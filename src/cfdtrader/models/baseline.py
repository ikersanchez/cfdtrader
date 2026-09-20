"""Modelo baseline de probabilidad: logistica con elastic net sobre 10 features (#24).

Responde **una** pregunta —«la sesion `t` cierra por encima de su apertura?»— para la
direccion **larga unica**: ``P(y = 1)`` con ``y = 1{ret_long > 0}``.

Este modulo es **puro** (A1): del frame etiquetado a las predicciones, sin almacen y sin
capa de informe. No importa ``cfdtrader.data``, ``cfdtrader.analysis``, ``cfdtrader.features``,
``duckdb``, ``backtest`` ni ``model_selection``: el plan de particiones de #12 llega ya
traducido a **posiciones** (:class:`SplitAssignment`), el canonicamente hasheable de #13 y el
registro de #16 viven fuera, y el ``model_sha256`` lo calcula quien puede importarlos.

Disponibilidad temporal (A2): la fila de diseno de `t` es la fila **completa** de features de
`t-1` (:data:`DESIGN_LAG_SESSIONS`), la sesion anterior **del diario**. Una sola regla, sin
excepciones y **sin** filtrar columnas por ``required_as_of``: ``atr_norm`` declara «cierre de
la sesion `t`» en ``volatility_v1`` y «cierre de la sesion `t-1`» en ``technical_v1`` (#72), asi
que filtrar por ``required_as_of`` no tendria respuesta unica.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final, Protocol, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

__all__ = [
    "BASELINE_FEATURES",
    "DECISION_THRESHOLD",
    "DESIGN_LAG_SESSIONS",
    "DESIGN_SESSION_COLUMN",
    "HYPERPARAMETERS",
    "MODEL_DOES_NOT_DO",
    "SEED",
    "BaselineError",
    "BaselineModel",
    "DesignFrame",
    "FoldFit",
    "InvalidDesignFrameError",
    "SplitAssignment",
    "UnknownFeatureError",
    "design_frame",
    "fit_baseline",
    "long_signal",
    "probabilities",
    "sigmoid",
]

#: Semilla declarada a priori: una sola, sin barridos (A7).
SEED: Final[int] = 20260920

#: Corrimiento de diseno en **sesiones del diario** (A2): la fila de `t` es la de `t-1`.
DESIGN_LAG_SESSIONS: Final[int] = 1

#: Umbral de decision declarado a priori (A11): `p >= 0,5` ⇒ largo. El umbral economico de
#: §4.4 y el *sizing* son #27 y #60, **no** esto.
DECISION_THRESHOLD: Final[float] = 0.5

#: Columna de auditoria del corrimiento: de que sesion viene la fila de diseno.
DESIGN_SESSION_COLUMN: Final[str] = "design_feature_session"

#: Las **10** features, fijadas **a priori** (A4): sin seleccion guiada por datos, luego sin
#: *leakage* por construccion. Nunca dos del mismo bloque redundante, y las cinco familias
#: representadas: volatilidad (`har_forecast`, `vix_zscore`), tecnica (`dist_sma_20_z`),
#: contexto (`asia_overnight_1`, `europe_prev_1`, `dxy_ret_1`, `sector_dispersion_1`), macro
#: (`ust_10y_chg_5`) y regimen (`garch_forecast_z`, `is_es_roll_session`).
#:
#: Fuera a proposito: `sessions_to_opex` (17 nulos de cola, #79), `pendiente_2s10s` y
#: `pendiente_2s10s_chg_5` (dependencia lineal exacta: `ust_10y - ust_2y`) y el resto de cada
#: bloque redundante entre si (`corr_*`, `vix_level`/`vix_percentile`, `har_lag1/4/17`,
#: `atr_norm`/`atr_norm_z`, `rv_percentile`, `ret_1/5/21`, `dist_sma_20`, `rsi_14`,
#: `range_pos_20`, `vol_break_20`, `sector_count`, `sector_dispersion_1_z`, `fed_funds*`,
#: `ust_10y`, `ust_2y*`, `cpi_yoy`, `pce_yoy`, `dxy`, `dxy_z`, `ust_10y_z`, `garch_forecast`,
#: `efficiency_ratio_20`, `day_of_week`, `true_range`, `parkinson_rv`, `ret_log`, `ret_sq`,
#: `beta_vix_60`, `corr_*`).
BASELINE_FEATURES: Final[tuple[str, ...]] = (
    "har_forecast",
    "vix_zscore",
    "dist_sma_20_z",
    "asia_overnight_1",
    "europe_prev_1",
    "dxy_ret_1",
    "sector_dispersion_1",
    "ust_10y_chg_5",
    "garch_forecast_z",
    "is_es_roll_session",
)

#: Hiperparametros **fijos a priori** (A7), sin busqueda ni barrido: viajan a la configuracion
#: registrada (#16) y por tanto al ``run_sha256``.
#:
#: ``penalty`` se publica con la ortografia que fija la issue, pero **no** se pasa: en
#: scikit-learn 1.9.1 un ``penalty`` explicito esta deprecado y el mismo estimador
#: (elastic net) se pide con ``0 < l1_ratio < 1``. El valor declarado y el efectivo coinciden,
#: y ``penalty_translation`` lo deja escrito para que no parezca un descuido.
#:
#: ``max_iter`` y ``tol`` son **cotas declaradas**, no ajustes: la convergencia se publica por
#: fold (``n_iter``, ``converged``) para que se pueda comprobar que no se alcanzo la cota.
HYPERPARAMETERS: Final[dict[str, object]] = {
    "penalty": "elasticnet",
    "solver": "saga",
    "l1_ratio": 0.5,
    "C": 1.0,
    "fit_intercept": True,
    "max_iter": 1000,
    "tol": 1e-4,
    "random_state": SEED,
}

#: Que **no** hace este modulo, legible por maquina. Cada frontera con su issue.
MODEL_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_lee_el_almacen",
        "issue": "#73",
        "statement": (
            "no lee `raw.*` ni `derived.*` ni escribe `derived.features_daily`: recibe el frame "
            "etiquetado ya construido (el adaptador es `analysis.feature_frame`, y la "
            "persistencia de features es #73)"
        ),
    },
    {
        "id": "no_construye_el_plan",
        "issue": "#12",
        "statement": (
            "no construye ni reimplementa las particiones: recibe el plan ya traducido a "
            "posiciones (`SplitAssignment`) y no decide purga ni embargo"
        ),
    },
    {
        "id": "no_calibra",
        "issue": "#25",
        "statement": (
            "no calibra las probabilidades (Platt/isotonica): publica la curva **sin** calibrar "
            "y la calibracion es #25"
        ),
    },
    {
        "id": "no_decide_el_umbral_economico",
        "issue": "#27",
        "statement": (
            "el umbral 0,5 de A11 es el del informe, no el del sistema: el umbral economico y el "
            "*sizing* son #27 y #60"
        ),
    },
)

#: Tipos numericos que se aceptan al leer un frame de diseno.
_FLOAT_DTYPE: Final[frozenset[str]] = frozenset({"Float64", "Float32", "Int64", "Int32"})


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados
# ─────────────────────────────────────────────────────────────────────────────
class BaselineError(Exception):
    """Raiz de los errores del modelo baseline."""


class InvalidDesignFrameError(BaselineError):
    """El frame de diseno no trae lo que el modelo necesita (columnas, filas)."""


class UnknownFeatureError(BaselineError):
    """Se pidio una feature que no esta entre las 10 declaradas (A4)."""


# ─────────────────────────────────────────────────────────────────────────────
# El frame de diseno (A2, A3): corrimiento de una sesion + la etiqueta
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class DesignFrame:
    """La matriz de diseno: una fila por sesion etiquetada, features de `t-1` y `y` de `t`.

    ``n_shifted_rows`` son las sesiones etiquetadas que el corrimiento **pierde** (la primera
    sesion del diario no tiene anterior): se publican, nunca se rellenan. ``n_nulls_in_features``
    son los nulos de las 10 columnas de diseno: un nulo aqui se publica y se rechaza antes de
    entrenar, no se imputa.
    """

    frame: pl.DataFrame
    n_sessions: int
    n_labels: int
    n_shifted_rows: int
    n_nulls_in_features: int
    design_lag_sessions: int

    @property
    def sessions(self) -> tuple[date, ...]:
        """Las sesiones del diseno, en orden."""
        return tuple(cast("list[date]", self.frame.get_column("session").to_list()))

    @property
    def positives(self) -> int:
        """Numero de ``y == 1`` (A3)."""
        return int(self.frame.get_column("y").sum())


def _require_frame(frame: object, *, what: str) -> pl.DataFrame:
    """El argumento tiene que ser un ``pl.DataFrame`` (error tipado, no ``AttributeError``)."""
    if not isinstance(frame, pl.DataFrame):
        raise InvalidDesignFrameError(f"{what} tiene que ser un pl.DataFrame, no {type(frame)}")
    return frame


def _require_instance(value: object, expected: type[object], *, field: str) -> None:
    """El argumento tiene que ser del tipo declarado (error tipado, no ``AttributeError``).

    Se usa un helper en vez de un ``isinstance`` en linea porque comprobar el tipo de un
    argumento **ya anotado** es lo que marca ``reportUnnecessaryIsInstance`` en pyright
    estricto: aqui el valor entra como ``object`` y la comprobacion si aporta.
    """
    if not isinstance(value, expected):
        raise InvalidDesignFrameError(
            f"{field} tiene que ser {expected.__name__}, no {type(value).__name__}"
        )


def _require_columns(frame: pl.DataFrame, *, columns: Sequence[str], what: str) -> None:
    """Las columnas declaradas tienen que existir (A4)."""
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise UnknownFeatureError(
            f"{what} no trae las columnas {missing}: las features declaradas son "
            f"{list(BASELINE_FEATURES)} y no se sustituyen por otras (A4)"
        )


def design_frame(features: pl.DataFrame, *, labels: pl.DataFrame) -> DesignFrame:
    """Construye la matriz de diseno desde el frame de features y las etiquetas (A2, A3).

    La fila de diseno de la sesion `t` es la fila de features de la **sesion anterior del
    diario** (:data:`DESIGN_LAG_SESSIONS` = 1): una sola regla, aplicada a las 53 columnas, sin
    filtrar por ``required_as_of``. La sesion de origen viaja en
    :data:`DESIGN_SESSION_COLUMN` para poder auditarla.

    ``labels`` trae ``session`` y ``ret_long``; el objetivo es ``y = 1{ret_long > 0}`` (A3).
    Las sesiones que no estan en las etiquetas se quedan fuera del diseno, y las etiquetas que
    no tienen sesion anterior se cuentan en ``n_shifted_rows`` en vez de rellenarse.
    """
    feature_frame = _require_frame(features, what="features")
    label_frame = _require_frame(labels, what="labels")
    _require_columns(feature_frame, columns=("session", *BASELINE_FEATURES), what="features")
    _require_columns(label_frame, columns=("session", "ret_long"), what="labels")

    ordered = feature_frame.sort("session")
    shifted = ordered.select(
        [
            pl.col("session"),
            pl.col("session").shift(DESIGN_LAG_SESSIONS).alias(DESIGN_SESSION_COLUMN),
            *[pl.col(name).shift(DESIGN_LAG_SESSIONS) for name in BASELINE_FEATURES],
        ]
    )
    joined = shifted.join(
        label_frame.select("session", "ret_long").sort("session"), on="session", how="inner"
    )
    joined = joined.with_columns(
        (pl.col("ret_long") > 0).cast(pl.Int64).alias("y"),
    ).sort("session")

    nulls = sum(joined.get_column(name).null_count() for name in BASELINE_FEATURES)
    return DesignFrame(
        frame=joined,
        n_sessions=joined.height,
        n_labels=label_frame.height,
        n_shifted_rows=label_frame.height - joined.height,
        n_nulls_in_features=nulls,
        design_lag_sessions=DESIGN_LAG_SESSIONS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# El modelo: elastic net ajustado fold a fold (A6, A7)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """Un fold del plan de #12 traducido a **posiciones** del frame de diseno.

    Este modulo no conoce ``SplitPlan`` (vive en ``cfdtrader.backtest``, que tiene prohibido
    importar): el adaptador traduce el plan y aqui solo se consumen posiciones.
    """

    index: int
    train: tuple[int, ...]
    test: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FoldFit:
    """Lo que se ajusta en **un** fold, fold a fold y sin estado compartido (A7).

    ``mean`` y ``scale`` son el escalado de **ese** train: ajustar el escalado con el test (o
    con la muestra entera) seria *look-ahead*. ``n_iter``/``converged`` publican la
    convergencia del solver en vez de afirmarla.
    """

    index: int
    n_train: int
    n_test: int
    train_first_session: date
    train_last_session: date
    train_positives: int
    train_base_rate: float
    coefficients: tuple[float, ...]
    intercept: float
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    n_iter: int
    converged: bool
    test_positions: tuple[int, ...]

    def to_payload(self) -> dict[str, object]:
        """El fold como JSON puro: coeficientes, intercepto y escalado, sin `pickle` (A12)."""
        return {
            "index": self.index,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_first_session": self.train_first_session.isoformat(),
            "train_last_session": self.train_last_session.isoformat(),
            "train_positives": self.train_positives,
            "train_base_rate": self.train_base_rate,
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "n_iter": self.n_iter,
            "converged": self.converged,
            "test_positions": list(self.test_positions),
        }


@dataclass(frozen=True, slots=True)
class BaselineModel:
    """El modelo ajustado: un :class:`FoldFit` por fold, mas las constantes declaradas.

    No guarda el objeto de scikit-learn: lo que se serializa son los coeficientes, el
    intercepto y el escalado, y la probabilidad se recalcula con :func:`sigmoid`. Asi el
    ``model.json`` es suficiente para reproducir las predicciones (A12) y no hace falta
    `pickle`.
    """

    features: tuple[str, ...]
    hyperparameters: Mapping[str, object]
    seed: int
    folds: tuple[FoldFit, ...]

    def fold_for(self, position: int) -> FoldFit | None:
        """El fold cuyo *test* contiene esa posicion, o ``None`` si esta fuera de todo test."""
        for fold in self.folds:
            if position in fold.test_positions:
                return fold
        return None

    def to_payload(self) -> dict[str, object]:
        """El modelo completo como JSON puro, con la spec de features que lo define."""
        return {
            "features": list(self.features),
            "hyperparameters": dict(self.hyperparameters),
            "seed": self.seed,
            "design_lag_sessions": DESIGN_LAG_SESSIONS,
            "decision_threshold": DECISION_THRESHOLD,
            "folds": [fold.to_payload() for fold in self.folds],
        }


def _matrix(frame: pl.DataFrame) -> NDArray[np.float64]:
    """Las 10 columnas declaradas como matriz ``(n, 10)`` de ``float64``, en su orden (A4)."""
    missing = [name for name in BASELINE_FEATURES if name not in frame.columns]
    if missing:
        raise UnknownFeatureError(
            f"el frame de diseno no trae {missing}: el modelo solo acepta las 10 features "
            f"declaradas, en su orden (A4)"
        )
    for name in BASELINE_FEATURES:
        dtype = str(frame.get_column(name).dtype)
        if dtype not in _FLOAT_DTYPE:
            raise InvalidDesignFrameError(
                f"la columna '{name}' tiene tipo {dtype}: el modelo solo acepta columnas "
                "numericas (los booleanos de polars se convierten a 0/1 float antes)"
            )
    selected = frame.select(list(BASELINE_FEATURES)).cast(pl.Float64)
    return cast("NDArray[np.float64]", selected.to_numpy())


def _outcomes(matrix: pl.DataFrame, *, column: str = "y") -> NDArray[np.float64]:
    """La etiqueta ``y`` como vector ``float64`` (el solver la exige numerica)."""
    if column not in matrix.columns:
        raise InvalidDesignFrameError(
            f"el frame de diseno no trae la columna '{column}': sin etiqueta no hay ajuste"
        )
    return cast("NDArray[np.float64]", matrix.get_column(column).cast(pl.Float64).to_numpy())


def sigmoid(score: float) -> float:
    """``1 / (1 + exp(-score))`` estable en los dos extremos (no desborda)."""
    if score >= 0.0:
        return 1.0 / (1.0 + math.exp(-score))
    exponential = math.exp(score)
    return exponential / (1.0 + exponential)


def _require_positions(positions: Sequence[int], *, n_sessions: int, what: str) -> tuple[int, ...]:
    """Las posiciones tienen que existir en el frame y no venir vacias."""
    values = tuple(positions)
    if not values:
        raise InvalidDesignFrameError(f"{what} viene vacio: un split sin muestra no es un split")
    out_of_range = sorted({value for value in values if not 0 <= value < n_sessions})
    if out_of_range:
        raise InvalidDesignFrameError(
            f"{what} apunta fuera del frame ({n_sessions} filas): {out_of_range}. Las posiciones "
            "se refieren al frame de diseno, que tiene que venir en el orden del universo"
        )
    return values


def fit_baseline(
    design: DesignFrame,
    *,
    splits: Sequence[SplitAssignment],
    hyperparameters: Mapping[str, object] | None = None,
    seed: int = SEED,
) -> BaselineModel:
    """Ajusta la logistica con elastic net fold a fold, con el escalado del train (A6, A7).

    Cada fold es independiente: el escalado se ajusta **solo** con su train y el estimador
    tambien. No hay busqueda de hiperparametros ni barrido (A7): ``hyperparameters`` es la
    constante declarada y el *seed* viaja con ella.
    """
    _require_instance(design, DesignFrame, field="design")
    assignments = tuple(splits)
    if not assignments:
        raise InvalidDesignFrameError("no hay folds que ajustar: el plan de #12 no vino vacio")
    matrix = _matrix(design.frame)
    label = _outcomes(design.frame)
    if matrix.shape[0] != label.shape[0]:
        raise InvalidDesignFrameError(
            f"la matriz tiene {matrix.shape[0]} filas y la etiqueta {label.shape[0]}: el frame "
            "de diseno esta desalineado"
        )
    parameters = dict(HYPERPARAMETERS if hyperparameters is None else hyperparameters)
    folds: list[FoldFit] = []
    for assignment in assignments:
        label_train = f"el train del fold {assignment.index}"
        label_test = f"el test del fold {assignment.index}"
        train = _require_positions(assignment.train, n_sessions=matrix.shape[0], what=label_train)
        test = _require_positions(assignment.test, n_sessions=matrix.shape[0], what=label_test)
        overlap = sorted(set(train).intersection(test))
        if overlap:
            raise InvalidDesignFrameError(
                f"el train y el test del fold {assignment.index} se solapan en {overlap}: el "
                "escalado se ajustaria con el test (A7)"
            )
        folds.append(
            _fit_fold(
                matrix,
                label,
                sessions=design.sessions,
                assignment=assignment,
                train=train,
                test=test,
                parameters=parameters,
                seed=seed,
            )
        )
    return BaselineModel(
        features=BASELINE_FEATURES,
        hyperparameters=parameters,
        seed=seed,
        folds=tuple(folds),
    )


def _fit_fold(
    matrix: NDArray[np.float64],
    label: NDArray[np.float64],
    *,
    sessions: tuple[date, ...],
    assignment: SplitAssignment,
    train: tuple[int, ...],
    test: tuple[int, ...],
    parameters: Mapping[str, object],
    seed: int,
) -> FoldFit:
    """Ajusta **un** fold: escalado del train, estimador del train, convergencia publicada."""
    train_matrix = matrix[list(train), :]
    mean = train_matrix.mean(axis=0)
    scale = train_matrix.std(axis=0)
    # Una columna constante no se puede estandarizar: su escala es 1 y el coeficiente se queda
    # en lo que decida la regularizacion. Nunca se divide por cero.
    scale = np.where(scale == 0.0, 1.0, scale)
    standardised = (train_matrix - mean) / scale
    coefficients, intercept, iterations = _solve(
        standardised, label[list(train)], parameters=parameters, seed=seed
    )
    limit = int(cast("int", parameters["max_iter"]))
    positives = int(label[list(train)].sum())
    return FoldFit(
        index=assignment.index,
        n_train=len(train),
        n_test=len(test),
        train_first_session=sessions[train[0]],
        train_last_session=sessions[train[-1]],
        train_positives=positives,
        train_base_rate=positives / len(train),
        coefficients=coefficients,
        intercept=intercept,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        n_iter=iterations,
        converged=iterations < limit,
        test_positions=test,
    )


class _FittedLogistic(Protocol):
    """Vista tipada de lo unico que se lee de scikit-learn (que no publica anotaciones).

    El estimador entra por ``cast`` y sale por aqui: el resto del modulo trabaja con ``float``
    de verdad y no hace falta silenciar cada acceso a un atributo ajustado.
    """

    coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]
    n_iter_: NDArray[np.int64]

    def fit(self, features: NDArray[np.float64], outcomes: NDArray[np.float64]) -> object: ...


def _solve(
    standardised: NDArray[np.float64],
    label: NDArray[np.float64],
    *,
    parameters: Mapping[str, object],
    seed: int,
) -> tuple[tuple[float, ...], float, int]:
    """Ajusta el estimador y devuelve ``(coeficientes, intercepto, iteraciones)``.

    La ortografia del estimador es la de la issue salvo ``penalty``: en scikit-learn 1.9.1 un
    ``penalty`` explicito esta deprecado y el mismo estimador (elastic net) se pide con
    ``0 < l1_ratio < 1``. El valor declarado y el efectivo coinciden, y ambos viajan al informe.
    """
    estimator = cast(
        "_FittedLogistic",
        LogisticRegression(
            solver=str(parameters["solver"]),
            l1_ratio=float(cast("float", parameters["l1_ratio"])),
            C=float(cast("float", parameters["C"])),
            fit_intercept=bool(parameters["fit_intercept"]),
            max_iter=int(cast("int", parameters["max_iter"])),
            tol=float(cast("float", parameters["tol"])),
            random_state=seed,
        ),
    )
    estimator.fit(standardised, label)
    coefficients = tuple(float(value) for value in np.asarray(estimator.coef_).ravel())
    intercept = float(np.asarray(estimator.intercept_).ravel()[0])
    iterations = int(np.asarray(estimator.n_iter_).ravel()[0])
    return coefficients, intercept, iterations


def probabilities(
    model: BaselineModel,
    frame: pl.DataFrame,
) -> tuple[float | None, ...]:
    """Una probabilidad por fila: la del fold cuyo *test* la contiene, o ``None`` (A6, A9).

    Fuera de todo *test* no hay prediccion **honesta**: cada fold predice con el modelo que
    **no** vio esas sesiones, y una sesion que no cae en ningun test no se predice con un
    modelo que la vio en su train. ``None`` se publica como tal, nunca como ``0``.
    """
    matrix = _matrix(frame)
    out: list[float | None] = [None] * matrix.shape[0]
    for fold in model.folds:
        positions = list(fold.test_positions)
        mean = np.asarray(fold.mean, dtype=np.float64)
        scale = np.asarray(fold.scale, dtype=np.float64)
        coefficients = np.asarray(fold.coefficients, dtype=np.float64)
        scores = ((matrix[positions, :] - mean) / scale) @ coefficients + fold.intercept
        for position, score in zip(positions, cast("Sequence[float]", scores), strict=True):
            out[position] = sigmoid(float(score))
    return tuple(out)


def long_signal(probability: float) -> bool:
    """La decision declarada (A11): ``True`` (largo) si ``p >= DECISION_THRESHOLD``."""
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise BaselineError(
            f"una probabilidad de {probability!r} no es admisible: el decider de A11 solo lee "
            "una probabilidad entre 0 y 1"
        )
    return probability >= DECISION_THRESHOLD
