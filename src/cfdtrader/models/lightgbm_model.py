"""Modelo LightGBM reducido sobre las mismas 10 features, con el mismo protocolo (#26).

Segunda **familia** del modelo supervisado: un ``LGBMClassifier`` con
``min_child_samples`` alto (``LIGHTGBM_HYPERPARAMETERS``), ajustado fold a fold sobre la
**misma** matriz de diseno de #24 y calibrado por el **mismo** camino de
``models.calibration`` (#25). Responde la misma pregunta —«la sesion `t` cierra por encima
de su apertura?»— para la direccion **larga unica**.

Este modulo es **puro** (A1): del frame etiquetado a las predicciones, sin almacen y sin
capa de informe. No importa ``cfdtrader.data``, ``cfdtrader.analysis``, ``cfdtrader.backtest``
ni ``duckdb``; de ``cfdtrader.models`` reutiliza los **contratos compartidos** de
``models.baseline`` (``DesignFrame``, ``SplitAssignment``, la matriz de las 10 features y su
validacion) y los helpers de ``models.calibration``. El ``model_sha256`` y el ``model.json``
los escribe quien puede importar el canonicamente hasheable de #13
(``analysis.model_comparison``).

Hiperparametros **fijos** (A4), sin busqueda ni barrido: ``LIGHTGBM_HYPERPARAMETERS`` es la
constante declarada, viaja a la configuracion registrada de #16 y por tanto al
``run_sha256``. Otra variante exige una entrada nueva en ``runs/`` con otro ``variant_id``
(la busqueda es #82).

Determinismo (A5): el ajuste esta **pinnado** (``deterministic=True``,
``force_row_wise=True``, ``num_threads=1``, ``random_state`` fijo) y el modulo lo **mide**:
cada fold se ajusta dos veces y los dos textos de ``model_to_string()`` tienen que coincidir
byte a byte, y ademas el ``booster_model`` publicado se recarga y tiene que reproducir las
probabilidades publicadas (``RELOAD_TOLERANCE``). Si algo diverge, ``LightGBMDeterminismError``
y no se escribe nada.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final, cast

import lightgbm
import numpy as np
import polars as pl
from lightgbm import Booster, LGBMClassifier
from numpy.typing import NDArray

from cfdtrader.models.baseline import (
    BASELINE_FEATURES,
    DECISION_THRESHOLD,
    DESIGN_LAG_SESSIONS,
    SEED,
    DesignFrame,
    SplitAssignment,
    _matrix,  # pyright: ignore[reportPrivateUsage]
    _outcomes,  # pyright: ignore[reportPrivateUsage]
    _require_positions,  # pyright: ignore[reportPrivateUsage]
)
from cfdtrader.models.calibration import (
    Calibration,
    fit_calibration,
    sigmoid,
    split_train_for_calibration,
)

__all__ = [
    "LIGHTGBM_HYPERPARAMETERS",
    "LMODEL_DOES_NOT_DO",
    "RELOAD_TOLERANCE",
    "InvalidLightGBMInputError",
    "LightGBMDeterminismError",
    "LightGBMError",
    "LightGBMFoldFit",
    "LightGBMModel",
    "calibrated_probabilities",
    "fit_lightgbm",
    "probabilities",
    "reload_probabilities",
]

#: Hiperparametros **fijos a priori** (A4), sin busqueda ni barrido: viajan a la configuracion
#: registrada (#16) y por tanto al ``run_sha256`` del experimento.
#:
#: La familia es **reducida a proposito**: ``num_leaves = 4`` y ``max_depth = 2`` dan arboles
#: de profundidad 2 (tres cortes), y ``min_child_samples = 200`` es del 7,6-9,1 % del *train*
#: menor (2.187 sesiones): cada hoja exige una muestra grande, que es la lectura de
#: ``tech_stack.md`` («LightGBM por ``min_child_samples`` alto») y el motivo de que la
#: busqueda de hiperparametros sea **#82** y no esta tarea.
#:
#: ``deterministic = True``, ``force_row_wise = True`` y ``num_threads = 1`` son las tres
#: condiciones del determinismo **medido** (A5): con 16 hilos el modelo cambia, con y sin
#: ``deterministic``. ``subsample_freq = 0`` desactiva el muestreo (``subsample = 1.0``).
LIGHTGBM_HYPERPARAMETERS: Final[dict[str, object]] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 4,
    "max_depth": 2,
    "min_child_samples": 200,
    "subsample": 1.0,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "reg_lambda": 0.0,
    "random_state": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
    "verbose": -1,
}

#: Tolerancia con la que el ``booster_model`` **recargado** reproduce lo publicado (A5). Es
#: ``1e-12`` y no ``0`` porque el texto del modelo se reconstruye desde cero y la prediccion
#: vuelve a sumar las hojas; el valor medido es ``0.0`` exacto en las diez folds.
RELOAD_TOLERANCE: Final[float] = 1e-12

#: Que **no** hace este modulo, legible por maquina. Cada frontera con su issue.
LMODEL_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_barre_hiperparametros",
        "issue": "#82",
        "statement": (
            "los hiperparametros son **fijos** y no hay bucle de barrido: la constante "
            "`LIGHTGBM_HYPERPARAMETERS` viaja a la configuracion y el CLI no acepta "
            "banderas que la muevan. La busqueda de hiperparametros y de subconjunto de "
            "features es #82"
        ),
    },
    {
        "id": "no_lee_el_almacen",
        "issue": "#73",
        "statement": (
            "no lee `raw.*` ni `derived.*` ni escribe `derived.features_daily`: recibe el frame "
            "etiquetado ya construido (`analysis.feature_frame`) y la persistencia es #73"
        ),
    },
    {
        "id": "no_construye_el_plan",
        "issue": "#12",
        "statement": (
            "no construye ni reimplementa las particiones: recibe el plan ya traducido a "
            "posiciones (`SplitAssignment`), exactamente el mismo que el modelo lineal"
        ),
    },
    {
        "id": "no_decide_el_umbral_economico",
        "issue": "#27",
        "statement": (
            "el umbral 0,5 es el del informe, no el del sistema: el umbral economico y el "
            "*sizing* son #27 y #60"
        ),
    },
)

#: Tipos que se aceptan al leer el frame de diseno (el mismo contrato que #24).
_FLOAT_DTYPE: Final[frozenset[str]] = frozenset({"Float64", "Float32", "Int64", "Int32"})


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados
# ─────────────────────────────────────────────────────────────────────────────
class LightGBMError(Exception):
    """Raiz de los errores del modelo LightGBM."""


class InvalidLightGBMInputError(LightGBMError):
    """El frame de diseno o los folds no traen lo que el modelo LightGBM necesita."""


class LightGBMDeterminismError(LightGBMError):
    """El ajuste **no** es reproducible: dos ajustes del mismo fold difieren (A5)."""


# ─────────────────────────────────────────────────────────────────────────────
# Un fold y el modelo
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class LightGBMFoldFit:
    """Lo que se ajusta en **un** fold, sin estado compartido con los demas (A5).

    ``booster_model`` es el texto del ``Booster`` (``model_to_string()``), **sin** `pickle`:
    es lo que hace que ``model.json`` reproduzca las predicciones. ``test_probabilities`` son
    las probabilidades **crudas** de las sesiones de *test* y ``test_scores`` los margenes
    (el *logit*) con los que se ajusta y se aplica el calibrador. ``calibration`` es el
    calibrador ajustado con la **cola** del train (A6, #25), un estado publicado.
    """

    index: int
    n_train: int
    n_test: int
    train_first_session: date
    train_last_session: date
    train_positives: int
    train_base_rate: float
    n_trees: int
    booster_model: str
    test_positions: tuple[int, ...]
    test_probabilities: tuple[float, ...]
    test_scores: tuple[float, ...]
    calibration: Calibration

    def to_payload(self) -> dict[str, object]:
        """El fold como JSON puro: el modelo en texto y las predicciones del *test* (A5)."""
        return {
            "index": self.index,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_first_session": self.train_first_session.isoformat(),
            "train_last_session": self.train_last_session.isoformat(),
            "train_positives": self.train_positives,
            "train_base_rate": self.train_base_rate,
            "n_trees": self.n_trees,
            "booster_model": self.booster_model,
            "test_positions": list(self.test_positions),
            "test_probabilities": list(self.test_probabilities),
            "test_scores": list(self.test_scores),
            "calibration": self.calibration.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class LightGBMModel:
    """El modelo ajustado: un :class:`LightGBMFoldFit` por fold y las constantes declaradas.

    No guarda el objeto de LightGBM en memoria mas alla del texto: lo serializable es
    ``booster_model`` y las predicciones de *test*, y el ``model.json`` es suficiente para
    reproducirlas (A5) sin `pickle`.
    """

    features: tuple[str, ...]
    hyperparameters: Mapping[str, object]
    seed: int
    folds: tuple[LightGBMFoldFit, ...]

    def fold_for(self, position: int) -> LightGBMFoldFit | None:
        """El fold cuyo *test* contiene esa posicion, o ``None`` si esta fuera de todo test."""
        for fold in self.folds:
            if position in fold.test_positions:
                return fold
        return None

    def to_payload(self) -> dict[str, object]:
        """El modelo completo como JSON puro, con la version de la libreria que lo ajusto."""
        return {
            "library": {"name": "lightgbm", "version": lightgbm.__version__},
            "features": list(self.features),
            "hyperparameters": dict(self.hyperparameters),
            "seed": self.seed,
            "design_lag_sessions": DESIGN_LAG_SESSIONS,
            "decision_threshold": DECISION_THRESHOLD,
            "folds": [fold.to_payload() for fold in self.folds],
        }


def _require_design_columns(frame: pl.DataFrame) -> None:
    """Las 10 columnas declaradas tienen que estar y ser numericas (el contrato de #24)."""
    missing = sorted(name for name in BASELINE_FEATURES if name not in frame.columns)
    if missing:
        raise InvalidLightGBMInputError(
            f"el frame de diseno no trae {missing}: la familia LightGBM usa **las mismas** 10 "
            f"features declaradas ({list(BASELINE_FEATURES)}) y no se sustituyen por otras (A4)"
        )
    wrong = sorted(
        name for name in BASELINE_FEATURES if str(frame.get_column(name).dtype) not in _FLOAT_DTYPE
    )
    if wrong:
        raise InvalidLightGBMInputError(
            f"las columnas {wrong} no son numericas: el estimador solo acepta columnas "
            "numericas (los booleanos de polars se convierten a 0/1 float antes)"
        )


def _require_instance(value: object, expected: type[object], *, field: str) -> None:
    """El argumento tiene que ser del tipo declarado (error tipado, no ``AttributeError``)."""
    if not isinstance(value, expected):
        raise InvalidLightGBMInputError(
            f"{field} tiene que ser {expected.__name__}, no {type(value).__name__}"
        )


def _fit_once(
    train: NDArray[np.float64],
    label: NDArray[np.int64],
    parameters: Mapping[str, object],
) -> tuple[str, Booster]:
    """Ajusta **una** vez y devuelve ``(texto del modelo, booster)``.

    Es la **unica** llamada al estimador del modulo y se hace con el diccionario declarado
    entero: sin literales de hiperparametros y sin bucle de barrido (A4).
    """
    estimator = LGBMClassifier(**parameters)  # pyright: ignore[reportArgumentType]
    estimator.fit(train, label)  # pyright: ignore[reportUnknownMemberType]
    booster = estimator.booster_
    return str(booster.model_to_string()), booster


def _require_deterministic(
    first: str,
    second: str,
    *,
    index: int,
) -> None:
    """Los dos ajustes del mismo fold tienen que dar el **mismo** texto (A5)."""
    if first == second:
        return
    raise LightGBMDeterminismError(
        f"los dos ajustes del fold {index} producen modelos distintos "
        f"({len(first)} y {len(second)} caracteres): con `deterministic=True`, "
        "`force_row_wise=True`, `num_threads=1` y `random_state` fijo el texto de "
        "`model_to_string()` tiene que ser identico byte a byte. No se publica un modelo "
        "cuya reproducibilidad no esta medida (A5)"
    )


def _require_reload(
    fold: LightGBMFoldFit,
    matrix: NDArray[np.float64],
) -> None:
    """El texto publicado tiene que reproducir lo publicado al recargarlo (A5)."""
    booster = Booster(model_str=fold.booster_model)
    raw = cast("Any", booster.predict(matrix[list(fold.test_positions), :]))
    predicted = tuple(float(value) for value in np.asarray(raw))
    worst = max(
        (abs(one - other) for one, other in zip(predicted, fold.test_probabilities, strict=True)),
        default=0.0,
    )
    if worst > RELOAD_TOLERANCE:
        raise LightGBMDeterminismError(
            f"el `booster_model` del fold {fold.index} recargado no reproduce las "
            f"probabilidades publicadas: discrepancia maxima {worst!r} > {RELOAD_TOLERANCE!r}. "
            "El texto publicado no seria suficiente para reproducir el modelo (A5)"
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
    label_horizon: Sequence[int],
) -> LightGBMFoldFit:
    """Ajusta **un** fold dos veces, mide el determinismo y calibra con la cola (A5, A6).

    El *test* no entra ni en el estimador ni en el calibrador: la calibracion usa la cola
    purgada del **train** y sus puntuaciones salen del mismo booster (A6).
    """
    train_matrix = matrix[list(train), :]
    train_label = label[list(train)].astype(np.int64)
    first_text, first_booster = _fit_once(train_matrix, train_label, parameters)
    second_text, _ = _fit_once(train_matrix, train_label, parameters)
    _require_deterministic(first_text, second_text, index=assignment.index)

    test_matrix = matrix[list(test), :]
    fitted = cast("Any", first_booster.predict(test_matrix))
    test_probabilities = tuple(float(value) for value in fitted)
    margins = cast("Any", first_booster.predict(test_matrix, raw_score=True))
    test_scores = tuple(float(value) for value in margins)
    split = split_train_for_calibration(train, label_horizon=label_horizon)
    calibration_positions = list(split.calibration)
    calibration_scores = cast(
        "Any", first_booster.predict(matrix[calibration_positions, :], raw_score=True)
    )
    calibration = fit_calibration(
        split,
        scores=[float(value) for value in np.asarray(calibration_scores)],
        outcomes=[int(value) for value in label[calibration_positions]],
    )
    positives = int(label[list(train)].sum())
    fold = LightGBMFoldFit(
        index=assignment.index,
        n_train=len(train),
        n_test=len(test),
        train_first_session=sessions[train[0]],
        train_last_session=sessions[train[-1]],
        train_positives=positives,
        train_base_rate=positives / len(train),
        n_trees=int(first_booster.num_trees()),
        booster_model=first_text,
        test_positions=test,
        test_probabilities=test_probabilities,
        test_scores=test_scores,
        calibration=calibration,
    )
    _require_reload(fold, matrix)
    return fold


def fit_lightgbm(
    design: DesignFrame,
    *,
    splits: Sequence[SplitAssignment],
    hyperparameters: Mapping[str, object] | None = None,
    seed: int = SEED,
    label_horizon: Sequence[int] | None = None,
) -> LightGBMModel:
    """Ajusta el LightGBM reducido fold a fold, con el mismo protocolo que #24 (A4, A5).

    Cada fold es independiente y se ajusta **dos veces** para medir el determinismo: si los
    dos textos difieren, ``LightGBMDeterminismError`` y el llamante no escribe nada. La
    calibracion es la de #25, con la cola purgada del *train* de ese mismo fold.
    """
    _require_instance(design, DesignFrame, field="design")
    assignments = tuple(splits)
    if not assignments:
        raise InvalidLightGBMInputError("no hay folds que ajustar: el plan de #12 no vino vacio")
    _require_design_columns(design.frame)
    matrix = _matrix(design.frame)
    label = _outcomes(design.frame)
    horizon = (
        (0,) * matrix.shape[0]
        if label_horizon is None
        else tuple(int(value) for value in label_horizon)
    )
    parameters = dict(LIGHTGBM_HYPERPARAMETERS if hyperparameters is None else hyperparameters)
    folds: list[LightGBMFoldFit] = []
    for assignment in assignments:
        train = _require_positions(
            assignment.train,
            n_sessions=matrix.shape[0],
            what=f"el train del fold {assignment.index}",
        )
        test = _require_positions(
            assignment.test,
            n_sessions=matrix.shape[0],
            what=f"el test del fold {assignment.index}",
        )
        overlap = sorted(set(train).intersection(test))
        if overlap:
            raise InvalidLightGBMInputError(
                f"el train y el test del fold {assignment.index} se solapan en {overlap}: el "
                "modelo se ajustaria con el test"
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
                label_horizon=horizon,
            )
        )
    return LightGBMModel(
        features=BASELINE_FEATURES,
        hyperparameters=parameters,
        seed=seed,
        folds=tuple(folds),
    )


def probabilities(
    model: LightGBMModel,
    frame: pl.DataFrame,
) -> tuple[float | None, ...]:
    """Una probabilidad **cruda** por fila: la del fold cuyo *test* la contiene, o ``None``.

    Fuera de todo *test* no hay prediccion honesta: ``None`` se publica como tal, nunca
    como ``0`` (el contrato de #24, reutilizado tal cual).
    """
    _require_design_columns(frame)
    size = int(_matrix(frame).shape[0])
    out: list[float | None] = [None] * size
    for fold in model.folds:
        for position, value in zip(fold.test_positions, fold.test_probabilities, strict=True):
            out[position] = value
    return tuple(out)


def calibrated_probabilities(
    model: LightGBMModel,
    frame: pl.DataFrame,
) -> tuple[float | None, ...]:
    """Una probabilidad **calibrada** por fila, o ``None`` fuera de todo *test* (A6).

    Es la que decide. Un fold sin calibrador publicado (``method: "none"``) pasa su
    probabilidad cruda, exactamente como ``models.baseline``.
    """
    _require_design_columns(frame)
    size = int(_matrix(frame).shape[0])
    out: list[float | None] = [None] * size
    for fold in model.folds:
        calibrated = fold.calibration.calibrate(list(fold.test_scores))
        for position, score, value in zip(
            fold.test_positions, fold.test_scores, calibrated, strict=True
        ):
            out[position] = sigmoid(float(score)) if value is None else value
    return tuple(out)


def reload_probabilities(
    fold: LightGBMFoldFit,
    frame: pl.DataFrame,
) -> tuple[float, ...]:
    """Recarga el ``booster_model`` desde su texto y recalcula las probabilidades (A5).

    Es la comprobacion de que lo publicado **basta**: `numpy` y el texto del modelo, sin el
    objeto ajustado en memoria y sin `pickle`. Un texto que no reproduzca lo publicado es un
    ``LightGBMDeterminismError``.
    """
    matrix = _matrix(frame)
    positions = _require_positions(
        fold.test_positions, n_sessions=matrix.shape[0], what=f"el test del fold {fold.index}"
    )
    booster = Booster(model_str=fold.booster_model)
    raw = cast("Any", booster.predict(matrix[list(positions), :]))
    predicted = tuple(float(value) for value in np.asarray(raw))
    worst = max(
        (abs(one - other) for one, other in zip(predicted, fold.test_probabilities, strict=True)),
        default=0.0,
    )
    if worst > RELOAD_TOLERANCE:
        raise LightGBMDeterminismError(
            f"el `booster_model` del fold {fold.index} recargado no reproduce las "
            f"probabilidades publicadas: discrepancia maxima {worst!r} > {RELOAD_TOLERANCE!r} (A5)"
        )
    return predicted
