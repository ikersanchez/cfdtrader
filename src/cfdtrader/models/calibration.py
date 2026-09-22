"""Calibracion de probabilidades dentro del *train*, sin ver el *test* (#25).

Que una probabilidad de 0,60 signifique de verdad un 60 %: el calibrador se ajusta con la
**cola** del *train* de cada fold (el 20 % que queda despues del bloque de ajuste) y nunca con
el *test*, que no entra ni en el estimador ni en el calibrador.

Este modulo es **puro** (A1): `numpy` y estimadores **publicos** de `scikit-learn`, sin
`cfdtrader.data`, `cfdtrader.analysis`, `cfdtrader.backtest`, `duckdb`, `model_selection` ni
serializadores binarios. No conoce el almacen, ni el plan de particiones, ni el registro:
recibe **posiciones** y **puntuaciones** y devuelve el calibrador como *floats*.

El dominio del calibrador es el **score** (el *logit* del fold, lo que `sigmoid` invierte), no
la probabilidad ya redondeada: `sigmoid(score)` y `score` son la misma cantidad salvo el
redondeo del enlace, y usar el score directo hace que la probabilidad publicada se reconstruya
**exacta** (tolerancia 0) desde los parametros del JSON (A8). La interpolacion de la isotonica
se recorta en los extremos (`np.interp`), asi que tampoco alli hace falta el objeto ajustado.

`sigmoid` vive aqui porque es el enlace que comparten el modelo crudo (`models.baseline` la
**importa**: una sola definicion) y el calibrador de Platt.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

__all__ = [
    "CALIBRATION_FRACTION",
    "CALIBRATION_HYPERPARAMETERS",
    "METHOD_ISOTONIC",
    "METHOD_NONE",
    "METHOD_PLATT",
    "MIN_CALIBRATION_SESSIONS",
    "PLATT_HYPERPARAMETERS",
    "PLATT_MAX_CALIBRATION_SESSIONS",
    "REASON_RANKING_INVERTED",
    "REASON_SINGLE_CLASS",
    "REASON_TRAIN_TOO_SMALL",
    "RECONSTRUCTION_FORMAT",
    "Calibration",
    "CalibrationError",
    "InvalidCalibrationInputError",
    "TrainSplit",
    "fit_calibration",
    "method_counts",
    "select_method",
    "sigmoid",
    "split_train_for_calibration",
]

#: Fraccion del *train* que se reserva para calibrar: la **cola** (A2).
CALIBRATION_FRACTION: Final[float] = 0.2

#: Por debajo de esto no se calibra: se publica `method: "none"` con el motivo (A5).
MIN_CALIBRATION_SESSIONS: Final[int] = 30

#: Regla del metodo (A4): Platt por debajo, isotonica a partir de aqui. Se evalua **por fold**
#: con el recuento medido, nunca con un promedio de metodos.
PLATT_MAX_CALIBRATION_SESSIONS: Final[int] = 500

#: Los tres estados de `method`: dos calibradores y el «sin calibrar» **publicado**.
METHOD_NONE: Final[str] = "none"
METHOD_PLATT: Final[str] = "platt"
METHOD_ISOTONIC: Final[str] = "isotonic"

#: Los tres motivos admisibles de `method: "none"` (A5). Son estados publicados, no fallos.
REASON_TRAIN_TOO_SMALL: Final[str] = "train_too_small"
REASON_SINGLE_CLASS: Final[str] = "single_class_calibration"
REASON_RANKING_INVERTED: Final[str] = "ranking_inverted"

#: Las **tres** constantes de calibracion, en el vocabulario de la configuracion registrada de
#: #16: viajan a `config.json` y por tanto al `run_sha256` (A9).
CALIBRATION_HYPERPARAMETERS: Final[dict[str, object]] = {
    "calibration_fraction": CALIBRATION_FRACTION,
    "min_calibration_sessions": MIN_CALIBRATION_SESSIONS,
    "platt_max_calibration_sessions": PLATT_MAX_CALIBRATION_SESSIONS,
}

#: Hiperparametros **fijos** del calibrador de Platt: una columna y un solver determinista.
PLATT_HYPERPARAMETERS: Final[dict[str, object]] = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "tol": 1e-4,
}

#: Como se reconstruye la probabilidad publicada desde los parametros del JSON (A8).
RECONSTRUCTION_FORMAT: Final[str] = (
    "platt: `p = sigmoid(coef * (score - mean) / scale + intercept)`; isotonica: "
    "`p = np.interp(score, thresholds, values)` (lineal, recortada en los extremos); none: la "
    "probabilidad cruda del fold. `score` es la puntuacion del fold, que publica "
    "`models.baseline.scores`"
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados
# ─────────────────────────────────────────────────────────────────────────────
class CalibrationError(Exception):
    """Raiz de los errores de la calibracion."""


class InvalidCalibrationInputError(CalibrationError):
    """El reparto o las puntuaciones no son lo que el calibrador necesita."""


# ─────────────────────────────────────────────────────────────────────────────
# El enlace, compartido con el modelo crudo
# ─────────────────────────────────────────────────────────────────────────────
def sigmoid(score: float) -> float:
    """``1 / (1 + exp(-score))`` estable en los dos extremos (no desborda).

    La definicion vive en **este** modulo y `models/baseline.py` la importa: una sola
    implementacion para el modelo crudo y para el calibrador de Platt.
    """
    if score >= 0.0:
        return 1.0 / (1.0 + math.exp(-score))
    exponential = math.exp(score)
    return exponential / (1.0 + exponential)


# ─────────────────────────────────────────────────────────────────────────────
# El reparto del train (A2)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class TrainSplit:
    """El *train* de un fold partido en `fit` (cabeza) y `calibration` (cola), purgado.

    ``purge_sessions`` son las sesiones que la regla literal de §11.1
    (``i + h[i] >= calibration_start``) aparta de la **cola del `fit`**: con ``h = 0`` son 0 y
    ``exclusions_are_no_op`` lo publica. ``calibration`` es siempre
    ``floor(CALIBRATION_FRACTION x n_train)`` sesiones del final del train.
    """

    fit: tuple[int, ...]
    calibration: tuple[int, ...]
    calibration_start: int
    purge_sessions: int
    exclusions_are_no_op: bool

    @property
    def n_fit(self) -> int:
        """Sesiones que quedan en el bloque de ajuste despues de la purga."""
        return len(self.fit)

    @property
    def n_calibration(self) -> int:
        """Sesiones del bloque de calibracion."""
        return len(self.calibration)

    def to_payload(self) -> dict[str, object]:
        """El reparto como JSON puro: recuentos, frontera y el eco de los no-ops."""
        return {
            "n_fit": self.n_fit,
            "n_calibration": self.n_calibration,
            "calibration_start": self.calibration_start,
            "purge_sessions": self.purge_sessions,
            "exclusions_are_no_op": self.exclusions_are_no_op,
        }


def split_train_for_calibration(
    train: Sequence[int], *, label_horizon: Sequence[int]
) -> TrainSplit:
    """Parte el *train* del fold en `fit` (80 %, cabeza) y `calibration` (20 %, cola) (A2).

    ``label_horizon`` es el horizonte **por posicion** del frame de diseno, el mismo vector que
    consume el plan de #12, y la purga es la regla literal de §11.1 aplicada a la frontera del
    bloque de calibracion: ``i + h[i] >= calibration_start``. Con ``h = 0`` aparta 0 sesiones y
    se publica como no-op; con ``h = 2`` aparta las dos ultimas del `fit`.

    El *test* del fold no pasa por aqui: no entra ni en el `fit` ni en la `calibration`.
    """
    ordered = tuple(sorted({int(value) for value in train}))
    if not ordered:
        raise InvalidCalibrationInputError(
            "el train de un fold no puede venir vacio: sin train no hay reparto que hacer (A2)"
        )
    horizon = tuple(int(value) for value in label_horizon)
    if len(horizon) <= ordered[-1]:
        raise InvalidCalibrationInputError(
            f"`label_horizon` trae {len(horizon)} posiciones y el train llega a la posicion "
            f"{ordered[-1]}: el horizonte se indexa por posicion del frame de diseno (A2)"
        )
    negative = [position for position in ordered if horizon[position] < 0]
    if negative:
        raise InvalidCalibrationInputError(
            f"`label_horizon` no puede ser negativo; posiciones {negative} (A2)"
        )
    boundary = len(ordered) - math.floor(CALIBRATION_FRACTION * len(ordered))
    head = ordered[:boundary]
    tail = ordered[boundary:]
    start = tail[0] if tail else ordered[-1] + 1
    dropped = {position for position in head if position + horizon[position] >= start}
    return TrainSplit(
        fit=tuple(position for position in head if position not in dropped),
        calibration=tail,
        calibration_start=start,
        purge_sessions=len(dropped),
        exclusions_are_no_op=not dropped,
    )


# ─────────────────────────────────────────────────────────────────────────────
# El calibrador ajustado (A3, A4, A5)
# ─────────────────────────────────────────────────────────────────────────────
def select_method(n_calibration: int) -> str:
    """La regla declarada (A4): Platt si ``n < 500``, isotonica si no, **por fold**."""
    if n_calibration < 0:
        raise InvalidCalibrationInputError(
            f"`n_calibration` no puede ser negativo: {n_calibration} (A4)"
        )
    return METHOD_PLATT if n_calibration < PLATT_MAX_CALIBRATION_SESSIONS else METHOD_ISOTONIC


def method_counts(methods: Sequence[str]) -> dict[str, int]:
    """Histograma de metodos (A4): los tres estados, tambien los que salen 0."""
    return {
        METHOD_PLATT: sum(1 for value in methods if value == METHOD_PLATT),
        METHOD_ISOTONIC: sum(1 for value in methods if value == METHOD_ISOTONIC),
        METHOD_NONE: sum(1 for value in methods if value == METHOD_NONE),
    }


@dataclass(frozen=True, slots=True)
class Calibration:
    """El calibrador de **un** fold: metodo, recuentos y parametros (o ``none`` y su motivo).

    ``method: "none"`` es un estado **publicado**, nunca una calibracion mala: ese fold pasa la
    probabilidad cruda tal cual y el motivo dice por que (A5). ``coef``/``intercept``/``mean``/
    ``scale`` son de Platt; ``thresholds``/``values`` de la isotonica.
    """

    method: str
    reason: str | None
    n_fit: int
    n_calibration: int
    n_positives: int
    calibration_positions: tuple[int, ...]
    purge_sessions: int
    exclusions_are_no_op: bool
    coef: float | None = None
    intercept: float | None = None
    mean: float | None = None
    scale: float | None = None
    thresholds: tuple[float, ...] = ()
    values: tuple[float, ...] = ()

    @property
    def calibrated(self) -> bool:
        """``True`` solo si hay un calibrador ajustado; ``none`` no calibra nada."""
        return self.method != METHOD_NONE

    def parameters(self) -> dict[str, object] | None:
        """Los parametros publicados, o ``None`` cuando el fold no calibra (decision 3)."""
        if self.method == METHOD_PLATT:
            return {
                "coef": self.coef,
                "intercept": self.intercept,
                "mean": self.mean,
                "scale": self.scale,
                "hyperparameters": dict(PLATT_HYPERPARAMETERS),
            }
        if self.method == METHOD_ISOTONIC:
            return {
                "thresholds": list(self.thresholds),
                "values": list(self.values),
                "increasing": True,
                "out_of_bounds": "clip",
            }
        return None

    def calibrate(self, scores: Sequence[float]) -> tuple[float | None, ...]:
        """La probabilidad calibrada de cada puntuacion, en orden (A5, A7).

        Devuelve ``None`` por posicion cuando el fold no calibra: quien llama decide pasar la
        cruda, que es lo que exige A5 («sus sesiones pasan con la cruda»), pero el modulo no
        esconde esa decision.
        """
        if self.method == METHOD_NONE:
            return tuple(None for _ in scores)
        values = np.asarray([float(value) for value in scores], dtype=np.float64)
        if self.method == METHOD_PLATT:
            mean = float(cast("float", self.mean))
            scale = float(cast("float", self.scale))
            logits = (values - mean) / scale * float(cast("float", self.coef)) + float(
                cast("float", self.intercept)
            )
            return tuple(sigmoid(float(value)) for value in logits)
        interpolated = np.interp(
            values, np.asarray(self.thresholds, dtype=np.float64), np.asarray(self.values)
        )
        return tuple(float(value) for value in interpolated)

    def to_payload(self) -> dict[str, object]:
        """El calibrador como JSON puro: metodo, recuentos, parametros y como se reproduce."""
        return {
            "method": self.method,
            "reason": self.reason,
            "calibrated": self.calibrated,
            "n_fit": self.n_fit,
            "n_calibration": self.n_calibration,
            "n_positives": self.n_positives,
            "calibration_positions": list(self.calibration_positions),
            "purge_sessions": self.purge_sessions,
            "exclusions_are_no_op": self.exclusions_are_no_op,
            "parameters": self.parameters(),
            "reconstruction": RECONSTRUCTION_FORMAT,
            "note": (
                "el calibrador se ajusta **solo** con las puntuaciones de "
                "`calibration_positions`, la cola purgada del train de ese fold: el test no "
                "entra ni en el estimador ni aqui (A2/A3); `none` significa que sus sesiones "
                "pasan con la probabilidad cruda (A5)"
            ),
        }


class _FittedLogistic(Protocol):
    """Vista tipada de lo unico que se lee del estimador de Platt (sklearn no se anota)."""

    coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]

    def fit(self, features: NDArray[np.float64], outcomes: NDArray[np.float64]) -> object: ...


class _FittedIsotonic(Protocol):
    """Vista tipada de lo unico que se lee de la isotonica: los umbrales y el sentido."""

    X_thresholds_: NDArray[np.float64]
    y_thresholds_: NDArray[np.float64]
    increasing_: bool

    def fit(self, features: NDArray[np.float64], outcomes: NDArray[np.float64]) -> object: ...


@dataclass(frozen=True, slots=True)
class _Fitted:
    """Lo ajustado en un fold, antes de decidir si se publica (A4)."""

    increasing: bool
    coef: float | None = None
    intercept: float | None = None
    mean: float | None = None
    scale: float | None = None
    thresholds: tuple[float, ...] = ()
    values: tuple[float, ...] = ()


def _as_calibration(
    split: TrainSplit,
    *,
    method: str,
    reason: str | None,
    n_positives: int,
    fitted: _Fitted | None,
) -> Calibration:
    """Arma el calibrador publicado: los recuentos del reparto y lo ajustado (si lo hay)."""
    return Calibration(
        method=method,
        reason=reason,
        n_fit=split.n_fit,
        n_calibration=split.n_calibration,
        n_positives=n_positives,
        calibration_positions=split.calibration,
        purge_sessions=split.purge_sessions,
        exclusions_are_no_op=split.exclusions_are_no_op,
        coef=None if fitted is None else fitted.coef,
        intercept=None if fitted is None else fitted.intercept,
        mean=None if fitted is None else fitted.mean,
        scale=None if fitted is None else fitted.scale,
        thresholds=() if fitted is None else fitted.thresholds,
        values=() if fitted is None else fitted.values,
    )


def fit_calibration(
    split: TrainSplit,
    *,
    scores: Sequence[float],
    outcomes: Sequence[int],
) -> Calibration:
    """Ajusta el calibrador del fold con las puntuaciones de su bloque de calibracion (A3/A4).

    La regla del metodo se evalua **aqui**, con el recuento medido: Platt si
    ``n_calibration < 500``, isotonica si no (A4). Si el reparto es demasiado corto, la
    calibracion es de una sola clase o el ajuste **invierte** el orden, se publica
    ``method: "none"`` con el motivo y ese fold pasa la cruda (A5): no se inventa un calibrador
    que empeore la probabilidad publicada.
    """
    values = np.asarray([float(value) for value in scores], dtype=np.float64)
    labels = np.asarray([float(bool(value)) for value in outcomes], dtype=np.float64)
    if values.shape[0] != split.n_calibration or labels.shape[0] != split.n_calibration:
        raise InvalidCalibrationInputError(
            f"el bloque de calibracion tiene {split.n_calibration} sesiones y llegaron "
            f"{values.shape[0]} puntuaciones y {labels.shape[0]} etiquetas: van alineadas y en "
            "el orden de `calibration_positions` (A3)"
        )
    n_positives = int(labels.sum())
    if split.n_calibration < MIN_CALIBRATION_SESSIONS:
        return _as_calibration(
            split,
            method=METHOD_NONE,
            reason=REASON_TRAIN_TOO_SMALL,
            n_positives=n_positives,
            fitted=None,
        )
    if n_positives == 0 or n_positives == split.n_calibration:
        return _as_calibration(
            split,
            method=METHOD_NONE,
            reason=REASON_SINGLE_CLASS,
            n_positives=n_positives,
            fitted=None,
        )
    method = select_method(split.n_calibration)
    fitted = _fit_platt(values, labels) if method == METHOD_PLATT else _fit_isotonic(values, labels)
    if not fitted.increasing:
        return _as_calibration(
            split,
            method=METHOD_NONE,
            reason=REASON_RANKING_INVERTED,
            n_positives=n_positives,
            fitted=None,
        )
    return _as_calibration(
        split, method=method, reason=None, n_positives=n_positives, fitted=fitted
    )


def _fit_platt(scores: NDArray[np.float64], outcomes: NDArray[np.float64]) -> _Fitted:
    """Ajusta la logistica de **una** columna sobre el *logit* estandarizado (decision 3).

    Se usa el estimador publico de `scikit-learn`, no un calibrador que envuelva el modelo: lo
    ajustado se publica como *floats* (`coef`, `intercept`, `mean`, `scale`) y el JSON
    reproduce la probabilidad sin `pickle`.
    """
    mean = float(scores.mean())
    scale = float(scores.std())
    if scale == 0.0 or not math.isfinite(scale):
        # Todas las puntuaciones iguales: no hay orden que preservar, luego no se calibra (A4).
        return _Fitted(increasing=False, coef=0.0, intercept=0.0, mean=mean, scale=1.0)
    standardised = (scores - mean) / scale
    estimator = cast(
        "_FittedLogistic",
        LogisticRegression(
            solver=str(PLATT_HYPERPARAMETERS["solver"]),
            max_iter=int(cast("int", PLATT_HYPERPARAMETERS["max_iter"])),
            tol=float(cast("float", PLATT_HYPERPARAMETERS["tol"])),
        ),
    )
    estimator.fit(standardised.reshape(-1, 1), outcomes)
    coef = float(np.asarray(estimator.coef_).ravel()[0])
    intercept = float(np.asarray(estimator.intercept_).ravel()[0])
    return _Fitted(increasing=coef > 0.0, coef=coef, intercept=intercept, mean=mean, scale=scale)


def _fit_isotonic(scores: NDArray[np.float64], outcomes: NDArray[np.float64]) -> _Fitted:
    """Ajusta la isotonica publica sobre el *logit* y publica sus umbrales.

    ``increasing="auto"`` deja que el ajuste **elija el sentido**: si elige decreciente, el
    calibrador invertiria el orden de las probabilidades y se rechaza (A4) en vez de publicarse.
    """
    # La firma publicada de `scikit-learn` tipa `increasing` como `bool`, pero el valor
    # declarado es `"auto"`: el ajuste elige el sentido y por eso se puede rechazar (A4).
    increasing = cast("bool", "auto")
    estimator = cast(
        "_FittedIsotonic",
        IsotonicRegression(increasing=increasing, out_of_bounds="clip"),
    )
    estimator.fit(scores, outcomes)
    return _Fitted(
        increasing=bool(estimator.increasing_),
        thresholds=tuple(float(value) for value in np.asarray(estimator.X_thresholds_).ravel()),
        values=tuple(float(value) for value in np.asarray(estimator.y_thresholds_).ravel()),
    )
