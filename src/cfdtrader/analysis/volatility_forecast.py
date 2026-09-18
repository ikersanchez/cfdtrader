"""Volatilidad realizada, VIX y primer *forecast* con veredicto *walk-forward* (tarea #7).

Responde a la pregunta de la tarea: **¿qué candidato de volatilidad se elige para
dimensionar las barreras de #10 y alimentar el informe de Fase 0 de #9?** No
construye ningún agente ni usa el LLM (`_docs/tasks.md`, Fase 0).

Esquema **pre-registrado** (no se cambia después de ver el resultado)
-------------------------------------------------------------------

- Ventana **expansiva**, ``min_train = 500`` sesiones.
- **Reajuste cada 21 sesiones para TODOS los candidatos** (el mismo ritmo para
  todos: ninguno juega con ventaja).
- Pronósticos **a un paso**, evaluados sobre todas las sesiones posteriores al
  calentamiento (22 sesiones de HAR) y al mínimo de entrenamiento.
- Candidatos: **exactamente cuatro**, sin búsqueda de hiperparámetros —
  ``rw`` (persistencia, el *baseline* que hay que batir), ``har``,
  ``har_vix`` (HAR + features de VIX) y ``garch`` (GARCH(1,1) con ``arch``).
- Métrica primaria: **QLIKE** en escala de varianza
  ``L = mean(ln v̂ + v / v̂)`` (robusta frente al ruido del proxy). Secundaria:
  **MSE del logaritmo** de la varianza. El ranking es por QLIKE medio.
- Regla de selección: gana el candidato que sea el mejor **o esté dentro del 2 %
  relativo** del mejor en **los dos** objetivos **y** que supere a ``rw`` en
  **más del 2 % relativo** de QLIKE en **los dos**. Si ninguno lo cumple, el
  veredicto es ``no_better_than_naive`` y el *fallback* es ``rw``; si los dos
  objetivos se contradicen, ``inconclusive``.

Objetivos y unidades
--------------------

- **Primario** — Parkinson, ``rv_t = (ln(H_t/L_t))² / (4 ln 2)``. Usa solo
  ``high``/``low``: es **inmune** al artefacto del ``open`` repetido (#52).
- **Secundario** — ``r_t²`` con ``r_t = ln(C_t/O_t)``. **No** es inmune a #52,
  porque lee el ``open``.

Unidades: **fracción²** en el cálculo y **bp** (×10⁴, y su raíz) al informar, como
en ``analysis/drift.py``.

Muestra limpia y exclusiones
----------------------------

La muestra limpia se deriva con la **definición única** de #52, importada de
:mod:`cfdtrader.analysis.drift` (tolerancia del 5 % de ``open`` repetido, primer
año desde el que ningún año posterior la supera, mínimo de 250 sesiones). Nunca
con un año escrito a mano. Se excluyen del ajuste y de la evaluación —sin imputar
a cero ni con la media— las sesiones con ``open`` repetido, con ``high``/``low``/
``close`` nulos y las **medias sesiones** (detectadas con
``data.calendar.MarketCalendar.is_half_day``, no con una fecha a mano).

Las medias sesiones se **excluyen al evaluar** y se **escalan al servir**: el API
de servicio devuelve el *forecast* multiplicado por ``√(3,5/6,5)``, porque una
media sesión dura 3,5 h de las 6,5 h de una sesión completa (`plan.md` §4.1 permite
normalizar por duración o excluir el día; aquí se hace lo primero al servir y lo
segundo al evaluar, y se declara).

Determinismo y red
------------------

Dos ejecuciones con el mismo ``--now`` producen un JSON idéntico byte a byte,
GARCH incluido. No hay ninguna llamada HTTP en este módulo: todo sale del almacén.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, cast

import arch
import duckdb
import numpy as np
import polars as pl
from loguru import logger
from scipy import stats  # pyright: ignore

from cfdtrader.analysis.drift import (
    MIN_CLEAN_SESSIONS,
    STALE_OPEN_TOLERANCE,
    clean_sample,
    clean_sample_cutoff,
    session_stale_open,
    yearly_stale_open_share,
)
from cfdtrader.data.calendar import FULL_SESSION_HOURS, HALF_SESSION_HOURS, MarketCalendar
from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.store import Store, UnknownDatasetError
from cfdtrader.features.volatility import HAR_WARMUP, add_features, fit_log_har

__all__ = [
    "CANDIDATES",
    "MIN_TRAIN",
    "REFIT_EVERY",
    "RELATIVE_TOLERANCE",
    "Candidate",
    "Fold",
    "Metrics",
    "Target",
    "Verdict",
    "VolatilityStudy",
    "analyse",
    "main",
    "render_markdown",
    "scale_sigma_for_duration",
    "select_candidate",
    "write_report",
]

#: Serie analizada: el subyacente del CFD. El CFD no tiene fuente (#50).
SERIES_ID: Final[str] = "^GSPC"

#: Serie del VIX, que en el S&P 500 es un índice real y líquido, no un proxy.
VIX_SERIES_ID: Final[str] = "^VIX"

SOURCE: Final[str] = "yfinance"

#: Sesiones mínimas de entrenamiento antes de empezar a evaluar (A12).
MIN_TRAIN: Final[int] = 500

#: Reajuste del modelo cada N sesiones, **el mismo para todos los candidatos** (A12).
REFIT_EVERY: Final[int] = 21

#: Horizonte del pronóstico: siempre a un paso (A12).
HORIZON: Final[int] = 1

#: Tolerancia relativa de la regla de selección (A16): 2 %.
RELATIVE_TOLERANCE: Final[float] = 0.02

#: Escalado del *forecast* al servir una media sesión: ``√(3,5/6,5)`` (A20).
HALF_DAY_SIGMA_SCALE: Final[float] = math.sqrt(HALF_SESSION_HOURS / FULL_SESSION_HOURS)

CANDIDATES: Final[tuple[str, ...]] = ("rw", "har", "har_vix", "garch")

#: Columnas de VIX que produce `features.volatility.vix_features`.
_VIX_COLUMNS: Final[tuple[str, ...]] = ("vix_level", "vix_zscore", "vix_percentile")


class Candidate(StrEnum):
    """Los cuatro candidatos pre-registrados, ni uno más."""

    RW = "rw"
    """Persistencia: ``rv̂_t = rv_{t-1}``. El *baseline* que hay que batir."""

    HAR = "har"
    """HAR en log-varianza con retardos 1 / 4 / 17 sesiones."""

    HAR_VIX = "har_vix"
    """HAR más las features de VIX de la sesión ``t-1``."""

    GARCH = "garch"
    """GARCH(1,1) media cero, errores normales, sin regresores exógenos."""


class Target(StrEnum):
    """Los dos objetivos de A1, evaluados por separado."""

    PARKINSON = "parkinson"
    """Primario: varianza realizada de Parkinson. Inmune a #52."""

    RET_SQ = "ret_sq"
    """Secundario: cuadrado del retorno de sesión. **No** inmune a #52."""


class Verdict(StrEnum):
    """Veredicto de la regla de selección pre-registrada."""

    SELECTED = "selected"
    """Hay un candidato que cumple las dos condiciones en los dos objetivos."""

    NO_BETTER_THAN_NAIVE = "no_better_than_naive"
    """Ninguno bate a ``rw``: el *fallback* es la persistencia."""

    INCONCLUSIVE = "inconclusive"
    """Los dos objetivos apuntan a candidatos distintos: no se elige."""


class Status(StrEnum):
    """Estado de un candidato en el estudio."""

    OK = "ok"
    UNAVAILABLE = "unavailable"


TARGETS: Final[tuple[Target, ...]] = (Target.PARKINSON, Target.RET_SQ)

#: Fórmulas y unidades que se declaran en el informe (A1).
TARGET_FORMULAS: Final[dict[str, str]] = {
    Target.PARKINSON: "rv_t = (ln(H_t / L_t))^2 / (4 ln 2)",
    Target.RET_SQ: "r_t^2 con r_t = ln(C_t / O_t)",
}

#: Columnas del frame de trabajo que alimentan cada objetivo.
TARGET_COLUMNS: Final[dict[str, str]] = {
    Target.PARKINSON: "parkinson_rv",
    Target.RET_SQ: "ret_sq",
}

#: El objetivo primario no lee el `open`; el secundario sí (#52).
TARGET_IMMUNE_TO_STALE_OPEN: Final[dict[str, bool]] = {
    Target.PARKINSON: True,
    Target.RET_SQ: False,
}


# ─────────────────────────────────────────────────────────────────────────────
# Resultado
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Metrics:
    """Métricas de un candidato sobre un objetivo y un tramo."""

    qlike: float
    mse_log: float
    sessions: int


@dataclass(frozen=True, slots=True)
class Fold:
    """Un tramo de reajuste del *walk-forward*."""

    index: int
    start: str
    end: str
    sessions: int
    metrics: dict[str, dict[str, Metrics | None]]


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Resultado de un candidato, con el motivo si no se pudo estimar."""

    name: str
    status: str
    reason: str | None
    metrics: dict[str, Metrics | None]
    median_sigma_bp: float | None


@dataclass(frozen=True, slots=True)
class Anchors:
    """Anclaje para #10 (A21): *forecast* y movimiento medidos, no inventados."""

    used_candidate: str
    median_forecast_sigma_bp: float
    median_abs_open_close_bp: float
    absolute_move_sessions: int
    per_candidate_sigma_bp: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class Selection:
    """Decisión de la regla pre-registrada, con la aritmética que la justifica."""

    selected: str | None
    verdict: str
    metric: str
    rule: tuple[str, ...]
    constants: dict[str, object]
    arithmetic: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class VolatilityStudy:
    """Resultado completo del estudio, listo para el informe."""

    series_id: str
    vix_series_id: str
    source: str
    as_of: datetime
    first_session: str
    last_session: str
    sessions: int
    stale_open_share: float
    clean_from: str | None
    clean_sessions: int
    by_year: tuple[dict[str, object], ...]
    exclusions: dict[str, int]
    exclusion_reasons: dict[str, str]
    evaluated_sessions: int
    first_evaluated: str | None
    last_evaluated: str | None
    folds: tuple[Fold, ...]
    candidates: tuple[CandidateResult, ...]
    selection: Selection
    anchors: Anchors
    rank_correlation: float | None
    rank_correlation_note: str
    limitations: tuple[str, ...]
    notes: tuple[str, ...] = field(default=())

    def candidate(self, name: str) -> CandidateResult:
        """Resultado de ese candidato."""
        for result in self.candidates:
            if result.name == name:
                return result
        raise KeyError(name)


# ─────────────────────────────────────────────────────────────────────────────
# Datos
# ─────────────────────────────────────────────────────────────────────────────
def _literal(value: str) -> str:
    """Literal SQL seguro: los identificadores vienen del registro validado."""
    return "'" + value.replace("'", "''") + "'"


def load_market(store: Store, *, series_id: str = SERIES_ID) -> pl.DataFrame:
    """Sesiones diarias de esa serie con OHLC, ordenadas por ``as_of``.

    Se usa ``store.sql()``, **no** ``read_pit``: la semántica de ``read_pit`` es
    «qué sabíamos en T», y con un ``fetched_at`` de 2026 y un ``at`` histórico no
    devuelve nada. Aquí se quiere «qué datos existían», que es el estado vigente
    del dataset.
    """
    query = (
        "SELECT as_of, open, high, low, close FROM raw.market_daily "  # noqa: S608
        f"WHERE series_id = {_literal(series_id)} "
        "AND open IS NOT NULL AND close IS NOT NULL ORDER BY as_of"
    )
    try:
        frame = store.sql(query).sort("as_of")
    except (UnknownDatasetError, duckdb.Error) as error:
        raise ConfigurationError(
            f"no hay dataset de mercado diario en {store.root}: {error}. "
            "Ejecuta antes la ingesta de mercado (tarea #3)."
        ) from error
    if frame.height < 2:
        raise ConfigurationError(
            f"no hay sesiones suficientes de {series_id!r} en el almacén: {frame.height}. "
            "Ejecuta antes la ingesta de mercado (tarea #3)."
        )
    return frame


def load_vix(store: Store, *, series_id: str = VIX_SERIES_ID) -> pl.DataFrame | None:
    """Cierres del VIX por sesión, o ``None`` si no está en el almacén.

    Que falte el VIX **no** aborta el estudio: el candidato ``har_vix`` se declara
    ``unavailable`` con el motivo y el informe se escribe igualmente (A13).
    """
    query = (
        "SELECT as_of, close AS vix_close FROM raw.market_daily "  # noqa: S608
        f"WHERE series_id = {_literal(series_id)} AND close IS NOT NULL ORDER BY as_of"
    )
    try:
        frame = store.sql(query)
    except (UnknownDatasetError, duckdb.Error) as error:
        logger.warning("sin VIX en el almacén ({}): har_vix no se podrá estimar", error)
        return None
    if frame.height < 2:
        return None
    return frame.with_columns(
        pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("session")
    ).select("session", "vix_close")


def _column(frame: pl.DataFrame, name: str) -> np.ndarray:
    """Columna como array de ``numpy``, con los nulos convertidos en ``NaN``."""
    return np.asarray(
        frame.get_column(name).cast(pl.Float64).fill_null(float("nan")).to_numpy(), dtype=float
    )


# ─────────────────────────────────────────────────────────────────────────────
# Muestra: definición única de #52 y exclusiones
# ─────────────────────────────────────────────────────────────────────────────
def _stale_open_rows(base: pl.DataFrame) -> dict[int, int]:
    """Sesiones con el ``open`` repetido, por año."""
    grouped = (
        base.group_by(pl.col("session").dt.year().alias("year"))
        .agg(pl.col("open_stale").sum().alias("stale_open_sessions"))
        .sort("year")
    )
    return {
        int(str(row["year"])): int(str(row["stale_open_sessions"]))
        for row in grouped.iter_rows(named=True)
    }


def _by_year(base: pl.DataFrame, clean_year: int | None) -> tuple[dict[str, object], ...]:
    """A2: por año, la proporción de ``open`` repetido y si el año es limpio."""
    stale_rows = _stale_open_rows(base)
    return tuple(
        {
            "year": year,
            "sessions": sessions,
            "stale_open_sessions": stale_rows.get(year, 0),
            "stale_open_share": stale_share,
            "clean": clean_year is not None and year >= clean_year,
        }
        for year, sessions, stale_share in yearly_stale_open_share(base)
    )


def build_sample(
    store: Store, *, series_id: str = SERIES_ID
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Muestra de trabajo (features) y resumen de la muestra y las exclusiones.

    Devuelve el frame **con features** (ya sin el artefacto #52, sin nulos y sin
    medias sesiones) y el resumen que va al informe.
    """
    market = load_market(store, series_id=series_id)
    base = session_stale_open(market).filter(pl.col("prev_close").is_not_null())
    cutoff = clean_sample_cutoff(base)
    # La **misma** función que usa el estudio del drift, con su mínimo de 250
    # sesiones: la regla de la muestra limpia (#52) es una sola en todo el proyecto.
    clean = clean_sample(base, cutoff=cutoff)
    if clean is None:
        raise ConfigurationError(
            "la muestra limpia de #52 no llega al mínimo de "
            f"{MIN_CLEAN_SESSIONS} sesiones (corte calculado: {cutoff})."
        )
    region = base if cutoff is None else base.filter(pl.col("session") >= cutoff)
    stale_excluded = region.height - clean.height

    has_ohlc = (
        pl.col("high").is_not_null() & pl.col("low").is_not_null() & pl.col("close").is_not_null()
    )
    with_ohlc = clean.filter(has_ohlc)
    null_excluded = clean.height - with_ohlc.height

    calendar = MarketCalendar()
    half_days = [
        day for day in with_ohlc.get_column("session").to_list() if calendar.is_half_day(day)
    ]
    working = with_ohlc.filter(~pl.col("session").is_in(half_days))
    half_excluded = with_ohlc.height - working.height

    vix = load_vix(store)
    if vix is not None:
        working = working.join(vix, on="session", how="left")

    features = add_features(working)
    summary: dict[str, object] = {
        "sessions": base.height,
        "stale_open_share": _mean(base.get_column("open_stale")),
        "clean_from": None if cutoff is None else cutoff.isoformat(),
        "clean_sessions": clean.height,
        "by_year": _by_year(base, None if cutoff is None else cutoff.year),
        "exclusions": {
            "stale_open": stale_excluded,
            "null_ohlc": null_excluded,
            "half_day": half_excluded,
            "total": stale_excluded + null_excluded + half_excluded,
        },
        "evaluated_sessions": features.height,
        "vix_sessions": features.get_column("vix_level").is_not_null().sum()
        if vix is not None
        else 0,
    }
    return features, summary


def _mean(column: pl.Series) -> float:
    """Media de una columna booleana, o 0.0 si está vacía."""
    value = column.mean()
    return float(value) if isinstance(value, (int, float)) else 0.0


def _as_float(value: object) -> float:
    """Número de un valor del resumen, o 0.0 si no lo es."""
    return float(value) if isinstance(value, (int, float)) else 0.0


def _as_int(value: object) -> int:
    """Entero de un valor del resumen, o 0 si no lo es."""
    return int(value) if isinstance(value, int) else 0


# ─────────────────────────────────────────────────────────────────────────────
# Modelos
# ─────────────────────────────────────────────────────────────────────────────
def _har_design(frame: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Matriz de regresores del HAR (o HAR+VIX) del frame completo."""
    return np.column_stack([_column(frame, name) for name in columns])


def _har_forecasts(
    frame: pl.DataFrame, *, columns: Sequence[str], bounds: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, str | None]:
    """Pronósticos one-step del HAR (o HAR+VIX) con reajuste en cada fold.

    El ajuste es en ``ln(rv)`` por mínimos cuadrados con ``numpy`` (sin
    ``statsmodels``): la varianza pronosticada es siempre ``> 0`` y, si el ajuste
    no es posible, queda ``NaN`` (que se reporta como *no disponible*), nunca 0,
    ``inf`` ni un valor inventado.
    """
    target = _column(frame, TARGET_COLUMNS[Target.PARKINSON])
    with np.errstate(divide="ignore", invalid="ignore"):
        log_target = np.where(target > 0.0, np.log(target), np.nan)
    design = _har_design(frame, columns)
    forecasts = np.full(target.shape, np.nan)
    usable = np.isfinite(design).all(axis=1) & np.isfinite(log_target)

    for start, end in bounds:
        train = np.flatnonzero(usable[:start])
        fitted = fit_log_har(design[train], log_target[train]) if train.size else None
        if fitted is None:
            return forecasts, (
                f"el ajuste en ln(rv) no es posible (filas de entrenamiento: {train.size})"
            )
        for index in range(start, end):
            if not usable[index]:
                continue
            predicted = fitted.predict(*(float(value) for value in design[index]))
            if predicted is not None:
                forecasts[index] = predicted
    return forecasts, None


@dataclass(frozen=True, slots=True)
class _GarchFit:
    """Parámetros ajustados de GARCH(1,1) y el estado de la varianza condicional."""

    omega: float
    alpha: float
    beta: float
    sigma2: float
    epsilon2: float


def _fit_garch(returns: np.ndarray) -> _GarchFit:
    """Ajusta GARCH(1,1) con ``arch``: media cero, errores normales, sin exógenas.

    ``rescale=False`` a propósito: los parámetros quedan en las unidades de la
    serie (fracciones), no en una escala interna, para que la recursión de la
    varianza condicional sea exacta.
    """
    model = arch.arch_model(
        returns, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False
    )
    result = cast("Any", model.fit(disp="off", show_warning=False))
    params = result.params
    fit = _GarchFit(
        omega=float(params["omega"]),
        alpha=float(params["alpha[1]"]),
        beta=float(params["beta[1]"]),
        sigma2=float(result.conditional_volatility[-1]) ** 2,
        epsilon2=float(result.resid[-1]) ** 2,
    )
    if not all(math.isfinite(value) for value in (fit.omega, fit.alpha, fit.beta, fit.sigma2)):
        raise ValueError("el ajuste GARCH no ha dado parámetros finitos")
    if fit.omega <= 0.0 or fit.alpha < 0.0 or fit.beta < 0.0 or fit.sigma2 <= 0.0:
        raise ValueError("el ajuste GARCH no ha dado una varianza admisible")
    return fit


def _garch_forecasts(
    returns: np.ndarray, *, bounds: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, str | None]:
    """Pronósticos de varianza a un paso de GARCH(1,1) con reajuste en cada fold.

    Dentro del fold los parámetros están **congelados** (el reajuste es cada 21
    sesiones, igual que para los demás candidatos) y la varianza condicional se
    actualiza con la recursión de GARCH(1,1):
    ``sigma²_t = ω + alpha·ε²_{t-1} + β·sigma²_{t-1}``. Es exactamente el pronóstico a un paso
    con la información disponible antes de la sesión ``t``.
    """
    forecasts = np.full(returns.shape, np.nan)
    for start, end in bounds:
        try:
            fit = _fit_garch(returns[:start])
        except Exception as error:  # `arch` falla de muchas formas distintas
            return forecasts, f"GARCH(1,1) no estimable: {type(error).__name__}: {error}"
        sigma2, epsilon2 = fit.sigma2, fit.epsilon2
        for index in range(start, end):
            forecasts[index] = sigma2
            epsilon2 = returns[index] ** 2
            sigma2 = fit.omega + fit.alpha * epsilon2 + fit.beta * sigma2
    return forecasts, None


def _metrics(
    forecast: np.ndarray, target: np.ndarray, *, mask: np.ndarray | None = None
) -> Metrics | None:
    """QLIKE en varianza y MSE en log-varianza sobre las filas evaluables.

    Solo cuentan las filas con pronóstico **finito y positivo** y objetivo
    **finito y positivo**: un objetivo nulo no tiene logaritmo y un pronóstico no
    positivo haría infinito el QLIKE. Cuántas han contado se reporta.
    """
    valid = np.isfinite(forecast) & (forecast > 0.0) & np.isfinite(target) & (target > 0.0)
    if mask is not None:
        valid &= mask
    if not bool(valid.any()):
        return None
    predicted = forecast[valid]
    observed = target[valid]
    qlike = float(np.mean(np.log(predicted) + observed / predicted))
    mse_log = float(np.mean((np.log(predicted) - np.log(observed)) ** 2))
    return Metrics(qlike=qlike, mse_log=mse_log, sessions=int(predicted.size))


def _fold_bounds(
    sessions: int, *, min_train: int = MIN_TRAIN, refit_every: int = REFIT_EVERY
) -> list[tuple[int, int]]:
    """Tramos de reajuste: ventana expansiva, reajuste cada ``refit_every`` sesiones."""
    bounds: list[tuple[int, int]] = []
    start = min_train
    while start < sessions:
        end = min(start + refit_every, sessions)
        bounds.append((start, end))
        start = end
    return bounds


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class WalkForward:
    """Resultado del *walk-forward*: folds, candidatos y correlación de rangos.

    ``forecasts`` expone el pronóstico de varianza de cada candidato (fracción²,
    ``NaN`` donde no hay). No va al informe, pero permite verificar que la sesión
    ``t`` no cambia al añadir datos posteriores (A10) y deja el dato listo para
    quien lo consuma después (#23).
    """

    bounds: tuple[tuple[int, int], ...]
    folds: tuple[Fold, ...]
    candidates: tuple[CandidateResult, ...]
    rank_correlation: float | None
    rank_correlation_note: str
    forecasts: dict[str, np.ndarray]


def walk_forward(frame: pl.DataFrame) -> WalkForward:
    """Ejecuta el esquema pre-registrado para los cuatro candidatos."""
    sessions = frame.height
    bounds = _fold_bounds(sessions)
    if not bounds:
        raise ConfigurationError(
            f"no se puede estimar ningún candidato: {sessions} sesiones utilizables y el "
            f"mínimo de entrenamiento es {MIN_TRAIN}."
        )

    parkinson = _column(frame, TARGET_COLUMNS[Target.PARKINSON])
    ret_sq = _column(frame, TARGET_COLUMNS[Target.RET_SQ])
    returns = _column(frame, "ret_log")
    targets: dict[str, np.ndarray] = {
        Target.PARKINSON: parkinson,
        Target.RET_SQ: ret_sq,
    }

    # `rw`: persistencia, sin parámetros. Con `rv_{t-1}` como pronóstico de `rv_t`.
    forecasts: dict[str, np.ndarray] = {Candidate.RW: np.roll(parkinson, 1)}
    forecasts[Candidate.RW][0] = np.nan
    reasons: dict[str, str | None] = {Candidate.RW: None}

    har_forecast, har_reason = _har_forecasts(
        frame, columns=("har_lag1", "har_lag4", "har_lag17"), bounds=bounds
    )
    forecasts[Candidate.HAR] = har_forecast
    reasons[Candidate.HAR] = har_reason

    if all(name in frame.columns for name in _VIX_COLUMNS):
        har_vix, har_vix_reason = _har_forecasts(
            frame,
            columns=("har_lag1", "har_lag4", "har_lag17", *_VIX_COLUMNS),
            bounds=bounds,
        )
    else:
        har_vix, har_vix_reason = (
            np.full(sessions, np.nan),
            (f"no hay {VIX_SERIES_ID} en el almacén para construir las features de VIX"),
        )
    forecasts[Candidate.HAR_VIX] = har_vix
    reasons[Candidate.HAR_VIX] = har_vix_reason

    try:
        garch_forecast, garch_reason = _garch_forecasts(returns, bounds=bounds)
    except Exception as error:  # `arch` falla de muchas formas distintas
        garch_forecast, garch_reason = (
            np.full(sessions, np.nan),
            f"GARCH(1,1) no estimable: {type(error).__name__}: {error}",
        )
    forecasts[Candidate.GARCH] = garch_forecast
    reasons[Candidate.GARCH] = garch_reason

    evaluated = np.zeros(sessions, dtype=bool)
    evaluated[bounds[0][0] :] = True

    results: list[CandidateResult] = []
    for name in CANDIDATES:
        forecast = forecasts[name]
        overall: dict[str, Metrics | None] = {}
        for target in TARGETS:
            overall[target.value] = _metrics(forecast, targets[target.value], mask=evaluated)
        sigma = forecast[evaluated]
        sigma = sigma[np.isfinite(sigma) & (sigma > 0.0)]
        status = (
            Status.OK
            if any(value is not None for value in overall.values())
            else Status.UNAVAILABLE
        )
        reason = reasons[name]
        if status is Status.UNAVAILABLE and reason is None:
            reason = "no se ha podido producir ningún pronóstico evaluable"
        results.append(
            CandidateResult(
                name=name,
                status=status.value,
                reason=reason if status is Status.UNAVAILABLE else None,
                metrics=overall,
                median_sigma_bp=float(np.median(np.sqrt(sigma))) * 10_000.0 if sigma.size else None,
            )
        )

    folds: list[Fold] = []
    for index, (start, end) in enumerate(bounds):
        mask = np.zeros(sessions, dtype=bool)
        mask[start:end] = True
        per_candidate: dict[str, dict[str, Metrics | None]] = {}
        for name in CANDIDATES:
            per_candidate[name] = {
                target.value: _metrics(forecasts[name], targets[target.value], mask=mask)
                for target in TARGETS
            }
        folds.append(
            Fold(
                index=index,
                start=str(frame.get_column("session")[start]),
                end=str(frame.get_column("session")[end - 1]),
                sessions=end - start,
                metrics=per_candidate,
            )
        )

    correlation, note = _rank_correlation(results)
    return WalkForward(
        bounds=tuple(bounds),
        folds=tuple(folds),
        candidates=tuple(results),
        rank_correlation=correlation,
        rank_correlation_note=note,
        forecasts=forecasts,
    )


def _rank_correlation(results: Sequence[CandidateResult]) -> tuple[float | None, str]:
    """Correlación de rangos de Spearman entre los dos rankings de QLIKE (A15).

    Con dos candidatos por objetivo el coeficiente no siempre está definido (una
    serie constante): en ese caso se declara en vez de devolver un número falso.
    """
    first, second = Target.PARKINSON.value, Target.RET_SQ.value
    complete = [
        result
        for result in results
        if result.metrics.get(first) is not None and result.metrics.get(second) is not None
    ]
    if len(complete) < 3:
        return None, (
            "no se calcula: hacen falta al menos tres candidatos con las dos métricas "
            f"(hay {len(complete)})."
        )
    primary = [cast("Metrics", result.metrics[first]).qlike for result in complete]
    secondary = [cast("Metrics", result.metrics[second]).qlike for result in complete]
    result_any: Any = cast(
        "Any",
        stats.spearmanr(primary, secondary),  # pyright: ignore[reportUnknownMemberType]
    )
    coefficient = getattr(result_any, "statistic", None)
    if coefficient is None:
        coefficient = getattr(result_any, "correlation", None)
    if not isinstance(coefficient, (int, float)) or math.isnan(coefficient):
        return None, "no definida (alguna serie de QLIKE es constante)."
    return float(coefficient), f"Spearman sobre {len(complete)} candidatos."


# ─────────────────────────────────────────────────────────────────────────────
# Regla de selección pre-registrada (A16)
# ─────────────────────────────────────────────────────────────────────────────
SELECTION_RULE: Final[tuple[str, ...]] = (
    "Gana el candidato que sea el mejor o esté dentro del 2 % relativo del mejor "
    "en los DOS objetivos y que además supere a `rw` en más del 2 % relativo de "
    "QLIKE en los DOS objetivos.",
    "Si ninguno lo cumple, el veredicto es `no_better_than_naive` y el fallback es `rw`.",
    "Si los dos objetivos se contradicen (el mejor de cada uno es distinto y ninguno "
    "cumple la condición completa), el veredicto es `inconclusive` y se declaran los "
    "candidatos empatados.",
    "Si varios candidatos cumplen, gana el de mejor QLIKE en el objetivo PRIMARIO "
    "(el primario manda). Las tolerancias son las declaradas y no se cambian después "
    "de ver el resultado.",
)


def select_candidate(results: Sequence[CandidateResult]) -> Selection:
    """Aplica la regla pre-registrada, tal cual, y devuelve la aritmética."""
    primary, secondary = Target.PARKINSON.value, Target.RET_SQ.value
    available = {
        result.name: result
        for result in results
        if result.status == Status.OK.value
        and result.metrics.get(primary) is not None
        and result.metrics.get(secondary) is not None
    }

    def qlike(name: str, target: str) -> float:
        return cast("Metrics", available[name].metrics[target]).qlike

    arithmetic: list[dict[str, object]] = []
    verdict = Verdict.NO_BETTER_THAN_NAIVE.value
    selected: str | None = None

    if available:
        best_primary = min(available, key=lambda name: qlike(name, primary))
        best_secondary = min(available, key=lambda name: qlike(name, secondary))
        rw_primary = qlike(Candidate.RW, primary) if Candidate.RW in available else None
        rw_secondary = qlike(Candidate.RW, secondary) if Candidate.RW in available else None

        eligible: list[str] = []
        for name in available:
            within_primary = qlike(name, primary) <= qlike(best_primary, primary) * (
                1 + RELATIVE_TOLERANCE
            )
            within_secondary = qlike(name, secondary) <= qlike(best_secondary, secondary) * (
                1 + RELATIVE_TOLERANCE
            )
            beats_primary = rw_primary is not None and qlike(name, primary) < rw_primary * (
                1 - RELATIVE_TOLERANCE
            )
            beats_secondary = rw_secondary is not None and qlike(name, secondary) < rw_secondary * (
                1 - RELATIVE_TOLERANCE
            )
            qualifies = within_primary and within_secondary and beats_primary and beats_secondary
            if qualifies:
                eligible.append(name)
            arithmetic.append(
                {
                    "candidate": name,
                    "qlike_primary": qlike(name, primary),
                    "qlike_secondary": qlike(name, secondary),
                    "relative_vs_best_primary": qlike(name, primary) / qlike(best_primary, primary)
                    - 1.0,
                    "relative_vs_best_secondary": qlike(name, secondary)
                    / qlike(best_secondary, secondary)
                    - 1.0,
                    "relative_vs_rw_primary": (qlike(name, primary) / rw_primary - 1.0)
                    if rw_primary
                    else None,
                    "relative_vs_rw_secondary": (qlike(name, secondary) / rw_secondary - 1.0)
                    if rw_secondary
                    else None,
                    "within_2pct_of_best": within_primary and within_secondary,
                    "beats_rw_by_more_than_2pct": beats_primary and beats_secondary,
                    "eligible": qualifies,
                }
            )

        if eligible:
            selected = min(eligible, key=lambda name: qlike(name, primary))
            verdict = Verdict.SELECTED.value
        elif best_primary != best_secondary:
            verdict = Verdict.INCONCLUSIVE.value
        else:
            verdict = Verdict.NO_BETTER_THAN_NAIVE.value

    constants: dict[str, object] = {
        "metric": "qlike",
        "primary": primary,
        "secondary": secondary,
        "candidates": list(CANDIDATES),
        "relative_tolerance": RELATIVE_TOLERANCE,
        "fallback": Candidate.RW.value,
        "selection_rule": "within 2% of best in both AND beat rw by >2% in both",
        "min_train": MIN_TRAIN,
        "refit_every": REFIT_EVERY,
        "horizon": HORIZON,
    }
    return Selection(
        selected=selected,
        verdict=verdict,
        metric="qlike",
        rule=SELECTION_RULE,
        constants=constants,
        arithmetic=tuple(arithmetic),
    )


# ─────────────────────────────────────────────────────────────────────────────
# API de servicio (A20)
# ─────────────────────────────────────────────────────────────────────────────
def scale_sigma_for_duration(
    sigma_fraction: float, *, duration_hours: float = FULL_SESSION_HOURS
) -> float:
    """sigma de servicio de una sesión de esa duración, en fracción.

    Regla declarada (A20): la volatilidad escala con la **raíz del tiempo**, así que
    una media sesión de 3,5 h lleva ``sigma × √(3,5 / 6,5)``. Se **excluye** del ajuste
    y de la comparación del estudio (una media sesión no es comparable con una
    sesión completa), pero **al servir** el *forecast* se escala en vez de dejarlo
    sin respuesta: es la opción que `plan.md` §4.1 permite como alternativa a
    excluir el día.
    """
    if duration_hours <= 0.0:
        raise ValueError("la duración de la sesión debe ser positiva")
    return sigma_fraction * math.sqrt(duration_hours / FULL_SESSION_HOURS)


# ─────────────────────────────────────────────────────────────────────────────
# Estudio
# ─────────────────────────────────────────────────────────────────────────────
def analyse(
    *,
    data_root: Path,
    now: datetime,
    series_id: str = SERIES_ID,
    vix_series_id: str = VIX_SERIES_ID,
    source: str = SOURCE,
    reports_dir: Path | None = None,
) -> VolatilityStudy:
    """Lee el almacén, ejecuta el *walk-forward*, decide y escribe el informe.

    Lanza :class:`ConfigurationError` si el almacén no da para estimar **ningún**
    candidato: en ese caso **no se escribe ningún informe** (A23).
    """
    store = Store(data_root)
    frame, sample = build_sample(store, series_id=series_id)
    walk = walk_forward(frame)

    if all(result.status == Status.UNAVAILABLE.value for result in walk.candidates):
        reasons = "; ".join(f"{r.name}: {r.reason}" for r in walk.candidates)
        raise ConfigurationError(f"no se puede estimar ningún candidato. {reasons}")

    selection = select_candidate(walk.candidates)
    bounds = walk.bounds
    first_index, last_index = bounds[0][0], bounds[-1][1] - 1
    chosen = selection.selected or Candidate.RW.value
    chosen_result = next(result for result in walk.candidates if result.name == chosen)

    session_column = frame.get_column("session")
    simple_return = (_column(frame, "close") / _column(frame, "open") - 1.0) * 10_000.0
    absolute = simple_return[np.isfinite(simple_return)]

    study = VolatilityStudy(
        series_id=series_id,
        vix_series_id=vix_series_id,
        source=source,
        as_of=now,
        first_session=str(session_column[0]),
        last_session=str(session_column[-1]),
        sessions=_as_int(sample["sessions"]),
        stale_open_share=_as_float(sample["stale_open_share"]),
        clean_from=cast("str | None", sample["clean_from"]),
        clean_sessions=_as_int(sample["clean_sessions"]),
        by_year=cast("tuple[dict[str, object], ...]", sample["by_year"]),
        exclusions=cast("dict[str, int]", sample["exclusions"]),
        exclusion_reasons={
            "stale_open": (
                "el `open` de la sesión es el cierre anterior repetido (artefacto #52): "
                "el tramo nocturno sería cero por construcción"
            ),
            "null_ohlc": "`high`, `low` o `close` nulos: no se imputan a cero ni con la media",
            "half_day": (
                "media sesión (cierre a las 13:00 ET, 3,5 h): no es comparable con una sesión "
                "completa; se escala al servir con √(3,5/6,5)"
            ),
        },
        evaluated_sessions=frame.height,
        first_evaluated=str(session_column[first_index]),
        last_evaluated=str(session_column[last_index]),
        folds=walk.folds,
        candidates=walk.candidates,
        selection=selection,
        anchors=Anchors(
            used_candidate=chosen,
            median_forecast_sigma_bp=chosen_result.median_sigma_bp
            if chosen_result.median_sigma_bp is not None
            else 0.0,
            median_abs_open_close_bp=float(np.median(np.abs(absolute))) if absolute.size else 0.0,
            absolute_move_sessions=int(absolute.size),
            per_candidate_sigma_bp={
                result.name: result.median_sigma_bp for result in walk.candidates
            },
        ),
        rank_correlation=walk.rank_correlation,
        rank_correlation_note=walk.rank_correlation_note,
        limitations=(
            "Se mide sobre `^GSPC`, **no** sobre el CFD: no hay fuente del bid/ask ni del "
            "intradía del SPX500:CFD (issue #50).",
            "El objetivo es un **proxy** de la varianza, no la varianza observada. La "
            "referencia intradía (RV de barras de 5 min) es la issue #57.",
            "La muestra anterior al corte limpio está contaminada por el artefacto del "
            "`open` repetido (#52). El objetivo primario (Parkinson) es inmune por "
            "construcción porque no lee el `open`; el secundario (`r_t²`) **no** lo es.",
            "Se comparan 4 candidatos **pre-registrados**, no el resultado de una búsqueda: "
            "la búsqueda de especificaciones e hiperparámetros es la issue #58 (y exige la "
            "corrección por comparaciones múltiples de #16).",
            "Un resultado bueno o malo aquí **no dice nada** sobre la viabilidad de la "
            "estrategia: eso es la tarea #9 (informe de Fase 0).",
        ),
        notes=(
            f"Esquema pre-registrado: ventana expansiva, min_train = {MIN_TRAIN} sesiones, "
            f"reajuste cada {REFIT_EVERY} sesiones para TODOS los candidatos, pronóstico a un "
            "paso. Ningún candidato se reajusta más a menudo que otro.",
            "Las medias sesiones se excluyen del ajuste y de la comparación; al **servir** el "
            f"forecast se multiplica por √(3,5/6,5) = {HALF_DAY_SIGMA_SCALE:.6f} "
            f"(`{scale_sigma_for_duration.__name__}`).",
            "La clave de sesión se deriva en `America/New_York`, nunca de la fecha UTC: las "
            "sesiones a caballo del cambio de hora quedan contiguas y sin duplicados.",
            f"La regla de la muestra limpia (#52) es la **definición única** importada de "
            f"`analysis/drift.py` (tolerancia del {STALE_OPEN_TOLERANCE:.0%} de `open` "
            f"repetido, mínimo de {MIN_CLEAN_SESSIONS} sesiones).",
            f"El HAR se estima con `numpy.linalg.lstsq` en `ln(rv)` (sin `statsmodels`: no "
            f"está aprobado y aquí no se pide inferencia). Calentamiento del HAR: "
            f"{HAR_WARMUP} sesiones.",
            "Determinismo: dos ejecuciones con el mismo `--now` producen el mismo JSON byte a "
            "byte, GARCH incluido.",
        ),
    )
    if reports_dir is not None:
        json_path, markdown_path = write_report(study, reports_dir)
        logger.info("informe de volatilidad: {} y {}", json_path, markdown_path)
    return study


# ─────────────────────────────────────────────────────────────────────────────
# Informe
# ─────────────────────────────────────────────────────────────────────────────
def report_payload(study: VolatilityStudy) -> dict[str, object]:
    """Payload JSON, con el bloque legible por máquina que consume #23 (A17)."""
    return {
        "as_of": study.as_of,
        "series_id": study.series_id,
        "vix_series_id": study.vix_series_id,
        "source": study.source,
        "targets": {
            target.value: {
                "formula": TARGET_FORMULAS[target],
                "units": "fracción² en el cálculo; bp (×10⁴) y su raíz al informar",
                "immune_to_stale_open_52": TARGET_IMMUNE_TO_STALE_OPEN[target],
            }
            for target in TARGETS
        },
        "sample": {
            "first_session": study.first_session,
            "last_session": study.last_session,
            "sessions": study.sessions,
            "stale_open_share": study.stale_open_share,
            "clean_from": study.clean_from,
            "clean_sessions": study.clean_sessions,
            "by_year": list(study.by_year),
            "exclusions": study.exclusions,
            "exclusion_reasons": study.exclusion_reasons,
        },
        "features": {
            "evaluated_sessions": study.evaluated_sessions,
            "first_evaluated": study.first_evaluated,
            "last_evaluated": study.last_evaluated,
        },
        "walk_forward": {
            "window": "expanding",
            "min_train": MIN_TRAIN,
            "refit_every": REFIT_EVERY,
            "horizon": HORIZON,
            "folds": len(study.folds),
        },
        "selection": {
            "selected": study.selection.selected,
            "verdict": study.selection.verdict,
            "metric": study.selection.metric,
            "rule": list(study.selection.rule),
            "constants": study.selection.constants,
            "arithmetic": list(study.selection.arithmetic),
        },
        "candidates": {
            result.name: {
                "status": result.status,
                "reason": result.reason,
                **{
                    target.value: None
                    if result.metrics.get(target.value) is None
                    else asdict(cast("Metrics", result.metrics[target.value]))
                    for target in TARGETS
                },
            }
            for result in study.candidates
        },
        "folds": [
            {
                "index": fold.index,
                "start": fold.start,
                "end": fold.end,
                "sessions": fold.sessions,
                **{
                    name: {
                        target.value: None
                        if fold.metrics[name][target.value] is None
                        else asdict(cast("Metrics", fold.metrics[name][target.value]))
                        for target in TARGETS
                    }
                    for name in CANDIDATES
                },
            }
            for fold in study.folds
        ],
        "rank_correlation": {
            "spearman": study.rank_correlation,
            "note": study.rank_correlation_note,
        },
        "anchors": asdict(study.anchors),
        "limitations": list(study.limitations),
        "notes": list(study.notes),
    }


def write_report(study: VolatilityStudy, directory: Path) -> tuple[Path, Path]:
    """Escribe el informe del estudio en JSON y Markdown."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"volatility_forecast_{study.as_of.date().isoformat()}"
    json_path = directory / f"{stem}.json"
    markdown_path = directory / f"{stem}.md"
    payload = report_payload(study)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(study), encoding="utf-8")
    return json_path, markdown_path


def _target_table(study: VolatilityStudy, target: Target, *, title: str) -> list[str]:
    """Tabla de QLIKE y MSE de todos los candidatos para un objetivo."""
    lines = [
        f"### {title}",
        "",
        "| candidato | estado | QLIKE | MSE log | sesiones |",
        "|---|---|---|---|---|",
    ]
    for result in study.candidates:
        metrics = result.metrics.get(target.value)
        if metrics is None:
            lines.append(f"| `{result.name}` | `{result.status}` | — | — | 0 |")
            continue
        lines.append(
            f"| `{result.name}` | `{result.status}` | {metrics.qlike:.6f} | "
            f"{metrics.mse_log:.6f} | {metrics.sessions} |"
        )
    lines.append("")
    return lines


def render_markdown(study: VolatilityStudy) -> str:
    """Informe legible, empezando por lo que se decide."""
    selection = study.selection
    verdict_text = {
        Verdict.SELECTED: f"Candidato elegido: **`{selection.selected}`**.",
        Verdict.NO_BETTER_THAN_NAIVE: (
            "Ningún candidato bate a la persistencia: el **fallback es `rw`**."
        ),
        Verdict.INCONCLUSIVE: (
            "Los dos objetivos apuntan a candidatos distintos: **no se elige** ninguno."
        ),
    }[Verdict(selection.verdict)]

    lines = [
        "# Volatilidad realizada, VIX y primer *forecast* (tarea #7)",
        "",
        f"- **Serie:** `{study.series_id}` (fuente `{study.source}`), VIX: `{study.vix_series_id}`",
        f"- **Sesiones utilizables:** {study.evaluated_sessions} "
        f"(de {study.first_session} a {study.last_session})",
        f"- **Calculado:** {study.as_of.isoformat()}",
        "- **Métrica pre-registrada:** QLIKE en varianza (ranking por QLIKE medio); "
        "MSE del logaritmo como secundaria",
        f"- **Veredicto:** `{selection.verdict}` — {verdict_text}",
        "",
        "## Anclaje para #10 (barreras sobre números medidos)",
        "",
        f"- Mediana del *forecast* sigma de `{study.anchors.used_candidate}` en la ventana de "
        f"evaluación: **{study.anchors.median_forecast_sigma_bp:.1f} bp por sesión**.",
        f"- Mediana de `|open→close|` de la muestra limpia: "
        f"**{study.anchors.median_abs_open_close_bp:.1f} bp** "
        f"({study.anchors.absolute_move_sessions} sesiones).",
        "",
        "| candidato | sigma mediana (bp/sesión) |",
        "|---|---|",
    ]
    for name, sigma in study.anchors.per_candidate_sigma_bp.items():
        value = "—" if sigma is None else f"{sigma:.1f}"
        lines.append(f"| `{name}` | {value} |")

    lines.extend(
        [
            "",
            "## Objetivos y unidades",
            "",
            f"- **Primario** — Parkinson: `{TARGET_FORMULAS[Target.PARKINSON]}`. Usa solo "
            "`high`/`low`: **inmune** al artefacto del `open` repetido (#52) por construcción.",
            f"- **Secundario** — `{TARGET_FORMULAS[Target.RET_SQ]}`. Lee el `open`: **no** es "
            "inmune a #52.",
            "- Unidades: fracción² en el cálculo; bp (×10⁴, y su raíz) al informar, como en "
            "`analysis/drift.py`.",
            "",
            "## Muestra limpia (#52) y exclusiones",
            "",
            f"- Sesiones en el almacén: **{study.sessions}**; proporción con el `open` repetido "
            f"(cierre anterior): {study.stale_open_share:.1%}.",
            f"- Año desde el que la muestra es limpia (**calculado con el dato**, no escrito a "
            f"mano): **{study.clean_from}**; sesiones de la muestra limpia: "
            f"**{study.clean_sessions}**.",
            f"- Excluidas del ajuste y de la evaluación: `open` repetido "
            f"**{study.exclusions['stale_open']}**, `high`/`low`/`close` nulos "
            f"**{study.exclusions['null_ohlc']}**, medias sesiones "
            f"**{study.exclusions['half_day']}** (total {study.exclusions['total']}).",
            "- Motivos, uno a uno: "
            + "; ".join(f"**{key}** — {text}" for key, text in study.exclusion_reasons.items())
            + ".",
            "",
            "| año | sesiones | `open` repetido | proporción | ¿limpio? |",
            "|---|---|---|---|---|",
        ]
    )
    for row in study.by_year:
        share = _as_float(row["stale_open_share"])
        lines.append(
            f"| {row['year']} | {row['sessions']} | {row['stale_open_sessions']} | "
            f"{share:.1%} | {'sí' if row['clean'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Esquema *walk-forward* pre-registrado",
            "",
            f"- Ventana **expansiva**, `min_train = {MIN_TRAIN}` sesiones.",
            f"- **Reajuste cada {REFIT_EVERY} sesiones para TODOS los candidatos** (mismo ritmo "
            "para todos: ninguno juega con ventaja).",
            f"- Pronósticos **a un paso** ({HORIZON}), evaluados desde "
            f"{study.first_evaluated} hasta {study.last_evaluated}.",
            f"- Folds: **{len(study.folds)}** tramos.",
            "",
            "| fold | inicio | fin | sesiones |",
            "|---|---|---|---|",
        ]
    )
    for fold in study.folds:
        lines.append(f"| {fold.index} | {fold.start} | {fold.end} | {fold.sessions} |")

    lines.extend(["", "## Métricas por candidato", ""])
    lines.extend(
        _target_table(
            study, Target.PARKINSON, title=f"Objetivo primario — {Target.PARKINSON.value}"
        )
    )
    lines.extend(
        _target_table(study, Target.RET_SQ, title=f"Objetivo secundario — {Target.RET_SQ.value}")
    )
    correlation = (
        "no calculada" if study.rank_correlation is None else f"{study.rank_correlation:+.3f}"
    )
    lines.extend(
        [
            f"**Correlación de rangos de Spearman entre los dos rankings:** {correlation}. "
            f"{study.rank_correlation_note}",
            "",
            "Se declara que el objetivo **primario (Parkinson) es inmune a #52** (no lee el "
            "`open`) y el **secundario no**: la correlación se lee sabiendo eso.",
            "",
            "## Regla de selección pre-registrada y su aritmética",
            "",
        ]
    )
    lines.extend(f"{index}. {item}" for index, item in enumerate(selection.rule, start=1))
    lines.extend(
        [
            "",
            "| candidato | QLIKE priori | QLIKE secu | vs mejor (pri) | vs mejor (sec) | "
            "vs `rw` (pri) | vs `rw` (sec) | ¿elegible? |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in selection.arithmetic:
        lines.append(
            f"| `{row['candidate']}` | {_as_float(row['qlike_primary']):.6f} | "
            f"{_as_float(row['qlike_secondary']):.6f} | "
            f"{_as_float(row['relative_vs_best_primary']):+.2%} | "
            f"{_as_float(row['relative_vs_best_secondary']):+.2%} | "
            f"{_as_float(row['relative_vs_rw_primary']):+.2%} | "
            f"{_as_float(row['relative_vs_rw_secondary']):+.2%} | "
            f"{'sí' if row['eligible'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"**Veredicto:** `{selection.verdict}`"
            + (f" ⇒ `{selection.selected}`" if selection.selected else " ⇒ fallback `rw`")
            + ".",
        ]
    )

    unavailable = [
        result for result in study.candidates if result.status == Status.UNAVAILABLE.value
    ]
    lines.extend(["", "## Candidatos no disponibles", ""])
    if unavailable:
        lines.extend(f"- `{result.name}`: {result.reason}" for result in unavailable)
    else:
        lines.append("- Ninguno: los cuatro candidatos se han podido estimar.")

    lines.extend(["", "## Limitaciones (declaradas, no escondidas)", ""])
    lines.extend(f"- {item}" for item in study.limitations)
    lines.extend(["", "## Notas", ""])
    lines.extend(f"- {item}" for item in study.notes)
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del estudio."""
    parser = argparse.ArgumentParser(
        prog="cfdtrader.analysis.volatility_forecast", description=__doc__
    )
    parser.add_argument("--data-root", type=Path, default=None, help="raíz del almacén")
    parser.add_argument("--settings", type=Path, default=None, help="ruta de settings.yaml")
    parser.add_argument("--series", default=SERIES_ID, help="serie a analizar")
    parser.add_argument("--vix-series", default=VIX_SERIES_ID, help="serie del VIX")
    parser.add_argument("--now", default=None, help="instante de referencia ISO (tests)")
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.settings)
    except ConfigurationError as error:
        logger.error("configuración inválida: {}", error)
        return 1

    data_root = args.data_root if args.data_root is not None else settings.data.root
    now = _parse_now(args.now)
    try:
        study = analyse(
            data_root=data_root,
            now=now,
            series_id=args.series,
            vix_series_id=args.vix_series,
            reports_dir=data_root / "derived" / "reports",
        )
    except ConfigurationError as error:
        logger.error("no se puede hacer el estudio: {}", error)
        return 2

    logger.info(
        "volatilidad: veredicto {} (candidato {}), sigma mediana {:.1f} bp/sesión",
        study.selection.verdict,
        study.selection.selected or Candidate.RW.value,
        study.anchors.median_forecast_sigma_bp,
    )
    return 0


def _parse_now(value: str | None) -> datetime:
    """Instante de referencia: el de verdad, o el que fije el test."""
    if value is None:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
