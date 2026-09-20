"""Features de regimen y volatilidad (`regime_v1`, tarea #23).

**Familia cerrada.** Este modulo entrega el calculo completo de ``regime_v1``: las
siete columnas declaradas en
:data:`cfdtrader.features.store.REGIME_FEATURE_CATALOG` — el percentil expandido de
la volatilidad realizada de Parkinson, el **pronostico GARCH(1,1) elegido en #7**
con su z-score robusto, el *efficiency ratio* de Kaufman y tres columnas de
calendario (dia de la semana, sesiones hasta el proximo vencimiento mensual y
marca de vencimiento trimestral). No entrega ninguna otra: una columna fuera del
catalogo no es de esta familia.

**No reimplementa.** La varianza realizada de Parkinson y el retorno de sesion se
importan de :mod:`cfdtrader.features.volatility` (#7), la **definicion unica** del
proyecto; el percentil de ventana expandida y la normalizacion robusta se importan
de ahi mismo (#7) y de :mod:`cfdtrader.features.store` (#19); y el motor
GARCH(1,1) —ajuste, pronostico a un paso y tramos de reajuste— se importa de
:mod:`cfdtrader.features.volatility`, que es donde vive desde #23 para que
``features/`` no dependa de la capa de estudios (``tech_stack.md`` §4.6 y regla 5).

Sin *look-ahead*
----------------

La fila ``t`` es el **estado al cierre de ``t``** y ninguna columna lee una sesion
posterior. El catalogo declara, columna a columna, que sesion mira cada una
(``required_as_of``):

- ``rv_percentile`` compara la volatilidad realizada de ``t-1`` con las
  **anteriores** a ella: cierre de ``t-1``.
- ``garch_forecast`` es el pronostico **de la sesion ``t``** hecho con ``ret_log``
  hasta ``T = t-1`` (es el emparejamiento del QLIKE de #7, sin un paso de retraso);
  su ``_z`` hereda el mismo corte.
- ``efficiency_ratio_20`` cierra en ``t``: es la unica columna de precio que usa la
  sesion corriente.
- Las tres de calendario se conocen de antemano.

Lo no computable se publica ``null``
------------------------------------

Un ajuste GARCH que no se puede estimar no lanza: deja el pronostico a ``null``.
Un ``NaN`` o un ``inf`` se convierten en ``null`` antes de salir, y
:func:`cfdtrader.features.store.daily_records` lo vuelve a verificar al escribir.

El modulo es **puro** (entra un ``pl.DataFrame``, sale otro), **no lee el reloj**,
no consulta el calendario del proyecto (los vencimientos se derivan del **propio
frame**) y no toca el sistema de ficheros: la persistencia la hace
:func:`cfdtrader.features.store.save_daily` con la spec de :func:`regime_spec`.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Final, cast

import numpy as np
import polars as pl

from cfdtrader.features.store import (
    DEFAULT_REGIME_SOURCES,
    DEFAULT_REGIME_WINDOWS,
    REGIME_EFFICIENCY_WINDOW,
    REGIME_FEATURE_COLUMNS,
    REGIME_FEATURE_SET,
    REGIME_MIN_SESSIONS,
    FeatureSpec,
    InvalidFeatureMatrixError,
    RegimeInputError,
    normalise_expanding,
)
from cfdtrader.features.volatility import (
    GARCH_MIN_TRAIN,
    GARCH_REFIT_EVERY,
    _expanding_percentile,  # pyright: ignore[reportPrivateUsage]
    garch_fold_bounds,
    garch_forecasts,
    parkinson_variance,
    session_returns,
)

__all__ = [
    "REGIME_INPUT_COLUMNS",
    "regime_matrix",
    "regime_spec",
]

#: Columnas de entrada que necesita el calculo. ``open`` **si** hace falta: el
#: regresor del GARCH es ``ret_log = ln(C/O)``, el mismo objetivo de #7.
REGIME_INPUT_COLUMNS: Final[tuple[str, ...]] = ("session", "open", "high", "low", "close")

#: Nombre de la columna del objetivo de #7 que alimenta el GARCH. Son las claves
#: de ``CatalogEntry.name``: los nombres del codigo y los del catalogo coinciden.
_PARKINSON: Final[str] = "parkinson_rv"
_RET_LOG: Final[str] = "ret_log"
_RV_PERCENTILE: Final[str] = "rv_percentile"
_GARCH_FORECAST: Final[str] = "garch_forecast"
_EFFICIENCY_RATIO: Final[str] = "efficiency_ratio_20"
_DAY_OF_WEEK: Final[str] = "day_of_week"
_SESSIONS_TO_OPEX: Final[str] = "sessions_to_opex"
_ES_ROLL: Final[str] = "is_es_roll_session"

#: Meses cuyo vencimiento es el *roll* trimestral del futuro ES (mar/jun/sep/dic).
_QUARTER_MONTHS: Final[tuple[int, ...]] = (3, 6, 9, 12)

#: Columnas que se publican como entero (las tres de calendario).
_INTEGER_COLUMNS: Final[tuple[str, ...]] = (_DAY_OF_WEEK, _SESSIONS_TO_OPEX, _ES_ROLL)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de entrada
# ─────────────────────────────────────────────────────────────────────────────
def _numeric_values(frame: pl.DataFrame, *, column: str) -> list[float | None]:
    """Columna del frame como ``float`` con ``None`` en los huecos.

    Un valor no numerico es un error tipado, y se comprueba **antes** de construir
    cualquier expresion: una columna de texto tiene que dar este error, no uno de
    polars. Un ``NaN`` o un ``inf`` es un hueco (``None``).
    """
    values: list[float | None] = []
    for value in cast("list[object]", frame.get_column(column).to_list()):
        if value is None:
            values.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RegimeInputError(f"la columna '{column}' no es numerica: {type(value).__name__}")
        number = float(value)
        values.append(number if math.isfinite(number) else None)
    return values


def _finite_or_null(frame: pl.DataFrame, *, columns: tuple[str, ...]) -> pl.DataFrame:
    """Convierte ``NaN``/``inf`` en ``null`` en las columnas de coma flotante."""
    return frame.with_columns(
        *(
            pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).alias(name)
            for name in columns
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Calendario derivado del propio frame (sin `MarketCalendar` ni IO)
# ─────────────────────────────────────────────────────────────────────────────
def _third_friday(year: int, month: int) -> date:
    """Tercer viernes del mes: la fecha de vencimiento de las opciones mensuales."""
    first = date(year, month, 1)
    first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
    return first_friday + timedelta(days=14)


def _opex_session(sessions: Sequence[date], target: date) -> date:
    """Sesion OPEX de un mes: el tercer viernes **rodado** al ultimo dia de sesion ``<=`` el.

    Si el tercer viernes no es sesion (festivo) —Viernes Santo, Juneteenth— el
    vencimiento cae en la sesion inmediatamente anterior. Es la regla que la
    realidad impone y la que difiere del calendario trimestral del proyecto:
    ``MarketCalendar.is_es_roll`` marca el tercer viernes **nominal**, aunque el
    mercado este cerrado (issue #77).
    """
    position = bisect.bisect_right(sessions, target) - 1
    if position < 0:
        raise InvalidFeatureMatrixError(
            f"el vencimiento {target.isoformat()} cae antes de la primera sesion del frame"
        )
    return sessions[position]


def _opex_sessions(sessions: Sequence[date]) -> tuple[date, ...]:
    """Sesiones OPEX del frame: una por mes, de la primera a la ultima sesion.

    Un mes cuyo tercer viernes cae **fuera** del frame no aporta vencimiento: el
    frame no puede inventar una sesion que no tiene. Por eso los ultimos dias sin
    vencimiento por delante quedan a ``null`` en ``sessions_to_opex``.
    """
    first, last = sessions[0], sessions[-1]
    found: list[date] = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        target = _third_friday(year, month)
        if first <= target <= last:
            found.append(_opex_session(sessions, target))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return tuple(dict.fromkeys(found))


def _calendar_columns(sessions: Sequence[date]) -> tuple[list[int], list[int | None], list[int]]:
    """``day_of_week``, ``sessions_to_opex`` e ``is_es_roll_session``, sesion a sesion.

    ``sessions_to_opex`` cuenta **posiciones hasta la proxima sesion OPEX** (0 en
    ella misma) y es ``None`` cuando la siguiente cae fuera del frame. El
    vencimiento de un mes es el tercer viernes rodado (ver
    :func:`_opex_session`).
    """
    opex = set(_opex_sessions(sessions))
    day_of_week = [session.isoweekday() for session in sessions]
    to_opex: list[int | None] = [None] * len(sessions)
    following: int | None = None
    for position in range(len(sessions) - 1, -1, -1):
        if sessions[position] in opex:
            following = position
        if following is not None:
            to_opex[position] = following - position
    is_roll = [
        1 if (session in opex and session.month in _QUARTER_MONTHS) else 0 for session in sessions
    ]
    return day_of_week, to_opex, is_roll


# ─────────────────────────────────────────────────────────────────────────────
# GARCH(1,1): el pronostico elegido en #7, una fila por sesion
# ─────────────────────────────────────────────────────────────────────────────
def _garch_variance(frame: pl.DataFrame) -> np.ndarray:
    """Varianza pronosticada (fraccion²) de cada sesion con la informacion ``<= t-1``.

    Usa el **mismo** motor que el estudio de #7 —tramos de reajuste cada 21
    sesiones y recursion de la varianza condicional dentro del tramo—, importado
    de :mod:`cfdtrader.features.volatility`: aqui no hay una segunda copia del
    GARCH. Una serie que no se puede estimar (menos de ``GARCH_MIN_TRAIN``
    sesiones, valores no finitos o un ajuste que falla) deja ``NaN``, que sale como
    ``null``: nunca una excepcion ni un numero inventado.
    """
    returns = np.asarray(frame.get_column(_RET_LOG).to_numpy(), dtype=float)
    if not bool(np.isfinite(returns).all()):
        return np.full(returns.shape, np.nan)
    bounds = garch_fold_bounds(
        returns.shape[0], min_train=GARCH_MIN_TRAIN, refit_every=GARCH_REFIT_EVERY
    )
    forecasts, _reason = garch_forecasts(returns, bounds=bounds)
    return np.asarray(forecasts, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Spec y matriz
# ─────────────────────────────────────────────────────────────────────────────
def regime_spec() -> FeatureSpec:
    """Spec por defecto de ``regime_v1``: el catalogo entero y una sola fuente.

    Los valores por defecto de :class:`FeatureSpec` son los de ``volatility_v1``
    (familias distintas), asi que la spec de regimen se declara **explicitamente**:
    sus ventanas son las de su catalogo y su fuente es el indice, **sin** VIX.
    """
    return FeatureSpec(
        feature_set=REGIME_FEATURE_SET,
        windows=DEFAULT_REGIME_WINDOWS,
        sources=DEFAULT_REGIME_SOURCES,
    )


def regime_matrix(frame: pl.DataFrame, *, spec: FeatureSpec) -> pl.DataFrame:
    """Matriz persistible de ``regime_v1``: ``session`` + las siete features (A1).

    Parameters
    ----------
    frame:
        Frame con ``session`` y OHLC. Las cinco columnas son obligatorias: el
        GARCH consume ``ret_log = ln(C/O)`` y la volatilidad realizada necesita
        ``high``/``low``.
    spec:
        Spec de la familia de regimen. Tiene que declarar ``regime_v1``: las otras
        matrices tienen su propia funcion de calculo.

    Returns
    -------
    pl.DataFrame
        ``session`` mas una columna por entrada del catalogo, **en el orden del
        catalogo**, ordenado por sesion y con lo no computable a ``null``.
    """
    if spec.feature_set != REGIME_FEATURE_SET:
        raise InvalidFeatureMatrixError(
            f"'regime_matrix' es la entrada de '{REGIME_FEATURE_SET}': la familia "
            f"'{spec.feature_set}' tiene su propia funcion de calculo"
        )
    missing = [name for name in REGIME_INPUT_COLUMNS if name not in frame.columns]
    if missing:
        raise RegimeInputError(
            f"faltan columnas de entrada para calcular la familia de regimen: {missing}"
        )

    ordered = frame.sort("session")
    # La validacion de tipos va **antes** de cualquier aritmetica: una columna de
    # texto tiene que dar un error tipado, no un fallo de polars a mitad del calculo.
    for name in ("open", "high", "low", "close"):
        _numeric_values(ordered, column=name)
    # Los precios se calculan en coma flotante: una columna de enteros o de todo
    # nulos (dtype `Null`) no puede tumbar una expresion de polars.
    ordered = ordered.with_columns(
        *(pl.col(name).cast(pl.Float64) for name in ("open", "high", "low", "close"))
    )
    sessions = [cast("date", value) for value in ordered.get_column("session").to_list()]

    # El objetivo de #7 y el regresor del GARCH salen de la definicion unica.
    computed = session_returns(parkinson_variance(ordered))
    computed = computed.with_columns(
        pl.Series(
            _RV_PERCENTILE,
            _expanding_percentile(computed, _PARKINSON, REGIME_MIN_SESSIONS),
            nan_to_null=True,
        ),
        pl.Series(_GARCH_FORECAST, _garch_variance(computed), nan_to_null=True),
    )
    # El *efficiency ratio* cierra en `t`: numerador de 20 sesiones y denominador
    # con las 20 diferencias que terminan en `t`. Sin ventana parcial (`20` nulos).
    movement = (pl.col("close") - pl.col("close").shift(REGIME_EFFICIENCY_WINDOW)).abs()
    path = (
        pl.col("close")
        .diff()
        .abs()
        .rolling_sum(window_size=REGIME_EFFICIENCY_WINDOW, min_samples=REGIME_EFFICIENCY_WINDOW)
    )
    computed = computed.with_columns(
        pl.when(path > 0.0).then(movement / path).otherwise(None).alias(_EFFICIENCY_RATIO),
    )
    day_of_week, to_opex, is_roll = _calendar_columns(sessions)
    computed = computed.with_columns(
        pl.Series(_DAY_OF_WEEK, day_of_week, dtype=pl.Int64),
        pl.Series(_SESSIONS_TO_OPEX, to_opex, dtype=pl.Int64),
        pl.Series(_ES_ROLL, is_roll, dtype=pl.Int64),
    )
    # Un `NaN`/`inf` se limpia antes de normalizar (`normalise_expanding` rechaza
    # los valores no finitos) y el resultado se normaliza con la regla de #19.
    computed = _finite_or_null(computed, columns=(_RV_PERCENTILE, _GARCH_FORECAST))
    computed = normalise_expanding(computed, _GARCH_FORECAST, min_sessions=REGIME_MIN_SESSIONS)
    columns = (_RV_PERCENTILE, _GARCH_FORECAST, f"{_GARCH_FORECAST}_z", _EFFICIENCY_RATIO)
    computed = _finite_or_null(computed, columns=columns)
    return computed.select(["session", *REGIME_FEATURE_COLUMNS]).sort("session")
