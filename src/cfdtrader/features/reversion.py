"""Features de sobre-extension y reversion (`reversion_v1`, tarea #159).

**Familia candidata, no produccion.** Este modulo entrega el calculo completo de
``reversion_v1``: las seis columnas declaradas en
:data:`cfdtrader.features.store.REVERSION_FEATURE_CATALOG` —la *term structure* del
VIX en sus tres vertices, la amplitud sectorial, la racha alcista del indice y el
percentil del retorno de cinco sesiones—. Ninguna de las seis esta en
``models.baseline.BASELINE_FEATURES``: se materializan para poder **medirlas**. Una
columna fuera del catalogo no es de esta familia.

**Que se mide aqui y por que.** El 2026-10-16 el modelo servido dio ``p(up) = 0,5959``
(LONG) y la descomposicion SHAP del margen mostro que **ninguna** de las diez columnas
del control codifica "sobre-extension": el optimismo de otras plazas entraba como
*alcista* (``asia_overnight_1`` +0,163, ``europe_prev_1`` +0,042). Estas seis
candidatas son la materializacion de la tabla de #159, medida con el snapshot de
decision (08:45-09:00 ET) antes de escribirlas: su insumo ya esta en el almacen y ya
es visible a esa hora. La decision de presupuesto (``amplia`` / ``intercambia`` /
``mantiene_control``) es la issue de seguimiento con ``analysis/feature_budget``,
exactamente como #146 siguio a #143.

**No reimplementa.** El percentil de ventana expandida sigue la **misma** convencion
que ``vix_percentile`` de #7 (:func:`cfdtrader.features.volatility.vix_features`):
fraccion de los valores **anteriores** que no superan al actual, con un minimo de
sesiones declarado y nunca sobre la muestra completa (`_docs/plan.md` §9). La
convencion esta cruzada celda a celda contra esa columna por un test de esta familia;
no hay una segunda definicion de "percentil de ventana expandida".

Interfaz
--------

:func:`reversion_matrix` recibe un ``Mapping`` con **una entrada por serie**
(:data:`cfdtrader.features.store.REVERSION_SERIES`: 15 claves) y no un frame largo,
por el mismo motivo que las familias de contexto (#21) y commodities (#143): cada
mercado trae su **propio calendario** y un frame largo obligaria a inventar la
alineacion antes de calcular. De cada frame se leen **solo** ``session`` y ``close``;
del de ``^GSPC``, ademas, el ``as_of`` de cada fila, que se **copia** (el modulo no lee
el reloj). Falta una serie, sobra una clave, falta ``session`` o ``close`` o se repite
una sesion: :class:`cfdtrader.features.store.ReversionInputError`, con la serie en el
mensaje.

Insumos y procedencia
---------------------

============================  ==============================  ====================
columna                       insumo                          fuente
============================  ==============================  ====================
``vix_ts_slope``              ``^VIX3M`` y ``^VIX9D``         ``raw.market_daily``
``vix_contango``              ``^VIX`` y ``^VIX3M``           ``raw.market_daily``
``vix9d_vix``                 ``^VIX9D`` y ``^VIX``           ``raw.market_daily``
``breadth``                   los once ETF sectoriales        ``raw.sectors``
``streak_up``                 ``^GSPC``                       ``raw.market_daily``
``ret_5_pct``                 ``^GSPC``                       ``raw.market_daily``
============================  ==============================  ====================

Los tres vertices del VIX se ingieren desde #58 (``^VIX9D`` desde 2011-01-03,
``^VIX3M`` desde 2006-07-17) y **ninguna** de las 57 columnas del catalogo anterior
los usaba: solo los leia el estudio del forecast de volatilidad. Ese es el hallazgo que
justifica esta familia.

Las formulas
------------

Con ``t`` la sesion de la fila, ``s(t)`` la **ultima sesion publicada** de cada serie
(``< t``, el rezago declarado de la familia) y ``C`` su cierre::

    vix_ts_slope_t = VIX3M_{s(t)} / VIX9D_{s(t)} - 1
    vix_contango_t = VIX_{s(t)}   / VIX3M_{s(t)} - 1
    vix9d_vix_t    = VIX9D_{s(t)} / VIX_{s(t)}   - 1
    breadth_t      = #{sectores con retorno > 0 en s(t)} / #{sectores con dato en s(t)}
    streak_up_t    = sesiones consecutivas con C_u > C_{u-1} al cierre de s(t)
    ret_5_pct_t    = percentil expandido de ln(C_v / C_{v-5}) en v = s(t)

Tres de las seis son **algebraicamente dependientes**, y se declara en vez de
esconderse::

    (1 + vix_contango) * (1 + vix9d_vix) * (1 + vix_ts_slope) = 1

de modo que dos cualesquiera determinan la tercera. El catalogo publica las tres —son
tres candidatas distintas de la tabla de #159, con percentiles muy distintos— y la
regla de intercambio del presupuesto (#146) es quien decide el solape con
``vix_zscore``/``vix_percentile``. ``breadth`` se solapa parcialmente con
``sector_dispersion_1`` (misma fuente, otra formula) y ``ret_5_pct`` normaliza el
``ret_5`` de ``technical_v1``: la candidata es su **percentil**, no el retorno.

El **denominador** de ``breadth`` es el numero de sectores **con dato** en ``s(t)``, el
mismo que publica ``sector_count`` en ``context_v1`` (#21): los ETF que aun no existian
no aportan, no se les inventa un cero ni se anula la amplitud por su culpa (``XLRE``
nace en 2015 y ``XLC`` en 2018). Un test de esta familia cruza ese denominador contra
``sector_count`` de la **misma** matriz.

Sin *look-ahead*
----------------

- La fila ``t`` usa la **ultima sesion publicada** de cada serie con el rezago
  declarado (**1 sesion**): el cierre de ``^VIX``/``^VIX9D``/``^VIX3M`` de ``t-1``, los
  retornos sectoriales de ``t-1`` y los cierres del indice hasta ``t-1``. Nunca una
  sesion posterior ni una barra del ancla de ``t``.
- La unica columna cuya formula **si** mira el cierre de ``t`` es ``ret_5_pct``, porque
  su base es ``ln(C_t / C_{t-5})``: por eso su ``required_as_of`` es el cierre de la
  sesion ``t`` y su ``design_lag`` es 1, igual que ``ret_5`` en ``technical_v1``.
- El percentil de ``ret_5_pct`` es de ventana **expandida** (minimo
  :data:`cfdtrader.features.store.REVERSION_MIN_SESSIONS` sesiones), nunca de la
  muestra completa.
- Un festivo ajeno **no** produce un cero: el retorno de una serie es el de su ultima
  sesion con cierre, y si no hay historia la columna es ``null``. Nada se rellena hacia
  adelante.

``^VIX9D`` antes del 2011-01-01 no existe (``min_start`` de #58): las candidatas que lo
leen salen ``null`` —un dato declarado, no un cero— y **no** se rellenan con otro
vertice.

Lo no computable se publica ``null``
------------------------------------

Un cierre nulo o no positivo deja la division sin definir y el resultado se publica
``null``: un ``null`` es un dato ("aqui no hay valor"), un ``NaN`` es un veneno que se
propaga. La escritura lo verifica otra vez:
:func:`cfdtrader.features.store.daily_records` rechaza un ``NaN`` o un ``inf`` en
cualquier columna del catalogo.

El modulo es **puro** (entra un ``Mapping`` de frames, sale otro frame), **no lee el
reloj** y no toca el sistema de ficheros: la persistencia la hace
:func:`cfdtrader.features.store.save_daily` con la spec de :func:`reversion_spec`.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final

import polars as pl

from cfdtrader.features.store import (
    DEFAULT_REVERSION_SOURCES,
    DEFAULT_REVERSION_WINDOWS,
    REVERSION_ANCHOR_SERIES,
    REVERSION_FEATURE_COLUMNS,
    REVERSION_FEATURE_SET,
    REVERSION_MIN_SESSIONS,
    REVERSION_RETURN_WINDOW,
    REVERSION_SECTOR_SERIES,
    REVERSION_SERIES,
    FeatureSpec,
    InvalidFeatureMatrixError,
    ReversionInputError,
)

__all__ = [
    "REVERSION_INPUT_COLUMNS",
    "reversion_matrix",
    "reversion_spec",
]

#: Columnas de entrada que necesita el calculo: la sesion y el cierre de cada serie.
REVERSION_INPUT_COLUMNS: Final[tuple[str, ...]] = ("session", "close")

#: Columna del instante de cada fila del ancla: se **copia**, no se calcula.
AS_OF_COLUMN: Final[str] = "as_of"

#: Sesiones que la familia retrocede para leer el cierre de cada serie (el rezago declarado).
REVERSION_LAG_SESSIONS: Final[int] = 1

#: Los tres vertices, por nombre, en el orden del catalogo.
_SLOPE_LONG: Final[str] = "^VIX3M"
_SLOPE_SHORT: Final[str] = "^VIX9D"
_VIX: Final[str] = "^VIX"

#: Nombres de las seis columnas del catalogo.
_TS_SLOPE: Final[str] = "vix_ts_slope"
_CONTANGO: Final[str] = "vix_contango"
_VIX9D_VIX: Final[str] = "vix9d_vix"
_BREADTH: Final[str] = "breadth"
_STREAK_UP: Final[str] = "streak_up"
_RET_5_PCT: Final[str] = "ret_5_pct"

#: Las tres columnas de la *term structure* y el par de vertices del que sale cada una.
_TERM_STRUCTURE: Final[tuple[tuple[str, str, str], ...]] = (
    (_TS_SLOPE, _SLOPE_LONG, _SLOPE_SHORT),
    (_CONTANGO, _VIX, _SLOPE_LONG),
    (_VIX9D_VIX, _SLOPE_SHORT, _VIX),
)


# ─────────────────────────────────────────────────────────────────────────────
# Series de entrada
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class _Series:
    """Una serie de entrada, ordenada por sesion, con sus retornos ya calculados.

    ``returns[i]`` es ``ln(C_i / C_{i-1})`` en el calendario de **esta** serie y vale ``None``
    cuando no es computable (primera sesion, cierre nulo o no positivo). No es un dato que
    falte: es la forma de no inventar un cero.
    """

    series_id: str
    sessions: list[date]
    closings: list[float | None]
    returns: list[float | None]


def _sessions(frame: pl.DataFrame, *, series_id: str) -> list[date]:
    """Sesiones de un frame, en orden de fila y sin repetir (A4)."""
    if "session" not in frame.columns:
        raise ReversionInputError(
            f"la serie '{series_id}' no trae la columna 'session': de cada serie se leen "
            f"{list(REVERSION_INPUT_COLUMNS)}"
        )
    values: list[date] = []
    seen: set[date] = set()
    for value in frame.get_column("session").to_list():
        if not isinstance(value, date) or isinstance(value, datetime):
            raise ReversionInputError(
                f"la columna 'session' de '{series_id}' trae {value!r}, que no es una fecha"
            )
        if value in seen:
            raise ReversionInputError(
                f"la serie '{series_id}' repite la sesion {value.isoformat()}: una serie de "
                "entrada tiene una fila por sesion y la eleccion de cual vale no se hace aqui"
            )
        seen.add(value)
        values.append(value)
    return values


def _closings(frame: pl.DataFrame, *, series_id: str) -> list[float | None]:
    """Cierres de un frame, con ``None`` en los huecos (un ``NaN`` es un hueco, no un dato)."""
    if "close" not in frame.columns:
        raise ReversionInputError(
            f"la serie '{series_id}' no trae la columna 'close': de cada serie se leen "
            f"{list(REVERSION_INPUT_COLUMNS)}"
        )
    values: list[float | None] = []
    for value in frame.get_column("close").to_list():
        if value is None:
            values.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReversionInputError(
                f"la columna 'close' de '{series_id}' trae {value!r}: tiene que ser numerica"
            )
        number = float(value)
        values.append(number if math.isfinite(number) else None)
    return values


def _instants(frame: pl.DataFrame, *, series_id: str) -> list[datetime]:
    """El ``as_of`` de cada fila del ancla: se copia al frame de salida, no se recalcula."""
    if AS_OF_COLUMN not in frame.columns:
        raise ReversionInputError(
            f"el ancla '{series_id}' no trae la columna '{AS_OF_COLUMN}': la fila persistida es "
            "la del instante del almacen y no se inventa un reloj"
        )
    instants: list[datetime] = []
    for value in frame.get_column(AS_OF_COLUMN).to_list():
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ReversionInputError(
                f"el '{AS_OF_COLUMN}' de '{series_id}' ({value!r}) no es un instante con zona "
                "horaria"
            )
        instants.append(value)
    return instants


def _log_returns(closings: list[float | None]) -> list[float | None]:
    """``ln(C_i / C_{i-1})``, con ``None`` en el primer elemento y donde no es computable."""
    returns: list[float | None] = [None] * len(closings)
    for index in range(1, len(closings)):
        current = closings[index]
        previous = closings[index - 1]
        if current is None or previous is None or current <= 0.0 or previous <= 0.0:
            continue
        returns[index] = math.log(current / previous)
    return returns


def _series_of(frame: pl.DataFrame, *, series_id: str) -> _Series:
    """Una serie de un frame, con sus dos columnas ordenadas por sesion."""
    sessions = _sessions(frame, series_id=series_id)
    closings = _closings(frame, series_id=series_id)
    triples = sorted(zip(sessions, closings, strict=True))
    ordered_sessions = [item[0] for item in triples]
    ordered_closings = [item[1] for item in triples]
    return _Series(
        series_id=series_id,
        sessions=ordered_sessions,
        closings=ordered_closings,
        returns=_log_returns(ordered_closings),
    )


def _anchored_series(frame: pl.DataFrame) -> tuple[_Series, list[datetime]]:
    """El ancla: su serie y el ``as_of`` de cada sesion, ordenados igual."""
    series_id = REVERSION_ANCHOR_SERIES
    sessions = _sessions(frame, series_id=series_id)
    closings = _closings(frame, series_id=series_id)
    instants = _instants(frame, series_id=series_id)
    triples = sorted(zip(sessions, closings, instants, strict=True))
    ordered_sessions = [item[0] for item in triples]
    ordered_closings = [item[1] for item in triples]
    ordered_instants = [item[2] for item in triples]
    for session, instant in zip(ordered_sessions, ordered_instants, strict=True):
        if instant.date() != session:
            raise ReversionInputError(
                f"la sesion {session.isoformat()} de '{series_id}' y su 'as_of' "
                f"({instant.isoformat()}) no corresponden al mismo dia: el 'as_of' de la fila "
                "es el que entra en el hash"
            )
    anchor = _Series(
        series_id=series_id,
        sessions=ordered_sessions,
        closings=ordered_closings,
        returns=_log_returns(ordered_closings),
    )
    return anchor, ordered_instants


# ─────────────────────────────────────────────────────────────────────────────
# Alineacion y formulas
# ─────────────────────────────────────────────────────────────────────────────
def _last_index(series: _Series, target: date) -> int | None:
    """Posicion de la **ultima sesion publicada** de la serie, con el rezago declarado.

    ``REVERSION_LAG_SESSIONS`` es 1, asi que la fila ``t`` lee la ultima sesion ``< target``:
    no se rellena ni se desplaza nada.
    """
    position = bisect.bisect_left(series.sessions, target) - REVERSION_LAG_SESSIONS
    return position if position >= 0 else None


def _closing_at(series: _Series, target: date) -> float | None:
    """Cierre publicado de la serie en su ultima sesion con el rezago declarado.

    Un cierre nulo o no positivo no es un cierre: devuelve ``None`` en vez de un cero.
    """
    position = _last_index(series, target)
    if position is None:
        return None
    value = series.closings[position]
    if value is None or value <= 0.0:
        return None
    return value


def _ratio_at(numerator: _Series, denominator: _Series, target: date) -> float | None:
    """``CIERRE(num) / CIERRE(den) - 1`` con los cierres publicados, o ``None``."""
    top = _closing_at(numerator, target)
    bottom = _closing_at(denominator, target)
    if top is None or bottom is None:
        return None
    return top / bottom - 1.0


def _breadth_at(series: Mapping[str, _Series], target: date) -> float | None:
    """Fraccion de los sectores **con dato** en su ultima sesion ``< t`` con retorno > 0.

    El denominador son los sectores con retorno computable en esa sesion —el mismo numero que
    publica ``sector_count`` en ``context_v1``—: un ETF que todavia no existia no aporta, no se
    le inventa un cero ni se anula la amplitud por su culpa.
    """
    advances = 0
    present = 0
    for name in REVERSION_SECTOR_SERIES:
        position = _last_index(series[name], target)
        if position is None:
            continue
        value = series[name].returns[position]
        if value is None:
            continue
        present += 1
        if value > 0.0:
            advances += 1
    if present == 0:
        return None
    return advances / present


def _streak_at(anchor: _Series, index: int) -> float | None:
    """Sesiones consecutivas alcistas **hasta la sesion ``index``**, o ``None`` sin historia.

    La racha se lee en los **cierres** (``C_u > C_{u-1}``) y no en un logaritmo: el signo no
    admite una diferencia de 1 ULP. Un hueco en los cierres rompe la racha (el estado no se
    arrastra). La fila de la sesion ``t`` publica la racha **hasta ``t-1``**.
    """
    current = index - REVERSION_LAG_SESSIONS
    if current < 0:
        return None
    streak = 0
    position = current
    while position >= 1:
        value = anchor.closings[position]
        previous = anchor.closings[position - 1]
        if value is None or previous is None or value <= previous:
            break
        streak += 1
        position -= 1
    return float(streak)


def _return_at(anchor: _Series, index: int, *, window: int) -> float | None:
    """``ln(C_i / C_{i-window})`` sobre el ancla, o ``None`` si la ventana no cabe.

    Es la base de ``ret_5_pct``: la **misma** definicion que ``ret_5`` en ``technical_v1``
    (``ln(C_t / C_{t-5})``), y un test de esta familia la cruza contra esa columna.
    """
    position = index - window
    if position < 0:
        return None
    current = anchor.closings[index]
    previous = anchor.closings[position]
    if current is None or previous is None or current <= 0.0 or previous <= 0.0:
        return None
    return math.log(current / previous)


def _expanding_percentile(values: list[float | None], *, min_sessions: int) -> list[float | None]:
    """Percentil de cada valor dentro de los **anteriores**: la convencion de `vix_percentile`.

    Fraccion de los valores anteriores (no nulos) que **no superan** al actual, exigiendo un
    minimo de ``min_sessions`` valores anteriores. Mientras no haya historia suficiente el
    valor es ``None``, nunca una muestra completa (`_docs/plan.md` §9): normalizar sobre todo
    el conjunto es *look-ahead*. Se sostiene una copia ordenada de la historia para no recorrer
    el vector entero en cada fila.
    """
    out: list[float | None] = [None] * len(values)
    history: list[float] = []
    for index, value in enumerate(values):
        if value is None:
            continue
        if len(history) >= min_sessions:
            out[index] = bisect.bisect_right(history, value) / len(history)
        bisect.insort(history, value)
    return out


def _finite_or_null(frame: pl.DataFrame, *, columns: tuple[str, ...]) -> pl.DataFrame:
    """Convierte ``NaN``/``inf`` en ``null`` en las columnas indicadas (A11).

    Es la misma fontaneria que usan las familias tecnica y de contexto: ``is_finite`` propaga
    los nulos, asi que la conversion no inventa un valor donde no lo habia.
    """
    return frame.with_columns(
        *(
            pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).alias(name)
            for name in columns
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Spec y matriz
# ─────────────────────────────────────────────────────────────────────────────
def reversion_spec() -> FeatureSpec:
    """Spec por defecto de ``reversion_v1``: el catalogo entero y las quince series.

    Los valores por defecto de :class:`FeatureSpec` son los de ``volatility_v1`` (familias
    distintas), asi que la spec de sobre-extension se declara **explicitamente**: sus ventanas
    son las de su catalogo y sus fuentes, el ancla, los tres vertices del VIX y los once
    sectores.
    """
    return FeatureSpec(
        feature_set=REVERSION_FEATURE_SET,
        windows=DEFAULT_REVERSION_WINDOWS,
        sources=DEFAULT_REVERSION_SOURCES,
    )


def reversion_matrix(frames: Mapping[str, pl.DataFrame], *, spec: FeatureSpec) -> pl.DataFrame:
    """Matriz persistible de ``reversion_v1``: ``session`` + ``as_of`` + las seis features.

    Parameters
    ----------
    frames:
        ``Mapping`` con una entrada por serie de :data:`REVERSION_SERIES` (15 claves). De cada
        una se leen ``session`` y ``close``; del ancla, ademas, el ``as_of``. Un frame **sin
        filas** es una serie sin historia: sus features salen ``null`` (y no cuentan en el
        denominador de ``breadth``).
    spec:
        Spec de la familia de sobre-extension. Tiene que declarar ``reversion_v1``: cada
        familia tiene su propia funcion de calculo.

    Returns
    -------
    pl.DataFrame
        Una fila por sesion de ``^GSPC``, en orden, con ``session``, ``as_of`` y las seis
        columnas del catalogo, en su orden; lo no computable a ``null``.
    """
    if spec.feature_set != REVERSION_FEATURE_SET:
        raise InvalidFeatureMatrixError(
            f"'reversion_matrix' es la entrada de '{REVERSION_FEATURE_SET}': la familia "
            f"'{spec.feature_set}' tiene su propia funcion de calculo"
        )
    missing = sorted(set(REVERSION_SERIES) - set(frames))
    unknown = sorted(set(frames) - set(REVERSION_SERIES))
    if missing or unknown:
        raise ReversionInputError(
            f"el 'Mapping' de series no es el de '{REVERSION_FEATURE_SET}': faltan {missing} y "
            f"sobran {unknown}"
        )

    anchor, instants = _anchored_series(frames[REVERSION_ANCHOR_SERIES])
    series = {
        name: _series_of(frames[name], series_id=name)
        for name in REVERSION_SERIES
        if name != REVERSION_ANCHOR_SERIES
    }

    term_structure: dict[str, list[float | None]] = {name: [] for name, _, _ in _TERM_STRUCTURE}
    breadth: list[float | None] = []
    streak: list[float | None] = []
    base_returns: list[float | None] = []
    for index, session in enumerate(anchor.sessions):
        for name, numerator, denominator in _TERM_STRUCTURE:
            term_structure[name].append(_ratio_at(series[numerator], series[denominator], session))
        breadth.append(_breadth_at(series, session))
        streak.append(_streak_at(anchor, index))
        base_returns.append(_return_at(anchor, index, window=REVERSION_RETURN_WINDOW))

    data: dict[str, pl.Series] = {
        "session": pl.Series("session", anchor.sessions, dtype=pl.Date()),
        AS_OF_COLUMN: pl.Series(AS_OF_COLUMN, instants, dtype=pl.Datetime("us", "UTC")),
    }
    for name, _, _ in _TERM_STRUCTURE:
        data[name] = pl.Series(name, term_structure[name], dtype=pl.Float64)
    data[_BREADTH] = pl.Series(_BREADTH, breadth, dtype=pl.Float64)
    data[_STREAK_UP] = pl.Series(_STREAK_UP, streak, dtype=pl.Float64)
    # El percentil es de ventana expandida y se calcula sobre el `ret_5` del propio ancla: es la
    # misma definicion que publica `technical_v1` (un test la cruza contra esa columna).
    data[_RET_5_PCT] = pl.Series(
        _RET_5_PCT,
        _expanding_percentile(base_returns, min_sessions=REVERSION_MIN_SESSIONS),
        dtype=pl.Float64,
    )

    frame = _finite_or_null(pl.DataFrame(data), columns=REVERSION_FEATURE_COLUMNS)
    return frame.select(["session", AS_OF_COLUMN, *REVERSION_FEATURE_COLUMNS]).sort("session")
