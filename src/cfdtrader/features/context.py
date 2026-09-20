"""Features de contexto de mercado (`context_v1`, tarea #21).

**Familia cerrada.** Este modulo entrega el calculo completo de ``context_v1``:
las once columnas declaradas en
:data:`cfdtrader.features.store.CONTEXT_FEATURE_CATALOG` (cuatro correlaciones
moviles con los indices europeos y el Nikkei, el overnight asiatico, el cierre
europeo anterior, la beta del VIX, el retorno del dolar, la dispersion sectorial
y su z-score de ventana expandida, y el recuento de sectores con dato). La matriz
**no** publica ningun retorno del S&P 500: ``ret_1``/``ret_5``/``ret_21`` son de
``technical_v1`` (#20) y el solape de nombres entre familias sigue siendo solo
``atr_norm`` (politica de nombres, #72).

**No reimplementa.** La normalizacion robusta de ventana expandida se **importa**
de :mod:`cfdtrader.features.store` (#19); aqui no hay una segunda formula de
mediana ni de MAD.

Interfaz
--------

:func:`context_matrix` recibe un ``Mapping`` con **una entrada por serie**
(:data:`cfdtrader.features.store.CONTEXT_SERIES`: 19 claves) y no un frame
largo. El motivo es que cada mercado trae su **propio calendario**: un frame
largo obligaria a inventar la alineacion antes de calcular, y la alineacion es
justo la decision que esta familia tiene que hacer explicita. De cada frame se
leen **solo** ``session`` y ``close``; del de ``^GSPC``, ademas, el ``as_of`` de
cada fila, que se **copia** (el modulo no lee el reloj). Falta una serie, sobra
una clave, falta ``session`` o ``close`` o se repite una sesion:
:class:`cfdtrader.features.store.ContextInputError`, con la serie en el mensaje.

Alineacion entre mercados
-------------------------

El ``as_of`` del almacen es el **sello del cierre de la sesion americana** en
**todas** las series, extranjeras incluidas: no dice cuando cerro un mercado
ajeno. La alineacion sale, por tanto, del **orden de los cierres** y se declara
en :data:`CONTEXT_MARKET_LAG`:

- **0 sesiones** (``^N225``, ``^HSI``): Tokio y Hong Kong cierran **antes** de la
  apertura americana, asi que su sesion ``t`` ya esta publicada cuando el S&P
  abre.
- **1 sesion** (``^GDAXI``, ``^FTSE``, ``^STOXX50E``, ``DX-Y.NYB``, ``^VIX``):
  Europa, el dolar y el VIX cierran **despues**; su ultimo dato conocido a ``t``
  es el de la sesion anterior.

Con el rezago, la sesion ``t`` usa la **ultima sesion publicada** de cada mercado
(la ultima ``<= t`` con rezago 0, la ultima ``< t`` con rezago 1). Un festivo
ajeno **no** produce un cero: el retorno de una sesion con hueco es el de su
ultima sesion con cierre, y si no hay historia la feature es ``null``. Nada se
rellena hacia adelante.

El retorno de una serie se calcula siempre en **su** calendario
(``ln(C_i / C_{i-1})``, donde ``i-1`` es la sesion anterior de **esa** serie):
un festivo ajeno nunca puede dar un retorno de cero.

Sin *look-ahead*
----------------

- Las features de una sesion (``asia_overnight_1``, ``europe_prev_1``,
  ``dxy_ret_1``, la dispersion y el recuento) usan la ultima sesion publicada de
  cada mercado, con el rezago de :data:`CONTEXT_MARKET_LAG`. Nunca una sesion
  posterior.
- Las **estadisticas de ventana** (las cuatro correlaciones y la beta) usan pares
  de la **misma** sesion ``s <= t-1`` del S&P, cada serie en su ultima sesion
  ``<= s``. Es la unica forma de que las cuatro correlaciones sean comparables
  entre si: si cada serie entrase en su propia sesion mas reciente, dos
  correlaciones no medirian lo mismo.
- El z-score es de ventana **expandida** (mediana y MAD de la historia ``<= t``),
  nunca de la muestra completa.
- El modulo **no** aplica el corte de muestra limpia de ``analysis.drift``: es
  una restriccion de *estudio*, no del almacen (`_docs/plan.md` §9).

Lo no computable se publica ``null``
------------------------------------

Una correlacion o una beta con menos de su ventana completa, una media a la que
le falta un componente, una serie sin historia, una dispersion con menos de dos
sectores con dato: todo eso es ``null``, nunca ``0`` ni ``NaN``. Un ``close`` no
positivo, o nulo, deja el retorno de esa sesion en ``null`` en lugar de
propagar un logaritmo imposible. La escritura lo verifica otra vez:
:func:`cfdtrader.features.store.daily_records` rechaza un ``NaN`` o un ``inf`` en
cualquier columna del catalogo.

El modulo es **puro** (entra un ``Mapping`` de frames, sale un ``pl.DataFrame``),
**no lee el reloj** y no toca el sistema de ficheros: la persistencia la hace
:func:`cfdtrader.features.store.save_daily` con la spec de :func:`context_spec`.
"""

from __future__ import annotations

import math
import statistics
from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final, cast

import polars as pl

from cfdtrader.features.store import (
    CONTEXT_CORRELATION_WINDOW,
    CONTEXT_FEATURE_COLUMNS,
    CONTEXT_FEATURE_SET,
    CONTEXT_MIN_SESSIONS,
    CONTEXT_SECTOR_SERIES,
    CONTEXT_SERIES,
    DEFAULT_CONTEXT_SOURCES,
    DEFAULT_CONTEXT_WINDOWS,
    ContextInputError,
    FeatureSpec,
    InvalidFeatureMatrixError,
    normalise_expanding,
)

__all__ = [
    "CONTEXT_INPUT_COLUMNS",
    "CONTEXT_MARKET_LAG",
    "context_matrix",
    "context_spec",
]

#: Columnas que se leen de **cada** frame de entrada. ``session`` es el ancla
#: temporal, no una feature; ``close`` es el unico precio que esta familia lee.
CONTEXT_INPUT_COLUMNS: Final[tuple[str, ...]] = ("session", "close")

#: Columna adicional que solo se lee del frame del ancla (``^GSPC``).
AS_OF_COLUMN: Final[str] = "as_of"

#: La serie que fija el **universo** de la matriz y el ``as_of`` de cada fila.
ANCHOR_SERIES: Final[str] = "^GSPC"

#: Rezago declarado de cada mercado **ajeno** al calendario americano, en sesiones
#: del S&P. No es una preferencia: el ``as_of`` del almacen sella el cierre
#: americano en todas las series, asi que lo unico que queda es el orden de los
#: cierres. Tokio y Hong Kong cierran antes de la apertura de Nueva York (rezago
#: 0); Europa, el dolar y el VIX, despues (rezago 1). El S&P no aparece: es el
#: ancla, su rezago es 0 por definicion y no se elige.
CONTEXT_MARKET_LAG: Final[dict[str, int]] = {
    "^N225": 0,
    "^HSI": 0,
    "^GDAXI": 1,
    "^FTSE": 1,
    "^STOXX50E": 1,
    "DX-Y.NYB": 1,
    "^VIX": 1,
}

#: Las cuatro correlaciones moviles y la serie con la que se calculan.
_CORRELATIONS: Final[tuple[tuple[str, str], ...]] = (
    ("corr_dax_60", "^GDAXI"),
    ("corr_ftse_60", "^FTSE"),
    ("corr_stoxx_60", "^STOXX50E"),
    ("corr_nikkei_60", "^N225"),
)

#: Los dos indices asiaticos cuya media es el overnight.
_ASIA: Final[tuple[str, ...]] = ("^N225", "^HSI")

#: Los tres indices europeos cuya media es el cierre previo.
_EUROPE: Final[tuple[str, ...]] = ("^GDAXI", "^FTSE", "^STOXX50E")

#: Nombres de las features sin ventana expandida (los del catalogo de #19).
_ASIA_OVERNIGHT: Final[str] = "asia_overnight_1"
_EUROPE_PREVIOUS: Final[str] = "europe_prev_1"
_BETA_VIX: Final[str] = "beta_vix_60"
_DXY_RETURN: Final[str] = "dxy_ret_1"
_DISPERSION: Final[str] = "sector_dispersion_1"
_SECTOR_COUNT: Final[str] = "sector_count"
_DISPERSION_Z: Final[str] = "sector_dispersion_1_z"

#: Columnas base: las once del catalogo menos la que anade la normalizacion de
#: #19, que se importa y no se calcula aqui.
_BASE_COLUMNS: Final[tuple[str, ...]] = tuple(
    name for name in CONTEXT_FEATURE_COLUMNS if name != _DISPERSION_Z
)


# ─────────────────────────────────────────────────────────────────────────────
# Series de entrada
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class _Series:
    """Una serie de entrada, ordenada por sesion, con sus retornos ya calculados.

    ``returns[i]`` es ``ln(C_i / C_{i-1})`` en el calendario de **esta** serie y
    vale ``None`` cuando no es computable (primera sesion, cierre nulo o no
    positivo). No es un dato que falte: es la forma de no inventar un cero.
    """

    series_id: str
    sessions: list[date]
    returns: list[float | None]


def _sessions(frame: pl.DataFrame, *, series_id: str) -> list[date]:
    """Sesiones de un frame, en orden de fila y sin repetir (A4)."""
    if "session" not in frame.columns:
        raise ContextInputError(
            f"la serie '{series_id}' no trae la columna 'session': de cada serie se leen "
            f"{list(CONTEXT_INPUT_COLUMNS)}"
        )
    column = frame.get_column("session")
    dtype = column.dtype
    if isinstance(dtype, pl.Datetime):
        column = column.dt.date()
    elif not isinstance(dtype, pl.Date):
        raise ContextInputError(
            f"'session' de '{series_id}' tiene que ser pl.Date o pl.Datetime, no {dtype}"
        )

    sessions: list[date] = []
    for value in column.to_list():
        if value is None:
            raise ContextInputError(f"la serie '{series_id}' trae una 'session' nula")
        sessions.append(cast("date", value))
    if len(set(sessions)) != len(sessions):
        repeated = sorted({item for item in sessions if sessions.count(item) > 1})
        raise ContextInputError(
            f"la serie '{series_id}' repite estas sesiones: "
            f"{[item.isoformat() for item in repeated]}"
        )
    return sessions


def _closings(frame: pl.DataFrame, *, series_id: str) -> list[float | None]:
    """``close`` como floats, con ``None`` en los huecos y error en lo no numerico (A4)."""
    if "close" not in frame.columns:
        raise ContextInputError(
            f"la serie '{series_id}' no trae la columna 'close': de cada serie se leen "
            f"{list(CONTEXT_INPUT_COLUMNS)}"
        )
    values: list[float | None] = []
    for value in cast("list[object]", frame.get_column("close").to_list()):
        if value is None:
            values.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContextInputError(
                f"la columna 'close' de '{series_id}' no es numerica: {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ContextInputError(
                f"la columna 'close' de '{series_id}' trae un valor no finito ({number!r}): lo "
                "no computable se publica como null, no como NaN ni inf"
            )
        values.append(number)
    return values


def _instants(frame: pl.DataFrame, *, series_id: str) -> list[datetime]:
    """``as_of`` de los frames del ancla, normalizado a UTC (A4, A11)."""
    if AS_OF_COLUMN not in frame.columns:
        raise ContextInputError(
            f"la serie '{series_id}' no trae la columna '{AS_OF_COLUMN}': el ancla del "
            "universo tiene que traer el cierre de sesion de cada fila"
        )
    column = frame.get_column(AS_OF_COLUMN)
    dtype = column.dtype
    if not isinstance(dtype, pl.Datetime):
        raise ContextInputError(
            f"'{AS_OF_COLUMN}' de '{series_id}' tiene que ser pl.Datetime, no {dtype}"
        )
    if dtype.time_zone is None:
        raise ContextInputError(
            f"'{AS_OF_COLUMN}' de '{series_id}' no lleva zona horaria: un instante sin zona "
            "no puede anclar una fila"
        )
    instants: list[datetime] = []
    for value in column.dt.convert_time_zone("UTC").to_list():
        if value is None:
            raise ContextInputError(f"la serie '{series_id}' trae un '{AS_OF_COLUMN}' nulo")
        instants.append(cast("datetime", value))
    return instants


def _log_returns(closings: list[float | None]) -> list[float | None]:
    """``ln(C_i / C_{i-1})`` en el calendario propio de la serie (decision 1a).

    La primera sesion de la serie no tiene retorno. Un cierre nulo o no positivo
    deja el retorno de esa sesion en ``None``: se prefiere un hueco declarado a un
    logaritmo inventado.
    """
    if not closings:
        return []
    returns: list[float | None] = [None]
    for index in range(1, len(closings)):
        current = closings[index]
        previous = closings[index - 1]
        if current is None or previous is None or current <= 0.0 or previous <= 0.0:
            returns.append(None)
            continue
        returns.append(math.log(current / previous))
    return returns


def _series_of(frame: pl.DataFrame, *, series_id: str) -> _Series:
    """Serie de entrada ordenada por sesion, con sus retornos."""
    sessions = _sessions(frame, series_id=series_id)
    closings = _closings(frame, series_id=series_id)
    pairs = sorted(zip(sessions, closings, strict=True))
    return _Series(
        series_id=series_id,
        sessions=[item[0] for item in pairs],
        returns=_log_returns([item[1] for item in pairs]),
    )


def _anchored_series(frame: pl.DataFrame) -> tuple[_Series, list[datetime]]:
    """El ancla: su serie y el ``as_of`` de cada sesion, ordenados igual (A11)."""
    sessions = _sessions(frame, series_id=ANCHOR_SERIES)
    closings = _closings(frame, series_id=ANCHOR_SERIES)
    instants = _instants(frame, series_id=ANCHOR_SERIES)
    triples = sorted(zip(sessions, closings, instants, strict=True))
    ordered_sessions = [item[0] for item in triples]
    ordered_closings = [item[1] for item in triples]
    ordered_instants = [item[2] for item in triples]
    for session, instant in zip(ordered_sessions, ordered_instants, strict=True):
        if instant.date() != session:
            raise ContextInputError(
                f"la sesion {session.isoformat()} de '{ANCHOR_SERIES}' y su 'as_of' "
                f"({instant.isoformat()}) no corresponden al mismo dia: el 'as_of' de la fila "
                "es el que entra en el hash"
            )
    anchor = _Series(
        series_id=ANCHOR_SERIES,
        sessions=ordered_sessions,
        returns=_log_returns(ordered_closings),
    )
    return anchor, ordered_instants


# ─────────────────────────────────────────────────────────────────────────────
# Alineacion y formulas
# ─────────────────────────────────────────────────────────────────────────────
def _last_index(series: _Series, target: date, *, lag: int) -> int | None:
    """Posicion de la **ultima sesion publicada** de la serie respecto a ``target``.

    ``lag = 0`` ⇒ ultima sesion ``<= target`` (mercado que ya cerro); ``lag = 1``
    ⇒ ultima sesion ``< target`` (mercado que aun no ha cerrado). Es la unica
    forma de leer el calendario ajeno: no se rellena ni se desplaza nada.
    """
    if lag == 0:
        position = bisect_right(series.sessions, target) - 1
    else:
        position = bisect_left(series.sessions, target) - 1
    return position if position >= 0 else None


def _return_at(series: _Series, target: date, *, lag: int) -> float | None:
    """Retorno publicado de la serie en su ultima sesion con el rezago pedido."""
    position = _last_index(series, target, lag=lag)
    if position is None:
        return None
    return series.returns[position]


def _mean_of(values: list[float | None]) -> float | None:
    """Media de los componentes, o ``None`` si falta **cualquiera** de ellos (A5).

    Un mercado sin historia no se omite en silencio: la media de la que forma
    parte no es computable y se publica ``null``.
    """
    if any(value is None for value in values):
        return None
    numbers = [cast("float", value) for value in values]
    return math.fsum(numbers) / len(numbers)


def _centred_sums(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    """Sumas centradas ``(suma(x*y), suma(x^2), suma(y^2))``, o ``None`` sin varianza.

    La varianza cero se detecta con ``max - min == 0`` **exacto** y no con la suma
    de cuadrados: la de una serie constante puede quedar en un residuo de redondeo
    (``~1e-18``) y publicar una correlacion de ``~1e16`` en vez de ``null``.
    """
    if max(xs) - min(xs) == 0.0 or max(ys) - min(ys) == 0.0:
        return None
    count = len(xs)
    mean_x = math.fsum(xs) / count
    mean_y = math.fsum(ys) / count
    deviations_x = [value - mean_x for value in xs]
    deviations_y = [value - mean_y for value in ys]
    return (
        math.fsum(left * right for left, right in zip(deviations_x, deviations_y, strict=True)),
        math.fsum(value * value for value in deviations_x),
        math.fsum(value * value for value in deviations_y),
    )


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    """Pearson de dos listas del mismo tamano, o ``None`` si no es computable (A6).

    Un denominador que desborda a cero (dos series de escala subnormal) tambien es
    ``None``: en coma flotante de doble precision esa correlacion no existe.
    """
    sums = _centred_sums(xs, ys)
    if sums is None:
        return None
    covariance, variance_x, variance_y = sums
    denominator = math.sqrt(variance_x * variance_y)
    if denominator == 0.0:
        return None
    return covariance / denominator


def _slope(xs: list[float], ys: list[float]) -> float | None:
    """Pendiente OLS de ``ys`` sobre ``xs``, o ``None`` sin varianza en ``xs`` (A8)."""
    sums = _centred_sums(xs, ys)
    if sums is None:
        return None
    covariance, variance_x, _ = sums
    if variance_x == 0.0:
        return None
    return covariance / variance_x


def _window_pairs(
    anchor: _Series, index: int, other: _Series
) -> tuple[list[float], list[float]] | None:
    """Los ``CONTEXT_CORRELATION_WINDOW`` pares de la ventana ``s <= t-1`` (decision 1c).

    Cada par es ``(retorno del S&P en s, retorno de la otra serie en su ultima
    sesion <= s)``. Si falta cualquier par de la ventana, la estadistica entera es
    ``None``: no se calcula sobre una ventana incompleta.
    """
    start = index - CONTEXT_CORRELATION_WINDOW
    if start < 0:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for position in range(start, index):
        left = anchor.returns[position]
        if left is None:
            return None
        right = _return_at(other, anchor.sessions[position], lag=0)
        if right is None:
            return None
        xs.append(left)
        ys.append(right)
    return xs, ys


def _correlation_at(anchor: _Series, index: int, other: _Series) -> float | None:
    """Correlacion movil de la sesion ``index`` del ancla con otra serie (A6)."""
    pairs = _window_pairs(anchor, index, other)
    if pairs is None:
        return None
    return _correlation(*pairs)


def _beta_at(anchor: _Series, index: int, other: _Series) -> float | None:
    """Pendiente OLS del retorno del S&P sobre el de la otra serie (A8)."""
    pairs = _window_pairs(anchor, index, other)
    if pairs is None:
        return None
    return _slope(*pairs)


def _sector_returns(series: dict[str, _Series], target: date) -> list[float]:
    """Retornos de los ETF sectoriales **con dato** en su ultima sesion ``< t`` (A10).

    Un ETF que todavia no existia no aporta: no se le inventa un cero ni se anula
    la dispersion por su culpa (``XLRE`` nace en 2015 y ``XLC`` en 2018).
    """
    returns: list[float] = []
    for name in CONTEXT_SECTOR_SERIES:
        value = _return_at(series[name], target, lag=1)
        if value is not None:
            returns.append(value)
    return returns


def _finite_or_null(frame: pl.DataFrame, *, columns: tuple[str, ...]) -> pl.DataFrame:
    """Convierte ``NaN``/``inf`` en ``null`` en las columnas indicadas (A11).

    Es la misma fontaneria que usa la familia tecnica: ``is_finite`` propaga los
    nulos, asi que la conversion no inventa un valor donde no lo habia.
    """
    return frame.with_columns(
        *(
            pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).alias(name)
            for name in columns
        )
    )


def _base_frame(
    anchor: _Series,
    instants: list[datetime],
    columns: dict[str, list[float | None]],
    counts: list[int],
) -> pl.DataFrame:
    """Frame con ``session``, ``as_of`` y las diez columnas sin normalizar."""
    data: dict[str, pl.Series] = {
        "session": pl.Series("session", anchor.sessions, dtype=pl.Date()),
        AS_OF_COLUMN: pl.Series(AS_OF_COLUMN, instants, dtype=pl.Datetime("us", "UTC")),
    }
    for name in _BASE_COLUMNS:
        if name == _SECTOR_COUNT:
            data[name] = pl.Series(name, counts, dtype=pl.Int64())
        else:
            data[name] = pl.Series(name, columns[name], dtype=pl.Float64())
    return pl.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# Spec y matriz
# ─────────────────────────────────────────────────────────────────────────────
def context_spec() -> FeatureSpec:
    """Spec por defecto de ``context_v1``: el catalogo entero y las 19 series.

    Los valores por defecto de :class:`FeatureSpec` son los de ``volatility_v1``
    (familias distintas), asi que la spec de contexto se declara
    **explicitamente**: sus ventanas son las de su catalogo y sus fuentes, las 19
    series que alimentan la alineacion.
    """
    return FeatureSpec(
        feature_set=CONTEXT_FEATURE_SET,
        windows=DEFAULT_CONTEXT_WINDOWS,
        sources=DEFAULT_CONTEXT_SOURCES,
    )


def context_matrix(frames: Mapping[str, pl.DataFrame], *, spec: FeatureSpec) -> pl.DataFrame:
    """Matriz persistible de ``context_v1``: ``session`` + ``as_of`` + 11 features (A3).

    Parameters
    ----------
    frames:
        Un frame por serie de :data:`cfdtrader.features.store.CONTEXT_SERIES`. De
        cada uno se leen ``session`` y ``close``; del de ``^GSPC``, tambien
        ``as_of``. Un frame **sin filas** es una serie sin historia: sus features
        salen ``null`` (y no cuentan en ``sector_count``).
    spec:
        Spec de la familia de contexto. Tiene que declarar ``context_v1``: cada
        familia tiene su propia funcion de calculo.

    Returns
    -------
    pl.DataFrame
        Una fila por sesion de ``^GSPC``, en orden, con ``session``, ``as_of`` y
        las once columnas del catalogo, en su orden; lo no computable a ``null``.
    """
    if spec.feature_set != CONTEXT_FEATURE_SET:
        raise InvalidFeatureMatrixError(
            f"'context_matrix' es la entrada de '{CONTEXT_FEATURE_SET}': la familia "
            f"'{spec.feature_set}' tiene su propia funcion de calculo"
        )
    missing = sorted(set(CONTEXT_SERIES) - set(frames))
    unknown = sorted(set(frames) - set(CONTEXT_SERIES))
    if missing or unknown:
        raise ContextInputError(
            f"el 'Mapping' de series no es el de '{CONTEXT_FEATURE_SET}': faltan {missing} y "
            f"sobran {unknown}"
        )

    anchor, instants = _anchored_series(frames[ANCHOR_SERIES])
    series = {
        name: _series_of(frames[name], series_id=name)
        for name in CONTEXT_SERIES
        if name != ANCHOR_SERIES
    }
    series[ANCHOR_SERIES] = anchor

    columns: dict[str, list[float | None]] = {name: [] for name in _BASE_COLUMNS}
    counts: list[int] = []
    for index, session in enumerate(anchor.sessions):
        for name, partner in _CORRELATIONS:
            columns[name].append(_correlation_at(anchor, index, series[partner]))
        columns[_ASIA_OVERNIGHT].append(
            _mean_of([_return_at(series[name], session, lag=0) for name in _ASIA])
        )
        columns[_EUROPE_PREVIOUS].append(
            _mean_of([_return_at(series[name], session, lag=1) for name in _EUROPE])
        )
        columns[_BETA_VIX].append(_beta_at(anchor, index, series["^VIX"]))
        columns[_DXY_RETURN].append(_return_at(series["DX-Y.NYB"], session, lag=1))

        sector_returns = _sector_returns(series, session)
        counts.append(len(sector_returns))
        columns[_DISPERSION].append(
            statistics.stdev(sector_returns) if len(sector_returns) >= 2 else None
        )

    frame = _base_frame(anchor, instants, columns, counts)
    # La `_z` es de #19: ventana expandida con el minimo declarado, importada.
    frame = normalise_expanding(frame, _DISPERSION, min_sessions=CONTEXT_MIN_SESSIONS)
    frame = _finite_or_null(frame, columns=CONTEXT_FEATURE_COLUMNS)
    return frame.select(["session", AS_OF_COLUMN, *CONTEXT_FEATURE_COLUMNS]).sort("session")
