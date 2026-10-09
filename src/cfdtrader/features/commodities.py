"""Features de commodities y FX (`commodities_v1`, tarea #143).

**Familia cerrada.** Este modulo entrega el calculo completo de ``commodities_v1``:
las cinco columnas declaradas en
:data:`cfdtrader.features.store.COMMODITIES_FEATURE_CATALOG` — el retorno del WTI a
una y a cinco sesiones, su z-score de ventana expandida, el retorno del oro y el
retorno del EUR/USD. No entrega ninguna otra: una columna fuera del catalogo no es
de esta familia.

**Que series y por que.** La decision esta escrita, con su numero medido, en
``_docs/commodity_fx_features_2026-10-09.md``, y aqui solo se materializa:
``BZ=F`` (Brent) queda **fuera** por la regla pre-declarada del documento
(redundante con el WTI, ``r = +0.86`` a una sesion sobre la ventana comun) y
``EURUSD=X`` **entra** porque la premisa contraria no se sostiene en el dato: su
retorno de una sesion correlaciona ``-0.32`` con el DXY, no ``-0.95`` (ese ``-0.95``
es la correlacion de **niveles**, ``-0.97``, y la de horizontes de un mes o mas).
``DX-Y.NYB`` no entra: su retorno ya es ``dxy_ret_1`` desde #21.

**No reimplementa.** La normalizacion robusta de ventana expandida se **importa**
de :mod:`cfdtrader.features.store` (#19); aqui no hay una segunda formula de
mediana ni de MAD.

Interfaz
--------

:func:`commodities_matrix` recibe un ``Mapping`` con **una entrada por serie**
(:data:`cfdtrader.features.store.COMMODITIES_SERIES`: 4 claves) y no un frame
largo, por el mismo motivo que la familia de contexto (#21): cada mercado trae su
**propio calendario** y un frame largo obligaria a inventar la alineacion antes de
calcular. De cada frame se leen **solo** ``session`` y ``close``; del de ``^GSPC``,
ademas, el ``as_of`` de cada fila, que se **copia** (el modulo no lee el reloj).
Falta una serie, sobra una clave, falta ``session`` o ``close`` o se repite una
sesion: :class:`cfdtrader.features.store.CommoditiesInputError`, con la serie en el
mensaje.

Alineacion entre mercados
-------------------------

El ``as_of`` del almacen es el **sello del cierre de la sesion americana** en
**todas** las series, asi que no dice cuando cerro un mercado ajeno; lo unico que
queda es el **orden de los cierres**. Las tres series admitidas cierran **despues**
del cierre del S&P —el futuro del WTI y el del oro a las 17:00 ET
(``_docs/plan.md`` §8.5) y el EUR/USD con el cierre de Nueva York—, de modo que su
ultima sesion **cerrada** cuando la sesion ``t`` del indice acaba es la ``< t``.
Ese es el rezago **declarado** de la familia, el mismo que usa ``dxy_ret_1`` en
:mod:`cfdtrader.features.context`: **1 sesion**, sin un mapa por serie porque las
tres lo comparten.

El retorno de una serie se calcula siempre en **su** calendario
(``ln(C_i / C_{i-1})``): un festivo ajeno **no** produce un cero, produce el retorno
de su ultima sesion con cierre, y si no hay historia la feature es ``null``. Nada se
rellena hacia adelante.

Sin *look-ahead*
----------------

- Toda columna de la fila ``t`` usa la **ultima sesion ``< t``** de su serie, con la
  excepcion declarada de ``oil_ret_1_z``, que es una normalizacion de ventana
  **expandida** ``<= t`` (mediana y MAD de la historia, nunca de la muestra
  completa) y por tanto es la medida de «cuanto de raro es el dato de hoy frente a
  su historia», no un dato de ``t``.
- El corrimiento de disponibilidad de una sesion adicional que aplica el modelo vive
  en :func:`cfdtrader.models.baseline.design_frame`, no aqui: esta familia publica el
  estado **al cierre de ``t``**, igual que las otras cinco.
- El modulo **no** aplica el corte de muestra limpia de ``analysis.drift``: es una
  restriccion de *estudio*, no del almacen (`_docs/plan.md` §9).

Lo no computable se publica ``null``
------------------------------------

Una serie sin historia suficiente, un ``close`` nulo o no positivo, una ventana de
cinco sesiones que no cabe en la historia de la serie: todo eso es ``null``, nunca
``0`` ni ``NaN``. La escritura lo verifica otra vez:
:func:`cfdtrader.features.store.daily_records` rechaza un ``NaN`` o un ``inf`` en
cualquier columna del catalogo.

El modulo es **puro** (entra un ``Mapping`` de frames, sale un ``pl.DataFrame``),
**no lee el reloj** y no toca el sistema de ficheros: la persistencia la hace
:func:`cfdtrader.features.store.save_daily` con la spec de
:func:`commodities_spec`.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Final, cast

import polars as pl

from cfdtrader.features.store import (
    COMMODITIES_ANCHOR_SERIES,
    COMMODITIES_FEATURE_COLUMNS,
    COMMODITIES_FEATURE_SET,
    COMMODITIES_MIN_SESSIONS,
    COMMODITIES_RETURN_WINDOW,
    COMMODITIES_SERIES,
    DEFAULT_COMMODITIES_SOURCES,
    DEFAULT_COMMODITIES_WINDOWS,
    CommoditiesInputError,
    FeatureSpec,
    InvalidFeatureMatrixError,
    normalise_expanding,
)

__all__ = [
    "COMMODITIES_INPUT_COLUMNS",
    "COMMODITIES_LAG_SESSIONS",
    "commodities_matrix",
    "commodities_spec",
]

#: Columnas que se leen de **cada** frame de entrada. ``session`` es el ancla
#: temporal, no una feature; ``close`` es el unico precio que esta familia lee.
COMMODITIES_INPUT_COLUMNS: Final[tuple[str, ...]] = ("session", "close")

#: Columna adicional que solo se lee del frame del ancla (``^GSPC``).
AS_OF_COLUMN: Final[str] = "as_of"

#: Rezago declarado de las **tres** series admitidas, en sesiones del S&P. No es una
#: preferencia: el ``as_of`` del almacen sella el cierre americano en todas las
#: series, y las tres (futuro del WTI, futuro del oro y EUR/USD) cierran **despues**
#: de la sesion del indice, asi que lo unico que queda es el orden de los cierres.
#: Es el mismo rezago que usa ``dxy_ret_1`` en la familia de contexto.
COMMODITIES_LAG_SESSIONS: Final[int] = 1

#: Nombres de las features (las claves de ``CatalogEntry.name``): los del codigo y
#: los del catalogo coinciden.
_OIL_RETURN: Final[str] = "oil_ret_1"
_OIL_RETURN_LONG: Final[str] = "oil_ret_5"
_OIL_RETURN_Z: Final[str] = f"{_OIL_RETURN}_z"
_GOLD_RETURN: Final[str] = "gold_ret_1"
_EURUSD_RETURN: Final[str] = "eurusd_ret_1"

#: Serie de cada columna con ventana, y la columna de la que se calcula el z.
_SERIES_BY_COLUMN: Final[dict[str, str]] = {
    _OIL_RETURN: "CL=F",
    _OIL_RETURN_LONG: "CL=F",
    _GOLD_RETURN: "GC=F",
    _EURUSD_RETURN: "EURUSD=X",
}

#: Columnas base: las cinco del catalogo menos la que anade la normalizacion de #19,
#: que se importa y no se calcula aqui.
_BASE_COLUMNS: Final[tuple[str, ...]] = tuple(
    name for name in COMMODITIES_FEATURE_COLUMNS if name != _OIL_RETURN_Z
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
    ``closings`` se conserva porque ``oil_ret_5`` es un retorno **acumulado** y se
    calcula sobre los cierres, no encadenando cinco retornos que pueden faltar.
    """

    series_id: str
    sessions: list[date]
    closings: list[float | None]
    returns: list[float | None]


def _sessions(frame: pl.DataFrame, *, series_id: str) -> list[date]:
    """Sesiones de un frame, en orden de fila y sin repetir."""
    if "session" not in frame.columns:
        raise CommoditiesInputError(
            f"la serie '{series_id}' no trae la columna 'session': de cada serie se leen "
            f"{list(COMMODITIES_INPUT_COLUMNS)}"
        )
    column = frame.get_column("session")
    dtype = column.dtype
    if isinstance(dtype, pl.Datetime):
        column = column.dt.date()
    elif not isinstance(dtype, pl.Date):
        raise CommoditiesInputError(
            f"'session' de '{series_id}' tiene que ser pl.Date o pl.Datetime, no {dtype}"
        )

    sessions: list[date] = []
    for value in column.to_list():
        if value is None:
            raise CommoditiesInputError(f"la serie '{series_id}' trae una 'session' nula")
        sessions.append(cast("date", value))
    if len(set(sessions)) != len(sessions):
        repeated = sorted({item for item in sessions if sessions.count(item) > 1})
        raise CommoditiesInputError(
            f"la serie '{series_id}' repite estas sesiones: "
            f"{[item.isoformat() for item in repeated]}"
        )
    return sessions


def _closings(frame: pl.DataFrame, *, series_id: str) -> list[float | None]:
    """``close`` como floats, con ``None`` en los huecos y error en lo no numerico."""
    if "close" not in frame.columns:
        raise CommoditiesInputError(
            f"la serie '{series_id}' no trae la columna 'close': de cada serie se leen "
            f"{list(COMMODITIES_INPUT_COLUMNS)}"
        )
    values: list[float | None] = []
    for value in cast("list[object]", frame.get_column("close").to_list()):
        if value is None:
            values.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CommoditiesInputError(
                f"la columna 'close' de '{series_id}' no es numerica: {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise CommoditiesInputError(
                f"la columna 'close' de '{series_id}' trae un valor no finito ({number!r}): lo "
                "no computable se publica como null, no como NaN ni inf"
            )
        values.append(number)
    return values


def _instants(frame: pl.DataFrame, *, series_id: str) -> list[datetime]:
    """``as_of`` de los frames del ancla, normalizado a UTC."""
    if AS_OF_COLUMN not in frame.columns:
        raise CommoditiesInputError(
            f"la serie '{series_id}' no trae la columna '{AS_OF_COLUMN}': el ancla del "
            "universo tiene que traer el cierre de sesion de cada fila"
        )
    column = frame.get_column(AS_OF_COLUMN)
    dtype = column.dtype
    if not isinstance(dtype, pl.Datetime):
        raise CommoditiesInputError(
            f"'{AS_OF_COLUMN}' de '{series_id}' tiene que ser pl.Datetime, no {dtype}"
        )
    if getattr(dtype, "time_zone", None) is not None:
        column = column.dt.convert_time_zone("UTC")

    instants: list[datetime] = []
    for value in column.to_list():
        if value is None:
            raise CommoditiesInputError(f"la serie '{series_id}' trae un '{AS_OF_COLUMN}' nulo")
        instant = cast("datetime", value)
        instants.append(instant.replace(tzinfo=UTC) if instant.tzinfo is None else instant)
    return instants


def _log_returns(closings: list[float | None]) -> list[float | None]:
    """``ln(C_i / C_{i-1})`` en el calendario propio de la serie.

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
    ordered_closings = [item[1] for item in pairs]
    return _Series(
        series_id=series_id,
        sessions=[item[0] for item in pairs],
        closings=ordered_closings,
        returns=_log_returns(ordered_closings),
    )


def _anchored_series(frame: pl.DataFrame) -> tuple[_Series, list[datetime]]:
    """El ancla: su serie y el ``as_of`` de cada sesion, ordenados igual."""
    series_id = COMMODITIES_ANCHOR_SERIES
    sessions = _sessions(frame, series_id=series_id)
    closings = _closings(frame, series_id=series_id)
    instants = _instants(frame, series_id=series_id)
    triples = sorted(zip(sessions, closings, instants, strict=True))
    ordered_sessions = [item[0] for item in triples]
    ordered_closings = [item[1] for item in triples]
    ordered_instants = [item[2] for item in triples]
    for session, instant in zip(ordered_sessions, ordered_instants, strict=True):
        if instant.date() != session:
            raise CommoditiesInputError(
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

    ``COMMODITIES_LAG_SESSIONS`` es 1, asi que la fila ``t`` lee la ultima sesion
    ``< target`` (la serie aun no ha cerrado cuando la sesion ``t`` del indice
    acaba). Es la unica forma de leer el calendario ajeno: no se rellena ni se
    desplaza nada.
    """
    position = bisect_left(series.sessions, target) - COMMODITIES_LAG_SESSIONS
    return position if position >= 0 else None


def _return_at(series: _Series, target: date) -> float | None:
    """Retorno publicado de la serie en su ultima sesion con el rezago declarado."""
    position = _last_index(series, target)
    if position is None:
        return None
    return series.returns[position]


def _cumulative_return_at(series: _Series, target: date) -> float | None:
    """Retorno acumulado de ``COMMODITIES_RETURN_WINDOW`` sesiones, o ``None``.

    Es ``ln(C_i / C_{i-w})`` sobre los **cierres**, con ``i`` la ultima sesion
    publicada y ``w`` la ventana declarada. Si la serie no tiene ``w`` sesiones
    antes de ``i``, o cualquiera de los dos cierres no es positivo, el resultado es
    ``None``: una ventana que no cabe no se rellena.
    """
    position = _last_index(series, target)
    if position is None or position - COMMODITIES_RETURN_WINDOW < 0:
        return None
    current = series.closings[position]
    previous = series.closings[position - COMMODITIES_RETURN_WINDOW]
    if current is None or previous is None or current <= 0.0 or previous <= 0.0:
        return None
    return math.log(current / previous)


def _finite_or_null(frame: pl.DataFrame, *, columns: tuple[str, ...]) -> pl.DataFrame:
    """Convierte ``NaN``/``inf`` en ``null`` en las columnas indicadas.

    ``is_finite`` propaga los nulos, asi que la conversion no inventa un valor
    donde no lo habia.
    """
    return frame.with_columns(
        *(
            pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).alias(name)
            for name in columns
        )
    )


def _base_frame(
    anchor: _Series, instants: list[datetime], columns: dict[str, list[float | None]]
) -> pl.DataFrame:
    """Frame con ``session``, ``as_of`` y las cuatro columnas sin normalizar."""
    data: dict[str, pl.Series] = {
        "session": pl.Series("session", anchor.sessions, dtype=pl.Date()),
        AS_OF_COLUMN: pl.Series(AS_OF_COLUMN, instants, dtype=pl.Datetime("us", "UTC")),
    }
    for name in _BASE_COLUMNS:
        data[name] = pl.Series(name, columns[name], dtype=pl.Float64())
    return pl.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# Spec y matriz
# ─────────────────────────────────────────────────────────────────────────────
def commodities_spec() -> FeatureSpec:
    """Spec por defecto de ``commodities_v1``: el catalogo entero y las cuatro series.

    Los valores por defecto de :class:`FeatureSpec` son los de ``volatility_v1``
    (familias distintas), asi que la spec de commodities se declara
    **explicitamente**: sus ventanas son las de su catalogo y sus fuentes, el ancla
    y las tres series admitidas.
    """
    return FeatureSpec(
        feature_set=COMMODITIES_FEATURE_SET,
        windows=DEFAULT_COMMODITIES_WINDOWS,
        sources=DEFAULT_COMMODITIES_SOURCES,
    )


def commodities_matrix(frames: Mapping[str, pl.DataFrame], *, spec: FeatureSpec) -> pl.DataFrame:
    """Matriz persistible de ``commodities_v1``: ``session`` + ``as_of`` + las cinco features.

    Parameters
    ----------
    frames:
        ``Mapping`` con una entrada por serie de ``COMMODITIES_SERIES``.
    spec:
        Spec de la familia. Tiene que declarar ``commodities_v1``: las otras
        familias tienen su propia funcion de calculo.

    Returns
    -------
    pl.DataFrame
        Una fila por sesion de ``^GSPC``, en orden, con ``session``, ``as_of`` y
        las cinco columnas del catalogo, en su orden; lo no computable a ``null``.
    """
    if spec.feature_set != COMMODITIES_FEATURE_SET:
        raise InvalidFeatureMatrixError(
            f"'commodities_matrix' es la entrada de '{COMMODITIES_FEATURE_SET}': la familia "
            f"'{spec.feature_set}' tiene su propia funcion de calculo"
        )
    missing = sorted(set(COMMODITIES_SERIES) - set(frames))
    unknown = sorted(set(frames) - set(COMMODITIES_SERIES))
    if missing or unknown:
        raise CommoditiesInputError(
            f"el 'Mapping' de series no es el de '{COMMODITIES_FEATURE_SET}': faltan {missing} y "
            f"sobran {unknown}"
        )

    anchor, instants = _anchored_series(frames[COMMODITIES_ANCHOR_SERIES])
    series = {
        name: _series_of(frames[name], series_id=name)
        for name in COMMODITIES_SERIES
        if name != COMMODITIES_ANCHOR_SERIES
    }

    columns: dict[str, list[float | None]] = {name: [] for name in _BASE_COLUMNS}
    for session in anchor.sessions:
        for name, series_id in _SERIES_BY_COLUMN.items():
            if name == _OIL_RETURN_LONG:
                columns[name].append(_cumulative_return_at(series[series_id], session))
            else:
                columns[name].append(_return_at(series[series_id], session))

    frame = _base_frame(anchor, instants, columns)
    # La `_z` es de #19: ventana expandida con el minimo declarado, importada.
    frame = normalise_expanding(frame, _OIL_RETURN, min_sessions=COMMODITIES_MIN_SESSIONS)
    frame = _finite_or_null(frame, columns=COMMODITIES_FEATURE_COLUMNS)
    return frame.select(["session", AS_OF_COLUMN, *COMMODITIES_FEATURE_COLUMNS]).sort("session")
