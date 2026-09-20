"""Features macro alineadas *point-in-time* (`macro_v1`, tarea #22).

**Familia cerrada.** Este modulo entrega el calculo completo de ``macro_v1``: las
trece columnas declaradas en
:data:`cfdtrader.features.store.MACRO_FEATURE_CATALOG` (nivel y cambio de los
cuatro drivers de tipos, la inflacion interanual de CPI y PCE, el nivel del
indice dolar y dos z-scores de ventana expandida). ``PAYEMS`` esta almacenada y
**fuera** de esta familia: su sitio es ``macro_v2`` (#76).

**No reimplementa.** La normalizacion robusta de ventana expandida se **importa**
de :mod:`cfdtrader.features.store` (#19): aqui no hay una segunda formula de
mediana ni de MAD.

Interfaz
--------

:func:`macro_matrix` recibe un ``Mapping`` con **una entrada por serie** (las
seis de :data:`cfdtrader.features.store.MACRO_SERIES` mas las dos de
:data:`cfdtrader.features.store.MACRO_MARKET_SERIES`) y no un frame largo. El
motivo es que una publicacion macro **no** cae el dia de su referencia: lo que
hay que alinear es el ``published_at`` de cada observacion contra el cierre de
cada sesion, y eso exige tener cada serie con su propia rejilla. Falta una serie,
sobra una clave, falta ``published_at`` o ``value``, el ``as_of`` de una serie
macro no es una fecha o el de una barra no es un instante con zona:
:class:`cfdtrader.features.store.MacroInputError`, con la serie en el mensaje.

El ``as_of`` **no** significa lo mismo en las dos clases de serie, y confundirlas
es un error tipado:

- de una serie de ``raw.macro``, el ``as_of`` es la **fecha de referencia** del
  dato (``pl.Date``): el mes al que se refiere el CPI, el dia que observo la DFF;
- de una barra de ``raw.market_daily``, el ``as_of`` es el **instante del
  cierre** (``pl.Datetime`` con zona UTC). De la barra del ancla se lee ademas
  ``session``, que fija el universo de la matriz; de la del DXY, su ``close``.

La regla de alineacion (R)
--------------------------

En la fila de la sesion ``t`` solo entra lo que ya estaba publicado al cierre de
``t``: ``published_at <= as_of(t)``, comparando **instantes UTC**. Nunca fechas
sueltas (una publicacion de las 20:30 UTC pertenece al dia siguiente en Madrid, y
la referencia de Madrid no es la del proyecto) y nunca la fecha de referencia.
De las candidatas gana la de **mayor** ``published_at``; un empate (una
publicacion doble, como el PCEPI del 2026-01-22, que publica de golpe octubre y
noviembre) lo rompe la **mayor fecha de referencia**. Si el empate alcanza a la
misma referencia, el dato es ambiguo y no se elige: ``MacroInputError``.

Un nulo no se interpola y un valor viejo **no** se censura: si la fuente se para
(como en el *shutdown* de 2025), la ultima observacion publicada se sigue
transportando sesion a sesion durante 38 o 49 sesiones si hace falta. Lo que no
es computable se publica ``null``, nunca ``0``, ``NaN`` ni ``inf``, y **nunca** la
primera observacion futura.

Las columnas
------------

- ``fed_funds``, ``ust_10y``, ``ust_2y`` y ``pendiente_2s10s`` son **niveles**
  transportados; ``pendiente_2s10s`` se toma de la serie ``T10Y2Y`` publicada,
  **sin** derivarla como ``ust_10y - ust_2y``.
- Sus cuatro ``_chg_5`` son ``transportado(t) - transportado(t-5)``, con ``t-5``
  la sesion **cinco posiciones** antes en el ancla (no cinco publicaciones ni
  cinco dias naturales), y en **puntos porcentuales**, no en ``%`` relativo.
- ``cpi_yoy`` y ``pce_yoy`` son ``100 * (v_m / v_{m-12} - 1)``, con ``m`` la
  referencia de la ultima publicacion que cumple R y ``v_{m-12}`` la de su misma
  fecha un ano antes, que tambien tiene que estar publicada. Sin
  desestacionalizar.
- ``dxy`` es el **nivel** del indice dolar: el ``close`` de la barra con mayor
  ``as_of <= as_of(t)`` (la igualdad cuenta), o ``null`` solo sin candidata. Su
  variacion ya vive en ``context_v1`` como ``dxy_ret_1`` (#21), y esta familia
  **no** la republica: la divergencia (aqui nivel y sin rezago, alli retorno con
  rezago de una sesion) es deliberada.
- ``ust_10y_z`` y ``dxy_z`` son los dos niveles cuyo escalon absoluto cambia de
  regimen, normalizados con la ventana expandida de #19.

Sin *look-ahead*
----------------

La fila ``t`` es el estado **al cierre de** ``t``, como en las otras tres
familias: el ``published_at`` de todo lo que entra es anterior o igual a
``as_of(t)`` y el DXY solo puede aportar una barra ya cerrada. El uso de la fila
para decidir en la apertura de ``t+1`` (desplazarla una sesion) es de **#73**,
no de este modulo. El modulo **no** aplica el corte de muestra limpia de
``analysis.drift`` ni reserva *holdout* (#68): son restricciones de *estudio*.

El modulo es **puro** (entra un ``Mapping`` de frames, sale un ``pl.DataFrame``),
**no lee el reloj** y no toca el sistema de ficheros: la persistencia la hace
:func:`cfdtrader.features.store.save_daily` con la spec de :func:`macro_spec`.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final, cast

import polars as pl

from cfdtrader.features.store import (
    DEFAULT_MACRO_SOURCES,
    DEFAULT_MACRO_WINDOWS,
    MACRO_CHG_WINDOW,
    MACRO_FEATURE_COLUMNS,
    MACRO_FEATURE_SET,
    MACRO_MARKET_SERIES,
    MACRO_MIN_SESSIONS,
    MACRO_SERIES,
    NORMALISED_SUFFIX,
    FeatureSpec,
    InvalidFeatureMatrixError,
    MacroInputError,
    normalise_expanding,
)

__all__ = [
    "ANCHOR_SERIES",
    "DXY_SERIES",
    "MACRO_INPUT_COLUMNS",
    "macro_matrix",
    "macro_spec",
]

#: La serie de mercado que fija el **universo** de la matriz y el ``as_of`` de
#: cada fila.
ANCHOR_SERIES: Final[str] = "^GSPC"

#: La barra de mercado cuyo **nivel** publica esta familia.
DXY_SERIES: Final[str] = "DX-Y.NYB"

#: Nombre de la columna de sesion de las barras de mercado.
SESSION_COLUMN: Final[str] = "session"

#: Nombre de la columna temporal en las dos clases de serie. En una serie macro es
#: la **fecha de referencia** (``pl.Date``); en una barra, el **instante** del
#: cierre (``pl.Datetime`` con zona). Es a proposito que compartan nombre: es el
#: mismo concepto ("el sello del dato") leido con dos tipos distintos, y confundir
#: los tipos es :class:`cfdtrader.features.store.MacroInputError`.
AS_OF_COLUMN: Final[str] = "as_of"

#: Instante en el que la fuente publico la observacion (solo series macro).
PUBLISHED_COLUMN: Final[str] = "published_at"

#: Valor observado (solo series macro).
VALUE_COLUMN: Final[str] = "value"

#: Cierre de la barra (solo la del DXY).
CLOSE_COLUMN: Final[str] = "close"

#: Columnas que se leen de **cada** serie macro.
MACRO_INPUT_COLUMNS: Final[tuple[str, ...]] = (AS_OF_COLUMN, PUBLISHED_COLUMN, VALUE_COLUMN)

#: Columnas que se leen de la barra del ancla.
ANCHOR_INPUT_COLUMNS: Final[tuple[str, ...]] = (SESSION_COLUMN, AS_OF_COLUMN)

#: Columnas que se leen de la barra del DXY.
DXY_INPUT_COLUMNS: Final[tuple[str, ...]] = (SESSION_COLUMN, AS_OF_COLUMN, CLOSE_COLUMN)

#: Las cuatro parejas ``(nivel, serie de la que sale, cambio)``. El orden es el del
#: catalogo y el cambio se calcula **posicionalmente** sobre el nivel ya
#: transportado, nunca sobre la serie original.
_LEVELS: Final[tuple[tuple[str, str, str], ...]] = (
    ("fed_funds", "DFF", "fed_funds_chg_5"),
    ("ust_10y", "DGS10", "ust_10y_chg_5"),
    ("ust_2y", "DGS2", "ust_2y_chg_5"),
    ("pendiente_2s10s", "T10Y2Y", "pendiente_2s10s_chg_5"),
)

#: Las dos inflaciones interanuales y la serie de la que sale cada una.
_YOY: Final[tuple[tuple[str, str], ...]] = (("cpi_yoy", "CPIAUCSL"), ("pce_yoy", "PCEPI"))

#: Nombre de la columna del nivel del indice dolar.
_DXY_COLUMN: Final[str] = "dxy"

#: Las columnas que **no** se calculan directamente sino normalizando otro nivel,
#: con la ventana expandida que se importa de #19.
_NORMALISED: Final[tuple[str, ...]] = ("ust_10y", _DXY_COLUMN)

#: Los nombres de las columnas normalizadas: los del catalogo, derivados del sufijo
#: de #19 para que no puedan divergir de lo que produce ``normalise_expanding``.
_NORMALISED_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"{column}{NORMALISED_SUFFIX}" for column in _NORMALISED
)

#: Las columnas que se calculan directamente: el catalogo menos las dos normalizadas.
_BASE_COLUMNS: Final[tuple[str, ...]] = tuple(
    name for name in MACRO_FEATURE_COLUMNS if name not in _NORMALISED_COLUMNS
)


# ─────────────────────────────────────────────────────────────────────────────
# Lectura de las series de entrada
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class _Observation:
    """Una observacion macro con valor y con instante de publicacion.

    Las filas con ``published_at`` o ``value`` nulos **no** llegan aqui: no son
    candidatas a nada (decision 2 de #22) y no se les inventa un valor.
    """

    reference: date
    published_at: datetime
    value: float


@dataclass(frozen=True, slots=True)
class _MacroSeries:
    """Una serie macro lista para resolver la regla R.

    ``observations`` esta ordenada por ``(published_at, reference)``, que es
    exactamente el orden con el que se rompe el empate de una publicacion doble:
    la ultima candidata de la ventana es la de mayor instante y, entre las que
    comparten instante, la de mayor referencia.
    """

    series_id: str
    observations: list[_Observation]
    published: list[datetime]
    by_reference: dict[date, list[_Observation]]


def _require_columns(frame: pl.DataFrame, *, series_id: str, columns: tuple[str, ...]) -> None:
    """Comprueba que el frame trae las columnas que esa serie tiene que traer (A1)."""
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise MacroInputError(
            f"la serie '{series_id}' no trae las columnas {missing}: de ella se leen "
            f"{list(columns)}"
        )


def _sessions(frame: pl.DataFrame, *, series_id: str) -> list[date]:
    """Sesiones de una barra de mercado, sin repetir y sin nulos (A1)."""
    _require_columns(frame, series_id=series_id, columns=(SESSION_COLUMN,))
    column = frame.get_column(SESSION_COLUMN)
    dtype = column.dtype
    if isinstance(dtype, pl.Datetime):
        column = column.dt.date()
    elif not isinstance(dtype, pl.Date):
        raise MacroInputError(
            f"'{SESSION_COLUMN}' de '{series_id}' tiene que ser pl.Date o pl.Datetime, no {dtype}"
        )
    sessions: list[date] = []
    for value in column.to_list():
        if value is None:
            raise MacroInputError(f"la serie '{series_id}' trae una '{SESSION_COLUMN}' nula")
        sessions.append(cast("date", value))
    if len(set(sessions)) != len(sessions):
        repeated = sorted({item for item in sessions if sessions.count(item) > 1})
        raise MacroInputError(
            f"la serie '{series_id}' repite estas sesiones: "
            f"{[item.isoformat() for item in repeated]}"
        )
    return sessions


def _references(frame: pl.DataFrame, *, series_id: str) -> list[date]:
    """Fechas de **referencia** de una serie macro: ``pl.Date`` y sin nulos (A1).

    Un ``pl.Datetime`` aqui es el tipo de una barra de mercado puesta en el sitio
    de una serie macro: el error se lanza en vez de convertir, porque convertir
    seria elegir por quien llama.
    """
    _require_columns(frame, series_id=series_id, columns=(AS_OF_COLUMN,))
    column = frame.get_column(AS_OF_COLUMN)
    dtype = column.dtype
    if not isinstance(dtype, pl.Date):
        raise MacroInputError(
            f"'{AS_OF_COLUMN}' de '{series_id}' tiene que ser pl.Date (la fecha de referencia "
            f"de una serie macro), no {dtype}: el 'as_of' de una barra de mercado si es un "
            "instante"
        )
    references: list[date] = []
    for value in column.to_list():
        if value is None:
            raise MacroInputError(
                f"la serie '{series_id}' trae una '{AS_OF_COLUMN}' nula: sin fecha de "
                "referencia una observacion no se puede transportar"
            )
        references.append(cast("date", value))
    return references


def _instants(frame: pl.DataFrame, *, series_id: str, column: str) -> list[datetime | None]:
    """Instantes UTC de una columna; un ``Datetime`` sin zona es un error (A6).

    Devuelve ``None`` donde la celda es nula: en una serie macro eso significa
    "no es candidata" y lo decide quien llama, no este helper.
    """
    _require_columns(frame, series_id=series_id, columns=(column,))
    series = frame.get_column(column)
    dtype = series.dtype
    if not isinstance(dtype, pl.Datetime):
        raise MacroInputError(f"'{column}' de '{series_id}' tiene que ser pl.Datetime, no {dtype}")
    if dtype.time_zone is None:
        raise MacroInputError(
            f"'{column}' de '{series_id}' no lleva zona horaria: un instante sin zona no puede "
            "compararse con el cierre de una sesion"
        )
    instants: list[datetime | None] = []
    for value in series.dt.convert_time_zone("UTC").to_list():
        if value is None:
            instants.append(None)
            continue
        instants.append(cast("datetime", value))
    return instants


def _numbers(frame: pl.DataFrame, *, series_id: str, column: str) -> list[float | None]:
    """Columna numerica como floats, con ``None`` en los nulos y error en lo no finito.

    Un valor de texto, ``NaN`` o ``inf`` es un error tipado (A1): lo no computable
    se publica como ``null`` cuando falta, no cuando esta mal.
    """
    _require_columns(frame, series_id=series_id, columns=(column,))
    values: list[float | None] = []
    for value in cast("list[object]", frame.get_column(column).to_list()):
        if value is None:
            values.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MacroInputError(
                f"la columna '{column}' de '{series_id}' no es numerica: {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise MacroInputError(
                f"la columna '{column}' de '{series_id}' trae un valor no finito ({number!r}): "
                "lo no computable se publica como null, no como NaN ni inf"
            )
        values.append(number)
    return values


def _macro_series(frame: pl.DataFrame, *, series_id: str) -> _MacroSeries:
    """Una serie macro lista para la regla R (A1, A6).

    Las filas sin ``published_at`` o sin ``value`` se descartan en silencio: la
    decision 2 las excluye de las candidatas, no las convierte en error. Las que
    comparten instante **y** referencia si son un error: el empate es persistente
    y elegir una seria inventar el desempate.
    """
    references = _references(frame, series_id=series_id)
    instants = _instants(frame, series_id=series_id, column=PUBLISHED_COLUMN)
    values = _numbers(frame, series_id=series_id, column=VALUE_COLUMN)

    observations: list[_Observation] = []
    seen: set[tuple[datetime, date]] = set()
    for reference, published, value in zip(references, instants, values, strict=True):
        if published is None or value is None:
            continue
        key = (published, reference)
        if key in seen:
            raise MacroInputError(
                f"la serie '{series_id}' trae dos observaciones con el mismo instante de "
                f"publicacion ({published.isoformat()}) y la misma referencia "
                f"({reference.isoformat()}): el empate no se puede deshacer"
            )
        seen.add(key)
        observations.append(_Observation(reference, published, value))

    observations.sort(key=lambda observation: (observation.published_at, observation.reference))
    by_reference: dict[date, list[_Observation]] = {}
    for observation in observations:
        by_reference.setdefault(observation.reference, []).append(observation)
    return _MacroSeries(
        series_id=series_id,
        observations=observations,
        published=[observation.published_at for observation in observations],
        by_reference=by_reference,
    )


def _anchor(frame: pl.DataFrame) -> tuple[list[date], list[datetime]]:
    """Sesiones e instantes de cierre del ancla, ordenados y coherentes (A1).

    El ``as_of`` de una fila tiene que caer el **mismo dia** que su ``session``:
    es el sello que entra en el hash de ``features_version``, y desplazarlo seria
    colar una fila con el cierre de otro dia.
    """
    sessions = _sessions(frame, series_id=ANCHOR_SERIES)
    instants = _instants(frame, series_id=ANCHOR_SERIES, column=AS_OF_COLUMN)
    pairs: list[tuple[date, datetime]] = []
    for session, instant in zip(sessions, instants, strict=True):
        if instant is None:
            raise MacroInputError(f"la serie '{ANCHOR_SERIES}' trae un '{AS_OF_COLUMN}' nulo")
        if instant.date() != session:
            raise MacroInputError(
                f"la sesion {session.isoformat()} de '{ANCHOR_SERIES}' y su 'as_of' "
                f"({instant.isoformat()}) no corresponden al mismo dia: el 'as_of' de la fila "
                "es el que ancla la alineacion"
            )
        pairs.append((session, instant))
    pairs.sort(key=lambda pair: pair[0])
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _dxy_bars(frame: pl.DataFrame) -> tuple[list[datetime], list[float]]:
    """Barras del DXY ordenadas por instante, con su cierre (A12).

    Una barra sin ``close`` no es una barra: no se le inventa un nivel y no entra
    en la lista de candidatas. Una barra cuyo ``as_of`` no cae el dia de su
    ``session`` si es un error: significaria que el cierre es de otro dia.
    """
    sessions = _sessions(frame, series_id=DXY_SERIES)
    instants = _instants(frame, series_id=DXY_SERIES, column=AS_OF_COLUMN)
    closes = _numbers(frame, series_id=DXY_SERIES, column=CLOSE_COLUMN)
    bars: list[tuple[datetime, float]] = []
    for session, instant, close in zip(sessions, instants, closes, strict=True):
        if instant is None:
            raise MacroInputError(f"la serie '{DXY_SERIES}' trae un '{AS_OF_COLUMN}' nulo")
        if instant.date() != session:
            raise MacroInputError(
                f"la sesion {session.isoformat()} de '{DXY_SERIES}' y su 'as_of' "
                f"({instant.isoformat()}) no corresponden al mismo dia: el 'as_of' es el "
                "sello de su cierre"
            )
        if close is None:
            continue
        bars.append((instant, close))
    bars.sort(key=lambda bar: bar[0])
    return [bar[0] for bar in bars], [bar[1] for bar in bars]


# ─────────────────────────────────────────────────────────────────────────────
# La regla R y sus formulas
# ─────────────────────────────────────────────────────────────────────────────
def _transported(series: _MacroSeries, instant: datetime) -> _Observation | None:
    """La observacion vigente en ``instant``: la ultima con ``published_at <=`` (A6).

    Sin candidatas devuelve ``None``, que es un ``null`` en la matriz: no se
    interpola, no se censura y **nunca** se toma una publicacion futura.
    """
    index = bisect_right(series.published, instant) - 1
    if index < 0:
        return None
    return series.observations[index]


def _level(series: _MacroSeries, instant: datetime) -> float | None:
    """Nivel transportado de una serie, o ``None`` si aun no hay nada publicado."""
    observation = _transported(series, instant)
    return None if observation is None else observation.value


def _changes(values: list[float | None], *, window: int) -> list[float | None]:
    """``values[i] - values[i-window]``, en puntos porcentuales y posicional (A10).

    Es un cambio **entre sesiones del ancla**, no entre observaciones: si la
    fuente se para, las dos puntas pueden ser la misma publicacion y el cambio
    sale ``0,0``, que es la informacion correcta (en esas cinco sesiones no se
    supo nada nuevo). Las ``window`` primeras sesiones no tienen la punta
    anterior y quedan a ``null``.
    """
    changes: list[float | None] = []
    for index, value in enumerate(values):
        previous = values[index - window] if index >= window else None
        if value is None or previous is None:
            changes.append(None)
            continue
        changes.append(value - previous)
    return changes


def _same_day_previous_year(reference: date) -> date | None:
    """La misma fecha un ano antes, o ``None`` si esa fecha no existe (29 de febrero)."""
    try:
        return date(reference.year - 1, reference.month, reference.day)
    except ValueError:
        return None


def _reference_value(series: _MacroSeries, reference: date, instant: datetime) -> float | None:
    """Valor de referencia ``reference`` **si ya estaba publicado** en ``instant`` (A11).

    Es la mitad poco obvia de ``cpi_yoy``: la referencia de hace un ano tiene que
    cumplir R por si misma, no basta con que exista.
    """
    candidates = [
        observation
        for observation in series.by_reference.get(reference, [])
        if observation.published_at <= instant
    ]
    if not candidates:
        return None
    return candidates[-1].value


def _yoy(series: _MacroSeries, instant: datetime) -> float | None:
    """``100 * (v_m / v_{m-12} - 1)`` con ``m`` la referencia vigente (A11).

    Un denominador cero no es un infinito: la tasa no es computable y se publica
    ``null``.
    """
    observation = _transported(series, instant)
    if observation is None:
        return None
    reference = _same_day_previous_year(observation.reference)
    if reference is None:
        return None
    base = _reference_value(series, reference, instant)
    if base is None or base == 0.0:
        return None
    return 100.0 * (observation.value / base - 1.0)


def _dxy_level(instants: list[datetime], closes: list[float], instant: datetime) -> float | None:
    """Nivel del DXY: el cierre de la ultima barra cerrada a ``instant`` (A12).

    La **igualdad** cuenta: la barra de la misma sesion cierra al mismo instante
    que el ancla, asi que es la que se publica. Sin candidata, ``null``.
    """
    index = bisect_right(instants, instant) - 1
    if index < 0:
        return None
    return closes[index]


# ─────────────────────────────────────────────────────────────────────────────
# Spec y matriz
# ─────────────────────────────────────────────────────────────────────────────
def macro_spec() -> FeatureSpec:
    """Spec por defecto de ``macro_v1``: el catalogo entero y las ocho series.

    Los valores por defecto de :class:`FeatureSpec` son los de ``volatility_v1``
    (familias distintas), asi que la spec macro se declara **explicitamente**: sus
    ventanas son las de su catalogo y sus fuentes, las seis series macro mas las
    dos barras de mercado.
    """
    return FeatureSpec(
        feature_set=MACRO_FEATURE_SET,
        windows=DEFAULT_MACRO_WINDOWS,
        sources=DEFAULT_MACRO_SOURCES,
    )


def _base_frame(
    sessions: list[date],
    instants: list[datetime],
    columns: dict[str, list[float | None]],
) -> pl.DataFrame:
    """Frame con ``session``, ``as_of`` y las once columnas sin normalizar."""
    data: dict[str, pl.Series] = {
        SESSION_COLUMN: pl.Series(SESSION_COLUMN, sessions, dtype=pl.Date()),
        AS_OF_COLUMN: pl.Series(AS_OF_COLUMN, instants, dtype=pl.Datetime("us", "UTC")),
    }
    for name in _BASE_COLUMNS:
        data[name] = pl.Series(name, columns[name], dtype=pl.Float64())
    return pl.DataFrame(data)


def macro_matrix(frames: Mapping[str, pl.DataFrame], *, spec: FeatureSpec) -> pl.DataFrame:
    """Matriz persistible de ``macro_v1``: ``session`` + ``as_of`` + 13 features (A1).

    Parameters
    ----------
    frames:
        Un frame por serie de ``MACRO_SERIES + MACRO_MARKET_SERIES``. De cada serie
        macro se leen ``as_of`` (referencia), ``published_at`` y ``value``; de la
        barra del ancla, ``session`` y ``as_of``; de la del DXY, tambien ``close``.
        Un frame **sin filas** es una serie sin historia: sus features salen
        ``null``.
    spec:
        Spec de la familia macro. Tiene que declarar ``macro_v1``: cada familia
        tiene su propia funcion de calculo.

    Returns
    -------
    pl.DataFrame
        Una fila por sesion de ``^GSPC``, en orden, con ``session``, ``as_of`` y
        las trece columnas del catalogo, en su orden; lo no computable a ``null``.
    """
    if spec.feature_set != MACRO_FEATURE_SET:
        raise InvalidFeatureMatrixError(
            f"'macro_matrix' es la entrada de '{MACRO_FEATURE_SET}': la familia "
            f"'{spec.feature_set}' tiene su propia funcion de calculo"
        )
    expected = (*MACRO_SERIES, *MACRO_MARKET_SERIES)
    missing = sorted(set(expected) - set(frames))
    unknown = sorted(set(frames) - set(expected))
    if missing or unknown:
        raise MacroInputError(
            f"el 'Mapping' de series no es el de '{MACRO_FEATURE_SET}': faltan {missing} y "
            f"sobran {unknown}"
        )

    sessions, instants = _anchor(frames[ANCHOR_SERIES])
    series = {name: _macro_series(frames[name], series_id=name) for name in MACRO_SERIES}
    bar_instants, bar_closes = _dxy_bars(frames[DXY_SERIES])

    columns: dict[str, list[float | None]] = {}
    for name, series_id, change in _LEVELS:
        values = [_level(series[series_id], instant) for instant in instants]
        columns[name] = values
        columns[change] = _changes(values, window=MACRO_CHG_WINDOW)
    for name, series_id in _YOY:
        columns[name] = [_yoy(series[series_id], instant) for instant in instants]
    columns[_DXY_COLUMN] = [_dxy_level(bar_instants, bar_closes, instant) for instant in instants]

    frame = _base_frame(sessions, instants, columns)
    # Las dos `_z` son de #19: ventana expandida con el minimo declarado, importada.
    for column in _NORMALISED:
        frame = normalise_expanding(frame, column, min_sessions=MACRO_MIN_SESSIONS)
    return frame.select([SESSION_COLUMN, AS_OF_COLUMN, *MACRO_FEATURE_COLUMNS]).sort(SESSION_COLUMN)
