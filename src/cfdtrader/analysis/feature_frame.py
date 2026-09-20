"""Adaptador del almacen al frame de features etiquetado, con el corrimiento de diseno (#24).

Es el **unico** modulo que lee el almacen para entrenar: arma las cinco familias llamando a
sus funciones de calculo y aplica el corrimiento temporal de una sesion que vive en
``models/baseline``. Lo reutilizan #25, #27, #28 y #73 (la persistencia real de
``derived.features_daily``).

Reglas que se respetan aqui:

- Se lee **siempre** con ``store.sql()``, **nunca** con ``read_pit``: la pregunta es «que datos
  hay», no «que sabiamos en T» (con ``fetched_at`` de hoy y un ``at`` historico, ``read_pit``
  devuelve vacio).
- ``session`` se deriva de ``as_of`` en ``America/New_York`` y se trunca a fecha, igual que en
  ``analysis.backtest_report`` y ``analysis.drift``: la fecha **UTC** no sirve (una sesion a
  caballo del cambio de hora se duplicaria).
- Las cinco familias se llaman con su ``spec`` explicito (es *keyword-only* **sin** valor por
  defecto) y con **exactamente** las claves que declara ``features.store``.
- ``atr_norm`` esta en dos familias (`volatility_v1` y `technical_v1`, #72): la duplicada se
  conserva aparte, se comprueba que las dos son iguales y solo entonces se descarta una. Si
  diverge, es un error, no una eleccion silenciosa.
- ``is_es_roll_session`` entra como 0/1 ``float`` (polars devuelve ``Boolean``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final, cast

import duckdb
import polars as pl

from cfdtrader.data.store import Store, UnknownDatasetError
from cfdtrader.features import store as feature_store
from cfdtrader.features.context import context_matrix, context_spec
from cfdtrader.features.macro import (
    ANCHOR_INPUT_COLUMNS,
    DXY_INPUT_COLUMNS,
    DXY_SERIES,
    MACRO_INPUT_COLUMNS,
    macro_matrix,
    macro_spec,
)
from cfdtrader.features.regime import regime_matrix, regime_spec
from cfdtrader.features.technical import technical_matrix, technical_spec
from cfdtrader.features.volatility import add_features
from cfdtrader.models.baseline import BASELINE_FEATURES, DesignFrame, design_frame

__all__ = [
    "FAMILY_ORDER",
    "FEATURES_LIMITATIONS",
    "MACRO_QUERY",
    "MARKET_QUERY",
    "SECTORS_QUERY",
    "FeatureFrame",
    "FeatureFrameError",
    "FeatureMatrix",
    "MissingFeatureDatasetError",
    "build_feature_frame",
    "build_feature_matrix",
    "load_labels",
]

#: Consultas del almacen. Todas leen el estado **vigente** del dataset: nada de ``read_pit``.
MARKET_QUERY: Final[str] = (
    "SELECT series_id, as_of, open, high, low, close FROM raw.market_daily "
    "ORDER BY series_id, as_of"
)

SECTORS_QUERY: Final[str] = (
    "SELECT series_id, as_of, open, high, low, close FROM raw.sectors ORDER BY series_id, as_of"
)

MACRO_QUERY: Final[str] = (
    "SELECT series_id, as_of, published_at, value FROM raw.macro ORDER BY series_id, as_of"
)

#: Las etiquetas de #10: una fila por sesion, con las dos direcciones (el `series_id` entra
#: como literal validado por el llamante, como en `analysis.backtest_report`).
LABELS_COLUMNS: Final[tuple[str, ...]] = ("session", "ret_long", "is_half_day", "k_sigma")

#: Orden de ensamblado de las familias. Es el orden en que se publican las columnas y no
#: depende del contenido: cambiarlo cambia el ``matrix_sha256``.
FAMILY_ORDER: Final[tuple[str, ...]] = (
    feature_store.VOLATILITY_FEATURE_SET,
    feature_store.TECHNICAL_FEATURE_SET,
    feature_store.CONTEXT_FEATURE_SET,
    feature_store.MACRO_FEATURE_SET,
    feature_store.REGIME_FEATURE_SET,
)

#: Columnas del catalogo por familia, en su orden.
COLUMNS_BY_FAMILY: Final[dict[str, tuple[str, ...]]] = {
    feature_store.VOLATILITY_FEATURE_SET: feature_store.FEATURE_COLUMNS,
    feature_store.TECHNICAL_FEATURE_SET: feature_store.TECHNICAL_FEATURE_COLUMNS,
    feature_store.CONTEXT_FEATURE_SET: feature_store.CONTEXT_FEATURE_COLUMNS,
    feature_store.MACRO_FEATURE_SET: feature_store.MACRO_FEATURE_COLUMNS,
    feature_store.REGIME_FEATURE_SET: feature_store.REGIME_FEATURE_COLUMNS,
}

#: Limites declarados del adaptador, en prosa (viajan al informe).
FEATURES_LIMITATIONS: Final[tuple[str, ...]] = (
    "las features **no** estan persistidas: `data/derived/features_daily` no existe todavia "
    "(lo escribe #73, y hoy nadie escribe ese dataset fuera de `tmp_path`), asi que el "
    "adaptador lee el almacen y vuelve a llamar a las cinco familias en cada corrida",
    "`is_es_roll_session` viaja como 0/1 `float` aunque polars lo produzca `Boolean`: el motor "
    "de features publica ese 0/1, no un booleano",
    "este modulo **no** filtra por la muestra limpia de #52: eso es una restriccion de "
    "*estudio*, no del almacen, y la aplica el adaptador de #69 (el universo del arnes)",
    "`atr_norm` aparece en `volatility_v1` y en `technical_v1` (#72); se comprueba que las dos "
    "son identicas y se publica **una** columna",
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados
# ─────────────────────────────────────────────────────────────────────────────
class FeatureFrameError(Exception):
    """Raiz de los errores del adaptador de features."""


class MissingFeatureDatasetError(FeatureFrameError):
    """Falta un dataset del almacen (``raw.market_daily``, ``raw.macro``, ``derived.labels``)."""


class DuplicateFeatureColumnError(FeatureFrameError):
    """Dos familias publican la misma columna con valores **distintos** (#72)."""


# ─────────────────────────────────────────────────────────────────────────────
# Lectura del almacen
# ─────────────────────────────────────────────────────────────────────────────
def _query(store: Store, query: str, *, dataset: str) -> pl.DataFrame:
    """Ejecuta la consulta y traduce la ausencia del dataset a error tipado (como #69)."""
    try:
        return store.sql(query)
    except (UnknownDatasetError, duckdb.Error) as error:
        raise MissingFeatureDatasetError(
            f"no se puede leer {dataset} en {store.root}: {error}. Ejecuta antes la ingesta "
            "que produce ese dataset"
        ) from error


def _optional_query(
    store: Store, query: str, *, dataset: str, schema: Mapping[str, pl.DataType]
) -> pl.DataFrame:
    """Como :func:`_query`, pero un dataset ausente es «esa familia no tiene historia».

    ``raw.sectors`` y ``raw.macro`` son opcionales: sin ellos las features de esa familia salen
    ``null`` y se **declaran** (la convencion de #21: un frame sin filas es una serie sin
    historia). Lo que no es opcional es el diario del ancla ni las etiquetas.
    """
    try:
        return store.sql(query)
    except (UnknownDatasetError, duckdb.Error):
        return _empty(schema)


def _with_session(frame: pl.DataFrame) -> pl.DataFrame:
    """Anade ``session``: ``as_of`` en ``America/New_York`` truncado a fecha."""
    if frame.height == 0:
        return frame.with_columns(pl.lit(None, dtype=pl.Date()).alias("session"))
    return frame.with_columns(
        pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("session")
    )


def _by_series(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Trocea un frame multi-serie en un frame por ``series_id``, ordenado por ``as_of``."""
    out: dict[str, pl.DataFrame] = {}
    if frame.height == 0:
        return out
    for series_id in frame.get_column("series_id").unique().sort().to_list():
        out[str(series_id)] = frame.filter(pl.col("series_id") == series_id).sort("as_of")
    return out


def _empty(schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    """Frame vacio con el esquema que espera la familia: «esa serie no tiene historia»."""
    return pl.DataFrame(schema=dict(schema))


#: Esquemas de los frames vacios que sustituyen a un dataset o a una serie ausente.
MARKET_SCHEMA: Final[dict[str, pl.DataType]] = {
    "series_id": pl.String(),
    "as_of": pl.Datetime("us", "UTC"),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
}

MACRO_SCHEMA: Final[dict[str, pl.DataType]] = {
    "series_id": pl.String(),
    "as_of": pl.Date(),
    "published_at": pl.Datetime("us", "UTC"),
    "value": pl.Float64(),
}


def _require_datasets(store: Store, *, names: Sequence[str]) -> None:
    """Los tres datasets que el adaptador necesita, con su error tipado si falta alguno."""
    available = set(store.datasets("raw")).union(store.datasets("derived"))
    missing = sorted({name for name in names if name not in available})
    if missing:
        raise MissingFeatureDatasetError(
            f"el almacen {store.root} no tiene {missing}: el adaptador necesita "
            f"{list(names)} para armar las cinco familias. Construye el almacen antes (los "
            "tests que no lo necesitan no leen este modulo)"
        )


def load_labels(store: Store, *, series_id: str = "^GSPC") -> pl.DataFrame:
    """Las etiquetas de esa serie: ``session``, ``ret_long`` y su procedencia (A3)."""
    if "'" in series_id or ";" in series_id:
        raise MissingFeatureDatasetError(
            f"el `series_id` {series_id!r} no es un identificador admisible: se compone un "
            "literal SQL y no se interpola texto libre"
        )
    query = (
        "SELECT session, ret_long, is_half_day, k_sigma FROM derived.labels "  # noqa: S608
        f"WHERE series_id = '{series_id}' ORDER BY session"
    )
    frame = _query(store, query, dataset="derived.labels")
    if frame.height == 0:
        raise MissingFeatureDatasetError(
            f"`derived.labels` no tiene ninguna sesion de {series_id!r}: sin etiquetas no hay "
            "objetivo que modelar"
        )
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Las cinco familias
# ─────────────────────────────────────────────────────────────────────────────
def _volatility_frame(anchor: pl.DataFrame, vix: pl.DataFrame) -> pl.DataFrame:
    """``volatility_v1``: el OHLC del ancla mas ``vix_close`` (el cierre de ``^VIX``)."""
    vix_close = vix.select(
        pl.col("session"),
        pl.col("close").alias("vix_close"),
    )
    inputs = anchor.select("session", "open", "high", "low", "close").join(
        vix_close, on="session", how="left"
    )
    return add_features(inputs)


def _context_frames(
    market: Mapping[str, pl.DataFrame], sectors: Mapping[str, pl.DataFrame]
) -> tuple[dict[str, pl.DataFrame], tuple[str, ...]]:
    """Los 19 frames de ``context_v1``: 8 de ``raw.market_daily`` y 11 de ``raw.sectors``."""
    frames: dict[str, pl.DataFrame] = {}
    missing: list[str] = []
    for series_id in feature_store.CONTEXT_SERIES:
        source = sectors if series_id in feature_store.CONTEXT_SECTOR_SERIES else market
        raw = source.get(series_id)
        if raw is None:
            missing.append(series_id)
            frames[series_id] = _empty(
                {"session": pl.Date(), "as_of": pl.Datetime("us", "UTC"), "close": pl.Float64()}
            )
            continue
        frame = _with_session(raw)
        frames[series_id] = (
            frame.select("session", "as_of", "close")
            if series_id == feature_store.CONTEXT_MARKET_SERIES[0]
            else frame.select("session", "close")
        )
    return frames, tuple(missing)


def _macro_frames(
    market: Mapping[str, pl.DataFrame], macro: Mapping[str, pl.DataFrame]
) -> tuple[dict[str, pl.DataFrame], tuple[str, ...]]:
    """Los 8 frames de ``macro_v1``: seis series macro (``as_of`` = referencia) y dos barras."""
    frames: dict[str, pl.DataFrame] = {}
    missing: list[str] = []
    schema = {"as_of": pl.Date(), "published_at": pl.Datetime("us", "UTC"), "value": pl.Float64()}
    for series_id in feature_store.MACRO_SERIES:
        raw = macro.get(series_id)
        if raw is None:
            missing.append(series_id)
            frames[series_id] = _empty(schema)
            continue
        frames[series_id] = raw.select(*MACRO_INPUT_COLUMNS)
    anchor = market.get(feature_store.CONTEXT_MARKET_SERIES[0])
    frames[feature_store.CONTEXT_MARKET_SERIES[0]] = (
        _empty({"session": pl.Date(), "as_of": pl.Datetime("us", "UTC")})
        if anchor is None
        else _with_session(anchor).select(*ANCHOR_INPUT_COLUMNS)
    )
    dxy = market.get(DXY_SERIES)
    frames[DXY_SERIES] = (
        _empty({"session": pl.Date(), "as_of": pl.Datetime("us", "UTC"), "close": pl.Float64()})
        if dxy is None
        else _with_session(dxy).select(*DXY_INPUT_COLUMNS)
    )
    return frames, tuple(missing)


# ─────────────────────────────────────────────────────────────────────────────
# Ensamblado de la matriz (A1, A4)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """La matriz de las cinco familias: una fila por sesion del diario, 52 columnas unicas.

    ``duplicated_columns`` son las columnas que publican **dos** familias y que se comprobaron
    identicas antes de quedarse con una sola (#72). ``missing_series`` son las series que el
    almacen no tenia: sus features salen ``null``, declaradas, nunca inventadas.
    """

    frame: pl.DataFrame
    n_sessions: int
    n_columns: int
    duplicated_columns: tuple[str, ...]
    missing_series: tuple[str, ...]
    feature_spec_sha256: dict[str, str]
    feature_code_version: int
    matrix_sha256: str
    first_session: date
    last_session: date

    @property
    def columns(self) -> tuple[str, ...]:
        """Las columnas de features, sin ``session``, en el orden publicado."""
        return tuple(name for name in self.frame.columns if name != "session")


def _family_columns(family: str) -> tuple[str, ...]:
    """Las columnas declaradas de esa familia."""
    try:
        return COLUMNS_BY_FAMILY[family]
    except KeyError as error:
        raise FeatureFrameError(
            f"la familia {family!r} no tiene columnas declaradas en COLUMNS_BY_FAMILY"
        ) from error


def _join(
    matrix: pl.DataFrame, frame: pl.DataFrame, *, family: str
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """Une una familia, apartando las columnas repetidas para poder compararlas."""
    columns = _family_columns(family)
    present = [name for name in columns if name in matrix.columns]
    aliased = [f"{name}__{family}" for name in present]
    prepared = frame.select("session", *columns).rename(dict(zip(present, aliased, strict=True)))
    joined = matrix.join(prepared, on="session", how="left", validate="1:1")
    return joined, tuple(present)


def build_feature_matrix(store: Store, *, series_id: str = "^GSPC") -> FeatureMatrix:
    """Arma la matriz de las cinco familias sobre el **diario** del ancla (A1, A4).

    Una sola pasada por familia, con su ``spec`` explicito. La fila de la sesion `t` es el
    estado de las features **al cierre de `t`**; el corrimiento de disponibilidad lo aplica
    despues :func:`cfdtrader.models.baseline.design_frame`, que es donde vive esa regla.
    """
    _require_datasets(store, names=("market_daily", "labels"))
    market = _by_series(_with_session(_query(store, MARKET_QUERY, dataset="raw.market_daily")))
    sectors = _by_series(
        _with_session(
            _optional_query(store, SECTORS_QUERY, dataset="raw.sectors", schema=MARKET_SCHEMA)
        )
    )
    macro_series = _by_series(
        _optional_query(store, MACRO_QUERY, dataset="raw.macro", schema=MACRO_SCHEMA)
    )

    anchor = market.get(series_id)
    if anchor is None or anchor.height == 0:
        raise MissingFeatureDatasetError(
            f"`raw.market_daily` no tiene ninguna sesion de {series_id!r}: sin el ancla no hay "
            "matriz de features"
        )
    vix = market.get("^VIX")
    if vix is None or vix.height == 0:
        raise MissingFeatureDatasetError(
            "`raw.market_daily` no tiene el cierre de '^VIX': `volatility_v1` exige `vix_close` y "
            "no se sustituye por otro proxy de volatilidad"
        )

    family_frames: dict[str, pl.DataFrame] = {
        feature_store.VOLATILITY_FEATURE_SET: _volatility_frame(anchor, vix),
        feature_store.TECHNICAL_FEATURE_SET: technical_matrix(
            anchor.select("session", "open", "high", "low", "close"), spec=technical_spec()
        ),
        feature_store.REGIME_FEATURE_SET: regime_matrix(
            anchor.select("session", "open", "high", "low", "close"), spec=regime_spec()
        ),
    }
    context_frames, missing_context = _context_frames(market, sectors)
    family_frames[feature_store.CONTEXT_FEATURE_SET] = context_matrix(
        context_frames, spec=context_spec()
    )
    macro_frames, missing_macro = _macro_frames(market, macro_series)
    family_frames[feature_store.MACRO_FEATURE_SET] = macro_matrix(macro_frames, spec=macro_spec())

    matrix = anchor.select("session")
    duplicated: list[str] = []
    for family in FAMILY_ORDER:
        frame = family_frames[family]
        if frame.height != matrix.height:
            raise FeatureFrameError(
                f"la familia {family!r} devuelve {frame.height} filas y el ancla tiene "
                f"{matrix.height}: la matriz se une por `session` y no se pueden perder sesiones "
                "en el camino"
            )
        matrix, present = _join(matrix, frame, family=family)
        duplicated.extend(present)
    duplicated_columns = tuple(dict.fromkeys(duplicated))
    matrix = _collapse_duplicates(matrix, duplicated_columns)
    matrix = _normalise_dtypes(matrix)

    expected = ("session", *feature_store.ALL_FEATURE_COLUMNS)
    unknown = sorted(set(matrix.columns) - set(expected))
    if unknown:
        raise FeatureFrameError(
            f"la matriz publica columnas que no estan en el catalogo: {unknown}. Las features "
            "son las declaradas en `features.store`, ni una mas"
        )
    matrix = matrix.select(*expected)

    sessions = cast("list[date]", matrix.get_column("session").to_list())
    return FeatureMatrix(
        frame=matrix,
        n_sessions=matrix.height,
        n_columns=len(feature_store.ALL_FEATURE_COLUMNS),
        duplicated_columns=duplicated_columns,
        missing_series=(*missing_context, *missing_macro),
        feature_spec_sha256={
            family: feature_store.feature_spec_sha256(_spec_for(family)) for family in FAMILY_ORDER
        },
        feature_code_version=feature_store.FEATURE_CODE_VERSION,
        matrix_sha256=feature_store.matrix_sha256(matrix),
        first_session=sessions[0],
        last_session=sessions[-1],
    )


def _collapse_duplicates(matrix: pl.DataFrame, names: Sequence[str]) -> pl.DataFrame:
    """Comprueba que las columnas repetidas son identicas y descarta la copia (#72)."""
    for name in names:
        copies = [column for column in matrix.columns if column.startswith(f"{name}__")]
        for copy in copies:
            left = matrix.get_column(name)
            right = matrix.get_column(copy)
            equal = left.equals(right)
            if not equal:
                difference = _first_difference(left, right)
                raise DuplicateFeatureColumnError(
                    f"la columna '{name}' y su copia '{copy}' no coinciden (primera diferencia en "
                    f"la fila {difference}): la familia que la duplica (#72) tiene que producir el "
                    "mismo numero; quedarse con una sin comprobarlo seria elegir en silencio"
                )
        matrix = matrix.drop(copies)
    return matrix


def _first_difference(left: pl.Series, right: pl.Series) -> int:
    """Posicion de la primera diferencia entre dos series, o ``-1`` si no hay ninguna."""
    for index, (one, other) in enumerate(zip(left.to_list(), right.to_list(), strict=True)):
        if one is None and other is None:
            continue
        if one != other or (one is None) != (other is None):
            return index
    return -1


def _normalise_dtypes(matrix: pl.DataFrame) -> pl.DataFrame:
    """Las cinco familias como ``Float64``, con lo no finito a ``null`` (A4).

    ``is_es_roll_session`` entra como 0/1 ``float`` aunque polars lo produzca ``Boolean``: el
    motor de features publica ese 0/1. ``nan``/``inf`` no son valores publicables: se declaran
    como ``null``, nunca como un numero inventado.
    """
    expressions: list[pl.Expr] = []
    for name in matrix.columns:
        if name == "session":
            continue
        column = matrix.get_column(name)
        expression = (
            pl.col(name).cast(pl.Int8) if isinstance(column.dtype, pl.Boolean) else pl.col(name)
        )
        numbers = expression.cast(pl.Float64)
        expressions.append(pl.when(numbers.is_finite()).then(numbers).otherwise(None).alias(name))
    return matrix.with_columns(expressions)


def _spec_for(family: str) -> feature_store.FeatureSpec:
    """La spec de esa familia, sin pasar por el almacen (es una constante del modulo)."""
    if family == feature_store.VOLATILITY_FEATURE_SET:
        return feature_store.FeatureSpec()
    if family == feature_store.TECHNICAL_FEATURE_SET:
        return technical_spec()
    if family == feature_store.CONTEXT_FEATURE_SET:
        return context_spec()
    if family == feature_store.MACRO_FEATURE_SET:
        return macro_spec()
    if family == feature_store.REGIME_FEATURE_SET:
        return regime_spec()
    raise FeatureFrameError(
        f"la familia {family!r} no tiene spec: el adaptador solo conoce las cinco declaradas"
    )


# ─────────────────────────────────────────────────────────────────────────────
# El frame completo: matriz + etiquetas + diseno (A2, A3)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class FeatureFrame:
    """Todo lo que necesita el modelo: la matriz, las etiquetas y **la matriz de diseno**.

    ``design`` es el frame de diseno de :func:`cfdtrader.models.baseline.design_frame`: una fila
    por sesion etiquetada, con las 10 features de `t-1` y la etiqueta de `t`.
    """

    series_id: str
    matrix: FeatureMatrix
    labels: pl.DataFrame
    design: DesignFrame
    design_lag_sessions: int
    feature_columns: tuple[str, ...]

    @property
    def n_design_rows(self) -> int:
        """Sesiones con fila de diseno (el universo del modelo)."""
        return self.design.n_sessions

    @property
    def n_positives(self) -> int:
        """Sesiones con ``y == 1`` (A3)."""
        return self.design.positives

    @property
    def first_session(self) -> date:
        """Primera sesion del diseno."""
        return self.design.sessions[0]

    @property
    def last_session(self) -> date:
        """Ultima sesion del diseno."""
        return self.design.sessions[-1]

    @property
    def n_nulls_in_features(self) -> int:
        """Nulos en las 10 columnas de diseno (A3): tienen que ser 0 y se publican."""
        return self.design.n_nulls_in_features

    @property
    def n_half_days(self) -> int:
        """Sesiones etiquetadas como media sesion (dato de #10, solo se publica)."""
        if "is_half_day" not in self.labels.columns:
            return 0
        return int(self.labels.get_column("is_half_day").sum())


def build_feature_frame(store: Store, *, series_id: str = "^GSPC") -> FeatureFrame:
    """Arma la matriz, las etiquetas y el frame de diseno con el corrimiento de una sesion.

    El corrimiento **no** pierde ninguna sesion etiquetada: la anterior a 2016-01-07
    (2016-01-06) esta en el diario y su fila de features existe. ``design.n_shifted_rows`` lo
    publica medido, sin afirmarlo.
    """
    matrix = build_feature_matrix(store, series_id=series_id)
    labels = load_labels(store, series_id=series_id)
    design = design_frame(matrix.frame, labels=labels)
    if design.n_sessions == 0:
        raise FeatureFrameError(
            "la matriz de diseno sale vacia: ninguna sesion etiquetada tiene fila de features de "
            "la sesion anterior. Revisa que el diario y las etiquetas compartan sesiones"
        )
    return FeatureFrame(
        series_id=series_id,
        matrix=matrix,
        labels=labels,
        design=design,
        design_lag_sessions=design.design_lag_sessions,
        feature_columns=BASELINE_FEATURES,
    )
