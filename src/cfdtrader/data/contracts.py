"""Firmas y ``layout_version`` de cada dataset del almacén (`tech_stack.md` §12.10).

El contrato de columnas obligatorias vive en `cfdtrader.data.store`. Este módulo
declara lo que el almacén **no** puede saber por sí solo: la **firma del *payload***
—nombre y tipo de cada columna propia del dataset— y su ``layout_version``, por
``(capa, dataset)``.

Por qué se versiona **en el código** y no en el dato: la firma *es* el esquema del
Parquet, así que un fichero antiguo se detecta **leyéndolo**. Escribir la versión en
cada fila habría contaminado las seis columnas obligatorias y roto la inmutabilidad de
`raw` en cada cambio de contrato.

Los tres casos de §12.10, y cómo se declaran aquí:

- **Columna nueva (aditiva):** sube ``layout_version`` y la columna se declara con
  ``since`` = la versión en la que aparece. Un fichero anterior a esa versión **se
  lee**: la columna ausente sale ``null`` **declarado**, nunca ``0``.
- **Cambio de tipo:** el lector **rechaza** el fichero y nombra fichero, columna y los
  dos tipos (:class:`cfdtrader.data.store.LayoutMismatchError`). No hay coerción.
- **Cambio de semántica:** exige **columna nueva con nombre nuevo**. Una columna que
  no está en la firma es un error, no algo que se ignora en silencio.

Un dataset sin declaración **no se valida** (el almacén lee lo que haya, como antes de
#139): la declaración es una lista blanca, y un test comprueba que *todos* los datasets
que el proyecto escribe están en ella, para que no haya huecos por olvido.

**Identidad ampliada (#144):** además del *payload*, un dataset puede declarar
``identity_columns`` —columnas de *payload* que se suman a ``(source, series_id, as_of)``
para identificar el registro—. Solo lo necesita un dataset donde ese trío no sea único
(``news_headlines``). Ampliar la identidad **no** cambia la firma del Parquet, así que no
sube ``layout_version``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

#: Vocabulario declarado de tipos. Es el mismo que el de las seis columnas obligatorias,
#: a propósito: un *payload* no necesita un lenguaje de tipos más rico que el contrato.
DTYPES: Final[frozenset[str]] = frozenset({"str", "int", "float", "bool", "date", "datetime"})

#: Tipos de Polars → tipo declarado. Lo usa el lector para comparar firmas.
POLARS_DTYPES: Final[dict[str, str]] = {
    "String": "str",
    "Int8": "int",
    "Int16": "int",
    "Int32": "int",
    "Int64": "int",
    "UInt8": "int",
    "UInt16": "int",
    "UInt32": "int",
    "UInt64": "int",
    "Float32": "float",
    "Float64": "float",
    "Boolean": "bool",
    "Date": "date",
    "Datetime": "datetime",
}

#: Tipos declarados que el lector acepta para la columna ``as_of``: una serie macro
#: ancla en ``date`` y una barra en ``datetime`` UTC (contrato del almacén).
AS_OF_DTYPES: Final[frozenset[str]] = frozenset({"date", "datetime"})

#: Columnas obligatorias y su tipo declarado. ``as_of`` admite los dos anteriores.
REQUIRED_SIGNATURE: Final[dict[str, frozenset[str]]] = {
    "source": frozenset({"str"}),
    "series_id": frozenset({"str"}),
    "as_of": AS_OF_DTYPES,
    "fetched_at": frozenset({"datetime"}),
    "published_at": frozenset({"datetime"}),
    "version": frozenset({"int"}),
}


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Una columna de *payload*: su nombre, su tipo declarado y desde qué versión existe."""

    name: str
    dtype: str
    since: int = 1
    """``layout_version`` en la que la columna aparece. ``1`` = firma original."""

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(
                f"tipo declarado desconocido {self.dtype!r} para la columna {self.name!r}: "
                f"los admitidos son {sorted(DTYPES)}"
            )
        if self.name in REQUIRED_SIGNATURE:
            raise ValueError(
                f"{self.name!r} es una columna obligatoria del almacén: no puede declararse "
                "como payload de un dataset"
            )
        if self.since < 1:
            raise ValueError(f"'since' tiene que ser >= 1, no {self.since} ({self.name!r})")


@dataclass(frozen=True, slots=True)
class DatasetLayout:
    """Firma y ``layout_version`` vigentes de un dataset (§12.10)."""

    layer: str
    dataset: str
    layout_version: int
    payload: tuple[ColumnSpec, ...]
    identity_columns: tuple[str, ...] = ()
    """Columnas de *payload* que **amplían la identidad** del registro, además de
    ``(source, series_id, as_of)``. Por defecto ``()``: la identidad no cambia.

    Existe para datasets donde ``(source, series_id, as_of)`` **no** es único: en
    ``raw.news_headlines`` dos titulares del mismo feed pueden compartir instante y
    se distinguen por su ``headline_hash`` (#144). Ampliar la identidad **no** cambia
    el esquema del Parquet —``headline_hash`` ya era una columna de *payload*—, así
    que **no** exige subir ``layout_version`` ni migrar ficheros.
    """

    def __post_init__(self) -> None:
        declared = self.signature
        seen: set[str] = set()
        for name in self.identity_columns:
            if name in REQUIRED_SIGNATURE:
                raise ValueError(
                    f"{name!r} es una columna obligatoria del almacén: no puede ampliar "
                    f"la identidad de {self.layer}.{self.dataset}"
                )
            if name not in declared:
                raise ValueError(
                    f"{name!r} amplía la identidad de {self.layer}.{self.dataset} pero no está "
                    f"declarada en su payload: {list(declared)}"
                )
            if name in seen:
                raise ValueError(
                    f"{name!r} está repetida en identity_columns de {self.layer}.{self.dataset}"
                )
            seen.add(name)

    @property
    def signature(self) -> tuple[str, ...]:
        """Nombres de las columnas de *payload* de la firma vigente."""
        return tuple(spec.name for spec in self.payload)

    def version_of(self, columns: frozenset[str]) -> int:
        """``layout_version`` del fichero que trae esas columnas de *payload*.

        La firma **es** el esquema, así que la versión se **infiere**: un fichero con
        todas las columnas vigentes es de la versión vigente; al que solo le faltan
        columnas **aditivas** es de la versión anterior a la más antigua de las que le
        faltan (``min(since) - 1``). Al que le falta alguna de la firma **original**
        (``since == 1``) se le asigna ``0``: es anterior al primer *layout* declarado
        —o lo escribió una fuente que no trae ese *payload*—, así que **se lee**
        declarando esas columnas como ``null`` en vez de inventarlas.
        """
        missing = [spec for spec in self.payload if spec.name not in columns]
        if not missing:
            return self.layout_version
        additive = [spec for spec in missing if spec.since > 1]
        if len(additive) != len(missing):
            return 0
        return min(spec.since for spec in additive) - 1


def layout_of(layer: str, dataset: str) -> DatasetLayout | None:
    """Declaración del dataset, o ``None`` si no está declarado (y por tanto no se valida)."""
    return LAYOUTS.get((layer, dataset))


# ─────────────────────────────────────────────────────────────────────────────
# Declaraciones
# ─────────────────────────────────────────────────────────────────────────────
#: Columnas de *payload* de precios diarios: las comparten `market_daily` y `sectors`
#: (`market.PAYLOAD_COLUMNS` es su dueño; aquí se declara su **tipo**).
_PRICE_COLUMNS: Final[tuple[ColumnSpec, ...]] = (
    ColumnSpec("open", "float"),
    ColumnSpec("high", "float"),
    ColumnSpec("low", "float"),
    ColumnSpec("close", "float"),
    ColumnSpec("volume", "float"),
    ColumnSpec("adj_close", "float"),
)

#: Columna ``session`` (``date``) que llevan los datasets derivados por sesión.
_SESSION: Final[ColumnSpec] = ColumnSpec("session", "date")

#: Bloque de versión de la matriz de features (`features.store.VERSION_COLUMNS`).
_FEATURE_VERSION_COLUMNS: Final[tuple[ColumnSpec, ...]] = (
    ColumnSpec("features_version", "str"),
    ColumnSpec("feature_spec_sha256", "str"),
)

#: Columnas de la matriz de features (`features.store.ALL_FEATURE_COLUMNS`): el **catálogo**
#: lo posee el productor (#73), y aquí se declara su **firma**: los 52 nombres y su tipo.
#: Añadir una feature es cambiar la firma de `derived.features_daily` y por tanto subir su
#: `layout_version` (§12.10); el test que compara esta lista con el catálogo del productor
#: es lo que impide que eso pase inadvertido.
FEATURE_SPEC_COLUMNS: Final[tuple[ColumnSpec, ...]] = (
    ColumnSpec("true_range", "float"),
    ColumnSpec("atr_norm", "float"),
    ColumnSpec("parkinson_rv", "float"),
    ColumnSpec("ret_log", "float"),
    ColumnSpec("ret_sq", "float"),
    ColumnSpec("har_lag1", "float"),
    ColumnSpec("har_lag4", "float"),
    ColumnSpec("har_lag17", "float"),
    ColumnSpec("har_forecast", "float"),
    ColumnSpec("vix_level", "float"),
    ColumnSpec("vix_zscore", "float"),
    ColumnSpec("vix_percentile", "float"),
    ColumnSpec("ret_1", "float"),
    ColumnSpec("ret_5", "float"),
    ColumnSpec("ret_21", "float"),
    ColumnSpec("dist_sma_20", "float"),
    ColumnSpec("rsi_14", "float"),
    ColumnSpec("range_pos_20", "float"),
    ColumnSpec("vol_break_20", "float"),
    ColumnSpec("atr_norm_z", "float"),
    ColumnSpec("dist_sma_20_z", "float"),
    ColumnSpec("corr_dax_60", "float"),
    ColumnSpec("corr_ftse_60", "float"),
    ColumnSpec("corr_stoxx_60", "float"),
    ColumnSpec("corr_nikkei_60", "float"),
    ColumnSpec("asia_overnight_1", "float"),
    ColumnSpec("europe_prev_1", "float"),
    ColumnSpec("beta_vix_60", "float"),
    ColumnSpec("dxy_ret_1", "float"),
    ColumnSpec("sector_dispersion_1", "float"),
    ColumnSpec("sector_count", "float"),
    ColumnSpec("sector_dispersion_1_z", "float"),
    ColumnSpec("fed_funds", "float"),
    ColumnSpec("fed_funds_chg_5", "float"),
    ColumnSpec("ust_10y", "float"),
    ColumnSpec("ust_10y_chg_5", "float"),
    ColumnSpec("ust_2y", "float"),
    ColumnSpec("ust_2y_chg_5", "float"),
    ColumnSpec("pendiente_2s10s", "float"),
    ColumnSpec("pendiente_2s10s_chg_5", "float"),
    ColumnSpec("cpi_yoy", "float"),
    ColumnSpec("pce_yoy", "float"),
    ColumnSpec("dxy", "float"),
    ColumnSpec("ust_10y_z", "float"),
    ColumnSpec("dxy_z", "float"),
    ColumnSpec("rv_percentile", "float"),
    ColumnSpec("garch_forecast", "float"),
    ColumnSpec("garch_forecast_z", "float"),
    ColumnSpec("efficiency_ratio_20", "float"),
    ColumnSpec("day_of_week", "float"),
    ColumnSpec("sessions_to_opex", "float"),
    ColumnSpec("is_es_roll_session", "float"),
)


#: Firmas declaradas por ``(capa, dataset)``. **Toda** escritura del proyecto tiene que
#: aparecer aquí (`data/` y `derived/`), y un `layout_version` solo sube cuando la firma
#: cambia (§12.10).
LAYOUTS: Final[dict[tuple[str, str], DatasetLayout]] = {
    ("raw", "market_daily"): DatasetLayout(
        layer="raw", dataset="market_daily", layout_version=1, payload=_PRICE_COLUMNS
    ),
    ("raw", "sectors"): DatasetLayout(
        layer="raw", dataset="sectors", layout_version=1, payload=_PRICE_COLUMNS
    ),
    ("raw", "market_intraday"): DatasetLayout(
        layer="raw",
        dataset="market_intraday",
        layout_version=1,
        payload=(
            ColumnSpec("open", "float"),
            ColumnSpec("high", "float"),
            ColumnSpec("low", "float"),
            ColumnSpec("close", "float"),
            ColumnSpec("volume", "float"),
            ColumnSpec("interval", "str"),
            ColumnSpec("bid", "str"),
            ColumnSpec("ask", "str"),
        ),
    ),
    ("raw", "macro"): DatasetLayout(
        layer="raw",
        dataset="macro",
        layout_version=1,
        payload=(
            ColumnSpec("value", "float"),
            ColumnSpec("unit", "str"),
            ColumnSpec("name", "str"),
        ),
    ),
    ("raw", "news_headlines"): DatasetLayout(
        layer="raw",
        dataset="news_headlines",
        layout_version=1,
        payload=(
            ColumnSpec("title", "str"),
            ColumnSpec("url", "str"),
            ColumnSpec("headline_hash", "str"),
        ),
        identity_columns=("headline_hash",),
    ),
    ("raw", "earnings"): DatasetLayout(
        layer="raw",
        dataset="earnings",
        layout_version=1,
        payload=(
            ColumnSpec("name", "str"),
            ColumnSpec("event_date", "date"),
            ColumnSpec("moment", "str"),
            ColumnSpec("certainty", "str"),
            ColumnSpec("observed_at", "datetime"),
        ),
    ),
    ("derived", "features_daily"): DatasetLayout(
        layer="derived",
        dataset="features_daily",
        layout_version=1,
        payload=(*FEATURE_SPEC_COLUMNS, *_FEATURE_VERSION_COLUMNS),
    ),
    ("derived", "labels"): DatasetLayout(
        layer="derived",
        dataset="labels",
        layout_version=1,
        payload=(
            _SESSION,
            ColumnSpec("entry_px", "float"),
            ColumnSpec("entry_price_source", "str"),
            ColumnSpec("order_source", "str"),
            ColumnSpec("bars_observed", "int"),
            ColumnSpec("expected_bars", "int"),
            ColumnSpec("coverage", "float"),
            ColumnSpec("intraday_incomplete", "bool"),
            ColumnSpec("is_half_day", "bool"),
            ColumnSpec("ties_in_bar", "int"),
            ColumnSpec("fallback_ties", "int"),
            ColumnSpec("decided_bar", "int"),
            ColumnSpec("label_long", "str"),
            ColumnSpec("label_short", "str"),
            ColumnSpec("exit_long", "float"),
            ColumnSpec("exit_short", "float"),
            ColumnSpec("ret_long", "float"),
            ColumnSpec("ret_short", "float"),
            ColumnSpec("target_pct", "float"),
            ColumnSpec("stop_pct", "float"),
            ColumnSpec("sigma", "float"),
            ColumnSpec("sigma_carrier", "str"),
            ColumnSpec("forecast_candidate", "str"),
            ColumnSpec("selection_verdict", "str"),
            ColumnSpec("forecast_sha256", "str"),
            ColumnSpec("k_sigma", "float"),
        ),
    ),
}
