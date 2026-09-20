"""Persistencia versionada de la matriz de features (`derived.features_daily`, tarea #19).

Este modulo entrega **la infraestructura** del *feature store*: el esquema, la
identidad reproducible y la normalizacion robusta de ventana expandida. **No**
entrega familias de features: eso es #20-#23 (y la suite de integridad/cableado en
CI es #17). El catalogo que publica aqui es el de la familia de volatilidad, que
ya existe completa en :mod:`cfdtrader.features.volatility` (#7) y que este modulo
**no reimplementa**: la llama.

Politica de versionado
----------------------

**Que entra en el hash.** La *spec* tiene cinco campos declarados
(`feature_set`, `code_version`, `parameters`, `windows`, `sources`) y el hash
anade el `as_of` de la fila. Se serializa a JSON **canonico** (claves ordenadas,
separadores minimos, ASCII) y se aplica ``sha256``. Consecuencia buscada: el
**orden de insercion de un `dict` no cambia el hash**, y tocar cualquiera de los
seis campos con sentido si.

**Dos valores, no uno** (`tech_stack.md` §12.9):

- ``feature_spec_sha256`` — el digest de la spec **sin** ``as_of``. Identifica el
  *contrato de calculo*: dos sesiones del mismo codigo y los mismos parametros lo
  comparten. Es lo que se compara entre una recomputacion y lo guardado.
- ``features_version`` — ``"sha256:" + sha256(spec + as_of)``, **por sesion**.
  Distingue "cambio el codigo" de "cambio el dia", que es lo que hace posible la
  prueba anual de reconstruccion.

**Que NO entra.** El codigo de features **no** es el codigo fuente: es la
constante declarada :data:`FEATURE_CODE_VERSION`. Reformatear este modulo, mover
un comentario o reescribir un docstring **no** cambia ningun hash; subir la
constante, si. La razon es que un hash de los bytes del fichero cambiaria por un
`ruff format` y convertiria la prueba de reconstruccion en ruido.

**Cuando sube la constante.** Siempre que cambie el **resultado** del calculo de
cualquier feature: formula, ventana, fuente de entrada, redondeo o el orden de
las operaciones. Subirla es barato; no subirla es lo caro.

**Que obliga.** Un cambio de codigo de features **invalida los backtests
anteriores**: sus resultados ya no son reproducibles con el codigo nuevo y hay
que reevaluarlos (`_docs/plan.md` Apendice B). El *golden dataset* de
:data:`tests/fixtures/features/` congela el contrato: si el digest de la matriz
cambia sin que `FEATURE_CODE_VERSION` haya subido, el test falla. Nunca se
"arregla" una feature en produccion sin versionarla.

Esquema persistido (`tech_stack.md` §12.4)
-----------------------------------------

Una fila **por sesion**, en **formato ancho**:

===============  ==================================================================
Columna          Origen
===============  ==================================================================
``source``       :data:`FEATURES_SOURCE` (``cfdtrader.features.store``)
``series_id``    la serie de entrada (``^GSPC``)
``as_of``        el **cierre de sesion en UTC**, que pasa el llamante
``fetched_at``   lo pasa el llamante; el modulo **no lee el reloj**
``published_at`` ``NULL``: un dataset derivado no tiene editor externo
``version``      lo posee el ``Store``; el modulo nunca escribe este campo
``features_version`` / ``feature_spec_sha256``  el contrato de arriba
``<feature>``    una columna por feature del catalogo
===============  ==================================================================

La capa es siempre ``derived``: ``raw`` no se toca desde aqui. La revision
vigente se lee con ``store.sql(...)`` (o con :func:`load_daily`); ``read_pit``
responde "que sabiamos en T" y no sirve para un estudio historico.

Sin *look-ahead*
----------------

- La normalizacion es de **ventana expandida** (mediana y MAD de la historia
  ``<= t``), nunca de la muestra completa (`_docs/plan.md` §9): anadir sesiones
  posteriores no cambia el valor de una sesion pasada.
- El modulo **no** filtra sesiones ni aplica el corte de muestra limpia de
  ``analysis.drift``: eso es una restriccion de *estudio*, no del almacen.

Familias de features
--------------------

El modulo publica el **registro de familias**: cada ``feature_set`` declarado
(:data:`CATALOG_BY_FEATURE_SET`) tiene su catalogo y su ``source``
(:data:`SOURCE_BY_FEATURE_SET`). Una familia sin registrar es un error tipado:
no se inventa un catalogo por defecto.

La identidad del almacen es ``(source, series_id, as_of)`` y el esquema de #19
**no** guarda el ``feature_set`` (#49 todavia abierto), asi que el ``source`` es
lo unico que separa dos familias dentro del **mismo** dataset: sin el, escribir
la familia tecnica sustituiria las filas de volatilidad de la misma serie y el
mismo dia.

=====================  ================================  ======================
``feature_set``        ``source``                        catalogo
=====================  ================================  ======================
``volatility_v1``      ``cfdtrader.features.store``      :data:`FEATURE_CATALOG`
``technical_v1``       ``cfdtrader.features.technical``  :data:`TECHNICAL_FEATURE_CATALOG`
``context_v1``         ``cfdtrader.features.context``    :data:`CONTEXT_FEATURE_CATALOG`
=====================  ================================  ======================

Los valores por defecto de :class:`FeatureSpec` siguen siendo los de
``volatility_v1``: #20 y #21 anaden features nuevas, no cambian el resultado de
ninguna existente, asi que :data:`FEATURE_CODE_VERSION` **no** se mueve por
anadirlas. :func:`build_matrix` sigue siendo la entrada de ``volatility_v1``
(congelada por su *golden*); las otras dos familias tienen su propia funcion de
calculo, ``cfdtrader.features.technical.technical_matrix`` y
``cfdtrader.features.context.context_matrix``.

La familia de **contexto de mercado** (#21) es la unica que recibe **muchas**
series: su entrada es un ``Mapping`` con una entrada por serie
(:data:`CONTEXT_SERIES`), porque cada mercado trae su propio calendario y hay que
poder alinearlos sin inventar un frame largo. Un ``Mapping`` incompleto, con una
clave de mas o con una serie mal formada es :class:`ContextInputError`: el error
nombra la serie, que es lo unico que se puede arreglar desde fuera.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Final, cast

import numpy as np
import polars as pl

from cfdtrader.data.store import Layer, StorageError, Store, UnknownDatasetError, WriteOutcome
from cfdtrader.features.volatility import (
    ATR_WINDOW,
    HAR_LAG_MONTHLY,
    HAR_LAG_WEEKLY,
    VIX_MIN_SESSIONS,
    add_features,
)

__all__ = [
    "ALL_FEATURE_COLUMNS",
    "CATALOG_BY_FEATURE_SET",
    "CONTEXT_CORRELATION_WINDOW",
    "CONTEXT_FEATURES_SOURCE",
    "CONTEXT_FEATURE_CATALOG",
    "CONTEXT_FEATURE_COLUMNS",
    "CONTEXT_FEATURE_SET",
    "CONTEXT_MARKET_SERIES",
    "CONTEXT_MIN_SESSIONS",
    "CONTEXT_SECTOR_SERIES",
    "CONTEXT_SERIES",
    "DEFAULT_CONTEXT_SOURCES",
    "DEFAULT_CONTEXT_WINDOWS",
    "DEFAULT_TECHNICAL_SOURCES",
    "DEFAULT_TECHNICAL_WINDOWS",
    "FEATURES_DATASET",
    "FEATURES_LAYER",
    "FEATURES_SOURCE",
    "FEATURE_CATALOG",
    "FEATURE_CODE_VERSION",
    "FEATURE_VERSION_PREFIX",
    "MAD_SCALE",
    "NORMALISED_SUFFIX",
    "RANGE_WINDOW",
    "RETURN_LAGS",
    "RSI_WINDOW",
    "SOURCE_BY_FEATURE_SET",
    "TECHNICAL_FEATURES_SOURCE",
    "TECHNICAL_FEATURE_CATALOG",
    "TECHNICAL_FEATURE_COLUMNS",
    "TECHNICAL_FEATURE_SET",
    "TECHNICAL_MIN_SESSIONS",
    "VOLATILITY_FEATURE_SET",
    "CatalogEntry",
    "ContextInputError",
    "FeatureSpec",
    "FeatureStoreError",
    "InvalidFeatureMatrixError",
    "InvalidFeatureSpecError",
    "InvalidSeriesIdError",
    "build_matrix",
    "daily_records",
    "feature_spec_sha256",
    "features_version",
    "load_daily",
    "matrix_sha256",
    "normalise_expanding",
    "save_daily",
    "spec_payload",
]

#: Capa del almacen donde vive la matriz de features: es recalculable.
FEATURES_LAYER: Final[Layer] = "derived"

#: Dataset de la matriz de features (`tech_stack.md` §12.4).
FEATURES_DATASET: Final[str] = "features_daily"

#: ``source`` declarado de las filas: es un dataset derivado, no una fuente externa.
FEATURES_SOURCE: Final[str] = "cfdtrader.features.store"

#: Identificador del conjunto de features que este contrato publica.
VOLATILITY_FEATURE_SET: Final[str] = "volatility_v1"

#: Version declarada del **codigo de calculo** (A2). Arranca en 1 y **solo** sube
#: cuando cambia el resultado de alguna feature; reformatear el modulo no cuenta.
FEATURE_CODE_VERSION: Final[int] = 1

#: Sufijo de la columna normalizada que anade :func:`normalise_expanding`.
NORMALISED_SUFFIX: Final[str] = "_z"

#: Factor de consistencia de la MAD con la desviacion tipica de una normal.
#: ``MAD_SCALE = 1.4826`` es ``1 / Phi^{-1}(3/4)``: con el, ``MAD_SCALE * MAD``
#: estima ``sigma`` bajo normalidad.
MAD_SCALE: Final[float] = 1.4826

#: Prefijo de ``features_version``: deja ver de un vistazo que el valor es un sha256.
FEATURE_VERSION_PREFIX: Final[str] = "sha256:"

#: Columnas de identidad de la matriz de entrada (y de la persistida).
IDENTITY_COLUMNS: Final[tuple[str, ...]] = ("session", "as_of")

#: Columnas del contrato de versionado que la matriz puede traer ya calculadas:
#: se **verifican**, nunca se sobreescriben en silencio (A7).
VERSION_COLUMNS: Final[tuple[str, ...]] = ("features_version", "feature_spec_sha256")

#: Series de entrada admitidas como nombre de columna y en SQL: nombre simple.
_SERIES_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9^][A-Za-z0-9^._=-]*$")


# ─────────────────────────────────────────────────────────────────────────────
# Errores
# ─────────────────────────────────────────────────────────────────────────────
class FeatureStoreError(StorageError):
    """Base de los errores del *feature store*.

    Hereda de la jerarquia del almacen (`.data.store.StorageError`) porque un
    fallo de esta capa es un fallo del almacen: asi, quien capture
    ``StorageError`` no se deja escapar una matriz mal formada.
    """


class InvalidFeatureSpecError(FeatureStoreError):
    """La *spec* no es valida: falta un campo, no es serializable o contradice al catalogo."""


class InvalidFeatureMatrixError(FeatureStoreError):
    """La matriz no cumple el contrato: columna ausente, `NaN`/`inf`, sesion repetida..."""


class InvalidSeriesIdError(FeatureStoreError):
    """``series_id`` vacio o con caracteres que no pueden viajar a una ruta o a SQL."""


class ContextInputError(FeatureStoreError):
    """El ``Mapping`` de series de la familia de contexto no cumple su contrato.

    La familia ``context_v1`` (#21) no recibe un frame: recibe una serie por
    mercado, porque cada uno trae su calendario. Falta una serie, sobra una
    clave, falta ``session`` o ``close``, una sesion se repite o un ``as_of`` no
    corresponde a su sesion: todo eso es un error de **entrada**, y el mensaje
    nombra la serie, porque es lo unico que se puede arreglar desde fuera.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Catalogo
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """Una feature del catalogo, documentada como pide `_docs/plan.md` §9.

    Parameters
    ----------
    name:
        Nombre de la columna persistida.
    formula:
        Formula en texto, sin dependencias de la implementacion.
    window:
        Sesiones de historia que necesita: las que entran en la formula o, si la
        ventana es expandida, el minimo de sesiones. ``None`` = sin ventana fija.
    source:
        De donde sale el dato de entrada (``raw.market_daily``).
    required_as_of:
        Cierre de sesion a partir del cual la feature esta completamente definida.
    """

    name: str
    formula: str
    window: int | None
    source: str
    required_as_of: str


#: Catalogo completo de la familia de volatilidad (#7). Toda columna persistida
#: aparece aqui, y las ventanas son las de `cfdtrader.features.volatility`: se
#: importan, no se copian, para que no puedan divergir.
FEATURE_CATALOG: Final[tuple[CatalogEntry, ...]] = (
    CatalogEntry(
        "true_range",
        "max(H-L, abs(H-C_{t-1}), abs(L-C_{t-1}))",
        1,
        "raw.market_daily",
        "cierre de la sesion t",
    ),
    CatalogEntry(
        "atr_norm",
        f"media de true_range de las {ATR_WINDOW} sesiones anteriores / C_{{t-1}}",
        ATR_WINDOW,
        "raw.market_daily",
        "cierre de la sesion t",
    ),
    CatalogEntry(
        "parkinson_rv",
        "(ln(H/L))^2 / (4 * ln 2)",
        1,
        "raw.market_daily",
        "cierre de la sesion t",
    ),
    CatalogEntry(
        "ret_log",
        "ln(C/O)",
        1,
        "raw.market_daily",
        "cierre de la sesion t",
    ),
    CatalogEntry(
        "ret_sq",
        "(ln(C/O))^2",
        1,
        "raw.market_daily",
        "cierre de la sesion t",
    ),
    CatalogEntry(
        "har_lag1",
        "parkinson_rv de la sesion t-1",
        1,
        "raw.market_daily",
        "cierre de la sesion t-1",
    ),
    CatalogEntry(
        "har_lag4",
        f"media de parkinson_rv de t-2 ... t-{1 + HAR_LAG_WEEKLY}",
        HAR_LAG_WEEKLY,
        "raw.market_daily",
        "cierre de la sesion t-1",
    ),
    CatalogEntry(
        "har_lag17",
        f"media de parkinson_rv de t-6 ... t-{1 + HAR_LAG_WEEKLY + HAR_LAG_MONTHLY}",
        HAR_LAG_MONTHLY,
        "raw.market_daily",
        "cierre de la sesion t-1",
    ),
    CatalogEntry(
        "har_forecast",
        "exp(pronostico HAR a un paso, ajustado con las sesiones < t)",
        None,
        "raw.market_daily",
        "cierre de la sesion t-1",
    ),
    CatalogEntry(
        "vix_level",
        "cierre del VIX de la sesion t-1",
        1,
        "raw.market_daily",
        "cierre de la sesion t-1",
    ),
    CatalogEntry(
        "vix_zscore",
        f"(vix_level - media expandida) / desviacion expandida (ddof=1), {VIX_MIN_SESSIONS} "
        "sesiones como minimo",
        VIX_MIN_SESSIONS,
        "raw.market_daily",
        "cierre de la sesion t-1",
    ),
    CatalogEntry(
        "vix_percentile",
        f"fraccion de los niveles anteriores <= vix_level, {VIX_MIN_SESSIONS} sesiones como minimo",
        VIX_MIN_SESSIONS,
        "raw.market_daily",
        "cierre de la sesion t-1",
    ),
)

#: Catalogo indexado por nombre, para las comprobaciones cruzadas.
CATALOG_BY_NAME: Final[dict[str, CatalogEntry]] = {entry.name: entry for entry in FEATURE_CATALOG}

#: Columnas de feature que persiste la matriz: **todas** las del catalogo.
FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(entry.name for entry in FEATURE_CATALOG)

#: Ventanas por defecto declaradas en la spec: las del catalogo, sin excepciones.
DEFAULT_WINDOWS: Final[dict[str, int | None]] = {
    entry.name: entry.window for entry in FEATURE_CATALOG
}

#: Fuentes de entrada por defecto del conjunto de volatilidad: el indice y su VIX.
DEFAULT_SOURCES: Final[tuple[tuple[str, str], ...]] = (
    ("raw.market_daily", "^GSPC"),
    ("raw.market_daily", "^VIX"),
)

#: Columnas de entrada que necesita el calculo de la familia de volatilidad.
INPUT_COLUMNS: Final[tuple[str, ...]] = ("session", "open", "high", "low", "close")


# ─────────────────────────────────────────────────────────────────────────────
# Catalogo de la familia tecnica (#20)
# ─────────────────────────────────────────────────────────────────────────────
#: Identificador del conjunto de features tecnicas (`_docs/plan.md` §9).
TECHNICAL_FEATURE_SET: Final[str] = "technical_v1"

#: ``source`` de las filas tecnicas. **No** es decorativo: es el discriminador de
#: familia dentro de ``derived.features_daily`` mientras el ``feature_set`` no
#: forme parte de la identidad del almacen (#49).
TECHNICAL_FEATURES_SOURCE: Final[str] = "cfdtrader.features.technical"

#: Sesiones minimas de la ventana **expandida** de las dos ``_z`` tecnicas.
TECHNICAL_MIN_SESSIONS: Final[int] = 250

#: Retardo de cada retorno multi-ventana, en sesiones.
RETURN_LAGS: Final[tuple[tuple[str, int], ...]] = (("ret_1", 1), ("ret_5", 5), ("ret_21", 21))

#: Ventana del RSI de Wilder (sesiones de historia del suavizado).
RSI_WINDOW: Final[int] = 14

#: Ventana de las features de media, rango y ruptura (`dist_sma_20`, `range_pos_20`,
#: `vol_break_20`; el sufijo _20 del nombre es este mismo numero).
RANGE_WINDOW: Final[int] = 20

#: Catalogo completo de la familia tecnica (10 entradas, #20). Vive aqui, con la
#: spec, y no en ``technical.py``: el catalogo es la **declaracion** del contrato
#: y las formulas lo leen, de modo que no puede haber una segunda copia de las
#: ventanas. ``technical.py`` importa de este modulo; la dependencia va en un solo
#: sentido y no hay ciclos.
TECHNICAL_FEATURE_CATALOG: Final[tuple[CatalogEntry, ...]] = (
    CatalogEntry(
        "ret_1",
        "ln(C_t / C_{t-1})",
        1,
        "raw.market_daily",
        "cierre de la sesion t",
    ),
    CatalogEntry(
        "ret_5",
        "ln(C_t / C_{t-5})",
        5,
        "raw.market_daily",
        "cierre de la sesion t",
    ),
    CatalogEntry(
        "ret_21",
        "ln(C_t / C_{t-21})",
        21,
        "raw.market_daily",
        "cierre de la sesion t",
    ),
    CatalogEntry(
        "atr_norm",
        f"media de true_range de t-{ATR_WINDOW} ... t-1 / C_{{t-1}} (importada de #7)",
        ATR_WINDOW,
        "raw.market_daily",
        "cierre de la sesion t-1",
    ),
    CatalogEntry(
        "dist_sma_20",
        f"C_t / media(C_{{t-{RANGE_WINDOW - 1}}} ... C_t) - 1",
        RANGE_WINDOW,
        "raw.market_daily",
        "cierre de la sesion t",
    ),
    CatalogEntry(
        "rsi_14",
        f"RSI de Wilder ({RSI_WINDOW}) sobre d_t = C_t - C_{{t-1}}, alpha = 1/{RSI_WINDOW}, "
        f"semilla = media simple de los {RSI_WINDOW} primeros d",
        RSI_WINDOW,
        "raw.market_daily",
        "cierre de la sesion t",
    ),
    CatalogEntry(
        "range_pos_20",
        f"(C_t - min(low de t-{RANGE_WINDOW - 1} ... t)) / "
        f"(max(high de t-{RANGE_WINDOW - 1} ... t) - min(low de t-{RANGE_WINDOW - 1} ... t))",
        RANGE_WINDOW,
        "raw.market_daily",
        "cierre de la sesion t",
    ),
    CatalogEntry(
        "vol_break_20",
        f"true_range_t / media(true_range de t-{RANGE_WINDOW} ... t-1)",
        RANGE_WINDOW,
        "raw.market_daily",
        "cierre de la sesion t",
    ),
    CatalogEntry(
        "atr_norm_z",
        f"normalise_expanding(atr_norm, min_sessions={TECHNICAL_MIN_SESSIONS})",
        TECHNICAL_MIN_SESSIONS,
        "raw.market_daily",
        "cierre de la sesion t-1",
    ),
    CatalogEntry(
        "dist_sma_20_z",
        f"normalise_expanding(dist_sma_20, min_sessions={TECHNICAL_MIN_SESSIONS})",
        TECHNICAL_MIN_SESSIONS,
        "raw.market_daily",
        "cierre de la sesion t",
    ),
)

#: Columnas de feature que persiste la matriz tecnica: **todas** las del catalogo.
TECHNICAL_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    entry.name for entry in TECHNICAL_FEATURE_CATALOG
)

#: Ventanas por defecto de la spec tecnica: las del catalogo, sin excepciones.
DEFAULT_TECHNICAL_WINDOWS: Final[dict[str, int | None]] = {
    entry.name: entry.window for entry in TECHNICAL_FEATURE_CATALOG
}

#: Fuentes de entrada por defecto del conjunto tecnico: el indice, **sin** VIX.
DEFAULT_TECHNICAL_SOURCES: Final[tuple[tuple[str, str], ...]] = (("raw.market_daily", "^GSPC"),)

# ─────────────────────────────────────────────────────────────────────────────
# Catalogo de la familia de contexto de mercado (#21)
# ─────────────────────────────────────────────────────────────────────────────
#: Identificador del conjunto de features de contexto (`_docs/plan.md` §7.1).
CONTEXT_FEATURE_SET: Final[str] = "context_v1"

#: ``source`` de las filas de contexto (mismo papel que el de la familia tecnica).
CONTEXT_FEATURES_SOURCE: Final[str] = "cfdtrader.features.context"

#: Ventana de las cuatro correlaciones moviles y de la beta del VIX, en sesiones
#: del S&P 500: entra en el nombre de la columna.
CONTEXT_CORRELATION_WINDOW: Final[int] = 60

#: Sesiones minimas de la ventana **expandida** de ``sector_dispersion_1_z``.
CONTEXT_MIN_SESSIONS: Final[int] = 250

#: Series de ``raw.market_daily``: el indice (que es el ancla del calendario) y
#: las siete series de contexto (VIX, tres indices europeos, dos asiaticos y DXY).
CONTEXT_MARKET_SERIES: Final[tuple[str, ...]] = (
    "^GSPC",
    "^VIX",
    "^GDAXI",
    "^FTSE",
    "^STOXX50E",
    "^N225",
    "^HSI",
    "DX-Y.NYB",
)

#: Los **once** ETF sectoriales de ``raw.sectors`` (`_docs/plan.md` §7.1). Orden
#: alfabetico a proposito: el orden no entra en ningun hash, y asi no se puede
#: confundir con una jerarquia.
CONTEXT_SECTOR_SERIES: Final[tuple[str, ...]] = (
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
)

#: Las **19** series de entrada de la familia, en orden: 8 de ``raw.market_daily``
#: y los 11 ETF de ``raw.sectors``. No es una lista descriptiva: el ``Mapping``
#: que recibe ``context_matrix`` tiene que tener exactamente estas claves.
CONTEXT_SERIES: Final[tuple[str, ...]] = (*CONTEXT_MARKET_SERIES, *CONTEXT_SECTOR_SERIES)

#: Catalogo completo de la familia de contexto (11 entradas, #21). Las formulas
#: viven en ``cfdtrader.features.context``; aqui esta la **declaracion** del
#: contrato (ventana, fuente y cierre del que depende cada columna).
CONTEXT_FEATURE_CATALOG: Final[tuple[CatalogEntry, ...]] = (
    CatalogEntry(
        "corr_dax_60",
        "Pearson de (retorno logaritmico del S&P, retorno del DAX en su ultima sesion "
        f"<= s) sobre las {CONTEXT_CORRELATION_WINDOW} sesiones s <= t-1",
        CONTEXT_CORRELATION_WINDOW,
        "raw.market_daily",
        "cierre de la sesion t-1",
    ),
    CatalogEntry(
        "corr_ftse_60",
        "Pearson de (retorno logaritmico del S&P, retorno del FTSE 100 en su ultima "
        f"sesion <= s) sobre las {CONTEXT_CORRELATION_WINDOW} sesiones s <= t-1",
        CONTEXT_CORRELATION_WINDOW,
        "raw.market_daily",
        "cierre de la sesion t-1",
    ),
    CatalogEntry(
        "corr_stoxx_60",
        "Pearson de (retorno logaritmico del S&P, retorno del EURO STOXX 50 en su "
        f"ultima sesion <= s) sobre las {CONTEXT_CORRELATION_WINDOW} sesiones s <= t-1",
        CONTEXT_CORRELATION_WINDOW,
        "raw.market_daily",
        "cierre de la sesion t-1",
    ),
    CatalogEntry(
        "corr_nikkei_60",
        "Pearson de (retorno logaritmico del S&P, retorno del Nikkei 225 en su ultima "
        f"sesion <= s) sobre las {CONTEXT_CORRELATION_WINDOW} sesiones s <= t-1",
        CONTEXT_CORRELATION_WINDOW,
        "raw.market_daily",
        "cierre de la sesion t-1",
    ),
    CatalogEntry(
        "asia_overnight_1",
        "media de los retornos del Nikkei 225 y del Hang Seng de su ultima sesion <= t",
        1,
        "raw.market_daily",
        "cierre asiatico de la sesion t",
    ),
    CatalogEntry(
        "europe_prev_1",
        "media de los retornos del DAX, del FTSE 100 y del EURO STOXX 50 de su ultima sesion < t",
        1,
        "raw.market_daily",
        "cierre europeo de la sesion t-1",
    ),
    CatalogEntry(
        "beta_vix_60",
        "pendiente OLS del retorno logaritmico del S&P sobre el del VIX en los "
        f"{CONTEXT_CORRELATION_WINDOW} pares s <= t-1",
        CONTEXT_CORRELATION_WINDOW,
        "raw.market_daily",
        "cierre de la sesion t-1",
    ),
    CatalogEntry(
        "dxy_ret_1",
        "retorno logaritmico del indice dolar de la ultima sesion del DXY < t",
        1,
        "raw.market_daily",
        "cierre de la sesion t-1",
    ),
    CatalogEntry(
        "sector_dispersion_1",
        "desviacion estandar muestral (ddof=1) del retorno de los ETF sectoriales con "
        "dato en su ultima sesion < t",
        1,
        "raw.sectors",
        "cierre de la sesion t-1",
    ),
    CatalogEntry(
        "sector_count",
        "numero de ETF sectoriales con retorno disponible (0-11) en su ultima sesion < t",
        1,
        "raw.sectors",
        "cierre de la sesion t-1",
    ),
    CatalogEntry(
        "sector_dispersion_1_z",
        f"normalise_expanding(sector_dispersion_1, min_sessions={CONTEXT_MIN_SESSIONS})",
        CONTEXT_MIN_SESSIONS,
        "raw.sectors",
        "cierre de la sesion t-1",
    ),
)

#: Columnas de feature que persiste la matriz de contexto: **todas** las del catalogo.
CONTEXT_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    entry.name for entry in CONTEXT_FEATURE_CATALOG
)

#: Ventanas por defecto de la spec de contexto: las del catalogo, sin excepciones.
DEFAULT_CONTEXT_WINDOWS: Final[dict[str, int | None]] = {
    entry.name: entry.window for entry in CONTEXT_FEATURE_CATALOG
}

#: Fuentes de entrada por defecto de la familia de contexto: las 19 series.
DEFAULT_CONTEXT_SOURCES: Final[tuple[tuple[str, str], ...]] = (
    *(("raw.market_daily", series_id) for series_id in CONTEXT_MARKET_SERIES),
    *(("raw.sectors", series_id) for series_id in CONTEXT_SECTOR_SERIES),
)


# ─────────────────────────────────────────────────────────────────────────────
# Registro de familias
# ─────────────────────────────────────────────────────────────────────────────
#: Registro de familias: el catalogo de cada ``feature_set`` declarado.
CATALOG_BY_FEATURE_SET: Final[dict[str, tuple[CatalogEntry, ...]]] = {
    VOLATILITY_FEATURE_SET: FEATURE_CATALOG,
    TECHNICAL_FEATURE_SET: TECHNICAL_FEATURE_CATALOG,
    CONTEXT_FEATURE_SET: CONTEXT_FEATURE_CATALOG,
}

#: ``source`` con el que se persiste cada familia (el discriminador de la
#: decision 2 de #20). Toda familia del registro tiene que estar aqui.
SOURCE_BY_FEATURE_SET: Final[dict[str, str]] = {
    VOLATILITY_FEATURE_SET: FEATURES_SOURCE,
    TECHNICAL_FEATURE_SET: TECHNICAL_FEATURES_SOURCE,
    CONTEXT_FEATURE_SET: CONTEXT_FEATURES_SOURCE,
}

#: Todas las columnas de feature conocidas, de las tres familias: es la lista con
#: la que el digest de una matriz comprueba que no haya ``NaN`` ni ``inf``. Se
#: deduplica porque ``atr_norm`` existe en dos catalogos (el solape lo declara y
#: lo resuelve #72, no esta capa). Que las 11 columnas de contexto entren aqui no
#: es cosmetico: sin ellas, ``matrix_sha256`` **no** detectaria un ``NaN`` ni un
#: ``inf`` en la matriz de contexto (A1 de #21).
ALL_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys((*FEATURE_COLUMNS, *TECHNICAL_FEATURE_COLUMNS, *CONTEXT_FEATURE_COLUMNS))
)


# ─────────────────────────────────────────────────────────────────────────────
# Spec e identidad
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Contrato de calculo de una matriz de features.

    Parameters
    ----------
    feature_set:
        Nombre del conjunto (``volatility_v1``).
    code_version:
        Version **declarada** del codigo de calculo. Por defecto toma
        :data:`FEATURE_CODE_VERSION`, de modo que subir la constante cambia la
        identidad de todas las matrices construidas con la spec por defecto.
    parameters:
        Parametros del calculo, serializables a JSON.
    windows:
        Ventana declarada por feature. Es un **subconjunto** del catalogo: cada
        clave tiene que existir en el catalogo y declarar la **misma** ventana
        (una discrepancia es un error, no un valor distinto).
    sources:
        Fuentes de entrada como pares ``(dataset, serie)`` de ``raw.*``.
    """

    feature_set: str = VOLATILITY_FEATURE_SET
    code_version: int = FEATURE_CODE_VERSION
    parameters: Mapping[str, object] = field(default_factory=dict[str, object])
    windows: Mapping[str, int | None] = field(default_factory=lambda: dict(DEFAULT_WINDOWS))
    sources: tuple[tuple[str, str], ...] = DEFAULT_SOURCES

    def __post_init__(self) -> None:
        """Valida la spec **al construirla**: una spec mal formada no deberia existir.

        Se lanza un error tipado propio y no un ``ValueError``: un ``__post_init__``
        que lanza algo distinto de ``ValueError``/``AssertionError`` se propaga tal
        cual, sin que ``dataclasses`` lo envuelva.
        """
        _validate_spec(self)


def _is_text(value: object) -> bool:
    """``True`` solo para ``str`` de verdad.

    Va en un helper que recibe ``object`` a proposito: un ``isinstance`` sobre un
    valor ya anotado no comprueba nada (pyright lo marca como innecesario) y
    ademas quedaria muerto si el tipo declarado cambia.
    """
    return isinstance(value, str)


def _is_int(value: object) -> bool:
    """``True`` solo para ``int`` de verdad: ``bool`` es subclase y no cuenta."""
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_json(payload: object) -> str:
    """JSON canonico: claves ordenadas, sin espacios y solo ASCII.

    Es la unica forma de que el hash no dependa del orden de un ``dict`` ni de la
    plataforma. Un valor no serializable o no finito es un error tipado, nunca un
    hash calculado sobre una representacion distinta.
    """
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise InvalidFeatureSpecError(
            f"la spec no es serializable a JSON canonico: {error}"
        ) from error


def _catalog_for(feature_set: str) -> tuple[CatalogEntry, ...]:
    """Catalogo registrado de una familia.

    Una familia sin registrar es un error tipado, nunca un catalogo vacio: una
    spec que no casa con ningun contrato no puede llegar a escribir nada.
    """
    catalog = CATALOG_BY_FEATURE_SET.get(feature_set)
    if catalog is None:
        raise InvalidFeatureSpecError(
            f"la familia de features '{feature_set}' no esta registrada: las declaradas son "
            f"{sorted(CATALOG_BY_FEATURE_SET)}"
        )
    return catalog


def _source_for(feature_set: str) -> str:
    """``source`` con el que se persiste una familia (lo que la separa de las demas)."""
    _catalog_for(feature_set)
    return SOURCE_BY_FEATURE_SET[feature_set]


def _feature_columns_for(feature_set: str) -> tuple[str, ...]:
    """Columnas de feature publicadas por una familia, en el orden del catalogo."""
    return tuple(entry.name for entry in _catalog_for(feature_set))


def _validate_spec(spec: FeatureSpec) -> None:
    """Comprueba los tipos de la spec y que sus ventanas no contradigan a su catalogo.

    Una familia **registrada** se valida contra **su** catalogo. Una familia sin
    registrar se valida contra :data:`FEATURE_CATALOG`, que es exactamente el
    contrato congelado de #19: su spec se tiene que poder **hashear** (el digest
    identifica un contrato aunque la familia no este dada de alta) y lo que cae es
    la **resolucion** de la familia, en :func:`_source_for`. Cambiar esto romperia
    ``test_a1`` de #19, que exige que ``volatility_v2`` siga dando un digest.
    """
    if not _is_text(spec.feature_set) or not spec.feature_set.strip():
        raise InvalidFeatureSpecError("'feature_set' es obligatorio y no puede estar vacio")
    catalog = CATALOG_BY_FEATURE_SET.get(spec.feature_set, FEATURE_CATALOG)
    if not _is_int(spec.code_version):
        raise InvalidFeatureSpecError(
            f"'code_version' debe ser un entero >= 1, no {type(spec.code_version).__name__}"
        )
    if spec.code_version < 1:
        raise InvalidFeatureSpecError(f"'code_version' debe ser >= 1, no {spec.code_version}")
    if not spec.windows:
        raise InvalidFeatureSpecError(
            "'windows' no puede estar vacio: declara al menos la ventana de una feature"
        )
    by_name = {entry.name: entry for entry in catalog}
    for name, window in spec.windows.items():
        entry = by_name.get(name)
        if entry is None:
            raise InvalidFeatureSpecError(
                f"'{name}' no esta en el catalogo de '{spec.feature_set}': las features "
                f"declaradas son {sorted(by_name)}"
            )
        if window != entry.window:
            raise InvalidFeatureSpecError(
                f"la ventana declarada de '{name}' ({window!r}) no coincide con la del catalogo "
                f"({entry.window!r}): el catalogo y la spec tienen que decir lo mismo"
            )


def spec_payload(spec: FeatureSpec) -> dict[str, object]:
    """Forma canonica de la spec: los **cinco** campos que entran en el hash.

    ``windows`` y ``sources`` se ordenan aqui, de modo que reordenarlos en la
    entrada no cambia el digest.
    """
    _validate_spec(spec)
    sources: list[list[str]] = [list(pair) for pair in sorted(spec.sources)]
    payload: dict[str, object] = {
        "feature_set": spec.feature_set,
        "code_version": spec.code_version,
        "parameters": dict(spec.parameters),
        "windows": {name: spec.windows[name] for name in sorted(spec.windows)},
        "sources": sources,
    }
    _canonical_json(payload)
    return payload


def feature_spec_sha256(spec: FeatureSpec) -> str:
    """Digest del contrato de calculo **sin** ``as_of`` (A3).

    Dos sesiones distintas del mismo codigo y los mismos parametros comparten
    este valor; es lo que se compara entre una recomputacion y lo guardado.
    """
    return hashlib.sha256(_canonical_json(spec_payload(spec)).encode("utf-8")).hexdigest()


def features_version(spec: FeatureSpec, as_of: datetime) -> str:
    """Identidad **por sesion**: ``"sha256:" + sha256(spec + as_of)`` (A1).

    Parameters
    ----------
    spec:
        Contrato de calculo.
    as_of:
        Cierre de sesion en UTC de la fila. Tiene que llevar zona horaria: un
        instante ambiguo no puede entrar en un hash.
    """
    instant = _as_utc(as_of, field="as_of", error=InvalidFeatureSpecError)
    payload = {**spec_payload(spec), "as_of": instant.isoformat()}
    return (
        FEATURE_VERSION_PREFIX
        + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    )


def _as_utc(value: object, *, field: str, error: type[FeatureStoreError]) -> datetime:
    """Normaliza a UTC un instante con zona; un ``datetime`` naive es un error."""
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise error(
                f"'{field}' tiene que llevar zona horaria (UTC), no un datetime naive: {value!r}"
            )
        return value.astimezone(UTC)
    raise error(f"'{field}' debe ser un datetime UTC, no {type(value).__name__}")


# ─────────────────────────────────────────────────────────────────────────────
# Normalizacion robusta de ventana expandida
# ─────────────────────────────────────────────────────────────────────────────
def _column_values(series: pl.Series, *, column: str) -> list[float | None]:
    """Columna como floats con ``None`` en los nulos.

    Un valor no numerico, ``NaN`` o ``inf`` es un error tipado (A10): lo no
    computable se publica como ``null``, nunca como ``NaN`` ni ``inf``.
    """
    values: list[float | None] = []
    for value in cast("list[object]", series.to_list()):
        if value is None:
            values.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidFeatureMatrixError(
                f"la columna '{column}' no es numerica: {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise InvalidFeatureMatrixError(
                f"la columna '{column}' contiene un valor no finito ({number!r}): lo no "
                "computable se publica como null, no como NaN ni inf"
            )
        values.append(number)
    return values


def normalise_expanding(frame: pl.DataFrame, column: str, *, min_sessions: int) -> pl.DataFrame:
    """Anade ``{column}_z``: z-score robusto de **ventana expandida** (A8).

    $$z_t = \\frac{x_t - \\mathrm{mediana}_{\\le t}}{1{,}4826 \\cdot \\mathrm{MAD}_{\\le t}}$$

    ``MAD_{\\le t}`` es la mediana de ``|x_i - mediana_{\\le t}|`` para ``i <= t``,
    con la **misma** ventana que la mediana: nunca la muestra completa
    (`_docs/plan.md` §9; normalizar sobre todo el conjunto es *look-ahead*).

    Las ``min_sessions - 1`` primeras sesiones quedan a ``null``, y tambien la
    sesion cuyo ``MAD`` sea exactamente ``0``: dividir por el daria ``inf``, y un
    ``inf`` no es un dato. Un nulo de la columna no cuenta como sesion de
    historia y devuelve ``null``; un ``NaN`` o un ``inf`` es un error tipado.

    Parameters
    ----------
    frame:
        Frame con la columna a normalizar. El orden de las filas se conserva.
    column:
        Nombre de la columna de features.
    min_sessions:
        Sesiones de historia exigidas. **Sin valor por defecto**: la ventana es
        una decision declarada, no algo que el modulo elija por su cuenta.

    Returns
    -------
    pl.DataFrame
        El mismo frame con la columna ``{column}_z`` anadida.
    """
    if not _is_int(min_sessions) or min_sessions < 1:
        raise InvalidFeatureSpecError(
            f"'min_sessions' debe ser un entero >= 1, no {min_sessions!r}"
        )
    if column not in frame.columns:
        raise InvalidFeatureMatrixError(
            f"la columna '{column}' no esta en el frame: hay {sorted(frame.columns)}"
        )

    values = _column_values(frame.get_column(column), column=column)
    history: list[float] = []
    normalised: list[float | None] = []
    for value in values:
        if value is None:
            normalised.append(None)
            continue
        history.append(value)
        if len(history) < min_sessions:
            normalised.append(None)
            continue
        window = np.asarray(history, dtype=float)
        centre = float(np.median(window))
        mad = float(np.median(np.abs(window - centre)))
        if mad == 0.0:
            normalised.append(None)
            continue
        normalised.append((value - centre) / (MAD_SCALE * mad))

    return frame.with_columns(
        pl.Series(f"{column}{NORMALISED_SUFFIX}", normalised, dtype=pl.Float64)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Matriz de features
# ─────────────────────────────────────────────────────────────────────────────
def build_matrix(frame: pl.DataFrame, *, spec: FeatureSpec) -> pl.DataFrame:
    """Calcula la matriz persistible: ``session`` + **todas** las features del catalogo.

    Reutiliza la API publica de :mod:`cfdtrader.features.volatility` (#7): la
    volatilidad no se reimplementa aqui. El frame de entrada tiene que traer
    ``session``, ``open``, ``high``, ``low``, ``close`` y ``vix_close``; las
    columnas de entrada no se persisten (solo las features).

    Returns
    -------
    pl.DataFrame
        ``session`` mas una columna por feature del catalogo, ordenado por sesion
        y con lo no computable a ``null``.
    """
    _validate_spec(spec)
    if spec.feature_set != VOLATILITY_FEATURE_SET:
        raise InvalidFeatureMatrixError(
            f"'build_matrix' es la entrada de '{VOLATILITY_FEATURE_SET}': la familia "
            f"'{spec.feature_set}' tiene su propia funcion de calculo"
        )
    missing = [name for name in (*INPUT_COLUMNS, "vix_close") if name not in frame.columns]
    if missing:
        raise InvalidFeatureMatrixError(
            f"faltan columnas de entrada para calcular la familia de volatilidad: {missing}"
        )

    computed = add_features(frame)
    feature_columns = [name for name in FEATURE_COLUMNS if name in computed.columns]
    absent = sorted(set(FEATURE_COLUMNS) - set(feature_columns))
    if absent:
        raise InvalidFeatureMatrixError(
            f"el calculo no produjo estas features del catalogo: {absent}"
        )

    matrix = computed.select(["session", *FEATURE_COLUMNS]).sort("session")
    return _require_finite(matrix, FEATURE_COLUMNS)


def matrix_sha256(matrix: pl.DataFrame) -> str:
    """Digest canonico de una matriz de features, para comparar dos calculos.

    Las columnas se ordenan alfabeticamente y las filas por ``session`` (y
    ``as_of`` si lo traen), asi que el digest **no** depende del orden en que se
    construyo el frame. Es lo que congela el *golden dataset* (A4) y lo que
    permite recomputar y comparar con lo guardado.
    """
    frame = matrix.select(sorted(matrix.columns))
    ordering = [name for name in IDENTITY_COLUMNS if name in frame.columns]
    if ordering:
        frame = frame.sort(ordering)

    # El catalogo completo de las dos familias, no solo las columnas presentes: asi
    # una matriz parcial (un subconjunto de features) tambien se puede hashear.
    frame = _require_finite(frame, ALL_FEATURE_COLUMNS)

    rows: list[list[object]] = []
    for row in frame.iter_rows():
        rows.append([_json_scalar(value) for value in row])
    payload: dict[str, object] = {"columns": frame.columns, "rows": rows}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _json_scalar(value: object) -> object:
    """Valor de una celda en forma serializable: los instantes, en ISO 8601."""
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value


def _require_finite(frame: pl.DataFrame, columns: list[str] | tuple[str, ...]) -> pl.DataFrame:
    """Comprueba que las columnas de feature son numericas y finitas (A10)."""
    for name in columns:
        if name in frame.columns:
            _column_values(frame.get_column(name), column=name)
    return frame


def _session_dates(matrix: pl.DataFrame) -> list[date]:
    """Sesiones de la matriz, en orden de fila; admite ``Date`` y ``Datetime``."""
    series = matrix.get_column("session")
    if isinstance(series.dtype, pl.Datetime):
        series = series.dt.date()
    elif not isinstance(series.dtype, pl.Date):
        raise InvalidFeatureMatrixError(
            f"'session' debe ser pl.Date o pl.Datetime, no {series.dtype}"
        )
    return [cast("date", value) for value in series.to_list()]


def _instants(matrix: pl.DataFrame, *, field: str) -> list[datetime]:
    """Instantes UTC de una columna; un ``Datetime`` sin zona es un error."""
    series = matrix.get_column(field)
    dtype = series.dtype
    if not isinstance(dtype, pl.Datetime):
        raise InvalidFeatureMatrixError(f"'{field}' debe ser pl.Datetime con zona UTC, no {dtype}")
    if dtype.time_zone is None:
        raise InvalidFeatureMatrixError(
            f"'{field}' no lleva zona horaria: un instante sin zona no puede entrar en el hash"
        )
    return [
        _as_utc(value, field=field, error=InvalidFeatureMatrixError) for value in series.to_list()
    ]


def _require_series_id(series_id: object) -> str:
    """``series_id`` simple: forma parte de rutas y de la consulta de lectura."""
    if not isinstance(series_id, str) or not _SERIES_RE.match(series_id):
        raise InvalidSeriesIdError(
            f"'series_id' tiene que ser un nombre simple (letras, digitos, '^', '.', '_', '-'), "
            f"no {series_id!r}"
        )
    return series_id


def _optional_column(matrix: pl.DataFrame, name: str) -> list[object] | None:
    """Columna opcional del contrato ya calculada por el llamante, si la trae."""
    if name not in matrix.columns:
        return None
    return list(matrix.get_column(name).to_list())


def daily_records(
    matrix: pl.DataFrame,
    *,
    spec: FeatureSpec,
    series_id: str,
    fetched_at: datetime,
) -> list[dict[str, object]]:
    """Registros del ``Store`` para una matriz de features, **sin tocar el disco**.

    Es la funcion que define que se escribe, y por eso es la que se puede
    inspeccionar en un test: **no** incluye ``version`` (lo posee el almacen) y
    **no** inventa ``fetched_at`` (lo pasa el llamante; el modulo no lee el
    reloj).

    La matriz tiene que traer ``session`` y ``as_of`` (el cierre de sesion en
    UTC), todas las columnas del catalogo **de su familia** y, opcionalmente,
    ``features_version`` y/o ``feature_spec_sha256`` ya calculados: si vienen, se
    **verifican** contra el recalculo y una discrepancia es un error tipado (A7).
    Cualquier otra columna es un error (A10): la matriz persistida es exactamente
    ``session``+``as_of``+features. El ``source`` del registro sale de la familia
    de la spec, no del llamante: es lo que impide que una familia sustituya a la
    otra en el mismo dataset.

    Returns
    -------
    list[dict[str, object]]
        Un registro por sesion, ordenado por sesion.
    """
    _validate_spec(spec)
    prepared_series = _require_series_id(series_id)
    fetched = _as_utc(fetched_at, field="fetched_at", error=InvalidFeatureMatrixError)

    source = _source_for(spec.feature_set)
    feature_columns = _feature_columns_for(spec.feature_set)
    required = [*IDENTITY_COLUMNS, *feature_columns]
    missing = [name for name in required if name not in matrix.columns]
    if missing:
        raise InvalidFeatureMatrixError(f"faltan columnas en la matriz: {missing}")
    known = {*required, *VERSION_COLUMNS}
    unknown = sorted(set(matrix.columns) - known)
    if unknown:
        raise InvalidFeatureMatrixError(
            f"columnas fuera del catalogo: {unknown}. Toda columna persistida tiene que estar "
            f"en el catalogo de '{spec.feature_set}'"
        )

    sessions = _session_dates(matrix)
    if len(set(sessions)) != len(sessions):
        repeated = sorted({session for session in sessions if sessions.count(session) > 1})
        raise InvalidFeatureMatrixError(f"la matriz repite estas sesiones: {repeated}")
    instants = _instants(matrix, field="as_of")

    features = {
        name: _column_values(matrix.get_column(name), column=name) for name in feature_columns
    }
    supplied_version = _optional_column(matrix, "features_version")
    supplied_spec = _optional_column(matrix, "feature_spec_sha256")

    digest = feature_spec_sha256(spec)
    if supplied_spec is not None and any(value != digest for value in supplied_spec):
        raise InvalidFeatureMatrixError(
            "la matriz trae un 'feature_spec_sha256' que no corresponde a su spec: el contrato "
            "de calculo cambia, no se sobreescribe en silencio"
        )

    records: list[dict[str, object]] = []
    for index, session in enumerate(sessions):
        instant = instants[index]
        if instant.date() != session:
            raise InvalidFeatureMatrixError(
                f"la sesion {session.isoformat()} y su 'as_of' ({instant.isoformat()}) no "
                "corresponden al mismo dia: el 'as_of' de la fila es exactamente el que entra "
                "en el hash"
            )
        if fetched < instant:
            # El propio Store lo rechazaria, pero aqui el error dice *que fila* falla.
            raise InvalidFeatureMatrixError(
                f"'fetched_at' ({fetched.isoformat()}) es anterior al 'as_of' de la sesion "
                f"{session.isoformat()} ({instant.isoformat()})"
            )
        version = features_version(spec, instant)
        if supplied_version is not None and supplied_version[index] != version:
            raise InvalidFeatureMatrixError(
                f"la matriz trae un 'features_version' que no corresponde a su 'as_of' "
                f"({supplied_version[index]!r} != {version!r}): el 'as_of' de la fila tiene que "
                "ser exactamente el que entra en el hash"
            )
        record: dict[str, object] = {
            "source": source,
            "series_id": prepared_series,
            "as_of": instant,
            "fetched_at": fetched,
            "published_at": None,
            "features_version": version,
            "feature_spec_sha256": digest,
        }
        for name, values in features.items():
            record[name] = values[index]
        records.append(record)

    records.sort(key=lambda record: cast("datetime", record["as_of"]))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Persistencia
# ─────────────────────────────────────────────────────────────────────────────
def save_daily(
    store: Store,
    *,
    spec: FeatureSpec,
    matrix: pl.DataFrame,
    series_id: str,
    fetched_at: datetime,
) -> WriteOutcome:
    """Persiste la matriz en ``derived.features_daily`` y devuelve que paso.

    La escritura va **siempre** por ``Store.replace``: una feature es un dato
    *derived* recalculable, asi que una revision nueva sustituye a la vigente
    (dejando la anterior en disco) en vez de chocar con la inmutabilidad de
    ``raw``. Contenido identico ⇒ ``WriteOutcome.UNCHANGED`` y ningun Parquet
    nuevo; contenido distinto (por ejemplo, tras subir ``code_version``) ⇒
    ``version = max + 1``.

    Parameters
    ----------
    store:
        Almacen donde escribir. La raiz la decide quien llama: este modulo no
        conoce ninguna ruta concreta.
    spec, matrix, series_id, fetched_at:
        Ver :func:`daily_records`.
    """
    records = daily_records(matrix, spec=spec, series_id=series_id, fetched_at=fetched_at)
    return store.replace(FEATURES_LAYER, FEATURES_DATASET, records)


def load_daily(
    store: Store, *, series_id: str, feature_set: str = VOLATILITY_FEATURE_SET
) -> pl.DataFrame:
    """Revision **vigente** de una familia de features de una serie, por sesion.

    Lee con ``store.sql()`` a proposito: las vistas del almacen exponen una sola
    fila por identidad (la de mayor ``version``), que es justo lo que se quiere
    comparar contra una recomputacion. ``read_pit`` responde otra pregunta.

    ``feature_set`` filtra por el ``source`` de la familia (decision 2 de #20):
    las dos familias comparten dataset y hay que poder leer una sin la otra. La
    familia pedida **tiene** que tener filas: devolver un frame vacio por un nombre
    mal escrito seria el fallo silencioso que este almacen no admite.
    """
    prepared = _require_series_id(series_id)
    source = _source_for(feature_set)
    if FEATURES_DATASET not in store.datasets(FEATURES_LAYER):
        raise UnknownDatasetError(
            f"'{FEATURES_LAYER}.{FEATURES_DATASET}' no tiene ningun Parquet en {store.root}: "
            "no hay matriz de features que leer"
        )
    literal = prepared.replace("'", "''")
    source_literal = source.replace("'", "''")
    query = (
        f"SELECT * FROM {FEATURES_LAYER}.{FEATURES_DATASET} "  # noqa: S608 - nombre fijo + serie validada
        f"WHERE series_id = '{literal}' AND source = '{source_literal}' ORDER BY as_of"
    )
    frame = store.sql(query)
    if frame.height == 0:
        raise UnknownDatasetError(
            f"'{FEATURES_LAYER}.{FEATURES_DATASET}' no tiene filas de la familia "
            f"'{feature_set}' (source='{source}') para la serie '{prepared}'"
        )
    return frame
