"""Almacén *point-in-time* sobre Parquet, con DuckDB como motor de consulta.

Este módulo es el **único punto de acceso** al almacén de datos del proyecto
(`_docs/plan.md` §8.2, §8.4 · `_docs/tech_stack.md` §4.3, §12.4, §12.7, §12.9).
Su contrato está escrito aquí dentro porque el código es lo que se ejecuta: la
issue que lo pidió (`#2`) y el *grooming* que lo cerró no son el artefacto.

Contrato de columnas obligatorias
---------------------------------

Todo registro, en `raw` y en `derived`, tiene estas seis columnas:

===============  ===========================  =========  ===========================================
Columna          Tipo                         Obligat.   Regla
===============  ===========================  =========  ===========================================
``source``       ``str`` no vacío             sí         Quién produjo el dato. En `raw`, el
                                                         adaptador (``yfinance``, ``fred``); en
                                                         `derived`, el módulo que lo calculó
                                                         (``features.technical``).
``series_id``    ``str`` no vacío             sí         Dimensión de entidad dentro del dataset
                                                         (``^GSPC``, ``CPIAUCSL``, ``XLK``). Si el
                                                         dataset no tiene entidad, se usa el nombre
                                                         del dataset.
``as_of``        ``date`` **o** ``datetime``  sí         Aquello **a lo que se refiere** el dato.
                 UTC                                     Nunca ``datetime`` sin zona. Una barra
                                                         diaria se ancla al cierre de sesión
                                                         (16:00 ET) en UTC, **no** a medianoche.
``fetched_at``   ``datetime`` UTC             sí         Cuándo lo obtuvimos. Invariante:
                                                         ``fetched_at >= as_of``.
``published_at`` ``datetime`` UTC | ``NULL``  opcional   Cuándo lo publicó la fuente. **Prohibido**
                                                         rellenarlo con ``fetched_at`` si la fuente
                                                         no lo da. Invariante:
                                                         ``published_at <= fetched_at``.
``version``      ``int >= 1``                 sí         **Contador de revisión del dato** para esa
                                                         identidad. 1 = primer valor visto; 2, 3…
                                                         = revisiones posteriores. **No** es la
                                                         versión del esquema (#49).
===============  ===========================  =========  ===========================================

Cualquier columna adicional es *payload* del dataset y la declara la tarea que
lo ingesta o lo calcula: el esquema concreto de cada dataset (``market_daily``,
``macro``, ``features_daily``…) **no** forma parte de este contrato.

**Identidad de un registro:** ``(dataset, source, series_id, as_of)``, dentro de
una capa. Un ``dataset`` es el nombre de la tabla documentada (``market_daily``,
``macro``…); vive como directorio, no como columna. La capa (``raw`` o
``derived``) se pasa como argumento aparte y no forma parte del nombre.

``dataset`` y ``source`` acaban en rutas y en nombres de vista, así que se
exigen nombres simples: ``dataset`` empieza por letra y solo usa letras, dígitos
o ``_``; ``source`` admite además ``.`` y ``-`` (``yfinance``, ``ecb_sdw``).
Cualquier otro valor se rechaza con ``InvalidRecordError`` antes de tocar el
disco.

Layout físico
-------------

``<raíz>/<capa>/<dataset>/source=<source>/year=<AAAA de as_of>/*.parquet``

- Parquet comprimido en **ZSTD**, una partición por ``source`` y por **año del
  ``as_of``** (nunca del ``fetched_at``).
- **Ninguna escritura modifica un fichero existente**: cada escritura crea
  ficheros nuevos. Es lo que hace que ``raw`` sea inmutable de verdad.
- DuckDB es **solo motor de consulta** y lee los Parquet directamente. No se
  persiste ningún catálogo ``.duckdb`` con copia del dato: el Parquet es la
  única fuente de verdad.
- La raíz es un **parámetro explícito** del objeto: ninguna constante del módulo
  escribe en ``data/``. El cableado a la configuración real es de la tarea que
  necesite el almacén por primera vez.

Semántica de escritura
----------------------

- ``append(...)``: identidad nueva ⇒ ``version = 1``. Identidad existente con
  contenido idéntico ⇒ **no-op** (idempotente: un reintento de ingesta no rompe
  el almacén). Identidad existente con contenido **distinto** ⇒
  ``ImmutableWriteError``, nunca sobrescritura silenciosa.
- ``append_revision(...)``: explícito, para cuando la fuente revisa un dato
  (FRED revisa CPI/PCE/NFP). ``version = max + 1``. Si el contenido coincide con
  la última revisión ⇒ no-op.
- ``replace(...)``: **solo ``derived``**, que es recalculable. En ``raw`` falla
  con ``ImmutableWriteError``. Reescribe el valor de la identidad y deja **una
  sola fila por identidad** en el estado consultable: la revisión nueva
  sustituye a la anterior tanto en ``read_pit`` como en las vistas SQL. El valor
  anterior deja de ser visible pero **no se borra del disco** (se recupera con
  ``read_pit`` de un instante pasado). `raw` nunca se sobrescribe.

El **contenido** de un registro es todo salvo ``version`` y ``fetched_at``:
``fetched_at`` es cuándo lo obtuvimos (varía en cada reintento legítimo) y
``version`` lo gestiona el almacén. ``published_at`` sí forma parte del
contenido. La comparación se hace contra la **última versión almacenada** de la
identidad, esté visible o no en este instante.

El almacén es el propietario de ``version``: si un registro trae una, se valida
(entero ``>= 1``) pero **se ignora**; el valor lo calcula el almacén según la
operación. Todos los registros de una misma escritura deben compartir las mismas
columnas: una escritura, un esquema.

Lectura *point-in-time*
-----------------------

``read_pit(capa, dataset, at, series_id=None)`` devuelve el estado del mundo
**tal y como se conocía en ``at``**:

- Visibilidad: si ``published_at`` está definido, ``published_at <= at``; si es
  ``NULL``, la visibilidad cae a ``fetched_at <= at``.
- ``as_of`` **nunca** decide visibilidad: solo sitúa el dato en el tiempo. Un
  dato referido a 2020 pero publicado hoy no existía ayer.
- Por identidad devuelve la fila de **mayor ``version``** entre las visibles.
- La sesión de DuckDB se fija en UTC, así que los ``datetime`` vuelven en UTC.

``sql(query)`` consulta cualquier dataset con SQL de DuckDB sobre los Parquet.
Para eso registra una vista por dataset en los esquemas ``raw`` y ``derived``,
de modo que ``SELECT * FROM raw.market_daily`` funciona sin escribir rutas.

Las vistas exponen **una sola fila por identidad**: la **revisión vigente**, la
de mayor ``version`` (lo mismo que devuelve ``read_pit``, sin el filtro de
visibilidad temporal). Así, tras un ``replace`` en ``derived`` o un
``append_revision``, el dataset **no** muestra a la vez el valor viejo y el
nuevo. La historia completa sigue en los Parquet y se reconstruye con
``read_pit(at=...)``.

Contrato de *layout* (`tech_stack.md` §12.10)
---------------------------------------------

Las seis columnas obligatorias de arriba son el contrato **común**. El *payload* de
cada dataset lo declara `cfdtrader.data.contracts` —firma por ``(capa, dataset)`` y
``layout_version``— y este módulo la **hace cumplir al leer**:

- **Cambio de tipo** de una columna: :class:`LayoutMismatchError`, que nombra el
  **fichero**, la **columna** y los **dos tipos**. No hay coerción silenciosa.
- **Columna desconocida** (la firma no la declara): también es error. Un cambio de
  semántica exige columna **nueva con nombre nuevo**; nunca se reinterpreta la vieja.
- **Columna ausente**: si el fichero **es** un *layout* declarado más antiguo, la columna se
  materializa a ``null`` **declarado** —nunca ``0``— y el lector publica el *layout* con
  :meth:`Store.layout` y el log. Si al fichero le faltan columnas de la firma **original**,
  no es ningún *layout* declarado: se lee **lo que trae** y el hueco se declara igual, sin
  inventar columnas que nadie escribió.

Un dataset **sin declaración** no se valida: la lista de firmas es una lista blanca, y
un test comprueba que todos los datasets que el proyecto escribe están en ella.

Un fichero de *layout* antiguo de `raw` se **cuarentena** con :meth:`Store.quarantine`
—se **mueve** a ``_legacy/<layout_version>/``, sin reescribirlo, y la vista deja de
verlo— y se re-ingesta. En `derived` la política es recomputar y sustituir con
``replace``; la cuarentena es solo de `raw`.
"""

from __future__ import annotations

import dataclasses
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast, final

import duckdb
import polars as pl
from loguru import logger

from cfdtrader.data.contracts import (
    POLARS_DTYPES,
    REQUIRED_SIGNATURE,
    DatasetLayout,
    layout_of,
)

__all__ = [
    "ImmutableWriteError",
    "InvalidRecordError",
    "Layer",
    "LayoutFile",
    "LayoutMismatchError",
    "LayoutReading",
    "Record",
    "StorageError",
    "Store",
    "UnknownDatasetError",
    "WriteOutcome",
]

#: Capas del almacén. `raw` es inmutable; `derived` es recalculable.
type Layer = Literal["raw", "derived"]

#: Un registro de entrada: columnas obligatorias más el *payload* del dataset.
type Record = Mapping[str, object]

LAYERS: tuple[Layer, ...] = ("raw", "derived")

#: Las seis columnas obligatorias. Ninguna se puede omitir ni reutilizar como payload.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "source",
    "series_id",
    "as_of",
    "fetched_at",
    "published_at",
    "version",
)

#: Columnas cuyo valor define el contenido, además del payload. `fetched_at` y
#: `version` quedan fuera a propósito (ver el docstring del módulo).
_CONTENT_COLUMNS: tuple[str, ...] = ("source", "series_id", "as_of", "published_at")

_RESERVED: frozenset[str] = frozenset(REQUIRED_COLUMNS)

#: Un dataset vive como directorio y como vista `capa.dataset`: nombre simple.
_DATASET_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

#: `source` también forma parte de la ruta (`source=<source>`): sin separadores.
_SOURCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_PARQUET_SUFFIX = ".parquet"

#: Directorio de cuarentena de los ficheros de *layout* antiguo de `raw` (§12.10). Vive
#: **dentro** del dataset, así que no aparece como dataset, y queda fuera de las vistas.
_LEGACY_DIR = "_legacy"

#: Tipo de DuckDB con el que se materializa a ``NULL`` una columna declarada que
#: **ningún** fichero trae: el lector declara el tipo, no lo adivina.
_DUCKDB_TYPES: dict[str, str] = {
    "str": "VARCHAR",
    "int": "BIGINT",
    "float": "DOUBLE",
    "bool": "BOOLEAN",
    "date": "DATE",
    "datetime": "TIMESTAMP",
}

#: Tipo de Polars con el que se **escribe** cada tipo declarado. Se usa cuando un lote trae
#: la columna entera a ``NULL``: el Parquet de un dataset declarado lleva su tipo declarado.
_POLARS_TYPES: dict[str, pl.DataType] = {
    "str": pl.String(),
    "int": pl.Int64(),
    "float": pl.Float64(),
    "bool": pl.Boolean(),
    "date": pl.Date(),
    "datetime": pl.Datetime("us", "UTC"),
}

#: Expresión de ventana que define la **revisión vigente** de una identidad: la
#: de mayor ``version``. La usan la lectura *point-in-time* (que además filtra
#: por visibilidad) y las vistas SQL (que exponen el estado actual).
_CURRENT_ROW_WINDOW = (
    "row_number() OVER (PARTITION BY source, series_id, as_of ORDER BY version DESC)"
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores
# ─────────────────────────────────────────────────────────────────────────────
class StorageError(Exception):
    """Base de todos los errores del almacén."""


class InvalidRecordError(StorageError):
    """El registro no cumple el contrato: columna ausente, vacía o incoherente."""


class ImmutableWriteError(StorageError):
    """Intento de sobrescribir contenido ya almacenado con otro distinto."""


class UnknownDatasetError(StorageError):
    """Lectura de un dataset que todavía no tiene ningún fichero Parquet."""


class LayoutMismatchError(StorageError):
    """La firma de un Parquet no es la declarada para su dataset (`tech_stack.md` §12.10).

    Lleva el **fichero**, la **columna** y los **dos tipos** como atributos, para que quien
    la capture no tenga que leer el mensaje. Se levanta en los dos casos en que leer sería
    mentir: una columna con **otro tipo** y una columna que la firma **no declara**.
    """

    def __init__(
        self,
        *,
        path: Path,
        column: str,
        expected: str | None,
        found: str | None,
        reason: str,
    ) -> None:
        self.path = path
        self.column = column
        self.expected = expected
        self.found = found
        self.reason = reason
        super().__init__(
            f"{path}: la columna {column!r} no cumple la firma declarada. {reason} "
            f"(tipo en el fichero: {found or 'ausente'}; tipo declarado: {expected or 'ninguno'})"
        )


@dataclass(frozen=True, slots=True)
class LayoutFile:
    """Firma de **un** Parquet del dataset: su *layout*, sus columnas y las que no trae."""

    path: Path
    layout_version: int
    columns: tuple[str, ...]
    """Columnas del fichero, obligatorias incluidas."""

    missing: tuple[str, ...]
    """Columnas de la firma vigente que este fichero no trae: el lector las devuelve a ``null``."""


@dataclass(frozen=True, slots=True)
class LayoutReading:
    """Lo que el lector vio en un dataset: una versión por fichero y qué sale a ``null``.

    ``old_layout`` y ``null_columns`` son la **declaración** que §12.10 exige antes de
    responder a una consulta que mezcla dos *layouts*: un histórico mezclado se puede leer,
    pero no en silencio.
    """

    layer: str
    dataset: str
    declared_version: int
    files: tuple[LayoutFile, ...]

    @property
    def versions(self) -> tuple[int, ...]:
        """Versiones de *layout* presentes, de menor a mayor."""
        return tuple(sorted({file.layout_version for file in self.files}))

    @property
    def old_layout(self) -> bool:
        """``True`` si algún fichero es anterior al *layout* vigente (histórico mezclado)."""
        return any(file.layout_version < self.declared_version for file in self.files)

    @property
    def null_columns(self) -> tuple[str, ...]:
        """Columnas que el lector devuelve a ``null`` por venir de un *layout* anterior."""
        return tuple(sorted({name for file in self.files for name in file.missing}))

    @property
    def present_columns(self) -> frozenset[str]:
        """Columnas que trae **algún** fichero del dataset (la unión de todas)."""
        return frozenset(name for file in self.files for name in file.columns)

    def describe(self) -> str:
        """Frase con la que el lector **publica** el *layout* leído."""
        head = (
            f"'{self.layer}.{self.dataset}': layout vigente {self.declared_version}, "
            f"ficheros en {list(self.versions)}"
        )
        if not self.old_layout:
            return f"{head}; ninguno anterior al vigente"
        return (
            f"{head}; **histórico mezclado**: las columnas "
            f"{list(self.null_columns)} salen a null en los ficheros anteriores"
        )


class WriteOutcome(StrEnum):
    """Resultado de una escritura, para que quien ingesta pueda trazar qué pasó."""

    CREATED = "created"
    """Se escribieron filas nuevas."""

    UNCHANGED = "unchanged"
    """El contenido ya estaba almacenado: no se escribió nada."""


# ─────────────────────────────────────────────────────────────────────────────
# Normalización de registros
# ─────────────────────────────────────────────────────────────────────────────
def _canonical(value: object) -> str:
    """Forma canónica y estable de un valor, para comparar contenidos.

    Se usa ``str``/``repr`` en lugar de ``==`` para que dos lecturas del mismo
    dato coincidan aunque cambie el tipo contenedor (``NaN``, escalares de
    Polars frente a tipos de Python…).
    """
    if value is None:
        return "null"
    if isinstance(value, datetime):
        return f"datetime:{value.astimezone(UTC).isoformat()}"
    if isinstance(value, date):
        return f"date:{value.isoformat()}"
    if isinstance(value, bool):
        return f"bool:{value}"
    if isinstance(value, (int, float, str)):
        return f"{type(value).__name__}:{value}"
    return f"{type(value).__name__}:{value!r}"


def _identity_key(source: str, series_id: str, as_of: date | datetime) -> tuple[str, str, str]:
    """Clave de identidad comparable entre un registro y una fila releída del almacén.

    Se usa la forma canónica de ``as_of`` en lugar del valor: el mismo instante
    vuelve del almacén como ``date`` (series macro) o como ``datetime`` con zona
    UTC (barras), y comparar objetos de distinto tipo no sirve.
    """
    return (source, series_id, _canonical(as_of))


def _content_key(values: Mapping[str, object]) -> tuple[str, ...]:
    """Huella del contenido: todo menos ``version`` y ``fetched_at``."""
    parts = [f"{name}={_canonical(values.get(name))}" for name in _CONTENT_COLUMNS]
    parts.extend(
        f"{name}={_canonical(values[name])}"
        for name in sorted(k for k in values if k not in _RESERVED)
    )
    return tuple(parts)


@dataclass(frozen=True, slots=True)
class PreparedRecord:
    """Registro validado y normalizado, listo para escribirse."""

    layer: Layer
    dataset: str
    source: str
    series_id: str
    as_of: date | datetime
    fetched_at: datetime
    published_at: datetime | None
    version: int
    payload: tuple[tuple[str, object], ...]
    """Columnas propias del dataset, ordenadas por nombre para ser deterministas."""

    def as_values(self) -> dict[str, object]:
        """Todas las columnas del registro, en orden de escritura."""
        values: dict[str, object] = {
            "source": self.source,
            "series_id": self.series_id,
            "as_of": self.as_of,
            "fetched_at": self.fetched_at,
            "published_at": self.published_at,
            "version": self.version,
        }
        values.update(self.payload)
        return values

    def content_key(self) -> tuple[str, ...]:
        """Contenido del registro, comparable con una fila releída del almacén."""
        return _content_key(self.as_values())

    @property
    def year(self) -> int:
        """Año de ``as_of`` en UTC: la partición depende de a qué se refiere el dato."""
        return self.as_of.year


def _as_utc(value: object, *, field: str) -> datetime:
    """Convierte a UTC exigiendo zona horaria explícita."""
    if not isinstance(value, datetime):
        raise InvalidRecordError(
            f"'{field}' debe ser un datetime, no {type(value).__name__} (registro: {field})"
        )
    if value.tzinfo is None or value.utcoffset() is None:
        raise InvalidRecordError(
            f"'{field}' no tiene zona horaria: el almacén guarda todo en UTC y no adivina offsets"
        )
    return value.astimezone(UTC)


def _as_instant(value: object, *, field: str) -> datetime:
    """Instante UTC a partir de un ``datetime`` con zona o de un ``date`` (00:00 UTC)."""
    if isinstance(value, datetime):
        return _as_utc(value, field=field)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=UTC)
    raise InvalidRecordError(
        f"'{field}' debe ser date o datetime, no {type(value).__name__} (registro: {field})"
    )


def _require_text(values: Mapping[str, object], *, field: str) -> str:
    """Columna de texto obligatoria y no vacía."""
    value = values.get(field)
    if not isinstance(value, str):
        raise InvalidRecordError(
            f"'{field}' es obligatoria y debe ser str, no {type(value).__name__} "
            f"(registro: {field})"
        )
    if not value.strip():
        raise InvalidRecordError(f"'{field}' no puede estar vacía (registro: {field})")
    return value


def _require_as_of(values: Mapping[str, object]) -> date | datetime:
    """``as_of`` obligatorio: ``date`` para series, ``datetime`` UTC para barras."""
    if "as_of" not in values:
        raise InvalidRecordError("'as_of' es obligatoria (registro: as_of)")
    value = values["as_of"]
    if isinstance(value, datetime):
        return _as_utc(value, field="as_of")
    if isinstance(value, date):
        return value
    raise InvalidRecordError(
        f"'as_of' debe ser date o datetime, no {type(value).__name__} (registro: as_of)"
    )


def _optional_utc(values: Mapping[str, object], *, field: str) -> datetime | None:
    """Columna opcional de instante UTC. Ausente o ``None`` ⇒ ``NULL``, nunca relleno."""
    value = values.get(field)
    if value is None:
        return None
    return _as_utc(value, field=field)


def _require_version(values: Mapping[str, object]) -> int:
    """``version`` la gestiona el almacén; si el registro trae una, se valida."""
    value = values.get("version")
    if value is None:
        return 1
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidRecordError(
            f"'version' debe ser un entero >= 1, no {type(value).__name__} (registro: version)"
        )
    if value < 1:
        raise InvalidRecordError(f"'version' debe ser >= 1, no {value} (registro: version)")
    return value


def _prepare(layer: Layer, dataset: str, record: Record) -> PreparedRecord:
    """Valida un registro contra el contrato y lo devuelve normalizado."""
    values = dict(record)
    source = _require_text(values, field="source")
    if not _SOURCE_RE.match(source):
        raise InvalidRecordError(
            f"'source' contiene caracteres no admitidos: {source!r} "
            "(solo letras, dígitos, '.', '_' y '-')"
        )
    series_id = _require_text(values, field="series_id")
    as_of = _require_as_of(values)
    fetched_at = _as_utc(values.get("fetched_at"), field="fetched_at")
    published_at = _optional_utc(values, field="published_at")
    version = _require_version(values)

    if published_at is not None and published_at > fetched_at:
        raise InvalidRecordError(
            f"'published_at' ({published_at.isoformat()}) no puede ser posterior a "
            f"'fetched_at' ({fetched_at.isoformat()}) (registro: published_at)"
        )
    if fetched_at < _as_instant(as_of, field="as_of"):
        raise InvalidRecordError(
            f"'fetched_at' ({fetched_at.isoformat()}) no puede ser anterior a "
            f"'as_of' ({as_of.isoformat()}) (registro: fetched_at)"
        )

    payload = tuple(sorted((key, value) for key, value in values.items() if key not in _RESERVED))
    return PreparedRecord(
        layer=layer,
        dataset=dataset,
        source=source,
        series_id=series_id,
        as_of=as_of,
        fetched_at=fetched_at,
        published_at=published_at,
        version=version,
        payload=payload,
    )


def _as_of_is_date(record: PreparedRecord) -> bool:
    """``True`` si el ``as_of`` del registro es un ``date`` puro (serie macro)."""
    return not isinstance(record.as_of, datetime)


def _file_dtypes(path: Path) -> dict[str, str]:
    """Tipo **declarado** de cada columna de un Parquet, en el vocabulario de `contracts`."""
    declared: dict[str, str] = {}
    for name, dtype in pl.read_parquet_schema(path).items():
        base = str(dtype).split("(")[0]
        mapped = POLARS_DTYPES.get(base)
        if mapped is None:
            raise StorageError(
                f"{path}: el lector no sabe comparar el tipo {base!r} de la columna "
                f"{str(name)!r} con la firma declarada (conoce {sorted(POLARS_DTYPES)})"
            )
        declared[str(name)] = mapped
    return declared


def _inspect_file(path: Path, declaration: DatasetLayout, dtypes: Mapping[str, str]) -> LayoutFile:
    """*Layout* de un fichero y las columnas que no trae, o :class:`LayoutMismatchError`.

    Es el único sitio donde se decide si un fichero **se puede leer**: un tipo distinto o
    una columna que la firma no declara son errores (se leería un dato que no es el
    declarado); una columna **ausente** no lo es —se declara como ``null``—.
    """
    declared = {spec.name: spec for spec in declaration.payload}
    for name, dtype in dtypes.items():
        if name in REQUIRED_SIGNATURE:
            continue
        spec = declared.get(name)
        if spec is None:
            raise LayoutMismatchError(
                path=path,
                column=name,
                expected=None,
                found=dtype,
                reason=(
                    f"'{declaration.layer}.{declaration.dataset}' no declara esa columna: un "
                    "cambio de semantica exige columna nueva con nombre nuevo, no reinterpretar "
                    "la vieja"
                ),
            )
        if spec.dtype != dtype:
            raise LayoutMismatchError(
                path=path,
                column=name,
                expected=spec.dtype,
                found=dtype,
                reason=f"'{declaration.layer}.{declaration.dataset}' la declara de otro tipo",
            )
    for name, allowed in REQUIRED_SIGNATURE.items():
        expected = " o ".join(sorted(allowed))
        if name not in dtypes:
            raise LayoutMismatchError(
                path=path,
                column=name,
                expected=expected,
                found=None,
                reason="faltan columnas obligatorias del contrato del almacen",
            )
        if dtypes[name] not in allowed:
            raise LayoutMismatchError(
                path=path,
                column=name,
                expected=expected,
                found=dtypes[name],
                reason="las seis columnas obligatorias tienen un tipo declarado",
            )
    payload = frozenset(name for name in dtypes if name not in REQUIRED_SIGNATURE)
    version = declaration.version_of(payload)
    return LayoutFile(
        path=path,
        layout_version=version,
        columns=tuple(dtypes),
        missing=tuple(spec.name for spec in declaration.payload if spec.name not in dtypes),
    )


def _as_records(record: Record | Sequence[Record]) -> Sequence[object]:
    """Normaliza el argumento de escritura a una secuencia de registros sin tipar.

    Devuelve ``object`` a propósito: los tests y los adaptadores llaman desde
    Python sin comprobación de tipos, y un elemento que no sea un mapping tiene
    que dar un error del almacén, no un ``TypeError``.
    """
    if isinstance(record, Mapping):
        return (record,)
    return list(record)


# ─────────────────────────────────────────────────────────────────────────────
# Almacén
# ─────────────────────────────────────────────────────────────────────────────
@final
class Store:
    """Acceso al almacén *point-in-time* descrito en el docstring del módulo.

    Parameters
    ----------
    root:
        Raíz del almacén (``data/`` en producción). Es un parámetro explícito:
        construir el objeto **no** toca el disco y ninguna constante del módulo
        escribe en ninguna ruta concreta.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        #: Caché de firmas leídas por ``(capa, dataset)``, con la huella de los ficheros para
        #: invalidarla. Inspeccionar el esquema de cada Parquet en **cada** consulta sería un
        #: coste absurdo en un informe que hace decenas: la huella (ruta, tamaño, ``mtime``) es
        #: barata y cambia en cuanto alguien escribe.
        self._layouts: dict[
            tuple[str, str], tuple[tuple[tuple[str, int, int], ...], LayoutReading | None]
        ] = {}

    @property
    def root(self) -> Path:
        """Raíz del almacén."""
        return self._root

    # ── Escritura ────────────────────────────────────────────────────────────
    def append(self, layer: Layer, dataset: str, record: Record | Sequence[Record]) -> WriteOutcome:
        """Escribe registros nuevos. Falla si la identidad ya existe con otro contenido."""
        return self._write(layer, dataset, record, mode="append")

    def append_revision(
        self, layer: Layer, dataset: str, record: Record | Sequence[Record]
    ) -> WriteOutcome:
        """Escribe una revisión de la fuente sobre una identidad existente
        (``version = max + 1``)."""
        return self._write(layer, dataset, record, mode="revision")

    def replace(
        self, layer: Layer, dataset: str, record: Record | Sequence[Record]
    ) -> WriteOutcome:
        """Recalcula un valor de ``derived``. En ``raw`` siempre falla."""
        return self._write(layer, dataset, record, mode="replace")

    # ── Lectura ──────────────────────────────────────────────────────────────
    def read_pit(
        self,
        layer: Layer,
        dataset: str,
        at: date | datetime,
        series_id: str | None = None,
    ) -> pl.DataFrame:
        """Estado del almacén tal y como se conocía en ``at`` (ver el módulo).

        ``at`` como ``date`` significa las 00:00 UTC de ese día. Un dataset sin
        ningún Parquet levanta ``UnknownDatasetError``: una lectura vacía por un
        nombre mal escrito es un fallo silencioso que este proyecto no se puede
        permitir.
        """
        instant = _as_instant(at, field="at")
        reading = self._declare_layout(layer, dataset)
        paths = self._parquet_files(layer, dataset)
        if not paths:
            raise UnknownDatasetError(
                f"el dataset '{layer}.{dataset}' no tiene ningún fichero Parquet en {self._root}"
            )

        conditions = [
            "((published_at IS NOT NULL AND published_at <= ?) "
            "OR (published_at IS NULL AND fetched_at <= ?))",
        ]
        params: list[object] = [instant, instant]
        if series_id is not None:
            conditions.append("series_id = ?")
            params.append(series_id)

        select = "*" if reading is None else self._layout_projection(layer, dataset, reading)
        query = (
            f"WITH visible AS (SELECT {select} FROM "
            f"{_read_parquet_expr(paths)} WHERE {' AND '.join(conditions)}), "
            f"ranked AS (SELECT *, {_CURRENT_ROW_WINDOW} AS pit_rank FROM visible) "
            "SELECT * EXCLUDE (pit_rank) FROM ranked WHERE pit_rank = 1 "
            "ORDER BY series_id, as_of"
        )
        return self._fetch(query, params)

    def sql(self, query: str) -> pl.DataFrame:
        """Ejecuta SQL de DuckDB sobre los Parquet, con una vista por dataset registrada."""
        return self._fetch(query, [])

    def datasets(self, layer: Layer) -> list[str]:
        """Datasets con directorio en esa capa, en orden alfabético."""
        self._validate_layer(layer)
        layer_dir = self._layer_dir(layer)
        if not layer_dir.is_dir():
            return []
        return sorted(path.name for path in layer_dir.iterdir() if path.is_dir())

    # ── Contrato de *layout* (§12.10) ────────────────────────────────────────
    def layout(self, layer: Layer, dataset: str) -> LayoutReading | None:
        """Firma leída de **cada** Parquet del dataset, con su *layout* y sus ``null``.

        Devuelve ``None`` si el dataset no tiene declaración: §12.10 solo se puede hacer
        cumplir sobre una firma declarada, y el test que comprueba que *todos* los datasets
        del proyecto están declarados es lo que cierra ese hueco.

        Levanta :class:`LayoutMismatchError` en cuanto un fichero cambia el tipo de una
        columna o trae una que la firma no declara. Que a un fichero le falten columnas
        **no** es un error: queda declarado en ``old_layout``/``null_columns``, y si es un
        *layout* declarado más antiguo, las aditivas que no trae se materializan a ``null``
        (:meth:`read_pit`).
        """
        declaration = layout_of(layer, dataset)
        paths = self._parquet_files(layer, dataset)
        fingerprint = tuple(
            (str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths
        )
        cached = self._layouts.get((layer, dataset))
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
        reading = (
            None
            if declaration is None
            else LayoutReading(
                layer=layer,
                dataset=dataset,
                declared_version=declaration.layout_version,
                files=tuple(_inspect_file(path, declaration, _file_dtypes(path)) for path in paths),
            )
        )
        self._layouts[(layer, dataset)] = (fingerprint, reading)
        return reading

    def quarantine(self, layer: Layer, dataset: str) -> tuple[Path, ...]:
        """Mueve los ficheros de *layout* antiguo de `raw` a ``_legacy/<version>/`` (§12.10).

        El fichero se **mueve** (``os.replace`` dentro de la misma raíz), nunca se
        reescribe: los bytes son los mismos antes y después, que es lo que exige la
        inmutabilidad de `raw` (§12.4). Al quedar bajo ``_legacy/`` sale de las vistas, así
        que deja de leerse, y **la re-ingesta es de quien llama**: esto no inventa datos ni
        decide qué *layout* es el vigente —eso lo dice la declaración—. Nada de ``_legacy/``
        se borra aquí: la poda es #44 y la decide el propietario.

        Devuelve los ficheros movidos (vacío si no había ninguno anterior al vigente).
        """
        if layer != "raw":
            raise StorageError(
                f"'{layer}.{dataset}': la cuarentena es solo de 'raw', que es lo inmutable. "
                "En 'derived' §12.10 pide recomputar el dataset y sustituirlo con 'replace' "
                "(el dato no es irreversible), y borrar los ficheros viejos es una poda (#44)"
            )
        reading = self.layout(layer, dataset)
        if reading is None:
            raise StorageError(
                f"'{layer}.{dataset}' no tiene firma declarada: sin declaracion no se puede "
                "saber que ficheros son de un layout anterior (§12.10)"
            )
        moved: list[Path] = []
        root = self._dataset_dir(layer, dataset)
        for file in reading.files:
            if file.layout_version >= reading.declared_version:
                continue
            target = root / _LEGACY_DIR / str(file.layout_version) / file.path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(file.path, target)
            moved.append(target)
        if moved:
            logger.warning(
                "contrato del almacen: cuarentenados {} ficheros de '{}.{}' en '{}'",
                len(moved),
                layer,
                dataset,
                _LEGACY_DIR,
            )
        return tuple(moved)

    def _declare_layout(self, layer: Layer, dataset: str) -> LayoutReading | None:
        """Lee la firma del dataset y **publica** el *layout* si mezcla dos versiones (§12.10).

        Devuelve la lectura para que quien la pida no tenga que volver a inspeccionar los
        ficheros (`read_pit` la usa para materializar las columnas que falten).
        """
        reading = self.layout(layer, dataset)
        if reading is not None and reading.old_layout:
            logger.warning("contrato del almacen: {}", reading.describe())
        return reading

    def _layout_projection(self, layer: Layer, dataset: str, reading: LayoutReading) -> str:
        """Lista de columnas de la lectura, con las que **ningún** fichero trae creadas a ``null``.

        Sin esto, un dataset cuyos ficheros son **todos** anteriores al *layout* vigente
        devolvería una tabla **sin** esas columnas (el ``null`` implícito de
        ``union_by_name`` solo aparece cuando algún fichero sí las trae). Se materializan
        como ``NULL`` **declarado**: nunca ``0``, nunca un valor inventado.
        """
        declaration = layout_of(layer, dataset)
        if declaration is None:
            return "*"
        if any(file.layout_version == 0 for file in reading.files):
            # A algún fichero le faltan columnas de la firma **original**: no es ningún layout
            # declarado —el lector ya lo publica en `null_columns`— así que se lee **lo que
            # trae**. Inventarle las columnas que nadie escribió sería peor que declarar el
            # hueco, y el `null` implícito de `union_by_name` ya cubre lo que sí falta en un
            # histórico mezclado.
            return "*"
        present = reading.present_columns
        absent = [spec for spec in declaration.payload if spec.name not in present]
        if not absent:
            return "*"
        columns = [f'"{name}"' for name in REQUIRED_SIGNATURE]
        for spec in declaration.payload:
            if spec.name in present:
                columns.append(f'"{spec.name}"')
            else:
                columns.append(f'CAST(NULL AS {_DUCKDB_TYPES[spec.dtype]}) AS "{spec.name}"')
        return ", ".join(columns)

    # ── Escritura: preparación ───────────────────────────────────────────────
    def _write(
        self,
        layer: Layer,
        dataset: str,
        record: Record | Sequence[Record],
        *,
        mode: Literal["append", "revision", "replace"],
    ) -> WriteOutcome:
        self._validate_layer(layer)
        self._validate_dataset(dataset)
        if mode == "replace" and layer != "derived":
            raise ImmutableWriteError(
                f"'{layer}.{dataset}' es inmutable: 'replace' solo se admite en la capa 'derived' "
                "(una revisión de la fuente se escribe con 'append_revision')"
            )

        prepared = self._prepare_batch(layer, dataset, record)
        self._validate_as_of_kind(layer, dataset, prepared)
        stored_by_identity = self._latest_rows(prepared)

        planned: list[PreparedRecord] = []
        for item in prepared:
            identity = _identity_key(item.source, item.series_id, item.as_of)
            stored = stored_by_identity.get(identity)
            if stored is None:
                planned.append(dataclasses.replace(item, version=1))
                continue
            if _content_key(stored) == item.content_key():
                continue
            if mode == "append":
                raise ImmutableWriteError(
                    f"'{layer}.{dataset}' ya tiene contenido distinto para la identidad "
                    f"(source={item.source!r}, series_id={item.series_id!r}, "
                    f"as_of={item.as_of!r}). "
                    "Un dato crudo no se sobrescribe: usa 'append_revision' si la fuente lo revisó."
                )
            planned.append(dataclasses.replace(item, version=_stored_version(stored) + 1))

        if not planned:
            return WriteOutcome.UNCHANGED
        self._write_rows(planned)
        return WriteOutcome.CREATED

    def _prepare_batch(
        self, layer: Layer, dataset: str, record: Record | Sequence[Record]
    ) -> list[PreparedRecord]:
        """Valida todos los registros antes de escribir nada: la escritura es todo o nada."""
        items = _as_records(record)
        if not items:
            raise InvalidRecordError("la escritura no contiene ningún registro")

        prepared: list[PreparedRecord] = []
        for item in items:
            if not isinstance(item, Mapping):
                raise InvalidRecordError(
                    f"cada registro debe ser un mapping de columnas, no {type(item).__name__}"
                )
            prepared.append(_prepare(layer, dataset, cast(Record, item)))

        columns = {tuple(record.as_values()) for record in prepared}
        if len(columns) > 1:
            raise InvalidRecordError(
                "todos los registros de una escritura deben tener las mismas columnas "
                "(una escritura, un esquema)"
            )
        identities = [(record.source, record.series_id, record.as_of) for record in prepared]
        if len(set(identities)) != len(identities):
            raise InvalidRecordError("la escritura repite una misma identidad más de una vez")
        return prepared

    def _validate_as_of_kind(
        self, layer: Layer, dataset: str, prepared: Sequence[PreparedRecord]
    ) -> None:
        """Un dataset mezcla ``date`` y ``datetime`` en ``as_of`` o no lo mezcla, pero no ambas."""
        stored_is_date = self._stored_as_of_is_date(layer, dataset)
        if stored_is_date is None:
            return
        if {_as_of_is_date(record) for record in prepared} != {stored_is_date}:
            stored_kind = "date" if stored_is_date else "datetime UTC"
            raise InvalidRecordError(
                f"'{layer}.{dataset}' ya almacena 'as_of' de tipo {stored_kind}: un dataset usa un "
                "solo tipo (date para series macro, datetime UTC para barras)"
            )

    # ── Escritura: a disco ───────────────────────────────────────────────────
    def _write_rows(self, records: Sequence[PreparedRecord]) -> None:
        """Un fichero nuevo por partición (mismo ``source`` y mismo año de ``as_of``)."""
        groups: dict[tuple[str, int], list[PreparedRecord]] = {}
        for record in records:
            groups.setdefault((record.source, record.year), []).append(record)

        for (source, year), group in groups.items():
            directory = self._partition_dir(group[0].layer, group[0].dataset, source, year)
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"part-{_timestamp()}-{uuid.uuid4().hex[:8]}{_PARQUET_SUFFIX}"
            temporary = target.with_name(target.name + ".tmp")
            try:
                _to_frame(group).write_parquet(temporary, compression="zstd")
                os.replace(temporary, target)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise

    def _latest_rows(
        self, records: Sequence[PreparedRecord]
    ) -> dict[tuple[str, str, str], dict[str, object]]:
        """Última revisión almacenada de cada identidad del lote, sin filtro de visibilidad.

        Comprobar la identidad era el coste dominante de escribir, porque se hacía
        **registro a registro** y cada comprobación abría conexión y volvía a
        registrar la vista de *todos* los datasets del almacén. Medido sobre una
        copia del almacén real (352 ficheros Parquet): ~76-103 ms por fila.
        Escribir una serie de 21 años son ~5.000 filas, es decir horas para añadir
        lo mismo que cabe en un fichero.

        Aquí se agrupa por partición —``source`` y año de ``as_of``, exactamente lo
        que va a escribir `_write_rows`— y cada partición se resuelve con **una**
        consulta que ya devuelve la revisión vigente de cada identidad (la ventana
        de ``version`` máxima, la misma que usan las vistas SQL y `read_pit`).

        Returns
        -------
        dict[tuple[str, str, str], dict[str, object]]
            Identidad canónica → fila vigente. Si una identidad no aparece, es que
            no está almacenada.
        """
        groups: dict[tuple[Layer, str, str, int], list[PreparedRecord]] = {}
        for record in records:
            key = (record.layer, record.dataset, record.source, record.year)
            groups.setdefault(key, []).append(record)

        stored: dict[tuple[str, str, str], dict[str, object]] = {}
        for (layer, dataset, source, year), group in groups.items():
            paths = self._parquet_files(layer, dataset, source=source, year=year)
            if not paths:
                continue
            series = sorted({item.series_id for item in group})
            placeholders = ", ".join("?" for _ in series)
            query = (
                "WITH ranked AS (SELECT *, "
                f"{_CURRENT_ROW_WINDOW} AS pit_rank FROM {_read_parquet_expr(paths)} "
                f"WHERE series_id IN ({placeholders})) "
                "SELECT * EXCLUDE (pit_rank) FROM ranked WHERE pit_rank = 1"
            )
            for row in self._fetch(query, list(series)).to_dicts():
                key = _identity_key(str(row["source"]), str(row["series_id"]), row["as_of"])
                stored[key] = row
        return stored

    def _stored_as_of_is_date(self, layer: Layer, dataset: str) -> bool | None:
        """Tipo de ``as_of`` ya almacenado en el dataset, o ``None`` si está vacío."""
        paths = self._parquet_files(layer, dataset)
        if not paths:
            return None
        schema = pl.read_parquet_schema(paths[0])
        return not isinstance(schema["as_of"], pl.Datetime)

    # ── Rutas ────────────────────────────────────────────────────────────────
    def _layer_dir(self, layer: Layer) -> Path:
        return self._root / layer

    def _dataset_dir(self, layer: Layer, dataset: str) -> Path:
        """Directorio del dataset, validando capa y nombre **también en lecturas**."""
        self._validate_layer(layer)
        self._validate_dataset(dataset)
        return self._layer_dir(layer) / dataset

    def _partition_dir(self, layer: Layer, dataset: str, source: str, year: int) -> Path:
        return self._dataset_dir(layer, dataset) / f"source={source}" / f"year={year}"

    def _parquet_files(
        self,
        layer: Layer,
        dataset: str,
        *,
        source: str | None = None,
        year: int | None = None,
    ) -> list[Path]:
        """Ficheros Parquet del dataset, opcionalmente acotados a una partición.

        Los de ``_legacy/`` **no** cuentan: están cuarentenados, así que no se leen (§12.10).
        """
        base = self._dataset_dir(layer, dataset)
        if source is not None:
            base = base / f"source={source}"
        if year is not None:
            base = base / f"year={year}"
        if not base.is_dir():
            return []
        return sorted(
            path for path in base.rglob(f"*{_PARQUET_SUFFIX}") if _LEGACY_DIR not in path.parts
        )

    def _validate_layer(self, layer: str) -> None:
        if layer not in LAYERS:
            raise InvalidRecordError(
                f"capa desconocida: {layer!r}; se esperaba una de {list(LAYERS)}"
            )

    def _validate_dataset(self, dataset: str) -> None:
        if not _DATASET_RE.match(dataset):
            raise InvalidRecordError(
                f"nombre de dataset no válido: {dataset!r} "
                "(debe empezar por letra y contener solo letras, dígitos o '_')"
            )

    # ── Consulta ─────────────────────────────────────────────────────────────
    def _connect(self) -> duckdb.DuckDBPyConnection:
        """Conexión en memoria, en UTC y con una vista por dataset registrada."""
        connection = duckdb.connect(config={"TimeZone": "UTC"})
        try:
            self._register_views(connection)
        except BaseException:
            connection.close()
            raise
        return connection

    def _register_views(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Expone cada dataset como ``raw.<dataset>`` / ``derived.<dataset>``.

        La vista devuelve el **estado actual**: una sola fila por identidad, la
        de mayor ``version``. Sin ese filtro, un ``replace`` en ``derived`` (o
        un ``append_revision``) dejaría dos filas por identidad y el valor ya
        sustituido seguiría siendo legible por SQL.
        """
        for layer in LAYERS:
            connection.execute(f'CREATE SCHEMA IF NOT EXISTS "{layer}"')
            for dataset in self.datasets(layer):
                paths = self._parquet_files(layer, dataset)
                if not paths:
                    continue
                reading = self._declare_layout(layer, dataset)
                select = (
                    "*" if reading is None else self._layout_projection(layer, dataset, reading)
                )
                connection.execute(
                    f'CREATE OR REPLACE VIEW "{layer}"."{dataset}" AS '
                    "SELECT * EXCLUDE (pit_rank) FROM ("
                    f"SELECT {select}, {_CURRENT_ROW_WINDOW} AS pit_rank "
                    f"FROM {_read_parquet_expr(paths)}) WHERE pit_rank = 1"
                )

    def _fetch(self, query: str, params: Sequence[object]) -> pl.DataFrame:
        """Ejecuta una consulta y devuelve el resultado como DataFrame de Polars."""
        connection = self._connect()
        try:
            ret = connection.execute(query, list(params)) if params else connection.sql(query)
            return ret.pl()
        finally:
            connection.close()


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de escritura
# ─────────────────────────────────────────────────────────────────────────────
def _stored_version(row: Mapping[str, object]) -> int:
    """``version`` de una fila releída del almacén."""
    value = row.get("version")
    if isinstance(value, bool) or not isinstance(value, int):
        raise StorageError(f"el almacén devolvió un 'version' no entero: {value!r}")
    return value


def _timestamp() -> str:
    """Marca temporal UTC para el nombre del fichero (legible al depurar)."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")


def _to_frame(records: Sequence[PreparedRecord]) -> pl.DataFrame:
    """Construye el DataFrame con el esquema explícito del contrato."""
    names = list(records[0].as_values())
    columns = [record.as_values() for record in records]
    data: dict[str, list[object]] = {name: [column[name] for column in columns] for name in names}

    as_of_type: pl.DataType = pl.Date() if _as_of_is_date(records[0]) else pl.Datetime("us", "UTC")
    dtypes: dict[str, pl.DataType] = {
        "source": pl.String(),
        "series_id": pl.String(),
        "as_of": as_of_type,
        "fetched_at": pl.Datetime("us", "UTC"),
        "published_at": pl.Datetime("us", "UTC"),
        "version": pl.Int64(),
    }
    declaration = layout_of(records[0].layer, records[0].dataset)
    declared = {spec.name: spec.dtype for spec in declaration.payload} if declaration else {}
    for name in names:
        if name not in _RESERVED and all(value is None for value in data[name]):
            # Polars inferiría `Null` y Parquet no tiene ese tipo. En un dataset **declarado**
            # se escribe el tipo que declara su firma —si no, el almacén escribiría una firma
            # que su propio lector rechazaría— y en uno sin declarar, String.
            dtypes[name] = _POLARS_TYPES[declared[name]] if name in declared else pl.String()
    return pl.DataFrame(data).with_columns(
        pl.col(name).cast(dtype) for name, dtype in dtypes.items()
    )


def _read_parquet_expr(paths: Sequence[Path]) -> str:
    """Expresión ``read_parquet`` con rutas literales y sin particionado implícito."""
    literals = ", ".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"read_parquet([{literals}], hive_partitioning=false, union_by_name=true)"
