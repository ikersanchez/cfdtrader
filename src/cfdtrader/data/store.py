"""Almacén *point-in-time* sobre Parquet, con DuckDB como motor de consulta.

Este módulo es el **único punto de acceso** al almacén de datos del proyecto
(`_docs/plan.md` §8.2, §8.4 · `_docs/tech_stack.md` §4.3, §12.4, §12.7, §12.9).
Su contrato está escrito aquí dentro porque el código es lo que se ejecuta: la
issue que lo pidió (`#2`) y el *grooming* que lo cerró no son el artefacto.

Contrato de columnas obligatorias
---------------------------------

Todo registro, en `raw` y en `derived`, tiene estas seis columnas:

===============  ===========================  =========  ============================================
Columna          Tipo                         Obligat.   Regla
===============  ===========================  =========  ============================================
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
===============  ===========================  =========  ============================================

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
from typing import Literal, TypeAlias, cast, final

import duckdb
import polars as pl

__all__ = [
    "InvalidRecordError",
    "ImmutableWriteError",
    "Layer",
    "Record",
    "StorageError",
    "Store",
    "UnknownDatasetError",
    "WriteOutcome",
]

#: Capas del almacén. `raw` es inmutable; `derived` es recalculable.
Layer: TypeAlias = Literal["raw", "derived"]

#: Un registro de entrada: columnas obligatorias más el *payload* del dataset.
Record: TypeAlias = Mapping[str, object]

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
        f"{name}={_canonical(values[name])}" for name in sorted(k for k in values if k not in _RESERVED)
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
            f"'{field}' es obligatoria y debe ser str, no {type(value).__name__} (registro: {field})"
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
        """Escribe una revisión de la fuente sobre una identidad existente (``version = max + 1``)."""
        return self._write(layer, dataset, record, mode="revision")

    def replace(self, layer: Layer, dataset: str, record: Record | Sequence[Record]) -> WriteOutcome:
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

        query = (
            "WITH visible AS (SELECT * FROM "
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
                    f"(source={item.source!r}, series_id={item.series_id!r}, as_of={item.as_of!r}). "
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
        """Ficheros Parquet del dataset, opcionalmente acotados a una partición."""
        base = self._dataset_dir(layer, dataset)
        if source is not None:
            base = base / f"source={source}"
        if year is not None:
            base = base / f"year={year}"
        if not base.is_dir():
            return []
        return sorted(base.rglob(f"*{_PARQUET_SUFFIX}"))

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
                connection.execute(
                    f'CREATE OR REPLACE VIEW "{layer}"."{dataset}" AS '
                    "SELECT * EXCLUDE (pit_rank) FROM ("
                    f"SELECT *, {_CURRENT_ROW_WINDOW} AS pit_rank "
                    f"FROM {_read_parquet_expr(paths)}) WHERE pit_rank = 1"
                )

    def _fetch(self, query: str, params: Sequence[object]) -> pl.DataFrame:
        """Ejecuta una consulta y devuelve el resultado como DataFrame de Polars."""
        connection = self._connect()
        try:
            if params:
                ret = connection.execute(query, list(params))
            else:
                ret = connection.sql(query)
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
    data: dict[str, list[object]] = {
        name: [column[name] for column in columns] for name in names
    }

    as_of_type: pl.DataType = pl.Date() if _as_of_is_date(records[0]) else pl.Datetime("us", "UTC")
    dtypes: dict[str, pl.DataType] = {
        "source": pl.String(),
        "series_id": pl.String(),
        "as_of": as_of_type,
        "fetched_at": pl.Datetime("us", "UTC"),
        "published_at": pl.Datetime("us", "UTC"),
        "version": pl.Int64(),
    }
    for name in names:
        if name not in _RESERVED and all(value is None for value in data[name]):
            # Polars inferiría `Null` y Parquet no tiene ese tipo: se fija String y el valor es NULL.
            dtypes[name] = pl.String()
    return pl.DataFrame(data).with_columns(pl.col(name).cast(dtype) for name, dtype in dtypes.items())


def _read_parquet_expr(paths: Sequence[Path]) -> str:
    """Expresión ``read_parquet`` con rutas literales y sin particionado implícito."""
    literals = ", ".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"read_parquet([{literals}], hive_partitioning=false, union_by_name=true)"
