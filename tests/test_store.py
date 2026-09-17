"""Tests del almacén *point-in-time* (tarea #2).

Todo se escribe bajo ``tmp_path``: ningún test toca el ``data/`` del repositorio.
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import duckdb
import polars as pl
import pytest

from cfdtrader.data.store import (
    ImmutableWriteError,
    InvalidRecordError,
    Layer,
    StorageError,
    Store,
    UnknownDatasetError,
    WriteOutcome,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Una barra diaria ancla su ``as_of`` al cierre de sesión (16:00 ET), no a medianoche:
#: 2024-03-15 16:00 EDT = 2024-03-15 20:00 UTC.
SESSION_CLOSE = datetime(2024, 3, 15, 20, 0, tzinfo=UTC)
AFTER_CLOSE = datetime(2024, 3, 15, 21, 0, tzinfo=UTC)
READ_NOW = datetime(2024, 3, 16, 12, 0, tzinfo=UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def _bar(**overrides: object) -> dict[str, object]:
    """Registro de barras diarias (``as_of`` de tipo ``datetime`` UTC)."""
    record: dict[str, object] = {
        "source": "yfinance",
        "series_id": "^GSPC",
        "as_of": SESSION_CLOSE,
        "fetched_at": AFTER_CLOSE,
        "published_at": None,
        "close": 5123.4,
    }
    record.update(overrides)
    return record


def _series(**overrides: object) -> dict[str, object]:
    """Registro de una serie macro (``as_of`` de tipo ``date``)."""
    record: dict[str, object] = {
        "source": "fred",
        "series_id": "CPIAUCSL",
        "as_of": date(2024, 1, 1),
        "fetched_at": datetime(2024, 2, 13, 13, 31, tzinfo=UTC),
        "published_at": datetime(2024, 2, 13, 13, 30, tzinfo=UTC),
        "value": 308.4,
    }
    record.update(overrides)
    return record


def _files(root: Path) -> list[str]:
    """Ficheros Parquet del almacén, en rutas relativas y ordenadas."""
    return sorted(str(path.relative_to(root)) for path in root.rglob("*.parquet"))


def _fingerprint(root: Path) -> dict[str, str]:
    """Ruta relativa → sha256 de cada fichero: detecta listados o bytes distintos."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Contrato y ciclo de vida del módulo
# ─────────────────────────────────────────────────────────────────────────────
def test_importing_the_module_touches_no_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import cfdtrader.data.store"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_constructing_the_store_touches_no_directory(tmp_path: Path) -> None:
    root = tmp_path / "almacen"
    Store(root)
    assert not root.exists()


def test_module_documents_the_contract() -> None:
    """El contrato vive dentro del módulo, no solo en la issue."""
    doc = importlib.import_module("cfdtrader.data.store").__doc__ or ""
    for expected in (
        "source",
        "series_id",
        "as_of",
        "fetched_at",
        "published_at",
        "version",
        "16:00 ET",
        "(dataset, source, series_id, as_of)",
        "ImmutableWriteError",
        "published_at <= at",
    ):
        assert expected in doc, f"el docstring del módulo no documenta {expected!r}"


def test_error_hierarchy_starts_at_storage_error() -> None:
    for error in (InvalidRecordError, ImmutableWriteError, UnknownDatasetError):
        assert issubclass(error, StorageError)


# ─────────────────────────────────────────────────────────────────────────────
# Escritura y relectura
# ─────────────────────────────────────────────────────────────────────────────
def test_round_trip_returns_identity_and_six_columns(store: Store) -> None:
    record = _bar()
    assert store.append("raw", "market_daily", record) is WriteOutcome.CREATED

    frame = store.read_pit("raw", "market_daily", READ_NOW)
    assert frame.height == 1
    row = frame.to_dicts()[0]

    assert (row["source"], row["series_id"], row["as_of"]) == ("yfinance", "^GSPC", SESSION_CLOSE)
    assert row["fetched_at"] == AFTER_CLOSE
    assert row["published_at"] is None
    assert row["version"] == 1
    assert row["close"] == 5123.4
    assert store.datasets("raw") == ["market_daily"]


def test_layout_partitions_by_source_and_as_of_year_with_zstd(store: Store, tmp_path: Path) -> None:
    store.append("raw", "market_daily", _bar(source="yfinance"))
    store.append("raw", "market_daily", _bar(source="stooq", series_id="^GSPC_2"))

    files = sorted(tmp_path.rglob("*.parquet"))
    assert len(files) == 2
    relative = files[0].relative_to(tmp_path)
    assert relative.parts[0] == "raw"
    assert relative.parts[1] == "market_daily"
    assert relative.parent.parent.name.startswith("source=")
    assert relative.parent.name == "year=2024"
    assert sorted(path.parent.parent.name for path in files) == ["source=stooq", "source=yfinance"]

    connection = duckdb.connect()
    try:
        compressions = {
            row[0]
            for path in files
            for row in connection.execute(
                "SELECT DISTINCT compression FROM parquet_metadata(?)", [str(path)]
            ).fetchall()
        }
    finally:
        connection.close()
    assert compressions == {"ZSTD"}


def test_date_and_datetime_as_of_coexist_across_datasets(store: Store) -> None:
    store.append("raw", "macro", _series())
    store.append("raw", "market_daily", _bar())

    bars = store.read_pit("raw", "market_daily", READ_NOW)
    series = store.read_pit("raw", "macro", datetime(2024, 3, 1, tzinfo=UTC))

    assert bars["as_of"][0] == SESSION_CLOSE
    assert bars.schema["as_of"] == pl.Datetime("us", "UTC")
    assert series["as_of"][0] == date(2024, 1, 1)
    assert series.schema["as_of"] == pl.Date


def test_mixing_as_of_kinds_inside_one_dataset_is_rejected(store: Store) -> None:
    store.append("raw", "macro", _series())
    with pytest.raises(InvalidRecordError, match="un solo tipo"):
        store.append("raw", "macro", _bar(series_id="FEDFUNDS"))
    assert len(_files(store.root)) == 1


def test_missing_published_at_is_stored_as_null(store: Store) -> None:
    store.append("raw", "market_daily", _bar())

    frame = store.read_pit("raw", "market_daily", READ_NOW)
    assert frame["published_at"].null_count() == 1
    assert frame["published_at"][0] is None
    assert frame["fetched_at"][0] != frame["published_at"][0]

    nulls = store.sql("SELECT count(*) AS n FROM raw.market_daily WHERE published_at IS NULL")
    assert nulls["n"][0] == 1


def test_offset_datetime_is_stored_and_read_in_utc(store: Store) -> None:
    store.append(
        "raw",
        "market_daily",
        _bar(fetched_at=datetime(2024, 3, 15, 23, 0, tzinfo=timezone(timedelta(hours=2)))),
    )

    frame = store.read_pit("raw", "market_daily", READ_NOW)
    value = frame["fetched_at"][0]
    assert isinstance(value, datetime)
    assert frame.schema["fetched_at"] == pl.Datetime("us", "UTC")
    assert value.utcoffset() == timedelta(0)
    assert value == AFTER_CLOSE


def test_batch_write_uses_one_file_per_partition(store: Store) -> None:
    records = [_bar(series_id="^GSPC"), _bar(series_id="SPY")]
    assert store.append("raw", "market_daily", records) is WriteOutcome.CREATED

    assert store.read_pit("raw", "market_daily", READ_NOW).height == 2
    assert len(_files(store.root)) == 1


def test_identity_check_reads_once_per_partition_not_once_per_record(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#53: comprobar la identidad costaba una lectura del almacén **por registro**.

    Cada comprobación abría conexión y volvía a registrar la vista de *todos* los
    datasets: medido sobre el almacén real, ~76-103 ms por fila. Escribir una serie
    de 21 años son ~5.000 filas, o sea horas para meter lo que cabe en un fichero.
    La comprobación se agrupa por partición, así que el número de lecturas depende
    del número de particiones y **no** del número de registros.
    """
    records = [
        _bar(
            series_id="^GSPC",
            as_of=SESSION_CLOSE + timedelta(days=offset),
            fetched_at=SESSION_CLOSE + timedelta(days=offset, hours=1),
            close=5000.0 + offset,
        )
        for offset in range(60)
    ]
    assert store.append("raw", "market_daily", records) is WriteOutcome.CREATED

    reads: list[str] = []
    # Es una prueba de regresión de rendimiento: hay que contar lecturas reales.
    original_fetch = Store._fetch  # pyright: ignore[reportPrivateUsage]

    def spy(self: Store, query: str, params: Sequence[object]) -> pl.DataFrame:
        reads.append(query)
        return original_fetch(self, query, params)

    monkeypatch.setattr(Store, "_fetch", spy)
    # El mismo contenido: la comprobación recorre los 60 registros y no escribe nada.
    assert store.append("raw", "market_daily", records) is WriteOutcome.UNCHANGED

    assert len(reads) == 1, f"60 registros de una partición deben ser 1 lectura, no {len(reads)}"


def test_identity_check_spans_every_year_partition_of_the_batch(store: Store) -> None:
    """Un lote de 21 años cae en 21 particiones: todas deben comprobarse.

    Si la comprobación mirase solo la partición del primer registro, reescribir el
    histórico de una serie pasaría como si estuviera vacío.
    """
    years = list(range(2005, 2025))
    records = [
        _series(
            as_of=date(year, 1, 1),
            published_at=datetime(year, 2, 1, 13, 30, tzinfo=UTC),
            fetched_at=datetime(year, 2, 1, 13, 31, tzinfo=UTC),
            value=float(year),
        )
        for year in years
    ]
    assert store.append("raw", "macro", records) is WriteOutcome.CREATED

    assert store.append("raw", "macro", records) is WriteOutcome.UNCHANGED
    # Un cambio en el último año (el de la última partición) también se detecta.
    changed = [*records[:-1], {**records[-1], "value": 1.0}]
    with pytest.raises(ImmutableWriteError):
        store.append("raw", "macro", changed)


# ─────────────────────────────────────────────────────────────────────────────
# Rechazo de registros inválidos
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "override",
    [
        {"source": ""},
        {"source": "   "},
        {"source": "yfinance/v2"},
        {"series_id": ""},
        {"series_id": "   "},
        {"as_of": "2024-03-15"},
        {"fetched_at": datetime(2024, 3, 1, tzinfo=UTC)},
        {"published_at": datetime(2024, 4, 1, tzinfo=UTC)},
        {"version": 0},
        {"version": -3},
    ],
    ids=[
        "source-vacio",
        "source-en-blanco",
        "source-con-separador-de-ruta",
        "series-id-vacio",
        "series-id-en-blanco",
        "as-of-no-temporal",
        "fetched-at-anterior-a-as-of",
        "published-at-posterior-a-fetched-at",
        "version-cero",
        "version-negativa",
    ],
)
def test_invalid_writes_are_rejected_without_writing(
    store: Store, override: dict[str, object]
) -> None:
    with pytest.raises(InvalidRecordError):
        store.append("raw", "market_daily", _bar(**override))
    assert list(store.root.rglob("*")) == []


@pytest.mark.parametrize("field", ["as_of", "fetched_at", "published_at"])
def test_naive_datetime_is_rejected(store: Store, field: str) -> None:
    with pytest.raises(InvalidRecordError, match="zona horaria"):
        store.append("raw", "market_daily", _bar(**{field: datetime(2024, 3, 15, 21)}))
    assert list(store.root.rglob("*")) == []


@pytest.mark.parametrize("dataset", ["", "market-daily", "market.daily", "1daily", "market daily"])
def test_invalid_dataset_names_are_rejected(store: Store, dataset: str) -> None:
    with pytest.raises(InvalidRecordError, match="dataset"):
        store.append("raw", dataset, _bar())


def test_invalid_dataset_names_are_rejected_on_read(store: Store) -> None:
    with pytest.raises(InvalidRecordError, match="dataset"):
        store.read_pit("raw", "../derived", READ_NOW)
    with pytest.raises(InvalidRecordError, match="capa"):
        store.datasets(cast(Layer, "journal"))


def test_invalid_layer_is_rejected(store: Store) -> None:
    with pytest.raises(InvalidRecordError, match="capa"):
        store.append(cast(Layer, "journal"), "market_daily", _bar())


def test_one_write_one_schema(store: Store) -> None:
    with pytest.raises(InvalidRecordError, match="mismas columnas"):
        store.append("raw", "market_daily", [_bar(), _bar(series_id="SPY", extra=1)])


def test_repeated_identity_inside_one_write_is_rejected(store: Store) -> None:
    with pytest.raises(InvalidRecordError, match="identidad"):
        store.append("raw", "market_daily", [_bar(), _bar()])
    assert list(store.root.rglob("*")) == []


def test_caller_supplied_version_is_validated_but_the_store_owns_it(store: Store) -> None:
    assert store.append("raw", "market_daily", _bar(version=7)) is WriteOutcome.CREATED
    assert store.read_pit("raw", "market_daily", READ_NOW)["version"][0] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Inmutabilidad de `raw`
# ─────────────────────────────────────────────────────────────────────────────
def test_repeated_identical_append_is_a_noop(store: Store) -> None:
    record = _bar()
    assert store.append("raw", "market_daily", record) is WriteOutcome.CREATED
    before = _fingerprint(store.root)

    assert store.append("raw", "market_daily", record) is WriteOutcome.UNCHANGED
    assert _fingerprint(store.root) == before
    assert store.read_pit("raw", "market_daily", READ_NOW).height == 1


def test_retry_with_later_fetched_at_is_still_a_noop(store: Store) -> None:
    """Un reintento de ingesta trae otro ``fetched_at``: no es contenido distinto."""
    store.append("raw", "market_daily", _bar())
    before = _fingerprint(store.root)

    retry = _bar(fetched_at=datetime(2024, 3, 18, 9, 0, tzinfo=UTC))
    assert store.append("raw", "market_daily", retry) is WriteOutcome.UNCHANGED

    assert _fingerprint(store.root) == before
    frame = store.read_pit("raw", "market_daily", READ_NOW)
    assert frame.height == 1
    assert frame["fetched_at"][0] == AFTER_CLOSE  # se conserva el primer momento en que se conoció


def test_conflicting_append_raises_and_leaves_files_untouched(store: Store) -> None:
    store.append("raw", "market_daily", _bar())
    before = _fingerprint(store.root)

    with pytest.raises(ImmutableWriteError, match="no se sobrescribe"):
        store.append("raw", "market_daily", _bar(close=1.0))

    assert _fingerprint(store.root) == before


def test_every_write_creates_new_files_and_never_modifies_old_ones(store: Store) -> None:
    store.append("raw", "market_daily", _bar(series_id="^GSPC"))
    first = _fingerprint(store.root)

    store.append("raw", "market_daily", _bar(series_id="SPY"))
    after = _fingerprint(store.root)

    assert len(after) == 2
    assert all(after[path] == digest for path, digest in first.items())


def test_replace_is_refused_in_raw(store: Store) -> None:
    store.append("raw", "market_daily", _bar())
    before = _fingerprint(store.root)

    with pytest.raises(ImmutableWriteError, match="inmutable"):
        store.replace("raw", "market_daily", _bar(close=1.0))

    assert _fingerprint(store.root) == before


def test_replace_in_derived_keeps_a_single_row_per_identity(store: Store) -> None:
    feature = _bar(as_of=date(2024, 3, 15), series_id="2024-03-15", value=1.0)
    revised = {**feature, "value": 2.0}
    assert store.replace("derived", "features_daily", feature) is WriteOutcome.CREATED
    assert store.replace("derived", "features_daily", revised) is WriteOutcome.CREATED
    assert store.replace("derived", "features_daily", revised) is WriteOutcome.UNCHANGED

    frame = store.read_pit("derived", "features_daily", READ_NOW)
    assert frame.height == 1
    assert frame["value"][0] == 2.0
    assert frame["version"][0] == 2

    # El estado consultable del dataset también tiene una sola fila por identidad:
    # el valor sustituido no se puede leer por SQL. La historia sigue en disco.
    assert store.sql("SELECT value, version FROM derived.features_daily").to_dicts() == [
        {"value": 2.0, "version": 2}
    ]
    assert len(_files(store.root)) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Revisiones y visibilidad point-in-time
# ─────────────────────────────────────────────────────────────────────────────
def test_append_revision_assigns_next_version_and_keeps_history(store: Store) -> None:
    revised = _series(
        value=309.0,
        published_at=datetime(2024, 3, 14, 12, 30, tzinfo=UTC),
        fetched_at=datetime(2024, 3, 14, 12, 31, tzinfo=UTC),
    )
    assert store.append("raw", "macro", _series()) is WriteOutcome.CREATED
    assert store.append_revision("raw", "macro", revised) is WriteOutcome.CREATED
    assert store.append_revision("raw", "macro", revised) is WriteOutcome.UNCHANGED

    before = store.read_pit("raw", "macro", datetime(2024, 2, 20, tzinfo=UTC))
    after = store.read_pit("raw", "macro", datetime(2024, 3, 20, tzinfo=UTC))

    assert (before.height, before["value"][0], before["version"][0]) == (1, 308.4, 1)
    assert (after.height, after["value"][0], after["version"][0]) == (1, 309.0, 2)

    # La revisión añade un fichero; el valor original sigue en disco, solo deja de ser visible.
    assert len(_files(store.root)) == 2


def test_point_in_time_visibility_ignores_as_of(store: Store) -> None:
    yesterday = datetime(2024, 6, 10, 22, 0, tzinfo=UTC)
    today = datetime(2024, 6, 11, 22, 0, tzinfo=UTC)

    store.append(
        "raw",
        "macro",
        _series(
            series_id="OLD_REF",
            as_of=date(2020, 1, 1),
            published_at=yesterday - timedelta(hours=1),
            fetched_at=yesterday - timedelta(minutes=55),
        ),
    )
    store.append(
        "raw",
        "macro",
        _series(
            series_id="OLD_REF_TODAY",
            as_of=date(2019, 1, 1),
            published_at=today - timedelta(hours=1),
            fetched_at=today - timedelta(minutes=55),
        ),
    )

    before_anything = store.read_pit("raw", "macro", yesterday - timedelta(days=1))
    at_yesterday = store.read_pit("raw", "macro", yesterday)
    at_today = store.read_pit("raw", "macro", today)

    assert before_anything.height == 0
    assert before_anything.columns  # el esquema sí se conoce aunque no haya filas visibles
    assert at_yesterday["series_id"].to_list() == ["OLD_REF"]
    assert sorted(at_today["series_id"].to_list()) == ["OLD_REF", "OLD_REF_TODAY"]


def test_visibility_of_null_published_at_falls_back_to_fetched_at(store: Store) -> None:
    store.append("raw", "market_daily", _bar())  # published_at NULL, fetched 2024-03-15 21:00 UTC

    assert store.read_pit("raw", "market_daily", AFTER_CLOSE - timedelta(minutes=1)).height == 0
    assert store.read_pit("raw", "market_daily", AFTER_CLOSE).height == 1


def test_read_pit_can_filter_by_series_and_date(store: Store) -> None:
    store.append("raw", "market_daily", _bar(series_id="^GSPC"))
    store.append("raw", "market_daily", _bar(series_id="SPY"))

    assert store.read_pit("raw", "market_daily", date(2024, 3, 16), series_id="SPY")[
        "series_id"
    ].to_list() == ["SPY"]


def test_reading_an_empty_dataset_is_an_explicit_error(store: Store) -> None:
    with pytest.raises(UnknownDatasetError, match="market_daily"):
        store.read_pit("raw", "market_daily", READ_NOW)


# ─────────────────────────────────────────────────────────────────────────────
# SQL sobre Parquet, sin base de datos persistida
# ─────────────────────────────────────────────────────────────────────────────
def test_sql_queries_parquet_and_creates_no_database(store: Store, tmp_path: Path) -> None:
    store.append("raw", "market_daily", _bar())
    store.append("derived", "features_daily", _bar(as_of=date(2024, 3, 15), series_id="2024-03-15"))

    frame = store.sql("SELECT source, series_id, version FROM raw.market_daily")
    assert frame.to_dicts() == [{"source": "yfinance", "series_id": "^GSPC", "version": 1}]

    joined = store.sql(
        "SELECT count(*) AS n FROM raw.market_daily AS r "
        "JOIN derived.features_daily AS d USING (source)"
    )
    assert joined["n"][0] == 1

    assert list(tmp_path.rglob("*.duckdb")) == []
    assert {path.suffix for path in tmp_path.rglob("*") if path.is_file()} == {".parquet"}


def test_sql_view_shows_one_row_per_identity_without_losing_the_history(store: Store) -> None:
    """La vista SQL expone el estado vigente; las revisiones anteriores siguen en disco."""
    revised = _series(
        value=309.0,
        published_at=datetime(2024, 3, 14, 12, 30, tzinfo=UTC),
        fetched_at=datetime(2024, 3, 14, 12, 31, tzinfo=UTC),
    )
    store.append("raw", "macro", _series())
    store.append_revision("raw", "macro", revised)

    assert store.sql("SELECT count(*) AS n FROM raw.macro")["n"][0] == 1
    assert store.sql("SELECT value, version FROM raw.macro").to_dicts() == [
        {"value": 309.0, "version": 2}
    ]

    # Nada se ha borrado: siguen los dos ficheros y la lectura antigua ve el valor original.
    assert len(_files(store.root)) == 2
    before = store.read_pit("raw", "macro", datetime(2024, 2, 20, tzinfo=UTC))
    assert (before["value"][0], before["version"][0]) == (308.4, 1)


def test_the_data_directory_guard_detects_a_write(tmp_path: Path) -> None:
    """#54: el guardián de A22 tiene que detectar de verdad una escritura.

    Un guardián que no ve nada no vigila nada. Se comprueba sobre un directorio
    de prueba que la huella cambia al crear, modificar y borrar un fichero. El
    guardián real (la sesión no toca `data/`) vive en `tests/conftest.py`; antes
    se comprobaba exigiendo que `data/` no tuviera Parquet, lo que hacía fallar
    la puerta en cuanto se ejecutaba la ingesta real de A14 sin que ningún test
    hubiera escrito nada.
    """
    from conftest import fingerprint

    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "part-1.parquet").write_bytes(b"uno")
    before = fingerprint(tmp_path)

    (tmp_path / "raw" / "part-2.parquet").write_bytes(b"dos")
    with_two = fingerprint(tmp_path)
    assert set(with_two) - set(before) == {"raw/part-2.parquet"}

    (tmp_path / "raw" / "part-1.parquet").write_bytes(b"otro")
    assert fingerprint(tmp_path)["raw/part-1.parquet"] != before["raw/part-1.parquet"]

    (tmp_path / "raw" / "part-2.parquet").unlink()
    assert set(with_two) - set(fingerprint(tmp_path)) == {"raw/part-2.parquet"}
