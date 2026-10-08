"""Tests del contrato de *layout* del almacén (`tech_stack.md` §12.10, tarea #139).

Lo que se comprueba es la **maquinaria**: que cada dataset que el proyecto escribe
declara su firma, que un cambio de tipo o una columna desconocida se rechazan con un
error que nombra fichero, columna y los dos tipos, que una columna ausente se lee a
``null`` **declarado** (nunca ``0``) y que la cuarentena de `raw` **mueve** el fichero
sin reescribirlo.
"""

from __future__ import annotations

import ast
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from cfdtrader.data.contracts import LAYOUTS, ColumnSpec, layout_of
from cfdtrader.data.store import LayoutMismatchError, Store

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "cfdtrader"
CONTRACTS = SRC / "data" / "contracts.py"

NOW = datetime(2024, 3, 15, 21, 0, tzinfo=UTC)
READ_NOW = datetime(2024, 4, 1, 12, 0, tzinfo=UTC)


def _macro(**overrides: object) -> dict[str, object]:
    """Registro de `raw.macro` con el *payload* declarado completo."""
    record: dict[str, object] = {
        "source": "fred",
        "series_id": "CPIAUCSL",
        "as_of": date(2024, 1, 1),
        "fetched_at": datetime(2024, 2, 13, 13, 31, tzinfo=UTC),
        "published_at": None,
        "value": 308.4,
        "unit": "index",
        "name": "CPI",
    }
    record.update(overrides)
    return record


def _files(root: Path) -> list[Path]:
    """Ficheros Parquet del almacén, en orden."""
    return sorted(root.rglob("*.parquet"))


# ─────────────────────────────────────────────────────────────────────────────
# A1 — La declaración coincide con lo que se escribe
# ─────────────────────────────────────────────────────────────────────────────
def _module_declarations(tree: ast.AST) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Constantes de texto de un módulo y las colecciones literales de nombres.

    Devuelve ``(constantes, colecciones)``: ``LABELS_DATASET -> "labels"`` y
    ``PAYLOAD_COLUMNS -> {"market_daily", "sectors", ...}`` (las **claves** de un
    diccionario son los nombres de dataset; los valores, no).
    """
    constants: dict[str, str] = {}
    collections: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
        if not isinstance(target, ast.Name) or node.value is None:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            constants[target.id] = value.value
        elif isinstance(value, ast.Dict):
            collections[target.id] = {
                key.value
                for key in value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    return constants, collections


def _writers() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Los ``(capa, dataset)`` que el proyecto escribe, leídos del **código**.

    Devuelve dos cosas: los pares que el escaneo **resuelve** y los que cada módulo que
    escribe **declara** en sus constantes. Los dos hacen falta porque
    ``market._write`` recibe el nombre del dataset **por parámetro**: ahí no se puede
    resolver leyendo la llamada, y se completa con lo que el propio módulo declara
    (``PAYLOAD_COLUMNS`` y las constantes con ``DATASET`` en el nombre). La capa es
    siempre un literal en este proyecto, porque su tipo es ``Literal["raw","derived"]``.
    """
    from cfdtrader.data.store import LAYERS

    resolved: set[tuple[str, str]] = set()
    declared_by_writers: set[tuple[str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants, collections = _module_declarations(tree)
        layers: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            call = ast.unparse(node.func).split(".")[-1]
            if call not in {"append", "append_revision", "replace"}:
                continue
            layer_node, dataset = node.args[0], node.args[1]
            if isinstance(layer_node, ast.Constant) and isinstance(layer_node.value, str):
                layer = layer_node.value
            elif isinstance(layer_node, ast.Name):
                layer = constants.get(layer_node.id)
            else:
                layer = None
            if layer not in LAYERS:
                continue
            layers.add(layer)
            if isinstance(dataset, ast.Constant) and isinstance(dataset.value, str):
                resolved.add((layer, dataset.value))
            elif isinstance(dataset, ast.Name) and dataset.id in constants:
                resolved.add((layer, constants[dataset.id]))
        if not layers:
            continue
        names = set(collections.get("PAYLOAD_COLUMNS", set()))
        names |= {value for name, value in constants.items() if "DATASET" in name}
        for layer in layers:
            declared_by_writers |= {(layer, name) for name in names}
    return resolved, declared_by_writers


def test_every_dataset_the_project_writes_declares_its_signature() -> None:
    """A1 (#139): ningún dataset se escribe sin firma declarada, y ninguna firma sobra.

    En los dos sentidos: un escritor sin declaración es un dataset que nadie valida, y
    una declaración sin escritor es una firma que ya no es de nadie.
    """
    resolved, declared_by_writers = _writers()
    assert resolved <= set(LAYOUTS), sorted(resolved - set(LAYOUTS))
    assert set(LAYOUTS) <= resolved | declared_by_writers, sorted(
        set(LAYOUTS) - resolved - declared_by_writers
    )


def test_the_declaration_matches_the_columns_the_producers_write() -> None:
    """A1 (#139): la firma declarada es la que los productores escriben.

    Se compara contra la fuente que **posee** cada lista de columnas, para que añadir una
    columna sin subir el `layout_version` rompa aquí y no en producción.
    """
    from cfdtrader.data import market, news
    from cfdtrader.features import store as features_store

    for dataset, columns in market.PAYLOAD_COLUMNS.items():
        declaration = layout_of("raw", dataset)
        assert declaration is not None, f"'raw.{dataset}' se escribe y no está declarado"
        assert set(declaration.signature) == set(columns)

    features = layout_of("derived", "features_daily")
    assert features is not None
    assert set(features.signature) == {
        *features_store.ALL_FEATURE_COLUMNS,
        *features_store.VERSION_COLUMNS,
    }

    headline = news.Headline(
        feed="feed",
        source="yfinance",
        title="titular",
        url="https://example.invalid/1",
        published_at=NOW,
    )
    row = news.headline_records([headline], fetched_at=NOW)[0]
    declared = layout_of("raw", "news_headlines")
    assert declared is not None
    mandatory = {"source", "series_id", "as_of", "fetched_at", "published_at", "version"}
    assert set(declared.signature) == set(row) - mandatory


def test_the_declaration_is_readable_by_machine() -> None:
    """A1 (#139): la declaración vive en el código, con versión y tipo por columna."""
    assert "LAYOUTS" in CONTRACTS.read_text(encoding="utf-8")
    for (layer, dataset), declaration in LAYOUTS.items():
        assert (declaration.layer, declaration.dataset) == (layer, dataset)
        assert declaration.layout_version >= 1
        assert declaration.payload
        assert len(declaration.signature) == len(set(declaration.signature))


def test_a_spec_outside_the_vocabulary_or_a_required_column_is_rejected() -> None:
    """A1 (#139): la firma se valida a sí misma: tipos declarados y columnas no obligatorias."""
    with pytest.raises(ValueError, match="tipo declarado desconocido"):
        _ = ColumnSpec("close", "decimal")
    with pytest.raises(ValueError, match="columna obligatoria"):
        _ = ColumnSpec("version", "int")
    with pytest.raises(ValueError, match="'since'"):
        _ = ColumnSpec("close", "float", since=0)


# ─────────────────────────────────────────────────────────────────────────────
# A2/A4 — Un cambio de tipo y una columna desconocida se rechazan al leer
# ─────────────────────────────────────────────────────────────────────────────
def _foreign_file(
    root: Path,
    layer: str,
    dataset: str,
    payload: dict[str, object],
    *,
    source: str = "stooq",
    series_id: str = "CPIAUCSL",
) -> Path:
    """Escribe un Parquet **a mano**, como lo haría otra versión del productor.

    Es la única forma honesta de probar el lector: un fichero que el ``Store`` actual no
    habría escrito (otro tipo, otra firma) pero que está en el almacén.
    """
    import polars as pl

    row: dict[str, object] = {
        "source": source,
        "series_id": series_id,
        "as_of": date(2024, 1, 1),
        "fetched_at": datetime(2024, 2, 13, 13, 31, tzinfo=UTC),
        "published_at": datetime(2024, 2, 13, 13, 30, tzinfo=UTC),
        "version": 1,
        **payload,
    }
    directory = root / layer / dataset / f"source={source}" / "year=2024"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "part-manual.parquet"
    pl.DataFrame([row]).write_parquet(path)
    return path


def _old_layout(monkeypatch: pytest.MonkeyPatch, *, extra: ColumnSpec) -> None:
    """Declara `raw.macro` en la versión siguiente, con una columna **aditiva** más."""
    from cfdtrader.data import contracts

    current = contracts.LAYOUTS[("raw", "macro")]
    monkeypatch.setitem(
        contracts.LAYOUTS,
        ("raw", "macro"),
        contracts.DatasetLayout(
            layer="raw",
            dataset="macro",
            layout_version=current.layout_version + 1,
            payload=(*current.payload, extra),
        ),
    )


def test_a_type_change_is_a_typed_error_that_names_file_column_and_types(tmp_path: Path) -> None:
    """A2 (#139): tipo distinto ⇒ error tipado con fichero, columna y los **dos** tipos."""
    store = Store(tmp_path)
    store.append("raw", "macro", _macro())
    path = _foreign_file(tmp_path, "raw", "macro", {"value": "308.4", "unit": "i", "name": "CPI"})

    with pytest.raises(LayoutMismatchError) as error:
        _ = store.layout("raw", "macro")
    assert error.value.path == path
    assert error.value.column == "value"
    assert (error.value.expected, error.value.found) == ("float", "str")
    message = str(error.value)
    assert str(path) in message and "'value'" in message and "float" in message and "str" in message

    # Las **dos** puertas de lectura lo rechazan, no solo el inspector:
    with pytest.raises(LayoutMismatchError):
        _ = store.read_pit("raw", "macro", READ_NOW)
    with pytest.raises(LayoutMismatchError):
        _ = store.sql("SELECT * FROM raw.macro")


def test_an_unknown_column_is_a_typed_error_and_not_an_ignored_column(tmp_path: Path) -> None:
    """A4 (#139): una columna que la firma no declara **no** se ignora en silencio."""
    store = Store(tmp_path)
    store.append("raw", "macro", _macro())
    _ = _foreign_file(
        tmp_path,
        "raw",
        "macro",
        {"value": 308.4, "unit": "index", "name": "CPI", "frequency": "monthly"},
    )

    with pytest.raises(LayoutMismatchError) as error:
        _ = store.sql("SELECT * FROM raw.macro")
    assert error.value.column == "frequency"
    assert error.value.expected is None
    assert error.value.found == "str"
    assert "columna nueva con nombre nuevo" in str(error.value)


def test_a_file_that_breaks_the_mandatory_contract_is_a_typed_error(tmp_path: Path) -> None:
    """A2 (#139): las seis columnas obligatorias también tienen tipo declarado."""
    store = Store(tmp_path)
    _ = _foreign_file(tmp_path, "raw", "macro", {"value": 308.4, "unit": "i", "name": "CPI"})
    path = tmp_path / "raw" / "macro" / "source=stooq" / "year=2024" / "part-manual.parquet"
    import polars as pl

    pl.DataFrame(
        [
            {
                "source": "stooq",
                "series_id": "CPIAUCSL",
                "as_of": date(2024, 1, 1),
                "fetched_at": datetime(2024, 2, 13, 13, 31, tzinfo=UTC),
                "version": 1,
                "value": 308.4,
                "unit": "index",
                "name": "CPI",
            }
        ]
    ).write_parquet(path)

    with pytest.raises(LayoutMismatchError) as error:
        _ = store.layout("raw", "macro")
    assert error.value.column == "published_at"
    assert error.value.found is None
    assert "datetime" in str(error.value)


# ─────────────────────────────────────────────────────────────────────────────
# A3/A6 — Columna aditiva ausente: se lee a `null` y el layout se declara
# ─────────────────────────────────────────────────────────────────────────────
def test_a_missing_additive_column_is_read_as_null_and_declares_old_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A3 (#139): columna aditiva ausente ⇒ ``null`` declarado, nunca ``0``, y ``old_layout``."""
    _old_layout(monkeypatch, extra=ColumnSpec("frequency", "str", since=2))
    store = Store(tmp_path)
    store.append("raw", "macro", _macro())

    reading = store.layout("raw", "macro")
    assert reading is not None
    assert reading.versions == (1,)
    assert reading.old_layout is True
    assert reading.null_columns == ("frequency",)

    frame = store.read_pit("raw", "macro", READ_NOW)
    assert frame["frequency"].to_list() == [None]
    assert frame["value"].to_list() == [308.4]
    # La columna existe **declarada**, con su tipo: no se rellena con 0 ni se omite.
    assert frame["frequency"].dtype == frame["unit"].dtype
    assert store.sql("SELECT frequency FROM raw.macro")["frequency"].to_list() == [None]


def test_a_mixed_history_is_declared_instead_of_merged_in_silence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A6 (#139): un histórico con dos *layouts* se lee **declarándolo**."""
    from loguru import logger

    _old_layout(monkeypatch, extra=ColumnSpec("frequency", "str", since=2))
    store = Store(tmp_path)
    store.append("raw", "macro", _macro())
    _ = _foreign_file(
        tmp_path,
        "raw",
        "macro",
        {"value": 310.0, "unit": "index", "name": "CPI", "frequency": "monthly"},
    )

    reading = store.layout("raw", "macro")
    assert reading is not None
    assert reading.versions == (1, 2)
    assert reading.old_layout is True
    described = reading.describe()
    assert "layout vigente 2" in described
    assert "mezclado" in described and "frequency" in described

    captured: list[str] = []

    def sink(message: Any) -> None:
        captured.append(str(message.record["message"]))

    sink_id = logger.add(sink, level="WARNING")
    try:
        frame = store.read_pit("raw", "macro", READ_NOW)
    finally:
        logger.remove(sink_id)
    assert any("histórico mezclado" in text for text in captured)

    # El fichero viejo sale a `null` en la columna aditiva, y el nuevo la trae: ni `0` ni
    # una media de las dos cosas.
    values = dict(zip(frame["source"].to_list(), frame["frequency"].to_list(), strict=True))
    assert values == {"fred": None, "stooq": "monthly"}


def test_the_layout_version_is_inferred_from_the_columns() -> None:
    """A3 (#139): la firma **es** el esquema, así que la versión se infiere leyendo."""
    from cfdtrader.data.contracts import DatasetLayout

    layout = DatasetLayout(
        layer="raw",
        dataset="probe",
        layout_version=3,
        payload=(
            ColumnSpec("a", "float"),
            ColumnSpec("b", "float", since=2),
            ColumnSpec("c", "float", since=3),
        ),
    )
    assert layout.version_of(frozenset({"a", "b", "c"})) == 3
    assert layout.version_of(frozenset({"a", "b"})) == 2
    assert layout.version_of(frozenset({"a"})) == 1
    assert layout.version_of(frozenset()) == 0


# ─────────────────────────────────────────────────────────────────────────────
# A5/A7 — La cuarentena mueve, `raw` sigue inmutable
# ─────────────────────────────────────────────────────────────────────────────
def test_the_quarantine_moves_the_file_without_rewriting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A5 (#139): la cuarentena **mueve** el fichero; los bytes son idénticos."""
    _old_layout(monkeypatch, extra=ColumnSpec("frequency", "str", since=2))
    store = Store(tmp_path)
    store.append("raw", "macro", _macro())
    original = _files(tmp_path)[0]
    content = original.read_bytes()
    stamp = original.stat().st_mtime_ns

    moved = store.quarantine("raw", "macro")

    assert len(moved) == 1
    target = moved[0]
    assert "_legacy" in target.parts
    assert "1" in target.parts
    assert target.read_bytes() == content
    assert target.stat().st_mtime_ns == stamp
    assert not original.exists()


def test_the_quarantined_file_leaves_the_views_and_the_dataset_is_read_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A5 (#139): `_legacy/` queda fuera de las vistas, así que deja de leerse."""
    from cfdtrader.data.store import UnknownDatasetError

    _old_layout(monkeypatch, extra=ColumnSpec("frequency", "str", since=2))
    store = Store(tmp_path)
    store.append("raw", "macro", _macro())
    store.quarantine("raw", "macro")

    reading = store.layout("raw", "macro")
    assert reading is not None and reading.files == ()
    with pytest.raises(UnknownDatasetError):
        _ = store.read_pit("raw", "macro", READ_NOW)
    with pytest.raises(Exception, match=r"does not exist|no tiene"):
        _ = store.sql("SELECT * FROM raw.macro")
    # Nada se ha borrado: sigue en `_legacy/`, que es de donde se re-ingesta.
    assert len(_files(tmp_path)) == 1
    assert store.quarantine("raw", "macro") == ()


def test_the_quarantine_is_raw_only_and_needs_a_declared_signature(tmp_path: Path) -> None:
    """A5 (#139): en `derived` la política es recomputar; sin declaración no hay cuarentena."""
    from cfdtrader.data.store import StorageError

    store = Store(tmp_path)
    with pytest.raises(StorageError, match="solo de 'raw'"):
        _ = store.quarantine("derived", "labels")
    with pytest.raises(StorageError, match="no tiene firma declarada"):
        _ = store.quarantine("raw", "probe")


def test_raw_is_written_only_by_append(tmp_path: Path) -> None:
    """A7 (#139): el único camino que escribe en `raw` es ``append`` (y su revisión explícita)."""
    from cfdtrader.data.store import ImmutableWriteError

    store = Store(tmp_path)
    store.append("raw", "macro", _macro())
    with pytest.raises(ImmutableWriteError):
        _ = store.replace("raw", "macro", _macro(value=1.0))
    assert store.sql("SELECT value FROM raw.macro")["value"].to_list() == [308.4]


# ─────────────────────────────────────────────────────────────────────────────
# Lista blanca: un dataset sin declaración no se valida (y se lee como siempre)
# ─────────────────────────────────────────────────────────────────────────────
def test_a_dataset_without_a_declaration_is_read_without_being_validated(tmp_path: Path) -> None:
    """A1 (#139): la declaración es una lista blanca; un dataset de prueba no se valida."""
    store = Store(tmp_path)
    store.append(
        "raw",
        "probe",
        {
            "source": "manual",
            "series_id": "manual",
            "as_of": NOW,
            "fetched_at": NOW,
            "published_at": None,
            "whatever": 1.0,
            "otra_cosa": "texto",
        },
    )

    assert store.layout("raw", "probe") is None
    assert store.read_pit("raw", "probe", READ_NOW).height == 1
    assert store.sql("SELECT otra_cosa FROM raw.probe")["otra_cosa"].to_list() == ["texto"]
