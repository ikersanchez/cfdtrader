"""Tests de la identidad ampliada por dataset (`#144`).

La identidad del almacén es ``(source, series_id, as_of)``, y en `raw.news_headlines`
eso **no** es único: dos titulares del mismo feed pueden compartir instante y solo se
distinguen por su ``headline_hash``. Este módulo fija que un dataset pueda **ampliar**
su identidad con columnas de *payload* ya declaradas —sin cambiar el Parquet ni el
``layout_version``— y que el resto se comporte exactamente como antes.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from cfdtrader.data.contracts import LAYOUTS, ColumnSpec, DatasetLayout, layout_of
from cfdtrader.data.news import ingest as news_ingest
from cfdtrader.data.sources.news import Headline
from cfdtrader.data.store import (
    ImmutableWriteError,
    InvalidRecordError,
    Store,
    WriteOutcome,
)

#: Instante de publicación: el mismo para los titulares que "colisionan".
PUB = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)

#: Cuándo se recogieron: posterior al instante de publicación.
FETCHED = datetime(2026, 10, 9, 12, 45, tzinfo=UTC)

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64


def _headline(**overrides: object) -> dict[str, object]:
    """Registro de `raw.news_headlines` con el *payload* declarado completo."""
    record: dict[str, object] = {
        "source": "rss",
        "series_id": "cnbc",
        "as_of": PUB,
        "fetched_at": FETCHED,
        "published_at": PUB,
        "title": "Titular",
        "url": "https://example.invalid/1",
        "headline_hash": _HASH_A,
    }
    record.update(overrides)
    return record


def _macro(**overrides: object) -> dict[str, object]:
    """Registro de `raw.macro` (dataset **sin** identidad ampliada)."""
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


def _layout(*identity_columns: str) -> DatasetLayout:
    """Declaración de prueba con una columna de *payload* ampliable."""
    return DatasetLayout(
        layer="raw",
        dataset="probe",
        layout_version=1,
        payload=(ColumnSpec("headline_hash", "str"),),
        identity_columns=identity_columns,
    )


# ─────────────────────────────────────────────────────────────────────────────
# A1 · La declaración se valida contra el payload
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_a_valid_extension_is_accepted() -> None:
    """A1: una columna de *payload* declarada puede ampliar la identidad."""
    assert _layout("headline_hash").identity_columns == ("headline_hash",)


def test_a1_a_column_outside_the_payload_is_rejected() -> None:
    """A1: ampliar la identidad con una columna no declarada es error."""
    with pytest.raises(ValueError, match="no está declarada"):
        _layout("url")


def test_a1_a_required_column_cannot_extend_the_identity() -> None:
    """A1: las seis columnas obligatorias ya forman la identidad base."""
    with pytest.raises(ValueError, match="obligatoria"):
        _layout("series_id")


def test_a1_a_repeated_column_is_rejected() -> None:
    """A1: una columna repetida es una declaración ambigua."""
    with pytest.raises(ValueError, match="repetida"):
        _layout("headline_hash", "headline_hash")


# ─────────────────────────────────────────────────────────────────────────────
# A2 · Solo noticias amplía su identidad, y sin subir el layout
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_only_the_news_dataset_extends_its_identity() -> None:
    """A2: `news_headlines` declara `headline_hash`; ningún otro declara nada."""
    news = layout_of("raw", "news_headlines")
    assert news is not None
    assert news.identity_columns == ("headline_hash",)

    declaring = {key for key, layout in LAYOUTS.items() if layout.identity_columns}
    assert declaring == {("raw", "news_headlines")}


def test_a2_the_layout_version_is_not_bumped() -> None:
    """A2: ampliar la identidad no cambia el Parquet, así que no hay migración."""
    news = layout_of("raw", "news_headlines")
    assert news is not None
    assert news.layout_version == 1
    assert "headline_hash" in news.signature


# ─────────────────────────────────────────────────────────────────────────────
# A3 · La identidad incluye las columnas declaradas
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_the_identity_key_grows_with_the_declared_columns() -> None:
    """A3: dos valores distintos de la columna extra dan dos identidades."""
    from cfdtrader.data.store import _identity_key

    base = _identity_key("rss", "cnbc", PUB)
    with_a = _identity_key("rss", "cnbc", PUB, (_HASH_A,))
    with_b = _identity_key("rss", "cnbc", PUB, (_HASH_B,))

    assert base != with_a
    assert with_a != with_b
    assert len(base) == 3
    assert len(with_a) == 4


# ─────────────────────────────────────────────────────────────────────────────
# A4 · La ventana de revisión vigente se construye por dataset
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_the_current_row_window_partitions_by_the_full_identity() -> None:
    """A4: sin columnas extra la ventana es la clásica; con ellas, la completa."""
    from cfdtrader.data.store import _current_row_window

    assert (
        _current_row_window()
        == "row_number() OVER (PARTITION BY source, series_id, as_of ORDER BY version DESC)"
    )
    assert (
        _current_row_window(("headline_hash",))
        == "row_number() OVER (PARTITION BY source, series_id, as_of, headline_hash "
        "ORDER BY version DESC)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# A5/A6/A8/A9/A10 · El caso que abortaba, extremo a extremo sobre el almacén
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_two_headlines_of_the_same_feed_and_instant_are_two_rows(tmp_path: Path) -> None:
    """A5: mismo `source`/`series_id`/`as_of`, distinto `headline_hash` ⇒ dos filas."""
    store = Store(tmp_path)
    first = _headline(headline_hash=_HASH_A, title="Titular A", url="https://example.invalid/a")
    second = _headline(headline_hash=_HASH_B, title="Titular B", url="https://example.invalid/b")

    assert store.append("raw", "news_headlines", [first, second]) is WriteOutcome.CREATED

    frame = store.sql("SELECT headline_hash FROM raw.news_headlines ORDER BY headline_hash")
    assert frame.get_column("headline_hash").to_list() == [_HASH_A, _HASH_B]


def test_a6_a_batch_that_repeats_the_full_identity_is_rejected(tmp_path: Path) -> None:
    """A6: repetir la identidad **completa** sigue siendo un lote inválido."""
    store = Store(tmp_path)
    with pytest.raises(InvalidRecordError, match="identidad"):
        store.append("raw", "news_headlines", [_headline(), _headline()])


def test_a7_a_missing_or_empty_identity_column_is_a_typed_error(tmp_path: Path) -> None:
    """A7: la columna de identidad no puede faltar ni quedar vacía; el error la nombra."""
    store = Store(tmp_path)

    missing = _headline()
    del missing["headline_hash"]
    with pytest.raises(InvalidRecordError, match="headline_hash"):
        store.append("raw", "news_headlines", missing)

    with pytest.raises(InvalidRecordError, match="headline_hash"):
        store.append("raw", "news_headlines", _headline(headline_hash=None))

    with pytest.raises(InvalidRecordError, match="headline_hash"):
        store.append("raw", "news_headlines", _headline(headline_hash="  "))


def test_a8_rewriting_the_same_batch_is_idempotent(tmp_path: Path) -> None:
    """A8: reescribir el mismo lote es `UNCHANGED` y no duplica filas."""
    store = Store(tmp_path)
    batch = [_headline(headline_hash=_HASH_A), _headline(headline_hash=_HASH_B)]

    assert store.append("raw", "news_headlines", batch) is WriteOutcome.CREATED
    assert store.append("raw", "news_headlines", batch) is WriteOutcome.UNCHANGED

    assert store.sql("SELECT count(*) AS n FROM raw.news_headlines").item(0, "n") == 2


def test_a9_different_content_on_the_same_identity_is_still_immutable(tmp_path: Path) -> None:
    """A9: la identidad ampliada no relaja la inmutabilidad de `raw`."""
    store = Store(tmp_path)
    store.append("raw", "news_headlines", _headline(title="Titular A"))

    with pytest.raises(ImmutableWriteError):
        store.append("raw", "news_headlines", _headline(title="Otro titular"))


def test_a10_read_pit_returns_one_row_per_full_identity(tmp_path: Path) -> None:
    """A10: `read_pit` ve las dos identidades, no una."""
    store = Store(tmp_path)
    store.append(
        "raw",
        "news_headlines",
        [_headline(headline_hash=_HASH_A), _headline(headline_hash=_HASH_B)],
    )

    assert len(store.read_pit("raw", "news_headlines", at=FETCHED)) == 2


# ─────────────────────────────────────────────────────────────────────────────
# A11 · Sin identidad ampliada, nada cambia (regresión)
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_a_dataset_without_identity_columns_behaves_as_before(tmp_path: Path) -> None:
    """A11: `raw.macro` sigue con una fila por `(source, series_id, as_of)`."""
    store = Store(tmp_path)
    first = _macro(value=308.4)
    second = _macro(as_of=date(2024, 2, 1), value=310.0)

    assert store.append("raw", "macro", [first, second]) is WriteOutcome.CREATED
    assert store.sql("SELECT count(*) AS n FROM raw.macro").item(0, "n") == 2

    store.append_revision("raw", "macro", _macro(value=308.9))
    frame = store.sql("SELECT series_id, value, version FROM raw.macro ORDER BY as_of")
    assert frame.get_column("version").to_list() == [2, 1]
    assert frame.get_column("value").to_list() == [308.9, 310.0]


# ─────────────────────────────────────────────────────────────────────────────
# A12 · El camino de la ingesta (el caso de `google-news`, sin red)
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_the_news_ingest_writes_same_instant_headlines_of_one_feed(tmp_path: Path) -> None:
    """A12: la ingesta escribe dos titulares del mismo feed al mismo instante."""
    store = Store(tmp_path)
    headlines = [
        Headline(source="rss", feed="gnews", title="Uno", url="https://x/1", published_at=PUB),
        Headline(source="rss", feed="gnews", title="Dos", url="https://x/2", published_at=PUB),
        Headline(source="rss", feed="gnews", title="Tres", url="https://x/3", published_at=PUB),
    ]

    report = news_ingest(store=store, headlines=headlines, now=FETCHED)

    assert report.outcome == "created"
    assert report.new == 3
    assert store.sql("SELECT count(*) AS n FROM raw.news_headlines").item(0, "n") == 3
