"""Tests del productor real de ``derived.features_daily`` (#73).

Todo se mide sobre un almacen **temporal** (``tmp_path``): la fixture de sesion de
``tests/conftest.py`` huella el ``data/`` del repositorio y este modulo **escribe**.

Criterios: A1 el cargador de ``context_v1``, A2 las cinco matrices por familia, A3 el ancla
declarada, A4 las seis familias persistidas, A5 la CLI no lee el reloj, A6 ``--as-of``
obligatorio y cero ficheros, A7 no se publica el futuro, A8 procedencia por fila, A9
``fetched_at`` anterior al ``as_of`` es error, A10 repetir no escribe, A11 pureza.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import polars as pl
import pytest

from cfdtrader.analysis import feature_frame, features_daily
from cfdtrader.analysis.feature_frame import ANCHOR_SERIES, FAMILY_ORDER
from cfdtrader.data.store import Store, WriteOutcome
from cfdtrader.features import store as feature_store

VIX: Final[str] = "^VIX"
MODULE_PATH: Final[Path] = Path(features_daily.__file__)
FAMILIES: Final[tuple[str, ...]] = FAMILY_ORDER

#: Días que se materializan en el almacén temporal: suficiente para las ventanas de #19-#23.
HISTORY_DAYS: Final[int] = 40

#: Instante de la corrida, muy posterior a las sesiones sintéticas de 2026-01/02.
FETCHED_AT: Final[str] = "2026-09-20T00:00:00+00:00"

#: Sesión de referencia de la corrida: posterior a todas las sesiones sintéticas.
AS_OF: Final[str] = "2026-03-02T12:00:00-05:00"


def _market_records(
    days: Sequence[tuple[int, int, int]], *, series_id: str
) -> list[dict[str, object]]:
    """Filas diarias sinteticas de esa serie, con un ``open`` que no repite el cierre previo."""
    records: list[dict[str, object]] = []
    close = 100.0
    for index, (year, month, day) in enumerate(days):
        open_px = close * 1.0009
        close = open_px * 1.0004
        records.append(
            {
                "source": "yfinance",
                "series_id": series_id,
                "as_of": datetime(year, month, day, 21, tzinfo=UTC),
                "fetched_at": datetime(2026, 9, 20, tzinfo=UTC),
                "published_at": None,
                "open": open_px,
                "high": max(open_px, close) * 1.001,
                "low": min(open_px, close) * 0.999,
                "close": close,
                "volume": 1_000.0 + index,
            }
        )
    return records


def _label_records(days: Sequence[tuple[int, int, int]]) -> list[dict[str, object]]:
    """Etiquetas de #10: una fila por sesion, con el objetivo y la volatilidad prevista."""
    return [
        {
            "source": "cfdtrader.models.labels",
            "series_id": ANCHOR_SERIES,
            "as_of": datetime(year, month, day, 21, tzinfo=UTC),
            "fetched_at": datetime(2026, 9, 20, tzinfo=UTC),
            "published_at": None,
            "session": date(year, month, day),
            "ret_long": 0.001,
            "is_half_day": False,
            "k_sigma": 1.0,
        }
        for year, month, day in days
    ]


def _business_days(count: int, *, start: date = date(2026, 1, 5)) -> list[tuple[int, int, int]]:
    """Dias laborables consecutivos desde ``start``."""
    out: list[tuple[int, int, int]] = []
    current = start
    while len(out) < count:
        if current.weekday() < 5:
            out.append((current.year, current.month, current.day))
        current += timedelta(days=1)
    return out


def _build_store(root: Path, *, extra_sessions: int = 0) -> Store:
    """Almacen temporal con el diario del ancla, el `^VIX` y las etiquetas de #10."""
    store = Store(root)
    days = _business_days(HISTORY_DAYS)
    store.append(
        "raw",
        "market_daily",
        [*_market_records(days, series_id=ANCHOR_SERIES), *_market_records(days, series_id=VIX)],
    )
    store.append("derived", "labels", _label_records(days))
    for offset in range(extra_sessions):
        future = date(2026, 3, 2) + timedelta(days=offset)
        store.append(
            "raw",
            "market_daily",
            _market_records([(future.year, future.month, future.day)], series_id=ANCHOR_SERIES),
        )
    return store


def _fingerprint(root: Path) -> dict[str, str]:
    """Ruta relativa → tamano, para comprobar que la CLI no escribio nada."""
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): str(path.stat().st_size)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _cli(root: Path, *extra: str) -> list[str]:
    """Argumentos de la CLI con la raiz temporal delante."""
    return ["--data-root", str(root), *extra]


def _persisted(store: Store) -> pl.DataFrame:
    """**Todas** las filas del dataset, sin filtrar por familia (para mirar la procedencia)."""
    return store.sql(
        f"SELECT * FROM derived.{feature_store.FEATURES_DATASET} "  # noqa: S608 - nombre fijo
        "ORDER BY as_of, source"
    )


def _market_sessions(store: Store) -> list[date]:
    """Las sesiones del diario del ancla (para comprobar que la sesion hostil existe)."""
    frame = store.sql(
        "SELECT as_of FROM raw.market_daily "  # noqa: S608 - nombre fijo
        f"WHERE series_id = '{ANCHOR_SERIES}'"
    )
    return [instant.astimezone(UTC).date() for instant in frame.get_column("as_of").to_list()]


def _sessions(frame: pl.DataFrame) -> set[date]:
    """Las sesiones de las filas: el dia ET de su ``as_of``.

    El dataset **no** persiste ``session``: la fila lleva su ``as_of`` y la sesion se deriva de
    el (el mismo criterio que el adaptador).
    """
    derived = frame.get_column("as_of").dt.convert_time_zone("America/New_York").dt.date()
    return set(derived.to_list())


# ─────────────────────────────────────────────────────────────────────────────
# A1 · El cargador de `context_v1` (#73)
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_context_loader_reads_the_store_and_declares_what_is_missing(tmp_path: Path) -> None:
    store = _build_store(tmp_path / "almacen")
    present = feature_frame.load_context_inputs(store)

    assert set(present) == {ANCHOR_SERIES, VIX}, "solo lo que el almacen tiene"
    assert present[ANCHOR_SERIES].columns == ["session", "as_of", "close"]
    assert present[VIX].columns == ["session", "close"], "el `as_of` es del ancla"
    assert present[ANCHOR_SERIES].height == HISTORY_DAYS

    missing = feature_frame.missing_context_series(store)
    assert set(missing) == set(feature_store.CONTEXT_SERIES) - set(present)


def test_a1_the_loader_needs_the_anchor_and_invents_no_series(tmp_path: Path) -> None:
    """Sobre un almacen vacio el cargador falla con error tipado: no inventa ninguna serie."""
    with pytest.raises(feature_frame.MissingFeatureDatasetError, match="market_daily"):
        feature_frame.load_context_inputs(Store(tmp_path / "vacio"))


# ─────────────────────────────────────────────────────────────────────────────
# A2 · Las cinco matrices **por familia**
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_adapter_exposes_the_six_families_separately(tmp_path: Path) -> None:
    store = _build_store(tmp_path / "almacen")
    built = feature_frame.build_family_frames(store)

    assert set(built.frames) == set(FAMILIES)
    for family, family_frame in built.frames.items():
        assert family_frame.columns[0] == "session"
        assert set(feature_frame.COLUMNS_BY_FAMILY[family]) <= set(family_frame.columns)
        assert family_frame.height == HISTORY_DAYS
    assert built.instants.columns == ["session", "as_of"]
    assert len(built.missing_series) == 26, (
        "17 de contexto + 6 macro + 3 de commodities: las que no estan"
    )


# ─────────────────────────────────────────────────────────────────────────────
# A3 · El ancla y el `series_id` persistido
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_the_anchor_series_is_declared() -> None:
    assert ANCHOR_SERIES == "^GSPC"
    assert feature_frame.VIX_SERIES == "^VIX"


# ─────────────────────────────────────────────────────────────────────────────
# A4 · Las seis familias quedan en el dataset y se releen
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_the_cli_persists_the_six_families(tmp_path: Path) -> None:
    root = tmp_path / "almacen"
    store = _build_store(root)
    last_session = date(*_business_days(HISTORY_DAYS)[-1])

    assert features_daily.main(_cli(root, "--as-of", AS_OF, "--fetched-at", FETCHED_AT)) == 0

    for family in FAMILIES:
        frame = feature_store.load_daily(store, series_id=ANCHOR_SERIES, feature_set=family)
        assert frame.height == HISTORY_DAYS, family
        assert max(_sessions(frame)) == last_session, family
        assert frame.get_column("series_id").unique().to_list() == [ANCHOR_SERIES], family


# ─────────────────────────────────────────────────────────────────────────────
# A5 y A11 · La CLI no lee el reloj y la sesion es la de Nueva York
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_the_module_does_not_read_the_clock() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    for needle in ("datetime.now", "date.today", "time.time", "utcnow", "ZoneInfo"):
        assert needle not in source, needle


def test_a11_the_session_comes_from_new_york_and_not_from_utc() -> None:
    """Un instante de madrugada UTC pertenece a la sesion del dia **anterior** en ET."""
    assert features_daily.session_of(datetime(2026, 3, 3, 1, 30, tzinfo=UTC)) == date(2026, 3, 2)
    with pytest.raises(features_daily.InvalidInstantError, match="zona horaria"):
        features_daily.session_of(datetime(2026, 3, 2, 12, 0))


# ─────────────────────────────────────────────────────────────────────────────
# A6 · Sin los dos instantes: `rc=2` y **cero** ficheros
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_the_cli_refuses_without_the_instants_and_writes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "almacen"
    _build_store(root)
    before = _fingerprint(root)

    assert features_daily.main(_cli(root)) == 2, "sin ninguno"
    assert features_daily.main(_cli(root, "--as-of", AS_OF)) == 2, "sin --fetched-at"
    assert features_daily.main(_cli(root, "--fetched-at", FETCHED_AT)) == 2, "sin --as-of"

    assert _fingerprint(root) == before, "sin instantes no se escribe ni un fichero"


# ─────────────────────────────────────────────────────────────────────────────
# A7 · No se publica una sesion que no habia cerrado
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_no_session_after_the_reference_is_published(tmp_path: Path) -> None:
    root = tmp_path / "almacen"
    store = _build_store(root, extra_sessions=1)
    reference = features_daily.session_of(datetime.fromisoformat(AS_OF))
    assert reference == date(2026, 3, 2)
    assert reference in set(_market_sessions(store)), "la sesion hostil esta en el diario"

    assert features_daily.main(_cli(root, "--as-of", AS_OF, "--fetched-at", FETCHED_AT)) == 0

    sessions = _sessions(_persisted(store))
    assert max(sessions) < reference, (
        "la sesion del `--as-of` no habia cerrado en ese instante: no puede tener features"
    )


def test_a6_a_naive_instant_is_rejected_by_the_cli(tmp_path: Path) -> None:
    """Una hora sin zona no es un instante: la CLI la rechaza con su codigo 2."""
    with pytest.raises(SystemExit) as exit_info:
        features_daily.main(
            _cli(tmp_path / "almacen", "--as-of", "2026-03-02T12:00:00", "--fetched-at", FETCHED_AT)
        )
    assert exit_info.value.code == 2
    assert not (tmp_path / "almacen" / "derived").exists()


# ─────────────────────────────────────────────────────────────────────────────
# A8 · Cada fila lleva la procedencia de **su** familia
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_every_row_carries_its_family_provenance(tmp_path: Path) -> None:
    root = tmp_path / "almacen"
    store = _build_store(root)
    assert features_daily.main(_cli(root, "--as-of", AS_OF, "--fetched-at", FETCHED_AT)) == 0

    rows = _persisted(store)
    assert rows.height == HISTORY_DAYS * len(FAMILIES)
    checked = 0
    for family in FAMILIES:
        spec = feature_frame.family_spec(family)
        digest = feature_store.feature_spec_sha256(spec)
        for row in rows.filter(pl.col("feature_spec_sha256") == digest).iter_rows(named=True):
            assert row["features_version"] == feature_store.features_version(spec, row["as_of"])
            assert row["fetched_at"] == datetime.fromisoformat(FETCHED_AT)
            assert row["source"], "cada familia trae su `source`"
            checked += 1
    assert checked == rows.height, "toda fila se reconoce por la spec de su familia"
    assert rows.get_column("source").n_unique() == len(FAMILIES)


# ─────────────────────────────────────────────────────────────────────────────
# A9 · `fetched_at` anterior al `as_of` de una fila es error tipado
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_a_fetched_at_before_the_session_is_a_typed_error(tmp_path: Path) -> None:
    root = tmp_path / "almacen"
    store = _build_store(root)
    early = datetime(2026, 2, 27, 20, tzinfo=UTC)

    with pytest.raises(feature_store.InvalidFeatureMatrixError, match="anterior"):
        features_daily.persist_family_frames(
            store, fetched_at=early, sessions_before=date(2026, 3, 2)
        )

    before = _fingerprint(root)
    assert features_daily.main(_cli(root, "--as-of", AS_OF, "--fetched-at", early.isoformat())) == 2
    assert _fingerprint(root) == before, "un `fetched_at` imposible no escribe nada"


# ─────────────────────────────────────────────────────────────────────────────
# A10 · El dataset es *derived*: repetir no crea Parquet
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_repeating_the_run_writes_nothing_new(tmp_path: Path) -> None:
    root = tmp_path / "almacen"
    store = _build_store(root)
    assert features_daily.main(_cli(root, "--as-of", AS_OF, "--fetched-at", FETCHED_AT)) == 0
    before = _fingerprint(root)

    outcomes = features_daily.persist_family_frames(
        store,
        fetched_at=datetime.fromisoformat(FETCHED_AT),
        sessions_before=date(2026, 3, 2),
    )

    assert set(outcomes) == set(FAMILIES)
    assert set(outcomes.values()) == {WriteOutcome.UNCHANGED}
    assert _fingerprint(root) == before, "contenido identico: ni un Parquet nuevo"
