"""Tests de la ingesta de mercado y del control de calidad (tarea #3).

Todo se escribe bajo ``tmp_path``: ningún test toca el ``data/`` del repositorio
ni abre red (el adaptador se inyecta con un doble).
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from cfdtrader.data.coverage import CoverageReport, SeriesRow
from cfdtrader.data.market import PAYLOAD_COLUMNS, ingest
from cfdtrader.data.quality import (
    CODE_DUPLICATE,
    CODE_GAP,
    CODE_NEGATIVE_VOLUME,
    CODE_NULL_OHLC,
    CODE_NULL_VOLUME,
    CODE_OHLC_INCOHERENT,
    CODE_OUT_OF_WINDOW,
    CODE_STALE,
    compare_sources,
    validate,
)
from cfdtrader.data.sources.base import (
    AssetClass,
    FetchRequest,
    FetchResult,
    SeriesSpec,
    SourceAdapter,
    SourceStatus,
)
from cfdtrader.data.sources.frames import empty_canonical_frame
from cfdtrader.data.sources.registry import SeriesRegistry
from cfdtrader.data.store import Store

#: 2024-06-07 (viernes) y 2024-06-10 (lunes) cierran a las 20:00 UTC en EDT.
FRIDAY = datetime(2024, 6, 7, 20, 0, tzinfo=UTC)
MONDAY = datetime(2024, 6, 10, 20, 0, tzinfo=UTC)

#: 14:00 ET del lunes: la sesión del día sigue abierta.
MONDAY_BEFORE_CLOSE = datetime(2024, 6, 10, 18, 0, tzinfo=UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def _frame(
    rows: list[tuple[datetime, float, float, float, float, float | None]],
    *,
    interval: str | None = None,
) -> pl.DataFrame:
    """Frame canónico a partir de ``(as_of, open, high, low, close, volume)``."""
    data: dict[str, list[object]] = {
        "as_of": [row[0] for row in rows],
        "open": [row[1] for row in rows],
        "high": [row[2] for row in rows],
        "low": [row[3] for row in rows],
        "close": [row[4] for row in rows],
        "volume": [row[5] for row in rows],
        "adj_close": [row[4] for row in rows],
        "bid": [None] * len(rows),
        "ask": [None] * len(rows),
    }
    if interval is not None:
        data["interval"] = [interval] * len(rows)
    return pl.DataFrame(
        data,
        schema_overrides={
            "as_of": pl.Datetime("us", "UTC"),
            "open": pl.Float64(),
            "high": pl.Float64(),
            "low": pl.Float64(),
            "close": pl.Float64(),
            "volume": pl.Float64(),
            "adj_close": pl.Float64(),
            "bid": pl.Float64(),
            "ask": pl.Float64(),
        },
    )


def _bar(
    as_of: datetime, close: float, *, volume: float | None = 1000.0
) -> tuple[datetime, float, float, float, float, float | None]:
    """Una barra coherente alrededor de ese cierre."""
    return (as_of, close - 5.0, close + 5.0, close - 10.0, close, volume)


def _spec(
    series_id: str = "^TEST",
    *,
    dataset: str = "market_daily",
    asset_class: AssetClass = AssetClass.INDEX,
    granularity: str = "daily",
    interval: str = "1d",
) -> SeriesSpec:
    """Especificación de serie para los tests, con la fuente `fake`."""
    return SeriesSpec(
        series_id=series_id,
        dataset=dataset,
        asset_class=asset_class,
        granularity=granularity,
        interval=interval,
        primary="fake",
        volume_expected=asset_class is not AssetClass.INDEX,
    )


def _registry(*specs: SeriesSpec) -> SeriesRegistry:
    """Registro mínimo con las series de test y sin series sin fuente."""
    return SeriesRegistry(version=1, series=tuple(specs), unavailable=())


class FakeAdapter(SourceAdapter):
    """Adaptador doble: devuelve el frame preparado y declara el estado pedido."""

    name = "fake"

    def __init__(
        self,
        frames: dict[str, pl.DataFrame] | None = None,
        *,
        attempts: int = 1,
    ) -> None:
        self._frames = frames or {}
        self._attempts = attempts

    def fetch(self, request: FetchRequest) -> FetchResult:
        frame = self._frames.get(request.spec.series_id)
        if frame is None:
            return FetchResult(
                spec=request.spec,
                source=self.name,
                status=SourceStatus.UNAVAILABLE,
                frame=empty_canonical_frame(),
                attempts=self._attempts,
                error="la fuente doble no tiene datos para esa serie",
            )
        return FetchResult(
            spec=request.spec,
            source=self.name,
            status=SourceStatus.OK,
            frame=frame,
            attempts=self._attempts,
        )


def _row(report: CoverageReport, series_id: str) -> SeriesRow:
    """La fila del informe de esa serie."""
    for row in (*report.daily, *report.intraday):
        if row.series_id == series_id:
            return row
    raise AssertionError(f"{series_id!r} no aparece en el informe")


# ─────────────────────────────────────────────────────────────────────────────
# Calidad (A12, A13)
# ─────────────────────────────────────────────────────────────────────────────
def test_duplicate_identities_are_detected_and_rejected() -> None:
    """A12(a): dos filas duplicadas ⇒ dos rechazadas, ninguna escrita."""
    spec = _spec()
    frame = _frame([_bar(FRIDAY, 100.0), _bar(FRIDAY, 100.0)])

    report = validate(frame, spec=spec, source="fake", now=MONDAY)

    issue = report.issue(CODE_DUPLICATE)
    assert issue is not None
    assert issue.rows == 2
    assert report.accepted.height == 0
    assert report.rejected_rows == 2


def test_missing_sessions_are_listed_with_their_dates() -> None:
    """A12(g): tres sesiones ausentes ⇒ tres huecos, con sus fechas exactas."""
    spec = _spec()
    sessions = [
        datetime(2024, 6, 3, 20, 0, tzinfo=UTC),  # lunes
        datetime(2024, 6, 4, 20, 0, tzinfo=UTC),  # martes
        # faltan miércoles 5, jueves 6 y viernes 7
        datetime(2024, 6, 10, 20, 0, tzinfo=UTC),  # lunes siguiente
    ]
    frame = _frame([_bar(day, 100.0 + index) for index, day in enumerate(sessions)])

    report = validate(frame, spec=spec, source="fake", now=MONDAY)

    assert report.gaps == ("2024-06-05", "2024-06-06", "2024-06-07")
    gap_issue = report.issue(CODE_GAP)
    assert gap_issue is not None
    # Descriptivo, no autoritativo: no se afirma que sean festivos.
    assert "no se afirma que sean festivos" in gap_issue.detail


def test_repeated_close_marks_stale_with_exact_dates() -> None:
    """A12(f): `close` idéntico en 5 sesiones seguidas ⇒ `stale` con las fechas."""
    spec = _spec()
    sessions = [
        datetime(2024, 6, 3, 20, 0, tzinfo=UTC),
        datetime(2024, 6, 4, 20, 0, tzinfo=UTC),
        datetime(2024, 6, 5, 20, 0, tzinfo=UTC),
        datetime(2024, 6, 6, 20, 0, tzinfo=UTC),
        datetime(2024, 6, 7, 20, 0, tzinfo=UTC),
    ]
    frame = _frame([_bar(day, 5000.0) for day in sessions])

    report = validate(frame, spec=spec, source="fake", now=MONDAY)

    assert report.stale is True
    assert report.stale_dates == tuple(day.date().isoformat() for day in sessions)
    assert report.issue(CODE_STALE) is not None
    # El dato congelado se marca, no se borra: el informe existe para investigarlo.
    assert report.accepted.height == 5


def test_null_and_incoherent_ohlc_are_rejected() -> None:
    """A12(b)(c): OHLC nulo y `high < max(open, close)` se rechazan con sus fechas."""
    spec = _spec()
    frame = pl.DataFrame(
        {
            "as_of": [FRIDAY, MONDAY],
            "open": [100.0, 100.0],
            "high": [105.0, 95.0],  # la segunda fila es incoherente: 95 < max(100, 101)
            "low": [95.0, 90.0],
            "close": [None, 101.0],  # la primera no trae cierre
            "volume": [1000.0, 1000.0],
            "adj_close": [None, 101.0],
            "bid": [None, None],
            "ask": [None, None],
        },
        schema_overrides={"as_of": pl.Datetime("us", "UTC")},
    )

    report = validate(frame, spec=spec, source="fake", now=MONDAY)

    assert report.issue(CODE_NULL_OHLC) is not None
    assert report.issue(CODE_OHLC_INCOHERENT) is not None
    assert report.accepted.height == 0
    assert report.rejected_rows == 2


def test_volume_rules_depend_on_the_asset_class() -> None:
    """A12(d): volumen negativo siempre fuera; volumen nulo solo en índices."""
    negative = _frame([_bar(FRIDAY, 100.0, volume=-1.0)])
    assert (
        validate(negative, spec=_spec(), source="fake", now=MONDAY).issue(CODE_NEGATIVE_VOLUME)
        is not None
    )

    null_volume = _frame([_bar(FRIDAY, 100.0, volume=None)])
    as_index = validate(null_volume, spec=_spec(), source="fake", now=MONDAY)
    assert as_index.accepted.height == 1
    assert as_index.issue(CODE_NULL_VOLUME) is None

    as_etf = validate(
        null_volume, spec=_spec("XLK", asset_class=AssetClass.ETF), source="fake", now=MONDAY
    )
    assert as_etf.issue(CODE_NULL_VOLUME) is not None
    assert as_etf.accepted.height == 0


def test_intraday_bar_outside_the_cash_window_is_rejected() -> None:
    """A12(e): una barra de las 08:00 ET queda fuera de la ventana 09:30–16:00."""
    spec = _spec("^TEST", dataset="market_intraday", granularity="intraday", interval="5m")
    before_open = datetime(2024, 6, 10, 12, 0, tzinfo=UTC)  # 08:00 ET
    morning = datetime(2024, 6, 10, 14, 0, tzinfo=UTC)  # 10:00 ET
    close = datetime(2024, 6, 10, 20, 0, tzinfo=UTC)  # 16:00 ET
    frame = _frame([_bar(before_open, 100.0), _bar(morning, 101.0), _bar(close, 102.0)])

    report = validate(frame, spec=spec, source="fake", now=MONDAY)

    issue = report.issue(CODE_OUT_OF_WINDOW)
    assert issue is not None and issue.rows == 1
    assert report.accepted.height == 2


def test_validation_is_deterministic_and_never_reads_the_clock() -> None:
    """A13: mismo `now` ⇒ mismo veredicto, y ninguna llamada interna al reloj."""
    spec = _spec()
    frame = _frame([_bar(FRIDAY, 100.0), _bar(MONDAY, 101.0)])

    first = validate(frame, spec=spec, source="fake", now=MONDAY)
    second = validate(frame, spec=spec, source="fake", now=MONDAY)

    assert first.accepted.equals(second.accepted)
    assert first.gaps == second.gaps
    assert first.checked_at == second.checked_at

    source = Path("src/cfdtrader/data/quality.py").read_text(encoding="utf-8")
    # La comprobación es sobre el AST, no sobre el texto: el docstring *habla* de
    # `datetime.now()`, y eso no es una llamada.
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "now"
    ]
    assert calls == []


def test_cross_source_comparison_only_flags_the_same_logical_series() -> None:
    """A12(h): misma serie lógica con >10 pb de diferencia ⇒ incoherencia declarada."""
    left = _frame([_bar(FRIDAY, 100.0), _bar(MONDAY, 200.0)])
    right = _frame([_bar(FRIDAY, 100.0), _bar(MONDAY, 200.4)])  # 20 pb

    issue = compare_sources(
        left, right, series_id="^GSPC", left_source="yfinance", right_source="stooq"
    )

    assert issue is not None
    assert issue.dates == (str(MONDAY),)
    assert "misma serie ^GSPC" in issue.detail

    identical = compare_sources(
        left, left, series_id="^GSPC", left_source="yfinance", right_source="stooq"
    )
    assert identical is None


# ─────────────────────────────────────────────────────────────────────────────
# Ingesta (A7, A10, A11)
# ─────────────────────────────────────────────────────────────────────────────
def test_open_session_is_discarded_and_counted(tmp_path: Path) -> None:
    """A11: la barra diaria de hoy, sin cerrar, no se escribe y se declara."""
    spec = _spec()
    frame = _frame([_bar(FRIDAY, 5000.0), _bar(MONDAY, 5010.0)])
    report = ingest(
        registry=_registry(spec),
        data_root=tmp_path,
        adapters={"fake": FakeAdapter({"^TEST": frame})},
        now=MONDAY_BEFORE_CLOSE,
    )

    outcome = _row(report, "^TEST")
    assert outcome.status is SourceStatus.OK
    assert outcome.discarded_open_session == 1
    assert outcome.rows_written == 1
    assert outcome.span_start is not None and outcome.span_start.startswith("2024-06-07")


def test_second_identical_run_is_a_noop(tmp_path: Path) -> None:
    """A10: una segunda ejecución idéntica no escribe nada y no cambia el almacén."""
    spec = _spec()
    frame = _frame([_bar(FRIDAY, 5000.0), _bar(MONDAY, 5010.0)])
    adapters = {"fake": FakeAdapter({"^TEST": frame})}

    first = ingest(
        registry=_registry(spec), data_root=tmp_path, adapters=adapters, now=MONDAY_BEFORE_CLOSE
    )
    files_after_first = sorted((tmp_path / "raw").rglob("*.parquet"))
    second = ingest(
        registry=_registry(spec), data_root=tmp_path, adapters=adapters, now=MONDAY_BEFORE_CLOSE
    )

    assert _row(first, "^TEST").rows_new == 1
    assert _row(second, "^TEST").rows_new == 0
    assert _row(second, "^TEST").rows_written == 1
    assert any("unchanged" in note for note in _row(second, "^TEST").notes)
    assert sorted((tmp_path / "raw").rglob("*.parquet")) == files_after_first


def test_second_run_does_not_send_the_history_to_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La ejecución diaria no reenvía el histórico: solo lo nuevo y lo revisado.

    El almacén deduplica, así que reenviarlo todo es correcto pero carísimo: su
    comprobación de identidad es registro a registro. Con 5.461 filas por serie la
    ejecución diaria tardaba horas. Aquí se comprueba que la segunda vuelta no
    llama a `append` en absoluto, no solo que no cree ficheros.
    """
    spec = _spec()
    frame = _frame([_bar(FRIDAY, 5000.0), _bar(MONDAY, 5010.0)])
    adapters = {"fake": FakeAdapter({"^TEST": frame})}
    calls: list[str] = []
    original_append = Store.append
    original_revision = Store.append_revision

    def spy_append(self: Store, layer: str, dataset: str, records: object) -> object:
        calls.append("append")
        return original_append(self, layer, dataset, records)  # type: ignore[arg-type]

    def spy_revision(self: Store, layer: str, dataset: str, records: object) -> object:
        calls.append("append_revision")
        return original_revision(self, layer, dataset, records)  # type: ignore[arg-type]

    monkeypatch.setattr(Store, "append", spy_append)
    monkeypatch.setattr(Store, "append_revision", spy_revision)

    ingest(registry=_registry(spec), data_root=tmp_path, adapters=adapters, now=MONDAY_BEFORE_CLOSE)
    assert calls == ["append"], "la primera vuelta escribe las dos barras nuevas"
    calls.clear()
    ingest(registry=_registry(spec), data_root=tmp_path, adapters=adapters, now=MONDAY_BEFORE_CLOSE)

    assert calls == [], "la segunda vuelta no debe tocar el almacén"


def test_a_revised_bar_is_stored_as_version_two(tmp_path: Path) -> None:
    """A10: la fuente corrige un valor ⇒ `append_revision` y dos versiones en la historia."""
    spec = _spec()
    store = Store(tmp_path)
    first_now = MONDAY_BEFORE_CLOSE
    second_now = datetime(2024, 6, 11, 6, 0, tzinfo=UTC)
    original = _frame([_bar(FRIDAY, 5000.0)])
    revised = _frame([_bar(FRIDAY, 5005.0)])

    ingest(
        registry=_registry(spec),
        data_root=tmp_path,
        adapters={"fake": FakeAdapter({"^TEST": original})},
        now=first_now,
    )
    ingest(
        registry=_registry(spec),
        data_root=tmp_path,
        adapters={"fake": FakeAdapter({"^TEST": revised})},
        now=second_now,
    )

    early = store.read_pit("raw", "market_daily", at=first_now)
    late = store.read_pit("raw", "market_daily", at=second_now)

    assert early.get_column("version").to_list() == [1]
    assert early.get_column("close").to_list() == [5000.0]
    assert late.get_column("version").to_list() == [2]
    assert late.get_column("close").to_list() == [5005.0]
    # El estado consultable tiene una sola fila por identidad: la revisión vigente.
    assert store.sql("SELECT * FROM raw.market_daily").height == 1


def test_a_source_that_fails_does_not_create_an_empty_dataset(tmp_path: Path) -> None:
    """A7 y A14: una serie que no entrega datos no aparece con 0 filas, y no aborta."""
    good = _spec("^GOOD")
    bad = _spec("^BAD")
    report = ingest(
        registry=_registry(good, bad),
        data_root=tmp_path,
        adapters={"fake": FakeAdapter({"^GOOD": _frame([_bar(FRIDAY, 100.0)])})},
        now=MONDAY_BEFORE_CLOSE,
    )

    assert Store(tmp_path).datasets("raw") == ["market_daily"]
    assert _row(report, "^GOOD").rows_written == 1
    assert _row(report, "^BAD").status is SourceStatus.UNAVAILABLE
    assert _row(report, "^BAD").rows_written == 0


def test_payload_columns_match_the_dataset_schema() -> None:
    """A8: los datasets de mercado declaran sus columnas de payload, y coinciden."""
    assert set(PAYLOAD_COLUMNS) == {"market_daily", "sectors", "market_intraday"}
    assert "adj_close" in PAYLOAD_COLUMNS["market_daily"]
    assert "interval" in PAYLOAD_COLUMNS["market_intraday"]
    assert "bid" in PAYLOAD_COLUMNS["market_intraday"]
    assert "ask" in PAYLOAD_COLUMNS["market_intraday"]


def test_intraday_records_keep_published_at_null(tmp_path: Path) -> None:
    """A8: el `published_at` de una barra es NULL; nunca se rellena con `fetched_at`."""
    spec = _spec("^TEST", dataset="market_intraday", granularity="intraday", interval="5m")
    bar = datetime(2024, 6, 10, 14, 0, tzinfo=UTC)  # 10:00 ET
    ingest(
        registry=_registry(spec),
        data_root=tmp_path,
        adapters={"fake": FakeAdapter({"^TEST": _frame([_bar(bar, 100.0)], interval="5m")})},
        now=MONDAY,
    )

    row = (
        Store(tmp_path).sql("SELECT published_at, interval FROM raw.market_intraday").to_dicts()[0]
    )
    assert row["published_at"] is None
    assert row["interval"] == "5m"


def test_monkeypatched_adapters_are_the_only_way_this_module_talks_to_a_source() -> None:
    """A1: el registro y la orquestación no importan ninguna fuente directamente."""
    source = "".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "src/cfdtrader/data/market.py",
            "src/cfdtrader/data/coverage.py",
            "src/cfdtrader/data/sources/registry.py",
        )
    )
    assert "import yfinance" not in source
