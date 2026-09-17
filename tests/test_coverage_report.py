"""Tests del informe de cobertura y del bloqueo comprobable (tarea #3).

El informe es el artefacto verificable de la tarea: se comprueba que **no es
optimista** (A16), que declara el bloqueo de la Fase 1 (A21) y que separa diario
e intradía declarando el límite rodante del proveedor (A18).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from cfdtrader.data.coverage import (
    BLOCKER_CFD_MISSING,
    CoverageReport,
    build_report,
    render_markdown,
    write_report,
)
from cfdtrader.data.market import EXIT_NOT_READY, EXIT_OK, ingest, main
from cfdtrader.data.sources.base import (
    AssetClass,
    FetchRequest,
    FetchResult,
    SeriesSpec,
    SourceAdapter,
    SourceStatus,
)
from cfdtrader.data.sources.frames import empty_canonical_frame
from cfdtrader.data.sources.registry import SeriesRegistry, UnavailableSeries
from cfdtrader.data.store import Store

FRIDAY = datetime(2024, 6, 7, 20, 0, tzinfo=UTC)
NOW = datetime(2024, 6, 10, 18, 0, tzinfo=UTC)

#: Serie sin fuente, como la del registro real: el CFD del proyecto.
CFD_ENTRY = UnavailableSeries(
    series_id="SPX500:CFD",
    status="unavailable",
    bid_ask=False,
    reason="no hay fuente pública del intradía ni del bid/ask del SPX500:CFD",
    checked_on=datetime(2026, 9, 17, tzinfo=UTC).date(),
    follow_up_issue=50,
    documentation="_docs/data_sources.md",
)


def _spec(
    series_id: str = "^GSPC",
    *,
    dataset: str = "market_daily",
    granularity: str = "daily",
    interval: str = "1d",
    history_window_limit_days: int | None = None,
    lookback_period: str | None = None,
    min_start: date | None = None,
) -> SeriesSpec:
    """Especificación de serie con la fuente `fake`."""
    return SeriesSpec(
        series_id=series_id,
        dataset=dataset,
        asset_class=AssetClass.INDEX,
        granularity=granularity,
        interval=interval,
        primary="fake",
        min_start=min_start,
        lookback_period=lookback_period,
        history_window_limit_days=history_window_limit_days,
    )


def _frame(as_of: list[datetime]) -> pl.DataFrame:
    """Frame canónico mínimo con esas marcas temporales."""
    return pl.DataFrame(
        {
            "as_of": as_of,
            "open": [100.0] * len(as_of),
            "high": [105.0] * len(as_of),
            "low": [95.0] * len(as_of),
            "close": [101.0] * len(as_of),
            "volume": [1000.0] * len(as_of),
            "adj_close": [101.0] * len(as_of),
            "bid": [None] * len(as_of),
            "ask": [None] * len(as_of),
        },
        schema_overrides={"as_of": pl.Datetime("us", "UTC")},
    )


class FakeAdapter(SourceAdapter):
    """Adaptador doble: entrega el frame preparado para esa serie."""

    name = "fake"

    def __init__(self, frames: dict[str, pl.DataFrame] | None = None) -> None:
        self._frames = frames or {}

    def fetch(self, request: FetchRequest) -> FetchResult:
        frame = self._frames.get(request.spec.series_id)
        if frame is None:
            return FetchResult(
                spec=request.spec,
                source=self.name,
                status=SourceStatus.UNAVAILABLE,
                frame=empty_canonical_frame(),
                error="la fuente doble no tiene datos para esa serie",
            )
        return FetchResult(
            spec=request.spec,
            source=self.name,
            status=SourceStatus.OK,
            frame=frame,
            attempts=2,
        )


def _run(
    tmp_path: Path, *, registry: SeriesRegistry, frames: dict[str, pl.DataFrame]
) -> CoverageReport:
    """Ejecuta la ingesta con el adaptador doble y el informe en ``tmp_path``."""
    return ingest(
        registry=registry,
        data_root=tmp_path,
        adapters={"fake": FakeAdapter(frames)},
        now=NOW,
        reports_dir=tmp_path / "derived" / "reports",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Estructura y contenido del informe (A15, A16, A18)
# ─────────────────────────────────────────────────────────────────────────────
def test_report_has_one_row_per_series_and_source(tmp_path: Path) -> None:
    """A15: una fila por (serie, fuente) con los campos declarados."""
    registry = SeriesRegistry(
        version=1,
        series=(_spec("^GSPC"), _spec("SPY")),
        unavailable=(CFD_ENTRY,),
    )
    report = _run(
        tmp_path,
        registry=registry,
        frames={"^GSPC": _frame([FRIDAY]), "SPY": _frame([FRIDAY])},
    )

    assert len(report.daily) == 2
    row = report.daily[0]
    assert (row.series_id, row.source) == ("^GSPC", "fake")
    assert row.status is SourceStatus.OK
    assert row.attempts == 2
    assert row.rows_written == 1
    assert row.rows_new == 1
    assert row.span_start is not None and row.span_start.startswith("2024-06-07")
    assert row.span_end is not None
    assert row.granularity == "daily"
    assert row.timezone == "UTC"
    assert row.bid_ask in {"yes", "no", "na"}
    assert row.stale is False
    assert row.gaps == ()
    assert row.span_ok is True
    assert row.notes


def test_span_ok_uses_the_first_business_day_not_the_calendar_date(tmp_path: Path) -> None:
    """A17: `min_start` es una fecha de calendario y la serie guarda sesiones.

    2005-01-01 fue sábado **y** festivo, así que exigir que la primera barra sea
    de ese día es imposible: `^GSPC` decía `span_ok: no` con la ventana completa,
    y el indicador no podía valer `true` para ninguna serie con ese `min_start`.
    El listón pasa a ser el primer día laborable (2005-01-03, el día que abrió el
    mercado). Una serie cuyo primer dato es **posterior** sigue siendo `false`,
    con las fechas reales.
    """
    complete = SeriesRegistry(version=1, series=(_spec(min_start=date(2005, 1, 1)),))
    report = _run(
        tmp_path / "a",
        registry=complete,
        frames={"^GSPC": _frame([datetime(2005, 1, 3, 21, 0, tzinfo=UTC)])},
    )
    assert report.daily[0].span_ok is True

    late = SeriesRegistry(version=1, series=(_spec("XLRE", min_start=date(2015, 10, 1)),))
    late_report = _run(
        tmp_path / "b",
        registry=late,
        frames={"XLRE": _frame([datetime(2015, 10, 8, 20, 0, tzinfo=UTC)])},
    )
    assert late_report.daily[0].span_ok is False
    assert late_report.daily[0].span_start is not None
    assert late_report.daily[0].span_start.startswith("2015-10-08")

    weekend = SeriesRegistry(version=1, series=(_spec(min_start=date(2005, 1, 2)),))
    weekend_report = _run(
        tmp_path / "c",
        registry=weekend,
        frames={"^GSPC": _frame([datetime(2005, 1, 3, 21, 0, tzinfo=UTC)])},
    )
    assert weekend_report.daily[0].span_ok is True

    no_floor = SeriesRegistry(version=1, series=(_spec(min_start=None),))
    no_floor_report = _run(
        tmp_path / "d",
        registry=no_floor,
        frames={"^GSPC": _frame([datetime(2005, 1, 3, 21, 0, tzinfo=UTC)])},
    )
    assert no_floor_report.daily[0].span_ok is True


def test_rows_written_matches_the_store_and_ok_never_has_zero_rows(tmp_path: Path) -> None:
    """A16: nada figura `ok` sin filas, y `rows_written` cuadra con el almacén."""
    registry = SeriesRegistry(version=1, series=(_spec("^GSPC"),))
    report = _run(
        tmp_path,
        registry=registry,
        frames={"^GSPC": _frame([FRIDAY])},
    )
    store = Store(tmp_path)

    for row in report.daily:
        # Los valores vienen del informe que acabamos de construir, no de fuera:
        # no hay entrada de usuario en esta consulta de comprobación.
        query = (
            "SELECT count(*) AS n FROM raw.market_daily "  # noqa: S608
            f"WHERE source = '{row.source}' "
            f"AND series_id = '{row.series_id}'"
        )
        stored = store.sql(query).item(0, "n")
        assert row.rows_written == stored
        if row.status is SourceStatus.OK:
            assert row.rows_written > 0


def test_intraday_declares_the_rolling_window_limit(tmp_path: Path) -> None:
    """A18: el intradía declara la ventana y el límite del proveedor, sin inventar umbrales."""
    registry = SeriesRegistry(
        version=1,
        series=(
            _spec(
                "^GSPC",
                dataset="market_intraday",
                granularity="intraday",
                interval="5m",
                lookback_period="60d",
                history_window_limit_days=60,
            ),
        ),
    )
    report = _run(
        tmp_path,
        registry=registry,
        frames={"^GSPC": _frame([datetime(2024, 6, 10, 14, 0, tzinfo=UTC)])},
    )

    row = report.intraday[0]
    assert row.history_window_limit_days == 60
    assert row.history_window_limited is True
    assert report.daily == ()


def test_report_is_written_as_json_and_markdown(tmp_path: Path) -> None:
    """A14: el informe queda escrito en `derived/reports/` en JSON y Markdown."""
    registry = SeriesRegistry(version=1, series=(_spec("^GSPC"),), unavailable=(CFD_ENTRY,))
    report = _run(tmp_path, registry=registry, frames={"^GSPC": _frame([FRIDAY])})

    json_path, markdown_path = write_report(report, tmp_path / "derived" / "reports")

    assert json_path.name == "market_coverage_2024-06-10.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["phase1_ready"] is False
    assert payload["daily"][0]["rows_written"] == 1
    assert payload["unavailable"][0]["series_id"] == "SPX500:CFD"

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Diario" in markdown and "## Intradía" in markdown
    assert f"BLOQUEO `{BLOCKER_CFD_MISSING}`" in markdown
    assert "#50" in markdown


def test_intraday_and_daily_are_separated_in_the_markdown() -> None:
    """A18: el informe separa diario e intradía, también en Markdown."""
    report = build_report(
        outcomes=(),
        unavailable=(CFD_ENTRY,),
        now=NOW,
        data_root=Path("data"),
    )
    markdown = render_markdown(report)
    assert "## Diario" in markdown
    assert "_Sin series en esta granularidad._" in markdown


# ─────────────────────────────────────────────────────────────────────────────
# Bloqueo de la Fase 1 (A21)
# ─────────────────────────────────────────────────────────────────────────────
def test_phase1_is_blocked_without_a_cfd_source(tmp_path: Path) -> None:
    """A21: con datos solo del índice, `phase1_ready` es False y el código lo dice."""
    registry = SeriesRegistry(version=1, series=(_spec("^GSPC"),), unavailable=(CFD_ENTRY,))
    report = _run(tmp_path, registry=registry, frames={"^GSPC": _frame([FRIDAY])})

    assert report.phase1_ready is False
    assert [blocker.code for blocker in report.blockers] == [BLOCKER_CFD_MISSING]
    blocker = report.blockers[0]
    assert blocker.series_id == "SPX500:CFD"
    assert blocker.follow_up_issue == 50


def test_require_ready_exits_with_two_but_still_writes_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A21: sin el flag sale 0 y el informe queda escrito; con el flag, salida 2."""
    registry = SeriesRegistry(version=1, series=(_spec("^GSPC"),), unavailable=(CFD_ENTRY,))
    adapters: dict[str, SourceAdapter] = {"fake": FakeAdapter({"^GSPC": _frame([FRIDAY])})}

    def fake_build_adapters(**kwargs: object) -> dict[str, SourceAdapter]:
        return adapters

    def fake_load_registry(path: Path | str | None = None) -> SeriesRegistry:
        return registry

    monkeypatch.setattr("cfdtrader.data.market.build_adapters", fake_build_adapters)
    monkeypatch.setattr("cfdtrader.data.market.load_registry", fake_load_registry)

    reports = tmp_path / "derived" / "reports"
    assert main(["--data-root", str(tmp_path), "--now", NOW.isoformat()]) == EXIT_OK
    assert list(reports.glob("market_coverage_*.json"))

    assert (
        main(["--data-root", str(tmp_path), "--now", NOW.isoformat(), "--require-ready"])
        == EXIT_NOT_READY
    )
    assert EXIT_NOT_READY == 2
