"""Candidatas históricas opt-in de #156; fixtures aisladas, sin red ni escritura en data/runs."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from cfdtrader.analysis import feature_budget, feature_frame
from cfdtrader.data.store import Store
from cfdtrader.features import historical_market
from cfdtrader.features.historical_market import (
    HISTORICAL_MARKET_CANDIDATE_CATALOG,
    HISTORICAL_MARKET_CANDIDATE_COLUMNS,
    HISTORICAL_MARKET_CANDIDATE_SET,
    HISTORICAL_MARKET_DESIGN_LAG_BY_FEATURE,
    HistoricalMarketDataError,
    build_historical_market_candidates,
)
from cfdtrader.models.baseline import BASELINE_FEATURES, design_frame

_DATES: Final[tuple[date, ...]] = (
    date(2026, 1, 5),
    date(2026, 1, 6),
    date(2026, 1, 8),
    date(2026, 1, 9),
)


def _bars(
    *,
    spx: tuple[tuple[float | None, float | None], ...] | None = None,
    es: dict[date, float | None] | None = None,
    include_source: bool = True,
) -> pl.DataFrame:
    """Crea barras compactas en fechas explícitas; ES puede tener otro calendario."""
    spx_values = spx or ((90.0, 100.0), (110.0, 120.0), (130.0, 140.0), (150.0, 160.0))
    es_values = dict.fromkeys(_DATES, 101.0) if es is None else es
    rows: list[dict[str, object]] = []
    for session, (open_px, close_px) in zip(_DATES, spx_values, strict=True):
        rows.append(
            {
                "series_id": "^GSPC",
                "session": session,
                "open": open_px,
                "close": close_px,
                **({"source": "yfinance"} if include_source else {}),
            }
        )
    for session, close_px in es_values.items():
        rows.append(
            {
                "series_id": "ES=F",
                "session": session,
                "open": close_px,
                "close": close_px,
                **({"source": "yfinance"} if include_source else {}),
            }
        )
    return pl.DataFrame(rows, strict=False)


def _stored_bar(
    series_id: str, session: date, open_px: float, close_px: float
) -> dict[str, object]:
    """Registro de mercado válido para el almacén aislado de pruebas."""
    close_in_utc = datetime.combine(
        session, time(16), tzinfo=ZoneInfo("America/New_York")
    ).astimezone(UTC)
    return {
        "source": "yfinance",
        "series_id": series_id,
        "as_of": close_in_utc,
        "fetched_at": datetime(2026, 10, 10, tzinfo=UTC),
        "published_at": None,
        "open": open_px,
        "high": max(open_px, close_px) + 1.0,
        "low": min(open_px, close_px) - 1.0,
        "close": close_px,
        "volume": 1000.0,
    }


def test_catalog_isolated_from_commodities_and_declares_exact_point_in_time_formulas() -> None:
    assert HISTORICAL_MARKET_CANDIDATE_SET == "historical_market_v1"
    assert HISTORICAL_MARKET_CANDIDATE_COLUMNS == ("es_basis", "spx_overnight_ret_1")
    assert tuple(entry.formula for entry in HISTORICAL_MARKET_CANDIDATE_CATALOG) == (
        "ES=F_close(t−1) / ^GSPC_close(t−1) − 1",  # noqa: RUF001
        "ln(^GSPC_open(t−1) / ^GSPC_close(t−2))",  # noqa: RUF001
    )
    assert HISTORICAL_MARKET_DESIGN_LAG_BY_FEATURE == {
        "es_basis": 0,
        "spx_overnight_ret_1": 0,
    }
    assert not set(HISTORICAL_MARKET_CANDIDATE_COLUMNS).intersection(
        feature_budget.CANDIDATE_FEATURES
    )
    assert feature_budget.CANDIDATE_FEATURES == (
        "oil_ret_1",
        "oil_ret_5",
        "oil_ret_1_z",
        "gold_ret_1",
        "eurusd_ret_1",
    )
    assert BASELINE_FEATURES == feature_budget.CONTROL_FEATURES


def test_values_align_to_gspc_calendar_and_shift_into_design_row_t() -> None:
    bars = _bars(es={_DATES[0]: 101.0, _DATES[2]: 142.0, _DATES[3]: 161.0})
    built = build_historical_market_candidates(bars)
    assert built.frame.get_column("session").to_list() == list(_DATES)

    basis = built.frame.get_column("es_basis").to_list()
    overnight = built.frame.get_column("spx_overnight_ret_1").to_list()
    assert basis[0] is None
    assert basis[1] == pytest.approx(0.01)
    assert basis[2] is None  # ES no tiene la sesión ^GSPC t-1: no se arrastra t-2.
    assert basis[3] == pytest.approx(142.0 / 140.0 - 1.0)
    assert overnight[0] is None
    assert overnight[1] is None
    assert overnight[2] == pytest.approx(math.log(110.0 / 100.0))
    assert overnight[3] == pytest.approx(math.log(130.0 / 120.0))

    labels = pl.DataFrame(
        {
            "session": list(_DATES),
            "ret_long": [0.1, -0.1, 0.2, -0.2],
        }
    )
    design = design_frame(
        built.frame,
        labels=labels,
        selected=HISTORICAL_MARKET_CANDIDATE_COLUMNS,
        availability=HISTORICAL_MARKET_DESIGN_LAG_BY_FEATURE,
    )
    assert dict(design.design_lag_by_feature) == {
        "es_basis": 0,
        "spx_overnight_ret_1": 0,
    }
    # design_lag=0 conserva el valor asignado a t, no el de t-1.
    assert design.frame.filter(pl.col("session") == _DATES[1])["es_basis"].item() == pytest.approx(
        0.01
    )


def test_missing_sessions_and_invalid_prices_remain_null_without_forward_fill() -> None:
    spx = ((90.0, 100.0), (0.0, 0.0), (0.0, 0.0), (150.0, 160.0))
    bars = _bars(spx=spx, es={_DATES[0]: 0.0, _DATES[2]: 142.0})
    built = build_historical_market_candidates(bars)
    basis = built.frame.get_column("es_basis").to_list()
    overnight = built.frame.get_column("spx_overnight_ret_1").to_list()
    assert basis[1] is None  # cierre ES t-1 inválido
    assert basis[2] is None  # ES no tiene la sesión previa exacta
    assert basis[3] is None  # cierre SPX t-1 inválido
    assert overnight[2] is None  # apertura SPX t-1 inválida
    assert overnight[3] is None  # cierre SPX t-2 inválido
    assert all(value is None or math.isfinite(value) for value in (*basis, *overnight))
    for audit in built.audit["candidate_rows"].values():  # type: ignore[union-attr]
        assert audit["valid_rows"] + audit["missing_or_invalid_rows"] == audit["rows"]
        assert audit["reconciles"] is True


def test_bad_prices_are_rejected_and_missing_series_is_not_invented() -> None:
    for value in (None, True, 0.0, -1.0, float("nan"), float("inf"), object(), "bad"):
        assert historical_market._valid_price(value) is None  # pyright: ignore[reportPrivateUsage]
    assert historical_market._valid_price(100.0) == 100.0  # pyright: ignore[reportPrivateUsage]

    only_spx = _bars(es={})
    built = build_historical_market_candidates(only_spx)
    assert built.frame.get_column("es_basis").null_count() == len(_DATES)
    input_audit = built.audit["input_series"]
    assert input_audit["ES=F"]["input_rows"] == 0  # type: ignore[index]
    assert input_audit["^GSPC"]["input_rows"] == len(_DATES)  # type: ignore[index]


def test_audit_uses_reference_date_and_reports_dynamic_ranges_and_reconciled_rows() -> None:
    sessions = (date(2026, 10, 8), date(2026, 10, 9), date(2026, 10, 12))
    rows = [
        {
            "series_id": series,
            "session": session,
            "open": 100.0,
            "close": 101.0,
            "source": source,
        }
        for session in sessions
        for series, source in (("^GSPC", "yfinance"), ("ES=F", "yfinance"))
    ]
    built = build_historical_market_candidates(pl.DataFrame(rows))
    assert built.audit["reference_snapshot_date"] == "2026-10-09"
    series_audit = built.audit["input_series"]
    assert series_audit["^GSPC"]["input_rows"] == 2  # type: ignore[index]
    assert series_audit["^GSPC"]["first_session"] == "2026-10-08"  # type: ignore[index]
    assert series_audit["^GSPC"]["last_session"] == "2026-10-09"  # type: ignore[index]
    assert built.audit["candidate_rows"]["es_basis"]["rows"] == 2  # type: ignore[index]
    assert built.audit["candidate_rows"]["spx_overnight_ret_1"]["reconciles"] is True  # type: ignore[index]


def test_duplicate_or_invalid_sessions_fail_instead_of_choosing_a_source() -> None:
    bars = _bars()
    duplicate = pl.concat([bars, bars.filter(pl.col("session") == _DATES[0]).head(1)])
    with pytest.raises(HistoricalMarketDataError, match="sesion duplicada"):
        build_historical_market_candidates(duplicate)

    missing_session = bars.with_columns(
        pl.when(pl.col("series_id") == "ES=F")
        .then(pl.lit(None, dtype=pl.Date()))
        .otherwise(pl.col("session"))
        .alias("session")
    )
    with pytest.raises(HistoricalMarketDataError, match="sesion ausente"):
        build_historical_market_candidates(missing_session)


def test_invalid_schema_and_empty_index_are_handled_explicitly() -> None:
    with pytest.raises(HistoricalMarketDataError, match="faltan columnas"):
        build_historical_market_candidates(pl.DataFrame({"series_id": ["^GSPC"]}))

    empty = pl.DataFrame(
        schema={
            "series_id": pl.String(),
            "session": pl.Date(),
            "open": pl.Float64(),
            "close": pl.Float64(),
        }
    )
    result = build_historical_market_candidates(empty)
    assert result.frame.height == 0
    assert result.frame.columns == ["session", *HISTORICAL_MARKET_CANDIDATE_COLUMNS]
    assert result.audit["candidate_rows"]["es_basis"]["reconciles"] is True  # type: ignore[index]


def test_feature_frame_loader_reads_only_local_store_and_converts_et_sessions(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "store")
    days = (date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7))
    bars = [
        _stored_bar("^GSPC", day, 100.0 + index, 101.0 + index) for index, day in enumerate(days)
    ] + [_stored_bar("ES=F", day, 101.0 + index, 102.0 + index) for index, day in enumerate(days)]
    store.append("raw", "market_daily", bars)

    loaded = feature_frame.load_historical_market_candidates(store)
    assert loaded.frame.get_column("session").to_list() == list(days)
    assert loaded.audit["input_series"]["ES=F"]["sources"] == ["yfinance"]  # type: ignore[index]
    assert "raw.market_daily" in feature_frame.HISTORICAL_MARKET_QUERY
    assert not list((tmp_path / "store").rglob("*.json"))


def test_pre_registered_historical_space_has_control_expansion_and_two_swaps() -> None:
    control = list(BASELINE_FEATURES)
    values = {
        name: [float(index) + position for index in range(40)]
        for position, name in enumerate(control)
    }
    values.update(
        {
            name: [math.sin(index + offset) for index in range(40)]
            for offset, name in enumerate(HISTORICAL_MARKET_CANDIDATE_COLUMNS)
        }
    )
    matrix = pl.DataFrame(values)
    sets = feature_budget.planned_sets(
        matrix,
        candidates=feature_budget.HISTORICAL_MARKET_CANDIDATES,
        candidate_set=HISTORICAL_MARKET_CANDIDATE_SET,
        expanded_name=feature_budget.HISTORICAL_EXPANDED_SET,
        swap_prefix=feature_budget.HISTORICAL_SWAP_PREFIX,
        expanded_source="BASELINE_FEATURES + historical_market_v1",
        expanded_provenance="control + exactamente dos candidatas",
    )
    assert [item.name for item in sets] == [
        feature_budget.CONTROL_SET,
        feature_budget.HISTORICAL_EXPANDED_SET,
        "swap_historical_es_basis",
        "swap_historical_spx_overnight_ret_1",
    ]
    assert [item.n_features for item in sets] == [10, 12, 10, 10]
    assert sets[1].columns == (*BASELINE_FEATURES, *HISTORICAL_MARKET_CANDIDATE_COLUMNS)
    assert len({item.name for item in sets}) == 4


def test_cli_defaults_to_existing_commodity_group_and_separates_historical_report_name() -> None:
    report = feature_budget.BudgetReport(
        as_of=datetime(2026, 10, 11, tzinfo=UTC),
        report_date=date(2026, 10, 11),
        payload={},
        report_sha256="sha256:test",
        sets=(),
        verdict=feature_budget.Verdict("not_evaluable", None, None, 0),
        decision="mantiene_control",
        adopted=feature_budget.SetMeasurement(
            name="control_v1",
            kind="control",
            columns=BASELINE_FEATURES,
            usable=False,
            n_traded=0,
            n_test_sessions=0,
            sharpe=None,
            brier=None,
            log_loss=None,
            matrix_sha256="sha256:test",
            features_version="sha256:test",
            feature_spec_sha256={},
            feature_code_version=2,
        ),
        candidate_group=feature_budget.HISTORICAL_MARKET_CANDIDATE_GROUP,
    )
    assert report.report_stem == "feature_budget_historical_market_2026-10-11"
    assert feature_budget.COMMODITIES_CANDIDATE_GROUP == "commodities"
