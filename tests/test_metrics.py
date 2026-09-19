"""Tests for the net performance and calibration boundary (#15)."""

from __future__ import annotations

import math
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from cfdtrader.backtest.engine import Decision, Direction, SessionOutcome
from cfdtrader.backtest.metrics import (
    MetricsInputError,
    bootstrap_confidence_interval,
    brier_score,
    calculate_metrics,
    calibration_curve,
    drawdown_metrics,
    generate_quantstats_report,
    log_loss,
    sharpe_ratio,
)


def _outcome(
    day: int,
    *,
    status: str = "traded",
    pnl_net_pct: float | None = 1.0,
    probability: float | None = None,
) -> SessionOutcome:
    decision = None
    if status == "traded":
        decision = Decision(
            direction=Direction.LONG,
            reason="test",
            notional_usd=Decimal("10000"),
            probability=probability,
        )
    return SessionOutcome(
        fold_index=0,
        session=date(2026, 1, day),
        session_index=day,
        status=status,
        reason=None,
        skip_reason=None,
        gap_px=None,
        decision=decision,
        entry_session=None,
        exit_session=None,
        entry_px=None,
        exit_px=None,
        exit_reason=None,
        exit_bar_index=None,
        notional_usd=None,
        gross_pct=None,
        pnl_declared_pct=None,
        pnl_net_pct=pnl_net_pct,
        pnl_net_reason=None,
        cost=None,
    )


def test_standalone_metrics_use_decimal_returns_and_known_drawdown() -> None:
    returns = (0.1, -0.1, -0.1, 0.3)

    maximum, duration = drawdown_metrics(returns)

    assert maximum == pytest.approx(0.19)
    assert duration == 2
    assert sharpe_ratio((0.01, 0.02, 0.03)) > 0.0


def test_bootstrap_is_seeded_and_returns_a_known_estimate() -> None:
    values = (0.0, 0.1, 0.2)

    first = bootstrap_confidence_interval(
        values, lambda sample: sum(sample) / len(sample), n_bootstrap=200, seed=17
    )
    second = bootstrap_confidence_interval(
        values, lambda sample: sum(sample) / len(sample), n_bootstrap=200, seed=17
    )

    assert first == second
    assert first.estimate == pytest.approx(0.1)
    assert first.lower <= first.estimate <= first.upper


def test_probability_metrics_and_calibration_bins_are_included() -> None:
    assert brier_score((0.8, 0.2), (True, False)) == pytest.approx(0.04)
    assert log_loss((0.8, 0.2), (True, False)) == pytest.approx(-math.log(0.8))
    curve = calibration_curve((0.1, 0.9), (False, True), n_bins=2)

    assert curve[0].count == 1
    assert curve[0].predicted_probability == pytest.approx(0.1)
    assert curve[1].observed_frequency == pytest.approx(1.0)


def test_calculate_metrics_is_net_and_keeps_no_trade_in_risk_series() -> None:
    outcomes = (
        _outcome(1, pnl_net_pct=2.0, probability=0.6),
        _outcome(2, status="no_trade", pnl_net_pct=None),
        _outcome(3, pnl_net_pct=-1.0, probability=0.4),
        _outcome(4, status="skipped", pnl_net_pct=None),
    )

    metrics = calculate_metrics(outcomes, n_bootstrap=100, seed=4, n_calibration_bins=2)

    assert metrics.n_sessions == 4
    assert metrics.n_trades == 2
    assert metrics.n_skipped == 1
    assert metrics.returns == pytest.approx((0.02, 0.0, -0.01))
    assert metrics.ev_per_trade_pct == pytest.approx(0.5)
    assert metrics.hit_rate == pytest.approx(0.5)
    assert metrics.brier_score == pytest.approx(0.16)
    assert metrics.calibration[0].count == 1
    assert metrics.calibration[1].count == 1


def test_net_metrics_reject_unmeasured_pnl_instead_of_using_declared_pnl() -> None:
    with pytest.raises(MetricsInputError, match="pnl_net_pct es null"):
        calculate_metrics((_outcome(1, pnl_net_pct=None),), n_bootstrap=10)


def test_quantstats_is_only_a_report_adapter(tmp_path: Path) -> None:
    target = tmp_path / "report.html"

    returned = generate_quantstats_report(
        (0.01, -0.005, 0.02),
        target,
        dates=(date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)),
    )

    assert returned == target
    assert target.exists()
    assert target.stat().st_size > 0
