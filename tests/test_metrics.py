"""Tests de las metricas netas y de calibracion (#15).

Un test por criterio de aceptacion (A1-A35), nombrados ``test_a{i}_...``.  Lo importante
que se comprueba aqui:

- ``alpha_pct`` es el alfa de Jensen y la diferencia de medias vive, con nombre honesto, en
  ``mean_excess_return_pct`` (A5-A8);
- la duracion del drawdown se cuenta en sesiones de la serie de riesgo (A2) y los recuentos
  cuadran con los tres estados (A3);
- ningun ``float`` publicado es ``inf`` ni ``nan`` y el payload serializa con JSON estricto
  (A10-A13);
- un ``pnl_net_pct`` nulo es un error tipado por las dos rutas de entrada, nunca un cero ni
  el P&L declarado (A22, A23);
- la ruta ``BacktestRun`` se ejercita con una corrida **real** del motor (A24);
- el *bootstrap* cabe en el presupuesto de tiempo, no materializa la matriz completa y es
  reproducible byte a byte entre procesos (A29, A30, A33).

Los importes declarados se calculan **a mano** (0,24 $ por ida y vuelta corta de una noche
sobre 10.000 $) y el coste se comprueba contra la tabla de #11, no contra el modulo.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import time
import tracemalloc
from collections.abc import Callable, Iterator, Sequence
from datetime import date, timedelta
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import cast

import pytest

from cfdtrader.analysis.cost_audit import Side
from cfdtrader.backtest import metrics as metrics_module
from cfdtrader.backtest.costs import (
    CostBreakdown,
    SlippageParameter,
    cost_breakdown,
    declared_cost_model,
    declared_slippage_assumption,
)
from cfdtrader.backtest.engine import (
    BacktestRun,
    Decision,
    Direction,
    SessionInput,
    SessionOutcome,
    SessionView,
    run_walk_forward,
)
from cfdtrader.backtest.metrics import (
    CalibrationBin,
    ConfidenceInterval,
    MetricsError,
    MetricsInputError,
    PerformanceMetrics,
    bootstrap_ci,
    bootstrap_confidence_interval,
    brier_score,
    calculate_metrics,
    calibration_curve,
    drawdown_metrics,
    equity_curve,
    generate_quantstats_report,
    log_loss,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)
from cfdtrader.backtest.splits import Fold, SplitPlan, walk_forward_splits

NOTIONAL = Decimal("10000")
REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = Path(str(metrics_module.__file__))
SOURCE = MODULE_PATH.read_text(encoding="utf-8")
TEST_SOURCE = Path(__file__).read_text(encoding="utf-8")

#: Un test por criterio ``Ai``: el rango completo que A28 comprueba por ``grep``.
CRITERIA = range(1, 36)

#: Funciones publicas del modulo: ninguna puede quedarse sin al menos un test (A27).
PUBLIC_FUNCTIONS = (
    "bootstrap_ci",
    "bootstrap_confidence_interval",
    "brier_score",
    "calculate_metrics",
    "calibration_curve",
    "drawdown_metrics",
    "equity_curve",
    "generate_quantstats_report",
    "log_loss",
    "max_drawdown",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
)

GOLDEN_PNLS = (1.0, -0.5, 0.8, -0.3, 0.4, -0.2)
GOLDEN_DAYS = (5, 6, 7, 8, 9, 12)
GOLDEN_NO_TRADE_DAY = 13
GOLDEN_SKIPPED_DAY = 14

#: Congelado con ``numpy`` 2.5.3, la semilla 42 y 1.000 remuestreos: el golden de A30.
GOLDEN_SHARPE_INTERVAL = (4.7441028973888235, -12.651542928099262, 20.167402484031086)
GOLDEN_SORTINO_INTERVAL = (11.679942321414904, -9.900562710028924, 78.00000000000001)

#: La serie de riesgo de A29 y A33: 3.192 sesiones, como la muestra limpia de #52.
BIG_SESSIONS = 3192

#: Presupuesto de A33 para el pico de memoria adicional del *bootstrap*.
MEMORY_BUDGET_BYTES = 512 * 1024 * 1024


def _outcome(
    session: date,
    *,
    status: str = "traded",
    pnl_net_pct: float | None = 1.0,
    pnl_declared_pct: float | None = 5.0,
    probability: float | None = None,
    cost: CostBreakdown | None = None,
) -> SessionOutcome:
    """Una ``SessionOutcome`` sintetica con el estado, el P&L y el coste que pida el caso."""
    decision = None
    if status == "traded":
        decision = Decision(
            direction=Direction.LONG,
            reason="prueba",
            notional_usd=NOTIONAL,
            probability=probability,
        )
    return SessionOutcome(
        fold_index=0,
        session=session,
        session_index=session.toordinal(),
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
        pnl_declared_pct=pnl_declared_pct,
        pnl_net_pct=pnl_net_pct,
        pnl_net_reason=None,
        cost=cost,
    )


def _sessions(count: int, *, start: date = date(2026, 1, 5)) -> list[date]:
    """``count`` sesiones consecutivas de lunes a viernes (posiciones, no calendario)."""
    days: list[date] = []
    session = start
    while len(days) < count:
        if session.weekday() < 5:
            days.append(session)
        session += timedelta(days=1)
    return days


def _series(*pnls: float, start: date = date(2026, 1, 5)) -> tuple[SessionOutcome, ...]:
    """Una serie de sesiones operadas con esos P&L netos, en sesiones consecutivas."""
    days = _sessions(len(pnls), start=start)
    return tuple(_outcome(day, pnl_net_pct=pnl) for day, pnl in zip(days, pnls, strict=True))


def _golden_outcomes() -> tuple[SessionOutcome, ...]:
    """La serie congelada de A30: seis operadas, una ``no_trade`` y una ``skipped``."""
    golden = tuple(
        _outcome(date(2026, 1, day), pnl_net_pct=pnl)
        for day, pnl in zip(GOLDEN_DAYS, GOLDEN_PNLS, strict=True)
    )
    return (
        *golden,
        _outcome(date(2026, 1, GOLDEN_NO_TRADE_DAY), status="no_trade", pnl_net_pct=None),
        _outcome(date(2026, 1, GOLDEN_SKIPPED_DAY), status="skipped", pnl_net_pct=None),
    )


def _flatten(text: str) -> str:
    """El texto con los espacios normalizados: un docstring se parte en lineas."""
    return " ".join(text.split())


def _code_only() -> str:
    """El codigo del modulo sin **ningun** literal de cadena (ni docstrings)."""
    masked: set[int] = set()
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            masked.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return "\n".join(
        "" if number in masked else row for number, row in enumerate(SOURCE.splitlines(), start=1)
    )


def _walk_floats(node: object) -> Iterator[float]:
    """Todos los ``float`` de un payload serializado, a cualquier profundidad (A12)."""
    if isinstance(node, float):
        yield node
        return
    if isinstance(node, dict):
        for value in cast("dict[str, object]", node).values():
            yield from _walk_floats(value)
        return
    if isinstance(node, (list, tuple)):
        for value in cast("list[object] | tuple[object, ...]", node):
            yield from _walk_floats(value)


def _measured_zero_slippage() -> SlippageParameter:
    """Un *slippage* **medido y nulo**: la tabla declarada de #8 queda intacta (A25)."""
    return SlippageParameter.measured(
        pct_of_notional=Decimal("0"),
        source="prueba",
        reason="medicion de prueba con termino nulo",
    )


def _short_round_trip_cost() -> CostBreakdown:
    """Ida y vuelta corta de **una noche** sobre 10.000 $: 0,24 $ / 0,0024 % (plan.md 3.3)."""
    return cost_breakdown(
        model=declared_cost_model(),
        slippage=_measured_zero_slippage(),
        notional_usd=NOTIONAL,
        side=Side.SHORT,
        nights=1,
        overnight_reason="prueba de coste: una noche declarada",
    )


def _long_decider() -> Callable[[SessionView], Decision]:
    """Un decididor largo con barreras simetricas alrededor del ``open``."""

    def decide(view: SessionView) -> Decision:
        if view.open_px is None:
            return Decision(direction=Direction.LONG, reason="prueba", notional_usd=NOTIONAL)
        return Decision(
            direction=Direction.LONG,
            reason="prueba",
            stop_px=view.open_px - 1.0,
            target_px=view.open_px + 1.0,
            notional_usd=NOTIONAL,
        )

    return decide


def _nothing_decider() -> Callable[[SessionView], Decision]:
    """Un decididor que no opera: ``NOTHING`` con su motivo."""

    def decide(_view: SessionView) -> Decision:
        return Decision(direction=Direction.NOTHING, reason="prueba: sin senal")

    return decide


def _flat_inputs(days: Sequence[date], *, base: float = 100.0) -> list[SessionInput]:
    """Sesiones planas: ``open`` = ``base + i``, rango de un punto y cierre por encima."""
    return [
        SessionInput(
            session=day,
            open_px=base + index,
            high_px=base + index + 1.0,
            low_px=base + index - 1.0,
            close_px=base + index + 0.5,
        )
        for index, day in enumerate(days)
    ]


def _full_coverage_plan(days: Sequence[date], *, block: int) -> SplitPlan:
    """Un ``SplitPlan`` de #12 **construido a mano** cuyos folds cubren toda la muestra.

    ``walk_forward_splits`` exige ``n_splits * test_size < n`` (A25 de #12), asi que nunca
    deja ``uncovered`` vacio.  Con un plan de #12, ``metrics.n_sessions == run.n_sessions``
    es imposible por construccion, y A24 lo pide: el plan se construye con los campos
    publicos de #12 (el motor los consume tal cual) y la tension se declara en A24.
    """
    folds: list[Fold] = []
    for index, start in enumerate(range(0, len(days), block)):
        folds.append(
            Fold(
                index=index,
                test_start=start,
                test_stop=min(start + block, len(days)),
                train=tuple(range(start)),
                purged=(),
                embargoed=(),
            )
        )
    return SplitPlan(
        n_sessions=len(days),
        inputs={"n_splits": len(folds), "test_size": block, "hand_built": True},
        folds=tuple(folds),
        uncovered=(),
        purge_total=0,
        embargo_total=0,
        embargo_in_train_total=0,
        exclusions_are_no_op=True,
        plan_sha256="plan-construido-a-mano-para-#15",
    )


def _run(
    days: Sequence[date],
    *,
    slippage: SlippageParameter,
    decider: Callable[[SessionView], Decision] | None = None,
    block: int = 6,
) -> BacktestRun:
    """Ejecuta el motor de #13 sobre la muestra completa con el *slippage* declarado."""
    plan = _full_coverage_plan(days, block=block)
    decide = _long_decider() if decider is None else decider
    return run_walk_forward(
        _flat_inputs(days),
        split_plan=plan,
        cost_model=declared_cost_model(),
        slippage=slippage,
        decide_by_fold=[decide] * len(plan.folds),
    )


@lru_cache(maxsize=1)
def _big_outcomes() -> tuple[SessionOutcome, ...]:
    """La serie de riesgo de 3.192 sesiones que piden A29 y A33 (se construye una vez)."""
    days = _sessions(BIG_SESSIONS)
    return tuple(
        _outcome(day, pnl_net_pct=((index % 7) - 3) * 0.1) for index, day in enumerate(days)
    )


def test_a1_catalog_is_published_without_misleading_names() -> None:
    names = [field.name for field in dataclasses.fields(PerformanceMetrics)]

    assert "n_no_trade" in names
    assert "max_drawdown_duration_sessions" in names
    assert "max_drawdown_duration" not in names
    assert "drawdown_duration" not in names
    assert not hasattr(PerformanceMetrics, "max_drawdown_duration")
    assert not hasattr(PerformanceMetrics, "drawdown_duration")
    for required in (
        "n_sessions",
        "n_trades",
        "n_no_trade",
        "n_skipped",
        "returns",
        "equity",
        "total_return_pct",
        "sharpe",
        "sharpe_ci",
        "sortino",
        "sortino_ci",
        "ev_per_trade_pct",
        "hit_rate",
        "payoff_ratio",
        "profit_factor",
        "average_win_pct",
        "average_loss_pct",
        "max_drawdown_pct",
        "max_drawdown_duration_sessions",
        "session_span_days",
        "trades_per_year",
        "total_cost_pct",
        "total_cost_usd",
        "brier_score",
        "log_loss",
        "calibration",
        "benchmark_return_pct",
        "mean_excess_return_pct",
        "alpha_pct",
        "beta",
        "benchmark_sharpe",
    ):
        assert required in names, f"falta {required} en el catalogo publicado (A1)"


def test_a2_drawdown_duration_counts_risk_sessions() -> None:
    outcomes = (
        _outcome(date(2026, 1, 5), pnl_net_pct=-10.0),
        _outcome(date(2026, 1, 6), status="skipped", pnl_net_pct=None),
        _outcome(date(2026, 1, 7), status="skipped", pnl_net_pct=None),
        _outcome(date(2026, 1, 8), pnl_net_pct=-10.0),
        _outcome(date(2026, 1, 9), pnl_net_pct=30.0),
    )
    metrics = calculate_metrics(outcomes, n_bootstrap=50)

    assert metrics.returns == (-0.1, -0.1, 0.3)
    assert metrics.max_drawdown_duration_sessions == 2
    assert metrics.max_drawdown_pct == pytest.approx(19.0)
    assert drawdown_metrics(metrics.returns) == pytest.approx((0.19, 2))
    assert "counts observations of the **risk series**" in _flatten(metrics_module.__doc__ or "")


def test_a3_count_and_length_identities() -> None:
    outcomes = (
        _outcome(date(2026, 1, 5), pnl_net_pct=2.0),
        _outcome(date(2026, 1, 6), status="no_trade", pnl_net_pct=None),
        _outcome(date(2026, 1, 7), status="skipped", pnl_net_pct=None),
        _outcome(date(2026, 1, 8), pnl_net_pct=-1.0),
    )
    metrics = calculate_metrics(outcomes, n_bootstrap=50)

    assert metrics.n_sessions == metrics.n_trades + metrics.n_no_trade + metrics.n_skipped
    assert metrics.n_sessions == len(outcomes) == 4
    assert len(metrics.returns) == metrics.n_sessions - metrics.n_skipped
    assert len(metrics.returns) == len(metrics.equity)
    assert metrics.equity[0] == pytest.approx(1.0 + metrics.returns[0])
    assert metrics.total_return_pct == pytest.approx((metrics.equity[-1] - 1.0) * 100.0)
    assert equity_curve(metrics.returns) == metrics.equity


def test_a4_units_convention_is_declared_and_checked() -> None:
    docstring = _flatten(metrics_module.__doc__ or "")
    assert "decimal returns (``0.01`` means one percent)" in docstring
    assert "percentage points" in docstring
    assert "positive loss magnitude" in docstring
    assert "dimensionless" in docstring

    metrics = calculate_metrics(_series(1.0, -2.0), n_bootstrap=50)

    assert metrics.returns == (0.01, -0.02)
    assert metrics.ev_per_trade_pct == pytest.approx(-0.5)
    assert metrics.average_win_pct == pytest.approx(1.0)
    assert metrics.average_loss_pct == pytest.approx(-2.0)
    assert max_drawdown(metrics.returns) >= 0.0
    assert metrics.max_drawdown == pytest.approx(metrics.max_drawdown_pct / 100.0)


def test_a5_alpha_is_jensens_alpha() -> None:
    benchmark = (0.01, -0.005, 0.02)
    metrics = calculate_metrics(
        _series(2.0, -1.0, 4.0), n_bootstrap=100, benchmark_returns=benchmark
    )

    assert metrics.beta == 2.0
    assert metrics.alpha_pct == pytest.approx(0.0, abs=1e-12)
    assert metrics.mean_excess_return_pct == pytest.approx(0.8333333333333334)


def test_a6_zero_variance_benchmark_leaves_beta_and_alpha_null() -> None:
    metrics = calculate_metrics(
        _series(1.0, 1.0, 1.0), n_bootstrap=100, benchmark_returns=(0.005, 0.005, 0.005)
    )

    assert metrics.beta is None
    assert metrics.alpha_pct is None
    assert metrics.benchmark_return_pct == pytest.approx(1.5075, abs=1e-4)
    assert metrics.benchmark_sharpe == 0.0


def test_a7_mean_excess_return_keeps_the_mean_difference() -> None:
    benchmark = (0.01, -0.005, 0.02)
    with_benchmark = calculate_metrics(
        _series(2.0, -1.0, 4.0), n_bootstrap=100, benchmark_returns=benchmark
    )
    without_benchmark = calculate_metrics(_series(2.0, -1.0, 4.0), n_bootstrap=100)

    assert with_benchmark.mean_excess_return_pct == pytest.approx(0.8333333333333334)
    assert without_benchmark.mean_excess_return_pct is None
    assert with_benchmark.alpha_pct != with_benchmark.mean_excess_return_pct


def test_a8_no_benchmark_means_no_zeros() -> None:
    metrics = calculate_metrics(_series(1.0, -1.0), n_bootstrap=50)

    assert metrics.benchmark_return_pct is None
    assert metrics.mean_excess_return_pct is None
    assert metrics.alpha_pct is None
    assert metrics.beta is None
    assert metrics.benchmark_sharpe is None


def test_a9_malformed_benchmark_is_a_typed_error() -> None:
    with pytest.raises(MetricsInputError, match="debe tener la misma longitud"):
        calculate_metrics(_series(1.0, -1.0), n_bootstrap=50, benchmark_returns=(0.01,))
    with pytest.raises(MetricsInputError, match="finitos"):
        calculate_metrics(_series(1.0, -1.0), n_bootstrap=50, benchmark_returns=(0.01, math.nan))

    docstring = _flatten(metrics_module.__doc__ or "")
    assert "already aligned session by session" in docstring
    assert "does not align series by date" in docstring
    code = _code_only()
    assert "read_pit" not in code
    assert "Store(" not in code


def test_a10_profit_factor_is_never_infinite() -> None:
    assert profit_factor((0.01, 0.02)) is None
    assert profit_factor((0.01, -0.02)) == pytest.approx(0.5)

    gains_only = calculate_metrics(_series(1.0, 2.0), n_bootstrap=50)
    balanced = calculate_metrics(_series(1.0, -2.0), n_bootstrap=50)
    no_trades = calculate_metrics(
        (_outcome(date(2026, 1, 5), status="no_trade", pnl_net_pct=None),), n_bootstrap=50
    )

    assert gains_only.profit_factor is None
    assert gains_only.n_trades == 2
    assert gains_only.average_loss_pct is None
    assert balanced.profit_factor == pytest.approx(0.5)
    assert balanced.gain_loss_ratio == balanced.payoff_ratio == pytest.approx(0.5)
    assert no_trades.profit_factor is None
    assert no_trades.n_trades == 0
    assert no_trades.hit_rate is None
    assert "``profit_factor`` is ``None`` when there are no losing trades" in _flatten(
        metrics_module.__doc__ or ""
    )


def test_a11_empty_calibration_bins_publish_none() -> None:
    curve = calibration_curve((0.1,), (True,), n_bins=5)

    assert [item.count for item in curve] == [1, 0, 0, 0, 0]
    assert curve[0].predicted_probability == pytest.approx(0.1)
    assert curve[0].observed_frequency == pytest.approx(1.0)
    for empty in curve[1:]:
        assert empty.predicted_probability is None
        assert empty.observed_frequency is None
    assert {field.name for field in dataclasses.fields(CalibrationBin)} == {
        "lower",
        "upper",
        "predicted_probability",
        "observed_frequency",
        "count",
    }

    declared = calculate_metrics(
        (
            _outcome(date(2026, 1, 5), pnl_net_pct=2.0, probability=0.6),
            _outcome(date(2026, 1, 6), pnl_net_pct=-1.0, probability=0.4),
        ),
        n_bootstrap=50,
        n_calibration_bins=2,
    )
    assert declared.brier_score == pytest.approx(0.16)
    assert declared.log_loss == pytest.approx(-math.log(0.6))
    assert [item.count for item in declared.calibration] == [1, 1]
    assert "an empty calibration bin carries ``None`` instead of ``nan``" in _flatten(
        metrics_module.__doc__ or ""
    )


def test_a12_no_published_float_is_inf_or_nan() -> None:
    scenarios = (
        calculate_metrics(_series(1.0, 2.0), n_bootstrap=50),
        calculate_metrics((_outcome(date(2026, 1, 5), pnl_net_pct=2.0),), n_bootstrap=50),
        calculate_metrics(
            (_outcome(date(2026, 1, 5), pnl_net_pct=2.0, probability=0.1),), n_bootstrap=50
        ),
        calculate_metrics(
            (
                _outcome(date(2026, 1, 5), status="no_trade", pnl_net_pct=None),
                _outcome(date(2026, 1, 6), status="no_trade", pnl_net_pct=None),
            ),
            n_bootstrap=50,
        ),
        calculate_metrics(_series(1.0, 1.0, 1.0), n_bootstrap=50),
        calculate_metrics(
            _series(2.0, -1.0, 4.0), n_bootstrap=50, benchmark_returns=(0.01, -0.005, 0.02)
        ),
    )

    for metrics in scenarios:
        payload = metrics.to_payload()
        for value in _walk_floats(payload):
            assert math.isfinite(value), f"valor no finito publicado: {value!r}"
        json.dumps(payload, allow_nan=False)

    with pytest.raises(ValueError):
        json.dumps({"profit_factor": math.inf}, allow_nan=False)


def test_a13_log_loss_extremes_are_finite() -> None:
    assert log_loss((1.0,), (False,)) == 34.538776394910684
    assert log_loss((1.0,), (True,)) < 1e-14
    assert inspect.signature(log_loss).parameters["epsilon"].default == 1e-15
    for bad in (0.0, -0.1, 0.5, 0.9):
        with pytest.raises(MetricsInputError):
            log_loss((0.5,), (True,), epsilon=bad)


def test_a14_inadmissible_inputs_raise_metrics_input_error() -> None:
    assert issubclass(MetricsInputError, MetricsError)
    with pytest.raises(MetricsInputError):
        brier_score((1.5,), (True,))
    with pytest.raises(MetricsInputError):
        brier_score((0.5, 0.5), (True,))
    with pytest.raises(MetricsInputError):
        brier_score((), ())
    with pytest.raises(MetricsInputError):
        calibration_curve((), ())
    with pytest.raises(MetricsInputError):
        calibration_curve((0.5,), (True,), n_bins=0)
    with pytest.raises(MetricsInputError):
        sharpe_ratio(())
    with pytest.raises(MetricsInputError):
        sortino_ratio(())
    with pytest.raises(MetricsInputError):
        equity_curve(())
    with pytest.raises(MetricsInputError):
        sharpe_ratio((math.nan, 0.01))
    with pytest.raises(MetricsInputError):
        sharpe_ratio((0.01,), annualization=0)
    with pytest.raises(MetricsInputError):
        sortino_ratio((0.01,), annualization=0)

    average = bootstrap_confidence_interval(
        (0.01, -0.02, 0.03),
        lambda sample: sum(sample) / len(sample),
        n_bootstrap=8,
        seed=3,
    )
    assert average.n_bootstrap == 8
    assert average.seed == 3
    with pytest.raises(MetricsInputError):
        bootstrap_confidence_interval((), lambda sample: sum(sample), n_bootstrap=8)
    with pytest.raises(MetricsInputError):
        bootstrap_confidence_interval((0.01,), lambda sample: math.inf, n_bootstrap=8)
    with pytest.raises(MetricsInputError):
        bootstrap_confidence_interval(
            (0.01,), lambda sample: sum(sample), n_bootstrap=8, confidence_level=1.5
        )
    with pytest.raises(MetricsInputError):
        bootstrap_confidence_interval((0.01,), lambda sample: sum(sample), n_bootstrap=8, seed=-1)

    with pytest.raises(MetricsInputError, match="n_bootstrap"):
        bootstrap_confidence_interval((0.01,), lambda sample: sum(sample), n_bootstrap=0)
    with pytest.raises(MetricsInputError, match="n_bootstrap"):
        calculate_metrics(_series(1.0), n_bootstrap=0)
    with pytest.raises(MetricsInputError, match="n_calibration_bins"):
        calculate_metrics(_series(1.0), n_bootstrap=50, n_calibration_bins=0)
    with pytest.raises(MetricsInputError, match="perder"):
        sharpe_ratio((-1.0, 0.01))

    seen: list[int] = []

    def occasionally_infinite(sample: Sequence[float]) -> float:
        seen.append(len(sample))
        return 0.0 if len(seen) == 1 else math.inf

    with pytest.raises(MetricsInputError, match="finitos"):
        bootstrap_confidence_interval((0.01, -0.02), occasionally_infinite, n_bootstrap=8, seed=3)

    degenerate = bootstrap_confidence_interval(
        (0.01, -0.02, 0.03),
        lambda sample: sum(sample) / len(sample),
        confidence_level=0.5,
        n_bootstrap=5,
        seed=3,
    )
    assert degenerate.confidence_level == 0.5
    assert degenerate.lower <= degenerate.upper

    mixed = (
        _outcome(date(2026, 1, 5), pnl_net_pct=1.0, probability=0.6),
        _outcome(date(2026, 1, 6), pnl_net_pct=-1.0),
    )
    with pytest.raises(MetricsInputError, match="declarar probability"):
        calculate_metrics(mixed, n_bootstrap=50)


def test_a15_empty_and_all_skipped_series() -> None:
    with pytest.raises(MetricsInputError, match="el resultado no contiene sesiones"):
        calculate_metrics(())

    all_skipped = (
        _outcome(date(2026, 1, 5), status="skipped", pnl_net_pct=None),
        _outcome(date(2026, 1, 6), status="skipped", pnl_net_pct=None),
    )
    with pytest.raises(MetricsInputError, match="se necesita al menos un retorno"):
        calculate_metrics(all_skipped, n_bootstrap=50)


def test_a16_all_no_trade() -> None:
    outcomes = tuple(
        _outcome(date(2026, 1, day), status="no_trade", pnl_net_pct=None) for day in (5, 6, 7)
    )
    metrics = calculate_metrics(outcomes, n_bootstrap=100)

    assert metrics.returns == (0.0, 0.0, 0.0)
    assert metrics.n_trades == 0
    assert metrics.sharpe == 0.0
    assert metrics.sortino == 0.0
    assert metrics.sharpe_ci.lower == 0.0
    assert metrics.sharpe_ci.upper == 0.0
    assert metrics.sortino_ci.lower == 0.0
    assert metrics.sortino_ci.upper == 0.0
    assert metrics.max_drawdown_pct == 0.0
    assert metrics.max_drawdown_duration_sessions == 0
    assert metrics.calibration == ()
    for value in (
        metrics.ev_per_trade_pct,
        metrics.hit_rate,
        metrics.payoff_ratio,
        metrics.profit_factor,
        metrics.average_win_pct,
        metrics.average_loss_pct,
        metrics.brier_score,
        metrics.log_loss,
        metrics.total_cost_pct,
        metrics.total_cost_usd,
    ):
        assert value is None, f"A16 exige null, no {value!r}"


def test_a17_no_trade_is_not_a_trade() -> None:
    outcomes = (
        _outcome(date(2026, 1, 5), pnl_net_pct=2.0),
        _outcome(date(2026, 1, 6), status="no_trade", pnl_net_pct=None),
    )
    metrics = calculate_metrics(outcomes, n_bootstrap=50)

    assert metrics.returns == (0.02, 0.0)
    assert metrics.n_trades == 1
    assert metrics.n_no_trade == 1
    assert metrics.ev_per_trade_pct == pytest.approx(2.0)
    assert metrics.hit_rate == 1.0


def test_a18_constant_returns() -> None:
    up = calculate_metrics(_series(1.0, 1.0, 1.0), n_bootstrap=100)
    down = calculate_metrics(_series(-1.0, -1.0, -1.0), n_bootstrap=100)

    assert up.sharpe == 0.0
    assert up.sortino == 0.0
    assert up.sharpe_ci.lower == up.sharpe_ci.upper == 0.0
    assert up.total_return_pct == pytest.approx(3.0301, abs=1e-4)
    assert up.max_drawdown_pct == 0.0
    assert up.max_drawdown_duration_sessions == 0
    assert up.profit_factor is None

    assert down.sharpe == 0.0
    assert down.max_drawdown_pct == pytest.approx(2.9701, abs=1e-4)
    assert down.max_drawdown_duration_sessions == 3


def test_a19_single_trade() -> None:
    metrics = calculate_metrics((_outcome(date(2026, 1, 5), pnl_net_pct=2.0),), n_bootstrap=100)

    assert metrics.n_trades == 1
    assert metrics.hit_rate == 1.0
    assert metrics.average_win_pct == pytest.approx(2.0)
    assert metrics.payoff_ratio is None
    assert metrics.average_loss_pct is None
    assert metrics.ev_per_trade_pct == pytest.approx(2.0)
    assert metrics.profit_factor is None


def test_a20_turnover_needs_a_span() -> None:
    one_session = calculate_metrics((_outcome(date(2026, 1, 5), pnl_net_pct=1.0),), n_bootstrap=50)
    short_span = calculate_metrics(_series(1.0, 1.0), n_bootstrap=50)
    wide_days = (date(2026, 1, 5), date(2026, 1, 20), date(2026, 2, 4))
    wide = calculate_metrics(
        tuple(_outcome(day, pnl_net_pct=1.0) for day in wide_days), n_bootstrap=50
    )

    assert one_session.trades_per_year is None
    assert one_session.session_span_days == 0
    assert short_span.trades_per_year is None
    assert short_span.session_span_days == 1
    assert wide.session_span_days == 30
    assert wide.trades_per_year is not None
    assert wide.trades_per_year == pytest.approx(wide.n_trades * 365.25 / wide.session_span_days)
    assert "shorter than 30 natural days" in _flatten(metrics_module.__doc__ or "")


def test_a21_unknown_status_is_an_error() -> None:
    weird = _outcome(date(2026, 1, 5), status="weird", pnl_net_pct=1.0)
    with pytest.raises(MetricsInputError) as error:
        calculate_metrics((weird,), n_bootstrap=50)

    assert "2026-01-05" in str(error.value)
    assert "weird" in str(error.value)


def test_a22_null_net_pnl_is_an_error_by_both_routes() -> None:
    declared_only = (_outcome(date(2026, 1, 5), pnl_net_pct=None, pnl_declared_pct=5.0),)
    with pytest.raises(MetricsInputError, match="pnl_net_pct es null") as error:
        calculate_metrics(declared_only, n_bootstrap=50)
    assert "2026-01-05" in str(error.value)

    days = _sessions(12)
    assumed_run = _run(days, slippage=declared_slippage_assumption())
    with pytest.raises(MetricsInputError, match="pnl_net_pct es null"):
        calculate_metrics(assumed_run, n_bootstrap=50)

    assert "pnl_declared_pct" not in _code_only()


def test_a23_slippage_assumption_leaves_net_metrics_null() -> None:
    assumption = declared_slippage_assumption()
    assert assumption.state.value == "assumed"
    assert assumption.is_measurement is False

    days = _sessions(12)
    traded_run = _run(days, slippage=assumption)
    assert traded_run.traded == 12
    with pytest.raises(MetricsInputError, match="pnl_net_pct es null"):
        calculate_metrics(traded_run, n_bootstrap=50)

    flat_run = _run(days, slippage=assumption, decider=_nothing_decider())
    metrics = calculate_metrics(flat_run, n_bootstrap=50)
    assert metrics.n_trades == 0
    assert metrics.total_cost_pct is None
    assert metrics.total_cost_usd is None

    docstring = _flatten(metrics_module.__doc__ or "")
    assert "#62" in docstring
    assert "#60" in docstring
    assert "assumed" in docstring


def test_a24_backtest_run_route_matches_the_engine() -> None:
    days = _sessions(12)
    run = _run(days, slippage=_measured_zero_slippage())
    metrics = calculate_metrics(run, n_bootstrap=100)

    assert run.not_in_any_test == 0
    assert metrics.n_sessions == run.n_sessions
    assert metrics.n_trades == run.traded
    assert metrics.n_no_trade == run.no_trade
    assert metrics.n_skipped == run.skipped
    assert len(metrics.returns) == run.n_sessions - run.skipped

    generated = walk_forward_splits(
        days, label_horizon=[0] * len(days), n_splits=2, test_size=3, embargo_sessions=1
    )
    generated_run = run_walk_forward(
        _flat_inputs(days),
        split_plan=generated,
        cost_model=declared_cost_model(),
        slippage=_measured_zero_slippage(),
        decide_by_fold=[_long_decider()] * len(generated.folds),
    )
    generated_metrics = calculate_metrics(generated_run, n_bootstrap=50)

    assert generated_run.not_in_any_test > 0
    assert generated_metrics.n_sessions == (
        generated_run.n_sessions - generated_run.not_in_any_test
    )
    assert generated_metrics.n_trades == generated_run.traded
    assert generated_metrics.n_no_trade == generated_run.no_trade


def test_a25_total_cost_is_exact_decimal() -> None:
    cost = _short_round_trip_cost()
    assert cost.c_total_usd == Decimal("0.24")
    assert cost.c_total_pct == Decimal("0.0024")

    outcomes = tuple(
        _outcome(date(2026, 1, day), pnl_net_pct=1.0, cost=_short_round_trip_cost())
        for day in (5, 6, 7)
    )
    metrics = calculate_metrics(outcomes, n_bootstrap=50)

    assert metrics.total_cost_usd == Decimal("0.72")
    assert metrics.total_cost_pct == pytest.approx(0.0072)

    partial = (
        _outcome(date(2026, 1, 5), pnl_net_pct=1.0, cost=_short_round_trip_cost()),
        _outcome(date(2026, 1, 6), pnl_net_pct=1.0, cost=None),
        _outcome(date(2026, 1, 7), pnl_net_pct=1.0, cost=_short_round_trip_cost()),
    )
    metrics_partial = calculate_metrics(partial, n_bootstrap=50)
    assert metrics_partial.total_cost_usd is None
    assert metrics_partial.total_cost_pct is None

    unmeasured = cost_breakdown(
        model=declared_cost_model(),
        slippage=declared_slippage_assumption(),
        notional_usd=NOTIONAL,
        side=Side.SHORT,
    )
    assert unmeasured.c_total_pct is None
    metrics_unmeasured = calculate_metrics(
        (_outcome(date(2026, 1, 5), pnl_net_pct=1.0, cost=unmeasured),), n_bootstrap=50
    )
    assert metrics_unmeasured.total_cost_usd is None
    assert metrics_unmeasured.total_cost_pct is None


def test_a26_benchmark_metrics_is_covered() -> None:
    benchmark_metrics = metrics_module._benchmark_metrics  # pyright: ignore[reportPrivateUsage]

    strategy = (0.02, -0.01, 0.04)
    benchmark = (0.01, -0.005, 0.02)
    result = benchmark_metrics(strategy, benchmark)

    assert result[3] == 2.0
    assert result[2] == pytest.approx(0.0, abs=1e-12)
    assert result[1] == pytest.approx(0.8333333333333334)
    assert result[4] == pytest.approx(sharpe_ratio(benchmark))
    assert result[0] == pytest.approx(2.5049, abs=1e-4)

    flat = benchmark_metrics(strategy, (0.005, 0.005, 0.005))
    assert flat[3] is None
    assert flat[2] is None
    assert flat[0] == pytest.approx(1.5075, abs=1e-4)

    assert benchmark_metrics(strategy, None) == (None, None, None, None, None)
    with pytest.raises(MetricsInputError, match="debe tener la misma longitud"):
        benchmark_metrics(strategy, (0.01,))
    with pytest.raises(MetricsInputError, match="finitos"):
        benchmark_metrics(strategy, (0.01, 0.02, math.inf))


def test_a27_public_functions_have_at_least_one_test() -> None:
    assert bootstrap_ci is bootstrap_confidence_interval
    assert set(PUBLIC_FUNCTIONS) <= set(metrics_module.__all__)
    without_test = [name for name in PUBLIC_FUNCTIONS if not re.search(rf"\b{name}\b", TEST_SOURCE)]
    assert not without_test, f"funciones publicas sin test: {without_test}"
    assert len(re.findall(r"^def test_", TEST_SOURCE, re.M)) >= 30


def test_a28_every_criterion_has_a_named_test() -> None:
    missing = [
        number
        for number in CRITERIA
        if re.search(rf"^def test_a{number}_", TEST_SOURCE, re.M) is None
    ]
    assert not missing, f"criterios sin test propio: {missing}"
    assert len(re.findall(r"^def test_", TEST_SOURCE, re.M)) >= 30


def test_a29_bootstrap_budget_is_respected() -> None:
    outcomes = _big_outcomes()

    started = time.perf_counter()
    metrics = calculate_metrics(outcomes, n_bootstrap=10_000)
    total_seconds = time.perf_counter() - started

    assert metrics.n_sessions == BIG_SESSIONS
    assert metrics.sharpe_ci.n_bootstrap == 10_000
    assert metrics.sortino_ci.n_bootstrap == 10_000
    assert total_seconds <= 10.0, f"calculate_metrics tardo {total_seconds:.2f} s (limite 10 s)"

    returns = metrics.returns
    started = time.perf_counter()
    metrics_module._vectorized_confidence_interval(  # pyright: ignore[reportPrivateUsage]
        returns,
        kind="sharpe",
        estimate=metrics.sharpe,
        confidence_level=0.95,
        n_bootstrap=10_000,
        seed=42,
    )
    sharpe_seconds = time.perf_counter() - started

    started = time.perf_counter()
    metrics_module._vectorized_confidence_interval(  # pyright: ignore[reportPrivateUsage]
        returns,
        kind="sortino",
        estimate=metrics.sortino,
        confidence_level=0.95,
        n_bootstrap=10_000,
        seed=43,
    )
    sortino_seconds = time.perf_counter() - started

    assert sharpe_seconds <= 5.0, f"intervalo de Sharpe: {sharpe_seconds:.2f} s"
    assert sortino_seconds <= 5.0, f"intervalo de Sortino: {sortino_seconds:.2f} s"


_GOLDEN_PROBE = textwrap.dedent(
    """
    from datetime import date
    from decimal import Decimal

    from cfdtrader.backtest.engine import Decision, Direction, SessionOutcome
    from cfdtrader.backtest.metrics import calculate_metrics

    GOLDEN_PNLS = (1.0, -0.5, 0.8, -0.3, 0.4, -0.2)
    GOLDEN_DAYS = (5, 6, 7, 8, 9, 12)


    def outcome(day, status, pnl_net_pct):
        decision = None
        if status == "traded":
            decision = Decision(
                direction=Direction.LONG,
                reason="golden",
                notional_usd=Decimal("10000"),
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
            pnl_declared_pct=5.0,
            pnl_net_pct=pnl_net_pct,
            pnl_net_reason=None,
            cost=None,
        )


    outcomes = tuple(
        outcome(day, "traded", pnl)
        for day, pnl in zip(GOLDEN_DAYS, GOLDEN_PNLS, strict=True)
    )
    outcomes += (outcome(13, "no_trade", None), outcome(14, "skipped", None))
    metrics = calculate_metrics(outcomes, n_bootstrap=1000, seed=42)
    for interval in (metrics.sharpe_ci, metrics.sortino_ci):
        print(
            " ".join(
                float.hex(value)
                for value in (interval.estimate, interval.lower, interval.upper)
            )
        )
    """
)


def test_a30_bootstrap_is_reproducible_across_processes() -> None:
    metrics = calculate_metrics(_golden_outcomes(), n_bootstrap=1000, seed=42)

    assert (
        metrics.sharpe_ci.estimate,
        metrics.sharpe_ci.lower,
        metrics.sharpe_ci.upper,
    ) == GOLDEN_SHARPE_INTERVAL
    assert (
        metrics.sortino_ci.estimate,
        metrics.sortino_ci.lower,
        metrics.sortino_ci.upper,
    ) == GOLDEN_SORTINO_INTERVAL

    outputs: list[str] = []
    for hash_seed in ("0", "1"):
        environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
        completed = subprocess.run(  # noqa: S603 - el interprete de la sesion
            [sys.executable, "-c", _GOLDEN_PROBE],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout)

    assert outputs[0] == outputs[1]
    assert outputs[0].splitlines()[0] == " ".join(
        float.hex(value) for value in GOLDEN_SHARPE_INTERVAL
    )

    docstring = _flatten(metrics_module.__doc__ or "")
    assert "numpy.random.RandomState" in docstring
    assert "Mersenne Twister" in docstring
    assert "frozen" in docstring


def test_a31_seeds_are_declared_and_ordering_holds() -> None:
    metrics = calculate_metrics(_golden_outcomes(), n_bootstrap=500, seed=11)

    fields = {field.name for field in dataclasses.fields(ConfidenceInterval)}
    assert {"estimate", "lower", "upper", "confidence_level", "n_bootstrap", "seed"} <= fields
    assert metrics.sharpe_ci.n_bootstrap == 500
    assert metrics.sortino_ci.n_bootstrap == 500
    assert metrics.sharpe_ci.seed == 11
    assert metrics.sortino_ci.seed == 12
    assert metrics.sharpe_ci.seed != metrics.sortino_ci.seed
    assert metrics.sharpe_ci.estimate == metrics.sharpe
    assert metrics.sortino_ci.estimate == metrics.sortino
    for interval in (metrics.sharpe_ci, metrics.sortino_ci):
        assert interval.lower <= interval.estimate <= interval.upper

    coarse = calculate_metrics(_golden_outcomes(), n_bootstrap=5, seed=3, confidence_level=0.5)
    assert coarse.sharpe_ci.confidence_level == 0.5
    assert coarse.sortino_ci.confidence_level == 0.5


def test_a32_determinism_and_no_global_random() -> None:
    outcomes = _golden_outcomes()
    first = calculate_metrics(outcomes, n_bootstrap=300, seed=7)
    second = calculate_metrics(outcomes, n_bootstrap=300, seed=7)

    assert first == second
    assert repr(first) == repr(second)

    random.seed(0)
    with_zero = calculate_metrics(outcomes, n_bootstrap=300, seed=7)
    random.seed(999)
    with_999 = calculate_metrics(outcomes, n_bootstrap=300, seed=7)

    assert with_zero == with_999
    assert with_zero.sharpe_ci == with_999.sharpe_ci
    assert with_zero.sortino_ci == with_999.sortino_ci


def test_a33_memory_is_bounded() -> None:
    outcomes = _big_outcomes()
    tracemalloc.start()
    try:
        metrics = calculate_metrics(outcomes, n_bootstrap=10_000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert metrics.sharpe_ci.n_bootstrap == 10_000
    assert peak < MEMORY_BUDGET_BYTES, f"pico de memoria {peak / 1024 / 1024:.1f} MB"


_IMPORT_PROBE = textwrap.dedent(
    """
    import sys

    import cfdtrader.backtest.metrics

    print(",".join(sorted({"pandas", "quantstats"} & set(sys.modules))))
    """
)


def test_a34_quantstats_is_only_an_output_adapter(tmp_path: Path) -> None:
    completed = subprocess.run(  # noqa: S603 - el interprete de la sesion
        [sys.executable, "-c", _IMPORT_PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", (
        "importar el modulo deja librerias pesadas en sys.modules: " + completed.stdout
    )

    target = tmp_path / "metrics.html"
    returned = generate_quantstats_report(
        (0.01, -0.005, 0.02),
        target,
        dates=(date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)),
    )
    assert returned == target
    assert target.stat().st_size > 0

    with_benchmark = tmp_path / "metrics_benchmark.html"
    generate_quantstats_report(
        (0.01, -0.005, 0.02),
        with_benchmark,
        benchmark_returns=(0.005, -0.0025, 0.01),
    )
    assert with_benchmark.stat().st_size > 0

    with pytest.raises(MetricsInputError):
        generate_quantstats_report(
            (0.01,), tmp_path / "bad_dates.html", dates=(date(2026, 1, 5), date(2026, 1, 6))
        )
    with pytest.raises(MetricsInputError):
        generate_quantstats_report(
            (0.01,), tmp_path / "bad_benchmark.html", benchmark_returns=(0.01, 0.02)
        )


def test_a35_tests_do_not_dirty_the_repository() -> None:
    git = shutil.which("git")
    assert git is not None, "hace falta git para comprobar que el arbol queda limpio"
    completed = subprocess.run(  # noqa: S603 - el git del sistema
        [git, "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", (
        "los tests han dejado el arbol de trabajo sucio:\n" + completed.stdout
    )
