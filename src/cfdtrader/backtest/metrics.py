"""Net performance and probability metrics for backtest results (#15).

The walk-forward engine deliberately returns one :class:`SessionOutcome` per test
session.  This module is the aggregation boundary: it consumes those outcomes and
never substitutes ``pnl_declared_pct`` for ``pnl_net_pct``.  A result with an
unmeasured net cost is therefore rejected instead of being presented as a
cost-free performance number.

Returns passed to the standalone metric functions are decimal returns (``0.01``
means one percent).  The engine stores percentages, so :func:`calculate_metrics`
performs that conversion once, at this boundary.

QuantStats is intentionally only an output adapter.  The calculations used for
the report and for tests remain local and auditable.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Final, overload

from cfdtrader.backtest.engine import BacktestRun, SessionOutcome

__all__ = [
    "CalibrationBin",
    "ConfidenceInterval",
    "MetricsError",
    "MetricsInputError",
    "PerformanceMetrics",
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
]

TRADING_DAYS_PER_YEAR: Final[int] = 252
DEFAULT_CONFIDENCE_LEVEL: Final[float] = 0.95
DEFAULT_BOOTSTRAP_SAMPLES: Final[int] = 10_000
DEFAULT_BOOTSTRAP_SEED: Final[int] = 42
DEFAULT_CALIBRATION_BINS: Final[int] = 10
LOG_LOSS_EPSILON: Final[float] = 1e-15


class MetricsError(Exception):
    """Base class for errors raised by the metrics layer."""


class MetricsInputError(MetricsError, ValueError):
    """The supplied outcomes or metric parameters are not admissible."""


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A percentile-bootstrap confidence interval."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_bootstrap: int
    seed: int


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One equal-width bin of the predicted-probability calibration curve."""

    lower: float
    upper: float
    predicted_probability: float
    observed_frequency: float
    count: int


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """All net performance and probability metrics for one evaluated sample.

    Percent fields use percentage points (for example ``1.25`` means 1.25%),
    while ratios and Sharpe/Sortino are dimensionless.  ``max_drawdown_pct`` is
    a positive loss magnitude, not a negative return.
    """

    n_sessions: int
    n_trades: int
    n_skipped: int
    returns: tuple[float, ...]
    equity: tuple[float, ...]
    total_return_pct: float
    sharpe: float
    sortino: float
    sharpe_ci: ConfidenceInterval
    sortino_ci: ConfidenceInterval
    ev_per_trade_pct: float | None
    hit_rate: float | None
    payoff_ratio: float | None
    profit_factor: float | None
    average_win_pct: float | None
    average_loss_pct: float | None
    max_drawdown_pct: float
    max_drawdown_duration: int
    trades_per_year: float
    total_cost_pct: float | None
    total_cost_usd: Decimal | None
    brier_score: float | None
    log_loss: float | None
    calibration: tuple[CalibrationBin, ...]
    benchmark_return_pct: float | None
    alpha_pct: float | None
    beta: float | None
    benchmark_sharpe: float | None

    @property
    def max_drawdown(self) -> float:
        """The maximum drawdown as a positive decimal magnitude."""
        return self.max_drawdown_pct / 100.0

    @property
    def gain_loss_ratio(self) -> float | None:
        """Alias for the mean winning trade divided by mean losing trade."""
        return self.payoff_ratio

    @property
    def drawdown_duration(self) -> int:
        """Alias for the longest drawdown duration in observations."""
        return self.max_drawdown_duration


@overload
def _outcomes(result: BacktestRun) -> tuple[SessionOutcome, ...]: ...


@overload
def _outcomes(result: Iterable[SessionOutcome]) -> tuple[SessionOutcome, ...]: ...


def _outcomes(result: BacktestRun | Iterable[SessionOutcome]) -> tuple[SessionOutcome, ...]:
    if isinstance(result, BacktestRun):
        return tuple(outcome for fold in result.folds for outcome in fold.sessions)
    return tuple(result)


def _validate_confidence(confidence_level: float) -> None:
    if not 0.0 < confidence_level < 1.0 or not math.isfinite(confidence_level):
        raise MetricsInputError("confidence_level debe estar estrictamente entre 0 y 1")


def _validate_bootstrap(n_bootstrap: int, seed: int) -> None:
    if n_bootstrap < 1:
        raise MetricsInputError("n_bootstrap debe ser positivo")


def _validate_returns(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise MetricsInputError("se necesita al menos un retorno")
    if not all(math.isfinite(value) for value in result):
        raise MetricsInputError("los retornos deben ser finitos")
    if any(value <= -1.0 for value in result):
        raise MetricsInputError("un retorno no puede perder más del 100 %")
    return result


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def sharpe_ratio(returns: Sequence[float], *, annualization: int = TRADING_DAYS_PER_YEAR) -> float:
    """Return annualised Sharpe with a zero risk-free rate.

    A zero-variance series returns ``0`` rather than an infinite statistic: it
    contains no evidence of risk-adjusted performance, including the no-trade
    baseline.
    """
    values = _validate_returns(returns)
    if annualization < 1:
        raise MetricsInputError("annualization debe ser positivo")
    deviation = _sample_std(values)
    return 0.0 if deviation == 0.0 else _mean(values) / deviation * math.sqrt(annualization)


def sortino_ratio(
    returns: Sequence[float],
    *,
    annualization: int = TRADING_DAYS_PER_YEAR,
    minimum_acceptable_return: float = 0.0,
) -> float:
    """Return annualised Sortino using downside deviation below zero by default."""
    values = _validate_returns(returns)
    if annualization < 1 or not math.isfinite(minimum_acceptable_return):
        raise MetricsInputError("annualization y minimum_acceptable_return deben ser válidos")
    downside = [min(value - minimum_acceptable_return, 0.0) for value in values]
    deviation = math.sqrt(_mean([value * value for value in downside]))
    if deviation == 0.0:
        return 0.0
    return (_mean(values) - minimum_acceptable_return) / deviation * math.sqrt(annualization)


def bootstrap_confidence_interval(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> ConfidenceInterval:
    """Calculate a deterministic percentile-bootstrap confidence interval.

    Resampling uses Python's seeded Mersenne Twister so the procedure is stable
    across NumPy versions and can be reproduced without a global random state.
    """
    import random

    sample = _validate_returns(values)
    _validate_confidence(confidence_level)
    _validate_bootstrap(n_bootstrap, seed)
    estimate = float(statistic(sample))
    if not math.isfinite(estimate):
        raise MetricsInputError("statistic debe devolver un valor finito")
    rng = random.Random(seed)  # noqa: S311 - reproducible simulation, not cryptography
    bootstrapped: list[float] = []
    for _ in range(n_bootstrap):
        resample = tuple(sample[rng.randrange(len(sample))] for _ in sample)
        value = float(statistic(resample))
        if not math.isfinite(value):
            raise MetricsInputError("statistic debe devolver valores finitos")
        bootstrapped.append(value)
    bootstrapped.sort()
    alpha = (1.0 - confidence_level) / 2.0
    return ConfidenceInterval(
        estimate=estimate,
        lower=_linear_quantile(bootstrapped, alpha),
        upper=_linear_quantile(bootstrapped, 1.0 - alpha),
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


bootstrap_ci = bootstrap_confidence_interval


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def equity_curve(returns: Sequence[float]) -> tuple[float, ...]:
    """Return compounded equity, starting at one after the first observation."""
    values = _validate_returns(returns)
    equity: list[float] = []
    current = 1.0
    for value in values:
        current *= 1.0 + value
        equity.append(current)
    return tuple(equity)


def drawdown_metrics(returns: Sequence[float]) -> tuple[float, int]:
    """Return ``(maximum drawdown magnitude, longest duration)``."""
    equity = equity_curve(returns)
    peak = 1.0
    maximum = 0.0
    current_duration = 0
    maximum_duration = 0
    for value in equity:
        if value >= peak:
            peak = value
            current_duration = 0
            continue
        current_duration += 1
        maximum_duration = max(maximum_duration, current_duration)
        maximum = max(maximum, 1.0 - value / peak)
    return maximum, maximum_duration


def max_drawdown(returns: Sequence[float]) -> float:
    """Return maximum drawdown as a positive decimal magnitude."""
    return drawdown_metrics(returns)[0]


def brier_score(probabilities: Sequence[float], outcomes: Sequence[bool | int | float]) -> float:
    """Return the mean squared probability error."""
    probs, observed = _probability_inputs(probabilities, outcomes)
    return _mean(
        [(probability - target) ** 2 for probability, target in zip(probs, observed, strict=True)]
    )


def log_loss(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int | float],
    *,
    epsilon: float = LOG_LOSS_EPSILON,
) -> float:
    """Return binary log-loss, clipping only at machine-safe probability bounds."""
    probs, observed = _probability_inputs(probabilities, outcomes)
    if not 0.0 < epsilon < 0.5:
        raise MetricsInputError("epsilon debe estar entre 0 y 0,5")
    return -_mean(
        [
            target * math.log(max(epsilon, min(1.0 - epsilon, probability)))
            + (1.0 - target) * math.log(max(epsilon, min(1.0 - epsilon, 1.0 - probability)))
            for probability, target in zip(probs, observed, strict=True)
        ]
    )


def _probability_inputs(
    probabilities: Sequence[float], outcomes: Sequence[bool | int | float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(probabilities) != len(outcomes) or not probabilities:
        raise MetricsInputError(
            "probabilidades y resultados deben tener la misma longitud no vacía"
        )
    probs = tuple(float(value) for value in probabilities)
    observed = tuple(float(bool(value)) for value in outcomes)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probs):
        raise MetricsInputError("las probabilidades deben estar entre 0 y 1")
    return probs, observed


def calibration_curve(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int | float],
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> tuple[CalibrationBin, ...]:
    """Return an equal-width reliability curve, including empty bins as ``count=0``."""
    probs, observed = _probability_inputs(probabilities, outcomes)
    if n_bins < 1:
        raise MetricsInputError("n_bins debe ser positivo")
    grouped: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for probability, target in zip(probs, observed, strict=True):
        index = min(n_bins - 1, int(probability * n_bins))
        grouped[index].append((probability, target))
    width = 1.0 / n_bins
    return tuple(
        CalibrationBin(
            lower=index * width,
            upper=(index + 1) * width,
            predicted_probability=_mean([item[0] for item in values]) if values else math.nan,
            observed_frequency=_mean([item[1] for item in values]) if values else math.nan,
            count=len(values),
        )
        for index, values in enumerate(grouped)
    )


def profit_factor(returns: Sequence[float]) -> float:
    """Return gross profits divided by gross losses in a decimal return series."""
    values = _validate_returns(returns)
    gains = sum(value for value in values if value > 0.0)
    losses = -sum(value for value in values if value < 0.0)
    if losses == 0.0:
        return math.inf if gains > 0.0 else 0.0
    return gains / losses


def _trade_returns(
    outcomes: Sequence[SessionOutcome],
) -> tuple[tuple[float, ...], tuple[SessionOutcome, ...]]:
    trades: list[float] = []
    traded_outcomes: list[SessionOutcome] = []
    for outcome in outcomes:
        if outcome.status != "traded":
            continue
        if outcome.pnl_net_pct is None:
            raise MetricsInputError(
                f"{outcome.session.isoformat()}: pnl_net_pct es null; no se puede publicar "
                "una métrica neta usando pnl_declared_pct"
            )
        trades.append(outcome.pnl_net_pct / 100.0)
        traded_outcomes.append(outcome)
    return tuple(trades), tuple(traded_outcomes)


def _session_returns(outcomes: Sequence[SessionOutcome]) -> tuple[float, ...]:
    returns: list[float] = []
    for outcome in outcomes:
        if outcome.status == "skipped":
            continue
        if outcome.status == "no_trade":
            returns.append(0.0)
            continue
        if outcome.pnl_net_pct is None:
            raise MetricsInputError(
                f"{outcome.session.isoformat()}: pnl_net_pct es null; no se puede publicar "
                "una métrica neta usando pnl_declared_pct"
            )
        returns.append(outcome.pnl_net_pct / 100.0)
    return _validate_returns(returns)


def _probability_metrics(
    traded: Sequence[SessionOutcome],
    *,
    n_bins: int,
) -> tuple[float | None, float | None, tuple[CalibrationBin, ...]]:
    with_probability = [
        outcome
        for outcome in traded
        if outcome.decision is not None and outcome.decision.probability is not None
    ]
    if not with_probability:
        return None, None, ()
    if len(with_probability) != len(traded):
        raise MetricsInputError("todas las operaciones deben declarar probability o ninguna")
    probabilities: list[float] = []
    for outcome in with_probability:
        decision = outcome.decision
        if decision is None or decision.probability is None:
            raise MetricsInputError("probability no puede desaparecer durante la agregación")
        probabilities.append(decision.probability)
    probability_values = tuple(probabilities)
    observed = tuple(
        bool(outcome.pnl_net_pct is not None and outcome.pnl_net_pct > 0.0)
        for outcome in with_probability
    )
    return (
        brier_score(probability_values, observed),
        log_loss(probability_values, observed),
        calibration_curve(probability_values, observed, n_bins=n_bins),
    )


def _cost_totals(traded: Sequence[SessionOutcome]) -> tuple[float | None, Decimal | None]:
    if not traded or any(outcome.cost is None for outcome in traded):
        return None, None
    costs = tuple(outcome.cost for outcome in traded if outcome.cost is not None)
    if any(cost.c_total_pct is None or cost.c_total_usd is None for cost in costs):
        return None, None
    return (
        float(
            sum((cost.c_total_pct for cost in costs if cost.c_total_pct is not None), Decimal("0"))
        ),
        sum((cost.c_total_usd for cost in costs if cost.c_total_usd is not None), Decimal("0")),
    )


def _benchmark_metrics(
    returns: Sequence[float], benchmark_returns: Sequence[float] | None
) -> tuple[float | None, float | None, float | None, float | None]:
    if benchmark_returns is None:
        return None, None, None, None
    benchmark = _validate_returns(benchmark_returns)
    if len(benchmark) != len(returns):
        raise MetricsInputError("benchmark_returns debe tener la misma longitud que los retornos")
    mean_benchmark = _mean(benchmark)
    mean_strategy = _mean(returns)
    variance = sum((value - mean_benchmark) ** 2 for value in benchmark)
    covariance = sum(
        (strategy - mean_strategy) * (benchmark_value - mean_benchmark)
        for strategy, benchmark_value in zip(returns, benchmark, strict=True)
    )
    beta = None if variance == 0.0 else covariance / variance
    return (
        (math.prod(1.0 + value for value in benchmark) - 1.0) * 100.0,
        (mean_strategy - mean_benchmark) * 100.0,
        beta,
        sharpe_ratio(benchmark),
    )


def calculate_metrics(
    result: BacktestRun | Iterable[SessionOutcome],
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    n_calibration_bins: int = DEFAULT_CALIBRATION_BINS,
    benchmark_returns: Sequence[float] | None = None,
) -> PerformanceMetrics:
    """Aggregate a run or session outcomes into net performance metrics.

    Non-trading sessions contribute a zero return to the daily risk series;
    skipped sessions are excluded.  A traded session without a measured
    ``pnl_net_pct`` is a hard error, preserving the project's ``null != 0`` rule.
    """
    outcomes = _outcomes(result)
    if not outcomes:
        raise MetricsInputError("el resultado no contiene sesiones")
    _validate_confidence(confidence_level)
    _validate_bootstrap(n_bootstrap, seed)
    if n_calibration_bins < 1:
        raise MetricsInputError("n_calibration_bins debe ser positivo")
    returns = _session_returns(outcomes)
    trades, traded_outcomes = _trade_returns(outcomes)
    equity = equity_curve(returns)
    max_dd, dd_duration = drawdown_metrics(returns)
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    sharpe_ci = bootstrap_confidence_interval(
        returns,
        sharpe_ratio,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    sortino_ci = bootstrap_confidence_interval(
        returns,
        sortino_ratio,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
    )
    wins = tuple(value for value in trades if value > 0.0)
    losses = tuple(value for value in trades if value < 0.0)
    first = next((outcome.session for outcome in outcomes if outcome.status != "skipped"), None)
    last = next(
        (outcome.session for outcome in reversed(outcomes) if outcome.status != "skipped"),
        None,
    )
    span_days = 0 if first is None or last is None else max(1, (last - first).days)
    years = span_days / 365.25 if span_days else 1.0 / TRADING_DAYS_PER_YEAR
    brier, loss, calibration = _probability_metrics(traded_outcomes, n_bins=n_calibration_bins)
    cost_pct, cost_usd = _cost_totals(traded_outcomes)
    benchmark_return, alpha, beta, benchmark_sharpe = _benchmark_metrics(returns, benchmark_returns)
    return PerformanceMetrics(
        n_sessions=len(outcomes),
        n_trades=len(trades),
        n_skipped=sum(outcome.status == "skipped" for outcome in outcomes),
        returns=returns,
        equity=equity,
        total_return_pct=(equity[-1] - 1.0) * 100.0,
        sharpe=sharpe,
        sortino=sortino,
        sharpe_ci=sharpe_ci,
        sortino_ci=sortino_ci,
        ev_per_trade_pct=None if not trades else _mean(trades) * 100.0,
        hit_rate=None if not trades else len(wins) / len(trades),
        payoff_ratio=None if not wins or not losses else _mean(wins) / abs(_mean(losses)),
        profit_factor=None if not trades else profit_factor(trades),
        average_win_pct=None if not wins else _mean(wins) * 100.0,
        average_loss_pct=None if not losses else _mean(losses) * 100.0,
        max_drawdown_pct=max_dd * 100.0,
        max_drawdown_duration=dd_duration,
        trades_per_year=len(trades) / years,
        total_cost_pct=None if cost_pct is None else float(cost_pct),
        total_cost_usd=cost_usd,
        brier_score=brier,
        log_loss=loss,
        calibration=calibration,
        benchmark_return_pct=benchmark_return,
        alpha_pct=alpha,
        beta=beta,
        benchmark_sharpe=benchmark_sharpe,
    )


def generate_quantstats_report(
    returns: Sequence[float],
    output_path: str | Path,
    *,
    dates: Sequence[date] | None = None,
    benchmark_returns: Sequence[float] | None = None,
    title: str = "cfdtrader net performance",
) -> Path:
    """Write a QuantStats HTML tear sheet and return its path.

    QuantStats receives decimal returns, as required by its public API.  Dates
    default to consecutive weekdays only as a presentation convenience; callers
    with actual sessions should pass their dates explicitly.
    """
    import pandas as pd
    import quantstats as qs

    values = _validate_returns(returns)
    if dates is None:
        index = pd.date_range(  # pyright: ignore[reportUnknownMemberType]
            "2000-01-03", periods=len(values), freq="B"
        )
    else:
        if len(dates) != len(values):
            raise MetricsInputError("dates debe tener la misma longitud que returns")
        index = pd.DatetimeIndex(dates)
    series = pd.Series(values, index=index, name="strategy")
    benchmark = None
    if benchmark_returns is not None:
        benchmark_values = _validate_returns(benchmark_returns)
        if len(benchmark_values) != len(values):
            raise MetricsInputError("benchmark_returns debe tener la misma longitud que returns")
        benchmark = pd.Series(benchmark_values, index=index, name="benchmark")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if benchmark is not None:
        qs.reports.html(  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            series, benchmark=benchmark, output=str(target), title=title
        )
    else:
        qs.reports.html(  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            series, output=str(target), title=title
        )
    return target
