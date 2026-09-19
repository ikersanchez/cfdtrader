"""Net performance and probability metrics for backtest results (#15).

The walk-forward engine (#13) deliberately returns one :class:`SessionOutcome` per test
session.  This module is the aggregation boundary: it consumes those outcomes and never
substitutes ``pnl_declared_pct`` for ``pnl_net_pct``.  A result with an unmeasured net
cost is therefore rejected instead of being presented as a cost-free performance number.

Returns passed to the standalone metric functions are decimal returns (``0.01`` means one
percent).  The engine stores percentages, so :func:`calculate_metrics` performs that
conversion once, at this boundary.  Units are declared and tested (A4): ``returns`` and
``equity`` are decimals, every ``*_pct`` field is in percentage points,
``max_drawdown_pct`` is a positive loss magnitude, and Sharpe, Sortino, beta, payoff and
profit factor are dimensionless.

Contracts this module fixes, because #15 found them ambiguous:

- ``alpha_pct`` is **Jensen's alpha** (``mean_strategy - beta * mean_benchmark`` with a
  zero risk-free rate), never the difference of means.  That difference survives, with an
  honest name, in ``mean_excess_return_pct``.  A zero-variance benchmark leaves ``beta``
  and ``alpha_pct`` as ``None``: ``null != 0``.
- ``max_drawdown_duration_sessions`` counts observations of the **risk series**
  (``traded`` + ``no_trade``; a ``skipped`` session never counts), never calendar days.
- ``profit_factor`` is ``None`` when there are no losing trades, because ``inf`` is not
  JSON: the module never publishes ``inf`` or ``nan``, and an empty calibration bin
  carries ``None`` instead of ``nan``.
- ``trades_per_year`` is ``None`` while the sample span is shorter than 30 natural days.
  Annualising a single session is an artefact, not a datum; ``session_span_days`` is
  published so the turnover can be audited.
- A session whose ``status`` is neither ``traded``, ``no_trade`` nor ``skipped`` is a
  :class:`MetricsInputError` naming the session and the status: it is never read as an
  operation.

Costs: with the *slippage* the engine produces today (the declared pessimistic assumption
of #64: ``state = "assumed"``) ``c_total_pct`` is ``null``, so every traded session has
``pnl_net_pct`` ``null`` and :func:`calculate_metrics` raises :class:`MetricsInputError`
instead of publishing a net metric built on ``pnl_declared_pct``.  That is the correct
behaviour until the execution *slippage* is measured (**#62**) and the size of ``R`` is
decided (**#60**): substituting either of them here is forbidden.

The *bootstrap* is vectorised with NumPy and a **legacy** ``numpy.random.RandomState``
(Mersenne Twister).  NumPy keeps the ``RandomState`` stream frozen for compatibility, so
the same input and the same seed give the same interval byte to byte, in this process and
in any other one.  The resamples are drawn in blocks, so the full
``n_bootstrap x n_sessions`` matrix is never materialised.

Benchmark: ``benchmark_returns`` arrives **already aligned session by session** with the
risk series.  This module does not align series by date, does not read the ``Store``
(adapter #69) and does not write the report (#18).

QuantStats is intentionally only an output adapter, imported lazily inside
:func:`generate_quantstats_report`.  The calculations used for the report and for tests
remain local and auditable, and pandas appears only because QuantStats requires it
(``tech_stack.md`` section 4.4).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

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

#: Below this span (natural days) the turnover is not published: see ``trades_per_year``.
MIN_TURNOVER_SPAN_DAYS: Final[int] = 30

#: Resamples drawn per block: the ``n_bootstrap x n_sessions`` matrix is never materialised.
BOOTSTRAP_CHUNK_SIZE: Final[int] = 512

#: ``numpy.random.RandomState`` accepts seeds in ``[0, 2**32)``.
MAX_SEED: Final[int] = 2**32

#: The two annualised risk ratios the block bootstrap computes directly (A29, A33).
StatisticKind = Literal["sharpe", "sortino"]

#: The three states the engine publishes; anything else is a contract breach (A21).
KNOWN_STATUSES: Final[frozenset[str]] = frozenset({"traded", "no_trade", "skipped"})


class MetricsError(Exception):
    """Base class for errors raised by the metrics layer."""


class MetricsInputError(MetricsError, ValueError):
    """The supplied outcomes or metric parameters are not admissible."""


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A percentile-bootstrap confidence interval, with the PRNG seed published (A31)."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_bootstrap: int
    seed: int


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One equal-width bin of the predicted-probability calibration curve.

    An empty bin publishes ``count == 0`` with ``predicted_probability`` and
    ``observed_frequency`` set to ``None``: ``nan`` is never published (A11).
    """

    lower: float
    upper: float
    predicted_probability: float | None
    observed_frequency: float | None
    count: int


def _interval_payload(interval: ConfidenceInterval) -> dict[str, object]:
    """The interval as plain values, including the ``seed`` and ``n_bootstrap`` of A31."""
    return {
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "confidence_level": interval.confidence_level,
        "n_bootstrap": interval.n_bootstrap,
        "seed": interval.seed,
    }


def _bin_payload(item: CalibrationBin) -> dict[str, object]:
    """One calibration bin as plain values: an empty bin publishes ``None``, never ``nan``."""
    return {
        "lower": item.lower,
        "upper": item.upper,
        "predicted_probability": item.predicted_probability,
        "observed_frequency": item.observed_frequency,
        "count": item.count,
    }


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """All net performance and probability metrics for one evaluated sample.

    Percent fields use percentage points (for example ``1.25`` means 1.25%), ``returns``
    and ``equity`` use decimals (``0.01`` means one percent), ``max_drawdown_pct`` is a
    positive loss magnitude, and ratios and Sharpe/Sortino are dimensionless.

    A field is ``None`` when the sample does not contain the evidence it needs
    (``null != 0``): no trades, no losses, no declared probabilities or no measured cost.
    No field is ever ``inf`` or ``nan``.

    ``max_drawdown_duration_sessions`` counts observations of the risk series, never days;
    ``session_span_days`` carries the span that ``trades_per_year`` annualises.
    """

    n_sessions: int
    n_trades: int
    n_no_trade: int
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
    max_drawdown_duration_sessions: int
    session_span_days: int
    trades_per_year: float | None
    total_cost_pct: float | None
    total_cost_usd: Decimal | None
    brier_score: float | None
    log_loss: float | None
    calibration: tuple[CalibrationBin, ...]
    benchmark_return_pct: float | None
    mean_excess_return_pct: float | None
    alpha_pct: float | None
    beta: float | None
    benchmark_sharpe: float | None

    @property
    def max_drawdown(self) -> float:
        """The maximum drawdown as a positive decimal magnitude."""
        return self.max_drawdown_pct / 100.0

    @property
    def gain_loss_ratio(self) -> float | None:
        """Alias for the mean winning trade divided by the mean losing trade."""
        return self.payoff_ratio

    def to_payload(self) -> dict[str, object]:
        """A JSON-safe view of every published field, with ``Decimal`` as an exact string.

        This is the object A12 walks: no ``float`` in it is ``inf`` or ``nan``, and
        ``json.dumps(payload, allow_nan=False)`` never raises.
        """
        cost_usd = self.total_cost_usd
        return {
            "n_sessions": self.n_sessions,
            "n_trades": self.n_trades,
            "n_no_trade": self.n_no_trade,
            "n_skipped": self.n_skipped,
            "returns": list(self.returns),
            "equity": list(self.equity),
            "total_return_pct": self.total_return_pct,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "sharpe_ci": _interval_payload(self.sharpe_ci),
            "sortino_ci": _interval_payload(self.sortino_ci),
            "ev_per_trade_pct": self.ev_per_trade_pct,
            "hit_rate": self.hit_rate,
            "payoff_ratio": self.payoff_ratio,
            "profit_factor": self.profit_factor,
            "average_win_pct": self.average_win_pct,
            "average_loss_pct": self.average_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_duration_sessions": self.max_drawdown_duration_sessions,
            "session_span_days": self.session_span_days,
            "trades_per_year": self.trades_per_year,
            "total_cost_pct": self.total_cost_pct,
            "total_cost_usd": None if cost_usd is None else format(cost_usd, "f"),
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "calibration": [_bin_payload(item) for item in self.calibration],
            "benchmark_return_pct": self.benchmark_return_pct,
            "mean_excess_return_pct": self.mean_excess_return_pct,
            "alpha_pct": self.alpha_pct,
            "beta": self.beta,
            "benchmark_sharpe": self.benchmark_sharpe,
        }


def _outcomes(result: BacktestRun | Iterable[SessionOutcome]) -> tuple[SessionOutcome, ...]:
    """The ordered session outcomes of a run (its folds) or of a plain iterable."""
    if isinstance(result, BacktestRun):
        return tuple(outcome for fold in result.folds for outcome in fold.sessions)
    return tuple(result)


def _validate_confidence(confidence_level: float) -> None:
    if not 0.0 < confidence_level < 1.0 or not math.isfinite(confidence_level):
        raise MetricsInputError("confidence_level debe estar estrictamente entre 0 y 1")


def _validate_bootstrap(n_bootstrap: int, seed: int) -> None:
    if n_bootstrap < 1:
        raise MetricsInputError("n_bootstrap debe ser positivo")
    if not 0 <= seed < MAX_SEED:
        raise MetricsInputError(
            f"seed debe estar en [0, {MAX_SEED}); {seed} no lo esta: el generador declarado "
            "no admite ese valor"
        )


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
    """Sample standard deviation, exactly ``0`` when the series is constant (A16, A18).

    ``max - min`` is the exact test: a floating-point ``sqrt`` of a constant series can
    leave a tiny non-zero residue, and a zero-variance series must report a Sharpe of
    ``0`` rather than a huge ratio built on that residue.
    """
    if len(values) < 2 or max(values) == min(values):
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

    The resamples come from the declared ``numpy.random.RandomState`` stream seeded with
    ``seed`` (a legacy Mersenne Twister whose stream NumPy keeps frozen), so the procedure
    is reproducible without touching any global random state.  The resamples are drawn in
    blocks: the full ``n_bootstrap x n_sessions`` matrix is never materialised (A33).

    :func:`calculate_metrics` uses the vectorised Sharpe/Sortino path for its own
    intervals; this generic entry point stays for any other statistic.
    """
    sample = _validate_returns(values)
    _validate_confidence(confidence_level)
    _validate_bootstrap(n_bootstrap, seed)
    estimate = float(statistic(sample))
    if not math.isfinite(estimate):
        raise MetricsInputError("statistic debe devolver un valor finito")
    size = len(sample)
    rng = np.random.RandomState(seed)
    bootstrapped: list[float] = []
    remaining = n_bootstrap
    while remaining > 0:
        rows = min(BOOTSTRAP_CHUNK_SIZE, remaining)
        for indices in rng.randint(0, size, size=(rows, size)):
            resample = tuple(sample[int(index)] for index in indices)
            value = float(statistic(resample))
            if not math.isfinite(value):
                raise MetricsInputError("statistic debe devolver valores finitos")
            bootstrapped.append(value)
        remaining -= rows
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
    """Linear-interpolation quantile of an ascending sequence."""
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def _sorted_quantile(values: npt.NDArray[np.float64], probability: float) -> float:
    """Linear-interpolation quantile of an ascending array, same rule as the generic one."""
    position = (int(values.size) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def _bootstrap_draws(
    sample: npt.NDArray[np.float64],
    *,
    kind: StatisticKind,
    n_bootstrap: int,
    seed: int,
) -> npt.NDArray[np.float64]:
    """The bootstrap distribution of one annualised risk ratio, in blocks (A29, A33).

    Both ratios follow the standalone estimators exactly: Sharpe divides by the sample
    standard deviation with ``ddof = 1`` and Sortino by the root mean square of the
    downside.  A constant resample is detected with ``max - min == 0`` -- the exact test --
    because the floating-point deviation of a constant array is a tiny non-zero residue
    that would otherwise become a huge ratio.
    """
    size = int(sample.size)
    annualization = math.sqrt(TRADING_DAYS_PER_YEAR)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    rng = np.random.RandomState(seed)
    remaining = n_bootstrap
    cursor = 0
    while remaining > 0:
        rows = min(BOOTSTRAP_CHUNK_SIZE, remaining)
        chunk = sample[rng.randint(0, size, size=(rows, size))]
        means = chunk.mean(axis=1)
        constant = np.asarray(chunk.max(axis=1) - chunk.min(axis=1)) == 0.0
        if kind == "sharpe":
            spread = np.zeros(rows, dtype=np.float64) if size < 2 else chunk.std(axis=1, ddof=1)
            deviations = np.where(constant, 0.0, spread)
        else:
            downside = np.minimum(chunk, 0.0)
            deviations = np.sqrt((downside * downside).mean(axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = means / deviations * annualization
        draws[cursor : cursor + rows] = np.where(deviations == 0.0, 0.0, ratios)
        cursor += rows
        remaining -= rows
    draws.sort()
    return draws


def _vectorized_confidence_interval(
    returns: tuple[float, ...],
    *,
    kind: StatisticKind,
    estimate: float,
    confidence_level: float,
    n_bootstrap: int,
    seed: int,
) -> ConfidenceInterval:
    """The block-bootstrap interval of Sharpe or Sortino for the risk series (A29, A31)."""
    sample = np.asarray(returns, dtype=np.float64)
    draws = _bootstrap_draws(sample, kind=kind, n_bootstrap=n_bootstrap, seed=seed)
    alpha = (1.0 - confidence_level) / 2.0
    return ConfidenceInterval(
        estimate=estimate,
        lower=_sorted_quantile(draws, alpha),
        upper=_sorted_quantile(draws, 1.0 - alpha),
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


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
    """Return an equal-width reliability curve, including the empty bins.

    An empty bin publishes ``count == 0`` and ``None`` for its two moments: ``nan`` is not
    a published value in this module (A11).
    """
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
            predicted_probability=_mean([item[0] for item in values]) if values else None,
            observed_frequency=_mean([item[1] for item in values]) if values else None,
            count=len(values),
        )
        for index, values in enumerate(grouped)
    )


def profit_factor(returns: Sequence[float]) -> float | None:
    """Gross profits divided by gross losses, or ``None`` when there are no losses (A10).

    ``None`` is neither ``0`` nor ``inf``: a series with gains but no losses has an
    undefined profit factor, and ``json.dumps`` would turn ``inf`` into ``Infinity``, which
    is not valid JSON.  :func:`calculate_metrics` distinguishes this case from "no trades"
    through the fields that are still published (``n_trades``, ``hit_rate``,
    ``average_loss_pct``).
    """
    values = _validate_returns(returns)
    gains = sum(value for value in values if value > 0.0)
    losses = -sum(value for value in values if value < 0.0)
    if losses == 0.0:
        return None
    return gains / losses


def _require_known_status(outcome: SessionOutcome) -> str:
    """Reject any state the engine never publishes, naming session and status (A21)."""
    status = outcome.status
    if status not in KNOWN_STATUSES:
        raise MetricsInputError(
            f"{outcome.session.isoformat()}: estado de sesion desconocido {status!r}; solo se "
            "admite 'traded', 'no_trade' o 'skipped' y nunca se lee como una operacion"
        )
    return status


def _net_return(outcome: SessionOutcome) -> float:
    """The net decimal return of one traded session, or a hard error (A22)."""
    value = outcome.pnl_net_pct
    if value is None:
        raise MetricsInputError(
            f"{outcome.session.isoformat()}: pnl_net_pct es null; no se puede publicar "
            "una métrica neta usando pnl_declared_pct (null != 0)"
        )
    return value / 100.0


def _risk_returns(outcomes: Sequence[SessionOutcome]) -> tuple[float, ...]:
    """The risk series: every ``traded`` and ``no_trade`` session, in order (A2, A16).

    A ``no_trade`` session contributes a zero decimal return and therefore dilutes the
    series; a ``skipped`` session is excluded from it.
    """
    returns: list[float] = []
    for outcome in outcomes:
        status = _require_known_status(outcome)
        if status == "skipped":
            continue
        returns.append(0.0 if status == "no_trade" else _net_return(outcome))
    return _validate_returns(returns)


def _trade_returns(
    outcomes: Sequence[SessionOutcome],
) -> tuple[tuple[float, ...], tuple[SessionOutcome, ...]]:
    """The traded sessions and their net decimal returns; a null net P&L is an error (A22)."""
    trades: list[float] = []
    traded_outcomes: list[SessionOutcome] = []
    for outcome in outcomes:
        if _require_known_status(outcome) != "traded":
            continue
        trades.append(_net_return(outcome))
        traded_outcomes.append(outcome)
    return tuple(trades), tuple(traded_outcomes)


def _probability_metrics(
    traded: Sequence[SessionOutcome],
    *,
    n_bins: int,
) -> tuple[float | None, float | None, tuple[CalibrationBin, ...]]:
    """Brier, log-loss and the reliability curve of the declared probabilities (A11).

    Either **all** traded sessions declare a probability or none does: a partial set would
    silently change the sample the calibration is read on, so it is an error.
    """
    declared: list[float] = []
    observed: list[bool] = []
    answered = 0
    for outcome in traded:
        probability = None if outcome.decision is None else outcome.decision.probability
        if probability is None:
            continue
        answered += 1
        declared.append(probability)
        observed.append(outcome.pnl_net_pct is not None and outcome.pnl_net_pct > 0.0)
    if not declared:
        return None, None, ()
    if answered != len(traded):
        raise MetricsInputError("todas las operaciones deben declarar probability o ninguna")
    probabilities = tuple(declared)
    outcomes = tuple(observed)
    return (
        brier_score(probabilities, outcomes),
        log_loss(probabilities, outcomes),
        calibration_curve(probabilities, outcomes, n_bins=n_bins),
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
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Benchmark return, mean excess, Jensen's alpha, beta and benchmark Sharpe (A5-A9).

    ``returns`` and ``benchmark_returns`` are the two risk series, **already aligned
    session by session**: this module never aligns by date.  With
    ``benchmark_returns is None`` all five values are ``None`` (``null != 0``); with a
    zero-variance benchmark ``beta`` and ``alpha_pct`` are ``None``, because no beta can be
    estimated from it and publishing an "alpha" without a beta is not allowed.
    """
    if benchmark_returns is None:
        return None, None, None, None, None
    benchmark = _validate_returns(benchmark_returns)
    if len(benchmark) != len(returns):
        raise MetricsInputError(
            "benchmark_returns debe tener la misma longitud que la serie de riesgo "
            f"({len(benchmark)} != {len(returns)}): la serie llega ya alineada sesion a "
            "sesion y este modulo no alinea por fecha"
        )
    mean_benchmark = _mean(benchmark)
    mean_strategy = _mean(returns)
    variance = sum((value - mean_benchmark) ** 2 for value in benchmark)
    covariance = sum(
        (strategy - mean_strategy) * (benchmark_value - mean_benchmark)
        for strategy, benchmark_value in zip(returns, benchmark, strict=True)
    )
    beta = None if variance == 0.0 else covariance / variance
    alpha = None if beta is None else mean_strategy - beta * mean_benchmark
    return (
        (math.prod(1.0 + value for value in benchmark) - 1.0) * 100.0,
        (mean_strategy - mean_benchmark) * 100.0,
        None if alpha is None else alpha * 100.0,
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
    """Aggregate a run or a sequence of session outcomes into net metrics.

    A ``no_trade`` session contributes a zero decimal return to the risk series; a
    ``skipped`` session is excluded from it.  A traded session without a measured
    ``pnl_net_pct`` is a hard error, preserving the project's ``null != 0`` rule: with the
    *slippage* assumption of #64 the whole run is rejected instead of being read as
    cost-free (A22, A23).

    The count identities hold by construction and are documented and tested (A3):
    ``n_sessions == n_trades + n_no_trade + n_skipped == len(outcomes)``,
    ``len(returns) == n_sessions - n_skipped == len(equity)`` and
    ``total_return_pct == (equity[-1] - 1) * 100``.
    """
    outcomes = _outcomes(result)
    if not outcomes:
        raise MetricsInputError("el resultado no contiene sesiones")
    _validate_confidence(confidence_level)
    _validate_bootstrap(n_bootstrap, seed)
    if n_calibration_bins < 1:
        raise MetricsInputError("n_calibration_bins debe ser positivo")
    returns = _risk_returns(outcomes)
    trades, traded_outcomes = _trade_returns(outcomes)
    n_trades = len(trades)
    n_no_trade = sum(1 for outcome in outcomes if outcome.status == "no_trade")
    n_skipped = sum(1 for outcome in outcomes if outcome.status == "skipped")
    equity = equity_curve(returns)
    max_dd, dd_duration = drawdown_metrics(returns)
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    sharpe_ci = _vectorized_confidence_interval(
        returns,
        kind="sharpe",
        estimate=sharpe,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    sortino_ci = _vectorized_confidence_interval(
        returns,
        kind="sortino",
        estimate=sortino,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
    )
    wins = tuple(value for value in trades if value > 0.0)
    losses = tuple(value for value in trades if value < 0.0)
    sessions = [outcome.session for outcome in outcomes if outcome.status != "skipped"]
    session_span_days = (max(sessions) - min(sessions)).days if sessions else 0
    brier, loss, calibration = _probability_metrics(traded_outcomes, n_bins=n_calibration_bins)
    cost_pct, cost_usd = _cost_totals(traded_outcomes)
    benchmark_return, mean_excess, alpha, beta, benchmark_sharpe = _benchmark_metrics(
        returns, benchmark_returns
    )
    return PerformanceMetrics(
        n_sessions=len(outcomes),
        n_trades=n_trades,
        n_no_trade=n_no_trade,
        n_skipped=n_skipped,
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
        max_drawdown_duration_sessions=dd_duration,
        session_span_days=session_span_days,
        trades_per_year=(
            None
            if session_span_days < MIN_TURNOVER_SPAN_DAYS
            else n_trades * 365.25 / session_span_days
        ),
        total_cost_pct=None if cost_pct is None else float(cost_pct),
        total_cost_usd=cost_usd,
        brier_score=brier,
        log_loss=loss,
        calibration=calibration,
        benchmark_return_pct=benchmark_return,
        mean_excess_return_pct=mean_excess,
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
