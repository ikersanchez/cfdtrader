"""Tests de los baselines triviales (#14).

Un test por criterio de aceptación (A1-A36). Lo importante que se comprueba aquí:

- los **seis** identificadores de `plan.md` §11.2, y ningún otro (A2);
- el **listón A** explícitamente identificado (``always_long`` ``open`` -> cierre, solo el
  diferencial) y los listones **B** y **C** fuera, declarados y sin puerta de entrada (A3, A4);
- ``no_trade`` es un artefacto comprobable: ni opera, ni paga, ni produce coste (A6, A7);
- el coste de cada operación es el **declarado** de #11/#8 (0,0042 % de diferencial, sin
  tenencia, sin divisa y sin comisión) y con el supuesto de #64 ``pnl_net_pct`` queda en
  ``null`` con motivo, nunca en 0 (A9, A10);
- **no hay *look-ahead***: mutar ``close_px[i]`` no cambia la decisión de ``i``, y añadir
  sesiones posteriores tampoco (A14);
- la regla aleatoria opera exactamente ``floor(frequency · n_test)`` sesiones distintas, con
  ``Fraction`` exacta y semilla obligatoria, y es reproducible entre procesos y con
  ``PYTHONHASHSEED`` distinto (A19-A23, A26);
- el módulo es **puro**: biblioteca estándar, sin reloj, sin disco y sin métricas (A30, A31).

Los importes declarados se comprueban **contra la tabla de #11**, no contra el resultado del
módulo, y el orden de extracción del azar se reproduce **a mano** con ``random.Random(seed)``.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import os
import random
import re
import subprocess
import sys
import textwrap
from collections.abc import Sequence
from dataclasses import fields, replace
from datetime import date, timedelta
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import pytest

from cfdtrader.backtest import baselines
from cfdtrader.backtest.baselines import (
    ALWAYS_LONG,
    ALWAYS_SHORT,
    BASELINE_IDS,
    GAP_REVERSAL,
    LIMITATIONS,
    MOMENTUM_5D,
    NO_TRADE,
    RANDOM_MATCHED,
    BaselinesError,
    Bias,
    InvalidBaselineParameterError,
    always_long_bias,
    always_short_bias,
    baseline_deciders,
    gap_reversal_bias,
    make_decision_fn,
    momentum_5d_signal,
    no_trade_bias,
    random_matched_deciders,
    random_matched_signal,
    run_baseline,
    run_random_matched,
)
from cfdtrader.backtest.costs import (
    Side,
    declared_cost_model,
    declared_slippage_assumption,
)
from cfdtrader.backtest.engine import (
    BacktestRun,
    Decision,
    DecisionFn,
    Direction,
    EngineInputError,
    SessionInput,
    SessionOutcome,
    SessionView,
)
from cfdtrader.backtest.splits import (
    InsufficientSessionsError,
    SplitPlan,
    walk_forward_splits,
)

MODULE_PATH = Path(str(baselines.__file__))
SOURCE = MODULE_PATH.read_text(encoding="utf-8")
REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DOCSTRING = (REPO_ROOT / "src" / "cfdtrader" / "backtest" / "__init__.py").read_text(
    encoding="utf-8"
)

#: Lo que A1 exige como mínimo en ``__all__``.
REQUIRED_PUBLIC = (
    "BASELINE_IDS",
    "BaselinesError",
    "InvalidBaselineParameterError",
    "Bias",
    "baseline_deciders",
    "random_matched_deciders",
    "momentum_5d_signal",
    "random_matched_signal",
    "no_trade_bias",
    "always_long_bias",
    "always_short_bias",
    "gap_reversal_bias",
    "run_baseline",
    "run_random_matched",
    "LIMITATIONS",
    "BASELINES_DOES_NOT_DO",
    "FOLLOW_UPS",
)

#: Los seis identificadores exactos de A2, en el orden de la tabla de §11.2.
SIX_IDS = (
    "no_trade",
    "always_long",
    "always_short",
    "momentum_5d",
    "gap_reversal",
    "random_matched",
)

#: Módulos que el núcleo puede importar (A31): biblioteca estándar y los contratos de #11-#13.
ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "collections.abc",
        "dataclasses",
        "datetime",
        "decimal",
        "fractions",
        "random",
        "typing",
        "cfdtrader.backtest.costs",
        "cfdtrader.backtest.engine",
        "cfdtrader.backtest.splits",
    }
)

#: Lo que el módulo **no** puede importar (A31).
FORBIDDEN_IMPORTS = (
    "polars",
    "duckdb",
    "pandas",
    "numpy",
    "sklearn",
    "loguru",
    "pydantic",
    "cfdtrader.analysis",
    "cfdtrader.data",
    "cfdtrader.agents",
    "cfdtrader.orchestration",
)

#: Lo que el código **no** puede contener (A26, A29, A31): reloj, azar sin semilla y disco.
FORBIDDEN_TOKENS = (
    "datetime.now",
    "utcnow",
    "date.today",
    "time.time",
    "uuid",
    "random.seed(",
    "random.Random()",
    "Store(",
    "Store.",
    "read_pit",
    "open(",
    "Path(",
    "config/",
    "sql(",
)

#: Métricas que **no** se calculan aquí (A30): son #15.
FORBIDDEN_METRICS = (
    "sharpe",
    "sortino",
    "hit_rate",
    "profit_factor",
    "drawdown",
    "brier",
    "log_loss",
    "bootstrap",
    "equity",
    "expected_value",
)

#: Librerías de backtest prohibidas por ``tech_stack.md`` §4.7 (A36).
FORBIDDEN_BACKTEST_LIBS = ("mlfinlab", "statsmodels", "backtrader", "zipline", "backtesting.py")

NOTIONAL = Decimal("10000")
DECLARED = declared_cost_model()
ASSUMED = declared_slippage_assumption()
START = date(2026, 1, 5)

#: El escenario fijo y reproducible del subproceso de A20 (idéntico al de este proceso).
_SECOND_PROCESS_SCRIPT = textwrap.dedent(
    """
    from datetime import date, timedelta
    from decimal import Decimal
    from fractions import Fraction

    from cfdtrader.backtest import baselines
    from cfdtrader.backtest.costs import declared_cost_model, declared_slippage_assumption
    from cfdtrader.backtest.engine import SessionInput
    from cfdtrader.backtest.splits import walk_forward_splits

    days = []
    day = date(2026, 1, 5)
    while len(days) < 12:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    inputs = [
        SessionInput(
            session=item,
            open_px=100.0 + index,
            high_px=101.0 + index,
            low_px=99.0 + index,
            close_px=100.5 + index,
        )
        for index, item in enumerate(days)
    ]
    plan = walk_forward_splits(
        days, label_horizon=[0] * 12, n_splits=2, test_size=3, embargo_sessions=1
    )
    result = baselines.run_random_matched(
        inputs,
        split_plan=plan,
        cost_model=declared_cost_model(),
        slippage=declared_slippage_assumption(),
        notional_usd=Decimal("10000"),
        frequency=Fraction(1, 3),
        seed=7,
    )
    print(result.run_sha256)
    """
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades del test
# ─────────────────────────────────────────────────────────────────────────────
class ViewProbe:
    """Vista mínima (A13, A28): solo la sesión y, si acaso, el *gap* que publica #13.

    Cualquier otro atributo **explota**: si una ``DecisionFn`` consulta un precio de la vista
    o el futuro de la sesión, el test lo ve en vez de taparlo.
    """

    def __init__(self, session: date = START, gap_px: float | None = None) -> None:
        self.session = session
        self.gap_px = gap_px

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"la DecisionFn ha consultado view.{name} (A13/A28)")


def business_sessions(count: int, *, start: date = START) -> list[date]:
    """``count`` sesiones de lunes a viernes consecutivos (posiciones, no calendario)."""
    out: list[date] = []
    day = start
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def session_input(
    session: date,
    *,
    open_px: float | None = 100.0,
    high_px: float | None = 101.0,
    low_px: float | None = 99.0,
    close_px: float | None = 100.5,
) -> SessionInput:
    """Un ``SessionInput`` sintético con los precios que pida el caso."""
    return SessionInput(
        session=session, open_px=open_px, high_px=high_px, low_px=low_px, close_px=close_px
    )


def flat_inputs(days: Sequence[date], *, base: float = 100.0) -> list[SessionInput]:
    """Sesiones planas y crecientes: ``open`` = ``base + i`` y cierre un punto por encima."""
    return [
        session_input(
            day,
            open_px=base + position,
            high_px=base + position + 1.0,
            low_px=base + position - 1.0,
            close_px=base + position + 0.5,
        )
        for position, day in enumerate(days)
    ]


def make_plan(
    days: Sequence[date], *, n_splits: int = 2, test_size: int = 3, embargo: int = 1
) -> SplitPlan:
    """El plan de #12 para esas sesiones, con el horizonte de #10 (``h = 0``)."""
    return walk_forward_splits(
        days,
        label_horizon=[0] * len(days),
        n_splits=n_splits,
        test_size=test_size,
        embargo_sessions=embargo,
    )


def run_for(
    baseline: str,
    inputs: Sequence[SessionInput],
    plan: SplitPlan,
    *,
    notional: Decimal = NOTIONAL,
    frequency: Fraction = Fraction(1, 3),
    seed: int = 7,
) -> BacktestRun:
    """Corre un baseline por la única vía posible: el motor de #13, con coste explícito."""
    if baseline == RANDOM_MATCHED:
        return run_random_matched(
            inputs,
            split_plan=plan,
            cost_model=DECLARED,
            slippage=ASSUMED,
            notional_usd=notional,
            frequency=frequency,
            seed=seed,
        )
    return run_baseline(
        inputs,
        split_plan=plan,
        cost_model=DECLARED,
        slippage=ASSUMED,
        notional_usd=notional,
        baseline=baseline,
    )


def deciders_for(
    baseline: str,
    inputs: Sequence[SessionInput],
    plan: SplitPlan,
    *,
    notional: Decimal = NOTIONAL,
    frequency: Fraction = Fraction(1, 3),
    seed: int = 7,
) -> tuple[DecisionFn, ...]:
    """Las ``DecisionFn`` del baseline, una por fold, como las entrega el módulo."""
    if baseline == RANDOM_MATCHED:
        return random_matched_deciders(
            inputs, plan, notional_usd=notional, frequency=frequency, seed=seed
        )
    return baseline_deciders(inputs, plan, baseline=baseline, notional_usd=notional)


def outcomes(result: BacktestRun) -> list[SessionOutcome]:
    """Todas las ``SessionOutcome`` de la corrida, en el orden publicado."""
    return [outcome for fold in result.folds for outcome in fold.sessions]


def decisions_of(result: BacktestRun) -> dict[date, Decision]:
    """La ``Decision`` de cada sesión de *test*, indexada por sesión."""
    out: dict[date, Decision] = {}
    for outcome in outcomes(result):
        assert outcome.decision is not None
        out[outcome.session] = outcome.decision
    return out


def outcome_rows(result: BacktestRun) -> list[tuple[object, ...]]:
    """La huella por sesión que A26 compara: sesión, estado, decisión y P&L declarado."""
    return [
        (
            outcome.session,
            outcome.status,
            outcome.skip_reason,
            None if outcome.decision is None else outcome.decision.direction,
            outcome.entry_px,
            outcome.exit_px,
            outcome.exit_reason,
            outcome.notional_usd,
            outcome.gross_pct,
            outcome.pnl_declared_pct,
            outcome.pnl_net_pct,
        )
        for outcome in outcomes(result)
    ]


def positions_of(plan: SplitPlan) -> list[int]:
    """La unión de los ``test`` de los folds, sin repetición (A19)."""
    return sorted({position for fold in plan.folds for position in fold.test})


def traded_of(result: BacktestRun) -> list[SessionOutcome]:
    """Las sesiones realmente operadas."""
    return [outcome for outcome in outcomes(result) if outcome.status == "traded"]


def _fingerprint(root: Path) -> dict[str, str]:
    """Ruta relativa -> sha256 de cada fichero: la huella de ``tests/conftest.py`` (A33)."""
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _code_only() -> str:
    """El código del módulo sin **ningún** literal de cadena (ni docstrings ni declaraciones)."""
    masked_lines: set[int] = set()
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            masked_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return "\n".join(
        "" if number in masked_lines else row
        for number, row in enumerate(SOURCE.splitlines(), start=1)
    )


def _imported_modules() -> set[str]:
    """Módulos importados por el módulo (para demostrar que A31 se cumple)."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def _function_source(name: str) -> str:
    """El texto de una función del módulo **sin** su docstring, por su nombre."""
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            rows = SOURCE.splitlines()[node.lineno - 1 : (node.end_lineno or node.lineno)]
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                skip = (first.value.end_lineno or first.value.lineno) - node.lineno + 1
                rows = rows[skip:]
            return "\n".join(rows)
    raise AssertionError(f"{name} no esta en el modulo")


def flatten(text: str) -> str:
    """El texto con los espacios normalizados: un docstring se parte en líneas."""
    return " ".join(text.split())


def _subprocess_hash(hash_seed: str) -> str:
    """El ``run_sha256`` de A20 calculado en **otro** proceso con ese ``PYTHONHASHSEED``."""
    completed = subprocess.run(  # noqa: S603 - comando fijo (el interprete de la sesion)
        [sys.executable, "-c", _SECOND_PROCESS_SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


# ─────────────────────────────────────────────────────────────────────────────
# A1, A2 — ficheros, API pública y los seis identificadores
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_module_files_and_public_api() -> None:
    assert MODULE_PATH.is_file()
    assert MODULE_PATH.name == "baselines.py"
    assert MODULE_PATH.parent.name == "backtest"
    assert (REPO_ROOT / "tests" / "test_baselines.py").is_file()
    declared = tuple(baselines.__all__)
    for name in REQUIRED_PUBLIC:
        assert name in declared, f"{name} falta en __all__"
    assert issubclass(BaselinesError, Exception)
    assert issubclass(InvalidBaselineParameterError, BaselinesError)
    assert InvalidBaselineParameterError is not BaselinesError
    assert isinstance(BASELINE_IDS, tuple)
    assert callable(baseline_deciders)
    assert callable(random_matched_deciders)
    for name in ("no_trade_bias", "always_long_bias", "always_short_bias", "gap_reversal_bias"):
        assert callable(getattr(baselines, name))


def test_a2_exactly_six_identifiers_and_no_others() -> None:
    assert BASELINE_IDS == SIX_IDS
    assert len(BASELINE_IDS) == 6
    assert len(set(BASELINE_IDS)) == 6
    assert SIX_IDS[:-1] == baselines.DETERMINISTIC_IDS
    assert RANDOM_MATCHED not in baselines.DETERMINISTIC_IDS
    # No hay un séptimo identificador escondido: cualquier otro nombre es error tipado.
    for unknown in ("buy_and_hold", "sma_cross", "momentum_10d", "", "NO_TRADE"):
        with pytest.raises(InvalidBaselineParameterError):
            baseline_deciders(
                flat_inputs(business_sessions(12)),
                make_plan(business_sessions(12)),
                baseline=unknown,
                notional_usd=NOTIONAL,
            )


# ─────────────────────────────────────────────────────────────────────────────
# A3, A4 — el listón A dentro, y los listones B y C fuera
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_always_long_is_the_a_liston() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    result = run_for(ALWAYS_LONG, inputs, plan)
    traded = traded_of(result)
    assert len(traded) == len(positions_of(plan))
    for outcome, position in zip(traded, positions_of(plan), strict=True):
        item = inputs[position]
        assert outcome.decision is not None
        assert outcome.decision.direction is Direction.LONG
        assert outcome.decision.reason.strip()
        assert outcome.entry_px == item.open_px  # entrada = open de la subasta
        assert outcome.exit_px == item.close_px  # salida = cierre de la MISMA sesion
        assert outcome.exit_reason == "session_close"
        assert outcome.cost is not None
        assert outcome.cost.nights == 0  # sin overnight (regla 6)
    doc = flatten(baselines.__doc__ or "")
    assert "listón A" in doc or "liston A" in doc
    assert "session_close" in (always_long_bias.__doc__ or "")


def test_a4_b_and_c_listones_are_out_declared_and_unaskable() -> None:
    text = " ".join(LIMITATIONS)
    assert "#70" in text, "el liston B tiene que estar declarado con su issue"
    assert "#28" in text, "el liston C tiene que estar declarado con su issue"
    assert "no es invertible" in text
    assert "corta la noche" in text or "cruza la noche" in text
    # Ninguna puerta de entrada: ni identificador, ni alias, ni parametro.
    forbidden = ("close", "overnight", "index")
    for name in baselines.__all__:
        assert not any(token in name.lower() for token in forbidden), name
    for function in (baseline_deciders, run_baseline, run_random_matched, random_matched_deciders):
        for parameter in inspect.signature(function).parameters:
            assert not any(token in parameter.lower() for token in forbidden), parameter
    for candidate in ("close_to_close", "overnight", "index", "hold", "buy_and_hold"):
        with pytest.raises(InvalidBaselineParameterError):
            run_baseline(
                flat_inputs(business_sessions(12)),
                split_plan=make_plan(business_sessions(12)),
                cost_model=DECLARED,
                slippage=ASSUMED,
                notional_usd=NOTIONAL,
                baseline=candidate,
            )


# ─────────────────────────────────────────────────────────────────────────────
# A5 — direccion en todas las sesiones, sin consultar ningun precio
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_direction_without_prices_and_missing_open_is_skipped() -> None:
    probe = cast("SessionView", ViewProbe())
    assert always_long_bias(probe).direction is Direction.LONG
    assert always_short_bias(probe).direction is Direction.SHORT
    assert no_trade_bias(probe).direction is Direction.NOTHING
    for bias in (always_long_bias(probe), always_short_bias(probe), no_trade_bias(probe)):
        assert bias.reason.strip()
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    inputs[6] = replace(inputs[6], open_px=None)  # una sesion de test sin apertura
    for baseline in (ALWAYS_LONG, ALWAYS_SHORT):
        result = run_for(baseline, inputs, plan)
        skipped = [outcome for outcome in outcomes(result) if outcome.status == "skipped"]
        assert len(skipped) == 1
        assert skipped[0].session == days[6]
        assert skipped[0].skip_reason == "missing_prices"
        assert skipped[0].decision is not None
        expected = Direction.LONG if baseline == ALWAYS_LONG else Direction.SHORT
        assert skipped[0].decision.direction is expected
        assert skipped[0].decision.reason.strip()


# ─────────────────────────────────────────────────────────────────────────────
# A6, A7 — no operar: ni operacion, ni coste, ni P&L
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_no_trade_never_trades() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    result = run_for(NO_TRADE, flat_inputs(days), plan)
    n_test = len(positions_of(plan))
    assert result.traded == 0
    assert result.no_trade == n_test
    assert result.skipped == 0
    for outcome in outcomes(result):
        assert outcome.decision is not None
        assert outcome.decision.direction is Direction.NOTHING
        assert outcome.notional_usd is None


def test_a7_no_trade_pays_nothing_and_the_convention_is_declared() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    result = run_for(NO_TRADE, flat_inputs(days), plan)
    sessions = outcomes(result)
    assert len(sessions) == len(positions_of(plan))
    assert all(outcome.cost is None for outcome in sessions)
    assert all(outcome.exit_reason is None for outcome in sessions)
    total = sum(
        (outcome.cost.c_declared_usd for outcome in sessions if outcome.cost is not None),
        Decimal(0),
    )
    assert total == Decimal("0")
    doc = flatten(baselines.__doc__ or "")
    assert "Sharpe = 0" in doc
    assert "sharpe" not in _code_only().lower(), "la convencion se declara, no se calcula"
    assert any("Sharpe = 0" in limitation for limitation in LIMITATIONS)


# ─────────────────────────────────────────────────────────────────────────────
# A8 — la unica via de ejecucion es run_walk_forward
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_the_only_execution_path_is_the_engine() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    code = _code_only()
    assert "cost_breakdown" not in code, "el modulo no puede cobrar por su cuenta (A8)"
    assert "CostBreakdown" not in code
    for baseline in BASELINE_IDS:
        deciders = deciders_for(baseline, inputs, plan)
        assert isinstance(deciders, tuple)
        assert len(deciders) == len(plan.folds), "una DecisionFn por fold, en su orden (A8)"
        assert all(callable(decider) for decider in deciders)
        result = run_for(baseline, inputs, plan)
        assert isinstance(result, BacktestRun)
        if baseline != NO_TRADE:
            assert any(outcome.cost is not None for outcome in outcomes(result))


# ─────────────────────────────────────────────────────────────────────────────
# A9, A10, A11 — la tabla declarada, el slippage asumido y el nocional obligatorio
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_every_traded_session_pays_the_declared_table() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    for baseline in (ALWAYS_LONG, ALWAYS_SHORT, MOMENTUM_5D, GAP_REVERSAL, RANDOM_MATCHED):
        for outcome in traded_of(run_for(baseline, inputs, plan)):
            cost = outcome.cost
            assert cost is not None, baseline
            assert cost.nights == 0
            assert cost.spread_pct == Decimal("0.0042")
            assert cost.fx_pct == 0
            assert cost.commission_pct == 0
            assert cost.carry_pct == 0
            assert cost.c_declared_pct > 0
            assert outcome.decision is not None
            expected = Side.LONG if outcome.decision.direction is Direction.LONG else Side.SHORT
            assert cost.side is expected
            assert outcome.notional_usd == NOTIONAL


def test_a10_assumed_slippage_leaves_the_net_pnl_null() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    for baseline in BASELINE_IDS:
        for outcome in traded_of(run_for(baseline, inputs, plan)):
            assert outcome.cost is not None
            assert outcome.cost.c_total_pct is None
            assert outcome.cost.c_total_usd is None
            assert outcome.pnl_net_pct is None, "nunca se rellena con 0"
            assert outcome.pnl_net_reason
            assert "null" in outcome.pnl_net_reason
            assert outcome.pnl_declared_pct is not None
    # ``slippage`` y ``cost_model`` son keyword-only y sin valor por defecto.
    for function in (run_baseline, run_random_matched):
        signature = inspect.signature(function)
        for name in ("cost_model", "slippage", "notional_usd"):
            parameter = signature.parameters[name]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
            assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        cast("Any", run_baseline)(
            inputs,
            split_plan=plan,
            cost_model=DECLARED,
            notional_usd=NOTIONAL,
            baseline=ALWAYS_LONG,
        )
    assert "slippage" not in inspect.signature(baseline_deciders).parameters


def test_a11_the_notional_is_mandatory_explicit_and_flat() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    signature = inspect.signature(baseline_deciders)
    parameter = signature.parameters["notional_usd"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    assert inspect.signature(run_baseline).parameters["notional_usd"].default is (
        inspect.Parameter.empty
    )
    for bad in (None, 10000.0, 1, "10000", Decimal("0"), Decimal("-1")):
        with pytest.raises(InvalidBaselineParameterError):
            baseline_deciders(inputs, plan, baseline=ALWAYS_LONG, notional_usd=cast("Decimal", bad))
    with pytest.raises(TypeError):
        cast("Any", baseline_deciders)(inputs, plan, baseline=ALWAYS_LONG)
    for outcome in traded_of(run_for(ALWAYS_LONG, inputs, plan)):
        assert outcome.notional_usd == NOTIONAL
        assert type(outcome.notional_usd) is Decimal
        assert outcome.decision is not None
        assert outcome.decision.notional_usd == NOTIONAL
    text = " ".join(LIMITATIONS)
    assert "#27" in text
    assert "ilustrativo" in text
    code = _code_only().lower()
    for token in ("capital", "leverage", "apalancamiento", "riesgo_por_operacion"):
        assert token not in code


# ─────────────────────────────────────────────────────────────────────────────
# A12, A13 — momentum 5d: la senal se resuelve fuera del bucle
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_momentum_uses_the_five_previous_closes() -> None:
    days = business_sessions(8)
    closes = (100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 100.0)
    inputs = [session_input(day, close_px=close) for day, close in zip(days, closes, strict=True)]
    signal = momentum_5d_signal(inputs)
    # en la posicion 6: close[5]=105 > close[0]=100 -> largo (retorno de cinco dias > 0)
    assert signal[days[6]].direction is Direction.LONG
    # en la posicion 7: close[6]=106 > close[1]=101 -> largo; y close[7] no interviene
    assert signal[days[7]].direction is Direction.LONG
    falling = [session_input(day, close_px=200.0 - position) for position, day in enumerate(days)]
    negative = momentum_5d_signal(falling)
    assert negative[days[6]].direction is Direction.SHORT
    assert negative[days[7]].direction is Direction.SHORT
    # el cierre de la propia sesion t no cambia su senal (A12/A14)
    mutated = list(inputs)
    mutated[6] = replace(mutated[6], close_px=1.0)
    assert momentum_5d_signal(mutated)[days[6]] == signal[days[6]]
    assert {position: value.direction for position, value in signal.items()} == {
        day: momentum_5d_signal(inputs)[day].direction for day in days
    }


def test_a13_momentum_signal_is_resolved_outside_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    calls: list[int] = []
    original = baselines.momentum_5d_signal

    def spy(items: Sequence[SessionInput]) -> dict[date, Bias]:
        calls.append(len(items))
        return original(items)

    monkeypatch.setattr(baselines, "momentum_5d_signal", spy)
    deciders = baseline_deciders(inputs, plan, baseline=MOMENTUM_5D, notional_usd=NOTIONAL)
    assert calls == [12], "la senal se resuelve una sola vez, fuera del bucle (A13)"
    decision = deciders[0](cast("SessionView", ViewProbe(days[6])))
    assert decision.direction is Direction.LONG
    assert calls == [12], "decidir no recalcula la senal"
    fields_of_view = {field.name for field in fields(SessionView)}
    assert fields_of_view == {"session", "open_px", "gap_px", "context"}
    assert "close_px" not in fields_of_view
    assert "close_px" not in _function_source("_signal_resolver")


# ─────────────────────────────────────────────────────────────────────────────
# A14 — sin look-ahead, probado sesion a sesion
# ─────────────────────────────────────────────────────────────────────────────
def test_a14_no_look_ahead_of_the_session() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    target = 6
    for baseline in BASELINE_IDS:
        before = decisions_of(run_for(baseline, inputs, plan))
        mutated = list(inputs)
        mutated[target] = replace(mutated[target], close_px=999.0)
        after = decisions_of(run_for(baseline, mutated, plan))
        assert after[days[target]] == before[days[target]], baseline
    # ... y si puede cambiar la de i+1 .. i+5: la senal de momentum mira los cierres previos.
    falling = [session_input(day, close_px=100.0 - position) for position, day in enumerate(days)]
    baseline_signal = momentum_5d_signal(falling)
    assert baseline_signal[days[7]].direction is Direction.SHORT
    mutated = list(falling)
    mutated[6] = replace(mutated[6], close_px=1000.0)
    assert momentum_5d_signal(mutated)[days[7]].direction is Direction.LONG
    assert momentum_5d_signal(mutated)[days[6]] == baseline_signal[days[6]]
    # anadir sesiones posteriores tampoco cambia la decision de las anteriores.
    short_days = business_sessions(12)
    long_days = business_sessions(18)
    short_plan = make_plan(short_days, n_splits=3, test_size=3)
    long_plan = make_plan(long_days, n_splits=5, test_size=3)
    shared = sorted(set(positions_of(short_plan)).intersection(positions_of(long_plan)))
    assert shared == positions_of(short_plan)
    short_inputs = flat_inputs(short_days)
    long_inputs = flat_inputs(long_days)
    assert short_inputs == long_inputs[:12]
    for baseline in baselines.DETERMINISTIC_IDS:
        short_decisions = decisions_of(run_for(baseline, short_inputs, short_plan))
        long_decisions = decisions_of(run_for(baseline, long_inputs, long_plan))
        for position in shared:
            assert long_decisions[long_days[position]] == short_decisions[short_days[position]], (
                baseline,
                position,
            )
    # La regla aleatoria no depende de ningun precio, pero su muestra si manda: se declara.
    random_before = decisions_of(run_for(RANDOM_MATCHED, short_inputs, short_plan))
    random_mutated = list(short_inputs)
    random_mutated[target] = replace(random_mutated[target], close_px=999.0)
    random_after = decisions_of(run_for(RANDOM_MATCHED, random_mutated, short_plan))
    assert random_after[short_days[target]] == random_before[short_days[target]]
    assert "muestra de test" in " ".join(LIMITATIONS)


# ─────────────────────────────────────────────────────────────────────────────
# A15, A16 — historia insuficiente y empate
# ─────────────────────────────────────────────────────────────────────────────
def test_a15_momentum_without_history_returns_nothing() -> None:
    days = business_sessions(8)
    inputs = [session_input(day, close_px=100.0 + position) for position, day in enumerate(days)]
    signal = momentum_5d_signal(inputs)
    for position in range(6):
        bias = signal[days[position]]
        assert bias.direction is Direction.NOTHING
        assert "historia insuficiente" in bias.reason
    assert signal[days[6]].direction is Direction.LONG
    missing = list(inputs)
    missing[2] = replace(missing[2], close_px=None)
    assert momentum_5d_signal(missing)[days[6]].direction is Direction.NOTHING
    assert "ausente" in momentum_5d_signal(missing)[days[6]].reason
    zero = list(inputs)
    zero[1] = replace(zero[1], close_px=0.0)
    assert momentum_5d_signal(zero)[days[6]].direction is Direction.NOTHING
    assert "ausente" in momentum_5d_signal(zero)[days[6]].reason
    # nunca lanza y jamas asume una direccion en las sesiones sin senal
    assert set(momentum_5d_signal(inputs)) == set(days)


def test_a16_a_momentum_tie_is_nothing() -> None:
    days = business_sessions(8)
    closes = (100.0, 101.0, 102.0, 103.0, 104.0, 100.0, 99.0, 98.0)
    inputs = [session_input(day, close_px=close) for day, close in zip(days, closes, strict=True)]
    bias = momentum_5d_signal(inputs)[days[6]]
    assert bias.direction is Direction.NOTHING
    assert "empate" in bias.reason
    assert "0" in bias.reason
    assert bias.reason.strip()


# ─────────────────────────────────────────────────────────────────────────────
# A17, A18 — reversion de gap
# ─────────────────────────────────────────────────────────────────────────────
def test_a17_gap_reversal_is_contrary_to_the_view_gap() -> None:
    assert gap_reversal_bias(cast("SessionView", ViewProbe(gap_px=0.0025))).direction is (
        Direction.SHORT
    )
    assert gap_reversal_bias(cast("SessionView", ViewProbe(gap_px=-0.0025))).direction is (
        Direction.LONG
    )
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    for outcome in outcomes(run_for(GAP_REVERSAL, inputs, plan)):
        assert outcome.gap_px is not None
        assert outcome.decision is not None
        expected = Direction.SHORT if outcome.gap_px > 0 else Direction.LONG
        assert outcome.decision.direction is expected
    assert "close_px" not in _function_source("gap_reversal_bias")
    assert "anterior de la secuencia" in flatten(gap_reversal_bias.__doc__ or "")


def test_a18_gap_zero_and_gap_none_are_two_distinct_nothings() -> None:
    tie = gap_reversal_bias(cast("SessionView", ViewProbe(gap_px=0.0)))
    unknown = gap_reversal_bias(cast("SessionView", ViewProbe(gap_px=None)))
    assert tie.direction is Direction.NOTHING
    assert unknown.direction is Direction.NOTHING
    assert tie.reason != unknown.reason
    assert "0" in tie.reason
    assert "primera sesion" in flatten(unknown.reason)


# ─────────────────────────────────────────────────────────────────────────────
# A19 - A23 — la regla aleatoria
# ─────────────────────────────────────────────────────────────────────────────
def test_a19_random_operates_the_exact_frequency() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    n_test = len(positions_of(plan))
    for frequency, expected in (
        (Fraction(0), 0),
        (Fraction(1, 3), n_test // 3),
        (Fraction(1, 2), n_test // 2),
        (Fraction(1), n_test),
    ):
        result = run_for(RANDOM_MATCHED, inputs, plan, frequency=frequency)
        assert result.traded == expected, frequency
        assert result.traded + result.no_trade == n_test
        chosen = {outcome.session for outcome in traded_of(result)}
        assert len(chosen) == expected  # sesiones distintas, sin repeticion
        assert chosen <= {days[position] for position in positions_of(plan)}
    with pytest.raises(InvalidBaselineParameterError):
        baseline_deciders(inputs, plan, baseline=RANDOM_MATCHED, notional_usd=NOTIONAL)


def test_a20_random_is_deterministic_across_processes() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    first = run_for(RANDOM_MATCHED, inputs, plan, seed=7)
    second = run_for(RANDOM_MATCHED, inputs, plan, seed=7)
    other = run_for(RANDOM_MATCHED, inputs, plan, seed=8)
    assert first.run_sha256 == second.run_sha256
    assert first.run_sha256 != other.run_sha256
    assert outcome_rows(first) == outcome_rows(second)
    assert _subprocess_hash("0") == _subprocess_hash("12345") == first.run_sha256
    code = _code_only()
    assert "random.Random(" in code
    assert "numpy" not in _imported_modules()
    for token in ("random.seed(", "random.Random()"):
        assert token not in code


def test_a21_the_selection_is_resolved_outside_the_loop_and_is_pure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    calls: list[int] = []
    original = baselines.random_matched_signal

    def spy(
        sessions: Sequence[SessionInput], *, frequency: Fraction, seed: int
    ) -> dict[date, Bias]:
        calls.append(len(sessions))
        return original(sessions, frequency=frequency, seed=seed)

    monkeypatch.setattr(baselines, "random_matched_signal", spy)
    deciders = random_matched_deciders(
        inputs, plan, notional_usd=NOTIONAL, frequency=Fraction(1, 3), seed=7
    )
    assert calls == [len(positions_of(plan))], "la seleccion se resuelve fuera del bucle (A21)"
    view = cast("SessionView", ViewProbe(days[positions_of(plan)[0]]))
    first = deciders[0](view)
    assert deciders[0](view) == first
    assert deciders[-1](view) == first
    assert calls == [len(positions_of(plan))]
    function = _function_source("make_decision_fn")
    assert "append" not in function, "la DecisionFn no acumula estado (A21)"
    for token in ("global ", "nonlocal "):
        assert token not in function
    for _ in range(50):
        deciders[0](cast("SessionView", ViewProbe(days[-1])))
    assert deciders[0](view) == first
    assert calls == [len(positions_of(plan))]


def test_a22_random_edge_cases() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    n_test = len(positions_of(plan))
    never = run_for(RANDOM_MATCHED, inputs, plan, frequency=Fraction(0))
    assert never.traded == 0
    assert all(
        outcome.decision is not None and outcome.decision.direction is Direction.NOTHING
        for outcome in outcomes(never)
    )
    always = run_for(RANDOM_MATCHED, inputs, plan, frequency=Fraction(1))
    assert always.traded == n_test
    assert all(outcome.status == "traded" for outcome in outcomes(always))
    third = run_for(RANDOM_MATCHED, inputs, plan, frequency=Fraction(1, 3))
    assert third.traded == n_test // 3
    assert third.no_trade == n_test - n_test // 3
    for bad_frequency in (Fraction(-1, 2), Fraction(3, 2), Fraction(2), Fraction(11, 10)):
        with pytest.raises(InvalidBaselineParameterError):
            run_for(RANDOM_MATCHED, inputs, plan, frequency=bad_frequency)
    with pytest.raises(InvalidBaselineParameterError):
        run_for(RANDOM_MATCHED, inputs, plan, frequency=cast("Fraction", 0.5))


def test_a23_random_direction_and_documented_extraction_order() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    positions = positions_of(plan)
    result = run_for(RANDOM_MATCHED, inputs, plan, frequency=Fraction(1, 3), seed=7)
    chosen = {outcome.session: outcome for outcome in outcomes(result)}
    traded = [outcome for outcome in outcomes(result) if outcome.status == "traded"]
    assert traded
    for outcome in traded:
        assert outcome.decision is not None
        assert outcome.decision.direction is not Direction.NOTHING
    # El orden de extraccion documentado: sesiones primero y lados despues, en orden.
    generator = random.Random(7)  # noqa: S311 - reproduccion del orden documentado (A23)
    expected_positions = sorted(generator.sample(range(len(positions)), len(positions) // 3))
    expected_directions: dict[date, Direction] = {}
    for position in expected_positions:
        side = Direction.LONG if generator.random() < 0.5 else Direction.SHORT
        expected_directions[days[positions[position]]] = side
    for session, direction in expected_directions.items():
        outcome = chosen[session]
        assert outcome.decision is not None
        assert outcome.decision.direction is direction
    assert len(expected_directions) == result.traded
    doc = flatten(baselines.__doc__ or "")
    assert "sesiones primero" in doc or "sesiones primero, lados despues" in doc
    assert "sample" in (random_matched_signal.__doc__ or "")


# ─────────────────────────────────────────────────────────────────────────────
# A24, A25 — errores tipados y errores de #12/#13 sin envolver
# ─────────────────────────────────────────────────────────────────────────────
def test_a24_invalid_parameters_raise_a_typed_error() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    with pytest.raises(InvalidBaselineParameterError):
        run_for(RANDOM_MATCHED, inputs, plan, frequency=Fraction(3, 2))
    with pytest.raises(InvalidBaselineParameterError):
        run_for(RANDOM_MATCHED, inputs, plan, frequency=Fraction(-1, 5))
    with pytest.raises(InvalidBaselineParameterError):
        random_matched_deciders(
            inputs, plan, notional_usd=NOTIONAL, frequency=Fraction(1, 2), seed=cast("int", None)
        )
    with pytest.raises(InvalidBaselineParameterError):
        random_matched_deciders(
            inputs, plan, notional_usd=NOTIONAL, frequency=Fraction(1, 2), seed=cast("int", True)
        )
    with pytest.raises(InvalidBaselineParameterError):
        random_matched_deciders(
            inputs, plan, notional_usd=NOTIONAL, frequency=Fraction(1, 2), seed=cast("int", "7")
        )
    with pytest.raises(TypeError):
        cast("Any", random_matched_deciders)(
            inputs, plan, notional_usd=NOTIONAL, frequency=Fraction(1, 2)
        )
    with pytest.raises(InvalidBaselineParameterError):
        baseline_deciders(inputs, plan, baseline=cast("str", "sma_cross"), notional_usd=NOTIONAL)
    duplicated = [inputs[0], inputs[0]]
    with pytest.raises(InvalidBaselineParameterError):
        baseline_deciders(duplicated, plan, baseline=ALWAYS_LONG, notional_usd=NOTIONAL)
    with pytest.raises(InvalidBaselineParameterError):
        momentum_5d_signal(list(reversed(inputs)))
    with pytest.raises(InvalidBaselineParameterError):
        random_matched_signal(
            cast("Sequence[SessionInput]", "nfo"), frequency=Fraction(1, 2), seed=7
        )
    with pytest.raises(InvalidBaselineParameterError):
        baseline_deciders(
            inputs, cast("SplitPlan", "plan"), baseline=ALWAYS_LONG, notional_usd=NOTIONAL
        )
    with pytest.raises(InvalidBaselineParameterError):
        Bias(direction=Direction.NOTHING, reason="   ")
    for bias in (no_trade_bias, always_long_bias, always_short_bias, gap_reversal_bias):
        assert callable(bias)


def test_a25_engine_and_splits_errors_propagate_unwrapped() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    with pytest.raises(EngineInputError):
        run_baseline(
            [],
            split_plan=plan,
            cost_model=DECLARED,
            slippage=ASSUMED,
            notional_usd=NOTIONAL,
            baseline=ALWAYS_LONG,
        )
    with pytest.raises(EngineInputError):
        run_baseline(
            inputs[:6],
            split_plan=plan,
            cost_model=DECLARED,
            slippage=ASSUMED,
            notional_usd=NOTIONAL,
            baseline=ALWAYS_LONG,
        )
    with pytest.raises(InsufficientSessionsError):
        walk_forward_splits(
            days[:4], label_horizon=[0] * 4, n_splits=2, test_size=3, embargo_sessions=1
        )
    assert not issubclass(EngineInputError, BaselinesError)
    assert not issubclass(InsufficientSessionsError, BaselinesError)
    assert "EngineInputError" not in baselines.__all__
    assert "InsufficientSessionsError" not in baselines.__all__
    code = _code_only()
    assert "except EngineError" not in code
    assert "except SplitsError" not in code
    assert code.count("except ") == 1, "el unico except del modulo es el KeyError de la senal"


# ─────────────────────────────────────────────────────────────────────────────
# A26, A27 — determinismo global y cobertura exacta
# ─────────────────────────────────────────────────────────────────────────────
def test_a26_identical_runs_are_byte_identical() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    for baseline in BASELINE_IDS:
        first = run_for(baseline, inputs, plan)
        second = run_for(baseline, inputs, plan)
        assert first.run_sha256 == second.run_sha256, baseline
        assert outcome_rows(first) == outcome_rows(second), baseline
    code = _code_only()
    for token in FORBIDDEN_TOKENS:
        assert token not in code, token
    assert "set(" not in code, "el orden de un set no es una fuente de decision (A26)"


def test_a27_coverage_is_exact_and_skipped_only_by_missing_prices() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    inputs[6] = replace(inputs[6], open_px=None)
    positions = positions_of(plan)
    for baseline in BASELINE_IDS:
        result = run_for(baseline, inputs, plan)
        sessions = outcomes(result)
        assert result.traded + result.no_trade + result.skipped == len(positions), baseline
        expected_order = [days[position] for position in positions]
        assert [outcome.session for outcome in sessions] == expected_order
        assert len({outcome.session for outcome in sessions}) == len(positions)
        for outcome in sessions:
            if outcome.status == "skipped":
                assert outcome.skip_reason == "missing_prices"
            else:
                assert outcome.skip_reason is None


# ─────────────────────────────────────────────────────────────────────────────
# A28, A29, A30 — la vista, la media sesion y las metricas
# ─────────────────────────────────────────────────────────────────────────────
def test_a28_the_deciders_only_see_the_view_of_the_session() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    resolver = make_decision_fn(always_long_bias, notional_usd=NOTIONAL)
    assert list(inspect.signature(resolver).parameters) == ["view"]
    decision = resolver(cast("SessionView", ViewProbe(days[0])))
    assert decision.direction is Direction.LONG
    for baseline in BASELINE_IDS:
        for decider in deciders_for(baseline, inputs, plan):
            probe = cast("SessionView", ViewProbe(days[6], gap_px=0.01))
            assert decider(probe).direction in tuple(Direction)
    assert "high_px" not in _code_only()
    assert "low_px" not in _code_only()
    assert "open_px" not in _code_only()


def test_a29_a_half_session_is_one_more_session_and_there_is_no_clock() -> None:
    assert re.search(r"\d{1,2}:\d{2}", SOURCE) is None, "no puede haber ninguna hora literal (A29)"
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    inputs[8] = replace(inputs[8], close_px=108.25)  # media sesion: su propio cierre
    result = run_for(ALWAYS_LONG, inputs, plan)
    traded = traded_of(result)
    assert len(traded) == len(positions_of(plan))
    for outcome in traded:
        assert outcome.exit_reason == "session_close"
        assert outcome.exit_session == outcome.entry_session == outcome.session
    half = [outcome for outcome in traded if outcome.session == days[8]]
    assert len(half) == 1
    assert half[0].exit_px == 108.25
    code = _code_only().lower()
    for token in ("duration", "timedelta", "hour", "minute"):
        assert token not in code


def test_a30_no_performance_metric_is_computed_here() -> None:
    code = _code_only().lower()
    for token in FORBIDDEN_METRICS:
        assert token not in code, token
    text = " ".join(LIMITATIONS)
    assert "recuentos" in text
    assert "#15" in text
    days = business_sessions(12)
    plan = make_plan(days)
    result = run_for(ALWAYS_LONG, flat_inputs(days), plan)
    assert isinstance(result, BacktestRun)
    assert result.traded == len(positions_of(plan))
    assert not hasattr(result, "sharpe")


# ─────────────────────────────────────────────────────────────────────────────
# A31 — pureza respecto al disco
# ─────────────────────────────────────────────────────────────────────────────
def test_a31_the_module_is_pure_stdlib_and_touches_no_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = _imported_modules()
    assert imported, "el modulo no importa nada: el test no esta mirando el fichero correcto"
    assert imported <= ALLOWED_IMPORTS, f"imports no declarados: {imported - ALLOWED_IMPORTS}"
    for name in FORBIDDEN_IMPORTS:
        assert name not in imported
    code = _code_only()
    for token in FORBIDDEN_TOKENS:
        assert token not in code, token
    monkeypatch.chdir(tmp_path)
    days = business_sessions(12)
    plan = make_plan(days)
    run_for(ALWAYS_LONG, flat_inputs(days), plan)
    run_for(RANDOM_MATCHED, flat_inputs(days), plan)
    assert list(tmp_path.iterdir()) == []


# ─────────────────────────────────────────────────────────────────────────────
# A32, A33 — cobertura de escenarios y datos de prueba
# ─────────────────────────────────────────────────────────────────────────────
def test_a32_the_suite_covers_every_declared_scenario() -> None:
    module = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names = {
        node.name
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }
    required = (
        "test_a2_exactly_six_identifiers_and_no_others",
        "test_a6_no_trade_never_trades",
        "test_a7_no_trade_pays_nothing_and_the_convention_is_declared",
        "test_a3_always_long_is_the_a_liston",
        "test_a5_direction_without_prices_and_missing_open_is_skipped",
        "test_a9_every_traded_session_pays_the_declared_table",
        "test_a10_assumed_slippage_leaves_the_net_pnl_null",
        "test_a12_momentum_uses_the_five_previous_closes",
        "test_a15_momentum_without_history_returns_nothing",
        "test_a16_a_momentum_tie_is_nothing",
        "test_a17_gap_reversal_is_contrary_to_the_view_gap",
        "test_a18_gap_zero_and_gap_none_are_two_distinct_nothings",
        "test_a22_random_edge_cases",
        "test_a20_random_is_deterministic_across_processes",
        "test_a23_random_direction_and_documented_extraction_order",
        "test_a26_identical_runs_are_byte_identical",
        "test_a27_coverage_is_exact_and_skipped_only_by_missing_prices",
        "test_a29_a_half_session_is_one_more_session_and_there_is_no_clock",
        "test_a24_invalid_parameters_raise_a_typed_error",
        "test_a25_engine_and_splits_errors_propagate_unwrapped",
    )
    missing = [name for name in required if name not in names]
    assert missing == [], f"escenarios de A32 sin test: {missing}"


def test_a33_the_tests_do_not_touch_the_repository_data(tmp_path: Path) -> None:
    repository_data = REPO_ROOT / "data"
    before = _fingerprint(repository_data)
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    for baseline in BASELINE_IDS:
        run_for(baseline, inputs, plan)
    scratch = tmp_path / "baselines"
    scratch.mkdir()
    assert _fingerprint(repository_data) == before
    assert list(scratch.iterdir()) == []


# ─────────────────────────────────────────────────────────────────────────────
# A34, A35, A36 — cobertura de criterios, docstring del paquete y tamano del nucleo
# ─────────────────────────────────────────────────────────────────────────────
def test_a34_the_suite_covers_every_criterion() -> None:
    module = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names = [
        node.name
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_a")
    ]
    covered = {int(name.split("_")[1][1:]) for name in names}
    missing = sorted(set(range(1, 37)) - covered)
    assert missing == [], f"criterios sin test: {missing}"
    assert len(names) >= 36
    for table in (baselines.BASELINES_DOES_NOT_DO, baselines.FOLLOW_UPS):
        assert isinstance(table, tuple) and table
        for entry in table:
            assert entry.get("issue", "").startswith("#")
            assert len(entry) >= 3
    frontiers = {entry["issue"] for entry in baselines.BASELINES_DOES_NOT_DO}
    for issue in ("#13", "#15", "#16", "#18", "#27", "#62", "#69", "#70"):
        assert issue in frontiers, issue


def test_a35_package_docstring_mentions_the_new_module() -> None:
    assert "cfdtrader.backtest.baselines" in PACKAGE_DOCSTRING
    assert "#14" in PACKAGE_DOCSTRING
    assert "cfdtrader.backtest.engine" in PACKAGE_DOCSTRING
    assert "cfdtrader.backtest.costs" in PACKAGE_DOCSTRING
    assert "cfdtrader.backtest.splits" in PACKAGE_DOCSTRING
    doc = flatten(baselines.__doc__ or "")
    for token in (
        "#13 ejecuta y cobra",
        "listón A",
        "#70",
        "#28",
        "SessionView",
        "gap_px",
        "plan.md",
        "#69",
        "random",
    ):
        assert token in doc, f"el docstring del modulo no traza {token}"


def test_a36_the_core_is_short_and_no_backtest_library_is_used() -> None:
    core = (
        "no_trade_bias",
        "always_long_bias",
        "always_short_bias",
        "gap_reversal_bias",
        "momentum_5d_signal",
        "_momentum_bias",
        "random_matched_signal",
    )
    total = 0
    found: set[str] = set()
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.FunctionDef) and node.name in core:
            found.add(node.name)
            rows = SOURCE.splitlines()[node.lineno - 1 : (node.end_lineno or node.lineno)]
            first = node.body[0] if node.body else None
            doc_lines = 0
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                doc_lines = (first.value.end_lineno or first.value.lineno) - first.value.lineno + 1
            body = rows[doc_lines:]
            total += sum(1 for row in body if row.strip() and not row.strip().startswith("#"))
    assert found == set(core)
    assert 40 <= total <= 150, f"el nucleo tiene {total} lineas efectivas"
    for name in FORBIDDEN_BACKTEST_LIBS:
        assert name not in SOURCE
        assert name not in _imported_modules()
