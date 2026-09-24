"""Tests del motor *walk-forward* (#13).

Un test por criterio de aceptación (A1-A35). Lo importante que se comprueba aquí:

- la vista de decisión **no** expone el futuro de la sesión (A5) y el decididor se llama
  **una sola vez** por sesión de *test*, dentro de su fold (A4);
- el *gap* se mide contra el cierre de la sesión anterior **de la secuencia** y **no** se
  opera (A6);
- la salida se resuelve con la primera barrera tocada, a su precio, y en empate gana la
  adversa (A21, A22);
- el coste lo cobra **#11**: una sola llamada por operación, con los mismos argumentos, y
  el `CostBreakdown` del resultado es **exactamente** el suyo (A15, A16);
- la purga y el embargo del plan se **publican**, no se afirman (A8), y ninguna sesión se
  pierde por el camino (A9);
- la corrida es determinista byte a byte, también **entre procesos** (A28, A29).

Los importes declarados se calculan **a mano** en el test (0,42 $ / 0,0042 % sobre 10.000 $
de nocional) y el P&L se comprueba contra la fórmula, no contra el resultado del módulo.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import subprocess
import sys
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import FrozenInstanceError, fields
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from cfdtrader.analysis.cost_audit import MeasureState, Side
from cfdtrader.backtest import costs, engine, splits
from cfdtrader.backtest.costs import (
    CostBreakdown,
    CostError,
    CostModel,
    CostModelError,
    FinancingCut,
    SlippageParameter,
    cost_breakdown,
    declared_cost_model,
    declared_slippage_assumption,
)
from cfdtrader.backtest.engine import (
    BacktestRun,
    Bar,
    Decision,
    DecisionError,
    Direction,
    EmptyTestSetError,
    EngineError,
    EngineInputError,
    SessionInput,
    SessionOutcome,
    SessionView,
    canonical_text,
    run_walk_forward,
)
from cfdtrader.backtest.splits import (
    Fold,
    InsufficientSessionsError,
    SplitPlan,
    SplitsError,
    walk_forward_splits,
)

MODULE_PATH = Path(str(engine.__file__))
SOURCE = MODULE_PATH.read_text(encoding="utf-8")
REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DOCSTRING = (REPO_ROOT / "src" / "cfdtrader" / "backtest" / "__init__.py").read_text(
    encoding="utf-8"
)

#: Lo que A1 exige como mínimo en ``__all__``.
REQUIRED_PUBLIC = (
    "run_walk_forward",
    "SessionInput",
    "Bar",
    "SessionView",
    "Decision",
    "Direction",
    "SessionOutcome",
    "FoldOutcome",
    "BacktestRun",
    "EngineError",
    "EngineInputError",
    "EmptyTestSetError",
    "DecisionError",
    "RUN_HASH_FORMAT",
    "ENGINE_DOES_NOT_DO",
    "FOLLOW_UPS",
)

#: Las fronteras que A31 exige cubrir entre ``ENGINE_DOES_NOT_DO`` y ``FOLLOW_UPS``.
REQUIRED_FRONTIER_ISSUES = (
    "#14",
    "#15",
    "#16",
    "#17",
    "#18",
    "#19",
    "#24",
    "#25",
    "#27",
    "#28",
    "#50",
    "#60",
    "#62",
    "#63",
    "#66",
    "#67",
    "#68",
)

#: Módulos que el motor puede importar (A30): biblioteca estándar y los contratos de #11/#12.
ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "collections.abc",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "hashlib",
        "json",
        "math",
        "typing",
        "cfdtrader.backtest.costs",
        "cfdtrader.backtest.splits",
    }
)

#: Lo que el núcleo **no** puede importar (A30).
FORBIDDEN_IMPORTS = (
    "duckdb",
    "polars",
    "openai",
    "langgraph",
    "cfdtrader.agents",
    "cfdtrader.orchestration",
    "cfdtrader.data",
    "cfdtrader.analysis",
    "numpy",
    "pandas",
    "sklearn",
)

#: Lo que el núcleo **no** puede contener en su texto (A2, A29).
FORBIDDEN_TOKENS = (
    "datetime.now",
    "utcnow",
    "date.today",
    "time.time",
    "random",
    "uuid",
)

#: Librerías de backtest prohibidas por ``tech_stack.md`` §4.7 (A35).
FORBIDDEN_BACKTEST_LIBS = ("mlfinlab", "statsmodels", "backtrader", "zipline", "backtesting.py")

NOTIONAL = Decimal("10000")
DECLARED = declared_cost_model()
ASSUMED = declared_slippage_assumption()
START = date(2026, 1, 5)
REAL_COST_BREAKDOWN = cost_breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades del test
# ─────────────────────────────────────────────────────────────────────────────
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
    bars: Sequence[Bar] | None = None,
    context: object | None = None,
) -> SessionInput:
    """Un ``SessionInput`` sintético con los precios que pida el caso."""
    return SessionInput(
        session=session,
        open_px=open_px,
        high_px=high_px,
        low_px=low_px,
        close_px=close_px,
        context=context,
        bars=bars,
    )


def flat_inputs(days: Sequence[date], *, base: float = 100.0) -> list[SessionInput]:
    """Sesiones planas: ``open`` = ``base + i``, rango de un punto y cierre por encima."""
    return [
        session_input(
            day,
            open_px=base + index,
            high_px=base + index + 1.0,
            low_px=base + index - 1.0,
            close_px=base + index + 0.5,
        )
        for index, day in enumerate(days)
    ]


def make_plan(
    days: Sequence[date],
    *,
    n_splits: int = 2,
    test_size: int = 3,
    embargo: int = 1,
) -> SplitPlan:
    """El plan de #12 para esas sesiones, con el horizonte de #10 (``h = 0``)."""
    return walk_forward_splits(
        days,
        label_horizon=[0] * len(days),
        n_splits=n_splits,
        test_size=test_size,
        embargo_sessions=embargo,
    )


def long_decider(
    *,
    reason: str = "gate: largo",
    stop_offset: float = 1.0,
    target_offset: float = 1.0,
    notional: object = NOTIONAL,
) -> Callable[[SessionView], Decision]:
    """Un decididor largo con barreras simétricas alrededor del ``open``."""

    def decide(view: SessionView) -> Decision:
        if view.open_px is None:
            return Decision(
                direction=Direction.LONG,
                reason=reason,
                notional_usd=cast("Decimal | None", notional),
            )
        return Decision(
            direction=Direction.LONG,
            reason=reason,
            stop_px=view.open_px - stop_offset,
            target_px=view.open_px + target_offset,
            notional_usd=cast("Decimal | None", notional),
        )

    return decide


def short_decider(
    *, reason: str = "gate: corto", stop_offset: float = 1.0, target_offset: float = 1.0
) -> Callable[[SessionView], Decision]:
    """Un decididor corto: la geometría de A23 se invierte."""

    def decide(view: SessionView) -> Decision:
        if view.open_px is None:
            return Decision(direction=Direction.SHORT, reason=reason, notional_usd=NOTIONAL)
        return Decision(
            direction=Direction.SHORT,
            reason=reason,
            stop_px=view.open_px + stop_offset,
            target_px=view.open_px - target_offset,
            notional_usd=NOTIONAL,
        )

    return decide


def close_decider(*, reason: str = "gate: al cierre") -> Callable[[SessionView], Decision]:
    """Un decididor largo **sin barreras**: la salida es el cierre de la sesión (A21-i)."""

    def decide(_view: SessionView) -> Decision:
        return Decision(direction=Direction.LONG, reason=reason, notional_usd=NOTIONAL)

    return decide


def nothing_decider(*, reason: str = "gate: sin senal") -> Callable[[SessionView], Decision]:
    """Un decididor que no opera: ``NOTHING`` con su motivo (A12)."""

    def decide(_view: SessionView) -> Decision:
        return Decision(direction=Direction.NOTHING, reason=reason)

    return decide


def selective_decider(
    *,
    nothing_for: frozenset[date] = frozenset(),
    base: Callable[[SessionView], Decision] | None = None,
) -> Callable[[SessionView], Decision]:
    """Un decididor que devuelve ``NOTHING`` en unas sesiones concretas y ``base`` en el resto."""
    inner = base if base is not None else long_decider()

    def decide(view: SessionView) -> Decision:
        if view.session in nothing_for:
            return Decision(direction=Direction.NOTHING, reason="gate: sin senal")
        return inner(view)

    return decide


class Spy:
    """Espía de la función de decisión: cuenta llamadas y guarda las vistas (A4, A5)."""

    def __init__(self, tag: str, inner: Callable[[SessionView], Decision]) -> None:
        self.tag = tag
        self.inner = inner
        self.calls: list[tuple[str, date]] = []
        self.views: list[SessionView] = []

    def __call__(self, view: SessionView) -> Decision:
        self.calls.append((self.tag, view.session))
        self.views.append(view)
        return self.inner(view)


class CostSpy:
    """Espía del motor de costes de #11: guarda los argumentos de cada llamada (A14, A15)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> CostBreakdown:
        self.calls.append(dict(kwargs))
        return cast("Any", REAL_COST_BREAKDOWN)(**kwargs)


def run(
    inputs: Sequence[SessionInput],
    plan: SplitPlan,
    deciders: Sequence[Callable[[SessionView], Decision]],
    *,
    cost_model: CostModel = DECLARED,
    slippage: SlippageParameter = ASSUMED,
    financing_cut: FinancingCut | None = None,
) -> BacktestRun:
    """Ejecuta el motor con el modelo declarado (#8) y el supuesto de #64 por defecto."""
    return run_walk_forward(
        inputs,
        split_plan=plan,
        cost_model=cost_model,
        slippage=slippage,
        decide_by_fold=deciders,
        financing_cut=financing_cut,
    )


def outcomes(run_result: BacktestRun) -> list[SessionOutcome]:
    """Todas las ``SessionOutcome`` de la corrida, en el orden publicado."""
    return [outcome for fold in run_result.folds for outcome in fold.sessions]


def flatten(text: str) -> str:
    """El texto con los espacios normalizados: un docstring se parte en lineas."""
    return " ".join(text.split())


def block(source: dict[str, object], key: str) -> dict[str, object]:
    """Un bloque del informe, con el tipo declarado (el informe es ``dict[str, object]``)."""
    return cast("dict[str, object]", source[key])


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
    """Módulos importados por el motor (para demostrar que A30 se cumple)."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def _gaps() -> Any:
    """``_gaps`` por reflexión: es privado, pero A6 exige que ``gap_px[0]`` sea ``None``."""
    return engine._gaps  # pyright: ignore[reportPrivateUsage]


def _scenario() -> tuple[list[SessionInput], SplitPlan, list[Spy]]:
    """El escenario fijo y reproducible de A29 (idéntico en el proceso y en el subproceso."""
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    deciders = [Spy("fold-0", long_decider()), Spy("fold-1", long_decider())]
    return inputs, plan, deciders


# ─────────────────────────────────────────────────────────────────────────────
# A1, A2 — ficheros, API pública, firma y pureza
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_module_files_and_public_api() -> None:
    assert MODULE_PATH.is_file()
    assert MODULE_PATH.name == "engine.py"
    assert MODULE_PATH.parent.name == "backtest"
    assert (REPO_ROOT / "tests" / "test_engine.py").is_file()
    declared = tuple(engine.__all__)
    for name in REQUIRED_PUBLIC:
        assert name in declared, f"{name} falta en __all__"
        assert hasattr(engine, name), f"{name} no existe en el modulo"
    assert issubclass(EngineInputError, EngineError)
    assert issubclass(EmptyTestSetError, EngineError)
    assert issubclass(DecisionError, EngineError)


def test_a2_signature_is_keyword_only_and_has_no_invented_defaults() -> None:
    parameters = list(inspect.signature(run_walk_forward).parameters.values())
    assert parameters[0].name == "inputs"
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for parameter in parameters[1:]:
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, parameter.name
    defaults = {parameter.name: parameter.default for parameter in parameters}
    assert defaults["financing_cut"] is None  # unico valor por defecto, y declarado
    for name in ("split_plan", "cost_model", "slippage", "decide_by_fold"):
        assert defaults[name] is inspect.Parameter.empty, f"{name} no lleva valor por defecto"


def test_a2_module_is_pure_no_clock_no_randomness_no_global_state() -> None:
    for token in FORBIDDEN_TOKENS:
        assert token not in SOURCE, f"el modulo menciona {token}"
    offenders = [
        ast.unparse(node)
        for node in ast.parse(SOURCE).body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, (ast.Dict, ast.List, ast.Set))
        and "__all__" not in {target.id for target in node.targets if isinstance(target, ast.Name)}
    ]
    assert offenders == [], f"estado global mutable en el modulo: {offenders}"
    assert "environ" not in SOURCE
    assert "getenv" not in SOURCE


# ─────────────────────────────────────────────────────────────────────────────
# A3 — contratos congelados con slots
# ─────────────────────────────────────────────────────────────────────────────
#: Campos exigidos por A3, por contrato.
CONTRACT_FIELDS: dict[str, tuple[str, ...]] = {
    "SessionInput": ("session", "open_px", "high_px", "low_px", "close_px", "context", "bars"),
    "Bar": ("high_px", "low_px"),
    "SessionView": ("session", "open_px", "gap_px", "context"),
    "Decision": ("direction", "reason", "stop_px", "target_px", "notional_usd", "probability"),
}


def test_a3_contracts_are_frozen_slots_dataclasses() -> None:
    classes = {
        "SessionInput": SessionInput,
        "Bar": Bar,
        "SessionView": SessionView,
        "Decision": Decision,
    }
    for name, cls in classes.items():
        assert fields(cls), name
        assert tuple(field.name for field in fields(cls)) == CONTRACT_FIELDS[name], name
    session = START
    instances = {
        "SessionInput": session_input(session),
        "Bar": Bar(high_px=101.0, low_px=99.0),
        "SessionView": SessionView(session=session, open_px=100.0, gap_px=None, context=None),
        "Decision": Decision(direction=Direction.NOTHING, reason="prueba"),
    }
    for name, instance in instances.items():
        assert not hasattr(instance, "__dict__"), f"{name} no tiene slots"
        target = cast("Any", instance)
        attribute = CONTRACT_FIELDS[name][0]
        with pytest.raises(FrozenInstanceError):
            setattr(target, attribute, None)


def test_a3_direction_and_decision_fn_are_declared() -> None:
    assert [member.value for member in Direction] == ["long", "short", "nothing"]
    assert Direction.LONG.value == "long"
    assert Direction.SHORT.value == "short"
    assert Direction.NOTHING.value == "nothing"
    decider: engine.DecisionFn = long_decider()
    assert callable(decider)


# ─────────────────────────────────────────────────────────────────────────────
# A4 — una llamada por sesion de test, dentro de su fold
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_one_call_per_test_session_in_its_own_fold() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    spies = [Spy("fold-0", long_decider()), Spy("fold-1", long_decider())]
    result = run(inputs, plan, spies)
    expected = sum(len(fold.test) for fold in plan.folds)
    assert sum(len(spy.calls) for spy in spies) == expected
    for index, fold in enumerate(plan.folds):
        seen = {session for tag, session in spies[index].calls if tag == f"fold-{index}"}
        assert len(spies[index].calls) == len(fold.test)
        assert seen == {days[position] for position in fold.test}
        # ninguna llamada del fold j fuera de su test
        others = [tag for tag in (spy.tag for spy in spies) if tag != f"fold-{index}"]
        assert others
    assert len(outcomes(result)) == expected


# ─────────────────────────────────────────────────────────────────────────────
# A5 — la vista no expone el futuro de la sesion en curso
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_session_view_hides_the_future_of_the_session() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    hidden = ("high_px", "low_px", "close_px")
    spies = [Spy("fold-0", long_decider()), Spy("fold-1", long_decider())]
    run(inputs, plan, spies)
    views = [view for spy in spies for view in spy.views]
    assert views
    for view in views:
        for name in hidden:
            assert not hasattr(view, name), f"la vista expone {name}"
    assert tuple(field.name for field in fields(SessionView)) == CONTRACT_FIELDS["SessionView"]


def test_a5_mutating_close_high_low_changes_only_the_pnl() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    mutated = [
        session_input(
            item.session,
            open_px=cast("float", item.open_px),
            high_px=cast("float", item.high_px) + 1.0,
            low_px=cast("float", item.low_px) - 1.0,
            close_px=cast("float", item.close_px) + 10.0,
        )
        for item in inputs
    ]
    # barreras lejanas: la salida es el cierre, asi que mutar el cierre mueve el P&L
    deciders = [long_decider(stop_offset=5.0, target_offset=5.0)] * 2
    before = run(inputs, plan, deciders)
    after = run(mutated, plan, deciders)
    shape = [
        (outcome.status, outcome.entry_px, outcome.reason, outcome.exit_reason)
        for outcome in outcomes(before)
    ]
    assert shape == [
        (outcome.status, outcome.entry_px, outcome.reason, outcome.exit_reason)
        for outcome in outcomes(after)
    ]
    assert [outcome.gross_pct for outcome in outcomes(before)] != [
        outcome.gross_pct for outcome in outcomes(after)
    ]
    # ni el `entry_px` ni el `exit_px` de la sesion 0 dependen de `close_px`
    assert outcomes(before)[0].entry_px == outcomes(after)[0].entry_px


# ─────────────────────────────────────────────────────────────────────────────
# A6 — el gap se mide y no se opera
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_gap_is_measured_against_the_previous_session_of_the_sequence() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    result = run(inputs, plan, [long_decider(), long_decider()])
    gaps = _gaps()(tuple(inputs))
    assert gaps[0] is None  # declarado en la primera posicion, nunca 0
    for index in range(1, len(inputs)):
        previous = cast("float", inputs[index - 1].close_px)
        current = cast("float", inputs[index].open_px)
        assert gaps[index] == pytest.approx(current / previous - 1.0)
    by_index = {outcome.session_index: outcome for outcome in outcomes(result)}
    for index, outcome in by_index.items():
        assert outcome.gap_px == pytest.approx(gaps[index])


def test_a6_no_outcome_crosses_the_overnight_gap() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    result = run(inputs, plan, [long_decider(), long_decider()])
    traded = [outcome for outcome in outcomes(result) if outcome.status == "traded"]
    assert traded
    for outcome in traded:
        item = inputs[outcome.session_index]
        assert outcome.entry_px == item.open_px  # exactamente el open, no el cierre previo
        assert outcome.exit_px is not None
        expected = (
            outcome.exit_px / cast("float", item.open_px) - 1.0
            if outcome.decision is not None and outcome.decision.direction is Direction.LONG
            else None
        )
        assert outcome.gross_pct == pytest.approx(cast("float", expected))


# ─────────────────────────────────────────────────────────────────────────────
# A7, A8 — el plan se consume; purga y embargo se publican
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_plan_is_consumed_not_recomputed() -> None:
    parameters = inspect.signature(run_walk_forward).parameters
    section_parameters = {parameter.name for parameter in parameters.values()}
    for name in ("n_splits", "test_size", "embargo_sessions", "max_train_size", "label_horizon"):
        assert name not in section_parameters
    assert "walk_forward_splits" not in SOURCE
    assert "purge" not in _code_only().replace("purge_total", "")


def test_a7_mismatch_between_plan_and_inputs_raises_with_both_sizes() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)[:11]
    with pytest.raises(EngineInputError) as info:
        run(inputs, plan, [long_decider(), long_decider()])
    message = str(info.value)
    assert str(plan.n_sessions) in message
    assert str(len(inputs)) in message


def test_a8_purge_and_embargo_are_published_not_asserted() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    result = run(inputs, plan, [long_decider(), long_decider()])
    assert result.plan_sha256 == plan.plan_sha256
    assert result.purge_total == plan.purge_total
    assert result.embargo_total == plan.embargo_total
    assert result.embargo_in_train_total == plan.embargo_in_train_total
    assert result.exclusions_are_no_op is plan.exclusions_are_no_op
    assert result.uncovered == plan.uncovered
    report_plan = block(result.report, "plan")
    assert report_plan["purge_total"] == plan.purge_total
    assert report_plan["embargo_total"] == plan.embargo_total
    assert report_plan["exclusions_are_no_op"] is plan.exclusions_are_no_op
    assert list(cast("list[object]", report_plan["uncovered"])) == list(plan.uncovered)
    # con el horizonte de #10 las dos exclusiones son no-ops, y el docstring lo declara
    assert plan.exclusions_are_no_op is True
    assert "no-ops estructurales" in flatten(engine.__doc__ or "")


# ─────────────────────────────────────────────────────────────────────────────
# A9, A10, A11 — conservacion, test vacio y orden
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_uncovered_positions_are_not_evaluated_but_counted() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    spies = [Spy("fold-0", long_decider()), Spy("fold-1", long_decider())]
    result = run(inputs, plan, spies)
    evaluated = {session for spy in spies for _tag, session in spy.calls}
    uncovered_sessions = {days[position] for position in plan.uncovered}
    assert not (evaluated & uncovered_sessions)
    assert result.not_in_any_test == len(plan.uncovered)
    assert (
        result.n_sessions
        == result.traded + result.no_trade + result.skipped + result.not_in_any_test
    )
    counts = block(result.report, "counts")
    assert counts["conservation_holds"] is True
    assert counts["not_in_any_test"] == len(plan.uncovered)


def test_a10_empty_test_raises_and_never_returns_a_fold() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    empty = Fold(
        index=0,
        test_start=5,
        test_stop=5,
        train=tuple(range(5)),
        purged=(),
        embargoed=(),
    )
    broken = SplitPlan(
        n_sessions=plan.n_sessions,
        inputs=plan.inputs,
        folds=(empty,),
        uncovered=plan.uncovered,
        purge_total=plan.purge_total,
        embargo_total=plan.embargo_total,
        embargo_in_train_total=plan.embargo_in_train_total,
        exclusions_are_no_op=plan.exclusions_are_no_op,
        plan_sha256=plan.plan_sha256,
    )
    with pytest.raises(EmptyTestSetError):
        run(flat_inputs(days), broken, [long_decider()])


def test_a11_folds_and_sessions_keep_the_plan_order() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    result = run(inputs, plan, [long_decider(), long_decider()])
    assert [fold.index for fold in result.folds] == [fold.index for fold in plan.folds]
    for outcome_fold, plan_fold in zip(result.folds, plan.folds, strict=True):
        indexes = [outcome.session_index for outcome in outcome_fold.sessions]
        assert indexes == list(plan_fold.test)
        assert indexes == sorted(indexes)
        assert outcome_fold.test_start == plan_fold.test_start
        assert outcome_fold.test_stop == plan_fold.test_stop
    again = run(inputs, plan, [long_decider(), long_decider()])
    assert again.folds == result.folds


# ─────────────────────────────────────────────────────────────────────────────
# A12, A13 — una salida por sesion; no_trade y skipped
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_exactly_one_outcome_per_test_session_with_the_gate_reason() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    inputs[7] = session_input(days[7], open_px=None)
    nothing_for = frozenset({days[6]})
    result = run([*inputs], plan, [selective_decider(nothing_for=nothing_for)] * 2)
    expected = sum(len(fold.test) for fold in plan.folds)
    assert len(outcomes(result)) == expected
    statuses = {outcome.status for outcome in outcomes(result)}
    assert statuses <= {"traded", "no_trade", "skipped"}
    no_trade = [outcome for outcome in outcomes(result) if outcome.status == "no_trade"]
    assert [outcome.session for outcome in no_trade] == [days[6]]
    assert no_trade[0].reason == "gate: sin senal"
    assert no_trade[0].decision is not None
    assert result.traded + result.no_trade + result.skipped == expected
    assert result.no_trade == 1
    assert result.skipped == 1
    counts = block(result.report, "counts")
    for name in ("traded", "no_trade", "skipped"):
        assert counts[name] == getattr(result, name)


def test_a13_skipped_session_keeps_the_decision_and_the_reason() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    inputs[8] = session_input(days[8], open_px=None)
    spies = [Spy("fold-0", long_decider()), Spy("fold-1", long_decider())]
    result = run(inputs, plan, spies)
    skipped = [outcome for outcome in outcomes(result) if outcome.status == "skipped"]
    assert [outcome.session for outcome in skipped] == [days[8]]
    outcome = skipped[0]
    assert outcome.skip_reason == engine.SKIP_MISSING_PRICES
    assert outcome.reason == "gate: largo"
    assert outcome.decision is not None
    assert outcome.cost is None
    # el motor SI llama al decididor de esa sesion: la senal no se pierde
    assert any(session == days[8] for spy in spies for _tag, session in spy.calls)
    # un precio no positivo tampoco se simula
    zero = flat_inputs(days)
    zero[9] = session_input(days[9], open_px=0.0)
    result_zero = run(zero, plan, [long_decider(), long_decider()])
    zero_outcome = next(outcome for outcome in outcomes(result_zero) if outcome.session == days[9])
    assert zero_outcome.status == "skipped"
    assert zero_outcome.skip_reason == engine.SKIP_MISSING_PRICES


# ─────────────────────────────────────────────────────────────────────────────
# A14, A15, A16 — el coste: cero con NOTHING, una llamada por operacion, igual en las dos patas
# ─────────────────────────────────────────────────────────────────────────────
def test_a14_nothing_pays_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    spy = CostSpy()
    monkeypatch.setattr(engine, "cost_breakdown", spy)
    result = run(inputs, plan, [nothing_decider(), nothing_decider()])
    assert spy.calls == []
    assert result.traded == 0
    assert all(outcome.cost is None for outcome in outcomes(result))
    assert all(outcome.pnl_declared_pct is None for outcome in outcomes(result))


def test_a15_one_cost_call_per_operation_with_the_declared_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    spy = CostSpy()
    monkeypatch.setattr(engine, "cost_breakdown", spy)
    cut = FinancingCut.unverified()
    result = run(inputs, plan, [long_decider(), long_decider()], financing_cut=cut)
    traded = [outcome for outcome in outcomes(result) if outcome.status == "traded"]
    assert len(spy.calls) == len(traded) == result.traded
    for call, outcome in zip(spy.calls, traded, strict=True):
        assert call["model"] is DECLARED
        assert call["slippage"] is ASSUMED
        assert call["side"] is Side.LONG
        assert call["nights"] == 0
        assert call["overnight_reason"] is None
        assert call["financing_cut"] is cut
        assert call["notional_usd"] == outcome.notional_usd == NOTIONAL
        assert type(call["notional_usd"]) is Decimal
    short = run(inputs, plan, [short_decider(), short_decider()])
    short_traded = [outcome for outcome in outcomes(short) if outcome.status == "traded"]
    assert short_traded
    for outcome in short_traded:
        assert outcome.cost is not None
        assert outcome.cost.side is Side.SHORT


def test_a16_the_cost_is_identical_in_both_legs() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = [
        session_input(day, open_px=100.0, high_px=100.5, low_px=99.5, close_px=100.0)
        for day in days
    ]
    result = run(inputs, plan, [long_decider(stop_offset=5.0, target_offset=5.0)] * 2)
    traded = [outcome for outcome in outcomes(result) if outcome.status == "traded"]
    assert traded
    for outcome in traded:
        cost = outcome.cost
        assert cost is not None
        assert cost.spread_entry_pct == cost.spread_exit_pct == Decimal("0.0021")
        assert cost.spread_entry_usd == cost.spread_exit_usd == Decimal("0.21")
        assert cost.spread_pct == Decimal("0.0042")
        assert outcome.exit_reason == "session_close"
        assert outcome.gross_pct == pytest.approx(0.0)
        # #80: la unidad del motor es la **fraccion** del nocional. El coste declarado
        # (`c_declared_pct` = 0.0042 %) se resta convertido con `c_fraction_of_notional`
        # (0.000042), nunca 100x.
        assert outcome.pnl_declared_pct == pytest.approx(-0.000042, abs=1e-12)
        assert outcome.pnl_declared_pct == pytest.approx(-float(cost.c_fraction_of_notional))
        expected = cost_breakdown(
            model=DECLARED,
            slippage=ASSUMED,
            notional_usd=NOTIONAL,
            side=Side.LONG,
            nights=0,
            overnight_reason=None,
            financing_cut=None,
        )
        assert outcome.cost == expected  # `==` sobre Decimal, sin tolerancia


# ─────────────────────────────────────────────────────────────────────────────
# A17 — sin overnight, por construccion
# ─────────────────────────────────────────────────────────────────────────────
def test_a17_no_overnight_by_construction() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    result = run(inputs, plan, [long_decider(), long_decider()])
    traded = [outcome for outcome in outcomes(result) if outcome.status == "traded"]
    assert traded
    for outcome in traded:
        cost = outcome.cost
        assert cost is not None
        assert cost.nights == 0
        assert cost.carry_pct == 0
        assert cost.carry_usd == 0
        assert "regla 6" in cost.carry_reason
        assert outcome.entry_session == outcome.exit_session == outcome.session
    code = _code_only()
    assert "nights=0" in code
    assert "overnight_reason=None" in code
    assert "nights=1" not in code


# ─────────────────────────────────────────────────────────────────────────────
# A18 — los tres estados del slippage no se fusionan
# ─────────────────────────────────────────────────────────────────────────────
def _traded_slippage(slippage: SlippageParameter) -> tuple[BacktestRun, SessionOutcome]:
    days = business_sessions(12)
    plan = make_plan(days)
    result = run(flat_inputs(days), plan, [long_decider(), long_decider()], slippage=slippage)
    outcome = next(item for item in outcomes(result) if item.status == "traded")
    return result, outcome


def test_a18_the_three_slippage_states_are_not_merged() -> None:
    measured = SlippageParameter.measured(
        pct_of_notional=Decimal("0.002"),
        source="medicion sintetica del test de #13",
        reason="medicion declarada para probar el estado measured",
    )
    result, outcome = _traded_slippage(measured)
    cost = outcome.cost
    assert cost is not None
    assert cost.c_total_pct is not None
    assert outcome.pnl_net_pct is not None
    # #80: el total tambien se convierte a fraccion antes de restarlo (`gross - total/100`);
    # `c_total_pct` = 0.0062 % (0.0042 declarado + 0.002 medido) -> 0.000062 en fraccion.
    assert float(cost.c_total_pct) == pytest.approx(0.0062)
    assert outcome.pnl_net_pct == pytest.approx(
        cast("float", outcome.gross_pct) - float(cost.c_total_pct) / 100
    )
    assert outcome.pnl_net_pct == pytest.approx(cast("float", outcome.gross_pct) - 0.000062)
    assert outcome.pnl_net_reason is None
    assert block(result.report, "slippage")["state"] == MeasureState.MEASURED.value

    assumed_result, assumed_outcome = _traded_slippage(ASSUMED)
    assumed_cost = assumed_outcome.cost
    assert assumed_cost is not None
    assert assumed_cost.c_total_pct is None
    assert assumed_outcome.pnl_net_pct is None
    assert assumed_outcome.pnl_net_pct != 0
    assert assumed_outcome.pnl_net_reason
    assert "assumed" in assumed_outcome.pnl_net_reason
    assumed_block = block(assumed_result.report, "slippage")
    assert assumed_block["state"] == MeasureState.ASSUMED.value
    assert assumed_block["is_measurement"] is False
    illustrative = block(assumed_block, "illustrative_equivalence")
    assert illustrative["label"] == "illustrative"
    assert illustrative["decision"] is False
    assert illustrative["feeds_pnl_net_pct"] is False
    assert illustrative["equivalent_pct_of_notional"] == "0.2"
    assert illustrative["equivalent_bp_of_notional"] == "20"

    unmeasured = SlippageParameter.unmeasured(reason="no hay medicion ni supuesto declarado")
    unmeasured_result, unmeasured_outcome = _traded_slippage(unmeasured)
    assert unmeasured_outcome.pnl_net_pct is None
    assert unmeasured_outcome.pnl_net_reason
    assert block(unmeasured_result.report, "slippage")["state"] == MeasureState.UNMEASURED.value


# ─────────────────────────────────────────────────────────────────────────────
# A19, A20 — nocional explicito; entrada y salida dentro de la sesion
# ─────────────────────────────────────────────────────────────────────────────
def test_a19_notional_is_explicit_and_without_default() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    cases = {
        "ausente": None,
        "cero": Decimal("0"),
        "negativo": Decimal("-100"),
        "float de relleno": 10000.0,
        "texto": "10000",
    }
    for value in cases.values():
        with pytest.raises(DecisionError):
            run(inputs, plan, [long_decider(notional=value), long_decider(notional=value)])
    assert NOTIONAL > 0


def test_a20_entry_is_the_auction_open_and_the_exit_stays_in_the_session() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    result = run(inputs, plan, [long_decider(), long_decider()])
    traded = [outcome for outcome in outcomes(result) if outcome.status == "traded"]
    assert traded
    for outcome in traded:
        assert outcome.entry_px == inputs[outcome.session_index].open_px
        assert outcome.exit_session == outcome.entry_session == outcome.session
        assert outcome.session in days
    code = _code_only()
    assert "entry_px=open_px" in code


# ─────────────────────────────────────────────────────────────────────────────
# A21, A22 — resolucion de la salida declarada
# ─────────────────────────────────────────────────────────────────────────────
def _single_session_case(
    *, stop_offset: float, target_offset: float, bars: Sequence[Bar] | None, bar_reason: str
) -> SessionOutcome:
    """Una corrida de un unico fold con una sola sesion de test, para leer su salida."""
    days = business_sessions(3)
    plan = make_plan(days, n_splits=1, test_size=1, embargo=0)
    inputs = flat_inputs(days)
    inputs[-1] = session_input(
        days[-1], open_px=100.0, high_px=101.0, low_px=99.0, close_px=100.4, bars=bars
    )
    result = run(inputs, plan, [long_decider(stop_offset=stop_offset, target_offset=target_offset)])
    outcome = result.folds[0].sessions[0]
    assert outcome.reason == bar_reason
    return outcome


def test_a21_exit_resolution_is_declared() -> None:
    target = _single_session_case(
        stop_offset=5.0, target_offset=1.0, bars=None, bar_reason="gate: largo"
    )
    assert target.exit_reason == "target"
    assert target.exit_px == pytest.approx(101.0)
    stop = _single_session_case(
        stop_offset=1.0, target_offset=5.0, bars=None, bar_reason="gate: largo"
    )
    assert stop.exit_reason == "stop"
    assert stop.exit_px == pytest.approx(99.0)
    untouched = _single_session_case(
        stop_offset=5.0, target_offset=5.0, bars=None, bar_reason="gate: largo"
    )
    assert untouched.exit_reason == "session_close"
    assert untouched.exit_px == pytest.approx(100.4)
    # sin barreras, la salida es el cierre (A21-i)
    days = business_sessions(3)
    plan = make_plan(days, n_splits=1, test_size=1, embargo=0)
    flat = flat_inputs(days)
    result = run(flat, plan, [close_decider()])
    without = result.folds[0].sessions[0]
    assert without.exit_reason == "session_close"
    assert without.exit_px == flat[-1].close_px
    assert without.exit_bar_index is None
    assert target.exit_bar_index == 0  # camino del respaldo diario: una sola barra


def test_a21_a_tie_in_the_same_bar_goes_to_the_adverse_barrier() -> None:
    days = business_sessions(3)
    plan = make_plan(days, n_splits=1, test_size=1, embargo=0)
    inputs = flat_inputs(days)
    inputs[-1] = session_input(days[-1], open_px=100.0, high_px=101.0, low_px=99.0, close_px=100.4)
    result = run(inputs, plan, [long_decider(stop_offset=1.0, target_offset=1.0)])
    outcome = result.folds[0].sessions[0]
    assert outcome.exit_reason == "stop"
    assert outcome.exit_px == pytest.approx(99.0)


def test_a21_the_intraday_path_follows_the_bars_in_order() -> None:
    bars = (
        Bar(high_px=100.5, low_px=99.5),
        Bar(high_px=101.5, low_px=100.5),
        Bar(high_px=102.0, low_px=101.6),
    )
    outcome = _single_session_case(
        stop_offset=5.0, target_offset=1.0, bars=bars, bar_reason="gate: largo"
    )
    assert outcome.exit_reason == "target"
    assert outcome.exit_bar_index == 1


def test_a22_exit_price_is_the_barrier_price_and_the_index_is_published() -> None:
    bars = (Bar(high_px=101.5, low_px=99.5),)
    days = business_sessions(3)
    plan = make_plan(days, n_splits=1, test_size=1, embargo=0)
    inputs = flat_inputs(days)
    inputs[-1] = session_input(
        days[-1], open_px=100.0, high_px=101.5, low_px=99.5, close_px=100.4, bars=bars
    )

    def decider(_view: SessionView) -> Decision:
        return Decision(
            direction=Direction.LONG,
            reason="gate: objetivo pegado",
            stop_px=98.5,
            target_px=100.5,
            notional_usd=NOTIONAL,
        )

    result = run(inputs, plan, [decider])
    outcome = result.folds[0].sessions[0]
    assert outcome.exit_reason == "target"
    assert outcome.exit_px == pytest.approx(100.5)  # el precio de la barrera, no 101,5
    assert outcome.exit_bar_index == 0
    assert outcome.exit_reason in {"target", "stop", "session_close"}


# ─────────────────────────────────────────────────────────────────────────────
# A23 — geometria de la decision
# ─────────────────────────────────────────────────────────────────────────────
def test_a23_invalid_geometry_raises() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)

    def bad_long(_view: SessionView) -> Decision:
        return Decision(
            direction=Direction.LONG,
            reason="gate: geometria rota",
            stop_px=101.0,
            target_px=105.0,
            notional_usd=NOTIONAL,
        )

    def bad_short(_view: SessionView) -> Decision:
        return Decision(
            direction=Direction.SHORT,
            reason="gate: geometria rota",
            stop_px=105.0,
            target_px=101.0,
            notional_usd=NOTIONAL,
        )

    with pytest.raises(DecisionError):
        run(inputs, plan, [bad_long, bad_long])
    with pytest.raises(DecisionError):
        run(inputs, plan, [bad_short, bad_short])


def test_a23_a_single_barrier_is_rejected() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)

    def only_stop(_view: SessionView) -> Decision:
        return Decision(
            direction=Direction.LONG,
            reason="gate: solo stop",
            stop_px=98.0,
            notional_usd=NOTIONAL,
        )

    with pytest.raises(DecisionError):
        run(inputs, plan, [only_stop, only_stop])


# ─────────────────────────────────────────────────────────────────────────────
# A24, A25 — medias sesiones; t0/t1
# ─────────────────────────────────────────────────────────────────────────────
def test_a24_a_half_session_is_one_more_session() -> None:
    days = business_sessions(3)
    plan = make_plan(days, n_splits=1, test_size=1, embargo=0)
    inputs = flat_inputs(days)
    inputs[-1] = session_input(days[-1], open_px=100.0, high_px=100.3, low_px=99.8, close_px=100.2)
    result = run(inputs, plan, [close_decider()])
    outcome = result.folds[0].sessions[0]
    assert outcome.status == "traded"
    assert outcome.exit_reason == "session_close"
    assert outcome.exit_px == pytest.approx(100.2)
    assert outcome.entry_session == outcome.exit_session == days[-1]
    for hour in ("09:30", "13:00", "16:00"):
        assert hour not in SOURCE, f"el modulo asume una hora fija: {hour}"


def test_a25_t0_and_t1_are_declared_in_the_docstring_and_the_report() -> None:
    doc = flatten(engine.__doc__ or "")
    for token in ("``t0``", "``t1``", "label_horizon = 0", "close_utc"):
        assert token in doc, f"el docstring no declara {token}"
    days = business_sessions(12)
    plan = make_plan(days)
    result = run(flat_inputs(days), plan, [long_decider(), long_decider()])
    declared = block(result.report, "t0_t1")
    assert declared["label_horizon"] == 0
    assert declared["overnight"] is False
    assert "subasta" in cast("str", declared["t0"])
    assert "misma sesion" in cast("str", declared["t1"])


# ─────────────────────────────────────────────────────────────────────────────
# A26, A27 — errores tipados y propagacion sin envolver
# ─────────────────────────────────────────────────────────────────────────────
def test_a26_invalid_inputs_raise_a_typed_error() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    deciders = [long_decider(), long_decider()]
    with pytest.raises(EngineInputError):
        run([], plan, deciders)  # inputs vacio
    with pytest.raises(EngineInputError):
        run(inputs, plan, [long_decider()])  # una funcion por fold, exactamente
    with pytest.raises(EngineError):
        run(inputs, plan, [long_decider(), long_decider(), long_decider()])
    duplicated = list(inputs)
    duplicated[3] = flat_inputs(days)[3]
    duplicated[4] = session_input(inputs[3].session)
    with pytest.raises(EngineInputError):
        run(duplicated, plan, deciders)  # sesiones duplicadas
    unordered = list(inputs)
    unordered[2], unordered[3] = unordered[3], unordered[2]
    with pytest.raises(EngineInputError):
        run(unordered, plan, deciders)  # sesiones no crecientes
    not_a_session = [session_input(cast("date", "2026-01-05")), *inputs[1:]]
    with pytest.raises(EngineInputError):
        run(not_a_session, plan, deciders)
    not_an_input = [cast("SessionInput", "no es una sesion"), *inputs[1:]]
    with pytest.raises(EngineInputError):
        run(not_an_input, plan, deciders)
    with pytest.raises(EngineInputError):
        run(cast("Any", inputs), cast("Any", "no es un plan"), deciders)
    with pytest.raises(EngineInputError):
        run(inputs, plan, deciders, cost_model=cast("CostModel", object()))
    with pytest.raises(EngineInputError):
        run(inputs, plan, deciders, slippage=cast("SlippageParameter", object()))

    def not_a_decision(_view: SessionView) -> Decision:
        return cast("Decision", "no es una Decision")

    with pytest.raises(EngineError):
        run(inputs, plan, [not_a_decision, not_a_decision])


def test_a26_unusable_bars_raise_instead_of_silently_skipping() -> None:
    days = business_sessions(3)
    plan = make_plan(days, n_splits=1, test_size=1, embargo=0)
    inputs = flat_inputs(days)
    inverted = [*inputs[:-1], session_input(days[-1], bars=(Bar(high_px=99.0, low_px=101.0),))]
    with pytest.raises(EngineInputError):
        run(inverted, plan, [long_decider()])
    without_high = [
        *inputs[:-1],
        session_input(days[-1], bars=(Bar(high_px=cast("float", None), low_px=100.0),)),
    ]
    with pytest.raises(EngineInputError):
        run(without_high, plan, [long_decider()])
    empty = [*inputs[:-1], session_input(days[-1], bars=())]
    with pytest.raises(EngineInputError):
        run(empty, plan, [long_decider()])
    not_a_bar = [*inputs[:-1], session_input(days[-1], bars=(cast("Bar", "no es una barra"),))]
    with pytest.raises(EngineInputError):
        run(not_a_bar, plan, [long_decider()])


def test_a27_cost_and_splits_errors_propagate_unwrapped() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    assert issubclass(CostModelError, CostError)
    assert issubclass(InsufficientSessionsError, SplitsError)
    unvalidated = CostModel.model_construct()
    with pytest.raises(CostModelError):
        run(inputs, plan, [long_decider(), long_decider()], cost_model=unvalidated)
    with pytest.raises(SplitsError):
        walk_forward_splits(
            days,
            label_horizon=[0] * len(days),
            n_splits=6,
            test_size=3,
            embargo_sessions=1,
        )
    code = _code_only()
    assert "except" not in code, "el motor captura errores de #11/#12"
    assert "try:" not in code


# ─────────────────────────────────────────────────────────────────────────────
# A28, A29 — serializacion canonica, hash y determinismo
# ─────────────────────────────────────────────────────────────────────────────
def test_a28_run_sha256_is_the_documented_canonical_text() -> None:
    text = canonical_text({"b": 1, "a": Decimal("0.0042"), "c": 1.5, "d": None})
    assert text == '{"a":"0.0042","b":1,"c":1.5,"d":null}'
    assert " " not in text
    for token in ("sha256", "UTF-8", "sort_keys", "Decimal", "float", "run_sha256"):
        assert token in engine.RUN_HASH_FORMAT, f"RUN_HASH_FORMAT no documenta {token}"
    inputs, plan, deciders = _scenario()
    result = run(inputs, plan, deciders)
    expected = hashlib.sha256(canonical_text(result.report).encode("utf-8")).hexdigest()
    assert result.run_sha256 == expected
    assert len(result.run_sha256) == 64
    assert result.published_report()["run_sha256"] == result.run_sha256
    assert "run_sha256" not in result.report


def test_a28_every_input_change_changes_the_hash() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    inputs = flat_inputs(days)
    deciders = [long_decider(), long_decider()]
    base = run(inputs, plan, deciders)
    other_plan = make_plan(days, n_splits=2, test_size=2, embargo=2)
    other_model = DECLARED.model_copy(update={"name": "otro modelo declarado"})
    other_slippage = SlippageParameter.unmeasured(reason="sin medicion ni supuesto")
    mutated = flat_inputs(days)
    mutated[-1] = session_input(days[-1], close_px=999.0, open_px=999.0)
    variants = {
        "precio": run(mutated, plan, deciders),
        "plan": run(inputs, other_plan, deciders),
        "modelo de coste": run(inputs, plan, deciders, cost_model=other_model),
        "slippage": run(inputs, plan, deciders, slippage=other_slippage),
        "decision": run(inputs, plan, [long_decider(reason="otro motivo"), long_decider()]),
    }
    for name, variant in variants.items():
        assert variant.run_sha256 != base.run_sha256, name
    assert run(inputs, plan, deciders).run_sha256 == base.run_sha256


def test_a29_identical_runs_are_structurally_equal() -> None:
    inputs, plan, _deciders = _scenario()
    first = run(inputs, plan, [long_decider(), long_decider()])
    second = run(inputs, plan, [long_decider(), long_decider()])
    assert first.run_sha256 == second.run_sha256
    assert first.folds == second.folds
    assert first.report == second.report
    assert first == second


_SECOND_PROCESS_SCRIPT = textwrap.dedent(
    """
    from datetime import date, timedelta
    from decimal import Decimal

    from cfdtrader.backtest.costs import declared_cost_model, declared_slippage_assumption
    from cfdtrader.backtest.engine import (
        Decision, Direction, SessionInput, run_walk_forward,
    )
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

    def decide(view):
        return Decision(
            direction=Direction.LONG,
            reason="gate: largo",
            stop_px=view.open_px - 1.0,
            target_px=view.open_px + 1.0,
            notional_usd=Decimal("10000"),
        )

    result = run_walk_forward(
        inputs,
        split_plan=plan,
        cost_model=declared_cost_model(),
        slippage=declared_slippage_assumption(),
        decide_by_fold=[decide, decide],
    )
    print(result.run_sha256)
    """
)


def test_a29_a_second_process_prints_the_same_hash() -> None:
    inputs, plan, _deciders = _scenario()
    in_process = run(inputs, plan, [long_decider(), long_decider()])
    completed = subprocess.run(  # noqa: S603 - comando fijo (el interprete de la sesion)
        [sys.executable, "-c", _SECOND_PROCESS_SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == in_process.run_sha256


# ─────────────────────────────────────────────────────────────────────────────
# A30, A31, A32, A33 — pureza de disco, sin LLM, fronteras y limitaciones
# ─────────────────────────────────────────────────────────────────────────────
def test_a30_module_imports_only_the_standard_library_and_the_contracts() -> None:
    imported = _imported_modules()
    assert imported, "el modulo no importa nada: el test no esta mirando el fichero correcto"
    assert imported <= ALLOWED_IMPORTS, f"imports no declarados: {imported - ALLOWED_IMPORTS}"
    for name in FORBIDDEN_IMPORTS:
        assert name not in imported
    code = _code_only()
    for token in ("Store(", "Store.", "duckdb", "polars", "config/", "read_pit"):
        assert token not in code, f"el motor toca {token}"


def test_a30_the_report_declares_the_llm_overlay_disabled() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    result = run(flat_inputs(days), plan, [long_decider(), long_decider()])
    assert result.report["llm_overlay"] == "disabled"
    for name in ("llm", "llm_overlay", "overlay"):
        assert name not in inspect.signature(run_walk_forward).parameters
    assert "orchestration" not in _imported_modules()
    assert "agents" not in _imported_modules()


def test_a31_does_not_do_and_follow_ups_are_machine_readable() -> None:
    tables = {
        "ENGINE_DOES_NOT_DO": engine.ENGINE_DOES_NOT_DO,
        "FOLLOW_UPS": engine.FOLLOW_UPS,
    }
    for name, table in tables.items():
        assert isinstance(table, tuple) and table, name
        for entry in table:
            assert isinstance(entry, dict)
            assert entry.get("issue", "").startswith("#"), f"{name}: entrada sin issue"
            assert len(entry) >= 3, f"{name}: entrada incompleta ({entry})"
        covered = {entry["issue"] for entry in table}
        missing = [issue for issue in REQUIRED_FRONTIER_ISSUES if issue not in covered]
        assert missing == [], f"{name} no cubre {missing}"


def test_a32_the_report_publishes_its_limitations() -> None:
    days = business_sessions(12)
    plan = make_plan(days)
    result = run(flat_inputs(days), plan, [long_decider(), long_decider()])
    assert result.report["gate"] == "fail"
    assert result.report["phase1_ready"] is False
    limitations = cast("list[object]", result.report["limitations"])
    assert len(limitations) >= 5
    text = " ".join(str(item) for item in limitations)
    for token in (
        "fail",
        "phase1_ready",
        "no son una validacion de la estrategia",
        "declarados y no medidos",
        "supuesto pesimista declarado",
        "pnl_net_pct",
        "corte de financiacion sigue sin verificar",
        "proxies",
        "respaldo diario",
    ):
        assert token in text, f"las limitaciones no mencionan {token}"


def test_a33_docstrings_declare_the_vocabulary_and_the_frontiers() -> None:
    doc = flatten(engine.__doc__ or "")
    for token in (
        "#11 cobra; #13 recorre y simula",
        "SessionView",
        "gap_px",
        "high_px",
        "no-ops estructurales",
        "#69",
        "llm_overlay",
        "plan.md",
        "tech_stack.md",
    ):
        assert token in doc, f"el docstring del modulo no traza {token}"


def test_a33_package_docstring_mentions_the_new_module() -> None:
    assert "cfdtrader.backtest.engine" in PACKAGE_DOCSTRING
    assert "#13" in PACKAGE_DOCSTRING
    assert "cfdtrader.backtest.costs" in PACKAGE_DOCSTRING
    assert "cfdtrader.backtest.splits" in PACKAGE_DOCSTRING


# ─────────────────────────────────────────────────────────────────────────────
# A34, A35 — cobertura de la suite y puertas
# ─────────────────────────────────────────────────────────────────────────────
def test_a34_the_suite_covers_every_criterion() -> None:
    module = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names = [
        node.name
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_a")
    ]
    covered = {int(name.split("_")[1][1:]) for name in names}
    missing = sorted(set(range(1, 36)) - covered)
    assert missing == [], f"criterios sin test: {missing}"
    assert len(names) >= 35


def test_a35_core_is_about_two_hundred_lines() -> None:
    core = ("_resolve_exit", "_gross_pct", "_intraday_path", "_evaluate_session", "_run_folds")
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
    assert total <= 200, f"el nucleo tiene {total} lineas efectivas"


def test_a35_no_forbidden_backtest_library_is_used() -> None:
    for name in FORBIDDEN_BACKTEST_LIBS:
        assert name not in SOURCE
        assert name not in _imported_modules()


def test_a35_a_run_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    days = business_sessions(12)
    plan = make_plan(days)
    result = run(flat_inputs(days), plan, [long_decider(), long_decider()])
    assert result.run_sha256
    assert list(tmp_path.iterdir()) == []
    assert costs.__name__ == "cfdtrader.backtest.costs"
    assert splits.__name__ == "cfdtrader.backtest.splits"
