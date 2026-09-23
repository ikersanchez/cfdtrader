"""Tests del gate de decision y *sizing* (#27): A1-A14.

El modulo es **puro**: todo se prueba con entradas declaradas a mano (un `CostBreakdown` real
del bloque de #11, un `MarketCalendar` real y unos `GateParameters` completos). Los tres
casos calculados a mano —nocional desde el stop (A5), EV del 0,5958 % (A6) y las reglas 8 y 9—
fijan la aritmetica, y el resto de criterios fijan los estados, los bloqueos y la pureza.

Los numeros que #60 no ha decidido (1 %, 2 %, 3c, 0,58) **no** estan en el modulo: los tests
los declaran en `GateParameters` como cualquier llamante.
"""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
import textwrap
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final, cast

import pytest
import yaml
from pydantic import ValidationError

from cfdtrader.backtest.costs import (
    CostBreakdown,
    Side,
    SlippageParameter,
    cost_breakdown,
    declared_cost_model,
    declared_slippage_assumption,
)
from cfdtrader.backtest.engine import Decision, Direction
from cfdtrader.data.calendar import MarketCalendar
from cfdtrader.decision import gate
from cfdtrader.decision.gate import (
    GateInputError,
    GateOutput,
    GateParameters,
    GateStatus,
    evaluate_gate,
    to_engine_decision,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODULE: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "decision" / "gate.py"
SETTINGS: Final[Path] = REPO_ROOT / "config" / "settings.yaml"

#: Sesion completa (miercoles), su "hoy" y el instante del snapshot: todo declarado.
SESSION: Final[date] = date(2026, 9, 23)
TODAY: Final[date] = SESSION
AS_OF: Final[datetime] = datetime(2026, 9, 23, 12, 45, tzinfo=UTC)

#: Media sesion real del calendario: el viernes despues de Accion de Gracias de 2026.
HALF_SESSION: Final[date] = date(2026, 11, 27)

#: Espacio de nombres de la firma declarada (A1): ni un argumento de mas ni de menos.
DECLARED_SIGNATURE: Final[tuple[str, ...]] = (
    "session",
    "as_of",
    "today",
    "calendar",
    "prob_up_calibrated",
    "expected_move_pct",
    "expected_move_basis",
    "cost",
    "capital_usd",
    "snapshot_ok",
    "stop_pct",
    "target_pct",
    "fomc_dates",
    "params",
    "trades_today",
    "daily_pnl_pct",
    "weekly_pnl_pct",
    "monthly_pnl_pct",
    "observation_sessions_remaining",
)

CALENDAR: Final[MarketCalendar] = MarketCalendar(years=tuple(range(2020, 2030)))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures declaradas a mano
# ─────────────────────────────────────────────────────────────────────────────
def measured(pct: str) -> SlippageParameter:
    """Un `slippage` **medido** (el unico estado que cierra el total de #11)."""
    return SlippageParameter.measured(
        pct_of_notional=Decimal(pct), source="test_gate (#27)", reason=f"medicion declarada {pct} %"
    )


def unmeasured() -> SlippageParameter:
    """Sin medicion ni supuesto: `pct_of_notional` es `None` y el total no se cierra."""
    return SlippageParameter.unmeasured(reason="test_gate (#27): no hay medicion de slippage")


def cost(
    slippage: SlippageParameter,
    *,
    nights: int = 0,
    side: Side = Side.LONG,
    notional: str = "10000",
) -> CostBreakdown:
    """El coste de una operacion, calculado por el bloque real de #11 (no aqui)."""
    return cost_breakdown(
        model=declared_cost_model(),
        slippage=slippage,
        notional_usd=Decimal(notional),
        side=side,
        nights=nights,
        overnight_reason=None if nights == 0 else "test_gate: noche declarada a proposito",
    )


def params(**overrides: object) -> GateParameters:
    """Los parametros **decididos** de una llamada normal; cada test cambia lo que necesita."""
    base: dict[str, object] = {
        "broker": "cfd-broker-declarado",
        "risk_per_trade_pct": Decimal("1"),
        "ev_threshold_pct": Decimal("0.0084"),
        "max_daily_loss_pct": Decimal("2"),
        "max_weekly_loss_pct": Decimal("5"),
        "max_monthly_loss_pct": Decimal("10"),
        "r_pct": Decimal("1"),
        "tier_a_cost_multiple": Decimal("3"),
        "tier_b_cost_multiple": Decimal("2"),
        "tier_a_min_probability": Decimal("0.58"),
        "authorized_tiers": ("A",),
    }
    base.update(overrides)
    return GateParameters(**base)  # type: ignore[arg-type]


def call(**overrides: object) -> GateOutput:
    """Una llamada completa al gate con todo declarado; cada test cambia lo que necesita."""
    kwargs: dict[str, object] = {
        "session": SESSION,
        "as_of": AS_OF,
        "today": TODAY,
        "calendar": CALENDAR,
        "prob_up_calibrated": 0.6,
        "expected_move_pct": Decimal("1.0"),
        "expected_move_basis": "sigma_k: k declarado por #60 (test)",
        "cost": cost(measured("0")),
        "capital_usd": Decimal("10000"),
        "snapshot_ok": True,
        "stop_pct": Decimal("0.5"),
        "target_pct": Decimal("1.0"),
        "fomc_dates": (),
        "params": params(),
    }
    kwargs.update(overrides)
    return evaluate_gate(**kwargs)  # type: ignore[arg-type]


def codes(output: GateOutput) -> dict[str, str]:
    """Codigo de bloqueo -> numero de regla, para leer los bloqueos sin depender del orden."""
    return {entry["code"]: entry["rule"] for entry in output.blockers}


def rule_outcomes(output: GateOutput) -> dict[str, str]:
    """Regla -> resultado, tal cual se publica en `rules[]`."""
    return {entry["rule"]: entry["outcome"] for entry in output.rules}


def module_tree() -> ast.Module:
    """El AST del modulo: las comprobaciones estructurales se hacen por AST, no por texto."""
    return ast.parse(MODULE.read_text(encoding="utf-8"))


def called_names(tree: ast.Module) -> set[str]:
    """Nombres invocados en el modulo (`open(...)`, `datetime.now(...)`, ...)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                names.add(function.id)
            elif isinstance(function, ast.Attribute):
                names.add(function.attr)
    return names


def imported_modules(tree: ast.Module) -> set[str]:
    """Modulos importados, por nombre completo y por raiz."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules | {name.split(".")[0] for name in modules}


# ─────────────────────────────────────────────────────────────────────────────
# A1 — el modulo existe, exporta el contrato y la firma declarada
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_modulo_y_firma_declarada() -> None:
    assert MODULE.is_file()
    assert {
        "evaluate_gate",
        "GateOutput",
        "GateStatus",
        "GateParameters",
        "to_engine_decision",
    } <= (set(gate.__all__))
    for name in gate.__all__:
        assert hasattr(gate, name), name

    signature = inspect.signature(evaluate_gate)
    assert tuple(signature.parameters) == DECLARED_SIGNATURE
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    defaults = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    assert set(defaults) == {
        "trades_today",
        "daily_pnl_pct",
        "weekly_pnl_pct",
        "monthly_pnl_pct",
        "observation_sessions_remaining",
    }
    assert defaults["trades_today"] == 0
    assert defaults["observation_sessions_remaining"] == 0
    assert defaults["daily_pnl_pct"] is None
    assert len(gate.RULES) == 18
    assert [entry["rule"] for entry in gate.RULES] == [str(number) for number in range(1, 19)]


# ─────────────────────────────────────────────────────────────────────────────
# A2 — determinismo entre procesos con PYTHONHASHSEED distinto
# ─────────────────────────────────────────────────────────────────────────────
_PROBE: Final[str] = textwrap.dedent(
    """
    from datetime import date, datetime, timezone
    from decimal import Decimal

    from cfdtrader.backtest.costs import (
        Side,
        SlippageParameter,
        cost_breakdown,
        declared_cost_model,
    )
    from cfdtrader.data.calendar import MarketCalendar
    from cfdtrader.decision.gate import GateParameters, evaluate_gate

    cost = cost_breakdown(
        model=declared_cost_model(),
        slippage=SlippageParameter.measured(
            pct_of_notional=Decimal("0.2"), source="sonda A2", reason="sonda A2"
        ),
        notional_usd=Decimal("10000"),
        side=Side.LONG,
        nights=0,
    )
    output = evaluate_gate(
        session=date(2026, 9, 23),
        as_of=datetime(2026, 9, 23, 12, 45, tzinfo=timezone.utc),
        today=date(2026, 9, 23),
        calendar=MarketCalendar(years=tuple(range(2020, 2030))),
        prob_up_calibrated=0.6,
        expected_move_pct=Decimal("1.0"),
        expected_move_basis="sigma_k: k declarado por #60 (test)",
        cost=cost,
        capital_usd=Decimal("10000"),
        snapshot_ok=True,
        stop_pct=Decimal("0.5"),
        target_pct=Decimal("1.0"),
        fomc_dates=(),
        params=GateParameters(
            broker="cfd-broker-declarado",
            risk_per_trade_pct=Decimal("1"),
            ev_threshold_pct=Decimal("0.0084"),
            max_daily_loss_pct=Decimal("2"),
            max_weekly_loss_pct=Decimal("5"),
            max_monthly_loss_pct=Decimal("10"),
            r_pct=Decimal("1"),
            tier_a_cost_multiple=Decimal("3"),
            tier_b_cost_multiple=Decimal("2"),
            tier_a_min_probability=Decimal("0.58"),
            authorized_tiers=("A",),
        ),
    )
    print(output.gate_sha256)
    print(output.status.value, output.direction.value, output.tier, output.notional_usd)
    """
)


def _probe(hash_seed: str) -> str:
    completed = subprocess.run(  # noqa: S603 - el interprete de la sesion
        [sys.executable, "-c", _PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_a2_determinismo_entre_procesos() -> None:
    first = call(cost=cost(measured("0.2")))
    second = call(cost=cost(measured("0.2")))
    assert first == second
    assert first.gate_sha256 == second.gate_sha256
    assert first.gate_sha256.startswith("sha256:")
    assert len(first.gate_sha256) == len("sha256:") + 64

    outputs = [_probe("0"), _probe("1")]
    assert outputs[0] == outputs[1]
    lines = outputs[0].splitlines()
    assert lines[0] == first.gate_sha256
    assert lines[1] == "recommendation long A 20000.00"


def test_a2_hash_del_payload_sin_la_clave() -> None:
    output = call()
    moved = output.model_copy(update={"gate_sha256": "sha256:no-cuenta"})
    assert gate.gate_sha256(moved) == output.gate_sha256
    assert gate.gate_sha256(output) == output.gate_sha256
    assert gate.GATE_HASH_PREFIX == "sha256:"
    assert "sha256" in gate.GATE_HASH_FORMAT


# ─────────────────────────────────────────────────────────────────────────────
# A3 — pureza: sin reloj, sin red, sin disco, sin leer la configuracion
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_sin_reloj_ni_io() -> None:
    tree = module_tree()
    assert (
        imported_modules(tree)
        & {
            "time",
            "socket",
            "httpx",
            "requests",
            "os",
            "pathlib",
            "shutil",
            "subprocess",
            "urllib",
        }
        == set()
    )
    assert (
        called_names(tree)
        & {
            "now",
            "today",
            "utcnow",
            "open",
            "read_text",
            "read_bytes",
            "load_calendar",
            "safe_load",
        }
        == set()
    )
    assert not any(isinstance(node, (ast.Global, ast.Nonlocal)) for node in ast.walk(tree))
    literals = {
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and node.value
    }
    assert not [text for text in literals if isinstance(text, str) and "config/" in text]
    # El modulo no construye el calendario ni lo carga: lo recibe (su import es solo el tipo).
    assert "MarketCalendar" in {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "load_calendar" not in called_names(tree)


def test_a3_el_gate_no_toca_el_disco(tmp_path: Path) -> None:
    """Una llamada no escribe ni lee nada: la unica raiz que existe es la que el test declara."""
    marker = tmp_path / "no-debe-existir"
    before = sorted(tmp_path.iterdir())
    call()
    assert sorted(tmp_path.iterdir()) == before
    assert not marker.exists()


# ─────────────────────────────────────────────────────────────────────────────
# A4 — la `Direction` es la del motor y `to_engine_decision` devuelve un `Decision` valido
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_direction_compartida() -> None:
    from cfdtrader.backtest import engine

    assert gate.Direction is engine.Direction
    output = call()
    assert output.direction is Direction.LONG
    decision = to_engine_decision(output)
    assert isinstance(decision, Decision)
    assert decision.direction is engine.Direction.LONG
    assert isinstance(decision.notional_usd, Decimal)
    assert decision.notional_usd > 0
    assert decision.probability == 0.6
    assert decision.stop_px is None and decision.target_px is None
    assert output.to_engine_decision() == decision

    # El motor de #13 acepta la decision: nocional explicito y positivo, barreras las dos o
    # ninguna. Se le pasan sus **propias** comprobaciones para que el "valido" sea medido.
    engine._require_notional(decision)  # pyright: ignore[reportPrivateUsage]
    engine._require_geometry(decision=decision, open_px=5000.0)  # pyright: ignore[reportPrivateUsage]

    with_prices = to_engine_decision(output, entry_px=5000.0)
    assert with_prices.stop_px is not None and with_prices.target_px is not None
    assert with_prices.stop_px < 5000.0 < with_prices.target_px
    engine._require_geometry(  # pyright: ignore[reportPrivateUsage]
        decision=with_prices, open_px=5000.0
    )

    short = call(prob_up_calibrated=0.4)
    assert short.direction is Direction.SHORT
    short_prices = to_engine_decision(short, entry_px=5000.0)
    assert short_prices.stop_px is not None and short_prices.target_px is not None
    assert short_prices.target_px < 5000.0 < short_prices.stop_px
    engine._require_geometry(  # pyright: ignore[reportPrivateUsage]
        decision=short_prices, open_px=5000.0
    )

    nothing = call(trades_today=1)
    assert nothing.direction is Direction.NOTHING
    stopped = to_engine_decision(nothing)
    assert stopped.direction is Direction.NOTHING
    assert stopped.notional_usd is None


# ─────────────────────────────────────────────────────────────────────────────
# A5 — el nocional sale del riesgo y del stop; el apalancamiento es derivado
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_sizing_desde_el_stop() -> None:
    output = call()
    assert output.notional_usd == Decimal("20000.00")
    assert output.leverage_implied == Decimal("2.0000")
    assert output.notional_usd == Decimal("10000") * Decimal("1") / Decimal("0.5")

    wider = call(capital_usd=Decimal("100000"), stop_pct=Decimal("0.3"))
    assert wider.notional_usd == Decimal("333333.33")  # cuantizado a centimos
    assert wider.leverage_implied == Decimal("3.3333")

    smaller = call(prob_up_calibrated=0.4, target_pct=Decimal("1.0"))
    assert smaller.notional_usd == Decimal("20000.00")
    assert smaller.leverage_implied == Decimal("2.0000")

    assert "leverage" not in inspect.signature(evaluate_gate).parameters
    assert "leverage_implied" not in inspect.signature(evaluate_gate).parameters


# ─────────────────────────────────────────────────────────────────────────────
# A6 — unidades en porcentaje, copiadas del `CostBreakdown` sin conversion
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_unidades_del_ev() -> None:
    breakdown = cost(measured("0"))
    output = call(cost=breakdown)
    assert breakdown.c_declared_pct == Decimal("0.0042")
    assert breakdown.c_total_pct == Decimal("0.0042")
    # Copiado tal cual: ni x100 ni /100.
    assert output.cost_pct == breakdown.c_declared_pct
    assert output.cost_total_pct == breakdown.c_total_pct
    assert output.cost_pct != breakdown.c_declared_pct * 100
    assert output.cost_pct != breakdown.c_declared_pct / 100
    # Caso a mano: 0,60 x 1,0 % - 0,0042 % = 0,5958 %.
    assert output.ev_declared_pct == Decimal("0.5958")
    assert output.ev_net_pct == Decimal("0.5958")
    forbidden = {
        "fraccion": Decimal("0.005958"),
        "x100": Decimal("59.58"),
        "restando_100x_el_coste": Decimal("0.18"),
    }
    for label, value in forbidden.items():
        assert output.ev_declared_pct != value, label
    assert output.cost_pct != Decimal("0.42")


def test_a6_el_ev_usa_la_probabilidad_de_la_direccion() -> None:
    long = call(prob_up_calibrated=0.6)
    short = call(prob_up_calibrated=0.4)
    assert long.direction is Direction.LONG
    assert short.direction is Direction.SHORT
    assert long.ev_declared_pct == short.ev_declared_pct == Decimal("0.5958")


# ─────────────────────────────────────────────────────────────────────────────
# A7 — los tres estados de slippage no se fusionan
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_slippage_no_se_fusiona() -> None:
    medido = call(cost=cost(measured("0.2")))
    asumido = call(cost=cost(declared_slippage_assumption()))
    sin_medir = call(cost=cost(unmeasured()))

    assert medido.slippage_state == "measured"
    assert asumido.slippage_state == "assumed"
    assert sin_medir.slippage_state == "unmeasured"
    assert len({medido.slippage_state, asumido.slippage_state, sin_medir.slippage_state}) == 3

    # El EV declarado se publica siempre y no depende del estado del slippage.
    for output in (medido, asumido, sin_medir):
        assert output.ev_declared_pct == Decimal("0.5958")
        assert output.cost_pct == Decimal("0.0042")

    # El total solo se cierra con una medicion: los otros dos dejan el EV neto en None.
    assert medido.cost_total_pct == Decimal("0.2042")
    assert medido.ev_net_pct == Decimal("0.3958")
    assert asumido.cost_total_pct is None
    assert asumido.ev_net_pct is None
    assert sin_medir.cost_total_pct is None
    assert sin_medir.ev_net_pct is None
    assert asumido.ev_net_pct is not Decimal(0)
    assert sin_medir.ev_net_pct is not Decimal(0)

    # Un total que no se puede cerrar no se inventa: la regla 9 lo bloquea explicitamente (y
    # la senal, sin EV neto demostrable, no pasa del tier C).
    for output in (asumido, sin_medir):
        assert output.status is GateStatus.RECOMMENDATION
        assert output.direction is Direction.NOTHING
        assert output.tier == "C"
        assert codes(output) == {"ev_neto_no_calculable": "9", "tier_no_autorizado": "10"}
        assert "#62" in output.blockers[0]["detail"]
        assert rule_outcomes(output)["9"] == "blocked"

    assert len({medido.gate_sha256, asumido.gate_sha256, sin_medir.gate_sha256}) == 3

    # El motivo que viaja al motor tampoco inventa un EV neto cuando no lo hay.
    reason = to_engine_decision(asumido).reason
    assert "ev_net_pct" not in reason
    assert "ev_neto_no_calculable(regla 9)" in reason


# ─────────────────────────────────────────────────────────────────────────────
# A8 — todo `None` es `undecided`; los umbrales llegan declarados, no cableados
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_undecided_y_sin_literales() -> None:
    output = call(params=GateParameters())
    assert output.status is GateStatus.NO_RECOMMENDATION_UNDECIDED
    assert output.direction is None
    assert output.notional_usd is None
    assert output.leverage_implied is None
    assert output.blockers == ()
    assert [entry["parameter"] for entry in output.undecided] == [
        name for name, _issue, _note in gate.PARAMETER_ISSUES
    ]
    issues = {entry["parameter"]: entry["issue"] for entry in output.undecided}
    assert len(issues) == 11
    assert issues["broker"] == "#59"
    assert set(issues.values()) == {"#59", "#60"}
    assert all(issues[name] == "#60" for name in issues if name != "broker")
    assert all(entry["detail"] for entry in output.undecided)
    assert set(output.params) == set(issues)
    assert set(output.params.values()) == {None}
    assert output.tier == "C"

    # Una sola decision sin cerrar ya bloquea: no se evalua nada con un umbral que no existe.
    partial = call(params=params(ev_threshold_pct=None))
    assert partial.status is GateStatus.NO_RECOMMENDATION_UNDECIDED
    assert [entry["parameter"] for entry in partial.undecided] == ["ev_threshold_pct"]

    # Dos juegos de umbrales distintos dan decisiones distintas (y hashes distintos).
    estricto = call(params=params(ev_threshold_pct=Decimal("0.9")))
    laxo = call(params=params(ev_threshold_pct=Decimal("0.0")))
    assert estricto.direction is Direction.NOTHING
    assert estricto.tier == "A"
    assert codes(estricto) == {"ev_bajo_el_umbral": "9"}
    assert laxo.direction is Direction.LONG
    assert laxo.blockers == ()
    assert estricto.gate_sha256 != laxo.gate_sha256

    # Ni el capricho de cablear un tier: el mismo EV es B con la probabilidad de C y A sin ella.
    con_minimo = call(prob_up_calibrated=0.55)
    sin_minimo = call(prob_up_calibrated=0.55, params=params(tier_a_min_probability=Decimal("0.5")))
    assert con_minimo.tier == "B"
    assert sin_minimo.tier == "A"

    # Y la configuracion del repositorio no gana un bloque de umbrales.
    document = yaml.safe_load(SETTINGS.read_text(encoding="utf-8"))
    assert (
        set(all_keys(document))
        & {
            "thresholds",
            "umbrales",
            "gate",
            "tiers",
            "risk",
            "risks",
            "capital",
            "ev_threshold_pct",
            "risk_per_trade_pct",
            "max_daily_loss_pct",
            "authorized_tiers",
        }
        == set()
    )


def all_keys(node: object) -> set[str]:
    """Todas las claves de un documento YAML, a cualquier profundidad."""
    if isinstance(node, dict):
        mapping = cast("dict[object, object]", node)
        found = {str(key) for key in mapping}
        for value in mapping.values():
            found |= all_keys(value)
        return found
    if isinstance(node, list):
        found: set[str] = set()
        for item in cast("list[object]", node):
            found |= all_keys(item)
        return found
    return set()


# ─────────────────────────────────────────────────────────────────────────────
# A9 — reglas 1, 3, 4 y 5 con el signo correcto
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_kill_switches() -> None:
    una_operacion = call(trades_today=1)
    assert una_operacion.status is GateStatus.RECOMMENDATION
    assert una_operacion.direction is Direction.NOTHING
    assert codes(una_operacion) == {"max_una_operacion_por_sesion": "1"}
    assert call(trades_today=0).direction is Direction.LONG

    diaria = call(daily_pnl_pct=Decimal("-2"))
    assert codes(diaria) == {"perdida_diaria": "3"}
    assert diaria.direction is Direction.NOTHING
    assert call(daily_pnl_pct=Decimal("-1.9")).direction is Direction.LONG
    assert call(daily_pnl_pct=Decimal("-2.5")).direction is Direction.NOTHING
    assert call(daily_pnl_pct=Decimal("2")).direction is Direction.LONG

    semanal = call(weekly_pnl_pct=Decimal("-5"))
    assert codes(semanal) == {"perdida_semanal": "4"}
    assert call(weekly_pnl_pct=Decimal("-4.99")).direction is Direction.LONG

    mensual = call(monthly_pnl_pct=Decimal("-10"))
    assert codes(mensual) == {"perdida_mensual": "5"}
    assert call(monthly_pnl_pct=Decimal("-9.99")).direction is Direction.LONG

    # Los umbrales son los declarados: con otro limite, la misma perdida no bloquea.
    otro_limite = call(daily_pnl_pct=Decimal("-2"), params=params(max_daily_loss_pct=Decimal("3")))
    assert otro_limite.direction is Direction.LONG

    # Los bloqueos se acumulan: un dia de FOMC con la operacion ya hecha trae los dos motivos.
    acumulados = call(trades_today=1, fomc_dates=(SESSION,))
    assert codes(acumulados) == {"max_una_operacion_por_sesion": "1", "dia_de_fomc": "17"}
    assert rule_outcomes(acumulados)["1"] == "blocked"
    assert rule_outcomes(acumulados)["17"] == "blocked"
    assert rule_outcomes(acumulados)["10"] == "not_evaluated"


# ─────────────────────────────────────────────────────────────────────────────
# A10 — reglas 6, 7, 8, 9 y 12
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_reglas_de_riesgo() -> None:
    # Regla 6: intradia puro. Un coste con noches se rechaza, no se ignora.
    assert call(cost=cost(measured("0"))).cost_pct == Decimal("0.0042")
    con_noche = cost(measured("0"), nights=1)
    assert con_noche.nights == 1
    with pytest.raises(GateInputError, match="nights"):
        call(cost=con_noche)
    assert rule_outcomes(call())["6"] == "pass"

    # Regla 7: sin stop (o con un stop no positivo) no hay decision direccional.
    with pytest.raises(GateInputError, match="stop_pct"):
        call(stop_pct=cast("Decimal", None))
    with pytest.raises(GateInputError, match="stop_pct"):
        call(stop_pct=Decimal("0"))
    with pytest.raises(GateInputError, match="stop_pct"):
        call(stop_pct=Decimal("-0.5"))

    # Regla 8: el objetivo tiene que cubrir 2x el coste declarado.
    justo = call(target_pct=Decimal("0.0084"))
    assert justo.direction is Direction.LONG
    assert rule_outcomes(justo)["8"] == "pass"
    corto = call(target_pct=Decimal("0.008"))
    assert corto.direction is Direction.NOTHING
    assert codes(corto) == {"objetivo_bajo_el_coste": "8"}
    assert rule_outcomes(corto)["8"] == "blocked"

    # Regla 9: umbral estricto.
    assert call(params=params(ev_threshold_pct=Decimal("0.5957"))).direction is Direction.LONG
    en_el_umbral = call(params=params(ev_threshold_pct=Decimal("0.5958")))
    assert en_el_umbral.direction is Direction.NOTHING
    assert codes(en_el_umbral) == {"ev_bajo_el_umbral": "9"}

    # Regla 12: dos llamadas identicas no modifican nada (funcion pura y sin estado).
    first = call()
    second = call()
    assert first == second
    assert first.rules == second.rules
    assert first.gate_sha256 == second.gate_sha256
    assert not any(isinstance(node, ast.Global) for node in ast.walk(module_tree()))
    assert rule_outcomes(first)["12"] == "pass"


# ─────────────────────────────────────────────────────────────────────────────
# A11 — regla 10 (tiers) y la regla 11 declarada como delegada
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_tiers() -> None:
    # Tier A: EV neto por encima del multiplo declarado y probabilidad por encima del minimo.
    assert call(cost=cost(measured("0.2"))).tier == "A"
    assert call(cost=cost(measured("0.2"))).direction is Direction.LONG

    # Tier B: falta la probabilidad minima => se publica como B y devuelve NOTHING.
    tier_b = call(cost=cost(measured("0.2")), prob_up_calibrated=0.55)
    assert tier_b.tier == "B"
    assert tier_b.status is GateStatus.RECOMMENDATION
    assert tier_b.direction is Direction.NOTHING
    assert codes(tier_b) == {"tier_no_autorizado": "10"}
    assert "tier B" in tier_b.blockers[0]["detail"]

    # Tier C: el EV neto no llega al multiplo del tier B.
    tier_c = call(
        cost=cost(measured("0.01")),
        expected_move_pct=Decimal("0.035"),
        params=params(ev_threshold_pct=Decimal("0")),
    )
    assert tier_c.tier == "C"
    assert tier_c.status is GateStatus.RECOMMENDATION
    assert tier_c.direction is Direction.NOTHING
    assert codes(tier_c) == {"tier_no_autorizado": "10"}

    # El tier autorizado es un parametro: autorizar el C lo deja operar.
    all_tiers = call(
        cost=cost(measured("0.01")),
        expected_move_pct=Decimal("0.035"),
        params=params(ev_threshold_pct=Decimal("0"), authorized_tiers=("A", "B", "C")),
    )
    assert all_tiers.tier == "C"
    assert all_tiers.direction is Direction.LONG
    assert all_tiers.blockers == ()

    # La regla 11 no es por sesion: se declara delegada a #28, en RULES y en LIMITATIONS.
    assert rule_outcomes(call())["11"] == "delegated"
    reglas = {entry["rule"]: entry for entry in gate.RULES}
    assert reglas["11"]["owner"] == "#28"
    assert reglas["11"]["issue"] == "#28"
    limitations = {entry["id"]: entry for entry in gate.LIMITATIONS}
    assert limitations["regla_11"]["issue"] == "#28"
    assert "#28" in limitations["regla_11"]["statement"]
    assert len(gate.RULES) == 18
    assert all({"rule", "title", "owner", "issue", "note"} <= set(entry) for entry in gate.RULES)


# ─────────────────────────────────────────────────────────────────────────────
# A12 — reglas 13, 14 y 15: los estados no se fusionan
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_estados() -> None:
    members = {member.value: member for member in GateStatus}
    assert len(members) == 5
    assert "nothing" not in members
    assert members["no_recommendation_undecided"] not in {
        members["recommendation"],
        members["no_recommendation_stale_data"],
        members["no_recommendation_data_quality"],
        members["error"],
    }

    ayer = call(as_of=datetime(2026, 9, 22, 12, 45, tzinfo=UTC))
    assert ayer.status is GateStatus.NO_RECOMMENDATION_STALE_DATA
    assert ayer.direction is None
    assert ayer.notional_usd is None
    assert codes(ayer) == {"as_of_no_es_de_hoy": "13"}
    assert rule_outcomes(ayer)["13"] == "blocked"
    assert call().as_of.date() == TODAY

    calidad = call(snapshot_ok=False)
    assert calidad.status is GateStatus.NO_RECOMMENDATION_DATA_QUALITY
    assert calidad.direction is None
    assert codes(calidad) == {"snapshot_no_valido": "14"}

    observacion = call(observation_sessions_remaining=5)
    assert observacion.direction is Direction.NOTHING
    assert observacion.status is GateStatus.RECOMMENDATION
    assert codes(observacion) == {"modo_observacion": "15"}
    assert call(observation_sessions_remaining=0).direction is Direction.LONG

    # "Hoy no veo oportunidad" y "no se" son cosas distintas y se distinguen sin ambiguedad.
    nada = call(trades_today=1)
    assert nada.status is GateStatus.RECOMMENDATION
    assert nada.direction is Direction.NOTHING
    assert ayer.direction is not Direction.NOTHING
    assert {ayer.direction, calidad.direction} == {None}
    assert GateStatus.RECOMMENDATION.value not in {ayer.status.value, calidad.status.value}


# ─────────────────────────────────────────────────────────────────────────────
# A13 — reglas 16, 17 y 18
# ─────────────────────────────────────────────────────────────────────────────
class _SpyCalendar(MarketCalendar):
    """Calendario real que **cuenta** las consultas a ``is_half_day`` (regla 18)."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.half_day_calls: list[date] = []

    def is_half_day(self, day: date) -> bool:
        self.half_day_calls.append(day)
        return super().is_half_day(day)


def test_a13_bracket_fomc_media_sesion() -> None:
    spy = _SpyCalendar(years=tuple(range(2020, 2030)))
    assert CALENDAR.is_half_day(HALF_SESSION) is True
    assert CALENDAR.is_half_day(SESSION) is False

    # Regla 16: el bracket es obligatorio y las barreras van las dos o ninguna.
    assert call().bracket_required is True
    assert call(trades_today=1).bracket_required is True
    assert call(params=GateParameters()).bracket_required is True
    assert call(as_of=datetime(2026, 9, 22, 12, 45, tzinfo=UTC)).bracket_required is True

    sin_objetivo = call(target_pct=None)
    assert sin_objetivo.direction is Direction.NOTHING
    assert codes(sin_objetivo) == {"bracket_sin_objetivo": "16"}
    assert sin_objetivo.stop_pct == Decimal("0.5")
    assert sin_objetivo.target_pct is None
    assert sin_objetivo.stop_px is None and sin_objetivo.target_px is None
    decision = to_engine_decision(sin_objetivo, entry_px=5000.0)
    assert decision.stop_px is None and decision.target_px is None

    con_objetivo = to_engine_decision(call(), entry_px=5000.0)
    assert con_objetivo.stop_px is not None and con_objetivo.target_px is not None

    # Regla 17: la sesion esta en el conjunto declarado de FOMC (la ingesta es #34).
    fomc = call(fomc_dates=(date(2026, 9, 22), SESSION))
    assert fomc.is_fomc_session is True
    assert fomc.fomc_dates_count == 2
    assert fomc.direction is Direction.NOTHING
    assert codes(fomc) == {"dia_de_fomc": "17"}
    assert call(fomc_dates=(date(2026, 9, 22),)).is_fomc_session is False

    # Regla 18: se pregunta al calendario, que es quien sabe de medias sesiones.
    media = evaluate_gate(
        session=HALF_SESSION,
        as_of=AS_OF,
        today=TODAY,
        calendar=spy,
        prob_up_calibrated=0.6,
        expected_move_pct=Decimal("1.0"),
        expected_move_basis="sigma_k (test)",
        cost=cost(measured("0")),
        capital_usd=Decimal("10000"),
        snapshot_ok=True,
        stop_pct=Decimal("0.5"),
        target_pct=Decimal("1.0"),
        fomc_dates=(),
        params=params(),
    )
    assert spy.half_day_calls == [HALF_SESSION]
    assert media.is_half_session is True
    assert media.status is GateStatus.RECOMMENDATION
    assert media.direction is Direction.NOTHING
    assert codes(media) == {"media_sesion": "18"}

    # El AST no trae ninguna tabla de festivos ni de medias sesiones: se pregunta al calendario.
    tree = module_tree()
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "date"
    ]
    assert "holidays" not in imported_modules(tree)
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, date)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Apoyo: validacion de entradas, parametros y traduccion al motor
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("session", datetime(2026, 9, 23, 12, 0, tzinfo=UTC)),
        ("session", "2026-09-23"),
        ("as_of", SESSION),
        ("today", datetime(2026, 9, 23, 0, 0, tzinfo=UTC)),
        ("calendar", object()),
        ("cost", object()),
        ("params", object()),
        ("prob_up_calibrated", 1.4),
        ("prob_up_calibrated", "0.6"),
        ("prob_up_calibrated", True),
        ("expected_move_pct", 1.0),
        ("expected_move_pct", Decimal("-1")),
        ("expected_move_basis", "  "),
        ("capital_usd", Decimal("0")),
        ("capital_usd", 10000),
        ("capital_usd", Decimal("NaN")),
        ("stop_pct", "0.5"),
        ("target_pct", Decimal("0")),
        ("target_pct", Decimal("-1")),
        ("fomc_dates", "2026-09-23"),
        ("fomc_dates", (1,)),
        ("trades_today", -1),
        ("trades_today", True),
        ("trades_today", "1"),
        ("observation_sessions_remaining", -3),
        ("daily_pnl_pct", -2.0),
        ("weekly_pnl_pct", "5"),
        ("snapshot_ok", 1),
    ),
)
def test_apoyo_entradas_invalidas(field: str, value: object) -> None:
    with pytest.raises(GateInputError):
        call(**{field: value})


def test_apoyo_parametros_invalidos() -> None:
    with pytest.raises(GateInputError, match="broker"):
        GateParameters(broker="   ")
    with pytest.raises(GateInputError, match="risk_per_trade_pct"):
        GateParameters(risk_per_trade_pct=Decimal("0"))
    with pytest.raises(GateInputError, match="ev_threshold_pct"):
        GateParameters(ev_threshold_pct=Decimal("-1"))
    with pytest.raises(GateInputError, match="max_weekly_loss_pct"):
        GateParameters(max_weekly_loss_pct=Decimal("-5"))
    with pytest.raises(GateInputError, match="r_pct"):
        GateParameters(r_pct=Decimal("0"))
    with pytest.raises(GateInputError, match="tier_a_min_probability"):
        GateParameters(tier_a_min_probability=Decimal("1"))
    with pytest.raises(GateInputError, match="tier_a_min_probability"):
        GateParameters(tier_a_min_probability=Decimal("0"))
    with pytest.raises(GateInputError, match="authorized_tiers"):
        GateParameters(authorized_tiers=("D",))
    with pytest.raises(GateInputError, match="tier_a_cost_multiple"):
        GateParameters(tier_a_cost_multiple=Decimal("1"), tier_b_cost_multiple=Decimal("2"))

    # Un `None` no se valida (es "sin decidir") y el 0 del umbral de EV si es legitimo.
    assert GateParameters().model_fields_set == frozenset()
    assert GateParameters(ev_threshold_pct=Decimal("0")).ev_threshold_pct == Decimal("0")
    assert GateParameters(authorized_tiers=("A", "B")).authorized_tiers == ("A", "B")
    with pytest.raises(ValidationError):
        GateParameters.model_validate({"inexistente": "x"})


def test_apoyo_traduccion_al_motor() -> None:
    with pytest.raises(GateInputError, match="GateOutput"):
        to_engine_decision(cast("GateOutput", object()))
    output = call()
    with pytest.raises(GateInputError, match="entry_px"):
        to_engine_decision(output, entry_px=cast("float", "5000"))
    with pytest.raises(GateInputError, match="entry_px"):
        to_engine_decision(output, entry_px=0.0)
    with pytest.raises(GateInputError, match="entry_px"):
        to_engine_decision(output, entry_px=float("inf"))
    with pytest.raises(GateInputError, match="notional_usd"):
        to_engine_decision(output.model_copy(update={"notional_usd": None}))

    # `prob_up_calibrated` en el 0,5 exacto mira arriba (el criterio de #24/#26), pero la
    # probabilidad minima del tier A no llega: se registra como B y devuelve NOTHING.
    mitad = call(prob_up_calibrated=0.5)
    assert mitad.direction is Direction.NOTHING
    assert mitad.ev_declared_pct == Decimal("0.4958")
    assert mitad.tier == "B"
    assert mitad.leverage_implied is None
    assert to_engine_decision(output, entry_px=100.0).stop_px == 99.5


def test_apoyo_payload_y_detalles() -> None:
    output = call()
    assert output.expected_move_basis == "sigma_k: k declarado por #60 (test)"
    assert output.fomc_dates_count == 0
    assert output.params["authorized_tiers"] == "A"
    assert output.params["risk_per_trade_pct"] == "1"
    assert output.params["ev_threshold_pct"] == "0.0084"
    assert output.slippage_state == "measured"
    assert output.stop_pct == Decimal("0.5")
    assert output.target_pct == Decimal("1.0")
    assert output.expected_move_pct == Decimal("1.0")
    assert "ev_net_pct=0.5958" in to_engine_decision(output).reason
    assert "status=recommendation" in to_engine_decision(output).reason
    assert "max_una_operacion_por_sesion" in to_engine_decision(call(trades_today=1)).reason
    assert set(output.model_dump()) == {
        "session",
        "as_of",
        "today",
        "status",
        "direction",
        "tier",
        "prob_up_calibrated",
        "expected_move_pct",
        "expected_move_basis",
        "cost_pct",
        "cost_total_pct",
        "slippage_state",
        "ev_declared_pct",
        "ev_net_pct",
        "stop_pct",
        "target_pct",
        "stop_px",
        "target_px",
        "notional_usd",
        "leverage_implied",
        "bracket_required",
        "trades_today",
        "observation_sessions_remaining",
        "is_fomc_session",
        "is_half_session",
        "fomc_dates_count",
        "params",
        "blockers",
        "undecided",
        "rules",
        "gate_sha256",
    }
    # Cambiar cualquier entrada declarada cambia el hash de la salida.
    assert call(snapshot_ok=False).gate_sha256 != output.gate_sha256
    assert gate.FOLLOW_UPS, "las fronteras declaradas no pueden estar vacias"
    assert gate.LIMITATIONS
