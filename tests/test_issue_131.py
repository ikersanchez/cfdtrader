"""Tests del cableado del `R` decidido en #60 (tarea **#131**): A2, A4-A9, A11 y A12.

El supuesto de *slippage* declarado en #64 se **cuantifica** con el `R` que el propietario decidió
en #60 (1,00 % del nocional) y **sigue siendo un supuesto**: `state = "assumed"` e
`is_measurement = false` viajan tal cual. La consecuencia es la que manda `plan.md` §19.12: la
regla 9 se verifica sobre el **coste declarado** y el EV bajo el supuesto se publica como
**sensibilidad**, también cuando sale negativo.

Nada de red, reloj ni disco: el motor de costes y el gate son funciones puras, así que todo se mide
con argumentos explícitos.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final

import pytest

from cfdtrader.analysis.backtest_report import NOTIONAL_USD
from cfdtrader.analysis.pipeline_report import (
    EXPECTED_MOVE_BASIS,
    SCENARIO_R_PCT,
    scenario_parameters,
)
from cfdtrader.backtest import costs
from cfdtrader.backtest.costs import (
    CostModelError,
    MeasureState,
    Side,
    SlippageParameter,
    cost_breakdown,
    declared_cost_model,
    declared_slippage_assumption,
)
from cfdtrader.data.calendar import EASTERN, load_calendar
from cfdtrader.decision import gate
from cfdtrader.delivery import run_daily

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
RUN_DAILY_SOURCE: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "delivery" / "run_daily.py"

SESSION: Final[date] = date(2026, 9, 17)
AS_OF: Final[datetime] = datetime(2026, 9, 17, 12, 0, tzinfo=UTC).astimezone(EASTERN)

#: El supuesto cuantificado con el `R` de #60: 20 % de 1,00 % = **0,2 %** (20 bp).
EXPECTED_SLIPPAGE_PCT: Final[Decimal] = Decimal("0.2")


def _cost(r_pct: Decimal | None = None) -> costs.CostBreakdown:
    """El coste declarado con el supuesto sin cuantificar (por defecto) o cuantificado por `R`."""
    slippage = (
        declared_slippage_assumption() if r_pct is None else declared_slippage_assumption(r_pct)
    )
    return cost_breakdown(
        model=declared_cost_model(),
        slippage=slippage,
        notional_usd=NOTIONAL_USD,
        side=Side.LONG,
        nights=0,
    )


def _output(
    cost: costs.CostBreakdown,
    params: gate.GateParameters,
    *,
    probability: float = 0.62,
    move: str = "1.5",
) -> gate.GateOutput:
    """El gate real, con sus argumentos explícitos: no hay almacén ni reloj de por medio."""
    stop = Decimal(move)
    return gate.evaluate_gate(
        session=SESSION,
        as_of=AS_OF,
        today=SESSION,
        calendar=load_calendar(),
        prob_up_calibrated=probability,
        expected_move_pct=Decimal(move),
        expected_move_basis=EXPECTED_MOVE_BASIS,
        cost=cost,
        capital_usd=NOTIONAL_USD,
        snapshot_ok=True,
        stop_pct=stop,
        target_pct=Decimal("2") * stop,
        fomc_dates=(),
        params=params,
        trades_today=0,
    )


def _codes(output: gate.GateOutput) -> dict[str, str]:
    return {entry["code"]: entry["rule"] for entry in output.blockers}


def _rule_detail(output: gate.GateOutput, rule: str) -> str:
    return next(entry["detail"] for entry in output.rules if entry["rule"] == rule)


# ─────────────────────────────────────────────────────────────────────────────
# A2 · El supuesto cuantificado deriva exacto y sigue siendo un supuesto
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_quantified_assumption_derives_exactly_and_is_still_assumed() -> None:
    quantified = declared_slippage_assumption(SCENARIO_R_PCT)
    assert quantified.state is MeasureState.ASSUMED
    assert quantified.is_measurement is False
    assert quantified.r_pct == SCENARIO_R_PCT
    assert quantified.pct_of_notional == EXPECTED_SLIPPAGE_PCT
    assert quantified.pct_of_notional == quantified.pct_of_r * quantified.r_pct / Decimal(100)
    assert quantified.decided_on == "2026-09-18"
    # El marcador de la decision de #60 es el propio `r_pct`: no hay campo nuevo que se pueda
    # publicar de mas (el informe del motor tiene que seguir siendo byte a byte el mismo).
    assert set(quantified.model_dump()) == set(declared_slippage_assumption().model_dump())


def test_a2_a_typed_number_is_rejected_without_the_declared_derivation() -> None:
    payload = {
        "state": MeasureState.ASSUMED,
        "is_measurement": False,
        "pct_of_r": Decimal("20"),
        "r_pct": Decimal("1"),
        "pct_of_notional": Decimal("0.19"),  # no sale de 20 % x 1 %
        "source": "origen",
        "reason": "motivo",
        "decided_on": "2026-09-18",
        "follow_up_issue": "#62",
    }
    with pytest.raises(CostModelError, match="pct_of_notional"):
        SlippageParameter.model_validate(payload)
    # Sin la marca de la decision de #60, el `R` sigue prohibido (el contrato de #64 no se mueve).
    with pytest.raises(CostModelError, match="r_pct"):
        SlippageParameter.model_validate({**payload, "pct_of_notional": None})


# ─────────────────────────────────────────────────────────────────────────────
# A4 · El motor cierra el total con el `R` decidido y `r_pct` deja de ser null
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_the_engine_closes_the_total_with_the_decided_r() -> None:
    plain = _cost()
    quantified = _cost(SCENARIO_R_PCT)
    assert plain.c_total_pct is None
    assert plain.nulls, "sin cuantificar, el hueco del *slippage* sigue declarado"
    assert quantified.slippage_pct == EXPECTED_SLIPPAGE_PCT
    assert quantified.c_total_pct == quantified.c_declared_pct + EXPECTED_SLIPPAGE_PCT
    assert quantified.nulls == ()
    payload = costs.report_payload(slippage=declared_slippage_assumption(SCENARIO_R_PCT))
    assert payload["slippage"]["state"] == "assumed"
    assert payload["slippage"]["is_measurement"] is False
    assert payload["slippage"]["r_pct"] == "1"
    assert payload["c_total"]["pct"] == "0.2042"
    # El informe **no** gana ninguna clave: el bloque del *slippage* ya dice que el supuesto esta
    # cuantificado con `r_pct` y `pct_of_notional`, y anadir una clave a un bloque publicado
    # invalidaria el artefacto ya emitido (`phase1_backtest`, #29) sin necesidad.
    assert set(payload["slippage"]) == set(
        costs.report_payload(slippage=declared_slippage_assumption())["slippage"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# A5 · Los dos `r_pct` quedan conectados
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_the_two_r_pct_must_agree() -> None:
    cost = _cost(Decimal("1.5"))
    params = scenario_parameters(cost_pct=cost.c_declared_pct)
    assert params.r_pct == SCENARIO_R_PCT
    with pytest.raises(gate.GateInputError, match="r_pct"):
        _output(cost, params)


# ─────────────────────────────────────────────────────────────────────────────
# A6/A7 · La regla 9 se decide sobre el coste declarado y publica la sensibilidad
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_a7_the_declared_basis_decides_and_publishes_the_sensitivity() -> None:
    cost = _cost(SCENARIO_R_PCT)
    params = scenario_parameters(cost_pct=cost.c_declared_pct)
    output = _output(cost, params)
    assert output.ev_basis == gate.BASIS_DECLARED
    assert output.slippage_state == "assumed"
    assert output.ev_decision_pct == output.ev_declared_pct
    assert output.ev_decision_pct > params.ev_threshold_pct  # type: ignore[operator]
    assert output.ev_net_is_sensitivity is True
    assert output.ev_net_pct == output.ev_declared_pct - EXPECTED_SLIPPAGE_PCT
    assert output.tier == gate.TIER_A
    assert _codes(output) == {}
    assert "sensibilidad bajo el supuesto declarado" in _rule_detail(output, "9")
    # El neto bajo el supuesto se publica **aunque sea negativo**: es la sensibilidad, no 0.
    negative = _output(cost, params, move="0.2")
    assert negative.ev_net_pct is not None and negative.ev_net_pct < 0
    assert negative.ev_net_is_sensitivity is True


# ─────────────────────────────────────────────────────────────────────────────
# A8/A9 · Sin cuantificar nada cambia; con el *slippage* medido decide el neto
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_without_the_decided_r_nothing_changes() -> None:
    cost = _cost()
    output = _output(cost, scenario_parameters(cost_pct=cost.c_declared_pct))
    assert output.ev_basis == gate.BASIS_NONE
    assert output.ev_net_pct is None
    assert output.ev_net_is_sensitivity is False
    assert output.ev_decision_pct is None
    assert output.tier == gate.TIER_C
    assert _codes(output) == {"ev_neto_no_calculable": "9", "tier_no_autorizado": "10"}


def test_a9_a_measured_slippage_still_decides_on_the_net() -> None:
    measured = SlippageParameter.measured(
        pct_of_notional=Decimal("0.0001"),
        source="prueba unitaria",
        reason="medicion declarada en la prueba",
    )
    cost = cost_breakdown(
        model=declared_cost_model(),
        slippage=measured,
        notional_usd=NOTIONAL_USD,
        side=Side.LONG,
        nights=0,
    )
    output = _output(cost, scenario_parameters(cost_pct=cost.c_declared_pct))
    assert output.ev_basis == gate.BASIS_MEASURED
    assert output.ev_decision_pct == output.ev_net_pct
    assert output.ev_net_is_sensitivity is False


# ─────────────────────────────────────────────────────────────────────────────
# A11/A12 · El camino diario: una sola fuente para el `R` y el texto que lo declara
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_a12_the_daily_path_wires_the_decided_r_from_a_single_source() -> None:
    source = RUN_DAILY_SOURCE.read_text(encoding="utf-8")
    assert "declared_slippage_assumption(SCENARIO_R_PCT)" in source, (
        "el camino diario tiene que cuantificar el supuesto con el `R` de S1, importado"
    )
    fence = "\n".join(run_daily.HONESTY_FENCE)
    assert "coste **declarado**" in fence
    assert "sensibilidad" in fence and "#62" in fence
    assert run_daily.scenario_parameters(cost_pct=Decimal("0.0042")).r_pct == SCENARIO_R_PCT
