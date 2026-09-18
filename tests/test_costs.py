"""Tests del motor de costes (#11): tabla declarada, *slippage* obligatorio y CLI.

Un test por criterio de aceptación: la tabla declarada se reproduce **exactamente**
(``Decimal``, sin tolerancia), el *slippage* es un parámetro obligatorio con tres estados
que **nunca** se fusionan, el corte de financiación está declarado como sin verificar y el
módulo es puro (sin I/O, sin red, sin reloj, sin ``data/``).

Los importes se calculan **a mano** en el test (0,42 / 0,24 / 2,24 $ y
0,0042 / 0,0024 / 0,0224 %), no se copian del resultado del módulo.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import sys
import tokenize
from datetime import time
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from cfdtrader.analysis import cost_audit
from cfdtrader.analysis.cost_audit import (
    CARRY_LONG_PCT_PER_NIGHT,
    CARRY_LONG_USD_PER_NIGHT,
    CARRY_SHORT_PCT_PER_NIGHT,
    CARRY_SHORT_USD_PER_NIGHT,
    DECLARED_FINANCING_CUT,
    FX_COST_PCT,
    REFERENCE_NOTIONAL_USD,
    SLIPPAGE_ASSUMPTION_PCT_OF_R,
    SPREAD_PCT,
    SPREAD_USD,
    WINDOW_CONFIRMED_ON,
    MeasureState,
    Side,
)
from cfdtrader.backtest import costs

NOTIONAL = REFERENCE_NOTIONAL_USD
MODULE_PATH = Path(costs.__file__)
SOURCE = MODULE_PATH.read_text(encoding="utf-8")
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Los nombres que A2 exige en la API pública.
REQUIRED_PUBLIC = (
    "CostError",
    "CostModelError",
    "CostInputError",
    "CostModel",
    "CostBreakdown",
    "SlippageParameter",
    "FinancingCut",
    "Side",
    "declared_cost_model",
    "declared_slippage_assumption",
    "cost_breakdown",
    "report_payload",
    "render_markdown",
    "main",
)

#: Los seis campos de coste de A3.
COST_FIELDS = (
    "spread_entry_pct",
    "spread_exit_pct",
    "carry_long_pct_per_night",
    "carry_short_pct_per_night",
    "fx_pct",
    "commission_pct",
)

#: Claves cuyas hojas son importes: tienen que ser cadenas decimales exactas (A27).
QUANTITY_KEYS = frozenset(
    {
        "usd",
        "pct",
        "bp",
        "ratio",
        "nights",
        "notional_usd",
        "pct_of_notional",
        "pct_of_r",
        "r_pct",
        "value_pct_of_r",
        "value_bp",
        "value_pct_of_notional",
        "value_usd",
        "c_pct_of_notional",
        "c_fraction_of_notional",
        "slippage_over_spread_ratio",
        "ratio_quantum",
        "p90_bp",
        "median_bp",
        "max_bp",
        "declared_spread_bp",
        "sessions_above_10_bp",
        "sessions_measured",
        "holding_nights",
        "reference_notional_usd",
        "halves_usd_on_reference_notional",
        "declared_spread_usd",
        "long_usd_per_night",
        "short_usd_per_night",
        "long_pct_per_night",
        "short_pct_per_night",
        "long_usd_on_reference_notional",
        "short_usd_on_reference_notional",
    }
)

#: Palabras que solo pueden aparecer en el módulo como declaración, no como cálculo.
SIZING_NAMES = frozenset(
    {
        "notional_usd",
        "riesgo_por_operacion",
        "distancia_al_stop",
        "apalancamiento",
        "leverage",
        "position_size",
        "risk_per_trade",
        "sizing",
    }
)

OVERNIGHT = "escenario declarado de prueba (una noche)"


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades del test
# ─────────────────────────────────────────────────────────────────────────────
def _breakdown(**kwargs: Any) -> costs.CostBreakdown:
    """Un desglose con el escenario declarado; los argumentos se declaran en el test."""
    model = kwargs.pop("model", None) or costs.declared_cost_model()
    slippage = kwargs.pop("slippage", None) or costs.declared_slippage_assumption()
    nights = kwargs.pop("nights", 0)
    if nights >= 1:
        kwargs.setdefault("overnight_reason", OVERNIGHT)
    return costs.cost_breakdown(
        model=model,
        slippage=slippage,
        notional_usd=kwargs.pop("notional_usd", NOTIONAL),
        side=kwargs.pop("side", Side.LONG),
        nights=nights,
        **kwargs,
    )


def _leaves(node: object, path: str = "") -> list[tuple[str, object]]:
    """Todas las hojas del payload, con su ruta (para los escaneos de A27)."""
    if isinstance(node, dict):
        mapping = cast("dict[str, object]", node)
        return [item for key, value in mapping.items() for item in _leaves(value, f"{path}.{key}")]
    if isinstance(node, list):
        items = cast("list[object]", node)
        return [
            item for index, value in enumerate(items) for item in _leaves(value, f"{path}[{index}]")
        ]
    return [(path, node)]


def _keys(node: object) -> set[str]:
    """Todas las claves del payload (para comprobar que no hay agregados)."""
    if isinstance(node, dict):
        mapping = cast("dict[str, object]", node)
        return set(mapping) | {item for value in mapping.values() for item in _keys(value)}
    if isinstance(node, list):
        return {item for value in cast("list[object]", node) for item in _keys(value)}
    return set()


def _assigned_names(source: str) -> set[str]:
    """Nombres que el módulo **calcula** dentro de sus funciones (no campos ni constantes)."""
    assigned: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Assign):
                assigned.update(t.id for t in inner.targets if isinstance(t, ast.Name))
            elif (
                isinstance(inner, ast.AnnAssign)
                and inner.value is not None
                and isinstance(inner.target, ast.Name)
            ):
                assigned.add(inner.target.id)
    return assigned


def _docstring_and_comment_free_source() -> str:
    """El código del módulo sin comentarios ni docstrings (donde no cabe la tabla, A6)."""
    lines = SOURCE.splitlines()
    masked = [list(line) for line in lines]
    docstring_lines: set[int] = set()
    for node in ast.walk(ast.parse(SOURCE)):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstring_lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    for token in tokenize.generate_tokens(io.StringIO(SOURCE).readline):
        if token.type != tokenize.COMMENT:
            continue
        row, column = token.start
        end_row, end_column = token.end
        if row == end_row:
            for index in range(column, min(end_column, len(masked[row - 1]))):
                masked[row - 1][index] = " "
    kept = [
        "" if number in docstring_lines else "".join(row)
        for number, row in enumerate(masked, start=1)
    ]
    return "\n".join(kept)


def _payload_hash(payload: dict[str, Any]) -> str:
    """sha256 del payload serializado de forma determinista (A24)."""
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stray_costs_files() -> list[str]:
    """Ficheros ``costs_*`` que aparezcan en el repositorio (la CLI no debe crear ninguno)."""
    return sorted(
        str(path.relative_to(REPO_ROOT)) for path in REPO_ROOT.rglob("costs_*") if path.is_file()
    )


def _declared_fields(**overrides: Any) -> dict[str, Any]:
    """Los seis campos de coste declarados, con los retoques que pida el test."""
    fields: dict[str, Any] = {
        "spread_entry_pct": SPREAD_PCT / Decimal(2),
        "spread_exit_pct": SPREAD_PCT / Decimal(2),
        "carry_long_pct_per_night": CARRY_LONG_PCT_PER_NIGHT,
        "carry_short_pct_per_night": CARRY_SHORT_PCT_PER_NIGHT,
        "fx_pct": FX_COST_PCT,
        "commission_pct": Decimal("0.00"),
    }
    fields.update(overrides)
    return fields


# ─────────────────────────────────────────────────────────────────────────────
# A1, A2 — ficheros, frontera declarada y API pública
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_module_and_docstring_declare_the_frontier() -> None:
    assert MODULE_PATH.is_file()
    assert MODULE_PATH.name == "costs.py"
    assert MODULE_PATH.parent.name == "backtest"
    assert (REPO_ROOT / "tests" / "test_costs.py").is_file()
    docstring = costs.__doc__ or ""
    for issue in ("#8", "#13", "#9", "#60"):
        assert issue in docstring, issue
    assert "no mide" in docstring and "no recorre sesiones" in docstring


def test_a2_public_api_is_explicit_and_complete() -> None:
    assert isinstance(costs.__all__, (list, tuple))
    assert set(REQUIRED_PUBLIC) <= set(costs.__all__)
    assert not [name for name in costs.__all__ if name.startswith("_")]
    for name in costs.__all__:
        assert getattr(costs, name) is not None, name


# ─────────────────────────────────────────────────────────────────────────────
# A3, A4, A5 — sin coste por defecto y errores tipados
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_every_cost_field_is_required_and_named_in_the_error() -> None:
    with pytest.raises(ValidationError) as excinfo:
        costs.CostModel.model_validate({})
    message = str(excinfo.value)
    for field in COST_FIELDS:
        assert field in message, field


def test_a3_no_zero_cost_default_in_the_source() -> None:
    for forbidden in ("ZERO_COST", "DEFAULT_COST", "FREE_"):
        assert forbidden not in SOURCE


def test_a4_model_and_slippage_have_no_default() -> None:
    signature = inspect.signature(costs.cost_breakdown)
    for name in ("model", "slippage"):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty
    assert signature.parameters["notional_usd"].default is inspect.Parameter.empty
    assert signature.parameters["nights"].default == 0


def test_a4_calling_without_the_explicit_cost_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="slippage"):
        costs.cost_breakdown(  # pyright: ignore[reportCallIssue]
            model=costs.declared_cost_model(), notional_usd=NOTIONAL, side=Side.LONG
        )
    with pytest.raises(TypeError, match="model"):
        costs.cost_breakdown(  # pyright: ignore[reportCallIssue]
            slippage=costs.declared_slippage_assumption(), notional_usd=NOTIONAL, side=Side.LONG
        )


def test_a4_an_unvalidated_model_is_rejected() -> None:
    unvalidated = costs.CostModel.model_construct()
    assert unvalidated.model_fields_set == set()
    with pytest.raises(costs.CostModelError, match="model"):
        _breakdown(model=unvalidated)


def test_a5_error_hierarchy_and_messages() -> None:
    assert issubclass(costs.CostModelError, costs.CostError)
    assert issubclass(costs.CostInputError, costs.CostError)

    cases: list[tuple[type[Exception], str, str]] = []
    with pytest.raises(costs.CostModelError) as fx_error:
        costs.CostModel.model_validate(_declared_fields(fx_pct=Decimal("0.01")))
    cases.append((costs.CostModelError, "fx_pct", str(fx_error.value)))
    with pytest.raises(costs.CostInputError) as nights_error:
        _breakdown(nights=-1)
    cases.append((costs.CostInputError, "nights", str(nights_error.value)))
    with pytest.raises(costs.CostModelError) as slippage_error:
        costs.SlippageParameter.model_validate(
            {"state": "unmeasured", "reason": "prueba", "pct_of_notional": Decimal("0")}
        )
    cases.append((costs.CostModelError, "pct_of_notional", str(slippage_error.value)))
    with pytest.raises(costs.CostInputError) as notional_error:
        _breakdown(notional_usd=Decimal("0"))
    cases.append((costs.CostInputError, "notional_usd", str(notional_error.value)))

    for error_type, field, message in cases:
        assert error_type.__name__ in {"CostModelError", "CostInputError"}
        assert field in message, (field, message)
        assert ":" in message  # «campo: motivo»


# ─────────────────────────────────────────────────────────────────────────────
# A6 — la tabla declarada se importa, no se re-declara
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_the_declared_model_is_built_from_the_objects_of_8() -> None:
    model = costs.declared_cost_model()
    assert costs.SPREAD_PCT is cost_audit.SPREAD_PCT
    assert costs.REFERENCE_NOTIONAL_USD is cost_audit.REFERENCE_NOTIONAL_USD
    assert costs.CARRY_LONG_PCT_PER_NIGHT is cost_audit.CARRY_LONG_PCT_PER_NIGHT
    assert costs.CARRY_SHORT_PCT_PER_NIGHT is cost_audit.CARRY_SHORT_PCT_PER_NIGHT
    assert costs.FX_COST_PCT is cost_audit.FX_COST_PCT
    assert model.carry_long_pct_per_night is CARRY_LONG_PCT_PER_NIGHT
    assert model.carry_short_pct_per_night is CARRY_SHORT_PCT_PER_NIGHT
    assert model.fx_pct is FX_COST_PCT
    assert model.spread_entry_pct is costs.DECLARED_SPREAD_HALF_PCT
    assert model.spread_entry_pct + model.spread_exit_pct == SPREAD_PCT


def test_a6_the_declared_literals_do_not_appear_in_the_code() -> None:
    code = _docstring_and_comment_free_source()
    for literal in ("0.42", "0.0042", "-0.18", "-0.0018", "1.82", "0.0182", "10000"):
        assert literal not in code, literal


# ─────────────────────────────────────────────────────────────────────────────
# A7, A8 — los totales declarados, exactos
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_declared_round_trip_totals_in_usd() -> None:
    assert _breakdown(side=Side.SHORT, nights=0).c_declared_usd == Decimal("0.42")
    assert _breakdown(side=Side.LONG, nights=0).c_declared_usd == Decimal("0.42")
    assert _breakdown(side=Side.SHORT, nights=1).c_declared_usd == Decimal("0.24")
    assert _breakdown(side=Side.LONG, nights=1).c_declared_usd == Decimal("2.24")


def test_a8_declared_round_trip_totals_in_pct() -> None:
    assert _breakdown(side=Side.SHORT, nights=0).c_declared_pct == Decimal("0.0042")
    assert _breakdown(side=Side.LONG, nights=0).c_declared_pct == Decimal("0.0042")
    assert _breakdown(side=Side.SHORT, nights=1).c_declared_pct == Decimal("0.0024")
    assert _breakdown(side=Side.LONG, nights=1).c_declared_pct == Decimal("0.0224")


def test_a8_pct_and_usd_derive_from_each_other() -> None:
    for nights in (0, 1):
        breakdown = _breakdown(nights=nights)
        assert breakdown.c_declared_usd == NOTIONAL * breakdown.c_declared_pct / Decimal(100)
        assert breakdown.spread_usd == NOTIONAL * breakdown.spread_pct / Decimal(100)
        assert breakdown.carry_usd == NOTIONAL * breakdown.carry_pct / Decimal(100)


# ─────────────────────────────────────────────────────────────────────────────
# A9, A10, A11, A12 — las mitades del diferencial, la divisa, la comisión, la tenencia
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_half_the_spread_entering_and_half_leaving() -> None:
    breakdown = _breakdown()
    assert breakdown.spread_entry_pct == Decimal("0.0021")
    assert breakdown.spread_exit_pct == Decimal("0.0021")
    assert breakdown.spread_entry_usd == Decimal("0.21")
    assert breakdown.spread_exit_usd == Decimal("0.21")
    assert Decimal("0.21") + Decimal("0.21") == Decimal("0.42")
    assert breakdown.spread_entry_usd + breakdown.spread_exit_usd == SPREAD_USD
    assert breakdown.spread_entry_pct + breakdown.spread_exit_pct == SPREAD_PCT
    assert breakdown.spread_usd == Decimal("0.42")
    assert breakdown.spread_entry_source and breakdown.spread_exit_source


def test_a9_the_payload_publishes_both_halves_and_their_sum() -> None:
    payload = costs.report_payload()
    assert payload["spread"]["entry"]["pct"] == "0.0021"
    assert payload["spread"]["exit"]["pct"] == "0.0021"
    assert payload["spread"]["total_pct"] == "0.0042"
    assert payload["spread"]["cross_check_against_8"]["matches_declared"] == "true"


def test_a9_an_asymmetric_spread_needs_a_source_for_each_half() -> None:
    with pytest.raises(costs.CostModelError, match="spread_entry_source"):
        costs.CostModel.model_validate(
            _declared_fields(spread_entry_pct=Decimal("0.0030"), spread_exit_pct=Decimal("0.0012"))
        )
    accepted = costs.CostModel.model_validate(
        _declared_fields(
            spread_entry_pct=Decimal("0.0030"),
            spread_exit_pct=Decimal("0.0012"),
            spread_entry_source="bid/ask de la entrada (declarado)",
            spread_exit_source="bid/ask de la salida (declarado)",
        )
    )
    assert accepted.spread_entry_pct != accepted.spread_exit_pct


def test_a10_fx_is_zero_with_provenance_and_never_a_mute_zero() -> None:
    breakdown = _breakdown()
    assert breakdown.fx_pct == Decimal("0")
    assert breakdown.fx_usd == Decimal("0")
    assert breakdown.fx_state is MeasureState.MEASURED
    assert breakdown.fx_source is not None
    assert "USD" in (breakdown.fx_reason or "")
    assert cost_audit.DECLARED_SETTLEMENT_CURRENCY in (breakdown.fx_reason or "")
    assert WINDOW_CONFIRMED_ON.isoformat() in (breakdown.fx_reason or "")
    assert "#8" in (breakdown.fx_reason or "")


def test_a10_a_non_zero_fx_without_source_or_reason_raises() -> None:
    with pytest.raises(costs.CostModelError, match="fx_pct"):
        costs.CostModel.model_validate(_declared_fields(fx_pct=Decimal("0.01")))
    accepted = costs.CostModel.model_validate(
        _declared_fields(fx_pct=Decimal("0.01"), fx_source="origen", fx_reason="motivo")
    )
    assert accepted.fx_pct == Decimal("0.01")


def test_a11_commission_is_mandatory() -> None:
    fields = _declared_fields()
    fields.pop("commission_pct")
    with pytest.raises(ValidationError) as excinfo:
        costs.CostModel.model_validate(fields)
    assert "commission_pct" in str(excinfo.value)
    assert costs.CostModel.model_fields["commission_pct"].is_required()


def test_a11_a_declared_commission_moves_the_total_exactly() -> None:
    plain = _breakdown()
    richer = costs.declared_cost_model().model_copy(update={"commission_pct": Decimal("0.01")})
    with_commission = _breakdown(model=richer)
    assert with_commission.commission_pct == Decimal("0.01")
    assert with_commission.commission_usd == Decimal("1")
    assert with_commission.c_declared_pct - plain.c_declared_pct == Decimal("0.01")
    assert with_commission.c_declared_usd - plain.c_declared_usd == Decimal("1")
    assert with_commission.c_declared_usd == Decimal("1.42")


def test_a11_the_engine_never_fills_the_commission() -> None:
    payload = costs.report_payload()
    assert payload["commission"]["filled_by_the_engine"] is False
    assert payload["commission"]["pct"] == "0"
    assert payload["commission"]["source"] and payload["commission"]["reason"]
    assert payload["commission"]["state"] == "measured"


def test_a12_carry_is_asymmetric_and_exact() -> None:
    short_one = _breakdown(side=Side.SHORT, nights=1)
    long_one = _breakdown(side=Side.LONG, nights=1)
    assert short_one.carry_usd == Decimal("-0.18")
    assert short_one.carry_pct == Decimal("-0.0018")
    assert long_one.carry_usd == Decimal("1.82")
    assert long_one.carry_pct == Decimal("0.0182")
    assert _breakdown(side=Side.LONG, nights=3).carry_usd == Decimal("5.46")
    assert _breakdown(side=Side.SHORT, nights=3).carry_usd == Decimal("-0.54")
    for side in (Side.SHORT, Side.LONG):
        assert _breakdown(side=side, nights=0).carry_usd == Decimal("0")
        assert _breakdown(side=side, nights=0).carry_pct == Decimal("0")
    assert short_one.carry_pct_per_night is CARRY_SHORT_PCT_PER_NIGHT
    assert long_one.carry_pct_per_night is CARRY_LONG_PCT_PER_NIGHT


def test_a12_the_sign_convention_is_published() -> None:
    carry = costs.report_payload()["carry"]
    assert carry["sign_convention"] == "negativo = el lado **cobra**; positivo = el lado **paga**"
    assert carry["long_usd_per_night"] == "1.82"
    assert carry["short_usd_per_night"] == "-0.18"
    assert carry["cross_check_against_8"]["matches_declared"] == "true"
    assert Decimal("1.82") == CARRY_LONG_USD_PER_NIGHT
    assert Decimal("-0.18") == CARRY_SHORT_USD_PER_NIGHT


# ─────────────────────────────────────────────────────────────────────────────
# A13, A14, A15 — intradía puro por defecto, corte declarado y nunca derivado
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_zero_nights_does_not_charge_carry_and_explains_why() -> None:
    breakdown = _breakdown(nights=0)
    assert breakdown.nights == 0
    assert breakdown.carry_pct == Decimal("0")
    assert breakdown.carry_state is MeasureState.MEASURED
    assert breakdown.carry_source
    assert "regla 6" in breakdown.carry_reason
    assert "plan.md §12" in breakdown.carry_reason


def test_a13_negative_nights_raise() -> None:
    with pytest.raises(costs.CostInputError, match="nights") as excinfo:
        _breakdown(nights=-1)
    assert "no puede ser negativo" in str(excinfo.value)


def test_a13_a_night_needs_an_explicit_reason() -> None:
    with pytest.raises(costs.CostInputError, match="overnight_reason") as excinfo:
        _breakdown(nights=1, overnight_reason=None)
    message = str(excinfo.value)
    assert "plan.md §12" in message
    assert "regla 6" in message
    assert "regla 16" in message
    with pytest.raises(costs.CostInputError, match="overnight_reason"):
        _breakdown(nights=1, overnight_reason="   ")


def test_a13_the_overnight_reason_is_published() -> None:
    breakdown = _breakdown(nights=1, overnight_reason="cierre manual no ejecutado")
    assert breakdown.overnight_reason == "cierre manual no ejecutado"
    assert "cierre manual no ejecutado" in breakdown.carry_reason
    assert (
        costs.report_payload(nights=2, overnight_reason="dos noches declaradas")["scenario"][
            "overnight_reason"
        ]
        == "dos noches declaradas"
    )


def test_a14_the_financing_cut_is_unverified_today() -> None:
    cut = costs.FinancingCut.unverified()
    assert cut.state is MeasureState.UNMEASURED
    assert cut.cut_et is None
    assert cut.source
    assert "sin verificar" in cut.reason
    assert cut.broker_question == _broker_question()


def test_a14_the_cut_of_8_is_imported_and_still_none() -> None:
    assert DECLARED_FINANCING_CUT is None
    assert cost_audit.DECLARED_FINANCING_CUT is None
    assert costs.DECLARED_FINANCING_CUT is cost_audit.DECLARED_FINANCING_CUT
    assert _breakdown().financing_cut.state is MeasureState.UNMEASURED


def test_a14_no_cut_hour_and_no_clock_in_the_module() -> None:
    for needle in (
        "16:00",
        "time(16",
        "timedelta(hours=16)",
        "15:45",
        "17:00",
        "datetime.now",
        "date.today",
        "time.time",
        "ZoneInfo",
    ):
        assert needle not in SOURCE, needle


def test_a15_a_declared_cut_is_representable() -> None:
    cut = costs.FinancingCut.verified(cut_et=time(17, 0), source="contestado por el bróker (#59)")
    assert cut.state is MeasureState.MEASURED
    assert cut.cut_et == time(17, 0)
    payload = costs.report_payload(financing_cut=cut)
    assert payload["financing_cut"]["state"] == "measured"
    assert payload["financing_cut"]["cut_et"] == "17:00:00"
    breakdown = _breakdown(financing_cut=cut)
    assert breakdown.financing_cut.cut_et == time(17, 0)


def test_a15_the_default_cut_is_published_as_unmeasured() -> None:
    payload = costs.report_payload()
    assert payload["financing_cut"]["cut_et"] is None
    assert payload["financing_cut"]["state"] == "unmeasured"
    assert payload["financing_cut"]["sixteen_is_not_assumed"] == (
        "asumir una hora de corte fija está prohibido"
    )
    assert "#12" in payload["financing_cut"]["note"]
    assert "#13" in payload["financing_cut"]["note"]


def test_a15_nights_are_never_derived_from_an_instant() -> None:
    """`nights` es siempre una entrada: ninguna función mezcla un instante con noches."""
    instants = {"time", "datetime", "date"}
    for node in ast.walk(ast.parse(SOURCE)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = [*node.args.args, *node.args.kwonlyargs]
        annotations = {
            ast.unparse(argument.annotation).split(" |")[0]
            for argument in arguments
            if argument.annotation is not None
        }
        takes_nights = any(argument.arg == "nights" for argument in arguments)
        assert not (takes_nights and annotations & instants), node.name


# ─────────────────────────────────────────────────────────────────────────────
# A16, A17, A18, A19, A20, A21 — el *slippage*
# ─────────────────────────────────────────────────────────────────────────────
def _assumed_payload() -> dict[str, Any]:
    return {
        "state": "assumed",
        "is_measurement": False,
        "pct_of_r": Decimal("20"),
        "source": "procedencia",
        "reason": "motivo",
        "decided_on": "2026-09-18",
    }


def test_a16_assumed_cannot_be_declared_as_a_measurement() -> None:
    with pytest.raises(costs.CostModelError, match="is_measurement"):
        costs.SlippageParameter.model_validate({**_assumed_payload(), "is_measurement": True})


def test_a16_measured_cannot_be_declared_as_an_assumption() -> None:
    with pytest.raises(costs.CostModelError, match="is_measurement"):
        costs.SlippageParameter.model_validate(
            {
                "state": "measured",
                "is_measurement": False,
                "pct_of_notional": Decimal("0.02"),
                "source": "origen",
                "reason": "motivo",
            }
        )


def test_a16_the_three_states_are_separated_and_nothing_aggregates_them() -> None:
    payload = costs.report_payload()
    blocks = {
        key: payload[key]
        for key in ("slippage_measured", "slippage_assumption", "slippage_unmeasured")
    }
    assert {block["state"] for block in blocks.values()} == {"measured", "assumed", "unmeasured"}
    keys = _keys(payload)
    aggregating = [
        key
        for key in keys
        if "slippage" in key
        and any(
            word in key
            for word in ("total", "merged", "combined", "normalized", "unified", "_sum", "sum_")
        )
    ]
    assert aggregating == []


def test_a16_a_missing_reason_raises() -> None:
    for state in ("measured", "assumed", "unmeasured"):
        with pytest.raises(costs.CostModelError, match="reason"):
            costs.SlippageParameter.model_validate({"state": state, "reason": " "})


def test_a17_unmeasured_with_a_zero_raises() -> None:
    with pytest.raises(costs.CostModelError, match="pct_of_notional"):
        costs.SlippageParameter.model_validate(
            {"state": "unmeasured", "reason": "sin ejecuciones", "pct_of_notional": Decimal("0")}
        )


def test_a17_unmeasured_is_null_and_has_no_zero_in_the_json() -> None:
    unmeasured = costs.SlippageParameter.unmeasured(reason="no hay ninguna ejecución real (#62)")
    assert unmeasured.pct_of_notional is None
    assert unmeasured.reason
    payload = costs.report_payload(slippage=unmeasured)
    block = payload["slippage"]
    assert block["state"] == "unmeasured"
    assert block["pct_of_notional"] is None
    for key, value in block.items():
        assert value != 0, key
        assert value != "0", key


def test_a18_the_declared_assumption_is_the_one_of_64() -> None:
    assumption = costs.declared_slippage_assumption()
    assert assumption.state is MeasureState.ASSUMED
    assert assumption.is_measurement is False
    assert assumption.pct_of_r is SLIPPAGE_ASSUMPTION_PCT_OF_R
    assert assumption.pct_of_r == Decimal("20")
    assert assumption.r_pct is None
    assert assumption.pct_of_notional is None
    assert assumption.source == cost_audit.SLIPPAGE_ASSUMPTION_PROVENANCE
    assert assumption.decided_on == cost_audit.SLIPPAGE_ASSUMPTION_DECIDED_ON
    assert assumption.follow_up_issue == "#60"
    assert costs.SLIPPAGE_ASSUMPTION_PCT_OF_R is cost_audit.SLIPPAGE_ASSUMPTION_PCT_OF_R
    assert costs.R_ILLUSTRATIVE_PCT is cost_audit.R_ILLUSTRATIVE_PCT


def test_a18_no_path_turns_the_assumption_into_a_measurement() -> None:
    assumption = costs.declared_slippage_assumption()
    payload = costs.report_payload()
    assert payload["slippage"]["state"] == "assumed"
    assert payload["slippage"]["is_measurement"] is False
    assert payload["slippage_assumption"]["is_measurement"] is False
    assert payload["slippage_assumption"]["state"] == "assumed"
    measured_block = payload["slippage_measured"]
    assert measured_block["state"] == "measured"
    assert measured_block["is_measurement"] is True
    assert measured_block["available"] is False
    assert measured_block["pct_of_notional"] is None
    assert payload["slippage_unmeasured"]["is_measurement"] is None
    with pytest.raises(costs.CostModelError, match="r_pct"):
        costs.SlippageParameter.model_validate({**_assumed_payload(), "r_pct": Decimal("1")})
    assert assumption.state is MeasureState.ASSUMED


def test_a18_the_payload_publishes_the_block_of_8_verbatim() -> None:
    assert costs.report_payload()["slippage_assumption"] == cost_audit.slippage_assumption_block()


def test_a19_the_assumption_leaves_the_total_null() -> None:
    payload = costs.report_payload()
    assert payload["c_total"]["pct"] is None
    assert payload["c_total"]["usd"] is None
    assert payload["c_declared"]["pct"] == "0.0042"
    assert payload["nulls"], "el hueco del *slippage* tiene que estar en nulls"
    entry = payload["nulls"][0]
    assert entry["field"] == "slippage.pct_of_notional"
    assert entry["state"] == "assumed"
    assert entry["reason"]
    breakdown = _breakdown()
    assert breakdown.c_total_pct is None
    assert breakdown.c_total_usd is None
    assert breakdown.nulls and breakdown.nulls[0]["follow_up_issue"] in {"#60", "#62"}


def test_a19_the_illustrative_equivalence_is_opt_in_and_never_feeds_the_total() -> None:
    without = costs.report_payload()
    with_it = costs.report_payload(r_illustrative_pct=Decimal("1"))
    assert without["illustrative_equivalence"] is None
    equivalence = with_it["illustrative_equivalence"]
    assert equivalence is not None
    assert equivalence["illustrative"] == "true"
    assert equivalence["decision"] == "false"
    assert equivalence["r_issue"] == "#60"
    assert equivalence["pct_of_notional"] == "0.2"
    assert equivalence["bp_of_notional"] == "20"
    assert equivalence["usd_on_notional"] == "20"
    for payload in (without, with_it):
        assert payload["c_total"]["pct"] is None
        assert payload["c_total"]["usd"] is None
    assert _breakdown().illustrative_equivalence is None
    assert _breakdown(r_illustrative_pct=Decimal("1")).illustrative_equivalence is not None


def test_a20_a_measured_slippage_closes_the_total_exactly() -> None:
    measured = costs.SlippageParameter.measured(
        pct_of_notional=Decimal("0.02"), source="prueba de medición", reason="prueba de medición"
    )
    for side in (Side.SHORT, Side.LONG):
        for nights in (0, 1):
            breakdown = _breakdown(slippage=measured, side=side, nights=nights)
            assert breakdown.c_total_pct == breakdown.c_declared_pct + Decimal("0.02")
            assert breakdown.c_total_usd == breakdown.c_declared_usd + Decimal("2")
            assert breakdown.slippage_usd == Decimal("2")
            assert breakdown.nulls == ()
    payload = costs.report_payload(slippage=measured)
    assert payload["nulls"] == []
    assert payload["c_total"]["pct"] == "0.0242"
    assert payload["c_total"]["usd"] == "2.42"


def test_a21_the_slippage_is_the_dominant_term() -> None:
    big = costs.SlippageParameter.measured(
        pct_of_notional=Decimal("0.2"), source="origen", reason="motivo"
    )
    breakdown = _breakdown(slippage=big)
    assert breakdown.slippage_over_spread_ratio == Decimal("47.6190")
    assert breakdown.slippage_over_spread_ratio == (Decimal("0.2") / SPREAD_PCT).quantize(
        Decimal("0.0001")
    )
    assert breakdown.slippage_dominates is True
    payload = costs.report_payload(slippage=big)
    assert payload["slippage_dominance"]["slippage_over_spread_ratio"] == "47.619"
    assert payload["slippage_dominance"]["slippage_dominates"] is True


def test_a21_a_slippage_below_the_spread_does_not_dominate() -> None:
    small = costs.SlippageParameter.measured(
        pct_of_notional=Decimal("0.002"), source="origen", reason="motivo"
    )
    breakdown = _breakdown(slippage=small)
    ratio = breakdown.slippage_over_spread_ratio
    assert ratio is not None
    assert ratio < Decimal("1")
    assert breakdown.slippage_dominates is False


def test_a21_without_a_number_the_ratio_is_null_not_zero() -> None:
    breakdown = _breakdown()
    assert breakdown.slippage_over_spread_ratio is None
    assert breakdown.slippage_dominates is None
    payload = costs.report_payload()
    assert payload["slippage_dominance"]["slippage_over_spread_ratio"] is None
    assert payload["slippage_dominance"]["slippage_dominates"] is None


def test_a21_the_report_quotes_the_declared_justification() -> None:
    justification = costs.report_payload()["slippage_dominance"]["justification"]
    assert "plan.md §4.4" in justification
    assert "cincuenta veces" in justification
    assert "declarada" in justification


# ─────────────────────────────────────────────────────────────────────────────
# A22, A23, A24 — nocional, pureza y determinismo
# ─────────────────────────────────────────────────────────────────────────────
def test_a22_the_cost_scales_with_the_notional() -> None:
    small = _breakdown(nights=1, notional_usd=Decimal("10000"))
    big = _breakdown(nights=1, notional_usd=Decimal("20000"))
    assert small.c_declared_pct == big.c_declared_pct
    assert big.c_declared_usd == small.c_declared_usd * 2
    assert small.c_declared_usd == Decimal("2.24")
    assert big.c_declared_usd == Decimal("4.48")
    assert big.spread_usd == small.spread_usd * 2
    assert big.carry_usd == small.carry_usd * 2


def test_a22_the_notional_must_be_positive() -> None:
    for value in (Decimal("0"), Decimal("-1")):
        with pytest.raises(costs.CostInputError, match="notional_usd") as excinfo:
            _breakdown(notional_usd=value)
        assert "mayor que 0" in str(excinfo.value)
    with pytest.raises(costs.CostInputError, match="notional_usd"):
        _breakdown(notional_usd=0.5)  # pyright: ignore[reportArgumentType]
    with pytest.raises(costs.CostInputError, match="notional_usd"):
        _breakdown(notional_usd="10000")  # pyright: ignore[reportArgumentType]


def test_a22_the_module_does_not_do_sizing() -> None:
    assert not [
        name
        for name in costs.__all__
        if any(
            word in name.lower()
            for word in ("sizing", "leverage", "riesgo", "apalancamiento", "stop")
        )
    ]
    assert not (SIZING_NAMES & _assigned_names(SOURCE))
    assigned = _assigned_names(SOURCE)
    assert "notional_usd" not in assigned
    assert "nocional" not in {name.lower() for name in costs.__all__}


def test_a23_the_direct_imports_are_limited() -> None:
    allowed_third_party = {"pydantic", "loguru"}
    allowed_first_party = {"cfdtrader.analysis.cost_audit"}
    modules: set[str] = set()
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    for module in modules:
        root = module.split(".")[0]
        assert (
            root in sys.stdlib_module_names
            or root in allowed_third_party
            or module in allowed_first_party
        ), module


def test_a23_there_is_no_io_network_or_clock_in_the_module() -> None:
    for needle in (
        "httpx",
        "requests",
        "urllib",
        "socket",
        "duckdb",
        "polars",
        "cfdtrader.data.store",
        "cfdtrader.data.calendar",
        "cfdtrader.data.market",
        "Path.read_text",
        "Path.open",
        "open(",
        "datetime.now",
        "date.today",
        "time.time",
    ):
        assert needle not in SOURCE, needle


def test_a23_the_breakdown_runs_in_memory() -> None:
    """Sin `tmp_path`: el motor no toca el disco para calcular un coste."""
    breakdown = _breakdown(side=Side.SHORT, nights=1)
    assert breakdown.c_declared_usd == Decimal("0.24")
    assert not _stray_costs_files()


def test_a24_two_calls_are_identical_field_by_field_and_byte_by_byte() -> None:
    first = _breakdown(side=Side.LONG, nights=1)
    second = _breakdown(side=Side.LONG, nights=1)
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    for field in costs.CostBreakdown.model_fields:
        assert getattr(first, field) == getattr(second, field)


def test_a24_the_payload_hash_is_stable_across_calls() -> None:
    first = costs.report_payload()
    second = costs.report_payload()
    assert _payload_hash(first) == _payload_hash(second)
    assert (
        _payload_hash(first)
        == hashlib.sha256(
            json.dumps(first, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
    )
    markdown_first = costs.render_markdown(first)
    assert markdown_first == costs.render_markdown(second)


# ─────────────────────────────────────────────────────────────────────────────
# A25, A26, A27, A28 — el informe
# ─────────────────────────────────────────────────────────────────────────────
def test_a25_the_report_declares_what_the_module_does_not_do() -> None:
    payload = costs.report_payload()
    items = payload["does_not_do"]
    assert len(items) == 6
    assert {item["issue"] for item in items} >= {"#8", "#13", "#9", "#60", "#27"}
    statements = " ".join(item["statement"] for item in items)
    for needle in (
        "config/cost_observations.yaml",
        "point-in-time",
        "bid",
        "gap",
        "R",
        "sizing",
        "almacén",
        "p*",
    ):
        assert needle in statements, needle
    assert "no lee el almacén" in statements


def test_a26_the_units_are_the_ones_the_gate_consumes() -> None:
    units = costs.report_payload()["units"]
    assert units["c_pct_of_notional"] == "0.0042"
    assert units["c_fraction_of_notional"] == "0.000042"
    assert "#9" in units["note"]
    assert "(R + c) / 2R" in units["note"]
    assert "fracción" in units["fraction_of_notional"]
    serialized = json.dumps(costs.report_payload(), ensure_ascii=False)
    assert "p_star" not in serialized
    assert "breakeven" not in serialized
    keys = _keys(costs.report_payload())
    assert "p_star" not in keys
    assert "breakeven" not in keys


def test_a27_the_report_has_every_required_block() -> None:
    payload = costs.report_payload()
    for key in (
        "declared_table",
        "spread",
        "carry",
        "fx",
        "commission",
        "financing_cut",
        "slippage",
        "slippage_assumption",
        "slippage_measured",
        "slippage_unmeasured",
        "nulls",
        "limitations",
        "units",
        "does_not_do",
    ):
        assert key in payload, key
    rows = payload["declared_table"]["scenarios"]
    assert {row["direction"] for row in rows} == {"long", "short"}
    assert {"0", "1"} <= {row["nights"] for row in rows}
    round_trip = payload["declared_table"]["round_trip"]
    assert round_trip["intraday_pure"]["usd"] == "0.42"
    assert round_trip["intraday_pure"]["pct"] == "0.0042"
    assert round_trip["with_one_night"]["short"]["usd"] == "0.24"
    assert round_trip["with_one_night"]["short"]["pct"] == "0.0024"
    assert round_trip["with_one_night"]["long"]["usd"] == "2.24"
    assert round_trip["with_one_night"]["long"]["pct"] == "0.0224"
    assert payload["fx"]["reason"]
    assert payload["financing_cut"]["state"] == "unmeasured"
    assert isinstance(json.loads(json.dumps(payload, ensure_ascii=False)), dict)


def test_a27_the_amounts_are_exact_decimal_strings() -> None:
    payload = costs.report_payload()
    for path, value in _leaves(payload):
        if path.startswith(".slippage_assumption"):
            # Bloque de #8 publicado **tal cual** (A18: es una sola definición): su evidencia
            # medida lleva sus propios números y no es un importe de este motor.
            continue
        assert value is None or isinstance(value, (str, bool)), f"{path}: {value!r}"
    for path, value in _leaves(payload):
        key = path.rsplit(".", 1)[-1].split("[")[0]
        if isinstance(value, str) and key in QUANTITY_KEYS:
            assert "e" not in value and "E" not in value, f"{path}: {value}"


def test_a27_the_exception_of_block_8_is_exactly_that_block() -> None:
    payload = costs.report_payload()
    assert payload["slippage_assumption"] == cost_audit.slippage_assumption_block()
    assert payload["slippage_assumption"]["state"] == "assumed"


def test_a28_the_markdown_has_the_declared_table_and_the_three_states() -> None:
    markdown = costs.render_markdown()
    for amount in ("0.42", "0.24", "2.24", "0.0042", "0.0024", "0.0224"):
        assert amount in markdown, amount
    for section in (
        "plan.md §3.3",
        "#8",
        "Las dos mitades del diferencial",
        "Tenencia",
        "Divisa, comisión y corte de financiación",
        "Los tres estados del *slippage*",
        "Dominancia del *slippage*",
        "Limitaciones",
        "Seguimientos abiertos",
        "Unidades que consume #9",
    ):
        assert section in markdown, section
    for state in ("measured", "assumed", "unmeasured"):
        assert f"`{state}`" in markdown
    for limitation in ("59 sesiones", "5 min", "^GSPC", "no** sobre el CFD"):
        assert limitation in markdown, limitation
    for issue in ("#50", "#59", "#60", "#62", "#65", "#66"):
        assert issue in markdown, issue


def test_a28_the_markdown_and_the_json_say_the_same_numbers() -> None:
    payload = costs.report_payload()
    markdown = costs.render_markdown(payload)
    for row in payload["declared_table"]["scenarios"]:
        assert row["c_declared"]["usd"] in markdown
        assert row["c_declared"]["pct"] in markdown
    assert payload["c_total"]["usd"] is None
    assert "`null`" in markdown
    assert costs.render_markdown(payload) == costs.render_markdown()


# ─────────────────────────────────────────────────────────────────────────────
# A29, A30 — la CLI y los casos calculados a mano
# ─────────────────────────────────────────────────────────────────────────────
def test_a29_the_cli_prints_without_writing_anything(capsys: pytest.CaptureFixture[str]) -> None:
    before = _stray_costs_files()
    assert costs.main([]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    for needle in (
        "0.42",
        "0.24",
        "2.24",
        "0.0042",
        "0.0024",
        "0.0224",
        "assumed",
        "measured",
        "unmeasured",
    ):
        assert needle in captured.out, needle
    assert _stray_costs_files() == before


def test_a29_the_cli_writes_only_inside_out_dir(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    before = _stray_costs_files()
    out_dir = tmp_path / "informes"
    assert costs.main(["--out-dir", str(out_dir), "--as-of", "2026-09-18"]) == 0
    capsys.readouterr()
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "costs_2026-09-18.json",
        "costs_2026-09-18.md",
    ]
    assert _stray_costs_files() == before
    payload = json.loads((out_dir / "costs_2026-09-18.json").read_text(encoding="utf-8"))
    assert payload["declared_table"]["round_trip"]["intraday_pure"]["usd"] == "0.42"
    assert (out_dir / "costs_2026-09-18.md").read_text(encoding="utf-8") == (
        costs.render_markdown(payload) + "\n"
    )
    assert costs.main(["--out-dir", str(out_dir), "--as-of", "2026-09-18"]) == 0
    assert (out_dir / "costs_2026-09-18.json").read_text(encoding="utf-8") == json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"


def test_a29_the_cli_returns_2_without_writing_on_invalid_input(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out_dir = tmp_path / "informes"
    base = ["--out-dir", str(out_dir), "--as-of", "2026-09-18"]
    cases = (
        ([*base, "--nights", "-1"], "nights"),
        ([*base, "--notional-usd", "0"], "notional_usd"),
        ([*base, "--nights", "1"], "overnight_reason"),
        ([*base, "--slippage-bp", "20"], "término numérico"),
        ([*base, "--slippage-state", "measured"], "slippage"),
        (["--out-dir", str(out_dir)], "--as-of"),
        ([*base, "--side", "lateral"], None),
    )
    for argv, needle in cases:
        if needle is None:
            with pytest.raises(SystemExit):
                costs.main(argv)
            capsys.readouterr()
            continue
        assert costs.main(argv) == 2
        captured = capsys.readouterr()
        assert needle in captured.err, (needle, captured.err)
        assert not out_dir.exists()


def test_a29_the_cli_accepts_a_declared_measured_term(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out_dir = tmp_path / "informes"
    argv = [
        "--out-dir",
        str(out_dir),
        "--as-of",
        "2026-09-18",
        "--slippage-state",
        "measured",
        "--slippage-bp",
        "20",
        "--side",
        "long",
    ]
    assert costs.main(argv) == 0
    capsys.readouterr()
    payload = json.loads((out_dir / "costs_2026-09-18.json").read_text(encoding="utf-8"))
    assert payload["slippage"]["state"] == "measured"
    assert payload["slippage"]["pct_of_notional"] == "0.2"
    assert payload["nulls"] == []
    assert payload["c_total"]["pct"] == "0.2042"


def test_a30_the_hand_computed_table_is_the_declared_one() -> None:
    """Los números de A7/A8/A12/A20/A22, calculados a mano: 0,42 / 0,24 / 2,24 USD."""
    assert Decimal("0.21") + Decimal("0.21") == Decimal("0.42")
    assert Decimal("0.42") + Decimal("-0.18") == Decimal("0.24")
    assert Decimal("0.42") + Decimal("1.82") == Decimal("2.24")
    assert Decimal("1.82") * 3 == Decimal("5.46")
    assert Decimal("0.42") + Decimal("5.46") == Decimal("5.88")
    assert Decimal("0.0042") + Decimal("-0.0018") == Decimal("0.0024")
    assert Decimal("0.0042") + Decimal("0.0182") == Decimal("0.0224")
    assert _breakdown(side=Side.LONG, nights=3).c_declared_usd == Decimal("5.88")
    assert Decimal(10000) * Decimal("0.0224") / Decimal(100) == Decimal("2.24")


def test_a30_an_error_per_declared_condition_with_its_type_and_message() -> None:
    with pytest.raises(costs.CostModelError) as unvalidated:
        _breakdown(model=costs.CostModel.model_construct())
    assert "model_construct" in str(unvalidated.value)

    with pytest.raises(costs.CostInputError) as negative:
        _breakdown(nights=-2)
    assert "nights" in str(negative.value)

    with pytest.raises(costs.CostInputError) as overnight:
        _breakdown(nights=1, overnight_reason="")
    assert "overnight_reason" in str(overnight.value)

    with pytest.raises(costs.CostModelError) as zero_slippage:
        costs.SlippageParameter.model_validate(
            {"state": "unmeasured", "reason": "sin ejecuciones", "pct_of_notional": Decimal("0")}
        )
    assert "null" in str(zero_slippage.value)

    with pytest.raises(costs.CostInputError) as notional:
        _breakdown(notional_usd=Decimal("-0.01"))
    assert "notional_usd" in str(notional.value)


# ─────────────────────────────────────────────────────────────────────────────
# A32, A33, A34 — convenciones, modelos y limitaciones
# ─────────────────────────────────────────────────────────────────────────────
def test_a32_repository_conventions() -> None:
    assert SOURCE.startswith('"""')
    assert "from __future__ import annotations" in SOURCE
    assert "float(" not in SOURCE
    assert "import numpy" not in SOURCE
    assert "import polars" not in SOURCE
    assert SOURCE.count("Final[") >= 10
    assert "argparse" in SOURCE
    unannotated = {
        target.id
        for node in ast.parse(SOURCE).body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert unannotated <= {"__all__"}
    logger_uses = [
        node.lineno
        for node in ast.walk(ast.parse(SOURCE))
        if isinstance(node, ast.Name) and node.id == "logger"
    ]
    main_node = next(
        node
        for node in ast.parse(SOURCE).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assert logger_uses
    assert all(
        main_node.lineno <= line <= (main_node.end_lineno or main_node.lineno)
        for line in logger_uses
    )


def test_a33_the_four_models_are_frozen_and_forbid_extra() -> None:
    for model in (
        costs.CostModel,
        costs.CostBreakdown,
        costs.SlippageParameter,
        costs.FinancingCut,
    ):
        assert model.model_config.get("frozen") is True
        assert model.model_config.get("extra") == "forbid"
    breakdown = _breakdown()
    with pytest.raises((ValidationError, TypeError)):
        breakdown.nights = 4  # type: ignore[misc]
    with pytest.raises(ValidationError):
        costs.CostModel.model_validate({**_declared_fields(), "campo_desconocido": Decimal("1")})
    with pytest.raises(ValidationError):
        costs.SlippageParameter.model_validate(
            {"state": "unmeasured", "reason": "sin ejecuciones", "campo_desconocido": "x"}
        )


def test_a33_the_sides_and_states_are_the_ones_of_8() -> None:
    assert costs.Side is cost_audit.Side
    assert costs.MeasureState is cost_audit.MeasureState
    assert costs.Side.SHORT is cost_audit.Side.SHORT
    assert costs.Side.LONG is cost_audit.Side.LONG
    assert costs.MeasureState.ASSUMED is cost_audit.MeasureState.ASSUMED
    assert "class Side" not in SOURCE
    assert "class MeasureState" not in SOURCE
    assert "StrEnum" not in SOURCE


def test_a34_the_limitations_are_published_in_json_and_markdown() -> None:
    payload = costs.report_payload()
    markdown = costs.render_markdown(payload)
    limitations = " ".join(payload["limitations"])
    assert len(payload["limitations"]) >= 6
    for needle in (
        "§3.3",
        "#8",
        "#59",
        "cut_et: null",
        'state: "unmeasured"',
        "asumir una hora de corte fija está prohibido",
        "20 % de `R`",
        "is_measurement: false",
        "#62",
        "#60",
        "#66",
        "#50",
        "#52",
        "#65",
    ):
        assert needle in limitations, needle
    for needle in ("#66", "#50", "#52", "#59", "#60", "#62", "#65"):
        assert needle in markdown, needle
    assert all(item["issue"] in markdown for item in payload["follow_ups"])


def _broker_question() -> str:
    """La pregunta literal al bróker, tal cual la publica #8."""
    return next(
        item["question"] for item in cost_audit.BROKER_QUESTIONS if item["id"] == "financing_cut"
    )
