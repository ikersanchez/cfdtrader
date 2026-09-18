"""Tests de la auditoría de costes declarados del ``SPX500:CFD`` (tarea #8).

Los números esperados se calculan **a mano** en el propio test: el diferencial
declarado (0,42 $ / 0,0042 %), los totales de ida y vuelta (0,24 $ corto, 2,24 $
largo) y el caso del spread por tramo (1,20 $ ⇒ 0,0120 %, 0,78 $ por encima del
declarado). Ningún valor esperado se copia de la salida del código.

Todo lo que escribe, escribe en ``tmp_path``: la fixture de sesión de
``tests/conftest.py`` huella el ``data/`` del repositorio y no puede cambiar.
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import pytest
import yaml
from pydantic import ValidationError

from cfdtrader.analysis import cost_audit
from cfdtrader.analysis.cost_audit import (
    ANNUALISATION_TOLERANCE,
    BROKER_QUESTIONS,
    CARRY_LONG_PCT_PER_NIGHT,
    CARRY_LONG_USD_PER_NIGHT,
    CARRY_SHORT_PCT_PER_NIGHT,
    CARRY_SHORT_USD_PER_NIGHT,
    DEFAULT_TEMPLATE_PATH,
    HOLDING_NIGHTS,
    LIMITATIONS,
    PAIR_TOLERANCE_SECONDS,
    REFERENCE_NOTIONAL_USD,
    SESSION_TRANCHES,
    SIZE_LADDER_USD,
    SPREAD_PCT,
    SPREAD_USD,
    CostAudit,
    CostObservations,
    CostTemplateError,
    FxCost,
    MeasureState,
    consolidate,
    load_template,
    main,
    render_markdown,
    session_tranches,
    tranche_of,
)
from cfdtrader.data.calendar import MarketCalendar, load_calendar

#: Instante de referencia de los tests (medianoche UTC del día auditado).
NOW = datetime(2026, 9, 18, tzinfo=UTC)

#: Sesión auditada: viernes 2026-09-18, sesión completa.
SESSION_DAY = date(2026, 9, 18)

#: Media sesión conocida: el día después de Acción de Gracias de 2026.
HALF_DAY = date(2026, 11, 27)

#: Nombres de los cinco tramos, en orden.
TRANCHE_NAMES = [spec.name for spec in SESSION_TRANCHES]

#: Claves que llevan el valor de una medida (las que vigila A24).
VALUE_KEYS = ("value_usd", "value_pct", "value_points", "value_utc")


#: Raíz del repositorio y su ``data/``, para vigilar que la suite no lo toca (A26).
REPO_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_DATA = REPO_ROOT / "data"


def _fingerprint(root: Path) -> dict[str, str]:
    """Ruta relativa → sha256, la misma huella que usa la fixture de ``conftest.py``."""
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _payload(**overrides: object) -> dict[str, object]:
    """Plantilla mínima válida (equivalente a la vacía) con lo que se quiera cambiar."""
    base: dict[str, object] = {
        "trading_window": None,
        "settlement_currency": None,
        "financing_cut": None,
        "minimum_commission_usd": None,
        "spread_observations": [],
        "tracking_pairs": [],
        "executions": [],
    }
    base.update(overrides)
    return base


def _observations(**overrides: object) -> CostObservations:
    return CostObservations.model_validate(_payload(**overrides))


def _consolidate(
    *, template_path: Path | str = DEFAULT_TEMPLATE_PATH, **overrides: object
) -> CostAudit:
    """Consolida en memoria: modelo validado dentro, informe fuera (sin tocar disco)."""
    return consolidate(
        _observations(**overrides),
        calendar=load_calendar(),
        now=NOW,
        template_path=template_path,
    )


def _spread(
    *, hour_et: int, minute_et: int, bid: float, ask: float, notional: float = 10_000.0
) -> dict[str, object]:
    """Observación de spread con el instante expresado en hora ET (como lo anota un humano)."""
    instant = datetime(
        2026, 9, 18, hour_et, minute_et, tzinfo=ZoneInfo("America/New_York")
    ).astimezone(UTC)
    return {
        "timestamp_utc": instant.isoformat(),
        "notional_usd": notional,
        "bid": bid,
        "ask": ask,
    }


def _walk(node: object, key: str = "") -> list[tuple[str, object]]:
    """(clave, valor) de todo el árbol, conservando la clave que lo contiene."""
    found: list[tuple[str, object]] = []
    if isinstance(node, dict):
        for child_key, child in cast("dict[object, object]", node).items():
            found.extend(_walk(child, str(child_key)))
    elif isinstance(node, list):
        for item in cast("list[object]", node):
            found.extend(_walk(item, key))
    else:
        found.append((key, node))
    return found


def _numeric(raw: object) -> Decimal | None:
    """El valor como ``Decimal`` si es un número (o una cadena numérica)."""
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        return Decimal(str(raw))
    except (InvalidOperation, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# A1 · La ventana está confirmada y se registra con su procedencia
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_confirmed_window_is_recorded_with_provenance() -> None:
    audit = _consolidate()
    window = audit.payload["confirmed_inputs"]["trading_window"]

    assert window["state"] == "confirmed"
    assert window["source"] == "declaración del usuario"
    assert window["confirmed_on"] == "2026-09-18"
    assert window["reference_timezone"] == "America/New_York"
    assert window["value"] == "sesión regular del S&P 500"
    assert "unverified" not in json.dumps(window)
    assert "pending" not in json.dumps(window)

    text = render_markdown(audit)
    assert "sesión regular del S&P 500" in text
    assert "declaración del usuario" in text


# ─────────────────────────────────────────────────────────────────────────────
# A2 · Las fronteras salen del calendario y aguantan el cambio de hora
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_session_borders_come_from_the_calendar_across_dst() -> None:
    calendar = load_calendar()
    before_dst = date(2026, 3, 6)  # viernes, horario de invierno: ET -05:00
    after_dst = date(2026, 3, 9)  # lunes, horario de verano: ET -04:00

    open_before = calendar.session(before_dst).open_utc
    open_after = calendar.session(after_dst).open_utc
    assert open_before is not None and open_after is not None
    assert open_before.utcoffset() == timedelta(0)
    # El instante UTC de la apertura se desplaza exactamente 1 h manteniendo las 09:30 ET:
    # 14:30 UTC con ET -05:00 frente a 13:30 UTC con ET -04:00.
    assert (open_before.hour, open_before.minute) == (14, 30)
    assert (open_after.hour, open_after.minute) == (13, 30)
    shift = timedelta(hours=open_before.hour, minutes=open_before.minute) - timedelta(
        hours=open_after.hour, minutes=open_after.minute
    )
    assert shift == timedelta(hours=1)
    assert MarketCalendar.to_et(open_before).utcoffset() == timedelta(hours=-5)
    assert MarketCalendar.to_et(open_after).utcoffset() == timedelta(hours=-4)

    # La franja asignada a la misma hora local ET no cambia con el DST.
    for day, hour in ((before_dst, 11), (after_dst, 11)):
        local = datetime(day.year, day.month, day.day, hour, tzinfo=ZoneInfo("America/New_York"))
        tranches = session_tranches(calendar, day)
        assert tranche_of(local.astimezone(UTC), tranches) == "mediodia"

    assert session_tranches(calendar, before_dst)[2].label_et == "11:00 ET -05:00"
    assert session_tranches(calendar, after_dst)[2].label_et == "11:00 ET -04:00"


def test_a2_no_literal_clock_times_in_the_module() -> None:
    source = Path(cost_audit.__file__).read_text(encoding="utf-8")
    for literal in ("09:30", "16:00", "13:00", "14:30", "22:00", "09:20", "15:45"):
        assert literal not in source, f"el módulo no puede fijar la hora {literal}"


# ─────────────────────────────────────────────────────────────────────────────
# A3 · Divisa de liquidación: un cero **con motivo**, nunca un cero silencioso
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_fx_cost_is_zero_with_origin() -> None:
    payload = _consolidate().payload
    fx = payload["fx_cost"]

    assert fx["state"] == MeasureState.MEASURED.value
    assert Decimal(fx["value_usd"]) == 0
    assert Decimal(fx["value_pct"]) == 0
    assert fx["source"] and fx["reason"]
    assert "USD" in fx["reason"]
    assert payload["confirmed_inputs"]["settlement_currency"]["value"] == "USD"
    # la exposición de divisa es otra cosa y se declara fuera de alcance
    assert "#27" in fx["note"]


def test_a3_fx_cost_without_reason_does_not_validate() -> None:
    without_reason: dict[str, object] = {
        "value_usd": Decimal("0"),
        "value_pct": Decimal("0"),
        "source": "declaración del usuario",
    }
    with pytest.raises(ValidationError):
        FxCost.model_validate(without_reason)
    with pytest.raises(ValidationError):
        FxCost.model_validate({**without_reason, "reason": ""})


def test_a3_a_currency_other_than_usd_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CostObservations.model_validate(_payload(settlement_currency="EUR"))


# ─────────────────────────────────────────────────────────────────────────────
# A4 · El corte de financiación sigue sin verificar y no tiene valor por defecto
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_financing_cut_is_unverified_with_the_literal_question() -> None:
    cut = _consolidate().payload["financing_cut"]

    assert cut["state"] == "unverified"
    assert cut["value_utc"] is None
    assert cut["value_et"] is None
    assert cut["reason"]

    question = cut["broker_question"]
    assert question == cost_audit.FINANCING_CUT_QUESTION
    for fragment in (
        "instante exacto",
        "swap",
        "SPX500:CFD",
        "America/New_York",
        "noche de calendario o por sesión",
    ):
        assert fragment in question


def test_a4_there_is_no_default_for_the_cut() -> None:
    assert cost_audit.DECLARED_FINANCING_CUT is None
    assert CostObservations.model_fields["financing_cut"].is_required()
    # Una plantilla sin el campo no es válida: no hay valor por defecto que lo cubra.
    incomplete = {key: value for key, value in _payload().items() if key != "financing_cut"}
    with pytest.raises(ValidationError):
        CostObservations.model_validate(incomplete)
    # el informe se escribe igualmente con el corte a null (A20 lo prueba de punta a punta)
    assert _consolidate().payload["financing_cut"]["state"] == "unverified"


# ─────────────────────────────────────────────────────────────────────────────
# A5 · El slippage no está medido y no se rellena con nada
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_slippage_is_unmeasured_with_how_to_fill() -> None:
    block = _consolidate().payload["slippage_ejecucion"]

    assert block["state"] == MeasureState.UNMEASURED.value
    assert block["value_usd"] is None
    assert block["value_pct"] is None
    assert block["value_points"] is None
    assert "no existe ninguna ejecución real" in block["reason"]

    how_to_fill = block["how_to_fill"]
    assert "precio de referencia" in how_to_fill
    assert "10–15 veces" in how_to_fill
    assert "apertura" in how_to_fill
    assert "relleno" in block["forbidden"]


def test_a5_a_filled_execution_is_measured_with_the_declared_sign_convention() -> None:
    execution = {
        "timestamp_utc": "2026-09-18T13:30:05+00:00",
        "side": "long",
        "notional_usd": 10_000,
        "reference_price": 5000.0,
        "filled_price": 5001.0,  # 0,02 % por encima de la referencia
    }
    block = _consolidate(executions=[execution]).payload["slippage_ejecucion"]

    assert block["state"] == MeasureState.MEASURED.value
    assert Decimal(block["value_pct"]) == Decimal("0.02")
    assert Decimal(block["value_usd"]) == Decimal("2.00")


# ─────────────────────────────────────────────────────────────────────────────
# A6 · La puerta (b) de #9 no es evaluable y aquí no se emite veredicto
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_phase0_gate_b_is_not_evaluable_and_no_verdict_is_emitted() -> None:
    payload = _consolidate().payload
    gate = payload["phase0_gate_b"]

    assert gate["evaluable"] is False
    assert gate["reason"]
    assert gate["threshold_pct_of_r"] == "20"
    assert gate["verdict_owner"] == "#9"
    assert "verdict" not in payload
    assert not [key for key in payload if "verdict" in key or "continu" in key]


# ─────────────────────────────────────────────────────────────────────────────
# A7, A8, A9, A10 · La tabla declarada
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_declared_constants_with_units_and_reference_notional() -> None:
    declared: dict[str, tuple[object, object]] = {
        "reference_notional_usd": (REFERENCE_NOTIONAL_USD, Decimal("10000")),
        "holding_nights": (HOLDING_NIGHTS, 1),
        "spread_usd": (SPREAD_USD, Decimal("0.42")),
        "spread_pct": (SPREAD_PCT, Decimal("0.0042")),
        "carry_short_usd_per_night": (CARRY_SHORT_USD_PER_NIGHT, Decimal("-0.18")),
        "carry_short_pct_per_night": (CARRY_SHORT_PCT_PER_NIGHT, Decimal("-0.0018")),
        "carry_long_usd_per_night": (CARRY_LONG_USD_PER_NIGHT, Decimal("1.82")),
        "carry_long_pct_per_night": (CARRY_LONG_PCT_PER_NIGHT, Decimal("0.0182")),
    }
    for name, (actual, expected) in declared.items():
        assert actual == expected, f"{name} no coincide con la tabla declarada de §3.3"

    rows = _consolidate().payload["declared_table"]["rows"]
    assert len(rows) == 4
    assert [row["amount"]["usd"] for row in rows] == ["0.42", "-0.18", "1.82", "0"]
    assert [row["amount"]["pct"] for row in rows] == ["0.0042", "-0.0018", "0.0182", "0"]


def test_a8_round_trip_totals_are_exact() -> None:
    round_trip = _consolidate().payload["declared_table"]["round_trip"]
    short = round_trip["short"]["amount"]
    long = round_trip["long"]["amount"]

    assert Decimal(short["usd"]) == Decimal("0.24")
    assert Decimal(short["pct"]) == Decimal("0.0024")
    assert Decimal(long["usd"]) == Decimal("2.24")
    assert Decimal(long["pct"]) == Decimal("0.0224")

    # La comparación es exacta, no «≈»: 0,42 - 0,18 = 0,24 y 0,42 + 1,82 = 2,24.
    assert Decimal("0.42") - Decimal("0.18") == Decimal(short["usd"])
    assert Decimal("0.42") + Decimal("1.82") == Decimal(long["usd"])
    assert round_trip["holding_nights"] == 1


def test_a9_annualised_figures_are_declared_and_not_derived() -> None:
    annual = _consolidate().payload["declared_table"]["annualisation"]

    assert Decimal(annual["short"]["pct"]) == Decimal("-0.6528")
    assert Decimal(annual["long"]["pct"]) == Decimal("6.6647")
    assert annual["derived_from_daily"] is False
    # No se derivan: 365 noches del valor por-noche no dan la cifra declarada.
    assert Decimal("-0.0018") * 365 != Decimal("-0.6528")
    assert Decimal("0.0182") * 365 != Decimal("6.6647")

    ratio_short = Decimal(annual["short"]["ratio_annualised_over_per_night"])
    ratio_long = Decimal(annual["long"]["ratio_annualised_over_per_night"])
    assert ratio_short.quantize(Decimal("0.1")) == Decimal("362.7")
    assert ratio_long.quantize(Decimal("0.1")) == Decimal("366.2")
    assert annual["annualisation_consistent"] is True
    assert Decimal(annual["tolerance"]) == ANNUALISATION_TOLERANCE


def test_a9_an_inconsistent_annualisation_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    # Si las dos ratios se separasen más del 1 % declarado, la marca sería falsa.
    monkeypatch.setattr(cost_audit, "ANNUALISED_LONG_PCT", Decimal("10.0"))
    annual = _consolidate().payload["declared_table"]["annualisation"]

    assert annual["annualisation_consistent"] is False
    assert Decimal(annual["relative_difference"]) > ANNUALISATION_TOLERANCE


def test_a10_no_amount_without_unit_and_notional() -> None:
    declared = _consolidate().payload["declared_table"]
    reference = declared["reference_notional_usd"]

    amounts = [row["amount"] for row in declared["rows"]]
    amounts += [declared["round_trip"][side]["amount"] for side in ("short", "long")]
    for amount in amounts:
        assert amount["usd_unit"]
        assert amount["pct_unit"]
        assert amount["notional_usd"] == reference

    for side in ("short", "long"):
        block = declared["annualised"][side]
        assert block["pct_unit"] == "% anual"
        assert block["notional_usd"] == reference


# ─────────────────────────────────────────────────────────────────────────────
# A11 · Los cinco tramos, relativos a la sesión
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_five_tranches_relative_to_the_session() -> None:
    calendar = load_calendar()
    tranches = session_tranches(calendar, SESSION_DAY)

    assert [tranche.name for tranche in tranches] == TRANCHE_NAMES
    assert [tranche.label_et for tranche in tranches] == [
        "09:20 ET -04:00",
        "09:35 ET -04:00",
        "11:00 ET -04:00",
        "15:45 ET -04:00",
        "16:00 ET -04:00",
    ]
    for tranche in tranches:
        assert tranche.reference_utc.utcoffset() == timedelta(0)
        assert tranche.reference_et.endswith(("-04:00", "-05:00"))


def test_a11_half_days_shift_the_tranches_with_the_close() -> None:
    calendar = load_calendar()
    assert calendar.is_half_day(HALF_DAY)
    labels = [tranche.label_et for tranche in session_tranches(calendar, HALF_DAY)]

    assert labels[:3] == ["09:20 ET -05:00", "09:35 ET -05:00", "11:00 ET -05:00"]
    assert labels[3] == "12:45 ET -05:00"
    assert labels[4] == "13:00 ET -05:00"


def test_a11_observations_outside_the_session_are_declared() -> None:
    observations = [_spread(hour_et=5, minute_et=0, bid=5000.0, ask=5000.42)]
    block = _consolidate(spread_observations=observations).payload["spread_cotizado"]

    assert block["state"] == MeasureState.UNMEASURED.value
    assert block["out_of_session"] == 1
    assert len(block["by_tranche"]) == 5


# ─────────────────────────────────────────────────────────────────────────────
# A12 · Consolidación por tamaño
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_by_size_ladder_is_proportional_without_a_minimum() -> None:
    by_size = _consolidate().payload["by_size"]

    assert by_size["notionals_usd"] == ["1000", "5000", "10000"]
    assert [Decimal(notional) for notional in by_size["notionals_usd"]] == list(SIZE_LADDER_USD)
    assert by_size["size_scaling_anomaly"] is False
    assert by_size["percentage_is_constant"] is True
    for row in by_size["rows"]:
        assert Decimal(row["round_trip_short_pct"]) == Decimal("0.0024")
        assert Decimal(row["round_trip_long_pct"]) == Decimal("0.0224")


def test_a12_a_declared_minimum_commission_marks_the_anomaly() -> None:
    by_size = _consolidate(minimum_commission_usd=Decimal("0.50")).payload["by_size"]

    assert by_size["size_scaling_anomaly"] is True
    assert by_size["percentage_is_constant"] is False
    small = next(row for row in by_size["rows"] if row["notional_usd"] == "1000")
    assert Decimal(small["spread_usd"]) == Decimal("0.50")  # el mínimo manda
    assert Decimal(small["round_trip_short_pct"]) == Decimal("0.0482")
    assert Decimal(by_size["detail"][0]["vs_reference_short_pct"]) > 0
    assert "mínimo absoluto" in by_size["reason"]


# ─────────────────────────────────────────────────────────────────────────────
# A13, A14 · Tres bloques separados, sin total y con null en vez de 0
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_the_three_measures_are_three_blocks_and_nothing_sums_them() -> None:
    payload = _consolidate().payload
    for name in ("spread_cotizado", "tracking_difference", "slippage_ejecucion"):
        assert name in payload
        assert {"state", "value_usd", "value_pct", "reason"} <= set(payload[name])

    forbidden = {"total", "total_cost", "cost_total", "combined_cost", "grand_total", "sum_cost"}
    for key, _ in _walk(payload):
        assert key not in forbidden, f"el informe no puede combinar las tres medidas: {key}"


def test_a14_empty_template_gives_three_unmeasured_and_null_values() -> None:
    payload = _consolidate().payload

    for name in ("spread_cotizado", "tracking_difference", "slippage_ejecucion"):
        block = payload[name]
        assert block["state"] == MeasureState.UNMEASURED.value
        assert block["value_usd"] is None
        assert block["value_pct"] is None
        assert block["value_points"] is None
        assert block["source"] is None
        assert block["reason"]
        assert block["observations"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# A15 · Spread por tramo, con aritmética verificable a mano
# ─────────────────────────────────────────────────────────────────────────────
def test_a15_spread_by_tranche_is_hand_computed() -> None:
    # Fixture: 09:20 ET ⇒ ask - bid = 1,20 $ (0,0120 %); 11:00 ET ⇒ 0,42 $ (0,0042 %).
    observations = [
        _spread(hour_et=9, minute_et=20, bid=5000.0, ask=5001.2),
        _spread(hour_et=11, minute_et=0, bid=5000.0, ask=5000.42),
    ]
    block = _consolidate(spread_observations=observations).payload["spread_cotizado"]

    assert block["state"] == MeasureState.MEASURED.value
    assert block["observations"] == 2
    assert Decimal(block["value_usd"]) == Decimal("0.81")  # (1,20 + 0,42) / 2

    rows = {row["tranche"]: row for row in block["by_tranche"]}
    pre = rows["pre_subasta"]
    assert Decimal(pre["spread_usd"]) == Decimal("1.20")
    assert Decimal(pre["spread_pct"]) == Decimal("0.0120")
    assert Decimal(pre["observed_minus_declared_usd"]) == Decimal("0.78")
    assert Decimal(pre["observed_minus_declared_pct"]) == Decimal("0.0078")
    assert pre["wider_than_declared"] is True
    assert pre["observations"] == 1
    assert pre["timestamps_utc"] == ["2026-09-18T13:20:00+00:00"]

    mid = rows["mediodia"]
    assert Decimal(mid["spread_usd"]) == Decimal("0.42")
    assert Decimal(mid["spread_pct"]) == Decimal("0.0042")
    assert Decimal(mid["observed_minus_declared_usd"]) == Decimal("0.00")
    assert Decimal(mid["observed_minus_declared_pct"]) == Decimal("0.00")
    assert mid["wider_than_declared"] is False

    # Los tramos sin dato siguen apareciendo, en `unmeasured`.
    assert rows["cierre"]["state"] == MeasureState.UNMEASURED.value
    assert len(block["by_tranche"]) == 5


# ─────────────────────────────────────────────────────────────────────────────
# A16 · Tracking difference: solo pares del mismo instante
# ─────────────────────────────────────────────────────────────────────────────
def _pair(*, lag_seconds: int, cfd: float = 5000.5, index: float = 5000.0) -> dict[str, object]:
    instant = datetime(2026, 9, 18, 13, 20, tzinfo=UTC)
    return {
        "series_id": "^GSPC",
        "cfd_timestamp_utc": instant.isoformat(),
        "cfd_price": cfd,
        "index_timestamp_utc": (instant + timedelta(seconds=lag_seconds)).isoformat(),
        "index_price": index,
    }


def test_a16_tracking_difference_uses_paired_instants_only() -> None:
    block = _consolidate(tracking_pairs=[_pair(lag_seconds=0)]).payload["tracking_difference"]

    assert block["state"] == MeasureState.MEASURED.value
    assert block["observations"] == 1
    assert Decimal(block["value_points"]) == Decimal("0.5")
    assert Decimal(block["value_pct"]) == Decimal("0.01")
    assert block["pairs"][0]["in_tolerance"] is True
    assert block["tolerance_seconds"] == PAIR_TOLERANCE_SECONDS


def test_a16_a_pair_beyond_the_tolerance_is_excluded_and_counted() -> None:
    block = _consolidate(tracking_pairs=[_pair(lag_seconds=2)]).payload["tracking_difference"]

    assert block["state"] == MeasureState.UNMEASURED.value
    assert block["value_points"] is None
    assert block["pairs"][0]["in_tolerance"] is False
    assert block["pairs"][0]["lag_seconds"] == 2.0
    assert "tolerancia" in block["reason"]
    assert block["pairs_excluded_out_of_tolerance"] == 1


def test_a16_mixing_pairs_counts_only_the_paired_one() -> None:
    block = _consolidate(tracking_pairs=[_pair(lag_seconds=0), _pair(lag_seconds=2)]).payload[
        "tracking_difference"
    ]

    assert block["observations"] == 1
    assert [pair["in_tolerance"] for pair in block["pairs"]] == [True, False]


# ─────────────────────────────────────────────────────────────────────────────
# A17 · Timestamps en UTC, intervalos en America/New_York
# ─────────────────────────────────────────────────────────────────────────────
def test_a17_timestamps_are_utc_and_intervals_are_et() -> None:
    observations = [_spread(hour_et=9, minute_et=20, bid=5000.0, ask=5001.2)]
    payload = _consolidate(
        spread_observations=observations, tracking_pairs=[_pair(lag_seconds=0)]
    ).payload

    checked_utc = 0
    for key, value in _walk(payload):
        if not isinstance(value, str):
            continue
        if key.endswith("_utc"):
            assert value.endswith(("+00:00", "Z")), f"{key} no está en UTC: {value}"
            assert datetime.fromisoformat(value).tzinfo is not None
            checked_utc += 1
        if key.endswith("_et"):
            assert value.endswith(("-04:00", "-05:00")), f"{key} no está en ET: {value}"
    assert checked_utc >= 10

    labels = [row["label_et"] for row in payload["spread_cotizado"]["by_tranche"]]
    assert all(label.endswith(("-04:00", "-05:00")) for label in labels)


# ─────────────────────────────────────────────────────────────────────────────
# A18, A19 · La plantilla de captura
# ─────────────────────────────────────────────────────────────────────────────
def test_a18_the_committed_template_is_blank_commented_and_strict() -> None:
    text = DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)

    assert json.dumps(sorted(raw)) == json.dumps(
        sorted(
            [
                "trading_window",
                "settlement_currency",
                "financing_cut",
                "minimum_commission_usd",
                "spread_observations",
                "tracking_pairs",
                "executions",
            ]
        )
    )
    for key in ("trading_window", "settlement_currency", "financing_cut", "minimum_commission_usd"):
        assert raw[key] is None, f"{key} debe ir a null en la plantilla commiteada"
    for key in ("spread_observations", "tracking_pairs", "executions"):
        assert raw[key] == [], f"{key} debe ser una lista vacía en la plantilla commiteada"
    assert text.count("\n# ") > 30, "la plantilla debe estar comentada"

    # Un campo desconocido no se ignora en silencio.
    with pytest.raises(ValidationError):
        CostObservations.model_validate(_payload(campo_desconocido=1))
    assert CostObservations.model_config.get("extra") == "forbid"

    # La plantilla vacía valida (es la que hay commiteada).
    assert load_template(DEFAULT_TEMPLATE_PATH).spread_observations == ()


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param({"notional_usd": 10_000, "bid": 5000.0, "ask": 5000.42}, id="sin_timestamp"),
        pytest.param(
            {
                "timestamp_utc": "2026-09-18T13:20:00",
                "notional_usd": 10_000,
                "bid": 5000.0,
                "ask": 5000.42,
            },
            id="timestamp_sin_zona",
        ),
        pytest.param(
            {
                "timestamp_utc": "2026-09-18T13:20:00+00:00",
                "notional_usd": 0,
                "bid": 5000.0,
                "ask": 5000.42,
            },
            id="nocional_cero",
        ),
        pytest.param(
            {
                "timestamp_utc": "2026-09-18T13:20:00+00:00",
                "notional_usd": -100,
                "bid": 5000.0,
                "ask": 5000.42,
            },
            id="nocional_negativo",
        ),
        pytest.param(
            {
                "timestamp_utc": "2026-09-18T13:20:00+00:00",
                "notional_usd": 10_000,
                "bid": 5000.42,
                "ask": 5000.0,
            },
            id="ask_menor_que_bid",
        ),
    ],
)
def test_a19_validation_rejects_bad_observations(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CostObservations.model_validate(_payload(spread_observations=[bad]))


def test_a19_the_errors_are_typed_and_explain_the_field() -> None:
    bad = {
        "timestamp_utc": "2026-09-18T13:20:00",  # sin zona horaria
        "notional_usd": 10_000,
        "bid": 5000.0,
        "ask": 5000.42,
    }
    with pytest.raises(ValidationError) as error:
        CostObservations.model_validate(_payload(spread_observations=[bad]))

    text = str(error.value)
    assert "timestamp_utc" in text
    assert "zona horaria" in text
    assert not isinstance(error.value, AssertionError)


# ─────────────────────────────────────────────────────────────────────────────
# A20, A21, A22, A23 · CLI, pureza, códigos de salida y determinismo
# ─────────────────────────────────────────────────────────────────────────────
def _cli_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--data-root",
        str(tmp_path),
        "--now",
        "2026-09-18T00:00:00+00:00",
        *extra,
    ]


def test_a20_the_empty_template_produces_a_full_report(tmp_path: Path) -> None:
    assert main(_cli_args(tmp_path)) == 0

    reports = tmp_path / "derived" / "reports"
    json_path = reports / "cost_audit_2026-09-18.json"
    markdown_path = reports / "cost_audit_2026-09-18.md"
    assert json_path.is_file()
    assert markdown_path.is_file()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["declared_table"]["round_trip"]["short"]["amount"]["usd"] == "0.24"
    assert payload["declared_table"]["round_trip"]["long"]["amount"]["usd"] == "2.24"
    for name in ("spread_cotizado", "tracking_difference", "slippage_ejecucion"):
        assert payload[name]["state"] == "unmeasured"


def test_a21_the_consolidation_is_pure_and_touches_no_disk(tmp_path: Path) -> None:
    ghost = tmp_path / "no-existe.yaml"
    audit = _consolidate(template_path=ghost)

    assert isinstance(audit, CostAudit)
    assert audit.payload["template_path"] == str(ghost)
    assert list(tmp_path.iterdir()) == [], "consolidar no puede escribir nada"
    assert audit.report_stem == "cost_audit_2026-09-18"


def test_a22_exit_code_2_when_nothing_can_be_consolidated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = tmp_path / "rota.yaml"
    broken.write_text("trading_window: null\nsettlement_currency: null\n", encoding="utf-8")

    assert main(_cli_args(tmp_path, "--template", str(broken))) == 2
    captured = capsys.readouterr()
    assert captured.err.strip()
    assert "no se puede consolidar" in captured.err
    assert not (tmp_path / "derived").exists(), "con exit 2 no se escribe ningún informe"


def test_a22_exit_code_2_for_an_unreadable_template_never_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "no-esta.yaml"
    assert main(_cli_args(tmp_path, "--template", str(missing))) == 2
    assert capsys.readouterr().err.strip()
    assert not (tmp_path / "derived").exists()


def test_a22_exit_code_2_for_a_template_with_a_bad_observation(tmp_path: Path) -> None:
    bad = yaml.safe_dump(
        _payload(
            spread_observations=[
                {
                    "timestamp_utc": "2026-09-18T13:20:00+00:00",
                    "notional_usd": 10_000,
                    "bid": 5,
                    "ask": 1,
                }
            ]
        )
    )
    broken = tmp_path / "mala.yaml"
    broken.write_text(bad, encoding="utf-8")

    assert main(_cli_args(tmp_path, "--template", str(broken))) == 2
    assert not (tmp_path / "derived").exists()


def test_a23_two_runs_with_the_same_now_are_byte_identical(tmp_path: Path) -> None:
    assert main(_cli_args(tmp_path)) == 0
    reports = tmp_path / "derived" / "reports"
    first_json = (reports / "cost_audit_2026-09-18.json").read_bytes()
    first_md = (reports / "cost_audit_2026-09-18.md").read_bytes()

    assert main(_cli_args(tmp_path)) == 0
    second_json = (reports / "cost_audit_2026-09-18.json").read_bytes()
    second_md = (reports / "cost_audit_2026-09-18.md").read_bytes()

    assert second_json == first_json
    assert second_md == first_md
    assert hashlib.sha256(second_json).hexdigest() == hashlib.sha256(first_json).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# A24 · Ningún cero sin estado `measured` y sin procedencia
# ─────────────────────────────────────────────────────────────────────────────
def test_a24_no_bare_zero_anywhere_in_the_guarded_blocks() -> None:
    observations = [_spread(hour_et=9, minute_et=20, bid=5000.0, ask=5001.2)]
    audited = [
        _consolidate().payload,
        _consolidate(
            spread_observations=observations, tracking_pairs=[_pair(lag_seconds=0)]
        ).payload,
    ]

    for payload in audited:
        for name in (
            "spread_cotizado",
            "tracking_difference",
            "slippage_ejecucion",
            "slippage_asumido",
            "financing_cut",
            "fx_cost",
        ):
            block = payload[name]
            for key in VALUE_KEYS:
                value = _numeric(block.get(key))
                if value is None or value != 0:
                    continue
                assert block.get("state") == MeasureState.MEASURED.value, f"{name}.{key} es 0"
                assert block.get("source"), f"{name}.{key} es 0 sin fuente"
                assert block.get("reason"), f"{name}.{key} es 0 sin motivo"


# ─────────────────────────────────────────────────────────────────────────────
# #64 etapa 2 · El supuesto pesimista del *slippage* (A14-A22)
# ─────────────────────────────────────────────────────────────────────────────
def _assumption() -> dict[str, object]:
    return cast("dict[str, object]", _consolidate().payload["slippage_asumido"])


def test_a14_the_assumption_constant_and_the_block_agree_and_it_is_not_a_measurement() -> None:
    """A14: constante declarada con nombre, valor, unidad, procedencia y motivo."""
    block = _assumption()
    assert block["name"] == "slippage_ejecucion_asumido"
    assert block["value_pct_of_r"] == str(cost_audit.SLIPPAGE_ASSUMPTION_PCT_OF_R)
    assert block["value_pct_of_r_unit"] == "% de `R`"
    assert block["provenance"] == "decision del propietario 2026-09-18"
    assert block["decided_on"] == "2026-09-18"
    assert block["is_measurement"] is False
    assert block["reason"] and block["owner_quote"]
    assert block["state"] == MeasureState.ASSUMED.value


def test_a15_the_assumption_is_a_ratio_over_r_and_the_bp_equivalent_is_illustrative() -> None:
    """A15: 100 % del margen de la puerta (b) = 20 % de `R`; el bp sale de esa relación."""
    block = _assumption()
    assert block["share_of_gate_b_allowance_pct"] == "100"
    assert block["gate_b_allowance_pct_of_r"] == "20"
    assert block["value_pct_of_r"] == "20"
    assert block["r_pct"] is None, "`R` sigue pendiente: nunca un valor por defecto"
    assert block["r_state"] == "unresolved" and block["r_issue"] == "#60"

    stated = cast("dict[str, object]", block["illustrative_equivalence"])
    assert stated["r_pct"] == "1"
    assert stated["value_bp"] == "20"
    assert stated["value_pct_of_notional"] == "0.2"
    assert stated["value_usd"] == "20"
    assert stated["notional_usd"] == "10000"
    assert stated["is_decision"] is False
    assert "no" in str(stated["warning"])

    # Con otro `R` ilustrativo, el equivalente cambia y la ratio sobre `R` **no**.
    halves = cost_audit.slippage_assumption_block(r_illustrative_pct=Decimal("0.5"))
    assert halves["value_pct_of_r"] == "20"
    other = cast("dict[str, object]", halves["illustrative_equivalence"])
    assert other["value_bp"] == "10"
    assert other["value_usd"] == "10"
    assert other["r_pct"] == "0.5"


def test_a16_the_assumption_cites_the_measurement_its_reason_and_its_limitation() -> None:
    """A16: motivo de no-cero, evidencia medida y limitación declarada."""
    block = _assumption()
    assert "no puede ser 0" in str(block["why_not_zero"])
    assert "≈30×" in str(block["why_not_zero"]) or "30x" in str(block["why_not_zero"])

    evidence = cast("dict[str, object]", block["measured_evidence"])
    assert evidence["median_bp"] == 13.2
    assert evidence["p90_bp"] == 26.6
    assert evidence["max_bp"] == 46.8
    assert evidence["sessions_above_10_bp"] == 36
    assert evidence["sessions_measured"] == 59
    assert evidence["declared_spread_bp"] == 0.42
    assert evidence["state"] == MeasureState.MEASURED.value and evidence["is_measurement"] is True
    assert "5 min" in str(evidence["definition"]) and "subasta" in str(evidence["definition"])

    limitations = " ".join(cast("list[str]", block["limitations"]))
    assert "59 sesiones" in limitations and "sub-minuto" in limitations

    blob = json.dumps(block, ensure_ascii=False)
    assert "asunción" in blob and "no" in blob
    assert "asunción" in str(block["assumption_note"])
    assert "no** una **medición" in str(block["assumption_note"])


def test_a17_the_assumption_does_not_feed_the_unmeasured_slippage() -> None:
    """A17: los tres estados son distinguibles y nadie rellena `slippage_ejecucion`."""
    payload = _consolidate().payload
    measured = payload["spread_cotizado"]
    assumed = payload["slippage_asumido"]
    unmeasured = payload["slippage_ejecucion"]

    assert measured["state"] == MeasureState.UNMEASURED.value
    assert assumed["state"] == MeasureState.ASSUMED.value
    assert unmeasured["state"] == MeasureState.UNMEASURED.value
    assert {assumed["state"], unmeasured["state"], MeasureState.MEASURED.value} == {
        "assumed",
        "unmeasured",
        "measured",
    }, "los tres estados deben existir y ser distinguibles"

    assert unmeasured["value_pct"] is None and unmeasured["value_usd"] is None
    assert unmeasured["source"] is None
    assert unmeasured["reason"] and unmeasured["how_to_fill"] == cost_audit.SLIPPAGE_HOW_TO_FILL
    assert "slippage_asumido" not in json.dumps(unmeasured, ensure_ascii=False)
    # Ningún valor del supuesto vive dentro de la medida sin medir: sigue en `null`.
    for key in ("value_usd", "value_pct", "value_points"):
        assert unmeasured[key] is None, f"`{key}` no puede venir del supuesto"
    assert "value_pct_of_r" not in unmeasured and "value_bp" not in unmeasured


def test_a18_the_assumption_is_a_separate_block_and_nothing_sums_it() -> None:
    """A18: sigue sin existir ningún «coste total» que fusione los bloques."""
    payload = _consolidate().payload
    for name in (
        "spread_cotizado",
        "tracking_difference",
        "slippage_ejecucion",
        "slippage_asumido",
    ):
        assert name in payload
        assert {"state", "reason"} <= set(payload[name])

    forbidden = {"total", "total_cost", "cost_total", "combined_cost", "grand_total", "sum_cost"}
    for key, _ in _walk(payload):
        assert key not in forbidden, f"el informe no puede combinar los bloques: {key}"


def test_a19_a_numeric_assumption_value_without_state_unit_and_provenance_fails() -> None:
    """A19: recorrido del JSON completo; `assumed` sin procedencia no pasa."""
    block = _assumption()
    assert _assumed_is_declared(block)

    for key in ("value_pct_of_r", "share_of_gate_b_allowance_pct"):
        assert block[key] is not None
        assert block[f"{key}_unit"]

    assert not _assumed_is_declared({**block, "provenance": ""})
    assert not _assumed_is_declared({**block, "is_measurement": True})
    assert not _assumed_is_declared({**block, "state": MeasureState.MEASURED.value})

    equivalence = cast("dict[str, object]", block["illustrative_equivalence"])
    assert equivalence["value_bp_unit"] and equivalence["value_usd_unit"]
    assert equivalence["value_pct_of_notional_unit"]
    assert equivalence["notional_usd"]


def _assumed_is_declared(block: dict[str, object]) -> bool:
    """Regla de A19: `assumed` exige estado, unidad, procedencia y no ser medición."""
    return (
        block.get("state") == MeasureState.ASSUMED.value
        and block.get("is_measurement") is False
        and bool(block.get("provenance"))
        and bool(block.get("decided_on"))
        and bool(block.get("value_pct_of_r_unit"))
        and block.get("reason") is not None
    )


def test_a20_the_blank_template_still_produces_the_assumption_block(tmp_path: Path) -> None:
    """A20: la plantilla vacía sigue valiendo y el supuesto no es un campo suyo."""
    template = tmp_path / "vacia.yaml"
    template.write_text(yaml.safe_dump(_payload()), encoding="utf-8")
    loads = cost_audit.load_template(template)
    assert loads.executions == (), "el supuesto no puede llegar por la plantilla"

    audit = consolidate(loads, calendar=load_calendar(), now=NOW, template_path=template)
    payload = audit.payload
    assert payload["slippage_asumido"]["state"] == MeasureState.ASSUMED.value
    for name in ("spread_cotizado", "tracking_difference", "slippage_ejecucion"):
        assert payload[name]["state"] == MeasureState.UNMEASURED.value

    # El YAML commiteado solo menciona el supuesto en comentarios.
    text = DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8")
    assert "slippage_asumido" in text
    for line in text.splitlines():
        if "slippage_asumido" in line:
            assert line.lstrip().startswith("#"), f"no puede ser un campo YAML: {line!r}"


def test_a21_the_assumption_never_makes_gate_b_a_pass() -> None:
    """A21: «asumir tu propio peor caso no puede ser un aprobado»."""
    gate = _consolidate().payload["phase0_gate_b"]
    assert gate["evaluable"] is False
    assert gate["assumption_is_measurement"] is False
    assert "asumir tu propio peor caso no puede ser un aprobado" in str(gate["reason"])
    assert gate["assumption"]["state"] == MeasureState.ASSUMED.value
    assert gate["assumption"]["is_measurement"] is False

    # Con un *slippage* sintético **medido**, la condición pasa a ser evaluable... pero el
    # supuesto sigue sin ser una medición y `R` sigue sin decidirse (lo dice #9).
    execution = {
        "timestamp_utc": "2026-09-18T13:30:05+00:00",
        "side": "long",
        "notional_usd": 10_000,
        "reference_price": 5000.0,
        "filled_price": 5002.0,
    }
    measured = _consolidate(executions=[execution]).payload["phase0_gate_b"]
    assert measured["evaluable"] is True
    assert measured["assumption_is_measurement"] is False
    assert measured["assumption"]["r_pct"] is None


def test_a22_the_report_says_where_the_assumption_is_enforced() -> None:
    """A22: la cadena #11 → #13 → #28, en máquina y en prosa."""
    block = _assumption()
    downstream = cast("dict[str, object]", block["enforced_downstream"])
    assert downstream["chain"] == ["#11", "#13", "#28"]
    assert downstream["status"] == "declaracion"
    assert "declaración" in str(downstream["note"])
    text = render_markdown(_consolidate())
    assert "## El supuesto pesimista del *slippage* (declarado el 2026-09-18)" in text
    assert "#11" in text and "#13" in text and "#28" in text
    assert "asumir tu propio peor caso no es una medición" in text
    assert "20 bp" in text, "el equivalente ilustrativo se publica"


# ─────────────────────────────────────────────────────────────────────────────
# A25, A26, A27, A29, A30 · Ingeniería y calidad
# ─────────────────────────────────────────────────────────────────────────────
def test_a25_the_new_module_does_not_import_network_libraries() -> None:
    source = Path(cost_audit.__file__).read_text(encoding="utf-8")
    for banned in ("httpx", "requests", "urllib", "socket", "http.client", "aiohttp"):
        assert banned not in source, f"el módulo nuevo no puede usar {banned}"


def test_a26_the_cli_writes_only_inside_the_given_root(tmp_path: Path) -> None:
    before = _fingerprint(REPOSITORY_DATA)
    assert main(_cli_args(tmp_path)) == 0
    assert _fingerprint(REPOSITORY_DATA) == before
    assert (tmp_path / "derived" / "reports" / "cost_audit_2026-09-18.json").is_file()


def test_a27_the_module_imports_only_approved_libraries() -> None:
    tree = ast.parse(Path(cost_audit.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    unused_here = {
        "polars",
        "numpy",
        "pandas",
        "scipy",
        "httpx",
        "arch",
        "sklearn",
        "lightgbm",
        "duckdb",
        "tenacity",
    }
    assert imported.isdisjoint(unused_here), f"este módulo no necesita: {imported & unused_here}"
    assert {"pydantic", "yaml", "loguru", "cfdtrader"} <= imported


def test_a29_the_module_docstring_declares_the_scope() -> None:
    docstring = cost_audit.__doc__ or ""
    assert "plan.md" in docstring
    assert "no se puede medir" in docstring
    assert "no existe ninguna ejecución real" in docstring
    assert "#11" in docstring
    assert "modelo de coste" in docstring


def test_a30_the_markdown_report_has_the_required_sections() -> None:
    audit = _consolidate()
    text = render_markdown(audit)

    for heading in (
        "# Auditoría de costes declarados del `SPX500:CFD` (tarea #8)",
        "## Tabla declarada (`plan.md` §3.3)",
        "## Coste por tramo de sesión",
        "## Coste por tamaño",
        "## Las tres medidas (nunca se fusionan)",
        "## Corte de financiación",
        "## Puerta (b) de la Fase 0 (#9)",
        "## Preguntas pendientes al bróker",
        "## Limitaciones (declaradas, no escondidas)",
    ):
        assert heading in text, f"falta la sección: {heading}"

    assert "**0.24**" in text and "**2.24**" in text
    assert "No hay ejecución real" in text
    assert "corte de financiación está sin verificar" in text
    assert "documento del bróker, no de una medición" in text
    assert "no se puede evaluar" in text
    for question in BROKER_QUESTIONS:
        assert question["question"] in text
    for limitation in LIMITATIONS:
        assert limitation in text


def test_the_declared_shares_match_the_declared_dollar_amounts() -> None:
    """La tabla declarada es coherente consigo misma: el % sale del $ y del nocional."""
    assert Decimal("0.42") / REFERENCE_NOTIONAL_USD * 100 == cost_audit.SPREAD_PCT
    assert Decimal("-0.18") / REFERENCE_NOTIONAL_USD * 100 == CARRY_SHORT_PCT_PER_NIGHT
    assert Decimal("1.82") / REFERENCE_NOTIONAL_USD * 100 == CARRY_LONG_PCT_PER_NIGHT


def test_a18_a_malformed_template_raises_a_typed_error(tmp_path: Path) -> None:
    broken = tmp_path / "roto.yaml"
    broken.write_text("spread_observations: [\n", encoding="utf-8")
    with pytest.raises(CostTemplateError):
        load_template(broken)
