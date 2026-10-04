"""El material para decidir `R` y los umbrales: se recalcula, no se copia (tarea #131).

`_docs/r_y_umbrales_material_2026-10-04.md` es una hoja para una **decisión del propietario** (#60).
Lo que se blinda aquí es que sus cifras **salen del código** y siguen saliendo:

- las de la tabla (`p*` y `EV` por `R` candidato) se recomputan con la API publica
  (`analysis.cost_audit.slippage_assumption_block` + `backtest.costs.DECLARED_SPREAD_HALF_PCT`);
- la evidencia **medida** de la apertura (mediana, p90, maximo) se lee del artefacto, no se copia;
- el cruce entre el supuesto y lo medido se recalcula;
- y todo lo anterior se comprueba **contra el texto del documento**, para que si el supuesto, el
  diferencial o la evidencia cambian, la hoja **falle en voz alta** en vez de quedarse vieja.

No hay modulo nuevo de `src/`: es una prueba de **contrato de documento**, asi que el suelo de
cobertura (90 % sentencias / 85 % ramas) **no aplica** y se declara aqui.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Final

from cfdtrader.analysis.cost_audit import slippage_assumption_block
from cfdtrader.backtest.costs import (
    DECLARED_SPREAD_HALF_PCT,
    SLIPPAGE_ASSUMPTION_PCT_OF_R,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MATERIAL_PATH: Final[Path] = REPO_ROOT / "_docs" / "r_y_umbrales_material_2026-10-04.md"

#: El diferencial de **ida y vuelta**: la constante del repo es el medio diferencial.
SPREAD_ROUND_TRIP_PCT: Final[Decimal] = DECLARED_SPREAD_HALF_PCT * 2

#: Los `R` candidatos de la hoja, en % del nocional (no en fraccion).
CANDIDATE_R_PCT: Final[tuple[Decimal, ...]] = (
    Decimal("0.5"),
    Decimal("0.75"),
    Decimal("1.0"),
    Decimal("1.5"),
    Decimal("2.0"),
)

#: La probabilidad calibrada de la ultima ejecucion real del camino diario (2026-09-17).
OBSERVED_P: Final[Decimal] = Decimal("0.5054")


def _material_text() -> str:
    return MATERIAL_PATH.read_text(encoding="utf-8")


def _es(value: Decimal, digits: int = 2) -> str:
    """El numero en la convencion del documento: decimales con coma."""
    return f"{value:.{digits}f}".replace(".", ",")


def assumed_slippage_pct_of_notional(r_pct: Decimal) -> Decimal:
    """El supuesto (20 % de `R`) en % del nocional, por la API publica."""
    block = slippage_assumption_block(r_illustrative_pct=r_pct)
    return Decimal(str(block["illustrative_equivalence"]["value_pct_of_notional"]))


def total_cost_pct(r_pct: Decimal) -> Decimal:
    """`c` del bracket: diferencial de ida y vuelta mas el supuesto."""
    return SPREAD_ROUND_TRIP_PCT + assumed_slippage_pct_of_notional(r_pct)


def break_even_pct(r_pct: Decimal, c_pct: Decimal) -> Decimal:
    """`p* = (R + c) / 2R` (`plan.md` §4.4), en %."""
    return (r_pct + c_pct) / (2 * r_pct)


def ev_pct(r_pct: Decimal, c_pct: Decimal, p: Decimal) -> Decimal:
    """`EV` de un bracket simetrico, en % del nocional: `(2p - 1) * R - c`."""
    return (2 * p - 1) * r_pct - c_pct


def _measured_opening_bp() -> dict[str, Decimal]:
    """La evidencia **medida** de la apertura, tal como la publica el artefacto."""
    block = slippage_assumption_block()
    evidence = block["measured_evidence"]
    return {
        "median": Decimal(str(evidence["median_bp"])),
        "p90": Decimal(str(evidence["p90_bp"])),
        "max": Decimal(str(evidence["max_bp"])),
        "spread": Decimal(str(evidence["declared_spread_bp"])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# El diferencial declarado que usa la hoja
# ─────────────────────────────────────────────────────────────────────────────
def test_the_document_uses_the_round_trip_spread() -> None:
    """`DECLARED_SPREAD_HALF_PCT` es el **medio** diferencial; la ida y vuelta lo duplica."""
    assert Decimal("0.0042") == SPREAD_ROUND_TRIP_PCT
    assert "0,0042 % = 0,42 bp" in _material_text()


# ─────────────────────────────────────────────────────────────────────────────
# `p*` por `R` candidato: la tabla de la hoja
# ─────────────────────────────────────────────────────────────────────────────
def test_the_assumption_table_of_p_star_matches_the_document() -> None:
    text = _material_text()
    for r_pct in CANDIDATE_R_PCT:
        c_pct = total_cost_pct(r_pct)
        with_assumption = break_even_pct(r_pct, c_pct) * 100
        with_spread = break_even_pct(r_pct, SPREAD_ROUND_TRIP_PCT) * 100
        assert _es(with_assumption) in text, f"R = {r_pct}: falta {_es(with_assumption)}"
        assert _es(with_spread) in text, f"R = {r_pct}: falta {_es(with_spread)}"


def test_the_assumption_is_twenty_percent_of_r() -> None:
    """La equivalencia que da la hoja sale de la ratio declarada, no de un numero cableado."""
    assert Decimal("20") == SLIPPAGE_ASSUMPTION_PCT_OF_R
    for r_pct in CANDIDATE_R_PCT:
        expected = SLIPPAGE_ASSUMPTION_PCT_OF_R * r_pct / Decimal(100)
        assert assumed_slippage_pct_of_notional(r_pct) == expected


def test_the_spread_only_row_reproduces_plan_section_4_4() -> None:
    """`p*` con `c = 0,0042 %` y `R = 1 %` es el 50,21 % que declara `plan.md` §4.4."""
    p_star = break_even_pct(Decimal("1.0"), SPREAD_ROUND_TRIP_PCT) * 100
    assert _es(p_star) == "50,21"
    assert "50,21 %" in _material_text()


def test_the_row_of_r_one_percent_is_the_highlighted_one() -> None:
    """La hoja destaca `R = 1 %`: los tres valores de esa fila tienen que cuadrar."""
    c_pct = total_cost_pct(Decimal("1.0"))
    assert _es(c_pct, 4) == "0,2042"
    assert _es(c_pct * 100, 2) == "20,42"
    assert _es(break_even_pct(Decimal("1.0"), c_pct) * 100) == "60,21"


# ─────────────────────────────────────────────────────────────────────────────
# La conclusion que sostiene la hoja: el EV negativo en toda la banda
# ─────────────────────────────────────────────────────────────────────────────
def test_the_ev_is_negative_for_every_candidate_and_says_so() -> None:
    text = _material_text()
    for r_pct in CANDIDATE_R_PCT:
        value = ev_pct(r_pct, total_cost_pct(r_pct), OBSERVED_P)
        assert value < 0, f"R = {r_pct}: el EV no es negativo ({value})"
        assert _es(abs(value), 4) in text, f"R = {r_pct}: falta |EV| = {_es(abs(value), 4)}"


def test_the_declared_assumption_lifts_the_bar_from_50_to_60() -> None:
    """Con el supuesto, `p*` ronda el 60 % en toda la banda: no es un ajuste fino."""
    for r_pct in CANDIDATE_R_PCT:
        p_star = break_even_pct(r_pct, total_cost_pct(r_pct)) * 100
        assert Decimal("59.5") < p_star < Decimal("61"), f"R = {r_pct}: p* = {p_star}"


# ─────────────────────────────────────────────────────────────────────────────
# El cruce con la evidencia MEDIDA de la apertura
# ─────────────────────────────────────────────────────────────────────────────
def test_the_measured_evidence_is_the_one_the_document_cites() -> None:
    measured = _measured_opening_bp()
    assert measured["median"] == Decimal("13.2")
    assert measured["p90"] == Decimal("26.6")
    assert measured["max"] == Decimal("46.8")
    assert measured["spread"] == Decimal("0.42")
    text = _material_text()
    assert "13,2 bp" in text and "26,6 bp" in text and "46,8 bp" in text


def test_the_assumption_crosses_the_measured_opening_where_the_document_says() -> None:
    """El supuesto iguala la mediana, el p90 y el maximo medidos en los `R` que la hoja declara."""
    text = _material_text()
    measured = _measured_opening_bp()
    for name in ("median", "p90", "max"):
        crossing = measured[name] / SLIPPAGE_ASSUMPTION_PCT_OF_R
        assert _es(crossing) in text, f"{name}: falta el cruce en R = {_es(crossing)}"


def test_below_the_crossing_the_pessimistic_assumption_is_optimistic() -> None:
    """Por debajo de la mediana medida, el supuesto deja de ser pesimista: es peor que el dato."""
    measured = _measured_opening_bp()
    below = Decimal("0.5")
    assertion_bp = SLIPPAGE_ASSUMPTION_PCT_OF_R * below
    assert assertion_bp < measured["median"]
    assert "no es pesimista, es optimista" in _material_text()


# ─────────────────────────────────────────────────────────────────────────────
# La hoja no decide: lo declara
# ─────────────────────────────────────────────────────────────────────────────
def test_the_document_does_not_decide_r() -> None:
    text = _material_text()
    assert "no la decisión" in text
    assert "quien decide es el propietario" in text
    assert "#60" in text and "#131" in text
