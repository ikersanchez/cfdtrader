"""Tests del overlay del NewsAgent (tarea #35): A1, A2, A10 (parte pura) y bordes.

Se prueba **sin proveedor y sin gate**: lo que vive aquí es la decisión del overlay y su efecto
sobre la probabilidad. La mitad del gate (que un veto no invierta la dirección, que el veto sea la
regla 20) se prueba donde vive el gate.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from cfdtrader.agents.news import NewsEvent, NewsExtraction
from cfdtrader.decision.overlay import (
    ADJUSTMENT_BY_MAGNITUDE,
    MAX_ADJUSTMENT_PCT,
    VETO_MIN_CONFIDENCE,
    OverlayDecision,
    OverlayState,
    apply_overlay,
    disabled_overlay,
    overlay_from_extraction,
    vetoes,
)
from cfdtrader.journal.decision_log import LLM_OVERLAYS

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODULE: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "decision" / "overlay.py"
PROMPT_HASH: Final[str] = "sha256:" + "a" * 64


def _event(
    *, magnitude: str = "low", sentiment: str = "bullish", confidence: float = 0.5
) -> NewsEvent:
    return NewsEvent.model_validate(
        {
            "headline_hash": "sha256:" + "b" * 64,
            "event_type": "monetary_policy",
            "sentiment": sentiment,
            "magnitude": magnitude,
            "confidence": confidence,
            "horizon": "intraday",
            "rationale": "una frase de prueba",
        }
    )


def _extraction(*events: NewsEvent) -> NewsExtraction:
    return NewsExtraction(events=events, prompt_hash=PROMPT_HASH, model="modelo-de-prueba")


def _imported_modules(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


# ─────────────────────────────────────────────────────────────────────────────
# A1 · El módulo, su tipo y su vocabulario
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_module_and_its_type_exist() -> None:
    assert MODULE.is_file()
    assert MAX_ADJUSTMENT_PCT == 10.0
    assert OverlayDecision.model_config.get("frozen") is True
    assert OverlayDecision.model_config.get("extra") == "forbid"
    assert set(OverlayDecision.model_fields) == {
        "state",
        "adjustment_pct",
        "reasons",
        "prompt_hash",
    }

    with pytest.raises(ValidationError):
        OverlayDecision.model_validate({"state": "applied", "campo_de_mas": 1})


def test_a1_the_vocabulary_is_the_one_39_declared() -> None:
    assert {member.value for member in OverlayState} == set(LLM_OVERLAYS)


def test_a1_the_overlay_does_not_know_the_provider() -> None:
    modules = _imported_modules(ast.parse(MODULE.read_text(encoding="utf-8")))
    assert "openai" not in modules
    assert not any(name.endswith("llm.openai_client") for name in modules)
    assert not any(name.startswith("cfdtrader.llm") for name in modules), (
        "el overlay no debe depender de la capa LLM: es una decision, no una llamada"
    )


# ─────────────────────────────────────────────────────────────────────────────
# A2 · El techo de ±10 pp es duro
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_adjustment_bound_is_in_the_type() -> None:
    for out_of_range in (50.0, -50.0, 10.01, -10.01):
        with pytest.raises(ValidationError):
            OverlayDecision(state=OverlayState.APPLIED, adjustment_pct=out_of_range)

    assert OverlayDecision(state=OverlayState.APPLIED, adjustment_pct=10.0).adjustment_pct == 10.0
    assert OverlayDecision(state=OverlayState.APPLIED, adjustment_pct=-10.0).adjustment_pct == -10.0


def test_a2_apply_never_moves_more_than_ten_points() -> None:
    for requested in (x / 10.0 for x in range(-100, 101)):
        decision = OverlayDecision(
            state=OverlayState.APPLIED,
            adjustment_pct=max(-MAX_ADJUSTMENT_PCT, min(MAX_ADJUSTMENT_PCT, requested)),
        )
        for prob in (0.0, 0.25, 0.5, 0.75, 1.0):
            moved = apply_overlay(prob, decision)
            assert 0.0 <= moved <= 1.0
            assert abs(moved - prob) <= MAX_ADJUSTMENT_PCT / 100.0 + 1e-12


def test_a2_without_overlay_the_probability_is_intact() -> None:
    assert apply_overlay(0.5, None) == 0.5
    for state in (
        OverlayState.VETO,
        OverlayState.DISABLED_BUDGET,
        OverlayState.DISABLED_ERROR,
        OverlayState.DISABLED_TIMEOUT,
    ):
        assert apply_overlay(0.5, OverlayDecision(state=state)) == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# A10 · El overlay no puede crear una dirección donde no la había (parte pura)
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_the_overlay_cannot_create_direction() -> None:
    """Un veto no toca la probabilidad: se expresa bloqueando, no empujando el número."""
    vetoing = overlay_from_extraction(
        _extraction(_event(magnitude="high", sentiment="bullish", confidence=0.95))
    )
    assert vetoing.state is OverlayState.VETO
    assert vetoing.adjustment_pct == 0.0
    assert vetoes(vetoing)
    assert apply_overlay(0.99, vetoing) == 0.99, "un veto alcista no puede subir la probabilidad"

    neutral = OverlayDecision(state=OverlayState.APPLIED)
    assert not vetoes(neutral)
    assert not vetoes(None)


# ─────────────────────────────────────────────────────────────────────────────
# Bordes del constructor y de los estados desactivados
# ─────────────────────────────────────────────────────────────────────────────
def test_a_veto_is_uncertainty_and_not_direction() -> None:
    for sentiment in ("bullish", "bearish", "neutral"):
        decision = overlay_from_extraction(
            _extraction(_event(magnitude="high", sentiment=sentiment, confidence=0.9))
        )
        assert decision.state is OverlayState.VETO, f"{sentiment} de magnitud alta debe vetar"


def test_the_adjustment_sums_and_clips() -> None:
    applied = overlay_from_extraction(
        _extraction(
            _event(magnitude="high", sentiment="bullish"),
            _event(magnitude="medium", sentiment="bearish"),
        )
    )
    assert applied.state is OverlayState.APPLIED
    assert applied.adjustment_pct == pytest.approx(
        ADJUSTMENT_BY_MAGNITUDE["high"] - ADJUSTMENT_BY_MAGNITUDE["medium"]
    )

    clipped = overlay_from_extraction(
        _extraction(*[_event(magnitude="high", sentiment="bullish") for _ in range(5)])
    )
    assert clipped.adjustment_pct == MAX_ADJUSTMENT_PCT, "la suma se recorta al techo"


def test_an_empty_extraction_applies_with_no_adjustment() -> None:
    decision = overlay_from_extraction(_extraction())
    assert decision.state is OverlayState.APPLIED
    assert decision.adjustment_pct == 0.0
    assert decision.prompt_hash == PROMPT_HASH
    assert decision.reasons == ()


def test_a_neutral_event_moves_nothing() -> None:
    """``neutral`` no empuja la probabilidad en ningún sentido, ni con magnitud alta."""
    decision = overlay_from_extraction(
        _extraction(_event(magnitude="high", sentiment="neutral", confidence=0.5))
    )
    assert decision.state is OverlayState.APPLIED
    assert decision.adjustment_pct == 0.0


def test_the_veto_threshold_is_where_it_says_it_is() -> None:
    almost = VETO_MIN_CONFIDENCE - 0.01
    below = overlay_from_extraction(
        _extraction(_event(magnitude="high", sentiment="bearish", confidence=almost))
    )
    assert below.state is OverlayState.APPLIED

    at = overlay_from_extraction(
        _extraction(_event(magnitude="high", sentiment="bearish", confidence=VETO_MIN_CONFIDENCE))
    )
    assert at.state is OverlayState.VETO


def test_disabled_overlay_only_accepts_disabled_states() -> None:
    for state in (
        OverlayState.DISABLED_BUDGET,
        OverlayState.DISABLED_ERROR,
        OverlayState.DISABLED_TIMEOUT,
    ):
        decision = disabled_overlay(state, reasons=("motivo",))
        assert decision.adjustment_pct == 0.0
        assert decision.reasons == ("motivo",)

    for state in (OverlayState.APPLIED, OverlayState.VETO):
        with pytest.raises(ValueError, match="no es un estado desactivado"):
            disabled_overlay(state)
