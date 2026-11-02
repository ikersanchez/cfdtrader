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
from cfdtrader.decision import overlay as overlay_module
from cfdtrader.decision.overlay import (
    ADJUSTMENT_BY_MAGNITUDE,
    MAX_ADJUSTMENT_PCT,
    TILT_DEADBAND_PCT,
    VETO_MIN_CONFIDENCE,
    OverlayDecision,
    OverlayState,
    OverlayTilt,
    apply_overlay,
    disabled_overlay,
    overlay_from_extraction,
    tilt_from_events,
    vetoes,
)
from cfdtrader.journal.decision_log import LLM_OVERLAYS

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODULE: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "decision" / "overlay.py"
PROMPT_HASH: Final[str] = "sha256:" + "a" * 64

#: Las once claves que publica el sesgo del lote (#161), en el orden del contrato.
TILT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "net_pct",
        "clamped",
        "ratio",
        "side",
        "weighted",
        "n_events",
        "n_neutral",
        "consensus",
        "horizon",
        "dominant",
        "capped_events",
    }
)


def _event(
    *,
    magnitude: str = "low",
    sentiment: str = "bullish",
    confidence: float = 0.5,
    event_type: str = "monetary_policy",
    horizon: str = "intraday",
    rationale: str = "una frase de prueba",
) -> NewsEvent:
    return NewsEvent.model_validate(
        {
            "headline_hash": "sha256:" + "b" * 64,
            "event_type": event_type,
            "sentiment": sentiment,
            "magnitude": magnitude,
            "confidence": confidence,
            "horizon": horizon,
            "rationale": rationale,
        }
    )


def _extraction(*events: NewsEvent, capped_events: int = 0) -> NewsExtraction:
    return NewsExtraction(
        events=events,
        prompt_hash=PROMPT_HASH,
        model="modelo-de-prueba",
        capped_events=capped_events,
    )


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
        "tilt",
    }

    with pytest.raises(ValidationError):
        OverlayDecision.model_validate({"state": "applied", "campo_de_mas": 1})

    # El sesgo del lote (#161): un tipo congelado y cerrado con exactamente once campos.
    assert OverlayTilt.model_config.get("frozen") is True
    assert OverlayTilt.model_config.get("extra") == "forbid"
    assert set(OverlayTilt.model_fields) == TILT_FIELDS
    assert {"OverlayTilt", "tilt_from_events"} <= set(overlay_module.__all__)

    with pytest.raises(ValidationError):
        OverlayTilt.model_validate({**tilt_from_events(()).model_dump(), "campo_de_mas": 1})


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


# ─────────────────────────────────────────────────────────────────────────────
# El sesgo del lote (#161): aritmética declarada, sin autoridad sobre la decisión
# ─────────────────────────────────────────────────────────────────────────────
def test_tilt_clamped_marks_the_recorte_without_changing_the_ceiling() -> None:
    """Seis ``high`` alcistas: ``net_pct`` 60 sin recortar, ajuste recortado a +10 pp."""
    up = overlay_from_extraction(_extraction(*[_event(magnitude="high") for _ in range(6)]))
    assert up.tilt is not None
    assert up.tilt.net_pct == pytest.approx(60.0)
    assert up.tilt.clamped is True
    assert up.adjustment_pct == MAX_ADJUSTMENT_PCT

    down = overlay_from_extraction(
        _extraction(*[_event(magnitude="high", sentiment="bearish") for _ in range(6)])
    )
    assert down.tilt is not None
    assert down.tilt.net_pct == pytest.approx(-60.0)
    assert down.tilt.clamped is True
    assert down.adjustment_pct == -MAX_ADJUSTMENT_PCT

    flat = overlay_from_extraction(
        _extraction(_event(magnitude="high"), _event(magnitude="high", sentiment="bearish"))
    )
    assert flat.tilt is not None
    assert flat.tilt.net_pct == pytest.approx(0.0)
    assert flat.tilt.clamped is False
    assert flat.adjustment_pct == 0.0


def test_tilt_composition_counts_the_neutrals_in_the_gross_weight() -> None:
    """``low bullish, low bullish, low bearish, low neutral``: ratio 0,25 y consenso 0,5."""
    tilt = tilt_from_events(
        (
            _event(),
            _event(),
            _event(sentiment="bearish"),
            _event(sentiment="neutral"),
        )
    )
    assert tilt.net_pct == pytest.approx(2.5)
    assert tilt.ratio == pytest.approx(0.25)
    assert tilt.weighted == {"bullish": 5.0, "bearish": 2.5, "neutral": 2.5}
    assert tilt.consensus == pytest.approx(0.5)
    assert tilt.n_events == 4
    assert tilt.n_neutral == 1
    assert tilt.side == "neutral", "la banda muerta es inclusiva: 2,5 no declara lado"


def test_tilt_deadband_is_inclusive_and_anchored_to_the_low_magnitude() -> None:
    assert TILT_DEADBAND_PCT == ADJUSTMENT_BY_MAGNITUDE["low"] == 2.5

    mixed = tilt_from_events((_event(), _event(sentiment="bearish")))
    assert mixed.net_pct == pytest.approx(0.0)
    assert mixed.side == "neutral"

    one_low = tilt_from_events((_event(),))
    assert one_low.net_pct == pytest.approx(TILT_DEADBAND_PCT)
    assert one_low.side == "neutral", "un único titular de magnitud baja no puede declarar lado"

    two_up = tilt_from_events((_event(), _event()))
    assert two_up.net_pct == pytest.approx(5.0)
    assert two_up.side == "long"

    two_down = tilt_from_events((_event(sentiment="bearish"), _event(sentiment="bearish")))
    assert two_down.net_pct == pytest.approx(-5.0)
    assert two_down.side == "short"


def test_tilt_empty_declares_the_lote_vacio_without_inventing_numbers() -> None:
    tilt = tilt_from_events(())
    assert tilt.net_pct == 0.0
    assert tilt.clamped is False
    assert tilt.ratio == 0.0
    assert tilt.side == "neutral"
    assert tilt.weighted == {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
    assert tilt.n_events == 0
    assert tilt.n_neutral == 0
    assert tilt.consensus is None
    assert tilt.horizon == {"intraday": 0, "days": 0}
    assert tilt.dominant is None
    assert tilt.capped_events == 0


def test_tilt_horizon_and_dominant_are_read_from_the_batch() -> None:
    first_high = _event(
        magnitude="high",
        event_type="inflation",
        horizon="intraday",
        rationale="el primero de los dos altos",
    )
    second_high = _event(
        magnitude="high",
        sentiment="bearish",
        event_type="geopolitics",
        horizon="days",
        rationale="el segundo de los dos altos",
    )
    tilt = tilt_from_events((_event(magnitude="medium", horizon="days"), first_high, second_high))

    assert tilt.horizon == {"intraday": 1, "days": 2}
    assert tilt.dominant is not None
    assert tilt.dominant.event_type.value == "inflation"
    assert tilt.dominant.sentiment.value == "bullish"
    assert tilt.dominant.magnitude.value == "high"
    assert tilt.dominant.rationale == "el primero de los dos altos", "en un empate gana el primero"


def test_tilt_in_veto_is_still_published() -> None:
    """El overlay vetado **también** publica su sesgo: son dos ejes distintos (#161)."""
    decision = overlay_from_extraction(
        _extraction(
            _event(magnitude="high", sentiment="bearish", confidence=0.9),
            _event(magnitude="high", sentiment="bearish", confidence=0.9),
        )
    )
    assert decision.state is OverlayState.VETO
    assert decision.adjustment_pct == 0.0
    assert decision.tilt is not None
    assert decision.tilt.net_pct == pytest.approx(-20.0)
    assert decision.tilt.clamped is True
    assert decision.tilt.side == "short"
    assert decision.tilt.n_events == 2

    capped = overlay_from_extraction(_extraction(_event(magnitude="medium"), capped_events=3))
    assert capped.tilt is not None
    assert capped.tilt.capped_events == 3


def test_tilt_disabled_leaves_no_tilt_at_all() -> None:
    for state in (
        OverlayState.DISABLED_BUDGET,
        OverlayState.DISABLED_ERROR,
        OverlayState.DISABLED_TIMEOUT,
    ):
        assert disabled_overlay(state).tilt is None, "sin extracción no hay lote que describir"

    assert overlay_from_extraction(_extraction()).tilt is not None
