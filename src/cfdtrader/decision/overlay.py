"""Overlay del `NewsAgent` sobre la decisión (`tech_stack.md` §4.9) — tarea #35.

La pregunta que responde este módulo: **¿cuánto poder tiene el LLM sobre la recomendación, y
cómo se demuestra que no tiene más?**

Exactamente el que §4.9 le concede, y ninguno más:

- **Veto binario.** Un evento de magnitud alta y confianza alta convierte el día en ``NOTHING``.
  Es un veto **de incertidumbre, no de dirección**: el overlay no sabe si el gate iba a decir
  ``LONG`` o ``SHORT``, y no le importa — dice «hoy las noticias pesan demasiado como para fiarse
  del modelo». Por eso jamás puede **crear** una dirección donde no la había.
- **Ajuste acotado a ±10 puntos porcentuales** sobre la probabilidad calibrada. El límite es un
  **techo duro**, no una sugerencia: se aplica recortando, siempre.
- **El overlay no calcula números ni decide la dirección** (§4.9, «regla de diseño que no cambia»).
  Los ajustes de ``ADJUSTMENT_BY_MAGNITUDE`` son una **decisión declarada** del propietario, no un
  número que salga del modelo; están aquí, en una sola tabla, para poder revisarlos.

Este módulo es **puro**: recibe una extracción ya validada por #32 y produce una decisión tipada y
determinista. No habla con el proveedor, no lee el reloj y no toca el disco. Depende del
**vocabulario** de eventos de :mod:`cfdtrader.agents.news` (necesita comparar magnitudes y
sentimientos), nunca de su cliente.

Por qué el vocabulario de estados aparece en dos módulos
-------------------------------------------------------
Los cinco estados de §12.5 son los mismos que publica la capa de coste (#33) y los que escribe el
diario (#39 los declara en ``LLM_OVERLAYS``). **No se comparten por import, y es deliberado:** el
reparto de capas está probado en los dos sentidos — #33 comprueba que ``llm/budget.py`` no importa
``cfdtrader.decision`` y este módulo comprueba que no importa ``cfdtrader.llm``; compartir el enum
obligaría a romper una de las dos reglas. El origen de verdad es ``LLM_OVERLAYS`` y **cada** módulo
tiene su prueba que compara su vocabulario con él: una sexta variante hace fallar las dos.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from cfdtrader.agents.news import Magnitude, NewsEvent, NewsExtraction, Sentiment

__all__ = [
    "ADJUSTMENT_BY_MAGNITUDE",
    "MAX_ADJUSTMENT_PCT",
    "VETO_MIN_CONFIDENCE",
    "OverlayDecision",
    "OverlayState",
    "apply_overlay",
    "disabled_overlay",
    "overlay_from_extraction",
    "vetoes",
]

#: Techo duro de §4.9: el ajuste no puede alejarse más de esto, en puntos porcentuales.
MAX_ADJUSTMENT_PCT: Final[float] = 10.0

#: Ajuste **declarado** por magnitud, antes de recortar. La suma se recorta a ±10 pp. Es una
#: decisión del propietario y vive en una sola tabla para poder revisarla (§11 bis, #60).
ADJUSTMENT_BY_MAGNITUDE: Final[dict[str, float]] = {
    Magnitude.LOW.value: 2.5,
    Magnitude.MEDIUM.value: 5.0,
    Magnitude.HIGH.value: 10.0,
}

#: Confianza mínima para que un evento de magnitud alta vete la sesión.
VETO_MIN_CONFIDENCE: Final[float] = 0.8


class OverlayState(StrEnum):
    """Los cinco estados de §12.5, declarados por #39. Aquí no se inventa ningún vocabulario."""

    APPLIED = "applied"
    VETO = "veto"
    DISABLED_BUDGET = "disabled_budget"
    DISABLED_ERROR = "disabled_error"
    DISABLED_TIMEOUT = "disabled_timeout"


class OverlayDecision(BaseModel):
    """Lo que el overlay pide: un estado, un ajuste ya acotado y por qué.

    El techo de ±10 pp está **en el tipo**: construir una decisión con 50 pp es un error de
    validación, no algo que se recorte por dentro y nadie note.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: OverlayState
    adjustment_pct: float = Field(default=0.0, ge=-MAX_ADJUSTMENT_PCT, le=MAX_ADJUSTMENT_PCT)
    reasons: tuple[str, ...] = Field(
        default=(), description="los hechos que justifican el estado, en orden de aparicion"
    )
    prompt_hash: str | None = Field(
        default=None, description="hash del prompt que produjo los eventos; lo persiste el diario"
    )


def _contribution(event: NewsEvent) -> float:
    """Cuánto pide ese evento, con signo. ``neutral`` no pide nada."""
    weight = ADJUSTMENT_BY_MAGNITUDE[event.magnitude.value]
    if event.sentiment is Sentiment.BULLISH:
        return weight
    if event.sentiment is Sentiment.BEARISH:
        return -weight
    return 0.0


def _is_vetoing(event: NewsEvent) -> bool:
    """Un evento de magnitud alta y confianza alta basta para no fiarse del modelo hoy."""
    return event.magnitude is Magnitude.HIGH and event.confidence >= VETO_MIN_CONFIDENCE


def overlay_from_extraction(
    extraction: NewsExtraction, *, prompt_hash: str | None = None
) -> OverlayDecision:
    """La decisión del overlay a partir de una extracción de #32 ya validada.

    Sin eventos, el overlay **se aplica** con ajuste 0: no hay nada que vetar ni que ajustar, y
    eso es distinto de estar desactivado. El estado ``applied`` es el único que este constructor
    produce; los ``disabled_*`` los decide la capa de coste y llegan por :func:`disabled_overlay`.
    """
    events = extraction.events
    chosen = extraction.prompt_hash if prompt_hash is None else prompt_hash
    vetoing = tuple(event for event in events if _is_vetoing(event))
    if vetoing:
        return OverlayDecision(
            state=OverlayState.VETO,
            adjustment_pct=0.0,
            reasons=tuple(
                f"{event.event_type.value}/{event.magnitude.value}: {event.rationale}"
                for event in vetoing
            ),
            prompt_hash=chosen,
        )
    total = sum(_contribution(event) for event in events)
    return OverlayDecision(
        state=OverlayState.APPLIED,
        adjustment_pct=max(-MAX_ADJUSTMENT_PCT, min(MAX_ADJUSTMENT_PCT, total)),
        reasons=tuple(f"{event.event_type.value}/{event.sentiment.value}" for event in events),
        prompt_hash=chosen,
    )


def disabled_overlay(
    state: OverlayState, *, reasons: Sequence[str] = (), prompt_hash: str | None = None
) -> OverlayDecision:
    """Una decisión **desactivada**: ajuste 0 y motivo. El pipeline sigue sin overlay."""
    if state is OverlayState.APPLIED or state is OverlayState.VETO:
        raise ValueError(f"disabled_overlay: {state.value!r} no es un estado desactivado")
    return OverlayDecision(
        state=state, adjustment_pct=0.0, reasons=tuple(reasons), prompt_hash=prompt_hash
    )


def vetoes(decision: OverlayDecision | None) -> bool:
    """Si esa decisión veta la sesión."""
    return decision is not None and decision.state is OverlayState.VETO


def apply_overlay(prob_up_calibrated: float, decision: OverlayDecision | None) -> float:
    """La probabilidad con el ajuste aplicado, acotada a ``[0, 1]`` y a ±10 pp.

    Sin overlay, o con el overlay desactivado o vetando, devuelve la probabilidad **intacta**: el
    veto no se expresa moviendo la probabilidad, se expresa bloqueando la dirección.
    """
    if decision is None or decision.state is not OverlayState.APPLIED:
        return prob_up_calibrated
    bounded = max(-MAX_ADJUSTMENT_PCT, min(MAX_ADJUSTMENT_PCT, decision.adjustment_pct))
    return min(1.0, max(0.0, prob_up_calibrated + bounded / 100.0))
