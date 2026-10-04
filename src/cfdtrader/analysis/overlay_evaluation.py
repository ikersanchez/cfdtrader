"""Evaluación incremental del overlay LLM — tarea #38 (puerta de salida de la Fase 3).

La pregunta que responde este módulo: **¿el LLM aporta algo medible o solo redacta?**

La respuesta honesta está condicionada por un hecho que `tech_stack.md` §4.9 fija palabra por
palabra: **el backtest corre siempre sin overlay**, porque no existe un archivo histórico de
noticias con ``published_at`` fiable a coste razonable. Por tanto **no hay medición pareada**
(con/sin overlay) que comparar: el Brier y el Sharpe OOS ``con`` overlay **no son obtenibles hacia
atrás**. Es lo que la puerta de la Fase 3, registrada en #120 el 2026-10-03 *antes de ver
resultados*, admite como **``not_evaluable``**, con su consecuencia declarada de antemano: el LLM
queda **solo como redactor** y la evidencia se traslada al *paper trading* de la Fase 4 (#45).

El módulo **no inventa números**: cuando falta la medición, los campos van ``None`` y el veredicto
es ``not_evaluable``. Cuando hay archivo (o *paper trading* acumulado), compara de verdad.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict

__all__ = [
    "PAPER_TRADING_DESIGN",
    "OverlayEvaluation",
    "OverlayVerdict",
    "evaluate_overlay",
    "render_markdown",
]

#: Método alternativo de evaluación: *paper trading* prospectivo (Fase 4, #45).
PAPER_TRADING_DESIGN: Final[str] = (
    "Registro prospectivo (Fase 4, #45): cada dia se guardan la recomendacion con el overlay y "
    "sin el, y la accion humana; al cierre se registran el P&L y la atribucion, y se compara "
    "hacia delante. Sin archivo historico de noticias, es el unico metodo honesto."
)


class OverlayVerdict(StrEnum):
    """El veredicto de la puerta: el overlay mejora, empeora o **no se puede evaluar**."""

    IMPROVED = "improved"
    WORSE = "worse"
    NOT_EVALUABLE = "not_evaluable"


class OverlayEvaluation(BaseModel):
    """El informe de la evaluación: el veredicto y su porqué, sin adornos.

    ``brier_*`` y ``sharpe_*`` son ``None`` cuando no se pudieron medir (nunca un 0 de relleno): un
    valor que no se midió **no** se escribe como cero.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: OverlayVerdict
    reasons: tuple[str, ...] = ()
    news_archive_available: bool
    brier_with_overlay: float | None = None
    brier_without_overlay: float | None = None
    sharpe_with_overlay: float | None = None
    sharpe_without_overlay: float | None = None
    paper_trading_design: str = PAPER_TRADING_DESIGN


def evaluate_overlay(
    *,
    news_archive_available: bool,
    brier_with_overlay: float | None = None,
    brier_without_overlay: float | None = None,
    sharpe_with_overlay: float | None = None,
    sharpe_without_overlay: float | None = None,
) -> OverlayEvaluation:
    """Determina el veredicto a partir de la evidencia disponible, sin inventar nada.

    Sin archivo histórico de noticias **no hay** medición pareada: el veredicto es
    ``not_evaluable`` y los números quedan ``None``. Con archivo, se exige la comparación de
    **Brier** (menor es mejor); el Sharpe OOS se publica además si está.
    """
    if not news_archive_available:
        return OverlayEvaluation(
            verdict=OverlayVerdict.NOT_EVALUABLE,
            reasons=(
                "el backtest corre siempre sin overlay: no hay archivo historico de noticias con "
                "published_at fiable (tech_stack.md §4.9)",
                "sin medicion pareada (con/sin overlay) no hay Brier ni Sharpe OOS que comparar: "
                "no se inventa ningun numero",
                "consecuencia declarada de antemano: el LLM queda solo como redactor y la "
                "evidencia se traslada al paper trading de la Fase 4 (#45)",
            ),
            news_archive_available=False,
        )
    if brier_with_overlay is None or brier_without_overlay is None:
        return OverlayEvaluation(
            verdict=OverlayVerdict.NOT_EVALUABLE,
            reasons=(
                "hay archivo, pero falta la medicion pareada de Brier: el overlay no se evalua a "
                "medias",
            ),
            news_archive_available=True,
            sharpe_with_overlay=sharpe_with_overlay,
            sharpe_without_overlay=sharpe_without_overlay,
        )
    verdict = (
        OverlayVerdict.IMPROVED
        if brier_with_overlay < brier_without_overlay
        else OverlayVerdict.WORSE
    )
    return OverlayEvaluation(
        verdict=verdict,
        reasons=(
            f"Brier con overlay {brier_with_overlay:.6f} frente a "
            f"{brier_without_overlay:.6f} sin overlay (menor es mejor)",
        ),
        news_archive_available=True,
        brier_with_overlay=brier_with_overlay,
        brier_without_overlay=brier_without_overlay,
        sharpe_with_overlay=sharpe_with_overlay,
        sharpe_without_overlay=sharpe_without_overlay,
    )


def render_markdown(evaluation: OverlayEvaluation, *, as_of: str) -> str:
    """El informe en markdown, con el veredicto **literal** y su consecuencia."""

    def _num(value: float | None) -> str:
        return "no medido" if value is None else f"{value:.6f}"

    lines = [
        "# Evaluacion incremental del overlay LLM (tarea #38)",
        "",
        f"- **Fecha:** {as_of}",
        f"- **Veredicto de la puerta de la Fase 3:** `{evaluation.verdict.value}`",
        "- **Archivo historico de noticias disponible:** "
        f"{'si' if evaluation.news_archive_available else 'no'}",
        "",
        "## Brier y Sharpe OOS, con y sin overlay",
        "",
        "| Metrica | Con overlay | Sin overlay |",
        "|---|---|---|",
        f"| Brier score | {_num(evaluation.brier_with_overlay)} | "
        f"{_num(evaluation.brier_without_overlay)} |",
        f"| Sharpe OOS | {_num(evaluation.sharpe_with_overlay)} | "
        f"{_num(evaluation.sharpe_without_overlay)} |",
        "",
        "## Motivos",
        "",
    ]
    lines.extend(f"- {reason}" for reason in evaluation.reasons)
    lines.extend(
        [
            "",
            "## Metodo alternativo de evaluacion",
            "",
            evaluation.paper_trading_design,
            "",
            "## Valla de honestidad",
            "",
            "- No hay edge demostrado. La ejecucion es manual y esto es apoyo a la decision,",
            "  no una estrategia validada.",
            "- `not_evaluable` es un veredicto valido de la puerta, no un fracaso que maquillar.",
            "",
        ]
    )
    return "\n".join(lines)
