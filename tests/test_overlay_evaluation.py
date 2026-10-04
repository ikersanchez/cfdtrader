"""Tests de la evaluación incremental del overlay (#38, puerta de la Fase 3).

El artefacto verificable es el **veredicto**: sin archivo histórico de noticias la medición pareada
no existe, así que el veredicto es `not_evaluable` y los números quedan `None` (nunca un 0). Con
medición, se compara de verdad. Ninguna prueba abre red ni consulta el reloj.
"""

from __future__ import annotations

from pathlib import Path

from cfdtrader.analysis import overlay_evaluation as module
from cfdtrader.analysis.overlay_evaluation import (
    OverlayVerdict,
    evaluate_overlay,
    render_markdown,
)


def test_38_without_a_news_archive_the_verdict_is_not_evaluable() -> None:
    """Sin archivo histórico no hay medición pareada: `not_evaluable` y ningún número inventado."""
    evaluation = evaluate_overlay(news_archive_available=False)

    assert evaluation.verdict is OverlayVerdict.NOT_EVALUABLE
    assert evaluation.brier_with_overlay is None
    assert evaluation.brier_without_overlay is None
    assert evaluation.sharpe_with_overlay is None
    assert any("§4.9" in reason for reason in evaluation.reasons)
    assert any("#45" in reason for reason in evaluation.reasons)
    # La consecuencia declarada de antemano: el LLM queda como redactor.
    assert any("redactor" in reason for reason in evaluation.reasons)


def test_38_with_an_archive_but_no_paired_measurement_is_still_not_evaluable() -> None:
    """Tener archivo no basta sin la medición pareada: no se evalúa a medias."""
    evaluation = evaluate_overlay(news_archive_available=True, sharpe_with_overlay=1.0)

    assert evaluation.verdict is OverlayVerdict.NOT_EVALUABLE
    assert evaluation.brier_with_overlay is None
    assert evaluation.sharpe_with_overlay == 1.0


def test_38_a_lower_brier_with_the_overlay_is_an_improvement() -> None:
    """Con medición real, menor Brier gana; mayor Brier, pierde."""
    better = evaluate_overlay(
        news_archive_available=True, brier_with_overlay=0.24, brier_without_overlay=0.25
    )
    worse = evaluate_overlay(
        news_archive_available=True, brier_with_overlay=0.26, brier_without_overlay=0.25
    )

    assert better.verdict is OverlayVerdict.IMPROVED
    assert worse.verdict is OverlayVerdict.WORSE


def test_38_the_report_prints_the_verdict_literally_and_never_zero_for_unmeasured() -> None:
    """El informe escribe `not_evaluable` tal cual y `no medido` (nunca un 0 de relleno)."""
    markdown = render_markdown(evaluate_overlay(news_archive_available=False), as_of="2026-10-04")

    assert "`not_evaluable`" in markdown
    assert "no medido" in markdown
    assert "no hay edge demostrado" in markdown.lower() or "No hay edge demostrado" in markdown
    assert "#45" in markdown


def test_38_the_module_has_no_clock_and_no_network() -> None:
    """El módulo decide sin reloj y sin red."""
    source = Path(module.__file__).read_text(encoding="utf-8")
    for forbidden in ("datetime.now", "date.today", "time.time"):
        assert forbidden not in source, forbidden
    for network in ("import yfinance", "import requests", "import urllib", "import openai"):
        assert network not in source, network
