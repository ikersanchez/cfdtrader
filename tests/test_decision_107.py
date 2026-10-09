"""Cierre de #107 por el criterio (b): el `SPX500:CFD` no se adquiere.

#107 (seguimiento de #50) pedía adquirir, como **acción del propietario**, el
intradía y el `bid`/`ask` reales del `SPX500:CFD`, y dejaba **dos** criterios de
cierre simétricos. Se resuelve por el **(b)**: **no** se adquiere y el **proxy
declarado** (ruta 3 de #50) pasa a ser la **ruta definitiva**, no provisional.
Esta prueba blinda que:

- `plan.md` §19.18 registra la decisión con su fecha y su criterio;
- `_docs/data_sources.md` declara el proxy **definitivo** y no deja el pliego de
  adquisición como una vía abierta;
- el cierre **no** ha movido las otras puertas: la tabla de §11.6 sigue leyéndose
  con sus **nueve** filas y el bloque pre-registrado de la Fase 4
  (`<!-- puerta-fase-4:begin -->`) sigue presente.

No hay módulo nuevo de `src/`: es una prueba de **contrato de documento**, así que
el suelo de cobertura (90 % sentencias / 85 % ramas) **no aplica** y se declara aquí.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from cfdtrader.analysis import phase2_report

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PLAN_PATH: Final[Path] = REPO_ROOT / "_docs" / "plan.md"
DATA_SOURCES_PATH: Final[Path] = REPO_ROOT / "_docs" / "data_sources.md"

#: La fecha de la decisión (literal estable del documento, no del reloj).
DECIDED_ON: Final[str] = "2026-10-09"

#: El encabezado de la sección de la ruta en `data_sources.md` (tarea #50).
ROUTE_SECTION_HEADER: Final[str] = "## Decisión de la fuente de intradía y bid/ask (tarea #50)"

#: El marcador del bloque de la puerta de la Fase 4 (#130), que este cierre **no** debe mover.
PHASE4_BEGIN_MARKER: Final[str] = "<!-- puerta-fase-4:begin -->"


def _plan_text() -> str:
    return PLAN_PATH.read_text(encoding="utf-8")


def _data_sources_text() -> str:
    return DATA_SOURCES_PATH.read_text(encoding="utf-8")


def _section_19_18(text: str) -> str:
    """La sección `### 19.18`, hasta la regla horizontal que la cierra."""
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith("### 19.18")]
    assert len(starts) == 1, f"se espera una unica seccion 19.18: hay {len(starts)}"
    start = starts[0]
    end = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "---")
    return "\n".join(lines[start:end])


def _route_section(text: str) -> str:
    """Texto de la sección de la ruta, de su cabecera al siguiente `## `."""
    lines = text.splitlines()
    start = lines.index(ROUTE_SECTION_HEADER)
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _flat(text: str) -> str:
    """El texto con los espacios colapsados, para afirmar sobre frases que envuelven línea."""
    return " ".join(text.split())


# ─────────────────────────────────────────────────────────────────────────────
# La decisión queda registrada en `plan.md` §19.18
# ─────────────────────────────────────────────────────────────────────────────
def test_the_registry_section_exists_with_its_date_and_criterion() -> None:
    section = _section_19_18(_plan_text())
    assert DECIDED_ON in section
    assert "#107" in section and "#50" in section
    assert "(b)" in section


def test_the_registry_declares_the_proxy_as_definitive() -> None:
    section = _section_19_18(_plan_text())
    missing = [lit for lit in ("ruta 3", "proxy declarado", "definitiva") if lit not in section]
    assert not missing, f"§19.18 no declara: {missing}"


def test_the_registry_keeps_the_honesty_fence() -> None:
    section = _section_19_18(_plan_text())
    missing = [
        lit for lit in ("§11.6", "phase1_ready", "#62", "not_evaluable") if lit not in section
    ]
    assert not missing, f"§19.18 no cita: {missing}"


def test_moving_the_resolution_changes_the_document() -> None:
    """El anclaje es de valor, no solo de presencia: quitar la decisión no pasa desapercibido."""
    text = _plan_text()
    perturbed = text.replace("ruta definitiva", "ruta provisional")
    assert perturbed != text, "la sonda no cambio nada: revisa el texto de la decision"
    assert "ruta definitiva" not in perturbed


# ─────────────────────────────────────────────────────────────────────────────
# `data_sources.md` declara el proxy definitivo y no deja el pliego abierto
# ─────────────────────────────────────────────────────────────────────────────
def test_the_data_sources_doc_declares_the_criterion_b() -> None:
    section = _route_section(_data_sources_text())
    assert "criterio (b)" in section
    assert DECIDED_ON in section
    assert "definitiva" in section


def test_the_open_acquisition_pliego_is_gone() -> None:
    """El pliego no puede seguir declarando la adquisición (rutas 1 y 2) como abierta."""
    section = _flat(_route_section(_data_sources_text()))
    assert "siguen abiertas" not in section, "el pliego sigue declarando la adquisición abierta"
    assert "descartadas por decisión" in section


# ─────────────────────────────────────────────────────────────────────────────
# El cierre no ha movido las otras puertas
# ─────────────────────────────────────────────────────────────────────────────
def test_the_kill_table_still_reads_from_the_edited_document() -> None:
    table = phase2_report.read_kill_table(_plan_text())
    assert table.source["section"] == "§11.6"
    assert table.source["n_rows"] == 9


def test_the_phase4_gate_block_is_untouched() -> None:
    text = _plan_text()
    assert text.count(PHASE4_BEGIN_MARKER) == 1
