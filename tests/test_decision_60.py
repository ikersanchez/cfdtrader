"""Cierre de la decisión abierta 5 (umbrales y tamaño de `R`) — tarea #60.

`tech_stack.md` §11 bis lista ocho decisiones que el documento no puede cerrar solo. La
**decisión 5** —«umbrales concretos (EV mínimo, riesgo por operación, pérdidas máximas, tamaño de
`R`)»— era la que **#9 no podía suplir** y la que **#131** marcó como **no diferible**: sin `R` el
`EV` neto es `null`, la regla 9 bloquea (`ev_neto_no_calculable`) y la Fase 4 no puede observar.
Esta prueba blinda que:

- los cuatro valores están **declarados** en `plan.md` (§12 y §19.12) y en `tech_stack.md` §11 bis;
- la **decisión 5** está marcada como **CERRADA**, con fecha y motivo, y la fila antigua
  («pendiente de **#60**») ha desaparecido;
- el cierre **no** ha movido las otras puertas: la tabla de §11.6 sigue leyéndose con sus **nueve**
  filas y el bloque pre-registrado de la Fase 4 (`<!-- puerta-fase-4:begin -->`) sigue presente.

No hay módulo nuevo de `src/`: es una prueba de **contrato de documento**, así que el suelo de
cobertura (90 % sentencias / 85 % ramas) **no aplica** y se declara aquí.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from cfdtrader.analysis import phase2_report

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PLAN_PATH: Final[Path] = REPO_ROOT / "_docs" / "plan.md"
TECH_STACK_PATH: Final[Path] = REPO_ROOT / "_docs" / "tech_stack.md"

#: El signo menos del documento es U+2212; se construye por escape para no disparar RUF001.
MINUS: Final[str] = "\u2212"

#: Los cuatro valores (decisión abierta 5, #60): `(literal en plan.md, literal en tech_stack.md)`.
DECIDED_VALUES: Final[dict[str, tuple[str, str]]] = {
    "R": ("1,00 % del nocional", "1,00 %"),
    "riesgo_por_operacion": ("1 % del capital", "1 %"),
    "perdida_diaria": (f"{MINUS}2 %", f"{MINUS}2 %"),
    "ev_minimo": ("2 × c", "2 × c"),
}

#: La fecha de la decisión (literal estable del documento, no del reloj).
DECIDED_ON: Final[str] = "2026-10-04"

#: El marcador del bloque de la puerta de la Fase 4 (#130), que este cierre **no** debe mover.
PHASE4_BEGIN_MARKER: Final[str] = "<!-- puerta-fase-4:begin -->"


def _plan_text() -> str:
    return PLAN_PATH.read_text(encoding="utf-8")


def _tech_stack_text() -> str:
    return TECH_STACK_PATH.read_text(encoding="utf-8")


def _section_19_12(text: str) -> str:
    """La sección `### 19.12`, hasta la regla horizontal que la cierra."""
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith("### 19.12")]
    assert len(starts) == 1, f"se espera una unica seccion 19.12: hay {len(starts)}"
    start = starts[0]
    end = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "---")
    return "\n".join(lines[start:end])


def _closing_box(text: str) -> str:
    """El bloque «Umbrales DECIDIDOS» del §12 (la cita que abre con `> **Umbrales DECIDIDOS`)."""
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith("> **Umbrales DECIDIDOS")]
    assert len(starts) == 1, f"se espera un unico bloque «Umbrales DECIDIDOS»: hay {len(starts)}"
    start = starts[0]
    end = next(i for i in range(start + 1, len(lines)) if not lines[i].startswith(">"))
    return "\n".join(lines[start:end])


def _decision_5_row(text: str) -> str:
    """La fila 5 de la tabla de §11 bis, por su texto, sin depender del número de línea."""
    rows = [
        line
        for line in text.splitlines()
        if line.startswith("| 5 |") and "Umbrales concretos" in line
    ]
    assert len(rows) == 1, f"se espera una unica fila 5 de §11 bis: hay {len(rows)}"
    return rows[0]


def _section_11_bis(text: str) -> str:
    """La sección `## 11 bis`, hasta el siguiente encabezado de nivel dos."""
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith("## 11 bis")]
    assert len(starts) == 1, f"se espera una unica seccion 11 bis: hay {len(starts)}"
    start = starts[0]
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## "))
    return "\n".join(lines[start:end])


# ─────────────────────────────────────────────────────────────────────────────
# Los cuatro valores quedan declarados en el documento
# ─────────────────────────────────────────────────────────────────────────────
def test_the_four_values_are_declared_in_section_12() -> None:
    box = _closing_box(_plan_text())
    missing = [name for name, (plan_needle, _) in DECIDED_VALUES.items() if plan_needle not in box]
    assert not missing, f"el bloque de §12 no declara: {missing}"


def test_the_four_values_are_declared_in_section_19_12() -> None:
    section = _section_19_12(_plan_text())
    missing = [
        name for name, (plan_needle, _) in DECIDED_VALUES.items() if plan_needle not in section
    ]
    assert not missing, f"§19.12 no declara: {missing}"


def test_the_registry_carries_the_date_and_the_motive() -> None:
    section = _section_19_12(_plan_text())
    assert DECIDED_ON in section
    assert "#60" in section and "#131" in section


def test_moving_a_value_changes_the_document() -> None:
    """El anclaje es de valor, no solo de presencia: bajar `R` no puede pasar desapercibido."""
    text = _plan_text()
    perturbed = text.replace("1,00 % del nocional", "0,50 % del nocional")
    assert perturbed != text, "la sonda no cambio nada: revisa el texto de la decision"
    assert "1,00 % del nocional" not in perturbed


# ─────────────────────────────────────────────────────────────────────────────
# La decisión 5 está cerrada en `tech_stack.md` §11 bis
# ─────────────────────────────────────────────────────────────────────────────
def test_decision_5_is_closed_with_its_values() -> None:
    row = _decision_5_row(_tech_stack_text())
    assert "CERRADA" in row and DECIDED_ON in row, row
    missing = [name for name, (_, tech_needle) in DECIDED_VALUES.items() if tech_needle not in row]
    assert not missing, f"la fila 5 no declara: {missing}"


def test_the_stale_open_marker_is_gone() -> None:
    """La fila antigua («pendiente de **#60**») tiene que haber desaparecido, no convivir."""
    section = _section_11_bis(_tech_stack_text())
    assert "pendiente de **#60**" not in section, (
        "§11 bis sigue marcando la decision 5 como abierta"
    )


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
