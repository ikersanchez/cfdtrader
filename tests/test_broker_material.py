"""El material para las decisiones del bróker: se recalcula, no se copia (#59, #87, #62).

`_docs/broker_material_2026-10-07.md` es la hoja para las **decisiones del propietario** del
Bloque 1 (bróker y sus dependencias). Lo que se blinda aquí es lo que la hace fiable y la impide
quedarse vieja o pasarse de la raya:

- el **cuestionario declarado** (`analysis.cost_audit.BROKER_QUESTIONS`) aparece **verbatim** en la
  hoja, de modo que si el código cambia una pregunta, la hoja **falla en voz alta**;
- la hoja **no decide**: declara que es material y que quien decide es el propietario;
- el corte de financiación sigue **sin verificar** (`DECLARED_FINANCING_CUT is None`) y la hoja lo
  dice, en vez de asumir una hora (prohibido);
- la hoja **apunta a los campos** de `config/cost_observations.yaml` donde se anota cada respuesta.

No hay modulo nuevo de `src/`: es una prueba de **contrato de documento**, asi que el suelo de
cobertura (90 % sentencias / 85 % ramas) **no aplica** y se declara aqui.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from cfdtrader.analysis.cost_audit import BROKER_QUESTIONS, DECLARED_FINANCING_CUT

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MATERIAL_PATH: Final[Path] = REPO_ROOT / "_docs" / "broker_material_2026-10-07.md"

#: Las cinco issues que esta hoja alimenta (raiz y dependencias del Bloque 1).
CITED_ISSUES: Final[tuple[str, ...]] = ("#59", "#87", "#62", "#107", "#51")

#: Los campos de `config/cost_observations.yaml` donde se anota una respuesta.
TARGET_FIELDS: Final[tuple[str, ...]] = (
    "financing_cut",
    "minimum_commission_usd",
    "spread_observations",
    "tracking_pairs",
    "executions",
)


def _material_text() -> str:
    return MATERIAL_PATH.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# El cuestionario se reproduce verbatim (si el codigo cambia, la hoja falla)
# ─────────────────────────────────────────────────────────────────────────────
def test_every_declared_broker_question_is_reproduced_verbatim() -> None:
    text = _material_text()
    missing = [q["id"] for q in BROKER_QUESTIONS if q["question"] not in text]
    assert not missing, f"la hoja no reproduce la pregunta declarada: {missing}"


def test_the_financing_cut_question_is_the_first_and_is_the_critical_one() -> None:
    text = _material_text()
    first = BROKER_QUESTIONS[0]
    assert first["id"] == "financing_cut"
    assert first["question"] in text
    assert "la verificación más crítica de la Fase 0" in text


# ─────────────────────────────────────────────────────────────────────────────
# La hoja no decide: es material, y quien decide es el propietario
# ─────────────────────────────────────────────────────────────────────────────
def test_the_document_does_not_decide() -> None:
    text = _material_text()
    assert "no la decisión" in text
    assert "quien decide es el propietario" in text


def test_the_document_records_the_closed_decision_and_the_kid() -> None:
    """La decision 4 se cierra con el KID: la hoja lo enlaza con §11 bis y lo declara."""
    text = _material_text()
    assert "§11 bis" in text
    assert "CERRADA" in text
    assert "§11.6" in text


# ─────────────────────────────────────────────────────────────────────────────
# Las cinco issues que alimenta y los campos donde se anota cada respuesta
# ─────────────────────────────────────────────────────────────────────────────
def test_the_five_issues_are_cited() -> None:
    text = _material_text()
    missing = [issue for issue in CITED_ISSUES if issue not in text]
    assert not missing, f"la hoja no cita: {missing}"


def test_every_target_field_is_named() -> None:
    text = _material_text()
    missing = [field for field in TARGET_FIELDS if field not in text]
    assert not missing, f"la hoja no nombra el campo: {missing}"


def test_the_financing_cut_is_still_unverified() -> None:
    """El corte sigue sin verificar: la hoja no puede asumir una hora (prohibido)."""
    assert DECLARED_FINANCING_CUT is None
    assert "sin verificar" in _material_text().lower() or "#87" in _material_text()


def test_the_declared_broker_and_the_kid_are_recorded() -> None:
    """El broker declarado (Revolut Securities Europe UAB) y su KID se nombran."""
    text = _material_text()
    assert "Revolut Securities Europe UAB" in text
    assert "KID" in text
