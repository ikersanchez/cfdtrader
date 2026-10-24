"""Expectativas macro de #75: retirada explicita de dos salidas que no se pueden construir.

La decision vive en `_docs/plan.md` §7.1 y en su registro de cambios. Este fichero la blinda:
el `MacroAgent` no promete `Δfed_funds_esperado` ni `sorpresa_macro_hoy`, el observado
`fed_funds_chg_5` sigue en `macro_v1` y la nota declara por que no se ingesta consenso.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from cfdtrader.features import store as feature_store

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PLAN_PATH: Final[Path] = REPO_ROOT / "_docs" / "plan.md"

REMOVED: Final[tuple[str, ...]] = ("Δfed_funds_esperado", "sorpresa_macro_hoy")


def _macro_agent_row(text: str) -> str:
    """La fila del catalogo de §7.1 dedicada al `MacroAgent`."""
    matches = re.findall(r"^\| \*\*MacroAgent\*\* \|.*\|$", text, flags=re.MULTILINE)
    assert len(matches) == 1, matches
    return matches[0]


def test_the_macro_agent_no_longer_promises_the_two_unbuilt_outputs() -> None:
    """§7.1: las dos salidas retiradas no estan en la fila del catalogo; las observadas si."""
    text = PLAN_PATH.read_text(encoding="utf-8")
    row = _macro_agent_row(text)
    for name in REMOVED:
        assert name not in row, name
    for kept in ("fed_funds", "ust_10y", "Δust_10y_5d", "pendiente_2s10s", "cpi_yoy", "dxy"):
        assert kept in row, kept
    # El observado si existe y es el que se promete en su lugar.
    assert "fed_funds_chg_5" in text


def test_the_decision_is_written_with_its_reason_and_its_scope() -> None:
    """La nota declara el motivo (sin consenso) y el camino si algun dia hay fuente."""
    text = PLAN_PATH.read_text(encoding="utf-8")
    note = "Alcance declarado del `MacroAgent` (#75)"
    assert note in text
    section = text.split(note, 1)[1].split("**Nota específica del S&P 500:**", 1)[0]
    assert "#75" in section
    assert "consenso" in section
    assert "macro_v1" in section and "#22" in section
    assert "macro_v2" in section
    assert "point-in-time" in section
    assert "no se promete" in section


def test_the_observed_series_stays_in_the_macro_catalog() -> None:
    """La retirada de la expectativa no toca el catalogo observado de `macro_v1`."""
    assert "fed_funds_chg_5" in feature_store.MACRO_FEATURE_COLUMNS
    assert "fed_funds" in feature_store.MACRO_FEATURE_COLUMNS
    for name in REMOVED:
        assert name not in feature_store.MACRO_FEATURE_COLUMNS, name
