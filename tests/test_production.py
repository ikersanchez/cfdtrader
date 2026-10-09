"""Tests del contrato de #47: la puesta en produccion con tamano minimo.

El modulo declara las reglas (solo tier A, tamano minimo, valla activa, cierre obligatorio, sin
automatismo) y compone la **tarjeta de operacion** del dia leyendo la pista del diario. Nada de esto
afirma un *edge*: la tarjeta es el procedimiento, y «no operar» es un resultado legitimo con motivo.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Final

import pytest

from cfdtrader.delivery import production
from cfdtrader.delivery.production import (
    AUTHORIZED_TIERS,
    CHECKLIST,
    COMMANDS,
    HONESTY_FENCE,
    PRODUCTION_RULES,
    SIZE_POLICY,
    TASK,
    commands_for,
    main,
    operating_card,
    render_card,
)
from cfdtrader.journal.decision_log import Journal

SESSION: Final[date] = date(2026, 10, 9)
ZERO_SHA: Final[str] = "sha256:" + "0" * 64


def _row(*, status: str = "recommendation", direction: object = "long", tier: object = "A") -> dict:
    """Una fila completa de ``journal.decisions`` con lo que la tarjeta necesita."""
    return {
        "trade_date": SESSION.isoformat(),
        "as_of": "2026-10-09T12:45:00+00:00",
        "status": status,
        "features_version": ZERO_SHA,
        "model_version": "1" * 64,
        "prompt_hashes": {},
        "git_commit": "2" * 40,
        "prob_up_raw": 0.61,
        "prob_up_calibrated": 0.61,
        "expected_move_pct": 0.55,
        "cost_pct": 0.0042,
        "ev_net_pct": None,
        "direction": direction,
        "stop_pct": 0.5463,
        "target_pct": 1.0926,
        "size_notional_eur": None,
        "size_fraction": None,
        "leverage_implied": 1.8305,
        "tier": tier,
        "blocking_events": [],
        "bull_case": [],
        "bear_case": [],
        "llm_overlay": None,
        "report_text": "informe de prueba",
    }


def test_47_the_module_exports_the_declared_contract() -> None:
    assert set(production.__all__) == {
        "AUTHORIZED_TIERS",
        "CHECKLIST",
        "COMMANDS",
        "HONESTY_FENCE",
        "MODULE",
        "PRODUCTION_RULES",
        "SIZE_POLICY",
        "TASK",
        "OperatingCard",
        "ProductionError",
        "commands_for",
        "main",
        "operating_card",
        "render_card",
    }
    assert TASK == "#47"
    assert AUTHORIZED_TIERS == ("A",)


def test_47_the_rules_cover_the_pieces_of_the_lane() -> None:
    assert [rule["id"] for rule in PRODUCTION_RULES] == [
        "solo_tier_a",
        "tamano_minimo",
        "valla_de_cartera",
        "cierre_obligatorio",
        "una_operacion",
        "sin_automatismo",
    ]
    assert all(rule["rule"] and rule["statement"] for rule in PRODUCTION_RULES)
    # El tamano es un techo, no un objetivo, y lo dice el propio texto.
    assert "techo" in SIZE_POLICY


def test_47_the_checklist_covers_the_day_from_the_ingest_to_the_fence() -> None:
    text = "\n".join(CHECKLIST)
    for anchor in ("08:00", "08:45", "09:30", "15:45", "16:00", "16:15", "16:30"):
        assert anchor in text
    # La verificacion de las 15:45 no tiene alarma: es del operador, por diseno.
    assert "no hay alarma" in text


def test_47_the_commands_are_declared_with_their_anchor() -> None:
    ids = [command["id"] for command in COMMANDS]
    assert ids == ["camino_diario", "billete", "registro", "valla", "veredicto_paper"]
    assert all("uv run python -m" in command["command"] for command in COMMANDS)
    filled = commands_for(journal_root="mi_diario", session=SESSION)
    assert "mi_diario" in filled[1]["command"]
    assert SESSION.isoformat() in filled[1]["command"]


def test_47_without_a_pista_the_session_is_not_operated(tmp_path: Path) -> None:
    card = operating_card(tmp_path / "journal", SESSION)
    assert card.operable is False
    assert "sin pista registrada" in card.reason
    assert card.ticket is None


def test_47_a_nothing_session_is_not_operated(tmp_path: Path) -> None:
    journal_root = tmp_path / "journal"
    Journal(journal_root).write("decisions", _row(direction="nothing", tier="C"))
    card = operating_card(journal_root, SESSION)
    assert card.operable is False
    assert "no hay nada que operar" in card.reason


def test_47_a_tier_b_signal_is_not_operated(tmp_path: Path) -> None:
    """La regla 10: los tiers B y C se registran y **no** se operan."""
    journal_root = tmp_path / "journal"
    Journal(journal_root).write("decisions", _row(tier="B"))
    card = operating_card(journal_root, SESSION)
    assert card.operable is False
    assert "regla 10" in card.reason


def test_47_a_tier_a_signal_is_operated_with_its_ticket(tmp_path: Path) -> None:
    journal_root = tmp_path / "journal"
    Journal(journal_root).write("decisions", _row())
    card = operating_card(journal_root, SESSION)
    assert card.operable is True
    assert card.direction == "long" and card.tier == "A"
    assert card.notional_usd == Decimal("18305.0000")
    ticket = card.ticket
    assert ticket is not None
    # El billete de #84 viaja entero y con su render; el techo de nocional es el del gate.
    assert ticket["stop_pct"] == "0.5463"
    assert "Billete de ejecucion (#84)" in str(ticket["rendered"])
    # La tarjeta no promete nada: el tamano es un techo, no un objetivo.
    assert "menor tamano admisible" in card.reason


def test_47_the_card_renders_the_rules_the_steps_the_commands_and_the_fence(tmp_path: Path) -> None:
    journal_root = tmp_path / "journal"
    Journal(journal_root).write("decisions", _row())
    text = render_card(operating_card(journal_root, SESSION))
    assert "Tarjeta de operacion (#47)" in text
    assert "**si**" in text
    assert "solo_tier_a" in text
    assert "Comandos" in text
    assert "no hay edge demostrado" in text


def test_47_the_cli_prints_the_card(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    journal_root = tmp_path / "journal"
    Journal(journal_root).write("decisions", _row())
    code = main(["--journal-root", str(journal_root), "--session", SESSION.isoformat()])
    captured = capsys.readouterr()
    assert code == 0
    assert "Tarjeta de operacion (#47)" in captured.out
    assert main(["--journal-root", str(journal_root), "--session", "no-es-fecha"]) == 2


def test_47_the_honesty_fence_says_there_is_no_edge() -> None:
    text = " ".join(HONESTY_FENCE)
    assert "no hay edge demostrado" in text
    assert "negativo" in text
    assert "manual" in text
