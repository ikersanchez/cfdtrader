"""Tests del contrato de #84: el *bracket* contra el broker real y el registro de la sesion.

El modulo compone el **billete** de ejecucion de una sesion, declara el protocolo de fallo y escribe
la fila **real** en `journal.trades` (§12.5). Las pruebas son puras y sobre `tmp_path` (la sesion de
tests no toca `data/` ni `runs/`, lo blinda `tests/conftest.py`); el cotejo de la aritmetica de
precios se hace **contra la del gate**, que es quien la define (#27, A23).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final

import pytest

from cfdtrader.analysis import paper_trading, portfolio_rules
from cfdtrader.backtest.engine import Direction
from cfdtrader.decision.gate import GateOutput, GateStatus
from cfdtrader.delivery import bracket
from cfdtrader.delivery.bracket import (
    ANCHORS,
    EXIT_REASONS,
    INCIDENTS,
    MODULE,
    REPORT_DOES_NOT_DO,
    TASK,
    BracketError,
    ExecutionFacts,
    execution_pnl_pct,
    main,
    read_trade,
    record_trade,
    render_ticket,
    ticket,
    ticket_from_output,
)
from cfdtrader.journal.decision_log import TABLE_COLUMNS, Journal

SESSION: Final[date] = date(2026, 10, 9)
ENTRY_TIME: Final[datetime] = datetime(2026, 10, 9, 13, 30, tzinfo=UTC)
EXIT_TIME: Final[datetime] = datetime(2026, 10, 9, 20, 0, tzinfo=UTC)
ZERO_SHA: Final[str] = "sha256:" + "0" * 64


def _decision_row(
    *,
    direction: str | None = "long",
    stop_pct: object = 0.5463,
    target_pct: object = 1.0926,
    leverage_implied: object = 1.8305,
    tier: object = "A",
) -> dict[str, object]:
    """Una fila completa de ``journal.decisions`` (las 24 columnas del esquema de #39)."""
    return {
        "trade_date": SESSION.isoformat(),
        "as_of": "2026-10-09T12:45:00+00:00",
        "status": "recommendation",
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
        "stop_pct": stop_pct,
        "target_pct": target_pct,
        "size_notional_eur": None,
        "size_fraction": None,
        "leverage_implied": leverage_implied,
        "tier": tier,
        "blocking_events": [],
        "bull_case": [],
        "bear_case": [],
        "llm_overlay": None,
        "report_text": "informe de prueba",
    }


def _facts(
    *,
    direction: str = "long",
    entry_px: float = 5000.0,
    exit_px: float = 4975.0,
    costs_pct: Decimal = Decimal("0.0042"),
    exit_reason: str = "stop",
    closed_by_close: bool = True,
    notional: Decimal = Decimal("18305"),
) -> ExecutionFacts:
    return ExecutionFacts(
        session=SESSION,
        direction=direction,
        entry_px=entry_px,
        exit_px=exit_px,
        notional=notional,
        costs_pct=costs_pct,
        exit_reason=exit_reason,
        entry_time=ENTRY_TIME,
        exit_time=EXIT_TIME,
        closed_by_close=closed_by_close,
    )


# ─────────────────────────────────────────────────────────────────────────────
# A1 · El modulo y su contrato declarado
# ─────────────────────────────────────────────────────────────────────────────
def test_84_the_module_exports_the_declared_contract() -> None:
    assert set(bracket.__all__) == {
        "ANCHORS",
        "BRACKET_LEGS",
        "EXIT_REASONS",
        "INCIDENTS",
        "MODULE",
        "REPORT_DOES_NOT_DO",
        "TASK",
        "BracketError",
        "BracketTicket",
        "ExecutionFacts",
        "execution_pnl_pct",
        "main",
        "read_trade",
        "record_trade",
        "render_ticket",
        "ticket",
    }
    assert TASK == "#84"
    assert MODULE == "cfdtrader.delivery.bracket"
    assert bracket.BRACKET_LEGS == ("stop", "objetivo")


def test_84_the_exit_reasons_are_the_ones_of_the_recomputation() -> None:
    """Un solo vocabulario: el que #45 usa al recomputar el resultado de la sesion."""
    assert tuple(EXIT_REASONS) == (
        paper_trading.EXIT_TARGET,
        paper_trading.EXIT_STOP,
        paper_trading.EXIT_CLOSE,
    )


def test_84_the_steps_cover_the_flow_of_section_13() -> None:
    et = [anchor["et"] for anchor in ANCHORS]
    assert et == [
        "09:00",
        "09:20-09:30",
        "09:30",
        "09:30 (inmediatamente)",
        "15:45",
        "16:00",
        "16:15",
    ]
    assert all(anchor["madrid"] and anchor["step"] and anchor["action"] for anchor in ANCHORS)
    # El cierre a las 16:00 ET y la verificacion de las 15:45 estan los dos, y el segundo dice que
    # **no hay alarma**: es la decision de diseno de §12 regla 16, no un olvido.
    assert "16:00" in et
    assert any("No hay alarma" in anchor["action"] for anchor in ANCHORS)


def test_84_the_failure_protocol_is_declared_and_none_of_its_modes_is_silent() -> None:
    ids = [incident["id"] for incident in INCIDENTS]
    assert ids == [
        "broker_rechaza_el_bracket",
        "una_sola_pata",
        "sin_relleno",
        "paso_la_noche",
        "salto_el_bracket",
    ]
    for incident in INCIDENTS:
        assert incident["when"] and incident["action"] and incident["record"]
    # El incumplimiento de las 16:00 ET se registra con `closed_by_close = false`.
    overnight = next(item for item in INCIDENTS if item["id"] == "paso_la_noche")
    assert "closed_by_close = false" in overnight["record"]
    # Lo que no se opera no se maquilla con una fila: no hay trade que registrar.
    rejected = next(item for item in INCIDENTS if item["id"] == "broker_rechaza_el_bracket")
    assert "no se opera" in rejected["action"]


def test_84_the_module_declares_what_it_does_not_do() -> None:
    assert [entry["id"] for entry in REPORT_DOES_NOT_DO] == [
        "no_habla_con_el_broker",
        "no_decide_la_geometria",
    ]
    assert all(entry["statement"] for entry in REPORT_DOES_NOT_DO)


# ─────────────────────────────────────────────────────────────────────────────
# A2 · La aritmetica de precios es la del gate (no una segunda)
# ─────────────────────────────────────────────────────────────────────────────
class _Geometry:
    """Duck type de la geometria del gate: ``_barrier_prices`` solo lee estos tres campos."""

    def __init__(self, direction: Direction, stop_pct: Decimal, target_pct: Decimal) -> None:
        self.direction = direction
        self.stop_pct = stop_pct
        self.target_pct = target_pct


def test_84_the_price_arithmetic_matches_the_gate() -> None:
    """Si #27 cambia su convencion de barreras, este test cae: no hay dos aritmeticas."""
    from cfdtrader.decision.gate import (
        _barrier_prices as gate_barrier_prices,  # pyright: ignore[reportPrivateUsage]
    )

    for direction, entry in (
        (Direction.LONG, 5000.0),
        (Direction.SHORT, 5000.0),
        (Direction.LONG, 7777.5),
    ):
        stop_pct, target_pct = Decimal("1.00"), Decimal("2.00")
        geometry = _Geometry(direction, stop_pct, target_pct)
        expected_stop, expected_target = gate_barrier_prices(geometry, entry)  # type: ignore[arg-type]
        assert bracket._barrier_prices(  # pyright: ignore[reportPrivateUsage]
            direction=direction.value,
            stop_pct=stop_pct,
            target_pct=target_pct,
            entry_px=entry,
        ) == (expected_stop, expected_target)


def _prices(billete: bracket.BracketTicket) -> tuple[float, float, float]:
    """Los tres precios ya estrechados a `float`: el test comprueba antes que existen."""
    assert billete.entry_px is not None
    assert billete.stop_px is not None and billete.target_px is not None
    return billete.entry_px, billete.stop_px, billete.target_px


def test_84_a_long_bracket_puts_the_stop_below_and_the_target_above() -> None:
    billete = ticket(
        session=SESSION,
        direction="long",
        tier="A",
        notional_usd=Decimal("18305"),
        stop_pct=Decimal("1"),
        target_pct=Decimal("2"),
        entry_px=5000.0,
    )
    entry, stop, target = _prices(billete)
    assert billete.stop_px == pytest.approx(4950.0)
    assert billete.target_px == pytest.approx(5100.0)
    assert stop < entry < target


def test_84_a_short_bracket_is_the_mirror_image() -> None:
    billete = ticket(
        session=SESSION,
        direction="short",
        tier="A",
        notional_usd=Decimal("18305"),
        stop_pct=Decimal("1"),
        target_pct=Decimal("2"),
        entry_px=5000.0,
    )
    entry, stop, target = _prices(billete)
    assert target < entry < stop


def test_84_without_a_declared_entry_the_prices_are_none_never_invented() -> None:
    """La entrada es el relleno de la subasta: sin el, el billete publica los `%` y nada mas."""
    billete = ticket(
        session=SESSION,
        direction="long",
        tier="A",
        notional_usd=Decimal("18305"),
        stop_pct=Decimal("1"),
        target_pct=Decimal("2"),
    )
    assert billete.entry_px is None
    assert billete.stop_px is None and billete.target_px is None
    assert billete.payload()["stop_pct"] == "1"


# ─────────────────────────────────────────────────────────────────────────────
# A3 · Sin objetivo no hay billete: la regla 16 es "las dos o ninguna"
# ─────────────────────────────────────────────────────────────────────────────
def test_84_without_a_target_there_is_no_ticket() -> None:
    with pytest.raises(BracketError, match="no se opera"):
        ticket(
            session=SESSION,
            direction="long",
            tier="A",
            notional_usd=Decimal("18305"),
            stop_pct=Decimal("1"),
            target_pct=None,
        )


def test_84_a_non_directional_session_has_no_ticket() -> None:
    for direction in ("nothing", "", "LONG"):
        with pytest.raises(BracketError):
            ticket(
                session=SESSION,
                direction=direction,
                tier="C",
                notional_usd=Decimal("18305"),
                stop_pct=Decimal("1"),
                target_pct=Decimal("2"),
            )


def test_84_a_non_positive_notional_or_stop_is_a_typed_error() -> None:
    with pytest.raises(BracketError):
        ticket(
            session=SESSION,
            direction="long",
            tier="A",
            notional_usd=Decimal("0"),
            stop_pct=Decimal("1"),
            target_pct=Decimal("2"),
        )
    with pytest.raises(BracketError):
        ticket(
            session=SESSION,
            direction="long",
            tier="A",
            notional_usd=Decimal("18305"),
            stop_pct=Decimal("0"),
            target_pct=Decimal("2"),
        )


def _output(
    *, status: GateStatus, direction: Direction | None, notional: Decimal | None
) -> GateOutput:
    """Un `GateOutput` minimo con los campos que el billete lee (sin pasar por el pipeline)."""
    return GateOutput.model_construct(
        session=SESSION,
        as_of=datetime(2026, 10, 9, 12, 45, tzinfo=UTC),
        today=SESSION,
        status=status,
        direction=direction,
        tier="A",
        prob_up_calibrated=0.61,
        expected_move_pct=Decimal("0.55"),
        expected_move_basis="declarado",
        cost_pct=Decimal("0.0042"),
        stop_pct=Decimal("0.5463"),
        target_pct=Decimal("1.0926"),
        notional_usd=notional,
        leverage_implied=Decimal("1.8305"),
        gate_sha256=ZERO_SHA,
    )


def test_84_the_ticket_from_the_gate_carries_its_digest() -> None:
    billete = ticket_from_output(
        _output(
            status=GateStatus.RECOMMENDATION,
            direction=Direction.SHORT,
            notional=Decimal("18305"),
        ),
        entry_px=5000.0,
    )
    assert billete.direction == "short"
    assert billete.gate_sha256 == ZERO_SHA
    assert billete.source == "gate_sha256"
    # En corto el stop va por encima y el objetivo por debajo.
    _entry, stop, target = _prices(billete)
    assert target < 5000.0 < stop


def test_84_a_nothing_or_non_recommendation_output_has_no_ticket() -> None:
    with pytest.raises(BracketError, match="no operar"):
        ticket_from_output(
            _output(
                status=GateStatus.RECOMMENDATION,
                direction=Direction.NOTHING,
                notional=Decimal("18305"),
            )
        )
    with pytest.raises(BracketError, match="no operar"):
        ticket_from_output(
            _output(
                status=GateStatus.NO_RECOMMENDATION_STALE_DATA,
                direction=None,
                notional=None,
            )
        )
    with pytest.raises(BracketError):
        ticket_from_output(
            _output(
                status=GateStatus.RECOMMENDATION,
                direction=Direction.LONG,
                notional=None,
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# A4 · El retorno neto en `%` del nocional (el contrato que lee #83)
# ─────────────────────────────────────────────────────────────────────────────
def test_84_the_net_return_is_the_price_move_minus_the_effective_cost() -> None:
    assert execution_pnl_pct(_facts(direction="long")) == Decimal("100") * Decimal(
        repr((4975.0 - 5000.0) / 5000.0)
    ) - Decimal("0.0042")
    assert execution_pnl_pct(_facts(direction="long")) == pytest.approx(Decimal("-0.5042"))
    assert execution_pnl_pct(_facts(direction="short")) == pytest.approx(Decimal("0.4958"))


def test_84_an_invalid_entry_or_direction_is_a_typed_error() -> None:
    with pytest.raises(BracketError):
        execution_pnl_pct(_facts(entry_px=0.0))
    with pytest.raises(BracketError):
        execution_pnl_pct(_facts(direction="nothing"))


# ─────────────────────────────────────────────────────────────────────────────
# A5 · El registro real en `journal.trades` (§12.5), con su esquema
# ─────────────────────────────────────────────────────────────────────────────
def test_84_the_recorded_row_has_the_ten_columns_of_the_closed_schema(tmp_path: Path) -> None:
    journal_root = tmp_path / "journal"
    path = record_trade(journal_root, _facts())
    assert path.is_file()
    row = read_trade(journal_root, SESSION)
    assert set(row) == set(TABLE_COLUMNS["trades"])
    assert row["trade_date"] == SESSION.isoformat()
    assert row["pnl_pct"] == pytest.approx(-0.5042)
    assert row["costs_pct"] == pytest.approx(0.0042)
    assert row["exit_reason"] == "stop"
    assert row["closed_by_close"] is True


def test_84_an_overnight_breach_is_recorded_as_such(tmp_path: Path) -> None:
    """El incumplimiento de las 16:00 ET se registra: `closed_by_close = false`."""
    journal_root = tmp_path / "journal"
    record_trade(journal_root, _facts(exit_reason="close", closed_by_close=False))
    row = read_trade(journal_root, SESSION)
    assert row["closed_by_close"] is False


def test_84_the_journal_never_silently_overwrites_a_session(tmp_path: Path) -> None:
    journal_root = tmp_path / "journal"
    record_trade(journal_root, _facts())
    with pytest.raises(BracketError, match="no se puede registrar"):
        record_trade(journal_root, _facts(exit_px=4900.0))
    # Y la fila original sigue intacta.
    assert read_trade(journal_root, SESSION)["exit_price"] == pytest.approx(4975.0)


def test_84_an_invalid_reason_or_incoherent_times_is_a_typed_error(tmp_path: Path) -> None:
    journal_root = tmp_path / "journal"
    with pytest.raises(BracketError):
        record_trade(journal_root, _facts(exit_reason="inventado"))
    with pytest.raises(BracketError):
        record_trade(journal_root, _facts(direction="nothing"))
    with pytest.raises(BracketError):
        record_trade(journal_root, _facts(costs_pct=Decimal("-1")))
    assert not (journal_root / "trades").exists()


def test_84_a_missing_row_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(BracketError):
        read_trade(tmp_path / "journal", SESSION)


def test_84_the_written_row_is_what_the_portfolio_fence_reads(tmp_path: Path) -> None:
    """Cierra el circuito con #83: la fila real pasa a `%` del capital con apalancamiento."""
    journal_root = tmp_path / "journal"
    notional = Decimal("18305")
    record_trade(journal_root, _facts(notional=notional))
    ledger = portfolio_rules.closed_trades_from_journal(journal_root)
    assert ledger.skipped == ()
    (entry,) = ledger.trades
    assert entry.leverage == notional / Decimal("10000")
    assert entry.capital_pct == entry.notional_pct * entry.leverage
    totals = portfolio_rules.accumulate(ledger.trades, session=date(2026, 10, 10))
    assert totals.weekly_pnl_pct == entry.capital_pct


# ─────────────────────────────────────────────────────────────────────────────
# A6 · El billete impreso
# ─────────────────────────────────────────────────────────────────────────────
def test_84_the_ticket_lists_the_steps_the_protocol_and_the_prices() -> None:
    text = render_ticket(
        ticket(
            session=SESSION,
            direction="long",
            tier="A",
            notional_usd=Decimal("18305"),
            stop_pct=Decimal("0.5463"),
            target_pct=Decimal("1.0926"),
            leverage_implied=Decimal("1.8305"),
            entry_px=5000.0,
            source="journal.decisions/2026-10-09.json",
        )
    )
    assert "Billete de ejecucion (#84)" in text
    for incident in INCIDENTS:
        assert incident["id"] in text
    assert "09:30 (inmediatamente)" in text
    assert "15:45" in text and "16:00" in text
    assert "stop **4972.685**" in text and "objetivo **5054.63**" in text


def test_84_the_ticket_says_when_the_entry_is_not_declared() -> None:
    text = render_ticket(
        ticket(
            session=SESSION,
            direction="long",
            tier="A",
            notional_usd=Decimal("18305"),
            stop_pct=Decimal("0.5463"),
            target_pct=Decimal("1.0926"),
        )
    )
    assert "no se inventa" in text


# ─────────────────────────────────────────────────────────────────────────────
# A7 · La CLI
# ─────────────────────────────────────────────────────────────────────────────
def _write_decision(journal_root: Path, **kwargs: object) -> None:
    Journal(journal_root).write("decisions", _decision_row(**kwargs))  # type: ignore[arg-type]


def test_84_the_cli_without_a_session_row_is_an_error(tmp_path: Path) -> None:
    assert (
        main(["--journal-root", str(tmp_path / "journal"), "--session", SESSION.isoformat()]) == 2
    )
    assert main(["--journal-root", str(tmp_path / "journal"), "--session", "no-es-fecha"]) == 2


def test_84_the_cli_refuses_to_have_a_ticket_for_a_nothing_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    journal_root = tmp_path / "journal"
    _write_decision(journal_root, direction="nothing", tier="C")
    code = main(["--journal-root", str(journal_root), "--session", SESSION.isoformat()])
    captured = capsys.readouterr()
    assert code == 2
    assert "no se opera" in captured.err


def test_84_the_cli_prints_the_ticket_of_a_directional_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    journal_root = tmp_path / "journal"
    _write_decision(journal_root)
    code = main(
        [
            "--journal-root",
            str(journal_root),
            "--session",
            SESSION.isoformat(),
            "--entry-px",
            "5000",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "Billete de ejecucion (#84)" in captured.out
    assert "journal.decisions/2026-10-09.json" in captured.out
    # El nocional sale del apalancamiento publicado por el gate x el capital declarado.
    assert "**nocional:** 18305.0000 USD" in captured.out
    assert "stop **4972.685**" in captured.out


def test_84_the_cli_records_the_real_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    journal_root = tmp_path / "journal"
    _write_decision(journal_root)
    base = ["--journal-root", str(journal_root), "--session", SESSION.isoformat(), "--record"]
    assert main(base) == 2  # sin datos reales no se escribe nada
    code = main(
        [
            *base,
            "--entry-px",
            "5000",
            "--exit-px",
            "4975",
            "--exit-reason",
            "stop",
            "--costs-pct",
            "0.0042",
            "--entry-time",
            ENTRY_TIME.isoformat(),
            "--exit-time",
            EXIT_TIME.isoformat(),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "operacion registrada" in captured.out
    assert "closed_by_close: True" in captured.out
    row = read_trade(journal_root, SESSION)
    assert row["exit_reason"] == "stop"
    assert row["pnl_pct"] == pytest.approx(-0.5042)


def test_84_the_cli_records_an_overnight_breach_and_warns_about_a_deviation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    journal_root = tmp_path / "journal"
    _write_decision(journal_root)
    code = main(
        [
            "--journal-root",
            str(journal_root),
            "--session",
            SESSION.isoformat(),
            "--record",
            "--entry-px",
            "5000",
            "--exit-px",
            "5025",
            "--exit-reason",
            "close",
            "--costs-pct",
            "0.0182",
            "--entry-time",
            ENTRY_TIME.isoformat(),
            "--exit-time",
            EXIT_TIME.isoformat(),
            "--direction",
            "short",
            "--overnight",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "closed_by_close: False" in captured.out
    assert "no es la registrada en la pista" in captured.err
    row = read_trade(journal_root, SESSION)
    assert row["closed_by_close"] is False
    # En corto, subir de 5000 a 5025 pierde.
    assert row["pnl_pct"] == pytest.approx(-0.5182)
