"""Tests del contrato de #83: la acumulacion del P&L realizado para el *kill switch*.

El modulo publica las **tres cifras de cartera** de §12 reglas 3, 4 y 5 (dia natural ET, semana
ISO-8601, mes natural) que el gate de #27 recibe como argumentos explicitos, con dos vias: la
operacion **real** (`journal.trades`, #47; vacia en observacion) y la **recomputacion** con la
maquina de #45. Los tests del nucleo son puros; los de las dos vias usan un diario y un almacen
**sinteticos** en `tmp_path` (la sesion de tests no toca `data/` ni `runs/`, lo blinda
`tests/conftest.py`).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Final, cast

import pytest

from cfdtrader.analysis import portfolio_rules
from cfdtrader.analysis.portfolio_rules import (
    CAPITAL_USD,
    LIMITS,
    MODULE,
    SOURCE_RECOMPUTED,
    STATE_IN_PROGRESS,
    STATE_NO_HISTORY,
    ClosedTrade,
    PortfolioRulesError,
    TradeLedger,
    accumulate,
    analyse,
    assess,
    closed_trades_from_journal,
    main,
    recompute_closed_trades,
    render_markdown,
    walk,
    write_report,
)
from cfdtrader.data.store import Store
from cfdtrader.journal.decision_log import Journal, write_record

SOURCE: Final[str] = Path(str(portfolio_rules.__file__)).read_text(encoding="utf-8")

#: La sesion que se decide en casi todos los tests: viernes de la semana ISO 40 de 2026.
SESSION: Final[date] = date(2026, 10, 2)
#: La fecha de generacion declarada (el modulo no lee el reloj).
AS_OF: Final[datetime] = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
ZERO_SHA: Final[str] = "sha256:" + "0" * 64


def trade(day: date, capital_pct: str, *, leverage: str = "1.818") -> ClosedTrade:
    """Una operacion cerrada con su ``%`` del capital ya convertido (fija y legible)."""
    notional = Decimal(capital_pct) / Decimal(leverage)
    return ClosedTrade(
        trade_date=day,
        capital_pct=Decimal(capital_pct),
        notional_pct=notional,
        leverage=Decimal(leverage),
    )


def _trade_row(
    day: date,
    *,
    pnl_pct: object = -0.55,
    notional: object = 18180.0,
    exit_reason: object = "stop",
) -> dict[str, object]:
    """Una fila completa de ``journal.trades`` (las 10 columnas del esquema cerrado de #39)."""
    return {
        "trade_date": day.isoformat(),
        "entry_price": 5000.0,
        "exit_price": 4990.0,
        "entry_time": f"{day.isoformat()}T13:30:00+00:00",
        "exit_time": f"{day.isoformat()}T20:00:00+00:00",
        "notional": notional,
        "pnl_pct": pnl_pct,
        "costs_pct": 0.0042,
        "exit_reason": exit_reason,
        "closed_by_close": True,
    }


def _decision_row(
    day: date,
    *,
    direction: str = "long",
    stop_pct: object = 0.005,
    target_pct: object = 0.01,
    cost_pct: object = 0.0042,
    leverage_implied: object = 1.818,
) -> dict[str, object]:
    """Una fila completa de ``journal.decisions`` (las 24 columnas del esquema de #39)."""
    return {
        "trade_date": day.isoformat(),
        "as_of": "2026-09-17T12:00:00+00:00",
        "status": "recommendation",
        "features_version": ZERO_SHA,
        "model_version": "1" * 64,
        "prompt_hashes": {},
        "git_commit": "2" * 40,
        "prob_up_raw": 0.6,
        "prob_up_calibrated": 0.6,
        "expected_move_pct": 0.5,
        "cost_pct": cost_pct,
        "ev_net_pct": None,
        "direction": direction,
        "stop_pct": stop_pct,
        "target_pct": target_pct,
        "size_notional_eur": None,
        "size_fraction": None,
        "leverage_implied": leverage_implied,
        "tier": "A",
        "blocking_events": [],
        "bull_case": [],
        "bear_case": [],
        "llm_overlay": None,
        "report_text": "informe de prueba",
    }


def _market_daily(days: Sequence[date]) -> list[dict[str, object]]:
    """Sesiones sinteticas: ``open`` y ``close`` 5000, ``high`` 5050 y ``low`` 4950."""
    fetched_at = datetime.combine(days[-1], datetime.min.time(), tzinfo=UTC) + timedelta(hours=23)
    return [
        {
            "source": "yfinance",
            "series_id": "^GSPC",
            "as_of": datetime(day.year, day.month, day.day, 21, 0, tzinfo=UTC),
            "fetched_at": fetched_at,
            "published_at": None,
            "open": 5000.0,
            "high": 5050.0,
            "low": 4950.0,
            "close": 5000.0,
            "volume": 1_000_000.0,
            "adj_close": 5000.0,
        }
        for day in days
    ]


# ─────────────────────────────────────────────────────────────────────────────
# A1 · El modulo y su contrato declarado
# ─────────────────────────────────────────────────────────────────────────────
def test_83_the_module_exports_the_declared_contract() -> None:
    assert set(portfolio_rules.__all__) == {
        "CAPITAL_PROVENANCE",
        "CAPITAL_USD",
        "DAY_TIMEZONE",
        "LIMITS",
        "MODULE",
        "REPORT_PREFIX",
        "SHA256_PREFIX",
        "SOURCE_RECOMPUTED",
        "SOURCE_TRADES",
        "STATE_IN_PROGRESS",
        "STATE_NO_HISTORY",
        "STATE_UNAVAILABLE",
        "TASK",
        "WINDOWS",
        "ClosedTrade",
        "LossLimits",
        "PortfolioRulesError",
        "PortfolioRulesReport",
        "WindowTotals",
        "accumulate",
        "analyse",
        "assess",
        "closed_trades_from_journal",
        "main",
        "recompute_closed_trades",
        "render_markdown",
        "walk",
        "write_report",
    }
    assert portfolio_rules.TASK == "#83"
    assert MODULE == "cfdtrader.analysis.portfolio_rules"
    assert portfolio_rules.REPORT_PREFIX == "portfolio_rules"
    assert portfolio_rules.DAY_TIMEZONE == "America/New_York"


def test_83_the_three_windows_declare_their_bucket_rearm_and_rule() -> None:
    assert len(portfolio_rules.WINDOWS) == 3
    assert [window["rule"] for window in portfolio_rules.WINDOWS] == ["3", "4", "5"]
    assert [window["field"] for window in portfolio_rules.WINDOWS] == [
        "daily_pnl_pct",
        "weekly_pnl_pct",
        "monthly_pnl_pct",
    ]
    limits = {
        "3": LIMITS.max_daily_loss_pct,
        "4": LIMITS.max_weekly_loss_pct,
        "5": LIMITS.max_monthly_loss_pct,
    }
    for window in portfolio_rules.WINDOWS:
        assert window["bucket"] and window["rearm"] and window["note"]
        assert window["issue"] == "#60"
        assert Decimal(window["limit_pct"]) == limits[window["rule"]]
    # La semana es ISO-8601 (lunes a domingo) y el rearme esta declarado, no implícito.
    assert "lunes" in portfolio_rules.WINDOWS[1]["bucket"]
    assert "lunes" in portfolio_rules.WINDOWS[1]["rearm"]


def test_83_the_thresholds_are_the_ones_the_s1_scenario_serves_the_gate() -> None:
    """La copia se **verifica**: los umbrales son los de #60 (§19.12), no unos de este modulo."""
    from cfdtrader.analysis.pipeline_report import scenario_parameters

    params = scenario_parameters(cost_pct=Decimal("0.0042"))
    assert LIMITS.max_daily_loss_pct == params.max_daily_loss_pct == Decimal("2")
    assert LIMITS.max_weekly_loss_pct == params.max_weekly_loss_pct == Decimal("5")
    assert LIMITS.max_monthly_loss_pct == params.max_monthly_loss_pct == Decimal("10")


# ─────────────────────────────────────────────────────────────────────────────
# A2 · Punto en el tiempo: la sesion que se decide no entra
# ─────────────────────────────────────────────────────────────────────────────
def test_83_the_session_being_decided_does_not_enter_its_own_windows() -> None:
    """Usar el resultado de la sesion ``t`` para decidir ``t`` seria *look-ahead*."""
    totals = accumulate([trade(SESSION, "-9.00")], session=SESSION)
    assert totals.state == STATE_IN_PROGRESS
    assert totals.history_trades == 1
    assert totals.considered_trades == 0
    assert totals.daily_pnl_pct == Decimal("0")
    assert totals.weekly_pnl_pct == Decimal("0")
    assert totals.monthly_pnl_pct == Decimal("0")
    assert totals.last_closed == SESSION


# ─────────────────────────────────────────────────────────────────────────────
# A3 · Los cubos son de calendario, y el dia ET no es el dia UTC cualquiera
# ─────────────────────────────────────────────────────────────────────────────
def test_83_the_buckets_are_et_day_iso_week_and_calendar_month() -> None:
    """Sesion viernes ISO-40 de octubre: cada operacion cae o no segun su **fecha de cierre**."""
    totals = accumulate(
        [
            trade(date(2026, 10, 1), "-1.00"),  # W40, octubre
            trade(date(2026, 9, 30), "-2.00"),  # W40, **septiembre**
            trade(date(2026, 9, 27), "-4.00"),  # W39 (domingo), septiembre
        ],
        session=SESSION,
    )
    assert totals.daily_pnl_pct == Decimal("0")
    assert totals.daily_trades == 0
    assert totals.weekly_pnl_pct == Decimal("-3.00")
    assert totals.weekly_trades == 2
    assert totals.monthly_pnl_pct == Decimal("-1.00")
    assert totals.monthly_trades == 1
    assert totals.considered_trades == 3


def test_83_a_sunday_closes_the_previous_iso_week_not_the_one_that_opens() -> None:
    """ISO-8601 empieza en lunes: el domingo anterior pertenece a la semana que **acaba**."""
    session = date(2026, 10, 5)  # lunes, W41
    totals = accumulate(
        [trade(date(2026, 10, 4), "-6.00")],  # domingo, W40
        session=session,
    )
    assert totals.weekly_pnl_pct == Decimal("0")
    assert totals.weekly_trades == 0
    assert totals.monthly_pnl_pct == Decimal("-6.00")  # octubre si es su mes
    assert totals.monthly_trades == 1


# ─────────────────────────────────────────────────────────────────────────────
# A4 · Sin historial no hay cifra: `None`, nunca un `0` de relleno
# ─────────────────────────────────────────────────────────────────────────────
def test_83_without_history_the_three_figures_are_none_never_zero() -> None:
    totals = accumulate([], session=SESSION)
    assert totals.state == STATE_NO_HISTORY
    assert totals.for_gate() == (None, None, None)
    assert totals.daily_pnl_pct is None
    assert totals.weekly_pnl_pct is None
    assert totals.monthly_pnl_pct is None
    assert totals.history_trades == 0
    assert totals.last_closed is None
    assert "no tiene ninguna operacion cerrada" in totals.reason


def test_83_a_session_without_a_trade_contributes_nothing_and_resets_nothing() -> None:
    """Una sesion sin operacion (o media sesion, regla 18) no es una operacion de P&L cero."""
    totals = accumulate(
        [trade(date(2026, 9, 30), "-2.00"), trade(date(2026, 10, 1), "-1.00")],
        session=SESSION,
    )
    assert totals.weekly_pnl_pct == Decimal("-3.00")
    assert totals.history_trades == 2  # no aparece ninguna fila "sin operar"


# ─────────────────────────────────────────────────────────────────────────────
# A5 · Exactitud: `Decimal`, sin `float` en el nucleo y sin redondear lo que decide
# ─────────────────────────────────────────────────────────────────────────────
def test_83_the_sum_is_exact_and_never_rounded() -> None:
    totals = accumulate(
        [trade(date(2026, 9, 30), "-0.0001"), trade(date(2026, 10, 1), "-0.0002")],
        session=SESSION,
    )
    assert totals.weekly_pnl_pct == Decimal("-0.0003")
    assert isinstance(totals.weekly_pnl_pct, Decimal)


def test_83_the_capital_is_the_declared_one() -> None:
    assert Decimal("10000") == CAPITAL_USD
    assert "NOTIONAL_USD" in portfolio_rules.CAPITAL_PROVENANCE


# ─────────────────────────────────────────────────────────────────────────────
# A6 · El veredicto informativo y su frontera
# ─────────────────────────────────────────────────────────────────────────────
def test_83_the_assessment_uses_the_sign_and_boundary_of_section_12() -> None:
    boundary = accumulate(
        [trade(date(2026, 9, 30), "-3.00"), trade(date(2026, 10, 1), "-2.00")],
        session=SESSION,
    )
    verdicts = {entry["rule"]: entry for entry in assess(boundary)}
    assert verdicts["4"]["value_pct"] == "-5.00"
    assert verdicts["4"]["threshold_pct"] == "-5"
    assert verdicts["4"]["breached"] is True
    assert verdicts["4"]["verdict"] == "incumplida"
    # Dentro por una centesima: no se bloquea.
    inside = accumulate(
        [trade(date(2026, 9, 30), "-3.00"), trade(date(2026, 10, 1), "-1.99")],
        session=SESSION,
    )
    inside_verdicts = {entry["rule"]: entry for entry in assess(inside)}
    assert inside_verdicts["4"]["breached"] is False
    assert inside_verdicts["4"]["verdict"] == "dentro_del_limite"
    assert inside_verdicts["3"]["verdict"] == "dentro_del_limite"


def test_83_the_assessment_of_an_empty_portfolio_says_no_history_not_zero() -> None:
    verdicts = assess(accumulate([], session=SESSION))
    assert [entry["verdict"] for entry in verdicts] == ["sin_historial"] * 3
    assert all(entry["value_pct"] is None for entry in verdicts)
    assert all(entry["breached"] is None for entry in verdicts)


# ─────────────────────────────────────────────────────────────────────────────
# A7 · Errores tipados y fronteras declaradas
# ─────────────────────────────────────────────────────────────────────────────
def test_83_a_datetime_is_not_a_session() -> None:
    with pytest.raises(PortfolioRulesError):
        accumulate([], session=datetime(2026, 10, 2, 12, 0, tzinfo=UTC))


def test_83_an_unknown_source_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(PortfolioRulesError):
        analyse(
            session=SESSION,
            as_of=AS_OF,
            journal_root=tmp_path / "journal",
            source="inventada",
            write=False,
        )


def test_83_the_recomputed_path_requires_the_store(tmp_path: Path) -> None:
    with pytest.raises(PortfolioRulesError):
        analyse(
            session=SESSION,
            as_of=AS_OF,
            journal_root=tmp_path / "journal",
            source=SOURCE_RECOMPUTED,
            write=False,
        )


def test_83_writing_without_a_reports_dir_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(PortfolioRulesError):
        analyse(
            session=SESSION,
            as_of=AS_OF,
            journal_root=tmp_path / "journal",
            reports_dir=None,
            write=True,
        )


def test_83_an_unreadable_percentage_is_a_typed_error() -> None:
    with pytest.raises(PortfolioRulesError):
        portfolio_rules._dec("no-es-un-numero")  # pyright: ignore[reportPrivateUsage]


# ─────────────────────────────────────────────────────────────────────────────
# A8 · Pureza, determinismo y ausencia de reloj y de red
# ─────────────────────────────────────────────────────────────────────────────
def test_83_the_module_never_reads_the_clock_nor_the_network() -> None:
    for forbidden in (
        "import yfinance",
        "import requests",
        "import urllib",
        "import socket",
        "import httpx",
        "import aiohttp",
    ):
        assert forbidden not in SOURCE, forbidden
    for clock in (".now(", ".today(", ".utcnow("):
        assert clock not in SOURCE, f"el modulo no debe leer el reloj ({clock})"


def test_83_the_accumulation_is_a_deterministic_pure_function() -> None:
    trades = (trade(date(2026, 10, 1), "-1.00"), trade(date(2026, 9, 30), "-2.00"))
    first = accumulate(trades, session=SESSION)
    second = accumulate(reversed(trades), session=SESSION)
    assert first == second  # el orden de entrada no cambia la ventana
    assert first.weekly_pnl_pct == Decimal("-3.00")


def test_83_walk_accumulates_session_by_session_without_looking_ahead() -> None:
    trades = [trade(date(2026, 9, 30), "-2.00"), trade(date(2026, 10, 1), "-1.00")]
    sessions = [date(2026, 9, 30), date(2026, 10, 1), SESSION]
    walked = walk(trades, sessions)
    assert [entry.considered_trades for entry in walked] == [0, 1, 2]
    assert [entry.weekly_pnl_pct for entry in walked] == [
        Decimal("0"),
        Decimal("-2.00"),
        Decimal("-3.00"),
    ]


def test_83_state_of_the_percentage() -> None:
    """El modulo publica ``STATE_UNAVAILABLE`` y no lo usa: no hay tercer camino silencioso."""
    assert portfolio_rules.STATE_IN_PROGRESS == "en_curso"
    assert portfolio_rules.STATE_NO_HISTORY == "sin_historial"
    assert portfolio_rules.STATE_UNAVAILABLE == "unavailable"


# ─────────────────────────────────────────────────────────────────────────────
# A9 · La via real: `journal.trades` (§12.5)
# ─────────────────────────────────────────────────────────────────────────────
def test_83_the_journal_ledger_converts_from_notional_to_capital(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal")
    write_record(journal, "trades", _trade_row(date(2026, 10, 1), pnl_pct=-0.55, notional=18180.0))
    ledger = closed_trades_from_journal(tmp_path / "journal")
    assert isinstance(ledger, TradeLedger)
    assert ledger.source == "journal.trades"
    assert ledger.skipped == ()
    (entry,) = ledger.trades
    assert entry.trade_date == date(2026, 10, 1)
    assert entry.notional_pct == Decimal("-0.55")
    assert entry.leverage == Decimal("1.818")
    # -0,55 % del nocional es -0,55 x 1,818 = -0,9999 % del capital: justo el 1 % de la regla 2.
    assert entry.capital_pct == Decimal("-0.9999")
    assert entry.exit_reason == "stop"


def test_83_the_journal_ledger_is_empty_in_observation(tmp_path: Path) -> None:
    """En observacion `journal.trades` esta vacio y eso es lo correcto (§19.11)."""
    ledger = closed_trades_from_journal(tmp_path / "journal")
    assert ledger.trades == ()
    assert ledger.skipped == ()
    totals = accumulate(ledger.trades, session=SESSION)
    assert totals.for_gate() == (None, None, None)


def test_83_a_trade_row_without_figures_is_declared_not_filled_with_zero(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal")
    write_record(journal, "trades", _trade_row(date(2026, 10, 1), notional=None))
    ledger = closed_trades_from_journal(tmp_path / "journal")
    assert ledger.trades == ()
    (skipped,) = ledger.skipped
    assert skipped["trade_date"] == "2026-10-01"
    assert "no se rellena con un cero" in skipped["reason"]


def test_83_a_trade_row_without_a_legible_date_is_declared(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal")
    row = _trade_row(date(2026, 10, 1))
    row["trade_date"] = "no-es-fecha"
    write_record(journal, "trades", row)
    ledger = closed_trades_from_journal(tmp_path / "journal")
    assert ledger.trades == ()
    assert ledger.skipped[0]["reason"] == "sin fecha legible"


# ─────────────────────────────────────────────────────────────────────────────
# A10 · La via recomputada: `journal.decisions` + el almacen (maquina de #45)
# ─────────────────────────────────────────────────────────────────────────────
def _store_with_market(tmp_path: Path, days: Sequence[date]) -> Store:
    """Almacen sintetico con las sesiones del indice en ``raw.market_daily``."""
    store = Store(tmp_path / "store")
    store.append("raw", "market_daily", _market_daily(days))
    return store


def test_83_the_recomputed_ledger_uses_the_published_leverage(tmp_path: Path) -> None:
    days = (date(2026, 10, 1), SESSION)
    store = _store_with_market(tmp_path, days)
    journal_root = tmp_path / "journal"
    journal = Journal(journal_root)
    for day in days:
        write_record(journal, "decisions", _decision_row(day))

    ledger = recompute_closed_trades(store, journal_root)
    assert ledger.source == "journal.decisions+almacen"
    assert ledger.skipped == ()
    assert [entry.trade_date for entry in ledger.trades] == list(days)
    for entry in ledger.trades:
        # La conversion es la invariante: `%` del capital = `%` del nocional x apalancamiento.
        assert entry.capital_pct == entry.notional_pct * entry.leverage
        assert entry.leverage == Decimal("1.818")
    # El resultado se recomputa con el **mismo** coste declarado que cobro el gate (§19.14).
    first = ledger.trades[0]
    assert abs(first.notional_pct - Decimal("-0.92")) < Decimal("0.000001")


def test_83_the_recomputed_ledger_does_not_count_the_session_being_decided(tmp_path: Path) -> None:
    days = (date(2026, 10, 1), SESSION)
    store = _store_with_market(tmp_path, days)
    journal_root = tmp_path / "journal"
    journal = Journal(journal_root)
    for day in days:
        write_record(journal, "decisions", _decision_row(day))
    ledger = recompute_closed_trades(store, journal_root)
    totals = accumulate(ledger.trades, session=SESSION)
    assert totals.history_trades == 2
    assert totals.considered_trades == 1
    assert totals.weekly_trades == 1
    assert totals.weekly_pnl_pct == ledger.trades[0].capital_pct
    assert totals.daily_pnl_pct == Decimal("0")


def test_83_a_recomputed_row_without_leverage_is_declared(tmp_path: Path) -> None:
    store = _store_with_market(tmp_path, (date(2026, 10, 1),))
    journal_root = tmp_path / "journal"
    write_record(
        Journal(journal_root), "decisions", _decision_row(date(2026, 10, 1), leverage_implied=None)
    )
    ledger = recompute_closed_trades(store, journal_root)
    assert ledger.trades == ()
    assert "`leverage_implied`" in ledger.skipped[0]["reason"]


def test_83_a_recomputed_row_without_market_data_is_declared(tmp_path: Path) -> None:
    store = _store_with_market(tmp_path, (date(2026, 10, 1),))
    journal_root = tmp_path / "journal"
    write_record(Journal(journal_root), "decisions", _decision_row(date(1990, 1, 2)))
    ledger = recompute_closed_trades(store, journal_root)
    assert ledger.trades == ()
    assert "no se puede recomputar" in ledger.skipped[0]["reason"]


# ─────────────────────────────────────────────────────────────────────────────
# A11 · El informe: payload, digest, markdown y escritura
# ─────────────────────────────────────────────────────────────────────────────
def test_83_analyse_publishes_the_three_figures_and_what_the_gate_receives(tmp_path: Path) -> None:
    journal_root = tmp_path / "journal"
    journal = Journal(journal_root)
    for day in (date(2026, 9, 30), date(2026, 10, 1)):
        write_record(journal, "trades", _trade_row(day, pnl_pct=-0.55))

    report = analyse(
        session=SESSION,
        as_of=AS_OF,
        journal_root=journal_root,
        reports_dir=tmp_path / "reports",
        write=True,
    )
    payload = report.payload
    accumulation = cast("Mapping[str, object]", payload["accumulation"])
    gate_input = cast("Mapping[str, object]", payload["gate_input"])
    assert accumulation["state"] == STATE_IN_PROGRESS
    assert accumulation["weekly_pnl_pct"] == "-1.9998"
    assert accumulation["weekly_trades"] == 2
    assert gate_input["weekly_pnl_pct"] == "-1.9998"
    assert gate_input["daily_pnl_pct"] == "0"
    # El 30-sep es de **septiembre**: entra en la semana ISO-40, no en el mes de octubre.
    assert gate_input["monthly_pnl_pct"] == "-0.9999"
    assert payload["task"] == "#83"
    assert payload["is_measurement"] is False
    # La via real es la de #47 y el informe lo dice.
    source = cast("Mapping[str, object]", payload["source"])
    assert source["kind"] == "journal.trades"
    # El digest cubre el payload sin su propia clave.
    body = {key: value for key, value in payload.items() if key != "report_sha256"}
    digest = hashlib.sha256(portfolio_rules.canonical_text(body).encode("utf-8")).hexdigest()
    assert report.report_sha256 == "sha256:" + digest
    json_path = tmp_path / "reports" / f"portfolio_rules_{SESSION.isoformat()}.json"
    markdown_path = tmp_path / "reports" / f"portfolio_rules_{SESSION.isoformat()}.md"
    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written["report_sha256"] == report.report_sha256
    text = markdown_path.read_text(encoding="utf-8")
    assert "Valla de cartera del *kill switch*" in text
    assert "-1.9998" in text
    assert "es una afirmacion de *edge*" in text


def test_83_analyse_without_history_sends_none_to_the_gate(tmp_path: Path) -> None:
    """Hasta que exista una operacion real, el gate no cambia de comportamiento."""
    report = analyse(session=SESSION, as_of=AS_OF, journal_root=tmp_path / "journal", write=False)
    gate_input = cast("Mapping[str, object]", report.payload["gate_input"])
    assert gate_input["daily_pnl_pct"] is None
    assert gate_input["weekly_pnl_pct"] is None
    assert gate_input["monthly_pnl_pct"] is None
    assert "sin_historial" in render_markdown(report)


def test_83_the_report_reports_a_breach_but_does_not_decide_it(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal")
    for day in (date(2026, 9, 30), date(2026, 10, 1)):
        write_record(journal, "trades", _trade_row(day, pnl_pct=-5.5))
    report = analyse(session=SESSION, as_of=AS_OF, journal_root=tmp_path / "journal", write=False)
    assessment = cast("list[dict[str, object]]", report.payload["assessment"])
    weekly = next(entry for entry in assessment if entry["rule"] == "4")
    assert weekly["breached"] is True
    does_not_do = cast("list[dict[str, object]]", report.payload["does_not_do"])
    assert any(entry["id"] == "no_evalua_las_reglas" for entry in does_not_do)
    honest = cast("Mapping[str, object]", report.payload["honesty"])
    assert honest["phase2"] == "`not_evaluable`/`fail`; `phase2_ready = false`"


def test_83_write_report_is_the_only_writer(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    report = analyse(session=SESSION, as_of=AS_OF, journal_root=tmp_path / "journal", write=False)
    assert not reports_dir.exists()
    write_report(report, reports_dir)
    assert sorted(path.name for path in reports_dir.iterdir()) == [
        f"portfolio_rules_{SESSION.isoformat()}.json",
        f"portfolio_rules_{SESSION.isoformat()}.md",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# A12 · La CLI
# ─────────────────────────────────────────────────────────────────────────────
def test_83_the_cli_requires_the_declared_session_and_instant(tmp_path: Path) -> None:
    base = ["--journal-root", str(tmp_path / "journal")]
    assert main(base) == 2
    assert main([*base, "--session", SESSION.isoformat()]) == 2
    assert main([*base, "--as-of", AS_OF.isoformat()]) == 2
    assert main([*base, "--session", "no-es-fecha", "--as-of", AS_OF.isoformat()]) == 2
    assert main([*base, "--session", SESSION.isoformat(), "--as-of", "2026-10-02T12:00:00"]) == 2


def test_83_the_cli_needs_the_store_for_the_recomputed_path(tmp_path: Path) -> None:
    code = main(
        [
            "--journal-root",
            str(tmp_path / "journal"),
            "--session",
            SESSION.isoformat(),
            "--as-of",
            AS_OF.isoformat(),
            "--source",
            SOURCE_RECOMPUTED,
        ]
    )
    assert code == 2


def test_83_the_cli_publishes_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    journal_root = tmp_path / "journal"
    write_record(Journal(journal_root), "trades", _trade_row(date(2026, 10, 1), pnl_pct=-0.55))
    code = main(
        [
            "--journal-root",
            str(journal_root),
            "--session",
            SESSION.isoformat(),
            "--as-of",
            AS_OF.isoformat(),
            "--reports-dir",
            str(tmp_path / "reports"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "estado: en_curso" in captured.out
    assert "semana: -0.9999" in captured.out
    assert (tmp_path / "reports" / f"portfolio_rules_{SESSION.isoformat()}.json").is_file()


def test_83_the_cli_report_is_byte_identical_across_processes(tmp_path: Path) -> None:
    """El JSON canonico y su digest no dependen de ``PYTHONHASHSEED``."""
    journal_root = tmp_path / "journal"
    write_record(Journal(journal_root), "trades", _trade_row(date(2026, 10, 1)))
    seen: set[bytes] = set()
    for seed in ("0", "1"):
        reports_dir = tmp_path / f"reports_{seed}"
        completed = subprocess.run(  # noqa: S603 - el ejecutable es el interprete de la sesion
            [
                sys.executable,
                "-m",
                "cfdtrader.analysis.portfolio_rules",
                "--journal-root",
                str(journal_root),
                "--session",
                SESSION.isoformat(),
                "--as-of",
                AS_OF.isoformat(),
                "--reports-dir",
                str(reports_dir),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            cwd=str(Path(__file__).resolve().parents[1]),
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        seen.add((reports_dir / f"portfolio_rules_{SESSION.isoformat()}.json").read_bytes())
    assert len(seen) == 1
