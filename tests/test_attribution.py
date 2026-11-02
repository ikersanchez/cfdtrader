"""Tests de #162: la atribucion por agente del *paper* (tilt + resultado de sesion).

La fila de `journal.attribution` se compone de dos fuentes que ya existen: el tilt que publica
#161 en `journal.agent_signals.evidence.tilt` y la serie de paper de #45 (`journal.decisions` +
almacen). Los tests construyen un almacen y un diario **sinteticos** en `tmp_path`: la suite no
escribe en el `data/` ni en el `journal/` del repositorio (fixtures de `tests/conftest.py`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from datetime import time as dtime
from pathlib import Path
from typing import Final, cast

import pytest

from cfdtrader.analysis.paper_trading import DecisionOutcome
from cfdtrader.data.store import Store
from cfdtrader.journal import attribution
from cfdtrader.journal.attribution import (
    AGENT,
    DOWN,
    FLAT,
    LONG,
    NEUTRAL,
    SHORT,
    UP,
    InvalidTiltError,
    MissingMarketDataError,
    attribute_sessions,
    build_attribution,
    main,
    record_attribution,
)
from cfdtrader.journal.decision_log import Journal, read_record, write_record

DAY_UP: Final[date] = date(2026, 10, 5)
DAY_DOWN: Final[date] = date(2026, 10, 6)
DAY_FLAT: Final[date] = date(2026, 10, 7)
ZERO_SHA: Final[str] = "sha256:" + "0" * 64

#: El tilt minimo con la forma que publica #161 (las once claves declaradas).
TILT: Final[dict[str, object]] = {
    "net_pct": 5.0,
    "clamped": False,
    "ratio": 0.5,
    "side": LONG,
    "weighted": {"bullish": 5.0, "bearish": 0.0, "neutral": 5.0},
    "n_events": 2,
    "n_neutral": 1,
    "consensus": 0.5,
    "horizon": {"intraday": 1, "days": 1},
    "dominant": None,
    "capped_events": 0,
}


def _tilt(side: str) -> dict[str, object]:
    """Un tilt con ese lado, con la forma completa de #161."""
    return {**TILT, "side": side}


def _daily_record(day: date, *, open_px: float, close_px: float) -> dict[str, object]:
    """Una sesion diaria de la serie del CFD (los campos que exige `raw.market_daily`)."""
    return {
        "source": "yfinance",
        "series_id": "^GSPC",
        "as_of": datetime.combine(day, dtime(20, 0), tzinfo=UTC),
        "fetched_at": datetime.combine(day, dtime(23, 0), tzinfo=UTC),
        "published_at": None,
        "open": float(open_px),
        "high": float(max(open_px, close_px)) * 1.01,
        "low": float(min(open_px, close_px)) * 0.99,
        "close": float(close_px),
        "volume": 1_000_000.0,
        "adj_close": float(close_px),
    }


def _build_store(root: Path, bars: Sequence[tuple[date, float, float]]) -> Store:
    """Almacen temporal con esas sesiones diarias (sin intradia: respaldo diario)."""
    store = Store(root / "data")
    store.append(
        "raw",
        "market_daily",
        [_daily_record(day, open_px=open_px, close_px=close_px) for day, open_px, close_px in bars],
    )
    return store


def _decision(
    day: date,
    *,
    direction: str = LONG,
    stop_pct: float = 0.005,
    target_pct: float = 0.01,
    cost_pct: float = 0.0042,
) -> dict[str, object]:
    """Una fila completa de `journal.decisions` (las 24 columnas del esquema cerrado de #39)."""
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
        "leverage_implied": None,
        "tier": "B",
        "blocking_events": [],
        "bull_case": [],
        "bear_case": [],
        "llm_overlay": "applied",
        "report_text": "informe de prueba",
    }


def _signal(day: date, *, side: str | None, agent: str = AGENT) -> dict[str, object]:
    """Una fila de `journal.agent_signals`; `side=None` simula una fila anterior a #161."""
    evidence: dict[str, object] = {
        "state": "applied",
        "direction": "long",
        "gate_direction_with_overlay": "long",
    }
    if side is not None:
        evidence["tilt"] = _tilt(side)
    return {
        "trade_date": day.isoformat(),
        "agent": agent,
        "prob_up": 0.6,
        "confidence": None,
        "veto": False,
        "veto_reason": None,
        "evidence": evidence,
    }


def _journal(
    *,
    signals: Sequence[Mapping[str, object]],
    decisions: Sequence[Mapping[str, object]],
    root: Path,
) -> Path:
    """Diario temporal con las decisiones y senales declaradas."""
    journal_root = root / "journal"
    journal = Journal(journal_root)
    for decision in decisions:
        write_record(journal, "decisions", decision)
    for signal in signals:
        write_record(journal, "agent_signals", signal)
    return journal_root


def _identity(day: date) -> dict[str, str]:
    """La identidad compuesta de `journal.attribution` (#123)."""
    return {"trade_date": day.isoformat(), "agent": AGENT}


def _fingerprint(directory: Path) -> dict[str, str]:
    """Ruta relativa -> sha256, para demostrar que un directorio no cambia."""
    if not directory.exists():
        return {}
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*.json"))
    }


def test_162_the_package_reexports_the_module() -> None:
    """El paquete del diario expone el modulo sin arrastrarlo al importar (PEP 562)."""
    import cfdtrader.journal as package

    assert package.attribution is attribution
    assert "attribution" in package.__all__


def test_162_the_pipeline_runs_the_attribution_after_the_paper() -> None:
    """El paso operativo existe y va **despues** de la puerta del paper (#162)."""
    pipeline = (Path(__file__).resolve().parents[1] / "run_pipeline.sh").read_text(encoding="utf-8")

    paper = pipeline.index("cfdtrader.analysis.paper_trading")
    attribution_step = pipeline.index("cfdtrader.journal.attribution")
    assert paper < attribution_step


# ─────────────────────────────────────────────────────────────────────────────
# El juicio por agente y la evidencia publicada
# ─────────────────────────────────────────────────────────────────────────────
def test_162_the_row_carries_the_tilt_and_the_session_result(tmp_path: Path) -> None:
    """#162: el tilt **verbatim** y el resultado de la sesion, sin tocar el gate."""
    store = _build_store(tmp_path, [(DAY_UP, 100.0, 110.0), (DAY_DOWN, 100.0, 90.0)])
    journal_root = _journal(
        signals=[_signal(DAY_UP, side=LONG), _signal(DAY_DOWN, side=SHORT)],
        decisions=[_decision(DAY_UP), _decision(DAY_DOWN)],
        root=tmp_path,
    )

    report = record_attribution(store=store, journal_root=journal_root)

    assert len(report.rows) == 2 and len(report.written) == 2 and report.existing == ()
    journal = Journal(journal_root)
    up = cast(
        "Mapping[str, object]",
        read_record(journal, "attribution", _identity(DAY_UP)),
    )
    assert up["trade_date"] == DAY_UP.isoformat() and up["agent"] == AGENT
    assert up["would_have_won"] is True
    evidence = cast("Mapping[str, object]", up["evidence"])
    assert evidence["tilt"] == _tilt(LONG)
    session = cast("Mapping[str, object]", evidence["session"])
    assert session["direction"] == UP
    assert session["return_pct"] == pytest.approx(10.0)
    assert session["open"] == 100.0 and session["close"] == 110.0
    paper = cast("Mapping[str, object]", evidence["paper"])
    assert paper["direction"] == LONG
    assert "§19.19" in str(evidence["note"])

    down = cast(
        "Mapping[str, object]",
        read_record(journal, "attribution", _identity(DAY_DOWN)),
    )
    assert down["would_have_won"] is True
    down_evidence = cast("Mapping[str, object]", down["evidence"])
    assert cast("Mapping[str, object]", down_evidence["session"])["direction"] == DOWN


def test_162_a_wrong_side_is_a_loss(tmp_path: Path) -> None:
    """Un `short` en una sesion al alza no gana: el juicio es del agente, no del modelo."""
    store = _build_store(tmp_path, [(DAY_UP, 100.0, 110.0)])
    journal_root = _journal(
        signals=[_signal(DAY_UP, side=SHORT)],
        decisions=[_decision(DAY_UP)],
        root=tmp_path,
    )

    report = record_attribution(store=store, journal_root=journal_root)

    assert report.rows[0].would_have_won is False


def test_162_a_neutral_tilt_has_no_verdict(tmp_path: Path) -> None:
    """Un tilt neutral no hace afirmacion direccional: `None`, no un falso aprobado."""
    store = _build_store(tmp_path, [(DAY_UP, 100.0, 110.0)])
    journal_root = _journal(
        signals=[_signal(DAY_UP, side=NEUTRAL)],
        decisions=[_decision(DAY_UP)],
        root=tmp_path,
    )

    report = record_attribution(store=store, journal_root=journal_root)

    assert report.rows[0].would_have_won is None
    journal = Journal(journal_root)
    row = read_record(journal, "attribution", _identity(DAY_UP))
    assert row["would_have_won"] is None


def test_162_a_flat_session_is_not_a_win(tmp_path: Path) -> None:
    """Un cierre exactamente plano no da la razon a ninguno de los dos lados."""
    store = _build_store(tmp_path, [(DAY_FLAT, 100.0, 100.0)])
    journal_root = _journal(
        signals=[_signal(DAY_FLAT, side=LONG)],
        decisions=[_decision(DAY_FLAT)],
        root=tmp_path,
    )

    report = record_attribution(store=store, journal_root=journal_root)

    assert report.rows[0].would_have_won is False
    session = cast("Mapping[str, object]", report.rows[0].evidence["session"])
    assert session["direction"] == FLAT


# ─────────────────────────────────────────────────────────────────────────────
# Fronteras: sin serie de paper, sin tilt, sin senal y diario inmutable
# ─────────────────────────────────────────────────────────────────────────────
def test_162_only_sessions_with_a_paper_outcome_are_attributed(tmp_path: Path) -> None:
    """Sin serie de paper no hay fila: el hueco se declara, no se rellena hacia atras."""
    store = _build_store(tmp_path, [(DAY_UP, 100.0, 110.0), (DAY_DOWN, 100.0, 90.0)])
    journal_root = _journal(
        signals=[_signal(DAY_UP, side=LONG), _signal(DAY_DOWN, side=SHORT)],
        # Solo la primera sesion tiene recomendacion direccional recomputable.
        decisions=[_decision(DAY_UP)],
        root=tmp_path,
    )

    report = record_attribution(store=store, journal_root=journal_root)

    assert [row.trade_date for row in report.rows] == [DAY_UP]
    assert any(
        entry["trade_date"] == DAY_DOWN.isoformat() and "sin serie de paper" in entry["reason"]
        for entry in report.skipped
    )


def test_162_a_signal_without_tilt_is_not_backfilled(tmp_path: Path) -> None:
    """Las filas anteriores a #161 no traen tilt: se declaran omitidas, no se reescriben."""
    store = _build_store(tmp_path, [(DAY_UP, 100.0, 110.0)])
    journal_root = _journal(
        signals=[_signal(DAY_UP, side=None)],
        decisions=[_decision(DAY_UP)],
        root=tmp_path,
    )

    report = record_attribution(store=store, journal_root=journal_root)

    assert report.rows == ()
    assert any("no trae `tilt`" in entry["reason"] for entry in report.skipped)


def test_162_a_session_without_a_news_signal_is_declared(tmp_path: Path) -> None:
    """La serie de paper sin senal de noticias no inventa un agente que no hablo."""
    store = _build_store(tmp_path, [(DAY_UP, 100.0, 110.0)])
    journal_root = _journal(
        signals=[_signal(DAY_UP, side=LONG, agent="technical")],
        decisions=[_decision(DAY_UP)],
        root=tmp_path,
    )

    report = record_attribution(store=store, journal_root=journal_root)

    assert report.rows == ()
    assert any("no hay senal del agente `news`" in entry["reason"] for entry in report.skipped)


def test_162_the_journal_is_append_only(tmp_path: Path) -> None:
    """Una identidad ya escrita no se reescribe aunque el veredicto calculado cambie."""
    store = _build_store(tmp_path, [(DAY_UP, 100.0, 110.0)])
    journal_root = _journal(
        signals=[_signal(DAY_UP, side=LONG)],
        decisions=[_decision(DAY_UP)],
        root=tmp_path,
    )
    journal = Journal(journal_root)
    journal.write(
        "attribution",
        {
            "trade_date": DAY_UP.isoformat(),
            "agent": AGENT,
            "would_have_won": False,
            "evidence": {"pre": "existente"},
        },
    )
    path = journal.path("attribution", _identity(DAY_UP))
    before = path.read_bytes()

    report = record_attribution(store=store, journal_root=journal_root)

    assert report.written == () and len(report.existing) == 1
    assert path.read_bytes() == before
    row = read_record(journal, "attribution", _identity(DAY_UP))
    assert row["would_have_won"] is False
    assert row["evidence"] == {"pre": "existente"}


def test_162_only_attribution_is_written(tmp_path: Path) -> None:
    """El gate, `journal.decisions` y `agent_signals` no cambian: solo se rellena la tabla."""
    store = _build_store(tmp_path, [(DAY_UP, 100.0, 110.0)])
    journal_root = _journal(
        signals=[_signal(DAY_UP, side=LONG)],
        decisions=[_decision(DAY_UP)],
        root=tmp_path,
    )
    decisions_before = _fingerprint(journal_root / "decisions")
    signals_before = _fingerprint(journal_root / "agent_signals")

    record_attribution(store=store, journal_root=journal_root)

    assert _fingerprint(journal_root / "decisions") == decisions_before
    assert _fingerprint(journal_root / "agent_signals") == signals_before
    assert len(list((journal_root / "attribution").glob("*.json"))) == 1


def test_162_dry_run_writes_nothing(tmp_path: Path) -> None:
    """`write=False` calcula las filas y deja el diario intacto."""
    store = _build_store(tmp_path, [(DAY_UP, 100.0, 110.0)])
    journal_root = _journal(
        signals=[_signal(DAY_UP, side=LONG)],
        decisions=[_decision(DAY_UP)],
        root=tmp_path,
    )

    report = record_attribution(store=store, journal_root=journal_root, write=False)

    assert len(report.rows) == 1 and report.written == ()
    assert not (journal_root / "attribution").exists()


def test_162_build_attribution_does_not_write(tmp_path: Path) -> None:
    """El constructor puro devuelve las filas y no crea la tabla."""
    store = _build_store(tmp_path, [(DAY_UP, 100.0, 110.0)])
    journal_root = _journal(
        signals=[_signal(DAY_UP, side=LONG)],
        decisions=[_decision(DAY_UP)],
        root=tmp_path,
    )

    rows = build_attribution(store=store, journal_root=journal_root)

    assert len(rows) == 1 and not (journal_root / "attribution").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Funcion pura, errores tipados y CLI
# ─────────────────────────────────────────────────────────────────────────────
def _outcome(day: date, *, direction: str = LONG) -> DecisionOutcome:
    """Un `DecisionOutcome` de #45 para probar la funcion pura sin almacen."""
    return DecisionOutcome(
        trade_date=day,
        direction=direction,
        entry_px=100.0,
        stop_pct=0.005,
        target_pct=0.01,
        exit_px=101.0,
        exit_reason="target",
        gross_return_pct=1.0,
        cost_pct=0.42,
        net_return_pct=0.58,
    )


def test_162_the_pure_function_sorts_by_session_and_skips_what_it_cannot_judge() -> None:
    """La funcion pura no inventa filas ni desordena: solo las sesiones con tilt y outcome."""
    outcomes = {DAY_UP: _outcome(DAY_UP), DAY_DOWN: _outcome(DAY_DOWN)}
    daily = {
        DAY_UP: {"open": 100.0, "close": 110.0},
        DAY_DOWN: {"open": 100.0, "close": 90.0},
        DAY_FLAT: {"open": 100.0, "close": 100.0},
    }
    tilts = {
        DAY_DOWN: _tilt(SHORT),
        DAY_FLAT: _tilt(LONG),  # sin outcome: no hay fila
        DAY_UP: _tilt(LONG),
    }

    rows = attribute_sessions(outcomes=outcomes, daily=daily, tilts=tilts)

    assert [row.trade_date for row in rows] == [DAY_UP, DAY_DOWN]
    assert [row.would_have_won for row in rows] == [True, True]


def test_162_an_invalid_tilt_side_is_a_typed_error() -> None:
    """Un tilt sin `side` declarado no se adivina: error tipado con la sesion."""
    with pytest.raises(InvalidTiltError, match=DAY_UP.isoformat()):
        attribute_sessions(
            outcomes={DAY_UP: _outcome(DAY_UP)},
            daily={DAY_UP: {"open": 100.0, "close": 110.0}},
            tilts={DAY_UP: {"side": "sideways"}},
        )


def test_162_a_non_positive_open_is_a_typed_error() -> None:
    """Sin `open` positivo no hay retorno de sesion que atribuir."""
    with pytest.raises(attribution.AttributionError, match="no es positivo"):
        attribute_sessions(
            outcomes={DAY_UP: _outcome(DAY_UP)},
            daily={DAY_UP: {"open": 0.0, "close": 110.0}},
            tilts={DAY_UP: _tilt(LONG)},
        )


def test_162_without_market_data_the_error_is_typed(tmp_path: Path) -> None:
    """Sin `raw.market_daily` la atribucion falla con un motivo, no con un `KeyError`."""
    with pytest.raises(MissingMarketDataError, match=r"raw\.market_daily"):
        record_attribution(
            store=Store(tmp_path / "vacio"),
            journal_root=tmp_path / "journal",
        )


def test_162_the_cli_writes_and_reports_the_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """El CLI manual rellena la tabla y publica el recuento honesto."""
    store = _build_store(tmp_path, [(DAY_UP, 100.0, 110.0)])
    journal_root = _journal(
        signals=[_signal(DAY_UP, side=LONG)],
        decisions=[_decision(DAY_UP)],
        root=tmp_path,
    )

    rc = main(["--data-root", str(store.root), "--journal-root", str(journal_root)])

    assert rc == 0
    captured = capsys.readouterr()
    assert "1 filas candidatas" in captured.out
    assert "1 escritas" in captured.out
    assert "0 ya en el diario" in captured.out
    assert len(list((journal_root / "attribution").glob("*.json"))) == 1


def test_162_the_cli_dry_run_does_not_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--dry-run` publica lo que escribiria y deja el diario intacto."""
    store = _build_store(tmp_path, [(DAY_UP, 100.0, 110.0)])
    journal_root = _journal(
        signals=[_signal(DAY_UP, side=LONG)],
        decisions=[_decision(DAY_UP)],
        root=tmp_path,
    )

    rc = main(
        [
            "--data-root",
            str(store.root),
            "--journal-root",
            str(journal_root),
            "--dry-run",
        ]
    )

    assert rc == 0
    assert "0 escritas" in capsys.readouterr().out
    assert not (journal_root / "attribution").exists()


def test_162_the_cli_reports_a_missing_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sin almacen el CLI devuelve 2 y dice que falta la serie diaria."""
    rc = main(
        [
            "--data-root",
            str(tmp_path / "vacio"),
            "--journal-root",
            str(tmp_path / "journal"),
        ]
    )

    assert rc == 2
    assert "raw.market_daily" in capsys.readouterr().err
