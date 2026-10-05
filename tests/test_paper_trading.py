"""Tests del contrato de #45: el informe de *paper trading* de la Fase 4.

El modulo **recomputa** el resultado de cada recomendacion desde `journal.decisions` + el almacen
(nunca lee `journal.trades`, reservado a #47) y aplica la puerta de §16 al pie de la letra. Las
pruebas con el almacen real se **saltan** si no hay `data/` (clon limpio, CI); las puras no.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final, cast

import pytest

from cfdtrader.analysis import paper_trading
from cfdtrader.analysis.paper_trading import (
    MIN_SESSIONS,
    REFERENCE_ARM,
    STATE_DIVERGES,
    STATE_IN_RANGE,
    STATE_NOT_EVALUABLE,
    MalformedReferenceError,
    MissingReferenceArtifactError,
    _outcome,  # pyright: ignore[reportPrivateUsage]
    _recommendations,  # pyright: ignore[reportPrivateUsage]
    _reference_block,  # pyright: ignore[reportPrivateUsage]
    _reference_series,  # pyright: ignore[reportPrivateUsage]
    _resolve,  # pyright: ignore[reportPrivateUsage]
    _verdict,  # pyright: ignore[reportPrivateUsage]
    analyse,
    main,
    render_markdown,
)
from cfdtrader.journal.decision_log import Journal, write_record

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"
REPORTS: Final[Path] = REAL_DATA / "derived" / "reports"
REFERENCE: Final[Path] = REPORTS / "pipeline_backtest_2026-09-23.json"
SOURCE: Final[str] = Path(str(paper_trading.__file__)).read_text(encoding="utf-8")

AS_OF: Final[datetime] = datetime(2026, 10, 5, tzinfo=UTC)
ZERO_SHA: Final[str] = "sha256:" + "0" * 64
COUNT_KEYS: Final[tuple[str, ...]] = ("recommendation", "nothing", "no_recommendation", "error")


def _decision(
    trade_date: date,
    *,
    status: str = "recommendation",
    direction: str | None = "long",
    stop_pct: float | None = 0.005,
    target_pct: float | None = 0.01,
    cost_pct: float | None = 0.0042,
) -> dict[str, object]:
    """Una fila completa de ``journal.decisions`` (las 24 columnas del esquema cerrado de #39)."""
    return {
        "trade_date": trade_date.isoformat(),
        "as_of": "2026-09-17T12:00:00+00:00",
        "status": status,
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
        "tier": "C" if direction == "nothing" else "B",
        "blocking_events": [],
        "bull_case": [],
        "bear_case": [],
        "llm_overlay": None,
        "report_text": "informe de prueba",
    }


def _daily(
    open_px: float = 100.0, *, high: float = 106.0, low: float = 98.0
) -> dict[date, dict[str, float]]:
    day = date(2026, 9, 17)
    return {day: {"open": open_px, "high": high, "low": low, "close": 100.0}}


# ─────────────────────────────────────────────────────────────────────────────
# A3/A4 · El orden de las barreras y la recomputacion del neto
# ─────────────────────────────────────────────────────────────────────────────
def _hit(
    direction: str, bars: list[tuple[float, float]], upper: float, lower: float
) -> tuple[str, float]:
    """Atajo de ``_resolve`` con un cierre fijo (100.0)."""
    return _resolve(direction=direction, bars=bars, upper=upper, lower=lower, close=100.0)


def test_45_the_resolve_orders_the_barriers_by_direction() -> None:
    assert _hit("long", [(105.0, 99.5)], 105.0, 99.0) == ("target", 105.0)
    assert _hit("long", [(104.0, 98.5)], 105.0, 99.0) == ("stop", 99.0)
    assert _hit("short", [(100.5, 94.5)], 101.0, 95.0) == ("target", 95.0)
    assert _hit("long", [(101.0, 99.5)], 105.0, 99.0) == ("close", 100.0)


def test_45_a_bar_that_touches_both_barriers_is_resolved_adversely() -> None:
    assert _hit("long", [(106.0, 98.0)], 105.0, 99.0) == ("stop", 99.0)
    assert _hit("short", [(102.0, 94.0)], 101.0, 95.0) == ("stop", 101.0)


def test_45_the_outcome_recomputes_gross_minus_declared_cost() -> None:
    # Barrera de arriba 101.0 (target_pct 1 %), de abajo 99.5 (stop_pct 0.5 %). Una barra que
    # **solo** toca la de arriba -> target; (low 99.5 tocaria las dos y resolveria al stop adverso).
    outcome = _outcome(
        _decision(date(2026, 9, 17), direction="long"),
        daily=_daily(),
        intraday={date(2026, 9, 17): [(101.0, 100.5)]},
    )
    assert outcome is not None
    assert outcome.exit_reason == "target"
    assert outcome.gross_return_pct == pytest.approx(1.0)
    assert outcome.cost_pct == pytest.approx(0.42)
    assert outcome.net_return_pct == pytest.approx(1.0 - 0.42)


def test_45_a_decision_without_market_data_has_no_outcome() -> None:
    assert _outcome(_decision(date(2020, 1, 2)), daily={}, intraday={}) is None
    assert (
        _outcome(_decision(date(2026, 9, 17), stop_pct=None), daily=_daily(), intraday={}) is None
    )


# ─────────────────────────────────────────────────────────────────────────────
# A5 · Recuentos y estadistico
# ─────────────────────────────────────────────────────────────────────────────
def test_45_the_recommendations_split_the_counts() -> None:
    rows = [
        _decision(date(2026, 9, 17), direction="long"),
        _decision(date(2026, 9, 18), direction="short"),
        _decision(date(2026, 9, 21), direction="nothing", stop_pct=None, target_pct=None),
        _decision(date(2026, 9, 22), status="no_recommendation_stale_data", direction=None),
        _decision(date(2026, 9, 23), status="error", direction=None),
    ]
    emitted, counts = _recommendations(rows)
    assert [row["direction"] for row in emitted] == ["long", "short"]
    assert counts == {"recommendation": 3, "nothing": 1, "no_recommendation": 1, "error": 1}


# ─────────────────────────────────────────────────────────────────────────────
# A6/A7/A8 · La regla de §16
# ─────────────────────────────────────────────────────────────────────────────
def test_45_the_verdict_is_not_evaluable_below_the_minimum() -> None:
    reference = {"mean_pct": 0.0, "sigma_pct": 1.0}
    for n in (0, MIN_SESSIONS - 1):
        block = _verdict(n_sessions=n, mean_paper_pct=None, reference=reference)
        assert block["state"] == STATE_NOT_EVALUABLE
        assert block["threshold_pct"] is None


def test_45_the_verdict_applies_the_two_sigma_rule() -> None:
    reference = {"mean_pct": 0.0, "sigma_pct": 1.0}
    # N = 100 -> umbral = 2 / 10 = 0.2; una media de 0.1 no diverge, una de 0.5 si.
    inside = _verdict(n_sessions=100, mean_paper_pct=0.1, reference=reference)
    assert inside["state"] == STATE_IN_RANGE and inside["diverges"] is False
    assert inside["threshold_pct"] == pytest.approx(0.2)
    outside = _verdict(n_sessions=100, mean_paper_pct=0.5, reference=reference)
    assert outside["state"] == STATE_DIVERGES and outside["diverges"] is True


# ─────────────────────────────────────────────────────────────────────────────
# A6 · La referencia es la serie declarada del brazo de referencia
# ─────────────────────────────────────────────────────────────────────────────
def test_45_a_missing_reference_artifact_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(MissingReferenceArtifactError):
        _reference_block(tmp_path / "no_existe.json")


def test_45_a_malformed_reference_is_a_typed_error() -> None:
    with pytest.raises(MalformedReferenceError):
        _reference_series({})
    with pytest.raises(MalformedReferenceError):
        _reference_series({"arms": {REFERENCE_ARM: {}}})


# ─────────────────────────────────────────────────────────────────────────────
# A11 · Sin reloj y sin red
# ─────────────────────────────────────────────────────────────────────────────
def test_45_the_module_never_reads_the_clock_nor_the_network() -> None:
    for forbidden in (
        "import yfinance",
        "import requests",
        "import urllib",
        "import socket",
        "import httpx",
        "import aiohttp",
        "import fredapi",
    ):
        assert forbidden not in SOURCE, forbidden
    for clock in (".now(", ".today(", ".utcnow("):
        assert clock not in SOURCE, f"el modulo no debe leer el reloj ({clock})"


# ─────────────────────────────────────────────────────────────────────────────
# A2 · La CLI (as-of obligatorio)
# ─────────────────────────────────────────────────────────────────────────────
def test_45_the_cli_requires_a_declared_as_of(tmp_path: Path) -> None:
    base = ["--data-root", "data", "--journal-root", str(tmp_path), "--reference-artifact", "ref"]
    assert main(base) == 2
    assert main([*base, "--as-of", "no-es-iso"]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# A12/A13 · Extremo a extremo sobre el almacen real (se salta sin `data/`)
# ─────────────────────────────────────────────────────────────────────────────
def _real_sessions(limit: int) -> list[date]:
    from cfdtrader.data.store import Store

    daily = paper_trading._daily_by_session(Store(REAL_DATA))  # pyright: ignore[reportPrivateUsage]
    return sorted(daily)[-limit:]


def _counts(value: object) -> dict[str, int]:
    mapping = cast("Mapping[str, object]", value)
    return {key: int(str(count)) for key, count in mapping.items()}


@pytest.mark.skipif(not REFERENCE.is_file(), reason="sin el artefacto de referencia de #28")
def test_45_analyse_writes_json_and_markdown_and_never_an_invented_zero(tmp_path: Path) -> None:
    journal_root = tmp_path / "journal"
    journal = Journal(journal_root)
    sessions = _real_sessions(3)
    for day in sessions:
        write_record(journal, "decisions", _decision(day, direction="long"))
    # Una fila sin datos de mercado (fecha absurda) para probar el "no se".
    write_record(journal, "decisions", _decision(date(1990, 1, 2), direction="short"))

    report = analyse(
        store=paper_trading.Store(REAL_DATA),
        journal_root=journal_root,
        reference_artifact=REFERENCE,
        as_of=AS_OF,
        reports_dir=tmp_path / "reports",
        write=True,
    )
    payload = report.payload
    sample = cast("Mapping[str, object]", payload["sample"])
    assert sample["n_sessions"] == len(sessions)
    assert sample["missing_outcomes"] == 1
    assert _counts(sample["counts"])["recommendation"] == len(sessions) + 1
    metrics = cast("Mapping[str, object]", payload["metrics"])
    verdict = cast("Mapping[str, object]", metrics["verdict"])
    assert verdict["state"] == STATE_NOT_EVALUABLE
    assert verdict["mean_paper_pct"] is not None
    assert verdict["threshold_pct"] is None
    stem = f"paper_trading_{report.report_date}"
    payload_json = json.loads((tmp_path / "reports" / f"{stem}.json").read_text(encoding="utf-8"))
    assert payload_json["report_sha256"] == report.report_sha256
    assert "es una afirmacion de *edge* ni una validacion" in render_markdown(report)
    body = {key: value for key, value in payload.items() if key != "report_sha256"}
    digest = hashlib.sha256(paper_trading.canonical_text(body).encode("utf-8")).hexdigest()
    assert report.report_sha256 == "sha256:" + digest


@pytest.mark.skipif(not REFERENCE.is_file(), reason="sin el artefacto de referencia de #28")
def test_45_write_false_writes_nothing(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    analyse(
        store=paper_trading.Store(REAL_DATA),
        journal_root=tmp_path / "journal",
        reference_artifact=REFERENCE,
        as_of=AS_OF,
        reports_dir=reports_dir,
        write=False,
    )
    assert not reports_dir.exists()


@pytest.mark.skipif(not REFERENCE.is_file(), reason="sin el artefacto de referencia de #28")
def test_45_the_cli_is_byte_identical_across_processes(tmp_path: Path) -> None:
    journal_root = tmp_path / "journal"
    journal = Journal(journal_root)
    for day in _real_sessions(2):
        write_record(journal, "decisions", _decision(day, direction="long"))
    name = f"paper_trading_{AS_OF.date().isoformat()}.json"
    seen: set[str] = set()
    for seed in ("0", "1", "random"):
        out = tmp_path / f"run_{seed}"
        out.mkdir()
        result = subprocess.run(  # noqa: S603 - comando fijo
            [
                sys.executable,
                "-m",
                "cfdtrader.analysis.paper_trading",
                "--data-root",
                str(REAL_DATA),
                "--journal-root",
                str(journal_root),
                "--reference-artifact",
                str(REFERENCE),
                "--reports-dir",
                str(out),
                "--as-of",
                AS_OF.isoformat(),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert result.returncode == 0, result.stderr
        seen.add(hashlib.sha256((out / name).read_bytes()).hexdigest())
    assert len(seen) == 1, "el informe no es determinista entre procesos"
