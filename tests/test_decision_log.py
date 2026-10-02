"""Tests de la capa de persistencia del diario de decisiones (tarea #39).

Cubren el esquema cerrado de las 8 tablas, la inmutabilidad, la determinismo entre procesos, los
cuatro estados de ``plan.md`` §19.2 con la misma forma y los campos declarados ausentes de la
Fase 3. Ninguna prueba escribe bajo el ``data/`` ni el ``runs/`` del repositorio: la raiz del
diario es siempre ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from cfdtrader.backtest.engine import Direction, canonical_text
from cfdtrader.data.store import WriteOutcome
from cfdtrader.decision.gate import GateOutput, GateStatus
from cfdtrader.journal import decision_log
from cfdtrader.journal.decision_log import (
    DECISION_STATUSES,
    DIGEST_KEY,
    DIRECTIONS,
    JOURNAL_TABLES,
    LLM_OVERLAYS,
    OPS_TABLES,
    SHA256_PREFIX,
    TABLE_COLUMNS,
    TABLES,
    ClosedSchemaError,
    DecisionLogError,
    InvalidDirectionError,
    InvalidStatusError,
    Journal,
    JournalIntegrityError,
    JournalRewriteError,
    MissingIdentityError,
    RecordNotFoundError,
    UnknownTableError,
    build_decision,
    counts_by_status,
    identity_column,
    main,
    read_decision,
    read_decisions,
    read_record,
    write_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Los 24 campos obligatorios de ``journal.decisions``, en el orden de §12.5.
DECISION_COLUMNS = (
    "trade_date",
    "as_of",
    "status",
    "features_version",
    "model_version",
    "prompt_hashes",
    "git_commit",
    "prob_up_raw",
    "prob_up_calibrated",
    "expected_move_pct",
    "cost_pct",
    "ev_net_pct",
    "direction",
    "stop_pct",
    "target_pct",
    "size_notional_eur",
    "size_fraction",
    "leverage_implied",
    "tier",
    "blocking_events",
    "bull_case",
    "bear_case",
    "llm_overlay",
    "report_text",
)

_AS_OF = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
_FEATURES_VERSION = "a" * 64
_MODEL_VERSION = "b" * 64

_BASE_OUTPUT = GateOutput(
    session=date(2026, 10, 2),
    as_of=_AS_OF,
    today=date(2026, 10, 2),
    status=GateStatus.RECOMMENDATION,
    direction=Direction.NOTHING,
    tier="C",
    prob_up_calibrated=0.51,
    expected_move_pct=Decimal("1.20"),
    expected_move_basis="garch_forecast",
    cost_pct=Decimal("0.0042"),
    cost_total_pct=None,
    slippage_state="assumed",
    ev_declared_pct=Decimal("0.001"),
    ev_net_pct=None,
    stop_pct=Decimal("0.60"),
    target_pct=Decimal("0.90"),
    notional_usd=None,
    leverage_implied=None,
    bracket_required=True,
    trades_today=0,
    observation_sessions_remaining=0,
    is_fomc_session=False,
    is_half_session=False,
    fomc_dates_count=0,
    params={},
    blockers=(),
    undecided=(),
    rules=(),
    gate_sha256=SHA256_PREFIX + "0" * 64,
)


def _gate_output(**overrides: object) -> GateOutput:
    """Un ``GateOutput`` valido con los campos indicados sobrescritos."""
    return _BASE_OUTPUT.model_copy(update=dict(overrides))


def _build(**overrides: object) -> dict[str, object]:
    """``build_decision`` con los argumentos obligatorios ya fijados."""
    fields: dict[str, object] = {
        "trade_date": "2026-10-02",
        "as_of": "2026-10-02T12:00:00+00:00",
        "features_version": _FEATURES_VERSION,
        "model_version": _MODEL_VERSION,
        "git_commit": "0edfb79",
        "report_text": "estado: recommendation\n",
        "output": _BASE_OUTPUT,
    }
    fields.update(overrides)
    return build_decision(**cast("Any", fields))


def _sample_payloads() -> dict[str, dict[str, object]]:
    """Un payload valido (todas sus columnas) por cada una de las 8 tablas."""
    return {
        "decisions": _build(),
        "agent_signals": {
            "trade_date": "2026-10-02",
            "agent": "technical",
            "prob_up": 0.55,
            "confidence": 0.70,
            "veto": False,
            "veto_reason": None,
            "evidence": {"rsi": 60},
        },
        "trades": {
            "trade_date": "2026-10-02",
            "entry_price": 5000.0,
            "exit_price": 5010.0,
            "entry_time": "2026-10-02T14:30:00+00:00",
            "exit_time": "2026-10-02T20:00:00+00:00",
            "notional": 1000.0,
            "pnl_pct": 0.2,
            "costs_pct": 0.0042,
            "exit_reason": "target",
            "closed_by_close": True,
        },
        "overrides": {
            "trade_date": "2026-10-02",
            "model_recommendation": "long",
            "human_action": "skip",
            "reason": "manual",
            "declared_confidence": 0.3,
        },
        "attribution": {
            "trade_date": "2026-10-02",
            "agent": "technical",
            "would_have_won": True,
            "evidence": {"pnl_pct": 0.2},
        },
        "run_log": {
            "run_id": "run-1",
            "as_of": "2026-10-02T12:00:00+00:00",
            "stage": "ingest",
            "duration_ms": 120,
            "ok": True,
            "error": None,
        },
        "llm_calls": {
            "call_id": "call-1",
            "as_of": "2026-10-02T12:00:00+00:00",
            "provider": "openai",
            "model": "gpt",
            "system_fingerprint": "fp",
            "purpose": "overlay",
            "tokens_in": 100,
            "tokens_out": 50,
            "cache_hit": False,
            "cost_estimate": 0.01,
            "latency_ms": 800,
            "ok": True,
        },
        "backtest_runs": {
            "run_sha256": "c" * 64,
            "config": {"seed": 1},
            "metrics": {"sharpe": 0.1},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Esquema
# ─────────────────────────────────────────────────────────────────────────────
def test_table_columns_declares_the_eight_tables() -> None:
    assert set(TABLE_COLUMNS) == set(TABLES)
    assert (*JOURNAL_TABLES, *OPS_TABLES) == TABLES
    assert len(TABLES) == 8


def test_decisions_columns_are_exactly_the_24_in_order() -> None:
    assert TABLE_COLUMNS["decisions"] == DECISION_COLUMNS
    assert len(TABLE_COLUMNS["decisions"]) == 24


@pytest.mark.parametrize(
    ("table", "columns"),
    [
        (
            "agent_signals",
            ("trade_date", "agent", "prob_up", "confidence", "veto", "veto_reason", "evidence"),
        ),
        (
            "trades",
            (
                "trade_date",
                "entry_price",
                "exit_price",
                "entry_time",
                "exit_time",
                "notional",
                "pnl_pct",
                "costs_pct",
                "exit_reason",
                "closed_by_close",
            ),
        ),
        (
            "overrides",
            ("trade_date", "model_recommendation", "human_action", "reason", "declared_confidence"),
        ),
        ("attribution", ("trade_date", "agent", "would_have_won", "evidence")),
        ("run_log", ("run_id", "as_of", "stage", "duration_ms", "ok", "error")),
        (
            "llm_calls",
            (
                "call_id",
                "as_of",
                "provider",
                "model",
                "system_fingerprint",
                "purpose",
                "tokens_in",
                "tokens_out",
                "cache_hit",
                "cost_estimate",
                "latency_ms",
                "ok",
            ),
        ),
        ("backtest_runs", ("run_sha256", "config", "metrics")),
    ],
)
def test_the_other_seven_tables_have_the_declared_columns(
    table: str, columns: tuple[str, ...]
) -> None:
    assert TABLE_COLUMNS[table] == columns


@pytest.mark.parametrize("table", TABLES)
def test_identity_column_is_the_first_column_of_each_table(table: str) -> None:
    assert identity_column(table) == TABLE_COLUMNS[table][0]


@pytest.mark.parametrize("table", TABLES)
def test_round_trip_for_the_eight_tables(tmp_path: Path, table: str) -> None:
    journal = Journal(tmp_path)
    payload = _sample_payloads()[table]
    identity = payload[identity_column(table)]
    assert journal.write(table, payload) is WriteOutcome.CREATED
    assert journal.read(table, identity) == payload
    assert read_record(journal, table, identity) == payload
    assert write_record(journal, table, payload) is WriteOutcome.UNCHANGED


def test_journal_directory_and_path(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    assert journal.directory("decisions") == tmp_path / "decisions"
    assert journal.path("decisions", "2026-10-02") == tmp_path / "decisions" / "2026-10-02.json"


def test_journal_coerces_a_text_root(tmp_path: Path) -> None:
    journal = Journal(cast("Path", str(tmp_path)))
    assert journal.root == tmp_path
    journal.write("overrides", {"trade_date": "2026-10-02", "reason": "x"})
    assert read_record(journal, "overrides", date(2026, 10, 2))["reason"] == "x"


def test_the_package_reexports_the_module() -> None:
    import cfdtrader.journal as package

    assert package.decision_log is decision_log
    assert decision_log.Journal is Journal


# ─────────────────────────────────────────────────────────────────────────────
# Esquema cerrado y errores tipados
# ─────────────────────────────────────────────────────────────────────────────
def test_extra_key_is_a_closed_schema_error(tmp_path: Path) -> None:
    payload = _sample_payloads()["attribution"]
    payload["extra"] = 1
    with pytest.raises(ClosedSchemaError, match="no son columnas"):
        Journal(tmp_path).write("attribution", payload)


def test_non_text_key_is_a_closed_schema_error(tmp_path: Path) -> None:
    bad = cast("Mapping[str, object]", {"trade_date": "2026-10-02", 1: "x"})
    with pytest.raises(ClosedSchemaError):
        Journal(tmp_path).write("attribution", bad)


def test_unknown_table_is_rejected(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    with pytest.raises(UnknownTableError):
        journal.write("no_existe", {})
    with pytest.raises(UnknownTableError):
        journal.read("no_existe", "x")
    with pytest.raises(UnknownTableError):
        identity_column("no_existe")


def test_missing_identity_is_rejected(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    with pytest.raises(MissingIdentityError):
        journal.write("attribution", {"agent": "technical"})


def test_identity_must_be_a_safe_non_empty_text(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    with pytest.raises(MissingIdentityError):
        journal.write("attribution", {"trade_date": "", "agent": "technical"})
    for unsafe in (".", "..", "../x", "a/b", "a\\b"):
        with pytest.raises(MissingIdentityError, match="nombre de fichero seguro"):
            journal.write("attribution", {"trade_date": unsafe, "agent": "technical"})


def test_non_json_value_is_rejected(tmp_path: Path) -> None:
    payload = _sample_payloads()["run_log"]
    payload["error"] = object()
    with pytest.raises(DecisionLogError, match="solo admite tipos JSON"):
        Journal(tmp_path).write("run_log", payload)


def test_decimal_and_instants_are_serialized_to_json_pure(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    payload = _sample_payloads()["run_log"]
    payload["as_of"] = _AS_OF
    payload["duration_ms"] = Decimal("120")
    journal.write("run_log", payload)
    assert journal.read("run_log", "run-1")["as_of"] == "2026-10-02T12:00:00+00:00"
    assert journal.read("run_log", "run-1")["duration_ms"] == "120"


def test_record_not_found(tmp_path: Path) -> None:
    with pytest.raises(RecordNotFoundError, match="no existe la identidad"):
        Journal(tmp_path).read("decisions", "2026-01-01")


# ─────────────────────────────────────────────────────────────────────────────
# Inmutabilidad y determinismo
# ─────────────────────────────────────────────────────────────────────────────
def test_same_content_is_unchanged_and_does_not_change_the_bytes(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    path = journal.path("decisions", "2026-10-02")
    assert journal.write("decisions", _build()) is WriteOutcome.CREATED
    before = path.read_bytes()
    assert journal.write("decisions", _build()) is WriteOutcome.UNCHANGED
    assert path.read_bytes() == before


def test_different_content_raises_and_leaves_the_file_intact(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    assert journal.write("decisions", _build()) is WriteOutcome.CREATED
    path = journal.path("decisions", "2026-10-02")
    before = path.read_bytes()
    changed = _build(report_text="otro informe\n")
    with pytest.raises(JournalRewriteError, match="otro"):
        journal.write("decisions", changed)
    assert path.read_bytes() == before


def test_document_digest_is_self_consistent(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.write("decisions", _build())
    document = cast(
        "dict[str, object]",
        json.loads(journal.path("decisions", "2026-10-02").read_text(encoding="utf-8")),
    )
    digest = document.pop(DIGEST_KEY)
    assert isinstance(digest, str)
    assert digest.startswith(SHA256_PREFIX)
    assert len(digest) == len(SHA256_PREFIX) + 64
    expected = SHA256_PREFIX + hashlib.sha256(canonical_text(document).encode("utf-8")).hexdigest()
    assert digest == expected


def test_integrity_error_when_the_digest_is_tampered(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.write("decisions", _build())
    path = journal.path("decisions", "2026-10-02")
    document = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    document[DIGEST_KEY] = SHA256_PREFIX + "0" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(JournalIntegrityError, match="autoconsistente"):
        journal.read("decisions", "2026-10-02")


def test_integrity_error_when_the_digest_is_missing(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    path = journal.path("decisions", "2026-10-02")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"trade_date": "2026-10-02"}), encoding="utf-8")
    with pytest.raises(JournalIntegrityError, match="falta el digest"):
        journal.read("decisions", "2026-10-02")


def test_integrity_error_when_the_json_is_not_an_object(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    path = journal.path("decisions", "2026-10-02")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(JournalIntegrityError, match="no es un objeto JSON"):
        journal.read("decisions", "2026-10-02")


def test_integrity_error_when_the_json_is_invalid(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    path = journal.path("decisions", "2026-10-02")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{no es json", encoding="utf-8")
    with pytest.raises(JournalIntegrityError, match="JSON invalido"):
        journal.read("decisions", "2026-10-02")


_DETERMINISM_SCRIPT = """
import sys
from pathlib import Path

from cfdtrader.journal.decision_log import Journal

payload = {
    "trade_date": "2026-10-02",
    "as_of": "2026-10-02T12:00:00+00:00",
    "status": "recommendation",
    "features_version": "a" * 64,
    "model_version": "b" * 64,
    "prompt_hashes": {},
    "git_commit": "0edfb79",
    "prob_up_raw": 0.5,
    "prob_up_calibrated": 0.51,
    "expected_move_pct": 1.2,
    "cost_pct": 0.0042,
    "ev_net_pct": None,
    "direction": "nothing",
    "stop_pct": 0.6,
    "target_pct": 0.9,
    "size_notional_eur": None,
    "size_fraction": None,
    "leverage_implied": None,
    "tier": "C",
    "blocking_events": ["ev_net_no_calculable"],
    "bull_case": [],
    "bear_case": [],
    "llm_overlay": None,
    "report_text": "informe: sesión 2026-10-02",
}
Journal(Path(sys.argv[1])).write("decisions", payload)
"""


def test_determinism_across_processes(tmp_path: Path) -> None:
    first = _write_in_fresh_process(tmp_path / "seed0", "0")
    second = _write_in_fresh_process(tmp_path / "seed1", "1")
    assert first == second
    assert b"sesi\xc3\xb3n" in first  # ensure_ascii=False conserva el acento


def _write_in_fresh_process(root: Path, seed: str) -> bytes:
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(REPO_ROOT / "src")}
    subprocess.run(  # noqa: S603 - el interprete del propio entorno de pruebas
        [sys.executable, "-c", _DETERMINISM_SCRIPT, str(root)],
        check=True,
        env=env,
        cwd=REPO_ROOT,
    )
    return (root / "decisions" / "2026-10-02.json").read_bytes()


def test_module_has_no_clock_and_no_network() -> None:
    source = Path(decision_log.__file__).read_text(encoding="utf-8")
    for forbidden in ("datetime.now", "date.today", "time.time"):
        assert forbidden not in source
    for network in ("import yfinance", "import requests", "import urllib"):
        assert network not in source


# ─────────────────────────────────────────────────────────────────────────────
# Los cuatro estados de §19.2 y las versiones
# ─────────────────────────────────────────────────────────────────────────────
def test_the_four_statuses_share_the_same_shape() -> None:
    cases = [
        _build(status="recommendation", output=_gate_output(direction=Direction.NOTHING)),
        _build(
            status="recommendation",
            output=_gate_output(
                direction=Direction.LONG,
                notional_usd=Decimal("1000"),
                leverage_implied=Decimal("0.5000"),
                cost_total_pct=Decimal("0.0042"),
                ev_net_pct=Decimal("0.0010"),
            ),
            report_text="estado: recommendation\ndireccion: LONG\n",
        ),
        _build(
            status="no_recommendation_stale_data",
            output=None,
            report_text="estado: no_recommendation_stale_data\n",
        ),
        _build(
            status="no_recommendation_data_quality",
            output=None,
            report_text="estado: no_recommendation_data_quality\n",
        ),
        _build(status="error", output=None, report_text="estado: error\n"),
    ]
    expected_statuses = [
        "recommendation",
        "recommendation",
        "no_recommendation_stale_data",
        "no_recommendation_data_quality",
        "error",
    ]
    for case, expected in zip(cases, expected_statuses, strict=True):
        assert set(case) == set(DECISION_COLUMNS)
        assert case["status"] == expected
        assert isinstance(case["report_text"], str) and case["report_text"]


def test_states_without_recommendation_are_not_collapsed_into_nothing() -> None:
    long_case = _build(
        status="recommendation",
        output=_gate_output(direction=Direction.LONG, notional_usd=Decimal("1000")),
    )
    stale = _build(status="no_recommendation_stale_data", output=None)
    assert long_case["direction"] == "long"
    assert stale["direction"] is None
    assert long_case["direction"] != stale["direction"]


def test_build_decision_derives_the_fields_from_the_gate_output() -> None:
    output = _gate_output(
        direction=Direction.SHORT,
        tier="B",
        leverage_implied=Decimal("2.5000"),
        cost_total_pct=Decimal("0.0100"),
        ev_net_pct=Decimal("0.0020"),
        blockers=(
            {"rule": "13", "code": "as_of_no_es_hoy", "detail": "x"},
            {"rule": "9", "code": "ev_net_no_calculable", "detail": "y"},
        ),
    )
    payload = _build(status="recommendation", output=output)
    assert payload["direction"] == "short"
    assert payload["tier"] == "B"
    assert payload["leverage_implied"] == 2.5
    assert payload["ev_net_pct"] == 0.002
    assert payload["blocking_events"] == ["as_of_no_es_hoy", "ev_net_no_calculable"]
    assert payload["prob_up_calibrated"] == 0.51


def test_ev_net_is_null_never_zero_when_the_total_cost_is_null() -> None:
    payload = _build(
        status="recommendation",
        output=_gate_output(cost_total_pct=None, ev_net_pct=Decimal("0.5")),
    )
    assert payload["ev_net_pct"] is None


def test_blocking_events_are_the_blocker_codes() -> None:
    empty = _build()
    assert empty["blocking_events"] == []
    provided = _build(
        status="no_recommendation_data_quality",
        output=None,
        blocking_events=["calidad_de_datos", "sin_sigma"],
    )
    assert provided["blocking_events"] == ["calidad_de_datos", "sin_sigma"]


def test_features_and_model_versions_are_persisted_by_the_caller() -> None:
    payload = _build(features_version=_FEATURES_VERSION, model_version=_MODEL_VERSION)
    assert payload["features_version"] == _FEATURES_VERSION
    assert payload["model_version"] == _MODEL_VERSION
    assert payload["git_commit"] == "0edfb79"
    for field_name in ("features_version", "model_version", "git_commit", "report_text"):
        with pytest.raises(DecisionLogError):
            _build(**{field_name: ""})
        with pytest.raises(DecisionLogError):
            _build(**{field_name: cast("str", 123)})


def test_phase3_fields_are_declared_absent_until_someone_provides_them() -> None:
    payload = _build()
    assert payload["prompt_hashes"] == {}
    assert payload["bull_case"] == []
    assert payload["bear_case"] == []
    assert payload["llm_overlay"] is None
    assert payload["prob_up_raw"] is None
    assert payload["size_notional_eur"] is None
    assert payload["size_fraction"] is None
    assert set(LLM_OVERLAYS) == {
        "applied",
        "veto",
        "disabled_budget",
        "disabled_error",
        "disabled_timeout",
    }

    provided = _build(
        prob_up_raw=0.4,
        prompt_hashes={"system": SHA256_PREFIX + "d" * 64},
        bull_case=["caso alcista"],
        bear_case=["caso bajista"],
        llm_overlay="veto",
    )
    assert provided["prob_up_raw"] == 0.4
    assert provided["prompt_hashes"] == {"system": SHA256_PREFIX + "d" * 64}
    assert provided["bull_case"] == ["caso alcista"]
    assert provided["bear_case"] == ["caso bajista"]
    assert provided["llm_overlay"] == "veto"


def test_report_text_is_persisted_verbatim(tmp_path: Path) -> None:
    from cfdtrader.delivery.run_daily import render

    output = _gate_output()
    text = render(
        status=GateStatus.RECOMMENDATION,
        session=date(2026, 10, 2),
        as_of=_AS_OF,
        snapshot_session=date(2026, 10, 1),
        model_source="runs/x/model.json",
        message="pista evaluada",
        output=output,
    )
    journal = Journal(tmp_path)
    journal.write("decisions", _build(output=output, report_text=text))
    assert journal.read("decisions", "2026-10-02")["report_text"] == text


def test_trade_date_and_as_of_are_normalized(tmp_path: Path) -> None:
    payload = _build(trade_date=date(2026, 10, 2), as_of=_AS_OF)
    assert payload["trade_date"] == "2026-10-02"
    assert payload["as_of"] == "2026-10-02T12:00:00+00:00"
    from_datetime = _build(trade_date=_AS_OF)
    assert from_datetime["trade_date"] == "2026-10-02"
    with pytest.raises(DecisionLogError, match="fecha ISO"):
        _build(trade_date="no-es-fecha")
    journal = Journal(tmp_path)
    journal.write("decisions", payload)
    assert journal.read_decision(date(2026, 10, 2))["trade_date"] == "2026-10-02"
    assert read_decision(journal, date(2026, 10, 2))["trade_date"] == "2026-10-02"


def test_the_status_can_be_derived_from_the_gate_output() -> None:
    stale = _build(
        status=None,
        output=_gate_output(status=GateStatus.NO_RECOMMENDATION_STALE_DATA, direction=None),
    )
    assert stale["status"] == "no_recommendation_stale_data"
    error = _build(status=GateStatus.ERROR, output=None)
    assert error["status"] == "error"


def test_status_outside_the_four_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidStatusError, match="no_recommendation_undecided"):
        _build(status="no_recommendation_undecided", output=None)
    with pytest.raises(InvalidStatusError):
        _build(status="otro", output=None)
    with pytest.raises(InvalidStatusError, match="se requiere `status`"):
        _build(status=None, output=None)
    undecided = _gate_output(status=GateStatus.NO_RECOMMENDATION_UNDECIDED, direction=None)
    with pytest.raises(InvalidStatusError):
        _build(status=None, output=undecided)
    payload = _build()
    del payload["status"]
    with pytest.raises(InvalidStatusError, match="exige `status`"):
        Journal(tmp_path).write("decisions", payload)
    payload = _build()
    payload["status"] = "no_recommendation_undecided"
    with pytest.raises(InvalidStatusError):
        Journal(tmp_path).write("decisions", payload)


def test_direction_is_consistent_with_the_status(tmp_path: Path) -> None:
    with pytest.raises(InvalidDirectionError, match="exige `direction`"):
        _build(status="recommendation", output=_gate_output(direction=None))
    assert _build(status="error", output=None)["direction"] is None
    journal = Journal(tmp_path)
    payload = _build()
    payload["direction"] = None
    with pytest.raises(InvalidDirectionError):
        journal.write("decisions", payload)
    payload = _build()
    payload["direction"] = "up"
    with pytest.raises(InvalidDirectionError, match="direction invalida"):
        journal.write("decisions", payload)
    payload = _build(status="no_recommendation_stale_data", output=None)
    payload["direction"] = "long"
    with pytest.raises(InvalidDirectionError):
        journal.write("decisions", payload)
    assert set(DIRECTIONS) == {"long", "short", "nothing"}
    assert set(DECISION_STATUSES) == {
        "recommendation",
        "no_recommendation_stale_data",
        "no_recommendation_data_quality",
        "error",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Lectura agregada, recuento y CLI
# ─────────────────────────────────────────────────────────────────────────────
def test_read_decisions_is_ordered_by_trade_date(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    assert read_decisions(journal) == []
    for trade_date in ("2026-10-03", "2026-10-01", "2026-10-02"):
        journal.write("decisions", _build(trade_date=trade_date))
    ordered = journal.read_decisions()
    assert [record["trade_date"] for record in ordered] == [
        "2026-10-01",
        "2026-10-02",
        "2026-10-03",
    ]
    assert read_decisions(tmp_path) == ordered
    assert read_decisions(str(tmp_path)) == ordered


def test_counts_by_status(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    assert counts_by_status(journal) == dict.fromkeys(DECISION_STATUSES, 0)
    journal.write("decisions", _build(trade_date="2026-10-01"))
    journal.write("decisions", _build(trade_date="2026-10-02", status="error", output=None))
    journal.write(
        "decisions",
        _build(trade_date="2026-10-03", status="no_recommendation_stale_data", output=None),
    )
    expected = {
        "recommendation": 1,
        "no_recommendation_stale_data": 1,
        "no_recommendation_data_quality": 0,
        "error": 1,
    }
    assert counts_by_status(journal) == expected
    assert journal.counts_by_status() == expected


def test_main_prints_the_counts_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    journal = Journal(tmp_path)
    journal.write("decisions", _build(trade_date="2026-10-01"))
    journal.write("decisions", _build(trade_date="2026-10-02", status="error", output=None))
    before = sorted(path.name for path in (tmp_path / "decisions").iterdir())

    code = main(["--root", str(tmp_path)])
    captured = capsys.readouterr()

    assert code == 0
    assert "recommendation: 1" in captured.out
    assert "no_recommendation_stale_data: 0" in captured.out
    assert "no_recommendation_data_quality: 0" in captured.out
    assert "error: 1" in captured.out
    assert "total: 2" in captured.out
    assert sorted(path.name for path in (tmp_path / "decisions").iterdir()) == before
