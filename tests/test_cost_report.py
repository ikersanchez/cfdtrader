"""Informe mensual de gasto del LLM: las cuatro cifras de §6.3.5 y sus limites (tarea #117).

Lo que se blinda aqui:

- el **gasto** sale de `ops.llm_calls` y una llamada sin tarifa **no** se rellena con 0 (C4, C5);
- el **denominador** del coste por titular sale del `manifest` (#129), y una sesion **sin** conteo
  se cuenta aparte en vez de inventarle titulares (C8);
- el gasto evitado por la deduplicacion viaja **etiquetado como estimacion** (C9);
- el contrato de **#119** se respeta: una fila de acierto no suma tokens ni gasto (C6, C7);
- el informe es **determinista**, **solo lee** y **no** escribe fuera de `--reports-dir` (C11, C12);
- los nombres de los conteos que lee son **exactamente** los que escribe el camino diario (C14).

No hay modulo nuevo de `tests` que cubra: el modulo nuevo es `ops/cost_report.py`, y las pruebas de
abajo lo recorren entero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import pytest

from cfdtrader.delivery import run_daily
from cfdtrader.journal.decision_log import TABLE_COLUMNS, Journal
from cfdtrader.llm.budget import (
    MONTHLY_WARNING,
    PRICES_EUR_PER_MTOKENS,
    PRICES_VERIFIED_ON,
    BatchCounts,
)
from cfdtrader.ops import cost_report

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MONTH: Final[str] = "2026-09"
SESSION: Final[str] = "2026-09-17"
AS_OF: Final[str] = "2026-09-17T12:45:00+00:00"

#: Una llamada del fixture: (cache_hit, coste, tokens_in, tokens_out, modelo).
Call = tuple[bool, float | None, int | None, int | None, str]

REAL: Final[Call] = (False, 0.000321, 814, 1730, "deepseek-flash")
HIT: Final[Call] = (True, None, None, None, "deepseek-flash")
NO_PRICE: Final[Call] = (False, None, 100, 50, "modelo-sin-tarifa")


def _write_call(journal: Journal, index: int, call: Call) -> None:
    hit, cost, tokens_in, tokens_out, model = call
    journal.write(
        "llm_calls",
        {
            "call_id": f"20260917T124500Z-{index:04d}",
            "as_of": AS_OF,
            "provider": "deepseek",
            "model": model,
            "system_fingerprint": None,
            "purpose": "extract",
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cache_hit": hit,
            "cost_estimate": cost,
            "latency_ms": 12,
            "ok": True,
        },
    )


def _write_manifest(
    journal_root: Path, *, counters: dict[str, int] | None, session: str = SESSION
) -> None:
    directory = journal_root / "ops" / session
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "run_id": session,
        "as_of": AS_OF,
        "git_commit": "0" * 40,
        "versions": {},
        "hashes": {},
        "stages": [],
        "ok": True,
    }
    if counters is not None:
        payload["counters"] = counters
    (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _counters(*, sent: int = 2, duplicates: int = 2, read: int = 5) -> dict[str, int]:
    return {
        "headlines_read": read,
        "headlines_duplicates": duplicates,
        "headlines_out_of_window": read - duplicates - sent,
        "headlines_prepared": sent,
        "headlines_sent": sent,
    }


@pytest.fixture
def journal_root(tmp_path: Path) -> Path:
    """Un diario con tres llamadas (real, acierto y sin tarifa) y una sesion con conteo."""
    root = tmp_path / "journal"
    handle = Journal(root / "ops")
    for index, call in enumerate((REAL, HIT, NO_PRICE), start=1):
        _write_call(handle, index, call)
    _write_manifest(root, counters=_counters())
    return root


def _report(root: Path, month: str = MONTH) -> dict[str, Any]:
    return cost_report.build_report(root, month)


def _snapshot(root: Path) -> dict[str, bytes]:
    """El contenido del arbol, para comprobar que el informe **no** lo toca (C12)."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# ─────────────────────────────────────────────────────────────────────────────
# C2 · Se lee por la API del diario (#39), no abriendo los JSON a mano
# ─────────────────────────────────────────────────────────────────────────────
def test_c2_the_calls_are_read_through_the_journal_api(journal_root: Path) -> None:
    rows = cost_report.read_calls(journal_root)
    assert len(rows) == 3
    assert set(rows[0]) == set(TABLE_COLUMNS["llm_calls"])


def test_c2_a_journal_without_calls_is_empty_and_not_an_error(tmp_path: Path) -> None:
    assert cost_report.read_calls(tmp_path / "no-existe") == ()


# ─────────────────────────────────────────────────────────────────────────────
# C3 · El mes se declara; un mes mal formado es error tipado
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad", ["2026/09", "2026-13", "", "septiembre", "2026"])
def test_c3_a_malformed_month_is_a_typed_error(bad: str) -> None:
    with pytest.raises(cost_report.ConfigurationError):
        cost_report.parse_month(bad)


def test_c3_the_cli_returns_2_and_writes_nothing(tmp_path: Path) -> None:
    reports = tmp_path / "rep"
    common = ["--journal-root", str(tmp_path), "--reports-dir", str(reports)]
    assert cost_report.main(["--month", "2026/09", *common]) == 2
    assert cost_report.main(["--month", MONTH]) == 2, "sin diario no hay llamadas que sumar"
    assert not reports.exists(), "un 2 no escribe nada"


# ─────────────────────────────────────────────────────────────────────────────
# C4/C5 · Gasto acumulado, y un coste desconocido que no se inventa
# ─────────────────────────────────────────────────────────────────────────────
def test_c4_the_spend_is_the_sum_of_the_declared_costs(journal_root: Path) -> None:
    spend = _report(journal_root)["spend_eur"]
    assert spend["total"] == "0.000321"
    assert spend["state"] == "measured"


def test_c5_a_call_without_a_tariff_is_counted_and_never_a_zero(journal_root: Path) -> None:
    report = _report(journal_root)
    assert report["calls"]["without_declared_cost"] == 2
    assert report["spend_eur"]["total"] != "0"
    assert report["spend_eur"]["complete"] is False
    assert "minimo conocido" in report["spend_eur"]["reason"]


def test_c5_a_month_without_calls_is_not_measurable_not_zero(tmp_path: Path) -> None:
    report = _report(tmp_path)
    assert report["calls"]["total"] == 0
    assert report["spend_eur"]["total"] is None
    assert report["spend_eur"]["state"] == "not_measurable"


# ─────────────────────────────────────────────────────────────────────────────
# C6/C7 · Aciertos de cache (#119) y tokens
# ─────────────────────────────────────────────────────────────────────────────
def test_c6_the_cache_hit_ratio_separates_hits_from_real_calls(journal_root: Path) -> None:
    calls = _report(journal_root)["calls"]
    assert (calls["total"], calls["real"], calls["cache_hits"]) == (3, 2, 1)
    assert calls["cache_hit_ratio"] == "0.333333"


def test_c7_a_cache_hit_adds_no_tokens(journal_root: Path) -> None:
    """El contrato de #119: la fila del acierto declara `null`, asi que no suma."""
    assert _report(journal_root)["tokens"] == {
        "in": 914,
        "out": 1780,
        "calls_without_tokens": 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# C8 · Coste por titular, con el denominador que persiste #129
# ─────────────────────────────────────────────────────────────────────────────
def test_c8_the_cost_per_headline_uses_the_persisted_denominator(journal_root: Path) -> None:
    per = _report(journal_root)["per_headline_eur"]
    assert per["headlines_sent"] == 2
    assert per["headlines_read"] == 5
    assert per["value"] == "0.00016", "0,000321 del mes entre 2 titulares enviados"
    assert per["state"] == "measured"
    assert per["sessions_with_counters"] == 1


def test_c8_a_session_without_counters_is_counted_apart(tmp_path: Path) -> None:
    """Una sesion sin conteo **no** aporta titulares: se declara, no se inventa."""
    root = tmp_path / "journal"
    _write_call(Journal(root / "ops"), 1, REAL)
    _write_manifest(root, counters=None)
    _write_manifest(root, counters=_counters(), session="2026-09-18")

    per = _report(root)["per_headline_eur"]
    assert per["sessions_with_counters"] == 1
    assert per["sessions_without_counters"] == 1
    assert per["headlines_sent"] == 2


def test_c8_without_any_counter_the_figure_is_null_with_a_reason(tmp_path: Path) -> None:
    root = tmp_path / "journal"
    _write_call(Journal(root / "ops"), 1, REAL)
    per = _report(root)["per_headline_eur"]
    assert per["value"] is None
    assert per["state"] == "not_measurable"
    assert "90 dias" in per["reason"], "el limite de retencion tiene que estar en el motivo"


# ─────────────────────────────────────────────────────────────────────────────
# C9 · El gasto que la deduplicacion evito va ETIQUETADO como estimacion
# ─────────────────────────────────────────────────────────────────────────────
def test_c9_the_avoided_spend_is_labelled_as_an_estimate(journal_root: Path) -> None:
    avoided = _report(journal_root)["avoided_spend_eur"]
    assert avoided["is_estimate"] is True
    assert avoided["duplicates"] == 2
    assert avoided["value"] == "0.000321", "2 duplicados x 0,00016 por titular"
    assert "contrafactual" in avoided["reason"], "no puede presentarse como medido"


# ─────────────────────────────────────────────────────────────────────────────
# C10 · Los precios viajan con su fecha
# ─────────────────────────────────────────────────────────────────────────────
def test_c10_the_report_declares_the_price_table_and_its_date(journal_root: Path) -> None:
    prices = _report(journal_root)["prices"]
    assert prices["verified_on"] == PRICES_VERIFIED_ON
    assert set(prices["table_eur_per_mtoken"]) == set(PRICES_EUR_PER_MTOKENS)


# ─────────────────────────────────────────────────────────────────────────────
# C11/C12 · Determinismo y solo lectura
# ─────────────────────────────────────────────────────────────────────────────
def test_c11_the_same_month_gives_the_same_report(journal_root: Path, tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    for target in (first, second):
        assert (
            cost_report.main(
                [
                    "--month",
                    MONTH,
                    "--journal-root",
                    str(journal_root),
                    "--reports-dir",
                    str(target),
                ]
            )
            == 0
        )
    for suffix in ("json", "md"):
        name = f"llm_cost_{MONTH}.{suffix}"
        assert (first / name).read_text(encoding="utf-8") == (second / name).read_text(
            encoding="utf-8"
        )


def test_c12_nothing_is_written_outside_the_reports_dir(
    journal_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _snapshot(journal_root)
    reports = tmp_path / "rep"
    assert (
        cost_report.main(
            ["--month", MONTH, "--journal-root", str(journal_root), "--reports-dir", str(reports)]
        )
        == 0
    )
    capsys.readouterr()

    assert _snapshot(journal_root) == before, "el diario es de solo lectura para este comando"
    assert {path.name for path in reports.iterdir()} == {
        f"llm_cost_{MONTH}.json",
        f"llm_cost_{MONTH}.md",
    }


# ─────────────────────────────────────────────────────────────────────────────
# C13 · El tope mensual
# ─────────────────────────────────────────────────────────────────────────────
def test_c13_a_breached_monthly_cap_carries_the_warning(
    journal_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_MONTHLY_BUDGET_EUR", "0.0001")
    cap = _report(journal_root)["monthly_cap_eur"]
    assert cap["declared"] == "0.0001"
    assert cap["breached"] is True
    assert cap["warning"] == MONTHLY_WARNING


def test_c13_without_a_declared_cap_nothing_is_invented(
    journal_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_MONTHLY_BUDGET_EUR", "")
    cap = _report(journal_root)["monthly_cap_eur"]
    assert cap == {"declared": None, "breached": False, "warning": None}


# ─────────────────────────────────────────────────────────────────────────────
# C14 · CLI manual, contrato de nombres e informe de ejemplo
# ─────────────────────────────────────────────────────────────────────────────
def test_c14_the_help_says_the_command_is_manual() -> None:
    text = cost_report._build_parser().format_help()  # pyright: ignore[reportPrivateUsage]
    assert "manual" in text
    assert "scheduler" in text


def test_c14_the_counter_names_are_the_ones_the_daily_path_writes() -> None:
    """Si #129 renombra un conteo, esto falla aqui en vez de dejar el denominador vacio."""
    counters = run_daily._headline_counters(  # pyright: ignore[reportPrivateUsage]
        BatchCounts(read=5, duplicates=2, out_of_window=1, sent=2, truncated=0), sent=0
    )
    assert set(cost_report.COUNTERS_READ) <= set(counters)


def test_c14_the_example_report_is_committed() -> None:
    example = REPO_ROOT / "examples" / "llm_cost_ejemplo_2026-09.md"
    assert example.is_file(), "la tarea pide un informe de ejemplo en `examples/`"
    assert "estimacion" in example.read_text(encoding="utf-8")
