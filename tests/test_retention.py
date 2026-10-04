"""Tests de la retención y la higiene de disco (tarea #44): A1-A13.

Lo que se blinda aquí:

- Los **presupuestos** y las **edades** son los declarados (§12.7), no números sueltos.
- `sizes` mide, marca lo que se pasa y **no borra nada**.
- `retention` **no borra sin `--apply`**: la simulación es el modo por defecto (A8).
- Un **modelo de producción no se purga jamás**: su `run_sha256` aparece como `model_version`
  en el journal (A12).

Ninguna prueba abre red ni escribe fuera de `tmp_path`.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cfdtrader.ops import retention
from cfdtrader.ops.retention import (
    BUDGETS,
    COMPONENTS,
    RETENTION_POLICIES,
    Component,
    ComponentSize,
    ConfigurationError,
    Roots,
    SizesReport,
    apply_retention,
    check_budgets,
    component_sizes,
    main,
    plan_retention,
    render_markdown,
)

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def _roots(tmp_path: Path) -> Roots:
    return Roots(data=tmp_path / "data", journal=tmp_path / "journal", runs=tmp_path / "runs")


def _write(path: Path, *, size: int = 16, age_days: int = 0) -> Path:
    """Crea un fichero con ese tamaño y esa antigüedad (respecto a ``NOW``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    when = (NOW - timedelta(days=age_days)).timestamp()
    os.utime(path, (when, when))
    return path


def _production_only_aaa(_journal: object) -> list[dict[str, str]]:
    """Doble del diario: solo el run `aaa` es producción (para A12)."""
    return [{"model_version": "aaa"}]


# ─────────────────────────────────────────────────────────────────────────────
# A2 · Los presupuestos están declarados
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_budgets_are_declared() -> None:
    assert set(BUDGETS) == {
        "raw",
        "derived",
        "http_cache",
        "journal",
        "ops_run_log",
        "ops_llm_cache",
        "runs",
    }
    assert all(budget > 0 for budget in BUDGETS.values())
    assert {component.name: component.budget_bytes for component in COMPONENTS} == BUDGETS


# ─────────────────────────────────────────────────────────────────────────────
# A3 · `sizes` mide cada componente y marca el exceso
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_component_sizes_measure_the_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retention, "COMPONENTS", (Component("raw", "data", "raw", 5),))
    roots = _roots(tmp_path)
    _write(roots.data / "raw" / "a.bin", size=10)
    _write(roots.data / "raw" / "sub" / "b.bin", size=10)

    (row,) = component_sizes(roots)

    assert row.bytes == 20
    assert row.over_budget is True, "20 > 5: se pasa del presupuesto"


def test_a3_a_missing_component_measures_zero(tmp_path: Path) -> None:
    rows = {row.component: row for row in component_sizes(_roots(tmp_path))}
    assert rows["raw"].bytes == 0
    assert rows["raw"].over_budget is False


# ─────────────────────────────────────────────────────────────────────────────
# A4 · El informe de `sizes` es determinista y se escribe donde toca
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_sizes_report_is_deterministic_and_written(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _write(roots.data / "raw" / "a.bin", size=100)

    first = check_budgets(roots, as_of=NOW)
    second = check_budgets(roots, as_of=NOW)

    assert isinstance(first, SizesReport)
    assert first.model_dump() == second.model_dump(), "mismo as_of, mismo informe"
    assert first.as_of == NOW.isoformat()

    reports = tmp_path / "reports"
    code = main(
        [
            "sizes",
            "--as-of",
            NOW.isoformat(),
            "--data-root",
            str(roots.data),
            "--journal-root",
            str(roots.journal),
            "--runs-root",
            str(roots.runs),
            "--reports-dir",
            str(reports),
        ]
    )
    assert code == 0
    json_path = reports / f"disk_sizes_{NOW.date().isoformat()}.json"
    assert json_path.is_file()
    assert json.loads(json_path.read_text(encoding="utf-8"))["as_of"] == NOW.isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# A5 · El componente por encima de presupuesto avisa
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_an_over_budget_component_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retention, "COMPONENTS", (Component("raw", "data", "raw", 5),))
    roots = _roots(tmp_path)
    _write(roots.data / "raw" / "a.bin", size=50)

    report = check_budgets(roots, as_of=NOW)

    assert report.over_budget == ("raw",)
    assert "AVISO: por encima" in render_markdown(report)


# ─────────────────────────────────────────────────────────────────────────────
# A6 · `sizes` no borra nada y escribe solo en `--reports-dir`
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_sizes_writes_only_the_report(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    keep = _write(roots.data / "raw" / "a.bin", size=100)
    reports = tmp_path / "reports"

    code = main(
        [
            "sizes",
            "--as-of",
            NOW.isoformat(),
            "--data-root",
            str(roots.data),
            "--journal-root",
            str(roots.journal),
            "--runs-root",
            str(roots.runs),
            "--reports-dir",
            str(reports),
        ]
    )

    assert code == 0
    assert keep.is_file(), "sizes no borra nada"
    assert {path.name for path in reports.iterdir()} == {
        f"disk_sizes_{NOW.date().isoformat()}.json",
        f"disk_sizes_{NOW.date().isoformat()}.md",
    }


# ─────────────────────────────────────────────────────────────────────────────
# A7 · Las tres retenciones están declaradas
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_the_retention_ages_are_the_declared_ones() -> None:
    ages = {policy.target: policy.max_age_days for policy in RETENTION_POLICIES}
    assert ages == {"run_log": 90, "llm_cache": 540, "model_binaries": 90}


# ─────────────────────────────────────────────────────────────────────────────
# A8 · La simulación es el modo por defecto: sin `--apply` no se borra nada
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_dry_run_never_deletes(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    log = _write(roots.journal / "ops" / "s1" / "run_log.jsonl", age_days=200)

    plan = plan_retention(roots, as_of=NOW)  # apply=False por defecto

    assert plan.apply is False
    assert len(plan.deletions) == 1
    assert log.is_file(), "sin --apply, el run_log sigue ahí"


# ─────────────────────────────────────────────────────────────────────────────
# A9 · `--apply` borra y cuenta; un `as_of` sin zona es error tipado
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_apply_deletes_and_counts(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    log = _write(roots.journal / "ops" / "s1" / "run_log.jsonl", size=32, age_days=200)
    manifest = _write(roots.journal / "ops" / "s1" / "manifest.json", size=8, age_days=200)

    plan = plan_retention(roots, as_of=NOW, apply=True)
    outcome = apply_retention(plan)

    assert len(outcome.deleted_paths) == 1
    assert not log.exists() and not manifest.exists(), "se borra la sesión entera"
    assert outcome.deleted_bytes >= 40
    assert outcome.failures == ()


def test_a9_a_naive_as_of_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        check_budgets(_roots(tmp_path), as_of=datetime(2026, 10, 4, 12, 0))


def test_a9_apply_tolerates_a_missing_path(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    log = _write(roots.journal / "ops" / "s1" / "run_log.jsonl", age_days=200)
    plan = plan_retention(roots, as_of=NOW, apply=True)
    log.unlink()
    log.parent.rmdir()  # la sesion entera desaparece entre el plan y la aplicacion

    outcome = apply_retention(plan)

    assert outcome.deleted_paths == ()
    assert outcome.failures == ()


def test_a9_apply_reports_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _roots(tmp_path)
    _write(roots.journal / "ops" / "s1" / "run_log.jsonl", age_days=200)
    plan = plan_retention(roots, as_of=NOW, apply=True)

    def _boom(_path: str) -> None:
        raise OSError("disco ocupado")

    monkeypatch.setattr(retention.shutil, "rmtree", _boom)
    outcome = apply_retention(plan)

    assert outcome.deleted_paths == ()
    assert len(outcome.failures) == 1 and "disco ocupado" in outcome.failures[0]


# ─────────────────────────────────────────────────────────────────────────────
# A10 · La purga de `run_log` solo toca las sesiones viejas
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_only_old_run_logs_are_purged(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    old = _write(roots.journal / "ops" / "old" / "run_log.jsonl", age_days=200)
    fresh = _write(roots.journal / "ops" / "fresh" / "run_log.jsonl", age_days=10)
    _write(roots.journal / "ops" / "noise" / "other.txt", age_days=200)  # sin run_log

    plan = plan_retention(roots, as_of=NOW, apply=True)
    paths = {deletion.path for deletion in plan.deletions}

    assert str(old.parent) in paths
    assert str(fresh.parent) not in paths
    assert not any("noise" in path for path in paths)


# ─────────────────────────────────────────────────────────────────────────────
# A11 · La caché del LLM se purga a los 18 meses
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_llm_cache_entries_older_than_18_months_are_purged(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    old = _write(roots.journal / "ops" / "llm_cache" / "xx" / "entry", age_days=600)
    fresh = _write(roots.journal / "ops" / "llm_cache" / "yy" / "entry", age_days=10)

    plan = plan_retention(roots, as_of=NOW, apply=True)
    paths = {deletion.path for deletion in plan.deletions}

    assert str(old) in paths
    assert str(fresh) not in paths


# ─────────────────────────────────────────────────────────────────────────────
# A12 · Binarios de modelos: se purga el no productivo, jamás el de producción
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_model_binaries_keep_the_production_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    production = _write(roots.runs / "aaa" / "model.json", age_days=200)
    experiment = _write(roots.runs / "bbb" / "model.json", age_days=200)
    monkeypatch.setattr(retention, "read_decisions", _production_only_aaa)

    plan = plan_retention(roots, as_of=NOW, apply=True)
    paths = {deletion.path for deletion in plan.deletions}

    assert plan.protected_runs == ("aaa",)
    assert str(experiment) in paths
    assert str(production) not in paths, "un modelo de producción no se purga jamás"


# ─────────────────────────────────────────────────────────────────────────────
# A13 · El CLI expone las dos operaciones con sus códigos
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_the_cli_has_two_commands_and_the_declared_codes(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _write(roots.journal / "ops" / "s1" / "run_log.jsonl", age_days=200)
    common = [
        "--as-of",
        NOW.isoformat(),
        "--data-root",
        str(roots.data),
        "--journal-root",
        str(roots.journal),
        "--runs-root",
        str(roots.runs),
        "--reports-dir",
        str(tmp_path / "reports"),
    ]

    assert main(["sizes", *common]) == 0
    assert main(["retention", *common]) == 0  # sin --apply: simula

    # Código 2: `--as-of` ausente o sin zona, y `--journal-root` ausente.
    assert main(["retention", "--journal-root", str(roots.journal)]) == 2
    naive = ["sizes", "--as-of", "2026-10-04T12:00:00", "--journal-root", str(roots.journal)]
    assert main(naive) == 2
    assert main(["sizes", "--as-of", NOW.isoformat()]) == 2

    assert "manual" in (retention.__doc__ or "").lower()
    assert {policy.target for policy in RETENTION_POLICIES} == {
        "run_log",
        "llm_cache",
        "model_binaries",
    }


def test_a13_the_retention_report_is_rendered(tmp_path: Path) -> None:
    plan = plan_retention(_roots(tmp_path), as_of=NOW)
    text = render_markdown(plan)
    assert "Retencion e higiene de disco" in text
    assert "simulacion" in text

    row = ComponentSize(component="raw", path="p", bytes=1, budget_bytes=2, over_budget=False)
    sizes = SizesReport(
        as_of=NOW.isoformat(), components=(row,), total_bytes=1, total_budget_bytes=2
    )
    assert render_markdown(sizes).startswith("# Higiene de disco")


# ─────────────────────────────────────────────────────────────────────────────
# Bordes: raíz desconocida, tamaños legibles y `retention --apply` de punta a punta
# ─────────────────────────────────────────────────────────────────────────────
def test_an_unknown_root_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        _roots(tmp_path).base("nope")


def test_budget_sizes_are_rendered_in_human_units() -> None:
    rows = (
        ComponentSize(component="a", path="p", bytes=1536, budget_bytes=1536, over_budget=False),
        ComponentSize(
            component="b", path="q", bytes=2 * 1024**3, budget_bytes=3 * 1024**3, over_budget=False
        ),
    )
    report = SizesReport(
        as_of=NOW.isoformat(), components=rows, total_bytes=1, total_budget_bytes=2
    )

    text = render_markdown(report)

    assert "1.5 KB" in text
    assert "2.0 GB" in text


def test_a13_the_cli_apply_deletes_end_to_end(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    log = _write(roots.journal / "ops" / "s1" / "run_log.jsonl", age_days=200)

    code = main(
        [
            "retention",
            "--apply",
            "--as-of",
            NOW.isoformat(),
            "--data-root",
            str(roots.data),
            "--journal-root",
            str(roots.journal),
            "--runs-root",
            str(roots.runs),
            "--reports-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert code == 0
    assert not log.exists(), "con --apply el run_log viejo desaparece"
