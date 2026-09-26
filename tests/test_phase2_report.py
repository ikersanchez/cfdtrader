"""Pruebas del informe de Fase 2 y de sus criterios de *kill* (`tasks.md`, tarea 29) — #29.

Un test por criterio (``test_a1_...`` … ``test_a21_...``), siempre con ``tmp_path`` y la
fixture de sesion de ``tests/conftest.py`` que **huella ``data/`` y ``runs/``**. Los bordes
incomodos (artefacto ausente, dos artefactos ambiguos, ``--as-of`` ausente, segunda pasada,
hash manipulado, ``write=False`` y corridas sinteticas de resultados opuestos) tienen su
propio caso: en este proyecto los defectos aparecen al **reejecutar**, no en la primera
pasada.

Las corridas sinteticas **no** re-ejecutan #28 ni #26 (eso serian minutos por test): se
construye el payload minimo que el informe consume y se le pasa la funcion **pura**
``evaluate_criteria``. Con dos payloads de resultados opuestos se comprueba que los umbrales
publicados son identicos y no se derivan de los resultados (A6, A11).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final, cast

import pytest

from cfdtrader.analysis import phase0_report, phase2_report
from cfdtrader.analysis.phase0_report import GateVerdict, HalfResult, aggregate_gate
from cfdtrader.analysis.phase2_report import (
    A13_DIVERGENCE,
    AmbiguousInputArtifactError,
    InvalidKillTableError,
    MissingInputArtifactError,
    MissingInputFieldError,
    VerdictError,
)
from cfdtrader.backtest.engine import canonical_text
from cfdtrader.data.store import Store

MODULE_PATH: Final[Path] = Path(str(phase2_report.__file__))
SOURCE: Final[str] = MODULE_PATH.read_text(encoding="utf-8")
TEST_PATH: Final[Path] = Path(__file__).resolve()
REPO_ROOT: Final[Path] = TEST_PATH.parents[1]
REAL_REPORTS: Final[Path] = REPO_ROOT / "data" / "derived" / "reports"
PIPELINE_ARTIFACT: Final[Path] = REAL_REPORTS / "pipeline_backtest_2026-09-23.json"
MODEL_ARTIFACT: Final[Path] = REAL_REPORTS / "model_comparison_2026-09-22.json"

#: Instante declarado de las corridas que fijan el hash dorado (A4).
NOW: Final[datetime] = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)

# #96: aqui vivia `GOLDEN_REPORT_SHA256`, el `report_sha256` dorado de la corrida real. Ese
# digest es el del payload de **este** informe, y dentro viaja el `sha256` del artefacto de #28
# (que #92 regenero al publicar `hit_rate_per_trade`), asi que **cualquier** tarea posterior lo
# invalida: un dorado asi convierte un trabajo ajeno en un fallo de #29. No se vuelve a cablear
# ningun digest regenerable. Lo que A4 vigila —prefijo `sha256:` + 64 hex, autoconsistencia del
# payload y determinismo entre procesos— se comprueba sin literal. Los unicos dorados estables
# de la fase son el `sha256` del bloque §11.6 de `plan.md` y las nueve filas, que siguen abajo.

#: `sha256` dorado del bloque de la tabla de §11.6 (A6): si `plan.md` cambia, esto falla.
GOLDEN_PLAN_SHA256: Final[str] = "3a42a85708eb78f4ca31dda740cdf5c6c6b6ece2bf350f59b5a922fadc6987f1"

#: Las nueve filas **literales** de §11.6, copiadas del documento (A6). Si una fila o un
#: umbral se desvia, el test falla.
GOLDEN_KILL_ROWS: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "⭐ **¿Existe edge demostrable?**",
        "**El IC 95 % de la tasa de acierto debe excluir el $p^*$ de break-even** (≈50,2 %), "
        "o el Sharpe OOS debe tener IC 95 % que excluya 0",
        "Parar. **Es el criterio principal, y con muestras pequeñas suele fallar por falta de "
        "datos, no por falta de edge**",
    ),
    ("PBO", "< 20%", "Simplificar modelo"),
    ("Deflated Sharpe", "> 0 significativo", "Rechazar la variante"),
    ("Drawdown máximo", "< 20% del capital", "Reducir tamaño o parar"),
    ('Bate a "no operar"', "Sí, significativamente", "Parar"),
    ("Bate a `siempre largo open→close`", "Sí, significativamente", "No hay alpha: reformular"),
    (
        "Bate a `siempre largo close→close` (CFD, con financiación)",
        "Sí, pero **insuficiente por sí solo**",
        "Si no lo bate, es grave: el sistema estaría peor que no hacer nada",
    ),
    ("Divergencia paper vs backtest", "< 2σ durante 3 meses", "Parar y auditar"),  # noqa: RUF001
    (
        "**Incumplimiento del cierre a las 16:00 ET** (22:00 Madrid)",
        "Cero tolerancia",
        "Revisar el mecanismo antes de seguir",
    ),
)

#: Ficheros congelados que A20 prohibe tocar. `pyproject.toml`/`uv.lock` no entran: su
#: cambio (`pytest-xdist` en el grupo `dev`) esta **autorizado por el propietario** y se
#: declara como **excepcion explicita a A21** en el comentario de la entrega de #29.
FORBIDDEN_PATHS: Final[frozenset[str]] = frozenset(
    {
        "src/cfdtrader/analysis/pipeline_report.py",
        "src/cfdtrader/analysis/model_comparison.py",
        "src/cfdtrader/analysis/experiment_log.py",
        "src/cfdtrader/analysis/phase0_report.py",
        "src/cfdtrader/analysis/phase1_report.py",
        "src/cfdtrader/backtest/engine.py",
        "src/cfdtrader/backtest/baselines.py",
        "src/cfdtrader/backtest/metrics.py",
        "src/cfdtrader/backtest/overfitting.py",
        "src/cfdtrader/backtest/costs.py",
        "src/cfdtrader/backtest/splits.py",
        "src/cfdtrader/decision/gate.py",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades y corridas sinteticas
# ─────────────────────────────────────────────────────────────────────────────
def _stars(pipeline: Mapping[str, object]) -> dict[str, object]:
    """Bloque `p*` derivado del coste declarado del payload, como lo hace `analyse`."""
    scenario = cast("Mapping[str, object]", pipeline["scenario"])
    return phase2_report.p_star_block(
        cost_pct=Decimal(cast("str", scenario["cost_basis_pct"])),
        cost_source=cast("str", scenario["cost_provenance"]),
    )


def synthetic_pipeline(
    *,
    net: Mapping[str, object] | None = None,
    drawdown: float | None = 5.88,
    cost_pct: str = "0.0042",
) -> dict[str, object]:
    """Payload minimo de #28 que el informe consume, sin re-ejecutar el pipeline."""
    metrics: dict[str, object] = {
        "max_drawdown_pct": {
            "estimate": drawdown,
            "lower": None if drawdown is None else drawdown / 3.0,
            "upper": None if drawdown is None else drawdown * 2.0,
            "basis": "declared_cost",
        }
    }
    return {
        "analysis": "cfdtrader.analysis.pipeline_report",
        "task": "#28",
        "generated_at": "2026-09-24T00:00:00+00:00",
        "report_sha256": "sha256:" + "0" * 64,
        "scenario": {"cost_basis_pct": cost_pct, "cost_provenance": "sintetico"},
        "table": {
            "rows": [
                {"row": "no_trade", "kind": "baseline", "is_invertible": True},
                {"row": "always_long", "kind": "baseline", "is_invertible": True},
                {"row": "always_short", "kind": "baseline", "is_invertible": True},
                {"row": "momentum_5d", "kind": "baseline", "is_invertible": True},
                {"row": "gap_reversal", "kind": "baseline", "is_invertible": True},
                {"row": "random_matched", "kind": "baseline", "is_invertible": True},
                {"row": "liston_a", "kind": "liston_a", "is_invertible": True},
                {"row": "liston_b", "kind": "liston_b", "is_invertible": True},
                {"row": "liston_c", "kind": "liston_c", "is_invertible": False},
            ]
        },
        "arms": {
            "coste_declarado": {"traded": 10, "no_trade": 490, "metrics": metrics},
        },
        "net_metrics": dict(net) if net is not None else _net_not_computable(),
    }


def _net_not_computable() -> dict[str, object]:
    """Bloque `net_metrics` de #28 tal cual lo publica hoy."""
    return {
        "state": "not_computable",
        "reason": "`pnl_net_pct` es `null` en todas las operaciones",
        "where": "cfdtrader.backtest.metrics.calculate_metrics",
        "follow_ups": ["#62", "#60"],
    }


def _net_computable(
    *, hit: tuple[float, float], sharpe: tuple[float, float], beats: Mapping[str, str]
) -> dict[str, object]:
    """Bloque `net_metrics` **computable** (corrida sintetica) para cubrir las ramas A9/A15."""
    return {
        "state": "computed",
        "basis": "net",
        "hit_rate": {"lower": hit[0], "upper": hit[1]},
        "sharpe": {"lower": sharpe[0], "upper": sharpe[1]},
        "beats": dict(beats),
    }


def synthetic_model(
    *, pbo: float = 0.05, dsr_half: str = "pass", dsr_value: float = 0.97
) -> dict[str, object]:
    """Payload minimo de #26 que el informe consume, sin recalcular PBO ni DSR."""
    return {
        "analysis": "cfdtrader.analysis.model_comparison",
        "task": "#26",
        "generated_at": "2026-09-24T00:00:00+00:00",
        "report_sha256": "sha256:" + "1" * 64,
        "deflated_sharpe_ratio": {"verdict": "significant", "dsr": dsr_value},
        "probability_of_backtest_overfitting": {
            "state": "evaluated",
            "pbo": pbo,
            "blocks": 10,
            "n_observations": 500,
            "verdict": "not_detected" if pbo < 0.2 else "detected",
        },
        "verdict": {
            "gate": {
                "halves": {
                    "deflated_sharpe_ratio": dsr_half,
                    "probability_of_backtest_overfitting": "pass",
                }
            }
        },
    }


def evaluate(
    pipeline: Mapping[str, object], model: Mapping[str, object]
) -> list[dict[str, object]]:
    """Evalua las nueve filas con la tabla real de §11.6 y un payload sintetico."""
    table = phase2_report.load_kill_table()
    return phase2_report.evaluate_criteria(
        pipeline=pipeline, model=model, table=table, stars=_stars(pipeline)
    )


def _row_of(criteria: Sequence[Mapping[str, object]], kind: str) -> Mapping[str, object]:
    """La fila evaluada con ese identificador, o fallo claro."""
    for entry in criteria:
        if entry["kind"] == kind:
            return entry
    raise AssertionError(f"no hay fila {kind!r} en {[e['kind'] for e in criteria]}")


def _run_module(*, reports_dir: Path, data_root: Path, as_of: str, seed: str | None) -> str:
    """Ejecuta el modulo en un **proceso nuevo** con su `PYTHONHASHSEED` (A4)."""
    environment = dict(os.environ)
    if seed is not None:
        environment["PYTHONHASHSEED"] = seed
    result = subprocess.run(  # noqa: S603 - el ejecutable es el interprete de la sesion
        [
            sys.executable,
            "-m",
            "cfdtrader.analysis.phase2_report",
            "--data-root",
            str(data_root),
            "--reports-dir",
            str(reports_dir),
            "--as-of",
            as_of,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture()
def reports(tmp_path: Path) -> Path:
    """Copia los artefactos reales de #28/#26 (solo lectura) dentro de `tmp_path`."""
    target = tmp_path / "derived" / "reports"
    target.mkdir(parents=True)
    for artifact in (PIPELINE_ARTIFACT, MODEL_ARTIFACT):
        shutil.copy2(artifact, target / artifact.name)
    return target


@pytest.fixture(scope="session")
def table() -> phase2_report.KillTable:
    """La tabla de §11.6 leida del documento real."""
    return phase2_report.load_kill_table()


@pytest.fixture()
def real_report(tmp_path_factory: pytest.TempPathFactory) -> phase2_report.Phase2Report:
    """El informe real (artefactos reales, `write=False`) para los criterios que lo leen."""
    root = tmp_path_factory.mktemp("phase2_real")
    target = root / "derived" / "reports"
    target.mkdir(parents=True)
    for artifact in (PIPELINE_ARTIFACT, MODEL_ARTIFACT):
        shutil.copy2(artifact, target / artifact.name)
    return phase2_report.analyse(store=Store(root), reports_dir=target, as_of=NOW, write=False)


# ─────────────────────────────────────────────────────────────────────────────
# A1 — API del modulo y CLI
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_module_api_and_cli(tmp_path: Path) -> None:
    import inspect

    signature = inspect.signature(phase2_report.analyse)
    assert set(signature.parameters) == {"store", "reports_dir", "as_of", "write"}
    assert signature.parameters["write"].default is True
    assert callable(phase2_report.main)

    target = tmp_path / "derived" / "reports"
    target.mkdir(parents=True)
    for artifact in (PIPELINE_ARTIFACT, MODEL_ARTIFACT):
        shutil.copy2(artifact, target / artifact.name)
    _run_module(reports_dir=target, data_root=tmp_path, as_of=NOW.isoformat(), seed=None)
    assert (target / "phase2_report_2026-09-24.json").is_file()
    assert (target / "phase2_report_2026-09-24.md").is_file()


# ─────────────────────────────────────────────────────────────────────────────
# A2 — ficheros, nada fuera de `--reports-dir`, `--as-of` obligatorio, segunda pasada
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_writes_only_report_pair_and_is_deterministic(tmp_path: Path, reports: Path) -> None:
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    report = phase2_report.analyse(
        store=Store(tmp_path), reports_dir=reports, as_of=NOW, write=True
    )
    assert report.report_stem == "phase2_report_2026-09-24"
    after = {p for p in tmp_path.rglob("*") if p.is_file()}
    assert after - before == {
        reports / "phase2_report_2026-09-24.json",
        reports / "phase2_report_2026-09-24.md",
    }
    json_bytes = (reports / "phase2_report_2026-09-24.json").read_bytes()
    md_bytes = (reports / "phase2_report_2026-09-24.md").read_bytes()
    phase2_report.analyse(store=Store(tmp_path), reports_dir=reports, as_of=NOW, write=True)
    assert (reports / "phase2_report_2026-09-24.json").read_bytes() == json_bytes
    assert (reports / "phase2_report_2026-09-24.md").read_bytes() == md_bytes


def test_a2_missing_as_of_exits_2_without_writing(tmp_path: Path, reports: Path) -> None:
    assert phase2_report.main(["--reports-dir", str(reports)]) == 2
    assert phase2_report.main(["--reports-dir", str(reports), "--as-of", "no-es-iso"]) == 2
    assert not list(reports.glob("phase2_report_*"))


def test_a2_write_false_writes_nothing(tmp_path: Path, reports: Path) -> None:
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    phase2_report.analyse(store=Store(tmp_path), reports_dir=reports, as_of=NOW, write=False)
    after = {p for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before


# ─────────────────────────────────────────────────────────────────────────────
# A3 — AST sin reloj, sin red, sin horas ET literales ni escrituras fuera del dir
# ─────────────────────────────────────────────────────────────────────────────
def _write_span(tree: ast.Module) -> tuple[int, int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Phase2Report":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "write":
                    return item.lineno, cast("int", item.end_lineno)
    raise AssertionError("no se encontro Phase2Report.write en el AST")


def test_a3_ast_forbids_clock_network_and_foreign_writes() -> None:
    tree = ast.parse(SOURCE)
    forbidden_attrs = {("datetime", "now"), ("date", "today"), ("time", "time")}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert (node.value.id, node.attr) not in forbidden_attrs
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            roots = {name.split(".")[0] for name in names}
            assert roots.isdisjoint({"time", "socket", "urllib", "httpx", "requests"})
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "open"
    start, end = _write_span(tree)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"write_text", "write_bytes"}
        ):
            assert start <= node.lineno <= end, "escritura fuera de Phase2Report.write (A3)"
    hours = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and re.search(r"\b\d{1,3}:\d{2}\b", node.value)
    ]
    assert hours == [], f"el modulo no puede contener horas literales: {hours} (A3)"


# ─────────────────────────────────────────────────────────────────────────────
# A4 — `report_sha256` canonico de #13, con prefijo, identico en procesos frescos
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_hash_format_and_golden(real_report: phase2_report.Phase2Report) -> None:
    """A4: el digest lleva el prefijo, cuadra con el payload y con el JSON publicado.

    #96: aqui se **fijaba** el literal `GOLDEN_REPORT_SHA256`. Ese digest es el del payload de
    este informe, y dentro viaja el `sha256` del artefacto de #28 (regenerable por cualquier
    tarea posterior: #92 lo regenero al publicar `hit_rate_per_trade`), asi que no puede ser un
    dorado. La corrida real sigue teniendo que publicar un digest con el formato declarado
    (prefijo `sha256:` + 64 hex), autoconsistente con su payload y sellado en el JSON; el
    determinismo entre procesos lo vigila `test_a4_fresh_processes_with_hashseeds`.
    """
    digest = real_report.report_sha256
    assert digest.startswith("sha256:")
    body = digest.removeprefix("sha256:")
    assert len(body) == 64
    assert all(character in "0123456789abcdef" for character in body)
    without_hash = dict(real_report.payload)
    recomputed = (
        "sha256:" + hashlib.sha256(canonical_text(without_hash).encode("utf-8")).hexdigest()
    )
    assert recomputed == digest
    # El payload publicado sella el hash **con** prefijo.
    published = json.loads(real_report.json_text())
    assert published["report_sha256"] == digest


def test_a4_fresh_processes_with_hashseeds(tmp_path: Path, reports: Path) -> None:
    first: bytes | None = None
    for seed in ("0", "1", "random"):
        _run_module(reports_dir=reports, data_root=tmp_path, as_of=NOW.isoformat(), seed=seed)
        payload = (reports / "phase2_report_2026-09-24.json").read_bytes()
        if first is None:
            first = payload
        assert payload == first


# ─────────────────────────────────────────────────────────────────────────────
# A5 — consumo en solo lectura de #28 y #26 con `select_artifact`
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_provenance_and_read_only(real_report: phase2_report.Phase2Report) -> None:
    provenance = cast("Mapping[str, object]", real_report.payload["provenance"])
    pipeline = cast("Mapping[str, object]", provenance["pipeline"])
    model = cast("Mapping[str, object]", provenance["model_comparison"])
    for block, artifact in ((pipeline, PIPELINE_ARTIFACT), (model, MODEL_ARTIFACT)):
        assert block["evidence"] == "artifact"
        assert block["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert isinstance(block["report_sha256"], str) and block["report_sha256"]
        assert block["generated_at"]
    assert real_report.pipeline.sha256 == hashlib.sha256(PIPELINE_ARTIFACT.read_bytes()).hexdigest()


def test_a5_zero_and_ambiguous_are_typed_errors(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(MissingInputArtifactError):
        phase2_report.load_input_artifact(
            empty, phase2_report.PIPELINE_CLASS, store=Store(tmp_path)
        )
    ambiguous = tmp_path / "ambiguous"
    ambiguous.mkdir()
    for name in ("pipeline_backtest_2026-09-23.json", "pipeline_backtest_2026-9-23.json"):
        shutil.copy2(PIPELINE_ARTIFACT, ambiguous / name)
    with pytest.raises(AmbiguousInputArtifactError):
        phase2_report.load_input_artifact(
            ambiguous, phase2_report.PIPELINE_CLASS, store=Store(tmp_path)
        )


def test_a5_missing_field_is_a_typed_error(tmp_path: Path, reports: Path) -> None:
    broken = json.loads(PIPELINE_ARTIFACT.read_text(encoding="utf-8"))
    broken.pop("scenario")
    (reports / PIPELINE_ARTIFACT.name).write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(MissingInputFieldError):
        phase2_report.analyse(store=Store(tmp_path), reports_dir=reports, as_of=NOW, write=False)


# ─────────────────────────────────────────────────────────────────────────────
# A6 — la tabla §11.6 se lee del documento; umbrales identicos entre corridas opuestas
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_table_is_read_from_the_document(table: phase2_report.KillTable) -> None:
    assert table.source["sha256"] == GOLDEN_PLAN_SHA256
    assert table.source["n_rows"] == 9
    assert table.source["section"] == "§11.6"
    assert table.source["path"] == "_docs/plan.md"
    rows = tuple((row.criterion, row.threshold, row.action) for row in table.rows)
    assert rows == GOLDEN_KILL_ROWS
    # Releyendo el documento, el hash se rehace y coincide (falla si una fila se desvia).
    reread = phase2_report.read_kill_table(
        (REPO_ROOT / "_docs" / "plan.md").read_text(encoding="utf-8")
    )
    assert reread.source["sha256"] == table.source["sha256"]
    assert (reread.rows[0].criterion, reread.rows[4].threshold) == (
        "⭐ **¿Existe edge demostrable?**",
        "Sí, significativamente",
    )


def test_a6_rows_are_the_nine_declared(table: phase2_report.KillTable) -> None:
    assert tuple(row.kind for row in table.rows) == phase2_report.ROW_KINDS
    altered = (
        (REPO_ROOT / "_docs" / "plan.md")
        .read_text(encoding="utf-8")
        .replace("| PBO | < 20% | Simplificar modelo |", "| PBO | < 15% | Simplificar modelo |")
    )
    mutated = phase2_report.read_kill_table(altered)
    assert mutated.source["sha256"] != table.source["sha256"]


def test_a6_opposite_synthetic_runs_share_identical_thresholds() -> None:
    all_pass = evaluate(
        synthetic_pipeline(
            net=_net_computable(
                hit=(0.60, 0.70),
                sharpe=(0.5, 1.5),
                beats={"no_trade": "pass", "liston_a": "pass", "liston_b": "pass"},
            )
        ),
        synthetic_model(pbo=0.01, dsr_half="pass"),
    )
    all_fail = evaluate(
        synthetic_pipeline(
            net=_net_computable(
                hit=(0.10, 0.20),
                sharpe=(-2.0, -1.0),
                beats={"no_trade": "fail", "liston_a": "fail", "liston_b": "fail"},
            ),
            drawdown=50.0,
        ),
        synthetic_model(pbo=0.9, dsr_half="fail"),
    )
    assert [row["threshold"] for row in all_pass] == [row["threshold"] for row in all_fail]
    assert [row["threshold"] for row in all_pass] == [
        threshold for _, threshold, _ in GOLDEN_KILL_ROWS
    ]


# ─────────────────────────────────────────────────────────────────────────────
# A7 — vocabulario de #9, `code`/`source_row` por fila y plegado de `aggregate_gate`
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_every_row_uses_the_vocabulary(real_report: phase2_report.Phase2Report) -> None:
    criteria = cast("list[Mapping[str, object]]", real_report.payload["criteria"])
    vocabulary = {member.value for member in HalfResult}
    for entry in criteria:
        assert entry["state"] in vocabulary
        assert isinstance(entry["code"], str) and entry["code"]
        assert entry["source_row"] == entry["criterion"]
        assert entry["threshold"] and entry["action"]


def test_a7_folding_matches_the_pairwise_rule(real_report: phase2_report.Phase2Report) -> None:
    gate = cast("Mapping[str, object]", real_report.payload["gate"])
    states = [
        str(entry["state"])
        for entry in cast("list[Mapping[str, object]]", real_report.payload["criteria"])
    ]
    folded = states[0]
    for state in states[1:]:
        folded = str(aggregate_gate(folded, state))
    assert gate["aggregate"] == folded
    for a in HalfResult:
        for b in HalfResult:
            expected = (
                GateVerdict.FAIL.value
                if str(HalfResult.FAIL) in {a.value, b.value}
                else GateVerdict.PASS.value
                if a.value == b.value == str(HalfResult.PASS)
                else GateVerdict.NOT_EVALUABLE.value
            )
            assert str(aggregate_gate(a, b)) == expected


# ─────────────────────────────────────────────────────────────────────────────
# A8 — `phase2_ready` y consistencia del veredicto
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_phase2_ready_and_verdict(real_report: phase2_report.Phase2Report) -> None:
    gate = cast("Mapping[str, object]", real_report.payload["gate"])
    assert real_report.payload["phase2_ready"] == (gate["aggregate"] == "pass")
    verdict = cast("Mapping[str, object]", real_report.payload["verdict"])
    assert verdict["state"] in {
        phase2_report.VERDICT_CONTINUE,
        phase2_report.VERDICT_SIMPLIFY,
        phase2_report.VERDICT_STOP,
        phase2_report.VERDICT_NOT_EVALUABLE,
    }
    assert verdict["state"] != phase2_report.VERDICT_CONTINUE
    assert verdict["consistent"] is True


def test_a8_inconsistent_verdict_raises() -> None:
    with pytest.raises(VerdictError):
        phase2_report.consistent_verdict(gate=GateVerdict.FAIL, verdict="continue")
    with pytest.raises(VerdictError):
        phase2_report.consistent_verdict(gate=GateVerdict.NOT_EVALUABLE, verdict="bogus")
    assert phase2_report.consistent_verdict(gate=GateVerdict.PASS, verdict="continue") == "continue"


def test_a8_verdicts_reachable_and_consistent() -> None:
    all_pass = evaluate(
        synthetic_pipeline(
            net=_net_computable(
                hit=(0.60, 0.70),
                sharpe=(0.5, 1.5),
                beats={"no_trade": "pass", "liston_a": "pass", "liston_b": "pass"},
            )
        ),
        synthetic_model(pbo=0.01, dsr_half="pass"),
    )
    gate_pass = phase2_report.gate_block(all_pass)
    assert gate_pass["aggregate"] == "not_evaluable"  # filas 8 y 9 siguen sin evaluar

    simplify_criteria = evaluate(
        synthetic_pipeline(
            net=_net_computable(
                hit=(0.60, 0.70),
                sharpe=(0.5, 1.5),
                beats={"no_trade": "pass", "liston_a": "pass", "liston_b": "pass"},
            )
        ),
        synthetic_model(pbo=0.9, dsr_half="pass"),
    )
    simplify = phase2_report.resolve_verdict(
        aggregate=str(phase2_report.gate_block(simplify_criteria)["aggregate"]),
        criteria=simplify_criteria,
    )
    assert simplify["state"] == phase2_report.VERDICT_SIMPLIFY

    stop_criteria = evaluate(
        synthetic_pipeline(
            net=_net_computable(
                hit=(0.60, 0.70),
                sharpe=(0.5, 1.5),
                beats={"no_trade": "pass", "liston_a": "pass", "liston_b": "pass"},
            )
        ),
        synthetic_model(pbo=0.01, dsr_half="fail"),
    )
    stop = phase2_report.resolve_verdict(
        aggregate=str(phase2_report.gate_block(stop_criteria)["aggregate"]),
        criteria=stop_criteria,
    )
    assert stop["state"] == phase2_report.VERDICT_STOP


# ─────────────────────────────────────────────────────────────────────────────
# A9 — criterio principal: `not_evaluable` con el artefacto real, nunca `pass`
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_main_row_not_evaluable_today(real_report: phase2_report.Phase2Report) -> None:
    criteria = cast("list[Mapping[str, object]]", real_report.payload["criteria"])
    main = criteria[0]
    assert main["kind"] == phase2_report.ROW_EDGE
    assert main["state"] == "not_evaluable"
    assert main["code"] == "net_metrics_not_computable"
    assert main["follow_ups"] == ["#62", "#60"]
    assert cast("Mapping[str, object]", real_report.payload["net_metrics"])["state"] == (
        "not_computable"
    )


def test_a9_never_turns_not_evaluable_into_pass() -> None:
    criteria = evaluate(
        synthetic_pipeline(net=_net_not_computable()),
        synthetic_model(pbo=0.01, dsr_half="pass"),
    )
    main = _row_of(criteria, phase2_report.ROW_EDGE)
    assert main["state"] == "not_evaluable"
    assert main["state"] != HalfResult.PASS


def test_a9_significant_and_against_branches() -> None:
    significant = evaluate(
        synthetic_pipeline(
            net=_net_computable(
                hit=(0.60, 0.70),
                sharpe=(0.5, 1.5),
                beats={"no_trade": "pass", "liston_a": "pass", "liston_b": "pass"},
            )
        ),
        synthetic_model(),
    )
    assert _row_of(significant, phase2_report.ROW_EDGE)["state"] == "pass"
    against = evaluate(
        synthetic_pipeline(
            net=_net_computable(
                hit=(0.10, 0.20),
                sharpe=(-2.0, -1.0),
                beats={"no_trade": "fail", "liston_a": "fail", "liston_b": "fail"},
            )
        ),
        synthetic_model(),
    )
    assert _row_of(against, phase2_report.ROW_EDGE)["state"] == "fail"
    assert _row_of(against, phase2_report.ROW_EDGE)["code"] == "edge_against"


# ─────────────────────────────────────────────────────────────────────────────
# A10 — `p*` derivado, no cableado
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_p_star_is_derived(real_report: phase2_report.Phase2Report) -> None:
    block = cast("Mapping[str, object]", real_report.payload["p_star"])
    scenarios = {
        cast("str", entry["r_pct"]): cast("str", entry["p_star_pct"])
        for entry in cast("list[Mapping[str, object]]", block["scenarios"])
    }
    assert scenarios["1.0"] == "50.21"
    assert scenarios["0.5"] == "50.42"
    assert scenarios["1.5"] == "50.14"
    assert block["cost_round_trip_pct"] == "0.0042"
    assert "derivation" in block
    assert block["cost_source"]
    assert phase0_report.declared_value("r_scenarios_pct") == "0.5 / 1.0 / 1.5"


def test_a10_no_hardcoded_threshold() -> None:
    assert "50.2" not in SOURCE
    assert "50.21" not in SOURCE
    assert "0.0042" not in SOURCE


# ─────────────────────────────────────────────────────────────────────────────
# A11 — liston B insuficiente por si solo; batir A y B no aprueba la fila principal
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_liston_b_insufficient_on_its_own(real_report: phase2_report.Phase2Report) -> None:
    criteria = cast("list[Mapping[str, object]]", real_report.payload["criteria"])
    liston_b = _row_of(criteria, phase2_report.ROW_LISTON_B)
    assert liston_b["insufficient_on_its_own"] is True
    assert "no demuestra nada" in cast("str", liston_b["note"])


def test_a11_beating_a_and_b_does_not_pass_the_main_row() -> None:
    criteria = evaluate(
        synthetic_pipeline(
            net=_net_computable(
                hit=(0.49, 0.51),
                sharpe=(-0.2, 0.2),
                beats={"no_trade": "pass", "liston_a": "pass", "liston_b": "pass"},
            )
        ),
        synthetic_model(),
    )
    assert _row_of(criteria, phase2_report.ROW_NO_TRADE)["state"] == "pass"
    assert _row_of(criteria, phase2_report.ROW_LISTON_A)["state"] == "pass"
    assert _row_of(criteria, phase2_report.ROW_LISTON_B)["state"] == "pass"
    assert _row_of(criteria, phase2_report.ROW_EDGE)["state"] == "not_evaluable"


# ─────────────────────────────────────────────────────────────────────────────
# A12 — los tres listones se distinguen por `kind`; C nunca es baseline
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_listones_and_baselines(real_report: phase2_report.Phase2Report) -> None:
    block = cast("Mapping[str, object]", real_report.payload["listones"])
    assert block["evidence"] == "artifact"
    assert block["is_validation"] is False
    kinds = cast("list[str]", block["kinds"])
    assert kinds.count("baseline") == 6
    assert set(kinds) == {"baseline", "liston_a", "liston_b", "liston_c"}
    listones = cast("list[Mapping[str, object]]", block["listones"])
    by_kind = {entry["kind"]: entry for entry in listones}
    assert by_kind["liston_c"]["is_invertible"] is False
    assert block["liston_c_is_invertible"] is False
    baselines = cast("list[Mapping[str, object]]", block["baselines"])
    assert all(entry["kind"] == "baseline" for entry in baselines)
    assert len(baselines) == block["n_baselines"] == 6


def test_a12_table_is_copied_and_reflects_the_artifact(tmp_path: Path) -> None:
    pipeline = synthetic_pipeline()
    table = cast("dict[str, object]", pipeline["table"])
    rows = cast("list[dict[str, object]]", table["rows"])
    rows.append({"row": "extra", "kind": "baseline", "is_invertible": True})
    report = _analyse_synthetic(tmp_path, pipeline, synthetic_model())
    block = cast("Mapping[str, object]", report.payload["listones"])
    assert block["n_baselines"] == 7


# ─────────────────────────────────────────────────────────────────────────────
# A13 — PBO y DSR se importan/copian; el AST no importa el calculo de sobreajuste
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_pbo_and_dsr_are_copied(real_report: phase2_report.Phase2Report) -> None:
    # #90: el PBO de #26 se **copia** del `model_comparison` regenerado. Al corregir el motor
    # (#80) el PBO pasa de `not_detected` (0.0476) a `detected` (0.6627) y su mitad de la puerta
    # de `pass` a `fail`; el `dsr` sigue `fail`/`not_significant` y la agregada sigue `fail`.
    # El `sr_variance` del DSR lo da el registro (`registry.sr_variance`), que aun mezcla el
    # `sharpe_per_session` pre-#80 de `runs/408fead` (fuera de alcance -> #97).
    criteria = cast("list[Mapping[str, object]]", real_report.payload["criteria"])
    pbo = _row_of(criteria, phase2_report.ROW_PBO)
    assert pbo["state"] == "fail"
    detail = cast("Mapping[str, object]", pbo["detail"])
    assert detail["blocks"] == 10
    assert detail["n_observations"] == 500
    assert detail["declared_verdict"] == "detected"
    assert detail["pbo_max"] == 0.2
    dsr = _row_of(criteria, phase2_report.ROW_DSR)
    assert dsr["state"] == "fail"
    assert cast("Mapping[str, object]", dsr["detail"])["declared_half"] == "fail"


def test_a13_ast_does_not_import_overfitting_calculations() -> None:
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "cfdtrader.backtest.overfitting":
            assert [alias.name for alias in node.names] == ["PBO_MAX"]
    assert A13_DIVERGENCE


def test_a13_pbo_threshold_follows_pbo_max() -> None:
    pass_row = evaluate(synthetic_pipeline(), synthetic_model(pbo=0.19))
    fail_row = evaluate(synthetic_pipeline(), synthetic_model(pbo=0.21))
    assert _row_of(pass_row, phase2_report.ROW_PBO)["state"] == "pass"
    assert _row_of(fail_row, phase2_report.ROW_PBO)["state"] == "fail"


# ─────────────────────────────────────────────────────────────────────────────
# A14 — drawdown desde #28 y umbral leido del literal
# ─────────────────────────────────────────────────────────────────────────────
def test_a14_drawdown_uses_artifact_metric(real_report: phase2_report.Phase2Report) -> None:
    criteria = cast("list[Mapping[str, object]]", real_report.payload["criteria"])
    drawdown = _row_of(criteria, phase2_report.ROW_DRAWDOWN)
    assert drawdown["state"] == "pass"
    detail = cast("Mapping[str, object]", drawdown["detail"])
    assert detail["limit_pct"] == "20"
    assert detail["arm"] == "coste_declarado"
    assert detail["basis"] == "declared_cost"


def test_a14_above_limit_fails_with_the_action() -> None:
    criteria = evaluate(synthetic_pipeline(drawdown=25.0), synthetic_model())
    drawdown = _row_of(criteria, phase2_report.ROW_DRAWDOWN)
    assert drawdown["state"] == "fail"
    assert drawdown["code"] == "drawdown_above_limit"
    assert drawdown["action"] == "Reducir tamaño o parar"


def test_a14_none_is_not_zero() -> None:
    criteria = evaluate(synthetic_pipeline(drawdown=None), synthetic_model())
    drawdown = _row_of(criteria, phase2_report.ROW_DRAWDOWN)
    assert drawdown["state"] == "not_evaluable"
    assert drawdown["code"] == "drawdown_not_computable"
    detail = cast("Mapping[str, object]", drawdown["detail"])
    metric = cast("Mapping[str, object]", detail["max_drawdown_pct"])
    assert metric["estimate"] is None


# ─────────────────────────────────────────────────────────────────────────────
# A15 — «bate a no operar» y «bate a liston A» exigen la base neta
# ─────────────────────────────────────────────────────────────────────────────
def test_a15_net_rows_not_evaluable_today(real_report: phase2_report.Phase2Report) -> None:
    criteria = cast("list[Mapping[str, object]]", real_report.payload["criteria"])
    for kind in (phase2_report.ROW_NO_TRADE, phase2_report.ROW_LISTON_A):
        row = _row_of(criteria, kind)
        assert row["state"] == "not_evaluable"
        assert row["code"] == "net_metrics_not_computable"


def test_a15_declared_cost_is_labelled(real_report: phase2_report.Phase2Report) -> None:
    block = cast("Mapping[str, object]", real_report.payload["declared_cost"])
    assert block["basis"] == "declared_cost"
    assert block["is_validation"] is False
    assert "nunca" in cast("str", block["note"])


# ─────────────────────────────────────────────────────────────────────────────
# A16 — las dos filas operativas se publican; `n_rows == 9` siempre
# ─────────────────────────────────────────────────────────────────────────────
def test_a16_operational_rows_are_published(real_report: phase2_report.Phase2Report) -> None:
    criteria = cast("list[Mapping[str, object]]", real_report.payload["criteria"])
    assert len(criteria) == 9
    paper = _row_of(criteria, phase2_report.ROW_PAPER)
    assert paper["state"] == "not_evaluable"
    assert paper["follow_ups"] == ["#45"]
    close = _row_of(criteria, phase2_report.ROW_CLOSE)
    assert close["state"] == "not_evaluable"
    assert close["follow_ups"] == ["#84"]
    assert cast("Mapping[str, object]", real_report.payload["criteria_source"])["n_rows"] == 9
    assert cast("Mapping[str, object]", real_report.payload["gate"])["n_rows"] == 9
    assert len(cast("list[object]", real_report.payload["kill_criteria"])) == 9


# ─────────────────────────────────────────────────────────────────────────────
# A17 — limites declarados y `null != 0`
# ─────────────────────────────────────────────────────────────────────────────
def test_a17_limitations_and_does_not_do(real_report: phase2_report.Phase2Report) -> None:
    limitations = " ".join(cast("list[str]", real_report.payload["limitations"]))
    for token in ("#62", "#60", "#70", "#68"):
        assert token in limitations
    does_not_do = cast("list[Mapping[str, object]]", real_report.payload["does_not_do"])
    issues = {entry["issue"] for entry in does_not_do}
    assert {"#62", "#60", "#70", "#16", "#68", "#45"} <= issues
    follow_ups = {
        entry["issue"]
        for entry in cast("list[Mapping[str, object]]", real_report.payload["follow_ups"])
    }
    assert {"#62", "#60", "#70", "#68", "#67", "#45", "#84", "#88"} <= follow_ups


def test_a17_no_none_written_as_zero(real_report: phase2_report.Phase2Report) -> None:
    criteria = cast("list[Mapping[str, object]]", real_report.payload["criteria"])
    assert all(
        entry["state"] != "pass"
        for entry in criteria
        if str(entry["code"]).endswith("not_computable")
    )


# ─────────────────────────────────────────────────────────────────────────────
# A18 — segunda pasada y cobertura del modulo nuevo (los numeros se miden por comando)
# ─────────────────────────────────────────────────────────────────────────────
def test_a18_second_pass_is_byte_identical(tmp_path: Path, reports: Path) -> None:
    first = phase2_report.analyse(store=Store(tmp_path), reports_dir=reports, as_of=NOW, write=True)
    first_json = (reports / f"{first.report_stem}.json").read_bytes()
    first_md = (reports / f"{first.report_stem}.md").read_bytes()
    second = phase2_report.analyse(
        store=Store(tmp_path), reports_dir=reports, as_of=NOW, write=True
    )
    assert second.report_sha256 == first.report_sha256
    assert (reports / f"{second.report_stem}.json").read_bytes() == first_json
    assert (reports / f"{second.report_stem}.md").read_bytes() == first_md


# ─────────────────────────────────────────────────────────────────────────────
# A19 — sin red: se genera leyendo solo ficheros locales
# ─────────────────────────────────────────────────────────────────────────────
def test_a19_no_network_is_used(
    tmp_path: Path, reports: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket

    def _blocked(*args: object, **kwargs: object) -> object:
        raise AssertionError("el informe no puede abrir un socket (A19)")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    report = phase2_report.analyse(
        store=Store(tmp_path), reports_dir=reports, as_of=NOW, write=True
    )
    assert (reports / f"{report.report_stem}.json").is_file()


# ─────────────────────────────────────────────────────────────────────────────
# A20 — solo se tocan los ficheros declarados
# ─────────────────────────────────────────────────────────────────────────────
def test_a20_only_named_files_change() -> None:
    git = shutil.which("git")
    if git is None:  # pragma: no cover - entorno sin git
        pytest.skip("git no disponible")
    result = subprocess.run(  # noqa: S603 - git local del repositorio
        [git, "-C", str(REPO_ROOT), "diff", "--name-only", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    changed = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    assert changed.isdisjoint(FORBIDDEN_PATHS), f"ficheros congelados tocados: {changed}"


# ─────────────────────────────────────────────────────────────────────────────
# A21 — sin dependencias nuevas
# ─────────────────────────────────────────────────────────────────────────────
def test_a21_no_new_dependency() -> None:
    tree = ast.parse(SOURCE)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "sys",
        "collections",
        "dataclasses",
        "datetime",
        "decimal",
        "functools",
        "pathlib",
        "typing",
        "loguru",
        "cfdtrader",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bordes adicionales: tabla invalida y ambiguedad de la tabla
# ─────────────────────────────────────────────────────────────────────────────
def test_kill_table_missing_section_raises() -> None:
    with pytest.raises(InvalidKillTableError):
        phase2_report.read_kill_table("## sin tabla\n")


def test_kill_table_wrong_row_count_raises() -> None:
    text = (REPO_ROOT / "_docs" / "plan.md").read_text(encoding="utf-8")
    without_last = text.replace(
        "| **Incumplimiento del cierre a las 16:00 ET** (22:00 Madrid) | Cero tolerancia | "
        "Revisar el mecanismo antes de seguir |\n",
        "",
    )
    with pytest.raises(InvalidKillTableError):
        phase2_report.read_kill_table(without_last)


# ─────────────────────────────────────────────────────────────────────────────
# Bordes de cobertura: todo a traves de la API publica del modulo
# ─────────────────────────────────────────────────────────────────────────────
_KILL_HEADER: Final[str] = "| Criterio | Umbral | Acción si falla |"


def _table_with_drawdown(threshold: str) -> phase2_report.KillTable:
    """Copia la tabla real cambiando el umbral de la fila de drawdown (clases publicas)."""
    base = phase2_report.load_kill_table()
    rows = tuple(
        phase2_report.KillRow(
            row_index=row.row_index,
            kind=row.kind,
            criterion=row.criterion,
            threshold=threshold if row.kind == phase2_report.ROW_DRAWDOWN else row.threshold,
            action=row.action,
        )
        for row in base.rows
    )
    return phase2_report.KillTable(source=base.source, rows=rows)


def _write_synthetic(
    tmp_path: Path, pipeline: Mapping[str, object], model: Mapping[str, object]
) -> tuple[Path, Path]:
    """Escribe un par de artefactos sinteticos de #28/#26 y devuelve (reports, raiz)."""
    reports = tmp_path / "derived" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / PIPELINE_ARTIFACT.name).write_text(json.dumps(pipeline), encoding="utf-8")
    (reports / MODEL_ARTIFACT.name).write_text(json.dumps(model), encoding="utf-8")
    return reports, tmp_path


def _analyse_synthetic(
    tmp_path: Path,
    pipeline: Mapping[str, object],
    model: Mapping[str, object],
    *,
    as_of: datetime = NOW,
    store_root: Path | None = None,
) -> phase2_report.Phase2Report:
    """Corre `analyse` sobre artefactos sinteticos, sin re-ejecutar #28 ni #26."""
    reports, root = _write_synthetic(tmp_path, pipeline, model)
    return phase2_report.analyse(
        store=Store(store_root if store_root is not None else root),
        reports_dir=reports,
        as_of=as_of,
        write=False,
    )


def test_edge_unknown_row_raises() -> None:
    real = (REPO_ROOT / "_docs" / "plan.md").read_text(encoding="utf-8")
    with pytest.raises(InvalidKillTableError):
        phase2_report.read_kill_table(real.replace("| PBO |", "| Criterio raro |"))


def test_edge_kill_table_missing_row_and_parsing() -> None:
    table = phase2_report.load_kill_table()
    with pytest.raises(InvalidKillTableError):
        table.row("no-existe")
    with pytest.raises(InvalidKillTableError):
        phase2_report.read_kill_table("### 11.6 x\n\nnada\n")
    with pytest.raises(InvalidKillTableError):
        phase2_report.read_kill_table(f"### 11.6 x\n\n{_KILL_HEADER}\n")
    with pytest.raises(InvalidKillTableError):
        phase2_report.read_kill_table(f"### 11.6 x\n\n{_KILL_HEADER}\n| xx |\n")
    with pytest.raises(InvalidKillTableError):
        phase2_report.read_kill_table(f"### 11.6 x\n\n{_KILL_HEADER}\n|---|---|---|\n| a | b |\n")
    with pytest.raises(InvalidKillTableError):
        phase2_report.read_kill_table(
            f"### 11.6 x\n\n{_KILL_HEADER}\n|---|---|---|\n| PBO | < 20% | Simplificar |\n"
        )
    real = (REPO_ROOT / "_docs" / "plan.md").read_text(encoding="utf-8")
    with pytest.raises(InvalidKillTableError):
        phase2_report.read_kill_table(real.replace("| Drawdown máximo |", "| PBO |"))


def test_edge_load_kill_table_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase2_report, "plan_path", lambda: tmp_path / "no-existe.md")
    with pytest.raises(InvalidKillTableError):
        phase2_report.load_kill_table()


def test_edge_typed_errors_from_artifacts(tmp_path: Path) -> None:
    non_string_cost = synthetic_pipeline()
    non_string_cost["scenario"] = {"cost_basis_pct": 0.0042, "cost_provenance": "s"}
    with pytest.raises(MissingInputFieldError):
        _analyse_synthetic(tmp_path, non_string_cost, synthetic_model())
    non_mapping_arms = synthetic_pipeline()
    non_mapping_arms["arms"] = 5
    with pytest.raises(MissingInputFieldError):
        _analyse_synthetic(tmp_path, non_mapping_arms, synthetic_model())
    non_number_traded = synthetic_pipeline()
    arm = cast(
        "dict[str, object]",
        cast("dict[str, object]", non_number_traded["arms"])["coste_declarado"],
    )
    arm["traded"] = "x"
    with pytest.raises(MissingInputFieldError):
        _analyse_synthetic(tmp_path, non_number_traded, synthetic_model())


def test_edge_optional_fields(tmp_path: Path) -> None:
    no_dsr = synthetic_model()
    cast("dict[str, object]", no_dsr["deflated_sharpe_ratio"]).pop("dsr")
    report = _analyse_synthetic(tmp_path, synthetic_pipeline(), no_dsr)
    criteria = cast("list[Mapping[str, object]]", report.payload["criteria"])
    assert _row_of(criteria, phase2_report.ROW_DSR)["state"] == "pass"
    null_dsr = synthetic_model()
    cast("dict[str, object]", null_dsr["deflated_sharpe_ratio"])["dsr"] = None
    null_report = _analyse_synthetic(tmp_path, synthetic_pipeline(), null_dsr)
    null_criteria = cast("list[Mapping[str, object]]", null_report.payload["criteria"])
    assert _row_of(null_criteria, phase2_report.ROW_DSR)["state"] == "pass"
    bad_dsr = synthetic_model()
    cast("dict[str, object]", bad_dsr["deflated_sharpe_ratio"])["dsr"] = "x"
    with pytest.raises(MissingInputFieldError):
        _analyse_synthetic(tmp_path, synthetic_pipeline(), bad_dsr)
    scalar_dsr = synthetic_model()
    scalar_dsr["deflated_sharpe_ratio"] = 5
    scalar_report = _analyse_synthetic(tmp_path, synthetic_pipeline(), scalar_dsr)
    scalar_criteria = cast("list[Mapping[str, object]]", scalar_report.payload["criteria"])
    assert _row_of(scalar_criteria, phase2_report.ROW_DSR)["state"] == "pass"


def test_edge_naive_as_of_and_foreign_store(tmp_path: Path) -> None:
    naive = _analyse_synthetic(
        tmp_path, synthetic_pipeline(), synthetic_model(), as_of=datetime(2026, 9, 24, 12, 0)
    )
    assert naive.report_date == "2026-09-24"
    foreign = _analyse_synthetic(
        tmp_path, synthetic_pipeline(), synthetic_model(), store_root=tmp_path / "otro-almacen"
    )
    provenance = cast("Mapping[str, object]", foreign.payload["provenance"])
    pipeline = cast("Mapping[str, object]", provenance["pipeline"])
    assert str(pipeline["path"]).startswith("/")


def test_edge_absent_net_metrics(tmp_path: Path) -> None:
    without = synthetic_pipeline()
    del without["net_metrics"]
    report = _analyse_synthetic(tmp_path, without, synthetic_model())
    assert cast("Mapping[str, object]", report.payload["net_metrics"])["state"] == "absent"
    criteria = cast("list[Mapping[str, object]]", report.payload["criteria"])
    assert _row_of(criteria, phase2_report.ROW_EDGE)["code"] == "net_metrics_not_computable"


def test_edge_drawdown_limit_parsing() -> None:
    pipeline = synthetic_pipeline(drawdown=25.0)
    stars = _stars(pipeline)
    criteria = phase2_report.evaluate_criteria(
        pipeline=pipeline,
        model=synthetic_model(),
        table=_table_with_drawdown("20.5 %"),
        stars=stars,
    )
    assert _row_of(criteria, phase2_report.ROW_DRAWDOWN)["state"] == "fail"
    with pytest.raises(InvalidKillTableError):
        phase2_report.evaluate_criteria(
            pipeline=pipeline,
            model=synthetic_model(),
            table=_table_with_drawdown("Cero tolerancia"),
            stars=stars,
        )


def test_edge_pbo_and_dsr_not_evaluable() -> None:
    pbo_model = synthetic_model()
    cast("dict[str, object]", pbo_model["probability_of_backtest_overfitting"])["state"] = (
        "not_evaluated"
    )
    assert (
        _row_of(evaluate(synthetic_pipeline(), pbo_model), phase2_report.ROW_PBO)["state"]
        == "not_evaluable"
    )
    dsr_model = synthetic_model()
    gate = cast("dict[str, object]", cast("dict[str, object]", dsr_model["verdict"])["gate"])
    gate["halves"] = {
        "deflated_sharpe_ratio": "not_evaluable",
        "probability_of_backtest_overfitting": "pass",
    }
    assert (
        _row_of(evaluate(synthetic_pipeline(), dsr_model), phase2_report.ROW_DSR)["state"]
        == "not_evaluable"
    )


def test_edge_net_basis_unknown_half() -> None:
    net = _net_computable(
        hit=(0.6, 0.7),
        sharpe=(0.5, 1.5),
        beats={"no_trade": "not_evaluable", "liston_a": "pass", "liston_b": "pass"},
    )
    criteria = evaluate(synthetic_pipeline(net=net), synthetic_model())
    assert _row_of(criteria, phase2_report.ROW_NO_TRADE)["state"] == "not_evaluable"


def test_edge_gate_block_empty_and_verdicts() -> None:
    with pytest.raises(VerdictError):
        phase2_report.gate_block([])
    all_pass = evaluate(
        synthetic_pipeline(
            net=_net_computable(
                hit=(0.6, 0.7),
                sharpe=(0.5, 1.5),
                beats={"no_trade": "pass", "liston_a": "pass", "liston_b": "pass"},
            )
        ),
        synthetic_model(),
    )
    gate = phase2_report.gate_block(all_pass)
    assert (
        phase2_report.resolve_verdict(aggregate="pass", criteria=all_pass)["state"]
        == phase2_report.VERDICT_CONTINUE
    )
    assert (
        phase2_report.resolve_verdict(aggregate=str(gate["aggregate"]), criteria=all_pass)["state"]
        == phase2_report.VERDICT_NOT_EVALUABLE
    )


def test_edge_main_writes_and_handles_errors(tmp_path: Path, reports: Path) -> None:
    assert phase2_report.main(["--reports-dir", str(reports), "--as-of", NOW.isoformat()]) == 0
    empty = tmp_path / "vacio"
    empty.mkdir()
    assert phase2_report.main(["--reports-dir", str(empty), "--as-of", NOW.isoformat()]) == 2


def test_edge_malformed_artifact(tmp_path: Path, reports: Path) -> None:
    (reports / PIPELINE_ARTIFACT.name).write_text("no es json", encoding="utf-8")
    with pytest.raises(phase2_report.MalformedInputArtifactError):
        phase2_report.load_input_artifact(
            reports, phase2_report.PIPELINE_CLASS, store=Store(tmp_path)
        )
    (reports / PIPELINE_ARTIFACT.name).write_text("[]", encoding="utf-8")
    with pytest.raises(phase2_report.MalformedInputArtifactError):
        phase2_report.load_input_artifact(
            reports, phase2_report.PIPELINE_CLASS, store=Store(tmp_path)
        )
