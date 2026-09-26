"""Tests del barrido de sensibilidad del gate (#86): A1 a A15.

La corrida real es **barata** comparada con #28 (~20 s: las entradas compartidas se calculan una
sola vez y cada celda cuesta ~1 s de gate + motor + un unico intervalo bootstrap), asi que el
informe completo se construye **una vez** por sesion (``report``) y todas las comprobaciones se
apoyan en el. Los tres procesos CLI frescos de A10 se lanzan **en paralelo** y escriben bajo
``tmp_path``, con ``TMPDIR`` y ``UV_CACHE_DIR`` redirigidos ahi: la sesion de tests no toca el
``data/`` ni el ``runs/`` del repositorio (lo comprueban las fixtures de ``conftest.py``).

Ningun test fija el digest de un artefacto regenerable: los unicos literales estables son la
rejilla declarada, los encabezados de la tabla y los recuentos medidos que los criterios 2 y 13
exigen. Los digests se comprueban por **formato**, por **autoconsistencia** recomputada desde el
``.json`` en disco y por **determinismo** entre procesos.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final, cast

import pytest

from cfdtrader.analysis import gate_sweep, pipeline_report
from cfdtrader.analysis.gate_sweep import (
    GRID_CELLS,
    HASH_PREFIX,
    INERT_PARAMETERS,
    INVARIANCE_CONTROLS,
    REFERENCE_CELL,
    REPORT_PREFIX,
    GridCell,
    analyse,
    main,
    render_markdown,
)
from cfdtrader.analysis.pipeline_report import (
    BASIS_DECLARED_COST,
    METRIC_NAMES,
    scenario_parameters,
)
from cfdtrader.backtest.engine import (
    STATUS_TRADED,
    BacktestRun,
    FoldOutcome,
    SessionOutcome,
    canonical_text,
)
from cfdtrader.backtest.metrics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
)
from cfdtrader.data.store import Store

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"
MODULE_PATH: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "analysis" / "gate_sweep.py"
TEST_PATH: Final[Path] = Path(__file__).resolve()
PIPELINE_ARTIFACT: Final[Path] = (
    REAL_DATA / "derived" / "reports" / "pipeline_backtest_2026-09-23.json"
)

#: Instante **declarado** de todas las corridas: el modulo nunca lee el reloj.
NOW: Final[datetime] = datetime(2026, 9, 23, 22, 0, tzinfo=UTC)
STEM: Final[str] = f"{REPORT_PREFIX}_{NOW.date().isoformat()}"

#: Commit de partida de la entrega: A15 compara contra el.
BASE_COMMIT: Final[str] = "6aa582d"

#: Los dos ficheros que la entrega debe traer en el diff, y los cuatro congelados (A15).
WRITTEN: Final[frozenset[str]] = frozenset(
    {
        "src/cfdtrader/analysis/gate_sweep.py",
        "tests/test_gate_sweep.py",
    }
)
FROZEN: Final[tuple[str, ...]] = (
    "src/cfdtrader/backtest/baselines.py",
    "src/cfdtrader/backtest/metrics.py",
    "src/cfdtrader/models/baseline.py",
    "src/cfdtrader/analysis/feature_frame.py",
)

#: Los helpers de #28 que la entrega **no redefine**: los que reutiliza, los importa.
REUSE_NAMES: Final[tuple[str, ...]] = (
    "_series_of_run",
    "_declared_return_pct",
    "_gate_outputs",
    "_with_context",
    "_deciders",
    "scenario_parameters",
    "ArmLedger",
)

#: Los helpers que la entrega importa y **usa**: su identidad de objeto se comprueba.
REUSED_HELPERS: Final[tuple[str, ...]] = (
    "_series_of_run",
    "_gate_outputs",
    "_with_context",
    "_deciders",
    "scenario_parameters",
    "ArmLedger",
)

#: Los once campos declarados del gate que cada celda publica, en el orden de la tabla.
GATE_FIELDS: Final[tuple[str, ...]] = (
    "broker",
    "risk_per_trade_pct",
    "ev_threshold_pct",
    "max_daily_loss_pct",
    "max_weekly_loss_pct",
    "max_monthly_loss_pct",
    "r_pct",
    "tier_a_cost_multiple",
    "tier_b_cost_multiple",
    "tier_a_min_probability",
    "authorized_tiers",
)

SOURCE: Final[str] = MODULE_PATH.read_text(encoding="utf-8")
TREE: Final[ast.Module] = ast.parse(SOURCE)

#: La rejilla declarada de la issue, **verbatim**: (cell_id, pmin, tiers, mult_a, mult_b, ev, risk).
DECLARED_GRID: Final[tuple[tuple[str, str, list[str], str, str, str, str], ...]] = (
    ("s1", "0.58", ["A"], "3", "2", "2*c", "1"),
    ("pmin_050", "0.50", ["A"], "3", "2", "2*c", "1"),
    ("pmin_055", "0.55", ["A"], "3", "2", "2*c", "1"),
    ("pmin_062", "0.62", ["A"], "3", "2", "2*c", "1"),
    ("pmin_070", "0.70", ["A"], "3", "2", "2*c", "1"),
    ("tiers_ab", "0.58", ["A", "B"], "3", "2", "2*c", "1"),
    ("tiers_ab_multb100", "0.58", ["A", "B"], "3", "100", "2*c", "1"),
    ("mult_a_100", "0.58", ["A"], "100", "2", "2*c", "1"),
    ("inv_risk2", "0.58", ["A"], "3", "2", "2*c", "2"),
    ("inv_thr32", "0.58", ["A"], "3", "2", "32*c", "1"),
)

#: Los ocho campos inertes declarados (criterio 12).
INERT_FIELDS: Final[tuple[str, ...]] = (
    "broker",
    "ev_threshold_pct",
    "max_daily_loss_pct",
    "max_weekly_loss_pct",
    "max_monthly_loss_pct",
    "r_pct",
    "risk_per_trade_pct",
    "tier_b_cost_multiple",
)

needs_store = pytest.mark.skipif(
    not (REAL_DATA / "derived" / "labels").exists(),
    reason="el almacen real no esta en el arbol: los numeros de A4-A13 son los suyos",
)

#: Palabras que no pueden aparecer en el artefacto: ninguna cifra de base medida (criterio 6).
FORBIDDEN_TOKENS: Final[tuple[str, ...]] = ("net", "neto", "neta", "pnl_net")

#: Fuentes de tiempo que el modulo no puede consultar (criterio 3).
FORBIDDEN_CLOCK_ATTRIBUTES: Final[tuple[str, ...]] = ("now", "utcnow", "today")

#: El unico fichero que la entrega podria anadir, ademas de los dos nuevos (criterio 15).
EXTRA_ALLOWED: Final[tuple[str, ...]] = ("src/cfdtrader/analysis/__init__.py",)


# ─────────────────────────────────────────────────────────────────────────────
# Acceso tipado al payload publicado
# ─────────────────────────────────────────────────────────────────────────────
def at(node: object, *keys: str) -> object:
    """Un valor anidado del payload, sin ``Any`` por el camino."""
    current: object = node
    for key in keys:
        current = cast("Mapping[str, object]", current)[key]
    return current


def as_map(node: object) -> dict[str, object]:
    """El nodo como mapping."""
    return cast("dict[str, object]", node)


def as_list(node: object) -> list[object]:
    """El nodo como lista."""
    return cast("list[object]", node)


def as_str(node: object) -> str:
    """El nodo como texto."""
    return cast("str", node)


def as_int(node: object) -> int:
    """El nodo como entero."""
    return cast("int", node)


def as_floats(node: object) -> list[float]:
    """El nodo como lista de numeros."""
    return [float(cast("float", item)) for item in as_list(node)]


def as_objects(node: object) -> list[dict[str, object]]:
    """Una lista de mappings del payload."""
    return [as_map(item) for item in as_list(node)]


def cells_by_id(payload: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Las celdas publicadas, indexadas por ``cell_id``."""
    return {as_str(cell["cell_id"]): cell for cell in as_objects(payload["cells"])}


def series_sha_of(cell: Mapping[str, object]) -> str:
    """El ``series_sha256`` declarado de una celda."""
    return as_str(at(cell, "declared_series", "series_sha256"))


def all_keys(node: object) -> list[str]:
    """Todas las claves de un payload JSON anidado, recursivamente."""
    found: list[str] = []
    if isinstance(node, Mapping):
        for key, value in cast("Mapping[str, object]", node).items():
            found.append(str(key))
            found.extend(all_keys(value))
    elif isinstance(node, list):
        for item in cast("list[object]", node):
            found.extend(all_keys(item))
    return found


def all_values(node: object) -> list[object]:
    """Todos los valores de un payload JSON anidado."""
    found: list[object] = []
    if isinstance(node, Mapping):
        for value in cast("Mapping[str, object]", node).values():
            found.append(value)
            found.extend(all_values(value))
    elif isinstance(node, list):
        for item in cast("list[object]", node):
            found.extend(all_values(item))
    return found


def _sha256_text(text: str) -> str:
    """El sha256 de un texto, con el prefijo declarado."""
    return f"{HASH_PREFIX}{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _git(*arguments: str) -> str:
    """Salida de ``git``, sin paginacion."""
    completed = subprocess.run(  # noqa: S603 - el git del sistema, comando fijo
        ["git", "--no-pager", *arguments],  # noqa: S607 - el git del sistema
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


@dataclass(frozen=True, slots=True)
class CliRun:
    """El resultado de una corrida de la CLI en un proceso fresco."""

    directory: Path
    code: int
    seconds: float
    stderr: str
    report_sha256: str
    json_sha256: str
    markdown_sha256: str


def _run_cli(directory: Path, hash_seed: str) -> CliRun:
    """Corre la CLI en un proceso fresco, con su ``PYTHONHASHSEED`` y su dir."""
    environment = {
        **os.environ,
        "PYTHONHASHSEED": hash_seed,
        "TMPDIR": str(directory),
        "UV_CACHE_DIR": str(directory / "uv-cache"),
    }
    started = time.monotonic()
    completed = subprocess.run(  # noqa: S603 - comando fijo (el interprete de la sesion)
        [
            sys.executable,
            "-m",
            "cfdtrader.analysis.gate_sweep",
            "--data-root",
            str(REAL_DATA),
            "--reports-dir",
            str(directory),
            "--as-of",
            NOW.isoformat(),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        cwd=REPO_ROOT,
    )
    seconds = time.monotonic() - started
    json_path = directory / f"{STEM}.json"
    markdown_path = directory / f"{STEM}.md"
    payload = as_map(json.loads(json_path.read_text(encoding="utf-8")))
    return CliRun(
        directory=directory,
        code=completed.returncode,
        seconds=seconds,
        stderr=completed.stderr,
        report_sha256=as_str(payload["report_sha256"]),
        json_sha256=hashlib.sha256(json_path.read_bytes()).hexdigest(),
        markdown_sha256=hashlib.sha256(markdown_path.read_bytes()).hexdigest(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def report(tmp_path_factory: pytest.TempPathFactory) -> gate_sweep.GateSweepReport:
    """La corrida real completa, **una vez** por sesion, escribiendo bajo ``tmp_path``."""
    root = tmp_path_factory.mktemp("gate_sweep")
    return analyse(store=Store(REAL_DATA), reports_dir=root, as_of=NOW, write=True)


# ─────────────────────────────────────────────────────────────────────────────
# A1 · Modulo, tests y disciplina de importaciones
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_module_and_test_file_exist() -> None:
    """A1: existen los dos ficheros nuevos y el modulo se importa."""
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    assert gate_sweep.ANALYSIS == "cfdtrader.analysis.gate_sweep"
    assert gate_sweep.__name__ == "cfdtrader.analysis.gate_sweep"


def test_a1_module_does_not_import_the_network() -> None:
    """A1: el barrido es local: ningun modulo de red en el AST."""
    forbidden = ("socket", "urllib", "http", "requests", "httpx", "aiohttp", "yfinance")
    imported: set[str] = set()
    for node in ast.walk(TREE):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(forbidden)


# ─────────────────────────────────────────────────────────────────────────────
# A2 · La rejilla declarada, verbatim
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a2_grid_is_verbatim(report: gate_sweep.GateSweepReport) -> None:
    """A2: ``grid.cells`` es la tabla declarada, celda a celda, con los once campos."""
    cells = as_objects(at(report.payload, "grid", "cells"))
    assert len(cells) == 10
    assert as_int(at(report.payload, "grid", "n_cells")) == 10
    assert len(GRID_CELLS) == len(DECLARED_GRID)
    for published, declared in zip(cells, DECLARED_GRID, strict=True):
        cell_id, pmin, tiers, mult_a, mult_b, ev, risk = declared
        assert published["cell_id"] == cell_id
        assert published["tier_a_min_probability"] == pmin
        assert published["authorized_tiers"] == tiers
        assert published["tier_a_cost_multiple"] == mult_a
        assert published["tier_b_cost_multiple"] == mult_b
        assert published["ev_threshold_pct"] == ev
        assert published["risk_per_trade_pct"] == risk
        assert set(published) == {"cell_id"} | set(GATE_FIELDS)


@needs_store
def test_a2_grid_axes_are_the_declared_ones(report: gate_sweep.GateSweepReport) -> None:
    """A2: los ejes publicados son los valores distintos de la tabla, sin ninguno de mas."""
    axes = as_map(at(report.payload, "grid", "axes"))
    assert axes["tier_a_min_probability"] == ["0.58", "0.50", "0.55", "0.62", "0.70"]
    assert axes["authorized_tiers"] == [["A"], ["A", "B"]]
    assert axes["tier_a_cost_multiple"] == ["3", "100"]
    assert axes["tier_b_cost_multiple"] == ["2", "100"]
    assert axes["ev_threshold_pct"] == ["2*c", "32*c"]
    assert axes["risk_per_trade_pct"] == ["1", "2"]
    assert set(axes) == {
        "tier_a_min_probability",
        "authorized_tiers",
        "tier_a_cost_multiple",
        "tier_b_cost_multiple",
        "ev_threshold_pct",
        "risk_per_trade_pct",
    }


@needs_store
def test_a2_cell_parameters_are_the_declared_grid(report: gate_sweep.GateSweepReport) -> None:
    """A2: los once parametros resueltos de cada celda salen de la fila declarada."""
    published = as_objects(at(report.payload, "grid", "cells"))
    for cell, declared in zip(published, DECLARED_GRID, strict=True):
        resolved = as_map(report.cell(as_str(cell["cell_id"])).params.model_dump())
        assert str(resolved["tier_a_min_probability"]) == as_str(cell["tier_a_min_probability"])
        assert list(cast("list[str]", resolved["authorized_tiers"])) == list(
            cast("list[str]", cell["authorized_tiers"])
        )
        assert str(resolved["tier_a_cost_multiple"]) == as_str(cell["tier_a_cost_multiple"])
        assert str(resolved["tier_b_cost_multiple"]) == as_str(cell["tier_b_cost_multiple"])
        assert str(resolved["risk_per_trade_pct"]) == as_str(cell["risk_per_trade_pct"])
        assert declared[0] == cell["cell_id"]


@needs_store
def test_a2_all_resolved_parameters_are_inside_the_grid(report: gate_sweep.GateSweepReport) -> None:
    """A2: ningun parametro fuera de la rejilla: los cinco fijos son los declarados."""
    fixed = as_map(at(report.payload, "grid", "fixed"))
    assert fixed == {
        "broker": "escenario:sin-decidir-#59",
        "max_daily_loss_pct": "2",
        "max_weekly_loss_pct": "5",
        "max_monthly_loss_pct": "10",
        "r_pct": "1",
    }
    for result in report.results:
        assert result.params.broker == fixed["broker"]
        assert str(result.params.max_daily_loss_pct) == fixed["max_daily_loss_pct"]
        assert str(result.params.max_weekly_loss_pct) == fixed["max_weekly_loss_pct"]
        assert str(result.params.max_monthly_loss_pct) == fixed["max_monthly_loss_pct"]
        assert str(result.params.r_pct) == fixed["r_pct"]


# ─────────────────────────────────────────────────────────────────────────────
# A3 · Un comando, sin reloj, y ``write = False`` no escribe
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a3_write_false_writes_nothing(tmp_path: Path, report: gate_sweep.GateSweepReport) -> None:
    """A3: ``write = False`` no escribe **nada** en el directorio pedido."""
    directory = tmp_path / "silencioso"
    silent = analyse(store=Store(REAL_DATA), reports_dir=directory, as_of=NOW, write=False)
    assert not directory.exists()
    assert silent.report_sha256 == report.report_sha256
    assert silent.json_text() == report.json_text()


def test_a3_missing_or_invalid_as_of_writes_nothing(tmp_path: Path) -> None:
    """A3: ``--as-of`` es obligatorio y no se lee el reloj: sin el, no se escribe nada."""
    directory = tmp_path / "sin-as-of"
    assert main(["--data-root", str(REAL_DATA), "--reports-dir", str(directory)]) == 2
    assert not directory.exists()
    assert (
        main(["--data-root", str(REAL_DATA), "--reports-dir", str(directory), "--as-of", "no-iso"])
        == 2
    )
    assert not directory.exists()


def test_a3_main_writes_to_the_requested_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: gate_sweep.GateSweepReport
) -> None:
    """A3: el camino de exito del CLI pasa ``write = True`` y devuelve ``0``."""
    calls: dict[str, object] = {}

    def fake_analyse(
        *, store: Store, reports_dir: Path, as_of: datetime, write: bool = True
    ) -> gate_sweep.GateSweepReport:
        calls.update({"reports_dir": reports_dir, "as_of": as_of, "write": write})
        return report

    monkeypatch.setattr(gate_sweep, "analyse", fake_analyse)
    directory = tmp_path / "cli"
    code = main(
        [
            "--data-root",
            str(REAL_DATA),
            "--reports-dir",
            str(directory),
            "--as-of",
            NOW.isoformat(),
        ]
    )
    assert code == 0
    assert calls["write"] is True
    assert calls["as_of"] == NOW


def test_a3_module_never_reads_the_clock() -> None:
    """A3: ninguna ruta del modulo consulta el reloj (``now``/``utcnow``/``today``)."""
    used: set[str] = set()
    for node in ast.walk(TREE):
        if isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Import):
            used.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            used.add(node.module.split(".")[0])
    assert used.isdisjoint(FORBIDDEN_CLOCK_ATTRIBUTES)
    assert "time" not in used
    assert "datetime" in used  # el modulo usa `datetime`, no `time`


# ─────────────────────────────────────────────────────────────────────────────
# A4 · El contrato publicado de cada celda
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a4_each_cell_publishes_the_contract(report: gate_sweep.GateSweepReport) -> None:
    """A4: cada celda publica su identidad, sus recuentos, su serie y su metrica."""
    expected_keys = {
        "cell_id",
        "parameters",
        "basis",
        "is_validation",
        "run_sha256",
        "plan_sha256",
        "n_test",
        "traded",
        "no_trade",
        "skipped",
        "direction_counts",
        "declared_cost_sum_pct",
        "exact_aggregates",
        "declared_series",
        "mean_return_pct",
        "rejections",
        "ledger",
    }
    for cell in as_objects(report.payload["cells"]):
        assert set(cell) == expected_keys
        assert cell["basis"] == BASIS_DECLARED_COST
        assert cell["is_validation"] is False
        assert set(as_map(cell["parameters"])) == set(GATE_FIELDS)
        series = as_map(cell["declared_series"])
        assert set(series) == {"units", "n", "series_pct", "series_sha256"}
        assert as_int(series["n"]) == len(as_list(series["series_pct"]))
        aggregates = as_map(cell["exact_aggregates"])
        assert aggregates["declared_cost_sum_pct"] == cell["declared_cost_sum_pct"]
        assert aggregates["n_test_without_expected_move"] == 0
        assert set(as_map(cell["direction_counts"])) == {"long", "short"}


@needs_store
def test_a4_declared_series_comes_from_the_reused_helper(
    report: gate_sweep.GateSweepReport,
) -> None:
    """A4: la serie es la de ``_series_of_run`` (cero exacto en ``no_trade``), no otra."""
    assert gate_sweep._series_of_run is pipeline_report._series_of_run  # pyright: ignore[reportPrivateUsage]
    for result in report.results:
        payload = gate_sweep._declared_series_payload(  # pyright: ignore[reportPrivateUsage]
            result.series_pct
        )
        assert payload["series_sha256"] == series_sha_of(
            cells_by_id(report.payload)[result.cell.cell_id]
        )
        expected = gate_sweep._series_of_run(result.run)  # pyright: ignore[reportPrivateUsage]
        assert tuple(as_floats(payload["series_pct"])) == expected
        assert result.series_pct == expected
        assert len(expected) == result.run.traded + result.run.no_trade
        assert sum(1 for value in expected if value == 0.0) >= result.run.no_trade


# ─────────────────────────────────────────────────────────────────────────────
# A5 · Una sola metrica con intervalo por celda
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a5_exactly_one_bootstrapped_metric_with_the_declared_seed(
    report: gate_sweep.GateSweepReport,
) -> None:
    """A5: ``mean_return_pct`` con semilla 43 y la configuracion de #28, y ninguna otra."""
    metric_keys = {"estimate", "lower", "upper", "basis", "n", "confidence_level", "n_bootstrap"}
    for cell in as_objects(report.payload["cells"]):
        block = as_map(cell["mean_return_pct"])
        assert metric_keys <= set(block)
        assert block["basis"] == BASIS_DECLARED_COST
        assert block["seed"] == 43
        assert block["seed"] == DEFAULT_BOOTSTRAP_SEED + METRIC_NAMES.index("mean_return_pct") + 1
        assert block["n_bootstrap"] == DEFAULT_BOOTSTRAP_SAMPLES
        assert block["confidence_level"] == DEFAULT_CONFIDENCE_LEVEL
        assert as_int(block["n"]) == 500
        candidates = [
            key
            for key, value in cell.items()
            if isinstance(value, Mapping) and "estimate" in cast("Mapping[str, object]", value)
        ]
        assert candidates == ["mean_return_pct"]
    assert as_list(at(report.payload, "checks", "bootstrapped_metrics_per_cell")) == [
        "mean_return_pct"
    ]


# ─────────────────────────────────────────────────────────────────────────────
# A6 · Ninguna metrica de base medida
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a6_payload_has_no_measured_basis(
    tmp_path: Path, report: gate_sweep.GateSweepReport
) -> None:
    """A6: sin claves ni valores prohibidos, y ``basis`` siempre ``declared_cost``."""
    directory = tmp_path / "limpio"
    json_path, markdown_path = report.write(directory)
    text = json_path.read_text(encoding="utf-8").lower()
    markdown = markdown_path.read_text(encoding="utf-8").lower()
    for token in FORBIDDEN_TOKENS:
        assert token not in text, token
        assert token not in markdown, token
    assert set(all_keys(report.payload)).isdisjoint({"net", "pnl_net"})
    bases = {
        value
        for value in all_values(report.payload)
        if isinstance(value, str) and value.endswith("_cost")
    }
    assert bases == {BASIS_DECLARED_COST}


def test_a6_module_never_calls_the_measured_metric_route() -> None:
    """A6: el AST no llama ``calculate_metrics`` ni lee un atributo de la base medida."""
    assert "calculate_metrics" not in SOURCE
    assert "pnl_net" not in SOURCE
    called = {
        node.attr
        for node in ast.walk(TREE)
        if isinstance(node, ast.Attribute) and node.attr.startswith(("net", "pnl_net"))
    }
    assert called == set()


# ─────────────────────────────────────────────────────────────────────────────
# A7 · La celda ``s1`` reproduce el brazo de coste declarado de #28
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a7_s1_reproduces_the_declared_arm(report: gate_sweep.GateSweepReport) -> None:
    """A7: ``s1`` == ``arms.coste_declarado`` del informe de #28, cifra a cifra."""
    assert PIPELINE_ARTIFACT.is_file()
    artifact = as_map(json.loads(PIPELINE_ARTIFACT.read_text(encoding="utf-8")))
    arm = as_map(at(artifact, "arms", "coste_declarado"))
    cell = cells_by_id(report.payload)[REFERENCE_CELL]

    assert as_int(cell["traded"]) == as_int(arm["traded"])
    assert as_int(cell["no_trade"]) == as_int(arm["no_trade"])
    assert as_int(cell["skipped"]) == as_int(arm["skipped"])
    assert series_sha_of(cell) == as_str(at(arm, "declared_series", "series_sha256"))
    assert as_floats(at(cell, "declared_series", "series_pct")) == as_floats(
        at(arm, "declared_series", "series_pct")
    )
    assert float(cast("float", cell["declared_cost_sum_pct"])) == float(
        cast("float", at(arm, "exact_aggregates", "declared_cost_sum_pct"))
    )
    assert as_map(cell["mean_return_pct"]) == as_map(at(arm, "metrics", "mean_return_pct"))


@needs_store
def test_a7_s1_is_the_declared_scenario(report: gate_sweep.GateSweepReport) -> None:
    """A7: la celda ``s1`` es, parametro a parametro, el escenario S1 de #28."""
    reference = report.cell(REFERENCE_CELL)
    assert reference.params == scenario_parameters(cost_pct=report.shared.cost.c_declared_pct)
    assert GRID_CELLS[0].cell_id == REFERENCE_CELL


# ─────────────────────────────────────────────────────────────────────────────
# A8 · Una celda sin operaciones: ``null`` con motivo, nunca 0
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a8_cell_without_trades_publishes_nulls_with_a_reason(
    report: gate_sweep.GateSweepReport,
) -> None:
    """A8: con 0 operadas no se inventa un 0: estimacion y limites ``null`` con ``reason``."""
    impossible = GridCell(
        cell_id="sin_operaciones",
        tier_a_min_probability=Decimal("0.58"),
        authorized_tiers=("A",),
        tier_a_cost_multiple=Decimal("1000000"),
        tier_b_cost_multiple=Decimal("1000000"),
        ev_threshold_multiple=Decimal("2"),
        risk_per_trade_pct=Decimal("1"),
    )
    result = gate_sweep.sweep_cell(report.shared, impossible)
    assert result.run.traded == 0
    payload = gate_sweep._cell_payload(  # pyright: ignore[reportPrivateUsage]
        result, shared=report.shared
    )
    block = as_map(payload["mean_return_pct"])
    assert block["estimate"] is None
    assert block["lower"] is None
    assert block["upper"] is None
    assert block["seed"] == 43
    assert "reason" in block
    assert as_int(block["n"]) == 500
    series = as_map(payload["declared_series"])
    assert all(value == 0.0 for value in as_floats(series["series_pct"]))
    assert as_map(payload["direction_counts"]) == {"long": 0, "short": 0}
    assert as_int(payload["traded"]) == 0
    assert as_int(payload["no_trade"]) == 500


def test_a8_metric_text_never_prints_a_fake_zero() -> None:
    """A8: la celda de Markdown de un intervalo ausente es ``null``, no ``0``."""
    text = gate_sweep._metric_text(  # pyright: ignore[reportPrivateUsage]
        {"estimate": None, "lower": None, "upper": None}
    )
    assert text == ("null", "null", "null")
    assert gate_sweep._metric_text(  # pyright: ignore[reportPrivateUsage]
        {"estimate": 0.5, "lower": -1.0, "upper": 2.0}
    ) == ("0.500000", "-1.000000", "2.000000")


def test_a8_direction_counts_needs_a_decision() -> None:
    """A8/A9: una sesion operada sin decision es un error tipado, no un cero silencioso."""
    outcome = SessionOutcome(
        fold_index=0,
        session=date(2026, 9, 23),
        session_index=0,
        status=STATUS_TRADED,
        reason="probando",
        skip_reason=None,
        gap_px=None,
        decision=None,
        entry_session=None,
        exit_session=None,
        entry_px=None,
        exit_px=None,
        exit_reason=None,
        exit_bar_index=None,
        notional_usd=None,
        gross_pct=None,
        pnl_declared_pct=None,
        pnl_net_pct=None,
        pnl_net_reason=None,
        cost=None,
    )
    fold = FoldOutcome(
        index=0, test_start=0, test_stop=1, sessions=(outcome,), traded=1, no_trade=0, skipped=0
    )
    run = BacktestRun(
        folds=(fold,),
        plan_sha256="0" * 64,
        purge_total=0,
        embargo_total=0,
        embargo_in_train_total=0,
        exclusions_are_no_op=True,
        uncovered=(),
        not_in_any_test=0,
        n_sessions=1,
        traded=1,
        no_trade=0,
        skipped=0,
        run_sha256="1" * 64,
        report={},
    )
    with pytest.raises(gate_sweep.GateSweepError):
        gate_sweep._direction_counts(run)  # pyright: ignore[reportPrivateUsage]


# ─────────────────────────────────────────────────────────────────────────────
# A9 · El contrato de sesion
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a9_session_contract_holds_in_every_cell(report: gate_sweep.GateSweepReport) -> None:
    """A9: ``traded + no_trade + skipped == n_test`` y ``long + short == traded``, por celda."""
    for cell in as_objects(report.payload["cells"]):
        counts = as_map(cell["direction_counts"])
        traded = as_int(cell["traded"])
        assert traded + as_int(cell["no_trade"]) + as_int(cell["skipped"]) == as_int(cell["n_test"])
        assert as_int(counts["long"]) + as_int(counts["short"]) == traded
        assert as_int(cell["skipped"]) == 0
        assert as_map(cell["ledger"])["traded"] == traded
        assert set(counts) == {"long", "short"}


# ─────────────────────────────────────────────────────────────────────────────
# A10 · Determinismo entre procesos
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a10_three_fresh_processes_agree(tmp_path: Path) -> None:
    """A10: tres procesos CLI frescos (``PYTHONHASHSEED`` 0, 1 y aleatorio) coinciden."""
    directories = [tmp_path / f"proceso-{index}" for index in range(3)]
    for directory in directories:
        directory.mkdir()
    seeds = ["0", "1", "random"]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(_run_cli, directory, seed)
            for directory, seed in zip(directories, seeds, strict=True)
        ]
        runs = [future.result() for future in futures]
    for run in runs:
        assert run.code == 0, run.stderr
        assert run.seconds < 300.0, "el barrido tiene que caber en el techo de 5 minutos"
    assert len({run.report_sha256 for run in runs}) == 1
    assert len({run.json_sha256 for run in runs}) == 1
    assert len({run.markdown_sha256 for run in runs}) == 1


# ─────────────────────────────────────────────────────────────────────────────
# A11 · Los digests, con el contrato de #89
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a11_digests_have_the_format_and_are_self_consistent(
    tmp_path: Path, report: gate_sweep.GateSweepReport
) -> None:
    """A11: ``sha256:`` + 64 hex, autoconsistencia recomputada y sin literales fijados."""
    directory = tmp_path / "digests"
    json_path, _ = report.write(directory)
    raw = json_path.read_text(encoding="utf-8")
    published = as_map(json.loads(raw))
    body = {key: value for key, value in published.items() if key != "report_sha256"}
    recomputed = _sha256_text(canonical_text(body))
    digest = as_str(published["report_sha256"])
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
    assert digest == report.report_sha256
    assert digest == recomputed
    assert as_str(report.payload["hash_format"]).startswith("sha256:<64 hex>")
    for cell in as_objects(published["cells"]):
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", series_sha_of(cell))
        declared_series = {
            key: value
            for key, value in as_map(cell["declared_series"]).items()
            if key != "series_sha256"
        }
        assert series_sha_of(cell) == _sha256_text(canonical_text(declared_series))


def test_a11_no_regenerable_digest_is_pinned() -> None:
    """A11: ni el modulo ni el test fijan el digest literal de un artefacto regenerable."""
    literals = re.findall(r"sha256:[0-9a-f]{64}", SOURCE + TEST_PATH.read_text(encoding="utf-8"))
    assert literals == []


# ─────────────────────────────────────────────────────────────────────────────
# A12 · La invariancia medida y los campos inertes
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a12_controls_share_the_reference_series(report: gate_sweep.GateSweepReport) -> None:
    """A12: ``inv_risk2`` e ``inv_thr32`` publican el mismo ``series_sha256`` que ``s1``."""
    cells = cells_by_id(report.payload)
    reference = series_sha_of(cells[REFERENCE_CELL])
    for control in INVARIANCE_CONTROLS:
        assert series_sha_of(cells[control]) == reference
        assert as_int(cells[control]["traded"]) == as_int(cells[REFERENCE_CELL]["traded"])
    holds = as_map(at(report.payload, "invariance", "holds"))
    assert holds == dict.fromkeys(INVARIANCE_CONTROLS, True)
    assert as_str(at(report.payload, "invariance", "reference")) == REFERENCE_CELL


@needs_store
def test_a12_the_eight_inert_fields_are_declared(report: gate_sweep.GateSweepReport) -> None:
    """A12: los ocho campos inertes, con su motivo, y la senal visible de la rejilla."""
    published = as_objects(report.payload["inert_parameters"])
    assert len(published) == 8
    assert tuple(as_str(item["field"]) for item in published) == INERT_FIELDS
    assert tuple(item["field"] for item in INERT_PARAMETERS) == INERT_FIELDS
    for item in published:
        assert len(as_str(item["reason"])) > 40
    traded = {as_int(cell["traded"]) for cell in as_objects(report.payload["cells"])}
    assert len(traded) >= 4
    assert as_str(report.payload["inertness_scope"]).startswith("estos ocho campos")


# ─────────────────────────────────────────────────────────────────────────────
# A13 · Senal visible y el hallazgo declarado
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a13_markdown_table_has_one_row_per_cell(report: gate_sweep.GateSweepReport) -> None:
    """A13: la tabla de resultados tiene 10 filas y al menos 4 valores distintos de ``traded``."""
    lines = render_markdown(report).splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("| celda | operadas")]
    assert len(starts) == 1
    rows: list[str] = []
    for line in lines[starts[0] + 2 :]:
        if not line.startswith("| `"):
            break
        rows.append(line)
    assert len(rows) == 10
    traded = {int(row.split("|")[2].strip()) for row in rows}
    assert len(traded) >= 4
    assert traded == {as_int(cell["traded"]) for cell in as_objects(report.payload["cells"])}


@needs_store
def test_a13_markdown_declares_the_short_finding(report: gate_sweep.GateSweepReport) -> None:
    """A13: el ``.md`` declara que en ``s1`` todas las operaciones son ``short`` (``1 - p``)."""
    markdown = render_markdown(report)
    assert "_favourable_probability" in markdown
    assert "1 - p" in markdown
    assert "todas las operaciones son `short`" in markdown
    assert "#98" in markdown
    s1 = cells_by_id(report.payload)[REFERENCE_CELL]
    assert as_int(s1["traded"]) == 31
    assert as_map(s1["direction_counts"]) == {"long": 0, "short": 31}
    assert as_str(at(report.payload, "finding", "cell")) == REFERENCE_CELL


@needs_store
def test_a13_markdown_is_deterministic_and_matches_the_payload(
    report: gate_sweep.GateSweepReport,
) -> None:
    """A13: el Markdown no trae cifras fuera del payload y se repite identico."""
    markdown = render_markdown(report)
    assert markdown == render_markdown(report)
    for cell in as_objects(report.payload["cells"]):
        estimate = as_map(cell["mean_return_pct"])["estimate"]
        expected = "null" if estimate is None else f"{float(cast('float', estimate)):.6f}"
        assert f"| `{cell['cell_id']}` | {cell['traded']} |" in markdown
        assert expected in markdown


# ─────────────────────────────────────────────────────────────────────────────
# A15 · Ficheros (SUBSET + DISJUNTO) y reuso estructural de #28
# ─────────────────────────────────────────────────────────────────────────────
def test_a15_written_files_and_frozen_modules() -> None:
    """A15: los dos ficheros de la entrega estan en el diff y ningun congelado se toca."""
    assert _git("status", "--porcelain").strip() == ""
    changed = set(_git("diff", "--name-only", f"{BASE_COMMIT}..HEAD").splitlines())
    assert set(WRITTEN) <= changed
    assert changed.isdisjoint(FROZEN)


def test_a15_helpers_are_imported_not_redefined() -> None:
    """A15: el AST no redefine ningun helper de #28 y la identidad de objetos se mantiene.

    ``_declared_return_pct`` no viaja importado al modulo nuevo porque no se usa alli
    directamente: lo usa ``_series_of_run`` de #28, que si se importa (identidad comprobada).
    Lo que A15 exige del AST es que el modulo **no lo redefina**.
    """
    defined = {
        node.name
        for node in TREE.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert defined.isdisjoint(REUSE_NAMES)
    imports = {
        node.module
        for node in ast.walk(TREE)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "cfdtrader.analysis.pipeline_report" in imports
    for name in REUSED_HELPERS:
        assert hasattr(gate_sweep, name), name
        assert getattr(gate_sweep, name) is getattr(pipeline_report, name), name
    for name in REUSE_NAMES:
        if not hasattr(gate_sweep, name):
            continue
        assert getattr(gate_sweep, name) is getattr(pipeline_report, name), name
    assert (
        gate_sweep._series_of_run is pipeline_report._series_of_run  # pyright: ignore[reportPrivateUsage]
    )


def test_a15_module_does_not_import_names_it_does_not_use() -> None:
    """A15: cada nombre importado de ``pipeline_report`` aparece usado en el modulo."""
    imported: list[str] = []
    for node in ast.walk(TREE):
        if isinstance(node, ast.ImportFrom) and node.module == "cfdtrader.analysis.pipeline_report":
            imported.extend(alias.asname or alias.name for alias in node.names)
    assert imported, "el modulo tiene que reutilizar los helpers de #28"
    used = {node.id for node in ast.walk(TREE) if isinstance(node, ast.Name)}
    used |= {node.attr for node in ast.walk(TREE) if isinstance(node, ast.Attribute)}
    assert set(imported) <= used


def test_a15_no_extra_file_is_declared() -> None:
    """A15: la entrega no toca ``__init__.py``: el diff no lo declara."""
    changed = set(_git("diff", "--name-only", f"{BASE_COMMIT}..HEAD").splitlines())
    assert changed.isdisjoint(EXTRA_ALLOWED)


# ─────────────────────────────────────────────────────────────────────────────
# Apoyo: helpers puros y errores tipados
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_support_missing_cell_is_a_typed_error(report: gate_sweep.GateSweepReport) -> None:
    """Una celda pedida que no esta en la rejilla falla con error tipado."""
    with pytest.raises(gate_sweep.GateSweepError):
        report.cell("no_existe")


@needs_store
def test_support_report_stem_and_cell_lookup(report: gate_sweep.GateSweepReport) -> None:
    """El nombre base del informe y la busqueda de celdas son los declarados."""
    assert report.report_stem == STEM
    assert [result.cell.cell_id for result in report.results] == [row[0] for row in DECLARED_GRID]
    assert report.shared.n_test == 500
    assert report.shared.n_test_without_expected_move == 0
    assert len(report.cell(REFERENCE_CELL).series_pct) == 500


def test_support_declared_grid_literals() -> None:
    """La rejilla del modulo y la tabla del test son la misma, literal a literal."""
    assert [cell.cell_id for cell in GRID_CELLS] == [row[0] for row in DECLARED_GRID]
    assert isinstance(GRID_CELLS[0], GridCell)
    assert gate_sweep.FIXED_PARAMETERS["broker"] == "escenario:sin-decidir-#59"


def test_support_sequence_helpers() -> None:
    """Los helpers puros de la rejilla: el multiplo en texto y los distintos ordenados."""
    assert gate_sweep._multiple_text(Decimal("2")) == "2*c"  # pyright: ignore[reportPrivateUsage]
    assert (
        gate_sweep._multiple_text(Decimal("32"))  # pyright: ignore[reportPrivateUsage]
        == "32*c"
    )
    assert gate_sweep._distinct([1, 2, 1, 3, 2]) == [1, 2, 3]  # pyright: ignore[reportPrivateUsage]
    assert gate_sweep._distinct(["a", "b", "a"]) == ["a", "b"]  # pyright: ignore[reportPrivateUsage]
