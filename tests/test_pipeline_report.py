"""Tests del informe del pipeline completo (#28): A1 a A15.

La corrida real es **cara** (~110 s: 73 intervalos bootstrap de 10.000 remuestreos sobre 500
sesiones con el helper puro de #15), asi que se hace **una sola vez** por sesion
(``real_report``) y todas las comprobaciones estructurales y numericas se apoyan en ella. La
segunda pasada de A2 y los dos procesos frescos de A3 (``PYTHONHASHSEED`` 0 y 1) se lanzan **en
paralelo** y escriben bajo ``tmp_path``, con ``TMPDIR`` y ``UV_CACHE_DIR`` redirigidos ahi: la
sesion de tests no toca el ``data/`` ni el ``runs/`` del repositorio (lo comprueban las fixtures
de ``conftest.py``).

El unico criterio que exige un ``analyse`` completo **sin** los intervalos (que nada se escriba
con ``write = False``) usa ``_patched_fast``: el doble deterministico de
``bootstrap_confidence_interval`` no cambia ninguna decision ni ninguna clave del payload, y
esperar otros ~110 s por el no mediria nada.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Generator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final, cast

import polars as pl
import pytest

from cfdtrader.analysis import pipeline_report
from cfdtrader.analysis.backtest_report import NOTIONAL_USD, PHASE1_PLAN, SERIES_ID, load_history
from cfdtrader.analysis.pipeline_report import (
    ARM_COSTE_DECLARADO,
    ARM_ESCENARIO,
    ARM_NAMES,
    ARM_OFICIAL,
    BASIS_DECLARED_COST,
    HASH_PREFIX,
    METRIC_NAMES,
    NO_INTERVAL_METRICS,
    REPORT_PREFIX,
    RULE_11_BAND,
    SERIES_UNITS,
    SESSION_RULES,
    TEMPORAL_MAPPING,
    AlignmentError,
    InvalidAsOfError,
    MissingAsOfError,
    PipelineReport,
    PipelineReportError,
    analyse,
    main,
    render_markdown,
    scenario_parameters,
)
from cfdtrader.backtest.baselines import BASELINE_IDS, NO_TRADE
from cfdtrader.backtest.costs import (
    CostBreakdown,
    Side,
    cost_breakdown,
    declared_cost_model,
    declared_slippage_assumption,
)
from cfdtrader.backtest.engine import (
    STATUS_NO_TRADE,
    STATUS_SKIPPED,
    STATUS_TRADED,
    BacktestRun,
    Direction,
    FoldOutcome,
    SessionOutcome,
    SessionView,
    canonical_text,
)
from cfdtrader.backtest.metrics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    ConfidenceInterval,
    bootstrap_confidence_interval,
)
from cfdtrader.data.store import Store
from cfdtrader.decision.gate import (
    DECISION_THRESHOLD,
    GATE_HASH_PREFIX,
    TARGET_MIN_COST_MULTIPLE,
    TIER_A,
    TIER_B,
    TIER_C,
    GateOutput,
    GateParameters,
    GateStatus,
    Tier,
)
from cfdtrader.models.baseline import BASELINE_FEATURES, DESIGN_LAG_SESSIONS

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"
MODULE_PATH: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "analysis" / "pipeline_report.py"
TEST_PATH: Final[Path] = Path(__file__).resolve()

#: Instante **declarado** de todas las corridas: el modulo nunca lee el reloj.
NOW: Final[datetime] = datetime(2026, 9, 23, 22, 0, tzinfo=UTC)
STEM: Final[str] = f"{REPORT_PREFIX}_{NOW.date().isoformat()}"

#: Commit de partida de la entrega (el arbol venia limpio en el): A15 compara contra el.
BASE_COMMIT: Final[str] = "6aa582d"

#: Los cinco ficheros que la entrega debe traer en el diff, y los cuatro congelados (A15).
#: #98: la entrega toca cinco ficheros (el modulo, el barrido y los tres tests) y ningun
#: congelado; la guarda sigue siendo SUBSET + DISJUNTO, nunca `changed <= ALLOWED` (#89).
WRITTEN: Final[frozenset[str]] = frozenset(
    {
        "src/cfdtrader/analysis/pipeline_report.py",
        "src/cfdtrader/analysis/gate_sweep.py",
        "tests/test_pipeline_report.py",
        "tests/test_gate_sweep.py",
        "tests/test_phase2_dominance.py",
    }
)
#: Los ficheros que la entrega de #80 **si** toca (motor, gate) se han retirado de
#: ``FROZEN``: este modulo no los toca, pero la entrega transversal de #80 si (#80 A6).
FROZEN: Final[tuple[str, ...]] = (
    "src/cfdtrader/backtest/baselines.py",
    "src/cfdtrader/backtest/metrics.py",
    "src/cfdtrader/models/baseline.py",
    "src/cfdtrader/analysis/feature_frame.py",
)

#: Modulos de red que el informe no puede importar (A1).
FORBIDDEN_IMPORTS: Final[tuple[str, ...]] = (
    "socket",
    "urllib",
    "http",
    "requests",
    "httpx",
    "aiohttp",
    "ftplib",
    "smtplib",
    "yfinance",
    "fredapi",
)

SOURCE: Final[str] = MODULE_PATH.read_text(encoding="utf-8")
TREE: Final[ast.Module] = ast.parse(SOURCE)
COMMON_KEYS: Final[tuple[str, ...]] = (
    "estimate",
    "lower",
    "upper",
    "basis",
    "n",
    "confidence_level",
    "n_bootstrap",
    "seed",
)

needs_store = pytest.mark.skipif(
    not (REAL_DATA / "derived" / "labels").exists(),
    reason="el almacen real no esta en el arbol: los numeros de A4-A13 son los suyos",
)


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


def as_float(node: object) -> float:
    """El nodo como numero."""
    return float(cast("float", node))


def as_objects(node: object) -> list[dict[str, object]]:
    """Una lista de mappings del payload."""
    return [as_map(item) for item in as_list(node)]


@dataclass(frozen=True, slots=True)
class CliRun:
    """El resultado de una corrida de la CLI en un proceso fresco."""

    directory: Path
    code: int
    stderr: str
    report_sha256: str
    json_sha256: str
    markdown_sha256: str

    def json_path(self) -> Path:
        """Ruta del JSON escrito por esa corrida."""
        return self.directory / f"{STEM}.json"

    def markdown_path(self) -> Path:
        """Ruta del Markdown escrito por esa corrida."""
        return self.directory / f"{STEM}.md"


def sessions_of(run: BacktestRun) -> tuple[SessionOutcome, ...]:
    """Las sesiones de *test* de la corrida, en orden."""
    return tuple(outcome for fold in run.folds for outcome in fold.sessions)


def row_block(report: PipelineReport, name: str) -> dict[str, object]:
    """El bloque publicado de una fila de la tabla o de un brazo."""
    if name in ARM_NAMES:
        return as_map(at(report.payload, "arms", name))
    for row in as_objects(at(report.payload, "table", "rows")):
        if row["row"] == name:
            return row
    raise AssertionError(f"el informe no trae la fila {name!r}")


def metrics_of(report: PipelineReport, name: str) -> dict[str, dict[str, object]]:
    """El bloque ``metrics`` de esa fila o brazo."""
    return cast("dict[str, dict[str, object]]", row_block(report, name)["metrics"])


def series_of(report: PipelineReport, name: str) -> tuple[float, ...]:
    """La serie declarada de esa fila (de un brazo, derivada aqui con la formula de A10)."""
    if name in ARM_NAMES:
        return declared_series(report.arm(name).run)
    return report.row(name).series_pct


def declared_series(run: BacktestRun) -> tuple[float, ...]:
    """La serie declarada de una corrida, recalculada en el test: ``100 x gross - c_declared``."""
    values: list[float] = []
    for outcome in sessions_of(run):
        if outcome.status == STATUS_SKIPPED:
            continue
        if outcome.status == STATUS_NO_TRADE:
            values.append(0.0)
            continue
        assert outcome.gross_pct is not None and outcome.cost is not None
        values.append(100.0 * outcome.gross_pct - float(outcome.cost.c_declared_pct))
    return tuple(values)


def metric_block(metrics: Mapping[str, dict[str, object]], name: str) -> dict[str, object]:
    """Una metrica con su **etiqueta** completa comprobada (A8)."""
    block = metrics[name]
    for key in COMMON_KEYS:
        assert key in block, f"{name} no publica {key}"
    assert block["basis"] == BASIS_DECLARED_COST
    assert block["n_bootstrap"] == DEFAULT_BOOTSTRAP_SAMPLES
    assert block["confidence_level"] == DEFAULT_CONFIDENCE_LEVEL
    seed = as_int(block["seed"])
    assert 0 <= seed <= 2**32 - 1
    return block


def beta_of(series: Sequence[float], benchmark: Sequence[float]) -> float:
    """La beta de la serie contra el benchmark, recalculada aqui (formula de A13)."""
    mean_benchmark = math.fsum(benchmark) / len(benchmark)
    mean_series = math.fsum(series) / len(series)
    variance = math.fsum((value - mean_benchmark) ** 2 for value in benchmark)
    if variance == 0.0:
        return math.nan
    covariance = math.fsum(
        (value - mean_series) * (reference - mean_benchmark)
        for value, reference in zip(series, benchmark, strict=True)
    )
    return covariance / variance


def _fast_interval(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> ConfidenceInterval:
    """Doble deterministico del intervalo: la estimacion, sin remuestreo."""
    estimate = float(statistic(values))
    return ConfidenceInterval(
        estimate=estimate,
        lower=estimate,
        upper=estimate,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


@contextmanager
def _patched_fast() -> Generator[None, None, None]:
    """Sustituye el helper de #15 por el doble y lo restaura al salir."""
    original = pipeline_report.bootstrap_confidence_interval
    pipeline_report.bootstrap_confidence_interval = _fast_interval
    try:
        yield
    finally:
        pipeline_report.bootstrap_confidence_interval = original


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def real_report(tmp_path_factory: pytest.TempPathFactory) -> Generator[PipelineReport, None, None]:
    """La corrida real completa, **una vez** por sesion, escribiendo bajo ``tmp_path``."""
    root = tmp_path_factory.mktemp("pipeline_report")
    yield analyse(store=Store(REAL_DATA), reports_dir=root, as_of=NOW, write=True)


def _run_cli(directory: Path, hash_seed: str) -> CliRun:
    """Corre la CLI en un proceso fresco, con su `PYTHONHASHSEED` y su dir."""
    environment = {
        **os.environ,
        "PYTHONHASHSEED": hash_seed,
        "TMPDIR": str(directory),
        "UV_CACHE_DIR": str(directory / "uv-cache"),
    }
    completed = subprocess.run(  # noqa: S603 - comando fijo (el interprete de la sesion)
        [
            sys.executable,
            "-m",
            "cfdtrader.analysis.pipeline_report",
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
    json_path = directory / f"{STEM}.json"
    markdown_path = directory / f"{STEM}.md"
    digest = ""
    json_digest = ""
    markdown_digest = ""
    if completed.returncode == 0 and json_path.is_file():
        digest = as_str(json.loads(json_path.read_text(encoding="utf-8"))["report_sha256"])
        json_digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
        markdown_digest = hashlib.sha256(markdown_path.read_bytes()).hexdigest()
    return CliRun(
        directory=directory,
        code=completed.returncode,
        stderr=completed.stderr,
        report_sha256=digest,
        json_sha256=json_digest,
        markdown_sha256=markdown_digest,
    )


@pytest.fixture(scope="session")
def fresh_runs(
    real_report: PipelineReport, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, CliRun]:
    """Dos procesos frescos (``PYTHONHASHSEED`` 0 y 1) **en paralelo**, cada uno con su dir.

    Depende de ``real_report``: la corrida de la sesion hace de tercer proceso (su
    ``PYTHONHASHSEED`` es el aleatorio del proceso de pytest, que es el caso ``random``) y sus
    ficheros son la primera pasada para la comparacion byte a byte de A2.
    """
    root = tmp_path_factory.mktemp("pipeline_fresh")
    with ThreadPoolExecutor(max_workers=2) as pool:
        zero = pool.submit(_run_cli, root / "seed0", "0")
        one = pool.submit(_run_cli, root / "seed1", "1")
        result_zero = zero.result()
        result_one = one.result()
    return {"seed0": result_zero, "seed1": result_one}


# ─────────────────────────────────────────────────────────────────────────────
# A1 · Modulo, CLI, ausencia de reloj y de red
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_module_api_and_no_clock() -> None:
    """A1: existe el modulo con la API minima y su AST no lee el reloj."""
    required = (
        "analyse",
        "main",
        "scenario_parameters",
        "render_markdown",
        "PipelineReport",
        "AlignmentError",
        "ARM_NAMES",
        "METRIC_NAMES",
        "RULE_11_BAND",
    )
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    assert [name for name in required if name not in pipeline_report.__all__] == []
    assert all(hasattr(pipeline_report, name) for name in required)

    forbidden = {
        ("datetime", "now"),
        ("datetime", "utcnow"),
        ("date", "today"),
        ("time", "time"),
    }
    for node in ast.walk(TREE):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert (node.value.id, node.attr) not in forbidden
        if isinstance(node, ast.Import):
            assert "time" not in [alias.name for alias in node.names]
        if isinstance(node, ast.ImportFrom):
            assert node.module != "time"


def test_a1_no_network_imports() -> None:
    """A1: el modulo no importa ningun cliente de red."""
    imported: set[str] = set()
    for node in ast.walk(TREE):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(FORBIDDEN_IMPORTS)


def _calls(name: str) -> bool:
    """``True`` si el modulo llama a esa funcion en cualquier forma."""
    for node in ast.walk(TREE):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id == name:
            return True
        if isinstance(function, ast.Attribute) and function.attr == name:
            return True
    return False


def test_a1_cli_runs_without_network(fresh_runs: Mapping[str, CliRun]) -> None:
    """A1: la CLI corre en un proceso fresco con `--as-of` y sin red."""
    for name in ("seed0", "seed1"):
        run = fresh_runs[name]
        assert run.code == 0, run.stderr
        assert run.json_sha256 and run.markdown_sha256
    for option in ("--data-root", "--reports-dir", "--settings", "--as-of"):
        assert option in SOURCE


# ─────────────────────────────────────────────────────────────────────────────
# A2 · Ficheros, `--as-of` y segunda pasada
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a2_writes_only_the_two_reports(real_report: PipelineReport) -> None:
    """A2: los dos ficheros con la fecha del `as_of` en UTC, y **nada** mas fuera de ahi."""
    root = real_report.reports_dir
    assert sorted(path.name for path in root.rglob("*") if path.is_file()) == [
        f"{STEM}.json",
        f"{STEM}.md",
    ]
    assert "pytest" in str(root)
    assert real_report.report_stem == STEM
    assert real_report.report_date == NOW.date()
    assert at(real_report.payload, "generated_at") == NOW.isoformat()


@needs_store
def test_a2_second_pass_is_byte_identical(
    real_report: PipelineReport, fresh_runs: Mapping[str, CliRun]
) -> None:
    """A2: la segunda pasada (otro proceso, mismo `as_of`) deja los dos ficheros identicos."""
    root = real_report.reports_dir
    fresh = fresh_runs["seed0"]
    assert hashlib.sha256((root / f"{STEM}.json").read_bytes()).hexdigest() == fresh.json_sha256
    assert hashlib.sha256((root / f"{STEM}.md").read_bytes()).hexdigest() == fresh.markdown_sha256
    assert fresh.json_path().read_text(encoding="utf-8") == (root / f"{STEM}.json").read_text(
        encoding="utf-8"
    )
    assert fresh.markdown_path().read_text(encoding="utf-8") == (root / f"{STEM}.md").read_text(
        encoding="utf-8"
    )


@needs_store
def test_a2_write_false_and_missing_as_of_write_nothing(tmp_path: Path) -> None:
    """A2: `write = False` no escribe nada y la CLI sin `--as-of` falla sin escribir."""
    silent = tmp_path / "silencioso"
    with _patched_fast():
        report = analyse(store=Store(REAL_DATA), reports_dir=silent, as_of=NOW, write=False)
    assert not silent.exists()
    assert report.report_sha256.startswith("sha256:")
    assert report.payload["basis"] == BASIS_DECLARED_COST

    empty = tmp_path / "vacio"
    assert main(["--reports-dir", str(empty)]) == 2
    assert not empty.exists()
    assert main(["--reports-dir", str(empty), "--as-of", "no-es-iso"]) == 2
    assert not empty.exists()
    with pytest.raises(MissingAsOfError):
        pipeline_report._parse_as_of(None)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(InvalidAsOfError):
        pipeline_report._parse_as_of("ayer")  # pyright: ignore[reportPrivateUsage]
    assert pipeline_report._parse_as_of("2026-09-23T22:00:00+00:00") == NOW  # pyright: ignore[reportPrivateUsage]


#: Operadas del artefacto previo **sintetico** del bloque `regeneration`: no es el 31 de S1, es
#: un valor propio del caso para no fijar ningun recuento regenerable (#98).
SYNTHETIC_PREVIOUS_TRADED: Final[int] = 7


def test_previous_artifact_is_keyword_only_and_the_cli_declares_the_flag() -> None:
    """A5/#98: `analyse` acepta `previous_artifact` (keyword-only, por defecto `None`).

    Sin la bandera la corrida es un no-op byte a byte: lo vigilan `test_a2_second_pass_is_byte_
    identical` y `test_a3_identical_across_fresh_processes` (la corrida de sesion y dos procesos
    frescos dan el mismo `report_sha256`).
    """
    parameters = inspect.signature(analyse).parameters
    assert "previous_artifact" in parameters
    declaration = parameters["previous_artifact"]
    assert declaration.kind is inspect.Parameter.KEYWORD_ONLY
    assert declaration.default is None
    assert "--previous-artifact" in SOURCE


@needs_store
def test_previous_artifact_block_is_top_level_and_measures_the_delta(tmp_path: Path) -> None:
    """A5/A6/#98: el bloque `regeneration` es de nivel superior y **mide** el antes/despues.

    Se comprueba con el bootstrap doblado (`_patched_fast`): el doble no cambia ninguna decision
    ni el bloque, y evita repetir los ~110 s de la corrida real por una propiedad estructural.
    """
    previous = tmp_path / "pipeline_backtest_2026-09-22.json"
    previous.write_text(
        json.dumps(
            {
                "arms": {
                    ARM_OFICIAL: {"traded": 0},
                    ARM_ESCENARIO: {"traded": 0},
                    ARM_COSTE_DECLARADO: {"traded": SYNTHETIC_PREVIOUS_TRADED},
                }
            }
        ),
        encoding="utf-8",
    )
    with _patched_fast():
        plain = analyse(
            store=Store(REAL_DATA), reports_dir=tmp_path / "sin-bandera", as_of=NOW, write=False
        )
        report = analyse(
            store=Store(REAL_DATA),
            reports_dir=tmp_path / "con-bandera",
            as_of=NOW,
            write=False,
            previous_artifact=previous,
        )
    assert "regeneration" not in plain.payload
    block = as_map(report.payload["regeneration"])
    assert set(report.payload) - set(plain.payload) == {"regeneration"}
    assert "arms" not in block
    assert block["previous_artifact"] == previous.name
    assert block["subject"] == "#98"
    assert block["previous_published_directions"] is False
    rows = {as_str(row["arm"]): row for row in as_objects(block["rows"])}
    assert set(rows) == set(ARM_NAMES)
    declared = rows[ARM_COSTE_DECLARADO]
    assert as_int(declared["traded_before"]) == SYNTHETIC_PREVIOUS_TRADED
    assert as_int(declared["traded_after"]) == report.arm(ARM_COSTE_DECLARADO).run.traded
    assert as_int(declared["delta_traded"]) == (
        report.arm(ARM_COSTE_DECLARADO).run.traded - SYNTHETIC_PREVIOUS_TRADED
    )
    assert declared["previous_published_direction_counts"] is False
    assert as_map(declared["direction_counts"]) == as_map(
        row_block(report, ARM_COSTE_DECLARADO)["direction_counts"]
    )
    markdown = render_markdown(report)
    assert markdown.count("## Regeneración") == 1
    assert f"`{previous.name}`" in markdown
    assert "#98" in markdown


# ─────────────────────────────────────────────────────────────────────────────
# A3 · `report_sha256`: formato, determinismo y valor fijado
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a3_hash_format_and_self_consistency(real_report: PipelineReport) -> None:
    """A3: el sha256 lleva el prefijo y es el del payload **sin** la clave del hash."""
    digest = real_report.report_sha256
    assert digest.startswith("sha256:")
    body = digest.removeprefix("sha256:")
    assert len(body) == 64
    assert all(character in "0123456789abcdef" for character in body)
    assert "report_sha256" not in real_report.payload
    assert hashlib.sha256(canonical_text(real_report.payload).encode("utf-8")).hexdigest() == body
    assert as_str(json.loads(real_report.json_text())["report_sha256"]) == digest


@needs_store
def test_a3_identical_across_fresh_processes(
    real_report: PipelineReport, fresh_runs: Mapping[str, CliRun]
) -> None:
    """A3: el mismo hash con `PYTHONHASHSEED` 0, 1 y el aleatorio de este proceso.

    #92: ademas, el bloque de la tasa por operacion es identico byte a byte entre la corrida de
    la sesion y los dos procesos frescos (criterio 7 de #92).
    """
    assert fresh_runs["seed0"].report_sha256 == real_report.report_sha256
    assert fresh_runs["seed1"].report_sha256 == real_report.report_sha256
    ours = metrics_of(real_report, ARM_COSTE_DECLARADO)["hit_rate_per_trade"]
    for name in ("seed0", "seed1"):
        fresh = as_map(json.loads(fresh_runs[name].json_path().read_text(encoding="utf-8")))
        published = as_map(at(fresh, "arms", ARM_COSTE_DECLARADO, "metrics"))["hit_rate_per_trade"]
        assert as_map(published) == ours


@needs_store
def test_a3_hash_is_fixed_with_prefix(real_report: PipelineReport) -> None:
    """A3: el digest es funcion del payload (no una constante cableada), con su prefijo.

    #96: aqui se **fijaba** el literal `report_sha256` de la corrida real. Ese digest depende
    de artefactos regenerables (#92 anadio `hit_rate_per_trade` al payload y lo cambio), asi
    que cualquier tarea posterior lo invalida: un dorado asi convierte un trabajo ajeno en un
    fallo de #28. El formato (`sha256:` + 64 hex), la autoconsistencia con el payload y el
    determinismo entre procesos ya los vigilan `test_a3_hash_format_and_self_consistency` y
    `test_a3_identical_across_fresh_processes`; aqui queda la sensibilidad: cambiar un campo
    del payload cambia el digest.
    """
    mutated = dict(real_report.payload)
    mutated["generated_at"] = "1999-01-01T00:00:00+00:00"
    recomputed = "sha256:" + hashlib.sha256(canonical_text(mutated).encode("utf-8")).hexdigest()
    assert recomputed != real_report.report_sha256


# ─────────────────────────────────────────────────────────────────────────────
# A4 · Los tres brazos
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a4_three_arms_with_identity(real_report: PipelineReport) -> None:
    """A4: tres brazos, cada uno con `n_test`, `traded`, `basis`, `is_validation` y `run_sha256`."""
    arms = as_map(at(real_report.payload, "arms"))
    assert list(arms) == list(ARM_NAMES)
    for name in ARM_NAMES:
        arm = as_map(arms[name])
        for key in ("n_test", "traded", "basis", "is_validation", "run_sha256"):
            assert key in arm, f"el brazo {name} no publica {key}"
        assert as_int(arm["n_test"]) == len(real_report.test_sessions)
        assert arm["basis"] == BASIS_DECLARED_COST
        assert arm["is_validation"] is False
        assert as_str(arm["run_sha256"]).strip()


@needs_store
def test_a4_oficial_has_no_declared_field(real_report: PipelineReport) -> None:
    """A4: `arms.oficial` va con `GateParameters()` y `model_fields_set == set()`."""
    gate = as_map(at(real_report.payload, "arms", ARM_OFICIAL, "gate"))
    assert gate["params_declared"] == []
    assert gate["n_params_declared"] == 0
    arm = real_report.arm(ARM_OFICIAL)
    assert isinstance(arm.params, GateParameters)
    assert arm.params.model_fields_set == set()
    assert cast("list[str]", gate["undecided"])


@needs_store
def test_a4_escenario_declares_the_eleven_fields(real_report: PipelineReport) -> None:
    """A4/A6: S1 declara los once parametros, cada uno con su procedencia."""
    gate = as_map(at(real_report.payload, "arms", ARM_ESCENARIO, "gate"))
    assert gate["n_params_declared"] == 11
    declared = cast("list[str]", gate["params_declared"])
    parameters = as_map(at(real_report.payload, "scenario", "parameters"))
    assert sorted(declared) == sorted(parameters)
    for entry in parameters.values():
        assert as_str(as_map(entry)["provenance"]).strip()
    params = real_report.arm(ARM_ESCENARIO).params
    assert all(getattr(params, name) is not None for name in declared)
    assert params.authorized_tiers == ("A",)
    assert params.broker == "escenario:sin-decidir-#59"


@needs_store
def test_a4_scenario_derives_the_ev_threshold_from_the_cost_table(
    real_report: PipelineReport,
) -> None:
    """A6: el umbral del EV es el multiplo declarado de la regla 8 por el coste de la tabla."""
    params = scenario_parameters(cost_pct=Decimal("0.0042"))
    assert params.ev_threshold_pct == TARGET_MIN_COST_MULTIPLE * Decimal("0.0042")
    assert params.ev_threshold_pct == Decimal("0.0084")
    scenario = as_map(at(real_report.payload, "scenario"))
    assert scenario["cost_basis_pct"] == "0.0042"
    assert scenario["id"] == "S1"
    assert real_report.arm(ARM_ESCENARIO).params.model_fields_set == params.model_fields_set
    assert as_str(scenario["ev_threshold_derivation"]).startswith("ev_threshold_pct = ")


@needs_store
def test_a4_direction_counts_are_long_short_only(real_report: PipelineReport) -> None:
    """A4/#98: `arms.<brazo>.direction_counts` = `{long, short}`, `long + short == traded`.

    La forma es la misma que la de `gate_sweep._direction_counts` (#86), pero la implementa este
    modulo: el bloque del brazo **no** es el bloque de la celda del barrido. En el brazo de coste
    declarado las dos direcciones son **no nulas**: el sesgo corto de #98 esta corregido.
    """
    for name in ARM_NAMES:
        block = row_block(real_report, name)
        counts = as_map(block["direction_counts"])
        assert set(counts) == {"long", "short"}
        assert "nothing" not in counts
        assert as_int(counts["long"]) + as_int(counts["short"]) == as_int(block["traded"])
        run = real_report.arm(name).run
        longs = sum(
            1
            for outcome in sessions_of(run)
            if outcome.status == STATUS_TRADED
            and outcome.decision is not None
            and str(outcome.decision.direction) == "long"
        )
        assert as_int(counts["long"]) == longs
    declared = as_map(row_block(real_report, ARM_COSTE_DECLARADO)["direction_counts"])
    assert as_int(declared["long"]) > 0
    assert as_int(declared["short"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# A5 · Los dos brazos del gate no operan, y se mide
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a5_gate_arms_operate_zero_sessions(real_report: PipelineReport) -> None:
    """A5: los dos brazos del gate operan 0 sesiones y su serie declarada es toda cero."""
    null_arms = as_map(at(real_report.payload, "null_arms"))
    assert {key: as_int(value) for key, value in as_map(null_arms["traded"]).items()} == {
        ARM_OFICIAL: 0,
        ARM_ESCENARIO: 0,
    }
    assert null_arms["all_no_trade"] is True
    assert as_map(null_arms["declared_return_series_all_zero"]) == {
        ARM_OFICIAL: True,
        ARM_ESCENARIO: True,
    }
    for name in (ARM_OFICIAL, ARM_ESCENARIO):
        arm = real_report.arm(name)
        assert arm.run.traded == 0
        assert arm.run.no_trade == arm.n_test
        assert set(declared_series(arm.run)) == {0.0}
        metrics = metrics_of(real_report, name)
        assert as_float(metric_block(metrics, "sharpe")["estimate"]) == 0.0
        assert as_float(metric_block(metrics, "mean_return_pct")["estimate"]) == 0.0


@needs_store
def test_a5_hash_is_compared_with_no_trade_and_reasons_are_published(
    real_report: PipelineReport,
) -> None:
    """A5: los tres `run_sha256` se publican juntos, se comparan y el motivo se **mide**."""
    null_arms = as_map(at(real_report.payload, "null_arms"))
    hashes = cast("dict[str, str]", null_arms["run_sha256"])
    assert set(hashes) == {NO_TRADE, ARM_OFICIAL, ARM_ESCENARIO}
    assert len(set(hashes.values())) == 3
    assert as_map(null_arms["equals_no_trade"]) == {ARM_OFICIAL: False, ARM_ESCENARIO: False}
    reasons = cast("dict[str, str]", null_arms["decision_reason"])
    assert len({reasons[NO_TRADE], reasons[ARM_OFICIAL], reasons[ARM_ESCENARIO]}) == 3
    assert "status=no_recommendation_undecided" in reasons[ARM_OFICIAL]
    assert "run_sha256" in as_str(null_arms["measured_on"])
    assert "no coinciden" in as_str(null_arms["why"])
    for name in (ARM_OFICIAL, ARM_ESCENARIO):
        assert real_report.arm(name).run.run_sha256 == hashes[name]


@needs_store
def test_a5_blocker_codes_and_undecided_status(real_report: PipelineReport) -> None:
    """A5: recuento por `code` de bloqueo y `no_recommendation_undecided` en las `n_test`."""
    null_arms = as_map(at(real_report.payload, "null_arms"))
    codes = {key: as_int(value) for key, value in as_map(null_arms["blocker_code_counts"]).items()}
    assert codes["ev_neto_no_calculable"] >= 1
    assert codes["tier_no_autorizado"] >= 1
    assert codes["ev_neto_no_calculable"] == codes["tier_no_autorizado"]
    statuses = cast("dict[str, dict[str, int]]", null_arms["status_counts"])
    assert statuses[ARM_OFICIAL] == {"no_recommendation_undecided": len(real_report.test_sessions)}
    assert statuses[ARM_ESCENARIO]
    assert all(
        real_report.arm(ARM_OFICIAL).outputs[session].status == "no_recommendation_undecided"
        for session in real_report.test_sessions
    )


# ─────────────────────────────────────────────────────────────────────────────
# A6 · La regla declarada del brazo de coste declarado
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a6_declared_rule_is_published_literal(real_report: PipelineReport) -> None:
    """A6: el brazo publica su regla literal, con las nueve reglas de sesion."""
    rule = as_str(at(real_report.payload, "arms", ARM_COSTE_DECLARADO, "declared_rule"))
    for token in SESSION_RULES:
        assert token in rule
    for token in (
        "ev_declared_pct > ev_threshold_pct",
        f"target_pct >= {TARGET_MIN_COST_MULTIPLE} x c_declared_pct",
        "authorized_tiers",
        "DECISION_THRESHOLD",
        "notional = capital_usd x risk_per_trade_pct / stop_pct",
        "entry_px = open",
    ):
        assert token in rule


def _declared_tier(output: GateOutput, params: GateParameters) -> str:
    """El tier A/B/C re-derivado sobre el EV declarado, recalculado en el test (A6, #98).

    La probabilidad a favor es la de la direccion del **decididor** (`p` si `p >=
    DECISION_THRESHOLD`, `1 - p` si no), no la de `output.direction` (que en este brazo es
    `nothing` en las 500 sesiones).
    """
    ev_declared = output.ev_declared_pct
    if ev_declared is None:
        return "C"
    favourable = output.prob_up_calibrated
    if favourable < DECISION_THRESHOLD:
        favourable = 1.0 - favourable
    if ev_declared > cast("Decimal", params.tier_a_cost_multiple) * output.cost_pct and Decimal(
        str(favourable)
    ) > cast("Decimal", params.tier_a_min_probability):
        return "A"
    if ev_declared > cast("Decimal", params.tier_b_cost_multiple) * output.cost_pct:
        return "B"
    return "C"


def test_favourable_probability_follows_the_decider_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A3/#98: `_favourable_probability` cita la constante del gate, no `max(p, 1-p)`.

    Con el umbral declarado (`0.5`) `p = 0.4` es corto ⇒ `0.6`; con el umbral parcheado a `0.3`
    la **misma** `p = 0.4` es larga ⇒ `0.4` (el maximo seria `0.6`, que es lo que discrimina).
    """
    favourable = pipeline_report._favourable_probability  # pyright: ignore[reportPrivateUsage]
    assert favourable(_gate_output(probability=0.4)) == Decimal("0.6")
    monkeypatch.setattr(pipeline_report, "GATE_DECISION_THRESHOLD", 0.3)
    assert favourable(_gate_output(probability=0.4)) == Decimal("0.4") != Decimal("0.6")
    assert favourable(_gate_output(probability=0.2)) == Decimal("0.8")
    assert favourable(_gate_output(probability=0.3)) == Decimal("0.3")
    assert favourable(_gate_output(probability=0.1)) == Decimal("0.9")


def test_favourable_ast_does_not_read_the_gate_direction() -> None:
    """A3/#98: el AST de `_favourable_probability` no lee `output.direction`.

    Los unicos nombres que usa son `prob_up_calibrated` (la probabilidad) y
    `GATE_DECISION_THRESHOLD` (la constante del gate): la direccion la decide el decididor.
    """
    function = next(
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "_favourable_probability"
    )
    attributes = {node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)}
    assert "direction" not in attributes
    assert "prob_up_calibrated" in attributes
    names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    assert "GATE_DECISION_THRESHOLD" in names


@needs_store
def test_a6_arm_operates_exactly_by_the_declared_rule(real_report: PipelineReport) -> None:
    """A6: la regla re-derivada sobre las salidas del gate coincide sesion a sesion."""
    arm = real_report.arm(ARM_COSTE_DECLARADO)
    params = arm.params
    threshold = cast("Decimal", params.ev_threshold_pct)
    authorized = cast("tuple[str, ...]", params.authorized_tiers)
    trades = 0
    for outcome in sessions_of(arm.run):
        output = arm.outputs[outcome.session]
        blocked = any(entry["rule"] in SESSION_RULES for entry in output.blockers)
        ev_declared = output.ev_declared_pct
        minimum = TARGET_MIN_COST_MULTIPLE * output.cost_pct
        expected = (
            not blocked
            and ev_declared is not None
            and ev_declared > threshold
            and output.target_pct is not None
            and output.target_pct >= minimum
            and _declared_tier(output, params) in authorized
        )
        assert (outcome.status == STATUS_TRADED) == expected, outcome.session.isoformat()
        if not expected:
            continue
        trades += 1
        decision = outcome.decision
        assert decision is not None
        expected_direction = "long" if output.prob_up_calibrated >= DECISION_THRESHOLD else "short"
        assert str(decision.direction) == expected_direction
        stop_px = cast("float", decision.stop_px)
        target_px = cast("float", decision.target_px)
        entry = cast("float", outcome.entry_px)
        if expected_direction == "long":
            assert stop_px < entry < target_px
        else:
            assert target_px < entry < stop_px
        assert outcome.notional_usd is not None
        assert outcome.notional_usd == (
            Decimal("10000") * cast("Decimal", params.risk_per_trade_pct) / output.stop_pct
        ).quantize(Decimal("0.01"))
        assert "arm=coste_declarado" in decision.reason
        assert "basis=declared_cost" in decision.reason
    assert trades == arm.run.traded


@needs_store
def test_a6_disagreements_and_not_a_validation(real_report: PipelineReport) -> None:
    """A6: publica `is_validation = false` y cuantas sesiones discrepa de los brazos del gate."""
    arm = row_block(real_report, ARM_COSTE_DECLARADO)
    assert arm["is_validation"] is False
    assert arm["basis"] == BASIS_DECLARED_COST
    assert as_int(arm["mismatches_with_other_arms"]) == as_int(arm["traded"])
    assert "coinciden" in as_str(as_map(at(real_report.payload, "null_arms"))["why"])
    rejections = {key: as_int(value) for key, value in as_map(arm["rejections"]).items()}
    assert sum(rejections.values()) + as_int(arm["traded"]) == as_int(arm["n_test"])
    assert set(rejections) <= {
        "gate_session_rule",
        "ev_declared_not_above_threshold",
        "target_below_cost_multiple",
        "tier_not_authorized",
    }
    for name in (ARM_OFICIAL, ARM_ESCENARIO):
        assert as_int(row_block(real_report, name)["mismatches_with_other_arms"]) == as_int(
            arm["traded"]
        )


# ─────────────────────────────────────────────────────────────────────────────
# A7 · La tabla unica: seis baselines y tres listones
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a7_table_has_the_six_baselines_and_the_three_listones(
    real_report: PipelineReport,
) -> None:
    """A7: nueve filas con las seis `BASELINE_IDS` y los tres listones."""
    rows = as_objects(at(real_report.payload, "table", "rows"))
    assert [row["row"] for row in rows] == [*BASELINE_IDS, "liston_a", "liston_b", "liston_c"]
    assert {row["kind"] for row in rows} == {"baseline", "liston_a", "liston_b", "liston_c"}
    assert len(rows) == 9
    for row in rows:
        assert as_str(row["provenance"]).strip()
        assert isinstance(row["is_invertible"], bool)
        assert row["basis"] == BASIS_DECLARED_COST
        assert row["is_validation"] is False


@needs_store
def test_a7_liston_a_is_the_always_long_row(real_report: PipelineReport) -> None:
    """A7: el liston A **es** la fila `always_long`: mismo hash, misma serie."""
    always_long = real_report.row("always_long")
    liston_a = real_report.row("liston_a")
    assert liston_a.kind == "liston_a"
    assert liston_a.is_invertible is True
    assert liston_a.run_sha256 == always_long.run_sha256
    assert liston_a.series_pct == always_long.series_pct
    assert liston_a.n_test == always_long.n_test


@needs_store
def test_a7_liston_b_charges_the_declared_financing(real_report: PipelineReport) -> None:
    """A7: el liston B es el C menos la financiacion por noche **importada** de la tabla."""
    carry = float(declared_cost_model().carry_long_pct_per_night)
    assert carry == 0.0182
    liston_b = real_report.row("liston_b")
    liston_c = real_report.row("liston_c")
    assert len(liston_b.series_pct) == len(liston_c.series_pct)
    for value_b, value_c in zip(liston_b.series_pct, liston_c.series_pct, strict=True):
        assert value_b == pytest.approx(value_c - carry)
    alpha = metric_block(metrics_of(real_report, "liston_b"), "alpha_pct")
    assert as_float(alpha["estimate"]) == pytest.approx(-carry, abs=1e-9)
    assert "0.0182" not in SOURCE


@needs_store
def test_a7_liston_c_is_the_reference_and_not_a_baseline(real_report: PipelineReport) -> None:
    """A7/A13: C es el indice puro, no invertible, y **no** cuenta entre los seis baselines."""
    liston_c = real_report.row("liston_c")
    assert liston_c.kind == "liston_c"
    assert liston_c.is_invertible is False
    assert liston_c.run_sha256 is None
    assert liston_c.row not in BASELINE_IDS
    metrics = metrics_of(real_report, "liston_c")
    assert as_float(metric_block(metrics, "beta")["estimate"]) == pytest.approx(1.0)
    assert as_float(metric_block(metrics, "alpha_pct")["estimate"]) == pytest.approx(0.0, abs=1e-12)
    assert at(real_report.payload, "series_id") == SERIES_ID


@needs_store
def test_a7_series_is_derived_in_coherent_units(real_report: PipelineReport) -> None:
    """A10: la serie de la tabla sale de `100 x gross_pct - c_declared_pct`, no del motor."""
    row = real_report.row("always_long")
    run = next(
        outcome.run for outcome in real_report.baselines if outcome.baseline == "always_long"
    )
    declared = list(declared_series(run))
    assert list(row.series_pct) == pytest.approx(declared)
    mixed: list[float] = []
    for outcome in sessions_of(run):
        assert outcome.gross_pct is not None and outcome.cost is not None
        mixed.append(outcome.gross_pct - float(outcome.cost.c_declared_pct))
    assert mixed[0] != pytest.approx(declared[0])


# ─────────────────────────────────────────────────────────────────────────────
# A8 · Intervalos bootstrap con etiqueta
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a8_every_metric_carries_its_interval_label(real_report: PipelineReport) -> None:
    """A8: las diez metricas de cada fila y de cada brazo llevan su etiqueta completa.

    No se exige que la estimacion caiga dentro del intervalo: el remuestreo percentil de un
    estadistico extremo (el drawdown maximo) no tiene por que contener el valor de la muestra
    original. Lo que se exige es que el intervalo exista, este ordenado y declare su base.
    """
    for name in (*BASELINE_IDS, "liston_a", "liston_b", "liston_c", *ARM_NAMES):
        metrics = metrics_of(real_report, name)
        assert list(cast("list[str]", metrics["metric_names"])) == list(METRIC_NAMES)
        series = series_of(real_report, name)
        varied = any(value != 0.0 for value in series)
        for metric in METRIC_NAMES:
            block = metric_block(metrics, metric)
            if metric in NO_INTERVAL_METRICS:
                assert block["lower"] is None and block["upper"] is None
                assert as_str(block["reason"]).strip()
                continue
            if block["estimate"] is None:
                # `hit_rate_per_trade` sin operaciones (#92): null, nunca 0 ni `[0, 0]`.
                assert metric == "hit_rate_per_trade"
                assert block["lower"] is None and block["upper"] is None
                assert as_int(block["n"]) == 0
                assert as_str(block["reason"]).strip()
                continue
            lower = as_float(block["lower"])
            upper = as_float(block["upper"])
            assert lower <= upper
            if varied and metric in {"mean_return_pct", "hit_rate", "sharpe"}:
                assert lower < upper
        assert as_str(metrics["bootstrap_note"]).strip()
    del real_report


@needs_store
def test_a8_seeds_are_derived_and_validated(real_report: PipelineReport) -> None:
    """A8: semillas derivadas de la declarada, distintas por metrica y <= 2**32 - 1."""
    metrics = metrics_of(real_report, "always_long")
    seeds = [
        as_int(metric_block(metrics, name)["seed"])
        for name in METRIC_NAMES
        if name not in NO_INTERVAL_METRICS
    ]
    assert len(set(seeds)) == len(seeds)
    assert all(seed >= DEFAULT_BOOTSTRAP_SEED for seed in seeds)
    derive = pipeline_report._derived_seed  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(PipelineReportError):
        derive(offset=-1)
    assert derive(offset=2**32 - DEFAULT_BOOTSTRAP_SEED) == 0
    assert derive(offset=2**32 - DEFAULT_BOOTSTRAP_SEED - 1) == 2**32 - 1


@needs_store
def test_a8_same_series_gives_the_same_interval(real_report: PipelineReport) -> None:
    """A8: la misma serie con la misma metrica publica el mismo intervalo (cache de series)."""
    always_long = metrics_of(real_report, "always_long")
    liston_a = metrics_of(real_report, "liston_a")
    for name in METRIC_NAMES:
        assert metric_block(always_long, name)["lower"] == metric_block(liston_a, name)["lower"]
        assert metric_block(always_long, name)["upper"] == metric_block(liston_a, name)["upper"]


# ─────────────────────────────────────────────────────────────────────────────
# A9 · Base declarada y ninguna metrica neta
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a9_declared_cost_at_the_top_and_in_every_arm(real_report: PipelineReport) -> None:
    """A9: `basis: declared_cost` y `is_validation: false` arriba y en cada brazo."""
    assert at(real_report.payload, "basis") == BASIS_DECLARED_COST
    assert at(real_report.payload, "is_validation") is False
    limits = as_map(at(real_report.payload, "limits"))
    assert limits["basis"] == BASIS_DECLARED_COST
    assert limits["is_validation"] is False
    for name in ARM_NAMES:
        arm = row_block(real_report, name)
        assert arm["basis"] == BASIS_DECLARED_COST
        assert arm["is_validation"] is False


@needs_store
def test_a9_net_metrics_are_not_computable_and_never_fabricated(
    real_report: PipelineReport,
) -> None:
    """A9: `net_metrics: not_computable` con motivo y seguimientos, y ninguna metrica neta."""
    net = as_map(at(real_report.payload, "net_metrics"))
    assert net["state"] == "not_computable"
    assert as_str(net["reason"]).strip()
    assert net["follow_ups"] == ["#62", "#60"]
    assert not _calls("calculate_metrics")
    assert as_map(at(real_report.payload, "limits"))["slippage_state"] == "assumed"
    assert as_map(at(real_report.payload, "limits"))["slippage_is_measurement"] is False
    for name in (*ARM_NAMES, "always_long"):
        metrics = metrics_of(real_report, name)
        assert not [key for key in metrics if key.startswith("net_")]
    assert as_map(at(real_report.payload, "limits"))["financing_cut"] is None


# ─────────────────────────────────────────────────────────────────────────────
# A10 · #80 declarado, no heredado
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_ast_does_not_read_the_engine_declared_pnl() -> None:
    """A10: el AST del modulo no lee el atributo del P&L declarado del motor."""
    for node in ast.walk(TREE):
        if isinstance(node, ast.Attribute):
            assert node.attr != "pnl_declared_pct"


@needs_store
def test_a10_the_defect_narrative_is_gone(real_report: PipelineReport) -> None:
    """A5: el bloque `check` de #80, su seccion Markdown y su seguimiento desaparecen."""
    assert "check" not in real_report.payload
    issues = [as_str(entry["issue"]) for entry in as_objects(at(real_report.payload, "follow_ups"))]
    assert "#80" not in issues
    markdown = render_markdown(real_report)
    assert "Chequeo de unidades" not in markdown
    assert "#80" not in markdown
    # El AST sigue sin leer el P&L declarado del motor: la derivacion propia no cambia.
    for node in ast.walk(TREE):
        if isinstance(node, ast.Attribute):
            assert node.attr != "pnl_declared_pct"


# ─────────────────────────────────────────────────────────────────────────────
# A11 · Mapeo temporal, regla 13 y ninguna hora ET literal
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a11_temporal_mapping_is_declared_and_measured(real_report: PipelineReport) -> None:
    """A11: `today = session` y `as_of` del almacen de esa misma sesion, nunca del reloj."""
    assert set(TEMPORAL_MAPPING) == {
        "today",
        "as_of",
        "entry_px",
        "probability",
        "expected_move_pct",
        "features",
        "clock",
    }
    assert at(real_report.payload, "temporal_mapping") == TEMPORAL_MAPPING
    history = load_history(Store(REAL_DATA))
    instants = {
        cast("date", row["session"]): cast("datetime", row["as_of"])
        for row in history.daily.select("session", "as_of").iter_rows(named=True)
    }
    arm = real_report.arm(ARM_ESCENARIO)
    for session in (real_report.test_sessions[0], real_report.test_sessions[-1]):
        output = arm.outputs[session]
        assert output.today == session
        assert output.as_of == instants[session]
        assert output.as_of.date() == session
    assert at(real_report.payload, "model", "design_lag_sessions") == DESIGN_LAG_SESSIONS


@needs_store
def test_a11_rule_13_passes_in_every_test_session(real_report: PipelineReport) -> None:
    """A11: la regla 13 (frescura) sale `pass` en las `n_test` sesiones del escenario."""
    for name in (ARM_ESCENARIO, ARM_OFICIAL):
        rules = cast(
            "dict[str, dict[str, int]]",
            at(real_report.payload, "arms", name, "gate", "rule_outcomes"),
        )
        assert rules["13"] == {"pass": len(real_report.test_sessions)}
        assert real_report.arm(name).n_test == len(real_report.test_sessions)


def test_a11_no_literal_eastern_hour() -> None:
    """A11: el AST no contiene ninguna hora ET literal ni fechas cableadas."""
    for node in ast.walk(TREE):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (date, datetime)):
                pytest.fail("el modulo publica una fecha literal")
            if isinstance(node.value, str) and len(node.value) in (5, 8) and ":" in node.value:
                pytest.fail(f"literal horaria en el modulo: {node.value!r}")
            if isinstance(node.value, str):
                assert "16:00" not in node.value
                assert "09:30" not in node.value
    assert "time(" not in SOURCE


# ─────────────────────────────────────────────────────────────────────────────
# A12 · La regla 11 medida
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a12_rule_11_share_is_published_and_measured(real_report: PipelineReport) -> None:
    """A12: por brazo, fraccion, % y su intervalo, contra la banda declarada."""
    rule_11 = as_map(at(real_report.payload, "rule_11"))
    assert rule_11["band"] == [RULE_11_BAND[0], RULE_11_BAND[1]]
    arms = cast("dict[str, dict[str, object]]", rule_11["arms"])
    assert list(arms) == list(ARM_NAMES)
    for name in ARM_NAMES:
        block = arms[name]
        run = real_report.arm(name).run
        assert as_int(block["traded"]) == run.traded
        assert as_int(block["n_test"]) == run.traded + run.no_trade + run.skipped
        assert block["share_fraction"] == f"{run.traded}/{as_int(block['n_test'])}"
        share = as_float(block["share"])
        assert share == pytest.approx(run.traded / as_int(block["n_test"]))
        assert as_float(block["share_pct"]) == pytest.approx(100.0 * share)
        assert block["measured"] is True
        interval = as_map(block["share_interval"])
        assert interval["basis"] == BASIS_DECLARED_COST
        assert interval["n_bootstrap"] == DEFAULT_BOOTSTRAP_SAMPLES
        assert as_float(interval["lower"]) <= as_float(interval["estimate"])
        assert as_float(interval["upper"]) >= as_float(interval["estimate"])


@needs_store
def test_a12_verdict_is_measured_not_asserted(real_report: PipelineReport) -> None:
    """A12: el veredicto se deduce de la cuota medida y de la banda declarada."""
    arms = cast("dict[str, dict[str, object]]", at(real_report.payload, "rule_11", "arms"))
    for name, block in arms.items():
        share = as_float(block["share"])
        expected = (
            "above" if share > RULE_11_BAND[1] else "below" if share < RULE_11_BAND[0] else "inside"
        )
        assert block["verdict"] == expected, name
    assert arms[ARM_OFICIAL]["verdict"] == "below"
    assert arms[ARM_ESCENARIO]["verdict"] == "below"


# ─────────────────────────────────────────────────────────────────────────────
# A13 · Alpha y beta separados
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a13_beta_alpha_benchmark_and_excess_per_arm_and_liston(
    real_report: PipelineReport,
) -> None:
    """A13: beta, alfa, retorno del benchmark y retorno en exceso, por brazo y por liston."""
    for name in (*ARM_NAMES, "liston_a", "liston_b", "liston_c"):
        metrics = metrics_of(real_report, name)
        for metric in ("beta", "alpha_pct", "benchmark_return_pct", "excess_return_pct"):
            assert metric_block(metrics, metric)["estimate"] is not None, f"{name}.{metric}"
    benchmark = real_report.row("liston_c").series_pct
    for name in (*ARM_NAMES, "liston_a"):
        expected = beta_of(series_of(real_report, name), benchmark)
        published = as_float(metric_block(metrics_of(real_report, name), "beta")["estimate"])
        assert published == pytest.approx(expected, abs=1e-12)
    metrics = metrics_of(real_report, ARM_COSTE_DECLARADO)
    excess = as_float(metric_block(metrics, "excess_return_pct")["estimate"])
    mean = as_float(metric_block(metrics, "mean_return_pct")["estimate"])
    assert excess == pytest.approx(mean - math.fsum(benchmark) / len(benchmark))
    benchmark_return = as_float(
        metric_block(metrics_of(real_report, "liston_c"), "benchmark_return_pct")["estimate"]
    )
    expected_return = (math.prod(1.0 + value / 100.0 for value in benchmark) - 1.0) * 100.0
    assert benchmark_return == pytest.approx(expected_return, rel=1e-9)
    assert math.isfinite(benchmark_return)
    assert benchmark_return != 0.0


@needs_store
def test_a13_markdown_says_which_of_the_two_carries_the_result(
    real_report: PipelineReport,
) -> None:
    """A13: el `.md` dice si el resultado lo carga el alpha o el beta, con los numeros medidos."""
    attribution = as_map(at(real_report.payload, "arm_comparison", "attribution"))
    loader = as_str(attribution["loader"])
    declared = metrics_of(real_report, ARM_COSTE_DECLARADO)
    beta = as_float(metric_block(declared, "beta")["estimate"])
    alpha = as_float(metric_block(declared, "alpha_pct")["estimate"])
    benchmark_mean_pct = as_float(attribution["benchmark_mean_pct"])
    contribution = as_float(attribution["benchmark_contribution_pct"])
    # el cargador **no** se acepta como etiqueta libre: sale de los numeros publicados
    assert contribution == pytest.approx(beta * benchmark_mean_pct, abs=1e-12)
    assert loader == ("beta" if abs(contribution) > abs(alpha) else "alpha")
    statement = as_str(attribution["statement"])
    assert loader in statement
    markdown = render_markdown(real_report)
    assert statement in markdown
    assert SERIES_ID in markdown
    assert "Atribucion: alpha contra beta" in markdown


@needs_store
def test_a13_attribution_units_are_percent_and_close_the_jensen_identity(
    real_report: PipelineReport,
) -> None:
    """A13: la atribucion publica en % (una sola convencion) y cierra la identidad de Jensen.

    Con el error de unidades 100x esto falla: ``benchmark_mean_pct`` saldria 100x la media por
    sesion (``6,369679``) y la aportacion seria ``-0,267822`` en vez de ``-0,002678``, asi que
    ``alpha + beta x media`` no daria el ``mean_return_pct`` publicado.
    """
    attribution = as_map(at(real_report.payload, "arm_comparison", "attribution"))
    listed_mean_pct = as_float(
        metric_block(metrics_of(real_report, "liston_c"), "mean_return_pct")["estimate"]
    )
    benchmark_mean_pct = as_float(attribution["benchmark_mean_pct"])
    # misma unidad y misma cifra que el informe ya publica para el liston C
    assert benchmark_mean_pct == pytest.approx(listed_mean_pct, abs=1e-12)
    declared = metrics_of(real_report, ARM_COSTE_DECLARADO)
    beta = as_float(metric_block(declared, "beta")["estimate"])
    alpha = as_float(metric_block(declared, "alpha_pct")["estimate"])
    mean = as_float(metric_block(declared, "mean_return_pct")["estimate"])
    contribution = as_float(attribution["benchmark_contribution_pct"])
    # la aportacion es ``beta x media``, **sin** volver a multiplicar por 100
    assert contribution == pytest.approx(beta * benchmark_mean_pct, abs=1e-12)
    assert contribution != pytest.approx(beta * benchmark_mean_pct * 100.0, abs=1e-9)
    # la identidad de Jensen cierra con los numeros publicados
    assert alpha + contribution == pytest.approx(mean, abs=1e-12)
    # y el cargador sale de la clasificacion corregida (no de una etiqueta fija)
    loader = as_str(attribution["loader"])
    assert loader == ("beta" if abs(contribution) > abs(alpha) else "alpha")
    # la etiqueta medida es la que aparece en la frase, sin fijar cual de las dos gana
    assert loader in as_str(attribution["statement"])


@pytest.mark.parametrize(
    ("metrics", "loader", "contribution", "alpha"),
    [
        ({"beta": {"estimate": 1.5}, "alpha_pct": {"estimate": 0.5}}, "beta", 3.0, 0.5),
        ({"beta": {"estimate": 0.5}, "alpha_pct": {"estimate": 1.5}}, "alpha", 1.0, 1.5),
        ({"beta": {"estimate": None}, "alpha_pct": {"estimate": 0.5}}, "no_atribuible", None, 0.5),
        ({"beta": {"estimate": 1.5}}, "no_atribuible", 3.0, None),
    ],
)
def test_a13_attribution_classifies_from_the_measured_numbers(
    metrics: dict[str, dict[str, float | None]],
    loader: str,
    contribution: float | None,
    alpha: float | None,
) -> None:
    """A13: el cargador sale de los numeros medidos (una sola convencion, %) y `no_atribuible`.

    El benchmark va en % (``1, 2, 3`` => media ``2``): con el error 100x la aportacion saldria
    ``200 x beta`` y el caso ``no_atribuible`` (sin beta o sin alfa) no tendria rama propia.
    """
    payload = pipeline_report._comparison_payload(  # pyright: ignore[reportPrivateUsage]
        rows=(),
        metrics_by_name={ARM_COSTE_DECLARADO: metrics},
        benchmark=(1.0, 2.0, 3.0),
    )
    attribution = as_map(payload["attribution"])
    assert as_float(attribution["benchmark_mean_pct"]) == 2.0
    assert attribution["benchmark_contribution_pct"] == contribution
    assert attribution["alpha_pct"] == alpha
    assert as_str(attribution["loader"]) == loader
    statement = as_str(attribution["statement"])
    if loader == "no_atribuible":
        assert "no hay atribucion" in statement
    else:
        assert loader in statement


@needs_store
def test_a13_series_are_aligned_session_by_session(real_report: PipelineReport) -> None:
    """A13: todas las series de riesgo miden lo mismo que las sesiones de *test*."""
    assert len(real_report.row("liston_c").series_pct) == len(real_report.test_sessions)
    for name in (*ARM_NAMES, "liston_a", "liston_b"):
        assert len(series_of(real_report, name)) == len(real_report.test_sessions)


# ─────────────────────────────────────────────────────────────────────────────
# A14 · Datos de prueba, segunda pasada y errores tipados
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a14_tests_write_only_under_tmp_path(real_report: PipelineReport) -> None:
    """A14: la corrida escribe bajo `tmp_path` y **solo** los dos ficheros del informe."""
    root = real_report.reports_dir
    assert root.is_dir()
    assert "pytest" in str(root)
    assert len([path for path in root.rglob("*") if path.is_file()]) == 2
    source = TEST_PATH.read_text(encoding="utf-8")
    assert "tmp_path" in source
    assert "PYTHONHASHSEED" in source
    assert "Store(REAL_DATA)" in source


def test_a14_cli_in_process_resolves_dirs_and_exits_zero(tmp_path: Path) -> None:
    """A14: `main` **en proceso** resuelve `--data-root`/`--reports-dir`, escribe y sale 0.

    Los dos procesos frescos de A2/A3 no registran estas lineas para `coverage`, asi que el punto
    de entrada se ejerce aqui: resolver el directorio de informes, delegar en `analyse` y publicar
    la tabla. Se usa `_patched_fast` (el doble deterministico de #15) para no repetir el remuestreo
    de la corrida de sesion; ninguna decision ni clave del payload cambia.
    """
    reports = tmp_path / "informes"
    with _patched_fast():
        code = main(
            [
                "--data-root",
                str(REAL_DATA),
                "--reports-dir",
                str(reports),
                "--as-of",
                NOW.isoformat(),
            ]
        )
    assert code == 0
    assert (reports / f"{STEM}.json").is_file()
    assert (reports / f"{STEM}.md").is_file()


def test_a14_cli_default_reports_dir_and_typed_exit_codes(tmp_path: Path) -> None:
    """A14: sin `--reports-dir` cuelga de `--data-root`; sin `--as-of` o con un almacen que
    falla, sale 2 y **no** escribe nada.
    """
    data_root = tmp_path / "almacen"
    # la salida tipada por `--as-of` ausente (no se escribe nada)
    assert main(["--data-root", str(data_root)]) == 2
    assert not (data_root / "derived" / "reports").exists()
    # `--as-of` presente pero un almacen vacio: `analyse` falla, sale 2 y no escribe
    empty = tmp_path / "vacio"
    empty.mkdir()
    assert main(["--data-root", str(empty), "--as-of", NOW.isoformat()]) == 2
    assert not (empty / "derived" / "reports").exists()


@needs_store
def test_a14_payload_is_json_pure_and_rehashable(real_report: PipelineReport) -> None:
    """A2/A3: el payload publicado se puede rehashear y es de tipos JSON puros."""
    assert (
        pipeline_report._digest(real_report.payload)  # pyright: ignore[reportPrivateUsage]
        == real_report.report_sha256
    )
    text = real_report.json_text()
    parsed = as_map(json.loads(text))
    assert parsed["report_sha256"] == real_report.report_sha256
    assert "NaN" not in text
    assert "Infinity" not in text
    assert text.endswith("\n")


@needs_store
def test_a14_errors_are_typed(real_report: PipelineReport) -> None:
    """A4: la traduccion del plan exige que el diseno y el universo sean la misma secuencia."""
    with pytest.raises(AlignmentError):
        pipeline_report._split_assignments(  # pyright: ignore[reportPrivateUsage]
            real_report.split_plan, n_design=1
        )
    with pytest.raises(PipelineReportError):
        real_report.arm("inexistente")
    with pytest.raises(PipelineReportError):
        real_report.row("inexistente")


# ─────────────────────────────────────────────────────────────────────────────
# A14 · Ramas de las guardas tipadas (unidad directa, sin corrida completa)
# ─────────────────────────────────────────────────────────────────────────────
# Las ramas que faltan son las guardas defensivas y los rechazos tipados del modulo: la corrida
# real solo recorre el camino feliz de cada una, asi que se ejercen **directamente** sobre el
# helper, con la entrada invalida que debe producir el error declarado. Todo es determinista y
# sin almacen: la corrida cara de la sesion no se repite. Los alias de abajo llevan el `ignore`
# de `pyright` una sola vez por helper en vez de una vez por llamada.
_as_utc = pipeline_report._as_utc  # pyright: ignore[reportPrivateUsage]
_plain = pipeline_report._plain  # pyright: ignore[reportPrivateUsage]
_session_instants = pipeline_report._session_instants  # pyright: ignore[reportPrivateUsage]
_daily_by_session = pipeline_report._daily_by_session  # pyright: ignore[reportPrivateUsage]
_expected_move_pct = pipeline_report._expected_move_pct  # pyright: ignore[reportPrivateUsage]
_derived_seed = pipeline_report._derived_seed  # pyright: ignore[reportPrivateUsage]
_declared_tier = pipeline_report._declared_tier  # pyright: ignore[reportPrivateUsage]
_barrier_prices = pipeline_report._barrier_prices  # pyright: ignore[reportPrivateUsage]
_declared_cost_decision = pipeline_report._declared_cost_decision  # pyright: ignore[reportPrivateUsage]
_bundle = pipeline_report._bundle  # pyright: ignore[reportPrivateUsage]
_deciders = pipeline_report._deciders  # pyright: ignore[reportPrivateUsage]
_declared_return_pct = pipeline_report._declared_return_pct  # pyright: ignore[reportPrivateUsage]
_series_of_run = pipeline_report._series_of_run  # pyright: ignore[reportPrivateUsage]
_declared_series_payload = pipeline_report._declared_series_payload  # pyright: ignore[reportPrivateUsage]
_close_to_close_pct = pipeline_report._close_to_close_pct  # pyright: ignore[reportPrivateUsage]
_beta = pipeline_report._beta  # pyright: ignore[reportPrivateUsage]
_alpha = pipeline_report._alpha  # pyright: ignore[reportPrivateUsage]
_metric_statistic = pipeline_report._metric_statistic  # pyright: ignore[reportPrivateUsage]
_add_excess_metric = pipeline_report._add_excess_metric  # pyright: ignore[reportPrivateUsage]
_counts_by = pipeline_report._counts_by  # pyright: ignore[reportPrivateUsage]
_rule_11_payload = pipeline_report._rule_11_payload  # pyright: ignore[reportPrivateUsage]

#: Sesion de las piezas de prueba: el modulo nunca lee el reloj, tampoco aqui.
SESSION: Final[date] = date(2026, 9, 23)
NEXT_SESSION: Final[date] = date(2026, 9, 24)


def declared_cost_breakdown() -> CostBreakdown:
    """El `CostBreakdown` declarado de #8/#11 con el nocional del informe: la misma pieza que
    `analyse`, para que las guardas bajo prueba vean las unidades reales.
    """
    return cost_breakdown(
        model=declared_cost_model(),
        slippage=declared_slippage_assumption(),
        notional_usd=NOTIONAL_USD,
        side=Side.LONG,
        nights=0,
    )


def _gate_output(
    *,
    ev_declared_pct: Decimal | None = None,
    target_pct: Decimal | None = None,
    cost_pct: Decimal = Decimal("0.0042"),
    probability: float = 0.5,
    direction: Direction | None = None,
    blockers: tuple[dict[str, str], ...] = (),
) -> GateOutput:
    """Una `GateOutput` minima pero **valida**: solo los campos que leen las guardas bajo prueba.

    `gate_sha256` va con el prefijo y sin digest (no se publica en el informe por esta via), que
    es tambien la politica de `detect-secrets` de #19/#20.
    """
    return GateOutput(
        session=SESSION,
        as_of=NOW,
        today=SESSION,
        status=GateStatus.RECOMMENDATION,
        direction=direction,
        tier=cast("Tier", TIER_C),
        prob_up_calibrated=probability,
        expected_move_pct=Decimal("2"),
        expected_move_basis="garch_forecast",
        cost_pct=cost_pct,
        slippage_state="assumed",
        ev_declared_pct=ev_declared_pct,
        stop_pct=Decimal("1"),
        target_pct=target_pct,
        trades_today=0,
        observation_sessions_remaining=0,
        is_fomc_session=False,
        is_half_session=False,
        fomc_dates_count=0,
        params={},
        blockers=blockers,
        gate_sha256=GATE_HASH_PREFIX,
    )


def _outcome(
    *,
    session: date,
    status: str,
    gross_pct: float | None = None,
    cost: CostBreakdown | None = None,
) -> SessionOutcome:
    """Una `SessionOutcome` de *test* con lo minimo que leen las guardas de series."""
    return SessionOutcome(
        fold_index=0,
        session=session,
        session_index=0,
        status=status,
        reason=None,
        skip_reason=STATUS_SKIPPED if status == STATUS_SKIPPED else None,
        gap_px=None,
        decision=None,
        entry_session=None,
        exit_session=None,
        entry_px=None,
        exit_px=None,
        exit_reason=None,
        exit_bar_index=None,
        notional_usd=None,
        gross_pct=gross_pct,
        pnl_declared_pct=None,
        pnl_net_pct=None,
        pnl_net_reason=None,
        cost=cost,
    )


def _run(
    *,
    folded: tuple[SessionOutcome, ...] = (),
    traded: int = 0,
    no_trade: int = 0,
    skipped: int = 0,
) -> BacktestRun:
    """Una `BacktestRun` minima: los recuentos que leen las guardas y las sesiones que hay."""
    folds = (
        (
            FoldOutcome(
                index=0,
                test_start=0,
                test_stop=len(folded),
                sessions=folded,
                traded=traded,
                no_trade=no_trade,
                skipped=skipped,
            ),
        )
        if folded
        else ()
    )
    return BacktestRun(
        folds=folds,
        plan_sha256=f"{GATE_HASH_PREFIX}plan",
        purge_total=0,
        embargo_total=0,
        embargo_in_train_total=0,
        exclusions_are_no_op=True,
        uncovered=(),
        not_in_any_test=0,
        n_sessions=len(folded),
        traded=traded,
        no_trade=no_trade,
        skipped=skipped,
        run_sha256=f"{GATE_HASH_PREFIX}run",
        report={},
    )


def _arm(name: str, run: BacktestRun) -> pipeline_report.ArmRun:
    """Un `ArmRun` minimo con los recuentos de esa corrida."""
    return pipeline_report.ArmRun(
        name=name,
        run=run,
        params=GateParameters(),
        outputs={},
        ledger=pipeline_report.ArmLedger(),
    )


def test_a14_guard_as_utc_normalises_naive_and_aware() -> None:
    """A2: `_as_utc` (l.524-525) rellena la zona de un `datetime` naive y deja el consciente."""
    naive = datetime(2026, 9, 23, 22, 0)
    normalised = _as_utc(naive)
    assert normalised == naive.replace(tzinfo=UTC)
    assert normalised.tzinfo is UTC
    aware = datetime(2026, 9, 23, 22, 0, tzinfo=UTC)
    assert _as_utc(aware) is aware


def test_a14_guard_plain_translates_only_json_types() -> None:
    """A2: `_plain` (l.541-554) rechaza `nan`/`inf` y los tipos que el JSON no admite."""
    assert _plain(True, where="x") is True
    assert _plain(Decimal("1.500"), where="x") == "1.500"
    assert (
        _plain(datetime(2026, 9, 23, 22, 0, tzinfo=UTC), where="x") == "2026-09-23T22:00:00+00:00"
    )
    assert _plain({"a": [1, None]}, where="x") == {"a": [1, None]}
    with pytest.raises(PipelineReportError, match="nan"):
        _plain(math.nan, where="x")
    with pytest.raises(PipelineReportError, match="nan"):
        _plain(math.inf, where="x")
    with pytest.raises(PipelineReportError, match="tipos JSON"):
        _plain(object(), where="x")


def test_a14_guard_warehouse_frames_skip_null_rows() -> None:
    """A4/A11: las lecturas del diario (l.632-633, 643-644) saltan las filas sin sesion."""
    daily = pl.DataFrame(
        {
            "session": [SESSION, None, NEXT_SESSION],
            "as_of": [
                None,
                datetime(2026, 9, 23, 20, 0, tzinfo=UTC),
                datetime(2026, 9, 24, 20, 0, tzinfo=UTC),
            ],
        }
    )
    assert _session_instants(daily) == {NEXT_SESSION: datetime(2026, 9, 24, 20, 0, tzinfo=UTC)}
    assert _daily_by_session(pl.DataFrame({"session": [SESSION, None]})) == {
        SESSION: {"session": SESSION}
    }


def test_a14_guard_expected_move_needs_the_garch_column() -> None:
    """A11: `_expected_move_pct` (l.656-669) exige la columna y salta los valores sin sentido."""
    with pytest.raises(PipelineReportError, match="garch_forecast"):
        _expected_move_pct(pl.DataFrame({"session": [SESSION]}))
    matrix = pl.DataFrame(
        {
            "session": [SESSION, NEXT_SESSION, date(2026, 9, 25), date(2026, 9, 28)],
            "garch_forecast": [0.25, None, math.nan, -1.0],
        }
    )
    assert _expected_move_pct(matrix) == {SESSION: Decimal("50.0")}


def test_a14_guard_derived_seed_invariant_and_dead_branch() -> None:
    """A8: `_derived_seed` (l.571-587) valida el offset y **siempre** cae en `[0, 2**32)`.

    La guarda `not 0 <= seed <= 2**32 - 1` (l.582-583) es **inalcanzable**: `seed` es el resto
    modulo `2**32` de un entero no negativo y el unico camino que podria salirse (offset
    negativo) ya se rechaza antes en la l.577. Se deja sin cubrir **a proposito** y se documenta
    aqui: la comprobacion no es codigo muerto (defiende el contrato de `RandomState`), pero
    ninguna entrada puede activarla, asi que no se le pone `pragma: no cover` disfrazado.
    """
    for offset in (0, 1, 42, 2**32 - 2, 2**53):
        seed = _derived_seed(offset=offset)
        assert 0 <= seed <= 2**32 - 1
    assert _derived_seed(offset=0) == DEFAULT_BOOTSTRAP_SEED
    assert _derived_seed(offset=1) == DEFAULT_BOOTSTRAP_SEED + 1
    with pytest.raises(PipelineReportError, match="desplazamiento"):
        _derived_seed(offset=-1)


def test_a14_guard_barrier_prices_need_entry_and_target() -> None:
    """A6/#64: `_barrier_prices` (l.795-801) no inventa barreras sin `open` ni geometria rara."""
    none_left = _barrier_prices(
        direction=Direction.LONG,
        entry_px=None,
        stop_pct=Decimal("1"),
        target_pct=Decimal("2"),
    )
    assert none_left == (None, None)
    none_target = _barrier_prices(
        direction=Direction.LONG,
        entry_px=100.0,
        stop_pct=Decimal("1"),
        target_pct=None,
    )
    assert none_target == (None, None)
    stop_px, target_px = _barrier_prices(
        direction=Direction.LONG,
        entry_px=100.0,
        stop_pct=Decimal("1"),
        target_pct=Decimal("2"),
    )
    assert (stop_px, target_px) == pytest.approx((99.0, 102.0))
    assert stop_px is not None and target_px is not None
    assert stop_px < 100.0 < target_px
    stop_px, target_px = _barrier_prices(
        direction=Direction.SHORT,
        entry_px=100.0,
        stop_pct=Decimal("1"),
        target_pct=Decimal("2"),
    )
    assert (stop_px, target_px) == pytest.approx((101.0, 98.0))
    assert stop_px is not None and target_px is not None
    assert target_px < 100.0 < stop_px


def test_a14_guard_declared_tier_derives_c_b_and_a() -> None:
    """A6: `_declared_tier` (l.763-772) re-deriva el tier sobre el EV declarado, no el neto."""
    params = scenario_parameters(cost_pct=Decimal("0.0042"))
    assert _declared_tier(_gate_output(ev_declared_pct=None), params) == TIER_C
    assert _declared_tier(_gate_output(ev_declared_pct=Decimal("0.005")), params) == TIER_C
    assert _declared_tier(_gate_output(ev_declared_pct=Decimal("0.01")), params) == TIER_B
    tier_a = _declared_tier(
        _gate_output(
            ev_declared_pct=Decimal("0.02"),
            probability=0.60,
            direction=Direction.LONG,
        ),
        params,
    )
    assert tier_a == TIER_A
    # #98: la direccion del gate (`nothing`) ya no decide: `p = 0.60` es tier A **sin** direccion
    # declarada (con la convencion vieja, `1 - p = 0.4` no llegaria a `0.58`), y `p = 0.50` es B.
    assert (
        _declared_tier(_gate_output(ev_declared_pct=Decimal("0.02"), probability=0.60), params)
        == TIER_A
    )
    assert (
        _declared_tier(_gate_output(ev_declared_pct=Decimal("0.02"), probability=0.50), params)
        == TIER_B
    )


def test_a14_guard_declared_cost_decision_rejects_below_threshold() -> None:
    """A6: la regla del brazo declarado (l.832-845) rechaza por EV y objetivo, sin inventar 0."""
    params = scenario_parameters(cost_pct=Decimal("0.0042"))
    ledger = pipeline_report.ArmLedger()
    decision = _declared_cost_decision(
        output=_gate_output(ev_declared_pct=None),
        params=params,
        capital_usd=NOTIONAL_USD,
        entry_px=100.0,
        ledger=ledger,
    )
    assert decision.direction is Direction.NOTHING
    assert "ev_declared_pct=null" in decision.reason
    assert ledger.rejections == {"ev_declared_not_above_threshold": 1}
    decision = _declared_cost_decision(
        output=_gate_output(ev_declared_pct=Decimal("0.0084")),
        params=params,
        capital_usd=NOTIONAL_USD,
        entry_px=100.0,
        ledger=ledger,
    )
    assert "ev_declared_pct=0.0084 <= ev_threshold_pct=0.0084" in decision.reason
    assert ledger.rejections == {"ev_declared_not_above_threshold": 2}
    target_ledger = pipeline_report.ArmLedger()
    decision = _declared_cost_decision(
        output=_gate_output(ev_declared_pct=Decimal("0.01"), target_pct=None),
        params=params,
        capital_usd=NOTIONAL_USD,
        entry_px=100.0,
        ledger=target_ledger,
    )
    assert "target_pct=null" in decision.reason
    assert target_ledger.rejections == {"target_below_cost_multiple": 1}
    decision = _declared_cost_decision(
        output=_gate_output(ev_declared_pct=Decimal("0.01"), target_pct=Decimal("0.001")),
        params=params,
        capital_usd=NOTIONAL_USD,
        entry_px=100.0,
        ledger=target_ledger,
    )
    assert "target_pct=0.001 < 0.0084" in decision.reason
    assert target_ledger.rejections == {"target_below_cost_multiple": 2}
    trade_ledger = pipeline_report.ArmLedger()
    decision = _declared_cost_decision(
        output=_gate_output(
            ev_declared_pct=Decimal("0.02"),
            target_pct=Decimal("0.02"),
            probability=0.60,
            direction=Direction.LONG,
        ),
        params=params,
        capital_usd=NOTIONAL_USD,
        entry_px=100.0,
        ledger=trade_ledger,
    )
    assert decision.direction is Direction.LONG
    assert decision.notional_usd == NOTIONAL_USD
    assert (decision.stop_px, decision.target_px) == pytest.approx((99.0, 100.02))
    assert trade_ledger.traded == 1


def test_a14_guard_bundle_and_missing_move_decider() -> None:
    """A4/A11: `_bundle` (l.981-982) exige la carga declarada; sin ella el decididor la declara.

    La sesion que viaja con `context = None` (sin `garch_forecast`, l.1012-1013) **no** se
    rompe: se declara como no operada y se cuenta en `without_expected_move`, nunca se inventa un
    movimiento por defecto.
    """
    view = SessionView(session=SESSION, open_px=100.0, gap_px=None, context=None)
    with pytest.raises(PipelineReportError, match="SessionBundle"):
        _bundle(view)
    bundle = pipeline_report.SessionBundle(
        probability=0.5, oficial=_gate_output(), escenario=_gate_output()
    )
    loaded = SessionView(session=SESSION, open_px=100.0, gap_px=None, context=bundle)
    assert _bundle(loaded) is bundle
    ledger = pipeline_report.ArmLedger()
    decide = _deciders(
        name=ARM_OFICIAL,
        params=GateParameters(),
        ledger=ledger,
        capital_usd=NOTIONAL_USD,
        n_folds=1,
    )[0]
    decision = decide(view)
    assert decision.direction is Direction.NOTHING
    assert "arm_missing_expected_move" in decision.reason
    assert "garch_forecast" in decision.reason
    assert ledger.without_expected_move == 1


def test_a14_guard_declared_return_and_series_skip() -> None:
    """A10: `_declared_return_pct` (l.1118-1119) exige operacion y coste; la serie (l.1134-1135)
    salta las sesiones `skipped` sin meterlas como cero.
    """
    cost = declared_cost_breakdown()
    with pytest.raises(PipelineReportError, match="retorno declarado"):
        _declared_return_pct(_outcome(session=SESSION, status=STATUS_TRADED))
    traded = _outcome(session=SESSION, status=STATUS_TRADED, gross_pct=0.01, cost=cost)
    skipped = _outcome(session=NEXT_SESSION, status=STATUS_SKIPPED)
    expected = 100.0 * 0.01 - float(cost.c_declared_pct)
    assert _declared_return_pct(traded) == pytest.approx(expected)
    run = _run(folded=(traded, skipped), traded=1, skipped=1)
    assert _series_of_run(run) == (pytest.approx(expected),)


def test_a14_guard_close_to_close_needs_the_daily_row() -> None:
    """A7: `_close_to_close_pct` (l.1153-1154) exige cierre y cierre previo en el diario."""
    with pytest.raises(PipelineReportError, match="cierre"):
        _close_to_close_pct(SESSION, {})
    with pytest.raises(PipelineReportError, match="cierre"):
        _close_to_close_pct(SESSION, {SESSION: {"close": 101.0, "prev_close": None}})
    assert _close_to_close_pct(SESSION, {SESSION: {"close": 101.0, "prev_close": 100.0}}) == (
        pytest.approx(1.0)
    )


def test_a14_guard_beta_alpha_of_a_flat_benchmark() -> None:
    """A13: con varianza 0 del benchmark no hay beta ni alfa: `nan`, nunca una cifra inventada."""
    flat = (0.5, 0.5, 0.5)
    assert math.isnan(_beta((0.01, -0.02, 0.03), flat))
    assert math.isnan(_beta((), ()))
    assert math.isnan(_alpha((0.01, -0.02, 0.03), flat))
    assert _beta((0.01, 0.02), (0.0, 0.02)) == pytest.approx(0.5)
    assert _alpha((0.01, 0.02), (0.0, 0.02)) == pytest.approx(0.01)


def test_a14_guard_metric_statistic_has_every_branch() -> None:
    """A8/A13: `_metric_statistic` (l.1266-1274) cubre `profit_factor`, `alpha_pct` y la caida."""
    benchmark = (0.0, 0.02)
    sample = (0.01, -0.02)
    assert _metric_statistic("mean_return_pct", benchmark)(sample) == pytest.approx(-0.005)
    assert _metric_statistic("profit_factor", benchmark)((0.02,)) == pytest.approx(0.0)
    assert _metric_statistic("alpha_pct", benchmark)(sample) == pytest.approx(
        _alpha(sample, benchmark)
    )
    assert _metric_statistic("no_existe", benchmark)(sample) == pytest.approx(-0.005 - 0.01)


def test_a14_guard_excess_metric_without_a_mean() -> None:
    """A13: `_add_excess_metric` (l.1371-1381) publica la metrica sin valor si no hay sesiones."""
    blocks: dict[str, object] = {}
    _add_excess_metric(
        blocks,
        mean_block={"estimate": None},
        benchmark=(0.0, 0.02),
        n=0,
        basis=BASIS_DECLARED_COST,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )
    excess = as_map(blocks["excess_return_pct"])
    assert excess["estimate"] is None
    assert excess["lower"] is None
    assert excess["upper"] is None
    assert "ninguna sesion" in as_str(excess["reason"])


def test_a14_guard_counts_by_orders_and_counts() -> None:
    """A3: `_counts_by` (l.1510-1516) ordena las claves para que el hash no dependa de un `dict`."""
    assert _counts_by(()) == {}
    assert _counts_by(("b", "a", "b")) == {"a": 1, "b": 2}


def test_a14_guard_rule_11_band_verdicts() -> None:
    """A12: la banda de la regla 11 (l.1839-1844) distingue por encima, dentro y por debajo."""
    cache: dict[tuple[str, tuple[float, ...]], dict[str, object]] = {}
    arms = (
        _arm(ARM_OFICIAL, _run(traded=1, no_trade=1)),
        _arm(ARM_ESCENARIO, _run(traded=1, no_trade=3)),
        _arm(ARM_COSTE_DECLARADO, _run(traded=0, no_trade=2)),
    )
    payload = _rule_11_payload(arms, cache=cache)
    published = as_map(at(payload, "arms"))
    verdicts = {name: as_str(as_map(published[name])["verdict"]) for name in ARM_NAMES}
    assert verdicts == {
        ARM_OFICIAL: "above",
        ARM_ESCENARIO: "inside",
        ARM_COSTE_DECLARADO: "below",
    }
    assert at(payload, "band") == [RULE_11_BAND[0], RULE_11_BAND[1]]
    assert as_float(as_map(published[ARM_OFICIAL])["share"]) == pytest.approx(0.5)
    assert as_map(published[ARM_ESCENARIO])["share_interval"] is None


# ─────────────────────────────────────────────────────────────────────────────
# #92 · Tasa de acierto por operacion junto a la de #28 (que es por sesion)
# ─────────────────────────────────────────────────────────────────────────────
# #98: aqui vivian los tres goldens literales de la corrida regenerable de S1. Al corregir
# `_favourable_probability` cambian, y un literal regenerable convierte cualquier tarea posterior
# en un fallo de #92: se re-deriva del run (estimacion y limites con la semilla publicada), nunca
# se sustituye por otro literal.


def traded_series(run: BacktestRun) -> tuple[float, ...]:
    """La serie declarada de **solo** las sesiones operadas, re-derivada aqui (#92)."""
    return tuple(
        _declared_return_pct(outcome)
        for outcome in sessions_of(run)
        if outcome.status == STATUS_TRADED
    )


def _synthetic_metrics(run: BacktestRun) -> dict[str, dict[str, object]]:
    """Las metricas de una corrida sintetica (sin almacen) que ejercen la denominacion (#92)."""
    series = declared_series(run)
    benchmark = tuple(0.01 * (index % 3) for index in range(len(series)))
    return cast(
        "dict[str, dict[str, object]]",
        pipeline_report._row_metrics(  # pyright: ignore[reportPrivateUsage]
            series_pct=series,
            traded_series_pct=traded_series(run),
            benchmark_pct=benchmark,
            basis=BASIS_DECLARED_COST,
            cache={},
        ),
    )


@needs_store
def test_a92_per_trade_rate_is_published_with_its_own_seed(real_report: PipelineReport) -> None:
    """#92/#98: la tasa por operacion se re-deriva del run; su semilla propia sigue."""
    arm = real_report.arm(ARM_COSTE_DECLARADO)
    returns = traded_series(arm.run)
    n_wins = sum(1 for value in returns if value > 0.0)
    metrics = metrics_of(real_report, ARM_COSTE_DECLARADO)
    block = metric_block(metrics, "hit_rate_per_trade")
    assert block["estimate"] == n_wins / len(returns)
    assert as_int(block["n"]) == len(returns)
    assert as_int(block["n_trades"]) == len(returns)
    assert as_int(block["n_wins"]) == n_wins
    assert 0.0 <= as_float(block["lower"]) <= as_float(block["upper"]) <= 1.0
    assert block["estimate"] == as_int(block["n_wins"]) / as_int(block["n_trades"])
    seeds = pipeline_report._seed_by_metric()  # pyright: ignore[reportPrivateUsage]
    seed = as_int(block["seed"])
    assert seed == 53 == _derived_seed(offset=11)
    assert seed >= DEFAULT_BOOTSTRAP_SEED
    historical = {name: seeds[name] for name in METRIC_NAMES if name != "hit_rate_per_trade"}
    assert seed not in set(historical.values())
    assert as_int(metrics["hit_rate"]["seed"]) == 44
    assert seed != as_int(metrics["hit_rate"]["seed"])


@needs_store
def test_a92_per_trade_estimate_is_the_rederived_ratio(real_report: PipelineReport) -> None:
    """Criterio 8 de #92: la estimacion publicada **es** `n_wins / n_trades` de la serie operada."""
    arm = real_report.arm(ARM_COSTE_DECLARADO)
    trades = [outcome for outcome in sessions_of(arm.run) if outcome.status == STATUS_TRADED]
    returns = tuple(_declared_return_pct(outcome) for outcome in trades)
    n_wins = sum(1 for value in returns if value > 0.0)
    assert len(returns) == arm.run.traded
    block = metrics_of(real_report, ARM_COSTE_DECLARADO)["hit_rate_per_trade"]
    assert as_int(block["n"]) == len(returns)
    assert as_int(block["n_wins"]) == n_wins
    assert block["estimate"] == n_wins / len(returns)
    assert block["wins_fraction"] == f"{n_wins}/{len(returns)}"
    interval = bootstrap_confidence_interval(
        tuple(value / 100.0 for value in returns),
        lambda sample: sum(1 for value in sample if value > 0.0) / max(len(sample), 1),
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
        n_bootstrap=DEFAULT_BOOTSTRAP_SAMPLES,
        seed=as_int(block["seed"]),
    )
    assert block["lower"] == interval.lower
    assert block["upper"] == interval.upper
    assert as_float(block["lower"]) < as_float(block["estimate"]) < as_float(block["upper"])


@needs_store
def test_a92_per_trade_does_not_touch_the_session_rate(real_report: PipelineReport) -> None:
    """#92/#98: `hit_rate` conserva nombre, posicion y semilla; su valor se re-deriva del run."""
    arm = real_report.arm(ARM_COSTE_DECLARADO)
    series = declared_series(arm.run)
    n_wins = sum(1 for value in series if value > 0.0)
    metrics = metrics_of(real_report, ARM_COSTE_DECLARADO)
    block = metric_block(metrics, "hit_rate")
    assert as_int(block["n"]) == len(series)
    assert as_int(block["n_wins"]) == n_wins
    assert block["estimate"] == n_wins / len(series)
    interval = bootstrap_confidence_interval(
        tuple(value / 100.0 for value in series),
        lambda sample: sum(1 for value in sample if value > 0.0) / max(len(sample), 1),
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
        n_bootstrap=DEFAULT_BOOTSTRAP_SAMPLES,
        seed=as_int(block["seed"]),
    )
    assert block["lower"] == interval.lower
    assert block["upper"] == interval.upper
    assert as_int(block["n"]) == 500
    assert as_int(block["n_bootstrap"]) == DEFAULT_BOOTSTRAP_SAMPLES
    assert as_int(block["seed"]) == 44
    names = list(cast("list[str]", metrics["metric_names"]))
    assert names == list(METRIC_NAMES)
    assert names.index("hit_rate") == 1
    assert names[10] == "hit_rate_per_trade"
    historical = (
        "mean_return_pct",
        "hit_rate",
        "sharpe",
        "sortino",
        "max_drawdown_pct",
        "profit_factor",
        "benchmark_return_pct",
        "excess_return_pct",
        "beta",
        "alpha_pct",
    )
    for name in (*BASELINE_IDS, "liston_a", "liston_b", "liston_c", *ARM_NAMES):
        other = metrics_of(real_report, name)
        assert all(metric in other for metric in historical)
        assert list(cast("list[str]", other["metric_names"])) == names


@needs_store
def test_a92_per_trade_denominations_are_declared(real_report: PipelineReport) -> None:
    """Criterio 3 de #92: cada tasa declara su denominacion, su nota, sus aciertos y su fraccion."""
    for name in (*BASELINE_IDS, "liston_a", "liston_b", "liston_c", *ARM_NAMES):
        metrics = metrics_of(real_report, name)
        session = metrics["hit_rate"]
        trade = metrics["hit_rate_per_trade"]
        assert session["denominator"] == "session"
        assert trade["denominator"] == "trade"
        for block in (session, trade):
            note = as_str(block["denominator_note"]).strip()
            assert note
            assert "no_trade" in note and "0 exacto" in note and "skipped" in note
            n = as_int(block["n"])
            n_wins = as_int(block["n_wins"])
            assert block["wins_fraction"] == f"{n_wins}/{n}"
            if n:
                assert block["estimate"] == n_wins / n


@needs_store
def test_a92_per_trade_denominators_are_measured(real_report: PipelineReport) -> None:
    """Criterio 5 de #92: los `n` salen de los recuentos publicados, no de una afirmacion."""
    for arm_name in ARM_NAMES:
        block = row_block(real_report, arm_name)
        metrics = metrics_of(real_report, arm_name)
        traded = as_int(block["traded"])
        no_trade = as_int(block["no_trade"])
        assert as_int(metrics["hit_rate"]["n"]) == traded + no_trade
        assert as_int(metrics["hit_rate_per_trade"]["n"]) == traded
    for row in as_objects(at(real_report.payload, "table", "rows")):
        metrics = cast("dict[str, dict[str, object]]", row["metrics"])
        traded = as_int(row["traded"])
        no_trade = as_int(row["no_trade"])
        assert as_int(metrics["hit_rate"]["n"]) == traded + no_trade
        assert as_int(metrics["hit_rate_per_trade"]["n"]) == traded
        assert as_int(metrics["hit_rate_per_trade"]["n"]) == as_int(
            as_map(row["rotation"])["traded"]
        )


@needs_store
def test_a92_per_trade_is_null_and_never_zero_without_trades(real_report: PipelineReport) -> None:
    """Criterio 6 de #92: cero operaciones ⇒ `null` en la tasa por operacion, nunca `0`."""
    for name in (ARM_OFICIAL, ARM_ESCENARIO, NO_TRADE):
        block = metrics_of(real_report, name)["hit_rate_per_trade"]
        assert block["estimate"] is None
        assert block["lower"] is None
        assert block["upper"] is None
        assert block["estimate"] != 0.0
        assert as_int(block["n"]) == 0
        assert as_int(block["n_trades"]) == 0
        assert as_int(block["n_wins"]) == 0
        assert as_str(block["reason"]).strip()


def test_a92_per_trade_denominators_on_a_synthetic_run() -> None:
    """Criterio 9(a) de #92: `traded + no_trade` y `traded`; una `skipped` de mas no los mueve."""
    cost = declared_cost_breakdown()
    win = _outcome(session=date(2026, 9, 1), status=STATUS_TRADED, gross_pct=0.01, cost=cost)
    loss = _outcome(session=date(2026, 9, 2), status=STATUS_TRADED, gross_pct=-0.01, cost=cost)
    flat = _outcome(session=date(2026, 9, 3), status=STATUS_NO_TRADE)
    skip = _outcome(session=date(2026, 9, 4), status=STATUS_SKIPPED)
    run = _run(folded=(win, loss, flat), traded=2, no_trade=1)
    metrics = _synthetic_metrics(run)
    assert as_int(metrics["hit_rate"]["n"]) == run.traded + run.no_trade == 3
    assert as_int(metrics["hit_rate_per_trade"]["n"]) == run.traded == 2
    assert as_int(metrics["hit_rate"]["n_wins"]) == 1
    assert len(traded_series(run)) == 2
    with_skip = _run(folded=(win, loss, flat, skip), traded=2, no_trade=1, skipped=1)
    skipped_metrics = _synthetic_metrics(with_skip)
    assert as_int(skipped_metrics["hit_rate"]["n"]) == 3
    assert as_int(skipped_metrics["hit_rate_per_trade"]["n"]) == 2


def test_a92_per_trade_zero_return_is_not_a_win() -> None:
    """Criterio 9(b) de #92: un retorno declarado exactamente `0.0` no cuenta como ganador."""
    cost = declared_cost_breakdown()
    flat_return = float(cost.c_declared_pct) / 100.0
    win = _outcome(session=date(2026, 9, 1), status=STATUS_TRADED, gross_pct=0.01, cost=cost)
    zero = _outcome(
        session=date(2026, 9, 2), status=STATUS_TRADED, gross_pct=flat_return, cost=cost
    )
    loss = _outcome(session=date(2026, 9, 3), status=STATUS_TRADED, gross_pct=-0.01, cost=cost)
    assert _declared_return_pct(zero) == 0.0
    run = _run(folded=(win, zero, loss), traded=3, no_trade=0)
    block = _synthetic_metrics(run)["hit_rate_per_trade"]
    assert as_int(block["n_wins"]) == 1
    assert as_int(block["n_trades"]) == 3
    assert block["estimate"] == 1 / 3
    assert as_int(_synthetic_metrics(run)["hit_rate"]["n_wins"]) == 1


@needs_store
def test_support_markdown_declares_both_denominations(real_report: PipelineReport) -> None:
    """Criterio 12 de #92: el Markdown declara la tasa por sesion y la de por operacion."""
    markdown = render_markdown(real_report)
    assert "`hit_rate` **por sesion**" in markdown
    assert "`hit_rate_per_trade` **por operacion**" in markdown
    metrics = metrics_of(real_report, ARM_COSTE_DECLARADO)
    session = metrics["hit_rate"]
    trade = metrics["hit_rate_per_trade"]
    assert f"`{session['wins_fraction']}`, n = {session['n']}" in markdown
    assert f"`{trade['wins_fraction']}`, n = {trade['n']}" in markdown


# ─────────────────────────────────────────────────────────────────────────────
# A15 · Nada congelado se toca
# ─────────────────────────────────────────────────────────────────────────────
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


def test_a15_frozen_modules_are_untouched() -> None:
    """A15: la entrega toca sus cinco ficheros y **ningun** modulo congelado."""
    assert _git("status", "--porcelain").strip() == ""
    changed = set(_git("diff", "--name-only", f"{BASE_COMMIT}..HEAD").splitlines())
    # #89: la igualdad contra `BASE_COMMIT` solo valia mientras `HEAD` fuese la punta de #28;
    # cualquier commit posterior anade ficheros al diff. La intencion de A15 es subconjunto
    # (los dos ficheros de la entrega estan) y disyuncion (ningun congelado esta), como en #26.
    assert set(WRITTEN) <= changed
    assert changed.isdisjoint(FROZEN)


def test_a15_consumed_modules_keep_their_contract() -> None:
    """A15: los modulos consumidos conservan las constantes que este informe publica."""
    from cfdtrader.decision import gate as gate_module

    expected: dict[str, object] = {
        "TARGET_MIN_COST_MULTIPLE": Decimal("2"),
        "DECISION_THRESHOLD": 0.5,
    }
    published = {name: getattr(gate_module, name) for name in expected}
    assert published == expected
    assert PHASE1_PLAN.n_splits == 10
    assert PHASE1_PLAN.test_size == 50
    assert float(declared_slippage_assumption().pct_of_r or Decimal(0)) == 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Apoyo: el modelo y el gate se consumen tal cual
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_support_model_is_the_calibrated_one(real_report: PipelineReport) -> None:
    """El informe publica la probabilidad calibrada del modelo de #24/#25 y su digest."""
    model = as_map(at(real_report.payload, "model"))
    assert as_int(model["n_folds"]) == len(real_report.split_plan.folds)
    assert as_float(model["decision_threshold"]) == 0.5
    assert as_int(as_map(model["probabilities"])["n_test_with_probability"]) == len(
        real_report.test_sessions
    )
    assert real_report.model.features == BASELINE_FEATURES
    for fold, plan_fold in zip(real_report.model.folds, real_report.split_plan.folds, strict=True):
        assert fold.test_positions == plan_fold.test


@needs_store
def test_support_gate_outputs_are_the_declared_map(real_report: PipelineReport) -> None:
    """El mapeo temporal produce una salida del gate por sesion de *test*, sin coste total."""
    arm = real_report.arm(ARM_ESCENARIO)
    assert len(arm.outputs) == len(real_report.test_sessions)
    for session, output in arm.outputs.items():
        assert session in set(real_report.test_sessions)
        assert output.expected_move_basis == "garch_forecast_sigma_1s"
        assert output.cost_total_pct is None
        assert output.ev_net_pct is None
        assert output.tier == "C"
        assert output.ev_declared_pct is not None
        assert output.stop_pct > 0


@needs_store
def test_support_markdown_covers_the_sections(real_report: PipelineReport) -> None:
    """El `.md` publica las mismas secciones que el payload, sin numeros inventados."""
    markdown = render_markdown(real_report)
    for heading in (
        "## Universo y plan",
        "## Mapeo temporal declarado",
        "## Los tres brazos",
        "## Tabla unica (seis baselines y los tres listones)",
        "## Metricas netas",
        "## Limitaciones y seguimientos",
    ):
        assert heading in markdown
    assert real_report.report_sha256 in markdown
    assert "NaN" not in markdown
    assert "profit factor" in markdown


@needs_store
def test_support_n_trade_never_traded_arm_is_declared(real_report: PipelineReport) -> None:
    """El brazo de coste declarado publica su nocional y su motivo por sesion."""
    arm = real_report.arm(ARM_COSTE_DECLARADO)
    reasons = {outcome.reason for outcome in sessions_of(arm.run) if outcome.reason}
    assert reasons
    assert all(as_str(reason).strip() for reason in reasons)
    assert any("arm=coste_declarado" in as_str(reason) for reason in reasons)


@needs_store
def test_support_no_trade_baseline_series_is_zero(real_report: PipelineReport) -> None:
    """El baseline `no_trade` publica una serie declarada de ceros y no opera ninguna sesion."""
    run = next(outcome.run for outcome in real_report.baselines if outcome.baseline == NO_TRADE)
    assert run.traded == 0
    assert set(declared_series(run)) == {0.0}
    assert real_report.row(NO_TRADE).traded == 0


# ─────────────────────────────────────────────────────────────────────────────
# T28d (#94) · La serie declarada por sesion y su `series_sha256` en `arms`
# ─────────────────────────────────────────────────────────────────────────────
def declared_series_block(report: PipelineReport, name: str) -> dict[str, object]:
    """El bloque ``declared_series`` publicado por ese brazo (del payload, no del objeto)."""
    return as_map(row_block(report, name)["declared_series"])


def _published_series(block: Mapping[str, object]) -> list[float]:
    """La lista ``series_pct`` del bloque, como numeros JSON."""
    return [as_float(value) for value in as_list(block["series_pct"])]


def _series_digest(block: Mapping[str, object]) -> str:
    """El sha256 del cuerpo del bloque **sin** su propia clave, con su prefijo (A3/T28d)."""
    body = {key: value for key, value in block.items() if key != "series_sha256"}
    digest = hashlib.sha256(canonical_text(body).encode("utf-8")).hexdigest()
    return f"{HASH_PREFIX}{digest}"


@needs_store
def test_t28d_declared_series_block_shape(real_report: PipelineReport) -> None:
    """Criterio 1: los tres brazos publican las cuatro claves y ``n == n_test - skipped``."""
    for name in ARM_NAMES:
        arm = as_map(at(real_report.payload, "arms", name))
        block = as_map(arm["declared_series"])
        assert set(block) == {"units", "n", "series_pct", "series_sha256"}
        assert as_str(block["units"]) == SERIES_UNITS
        series = as_list(block["series_pct"])
        assert all(isinstance(value, float) for value in series)
        assert as_int(block["n"]) == len(series)
        assert as_int(block["n"]) == as_int(arm["n_test"]) - as_int(arm["skipped"])


@needs_store
def test_t28d_declared_series_is_the_metric_series(real_report: PipelineReport) -> None:
    """Criterio 2: la serie publicada **es** la de las metricas, recomputada sesion a sesion."""
    for name in ARM_NAMES:
        arm = real_report.arm(name)
        published = tuple(_published_series(declared_series_block(real_report, name)))
        recomputed = declared_series(arm.run)
        assert len(published) == len(recomputed)
        for left, right in zip(published, recomputed, strict=True):
            assert left == right
        kept = [outcome for outcome in sessions_of(arm.run) if outcome.status != STATUS_SKIPPED]
        for outcome, value in zip(kept, published, strict=True):
            if outcome.status == STATUS_NO_TRADE:
                assert value == 0.0
            else:
                assert outcome.gross_pct is not None and outcome.cost is not None
                assert value == 100.0 * outcome.gross_pct - float(outcome.cost.c_declared_pct)


@needs_store
def test_t28d_declared_series_is_aligned_to_test_sessions(real_report: PipelineReport) -> None:
    """Criterio 3: hoy ``skipped == 0``, largo = 500 y el indice de cada entrada es su sesion."""
    index_of = {session: position for position, session in enumerate(real_report.test_sessions)}
    for name in ARM_NAMES:
        arm = real_report.arm(name)
        published = as_map(at(real_report.payload, "arms", name))
        series = _published_series(declared_series_block(real_report, name))
        assert as_int(published["skipped"]) == 0
        assert len(series) == len(real_report.test_sessions) == 500
        by_session = {outcome.session: outcome for outcome in sessions_of(arm.run)}
        for outcome in sessions_of(arm.run):
            position = index_of[outcome.session]
            if outcome.status == STATUS_NO_TRADE:
                assert series[position] == 0.0
            elif outcome.status == STATUS_TRADED:
                assert outcome.gross_pct is not None and outcome.cost is not None
                assert series[position] == 100.0 * outcome.gross_pct - float(
                    outcome.cost.c_declared_pct
                )
        for position, value in enumerate(series):
            if value != 0.0:
                assert by_session[real_report.test_sessions[position]].status == STATUS_TRADED


@needs_store
def test_t28d_series_digest_format_and_self_consistency_from_disk(
    real_report: PipelineReport,
) -> None:
    """Criterios 4 y 5: ``sha256:`` + 64 hex minuscula y digest recomputado del JSON en disco."""
    json_path = real_report.reports_dir / f"{STEM}.json"
    on_disk = as_map(json.loads(json_path.read_text(encoding="utf-8")))
    for name in ARM_NAMES:
        block = as_map(at(on_disk, "arms", name, "declared_series"))
        digest = as_str(block["series_sha256"])
        body = digest.removeprefix(HASH_PREFIX)
        assert digest.startswith(HASH_PREFIX)
        assert len(body) == 64
        assert all(character in "0123456789abcdef" for character in body)
        assert digest == _series_digest(block)


@needs_store
def test_t28d_declared_series_is_identical_across_fresh_processes(
    real_report: PipelineReport, fresh_runs: Mapping[str, CliRun]
) -> None:
    """Criterio 6: el bloque es identico byte a byte (y mismo ``series_sha256``) entre procesos."""
    ours = declared_series_block(real_report, ARM_COSTE_DECLARADO)
    for name in ("seed0", "seed1"):
        fresh = as_map(json.loads(fresh_runs[name].json_path().read_text(encoding="utf-8")))
        published = as_map(at(fresh, "arms", ARM_COSTE_DECLARADO, "declared_series"))
        assert published == ours
        assert as_str(published["series_sha256"]) == as_str(ours["series_sha256"])
        assert as_str(fresh["report_sha256"]) == real_report.report_sha256


@needs_store
def test_t28d_metrics_come_from_the_published_series(real_report: PipelineReport) -> None:
    """Criterio 7: `hit_rate` y `sharpe` se reproducen desde ``series_pct / 100`` y su ``seed``."""
    metrics = metrics_of(real_report, ARM_COSTE_DECLARADO)
    series = _published_series(declared_series_block(real_report, ARM_COSTE_DECLARADO))
    values = tuple(value / 100.0 for value in series)
    for name in ("hit_rate", "sharpe"):
        published = metrics[name]
        interval = bootstrap_confidence_interval(
            values,
            _metric_statistic(name, ()),
            confidence_level=DEFAULT_CONFIDENCE_LEVEL,
            n_bootstrap=DEFAULT_BOOTSTRAP_SAMPLES,
            seed=as_int(published["seed"]),
        )
        assert interval.estimate == published["estimate"]
        assert interval.lower == published["lower"]
        assert interval.upper == published["upper"]


@needs_store
def test_t28d_all_zero_arms_publish_the_null_series(real_report: PipelineReport) -> None:
    """Criterio 8: `oficial` y `escenario` publican ``n`` ceros exactos y su digest."""
    for name in (ARM_OFICIAL, ARM_ESCENARIO):
        arm = as_map(at(real_report.payload, "arms", name))
        block = as_map(arm["declared_series"])
        series = _published_series(block)
        assert as_int(block["n"]) == len(series) == len(real_report.test_sessions)
        assert series and all(value == 0.0 for value in series)
        assert math.fsum(series) == 0.0
        assert as_str(block["series_sha256"]).startswith(HASH_PREFIX)
        assert arm["declared_return_series_all_zero"] is True


def test_t28d_skipped_is_dropped_not_a_zero() -> None:
    """Criterio 9: la `skipped` se descarta (``n == n_test - skipped``) y el digest es estable."""
    cost = declared_cost_breakdown()
    traded = _outcome(session=date(2026, 9, 1), status=STATUS_TRADED, gross_pct=0.01, cost=cost)
    flat = _outcome(session=date(2026, 9, 2), status=STATUS_NO_TRADE)
    skip = _outcome(session=date(2026, 9, 3), status=STATUS_SKIPPED)
    run = _run(folded=(traded, flat, skip), traded=1, no_trade=1, skipped=1)
    block = _declared_series_payload(_series_of_run(run))
    series = _published_series(block)
    assert as_int(block["n"]) == len(series) == run.traded + run.no_trade == 2
    assert series[0] == pytest.approx(100.0 * 0.01 - float(cost.c_declared_pct))
    assert series[1] == 0.0
    again = _declared_series_payload(_series_of_run(run))
    assert as_str(again["series_sha256"]) == as_str(block["series_sha256"])
    assert as_str(block["series_sha256"]) == _series_digest(block)


def test_t28d_null_cost_is_a_typed_error() -> None:
    """Criterio 10: sin `gross_pct` o sin `CostBreakdown` la derivacion **lanza**."""
    cost = declared_cost_breakdown()
    without_gross = _outcome(session=SESSION, status=STATUS_TRADED, gross_pct=None, cost=cost)
    with pytest.raises(PipelineReportError, match="retorno declarado"):
        _declared_series_payload(_series_of_run(_run(folded=(without_gross,), traded=1)))
    without_cost = _outcome(session=SESSION, status=STATUS_TRADED, gross_pct=0.01, cost=None)
    with pytest.raises(PipelineReportError, match="retorno declarado"):
        _series_of_run(_run(folded=(without_cost,), traded=1))


@needs_store
def test_t28d_render_markdown_runs_with_the_new_block(real_report: PipelineReport) -> None:
    """Criterio 12: el Markdown se renderiza con el bloque nuevo, sin `KeyError`."""
    markdown = render_markdown(real_report)
    assert markdown.strip()
    assert real_report.report_sha256 in markdown
