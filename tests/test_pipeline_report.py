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

import pytest

from cfdtrader.analysis import pipeline_report
from cfdtrader.analysis.backtest_report import PHASE1_PLAN, SERIES_ID, load_history
from cfdtrader.analysis.pipeline_report import (
    ARM_COSTE_DECLARADO,
    ARM_ESCENARIO,
    ARM_NAMES,
    ARM_OFICIAL,
    BASIS_DECLARED_COST,
    METRIC_NAMES,
    NO_INTERVAL_METRICS,
    REPORT_PREFIX,
    RULE_11_BAND,
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
from cfdtrader.backtest.costs import declared_cost_model, declared_slippage_assumption
from cfdtrader.backtest.engine import (
    STATUS_NO_TRADE,
    STATUS_SKIPPED,
    STATUS_TRADED,
    BacktestRun,
    SessionOutcome,
    canonical_text,
)
from cfdtrader.backtest.metrics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    ConfidenceInterval,
)
from cfdtrader.data.store import Store
from cfdtrader.decision.gate import (
    DECISION_THRESHOLD,
    TARGET_MIN_COST_MULTIPLE,
    GateOutput,
    GateParameters,
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

#: Los dos ficheros que la entrega **solo** puede tocar, y los seis congelados (A15).
WRITTEN: Final[frozenset[str]] = frozenset(
    {
        "src/cfdtrader/analysis/pipeline_report.py",
        "tests/test_pipeline_report.py",
    }
)
FROZEN: Final[tuple[str, ...]] = (
    "src/cfdtrader/decision/gate.py",
    "src/cfdtrader/backtest/engine.py",
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
    """A3: el mismo hash con `PYTHONHASHSEED` 0, 1 y el aleatorio de este proceso."""
    assert fresh_runs["seed0"].report_sha256 == real_report.report_sha256
    assert fresh_runs["seed1"].report_sha256 == real_report.report_sha256


@needs_store
def test_a3_hash_is_fixed_with_prefix(real_report: PipelineReport) -> None:
    """A3: el test **fija** el digest, con su prefijo (un sha256 desnudo lo bloquea el hook)."""
    assert real_report.report_sha256 == (
        "sha256:687e6b67c3413292aba9b9413748eff0df114aff548e84957d28a4e0a01ffc59"
    )


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
    """El tier A/B/C re-derivado sobre el EV declarado, recalculado en el test (A6)."""
    ev_declared = output.ev_declared_pct
    if ev_declared is None:
        return "C"
    favourable = output.prob_up_calibrated
    if output.direction is not None and str(output.direction) != "long":
        favourable = 1.0 - favourable
    if ev_declared > cast("Decimal", params.tier_a_cost_multiple) * output.cost_pct and Decimal(
        str(favourable)
    ) > cast("Decimal", params.tier_a_min_probability):
        return "A"
    if ev_declared > cast("Decimal", params.tier_b_cost_multiple) * output.cost_pct:
        return "B"
    return "C"


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
    assert mixed[0] == pytest.approx(row.series_pct[0] / 100.0, rel=0.6)


# ─────────────────────────────────────────────────────────────────────────────
# A8 · Intervalos bootstrap con etiqueta
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a8_every_metric_carries_its_interval_label(real_report: PipelineReport) -> None:
    """A8: las diez metricas de cada fila y de cada brazo llevan su etiqueta completa."""
    for name in (*BASELINE_IDS, "liston_a", "liston_b", "liston_c", *ARM_NAMES):
        metrics = metrics_of(real_report, name)
        assert list(cast("list[str]", metrics["metric_names"])) == list(METRIC_NAMES)
        for metric in METRIC_NAMES:
            block = metric_block(metrics, metric)
            if metric in NO_INTERVAL_METRICS:
                assert block["lower"] is None and block["upper"] is None
                assert as_str(block["reason"]).strip()
                continue
            estimate = block["estimate"]
            lower = as_float(block["lower"])
            upper = as_float(block["upper"])
            if estimate is not None:
                assert lower <= as_float(estimate) <= upper
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
    assert "calculate_metrics" not in SOURCE
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
def test_a10_check_block_measures_the_hundredfold_discrepancy(
    real_report: PipelineReport,
) -> None:
    """A10: el bloque `check` publica `gross_pct`, `c_declared_pct` y la discrepancia medida."""
    check = as_map(at(real_report.payload, "check"))
    assert check["issue"] == "#80"
    gross = as_float(check["gross_pct"])
    cost = float(as_str(check["c_declared_pct"]))
    assert as_float(check["engine_subtraction"]) == pytest.approx(gross - cost)
    assert as_float(check["declared_subtraction"]) == pytest.approx(100.0 * gross - cost)
    assert as_float(check["discrepancy"]) == pytest.approx(99.0 * gross)
    issues = [as_str(entry["issue"]) for entry in as_objects(at(real_report.payload, "follow_ups"))]
    assert issues[0] == "#80"
    assert as_str(check["session"]) in {
        session.isoformat() for session in real_report.test_sessions
    }


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
    assert benchmark_return == pytest.approx(expected_return)
    assert math.isfinite(benchmark_return)
    assert benchmark_return != 0.0


@needs_store
def test_a13_markdown_says_which_of_the_two_carries_the_result(
    real_report: PipelineReport,
) -> None:
    """A13: el `.md` dice si el resultado lo carga el alpha o el beta, con los numeros."""
    attribution = as_map(at(real_report.payload, "arm_comparison", "attribution"))
    loader = as_str(attribution["loader"])
    assert loader in {"alpha", "beta", "no_atribuible"}
    statement = as_str(attribution["statement"])
    assert loader in statement
    markdown = render_markdown(real_report)
    assert statement in markdown
    assert SERIES_ID in markdown
    assert "Atribucion: alpha contra beta" in markdown


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
    """A15: `git diff <base>..HEAD` solo trae los dos ficheros de la entrega."""
    assert _git("status", "--porcelain").strip() == ""
    changed = set(_git("diff", "--name-only", f"{BASE_COMMIT}..HEAD").splitlines())
    assert changed == set(WRITTEN)
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
        "## Chequeo de unidades (#80)",
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
