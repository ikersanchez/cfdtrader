"""Pruebas del veredicto robusto por dominancia de la base declarada (`tasks.md`, T29c) — #93.

Un test por criterio (``test_a1_...`` … ``test_a18_...``), siempre con ``tmp_path`` y la fixture
de sesion de ``tests/conftest.py`` que **huella ``data/`` y ``runs/``**. Los bordes incomodos
(artefacto ausente, dos artefactos ambiguos, ``--as-of`` ausente o invalido, `write=False`,
serie degenerada, rejilla no ascendente, base que cruza, reproduccion que no cuadra) tienen su
propio caso: en este proyecto los defectos aparecen al **reejecutar**, no en la primera pasada.

**Coste de la suite de este fichero, y por que.** El artefacto de #28 **no** publica la serie
declarada por sesion (→ #94), asi que la corrida real **reejecuta el pipeline** de #28
(``analyse(..., write=False)``): ~2 minutos por proceso. El gasto se concentra en:

- **dos procesos frescos** (``PYTHONHASHSEED`` ``0`` y ``1``) **en paralelo**, que son los dos
  primeros casos de determinismo de A4 y a la vez la prueba de A1 (CLI y ficheros) y de A3 (el
  artefacto no cambia);
- **una corrida de la sesion** en el proceso de ``pytest`` (``PYTHONHASHSEED`` aleatorio, el
  tercer caso de A4) con el bootstrap de #28 **doblado** —ese doble no cambia nada de lo que este
  informe publica, porque sus metricas las calcula este modulo— y el bootstrap **real** en el
  modulo, que es lo que permite comprobar A9 contra el artefacto real;
- los casos sinteticos (A11-A14, A17) son **puros**: no reejecutan el pipeline, y bajan
  ``n_bootstrap`` para no gastar minutos por test.

Los dos tests que rompen a proposito (rejilla no ascendente y reproduccion que no cuadra) usan
artefactos copiados y con los dos bootstrap doblados: no necesitan numeros reales.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import itertools
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Generator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Final, cast

import pytest

from cfdtrader.analysis import phase2_dominance, pipeline_report
from cfdtrader.analysis.backtest_report import load_history
from cfdtrader.analysis.phase2_dominance import (
    CODE_BASE_CROSSES,
    CODE_DEGENERATE,
    CODE_FAIL,
    CODE_INCONCLUSIVE,
    DeclaredSeries,
    Dominance,
    DominanceReport,
    DominanceViolationError,
    InvalidSlippageGridError,
    Phase2DominanceError,
    ReproductionMismatchError,
    compute_dominance,
    declared_series_of,
    seed_of,
    slippage_levels,
)
from cfdtrader.backtest.engine import (
    STATUS_NO_TRADE,
    STATUS_SKIPPED,
    STATUS_TRADED,
    BacktestRun,
    canonical_text,
)
from cfdtrader.backtest.metrics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    ConfidenceInterval,
)
from cfdtrader.data.store import Store

MODULE_PATH: Final[Path] = Path(str(phase2_dominance.__file__))
TEST_PATH: Final[Path] = Path(__file__).resolve()
REPO_ROOT: Final[Path] = TEST_PATH.parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"
REAL_REPORTS: Final[Path] = REAL_DATA / "derived" / "reports"
PIPELINE_ARTIFACT: Final[Path] = REAL_REPORTS / "pipeline_backtest_2026-09-23.json"
MODEL_ARTIFACT: Final[Path] = REAL_REPORTS / "model_comparison_2026-09-22.json"

#: Instante declarado de las corridas que fijan el hash dorado (A4).
NOW: Final[datetime] = datetime(2026, 9, 24, 0, 0, tzinfo=UTC)

#: Nombre base del informe de esa corrida.
STEM: Final[str] = "phase2_dominance_2026-09-24"

# #95: aqui vivia `GOLDEN_REPORT_SHA256`, el `report_sha256` dorado de la corrida real. Ese
# digest es el del payload de **este** informe, y dentro viaja el `sha256` del artefacto de #28
# (que #92 regenero al publicar `hit_rate_per_trade`), asi que **cualquier** tarea posterior lo
# invalida: un dorado asi convierte un trabajo ajeno en un fallo de #93. No se vuelve a cablear
# ningun digest regenerable. Lo que A4 vigila —prefijo `sha256:` + 64 hex, autoconsistencia del
# payload, determinismo entre procesos y que `provenance.pipeline.sha256` sea el del **fichero**
# del artefacto— se comprueba sin literal. El unico dorado estable de la fase es el `sha256` del
# bloque §11.6 de `plan.md`, que ya fija `tests/test_phase2_report.py` (#29).

#: Los tres percentiles declarados de #8 que la rejilla tiene que incluir (A7).
GOLDEN_GRID_BP: Final[tuple[float, ...]] = (0.0, 13.2, 26.6, 46.8)

SOURCE: Final[str] = MODULE_PATH.read_text(encoding="utf-8")
TREE: Final[ast.Module] = ast.parse(SOURCE)

#: Claves que cada celda tiene que publicar (A10).
CELL_KEYS: Final[tuple[str, ...]] = (
    "slippage_bp",
    "state",
    "code",
    "crosses",
    "sharpe_excludes_zero_above",
    "sharpe_below",
    "sharpe",
    "hit_rate",
)
METRIC_KEYS: Final[tuple[str, ...]] = ("estimate", "lower", "upper")

#: Los ficheros que la entrega de #93 **debe** traer en el diff (A18): el modulo y sus tests. El
#: `__init__` no hizo falta (no exporta ningun informe de la fase), asi que no entra: un
#: subconjunto no exige que la tarea toque todo lo que podria tocar.
WRITTEN: Final[frozenset[str]] = frozenset(
    {
        "src/cfdtrader/analysis/phase2_dominance.py",
        "tests/test_phase2_dominance.py",
    }
)

#: Los modulos **realmente congelados** que A18 prohibe tocar. Tras #80 son exactamente estos
#: cuatro; las guardias A15 de `tests/test_pipeline_report.py` y A12 de
#: `tests/test_model_comparison.py` apuntan a los mismos.
FROZEN: Final[frozenset[str]] = frozenset(
    {
        "src/cfdtrader/backtest/baselines.py",
        "src/cfdtrader/backtest/metrics.py",
        "src/cfdtrader/models/baseline.py",
        "src/cfdtrader/analysis/feature_frame.py",
    }
)

# #95: #93 declaraba ademas «ajenos» y exigia `changed <= ALLOWED_PATHS` para **todo** el diff
# desde `BASE_COMMIT`, es decir que ningun fichero ajeno al de la entrega cambiara nunca mas.
# Eso no se sostiene: una tarea posterior legitima los toca (#92 ya regenero
# `pipeline_report.py` y su test). Se quedan solo como nota documental, **sin** asercion: la
# disyuncion permanente es contra FROZEN.
#   ajenos que #93 declaraba: src/cfdtrader/analysis/pipeline_report.py,
#       src/cfdtrader/analysis/phase2_report.py, src/cfdtrader/backtest/engine.py,
#       src/cfdtrader/backtest/costs.py

#: Commit del que arranca la tarea: el `git diff` del rango tiene que tocar solo lo declarado.
BASE_COMMIT: Final[str] = "35d2592"

needs_store = pytest.mark.skipif(
    not (REAL_DATA / "derived" / "labels").exists(),
    reason="el almacen real no esta en el arbol: los numeros de A1-A16 son los suyos",
)


# ─────────────────────────────────────────────────────────────────────────────
# Doble barato del bootstrap de #15 (el mismo de `tests/test_pipeline_report.py`)
# ─────────────────────────────────────────────────────────────────────────────
def _fast_interval(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    confidence_level: float,
    n_bootstrap: int,
    seed: int,
) -> ConfidenceInterval:
    """Doble determinista del intervalo: la estimacion, sin remuestreo."""
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
def _patched_fast(*, ours: bool) -> Generator[None, None, None]:
    """Dobla el bootstrap de #15 en #28 y, si se pide, en este modulo.

    La corrida de la sesion dobla **solo** el de #28: sus metricas no entran en el payload de
    este informe (las calcula el modulo), pero bajan la reejecucion de ~2 minutos a ~20 segundos.
    """
    original_pipeline = pipeline_report.bootstrap_confidence_interval
    original_ours = phase2_dominance.bootstrap_confidence_interval
    pipeline_report.bootstrap_confidence_interval = _fast_interval
    if ours:
        phase2_dominance.bootstrap_confidence_interval = _fast_interval
    try:
        yield
    finally:
        pipeline_report.bootstrap_confidence_interval = original_pipeline
        phase2_dominance.bootstrap_confidence_interval = original_ours


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
    return dict(cast("Mapping[str, object]", node))


def as_float(node: object, *keys: str) -> float:
    """Un numero anidado del payload, sin ``Any`` por el camino."""
    return float(cast("float", at(node, *keys)))


def as_text(node: object, *keys: str) -> str:
    """Una cadena anidada del payload."""
    return str(at(node, *keys))


def cells_of(report: DominanceReport) -> list[dict[str, object]]:
    """Las celdas publicadas, como mappings."""
    return [as_map(item) for item in cast("list[object]", report.payload["cells"])]


def _fingerprint(root: Path) -> dict[str, str]:
    """Ruta relativa → sha256 de cada fichero: detecta altas, bajas y cambios."""
    if not root.exists():  # pragma: no cover - el almacen real esta en el arbol en esta sesion
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _copy_artifacts(directory: Path) -> Path:
    """Copia los artefactos reales de #28/#26 (solo lectura) dentro del `tmp_path`."""
    reports = directory / "derived" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    for artifact in (PIPELINE_ARTIFACT, MODEL_ARTIFACT):
        shutil.copy2(artifact, reports / artifact.name)
    return reports


# ─────────────────────────────────────────────────────────────────────────────
# Los dos procesos frescos de A4, en paralelo, y la corrida de la sesion
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CliRun:
    """Lo que deja una corrida de la CLI en un proceso fresco."""

    directory: Path
    code: int
    stderr: str
    json_bytes: bytes
    markdown_bytes: bytes
    report_sha256: str
    reports_fingerprint_before: Mapping[str, str]
    reports_fingerprint_after: Mapping[str, str]


def _run_cli(directory: Path, hash_seed: str) -> CliRun:
    """Corre la CLI en un proceso fresco, con su `PYTHONHASHSEED` y su dir."""
    reports = _copy_artifacts(directory)
    before = _fingerprint(REAL_REPORTS)
    environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
    completed = subprocess.run(  # noqa: S603 - comando fijo (el interprete de la sesion)
        [
            sys.executable,
            "-m",
            "cfdtrader.analysis.phase2_dominance",
            "--data-root",
            str(REAL_DATA),
            "--reports-dir",
            str(reports),
            "--as-of",
            NOW.isoformat(),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        cwd=str(REPO_ROOT),
    )
    json_path = reports / f"{STEM}.json"
    markdown_path = reports / f"{STEM}.md"
    digest = ""
    if completed.returncode == 0 and json_path.is_file():
        digest = str(json.loads(json_path.read_text(encoding="utf-8"))["report_sha256"])
    return CliRun(
        directory=directory,
        code=completed.returncode,
        stderr=completed.stderr,
        json_bytes=json_path.read_bytes() if json_path.is_file() else b"",
        markdown_bytes=markdown_path.read_bytes() if markdown_path.is_file() else b"",
        report_sha256=digest,
        reports_fingerprint_before=before,
        reports_fingerprint_after=_fingerprint(REAL_REPORTS),
    )


@pytest.fixture(scope="session")
def fresh_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, CliRun]:
    """Dos procesos frescos (``PYTHONHASHSEED`` ``0`` y ``1``) **en paralelo**, cada uno con su dir.

    Cada uno paga una reejecucion completa del pipeline de #28 (~2 minutos), asi que se lanzan a
    la vez: en serie la suite de este fichero pasaria de tres a cinco minutos.
    """
    root = tmp_path_factory.mktemp("dominance_fresh")
    with ThreadPoolExecutor(max_workers=2) as pool:
        zero = pool.submit(_run_cli, root / "seed0", "0")
        one = pool.submit(_run_cli, root / "seed1", "1")
        return {"seed0": zero.result(), "seed1": one.result()}


@pytest.fixture(scope="session")
def dominance_report(tmp_path_factory: pytest.TempPathFactory) -> DominanceReport:
    """La corrida de la sesion (``PYTHONHASHSEED`` aleatorio) **una vez** por sesion.

    Dobla el bootstrap de #28 (sus metricas no entran en este payload) y deja **real** el de este
    modulo, que es el que reproduce el artefacto de #28 para A9.
    """
    reports = _copy_artifacts(tmp_path_factory.mktemp("dominance_session"))
    with _patched_fast(ours=False):
        return phase2_dominance.analyse(
            store=Store(REAL_DATA), reports_dir=reports, as_of=NOW, write=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# Series sinteticas para el calculo puro (A11-A14, A17)
# ─────────────────────────────────────────────────────────────────────────────
#: `p*` de la fila principal de §11.6 bajo el escenario de `R` mas exigente. Solo vive en el
#: test: el modulo lo **deriva** con `phase2_report.p_star_block` (A15).
P_STAR_SYNTHETIC: Final[Decimal] = Decimal("0.5021")

#: Rejilla sintetica minima para los casos puros: solo hace falta que arranque en `0`.
SYNTHETIC_LEVELS: Final[tuple[tuple[str, float], ...]] = (
    ("sin_slippage", 0.0),
    ("mediana", 13.2),
)

#: Muestras sinteticas, sin pipeline y sin almacen (A17).
N_SYNTHETIC: Final[int] = 40
N_BOOTSTRAP_SYNTHETIC: Final[int] = 200


def _series(values: Sequence[float], *, traded: Sequence[bool] | None = None) -> DeclaredSeries:
    """Una serie declarada sintetica: todas las sesiones operadas salvo que se diga otra cosa."""
    return DeclaredSeries(
        values_pct=tuple(values),
        traded=tuple(traded) if traded is not None else (True,) * len(values),
    )


def _positive_series() -> DeclaredSeries:
    """Serie claramente ganadora: su IC del Sharpe excluye el 0 **por arriba**."""
    return _series([0.5 + 0.005 * (index % 5) - 0.01 for index in range(N_SYNTHETIC)])


def _negative_series() -> DeclaredSeries:
    """Serie claramente perdedora: su IC del Sharpe excluye el 0 **por abajo**."""
    return _series([-0.5 - 0.005 * (index % 5) + 0.01 for index in range(N_SYNTHETIC)])


def _flat_series() -> DeclaredSeries:
    """Serie sin edge: media 0, el IC del Sharpe contiene el 0."""
    return _series([0.5 if index % 2 == 0 else -0.5 for index in range(N_SYNTHETIC)])


def _dominance(
    series: DeclaredSeries, levels: Sequence[tuple[str, float]] = SYNTHETIC_LEVELS
) -> Dominance:
    """El calculo puro sobre una serie sintetica, con pocos remuestreos."""
    return compute_dominance(
        series,
        levels=levels,
        p_star_fraction=P_STAR_SYNTHETIC,
        n_bootstrap=N_BOOTSTRAP_SYNTHETIC,
    )


# ─────────────────────────────────────────────────────────────────────────────
# A1 — API del modulo y CLI
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_module_api_and_cli(fresh_runs: Mapping[str, CliRun]) -> None:
    """A1: existe el modulo con su API minima y la CLI escribe el par JSON/Markdown."""
    required = (
        "analyse",
        "main",
        "compute_dominance",
        "declared_series_of",
        "render_markdown",
        "slippage_levels",
        "Dominance",
        "DominanceReport",
        "DeclaredSeries",
        "Phase2DominanceError",
    )
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    assert [name for name in required if name not in phase2_dominance.__all__] == []
    assert all(hasattr(phase2_dominance, name) for name in required)

    signature = inspect.signature(phase2_dominance.analyse)
    assert set(signature.parameters) == {"store", "reports_dir", "as_of", "write"}
    assert signature.parameters["write"].default is True
    assert callable(phase2_dominance.main)

    flags = _cli_flags()
    assert flags == ["--data-root", "--reports-dir", "--as-of"]

    run = fresh_runs["seed0"]
    assert run.code == 0, run.stderr
    reports = run.directory / "derived" / "reports"
    assert (reports / f"{STEM}.json").is_file()
    assert (reports / f"{STEM}.md").is_file()
    assert run.markdown_bytes.startswith(b"# Veredicto robusto de Fase 2")


@needs_store
def test_a1_cli_success_in_process(tmp_path: Path) -> None:
    """A1: la CLI sale con 0 y escribe el par, aunque el veredicto no sea aprobar."""
    reports = _copy_artifacts(tmp_path)
    with _patched_fast(ours=False):
        code = phase2_dominance.main(
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


def _cli_flags() -> list[str]:
    """Los `--flags` que la CLI declara, leidos del AST del modulo (A1)."""
    flags: list[str] = []
    for node in ast.walk(TREE):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "parser"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            flags.append(node.args[0].value)
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# A2 — `--as-of` obligatorio, reloj prohibido, `write=False` no escribe
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_missing_or_invalid_as_of_exits_2_without_writing(tmp_path: Path) -> None:
    """A2: sin `--as-of` (o no ISO-8601) la CLI sale con 2 y **no escribe nada**."""
    reports = _copy_artifacts(tmp_path)
    assert phase2_dominance.main(["--reports-dir", str(reports)]) == 2
    assert phase2_dominance.main(["--reports-dir", str(reports), "--as-of", "2026-13-45"]) == 2
    assert phase2_dominance.main(["--reports-dir", str(reports), "--as-of", "el reloj"]) == 2
    assert phase2_dominance.main(["--reports-dir", str(reports), "--as-of", ""]) == 2
    assert not list(reports.glob("phase2_dominance_*"))


def test_a2_ast_forbids_the_clock_and_the_network() -> None:
    """A2: el AST del modulo no lee el reloj ni abre la red."""
    forbidden_attributes = {
        ("datetime", "now"),
        ("datetime", "utcnow"),
        ("datetime", "today"),
        ("date", "today"),
        ("time", "time"),
        ("time", "monotonic"),
    }
    for node in ast.walk(TREE):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert (node.value.id, node.attr) not in forbidden_attributes, ast.dump(node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"now", "utcnow", "today", "time", "monotonic"}
    roots = _import_roots()
    assert roots.isdisjoint({"time", "socket", "urllib", "http", "httpx", "requests"})


def _import_roots() -> set[str]:
    """Los paquetes raiz que importa el modulo."""
    roots: set[str] = set()
    for node in ast.walk(TREE):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@needs_store
def test_a2_write_false_writes_nothing(tmp_path: Path) -> None:
    """A2: `write=False` no deja ningun fichero nuevo (ni en `--reports-dir`)."""
    reports = _copy_artifacts(tmp_path)
    before = {path for path in reports.rglob("*") if path.is_file()}
    with _patched_fast(ours=False):
        report = phase2_dominance.analyse(
            store=Store(REAL_DATA), reports_dir=reports, as_of=NOW, write=False
        )
    after = {path for path in reports.rglob("*") if path.is_file()}
    assert after == before
    assert report.report_sha256.startswith("sha256:")
    assert report.report_stem == STEM
    assert report.series.n_sessions == 500


# ─────────────────────────────────────────────────────────────────────────────
# A3 — artefactos de #28/#26: falta y ambiguedad; consumo en solo lectura
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_missing_and_ambiguous_artifacts_exit_2_without_writing(tmp_path: Path) -> None:
    """A3: sin artefacto, o con dos de la misma fecha, la CLI sale con 2 y no escribe."""
    empty = tmp_path / "vacio" / "derived" / "reports"
    empty.mkdir(parents=True)
    assert phase2_dominance.main(["--reports-dir", str(empty), "--as-of", NOW.isoformat()]) == 2
    assert not list(empty.glob("phase2_dominance_*"))

    ambiguous = tmp_path / "ambiguo"
    reports = _copy_artifacts(ambiguous)
    shutil.copy2(PIPELINE_ARTIFACT, reports / "pipeline_backtest_2026-9-23.json")
    assert phase2_dominance.main(["--reports-dir", str(reports), "--as-of", NOW.isoformat()]) == 2
    assert not list(reports.glob("phase2_dominance_*"))

    broken = tmp_path / "roto" / "derived" / "reports"
    broken.mkdir(parents=True)
    (broken / PIPELINE_ARTIFACT.name).write_text("no es json", encoding="utf-8")
    shutil.copy2(MODEL_ARTIFACT, broken / MODEL_ARTIFACT.name)
    assert phase2_dominance.main(["--reports-dir", str(broken), "--as-of", NOW.isoformat()]) == 2
    assert not list(broken.glob("phase2_dominance_*"))


def test_a3_malformed_artifact_fields_are_typed_errors(tmp_path: Path) -> None:
    """A3: un artefacto mal formado es error tipado y **no** gasta la reejecucion del pipeline."""
    broken_arms = _copy_artifacts(tmp_path / "arms")
    payload = json.loads(PIPELINE_ARTIFACT.read_text(encoding="utf-8"))
    payload["arms"] = []
    (broken_arms / PIPELINE_ARTIFACT.name).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Phase2DominanceError):
        phase2_dominance.analyse(
            store=Store(REAL_DATA),
            reports_dir=broken_arms,
            as_of=datetime(2026, 9, 24),  # sin zona: A2 lo interpreta como UTC
            write=True,
        )
    assert not list(broken_arms.glob("phase2_dominance_*"))

    broken_counts = _copy_artifacts(tmp_path / "counts")
    payload = json.loads(PIPELINE_ARTIFACT.read_text(encoding="utf-8"))
    payload["arms"]["coste_declarado"]["traded"] = "treinta y uno"
    (broken_counts / PIPELINE_ARTIFACT.name).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Phase2DominanceError):
        phase2_dominance.analyse(
            store=Store(REAL_DATA), reports_dir=broken_counts, as_of=NOW, write=True
        )
    assert not list(broken_counts.glob("phase2_dominance_*"))


def test_a3_pipeline_failure_exits_2_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A3: si el pipeline de #28 no se puede reejecutar, la CLI sale con 2 y no escribe."""
    reports = _copy_artifacts(tmp_path)

    def _boom(**_: object) -> object:
        raise pipeline_report.PipelineReportError("el almacen no esta disponible")

    monkeypatch.setattr(pipeline_report, "analyse", _boom)
    assert (
        phase2_dominance.main(
            [
                "--data-root",
                str(REAL_DATA),
                "--reports-dir",
                str(reports),
                "--as-of",
                NOW.isoformat(),
            ]
        )
        == 2
    )
    assert not list(reports.glob("phase2_dominance_*"))


@needs_store
def test_a3_artifacts_are_consumed_read_only(fresh_runs: Mapping[str, CliRun]) -> None:
    """A3: el `sha256` del artefacto de #28 no cambia tras la corrida (solo lectura)."""
    expected = hashlib.sha256(PIPELINE_ARTIFACT.read_bytes()).hexdigest()
    now = _fingerprint(REAL_REPORTS)
    for name, run in fresh_runs.items():
        assert run.reports_fingerprint_before == run.reports_fingerprint_after, name
        assert run.reports_fingerprint_before.get(PIPELINE_ARTIFACT.name) == expected
        assert now == run.reports_fingerprint_after


def test_a3_ast_writes_only_inside_write() -> None:
    """A3: en el AST, escribir solo ocurre dentro de `DominanceReport.write`."""
    span = _write_span()
    mutating = {
        "write_text",
        "write_bytes",
        "mkdir",
        "unlink",
        "remove",
        "touch",
        "rmdir",
    }
    for node in ast.walk(TREE):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in mutating
        ):
            assert span[0] <= node.lineno <= span[1], f"{node.func.attr} fuera de `write`"


def _write_span() -> tuple[int, int]:
    """El rango de lineas de `DominanceReport.write` en el AST."""
    for node in ast.walk(TREE):
        if isinstance(node, ast.ClassDef) and node.name == "DominanceReport":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "write":
                    return item.lineno, cast("int", item.end_lineno)
    raise AssertionError("no se encontro DominanceReport.write en el AST")


# ─────────────────────────────────────────────────────────────────────────────
# A4 — `report_sha256`: formato, determinismo byte a byte y semillas declaradas
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a4_hash_format_and_self_consistency(dominance_report: DominanceReport) -> None:
    """A4: el sha256 lleva el prefijo y es el del payload **sin** la clave del hash."""
    digest = dominance_report.report_sha256
    assert digest.startswith("sha256:")
    body = digest.removeprefix("sha256:")
    assert len(body) == 64
    assert all(character in "0123456789abcdef" for character in body)
    assert "report_sha256" not in dominance_report.payload
    assert (
        hashlib.sha256(canonical_text(dominance_report.payload).encode("utf-8")).hexdigest() == body
    )
    assert str(json.loads(dominance_report.json_text())["report_sha256"]) == digest


@needs_store
def test_a4_identical_across_fresh_processes(
    fresh_runs: Mapping[str, CliRun], dominance_report: DominanceReport
) -> None:
    """A4: mismo `--as-of` ⇒ JSON byte a byte identico con `PYTHONHASHSEED` 0, 1 y aleatorio."""
    ours = dominance_report.json_text().encode("utf-8")
    assert fresh_runs["seed0"].code == 0, fresh_runs["seed0"].stderr
    assert fresh_runs["seed1"].code == 0, fresh_runs["seed1"].stderr
    assert fresh_runs["seed0"].json_bytes == fresh_runs["seed1"].json_bytes == ours
    assert (
        fresh_runs["seed0"].report_sha256
        == fresh_runs["seed1"].report_sha256
        == dominance_report.report_sha256
    )


@needs_store
def test_a4_hash_is_fixed_with_prefix(fresh_runs: Mapping[str, CliRun]) -> None:
    """A4: el digest de un proceso fresco lleva el prefijo, cuadra con el fichero y con su fuente.

    #95: aqui se **fijaba** el literal `GOLDEN_REPORT_SHA256`. Ese digest es el del payload de
    este informe, que publica el `sha256` del artefacto de #28 (regenerable por cualquier tarea
    posterior: #92 lo regenero), asi que no puede ser un dorado. La corrida fresca sigue teniendo
    que publicar un digest con el formato declarado (prefijo `sha256:` + 64 hex, como el de la
    corrida de la sesion), autoconsistente con lo que escribio en disco, y una procedencia cuyo
    `sha256` sea el del **fichero** del artefacto que consume.
    """
    run = fresh_runs["seed0"]
    assert run.report_sha256.startswith("sha256:")
    body = run.report_sha256.removeprefix("sha256:")
    assert len(body) == 64
    assert all(character in "0123456789abcdef" for character in body)

    reports = run.directory / "derived" / "reports"
    published = as_map(json.loads((reports / f"{STEM}.json").read_text(encoding="utf-8")))
    assert published["report_sha256"] == run.report_sha256
    without_hash = {key: value for key, value in published.items() if key != "report_sha256"}
    recomputed = hashlib.sha256(canonical_text(without_hash).encode("utf-8")).hexdigest()
    assert recomputed == body

    provenance = as_map(at(published, "provenance", "pipeline"))
    assert provenance["path"] == PIPELINE_ARTIFACT.name
    assert (
        provenance["sha256"]
        == hashlib.sha256((reports / PIPELINE_ARTIFACT.name).read_bytes()).hexdigest()
    )


@needs_store
def test_a4_bootstrap_is_declared(dominance_report: DominanceReport) -> None:
    """A4: `scenario.bootstrap` publica `n_bootstrap`, `confidence_level` y `seed`."""
    bootstrap = as_map(at(dominance_report.payload, "scenario", "bootstrap"))
    assert bootstrap["n_bootstrap"] == DEFAULT_BOOTSTRAP_SAMPLES
    assert bootstrap["confidence_level"] == 0.95
    assert bootstrap["seed"] == 42
    assert bootstrap["seeds_by_metric"] == {
        "hit_rate": seed_of("hit_rate"),
        "sharpe": seed_of("sharpe"),
    }
    assert as_text(bootstrap, "seed_source").endswith("#15)")


# ─────────────────────────────────────────────────────────────────────────────
# A5 — etiquetado obligatorio y cero metricas netas
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a5_labelling_and_no_net_metrics(dominance_report: DominanceReport) -> None:
    """A5: supuesto declarado, `phase2_ready` siempre falso y ninguna metrica neta."""
    payload = dominance_report.payload
    assert payload["is_measurement"] is False
    assert payload["is_validation"] is False
    assert payload["basis"] == "declared_cost"
    assert payload["phase2_ready"] is False
    net = as_map(payload["net_metrics"])
    assert net["state"] != "computed"
    assert net["computed_here"] is False
    assert net["basis"] == "net"
    assert json.dumps(payload, sort_keys=True).count('"state": "computed"') == 0


def test_a5_ast_does_not_touch_the_net_pnl() -> None:
    """A5: el modulo no lee ni escribe el P&L neto del motor ni publica `computed`."""
    assert "pnl_net_pct" not in SOURCE
    assert "pnl_declared_pct" not in SOURCE
    assert '"state": "computed"' not in SOURCE
    for node in ast.walk(TREE):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"pnl_net_pct", "pnl_declared_pct"}


# ─────────────────────────────────────────────────────────────────────────────
# A6 — el brazo base declarado y sus recuentos **leidos** del artefacto
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a6_base_arm_counts_are_read_from_the_artifact(dominance_report: DominanceReport) -> None:
    """A6: `base_arm: coste_declarado` con 31 / 469 frente a 0 / 500 de los otros dos brazos."""
    artifact = json.loads(PIPELINE_ARTIFACT.read_text(encoding="utf-8"))
    base_arm = as_map(at(dominance_report.payload, "base_arm"))
    assert base_arm["name"] == "coste_declarado"
    assert base_arm["basis"] == "declared_cost"
    assert base_arm["is_validation"] is False
    assert base_arm["reason"]
    counts = as_map(base_arm["counts"])
    expected = as_map(at(artifact, "arms", "coste_declarado"))
    assert counts["traded"] == 31
    assert counts["no_trade"] == 469
    assert counts["traded"] == expected["traded"]
    assert counts["no_trade"] == expected["no_trade"]
    others = as_map(base_arm["other_arms"])
    for name in ("oficial", "escenario"):
        other = as_map(others[name])
        assert other["traded"] == 0
        assert other["no_trade"] == 500
    assert as_map(base_arm["rederived_counts"])["matches_artifact"] is True


# ─────────────────────────────────────────────────────────────────────────────
# A7 — la rejilla declarada de *slippage*: ascendente, con `0` exacto y procedencia #8
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a7_grid_is_declared_ascending_and_from_the_eight_report(
    dominance_report: DominanceReport,
) -> None:
    """A7: primer elemento exactamente `0`, ascendente, y con los tres percentiles de #8."""
    scenario = as_map(at(dominance_report.payload, "scenario"))
    grid = tuple(cast("list[float]", scenario["slippage_grid_bp"]))
    assert grid == GOLDEN_GRID_BP
    assert grid[0] == 0.0
    assert all(left < right for left, right in itertools.pairwise(grid))
    assert scenario["is_measurement"] is False
    provenance = as_map(scenario["grid_provenance"])
    assert provenance["issue"] == "#8"
    assert provenance["is_measurement"] is False
    assert provenance["sessions_measured"] == 59
    assert provenance["median_bp"] == 13.2
    assert provenance["p90_bp"] == 26.6
    assert provenance["max_bp"] == 46.8
    assert "no" in as_text(provenance, "why_not_a_measurement")
    labels = cast("list[str]", scenario["slippage_grid_labels"])
    assert slippage_levels() == tuple(zip(labels, grid, strict=True))


def test_a7_ast_does_not_wire_the_grid_nor_p_star() -> None:
    """A15: el modulo **deriva** la rejilla (#8) y `p*` (#9): no cablea sus valores."""
    numbers = {
        node.value
        for node in ast.walk(TREE)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    }
    assert numbers.isdisjoint({50.2, 50.21, 0.502, 0.5021})
    assert numbers.isdisjoint({13.2, 26.6, 46.8})
    called = {
        node.func.id
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {"slippage_assumption_block", "p_star_block"} <= called
    assert "median_bp" in SOURCE and "p90_bp" in SOURCE and "max_bp" in SOURCE


def test_a7_non_ascending_grid_is_a_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A7: una rejilla que no arranca en `0` o no sube es un error tipado, no un aviso."""

    def _bad() -> dict[str, object]:
        return {
            "measured_evidence": {
                "median_bp": 13.2,
                "p90_bp": 26.6,
                "max_bp": 13.2,
            }
        }

    monkeypatch.setattr(phase2_dominance, "slippage_assumption_block", _bad)
    with pytest.raises(InvalidSlippageGridError):
        slippage_levels()


# ─────────────────────────────────────────────────────────────────────────────
# A8 — unidades: `1 bp = 0,01 %`, una vez por sesion operada, sin tocar las que no operan
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_shift_subtracts_only_traded_sessions() -> None:
    """A8: el *slippage* se resta una vez por sesion **operada**; las demas quedan en `0` exacto."""
    series = _series([1.0, 0.0, -2.0, 0.0], traded=(True, False, True, False))
    assert series.shifted(slippage_bp=0.0) == series.values_pct
    assert series.shifted(slippage_bp=13.2) == (1.0 - 0.132, 0.0, -2.0 - 0.132, 0.0)
    assert phase2_dominance.BP_TO_PCT == 0.01
    assert series.n_traded == 2
    assert series.n_sessions == 4


@needs_store
def test_a8_units_block_declares_the_conversion(dominance_report: DominanceReport) -> None:
    """A8: el bloque `units` declara la conversion y el alcance del desplazamiento."""
    units = as_map(at(dominance_report.payload, "scenario", "units"))
    assert units["bp_to_pct"] == 0.01
    assert units["conversion"] == "1 bp = 0,01 % del nocional"
    assert "sesion operada" in str(units["applied"])
    assert "se tocan" in str(units["applied"])
    assert units["n_traded"] == 31
    assert units["n_no_trade"] == 469


# ─────────────────────────────────────────────────────────────────────────────
# A9 — la celda de `0` bp reproduce **exactamente** el artefacto de #28
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a9_zero_cell_reproduces_the_artifact(dominance_report: DominanceReport) -> None:
    """A9: estimacion, `lower` y `upper` de las dos mitades, con la semilla declarada de #28."""
    artifact = json.loads(PIPELINE_ARTIFACT.read_text(encoding="utf-8"))
    published = as_map(at(artifact, "arms", "coste_declarado", "metrics"))
    cell = cells_of(dominance_report)[0]
    assert cell["slippage_bp"] == 0.0
    assert cell["is_base"] is True
    for metric in ("hit_rate", "sharpe"):
        expected = as_map(published[metric])
        found = as_map(cell[metric])
        for key in METRIC_KEYS:
            assert found[key] == expected[key], f"{metric}.{key}"
        assert expected["seed"] == seed_of(metric)
        assert as_map(cell["seed"])[metric] == seed_of(metric)
    assert seed_of("hit_rate") == 44
    assert seed_of("sharpe") == 45
    reproduction = as_map(dominance_report.payload["reproduction"])
    assert reproduction["checked"] is True
    assert reproduction["reproduces"] is True
    assert reproduction["artifact_sha256"] == (
        "sha256:" + hashlib.sha256(PIPELINE_ARTIFACT.read_bytes()).hexdigest()
    )


@needs_store
def test_a9_reproduction_mismatch_is_a_typed_error(tmp_path: Path) -> None:
    """A9: si la reconstruccion no cuadra con el artefacto, error tipado y **nada** escrito."""
    reports = _copy_artifacts(tmp_path)
    broken = json.loads(PIPELINE_ARTIFACT.read_text(encoding="utf-8"))
    broken["arms"]["coste_declarado"]["metrics"]["hit_rate"]["estimate"] = 0.99
    (reports / PIPELINE_ARTIFACT.name).write_text(json.dumps(broken), encoding="utf-8")
    with _patched_fast(ours=True):  # noqa: SIM117 - el doble envuelve a la corrida entera
        with pytest.raises(ReproductionMismatchError):
            phase2_dominance.analyse(
                store=Store(REAL_DATA), reports_dir=reports, as_of=NOW, write=True
            )
    assert not list(reports.glob("phase2_dominance_*"))


@needs_store
def test_a9_reproduction_is_skipped_for_a_degenerate_base(
    dominance_report: DominanceReport,
) -> None:
    """A9: sin celdas (base degenerada) la reproduccion se declara **no comprobada**."""
    block = phase2_dominance._reproduction_block(  # pyright: ignore[reportPrivateUsage]
        payload=dominance_report.pipeline.payload,
        artifact=dominance_report.pipeline,
        cells=(),
    )
    assert block["checked"] is False
    assert "metrics" not in block
    assert "reproduces" not in block
    assert as_text(block, "artifact_sha256").startswith("sha256:")


# ─────────────────────────────────────────────────────────────────────────────
# A10 — campos de cada celda y banderas globales de cruce
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a10_cell_fields_and_global_crossing_flags(dominance_report: DominanceReport) -> None:
    """A10: cada celda publica sus metricas, `state`, `code` y banderas; el informe, el cruce."""
    payload = dominance_report.payload
    cells = cells_of(dominance_report)
    assert len(cells) == len(GOLDEN_GRID_BP)
    for cell in cells:
        assert set(CELL_KEYS) <= set(cell)
        for metric in ("hit_rate", "sharpe"):
            assert set(METRIC_KEYS) <= set(as_map(cell[metric]))
        assert cell["state"] in {"crosses", "fail", "not_evaluable"}
        assert isinstance(cell["code"], str) and cell["code"]
        assert cell["crosses"] is (as_float(cell, "sharpe", "lower") > 0.0)
        assert cell["sharpe_excludes_zero_above"] is cell["crosses"]
        assert cell["sharpe_below"] is (as_float(cell, "sharpe", "upper") < 0.0)
    crossed = cast("list[float]", payload["crossed_cells"])
    assert crossed == [cell["slippage_bp"] for cell in cells if cell["crosses"]]
    assert payload["no_cell_crosses"] is (not crossed)
    assert as_map(payload["dominance"])["crossed_cells"] == crossed


# ─────────────────────────────────────────────────────────────────────────────
# A11 — la mitad del Sharpe decide, en sus tres casos
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_three_cases_of_the_sharpe_half() -> None:
    """A11: `crosses` ⇔ `sharpe.lower > 0`; `fail` ⇔ `sharpe.upper < 0`; si no, `not_evaluable`."""
    crossing = _dominance(_positive_series())
    assert [cell["state"] for cell in crossing.cells] == ["crosses", "crosses"]
    assert all(cell["crosses"] for cell in crossing.cells)

    failing = _dominance(_negative_series())
    assert all(cell["state"] == "fail" for cell in failing.cells)
    for cell in failing.cells:
        assert cell["crosses"] is False
        assert cell["sharpe_below"] is True
        assert as_float(cell, "sharpe", "upper") < 0.0
        assert cell["state"] != "pass"
    assert failing.state == "fail"
    assert failing.code == CODE_FAIL

    flat = _dominance(_flat_series())
    base = flat.cells[0]
    assert base["state"] == "not_evaluable"
    assert base["crosses"] is False
    assert base["sharpe_below"] is False
    assert as_float(base, "sharpe", "lower") < 0.0 < as_float(base, "sharpe", "upper")
    assert all(cell["state"] in {"crosses", "fail", "not_evaluable"} for cell in flat.cells)
    assert flat.state == "not_evaluable"
    assert flat.code == CODE_INCONCLUSIVE


# ─────────────────────────────────────────────────────────────────────────────
# A12 — el veredicto: `fail` solo sin cruces; cualquier cruce ⇒ `not_evaluable`
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_verdict_fail_and_not_evaluable_branches() -> None:
    """A12: `fail` ⇔ la celda de `0` bp falla y ninguna cruza; si alguna cruza, `not_evaluable`."""
    failing = _dominance(_negative_series())
    assert failing.no_cell_crosses is True
    assert failing.crossed_cells == ()
    assert failing.state == "fail"
    assert failing.reason

    with_crossing = _dominance(_positive_series())
    assert with_crossing.no_cell_crosses is False
    assert with_crossing.crossed_cells == (0.0, 13.2)
    assert with_crossing.state == "not_evaluable"
    assert with_crossing.state not in {"pass", "continue"}
    block = with_crossing.as_dict()
    assert block["state"] == "not_evaluable"
    assert "`pass` y `continue` no se publican nunca" in str(block["note"])


# ─────────────────────────────────────────────────────────────────────────────
# A13 — un `crosses` en la propia base declarada **no** es aprobar
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_a_crossing_base_is_not_approval() -> None:
    """A13: la base declarada que cruza se publica `not_evaluable` con su `code` propio."""
    dominance = _dominance(_positive_series())
    base = dominance.cells[0]
    assert base["is_base"] is True
    assert base["crosses"] is True
    assert dominance.state == "not_evaluable"
    assert dominance.code == CODE_BASE_CROSSES
    assert "no demuestra la base neta" in dominance.reason
    assert dominance.as_dict()["state"] != "pass"


# ─────────────────────────────────────────────────────────────────────────────
# A14 — serie degenerada: `not_evaluable`, sin numeros fabricados
# ─────────────────────────────────────────────────────────────────────────────
def test_a14_degenerate_base_never_fails_nor_passes() -> None:
    """A14: 0 operaciones o todo ceros ⇒ `not_evaluable`, sin celdas ni metricas inventadas."""
    for series in (_series([0.0] * 5, traded=(False,) * 5), _series([])):
        dominance = _dominance(series)
        assert dominance.state == "not_evaluable"
        assert dominance.code == CODE_DEGENERATE
        assert dominance.cells == ()
        assert dominance.crossed_cells == ()
        assert dominance.no_cell_crosses is None
        assert dominance.check["violated"] is None
        block = dominance.as_dict()
        assert block["state"] not in {"fail", "pass"}
        assert "degenerada" in str(block["reason"])


# ─────────────────────────────────────────────────────────────────────────────
# A15 — `p*` derivado por escenario de `R`, con el **vinculante**
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a15_binding_p_star_is_the_most_demanding(dominance_report: DominanceReport) -> None:
    """A15: se publica `p*` por escenario de `R` y se usa el que mas exige."""
    stars = as_map(dominance_report.payload["p_star"])
    scenarios = [as_map(item) for item in cast("list[object]", stars["scenarios"])]
    binding = as_map(stars["binding"])
    fractions = [Decimal(str(item["p_star_fraction"])) for item in scenarios]
    assert Decimal(str(binding["p_star_fraction"])) == max(fractions)
    assert len(scenarios) >= 3
    for cell in cells_of(dominance_report):
        assert as_map(cell["hit_rate"])["p_star_fraction"] == str(binding["p_star_fraction"])
        assert as_map(cell["hit_rate"])["excludes_p_star_above"] is (
            Decimal(str(as_map(cell["hit_rate"])["lower"]))
            > Decimal(str(binding["p_star_fraction"]))
        )
    assert Decimal(str(binding["p_star_fraction"])) > Decimal("0.5")


# ─────────────────────────────────────────────────────────────────────────────
# A16 — la tabla §11.6 se lee del documento y la fila principal se sustituye
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a16_nine_rows_from_the_document_and_main_row_replaced(
    dominance_report: DominanceReport,
) -> None:
    """A16: `sha256` y 9 filas de §11.6, fila principal por dominancia y agregado con 9 filas."""
    payload = dominance_report.payload
    source = as_map(payload["criteria_source"])
    assert source["path"] == "_docs/plan.md"
    assert source["section"] == "§11.6"
    assert source["n_rows"] == 9
    assert str(source["sha256"])
    assert len(cast("list[object]", payload["kill_criteria"])) == 9
    rows = [as_map(item) for item in cast("list[object]", payload["criteria"])]
    assert len(rows) == 9
    main = rows[0]
    table = phase2_dominance.load_kill_table()
    assert main["kind"] == table.rows[0].kind
    assert main["source_row"] == table.rows[0].criterion
    assert main["threshold"] == table.rows[0].threshold
    assert main["action"] == table.rows[0].action
    assert as_text(main, "evaluated_by").endswith("(#93)")
    dominance = as_map(payload["dominance"])
    assert main["state"] == dominance["state"]
    assert main["code"] == dominance["code"]
    assert main["state"] in {"fail", "not_evaluable"}
    gate = as_map(payload["gate"])
    assert gate["n_rows"] == 9
    assert "aggregate_gate" in str(gate["aggregation_source"])
    verdict = as_map(payload["verdict"])
    assert verdict["state"] in {"stop", "simplify", "not_evaluable"}
    assert verdict["consistent"] is True


def test_a16_ast_reuses_the_nine_rows_machinery() -> None:
    """A16: la tabla y la agregacion se **importan** de #9/#29, no se reescriben."""
    imported = {
        alias.name
        for node in ast.walk(TREE)
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    }
    assert {"load_kill_table", "gate_block", "resolve_verdict", "evaluate_criteria"} <= imported
    called = {
        node.func.id
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {"load_kill_table", "gate_block", "evaluate_criteria", "resolve_verdict"} <= called
    assert "GATE_AGGREGATION_SOURCE" not in SOURCE


# ─────────────────────────────────────────────────────────────────────────────
# A17 — el calculo rejilla x serie -> celdas -> estado es una funcion pura
# ─────────────────────────────────────────────────────────────────────────────
def test_a17_compute_dominance_is_pure_and_checks_monotonicity() -> None:
    """A17: funcion pura, sin pipeline, con `dominance_check` de monotonia."""
    signature = inspect.signature(compute_dominance)
    assert set(signature.parameters) == {
        "series",
        "levels",
        "p_star_fraction",
        "n_bootstrap",
        "confidence_level",
    }
    series = _series([0.2, -0.1, 0.3, -0.2] * 5)
    first = _dominance(series)
    second = _dominance(series)
    assert first == second
    check = first.check
    assert check["mean_non_increasing"] is True
    assert check["hit_rate_non_increasing"] is True
    assert check["violated"] is False
    assert check["n_steps"] == len(first.cells) - 1
    for step in cast("list[Mapping[str, object]]", check["steps"]):
        assert float(cast("float", step["delta_mean_return_pct"])) <= 0.0
        assert float(cast("float", step["delta_hit_rate"])) <= 0.0
    assert check["error_type"] == "DominanceViolationError"
    assert "un *slippage* es un coste" in str(check["rule"]) or "no" in str(check["rule"])


def test_a17_a_grid_that_improves_the_series_is_a_typed_error() -> None:
    """A17: si el escenario **mejora** la media o la tasa de acierto, error tipado."""
    levels = (("sin_slippage", 0.0), ("rebaja_hipotetica", -5.0))
    with pytest.raises(DominanceViolationError):
        _dominance(_flat_series(), levels=levels)


# ─────────────────────────────────────────────────────────────────────────────
# A18 — sin dependencias nuevas y sin tocar los ficheros ajenos
# ─────────────────────────────────────────────────────────────────────────────
def test_a18_no_new_dependency() -> None:
    """A18: solo biblioteca estandar, `loguru` y el propio paquete."""
    assert _import_roots() <= {
        "__future__",
        "argparse",
        "hashlib",
        "itertools",
        "json",
        "math",
        "sys",
        "collections",
        "dataclasses",
        "datetime",
        "decimal",
        "pathlib",
        "typing",
        "loguru",
        "cfdtrader",
    }


def test_a18_frozen_modules_are_untouched() -> None:
    """A18: la entrega trae sus ficheros y **no** toca ningun modulo congelado.

    #95: antes exigia `changed <= ALLOWED_PATHS`, es decir que **todo** el cambio del repositorio
    desde `BASE_COMMIT` cupiera en los ficheros de #93. Cualquier tarea posterior que toque otro
    fichero rompia la suite. La intencion de A18 es subconjunto (los ficheros de #93 estan) y
    disyuncion (ningun congelado esta), como en #26 y en la relajacion A15 de #89.
    """
    git = shutil.which("git")
    if git is None:  # pragma: no cover - entorno sin git
        pytest.skip("git no disponible")
    committed = _git_lines(git, ["diff", "--name-only", f"{BASE_COMMIT}..HEAD"])
    working = {
        line.strip().split(maxsplit=1)[1] if len(line.strip().split(maxsplit=1)) > 1 else line
        for line in _git_lines(git, ["status", "--porcelain"])
        if line.strip()
    }
    changed = committed | working
    assert set(WRITTEN) <= changed, f"la entrega no trae: {sorted(set(WRITTEN) - changed)}"
    assert changed.isdisjoint(FROZEN), f"modulos congelados tocados: {sorted(changed & FROZEN)}"


def _git_lines(git: str, arguments: Sequence[str]) -> set[str]:
    """Las lineas no vacias de un comando git en la raiz del repositorio."""
    result = subprocess.run(  # noqa: S603 - git local del repositorio
        [git, "-C", str(REPO_ROOT), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


# ─────────────────────────────────────────────────────────────────────────────
# Bordes declarados: reconstruccion de la serie y errores tipados
# ─────────────────────────────────────────────────────────────────────────────
def _outcome(
    status: str,
    *,
    gross_pct: float | None,
    cost_pct: str,
    session: date = date(2026, 9, 1),
) -> object:
    """Una sesion sintetica del motor con lo que la reconstruccion necesita."""
    return SimpleNamespace(
        status=status,
        session=session,
        gross_pct=gross_pct,
        cost=None if cost_pct == "none" else SimpleNamespace(c_declared_pct=Decimal(cost_pct)),
    )


def _run(outcomes: Sequence[object]) -> BacktestRun:
    """Una corrida sintetica del motor con un solo fold."""
    return cast("BacktestRun", SimpleNamespace(folds=[SimpleNamespace(sessions=list(outcomes))]))


def test_edge_declared_series_reconstructs_the_three_statuses() -> None:
    """La serie declarada: `0` exacto en `no_trade`, fuera los `skipped` y el coste en `%`."""
    run = _run(
        [
            _outcome(STATUS_TRADED, gross_pct=0.01, cost_pct="0.0042"),
            _outcome(STATUS_NO_TRADE, gross_pct=None, cost_pct="none"),
            _outcome(STATUS_SKIPPED, gross_pct=None, cost_pct="none"),
            _outcome(STATUS_TRADED, gross_pct=-0.005, cost_pct="0.0042"),
        ]
    )
    series = declared_series_of(run)
    assert series.values_pct == (1.0 - 0.0042, 0.0, -0.5 - 0.0042)
    assert series.traded == (True, False, True)
    assert series.n_sessions == 3
    assert series.n_traded == 2
    assert series.digest().startswith("sha256:")
    assert series.as_dict()["n_no_trade"] == 1
    assert series.as_dict()["all_zero"] is False


def test_edge_declared_series_without_pnl_is_a_typed_error() -> None:
    """Una operacion sin P&L declarado no se rellena: error tipado."""
    run = _run([_outcome(STATUS_TRADED, gross_pct=None, cost_pct="0.0042")])
    with pytest.raises(Phase2DominanceError):
        declared_series_of(run)


def test_edge_declared_series_mismatch_is_a_typed_error() -> None:
    """Una serie y su mascara con longitudes distintas no se aceptan."""
    with pytest.raises(Phase2DominanceError):
        DeclaredSeries(values_pct=(1.0, 2.0), traded=(True,))


def test_edge_seed_of_an_unknown_metric_is_a_typed_error() -> None:
    """Una metrica fuera de `METRIC_NAMES` no tiene semilla declarada."""
    with pytest.raises(Phase2DominanceError):
        seed_of("metrica_inventada")


@needs_store
def test_edge_the_store_is_only_read() -> None:
    """La reejecucion del pipeline no escribe en el almacen real (solo lectura)."""
    history = load_history(Store(REAL_DATA))
    assert history.daily.height > 0
    assert not (REAL_DATA / "phase2_dominance_2026-09-24.json").exists()
