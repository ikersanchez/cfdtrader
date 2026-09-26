"""Tests del informe de comparacion de modelos (#26): A1, A2, A3 y A6-A14.

Los criterios publican **numeros reales**, asi que la corrida cara se hace sobre el almacen del
repositorio en **solo lectura** (la fixture de sesion de `tests/conftest.py` huella el `data/`
y el `runs/` antes y despues). Todo lo que se escribe va a `tmp_path`: el registro de la
corrida es una **copia** de las dos entradas congeladas de la familia lineal (#24/#25) mas las
dos entradas nuevas de LightGBM.

El determinismo entre procesos (A13) se mide con la CLI en procesos nuevos y `PYTHONHASHSEED`
0 / 1 / random, comparando `report_sha256`, los bytes de `.json`/`.md` y los de los dos
`model.json` nuevos.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pytest

from cfdtrader.analysis import model_comparison
from cfdtrader.analysis.backtest_report import PHASE1_PLAN
from cfdtrader.analysis.experiment_log import (
    Registry,
    RegistryEntry,
    TrialsMismatchError,
    deflate_block,
    load_registry,
)
from cfdtrader.analysis.model_comparison import (
    BASELINE_VARIANT_ID,
    FAMILY_ORDER,
    FROZEN_BASELINE,
    FROZEN_TOLERANCE,
    PBO_BLOCKS,
    PRIMARY_METRIC,
    REPORT_PREFIX,
    SELECTION_RULE,
    TIE_BREAKERS,
    VARIANT_ID,
    ModelComparisonError,
    ModelComparisonReport,
    analyse,
    main,
    render_markdown,
    require_matrix_matches_registry,
    selection_block,
)
from cfdtrader.backtest.costs import declared_cost_model, declared_slippage_assumption
from cfdtrader.data.settings import Settings
from cfdtrader.data.store import Store
from cfdtrader.models import lightgbm_model
from cfdtrader.models.baseline import BASELINE_FEATURES, DECISION_THRESHOLD, SEED
from cfdtrader.models.lightgbm_model import fit_lightgbm

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"
REPO_RUNS: Final[Path] = REPO_ROOT / "runs"
FROZEN_REPORTS: Final[Path] = REAL_DATA / "derived" / "reports"

#: Instante **declarado** de todas las corridas: el modulo nunca lee el reloj (A2).
NOW: Final[datetime] = datetime(2026, 9, 22, 22, 0, tzinfo=UTC)

#: Base de la entrega: `main == origin/main` al empezar la tarea (#26).
BASE_COMMIT: Final[str] = "9cb5068"

#: Ficheros que la entrega **no** puede tocar (A1, A12). #80 retira `engine.py` y `costs.py`:
#: la entrega transversal de #80 **si** toca el motor (es su arreglo) y la guardia de `git diff`
#: se reduce a los modulos que siguen congelados (A8).
UNTOUCHABLE: Final[tuple[str, ...]] = (
    "src/cfdtrader/models/baseline.py",
    "src/cfdtrader/models/calibration.py",
)

#: Los dos modulos nuevos y sus tests (A1, A14).
NEW_MODULES: Final[tuple[Path, ...]] = (
    REPO_ROOT / "src" / "cfdtrader" / "models" / "lightgbm_model.py",
    REPO_ROOT / "src" / "cfdtrader" / "analysis" / "model_comparison.py",
)
NEW_TESTS: Final[tuple[Path, ...]] = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parent / "test_lightgbm_model.py",
)

#: Los literales de coste de #8/#11: el modulo los **importa**, no los redeclara (A3).
COST_LITERALS: Final[frozenset[float]] = frozenset(
    {0.42, 0.0042, 1.82, 0.0182, -0.18, -0.0018, 0.0024, 0.0224}
)

#: Los cuatro numeros de la tabla, tal como se midieron en la corrida (A12).
MEASURED_BRIER: Final[tuple[float, ...]] = (
    0.2511418117147099,
    0.2564548794817189,
    0.2548043292843032,
    0.27316798097893297,
)

#: Huella de los informes congelados **antes** de que corra ninguna prueba (A7).
FROZEN_REPORTS_BEFORE: Final[dict[str, str]] = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(FROZEN_REPORTS.glob("baseline_2026-09-*"))
}

needs_store = pytest.mark.skipif(
    not (REAL_DATA / "derived" / "labels").exists(),
    reason="el almacen real no esta en el arbol: los numeros de A6/A7 son los suyos",
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def _copy_frozen_runs(destination: Path) -> tuple[str, ...]:
    """Copia a `destination` las entradas del registro de la familia lineal (las congeladas)."""
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for child in sorted(REPO_RUNS.iterdir()):
        if not child.is_dir():
            continue
        document = cast(
            "dict[str, object]",
            json.loads((child / "config.json").read_text(encoding="utf-8")),
        )
        config = cast("Mapping[str, object]", document["config"])
        if config["variant_id"] != BASELINE_VARIANT_ID:
            continue
        shutil.copytree(child, destination / child.name, dirs_exist_ok=True)
        copied.append(child.name)
    return tuple(copied)


def _block(report: ModelComparisonReport, *keys: str) -> dict[str, object]:
    """Un bloque anidado del payload, ya tipado."""
    node: dict[str, object] = report.payload
    for key in keys:
        node = cast("dict[str, object]", node[key])
    return node


def _rows(report: ModelComparisonReport) -> list[dict[str, object]]:
    """Las cuatro filas de la tabla comparativa."""
    return [
        cast("dict[str, object]", row)
        for row in cast("list[object]", _block(report, "comparison")["rows"])
    ]


def _variants(report: ModelComparisonReport) -> list[dict[str, object]]:
    """Las filas de `variants[]`, en el orden del registro."""
    return [
        cast("dict[str, object]", item) for item in cast("list[object]", report.payload["variants"])
    ]


def _cli(*arguments: str, seed: str | None = None) -> subprocess.CompletedProcess[str]:
    """Ejecuta el CLI en un **proceso nuevo**, con su `PYTHONHASHSEED` (A13)."""
    environment = dict(os.environ)
    if seed is not None:
        environment["PYTHONHASHSEED"] = seed
    return subprocess.run(  # noqa: S603 - el ejecutable es el interprete de la sesion
        [sys.executable, "-m", "cfdtrader.analysis.model_comparison", *arguments],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _report_hash(path: Path) -> str:
    """El `report_sha256` publicado en el JSON del informe."""
    document = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    return str(document["report_sha256"])


def _source(module: object) -> str:
    """El fuente de un modulo, para las comprobaciones por AST."""
    return Path(cast("Any", module).__file__).read_text(encoding="utf-8")


def _tree(module: object) -> ast.Module:
    """El AST de un modulo."""
    return ast.parse(_source(module))


def _imported_roots(tree: ast.Module) -> set[str]:
    """Las raices de los modulos importados (`x.y` -> `x`)."""
    return {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }


def _imported_modules(tree: ast.Module) -> set[str]:
    """Los modulos importados completos (`a.b` incluido)."""
    found = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    return {name for name in found if name}


def _numeric_literals(tree: ast.Module) -> list[float]:
    """Los literales numericos del modulo (A3: el coste no se redeclara)."""
    return [
        float(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ]


def _git(*arguments: str) -> str:
    """Salida de `git`, sin paginacion."""
    completed = subprocess.run(  # noqa: S603 - el git del sistema, comando fijo
        ["git", "--no-pager", *arguments],  # noqa: S607 - el git del sistema
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _synthetic_series(
    observations: int = 500, *, mean: float = 0.0015, sigma: float = 0.01, seed: int = SEED
) -> tuple[float, ...]:
    """Una serie determinista con Sharpe **positivo**, para hacer observable la deflacion (A9).

    `numpy.random.RandomState` congela el flujo entre versiones. Con las series reales de esta
    corrida (Sharpe negativo) el DSR se satura a `0.0` en cualquier registro, asi que no
    distinguiria tres intentos de diez.
    """
    state = np.random.RandomState(seed)
    return tuple(float(value) for value in state.normal(loc=mean, scale=sigma, size=observations))


# ─────────────────────────────────────────────────────────────────────────────
# Corridas de la suite
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def base_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """La raiz de trabajo de la suite: el registro con las dos entradas congeladas copiadas."""
    root = tmp_path_factory.mktemp("model_comparison")
    _copy_frozen_runs(root / "runs")
    return root


@pytest.fixture(scope="session")
def real_report(base_root: Path) -> ModelComparisonReport:
    """La corrida real completa, **una vez** por sesion, escribiendo en un directorio temporal."""
    return analyse(
        store=Store(REAL_DATA),
        reports_dir=base_root / "reports",
        runs_root=base_root / "runs",
        settings=Settings(),
        as_of=NOW,
        write=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# A1 - ficheros y fronteras
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_both_modules_and_their_tests_exist() -> None:
    """Los dos modulos nuevos y sus dos suites estan en el arbol (A1)."""
    for path in (*NEW_MODULES, *NEW_TESTS):
        assert path.is_file(), f"falta {path} (A1)"


def test_a1_the_pure_module_does_not_reach_the_store_or_the_report_layers() -> None:
    """`lightgbm_model` no importa `data`/`analysis`/`backtest` ni `duckdb` (A1, AST)."""
    tree = _tree(lightgbm_model)
    roots = _imported_roots(tree)
    assert "duckdb" not in roots
    modules = _imported_modules(tree)
    forbidden = sorted(
        name
        for name in modules
        if name.startswith(("cfdtrader.data", "cfdtrader.analysis", "cfdtrader.backtest"))
    )
    assert not forbidden, f"el modulo puro no importa {forbidden} (A1)"


def test_a1_only_the_report_module_reads_the_store_and_writes() -> None:
    """Solo `model_comparison` lee el almacen y escribe: el modelo no toca disco (A1)."""
    source = _source(lightgbm_model)
    for forbidden in (".sql(", "write_text", "write_bytes", "mkdir", "Store("):
        assert forbidden not in source, f"el modulo puro no puede hacer `{forbidden}` (A1)"
    report_tree = _tree(model_comparison)
    attributes = {node.attr for node in ast.walk(report_tree) if isinstance(node, ast.Attribute)}
    called = {
        node.func.id
        for node in ast.walk(report_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    # El informe no escribe `store.sql` a mano: lee el almacen por los **lectores** de #24/#69
    # (`load_history` y `build_feature_frame`), que son los que hablan con el `Store`. Lo que la
    # frontera exige es que **solo** este modulo lea y escriba (A1).
    assert {"load_history", "build_feature_frame"} <= called
    assert "write_text" in attributes


def test_a1_the_delivery_does_not_touch_the_frozen_files() -> None:
    """`git diff 9cb5068..HEAD` no toca los cuatro ficheros congelados ni las dos `runs/` (A1)."""
    changed = set(_git("diff", "--name-only", f"{BASE_COMMIT}..HEAD").splitlines())
    touched = sorted(name for name in UNTOUCHABLE if name in changed)
    assert not touched, f"la entrega no puede tocar {touched} (A1)"
    runs = sorted(name for name in changed if name.startswith("runs/"))
    assert not runs, f"la entrega no puede tocar `runs/`: {runs} (A1)"


# ─────────────────────────────────────────────────────────────────────────────
# A2 - CLI sin reloj
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_cli_without_as_of_exits_two_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sin `--as-of`: exit 2, motivo en `stderr` y **cero** ficheros (A2)."""
    assert main([]) == 2
    captured = capsys.readouterr()
    assert "--as-of" in captured.err
    assert list(tmp_path.iterdir()) == []


def test_a2_the_signature_and_the_absence_of_the_clock() -> None:
    """La firma de `analyse` es la declarada y ningun modulo nuevo lee el reloj (A2)."""
    import inspect

    parameters = inspect.signature(analyse).parameters
    assert list(parameters) == [
        "store",
        "reports_dir",
        "runs_root",
        "settings",
        "as_of",
        "write",
        "previous_artifact",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values()
    )
    assert parameters["write"].default is True
    # #90: la entrada opcional "artefacto previo" es contrato declarado y no viaja por defecto.
    assert parameters["previous_artifact"].default is None
    for module in (lightgbm_model, model_comparison):
        # El reloj se busca en el **AST**, no en el texto: los dos modulos **declaran** por
        # escrito que no lo leen, y una asercion sobre el fuente se encontraria a si misma.
        found = [
            ast.unparse(node)
            for node in ast.walk(_tree(module))
            if (isinstance(node, ast.Attribute) and node.attr in {"now", "utcnow", "today"})
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "time"
            )
        ]
        assert not found, f"{module.__name__} no puede leer el reloj (A2): {found}"


@needs_store
def test_a2_write_false_does_not_write_anything(tmp_path: Path) -> None:
    """`write=False` no escribe **nada**: ni el informe ni las carpetas del registro (A2)."""
    runs = tmp_path / "runs"
    _copy_frozen_runs(runs)
    before = sorted(path.name for path in runs.iterdir())
    reports = tmp_path / "reports"
    report = analyse(
        store=Store(REAL_DATA),
        reports_dir=reports,
        runs_root=runs,
        settings=Settings(),
        as_of=NOW,
        write=False,
    )
    assert report.report_sha256
    assert not reports.exists()
    assert sorted(path.name for path in runs.iterdir()) == before


# ─────────────────────────────────────────────────────────────────────────────
# A3 - protocolo importado, no redeclarado
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a3_the_protocol_is_imported_and_matches_the_frozen_baseline(
    real_report: ModelComparisonReport,
) -> None:
    """El payload publica el protocolo importado, identico al de la linea base (A3)."""
    protocol = _block(real_report, "protocol")
    frozen = cast(
        "dict[str, object]",
        json.loads((FROZEN_REPORTS / "baseline_2026-09-22.json").read_text(encoding="utf-8")),
    )
    frozen_plan = cast("dict[str, object]", frozen["plan"])
    assert protocol["plan_sha256"] == frozen_plan["plan_sha256"]
    assert protocol["plan_sha256"] == real_report.split_plan.plan_sha256
    assert protocol["n_folds"] == 10
    assert protocol["n_test"] == 500
    assert protocol["features"] == list(BASELINE_FEATURES)
    assert protocol["decision_threshold"] == DECISION_THRESHOLD == 0.5
    assert protocol["calibration_bins"] == 5
    cost = cast("dict[str, object]", protocol["cost"])
    assert cost["basis"] == "declared_cost"
    assert protocol["plan_params"] == {
        "n_splits": PHASE1_PLAN.n_splits,
        "test_size": PHASE1_PLAN.test_size,
        "embargo_sessions": PHASE1_PLAN.embargo_sessions,
        "max_train_size": PHASE1_PLAN.max_train_size,
        "label_horizon": PHASE1_PLAN.label_horizon,
    }


def test_a3_the_module_does_not_redeclare_plan_threshold_bins_or_cost() -> None:
    """El modulo importa el plan, el umbral, los bins y el coste: no los reescribe (A3)."""
    source = _source(model_comparison)
    tree = _tree(model_comparison)
    for literal in sorted(COST_LITERALS):
        assert literal not in _numeric_literals(tree), (
            f"el modulo no puede contener el literal de coste {literal} (A3)"
        )
    assert "from cfdtrader.analysis.backtest_report import" in source
    assert "PHASE1_PLAN" in source
    assert "DECISION_THRESHOLD" in source
    assert "CALIBRATION_BINS" in source
    assert "declared_cost_model()" in source
    assert "declared_slippage_assumption()" in source
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "PHASE1_PLAN" not in assigned
    assert "DECISION_THRESHOLD" not in assigned
    assert "CALIBRATION_BINS" not in assigned


# ─────────────────────────────────────────────────────────────────────────────
# A6 - paridad de calibracion
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a6_both_families_publish_the_same_calibration_split(
    real_report: ModelComparisonReport,
) -> None:
    """Las dos familias reparten el *train* igual, fold a fold, y el histograma coincide (A6)."""
    parity = _block(real_report, "calibration_parity")
    per_family = cast("dict[str, dict[str, object]]", parity["per_family"])
    assert sorted(per_family) == sorted(FAMILY_ORDER)
    expected_sizes = [437, 447, 457, 467, 477, 487, 497, 507, 517, 527]
    for family in FAMILY_ORDER:
        assert per_family[family]["n_calibration"] == expected_sizes
        assert per_family[family]["methods"] == {"platt": 7, "isotonic": 3, "none": 0}
    assert parity["identical"] is True
    assert parity["n_families_compared"] == 2
    assert _block(real_report, "comparison")["n_rows"] == 4


@needs_store
def test_a6_every_variant_decides_with_its_own_probability(
    real_report: ModelComparisonReport,
) -> None:
    """Cada variante opera con `p >= 0,5` sobre **su** probabilidad: el umbral no se mueve (A6)."""
    for variant in _variants(real_report):
        probabilities = cast("list[float]", variant["deciding_probabilities"])
        assert len(probabilities) == 500
        traded = sum(1 for value in probabilities if value >= DECISION_THRESHOLD)
        assert traded == variant["n_traded"], variant["run_sha256"]
        assert variant["zeros"] == 500 - traded


# ─────────────────────────────────────────────────────────────────────────────
# A7 - la linea base cuadra con lo publicado
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a7_the_frozen_baseline_is_reproduced_to_the_declared_tolerance(
    real_report: ModelComparisonReport,
) -> None:
    """Las dos series de la linea base reproducen las cuatro cifras congeladas (A7)."""
    frozen = _block(real_report, "frozen_baseline")
    assert frozen["verified"] is True
    rows = cast("list[dict[str, object]]", frozen["rows"])
    assert len(rows) == 2
    for row in rows:
        key = "calibrated" if row["calibrated"] else "raw"
        reference = FROZEN_BASELINE[key]
        assert row["n_traded"] == reference["n_traded"]
        for name in ("brier_score", "log_loss", "pnl_declared_sum"):
            observed = float(cast("float", row[name]))
            assert abs(observed - reference[name]) <= FROZEN_TOLERANCE, name
    for name, digest in FROZEN_REPORTS_BEFORE.items():
        path = FROZEN_REPORTS / name
        import hashlib

        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


@needs_store
def test_a7_a_line_base_that_does_not_square_is_a_typed_error(
    real_report: ModelComparisonReport, tmp_path: Path
) -> None:
    """Si la reconstruccion no cuadra con lo congelado, error tipado, no una cifra (A7)."""
    entry = next(
        item for item in real_report.registry.entries if item.variant_id == BASELINE_VARIANT_ID
    )
    runs = tmp_path / "runs"
    _copy_frozen_runs(runs)
    model_path = runs / entry.run_sha256 / "model.json"
    document = cast("dict[str, object]", json.loads(model_path.read_text(encoding="utf-8")))
    model = cast("dict[str, object]", document["model"])
    folds = cast("list[dict[str, object]]", model["folds"])
    coefficients = cast("list[float]", folds[0]["coefficients"])
    coefficients[0] = coefficients[0] + 5.0
    model_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ModelComparisonError):
        model_comparison.reconstruct_baseline(
            entry,
            runs_root=runs,
            universe=real_report.universe,
            frame=real_report.features,
            plan=real_report.split_plan,
            cost_model=declared_cost_model(),
            slippage=declared_slippage_assumption(),
        )


@needs_store
def test_a7_the_registry_grows_to_four_entries(base_root: Path) -> None:
    """Tras la corrida, el registro de trabajo tiene **cuatro** entradas (A7)."""
    registry = load_registry(base_root / "runs")
    assert registry.n_trials == 4
    assert sorted(registry.variant_ids) == sorted([BASELINE_VARIANT_ID] * 2 + [VARIANT_ID] * 2)


# ─────────────────────────────────────────────────────────────────────────────
# A8 - serie por sesion y matriz rectangular
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a8_each_variant_publishes_five_hundred_values(
    real_report: ModelComparisonReport,
) -> None:
    """Cada variante publica 500 valores, con `0.0` donde no opera (A8)."""
    for variant in _variants(real_report):
        series = cast("list[float]", variant["series"])
        assert len(series) == 500
        zeros = sum(1 for value in series if value == 0.0)
        assert zeros == variant["zeros"]
        assert zeros >= 500 - int(cast("int", variant["n_traded"]))
        assert int(cast("int", variant["n_traded"])) + zeros == 500


@needs_store
def test_a8_the_matrix_is_rectangular_and_matches_the_registry(
    real_report: ModelComparisonReport,
) -> None:
    """La matriz publica 500 filas, 4 columnas y el `n_trials` del registro (A8)."""
    matrix = _block(real_report, "matrix")
    assert matrix["n_observations"] == 500
    assert matrix["n_variants"] == 4
    assert matrix["registry_n_trials"] == real_report.registry.n_trials == 4
    assert matrix["matrix_matches_registry"] is True
    assert matrix["blocks"] == PBO_BLOCKS
    columns = cast("list[dict[str, object]]", matrix["columns"])
    assert len(columns) == 4
    assert [column["run_sha256"] for column in columns] == [
        entry.run_sha256 for entry in real_report.registry.entries
    ]


@needs_store
def test_a8_the_two_sharpes_are_published_and_are_not_the_same(
    real_report: ModelComparisonReport,
) -> None:
    """El Sharpe del DSR (500 sesiones) y el `sharpe_per_session` se publican los dos (A8)."""
    dsr = _block(real_report, "deflated_sharpe_ratio")
    assert dsr["n_observations"] == 500
    assert dsr["registry_sharpe_per_session"] != dsr["sr_observed"]
    units = _block(real_report, "verdict", "sharpe_units")
    assert "no" in str(units["note"])


# ─────────────────────────────────────────────────────────────────────────────
# A9 - `n_trials` del registro
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a9_n_trials_and_sr_variance_come_from_the_registry(
    real_report: ModelComparisonReport, base_root: Path
) -> None:
    """`n_trials` y `sr_variance` son los del registro, con sus cuatro identidades (A9)."""
    registry = _block(real_report, "registry")
    assert registry["n_trials"] == 4
    assert registry["registry_sha256"] == real_report.registry.registry_sha256
    assert registry["sr_variance"] == real_report.registry.sr_variance
    entries = cast("list[dict[str, object]]", registry["entries"])
    assert len(entries) == 4
    assert sorted(str(entry["run_sha256"]) for entry in entries) == sorted(
        entry.run_sha256 for entry in real_report.registry.entries
    )
    assert len(load_registry(base_root / "runs").entries) == 4


@needs_store
def test_a9_three_columns_against_four_trials_is_a_typed_error(
    real_report: ModelComparisonReport,
) -> None:
    """`require_matrix_matches_registry` con 3 columnas de 4 intentos falla (A9)."""
    with pytest.raises(TrialsMismatchError):
        require_matrix_matches_registry(registry=real_report.registry, n_columns=3)
    require_matrix_matches_registry(registry=real_report.registry, n_columns=4)


def test_a9_the_cli_does_not_accept_n_trials() -> None:
    """El CLI no acepta `--n-trials`: los intentos se derivan del registro (A9)."""
    completed = _cli("--as-of", NOW.isoformat(), "--n-trials", "4")
    assert completed.returncode == 2
    assert "--n-trials" in completed.stderr


@needs_store
def test_a9_more_trials_deflate_the_same_series_more(
    real_report: ModelComparisonReport,
) -> None:
    """El DSR de la **misma** serie baja al crecer el registro: 3 y 10 intentos (A9)."""
    series = _synthetic_series()

    def registry_of(count: int) -> Registry:
        entries = tuple(
            RegistryEntry(
                run_sha256=f"fake{index:02d}",
                variant_id="fake_v1",
                sharpe_per_session=0.1 * index,
                n_observations=100,
            )
            for index in range(count)
        )
        return Registry(entries=entries, registry_sha256="fake")

    # #80 corrigio la unidad del P&L declarado (fraccion del nocional): la serie real ya **no**
    # tiene Sharpe negativo, asi que su DSR no se satura a `0.0` y su deflacion es casi plana:
    # no distingue tres intentos de diez. Por eso la deflacion se **mide** con una serie
    # determinista de Sharpe positivo y del orden del `SR0` esperado (A9).
    real = deflate_block(returns=real_report.evaluated[0].series, registry=registry_of(3))
    assert float(cast("float", real["dsr"])) >= 0.0
    small = deflate_block(returns=series, registry=registry_of(3))
    large = deflate_block(returns=series, registry=registry_of(10))
    assert small["n_trials"] == 3
    assert large["n_trials"] == 10
    assert large["dsr"] != small["dsr"]
    assert float(cast("float", large["dsr"])) < float(cast("float", small["dsr"]))


# ─────────────────────────────────────────────────────────────────────────────
# A10 - ninguna variante se rellena ni se cae
# ─────────────────────────────────────────────────────────────────────────────
def _candidates_with_extra(
    real_report: ModelComparisonReport,
    base_root: Path,
    extra: RegistryEntry,
) -> tuple[list[model_comparison.Candidate], Registry]:
    """Reconstruye los candidatos con una entrada sintetica anadida al registro real."""
    entries = (*real_report.registry.entries, extra)
    registry = Registry(entries=entries, registry_sha256="synthetic")
    known = {
        variant.run_sha256: variant
        for variant in real_report.evaluated
        if variant.variant_id == VARIANT_ID
    }
    candidates = model_comparison._candidates(  # pyright: ignore[reportPrivateUsage]
        registry=registry,
        runs_root=base_root / "runs",
        known=known,
        universe=real_report.universe,
        frame=real_report.features,
        plan=real_report.split_plan,
        cost_model=declared_cost_model(),
        slippage=declared_slippage_assumption(),
    )
    return list(candidates), registry


@needs_store
def test_a10_an_unknown_variant_is_declared_and_never_filled(
    real_report: ModelComparisonReport, base_root: Path
) -> None:
    """Un `variant_id` desconocido da `not_evaluable` con motivo, sin columna de ceros (A10)."""
    extra = RegistryEntry(
        run_sha256="deadbeef",
        variant_id="mystery_v1",
        sharpe_per_session=0.3,
        n_observations=10,
    )
    candidates, registry = _candidates_with_extra(real_report, base_root, extra)
    evaluated = [item for item in candidates if isinstance(item, model_comparison.Variant)]
    missing = [item for item in candidates if isinstance(item, model_comparison.NotEvaluable)]
    assert len(evaluated) == 4
    assert len(missing) == 1
    assert missing[0].run_sha256 == "deadbeef"
    assert missing[0].variant_id == "mystery_v1"
    assert "mystery_v1" in missing[0].reason
    assert missing[0].to_payload()["column"] is None
    selection = selection_block(candidates)
    assert selection["state"] == "not_evaluable"
    assert selection["selected"] is None
    assert len(cast("list[object]", selection["blockers"])) == 1
    verdict = model_comparison._verdict_block(  # pyright: ignore[reportPrivateUsage]
        evaluated=tuple(evaluated), registry=registry, selection=selection, candidates=candidates
    )
    assert verdict["state"] == "not_evaluable"
    assert cast("dict[str, object]", verdict["gate"])["aggregate"] != "pass"
    assert cast("dict[str, object]", verdict["deflated_sharpe_ratio"])["state"] == "not_evaluable"
    assert (
        cast("dict[str, object]", verdict["probability_of_backtest_overfitting"])["state"]
        == "not_evaluable"
    )


@needs_store
def test_a10_a_mismatched_observation_count_is_declared(
    real_report: ModelComparisonReport, base_root: Path
) -> None:
    """Un `n_observations` que no cuadra da `not_evaluable`, sin rellenar la serie (A10)."""
    entry = next(
        item for item in real_report.registry.entries if item.variant_id == BASELINE_VARIANT_ID
    )
    tampered = dataclasses.replace(entry, n_observations=entry.n_observations + 1)
    # La entrada envenenada **sustituye** a la del registro: si se anadiese, el registro tendria
    # dos entradas con la misma identidad y la reconstruible seguiria dando una columna (A10).
    entries = tuple(
        tampered if item.run_sha256 == entry.run_sha256 else item
        for item in real_report.registry.entries
    )
    registry = Registry(entries=entries, registry_sha256="synthetic")
    known = {
        variant.run_sha256: variant
        for variant in real_report.evaluated
        if variant.variant_id == VARIANT_ID
    }
    candidates = list(
        model_comparison._candidates(  # pyright: ignore[reportPrivateUsage]
            registry=registry,
            runs_root=base_root / "runs",
            known=known,
            universe=real_report.universe,
            frame=real_report.features,
            plan=real_report.split_plan,
            cost_model=declared_cost_model(),
            slippage=declared_slippage_assumption(),
        )
    )
    missing = [item for item in candidates if isinstance(item, model_comparison.NotEvaluable)]
    assert len(missing) == 1
    assert missing[0].run_sha256 == entry.run_sha256
    assert "n_observations" in missing[0].reason
    assert not any(
        item.variant_id == BASELINE_VARIANT_ID and item.run_sha256 == entry.run_sha256
        for item in candidates
        if isinstance(item, model_comparison.Variant)
    )


# ─────────────────────────────────────────────────────────────────────────────
# A11 - regla de seleccion probada
# ─────────────────────────────────────────────────────────────────────────────
def _synthetic_candidates(
    real_report: ModelComparisonReport,
    *,
    brier: Mapping[str, float],
    log_loss: Mapping[str, float],
) -> tuple[model_comparison.Candidate, ...]:
    """Los cuatro candidatos reales con las metricas **sustituidas** por las del caso (A11)."""
    out: list[model_comparison.Candidate] = []
    for variant in real_report.evaluated:
        out.append(
            dataclasses.replace(
                variant,
                brier_score=brier[variant.variant_id],
                log_loss_value=log_loss[variant.variant_id],
            )
        )
    return tuple(out)


def test_a11_the_rule_is_declared_and_published(real_report: ModelComparisonReport) -> None:
    """`rule`, `primary_metric`, `tie_breakers` y `FAMILY_ORDER` salen literales (A11)."""
    selection = _block(real_report, "selection")
    assert selection["rule"] == SELECTION_RULE
    assert selection["primary_metric"] == PRIMARY_METRIC == "brier_score"
    assert selection["tie_breakers"] == list(TIE_BREAKERS) == ["log_loss", "family_order"]
    assert selection["family_order"] == list(FAMILY_ORDER)
    assert selection["tie_tolerance"] == 1e-12
    assert selection["is_validation"] is False


@needs_store
def test_a11_a_tie_in_brier_and_log_loss_goes_to_the_simpler_family(
    real_report: ModelComparisonReport,
) -> None:
    """(a) Empate en Brier y log-loss: gana la familia mas simple (A11)."""
    candidates = _synthetic_candidates(
        real_report,
        brier=dict.fromkeys(FAMILY_ORDER, 0.25),
        log_loss=dict.fromkeys(FAMILY_ORDER, 0.7),
    )
    selection = selection_block(candidates)
    assert selection["state"] == "selected"
    selected = cast("dict[str, object]", selection["selected"])
    assert selected["variant_id"] == FAMILY_ORDER[0] == BASELINE_VARIANT_ID


@needs_store
def test_a11_a_tie_in_brier_is_broken_by_log_loss(real_report: ModelComparisonReport) -> None:
    """(b) Empate en Brier que desempata la log-loss (A11)."""
    candidates = _synthetic_candidates(
        real_report,
        brier=dict.fromkeys(FAMILY_ORDER, 0.25),
        log_loss={BASELINE_VARIANT_ID: 0.9, VARIANT_ID: 0.5},
    )
    selection = selection_block(candidates)
    selected = cast("dict[str, object]", selection["selected"])
    assert selected["variant_id"] == VARIANT_ID


@needs_store
def test_a11_a_worse_baseline_selects_lightgbm(real_report: ModelComparisonReport) -> None:
    """(c) Con la linea base peor en Brier gana LightGBM: no esta cableada (A11)."""
    candidates = _synthetic_candidates(
        real_report,
        brier={BASELINE_VARIANT_ID: 0.30, VARIANT_ID: 0.20},
        log_loss={BASELINE_VARIANT_ID: 0.60, VARIANT_ID: 0.55},
    )
    selection = selection_block(candidates)
    selected = cast("dict[str, object]", selection["selected"])
    assert selected["variant_id"] == VARIANT_ID
    assert selected["brier_score"] == 0.20


@needs_store
def test_a11_a_candidate_without_metric_leaves_no_selection(
    real_report: ModelComparisonReport,
) -> None:
    """(d) Un candidato sin metrica deja la seleccion en `not_evaluable` (A11)."""
    candidates: tuple[model_comparison.Candidate, ...] = (
        real_report.evaluated[0],
        model_comparison.NotEvaluable(
            run_sha256="nope", variant_id="mystery_v1", reason="sin metrica", error="X"
        ),
    )
    selection = selection_block(candidates)
    assert selection["state"] == "not_evaluable"
    assert selection["selected"] is None
    assert selection["n_blockers"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# A12 - cifras honestas
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a12_the_four_rows_are_measured_here_with_the_declared_basis(
    real_report: ModelComparisonReport,
) -> None:
    """Las cuatro filas publican Brier, log-loss, operadas y su delta contra #24 (A12)."""
    comparison = _block(real_report, "comparison")
    assert comparison["basis"] == "declared_cost"
    assert comparison["is_validation"] is False
    rows = _rows(real_report)
    assert [row["brier_score"] for row in rows] == list(MEASURED_BRIER)
    reference = cast("dict[str, object]", comparison["reference"])
    assert reference["variant_id"] == BASELINE_VARIANT_ID
    assert reference["run_sha256"] == real_report.evaluated[0].run_sha256
    for row in rows:
        assert row["basis"] == "declared_cost"
        assert row["is_validation"] is False
        for name in ("brier_score", "log_loss", "n_traded"):
            assert name in row
        delta = cast("dict[str, object]", row["delta_vs_baseline_raw"])
        assert delta["reference_run_sha256"] == reference["run_sha256"]
        assert float(cast("float", delta["brier_score"])) == float(
            cast("float", row["brier_score"])
        ) - float(cast("float", reference["brier_score"]))


@needs_store
def test_a12_the_net_metrics_are_refused_and_the_slippage_is_assumed(
    real_report: ModelComparisonReport,
) -> None:
    """`net_metrics` es `not_computable` y el *slippage* sigue siendo un supuesto (A12)."""
    net = _block(real_report, "net_metrics")
    assert net["state"] == "not_computable"
    assert net["follow_up"] == ["#62", "#60"]
    cost = _block(real_report, "declared_cost")
    slippage = cast("dict[str, object]", cost["slippage"])
    assert slippage["state"] == "assumed"
    assert slippage["is_measurement"] is False
    assert _block(real_report, "net_metrics")["state"] != "measured"


@needs_store
def test_a12_the_unit_bug_is_fixed_and_measured(real_report: ModelComparisonReport) -> None:
    """El bloque de #80 publica la constante medida por operacion, ya arreglado el motor (A12)."""
    bug = _block(real_report, "unit_bug_80")
    assert bug["issue"] == "#80"
    assert bug["state"] == "fixed_and_measured"
    assert abs(float(cast("float", bug["observed_per_operation"])) - 0.000042) <= 1e-15
    assert abs(float(cast("float", bug["correct_term_per_operation"])) - 0.000042) <= 1e-15
    assert abs(float(cast("float", bug["difference_per_operation"]))) <= 1e-12
    assert bug["n_operations"] == real_report.evaluated[0].n_traded
    assert "c_fraction_of_notional" in str(bug["statement"])
    assert "c_fraction_of_notional" in str(bug["resolution"])
    # A8: la guardia de `git diff` se reduce al contrato nuevo. El motor (#80) **si** entra en
    # el diff de esta entrega; lo que ya no se afirma es que resta 100x el coste declarado.
    changed = set(_git("diff", "--name-only", f"{BASE_COMMIT}..HEAD").splitlines())
    assert "src/cfdtrader/backtest/engine.py" in changed


@needs_store
def test_a12_the_table_is_not_copied_from_the_frozen_reports(
    real_report: ModelComparisonReport, tmp_path: Path
) -> None:
    """La tabla sale de `runs/`, no de los informes congelados: mutarlos no la cambia (A12)."""
    frozen = FROZEN_REPORTS / "baseline_2026-09-22.json"
    document = cast("dict[str, object]", json.loads(frozen.read_text(encoding="utf-8")))
    metrics = cast("dict[str, object]", document["probability_metrics"])
    original = metrics["brier_score"]
    metrics["brier_score"] = 0.999
    mutated = tmp_path / "reports"
    mutated.mkdir()
    (mutated / frozen.name).write_text(json.dumps(document), encoding="utf-8")
    mutated_document = cast(
        "dict[str, object]", json.loads((mutated / frozen.name).read_text(encoding="utf-8"))
    )
    assert (
        cast("dict[str, object]", mutated_document["probability_metrics"])["brier_score"] == 0.999
    )
    assert original != 0.999
    runs = tmp_path / "runs"
    _copy_frozen_runs(runs)
    reports = analyse(
        store=Store(REAL_DATA),
        reports_dir=mutated,
        runs_root=runs,
        settings=Settings(),
        as_of=NOW,
        write=False,
    )
    assert [row["brier_score"] for row in _rows(reports)] == list(MEASURED_BRIER)
    assert [row["brier_score"] for row in _rows(real_report)] == list(MEASURED_BRIER)
    source = _source(model_comparison)
    assert "baseline_2026" not in source
    # El informe **escribe** en el directorio de informes (`--reports-dir`, cuyo defecto es
    # `<data-root>/derived/reports`), asi que la ruta aparece en la ayuda del CLI; lo que A12
    # prohibe es **leer** los informes congelados. La unica lectura del modulo es el `model.json`
    # del registro, y ninguna cae sobre una ruta de informes.
    reads = [
        ast.unparse(node)
        for node in ast.walk(_tree(model_comparison))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"read_text", "read_bytes"}
    ]
    assert reads, "el informe lee el `model.json` del registro para reconstruir la linea base"
    assert not [call for call in reads if "reports" in call], reads


@needs_store
def test_a12_the_markdown_carries_the_table_and_the_rule(
    real_report: ModelComparisonReport,
) -> None:
    """El `.md` lleva la tabla de cuatro filas y la regla de seleccion (A12)."""
    text = render_markdown(real_report)
    assert "## Tabla comparativa (4 filas, las mismas 500 sesiones)" in text
    for variant in (BASELINE_VARIANT_ID, VARIANT_ID):
        assert text.count(f"| `{variant}` |") >= 3
    assert "## Regla de seleccion (pre-declarada)" in text
    assert "## Unidades del motor (#80, arreglado y medido)" in text
    assert (
        "0.000042"
        in cast("dict[str, str]", _block(real_report, "unit_bug_80")["constants"])["identity"]
    )
    assert text.endswith("\n")


# ─────────────────────────────────────────────────────────────────────────────
# A13 - determinismo del informe
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a13_the_second_pass_is_unchanged_and_hashes_the_same(
    real_report: ModelComparisonReport, base_root: Path
) -> None:
    """La segunda pasada deja las dos entradas nuevas en `unchanged` y hashea igual (A13)."""
    second = analyse(
        store=Store(REAL_DATA),
        reports_dir=base_root / "reports",
        runs_root=base_root / "runs",
        settings=Settings(),
        as_of=NOW,
        write=True,
    )
    assert second.report_sha256 == real_report.report_sha256
    assert second.json_text() == real_report.json_text()
    assert len(second.records) == 2
    for record in second.records:
        assert second.outcomes[record.run_sha256] == "unchanged", record.run_sha256


@needs_store
def test_a13_the_cli_is_byte_identical_across_processes_and_reports_dirs(
    real_report: ModelComparisonReport, base_root: Path, tmp_path: Path
) -> None:
    """`report_sha256`, `.json` y `.md` no cambian con el proceso ni con `--reports-dir` (A13)."""
    runs = tmp_path / "runs"
    _copy_frozen_runs(runs)
    reports = tmp_path / "reports"
    other = tmp_path / "other-reports"
    for seed, target in (("0", reports), ("1", reports), ("random", other)):
        completed = _cli(
            "--data-root",
            str(REAL_DATA),
            "--reports-dir",
            str(target),
            "--runs-root",
            str(runs),
            "--as-of",
            NOW.isoformat(),
            seed=seed,
        )
        assert completed.returncode == 0, completed.stderr
    name = f"{REPORT_PREFIX}_{NOW.date().isoformat()}"
    in_process_json = base_root / "reports" / f"{name}.json"
    in_process_markdown = base_root / "reports" / f"{name}.md"
    first = reports / f"{name}.json"
    second = other / f"{name}.json"
    assert _report_hash(first) == _report_hash(second) == real_report.report_sha256
    assert first.read_bytes() == second.read_bytes() == in_process_json.read_bytes()
    assert (reports / f"{name}.md").read_bytes() == (other / f"{name}.md").read_bytes()
    assert (reports / f"{name}.md").read_bytes() == in_process_markdown.read_bytes()
    cli_registry = load_registry(runs)
    assert cli_registry.registry_sha256 == real_report.registry.registry_sha256
    assert cli_registry.n_trials == 4
    for record in real_report.records:
        assert (runs / record.run_sha256 / "model.json").read_bytes() == (
            base_root / "runs" / record.run_sha256 / "model.json"
        ).read_bytes()


# ─────────────────────────────────────────────────────────────────────────────
# A14 - un test por criterio y las puertas
# ─────────────────────────────────────────────────────────────────────────────
def test_a14_every_criterion_has_a_test_in_the_two_new_suites() -> None:
    """Cada criterio A1-A14 tiene al menos un `test_aN_...` en las dos suites nuevas (A14)."""
    source = "\n".join(path.read_text(encoding="utf-8") for path in NEW_TESTS)
    tree = ast.parse(source)
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    missing = [
        f"A{index}"
        for index in range(1, 15)
        if not any(name.startswith(f"test_a{index}_") for name in names)
    ]
    assert not missing, f"criterios sin test: {missing} (A14)"


def test_a14_the_new_modules_have_no_pragma_except_the_entry_point() -> None:
    """Ningun modulo nuevo lleva `# pragma: no cover` salvo el guard de `__main__` (A14)."""
    for module in (model_comparison, lightgbm_model):
        text = _source(module)
        lines = [line for line in text.splitlines() if "pragma: no cover" in line]
        assert len(lines) <= 1, f"{module.__name__}: {lines}"
        if lines:
            assert "__main__" in lines[0], lines
            assert "PYTHONHASHSEED" not in lines[0]


def test_a14_the_new_modules_are_pure_python_and_the_seed_is_declared() -> None:
    """El modulo publica la semilla declarada y no reimplementa el enlace de #25 (A14)."""
    source = _source(model_comparison)
    assert "SEED" in source
    assert SEED == 20260920
    assert "PBO_BLOCKS" in source
    tree = _tree(lightgbm_model)
    names = {
        (node.module or "", alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert ("cfdtrader.models.calibration", "sigmoid") in names


def test_a14_the_fit_is_the_same_as_the_one_the_report_uses() -> None:
    """El informe ajusta LightGBM con la constante declarada, no con otra copia (A14)."""
    signature = __import__("inspect").signature(fit_lightgbm)
    assert "hyperparameters" in signature.parameters
    assert "splits" in signature.parameters
    source = _source(model_comparison)
    assert "hyperparameters=LIGHTGBM_HYPERPARAMETERS" in source
    assert "FROZEN_TOLERANCE" in source


# ─────────────────────────────────────────────────────────────────────────────
# Apoyo: la ruta de exito del CLI y las ramas que las corridas reales no tocan
#
# `pytest-cov` no instrumenta subprocesos (misma trampa que en #16): la ruta de exito de
# `main` solo cuenta como cubierta si se llama **en proceso**. Todo lo de abajo escribe en
# `tmp_path` o no escribe nada, asi que el `data/` y el `runs/` del repositorio no se tocan.
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a2_the_cli_in_process_writes_the_report_and_exits_zero(tmp_path: Path) -> None:
    """La ruta de exito del CLI, llamada **en proceso**, escribe el informe y devuelve 0 (A2)."""
    runs = tmp_path / "runs"
    _copy_frozen_runs(runs)
    reports = tmp_path / "reports"
    code = main(
        [
            "--data-root",
            str(REAL_DATA),
            "--reports-dir",
            str(reports),
            "--runs-root",
            str(runs),
            "--as-of",
            NOW.isoformat(),
        ]
    )
    assert code == 0
    stem = f"{REPORT_PREFIX}_{NOW.date().isoformat()}"
    assert (reports / f"{stem}.json").is_file()
    assert (reports / f"{stem}.md").is_file()
    assert load_registry(runs).n_trials == 4


def test_a2_an_as_of_that_is_not_iso_exits_two_with_a_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Un `--as-of` que no es ISO-8601 tambien sale con 2 y el motivo en `stderr` (A2)."""
    assert main(["--as-of", "no-es-iso-8601"]) == 2
    assert "--as-of" in capsys.readouterr().err


@needs_store
def test_a7_a_reconstruction_that_does_not_square_is_a_typed_error(
    real_report: ModelComparisonReport,
) -> None:
    """`_require_frozen_match` con una cifra que no cuadra lanza el error tipado (A7)."""
    tampered = dataclasses.replace(real_report.evaluated[0], brier_score=0.9)
    with pytest.raises(model_comparison.FrozenBaselineMismatchError):
        model_comparison._require_frozen_match(tampered)  # pyright: ignore[reportPrivateUsage]


def test_a8_an_empty_matrix_has_no_rows() -> None:
    """Sin variantes evaluadas la matriz no tiene filas (A8)."""
    assert model_comparison._series_matrix(()) == []  # pyright: ignore[reportPrivateUsage]


@needs_store
def test_a9_a_complete_matrix_without_selection_is_declared(
    real_report: ModelComparisonReport,
) -> None:
    """Con matriz completa pero sin seleccion, el DSR se declara `not_evaluable` (A9)."""
    empty_selection: dict[str, object] = {}
    verdict = model_comparison._verdict_block(  # pyright: ignore[reportPrivateUsage]
        evaluated=real_report.evaluated,
        registry=real_report.registry,
        selection=empty_selection,
        candidates=real_report.candidates,
    )
    assert verdict["state"] == "evaluated"
    dsr = cast("dict[str, object]", verdict["deflated_sharpe_ratio"])
    assert dsr["state"] == "not_evaluable"


@needs_store
def test_a10_a_missing_model_document_is_declared(
    real_report: ModelComparisonReport, tmp_path: Path
) -> None:
    """Una entrada cuyo `model.json` no esta se declara `not_evaluable`, no se rellena (A10)."""
    runs = tmp_path / "runs"
    _copy_frozen_runs(runs)
    entry = next(
        item for item in real_report.registry.entries if item.variant_id == BASELINE_VARIANT_ID
    )
    (runs / entry.run_sha256 / "model.json").unlink()
    known: dict[str, model_comparison.Variant] = {}
    candidates = model_comparison._candidates(  # pyright: ignore[reportPrivateUsage]
        registry=Registry(entries=(entry,), registry_sha256="synthetic"),
        runs_root=runs,
        known=known,
        universe=real_report.universe,
        frame=real_report.features,
        plan=real_report.split_plan,
        cost_model=declared_cost_model(),
        slippage=declared_slippage_assumption(),
    )
    assert len(candidates) == 1
    missing = candidates[0]
    assert isinstance(missing, model_comparison.NotEvaluable)
    assert missing.error == "MissingModelDocumentError"


def test_a12_the_number_helper_formats_every_json_type() -> None:
    """`_number` formatea `None`, `bool`, `int`, `float` y texto sin inventar cifras (A12)."""
    number = model_comparison._number  # pyright: ignore[reportPrivateUsage]
    assert number(None) == "`null`"
    assert number(True) == "true"
    assert number(7) == "7"
    assert number(0.125) == "0.125000"
    assert number("texto") == "texto"


@needs_store
def test_a12_the_comparison_without_the_raw_baseline_invents_no_delta(
    real_report: ModelComparisonReport,
) -> None:
    """Sin la cruda de #24 entre las evaluadas, la tabla no inventa un delta (A12)."""
    block = model_comparison._comparison_block(  # pyright: ignore[reportPrivateUsage]
        tuple(item for item in real_report.evaluated if item.calibrated),
    )
    assert block["reference"] is None
    rows = cast("list[dict[str, object]]", block["rows"])
    assert rows and all(row["delta_vs_baseline_raw"] is None for row in rows)


@needs_store
def test_a12_the_markdown_declares_a_report_without_selection(
    real_report: ModelComparisonReport,
) -> None:
    """El `.md` publica el veto cuando la seleccion no se puede resolver (A10, A12)."""
    payload = copy.deepcopy(real_report.payload)
    payload["selection"] = selection_block(
        (
            real_report.evaluated[0],
            model_comparison.NotEvaluable(
                run_sha256="nope", variant_id="mystery_v1", reason="sin metrica", error="X"
            ),
        )
    )
    text = render_markdown(dataclasses.replace(real_report, payload=payload))
    assert "Sin seleccion" in text
    assert "**Vetos** (candidato sin metrica)" in text
