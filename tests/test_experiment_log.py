"""Tests del registro de experimentos y del informe de sobreajuste (#16).

Un test por criterio. Aqui viven los del **registro** (`runs/<hash>/`), el CLI y el informe
(A1, A10-A15, A18, A20-A32 y A35); el nucleo puro esta en ``tests/test_overfitting.py``.

Ninguna prueba escribe en el ``data/`` ni en el ``runs/`` del repositorio: la fixture de
sesion de ``tests/conftest.py`` huella los dos directorios antes y despues (A27). El informe
compartido por varios tests sale de una unica corrida sobre ``tmp_path_factory``: la misma
raiz temporal, con el *stream* congelado de #15 y sin tocar nada del repositorio.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final, cast

import pytest

from cfdtrader.analysis import backtest_report, experiment_log, phase0_report
from cfdtrader.backtest import overfitting
from cfdtrader.backtest.engine import canonical_text
from cfdtrader.backtest.metrics import MetricsInputError, sharpe_ratio
from cfdtrader.data.store import WriteOutcome

MODULE_PATH = Path(str(experiment_log.__file__))
SOURCE = MODULE_PATH.read_text(encoding="utf-8")
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPOSITORY_RUNS: Final[Path] = REPO_ROOT / "runs"
AS_OF: Final[datetime] = datetime(2026, 9, 19, 12, tzinfo=UTC)
AS_OF_LATER: Final[datetime] = datetime(2026, 9, 20, 3, 30, tzinfo=UTC)

#: Lo que A1 permite tocar: los cinco ficheros de la tarea.
ALLOWED_PATHS: Final[frozenset[str]] = frozenset(
    {
        "src/cfdtrader/backtest/overfitting.py",
        "src/cfdtrader/analysis/experiment_log.py",
        "src/cfdtrader/backtest/__init__.py",
        "src/cfdtrader/analysis/__init__.py",
        "tests/test_overfitting.py",
        "tests/test_experiment_log.py",
        "tests/conftest.py",
    }
)

#: IDs estables de las dos tablas legibles por maquina (A30).
BOUNDARY_IDS: Final[frozenset[str]] = frozenset(
    entry["id"] for entry in overfitting.OVERFITTING_DOES_NOT_DO
)
FOLLOW_UP_IDS: Final[frozenset[str]] = frozenset(entry["id"] for entry in overfitting.FOLLOW_UPS)


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> experiment_log.OverfittingReport:
    """Una corrida completa (registro + informe) sobre una raiz temporal compartida."""
    root = tmp_path_factory.mktemp("t16")
    return experiment_log.analyse(
        runs_root=root / "runs", reports_dir=root / "reports", as_of=AS_OF, write=True
    )


def _config(
    variant_id: str,
    *,
    seed: int = 1,
    hyperparameters: Mapping[str, object] | None = None,
    window: Mapping[str, object] | None = None,
) -> experiment_log.ExperimentConfig:
    """Configuracion de sonda: todo JSON puro y sin ruta ni instante dentro."""
    return experiment_log.ExperimentConfig(
        variant_id=variant_id,
        features=("synthetic_random_draw",),
        hyperparameters=hyperparameters if hyperparameters is not None else {"kind": "probe"},
        seed=seed,
        series_id="SYNTHETIC",
        window=window if window is not None else {"kind": "index", "start": 0, "stop": 8},
    )


def _record(
    runs_root: Path,
    variant_id: str,
    *,
    sharpe: float,
    seed: int = 1,
    as_of: datetime = AS_OF,
    write: bool = True,
) -> experiment_log.ExperimentRecord:
    """Registra una variante de sonda con su Sharpe por sesion."""
    return experiment_log.record_experiment(
        runs_root=runs_root,
        config=_config(variant_id, seed=seed),
        result=experiment_log.ExperimentResult(sharpe_per_session=sharpe, n_observations=8),
        as_of=as_of,
        write=write,
    )


def _registry(runs_root: Path, count: int) -> experiment_log.Registry:
    """Registro de `count` variantes distintas: es la fuente de `n_trials` (A10)."""
    for index in range(count):
        _record(
            runs_root,
            f"probe-{index:02d}",
            sharpe=0.01 * (index + 1),
            seed=index + 1,
        )
    return experiment_log.load_registry(runs_root)


def _column(matrix: tuple[tuple[float, ...], ...], index: int) -> tuple[float, ...]:
    """La columna `index` de una matriz de variantes, como serie."""
    return tuple(row[index] for row in matrix)


def _payload(report: experiment_log.OverfittingReport) -> dict[str, object]:
    """El payload publicado del informe, leido del JSON (mismo texto que el fichero)."""
    return cast("dict[str, object]", json.loads(report.json_text()))


def _mapping(value: object) -> dict[str, object]:
    """Un sub-dict del payload: la forma la garantiza el propio JSON."""
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _rows(value: object) -> list[dict[str, object]]:
    """Las filas de una tabla del payload (una lista de sub-dicts)."""
    assert isinstance(value, list)
    return [cast("dict[str, object]", row) for row in cast("list[object]", value)]


def _number(value: object) -> float:
    """Un numero del payload: el JSON lo garantiza, el tipado estatico no lo sabe."""
    assert isinstance(value, int | float) and not isinstance(value, bool)
    return float(value)


def _series(seed: int = 20260919, n: int = 128) -> tuple[float, ...]:
    """Serie determinista para el DSR (sin azar propio del test)."""
    return tuple(0.0005 * math.sin(index / 4.0) + 0.0002 for index in range(n))


def _cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Lanza el modulo como proceso: es la unica forma de medir los codigos de salida."""
    return subprocess.run(  # noqa: S603 - comando fijo (el interprete de la sesion)
        [sys.executable, "-m", "cfdtrader.analysis.experiment_log", *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# A1 — ficheros y rutas fijados
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_files_and_paths_are_the_fixed_ones() -> None:
    for relative in (
        "src/cfdtrader/backtest/overfitting.py",
        "src/cfdtrader/analysis/experiment_log.py",
        "tests/test_overfitting.py",
        "tests/test_experiment_log.py",
    ):
        assert (REPO_ROOT / relative).is_file(), relative
    completed = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],  # noqa: S607 - el git del sistema, uso fijo
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    changed = {line for line in completed.stdout.split() if line}
    assert changed <= ALLOWED_PATHS, changed - ALLOWED_PATHS
    assert not any(
        path.startswith(
            ("data/", "config/", "_docs/", "src/cfdtrader/data/", "src/cfdtrader/models/")
        )
        for path in changed
    )


# ─────────────────────────────────────────────────────────────────────────────
# A10 — n_trials se deriva del registro, no se inyecta
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_n_trials_comes_from_the_registry(tmp_path: Path) -> None:
    returns = _series()
    with_three = _registry(tmp_path / "three", 3)
    with_thirty = _registry(tmp_path / "thirty", 30)
    assert with_three.n_trials == 3
    assert with_thirty.n_trials == 30
    small = experiment_log.deflate_block(returns=returns, registry=with_three)
    large = experiment_log.deflate_block(returns=returns, registry=with_thirty)
    assert small["dsr"] != large["dsr"]
    assert _number(large["dsr"]) < _number(small["dsr"]), "mas intentos ⇒ deflacion mas severa"
    assert large["n_trials"] == 30
    payload = with_thirty.to_payload()
    assert payload["n_trials"] == 30
    assert payload["registry_sha256"] == with_thirty.registry_sha256
    assert len(cast("list[object]", payload["variant_ids"])) == 30
    digests = [cast("str", entry["run_sha256"]) for entry in _rows(payload["entries"])]
    assert digests == sorted(digests)

    # el CLI no acepta `--n-trials`: exit 2 sin escribir nada
    completed = _cli(
        "--as-of", AS_OF.isoformat(), "--n-trials", "30", "--runs-root", str(tmp_path / "cli")
    )
    assert completed.returncode == 2
    assert "n-trials" in completed.stderr
    assert not (tmp_path / "cli").exists()


# ─────────────────────────────────────────────────────────────────────────────
# A11 — formato del registro
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_the_registry_format_is_the_declared_one(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    record = _record(runs_root, "format-00", sharpe=0.25)
    directory = runs_root / record.run_sha256
    assert sorted(path.name for path in directory.iterdir()) == [
        experiment_log.CONFIG_FILE,
        experiment_log.RESULT_FILE,
        experiment_log.SUMMARY_FILE,
    ]
    document = json.loads((directory / experiment_log.CONFIG_FILE).read_text(encoding="utf-8"))
    assert document["config"] == record.config.to_payload()
    assert document["hash_format"] == experiment_log.EXPERIMENT_HASH_FORMAT
    result = json.loads((directory / experiment_log.RESULT_FILE).read_text(encoding="utf-8"))
    assert result["result"]["sharpe_per_session"] == 0.25
    summary = (directory / experiment_log.SUMMARY_FILE).read_text(encoding="utf-8")
    assert record.run_sha256 in summary
    assert "por sesion" in summary
    assert "runs/" in summary
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "runs/" in gitignore
    assert "gitignorad" in SOURCE
    assert "gitignorad" in summary


# ─────────────────────────────────────────────────────────────────────────────
# A12 — identidad = hash del contenido de la configuracion
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_the_identity_is_the_configuration_content(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    config = _config("identity-00", seed=7, hyperparameters={"k": 3, "alpha": 0.1})
    expected = hashlib.sha256(canonical_text(config.to_payload()).encode("utf-8")).hexdigest()
    assert experiment_log.run_sha256(config) == expected
    assert "canonical_text" in experiment_log.EXPERIMENT_HASH_FORMAT

    first = experiment_log.record_experiment(
        runs_root=runs_root,
        config=config,
        result=experiment_log.ExperimentResult(sharpe_per_session=0.1, n_observations=8),
        as_of=AS_OF,
    )
    second = experiment_log.record_experiment(
        runs_root=runs_root,
        config=config,
        result=experiment_log.ExperimentResult(sharpe_per_session=0.1, n_observations=8),
        as_of=AS_OF_LATER,
    )
    assert first.run_sha256 == second.run_sha256 == expected
    assert first.directory == runs_root / expected
    assert first.directory.is_dir()

    variants = {
        "hiperparametro": _config("identity-00", seed=7, hyperparameters={"k": 4}),
        "semilla": _config("identity-00", seed=8, hyperparameters={"k": 3, "alpha": 0.1}),
        "ventana": _config(
            "identity-00",
            seed=7,
            hyperparameters={"k": 3, "alpha": 0.1},
            window={"kind": "index", "start": 1, "stop": 8},
        ),
        "variante": _config("identity-01", seed=7, hyperparameters={"k": 3, "alpha": 0.1}),
    }
    for label, variant in variants.items():
        assert experiment_log.run_sha256(variant) != expected, label
    # ni la ruta ni el instante entran en la identidad: la configuracion de disco es la misma
    stored = json.loads((runs_root / expected / experiment_log.CONFIG_FILE).read_text("utf-8"))
    assert stored["config"] == config.to_payload()
    assert not (runs_root / experiment_log.run_sha256(_config("identity-00", seed=8))).exists()


# ─────────────────────────────────────────────────────────────────────────────
# A13 — reescritura inmutable e idempotente
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_rewriting_is_immutable_and_idempotent(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    first = _record(runs_root, "rewrite-00", sharpe=0.1)
    assert first.outcome is WriteOutcome.CREATED
    assert first.written is True
    assert WriteOutcome.__module__ == "cfdtrader.data.store"

    directory = first.directory
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()
    }
    again = _record(runs_root, "rewrite-00", sharpe=0.1)
    assert again.outcome is WriteOutcome.UNCHANGED
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()
    }
    assert after == before

    with pytest.raises(experiment_log.ExperimentRewriteError):
        _record(runs_root, "rewrite-00", sharpe=0.2)
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()
    } == before

    # `write=False` no escribe ni el directorio
    dry = _record(runs_root / "dry", "rewrite-00", sharpe=0.1, write=False)
    assert dry.outcome is None and dry.written is False
    assert not (runs_root / "dry").exists()


# ─────────────────────────────────────────────────────────────────────────────
# A14 — el resultado tambien es inmutable (y delata el no-determinismo)
# ─────────────────────────────────────────────────────────────────────────────
def test_a14_the_result_is_immutable_and_detects_non_determinism(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    first = _record(runs_root, "result-00", sharpe=0.1)
    files = {
        name: (first.directory / name).read_bytes()
        for name in sorted(path.name for path in first.directory.iterdir())
    }
    second = _record(runs_root, "result-00", sharpe=0.1)
    assert second.outcome is WriteOutcome.UNCHANGED
    assert {name: (first.directory / name).read_bytes() for name in files} == files

    script = (
        "from datetime import UTC, datetime\n"
        "from pathlib import Path\n"
        "from cfdtrader.analysis import experiment_log as el\n"
        "config = el.ExperimentConfig(variant_id='result-00', features=('synthetic_random_draw',), "
        "hyperparameters={'kind': 'probe'}, seed=1, series_id='SYNTHETIC', "
        "window={'kind': 'index', 'start': 0, 'stop': 8})\n"
        "record = el.record_experiment(runs_root=Path('__ROOT__'), config=config, "
        "result=el.ExperimentResult(sharpe_per_session=0.1, n_observations=8), "
        "as_of=datetime(2026, 9, 21, tzinfo=UTC))\n"
        "print(record.outcome.value)\n"
    )
    other_root = tmp_path / "other"
    completed = subprocess.run(  # noqa: S603 - comando fijo (el interprete de la sesion)
        [sys.executable, "-c", script.replace("__ROOT__", str(other_root))],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "created"
    fresh = other_root / first.run_sha256
    assert {name: (fresh / name).read_bytes() for name in files} == files

    with pytest.raises(experiment_log.ExperimentRewriteError):
        _record(runs_root, "result-00", sharpe=0.9)


# ─────────────────────────────────────────────────────────────────────────────
# A15 — sin reloj y con --as-of obligatorio
# ─────────────────────────────────────────────────────────────────────────────
def test_a15_there_is_no_clock_and_as_of_is_mandatory(tmp_path: Path) -> None:
    clock_calls = re.compile(r"datetime\.now\(|datetime\.utcnow\(|date\.today\(|time\.time\(")
    for module in (SOURCE, Path(str(overfitting.__file__)).read_text(encoding="utf-8")):
        assert not clock_calls.search(module), module[:40]

    runs_root = tmp_path / "runs"
    reports = tmp_path / "reports"
    missing = _cli("--runs-root", str(runs_root), "--reports-dir", str(reports))
    assert missing.returncode == 2
    assert "falta --as-of" in missing.stderr
    assert not runs_root.exists() and not reports.exists()

    invalid = _cli("--as-of", "ayer", "--runs-root", str(runs_root), "--reports-dir", str(reports))
    assert invalid.returncode == 2
    assert "ISO-8601" in invalid.stderr
    assert not runs_root.exists() and not reports.exists()

    accepted = _cli(
        "--as-of", AS_OF.isoformat(), "--runs-root", str(runs_root), "--reports-dir", str(reports)
    )
    assert accepted.returncode == 0
    assert (reports / f"{experiment_log.REPORT_PREFIX}_{AS_OF.date().isoformat()}.json").is_file()


# ─────────────────────────────────────────────────────────────────────────────
# A18 — veredicto agregado mecanico y reutilizado
# ─────────────────────────────────────────────────────────────────────────────
def test_a18_the_aggregate_is_the_one_from_issue_9() -> None:
    assert experiment_log.aggregate_gate is phase0_report.aggregate_gate
    cases = {
        ("significant", "not_detected"): "pass",
        ("not_significant", "not_detected"): "fail",
        ("significant", "detected"): "fail",
        ("not_significant", "detected"): "fail",
        ("significant", "not_evaluable"): "not_evaluable",
        ("not_evaluable", "not_detected"): "not_evaluable",
        ("not_evaluable", "not_evaluable"): "not_evaluable",
    }
    for (dsr, pbo), expected in cases.items():
        assert experiment_log.aggregate_verdict(dsr_verdict=dsr, pbo_verdict=pbo) == expected
    with pytest.raises(experiment_log.VerdictConsistencyError):
        experiment_log.aggregate_verdict(dsr_verdict="inventado", pbo_verdict="detected")
    # forzar un `pass` con una mitad no evaluable es error de consistencia
    with pytest.raises(experiment_log.VerdictConsistencyError):
        experiment_log.require_consistent_aggregate(
            gate="pass", dsr_half="not_evaluable", pbo_half="pass"
        )
    with pytest.raises(experiment_log.VerdictConsistencyError):
        experiment_log.require_consistent_aggregate(gate="fail", dsr_half="pass", pbo_half="pass")
    experiment_log.require_consistent_aggregate(gate="pass", dsr_half="pass", pbo_half="pass")
    assert experiment_log.aggregate_gate("pass", "fail") is phase0_report.GateVerdict.FAIL


# ─────────────────────────────────────────────────────────────────────────────
# A20 — JSON estricto, sin nan ni inf
# ─────────────────────────────────────────────────────────────────────────────
def test_a20_the_payloads_are_strict_json() -> None:
    lossless = tuple(0.0005 * (index % 9 + 1) for index in range(200))
    zero_variance = overfitting.deflated_sharpe_ratio(_series(), n_trials=10, sr_variance=0.0)
    payloads: dict[str, object] = {
        "ruido": overfitting.deflated_sharpe_ratio(
            _column(overfitting.noise_matrix(), 0),
            n_trials=20,
            sr_variance=overfitting.variant_sharpe_variance(
                [
                    sharpe_ratio(_column(overfitting.noise_matrix(), index), annualization=1)
                    for index in range(20)
                ]
            ),
        ).to_payload(),
        "senal": overfitting.deflated_sharpe_ratio(
            _column(overfitting.signal_matrix(), 0), n_trials=20, sr_variance=0.002
        ).to_payload(),
        "degenerado": experiment_log.pbo_block(returns_matrix=((0.1,),) * 8, blocks=4),
        "cero_perdidas": overfitting.deflated_sharpe_ratio(
            lossless, n_trials=10, sr_variance=0.0004
        ).to_payload(),
        "varianza_cero": zero_variance.to_payload(),
    }
    assert zero_variance.sr0_expected_max == 0.0
    assert zero_variance.deflation == overfitting.DEFLATION_NONE
    for name, payload in payloads.items():
        text = json.dumps(payload, allow_nan=False)
        assert "NaN" not in text and "Infinity" not in text, name
        assert json.loads(text) == payload
    with pytest.raises(ValueError, match="Out of range float values"):
        json.dumps({"malo": float("nan")}, allow_nan=False)
    with pytest.raises(experiment_log.ExperimentLogError):
        experiment_log._jsonable(float("inf"), where="sonda")  # pyright: ignore[reportPrivateUsage]
    assert experiment_log._jsonable(0.1, where="sonda") == 0.1  # pyright: ignore[reportPrivateUsage]


# ─────────────────────────────────────────────────────────────────────────────
# A21 — umbrales constantes, no ajustables
# ─────────────────────────────────────────────────────────────────────────────
def test_a21_the_thresholds_are_constants_and_the_cli_cannot_move_them(
    tmp_path: Path, report: experiment_log.OverfittingReport
) -> None:
    assert overfitting.PBO_MAX == 0.20
    assert overfitting.DSR_CONFIDENCE_LEVEL == 0.95
    constants = _mapping(_payload(report)["constants"])
    provenance = cast("str", constants["provenance"])
    assert "§11.4" in provenance
    assert "§11.6" in provenance
    assert _number(constants["pbo_max"]) == overfitting.PBO_MAX
    assert _number(constants["dsr_confidence_level"]) == overfitting.DSR_CONFIDENCE_LEVEL
    for flag, value in (("--pbo-max", "0.5"), ("--confidence-level", "0.5")):
        completed = _cli(
            "--as-of", AS_OF.isoformat(), flag, value, "--runs-root", str(tmp_path / "cli")
        )
        assert completed.returncode == 2
        assert "unrecognized arguments" in completed.stderr
    assert not (tmp_path / "cli").exists()
    # el umbral no se deriva del resultado observado: es un literal del fuente. Se lee del
    # AST (no por coincidencia de texto) para que el tipo declarado no importe.
    tree = ast.parse(Path(str(overfitting.__file__)).read_text(encoding="utf-8"))
    literals = {
        node.target.id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert literals.get("PBO_MAX") == 0.20 and isinstance(literals["PBO_MAX"], float)
    assert literals.get("DSR_CONFIDENCE_LEVEL") == 0.95
    assert isinstance(literals["DSR_CONFIDENCE_LEVEL"], float)


# ─────────────────────────────────────────────────────────────────────────────
# A22 — not_evaluable explicito, nunca degradado
# ─────────────────────────────────────────────────────────────────────────────
def test_a22_not_evaluable_is_explicit_and_never_degraded(tmp_path: Path) -> None:
    single = _registry(tmp_path / "single", 1)
    assert single.n_trials == 1
    with pytest.raises(overfitting.InsufficientTrialsError):
        _ = single.sr_variance
    payload = single.to_payload()
    assert payload["sr_variance"] is None
    assert payload["sr_variance_state"] == "not_evaluable"
    assert payload["reason"] and payload["follow_up"]
    assert payload["rule"]

    block = experiment_log.deflate_block(returns=_series(), registry=single)
    assert block["state"] == "not_evaluable"
    assert block["verdict"] == overfitting.VERDICT_NOT_EVALUABLE
    assert "dsr" not in block and "sr0_expected_max" not in block
    assert block["reason"] and block["follow_up"]

    pbo = experiment_log.pbo_block(returns_matrix=((0.1,),) * 8, blocks=4)
    assert pbo["state"] == "not_evaluable" and pbo["reason"]
    assert "not_evaluable" in cast("str", pbo["note"]) and "pass" in cast("str", pbo["note"])

    with pytest.raises(experiment_log.EmptyRegistryError):
        experiment_log.load_registry(tmp_path / "vacio")


# ─────────────────────────────────────────────────────────────────────────────
# A23 — pnl_net_pct es null: la tension se declara
# ─────────────────────────────────────────────────────────────────────────────
def test_a23_the_null_net_return_is_declared_never_filled(
    tmp_path: Path, report: experiment_log.OverfittingReport
) -> None:
    published = _payload(report)
    net = _mapping(published["net_metrics"])
    assert net["state"] == "not_computable"
    assert net["where"] == "cfdtrader.backtest.metrics.calculate_metrics"
    assert net["follow_up"] == ["#62", "#60"]
    assert "#64" in cast("str", net["reason"])
    assert "supuesto" in cast("str", net["reason"])
    assert published["is_validation"] is False
    assert published["basis"] == experiment_log.BASIS_DECLARED_COST
    # el mismo bloque que publica el informe de Fase 1 de #18: el vocabulario se comparte
    assert backtest_report.NET_METRICS["state"] == net["state"]
    assert backtest_report.NET_METRICS["follow_up"] == net["follow_up"]

    runs_root = tmp_path / "runs"
    record = _record(runs_root, "net-00", sharpe=0.1)
    stored = json.loads(
        (record.directory / experiment_log.RESULT_FILE).read_text(encoding="utf-8")
    )["result"]
    assert stored["basis"] == "declared_cost"
    assert stored["is_validation"] is False
    assert stored["net_metrics"]["state"] == "not_computable"
    assert stored["net_metrics"]["follow_up"] == ["#62", "#60"]
    # ningun campo del informe afirma que la estrategia este validada
    text = json.dumps(published, ensure_ascii=False)
    assert 'is_validation": true' not in text
    assert "validada" in text


# ─────────────────────────────────────────────────────────────────────────────
# A24 — errores tipados con raiz y propagacion sin envolver
# ─────────────────────────────────────────────────────────────────────────────
def test_a24_the_errors_are_typed_and_the_foreign_ones_are_not_wrapped(tmp_path: Path) -> None:
    assert issubclass(experiment_log.ExperimentLogError, Exception)
    for error in (
        experiment_log.ExperimentRewriteError,
        experiment_log.EmptyRegistryError,
        experiment_log.RegistryIntegrityError,
        experiment_log.TrialsMismatchError,
        experiment_log.MissingAsOfError,
        experiment_log.InvalidAsOfError,
        experiment_log.VerdictConsistencyError,
    ):
        assert issubclass(error, experiment_log.ExperimentLogError)
    for error in overfitting.__all__:
        candidate = getattr(overfitting, error)
        if isinstance(candidate, type) and issubclass(candidate, Exception):
            assert issubclass(candidate, overfitting.OverfittingError)

    registry = _registry(tmp_path / "runs", 3)
    assert (
        experiment_log.deflate_block(returns=_series(), registry=registry)["state"] == "evaluated"
    )
    # un error de #15 sale con su tipo original: no lo envuelve la capa nueva
    with pytest.raises(MetricsInputError):
        experiment_log.deflate_block(returns=(-1.5, 0.1, 0.2), registry=registry)
    caught = {
        node.type.id
        for node in ast.walk(ast.parse(SOURCE))
        if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name)
    }
    assert not caught & {"SplitsError", "EngineInputError", "MetricsInputError"}
    # y el error tipado del registro corrupto: la identidad no cuadra con su contenido
    bad = tmp_path / "runs" / ("0" * 64)
    bad.mkdir(parents=True)
    (bad / experiment_log.CONFIG_FILE).write_text('{"config": {}}', encoding="utf-8")
    (bad / experiment_log.RESULT_FILE).write_text('{"result": {}}', encoding="utf-8")
    with pytest.raises(experiment_log.RegistryIntegrityError):
        experiment_log.load_registry(tmp_path / "runs")


# ─────────────────────────────────────────────────────────────────────────────
# A25 — frontera de capas verificada
# ─────────────────────────────────────────────────────────────────────────────
def test_a25_the_layer_boundary_holds() -> None:
    tree = ast.parse(Path(str(overfitting.__file__)).read_text(encoding="utf-8"))
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(
        module.startswith(("cfdtrader.data", "cfdtrader.analysis", "duckdb", "polars", "scipy"))
        for module in imported
    )
    # `__future__` entra por la convencion del proyecto (todas las anotaciones son perezosas),
    # no por una dependencia: la lista sigue siendo cerrada y sin capas prohibidas.
    assert imported == {
        "__future__",
        "itertools",
        "math",
        "dataclasses",
        "statistics",
        "typing",
        "collections.abc",
        "numpy",
        "numpy.typing",
        "cfdtrader.backtest.metrics",
    }

    for source_file in sorted((REPO_ROOT / "src" / "cfdtrader" / "backtest").glob("*.py")):
        file_tree = ast.parse(source_file.read_text(encoding="utf-8"))
        modules = {
            node.module or "" for node in ast.walk(file_tree) if isinstance(node, ast.ImportFrom)
        } | {
            alias.name
            for node in ast.walk(file_tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert not any(
            module.startswith(("cfdtrader.data", "duckdb", "polars")) for module in modules
        ), source_file.name

    log_tree = ast.parse(SOURCE)
    from_data = {
        (node.module or "")
        for node in ast.walk(log_tree)
        if isinstance(node, ast.ImportFrom) and "cfdtrader.data" in (node.module or "")
    }
    assert from_data == {"cfdtrader.data.store"}
    names = {
        alias.name
        for node in ast.walk(log_tree)
        if isinstance(node, ast.ImportFrom) and node.module == "cfdtrader.data.store"
        for alias in node.names
    }
    assert names == {"WriteOutcome"}


# ─────────────────────────────────────────────────────────────────────────────
# A26 — tests, uno por criterio
# ─────────────────────────────────────────────────────────────────────────────
def test_a26_there_is_one_test_per_criterion() -> None:
    names: dict[str, list[str]] = {}
    for filename in ("test_overfitting.py", "test_experiment_log.py"):
        text = (REPO_ROOT / "tests" / filename).read_text(encoding="utf-8")
        tree = ast.parse(text)
        functions = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_a")
        ]
        names[filename] = functions
        assert "tmp_path" in text
    found = [name for functions in names.values() for name in functions]
    assert len(found) == 35, found
    for number in range(1, 36):
        assert len([name for name in found if name.startswith(f"test_a{number}_")]) == 1, number
    assert len(names["test_overfitting.py"]) + len(names["test_experiment_log.py"]) == 35
    # ningun test escribe en el `runs/` ni en el `data/` del repositorio: la raiz de toda
    # escritura es temporal. Se comprueba sobre los argumentos reales (AST) y no por
    # coincidencia de texto, que se encontraria a si misma en esta misma linea.
    for filename in names:
        file_tree = ast.parse((REPO_ROOT / "tests" / filename).read_text(encoding="utf-8"))
        roots = [
            ast.unparse(keyword.value)
            for node in ast.walk(file_tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg in {"runs_root", "reports_dir"}
        ]
        assert roots, filename
        for root in roots:
            assert "REPOSITORY_RUNS" not in root and "REPORTS" not in root, (filename, root)


# ─────────────────────────────────────────────────────────────────────────────
# A27 — guardia del runs/ del repositorio
# ─────────────────────────────────────────────────────────────────────────────
def test_a27_the_repository_runs_directory_is_guarded(tmp_path: Path) -> None:
    conftest_source = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "_repository_runs_is_untouched" in conftest_source
    assert 'REPOSITORY_RUNS = REPO_ROOT / "runs"' in conftest_source
    assert "fingerprint(REPOSITORY_RUNS)" in conftest_source

    import conftest

    assert conftest.REPOSITORY_RUNS == REPOSITORY_RUNS
    guard = conftest._repository_runs_is_untouched  # pyright: ignore[reportPrivateUsage]
    # `_fixture_function_marker` es el nombre de esta version de pytest; `_pytestfixturefunction`
    # el de la anterior. Se admiten los dos en vez de apostar por uno solo.
    marker = getattr(guard, "_fixture_function_marker", None) or getattr(
        guard, "_pytestfixturefunction", None
    )
    assert marker is not None, "la guardia no esta declarada como fixture de pytest"
    assert marker.scope == "session"
    assert marker.autouse is True

    # la guardia no es decorativa: una escritura en el `runs/` del repositorio cambia la huella
    before = conftest.fingerprint(REPOSITORY_RUNS)
    probe = REPOSITORY_RUNS / "a27-guard-probe"
    assert not probe.exists()
    try:
        subprocess.run(  # noqa: S603 - comando fijo (el interprete de la sesion)
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(probe)!r}).write_text('x')",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        assert set(conftest.fingerprint(REPOSITORY_RUNS)) - set(before), (
            "la huella no ha detectado la escritura de prueba"
        )
    finally:
        probe.unlink(missing_ok=True)
    assert conftest.fingerprint(REPOSITORY_RUNS) == before

    # y la misma sesion de pytest falla si la escritura no se limpia: es la guardia, no un adorno
    suite = tmp_path / "suite"
    suite.mkdir()
    repo_conftest = REPO_ROOT / "tests" / "conftest.py"
    (suite / "conftest.py").write_text(
        "import importlib.util\n"
        "\n"
        "_spec = importlib.util.spec_from_file_location(\n"
        f"    '_repo_conftest', {str(repo_conftest)!r}\n"
        ")\n"
        "_module = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_module)\n"
        "_repository_runs_is_untouched = _module._repository_runs_is_untouched\n",
        encoding="utf-8",
    )
    (suite / "test_write.py").write_text(
        f"from pathlib import Path\n\n\ndef test_write() -> None:\n"
        f"    Path({str(probe)!r}).write_text('x', encoding='utf-8')\n",
        encoding="utf-8",
    )
    try:
        fired = subprocess.run(  # noqa: S603 - comando fijo (el interprete de la sesion)
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(suite)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert fired.returncode != 0, fired.stdout
        assert "ha modificado el runs/ del repositorio" in fired.stdout + fired.stderr
    finally:
        probe.unlink(missing_ok=True)
    assert conftest.fingerprint(REPOSITORY_RUNS) == before
    assert "runs/" in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# A28 — determinismo entre procesos
# ─────────────────────────────────────────────────────────────────────────────
def test_a28_the_registry_is_deterministic_across_processes(tmp_path: Path) -> None:
    script = (
        "import hashlib, json\n"
        "from datetime import UTC, datetime\n"
        "from pathlib import Path\n"
        "from cfdtrader.analysis import experiment_log as el\n"
        "root = Path('__ROOT__')\n"
        "for index in range(2):\n"
        "    config = el.ExperimentConfig(variant_id=f'cross-{index}', "
        "features=('synthetic_random_draw',), hyperparameters={'kind': 'cross', 'i': index}, "
        "seed=index + 1, series_id='SYNTHETIC', window={'kind': 'index', 'start': 0, 'stop': 8})\n"
        "    el.record_experiment(runs_root=root, config=config, "
        "result=el.ExperimentResult(sharpe_per_session=0.01 * (index + 1), n_observations=8), "
        "as_of=datetime(2026, 9, 19, 12, tzinfo=UTC))\n"
        "digests = sorted(path.name for path in root.iterdir())\n"
        "print(json.dumps(digests))\n"
        "for digest in digests:\n"
        "    for name in ('config.json', 'result.json'):\n"
        "        blob = (root / digest / name).read_bytes()\n"
        "        print(digest, name, hashlib.sha256(blob).hexdigest())\n"
    )
    results: list[str] = []
    for seed in ("0", "1", "random"):
        root = tmp_path / f"runs-{seed}"
        completed = subprocess.run(  # noqa: S603 - comando fijo (el interprete de la sesion)
            [sys.executable, "-c", script.replace("__ROOT__", str(root))],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        results.append(completed.stdout)
    assert len(set(results)) == 1

    # segunda pasada en el mismo proceso: no-op idempotente, mismos bytes
    runs_root = tmp_path / "second"
    first = _record(runs_root, "second-00", sharpe=0.3)
    payload_before = (first.directory / experiment_log.RESULT_FILE).read_bytes()
    assert _record(runs_root, "second-00", sharpe=0.3).outcome is WriteOutcome.UNCHANGED
    assert (first.directory / experiment_log.RESULT_FILE).read_bytes() == payload_before


# ─────────────────────────────────────────────────────────────────────────────
# A29 — puertas en verde (la parte que se puede medir desde el test)
# ─────────────────────────────────────────────────────────────────────────────
def test_a29_the_public_surface_is_documented_and_pragma_free() -> None:
    for module in (overfitting, experiment_log):
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        assert source.count("# pragma: no cover") <= 1, module.__name__
    cli_source = Path(str(experiment_log.__file__)).read_text(encoding="utf-8")
    assert cli_source.count("# pragma: no cover") == 1
    assert 'if __name__ == "__main__":  # pragma: no cover' in cli_source
    for module in (overfitting, experiment_log):
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        for public in module.__all__:
            if public.startswith("_"):
                continue
            attribute = getattr(module, public)
            if (callable(attribute) and not isinstance(attribute, type)) or (
                isinstance(attribute, type) and issubclass(attribute, Exception)
            ):
                assert (attribute.__doc__ or "").strip(), f"{module.__name__}.{public}"


# ─────────────────────────────────────────────────────────────────────────────
# A30 — fronteras legibles por maquina
# ─────────────────────────────────────────────────────────────────────────────
def test_a30_the_boundaries_are_machine_readable(report: experiment_log.OverfittingReport) -> None:
    for table in (overfitting.OVERFITTING_DOES_NOT_DO, overfitting.FOLLOW_UPS):
        assert isinstance(table, tuple)
        for entry in table:
            assert set(entry) == {"id", "issue", "reason"}
            assert entry["issue"].startswith("#")
            assert len(entry["reason"]) > 20
        assert len({entry["id"] for entry in table}) == len(table)
    assert (
        frozenset(
            {
                "no_es_la_suite_de_integridad",
                "no_mide_el_slippage",
                "no_decide_r_ni_umbrales",
                "no_es_cpcv",
                "no_toca_el_holdout",
                "no_entrena_modelos",
                "no_construye_features",
                "no_usa_el_store",
                "no_retencion_ops",
                "no_publica_metricas_netas",
                "no_valida_la_estrategia",
            }
        )
        == BOUNDARY_IDS
    )
    assert (
        frozenset(
            {
                "integrity_suite",
                "slippage_measurement",
                "r_and_thresholds",
                "cpcv_scheme",
                "final_holdout",
                "phase2_logistic",
                "phase2_boosting",
                "features",
                "retention",
                "documentation",
                "backlog_regularisation",
            }
        )
        == FOLLOW_UP_IDS
    )
    issues = {entry["issue"] for entry in overfitting.FOLLOW_UPS} | {
        entry["issue"] for entry in overfitting.OVERFITTING_DOES_NOT_DO
    }
    for required in ("#17", "#62", "#60", "#67", "#68", "#28/#29", "#19-#23", "#44", "#65"):
        assert required in issues, required
    boundaries = _mapping(_payload(report)["boundaries"])
    assert boundaries["follow_ups"] == [dict(entry) for entry in overfitting.FOLLOW_UPS]
    assert boundaries["overfitting_does_not_do"] == [
        dict(entry) for entry in overfitting.OVERFITTING_DOES_NOT_DO
    ]


# ─────────────────────────────────────────────────────────────────────────────
# A31 — CLI honesta
# ─────────────────────────────────────────────────────────────────────────────
def test_a31_the_cli_is_honest(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    reports = tmp_path / "reports"
    assert (
        experiment_log.main(
            [
                "--as-of",
                AS_OF.isoformat(),
                "--runs-root",
                str(runs_root),
                "--reports-dir",
                str(reports),
            ]
        )
        == 0
    )
    stem = f"{experiment_log.REPORT_PREFIX}_{AS_OF.date().isoformat()}"
    assert (reports / f"{stem}.json").is_file()
    assert (reports / f"{stem}.md").is_file()
    assert len([path for path in runs_root.iterdir() if path.is_dir()]) == 40
    payload = json.loads((reports / f"{stem}.json").read_text(encoding="utf-8"))
    assert payload["report_date"] == AS_OF.date().isoformat()
    assert payload["generated_at"] == AS_OF.isoformat()

    dry_runs = tmp_path / "dry-runs"
    dry_reports = tmp_path / "dry-reports"
    assert (
        experiment_log.main(
            [
                "--dry-run",
                "--as-of",
                AS_OF.isoformat(),
                "--runs-root",
                str(dry_runs),
                "--reports-dir",
                str(dry_reports),
            ]
        )
        == 0
    )
    assert not dry_runs.exists() and not dry_reports.exists()

    assert experiment_log.main(["--runs-root", str(tmp_path / "none")]) == 2
    assert not (tmp_path / "none").exists()


# ─────────────────────────────────────────────────────────────────────────────
# A32 — el registro no usa el Store
# ─────────────────────────────────────────────────────────────────────────────
def test_a32_the_registry_does_not_use_the_store(report: experiment_log.OverfittingReport) -> None:
    tree = ast.parse(SOURCE)
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "read_pit" not in attributes
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "Store" not in called
    assert "cfdtrader.data.store" in SOURCE
    assert "json" in SOURCE and "pathlib" in SOURCE
    assert "polars" not in SOURCE and "duckdb" not in SOURCE
    divergence = cast("str", _mapping(_payload(report)["boundaries"])["store_divergence"])
    assert "ops.backtest_runs" in divergence
    assert "#65" in divergence and "#44" in divergence
    assert "tech_stack.md" in divergence


# ─────────────────────────────────────────────────────────────────────────────
# A35 — informe legible desde el mismo payload determinista
# ─────────────────────────────────────────────────────────────────────────────
def test_a35_the_markdown_comes_from_the_same_payload(
    report: experiment_log.OverfittingReport,
) -> None:
    markdown = experiment_log.render_markdown(report)
    assert report.report_sha256 in markdown
    assert (
        hashlib.sha256(canonical_text(report.payload).encode("utf-8")).hexdigest()
        == report.report_sha256
    )
    published = {**report.payload, "report_sha256": report.report_sha256}
    assert json.loads(report.json_text()) == published
    assert "runs/" in markdown
    for outcome in report.experiments:
        assert markdown.count(f"`{outcome.selected.config.variant_id}`") >= 1
        for row in outcome.rows:
            assert f"`{row['variant_id']}`" in markdown
            assert str(row["run_sha256"]) in markdown
    assert markdown.count("| `noise-") == 20
    assert markdown.count("| `signal-") == 20
    signal = report.experiments[1]
    assert f"| {signal.pbo['pbo']!r} |" in markdown
    assert f"| {signal.dsr['dsr']!r} |" in markdown
    assert f"`{signal.gate['aggregate']}`" in markdown
    assert markdown.count("`not_significant`") >= 20


# ─────────────────────────────────────────────────────────────────────────────
# Apoyo (no es un criterio): integridad del registro y salidas de error del CLI
# ─────────────────────────────────────────────────────────────────────────────
def test_registry_integrity_and_failure_paths_are_typed(tmp_path: Path) -> None:
    """Ramas de error del registro y del CLI: tipadas y sin inventar un numero (A19-A22)."""
    jsonable = experiment_log._jsonable  # pyright: ignore[reportPrivateUsage]
    as_json_object = experiment_log._json_object  # pyright: ignore[reportPrivateUsage]
    as_utc = experiment_log._as_utc  # pyright: ignore[reportPrivateUsage]
    parse_as_of = experiment_log._parse_as_of  # pyright: ignore[reportPrivateUsage]
    synthetic_matrix = experiment_log._synthetic_matrix  # pyright: ignore[reportPrivateUsage]

    # `Decimal` viaja como cadena exacta; `Mapping` y secuencia se traducen; lo demas es error
    assert jsonable(Decimal("1.2300"), where="sonda") == "1.2300"
    assert jsonable({"b": 1, "a": (2, 3)}, where="sonda") == {"b": 1, "a": [2, 3]}
    with pytest.raises(experiment_log.ExperimentLogError):
        jsonable(object(), where="sonda")
    with pytest.raises(experiment_log.ExperimentLogError):
        as_json_object([1, 2], where="sonda")
    # un instante sin zona se interpreta como UTC; el Sharpe registrado tiene que ser finito
    assert as_utc(datetime(2026, 9, 19, 12)).tzinfo is UTC
    with pytest.raises(experiment_log.ExperimentLogError):
        experiment_log.ExperimentResult(
            sharpe_per_session=float("inf"), n_observations=8
        ).to_payload()
    with pytest.raises(experiment_log.InvalidAsOfError):
        parse_as_of("ayer")
    with pytest.raises(experiment_log.ExperimentLogError):
        synthetic_matrix("inventado")

    # registro incompleto y con JSON no valido: error tipado, nunca una lista vacia
    runs_root = tmp_path / "runs"
    broken = runs_root / ("f" * 64)
    broken.mkdir(parents=True)
    with pytest.raises(experiment_log.RegistryIntegrityError):
        experiment_log.load_registry(runs_root)
    (broken / experiment_log.CONFIG_FILE).write_text("{no json", encoding="utf-8")
    (broken / experiment_log.RESULT_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(experiment_log.RegistryIntegrityError):
        experiment_log.load_registry(runs_root)

    # un `result.json` que declara otra identidad o un Sharpe no numerico
    tampered = tmp_path / "tampered"
    good = _record(tampered, "integrity-00", sharpe=0.1)
    result_path = good.directory / experiment_log.RESULT_FILE
    document = json.loads(result_path.read_text(encoding="utf-8"))
    document["run_sha256"] = "0" * 64
    result_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(experiment_log.RegistryIntegrityError):
        experiment_log.load_registry(tampered)
    document["run_sha256"] = good.run_sha256
    document["result"]["sharpe_per_session"] = "alto"
    result_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(experiment_log.RegistryIntegrityError):
        experiment_log.load_registry(tampered)

    # un registro de una sola variante no permite derivar `V[SR]` (A22)
    single = _registry(tmp_path / "single", 1)
    with pytest.raises(experiment_log.TrialsMismatchError):
        experiment_log.require_trials_match_registry(n_trials=1, sr_variance=0.0, registry=single)

    # el CLI sale 2 y no escribe nada cuando el registro esta corrupto
    reports = tmp_path / "reports"
    failed = _cli(
        "--as-of", AS_OF.isoformat(), "--runs-root", str(runs_root), "--reports-dir", str(reports)
    )
    assert failed.returncode == 2
    assert "no se puede emitir el informe de sobreajuste" in failed.stderr
    assert not reports.exists()
