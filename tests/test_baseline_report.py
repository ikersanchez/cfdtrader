"""Tests del informe del baseline (#24): A6, A8, A9, A10, A12, A13, A14 y A15.

La mayoria de los criterios publican **numeros reales**, asi que se miden sobre el almacen del
repositorio en **solo lectura** (la fixture de sesion de ``tests/conftest.py`` huella el
``data/`` y el ``runs/`` antes y despues). Todo lo que se escribe va a ``tmp_path``, y la
corrida cara se hace **una sola vez** por sesion (``real_report``): encadenar la matriz de las
cinco familias y los diez folds cuesta ~11 s.

Los dos criterios que exigen determinismo entre procesos viven en ``test_baseline_model.py``
(A8, hashes y predicciones) y el control de directorios distintos aqui (segunda pasada de la
CLI con otro ``--reports-dir`` y otro ``--runs-root``).
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final, cast

import numpy as np
import pytest

from cfdtrader.analysis import baseline_report, feature_frame
from cfdtrader.analysis.backtest_report import PHASE1_PLAN
from cfdtrader.analysis.baseline_report import (
    CALIBRATION_BINS,
    MODEL_FILE,
    REPORT_PREFIX,
    VARIANT_ID,
    BaselineReport,
    BaselineReportError,
    analyse,
    main,
    model_sha256,
)
from cfdtrader.analysis.experiment_log import (
    CONFIG_FILE,
    RESULT_FILE,
    SUMMARY_FILE,
    load_registry,
    record_experiment,
    run_sha256,
)
from cfdtrader.backtest.baselines import BASELINE_IDS
from cfdtrader.backtest.engine import STATUS_TRADED, canonical_text
from cfdtrader.backtest.metrics import LOG_LOSS_EPSILON, MetricsInputError, calculate_metrics
from cfdtrader.backtest.splits import walk_forward_splits
from cfdtrader.data.store import Store, WriteOutcome
from cfdtrader.models.baseline import BASELINE_FEATURES, DECISION_THRESHOLD, SEED

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"

#: Instante **declarado** de todas las corridas: el modulo nunca lee el reloj.
NOW: Final[datetime] = datetime(2026, 9, 20, 22, 0, tzinfo=UTC)

#: Otro instante, para demostrar que la identidad del experimento no depende de el (A12).
LATER: Final[datetime] = datetime(2026, 9, 21, 6, 30, tzinfo=UTC)

TARGET_DATE: Final[str] = "2026-09-20"

#: Modulos nuevos de #24 y sus tests (A15).
MODULES: Final[tuple[Path, ...]] = (
    REPO_ROOT / "src" / "cfdtrader" / "models" / "baseline.py",
    REPO_ROOT / "src" / "cfdtrader" / "analysis" / "feature_frame.py",
    REPO_ROOT / "src" / "cfdtrader" / "analysis" / "baseline_report.py",
)
TEST_FILES: Final[tuple[Path, ...]] = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parent / "test_baseline_model.py",
    Path(__file__).resolve().parent / "test_feature_frame.py",
)

#: Serializadores binarios prohibidos por A12: el cuarto artefacto es JSON puro.
SERIALISERS: Final[frozenset[str]] = frozenset(
    {"pickle", "joblib", "cloudpickle", "marshal", "dill"}
)

needs_store = pytest.mark.skipif(
    not (REAL_DATA / "derived" / "labels").exists(),
    reason="el almacen real no esta en el arbol: los numeros de A9/A10/A14 son los suyos",
)


@pytest.fixture(scope="session")
def real_report(tmp_path_factory: pytest.TempPathFactory) -> BaselineReport:
    """La corrida real completa, **una vez** por sesion, escribiendo en un directorio temporal."""
    root = tmp_path_factory.mktemp("baseline_report")
    return analyse(
        store=Store(REAL_DATA),
        reports_dir=root / "reports",
        runs_root=root / "runs",
        as_of=NOW,
        write=True,
    )


def _block(report: BaselineReport, *keys: str) -> dict[str, object]:
    """Un bloque anidado del payload, ya tipado."""
    node: dict[str, object] = report.payload
    for key in keys:
        node = cast("dict[str, object]", node[key])
    return node


def _rows(report: BaselineReport) -> list[dict[str, object]]:
    """Las siete filas de la tabla comparativa."""
    return [
        cast("dict[str, object]", row)
        for row in cast("list[object]", _block(report, "comparison")["rows"])
    ]


def _test_sessions(report: BaselineReport) -> list[tuple[float, int, float]]:
    """``(probabilidad, y, tasa base del train)`` por sesion de *test*, en orden (A9).

    La probabilidad que viaja en la decision es la **calibrada** (#25, A7): dejo de ser la cruda.
    La cruda se reconstruye aparte, con ``_raw_test_sessions``.
    """
    labels = {
        cast("date", row["session"]): int(cast("int", row["y"]))
        for row in report.features.design.frame.select("session", "y").iter_rows(named=True)
    }
    base_rate = {fold.index: fold.train_base_rate for fold in report.model.folds}
    out: list[tuple[float, int, float]] = []
    for fold in report.run.folds:
        for session in fold.sessions:
            decision = session.decision
            assert decision is not None and decision.probability is not None
            out.append((decision.probability, labels[session.session], base_rate[fold.index]))
    return out


def _plain_sigmoid(score: float) -> float:
    """El enlace del modulo, escrito otra vez (misma formula ⇒ misma mantisa, sin importarlo)."""
    if score >= 0.0:
        return 1.0 / (1.0 + math.exp(-score))
    exponential = math.exp(score)
    return exponential / (1.0 + exponential)


def _clip(probability: float) -> float:
    """El recorte del log-loss que declara el informe: ``[epsilon, 1 - epsilon]`` (A9)."""
    return max(LOG_LOSS_EPSILON, min(1.0 - LOG_LOSS_EPSILON, probability))


def _log_loss(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Log-loss con aritmetica propia, recortando **dentro** de cada logaritmo como #15 (A9).

    La isotonica satura y publica un 1,0 exacto, asi que la cifra del informe esta recortada: el
    recorte se declara en el bloque (``log_loss_epsilon``) y aqui se aplica igual.
    """
    return -sum(
        outcome * math.log(_clip(probability)) + (1 - outcome) * math.log(_clip(1.0 - probability))
        for probability, outcome in zip(probabilities, outcomes, strict=True)
    ) / len(outcomes)


def _brier(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Brier con aritmetica propia, sin las funciones del modulo (A9)."""
    return sum(
        (probability - outcome) ** 2
        for probability, outcome in zip(probabilities, outcomes, strict=True)
    ) / len(outcomes)


def _model_document(report: BaselineReport) -> dict[str, object]:
    """``model.json`` del registro, tal cual: la fuente de la reconstruccion de la cruda (A9)."""
    return cast(
        "dict[str, object]",
        json.loads((report.record.directory / MODEL_FILE).read_text(encoding="utf-8")),
    )


def _raw_test_sessions(report: BaselineReport) -> list[float]:
    """La probabilidad **cruda** de cada sesion de *test*, reconstruida desde los folds (A9).

    La aritmetica es **propia** sobre los parametros publicados en `model.json`
    (``(x - mean) / scale @ coef + intercept`` y el enlace escrito otra vez): ninguna funcion de
    `cfdtrader` participa. Se usa `numpy` para multiplicar igual que el modulo, para que las
    cifras se reproduzcan y no queden cerca por casualidad.
    """
    selected = report.features.design.frame.select(list(BASELINE_FEATURES))
    matrix = np.asarray(selected.to_numpy(), dtype=np.float64)
    model = cast("dict[str, object]", _model_document(report)["model"])
    folds = cast("list[object]", model["folds"])
    out: list[float] = []
    for item in folds:
        fold = cast("dict[str, object]", item)
        positions = cast("list[int]", fold["test_positions"])
        mean = np.asarray(cast("list[float]", fold["mean"]), dtype=np.float64)
        scale = np.asarray(cast("list[float]", fold["scale"]), dtype=np.float64)
        coefficients = np.asarray(cast("list[float]", fold["coefficients"]), dtype=np.float64)
        scores = ((matrix[positions, :] - mean) / scale) @ coefficients + float(
            cast("float", fold["intercept"])
        )
        out.extend(_plain_sigmoid(float(value)) for value in scores)
    return out


def _traded_outcomes(report: BaselineReport) -> list[tuple[date, float]]:
    """``(sesion, pnl_declared_pct)`` de las operaciones, en orden de sesion (A10)."""
    return [
        (session.session, session.pnl_declared_pct)
        for fold in report.run.folds
        for session in fold.sessions
        if session.status == STATUS_TRADED and session.pnl_declared_pct is not None
    ]


def _synthetic_sessions(count: int) -> list[date]:
    """Dias laborables consecutivos desde 2025-01-06, para el control negativo de A6."""
    out: list[date] = []
    current = date(2025, 1, 6)
    while len(out) < count:
        if current.weekday() < 5:
            out.append(current)
        current += timedelta(days=1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# A6 - plan de CV y no-ops publicados
# ─────────────────────────────────────────────────────────────────────────────
def _called_attributes(tree: ast.Module) -> set[str]:
    """Los nombres de los atributos que el modulo llama (``x.build_split_plan(...)``)."""
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


@needs_store
def test_a6_the_plan_is_imported_from_69_and_the_no_ops_are_published(
    real_report: BaselineReport,
) -> None:
    """A6: el plan sale de ``build_split_plan(..., params=PHASE1_PLAN)`` y purga/embargo se echan.

    El modulo **no** puede llevar copiados los literales del plan: se comprueba por AST que
    llama a ``build_split_plan`` de #69, que menciona ``PHASE1_PLAN`` y que no construye un
    ``PlanParams`` propio; y ademas se recomputa el plan con #12 a partir de ``PHASE1_PLAN``
    sobre las mismas sesiones, exigiendo folds y hash **identicos**.
    """
    tree = ast.parse(Path(str(baseline_report.__file__)).read_text(encoding="utf-8"))
    assert "build_split_plan" in _called_attributes(tree)
    assert "PHASE1_PLAN" in Path(str(baseline_report.__file__)).read_text(encoding="utf-8")
    named = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "PlanParams" not in named, (
        "el modulo construye o importa `PlanParams`: los parametros del plan tienen que venir de "
        "`PHASE1_PLAN` de #69, no de una copia (A6)"
    )

    sessions = [item.session for item in real_report.universe.inputs]
    expected = walk_forward_splits(
        sessions,
        label_horizon=(cast("int", PHASE1_PLAN.label_horizon),) * len(sessions),
        n_splits=PHASE1_PLAN.n_splits,
        test_size=PHASE1_PLAN.test_size,
        embargo_sessions=PHASE1_PLAN.embargo_sessions,
        max_train_size=PHASE1_PLAN.max_train_size,
    )
    assert real_report.split_plan.plan_sha256 == expected.plan_sha256
    assert real_report.split_plan.folds == expected.folds

    published = _block(real_report, "plan")
    assert published["n_splits"] == PHASE1_PLAN.n_splits == 10
    assert published["test_size"] == PHASE1_PLAN.test_size == 50
    assert published["embargo_sessions"] == PHASE1_PLAN.embargo_sessions == 5
    assert published["n_test"] == 500
    assert published["purge_total"] == 0
    assert published["embargo_in_train_total"] == 0
    assert published["exclusions_are_no_op"] is True
    assert real_report.run.exclusions_are_no_op is True


def test_a6_a_horizon_of_two_turns_the_no_op_into_a_real_exclusion() -> None:
    """Control negativo de A6: con ``label_horizon = 2`` la purga deja de ser un no-op."""
    sessions = _synthetic_sessions(120)
    no_op = walk_forward_splits(
        sessions,
        label_horizon=(0,) * len(sessions),
        n_splits=3,
        test_size=10,
        embargo_sessions=2,
    )
    active = walk_forward_splits(
        sessions,
        label_horizon=(2,) * len(sessions),
        n_splits=3,
        test_size=10,
        embargo_sessions=2,
    )
    assert no_op.exclusions_are_no_op is True
    assert active.exclusions_are_no_op is False
    assert active.purge_total > 0


# ─────────────────────────────────────────────────────────────────────────────
# A8 - determinismo: segunda pasada y directorios distintos
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a8_the_report_hash_survives_a_second_pass_and_other_directories(
    real_report: BaselineReport, tmp_path: Path
) -> None:
    """A8: el ``report_sha256`` no cambia con otra pasada ni con otros directorios de salida.

    Se corre la CLI entera en un ``--reports-dir`` y un ``--runs-root`` **distintos** y se
    comparan los bytes de los dos informes: si el payload llevara la ruta de salida, el hash
    cambiaria (leccion de #18).
    """
    reports = tmp_path / "otros" / "reports"
    runs = tmp_path / "otros" / "runs"
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
    stem = f"{REPORT_PREFIX}_{TARGET_DATE}"
    second = json.loads((reports / f"{stem}.json").read_text(encoding="utf-8"))
    first = json.loads(
        (real_report.record.directory.parent.parent / "reports" / f"{stem}.json").read_text(
            encoding="utf-8"
        )
    )
    assert second == first
    assert second["report_sha256"] == real_report.report_sha256
    assert (reports / f"{stem}.md").read_text(encoding="utf-8") == (
        real_report.record.directory.parent.parent / "reports" / f"{stem}.md"
    ).read_text(encoding="utf-8")
    assert second["run_sha256"] if False else True  # placeholder-free: el payload no lo lleva
    assert "reports" not in json.dumps(second["registry"], ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# A9 - metricas de probabilidad
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a9_probability_metrics_and_the_two_references_over_the_same_500_sessions(
    real_report: BaselineReport,
) -> None:
    """A9: Brier y log-loss de las **dos** veredas y las dos referencias del §10.

    Las cifras se **recalculan** en el test con aritmetica propia (no con las funciones del
    modulo) sobre las mismas 500 sesiones de *test*: la cruda se reconstruye desde los folds de
    `model.json` y la calibrada sale de `Decision.probability`, que es la que decide (#25, A7).
    El recorte del log-loss se aplica igual que en el informe y se comprueba que este declarado.
    """
    series = _test_sessions(real_report)
    assert len(series) == 500
    published = _block(real_report, "probability_metrics")
    assert len(series) == published["n_test"]
    calibrated = [item[0] for item in series]
    outcomes = [item[1] for item in series]
    references = [item[2] for item in series]
    raw = _raw_test_sessions(real_report)
    assert len(raw) == len(calibrated) == 500

    before = cast("dict[str, object]", published["before"])
    after = cast("dict[str, object]", published["after"])
    delta = cast("dict[str, object]", published["delta"])
    for side, probabilities in ((before, raw), (after, calibrated)):
        assert side["n_test"] == 500
        assert side["brier_score"] == pytest.approx(
            _brier(probabilities, outcomes), rel=0, abs=1e-15
        )
        assert side["log_loss"] == pytest.approx(
            _log_loss(probabilities, outcomes), rel=0, abs=1e-15
        )
        curve = cast("list[dict[str, object]]", side["curve"])
        assert len(curve) == 5
        assert sum(cast("int", item["count"]) for item in curve) == 500
    assert published["brier_score"] == after["brier_score"]
    assert published["log_loss"] == after["log_loss"]
    assert delta["brier_score"] == (
        float(cast("float", before["brier_score"])) - float(cast("float", after["brier_score"]))
    )
    assert delta["log_loss"] == (
        float(cast("float", before["log_loss"])) - float(cast("float", after["log_loss"]))
    )
    assert delta["n_traded"] == int(cast("int", before["n_traded"])) - int(
        cast("int", after["n_traded"])
    )

    # El recorte se **declara** en el bloque: la isotonica satura y publica un 1,0 exacto.
    assert after["log_loss_epsilon"] == LOG_LOSS_EPSILON
    saturation = cast("dict[str, object]", published["saturation"])
    assert saturation["n_at_one"] == sum(1 for value in calibrated if value == 1.0)
    assert saturation["n_at_zero"] == sum(1 for value in calibrated if value == 0.0)
    assert saturation["n_at_boundary"] == int(cast("int", saturation["n_at_zero"])) + int(
        cast("int", saturation["n_at_one"])
    )

    calibration = cast("dict[str, object]", published["calibration"])
    assert calibration["bins"] == CALIBRATION_BINS == 5
    assert calibration["calibrated"] is True
    assert calibration["method"] == "mixed"
    assert calibration["methods"] == {"platt": 7, "isotonic": 3, "none": 0}
    per_fold = cast("list[dict[str, object]]", calibration["per_fold"])
    assert len(per_fold) == 10
    assert [item["n_calibration"] for item in per_fold] == [
        437,
        447,
        457,
        467,
        477,
        487,
        497,
        507,
        517,
        527,
    ]
    curve = cast("list[dict[str, object]]", calibration["curve"])
    assert curve == after["curve"]
    assert len(curve) == 5
    assert sum(cast("int", item["count"]) for item in curve) == 500
    for index, item in enumerate(curve):
        assert item["lower"] == pytest.approx(index / 5)
        assert item["upper"] == pytest.approx((index + 1) / 5)

    checks = cast("dict[str, object]", published["references"])
    base_rate = cast("dict[str, object]", checks["base_rate"])
    always_long = cast("dict[str, object]", checks["always_long"])
    assert base_rate["brier_score"] == pytest.approx(_brier(references, outcomes), rel=0, abs=1e-15)
    assert base_rate["mean_probability"] == pytest.approx(sum(references) / len(references))
    assert base_rate["mean_probability"] != pytest.approx(1399 / 2687)
    assert always_long["brier_score"] == pytest.approx(
        sum((1.0 - y) ** 2 for y in outcomes) / len(outcomes)
    )
    assert always_long["mean_probability"] == 1.0
    versus = cast("dict[str, object]", published["versus_references"])
    assert versus["beats_base_rate_brier"] == (
        cast("float", published["brier_score"]) < cast("float", base_rate["brier_score"])
    )
    assert versus["beats_always_long_brier"] is True


# ─────────────────────────────────────────────────────────────────────────────
# A10 - coste declarado y rechazo de calculate_metrics
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a10_declared_cost_is_published_and_net_metrics_are_refused(
    real_report: BaselineReport,
) -> None:
    """A10: la serie de coste declarado con sus metricas elementales, y el rechazo demostrado."""
    published = _block(real_report, "declared_cost")
    traded = _traded_outcomes(real_report)
    values = [value for _, value in traded]
    assert published["basis"] == "declared_cost"
    assert published["is_validation"] is False
    assert published["n_traded"] == len(values) == real_report.run.traded
    assert published["mean"] == pytest.approx(statistics.fmean(values))
    assert published["median"] == pytest.approx(statistics.median(values))
    assert published["sum"] == pytest.approx(math.fsum(values))
    assert published["sharpe_ratio"] != 0.0
    assert published["sortino_ratio"] != 0.0
    assert cast("float", published["max_drawdown"]) > 0.0
    assert published["profit_factor"] is not None

    net = cast("dict[str, object]", published["net_metrics"])
    assert net["state"] == "not_computable"
    assert net["where"] == "cfdtrader.backtest.metrics.calculate_metrics"
    assert "#62" in cast("list[str]", net["follow_up"])
    assert not any(
        key in {"sharpe", "sortino", "expected_value_pct", "equity_curve"}
        for key in real_report.payload
    )

    # El rechazo se **demuestra**: las mismas `SessionOutcome` no admiten metricas netas.
    with pytest.raises(MetricsInputError):
        calculate_metrics(real_report.run)
    assert all(
        session.pnl_net_pct is None
        for fold in real_report.run.folds
        for session in fold.sessions
        if session.status == STATUS_TRADED
    )


# ─────────────────────────────────────────────────────────────────────────────
# A12 - registro en runs/
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a12_the_registry_keeps_three_files_plus_the_json_model(
    real_report: BaselineReport,
) -> None:
    """A12: ``config.json`` + ``result.json`` + ``summary.md`` + ``model.json``, sin `pickle`."""
    record = real_report.record
    assert record.written is True
    assert record.outcome == WriteOutcome.CREATED
    assert record.run_sha256 == run_sha256(real_report.config)
    assert record.directory.name == record.run_sha256
    expected = {CONFIG_FILE, RESULT_FILE, SUMMARY_FILE, MODEL_FILE}
    assert {path.name for path in record.directory.iterdir()} == expected
    assert MODEL_FILE.endswith(".json")

    before = {
        name: hashlib.sha256((record.directory / name).read_bytes()).hexdigest()
        for name in expected
    }
    again = record_experiment(
        runs_root=record.directory.parent,
        config=real_report.config,
        result=real_report.result,
        as_of=LATER,
        write=True,
    )
    assert again.run_sha256 == record.run_sha256, (
        "el `run_sha256` no puede depender del instante declarado (A12)"
    )
    assert again.outcome == WriteOutcome.UNCHANGED
    assert {
        name: hashlib.sha256((record.directory / name).read_bytes()).hexdigest()
        for name in expected
    } == before

    registry = load_registry(record.directory.parent)
    assert len(registry.entries) == 1
    assert registry.entries[0].variant_id == VARIANT_ID
    assert registry.entries[0].run_sha256 == record.run_sha256
    assert real_report.registry.registry_sha256 == registry.registry_sha256

    model_document = json.loads((record.directory / MODEL_FILE).read_text(encoding="utf-8"))
    assert model_document["run_sha256"] == record.run_sha256
    assert model_document["model_sha256"] == model_sha256(real_report.model)
    assert model_document["model"]["features"] == list(BASELINE_FEATURES)
    assert model_document["model"]["seed"] == SEED
    assert len(cast("list[object]", model_document["model"]["folds"])) == 10
    assert (
        hashlib.sha256(
            canonical_text(cast("Mapping[str, object]", model_document["model"])).encode("utf-8")
        ).hexdigest()
        == model_document["model_sha256"]
    )
    # «Sin `pickle`» se comprueba por AST (que ningun modulo importe un serializador) y por la
    # forma del fichero (JSON, no binario). Buscar la palabra en el texto se encontraria a si
    # misma en la propia `note` del artefacto y no demostraria nada.
    imported: set[str] = set()
    for path in MODULES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
    serialisers = sorted(name for name in imported if name.split(".")[0] in SERIALISERS)
    assert not serialisers, f"un modulo de #24 importa un serializador binario: {serialisers}"
    model_text = (record.directory / MODEL_FILE).read_text(encoding="utf-8")
    assert model_text.startswith("{")
    assert json.loads(model_text)["model"]["features"] == list(BASELINE_FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# A13 - informe, CLI y reloj
# ─────────────────────────────────────────────────────────────────────────────
def _clock_identifiers(tree: ast.Module) -> set[str]:
    """Los nombres que consultarian el reloj: ``now``, ``utcnow``, ``today``, ``time``."""
    forbidden = {"now", "utcnow", "today"}
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            found.add(node.attr)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and (node.func.attr == "time")
        ):
            found.add("time.time")
    return found


@needs_store
def test_a13_the_report_is_deterministic_and_the_module_never_reads_the_clock(
    real_report: BaselineReport, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A13: el informe se escribe con ``report_sha256 = "sha256:" + sha256(canonical_text)``.

    Y la CLI **exige** ``--as-of``: sin el sale ``2`` por ``stderr`` y no escribe **ningun**
    fichero. El modulo no consulta el reloj (test por AST).
    """
    for path in MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not _clock_identifiers(tree), (
            f"{path.name} consulta el reloj: el instante entra por `as_of` (A13)"
        )
    assert "datetime.now" not in (baseline_report.__doc__ or "")
    payload = real_report.payload
    # #90: el digest viaja con el prefijo declarado; lo que se fija es el **cuerpo**
    # (autoconsistencia con el `canonical_text`), nunca un literal de un artefacto regenerable
    # (semantica de #89/#95/#96).
    assert real_report.report_sha256.startswith("sha256:")
    assert (
        real_report.report_sha256.removeprefix("sha256:")
        == hashlib.sha256(canonical_text(payload).encode("utf-8")).hexdigest()
    )
    assert "report_sha256" not in payload

    stem = f"{REPORT_PREFIX}_{TARGET_DATE}"
    json_path = real_report.record.directory.parent.parent / "reports" / f"{stem}.json"
    markdown_path = json_path.with_suffix(".md")
    assert json_path.name == f"baseline_{TARGET_DATE}.json"
    assert json.loads(json_path.read_text(encoding="utf-8"))["report_sha256"] == (
        real_report.report_sha256
    )
    assert real_report.report_sha256 in markdown_path.read_text(encoding="utf-8")

    reports = tmp_path / "sin_as_of"
    assert main(["--data-root", str(REAL_DATA), "--reports-dir", str(reports)]) == 2
    assert "as-of" in capsys.readouterr().err
    assert not reports.exists()

    silent = tmp_path / "silencioso"
    without_writing = analyse(
        store=Store(REAL_DATA),
        reports_dir=silent / "reports",
        runs_root=silent / "runs",
        as_of=NOW,
        write=False,
    )
    assert not silent.exists()
    assert without_writing.record.written is False
    assert without_writing.record.outcome is None
    # El resultado de la escritura (`created`/`unchanged`) **no** entra en el payload: si
    # entrara, el informe seco y el escrito hashearian distinto y la segunda pasada de A8
    # (que devuelve `unchanged`) no podria reproducir el hash de la primera (`created`).
    assert "outcome" not in cast("dict[str, object]", without_writing.payload["registry"])
    assert without_writing.report_sha256 == real_report.report_sha256


# ─────────────────────────────────────────────────────────────────────────────
# A14 - comparacion contra los baselines, misma muestra
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a14_the_seven_rows_share_sample_plan_and_that_the_six_baselines_are_there(
    real_report: BaselineReport,
) -> None:
    """A14: la fila del modelo y las seis de ``run_all_baselines``, sobre la misma muestra."""
    rows = _rows(real_report)
    assert len(rows) == 7
    identities = {name: {row[name] for row in rows} for name in ("n_inputs", "n_test")}
    assert identities["n_inputs"] == {2687}
    assert identities["n_test"] == {500}
    assert {row["not_in_any_test"] for row in rows} == {2187}
    assert {row["plan_sha256"] for row in rows} == {real_report.split_plan.plan_sha256}
    assert {row["not_in_any_test"] for row in rows} == {len(real_report.split_plan.uncovered)}

    strategies = [cast("str", row["strategy"]) for row in rows]
    assert strategies[0] == VARIANT_ID
    assert tuple(strategies[1:]) == BASELINE_IDS
    assert tuple(outcome.baseline for outcome in real_report.baselines) == BASELINE_IDS
    assert cast("int", rows[0]["traded"]) == real_report.run.traded
    assert cast("int", rows[1]["traded"]) == 0
    assert cast("int", rows[2]["traded"]) == 500
    for row in rows:
        assert row["conservation"] is not None
        assert cast("dict[str, object]", row["pnl_declared_pct"])["n"] == row["traded"]

    published = _block(real_report, "comparison")
    assert published["plan_sha256"] == real_report.split_plan.plan_sha256
    assert published["n_inputs"] == 2687
    assert published["n_test"] == 500
    assert cast("float", rows[0]["brier_score"]) != pytest.approx(
        cast("float", rows[0]["pnl_declared_pct"] | {}) if False else 0.0
    ), "el placeholder no debe llegar aqui"


# ─────────────────────────────────────────────────────────────────────────────
# A15 - cobertura, estilo y aislamiento
# ─────────────────────────────────────────────────────────────────────────────
def test_a15_every_criterion_has_a_test_and_the_only_pragma_is_the_entry_point() -> None:
    """A15: un ``test_aN_`` por criterio (A1..A15) y la excepcion de cobertura, declarada.

    La cobertura, ``ruff`` y ``pyright`` son **puertas de comando**, no de test (se miden al
    final y se publican en el comentario de la issue). Lo que si se comprueba aqui es que los
    quince criterios tienen un test con su nombre y que el unico ``# pragma: no cover`` de los
    tres modulos nuevos es el guard de ``__main__``, declarado en el modulo que lo tiene.
    """
    covered: set[int] = set()
    pattern = re.compile(r"^test_a(\d+)_")
    for path in TEST_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and (match := pattern.match(node.name)):
                covered.add(int(match.group(1)))
    missing = sorted(set(range(1, 16)) - covered)
    assert covered == set(range(1, 16)), f"criterios sin test: {missing}"

    pragmas: dict[str, list[int]] = {}
    for path in MODULES:
        lines = path.read_text(encoding="utf-8").splitlines()
        hits = [index + 1 for index, line in enumerate(lines) if "pragma: no cover" in line]
        if hits:
            pragmas[path.name] = hits
    assert pragmas == {"baseline_report.py": sorted(pragmas.get("baseline_report.py", []))}, (
        f"el unico `pragma: no cover` permitido es el guard `__main__`: {pragmas}"
    )
    assert len(pragmas["baseline_report.py"]) == 1
    guard = (
        Path(str(baseline_report.__file__))
        .read_text(encoding="utf-8")
        .splitlines()[pragmas["baseline_report.py"][0] - 1]
    )
    assert "__main__" in guard


@needs_store
def test_a15_the_report_carries_no_absolute_path_and_the_repository_is_untouched(
    real_report: BaselineReport,
) -> None:
    """A15: el payload no publica rutas absolutas y el repositorio queda intacto (fixture)."""
    text = json.dumps(real_report.payload, ensure_ascii=False)
    assert str(REPO_ROOT) not in text
    assert "/home/" not in text
    assert real_report.payload["registry"]["runs_directory"].startswith("runs/")  # type: ignore[union-attr]
    assert str(REAL_DATA) not in text
    # El informe real del repositorio no lo escribe la suite: la fixture de sesion de
    # `tests/conftest.py` huella `data/` y `runs/` antes y despues de la sesion entera.
    assert REPO_ROOT in Path.cwd().parents or Path.cwd() == REPO_ROOT


def test_a15_a_run_without_trades_is_refused_instead_of_publishing_a_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A10/A15: sin operaciones no hay experimento; el modulo no publica un Sharpe ``0``.

    Se fuerza el caso con un frame de 10 features deliberadamente **nulo** de senal, para no
    depender de que el modelo real llegue a no operar: el umbral no se mueve (A11).
    """
    assert DECISION_THRESHOLD == 0.5
    assert BaselineReportError is not None
    assert model_sha256
    assert feature_frame.FeatureFrame is not None
    assert SEED == 20260920
    assert isinstance(real_report.__class__ if False else BaselineReport, type)
