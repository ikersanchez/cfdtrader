"""Tests de la calibracion dentro del informe del baseline (#25): A3, A4, A6-A14.

La mayoria de los criterios publican **numeros reales**, asi que se miden sobre el almacen del
repositorio en **solo lectura** (la fixture de sesion de ``tests/conftest.py`` huella el ``data/``
y el ``runs/`` antes y despues). Todo lo que se escribe va a ``tmp_path``, y la corrida cara se
hace **una sola vez** por sesion (``real_report``): son ~14 s.

Los tests sinteticos (A2/A3/A4/A5) viven en ``tests/test_calibration.py``, junto al modulo puro.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import textwrap
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final, cast

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray

from cfdtrader.analysis import baseline_report
from cfdtrader.analysis.baseline_report import (
    CALIBRATION_BINS,
    MODEL_FILE,
    REGISTERED_HYPERPARAMETERS,
    REPORT_PREFIX,
    BaselineReport,
    analyse,
    main,
    model_sha256,
)
from cfdtrader.analysis.experiment_log import run_sha256
from cfdtrader.backtest.engine import canonical_text
from cfdtrader.backtest.metrics import LOG_LOSS_EPSILON
from cfdtrader.data.store import Store
from cfdtrader.models import baseline, calibration
from cfdtrader.models.baseline import (
    BASELINE_FEATURES,
    DECISION_THRESHOLD,
    HYPERPARAMETERS,
    SEED,
    SplitAssignment,
    design_frame,
    fit_baseline,
    long_signal,
)
from cfdtrader.models.calibration import (
    CALIBRATION_HYPERPARAMETERS,
    METHOD_ISOTONIC,
    METHOD_NONE,
    METHOD_PLATT,
    PLATT_HYPERPARAMETERS,
    Calibration,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"
TESTS_DIR: Final[Path] = Path(__file__).resolve().parent

#: Instante **declarado** de todas las corridas (el modulo nunca lee el reloj).
NOW: Final[datetime] = datetime(2026, 9, 22, 22, 0, tzinfo=UTC)
TARGET_DATE: Final[str] = "2026-09-22"

#: Modulos nuevos de #25 (A1) y los dos ficheros de test del criterio A14.
MODULES: Final[tuple[Path, ...]] = (
    REPO_ROOT / "src" / "cfdtrader" / "models" / "calibration.py",
    REPO_ROOT / "src" / "cfdtrader" / "models" / "baseline.py",
    REPO_ROOT / "src" / "cfdtrader" / "analysis" / "baseline_report.py",
)
TEST_FILES: Final[tuple[Path, ...]] = (
    Path(__file__).resolve(),
    TESTS_DIR / "test_calibration.py",
)

#: Suites congeladas (A13): ni una linea de estas se toca en #25.
FROZEN_SUITES: Final[tuple[str, ...]] = (
    "test_metrics.py",
    "test_splits.py",
    "test_engine.py",
    "test_overfitting.py",
    "test_experiment_log.py",
    "test_feature_store.py",
    "test_technical_features.py",
    "test_context_features.py",
    "test_macro_features.py",
    "test_regime_features.py",
    "test_baseline_model.py",
    "test_feature_frame.py",
    "test_backtest_report.py",
    "test_phase1_report.py",
)

#: Artefactos vivos de #69 y #24 que #25 **no** puede reescribir (A11). Se huellan al
#: **importar** el modulo, antes de que corra la fixture de sesion, para que la comparacion
#: posterior demuestre que la corrida real no toco el repositorio.
REPORT_ARTIFACTS: Final[tuple[str, ...]] = (
    "phase1_backtest_2026-09-19.json",
    "phase1_backtest_2026-09-19.md",
    "baseline_2026-09-20.json",
    "baseline_2026-09-20.md",
)


def _digests() -> dict[str, str]:
    """sha256 de cada informe del repositorio, o ``missing`` si no esta en el arbol."""
    root = REAL_DATA / "derived" / "reports"
    out: dict[str, str] = {}
    for name in REPORT_ARTIFACTS:
        path = root / name
        out[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
    return out


def _runs_entries() -> list[str]:
    """Las entradas del registro del repositorio, para comprobar que la suite no escribe."""
    root = REPO_ROOT / "runs"
    return sorted(path.name for path in root.iterdir()) if root.exists() else []


REPORTS_AT_IMPORT: Final[dict[str, str]] = _digests()
RUNS_AT_IMPORT: Final[list[str]] = _runs_entries()

needs_store = pytest.mark.skipif(
    not (REAL_DATA / "derived" / "labels").exists(),
    reason="el almacen real no esta en el arbol: los numeros de A3/A4/A6-A11 son los suyos",
)


@pytest.fixture(scope="session")
def real_report(tmp_path_factory: pytest.TempPathFactory) -> BaselineReport:
    """La corrida real calibrada, **una vez** por sesion, escribiendo en un directorio temporal."""
    root = tmp_path_factory.mktemp("baseline_calibration")
    return analyse(
        store=Store(REAL_DATA),
        reports_dir=root / "reports",
        runs_root=root / "runs",
        as_of=NOW,
        write=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de lectura del payload y de `model.json`
# ─────────────────────────────────────────────────────────────────────────────
def _block(report: BaselineReport, *keys: str) -> dict[str, object]:
    """Un bloque anidado del payload, ya tipado."""
    node: dict[str, object] = report.payload
    for key in keys:
        node = cast("dict[str, object]", node[key])
    return node


def _model_document(report: BaselineReport) -> dict[str, object]:
    """El cuarto artefacto del registro, tal cual se publico."""
    path = report.record.directory / MODEL_FILE
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _fold_documents(report: BaselineReport) -> list[dict[str, object]]:
    """Los diez folds de `model.json`, en orden."""
    document = _model_document(report)
    model = cast("dict[str, object]", document["model"])
    return [cast("dict[str, object]", item) for item in cast("list[object]", model["folds"])]


def _design_matrix(report: BaselineReport) -> NDArray[np.float64]:
    """La matriz de diseno de las 10 features, en el orden del frame (posiciones de #24)."""
    selected = report.features.design.frame.select(list(BASELINE_FEATURES)).cast(pl.Float64)
    return cast("NDArray[np.float64]", selected.to_numpy())


def _labels(report: BaselineReport) -> NDArray[np.float64]:
    """Las etiquetas ``y`` del frame de diseno, en orden de posicion."""
    column = report.features.design.frame.get_column("y").cast(pl.Float64)
    return cast("NDArray[np.float64]", column.to_numpy())


def _fold_scores(fold: Mapping[str, object], matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """El score del fold en las posiciones indicadas, con **aritmetica propia** (A3/A6/A8).

    Es la misma operacion que publica `models.baseline.scores` (`(x - mean) / scale @ coef +
    intercept`), escrita aqui desde cero sobre los parametros de `model.json`: si el JSON no
    bastara para reconstruir las puntuaciones, este test lo veria.
    """
    positions = cast("list[int]", fold["test_positions"])
    mean = np.asarray(cast("list[float]", fold["mean"]), dtype=np.float64)
    scale = np.asarray(cast("list[float]", fold["scale"]), dtype=np.float64)
    coefficients = np.asarray(cast("list[float]", fold["coefficients"]), dtype=np.float64)
    return ((matrix[positions, :] - mean) / scale) @ coefficients + float(
        cast("float", fold["intercept"])
    )


def _score_at(
    fold: Mapping[str, object], matrix: NDArray[np.float64], positions: Sequence[int]
) -> NDArray[np.float64]:
    """El score del fold en **esas** posiciones (para el bloque de calibracion de A3)."""
    mean = np.asarray(cast("list[float]", fold["mean"]), dtype=np.float64)
    scale = np.asarray(cast("list[float]", fold["scale"]), dtype=np.float64)
    coefficients = np.asarray(cast("list[float]", fold["coefficients"]), dtype=np.float64)
    selection = list(positions)
    return ((matrix[selection, :] - mean) / scale) @ coefficients + float(
        cast("float", fold["intercept"])
    )


def _plain_sigmoid(score: float) -> float:
    """El enlace del modulo, escrito otra vez (misma formula ⇒ misma mantisa, sin importarlo)."""
    if score >= 0.0:
        return 1.0 / (1.0 + math.exp(-score))
    exponential = math.exp(score)
    return exponential / (1.0 + exponential)


def _published_test_sessions(report: BaselineReport) -> list[float | None]:
    """La probabilidad **calibrada** publicada por sesion de *test*, en orden (A7)."""
    return [
        session.decision.probability if session.decision is not None else None
        for fold in report.run.folds
        for session in fold.sessions
    ]


def _reconstruct(fold: Mapping[str, object], scores: NDArray[np.float64]) -> list[float]:
    """La probabilidad publicada que `model.json` + `numpy` reconstruyen (A8)."""
    block = cast("dict[str, object]", fold["calibration"])
    parameters = block["parameters"]
    if parameters is None:
        return [_plain_sigmoid(float(value)) for value in scores]
    mapping = cast("dict[str, object]", parameters)
    if "coef" in mapping:
        mean = float(cast("float", mapping["mean"]))
        scale = float(cast("float", mapping["scale"]))
        coef = float(cast("float", mapping["coef"]))
        intercept = float(cast("float", mapping["intercept"]))
        return [
            _plain_sigmoid((float(value) - mean) / scale * coef + intercept) for value in scores
        ]
    thresholds = np.asarray(cast("list[float]", mapping["thresholds"]), dtype=np.float64)
    values = np.asarray(cast("list[float]", mapping["values"]), dtype=np.float64)
    return [float(value) for value in np.interp(scores, thresholds, values)]


def _series(report: BaselineReport, *, calibrated: bool) -> list[float]:
    """Las 500 probabilidades (crudas o calibradas) en orden de sesion, reconstruidas (A6)."""
    matrix = _design_matrix(report)
    out: list[float] = []
    for fold in _fold_documents(report):
        scores = _fold_scores(fold, matrix)
        out.extend(
            _reconstruct(fold, scores)
            if calibrated
            else [_plain_sigmoid(float(value)) for value in scores]
        )
    return out


def _brier(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Brier con aritmetica propia, sin las funciones del modulo."""
    return sum(
        (probability - outcome) ** 2
        for probability, outcome in zip(probabilities, outcomes, strict=True)
    ) / len(outcomes)


def _clip(probability: float) -> float:
    """El recorte declarado de #15: ``[LOG_LOSS_EPSILON, 1 - LOG_LOSS_EPSILON]``."""
    return max(LOG_LOSS_EPSILON, min(1.0 - LOG_LOSS_EPSILON, probability))


def _log_loss(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Log-loss con aritmetica propia, recortando **igual** que #15 (`LOG_LOSS_EPSILON`).

    La isotonica satura y publica un 1,0 (y podria publicar un 0,0) exactos: `math.log(0)` no
    existe, asi que la cifra publicada esta recortada y el recorte se **declara** en el bloque
    (`log_loss_epsilon`). Cada termino se recorta **dentro** de su logaritmo (`log(clip(p))` y
    `log(clip(1 - p))`, que es lo que hace `metrics.log_loss`), no la probabilidad una sola vez:
    recortar `p` y calcular despues `log(1 - p)` da otra mantisa y la cifra no coincidiria.
    """
    return -sum(
        outcome * math.log(_clip(probability)) + (1 - outcome) * math.log(_clip(1.0 - probability))
        for probability, outcome in zip(probabilities, outcomes, strict=True)
    ) / len(outcomes)


# ─────────────────────────────────────────────────────────────────────────────
# A3 - el calibrador solo ve el bloque de calibracion
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a3_the_calibrator_is_refit_by_hand_from_the_calibration_block_only(
    real_report: BaselineReport,
) -> None:
    """A3: reajustar a mano sobre `calibration_positions` reproduce los parametros publicados.

    Y **no** los reproduce ni sobre las posiciones del `fit` ni sobre las del *test*: si el
    calibrador se hubiera ajustado con cualquiera de las otras dos, el reajuste coincidiria.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    matrix = _design_matrix(real_report)
    labels = _labels(real_report)
    folds = _fold_documents(real_report)
    plan = real_report.split_plan
    for index in (0, 9):
        fold = folds[index]
        block = cast("dict[str, object]", fold["calibration"])
        positions = cast("list[int]", block["calibration_positions"])
        parameters = cast("dict[str, object]", block["parameters"])
        assert len(positions) == block["n_calibration"]
        assert set(positions).isdisjoint(cast("list[int]", fold["test_positions"]))
        scores = _score_at(fold, matrix, positions)
        outcomes = labels[positions]

        if block["method"] == METHOD_PLATT:
            published = float(cast("float", parameters["coef"]))
            mean, scale = (
                float(cast("float", parameters["mean"])),
                float(cast("float", parameters["scale"])),
            )
            assert mean == pytest.approx(float(scores.mean()), abs=0.0)
            assert scale == pytest.approx(float(scores.std()), abs=0.0)
            refit = cast(
                "baseline._FittedLogistic",  # pyright: ignore[reportPrivateUsage]
                LogisticRegression(
                    solver=str(PLATT_HYPERPARAMETERS["solver"]),
                    max_iter=int(cast("int", PLATT_HYPERPARAMETERS["max_iter"])),
                    tol=float(cast("float", PLATT_HYPERPARAMETERS["tol"])),
                ),
            )
            refit.fit(((scores - mean) / scale).reshape(-1, 1), outcomes)
            assert float(np.asarray(refit.coef_).ravel()[0]) == published
            assert float(np.asarray(refit.intercept_).ravel()[0]) == float(
                cast("float", parameters["intercept"])
            )
        else:
            assert block["method"] == METHOD_ISOTONIC
            published_thresholds = np.asarray(
                cast("list[float]", parameters["thresholds"]), dtype=np.float64
            )
            estimator = cast(
                "calibration._FittedIsotonic",  # pyright: ignore[reportPrivateUsage]
                IsotonicRegression(increasing=True, out_of_bounds="clip"),  # type: ignore[arg-type]
            )
            estimator.fit(scores, outcomes)
            assert np.asarray(estimator.X_thresholds_).shape == published_thresholds.shape
            assert np.array_equal(np.asarray(estimator.X_thresholds_), published_thresholds)

        # Los otros dos bloques **no** reproducen lo publicado.
        fit_positions = sorted(set(plan.folds[index].train) - set(positions))
        other = _score_at(fold, matrix, fit_positions)
        assert len(other) > len(scores)
        if block["method"] == METHOD_PLATT:
            assert float(other.mean()) != pytest.approx(float(cast("float", parameters["mean"])))
        else:
            marks = [10, 50, 90]
            assert not np.array_equal(np.percentile(other, marks), np.percentile(scores, marks))
        test_scores = _fold_scores(fold, matrix)
        assert len(test_scores) == len(cast("list[int]", fold["test_positions"]))
        assert not np.array_equal(test_scores, scores[: len(test_scores)])


def test_a3_mutating_the_test_sessions_leaves_the_calibration_identical() -> None:
    """A3: mutar las features y la ``y`` del *test* de un fold deja el calibrador igual."""
    design = _synthetic_design(rows=400)
    splits = (SplitAssignment(index=0, train=tuple(range(0, 300)), test=tuple(range(300, 340))),)
    model = fit_baseline(design, splits=splits)
    fold = model.folds[0]
    assert fold.calibration.method != METHOD_NONE
    assert fold.calibration.calibration_positions == tuple(range(240, 300))

    mutated = design.frame.with_columns(
        [
            *[
                pl.when(pl.int_range(pl.len()) >= 300)
                .then(pl.col(name) * 3.0 + 1.0)
                .otherwise(pl.col(name))
                .alias(name)
                for name in BASELINE_FEATURES
            ],
            pl.when(pl.int_range(pl.len()) >= 300)
            .then(1 - pl.col("y"))
            .otherwise(pl.col("y"))
            .alias("y"),
        ]
    )
    assert not mutated.select(list(BASELINE_FEATURES)).equals(
        design.frame.select(list(BASELINE_FEATURES))
    )
    refit = fit_baseline(_replaced(design, mutated), splits=splits)
    assert [item.calibration for item in refit.folds] == [item.calibration for item in model.folds]
    assert [item.coefficients for item in refit.folds] == [
        item.coefficients for item in model.folds
    ]


# ─────────────────────────────────────────────────────────────────────────────
# A4 - el metodo por fold, medido sobre el plan real
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a4_the_real_folds_publish_platt_and_isotonic_with_their_counts(
    real_report: BaselineReport,
) -> None:
    """A4: 437 en el fold 0 (Platt) y 527 en el 9 (isotonica), con el histograma medido."""
    folds = _fold_documents(real_report)
    blocks = [cast("dict[str, object]", fold["calibration"]) for fold in folds]
    assert [block["n_calibration"] for block in blocks] == [
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
    assert [cast("dict[str, object]", fold["calibration"])["method"] for fold in folds] == [
        METHOD_PLATT,
        METHOD_PLATT,
        METHOD_PLATT,
        METHOD_PLATT,
        METHOD_PLATT,
        METHOD_PLATT,
        METHOD_PLATT,
        METHOD_ISOTONIC,
        METHOD_ISOTONIC,
        METHOD_ISOTONIC,
    ]
    published = _block(real_report, "probability_metrics", "calibration")
    assert published["methods"] == {"platt": 7, "isotonic": 3, "none": 0}
    assert published["n_folds_calibrated"] == published["n_folds"] == 10
    assert published["method"] == "mixed"
    assert published["calibrated"] is True
    assert published["fully_calibrated"] is True
    assert published["bins"] == CALIBRATION_BINS == 5

    first = cast("dict[str, object]", blocks[0]["parameters"])
    assert cast("float", first["coef"]) > 0.0
    last = cast("dict[str, object]", blocks[9]["parameters"])
    assert last["increasing"] is True
    assert list(cast("list[float]", last["values"])) == sorted(cast("list[float]", last["values"]))
    for block in blocks:
        assert block["reason"] is None
        assert block["purge_sessions"] == 0
        assert block["exclusions_are_no_op"] is True
        assert int(cast("int", block["n_positives"])) > 0


# ─────────────────────────────────────────────────────────────────────────────
# A6 - la comparacion cruda frente a calibrada
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a6_the_two_sides_are_the_same_500_sessions_and_the_delta_is_signed(
    real_report: BaselineReport,
) -> None:
    """A6: Brier, log-loss, operadas y curva de las dos veredas, y ``delta = before - after``."""
    published = _block(real_report, "probability_metrics")
    before = cast("dict[str, object]", published["before"])
    after = cast("dict[str, object]", published["after"])
    delta = cast("dict[str, object]", published["delta"])
    labels = [int(value) for value in _labels(real_report)]
    positions = [
        position
        for fold in _fold_documents(real_report)
        for position in cast("list[int]", fold["test_positions"])
    ]
    assert positions == sorted(positions)
    outcomes = [labels[position] for position in positions]
    raw = _series(real_report, calibrated=False)
    calibrated = [value for value in _published_test_sessions(real_report) if value is not None]

    for side, series in ((before, raw), (after, calibrated)):
        assert side["n_test"] == len(series) == 500
        assert side["n_positives"] == sum(outcomes)
        assert side["brier_score"] == pytest.approx(_brier(series, outcomes), rel=0, abs=0.0)
        assert side["log_loss"] == pytest.approx(_log_loss(series, outcomes), rel=0, abs=0.0)
        curve = cast("list[dict[str, object]]", side["curve"])
        assert side["bins"] == CALIBRATION_BINS == 5
        assert len(curve) == 5
        assert sum(cast("int", row["count"]) for row in curve) == 500

    assert delta["brier_score"] == (
        float(cast("float", before["brier_score"])) - float(cast("float", after["brier_score"]))
    )
    assert delta["log_loss"] == (
        float(cast("float", before["log_loss"])) - float(cast("float", after["log_loss"]))
    )
    assert delta["n_traded"] == int(cast("int", before["n_traded"])) - int(
        cast("int", after["n_traded"])
    )
    assert delta["improves"] == (
        float(cast("float", delta["brier_score"])) > 0.0
        and float(cast("float", delta["log_loss"])) > 0.0
    )
    assert _block(real_report, "probability_metrics", "calibration")["curve"] == after["curve"]
    assert published["brier_score"] == after["brier_score"]
    assert published["log_loss"] == after["log_loss"]

    # La serie cruda es la que ya publico #24: la calibracion **no** la toca.
    assert float(cast("float", before["brier_score"])) == pytest.approx(0.2511418117147099)
    assert float(cast("float", before["log_loss"])) == pytest.approx(0.6953995223105118)


# ─────────────────────────────────────────────────────────────────────────────
# A7 - la decision usa la calibrada
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a7_the_decision_is_taken_with_the_calibrated_probability(
    real_report: BaselineReport,
) -> None:
    """A7: `run.traded` es el recuento calibrado, el crudo se publica y el umbral no se mueve."""
    published = _block(real_report, "probability_metrics")
    before = cast("dict[str, object]", published["before"])
    after = cast("dict[str, object]", published["after"])
    decision = _block(real_report, "decision")
    calibrated = [value for value in _published_test_sessions(real_report) if value is not None]
    raw = _series(real_report, calibrated=False)

    assert DECISION_THRESHOLD == 0.5 == decision["threshold"]
    assert len(calibrated) == 500
    assert sum(1 for value in calibrated if value >= DECISION_THRESHOLD) == after["n_traded"]
    assert sum(1 for value in raw if value >= DECISION_THRESHOLD) == before["n_traded"]
    assert real_report.run.traded == after["n_traded"] == decision["n_traded"]
    assert decision["n_traded_raw"] == before["n_traded"]
    assert int(cast("int", decision["n_no_trade"])) + int(cast("int", decision["n_traded"])) == 500
    assert before["n_traded"] != after["n_traded"], (
        "con este corpus el punto de corte se mueve: si algun dia coinciden, el test sigue "
        "valiendo pero el comentario de la issue tiene que decir el mismo numero"
    )

    # Los empates de la isotonica se resuelven con `>=`, sin desempatar a mano.
    tie = Calibration(
        method=METHOD_ISOTONIC,
        reason=None,
        n_fit=10,
        n_calibration=30,
        n_positives=15,
        calibration_positions=tuple(range(30)),
        purge_sessions=0,
        exclusions_are_no_op=True,
        thresholds=(0.0, 1.0),
        values=(0.25, 0.5),
    )
    assert tie.calibrate([1.0]) == (0.5,)
    assert long_signal(0.5) is True
    assert long_signal(0.4999999) is False


# ─────────────────────────────────────────────────────────────────────────────
# A8 - `model.json` reproduce las probabilidades publicadas
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a8_the_json_reproduces_the_published_probabilities_bit_for_bit(
    real_report: BaselineReport,
) -> None:
    """A8: con `model.json` y `numpy` (sin `sklearn`) las 500 calibradas coinciden **exactas**."""
    published = [value for value in _published_test_sessions(real_report) if value is not None]
    reconstructed = _series(real_report, calibrated=True)
    assert len(published) == len(reconstructed) == 500
    exact = sum(1 for left, right in zip(published, reconstructed, strict=True) if left == right)
    assert exact == 500
    assert published == reconstructed

    document = _model_document(real_report)
    model = cast("dict[str, object]", document["model"])
    folded = cast("list[object]", model["folds"])
    assert len(folded) == 10
    for fold in folded:
        block = cast("dict[str, object]", cast("dict[str, object]", fold)["calibration"])
        assert set(block) >= {
            "method",
            "reason",
            "n_calibration",
            "n_positives",
            "parameters",
            "purge_sessions",
            "exclusions_are_no_op",
            "calibration_positions",
        }
    assert document["model_sha256"] == model_sha256(real_report.model)
    assert (
        document["model_sha256"]
        == hashlib.sha256(canonical_text(model).encode("utf-8")).hexdigest()
    )
    assert document["run_sha256"] == real_report.record.run_sha256


# ─────────────────────────────────────────────────────────────────────────────
# A9 - identidad y determinismo
# ─────────────────────────────────────────────────────────────────────────────
CHILD: Final[str] = textwrap.dedent(
    """
    import hashlib, json, sys
    from datetime import datetime
    from pathlib import Path
    from cfdtrader.analysis.baseline_report import analyse, model_sha256
    from cfdtrader.data.store import Store

    report = analyse(
        store=Store(sys.argv[1]),
        reports_dir=Path(sys.argv[2]),
        runs_root=Path(sys.argv[3]),
        as_of=datetime.fromisoformat(sys.argv[4]),
        write=True,
    )
    stem = f"baseline_{report.report_date.isoformat()}"
    print(json.dumps({
        "report_sha256": report.report_sha256,
        "run_sha256": report.record.run_sha256,
        "model_sha256": model_sha256(report.model),
        "registry_sha256": report.registry.registry_sha256,
        "outcome": None if report.record.outcome is None else str(report.record.outcome),
        "json": hashlib.sha256((Path(sys.argv[2]) / f"{stem}.json").read_bytes()).hexdigest(),
        "md": hashlib.sha256((Path(sys.argv[2]) / f"{stem}.md").read_bytes()).hexdigest(),
    }))
    """
)


def _child(hash_seed: str, *, reports_dir: Path, runs_root: Path) -> dict[str, object]:
    """Corre la corrida real en un proceso aparte con esa semilla de ``hash``."""
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            CHILD,
            str(REAL_DATA),
            str(reports_dir),
            str(runs_root),
            NOW.isoformat(),
        ],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": hash_seed, "HOME": str(Path.home())},
    )
    return cast("dict[str, object]", json.loads(completed.stdout.strip().splitlines()[-1]))


@needs_store
def test_a9_the_report_is_deterministic_in_three_processes_and_a_second_pass(
    real_report: BaselineReport, tmp_path: Path
) -> None:
    """A9: `report_sha256`, `.json` y `.md` identicos en `PYTHONHASHSEED` 0/1/random.

    Las tres corridas comparten el `runs-root` (la primera lo crea, las otras dos encuentran el
    experimento `unchanged`) y escriben en `--reports-dir` **distintos**; los hashes del informe
    y de los dos ficheros tienen que coincidir entre si y con la corrida en proceso de la
    fixture, que usa otra ruta.
    """
    runs_root = tmp_path / "runs"
    results = [
        _child(seed, reports_dir=tmp_path / f"reports_{seed}", runs_root=runs_root)
        for seed in ("0", "1", "random")
    ]
    hashes = [{key: value for key, value in item.items() if key != "outcome"} for item in results]
    assert hashes[0] == hashes[1] == hashes[2]
    assert [item["outcome"] for item in results] == ["created", "unchanged", "unchanged"]
    assert results[0]["report_sha256"] == real_report.report_sha256
    assert results[0]["run_sha256"] == real_report.record.run_sha256
    assert results[0]["model_sha256"] == model_sha256(real_report.model)
    published = real_report.record.directory.parent.parent / "reports"
    assert (
        results[0]["json"]
        == hashlib.sha256(
            (published / f"{REPORT_PREFIX}_{TARGET_DATE}.json").read_bytes()
        ).hexdigest()
    )

    # La identidad del experimento no depende del instante ni de la ruta (A9).
    assert real_report.record.run_sha256 == run_sha256(real_report.config)
    assert model_sha256(real_report.model) == _model_document(real_report)["model_sha256"]
    assert str(REPO_ROOT) not in canonical_text(real_report.config.to_payload())
    assert real_report.config.hyperparameters == REGISTERED_HYPERPARAMETERS

    # Las tres constantes de calibracion entran en la configuracion registrada: cambiarlas
    # cambia el `run_sha256`.
    mutated = dataclasses.replace(
        real_report.config,
        hyperparameters={
            **real_report.config.hyperparameters,
            "platt_max_calibration_sessions": 400,
        },
    )
    assert run_sha256(mutated) != run_sha256(real_report.config)


# ─────────────────────────────────────────────────────────────────────────────
# A10 - el caso que no mejora es publicable
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_a_calibration_that_cannot_help_is_published_with_its_sign() -> None:
    """A10: sin nada que ganar, el informe publica ``delta <= 0`` con su signo y no falla."""
    outcomes = [1] * 250 + [0] * 250
    already = [0.75] * 250 + [0.25] * 250
    flat = baseline_report._probability_block(  # pyright: ignore[reportPrivateUsage]
        raw=already, calibrated=list(already), outcomes=outcomes, references=already, folds=()
    )
    delta = cast("dict[str, object]", flat["delta"])
    assert delta["brier_score"] == 0.0
    assert delta["log_loss"] == 0.0
    assert delta["n_traded"] == 0
    assert delta["improves"] is False
    calibration_block = cast("dict[str, object]", flat["calibration"])
    assert calibration_block["method"] == METHOD_NONE
    assert calibration_block["calibrated"] is False
    assert calibration_block["n_folds_calibrated"] == 0

    worse = baseline_report._probability_block(  # pyright: ignore[reportPrivateUsage]
        raw=already, calibrated=[0.5] * 500, outcomes=outcomes, references=already, folds=()
    )
    worse_delta = cast("dict[str, object]", worse["delta"])
    assert float(cast("float", worse_delta["brier_score"])) < 0.0
    assert float(cast("float", worse_delta["log_loss"])) < 0.0
    assert worse_delta["improves"] is False


@needs_store
def test_a10_the_report_runs_to_the_end_and_publishes_the_negative_delta(tmp_path: Path) -> None:
    """A10: la CLI sale ``0`` y el informe real publica un ``delta`` negativo **con su signo**."""
    reports = tmp_path / "reports"
    code = main(
        [
            "--data-root",
            str(REAL_DATA),
            "--reports-dir",
            str(reports),
            "--runs-root",
            str(tmp_path / "runs"),
            "--as-of",
            NOW.isoformat(),
        ]
    )
    assert code == 0
    document = cast(
        "dict[str, object]",
        json.loads((reports / f"{REPORT_PREFIX}_{TARGET_DATE}.json").read_text(encoding="utf-8")),
    )
    metrics = cast("dict[str, object]", document["probability_metrics"])
    delta = cast("dict[str, object]", metrics["delta"])
    before = cast("dict[str, object]", metrics["before"])
    after = cast("dict[str, object]", metrics["after"])
    assert delta["brier_score"] == (
        float(cast("float", before["brier_score"])) - float(cast("float", after["brier_score"]))
    )
    assert delta["improves"] is False
    assert float(cast("float", delta["brier_score"])) < 0.0


# ─────────────────────────────────────────────────────────────────────────────
# A11 - el veredicto heredado, intacto
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a11_the_inherited_verdict_is_intact_and_no_artifact_is_rewritten(
    real_report: BaselineReport,
) -> None:
    """A11: `gate`/`phase1_ready`/`is_validation` como los dejo #24, y los informes vivos igual."""
    assert real_report.payload["gate"] == "fail"
    assert real_report.payload["phase1_ready"] is False
    assert real_report.payload["is_validation"] is False
    limits = _block(real_report, "limits")
    assert limits["gate"] == "fail"
    assert limits["phase1_ready"] is False
    assert limits["is_validation"] is False
    assert limits["threshold"] == DECISION_THRESHOLD
    cost = _block(real_report, "declared_cost")
    assert cost["basis"] == "declared_cost"
    assert cost["is_validation"] is False
    net = cast("dict[str, object]", cost["net_metrics"])
    assert net["state"] == "not_computable"
    assert "#62" in cast("list[str]", net["follow_up"])

    # Ni #69 ni #24 se reescriben: la corrida real de la fixture escribe en `tmp_path` y la
    # huella de `data/derived/reports/` (tomada al **importar** este modulo) no cambia.
    assert _digests() == REPORTS_AT_IMPORT
    assert _runs_entries() == RUNS_AT_IMPORT


# ─────────────────────────────────────────────────────────────────────────────
# A12 - las fronteras declaradas
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a12_no_declared_frontier_points_at_25_any_more(real_report: BaselineReport) -> None:
    """A12: las tres listas declaran que **si** se calibra y nombran #26/#27/#60/#28/#68."""
    for entry in baseline_report.REPORT_DOES_NOT_DO:
        assert entry["issue"] != "#25", f"una frontera sigue apuntando a #25: {entry}"
    for entry in baseline_report.FOLLOW_UPS:
        assert entry["issue"] != "#25", f"un seguimiento sigue apuntando a #25: {entry}"
    for entry in baseline.MODEL_DOES_NOT_DO:
        assert entry["issue"] != "#25", f"el modelo sigue declarando #25: {entry}"
    assert any(entry["id"] == "calibra_en_el_train" for entry in baseline_report.REPORT_DOES_NOT_DO)
    assert any(entry["id"] == "calibra_en_el_train" for entry in baseline.MODEL_DOES_NOT_DO)
    assert any(entry["id"] == "no_toca_el_holdout" for entry in baseline_report.REPORT_DOES_NOT_DO)

    frontiers = {entry["issue"] for entry in baseline_report.REPORT_DOES_NOT_DO} | {
        entry["issue"] for entry in baseline_report.FOLLOW_UPS
    }
    assert {"#26", "#27", "#28", "#60", "#68"} <= frontiers
    assert "calibra" in " ".join(entry["statement"] for entry in baseline_report.REPORT_DOES_NOT_DO)
    assert not any("no estan calibradas" in item for item in baseline_report.LIMITATIONS)
    assert any("si** estan calibradas" in item for item in baseline_report.LIMITATIONS)

    payload = real_report.payload
    assert {entry["issue"] for entry in cast("list[dict[str, str]]", payload["does_not_do"])} == {
        entry["issue"] for entry in baseline_report.REPORT_DOES_NOT_DO
    }
    assert not any(
        entry["issue"] == "#25" for entry in cast("list[dict[str, str]]", payload["follow_ups"])
    )
    markdown_path = real_report.record.directory.parent.parent / "reports"
    markdown = (markdown_path / f"{REPORT_PREFIX}_{TARGET_DATE}.md").read_text(encoding="utf-8")
    assert "#25" not in markdown
    assert "cruda frente a calibrada" in markdown


# ─────────────────────────────────────────────────────────────────────────────
# A13 - las suites congeladas
# ─────────────────────────────────────────────────────────────────────────────
def _git(*args: str) -> str:
    """Salida de `git` en el repositorio, sin paginacion."""
    completed = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return completed.stdout


def test_a13_the_frozen_suites_are_untouched_and_the_suite_writes_nothing() -> None:
    """A13: las suites congeladas no conocen el modulo nuevo y el arbol esta limpio.

    La parte «el `runs/` del repositorio esta vacio» **no** es lo que comprueba la guardia de
    `tests/conftest.py`: alli se huella `runs/` antes y despues de la sesion y se exige que la
    suite **no escriba** (el repositorio conserva el experimento de #24). Aqui se comprueba la
    sustancia: ninguna suite congelada importa `models.calibration` (luego no se han adaptado a
    #25) y el arbol git esta limpio, que es la puerta de las guardias de `#15` y `#16`.
    """
    for name in FROZEN_SUITES:
        path = TESTS_DIR / name
        assert path.is_file(), f"falta la suite congelada {name}"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert "cfdtrader.models.calibration" not in imported, (
            f"{name} importa el modulo de calibracion: la suite congelada no se toca (A13)"
        )
        mentioned = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        assert not {"calibrated_probabilities", "REGISTERED_HYPERPARAMETERS"} & mentioned

    conftest = (TESTS_DIR / "conftest.py").read_text(encoding="utf-8")
    assert "_repository_data_is_untouched" in conftest
    assert "_repository_runs_is_untouched" in conftest
    assert _runs_entries() == RUNS_AT_IMPORT
    assert _git("status", "--porcelain") == "", (
        "el arbol tiene cambios sin commitear: las guardias de #15/#16 miden sobre el arbol limpio"
    )
    assert _git("diff", "--name-only", "--", *FROZEN_SUITES) == ""


# ─────────────────────────────────────────────────────────────────────────────
# A14 - cobertura, estilo y un test por criterio
# ─────────────────────────────────────────────────────────────────────────────
def test_a14_every_criterion_has_a_test_and_the_modules_have_no_new_pragma() -> None:
    """A14: un ``test_aN_`` por criterio (A1..A14) y sin `# pragma: no cover` anadido.

    Un criterio puede llevar mas de un test: ademas del que lo verifica hay tests de apoyo con
    el mismo prefijo (los controles negativos), asi que lo que se exige es que **ningun**
    criterio se quede sin test y que no haya un ``test_aN_`` de un criterio inexistente.

    La **cobertura**, `ruff`, `ruff format --check` y `pyright` son puertas de comando: se miden
    al final y se publican en el comentario de la issue. Lo que si se comprueba aqui es que los
    catorce criterios tienen test con su nombre, que los modulos de #25 no traen ningun `pragma`
    nuevo (el unico permitido es el guard ``__main__`` de `baseline_report.py`, por herencia de
    #24) y que los tests nuevos escriben en `tmp_path`.
    """
    covered: set[int] = set()
    names: set[str] = set()
    for path in TEST_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_a"):
                names.add(node.name)
                covered.add(int(node.name.split("_")[1][1:]))
    missing = sorted(set(range(1, 15)) - covered)
    assert covered == set(range(1, 15)), f"criterios sin test: {missing}"
    assert max(covered) == 14, (
        f"hay un `test_aN_` de un criterio que no existe (A1..A14): {sorted(names)}"
    )

    pragmas = {
        path.name: [
            index + 1
            for index, line in enumerate(path.read_text(encoding="utf-8").splitlines())
            if "pragma: no cover" in line
        ]
        for path in MODULES
    }
    assert pragmas["calibration.py"] == []
    assert pragmas["baseline.py"] == []
    assert len(pragmas["baseline_report.py"]) == 1
    guard = (
        Path(str(baseline_report.__file__))
        .read_text(encoding="utf-8")
        .splitlines()[pragmas["baseline_report.py"][0] - 1]
    )
    assert "__main__" in guard

    # El unico fichero de los dos que **escribe** en disco es este: corre el informe real en la
    # CLI y en la fixture, y todo lo que escribe va a `tmp_path`. `test_calibration.py` mide el
    # modulo puro con posiciones y puntuaciones sinteticas: no toca el disco, asi que pedirle
    # `tmp_path` seria exigirle un parametro que no tiene nada que hacer con el (tension
    # declarada de A14).
    tests_of_the_report = ast.parse(TEST_FILES[0].read_text(encoding="utf-8"))
    assert any(
        "tmp_path"
        in [argument.arg for argument in node.args.args]
        + [argument.arg for argument in node.args.kwonlyargs]
        for node in ast.walk(tests_of_the_report)
        if isinstance(node, ast.FunctionDef)
    ), f"{TEST_FILES[0].name} no escribe nada en `tmp_path` (A14)"
    assert "tmp_path" in (TESTS_DIR / "test_baseline_report.py").read_text(encoding="utf-8")
    assert CALIBRATION_HYPERPARAMETERS["min_calibration_sessions"] == 30
    assert CALIBRATION_HYPERPARAMETERS["calibration_fraction"] == 0.2
    assert HYPERPARAMETERS["solver"] == "saga"
    assert SEED == 20260920


# ─────────────────────────────────────────────────────────────────────────────
# Apoyo: diseno sintetico (A3) y utilidades de la corrida real
# ─────────────────────────────────────────────────────────────────────────────
def _noise(seed: str, index: int) -> float:
    """Ruido determinista en ``[-1, 1]`` a partir de ``sha256``."""
    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / (2**63) - 1.0


def _synthetic_design(rows: int = 400) -> baseline.DesignFrame:
    """Un frame de diseno sintetico con las 10 features y una etiqueta de las dos clases."""
    days: list[date] = []
    current = date(2024, 1, 2)
    while len(days) < rows:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    data: dict[str, list[object]] = {"session": list(days)}
    for position, name in enumerate(BASELINE_FEATURES):
        data[name] = [_noise(f"a25:{name}", index) + position / 100.0 for index in range(rows)]
    labels = pl.DataFrame(
        {"session": days[1:], "ret_long": [0.01 * (index % 3 - 1) for index in range(1, rows)]}
    )
    return design_frame(pl.DataFrame(data), labels=labels)


def _replaced(frame: baseline.DesignFrame, mutated: pl.DataFrame) -> baseline.DesignFrame:
    """El mismo `DesignFrame` con el frame mutado (los contadores no cambian)."""
    return baseline.DesignFrame(
        frame=mutated,
        n_sessions=frame.n_sessions,
        n_labels=frame.n_labels,
        n_shifted_rows=frame.n_shifted_rows,
        n_nulls_in_features=frame.n_nulls_in_features,
        design_lag_sessions=frame.design_lag_sessions,
    )
