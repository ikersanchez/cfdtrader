"""Tests del modelo LightGBM reducido (#26): A4 y A5.

Todo va sobre un frame de diseno **sintetico** de 2.400 sesiones: los criterios que miden
numeros reales viven en `tests/test_model_comparison.py` (que si lee el almacen). Aqui se
comprueba lo que es del modelo: los hiperparametros **fijos**, que hay **una sola** llamada al
estimador y sin barrido, y el determinismo **medido** (doble ajuste + recarga del texto).
"""

from __future__ import annotations

import ast
import itertools
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import polars as pl
import pytest

from cfdtrader.analysis import model_comparison
from cfdtrader.backtest.engine import canonical_text
from cfdtrader.models import lightgbm_model
from cfdtrader.models.baseline import (
    BASELINE_FEATURES,
    DESIGN_LAG_SESSIONS,
    DesignFrame,
    SplitAssignment,
)
from cfdtrader.models.lightgbm_model import (
    LIGHTGBM_HYPERPARAMETERS,
    RELOAD_TOLERANCE,
    InvalidLightGBMInputError,
    LightGBMDeterminismError,
    LightGBMModel,
    calibrated_probabilities,
    fit_lightgbm,
    probabilities,
    reload_probabilities,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODULE: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "models" / "lightgbm_model.py"

#: El diccionario del punto 2 del brief, escrito **otra vez** para compararlo (A4).
EXPECTED_HYPERPARAMETERS: Final[dict[str, object]] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 4,
    "max_depth": 2,
    "min_child_samples": 200,
    "subsample": 1.0,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "reg_lambda": 0.0,
    "random_state": 20260920,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
    "verbose": -1,
}

ROWS: Final[int] = 2_400
TRAIN_ROWS: Final[int] = 2_000


def _synthetic_frame(*, rows: int = ROWS, seed: int = 11) -> pl.DataFrame:
    """Features sinteticas deterministas y una etiqueta que depende de dos de ellas."""
    rng = np.random.RandomState(seed)
    features = {name: rng.normal(size=rows) for name in BASELINE_FEATURES}
    signal = features["har_forecast"] + 0.4 * features["vix_zscore"]
    outcomes = (signal + rng.normal(size=rows) > 0).astype(np.int64)
    sessions = [date(2020, 1, 1) + timedelta(days=index) for index in range(rows)]
    return pl.DataFrame({"session": sessions, **features, "y": outcomes})


def _design(frame: pl.DataFrame) -> DesignFrame:
    """El `DesignFrame` con los recuentos del frame, sin corrimiento (lo arma el sintetico)."""
    return DesignFrame(
        frame=frame,
        n_sessions=frame.height,
        n_labels=frame.height,
        n_shifted_rows=0,
        n_nulls_in_features=0,
        design_lag_sessions=DESIGN_LAG_SESSIONS,
    )


def _splits(*, folds: int = 2) -> tuple[SplitAssignment, ...]:
    """Folds contiguos de 200 sesiones sobre el sintetico, con train expansivo."""
    out: list[SplitAssignment] = []
    for index in range(folds):
        stop = TRAIN_ROWS + index * 200
        out.append(
            SplitAssignment(
                index=index,
                train=tuple(range(0, stop)),
                test=tuple(range(stop, stop + 200)),
            )
        )
    return tuple(out)


@pytest.fixture(scope="module")
def synthetic() -> tuple[DesignFrame, LightGBMModel]:
    """Un ajuste sintetico de 2 folds, una vez por modulo (es rapido pero no gratis)."""
    design = _design(_synthetic_frame())
    return design, fit_lightgbm(design, splits=_splits())


def _source(name: str) -> str:
    """El fuente del modulo nuevo, para las comprobaciones por AST."""
    module = lightgbm_model if name == "lightgbm_model" else model_comparison
    return Path(cast("Any", module).__file__).read_text(encoding="utf-8")


def _classifier_calls(tree: ast.Module) -> list[ast.Call]:
    """Las llamadas a `LGBMClassifier(...)` del modulo, por AST (A4)."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LGBMClassifier"
    ]


# ─────────────────────────────────────────────────────────────────────────────
# A4 - hiperparametros fijos, sin busqueda
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_the_hyperparameters_are_the_declared_literal() -> None:
    """El diccionario publicado es el del brief, incluidos `min_child_samples` y los hilos."""
    assert dict(LIGHTGBM_HYPERPARAMETERS) == EXPECTED_HYPERPARAMETERS
    assert LIGHTGBM_HYPERPARAMETERS["min_child_samples"] == 200
    assert LIGHTGBM_HYPERPARAMETERS["num_threads"] == 1
    assert LIGHTGBM_HYPERPARAMETERS["deterministic"] is True
    assert LIGHTGBM_HYPERPARAMETERS["force_row_wise"] is True


def test_a4_there_is_one_call_site_without_literals_and_without_sweep() -> None:
    """**Una** llamada al estimador, con el diccionario entero y sin bucle de barrido (A4)."""
    tree = ast.parse(_source("lightgbm_model"))
    calls = _classifier_calls(tree)
    assert len(calls) == 1, "tiene que haber una sola llamada a LGBMClassifier (A4)"
    call = calls[0]
    assert [keyword.arg for keyword in call.keywords] == [None], (
        "la unica llamada tiene que desempaquetar LIGHTGBM_HYPERPARAMETERS, no pasar "
        "hiperparametros sueltos (A4)"
    )
    unpacked = call.keywords[0].value
    assert isinstance(unpacked, ast.Name), (
        "la unica llamada tiene que desempaquetar el diccionario declarado, no literales (A4)"
    )
    numeric = [
        node
        for node in ast.walk(call)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
    ]
    assert not numeric, f"la llamada no lleva literales de hiperparametros: {numeric} (A4)"


def test_a4_there_is_no_sweep_over_candidate_parameters() -> None:
    """Sin `itertools`, sin `product`, sin `grid` y sin bucles sobre candidatos (A4)."""
    tree = ast.parse(_source("lightgbm_model"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "itertools" not in imported
    forbidden = ("sweep", "grid", "candidate", "search")
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and any(word in node.id.lower() for word in forbidden)
    } | {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and any(word in node.attr.lower() for word in forbidden)
    }
    assert not names, f"ningun barrido de hiperparametros en el modulo: {sorted(names)} (A4)"


# ─────────────────────────────────────────────────────────────────────────────
# A5 - determinismo medido y `model.json` reproducible
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_the_fit_is_reproducible_and_the_text_is_compared(
    synthetic: tuple[DesignFrame, LightGBMModel],
) -> None:
    """Los dos ajustes por fold coinciden: el modelo publicado es determinista (A5)."""
    _, model = synthetic
    assert len(model.folds) == 2
    for fold in model.folds:
        assert fold.booster_model.startswith("tree\nversion=") or "tree" in fold.booster_model[:40]
        assert fold.n_trees == 200
        assert len(fold.test_positions) == fold.n_test == 200
        assert len(fold.test_probabilities) == 200
        assert len(fold.test_scores) == 200
        assert all(0.0 <= value <= 1.0 for value in fold.test_probabilities)


def test_a5_a_divergent_fit_is_a_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un ajuste forzado a divergir da `LightGBMDeterminismError` y no publica nada (A5)."""
    design = _design(_synthetic_frame(rows=600, seed=5))

    class _FakeBooster:
        def __init__(self, text: str) -> None:
            self._text = text

        def model_to_string(self) -> str:
            return self._text

    class _FakeEstimator:
        """Dos ajustes distintos del mismo fold: el texto cambia a proposito."""

        texts = itertools.cycle(["primer-ajuste", "segundo-ajuste"])

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def fit(self, features: object, outcomes: object) -> _FakeEstimator:
            self.booster_ = _FakeBooster(next(self.texts))
            return self

    monkeypatch.setattr(lightgbm_model, "LGBMClassifier", _FakeEstimator)
    splits = (SplitAssignment(index=0, train=tuple(range(0, 400)), test=tuple(range(400, 600))),)
    with pytest.raises(LightGBMDeterminismError) as error:
        fit_lightgbm(design, splits=splits)
    assert "fold 0" in str(error.value)


def test_a5_the_reload_reproduces_the_published_probabilities(
    synthetic: tuple[DesignFrame, LightGBMModel],
) -> None:
    """Recargar el `booster_model` reproduce las probabilidades publicadas (A5)."""
    design, model = synthetic
    for fold in model.folds:
        reloaded = reload_probabilities(fold, design.frame)
        worst = max(
            (
                abs(one - other)
                for one, other in zip(reloaded, fold.test_probabilities, strict=True)
            ),
            default=0.0,
        )
        assert worst <= RELOAD_TOLERANCE, f"fold {fold.index}: discrepancia {worst!r} (A5)"


def test_a5_a_tampered_text_is_detected_by_the_reload(
    synthetic: tuple[DesignFrame, LightGBMModel],
) -> None:
    """El texto de **otro** fold no reproduce lo publicado: error tipado, no una cifra (A5)."""
    import dataclasses

    design, model = synthetic
    swapped = dataclasses.replace(model.folds[0], booster_model=model.folds[1].booster_model)
    with pytest.raises(LightGBMDeterminismError):
        reload_probabilities(swapped, design.frame)


def test_a5_the_model_payload_is_json_pure_and_carries_the_text(
    synthetic: tuple[DesignFrame, LightGBMModel],
) -> None:
    """El payload del modelo es JSON puro, lleva el texto y cambia si cambia el modelo (A5)."""
    _, model = synthetic
    payload = model.to_payload()
    text = canonical_text(payload)
    assert isinstance(text, str)
    folds = cast("list[dict[str, object]]", payload["folds"])
    assert all(isinstance(item["booster_model"], str) for item in folds)
    assert lightgbm_model.__name__
    first = model_comparison.model_sha256(model)
    assert first == model_comparison.model_sha256(model)
    assert isinstance(first, str) and len(first) == 64


def test_a5_probabilities_are_none_outside_every_test(
    synthetic: tuple[DesignFrame, LightGBMModel],
) -> None:
    """Fuera de todo *test* no hay prediccion honesta: `None`, nunca `0.0`."""
    design, model = synthetic
    raw = probabilities(model, design.frame)
    calibrated = calibrated_probabilities(model, design.frame)
    assert len(raw) == design.frame.height
    covered = {position for fold in model.folds for position in fold.test_positions}
    for index in range(len(raw)):
        if index in covered:
            assert raw[index] is not None
            assert calibrated[index] is not None
        else:
            assert raw[index] is None
            assert calibrated[index] is None


# ─────────────────────────────────────────────────────────────────────────────
# Contratos de entrada (apoyo)
# ─────────────────────────────────────────────────────────────────────────────
def test_the_design_contract_is_validated_with_typed_errors() -> None:
    """Columnas que faltan, folds vacios y solape train/test son errores tipados."""
    design = _design(_synthetic_frame(rows=600, seed=7))
    with pytest.raises(InvalidLightGBMInputError):
        fit_lightgbm(design, splits=())
    with pytest.raises(InvalidLightGBMInputError):
        fit_lightgbm(
            _design(_synthetic_frame(rows=600, seed=7).drop("har_forecast")),
            splits=(SplitAssignment(index=0, train=(0, 1, 2), test=(3, 4, 5)),),
        )
    with pytest.raises(InvalidLightGBMInputError):
        fit_lightgbm(
            design,
            splits=(SplitAssignment(index=0, train=(0, 1, 2, 3), test=(3, 4, 5)),),
        )


def test_the_default_hyperparameters_are_the_published_ones(
    synthetic: tuple[DesignFrame, LightGBMModel],
) -> None:
    """El modelo publica el diccionario con el que se ajusto, no una copia vacia."""
    _, model = synthetic
    assert dict(model.hyperparameters) == EXPECTED_HYPERPARAMETERS
    assert model.features == BASELINE_FEATURES
    assert model.seed == 20260920
