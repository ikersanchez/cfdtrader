"""Tests del modulo puro de calibracion (#25): A1, A2, A4 y A5.

Todo se mide con **posiciones y puntuaciones sinteticas**: el modulo no conoce el almacen ni el
plan, asi que no hace falta el `data/` del repositorio (eso lo cubren
``tests/test_baseline_calibration.py`` y el informe). El ruido es determinista
(``sha256(f"{seed}:{index}")``) y nunca un generador congruencial: dos semillas distintas de un
LCG producen la misma secuencia desplazada una constante.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Final, cast

import pytest

from cfdtrader.models import baseline, calibration
from cfdtrader.models.calibration import (
    CALIBRATION_FRACTION,
    CALIBRATION_HYPERPARAMETERS,
    METHOD_ISOTONIC,
    METHOD_NONE,
    METHOD_PLATT,
    MIN_CALIBRATION_SESSIONS,
    PLATT_MAX_CALIBRATION_SESSIONS,
    REASON_RANKING_INVERTED,
    REASON_SINGLE_CLASS,
    REASON_TRAIN_TOO_SMALL,
    Calibration,
    InvalidCalibrationInputError,
    fit_calibration,
    method_counts,
    select_method,
    sigmoid,
    split_train_for_calibration,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODULE_PATH: Final[Path] = Path(str(calibration.__file__))
TREE: Final[ast.Module] = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))

#: Modulos que ``models/calibration.py`` **no** puede importar (A1), por AST y no por texto.
FORBIDDEN_ROOTS: Final[tuple[str, ...]] = (
    "duckdb",
    "cfdtrader.data",
    "cfdtrader.analysis",
    "cfdtrader.backtest",
    "sklearn.model_selection",
)

#: Serializadores binarios prohibidos por A1: lo ajustado viaja como *floats*.
SERIALISERS: Final[frozenset[str]] = frozenset(
    {"pickle", "joblib", "cloudpickle", "marshal", "dill"}
)

#: Nombres que A1 prohibe **mencionar** en el modulo, aunque el enunciado los use para explicar
#: la divergencia declarada (la explicacion vive en el comentario de la issue y en el informe).
BANNED_NAMES: Final[frozenset[str]] = frozenset(
    {"CalibratedClassifierCV", "FrozenEstimator", "calibrated_classifiers_"}
)


def _noise(seed: str, index: int) -> float:
    """Ruido determinista en ``[-1, 1]`` a partir de ``sha256``."""
    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / (2**63) - 1.0


def _scores_and_outcomes(
    count: int, *, inverted: bool = False, seed: str = "a25"
) -> tuple[list[float], list[int]]:
    """``count`` puntuaciones con su etiqueta, ordenadas al reves si ``inverted``."""
    outcomes = [index % 2 for index in range(count)]
    scores = [
        ((0.5 - y) if inverted else (y - 0.5)) + 0.25 * _noise(f"{seed}:{index}", index)
        for index, y in enumerate(outcomes)
    ]
    return scores, outcomes


def _imported_modules(tree: ast.Module) -> set[str]:
    """Todos los modulos importados por el AST, con el nombre de cada simbolo importado."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _referenced_names(tree: ast.Module) -> set[str]:
    """Los identificadores y atributos que el modulo **menciona** (nunca el texto)."""
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }


# ─────────────────────────────────────────────────────────────────────────────
# A1 - modulos y fronteras
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_module_is_pure_and_keeps_its_layer() -> None:
    """A1: `calibration.py` existe, no cruza fronteras, ni lee ni escribe, ni serializa."""
    assert MODULE_PATH.is_file()
    imported = _imported_modules(TREE)
    forbidden = sorted(
        name
        for name in imported
        if any(name == root or name.startswith(f"{root}.") for root in FORBIDDEN_ROOTS)
    )
    assert not forbidden, f"`models/calibration.py` importa modulos prohibidos por A1: {forbidden}"
    serialisers = sorted(name for name in imported if name.split(".")[0] in SERIALISERS)
    assert not serialisers, f"el modulo importa un serializador binario: {serialisers}"

    mentioned = _referenced_names(TREE)
    assert not BANNED_NAMES & mentioned, (
        "el modulo menciona un calibrador envolvente: lo ajustado tiene que publicarse como "
        f"floats (A1), y los nombres prohibidos son {sorted(BANNED_NAMES)}"
    )

    calls = {
        node.func.attr
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not calls & {"sql", "read_parquet", "to_parquet", "write_text", "write_bytes", "mkdir"}
    assert not any(isinstance(node, ast.Name) and node.id == "open" for node in ast.walk(TREE)), (
        "el modulo de calibracion no abre ficheros: no lee ni escribe (A1)"
    )


def test_a1_the_sigmoid_has_a_single_definition_in_the_calibration_module() -> None:
    """A1: `sigmoid` vive en `calibration.py` y `baseline.py` la **importa**, no la duplica."""
    sources = _imported_modules(ast.parse(Path(str(baseline.__file__)).read_text(encoding="utf-8")))
    assert "cfdtrader.models.calibration.sigmoid" in sources
    definitions = {
        node.name
        for node in ast.walk(ast.parse(Path(str(baseline.__file__)).read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef)
    }
    assert "sigmoid" not in definitions, (
        "`baseline.py` vuelve a definir `sigmoid`: la unica definicion esta en "
        "`models.calibration` (A1)"
    )
    assert baseline.sigmoid is calibration.sigmoid
    assert sigmoid(0.0) == 0.5
    assert sigmoid(1000.0) == 1.0
    assert sigmoid(-1000.0) == 0.0
    assert sigmoid(2.0) == pytest.approx(0.8807970779778823, abs=0.0)


def _as_list(value: object) -> list[object]:
    """Un valor del payload que se sabe lista (para comprobar su longitud, sin indexar a ciegas)."""
    assert isinstance(value, list)
    return cast("list[object]", value)


# ─────────────────────────────────────────────────────────────────────────────
# A2 - el reparto del train
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_train_is_split_into_a_head_and_a_purged_tail() -> None:
    """A2: `fit` (cabeza) y `calibration` (cola) cubren el train, sin solape y con el 20 %."""
    train = tuple(range(1000, 1187))
    test = tuple(range(1187, 1237))
    split = split_train_for_calibration(train, label_horizon=(0,) * 1300)

    assert len(train) == 187
    assert split.n_calibration == math.floor(CALIBRATION_FRACTION * len(train)) == 37
    assert set(split.fit) | set(split.calibration) == set(train)
    assert not set(split.fit) & set(split.calibration)
    assert set(split.calibration).isdisjoint(test)
    assert split.calibration == tuple(range(1150, 1187))
    assert split.calibration_start == 1150
    assert split.n_fit == 150
    assert split.purge_sessions == 0
    assert split.exclusions_are_no_op is True
    payload = split.to_payload()
    assert payload["purge_sessions"] == 0
    assert payload["exclusions_are_no_op"] is True
    assert payload["n_calibration"] == 37
    assert json.dumps(payload)  # JSON puro, sin objetos


def test_a2_a_horizon_of_two_turns_the_purge_on() -> None:
    """Control negativo de A2: con ``label_horizon = 2`` la purga deja de ser un no-op."""
    train = tuple(range(1000, 1187))
    split = split_train_for_calibration(train, label_horizon=(2,) * 1300)
    assert split.purge_sessions == 2
    assert split.exclusions_are_no_op is False
    assert split.fit == tuple(range(1000, 1148))
    assert split.calibration == tuple(range(1150, 1187))


def test_a2_the_split_validates_its_inputs() -> None:
    """A2: los bordes del reparto son errores tipados, nunca un reparto a medias."""
    with pytest.raises(InvalidCalibrationInputError):
        split_train_for_calibration((), label_horizon=(0,))
    with pytest.raises(InvalidCalibrationInputError):
        split_train_for_calibration((5, 6, 7), label_horizon=(0, 0))
    with pytest.raises(InvalidCalibrationInputError):
        split_train_for_calibration((5, 6, 7), label_horizon=(0, 0, 0, 0))
    with pytest.raises(InvalidCalibrationInputError):
        split_train_for_calibration((0, 1, 2), label_horizon=(-1, 0, 0))


# ─────────────────────────────────────────────────────────────────────────────
# A4 - la regla del metodo y la monotonia
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_the_method_rule_is_evaluated_per_fold_with_the_measured_count() -> None:
    """A4: Platt por debajo de 500, isotonica a partir de ahi, con el recuento medido."""
    assert PLATT_MAX_CALIBRATION_SESSIONS == 500
    pairs = {
        437: METHOD_PLATT,
        499: METHOD_PLATT,
        500: METHOD_ISOTONIC,
        527: METHOD_ISOTONIC,
    }
    for n_calibration, expected in pairs.items():
        assert select_method(n_calibration) == expected
    assert (METHOD_PLATT, METHOD_ISOTONIC, METHOD_NONE) == ("platt", "isotonic", "none")
    with pytest.raises(InvalidCalibrationInputError):
        select_method(-1)
    assert method_counts(["platt", "isotonic", "isotonic", "none"]) == {
        "platt": 1,
        "isotonic": 2,
        "none": 1,
    }
    assert method_counts([]) == {"platt": 0, "isotonic": 0, "none": 0}


def test_a4_a_calibration_that_inverts_the_ranking_is_declared_and_not_published() -> None:
    """A4: un calibrador que invertiria el orden no se publica, se declara `ranking_inverted`."""
    scores, outcomes = _scores_and_outcomes(120, inverted=True, seed="inv-platt")
    split = split_train_for_calibration(tuple(range(0, 150)), label_horizon=(0,) * 400)
    platt = fit_calibration(split, scores=scores[:30], outcomes=outcomes[:30])
    assert platt.method == METHOD_NONE
    assert platt.reason == REASON_RANKING_INVERTED
    assert platt.parameters() is None

    # La misma inversión con un bloque grande: la isotonica elige el sentido decreciente.
    big_split = split_train_for_calibration(tuple(range(0, 2600)), label_horizon=(0,) * 2700)
    scores, outcomes = _scores_and_outcomes(520, inverted=True, seed="inv-iso")
    isotonic = fit_calibration(big_split, scores=scores, outcomes=outcomes)
    assert big_split.n_calibration == 520
    assert isotonic.method == METHOD_NONE
    assert isotonic.reason == REASON_RANKING_INVERTED


def test_a4_the_published_calibrators_are_monotone() -> None:
    """A4: Platt publica ``coef > 0`` y la isotonica ``increasing`` y umbrales crecientes."""
    scores, outcomes = _scores_and_outcomes(120, seed="mono-platt")
    split = split_train_for_calibration(tuple(range(0, 150)), label_horizon=(0,) * 400)
    platt = fit_calibration(split, scores=scores[:30], outcomes=outcomes[:30])
    assert platt.method == METHOD_PLATT
    assert platt.reason is None
    assert platt.coef is not None and platt.coef > 0.0
    parameters = platt.parameters()
    assert parameters is not None
    assert set(parameters) >= {"coef", "intercept", "mean", "scale"}
    calibrated = platt.calibrate(scores[:30])
    assert all(value is not None for value in calibrated)
    assert all(value is not None and 0.0 <= value <= 1.0 for value in calibrated)

    scores, outcomes = _scores_and_outcomes(520, seed="mono-iso")
    big_split = split_train_for_calibration(tuple(range(0, 2600)), label_horizon=(0,) * 2700)
    isotonic = fit_calibration(big_split, scores=scores, outcomes=outcomes)
    assert big_split.n_calibration == 520
    assert isotonic.method == METHOD_ISOTONIC
    assert isotonic.thresholds and isotonic.values
    assert list(isotonic.values) == sorted(isotonic.values)
    assert list(isotonic.thresholds) == sorted(isotonic.thresholds)
    published = isotonic.parameters()
    assert published is not None
    assert published["increasing"] is True
    assert len(_as_list(published["values"])) == len(isotonic.values)


# ─────────────────────────────────────────────────────────────────────────────
# A5 - estados no medibles, no degradados
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_unmeasurable_states_are_published_and_never_degraded() -> None:
    """A5: `none` con su motivo y la cruda tal cual; ninguna probabilidad sale de ``[0, 1]``."""
    short = split_train_for_calibration(tuple(range(0, 40)), label_horizon=(0,) * 100)
    assert short.n_calibration == 8 < MIN_CALIBRATION_SESSIONS
    scores, outcomes = _scores_and_outcomes(8, seed="short")
    too_small = fit_calibration(short, scores=scores, outcomes=outcomes)
    assert too_small.method == METHOD_NONE
    assert too_small.reason == REASON_TRAIN_TOO_SMALL
    assert too_small.calibrated is False
    assert too_small.to_payload()["parameters"] is None
    assert too_small.calibrate([0.0, 1.0, 2.0]) == (None, None, None)

    single = split_train_for_calibration(tuple(range(0, 150)), label_horizon=(0,) * 300)
    assert single.n_calibration == 30
    one_class = fit_calibration(single, scores=[0.0] * 30, outcomes=[1] * 30)
    assert one_class.method == METHOD_NONE
    assert one_class.reason == REASON_SINGLE_CLASS
    assert one_class.n_positives == 30
    assert set(one_class.calibrate([0.0])) == {None}

    zero_class = fit_calibration(single, scores=[0.0] * 30, outcomes=[0] * 30)
    assert zero_class.method == METHOD_NONE
    assert zero_class.reason == REASON_SINGLE_CLASS

    # Todas las puntuaciones iguales: no hay orden que preservar, luego no se calibra.
    flat = fit_calibration(single, scores=[1.234] * 30, outcomes=[index % 2 for index in range(30)])
    assert flat.method == METHOD_NONE
    assert flat.reason == REASON_RANKING_INVERTED

    for item in (too_small, one_class, zero_class, flat):
        payload = item.to_payload()
        assert payload["method"] == METHOD_NONE
        assert json.dumps(payload, allow_nan=False)
    assert too_small.to_payload()["reason"] == REASON_TRAIN_TOO_SMALL
    assert CALIBRATION_HYPERPARAMETERS == {
        "calibration_fraction": CALIBRATION_FRACTION,
        "min_calibration_sessions": MIN_CALIBRATION_SESSIONS,
        "platt_max_calibration_sessions": PLATT_MAX_CALIBRATION_SESSIONS,
    }


def test_a5_the_calibrator_validates_its_scores() -> None:
    """A5: puntuaciones y etiquetas van alineadas con `calibration_positions`."""
    split = split_train_for_calibration(tuple(range(0, 150)), label_horizon=(0,) * 300)
    scores, outcomes = _scores_and_outcomes(30, seed="align")
    with pytest.raises(InvalidCalibrationInputError):
        fit_calibration(split, scores=scores[:10], outcomes=outcomes[:10])
    with pytest.raises(InvalidCalibrationInputError):
        fit_calibration(split, scores=scores, outcomes=outcomes[:-1])


def test_a5_a_hand_built_none_calibration_declares_itself() -> None:
    """A5: una `Calibration` construida a mano tambien publica `none` con su motivo."""
    item = Calibration(
        method=METHOD_NONE,
        reason=REASON_RANKING_INVERTED,
        n_fit=10,
        n_calibration=0,
        n_positives=0,
        calibration_positions=(),
        purge_sessions=0,
        exclusions_are_no_op=True,
    )
    assert item.calibrated is False
    assert item.parameters() is None
    assert item.calibrate([0.25, 0.75]) == (None, None)
    payload = item.to_payload()
    assert payload["calibrated"] is False
    assert payload["reason"] == REASON_RANKING_INVERTED
    assert json.dumps(payload)
