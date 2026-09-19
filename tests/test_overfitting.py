"""Tests del nucleo de correccion por sobreajuste (#16): DSR y PBO por CSCV.

Un test por criterio de aceptacion. Aqui viven los del **nucleo puro** (A2-A9, A16, A17,
A19, A33 y A34); el registro de experimentos y el informe estan en
``tests/test_experiment_log.py``. Los valores del DSR se calculan **a mano** en el propio
test (constantes explicitas y ``statistics.NormalDist``), nunca se copian del modulo.

Las series sinteticas se construyen con ``sin``/``cos`` y con la semilla declarada: el test
no depende del azar de la corrida, y las dos validaciones de oro (ruido y senal) usan las
constantes con nombre del modulo.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Final, cast

import numpy as np
import pytest

from cfdtrader.analysis import experiment_log
from cfdtrader.backtest import overfitting
from cfdtrader.backtest.metrics import sharpe_ratio
from cfdtrader.backtest.overfitting import (
    BacktestOverfitting,
    CombinationBudgetError,
    DeflatedSharpe,
    DegenerateMatrixError,
    DegenerateSeriesError,
    InsufficientObservationsError,
    InsufficientTrialsError,
    InsufficientVariantsError,
    InvalidBlocksError,
    InvalidConfidenceLevelError,
    InvalidMatrixShapeError,
    InvalidSamplingSeedError,
    InvalidTrialsError,
    InvalidVarianceError,
    NonFiniteInputError,
    NonPositiveDenominatorError,
    OverfittingError,
    deflated_sharpe_ratio,
    noise_matrix,
    probability_of_backtest_overfitting,
    select_variant,
    signal_matrix,
    sr0_expected_max,
    variant_sharpe_variance,
)

MODULE_PATH = Path(str(overfitting.__file__))
SOURCE = MODULE_PATH.read_text(encoding="utf-8")
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPORTS: Final[Path] = REPO_ROOT / "data" / "derived" / "reports"
AS_OF: Final[datetime] = datetime(2026, 9, 19, 12, tzinfo=UTC)

#: Semilla declarada de los sondeos del test (ninguna entra en el modulo).
PROBE_SEED: Final[int] = 20260919

#: Serie con senal **moderada**: el DSR queda en la zona media, sin saturar en 0 ni en 1.
_LOW_MEAN: Final[float] = 0.0001
_HIGH_MEAN: Final[float] = 0.00012


def _series(
    *, mean: float = _LOW_MEAN, amplitude: float = 0.0012, n: int = 400
) -> tuple[float, ...]:
    """Serie determinista (sin azar) con Sharpe por sesion moderado."""
    return tuple(amplitude * math.sin(index / 5.0) + mean for index in range(n))


def _hand_moments(values: tuple[float, ...]) -> dict[str, float]:
    """Momentos poblacionales a mano: ``g3 = m3/m2**1.5`` y ``g4 = m4/m2**2`` (A2)."""
    size = len(values)
    mean = sum(values) / size
    m2 = sum((value - mean) ** 2 for value in values) / size
    m3 = sum((value - mean) ** 3 for value in values) / size
    m4 = sum((value - mean) ** 4 for value in values) / size
    return {"skewness": m3 / m2**1.5, "kurtosis": m4 / m2**2}


def _matrix(*, rows: int, columns: int, seed: int) -> tuple[tuple[float, ...], ...]:
    """Matriz `T x N` determinista y no degenerada, con el *stream* congelado de #15."""
    rng = np.random.RandomState(seed)
    return tuple(
        tuple(float(value) for value in row) for row in rng.normal(scale=0.01, size=(rows, columns))
    )


def _rows_with_mean(
    matrix: tuple[tuple[float, ...], ...], column: int, mean: float
) -> tuple[tuple[float, ...], ...]:
    """La misma matriz con una media plantada en una columna (dominancia real)."""
    return tuple(
        tuple(value + (mean if index == column else 0.0) for index, value in enumerate(row))
        for row in matrix
    )


def _fingerprint(root: Path) -> dict[str, str]:
    """Ruta relativa → sha256 de cada fichero (la huella de ``tests/conftest.py``)."""
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _column(matrix: tuple[tuple[float, ...], ...], index: int) -> tuple[float, ...]:
    """La columna `index` como serie."""
    return tuple(row[index] for row in matrix)


def _sharpes(matrix: tuple[tuple[float, ...], ...]) -> list[float]:
    """Sharpe por sesion de cada columna, con la funcion importada de #15."""
    return [
        sharpe_ratio(_column(matrix, index), annualization=1) for index in range(len(matrix[0]))
    ]


def _sample_std(values: tuple[float, ...]) -> float:
    """Desviacion estandar muestral a mano (``ddof = 1``), para la prueba de unidades (A8)."""
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


#: Serie de dos niveles que agota la correccion por no normalidad (A19): 60 valores en el
#: nivel bajo y 40 en el alto (``g3 = 0.4082...``, ``g4 = 7/6``) desplazados para que el
#: Sharpe por sesion caiga en el minimo del denominador (``SR* = 2 g3 / (g4 - 1)``).
#: Todos los retornos son **positivos y validos**: el caso no se fuerza con datos invalidos.
_TWO_LEVEL_SHIFT: Final[float] = 0.30151134457776385
DENOMINATOR_SERIES: Final[tuple[float, ...]] = (-0.05 + _TWO_LEVEL_SHIFT,) * 60 + (
    0.075 + _TWO_LEVEL_SHIFT,
) * 40


# ─────────────────────────────────────────────────────────────────────────────
# A2 — la formula del DSR es la de Bailey y Lopez de Prado, literal
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_formula_is_the_declared_one() -> None:
    values = _series()
    n_trials, variance = 12, 0.0025
    moments = _hand_moments(values)
    sr = sharpe_ratio(values, annualization=1)
    normal = NormalDist()
    manual_sr0 = math.sqrt(variance) * (
        (1.0 - overfitting.EULER_MASCHERONI) * normal.inv_cdf(1.0 - 1.0 / n_trials)
        + overfitting.EULER_MASCHERONI * normal.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    )
    denominator = 1.0 - moments["skewness"] * sr + (moments["kurtosis"] - 1.0) / 4.0 * sr * sr
    expected = normal.cdf((sr - manual_sr0) * math.sqrt(len(values) - 1) / math.sqrt(denominator))

    result = deflated_sharpe_ratio(values, n_trials=n_trials, sr_variance=variance)
    assert result.dsr == expected
    assert result.sr_observed == sr
    assert result.sr0_expected_max == manual_sr0
    assert result.skewness == moments["skewness"]
    assert result.kurtosis == moments["kurtosis"]
    assert result.non_normality_denominator == denominator
    # la curtosis es **no excedente**: en una serie casi normal vale 3, no 0
    normal_like = deflated_sharpe_ratio(_column(noise_matrix(), 0), n_trials=10, sr_variance=0.0008)
    assert 2.5 < normal_like.kurtosis < 3.5
    assert normal_like.kurtosis != normal_like.kurtosis - 3.0
    assert "Bailey" in SOURCE
    assert result.verdict in {overfitting.VERDICT_SIGNIFICANT, overfitting.VERDICT_NOT_SIGNIFICANT}


# ─────────────────────────────────────────────────────────────────────────────
# A3 — entradas explicitas del DSR
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_the_inputs_are_explicit_and_typed() -> None:
    parameters = inspect.signature(deflated_sharpe_ratio).parameters
    assert parameters["n_trials"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["sr_variance"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["confidence_level"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["confidence_level"].default == overfitting.DSR_CONFIDENCE_LEVEL

    values = _series()
    for trials in (1, 0, -3):
        with pytest.raises(InsufficientTrialsError):
            deflated_sharpe_ratio(values, n_trials=trials, sr_variance=0.01)
    for variance in (-0.001, float("inf"), float("nan")):
        with pytest.raises(InvalidVarianceError):
            deflated_sharpe_ratio(values, n_trials=5, sr_variance=variance)
    with pytest.raises(InvalidTrialsError):
        deflated_sharpe_ratio(values, n_trials=2**53, sr_variance=0.01)
    with pytest.raises(InsufficientObservationsError):
        deflated_sharpe_ratio((0.01,), n_trials=5, sr_variance=0.01)
    for confidence in (0.0, 1.0, -0.5, float("nan")):
        with pytest.raises(InvalidConfidenceLevelError):
            deflated_sharpe_ratio(values, n_trials=5, sr_variance=0.01, confidence_level=confidence)
    # sin varianza de variantes no se finge deflacion: SR0 = 0 y se declara
    flat = deflated_sharpe_ratio(values, n_trials=5, sr_variance=0.0)
    assert flat.sr0_expected_max == 0.0
    assert flat.deflation == overfitting.DEFLATION_NONE
    assert flat.note is not None
    assert sr0_expected_max(n_trials=5, sr_variance=0.0) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# A4 — que se publica del DSR
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_the_number_never_travels_without_a_verdict() -> None:
    required = {
        "dsr",
        "sr_observed",
        "sr0_expected_max",
        "n_trials",
        "sr_variance",
        "n_observations",
        "skewness",
        "kurtosis",
        "confidence_level",
        "deflation",
        "verdict",
    }
    for mean in (_LOW_MEAN, _HIGH_MEAN):
        result = deflated_sharpe_ratio(_series(mean=mean), n_trials=10, sr_variance=0.0025)
        payload = result.to_payload()
        assert required <= set(payload)
        assert isinstance(result, DeflatedSharpe)
        expected = (
            overfitting.VERDICT_SIGNIFICANT
            if cast("float", payload["dsr"]) >= cast("float", payload["confidence_level"])
            else overfitting.VERDICT_NOT_SIGNIFICANT
        )
        assert payload["verdict"] == expected
        assert payload["units"] == overfitting.PER_SESSION
        assert payload["threshold_rule"] == overfitting.THRESHOLD_RULE
        assert json.dumps(payload, allow_nan=False)
    verdicts = {
        deflated_sharpe_ratio(_series(mean=mean), n_trials=10, sr_variance=0.0025).verdict
        for mean in (_LOW_MEAN, _HIGH_MEAN)
    }
    assert verdicts == {overfitting.VERDICT_NOT_SIGNIFICANT, overfitting.VERDICT_SIGNIFICANT}


# ─────────────────────────────────────────────────────────────────────────────
# A5 — umbral declarado y tension citada
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_the_threshold_is_declared_and_the_verdict_crosses_it() -> None:
    assert overfitting.DSR_CONFIDENCE_LEVEL == 0.95
    assert "§11.6" in overfitting.THRESHOLD_RULE
    assert "> 0" in overfitting.THRESHOLD_RULE
    assert "no** como" in overfitting.THRESHOLD_RULE

    below = deflated_sharpe_ratio(_series(mean=_LOW_MEAN), n_trials=10, sr_variance=0.0025)
    above = deflated_sharpe_ratio(_series(mean=_HIGH_MEAN), n_trials=10, sr_variance=0.0025)
    assert below.dsr < overfitting.DSR_CONFIDENCE_LEVEL <= above.dsr
    assert below.verdict == overfitting.VERDICT_NOT_SIGNIFICANT
    assert above.verdict == overfitting.VERDICT_SIGNIFICANT
    assert below.threshold_rule == above.threshold_rule == overfitting.THRESHOLD_RULE
    assert below.confidence_level == above.confidence_level == overfitting.DSR_CONFIDENCE_LEVEL


# ─────────────────────────────────────────────────────────────────────────────
# A6 — PBO por CSCV
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_pbo_is_cscv_over_the_variant_matrix() -> None:
    plain = _matrix(rows=64, columns=6, seed=PROBE_SEED)
    dominant = _rows_with_mean(plain, 0, 0.01)

    with_signal = probability_of_backtest_overfitting(dominant, blocks=8)
    with_noise = probability_of_backtest_overfitting(plain, blocks=8)
    assert isinstance(with_signal, BacktestOverfitting)
    assert with_signal.pbo == 0.0
    assert with_signal.verdict == overfitting.VERDICT_NOT_DETECTED
    assert with_noise.pbo > 0.20
    assert with_noise.pbo == pytest.approx(0.45714285714285713, rel=1e-9)
    assert with_noise.verdict == overfitting.VERDICT_DETECTED

    payload = with_signal.to_payload()
    assert payload["blocks"] == 8
    assert payload["n_observations"] == 64
    assert payload["n_variants"] == 6
    assert payload["n_combinations"] == math.comb(8, 4)
    assert payload["n_combinations_drawn"] == math.comb(8, 4)
    assert payload["method"] == overfitting.METHOD_EXHAUSTIVE
    assert payload["sampling_seed"] is None
    assert payload["pbo_max"] == overfitting.PBO_MAX
    assert payload["units"] == overfitting.PER_SESSION
    assert 0.0 < cast("float", payload["omega_median"]) < 1.0
    assert math.isfinite(cast("float", payload["logit_median"]))
    assert math.isfinite(cast("float", payload["best_oos_sharpe_median"]))
    assert payload["omega_rule"] and payload["tie_rule"]
    assert json.dumps(payload, allow_nan=False)
    # el PBO es una fraccion exacta de las combinaciones evaluadas
    assert with_signal.pbo * math.comb(8, 4) == round(with_signal.pbo * math.comb(8, 4))
    # T=64 y S=2: la unica particion posible es 1 bloque contra 1
    assert probability_of_backtest_overfitting(plain, blocks=2).n_combinations == 2


# ─────────────────────────────────────────────────────────────────────────────
# A7 — presupuesto combinatorio explicito
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_the_combinatorial_budget_is_explicit() -> None:
    assert overfitting.MAX_COMBINATIONS == 12_870 == math.comb(16, 8)
    assert overfitting.DEFAULT_BLOCKS == 16

    matrix = _matrix(rows=100, columns=4, seed=PROBE_SEED)
    with pytest.raises(CombinationBudgetError):
        probability_of_backtest_overfitting(matrix, blocks=20)
    with pytest.raises(InvalidSamplingSeedError):
        probability_of_backtest_overfitting(matrix, blocks=20, sampling_seed=2**32)

    first = probability_of_backtest_overfitting(matrix, blocks=20, sampling_seed=11)
    second = probability_of_backtest_overfitting(matrix, blocks=20, sampling_seed=11)
    assert first.method == overfitting.METHOD_SAMPLED
    assert first.sampling_seed == 11
    assert first.n_combinations == math.comb(20, 10)
    assert first.n_combinations_drawn == overfitting.MAX_COMBINATIONS
    assert first.pbo == second.pbo
    assert first.to_payload() == second.to_payload()

    exhaustive = probability_of_backtest_overfitting(matrix, blocks=4)
    assert exhaustive.method == overfitting.METHOD_EXHAUSTIVE
    assert exhaustive.sampling_seed is None
    assert exhaustive.n_combinations == exhaustive.n_combinations_drawn == math.comb(4, 2)


# ─────────────────────────────────────────────────────────────────────────────
# A8 — una sola convencion de unidades
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_the_units_are_per_session_and_the_formula_is_not_invariant() -> None:
    values = _series(mean=_LOW_MEAN)
    variance = 0.0025
    per_session = deflated_sharpe_ratio(values, n_trials=10, sr_variance=variance)
    assert per_session.units == overfitting.PER_SESSION == "per_session"
    assert per_session.to_payload()["units"] == "per_session"
    assert per_session.sr_observed == sharpe_ratio(values, annualization=1)
    assert sharpe_ratio(values) == pytest.approx(per_session.sr_observed * math.sqrt(252))

    # el mismo caso **anualizado**: SR por sqrt(252) y V[SR] por 252 (desplazando la serie,
    # que es la unica forma de mover el SR sin cambiar la forma de la distribucion)
    standard = _sample_std(values)
    target = per_session.sr_observed * math.sqrt(252)
    shift = (target - per_session.sr_observed) * standard
    annualised = deflated_sharpe_ratio(
        tuple(value + shift for value in values), n_trials=10, sr_variance=variance * 252
    )
    assert annualised.sr_observed == pytest.approx(target)
    assert annualised.sr0_expected_max == pytest.approx(
        per_session.sr0_expected_max * math.sqrt(252)
    )
    assert annualised.skewness == pytest.approx(per_session.skewness)
    assert annualised.kurtosis == pytest.approx(per_session.kurtosis)

    # y la mezcla prohibida: SR por sesion con V[SR] anualizada
    mixed = deflated_sharpe_ratio(values, n_trials=10, sr_variance=variance * 252)

    # el numerador escala y el denominador no: los tres numeros son plausibles y distintos
    assert per_session.dsr == pytest.approx(0.8860068382928874, rel=1e-9)
    assert annualised.dsr == 1.0
    assert mixed.dsr == 0.0
    assert len({per_session.dsr, annualised.dsr, mixed.dsr}) == 3
    assert per_session.verdict == overfitting.VERDICT_NOT_SIGNIFICANT
    assert annualised.verdict == overfitting.VERDICT_SIGNIFICANT
    assert mixed.verdict == overfitting.VERDICT_NOT_SIGNIFICANT


# ─────────────────────────────────────────────────────────────────────────────
# A9 — reutilizar, no duplicar
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_the_point_sharpe_is_reused_and_never_reimplemented() -> None:
    values = _series()
    result = deflated_sharpe_ratio(values, n_trials=8, sr_variance=0.001)
    assert result.sr_observed == sharpe_ratio(values, annualization=1)
    assert overfitting.sharpe_ratio is sharpe_ratio
    assert "252" not in SOURCE
    tree = ast.parse(SOURCE)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "sharpe_ratio" in imported
    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert not defined & {"_mean", "_sample_std", "sharpe", "_sharpe", "stdev", "_std"}
    matrix = _matrix(rows=32, columns=5, seed=PROBE_SEED)
    expected = max(
        range(5), key=lambda index: sharpe_ratio(_column(matrix, index), annualization=1)
    )
    assert select_variant(matrix) == expected


# ─────────────────────────────────────────────────────────────────────────────
# A16 — ruido puro ⇒ no significativo (criterio de oro)
# ─────────────────────────────────────────────────────────────────────────────
def test_a16_pure_noise_is_not_significant() -> None:
    assert overfitting.SYNTHETIC_VARIANTS == 20
    assert overfitting.SYNTHETIC_OBSERVATIONS % overfitting.DEFAULT_BLOCKS == 0
    matrix = noise_matrix()
    assert matrix == noise_matrix()
    assert matrix != noise_matrix(seed=overfitting.NOISE_SEED + 1)

    sharpes = _sharpes(matrix)
    chosen = select_variant(matrix)
    result = deflated_sharpe_ratio(
        _column(matrix, chosen),
        n_trials=overfitting.SYNTHETIC_VARIANTS,
        sr_variance=variant_sharpe_variance(sharpes),
    )
    pbo = probability_of_backtest_overfitting(matrix, blocks=overfitting.DEFAULT_BLOCKS)
    assert result.verdict == overfitting.VERDICT_NOT_SIGNIFICANT
    assert result.dsr < overfitting.DSR_CONFIDENCE_LEVEL
    assert result.dsr == pytest.approx(0.8075739655292296, rel=1e-12)
    assert result.sr0_expected_max == pytest.approx(0.053550438391554496, rel=1e-12)
    # la banda declarada del test: el PBO del ruido cae alrededor de 0,5, no en un valor suelto
    assert 0.35 <= pbo.pbo <= 0.65
    assert pbo.pbo == pytest.approx(0.3825951825951826, rel=1e-12)
    assert pbo.verdict == overfitting.VERDICT_DETECTED


# ─────────────────────────────────────────────────────────────────────────────
# A17 — senal plantada ⇒ se detecta
# ─────────────────────────────────────────────────────────────────────────────
def test_a17_a_planted_signal_is_detected() -> None:
    assert overfitting.SIGNAL_MEAN > 0.0
    assert overfitting.SIGNAL_SEED != overfitting.NOISE_SEED
    matrix = signal_matrix()
    assert matrix == signal_matrix()
    assert select_variant(matrix) == 0

    sharpes = _sharpes(matrix)
    result = deflated_sharpe_ratio(
        _column(matrix, 0),
        n_trials=overfitting.SYNTHETIC_VARIANTS,
        sr_variance=variant_sharpe_variance(sharpes),
    )
    pbo = probability_of_backtest_overfitting(matrix, blocks=overfitting.DEFAULT_BLOCKS)
    noise_pbo = probability_of_backtest_overfitting(
        noise_matrix(), blocks=overfitting.DEFAULT_BLOCKS
    ).pbo
    assert result.verdict == overfitting.VERDICT_SIGNIFICANT
    assert result.dsr >= overfitting.DSR_CONFIDENCE_LEVEL
    assert result.dsr == pytest.approx(0.9949750545464502, rel=1e-12)
    assert pbo.pbo < overfitting.PBO_MAX
    assert pbo.pbo == pytest.approx(0.0004662004662004662, rel=1e-12)
    assert pbo.verdict == overfitting.VERDICT_NOT_DETECTED
    assert pbo.pbo < noise_pbo


# ─────────────────────────────────────────────────────────────────────────────
# A19 — degenerados con nombre propio
# ─────────────────────────────────────────────────────────────────────────────
def test_a19_degenerate_cases_have_their_own_error_class(tmp_path: Path) -> None:
    values = _series()
    matrix = _matrix(rows=96, columns=4, seed=PROBE_SEED)
    cases: list[tuple[str, Callable[[], object], type[OverfittingError]]] = [
        (
            "N < 2",
            lambda: deflated_sharpe_ratio(values, n_trials=1, sr_variance=0.01),
            InsufficientTrialsError,
        ),
        (
            "T < 2",
            lambda: deflated_sharpe_ratio((0.01,), n_trials=5, sr_variance=0.01),
            InsufficientObservationsError,
        ),
        (
            "varianza no estimable",
            lambda: deflated_sharpe_ratio(values, n_trials=5, sr_variance=-1.0),
            InvalidVarianceError,
        ),
        (
            "serie de desviacion cero",
            lambda: deflated_sharpe_ratio((0.01,) * 50, n_trials=5, sr_variance=0.01),
            DegenerateSeriesError,
        ),
        (
            "denominador no positivo",
            lambda: deflated_sharpe_ratio(DENOMINATOR_SERIES, n_trials=5, sr_variance=0.01),
            NonPositiveDenominatorError,
        ),
        (
            "matriz constante",
            lambda: probability_of_backtest_overfitting(((0.01,) * 4,) * 8, blocks=4),
            DegenerateMatrixError,
        ),
        (
            "T < 2 en la matriz",
            lambda: probability_of_backtest_overfitting(((0.01, 0.02),), blocks=2),
            InsufficientObservationsError,
        ),
        (
            "N < 2 en la matriz",
            lambda: probability_of_backtest_overfitting(tuple((v,) for v in values[:96]), blocks=8),
            InsufficientVariantsError,
        ),
        (
            "S impar",
            lambda: probability_of_backtest_overfitting(matrix, blocks=3),
            InvalidBlocksError,
        ),
        (
            "S no divide a T",
            lambda: probability_of_backtest_overfitting(matrix, blocks=10),
            InvalidBlocksError,
        ),
        (
            "matriz no rectangular",
            lambda: probability_of_backtest_overfitting(((0.1, 0.2), (0.3,)), blocks=2),
            InvalidMatrixShapeError,
        ),
        (
            "nan en la matriz",
            lambda: probability_of_backtest_overfitting(((0.1, float("nan")),) * 8, blocks=2),
            NonFiniteInputError,
        ),
    ]
    seen: set[type[OverfittingError]] = set()
    messages: list[str] = []
    for label, thunk, expected in cases:
        with pytest.raises(expected) as captured:
            thunk()
        assert str(captured.value), label
        seen.add(expected)
        messages.append(str(captured.value))
    # diez clases distintas para doce casos: los dos de bloques invalidos comparten clase
    # (S impar y S no divisor) y tambien los dos de observaciones, pero cada caso explica su
    # motivo con un mensaje propio: nunca una generica sin motivo
    assert len(seen) == 10
    assert len(set(messages)) == len(cases)

    # `n_trials` incoherente con el registro (A10, A19)
    runs_root = tmp_path / "runs"
    for index, seed in enumerate((1, 2)):
        experiment_log.record_experiment(
            runs_root=runs_root,
            config=experiment_log.ExperimentConfig(
                variant_id=f"probe-{index}",
                features=("synthetic_random_draw",),
                hyperparameters={"kind": "probe"},
                seed=seed,
                series_id="SYNTHETIC",
                window={"kind": "index", "start": 0, "stop": 1},
            ),
            result=experiment_log.ExperimentResult(
                sharpe_per_session=0.01 * (index + 1), n_observations=1
            ),
            as_of=AS_OF,
        )
    registry = experiment_log.load_registry(runs_root)
    with pytest.raises(experiment_log.TrialsMismatchError):
        experiment_log.require_trials_match_registry(
            n_trials=30, sr_variance=0.001, registry=registry
        )
    experiment_log.require_trials_match_registry(
        n_trials=registry.n_trials, sr_variance=registry.sr_variance, registry=registry
    )

    # un calculo no evaluable con datos **validos** se publica, no se rellena (A22)
    block = experiment_log.pbo_block(returns_matrix=tuple((v,) for v in values[:64]), blocks=8)
    assert block["state"] == experiment_log.VERDICT_NOT_EVALUABLE
    assert block["reason"] and "pbo" not in block


# ─────────────────────────────────────────────────────────────────────────────
# A33 — frontera con #67 (CSCV ≠ CPCV)
# ─────────────────────────────────────────────────────────────────────────────
def test_a33_cscv_is_not_cpcv() -> None:
    tree = ast.parse(SOURCE)
    imported_modules = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not any("splits" in module for module in imported_modules)
    assert not {node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)} & {
        "duckdb",
        "polars",
        "scipy",
    }
    # el modulo **nombra** el `SplitPlan` solo para declararlo fuera de alcance
    built = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "SplitPlan" not in built
    assert "walk_forward_splits" not in SOURCE
    assert not hasattr(overfitting, "SplitPlan")
    assert "#67" in experiment_log.CSCV_VERSUS_CPCV
    assert any(entry["issue"] == "#67" for entry in overfitting.FOLLOW_UPS)


# ─────────────────────────────────────────────────────────────────────────────
# A34 — el holdout de #68 y los artefactos existentes no se tocan
# ─────────────────────────────────────────────────────────────────────────────
def test_a34_the_holdout_and_the_existing_artifacts_are_untouched(tmp_path: Path) -> None:
    sources = {
        "overfitting": SOURCE.lower(),
        "experiment_log": Path(str(experiment_log.__file__)).read_text(encoding="utf-8").lower(),
    }
    for name, text in sources.items():
        for forbidden in (
            "ultimos 12 meses",
            "last_12_months",
            "holdout_period",
            "reserved_period",
        ):
            assert forbidden not in text, f"{name} declara un holdout que es #68"
    assert "#68" in experiment_log.HOLDOUT_BOUNDARY

    before_reports = _fingerprint(REPORTS)
    before_runs = _fingerprint(REPO_ROOT / "runs")
    experiment_log.analyse(
        runs_root=tmp_path / "runs", reports_dir=tmp_path / "reports", as_of=AS_OF, write=True
    )
    assert _fingerprint(REPORTS) == before_reports
    assert _fingerprint(REPO_ROOT / "runs") == before_runs
    assert _fingerprint(tmp_path / "reports")


# ─────────────────────────────────────────────────────────────────────────────
# Apoyo (no es un criterio): el nucleo es puro, no escribe y no lanza procesos
# ─────────────────────────────────────────────────────────────────────────────
def test_the_core_module_is_pure() -> None:
    """El nucleo no abre ficheros, no lanza procesos ni importa el sistema de ficheros."""
    tree = ast.parse(SOURCE)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called & {"open", "exec", "eval", "compile"}
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not imported & {"os", "subprocess", "pathlib", "shutil", "socket"}


# ─────────────────────────────────────────────────────────────────────────────
# Apoyo (no es un criterio): degenerados que no caben en un `test_aN`
# ─────────────────────────────────────────────────────────────────────────────
def test_degenerate_and_defensive_paths_are_typed() -> None:
    """Los degenerados que no caben en un `test_aN` siguen siendo error tipado (A19)."""
    with pytest.raises(NonFiniteInputError):
        deflated_sharpe_ratio((0.01, float("nan"), 0.02), n_trials=5, sr_variance=0.01)
    with pytest.raises(NonFiniteInputError):
        probability_of_backtest_overfitting(((0.01, 0.02), (float("inf"), 0.03)), blocks=2)
    with pytest.raises(InsufficientObservationsError):
        probability_of_backtest_overfitting((), blocks=2)
    with pytest.raises(InsufficientVariantsError):
        noise_matrix(n_variants=1, n_observations=16)
    # `V[SR]` de un vector constante: 0.0 **exacto** (nada de residuos de coma flotante)
    assert variant_sharpe_variance([0.25, 0.25, 0.25]) == 0.0
    # los empates de rango se reparten la posicion media: dos columnas identicas
    matrix = _matrix(rows=64, columns=4, seed=PROBE_SEED)
    tied = tuple((row[0], row[0], row[1], row[2]) for row in matrix)
    with_ties = probability_of_backtest_overfitting(tied, blocks=8)
    assert with_ties.n_variants == 4
    assert 0.0 <= with_ties.pbo <= 1.0
    assert json.dumps(with_ties.to_payload(), allow_nan=False)
