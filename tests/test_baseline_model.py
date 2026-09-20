"""Tests del modelo baseline puro (#24): fronteras, features, ajuste y decision.

Un test por criterio (``test_a1_...``, ``test_a4_...``, ``test_a5_...``, ``test_a7_...``,
``test_a8_...``, ``test_a11_...``); los de A2/A3 viven en ``test_feature_frame.py`` y los de
A6/A9/A10/A12/A13/A14/A15 en ``test_baseline_report.py``.

Todo se mide con **frames sinteticos deterministas** (nunca con el ``data/`` del repositorio):
el ruido es ``sha256(f"{seed}:{index}")`` y no un generador congruencial, porque dos semillas
distintas de un LCG producen la misma secuencia desplazada una constante.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Final, cast

import polars as pl
import pytest

from cfdtrader.analysis.baseline_report import model_sha256
from cfdtrader.backtest import engine
from cfdtrader.features import store as feature_store
from cfdtrader.models import baseline
from cfdtrader.models.baseline import (
    BASELINE_FEATURES,
    DECISION_THRESHOLD,
    DESIGN_LAG_SESSIONS,
    HYPERPARAMETERS,
    SEED,
    BaselineError,
    InvalidDesignFrameError,
    SplitAssignment,
    UnknownFeatureError,
    design_frame,
    fit_baseline,
    long_signal,
    probabilities,
)

MODULE_PATH: Final[Path] = Path(str(baseline.__file__))
SOURCE: Final[str] = MODULE_PATH.read_text(encoding="utf-8")
TREE: Final[ast.Module] = ast.parse(SOURCE)
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

#: Modulos que ``models/baseline.py`` **no** puede importar (A1), por AST y no por texto.
FORBIDDEN_ROOTS: Final[tuple[str, ...]] = (
    "duckdb",
    "cfdtrader.data",
    "cfdtrader.analysis",
    "cfdtrader.backtest",
    "sklearn.model_selection",
)

#: Modulos nuevos de #24 (A1).
NEW_MODULES: Final[tuple[Path, ...]] = (
    REPO_ROOT / "src" / "cfdtrader" / "models" / "baseline.py",
    REPO_ROOT / "src" / "cfdtrader" / "analysis" / "feature_frame.py",
    REPO_ROOT / "src" / "cfdtrader" / "analysis" / "baseline_report.py",
)


def _business_days(start: date, count: int) -> list[date]:
    """``count`` dias laborables consecutivos desde ``start``."""
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _noise(seed: str, index: int) -> float:
    """Ruido determinista en ``[-1, 1]`` a partir de ``sha256``: sin LCG ni semillas cruzadas."""
    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / (2**63) - 1.0


def _features_frame(*, rows: int = 240, seed: str = "a24") -> pl.DataFrame:
    """Frame de features: ``session`` y las 10 declaradas, con ``is_es_roll_session`` 0/1."""
    sessions = _business_days(date(2024, 1, 2), rows)
    data: dict[str, object] = {"session": sessions}
    for position, name in enumerate(BASELINE_FEATURES):
        data[name] = [_noise(f"{seed}:{name}", index) + position / 100.0 for index in range(rows)]
    data["is_es_roll_session"] = [float(index % 17 == 0) for index in range(rows)]
    return pl.DataFrame(data)


def _labels_frame(features: pl.DataFrame) -> pl.DataFrame:
    """Etiquetas deterministas: el signo de la suma de cuatro features, sin azar.

    La **primera** sesion del frame queda sin etiqueta a proposito: su fila de diseno no existe
    (no tiene sesion anterior) y el frame de diseno no puede llevar nulos. En el diario real
    ocurre lo mismo por construccion: las etiquetas de #10 empiezan once anos despues de la
    primera fila.
    """
    trimmed = features.slice(1)
    signal = [
        sum(cast("float", row[name]) for name in BASELINE_FEATURES[:4])
        for row in trimmed.iter_rows(named=True)
    ]
    return pl.DataFrame(
        {"session": trimmed["session"], "ret_long": [0.01 * value for value in signal]}
    )


def _design(*, rows: int = 240, seed: str = "a24") -> baseline.DesignFrame:
    """La matriz de diseno sintetica completa (features + etiquetas), con el corrimiento."""
    features = _features_frame(rows=rows, seed=seed)
    return design_frame(features, labels=_labels_frame(features))


def _assignments(rows: int, *, folds: int = 2, test_size: int = 40) -> tuple[SplitAssignment, ...]:
    """Folds contiguos al final del frame, como los de un plan *walk-forward*."""
    out: list[SplitAssignment] = []
    for index in range(folds):
        stop = rows - (folds - index - 1) * test_size
        start = stop - test_size
        out.append(
            SplitAssignment(
                index=index, train=tuple(range(0, start)), test=tuple(range(start, stop))
            )
        )
    return tuple(out)


def _evaluate(frame: pl.DataFrame, model: baseline.BaselineModel) -> list[float]:
    """Las probabilidades no nulas del frame, para comparar corridas."""
    return [value for value in probabilities(model, frame) if value is not None]


def _replaced(frame: baseline.DesignFrame, mutated: pl.DataFrame) -> baseline.DesignFrame:
    return baseline.DesignFrame(
        frame=mutated,
        n_sessions=frame.n_sessions,
        n_labels=frame.n_labels,
        n_shifted_rows=frame.n_shifted_rows,
        n_nulls_in_features=frame.n_nulls_in_features,
        design_lag_sessions=frame.design_lag_sessions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# A1 - modulos y frontera
# ─────────────────────────────────────────────────────────────────────────────
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


def test_a1_modules_exist_and_baseline_respects_its_layer() -> None:
    """A1: los tres modulos existen y ``models/baseline.py`` no cruza ninguna frontera."""
    for path in NEW_MODULES:
        assert path.is_file(), f"falta el modulo {path}"

    imported = _imported_modules(TREE)
    forbidden = sorted(
        name
        for name in imported
        if any(name == root or name.startswith(f"{root}.") for root in FORBIDDEN_ROOTS)
    )
    assert not forbidden, (
        "`models/baseline.py` importa modulos prohibidos por A1: "
        f"{forbidden}. El canonicamente hasheable, el registro y el motor viven fuera"
    )


def test_a1_only_the_feature_adapter_calls_store_sql() -> None:
    """A1: de los tres modulos nuevos, **solo** ``analysis/feature_frame.py`` llama a ``sql``.

    La comprobacion es por AST (una llamada a un atributo llamado ``sql``), nunca por
    coincidencia de texto: un docstring que mencione `store.sql()` no demuestra nada.
    """
    callers: list[str] = []
    for path in NEW_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "sql"
            for node in ast.walk(tree)
        ):
            callers.append(path.name)
    assert callers == ["feature_frame.py"], (
        "solo el adaptador de features puede llamar a `store.sql` (A1); los llamantes medidos "
        f"son {callers}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# A4 - seleccion de features
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_the_ten_features_are_declared_and_reach_the_configuration() -> None:
    """A4: diez features, en el catalogo, las cinco familias, sin las tres excluidas."""
    assert len(BASELINE_FEATURES) == 10
    assert len(set(BASELINE_FEATURES)) == 10
    assert all(name in feature_store.ALL_FEATURE_COLUMNS for name in BASELINE_FEATURES)
    assert not {"sessions_to_opex", "pendiente_2s10s", "pendiente_2s10s_chg_5"}.intersection(
        BASELINE_FEATURES
    )

    families: dict[str, str] = {}
    for feature_set, catalog in feature_store.CATALOG_BY_FEATURE_SET.items():
        for entry in catalog:
            families.setdefault(entry.name, feature_set)
    represented = sorted({families[name] for name in BASELINE_FEATURES})
    assert represented == sorted(feature_store.CATALOG_BY_FEATURE_SET), (
        f"las cinco familias tienen que estar representadas y ninguna puede faltar: {represented}"
    )

    design = _design()
    nulls = sum(design.frame.get_column(name).null_count() for name in BASELINE_FEATURES)
    assert nulls == 0
    model = fit_baseline(design, splits=_assignments(design.n_sessions))
    assert model.features == BASELINE_FEATURES
    assert all(fold.n_test == 40 for fold in model.folds)
    assert [fold.index for fold in model.folds] == [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# A5 - features frente al tamano muestral
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_positives_per_feature_is_above_ten() -> None:
    """A5: la aritmetica de §9 se cumple en las proporciones declaradas."""
    design = _design(rows=600)
    n_features = len(BASELINE_FEATURES)
    positives_per_feature = design.positives / n_features
    assert design.n_sessions == 599
    assert n_features == 10
    assert n_features <= 15
    assert positives_per_feature >= 10
    assert 0 < design.positives < design.n_sessions


# ─────────────────────────────────────────────────────────────────────────────
# A7 - hiperparametros sin leakage
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_hyperparameters_are_declared_and_the_scaling_is_train_only() -> None:
    """A7: las constantes viajan; el escalado se ajusta **solo** con el train de cada fold."""
    assert HYPERPARAMETERS["penalty"] == "elasticnet"
    assert HYPERPARAMETERS["solver"] == "saga"
    assert HYPERPARAMETERS["l1_ratio"] == 0.5
    assert HYPERPARAMETERS["C"] == 1.0
    assert HYPERPARAMETERS["random_state"] == SEED

    frame = _design()
    splits = _assignments(frame.n_sessions)
    model = fit_baseline(frame, splits=splits)
    train = list(splits[0].train)
    matrix = frame.frame.select(list(BASELINE_FEATURES)).cast(pl.Float64).to_numpy()
    assert model.folds[0].mean == pytest.approx(tuple(float(v) for v in matrix[train].mean(axis=0)))
    assert model.folds[0].scale == pytest.approx(tuple(float(v) for v in matrix[train].std(axis=0)))
    assert model.hyperparameters == HYPERPARAMETERS

    # Mutar las features **del test** y reentrenar: ni los coeficientes ni el escalado se mueven.
    mutated = frame.frame.with_columns(
        [
            pl.when(pl.int_range(pl.len()) >= splits[-1].test[0])
            .then(pl.col(name) * 3.0 + 1.0)
            .otherwise(pl.col(name))
            .alias(name)
            for name in BASELINE_FEATURES
        ]
    )
    assert not mutated.select(list(BASELINE_FEATURES)).equals(
        frame.frame.select(list(BASELINE_FEATURES))
    )
    refit = fit_baseline(_replaced(frame, mutated), splits=splits)
    assert [fold.coefficients for fold in refit.folds] == [
        fold.coefficients for fold in model.folds
    ]
    assert [fold.mean for fold in refit.folds] == [fold.mean for fold in model.folds]
    assert _evaluate(mutated, refit) != _evaluate(frame.frame, model)

    # Control positivo: mutar el **train** si mueve los coeficientes.
    scaled = frame.frame.with_columns(
        [(pl.col(name) * 3.0 - 1.0).alias(name) for name in BASELINE_FEATURES]
    )
    changed = fit_baseline(_replaced(frame, scaled), splits=splits)
    assert [fold.coefficients for fold in changed.folds] != [
        fold.coefficients for fold in model.folds
    ]


# ─────────────────────────────────────────────────────────────────────────────
# A8 - determinismo medido
# ─────────────────────────────────────────────────────────────────────────────
CHILD: Final[str] = textwrap.dedent(
    """
    import hashlib, json
    from datetime import date, timedelta
    import polars as pl
    from cfdtrader.analysis.experiment_log import ExperimentConfig, run_sha256
    from cfdtrader.backtest.engine import canonical_text
    from cfdtrader.models.baseline import (
        BASELINE_FEATURES, HYPERPARAMETERS, SEED, SplitAssignment, design_frame,
        fit_baseline, probabilities,
    )

    def noise(seed, index):
        digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / (2**63) - 1.0

    rows = 240
    days = []
    current = date(2024, 1, 2)
    while len(days) < rows:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    data = {"session": days}
    for position, name in enumerate(BASELINE_FEATURES):
        data[name] = [noise(f"a24:{name}", i) + position / 100.0 for i in range(rows)]
    labels = pl.DataFrame(
        {"session": days[1:], "ret_long": [0.01 * (i % 5 - 2) for i in range(1, rows)]}
    )
    frame = design_frame(pl.DataFrame(data), labels=labels)
    splits = (SplitAssignment(index=0, train=tuple(range(0, 160)), test=tuple(range(160, 200))),)
    model = fit_baseline(frame, splits=splits)
    config = ExperimentConfig(
        variant_id="synthetic_a8",
        features=BASELINE_FEATURES,
        hyperparameters=dict(HYPERPARAMETERS),
        seed=SEED,
        series_id="^GSPC",
        window={"rows": rows, "synthetic": True},
    )
    print(json.dumps({
        "run_sha256": run_sha256(config),
        "model_sha256": hashlib.sha256(canonical_text(model.to_payload()).encode()).hexdigest(),
        "coefficients": [repr(value) for value in model.folds[0].coefficients],
        "predictions": [repr(v) for v in probabilities(model, frame.frame) if v is not None],
    }))
    """
)


def _child_hashes(hashseed: str) -> dict[str, object]:
    """Corre el hijo con esa semilla de ``hash`` y devuelve su JSON."""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", CHILD],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": hashseed},
        cwd=str(REPO_ROOT),
    )
    return cast("dict[str, object]", json.loads(completed.stdout))


def test_a8_two_processes_give_the_same_hashes_and_predictions() -> None:
    """A8: dos procesos con distinta ``PYTHONHASHSEED`` dan el mismo modelo y las mismas ``p``.

    Se miden los tres objetos que A8 nombra: el ``run_sha256`` de la configuracion de #16, el
    ``model_sha256`` del payload del modelo y las predicciones. La tolerancia medida es ``0``
    (identico bit a bit), muy por debajo del ``1e-12`` que exige el criterio.
    """
    first = _child_hashes("0")
    second = _child_hashes("1")
    assert first["run_sha256"] == second["run_sha256"]
    assert first["model_sha256"] == second["model_sha256"]
    assert first["coefficients"] == second["coefficients"]
    deltas = [
        abs(float(left) - float(right))
        for left, right in zip(
            cast("list[str]", first["predictions"]),
            cast("list[str]", second["predictions"]),
            strict=True,
        )
    ]
    assert len(deltas) == 40
    assert max(deltas) <= 1e-12, f"las predicciones no son estables entre procesos: {max(deltas)}"

    # El mismo objeto, en este proceso: el digest se recalcula y coincide con el del hijo.
    frame = _design()
    model = fit_baseline(frame, splits=_assignments(frame.n_sessions))
    digest = hashlib.sha256(engine.canonical_text(model.to_payload()).encode("utf-8")).hexdigest()
    assert model_sha256(model) == digest


# ─────────────────────────────────────────────────────────────────────────────
# A11 - decision declarada
# ─────────────────────────────────────────────────────────────────────────────
def _view(session: date, probability: float | None) -> engine.SessionView:
    """Una vista con la probabilidad en su carga opaca, como la deja el adaptador."""
    return engine.SessionView(session=session, open_px=100.0, gap_px=None, context=probability)


def test_a11_the_decider_only_reads_the_view_and_uses_the_declared_threshold() -> None:
    """A11: ``p >= 0,5`` ⇒ ``LONG`` con esa ``p``; por debajo, ``NOTHING``."""
    from cfdtrader.analysis import baseline_report

    assert DECISION_THRESHOLD == 0.5
    assert long_signal(0.5) is True
    assert long_signal(0.4999) is False

    decide = baseline_report._decider(0)  # pyright: ignore[reportPrivateUsage]
    above = decide(_view(date(2025, 1, 2), 0.73))
    assert above.direction == engine.Direction.LONG
    assert above.probability == 0.73
    assert above.notional_usd is not None
    assert above.notional_usd > 0
    assert above.stop_px is None
    assert above.target_px is None

    below = decide(_view(date(2025, 1, 2), 0.4999))
    assert below.direction == engine.Direction.NOTHING
    assert below.probability == 0.4999
    assert below.notional_usd is None

    exact = decide(_view(date(2025, 1, 2), 0.5))
    assert exact.direction == engine.Direction.LONG

    with pytest.raises(engine.DecisionError):
        decide(_view(date(2025, 1, 2), None))
    with pytest.raises(engine.DecisionError):
        decide(_view(date(2025, 1, 2), 1.2))
    with pytest.raises(BaselineError):
        long_signal(float("nan"))
    with pytest.raises(BaselineError):
        long_signal(-0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Bordes del modulo puro (apoyo, no un criterio)
# ─────────────────────────────────────────────────────────────────────────────
def test_design_frame_shifts_one_diary_session_and_counts_what_it_loses() -> None:
    """El corrimiento es el de A2 y lo que no tiene sesion anterior se **cuenta**."""
    features = _features_frame(rows=6)
    labels = _labels_frame(features)
    complete = design_frame(features, labels=labels)
    assert complete.design_lag_sessions == DESIGN_LAG_SESSIONS == 1
    assert complete.n_sessions == 5
    assert complete.n_shifted_rows == 0
    assert complete.sessions == tuple(cast("list[object]", labels["session"].to_list()))

    partial = design_frame(features.slice(0, 4), labels=labels)
    assert partial.n_sessions == 3
    assert partial.n_shifted_rows == 2

    with pytest.raises(UnknownFeatureError):
        design_frame(features.select("session", *BASELINE_FEATURES[:3]), labels=labels)
    with pytest.raises(InvalidDesignFrameError):
        design_frame(cast("pl.DataFrame", "no es un frame"), labels=labels)
    with pytest.raises(InvalidDesignFrameError):
        design_frame(features, labels=cast("pl.DataFrame", "no es un frame"))


def test_fit_rejects_frames_and_splits_it_cannot_use() -> None:
    """Los bordes de ``fit_baseline`` son errores tipados, nunca un ajuste a medias."""
    frame = _design()
    rows = frame.n_sessions
    with pytest.raises(InvalidDesignFrameError):
        fit_baseline(cast("baseline.DesignFrame", "no es un frame"), splits=_assignments(rows))
    with pytest.raises(InvalidDesignFrameError):
        fit_baseline(frame, splits=())
    with pytest.raises(InvalidDesignFrameError):
        fit_baseline(frame, splits=(SplitAssignment(index=0, train=(), test=(210,)),))
    with pytest.raises(InvalidDesignFrameError):
        fit_baseline(frame, splits=(SplitAssignment(index=0, train=(0, 1, 2), test=(999,)),))
    with pytest.raises(InvalidDesignFrameError):
        fit_baseline(frame, splits=(SplitAssignment(index=0, train=(1, 2, 3), test=(2, 3, 4)),))
    with pytest.raises(InvalidDesignFrameError):
        fit_baseline(_replaced(frame, frame.frame.drop("y")), splits=_assignments(rows))


def test_probabilities_are_none_outside_every_test() -> None:
    """Fuera de todo *test* no hay prediccion honesta: ``None``, nunca ``0``."""
    frame = _design()
    splits = _assignments(frame.n_sessions)
    model = fit_baseline(frame, splits=splits)
    values = probabilities(model, frame.frame)
    assert len(values) == frame.n_sessions
    covered = {position for split in splits for position in split.test}
    assert all(value is None for position, value in enumerate(values) if position not in covered)
    assert all(value is not None for position, value in enumerate(values) if position in covered)
    assert all(value is None or 0.0 <= value <= 1.0 for value in values)
    assert model.fold_for(0) is None
    assert model.fold_for(splits[0].test[0]) is not None
    with pytest.raises(UnknownFeatureError):
        probabilities(model, frame.frame.select("session", *BASELINE_FEATURES[:4]))


def test_a_design_frame_carries_the_session_the_features_came_from() -> None:
    """La sesion de origen de cada fila de diseno viaja en el frame (auditoria de A2)."""
    features = _features_frame(rows=6)
    frame = design_frame(features, labels=_labels_frame(features))
    origin = frame.frame[baseline.DESIGN_SESSION_COLUMN].to_list()
    assert origin == cast("list[object]", features["session"].to_list())[:-1]


def test_model_payload_is_json_and_reproduces_its_own_identity() -> None:
    """El payload del modelo (A12) es JSON puro y su digest no depende de la ruta."""
    frame = _design()
    model = fit_baseline(frame, splits=_assignments(frame.n_sessions))
    payload = model.to_payload()
    text = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert json.loads(text) == payload
    assert payload["features"] == list(BASELINE_FEATURES)
    assert payload["design_lag_sessions"] == 1
    assert payload["decision_threshold"] == 0.5
    folds = cast("list[Mapping[str, object]]", payload["folds"])
    assert len(folds) == 2
    for fold in folds:
        assert set(fold) >= {"coefficients", "intercept", "mean", "scale", "test_positions"}
    assert (
        model_sha256(model)
        == hashlib.sha256(engine.canonical_text(payload).encode("utf-8")).hexdigest()
    )
    assert _evaluate(frame.frame, model) != []


def test_deciders_are_built_per_fold_and_the_reason_is_declared() -> None:
    """El motivo del decider es explicito y distinguible entre operar y no operar."""
    from cfdtrader.analysis import baseline_report

    portfolio: Sequence[engine.DecisionFn] = tuple(
        baseline_report._decider(index)  # pyright: ignore[reportPrivateUsage]
        for index in range(3)
    )
    assert len(portfolio) == 3
    assert portfolio[0](_view(date(2025, 1, 2), 0.9)).reason == baseline_report.LONG_REASON
    assert portfolio[1](_view(date(2025, 1, 2), 0.1)).reason == baseline_report.NO_TRADE_REASON
