"""Tests del modelo espejo de la direccion corta (#78): objetivo, identidad y ajuste.

Todo se mide con **frames sinteticos deterministas** (nunca con el ``data/`` del repositorio).
La maquinaria de ajuste es la de #24 **importada**: lo que aqui se comprueba es que el espejo
cambia **solo** el objetivo (``ret_short``) y declara su propia identidad, sin duplicar codigo.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Final, cast

import polars as pl
import pytest

from cfdtrader.analysis import feature_frame
from cfdtrader.features import store as feature_store
from cfdtrader.models import baseline, mirror
from cfdtrader.models.baseline import (
    BASELINE_FEATURES,
    DECISION_THRESHOLD,
    BaselineError,
    SplitAssignment,
    design_frame,
    probabilities,
)
from cfdtrader.models.mirror import (
    MIRROR_FEATURES,
    MIRROR_TARGET_COLUMN,
    MIRROR_TARGET_RULE,
    MIRROR_VARIANT_ID,
    MirrorError,
    fit_mirror,
    mirror_design_frame,
    mirror_identity,
    short_signal,
)

MIRROR_PATH: Final[Path] = Path(str(mirror.__file__))
SOURCE: Final[str] = MIRROR_PATH.read_text(encoding="utf-8")
TREE: Final[ast.Module] = ast.parse(SOURCE)
AVAILABILITY: Final[Mapping[str, int]] = feature_store.DESIGN_LAG_BY_FEATURE

#: Modulos que ``models/mirror.py`` **no** puede importar (A1): el espejo es puro como #24.
FORBIDDEN_PREFIXES: Final[tuple[str, ...]] = (
    "duckdb",
    "cfdtrader.data",
    "cfdtrader.analysis",
    "cfdtrader.backtest",
    "sklearn.model_selection",
)


def _business_days(start: date, count: int) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _noise(seed: str, index: int) -> float:
    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / (2**63) - 1.0


def _features_frame(*, rows: int = 240, seed: str = "a78") -> pl.DataFrame:
    sessions = _business_days(date(2024, 1, 2), rows)
    data: dict[str, object] = {"session": sessions}
    for position, name in enumerate(BASELINE_FEATURES):
        data[name] = [_noise(f"{seed}:{name}", index) + position / 100.0 for index in range(rows)]
    data["is_es_roll_session"] = [float(index % 17 == 0) for index in range(rows)]
    return pl.DataFrame(data)


def _labels_frame(features: pl.DataFrame) -> pl.DataFrame:
    """Las **dos** direcciones: la corta es el signo **contrario** de la larga (espejo exacto)."""
    trimmed = features.slice(1)
    signal = [
        sum(cast("float", row[name]) for name in BASELINE_FEATURES[:4])
        for row in trimmed.iter_rows(named=True)
    ]
    return pl.DataFrame(
        {
            "session": trimmed["session"],
            "ret_long": [0.01 * value for value in signal],
            "ret_short": [-0.01 * value for value in signal],
        }
    )


def _assignments(rows: int, *, folds: int = 2, test_size: int = 40) -> tuple[SplitAssignment, ...]:
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


def _mirror_design(*, rows: int = 240) -> baseline.DesignFrame:
    features = _features_frame(rows=rows)
    return mirror_design_frame(features, labels=_labels_frame(features), availability=AVAILABILITY)


def _long_design(*, rows: int = 240) -> baseline.DesignFrame:
    features = _features_frame(rows=rows)
    return design_frame(features, labels=_labels_frame(features), availability=AVAILABILITY)


def _imported_modules(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_c1_mirror_is_a_pure_module() -> None:
    """A1: ``models/mirror.py`` reutiliza #24 y no cruza ninguna frontera de capa."""
    forbidden = sorted(
        name
        for name in _imported_modules(TREE)
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in FORBIDDEN_PREFIXES)
    )
    assert not forbidden, f"el espejo importa modulos prohibidos: {forbidden}"
    assert "cfdtrader.models.baseline" in _imported_modules(TREE)


def test_c2_mirror_identity_declares_its_own_variant_and_target() -> None:
    """A2: la identidad del espejo es **propia** y declara el objetivo corto."""
    identity = mirror_identity()
    assert identity["variant_id"] == MIRROR_VARIANT_ID
    assert identity["variant_id"] != "baseline_logit_elasticnet_v1"
    assert identity["direction"] == "short"
    assert identity["target_column"] == MIRROR_TARGET_COLUMN
    assert identity["target_rule"] == MIRROR_TARGET_RULE
    assert identity["features"] == list(MIRROR_FEATURES)
    assert identity["features"] == list(BASELINE_FEATURES)
    assert identity["decision_threshold"] == DECISION_THRESHOLD
    # La identidad publica sus fronteras con su issue, como el resto del proyecto.
    assert {item["issue"] for item in mirror.MIRROR_DOES_NOT_DO} == {"#16", "#26", "#27"}


def test_c3_mirror_design_frame_targets_ret_short_and_flips_the_label() -> None:
    """A3: el frame del espejo declara ``ret_short`` y ``y = 1{ret_short > 0}`` fila a fila."""
    design = _mirror_design()
    assert design.target_column == "ret_short"
    assert design.target_rule == "ret_short > 0"
    assert design.features == MIRROR_FEATURES
    expected = [int(value > 0) for value in design.frame.get_column("ret_short").to_list()]
    assert design.frame.get_column("y").to_list() == expected


def test_c4_the_long_design_still_defaults_to_ret_long() -> None:
    """A3: sin ``target_column`` el frame de #24 no cambia (compatibilidad)."""
    design = _long_design()
    assert design.target_column == "ret_long"
    assert design.target_rule == "ret_long > 0"
    expected = [int(value > 0) for value in design.frame.get_column("ret_long").to_list()]
    assert design.frame.get_column("y").to_list() == expected


def test_c5_fit_mirror_rejects_the_long_design() -> None:
    """A4: el espejo **solo** acepta el objetivo corto; el frame de #24 es error tipado."""
    design = _long_design()
    with pytest.raises(MirrorError):
        fit_mirror(design, splits=_assignments(design.n_sessions))


def test_c6_fit_mirror_fits_the_short_objective_deterministically() -> None:
    """A5: el ajuste reutiliza #24 y es determinista (misma semilla, mismo payload)."""
    design = _mirror_design()
    splits = _assignments(design.n_sessions)
    first = fit_mirror(design, splits=splits)
    second = fit_mirror(design, splits=splits)
    assert first.to_payload() == second.to_payload()
    assert first.features == MIRROR_FEATURES
    assert first.design_lag_by_feature == design.design_lag_by_feature
    values = [value for value in probabilities(first, design.frame) if value is not None]
    assert values, "el espejo no predijo ninguna sesion de test"
    assert all(0.0 <= value <= 1.0 for value in values)


def test_c7_short_signal_is_the_declared_threshold() -> None:
    """A6: la decision del espejo es la **misma** regla declarada que la de #24."""
    assert short_signal(DECISION_THRESHOLD) is True
    assert short_signal(DECISION_THRESHOLD - 1e-9) is False
    with pytest.raises(BaselineError):
        short_signal(1.5)


def test_c8_labels_loader_declares_ret_short() -> None:
    """A7: el espejo tiene su cargador de etiquetas; la ruta larga de #24 no cambia."""
    assert "ret_short" in feature_frame.SHORT_LABELS_COLUMNS
    assert "ret_long" not in feature_frame.SHORT_LABELS_COLUMNS
    assert "load_short_labels" in feature_frame.__all__
    # La ruta larga sigue declarando **su** conjunto, sin `ret_short`.
    assert "ret_short" not in feature_frame.LABELS_COLUMNS
    source = Path(str(feature_frame.__file__)).read_text(encoding="utf-8")
    assert "SELECT session, ret_short FROM derived.labels" in source
