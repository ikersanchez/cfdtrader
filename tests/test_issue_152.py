"""Tests de #152: el `model.json` declara la alineacion de diseno con la que se entreno.

Cubre los criterios de la issue:

- **C1** — el payload del modelo (`BaselineModel` y `LightGBMModel`) publica
  `design_lag_by_feature`, una entrada por feature.
- **C2** — el valor sale del `DesignFrame` que ajusto el modelo, no de una consulta al catalogo
  en la capa de modelo (`models/*` no importa `features.store`).
- **C3** — el camino diario rechaza un modelo que **no** declara la alineacion (modelo viejo).
- **C4** — rechaza uno que declara otra distinta de la vigente (catalogo redeclarado sin
  re-entrenar), citando la columna que difiere.
- **C5** — con la alineacion vigente el modelo se sirve como siempre.
- **C6** — el rechazo es un error tipado del camino diario (`DeliveryError`), nunca un servicio
  silencioso.
- **C7** — el campo entra en la identidad del modelo (`model_sha256`).
- **C8** — determinismo: dos ajustes del mismo diseno publican el mismo payload.

Es hermetico: no toca el almacen ni el registro (`tests/conftest.py` huella `data/` y `runs/`).
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Final, cast

import polars as pl
import pytest

from cfdtrader.analysis.baseline_report import model_sha256
from cfdtrader.delivery import run_daily
from cfdtrader.features import store as feature_store
from cfdtrader.models.baseline import (
    BASELINE_FEATURES,
    DesignFrame,
    SplitAssignment,
    design_frame,
    fit_baseline,
)
from cfdtrader.models.lightgbm_model import fit_lightgbm

#: La **alineacion vigente** del catalogo restringida al contrato de #24 (`BASELINE_FEATURES`).
CURRENT: Final[dict[str, int]] = {
    name: feature_store.DESIGN_LAG_BY_FEATURE[name] for name in BASELINE_FEATURES
}

#: La columna que el caso "modelo viejo + catalogo nuevo" mueve (su valor se invierte).
FLIPPED: Final[str] = "is_es_roll_session"

#: Entradas del modelo sintetico: cualquier valor sirve, la prediccion no depende de ellas.
_FEATURES: Final[dict[str, float]] = dict.fromkeys(BASELINE_FEATURES, 3.0)

#: Hiperparametros minimos de LightGBM para que la familia se ajuste rapido en la prueba.
_LIGHTGBM_PARAMS: Final[dict[str, object]] = {
    "objective": "binary",
    "num_leaves": 3,
    "min_data_in_leaf": 4,
    "learning_rate": 0.3,
    "n_estimators": 8,
    "num_threads": 1,
    "seed": 7,
    "deterministic": True,
    "verbose": -1,
}


def _synthetic(rows: int = 40) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Features (las 10) y etiquetas deterministicas para un diseno pequeno.

    Cada columna vale ``posicion_en_la_sesion * 10 + posicion_de_la_columna``, igual que en
    #147/#150: asi la fila de diseno de `t` se lee a mano.
    """
    days = [date(2024, 1, 2) + timedelta(days=index) for index in range(rows)]
    data: dict[str, list[object]] = {"session": list(days)}
    for position, name in enumerate(BASELINE_FEATURES):
        data[name] = [float(index * 10 + position) for index in range(rows)]
    labels = pl.DataFrame(
        {"session": days[1:], "ret_long": [0.01 * (index % 3 - 1) for index in range(1, rows)]}
    )
    return pl.DataFrame(data), labels


def _design(*, rows: int = 40, availability: Mapping[str, int] | None = None) -> DesignFrame:
    """Un `DesignFrame` con el catalogo vigente (o con el mapa alternativo que se pida)."""
    features, labels = _synthetic(rows=rows)
    return design_frame(
        features,
        labels=labels,
        availability=feature_store.DESIGN_LAG_BY_FEATURE if availability is None else availability,
    )


def _splits(design: DesignFrame) -> tuple[SplitAssignment, ...]:
    """Un solo fold: el ultimo cuarto de las sesiones de diseno como *test*."""
    size = design.n_sessions
    cut = max(1, size - max(4, size // 4))
    return (SplitAssignment(index=0, train=tuple(range(0, cut)), test=tuple(range(cut, size))),)


# ─────────────────────────────────────────────────────────────────────────────
# C1 - el payload declara la alineacion, en las dos familias
# ─────────────────────────────────────────────────────────────────────────────
def test_c1_the_linear_model_publishes_the_design_alignment() -> None:
    """`to_payload()` publica `design_lag_by_feature`, una entrada por feature (C1)."""
    design = _design()
    model = fit_baseline(design, splits=_splits(design))
    payload = model.to_payload()
    published = cast("Mapping[str, int]", payload["design_lag_by_feature"])
    assert published == dict(design.design_lag_by_feature)
    assert list(published) == list(BASELINE_FEATURES)


def test_c1_the_lightgbm_model_publishes_the_design_alignment() -> None:
    """La familia LightGBM publica el mismo campo, con la misma alineacion (C1)."""
    design = _design(rows=60)
    model = fit_lightgbm(design, splits=_splits(design), hyperparameters=dict(_LIGHTGBM_PARAMS))
    payload = model.to_payload()
    published = cast("Mapping[str, int]", payload["design_lag_by_feature"])
    assert published == dict(design.design_lag_by_feature)


# ─────────────────────────────────────────────────────────────────────────────
# C2 - el valor sale del diseno, no del catalogo
# ─────────────────────────────────────────────────────────────────────────────
def test_c2_the_alignment_comes_from_the_design_not_the_catalog() -> None:
    """Con un diseno construido con **otro** mapa, el payload refleja el diseno (C2).

    Es la forma directa de fijar que el valor publicado es `design.design_lag_by_feature` y no
    una consulta al catalogo vigente dentro de la capa de modelo.
    """
    alternative = dict.fromkeys(BASELINE_FEATURES, 1)
    design = _design(availability=alternative)
    model = fit_baseline(design, splits=_splits(design))
    published = cast("Mapping[str, int]", model.to_payload()["design_lag_by_feature"])
    assert published == alternative
    assert published != CURRENT


def test_c2_the_model_layer_does_not_import_the_catalog() -> None:
    """`models/*` no consulta `features.store`: la alineacion viaja en el `DesignFrame` (C2).

    La comprobacion es estructural (AST), no de comportamiento: si la capa de modelo importara el
    catalogo, el payload podria declarar una alineacion distinta de la del frame que lo ajusto.
    """
    models_dir = Path(run_daily.__file__).resolve().parents[1] / "models"
    for name in ("baseline.py", "lightgbm_model.py"):
        tree = ast.parse((models_dir / name).read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not any(
            module == "cfdtrader.features" or module.startswith("cfdtrader.features.")
            for module in imported
        ), f"{name} importa el catalogo de features: la alineacion tiene que venir del diseno"


# ─────────────────────────────────────────────────────────────────────────────
# C3 / C4 / C5 / C6 - el camino diario verifica antes de puntuar
# ─────────────────────────────────────────────────────────────────────────────
def _document(alignment: Mapping[str, object] | None) -> dict[str, object]:
    """Un documento lineal bien formado; sin `alignment`, un modelo viejo (antes de #152)."""
    fold: dict[str, object] = {
        "index": 0,
        "coefficients": [0.5] * 10,
        "intercept": 0.25,
        "mean": [1.0] * 10,
        "scale": [2.0] * 10,
        "test_positions": [0],
    }
    model: dict[str, object] = {"features": list(BASELINE_FEATURES), "folds": [fold]}
    if alignment is not None:
        model["design_lag_by_feature"] = dict(alignment)
    return {"model": model, "note": ""}


def _write(tmp_path: Path, document: Mapping[str, object], *, name: str) -> Path:
    """Escribe `tmp_path/<name>/model.json` y devuelve su ruta."""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "model.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def test_c5_a_model_with_the_current_alignment_is_served(tmp_path: Path) -> None:
    """Con la alineacion vigente, la prediccion es la de siempre: sin regresion (C5)."""
    path = _write(tmp_path, _document(CURRENT), name="vigente")
    raw, calibrated = run_daily._predictions(  # pyright: ignore[reportPrivateUsage]
        path, _FEATURES
    )
    assert raw == pytest.approx(run_daily.sigmoid(5.25))
    assert calibrated == pytest.approx(run_daily.sigmoid(5.25))


def test_c3_a_model_without_the_alignment_is_rejected(tmp_path: Path) -> None:
    """Un modelo que no declara la alineacion se rechaza: no se puede verificar (C3)."""
    path = _write(tmp_path, _document(None), name="viejo")
    with pytest.raises(run_daily.DesignAlignmentError) as failure:
        run_daily._predictions(path, _FEATURES)  # pyright: ignore[reportPrivateUsage]
    assert "design_lag_by_feature" in str(failure.value)


def test_c4_a_model_with_a_stale_alignment_is_rejected(tmp_path: Path) -> None:
    """Modelo viejo + catalogo nuevo: la columna que difiere se cita y se rechaza (C4)."""
    stale = dict(CURRENT)
    stale[FLIPPED] = 1 - stale[FLIPPED]
    path = _write(tmp_path, _document(stale), name="desalineado")
    with pytest.raises(run_daily.DesignAlignmentError) as failure:
        run_daily._predictions(path, _FEATURES)  # pyright: ignore[reportPrivateUsage]
    assert FLIPPED in str(failure.value)


def test_c4_a_non_integer_or_extra_key_is_rejected(tmp_path: Path) -> None:
    """Un rezago no entero o una columna de mas tampoco son la alineacion vigente (C4)."""
    not_integer: dict[str, object] = dict(CURRENT)
    not_integer["har_forecast"] = 1.5
    with pytest.raises(run_daily.DesignAlignmentError):
        run_daily._predictions(  # pyright: ignore[reportPrivateUsage]
            _write(tmp_path, _document(not_integer), name="no-entero"), _FEATURES
        )
    extra: dict[str, object] = {**CURRENT, "una_columna_de_mas": 0}
    with pytest.raises(run_daily.DesignAlignmentError):
        run_daily._predictions(  # pyright: ignore[reportPrivateUsage]
            _write(tmp_path, _document(extra), name="de-mas"), _FEATURES
        )


def test_c6_the_rejection_is_a_typed_delivery_error() -> None:
    """El rechazo es un `DeliveryError` publicado, no un `KeyError` ni un silencio (C6)."""
    assert issubclass(run_daily.DesignAlignmentError, run_daily.DeliveryError)
    assert "DesignAlignmentError" in run_daily.__all__


# ─────────────────────────────────────────────────────────────────────────────
# C7 / C8 - el campo entra en la identidad y es determinista
# ─────────────────────────────────────────────────────────────────────────────
def test_c7_the_field_enters_the_model_identity() -> None:
    """Cambiar la alineacion cambia el `model_sha256`: el payload es su contenido (C7)."""
    design = _design()
    model = fit_baseline(design, splits=_splits(design))
    flipped = replace(
        model,
        design_lag_by_feature=tuple((name, 1 - lag) for name, lag in model.design_lag_by_feature),
    )
    assert model_sha256(model) != model_sha256(flipped)
    assert "design_lag_by_feature" in model.to_payload()


def test_c8_two_fits_of_the_same_design_publish_the_same_payload() -> None:
    """Dos ajustes del mismo diseno dan el mismo payload (C8)."""
    design = _design()
    first = fit_baseline(design, splits=_splits(design)).to_payload()
    second = fit_baseline(design, splits=_splits(design)).to_payload()
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False
    )
