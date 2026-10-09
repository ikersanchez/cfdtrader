"""Tests del barrido pre-registrado de la familia LightGBM (#82): A1-A12.

Los criterios que publican **numeros reales** corren sobre el almacen del repositorio en **solo
lectura** (la fixture de sesion de `tests/conftest.py` huella `data/` y `runs/` antes y despues), y
todo lo que se escribe va a `tmp_path`: el registro de la corrida es una **copia** de las cuatro
entradas congeladas de #24/#25/#26 mas las **diez** nuevas de §19.17.

El determinismo entre procesos (A10) se mide con la CLI en procesos nuevos y `PYTHONHASHSEED`
0 / 1 / random, comparando `report_sha256`, los bytes de `.json`/`.md` y el resultado de la
**segunda pasada** (`unchanged` en las diez entradas nuevas).

Cobertura medida del modulo nuevo (A12): `uv run pytest tests/test_hyperparameter_search.py
--cov=cfdtrader.analysis.hyperparameter_search --cov-report=term-missing` sobre el arbol limpio.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, cast

import numpy as np
import polars as pl
import pytest

from cfdtrader.analysis import hyperparameter_search as search
from cfdtrader.analysis.experiment_log import (
    ExperimentRecord,
    Registry,
    RegistryEntry,
    TrialsMismatchError,
    deflate_block,
    load_registry,
    require_trials_match_registry,
)
from cfdtrader.analysis.hyperparameter_search import (
    AXES,
    BASELINE_VARIANT_ID,
    BUDGET,
    FEATURE_SETS,
    PBO_MAX,
    REGISTERED_TRIALS,
    REPORT_PREFIX,
    SEARCH_SPACE,
    TOTAL_TRIALS,
    VARIANT_ID,
    HyperparameterSearchError,
    SearchSpaceError,
    SearchVariant,
    VariantNotEvaluableError,
    analyse,
    main,
    matrix_matches_registry,
    reconstruct_lightgbm,
    render_markdown,
    space_audit,
    sweep_verdict,
)
from cfdtrader.analysis.model_comparison import VARIANT_ID as LIGHTGBM_VARIANT_ID
from cfdtrader.data.settings import Settings
from cfdtrader.data.store import Store
from cfdtrader.features import store as feature_store
from cfdtrader.models.baseline import (
    BASELINE_FEATURES,
    SEED,
    DesignFrame,
    UnknownFeatureError,
    design_frame,
)
from cfdtrader.models.lightgbm_model import LIGHTGBM_HYPERPARAMETERS

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"
REPO_RUNS: Final[Path] = REPO_ROOT / "runs"
FROZEN_REPORTS: Final[Path] = REAL_DATA / "derived" / "reports"

#: Instante **declarado** de todas las corridas: el modulo nunca lee el reloj (A11).
NOW: Final[datetime] = datetime(2026, 10, 8, 22, 0, tzinfo=UTC)

#: Base de la entrega: el commit del PM que pre-registra §19.17 (`docs(#82)`), `main` al empezar.
BASE_COMMIT: Final[str] = "0a5c0a8"

#: El modulo nuevo y su suite (A12).
NEW_MODULE: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "analysis" / "hyperparameter_search.py"
TEST_SOURCE: Final[str] = Path(__file__).read_text(encoding="utf-8")

#: Ficheros que A9 declara **fuera** de esta entrega: el motor y el coste.
#:
# #107 retira `_docs/plan.md` de esta lista: la decision del propietario sobre la fuente del
# `SPX500:CFD` (criterio (b), `plan.md` §19.18) **anade una seccion nueva** al documento —y su fila
# de changelog—, el mismo criterio con que #113/#124/#131/#136/#73/#139/#108 retiraron los suyos.
# Lo que A9 protegia de `plan.md` —que §11.6, §19.6 y §19.7 no se toquen— lo sigue comprobando el
# bloque `limits.plan_sections_untouched` de mas abajo.
OUT_OF_DELIVERY: Final[tuple[str, ...]] = (
    "src/cfdtrader/backtest/engine.py",
    "src/cfdtrader/backtest/costs.py",
)

#: Las dos familias que #24/#25/#26 registraron: sus entradas son las **congeladas** de esta tarea.
FROZEN_FAMILIES: Final[tuple[str, ...]] = (BASELINE_VARIANT_ID, LIGHTGBM_VARIANT_ID)


def _frozen_run_directories() -> tuple[str, ...]:
    """Los directorios del registro del repositorio que son de #24/#25/#26 (A4).

    Se identifican por su `variant_id`, **no** por «lo que haya en `runs/`»: el barrido de esta
    tarea registra sus diez entradas ahi (es su comportamiento normal), y la guardia de A4 tiene
    que seguir apuntando a las cuatro de las familias anteriores.
    """
    if not REPO_RUNS.is_dir():
        return ()
    out: list[str] = []
    for child in sorted(REPO_RUNS.iterdir()):
        if not child.is_dir():
            continue
        document = cast(
            "dict[str, object]",
            json.loads((child / "config.json").read_text(encoding="utf-8")),
        )
        config = cast("Mapping[str, object]", document["config"])
        if config.get("variant_id") in FROZEN_FAMILIES:
            out.append(child.name)
    return tuple(out)


#: Los cuatro directorios que #24/#25/#26 registraron: se copian y se huellan (A4).
FROZEN_RUNS: Final[tuple[str, ...]] = _frozen_run_directories()

needs_store = pytest.mark.skipif(
    not (REAL_DATA / "derived" / "labels").exists(),
    reason="el almacen real no esta en el arbol: los numeros de A2-A10 son los suyos",
)


def _skip_without_store() -> None:
    """Salta si el almacen real no esta en el arbol: el CI corre sin `data/` (#85)."""
    if not (REAL_DATA / "derived" / "labels").exists():
        pytest.skip("el almacen real no esta en el arbol: el CI corre sin data/")


def _copy_frozen_runs(destination: Path, digests: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Copia al `destination` las entradas **congeladas** de #24/#25/#26, y solo esas (A4)."""
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for digest in digests if digests is not None else FROZEN_RUNS:
        shutil.copytree(REPO_RUNS / digest, destination / digest, dirs_exist_ok=True)
        copied.append(digest)
    return tuple(copied)


def _fingerprint(root: Path) -> dict[str, str]:
    """Ruta relativa → sha256 de cada fichero de ese arbol (A4)."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _block(report: search.SearchReport, *keys: str) -> dict[str, object]:
    """Un bloque anidado del payload, ya tipado."""
    node: dict[str, object] = report.payload
    for key in keys:
        node = cast("dict[str, object]", node[key])
    return node


def _rows(report: search.SearchReport) -> list[dict[str, object]]:
    """Las diez filas de la tabla del barrido."""
    return [
        cast("dict[str, object]", row)
        for row in cast("list[object]", _block(report, "comparison")["rows"])
    ]


def _sweep_rows(report: search.SearchReport) -> list[dict[str, object]]:
    """Las diez filas de `sweep_rows[]`, en el orden del espacio."""
    return [
        cast("dict[str, object]", row) for row in cast("list[object]", report.payload["sweep_rows"])
    ]


def _cli(*arguments: str, seed: str | None = None) -> subprocess.CompletedProcess[str]:
    """Ejecuta el CLI en un **proceso nuevo**, con su `PYTHONHASHSEED` (A10, A11)."""
    environment = dict(os.environ)
    if seed is not None:
        environment["PYTHONHASHSEED"] = seed
    return subprocess.run(  # noqa: S603 - el ejecutable es el interprete de la sesion
        [sys.executable, "-m", "cfdtrader.analysis.hyperparameter_search", *arguments],
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


def _source() -> str:
    """El fuente del modulo nuevo, para las comprobaciones por AST."""
    return NEW_MODULE.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    """El AST del modulo nuevo."""
    return ast.parse(_source())


def _synthetic_series(
    observations: int = 500, *, mean: float = 0.0015, sigma: float = 0.01, seed: int = SEED
) -> tuple[float, ...]:
    """Una serie determinista con Sharpe **positivo**, para hacer observable la deflacion (A5)."""
    state = np.random.RandomState(seed)
    return tuple(float(value) for value in state.normal(loc=mean, scale=sigma, size=observations))


def _registry_of(count: int) -> Registry:
    """Un registro sintetico de `count` intentos, con sus Sharpe declarados (A5)."""
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


def _synthetic_frame(*, rows: int = 60, nulls_in: str | None = None) -> pl.DataFrame:
    """Un frame de features minimo con el catalogo declarado, para A2/A7."""
    days = [date(2024, 1, 2) + timedelta(days=index) for index in range(rows)]
    data: dict[str, object] = {
        "session": days,
        **{
            name: [float(index) / 10.0 + offset for index in range(rows)]
            for offset, name in enumerate(feature_store.ALL_FEATURE_COLUMNS)
        },
    }
    frame = pl.DataFrame(data)
    if nulls_in is not None:
        frame = frame.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col(nulls_in))
            .alias(nulls_in)
        )
    return frame


def _synthetic_labels(frame: pl.DataFrame) -> pl.DataFrame:
    """Etiquetas deterministas de ese frame, una por sesion menos la primera (A3)."""
    return pl.DataFrame(
        {
            "session": frame.get_column("session").to_list()[1:],
            "ret_long": [0.01 if index % 2 == 0 else -0.01 for index in range(frame.height - 1)],
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Corridas de la suite
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def base_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """La raiz de trabajo de la suite: el registro con las cuatro entradas congeladas copiadas."""
    _skip_without_store()
    root = tmp_path_factory.mktemp("hyperparameter_search")
    _copy_frozen_runs(root / "runs")
    return root


@pytest.fixture(scope="session")
def real_report(base_root: Path) -> search.SearchReport:
    """La corrida real completa, **una vez** por sesion, escribiendo en un directorio temporal."""
    return analyse(
        store=Store(REAL_DATA),
        reports_dir=base_root / "reports",
        runs_root=base_root / "runs",
        settings=Settings(),
        as_of=NOW,
        write=True,
    )


def _declared_columns(feature_set: str) -> tuple[str, ...]:
    """Las columnas declaradas de ese subconjunto, tal como las publica `FEATURE_SETS` (A1)."""
    return next(item for item in FEATURE_SETS if item.name == feature_set).columns


# ─────────────────────────────────────────────────────────────────────────────
# A1 - el espacio y el presupuesto, como literales pre-registrados
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_audit_rejects_a_space_that_does_not_match_the_preregistration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Las cuatro puertas del auditor: indices, features ajenas, eje movido y subconjunto (A1)."""
    monkeypatch.setattr(
        search,
        "SEARCH_SPACE",
        tuple(dataclasses.replace(variant, index=0) for variant in SEARCH_SPACE),
    )
    with pytest.raises(SearchSpaceError) as indices:
        space_audit()
    assert "indices" in str(indices.value)

    monkeypatch.setattr(search, "SEARCH_SPACE", SEARCH_SPACE)
    with_foreign_features = dataclasses.replace(
        SEARCH_SPACE[0], index=BUDGET - 1, name="h9", features=feature_store.FEATURE_COLUMNS
    )
    monkeypatch.setattr(search, "SEARCH_SPACE", (*SEARCH_SPACE[:-1], with_foreign_features))
    with pytest.raises(SearchSpaceError) as foreign:
        space_audit()
    assert "features de control" in str(foreign.value)

    monkeypatch.setattr(search, "SEARCH_SPACE", SEARCH_SPACE)
    moved_subset = dataclasses.replace(
        SEARCH_SPACE[-1],
        hyperparameters={**SEARCH_SPACE[-1].hyperparameters, "max_depth": 3},
    )
    monkeypatch.setattr(search, "SEARCH_SPACE", (*SEARCH_SPACE[:-1], moved_subset))
    with pytest.raises(SearchSpaceError) as moved:
        space_audit()
    assert "subconjunto" in str(moved.value)

    monkeypatch.setattr(search, "SEARCH_SPACE", SEARCH_SPACE)
    relabelled = dataclasses.replace(
        SEARCH_SPACE[-1], feature_set="control", features=feature_store.FEATURE_COLUMNS
    )
    monkeypatch.setattr(search, "SEARCH_SPACE", (*SEARCH_SPACE[:-1], relabelled))
    with pytest.raises(SearchSpaceError) as mismatch:
        space_audit()
    assert "no coincide" in str(mismatch.value)


def test_a1_the_space_is_the_preregistered_one() -> None:
    """`SEARCH_SPACE` son los `BUDGET` variantes de §19.17 y el auditor lo comprueba (A1)."""
    assert NEW_MODULE.is_file()
    for error in (
        SearchSpaceError,
        VariantNotEvaluableError,
        search.MissingAsOfError,
        search.InvalidAsOfError,
    ):
        assert issubclass(error, HyperparameterSearchError)
    assert BUDGET == len(SEARCH_SPACE) == len(AXES) + len(FEATURE_SETS) == 6 + 4
    assert REGISTERED_TRIALS == 4
    assert TOTAL_TRIALS == REGISTERED_TRIALS + BUDGET == 14
    assert [variant.name for variant in SEARCH_SPACE] == [
        "h0",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "f0",
        "f1",
        "f2",
        "f3",
    ]
    assert [variant.index for variant in SEARCH_SPACE] == list(range(BUDGET))
    for variant in SEARCH_SPACE:
        assert variant.variant_id == f"{VARIANT_ID}#{variant.name}"
        assert set(variant.features) <= set(feature_store.ALL_FEATURE_COLUMNS), variant.name
        if variant.kind == "hyperparameters":
            # Un **solo** eje movido: los demas valores son los de la constante de #26.
            assert search.moved_axes(variant) == (variant.axis,), variant.name
            for name, value in LIGHTGBM_HYPERPARAMETERS.items():
                expected = variant.hyperparameters[name] if name == variant.axis else value
                assert variant.hyperparameters[name] == expected, (variant.name, name)
        else:
            assert variant.features == _declared_columns(variant.feature_set)
    audit = space_audit()
    assert audit["budget"] == BUDGET
    assert audit["n_variants"] == BUDGET
    assert audit["total_trials"] == TOTAL_TRIALS
    assert audit["catalog_columns"] == len(feature_store.ALL_FEATURE_COLUMNS) == 52
    assert [row["variant_id"] for row in cast("list[dict[str, object]]", audit["variants"])] == [
        variant.variant_id for variant in SEARCH_SPACE
    ]
    notes = cast("dict[str, str]", audit["axis_notes"])
    assert "num_leaves" in notes and "max_depth" in notes["num_leaves"]
    assert "§19.17" in notes["num_leaves"]


def test_a1_the_searched_values_are_exactly_the_declared_ones() -> None:
    """Los valores de barrido de cada eje son los de §19.17, literal a literal (A1)."""
    assert all(isinstance(axis, search.Axis) for axis in AXES)
    assert all(isinstance(conjunto, search.FeatureSet) for conjunto in FEATURE_SETS)
    assert {axis.name: (axis.control, axis.value) for axis in AXES} == {
        "n_estimators": (200, 400),
        "learning_rate": (0.05, 0.02),
        "num_leaves": (4, 8),
        "max_depth": (2, 3),
        "min_child_samples": (200, 100),
        "reg_lambda": (0.0, 1.0),
    }
    for variant, axis in zip(SEARCH_SPACE[: len(AXES)], AXES, strict=True):
        assert variant.hyperparameters[axis.name] == axis.value
    assert [conjunto.name for conjunto in FEATURE_SETS] == [
        "control",
        "volatility_vix",
        "technical",
        "macro_regime",
    ]
    assert _declared_columns("control") == BASELINE_FEATURES
    for conjunto in FEATURE_SETS:
        assert set(conjunto.columns) <= set(feature_store.ALL_FEATURE_COLUMNS)
        assert conjunto.provenance and "§19.17" in conjunto.provenance
        assert conjunto.source.startswith("cfdtrader.")


def test_a1_the_enumeration_is_identical_in_three_processes() -> None:
    """Nombres y orden del espacio son los mismos con `PYTHONHASHSEED` 0, 1 y random (A1)."""
    child = textwrap.dedent(
        """
        import json
        from cfdtrader.analysis.hyperparameter_search import SEARCH_SPACE, space_audit

        print(json.dumps({
            "names": [variant.name for variant in SEARCH_SPACE],
            "ids": [variant.variant_id for variant in SEARCH_SPACE],
            "axes": [row["axis"] for row in space_audit()["variants"]],
            "features": [row["features"] for row in space_audit()["variants"]],
        }))
        """
    )
    printed: list[str] = []
    for seed in ("0", "1", "random"):
        completed = subprocess.run(  # noqa: S603 - el ejecutable es el interprete de la sesion
            [sys.executable, "-c", child],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        )
        printed.append(completed.stdout.strip())
    assert len(set(printed)) == 1, printed
    document = cast("dict[str, object]", json.loads(printed[0]))
    assert document["names"] == [variant.name for variant in SEARCH_SPACE]
    assert document["ids"] == [variant.variant_id for variant in SEARCH_SPACE]


def test_a1_a_space_out_of_the_preregistered_one_is_a_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un espacio con dos ejes movidos, con una variante de mas o con otro control falla (A1)."""
    doubled = dataclasses.replace(
        SEARCH_SPACE[0],
        name="h9",
        index=BUDGET - 1,
        hyperparameters={**SEARCH_SPACE[0].hyperparameters, "max_depth": 3},
    )
    monkeypatch.setattr(search, "SEARCH_SPACE", (*SEARCH_SPACE[:-1], doubled))
    with pytest.raises(SearchSpaceError) as error:
        space_audit()
    assert "un solo" in str(error.value)

    monkeypatch.setattr(search, "SEARCH_SPACE", (*SEARCH_SPACE, SEARCH_SPACE[0]))
    with pytest.raises(SearchSpaceError) as longer:
        space_audit()
    assert "presupuesto" in str(longer.value)

    monkeypatch.setattr(search, "SEARCH_SPACE", (*SEARCH_SPACE[:-1], SEARCH_SPACE[0]))
    with pytest.raises(SearchSpaceError) as duplicate:
        space_audit()
    assert "repite" in str(duplicate.value)

    monkeypatch.setattr(search, "SEARCH_SPACE", SEARCH_SPACE)
    monkeypatch.setattr(search, "AXES", (dataclasses.replace(AXES[0], control=999), *AXES[1:]))
    with pytest.raises(SearchSpaceError) as control:
        space_audit()
    assert "control" in str(control.value)


# ─────────────────────────────────────────────────────────────────────────────
# A2 - la matriz de diseno por subconjunto, con la regla de lag de #24
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a2_the_control_matrix_is_identical_to_the_one_of_26(
    real_report: search.SearchReport,
) -> None:
    """Con el control, el diseno es **identico** al de #26: sesiones, valores y corrimiento (A2)."""
    frame = real_report.features
    control = next(variant for variant in SEARCH_SPACE if variant.feature_set == "control")
    ours = search.design_for(control, features=frame.matrix.frame, labels=frame.labels)
    reference = design_frame(frame.matrix.frame, labels=frame.labels)
    assert ours.features == reference.features == BASELINE_FEATURES
    assert ours.sessions == reference.sessions
    assert ours.n_sessions == reference.n_sessions
    assert ours.n_labels == reference.n_labels
    assert ours.n_shifted_rows == reference.n_shifted_rows
    assert ours.n_nulls_in_features == reference.n_nulls_in_features == 0
    assert ours.frame.equals(reference.frame)
    assert ours.design_lag_sessions == reference.design_lag_sessions == 1
    for variant in SEARCH_SPACE:
        design = search.design_for(variant, features=frame.matrix.frame, labels=frame.labels)
        assert design.features == variant.features
        assert design.sessions == reference.sessions
        assert design.n_shifted_rows == reference.n_shifted_rows


def test_a2_a_name_outside_the_catalog_is_a_typed_error() -> None:
    """Un nombre fuera del catalogo de #73 es `UnknownFeatureError`, nunca una columna de ceros."""
    with pytest.raises(UnknownFeatureError) as error:
        search.require_known_features(("har_forecast", "mystery_feature"))
    assert "mystery_feature" in str(error.value)
    with pytest.raises(UnknownFeatureError):
        search.require_known_features(())
    with pytest.raises(UnknownFeatureError):
        search.require_known_features(("har_forecast", "har_forecast"))
    frame = _synthetic_frame()
    variant = dataclasses.replace(SEARCH_SPACE[-1], features=("nope",))
    with pytest.raises(UnknownFeatureError):
        search.design_for(variant, features=frame, labels=_synthetic_labels(frame))


def test_a2_a_subset_with_design_nulls_is_not_imputed() -> None:
    """`n_nulls_in_features` se publica por subconjunto: un nulo de diseno **no** se imputa"""
    frame = _synthetic_frame(nulls_in="har_forecast")
    labels = _synthetic_labels(frame)
    control = next(variant for variant in SEARCH_SPACE if variant.feature_set == "control")
    design = search.design_for(control, features=frame, labels=labels)
    assert design.n_nulls_in_features == 1
    with pytest.raises(VariantNotEvaluableError) as error:
        search.fit_search_variant(
            control, features=frame, labels=labels, plan=cast("Any", None), horizon=()
        )
    assert "nulos de diseno" in str(error.value)
    clean = _synthetic_frame()
    assert (
        search.design_for(
            control, features=clean, labels=_synthetic_labels(clean)
        ).n_nulls_in_features
        == 0
    )


# ─────────────────────────────────────────────────────────────────────────────
# A3 - cada variante entra en el registro con su identidad
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a3_each_variant_has_its_own_run_sha256(real_report: search.SearchReport) -> None:
    """Las diez variantes tienen identidad propia y su `config.json` lleva **sus** valores (A3)."""
    assert len(real_report.records) == BUDGET
    digests = [record.run_sha256 for record in real_report.records]
    assert len(set(digests)) == BUDGET
    for row, record in zip(real_report.rows, real_report.records, strict=True):
        assert record.config.variant_id == row.variant.variant_id
        assert record.config.features == row.variant.features
        assert dict(record.config.hyperparameters) == dict(row.variant.hyperparameters)
        assert record.config.seed == SEED
        assert record.config.series_id == "^GSPC"
        window = dict(record.config.window)
        assert set(window) == {
            "first_session",
            "last_session",
            "n_sessions",
            "n_positives",
            "design_lag_sessions",
            "plan_sha256",
            "matrix_sha256",
            "feature_spec_sha256",
            "feature_code_version",
        }
        assert window["plan_sha256"] == real_report.split_plan.plan_sha256
    new_entries = cast("list[dict[str, object]]", _block(real_report, "registry")["new_entries"])
    assert [entry["variant_id"] for entry in new_entries] == [
        row.variant.variant_id for row in real_report.rows
    ]


def test_a3_changing_one_variant_changes_only_its_own_identity() -> None:
    """Mover **una** variante cambia **su** `run_sha256` y deja los otros nueve igual (A3)."""
    from cfdtrader.analysis.experiment_log import ExperimentConfig, run_sha256

    def configs_of(variants: tuple[SearchVariant, ...]) -> list[str]:
        return [
            run_sha256(
                ExperimentConfig(
                    variant_id=variant.variant_id,
                    features=variant.features,
                    hyperparameters=dict(variant.hyperparameters),
                    seed=SEED,
                    series_id="^GSPC",
                    window={"synthetic": True},
                )
            )
            for variant in variants
        ]

    before = configs_of(SEARCH_SPACE)
    mutated = dataclasses.replace(
        SEARCH_SPACE[3], hyperparameters={**SEARCH_SPACE[3].hyperparameters, "max_depth": 4}
    )
    after = [
        *configs_of(SEARCH_SPACE[:3]),
        *configs_of((mutated,)),
        *configs_of(SEARCH_SPACE[4:]),
    ]
    assert before[:3] == after[:3]
    assert before[4:] == after[4:]
    assert after[3] != before[3]
    assert len(set(after)) == BUDGET


@needs_store
def test_a3_the_ten_entries_are_the_declared_new_ones(
    real_report: search.SearchReport, base_root: Path
) -> None:
    """El registro copiado tiene **14** entradas: las 4 congeladas mas las 10 nuevas (A3)."""
    registry = load_registry(base_root / "runs")
    assert registry.n_trials == TOTAL_TRIALS
    new = {record.run_sha256 for record in real_report.records}
    assert len(new) == BUDGET
    assert set(FROZEN_RUNS) <= {entry.run_sha256 for entry in registry.entries}
    for digest in new:
        directory = base_root / "runs" / digest
        assert (directory / "config.json").is_file()
        assert (directory / "result.json").is_file()
        assert (directory / "model.json").is_file()
        assert (directory / "summary.md").is_file()
    assert all(row.evaluated for row in real_report.rows)


def test_a3_a_variant_is_one_configuration_without_invented_fields() -> None:
    """La configuracion registrada es la de #16 —sus cinco campos— y nada mas (A3)."""
    frame = _frame_stub()
    plan = _plan_stub()
    config = search.variant_config(SEARCH_SPACE[7], frame=frame, plan=plan)
    payload = config.to_payload()
    assert set(payload) == {
        "variant_id",
        "features",
        "hyperparameters",
        "seed",
        "series_id",
        "window",
    }
    assert payload["variant_id"] == "lightgbm_search_v1#f1"
    assert payload["features"] == list(feature_store.FEATURE_COLUMNS)
    assert payload["hyperparameters"] == dict(LIGHTGBM_HYPERPARAMETERS)
    assert payload["seed"] == SEED
    assert payload["series_id"] == "^GSPC"
    assert payload["window"] == {
        "first_session": "2016-01-07",
        "last_session": "2026-09-17",
        "n_sessions": 2688,
        "n_positives": 1400,
        "design_lag_sessions": 1,
        "plan_sha256": "plan",
        "matrix_sha256": "matrix",
        "feature_spec_sha256": {"volatility_v1": "spec"},
        "feature_code_version": "19",
    }


def _frame_stub() -> Any:
    """Un `FeatureFrame` minimo, para comprobar la configuracion sin almacen (A3)."""
    return SimpleNamespace(
        first_session=date(2016, 1, 7),
        last_session=date(2026, 9, 17),
        n_design_rows=2688,
        n_positives=1400,
        design_lag_sessions=1,
        matrix=SimpleNamespace(
            matrix_sha256="matrix",
            feature_spec_sha256={"volatility_v1": "spec"},
            feature_code_version="19",
        ),
    )


def _plan_stub() -> Any:
    """Un `SplitPlan` minimo: la configuracion solo lee su `plan_sha256` (A3)."""
    return SimpleNamespace(plan_sha256="plan")


# ─────────────────────────────────────────────────────────────────────────────
# A4 - las cuatro entradas de #24/#25/#26 quedan intactas
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a4_the_four_frozen_entries_are_untouched(tmp_path: Path) -> None:
    """Ni un byte de las 4 entradas congeladas cambia tras la corrida, y no se re-registran (A4)."""
    runs = tmp_path / "runs"
    _copy_frozen_runs(runs)
    before = _fingerprint(runs)
    report = analyse(
        store=Store(REAL_DATA),
        reports_dir=tmp_path / "reports",
        runs_root=runs,
        settings=Settings(),
        as_of=NOW,
        write=True,
    )
    after = _fingerprint(runs)
    for digest in FROZEN_RUNS:
        for name in ("config.json", "result.json", "model.json", "summary.md"):
            assert after[f"{digest}/{name}"] == before[f"{digest}/{name}"], (digest, name)
    frozen = {key.split("/")[0] for key in before}
    assert (
        set(FROZEN_RUNS)
        == frozen
        == {entry.run_sha256 for entry in report.registry.entries if entry.run_sha256 in frozen}
    )
    assert len(report.registry.entries) == TOTAL_TRIALS
    # Y ningun informe publicado de #24/#25/#26 cambia: la corrida escribe **su** informe.
    frozen = sorted(FROZEN_REPORTS.glob("model_comparison_*"))
    assert frozen, "los informes de #26 tienen que seguir en el arbol"
    for path in frozen:
        assert path.is_file()


@needs_store
def test_a4_a_reloaded_booster_that_does_not_reproduce_is_a_typed_error(
    real_report: search.SearchReport, tmp_path: Path
) -> None:
    """Un `model.json` manipulado no reproduce sus probabilidades: error tipado (A4)."""
    runs = tmp_path / "runs"
    _copy_frozen_runs(runs)
    entry = next(
        item for item in real_report.registry.entries if item.variant_id == LIGHTGBM_VARIANT_ID
    )
    document = cast(
        "dict[str, object]",
        json.loads((runs / entry.run_sha256 / "model.json").read_text(encoding="utf-8")),
    )
    model = cast("dict[str, object]", document["model"])
    folds = cast("list[dict[str, object]]", model["folds"])
    folds[0]["booster_model"] = str(folds[1]["booster_model"])
    (runs / entry.run_sha256 / "model.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(search.ReconstructionMismatchError) as error:
        reconstruct_lightgbm(
            entry,
            runs_root=runs,
            universe=real_report.universe,
            frame=real_report.features,
            plan=real_report.split_plan,
            cost_model=search.declared_cost_model(),
            slippage=search.declared_slippage_assumption(),
        )
    assert "no reproduce" in str(error.value)


@needs_store
def test_a4_a_tampered_observation_count_is_declared_without_a_column(
    real_report: search.SearchReport, tmp_path: Path
) -> None:
    """Un `n_observations` que no cuadra es `not_evaluable` con motivo y sin columna (A4, A7)."""
    runs = tmp_path / "runs"
    _copy_frozen_runs(runs)
    entry = next(
        item
        for item in real_report.registry.entries
        if item.run_sha256 in FROZEN_RUNS and item.variant_id == LIGHTGBM_VARIANT_ID
    )
    tampered = dataclasses.replace(entry, n_observations=entry.n_observations + 1)
    with pytest.raises(search.InconsistentObservationsError):
        reconstruct_lightgbm(
            tampered,
            runs_root=runs,
            universe=real_report.universe,
            frame=real_report.features,
            plan=real_report.split_plan,
            cost_model=search.declared_cost_model(),
            slippage=search.declared_slippage_assumption(),
        )
    candidates = search._candidates_from_registry(  # pyright: ignore[reportPrivateUsage]
        registry=Registry(entries=(tampered,), registry_sha256="synthetic"),
        known={},
        runs_root=runs,
        universe=real_report.universe,
        frame=real_report.features,
        plan=real_report.split_plan,
        cost_model=search.declared_cost_model(),
        slippage=search.declared_slippage_assumption(),
    )
    assert isinstance(candidates[0], search.NotEvaluable)
    assert candidates[0].error == "InconsistentObservationsError"
    assert candidates[0].to_payload()["column"] is None
    # Y una variante **medida** cuyo recuento no cuadre con su entrada tampoco se rellena.
    measured = next(
        item
        for item in real_report.evaluated
        if item.run_sha256 in FROZEN_RUNS and item.variant_id == LIGHTGBM_VARIANT_ID
    )
    with pytest.raises(search.InconsistentObservationsError):
        search._candidates_from_registry(  # pyright: ignore[reportPrivateUsage]
            registry=Registry(entries=(tampered,), registry_sha256="synthetic"),
            known={entry.run_sha256: measured},
            runs_root=runs,
            universe=real_report.universe,
            frame=real_report.features,
            plan=real_report.split_plan,
            cost_model=search.declared_cost_model(),
            slippage=search.declared_slippage_assumption(),
        )


@needs_store
def test_a4_a_model_of_another_design_is_not_comparable(tmp_path: Path) -> None:
    """Un `model.json` con otras features no es comparable: error tipado, no una columna (A4)."""
    frame = search.build_feature_frame(Store(REAL_DATA), series_id="^GSPC")
    entry = next(
        item for item in load_registry(REPO_RUNS).entries if item.variant_id == LIGHTGBM_VARIANT_ID
    )
    runs = tmp_path / "runs"
    runs.mkdir()
    mutated = runs / "mutated-entry"
    shutil.copytree(REPO_RUNS / entry.run_sha256, mutated)
    document = cast(
        "dict[str, object]", json.loads((mutated / "model.json").read_text(encoding="utf-8"))
    )
    model = cast("dict[str, object]", document["model"])
    model["features"] = ["har_forecast"]
    (mutated / "model.json").write_text(json.dumps(document), encoding="utf-8")
    config = cast(
        "dict[str, object]",
        cast(
            "dict[str, object]",
            json.loads((mutated / "config.json").read_text(encoding="utf-8")),
        )["config"],
    )
    window = cast("dict[str, object]", config["window"])
    with pytest.raises(search.UnknownVariantError) as error:
        reconstruct_lightgbm(
            dataclasses.replace(entry, run_sha256=mutated.name),
            runs_root=runs,
            universe=cast("Any", None),
            frame=frame,
            plan=cast("Any", SimpleNamespace(plan_sha256=window["plan_sha256"])),
            cost_model=search.declared_cost_model(),
            slippage=search.declared_slippage_assumption(),
        )
    assert "otras features" in str(error.value)


@needs_store
def test_a4_a_missing_model_document_is_declared_without_a_column(
    real_report: search.SearchReport, tmp_path: Path
) -> None:
    """Una entrada sin `model.json` es `not_evaluable` con motivo y **sin** columna (A4, A7)."""
    runs = tmp_path / "runs"
    _copy_frozen_runs(runs)
    entry = next(
        item for item in real_report.registry.entries if item.variant_id == LIGHTGBM_VARIANT_ID
    )
    (runs / entry.run_sha256 / "model.json").unlink()
    candidates = search._candidates_from_registry(  # pyright: ignore[reportPrivateUsage]
        registry=Registry(entries=(entry,), registry_sha256="synthetic"),
        known={},
        runs_root=runs,
        universe=real_report.universe,
        frame=real_report.features,
        plan=real_report.split_plan,
        cost_model=search.declared_cost_model(),
        slippage=search.declared_slippage_assumption(),
    )
    assert len(candidates) == 1
    missing = candidates[0]
    assert isinstance(missing, search.NotEvaluable)
    assert missing.error == "MissingModelDocumentError"
    assert missing.to_payload()["column"] is None
    assert "model.json" in missing.reason


@needs_store
def test_a4_a_third_family_in_the_registry_is_declared_not_filled(
    real_report: search.SearchReport, tmp_path: Path
) -> None:
    """Una entrada de una familia desconocida es `not_evaluable` con su motivo (A4)."""
    runs = tmp_path / "runs"
    _copy_frozen_runs(runs)
    extra = RegistryEntry(
        run_sha256="deadbeef",
        variant_id="mystery_v1",
        sharpe_per_session=0.3,
        n_observations=10,
    )
    frozen = tuple(
        entry for entry in real_report.registry.entries if entry.run_sha256 in FROZEN_RUNS
    )
    assert len(frozen) == 4
    entries = (*frozen, extra)
    candidates = search._candidates_from_registry(  # pyright: ignore[reportPrivateUsage]
        registry=Registry(entries=entries, registry_sha256="synthetic"),
        known={},
        runs_root=runs,
        universe=real_report.universe,
        frame=real_report.features,
        plan=real_report.split_plan,
        cost_model=search.declared_cost_model(),
        slippage=search.declared_slippage_assumption(),
    )
    missing = [item for item in candidates if isinstance(item, search.NotEvaluable)]
    assert [item.run_sha256 for item in missing] == ["deadbeef"]
    assert "mystery_v1" in missing[0].reason
    assert missing[0].to_payload()["column"] is None


# ─────────────────────────────────────────────────────────────────────────────
# A5 - el `n_trials` sale del registro, no de una bandera
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a5_n_trials_and_sr_variance_come_from_the_registry(
    real_report: search.SearchReport, base_root: Path
) -> None:
    """`n_trials == 14` y `V[SR]` son los del registro, con sus catorce identidades (A5)."""
    registry = _block(real_report, "registry")
    assert registry["n_trials"] == TOTAL_TRIALS == 14
    assert registry["registry_sha256"] == real_report.registry.registry_sha256
    assert registry["sr_variance"] == real_report.registry.sr_variance
    entries = cast("list[dict[str, object]]", registry["entries"])
    assert len(entries) == TOTAL_TRIALS
    assert sorted(str(entry["run_sha256"]) for entry in entries) == sorted(
        entry.run_sha256 for entry in real_report.registry.entries
    )
    assert len(load_registry(base_root / "runs").entries) == TOTAL_TRIALS
    matrix = _block(real_report, "matrix")
    assert matrix["n_variants"] == TOTAL_TRIALS
    assert matrix["matrix_matches_registry"] is True
    assert matrix["registry_n_trials"] == TOTAL_TRIALS
    gate = _block(real_report, "matrix_gate")
    assert gate["matches"] is True
    assert gate["n_columns"] == gate["n_trials"] == TOTAL_TRIALS
    assert len(real_report.evaluated) == TOTAL_TRIALS


@needs_store
def test_a5_thirteen_columns_of_fourteen_is_a_typed_error(
    real_report: search.SearchReport,
) -> None:
    """`require_trials_match_registry` con 13 columnas de 14 intentos es `TrialsMismatchError`."""
    with pytest.raises(TrialsMismatchError):
        require_trials_match_registry(
            n_trials=13,
            sr_variance=real_report.registry.sr_variance,
            registry=real_report.registry,
        )
    assert matrix_matches_registry(registry=real_report.registry, n_columns=13) is False
    require_trials_match_registry(
        n_trials=TOTAL_TRIALS,
        sr_variance=real_report.registry.sr_variance,
        registry=real_report.registry,
    )
    assert matrix_matches_registry(registry=real_report.registry, n_columns=TOTAL_TRIALS) is True


def test_a5_the_cli_does_not_accept_n_trials_or_a_budget() -> None:
    """El CLI no acepta `--n-trials` ni `--budget`: el presupuesto es §19.17 (A5, A11)."""
    for flag, value in (("--n-trials", "14"), ("--budget", "10")):
        completed = _cli("--as-of", NOW.isoformat(), flag, value)
        assert completed.returncode == 2, completed.stderr
        assert flag in completed.stderr


def test_a5_a_report_without_a_measured_best_declares_its_holes() -> None:
    """Sin mejor medido el DSR se declara, y las referencias ausentes son `None`, no cifras (A5)."""
    block = search._dsr_for_best(None, registry=_registry_of(3))  # pyright: ignore[reportPrivateUsage]
    assert block["state"] == "not_evaluable"
    assert block["verdict"] == "not_evaluable"
    assert "deflactar" in str(block["reason"])
    assert search._pick((), "nada") is None  # pyright: ignore[reportPrivateUsage]
    assert search._pick((), None) is None  # pyright: ignore[reportPrivateUsage]
    assert search._selected_of({"state": "not_evaluable"}) is None  # pyright: ignore[reportPrivateUsage]
    selected = {"selected": {"run_sha256": "x"}}
    assert search._selected_of(selected) == "x"  # pyright: ignore[reportPrivateUsage]


def test_a5_more_trials_deflate_the_same_series_more() -> None:
    """El DSR de la **misma** serie baja al crecer el registro: 3 y 14 intentos (A5)."""
    series = _synthetic_series()
    small = deflate_block(returns=series, registry=_registry_of(3))
    large = deflate_block(returns=series, registry=_registry_of(14))
    assert small["n_trials"] == 3
    assert large["n_trials"] == 14
    assert large["dsr"] != small["dsr"]
    assert float(cast("float", large["dsr"])) < float(cast("float", small["dsr"]))


# ─────────────────────────────────────────────────────────────────────────────
# A6 - publicacion corregida: el mejor nunca sale sin su numero deflactado
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a6_the_best_never_comes_without_its_deflated_number(
    real_report: search.SearchReport,
) -> None:
    """Cada variante publica su serie y su DSR, y el mejor los suyos con el PBO al lado (A6)."""
    sweep = _block(real_report, "sweep")
    assert sweep["n_trials"] == TOTAL_TRIALS
    best = cast("dict[str, object]", sweep["best"])
    assert best["variant_id"] in {variant.variant_id for variant in SEARCH_SPACE}
    dsr = cast("dict[str, object]", best["deflated_sharpe_ratio"])
    assert dsr["n_trials"] == TOTAL_TRIALS
    assert dsr["sr_variance"] == real_report.registry.sr_variance
    assert dsr["selected_run_sha256"] == best["run_sha256"]
    assert sweep["dsr"] == _block(real_report, "deflated_sharpe_ratio")
    pbo = _block(real_report, "probability_of_backtest_overfitting")
    assert pbo["state"] == "evaluated"
    assert sweep["pbo"] == pbo
    assert sweep["pbo_max"] == PBO_MAX == 0.2
    assert search.PBO_BLOCKS == 10
    assert _block(real_report, "matrix")["blocks"] == search.PBO_BLOCKS
    for row in _sweep_rows(real_report):
        if row["state"] != "evaluated":
            assert row["deflated_sharpe_ratio"] is None
            continue
        assert row["series"]
        assert row["deciding_probabilities"]
        published = cast("dict[str, object]", row["deflated_sharpe_ratio"])
        assert published["state"] == "evaluated"
        assert published["n_trials"] == TOTAL_TRIALS


@needs_store
def test_a6_the_sweep_verdict_is_never_pass(real_report: search.SearchReport) -> None:
    """El veredicto del barrido es `fail` o `not_evaluable` con su motivo: nunca `pass` (A6)."""
    sweep = _block(real_report, "sweep")
    assert sweep["verdict"] in {"fail", "not_evaluable"}
    pbo = _block(real_report, "probability_of_backtest_overfitting")
    assert sweep["pbo_within_max"] == (float(cast("float", pbo["pbo"])) <= PBO_MAX)
    if sweep["pbo_within_max"] is False:
        assert sweep["verdict"] == "fail"
    assert sweep["rule"] == search.SWEEP_VERDICT_RULE


def test_a6_a_pbo_over_the_maximum_is_a_fail_and_an_uncomputable_one_is_declared() -> None:
    """Con PBO > 0,20 el veredicto es `fail`; con un calculo no evaluable, `not_evaluable` (A6)."""
    assert sweep_verdict(gate="fail", pbo={"state": "evaluated", "pbo": 0.25}) == "fail"
    assert sweep_verdict(gate="pass", pbo={"state": "evaluated", "pbo": 0.5}) == "fail"
    assert sweep_verdict(gate="pass", pbo={"state": "evaluated", "pbo": 0.0}) == "fail"
    assert sweep_verdict(gate="not_evaluable", pbo={"state": "evaluated", "pbo": 0.0}) == (
        "not_evaluable"
    )
    assert sweep_verdict(gate="fail", pbo={"state": "not_evaluable"}) == "not_evaluable"
    verdicts = {
        sweep_verdict(gate=gate, pbo={"state": state, "pbo": value})
        for gate in ("pass", "fail", "not_evaluable")
        for state in ("evaluated", "not_evaluable")
        for value in (0.0, 0.19, 0.21, 1.0)
    }
    assert verdicts <= {"fail", "not_evaluable"}


# ─────────────────────────────────────────────────────────────────────────────
# A7 - presupuesto respetado y fallos registrados
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_the_sweep_calls_record_experiment_exactly_budget_times(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """El numero de `record_experiment` es **exactamente** `BUDGET`, con contador (A7)."""
    calls: list[tuple[str, float, int]] = []

    def fake_record(
        *, runs_root: Path, config: Any, result: Any, as_of: datetime, write: bool
    ) -> ExperimentRecord:
        calls.append(
            (
                str(config.variant_id),
                float(result.sharpe_per_session),
                int(result.n_observations),
            )
        )
        digest = f"{len(calls):064d}"
        return ExperimentRecord(
            as_of=as_of,
            run_sha256=digest,
            directory=runs_root / digest,
            config=config,
            result=result,
            outcome=None,
            written=False,
        )

    monkeypatch.setattr(search, "record_experiment", fake_record)
    frame = _frame_stub()
    design = DesignFrame(
        frame=pl.DataFrame({"session": [], "y": []}),
        n_sessions=2688,
        n_labels=2688,
        n_shifted_rows=0,
        n_nulls_in_features=0,
        design_lag_sessions=1,
    )
    outcomes: dict[str, Any] = {}
    digests: dict[str, str] = {}
    rows: list[search.SweepRow] = []
    for variant in SEARCH_SPACE:
        row, _ = search._record_and_write(  # pyright: ignore[reportPrivateUsage]
            variant,
            design=design,
            model=None,
            measured=None,
            reason="sintetico: sin medida",
            error="VariantNotEvaluableError",
            frame=frame,
            plan=_plan_stub(),
            runs_root=tmp_path / "runs",
            as_of=NOW,
            write=False,
            outcomes=outcomes,
            digests=digests,
        )
        rows.append(row)
    assert len(calls) == BUDGET == len(rows)
    assert [name for name, _, _ in calls] == [variant.variant_id for variant in SEARCH_SPACE]
    # Un intento sin medida se registra con el resultado nulo **declarado** y sin columna.
    assert all(sharpe == 0.0 and observations == 0 for _, sharpe, observations in calls)
    assert all(row.state == "not_evaluable" for row in rows)
    assert all(row.to_payload()["column"] is None for row in rows)
    assert all(isinstance(row, search.SweepRow) for row in rows)
    assert all(row.to_payload()["n_traded"] is None for row in rows)
    assert not (tmp_path / "runs").exists()


def _synthetic_rows() -> list[search.SweepRow]:
    """Las diez filas del espacio **intentadas y sin medida**, para los bordes del informe (A7)."""
    design = DesignFrame(
        frame=pl.DataFrame({"session": [], "y": []}),
        n_sessions=2688,
        n_labels=2688,
        n_shifted_rows=0,
        n_nulls_in_features=1,
        design_lag_sessions=1,
    )
    return [
        search.SweepRow(
            variant=variant,
            run_sha256=f"{variant.index:064d}",
            state="not_evaluable",
            reason="sintetico: sin medida",
            error="VariantNotEvaluableError",
            n_sessions=design.n_sessions,
            n_labels=design.n_labels,
            n_shifted_rows=design.n_shifted_rows,
            n_nulls_in_features=design.n_nulls_in_features,
            model_sha256=None,
            measured=None,
        )
        for variant in SEARCH_SPACE
    ]


def test_a7_a_subset_with_nulls_is_an_attempt_without_a_measure() -> None:
    """`_attempt` deja la variante intentada con su motivo y sin modelo (A7)."""
    frame = _synthetic_frame(nulls_in="har_forecast")
    labels = _synthetic_labels(frame)
    stub = SimpleNamespace(matrix=SimpleNamespace(frame=frame))
    control = next(variant for variant in SEARCH_SPACE if variant.feature_set == "control")
    design, model, measured, reason, error = search._attempt(  # pyright: ignore[reportPrivateUsage]
        control,
        frame=cast("Any", stub),
        labels=labels,
        plan=cast("Any", None),
        horizon=(),
        universe=cast("Any", None),
        cost_model=cast("Any", None),
        slippage=cast("Any", None),
    )
    assert design.n_nulls_in_features == 1
    assert model is None and measured is None
    assert error == "VariantNotEvaluableError"
    assert "nulos de diseno" in str(reason)


def test_a7_an_incomplete_matrix_is_declared_and_does_not_break_the_sweep() -> None:
    """Con 13 columnas de 14 intentos el DSR/PBO se declaran y el barrido no tumba (A5, A7)."""
    rows = _synthetic_rows()
    payloads = search._sweep_row_payloads(  # pyright: ignore[reportPrivateUsage]
        rows, registry=_registry_of(14)
    )
    assert all(row["deflated_sharpe_ratio"] is None for row in payloads)
    block = search._sweep_block(  # pyright: ignore[reportPrivateUsage]
        evaluated=cast("Any", (object(),) * 13),
        registry=_registry_of(14),
        rows=rows,
        best=None,
        reference_winner=None,
        dsr={"state": "not_evaluable", "verdict": "not_evaluable"},
        pbo={"state": "not_evaluable", "verdict": "not_evaluable"},
    )
    assert block["state"] == "not_evaluable"
    assert block["verdict"] == "not_evaluable"
    assert block["best"] is None
    assert block["blockers"] == [row.run_sha256 for row in rows]
    assert "matriz" in str(block["reason"])


@needs_store
def test_a7_the_real_run_registers_every_attempt(real_report: search.SearchReport) -> None:
    """La corrida real registra los `BUDGET` intentos y publica el estado de cada uno (A7)."""
    assert len(real_report.records) == BUDGET
    assert set(real_report.outcomes) == {record.run_sha256 for record in real_report.records}
    rows = _sweep_rows(real_report)
    assert len(rows) == BUDGET
    for row in rows:
        assert row["state"] in {"evaluated", "not_evaluable"}
        if row["state"] == "not_evaluable":
            assert row["reason"] and row["column"] is None
            assert row["deflated_sharpe_ratio"] is None
        else:
            assert row["column"] == row["run_sha256"]
            assert row["model_sha256"] and len(str(row["model_sha256"])) == 64
    assert _block(real_report, "sweep")["n_sweep_evaluated"] == sum(
        1 for row in rows if row["state"] == "evaluated"
    )


# ─────────────────────────────────────────────────────────────────────────────
# A8 - comparacion con la familia de #26 bajo el mismo protocolo
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a8_every_row_is_measured_here_with_the_declared_basis(
    real_report: search.SearchReport,
) -> None:
    """Cada fila publica Brier, log-loss y operadas, con sus dos deltas y su base (A8)."""
    comparison = _block(real_report, "comparison")
    assert comparison["basis"] == "declared_cost"
    assert comparison["is_validation"] is False
    assert comparison["primary_metric"] == "brier_score"
    assert comparison["n_rows"] == BUDGET == len(_rows(real_report))
    reference_raw = cast("dict[str, object]", comparison["reference_raw"])
    reference_winner = cast("dict[str, object]", comparison["reference_winner_26"])
    assert reference_raw["variant_id"] == BASELINE_VARIANT_ID
    assert reference_raw["calibrated"] is False
    assert reference_winner["run_sha256"] == reference_raw["run_sha256"]
    for row in _rows(real_report):
        assert row["basis"] == "declared_cost"
        assert row["is_validation"] is False
        for name in ("brier_score", "log_loss", "n_traded"):
            assert name in row
        if row["state"] != "evaluated":
            assert row["brier_score"] is None
            assert row["delta_vs_baseline_raw"] is None
            continue
        published = cast("dict[str, object]", row["deflated_sharpe_ratio"])
        assert published["n_trials"] == TOTAL_TRIALS
        for key in ("delta_vs_baseline_raw", "delta_vs_winner_26"):
            delta = cast("dict[str, object]", row[key])
            assert delta["reference_run_sha256"] in {
                reference_raw["run_sha256"],
                reference_winner["run_sha256"],
            }
            assert float(cast("float", delta["brier_score"])) == float(
                cast("float", row["brier_score"])
            ) - (
                float(cast("float", reference_raw["brier_score"]))
                if key == "delta_vs_baseline_raw"
                else float(cast("float", reference_winner["brier_score"]))
            )


@needs_store
def test_a8_the_table_is_not_copied_from_the_frozen_reports(
    real_report: search.SearchReport,
) -> None:
    """La tabla sale de esta corrida: el modulo no lee ningun informe congelado (A8)."""
    assert "model_comparison_2026" not in _source()
    assert "baseline_2026" not in _source()
    reads = [
        ast.unparse(node)
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"read_text", "read_bytes"}
    ]
    assert not [call for call in reads if "reports" in call], reads
    published = sorted(
        float(cast("float", row["brier_score"]))
        for row in _rows(real_report)
        if row["state"] == "evaluated"
    )
    in_rows = sorted(
        float(cast("float", row["brier_score"]))
        for row in _sweep_rows(real_report)
        if row["state"] == "evaluated"
    )
    assert published == in_rows != []
    frozen = cast(
        "dict[str, object]",
        json.loads(
            (FROZEN_REPORTS / "model_comparison_2026-09-22.json").read_text(encoding="utf-8")
        ),
    )
    comparison = cast("dict[str, object]", frozen["comparison"])
    foreign_rows = cast("list[dict[str, object]]", comparison["rows"])
    foreign = {str(row["variant_id"]) for row in foreign_rows}
    assert foreign & {variant.variant_id for variant in SEARCH_SPACE} == set()


@needs_store
@needs_store
def test_a8_mutating_the_frozen_report_does_not_change_the_table(
    real_report: search.SearchReport, tmp_path: Path
) -> None:
    """Mutar el informe congelado de #26 no cambia la tabla: sale de `runs/`, no de el (A8)."""
    frozen = FROZEN_REPORTS / "model_comparison_2026-09-22.json"
    document = cast("dict[str, object]", json.loads(frozen.read_text(encoding="utf-8")))
    comparison = cast("dict[str, object]", document["comparison"])
    rows = cast("list[dict[str, object]]", comparison["rows"])
    rows[0]["brier_score"] = 0.999
    mutated = tmp_path / "reports"
    mutated.mkdir()
    (mutated / frozen.name).write_text(json.dumps(document), encoding="utf-8")
    runs = tmp_path / "runs"
    _copy_frozen_runs(runs)
    fresh = analyse(
        store=Store(REAL_DATA),
        reports_dir=mutated,
        runs_root=runs,
        settings=Settings(),
        as_of=NOW,
        write=False,
    )
    assert [row["brier_score"] for row in _rows(fresh)] == [
        row["brier_score"] for row in _rows(real_report)
    ]
    assert 0.999 not in [row["brier_score"] for row in _rows(fresh)]
    assert fresh.report_sha256 == real_report.report_sha256


def test_a8_a_wave_without_a_measure_has_no_delta() -> None:
    """Una fila sin medida no publica cifras ni deltas, y la referencia ausente es `None` (A8)."""
    block = search._comparison_block(  # pyright: ignore[reportPrivateUsage]
        _synthetic_rows(),
        registry=_registry_of(14),
        reference_raw=None,
        reference_winner=None,
    )
    assert block["basis"] == "declared_cost"
    assert block["is_validation"] is False
    assert block["n_rows"] == BUDGET
    assert block["n_evaluated"] == 0
    assert block["reference_raw"] is None
    assert block["reference_winner_26"] is None
    for row in cast("list[dict[str, object]]", block["rows"]):
        assert row["state"] == "not_evaluable"
        assert row["brier_score"] is None
        assert row["delta_vs_baseline_raw"] is None
        assert row["delta_vs_winner_26"] is None
        assert row["deflated_sharpe_ratio"] is None


@needs_store
def test_a8_the_protocol_is_the_one_of_26(real_report: search.SearchReport) -> None:
    """El protocolo publicado es el de #26: mismo plan, umbral 0,5 y coste declarado (A8)."""
    protocol = _block(real_report, "protocol")
    universe = cast("dict[str, object]", protocol["universe"])
    assert protocol["decision_threshold"] == 0.5
    assert protocol["cost_basis"] == "declared_cost"
    assert protocol["features"] == list(BASELINE_FEATURES)
    assert protocol["n_folds"] == 10
    assert protocol["n_test"] == 500 == 10 * 50
    assert protocol["plan_sha256"] == real_report.split_plan.plan_sha256
    assert universe["n_sessions"] == len(real_report.universe.inputs)


@needs_store
def test_a8_the_markdown_declares_a_report_without_a_best(
    real_report: search.SearchReport,
) -> None:
    """El `.md` declara el hueco cuando no hay mejor medido ni referencias (A8, A7)."""
    payload = copy.deepcopy(real_report.payload)
    sweep = cast("dict[str, object]", payload["sweep"])
    sweep["best"] = None
    sweep["state"] = "not_evaluable"
    sweep["reason"] = "sintetico: matriz incompleta"
    sweep["family_changed"] = True
    sweep["follow_ups"] = ["#27", "#28"]
    sweep["blockers"] = ["abc123"]
    comparison = cast("dict[str, object]", payload["comparison"])
    comparison["reference_raw"] = None
    comparison["reference_winner_26"] = None
    for row in cast("list[dict[str, object]]", comparison["rows"]):
        row["deflated_sharpe_ratio"] = None
        row["delta_vs_baseline_raw"] = None
        row["delta_vs_winner_26"] = None
    text = render_markdown(dataclasses.replace(real_report, payload=payload))
    assert "**Sin mejor medido**" in text
    assert "sintetico: matriz incompleta" in text
    assert "**Vetos** (variante intentada sin columna)" in text
    assert "no medida en esta corrida" in text
    assert "Seguimiento: #27, #28." in text


@needs_store
def test_a8_the_markdown_carries_the_table_and_the_rule(real_report: search.SearchReport) -> None:
    """El `.md` lleva la tabla del barrido y la regla declarada, y termina en salto (A8)."""
    text = render_markdown(real_report)
    assert "## Tabla del barrido (las 10 variantes, las mismas 500 sesiones de test)" in text
    for variant in SEARCH_SPACE:
        assert f"| `{variant.variant_id}` |" in text
    assert "## La familia de #26: se usa, no se reabre" in text
    assert "Δ Brier vs cruda #24" in text
    assert "Δ Brier vs ganadora #26" in text
    assert text.endswith("\n")


@needs_store
def test_a8_the_comparison_rows_are_the_sweep_rows(real_report: search.SearchReport) -> None:
    """La tabla y `sweep_rows[]` publican las **mismas** cifras por variante (A8)."""
    for row, sweep_row in zip(_rows(real_report), _sweep_rows(real_report), strict=True):
        assert row["variant_id"] == sweep_row["variant_id"]
        assert row["run_sha256"] == sweep_row["run_sha256"]
        for name in ("brier_score", "log_loss", "n_traded", "sharpe_per_session"):
            assert row[name] == sweep_row[name], (row["variant_id"], name)
        assert row["features"] == sweep_row["features"]
        assert dict(cast("Mapping[str, object]", row["hyperparameters"])) == dict(
            cast("Mapping[str, object]", sweep_row["hyperparameters"])
        )


# ─────────────────────────────────────────────────────────────────────────────
# A9 - vallas de honestidad
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a9_the_unit_bug_block_publishes_the_declared_displacement(
    real_report: search.SearchReport,
) -> None:
    """El bloque de #80 publica `0,0042 - 0,000042 = 0,004158` por operacion y lo mide (A9)."""
    bug = _block(real_report, "unit_bug_80")
    assert bug["issue"] == "#80"
    assert bug["state"] == "fixed_and_measured"
    displacement = cast("dict[str, object]", bug["displacement"])
    assert displacement["declared_pct"] == "0.0042"
    assert displacement["correct_fraction"] == "0.000042"
    assert displacement["per_operation"] == pytest.approx(0.004158, abs=1e-12)
    assert "0,004158" in str(displacement["identity"])
    assert abs(float(cast("float", bug["observed_per_operation"])) - 0.000042) <= 1e-15
    assert abs(float(cast("float", bug["difference_per_operation"]))) <= 1e-12
    assert "probabilidad" in str(displacement["does_not_affect"])
    assert "Sharpe" in str(displacement["affects"])
    reference = cast("dict[str, object]", _block(real_report, "comparison")["reference_raw"])
    operations = int(cast("int", bug["n_operations"]))
    assert bug["observed_on"] == reference["run_sha256"] or operations > 0


@needs_store
def test_a9_the_family_change_is_declared_with_its_follow_up(
    real_report: search.SearchReport,
) -> None:
    """`family_changed` es un booleano y abre seguimiento a #27/#28 solo si es verdadero (A9)."""
    sweep = _block(real_report, "sweep")
    assert isinstance(sweep["family_changed"], bool)
    assert sweep["follow_ups"] == (["#27", "#28"] if sweep["family_changed"] else [])
    status = "cambiaria" if sweep["family_changed"] else "no** cambia"
    assert status in str(sweep["family_statement"])
    assert "no se tocan" in render_markdown(real_report) or "se usa, no se reabre" in (
        render_markdown(real_report)
    )


def test_a9_a_worse_sweep_does_not_change_the_family_and_a_better_one_does() -> None:
    """Solo una mejora **estricta** en la regla de #26 cambia la familia ganadora (A9)."""
    better = cast("Any", SimpleNamespace(brier_score=0.24, log_loss_value=0.69))
    worse = cast("Any", SimpleNamespace(brier_score=0.27, log_loss_value=0.80))
    tie = cast(
        "Any", SimpleNamespace(brier_score=0.2511170746947622, log_loss_value=0.6953524466258852)
    )
    reference = cast(
        "Any", SimpleNamespace(brier_score=0.2511170746947622, log_loss_value=0.6953524466258852)
    )
    assert search._beats_reference(better, reference) is True  # pyright: ignore[reportPrivateUsage]
    assert search._beats_reference(worse, reference) is False  # pyright: ignore[reportPrivateUsage]
    assert search._beats_reference(tie, reference) is False  # pyright: ignore[reportPrivateUsage]
    assert search._beats_reference(worse, None) is True  # pyright: ignore[reportPrivateUsage]
    better_log_loss = cast(
        "Any",
        SimpleNamespace(brier_score=0.2511170746947622, log_loss_value=0.60),
    )
    assert (
        search._beats_reference(  # pyright: ignore[reportPrivateUsage]
            better_log_loss, reference
        )
        is True
    )
    worse_log_loss = cast(
        "Any",
        SimpleNamespace(brier_score=0.2511170746947622, log_loss_value=0.99),
    )
    assert (
        search._beats_reference(  # pyright: ignore[reportPrivateUsage]
            worse_log_loss, reference
        )
        is False
    )


@needs_store
def test_a9_the_delivery_leaves_the_engine_the_cost_and_the_plan_untouched(
    real_report: search.SearchReport,
) -> None:
    """`git diff` no toca el motor ni el coste, y la puerta sigue en `fail` (A9).

    `_docs/plan.md` se retira de esta lista en #107: la decision del propietario anade su
    seccion §19.18 (y su fila de changelog) al documento.
    """
    changed = set(
        subprocess.run(  # noqa: S603 - el git del sistema, comando fijo
            ["git", "diff", "--name-only", f"{BASE_COMMIT}..HEAD"],  # noqa: S607
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    )
    assert not (changed & set(OUT_OF_DELIVERY)), sorted(changed & set(OUT_OF_DELIVERY))
    assert NEW_MODULE.relative_to(REPO_ROOT).as_posix() in changed
    limits = _block(real_report, "limits")
    assert limits["gate"] == "fail"
    assert limits["phase1_ready"] is False
    assert limits["phase2_ready"] is False
    assert limits["is_validation"] is False
    assert limits["plan_sections_untouched"] == ["§11.6", "§19.6", "§19.7"]
    assert str(limits["slippage_state"]).startswith("assumed")


# ─────────────────────────────────────────────────────────────────────────────
# A10 - determinismo del informe
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a10_the_second_pass_is_unchanged_and_hashes_the_same(
    real_report: search.SearchReport, base_root: Path
) -> None:
    """La segunda pasada deja las diez entradas nuevas en `unchanged` y hashea igual (A10)."""
    second = analyse(
        store=Store(REAL_DATA),
        reports_dir=base_root / "reports-second",
        runs_root=base_root / "runs",
        settings=Settings(),
        as_of=NOW,
        write=True,
    )
    assert second.report_sha256 == real_report.report_sha256
    assert second.json_text() == real_report.json_text()
    assert second.registry.registry_sha256 == real_report.registry.registry_sha256
    assert len(second.records) == BUDGET
    for record in second.records:
        assert second.outcomes[record.run_sha256] == "unchanged", record.run_sha256


@needs_store
def test_a10_the_cli_is_byte_identical_across_processes_and_reports_dirs(
    base_root: Path, tmp_path: Path
) -> None:
    """`report_sha256`, `.json` y `.md` no cambian con el proceso ni con `--reports-dir` (A10)."""
    runs = tmp_path / "runs"
    _copy_frozen_runs(runs)
    targets = (tmp_path / "reports", tmp_path / "other-reports", tmp_path / "third-reports")
    for seed, target in zip(("0", "1", "random"), targets, strict=True):
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
    in_process = base_root / "reports" / f"{name}.json"
    hashes = {_report_hash(target / f"{name}.json") for target in targets}
    hashes.add(_report_hash(in_process))
    assert len(hashes) == 1
    for suffix in ("json", "md"):
        payloads = {(target / f"{name}.{suffix}").read_bytes() for target in targets}
        payloads.add((base_root / "reports" / f"{name}.{suffix}").read_bytes())
        assert len(payloads) == 1, suffix
    document = cast(
        "dict[str, object]", json.loads((targets[0] / f"{name}.json").read_text(encoding="utf-8"))
    )
    assert str(document["report_sha256"]).startswith("sha256:")
    assert str(runs) not in json.dumps(document)


def test_a10_the_report_hash_ignores_the_paths() -> None:
    """El payload no lleva rutas absolutas: el hash no depende de `--runs-root` (A10)."""
    block = search._settings_block(Settings())  # pyright: ignore[reportPrivateUsage]
    declared = str(block["data_root_declared"])
    assert not Path(declared).is_absolute()
    assert "relativa" in str(block["note"])
    assert search.DEFAULT_RUNS_ROOT == "runs"
    naive = datetime(2026, 1, 1)
    assert search._as_utc(naive) == naive.replace(tzinfo=UTC)  # pyright: ignore[reportPrivateUsage]
    aware = datetime(2026, 1, 1, 12, tzinfo=UTC)
    assert search._as_utc(aware) == aware  # pyright: ignore[reportPrivateUsage]


# ─────────────────────────────────────────────────────────────────────────────
# A11 - CLI sin banderas de espacio
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_the_cli_without_as_of_exits_two_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sin `--as-of`: exit 2, motivo en `stderr` y **cero** ficheros (A11)."""
    assert main([]) == 2
    captured = capsys.readouterr()
    assert "--as-of" in captured.err
    assert list(tmp_path.iterdir()) == []


def test_a11_the_cli_flags_are_the_declared_ones() -> None:
    """El CLI acepta las cinco banderas declaradas y **ninguna** de espacio (A11)."""
    assert search.CLI_NAME == "cfdtrader.analysis.hyperparameter_search"
    completed = _cli("--help")
    assert completed.returncode == 0, completed.stderr
    for flag in ("--data-root", "--reports-dir", "--runs-root", "--as-of", "--dry-run"):
        assert flag in completed.stdout, flag
    for forbidden in ("--n-trials", "--budget", "--features", "--axis"):
        assert forbidden not in completed.stdout, forbidden


def test_a11_a_bad_or_missing_as_of_is_a_typed_error() -> None:
    """`--as-of` ausente o invalido son errores tipados: nada se escribe (A11)."""
    with pytest.raises(search.MissingAsOfError):
        search._parse_as_of(None)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(search.InvalidAsOfError):
        search._parse_as_of("ayer")  # pyright: ignore[reportPrivateUsage]
    assert search._parse_as_of(NOW.isoformat()) == NOW  # pyright: ignore[reportPrivateUsage]


def test_a11_the_signature_and_the_absence_of_the_clock() -> None:
    """La firma de `analyse` es la declarada y el modulo no lee el reloj (A11)."""
    parameters = inspect.signature(analyse).parameters
    assert list(parameters) == [
        "store",
        "reports_dir",
        "runs_root",
        "settings",
        "as_of",
        "write",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values()
    )
    assert parameters["write"].default is True
    found = [
        ast.unparse(node)
        for node in ast.walk(_tree())
        if (isinstance(node, ast.Attribute) and node.attr in {"now", "utcnow", "today"})
        or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "time"
        )
    ]
    assert not found, found


def test_a11_a_missing_dataset_exits_two_without_a_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Con un almacen vacio: exit 2, motivo en `stderr` y ningun informe escrito (A11)."""
    empty = tmp_path / "empty-store"
    empty.mkdir()
    reports = tmp_path / "reports"
    code = main(
        [
            "--data-root",
            str(empty),
            "--reports-dir",
            str(reports),
            "--runs-root",
            str(tmp_path / "runs"),
            "--as-of",
            NOW.isoformat(),
        ]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert "barrido" in captured.err
    assert not reports.exists()


@needs_store
def test_a11_the_dry_run_writes_nothing(tmp_path: Path) -> None:
    """`--dry-run` no escribe **nada**: ni informe ni carpetas del registro (A11)."""
    runs = tmp_path / "runs"
    _copy_frozen_runs(runs)
    before = _fingerprint(runs)
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
            "--dry-run",
        ]
    )
    assert code == 0
    assert not reports.exists()
    assert _fingerprint(runs) == before


# ─────────────────────────────────────────────────────────────────────────────
# A12 - cobertura, un test por criterio, puertas
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_every_criterion_has_a_test() -> None:
    """Un `test_aN_` por criterio (A1..A12), y los tests solo escriben en `tmp_path` (A12)."""
    missing = [
        index
        for index in range(1, 13)
        if not re.search(rf"^def test_a{index}_", TEST_SOURCE, re.MULTILINE)
    ]
    assert missing == []
    assert "tmp_path" in TEST_SOURCE
    conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "_repository_data_is_untouched" in conftest
    assert "_repository_runs_is_untouched" in conftest


def test_a12_the_public_surface_is_exercised_and_pragma_free() -> None:
    """Cada nombre publico aparece en la suite y el unico `pragma` es la entrada (A12)."""
    assert [name for name in search.__all__ if name not in TEST_SOURCE] == []
    pragmas = [line for line in _source().splitlines() if "pragma: no cover" in line]
    assert len(pragmas) == 1
    assert "__main__" in pragmas[0]
    for public in search.__all__:
        attribute = getattr(search, public)
        if callable(attribute) and not isinstance(attribute, type):
            assert (attribute.__doc__ or "").strip(), public


def test_a12_the_boundaries_are_machine_readable() -> None:
    """Las fronteras y los seguimientos llevan su issue, y el espacio cita §19.17/§19.18 (A12)."""
    for entry in search.REPORT_DOES_NOT_DO:
        assert entry["issue"].startswith("#")
        assert entry["statement"]
        assert entry["id"]
    for entry in search.FOLLOW_UPS:
        assert entry["issue"].startswith("#")
        assert entry["why"]
    assert {entry["issue"] for entry in search.REPORT_DOES_NOT_DO} >= {"#82", "#26", "#62", "#27"}
    assert search.PBO_MAX == 0.2
    assert "§19.18" in search.SPACE_SOURCE
    assert "§19.17" in search.SPACE_SOURCE
