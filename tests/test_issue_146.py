"""Presupuesto de features de #24 y re-medida de la Fase 2 (`#146`).

Blinda lo que el **codigo** declara, no los numeros vivos: la medicion decide y vive en
``_docs/feature_budget_2026-10-10.md``. Cubre: el modulo y su API (A1), las reglas pre-registradas
(A2), el espacio control/ampliado/intercambios (A3), la regla del intercambio por `|r|` y el
desempate declarado (A4), que los intercambios tienen el mismo tamano y una sola columna distinta
(A5), el veredicto pareado (A6), la puerta de significacion en la decision (A7), los bloques del PBO
(A8), la identidad del diseno (A9), la pureza del modulo (A10), el payload JSON estricto (A11) y el
documento de decision (A12).
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Final

import polars as pl
import pytest

from cfdtrader.analysis import feature_budget
from cfdtrader.features import store as feature_store
from cfdtrader.models.baseline import BASELINE_FEATURES

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODULE_PATH: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "analysis" / "feature_budget.py"
DECISION_PATH: Final[Path] = REPO_ROOT / "_docs" / "feature_budget_2026-10-10.md"

#: Las columnas del espacio: control + las cinco candidatas de #143, sin solaparse.
COLUMNS: Final[tuple[str, ...]] = (*BASELINE_FEATURES, *feature_store.COMMODITIES_FEATURE_COLUMNS)

_SESSIONS: Final[int] = 40


def _matrix(*, correlated: bool = False) -> pl.DataFrame:
    """Una matriz sintetica con las 15 columnas del espacio, sin nulos ni NaN (A3, A4)."""
    base = [float(index) for index in range(_SESSIONS)]
    data: dict[str, list[float]] = {}
    for position, name in enumerate(COLUMNS):
        data[name] = [math.sin((value + position) / 3.0) for value in base]
    if correlated:
        # `oil_ret_1` pasa a ser una copia exacta de `europe_prev_1`: la regla del intercambio
        # tiene que sacar `europe_prev_1` (la de mayor |r| con la candidata).
        data["oil_ret_1"] = list(data["europe_prev_1"])
    return pl.DataFrame(
        {name: pl.Series(values, dtype=pl.Float64) for name, values in data.items()}
    )


# ─────────────────────────────────────────────────────────────────────────────
# A1 - el modulo y su API
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_module_and_its_public_api() -> None:
    """A1: el modulo vive en el arbol y exporta lo que el informe necesita."""
    assert MODULE_PATH.is_file()
    for name in (
        "BUDGET_RULE",
        "INTERCHANGE_RULE",
        "VERDICT_RULE",
        "GATE_RULE",
        "planned_sets",
        "classify_verdict",
        "decide_budget",
        "measure",
    ):
        assert name in feature_budget.__all__, name
        assert hasattr(feature_budget, name), name


# ─────────────────────────────────────────────────────────────────────────────
# A2 - las reglas pre-registradas
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_rules_are_declared_and_complete() -> None:
    """A2: las cuatro reglas son texto declarado con sus decisiones y su valla."""
    assert "amplia" in feature_budget.BUDGET_RULE
    assert "intercambia" in feature_budget.BUDGET_RULE
    assert "mantiene_control" in feature_budget.BUDGET_RULE
    assert "10-15" in feature_budget.BUDGET_RULE
    assert "|r|" in feature_budget.INTERCHANGE_RULE
    assert "empate" in feature_budget.INTERCHANGE_RULE.lower()
    assert "mejora" in feature_budget.VERDICT_RULE
    assert "empata" in feature_budget.VERDICT_RULE
    assert "empeora" in feature_budget.VERDICT_RULE
    assert "not_evaluable" in feature_budget.VERDICT_RULE
    assert "significant" in feature_budget.GATE_RULE
    assert "detected" in feature_budget.GATE_RULE
    assert feature_budget.TOLERANCE_STANDARD_ERRORS > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# A3 - el espacio
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_the_space_is_control_expanded_and_one_swap_per_candidate() -> None:
    """A3: control, ampliado y un intercambio por candidata, todos dentro del limite 10-15."""
    sets = feature_budget.planned_sets(_matrix())
    names = [item.name for item in sets]
    assert names[0] == feature_budget.CONTROL_SET
    assert names[1] == feature_budget.EXPANDED_SET
    assert len(names) == 2 + len(feature_budget.CANDIDATE_FEATURES)
    kinds = {item.kind for item in sets}
    assert kinds == {"control", "expanded", "swap"}

    control = sets[0]
    expanded = sets[1]
    assert control.columns == feature_budget.CONTROL_FEATURES
    assert control.columns == BASELINE_FEATURES
    assert expanded.columns == (
        *feature_budget.CONTROL_FEATURES,
        *feature_budget.CANDIDATE_FEATURES,
    )
    assert expanded.n_features == 15
    assert feature_budget.BUDGET_MIN <= control.n_features <= feature_budget.BUDGET_MAX
    assert feature_budget.BUDGET_MIN <= expanded.n_features <= feature_budget.BUDGET_MAX


def test_a3_the_space_rejects_overlap_and_empty_inputs() -> None:
    """A3: control y candidatas solapados, o vacios, son error tipado, no un espacio valido."""
    with pytest.raises(feature_budget.BudgetError):
        feature_budget.planned_sets(_matrix(), candidates=("har_forecast",))
    with pytest.raises(feature_budget.BudgetError):
        feature_budget.planned_sets(_matrix(), candidates=())


# ─────────────────────────────────────────────────────────────────────────────
# A4 - la regla del intercambio
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_the_swap_drops_the_most_correlated_control_column() -> None:
    """A4: la columna que sale es la de mayor `|r|` con la candidata (medido antes del modelo)."""
    matrix = _matrix(correlated=True)
    assert feature_budget.redundant_control_feature(matrix, "oil_ret_1") == "europe_prev_1"


def test_a4_a_constant_candidate_breaks_the_tie_by_declared_order() -> None:
    """A4: con todos los `|r|` a cero (candidata constante), gana la primera del control."""
    matrix = _matrix().with_columns(pl.lit(1.0).alias("gold_ret_1"))
    assert (
        feature_budget.redundant_control_feature(matrix, "gold_ret_1")
        == (feature_budget.CONTROL_FEATURES[0])
    )


def test_a4_the_space_is_deterministic() -> None:
    """A4: el mismo frame da el mismo espacio, en el mismo orden (sin orden de contenedor)."""
    first = feature_budget.planned_sets(_matrix())
    second = feature_budget.planned_sets(_matrix())
    assert [item.to_payload() for item in first] == [item.to_payload() for item in second]


def test_a4_a_missing_column_is_an_error() -> None:
    """A4: una columna que la matriz no trae es error tipado, nunca una correlacion inventada."""
    with pytest.raises(feature_budget.BudgetError):
        feature_budget.redundant_control_feature(_matrix(), "no_existe")


# ─────────────────────────────────────────────────────────────────────────────
# A5 - los intercambios son del mismo tamano
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_every_swap_keeps_the_control_size_and_changes_one_column() -> None:
    """A5: un intercambio no amplia el presupuesto: mismo tamano y **una** columna distinta."""
    sets = feature_budget.planned_sets(_matrix(correlated=True))
    swaps = [item for item in sets if item.kind == "swap"]
    control = sets[0].columns
    assert len(swaps) == len(feature_budget.CANDIDATE_FEATURES)
    dropped: list[str] = []
    for swap in swaps:
        assert set(swap.columns) != set(control)
        assert len(swap.columns) == len(control) == feature_budget.BUDGET_MIN
        difference = [name for name in control if name not in swap.columns]
        assert len(difference) == 1, difference
        arrived = [name for name in swap.columns if name not in control]
        assert len(arrived) == 1
        assert arrived[0] in feature_budget.CANDIDATE_FEATURES
        dropped.append(difference[0])
    assert len(set(dropped)) == len(dropped), "una columna de control no puede salir dos veces"


# ─────────────────────────────────────────────────────────────────────────────
# A6 - el veredicto pareado
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_the_verdict_classifies_improve_tie_and_worsen() -> None:
    """A6: `mejora`/`empata`/`empeora` segun la regla pareada pre-registrada, con sus numeros."""
    control = [0.0] * 50
    stronger = [0.01] * 50
    # Ruido simetrico de media cero: la media pareada cae dentro del error estandar ⇒ `empata`.
    noise = [0.001 if index % 2 == 0 else -0.001 for index in range(50)]
    improves = feature_budget.classify_verdict(control, stronger)
    assert improves.verdict == feature_budget.VERDICT_IMPROVES
    assert improves.mean_difference == pytest.approx(0.01)
    assert improves.standard_error is not None

    worsens = feature_budget.classify_verdict(stronger, control)
    assert worsens.verdict == feature_budget.VERDICT_WORSENS

    ties = feature_budget.classify_verdict(control, noise)
    assert ties.verdict == feature_budget.VERDICT_TIES


def test_a6_the_verdict_needs_two_paired_sessions() -> None:
    """A6: con una sola sesion el error estandar no existe: error tipado, no un numero inventado."""
    with pytest.raises(feature_budget.BudgetError):
        feature_budget.classify_verdict([0.0], [0.1])
    with pytest.raises(feature_budget.BudgetError):
        feature_budget.classify_verdict([0.0, 0.0], [0.1])


# ─────────────────────────────────────────────────────────────────────────────
# A7 - la puerta de significacion en la decision
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_a_descriptive_improvement_without_the_gate_keeps_the_control() -> None:
    """A7: `mejora` sin DSR `significant` ni PBO no `detected` **no** mueve produccion."""
    best = next(item for item in feature_budget.planned_sets(_matrix()) if item.kind == "expanded")
    assert feature_budget.decide_budget(
        feature_budget.VERDICT_IMPROVES, best, significance_ok=False
    ) == (feature_budget.DECISION_KEEP, feature_budget.CONTROL_SET)
    assert feature_budget.decide_budget(
        feature_budget.VERDICT_IMPROVES, best, significance_ok=True
    ) == (feature_budget.DECISION_ADOPT, best.name)


def test_a7_only_an_improvement_adopts_and_swaps_intercambia() -> None:
    """A7: `empata`/`empeora`/`not_evaluable` mantienen; un swap adoptado es `intercambia`."""
    sets = feature_budget.planned_sets(_matrix())
    expanded = next(item for item in sets if item.kind == "expanded")
    swap = next(item for item in sets if item.kind == "swap")
    for verdict in (
        feature_budget.VERDICT_TIES,
        feature_budget.VERDICT_WORSENS,
        feature_budget.VERDICT_NOT_EVALUABLE,
    ):
        assert feature_budget.decide_budget(verdict, swap, significance_ok=True) == (
            feature_budget.DECISION_KEEP,
            feature_budget.CONTROL_SET,
        )
    assert feature_budget.decide_budget(
        feature_budget.VERDICT_IMPROVES, swap, significance_ok=True
    ) == (feature_budget.DECISION_SWAP, swap.name)
    assert feature_budget.decide_budget(
        feature_budget.VERDICT_IMPROVES, expanded, significance_ok=True
    ) == (feature_budget.DECISION_ADOPT, expanded.name)


def test_a7_the_gate_needs_evaluated_dsr_and_pbo() -> None:
    """A7: un DSR o un PBO no evaluables no son un aprobado: la puerta no pasa."""
    significance = feature_budget._significance  # pyright: ignore[reportPrivateUsage]
    evaluated_ok = {"state": "evaluated", "verdict": "significant", "dsr": 0.99}
    evaluated_pbo = {"state": "evaluated", "verdict": "not_detected", "pbo": 0.05}
    ok, _ = significance(evaluated_ok, evaluated_pbo)
    assert ok
    for broken in (
        {"state": "not_evaluable"},
        {"state": "evaluated", "verdict": "not_significant", "dsr": 0.5},
    ):
        failed, reason = significance(broken, evaluated_pbo)
        assert not failed
        assert reason


# ─────────────────────────────────────────────────────────────────────────────
# A8 - los bloques del PBO
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_the_pbo_blocks_are_a_valid_even_divisor_or_none() -> None:
    """A8: `S` par, >= 2 y <= DEFAULT_BLOCKS que divide a `T`; sin divisor valido, `None`."""
    blocks_for = feature_budget._blocks_for  # pyright: ignore[reportPrivateUsage]
    assert blocks_for(500) == 10
    assert blocks_for(16) == 16
    assert blocks_for(12) == 12
    assert blocks_for(7) is None
    assert blocks_for(15) is None


# ─────────────────────────────────────────────────────────────────────────────
# A9 - la identidad del diseno
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_the_features_version_is_deterministic_and_well_formed() -> None:
    """A9: `features_version` es `sha256:` + 64 hex, estable y sensible a la lista."""
    version_of = feature_budget._features_version  # pyright: ignore[reportPrivateUsage]
    first = version_of(("a", "b"), matrix_sha256="sha256:x", plan_sha256="sha256:y")
    again = version_of(("a", "b"), matrix_sha256="sha256:x", plan_sha256="sha256:y")
    other = version_of(("b", "a"), matrix_sha256="sha256:x", plan_sha256="sha256:y")
    assert first == again
    assert first != other
    assert first.startswith("sha256:")
    digest = first.removeprefix("sha256:")
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)


# ─────────────────────────────────────────────────────────────────────────────
# A10 - el modulo es puro
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_the_module_does_not_read_the_clock_the_network_or_the_environment() -> None:
    """A10: sin reloj, sin red, sin entorno: `--as-of` da el instante y el `Store`, los datos."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in ("datetime.now", "utcnow", "time.time", "os.environ", "open("):
        assert forbidden not in source, forbidden
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])
    assert not ({"requests", "httpx", "yfinance", "urllib", "socket"} & imported), sorted(imported)


# ─────────────────────────────────────────────────────────────────────────────
# A11 - el payload es JSON estricto
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_the_payloads_are_strict_json_and_none_is_not_zero() -> None:
    """A11: los payloads son JSON estricto (sin `NaN`) y un valor no medido sigue siendo `None`."""
    measurement = feature_budget.SetMeasurement(
        name=feature_budget.CONTROL_SET,
        kind="control",
        columns=feature_budget.CONTROL_FEATURES,
        usable=False,
        n_traded=0,
        n_test_sessions=0,
        sharpe=None,
        brier=None,
        log_loss=None,
        matrix_sha256="sha256:x",
        features_version="sha256:y",
        feature_spec_sha256={},
        feature_code_version=2,
        error="no opero",
    )
    payload = measurement.to_payload()
    assert payload["sharpe"] is None
    assert payload["n_features"] == 10
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert "NaN" not in text and "null" in text
    verdict = feature_budget.Verdict(
        verdict=feature_budget.VERDICT_NOT_EVALUABLE,
        mean_difference=None,
        standard_error=None,
        n_sessions=0,
    )
    assert json.dumps(verdict.to_payload(), allow_nan=False)
    assert verdict.to_payload()["rule"] == feature_budget.VERDICT_RULE


# ─────────────────────────────────────────────────────────────────────────────
# A12 - el documento de decision
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_the_decision_document_declares_rule_verdict_and_decision() -> None:
    """A12: el documento declara la regla, la ventana, el veredicto, la puerta y la decision."""
    assert DECISION_PATH.is_file()
    text = DECISION_PATH.read_text(encoding="utf-8")
    for needle in (
        "#146",
        "regla",
        "amplia",
        "intercambia",
        "mantiene_control",
        "casos completos",
        "mejora",
        "not_significant",
        "detected",
        "phase2_ready",
        "matrix_sha256",
        "features_version",
        "BASELINE_FEATURES",
    ):
        assert needle in text, needle
