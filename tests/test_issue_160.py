"""Presupuesto formal de `reversion_v1` (#160, seguimiento de #159).

Blinda lo que el **codigo** declara: el grupo `reversion` del CLI, el espacio pre-registrado
(control + seis adiciones de 11 + seis intercambios de 10, **nunca** el ampliado de 16), la
decision `amplia` sobre una adicion, el nombre del informe, el documento de decision y las vallas
de produccion. Los numeros vivos los publica la corrida real en
``_docs/feature_budget_reversion_2026-10-24.md``.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import polars as pl
import pytest

from cfdtrader.analysis import feature_budget
from cfdtrader.data.store import Store
from cfdtrader.features import store as feature_store
from cfdtrader.models.baseline import BASELINE_FEATURES

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DOC_PATH: Final[Path] = REPO_ROOT / "_docs" / "feature_budget_reversion_2026-10-24.md"
NOTE_PATH: Final[Path] = REPO_ROOT / "_docs" / "reversion_candidates_2026-10-16.md"

CANDIDATES: Final[tuple[str, ...]] = feature_store.REVERSION_FEATURE_COLUMNS
COLUMNS: Final[tuple[str, ...]] = (*BASELINE_FEATURES, *CANDIDATES)
_SESSIONS: Final[int] = 40


def _matrix(*, copies: dict[str, str] | None = None) -> pl.DataFrame:
    """Matriz sintetica con las 16 columnas del universo, sin nulos ni `NaN`."""
    data: dict[str, list[float]] = {
        name: [math.cos((index + position) / 2.5) for index in range(_SESSIONS)]
        for position, name in enumerate(COLUMNS)
    }
    for candidate, control in (copies or {}).items():
        data[candidate] = list(data[control])
    return pl.DataFrame({name: pl.Series(values) for name, values in data.items()})


def _space(matrix: pl.DataFrame | None = None) -> tuple[feature_budget.FeatureSet, ...]:
    """El espacio declarado del grupo `reversion`, tal y como lo pide `measure`."""
    return feature_budget.planned_sets(
        matrix if matrix is not None else _matrix(),
        candidates=feature_budget.REVERSION_CANDIDATES,
        candidate_set=feature_store.REVERSION_FEATURE_SET,
        swap_prefix=feature_budget.REVERSION_SWAP_PREFIX,
        add_prefix=feature_budget.REVERSION_ADD_PREFIX,
        per_candidate_additions=True,
    )


def _measurement(name: str, kind: str, columns: tuple[str, ...]) -> feature_budget.SetMeasurement:
    """Una medida sintetica valida para construir un `BudgetReport`."""
    return feature_budget.SetMeasurement(
        name=name,
        kind=kind,
        columns=columns,
        usable=True,
        n_traded=0,
        n_test_sessions=0,
        sharpe=None,
        brier=None,
        log_loss=None,
        matrix_sha256="sha256:test",
        features_version="sha256:test",
        feature_spec_sha256={},
        feature_code_version=2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# A1 — el catalogo y su disponibilidad
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_candidates_come_from_the_declared_family_with_lag_one() -> None:
    """Las seis candidatas son las del catalogo de #159 y declaran `design_lag = 1`."""
    assert feature_store.REVERSION_FEATURE_SET == "reversion_v1"
    assert CANDIDATES == (
        "vix_ts_slope",
        "vix_contango",
        "vix9d_vix",
        "breadth",
        "streak_up",
        "ret_5_pct",
    )
    assert feature_budget.REVERSION_CANDIDATES == CANDIDATES
    assert all(feature_store.DESIGN_LAG_BY_FEATURE[name] == 1 for name in CANDIDATES)
    assert not set(CANDIDATES) & set(BASELINE_FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# A2 — el espacio pre-registrado: seis adiciones y seis intercambios
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_space_is_control_six_additions_and_six_swaps() -> None:
    """El espacio es control + 6 `add_<candidata>` (11) + 6 `swap_<candidata>` (10)."""
    sets = _space()
    assert [item.name for item in sets] == [
        feature_budget.CONTROL_SET,
        *(f"add_{name}" for name in CANDIDATES),
        *(f"swap_{name}" for name in CANDIDATES),
    ]
    assert [item.n_features for item in sets] == [10] + [11] * 6 + [10] * 6
    assert [item.kind for item in sets] == ["control"] + ["add"] * 6 + ["swap"] * 6


def test_a2_the_expanded_set_of_sixteen_is_not_built() -> None:
    """El ampliado de las seis juntas (16) supera el limite 10-15 y **no** se construye."""
    sets = _space()
    assert not any(item.kind == "expanded" for item in sets)
    assert max(item.n_features for item in sets) == 11
    assert all(
        feature_budget.BUDGET_MIN <= item.n_features <= feature_budget.BUDGET_MAX for item in sets
    )


def test_a2_asking_for_the_expanded_alongside_additions_is_an_error() -> None:
    """`per_candidate_additions` y el ampliado son incompatibles: error tipado, no silencio."""
    with pytest.raises(feature_budget.BudgetError, match="no construyen el ampliado"):
        feature_budget.planned_sets(
            _matrix(),
            candidates=CANDIDATES,
            candidate_set=feature_store.REVERSION_FEATURE_SET,
            per_candidate_additions=True,
            expanded_source="BASELINE_FEATURES + reversion_v1",
        )


# ─────────────────────────────────────────────────────────────────────────────
# A3 — la composicion de cada conjunto
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_each_addition_is_control_plus_exactly_one_candidate() -> None:
    """Una adicion anade **una** candidata al control y conserva las diez columnas."""
    additions = [item for item in _space() if item.kind == "add"]
    assert len(additions) == len(CANDIDATES)
    for addition, candidate in zip(additions, CANDIDATES, strict=True):
        assert addition.columns == (*BASELINE_FEATURES, candidate)
        assert set(addition.columns) - set(BASELINE_FEATURES) == {candidate}
        assert addition.source.endswith(f"`{candidate}` (reversion_v1)")
        assert "10-15" in addition.provenance


def test_a3_each_swap_keeps_the_control_size_and_drops_one_distinct_column() -> None:
    """Cada intercambio tiene 10 columnas y no reutiliza una columna de control que ya salio."""
    swaps = [item for item in _space() if item.kind == "swap"]
    assert len(swaps) == len(CANDIDATES)
    dropped: list[str] = []
    for swap, candidate in zip(swaps, CANDIDATES, strict=True):
        assert len(swap.columns) == len(BASELINE_FEATURES) == 10
        assert candidate in swap.columns
        difference = [name for name in BASELINE_FEATURES if name not in swap.columns]
        assert len(difference) == 1
        dropped.append(difference[0])
        assert swap.source == (
            f"BASELINE_FEATURES - `{difference[0]}` + `{candidate}` (regla del intercambio)"
        )
    assert len(set(dropped)) == len(dropped)


def test_a3_the_swap_with_a_copied_candidate_drops_the_copied_control_column() -> None:
    """La regla del intercambio mide `|r|` **antes** del modelo: una copia exacta se ve."""
    matrix = _matrix(copies={"vix_ts_slope": "europe_prev_1"})
    swaps = [item for item in _space(matrix) if item.kind == "swap"]
    first = swaps[0]
    assert first.name == "swap_vix_ts_slope"
    assert "europe_prev_1" not in first.columns
    assert "vix_ts_slope" in first.columns


def test_a3_the_space_is_deterministic() -> None:
    """El mismo frame da el mismo espacio, en el mismo orden y con el mismo payload."""
    first = _space()
    second = _space()
    assert [item.to_payload() for item in first] == [item.to_payload() for item in second]


# ─────────────────────────────────────────────────────────────────────────────
# A4 — la decision `amplia` sobre una adicion
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_an_addition_adopted_with_the_gate_is_amplia_and_without_it_keeps_control() -> None:
    """`mejora` + DSR/PBO `amplia` la adicion; sin la puerta, `mantiene_control`."""
    addition = next(item for item in _space() if item.kind == "add")
    assert feature_budget.decide_budget(
        feature_budget.VERDICT_IMPROVES, addition, significance_ok=True
    ) == (feature_budget.DECISION_ADOPT, addition.name)
    assert feature_budget.decide_budget(
        feature_budget.VERDICT_IMPROVES, addition, significance_ok=False
    ) == (feature_budget.DECISION_KEEP, feature_budget.CONTROL_SET)
    for verdict in (
        feature_budget.VERDICT_TIES,
        feature_budget.VERDICT_WORSENS,
        feature_budget.VERDICT_NOT_EVALUABLE,
    ):
        assert feature_budget.decide_budget(verdict, addition, significance_ok=True) == (
            feature_budget.DECISION_KEEP,
            feature_budget.CONTROL_SET,
        )


def test_a4_an_unknown_group_is_a_typed_error_before_touching_the_store(tmp_path: Path) -> None:
    """El grupo se valida antes de leer el almacen: sin datos no hay medicion inventada."""
    with pytest.raises(feature_budget.BudgetError, match="grupo de candidatas desconocido"):
        feature_budget.measure(
            store=Store(tmp_path / "store"),
            as_of=datetime(2026, 10, 24, 22, tzinfo=UTC),
            candidate_group="no_existe",
        )


# ─────────────────────────────────────────────────────────────────────────────
# A5 — CLI e informe
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_the_cli_declares_the_three_groups_and_keeps_commodities_by_default() -> None:
    """`reversion` es una eleccion explicita; sin argumento, el grupo sigue siendo commodities."""
    parser = feature_budget.build_parser()
    assert parser.parse_args([]).candidate_group == feature_budget.COMMODITIES_CANDIDATE_GROUP
    assert (
        parser.parse_args(["--candidate-group", "reversion"]).candidate_group
        == feature_budget.REVERSION_CANDIDATE_GROUP
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--candidate-group", "no_existe"])


def test_a5_the_report_name_is_separate_and_the_decision_document_exists() -> None:
    """El informe no pisa el de #146 y el documento publica la decision con su identidad."""
    adopted = _measurement(feature_budget.CONTROL_SET, "control", feature_budget.CONTROL_FEATURES)
    report = feature_budget.BudgetReport(
        as_of=datetime(2026, 10, 24, 22, tzinfo=UTC),
        report_date=date(2026, 10, 24),
        payload={},
        report_sha256="sha256:test",
        sets=(),
        verdict=feature_budget.Verdict("empata", None, None, 0),
        decision=feature_budget.DECISION_KEEP,
        adopted=adopted,
        candidate_group=feature_budget.REVERSION_CANDIDATE_GROUP,
    )
    assert report.report_stem == "feature_budget_reversion_2026-10-24"
    assert (
        feature_budget.BudgetReport(
            as_of=report.as_of,
            report_date=report.report_date,
            payload={},
            report_sha256="sha256:test",
            sets=(),
            verdict=report.verdict,
            decision=report.decision,
            adopted=adopted,
        ).report_stem
        == "feature_budget_2026-10-24"
    )

    assert DOC_PATH.is_file()
    text = DOC_PATH.read_text(encoding="utf-8")
    for needle in (
        "#160",
        "reversion_v1",
        "mantiene_control",
        "DSR",
        "PBO",
        "10-15",
        "phase2_ready",
        "matrix_sha256",
        "report_sha256",
        "BASELINE_FEATURES",
    ):
        assert needle in text, needle
    note = NOTE_PATH.read_text(encoding="utf-8")
    assert "feature_budget_reversion_2026-10-24.md" in note


# ─────────────────────────────────────────────────────────────────────────────
# A6 — la produccion no cambia
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_the_control_and_the_default_group_do_not_move() -> None:
    """`BASELINE_FEATURES`, sus diez columnas y el grupo por defecto siguen intactos."""
    assert len(BASELINE_FEATURES) == 10
    assert feature_budget.CONTROL_FEATURES == BASELINE_FEATURES
    assert feature_budget.CANDIDATE_FEATURES == feature_store.COMMODITIES_FEATURE_COLUMNS
    assert feature_budget.CANDIDATE_SET == feature_store.COMMODITIES_FEATURE_SET
    assert feature_budget.BUDGET_MIN == 10 and feature_budget.BUDGET_MAX == 15
    assert not set(CANDIDATES) & set(BASELINE_FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# A7 — el bloque del payload y el Markdown de la decision
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_the_payload_block_declares_the_space_and_the_common_window() -> None:
    """El bloque de `reversion` declara catalogo, lags, ventana y que el ampliado no cabe."""
    block = feature_budget._reversion_experiment(  # pyright: ignore[reportPrivateUsage]
        candidate_features=CANDIDATES,
        candidate_matrix_sha256="sha256:matrix",
        n_candidate_design_rows=2688,
        sessions=(date(2016, 1, 7), date(2026, 9, 17)),
        n_sessions=2680,
        design_lag_by_feature=dict.fromkeys(CANDIDATES, 1),
        split_plan_audit={
            "purge_total": 0,
            "embargo_total": 45,
            "embargo_in_train_total": 0,
        },
    )
    assert block["candidate_set"] == "reversion_v1"
    assert block["expanded_set_admissible"] is False
    assert "16" in str(block["expanded_set_reason"])
    assert "10-15" in str(block["expanded_set_reason"])
    assert block["candidate_matrix_sha256"] == "sha256:matrix"
    assert block["design_lag_by_feature"] == dict.fromkeys(CANDIDATES, 1)
    assert len(block["catalog"]) == len(CANDIDATES)  # type: ignore[arg-type]
    window = block["common_complete_case_window"]
    assert isinstance(window, dict)
    assert window["rows_used"] == 2680
    assert window["rows_excluded"] == 8
    assert window["first_session_used"] == "2016-01-07"
    assert block["walk_forward_exclusions"] == {
        "purge_total": 0,
        "embargo_total": 45,
        "embargo_in_train_total": 0,
    }


def test_a7_the_markdown_declares_the_space_and_the_decision() -> None:
    """El Markdown de `reversion` publica la tabla, la puerta y el espacio declarado."""
    adopted = _measurement(feature_budget.CONTROL_SET, "control", feature_budget.CONTROL_FEATURES)
    payload: dict[str, object] = {
        "decision": {
            "decision": feature_budget.DECISION_KEEP,
            "adopted_set": feature_budget.CONTROL_SET,
            "n_features": 10,
            "gate": {"passed": False, "reason": "DSR no significant; PBO detected"},
        },
        "comparison": {
            "best_candidate": "swap_streak_up",
            "mean_return_difference": 0.000327,
            "standard_error": 0.000243,
            "rule": "regla pareada",
        },
        "identity": {
            "matrix_sha256": "sha256:matrix",
            "adopted_features_version": "sha256:adopted",
        },
        "deflated_sharpe_ratio": {
            "verdict": "not_significant",
            "dsr": 0.3478,
            "n_trials": 13,
            "selected_set": "swap_streak_up",
        },
        "probability_of_backtest_overfitting": {
            "verdict": "detected",
            "pbo": 0.5437,
            "blocks": 10,
            "pbo_max": 0.2,
        },
        "reversion_experiment": {
            "common_complete_case_window": {"rows_used": 2688, "rows_excluded": 0},
            "walk_forward_exclusions": {
                "purge_total": 0,
                "embargo_total": 45,
                "embargo_in_train_total": 0,
            },
        },
    }
    report = feature_budget.BudgetReport(
        as_of=datetime(2026, 10, 24, 22, tzinfo=UTC),
        report_date=date(2026, 10, 24),
        payload=payload,
        report_sha256="sha256:report",
        sets=(adopted,),
        verdict=feature_budget.Verdict("mejora", 0.000327, 0.000243, 500),
        decision=feature_budget.DECISION_KEEP,
        adopted=adopted,
        candidate_group=feature_budget.REVERSION_CANDIDATE_GROUP,
    )
    text = feature_budget._render_markdown(report)  # pyright: ignore[reportPrivateUsage]
    assert text.startswith("# Presupuesto de features de `reversion_v1`")
    assert "## Espacio y ventana" in text
    assert "16 columnas" in text and "10-15" in text
    assert "2.688" in text or "2688" in text
    assert "no se recalcula ninguna feature" in text
    assert "no se adopta nada en producción" in text
