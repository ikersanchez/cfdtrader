"""Tests de las particiones *walk-forward* con purga y embargo (#12).

Un test por criterio de aceptación (A1-A34). Lo importante que se comprueba aquí:

- los índices son **posiciones de sesión**, no fechas: un puente de calendario largo no
  cambia el tamaño de los bloques (A4, A16);
- la **purga** implementa literalmente ``i + h[i] >= test_start`` (A9) y con el horizonte de
  #10 (``h = 0``) es un **no-op medido**, no afirmado (A10, A18);
- el plan es **determinista** y reproducible por ``plan_sha256`` (A21, A23);
- cada error de A24, la falta de sesiones (A25) y el train vacío (A26) son **tipados**;
- el módulo es **puro**: solo biblioteca estándar, sin reloj, sin azar y sin disco (A22).

Los valores de purga y embargo se calculan **a mano** en el test, no se copian del módulo.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from cfdtrader.backtest import splits
from cfdtrader.backtest.splits import (
    Fold,
    InsufficientSessionsError,
    InvalidSplitParameterError,
    SplitPlan,
    SplitsError,
    walk_forward_splits,
)

MODULE_PATH = Path(str(splits.__file__))
SOURCE = MODULE_PATH.read_text(encoding="utf-8")
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Lo que A1 exige como mínimo en ``__all__``.
REQUIRED_PUBLIC = (
    "SplitsError",
    "InsufficientSessionsError",
    "Fold",
    "SplitPlan",
    "walk_forward_splits",
)

#: Los campos que A3 exige en ``SplitPlan`` y en ``Fold``.
PLAN_FIELDS = (
    "n_sessions",
    "inputs",
    "folds",
    "uncovered",
    "purge_total",
    "embargo_total",
    "plan_sha256",
)
FOLD_FIELDS = ("index", "test_start", "test_stop", "train", "purged", "embargoed")

#: Sesiones sintéticas (lunes a viernes; el módulo no mira el calendario real).
START = date(2026, 1, 5)


def business_sessions(count: int, *, start: date = START) -> list[date]:
    """``count`` sesiones de lunes a viernes consecutivos (posiciones, no calendario)."""
    out: list[date] = []
    day = start
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def sessions_with_long_gap(*, before: int, gap_days: int, after: int) -> list[date]:
    """Sesiones con un **puente de calendario** de ``gap_days`` días entre dos tramos."""
    first = business_sessions(before)
    return first + business_sessions(after, start=first[-1] + timedelta(days=gap_days))


def _code_only() -> str:
    """El código del módulo sin **ningún** literal de cadena (ni docstrings ni declaraciones)."""
    masked_lines: set[int] = set()
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            masked_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return "\n".join(
        "" if number in masked_lines else row
        for number, row in enumerate(SOURCE.splitlines(), start=1)
    )


def _imported_modules() -> set[str]:
    """Módulos importados por el módulo (para demostrar que A22/A27 usan stdlib)."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def _verify_fold() -> Any:
    """``_verify_fold`` por reflexión: es privado, pero A8/A19 exigen que verifique."""
    return splits._verify_fold  # pyright: ignore[reportPrivateUsage]


# ─────────────────────────────────────────────────────────────────────────────
# A1, A2 — ficheros, API pública y firma
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_module_files_and_public_api() -> None:
    assert MODULE_PATH.is_file()
    assert MODULE_PATH.name == "splits.py"
    assert MODULE_PATH.parent.name == "backtest"
    assert (REPO_ROOT / "tests" / "test_splits.py").is_file()
    assert isinstance(splits.__all__, (list, tuple))
    assert set(REQUIRED_PUBLIC) <= set(splits.__all__)
    assert not [name for name in splits.__all__ if name.startswith("_")]
    for name in splits.__all__:
        assert getattr(splits, name) is not None, name
    assert issubclass(InsufficientSessionsError, SplitsError)
    assert issubclass(InvalidSplitParameterError, SplitsError)
    assert issubclass(SplitsError, Exception)


def test_a2_signature_is_keyword_only_and_without_invented_defaults() -> None:
    signature = inspect.signature(walk_forward_splits)
    assert list(signature.parameters) == [
        "sessions",
        "label_horizon",
        "n_splits",
        "test_size",
        "embargo_sessions",
        "max_train_size",
    ]
    assert signature.parameters["sessions"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("label_horizon", "n_splits", "test_size", "embargo_sessions", "max_train_size"):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
    for name in ("label_horizon", "n_splits", "test_size", "embargo_sessions"):
        assert signature.parameters[name].default is inspect.Parameter.empty, name
    # ``None`` significa «expansivo»: no es un tamaño inventado (A20).
    assert signature.parameters["max_train_size"].default is None


# ─────────────────────────────────────────────────────────────────────────────
# A3 — la forma del plan
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_plan_and_fold_expose_the_declared_fields() -> None:
    sessions = business_sessions(30)
    plan = walk_forward_splits(
        sessions, label_horizon=[0] * 30, n_splits=3, test_size=4, embargo_sessions=2
    )
    for field in PLAN_FIELDS:
        assert hasattr(plan, field), field
    assert plan.n_sessions == 30
    assert plan.inputs["n_splits"] == 3
    assert plan.inputs["test_size"] == 4
    assert plan.inputs["embargo_sessions"] == 2
    assert plan.inputs["max_train_size"] is None
    assert plan.inputs["label_horizon"] == (0,) * 30
    assert plan.inputs["n_sessions"] == 30
    assert len(plan.folds) == 3
    assert isinstance(plan.uncovered, tuple)
    assert all(isinstance(i, int) for i in plan.uncovered)
    for fold in plan.folds:
        for field in FOLD_FIELDS:
            assert hasattr(fold, field), field
        assert isinstance(fold.index, int)
        assert isinstance(fold.test_start, int)
        assert isinstance(fold.test_stop, int)
        for values in (fold.train, fold.purged, fold.embargoed):
            assert isinstance(values, tuple)
            assert all(isinstance(i, int) and not isinstance(i, bool) for i in values)


# ─────────────────────────────────────────────────────────────────────────────
# A4 — posiciones de sesión, nunca aritmética de calendario
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_long_calendar_gap_does_not_change_fold_sizes() -> None:
    sessions = sessions_with_long_gap(before=32, gap_days=12, after=18)
    gaps = [i for i in range(len(sessions) - 1) if (sessions[i + 1] - sessions[i]).days > 10]
    assert gaps == [31]  # el puente cae dentro del primer bloque de test
    plan = walk_forward_splits(
        sessions, label_horizon=[0] * 50, n_splits=4, test_size=5, embargo_sessions=0
    )
    assert [(f.test_start, f.test_stop) for f in plan.folds] == [
        (30, 35),
        (35, 40),
        (40, 45),
        (45, 50),
    ]
    assert plan.folds[0].test == (30, 31, 32, 33, 34)  # 5 sesiones aunque el puente sea de 12 días
    assert all(len(f.test) == 5 for f in plan.folds)


def test_a4_no_calendar_arithmetic_in_the_code() -> None:
    code = _code_only()
    assert "timedelta" not in code
    assert "MarketCalendar" not in code
    assert "date.today" not in code


# ─────────────────────────────────────────────────────────────────────────────
# A5, A6, A7 — bloques de test y invariante dura
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_tests_are_contiguous_disjoint_and_cover_the_tail_exactly_once() -> None:
    sessions = business_sessions(23)
    plan = walk_forward_splits(
        sessions, label_horizon=[0] * 23, n_splits=3, test_size=5, embargo_sessions=1
    )
    assert len(plan.folds) == 3
    covered: list[int] = []
    for fold in plan.folds:
        assert len(fold.test) == 5
        assert fold.test_stop - fold.test_start == 5
        covered.extend(fold.test)
    assert len(covered) == len(set(covered)) == 3 * 5
    assert covered == list(range(23 - 15, 23))  # las últimas k*T posiciones, en orden
    for previous, current in zip(plan.folds, plan.folds[1:], strict=False):
        assert previous.test_stop == current.test_start  # contiguos


def test_a6_uncovered_is_published_and_nothing_is_dropped_silently() -> None:
    sessions = business_sessions(23)
    plan = walk_forward_splits(
        sessions, label_horizon=[0] * 23, n_splits=3, test_size=5, embargo_sessions=1
    )
    assert plan.uncovered == tuple(range(23 - 15))  # calentamiento inicial (8 posiciones)
    assert list(plan.uncovered) == sorted(plan.uncovered)
    assert len(plan.uncovered) + 3 * 5 == plan.n_sessions == 23
    covered = {i for fold in plan.folds for i in fold.test}
    assert covered.isdisjoint(plan.uncovered)


def test_a7_every_train_index_precedes_its_test_block() -> None:
    sessions = business_sessions(40)
    plan = walk_forward_splits(
        sessions, label_horizon=[1] * 40, n_splits=4, test_size=5, embargo_sessions=2
    )
    for fold in plan.folds:
        assert max(fold.train, default=-1) < fold.test_start
        assert set(fold.train).isdisjoint(fold.test)


# ─────────────────────────────────────────────────────────────────────────────
# A8, A19 — la verificación se hace siempre y no devuelve un plan roto
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_broken_invariants_raise_splits_error() -> None:
    verify = _verify_fold()
    # train con la última posición >= test_start
    with pytest.raises(SplitsError):
        verify(
            Fold(index=0, test_start=5, test_stop=8, train=(5,), purged=(), embargoed=()),
            label_horizon=(0,) * 10,
        )
    # train y embargo solapados
    with pytest.raises(SplitsError):
        verify(
            Fold(index=0, test_start=5, test_stop=6, train=(0,), purged=(), embargoed=(0,)),
            label_horizon=(0,) * 10,
        )
    # la purga dejó una muestra cuyo t1 invade el test
    with pytest.raises(SplitsError):
        verify(
            Fold(index=0, test_start=3, test_stop=4, train=(0,), purged=(), embargoed=()),
            label_horizon=(3,) * 10,
        )


def test_a19_verified_invariants_hold_in_every_fold() -> None:
    sessions = business_sessions(40)
    horizon = [2] * 40
    plan = walk_forward_splits(
        sessions, label_horizon=horizon, n_splits=4, test_size=5, embargo_sessions=3
    )
    for fold in plan.folds:
        assert set(fold.train).isdisjoint(fold.test)
        assert set(fold.train).isdisjoint(fold.embargoed)
        assert all(i + horizon[i] < fold.test_start for i in fold.train)


# ─────────────────────────────────────────────────────────────────────────────
# A9, A10, A13 — la purga es literal y con h = 0 es un no-op medido
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_purge_removes_exactly_the_invading_samples() -> None:
    sessions = business_sessions(30)
    horizon = [2] * 30
    plan = walk_forward_splits(
        sessions, label_horizon=horizon, n_splits=2, test_size=5, embargo_sessions=0
    )
    # fold 0: test_start = 20 -> se purga i si i + 2 >= 20 -> {18, 19}
    assert plan.folds[0].test_start == 20
    assert plan.folds[0].purged == (18, 19)
    assert plan.folds[0].train == tuple(range(18))
    # fold 1: test_start = 25 -> se purga i si i + 2 >= 25 -> {23, 24}
    assert plan.folds[1].test_start == 25
    assert plan.folds[1].purged == (23, 24)
    assert plan.folds[1].train == tuple(range(23))
    assert plan.purge_total == 4
    assert plan.exclusions_are_no_op is False


def test_a9_purge_is_the_literal_formula_not_a_tail_rule() -> None:
    sessions = business_sessions(20)
    # horizonte irregular: solo algunas muestras invaden el test
    horizon = [0] * 20
    horizon[14] = 6  # 14 + 6 = 20 >= 20 -> purgada
    horizon[18] = 1  # 18 + 1 = 19 < 20 -> NO purgada
    plan = walk_forward_splits(
        sessions, label_horizon=horizon, n_splits=1, test_size=5, embargo_sessions=0
    )
    assert plan.folds[0].test_start == 15
    assert plan.folds[0].purged == (14,)
    assert 13 in plan.folds[0].train and 14 not in plan.folds[0].train


def test_a10_with_the_horizon_of_10_the_purge_is_a_measured_no_op() -> None:
    sessions = business_sessions(50)
    plan = walk_forward_splits(
        sessions, label_horizon=[0] * 50, n_splits=5, test_size=6, embargo_sessions=3
    )
    assert all(fold.purged == () for fold in plan.folds)
    assert plan.purge_total == 0
    assert plan.embargo_in_train_total == 0
    assert plan.exclusions_are_no_op is True


# ─────────────────────────────────────────────────────────────────────────────
# A11, A12 — semántica de t1 y medias sesiones
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_docstring_declares_the_t1_semantics() -> None:
    docstring = splits.__doc__ or ""
    for needle in ("t0", "t1", "close_utc", "#10", "cierre de la misma sesion", "regla 6"):
        assert needle in docstring, needle


def test_a12_half_sessions_are_one_more_position_without_no_constante() -> None:
    sessions = business_sessions(30)
    # #10 declara labelled_half_sessions = 21 y h = 0 **también** en ellas.
    plan = walk_forward_splits(
        sessions, label_horizon=[0] * 30, n_splits=3, test_size=4, embargo_sessions=0
    )
    covered = [i for fold in plan.folds for i in fold.test]
    assert len(covered) == len(set(covered)) == 12
    assert sorted(covered + list(plan.uncovered)) == list(range(30))  # una posición más
    docstring = splits.__doc__ or ""
    assert "media sesion" in docstring and "13:00" in docstring
    # el código no asume duración de sesión ni t1 constante
    code = _code_only()
    for literal in ("13:00", "16:00", "HALF_SESSION", "FULL_SESSION", "BARS_PER_HOUR"):
        assert literal not in code, literal


# ─────────────────────────────────────────────────────────────────────────────
# A14, A15, A16, A17 — formas del embargo y disjunción de conjuntos
# ─────────────────────────────────────────────────────────────────────────────
def test_a14_train_purged_and_embargoed_are_disjoint_and_in_range() -> None:
    sessions = business_sessions(40)
    plan = walk_forward_splits(
        sessions, label_horizon=[1] * 40, n_splits=4, test_size=5, embargo_sessions=3
    )
    for fold in plan.folds:
        train, purged, embargoed = set(fold.train), set(fold.purged), set(fold.embargoed)
        assert train.isdisjoint(purged)
        assert train.isdisjoint(embargoed)
        assert purged.isdisjoint(embargoed)
        assert all(i < fold.test_start for i in train)
        assert all(i < fold.test_start for i in purged)
        assert all(fold.test_stop <= i < min(40, fold.test_stop + 3) for i in embargoed)
    assert plan.purge_total == sum(len(f.purged) for f in plan.folds)
    assert plan.embargo_total == sum(len(f.embargoed) for f in plan.folds)


def test_a15_embargo_is_mandatory_and_without_a_default_value() -> None:
    signature = inspect.signature(walk_forward_splits)
    parameter = signature.parameters["embargo_sessions"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    docstring = splits.__doc__ or ""
    assert "11.1" in docstring and "1-5 %" in docstring  # guia citada, no fijada
    with pytest.raises(TypeError):
        walk_forward_splits(  # pyright: ignore[reportCallIssue]
            business_sessions(30), label_horizon=[0] * 30, n_splits=2, test_size=5
        )


def test_a16_embargo_counts_sessions_even_across_a_calendar_gap() -> None:
    # el puente de 12 dias de calendario empieza justo en la posicion 35
    gapped = sessions_with_long_gap(before=35, gap_days=12, after=5)
    plain = business_sessions(40)
    assert (gapped[35] - gapped[34]).days >= 12
    kwargs: dict[str, Any] = {
        "label_horizon": [0] * 40,
        "n_splits": 2,
        "test_size": 5,
        "embargo_sessions": 4,
    }
    plan_gapped = walk_forward_splits(gapped, **kwargs)
    plan_plain = walk_forward_splits(plain, **kwargs)
    for plan in (plan_gapped, plan_plain):
        assert [fold.test_start for fold in plan.folds] == [30, 35]
        # embargo del fold 0: 4 **sesiones** (35..38), cruzando el puente de calendario
        assert plan.folds[0].embargoed == (35, 36, 37, 38)
        # recortado a [0, n): el ultimo fold no tiene hueco despues
        assert plan.folds[1].embargoed == ()
    # con el embargo a 0 no se marca nada
    zero_embargo = walk_forward_splits(
        gapped, label_horizon=[0] * 40, n_splits=2, test_size=5, embargo_sessions=0
    )
    assert zero_embargo.embargo_total == 0


def test_a17_no_embargoed_position_enters_any_train() -> None:
    sessions = business_sessions(44)
    plan = walk_forward_splits(
        sessions, label_horizon=[0] * 44, n_splits=4, test_size=5, embargo_sessions=3
    )
    for fold in plan.folds:
        assert set(fold.train).isdisjoint(fold.embargoed)
    assert plan.embargo_in_train_total == 0
    assert plan.embargo_total == sum(len(f.embargoed) for f in plan.folds)


# ─────────────────────────────────────────────────────────────────────────────
# A18 — el no-op estructural se declara y se mide
# ─────────────────────────────────────────────────────────────────────────────
def test_a18_no_op_is_declared_with_published_numbers() -> None:
    sessions = business_sessions(50)
    # horizonte de #10 (h = 0) y embargo > 0: el flag es True **medido**
    h_zero = walk_forward_splits(
        sessions, label_horizon=[0] * 50, n_splits=5, test_size=6, embargo_sessions=3
    )
    assert h_zero.purge_total == 0
    assert h_zero.embargo_total == sum(len(fold.embargoed) for fold in h_zero.folds)
    assert h_zero.embargo_in_train_total == 0
    assert h_zero.exclusions_are_no_op is True
    assert "no-op" in (splits.__doc__ or "")
    assert "exclusions_are_no_op" in SOURCE
    # con horizonte > 0 la purga sí trabaja y el flag es False
    h_two = walk_forward_splits(
        sessions, label_horizon=[2] * 50, n_splits=5, test_size=6, embargo_sessions=3
    )
    assert h_two.purge_total > 0
    assert h_two.exclusions_are_no_op is False


# ─────────────────────────────────────────────────────────────────────────────
# A20 — expansivo vs rodante
# ─────────────────────────────────────────────────────────────────────────────
def test_a20_expansive_and_rolling_trains_differ_where_they_should() -> None:
    sessions = business_sessions(30)
    horizon = [0] * 30
    expansive = walk_forward_splits(
        sessions, label_horizon=horizon, n_splits=2, test_size=5, embargo_sessions=0
    )
    rolling = walk_forward_splits(
        sessions,
        label_horizon=horizon,
        n_splits=2,
        test_size=5,
        embargo_sessions=0,
        max_train_size=4,
    )
    assert expansive.folds[0].train == tuple(range(20))
    assert rolling.folds[0].train == (16, 17, 18, 19)
    assert rolling.folds[1].train == (21, 22, 23, 24)
    assert all(fold.train == expansive.folds[fold.index].train[-4:] for fold in rolling.folds)
    assert expansive.folds[0].train != rolling.folds[0].train
    assert rolling.inputs["max_train_size"] == 4


# ─────────────────────────────────────────────────────────────────────────────
# A21, A23 — determinismo y hash reproducible
# ─────────────────────────────────────────────────────────────────────────────
def test_a21_identical_inputs_give_structurally_equal_plans() -> None:
    sessions = business_sessions(33)
    kwargs: dict[str, Any] = {
        "label_horizon": [1] * 33,
        "n_splits": 3,
        "test_size": 5,
        "embargo_sessions": 2,
        "max_train_size": 10,
    }
    first = walk_forward_splits(sessions, **kwargs)
    second = walk_forward_splits(sessions, **kwargs)
    assert first == second
    assert first.plan_sha256 == second.plan_sha256
    assert isinstance(second, SplitPlan)


def _expected_hash(*, n: int, k: int, t: int, e: int, m: int | None, horizon: Sequence[int]) -> str:
    payload = "\n".join(
        (
            f"n_sessions={n}",
            f"n_splits={k}",
            f"test_size={t}",
            f"embargo_sessions={e}",
            f"max_train_size={'none' if m is None else m}",
            f"label_horizon={','.join(str(h) for h in horizon)}",
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_a23_hash_is_the_documented_format_and_changes_with_every_parameter() -> None:
    sessions = business_sessions(30)
    horizon = [0] * 30
    base = walk_forward_splits(
        sessions, label_horizon=horizon, n_splits=2, test_size=5, embargo_sessions=2
    )
    assert base.plan_sha256 == _expected_hash(n=30, k=2, t=5, e=2, m=None, horizon=horizon)
    assert len(base.plan_sha256) == 64 and all(c in "0123456789abcdef" for c in base.plan_sha256)
    variants = {
        "n_sessions": walk_forward_splits(
            business_sessions(31),
            label_horizon=[0] * 31,
            n_splits=2,
            test_size=5,
            embargo_sessions=2,
        ),
        "n_splits": walk_forward_splits(
            sessions, label_horizon=horizon, n_splits=3, test_size=5, embargo_sessions=2
        ),
        "test_size": walk_forward_splits(
            sessions, label_horizon=horizon, n_splits=2, test_size=4, embargo_sessions=2
        ),
        "embargo_sessions": walk_forward_splits(
            sessions, label_horizon=horizon, n_splits=2, test_size=5, embargo_sessions=3
        ),
        "max_train_size": walk_forward_splits(
            sessions,
            label_horizon=horizon,
            n_splits=2,
            test_size=5,
            embargo_sessions=2,
            max_train_size=10,
        ),
        "label_horizon": walk_forward_splits(
            sessions,
            label_horizon=[1, *horizon[1:]],
            n_splits=2,
            test_size=5,
            embargo_sessions=2,
        ),
    }
    for name, plan in variants.items():
        assert plan.plan_sha256 != base.plan_sha256, name


# ─────────────────────────────────────────────────────────────────────────────
# A22, A27, A28 — pureza, entrada genérica y librerías prohibidas
# ─────────────────────────────────────────────────────────────────────────────
def test_a22_module_is_pure_and_uses_only_the_standard_library() -> None:
    allowed = {"__future__", "hashlib", "collections.abc", "dataclasses", "datetime", "typing"}
    assert _imported_modules() <= allowed, _imported_modules() - allowed
    code = _code_only()
    for forbidden in (
        "datetime.now",
        "utcnow",
        "date.today",
        "random",
        "numpy",
        "polars",
        "sklearn",
        "mlfinlab",
        "Store(",
        "MarketCalendar(",
        "config/",
        "open(",
        "write_text",
    ):
        assert forbidden not in code, forbidden


def test_a27_list_tuple_and_ndarray_give_the_same_plan() -> None:
    sessions = business_sessions(25)
    horizon = [1] * 25
    kwargs: dict[str, Any] = {
        "label_horizon": horizon,
        "n_splits": 2,
        "test_size": 5,
        "embargo_sessions": 2,
    }
    from_list = walk_forward_splits(sessions, **kwargs)
    from_tuple = walk_forward_splits(tuple(sessions), **kwargs)
    array = np.array(sessions, dtype=object)
    from_array = walk_forward_splits(cast("Sequence[date]", array), **kwargs)
    assert from_list == from_tuple == from_array
    assert from_list.plan_sha256 == from_array.plan_sha256


def test_a28_mlfinlab_sklearn_and_shuffled_splitters_are_absent() -> None:
    code = _code_only()
    for forbidden in ("mlfinlab", "sklearn", "KFold", "train_test_split", "TimeSeriesSplit"):
        assert forbidden not in code, forbidden
    assert "mlfinlab" not in _imported_modules()


# ─────────────────────────────────────────────────────────────────────────────
# A24, A25, A26 — errores tipados
# ─────────────────────────────────────────────────────────────────────────────
INVALID_CASES: dict[str, dict[str, Any]] = {
    "n_splits_cero": {"n_splits": 0, "test_size": 5, "embargo_sessions": 0},
    "n_splits_negativo": {"n_splits": -3, "test_size": 5, "embargo_sessions": 0},
    "test_size_cero": {"n_splits": 2, "test_size": 0, "embargo_sessions": 0},
    "test_size_negativo": {"n_splits": 2, "test_size": -1, "embargo_sessions": 0},
    "embargo_negativo": {"n_splits": 2, "test_size": 5, "embargo_sessions": -1},
    "max_train_cero": {
        "n_splits": 2,
        "test_size": 5,
        "embargo_sessions": 0,
        "max_train_size": 0,
    },
    "max_train_negativo": {
        "n_splits": 2,
        "test_size": 5,
        "embargo_sessions": 0,
        "max_train_size": -4,
    },
}


@pytest.mark.parametrize("name", sorted(INVALID_CASES))
def test_a24_invalid_numeral_parameters_raise_a_typed_error(name: str) -> None:
    sessions = business_sessions(30)
    with pytest.raises(InvalidSplitParameterError):
        walk_forward_splits(sessions, label_horizon=[0] * 30, **INVALID_CASES[name])


def test_a24_horizon_length_and_sign_and_ordering_are_validated() -> None:
    sessions = business_sessions(30)
    with pytest.raises(InvalidSplitParameterError):
        walk_forward_splits(
            sessions, label_horizon=[0] * 29, n_splits=2, test_size=5, embargo_sessions=0
        )
    with pytest.raises(InvalidSplitParameterError):
        walk_forward_splits(
            sessions,
            label_horizon=[0] * 29 + [-1],
            n_splits=2,
            test_size=5,
            embargo_sessions=0,
        )
    with pytest.raises(InvalidSplitParameterError):
        walk_forward_splits(
            [*sessions[:29], sessions[28]],
            label_horizon=[0] * 30,
            n_splits=2,
            test_size=5,
            embargo_sessions=0,
        )
    with pytest.raises(InvalidSplitParameterError):
        walk_forward_splits(
            list(reversed(sessions)),
            label_horizon=[0] * 30,
            n_splits=2,
            test_size=5,
            embargo_sessions=0,
        )


def test_a25_too_few_sessions_raise_with_the_sizes() -> None:
    sessions = business_sessions(30)
    with pytest.raises(InsufficientSessionsError, match="30"):
        walk_forward_splits(  # exactamente k * T == n
            sessions, label_horizon=[0] * 30, n_splits=3, test_size=10, embargo_sessions=0
        )
    with pytest.raises(InsufficientSessionsError):
        walk_forward_splits(  # k * T > n
            sessions, label_horizon=[0] * 30, n_splits=4, test_size=9, embargo_sessions=0
        )
    with pytest.raises(InsufficientSessionsError):
        walk_forward_splits([], label_horizon=[], n_splits=1, test_size=1, embargo_sessions=0)


def test_a26_an_empty_train_after_purge_raises_instead_of_returning_a_fold() -> None:
    sessions = business_sessions(6)
    with pytest.raises(InsufficientSessionsError, match="fold 0"):
        walk_forward_splits(
            sessions,
            label_horizon=[2, 0, 0, 0, 0, 0],
            n_splits=1,
            test_size=5,
            embargo_sessions=0,
        )


def test_a26_an_empty_train_after_rolling_purge_raises() -> None:
    sessions = business_sessions(7)
    # test_start = 2; h = 3 purga {0, 1} (0 + 3 >= 2 y 1 + 3 >= 2) -> train vacio
    with pytest.raises(InsufficientSessionsError):
        walk_forward_splits(
            sessions,
            label_horizon=[3, 3, 0, 0, 0, 0, 0],
            n_splits=1,
            test_size=5,
            embargo_sessions=0,
            max_train_size=1,
        )


# ─────────────────────────────────────────────────────────────────────────────
# A29, A31, A32, A33 — tamaño del núcleo y fronteras declaradas
# ─────────────────────────────────────────────────────────────────────────────
def test_a29_core_loop_is_short() -> None:
    function = next(
        node
        for node in ast.walk(ast.parse(SOURCE))
        if isinstance(node, ast.FunctionDef) and node.name == "walk_forward_splits"
    )
    loop = next(node for node in ast.walk(function) if isinstance(node, ast.For))
    lines = SOURCE.splitlines()[loop.lineno - 1 : loop.end_lineno]
    code = [line for line in lines if line.strip() and not line.strip().startswith("#")]
    assert len(code) <= 50, len(code)


def test_a31_docstring_defines_the_vocabulary_and_the_frontiers() -> None:
    docstring = splits.__doc__ or ""
    for needle in (
        "purga",
        "embargo",
        "sesiones",
        "expansivo",
        "rodante",
        "#13",
        "#16",
        "#67",
        "#68",
        "plan.md",
        "tech_stack.md",
    ):
        assert needle in docstring, needle


def test_a32_does_not_do_is_machine_readable_and_covers_the_frontiers() -> None:
    ids = {item["id"] for item in splits.SPLITS_DOES_NOT_DO}
    assert {"no_es_el_motor", "no_reserva_holdout", "no_usa_cpcv"} <= ids
    for item in splits.SPLITS_DOES_NOT_DO:
        assert item["issue"].startswith("#")
        assert item["statement"]
    statements = " ".join(item["statement"] for item in splits.SPLITS_DOES_NOT_DO).lower()
    assert "entrena" in statements and "holdout" in statements
    docstring = splits.__doc__ or ""
    assert "#24" in docstring and "#68" in docstring


def test_a33_package_docstring_mentions_the_new_module() -> None:
    import cfdtrader.backtest as backtest

    docstring = backtest.__doc__ or ""
    assert "splits" in docstring
    assert "walk-forward" in docstring
    assert "costs" in docstring  # la descripción del paquete sigue siendo coherente


# ─────────────────────────────────────────────────────────────────────────────
# A34 — la sesión de tests no escribe en data/ (lo vigila conftest.py)
# ─────────────────────────────────────────────────────────────────────────────
def test_a34_generating_a_plan_writes_nothing(tmp_path: Path) -> None:
    before = sorted(REPO_ROOT.glob("data/**/*"))
    plan = walk_forward_splits(
        business_sessions(20),
        label_horizon=[0] * 20,
        n_splits=2,
        test_size=4,
        embargo_sessions=1,
    )
    assert plan.n_sessions == 20
    assert sorted(REPO_ROOT.glob("data/**/*")) == before
    assert list(tmp_path.iterdir()) == []  # el módulo no escribe ni en un directorio temporal
