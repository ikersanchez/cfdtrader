"""Tests de la reserva del *holdout* final intocable (`plan.md` §11.4) — tarea #68.

Un test por criterio, ``test_aN_...``. El tramo reservado es una **decision declarada**, no
una medicion: los casos de aqui usan decisiones sinteticas para poder mover el universo a
voluntad, y uno comprueba la decision real contra el universo del almacen.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Final

import pytest

from cfdtrader.backtest import holdout
from cfdtrader.backtest.splits import walk_forward_splits

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODULE: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "backtest" / "holdout.py"

#: Doce sesiones seguidas: las nueve primeras usables, las tres ultimas reservadas.
SESSIONS: Final[tuple[date, ...]] = tuple(date(2025, 1, day) for day in range(1, 13))

#: Decision sintetica equivalente a la real: reserva el final de la muestra.
SYNTHETIC: Final[holdout.HoldoutDecision] = holdout.HoldoutDecision(
    first_session=date(2025, 1, 10),
    last_session=date(2025, 1, 12),
    n_sessions=3,
    reason="decision sintetica del test",
    decided_on=date(2026, 10, 6),
    decided_by="test",
)


def _decision_on(sessions: tuple[date, ...], *, count: int) -> holdout.HoldoutDecision:
    """Decision sintetica que reserva las ``count`` ultimas sesiones de ``sessions``."""
    return holdout.HoldoutDecision(
        first_session=sessions[-count],
        last_session=sessions[-1],
        n_sessions=count,
        reason="decision sintetica del test",
        decided_on=date(2026, 10, 6),
        decided_by="test",
    )


# ─────────────────────────────────────────────────────────────────────────────
# A1 · La decision esta declarada y no se consulta el reloj
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_decision_is_declared_with_its_period_and_its_motive() -> None:
    """El tramo, el motivo y la fecha son un literal declarado, no una medicion (A1)."""
    decision = holdout.HOLDOUT_DECISION

    assert (decision.first_session, decision.last_session) == (
        date(2025, 9, 18),
        date(2026, 9, 17),
    )
    assert decision.n_sessions == 251, "los ultimos 12 meses del universo etiquetado"
    assert decision.reason and decision.decided_by
    assert decision.decided_on == date(2026, 10, 6)
    assert decision.contains(date(2025, 9, 18)) and decision.contains(date(2026, 9, 17))
    assert not decision.contains(date(2025, 9, 17))


def test_a1_the_module_does_not_read_the_clock() -> None:
    """El "hoy" de la decision es un campo declarado: el modulo no consulta el reloj (A5)."""
    source = MODULE.read_text(encoding="utf-8")

    assert "datetime.now" not in source
    assert "date.today" not in source


# ─────────────────────────────────────────────────────────────────────────────
# A2 · El tramo es el final y nunca entra en la parte usable
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_tramo_is_the_final_and_never_enters_the_usable_part() -> None:
    """``reserve`` parte la muestra y el tramo reservado no aparece en ``usable`` (A2)."""
    sample = holdout.reserve(SESSIONS, decision=SYNTHETIC)

    assert sample.reserved == (date(2025, 1, 10), date(2025, 1, 11), date(2025, 1, 12))
    assert len(sample.usable) == len(SESSIONS) - SYNTHETIC.n_sessions
    assert not set(sample.usable) & set(sample.reserved)
    assert sample.usable[-1] < sample.reserved[0]
    assert len(sample.usable) + len(sample.reserved) == len(SESSIONS)


def test_a2_the_walk_forward_over_the_usable_part_never_touches_the_holdout() -> None:
    """#12 recibe solo ``usable``: ningun fold toca una sesion reservada (A2)."""
    sessions = tuple(date(2025, 1, 1) + timedelta(days=index) for index in range(60))
    decision = _decision_on(sessions, count=5)
    sample = holdout.reserve(sessions, decision=decision)
    plan = walk_forward_splits(
        sample.usable,
        label_horizon=[0] * len(sample.usable),
        n_splits=2,
        test_size=5,
        embargo_sessions=1,
    )

    touched = {
        sample.usable[index]
        for fold in plan.folds
        for index in (*fold.train, *fold.test, *fold.purged, *fold.embargoed)
    }
    assert touched, "el plan tiene que cubrir alguna sesion"
    assert not touched & set(sample.reserved)


# ─────────────────────────────────────────────────────────────────────────────
# A3 · Usar el tramo es un error tipado, no un aviso
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_using_the_reserved_tramo_is_a_typed_error() -> None:
    """``require_usable`` devuelve lo que no cae en el tramo y **falla** con lo que cae (A3)."""
    assert holdout.require_usable(SESSIONS[:9], decision=SYNTHETIC) == SESSIONS[:9]

    with pytest.raises(holdout.HoldoutAccessError, match="holdout final intocable"):
        holdout.require_usable(SESSIONS, decision=SYNTHETIC)
    with pytest.raises(holdout.HoldoutAccessError, match="1 sesion"):
        holdout.require_usable((SESSIONS[0], SESSIONS[-1]), decision=SYNTHETIC)
    assert issubclass(holdout.HoldoutAccessError, holdout.HoldoutError)


# ─────────────────────────────────────────────────────────────────────────────
# A4 · El informe declara el recuento, el rango y que queda fuera
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_the_report_declares_the_counts_the_range_and_what_stays_out() -> None:
    """El payload dice cuantas sesiones se reservan, su rango y que queda fuera (A4)."""
    report = holdout.holdout_report(SESSIONS, decision=SYNTHETIC)

    assert report["n_sessions_total"] == len(SESSIONS)
    assert report["n_reserved"] == SYNTHETIC.n_sessions
    assert report["n_usable"] == report["n_sessions_total"] - report["n_reserved"]
    assert report["reserved_range"] == ["2025-01-10", "2025-01-12"]
    assert report["usable_range"] == ["2025-01-01", "2025-01-09"]
    assert "fuera de" in str(report["excluded_from_folds"])
    decision = report["decision"]
    assert isinstance(decision, dict)
    assert decision["decided_by"] == "test" and decision["reason"]


# ─────────────────────────────────────────────────────────────────────────────
# A5 · Determinista, sin recolocarse con la muestra
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_it_is_deterministic_and_the_reserve_never_repositions_itself() -> None:
    """Misma entrada, mismo payload; y un universo que cambia **para** en vez de recolocarse."""
    first = holdout.holdout_report(SESSIONS, decision=SYNTHETIC)
    assert first == holdout.holdout_report(SESSIONS, decision=SYNTHETIC)

    # una sesion reservada que desaparece: el recuento ya no cuadra
    without = (*SESSIONS[:10], SESSIONS[11])
    with pytest.raises(holdout.HoldoutMismatchError, match="no se recoloca sola"):
        holdout.reserve(without, decision=SYNTHETIC)
    # una sesion despues del tramo: el holdout dejaria de ser el final
    with pytest.raises(holdout.HoldoutMismatchError, match="hay sesiones usables"):
        holdout.reserve((*SESSIONS, date(2025, 1, 13)), decision=SYNTHETIC)
    # desordenada, repetida o vacia: error tipado, nunca una particion silenciosa
    with pytest.raises(holdout.HoldoutMismatchError, match="crecientes"):
        holdout.reserve((SESSIONS[1], SESSIONS[0]), decision=SYNTHETIC)
    with pytest.raises(holdout.HoldoutMismatchError, match="crecientes"):
        holdout.reserve((SESSIONS[0], SESSIONS[0]), decision=SYNTHETIC)
    with pytest.raises(holdout.HoldoutMismatchError, match="vacia"):
        holdout.reserve((), decision=SYNTHETIC)
    # y una reserva que se comiera la muestra entera
    with pytest.raises(holdout.HoldoutMismatchError, match="muestra entera"):
        holdout.reserve(SESSIONS[:3], decision=SYNTHETIC)
    # un rango declarado que no coincide con el tramo que el universo tiene de verdad
    misaligned = holdout.HoldoutDecision(
        first_session=date(2025, 1, 11),
        last_session=date(2025, 1, 13),
        n_sessions=2,
        reason="rango desalineado",
        decided_on=date(2026, 10, 6),
        decided_by="test",
    )
    with pytest.raises(holdout.HoldoutMismatchError, match="no coincide con el rango declarado"):
        holdout.reserve(SESSIONS, decision=misaligned)
