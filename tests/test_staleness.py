"""Tests de la guardia de obsolescencia ``delivery/staleness.py`` (#40): A1-A4, A7 y A10-A11.

Los festivos y las medias sesiones son los **reales** (``2026-11-26``, Accion de Gracias, y
``2026-11-27``, media sesion) del calendario de ``config/calendar.yaml``: el modulo no tiene
tabla propia y estas pruebas lo demuestran comparando con ``MarketCalendar.session()``. Nada de
esto toca el ``data/`` ni el ``runs/`` del repositorio (lo blinda ``tests/conftest.py``): el
historial de ejecuciones se siembra con la capa de #39 (``build_decision`` + ``Journal``) en
``tmp_path``.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final, cast

from cfdtrader.data.calendar import MarketCalendar, load_calendar
from cfdtrader.decision.gate import GateStatus
from cfdtrader.delivery import staleness
from cfdtrader.journal import decision_log
from cfdtrader.journal.decision_log import Journal, build_decision

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

#: El fuente y su AST: la ausencia de reloj, de red y de escritura se mide en el arbol, no en la
#: prosa (el docstring del modulo **declara** por escrito que no las usa, y un `grep` textual se
#: encontraria a si mismo).
SOURCE: Final[str] = Path(staleness.__file__).read_text(encoding="utf-8")
TREE: Final[ast.Module] = ast.parse(SOURCE)

#: Commit inyectado por el llamante (el modulo no lee git, #112).
GIT_COMMIT: Final[str] = "5" * 40

#: Sesiones declaradas: nunca del reloj.
THURSDAY: Final[date] = date(2026, 9, 17)
FRIDAY: Final[date] = date(2026, 9, 18)
SATURDAY: Final[date] = date(2026, 9, 19)
MONDAY: Final[date] = date(2026, 9, 21)
SNAPSHOT: Final[date] = date(2026, 9, 16)
THANKSGIVING: Final[date] = date(2026, 11, 26)
BLACK_FRIDAY: Final[date] = date(2026, 11, 27)

#: Las sesiones que cierran la ventana de observacion: viernes, lunes, martes, miercoles y jueves.
OBSERVATION_SESSIONS: Final[tuple[date, ...]] = (
    THURSDAY,
    FRIDAY,
    MONDAY,
    date(2026, 9, 22),
    date(2026, 9, 23),
    date(2026, 9, 24),
)


def _moment(day: date) -> datetime:
    """Mediodia UTC de esa fecha: el instante entra declarado, nunca del reloj."""
    return datetime(day.year, day.month, day.day, 12, 0, tzinfo=UTC)


def _calendar() -> MarketCalendar:
    """El calendario real del proyecto (festivos NYSE de la libreria ``holidays``)."""
    return load_calendar()


def _seed(journal_root: Path, days: Sequence[date]) -> None:
    """Escribe una fila de ``journal.decisions`` por sesion, con la capa de #39."""
    for day in days:
        payload = build_decision(
            trade_date=day,
            as_of=_moment(day),
            features_version="features_de_prueba",
            model_version="modelo_de_prueba",
            git_commit=GIT_COMMIT,
            report_text="informe sembrado en la prueba (#40)",
            status=GateStatus.NO_RECOMMENDATION_STALE_DATA,
        )
        Journal(journal_root).write("decisions", payload)


# ─────────────────────────────────────────────────────────────────────────────
# A1: el modulo no lee el reloj, no usa red, no lanza procesos y no escribe
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_module_has_no_clock_no_network_no_process_and_writes_nothing() -> None:
    """A1: ni reloj, ni red, ni procesos, ni escritura, y solo los imports declarados (AST)."""
    calls = {ast.unparse(node.func) for node in ast.walk(TREE) if isinstance(node, ast.Call)}
    assert not [name for name in calls if name.endswith((".now", ".today", ".time"))]
    assert not [
        name
        for name in calls
        if name == "open" or name.endswith(("subprocess.run", "os.system", "os.popen"))
    ]
    attributes = {node.attr for node in ast.walk(TREE) if isinstance(node, ast.Attribute)}
    assert attributes.isdisjoint({"write_text", "write_bytes", "mkdir", "unlink"})

    imported = {
        alias.name
        for node in ast.walk(TREE)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imported == set()  # ni `yfinance`, ni `requests`, ni `urllib`: nada suelto
    modules = {
        node.module
        for node in ast.walk(TREE)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert modules <= {
        "__future__",
        "collections.abc",
        "dataclasses",
        "datetime",
        "enum",
        "itertools",
        "pathlib",
        "typing",
        "cfdtrader.data.calendar",
        "cfdtrader.journal.decision_log",
    }


# ─────────────────────────────────────────────────────────────────────────────
# A2: la clausura la decide el calendario, no una tabla nueva
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_calendar_decides_the_closure() -> None:
    """A2: el motivo es el de ``MarketCalendar.session()``; el modulo no declara festivos."""
    calendar = _calendar()
    thanksgiving = staleness.market_closure(as_of=_moment(THANKSGIVING), calendar=calendar)
    assert thanksgiving == calendar.session(THANKSGIVING).reason
    assert thanksgiving is not None
    assert "festivo" in thanksgiving and "Thanksgiving" in thanksgiving

    assert staleness.market_closure(as_of=_moment(SATURDAY), calendar=calendar) == "fin de semana"
    assert staleness.market_closure(as_of=_moment(THURSDAY), calendar=calendar) is None
    # Media sesion real (el dia despues de Accion de Gracias): es sesion, y el gate la bloquea.
    assert calendar.session(BLACK_FRIDAY).is_half_day is True
    assert staleness.market_closure(as_of=_moment(BLACK_FRIDAY), calendar=calendar) is None
    for literal in ("Thanksgiving", "Christmas", "Memorial", "Independence", "Labor Day"):
        assert literal not in SOURCE, f"el modulo no declara festivos: encontro {literal!r}"


# ─────────────────────────────────────────────────────────────────────────────
# A4: el veredicto de frescura
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_the_freshness_verdicts_are_explicit() -> None:
    """A4: ``proceed``, ``missing_previous_close``, ``snapshot_ahead`` y ``market_closed``."""
    calendar = _calendar()
    fresh = staleness.session_guard(
        as_of=_moment(THURSDAY), calendar=calendar, snapshot_session=SNAPSHOT
    )
    assert fresh.verdict is staleness.GuardVerdict.PROCEED
    assert fresh.blocks is False
    assert fresh.previous_session == SNAPSHOT
    assert fresh.is_session is True
    assert fresh.is_half_session is False
    assert fresh.closure_reason is None
    assert SNAPSHOT.isoformat() in fresh.message

    late = staleness.session_guard(
        as_of=_moment(MONDAY), calendar=calendar, snapshot_session=SNAPSHOT
    )
    assert late.verdict is staleness.GuardVerdict.MISSING_PREVIOUS_CLOSE
    assert late.blocks is True
    for token in (MONDAY.isoformat(), FRIDAY.isoformat(), SNAPSHOT.isoformat()):
        assert token in late.message, f"el motivo tiene que nombrar {token}"

    empty = staleness.session_guard(
        as_of=_moment(THURSDAY), calendar=calendar, snapshot_session=None
    )
    assert empty.verdict is staleness.GuardVerdict.MISSING_PREVIOUS_CLOSE
    assert SNAPSHOT.isoformat() in empty.message

    ahead = staleness.session_guard(
        as_of=_moment(SNAPSHOT), calendar=calendar, snapshot_session=SNAPSHOT
    )
    assert ahead.verdict is staleness.GuardVerdict.SNAPSHOT_AHEAD
    assert ahead.blocks is True

    closed = staleness.session_guard(
        as_of=_moment(THANKSGIVING), calendar=calendar, snapshot_session=date(2026, 11, 25)
    )
    assert closed.verdict is staleness.GuardVerdict.MARKET_CLOSED
    assert closed.is_session is False
    reason = cast("str", closed.closure_reason)
    assert "festivo" in reason and "Thanksgiving" in reason
    assert "el cierre de la sesion anterior" not in closed.message


# ─────────────────────────────────────────────────────────────────────────────
# A7: el contador de observacion, derivado del diario
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_the_observation_window_comes_from_the_journal(tmp_path: Path) -> None:
    """A7: 5 sesiones de observacion tras un hueco de mas de una semana; el diario es la fuente."""
    calendar = _calendar()
    root = tmp_path / "journal"
    assert staleness.execution_dates(root) == ()
    assert (
        staleness.observation_sessions_remaining(executions=(), calendar=calendar, session=THURSDAY)
        == 0
    )
    # El umbral declarado es "mas de una semana" (7 dias naturales) y sin huecos no hay ventana.
    assert (
        staleness.observation_sessions_remaining(
            executions=(date(2026, 9, 10),), calendar=calendar, session=THURSDAY
        )
        == 0
    )
    assert (
        staleness.observation_sessions_remaining(
            executions=(date(2026, 9, 9),), calendar=calendar, session=THURSDAY
        )
        == 5
    )
    assert (
        staleness.observation_sessions_remaining(
            executions=(SNAPSHOT,), calendar=calendar, session=THURSDAY
        )
        == 0
    )

    _seed(root, (date(2026, 9, 1), date(2026, 9, 2)))
    assert staleness.execution_dates(root) == (date(2026, 9, 1), date(2026, 9, 2))

    for position, day in enumerate(OBSERVATION_SESSIONS):
        expected = 5 - position
        guard = staleness.session_guard(
            as_of=_moment(day),
            calendar=calendar,
            snapshot_session=SNAPSHOT,
            executions=staleness.execution_dates(root),
        )
        assert guard.observation_sessions_remaining == expected, day
        if expected:
            notice = cast("str", guard.reincorporation_notice)
            assert "reincorporacion" in notice and f"{expected} de 5" in notice
        else:
            assert guard.reincorporation_notice is None
        # Cada corrida deja su fila: asi avanza la ventana. Del viernes al lunes la ventana
        # consume **una** sesion, no tres: el fin de semana no la gasta.
        _seed(root, (day,))
    assert staleness.execution_dates(root)[-1] == OBSERVATION_SESSIONS[-1]

    # Un dia de mercado cerrado no tiene contador: no hay sesion que revalidar.
    thanksgiving = staleness.session_guard(
        as_of=_moment(THANKSGIVING),
        calendar=calendar,
        snapshot_session=date(2026, 11, 25),
        executions=(date(2026, 11, 3),),
    )
    assert thanksgiving.observation_sessions_remaining == 0
    assert thanksgiving.reincorporation_notice is None
    assert thanksgiving.last_execution == date(2026, 11, 3)
    assert thanksgiving.absence_days == 23


# ─────────────────────────────────────────────────────────────────────────────
# A10: reuso por import
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_the_journal_is_read_through_the_imported_reader(tmp_path: Path) -> None:
    """A10: el diario se lee con ``read_decisions`` de #39; el modulo no abre los JSON a mano."""
    imported = {
        alias.name
        for node in ast.walk(TREE)
        if isinstance(node, ast.ImportFrom) and node.module == "cfdtrader.journal.decision_log"
        for alias in node.names
    }
    assert {"Journal", "read_decisions"} <= imported
    assert staleness.read_decisions is decision_log.read_decisions
    defined = {node.name for node in TREE.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assert "read_decisions" not in defined
    calls = {ast.unparse(node.func) for node in ast.walk(TREE) if isinstance(node, ast.Call)}
    assert not [name for name in calls if name.endswith((".read_text", ".glob", ".iterdir"))]

    root = tmp_path / "journal"
    _seed(root, (THURSDAY,))
    assert staleness.execution_dates(root) == (THURSDAY,)


# ─────────────────────────────────────────────────────────────────────────────
# A11: determinismo entre procesos
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_the_guard_is_deterministic_across_hash_seeds() -> None:
    """A11: dos procesos con ``PYTHONHASHSEED`` distinto publican el mismo veredicto byte a byte."""
    script = "\n".join(
        [
            "from datetime import UTC, date, datetime",
            "from cfdtrader.data.calendar import load_calendar",
            "from cfdtrader.delivery import staleness as st",
            "calendar = load_calendar()",
            "executions = (date(2026, 9, 1), date(2026, 9, 2))",
            "for day, snapshot in (",
            "    (date(2026, 9, 17), date(2026, 9, 16)),",
            "    (date(2026, 9, 21), date(2026, 9, 16)),",
            "    (date(2026, 11, 26), date(2026, 11, 25)),",
            "    (date(2026, 11, 27), date(2026, 11, 25)),",
            "):",
            "    guard = st.session_guard(",
            "        as_of=datetime(day.year, day.month, day.day, 12, 0, tzinfo=UTC),",
            "        calendar=calendar, snapshot_session=snapshot, executions=executions,",
            "    )",
            "    print(guard.verdict, guard.observation_sessions_remaining, guard.message)",
        ]
    )
    outputs: list[str] = []
    for seed in ("0", "1"):
        completed = subprocess.run(  # noqa: S603 - el ejecutable es el interprete de la sesion
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            cwd=REPO_ROOT,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout)

    assert outputs[0] == outputs[1]
    for token in ("proceed", "missing_previous_close", "market_closed"):
        assert token in outputs[0]
