"""Tests de la orquestación con LangGraph (#36).

Dos artefactos verificables: el grafo **coordina** nodos expertos en paralelo y los mergea; y el
**gate sigue siendo una función pura** que se ejecuta **sin arrancar LangGraph** (la garantía de
que el backtest sigue siendo retrotesteable). Ninguna prueba abre red.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Final

import pytest

from cfdtrader.agents.event_calendar import EventCalendarInputError
from cfdtrader.data.calendar import EASTERN, load_calendar
from cfdtrader.data.store import Store
from cfdtrader.orchestration import graph as graph_module
from cfdtrader.orchestration.graph import (
    EXPERT_NODES,
    FAN_IN_NOTICE,
    MERGE_NODE,
    PipelineState,
    build_pipeline_graph,
    run_pipeline,
)

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
SRC_ROOT: Final = REPO_ROOT / "src" / "cfdtrader"

#: Instante y sesión declarados (nunca del reloj).
SESSION: Final = date(2026, 9, 17)
MOMENT: Final = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
FETCHED: Final = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def _store(root: Path) -> Store:
    """Un almacén mínimo con una publicación macro y un resultado de mega-cap de ese día."""
    observed = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
    store = Store(root)
    store.append(
        "raw",
        "macro",
        [
            {
                "source": "fred",
                "series_id": "CPIAUCSL",
                "as_of": date(2026, 9, 1),
                "fetched_at": FETCHED,
                "published_at": datetime.combine(SESSION, time(8, 30), tzinfo=EASTERN).astimezone(
                    UTC
                ),
                "value": 1.0,
            }
        ],
    )
    store.append(
        "raw",
        "earnings",
        [
            {
                "source": "yfinance",
                "series_id": f"NVDA@{SESSION.isoformat()}",
                "as_of": observed,
                "fetched_at": FETCHED,
                "published_at": observed,
                "name": "NVIDIA",
                "event_date": SESSION,
                "moment": "amc",
                "certainty": "confirmed",
                "observed_at": observed,
            }
        ],
    )
    return store


def test_36_the_pipeline_state_is_typed_with_reducers() -> None:
    """El estado es Pydantic y sus listas se mergean con `operator.add` (paralelo sin carreras)."""
    state = PipelineState(session=SESSION, as_of=MOMENT)
    assert state.calendar_events == [] and state.publications == [] and state.earnings == []
    assert list(EXPERT_NODES) == [
        "calendar",
        "macro_publications",
        "mega_cap_earnings",
    ]
    assert MERGE_NODE == "merge"


def test_36_the_graph_merges_the_three_experts_and_reaches_the_fan_in(tmp_path: Path) -> None:
    """Los tres expertos corren y el *fan-in* los mergea en un estado tipado."""
    store = _store(tmp_path)
    graph = build_pipeline_graph(calendar=load_calendar(), store=store, moment=MOMENT)

    state = run_pipeline(graph, session=SESSION, as_of=MOMENT)

    assert state.session == SESSION
    assert any("CPIAUCSL" in item for item in state.publications)
    assert any("NVDA" in item for item in state.earnings)
    assert state.notices.count(FAN_IN_NOTICE) == 1


def test_36_a_broken_expert_degrades_to_a_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un experto que falla no tumba el grafo: deja su aviso y el resto sigue."""

    def _boom(*args: object, **kwargs: object) -> object:
        raise EventCalendarInputError("calendario roto")

    monkeypatch.setattr(graph_module, "calendar_signal", _boom)
    store = _store(tmp_path)
    graph = build_pipeline_graph(calendar=load_calendar(), store=store, moment=MOMENT)

    state = run_pipeline(graph, session=SESSION, as_of=MOMENT)

    assert any("calendario no calculable" in item for item in state.notices)
    assert FAN_IN_NOTICE in state.notices


def test_36_the_gate_runs_without_starting_langgraph() -> None:
    """El gate y el backtest se importan y ejecutan **sin** arrancar LangGraph (§4.10)."""
    script = (
        "import sys\n"
        "import cfdtrader.decision.gate as gate\n"
        "import cfdtrader.backtest.engine as engine\n"
        "assert 'langgraph' not in sys.modules, 'algo arranco LangGraph en el camino critico'\n"
        "print('ok', gate.GateStatus.RECOMMENDATION.value, bool(engine.Direction.NOTHING))\n"
    )
    completed = subprocess.run(  # noqa: S603 - el interprete de la sesion, guion fijo
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


def test_36_langgraph_is_imported_by_one_module_only() -> None:
    """Solo `orchestration/graph.py` importa `langgraph`: el resto del núcleo no lo arrastra."""
    importers: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "langgraph" or name.startswith("langgraph.") for name in names):
                importers.append(str(path.relative_to(REPO_ROOT)))
                break
    assert importers == ["src/cfdtrader/orchestration/graph.py"], importers
