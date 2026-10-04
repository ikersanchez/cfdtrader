# pyright: reportArgumentType=false, reportMissingTypeArgument=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""Orquestación del pipeline con LangGraph — tarea #36.

La pregunta que responde este módulo: **¿cómo se coordinan los nodos del día —en paralelo— sin
meter el orquestador en el camino crítico?**

`tech_stack.md` §4.10: LangGraph orquesta el **pipeline** (fetch → nodos expertos en paralelo →
*fan-in* → síntesis), pero el **gate de decisión** (`plan.md` §7.3) es una **función pura de
Python sin dependencias de orquestación**. Si decidir requiriese LangGraph, el backtest estaría
muerto.

Por eso este módulo es el **único** del proyecto que importa ``langgraph``: los expertos que
publica no calculan la dirección ni bloquean nada —solo reunen lo que el día ya sabe (calendario,
publicaciones macro, resultados de mega-caps) y lo dejan en un **estado tipado**—. La decisión la
toma ``decision.gate.evaluate_gate`` llamada **fuera** del grafo, y el backtest importa ese gate
directamente, sin arrancar el orquestador.

El punto de decisión humana de §7.3 **no** usa ``interrupt`` en esta primera versión (decisión
del propietario del 2026-10-03, `plan.md` §19.8): con ejecución manual y a demanda no existe un
grafo pausado al que reanudar; lo ejerce el propietario fuera del grafo.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from datetime import date, datetime
from typing import Annotated, Any, Final

import duckdb
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from cfdtrader.agents.event_calendar import EventCalendarInputError, calendar_signal
from cfdtrader.data.calendar import MarketCalendar
from cfdtrader.data.earnings import EarningsEvent, earnings_on
from cfdtrader.data.macro import MacroPublication, publications_on
from cfdtrader.data.settings import ConfigurationError
from cfdtrader.data.store import Store

__all__ = [
    "EXPERT_NODES",
    "FAN_IN_NOTICE",
    "MERGE_NODE",
    "PipelineState",
    "build_pipeline_graph",
    "run_pipeline",
]

#: Los nodos expertos que corren **en paralelo**. Ninguno decide ni bloquea.
EXPERT_NODES: Final[tuple[str, ...]] = ("calendar", "macro_publications", "mega_cap_earnings")

#: El nodo de *fan-in*: recibe las aristas de todos los expertos.
MERGE_NODE: Final[str] = "merge"

#: Aviso determinista que deja el *fan-in* (prueba de que el merge se ejecutó una sola vez).
FAN_IN_NOTICE: Final[str] = f"fan-in: {len(EXPERT_NODES)} expertos mergeados"

#: Un nodo del grafo: recibe el estado y devuelve su **parte** (LangGraph la mergea con el reducer).
Node = Callable[["PipelineState"], Mapping[str, object]]


class PipelineState(BaseModel):
    """El estado tipado del grafo: lo que el día ya sabe, **sin** dirección ni bloqueos.

    Las listas llevan el reducer ``operator.add``: un experto devuelve su parte y LangGraph la
    **une** con la de los demás, sin que ningún nodo tenga que leer el estado del otro.
    """

    model_config = ConfigDict(extra="forbid")

    session: date
    as_of: datetime
    calendar_events: Annotated[list[str], operator.add] = Field(default_factory=list)
    publications: Annotated[list[str], operator.add] = Field(default_factory=list)
    earnings: Annotated[list[str], operator.add] = Field(default_factory=list)
    notices: Annotated[list[str], operator.add] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Los nodos expertos (en paralelo) y el *fan-in*
# ─────────────────────────────────────────────────────────────────────────────
def _calendar_node(calendar: MarketCalendar, moment: datetime) -> Node:
    """El experto de calendario: publica los eventos del día, **informativos incluidos**."""

    def node(state: PipelineState) -> Mapping[str, object]:
        try:
            signal = calendar_signal(calendar, state.session, as_of=moment)
        except EventCalendarInputError as failure:
            return {"notices": [f"calendario no calculable: {failure}"]}
        return {
            "calendar_events": [
                f"{'evento_bloqueante' if event.blocking else 'evento'}: "
                f"{event.kind.value} | {event.name}"
                for event in signal.events
            ]
        }

    return node


def _macro_node(store: Store, moment: datetime) -> Node:
    """El experto macro: las publicaciones del día con su hora real y su disponibilidad."""

    def node(state: PipelineState) -> Mapping[str, object]:
        try:
            publications: tuple[MacroPublication, ...] = publications_on(
                store=store, session=state.session, as_of=moment
            )
        except (ConfigurationError, duckdb.Error) as failure:
            return {"notices": [f"publicaciones macro no legibles: {failure}"]}
        return {"publications": [f"{item.series_id} | {item.name}" for item in publications]}

    return node


def _earnings_node(store: Store, moment: datetime) -> Node:
    """El experto de mega-caps: los resultados del día, con su certeza (estimada/confirmada)."""

    def node(state: PipelineState) -> Mapping[str, object]:
        try:
            events: tuple[EarningsEvent, ...] = earnings_on(
                store, session=state.session, as_of=moment
            )
        except (ConfigurationError, duckdb.Error) as failure:
            return {"notices": [f"resultados de mega-caps no legibles: {failure}"]}
        return {
            "earnings": [
                f"{event.symbol} | {event.name} | {'confirmed' if event.blocking else 'estimated'}"
                for event in events
            ]
        }

    return node


def _merge_node(state: PipelineState) -> Mapping[str, object]:
    """El *fan-in*: recibe las aristas de todos los expertos y deja un aviso determinista."""
    return {"notices": [FAN_IN_NOTICE]}


# ─────────────────────────────────────────────────────────────────────────────
# Montaje y ejecución
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline_graph(*, calendar: MarketCalendar, store: Store, moment: datetime) -> Any:
    """Monta y compila el grafo del día: expertos en paralelo, *fan-in* y fin.

    Las entradas (``calendar``, ``store``, ``moment``) se **inyectan**: el grafo no lee la red,
    no consulta el reloj y no escribe. El **gate no está aquí**: se llama fuera, como función
    pura (`decision.gate.evaluate_gate`), que es lo que mantiene el backtest retrotesteable.
    """
    builder: StateGraph = StateGraph(PipelineState)
    builder.add_node("calendar", _calendar_node(calendar, moment))
    builder.add_node("macro_publications", _macro_node(store, moment))
    builder.add_node("mega_cap_earnings", _earnings_node(store, moment))
    builder.add_node(MERGE_NODE, _merge_node)
    for name in EXPERT_NODES:
        builder.add_edge(START, name)
        builder.add_edge(name, MERGE_NODE)
    builder.add_edge(MERGE_NODE, END)
    return builder.compile()


def run_pipeline(graph: Any, *, session: date, as_of: datetime) -> PipelineState:
    """Invoca el grafo y devuelve el **estado tipado**. El gate **no** se llama aquí.

    ``graph`` viene de :func:`build_pipeline_graph` (o de un doble en las pruebas): así el camino
    crítico no depende de que LangGraph arranque.
    """
    initial = PipelineState(session=session, as_of=as_of)
    result: object = graph.invoke(initial)
    return PipelineState.model_validate(result)
