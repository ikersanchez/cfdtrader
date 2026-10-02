"""Diario de decisiones y atribución.

El diario es el **dato irreversible** del sistema: sin él no hay auditoría ni
atribución de resultados (``_docs/plan.md`` §19 y ``tech_stack.md`` §12.5).
La capa de persistencia vive en :mod:`cfdtrader.journal.decision_log` (tarea #39):
este paquete lo reexporta **bajo demanda** (PEP 562) para que ``from
cfdtrader.journal import decision_log`` funcione sin arrastrar, al importar el
paquete, la cadena ``decision_log`` → ``data.store`` → ``duckdb``. El import
perezoso es deliberado: importar ``cfdtrader.journal`` debe ser barato y no toca
el motor de almacenamiento salvo que alguien pida el módulo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - solo para el tipado estático
    from cfdtrader.journal import decision_log as decision_log

__all__ = ["decision_log"]


def __getattr__(name: str) -> Any:
    """Reexporta ``decision_log`` la primera vez que se pide el atributo (PEP 562)."""
    if name == "decision_log":
        import cfdtrader.journal.decision_log as module

        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
