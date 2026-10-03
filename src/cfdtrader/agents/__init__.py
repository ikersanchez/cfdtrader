"""Agentes expertos (`_docs/plan.md` §7).

Regla de oro: el LLM no calcula números y no decide. Extrae eventos, veta y redacta.
La dirección la fija una función pura en ``cfdtrader.decision.gate``.

El arnés de backtest está en pie (Fases 0–2). En el carril A (`plan.md` §19.7) ya
vive el **núcleo derivable** del calendario de eventos (``event_calendar.py``,
tarea #34); el resto de la Fase 3 (agentes de noticias y LLM) sigue pendiente.
"""
