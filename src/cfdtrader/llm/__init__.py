"""Capa LLM: interfaz propia y cliente de proveedor (`tech_stack.md` §4.9) — tarea #31.

El **principio 9** del proyecto exige que el proveedor de LLM vaya detrás de una
interfaz propia: cambiar de OpenAI a DeepSeek (o a un modelo local) no debe tocar
``agents/`` ni ``orchestration/``. Aquí viven el protocolo agnóstico
(:mod:`cfdtrader.llm.base`) y la implementación del SDK
(:mod:`cfdtrader.llm.openai_client`), que es la **única** pieza que importa el SDK.

``agents/`` importa ``cfdtrader.llm.base`` (el protocolo), nunca ``openai_client``.
"""

from __future__ import annotations

__all__: list[str] = []
