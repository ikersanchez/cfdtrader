"""Decisión: el corazón del sistema.

``gate.py`` es una **función pura y determinista**: mismos inputs, mismo output,
byte a byte, siempre. No importa orquestación, ni LLM, ni red, ni reloj.
Es lo que permite retrotestear el sistema (``_docs/plan.md`` §6.2 y §7.3).
"""
