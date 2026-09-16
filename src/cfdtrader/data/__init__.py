"""Ingesta, almacén point-in-time y calidad de datos (`_docs/plan.md` §8).

Layout en disco:

- ``data/raw/`` — inmutable, *append-only*. Nunca se sobrescribe.
- ``data/derived/`` — recalculable. Se puede reconstruir desde ``raw/``.

Sin agentes y sin LLM en esta capa (Fase 0).
"""
