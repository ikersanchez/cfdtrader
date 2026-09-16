"""Ingeniería de features (`_docs/plan.md` §9).

Reglas de la capa:

- Cálculo determinista y auditable a mano (nada de librerías que oculten ventanas).
- Ventanas y normalización siempre sobre datos anteriores al instante de decisión.
- Cada matriz se versiona por ``sha256(código + parámetros + ventanas + fuente + as_of)``.

Se implementa en las tareas #19–#23.
"""
