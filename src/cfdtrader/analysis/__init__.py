"""Análisis y mediciones de la Fase 0 (`plan.md` §8.5).

No es un paquete de producción: aquí viven los **estudios** que deciden si el
proyecto sigue adelante, no el pipeline diario. Hoy contiene la descomposición
del drift (`drift.py`, tarea #6), que es la medición decisiva del proyecto.

Nota de estructura: `plan.md` §14 no preveía un paquete `analysis/`; los informes
de la Fase 0 se van a acumular (tareas #6, #7, #8 y #9), así que tienen un sitio
propio en vez de vivir sueltos dentro de `data/` o de `backtest/`.
"""
