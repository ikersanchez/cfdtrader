"""Adaptadores por fuente, detrás de interfaces propias.

Una fuente frágil nunca se usa directamente: todo adaptador valida y declara
*fuente de respaldo* (``_docs/tech_stack.md`` §3.3.3 y §4.5).

Previstos: yfinance, Stooq, FRED (macro US primaria), ECB SDW, GDELT, RSS.
Se implementan en las tareas #3 y #5.
"""
