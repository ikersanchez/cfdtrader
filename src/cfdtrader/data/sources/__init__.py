"""Adaptadores por fuente, detrás de interfaces propias.

Una fuente frágil nunca se usa directamente: todo adaptador valida y declara
*fuente de respaldo* (``_docs/tech_stack.md`` §3.3.3 y §4.5).

Previstos: yfinance, Stooq, FRED (macro US primaria), ECB SDW, GDELT, RSS.
Implementados: mercado (#3), macro (#5) y noticias (``news.py``: GDELT y RSS, #30).
"""
