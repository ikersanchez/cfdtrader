"""Entrega: informe diario y punto de entrada manual.

Todo lo de esta capa es presentación: si falla, el pipeline sigue produciendo
la recomendación (``_docs/tech_stack.md`` §4.12).

**No hay notificaciones ni canales externos.** La ejecución es manual y a
demanda (``run_daily.py``), y el informe se lee en la terminal o en el fichero
Markdown de ``data/derived/reports/``. El sistema no envía nada a ningún sitio:
ni Telegram, ni correo, ni *push* (``_docs/plan.md`` §13 y §12 regla 16).
"""
