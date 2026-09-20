"""Ingeniería de features (`_docs/plan.md` §9).

Reglas de la capa:

- Cálculo determinista y auditable a mano (nada de librerías que oculten ventanas).
- Ventanas y normalización siempre sobre datos anteriores al instante de decisión.
- Cada matriz se versiona por ``sha256(código + parámetros + ventanas + fuente + as_of)``.

Módulos:

- :mod:`cfdtrader.features.volatility` — **definición única** del ATR normalizado y
  de la familia de volatilidad del proyecto (Parkinson, HAR y VIX, tarea #7). Las
  tareas #20 (features técnicas, que también nombra el ``atr_norm``) y #23 (features
  de régimen y volatilidad) **importan de ahí**, no reimplementan las fórmulas.
- :mod:`cfdtrader.features.store` — **infraestructura de persistencia** (tarea #19):
  identidad reproducible (``features_version`` por sesión y ``feature_spec_sha256``
  por contrato), esquema ancho en ``derived.features_daily``, catálogo documentado y
  normalización robusta de ventana expandida. Las familias de features (#20–#23) se
  persistirán **a través de él**, no con su propio ``Store``.

Se implementa en las tareas #19–#23.
"""
