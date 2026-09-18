"""Arnés de backtest: motor propio, costes, purga y embargo, métricas.

Se construye **antes** que los agentes (``_docs/plan.md`` §2 y §11).
El LLM nunca entra en este camino.

Dependencias permitidas aquí: numpy, polars y scikit-learn (``tech_stack.md`` §3.3.2).

El **motor de costes** vive en ``cfdtrader.backtest.costs`` (tarea #11): modelo puro y
determinista del coste de ida y vuelta de una operación, con la tabla declarada de
``plan.md`` §3.3 **importada** de ``cfdtrader.analysis.cost_audit`` (tarea #8) y con el
*slippage* como parámetro obligatorio. Cobra; no mide ni recorre sesiones.

Las **particiones *walk-forward* con purga y embargo** viven en
``cfdtrader.backtest.splits`` (tarea #12): puro y determinista, decide qué sesiones entran
en el *train* y cuáles en el *test* (bloques de test contiguos, train estrictamente
anterior, expansivo o rodante) y publica un ``SplitPlan`` verificable y reproducible por
``plan_sha256``. Es la pieza que consumen #13, #16, #24 y #25; no entrena, no mide precios
y no reserva el *holdout* (eso es #68).
"""
