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

El **motor *walk-forward*** vive en ``cfdtrader.backtest.engine`` (tarea #13): puro y
determinista, recorre las sesiones de *test* del ``SplitPlan`` de #12, pide una ``Decision``
por sesión a partir de una vista **sin el futuro de la sesión**, simula la operación
intradía ``open`` -> ``close`` (entrada en la subasta de apertura, salida dentro de la
sesión, *gap* declarado y no operado) y devuelve un ``BacktestRun`` verificable, con el eco
del plan y su ``run_sha256`` reproducible byte a byte. El reparto es «**#11 cobra; #13
recorre y simula**»: el coste sale de una única llamada a ``cost_breakdown`` por operación.
No lee el ``Store``, no construye los ``SessionInput`` y no escribe el informe: eso es el
adaptador de almacén **#69**.

Los **baselines triviales** viven en ``cfdtrader.backtest.baselines`` (tarea #14): las seis
reglas de ``plan.md`` §11.2 —no operar, siempre largo (el listón **A** ``open`` -> cierre,
explícitamente identificado), siempre corto, momentum 5d, reversión de *gap* y regla
aleatoria con la misma frecuencia— como funciones **puras y deterministas** que entregan una
``DecisionFn`` por fold. El reparto es «**#13 ejecuta y cobra; #14 solo decide**»: el módulo
no llama a ``cost_breakdown``, no construye un ``CostBreakdown`` y su único camino de
ejecución es ``run_walk_forward``, que exige ``cost_model`` y ``slippage`` sin valor por
defecto, de modo que ningún baseline puede producir un resultado sin coste. Los listones
**B** (aguantar la posición, con la financiación del CFD) y **C** (índice puro, no
invertible) quedan **fuera** y declarados: son #70 y #28. La corrida real sobre el histórico
y la tabla comparativa de métricas netas son el informe de Fase 1 (**#18**), con las
métricas de **#15** y el adaptador de almacén **#69**.

Las **métricas netas y de calibración** viven en ``cfdtrader.backtest.metrics`` (tarea #15):
el límite de agregación que consume las ``SessionOutcome`` de #13 y publica Sharpe y Sortino
netos con intervalo *bootstrap*, EV por operación, hit rate, payoff, *profit factor*,
drawdown máximo y duración **en sesiones de la serie de riesgo**, Brier, *log-loss*, curva de
calibración, rotación y coste total acumulado, además de la separación alfa/beta contra el
*benchmark*. El reparto es «**#13 ejecuta; #15 agrega**»: nunca sustituye un ``pnl_net_pct``
nulo por ``pnl_declared_pct`` ni por ``0``, no publica ``inf`` ni ``nan`` (``null != 0``) y
suma el dinero en ``Decimal``. ``alpha_pct`` es el alfa de Jensen, no la diferencia de medias
(esa va en ``mean_excess_return_pct``), y ``trades_per_year`` solo se publica si el span
sostiene la anualización. Con el supuesto de *slippage* de #64 el total no se cierra y toda
corrida operada es un error **declarado** (#62 mide el *slippage* y #60 decide ``R``). No lee
el ``Store`` (adaptador #69), no alinea series por fecha, no escribe el informe (#18) y no
decide (el *gate* es #27).

La **corrección por sobreajuste** vive en ``cfdtrader.backtest.overfitting`` (tarea #16):
el **Deflated Sharpe Ratio** de Bailey y López de Prado (con el número **real** de variantes
probadas, derivado del registro de experimentos, nunca de un parámetro suelto) y la
**probabilidad de sobreajuste** por validación cruzada combinatoriamente simétrica (CSCV)
sobre la matriz de retornos de las variantes. Es puro: stdlib (``statistics.NormalDist``),
``numpy`` y los contratos de #13/#15, **sin** ``scipy``, sin ``cfdtrader.data`` y **sin**
``cfdtrader.backtest.splits`` (el CSCV no decide particiones de entrenamiento: eso es #67).
Todo entra **por sesión** y se publica con veredicto explícito; el instante y el disco son
de ``cfdtrader.analysis.experiment_log``, que es quien escribe ``runs/<hash>/``.
"""
