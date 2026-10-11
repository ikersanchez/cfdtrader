# Candidatas históricas ES/S&P (`#156`)

Este experimento mide dos columnas **opt-in** desde el `raw.market_daily` local. Se mantiene
separado de `commodities_v1` y de la matriz estándar de features: no cambia
`BASELINE_FEATURES`, `delivery.run_daily`, las recomendaciones públicas ni `phase2_ready`. Un
resultado descriptivo no demuestra edge ni aprueba la adopción.

## Catálogo point-in-time

| Feature | Fórmula en la fila de diseño `t` | Entrada | `design_lag` |
| --- | --- | --- | ---: |
| `es_basis` | `ES=F_close(t−1) / ^GSPC_close(t−1) − 1` | cierres de `ES=F` y `^GSPC` en sesiones del calendario `^GSPC` | 0 |
| `spx_overnight_ret_1` | `ln(^GSPC_open(t−1) / ^GSPC_close(t−2))` | apertura y cierres de `^GSPC`, alineados a su calendario de sesiones | 0 |

La fila de diseño `t` toma los valores calculados para la sesión `t`; esos valores solo usan
sesiones previas y ya están disponibles antes de la sesión `t`. Festivos o huecos no se rellenan:
un precio requerido ausente, nulo, no positivo o no finito produce `null`. Una sesión duplicada
para una serie detiene la medición para evitar escoger una procedencia arbitrariamente.

El preflight declara como referencia `2026-10-09`, pero cuenta filas, sesiones, fuentes, rangos y
precios inválidos desde el almacén disponible; no fija totales vivos. Para cada candidata cuadra
las filas válidas con las ausentes o inválidas. El modelo y todas las variantes usan la misma
ventana de casos completos, sin imputación.

## Reproducción

La vía lee exclusivamente el almacén existente y no hace peticiones de red:

```bash
UV_CACHE_DIR=/tmp/cfdtrader-uv-cache uv run python -m cfdtrader.analysis.feature_budget \
  --data-root data \
  --reports-dir /tmp/cfdtrader-feature-budget \
  --candidate-group historical_market \
  --as-of 2026-10-11T22:00:00+00:00
```

El reporte se llama `feature_budget_historical_market_<fecha>.{json,md}` para que no sobrescriba el
artefacto de commodities. El grupo predeterminado del CLI sigue siendo el de #146. La medición
reutiliza sus conjuntos control/ampliado/intercambios, comparador pareado, DSR, PBO y el protocolo
walk-forward; declara por separado cobertura, filas excluidas y las exclusiones medidas de purga y
embargo. `not_evaluable` nunca se interpreta como aprobado.
