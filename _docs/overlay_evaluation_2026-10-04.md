# Evaluacion incremental del overlay LLM (tarea #38)

- **Fecha:** 2026-10-04
- **Veredicto de la puerta de la Fase 3:** `not_evaluable`
- **Archivo historico de noticias disponible:** no

## Brier y Sharpe OOS, con y sin overlay

| Metrica | Con overlay | Sin overlay |
|---|---|---|
| Brier score | no medido | no medido |
| Sharpe OOS | no medido | no medido |

## Motivos

- el backtest corre siempre sin overlay: no hay archivo historico de noticias con published_at fiable (tech_stack.md §4.9)
- sin medicion pareada (con/sin overlay) no hay Brier ni Sharpe OOS que comparar: no se inventa ningun numero
- consecuencia declarada de antemano: el LLM queda solo como redactor y la evidencia se traslada al paper trading de la Fase 4 (#45)

## Metodo alternativo de evaluacion

Registro prospectivo (Fase 4, #45): cada dia se guardan la recomendacion con el overlay y sin el, y la accion humana; al cierre se registran el P&L y la atribucion, y se compara hacia delante. Sin archivo historico de noticias, es el unico metodo honesto.

## Valla de honestidad

- No hay edge demostrado. La ejecucion es manual y esto es apoyo a la decision,
  no una estrategia validada.
- `not_evaluable` es un veredicto valido de la puerta, no un fracaso que maquillar.
