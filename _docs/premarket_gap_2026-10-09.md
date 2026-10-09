# Decisión: el gap de pre-mercado del ES como dato declarado (`#141`)

> **Fecha:** 2026-10-09 · **Tarea:** #141 · **Bloqueado por datos:** #107 (intradía y `bid`/`ask`
> reales del CFD) · **Aval de honestidad:** esto **no** es una afirmación de *edge* ni una feature del
> modelo. La Fase 2 sigue `not_evaluable`/`fail` con `phase2_ready = false`, el carril B sigue
> bloqueado (`plan.md` §11.6 / §19.7) y la ejecución es **manual** (carril A).

## El hueco que cierra

`plan.md` §13 congela el snapshot a las **08:45 ET** y entrega a las 09:00 ET. El **2026-10-09** el
camino diario (`delivery.run_daily`) emitió `SHORT` con `prob_up = 0,41` sobre la fila del cierre del
2026-10-08, **mientras el futuro ES cotizaba en positivo en el pre-mercado**. El modelo no puede ver
ese dato: ninguna de las 10 features (`BASELINE_FEATURES`) usa el precio del futuro, y el único
dato ligado al ES es `is_es_roll_session` (`features/regime.py`), un **flag de calendario**.

## Lo medido antes de decidir (sobre `data/`, 2026-10-09)

| Hecho | Medición |
|---|---|
| La barra **diaria** de `ES=F` es una ventana de **24 h** que termina a las **16:00 ET** | `as_of` = `20:00 UTC` (= 16:00 ET), y `open(t) ≈ close(t-1)`: mediana `|open(t) − close(t-1)| = 1,0 pt`; su `open` **no** es el de las 09:30 (mediana 22 pts frente al primer bar de 5 min) |
| Ese movimiento de pre-mercado vive **entero** en la barra `t` | a las 08:45 la barra `t` **aún no existe** (su `as_of` es de las 16:00 de hoy) ⇒ usarla sería *look-ahead* |
| El intradía **guardado** de `ES=F` es solo RTH (09:30–16:00 ET) y ~60 días | `raw.market_intraday`: `2026-07-09 09:30 → 2026-10-08 16:00`, **sin** barras de pre-mercado |
| El pre-mercado **sí** es descargable en vivo, **sin historia** | `yf.Ticker("ES=F").history(interval="5m", prepost=True)` devuelve desde las `04:00 UTC`; el límite rodante de Yahoo para 5 m es **~60 días** |
| El único dato **diario** disponible es **redundante** | `corr(es_ret_1, ^GSPC ret_1) = **0,9720**` (2005-01-04 → 2026-10-08, n = 5 466): el mismo activo |

**Conclusión.** El gap de pre-mercado a las 08:45 ET **no es computable point-in-time** con una serie
**histórica** con los datos que hay: la barra diaria es de 24 h (y sería *look-ahead*) y el intradía
gratuito con horas extendidas es una ventana rodante de ~60 días. Es un problema de **datos**, no de
código.

## La decisión

**Se publica el gap de pre-mercado del ES como *dato declarado de la decision* (carril A), y NO como
feature del modelo.** El CLI `cfdtrader.analysis.premarket_gap` descarga las barras de 5 minutos de
`ES=F` **con horas extendidas**, toma la **última barra con instante <= 08:45 ET** y publica el
resultado; `run_daily --premarket-bars` lo incorpora al informe del día.

### Los tres números (nunca se suman)

Con `ES=F` cerrando a `5050,0` y el `^GSPC` a `5000,0` en la sesión anterior, y una barra de
pre-mercado a `5100,5`:

| Medida | Valor | Qué es |
|---|---|---|
| `overnight_move_pct` | **+1,00 %** | `ES(pre-mercado) / ES(cierre t-1) − 1`: el movimiento del **propio** futuro, **sin** *basis* |
| `gap_vs_index_pct` | **+2,01 %** | `ES(pre-mercado) / ^GSPC(cierre t-1) − 1`: el «gap» de la issue, que **incluye** el *basis* |
| `basis_pct` | **+1,00 %** | `ES(cierre t-1) / ^GSPC(cierre t-1) − 1`: cuánto del gap es *basis* estructural |

El *basis* (~+0,65 % medido en el almacén) es lo que separa los dos primeros: por eso se publican los
tres y **no** se presenta el segundo como «el movimiento».

### Garantías *point-in-time*

- Solo se usa la **última barra con instante <= 08:45 ET** (nunca el `open` ni el `close` de la sesión
  que se decide: el `open` de la subasta es el precio de entrada, #64, y usarlo sería *look-ahead*).
- El **cierre de referencia** es el último `raw.market_daily` con `as_of <= 08:45 ET`, que es la
  sesión **anterior** (la fila de `t` se sella a las 16:00 ET de hoy).
- Un `as_of` anterior al snapshot, una fuente caída, sin barras o sin referencia salen
  **`unavailable`** con su motivo: una ausencia declarada, jamás un `0`.

## Qué NO se hace, y por qué

- **No es una feature del modelo.** El modelo (y su re-medida de la Fase 2 con purga/embargo, DSR y
  PBO) exige una serie **histórica** de pre-mercado que las fuentes gratuitas no dan. Añadir hoy una
  feature del ES con el diario sería *look-ahead* (barra de 24 h) o redundante (`r = 0,972`).
- **La re-medida de la Fase 2 queda bloqueada por #107.** Sin historia de pre-mercado no hay nada que
  re-medir; el disparador de esta tarea era una **sesión observada**, no un resultado medido.
- **El presupuesto de features (#24) no cambia:** `models.baseline.BASELINE_FEATURES` sigue siendo la
  misma tupla de **10** y `delivery.run_daily` sigue puntuando exactamente esas 10.
- **`ES=F` es un proxy declarado** del subyacente del CFD (`plan.md` §3.1): `SPX500:CFD` sigue
  `unavailable` y sustituirlo es #107.

## Cómo se reproduce

```bash
# 1) Captura de las barras (a las 08:45 ET; queda en la traza del dia). Necesita red.
uv run python -m cfdtrader.analysis.premarket_gap \
    --as-of "$(date -u +%Y-%m-%d)T13:00:00+00:00" \
    --reports-dir data/derived/reports \
    --capture-bars journal/ops/$(date -u +%Y-%m-%d)/premarket_bars.json

# 2) El camino diario lee esas barras (sin red): publica el gap junto a la recomendacion.
uv run python -m cfdtrader.delivery.run_daily ... \
    --premarket-bars journal/ops/$(date -u +%Y-%m-%d)/premarket_bars.json
```

Sin `--es-bars` ni `--capture-bars`, el CLI intenta la descarga real; con `--es-bars` reproduce un gap
**sin red** desde barras ya capturadas (tests y auditoría).

## Lo que sigue bloqueado (y con qué issue)

- Serie histórica de pre-mercado del ES / intradía y `bid`/`ask` reales del CFD → **#107**.
- Fiabilidad de las fuentes gratuitas (Yahoo `429`, ventana rodante de 5 m) → **#51**.
