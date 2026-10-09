# HOWTO — Ejecutar el pipeline completo y obtener el reporte de decisión

Este documento explica, **sin depender de nadie**, cómo ejecutar de principio a fin el
pipeline de `cfdtrader` y obtener al final un **REPORTE FINAL** con todo lo necesario para
decidir: dirección (**LONG/SHORT**), **decisión final publicada**, **la variante con overlay**
(que puede convertirla en `NOTHING`), **probabilidad y el corte del tier A**, EV,
stop/objetivo, coste y tamaño.

> ⚠️ **No hay edge demostrado.** La Fase 2 dio `fail` en la base neta y el valor esperado
> medido sale **negativo**. Esto es un **asistente de decisión con ejecución manual**: el
> sistema **no coloca ninguna orden**. Operar es una decisión del propietario.

La fuente de verdad del procedimiento es [`_docs/runbook.md`](_docs/runbook.md); este HOWTO
lo resume y lo automatiza.

---

## TL;DR — un solo comando

```bash
uv sync                     # 1ª vez (instala el entorno)
cp .env.example .env        # 1ª vez; rellena LLM_API_KEY y FRED_API_KEY
chmod 600 .env

bash run_pipeline.sh        # pipeline COMPLETO + REPORTE FINAL
```

Al terminar imprime, al final de la salida, el **REPORTE FINAL DE LA DECISIÓN**
(ver [§4](#4-cómo-leer-el-reporte-final)).

Si solo quieres el **día de hoy** (sin la regeneración pesada de artefactos):

```bash
bash run_pipeline.sh --daily-only
```

---

## 1. Prerrequisitos (solo la primera vez)

```bash
uv sync
cp .env.example .env && chmod 600 .env   # LLM_API_KEY (obligatoria, Fase 3) y FRED_API_KEY
uv run pre-commit install                # opcional: hooks locales
```

- El entorno es `uv` + Python 3.12; `uv.lock` está commiteado (reproducibilidad).
- El modelo diario se resuelve con `--variant-id lightgbm_gbdt_v1` contra el registro
  `runs/`. El registro lo puebla la **regeneración** (paso 2 de `run_pipeline.sh`), así que
  la **primera** vez hay que correr el pipeline completo, no `--daily-only`.
- `--daily-only` reutiliza los artefactos ya publicados en `data/derived/reports/`.

---

## 2. Qué hace `run_pipeline.sh`

Un único comando que ejecuta los 8 pasos del runbook y termina con el reporte. Todas las
rutas y constantes son las del código actual.

| # | Paso | Módulo | Qué produce |
|---|---|---|---|
| 1 | **Ingesta** | `data.market`, `data.macro`, `data.earnings`, `data.news` | Refresca el almacén `data/` hasta el cierre de ayer |
| 2 | **Regeneración en orden** | `models.labels`, `analysis.volatility_forecast`, `analysis.backtest_report`, `analysis.phase1_report`, `analysis.baseline_report`, `analysis.model_comparison`, `analysis.pipeline_report`, `analysis.gate_sweep`, `analysis.phase2_dominance` | Reconstruye los artefactos de `data/derived/reports/` y el registro de modelos `runs/` |
| 3 | **Pre-mercado ES** | `analysis.premarket_gap` | Captura el gap del futuro ES (#141) en `journal/ops/<sesión>/premarket_bars.json` |
| 4 | **Camino diario** | `delivery.run_daily` | **La pista del día** → `journal/decisions/<sesión>.json` + `journal/ops/<sesión>/daily_report.txt` |
| 5 | **Valla de cartera** | `analysis.portfolio_rules` | Estado del *kill switch* (§12 reglas 3-5): pérdida diaria/semanal/mensual |
| 6 | **Tarjeta de operación** | `delivery.production` | **Si se opera** y con qué geometría, con la política de tamaño mínimo (#47) |
| 7 | **Puerta del paper** | `analysis.paper_trading` | Veredicto de §16 sobre el diario (con `N < 30` sale `not_evaluable`) |
| 8 | **REPORTE FINAL** | (lee `journal/decisions/<sesión>.json`) | Resumen de la decisión: LONG/SHORT, p, corte, tier, EV, stop/objetivo… |

Notas:

- El paso 2 **aparta** los ensayos del barrido (`lightgbm_search_v1#*`) del registro `runs/`
  mientras corre la cadena de modelos y los devuelve al terminar (§19.17): sin eso
  `model_comparison` deja la matriz incompleta y `phase2_dominance` no puede copiar el `pbo`.
- Los pasos que pueden devolver `rc≠0` legítimamente (p. ej. noticias sin titulares, o
  `NOTHING`/`not_evaluable`) **no abortan** el script: se registra el `rc` y se continúa.
- La `sesión` es la fecha de **hoy en UTC** y `--as-of` la del snapshot de la mañana
  (13:00 UTC ≈ 09:00 ET). La decisión usa el **cierre de ayer** (`lag-1`).


---

## 3. Ejecución paso a paso (manual, si prefieres control total)

Todas las órdenes, tal cual, desde la raíz del repo:

```bash
SESSION="$(date -u +%Y-%m-%d)"
TS="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
R="data/derived/reports"

# 1) Ingesta
uv run python -m cfdtrader.data.market   --data-root data --now "$TS"
uv run python -m cfdtrader.data.macro    --data-root data --now "$TS"
uv run python -m cfdtrader.data.earnings --data-root data --now "$TS"
uv run python -m cfdtrader.data.news     --data-root data --as-of "$TS" \
  --feed 'cnbc-top=https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114' \
  --feed 'cnbc-energy=https://www.cnbc.com/id/19836768/device/rss/rss.html' \
  --feed 'oilprice=https://oilprice.com/rss/main' \
  --feed 'google-news=https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en'

# 2) Regeneración en orden (ver _docs/process.md regla 2)
uv run python -m cfdtrader.models.labels                --data-root data --reports-dir "$R" --now 2026-09-18T00:00:00+00:00
uv run python -m cfdtrader.analysis.volatility_forecast --data-root data --now 2026-09-18T00:00:00+00:00
uv run python -m cfdtrader.analysis.backtest_report     --data-root data --reports-dir "$R" --as-of 2026-09-19T00:00:00+00:00
uv run python -m cfdtrader.analysis.phase1_report       --data-root data --reports-dir "$R" --as-of 2026-09-19T22:00:00+00:00
uv run python -m cfdtrader.analysis.baseline_report     --data-root data --reports-dir "$R" --runs-root runs --as-of 2026-09-22T22:00:00+00:00 --previous-artifact "$R/baseline_2026-09-22.json"
uv run python -m cfdtrader.analysis.baseline_report     --data-root data --reports-dir "$R" --runs-root runs --as-of 2026-09-22T22:00:00+00:00 --raw --previous-artifact "$R/baseline_2026-09-22.json"
#    → para model_comparison/pipeline_report/phase2_dominance, usar `run_pipeline.sh`
#      (aparta y restaura los ensayos del barrido; ver runbook §2).

# 3) Pre-mercado del ES
uv run python -m cfdtrader.analysis.premarket_gap \
    --as-of "$(date -u +%Y-%m-%d)T13:00:00+00:00" --reports-dir "$R" \
    --capture-bars "journal/ops/$SESSION/premarket_bars.json"

# 4) Camino diario — LA PISTA
uv run python -m cfdtrader.delivery.run_daily \
    --as-of "$(date -u +%Y-%m-%d)T13:00:00+00:00" \
    --variant-id lightgbm_gbdt_v1 \
    --journal-root journal --runs-root runs \
    --git-commit "$(git rev-parse HEAD)" \
    --premarket-bars "journal/ops/$SESSION/premarket_bars.json"

# 5) Valla de cartera
uv run python -m cfdtrader.analysis.portfolio_rules \
    --journal-root journal --session "$SESSION" \
    --as-of "$(date -u +%Y-%m-%d)T12:00:00+00:00" --reports-dir "$R"

# 6) Tarjeta de operación (¿se opera? ¿con qué tamaño?)
uv run python -m cfdtrader.delivery.production --journal-root journal --session "$SESSION"

# 7) Puerta del paper (§16)
uv run python -m cfdtrader.analysis.paper_trading \
    --data-root data --journal-root journal \
    --reference-artifact "$R/pipeline_backtest_2026-09-23.json" \
    --reports-dir "$R" --as-of "$(date -u +%Y-%m-%d)T22:00:00+00:00"
```

Códigos de salida de `run_daily`: `0` = informe emitido (incluido `NOTHING` justificado o
`no_recommendation_stale_data`); `2` = falta un argumento o el pipeline falló (con el motivo
por `stderr`).

---

## 4. Cómo leer el REPORTE FINAL

Ejemplo real (sesión 2026-10-09), producido por `run_pipeline.sh`:

```
  Estado del informe ......... recommendation
  DECISIÓN FINAL (sin overlay)  SHORT

  p(up) calibrada ............ 0.4100
  p a favor del lado ......... 0.5900   (p si p>=0.50, 1-p si no)
  Corte dirección (p) ........ 0.5000   -> SHORT
  Corte tier A (p a favor) ... 0.5800   -> PASA (> 0.58)
  Tier ....................... A

  ── Overlay de noticias (§19.19) ──
  Estado del overlay ......... veto
  Decisión CON overlay ....... NOTHING (veto del overlay)
  p(up) con overlay .......... 0.4100118532769206
  Motivo del veto ............ geopolitics/high: …; earnings/high: …
  (la recomendación PUBLICADA se emite sin overlay; esta línea es informativa)

  EV declarado (%) ........... 0.3247 %   (regla 9: > 3x coste)
  EV neto · sensibilidad (%) . 0.1247 %
  Coste declarado (%) ........ 0.0042 %
  Movimiento esperado (%) .... 0.5575 %
  Stop (%) ................... 0.5575 %
  Objetivo (%) ............... 1.1150 %

  Tamaño (fracción capital) .. null
  Nocional (EUR) ............. null
  Apalancamiento implícito ... 1.7938
  Overlay de noticias ........ veto
```

Cómo se interpreta (constantes **reales** del código, escenario declarado **S1**):

| Campo | Significado |
|---|---|
| **DECISIÓN FINAL (sin overlay)** | La que se **publica**: `LONG` / `SHORT` / `NOTHING (no se opera)` / `SIN RECOMENDACIÓN (estado)` |
| **p(up) calibrada** | Probabilidad calibrada de que el índice **suba** |
| **Corte dirección** | `DECISION_THRESHOLD = 0.50` (`decision/gate.py`): `p ≥ 0.50` → LONG; `p < 0.50` → SHORT |
| **p a favor** | La probabilidad del lado que se opera: `p` si `p ≥ 0.50`, si no `1 − p` |
| **Corte tier A** | `tier_a_min_probability = 0.58`: para operar hay que **superar 0.58** a favor, y `EV declarado > 3 × coste` (regla 10 de §12) |
| **Tier** | `A` (operables) · `B`/`C` (se registran y **no** se operan) |
| **EV declarado** | Valor esperado con el **coste declarado** (regla 9: `> 2 × coste`) |
| **EV neto · sensibilidad** | EV bajo el supuesto de *slippage* (20 % de `R`); es **sensibilidad**, no una medición |
| **Movimiento esperado / Stop / Objetivo** | Del `σ` de GARCH: stop = `1 × σ`, objetivo = `2 × stop` (reglas 7 y 8) |
| **Tamaño / Nocional** | `null` mientras no haya operación real; el nocional es un **techo** (riesgo 1 %, regla 2) |
| **Estado del overlay** | `veto` / `applied` / `disabled_*`; bajo §19.19 la recomendación **publicada** se emite **sin** overlay |
| **Decisión CON overlay** | Qué haría la señal **si el overlay actuara**: p. ej. `NOTHING (veto del overlay)`. `= sin overlay (no hay fila)` si la sesión no registró señal |
| **Motivo del veto** | Los titulares de alta magnitud/confianza que provocan el veto (regla 20 ⇒ `NOTHING`) |

**Las dos decisiones.** El reporte da, en este orden: (1) la **decisión final publicada**
(**sin** overlay) y (2) la **variante con overlay**, que es la que puede **cambiarla a
`NOTHING`** por un veto de noticias (§19.19 / #148, #149). La señal con overlay se guarda
aparte en `journal/agent_signals/<sesión>__news.json`; la publicada, en
`journal/decisions/<sesión>.json`. Si el overlay está `disabled_*` o no hay fila, la segunda
línea sale como `= sin overlay`.

- `NOTHING (no se opera)` aparece con su motivo: un `bloqueo: <regla>:<código>` por cada
  bloqueo del gate, un día de mercado cerrado (regla 19) o la reincorporación de la regla 15.
- Si `Estado del informe` es `no_recommendation_stale_data` o `no_recommendation_data_quality`,
  **no se opera**: el almacén no está al día o la fila de features es incompleta.

---

## 5. Dónde queda todo

| Qué | Ruta |
|---|---|
| Informe del día (texto que imprime `run_daily`) | `journal/ops/<sesión>/daily_report.txt` |
| **Fila del diario** (la que lee el REPORTE FINAL) | `journal/decisions/<sesión>.json` |
| **Señal del overlay** (variante «con overlay», #149) | `journal/agent_signals/<sesión>__news.json` |
| Traza estructurada (`run_log.jsonl`, `manifest.json`) | `journal/ops/<sesión>/` |
| Barras de pre-mercado ES | `journal/ops/<sesión>/premarket_bars.json` |
| Informes publicados (JSON + MD) | `data/derived/reports/` |
| Registro de modelos | `runs/` |
| Operación real (si se ejecuta) | `journal/trades/` |

`data/`, `runs/` y `journal/` están **gitignorados**: regenerarlos no cambia el árbol de git.

---

## 6. Ejecución real de una operación (solo si hay pista direccional)

La ejecución es **manual**; el sistema no coloca órdenes. El *bracket* y el registro real
(#84) se imprimen con:

```bash
# 09:30 ET — billete de ejecución (relleno de la subasta)
uv run python -m cfdtrader.delivery.bracket \
    --journal-root journal --session "$(date -u +%Y-%m-%d)" --entry-px 7839.0

# 16:15 ET — con los datos REALES de la sesión (escribe la fila de journal.trades)
uv run python -m cfdtrader.delivery.bracket \
    --journal-root journal --session "$(date -u +%Y-%m-%d)" \
    --record --entry-px 7839.0 --exit-px 7861.5 --exit-reason close --costs-pct 0.0042 \
    --entry-time "$(date -u +%Y-%m-%d)T13:30:00+00:00" \
    --exit-time  "$(date -u +%Y-%m-%d)T20:00:00+00:00"
```

Reglas duras activas (todas verificadas por `delivery.production`): **solo tier A**, **una
operación por sesión**, **sin overnight** (cierre obligatorio a las 16:00 ET) y la **valla de
cartera** (−2 % diario, −5 % semanal, −10 % mensual).

---

## 7. Valla de honestidad

Esto **no** es una estrategia validada: la Fase 2 sigue `fail`/`not_evaluable` con
`phase2_ready = false`, el carril B está bloqueado y `plan.md` §11.6 no se altera. El *paper*
con `N < 30` siempre da **`not_evaluable`** (nunca un aprobado por silencio). El valor está en
el **no-trade** y en la gestión de riesgo, y la ejecución siempre la decide una persona.
