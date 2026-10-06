# Runbook — Observación diaria de la Fase 4

> Procedimiento **operativo** de la observación (`plan.md` §16 y §19.14, tarea #45). No es una
> estrategia validada: es un **asistente de decisión** con **ejecución manual**, a demanda, sin
> scheduler ni notificaciones (`plan.md` §16). La valla de honestidad viaja en cada salida: **no hay
> edge demostrado**, Fase 2 `fail`/`not_evaluable`, carril A, `§11.6` sin tocar.

## Qué se persigue

Arrancar y mantener el **reloj de observación**: cada sesión registra su recomendación en
`journal.decisions`. A los `N ≥ 30` días, `cfdtrader.analysis.paper_trading` aplica la puerta de §16
(`|media_paper − media_backtest| > 2σ/√N`; con `N < 30` el veredicto es `not_evaluable`).

## Dónde vive todo

| Qué | Ruta | ¿Versionado? |
|---|---|---|
| Almacén (`raw/`, `derived/`, `cache/`) | `data/` | No (gitignored) |
| Registro de experimentos | `runs/` | No (gitignored) |
| **Diario de decisiones** | `journal/` | No (gitignored) |
| Traza estructurada (`run_log`/`manifest`) | `journal/ops/` | No |
| Informes publicados | `data/derived/reports/` | No |

## Cuándo

En la **mañana** de cada sesión, **antes de la apertura US** (09:30 ET). `--as-of` es la fecha de
**hoy** en ET: la decisión es para la sesión de hoy y usa el **cierre de ayer** (`lag-1`). Correrlo
el mismo día **después** del cierre deja el snapshot por delante del `--as-of` y el guardián de
`tech_stack.md` §8.4 lo declara `no_recommendation_stale_data` (una salida válida, pero no acumula).

## 0. Prerrequisitos (una vez)

```bash
uv sync
cp .env.example .env && chmod 600 .env        # LLM_API_KEY (obligatoria desde Fase 3), FRED_API_KEY
uv run pre-commit install
```

## 1. Ingesta — refresca el almacén hasta el cierre de ayer

```bash
ts=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
uv run python -m cfdtrader.data.market   --data-root data --now "$ts"
uv run python -m cfdtrader.data.macro    --data-root data --now "$ts"
uv run python -m cfdtrader.data.earnings --data-root data --now "$ts"          # lxml declarado en #134
uv run python -m cfdtrader.data.news     --data-root data --now "$ts" --query 'S&P 500'   # (o --feed <url>)
```

> `news` exige al menos un `--query` (GDELT) o un `--feed` (RSS): sin ninguno sale `rc=2` (llamada
> incompleta, no defecto). La ingesta de noticias solo alimenta el **overlay** del LLM, que es
> opcional por diseño: sin titulares el camino diario sigue produciendo su recomendación.

## 2. Regeneración en orden (`_docs/process.md`, regla 2)

Las fechas son las de los artefactos ya publicados (se regeneran **en su sitio**); `--previous-artifact`
es la copia publicada para que el bloque `regeneration` (#90) dé el delta.

```bash
uv run python -m cfdtrader.models.labels               --data-root data --reports-dir data/derived/reports --now 2026-09-18T00:00:00+00:00
uv run python -m cfdtrader.analysis.volatility_forecast --data-root data --now 2026-09-18T00:00:00+00:00
uv run python -m cfdtrader.analysis.backtest_report    --data-root data --reports-dir data/derived/reports --as-of 2026-09-19T00:00:00+00:00
uv run python -m cfdtrader.analysis.phase1_report      --data-root data --reports-dir data/derived/reports --as-of 2026-09-19T22:00:00+00:00
uv run python -m cfdtrader.analysis.baseline_report    --data-root data --reports-dir data/derived/reports --runs-root runs --as-of 2026-09-22T22:00:00+00:00
uv run python -m cfdtrader.analysis.baseline_report    --data-root data --reports-dir data/derived/reports --runs-root runs --as-of 2026-09-22T22:00:00+00:00 --raw
uv run python -m cfdtrader.analysis.model_comparison   --data-root data --reports-dir data/derived/reports --runs-root runs --as-of 2026-09-22T22:00:00+00:00 --previous-artifact data/derived/reports/model_comparison_2026-09-22.json
uv run python -m cfdtrader.analysis.pipeline_report    --data-root data --reports-dir data/derived/reports --as-of 2026-09-23T22:00:00+00:00 --previous-artifact data/derived/reports/pipeline_backtest_2026-09-23.json
uv run python -m cfdtrader.analysis.gate_sweep         --data-root data --reports-dir data/derived/reports --as-of 2026-09-23T22:00:00+00:00
uv run python -m cfdtrader.analysis.phase2_dominance   --data-root data --reports-dir data/derived/reports --as-of 2026-09-24T00:00:00+00:00 --previous-artifact data/derived/reports/phase2_dominance_2026-09-24.json
```

`rc=0` en todos, incluido cuando el veredicto es `fail`/`not_evaluable` (son resultados legítimos).
Regenerar los artefactos de `data/` **no** cambia el árbol de git.

## 3. Camino diario — la pista, o el estado «sin recomendación»

```bash
uv run python -m cfdtrader.delivery.run_daily \
    --as-of "$(date -u +%Y-%m-%d)T13:00:00+00:00" \
    --variant-id lightgbm_gbdt_v1 \
    --journal-root journal \
    --runs-root runs \
    --git-commit "$(git rev-parse HEAD)"
```

Salida `0`: informe emitido (`recommendation`, o `no_recommendation_stale_data`/`_data_quality`; un
día de mercado cerrado es `recommendation` con `NOTHING` justificado). Salida `2`: falta un argumento
o el pipeline falló, con el motivo por `stderr`. **Cada ejecución escribe su fila** en
`journal/decisions/` y deja `journal/ops/<sesión>/run_log.jsonl` + `manifest.json`.

## 4. Puerta de la Fase 4 — el veredicto del *paper*

```bash
uv run python -m cfdtrader.analysis.paper_trading \
    --data-root data --journal-root journal \
    --reports-dir data/derived/reports \
    --as-of "$(date -u +%Y-%m-%d)T22:00:00+00:00"
```

Con `N < 30` sale **`not_evaluable`** (normal durante la observación; nunca un aprobado por silencio).
Divergencia `> 2σ` obliga a **auditar antes de operar**.

## Checklist de la sesión

1. `git rev-parse HEAD` limpio y `uv sync` al día.
2. Ingesta (§1) con `rc=0`.
3. Regeneración (§2) con `rc=0`.
4. `run_daily` (§3) con `rc=0` y una fila nueva en `journal/decisions/`.
5. `paper_trading` (§4) y lectura del `state` del veredicto.
6. La decisión **se ejecuta a mano** (o no): el sistema **no** coloca órdenes. Con `tier B` (o un
   `bloqueo`), la salida es `NOTHING`.
