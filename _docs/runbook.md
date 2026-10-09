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
uv run python -m cfdtrader.data.news     --data-root data --as-of "$ts" \
  --feed 'cnbc-top=https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114' \
  --feed 'cnbc-energy=https://www.cnbc.com/id/19836768/device/rss/rss.html' \
  --feed 'oilprice=https://oilprice.com/rss/main' \
  --feed 'google-news=https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en'
```

> `news` exige al menos un `--query` (GDELT) o un `--feed` (RSS): sin ninguno sale `rc=2` (llamada
> incompleta, no defecto). El flag es `--as-of`, **no** `--now` (como el resto de ingestas).
> La ingesta de noticias solo alimenta el **overlay** del LLM, que es
> opcional por diseño: sin titulares el camino diario sigue produciendo su recomendación.
>
> **Aviso medido el 2026-10-09 (#142):** la API de **GDELT** (`--query`) responde **429**, así que
> hoy se ingiere por **RSS** (`--feed`). Fuentes ejecutables, licencia y límites en
> [`data_sources.md`](data_sources.md) §«Noticias» y en
> [`news_sources_2026-10-09.md`](news_sources_2026-10-09.md). **No** mezcles `google-news` con otros
> feeds si quieres que el lote no aborte por colisión de identidad (defecto abierto de #142).

## 2. Regeneración en orden (`_docs/process.md`, regla 2)

Las fechas son las de los artefactos ya publicados (se regeneran **en su sitio**); `--previous-artifact`
es la copia publicada para que el bloque `regeneration` (#90) dé el delta.

```bash
uv run python -m cfdtrader.models.labels               --data-root data --reports-dir data/derived/reports --now 2026-09-18T00:00:00+00:00
uv run python -m cfdtrader.analysis.volatility_forecast --data-root data --now 2026-09-18T00:00:00+00:00
uv run python -m cfdtrader.analysis.backtest_report    --data-root data --reports-dir data/derived/reports --as-of 2026-09-19T00:00:00+00:00
uv run python -m cfdtrader.analysis.phase1_report      --data-root data --reports-dir data/derived/reports --as-of 2026-09-19T22:00:00+00:00
uv run python -m cfdtrader.analysis.baseline_report    --data-root data --reports-dir data/derived/reports --runs-root runs --as-of 2026-09-22T22:00:00+00:00 --previous-artifact data/derived/reports/baseline_2026-09-22.json
uv run python -m cfdtrader.analysis.baseline_report    --data-root data --reports-dir data/derived/reports --runs-root runs --as-of 2026-09-22T22:00:00+00:00 --raw --previous-artifact data/derived/reports/baseline_2026-09-22.json
uv run python -m cfdtrader.analysis.model_comparison   --data-root data --reports-dir data/derived/reports --runs-root runs --as-of 2026-09-22T22:00:00+00:00 --previous-artifact data/derived/reports/model_comparison_2026-09-22.json
uv run python -m cfdtrader.analysis.pipeline_report    --data-root data --reports-dir data/derived/reports --as-of 2026-09-23T22:00:00+00:00 --previous-artifact data/derived/reports/pipeline_backtest_2026-09-23.json
uv run python -m cfdtrader.analysis.gate_sweep         --data-root data --reports-dir data/derived/reports --as-of 2026-09-23T22:00:00+00:00
uv run python -m cfdtrader.analysis.phase2_dominance   --data-root data --reports-dir data/derived/reports --as-of 2026-09-24T00:00:00+00:00 --previous-artifact data/derived/reports/phase2_dominance_2026-09-24.json
```

`rc=0` en todos, incluido cuando el veredicto es `fail`/`not_evaluable` (son resultados legítimos).
Regenerar los artefactos de `data/` **no** cambia el árbol de git.

### Poda del registro de modelos (`runs/`)

La identidad de la muestra (`matrix_sha256`, en `runs/<sha>/config.json`) **cambia cuando
crece el almacén**. Una entrada construida sobre una ventana anterior **no es reconstruible**
y `model_comparison` la declara `not_evaluable` con `StaleRunWindowError` —por diseño,
#136: se declara el motivo, no una discrepancia de cifras—. Eso deja la matriz **incompleta**,
la selección sin resolver y `phase2_dominance` sin `pbo`, así que el paso 10 falla.

El informe dice **él mismo** qué entradas sobran. Se podan (como en #136) y se repiten los
dos últimos pasos:

```bash
uv run python - <<'PY'
import json, pathlib, shutil

report = json.loads(
    pathlib.Path("data/derived/reports/model_comparison_2026-09-22.json").read_text(encoding="utf-8")
)
stale = [e["run_sha256"] for e in report["not_evaluable"] if e.get("error") == "StaleRunWindowError"]
for sha in stale:
    shutil.rmtree(pathlib.Path("runs") / sha)
print(f"podadas {len(stale)} entradas sobre otra ventana")
PY

# y se repiten el 7 y el 10 con el registro ya limpio
uv run python -m cfdtrader.analysis.model_comparison --data-root data --reports-dir data/derived/reports --runs-root runs --as-of 2026-09-22T22:00:00+00:00
uv run python -m cfdtrader.analysis.phase2_dominance --data-root data --reports-dir data/derived/reports --as-of 2026-09-24T00:00:00+00:00 --previous-artifact data/derived/reports/phase2_dominance_2026-09-24.json
```

Con el registro ya solo de la ventana vigente, el paso 7 vuelve a `selected` (**4/4**, medido el
2026-10-06) y el 10 emite. La poda es **la contrapartida** de la regla 1 de `process.md`: si la
ventana se mide en vez de fijarse, el registro que la declaraba hay que mantenerlo vivo.

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
    --reference-artifact data/derived/reports/pipeline_backtest_2026-09-23.json \
    --reports-dir data/derived/reports \
    --as-of "$(date -u +%Y-%m-%d)T22:00:00+00:00"
```

`--reference-artifact` es el informe del pipeline (#28): de él salen la media y la **σ** del backtest
(la **serie declarada** de `arms.coste_declarado`), que **no** se recalculan con la muestra del *paper*.

Con `N < 30` sale **`not_evaluable`** (normal durante la observación; nunca un aprobado por silencio).
Divergencia `> 2σ` obliga a **auditar antes de operar**.

> **Cuánto tarda en haber muestra.** `N` cuenta **solo las recomendaciones direccionales**
> (`long`/`short`); las de `nothing` y las «no se» se cuentan aparte y **no** entran en la media
> (§16). Como el gate solo autoriza el **tier A** (`p > 0,58`, §19.12), los días operados son
> minoría: llegar a `N ≥ 30` es cosa de **meses**, no de semanas. Es una consecuencia
> **pre-registrada** del diseño, no un defecto; forzar el ritmo sería mover la portería (§11.6).

## Checklist de la sesión

1. `git rev-parse HEAD` limpio y `uv sync` al día.
2. Ingesta (§1) con `rc=0`.
3. Regeneración (§2) con `rc=0`.
4. `run_daily` (§3) con `rc=0` y una fila nueva en `journal/decisions/`.
5. `paper_trading` (§4) y lectura del `state` del veredicto.
6. La decisión **se ejecuta a mano** (o no): el sistema **no** coloca órdenes. Con `tier B` (o un
   `bloqueo`), la salida es `NOTHING`.
