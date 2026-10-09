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
> [`news_sources_2026-10-09.md`](news_sources_2026-10-09.md).
>
> **Tolerancia por fuente y códigos de salida (#145).** Una fuente bloqueada, con `429` o caída
> **no** tumba el lote: se declara su estado en `report.sources`
> (`ok` | `blocked` | `rate_limited` | `unavailable` | `error`) y se sigue con las demás.
> - `rc=0` — alguna fuente entregó, o ninguna falló de verdad (todas `unavailable`: día sin noticias).
> - `rc=1` — **ninguna** fuente entregó y alguna falló. No se escriben filas inventadas.
> - `rc=2` — configuración inválida (motivo por `stderr`).

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

### El registro que ve `model_comparison` es el de **familias**, no el del barrido (`#82`)

`runs/` es un registro **compartido** (`DEFAULT_RUNS_ROOT`). El barrido de §19.17 (#82) registra ahí
sus `BUDGET = 10` ensayos (`lightgbm_search_v1#<nombre>`): tras el barrido el registro tiene **14**
entradas —las 4 de #24/#25/#26 más esas 10—, que es el estado **declarado** (es el `n_trials` que
publica el informe del barrido). Pero `model_comparison` reconstruye **una columna por entrada** y
sólo sabe reconstruir `baseline_logit_elasticnet_v1` y `lightgbm_gbdt_v1`:

- un ensayo `lightgbm_search_v1#*` —y **también cualquier entrada de LightGBM de una ventana
  anterior**: su mapa `known` sólo tiene las dos que acaba de re-entrenar— cae a
  `reconstruct_baseline` → `UnknownVariantError`;
- la matriz queda incompleta (`n_variants < registry_n_trials`), la selección se declara
  `not_evaluable` y **no hay `pbo`**: el pipeline no puede copiarlo y `phase2_dominance` falla con
  «falta el campo obligatorio `pbo`» (A10/A5).

El orden declarado es: **cadena §2 sobre el registro de modelos y, después, el barrido.** Para
regenerar los pasos 7-10 hay que **apartar** los ensayos del barrido, correr la cadena y
devolverlos; el registro de familias debe quedar en **4** entradas (2 lineales + las 2 de LightGBM
de la ventana vigente):

```bash
STASH=.scratch/runs_search; mkdir -p "$STASH"; trap 'for d in "$STASH"/*/; do mv "$d" runs/; done' EXIT
for d in runs/*/; do
    case "$(python3 -c "import json;print(json.load(open('$d/config.json'))['config']['variant_id'])")" in
        lightgbm_search_v1#*) mv "$d" "$STASH/" ;;
    esac
done
uv run python -m cfdtrader.analysis.model_comparison --data-root data --reports-dir data/derived/reports --runs-root runs --as-of 2026-09-22T22:00:00+00:00 --previous-artifact data/derived/reports/model_comparison_2026-09-22.json
uv run python -m cfdtrader.analysis.pipeline_report  --data-root data --reports-dir data/derived/reports --as-of 2026-09-23T22:00:00+00:00 --previous-artifact data/derived/reports/pipeline_backtest_2026-09-23.json
uv run python -m cfdtrader.analysis.phase2_dominance --data-root data --reports-dir data/derived/reports --as-of 2026-09-24T00:00:00+00:00 --previous-artifact data/derived/reports/phase2_dominance_2026-09-24.json
```

Con los ensayos apartados, el paso 7 vuelve a `selected` (**4/4**) y el 10 emite. La poda de la
subsección anterior **no** basta por sí sola cuando ya corrió el barrido: hay que apartarlos.



## 3. Camino diario — la pista, o el estado «sin recomendación»

Antes del camino, captura el pre-mercado del futuro ES (#141): es un **dato declarado** de la
decisión, no una feature del modelo. Se toma a las **08:45 ET** (el snapshot de `plan.md` §13).

```bash
uv run python -m cfdtrader.analysis.premarket_gap \
    --as-of "$(date -u +%Y-%m-%d)T13:00:00+00:00" \
    --reports-dir data/derived/reports \
    --capture-bars "journal/ops/$(date -u +%Y-%m-%d)/premarket_bars.json"
```

> Necesita red (descarga `ES=F` con horas extendidas). Si la fuente falla o no hay pre-mercado, el
> informe sale con su estado `unavailable` y **no** tumba nada; el camino diario sigue sin ese bloque.

```bash
uv run python -m cfdtrader.delivery.run_daily \
    --as-of "$(date -u +%Y-%m-%d)T13:00:00+00:00" \
    --variant-id lightgbm_gbdt_v1 \
    --journal-root journal \
    --runs-root runs \
    --git-commit "$(git rev-parse HEAD)" \
    --premarket-bars "journal/ops/$(date -u +%Y-%m-%d)/premarket_bars.json"
```

> `--premarket-bars` es **opcional**: sin él, el informe simplemente no trae el bloque del ES. El
> camino diario **no** descarga datos; solo lee las barras que capturó el paso anterior (#141).

Salida `0`: informe emitido (`recommendation`, o `no_recommendation_stale_data`/`_data_quality`; un
día de mercado cerrado es `recommendation` con `NOTHING` justificado). Salida `2`: falta un argumento
o el pipeline falló, con el motivo por `stderr`. **Cada ejecución escribe su fila** en
`journal/decisions/` y deja `journal/ops/<sesión>/run_log.jsonl` + `manifest.json`.

### La valla de cartera del *kill switch* (§12 reglas 3, 4 y 5) — #83

El camino diario **ya** pasa al gate las tres cifras de pérdida realizada (`daily_pnl_pct`,
`weekly_pnl_pct`, `monthly_pnl_pct`), recomputadas de `journal/trades/` (la operación **real**, #47).
Mientras ese directorio esté vacío —la situación de la **observación**, que no es un olvido,
`plan.md` §19.11— las tres van a `None` y el gate no cambia de comportamiento. En cuanto haya
operaciones cerradas, una pérdida acumulada que alcance el **−5 % semanal** o el **−10 % mensual**
bloquea la sesión en el gate (`bloqueo: 4:perdida_semanal` / `5:perdida_mensual`) y el conteo sale en
el `manifest` de la sesión (`portfolio_closed_trades`).

Para **ver el estado de la valla** antes de decidir (qué pérdida lleva el día, la semana ISO y el mes,
y contra qué umbral), el módulo publica su propio informe:

```bash
uv run python -m cfdtrader.analysis.portfolio_rules \
    --journal-root journal \
    --session "$(date -u +%Y-%m-%d)" \
    --as-of "$(date -u +%Y-%m-%d)T12:00:00+00:00" \
    --reports-dir data/derived/reports
```

> `--source trades` (por defecto) lee `journal.trades`; `--source recomputed` recomputa el resultado
> con la máquina de #45 desde `journal.decisions` más el almacén y exige `--data-root`. Las tres
> cifras **nunca se suman**: son % del **capital**, ya convertidas desde % del nocional con el
> apalancamiento de cada operación (§19.20). Sin operaciones cerradas el estado es `sin_historial` y
> las tres salen `null`, nunca `0`.

## 3 bis. Ejecución: el *bracket* y el registro real (#84)

Solo si la pista es **direccional**. El billete de ejecución y su protocolo de fallo salen del
módulo `delivery/bracket.py`:

```bash
# 09:30 ET, con el relleno de la subasta delante (los pasos y los precios del billete):
uv run python -m cfdtrader.delivery.bracket \
    --journal-root journal \
    --session "$(date -u +%Y-%m-%d)" \
    --entry-px 7839.0

# 16:15 ET, con los datos REALES de la sesión (escribe la fila de `journal.trades`):
uv run python -m cfdtrader.delivery.bracket \
    --journal-root journal \
    --session "$(date -u +%Y-%m-%d)" \
    --record --entry-px 7839.0 --exit-px 7861.5 \
    --exit-reason close --costs-pct 0.0042 \
    --entry-time "$(date -u +%Y-%m-%d)T13:30:00+00:00" \
    --exit-time "$(date -u +%Y-%m-%d)T20:00:00+00:00"
```

> La entrada es el **relleno de la subasta** y no se inventa: sin `--entry-px`, el billete publica
> los `%` y la instrucción de multiplicarlos por el relleno. `--overnight` marca el incumplimiento de
> las 16:00 ET (`closed_by_close = false`, con la financiación en `--costs-pct`). El motivo de salida
> es del vocabulario `target`/`stop`/`close`, el mismo que #45 usa al recomputar.
>
> **Las dos patas o ninguna.** Si el bróker no acepta el *bracket*, **no se opera** (y no se escribe
> ninguna fila); si acepta solo una, se **cierra de inmediato** y se registra el cierre real. El
> protocolo entero, con lo que se registra en cada caso, lo imprime el propio billete.

La fila escrita es la que lee la valla de cartera de §3 (`#83`): `pnl_pct` va **neto y en % del
nocional** (el recorrido de precio con signo menos el coste efectivo), y es #83 quien lo convierte a
`%` del capital con el apalancamiento.

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
