# Fuentes de datos — cobertura, licencias y límites

> Artefacto de la **tarea #3** (`_docs/tasks.md`). Documenta, por fuente, qué se
> obtiene de verdad, con qué licencia y con qué límites, y deja por escrito la
> **ausencia de fuente del `SPX500:CFD`** con la evidencia de la comprobación.
>
> Declaración de fuentes en `config/data_sources.yaml` (series y respaldos) y en
> `config/macro_series.yaml` (series macro). Verificación: **2026-09-17**.

## Resumen

| Fuente | Papel | Granularidad | Cobertura obtenida | bid/ask | Licencia y límites |
|---|---|---|---|---|---|
| **`yfinance`** (Yahoo Finance) | **Primaria** de mercado | Diaria e intradía 5m | Diaria desde 2005 (5.461 filas de `^GSPC`); intradía 5m: **60 días** (rodante) | **No** | Librería Apache-2.0. Los datos de Yahoo son de uso **estrictamente personal**: no se redistribuyen. *Scraper* no oficial → siempre detrás de adaptador, con respaldo y alerta de dato *stale* |
| **Stooq** | **Respaldo declarado** del diario | Diaria | **No usable hoy** (ver más abajo) | No | Gratuito para uso personal y educativo; el uso comercial puede estar restringido; no redistribuir |
| **FRED / ALFRED** | **Primaria** de macro US | Serie × fecha | ~2005 → hoy por serie (limitada por `min_start`) | — | Datos de FRED: uso libre **con atribución**. Requiere clave gratuita. ALFRED aporta las *vintages*, que es lo que hace posible el *point-in-time* |
| **`SPX500:CFD`** | Instrumento del proyecto | — | **SIN FUENTE** | **No** | Ver la sección «SIN FUENTE» |

Ninguna fuente entrega **bid/ask**: ni el índice, ni el futuro, ni Stooq. El
diferencial real del CFD solo se puede medir contra el bróker (tarea #8).

## `yfinance` (Yahoo Finance) — fuente primaria de mercado

- **Granularidad.** Diaria (`1d`) e intradía de 5 minutos (`5m`). El límite
  rodante del proveedor para `5m` es de **~60 días** (y ~7 días para `1m`), así
  que el intradía **no** es histórico: es una ventana móvil que hay que ir
  acumulando. El diario no tiene ese problema.
- **Cobertura obtenida** (medida el 2026-09-17, no estimada):

  | Consulta | Filas |
  |---|---|
  | `^GSPC` diario 2024-01-02 → 2024-01-10 | 6 |
  | `^GSPC` diario desde 2005-01-01 | **5.461** |
  | `^GSPC` 5m, `period="60d"` | **4.661** |
  | `ES=F` 5m, `period="60d"` | **13.561** |

- **Timezone.** Yahoo devuelve el índice intradía con zona `America/New_York`.
  El proyecto **normaliza a UTC** y ancla cada barra diaria al **cierre de sesión
  (16:00 ET)**, nunca a medianoche (`_docs/plan.md` §8.2): el `2024-03-15` (EDT)
  es `2024-03-15 20:00 UTC` y el `2024-01-15` (EST) es `21:00 UTC`.
- **`published_at`.** Yahoo **no** publica una marca por barra: la columna queda
  `NULL` y el almacén decide visibilidad con `fetched_at`. Rellenarla con
  `fetched_at` haría pasar una descarga por publicación y destruiría el
  *point-in-time*.
- **Licencia y límites.** La librería `yfinance` es Apache-2.0. Los datos son de
  Yahoo: uso personal, **sin redistribución**. Al ser un *scraper* no oficial,
  puede romperse cuando Yahoo cambie algo → adaptador propio, dos configuraciones
  de descarga (`yf.download` y `yf.Ticker().history`), reintentos con backoff y
  caché en disco de la respuesta cruda.
- **Series servidas hoy:** `^GSPC`, `ES=F`, `SPY`, `^VIX`, los 11 ETFs
  sectoriales SPDR (`XLK`, `XLF`, `XLE`, `XLV`, `XLY`, `XLP`, `XLI`, `XLU`,
  `XLB`, `XLRE`, `XLC`), contexto (`^GDAXI`, `^FTSE`, `^STOXX50E`, `^N225`,
  `^HSI`), divisa (`DX-Y.NYB`, `EURUSD=X`) y materias primas (`BZ=F`, `CL=F`,
  `GC=F`).

### Comandos de verificación

```console
$ uv run python -c "import yfinance as yf; print(len(yf.download('^GSPC', start='2005-01-01', auto_adjust=False, progress=False)))"
5461
$ uv run python -c "import yfinance as yf; print(len(yf.download('^GSPC', period='60d', interval='5m', auto_adjust=False, progress=False)))"
4661
$ uv run python -c "import yfinance as yf; print(len(yf.download('ES=F', period='60d', interval='5m', auto_adjust=False, progress=False)))"
13561
```

> Nota: un `429` visto con `curl` sobre un endpoint de Yahoo **no** significa que
> Yahoo no sirva el dato. `yfinance` resuelve el *crumb* y la sesión por su
> cuenta; la comprobación válida es con `yfinance`, no con `curl`.

## Stooq — respaldo declarado y **no usable hoy**

`tech_stack.md` §4.5 lo nombra como respaldo del histórico diario, y sigue
declarado como tal en `config/data_sources.yaml`. Pero **hoy no es usable
programáticamente**: responde `200` con un *challenge* JavaScript en lugar del
CSV. El adaptador lo traduce a `SourceBlockedError` con el `Content-Type` y los
primeros bytes en el mensaje, y el informe lo declara con estado `blocked`.

```console
$ curl -sS -o /dev/null -w 'HTTP %{http_code} | %{content_type} | %{size_download} bytes\n' 'https://stooq.com/q/d/l/?s=^spx&i=d'
HTTP 200 | text/html; charset=utf-8 | 796 bytes
$ curl -sS 'https://stooq.com/q/d/l/?s=^spx&i=d' | head -c 120
<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="robots" content="no
index,nofollow"></head><body><noscript>This site requires JavaScript to verify y
our browser. Please
```

Con `User-Agent` de navegador el resultado es **idéntico** (mismo cuerpo de 796
bytes), así que no es un problema de cabeceras: es verificación por JavaScript.

**Consecuencia:** el respaldo de facto del diario es `yfinance` con sus **dos
configuraciones de descarga**. No se contrata ni se inventa ninguna fuente nueva,
y el *challenge* no genera issue de seguimiento.

## FRED / ALFRED — fuente primaria de macro americana

- **Series** (`config/macro_series.yaml`): fed funds efectivo (`DFF`), Treasury a
  2 y 10 años (`DGS2`, `DGS10`), pendiente 2s10s (`T10Y2Y`), `CPIAUCSL`, `PCEPI`
  y nóminas no agrícolas (`PAYEMS`).
- **Point-in-time.** `as_of` es el **periodo observado** y `published_at` el
  **instante de publicación** en UTC: las series de calendario (BLS/BEA) toman la
  fecha de la *vintage* de ALFRED (`realtime_start`) a las **08:30 ET**; las
  series que fija el mercado usan el día al que se refieren más el
  desplazamiento declarado (Treasury a las 15:30 ET; fed funds efectivo al día
  siguiente a las 09:00 ET). Cuando no se puede determinar, queda `NULL`.
- **Las series de calendario piden la ventana de *vintages* completa**
  (`realtime_start = min_start`, `realtime_end = 9999-12-31`) y se guarda la
  **primera publicación** de cada observación. Comprobado contra la API el
  2026-09-17: sin pedirla, FRED devuelve la última vintage y cada observación trae
  `realtime_start` = el día de la consulta, de modo que las 260 observaciones de
  `CPIAUCSL` desde 2005 quedarían publicadas «hoy» y el *point-in-time* sería
  decorativo. El coste de pedirla es pequeño (0,1 MB para CPI, 1.367 filas) y el
  valor guardado es el **primer publicado**, que es el que movió el mercado; una
  vintage con `.` no es una publicación y no adelanta la fecha.
- **Requisito:** clave gratuita (`FRED_API_KEY`). Sin ella **no se ingesta nada**:
  cada serie se declara `unavailable` y el proceso sale con código 3, para que
  «no hay clave» no se confunda con «ya está ingestado».
- **Licencia.** Datos de FRED de uso libre **con atribución**.

```console
$ curl -sS -o /dev/null -w 'HTTP %{http_code}\n' 'https://api.stlouisfed.org/fred/series/observations?series_id=DFF&file_type=json'
HTTP 400   # falta api_key: la API responde, exige clave
```

## `SPX500:CFD` — SIN FUENTE

**Estado: no existe fuente pública gratuita del intradía histórico del
`SPX500:CFD` ni de su `bid`/`ask`.** No se sustituye por `^GSPC`, `ES=F` ni
`SPY`: son instrumentos distintos (`_docs/plan.md` §3.1) y hacerlo falsearía la
medición del diferencial, la financiación y el *tracking difference*, que es
justo lo que la Fase 0 tiene que medir. La adquisición del dato real se sigue en
**#107** y la ruta elegida se declara en la sección «Decisión de la fuente de
intradía y bid/ask (tarea #50)».

Comprobado el **2026-09-17** con estos comandos y estos resultados:

| Comprobación | Comando | Resultado observado |
|---|---|---|
| ¿Sirve el índice? | `yf.download('^GSPC', start='2005-01-01', auto_adjust=False)` | 5.461 filas → **sí** |
| ¿Sirve el futuro? | `yf.download('ES=F', period='60d', interval='5m')` | 13.561 filas → **sí** |
| ¿Sirve Stooq el diario? | `curl 'https://stooq.com/q/d/l/?s=^spx&i=d'` | `200 text/html`, *challenge* JS → **no** |
| ¿Hay bid/ask de la serie? | `yf.download('^GSPC', period='60d', interval='5m').columns` | no hay columnas `Bid`/`Ask` → **no** |
| ¿Hay clave de FRED? | `curl 'https://api.stlouisfed.org/fred/series/observations?series_id=DFF&file_type=json'` | `400`, falta `api_key` (no es un fallo de la fuente) |

**Ningún** instrumento de los que sirven estas fuentes entrega `bid`/`ask`: ni el
índice, ni el futuro, ni el ETF. El diferencial se mide contra el bróker en la
tarea #8, no contra una fuente pública.

### Cómo se declara esta ausencia en el código

- `config/data_sources.yaml` → bloque `unavailable:` con `status: unavailable`,
  `bid_ask: false`, `reason`, `checked_on` y `follow_up_issue: 107` (la adquisición del dato real se sigue en #107).
- El informe de cobertura emite `phase1_ready: false` con el bloqueo
  `cfd_source_missing`; con `--require-ready` el proceso sale con **2**.
- El test `test_cfd_has_no_alias_to_index_or_future` falla si alguien introduce
  un mapeo de `SPX500:CFD` a `^GSPC`, `ES=F` o `SPY`, aquí o en el mapa de
  símbolos de un adaptador.
- Tras la ejecución real **no hay ninguna fila con `series_id = 'SPX500:CFD'`** en
  `raw.market_daily` ni en `raw.market_intraday`, y ninguna fila de `^GSPC`,
  `ES=F` o `SPY` se etiqueta como CFD.

## Decisión de la fuente de intradía y bid/ask (tarea #50)

**Ruta elegida: ruta 3 — proxy declarado.** El proyecto **no** sustituye en
silencio el `SPX500:CFD`: lo declara. El tramo intradía se ordena con el proxy
`^GSPC` a 5 min de `raw.market_intraday` (ventana **rodante** del proveedor, ~60
días) y el precio de entrada usa el `open` diario de `^GSPC` como **proxy
declarado** del CFD. La procedencia viaja en los datos y en los informes
(`models/labels.py::ENTRY_PRICE_PROXY_OF`, `analysis/backtest_report.py::PRICE_PROXY_OF`
y `analysis/pipeline_report.py::renamed_series`), nunca como una sustitución
muda. Es el estado de facto del código; aquí queda **decidido y escrito**.

Esta decisión **no** cambia el estado de la Fase 1: `phase1_ready` sigue `false`
con el bloqueo `cfd_source_missing`, porque el CFD real sigue sin fuente.

### Límites del proxy

- Ventana **rodante de ~60 días** para el 5 min (~7 días para el 1 min): **no hay
  intradía de años**.
- **Sin bid/ask**: el spread real del CFD sigue sin medirse (el registro declara
  `bid_ask: false`).
- **No es la cotización del CFD**: no sirve para medir el *tracking difference*
  ni el coste real (§3.1).
- **No desbloquea la Fase 1**: `phase1_ready` sigue `false` con el bloqueo
  `cfd_source_missing`.

### Pliego de adquisición (acción del usuario, #107)

Las rutas **1 (contratar al bróker)** y **2 (exportar del bróker)** siguen
abiertas: la adquisición del dato real es **acción del usuario** y se sigue en
**#107**. Lo que se compraría es **fidelidad de ejecución**, no la economía del
tramo `open→close` (que solo lleva el diferencial declarado).

| Campo | Requisito |
|---|---|
| Instrumento | `SPX500:CFD` **cotizado por el propio bróker** (no el índice, ni el futuro, ni un ETF) |
| Campos | `timestamp`, `bid`, `ask`, `last` |
| Granularidad | 1 min (5 min aceptable) |
| Cobertura | ≥ 5 años |
| Zona horaria | UTC, con la sesión regular 09:30–16:00 ET |
| Licencia | licencia y retención por escrito |

**Regla de oro:** solo cuenta la cotización del propio bróker; ningún proxy
externo se registra como cotización del CFD.

### Seguimientos

**#10 y #11 están cerradas** (y **#52** también): no hay trabajo abierto ahí. El
único seguimiento abierto de la adquisición es **#107**. El proxy declarado **no**
mide el *tracking difference* ni el spread real del CFD.

## `open` diario de `^GSPC` — limitación conocida y decisión de la fuente de apertura

En parte del histórico, el `open` diario que sirve Yahoo para el **índice** es el
**cierre de la sesión anterior repetido**, no la apertura real:

| Año | Sesiones | `open` == cierre anterior |
|---|---|---|
| 2005 | 251 | 241 (96,0 %) |
| 2006 | 251 | 107 (42,6 %) |
| 2012 | 250 | 63 (25,2 %) |
| 2013 | 252 | 66 (26,2 %) |
| 2015 | 252 | 1 (0,4 %) |
| 2016 en adelante | — | 0 (0,0 %) |
| **Total** | 5.459 | 552 (10,1 %) |

Es un artefacto del **índice**, no del ETF (`SPY`) ni del futuro (`ES=F`), que sí
tienen apertura negociada. Importa porque cualquier partición del retorno que use
la apertura (`open→close` frente a `close→open`) queda sesgada en esas sesiones: el
tramo nocturno es cero por construcción y la sesión se queda con todo el retorno.

Medido (2026-09-17): con la muestra completa el estudio del drift daría
`intraday +2,47 pb` frente a `overnight +1,53 pb`; con la muestra limpia (desde
2014, 3.192 sesiones) da `intraday +2,00 pb` (p = 0,19, **no significativo**)
frente a `overnight +2,91 pb` (p = 0,0008). **La conclusión se invierte según se
limpie o no**, así que el estudio de la tarea #6 lo detecta, lo declara y decide
sobre la muestra limpia.

### Decisión (tarea #52, 2026-10-01)

Se **mantiene `^GSPC`** como fuente de apertura del estudio y el `open` repetido
queda como **límite asumido y declarado**, no corregido: el veredicto de #6 ya se
calcula sobre la era limpia. La decisión se midió comparando las tres fuentes
sobre la **misma ventana** (corte limpio `2014-01-01`, artefacto
`drift_open_source_2026-10-01.json`):

| fuente | `open` repetido (muestra) | sesiones (ventana) | `open` repetido (ventana) | intradía pb (p) | nocturno pb (p) | veredicto / puerta |
|---|---|---|---|---|---|---|
| `^GSPC` | 10,11 % | 3.192 | 0,09 % | +2,00 (0,1928) | +2,91 (0,0008) | `overnight` / `fail` |
| `SPY` | 0,90 % | 3.164 | 0,97 % | +1,75 (0,2289) | +3,03 (0,0151) | `overnight` / `fail` |
| `ES=F` | 10,82 % | 3.039 | 4,91 % | +5,67 (0,0035) | −1,02 (0,0740) | `intraday` / `pass` |

`SPY`, la apertura de subasta real del mismo mercado y horario (09:30–16:00 ET),
**confirma** `overnight`/`fail`. `ES=F` **no es sustituto**: es el futuro, de
sesión casi continua, así que su `open→close` no es la sesión regular del
instrumento y da otro veredicto; además arrastra un 4,91 % de `open` repetido en
la ventana. La comparación completa, con su digest y la forma del artefacto, vive
en `drift_open_source_2026-10-01.json`.

## Cómo reproducir la ingesta

```console
# Mercado (único comando de la tarea #3). Con --require-ready sale 2 mientras la
# Fase 1 siga bloqueada por falta de fuente del CFD.
uv run python -m cfdtrader.data.market --data-root data

# Macro (tarea #5). Necesita FRED_API_KEY en .env; sin ella sale con 3.
uv run python -m cfdtrader.data.macro --data-root data
```

Los informes quedan en `data/derived/reports/` (`market_coverage_<fecha>.json` y
`.md`, `macro_coverage_<fecha>.json` y `.md`). La caché de respuestas crudas vive
en `data/cache/<fuente>/<AAAA-MM-DD>/`; **su política de poda es la tarea #44** y
no se decide aquí.
