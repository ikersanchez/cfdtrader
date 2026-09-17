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
justo lo que la Fase 0 tiene que medir. Seguimiento en **#50**.

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
  `bid_ask: false`, `reason`, `checked_on` y `follow_up_issue: 50`.
- El informe de cobertura emite `phase1_ready: false` con el bloqueo
  `cfd_source_missing`; con `--require-ready` el proceso sale con **2**.
- El test `test_cfd_has_no_alias_to_index_or_future` falla si alguien introduce
  un mapeo de `SPX500:CFD` a `^GSPC`, `ES=F` o `SPY`, aquí o en el mapa de
  símbolos de un adaptador.
- Tras la ejecución real **no hay ninguna fila con `series_id = 'SPX500:CFD'`** en
  `raw.market_daily` ni en `raw.market_intraday`, y ninguna fila de `^GSPC`,
  `ES=F` o `SPY` se etiqueta como CFD.

## Limitación conocida del `open` diario de `^GSPC` (seguimiento en #52)

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
sobre la muestra limpia. La decisión de qué fuente usar para la apertura es la
**#52**.

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
