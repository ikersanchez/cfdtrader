# Fuentes de noticias gratuitas — medición y decisión (issue #142)

- **Fecha:** 2026-10-09 (medido ~07:45–08:05 UTC, desde este entorno)
- **Seguimiento de:** #30 (ingesta de noticias, ya implementada) · #51 (mismo ejercicio para mercado)
- **Veredicto (a):** lista de fuentes evaluadas con su **resultado medido** (abajo)
- **Veredicto (b):** **sí hay fuente gratuita ejecutable desde el código** — varios RSS/Atom sobre `RssAdapter`, **tras corregir un defecto del cliente HTTP** (ver «Defecto de código»).
- **Veredicto (c):** **archivo histórico libre: parcial.** GDELT es libre y retrospectivo (2013+), pero su marca temporal es **la de ingesta (ventana de 15 min)**, no la de publicación. Sirve para un backtest **conservador**, no para un `published_at` exacto.

## Cómo se midió

- Peticiones reales con `curl` (User-Agent de navegador) y, después, **a través del `CachedHttpClient` real** con `parse_rss_feed`/`RssAdapter` del repo.
- Criterios de aceptación (#142): `published_at` con precisión de **minuto**, cobertura histórica **≥ 5 años**, licencia que permita uso interno y guardar titular+URL, coste 0/bajo, y que **no esté geo/censurada**.

## (a) Fuentes evaluadas — resultado medido

### RSS/Atom — responden y parsean (adaptador existente, sin dependencia nueva)

| Fuente | HTTP | titulares | `published_at` | frescura del más nuevo |
|---|---|---|---|---|
| CNBC Top News | 200 | 30 | segundo | 0,23 h |
| CNBC World | 200 | 30 | segundo | 0,37 h |
| CNBC Energy | 200 | 30 | segundo | 12,75 h |
| Investing.com News | 200 | 10 | segundo | 0,14 h |
| Seeking Alpha Market Currents | 200 | 7 | segundo | 0,09 h |
| Google News (`stock market`) | 200 | 102 | segundo | 0,23 h |
| Google News (`iran oil sanctions`) | 200 | — | segundo | — |
| BBC World | 200 | 35 | segundo | 0,68 h |
| Al Jazeera | 200 | 25 | segundo | 0,85 h |
| OilPrice | 200 | 15 | segundo | 0,77 h |
| The Hill | 200 | 15 | segundo | 4,98 h |
| Nasdaq Markets | 200 | 15 | segundo | 0,37 h |
| FT Home | 200 | 10 | segundo | 3,27 h |
| MarketWatch Top | 200 | 10 | minuto | 6,90 h |
| WSJ Markets | 200 | 20 | minuto | **obsoleto (2025-01) → descartar** |

> El **RSS no tiene archivo**: el canal sirve una ventana móvil (últimos ~10-100 titulares). Vale para el overlay en vivo; **no** para retrotestear.

### RSS/Atom — no usables hoy

| Fuente | Resultado |
|---|---|
| Reuters (`feeds.reuters.com/...`) | host muerto (000) |
| Reuters (`reutersagency.com/feed/`) | 404 |
| Yahoo Finance (`finance.yahoo.com/news/rssindex`) | 404 |
| Politico | 403 |
| AP vía RSSHub (`rsshub.app/apnews/...`) | 403 |

### GDELT — la fuente **primaria** declarada (`tech_stack.md` §4.8): **no ejecutable**

| Petición | Resultado |
|---|---|
| DOC API `api.gdeltproject.org/api/v2/doc/doc` | **429**, luego *timeout* SSL (reintentos) |
| Archivo `data.gdeltproject.org/gdeltv2/lastupdate.txt` | 206 ✓ |
| GDELT 2.0 GKG `…/20160218230000.gkg.csv.zip` | 206, 14,2 MB ✓ |
| GDELT 1.0 GKG `…/gkg/20130401.gkg.csv.zip` | 206 ✓ |
| GDELT 1.0 GKG `…/gkg/20150217.gkg.csv.zip` | 206 ✓ |

### Agregadores *freemium* — requieren clave (no medida la cobertura, no hay clave en el entorno)

| Fuente | Sin clave |
|---|---|
| NewsAPI | 401 `apiKeyMissing` |
| NewsData.io | 401 |
| Marketaux | 401 |
| Finnhub | 401 |
| Alpha Vantage News | 200 con `demo` (solo demo) |

### X/Twitter — ninguna opción gratuita ejecutable

| Vía | Resultado |
|---|---|
| API oficial `api.twitter.com/2` | 401 |
| Nitter (`nitter.net`, `nitter.poast.org`) | sin respuesta (000) |
| Nitter (`nitter.space`, `…privacyredirect…`) | 403 / 404 |
| Nitter `nitter.tiekoetter.com/…/rss` | 200 **pero muro anti-bot** (`<item>`: 0) |
| `xcancel.com/…/rss` | 451 |
| RSSHub `rsshub.app/twitter/user/…` | 404 |
| `thetrumparchive.com` / `factba.se/topic/twitter` | 200 (webs, **sin API**) |

### Caso energía/Irán

| Fuente | Resultado |
|---|---|
| EIA Today in Energy (`eia.gov/rss/todayinenergy.xml`) | *timeout* (000) |
| CNBC Energy | 200 — sirvió, el 2026-10-09, «Treasury sanctions 17 tankers linked to Iran's 'shadow fleet'» |
| Google News `iran oil sanctions` | 200 |

## (c) Archivo histórico

| Corpus | Cobertura | Granularidad | Marca temporal |
|---|---|---|---|
| GDELT 1.0 GKG | 2013-04-01 → 2015-02-17 | fichero **diario** | del día del fichero |
| GDELT 2.0 GKG | 2015-02-18 → hoy | fichero cada **15 min** | de la ventana de ingesta |

**Comprobado en el GKG real** (fichero `20261009074500`): el campo 2 (`V2.1DATE`) vale
`20261009074500` en **las 773 filas** del fichero — es la **ventana de ingesta**, idéntica para
todo el fichero. `PAGE_PRECISEPUBLISHDATE` aparece **0 veces**; el fichero `mentions` usa la misma
marca. ⇒ **GDELT no da la fecha de publicación del artículo**, solo la de ingesta (≤ 15 min de
cuantización). Confirma el punto 3 de #142.

**Consecuencia honesta:** el archivo libre permite un backtest **conservador** (la noticia estuvo
disponible, como mucho, en esa ventana; nunca introduce *look-ahead*), pero **no** un
`published_at` de precisión de minuto. Un retrotest con `published_at` exacto sigue exigiendo el
archivo de pago (`tech_stack.md` §4.9, decisión 2). El **RSS no aporta archivo** (ventana móvil).

## Defectos de código encontrados

### Corregidos (bloqueaban el criterio (b))

`CachedHttpClient` (`src/cfdtrader/data/sources/http.py`) **rechazaba todos los RSS/Atom**:

1. `LOCK_DETECTOR` incluía `b"<?xml"` como firma de bloqueo, y un RSS/Atom **empieza** por ahí.
2. `ACCEPTED_CONTENT_TYPES` no incluía `text/xml`, `application/xml`, `application/rss+xml`,
   `application/atom+xml`, ni el subtipo ``+xml`` (algún medio sirve `rss+xml` a secas).
3. `RssAdapter` no enviaba `User-Agent`: `cnbc.com` (403 «Access Denied») y `nasdaq.com` cortan
   la conexión a `python-httpx`, aunque sirvan el feed a un navegador.

Corregido y con tests (`test_an_xml_feed_is_data_and_not_a_block`,
`test_a_bare_rss_xml_content_type_is_accepted`,
`test_an_html_error_page_wrapped_in_xml_is_still_blocked`,
`test_a12_the_rss_adapter_reads_a_feed_through_the_real_cached_client`,
`test_a13_the_rss_adapter_sends_a_browser_user_agent`). Las páginas HTML de error siguen
cazándose por `<!DOCTYPE`/`<html`.

### Abiertos (fuera de #142 — follow-ups)

1. **Identidad del almacén colisiona con noticias.** La identidad es `(source, series_id, as_of)`
   y en noticias `as_of = published_at` y `series_id = feed` (una etiqueta, no un id único). Dos
   titulares del **mismo feed** con el **mismo instante** chocan y el lote entero falla con
   `InvalidRecordError: la escritura repite una misma identidad más de una vez`. Medido: de 324
   titulares reales, **3 colisionan** (todos en `google-news`); los otros 12 feeds, 0. La ingesta
   de la prueba end-to-end **excluye** `google-news` por esto.
2. **Un feed bloqueado tumba el lote.** `news.main` propaga la excepción del primer `--feed` que
   responde *blocked*; no hay tolerancia por-feed (`ft-home` sirvió `text/html` en una de las
   rondas). El patrón de estados `rate_limited`/`blocked` existe en el cliente, pero el CLI
   no lo resume: aborta.
3. **La API de GDELT no responde** (429/*timeout*): la fuente **primaria** declarada en
   `tech_stack.md` §4.8 requiere un plan B ya (¿`gdeltdoc`, otro *endpoint*, o retirarla de
   «primaria»?).

## Prueba end-to-end (criterio (b))

Con el cliente corregido, `cfdtrader.data.news` **escribe de verdad** en `raw.news_headlines`:

```console
$ uv run python -m cfdtrader.data.news --data-root /tmp/news_probe_store --as-of "<ISO-8601Z>" \
    --feed 'cnbc-top=…' --feed 'cnbc-world=…' --feed 'cnbc-energy=…' --feed 'investing=…' \
    --feed 'seekingalpha=…' --feed 'oilprice=…' --feed 'nasdaq=…' --feed 'bbc-world=…' \
    --feed 'aljazeera=…' --feed 'marketwatch=…' --feed 'thehill=…'
{"as_of": "2026-10-09T07:58:05+00:00", "fetched": 222, "discarded_future": 0,
 "discarded_duplicate": 20, "new": 202, "outcome": "created", "headline_hashes": [ … ]}
exit=0

# raw.news_headlines
┌─────┬───────┐
│ n   ┆ feeds │
│ 202 ┆ 11    │
└─────┴───────┘
```

Muestra real (el caso energía/Irán que motiva #142 y #143):

| feed | `published_at` | titular |
|---|---|---|
| oilprice | 2026-10-08 21:00 UTC | Iran War Energy Shock Puts Hyd… |
| cnbc-energy | 2026-09-30 09:45 UTC | Trump denies offering Iran san… |
| oilprice | 2026-10-08 20:00 UTC | Hurricane Isaias Shuts In 1.28… |

> Se **excluyen** de esta corrida `google-news` (colisión de identidad, ver arriba) y `ft-home`
> (bloqueo). Con esos dos, el lote **aborta** — es el defecto #2, no un problema de la fuente.

## Veredicto

| Criterio de #142 | Resultado |
|---|---|
| (a) lista de fuentes evaluadas con resultado **medido** | ✅ arriba, con código HTTP |
| (b) **≥ 1 fuente gratuita ejecutable desde el código** | ✅ **12 feeds** vía `RssAdapter` (tras 3 correcciones del cliente); ingesta real de **202 filas** |
| (c) veredicto sobre **archivo histórico** | ⚠️ **parcial**: GDELT libre y retrospectivo (2013+), pero con marca temporal de **ingesta**, no de publicación |

## Reproducir

```console
# (B) Ingesta real de las fuentes ejecutables (tras las correcciones del cliente)
uv run python -m cfdtrader.data.news --data-root data --as-of "<ISO-8601 con zona>" \
  --feed 'cnbc-top=https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114' \
  --feed 'cnbc-energy=https://www.cnbc.com/id/19836768/device/rss/rss.html' \
  --feed 'oilprice=https://oilprice.com/rss/main' \
  --feed 'google-news=https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en'

# (C) Comprobar el archivo GDELT: la fecha es la ventana de ingesta, no la publicación
curl -sL 'https://data.gdeltproject.org/gdeltv2/lastupdate.txt'   # 206
```

## Valla de honestidad

- Encontrar fuentes **no** demuestra que las noticias tengan *edge*: sin archivo con `published_at`
  exacto no hay medición retrospectiva y `phase2_ready` sigue `false` / carril B bloqueado. La capa
  LLM sigue **solo como redactora** del informe.
- GDELT —fuente **primaria** declarada— **no** responde por su API; el archivo sí, con la limitación
  de la marca temporal. No se declara «resuelto» lo que solo funciona a medias.
- Las fuentes de **X/Twitter** que el propietario quería (Trump, geopolitica) **no** tienen opción
  gratuita ejecutable: quedan fuera con su evidencia (401/403/404/451 y muro anti-bot).
