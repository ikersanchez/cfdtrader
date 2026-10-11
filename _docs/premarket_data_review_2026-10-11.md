# Revisión: ¿hay fuente gratuita del pre-mercado del ES, o se puede adaptar el modelo/decisión?

| Campo | Valor |
|---|---|
| **Fecha** | 2026-10-11 |
| **Origen** | seguimiento de #141 / #148 / #149 / #107 |
| **Estado** | Revisión medida **y corregida**. §0/§2.1: **sí** hay fuente gratuita (Dukascopy) y **sí** cubre la ventana; §2.1.1: su **ToU prohíbe automatizarla** ⇒ sigue **bloqueada**, por **licencia**. **No** adopta nada: `phase2_ready = false`, carril B bloqueado. |
| **Aval de honestidad** | esto **no** afirma *edge*. Un «no hay serie» medido vale más que una feature prometida; y un «no se puede» por licencia vale más que una descarga que rompe un contrato. |

## La pregunta

¿Se puede conseguir **gratis** una serie **histórica point-in-time** del pre-mercado del ES (la
barra de las 08:45 ET) para promoverla a **feature** del modelo? Y si no, ¿qué se puede adaptar
**hoy** con lo que ya hay, tocando el modelo o la decisión final?

## 0. Qué está comprobado y qué no (corrección)

**Corrección de una exageración de la primera versión de esta nota.** Escribí «no hay fuente
gratuita». Eso **no estaba comprobado** y, medido, es **falso** (ver §2.1: **Dukascopy**). Lo que
está comprobado es mucho más estrecho:

| Afirmación | ¿Comprobada? |
|---|---|
| `yfinance` no da más de ~2 años a 1h ni ~60 días a 5m | **Sí**, medido (§1) |
| #149 midió **4** granularidades (`5m/30m/1h/1d`, todas de `yfinance`) y la mejor cubre **604** sesiones frescas frente a **2 688** | **Sí** (`premarket_history.json`) |
| **Ninguna** fuente gratuita tiene la serie histórica | **No** — era una inferencia, y §2.1 la desmiente |
| ¿Se puede **usar esa fuente desde el código**? | **No**: el ToU de Dukascopy prohíbe la automatización sin consentimiento escrito (§2.1.1). El bloqueo es de **licencia**, no de cobertura |
| #107 «bloquea» el dato | **Impreciso**: #107 lo cerró el **propietario** por su criterio (b) —**no adquirir la cotización del CFD del bróker**, que es de pago y es acción suya—, **no** por «no hay fuente gratuita». Son cosas distintas |

## 1. Lo medido hoy (sonda directa sobre `yfinance`, 2026-10-11)

`yfinance`, `history(period="max")`, con y sin horas extendidas (`prepost`):

| Serie | 1m | 5m / 15m / 30m | 1h | 1d |
|---|---|---|---|---|
| **`ES=F`** | 7 días (`2026-10-04`) | **~60 días** (`2026-08-12`) | **~2 años** (`2024-10-11`) | desde 2000 |
| `MES=F` | 7 días | ~60 días | ~2 años | desde 2019 |
| `SPY` | 7 días | ~60 días (con `prepost` **sí** trae pre-mercado: 7 707 vs 3 272 barras) | ~2 años | desde 1993 |
| `^GSPC` | 7 días | ~60 días | ~2 años | desde 1927 |

- Para `ES=F` el `prepost` **apenas cambia nada** en 1h (11 466 vs 11 422 barras): Yahoo sirve el
  futuro casi todo en horario regular, así que **no** hay horas extendidas históricas fiables.
- **Lo máximo gratis son ~2 años a 1h** (y 60 días a 5m). La ventana de entrenamiento del modelo son
  **2 688 sesiones (~10,6 años)**: **no cabe**. Es la misma conclusión que midió #149.

## 2. Fuentes gratuitas **no probadas antes** — y una que sí sirve

### 2.1 Dukascopy: histórico **tick gratuito** del S&P 500 CFD (medido hoy)

`datafeed.dukascopy.com` sirve ficheros `.bi5` (LZMA) con **ticks** gratuitos, por hora, de sus
instrumentos — incluido el índice **`USA500IDXUSD`** (el S&P 500, sesión de 24 h). Medido hoy:

```console
$ curl 'https://datafeed.dukascopy.com/datafeed/USA500IDXUSD/2024/05/03/08h_ticks.bi5'
HTTP 200  8 400 bytes  ->  1 706 ticks (ask/bid), precio ≈ 5 284,269
$ curl 'https://datafeed.dukascopy.com/datafeed/USA500IDXUSD/2014/05/03/12h_ticks.bi5'
HTTP 200  1 337 bytes  ->    161 ticks (ask/bid), precio ≈ 1 919,331   # el S&P estaba ~1 920
```

Los dos precios **cuadran con la realidad** de su fecha (el S&P estaba ~1 920 en junio de 2014 y
~5 284 en junio de 2024). Medición del **inicio** de la cobertura (fichero `12h` UTC = 08:00 ET):

| Fecha | Respuesta |
|---|---|
| 2009, 2010, 2011 | `404` (sin cobertura) |
| 2012-06, 2013-06 | `200`, **0 bytes** (hueco) |
| **2013-12 · 2014-01 · 2014-06 · 2024-06** | `200`, **con datos** (1 280–8 400 bytes) |

⇒ la cobertura arranca a finales de **2013**: **~12,5 años ≈ 3 000 sesiones**, **más** que las
**2 688** que exige el modelo. (Las peticiones intermitentes `503`/`000` son **throttling** del
proveedor, no ausencia de dato: el mismo fichero alterna `200` y `503`.)

Es **tick a tick** del propio subyacente, así que **reconstruye la barra de las 08:45 ET** (el
instante de decisión, 12:45/13:45 UTC) con la granularidad que se quiera. **Es un candidato gratuito
real que #149 no probó**: sus cuatro candidatas eran **todas** `yfinance`.

El dato es gratis y **llega**. Lo que no llega es la **vía**: el proveedor **prohíbe automatizarla**
(§2.1.1). Resumen de lo medido y de lo que queda:

- **Cobertura: medida** (tabla de arriba): desde finales de **2013** ⇒ **~3 000 sesiones**, por encima
  de las **2 688** que exige el modelo.
- **Continuidad fina: no medida, y no se va a medir por esta vía.** La sonda se topó con el **límite
  de peticiones** (el mismo fichero pasó de `200` a `503`/`000` al insistir) y **no se insistió**:
  insistir automatizadamente es justo lo que §2.1.1 descarta.
- **Licencia: medida** — la prohibición de §2.1.1 es el **bloqueo real** de esta vía.
- **Coherencia de instrumento**: es el CFD de **otro** bróker, no el del usuario (`SPX500:CFD`), pero
  **sí** es una cotización de CFD del S&P en sesión de 24 h — más parecida al instrumento operado que
  el futuro `ES=F`.

### 2.1.1 Legalidad: el `datafeed` **no** se puede automatizar (medido hoy)

Es lo que **cierra** esta vía por ahora. Comprobado hoy:

| Comprobación | Resultado |
|---|---|
| `https://datafeed.dukascopy.com/robots.txt` | **`404`** — el host del feed no publica reglas de rastreo |
| `https://www.dukascopy.com/robots.txt` | **`200`**, `Allow: /` salvo rutas internas (`/cgi-bin/`, `/fxspider/…`) ⇒ el sitio **no** veta el rastreo genérico |
| **Terms of Use** (`/swiss/english/legal-pages/terms-of-use/`) | **prohíben** la automatización (cita abajo) |

Cita textual de los *Terms of Use*:

> *«You shall not use or attempt to use any "scraper," "robot," "bot," "spider," "data mining,"
> "computer code," or any other automate device, program, tool, algorithm, process or methodology to
> access, acquire, copy, or monitor any portion of the WEBSITE, any data or content found on or
> accessed through the WEBSITE, or any other WEBSITE information **without the prior express written
> consent of DUKASCOPY**.»*

Y en el mismo documento: *«You may neither take nor attempt any action that … imposes … an
**unreasonable or disproportionately large load** … on the WEBSITE»*, y solo se permite guardar
copias *«for your own **personal, non-commercial use**»*.

**El matiz, dicho en voz alta:** la cláusula habla de «the WEBSITE» y el feed vive en otro host
(`datafeed.dukascopy.com`). Ese matiz **no autoriza**: el mismo documento declara **reservados todos
los derechos** sobre el contenido de Dukascopy, y la propia FAQ de la casa ofrece el dato gratis
**«through Dukascopy's Historical Data Export tool»** —es decir, la vía que **ellos** ofrecen es la
**herramienta**, no el `datafeed` por script—. Lectura prudente, y la que sigo: **automatizar la
descarga no está permitido sin consentimiento escrito previo**.

**Lo que hice hoy, declarado:** una **sonda acotada** de ~25 peticiones pequeñas para comprobar
existencia, cobertura y formato. Es una comprobación de viabilidad, no una adquisición sistemática, y
**se detuvo** en cuanto apareció el ToU; no se descargó ningún histórico.

**Vías que sí son legales, y la decisión que se tomó:**

1. **Herramienta oficial** — *Historical Data Export* (web) o **JForex → Historical Data Manager**:
   la casa da ahí el histórico tick/CSV **gratis** (su FAQ lo declara textualmente). Es una descarga
   **manual**, hecha por la persona ⇒ dentro del uso «personal, non-commercial» del ToU.
2. **Consentimiento escrito** de Dukascopy para automatizar el `datafeed` — el propio ToU lo
   contempla («*without the prior express written consent*»). Es un correo: **acción del propietario**.
3. **Decisión del propietario (2026-10-11): se procede por el `datafeed`, con la condición de no
   sobrecargar el servicio** — y esa condición queda **implementada**, no prometida: una petición por
   sesión, 30 s entre peticiones de red, un hilo, archivo local (ni una petición repetida), tope por
   ejecución y parada en seco ante `429`/`503`. El detalle, en
   `src/cfdtrader/data/sources/dukascopy.py` y en `_docs/data_sources.md` §«Dukascopy».
4. **Nada de esto reabre #107**, que va de la **cotización del CFD del bróker** (cerrada por el
   propietario el 2026-10-09). Aquí se habla de una **serie de pre-mercado para una feature**: otra
   necesidad, otra fuente, otra decisión.

### 2.2 Stooq (probado, no usable)

Diario e intradía devuelven un *challenge* de JavaScript («*This site requires JavaScript to verify
your browser*»): **no** es usable por API, como ya declaraba `data_sources.md`.

### 2.3 Proveedores de pago revisados (declarados, **sin contratar**)

| Proveedor | Plan gratuito | ¿Sirve? |
|---|---|---|
| Polygon.io | «Stocks Basic»: **solo acciones US**, 5 llamadas/min, **2 años** de histórico. Los **futuros** son producto aparte (de pago) | No (ni futuros ni 10 años) |
| Alpha Vantage | **25 llamadas/día**; el intradía histórico (parámetro `month`) es **premium** | No (el histórico de intraday no es gratis) |
| Twelve Data | Plan gratuito con pocas *credits*/min; histórico intradía amplio y **datos de horas extendidas** son de pago ($29–$999/mes) | No en el tramo gratuito |

**Veredicto corregido (datos):** está **medido** que `yfinance` no llega y que Stooq no es usable.
**No** está probado que no exista una fuente gratuita: **Dukascopy sí sirve ticks gratuitos** del
subyacente, y su **antigüedad** (el dato que decidiría) **está pendiente de medir**. La cotización
**en vivo** a las 08:45 está disponible; lo que se discute es cuánto **pasado** hay.

## 3. Qué **sí** se puede hacer con lo disponible (gratis)

### 3.1 Acumular el dato hacia adelante (la única vía gratuita a un feature)

El *plumbing* ya existe: `analysis.premarket_gap` captura las barras de las 08:45 a
`journal/ops/<sesión>/premarket_bars.json` (#141) y `run_daily --premarket-bars` las consume. Lo que
falta es **persistir el resultado** (los tres números: `overnight_move_pct`, `gap_vs_index_pct`,
`basis_pct`) como **serie point-in-time** en el almacén (`derived.premarket_daily`), de modo que,
sesión a sesión, se acumule la historia que hoy no existe. En **~1 año** habría material para
medir; en **~10**, para entrenar. Es gratis y no depende de ningún proveedor de pago.

### 3.2 Ampliar los **datos declarados** de la decisión (inmediato, carril A)

La regla 21 (#148) ya compara el movimiento del ES contra la dirección del modelo y **avisa**. El
mismo patrón se puede extender, sin historia, a los mercados que **también se mueven antes** de las
08:45 y ya están en el almacén: el movimiento **de la misma mañana** de Europa (el Stoxx/DAX llevan
horas abiertos), el dólar y el petróleo. No serían *features* (no hay historia entrenable), sino
**datos declarados + avisos del gate**. Es barato y honesto.

### 3.3 Candidatas **históricas gratis** que sí se pueden medir (presupuesto #24)

Dos series **point-in-time** con historia completa que **hoy no son features**:

- **Basis del futuro**: `ES=F(cierre t-1) / ^GSPC(cierre t-1) − 1`. Ambas barras cierran a las
  16:00 ET del día anterior ⇒ legible en el instantáneo (`design_lag = 0`). Historia: **5 478
  sesiones desde 2005-01-03**.
- **Retorno nocturno del S&P**: `close(t-2) → open(t-1)`, la mitad del *drift* que §1.1.a señala
  como la parte **fuerte** del día. Computable con el diario, con historia completa.

Ninguna de las dos se adopta aquí: se **miden** con la maquinaria de `feature_budget` (#146) y la
puerta de honestidad (DSR/PBO). La expectativa es **baja** (el basis es casi estructural; el `open`
diario arrastra el artefacto de #52), pero es gratis medirlo.

### 3.4 Lo que **no** se puede (con lo verificado)

- **Sustituir la barra de 24 h del ES por el pre-mercado**: su barra `t` cierra a las 16:00 ET y a
  las 08:45 **aún no existe**: usarla sería *look-ahead* (lo mide #141).
- **Un feature entrenado con `yfinance`**: medido, su ventana no llega (#149).
- **⚠️ NO se puede afirmar** que no exista fuente gratuita: **Dukascopy sí sirve el histórico tick del
  S&P, gratis, desde finales de 2013** (§2.1) ⇒ cubre las 2 688 sesiones que el modelo exige. Lo que
  **no** se puede es **automatizar la descarga**: su ToU lo prohíbe sin consentimiento escrito
  (§2.1.1). La promoción sigue **bloqueada**, pero por **licencia**, no por **cobertura** — y las
  vías legales están identificadas.

## 4. Recomendación (revisada a la luz de §2.1 y §2.1.1)

| Opción | Coste | Cuándo sirve | Riesgo |
|---|---|---|---|
| **2.1 Cobertura de Dukascopy** | — | **hecho**: sirve el tick del S&P desde finales de 2013 (~3 000 sesiones) | ninguno |
| **2.1b Conseguir el histórico por la vía legal** (herramienta oficial, a mano) | bajo | días, si el propietario la ejecuta | el volumen (~12 años de ticks) |
| **2.1c Pedir consentimiento escrito para automatizar el `datafeed`** | bajo | semanas (depende de Dukascopy) | que no contesten, o que digan que no |
| ~~2.1d Automatizar el `datafeed` por script~~ | — | **descartado** | **prohibido por el ToU** (§2.1.1) |
| **3.1 Persistir el snapshot (acumular)** | bajo | a partir de ~1 año (medir) / ~10 (entrenar) | ninguno; es solo acumular |
| **3.3 Medir basis y retorno nocturno** | bajo | inmediato | expectativa baja; no adopta nada |
| **3.2 Más datos declarados + avisos** | bajo | inmediato | un aviso que no decide (carril A) |
| **Contratar histórico point-in-time** | de pago | inmediato | decisión del propietario |

**Orden propuesto, corregido otra vez — y esta vez por una razón legal, no técnica:** la cobertura ya
está **medida y llega**; lo que bloquea la vía es el **ToU**. Así que lo primero no es «medir más»,
sino **(2.1b/2.1c) conseguir el dato por una puerta permitida**: la herramienta oficial a mano, o
pedir consentimiento escrito. En paralelo siguen en pie **(3.1)** acumular el snapshot propio (barato
y complementario) y **(3.3)** medir las dos candidatas históricas con la puerta de #146.

**Sin la puerta legal, el veredicto de #149 no cambia**: no hay serie utilizable **para el código**.

## 5. La ingesta tick→barra — **implementada**, con contrato de cortesía

`src/cfdtrader/data/sources/dukascopy.py` (con `tests/test_dukascopy_source.py`, 22 tests y sin red)
implementa la descarga y la decodificación **con el contrato de cortesía dentro del código**, no en
prosa: **una petición por sesión** (la hora UTC que contiene las 08:45 ET, no las 24), **30 s entre
peticiones de red**, **un solo hilo**, **archivo local** (una segunda pasada **no toca la red**),
**tope por ejecución** y **parada en seco** ante `429`/`503` (§2.1.1, decisión del propietario).

Lo que **queda pendiente** es la **promoción**, no la descarga: medir el candidato por la puerta de
#146/#149 y, **solo si pasa**, convertirlo en *feature*. Aquí no se toca nada de eso: `phase2_ready`
sigue `false` y el carril B sigue bloqueado.

**Probado contra la fuente** (2026-10-11, **3 peticiones** en total, 30 s aparte): `2024-06-03` →
`12:44:51Z` = **5 294,132**; `2024-06-04` → `12:44:57Z` = **5 265,376** (fichero `12h`, EDT); y
`2024-01-03` → `13:44:57Z` = **4 727,506** (fichero `13h`, EST: el DST cambia la hora, no solo el
sello). Los valores están en `_docs/data_sources.md` §«Dukascopy».

### 5.1 Qué se reutiliza (no se inventa nada)

- **El contrato de barras ya existe**: `analysis/premarket_gap.py` (#141) define
  `PremarketBar(instant, price)`, `SNAPSHOT_ET = 08:45 ET`, `save_bars` / `load_bars` y la captura
  `--capture-dir` / `--bars-dir` que reproduce **sin red**. La ingesta produce **esas** barras.
- **El criterio ya existe**: `analysis/premarket_history.py` (#149) mide las tres condiciones
  —point-in-time, frescura ≤ `MAX_STALENESS_MINUTES` (60), ventana ≥ `sessions_required()`—, y
  `sessions_required()` se lee de `derived.labels` (medido en el almacén, no estimado).
- **La puerta ya existe**: `analysis/feature_budget.py` (#146) con DSR/PBO.

### 5.2 Dónde vive el dato crudo

`data/raw/dukascopy/<instrumento>/<AAAA>/<MM>/<DD>/<HH>h_ticks.bi5` — dentro de `data/` (**ignorado por
git**: el ToU no permite redistribuir), y la ausencia se archiva como `<HH>h_ticks.empty`, para no
volver a preguntar por una hora que ya se sabe vacía.

### 5.3 Reglas de construcción (las de #141/#149, **no** otras)

1. **La barra del día es la última cotización con instante ≤ 08:45 ET** — 12:45/13:45 UTC según el DST,
   resuelto con `zoneinfo` (`data/calendar.py::EASTERN`), **nunca** con un offset fijo.
2. **`as_of` es el instante del snapshot**, no medianoche (la regla de `data/sources/frames.py`).
3. **Nada de la sesión que se decide.** Sin tick ≤ 08:45 ET el día es una **ausencia declarada** con su
   motivo, **nunca** un `0` (el contrato de `FetchResult`/`PremarketGap`).
4. **Frescura**: se mide y se declara; por encima de 60 min el candidato **no** vale sin decirlo.
5. **`bid`/`ask`**: el `.bi5` los trae (es ask/bid tick a tick) — es la primera fuente que daría
   `bid`/`ask` de verdad. `PremarketBar` hoy solo lleva `instant` y `price`: añadirlos sería una
   **extensión declarada** del contrato de #141, no un campo silencioso.

### 5.4 Piezas: qué está hecho y qué no

Hecho (`src/cfdtrader/data/sources/dukascopy.py`, con `tests/test_dukascopy_source.py`):

1. **Decodificación** del `.bi5` (LZMA *alone* → ticks), con el **mes a cero** de la ruta comprobado
   por test y el **recorte point-in-time** (último tick `<=` 08:45 ET, nunca uno de la sesión que se
   decide).
2. **Descarga de cortesía** (`DukascopyClient`): turno de 30 s, tope por ejecución, archivo local y
   parada en seco, **comprobados sin red ni esperas** (cliente y `sleep` inyectados).
3. **Salida en el formato de #141**: un test de ida y vuelta contra `premarket_gap.load_bars` impide
   que los dos formatos se separen.
4. **CLI** (`python -m cfdtrader.data.sources.dukascopy`): backfill por tramos, retomable; en la salida
   declara sesiones, barras, ausencias, notas, peticiones de red y horas servidas del archivo.

Pendiente (y **no** se toca aquí): la **medición** del candidato con la maquinaria de #146/#149 y, si
la puerta pasa, la promoción a *feature* (`features/store.py` + `analysis/feature_frame.py`) con su
`design_lag` declarado (la barra existe a las 08:45 ⇒ `0`). El `--bars-dir` de #149 ya puede leer la
salida de este módulo **sin conversión**.

### 5.5 Lo que falta para arrancar el backfill

- **Ejecutar el backfill por tramos** (`--start`/`--end`) en varias sesiones: ~2 500 peticiones a dos
  por minuto ≈ 20 h de reloj, **una sola vez** (después, todo sale del archivo local).
- **El correo de consentimiento** a Dukascopy sigue mereciendo la pena: es lo que convertiría esto de
  una decisión asumida en algo **autorizado**, y su ToU lo contempla expresamente.

**Lo que no se hará, pase lo que pase:** bajar el turno por debajo de lo que aguanta el servicio,
concurrencia, versionar el dato crudo o redistribuirlo.

## Cómo se reproduce la sonda

**La sonda de Dukascopy no se reproduce a escala**: su ToU lo prohíbe sin consentimiento escrito
(§2.1.1). Lo que se hizo fue **una** comprobación acotada de viabilidad (unos pocos ficheros), y el
comando se deja aquí **solo para dejar constancia del formato** — **no** es la vía de adquisición:

```bash
# NO usar esto para descargar el histórico: ver §2.1.1 (ToU de Dukascopy).
curl -s -o /tmp/duka.bi5 'https://datafeed.dukascopy.com/datafeed/USA500IDXUSD/2024/05/03/08h_ticks.bi5'
uv run python -c "import lzma; d=lzma.decompress(open('/tmp/duka.bi5','rb').read(), format=lzma.FORMAT_ALONE); print(len(d)//20, 'ticks')"
```

El `.bi5` es LZMA (*alone*); cada tick son 20 bytes big-endian
(`uint32` ms de la hora, `uint32` ask, `uint32` bid, `float32` askvol, `float32` bidvol), con el
precio escalado por los decimales del instrumento. La sonda que lo decodifica está en
`.scratch/probe_dukascopy.py`.

## Referencias

`_docs/premarket_gap_2026-10-09.md` (#141) · `_docs/plan.md` §19.18 (#107) · §19.23 (#148) ·
`analysis/premarket_history.py` (#149) · `_docs/data_sources.md` · `analysis/feature_budget.py` (#146).

