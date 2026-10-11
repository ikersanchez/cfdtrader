# Revisión: ¿hay fuente gratuita del pre-mercado del ES, o se puede adaptar el modelo/decisión?

| Campo | Valor |
|---|---|
| **Fecha** | 2026-10-11 |
| **Origen** | seguimiento de #141 / #148 / #149 / #107 |
| **Estado** | Revisión medida **y corregida** (§0/§2.1: **sí** hay candidata gratuita). **No** adopta nada: `phase2_ready = false`, carril B bloqueado. |
| **Aval de honestidad** | esto **no** afirma *edge*. Un «no hay serie» medido vale más que una feature prometida. |

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

Lo que **queda por medir** (no concluido en esta nota):

- **Fecha de inicio exacta y continuidad**: la sonda se topó con un **límite de peticiones** (el
  mismo fichero pasó de `200` a `503`/`000` al insistir), así que el alcance fino quedó incompleto.
  Hay que repetirlo **despacio** (una petición cada varios segundos) y comprobar que una sesión
  completa tiene sus 24 ficheros.
- **Licencia / ToS** de Dukascopy y **coste de ingeniería** (descarga masiva: ~24 ficheros/día ×
  ~250 días × ~12 años ≈ 70 000 ficheros).
- **Coherencia de instrumento**: es el CFD de **otro** bróker, no el del usuario (`SPX500:CFD`), pero
  **sí** es una cotización de CFD del S&P en sesión de 24 h — más parecida al instrumento operado que
  el futuro `ES=F`.

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
- **⚠️ NO se puede afirmar** que no exista fuente gratuita. **Dukascopy (§2.1) está pendiente de
  medir** y es la vía que puede reabrir la promoción a *feature*: si su histórico tick alcanza los
  ~10 años, la tarea deja de ser «contratar» y pasa a ser **ingerir un tick gratuito**.

## 4. Recomendación (revisada a la luz de §2.1)

| Opción | Coste | Cuándo sirve | Riesgo |
|---|---|---|---|
| **2.1 Medir la cobertura de Dukascopy** (fecha de inicio + continuidad, despacio) | bajo | inmediato | ninguno; es una medición |
| **2.1b Ingerir Dukascopy** *si la cobertura llega a ~10 años* | medio | al cerrar la medición | ToS y coste de ingeniería (tick a barra) |
| **3.1 Persistir el snapshot (acumular)** | bajo | a partir de ~1 año (medir) / ~10 (entrenar) | ninguno; es solo acumular |
| **3.3 Medir basis y retorno nocturno** | bajo | inmediato | expectativa baja; no adopta nada |
| **3.2 Más datos declarados + avisos** | bajo | inmediato | un aviso que no decide (carril A) |
| **Contratar histórico point-in-time** | de pago | inmediato | decisión del propietario |

**Orden propuesto, corregido:** **(2.1)** cerrar la medición de Dukascopy — es lo único que puede
convertir un «no hay serie» en un **feature entrenable gratis**, y ya está demostrado que el
proveedor sirve datos reales de ≥10 años. Si la cobertura se confirma, la tarea deja de ser
«contratar» y pasa a ser **ingerir**. En paralelo, **(3.1)** acumular el snapshot propio (barato y
complementario) y **(3.3)** medir las dos candidatas históricas con la puerta de #146.

## Cómo se reproduce la sonda

```bash
uv run python .scratch/probe_free_sources.py            # la tabla de la seccion 1 (yfinance)

# Dukascopy: un fichero tick (respetar el limite de peticiones: una cada varios segundos)
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

