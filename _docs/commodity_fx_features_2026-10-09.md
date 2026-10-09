# Decisión: commodities y FX como features (`commodities_v1`, #143)

> **Fecha:** 2026-10-09 · **Tarea:** #143 · **Seguimiento:** #146 (presupuesto de #24 y re-medida
> de la Fase 2) · **Aval de honestidad:** esto no es una afirmación de *edge*. El crudo «pesa»
> económicamente (`_docs/plan.md` §7.1) y eso **no** demuestra que su feature mejore el modelo
> fuera de muestra: `phase2_ready = false` y el carril B sigue bloqueado (plan.md §11.6 / §19.7).
> Esta decisión sólo decide **qué candidatas entran**, no que sirvan.

## El hueco que cierra

`_docs/plan.md` §7.1 y §8.1 declaran relevantes el crudo y el dólar y listan `DX-Y.NYB`,
`EURUSD=X`, `BZ=F`, `CL=F`, `GC=F`. Las cuatro series **se ingieren** (`raw.market_daily`,
`config/data_sources.yaml`, `asset_class: commodity`) y **ninguna feature las leía**: un `grep` de
`oil|crude|wti|brent|gold|eurusd` sobre `src/cfdtrader/features/` sólo encontraba `DX-Y.NYB`.
`DX-Y.NYB` ya era feature desde #21 (`dxy_ret_1`); las otras cuatro no eran nada.

## La regla, declarada antes de medir

> **R.** Una serie se **rechaza** si el coeficiente de Pearson de su retorno logarítmico **de una
> sesión** contra el de una serie **ya admitida** (o ya feature del catálogo) es `|r| >= 0.85`
> sobre la ventana común, y la admitida tiene **al menos tanta historia**. En cualquier otro caso
> la serie **entra** como candidata.
>
> Conjunto de comparación y orden: primero las series que ya publican una feature (`DX-Y.NYB`,
> vía `dxy_ret_1`, #21) y después las admitidas por esta misma regla, en el orden declarado
> `CL=F`, `BZ=F`, `GC=F`, `EURUSD=X`.

Dos consecuencias que conviene declarar de antemano:

- El horizonte de la regla es **una sesión** porque es el del sistema: la etiqueta es una barrera
  triple `k·σ` (con `k = 1.0`) evaluada sobre **las barras de la propia sesión** de entrada, y el
  `sigma` es el pronóstico de volatilidad de **esa** sesión (`models/labels.py`). No hay ningún
  horizonte de cinco o veinte sesiones en el sistema que justifique medir a ese plazo.
- `0.85` es una valla **declarada**, no estimada: separa «el mismo factor económico con ruido
  distinto» de «dos series que se mueven juntas». Se elige antes de ver los números y no se ajusta
  después.

## Lo medido

Pearson sobre retornos logarítmicos diarios, sesión derivada de `as_of` en `America/New_York` (la
misma convención que `analysis.feature_frame`), sobre el almacén `data/` el 2026-10-09:

| Serie | Qué es | Sesiones | Ventana |
| --- | --- | --- | --- |
| `CL=F` | WTI (crudo) | 5 472 | 2005-01-04 → 2026-10-08 |
| `BZ=F` | Brent | 4 708 | 2007-07-31 → 2026-10-08 |
| `GC=F` | Oro | 5 173 | 2005-01-06 → 2026-10-08 |
| `EURUSD=X` | EUR/USD | 5 519 | 2005-01-04 → 2026-10-08 |
| `DX-Y.NYB` | Índice dólar | 5 481 | 2005-01-04 → 2026-10-08 |

| Par | `r` a 1 sesión | n | Veredicto de la regla `R` |
| --- | --- | --- | --- |
| `BZ=F` vs `CL=F` | **+0.8596** | 4 705 | **rebasa la valla** |
| `EURUSD=X` vs `DX-Y.NYB` | **-0.3219** | 5 331 | no rebasa |
| `EURUSD=X` vs `CL=F` | +0.0860 | 5 322 | no rebasa |
| `EURUSD=X` vs `GC=F` | +0.1384 | 5 027 | no rebasa |
| `GC=F` vs `CL=F` | +0.1319 | 5 169 | no rebasa |
| `GC=F` vs `DX-Y.NYB` | -0.3487 | 5 170 | no rebasa |
| `CL=F` vs `DX-Y.NYB` | -0.1576 | 5 470 | no rebasa |
| `BZ=F` vs `DX-Y.NYB` | -0.1446 | 4 706 | no rebasa |


## La premisa de la issue no se sostenía (y se declara)

La issue afirmaba que «el `EURUSD` es casi su inverso» del DXY y ofrecía eso como motivo para
dejarlo fuera. **Medido, es falso en el horizonte que importa**, y verdadero en otro:

| Horizonte | `r(EURUSD=X, DX-Y.NYB)` | n |
| --- | --- | --- |
| **1 sesión** | **-0.3219** | 5 331 |
| 5 sesiones | -0.7955 | 5 327 |
| 21 sesiones | -0.9198 | 5 311 |
| 63 sesiones | -0.9533 | 5 269 |
| 252 sesiones | -0.9607 | 5 084 |
| **Niveles** (log) | **-0.9720** | 5 332 |

No es un artefacto de alineación: un desfase de ±1 sesión no mueve el `-0.32` (`+1`: `-0.3212`;
`-1`: `+0.0376`). La lectura es que el **nivel** del par y el del índice dólar son casi el mismo
número con el signo cambiado (el EUR pesa 57,6 % en la cesta del DXY), pero **el ruido diario no lo
es**: a un día cada serie lleva su propia sacudida (los otros cinco componentes de la cesta, los
diferenciales de tipos), y esa sacudida es justo lo que transporta una feature de retorno de una
sesión.

**Consecuencia declarada:** `eurusd_ret_1` **entra** como candidata. Si #146 encontrara que el
conjunto con el que se mide pide el horizonte largo, los números de la tabla (`-0.80` a 5 sesiones,
`-0.92` a 21) son los que habría que volver a mirar, y quedan publicados aquí para no repetir la
medición.

**La misma pregunta al revés:** `dxy_ret_1` (nivel de contexto, #21) y `eurusd_ret_1` (retorno) no
son la misma columna con otro nombre; son medidas distintas con un `r` de `-0.32` a un día.

## Decisión, serie a serie

| Serie | Decisión | Motivo (medido) |
| --- | --- | --- |
| `CL=F` (WTI) | **ENTRA** | Es la serie del caso geopolítico de #142 (Irán, Hormuz, sanciones) y la de **más historia** de los dos crudos (5 472 sesiones desde 2005-01-04 frente a las 4 708 de `BZ=F` desde 2007-07-31). `r` con el DXY: `-0.16`. |
| `BZ=F` (Brent) | **SE RECHAZA** | `r = +0.8596` contra `CL=F`, que rebasa la valla (`>= 0.85`) y además tiene **más historia**. Brent y WTI son el mismo factor económico con dos cotizaciones; añadir las dos engordaría el conjunto sin una hipótesis propia. |
| `GC=F` (Oro) | **ENTRA** | Es la serie de refugio / aversión al riesgo. Su `r` con el WTI es `+0.13` y con el DXY `-0.35`: no es ninguna de las dos. |
| `EURUSD=X` | **ENTRA** | La premisa del rechazo no se sostiene (arriba). `r = -0.32` con el DXY a un día. |

`DX-Y.NYB` no entra en la familia: su retorno **ya es** `dxy_ret_1` desde #21, y duplicarlo en un
segundo catálogo sería el defecto que #72 resolvió, no una decisión.

## Qué se rechaza además, y por qué (declarado, no olvidado)

- **La relación crudo~XLE** (`corr(oil, XLE, 60)`), que la issue dejaba como «si se justifica».
  **No entra.** Es exactamente la correlación rodante contra un ETF del índice que `_docs/plan.md`
  §9 y §17 señalan como riesgo de sobreajuste, exigiría una segunda rejilla de alineación entre el
  calendario del futuro y el del ETF sectorial, y la señal que busca —transmisión del crudo al
  sector energético— **ya viaja** en `oil_ret_1` y en `sector_dispersion_1` (que incluye `XLE`).
  Si #146 quiere medirla, se mide como experimento, no como columna del catálogo.
- **`gold_ret_5`, `eurusd_ret_5` y `oil_ret_5_z`.** No entran: no tienen una hipótesis propia
  declarada y cada una cuesta una columna del presupuesto de #24. `oil_ret_5` **sí** entra porque
  su hipótesis sí está declarada (abajo).

## Las cinco columnas y su hipótesis

| Columna | Serie | Hipótesis declarada |
| --- | --- | --- |
| `oil_ret_1` | `CL=F` | Un shock geopolítico mueve el crudo **hoy**; es la variable directa del caso de #142. |
| `oil_ret_5` | `CL=F` | El shock **persiste** varios días (Hormuz, sanciones) y el S&P no reacciona a la primera vela: mide la magnitud acumulada. |
| `oil_ret_1_z` | `CL=F` | «Cuánto de raro» es el movimiento de hoy frente a **su propia** historia (ventana expandida), la misma forma que `vix_zscore` y `sector_dispersion_1_z`: un `+0.05` de crudo no es un shock si el crudo se mueve así todas las semanas. |
| `gold_ret_1` | `GC=F` | Refugio: el oro sube cuando el mercado busca cobijo. |
| `eurusd_ret_1` | `EURUSD=X` | El par es la pata del dólar con más ruido diario propio (`r = -0.32` con el DXY): aporta lo que `dxy_ret_1` no lleva. |

## Alineación *point-in-time* (declarada)

- Las tres series cierran **después** del cierre del S&P (el futuro del WTI y el del oro a las
  17:00 ET, `_docs/plan.md` §8.5; el par con el cierre de Nueva York), así que la fila `t` lee la
  **última sesión `< t`** de cada serie: `COMMODITIES_LAG_SESSIONS = 1`, el mismo rezago que usa
  `dxy_ret_1` en `context_v1`. Nunca la sesión `t`.
- El retorno se calcula en **el calendario de su propia serie** (`ln(C_i / C_{i-1})`): un festivo
  ajeno **no** produce un cero, produce el retorno de la última sesión con cierre, y si no hay
  historia la feature es `null`.
- Lo que no se puede computar (serie sin historia, cierre nulo o no positivo, ventana de cinco
  sesiones que no cabe en la historia de la serie) se publica **`null`**, nunca `0` ni `NaN`.
- `oil_ret_1_z` es la única columna que mira la historia `<= t`: es una normalización de ventana
  **expandida** (mediana y MAD de la historia, jamás de la muestra completa, `plan.md` §9), es
  decir la medida de rareza de hoy, no un dato de hoy.

## Presupuesto de features (#24), declarado

El conjunto del modelo **no cambia en #143**: `models.baseline.BASELINE_FEATURES` sigue siendo la
misma tupla de **10** y `delivery.run_daily` sigue puntuando exactamente esas 10. Lo que se entrega
son **5 candidatas disponibles** en el catálogo (57 columnas en total) para la maquinaria de
selección que ya existe (`analysis.feature_frame`, #82). Decidir si el conjunto del modelo **se
amplía** (dentro del límite 10-15 de #24) o si se **intercambia** —y con qué regla— es **#146**,
junto con la re-medida de la Fase 2 y su veredicto.

## Cómo se reproduce

```bash
uv run python - <<'PY'
import math, polars as pl
from cfdtrader.data.store import Store

def ser(sid):
    f = Store("data").sql(
        f"SELECT as_of, close FROM raw.market_daily WHERE series_id = '{sid}' ORDER BY as_of"
    )
    f = f.with_columns(pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("s"))
    s, c = f["s"].to_list(), f["close"].to_list()
    return {s[i]: math.log(c[i] / c[i - 1]) for i in range(1, len(c)) if c[i] and c[i-1] and c[i] > 0 and c[i-1] > 0}

def pearson(a, b):
    k = sorted(set(a) & set(b)); x = [a[i] for i in k]; y = [b[i] for i in k]
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    num = sum((p-mx)*(q-my) for p, q in zip(x, y))
    dx = math.sqrt(sum((p-mx)**2 for p in x)); dy = math.sqrt(sum((q-my)**2 for q in y))
    return num/(dx*dy), n

r = {s: ser(s) for s in ("CL=F", "BZ=F", "GC=F", "EURUSD=X", "DX-Y.NYB")}
for a, b in (("BZ=F","CL=F"), ("EURUSD=X","DX-Y.NYB"), ("GC=F","CL=F"), ("CL=F","DX-Y.NYB")):
    print(a, b, pearson(r[a], r[b]))
PY
```

La columna de horizontes de la tabla del EUR/USD se obtiene sustituyendo el retorno de una sesión
por `ln(C_i / C_{i-w})` con `w` en `{5, 21, 63, 252}`, y la fila de **niveles** por `ln(C)` en la
ventana común.
