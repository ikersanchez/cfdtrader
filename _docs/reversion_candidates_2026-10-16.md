# Candidatas de sobre-extension y reversion (`reversion_v1`, tarea #159)

**Estado: candidatas materializadas, no adoptadas.** `BASELINE_FEATURES` sigue siendo la tupla de
**10** de #24 y la decision de presupuesto (`amplia` / `intercambia` / `mantiene_control`) es la
issue de seguimiento con `analysis/feature_budget`, exactamente como #146 siguio a #143. Carril B;
`phase2_ready = false`; `§11.6` sin tocar.

## Por que

El **2026-10-16** el modelo servido (`lightgbm_gbdt_v1`, `runs/b75cbaa6…`) dio `p(up) = 0,5959`
→ LONG, y la descomposicion SHAP del margen (`+0,3885` log-odds) mostro que **ninguna** de las 10
columnas del control codifica "sobre-extension / optimismo excesivo": el unico empuje bajista real
fue `sector_dispersion_1` (−0,060, percentil 87) y `asia_overnight_1` (+0,163) y `europe_prev_1`
(+0,042) leyeron el optimismo de otras plazas *como alcista*. Esta familia **no** promete arreglar
eso: existe para poder medirlo.

## Insumos, cobertura y visibilidad (medido)

Snapshot de decision: `Store.read_pit(..., at = 2026-10-16T13:00:00+00:00)` (08:45-09:00 ET).

| insumo | procedencia | cobertura | sesiones en la ventana de diseno | visible en el snapshot |
|---|---|---|---|---|
| `^VIX` | `config/data_sources.yaml` | 2005-01-03 → 2026-10-15 | 2.689 | si |
| `^VIX9D` | idem, `min_start: 2011-01-01` (#58) | 2011-01-03 → 2026-10-15 | 2.689 | si |
| `^VIX3M` | idem, `min_start: 2006-07-01` (#58) | 2006-07-17 → 2026-10-15 | 2.689 | si |
| 11 SPDR (`XL*`) | dataset `sectors` | 2005 / 2015 / 2018 → 2026-10-15 | 2.689 | si |

`fetched_at` de las tres series del VIX: 2026-10-16 11:08 UTC ≤ 13:00 UTC (el snapshot). Ventana
comun de casos completos de las seis juntas: **2.689 sesiones**, del mismo orden que las 2.685 que
uso el presupuesto de #146.

## Formulas

Con `t` la sesion de la fila, `s(t)` la **ultima sesion publicada** de cada serie (`< t`, el rezago
declarado de la familia) y `C` su cierre:

| candidata | formula | `required_as_of` | `design_lag` |
|---|---|---|---|
| `vix_ts_slope` | `VIX3M(s)/VIX9D(s) - 1` | cierre de la sesion t−1 | 1 |
| `vix_contango` | `VIX(s)/VIX3M(s) - 1` | cierre de la sesion t−1 | 1 |
| `vix9d_vix` | `VIX9D(s)/VIX(s) - 1` | cierre de la sesion t−1 | 1 |
| `breadth` | `#{sectores con retorno > 0 en s} / #{sectores con dato en s}` | cierre de la sesion t−1 | 1 |
| `streak_up` | sesiones consecutivas con `C_u > C_{u-1}` al cierre de `s` | cierre de la sesion t−1 | 1 |
| `ret_5_pct` | percentil expandido (min 250) de `ln(C_v/C_{v-5})` | cierre de la sesion t | 1 |

El `design_lag` es **1** en las seis, y en cinco es la opcion **conservadora declarada**: la fila
`t` de `vix_ts_slope`, `vix_contango`, `vix9d_vix`, `breadth` y `streak_up` solo usa sesiones
`<= t-1`, asi que por la regla de #153 podria declarar `0` (su fila `t` ya es computable en el
instantaneo); pasa a `0` el dia en que exista su lector pendiente, como en commodities (#143).
`ret_5_pct` **no** tiene eleccion: su base es `ret_5`, que lee el cierre del ancla de `t` — por eso
declara el mismo `required_as_of` y el mismo rezago que `ret_5` en `technical_v1`.

Valores publicados por el modulo en la fila de la sesion **2026-10-16** (es decir, leidos del
cierre del 2026-10-15):

| candidata | valor | percentil en la ventana |
|---|---|---|
| `vix_ts_slope` | +0,323550 | 65,7 |
| `vix_contango` | −0,109789 | 56,7 |
| `vix9d_vix` | −0,151276 | 19,7 |
| `breadth` | 0,727273 | 73,8 |
| `streak_up` | 1 | 71,2 |
| `ret_5_pct` | `null` en la fila pendiente; el valor que entra en el diseño es el de **2026-10-15**: 0,347215 | 34,7 |

Dos precisiones honestas sobre esa ultima fila: (1) `ret_5_pct` de la propia sesion `t` es `null`
(su base necesita el cierre de `t`, que no existe a las 08:45 ET) y el diseno lee la fila de `t-1`
por su `design_lag = 1`; (2) el percentil que publica la familia es de ventana **expandida sobre
toda la historia disponible** (5.481 sesiones desde 2005), no dentro de la ventana de diseno: la
sonda de la issue calculo 32,9 recortando a la ventana y el modulo publica 34,7 sobre la historia
completa. La definicion de la familia es la segunda, y `vix_percentile` de #7 usa la misma.

## La redundancia algebraica, declarada

Los tres vertices cumplen

```
(1 + vix_contango) * (1 + vix9d_vix) * (1 + vix_ts_slope) = 1
```

de modo que **dos cualesquiera determinan la tercera**. El catalogo publica las tres —son tres
candidatas distintas de la tabla de #159, con percentiles muy distintos (65,7 / 56,7 / 19,7)— y la
**regla de intercambio** del presupuesto (#146) es quien decide el solape con
`vix_zscore`/`vix_percentile`. Un test de la familia comprueba el producto a 1e-12 en cada fila: no
es un comentario, es una identidad medida.

Solapes con el catalogo vigente, tambien declarados:

- `breadth` frente a `sector_dispersion_1` (#21): **misma fuente** (los once SPDR), formula
  distinta —proporcion de signos frente a desviacion tipica de los retornos—. El **denominador**
  de `breadth` son los sectores **con dato** en `t-1`, el mismo numero que publica `sector_count`
  en `context_v1`; un test lo cruza contra esa columna en la misma matriz.
- `ret_5_pct` frente a `ret_5` (#20): la candidata es el **percentil** de un retorno que ya existe;
  un test comprueba que la base es exactamente la columna de `technical_v1`.
- ninguna columna del catalogo usaba `^VIX9D` ni `^VIX3M` antes de esta tarea: se ingieren desde
  #58 y solo los leia el estudio del forecast de volatilidad.

## Historico y coste en muestra

La matriz de features tiene 5.481 sesiones (2005-01-03 → 2026-10-15), pero la muestra del modelo
es la del **diario**: 2016-01-07 → 2026-09-17 (**2.688** sesiones), y ese corte lo ponen las
etiquetas (`derived.labels`, `sigma_carrier = walk_forward`), no las features. Consecuencia medida:
**las seis candidatas cubren las 2.689 sesiones de la ventana comun sin perder ninguna**, porque
`^VIX9D` (2011-01-03) tiene cinco años de margen sobre el arranque de la ventana. La unica
definicion que podia recortarla era el denominador de `breadth`:

- sectores **presentes** + `sector_count` como companero (precedente de `context_v1`, medido: 618
  sesiones con 10 sectores y 2.071 con 11, `sector_dispersion_1` con 0 nulos) → **coste cero**;
- los **11** exigidos (`XLC` empieza el 2018-06-20) → 2.072 sesiones, **−617 sesiones (−23 %)**.

La familia adopta la primera: es la que no pierde muestra y la que ya estaba en el catalogo.

Limite declarado: si algun dia la muestra se extendiera hacia atras de 2011, `vix_ts_slope` y
`vix9d_vix` serian `null` antes de esa fecha (su vertice no existe) y **no** se rellenan con otro
vertice. Hoy no ocurre: la muestra empieza cinco años despues.

## Lo que NO se admite, y por que

| fuente | motivo |
|---|---|
| put/call ratio, AAII, NAAIM, posicionamiento CFTC | no hay serie en el almacen ni historico *point-in-time* medido |
| spreads de credito HY, `^SKEW`, `^VVIX` | idem: primero se mide la fuente y su historico (patron #142/#149) |
| primas del ES fuera de la sesion US | es #149/#155 (serie *point-in-time*) y #157 (paridad *shadow train/serve*) |
| el espejo corto (#78/#158) | no entra en esta familia: la direccion corta tiene su propia identidad |

El cuerpo de candidatas queda **cerrado en seis**: anadir una septima exige su propia medicion de
disponibilidad, no una linea mas en esta tabla.

## Como se materializo (aditivo, sin mover el control)

- `src/cfdtrader/features/reversion.py` (familia cerrada, pura, sin reloj ni red) y su registro en
  `src/cfdtrader/features/store.py` como **septima** familia, con su `feature_spec` propia
  (`reversion_v1`) y su `source` (`cfdtrader.features.reversion`).
- Las seis columnas van **al final** de `ALL_FEATURE_COLUMNS` (57 → 63): ningun nombre previo
  cambia de sitio y los **digests de las seis familias anteriores no se mueven** (comprobado en la
  regeneracion del golden: 0 digests distintos y 0 recuentos de no nulos distintos en las 57
  columnas previas). `FEATURE_CODE_VERSION` **no** sube, por la misma regla declarada de #143: una
  familia nueva que solo **anade** columnas no cambia el resultado de ninguna existente.
- `derived.features_daily` sube su `layout_version` 2 → 3 (§12.10) y el golden de la matriz se
  re-congelo con el par (version, digests) nuevo.
- El control **no** se mueve: `BASELINE_FEATURES` sigue siendo la tupla de 10, su
  `DESIGN_LAG_BY_FEATURE` no cambia y la matriz de diseno sigue con sus 2.688 filas y sus 10
  columnas (test dedicado).

## Medición de poder predictivo (sonda del 2026-10-16)

Sonda de solo lectura `.scratch/measure_reversion_2026-10-16.py`: **no** es un artefacto publicado y
**no** decide nada (la decision formal se tomo despues en #160). Aplica el
protocolo de #146 importado —plan de #12, modelo de #24/#25, motor de #13, ventana comun de casos
completos, veredicto pareado y DSR/PBO de `analysis.feature_budget`— candidata a candidata, porque
`measure` estaba cableado al grupo de commodities: `control + candidata` (11 columnas, dentro del
presupuesto 10-15) y el **intercambio** declarado (`control + candidata − la columna de control con
mayor |r|`).

**Cautela de la sonda (declarada a posteriori).** Llamaba a `redundant_control_feature` **sin**
excluir una columna de control que ya hubiera salido, asi que sus intercambios podian repetir la
misma columna (`swap_breadth` y `swap_streak_up` sacaban las dos `europe_prev_1`; `vix_ts_slope` y
`vix_contango` podian repetir `vix_zscore`). No es el espacio pre-registrado: el informe formal de
#160, con la exclusion, es el que decide.

Ventana comun: **2.688 sesiones**, 10 pliegues, 500 sesiones de test por conjunto. Control:
`sharpe = −0,0345`, 416 operaciones.

| conjunto | columnas | ops | sharpe | veredicto pareado vs control |
|---|---|---|---|---|
| `control_v1` | 10 | 416 | −0,0345 | — |
| `add_vix_ts_slope` | 11 | 415 | −0,0333 | `empata` (+7,5e−06) |
| `add_vix_contango` | 11 | 418 | −0,0469 | `empeora` (−8,1e−05) |
| `add_vix9d_vix` | 11 | 421 | −0,0416 | `empeora` (−5,0e−05) |
| `add_breadth` | 11 | 409 | −0,0313 | `empata` |
| `add_streak_up` | 11 | 393 | −0,0328 | `empata` |
| `add_ret_5_pct` | 11 | 420 | −0,0316 | `empata` |
| `swap_vix_ts_slope` (−`vix_zscore`) | 10 | 408 | −0,0439 | `empata` |
| `swap_vix_contango` (−`vix_zscore`) | 10 | 417 | −0,0488 | `empeora` |
| `swap_vix9d_vix` (−`dist_sma_20_z`) | 10 | 409 | −0,0382 | `empata` |
| `swap_breadth` (−`europe_prev_1`) | 10 | 417 | −0,0020 | `mejora` (+2,08e−04, se 2,05e−04) |
| `swap_streak_up` (−`europe_prev_1`) | 10 | 428 | **+0,0097** | `mejora` (+3,00e−04, se 2,08e−04) |
| `swap_ret_5_pct` (−`dist_sma_20_z`) | 10 | 414 | −0,0387 | `empata` |

**Puerta de significacion: NO superada.** DSR `0,331297` (`not_significant`) con 13 ensayos
intentados, PBO `0,198413` (`not_detected`, a un pelo del umbral `0,2`), mejor conjunto
`swap_streak_up` y decision `mantiene_control -> control_v1`. Es el mismo patron que #146: una
mejora descriptiva de 2-3 puntos basicos por sesion que **no** sobrevive al deflactado por el numero
de ensayos, con todos los *sharpe* dentro de ±0,05 (ruido).

Tres cautelas que la tabla no dice por si sola:

- los dos `mejora` son **intercambios**, no adiciones: cambian dos cosas a la vez —entra la
  candidata y sale `europe_prev_1`, que es la columna de control con mayor `|r|` con ella—, asi que
  la mejora esta confundida con quitar esa columna (la que hoy empujo +0,042 en el SHAP);
- **ninguna adicion** mejora: `vix_contango` y `vix9d_vix` empeoran y las otras cuatro empatan;
- la ventana mide, no fija: 500 sesiones de test dan errores tipicos de 3e−05 a 2e−04 por sesion,
  del mismo orden que las diferencias medidas.

Conclusion de la sonda: **no hay poder predictivo demostrado**, asi que la produccion se queda como
esta (`BASELINE_FEATURES` intacto) y no se abre la adopcion. Las columnas quedaron materializadas
para el presupuesto formal, y ese presupuesto **ya se ejecuto** en #160 con el grupo `reversion`
declarado en `analysis/feature_budget`: ver
[`feature_budget_reversion_2026-10-24.md`](feature_budget_reversion_2026-10-24.md) y
`data/derived/reports/feature_budget_reversion_2026-10-24.{json,md}`. Con el espacio pre-registrado
(seis adiciones de 11 y seis intercambios de 10; el ampliado de 16 **no** cabe en 10-15) el
resultado es `mejora` descriptiva (`swap_streak_up`, +3,27e−04/sesion, se 2,43e−04), DSR `0,348`
(`not_significant`) y PBO `0,544` (`detected`) → `mantiene_control`. La sonda y el informe formal
coinciden en la decision (no se adopta nada); difieren en el detalle de la exclusion de columnas,
que en el espacio formal saca `asia_overnight_1` por `streak_up` y sube el PBO.

## La valla

Estas seis columnas **no** estan en produccion: `BASELINE_FEATURES` intacto, `phase2_ready = false`,
carril B bloqueado y `§11.6` sin tocar. La medicion con `analysis/feature_budget` (patron #143 →
#146) ya se hizo en #160: control + seis adiciones + seis intercambios sobre los mismos pliegues de
#12, con la puerta de significacion (DSR y PBO) que decidio #146, y salio `mantiene_control`.
Adoptar exigiria reentrenar y regenerar la cadena §2 completa, y en ese caso habria que re-declarar
la referencia de la puerta del *paper* en vez de recalcularla: cambiarla a mitad de observacion es
mover la porteria.
