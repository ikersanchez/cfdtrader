# Decisión: presupuesto de features de #24 y re-medida de la Fase 2 (`#146`)

> **Fecha:** 2026-10-10 · **Tarea:** #146 · **Origen:** #143 (candidatas de commodities/FX) ·
> **Aval de honestidad:** el crudo «pesa» económicamente (`_docs/plan.md` §7.1) y eso **no**
> demuestra que su feature mejore el modelo fuera de muestra. Esta re-medida **no** cambia
> producción: `BASELINE_FEATURES` sigue siendo la misma tupla de **10**, `phase2_ready = false` y el
> carril B sigue bloqueado (`plan.md` §11.6 / §19.7).

## El hueco que cierra

#143 decidió **qué** series entran como candidatas y materializó la familia `commodities_v1`
(`oil_ret_1`, `oil_ret_5`, `oil_ret_1_z`, `gold_ret_1`, `eurusd_ret_1`). Lo que no cabía allí —y
vive aquí— es la parte de **modelado**: si esas candidatas se quedan en el conjunto del modelo y si
el conjunto **mejora, empata o empeora** la Fase 2. El contrato de #24 son **10** columnas y el
límite declarado **10-15** (`plan.md` §9): meter candidatas exige decidir si el conjunto se **amplía**
o se **intercambia**, y esa decisión no se puede tomar sin medirla.

## La regla, declarada antes de ver los resultados

> **R (presupuesto).** Se prueba el **control** (las 10 de `BASELINE_FEATURES`) frente al
> **ampliado** (control + las 5 candidatas = 15, dentro del límite 10-15) y frente a un
> **intercambio por candidata** (mismo tamaño 10). La decisión solo puede ser `amplia`,
> `intercambia` o `mantiene_control`.
>
> **Intercambio.** Por cada candidata entra **una** columna del control y sale **la más redundante
> con ella**: la de mayor `|r|` de Pearson sobre el frame de features, con los empates resueltos por
> el orden declarado del control. La redundancia se mide **sobre las features**, antes de ver el
> modelo.
>
> **Veredicto.** `mejora` si la diferencia media pareada de retornos netos por sesión frente al
> control supera `1` error estándar (`media(d) > 1·se`), `empeora` si queda por debajo de `−1·se` y
> `empata` en otro caso. `not_evaluable` cuando el control o ningún candidato opera, o cuando el
> DSR/PBO no es estimable.
>
> **Puerta de significación (decisión).** La decisión solo **adopta** un conjunto si el veredicto es
> `mejora` **y** el DSR sale `significant` **y** el PBO no está `detected`. Una mejora descriptiva
> sin el respaldo de la maquinaria de sobreajuste **no** mueve producción.

**Ventana común de casos completos (declarada).** El modelo no acepta `NaN` y las features **no se
imputan** (`plan.md` §9). El WTI negativo del 2020-04 hace que `oil_ret_*` publique `null` (#143), así
que la re-medida usa la ventana **común de casos completos** de la unión control+candidatas: **2 685**
sesiones de diseño (2016-01-07 → 2026-09-17, 1 398 positivas), las mismas para todos los conjuntos,
de modo que las series pareadas son comparables. Se descartan **3** sesiones (2020-04-22, -23 y -29).
El protocolo es el de la Fase 2, **importado y no reimplementado**: el plan de #12 (purga y embargo),
el modelo de #24/#25 con su calibrador y el motor de #13 con el coste declarado de #11.

## La medición (2026-10-10, almacén `data/`)

Sharpe OOS por sesión (coste declarado), calibrado, sobre las 500 sesiones de test del plan:

| Conjunto | Tipo | n | Operadas | Sharpe OOS | Brier | Log-loss |
| --- | --- | --- | --- | --- | --- | --- |
| `control_v1` | control | 10 | 401 | +0.006631 | 0.255266 | 0.703829 |
| `expanded_v1` | ampliado | 15 | 399 | −0.006998 | 0.255343 | 0.833051 |
| `swap_oil_ret_1` | intercambio | 10 | 369 | **+0.044287** | 0.252517 | 0.698398 |
| `swap_oil_ret_5` | intercambio | 10 | 389 | +0.008616 | 0.256524 | 0.835643 |
| `swap_oil_ret_1_z` | intercambio | 10 | 404 | −0.005026 | 0.253586 | 0.700231 |
| `swap_gold_ret_1` | intercambio | 10 | 409 | −0.007948 | 0.257714 | 0.837576 |
| `swap_eurusd_ret_1` | intercambio | 10 | 415 | −0.015405 | 0.256732 | 0.771605 |

Las columnas que salen en cada intercambio (regla del intercambio, medida **antes** del modelo):
`europe_prev_1` (por `oil_ret_1`), `dist_sma_20_z` (por `oil_ret_5`), `dxy_ret_1` (por `oil_ret_1_z`),
`ust_10y_chg_5` (por `gold_ret_1`) y `asia_overnight_1` (por `eurusd_ret_1`).

## El veredicto y la decisión

- **Veredicto (descriptivo): `mejora`.** El mejor candidato es `swap_oil_ret_1`; la diferencia media
  pareada frente al control es `+0.000291` por sesión, con error estándar `+0.000250`
  (`media(d) > 1·se`). El punto sale a favor.
- **Puerta de significación: NO superada.** El **DSR** del conjunto seleccionado es
  `not_significant` (`dsr = 0.6569`, `n_trials = 7`, umbral 0.95) y el **PBO** por CSCV es
  `detected` (`pbo = 0.3730 > 0.20`, `blocks = 10`). La aparente ganancia **no** se distingue del
  sobreajuste.
- **Decisión: `mantiene_control`.** La mejora descriptiva no tiene el respaldo de la maquinaria de
  sobreajuste, así que la producción **no** cambia: `BASELINE_FEATURES` sigue siendo la tupla de 10.
  El ampliado **empeora** el Sharpe (`−0.006998`), de modo que tampoco se amplía a 15.

**Valla de honestidad (repetida a propósito).** `mejora` en el punto + puerta de significación no
superada ⇒ `mantiene_control`. Escribir «el crudo mejora el modelo» con estos números sería afirmar
sin medir: el DSR y el PBO son exactamente la maquinaria que lo desmiente.

## Identidad registrada

`feature_budget` publica la identidad del diseño elegido (la lista de features + la matriz + el plan):

- `matrix_sha256 = 44bc5c53147bc13c1586026f0b4f45b05bb66b78d5f1bb8f3da193090f06be39`
- `plan_sha256  = dfd25702c2cd179c8fc618440e229291a7c802e15f2e51764aed779f8b31bf33`
- `feature_code_version = 2`; `feature_spec_sha256` de las seis familias
  (`volatility_v1`, `technical_v1`, `context_v1`, `macro_v1`, `regime_v1`, `commodities_v1`).
- `adopted_features_version (= control) = sha256:229d24e39a58f7525201d16762f4ef7c89b837746c901579796276858a36a1e1`
- `report_sha256 = sha256:db455052c29d90970fbd5806960fabf63497946fa2ec7aa6aa2905d549fd77c2`

La matriz completa (57 columnas) y su spec **no** cambian al elegir un subconjunto: lo que se
registra es la **lista elegida** junto a las dos identidades que no dependen de ella. Con la decisión
`mantiene_control` no hay cambio de constante, así que **no** se regenera la cadena de Fase 2
(`_docs/process.md` regla 2): la regeneración en orden se dispara cuando la matriz o la constante
cambian, y aquí no.

## Por qué no cambia producción

- `BASELINE_FEATURES` sigue siendo la misma tupla de **10**; `delivery.run_daily` no se toca.
- `phase2_ready = false`; §11.6 y §19.6/§19.7 sin tocar; carril B.
- El gancho a la medición es **opt-in**: `analysis.feature_frame.build_feature_frame(features=...)`
  permite medir cualquier subconjunto del catálogo **sin** cambiar la constante de producción.

## Cómo se reproduce

```bash
uv run python -m cfdtrader.analysis.feature_budget \
    --data-root data --reports-dir data/derived/reports \
    --as-of 2026-10-10T22:00:00+00:00
```

Sale `feature_budget_2026-10-10.{json,md}` en `data/derived/reports/` (gitignorado). `--as-of` es
obligatorio: el módulo no lee el reloj. Los tests de `tests/test_issue_146.py` blindan lo que el
**código** declara (la regla, el espacio, la puerta y el vocabulario), no los números vivos.

## Seguimientos

- **#146** cierra con `mantiene_control`: no hay issue de cambio de constante que abrir, porque no se
  adopta ningún conjunto.
- El WTI negativo del 2020-04 deja `null` en `oil_ret_*` y reduce la ventana en 3 sesiones; si en el
  futuro se quiere una ventana sin ese hueco, es una decisión de datos, no de este presupuesto.
