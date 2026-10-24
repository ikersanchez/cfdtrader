# Presupuesto formal de `reversion_v1` (`#160`, seguimiento de #159)

> **Fecha:** 2026-10-24 · **Tarea:** #160 · **Origen:** #159 (familia materializada, no adoptada) ·
> **Aval de honestidad:** esto **no** demuestra edge. La producción **no** cambia:
> `BASELINE_FEATURES` sigue siendo la tupla de **10**, `phase2_ready = false` y el carril B sigue
> bloqueado (`plan.md` §11.6 / §19.7).

## El hueco que cierra

La nota de #159 decía que la decisión de presupuesto podía medirse «con el grupo `reversion`
declarado en `analysis/feature_budget`», pero ese grupo **no existía**: el módulo solo aceptaba
`commodities` y `historical_market`. #160 lo declara y ejecuta la medición pre-registrada. Las seis
columnas ya estaban en la matriz de **63** columnas: no se recalcula ninguna feature.

## La regla, declarada antes de ver los resultados

> **Espacio.** `control_v1` (10 columnas) + **seis adiciones de una candidata**
> (`control + candidata`, 11 columnas) + **seis intercambios** (10 columnas, sale la columna de
> control con mayor `|r|` de Pearson con la candidata, sin reutilizar una columna que ya haya
> salido; regla de #146).
>
> **El ampliado no es admisible.** Control + las seis candidatas son **16** columnas y superan el
> límite declarado **10-15** (`plan.md` §9). No se prueba ningún conjunto de 16.
>
> **Puerta.** La decisión solo adopta un conjunto si el veredicto es `mejora` **y** el DSR sale
> `significant` **y** el PBO no está `detected` (`<= 0.2`). Una mejora descriptiva sin la
> maquinaria de sobreajuste **mantiene el control**.

## La medición (2026-10-24, almacén `data/`)

Ventana común de casos completos: **2.688 sesiones** (2016-01-07 → 2026-09-17), 1.400 positivas,
**0** sesiones excluidas y **0** nulos. Plan de #12: 10 pliegues, embargo total 45, purga 0,
`exclusions_are_no_op = True`. Sharpe OOS por sesión (calibrado, coste declarado de #11):

| Conjunto | Tipo | n | Operadas | Sharpe OOS | Brier | Log-loss |
| --- | --- | --- | --- | --- | --- | --- |
| `control_v1` | control | 10 | 416 | −0,034462 | 0,254651 | 0,704553 |
| `add_vix_ts_slope` | add | 11 | 415 | −0,033324 | 0,254515 | 0,704572 |
| `add_vix_contango` | add | 11 | 418 | −0,046909 | 0,254851 | 0,704678 |
| `add_vix9d_vix` | add | 11 | 421 | −0,041637 | 0,254714 | 0,704764 |
| `add_breadth` | add | 11 | 409 | −0,031260 | 0,254113 | 0,702918 |
| `add_streak_up` | add | 11 | 393 | −0,032805 | 0,253167 | 0,700976 |
| `add_ret_5_pct` | add | 11 | 420 | −0,031582 | 0,254297 | 0,703838 |
| `swap_vix_ts_slope` | swap | 10 | 408 | −0,043921 | 0,255015 | 0,705604 |
| `swap_vix_contango` | swap | 10 | 421 | −0,049863 | 0,254593 | 0,703079 |
| `swap_vix9d_vix` | swap | 10 | 409 | −0,038168 | 0,254372 | 0,704472 |
| `swap_breadth` | swap | 10 | 417 | −0,002017 | 0,253367 | 0,700812 |
| `swap_streak_up` | swap | 10 | 402 | **+0,013066** | 0,253765 | 0,702164 |
| `swap_ret_5_pct` | swap | 10 | 427 | −0,037680 | 0,253956 | 0,702487 |

Intercambios (columna de control que sale → candidata que entra): `vix_zscore`→`vix_ts_slope`,
`garch_forecast_z`→`vix_contango`, `dist_sma_20_z`→`vix9d_vix`, `europe_prev_1`→`breadth`,
`asia_overnight_1`→`streak_up`, `ust_10y_chg_5`→`ret_5_pct`.

## El veredicto y la decisión

- **Veredicto (descriptivo): `mejora`.** El mejor conjunto es `swap_streak_up`: diferencia media
  pareada frente al control `+0,000327` por sesión, con error estándar `0,000243`.
- **Puerta de significación: NO superada.** **DSR** `0,347837` (`not_significant`) con
  `n_trials = 13` (los 13 conjuntos probados) y **PBO** `0,543651` (`detected`, umbral `0,2`).
- **Decisión: `mantiene_control`.** La producción no se toca: `BASELINE_FEATURES` sigue en 10 y
  no hay cambio de constante ni regeneración de la cadena §2.

**La sonda de #159 no era este espacio.** `.scratch/measure_reversion_2026-10-16.py` llamaba a
`redundant_control_feature` **sin** excluir la columna ya intercambiada, de modo que
`swap_streak_up` repetía `europe_prev_1` (y `swap_breadth` también); en el espacio pre-registrado
`breadth` se queda `europe_prev_1` y `streak_up` saca `asia_overnight_1`. Por eso la sonda daba
`+0,0097`/sesión con PBO `0,198`, y el informe formal da `+0,0131` con PBO `0,544`. Las dos
mediciones coinciden en lo que importa: **no se adopta nada**. La sonda queda como exploración, no
como decisión; la decisión es este informe.

## Identidad registrada

- `matrix_sha256 = 887390ebd35d5cbe72f6fc88703224a986f73e93367ff51bf8de849ff62267f3` (matriz de 63 columnas; el grupo no la cambia)
- `feature_spec_sha256[reversion_v1] = 0d6d149db484588b8ec6a3995d6985abb45e0de2b2b5e3ee52bbef513f0ddc8a`
- `plan_sha256 = 7618406dc9dcf52f72f99719a34aa68a3ad2251db071cfdae4018db2d2273494`
- `adopted_features_version (= control) = sha256:34c79a97940b99e26cf0338b2c2b9a3e12b1cd1944c5916d276592c120d3d1da`
- `report_sha256 = sha256:7bd952a1180956307344bbb00086e9d40b6c069f4429d833d71f9a0c76a1b88d`

## Por qué no cambia producción

- `BASELINE_FEATURES` sigue siendo la tupla de **10** y `delivery.run_daily` no se toca.
- `phase2_ready = false`; §11.6 y §19.6/§19.7 sin tocar; carril B.
- El gancho es **opt-in**: `analysis.feature_budget --candidate-group reversion`.
- El informe vive en `data/derived/reports/feature_budget_reversion_2026-10-24.{json,md}`
  (gitignorado, regenerable); esta nota es el artefacto versionado.

## Cómo se reproduce

```bash
uv run python -m cfdtrader.analysis.feature_budget \
  --data-root data --reports-dir data/derived/reports \
  --candidate-group reversion --as-of 2026-10-24T22:00:00+00:00
```

`--as-of` es obligatorio: el módulo no lee el reloj. `tests/test_issue_160.py` blinda el espacio, la
decisión, el CLI y el nombre del informe; **no** fija los números vivos.

## Seguimientos

- No hay issue de adopción: la puerta no se supera y la decisión es `mantiene_control`.
- Las seis columnas siguen materializadas para que cualquier revisión futura pueda repetir la
  medición sin recalcular nada. Añadir una séptima candidata exige su propia medición de
  disponibilidad (regla de #159), no una línea más en la tabla.
