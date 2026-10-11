# Decisión: auditoría de disponibilidad de las 57 columnas y medición de promoción

| Campo | Valor |
|---|---|
| **Fecha** | 2026-10-11 |
| **Origen** | «¿todas las variables se han evaluado con el dato más fresco?» · seguimiento de #147/#150/#153 |
| **Estado** | Implementado (auditoría + tests + regeneración). Medición **exploratoria**, no adopta nada. |
| **Aval de honestidad** | esto **no** afirma *edge*. `phase2_ready = false`, carril B bloqueado (`plan.md` §11.6 / §19.7). Adoptar un conjunto exige `mejora` **y** DSR `significant` **y** PBO no `detected`. |

## El hueco que cierra

#147 introdujo `design_lag` por columna y #150/#153 auditaron **solo las 10 de
`BASELINE_FEATURES`** (las que el modelo puntúa). Las otras **47** columnas del catálogo llevaban el
valor conservador por defecto (`1`) que puso #147, **sin auditar**: el test
`test_issue_147::test_c2_every_column_declares_its_availability` solo exigía que el valor fuera `0` o
`1`, no que fuera el correcto.

## La auditoría: tres grupos declarados (las 57 columnas)

La partición se fija por test (`tests/test_issue_153.py`) leyendo la **fórmula** de cada familia, no
la prosa del `req_as_of` (que en `atr_norm` de volatilidad decía `t` y su fórmula es `shift(1)`, es
decir `t-1`).

| Grupo | N | Regla |
|---|---|---|
| **Legibles en el instantáneo** (`design_lag = 0`) | 18 | su fila `t` ya está publicada o es computable a las 08:45 ET |
| **Forzadas por el reloj del ancla** | 13 | su fila `t` necesita la barra de `t` (cierre 16:00 ET): adelantarla sería *look-ahead* |
| **Forzadas por el corte macro** | 13 | la macro se define al `as_of` de la fila (16:00 ET) |
| **Conservadoras, sin lector pendiente** | 13 | su input cierra antes del instantáneo, pero su lector es una tarea aparte |

### Movidas a `0` en esta entrega (10 columnas, con lector pendiente, sin duplicar fórmulas)

- **volatilidad**: `har_lag1`, `har_lag4`, `har_lag17` (`parkinson_rv` desplazada), `vix_level`
  (`vix_close.shift(1)`), `vix_percentile`, `atr_norm` (`true_range` de `t-20…t-1` / `C_{t-1}`).
- **técnica**: `atr_norm_z` (normalización expandida de `atr_norm`).
- **régimen**: `garch_forecast` (`ret_log` hasta `T = t-1`), `rv_percentile` (la `rv` de `t-1` frente
  a la historia `< t-1`), `day_of_week` (calendario).

El lector pendiente es el de #153 (`feature_frame._pending_family_values`), ampliado con la rama
**técnica**; las columnas se añadieron a `PENDING_FAMILY_COLUMNS`. La unión de
`context.PENDING_COLUMNS` y `PENDING_FAMILY_COLUMNS` sigue siendo **exactamente** el conjunto con
`design_lag = 0` (invariante de #147, cruzado por dos tests).

### Qué queda fuera, y por qué

- **`sessions_to_opex`**: es de calendario pero arrastra nulos de cola (#79); su lector espera.
- **`corr_*_60`, `beta_vix_60`, `sector_count`, `sector_dispersion_1_z`** y **`commodities_v1`**
  (`oil_ret_*`, `gold_ret_1`, `eurusd_ret_1`): su input cierra antes del instantáneo —igual que
  `dxy_ret_1`, que ya es `0`—, pero el lector pendiente aún no tiene su fórmula. Conservan el `1`
  conservador, **declarado**, hasta su propia auditoría.

## Consecuencia sobre el modelo: **ninguna**

El `run_sha256` **no cambia**. La `window` de #16 publica `design_lag_by_feature` **solo de las
columnas seleccionadas**, y la producción selecciona las 10 de `BASELINE_FEATURES`, cuyas
disponibilidades no se han tocado. Medido en la regeneración en orden: `baseline_report` volvió a
emitir `run_sha256 = 7bac6a12…` y `c8214e97…` (los mismos) y `model_comparison` → **4/4 `selected`**.
El modelo servido sigue siendo válido y la guardia de #152
(`run_daily._require_design_alignment`) lo confirma: la alineación del `model.json` coincide con la
del catálogo.

Es decir: la auditoría deja el catálogo **coherente** y **prepara** una promoción futura (sin
*train/serve skew*), pero **no** cambia la decisión de hoy. Las 10 columnas auditadas **no** están en
el modelo.

## Medición de promoción (exploratoria, no adopta nada)

Con la maquinaria de `analysis.feature_budget` (#146), **importada** (protocolo de la Fase 2: purga y
embargo de #12, modelo de #24/#25 con su calibrador, motor de #13 con el coste de #11), sobre las 500
sesiones de test del plan. El presupuesto declarado de #24 es 10-15, así que las 10 candidatas se
midieron en **dos grupos de cinco** (control + 5 = 15).

| Grupo | Mejor candidato | Sharpe control | Sharpe mejor | Veredicto | Decisión |
|---|---|---|---|---|---|
| 1: `har_lag1/4/17`, `vix_level`, `vix_percentile` | `swap_vix_percentile` | −0.035270 | −0.003009 | `mejora` | **`mantiene_control`** |
| 2: `atr_norm`, `atr_norm_z`, `garch_forecast`, `rv_percentile`, `day_of_week` | `swap_garch_forecast` | −0.035270 | −0.031130 | `empata` | **`mantiene_control`** |

- El grupo 1 da `mejora` **descriptiva** (diferencia media pareada `+0.000261`, error estándar
  `0.000219`), pero la **puerta de significación no se supera** (DSR/PBO): la decisión es
  `mantiene_control`. Es el mismo patrón que #146 con las candidatas de commodities/FX.
- El grupo 2 **no** mejora (todos los intercambios con un Sharpe más negativo que el control).
- El conjunto **ampliado** (15 columnas) no ayuda en ninguno de los dos grupos.

**Conclusión.** Ninguna de las columnas auditadas mejora el modelo de forma que supere la puerta de
honestidad. `BASELINE_FEATURES` sigue siendo la misma tupla de **10** y producción no cambia. Lo que
sí aportaría información nueva (el pre-mercado del ES) sigue bloqueado por datos (#149/#107: no hay
serie histórica gratuita).

## Cómo se reproduce

```bash
# La auditoria: la particion de las 57 columnas y el motivo de cada movida
uv run pytest tests/test_issue_153.py -q

# La regeneracion en orden (sin red) y la medicion exploratoria
bash .scratch/regen_availability.sh
uv run python .scratch/explore_budget.py
```

## Archivos

`src/cfdtrader/features/store.py` (catálogo) · `src/cfdtrader/analysis/feature_frame.py` (lector
pendiente) · `tests/test_issue_153.py` (auditoría y re-baseline de #153) · `tests/test_issue_150.py`
(re-baseline del cruce de familias).

