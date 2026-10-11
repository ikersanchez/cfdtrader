# Revisión: ¿hay fuente gratuita del pre-mercado del ES, o se puede adaptar el modelo/decisión?

| Campo | Valor |
|---|---|
| **Fecha** | 2026-10-11 |
| **Origen** | seguimiento de #141 / #148 / #149 / #107 |
| **Estado** | Revisión medida. **No** adopta nada: `phase2_ready = false`, carril B bloqueado. |
| **Aval de honestidad** | esto **no** afirma *edge*. Un «no hay serie» medido vale más que una feature prometida. |

## La pregunta

¿Se puede conseguir **gratis** una serie **histórica point-in-time** del pre-mercado del ES (la
barra de las 08:45 ET) para promoverla a **feature** del modelo? Y si no, ¿qué se puede adaptar
**hoy** con lo que ya hay, tocando el modelo o la decisión final?

## 1. Lo medido hoy (sonda directa sobre la fuente gratuita, 2026-10-11)

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

## 2. Fuentes de pago revisadas (declaradas, **sin contratar**)

| Proveedor | Plan gratuito | ¿Sirve? |
|---|---|---|
| Polygon.io | «Stocks Basic»: **solo acciones US**, 5 llamadas/min, **2 años** de histórico. Los **futuros** son producto aparte (de pago) | No (ni futuros ni 10 años) |
| Alpha Vantage | **25 llamadas/día**; el intradía histórico (parámetro `month`) es **premium** | No (el histórico de intraday no es gratis) |
| Twelve Data | Plan gratuito con pocas *credits*/min; histórico intradía amplio y **datos de horas extendidas** son de pago ($29–$999/mes) | No en el tramo gratuito |

**Veredicto (datos):** **no hay** fuente gratuita con la serie histórica que exige un *feature* del
modelo. La cotización **en vivo** a las 08:45 sí está disponible; lo que falta es el **pasado**.

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

### 3.4 Lo que **no** se puede

- **Un feature entrenado del pre-mercado** con datos gratuitos: bloqueado por #107/#149 (no hay
  historia). Solo cambia si se **contrata** una fuente con histórico point-in-time (decisión de
  gasto, `tech_stack.md` §11 bis).
- **Sustituir la barra de 24 h del ES por el pre-mercado**: su barra `t` cierra a las 16:00 ET y a
  las 08:45 **aún no existe**: usarla sería *look-ahead* (lo mide #141).

## 4. Recomendación

| Opción | Coste | Cuándo sirve | Riesgo |
|---|---|---|---|
| **3.1 Persistir el snapshot (acumular)** | bajo | a partir de ~1 año (medir) / ~10 (entrenar) | ninguno; es solo acumular |
| **3.2 Más datos declarados + avisos** | bajo | inmediato | un aviso que no decide (carril A) |
| **3.3 Medir basis y retorno nocturno** | bajo | inmediato (medición) | expectativa baja; no adopta nada |
| **Contratar histórico point-in-time** | de pago | inmediato | decisión del propietario |

Orden propuesto: **(3.1)** montar la persistencia del snapshot (es la única vía gratuita a un
*feature* real, y empieza a acumular hoy) y **(3.3)** medir las dos candidatas históricas con la
puerta de #146. **(3.2)** es una extensión barata de la regla 21 si se quiere más contexto en la
decisión.

## Cómo se reproduce la sonda

```bash
uv run python .scratch/probe_free_sources.py     # la tabla de la seccion 1 (necesita red)
```

## Referencias

`_docs/premarket_gap_2026-10-09.md` (#141) · `_docs/plan.md` §19.18 (#107) · §19.23 (#148) ·
`analysis/premarket_history.py` (#149) · `_docs/data_sources.md` · `analysis/feature_budget.py` (#146).

