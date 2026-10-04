# Material para decidir `R` y los umbrales (decisión abierta 5, #60)

> **Qué es este documento.** Material **para una decisión del propietario**, no la decisión. No fija
> `R`, no cambia ninguna regla, no reabre `§11.6` y no toca la puerta pre-registrada de la Fase 4
> (`plan.md` §16, #130). Su único trabajo es dejar los números delante y separar lo **medido** de lo
> **asumido**, para que la decisión no se tome a ciegas.
>
> **Por qué se escribe ahora.** Arrancar la observación de la Fase 4 (#45) exige que el gate pueda
> autorizar una operación. Hoy no puede, y no por falta de datos: por una **decisión abierta**. Lo
> cuenta con reproducción **#131**.

---

## 1. El eslabón que falta, exactamente

1. El propietario declaró el 2026-09-18 un *slippage* **asumido** (`state="assumed"`,
   `is_measurement=False`) de **20 % de `R`** = el margen completo de la puerta (b). Es un supuesto
   **pesimista**, del que el propio código dice que «**no** sustituye a la medición pendiente (#62)».
2. Ese supuesto se publica **como ratio sobre `R`**, y `R` viaja como `null`: `pct_of_notional` es
   `None` *mientras `R` no esté decidido* (`backtest/costs.py`, `SlippageParameter`).
3. Sin valor numérico, **`c_total_pct` es `null`** ⇒ **`ev_net_pct` es `null`** (nunca se sustituye
   por 0).
4. La **regla 9** (`EV neto > umbral`, `plan.md` §12) no se puede verificar y bloquea
   (`ev_neto_no_calculable`); y el **tier A** exige `EV neto > 3c`, así que es inalcanzable (regla 10,
   `tier_no_autorizado`).

⇒ **Decidir `R` convierte el supuesto en un coste numérico y desbloquea el cálculo del EV.** Es
condición necesaria. **Si es suficiente es otra pregunta**, y la responde §3.

## 2. Qué es `R` aquí — y la ambigüedad que hay que cerrar con la decisión

- `plan.md` **§4.4** define `R` como la **amplitud del bracket** y da la condición de break-even
  `p* = (R + c) / 2R`. Su tabla (§4.4) usa **±`R` simétrico**.
- El modelo, en cambio, produce un bracket **asimétrico 2:1**. La última sesión ejecutada declaró
  `stop_pct = 0,5463 %` y `target_pct = 1,0926 %`. **Ninguno de los dos es «±R»**: decidir `R` exige
  decir **cuál** de las tres lecturas se adopta (distancia al stop · distancia al objetivo · amplitud
  simétrica declarada).
- `plan.md` **§12 regla 2** ata `R` al tamaño: el riesgo en el peor caso (stop alcanzado) ≤ **1 % del
  capital**. O sea: `R` no es solo un parámetro de la fórmula de break-even, **también fija el
  nocional**.

## 3. El material: qué implica cada `R` candidato

Diferencial de ida y vuelta declarado: **0,0042 % = 0,42 bp**. El supuesto es **20 % de `R`**.

| `R` (% del nocional) | Supuesto: *slippage* | `c` total | `p*` con el supuesto | `p*` con solo el diferencial | `EV` a `p = 0,5054` |
|---:|---:|---:|---:|---:|---:|
| 0,50 % | 10 bp | 0,1042 % | **60,42 %** | 50,42 % | **−0,0988 %** |
| 0,75 % | 15 bp | 0,1542 % | 60,28 % | 50,28 % | −0,1461 % |
| **1,00 %** | **20 bp** | **0,2042 %** | **60,21 %** | **50,21 %** | **−0,1934 %** |
| 1,50 % | 30 bp | 0,3042 % | 60,14 % | 50,14 % | −0,2880 % |
| 2,00 % | 40 bp | 0,4042 % | 60,10 % | 50,10 % | −0,3826 % |

- La columna `p*` con solo el diferencial reproduce el **50,21 %** de `plan.md` §4.4: la aritmética
  del documento y la del código coinciden.
- **El `EV` es negativo en las cinco filas** a la probabilidad calibrada que el sistema dio en su
  última ejecución real (`0,5054`). Con el supuesto declarado, **ningún `R` de la banda razonable
  produce un `EV` positivo**.
- `R` en la banda **0,5 %–1,1 %** es lo que el modelo produce de verdad (`stop_pct` 0,5463 %,
  `target_pct` 1,0926 % en la última sesión). No es una banda hipotética.

### El supuesto, confrontado con la apertura **medida**

La evidencia medida (`analysis/cost_audit.py`, 59 sesiones de `^GSPC` a 5 min, movimiento de la
subasta de apertura a 09:35 ET) da **mediana 13,2 bp**, **p90 26,6 bp** y **máximo 46,8 bp**, frente a
un diferencial declarado de 0,42 bp (≈30× menos).

| `R` | Supuesto (20 % de `R`) | ¿Dónde cae frente a lo medido? |
|---:|---:|---|
| **0,50 %** | 10 bp | **por debajo de la mediana medida (13,2 bp): no es pesimista, es optimista** |
| 0,66 % | 13,2 bp | iguala la mediana |
| 1,00 % | 20 bp | entre la mediana y el p90 |
| 1,33 % | 26,6 bp | iguala el p90 |
| 2,34 % | 46,8 bp | iguala el máximo medido |

Es decir: el supuesto solo es **pesimista** para `R ≳ 1,33 %`. En la banda que el modelo produce de
verdad (`R ≈ 0,5–1,1 %`), el «supuesto pesimista» **está en la mediana de lo medido**, no por encima.

## 4. La conclusión que hay que mirar de frente

**Decidir `R` no desbloquea una operación por sí solo.** Con el supuesto tal como está declarado, la
barrera económica pasa de **~50,2 %** (solo diferencial) a **~60 %** y el `EV` sale **negativo** a la
probabilidad que el sistema está dando. El gate seguiría diciendo `NOTHING`, ahora por un número en
vez de por un `null`. Y como la puerta de la Fase 4 (#130) mide la media de las recomendaciones
**emitidas** (`long`/`short`), el resultado sería el mismo que hoy: `N = 0` ⇒ **`not_evaluable`**
después de 2–3 meses.

Y hay un segundo cruce que conviene tener delante: `plan.md` **§4.6** calcula que un edge del 52 %
exige **~4.900 operaciones** para ser distinguible. La barrera del 60 % no es un ajuste fino: está
fuera del terreno que el propio plan declaró alcanzable.

## 5. Las opciones (no las decidimos aquí)

| Opción | Qué implica | Consecuencia |
|---|---|---|
| **Cerrar #60: fijar `R` y los umbrales, manteniendo el supuesto** | El coste deja de ser `null` | El gate calcula el EV… y sale **negativo**: seguiría sin autorizar. Desbloquea el **mecanismo**, no la **operación** |
| **Fijar `R` y revisar el supuesto a la baja, con la evidencia medida** (p. ej. la mediana de 13,2 bp en vez del 20 % de `R`) | El coste baja de ~20 bp a ~13 bp con `R = 1 %` | Acerca `p*` al 50,2 %, pero **convierte un supuesto pesimista en un supuesto benévolo**, y eso hay que declararlo en voz alta o es moverse la portería a uno mismo |
| **Fijar `R` y operar la observación con un coste declarado como *escenario*, publicando el `EV` bajo varios valores** | El gate autoriza bajo el escenario declarado, y cada sesión publica la sensibilidad | Produce evidencia **medible**, pero la salida tiene que declarar que el neto sale de un supuesto. Es la única que a la vez respeta la honestidad y permite observar |
| **No cerrar `R`** (dejarlo como está) | Nada cambia | La Fase 4 no produce nada: 2–3 meses sin evidencia y `not_evaluable` |

> ⚠️ En las cuatro, **quien decide es el propietario**: `#60` es una decisión abierta de
> `tech_stack.md` §11 bis, y `analysis/cost_audit.py` ya dejó escrito el principio que la gobierna —
> **una asunción no es una medición, y asumir tu propio peor caso no puede ser un aprobado**.

### Un precedente que ya existe en el repositorio (y que acota la opción 3)

No hay que inventarlo. El informe del pipeline (`analysis/pipeline_report.py`) ya publica un **tercer
brazo**, `coste_declarado`, que **no es una validación** (`is_validation = false`) y existe
precisamente para responder a «¿operaría el pipeline alguna sesión sin esperar a #60/#62?»:
re-deriva la decisión sobre el **coste declarado** —sin el término supuesto— con la regla literal
que el propio informe publica.

Y el motivo por el que su brazo hermano (`escenario`, S1) **no opera ninguna sesión** está escrito y
**medido**, no afirmado: con el supuesto en `assumed`, `c_total_pct` es `null`, el tier es **C por
construcción** y bloquean las reglas 9 y 10. Es **el mismo mecanismo** que bloquea el camino diario.

⇒ La opción 3 **no es una invención nueva**: es extender al camino diario un patrón que el repositorio
ya publica y ya etiqueta como lo que es. Ese es el argumento a favor; en contra sigue estando que
cualquier coste que no sea la medición (#62) es un supuesto, y hay que decirlo en la salida.


## 6. Procedencia y recomputabilidad

Todos los números de este documento salen de la API pública del repositorio, no de aritmética
manual: `analysis.cost_audit.slippage_assumption_block` (la equivalencia del supuesto),
`backtest.costs.DECLARED_SPREAD_HALF_PCT` (el diferencial) y `plan.md` §4.4 (la fórmula `p*`).
`tests/test_r_decision_material.py` los **recalcula** y comprueba que las cifras de arriba siguen
siendo las que da el código: si el supuesto, el diferencial o la evidencia medida cambian, esta
hoja **falla en voz alta** en vez de quedarse vieja.
