# Backlog de tareas — cfdtrader

| Campo | Valor |
|---|---|
| **Versión** | **2.1** |
| **Fecha** | 2026-09-16 |
| **Instrumento** | **SPX500:CFD**, horario definido en `America/New_York` · diferencial declarado 0,0042 % |
| **Documentos de referencia** | `plan.md` v2.1 (qué construir) · `tech_stack.md` v2.0 (con qué) |
| **Total de tareas** | 48, agrupadas en 6 fases |

---

## Cómo usar este backlog

**Reglas de las tareas**

1. **Una tarea = una sesión.** Si una tarea no cabe en una sesión de trabajo, está mal definida y hay que partirla.
2. **Cada tarea entrega un artefacto verificable**: un módulo con tests, un informe, o una medición. "Avanzar en X" no es una tarea.
3. **Cada tarea termina en un commit** con el número de tarea en el mensaje (`#12 splits con purga y embargo`).
4. **Las tareas de una misma fase son mayoritariamente secuenciales.** Las que se pueden hacer en paralelo se indican en la nota de la fase.
5. **No se empieza una fase sin haber pasado la puerta de salida de la anterior.** Las puertas de salida están definidas y son vallas reales, no trámites.

**Reglas del proyecto que condicionan el orden**

- **El backtest se construye antes que los agentes** (`plan.md` §2, principio: "Construir los agentes antes que el backtest" está en la lista de errores).
- **Fase 0 no construye ningún agente ni usa el LLM.** Mide la realidad y puede reencuadrar el proyecto entero.
- **El LLM entra en la Fase 3, nunca antes**, y jamás en el camino crítico del backtest (`tech_stack.md` §3.2).
- **No se instala nada del stack completo** (`tech_stack.md` §5.2) hasta que la fase correspondiente lo pida.

**Convención de nombres:** `#N` identifica la tarea en commits, ramas y comentarios.

---
---

# Fase 0 — Medir la realidad

> **Objetivo de la fase:** responder a "¿es viable esto?" antes de construir nada.
> **No se usa LLM. No se construyen agentes.**
> **⭐ Doble puerta de salida (tarea 9) — la de la v1.x queda derogada:**
> 1. **El drift debe estar en `open→close`.** Si la descomposición de la tarea 6 muestra que la rentabilidad se concentra en `close→open`, **la estrategia intradía opera la peor parte del día** y hay que rehacerla o abandonarla. **Es la condición con poder real para invalidar el proyecto.**
> 2. ***Slippage* pequeño frente a $R$.** Con un diferencial de solo 0,42 pb (tarea 8), el *slippage* pasa a ser el coste dominante. Si el medido supera ~20 % de $R$, replantear el momento de entrada.
>
> ❌ **Derogada: la antigua puerta "$p^* > 60\%$ ⇒ parar".** Con los costes reales $p^* \approx 50{,}2\%$, cualquier modelo sesgado la superaría. **La barrera ya no es económica, es estadística** (`plan.md` §4.6).

## 1. Bootstrap del proyecto con un test

Goal: Tener un repositorio Python instalable, con lint, tipos y un test que pasa.

Description: Crear la estructura de carpetas de `plan.md` §14 con `uv` como gestor, `pyproject.toml` con dependencias fijadas, `ruff` y `pyright` configurados, `pre-commit` con detección de secretos, y `.gitignore` que excluya `data/`, `.env` y artefactos. Escribir un `.env.example` con las claves previstas (`tech_stack.md` §4.2) y **un test trivial que pase**, ejecutable con `uv run pytest`. Terminar con el primer commit y la comprobación de que `ruff`, `pyright` y `pytest` corren en verde.

## 2. Esquema del almacén point-in-time

Goal: Definir y crear el esquema de almacenamiento con marcas `as_of`, antes de guardar un solo dato.

Description: Implementar el layout `data/raw/` (inmutable, *append-only*) y `data/derived/` (recalculable) en Parquet sobre DuckDB, según `tech_stack.md` §4.3 y §12.4. Definir las columnas obligatorias de todo registro (`source`, `fetched_at`, `as_of`, `published_at`, `version`) y escribir tests que verifiquen que no se puede sobrescribir un dato crudo. Entregar el módulo de acceso al almacén y un test que escriba y relea un registro de ejemplo.

## 3. Ingesta de datos de mercado del S&P 500

Goal: Descargar y almacenar el histórico del S&P 500 y de todo su contexto con validación y fuente de respaldo.

Description: Implementar los adaptadores diarios y **el adaptador intradía necesario para etiquetado y ejecución** detrás de interfaces propias. Para `SPX500:CFD`, documentar la fuente, granularidad, cobertura histórica, timezone, bid/ask y licencia; si no existe fuente suficiente, la tarea debe bloquear la Fase 1 en vez de sustituir silenciosamente el CFD por `^GSPC` o `ES=F`. Mantener reintentos, fallback y validaciones de datos *stale*, huecos y sesiones duplicadas, entregando un informe de cobertura separado por fuente.

## 4. Calendario de sesiones, festivos y DST de EE. UU.

Goal: Poder responder con certeza si una fecha es sesión válida del mercado americano y cuántas horas dura.

Description: Construir el módulo de calendario con **festivos de Estados Unidos** (no de España: los españoles solo afectan a tu disponibilidad), **medias sesiones** con cierre a las 13:00 ET y el calendario de **cambios de horario (DST)** de EE. UU. y Europa, según `plan.md` §8.3. Implementar la regla de diseño de que **la hora de referencia interna es `America/New_York`** y `Europe/Madrid` es solo presentación, guardando todo en UTC. Entregar el módulo con un test que verifique las **dos ventanas anuales de desfase** (marzo y finales de octubre) y los festivos y medias sesiones conocidos.

## 5. Ingesta de macro americana desde FRED

Goal: Tener Fed funds, tipos del Treasury, inflación y empleo en el almacén.

Description: Implementar el cliente de **FRED** con `httpx` y `tenacity` como **fuente macro primaria** (`tech_stack.md` §4.5): Fed funds, UST a 2 y 10 años, pendiente 2s10s, CPI, PCE y NFP. Añadir opcionalmente ECB SDW y Eurostat como contexto europeo, ya sin papel protagonista. Guardar las series con `as_of` y `published_at`, y entregar un test que verifique el *point-in-time* de al menos tres publicaciones conocidas de las 08:30 ET.

## 6. ⭐ Estudio de la tasa base y descomposición del drift

Goal: Determinar si el S&P 500 gana dinero en la sesión `open→close` o fuera de ella.

Description: Calcular sobre 10+ años la tasa base de sesiones alcistas y la distribución de `|open→close|`, segmentando por día de la semana y por tramos de volatilidad. Y sobre todo, **descomponer el drift en tres tramos: `open→close`, `close→open` y `close→close`**, comparando sus medias, medianas y significación estadística, **por año y por régimen de volatilidad** (`plan.md` §1.1.a y §8.5). Entregar un cuaderno de `marimo` y un informe en `data/derived/reports/`: **es la medición decisiva del proyecto**, porque si el drift está concentrado en el tramo nocturno, la estrategia intradía está operando sistemáticamente la peor parte del día.

## 7. Volatilidad realizada, VIX y primer forecast

Goal: Tener una estimación de volatilidad condicional lista para dimensionar y etiquetar.

Description: Implementar el ATR normalizado y la volatilidad realizada (HAR con retrasos 1d/1sem/1mes) sobre Polars, más un primer forecast con `arch` (GARCH) para comparar (`tech_stack.md` §4.6). Incorporar el **VIX como feature de régimen**, aprovechando que aquí es un índice real y líquido, no un proxy (§7.1 del `plan.md`). Entregar el modelo que mejor se comporte en walk-forward, con un test de que el forecast no usa datos posteriores al instante de decisión.

## 8. Verificar los costes declarados y medir el *slippage*

Goal: Confirmar que los costes del papel son los reales y cuantificar el único que no está declarado.

Description: Partir de la tabla de costes declarada para **SPX500:CFD** (`plan.md` §3.3): **diferencial 0,0042 %, tenencia +0,0182 %/noche en largo y −0,0018 %/noche en corto, cambio de divisa 0 %**. Verificar primero la hora exacta de corte de la financiación y la ventana real de cotización. Separar y medir por separado **spread cotizado**, **tracking difference** y **slippage de ejecución**: este último solo se considera medido si existe una ejecución real o una referencia ejecutable con timestamp preciso. Registrar todos los timestamps en UTC y expresar los tramos en `America/New_York`. Entregar la plantilla y el script que consolida el coste por tramo y tamaño.

## 9. Informe de Fase 0 y decisión de continuidad

Goal: Decidir con números si el proyecto sigue adelante, se reencuadra o se para.

Description: Consolidar los resultados de las tareas 6, 7 y 8 en el cálculo de la probabilidad de break-even $p^* = (R + c) / 2R$ —que con el coste declarado da **~50,2 %**— dejando claro por escrito que **un $p^*$ bajo ya no es criterio de viabilidad** (`plan.md` §4.6). Entregar el informe de Fase 0 con la **descomposición del drift**, el *slippage* medido, el corte de financiación confirmado y una **recomendación escrita y razonada**. **Doble puerta de salida: (a) si el drift se concentra en `close→open` en vez de en `open→close`, o (b) si el *slippage* sistemático supera ~20 % de $R$, no se pasa a Fase 1.**

---
---

# Fase 1 — Arnés de backtest

> **Objetivo de la fase:** tener un motor de evaluación en el que se pueda confiar antes de tener nada que evaluar.
> **Depende de:** Fase 0 superada.
> **Paralelizable:** tareas 10 y 11 entre sí; 12 es independiente de 10 y 11.
> **Puerta de salida (tarea 18):** el motor reproduce resultados de forma determinista y bate a los baselines o se para.

## 10. Etiquetado tri-barrera

Goal: Generar etiquetas que reflejen el resultado real de la operación, no solo la dirección.

Description: Implementar el etiquetado de tres barreras para **LONG y SHORT**, determinando qué barrera se toca primero con datos intradía y barreras proporcionales a la volatilidad prevista de la tarea 7 (`plan.md` §4.2). Fijar explícitamente la semántica de `target`, `stop` y `time`, la conversión a `p_win`/EV y el tratamiento conservador de empates dentro de una barra. Entregar el módulo, la columna de etiquetas en `derived.labels` y tests construidos a mano.

## 11. Motor de costes

Goal: Que ningún backtest pueda ejecutarse con coste cero, constante o incorrecto.

Description: Implementar el modelo de coste con las **cifras reales declaradas** (`plan.md` §3.3): diferencial de 0,0042 % por defecto, tenencia de +0,0182 %/noche en largo y −0,0018 %/noche en corto, cambio de divisa 0 %, más un parámetro de ***slippage* explícito** que sea el término dominante del modelo. Hacer que el motor **exija** un coste explícito y falle si no se le pasa, y que **asuma intradía puro sin noche por defecto**, con la financiación activada únicamente si la posición sobrepasa el corte. Entregar el módulo y tests con casos calculados a mano que reproduzcan exactamente los totales declarados (0,24 $ corto y 2,24 $ largo sobre 10.000 $).

## 12. Splits con purga y embargo

Goal: Eliminar la fuga de información entre entrenamiento y evaluación.

Description: Implementar la partición walk-forward con **purga** de las muestras cuyo horizonte de etiqueta se solapa con el test y **embargo** posterior, en lugar del K-Fold aleatorio (`plan.md` §11.1). Documentar el criterio y entregar tests que verifiquen que ninguna muestra de entrenamiento tiene solapamiento temporal con el test. Es el módulo que hace creíble todo lo demás, así que se implementa antes que cualquier modelo.

## 13. Motor de backtest walk-forward

Goal: Tener el bucle de evaluación que ejecuta la estrategia sesión a sesión.

Description: Implementar el motor de `plan.md` §15 como una función pura y determinista que recorre las sesiones, obtiene el snapshot point-in-time, llama al modelo, aplica el gate y simula la operación respetando `bid`/`ask` y el *gap* de apertura. Integrar purga y embargo y el modelo de costes de las tareas 11 y 12. Entregar el motor con un test de determinismo: dos ejecuciones idénticas deben dar resultados byte a byte iguales.

## 14. Baselines triviales

Goal: Saber cuál es el listón mínimo que hay que superar.

Description: Implementar los seis baselines de `plan.md` §11.2: no operar, siempre largo, siempre corto, momentum 5d, reversión de gap y regla aleatoria con la misma frecuencia de operación. Ejecutarlos todos sobre el histórico con costes reales y entregar la tabla comparativa de métricas netas. El baseline "no operar" (Sharpe = 0) es el más difícil de batir y el que decide si el proyecto tiene sentido.

## 15. Métricas netas y de calibración

Goal: Medir el rendimiento con las métricas correctas, no con accuracy.

Description: Implementar Sharpe y Sortino netos con **intervalo de confianza por bootstrap**, EV por operación, ratio ganancia/pérdida, drawdown máximo y duración, y las métricas de probabilidad (Brier score, log-loss y curva de calibración), según `plan.md` §11.3. Integrar `quantstats` para el informe visual. Entregar el módulo de métricas y un test que verifique el bootstrap contra un caso con resultado conocido.

## 16. Corrección por sobreajuste

Goal: Poder detectar cuándo un buen resultado es fruto del azar.

Description: Implementar el **Deflated Sharpe Ratio** y la **Probability of Backtest Overfitting** con el número real de variantes probadas (`plan.md` §11.4). Añadir el registro obligatorio de cada experimento (features, hiperparámetros, resultado) en `runs/<hash>/` con su configuración versionada. Entregar los dos cálculos y el mecanismo de registro, y validarlos con series sintéticas de ruido puro donde deben dar resultado no significativo.

## 17. Suite de tests de integridad

Goal: Blindar el proyecto contra los cuatro errores que lo invalidarían en silencio.

Description: Escribir los cuatro tests críticos de `plan.md` §11 y `tech_stack.md` §4.14: **no-look-ahead** (un feature en `t` no cambia al añadir datos de `t+1`), *golden dataset* de features con hash congelado, determinismo del gate y casos de coste calculados a mano. Añadir el *golden dataset* al CI de forma que el build falle si un hash cambia sin subir la versión declarada. Entregar la suite completa en verde; **el test de no-look-ahead es el más valioso de todo el proyecto**.

## 18. Informe de Fase 1 y puerta de salida

Goal: Verificar que el arnés es fiable antes de construir nada encima.

Description: Ejecutar los baselines con el motor completo, documentar los resultados y auditar que la purga, el coste y el determinismo funcionan como se especificó. Entregar el informe de Fase 1 con la tabla de baselines y la verificación del arnés. **Puerta de salida: si el motor no es determinista, o si "siempre largo" bate a todos los baselines de forma significativa, se para y se replantea la estrategia.**

---
---

# Fase 2 — Núcleo cuantitativo

> **Objetivo de la fase:** un modelo calibrado y un gate determinista que batan a los baselines.
> **Depende de:** Fase 1 superada.
> **Paralelizable:** las tareas 20, 21, 22 y 23 son independientes entre sí; todas dependen de 19.
> **Sin LLM en toda la fase.**
> **Puerta de salida (tarea 29):** se aplican los criterios de kill pre-registrados de `plan.md` §11.6.

## 19. Feature store con versionado y hash

Goal: Que cada cálculo de features sea reproducible y trazable.

Description: Implementar el módulo que persiste la matriz de features por sesión en `derived.features_daily` con `features_version = sha256(código + parámetros + ventanas + fuente + as_of)`, según `tech_stack.md` §12.4 y `plan.md` §9. Añadir normalización robusta sobre ventana expandida (nunca sobre toda la muestra, que sería *look-ahead*) y el test de *golden dataset* que congela el comportamiento. Entregar el módulo, el esquema y la política de versionado documentada.

## 20. Features técnicas

Goal: Tener la familia de features de precio y tendencia.

Description: Implementar **a mano** los 5–10 indicadores necesarios (retornos multi-ventana, ATR normalizado, distancia a medias, RSI, posición en rango, rupturas de volatilidad), justificado en `tech_stack.md` §4.6 para controlar la ventana exacta y el `as_of`. Cada feature se documenta con su fórmula, su ventana y el `as_of` que requiere. Entregar el módulo con tests por indicador y las features escritas en el almacén con su versión.

## 21. Features de contexto de mercado

Goal: Capturar lo que dicen los mercados relacionados sobre el S&P 500.

Description: Implementar correlaciones rodantes con DAX, FTSE, EuroStoxx 50 y Nikkei, el comportamiento nocturno de Asia, el retorno del S&P 500 del día anterior, la beta al VIX y la variación del DXY (`plan.md` §9). Implementar la **dispersión sectorial usando ETFs sectoriales americanos**, que es la solución adoptada contra el sesgo de supervivencia que tendría usar la composición actual del índice (`plan.md` §7.1). Entregar el módulo, las features versionadas y tests de que las correlaciones se calculan sobre ventana móvil y no sobre la muestra completa.

## 22. Features macro

Goal: Incorporar los drivers macro con fundamento económico específicos del S&P 500.

Description: Construir features de nivel y variación de la **Fed funds**, del **UST a 10 años y a 2 años**, de la **pendiente 2s10s**, del **DXY** y de la inflación (CPI/PCE), aprovechando que el índice está muy concentrado en mega-cap tecnológicas sensibles a la duración (`plan.md` §7.1). Respetar estrictamente el `published_at` real de cada serie, almacenado en UTC y comparado con el snapshot en `America/New_York`; no asumir una hora fija en Madrid. Cuando existan revisiones, conservar también el vintage o declarar la limitación. Entregar el módulo con tests point-in-time, incluidos los días de desfase DST.

## 23. Features de régimen y volatilidad

Goal: Que el modelo sepa en qué tipo de mercado está.

Description: Implementar percentil de volatilidad realizada, régimen de tendencia frente a rango, y el forecast de volatilidad elegido en la tarea 7 (`plan.md` §9). Añadir features de calendario: día de la semana y proximidad a vencimientos y rebalanceos. Entregar el módulo con tests y las features en el almacén; esta familia alimenta directamente el dimensionamiento del gate.

## 24. Modelo baseline con purged CV

Goal: Tener un primer modelo entrenado y evaluado de forma honesta.

Description: Entrenar una regresión logística regularizada (elastic net) sobre las features de las tareas 20–23 usando las etiquetas tri-barrera y los splits con purga y embargo, según `plan.md` §10. Documentar el número de features frente al tamaño muestral disponible para evitar el sobreajuste por exceso de parámetros. Entregar el módulo de entrenamiento, el modelo serializado en `runs/` y el informe de métricas netas frente a los baselines.

## 25. Calibración de probabilidades

Goal: Que una probabilidad de 0,60 signifique de verdad un 60%.

Description: Añadir calibración con `CalibratedClassifierCV`, **usando Platt en lugar de isotónica si hay menos de 500 muestras de calibración**, e integrar la calibración dentro del esquema de purga para no contaminarla (`plan.md` §10). Validar con la curva de calibración y el Brier score antes y después. Entregar el modelo calibrado y la comparación numérica que demuestre la mejora; sin esto, el EV y el sizing serían basura aunque el ranking fuese bueno.

## 26. Comparación con LightGBM y selección de modelo

Goal: Elegir el modelo definitivo con criterio, no por preferencia.

Description: Entrenar un LightGBM pequeño con `min_child_samples` alto y compararlo con el modelo lineal bajo el mismo protocolo de purga, métricas y calibración (`plan.md` §10). Aplicar Deflated Sharpe con el número real de variantes probadas y **registrar todos los experimentos** en `runs/`. Entregar el informe comparativo y la decisión documentada, sabiendo que con ~250–375 operaciones útiles un modelo simple suele ganar.

## 27. Gate de decisión y sizing

Goal: Implementar el corazón del sistema como una función pura y determinista.

Description: Implementar `decision/gate.py` tomando probabilidad calibrada, movimiento esperado, coste y snapshot, y devolviendo `LONG`/`SHORT`/`NOTHING` con stop, objetivo, nocional y tier, según `plan.md` §7.3 paso 4 y §12. Calcular el nocional desde el riesgo y la distancia al stop (**nunca desde el apalancamiento**) y aplicar las **reglas duras 1–18**, incluidas las específicas de este instrumento: el **mecanismo de cierre** (bracket obligatorio), los **días de FOMC** y las **medias sesiones**. Entregar el gate con el test de determinismo y tests de que respeta cada límite, cada bloqueo y cada regla de riesgo.

## 28. Backtest del pipeline completo contra baselines

Goal: Saber si el sistema completo aporta algo sobre lo trivial.

Description: Ejecutar el pipeline (features, modelo calibrado, gate y sizing) sobre todo el histórico con el motor de la Fase 1 y compararlo contra los seis baselines y contra el benchmark (`plan.md` §11.2 y §11.3). ⚠️ **El baseline "siempre largo" debe evaluarse en dos versiones: `open→close` y `close→close`**, porque en el S&P 500 el drift es fuerte y puede estar en el tramo nocturno (`plan.md` §11.2). Separar explícitamente alpha de beta para responder si el sistema gana por sí mismo o solo por estar expuesto al mercado. Entregar el informe de backtest completo con intervalos de confianza bootstrap en todas las métricas.

## 29. Informe de Fase 2 y criterios de kill

Goal: Aplicar los criterios de parada pre-registrados sin excepción.

Description: Evaluar el resultado de la tarea 28 contra la tabla de criterios de kill de `plan.md` §11.6 (Sharpe neto OOS, PBO, Deflated Sharpe, drawdown y superación de baselines), que **no se modifica después de ver los resultados**. Entregar el informe de Fase 2 con el veredicto y, si procede, la decisión de simplificar el modelo o parar. **Puerta de salida: si no se bate a "no operar" y a "siempre largo" de forma significativa, no se pasa a Fase 3.**

---
---

# Fase 3 — Capa LLM

> **Objetivo de la fase:** añadir el overlay de noticias con poder acotado y medir si aporta.
> **Depende de:** Fase 2 superada.
> **Paralelizable:** tareas 30 y 31 entre sí; 34 es independiente de 30–33.
> **Recordatorio:** el LLM tiene derecho a **veto** y a un ajuste de **±10 puntos porcentuales**. No decide ni calcula (`plan.md` §6.2).

## 30. Ingesta de noticias con published_at

Goal: Recoger titulares con su hora real de publicación y sin duplicados.

Description: Implementar la ingesta de GDELT y de RSS con `feedparser`, guardando **titular, URL, `published_at` y hash, pero nunca el cuerpo del artículo** (`tech_stack.md` §12.7). Añadir deduplicación por hash del titular normalizado más similitud difusa, sin *embeddings*, y respetar `tenacity` con backoff y caché en disco. Entregar el módulo, el esquema `raw.news_headlines` poblado y un test que verifique que nada se guarda con `published_at` posterior al instante de ejecución.

## 31. Interfaz LLMClient

Goal: Que el proveedor de LLM sea sustituible sin tocar el resto del sistema.

Description: Definir el protocolo interno `LLMClient` y su implementación sobre el SDK de `openai` con `base_url` y `model` configurables, de modo que OpenAI y DeepSeek se cubran con un solo cliente (`tech_stack.md` §4.9). Configurar `temperature=0`, el registro de `model` y `system_fingerprint`, y la configuración tipada con `pydantic-settings`. Entregar la interfaz, la implementación y un test con el proveedor simulado que verifique que `agents/` no importa nada específico del proveedor.

## 32. NewsAgent con salida estructurada

Goal: Extraer eventos de noticias como datos tipados, no como prosa.

Description: Implementar el `NewsAgent` que recibe lotes de titulares y devuelve `list[NewsEvent]` validado con Pydantic, **asumiendo que ningún proveedor garantiza el esquema**: validar siempre, reintentar con el error de validación en el mensaje y descartar el evento tras N intentos (`tech_stack.md` §4.9). Escribir los prompts en `Jinja2` y guardar su hash. Entregar el agente con tests sobre un conjunto fijo de titulares y la garantía de que un evento que no valida nunca llega al gate.

## 33. Caché, deduplicación y control de coste

Goal: Que el gasto del LLM esté acotado, medido y no pueda bloquear el pipeline.

Description: Implementar la caché en disco de `diskcache` con clave `hash(prompt + modelo + inputs)`, la deduplicación previa de titulares y las siete palancas de reducción de coste de `tech_stack.md` §6.3. Implementar los topes duros (tokens, llamadas, gasto diario y mensual, tiempo máximo, fallos consecutivos) y registrar cada llamada en `ops.llm_calls`. Entregar el módulo con un test que demuestre que **superar un tope desactiva el overlay y el pipeline sigue produciendo recomendación**, nunca se bloquea.

## 34. EventCalendarAgent

Goal: Identificar los días en los que la respuesta correcta es no operar.

Description: Implementar el agente que marca OPEX, *triple witching*, el **roll trimestral del futuro ES**, los **días de FOMC**, las publicaciones macro con sus timestamps oficiales en ET/UTC, las **medias sesiones americanas** y el calendario de resultados de las mega-caps. Consumir fuentes versionadas y point-in-time, producir una señal tipada con la lista de eventos bloqueantes y probar fechas conocidas, incluidos cambios DST.

## 35. Overlay del NewsAgent en el gate

Goal: Dar al LLM exactamente el poder que se le concedió, ni más ni menos.

Description: Integrar el `NewsAgent` en el gate como overlay con **veto binario** y ajuste acotado a ±10 puntos porcentuales, registrando qué hizo (`applied`, `veto`, `disabled_*`) en el diario (`tech_stack.md` §4.9). Asegurar que el overlay es opcional por diseño: sin él, la recomendación debe seguir emitiéndose. Entregar la integración con tests que verifiquen el límite del ajuste y que un veto no puede invertir la dirección.

## 36. Orquestación con LangGraph

Goal: Coordinar el pipeline sin meter el orquestador en el camino crítico.

Description: Montar el grafo de LangGraph con estado tipado por Pydantic, nodos expertos en paralelo, *fan-in* y punto de decisión humana vía `interrupt`, según `tech_stack.md` §4.10. Dejar el gate como función pura invocable fuera del grafo, y fijar la versión de LangGraph de forma exacta. Entregar el grafo con un test que verifique que el backtest puede ejecutarse **sin arrancar LangGraph**, que es la garantía de que el núcleo sigue siendo retrotesteable.

## 37. Informe diario

Goal: Producir el informe antes de la apertura americana, con horarios derivados de `America/New_York`.

Description: Implementar la plantilla `Jinja2` y la llamada al **modelo de mayor calidad** para redactar el informe a partir de datos ya calculados, incluyendo dirección, probabilidad calibrada, EV neto, stop, objetivo, tier y **el contra-argumento del `DevilAdvocateAgent`** (`plan.md` §7.1 y §13). Guardar el informe tal cual se emitió en `journal.decisions.report_text`. Entregar el módulo y un ejemplo de informe generado con datos reales de un día pasado.

## 38. Evaluación incremental del overlay

Goal: Determinar con evidencia si el LLM aporta algo o solo redacta.

Description: Comparar el Brier score y el Sharpe OOS **con el overlay activo y desactivado**, y documentar el resultado sin adornos, sabiendo que el backtest corre siempre **sin** overlay porque no hay archivo histórico de noticias (`tech_stack.md` §4.9). Diseñar el registro de *paper trading* prospectivo como método de evaluación alternativo. Entregar el informe de evaluación con una conclusión explícita: si el overlay no mejora nada medible, el LLM se queda **solo como redactor**.

---
---

# Fase 4 — Operación

> **Objetivo de la fase:** que el sistema funcione solo cada día, con degradación grácil, diario completo y **garantía de cierre a las 22:00**.
> **Depende de:** Fase 3 superada.
> **Paralelizable:** 41, 42 y 43 entre sí, una vez hecha 39.
> **Puerta de salida (tarea 45):** paper trading de 2–3 meses sin divergencia superior a 2σ frente al backtest.

## 39. Capa de persistencia del diario

Goal: Implementar el modelo de persistencia completo de `tech_stack.md` §12.

Description: Crear las tablas `journal.decisions`, `journal.agent_signals`, `journal.trades` y `journal.overrides` con las columnas obligatorias de §12.5, más `ops.run_log`, `ops.llm_calls` y `ops.backtest_runs` de §12.6. Implementar el campo `status` con sus cuatro valores y el registro de todas las versiones (`features_version`, `model_version`, `git_commit`, `prompt_hashes`). Entregar el módulo con tests de escritura y lectura y la verificación de que **una sesión sin recomendación se registra igual que una con recomendación**.

## 40. Guardia de obsolescencia y los cuatro estados

Goal: Que el sistema nunca emita una recomendación con datos que no corresponden a la sesión actual.

Description: Implementar la guardia de `tech_stack.md` §8.4 y las reglas duras 13–15 de `plan.md` §12: si el `as_of` no es la fecha de hoy no se emite recomendación accionable, y "no sé" se distingue de `NOTHING` en el registro y en la notificación. Implementar el modo observación de 5 sesiones tras una ausencia prolongada y el tratamiento de **festivos americanos y medias sesiones** (que en este instrumento devuelven `NOTHING` justificado, no "no sé"). Entregar el módulo con tests que simulen un PC apagado tres días, un festivo de EE. UU. y una media sesión, verificando que **no se emite ninguna recomendación accionable cuando no corresponde**.

## 41. Scheduler con systemd anclado a Nueva York

Goal: Que el pipeline se ejecute solo, siempre en el mismo punto relativo a la sesión americana y a su DST.

Description: Crear las unidades `systemd` de usuario (`.service` y `.timer`) con `OnCalendar`, `Timezone=America/New_York`, `Persistent=true` y `flock`, según `tech_stack.md` §4.11 y §8.1, más el timer secundario de registro de cierre a las 16:20 ET. Verificar que cada disparo se convierte correctamente a `Europe/Madrid` en ambas ventanas DST y que una ejecución recuperada por `Persistent=true` queda bloqueada por la guardia de sesión si llega tarde.

## 42. Notificaciones, alarma de cierre y heartbeat

Goal: Recibir el informe antes de la apertura y, sobre todo, **no olvidar cerrar a las 16:00 ET**.

Description: Implementar el envío del informe a Telegram con `httpx` contra la Bot API, más las alertas de fallo y de recomendación obsoleta. La alarma de cierre debe programarse a **15:45 `America/New_York`** y convertirse a Madrid solo para presentar el mensaje. Registrar el incumplimiento si la posición sigue abierta a las 16:00 ET (`plan.md` §12, regla 16). Entregar el módulo con pruebas de las dos ventanas DST.

## 43. Observabilidad y heartbeat

Goal: Que ningún fallo se manifieste en forma de silencio.

Description: Implementar el `run_log` por ejecución en JSONL con etapas, duraciones y errores, más el `manifest.json` con hashes y versiones (`tech_stack.md` §4.13). Implementar la alerta de *heartbeat*: si a las **09:40 ET** no ha llegado el informe (y si a las **16:20 ET** no se ha registrado el cierre), se notifica. Entregar el módulo de logging estructurado y un test que verifique que un fallo en cualquier etapa produce alerta y no un silencio.

## 44. Retención y job de tamaños

Goal: Evitar que el disco crezca sin control y sin que nadie lo note.

Description: Implementar el job mensual que mide el tamaño de cada directorio y **avisa si supera su presupuesto**, más el job trimestral que aplica la política de retención de `tech_stack.md` §12.7 y §12.9. Incluir la purga de la caché del LLM con más de 12–18 meses y de los binarios de modelos no productivos. Entregar los dos trabajos programados y un informe de ejemplo con los tamaños actuales frente a los presupuestados.

## 45. Paper trading prolongado

Goal: Comprobar si el sistema se comporta en la realidad como en el backtest.

Description: Ejecutar el pipeline en modo observación durante 2–3 meses registrando **todas** las recomendaciones y sus resultados, sin operar o con importe simbólico, según `plan.md` §16 Fase 4. Medir la divergencia frente al backtest con un umbral de 2σ y documentar cualquier desviación. Entregar el informe de paper trading con la comparación y la recomendación de pasar a capital real o no. **Puerta de salida: divergencia superior a 2σ obliga a auditar antes de operar.**

---
---

# Fase 5 — Producción

> **Objetivo de la fase:** operar con tamaño mínimo y mantener el sistema en el tiempo.
> **Depende de:** Fase 4 superada sin divergencia significativa.

## 46. Auditoría de reconstrucción

Goal: Demostrar que cualquier decisión pasada es reconstruible exactamente.

Description: Ejecutar la prueba de fuego de `tech_stack.md` §12.9: elegir una decisión de hace aproximadamente un año, hacer *checkout* del `git_commit` registrado, restaurar el `uv.lock`, recomputar las features desde `raw` y comparar con lo guardado en `journal.decisions`. Si no coincide, localizar la fuente de no determinismo antes de seguir. Entregar el informe de auditoría y, si procede, los arreglos de reproducibilidad.

## 47. Puesta en producción con tamaño mínimo

Goal: Operar capital real sin arriesgar más de lo que el sistema ha demostrado.

Description: Activar el sistema para operar solo señales **tier A** con tamaño mínimo y las reglas duras de `plan.md` §12 activas, incluido el límite de pérdida diaria y el *kill switch*. Registrar la ejecución real en `journal.trades` con precios y costes efectivos para poder auditar más adelante. Entregar el sistema en producción con el procedimiento de operación manual documentado y probado un día completo.

## 48. Revisión trimestral y poda de agentes

Goal: Mantener el sistema simple retirando lo que no aporta.

Description: Evaluar con datos qué agentes aportan valor marginal y cuáles no, aplicando la regla de `plan.md` §19.5 de **podar, no añadir**. Revisar el drift, recalibrar, auditar el coste real frente al modelado y actualizar los criterios de kill si la evidencia lo justifica. Entregar el informe trimestral y las decisiones de simplificación, con el registro en `tech_stack.md` §11 bis de cualquier decisión abierta que se haya cerrado.

---

# Resumen

| Fase | Tareas | Sesiones | Foco |
|---|---|---|---|
| 0 — Medir la realidad | 1–9 | 9 | Viabilidad: **drift, tasa base y coste real** |
| 1 — Arnés de backtest | 10–18 | 9 | Motor de evaluación fiable |
| 2 — Núcleo cuantitativo | 19–29 | 11 | Modelo calibrado y gate |
| 3 — Capa LLM | 30–38 | 9 | Overlay con poder acotado |
| 4 — Operación | 39–45 | 7 | Automatización, diario y **mecanismo de cierre** |
| 5 — Producción | 46–48 | 3 | Auditoría y mantenimiento |
| **Total** | **48** | **48** | |

**Las cuatro tareas que más peso tienen en el resultado del proyecto, y no son las que parecen:**

1. ⭐ **Tarea 6 — la descomposición del drift.** Es la única que puede **invalidar el planteamiento entero**: si el S&P 500 gana en el tramo nocturno y no en `open→close`, la estrategia intradía está operando la peor parte del día. Ninguna otra tarea tiene ese poder.
2. **Tarea 8 — verificar la hora de corte de la financiación y medir el *slippage*.** El diferencial real es de 0,42 pb, prácticamente un *tick* de futuro, así que **el *slippage* es el único coste que puede hundir el proyecto**. Y una hora de corte mal entendida multiplica el coste por más de cinco.
3. **Tarea 17 — test de no-look-ahead.** Es el que evita el fallo más caro y el que casi nadie escribe.
4. **Tarea 39 — el diario.** Es el único dato del sistema que no se puede reconstruir.

> ⚠️ **Advertencia que introduce la revisión de costes:** con $p^* \approx 50{,}2\%$, **ninguna tarea de este backlog va a producir una confirmación de que el sistema funciona** en un plazo razonable (§4.5 y §4.6 del `plan.md`). El valor verificable del proyecto está en las tareas **1–18** (proceso, motor, integridad) y en la **6** (la medición del drift), no en ver un P&L verde. Si esperas lo segundo, este backlog te va a decepcionar.

**La tarea más frágil en producción:** la **42** (alarma de cierre). El requisito de intradía puro depende de un acto manual a las 22:00, y **el coste de un descuido está cuantificado**: 0,0182 % de tenencia en un largo —más de cuatro veces el diferencial— más un *gap* de 17,5 horas.

**Decisiones abiertas que hay que cerrar antes de ciertas tareas** (`tech_stack.md` §11 bis): el bróker y el umbral de coste (antes de la 9), el horizonte y precio de entrada (antes de la 10), el *holdout* intocable (antes de la 24), el proveedor de LLM definitivo y la contratación de histórico de noticias (antes de la 31 y la 38).

---

## Registro de cambios

| Fecha | Versión | Cambio | Motivo |
|---|---|---|---|
| 2026-09-16 | 1.0 | Versión inicial del backlog: 48 tareas en 6 fases | Derivado de `plan.md` v1.1 y `tech_stack.md` v1.4 |
| 2026-09-16 | **2.0** | ⚠️ **REPLANTEO POR CAMBIO DE INSTRUMENTO A CFD DEL S&P 500.** Fase 0 reescrita: tareas 3–8 con los datos, el calendario y las fuentes americanas; **la tarea 6 pasa a ser la descomposición del drift** (antes solo tasa base) y se convierte en la medición decisiva; la tarea 8 incorpora la confirmación de la ventana de cotización y la divisa de liquidación. La puerta de salida de la Fase 0 pasa a tener **dos condiciones**. Fase 2: tareas 21, 22 y 27 adaptadas a los drivers y reglas del nuevo instrumento. Fase 3: tarea 34 incorpora FOMC, roll del ES, medias sesiones y **resultados de mega-caps**. Fase 4: tarea 40 con festivos US y medias sesiones, tarea 41 anclada a `America/New_York`, tarea 42 ampliada con la **alarma de cierre**. Resumen final con cuatro tareas críticas en lugar de tres | **Decisión del usuario: operar el CFD del S&P 500**, con decisión a las 15:00, entrada en la apertura US, cierre a las 22:00 y sin overnight |
| 2026-09-16 | **2.1** | ⭐ **COSTES REALES INCORPORADOS.** **Tarea 8 reformulada**: pasa de "confirmar la ventana y medir el coste" a **verificar los costes declarados**, priorizando la **hora exacta de corte de la financiación** y la **medición del *slippage***, que pasa a ser el coste dominante. **Tarea 9**: la antigua puerta "$p^* > 60\%$ ⇒ parar" queda **derogada** y se sustituye por *slippage* vs $R$. **Tarea 11**: el motor de costes se especifica con las cifras reales y un test que reproduzca los totales declarados. **Tarea 6** y resumen final matizados con la advertencia de que el sistema **no será validable por resultado** en plazo razonable | **Datos de coste aportados por el usuario.** Con un diferencial de 0,42 pb el listón económico desaparece ($p^* \approx 50{,}2\%$) y toda la dificultad se traslada al terreno estadístico (`plan.md` §4.6) |
