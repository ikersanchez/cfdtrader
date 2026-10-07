# Material para las decisiones del bróker (#59, #87, #62, #107, #51)

> **Qué es este documento.** Material **para decisiones del propietario**, no la decisión.
> No elige bróker, no fija `financing_cut`, no mide el *slippage*, no compra datos y no reabre
> `§11.6` ni el carril B. Su único trabajo es reunir en **un solo cuestionario al bróker** las
> preguntas que hoy están repartidas entre `analysis/cost_audit.py`, `plan.md` §8.5/§21 y
> `config/cost_observations.yaml`, y decir **dónde** se anota cada respuesta.
> ⚠️ **quien decide es el propietario**: `#59` es la decisión **4** de `tech_stack.md` §11 bis
> (**cerrada el 2026-10-07** con el KID de Revolut), y `#87` es «la verificación más crítica de la
> Fase 0».

> **Por qué se escribe ahora.** El Bloque 1 de la Fase 4 (observación, #45) es **acción del
> propietario**, no ingeniería: exige contactar con un tercero y/o gastar dinero. La **raíz es el
> bróker (#59)**; de él cuelgan **#87** (corte de financiación), **#62** (medición del *slippage*)
> y **#107** (intradía y `bid`/`ask` reales del `SPX500:CFD`). Una sola consulta bien hecha los
> alimenta a la vez: de ahí el cuestionario único de §2.

---

## 1. Las cinco decisiones, su dueño y dónde se anotan

| Issue | Decisión / dato | Naturaleza | Se anota en |
|---|---|---|---|
| **#59** | **Bróker definitivo** (nombre, tipo de cuenta, tabla de costes vigente, fecha) | Decisión del propietario (raíz) | `tech_stack.md` §11 bis (decisión 4 → CERRADA) · `plan.md` §3.3 si la tabla cambia |
| **#87** | **Hora exacta del corte de financiación** (*swap*) y su equivalente ET | Dato del bróker | `config/cost_observations.yaml` → `financing_cut` (ISO-8601 con zona) |
| **#62** | Medir el ***slippage* real** (10–15 ejecuciones en la apertura) y el spread real | Medición con dinero real | `config/cost_observations.yaml` → `executions`, `spread_observations`, `tracking_pairs` |
| **#107** | **Adquirir** el intradía y el `bid`/`ask` reales del `SPX500:CFD` (export del bróker o proveedor) | Decisión + coste | `_docs/data_sources.md` · `config/data_sources.yaml` (`unavailable:`) |
| **#51** | Plan B de fuentes gratuitas (Yahoo/Stooq): fuente diaria **y** intradía ejecutables | Decisión + posible adaptador | `_docs/data_sources.md` · `config/data_sources.yaml` |

> La **regla que manda**: se decide, se anota en el documento que la declara abierta **y entonces**
> se implementa (`tech_stack.md` §11 bis). Ninguna de estas se resuelve «sobre la marcha».

## 1 bis. El bróker declarado: Revolut (confirmado por el KID)

El propietario declara el **2026-10-07** que su bróker es **Revolut** y aporta el **Documento de
Datos Fundamentales (KID)** de su CFD de índice. Con el KID, la decisión 4 queda **confirmada**:

- **Entidad**: **Revolut Securities Europe UAB** (Lituania, nº 305799582), autorizada por el **Banco
  de Lituania** (agencia de corredores categoría A nº 6) → **EEE**; cuenta de **CFD**.
- **Divisa de liquidación**: **USD**.
- **Tabla de costes (KID, actualizado 2026-04-28)**: diferencial **0,0042 %** (0,42 $ sobre
  10.000 $), cambio de divisa **0 %**, **coste diario de tenencia** **−0,0018 % (corto) / +0,0182 %
  (largo)** por noche → **coincide exactamente** con la tabla declarada en `plan.md` §3.3.
- **Margen requerido: 5,0 %** (apalancamiento ≈ 20:1); el CFD puede cerrarse solo si las pérdidas
  agotan el margen.

**Dónde está cada respuesta en Revolut:**

| Pregunta | Dónde buscarla |
|---|---|
| 1–2 · instrumento, cuenta, divisa | Pantalla del instrumento en la app + **KID** (*Key Information Document*, PRIIPs) |
| 6 · spread por tramo | Spread **en vivo** en la pantalla del instrumento (anotar la hora ET) |
| 7 · comisión/spread mínimo | Documento de **costes y cargos** de Revolut |
| 9 · carry largo/corto | **KID** y la app (tipo *swap* por noche) |
| **10 ⭐ · corte de financiación** | Términos de CFDs / pantalla del instrumento: **a qué hora se aplica el *swap*** |
| 12–13 · export intradía `bid`/`ask` | Ver el aviso de abajo: **Revolut no es un proveedor de datos** |

**Tabla confirmada por el KID** (consultado el **2026-10-07**; KID actualizado el **2026-04-28**):

| Punto | Respuesta (KID) | Estado |
|---|---|---|
| Bróker / entidad | **Revolut Securities Europe UAB** (EEE/Lituania) | ✅ |
| Tipo de cuenta | **CFD** | ✅ |
| Divisa de liquidación | **USD** | ✅ |
| Spread | **0,0042 %** (0,42 $ / 10.000 $) | ✅ = §3.3 |
| Cambio de divisa | **0 %** | ✅ = §3.3 |
| Financiación (*swap*) | **−0,0018 % (corto) / +0,0182 % (largo)** por noche | ✅ = §3.3 |
| Margen requerido | **5,0 %** | ✅ (KID) |
| Hora de corte de la financiación (#87) | **el KID no la da** | ⏳ **#87** |

⚠️ **El «0,25 % por noche» anotado antes NO aparece en el KID y se descarta.** El KID fija el coste
de tenencia en **−0,0018 %/+0,0182 % por noche**, que coincide con §3.3; el `0,25 %` (mín. `0,01
USD`) no cuadra con nada de este documento (¿otra tarifa o error de lectura?) y **no** se usa.

**Con esto la decisión 4 queda CERRADA** (broker, tipo de cuenta, tabla y fecha): ver el cierre en
`tech_stack.md` §11 bis. Quedan **fuera** y siguen abiertas: **#87** (la hora de corte, que el KID
no da), **#62** (medir el *slippage* y el spread en vivo) y **#107** (intradía y `bid`/`ask`).

**Dos avisos honestos:**

- **#107 (intradía + `bid`/`ask` reales).** Revolut muestra precios **en vivo** y permite
  **exportar extractos de operaciones**, pero **no** sirve un histórico de 1 min con `bid`/`ask` de
  **≥ 5 años**. Es probable que #107 se cierre por la **vía (b)** (no adquirir; la ruta 3 —proxy
  declarado— pasa a definitiva), salvo que un **proveedor de datos** aparte lo cubra. No confundir
  «tengo precios en la app» con «tengo el histórico exportable».
- **Los números no se inventan.** El spread y el *swap* válidos son **los de tu cuenta y tu región
  en la fecha de consulta**; el **KID** y la app los publican, pero hay que **copiarlos con su fecha**.


## 2. El cuestionario único al bróker

Cada pregunta está numerada para poder responderlas de una vez. Las cinco primeras **ya existen**
en `analysis/cost_audit.py` (`BROKER_QUESTIONS`) y se reproducen aquí **verbatim**; las demás
vienen de `plan.md` §8.5 y §21 y de la plantilla `config/cost_observations.yaml`.

### A. Instrumento, cuenta y divisa (#59)

1. ¿Cuál es el **nombre exacto** del instrumento y el **tipo de cuenta** (réplica de contado o de
   futuro)? *(plan.md §21 pregunta 1)*
2. ¿En qué **divisa** se denomina y se liquida el `SPX500:CFD`, y cuál es el **coste de conversión**
   si el nocional no es USD? *(plan.md §21 pregunta 1 y §3.3)*

### B. Ventana, ejecución y calendario

3. ¿Cuál es la **ventana de cotización** real del CFD en ET (apertura y cierre) y su **equivalente
   UTC**, y cómo trata las **medias sesiones**? *(plan.md §8.5)*
4. ¿a qué hora efectiva se puede ejecutar la entrada en la subasta de apertura y el cierre obligatorio en su plataforma?
5. ¿Cuál es el **umbral de margen** de cierre forzoso y puede el bróker **cerrar antes** de que
   llegue mi stop? *(plan.md §3.4)*

### C. Coste declarado: spread, comisión mínima y escala

6. ¿Cuál es el **spread** (`ask − bid`) en vivo a las **09:20 / 09:35 / 11:00 / 15:45 / 16:00 ET**?
   *(plan.md §21 pregunta 2 y §8.5)*
7. ¿aplica el bróker una comisión mínima o un spread mínimo en puntos que haga que el coste en % sea mayor con nocionales pequeños?
   → `minimum_commission_usd`. *(plan.md §21 pregunta 5)*
8. ¿cuál es el tamaño de contrato del `SPX500:CFD`, es decir, la equivalencia exacta entre puntos de índice y $ de nocional?
   *(para leer el diferencial también en puntos)*
9. Tarifas de tenencia **largo y corto** por noche: confirmar `+0,0182 %` / `−0,0018 %` sobre el
   nocional. *(plan.md §3.3)*

### D. ⭐ Corte de financiación (#87)

> Es **la verificación más crítica de la Fase 0** (`plan.md` §8.5 y §21 pregunta 3): si el corte cae
> antes del cierre de la sesión US, el intradía puro paga tenencia igualmente.

10. ¿en qué instante exacto se cobra el coste de tenencia (*swap*) de una posición en `SPX500:CFD`, en hora del bróker y su equivalente en `America/New_York`? ¿Se cobra por noche de calendario o por sesión?

### E. Anualización

11. ¿sobre qué base anualiza el documento del bróker el coste de tenencia: 365 días naturales, noches de calendario o sesiones?

### F. Datos para #107 y #62

12. ¿**exporta** el bróker el histórico intradía del propio CFD (**1 min** preferido, 5 min
    aceptable) con `timestamp`, `bid`, `ask` y `last`? ¿Cobertura (**≥ 5 años** deseable), formato,
    **zona horaria** y **licencia** de uso interno? *(pliego de #107 y `_docs/data_sources.md`)*
13. ¿Cuál es la **tarifa** de esas exportaciones o de un feed equivalente?

## 3. Dónde se anota cada respuesta (y qué cierra)

| Respuesta | Campo / documento | Cierra |
|---|---|---|
| 1, 2 | `tech_stack.md` §11 bis (decisión 4, con tabla de costes y fecha) · `plan.md` §3.3 si cambia | #59 |
| 3, 4 | `plan.md` §8.5/§21 (ventana y hora efectiva) | #8 / #10 |
| 5 | `plan.md` §3.4 (margen y cierre forzoso) | #8 |
| 6 | `config/cost_observations.yaml` → `spread_observations` | #62 |
| 7 | `config/cost_observations.yaml` → `minimum_commission_usd` | #8 / #11 |
| 8 | `config/cost_observations.yaml` → `spread_observations` (lectura en puntos) | #8 / #11 |
| 9 | `config/cost_observations.yaml` (fija el carry declarado) | #8 |
| **10** | `config/cost_observations.yaml` → **`financing_cut`** (ISO-8601 con zona) | **#87** |
| 11 | `config/cost_observations.yaml` → base de la anualización | #8 |
| 12, 13 | `_docs/data_sources.md` (pliego + decisión) · `config/data_sources.yaml` | #107 |
| — | `config/cost_observations.yaml` → `executions` (10–15, dinero real) | #62 |

> Con las respuestas 1–3 y 10 se cierran #59 y #87, y `config/cost_observations.yaml` queda listo
> para que **#62** solo tenga que añadir las `executions` de las ejecuciones reales.

## 4. Lo que NO se decide aquí

- **No se elige bróker** ni se compara tabla de costes: eso es la decisión #59.
- **No se mide el *slippage*** (#62): exige ejecuciones reales con dinero real.
- **No se compra** el intradía del CFD (#107) ni una fuente de pago (#51): es gasto del propietario.
- **No se toca** `config/cost_observations.yaml` con valores de relleno: `null` = «no medido» y
  **nunca** se sustituye por `0`.
- **No se altera** `§11.6`, el carril B ni la puerta pre-registrada de la Fase 4 (tarea #45).

## 5. Referencias

- `plan.md` **§3.3** (tabla de costes declarada), **§3.4** (riesgo y margen), **§8.5** (mediciones
  previas obligatorias) y **§21** (preguntas abiertas, 1/2/3/5).
- `tech_stack.md` **§11 bis**, decisión abierta **4** (#59).
- `src/cfdtrader/analysis/cost_audit.py` (`BROKER_QUESTIONS`, `financing_cut` `unverified`,
  `slippage_ejecucion` `unmeasured`) y `config/cost_observations.yaml` (plantilla de captura, #8).
- `_docs/data_sources.md` (§«SIN FUENTE» y pliego de adquisición, #107) y `config/data_sources.yaml`
  (`unavailable: SPX500:CFD`).
