# Ejemplo de informe diario — sesión 2026-09-17 (tarea #37)

Informe **real** del camino diario para una sesión pasada, con el modelo de mayor calidad
redactando la narrativa y el contra-argumento (§4.9: el barato extrae, el de calidad redacta).

- **Comando:** `uv run python -m cfdtrader.delivery.run_daily --as-of 2026-09-17T12:00:00+00:00
  --variant-id baseline_logit_elasticnet_v1 --journal-root <tmp> --git-commit <sha>
  --data-root data --runs-root runs`
- **Datos:** el almacén real del repositorio (`data/`), snapshot del 2026-09-16.
- **Redacción:** `report_agent` (#37), plantilla `agents/prompts/daily_report.j2`. El proveedor
  devolvió `deepseek-flash` como modelo efectivo y el `prompt_hash` que acompaña al informe.
- **Vereducto del día:** `NOTHING` por `tier_no_autorizado` y `ev_neto_no_calculable`. El LLM
  **no** cambió la dirección ni los números: solo redactó y argumentó el caso contrario.

```text
estado: recommendation
sesion: 2026-09-17
as_of: 2026-09-17T12:00:00+00:00
snapshot_sesion: 2026-09-16
modelo: runs/1c70272a3846005af292b0b7fead5e55aa603383586915bc68193211690ecd18/model.json (variant_id: baseline_logit_elasticnet_v1)
direccion: NOTHING
prob_calibrada: 0.5053849302217571
ev_declarado_pct: 0.2719025909318174818498188105
ev_neto_pct: null
stop_pct: 0.5463213768773573
objetivo_pct: 1.0926427537547146
tier: C
gate_sha256: sha256:474e21cad3c3939345fc2fd3538b8a65eeb36af0a66123615294845c87b8208c
bloqueo: 9:ev_neto_no_calculable
bloqueo: 10:tier_no_autorizado
redaccion: Sesión sin dirección operativa: el sistema marca `nothing`. La probabilidad calibrada se queda en 0,505, es decir, prácticamente una moneda al aire, sin ventaja estadística que justifique exponer capital. El beneficio neto esperado no es calculable (`ev_neto_no_calculable`), y sin esa cifra no hay forma de comparar el riesgo asumido con la recompensa. A esto se suma que el tier asignado es C, un nivel no autorizado para operar por las reglas del sistema (`tier_no_autorizado`), lo que cierra la puerta a cualquier entrada independientemente del resto de métricas. El movimiento esperado de la sesión es de un 0,546%, con un stop del mismo tamaño y un objetivo del 1,093% (una relación 2:1 teórica), pero sobre un rango tan estrecho el coste de 0,0042% no compensa la incertidumbre direccional. La agenda viene limpia: sin publicaciones macro ni resultados de mega-caps que actúen como catalizador. Con ese panorama, la decisión sensata es mantenerse fuera y esperar una configuración con probabilidad calibrada fuera de la zona neutra y con tier habilitado.
contra_argumento: El movimiento esperado del 0,546% y el objetivo del 1,093% dibujan una relación 2:1 favorable en teoría, y el coste de 0,0042% es lo bastante bajo como para no comerse el recorrido si el precio se mueve a favor.
contra_argumento: La probabilidad calibrada de 0,505 está ligeramente por encima del 50%, lo que deja la balanza apenas inclinada hacia el lado comprador.
contra_argumento: No hay publicaciones macro ni resultados de mega-caps en el calendario, así que no hay un evento conocido que pueda romper la estructura de forma brusca.
contra_argumento: El stop se sitúa al mismo nivel que el movimiento esperado, lo que en un escenario de continuidad alcista permitiría capturar el recorrido completo sin sobreexponerse.
contra_argumento: Aun así, esta lectura no es una tesis operativa: el propio sistema no autoriza la entrada por tier C y por EV no calculable, por lo que quedarse fuera hoy es la decisión que respeta las reglas.
contra_argumento: Una probabilidad calibrada de 0,505 es indistinguible de una moneda al aire: no hay edge demostrado ni margen estadístico que justifique asumir riesgo direccional hoy.
contra_argumento: El beneficio neto esperado no es calculable (`ev_neto_no_calculable`), lo que impide verificar que la operación tenga sentido económico después de costes.
contra_argumento: El tier C está marcado como no autorizado (`tier_no_autorizado`), de modo que cualquier entrada vulneraría las propias reglas del sistema.
contra_argumento: Un movimiento esperado del 0,546% es un rango estrecho: basta un giro adverso dentro de la sesión para tocar el stop antes de que el objetivo del 1,093% tenga ocasión de desarrollarse.
contra_argumento: Sin catalizadores macro ni resultados de mega-caps, falta el combustible que suele empujar al índice fuera de rangos comprimidos, lo que favorece el ruido y las falsas rupturas.
contra_argumento: Con estos mimbres, operar sería apostar por una dirección con información insuficiente; no operar es la vía que preserva capital para una configuración con probabilidad y tier habilitados.
redaccion_modelo: deepseek-flash | prompt: sha256:37f6eeca1c3a7c69f26436cd47d51380c36eac97348a5e6bb82a64f835d3ed3f

no hay edge demostrado
ejecucion: manual (el sistema no coloca ordenes; las decide el operador)
naturaleza: apoyo a la decision, no una estrategia validada
escenario declarado: S1 (scenario_parameters); #59 y #60 siguen OPEN
```
