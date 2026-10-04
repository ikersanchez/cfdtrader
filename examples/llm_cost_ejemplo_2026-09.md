# Coste del LLM: 2026-09 (tarea #117)

- **Llamadas:** 3 (2 reales, 1 de cache)
- **Aciertos de cache:** 0.333333
- **Tokens:** 1424 de entrada, 2950 de salida (1 filas sin tokens: aciertos, #119)
- **Tarifa verificada el:** 2026-10-03

| Cifra | Valor | Estado |
|---|---:|---|
| Gasto del mes | 0.000563 | measured |
| Coste por titular procesado | 0.000031 | measured |
| Gasto que la deduplicacion evito | 0.000594 | **estimacion** |

> **Gasto incompleto:** hay llamadas sin tarifa declarada: el total es un minimo conocido, no el gasto
> **titulares_duplicados x coste medio por titular enviado** sobre 19 titulares duplicados. es un contrafactual, no una medicion: la palanca 3 existe justamente para no pagar esos titulares, asi que el gasto evitado no aparece en ninguna fila

- Titulares leidos: 48 | enviados: 18 | sesiones con conteo: 1 | sin conteo: 0
- Tope mensual declarado: None | superado: False
- Fuente de llamadas: `journal/ops/llm_calls`
- Fuente del conteo: `journal/ops/<sesion>/manifest.json`

> ⚠️ el numerador (ops.llm_calls) se conserva 18 meses; el denominador (el manifest) solo 90 dias (#44): las cifras por titular alcanzan al ultimo trimestre y se declaran como tales

> Comando **manual**: no hay *scheduler* ni servicio que lo dispare (`tech_stack.md` §4.11).
