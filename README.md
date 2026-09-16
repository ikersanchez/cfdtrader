# cfdtrader

Sistema multiagente de apoyo a la decisión para operar **intradía un CFD sobre el S&P 500**, con recomendación diaria `LONG` / `SHORT` / `NOTHING` y ejecución **siempre manual**.

> ⚠️ **Este repositorio es un proyecto de ingeniería, no asesoramiento financiero.** Los CFDs son productos apalancados de alto riesgo y la mayoría de las cuentas minoristas pierde dinero operando con ellos. Nada de lo que hay aquí debe interpretarse como una recomendación de inversión.

**Estado actual: especificación completa, sin código.** El proyecto está en Fase 0 (medición de viabilidad).

---

## Documentos

| Documento | Qué contiene |
|---|---|
| [`plan.md`](plan.md) | **Fuente de verdad funcional.** Qué se construye: definición del problema, arquitectura, agentes, riesgo, protocolo de evaluación, roadmap. |
| [`tech_stack.md`](tech_stack.md) | **Especificación técnica.** Con qué se construye: stack por capa, licencias, control de coste del LLM, operación en local y modelo de persistencia. Especificación **cerrada** (v2.0). |
| [`tasks.md`](tasks.md) | **Backlog.** 48 tareas de una sesión cada una, agrupadas en 6 fases, con puertas de salida. |

### Cómo se relacionan

```
plan.md          el QUÉ            (comportamiento)
tech_stack.md    el CON QUÉ        (herramientas)
tasks.md         el EN QUÉ ORDEN    (ejecución)
```

Ante conflicto: `plan.md` manda sobre el comportamiento y `tech_stack.md` sobre la herramienta.

---

## Las tres ideas que definen el diseño

**1. El LLM no decide ni calcula.** Extrae eventos de noticias, puede vetar y redacta el informe. La probabilidad y la decisión salen de una **función pura y determinista** en `decision/gate.py`. Un componente cuya versión cambia sin control no puede ser la fuente del edge — y tampoco puede retrotestearse.

**2. El backtest se construye antes que los agentes.** Con purga y embargo, costes reales y baselines triviales que hay que batir. La parte de evaluación no es un accesorio: es lo que decide si el resto tiene sentido.

**3. El sistema debe poder decir "no operar".** El valor está en el *no-trade*, la reducción de costes y la gestión de riesgo, no en predecir el mercado. Se espera operar un 10–30 % de las sesiones como máximo.

---

## Puesta en marcha

El repositorio remoto y las issues se crean con un único script:

```bash
gh auth login            # una vez
python3 setup_github.py --dry-run   # ver qué haría, sin tocar nada
python3 setup_github.py             # crear repo, labels, milestones e issues
```

El script es **idempotente** y lee `tasks.md` en cada ejecución, así que las issues siempre reflejan el backlog actual. `setup_github.py` es una herramienta de arranque de un solo uso: puede borrarse cuando el repositorio esté montado.

---

## Principios de trabajo

- **Ninguna tarea se cierra sin un artefacto verificable.** "Avanzar en X" no es una tarea.
- **Nada llega a producción sin backtest walk-forward con purga y embargo**, y sin batir a los baselines triviales.
- **Toda variante probada se registra.** El sobreajuste se corrige, no se ignora.
- **Point-in-time o nada.** Ninguna feature puede usar información publicada después del instante de decisión.
- **Sin datos ni secretos en el repositorio.** `data/` y `.env` están excluidos por `.gitignore`.

---

## Licencia

Sin licencia por el momento: **todos los derechos reservados**. Al no existir código propio todavía, no hay nada que licenciar; se decidirá cuando lo haya.
