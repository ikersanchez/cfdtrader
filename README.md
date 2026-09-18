# cfdtrader

Sistema multiagente de apoyo a la decisión para operar **intradía un CFD sobre el S&P 500**, con recomendación diaria `LONG` / `SHORT` / `NOTHING` y ejecución **siempre manual**.

> ⚠️ **Este repositorio es un proyecto de ingeniería, no asesoramiento financiero.** Los CFDs son productos apalancados de alto riesgo y la mayoría de las cuentas minoristas pierde dinero operando con ellos. Nada de lo que hay aquí debe interpretarse como una recomendación de inversión.

**Estado actual: especificación completa y repositorio Python instalable (tarea #1 hecha).** El proyecto está en Fase 0 (medición de viabilidad): todavía no hay datos ni mediciones.

---

## Documentos

Todos los documentos viven en `_docs/`.

| Documento | Qué contiene |
|---|---|
| [`plan.md`](_docs/plan.md) | **Fuente de verdad funcional.** Qué se construye: definición del problema, arquitectura, agentes, riesgo, protocolo de evaluación, roadmap. |
| [`tech_stack.md`](_docs/tech_stack.md) | **Especificación técnica.** Con qué se construye: stack por capa, licencias, control de coste del LLM, operación en local y modelo de persistencia. Especificación **cerrada** (v2.0). |
| [`tasks.md`](_docs/tasks.md) | **Backlog.** 46 tareas de una sesión cada una, agrupadas en 6 fases, con puertas de salida. |

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

## Desarrollo

Entorno gestionado con `uv` (Python 3.12). `uv.lock` está commiteado: es la garantía de reproducibilidad.

```bash
uv sync                                  # instala el entorno (stack mínimo viable, Fase 0–2)
uv run pre-commit install                # hooks locales: ruff + detección de secretos
uv run pytest                            # suite completa
uv run ruff check . && uv run ruff format --check .
uv run pyright                           # tipado estático en modo estricto
```

Las dependencias se declaran en `pyproject.toml` y **no se añade ninguna sin justificarla**: `tech_stack.md` §5.2 prohíbe instalar el stack completo antes de que la fase que lo necesita lo pida. Secretos: copiar `.env.example` a `.env` (ignorado por git) y `chmod 600 .env`.

---

## Puesta en marcha del repositorio remoto

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

Sin licencia por el momento: **todos los derechos reservados**. Solo hay esqueleto de código (tarea #1); se decidirá licencia cuando haya código con entidad suficiente.
