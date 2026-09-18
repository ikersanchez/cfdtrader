# Especificación del tech stack — Sistema de recomendación intradía sobre CFD del S&P 500

| Campo | Valor |
|---|---|
| **Versión** | **2.3** · instrumento S&P 500 |
| **Fecha** | 2026-09-18 |
| **Estado** | ✅ **Especificación cerrada.** Cambios posteriores solo mediante entrada en el registro y motivo medido |
| **Documento padre** | `plan.md` v2.3 (fuente de verdad funcional) |
| **Ámbito** | Stack técnico del sistema descrito en `plan.md` |
| **Instrumento** | **SPX500:CFD**, horario definido en `America/New_York` y presentado en `Europe/Madrid` |
| **Restricciones rectoras** | Ejecución en **PC propio** · prioridad a **software libre y open source** · coste marginal objetivo **≈0 €** |
| **Excepción conocida** | La capa LLM usa **API externa (OpenAI / DeepSeek)**. Es la única pieza no local y el único coste recurrente del proyecto |
| **Huella en disco** | **~4–8 GB** en total, de los cuales el grueso es el entorno Python, no los datos |
| **Núcleo histórico persistido** | **< 100 MB por década** (§12.3) |
| **Público** | Yo, dentro de 3–6 meses, cuando ya no recuerde por qué elegí cada cosa |

> **Aviso.** Documento de especificación técnica, no de asesoramiento financiero. Las licencias indicadas corresponden a información pública disponible en la fecha del documento: **verificar antes de redistribuir** el software o los datos.

---

## 0. Cómo usar este documento

- **`plan.md` dice QUÉ hay que construir. Este documento dice CON QUÉ.**
- Ante conflicto, `plan.md` manda sobre el comportamiento y este documento sobre la herramienta.
- Toda sustitución de una pieza del stack se anota aquí, en el registro de cambios, con fecha y motivo.
- Las celdas **Alternativa** no son relleno: son el plan B cuando una librería se abandone, cambie de licencia o se rompa.
- **Secciones normativas** (no orientativas, se implementan tal cual): **§3** (licencias y dependencias), **§4.9** (capa LLM), **§6.3** (control de coste), **§8.4** (guardia de obsolescencia) y **§12** (modelo de persistencia).

---

## 1. Principios y restricciones

| # | Principio | Consecuencia práctica |
|---|---|---|
| 1 | **Ejecución en un PC propio** | Todo el cómputo, los datos y las decisiones viven en tu máquina. **Única excepción: la capa LLM (API OpenAI/DeepSeek)** |
| 2 | **Open source y gratuito por defecto** | Toda pieza nueva debe justificar por qué no puede ser OSS. Las licencias permisivas (MIT/BSD/Apache) se prefieren sobre copyleft fuerte (GPL/AGPL). **El LLM por API es la excepción aceptada y documentada** |
| 3 | **Coste marginal ≈0 €** | Todo es gratuito salvo el LLM por API, cuyo coste esperado es de **céntimos a pocos euros al mes** (§6.3). Debe estar acotado, medido y con tope duro |
| 4 | **Cero infraestructura** | Sin servidores, sin colas, sin contenedores, sin orquestadores distribuidos |
| 5 | **Reproducibilidad por encima de comodidad** | Todo determinista, versionado y reconstruible desde los datos crudos |
| 6 | **Minimizar dependencias** | Cada librería es un pasivo de mantenimiento a 5 años. Si son 40 líneas propias, escríbelas |
| 7 | **El núcleo no depende de la periferia** | El backtest no importa LangGraph, ni el cliente LLM, ni la capa de informe. **El LLM nunca está en el camino crítico del backtest** |
| 8 | **Degradación grácil** | Si falla una API —**incluida la del LLM**— el pipeline continúa sin overlay o devuelve `NOTHING`. Nunca se bloquea por un tercero |
| 9 | **Toda dependencia externa, abstraída** | El proveedor de LLM va detrás de una interfaz propia: cambiar de OpenAI a DeepSeek (o a un modelo local) no debe tocar el resto del sistema |
| 10 | **Sobrevivir al abandono de librerías** | Formatos abiertos y estables (Parquet, JSON, CSV, SQL). Nunca un formato propietario como almacén primario |
| 11 | **Nunca emitir una recomendación con datos que no son de hoy** | Guardia de obsolescencia obligatoria (§8.4). Un dato viejo produce una recomendación peligrosa |

**La pregunta que hay que hacerse antes de añadir cualquier dependencia:**

> ¿Esto seguirá existiendo y funcionando dentro de 5 años, y sabré arreglarlo si se rompe?

Si la respuesta es no, la pieza va detrás de un **adaptador** propio, para poder sustituirla sin tocar el resto.

---

## 2. Hardware y sistema operativo del PC local

### 2.1 Dimensionamiento

| Recurso | Mínimo | Recomendado | Nota |
|---|---|---|---|
| CPU | 4 núcleos modernos | **8 núcleos** (Ryzen 5/7 o i5/i7 recientes) | Polars y LightGBM paralelizan bien |
| RAM | **8 GB** | **16 GB** | ⭐ Al usar el LLM por API **no necesitas RAM para modelos**. 16 GB es holgadísimo |
| Almacenamiento | **10 GB libres** | **20 GB libres** | ⭐ Los datos son minúsculos y no se almacenan modelos LLM. Ver desglose abajo |
| GPU | **no necesaria** | **no necesaria** | ⭐ Con LLM por API, una GPU no aporta nada a este proyecto |
| Red | Cualquiera | — | Necesaria en la ingesta y en la llamada al LLM |

**Desglose real de consumo de disco (esto es lo que hay que tener claro):**

| Componente | Volumen | Tamaño estimado |
|---|---|---|
| OHLCV diario, 10 años, ~25 tickers | ~62.500 filas | **~1–2 MB** |
| Barras de 5 min del índice, 10 años | ~212.000 filas | **~4 MB** |
| Barras de 5 min de ETFs sectoriales y contexto, 10 años | ~3,4 M filas | **~70 MB** |
| Series macro (FRED, ECB, Eurostat) | ~100.000 filas | **~1–5 MB** |
| ETFs sectoriales (11 series, 10 años) | ~27.500 filas | **~1 MB** |
| VIX y derivados de volatilidad | ~2.500 filas | < 1 MB |
| Calendario de resultados de mega-caps | ~5.000 eventos | < 1 MB |
| Noticias — solo eventos extraídos + URL + hash | ~250.000 eventos | **~50 MB** |
| Noticias — texto crudo de resumen | ~250.000 | **~250 MB** |
| Noticias — texto completo de artículo *(opcional)* | ~250.000 | **~2 GB** |
| **Caché de respuestas del LLM** (`diskcache`) | — | ⚠️ **~150 MB/año → ~750 MB en 5 años.** Requiere política de poda |
| Runs y experimentos (~200) | — | **~500 MB** |
| Decision journal | ~1.250 registros | **~3 MB** |
| **Entorno Python + caché de `uv`** | — | ⚠️ **~2–4 GB** ← **el mayor consumidor de todos** |
| Repositorio git + modelos ML | — | ~100 MB (y crece si se commitean binarios) |
| Backups locales del raw (×2) | — | ~0,5–4 GB según política de noticias |
| *Spill* temporal de DuckDB | — | reservar **2–5 GB libres** |
| **TOTAL REALISTA** | | **~4–8 GB** |

**Conclusión sobre el disco — el insight que importa:**

> En este proyecto **el entorno de desarrollo ocupa del orden de 1.000 veces más que los datos de mercado**. Una década de OHLCV diario son ~2 MB; solo `pyarrow` + `scipy` + `scikit-learn` + `marimo` ya superan 1 GB. **El disco no es un problema, y por eso no hay que dimensionar pensando en los datos.**

**Por qué 20 GB libres y no menos:** no por los datos, sino por la **holgura operativa** — el *spill* de DuckDB, el margen del sistema operativo, los backups en el mismo equipo y el escenario futuro de datos intradía de 1 minuto de varios años (~1 GB).

**Escenario futuro, si algún día se compran datos intradía finos:**

| Componente adicional | Tamaño |
|---|---|
| 1 min de los 35 valores, 10 años (~44 M filas) | ~1 GB |
| Texto completo de noticias para retrotestear la capa LLM | ~2 GB |
| **Total acumulado** | **~8–11 GB** |

Sigue cabiendo holgadamente en los 20 GB libres recomendados.

**Cómputo y memoria:** el proyecto es de **datos pequeños y cómputo ligero**. El pico realista de RAM es cargar 7–44 M filas de intradía en Polars (~0,5–1 GB) y entrenar un LightGBM sobre ~2.500 filas × ~15 features (instantáneo). Un equipo modesto sobra. No necesitas GPU ni un procesador potente porque el trabajo pesado —el modelo de lenguaje— lo hace un tercero.

### 2.2 Sistema operativo

| Opción | Valoración |
|---|---|
| **Linux (Ubuntu 24.04 LTS o Debian 12)** | ✅ **Recomendado.** Todo el tooling nativo y ninguna dependencia de plataforma |
| macOS | ✅ Válido. Sin diferencias relevantes: no hay scheduler ni servicio del sistema que migrar |
| Windows | ⚠️ Funciona, pero el tooling es incómodo. Si es tu caso, considera **WSL2** |

Como el entorno actual es Linux, **especificamos todo asumiendo Linux**.

### 2.3 Modo de ejecución

- **Ejecución manual y a demanda**, no servicio permanente. **No hay scheduler, ni timer, ni ningún disparo automático**: lanzas el pipeline a mano cuando quieres, normalmente dentro de la ventana que describe `plan.md` §13 (ingesta hacia las **08:00 ET**, informe hacia las **09:00 ET**). El registro de cierre es una invocación aparte y más ligera; Madrid solo se usa para presentar horarios.
- **Sin notificaciones ni canales externos.** El sistema **no envía nada a ningún sitio**: ni Telegram, ni correo, ni *push*, ni webhooks. El informe se lee en la terminal o en el fichero. No hay ningún secreto de notificación que custodiar.
- **Nada corriendo 24/7.** Sin daemons, sin servidores web, sin workers a la escucha, sin timers. Si el PC está apagado, simplemente no se ejecuta nada.
- **Modo offline parcial:** todo el cálculo de features, el backtest, el entrenamiento y el gate de decisión funcionan **sin conexión**. Solo requieren red la ingesta de datos y la **capa LLM (API)**.
- **Consecuencia de diseño:** el sistema debe poder ejecutarse con la capa LLM desactivada y producir una recomendación válida. El overlay de noticias **mejora** la decisión; no la habilita. Esto es lo que permite retrotestear el núcleo sin el LLM (`plan.md` §6.2).

---

## 3. Política de licencias y dependencias

### 3.1 Criterio de licencias

| Prioridad | Licencias | Uso |
|---|---|---|
| 1 | **MIT, BSD-2/3, Apache-2.0, ISC** | Preferidas. Sin obligaciones de copyleft. Cubre **todo el stack salvo la única excepción declarada**: el servicio LLM por API |
| 2 | MPL-2.0, LGPL-3.0 | Aceptables. LGPL obliga a permitir sustituir la librería si distribuyes binarios: **solo importa si distribuyes**, no en uso personal |
| 3 | GPL-3.0, AGPL-3.0 | Evitar en el núcleo. AGPL es problemática si algún día expones el sistema como servicio |
| — | Propietarias / *source-available* | Solo si no hay alternativa OSS razonable, y siempre fuera del camino crítico |

> **Nota clave para este proyecto:** todo el stack es de **uso personal**, no se distribuye. Eso relaja casi todas las obligaciones de licencia. Aun así, se prefiere licencias permisivas para no cerrar la puerta a publicar el proyecto en el futuro.

### 3.2 La capa LLM por API: licencias, datos y reproducibilidad

Al usar un LLM por API, el software de inferencia deja de ser tu problema y aparecen tres consideraciones que hay que tener por escrito:

**a) Licencia y modelo.** No consumes software, consumes un **servicio**. No hay licencia OSS que cumplir, pero sí términos de servicio que revisar. A cambio, no cargas con la obligación de verificar licencias de *weights* ni de mantener un servidor de inferencia.

**b) Tratamiento de datos.** Los titulares que envíes salen de tu PC y llegan a un tercero. Consecuencias:

- **Nunca enviar**: claves, datos personales, tu posición abierta, tu capital, ni el resultado del meta-learner.
- **Sí enviar**: únicamente titulares públicos y metadatos de calendario.
- Revisar la política de retención del proveedor y **desactivar el entrenamiento con tus datos** si la consola lo permite.
- En la UE, el RGPD aplica al proveedor como encargado del tratamiento: verificar dónde se procesan los datos.

**c) Reproducibilidad — el punto crítico.** Esto es un **downgrade real** respecto a un modelo local, y hay que mitigarlo explícitamente:

| Problema | Mitigación |
|---|---|
| El proveedor **actualiza el modelo** bajo el mismo nombre | Registrar en cada llamada: `model`, `system_fingerprint` (si el proveedor lo expone), `created`, `prompt_hash` y `seed`. Si el fingerprint cambia, la `prompt_version` cambia y **toda comparación histórica queda invalidada** |
| El proveedor **retira** un modelo | Fijar un identificador de modelo **con versión concreta**, nunca un alias móvil. Documentar la fecha de retirada si se anuncia |
| Las salidas **no son 100 % deterministas** | `temperature=0`, `seed` cuando esté soportado (es *best effort*), **caché en disco obligatoria** y validación Pydantic estricta |
| Coste variable | Topes duros de gasto y de llamadas (§6.3) |

**d) Jurisdicción y residencia de datos — decisión pendiente, no resuelta.** El documento propone DeepSeek como proveedor principal por coste, pero hay una consideración que **no puede resolverse solo con criterio técnico**:

| Aspecto | OpenAI | DeepSeek |
|---|---|---|
| Jurisdicción | EE. UU. (con opciones de residencia en la UE en planes de pago) | China |
| Retención por defecto | Configurable; modo sin retención disponible en la API | Consultar términos vigentes |
| Uso para entrenamiento | Desactivable en la API | Consultar términos vigentes |
| RGPD | Ofrece DPA y *SCCs* | **Verificar** si ofrece DPA y garantías equivalentes |

**Consecuencia:** enviar titulares públicos reduce mucho el riesgo (no hay datos personales), pero **no lo elimina** si alguna noticia contiene nombres o si el *payload* incluye metadatos no previstos. Antes de fijar el proveedor principal hay que **leer los términos de servicio, la política de retención y el régimen de tratamiento de datos de cada uno**, y decidir con esa información delante, no solo con la tarifa.

> **Recomendación:** mientras no se hayan leído y comparado los términos, trata este punto como **decisión abierta** (§11 bis). Si la privacidad se convierte en prioridad, el plan B local (Ollama) deja de ser contingencia y pasa a ser la opción principal.

> **Consecuencia arquitectónica, importante:** dado que el LLM **no es reproducible a largo plazo**, **nunca puede estar en el camino crítico del backtest**. Por eso el `plan.md` §6.2 le da derecho a veto y a ±10 puntos porcentuales, pero no a generar alpha. Un componente cuya versión cambia sin tu control no puede ser la fuente del edge.

### 3.3 Higiene de dependencias

1. **Pin exacto** de todo en `uv.lock`, que se **commitea**. Rangos conservadores en `pyproject.toml`.
2. **El núcleo de backtest solo depende de `numpy`, `polars` y `scikit-learn`.** Si algo se rompe en el ecosistema en 2030, el backtest sigue corriendo.
3. **Adaptadores para todo lo frágil.** `yfinance` detrás de una interfaz propia con validación y *fallback* a Stooq.
4. **Un cliente HTTP, un cliente LLM, un logger.** No dos librerías para lo mismo.
5. **Sin dependencias de interfaz en el núcleo.** `features/` y `backtest/` no importan LangGraph, ni el cliente LLM, ni la capa de informe.
6. **El cliente LLM va detrás de una interfaz propia.** Un protocolo interno (p. ej. `LLMClient`) con una implementación por proveedor. Cambiar de OpenAI a DeepSeek o a un modelo local no debe tocar `agents/` ni `orchestration/`.
7. **Revisar el árbol transitivo** antes de añadir algo pesado. `langgraph` arrastra bastante: queda confinado a `orchestration/`.
8. **`pip-audit` o `uv pip audit`** en el CI para vulnerabilidades conocidas.

---

## 4. Stack por capa

### 4.1 Entorno, tooling y calidad de código

| Propósito | Elección | Licencia | Alternativa | Nota |
|---|---|---|---|---|
| Versión de Python | **3.12** | PSF | 3.13 (verificar `arch`, `lightgbm`) | Compatibilidad del ecosistema científico |
| Gestor de entorno y dependencias | **uv** | MIT / Apache-2.0 | poetry, pip-tools | Resolución y lock muy rápidos; `uv run` sin activar entorno |
| Lockfile | `uv.lock` commiteado | — | `poetry.lock` | Reproducibilidad real |
| Lint + format | **ruff** | MIT | flake8 + black + isort | Una herramienta reemplaza tres |
| Tipado estático | **pyright** | MIT | mypy | Modo estricto. Con contratos Pydantic, es la mejor red de seguridad |
| Pre-commit | **pre-commit** | MIT | — | Ruff + detección de secretos + validación YAML |
| Auditoría de dependencias | **pip-audit** | Apache-2.0 | `uv pip audit` | En CI |
| Editor | **VS Code + Pylance** | MIT (VS Code: MIT con binarios propietarios) | — | Ya en uso |

### 4.2 Configuración y secretos

| Propósito | Elección | Licencia | Alternativa | Nota |
|---|---|---|---|---|
| Config tipada | **pydantic-settings** | MIT | dynaconf | Un error de config debe fallar al arrancar, no a mitad del pipeline |
| Config declarativa | **YAML** (`config/*.yaml`) validado con Pydantic | — | TOML | Legible y comentable. Nunca lógica en la config |
| Secretos (desarrollo local) | **`.env`** ignorado por git, permisos `600` | — | — | Ahora **más importante**: contiene la clave del proveedor LLM con valor económico |
| Secretos (endurecido) | **`pass`** o **SOPS + age** | GPL-2 / MPL-2 | keyring | Opcional. Cifra secretos en disco |
| Plantilla de secretos | `.env.example` commiteado | — | — | Documenta qué claves hacen falta sin exponerlas |

**Claves necesarias (previsto):**

| Variable | Origen | Obligatoria |
|---|---|---|
| `LLM_PROVIDER` | `openai` \| `deepseek` | Sí |
| `LLM_API_KEY` | Consola del proveedor elegido | Sí |
| `LLM_MODEL_EXTRACT` | Modelo barato para extracción de eventos | Sí |
| `LLM_MODEL_REPORT` | Modelo para redactar el informe (puede ser el mismo) | Sí |
| `LLM_BASE_URL` | *Endpoint* del proveedor (los dos son compatibles con el SDK de OpenAI) | Sí |
| `LLM_DAILY_BUDGET_EUR` | Tope duro de gasto diario (§6.3) | Sí |
| `FRED_API_KEY` | FRED, tier gratuito | Opcional |
| `MARKETAUX_API_KEY` / `FINNHUB_API_KEY` | Solo si GDELT + RSS no bastan | Opcional |

**Reglas:**

- La clave del LLM es un **secreto con coste asociado**: rota periódicamente y **nunca** la subas al repositorio ni la pegues en un cuaderno.
- Configura **límites de gasto en la consola del proveedor**, no solo en tu código. La defensa debe estar en ambos lados.
- El `.env` entra en el backup **cifrado**, nunca en el backup normal.

### 4.3 Almacenamiento

| Propósito | Elección | Licencia | Alternativa | Nota |
|---|---|---|---|---|
| **Motor analítico** | **DuckDB** | MIT | PostgreSQL + TimescaleDB | ⭐ **La decisión más importante del stack.** OLAP columnar embebido, SQL completo, sin servidor, lee Parquet directamente |
| Formato de persistencia | **Parquet** (vía `pyarrow`) | Apache-2.0 | — | Formato abierto, estable, inspeccionable dentro de 10 años |
| Compresión | **zstd** | BSD-3 | snappy | Mejor ratio que snappy a velocidad similar |
| Particionado | por `source/año` | — | — | Permite reconstruir y auditar por fuente |
| Layout | `data/raw/` inmutable · `data/derived/` recalculable | — | — | La base de la reproducibilidad |
| Formatos auxiliares | **JSON / JSONL** para logs y manifests · **CSV** para exportación | — | — | Universales, sin dependencias |

**Por qué DuckDB y no PostgreSQL:**

- DuckDB es **single-writer**. Si solo hay un proceso escribiendo (tu pipeline diario), es la opción superior: cero administración, cero servidor, rendimiento analítico muy superior.
- PostgreSQL + TimescaleDB solo se justifica si necesitas **varios procesos escribiendo a la vez** (p. ej. el pipeline más un servicio de informes en vivo).
- La migración es sencilla porque **hablas SQL en ambos casos**. Por eso DuckDB es una apuesta segura aunque luego crezcas.

### 4.4 Procesamiento de datos

| Propósito | Elección | Licencia | Alternativa | Nota |
|---|---|---|---|---|
| **ETL y cálculo de features** | **Polars** | MIT | pandas | Lazy, multihilo, API expresiva, sin índices. Muy superior para ETL |
| Compatibilidad | **pandas** | BSD-3 | — | Solo donde una librería lo exija (`arch`, algunas de ML) |
| Interoperabilidad | **pyarrow** | Apache-2.0 | — | Ya viene con Polars y pandas |
| Validación de datos | **pandera** | MIT | Great Expectations | Ligero. GE es sobredimensionado para este proyecto |
| Cálculo numérico | **numpy** | BSD-3 | — | Base de todo |
| Estadística | **scipy** | BSD-3 | — | Bootstrap, contrastes, intervalos de confianza |
| Series temporales / econometría | **statsmodels** | BSD-3 | — | Para contrastes y diagnósticos. **No** para GARCH (ver 4.7) |

### 4.5 Fuentes de datos e ingesta

| Propósito | Elección | Licencia | Alternativa | Nota |
|---|---|---|---|---|
| Cliente HTTP | **httpx** | BSD-3 | requests | Un único cliente para FRED, ECB, GDELT, RSS |
| Reintentos | **tenacity** | Apache-2.0 | — | Backoff exponencial. **Imprescindible** con APIs gratuitas |
| Índice, futuros, ETFs, FX | **yfinance** | Apache-2.0 | Stooq, EODHD | `^GSPC`, `ES=F`, `SPY`, `^VIX`, ETFs sectoriales, `DX-Y.NYB`, `BZ=F`. ⚠️ Scraper no oficial: **siempre detrás de un adaptador con validación y fallback** |
| Histórico diario fiable | **Stooq** (CSV directo) | — | yfinance | Más fiable para histórico largo que Yahoo |
| **Macro US** (tipos, CPI, PCE, NFP) | **httpx directo** contra **FRED** | datos FRED: uso libre con atribución | `fredapi` | ⭐ **Fuente primaria del proyecto.** Requiere clave gratuita |
| Macro europa (contexto) | httpx contra ECB Data Portal / SDW, Eurostat | reutilización libre con atribución | `pandasdmx` | Solo contexto de la sesión europea previa. Ya no es la columna vertebral |
| Noticias (eventos macro) | **GDELT** vía `gdeltdoc` o httpx | datos GDELT: uso libre | — | La mejor fuente gratuita de eventos globales |
| Noticias (RSS) | **feedparser** | BSD-2 | — | Reuters, CNBC, MarketWatch, Investing, Seeking Alpha |
| Noticias (ampliación) | Marketaux / Finnhub / Alpha Vantage / FMP | *freemium* | — | Tiers gratuitos limitados: **solo si GDELT + RSS no bastan** |
| **Calendario de resultados** | Calendario público + `yfinance` | — | FMP | ⚠️ Crítico: una mega-cap mueve el índice más que un dato macro |
| Calendario de festivos | **`holidays`** | MIT | tabla propia | **Estados Unidos** (mercado objetivo) más España (solo tu disponibilidad) |
| Zona horaria | **`zoneinfo`** (stdlib) | PSF | pytz | Guardar en UTC; la referencia interna es `America/New_York`, `Europe/Madrid` es solo presentación |

**Riesgos de las fuentes gratuitas y su mitigación:**

| Riesgo | Mitigación |
|---|---|
| `yfinance` se rompe cuando Yahoo cambia algo | Adaptador propio + `fallback` a Stooq + alerta si el dato es *stale* |
| Límites de peticiones (*rate limits*) | `tenacity` con backoff + caché en disco de toda respuesta cruda |
| Términos de uso de Yahoo restringen la redistribución | Uso estrictamente personal; **no** publicar los datos descargados |
| Una API desaparece | El **raw inmutable** ya descargado sigue siendo tuyo y válido |
| Fuente entrega datos con retraso o revisados | Esquema *point-in-time* con `as_of` (ver `plan.md` §8.2) |

> **Sobre Stooq:** gratuito para uso personal y educativo; el uso comercial puede estar restringido. **No distribuir los datos.**

### 4.6 Features e indicadores

| Propósito | Elección | Licencia | Alternativa | Nota |
|---|---|---|---|---|
| **Indicadores técnicos** | **Implementación propia** (5–10 indicadores) | — | `pandas-ta` (MIT), TA-Lib | ⭐ Recomendado: controlas la ventana exacta y el `as_of`, evitas dependencias frágiles y el test *golden* es trivial |
| Si se prefiere una librería | **pandas-ta** | MIT | TA-Lib | ⚠️ Su mantenimiento ha sido intermitente y el repositorio original estuvo inactivo. **Refuerza la recomendación de implementar los indicadores a mano** (principio 10) |
| Volatilidad condicional | **arch** | NCSA | statsmodels | ⭐ *La* librería para GARCH / EGARCH / HAR. `statsmodels` es lento para esto |
| Volatilidad realizada (HAR) | Implementación propia sobre Polars | — | `arch` | HAR son ~20 líneas con retrasos 1d/1sem/1mes |
| *Feature store* | **Módulo propio**: Parquet + hash | — | — | ❌ **No usar Feast**: es una herramienta de equipo, no de un usuario |

**Justificación de la implementación propia de indicadores:** el `src/cfdtrader/features/` debe ser el código más **auditable** del proyecto. Una librería externa te oculta si el RSI se calcula con 14 o 15 periodos, si usa el precio de cierre o el típico, o si hay *look-ahead*. Con 5–10 indicadores, 150 líneas propias te dan control total y un test *golden* que congela el comportamiento.

### 4.7 Machine learning

| Propósito | Elección | Licencia | Alternativa | Nota |
|---|---|---|---|---|
| Modelo base | **scikit-learn** | BSD-3 | — | `LogisticRegression(elasticnet)`, `HistGradientBoostingClassifier`, `Pipeline` |
| Gradient boosting | **LightGBM** | MIT | XGBoost, CatBoost | Con `min_child_samples` alto controla mejor el sobreajuste en muestras pequeñas |
| Calibración | `CalibratedClassifierCV` | BSD-3 | propio | **Usar Platt (`sigmoid`)** si hay < 500 muestras de calibración: la isotónica sobreajusta |
| Purga y embargo | **Implementación propia** (~50 líneas) | — | `sklearn.TimeSeriesSplit` + purga propia | ❌ `mlfinlab` fue de pago y problemático. Escríbelo, documéntalo y testéalo |
| Métricas de rendimiento | **quantstats** | Apache-2.0 | `empyrical` | Solo para generar el *tear sheet*, no como motor |
| Análisis de cartera y riesgo | **skfolio** | BSD-3 | `riskfolio-lib` (BSD-3) | Sucesor libre de las funciones de cartera de mlfinlab |
| Serialización de modelos | **joblib** / **ONNX** | BSD-3 / Apache-2.0 | `pickle` | ONNX si quieres independencia del entorno |
| Tracking de experimentos | **Sistema propio**: `runs/<hash>/` con JSON + modelo | — | MLflow (Apache-2.0) | ❌ **No usar MLflow al principio.** Sin servidor, reproducible y suficiente. Añadir si llegas a cientos de experimentos |
| Versionado de datos y modelos | **DVC** (opcional) | Apache-2.0 | git-lfs | Útil si los datasets crecen. Añade complejidad: posponer |

**Descartado deliberadamente: deep learning.** Con 250–375 operaciones útiles, una red neuronal —LSTM, transformer tabular, lo que sea— es sobreajuste garantizado. El propio `plan.md` (§4.5) fija que la muestra no da para más de ~10–15 features en un modelo lineal o GBM pequeño.

### 4.8 Motor de backtest

| Propósito | Elección | Licencia | Alternativa | Nota |
|---|---|---|---|---|
| **Motor principal** | **Implementación propia** (~200 líneas) | — | — | ⭐ **Decisión deliberada, no pereza** |
| Barrido de parámetros | **vectorbt** (versión OSS) | Apache-2.0 | — | Solo para análisis de sensibilidad. ⚠️ vectorbt PRO es comercial: usar la OSS |
| Informe de rendimiento | **quantstats** | Apache-2.0 | propio | Curva de equity, Sharpe, drawdown |
| Baselines | Implementación propia | — | — | Ver `plan.md` §11.2 |

**Por qué motor propio y no un framework:**

`backtrader`, `zipline` y `backtesting.py` introducen una capa de abstracción que **oculta precisamente lo que necesitas inspeccionar**: la semántica exacta de los costes, el orden de los eventos, el tratamiento del *gap* de apertura y la aplicación de la purga. En un proyecto donde el resultado depende de detalles sutiles, **el motor debe ser corto, aburrido y legible de arriba abajo en una sentada**. 200 líneas propias valen más que 20.000 que no controlas.

### 4.9 LLM y razonamiento — **vía API (OpenAI / DeepSeek)**

**Decisión:** la capa LLM se consume **por API**, no en local. OpenAI y DeepSeek son **ambos compatibles con el SDK de `openai`**, así que un único cliente con `base_url` y `model` configurables cubre los dos proveedores.

| Propósito | Elección | Licencia | Alternativa | Nota |
|---|---|---|---|---|
| **Cliente LLM** | **SDK `openai`** con `base_url` configurable | Apache-2.0 | `httpx` directo | ⭐ Un solo cliente para OpenAI y DeepSeek: DeepSeek expone API compatible con OpenAI |
| **Abstracción de proveedor** | **Interfaz propia** (`LLMClient`) + implementación por proveedor | — | LiteLLM (MIT) | ⭐ Obligatoria (principio 9). `agents/` no debe saber qué proveedor hay detrás |
| Proveedor principal | **DeepSeek** | servicio | OpenAI | Coste por token muy inferior para extracción de texto. Compatible con el mismo SDK |
| Proveedor alternativo | **OpenAI** | servicio | Anthropic, Google, Azure | Mejor ecosistema y salidas estructuradas maduras. Útil como *fallback* |
| **Modelo de extracción** (alto volumen) | Modelo **barato** de cada proveedor | servicio | — | Extrae eventos de titulares. Es la mayor parte del gasto: aquí va el modelo económico |
| **Modelo de informe** (bajo volumen) | Modelo de **mayor calidad** | servicio | — | Una llamada al día para redactar el informe. El coste es despreciable: gasta calidad aquí |
| **Salida estructurada** | **Modo JSON del proveedor + validación Pydantic estricta + reintento** | — | `instructor` (MIT) | ⚠️ **Las capacidades difieren entre proveedores** (ver aviso abajo). Si no valida el esquema tras N reintentos, se descarta el evento |
| Multi-proveedor (opcional) | **LiteLLM** | MIT | SDK directo | Solo si se quiere enrutado automático y *fallback* transparente entre proveedores |
| **Caché de respuestas** | **diskcache** | Apache-2.0 | JSON en disco | Clave = hash(prompt + modelo + inputs). **Obligatoria**: reproducibilidad y ahorro directo |
| Deduplicación previa | Hash del titular normalizado + similitud difusa sobre el texto | — | — | No pagues dos veces por la misma noticia. **No se usan *embeddings*: ver aviso abajo** |
| Control de coste | **tiktoken** (OpenAI) + contador del proveedor | MIT | — | ⚠️ El tokenizador de DeepSeek **no es** el de OpenAI: no calcules su coste con `tiktoken` |
| Presupuesto | **Tope duro diario** + contador acumulado | — | — | Si se supera ⇒ se desactiva el overlay y el pipeline sigue (§6.3) |
| Determinismo | `temperature=0` + `seed` (*best effort*) | — | — | No garantizado. Registrar `system_fingerprint` y `model` exactos |
| *Fallback* local (contingencia) | **Ollama** | MIT | `llama.cpp` | ⭐ Ya no es la opción principal, pero **es el plan B** si el coste se dispara o la API cae |

**Estrategia de dos modelos (optimización de coste):**

```
Alto volumen, alta frecuencia  →  modelo BARATO   →  extracción de eventos de noticias
Bajo volumen, 1 vez al día     →  modelo MEJOR    →  redacción del informe y del contra-argumento
Deduplicación de titulares     →  hash normalizado + difuso  →  sin coste, sin API
```

Esta separación es la palanca de coste más importante: no tiene sentido pagar un modelo de gama alta para etiquetar 200 titulares al día, ni escatimar en la única llamada diaria que produces y lees tú.

**⚠️ Dos avisos que obligan a que la interfaz `LLMClient` sea tolerante:**

1. **La salida estructurada NO es equivalente entre proveedores.** OpenAI ofrece *strict JSON Schema* (garantía de cumplimiento del esquema). DeepSeek ofrece **modo JSON** (garantiza que la salida *es* JSON válido, pero **no** que cumpla tu esquema). Consecuencia: la interfaz `LLMClient` **no puede asumir validación por el proveedor**; debe validar con Pydantic siempre, reintentar con el error de validación en el mensaje, y descartar el evento tras N intentos. Diseñar como si ningún proveedor garantizase el esquema.
2. **Los *embeddings* no son un servicio universal.** El plan original de deduplicar noticias con *embeddings* **no es viable con DeepSeek**, que no expone un *endpoint* de *embeddings*. Y para deduplicar titulares de prensa, unos *embeddings* son sobredimensionados: **un hash del titular normalizado (minúsculas, sin acentos, sin puntuación) más similitud difusa** resuelve el 95 % de los casos sin coste ni dependencia. Si algún día se necesitan *embeddings* de verdad, usar un modelo **local** pequeño (`sentence-transformers`, Apache-2.0) antes que añadir un segundo proveedor de API.

**Ventajas de la API frente a un modelo local:**

1. **Calidad muy superior** en extracción de matices, entidades y relaciones — que es exactamente la tarea del `NewsAgent`.
2. **Hardware irrelevante**: sin GPU, sin RAM para modelos, sin 20 GB de disco ocupados.
3. **Mantenimiento cero**: no actualizas pesos, no depuras `llama.cpp`, no peleas con cuantizaciones.
4. **Latencia** irrelevante de todos modos (una ejecución al día).

**Contrapartidas honestas y su mitigación:**

| Contrapartida | Mitigación |
|---|---|
| **Coste recurrente** (el único del proyecto) | Modelo barato para el volumen, caché agresiva, deduplicación, topes duros. Coste esperado: céntimos a pocos €/mes (§6.3) |
| **Reproducibilidad degradada** | Registrar `model` + `system_fingerprint` + `seed` + `prompt_hash`. **Nunca en el camino crítico del backtest** (§3.2) |
| **Dependencia de un tercero** | El overlay es **opcional**: si la API cae, el pipeline produce recomendación sin él. `LLMClient` permite cambiar de proveedor sin tocar el resto |
| **Los datos salen del PC** | Enviar **solo titulares públicos**. Nunca posiciones, capital ni salidas del modelo (§3.2) |
| **El proveedor retira o cambia el modelo** | Fijar identificadores **con versión concreta**, nunca alias móviles. Detectar cambio de `system_fingerprint` y alertar |
| **Límites de tasa (*rate limits*)** | `tenacity` con backoff, llamadas agrupadas (*batching*) y reintento en ventana posterior |
| **El proveedor cambia precios** | El contador de gasto es propio y acumulativo: te enteras el mismo día |

**Regla de diseño que no cambia:** el LLM **extrae eventos, veta y redacta**. No calcula números y no decide la dirección. Todo lo dicho en `plan.md` §6.2 y §7.3 sigue vigente palabra por palabra. Cambiar de Ollama a API **mejora la calidad de esa función**, pero no le amplía los permisos.

**Especificación explícita que faltaba:** **el backtest y el walk-forward se ejecutan siempre con el overlay LLM deshabilitado.** No es una opción de configuración ni un modo: es la única forma correcta de retrotestear, porque no existe un archivo histórico de noticias con `published_at` fiable a coste razonable (§6.2 del `plan.md`). El overlay se evalúa **aparte**, y solo de dos maneras posibles:

| Método de evaluación del overlay | Requisito | Coste |
|---|---|---|
| *Paper trading* prospectivo | Ninguno: se registra y se compara hacia delante | 0 € |
| Backtest con archivo histórico de noticias | Contratar un archivo con timestamps | 20–50 €/mes |

**Consecuencia:** mientras no se contrate archivo histórico, la capa LLM **no tiene evidencia retrospectiva que la respalde**. Por eso el principio 7 y el `plan.md` §6.2 le niegan el derecho a generar alpha: no es prudencia abstracta, es que **no hay forma de demostrarlo** con los datos disponibles.

### 4.10 Orquestación de agentes

| Propósito | Elección | Licencia | Alternativa | Nota |
|---|---|---|---|---|
| Grafo de agentes | **LangGraph**, **versión fijada y pineada** | MIT | `pydantic-ai` (MIT), pipeline propio | Evoluciona rápido: **pin exacto obligatorio** |
| Abstracciones de alto nivel | **Evitar LangChain** más allá de lo que LangGraph arrastre | MIT | SDK directo | Cambian mucho entre versiones |
| Contratos de datos | **Pydantic** | MIT | dataclasses | Ver `plan.md` §7.2 |
| Plantillas de prompts | **Jinja2** | BSD-3 | f-strings | Versionables y con hash |

**Nota de arquitectura:** LangGraph orquesta el **pipeline** (fetch → nodos expertos en paralelo → fan-in → síntesis). El **gate de decisión** (`plan.md` §7.3, paso 4) es una **función pura de Python sin dependencias de orquestación**. Si decidir requiriese LangGraph, el backtest estaría muerto.

### 4.11 Ejecución local

| Propósito | Elección | Licencia | Alternativa | Nota |
|---|---|---|---|---|
| **Disparador** | **Tú, a mano** | — | — | 🚫 **No hay scheduler**: sin `systemd timer`, sin cron, sin APScheduler. El pipeline se ejecuta cuando tú lo pides |
| Entrada del pipeline | `uv run python -m cfdtrader.delivery.run_daily` | — | script shell envoltorio | Un solo comando, sin argumentos obligatorios |
| Prevención de solapamiento | **Ninguna** | — | `flock` | Sin disparos automáticos no hay riesgo de solapamiento. Si algún día se automatiza, `flock` vuelve a ser obligatorio |
| Logs | **`run_log` en JSONL** + salida de la terminal | — | journald | El `run_log` por ejecución (§4.13) sustituye a `journalctl`, que ya no aplica |

**Por qué no hay scheduler:** el modo de operación fijado es **ejecución manual y a demanda con supervisión humana**. Un timer solo aporta valor en un flujo desatendido, y este no lo es: tú estás delante cuando el pipeline corre y cuando se cierra la posición. Eliminarlo quita de golpe un servicio del sistema, un fichero de *lock*, la recuperación de ejecuciones perdidas por `Persistent=true` y toda la casuística de horarios que no se disparan (PC apagado, suspensión, *linger*). **Si algún día se automatiza, esta sección es la que hay que recuperar.**

> ⚠️ **Lo que NO se retira es el anclaje a `America/New_York`.** Aunque no haya timer, sigue siendo obligatorio calcular en ET/UTC y presentar en Madrid: la sesión se desplaza una hora dos veces al año porque EE. UU. y Europa cambian de hora en fechas distintas (`plan.md` §8.3). La conversión **deja de ser responsabilidad del sistema operativo y pasa a serlo del código**.

### 4.12 Informe y salida

| Propósito | Elección | Licencia | Alternativa | Nota |
|---|---|---|---|---|
| **Informe en consola** | **Rich** | MIT | salida plana | La ejecución es manual y a demanda: tú lanzas el pipeline y lees el resultado |
| Plantillas del informe | **Jinja2** | BSD-3 | f-strings | Markdown/HTML |
| **Salida persistida** | **Markdown + JSON** en `data/derived/reports/` | — | — | Es el registro del informe. **No se envía a ningún sitio** |

**Sin notificaciones ni canales externos.** El sistema **no envía nada**: ni Telegram, ni correo, ni *push*, ni webhooks, ni ningún servicio de terceros. El informe se lee en la terminal o en el fichero Markdown de `data/derived/reports/`. Coherente con el principio 1 (todo vive en tu máquina), elimina una dependencia externa, un conjunto de secretos y una superficie de fallo.

> 🚫 **Descartadas (2026-09-18): Telegram (Bot API vía `httpx`), `ntfy`, `Apprise`, `smtplib` y `python-telegram-bot`.** Se descartaron al fijar el modo de operación **manual y a demanda**: no hay nada que notificar a distancia porque el usuario está delante cuando el sistema se ejecuta. Se retiraron también la alarma de cierre de las 15:45 ET y el *heartbeat*: la obligación de cerrar a las 16:00 ET sigue vigente (`plan.md` §12, regla 16), pero **la garantiza el usuario, no el software**.

### 4.13 Observabilidad

| Propósito | Elección | Licencia | Alternativa | Nota |
|---|---|---|---|---|
| Logging | **loguru** | MIT | structlog (MIT / Apache-2.0) | loguru por simplicidad; structlog si quieres JSON estructurado consultable |
| Trazas del pipeline | **JSONL** por ejecución + `manifest.json` | — | SQLite | Con hashes, versiones y duraciones |
| Calidad de datos | **pandera** + checks propios | MIT | — | Ver `plan.md` §8.4 |
| Fallos del pipeline | Traza en el `run_log` de la ejecución + código de salida ≠ 0 | — | — | El **silencio es el peor modo de fallo**: con ejecución manual, un fallo se ve al terminar y nunca se pierde el rastro |
| Errores no capturados | **Nada**: `run_log` + `manifest.json` | — | — | 🚫 **Sentry descartado:** mandar trazas a un tercero contradice la ejecución 100 % local y el principio 1. No hay capa de alertas que mantener |
| Monitorización de recursos | `psutil` (BSD-3) | BSD-3 | — | Opcional. Se consulta a mano, no alerta |

### 4.14 Testing y CI

| Propósito | Elección | Licencia | Alternativa | Nota |
|---|---|---|---|---|
| Framework de tests | **pytest** | MIT | unittest | — |
| Cobertura | **pytest-cov** | MIT | — | Métrica orientativa, no un objetivo |
| Tests basados en propiedades | **hypothesis** | MPL-2.0 | — | Ej.: el gate nunca da EV positivo si coste > movimiento esperado |
| CI | **GitHub Actions** (tier gratuito) | *freemium* | GitLab CI, **pre-commit + hook local** | El tier gratuito sobra para este proyecto |
| Alternativa 100 % local | **pre-commit + `uv run pytest`** como *hook* de pre-push | MIT | — | Cumple los principios de coste mínimo y de no depender de terceros |

**Los cuatro tests que más valor aportan (`plan.md` §11):**

| Test | Qué protege |
|---|---|
| **No-look-ahead** | Que un feature en `t` no cambia al añadir datos de `t+1`. **Es el más valioso de todos** |
| **Golden dataset de features** | Que un cambio silencioso de cálculo no invalide los backtests |
| **Determinismo del gate** | Mismo input → mismo output, byte a byte, N veces |
| **Casos de coste calculados a mano** | Que el modelo de costes no se desvía sin que te enteres |

### 4.15 Exploración y visualización

| Propósito | Elección | Licencia | Alternativa | Nota |
|---|---|---|---|---|
| Exploración interactiva | **marimo** | Apache-2.0 | Jupyter (BSD-3) | ⭐ Reactivo y **reproducible**: sin estado oculto ni orden de ejecución frágil |
| Cuadernos clásicos | **JupyterLab** | BSD-3 | — | Válido, pero exige disciplina manual |
| Visualización de señales y equity | **plotly** | MIT | matplotlib (PSF-like) | Interactivo, funciona local |
| Gráficos estáticos en informes | **matplotlib** | PSF-like | — | Para insertar en el Markdown |
| Panel de seguimiento (Fase 4+) | **Streamlit** | Apache-2.0 | Dash (MIT) | ❌ No en Fase 1: es procrastinación elegante |

> **Regla absoluta:** **notebook = exploración, nunca lógica de producción.** Toda lógica que se use dos veces migra a `src/` con su test.

---

## 5. Stack por fases: mínimo viable vs completo

### 5.1 Stack mínimo viable (Fases 0–2) — **esto es lo que instalar el primer día**

| Capa | Mínimo imprescindible |
|---|---|
| Entorno | Python 3.12 · uv · ruff · pyright |
| HTTP | httpx · tenacity |
| Datos | yfinance · Stooq · httpx (FRED) · `holidays` |
| Almacenamiento | DuckDB · pyarrow · Parquet |
| Procesamiento | Polars · numpy · scipy |
| Validación | pydantic · pydantic-settings · pandera |
| Indicadores | Implementación propia |
| ML | scikit-learn · LightGBM · arch |
| Backtest | Motor propio · quantstats |
| Config | YAML + Pydantic |
| Logging | loguru |
| Tests | pytest |
| Ejecución | **Manual desde terminal** |
| Salida | Fichero Markdown en `data/derived/reports/` |

**Total: ~23 paquetes de primer nivel** (más sus dependencias transitivas). Todo open source, todo gratuito, todo local. Es un stack deliberadamente pequeño: cada paquete añadido antes de tiempo es deuda de mantenimiento.

### 5.2 Stack completo (Fases 3–5) — **añadir solo cuando se necesite**

| Capa | Añadido | Fase |
|---|---|---|
| Ingesta | feedparser · gdeltdoc · GDELT | 3 |
| LLM | **SDK `openai`** (con `base_url` DeepSeek u OpenAI) · `LLMClient` propio · diskcache · tiktoken · **topes de gasto** | 3 |
| Orquestación | LangGraph · Jinja2 | 3 |
| Informe | Rich · Jinja2 | 4 |
| Observabilidad | JSONL + manifest | 4 |
| Análisis | vectorbt (OSS) · skfolio | 5 |
| Visualización | Streamlit · plotly | 5 |

**Disciplina:** no instalar nada de la tabla 5.2 hasta que la fase correspondiente lo requiera. Cada paquete añadido antes de tiempo es deuda de mantenimiento y superficie de fallo.

---

## 6. Presupuesto: coste mínimo vs coste con extras

### 6.1 Ruta de coste mínimo (objetivo)

| Componente | Coste | Límite / condición |
|---|---|---|
| Todo el software | **0 €** | Open source con licencias permisivas |
| Datos diarios de mercado | **0 €** | yfinance, Stooq |
| Macro y tipos | **0 €** | FRED (fuente primaria), ECB SDW y Eurostat (contexto) |
| Noticias | **0 €** | GDELT + RSS |
| Ejecución | **0 €** | Manual desde la terminal: sin scheduler ni servicios |
| Informe | **0 €** | Rich en consola + Markdown en `data/derived/reports/` |
| CI | **0 €** | GitHub Actions tier gratuito, o *hooks* locales |
| Almacenamiento | **0 €** | Disco del PC |
| **LLM por API** | **céntimos – pocos €/mes** | ⚠️ **Único coste recurrente.** Acotado, medido y con tope duro (§6.3) |
| **TOTAL MARGINAL** | **≈0 €** | Del orden del precio de un café al mes |

### 6.2 Costes opcionales y cuándo se justifican

| Partida | Coste orientativo | ¿Cuándo merece la pena? |
|---|---|---|
| **Datos intradía históricos** | ~20 €/mes | Hay intradía gratuito reciente para algunos proxies, pero **no se da por válido para `SPX500:CFD`** hasta verificar fuente, cobertura, granularidad y bid/ask en la tarea 3. Si no cubre el histórico requerido, se compra o se bloquea la Fase 1 |
| Datos de futuros (Databento) | pago por uso | Si finalmente se opera sobre el futuro Mini y se quiere rigor en la base |
| **Noticias con histórico** | 20–50 €/mes | Necesario para **retrotestear** la capa LLM. Sin esto, la capa LLM no es backtesteable (`plan.md` §6.2) |
| API de LLM con modelos mayores | +2–10 €/mes | Solo si se demuestra con evidencia que un modelo mayor mejora el Brier score y el Sharpe OOS de forma significativa |
| **Cambio de bróker** | — | ⭐ **No es un coste de stack, pero es la inversión con mejor retorno de todo el proyecto** (`plan.md` §4.4) |

**Orden de prioridad si algún día se decide gastar:**

1. **Reducir el coste de operar** (bróker con spread menor) — mejora el EV de cada operación, siempre.
2. **Datos intradía históricos** — sin ellos, el backtest intradía es ciego.
3. **Histórico de noticias** — solo si la vía LLM demuestra aportar valor.
4. **Nada más.**

### 6.3 Control de coste de la API del LLM (sección normativa)

El LLM es el **único componente con coste marginal por ejecución**. Por eso lleva controles explícitos, no basta con "ya lo miraremos":

**1. Estimación del gasto esperado**

| Concepto | Orden de magnitud diario |
|---|---|
| Titulares a procesar | ~50–200, agrupados en *lotes* |
| Llamadas de extracción | ~5–10 (batching, no una por titular) |
| Tokens de entrada (extracción) | ~30.000–60.000 |
| Tokens de salida (extracción) | ~5.000–10.000 |
| Llamadas de informe | 1 |
| **Gasto mensual estimado** (22 sesiones, modelo económico) | **del orden de céntimos a 2 €** |

⚠️ Las tarifas por token varían y los proveedores las cambian: **verificar precios vigentes y recalcular**. Los proveedores con *caché de contexto* (descuento sobre tokens de entrada ya vistos) reducen el coste de forma notable aquí, porque el *system prompt* es fijo y se repite a diario.

**2. Palancas para reducir el coste (por orden de impacto)**

| # | Palanca | Efecto |
|---|---|---|
| 1 | **Batching**: N titulares en una llamada, no N llamadas | Reduce la sobrecarga de prompt de sistema repetido |
| 2 | **Caché en disco** por hash(prompt + modelo + inputs) | Elimina llamadas repetidas por completo |
| 3 | **Deduplicación previa** de titulares | Evita pagar por la misma noticia 40 veces |
| 4 | **Modelo económico** para extracción | El mayor ahorro relativo |
| 5 | **Caché de contexto / prompt caching** del proveedor | Descuento sobre el *system prompt* repetido |
| 6 | **Truncar titulares** al contenido relevante | Menos tokens de entrada |
| 7 | **Filtrar antes de llamar**: solo noticias de las últimas X horas y con entidad relevante | Menos volumen total |

**3. Política de retención de disco**

> ⚠️ **Movida a §12.7.** La política de retención es ahora parte del **modelo de persistencia** (§12), que la define con su principio organizador y su justificación. Aquí solo se mantiene el principio de control:

**Regla:** un job mensual mide el tamaño de cada directorio y **avisa si alguno supera su presupuesto** (§12.7). Sin esta comprobación, en tres años hay 20 GB de caché que nadie mira.

**4. Topes duros obligatorios**

| Tope | Umbral | Comportamiento al superarlo |
|---|---|---|
| Tokens de entrada por ejecución | Configurable (`LLM_DAILY_BUDGET_EUR`) | Abortar el overlay y continuar sin él |
| Llamadas por ejecución | Configurable | Idem |
| **Gasto acumulado diario** | Tope en €  | Idem |
| **Gasto acumulado mensual** | Tope en € | Idem + **aviso registrado** en el informe y en el `run_log` |
| Tiempo máximo de la capa LLM | Configurable (p. ej. 5 min) | Abortar y continuar sin overlay |
| Fallos consecutivos de la API | 3 | Desactivar overlay ese día y registrarlo |

> **Regla dura:** superar cualquier tope **nunca** bloquea el pipeline. El sistema produce una recomendación sin overlay, lo registra en el `decision log` (`llm_overlay: "disabled_budget"`) y lo deja escrito en el informe y en el `run_log`. La decisión del día no depende de que la API esté disponible ni de que te quede presupuesto.

**5. Medición**

- Registrar en cada llamada: `provider`, `model`, `system_fingerprint`, tokens de entrada y salida, coste estimado, latencia y si hubo *hit* de caché.
- Informe mensual: gasto acumulado, coste por titular procesado, ratio de aciertos de caché, gasto desperdiciado por titulares duplicados.
- Si el coste por titular sube de forma inexplicable, algo ha cambiado en el proveedor: revisar antes de asumirlo.

---

## 7. Stack descartado y por qué

| Descartado | Motivo |
|---|---|
| **Kafka / RabbitMQ / Redis** | La latencia no es un problema. Cero necesidad de mensajería o caché distribuida |
| **Airflow / Prefect / Dagster** | Un DAG diario de 6 pasos es una función Python y un timer. Su infraestructura supera con creces la del proyecto |
| **Kubernetes / Docker** | Un solo proceso en una máquina. Docker solo si se migra a PostgreSQL y se quiere aislar |
| **MLflow / Weights & Biases** | Un directorio `runs/<hash>/` con JSON + modelo es suficiente, reproducible y sin servidor. W&B es un servicio en la nube sin justificación aquí |
| **Feast / feature stores** | Herramientas de equipo. El *feature store* es un módulo de ~100 líneas con Parquet + hash |
| **Celery** | No hay tareas distribuidas ni concurrencia |
| **Deep learning** (LSTM, transformers tabulares) | Muestra insuficiente (~250–375 operaciones). Sobreajuste garantizado |
| **TA-Lib** | Requiere compilar C, dependencia frágil, y solo se usan 5–10 indicadores |
| **`mlfinlab`** | De pago y problemático. Purga y embargo se implementan en ~50 líneas |
| **`backtrader` / `zipline` / `backtesting.py`** | Abstracción que oculta la semántica que hay que auditar |
| **LangChain a alto nivel** | Abstracciones inestables entre versiones |
| **Telegram y cualquier canal de notificación** (`python-telegram-bot`, `ntfy`, `Apprise`, `smtplib`) | La ejecución es manual y a demanda: no hay nada que notificar a distancia. Solo añadirían secretos, dependencias y superficie de fallo |
| **Streamlit / Grafana (Fase 1)** | No aportan nada al objetivo de la Fase 1. Fase 5 como pronto |
| **SQLite** | Mal rendimiento analítico columnar, tipos de fecha pobres |
| **MongoDB / InfluxDB** | No es el paradigma adecuado para datos tabulares y relacionales |
| **`pickle` como formato de intercambio** | Inseguro y frágil entre versiones. Usar Parquet, JSON u ONNX |
| **Cuadernos Jupyter en producción** | Estado oculto y orden de ejecución frágil: enemigo directo de la reproducibilidad |
| **Servicios cloud gestionados** (bases de datos, orquestadores, colas) | Nada que no puedas ejecutar en tu PC. El único servicio externo aceptado es la **API del LLM** |
| **Plataformas "todo en uno" de trading algorítmico** (QuantConnect, Alpaca-hosted, etc.) | Te atan a su nube, su *runtime* y su modelo de datos. El proyecto debe ser tuyo, en tu máquina y en formato abierto |
| **Bases de datos vectoriales** (Pinecone, Weaviate, Chroma) | No las necesitas: la deduplicación de noticias se resuelve con hashes y embeddings en memoria |
| **Fine-tuning de un LLM** | Coste, complejidad y un dataset de noticias etiquetadas que aún no existe. La extracción con *prompting* estructurado cubre la necesidad |
| **Agentes autónomos con capacidad de ejecutar órdenes** | Fuera de alcance por diseño: la ejecución es **siempre manual** (`plan.md` §2.12) |

---

## 8. Operación en el PC local

### 8.1 Ejecución manual

**No hay scheduler.** El pipeline se lanza a mano, cuando tú decides:

```bash
cd ~/cfdtrader
uv run python -m cfdtrader.delivery.run_daily
```

> **Nota sobre secretos:** al ejecutar a mano, `pydantic-settings` lee el `.env` desde el directorio de trabajo y el proceso hereda las variables de tu shell. El `.env` sigue necesitando permisos `600` y sigue fuera de git. ⚠️ **Verifica que la clave llega de verdad**: el fallo clásico es que el pipeline arranque, no encuentre la clave del LLM y emita la recomendación **sin overlay** sin que nadie lo note. Con ejecución manual hay una ventaja: lo ves en la terminal de esa misma ejecución.

Notas:

- ⭐ **El anclaje a `America/New_York` no desaparece: cambia de dueño.** Sin `systemd` no hay `Timezone=` que lo resuelva, así que la conversión ET/UTC ↔ Madrid **la hace el código**, no el sistema operativo (`plan.md` §8.3). Sigue siendo obligatorio calcular en ET/UTC y usar Madrid solo para presentar: la sesión se desplaza una hora dos veces al año.
- **La guardia de obsolescencia de §8.4 es obligatoria.** Sin timer que relance nada, sigue siendo posible ejecutar con datos de hace tres días (basta con no haber lanzado el pipeline), y **no debe producir una recomendación accionable**.
- **La comprobación de "¿es día de sesión válido?" la hace el propio pipeline**: los festivos de NYSE/Nasdaq y las medias sesiones son responsabilidad del código (`plan.md` §8.3).
- **No hay registro de cierre automatizado.** No hay un segundo disparo a las 16:20 ET que recoja el cierre: el cierre de la posición y su registro los haces tú en la misma sesión de trabajo (`plan.md` §12, regla 16).
- **Un fallo se ve al terminar.** Sin *heartbeat* ni alertas, la señal de que algo fue mal es el código de salida y la traza del `run_log`. Es la contrapartida aceptada de no tener notificaciones.
- **Ni `flock` ni `loginctl enable-linger`**: sin disparos automáticos no hay solapamiento posible y no hay unidades de usuario que mantener.

### 8.2 Backup y recuperación

| Qué | Cómo | Frecuencia |
|---|---|---|
| Código y configuración | **git** + remoto privado | En cada cambio |
| `data/raw/` (inmutable) | Copia a disco externo o `rsync` a otro equipo | Semanal |
| `data/derived/` | **No respaldar**: se reconstruye desde raw | — |
| **Caché de respuestas del LLM** | ⚠️ **Sí respaldar** (ver nota abajo) | Mensual |
| Modelos entrenados | Copia ligera (< 10 MB) | En cada reentrenamiento |
| `runs/` (experimentos) | Copia ligera | Mensual |
| `.env` | Copia **cifrada**, nunca en git | En cada cambio de clave |

> **Excepción a la regla de "no respaldar derived": la caché del LLM.** `data/derived/` se puede reconstruir porque todo deriva del raw. La caché del LLM **no**: reconstruirla exige volver a llamar a la API —pagando otra vez— y con un modelo que puede haber cambiado, de modo que la respuesta sería **distinta**. Es, por tanto, un **artefacto de reproducibilidad**, no un simple derivado. Si se pierde, se pierde la capacidad de reconstruir exactamente qué dijo el LLM los días pasados, y con ella la trazabilidad de la atribución (`plan.md` §19).
>
> **Decisión:** la caché del LLM se respalda con los mismos criterios que `data/raw/`, respetando la política de retención de §6.3 punto 3 (purgar > 12–18 meses) y **excluyendo cualquier contenido sensible** antes de copiarla.

**Prueba de recuperación anual:** reconstruir el proyecto desde cero en un directorio limpio a partir de `git` + `data/raw/` y comprobar que el backtest reproduce **exactamente** el mismo resultado. Si no lo hace, hay una dependencia no fijada o una fuente de no determinismo. **Esta prueba es el examen final del stack.**

### 8.3 Gestión del PC

- **Los datos viven en el PC, no en la nube.** Si el PC falla y no hay copia de `data/raw/`, se pierden años de descargas (recuperables parcialmente, pero a coste de tiempo).
- **No hay nada que se dispare solo.** Como la ejecución es manual, apagar el PC, suspenderlo o hibernarlo **no tiene ningún efecto sobre el pipeline**: simplemente no se ejecuta hasta que lo lances tú. Desaparece toda la casuística de disparos perdidos y recuperados.
- **Consumo**: el pipeline diario tarda minutos y consume poco. Con el LLM por API, **el consumo local es despreciable**: el trabajo pesado lo hace el proveedor. La carga del PC se limita a ingesta, features y entrenamiento puntual.

### 8.4 Guardia de obsolescencia (principio 11) — **obligatoria**

Al no haber scheduler, nadie ejecuta el pipeline por ti. Pero el peligro persiste y de hecho **aumenta**: si has estado tres días fuera y lanzas el pipeline sin más, podría producir una recomendación calculada con datos de hace tres sesiones, con la apariencia de ser la de hoy. **Una recomendación obsoleta es peor que ninguna recomendación**, porque se presenta con el mismo formato y la misma confianza que una buena.

**Reglas duras:**

| Condición | Comportamiento |
|---|---|
| `as_of` del snapshot **no es la fecha de hoy** | ❌ **No se emite recomendación accionable.** Se registra un aviso: "datos obsoletos, sin recomendación" |
| El cierre de la **sesión anterior** no está en el almacén | ❌ Sin recomendación (no se puede calcular el *gap*) |
| Han pasado **más de N días naturales** sin ejecución | ❌ Sin recomendación + aviso explícito de reincorporación |
| Festivo en EE. UU. (mercado cerrado) | ⏭️ No se ejecuta (o se ejecuta y devuelve `NOTHING` justificado) |
| ⚠️ **Media sesión US** (cierre 19:00 CET) | ❌ `NOTHING` por defecto: el rango esperado cae ~45% y el modelo no es válido (`plan.md` §12, regla 18) |
| El usuario ha estado ausente > 1 semana | ⚠️ Recomendación en **modo observación**: se muestra pero marcada como "no operar hasta revalidar" |

**Consecuencia de diseño:** el pipeline debe **distinguir tres estados de salida**, no dos:

1. **Recomendación accionable** (`LONG` / `SHORT` / `NOTHING`).
2. **Sin recomendación por datos insuficientes u obsoletos** — que es distinto de `NOTHING`.
3. **Error** (fallo técnico).

Confundir (2) con `NOTHING` es un error de concepto: `NOTHING` significa "he evaluado el mercado y hoy no veo oportunidad"; el estado (2) significa "**no sé**". El `decision log` debe registrarlos por separado, y el informe debe presentarlos de forma visualmente distinta.

> **Regla adicional de reincorporación:** tras una ausencia de más de una semana, el sistema entra en **modo observación durante 5 sesiones**. La primera semana después de una pausa es cuando más probable es que el usuario opere mal por exceso de confianza o por querer "recuperar" lo no operado. El sistema debe ayudar a no hacerlo.

---

## 9. Riesgos del stack y planes de contingencia

| Riesgo | Probabilidad | Impacto | Plan de contingencia |
|---|---|---|---|
| `yfinance` se rompe | **Alta** | Medio | Adaptador propio + `fallback` automático a Stooq + marcado de dato *stale* en el informe |
| Una API gratuita cambia de condiciones | Media | Medio | El raw ya descargado sigue siendo válido; el adaptador permite cambiar de fuente |
| Una librería queda sin mantenimiento | Media | Bajo–Medio | Licencias permisivas y formatos abiertos ⇒ sustitución posible. Por eso el núcleo usa pocas dependencias |
| LangGraph introduce *breaking changes* | Alta | Bajo | **Pin exacto** y confinado a `orchestration/`. El gate y el backtest no dependen de él |
| **El proveedor LLM cambia o retira el modelo** | **Alta** | Medio | Fijar identificadores con versión concreta; registrar `system_fingerprint`; alertar cuando cambie. Como el LLM no genera alpha, el impacto está acotado |
| **La API del LLM está caída a las 08:00** | Media | Bajo | El overlay es opcional: el pipeline produce recomendación sin él y lo registra. Reintentos con backoff. Proveedor alternativo configurable |
| **El coste del LLM se dispara** | Media | Medio | Topes duros diarios y mensuales, caché, batching, deduplicación y modelo económico. *Fallback* a Ollama local si es necesario |
| **Cambio de precios del proveedor** | Media | Bajo | Contador de gasto propio y acumulativo: se detecta el mismo día. Recalcular §6.3 |
| **Fuga de la clave de API** | Baja | **Alto** | `.env` con permisos `600`, fuera de git (pre-commit lo verifica), límites de gasto configurados **también en la consola del proveedor**, rotación periódica |
| **Envío inadvertido de datos sensibles al proveedor** | Baja | Medio | Lista blanca de campos enviados: solo titulares públicos y metadatos de calendario. Test que verifique el contenido del *payload* |
| Corrupción de datos en disco | Baja | **Alto** | Raw inmutable + backup semanal + prueba de recuperación anual |
| El backtest resulta no reproducible | Media | **Alto** | Test de golden dataset y de determinismo del gate. Se detecta antes de que importe |
| Falta de datos intradía para el backtest | **Alta** | Medio | Presupuestar EODHD, o rediseñar la estrategia a horizonte diario |
| **Sobreajuste** (riesgo dominante, no técnico) | **Alta** | **Muy alto** | Ninguna herramienta lo evita. Depende del protocolo de `plan.md` §11 |

> **Observación importante:** el mayor riesgo del proyecto **no es tecnológico, es estadístico** (el sobreajuste) y **económico** (los costes de operar). Ninguna elección de stack mitiga eso. El stack solo garantiza que, cuando el resultado sea malo, lo sepas con certeza y no por un artefacto de tu propia herramienta.

---

## 10. Criterios de migración (cuándo salir de cada pieza)

El stack está diseñado para poder sustituirse pieza a pieza. Umbrales concretos:

| Pieza | Migrar cuando | A |
|---|---|---|
| DuckDB | Varios procesos necesiten escribir a la vez, o el fichero supere la RAM del equipo de forma sostenida | PostgreSQL + TimescaleDB |
| Polars | Nunca (o solo si una librería concreta lo exige) | pandas puntual |
| Motor de backtest propio | Si se necesita barrer > 10.000 combinaciones de parámetros | vectorbt (OSS) como capa de análisis, **manteniendo el motor propio como referencia** |
| Tracking propio (`runs/`) | Si se superan ~200 experimentos y cuesta encontrar resultados | MLflow local |
| **API LLM (OpenAI / DeepSeek)** | Si el coste se vuelve significativo, si la privacidad pasa a importar, o si el proveedor deja de dar garantías de estabilidad del modelo | **Ollama + modelo local** (plan B ya documentado en §4.9). La interfaz `LLMClient` hace que el cambio no toque `agents/` ni `orchestration/` |
| Proveedor LLM concreto | Si el otro proveedor mejora en coste, calidad o estabilidad de modelo | El otro proveedor vía la misma interfaz. Ambos son compatibles con el SDK de `openai`: cambiar `base_url` y `model` |
| Sin visualización | Fase 4–5, para revisar la curva de equity | Streamlit |

**Regla de oro:** ninguna migración se hace por moda ni por "está más de moda". Se hace **cuando se cruza un umbral medido**, y se anota en el registro de cambios con la medición que la motivó.

---

## 11. Resumen ejecutivo del stack

| Capa | Elección | Licencia |
|---|---|---|
| Lenguaje | Python 3.12 | PSF |
| Entorno | uv | MIT / Apache-2.0 |
| Lint / tipos | ruff · pyright | MIT |
| Config | pydantic-settings + YAML | MIT |
| **Almacenamiento** | **DuckDB + Parquet** | MIT / Apache-2.0 |
| Procesamiento | Polars + numpy + scipy | MIT / BSD-3 |
| HTTP | httpx + tenacity | BSD-3 / Apache-2.0 |
| Datos mercado | yfinance + Stooq | Apache-2.0 |
| **Macro** | **FRED** (primaria) + ECB SDW y Eurostat (contexto) vía httpx | datos libres con atribución |
| Datos de mercado | yfinance: `^GSPC`, `ES=F`, `SPY`, `^VIX`, ETFs sectoriales | Apache-2.0 |
| Noticias | GDELT + feedparser | datos libres / BSD-2 |
| Indicadores | Implementación propia | — |
| Volatilidad | arch | NCSA |
| ML | scikit-learn + LightGBM | BSD-3 / MIT |
| Backtest | **Motor propio** | — |
| Reporte | quantstats | Apache-2.0 |
| **LLM** | **API OpenAI / DeepSeek** vía SDK `openai` + `LLMClient` propio | servicio (no OSS) |
| Caché LLM | diskcache | Apache-2.0 |
| *Fallback* LLM | Ollama (contingencia) | MIT |
| Orquestación | LangGraph (pin exacto) | MIT |
| Informe | Rich + Jinja2 | MIT / BSD-3 |
| Logging | loguru | MIT |
| Tests | pytest + hypothesis | MIT / MPL-2.0 |
| Exploración | marimo | Apache-2.0 |
| **Coste marginal** | **≈0 €** (solo el LLM) | **Cómputo y datos 100 % locales** |
— solo el LLM por API | Cómputo, datos y decisiones en local
**Las cuatro decisiones estructurales de este stack:**

1. **DuckDB + Parquet con esquema *point-in-time*** — es lo que hace el proyecto reproducible y auditable. Todo lo demás es reemplazable.
2. **Motor de backtest propio** — en un problema donde el resultado depende de semántica sutil de costes y purga, la transparencia vale más que las prestaciones.
3. **LLM por API detrás de una interfaz propia** — da la mejor calidad de extracción con el mínimo mantenimiento y hardware, a cambio de un coste pequeño y acotado. La interfaz `LLMClient` mantiene abierta la vuelta a local.
4. **El LLM nunca está en el camino crítico** — ni del backtest ni de la decisión. Puede vetar y redactar, no decidir ni calcular. Esto es lo que permite que un componente no reproducible y dependiente de un tercero no contamine el núcleo del sistema.

---

## 11 bis. Decisiones que este documento NO puede cerrar por ti

Lo que sigue **no son huecos del documento, son decisiones que dependen de información o prioridades que solo tú tienes**. Se listan para que no se den por resueltas:

| # | Decisión abierta | Información que falta | Bloquea a |
|---|---|---|---|
| 1 | **Proveedor principal: OpenAI o DeepSeek** | Leer y comparar términos de servicio, política de retención, DPA y jurisdicción de ambos (§3.2.d). El coste es un criterio, no el único | §4.9, Fase 3 |
| 2 | **¿Se contrata archivo histórico de noticias?** | Decidir si la capa LLM se evalúa solo prospectivamente (gratis, lento) o retrospectivamente (20–50 €/mes) | Fase 3 completa |
| 3 | **¿Se compran datos intradía históricos?** | `plan.md` §21 y Fase 0: depende de si el spread real deja margen. **No contratar antes de tener la medición de coste** | Fase 1–2 |
| 4 | **Bróker definitivo** | Medición de spread real en Fase 0 | Todo el proyecto (afecta al EV) |
| 5 | **Umbrales concretos** (EV mínimo, riesgo por operación, pérdidas máximas, tamaño de `R`) | `plan.md` §21 | Gate, backtest |
| 6 | **Horizonte y precio de entrada exactos** (`open` vs 09:00 vs primeros 30 min) | Fase 0 | Etiquetado tri-barrera |
| 7 | **Periodo histórico y *holdout* intocable** | Decisión personal, debe fijarse **antes** de ver resultados | Protocolo de evaluación |
| 8 | **Política de retención de noticias** (¿texto completo o solo resumen?) | Depende de la decisión 2 | Disco, §6.3.3 |

**Regla:** ninguna de estas se resuelve "sobre la marcha". Se decide, se anota en `plan.md` o aquí, **y entonces** se implementa. Decidir después de ver resultados es la definición operativa del sobreajuste.

---

## 12. Modelo de persistencia

> Sección normativa. Define **qué se guarda, dónde, durante cuánto tiempo y por qué**. Sustituye y amplía la política de retención de §6.3 punto 3.

### 12.1 Principio organizador: persistir lo irreversible

La pregunta correcta no es "¿cuánto dato he usado?" sino:

> **¿Podría reconstruir esto dentro de dos años?**

- Si **sí** → no hace falta guardarlo (o basta una ventana corta para depurar).
- Si **no** → se guarda **siempre**, con independencia de su tamaño.

Sin este criterio explícito, una política de retención se reinterpreta mal en seis meses y se acaba borrando lo insustituible o guardando lo trivial.

### 12.2 Las cuatro clases de dato

| Clase | Qué incluye | Retención | Por qué |
|---|---|---|---|
| **Irreversible** | Datos de mercado descargados, la decisión tomada, los eventos extraídos de noticias | **Permanente** | La fuente puede desaparecer, revisar el histórico o caducar el RSS. Y la decisión no se puede volver a preguntar |
| **Recomputable** | Features, señales de agentes, etiquetas, métricas | **Corta o ninguna** | Se regeneran desde `raw` + versión de código + `uv.lock` |
| **Coste, no dato** | Caché de prompt/respuesta del LLM | 12–18 meses | Solo ahorra dinero; perderla solo cuesta volver a pagar |
| **Operacional** | Logs de ejecución, artefactos de experimentos | 90 días / trimestral | Diagnóstico, no historia |

**El caso más importante es el diario de decisiones.** Es el único dato estrictamente irreversible del sistema, y no es el dato de mercado:

> Cada día que pasa, **el sistema que produjo esa recomendación deja de existir**: el modelo se reentrena, el código de features evoluciona, la caché se purga. Si no se registra la decisión, nunca se podrá volver a preguntar qué se habría recomendado ese día. Sin diario no hay atribución (`plan.md` §19), no hay detección de drift y no hay forma de distinguir un sistema que funciona de una racha de suerte.

### 12.3 La escala real (por qué casi nada merece poda)

Como el sistema decide **una vez al día**, los artefactos de decisión son **una fila por sesión**:

| Dataset | Filas en 10 años | Tamaño |
|---|---|---|
| `derived.features_daily` (~100 features) | 2.500 | **~2 MB** |
| `derived.labels` | 2.500 | < 1 MB |
| `journal.agent_signals` (10 agentes × día) | 25.000 | **~1 MB** |
| `journal.decisions` (fila completa con JSON) | 2.500 | **~25 MB** |
| `derived.events` (noticias extraídas) | 250.000 | **~50 MB** |
| `raw.market_daily` | 100.000 | **~2 MB** |
| **TOTAL del núcleo histórico** | | **< 100 MB por década** |

**Conclusión contraintuitiva:** **guardar es barato; decidir qué borrar es caro.** En este proyecto la poda agresiva cuesta más en complejidad y en riesgo de destruir algo irrecuperable que en disco. Lo único que crece de verdad son cuatro cosas —intradía fino, texto de noticias, caché del LLM y artefactos de experimentos— y son exactamente las que se podan.

### 12.4 Capa de investigación — retención permanente

| Dataset | Grano | Notas |
|---|---|---|
| `raw.market_daily` | ticker × sesión | Inmutable, *append-only*. Base del backtest |
| `raw.market_intraday` | ticker × barra | Solo si se compran datos finos |
| `raw.macro` | serie × fecha | FRED (primaria), ECB SDW, Eurostat |
| `raw.sectors` | ETF sectorial × sesión | Los 11 SPDR, **contra el sesgo de supervivencia** |
| `raw.news_headlines` | titular | ⚠️ **Titular + URL + `published_at` + hash. PROHIBIDO el cuerpo del artículo** (§12.7) |
| `derived.features_daily` | sesión × feature | Persistida **también** de forma permanente: permite detectar si la recomputación deriva |
| `derived.labels` | sesión | Etiquetas tri-barra |
| `derived.events` | evento | Extracción del LLM ya validada por Pydantic. Irreversible: el modelo que la generó se retirará |

### 12.5 Capa de diario — retención permanente (~30 MB por década)

| Tabla | Contenido |
|---|---|
| `journal.decisions` | **Una fila por sesión.** Ver columnas abajo |
| `journal.agent_signals` | `trade_date`, `agent`, `prob_up`, `confidence`, `veto`, `veto_reason`, `evidence` (JSON) |
| `journal.trades` | Operación real si la hubo: precios, horas, nocional, P&L, costes efectivos, motivo de salida |
| `journal.overrides` | `trade_date`, recomendación del modelo, acción humana, **motivo**, confianza declarada |
| `journal.attribution` | Post-mortem: qué agente habría acertado solo |

**Columnas obligatorias de `journal.decisions`:**

| Grupo | Campos |
|---|---|
| Identificación | `trade_date` (PK), `as_of` |
| **Estado** | `status` ∈ `recommendation` \| `no_recommendation_stale_data` \| `no_recommendation_data_quality` \| `error` |
| Versiones | `features_version`, `model_version`, `prompt_hashes` (JSON), `git_commit` |
| Cálculo | `prob_up_raw`, `prob_up_calibrated`, `expected_move_pct`, `cost_pct`, `ev_net_pct` |
| Recomendación | `direction`, `stop_pct`, `target_pct`, `size_notional_eur`, `size_fraction`, `leverage_implied`, `tier` |
| Contexto | `blocking_events`, `bull_case`, `bear_case` (JSON) |
| Overlay LLM | `llm_overlay` ∈ `applied` \| `veto` \| `disabled_budget` \| `disabled_error` \| `disabled_timeout` |
| Salida | `report_text` — el informe **tal cual se emitió**, sin reformatear |

> **El campo `status` es crítico:** distingue "hoy no hay oportunidad" de "no sé" (§8.4). Confundirlos destruye la capacidad de auditar el sistema.

### 12.6 Capa operacional — retención corta

| Tabla | Retención | Contenido |
|---|---|---|
| `ops.run_log` | **90 días** | Traza por ejecución: etapas, duraciones, errores |
| `ops.llm_calls` | **18 meses** | `provider`, `model`, `system_fingerprint`, `purpose`, tokens entrada/salida, `cache_hit`, coste estimado, latencia, éxito. **Alimenta el informe de coste de §6.3** |
| `ops.llm_cache` | **12–18 meses** | Caché de prompt/respuesta. Artefacto de coste, no de verdad |
| `ops.backtest_runs` | Config y métricas: **permanente** (KB) · Binarios: **trimestral** | Solo los binarios de modelos que llegaron a producción se conservan |

**Binarios de modelos de producción:** conservar el de **cada modelo que generó decisiones en vivo** (~1–3 MB cada uno, 12 al año ⇒ ~36 MB/año). No conservar los de experimentos descartados.

### 12.7 Política de retención consolidada

| Componente | Retención | Tamaño estabilizado |
|---|---|---|
| `raw.market_*` (diario e intradía) | **Permanente** | < 200 MB, o ~1,2 GB con intradía fino |
| `raw.macro` | **Permanente** | ~5 MB |
| `raw.news_headlines` | **Permanente** | ~50 MB en 5 años |
| **Cuerpo de artículos de noticias** | ❌ **NO SE GUARDA** | 0 |
| `derived.features_daily`, `derived.labels` | **Permanente** | ~3 MB por década |
| `derived.events` | **Permanente** | ~50 MB en 5 años |
| `journal.*` | **Permanente** | ~30 MB por década |
| `ops.run_log` | 90 días | < 50 MB |
| `ops.llm_calls` | 18 meses | ~10 MB |
| `ops.llm_cache` | 12–18 meses | ~150–250 MB |
| `ops.backtest_runs` (binarios) | Trimestral | ~500 MB |
| Modelos de producción | **Permanente** | ~36 MB/año |
| *Spill* temporal de DuckDB | Limpiar al final de cada ejecución | 0 (transitorio) |

**Los cuatro cambios respecto a la política anterior:**

1. ⚠️ **Se prohíbe guardar el cuerpo de los artículos de noticias.** Antes se contemplaban 2 GB. El motivo principal **no es el disco**: almacenar el texto íntegro de artículos de agencias y medios **puede infringir sus términos de uso y derechos de autor**. Se guarda titular + URL + `published_at` + hash.
2. **Se añaden a la tabla los cuatro datasets que faltaban**: features persistidas, diario de decisiones, operaciones realizadas y binarios de modelos de producción.
3. **Los binarios de modelos se acotan a los de producción.** No se acumulan los de cada experimento.
4. **Los eventos extraídos pasan a ser explícitamente irreversibles.** No basta con "son pequeños": el modelo que los generó se retirará y no se podrán regenerar igual.

### 12.8 Lo que NO se persiste

| No persistido | Motivo |
|---|---|
| **Cuerpo de artículos de noticias** | Términos de uso y derechos de autor; además es innecesario |
| **Texto completo de los prompts** | Viven en git, versionados junto al código. Se guarda su **hash**, que es lo que garantiza la trazabilidad |
| Estados intermedios del pipeline | Recomputables; solo añaden superficie de fallo |
| *Spill* y ficheros temporales | Transitorios por definición |
| Binarios de modelos experimentales descartados | Reentrenables; solo ocupan |
| Cualquier credencial, dato personal o posición abierta en la caché del LLM | Debe estar excluido por diseño (§3.2.b) |

### 12.9 Verificación del modelo de persistencia

| Frecuencia | Comprobación |
|---|---|
| **Mensual** | Medir el tamaño de cada directorio y **avisar si supera su presupuesto** (tabla §12.7). Sin esto, en tres años hay 20 GB de caché que nadie mira |
| **Trimestral** | Aplicar la poda y purgar `ops.backtest_runs` y binarios de modelos no productivos |
| **Anual** | ⭐ **Prueba de reconstrucción:** elegir una decisión de hace ~1 año, hacer checkout del `git_commit` registrado, restaurar el `uv.lock`, recomputar las features desde `raw` y **comparar con lo guardado en `journal.decisions`** |

> **La prueba anual es la prueba de fuego de todo el modelo.** Si la recomputación **no coincide** con lo registrado, hay una fuente de no determinismo —una dependencia sin fijar, un dato que se revisó, una feature con estado oculto— y el sistema **no es auditable**. Detectar eso una vez al año es infinitamente más barato que descubrirlo cuando una divergencia en producción no sepas explicar.
>
> Y si la recomputación coincide, has demostrado algo valioso: que el diario es suficiente para reconstruir cualquier decisión pasada, y que **no necesitas guardar los estados intermedios**. La prueba valida el diseño entero.

---

## Registro de cambios

| Fecha | Versión | Cambio | Motivo |
|---|---|---|---|
| 2026-09-16 | 1.0 | Versión inicial | Especificación del stack con prioridad open source y ejecución local en PC |
| 2026-09-16 | 1.1 | **La capa LLM pasa de Ollama local a API (OpenAI / DeepSeek)**. Se añade: §3.2 (licencias, datos y reproducibilidad por API), §4.2 (claves necesarias), §4.9 reescrita con estrategia de dos modelos, §6.3 (control de coste normativo), riesgos y criterios de migración asociados. Se revisan: principios 1/2/3 y 7–9, requisitos de hardware a la baja, modo offline y resumen ejecutivo | Decisión del usuario: usar el LLM por API en lugar de modelo local |
| 2026-09-16 | 1.2 | **Corrección del dimensionamiento de disco a la baja: de 64 GB a 20 GB libres recomendados.** Se rehace el desglose de consumo con cifras calculadas fila a fila, incluyendo tres partidas que faltaban (caché del LLM, runs de experimentos, entorno Python). Se añade §6.3 punto 3 (política de retención de disco) y el campo "Huella en disco" en la cabecera | La cifra anterior era un residuo de la etapa con Ollama (20 GB por modelo). Al pasar a API se ajustó a la mitad sin recalcular, y seguía siendo un 5–10× superior a la real |
| 2026-09-16 | 1.3 | **Revisión del documento.** Correcciones: principio 9 duplicado, entrada duplicada en el registro de cambios, afirmación imprecisa sobre `Type=oneshot`, recuento de paquetes en §5.1, fila residual en §9, celda confusa en §11, y "100 % del stack" en §3.1. Nuevo: principio 11 y **§8.4 guardia de obsolescencia**; comparación de **jurisdicción y residencia de datos** por proveedor (§3.2.d); **capacidades de salida estructurada dispares** entre proveedores (§4.9); corrección del uso de *embeddings* (§4.9); y **§11 bis: qué NO se puede decidir todavía** | Auditoría solicitada por el usuario |
| 2026-09-16 | 1.4 | ✅ **CIERRE DE LA ESPECIFICACIÓN.** Nuevo **§12 Modelo de persistencia**, normativo: principio de irreversibilidad (§12.1), las cuatro clases de dato (§12.2), escala real (§12.3), esquemas por capa —investigación, diario, operacional— (§12.4–12.6), **política de retención consolidada** (§12.7), lo que NO se persiste (§12.8) y la **prueba anual de reconstrucción** (§12.9). La política de retención sale de §6.3 punto 3, que pasa a ser un puntero, para tener una sola fuente de verdad. La entrega de noticias pasa de "texto completo opcional" a **prohibición de guardar el cuerpo del artículo** | Pregunta del usuario sobre qué persistir. El análisis mostró que (a) el principio correcto es "persistir lo irreversible", que no estaba enunciado; (b) cuatro datasets no aparecían en la política de retención; (c) guardar el cuerpo de artículos de prensa tiene riesgo de derechos de autor, no solo de disco |
| 2026-09-16 | **2.0** | ⚠️ **REVISIÓN POR CAMBIO DE INSTRUMENTO A CFD DEL S&P 500.** Cambian: título y cabecera. §2.1 desglose de disco (tickers y ETFs sectoriales americanos). §2.3 horario del pipeline (13:30 → informe 15:00 → registro 22:15). §4.5 fuentes: **FRED pasa a primaria** y ECB a contexto, se añaden `^GSPC`/`ES=F`/`SPY`/`^VIX`/ETFs sectoriales, calendario de resultados de mega-caps. §4.11 y §8.1: **el scheduler se ancla a `America/New_York` en lugar de `Europe/Madrid`**, con timer de registro de cierre. §5.1, §6.1, §6.2, §8.4, §11 y §12.4 actualizados. **No cambia** el modelo de persistencia (§12), ni la arquitectura, ni la capa LLM, ni el resto del stack | **Decisión del usuario: operar el CFD del S&P 500.** El anclaje del scheduler a la hora de Nueva York es el cambio técnico más importante: anclar a hora local produciría una ejecución una hora tarde durante ~4 semanas al año |
| 2026-09-16 | **2.1** | 🔧 **Corrección de dos horas heredadas del flujo matinal, tras fijar el calendario canónico en `plan.md` v2.2.** §2.3: la entrega del informe pasa de 09:30 a **09:00 ET** (09:30 ET es la apertura, no una hora de entrega). §4.13: el *heartbeat* pasa de 08:50 a **09:05 ET**, que es cuando el informe ya debería estar entregado | Al cerrar el calendario en `plan.md` §4.1 aparecieron dos horas incoherentes con él. **Queda pendiente una tercera, ajena a este cambio:** el registro de cierre es **16:15 ET** en `plan.md` §4.1/§13 y aquí §2.3, pero **16:20 ET** en el ejemplo de §8.1 y en las tareas #41 y #43. Se unifica cuando la tarea #41 implemente los timers |
| 2026-09-17 | **2.2** | 🔒 **Contrato del almacén cerrado en código (tarea #2).** Nuevo `src/cfdtrader/data/store.py`, único punto de acceso al almacén, con: layout `<raíz>/<capa>/<dataset>/source=<source>/year=<año de as_of>/*.parquet` en Parquet **ZSTD**; las **seis columnas obligatorias** de `plan.md` §8.2 más `series_id`; identidad `(dataset, source, series_id, as_of)`; `version` como **contador de revisión del dato** (no versión de esquema → #49); `append` idempotente e `ImmutableWriteError` ante contenido distinto, `append_revision` con `version = max + 1`, `replace` **solo** en `derived`; visibilidad *point-in-time* por `published_at` con caída a `fetched_at` cuando es `NULL`, y **`as_of` nunca decide visibilidad**; DuckDB como **motor de consulta puro** sobre los Parquet, sin catálogo `.duckdb`. Queda escrito en el módulo que el `as_of` de una barra diaria es el **cierre de sesión (16:00 ET) en UTC**, no medianoche; y las **vistas SQL exponen una sola fila por identidad** (la revisión vigente), de modo que un `replace` en `derived` no deja legible el valor sustituido | Primera tarea de código de la Fase 0 tras el bootstrap. **Corregido tras el QA de la #2:** la vista era un `SELECT *` sin filtrar por versión, así que un dataset de `derived` devolvía a la vez el valor viejo y el nuevo. Las decisiones de esquema se cerraron con el usuario en el *grooming* de la #2 y el contrato se escribió **dentro del módulo** para que no viva solo en la issue |
| 2026-09-18 | **2.3** | 🚫 **RETIRADA DE LA AUTOMATIZACIÓN Y DE LAS NOTIFICACIONES.** **§2.2:** la tabla de sistemas operativos deja de argumentar sobre schedulers; se especifica asumiendo **solo Linux**. **§2.3:** «ejecución batch diaria» → **ejecución manual y a demanda**; se declara que no hay scheduler, timer ni disparo automático, y que el sistema **no envía nada a ningún sitio**. **§3.1:** `systemd` deja de ser excepción de licencia (solo queda el servicio LLM por API). **§4.2:** fuera `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`. **§4.11** reescrita: de «Scheduling y ejecución local» con `systemd timer` + `flock` + journald a **«Ejecución local»** con disparador manual, sin scheduler, sin *lock* y con `run_log`; se conserva explícitamente el **anclaje a `America/New_York`**, que pasa del sistema operativo al código. **§4.12** reescrita: de «Notificaciones» (Telegram, Apprise, ntfy, `smtplib`, `python-telegram-bot`) a **«Informe y salida»** (Rich + Jinja2 + Markdown/JSON en `data/derived/reports/`). **§4.13:** fuera las filas de alertas de fallo y *heartbeat*; **Sentry descartado**; los fallos quedan en el `run_log` y el código de salida. **§5.2, §6.1, §10 y §11:** fuera las partidas de scheduling y notificaciones, dentro «Informe». **§6.4:** «alerta por Telegram» → **aviso registrado** en el informe y el `run_log`. **§8.1** reescrita: de los tres ficheros de unidad de `systemd` a un **comando manual**; se conserva la nota de secretos adaptada al `.env` local. **§8.3:** desaparece la casuística de disparos perdidos y de `Persistent=true`. **§8.4:** la guardia de obsolescencia se mantiene y su motivación se adapta (el riesgo **aumenta** sin scheduler, porque nadie te obliga a ejecutar). **§8.4** y **§9:** «notificación al usuario» → «el informe»; «alerta de dato *stale*» → «marcado en el informe» | **Decisión del usuario (2026-09-18): ejecución manual a demanda, sin Telegram y sin automatización.** No hay nada que notificar a distancia porque el usuario está delante cuando el sistema se ejecuta. La obligación de cierre a las 16:00 ET se mantiene, pero la garantiza el usuario, no el software |
