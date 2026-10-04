# Higiene de disco: tamanos frente a presupuesto (tarea #44)

- **Modo:** `sizes` (job mensual, manual)
- **as_of:** `2026-10-04T12:00:00+00:00`
- **Componentes por encima de presupuesto:** ninguno

| Componente | Ruta | Tamano | Presupuesto | Estado |
|---|---|---:|---:|---|
| `raw` | `data/raw` | 6.0 MB | 256.0 MB | ok |
| `derived` | `data/derived` | 2.4 MB | 64.0 MB | ok |
| `http_cache` | `data/cache` | 3.9 MB | 64.0 MB | ok |
| `journal` | `journal` | 0 B | 32.0 MB | ok |
| `ops_run_log` | `journal/ops` | 0 B | 50.0 MB | ok |
| `ops_llm_cache` | `journal/ops/llm_cache` | 0 B | 250.0 MB | ok |
| `runs` | `runs` | 2.6 MB | 512.0 MB | ok |
| **TOTAL** | | **14.8 MB** | **1.2 GB** | |

> Comando **manual**: no hay scheduler ni servicio que lo dispare (`tech_stack.md` §4.11).
