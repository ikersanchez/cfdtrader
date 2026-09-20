"""Ingeniería de features (`_docs/plan.md` §9).

Reglas de la capa:

- Cálculo determinista y auditable a mano (nada de librerías que oculten ventanas).
- Ventanas y normalización siempre sobre datos anteriores al instante de decisión.
- Cada matriz se versiona por ``sha256(código + parámetros + ventanas + fuente + as_of)``.

Módulos:

- :mod:`cfdtrader.features.volatility` — **definición única** del ATR normalizado y
  de la familia de volatilidad del proyecto (Parkinson, HAR y VIX, tarea #7). Las
  tareas #20 (features técnicas, que también nombra el ``atr_norm``) y #23 (features
  de régimen y volatilidad) **importan de ahí**, no reimplementan las fórmulas.
- :mod:`cfdtrader.features.technical` — familia **técnica** cerrada (tarea #20):
  retornos multi-ventana, ATR normalizado (importado de
  :mod:`cfdtrader.features.volatility`), distancia a la media móvil, RSI de Wilder,
  posición en el rango, ruptura del recorrido y dos z-scores de ventana expandida
  (importados de :mod:`cfdtrader.features.store`). Se persiste con el **mismo**
  ``Store`` y convive con la familia de volatilidad en ``derived.features_daily``,
  que las separa por ``source``. No lee el ``open``.
- :mod:`cfdtrader.features.context` — familia de **contexto de mercado** cerrada
  (tarea #21): correlaciones móviles del S&P con Europa y el Nikkei, overnight
  asiático, cierre europeo anterior, beta del VIX, retorno del dólar, dispersión
  sectorial y su z-score (importado de :mod:`cfdtrader.features.store`). Es la
  única familia con **muchas** series de entrada: recibe un ``Mapping`` con una
  entrada por serie (19) porque cada mercado trae su **propio** calendario, y
  declara el alineamiento entre ellos (``CONTEXT_MARKET_LAG``) en vez de
  esconderlo. No publica ningún retorno del S&P 500: los ``ret_1``/``ret_5``/
  ``ret_21`` siguen siendo de :mod:`cfdtrader.features.technical`.
- :mod:`cfdtrader.features.macro` — familia **macro** cerrada (tarea #22): nivel y
  cambio a cinco sesiones de la Fed funds, el UST 10y, el UST 2y y la pendiente
  2s10s, la inflación interanual del CPI y del PCE, el nivel del índice dólar y dos
  z-scores de ventana expandida (importados de :mod:`cfdtrader.features.store`).
  Es la familia que alinea **point-in-time**: cada fila usa solo lo que ya estaba
  publicado al cierre de su sesión, con el ``published_at`` real de cada
  observación y sin interpolar ni censurar un valor viejo. Tampoco escribe en el
  almacén por su cuenta (lo hace :func:`cfdtrader.features.store.save_daily`).
- :mod:`cfdtrader.features.regime` — familia de **régimen y volatilidad** cerrada
  (tarea #23): percentil expandido de la volatilidad realizada de Parkinson,
  pronóstico **GARCH(1,1)** (el candidato elegido en #7, con su motor importado de
  :mod:`cfdtrader.features.volatility`), su z-score robusto, el *efficiency ratio*
  de Kaufman y tres columnas de calendario derivadas **del propio frame** (día de
  la semana, sesiones hasta el vencimiento mensual y marca del trimestral). No
  publica ninguna columna de VIX ni duplica nada de las otras familias.
- :mod:`cfdtrader.features.store` — **infraestructura de persistencia** (tarea #19):
  identidad reproducible (``features_version`` por sesión y ``feature_spec_sha256``
  por contrato), esquema ancho en ``derived.features_daily``, catálogo documentado y
  normalización robusta de ventana expandida. Las familias de features (#20–#23) se
  persistirán **a través de él**, no con su propio ``Store``. También publica el
  **registro de familias** (``CATALOG_BY_FEATURE_SET``): cada ``feature_set`` tiene
  su catálogo y su ``source``, y una familia sin registrar es un error tipado.

Se implementa en las tareas #19–#23.
"""
