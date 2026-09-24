"""Motor *walk-forward* del arnés de backtest — tarea #13.

Reparto de responsabilidades (frontera explícita): **#11 cobra; #13 recorre y simula**.
Este módulo consume el plan de particiones de #12 tal cual (no reimplementa particiones) y
llama **una sola vez** a ``cost_breakdown`` de #11 por operación. No calcula diferencial,
tenencia, divisa ni comisión por su cuenta; no mide el *slippage*; no decide ``R``, el
tamaño ni el apalancamiento (#60, #27); no entrena nada (#24, #25); no calcula métricas
(#15), ni PBO (#16), ni la puerta de Fase 0 (#9). Traza: ``plan.md`` §2 (principio 5),
§3.2, §3.3, §3.5, §4.1, §4.2, §11.1-§11.3, §12 (reglas 6, 7, 8, 16 y 18), §14 y §15;
``tech_stack.md`` §3.2, §3.3 (puntos 2 y 5), §4.7, §4.8 (motor propio, ~200 líneas) y
§4.9 (el overlay LLM corre **siempre** deshabilitado).

``t0`` y ``t1``: la etiqueta de una sesión
------------------------------------------

Para la muestra ``i``, ``t0`` es el **open de la subasta** de apertura y ``t1``
es el cierre de la **misma** sesión (el ``close_utc`` que publica #10). El proyecto es
intradía puro ``open`` -> ``close``, sin *overnight* (``plan.md`` §12, regla 6), así que
``t1`` cae dentro de la propia sesión ``i`` y el **horizonte de etiqueta** vale
``label_horizon = 0`` en todas las muestras: **ninguna posición existe entre sesiones**. En
las **medias sesiones**, ``t1`` es su propio cierre, no un cierre fijo. El motor
no conoce ninguna hora ET literal y no asume ninguna duración de sesión (A24): una media
sesión es una sesión más.

La vista de decisión, sin el futuro de la sesión
------------------------------------------------

La función de decisión ve **solo** ``SessionView(session, open_px, gap_px, context)``: no
expone ``high_px``, ``low_px`` ni ``close_px`` de la sesión en curso (son el futuro de esa
sesión) ni ningún campo derivado de una sesión posterior. ``context`` es una carga opaca
que prepara el llamante (*feature store*, #19); el motor no construye *features*.

El ***gap*** **se mide y se publica; no se opera**
--------------------------------------------------

``gap_px[i] = open_px[i] / close_px[i-1] - 1``, con el cierre de la sesión **anterior de la
secuencia de entrada** (no del día natural anterior). ``gap_px[0]`` es ``None``
(declarado, nunca ``0``). El P&L **no** contiene ningún tramo ``close(i-1) -> open(i)``:
la entrada es exactamente ``open_px[i]`` (subasta de apertura: decisión del propietario del
2026-09-18, #61/#64) y la salida cae **siempre** dentro de la sesión ``i``.

Entrada, salida y empate
------------------------

Con barreras, se recorre el camino intradía (``bars`` en orden si existen; si no, el rango
diario ``[low_px, high_px]``) y gana la **primera** barrera tocada, al **precio de la
barrera** (sin redondear). Si en la misma barra se tocan las dos, gana la **adversa**
(``stop_px``): la misma regla conservadora que declara #10. Sin barreras, o si ninguna se
toca, la salida es ``close_px`` (``exit_reason = "session_close"``). El salto a través del
*stop* y el diferencial por tramo y por tamaño **no** se modelan: son #66 (y #62).

Purga y embargo: **se publican, no se afirman**
-----------------------------------------------

El ``BacktestRun`` hace eco literal de ``plan_sha256``, ``purge_total``, ``embargo_total``,
``embargo_in_train_total``, ``exclusions_are_no_op`` y ``uncovered`` del ``SplitPlan``.
Con el horizonte de #10 (``label_horizon = 0``) la purga y el embargo son **no-ops
estructurales** (el train es estrictamente anterior al test) y aquí **no** se presentan
como un filtro activo. La frontera es #12 (que los mide) y #67 (CPCV, el esquema donde sí
trabajan); la reserva del *holdout* es #68.

Determinismo, pureza y LLM
--------------------------

La salida es reproducible **byte a byte**: el informe se serializa de forma canónica
(``RUN_HASH_FORMAT``) y ``run_sha256`` es el sha256 de ese texto. El módulo es puro: sin
I/O, sin red, sin reloj, sin azar, sin estado global mutable y sin almacén. El LLM **nunca**
entra en este camino y el informe declara ``llm_overlay: "disabled"`` —declarado, **no**
configurable (``tech_stack.md`` §4.9).

Frontera con el adaptador de almacén
------------------------------------

Leer la historia del ``Store`` (con ``store.sql()``; para un estudio histórico
``read_pit`` devuelve vacío), construir los ``SessionInput`` y el ``context`` y escribir el
informe en ``data/derived/reports/`` es el **adaptador fino (#69)**, más su CLI y el
``--as-of``. Sin ese adaptador, este motor solo se ejercita con entradas sintéticas. La
tensión del enunciado («función pura y determinista» que a la vez obtiene el *snapshot*
*point-in-time*) se resuelve así: el **núcleo** es puro y recibe los datos ya preparados.

Dudas declaradas, no resueltas por cuenta propia: el ``R`` del *slippage* supuesto (#60), la
medición real del *slippage* (#62), el *gap* a través del *stop* y el diferencial por tramo
(#66) y la sigma con información del futuro (#63).

Unidades
--------

``gross_pct``, ``pnl_declared_pct`` y ``pnl_net_pct`` son **fracciones del nocional**
(``exit/entry - 1``), no puntos porcentuales: es la misma unidad que ``R``. El coste declarado
llega de #11 en **puntos porcentuales** (``c_declared_pct``: ``0,0042`` son ``0,0042 %``) y se
convierte a fracción con la **única fuente con nombre** ``c_fraction_of_notional`` de
``backtest.costs`` (``c_declared_pct / 100``), que el motor cita en vez de repetir la
división. Quien publica en puntos porcentuales multiplica por 100 **en su propio consumidor**;
la unidad del motor no cambia para contentar a un consumidor. El sufijo ``_pct`` de estos
campos se conserva: renombrarlos es #91.

Convención de tipos y aritmética
--------------------------------

Los importes y los porcentajes de coste siguen en ``decimal.Decimal`` **tal cual** los
devuelve #11 (se copian por identidad, nunca se re-redondean) y se publican como cadenas
decimales exactas. Los precios y el P&L simulado son ``float``, documentados como
aproximación. El bloque exacto de #11 viaja aparte (``SessionOutcome.cost``), sin mezclarse
con el P&L.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Final, cast

from cfdtrader.backtest.costs import (
    CostBreakdown,
    CostModel,
    FinancingCut,
    MeasureState,
    Side,
    SlippageParameter,
    cost_breakdown,
)
from cfdtrader.backtest.splits import SplitPlan

__all__ = [
    "ENGINE_DOES_NOT_DO",
    "FOLLOW_UPS",
    "LIMITATIONS",
    "RUN_HASH_FORMAT",
    "BacktestRun",
    "Bar",
    "Decision",
    "DecisionError",
    "DecisionFn",
    "Direction",
    "EmptyTestSetError",
    "EngineError",
    "EngineInputError",
    "FoldOutcome",
    "SessionInput",
    "SessionOutcome",
    "SessionView",
    "canonical_text",
    "run_walk_forward",
]

#: Formato estable del ``run_sha256`` (A28). Se hashea el **informe** serializado de forma
#: canónica: JSON con claves ordenadas, separadores ``,`` y ``:``, ``Decimal`` como cadena
#: decimal exacta, ``float`` vía ``repr`` y UTF-8. El informe que se hashea **no** incluye
#: la clave ``run_sha256`` (la añade ``BacktestRun.published_report``): un informe no se
#: hashea a sí mismo. Mismas entradas dan el mismo hash y cambiar un precio, el plan, el
#: modelo de coste, el *slippage* o una decisión lo cambia.
RUN_HASH_FORMAT: Final[str] = (
    "sha256(UTF-8) de json.dumps(informe, sort_keys=True, separators=(',', ':'), "
    "ensure_ascii=False), con Decimal como cadena decimal exacta y float via repr; el "
    "informe que se hashea no incluye la clave run_sha256"
)

#: Estados de una sesión evaluada (A12). Una por sesión de *test*, nunca descartada.
STATUS_TRADED: Final[str] = "traded"
STATUS_NO_TRADE: Final[str] = "no_trade"
STATUS_SKIPPED: Final[str] = "skipped"

#: Motivos de salida (A22).
EXIT_TARGET: Final[str] = "target"
EXIT_STOP: Final[str] = "stop"
EXIT_SESSION_CLOSE: Final[str] = "session_close"

#: Motivo de salto (A13): el ``SessionInput`` no se puede simular.
SKIP_MISSING_PRICES: Final[str] = "missing_prices"

#: Qué **no** hace el módulo, legible por máquina (A31). Cada frontera con su issue.
ENGINE_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_es_14",
        "issue": "#14",
        "statement": (
            "no implementa los baselines triviales (no operar, siempre largo/corto, "
            "momentum 5d, reversion de gap, regla aleatoria): solo ejecuta la funcion de "
            "decision que se le pasa"
        ),
    },
    {
        "id": "no_calcula_metricas",
        "issue": "#15",
        "statement": (
            "no calcula metricas netas (Sharpe/Sortino, EV, drawdown, intervalos), ni la "
            "curva de equity, ni ninguna agregacion: devuelve la serie por sesion"
        ),
    },
    {
        "id": "no_pbo_ni_deflated_sharpe",
        "issue": "#16",
        "statement": (
            "no calcula PBO ni Deflated Sharpe y no registra experimentos en runs/<hash>/: "
            "consume las particiones y produce la serie de resultados"
        ),
    },
    {
        "id": "no_es_la_suite_de_integridad",
        "issue": "#17",
        "statement": (
            "no es la suite de integridad (no-look-ahead sobre features, golden dataset, "
            "determinismo del gate): aqui solo se prueba este motor"
        ),
    },
    {
        "id": "no_es_el_informe_de_fase_1",
        "issue": "#18",
        "statement": (
            "no es el informe de Fase 1 ni su puerta de salida: publica sus limitaciones "
            "declaradas, no un veredicto"
        ),
    },
    {
        "id": "no_construye_features",
        "issue": "#19",
        "statement": (
            "no construye ni versiona la matriz de features: recibe el context ya "
            "preparado y lo trata como carga opaca"
        ),
    },
    {
        "id": "no_entrena",
        "issue": "#24",
        "statement": (
            "no entrena ningun modelo sobre las particiones: recibe decide_by_fold, una "
            "funcion por fold, y no la inspecciona"
        ),
    },
    {
        "id": "no_calibra",
        "issue": "#25",
        "statement": "no calibra probabilidades dentro del esquema de purga",
    },
    {
        "id": "no_decide_el_gate_ni_el_sizing",
        "issue": "#27",
        "statement": (
            "no implementa el gate de decision (tier, reglas duras 1-18) ni el sizing: "
            "Decision es un contrato minimo que el gate tendra que cumplir"
        ),
    },
    {
        "id": "no_es_el_pipeline_completo",
        "issue": "#28",
        "statement": (
            "no es el backtest del pipeline completo (features + modelo + gate + motor "
            "sobre el historico real)"
        ),
    },
    {
        "id": "no_tiene_intradia_real",
        "issue": "#50",
        "statement": (
            "no trae la fuente intradia real ni el bid/ask del SPX500:CFD: el diferencial "
            "es el declarado y simetrico de #11 y los precios son proxies"
        ),
    },
    {
        "id": "no_decide_r",
        "issue": "#60",
        "statement": (
            "no decide los umbrales ni el tamano de R: los consume; por eso el slippage "
            "supuesto no se puede cobrar y pnl_net_pct queda en None"
        ),
    },
    {
        "id": "no_mide_slippage",
        "issue": "#62",
        "statement": (
            "no mide el slippage real (10-15 ejecuciones en la apertura): lo recibe como "
            "parametro y publica su estado literal"
        ),
    },
    {
        "id": "no_arregla_la_sigma",
        "issue": "#63",
        "statement": (
            "no arregla la sigma con informacion del futuro heredada de la seleccion de "
            "candidato de #7"
        ),
    },
    {
        "id": "no_modela_el_gap_ni_el_diferencial_por_tramo",
        "issue": "#66",
        "statement": (
            "no modela el salto a traves del stop ni el diferencial por tramo de sesion y "
            "por tamano: cobra el diferencial declarado constante en dos mitades iguales"
        ),
    },
    {
        "id": "no_usa_cpcv",
        "issue": "#67",
        "statement": (
            "no implementa CPCV (el esquema donde la purga y el embargo hacen trabajo "
            "real): recorre los folds del plan de #12 tal cual"
        ),
    },
    {
        "id": "no_reserva_holdout",
        "issue": "#68",
        "statement": (
            "no reserva ni toca el holdout final intocable: solo evalua las sesiones de "
            "test que le entrega el plan"
        ),
    },
    {
        "id": "no_lee_el_almacen",
        "issue": "#69",
        "statement": (
            "no lee el Store (con store.sql()), ni construye los SessionInput/context, ni "
            "escribe el informe: eso es el adaptador fino #69, fuera de este modulo"
        ),
    },
    {
        "id": "no_calcula_costes",
        "issue": "#11",
        "statement": (
            "no calcula diferencial, tenencia, divisa ni comision: cobra con una sola "
            "llamada a cost_breakdown por operacion y copia su resultado por identidad"
        ),
    },
)

#: Seguimientos abiertos que este modulo deja declarados (A31/A33).
FOLLOW_UPS: Final[tuple[dict[str, str], ...]] = (
    {
        "issue": "#14",
        "topic": "baselines triviales",
        "why": "el motor solo ejecuta la funcion de decision que le pasen",
    },
    {
        "issue": "#15",
        "topic": "metricas netas, curva de equity y agregacion",
        "why": "aqui se devuelve la serie por sesion, no un rendimiento del sistema",
    },
    {
        "issue": "#16",
        "topic": "PBO, Deflated Sharpe y registro en runs/<hash>/",
        "why": "se calculan sobre las particiones, no dentro del motor",
    },
    {
        "issue": "#17",
        "topic": "suite de integridad",
        "why": "aqui solo se prueba este motor, no el pipeline entero",
    },
    {
        "issue": "#18",
        "topic": "informe de Fase 1 y su puerta de salida",
        "why": "el informe del motor no es el veredicto de la fase",
    },
    {
        "issue": "#19",
        "topic": "feature store",
        "why": "el context llega ya preparado y versionado; aqui es opaco",
    },
    {
        "issue": "#24",
        "topic": "entrenamiento del modelo",
        "why": "el motor recibe decide_by_fold y no entrena",
    },
    {
        "issue": "#25",
        "topic": "calibracion",
        "why": "no entra en el esquema de purga de este motor",
    },
    {
        "issue": "#27",
        "topic": "gate de decision y sizing",
        "why": "de ahi sale el nocional explicito que este motor consume",
    },
    {
        "issue": "#28",
        "topic": "backtest del pipeline completo",
        "why": "features + modelo + gate + motor sobre el historico real",
    },
    {
        "issue": "#50",
        "topic": "fuente intradia y bid/ask real del SPX500:CFD",
        "why": "traeria el diferencial asimetrico y los precios de ejecucion reales",
    },
    {
        "issue": "#57",
        "topic": "especificaciones del respaldo diario y de la muestra intradia",
        "why": "de ahi sale la proporcion de sesiones con camino intradia real",
    },
    {
        "issue": "#59",
        "topic": "broker definitivo",
        "why": "de ahi salen la comision real, el diferencial real y el corte de financiacion",
    },
    {
        "issue": "#60",
        "topic": "umbrales y tamano de R",
        "why": "sin R decidido el supuesto de slippage no se puede cobrar",
    },
    {
        "issue": "#62",
        "topic": "medir el slippage real",
        "why": "es lo que convierte el supuesto de #64 en una medicion",
    },
    {
        "issue": "#63",
        "topic": "sigma con informacion del futuro",
        "why": "heredada de la seleccion de candidato de #7",
    },
    {
        "issue": "#66",
        "topic": "diferencial por tramo y por tamano, y gap a traves del stop",
        "why": "hoy se cobra el diferencial declarado constante y simetrico",
    },
    {
        "issue": "#67",
        "topic": "CPCV",
        "why": "el esquema donde la purga y el embargo hacen trabajo real",
    },
    {
        "issue": "#68",
        "topic": "reserva y politica del holdout final intocable",
        "why": "`plan.md` §11.4 y §21 pregunta 10",
    },
    {
        "issue": "#69",
        "topic": "adaptador fino Store -> SessionInput + CLI + informe",
        "why": "sin el, el motor solo se ejercita con entradas sinteticas",
    },
)

#: Limitaciones que el informe publica (A32). No se esconden.
LIMITATIONS: Final[tuple[str, ...]] = (
    '**La puerta de Fase 0 esta en `fail`** (`gate: "fail"`, `phase1_ready: false`) y **los '
    "numeros de este arnes no son una validacion de la estrategia** (#9/#64): el motor "
    "entrega la serie de resultados, no un veredicto.",
    "**Los costes son declarados y no medidos**: la tabla de `plan.md` §3.3 reproducida por "
    "#8; el broker definitivo sigue sin decidir (#59), y de el salen la comision real, el "
    "diferencial real y la hora de corte.",
    "**El *slippage* es un supuesto pesimista declarado** (#64), **no** una medicion: "
    "mientras no se mida (#62) y `R` no este decidido (#60), `c_total_pct` es `null` y "
    "`pnl_net_pct` tambien. Su equivalente ilustrativo, si se publica, va etiquetado "
    "`illustrative` con `decision: false` y **nunca** alimenta el P&L.",
    "**El corte de financiacion sigue sin verificar** (#8/#59) y asumir una hora de corte "
    "fija esta prohibido: el motor no lo deduce de ningun *timestamp*.",
    "**Los precios son *proxies*** de `^GSPC` y no del `SPX500:CFD` (#50), y la muestra de "
    "intradia real es corta: la mayoria de las sesiones usan el respaldo diario (#10, #57).",
    "**La sigma con *look-ahead*** heredada de la seleccion de candidato de #7 sigue sin "
    "arreglar (#63), asi que las barreras de #10 pueden estar informadas por el futuro.",
    "**El *gap* a traves del *stop* no se modela** y el diferencial se cobra constante y "
    "simetrico (dos mitades iguales): el ensanchamiento por tramo y por tamano es #66.",
    "**La purga y el embargo son no-ops estructurales** con `label_horizon = 0` (#12): se "
    "publican con sus numeros. El esquema donde si trabajan es CPCV (#67) y el *holdout* "
    "final intocable es #68.",
    '**El overlay LLM va deshabilitado por declaracion** (`llm_overlay: "disabled"`, no '
    "configurable): el LLM no calcula ni decide en el camino critico (`tech_stack.md` §4.9).",
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados (A1, A26): nunca un resultado silencioso
# ─────────────────────────────────────────────────────────────────────────────
class EngineError(Exception):
    """Raiz de los errores del motor *walk-forward*."""


class EngineInputError(EngineError):
    """Las entradas del motor no son admisibles (A26)."""


class EmptyTestSetError(EngineInputError):
    """Un fold del plan trae un *test* vacio: el motor nunca devuelve un fold sin resultados."""


class DecisionError(EngineError):
    """La funcion de decision incumplio el contrato (A19, A23) o no devolvio una ``Decision``."""


# ─────────────────────────────────────────────────────────────────────────────
# Direccion (A3): el lado del motor; el `Side` de #11 se deriva de aqui
# ─────────────────────────────────────────────────────────────────────────────
class Direction(StrEnum):
    """Direccion declarada por la funcion de decision: ``LONG``, ``SHORT`` o ``NOTHING``."""

    LONG = "long"
    SHORT = "short"
    NOTHING = "nothing"


# ─────────────────────────────────────────────────────────────────────────────
# Contratos de entrada y de decision (A3)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Bar:
    """Una barra del camino intradia: solo el rango ``[low_px, high_px]``.

    El motor **no** deduce nada de una duracion ni de una hora: las barras llegan en el
    orden en que se recorren. Un ``high_px`` menor que ``low_px`` es un error tipado (A26).
    """

    high_px: float
    low_px: float


@dataclass(frozen=True, slots=True)
class SessionInput:
    """Una sesion lista para simular, preparada por el llamante (adaptador #69).

    ``session`` ordena la secuencia (estrictamente creciente, sin duplicados). Los precios
    son ``float`` y pueden faltar (``None``): entonces la sesion se registra como
    ``skipped`` con su motivo, nunca se descarta. ``bars`` es el camino intradia ordenado;
    ``None`` significa respaldo diario con el rango ``[low_px, high_px]``. ``context`` es
    una carga opaca para la funcion de decision (no la mira este motor).
    """

    session: date
    open_px: float | None
    high_px: float | None
    low_px: float | None
    close_px: float | None
    context: object | None = None
    bars: Sequence[Bar] | None = None


@dataclass(frozen=True, slots=True)
class SessionView:
    """Lo **unico** que ve la funcion de decision: la sesion en curso, sin su futuro.

    No expone ``high_px``, ``low_px`` ni ``close_px`` de la sesion en curso (son el futuro
    de esa sesion) ni dato alguno de sesiones posteriores. ``gap_px`` se mide con el cierre
    de la sesion **anterior de la secuencia** y vale ``None`` en la primera posicion.
    """

    session: date
    open_px: float | None
    gap_px: float | None
    context: object | None


@dataclass(frozen=True, slots=True)
class Decision:
    """El contrato minimo del gate (#27): direccion, motivo y, si se opera, el resto.

    ``notional_usd`` es **explicito y sin valor por defecto**: una decision ``LONG`` o
    ``SHORT`` sin un ``Decimal`` positivo es un ``DecisionError`` (A19). Con barreras, se
    declaran **las dos o ninguna** y su geometria se valida (A23).
    """

    direction: Direction
    reason: str
    stop_px: float | None = None
    target_px: float | None = None
    notional_usd: Decimal | None = None
    probability: float | None = None


#: Una funcion de decision por fold (A4): recibe la vista y devuelve una ``Decision``.
DecisionFn = Callable[[SessionView], Decision]


# ─────────────────────────────────────────────────────────────────────────────
# Resultados (A12, A17, A22): una `SessionOutcome` por sesion de *test*
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SessionOutcome:
    """El resultado de **una** sesion de *test*, con su estado y su coste exacto.

    ``status`` es ``traded``, ``no_trade`` o ``skipped``. ``reason`` es el motivo del gate
    (nunca se descarta en silencio) y ``skip_reason`` solo lo usa ``skipped``. ``cost`` es
    el ``CostBreakdown`` de #11 **tal cual** (por identidad) y ``pnl_net_pct`` es ``None``
    mientras el *slippage* no sea medido, con el motivo en ``pnl_net_reason``.
    """

    fold_index: int
    session: date
    session_index: int
    status: str
    reason: str | None
    skip_reason: str | None
    gap_px: float | None
    decision: Decision | None
    entry_session: date | None
    exit_session: date | None
    entry_px: float | None
    exit_px: float | None
    exit_reason: str | None
    exit_bar_index: int | None
    notional_usd: Decimal | None
    gross_pct: float | None
    pnl_declared_pct: float | None
    pnl_net_pct: float | None
    pnl_net_reason: str | None
    cost: CostBreakdown | None


@dataclass(frozen=True, slots=True)
class FoldOutcome:
    """Un fold del plan y sus sesiones de *test*, en orden ascendente (A11)."""

    index: int
    test_start: int
    test_stop: int
    sessions: tuple[SessionOutcome, ...]
    traded: int
    no_trade: int
    skipped: int


@dataclass(frozen=True, slots=True)
class BacktestRun:
    """La corrida completa: folds, eco del plan, recuentos, informe y ``run_sha256``.

    ``report`` es el informe canonico (el texto que se hashea, segun ``RUN_HASH_FORMAT``) y
    **no** incluye la clave ``run_sha256``; ``published_report()`` la anade. El eco del plan
    (A8) es literal: purga y embargo se publican, no se afirman.
    """

    folds: tuple[FoldOutcome, ...]
    plan_sha256: str
    purge_total: int
    embargo_total: int
    embargo_in_train_total: int
    exclusions_are_no_op: bool
    uncovered: tuple[int, ...]
    not_in_any_test: int
    n_sessions: int
    traded: int
    no_trade: int
    skipped: int
    run_sha256: str
    report: dict[str, object]

    def published_report(self) -> dict[str, object]:
        """El informe canonico mas su ``run_sha256`` (que no entra en el propio hash)."""
        return {**self.report, "run_sha256": self.run_sha256}


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades exactas (A28): cadenas decimales exactas y serializacion canonica
# ─────────────────────────────────────────────────────────────────────────────
def _decimal_text(value: Decimal) -> str:
    """``Decimal`` -> cadena decimal **exacta**, sin notacion cientifica."""
    return format(value, "f")


def _optional_decimal_text(value: Decimal | None) -> str | None:
    """``None`` sigue siendo ``None``: un valor no medido nunca se escribe como ``0``."""
    return None if value is None else _decimal_text(value)


def _compact_decimal_text(value: Decimal) -> str:
    """Cadena decimal exacta sin ceros de relleno: la convencion de #8 (20 bp, no 20,0)."""
    text = format(value.normalize(), "f")
    return "0" if text in {"-0", "0"} else text


def _date_text(value: date | None) -> str | None:
    """Una sesion se publica en ISO-8601; ``None`` sigue siendo ``None``."""
    return None if value is None else value.isoformat()


def _plain(value: object) -> object:
    """Traduce un informe a tipos JSON puros: ``Decimal`` como cadena exacta."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Decimal):
        return _decimal_text(value)
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        return {str(key): _plain(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        sequence = cast("list[object] | tuple[object, ...]", value)
        return [_plain(item) for item in sequence]
    raise EngineInputError(
        f"el informe solo admite tipos JSON, Decimal y secuencias; llego {type(value).__name__}"
    )


def canonical_text(report: Mapping[str, object]) -> str:
    """El texto canonico de un informe (A28): lo que se hashea para ``run_sha256``."""
    return json.dumps(
        _plain(dict(report)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _require_instance(value: object, expected: type[object], *, field: str) -> None:
    """Comprueba en tiempo de ejecucion el tipo declarado: un contrato roto es error tipado.

    Las anotaciones de esta API son el contrato; estas comprobaciones existen porque el
    llamante puede llegar sin tipar (un ``Any``) y A26 exige un error claro, nunca un
    resultado silencioso ni un ``AttributeError`` suelto.
    """
    if not isinstance(value, expected):
        raise EngineInputError(
            f"{field}: se espera {expected.__name__}, no {type(value).__name__} (A26)"
        )


def _price_or_none(value: object) -> float | None:
    """Un precio utilizable (numero finito y positivo) o ``None``: nunca se inventa un 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _touches(bar: Bar, level: float | None) -> bool:
    """La barra toca el nivel si esta dentro de su rango ``[low_px, high_px]``."""
    if level is None:
        return False
    return bar.low_px <= level <= bar.high_px


# ─────────────────────────────────────────────────────────────────────────────
# NUCLEO (A35): bucle + resolucion de la salida + llamada al coste
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_exit(
    *,
    decision: Decision,
    close_px: float | None,
    path: tuple[Bar, ...],
) -> tuple[float, str, int | None] | None:
    """Resuelve la salida (A21/A22); ``None`` si habria que cerrar y no hay cierre usable.

    Sin barreras, la salida es el cierre. Con barreras, se recorre el camino y gana la
    **primera** tocada, al **precio de la barrera**; en la misma barra gana la adversa
    (``stop_px``). Si no se toca ninguna, la salida vuelve a ser el cierre.
    """
    stop = decision.stop_px
    target = decision.target_px
    if stop is None and target is None:
        return None if close_px is None else (close_px, EXIT_SESSION_CLOSE, None)
    for index, bar in enumerate(path):
        if _touches(bar, stop):
            return (cast("float", stop), EXIT_STOP, index)
        if _touches(bar, target):
            return (cast("float", target), EXIT_TARGET, index)
    return None if close_px is None else (close_px, EXIT_SESSION_CLOSE, None)


def _gross_pct(*, entry_px: float, exit_px: float, direction: Direction) -> float:
    """P&L bruto simulado de la sesion: ``open`` -> salida, sin *overnight* ni *gap*."""
    factor = 1.0 if direction is Direction.LONG else -1.0
    return (exit_px / entry_px - 1.0) * factor


def _intraday_path(item: SessionInput) -> tuple[Bar, ...] | None:
    """El camino a recorrer: las ``bars`` si existen; si no, el rango diario (A21).

    Las ``bars`` mal declaradas (sin ``high``/``low`` utilizables, invertidas o vacias) son
    un error tipado (A26); el rango diario ausente o no positivo deja la sesion en
    ``skipped`` (A13), que es lo que el enunciado pide en cada caso.
    """
    if item.bars is not None:
        path = tuple(item.bars)
        if not path:
            raise EngineInputError("bars: el camino intradia llega vacio (A26)")
        for index, bar in enumerate(path):
            _require_instance(bar, Bar, field=f"bars[{index}]")
            high = _price_or_none(bar.high_px)
            low = _price_or_none(bar.low_px)
            if high is None or low is None:
                raise EngineInputError(
                    f"bars: la barra {index} no tiene high/low utilizables (A26)"
                )
            if high < low:
                raise EngineInputError(
                    f"bars: la barra {index} esta desordenada (high_px {high} < low_px {low}) (A26)"
                )
        return path
    high = _price_or_none(item.high_px)
    low = _price_or_none(item.low_px)
    if high is None or low is None or high < low:
        return None
    return (Bar(high_px=high, low_px=low),)


def _evaluate_session(
    *,
    index: int,
    fold_index: int,
    item: SessionInput,
    gap_px: float | None,
    decide: DecisionFn,
    cost_model: CostModel,
    slippage: SlippageParameter,
    financing_cut: FinancingCut | None,
) -> SessionOutcome:
    """Decide, simula y cobra **una** sesion de *test* (una llamada al decididor y una a #11)."""
    view = SessionView(
        session=item.session,
        open_px=item.open_px,
        gap_px=gap_px,
        context=item.context,
    )
    decision = decide(view)
    _require_instance(decision, Decision, field=f"decide_by_fold[{fold_index}]")
    if decision.direction is Direction.NOTHING:
        return _outcome(
            index=index,
            fold_index=fold_index,
            item=item,
            status=STATUS_NO_TRADE,
            reason=decision.reason,
            decision=decision,
            gap_px=gap_px,
        )
    open_px = _price_or_none(item.open_px)
    if open_px is None:
        return _skipped(
            index=index,
            fold_index=fold_index,
            item=item,
            decision=decision,
            gap_px=gap_px,
        )
    notional = _require_notional(decision)
    _require_geometry(decision=decision, open_px=open_px)
    path = _intraday_path(item)
    resolved = (
        None
        if path is None
        else _resolve_exit(decision=decision, close_px=_price_or_none(item.close_px), path=path)
    )
    if resolved is None:
        return _skipped(
            index=index,
            fold_index=fold_index,
            item=item,
            decision=decision,
            gap_px=gap_px,
        )
    exit_px, exit_reason, exit_bar_index = resolved
    gross_pct = _gross_pct(entry_px=open_px, exit_px=exit_px, direction=decision.direction)
    cost = cost_breakdown(
        model=cost_model,
        slippage=slippage,
        notional_usd=notional,
        side=_side_of(decision.direction),
        nights=0,
        overnight_reason=None,
        financing_cut=financing_cut,
    )
    # La unidad del motor es la **fraccion** del nocional (ver «Unidades» arriba): el coste de
    # #11 llega en % del nocional y se convierte con la unica fuente con nombre de ``costs``.
    pnl_declared_pct = gross_pct - float(cost.c_fraction_of_notional)
    pnl_net_pct = None if cost.c_total_pct is None else gross_pct - float(cost.c_total_pct) / 100
    return _outcome(
        index=index,
        fold_index=fold_index,
        item=item,
        status=STATUS_TRADED,
        reason=decision.reason,
        decision=decision,
        gap_px=gap_px,
        entry_px=open_px,
        exit_px=exit_px,
        exit_reason=exit_reason,
        exit_bar_index=exit_bar_index,
        notional_usd=notional,
        gross_pct=gross_pct,
        pnl_declared_pct=pnl_declared_pct,
        pnl_net_pct=pnl_net_pct,
        pnl_net_reason=_net_reason(cost),
        cost=cost,
    )


def _run_folds(
    *,
    items: tuple[SessionInput, ...],
    gaps: tuple[float | None, ...],
    split_plan: SplitPlan,
    cost_model: CostModel,
    slippage: SlippageParameter,
    decide_by_fold: Sequence[DecisionFn],
    financing_cut: FinancingCut | None,
) -> tuple[FoldOutcome, ...]:
    """Recorre los folds en orden y, dentro de cada uno, su *test* en orden ascendente (A11)."""
    folds: list[FoldOutcome] = []
    for fold_index, fold in enumerate(split_plan.folds):
        decide = decide_by_fold[fold_index]
        sessions = tuple(
            _evaluate_session(
                index=index,
                fold_index=fold_index,
                item=items[index],
                gap_px=gaps[index],
                decide=decide,
                cost_model=cost_model,
                slippage=slippage,
                financing_cut=financing_cut,
            )
            for index in fold.test
        )
        folds.append(
            FoldOutcome(
                index=fold.index,
                test_start=fold.test_start,
                test_stop=fold.test_stop,
                sessions=sessions,
                traded=_count(sessions, STATUS_TRADED),
                no_trade=_count(sessions, STATUS_NO_TRADE),
                skipped=_count(sessions, STATUS_SKIPPED),
            )
        )
    return tuple(folds)


# ─────────────────────────────────────────────────────────────────────────────
# Validacion (A26) y contrato de decision (A19, A23)
# ─────────────────────────────────────────────────────────────────────────────
def _validate(
    *,
    items: tuple[SessionInput, ...],
    split_plan: SplitPlan,
    cost_model: CostModel,
    slippage: SlippageParameter,
    decide_by_fold: Sequence[DecisionFn],
) -> None:
    """Rechaza con error tipado cualquier entrada inadmisible (A26). Nunca sigue en silencio."""
    if not items:
        raise EngineInputError("inputs: la secuencia de sesiones llega vacia (A26)")
    _require_instance(split_plan, SplitPlan, field="split_plan (de #12)")
    _require_instance(cost_model, CostModel, field="cost_model (de #11)")
    _require_instance(slippage, SlippageParameter, field="slippage (de #11)")
    if split_plan.n_sessions != len(items):
        raise EngineInputError(
            f"split_plan.n_sessions ({split_plan.n_sessions}) != len(inputs) ({len(items)}): "
            "el plan de #12 se consume tal cual y no se rehace aqui (A7/A26)"
        )
    if not split_plan.folds:
        raise EngineInputError("split_plan.folds: el plan no trae ningun fold (A26)")
    for index, item in enumerate(items):
        _require_instance(item, SessionInput, field=f"inputs[{index}]")
        _require_instance(item.session, date, field=f"inputs[{index}].session")
    for index in range(len(items) - 1):
        if items[index].session >= items[index + 1].session:
            raise EngineInputError(
                f"inputs: las sesiones deben ser estrictamente crecientes y sin duplicados; "
                f"se rompe en la posicion {index + 1} (A26)"
            )
    if len(decide_by_fold) != len(split_plan.folds):
        raise EngineInputError(
            f"decide_by_fold: hacen falta exactamente {len(split_plan.folds)} funciones (una por "
            f"fold), no {len(decide_by_fold)} (A4/A26)"
        )
    for fold_index, fold in enumerate(split_plan.folds):
        if not callable(decide_by_fold[fold_index]):
            raise EngineInputError(f"decide_by_fold[{fold_index}]: no es invocable (A26)")
        if not fold.test:
            raise EmptyTestSetError(
                f"el fold {fold_index} trae un test vacio (test_start={fold.test_start}, "
                f"test_stop={fold.test_stop}): el motor nunca devuelve un fold sin resultados (A10)"
            )


def _require_notional(decision: Decision) -> Decimal:
    """El nocional es explicito y **sin valor por defecto** (A19): lo decide el gate (#27)."""
    value = decision.notional_usd
    if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
        raise DecisionError(
            "decision.notional_usd: una decision LONG/SHORT exige un Decimal positivo explicito "
            f"(llego {value!r}); el motor no decide R, ni el tamano, ni el apalancamiento (A19)"
        )
    return value


def _require_geometry(*, decision: Decision, open_px: float) -> None:
    """Valida la geometria de la decision (A23) y que las barreras vengan las dos o ninguna."""
    stop = _price_or_none(decision.stop_px)
    target = _price_or_none(decision.target_px)
    if (stop is None) != (target is None):
        raise DecisionError(
            "decision.stop_px/decision.target_px: las barreras se declaran las dos o ninguna; "
            "una sola no resuelve la salida de forma declarada (A23)"
        )
    if stop is None or target is None:
        return
    if decision.direction is Direction.LONG and not stop < open_px < target:
        raise DecisionError(
            f"decision LONG: la geometria exige stop_px < open_px < target_px; llego "
            f"stop_px={stop}, open_px={open_px}, target_px={target} (A23)"
        )
    if decision.direction is Direction.SHORT and not target < open_px < stop:
        raise DecisionError(
            f"decision SHORT: la geometria exige target_px < open_px < stop_px; llego "
            f"target_px={target}, open_px={open_px}, stop_px={stop} (A23)"
        )


def _side_of(direction: Direction) -> Side:
    """El ``Side`` de #11 se deriva de la direccion; ``NOTHING`` no llega hasta aqui."""
    return Side.LONG if direction is Direction.LONG else Side.SHORT


def _count(sessions: tuple[SessionOutcome, ...], status: str) -> int:
    """Cuenta las sesiones con ese estado: los tres recuentos se publican (A12)."""
    return sum(1 for outcome in sessions if outcome.status == status)


def _net_reason(cost: CostBreakdown) -> str | None:
    """El motivo por el que ``pnl_net_pct`` es ``None`` (A18): nunca se sustituye por ``0``."""
    if cost.c_total_pct is not None:
        return None
    return (
        f"slippage.state = '{cost.slippage.state.value}': sin un % del nocional que cobrar, "
        f"c_total_pct es null y pnl_net_pct tambien (nunca 0). Motivo declarado: "
        f"{cost.slippage.reason}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Resultados: constructores y bloques del informe
# ─────────────────────────────────────────────────────────────────────────────
def _outcome(
    *,
    index: int,
    fold_index: int,
    item: SessionInput,
    status: str,
    reason: str | None,
    gap_px: float | None,
    decision: Decision | None = None,
    skip_reason: str | None = None,
    entry_px: float | None = None,
    exit_px: float | None = None,
    exit_reason: str | None = None,
    exit_bar_index: int | None = None,
    notional_usd: Decimal | None = None,
    gross_pct: float | None = None,
    pnl_declared_pct: float | None = None,
    pnl_net_pct: float | None = None,
    pnl_net_reason: str | None = None,
    cost: CostBreakdown | None = None,
) -> SessionOutcome:
    """Construye la ``SessionOutcome`` de una sesion; solo ``traded`` tiene sesion de entrada."""
    traded = status == STATUS_TRADED
    return SessionOutcome(
        fold_index=fold_index,
        session=item.session,
        session_index=index,
        status=status,
        reason=reason,
        skip_reason=skip_reason,
        gap_px=gap_px,
        decision=decision,
        entry_session=item.session if traded else None,
        exit_session=item.session if traded else None,
        entry_px=entry_px,
        exit_px=exit_px,
        exit_reason=exit_reason,
        exit_bar_index=exit_bar_index,
        notional_usd=notional_usd,
        gross_pct=gross_pct,
        pnl_declared_pct=pnl_declared_pct,
        pnl_net_pct=pnl_net_pct,
        pnl_net_reason=pnl_net_reason,
        cost=cost,
    )


def _skipped(
    *,
    index: int,
    fold_index: int,
    item: SessionInput,
    decision: Decision,
    gap_px: float | None,
) -> SessionOutcome:
    """La sesion no se puede simular: la decision y el motivo se conservan (A13)."""
    return _outcome(
        index=index,
        fold_index=fold_index,
        item=item,
        status=STATUS_SKIPPED,
        reason=decision.reason,
        decision=decision,
        gap_px=gap_px,
        skip_reason=SKIP_MISSING_PRICES,
    )


def _gaps(items: tuple[SessionInput, ...]) -> tuple[float | None, ...]:
    """``gap_px[i] = open_px[i] / close_px[i-1] - 1`` sobre la secuencia de entrada (A6)."""
    gaps: list[float | None] = []
    for index, item in enumerate(items):
        open_px = _price_or_none(item.open_px)
        previous = _price_or_none(items[index - 1].close_px) if index else None
        gaps.append(None if open_px is None or previous is None else open_px / previous - 1.0)
    return tuple(gaps)


def _model_block(model: CostModel) -> dict[str, object]:
    """El modelo de coste de #11 con sus cifras exactas: cambiarlo cambia el hash (A28)."""
    return {
        "name": model.name,
        "spread_entry_pct": _decimal_text(model.spread_entry_pct),
        "spread_exit_pct": _decimal_text(model.spread_exit_pct),
        "carry_long_pct_per_night": _decimal_text(model.carry_long_pct_per_night),
        "carry_short_pct_per_night": _decimal_text(model.carry_short_pct_per_night),
        "fx_pct": _decimal_text(model.fx_pct),
        "commission_pct": _decimal_text(model.commission_pct),
    }


def _slippage_block(slippage: SlippageParameter) -> dict[str, object]:
    """El *slippage* con su estado **literal**: los tres estados no se fusionan (A18)."""
    block: dict[str, object] = {
        "state": slippage.state.value,
        "is_measurement": slippage.is_measurement,
        "pct_of_notional": _optional_decimal_text(slippage.pct_of_notional),
        "pct_of_r": _optional_decimal_text(slippage.pct_of_r),
        "r_pct": _optional_decimal_text(slippage.r_pct),
        "source": slippage.source,
        "reason": slippage.reason,
        "decided_on": slippage.decided_on,
        "follow_up_issue": slippage.follow_up_issue,
        "illustrative_equivalence": None,
    }
    if slippage.state is MeasureState.ASSUMED and slippage.pct_of_r is not None:
        block["illustrative_equivalence"] = _illustrative_block(slippage.pct_of_r)
    return block


#: `R` **ilustrativo** con el que #8/#64 publica el equivalente en bp del supuesto: **no** es
#: una decision. El tamano de `R` es la decision abierta 5 -> #60 y viaja como `null`.
_ILLUSTRATIVE_R_PCT: Final[Decimal] = Decimal("1")


def _illustrative_block(pct_of_r: Decimal) -> dict[str, object]:
    """La equivalencia **ilustrativa** del supuesto: etiquetada y **fuera** del P&L (A18)."""
    pct = pct_of_r * _ILLUSTRATIVE_R_PCT / Decimal(100)
    return {
        "label": "illustrative",
        "decision": False,
        "basis": "supuesto declarado de #64: 20 % de R",
        "r_illustrative_pct": _decimal_text(_ILLUSTRATIVE_R_PCT),
        "equivalent_pct_of_notional": _compact_decimal_text(pct),
        "equivalent_bp_of_notional": _compact_decimal_text(pct * Decimal(100)),
        "feeds_pnl_net_pct": False,
    }


def _financing_cut_block(cut: FinancingCut | None) -> dict[str, object]:
    """El corte **declarado** de financiacion: hoy sin verificar (lo dice #11, ver A32)."""
    effective = cut if cut is not None else FinancingCut.unverified()
    return {
        "state": effective.state.value,
        "cut_et": None if effective.cut_et is None else effective.cut_et.isoformat(),
        "source": effective.source,
        "reason": effective.reason,
        "note": effective.note,
    }


def _cost_block(cost: CostBreakdown) -> dict[str, object]:
    """El bloque exacto de #11, con importes como cadenas decimales y sus ``null`` visibles."""
    return {
        "side": cost.side.value,
        "nights": cost.nights,
        "overnight_reason": cost.overnight_reason,
        "notional_usd": _decimal_text(cost.notional_usd),
        "spread_entry_pct": _decimal_text(cost.spread_entry_pct),
        "spread_exit_pct": _decimal_text(cost.spread_exit_pct),
        "spread_entry_usd": _decimal_text(cost.spread_entry_usd),
        "spread_exit_usd": _decimal_text(cost.spread_exit_usd),
        "spread_pct": _decimal_text(cost.spread_pct),
        "spread_usd": _decimal_text(cost.spread_usd),
        "carry_pct": _decimal_text(cost.carry_pct),
        "carry_usd": _decimal_text(cost.carry_usd),
        "carry_state": cost.carry_state.value,
        "carry_reason": cost.carry_reason,
        "fx_pct": _decimal_text(cost.fx_pct),
        "commission_pct": _decimal_text(cost.commission_pct),
        "c_declared_pct": _decimal_text(cost.c_declared_pct),
        "c_declared_usd": _decimal_text(cost.c_declared_usd),
        "slippage_state": cost.slippage.state.value,
        "slippage_pct": _optional_decimal_text(cost.slippage_pct),
        "c_total_pct": _optional_decimal_text(cost.c_total_pct),
        "c_total_usd": _optional_decimal_text(cost.c_total_usd),
        "financing_cut_state": cost.financing_cut.state.value,
        "nulls": [dict(item) for item in cost.nulls],
    }


def _session_block(outcome: SessionOutcome) -> dict[str, object]:
    """Una sesion del informe: estado, decision, P&L simulado y el coste exacto de #11."""
    decision = outcome.decision
    return {
        "fold": outcome.fold_index,
        "index": outcome.session_index,
        "session": outcome.session.isoformat(),
        "status": outcome.status,
        "reason": outcome.reason,
        "skip_reason": outcome.skip_reason,
        "gap_px": outcome.gap_px,
        "direction": None if decision is None else decision.direction.value,
        "stop_px": None if decision is None else decision.stop_px,
        "target_px": None if decision is None else decision.target_px,
        "probability": None if decision is None else decision.probability,
        "notional_usd": _optional_decimal_text(outcome.notional_usd),
        "entry_session": _date_text(outcome.entry_session),
        "exit_session": _date_text(outcome.exit_session),
        "entry_px": outcome.entry_px,
        "exit_px": outcome.exit_px,
        "exit_reason": outcome.exit_reason,
        "exit_bar_index": outcome.exit_bar_index,
        "gross_pct": outcome.gross_pct,
        "pnl_declared_pct": outcome.pnl_declared_pct,
        "pnl_net_pct": outcome.pnl_net_pct,
        "pnl_net_reason": outcome.pnl_net_reason,
        "cost": None if outcome.cost is None else _cost_block(outcome.cost),
    }


def _fold_block(fold: FoldOutcome) -> dict[str, object]:
    """Un fold del informe, con sus sesiones en orden ascendente (A11)."""
    return {
        "index": fold.index,
        "test_start": fold.test_start,
        "test_stop": fold.test_stop,
        "traded": fold.traded,
        "no_trade": fold.no_trade,
        "skipped": fold.skipped,
        "sessions": [_session_block(outcome) for outcome in fold.sessions],
    }


def _report(
    *,
    folds: tuple[FoldOutcome, ...],
    split_plan: SplitPlan,
    cost_model: CostModel,
    slippage: SlippageParameter,
    financing_cut: FinancingCut | None,
    n_sessions: int,
    traded: int,
    no_trade: int,
    skipped: int,
    not_in_any_test: int,
) -> dict[str, object]:
    """El informe canonico (A25-A32): lo que se serializa y se hashea."""
    return {
        "engine": "cfdtrader.backtest.engine",
        "task": "#13",
        "llm_overlay": "disabled",
        "hash_format": RUN_HASH_FORMAT,
        "gate": "fail",
        "phase1_ready": False,
        "t0_t1": {
            "t0": "open de la subasta de apertura, decision del propietario 2026-09-18",
            "t1": "cierre de la misma sesion (close_utc de #10); en medias sesiones, su cierre",
            "label_horizon": 0,
            "overnight": False,
        },
        "look_ahead": {
            "view": "SessionView(session, open_px, gap_px, context)",
            "exposes_high_low_close": False,
            "note": "la vista no expone el futuro de la sesion en curso ni sesiones posteriores",
        },
        "cost_model": _model_block(cost_model),
        "slippage": _slippage_block(slippage),
        "financing_cut": _financing_cut_block(financing_cut),
        "plan": {
            "plan_sha256": split_plan.plan_sha256,
            "n_sessions": split_plan.n_sessions,
            "purge_total": split_plan.purge_total,
            "embargo_total": split_plan.embargo_total,
            "embargo_in_train_total": split_plan.embargo_in_train_total,
            "exclusions_are_no_op": split_plan.exclusions_are_no_op,
            "uncovered": list(split_plan.uncovered),
            "inputs": dict(split_plan.inputs),
            "exclusions_note": (
                "la purga y el embargo se publican, no se afirman: con label_horizon = 0 son "
                "no-ops estructurales (frontera con #12; el esquema donde trabajan es #67)"
            ),
        },
        "counts": {
            "n_sessions": n_sessions,
            "traded": traded,
            "no_trade": no_trade,
            "skipped": skipped,
            "not_in_any_test": not_in_any_test,
            "conservation": "n_sessions == traded + no_trade + skipped + not_in_any_test",
            "conservation_holds": True,
        },
        "folds": [_fold_block(fold) for fold in folds],
        "limitations": list(LIMITATIONS),
        "does_not_do": list(ENGINE_DOES_NOT_DO),
        "follow_ups": list(FOLLOW_UPS),
    }


# ─────────────────────────────────────────────────────────────────────────────
# API publica (A2)
# ─────────────────────────────────────────────────────────────────────────────
def run_walk_forward(
    inputs: Sequence[SessionInput],
    *,
    split_plan: SplitPlan,
    cost_model: CostModel,
    slippage: SlippageParameter,
    decide_by_fold: Sequence[DecisionFn],
    financing_cut: FinancingCut | None = None,
) -> BacktestRun:
    """Recorre los *test* del plan de #12, simula cada sesion y cobra con #11 (A2).

    Es una funcion **pura y determinista**: no lee ni escribe nada, no consulta el reloj, no
    usa azar ni estado global, y mismas entradas dan el mismo ``run_sha256``. ``inputs``
    llega ya preparado por el llamante (adaptador #69); ``decide_by_fold`` trae
    **exactamente** una funcion por fold y el motor la llama una vez por sesion de *test*.
    ``financing_cut`` es el corte declarado que se pasa a #11; si no se declara, #11 usa el
    suyo (**sin verificar**, nunca deducido por este motor).
    """
    items = tuple(inputs)
    _validate(
        items=items,
        split_plan=split_plan,
        cost_model=cost_model,
        slippage=slippage,
        decide_by_fold=decide_by_fold,
    )
    folds = _run_folds(
        items=items,
        gaps=_gaps(items),
        split_plan=split_plan,
        cost_model=cost_model,
        slippage=slippage,
        decide_by_fold=decide_by_fold,
        financing_cut=financing_cut,
    )
    traded = sum(fold.traded for fold in folds)
    no_trade = sum(fold.no_trade for fold in folds)
    skipped = sum(fold.skipped for fold in folds)
    not_in_any_test = len(split_plan.uncovered)
    accounted = traded + no_trade + skipped + not_in_any_test
    if accounted != len(items):
        raise EngineInputError(
            f"la identidad de conservacion no se cumple (A9): n_sessions={len(items)} != "
            f"traded ({traded}) + no_trade ({no_trade}) + skipped ({skipped}) + "
            f"not_in_any_test ({not_in_any_test}) = {accounted}. El plan de #12 no particiona "
            "la muestra en ese orden"
        )
    report = _report(
        folds=folds,
        split_plan=split_plan,
        cost_model=cost_model,
        slippage=slippage,
        financing_cut=financing_cut,
        n_sessions=len(items),
        traded=traded,
        no_trade=no_trade,
        skipped=skipped,
        not_in_any_test=not_in_any_test,
    )
    digest = hashlib.sha256(canonical_text(report).encode("utf-8")).hexdigest()
    return BacktestRun(
        folds=folds,
        plan_sha256=split_plan.plan_sha256,
        purge_total=split_plan.purge_total,
        embargo_total=split_plan.embargo_total,
        embargo_in_train_total=split_plan.embargo_in_train_total,
        exclusions_are_no_op=split_plan.exclusions_are_no_op,
        uncovered=split_plan.uncovered,
        not_in_any_test=not_in_any_test,
        n_sessions=len(items),
        traded=traded,
        no_trade=no_trade,
        skipped=skipped,
        run_sha256=digest,
        report=report,
    )
