"""Baselines triviales del arnés de backtest — tarea #14.

Frontera explícita: **el motor de #13 ejecuta y cobra; este módulo solo decide**. Aquí viven
los seis baselines de `plan.md` §11.2 como funciones **puras y deterministas** sobre los
contratos de #13: se resuelve la señal de cada sesión fuera del bucle y se entrega una
``DecisionFn`` **por fold** (la misma función en todos los folds, porque un baseline no
entrena) lista para ``run_walk_forward``. La única vía de ejecución sigue siendo la de #13,
que exige ``cost_model`` y ``slippage`` **sin valor por defecto**; este módulo **no** llama a
``cost_breakdown`` ni construye un ``CostBreakdown``, así que ningún baseline puede producir
un resultado sin coste (A8) y un *slippage* en estado ``assumed`` deja ``pnl_net_pct`` en
``null`` con su motivo, nunca en 0 (A10).

Los seis baselines (A2)
-----------------------

+-----------------+---------------------------------------------------------------+
| identificador   | regla                                                         |
+=================+===============================================================+
| ``no_trade``    | no operar: ``NOTHING`` explícito en todas las sesiones. La     |
|                 | convención «Sharpe = 0» de §11.2 se **declara** aquí y **no**  |
|                 | se calcula: las métricas netas son #15                         |
+-----------------+---------------------------------------------------------------+
| ``always_long`` | **listón A**: compra en el ``open`` de la subasta y sale en el |
|                 | cierre de la **misma** sesión, todos los días, sin barreras    |
+-----------------+---------------------------------------------------------------+
| ``always_short``| espejo del anterior                                            |
+-----------------+---------------------------------------------------------------+
| ``momentum_5d`` | largo si el retorno de los cinco días previos es positivo;     |
|                 | corto si es negativo; ``NOTHING`` si falta historia o hay      |
|                 | empate (que no se rompe)                                      |
+-----------------+---------------------------------------------------------------+
| ``gap_reversal``| sesgo **contrario** al *gap* de apertura que publica #13       |
|                 | (``SessionView.gap_px``, medido contra el cierre de la sesión  |
|                 | anterior de la secuencia)                                     |
+-----------------+---------------------------------------------------------------+
| ``random_matched`` | control de significación estadística: opera exactamente     |
|                 | ``floor(frequency · n_test)`` sesiones de test **distintas**,  |
|                 | con el mismo flujo sembrado (sesiones primero, lados después)  |
+-----------------+---------------------------------------------------------------+

Los **tres listones** de §11.2, y cuál se implementa
----------------------------------------------------

`plan.md` §11.2 separa tres listones porque confundirlos invalida el análisis:

- **A. «siempre largo» ``open`` -> cierre** — **implementado** como ``always_long``: mismo
  horizonte, mismo instrumento y solo el diferencial. Es el listón **obligatorio** y el
  comparable directo de cualquier estrategia intradía.
- **B. «siempre largo» aguantando la posición** — **no** implementado: exige una posición que
  cruza la noche y paga la financiación del CFD, y #13 no puede producirla (``nights = 0``
  fijo, `plan.md` §12 regla 6). Va a **#70**.
- **C. índice puro (``^GSPC``)** — **no** implementado: **no es invertible**, es una
  referencia y no un baseline, y compararse con él es el error que sobreestima la gestión. Va
  a **#28** (con la separación alpha/beta).

Ni B ni C tienen identificador, alias ni parámetro en esta API: **no se pueden pedir**.

Determinismo, azar y pureza
---------------------------

El azar, cuando existe (solo ``random_matched``), viene de ``random`` con una semilla
**entera obligatoria**: sin semilla hay error tipado, nunca un ``random.Random()`` sin
argumento ni un ``random.seed()`` global. La selección se resuelve **fuera** del bucle, así que
cada ``DecisionFn`` devuelta es pura: la misma ``SessionView`` da siempre la misma
``Decision``, sin contadores ni estado mutable. Dos corridas idénticas dan el mismo
``run_sha256`` byte a byte, también entre procesos y con ``PYTHONHASHSEED`` distinto (A20,
A26). El módulo no consulta el reloj, no toca el disco y no lee el ``Store``: su núcleo es
biblioteca estándar (A31).

Qué **no** hace este módulo
---------------------------

Cada frontera viaja en ``BASELINES_DOES_NOT_DO`` con su issue, y los huecos abiertos en
``FOLLOW_UPS``: no ejecuta ni cobra (#13), no calcula métricas (#15), no calcula PBO ni
Deflated Sharpe (#16), no es la corrida real sobre el histórico ni la tabla comparativa
(#18), no decide el nocional real ni el *sizing* (#27), no mide el *slippage* (#62), no lee
el almacén (#69) y no implementa el listón B (#70).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from fractions import Fraction
from typing import Final, cast

from cfdtrader.backtest.costs import CostModel, FinancingCut, SlippageParameter
from cfdtrader.backtest.engine import (
    BacktestRun,
    Decision,
    DecisionFn,
    Direction,
    SessionInput,
    SessionView,
    run_walk_forward,
)
from cfdtrader.backtest.splits import SplitPlan

__all__ = [
    "ALWAYS_LONG",
    "ALWAYS_SHORT",
    "BASELINES_DOES_NOT_DO",
    "BASELINE_IDS",
    "FOLLOW_UPS",
    "GAP_REVERSAL",
    "LIMITATIONS",
    "MOMENTUM_5D",
    "NO_TRADE",
    "RANDOM_MATCHED",
    "BaselinesError",
    "Bias",
    "InvalidBaselineParameterError",
    "always_long_bias",
    "always_short_bias",
    "baseline_deciders",
    "gap_reversal_bias",
    "make_decision_fn",
    "momentum_5d_signal",
    "no_trade_bias",
    "random_matched_deciders",
    "random_matched_signal",
    "run_baseline",
    "run_random_matched",
]

#: Los seis identificadores estables de §11.2, **y ningún otro** (A2). Tupla de solo
#: lectura: pedir un baseline que no esté aquí es un error tipado (A24).
NO_TRADE: Final[str] = "no_trade"
ALWAYS_LONG: Final[str] = "always_long"
ALWAYS_SHORT: Final[str] = "always_short"
MOMENTUM_5D: Final[str] = "momentum_5d"
GAP_REVERSAL: Final[str] = "gap_reversal"
RANDOM_MATCHED: Final[str] = "random_matched"

BASELINE_IDS: Final[tuple[str, ...]] = (
    NO_TRADE,
    ALWAYS_LONG,
    ALWAYS_SHORT,
    MOMENTUM_5D,
    GAP_REVERSAL,
    RANDOM_MATCHED,
)

#: Las cinco reglas que no necesitan azar: se construyen con ``baseline_deciders``.
DETERMINISTIC_IDS: Final[tuple[str, ...]] = (
    NO_TRADE,
    ALWAYS_LONG,
    ALWAYS_SHORT,
    MOMENTUM_5D,
    GAP_REVERSAL,
)

#: Sesiones de cierre previo que consume ``momentum_5d``: ``close[t-1]`` frente a
#: ``close[t-6]`` (A12).
MOMENTUM_LOOKBACK: Final[int] = 5

#: Qué **no** hace el módulo, legible por máquina (A34). Cada frontera con su issue.
BASELINES_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_ejecuta_ni_cobra",
        "issue": "#13",
        "statement": (
            "no ejecuta, no simula precios, no cobra y no construye particiones: resuelve el "
            "sesgo por sesion y entrega las DecisionFn al motor de #13, cuya unica via es "
            "run_walk_forward con cost_model y slippage sin valor por defecto"
        ),
    },
    {
        "id": "no_calcula_metricas",
        "issue": "#15",
        "statement": (
            "no calcula ninguna metrica de rendimiento (Sharpe, Sortino, EV, hit rate, payoff, "
            "profit factor, drawdown, bootstrap, Brier, log-loss ni curva de equity): las "
            "metricas netas son #15"
        ),
    },
    {
        "id": "no_pbo_ni_deflated_sharpe",
        "issue": "#16",
        "statement": (
            "no calcula PBO ni Deflated Sharpe y no registra experimentos en runs/<hash>/"
        ),
    },
    {
        "id": "no_es_la_corrida_real",
        "issue": "#18",
        "statement": (
            "no corre los seis baselines sobre el historico real ni entrega la tabla "
            "comparativa de metricas netas: eso es el informe de Fase 1 (#18), cuyo "
            "prerrequisito es el adaptador #69"
        ),
    },
    {
        "id": "no_decide_el_nocional_real",
        "issue": "#27",
        "statement": (
            "no decide el nocional real, el sizing, el apalancamiento ni los umbrales: el "
            "nominal que viaja a Decision.notional_usd es un parametro obligatorio del llamante"
        ),
    },
    {
        "id": "no_mide_slippage",
        "issue": "#62",
        "statement": (
            "no mide el slippage: lo recibe el motor como parametro y el supuesto declarado de "
            "#64 deja pnl_net_pct en null"
        ),
    },
    {
        "id": "no_lee_el_almacen",
        "issue": "#69",
        "statement": (
            "no lee el Store, no construye los SessionInput ni el context y no escribe "
            "informes: sin el adaptador #69 solo se ejercita con entradas sinteticas"
        ),
    },
    {
        "id": "no_implementa_el_liston_b",
        "issue": "#70",
        "statement": (
            "no implementa el liston B (siempre largo aguantando la posicion): exige una "
            "posicion que cruza la noche y #13 la prohibe (regla 6, nights = 0 fijo)"
        ),
    },
    {
        "id": "no_implementa_el_liston_c",
        "issue": "#28",
        "statement": (
            "no implementa el liston C (indice puro ^GSPC) porque no es invertible: es una "
            "referencia, no un baseline, y la separacion alpha/beta es #28"
        ),
    },
)

#: Seguimientos abiertos que este módulo deja declarados (A34).
FOLLOW_UPS: Final[tuple[dict[str, str], ...]] = (
    {
        "issue": "#15",
        "topic": "metricas netas de costes",
        "why": "aqui solo se deciden las reglas; el rendimiento se mide alli",
    },
    {
        "issue": "#18",
        "topic": "corrida real sobre el historico y tabla comparativa de baselines",
        "why": "es el informe de Fase 1 y su puerta de salida; su prerrequisito duro es #69",
    },
    {
        "issue": "#27",
        "topic": "nocional real, tiers y reglas duras del gate",
        "why": "el nocional de aqui es un plano ilustrativo para comparar baselines",
    },
    {
        "issue": "#62",
        "topic": "medir el slippage real",
        "why": "es lo que cierra el total y convierte el supuesto de #64 en medicion",
    },
    {
        "issue": "#69",
        "topic": "adaptador Store -> SessionInput + CLI + informe",
        "why": "sin el, los baselines solo se ejercitan con entradas sinteticas",
    },
    {
        "issue": "#70",
        "topic": "liston B (posicion que cruza la noche, con financiacion)",
        "why": "exige una via overnight que #13 no tiene (regla 6)",
    },
)

#: Limitaciones que el módulo publica (A4, A11, A30). No se esconden.
LIMITATIONS: Final[tuple[str, ...]] = (
    "**De los tres listones de `plan.md` §11.2, aqui solo esta el A** (`always_long` "
    "`open` -> cierre, mismo horizonte y mismo instrumento): el liston B (aguantar la "
    "posicion, con la financiacion del CFD) es #70 porque #13 no puede producir una posicion "
    "que cruza la noche, y el liston C (indice puro) es #28 porque no es invertible: es una "
    "referencia, no un baseline.",
    "**El nocional es un plano ilustrativo** para comparar baselines entre si y lo declara el "
    "llamante: el modulo nunca lo deriva del capital, del riesgo, del apalancamiento ni de "
    "`R`, y el tamano real es #27.",
    "**Este modulo no calcula ninguna metrica de rendimiento**: lo unico que agrega son "
    "**recuentos** ya publicados por #13 (`traded`, `no_trade`, `skipped`, frecuencia de "
    "operacion y recuento de `exit_reason`). La convencion «Sharpe = 0» de §11.2 para «no "
    "operar» se declara aqui y no se calcula: las metricas netas, con su intervalo bootstrap, "
    "son #15.",
    "**La regla aleatoria es un control sobre la muestra de test**: para operar exactamente "
    "`floor(frequency * n_test)` sesiones distintas, su seleccion depende de cuantas sesiones "
    "de test hay. Anadir sesiones **posteriores** cambia el tamano de la muestra y por tanto el "
    "sorteo (con el mismo `seed` deja de ser comparable); lo que **no** depende de ningun "
    "precio es su decision, igual que en las otras cinco reglas. El liston B y el C no estan "
    "aqui (#70, #28) y ninguna de las seis reglas ejecuta sin coste: el unico camino es #13.",
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados (A1, A24): nunca un resultado silencioso
# ─────────────────────────────────────────────────────────────────────────────
class BaselinesError(Exception):
    """Raíz de los errores de los baselines triviales."""


class InvalidBaselineParameterError(BaselinesError):
    """Un parámetro de la llamada no es admisible (A24): error tipado, nunca silencioso."""


# ─────────────────────────────────────────────────────────────────────────────
# Bias: el sesgo declarado de una sesion (direccion + motivo, nunca una direccion muda)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Bias:
    """El sesgo declarado de una sesión: ``direction`` y ``reason`` no vacío (A5).

    Un ``NOTHING`` sin motivo no es admisible, y una dirección no se declara nunca sin
    explicar de dónde sale: el motivo viaja hasta ``SessionOutcome.reason``.
    """

    direction: Direction
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise InvalidBaselineParameterError(
                "Bias.reason: una direccion sin motivo no se admite (A5); cada baseline "
                "declara por que opera o por que no"
            )


#: Motivos declarados de cada regla. No son prosa decorativa: viajan al informe de #13.
NO_TRADE_REASON: Final[str] = (
    "baseline no_trade: no operar es un artefacto comprobable; la convencion «Sharpe = 0» de "
    "`plan.md` §11.2 se declara y no se calcula (las metricas son #15)"
)
ALWAYS_LONG_REASON: Final[str] = (
    "baseline always_long (liston A de `plan.md` §11.2): largo en el open de la subasta y "
    "salida en el cierre de la misma sesion, todos los dias"
)
ALWAYS_SHORT_REASON: Final[str] = (
    "baseline always_short: espejo del liston A: corto en el open de la subasta y salida en "
    "el cierre de la misma sesion, todos los dias"
)
MOMENTUM_NO_HISTORY_REASON: Final[str] = (
    "momentum_5d: historia insuficiente (hacen falta seis cierres previos, el de t-6 y el de "
    "t-1 entre ellos); sin la ventana completa no se asume ninguna direccion (A15)"
)
MOMENTUM_MISSING_CLOSE_REASON: Final[str] = (
    "momentum_5d: algun cierre previo de la ventana de cinco dias esta ausente o no es "
    "positivo; con la ventana incompleta no se asume direccion (A15)"
)
MOMENTUM_TIE_REASON: Final[str] = (
    "momentum_5d: el retorno de los cinco dias previos vale exactamente 0 (empate); no se "
    "redondea ni se rompe a favor de una direccion (A16)"
)
RANDOM_NOT_CHOSEN_REASON: Final[str] = (
    "baseline random_matched: sesion no elegida por el sorteo sembrado; el control opera "
    "exactamente la frecuencia declarada de las sesiones de test (A19)"
)
GAP_UNKNOWN_REASON: Final[str] = (
    "gap_reversal: el gap de apertura no existe en la primera sesion de la secuencia (no hay "
    "cierre anterior); no se interpreta como direccion (A18)"
)
GAP_TIE_REASON: Final[str] = (
    "gap_reversal: el gap de apertura vale exactamente 0 (empate); no se interpreta como "
    "direccion (A18)"
)


# ─────────────────────────────────────────────────────────────────────────────
# NUCLEO (A36): las seis reglas, puras y sin mirar el futuro
# ─────────────────────────────────────────────────────────────────────────────
def no_trade_bias(_view: SessionView) -> Bias:
    """«No operar»: ``NOTHING`` en **todas** las sesiones y sin consultar ningún precio (A6).

    La vista no se toca: el sesgo es constante y por eso esta regla es el artefacto
    comprobable de §11.2 («Sharpe = 0» como convención declarada, no calculada: #15).
    """
    return Bias(direction=Direction.NOTHING, reason=NO_TRADE_REASON)


def always_long_bias(_view: SessionView) -> Bias:
    """Largo en **todas** las sesiones, sin consultar ningún precio (A5).

    Es el **listón A** de `plan.md` §11.2: la entrada la fija el motor en el ``open`` de la
    subasta y la salida cae dentro de la misma sesión (``session_close``), así que esta regla
    solo declara la dirección.
    """
    return Bias(direction=Direction.LONG, reason=ALWAYS_LONG_REASON)


def always_short_bias(_view: SessionView) -> Bias:
    """Corto en **todas** las sesiones, sin consultar ningún precio (A5): espejo del listón A."""
    return Bias(direction=Direction.SHORT, reason=ALWAYS_SHORT_REASON)


def gap_reversal_bias(view: SessionView) -> Bias:
    """Sesgo **contrario** al *gap* de apertura que publica #13 (A17, A18).

    Consume ``SessionView.gap_px`` **sin recalcularlo**: #13 lo mide como
    ``open_px[i] / close_px[i-1] - 1`` con el cierre de la sesión **anterior de la secuencia**
    (no del día natural anterior). ``gap_px > 0`` es una apertura por encima del cierre previo
    y se responde corto; ``gap_px < 0``, largo. El empate (``0``) y la ausencia del *gap* en
    la primera sesión son dos ``NOTHING`` con motivos **distintos** y declarados.
    """
    gap = view.gap_px
    if gap is None:
        return Bias(direction=Direction.NOTHING, reason=GAP_UNKNOWN_REASON)
    if gap > 0:
        return Bias(
            direction=Direction.SHORT,
            reason=(
                f"gap_reversal: la apertura sube un {gap:.6f} sobre el cierre de la sesion "
                "anterior; el sesgo es contrario al gap, asi que corto (A17)"
            ),
        )
    if gap < 0:
        return Bias(
            direction=Direction.LONG,
            reason=(
                f"gap_reversal: la apertura cae un {gap:.6f} sobre el cierre de la sesion "
                "anterior; el sesgo es contrario al gap, asi que largo (A17)"
            ),
        )
    return Bias(direction=Direction.NOTHING, reason=GAP_TIE_REASON)


def momentum_5d_signal(inputs: Sequence[SessionInput]) -> dict[date, Bias]:
    """Señal de ``momentum_5d`` por sesión (A12, A13, A15, A16).

    Se resuelve **una vez**, sobre la secuencia de entrada (que sí trae los ``close_px`` de
    las sesiones pasadas) y **fuera** del bucle del motor: la ``DecisionFn`` resultante solo
    consulta la señal ya calculada por ``view.session``. La vista de #13 **no** expone ningún
    cierre, así que aquí no se intenta obtener la historia de ella.

    El retorno de los cinco días previos se lee como ``close[t-1]`` frente a ``close[t-6]``,
    **nunca** con un dato de la sesión ``t``: subida -> ``LONG``, bajada -> ``SHORT``,
    empate -> ``NOTHING`` declarado. Falta de historia (menos de seis sesiones previas) o
    algún cierre previo ausente o no positivo -> ``NOTHING``, nunca una dirección asumida.
    """
    items = _require_inputs(inputs)
    signal: dict[date, Bias] = {}
    for position, item in enumerate(items):
        signal[item.session] = _momentum_bias(items, position)
    return signal


def _momentum_bias(items: tuple[SessionInput, ...], position: int) -> Bias:
    """El sesgo de esa posición: la ventana completa o un ``NOTHING`` motivado (A15)."""
    window = range(position - MOMENTUM_LOOKBACK - 1, position)
    if position <= MOMENTUM_LOOKBACK:
        return Bias(direction=Direction.NOTHING, reason=MOMENTUM_NO_HISTORY_REASON)
    closes: list[float] = []
    for offset in window:
        close = _positive_price(items[offset].close_px)
        if close is None:
            return Bias(direction=Direction.NOTHING, reason=MOMENTUM_MISSING_CLOSE_REASON)
        closes.append(close)
    base = closes[0]
    latest = closes[-1]
    if latest > base:
        return Bias(
            direction=Direction.LONG,
            reason=(
                f"momentum_5d: el cierre previo ({latest}) supera al de cinco dias antes "
                f"({base}); el retorno de la ventana es positivo, asi que largo (A12)"
            ),
        )
    if latest < base:
        return Bias(
            direction=Direction.SHORT,
            reason=(
                f"momentum_5d: el cierre previo ({latest}) queda por debajo del de cinco dias "
                f"antes ({base}); el retorno de la ventana es negativo, asi que corto (A12)"
            ),
        )
    return Bias(direction=Direction.NOTHING, reason=MOMENTUM_TIE_REASON)


def random_matched_signal(
    sessions: Sequence[SessionInput], *, frequency: Fraction, seed: int
) -> dict[date, Bias]:
    """Selección sembrada de ``random_matched`` sobre la muestra de test (A19-A23).

    Función **pura**: la misma secuencia, la misma ``frequency`` y el mismo ``seed`` dan el
    mismo resultado, sin contadores ni estado mutable y sin depender de ``PYTHONHASHSEED``.
    Opera exactamente ``floor(frequency · len(sessions))`` sesiones **distintas** (A19) y el
    resto queda en ``NOTHING`` con su motivo. El orden de extracción está **documentado y
    fijo**: primero las sesiones (``random.Random(seed).sample`` sobre las posiciones) y
    después los lados, un ``random()`` por sesión elegida en orden ascendente de sesión
    (A23). Con ``frequency == 0`` no opera ninguna y con ``frequency == 1`` las opera todas.
    """
    items = _require_inputs(sessions)
    rate = _require_frequency(frequency)
    origin = _require_seed(seed)
    total = len(items)
    draws = rate * total
    traded = draws.numerator // draws.denominator  # floor exacto y >= 0 (A19)
    generator = random.Random(origin)  # noqa: S311 - azar sembrado y declarativo (A20)
    chosen = sorted(generator.sample(range(total), traded))
    signal: dict[date, Bias] = {}
    for position in chosen:  # sesiones primero, en orden ascendente (A23)
        long_side = generator.random() < 0.5
        signal[items[position].session] = Bias(
            direction=Direction.LONG if long_side else Direction.SHORT,
            reason=(
                f"baseline random_matched: sesion elegida por el sorteo (seed={origin}); el "
                "lado sale del mismo flujo sembrado, despues de las sesiones (A23)"
            ),
        )
    for item in items:
        signal.setdefault(
            item.session,
            Bias(direction=Direction.NOTHING, reason=RANDOM_NOT_CHOSEN_REASON),
        )
    return signal


# ─────────────────────────────────────────────────────────────────────────────
# Validacion (A24) y contrato de la DecisionFn (A6, A11, A21)
# ─────────────────────────────────────────────────────────────────────────────
def _require_instance(value: object, expected: type[object], *, field: str) -> None:
    """Comprueba en tiempo de ejecución el tipo declarado: un contrato roto es error tipado."""
    if not isinstance(value, expected):
        raise InvalidBaselineParameterError(
            f"{field}: se espera {expected.__name__}, no {type(value).__name__} (A24)"
        )


def _positive_price(value: object) -> float | None:
    """Un cierre utilizable (número positivo) o ``None``: nunca se inventa un cierre."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0.0 else None


def _require_session_input(value: object, *, field: str) -> SessionInput:
    """Una sesión de entrada con su fecha (A24): el resto de campos los valida #13."""
    if not isinstance(value, SessionInput):
        raise InvalidBaselineParameterError(
            f"{field}: se espera SessionInput, no {type(value).__name__} (A24)"
        )
    _require_instance(value.session, date, field=f"{field}.session")
    return value


def _require_inputs(value: object) -> tuple[SessionInput, ...]:
    """La secuencia de sesiones, **estrictamente creciente y sin duplicados** (A24).

    Una secuencia **vacía** pasa: quien la rechaza es #13 con ``EngineInputError``, y este
    módulo no envuelve ni sustituye los errores de #13 (A25).
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InvalidBaselineParameterError(
            f"inputs: se espera una secuencia de SessionInput, no {type(value).__name__} (A24)"
        )
    items = tuple(
        _require_session_input(item, field=f"inputs[{position}]")
        for position, item in enumerate(cast("Sequence[object]", value))
    )
    for position in range(len(items) - 1):
        if items[position].session >= items[position + 1].session:
            raise InvalidBaselineParameterError(
                "inputs: las sesiones tienen que ser estrictamente crecientes y sin "
                f"duplicados; se rompe en la posicion {position + 1} (A24)"
            )
    return items


def _require_plan(value: object) -> SplitPlan:
    """El plan de #12 se consume tal cual: ni se rehace aquí ni se acepta otra cosa (A24)."""
    if not isinstance(value, SplitPlan):
        raise InvalidBaselineParameterError(
            f"split_plan: se espera el SplitPlan de #12, no {type(value).__name__} (A24)"
        )
    return value


def _require_notional(value: object) -> Decimal:
    """El nocional es un ``Decimal`` explícito y positivo (A11, A24): nunca un ``float``."""
    if not isinstance(value, Decimal):
        raise InvalidBaselineParameterError(
            "notional_usd: hay que declararlo como `decimal.Decimal` exacto, no "
            f"{type(value).__name__} (A11/A24)"
        )
    if not value.is_finite() or value <= 0:
        raise InvalidBaselineParameterError(
            f"notional_usd: tiene que ser un importe positivo y finito, no {value} (A24)"
        )
    return value


def _require_frequency(value: object) -> Fraction:
    """La frecuencia es un ``Fraction`` **exacto** dentro de ``[0, 1]`` (A19, A22, A24)."""
    if not isinstance(value, Fraction):
        raise InvalidBaselineParameterError(
            "frequency: hay que declararla como `fractions.Fraction` exacta (nunca un "
            f"`float`), no {type(value).__name__} (A19/A24)"
        )
    if value < 0 or value > 1:
        raise InvalidBaselineParameterError(
            f"frequency: fuera de [0, 1] vale {value}; la frecuencia es una proporcion "
            "declarada de sesiones operadas (A22/A24)"
        )
    return value


def _require_seed(value: object) -> int:
    """La semilla es un entero explícito (A20, A24): sin ella el azar no es reproducible."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidBaselineParameterError(
            f"seed: hace falta un entero explicito (recibido {type(value).__name__}); sin "
            "semilla no hay sorteo reproducible y `random.Random()` sin argumento esta "
            "prohibido (A20/A24)"
        )
    return value


def _require_baseline(value: object) -> str:
    """El identificador tiene que ser uno de los seis declarados (A2, A24)."""
    if not isinstance(value, str) or value not in BASELINE_IDS:
        raise InvalidBaselineParameterError(
            f"baseline: identificador desconocido ({value!r}); los seis declarados son "
            f"{', '.join(BASELINE_IDS)} (A24)"
        )
    return value


def _require_deterministic_baseline(value: object) -> str:
    """Como ``_require_baseline`` pero enviando la regla aleatoria a su constructor."""
    name = _require_baseline(value)
    if name == RANDOM_MATCHED:
        raise InvalidBaselineParameterError(
            f"{RANDOM_MATCHED} exige `frequency` y `seed` sin valor por defecto: se construye "
            "con `random_matched_deciders` / `run_random_matched`, no con "
            "`baseline_deciders` / `run_baseline` (A19/A20)"
        )
    return name


def _signal_resolver(
    signal: Mapping[date, Bias], *, baseline: str
) -> Callable[[SessionView], Bias]:
    """La señal ya calculada, indexada por la sesión de la vista (A13, A21)."""

    def resolve(view: SessionView) -> Bias:
        try:
            return signal[view.session]
        except KeyError:
            raise InvalidBaselineParameterError(
                f"{baseline}: la sesion {view.session.isoformat()} no esta en la secuencia con "
                "la que se resolvio la senal; el motor solo evalua las sesiones que se le "
                "entregan (A24)"
            ) from None

    return resolve


def _test_pool(items: tuple[SessionInput, ...], plan: SplitPlan) -> tuple[SessionInput, ...]:
    """Las sesiones de test del plan, **en orden y sin repetición** (A19).

    Es la unión exacta de los ``test`` de los folds de #12. Las posiciones que se salen de la
    secuencia no se inventan aquí: un plan que no case con las entradas lo rechaza el motor de
    #13 con ``EngineInputError`` (A25).
    """
    positions = sorted({position for fold in plan.folds for position in fold.test})
    return tuple(items[position] for position in positions if position < len(items))


def make_decision_fn(
    resolve: Callable[[SessionView], Bias], *, notional_usd: Decimal
) -> DecisionFn:
    """Envuelve un sesgo por sesión en la ``DecisionFn`` que consume #13 (A6, A11, A21).

    El nocional es **obligatorio y explícito** (keyword-only, sin valor por defecto) y viaja a
    ``Decision.notional_usd`` solo cuando hay dirección; con ``NOTHING`` vale ``None``, nunca
    un 0 inventado. La función devuelta es **pura**: no guarda contadores ni estado mutable y
    la misma ``SessionView`` devuelve siempre la misma ``Decision``.
    """
    amount = _require_notional(notional_usd)

    def decide(view: SessionView) -> Decision:
        item = resolve(view)
        if item.direction is Direction.NOTHING:
            return Decision(direction=Direction.NOTHING, reason=item.reason)
        return Decision(direction=item.direction, reason=item.reason, notional_usd=amount)

    return decide


# ─────────────────────────────────────────────────────────────────────────────
# Construccion de las DecisionFn (A8): una por fold, en el orden del plan
# ─────────────────────────────────────────────────────────────────────────────
def baseline_deciders(
    inputs: Sequence[SessionInput],
    split_plan: SplitPlan,
    *,
    baseline: str,
    notional_usd: Decimal,
) -> tuple[DecisionFn, ...]:
    """Las cinco reglas sin azar, como una ``DecisionFn`` **por fold** (A8).

    Devuelve una entrada por cada fold de ``split_plan`` **en su orden exacto**, lista para
    ``decide_by_fold``. Todos los folds reciben la **misma** función porque un baseline no
    entrena: la decisión depende solo de la sesión y de la señal ya resuelta. La regla
    aleatoria **no** se construye aquí (exige ``frequency`` y ``seed``): usa
    ``random_matched_deciders``.
    """
    items = _require_inputs(inputs)
    plan = _require_plan(split_plan)
    amount = _require_notional(notional_usd)
    name = _require_deterministic_baseline(baseline)
    resolve = _deterministic_resolver(name, items)
    decide = make_decision_fn(resolve, notional_usd=amount)
    deciders: tuple[DecisionFn, ...] = tuple(decide for _ in plan.folds)
    return deciders


def _deterministic_resolver(
    baseline: str, items: tuple[SessionInput, ...]
) -> Callable[[SessionView], Bias]:
    """El sesgo por sesión de una de las cinco reglas deterministas (A5, A17)."""
    if baseline == NO_TRADE:
        return no_trade_bias
    if baseline == ALWAYS_LONG:
        return always_long_bias
    if baseline == ALWAYS_SHORT:
        return always_short_bias
    if baseline == GAP_REVERSAL:
        return gap_reversal_bias
    return _signal_resolver(momentum_5d_signal(items), baseline=MOMENTUM_5D)


def random_matched_deciders(
    inputs: Sequence[SessionInput],
    split_plan: SplitPlan,
    *,
    notional_usd: Decimal,
    frequency: Fraction,
    seed: int,
) -> tuple[DecisionFn, ...]:
    """El control aleatorio, como una ``DecisionFn`` **por fold** (A8, A19-A21).

    La selección se resuelve **fuera** del bucle, sobre la muestra de test del plan de #12
    (la unión exacta de sus ``test``, sin repetición): opera exactamente
    ``floor(frequency · n_test)`` sesiones distintas y el resto queda en ``NOTHING``. Ni
    ``frequency`` ni ``seed`` tienen valor por defecto.
    """
    items = _require_inputs(inputs)
    plan = _require_plan(split_plan)
    amount = _require_notional(notional_usd)
    signal = random_matched_signal(_test_pool(items, plan), frequency=frequency, seed=seed)
    resolver = _signal_resolver(signal, baseline=RANDOM_MATCHED)
    decide = make_decision_fn(resolver, notional_usd=amount)
    deciders: tuple[DecisionFn, ...] = tuple(decide for _ in plan.folds)
    return deciders


# ─────────────────────────────────────────────────────────────────────────────
# Ejecucion (A8, A10, A11): la unica via es el motor de #13
# ─────────────────────────────────────────────────────────────────────────────
def run_baseline(
    inputs: Sequence[SessionInput],
    *,
    split_plan: SplitPlan,
    cost_model: CostModel,
    slippage: SlippageParameter,
    notional_usd: Decimal,
    baseline: str,
    financing_cut: FinancingCut | None = None,
) -> BacktestRun:
    """Corre una de las cinco reglas deterministas con el motor de #13 (A8, A10, A11).

    ``cost_model`` y ``slippage`` son **obligatorios, keyword-only y sin valor por defecto**:
    el módulo no añade ninguna vía alternativa ni un valor de relleno, así que ninguna regla
    se puede ejecutar sin coste. Con el supuesto pesimista de #64 (``state = "assumed"``) el
    total no se cierra y ``pnl_net_pct`` queda en ``null`` con su motivo: aquí no se rellena
    con 0. ``notional_usd`` es el nocional **plano e ilustrativo** que declara el llamante.
    """
    deciders = baseline_deciders(inputs, split_plan, baseline=baseline, notional_usd=notional_usd)
    return run_walk_forward(
        inputs,
        split_plan=split_plan,
        cost_model=cost_model,
        slippage=slippage,
        decide_by_fold=deciders,
        financing_cut=financing_cut,
    )


def run_random_matched(
    inputs: Sequence[SessionInput],
    *,
    split_plan: SplitPlan,
    cost_model: CostModel,
    slippage: SlippageParameter,
    notional_usd: Decimal,
    frequency: Fraction,
    seed: int,
    financing_cut: FinancingCut | None = None,
) -> BacktestRun:
    """Corre el control aleatorio con el motor de #13 (A19-A23).

    Mismas reglas de coste que ``run_baseline``: ``cost_model`` y ``slippage`` sin valor por
    defecto. ``frequency`` (``Fraction`` exacta, en ``[0, 1]``) y ``seed`` (entero) son
    obligatorios: mismas entradas, misma frecuencia y misma semilla dan el mismo
    ``run_sha256``, también entre procesos y con ``PYTHONHASHSEED`` distinto.
    """
    deciders = random_matched_deciders(
        inputs,
        split_plan,
        notional_usd=notional_usd,
        frequency=frequency,
        seed=seed,
    )
    return run_walk_forward(
        inputs,
        split_plan=split_plan,
        cost_model=cost_model,
        slippage=slippage,
        decide_by_fold=deciders,
        financing_cut=financing_cut,
    )
