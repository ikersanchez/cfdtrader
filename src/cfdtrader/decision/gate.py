"""Gate de decision y *sizing* — tarea #27.

Este modulo es el **unico** sitio del proyecto donde una probabilidad calibrada se convierte
en una decision operativa: ``LONG``, ``SHORT`` o ``NOTHING``, con su stop, su objetivo, su
nocional, su tier y el detalle de **cada una de las 18 reglas duras** de ``plan.md`` §12 que
le toca aplicar aqui.

Funcion **pura y determinista**

``evaluate_gate`` no tiene reloj, ni red, ni disco, ni azar: ``as_of``, la fecha de "hoy" en
ET, el ``MarketCalendar`` **ya construido**, las fechas de FOMC, el P&L realizado y el
contador de observacion entran **como argumentos explicitos**. Mismos argumentos ⇒ misma
salida **byte a byte**, tambien entre procesos con ``PYTHONHASHSEED`` distinto. El modulo
**no** lee la configuracion del repositorio ni construye el calendario (construirlo si lee su
fichero de excepciones): el calendario llega de fuera.

Consume, no produce
-------------------

- La **probabilidad calibrada** viene de #25/#26 (``selection.selected.calibrated``): el gate no
  entrena, no calibra y no toca la probabilidad cruda.
- El **coste** viene de #11 (``backtest.costs.cost_breakdown``): el gate no recalcula un
  diferencial, una tenencia ni un *slippage*.
- El **stop** entra ya calculado (``stop_pct``): derivarlo de la volatilidad es una decision
  declarada de #60 y **no se inventa aqui**.

Unidades (trampa #80)
---------------------

``cost_pct`` y ``ev_declared_pct``/``ev_net_pct`` se publican en **porcentaje**, copiando el
campo del ``CostBreakdown`` **sin conversion**: ``c_declared_pct`` vale ``0,0042`` y eso son
``0,0042 %`` (su gemelo es ``c_fraction_of_notional``, ``0,000042``). El defecto de
``backtest/engine.py:840`` —restar un porcentaje a una fraccion, 100x el coste— **no se
replica ni se arregla aqui**: es **#80**. La convencion del modulo es una sola y esta escrita
en cada ``Field``: los porcentajes van en puntos porcentuales como numero (``2`` son 2 %).

El *slippage* no se fusiona con nada
------------------------------------

Los tres estados de #11 (``measured``/``assumed``/``unmeasured``) llegan **tal cual** dentro
del ``CostBreakdown``. ``ev_declared_pct`` se publica **siempre** (usa ``c_declared_pct``) y
``ev_net_pct`` **solo** si ``c_total_pct`` tiene valor: un *slippage* no medido deja el EV
neto en ``None`` y **nunca** en ``0``. Con el supuesto vigente (``assumed``, ``R`` sin
decidir, #60) el total es ``null``, la regla 9 no se puede verificar y el gate lo publica
como *blocker* que cita #62 y #60, sin convertir el supuesto en un numero.

Los cinco estados de salida
---------------------------

``GateStatus`` tiene los **cuatro estados de ``plan.md`` §19.2** (``recommendation``,
``no_recommendation_stale_data``, ``no_recommendation_data_quality``, ``error``) **mas**
``no_recommendation_undecided``, el estado bloqueado que obliga ``tech_stack.md`` §11 bis
mientras #59 (broker) y #60 (umbrales, riesgo y tamano de ``R``) sigan abiertos. Ninguno se
fusiona con ``NOTHING``: ``NOTHING`` significa "hoy no veo oportunidad" y es un
``recommendation`` con ``direction = nothing``; los estados "no se" llevan ``direction =
None`` y **no emiten nocional**. En ellos no hay juicio que hacer y el gate no lo fabrica.
``error`` es el fallo tecnico del pipeline (lo emite el ``run_log`` de #40/#43); este modulo,
ante una entrada inadmisible, lanza ``GateInputError``.

Decisiones abiertas que **no** se inventan
------------------------------------------

``GateParameters`` (Pydantic v2, ``frozen``) declara las decisiones que solo el propietario
cierra, todas con ``None`` = **sin decidir** (``tech_stack.md`` §11 bis): el broker (#59) y,
por #60, el umbral de EV, el riesgo por operacion, las perdidas maxima diaria/semanal/mensual,
el tamano de ``R`` y los tres umbrales del tier. Con ``GateParameters()`` —todo ``None``— el
gate devuelve ``no_recommendation_undecided`` y ``undecided[]`` nombra cada parametro que
falta **con su issue**. Las cifras del enunciado (1 %, 2 %, 3c, 0,58, el tamano de ``R``) no
estan cableadas en ninguna parte: son valores que el llamante declara.

Las 18 reglas de §12 y quien las aplica
---------------------------------------

===========  ==================================================  ==============================
Regla        Enunciado                                           Quien la aplica en #27
===========  ==================================================  ==============================
1            Maximo 1 operacion por sesion                       gate: input ``trades_today``
2            Riesgo por operacion <= 1 %                         gate por operacion; cartera #28
3            Perdida diaria -2 % ⇒ *kill switch*                 gate: input + umbral #60
4            Perdida semanal -5 % ⇒ parada                       gate: input; acumular #83
5            Perdida mensual -10 % ⇒ parada total                gate: input; acumular #83
6            Sin *overnight*                                     gate: ``nights = 0`` siempre
7            Stop obligatorio                                    gate: ``GateInputError`` sin stop
8            Objetivo >= 2x el coste de ida y vuelta             gate, con el coste declarado
9            EV neto > umbral de seguridad                       gate: ``params.ev_threshold_pct``
10           Solo operar tier A al principio                     gate: ``params.authorized_tiers``
11           Operar 10-30 % de los dias                          **delegada** a #28
12           Nunca ampliar perdedora ni mover el stop            gate: funcion pura y sin estado
13           Guardia de obsolescencia                            gate: ``as_of`` vs ``today``; #40
14           "No se" != ``NOTHING``                              gate: cinco estados; registro #40
15           Modo observacion de 5 sesiones tras una ausencia    gate: input; persistir #40
16           Cierre obligatorio (orden *bracket*)                gate: ``bracket_required``; #84
17           Dias de FOMC ⇒ ``NOTHING``                          gate: input; ingesta #34
18           Medias sesiones ⇒ ``NOTHING``                       gate: ``is_half_day``; flujo #40
===========  ==================================================  ==============================

Precedencia declarada de los estados: primero la **frescura** y la **calidad** de los datos
("no se" sobre el dato), despues las **decisiones abiertas** (``undecided``, "no se" sobre la
politica) y solo entonces los **bloqueos** de §12. Los bloqueos se evaluan **todos** (no se
corta en el primero) y se publican juntos: un dia de FOMC con una operacion ya hecha trae los
dos motivos, no uno.

``gate_sha256``
---------------

``gate_sha256 = "sha256:" + sha256(JSON canonico del payload sin esa clave)``. El prefijo es
obligatorio: un sha256 desnudo (64 hex) es lo que ``detect-secrets`` marca y bloquearia el
commit (politica de #19/#20). El payload incluye los parametros declarados, asi que cambiar un
umbral cambia el hash aunque la direccion coincida.

Que **no** hace este modulo (fronteras declaradas, con su issue)
----------------------------------------------------------------

- **No** es #28: no se cablea al motor, no ejecuta el *backtest* y no mide la regla 11 (ni el
  limite de riesgo en su lectura de **cartera**).
- **No** es #34: no ingiere el calendario de FOMC; recibe el conjunto de fechas declarado.
- **No** es #35: no aplica el *overlay* del LLM (veto ni ajuste de +-10 pp).
- **No** es #39/#40: no persiste nada en el diario, no implementa la guardia completa de
  obsolescencia ni el contador del modo observacion.
- **No** es #62: no mide el *slippage*; consume el estado que #11 le entrega.
- **No** es #80: no arregla las unidades de ``pnl_declared_pct`` en el motor.
- **No** es #83: no acumula el P&L diario, semanal ni mensual; lo recibe ya acumulado.
- **No** es #84: no coloca la orden *bracket* en el broker.
- **No** decide los umbrales ni el tamano de ``R``: son #59 y #60, con ``None`` = sin decidir.
- **No** deriva el stop de la volatilidad: recibe ``stop_pct`` ya calculado.

El nucleo solo usa **biblioteca estandar** (``hashlib``, ``json``, ``decimal``, ``dataclasses``)
y **Pydantic v2**; no usa ``numpy`` ni ``polars``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, localcontext
from enum import StrEnum
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from cfdtrader.backtest.costs import CostBreakdown
from cfdtrader.backtest.engine import Decision, Direction, canonical_text
from cfdtrader.data.calendar import MarketCalendar

__all__ = [
    "DECISION_THRESHOLD",
    "FOLLOW_UPS",
    "GATE_HASH_FORMAT",
    "GATE_HASH_PREFIX",
    "LIMITATIONS",
    "PARAMETER_ISSUES",
    "RULES",
    "TARGET_MIN_COST_MULTIPLE",
    "TIERS",
    "TIER_A",
    "TIER_B",
    "TIER_C",
    "GateError",
    "GateInputError",
    "GateOutput",
    "GateParameters",
    "GateStatus",
    "Tier",
    "evaluate_gate",
    "gate_sha256",
    "to_engine_decision",
]

# ─────────────────────────────────────────────────────────────────────────────
# Formato del hash (A2) y convenciones
# ─────────────────────────────────────────────────────────────────────────────
#: Formato estable de ``gate_sha256`` (A2). Se hashea el JSON canonico del payload **sin** la
#: clave ``gate_sha256`` (una salida no se hashea a si misma), con claves ordenadas,
#: separadores ``,`` y ``:``, ``Decimal`` como cadena decimal exacta y UTF-8.
GATE_HASH_FORMAT: Final[str] = (
    "sha256(UTF-8) de json.dumps(payload, sort_keys=True, separators=(',', ':'), "
    "ensure_ascii=False), con Decimal como cadena decimal exacta; el payload que se hashea no "
    "incluye la clave gate_sha256"
)
#: Prefijo obligatorio del digest (politica de #19/#20 frente a ``detect-secrets``).
GATE_HASH_PREFIX: Final[str] = "sha256:"

#: Cuantia del nocional (el dinero se publica con centimos) y del apalancamiento, que es
#: **derivado** (``notional / capital``) y nunca una entrada.
NOTIONAL_QUANTUM: Final[Decimal] = Decimal("0.01")
LEVERAGE_QUANTUM: Final[Decimal] = Decimal("0.0001")
#: Precision de la aritmetica del *sizing*: ``capital x riesgo / stop`` puede tener cola.
MONEY_PRECISION: Final[int] = 50

#: Umbral de direccion: el gate decide a favor del lado con ventaja declarada (``p >= 0,5``
#: mira arriba), el **mismo** criterio que #24/#26. No es un umbral de riesgo de #60: es el
#: signo de la ventaja y por eso no vive en ``GateParameters``.
DECISION_THRESHOLD: Final[float] = 0.5

#: Regla 8 de §12, **literal**: el objetivo tiene que cubrir al menos este multiplo del coste
#: de ida y vuelta declarado. No es una de las cifras que #60 decide (esas van declaradas en
#: ``GateParameters``): la regla viene escrita con su factor en ``plan.md`` §12.
TARGET_MIN_COST_MULTIPLE: Final[Decimal] = Decimal("2")

#: Tiers de confianza de §12. El tier se **deriva**; quien decide cual se autoriza es #60.
TIER_A: Final[str] = "A"
TIER_B: Final[str] = "B"
TIER_C: Final[str] = "C"
TIERS: Final[tuple[str, ...]] = (TIER_A, TIER_B, TIER_C)
Tier = Literal["A", "B", "C"]

#: Resultado de una regla en el registro que publica el gate.
_RULE_OK: Final[str] = "pass"
_RULE_BLOCKED: Final[str] = "blocked"
_RULE_DELEGATED: Final[str] = "delegated"
_RULE_NOT_EVALUATED: Final[str] = "not_evaluated"

#: Codigos de bloqueo (estables). Cada uno nombra la regla que lo produce.
CODE_AS_OF_NOT_TODAY: Final[str] = "as_of_no_es_de_hoy"
CODE_SNAPSHOT_NOT_OK: Final[str] = "snapshot_no_valido"
CODE_MAX_ONE_TRADE: Final[str] = "max_una_operacion_por_sesion"
CODE_DAILY_LOSS: Final[str] = "perdida_diaria"
CODE_WEEKLY_LOSS: Final[str] = "perdida_semanal"
CODE_MONTHLY_LOSS: Final[str] = "perdida_mensual"
CODE_OBSERVATION_MODE: Final[str] = "modo_observacion"
CODE_FOMC_DAY: Final[str] = "dia_de_fomc"
CODE_HALF_SESSION: Final[str] = "media_sesion"
CODE_BRACKET_TARGET_MISSING: Final[str] = "bracket_sin_objetivo"
CODE_TARGET_BELOW_COST: Final[str] = "objetivo_bajo_el_coste"
CODE_EV_NET_NOT_COMPUTABLE: Final[str] = "ev_neto_no_calculable"
CODE_EV_BELOW_THRESHOLD: Final[str] = "ev_bajo_el_umbral"
CODE_TIER_NOT_AUTHORIZED: Final[str] = "tier_no_autorizado"

# ─────────────────────────────────────────────────────────────────────────────
# Las 18 reglas de §12 (una por una, con quien las aplica)
# ─────────────────────────────────────────────────────────────────────────────
#: Las 18 reglas duras de ``plan.md`` §12. ``owner`` dice quien las aplica en esta tarea
#: (``gate`` o la issue que las completa, ``issue``). El gate **no** las salta en silencio:
#: publica el resultado de cada una en ``GateOutput.rules``.
RULES: Final[tuple[dict[str, str], ...]] = (
    {
        "rule": "1",
        "title": "Maximo 1 operacion por sesion",
        "owner": "gate",
        "issue": "",
        "note": "input trades_today: >= 1 bloquea la sesion entera, sin excepciones",
    },
    {
        "rule": "2",
        "title": "Riesgo por operacion <= 1 % del capital",
        "owner": "gate",
        "issue": "#28",
        "note": (
            "por operacion: el nocional sale del riesgo declarado y de la distancia al stop; la "
            "lectura de cartera es #28 y el valor del techo lo declara #60"
        ),
    },
    {
        "rule": "3",
        "title": "Perdida diaria -2 % => kill switch",
        "owner": "gate",
        "issue": "#83",
        "note": "input daily_pnl_pct contra el umbral de #60; acumular el dia es #83",
    },
    {
        "rule": "4",
        "title": "Perdida semanal -5 % => parada",
        "owner": "gate",
        "issue": "#83",
        "note": "input weekly_pnl_pct contra el umbral de #60; acumular la semana es #83",
    },
    {
        "rule": "5",
        "title": "Perdida mensual -10 % => parada total",
        "owner": "gate",
        "issue": "#83",
        "note": "input monthly_pnl_pct contra el umbral de #60; acumular el mes es #83",
    },
    {
        "rule": "6",
        "title": "Sin posiciones overnight",
        "owner": "gate",
        "issue": "",
        "note": "nights = 0 siempre: el gate rechaza un CostBreakdown con nights > 0",
    },
    {
        "rule": "7",
        "title": "Stop obligatorio definido antes de entrar",
        "owner": "gate",
        "issue": "",
        "note": "stop_pct es obligatorio y > 0: una decision direccional sin stop es error tipado",
    },
    {
        "rule": "8",
        "title": "Objetivo >= 2x el coste de ida y vuelta",
        "owner": "gate",
        "issue": "",
        "note": "target_pct contra el coste declarado del bloque de #11 (c_declared_pct)",
    },
    {
        "rule": "9",
        "title": "EV neto > umbral de seguridad",
        "owner": "gate",
        "issue": "#60",
        "note": (
            "el umbral es params.ev_threshold_pct; sin slippage medido el EV neto es null y la "
            "regla no se puede verificar (nunca se sustituye por 0)"
        ),
    },
    {
        "rule": "10",
        "title": "Solo operar senales tier A al principio",
        "owner": "gate",
        "issue": "#60",
        "note": (
            "los tiers B y C se publican con status = recommendation y devuelven NOTHING; que "
            "tiers se autorizan lo declara params.authorized_tiers"
        ),
    },
    {
        "rule": "11",
        "title": "NOTHING por defecto: operar 10-30 % de los dias como maximo",
        "owner": "#28",
        "issue": "#28",
        "note": (
            "no es una propiedad por sesion: es propiedad del backtest y se mide al cablear el "
            "gate al motor (#28)"
        ),
    },
    {
        "rule": "12",
        "title": "Nunca ampliar perdedora ni mover el stop en contra",
        "owner": "gate",
        "issue": "",
        "note": (
            "estructural: el modulo es una funcion pura y sin estado y no expone ninguna accion "
            "de aumentar ni de modificar una posicion abierta"
        ),
    },
    {
        "rule": "13",
        "title": "Guardia de obsolescencia: el as_of tiene que ser de hoy",
        "owner": "gate",
        "issue": "#40",
        "note": (
            "as_of.date() != today => no_recommendation_stale_data; el modulo de la guardia "
            "completa es #40"
        ),
    },
    {
        "rule": "14",
        "title": '"No se" no es NOTHING: cinco estados separados',
        "owner": "gate",
        "issue": "#40",
        "note": (
            "los cuatro estados de §19.2 mas no_recommendation_undecided; el registro del "
            "historico es #40 y el diario #39"
        ),
    },
    {
        "rule": "15",
        "title": "Modo observacion de 5 sesiones tras una ausencia",
        "owner": "gate",
        "issue": "#40",
        "note": "input observation_sessions_remaining > 0 => NOTHING; persistir el contador es #40",
    },
    {
        "rule": "16",
        "title": "Cierre obligatorio: orden bracket (objetivo y stop)",
        "owner": "gate",
        "issue": "#84",
        "note": (
            "bracket_required = True siempre y stop + objetivo los dos o ninguno; colocar la "
            "orden contra el broker real es #84"
        ),
    },
    {
        "rule": "17",
        "title": "Dias de FOMC: NOTHING",
        "owner": "gate",
        "issue": "#34",
        "note": "input fomc_dates (conjunto declarado); ingerir el calendario es #34",
    },
    {
        "rule": "18",
        "title": "Medias sesiones: excluidas por defecto",
        "owner": "gate",
        "issue": "#40",
        "note": (
            "reutiliza MarketCalendar.is_half_day, sin re-derivar festivos ni medias sesiones; "
            "el flujo completo es #40"
        ),
    },
)

#: Que **no** hace este modulo, legible por maquina (A8, A11). Cada frontera con su issue.
LIMITATIONS: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "regla_11",
        "issue": "#28",
        "statement": (
            "la regla 11 (operar solo el 10-30 % de los dias) **no** se aplica por sesion: es "
            "propiedad del backtest y se mide al cablear el gate al motor (#28)"
        ),
    },
    {
        "id": "regla_2_cartera",
        "issue": "#28",
        "statement": (
            "el riesgo por operacion se aplica aqui; su lectura de cartera (y el limite agregado) "
            "se mide en #28"
        ),
    },
    {
        "id": "reglas_3_5_cartera",
        "issue": "#83",
        "statement": "el P&L diario, semanal y mensual llegan ya acumulados: acumularlos es #83",
    },
    {
        "id": "reglas_13_15_persistencia",
        "issue": "#40",
        "statement": (
            "la guardia de obsolescencia completa, los cinco estados en el registro y el contador "
            "del modo observacion son #40"
        ),
    },
    {
        "id": "regla_17_ingesta",
        "issue": "#34",
        "statement": "el calendario de FOMC se recibe declarado; ingerirlo es #34",
    },
    {
        "id": "regla_16_orden",
        "issue": "#84",
        "statement": (
            "el gate declara bracket_required = True; colocar la orden bracket contra el broker "
            "real es #84"
        ),
    },
    {
        "id": "precio_de_entrada",
        "issue": "#84",
        "statement": (
            "el gate publica stop_pct y target_pct (distancias en %), no precios: convertir un "
            "porcentaje en precio necesita el `open` de la subasta de apertura (fijado en #64), "
            "que entra al colocar la orden (#84) y al cablear el motor (#28). Por eso stop_px y "
            "target_px valen None: el gate no recibe ningun precio y no se inventa uno"
        ),
    },
    {
        "id": "regimen_volatilidad",
        "issue": "#28",
        "statement": (
            "de las condiciones del tier A, esta tarea evalua el EV neto (> multiplo del coste) y "
            "la probabilidad calibrada. El 'regimen de volatilidad favorable' no tiene entrada "
            "declarada y no se evalua aqui: queda declarado para #28"
        ),
    },
    {
        "id": "overlay_llm",
        "issue": "#35",
        "statement": "el veto y el ajuste de +-10 pp del LLM son #35: el gate no los aplica",
    },
    {
        "id": "diario",
        "issue": "#39",
        "statement": "persistir la decision en el diario es #39: el gate no escribe nada",
    },
    {
        "id": "slippage_no_medido",
        "issue": "#62",
        "statement": (
            "el slippage se consume tal cual llega de #11: medirlo es #62 y el tamano de R que lo "
            "convertiria en % del nocional es #60. El gate no fusiona los tres estados"
        ),
    },
    {
        "id": "unidades_del_motor",
        "issue": "#80",
        "statement": (
            "el defecto de unidades de pnl_declared_pct en backtest/engine.py:840 no se arregla "
            "aqui (#80) y no se replica: este modulo tiene una sola convencion, el porcentaje"
        ),
    },
    {
        "id": "estado_error",
        "issue": "#40",
        "statement": (
            "GateStatus.ERROR existe (esta declarado en §19.2) pero no lo emite este modulo: un "
            "fallo tecnico es del pipeline y se registra en el run_log (#40/#43). Una entrada "
            "inadmisible es GateInputError"
        ),
    },
)

#: Issues con las que este modulo comparte frontera.
FOLLOW_UPS: Final[tuple[str, ...]] = (
    "#28",
    "#34",
    "#35",
    "#39",
    "#40",
    "#59",
    "#60",
    "#62",
    "#80",
    "#83",
    "#84",
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores y estados
# ─────────────────────────────────────────────────────────────────────────────
class GateError(Exception):
    """Raiz de los errores del gate."""


class GateInputError(GateError):
    """Una entrada del gate no es admisible: es un error tipado, nunca un resultado silencioso."""


class GateStatus(StrEnum):
    """Los cuatro estados de ``plan.md`` §19.2 **mas** el estado bloqueado de §11 bis.

    ``recommendation`` cubre ``LONG``, ``SHORT`` y ``NOTHING``: "hoy no veo oportunidad" es una
    recomendacion, no un "no se". Los estados ``no_recommendation_*`` llevan ``direction =
    None`` y ningun nocional; ninguno se fusiona con ``NOTHING``.
    """

    RECOMMENDATION = "recommendation"
    NO_RECOMMENDATION_STALE_DATA = "no_recommendation_stale_data"
    NO_RECOMMENDATION_DATA_QUALITY = "no_recommendation_data_quality"
    ERROR = "error"
    NO_RECOMMENDATION_UNDECIDED = "no_recommendation_undecided"


# ─────────────────────────────────────────────────────────────────────────────
# Parametros declarados: lo que decide el propietario, con None = sin decidir
# ─────────────────────────────────────────────────────────────────────────────
#: Cada parametro, la issue que lo cierra cuando falta (``tech_stack.md`` §11 bis) y que es.
#: El orden es fijo: ``undecided[]`` y ``params`` se publican en este orden, nunca en el de un
#: ``set``, para que el hash no dependa de nada que no este declarado.
PARAMETER_ISSUES: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "broker",
        "#59",
        "el broker decide instrumento, horario y ejecucion del CFD: sin el no hay operativa real",
    ),
    (
        "risk_per_trade_pct",
        "#60",
        "riesgo por operacion en % del capital (regla 2): el gate lo usa para el sizing",
    ),
    (
        "ev_threshold_pct",
        "#60",
        "umbral de seguridad del EV neto en % (regla 9, 'p. ej. > 2c'): no se cablea",
    ),
    ("max_daily_loss_pct", "#60", "perdida diaria maxima en % (regla 3, -2 % en el enunciado)"),
    ("max_weekly_loss_pct", "#60", "perdida semanal maxima en % (regla 4, -5 % en el enunciado)"),
    ("max_monthly_loss_pct", "#60", "perdida mensual maxima en % (regla 5, -10 % en el enunciado)"),
    ("r_pct", "#60", "tamano de R en %: sin el, el supuesto de slippage no tiene numero"),
    (
        "tier_a_cost_multiple",
        "#60",
        "multiplo del coste para el tier A (3c en el enunciado): el gate no lo cablea",
    ),
    (
        "tier_b_cost_multiple",
        "#60",
        "multiplo del coste para el tier B (2c en el enunciado): el gate no lo cablea",
    ),
    (
        "tier_a_min_probability",
        "#60",
        "probabilidad calibrada minima del tier A (0,58 en el enunciado): el gate no la cablea",
    ),
    (
        "authorized_tiers",
        "#60",
        "que tiers se autorizan a operar (regla 10): al principio, solo el A",
    ),
)

#: Campos que, si se declaran, tienen que ser estrictamente positivos.
_POSITIVE_PARAMETER_FIELDS: Final[tuple[str, ...]] = (
    "risk_per_trade_pct",
    "max_daily_loss_pct",
    "max_weekly_loss_pct",
    "max_monthly_loss_pct",
    "r_pct",
    "tier_b_cost_multiple",
    "tier_a_cost_multiple",
)

#: Campos que admiten el 0: un umbral de EV de 0 significa "cualquier EV neto positivo".
_NON_NEGATIVE_PARAMETER_FIELDS: Final[tuple[str, ...]] = ("ev_threshold_pct",)


class GateParameters(BaseModel):
    """Las decisiones que el gate **no** puede inventar, con ``None`` = sin decidir.

    Cada campo ausente se publica en ``GateOutput.undecided`` con la issue que lo cierra (#59
    el broker, #60 el resto). Con todos los campos en ``None`` el gate devuelve
    ``no_recommendation_undecided``: es el estado bloqueado que obliga ``tech_stack.md``
    §11 bis mientras las decisiones abiertas 4 y 5 sigan abiertas.

    Los porcentajes van en puntos porcentuales como numero (``2`` son 2 %) y el tamano de ``R``
    tambien: **ninguna** de las cifras del enunciado (1 %, 2 %, 3c, 0,58) vive en el codigo.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    broker: str | None = Field(
        default=None, description="broker declarado (decision abierta 4 => #59)"
    )
    risk_per_trade_pct: Decimal | None = Field(
        default=None, description="riesgo por operacion en % del capital (regla 2, #60)"
    )
    ev_threshold_pct: Decimal | None = Field(
        default=None, description="umbral de seguridad del EV neto en % (regla 9, #60)"
    )
    max_daily_loss_pct: Decimal | None = Field(
        default=None, description="perdida diaria maxima en % (regla 3, #60)"
    )
    max_weekly_loss_pct: Decimal | None = Field(
        default=None, description="perdida semanal maxima en % (regla 4, #60)"
    )
    max_monthly_loss_pct: Decimal | None = Field(
        default=None, description="perdida mensual maxima en % (regla 5, #60)"
    )
    r_pct: Decimal | None = Field(
        default=None, description="tamano de R en %: el supuesto de slippage se declara sobre R"
    )
    tier_a_cost_multiple: Decimal | None = Field(
        default=None, description="multiplo del coste exigido al tier A (regla 10, #60)"
    )
    tier_b_cost_multiple: Decimal | None = Field(
        default=None, description="multiplo del coste exigido al tier B (regla 10, #60)"
    )
    tier_a_min_probability: Decimal | None = Field(
        default=None,
        description="probabilidad calibrada minima del tier A (regla 10, #60), entre 0 y 1",
    )
    authorized_tiers: tuple[str, ...] | None = Field(
        default=None,
        description="tiers autorizados a operar (regla 10): subconjunto de (A, B, C)",
    )

    def model_post_init(self, _context: object, /) -> None:
        """Coherencia de los parametros **declarados**: un valor mal declarado es error tipado.

        Un campo en ``None`` es "sin decidir" y no se valida: lo que se rechaza es un valor
        declarado que no tiene sentido (un riesgo de 0, una probabilidad de 1,4, un tier que no
        existe).
        """
        declared = self.model_fields_set
        if not declared:
            return  # `GateParameters()` sin nada declarado: no hay nada que validar
        broker = self.broker
        if "broker" in declared and broker is not None and not broker.strip():
            raise GateInputError("params.broker: un broker declarado no puede estar en blanco")
        for name in _POSITIVE_PARAMETER_FIELDS:
            value = cast("Decimal | None", getattr(self, name))
            if value is not None and value <= 0:
                raise GateInputError(
                    f"params.{name}: un valor declarado tiene que ser estrictamente positivo "
                    f"(llego {_num(value)}); `None` es 'sin decidir', no un cero"
                )
        for name in _NON_NEGATIVE_PARAMETER_FIELDS:
            value = cast("Decimal | None", getattr(self, name))
            if value is not None and value < 0:
                raise GateInputError(
                    f"params.{name}: un umbral negativo no admite ninguna operacion (llego "
                    f"{_num(value)}); 0 significa 'cualquier EV neto positivo'"
                )
        probability = self.tier_a_min_probability
        if probability is not None and not Decimal(0) < probability < Decimal(1):
            raise GateInputError(
                "params.tier_a_min_probability: es una probabilidad, tiene que estar entre 0 y 1 "
                f"(llego {_num(probability)})"
            )
        tiers = self.authorized_tiers
        if tiers is not None and not set(tiers) <= set(TIERS):
            raise GateInputError(
                f"params.authorized_tiers: solo se autorizan tiers declarados {TIERS} "
                f"(llego {tiers!r})"
            )
        low = self.tier_b_cost_multiple
        high = self.tier_a_cost_multiple
        if low is not None and high is not None and high < low:
            raise GateInputError(
                "params.tier_a_cost_multiple/params.tier_b_cost_multiple: el tier A no puede ser "
                f"menos exigente que el B (A = {_num(high)}, B = {_num(low)})"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Salida del gate
# ─────────────────────────────────────────────────────────────────────────────
class GateOutput(BaseModel):
    """La salida completa del gate: la decision, su trazabilidad y el hash de todo lo anterior.

    ``direction`` es ``None`` en los estados "no se" (no hay juicio que hacer) y
    ``Direction.NOTHING`` cuando si lo hay y la conclusion es "hoy no veo oportunidad".
    ``tier`` es el tier **derivado** (``C`` si no se puede demostrar el A ni el B: sin EV neto
    no hay multiplo que comparar), ``blockers[]`` lleva **todas** las reglas que bloquearon la
    sesion y ``undecided[]`` los parametros sin decidir con su issue.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session: date = Field(description="sesion evaluada (ET)")
    as_of: datetime = Field(description="instante del snapshot que se evalua")
    today: date = Field(description="fecha de hoy en ET: entrada explicita, el gate no tiene reloj")
    status: GateStatus = Field(description="uno de los cinco estados; ninguno se fusiona")
    direction: Direction | None = Field(
        default=None, description="LONG/SHORT/NOTHING; None en los estados 'no se'"
    )
    tier: Tier = Field(description="tier derivado: A, B o C (los umbrales son de #60)")
    prob_up_calibrated: float = Field(description="probabilidad calibrada consumida de #26")
    expected_move_pct: Decimal = Field(description="movimiento esperado a favor, en %")
    expected_move_basis: str = Field(description="base declarada de `expected_move_pct`")
    cost_pct: Decimal = Field(description="coste declarado de ida y vuelta en % (c_declared_pct)")
    cost_total_pct: Decimal | None = Field(
        default=None, description="coste total en %: null mientras el slippage no sea medido"
    )
    slippage_state: str = Field(description="measured / assumed / unmeasured, tal cual llego")
    ev_declared_pct: Decimal | None = Field(
        default=None,
        description="EV con el coste declarado, en %: se publica siempre que haya probabilidad",
    )
    ev_net_pct: Decimal | None = Field(
        default=None, description="EV con el coste total, en %: null si el total es null, nunca 0"
    )
    stop_pct: Decimal = Field(description="distancia al stop en % del precio")
    target_pct: Decimal | None = Field(
        default=None, description="distancia al objetivo en %; None = sin objetivo declarado"
    )
    stop_px: float | None = Field(
        default=None,
        description="precio del stop: None porque el gate no recibe precio (la conversion, #84)",
    )
    target_px: float | None = Field(
        default=None, description="precio del objetivo: None por el mismo motivo"
    )
    notional_usd: Decimal | None = Field(
        default=None, description="nocional desde el riesgo y el stop; None si no se opera"
    )
    leverage_implied: Decimal | None = Field(
        default=None, description="apalancamiento **derivado**: notional / capital, nunca un input"
    )
    bracket_required: bool = Field(
        default=True, description="regla 16: el cierre es obligacion, la orden la coloca #84"
    )
    trades_today: int = Field(description="operaciones ya hechas hoy (regla 1, input)")
    observation_sessions_remaining: int = Field(
        description="sesiones de observacion que quedan (regla 15, input)"
    )
    is_fomc_session: bool = Field(description="regla 17: la sesion esta en el calendario de FOMC")
    is_half_session: bool = Field(description="regla 18: media sesion segun el MarketCalendar")
    fomc_dates_count: int = Field(description="tamano del conjunto de FOMC declarado")
    params: dict[str, str | None] = Field(
        description="parametros declarados en forma exacta (Decimal como cadena), tal cual"
    )
    blockers: tuple[dict[str, str], ...] = Field(
        default=(),
        description="todas las reglas que bloquearon la sesion, con su numero y su codigo",
    )
    undecided: tuple[dict[str, str], ...] = Field(
        default=(), description="parametros sin decidir con su issue (#59/#60)"
    )
    rules: tuple[dict[str, str], ...] = Field(
        default=(), description="resultado de cada una de las 18 reglas de §12"
    )
    gate_sha256: str = Field(description=f"{GATE_HASH_PREFIX}<64 hex>; ver GATE_HASH_FORMAT")

    def to_engine_decision(self, *, entry_px: float | None = None) -> Decision:
        """Atajo de ``to_engine_decision(output, entry_px=...)`` (A4)."""
        return to_engine_decision(self, entry_px=entry_px)


def _num(value: Decimal) -> str:
    """``Decimal`` -> cadena decimal exacta, sin notacion cientifica."""
    return format(value, "f")


def _probability_for(direction: Direction, probability: float) -> Decimal:
    """La probabilidad a favor de la direccion, exacta: ``p`` en largo y ``1 - p`` en corto."""
    value = probability if direction is Direction.LONG else 1.0 - probability
    return Decimal(str(value))


def _json_payload(output: GateOutput) -> dict[str, object]:
    """El payload canonico que se hashea (A2): solo tipos JSON puros.

    Se construye campo a campo y en este orden: nada depende del orden de un ``dict`` ni de la
    representacion de un ``Decimal``, asi que dos procesos con ``PYTHONHASHSEED`` distinto
    publican el mismo ``gate_sha256``.
    """
    return {
        "session": output.session.isoformat(),
        "as_of": output.as_of.isoformat(),
        "today": output.today.isoformat(),
        "status": output.status.value,
        "direction": None if output.direction is None else output.direction.value,
        "tier": output.tier,
        "prob_up_calibrated": output.prob_up_calibrated,
        "expected_move_pct": _num(output.expected_move_pct),
        "expected_move_basis": output.expected_move_basis,
        "cost_pct": _num(output.cost_pct),
        "cost_total_pct": None if output.cost_total_pct is None else _num(output.cost_total_pct),
        "slippage_state": output.slippage_state,
        "ev_declared_pct": None if output.ev_declared_pct is None else _num(output.ev_declared_pct),
        "ev_net_pct": None if output.ev_net_pct is None else _num(output.ev_net_pct),
        "stop_pct": _num(output.stop_pct),
        "target_pct": None if output.target_pct is None else _num(output.target_pct),
        "notional_usd": None if output.notional_usd is None else _num(output.notional_usd),
        "leverage_implied": (
            None if output.leverage_implied is None else _num(output.leverage_implied)
        ),
        "bracket_required": output.bracket_required,
        "trades_today": output.trades_today,
        "observation_sessions_remaining": output.observation_sessions_remaining,
        "is_fomc_session": output.is_fomc_session,
        "is_half_session": output.is_half_session,
        "fomc_dates_count": output.fomc_dates_count,
        "params": {name: output.params[name] for name, _issue, _note in PARAMETER_ISSUES},
        "blockers": [dict(entry) for entry in output.blockers],
        "undecided": [dict(entry) for entry in output.undecided],
        "rules": [dict(entry) for entry in output.rules],
    }


def gate_sha256(output: GateOutput) -> str:
    """``gate_sha256`` del payload **sin** la clave del hash (``GATE_HASH_FORMAT``, A2)."""
    digest = hashlib.sha256(canonical_text(_json_payload(output)).encode("utf-8")).hexdigest()
    return f"{GATE_HASH_PREFIX}{digest}"


def to_engine_decision(output: GateOutput, *, entry_px: float | None = None) -> Decision:
    """Traduce la salida del gate al ``Decision`` que el motor de #13 ya valida (A4, A19, A23).

    - ``NOTHING`` o un estado sin juicio (``direction is None``) devuelven una ``Decision``
      ``NOTHING`` **sin nocional**: el motor no le exige nocional a ``NOTHING`` (A19).
    - ``LONG``/``SHORT`` llevan el **nocional positivo** que el gate ya calculo (A19) y las
      barreras van **las dos o ninguna** (A23). Con ``entry_px`` —el ``open`` de la subasta de
      apertura, que fijo #64— se convierten ``stop_pct`` y ``target_pct`` en precios, de forma
      que la geometria (``stop < open < target`` en largo y la espejo en corto) se cumple **por
      construccion**; sin ``entry_px`` las dos barreras van a ``None``, porque el gate no recibe
      ningun precio y no se inventa uno.
    """
    output = _require_output(output)
    entry = _price_or_none(entry_px, field_name="entry_px")
    direction = output.direction
    if direction is None or direction is Direction.NOTHING:
        return Decision(
            direction=Direction.NOTHING,
            reason=_decision_reason(output),
            probability=output.prob_up_calibrated,
        )
    notional = output.notional_usd
    if not isinstance(notional, Decimal) or not notional.is_finite() or notional <= 0:
        raise GateInputError(
            "output.notional_usd: una decision direccional del gate lleva siempre un Decimal "
            f"positivo (llego {notional!r}); el motor lo exige (A19)"
        )
    stop_px, target_px = _barrier_prices(output, entry)
    return Decision(
        direction=direction,
        reason=_decision_reason(output),
        stop_px=stop_px,
        target_px=target_px,
        notional_usd=notional,
        probability=output.prob_up_calibrated,
    )


def _require_output(value: object) -> GateOutput:
    """El contrato de runtime de A4: un llamante sin tipar recibe un error claro (A4)."""
    if not isinstance(value, GateOutput):
        raise GateInputError(f"output: se espera GateOutput, no {type(value).__name__} (A4)")
    return value


def _price_or_none(value: object, *, field_name: str) -> float | None:
    """Un precio de entrada opcional: un numero finito y positivo, o ``None`` (nunca se inventa)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateInputError(
            f"{field_name}: se espera un numero (el `open` de la subasta) o None, no "
            f"{type(value).__name__}"
        )
    price = float(value)
    if not 0.0 < price < float("inf"):
        raise GateInputError(
            f"{field_name}: tiene que ser un precio finito y positivo, llego {value!r}"
        )
    return price


def _barrier_prices(
    output: GateOutput, entry_px: float | None
) -> tuple[float | None, float | None]:
    """Las dos barreras (precio) o ninguna: el gate no publica una sola (A23)."""
    if entry_px is None or output.target_pct is None:
        return None, None
    stop_fraction = float(output.stop_pct) / 100.0
    target_fraction = float(output.target_pct) / 100.0
    if output.direction is Direction.LONG:
        return entry_px * (1.0 - stop_fraction), entry_px * (1.0 + target_fraction)
    return entry_px * (1.0 + stop_fraction), entry_px * (1.0 - target_fraction)


def _decision_reason(output: GateOutput) -> str:
    """El motivo que viaja al motor: estado, direccion, tier y bloqueos, sin adornos."""
    direction = "nothing" if output.direction is None else output.direction.value
    parts = [f"status={output.status.value}", f"direction={direction}", f"tier={output.tier}"]
    if output.ev_net_pct is not None:
        parts.append(f"ev_net_pct={_num(output.ev_net_pct)}")
    for blocker in output.blockers:
        parts.append(f"{blocker['code']}(regla {blocker['rule']})")
    return "; ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Registro de las 18 reglas
# ─────────────────────────────────────────────────────────────────────────────
class _RuleLedger:
    """El paso de cada una de las 18 reglas de §12, en el orden declarado en ``RULES``.

    Es estado **local** de una llamada: el registro nace y muere dentro de ``evaluate_gate``
    (la funcion sigue siendo pura) y el resultado se publica en ``GateOutput.rules``.
    """

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, str]] = {
            entry["rule"]: {**entry, "outcome": _RULE_NOT_EVALUATED, "detail": ""}
            for entry in RULES
        }
        self.mark("6", _RULE_OK, "nights = 0 en el CostBreakdown: intradia puro")
        self.mark("7", _RULE_OK, "stop_pct declarado y > 0: el stop llega ya calculado")
        self.mark(
            "12",
            _RULE_OK,
            "funcion pura y sin estado: no existe accion de ampliar ni de mover el stop",
        )
        self.mark(
            "11",
            _RULE_DELEGATED,
            "propiedad del backtest, no de la sesion: se mide al cablear el motor (#28)",
        )

    def mark(self, rule: str, outcome: str, detail: str) -> None:
        """Anota el resultado de una regla (``pass``, ``blocked``, ``delegated``)."""
        entry = self._entries[rule]
        entry["outcome"] = outcome
        entry["detail"] = detail

    def block(self, rule: str, detail: str) -> None:
        """Anota que una regla bloqueo la sesion."""
        self.mark(rule, _RULE_BLOCKED, detail)

    def entries(self) -> tuple[dict[str, str], ...]:
        """Las 18 reglas, en el orden declarado (nunca en el orden de un ``set``)."""
        return tuple(self._entries[entry["rule"]] for entry in RULES)


# ─────────────────────────────────────────────────────────────────────────────
# Validacion de las entradas y derivacion comun
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class _Decided:
    """Los parametros de politica, ya garantizados no nulos.

    Solo se construye despues de comprobar que ``missing`` esta vacio: el gate no evalua una
    regla con un umbral que nadie ha decidido.
    """

    broker: str
    risk_per_trade_pct: Decimal
    ev_threshold_pct: Decimal
    max_daily_loss_pct: Decimal
    max_weekly_loss_pct: Decimal
    max_monthly_loss_pct: Decimal
    r_pct: Decimal
    tier_a_cost_multiple: Decimal
    tier_b_cost_multiple: Decimal
    tier_a_min_probability: Decimal
    authorized_tiers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Context:
    """Las entradas ya validadas mas lo que se deriva una sola vez (direccion, EV, calendario)."""

    session: date
    as_of: datetime
    today: date
    direction: Direction
    prob_up_calibrated: float
    expected_move_pct: Decimal
    expected_move_basis: str
    cost: CostBreakdown
    cost_pct: Decimal
    cost_total_pct: Decimal | None
    ev_declared_pct: Decimal
    ev_net_pct: Decimal | None
    capital_usd: Decimal
    snapshot_ok: bool
    stop_pct: Decimal
    target_pct: Decimal | None
    params: GateParameters
    missing: tuple[dict[str, str], ...]
    trades_today: int
    daily_pnl_pct: Decimal | None
    weekly_pnl_pct: Decimal | None
    monthly_pnl_pct: Decimal | None
    observation_sessions_remaining: int
    is_fomc_session: bool
    is_half_session: bool
    fomc_dates_count: int
    ledger: _RuleLedger = field(default_factory=_RuleLedger)


def _require_decimal(value: object, *, field_name: str) -> Decimal:
    """Un importe o un porcentaje son ``Decimal`` exactos: un ``float`` es error tipado."""
    if isinstance(value, bool) or not isinstance(value, Decimal):
        raise GateInputError(
            f"{field_name}: se espera un `decimal.Decimal` exacto (el dinero y los porcentajes no "
            f"son `float`), no {type(value).__name__}"
        )
    if not value.is_finite():
        raise GateInputError(f"{field_name}: se espera un Decimal finito, llego {value!r}")
    return value


def _optional_decimal(value: object, *, field_name: str) -> Decimal | None:
    """Un porcentaje opcional: ``None`` sigue siendo ``None`` (nunca se rellena con 0)."""
    if value is None:
        return None
    return _require_decimal(value, field_name=field_name)


def _require_int(value: object, *, field_name: str, minimum: int) -> int:
    """Un contador entero (``bool`` no cuenta) con su minimo declarado."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateInputError(f"{field_name}: se espera un entero, no {type(value).__name__}")
    if value < minimum:
        raise GateInputError(f"{field_name}: el minimo es {minimum}, llego {value}")
    return value


def _require_probability(value: object, *, field_name: str) -> float:
    """Una probabilidad calibrada en ``[0, 1]``, como ``float`` (lo que publica #26)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateInputError(
            f"{field_name}: se espera un `float` calibrado de #26, no {type(value).__name__}"
        )
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise GateInputError(f"{field_name}: es una probabilidad, tiene que estar en [0, 1]")
    return probability


def _require_iso_date(value: object, *, field_name: str) -> date:
    """Una fecha de calendario: un ``datetime`` no cuela como dia (arrastraria una hora)."""
    if isinstance(value, datetime) or not isinstance(value, date):
        raise GateInputError(
            f"{field_name}: se espera una `datetime.date`, no {type(value).__name__}"
        )
    return value


def _require_instant(value: object, *, field_name: str) -> datetime:
    """Un instante: el ``as_of`` del snapshot con el que se decide."""
    if not isinstance(value, datetime):
        raise GateInputError(
            f"{field_name}: se espera un `datetime.datetime`, no {type(value).__name__}"
        )
    return value


def _require_text(value: object, *, field_name: str) -> str:
    """Una cadena no vacia."""
    if not isinstance(value, str) or not value.strip():
        raise GateInputError(f"{field_name}: se espera una cadena no vacia")
    return value


def _require_flag(value: object, *, field_name: str) -> bool:
    """Un booleano de verdad: un ``0``/``1`` silencioso es un contrato roto, no un flag."""
    if not isinstance(value, bool):
        raise GateInputError(f"{field_name}: se espera un `bool`, no {type(value).__name__}")
    return value


def _missing_parameters(params: GateParameters) -> tuple[dict[str, str], ...]:
    """Los parametros sin decidir, en el orden declarado y cada uno con su issue (A8)."""
    return tuple(
        {"parameter": name, "issue": issue, "detail": note}
        for name, issue, note in PARAMETER_ISSUES
        if getattr(params, name) is None
    )


def _decided(params: GateParameters) -> _Decided:
    """Los parametros declarados, sin ninguno nulo: llamar **solo** con ``missing`` vacio."""
    return _Decided(
        broker=cast("str", params.broker),
        risk_per_trade_pct=cast("Decimal", params.risk_per_trade_pct),
        ev_threshold_pct=cast("Decimal", params.ev_threshold_pct),
        max_daily_loss_pct=cast("Decimal", params.max_daily_loss_pct),
        max_weekly_loss_pct=cast("Decimal", params.max_weekly_loss_pct),
        max_monthly_loss_pct=cast("Decimal", params.max_monthly_loss_pct),
        r_pct=cast("Decimal", params.r_pct),
        tier_a_cost_multiple=cast("Decimal", params.tier_a_cost_multiple),
        tier_b_cost_multiple=cast("Decimal", params.tier_b_cost_multiple),
        tier_a_min_probability=cast("Decimal", params.tier_a_min_probability),
        authorized_tiers=cast("tuple[str, ...]", params.authorized_tiers),
    )


def _parameters_payload(params: GateParameters) -> dict[str, str | None]:
    """Los parametros declarados en forma exacta: ``Decimal`` como cadena, ``None`` sin decidir."""
    payload: dict[str, str | None] = {}
    for name, _issue, _note in PARAMETER_ISSUES:
        value = getattr(params, name)
        if value is None:
            payload[name] = None
        elif isinstance(value, Decimal):
            payload[name] = _num(value)
        elif isinstance(value, str):
            payload[name] = value
        else:
            payload[name] = ",".join(cast("tuple[str, ...]", value))
    return payload


def _context(
    *,
    session: object,
    as_of: object,
    today: object,
    calendar: object,
    prob_up_calibrated: object,
    expected_move_pct: object,
    expected_move_basis: object,
    cost: object,
    capital_usd: object,
    snapshot_ok: object,
    stop_pct: object,
    target_pct: object,
    fomc_dates: object,
    params: object,
    trades_today: object,
    daily_pnl_pct: object,
    weekly_pnl_pct: object,
    monthly_pnl_pct: object,
    observation_sessions_remaining: object,
) -> _Context:
    """Valida las entradas, aplica las reglas 6 y 7 y deriva lo comun (direccion, EV, calendario).

    Los parametros van anotados como ``object`` a proposito: el contrato del gate se comprueba
    aqui con errores tipados, de forma que un llamante sin tipar reciba un ``GateInputError``
    claro y nunca un ``AttributeError`` suelto.
    """
    day = _require_iso_date(session, field_name="session")
    instant = _require_instant(as_of, field_name="as_of")
    now_day = _require_iso_date(today, field_name="today")
    if not isinstance(calendar, MarketCalendar):
        raise GateInputError(
            f"calendar: se espera un MarketCalendar ya construido, no {type(calendar).__name__}"
        )
    if not isinstance(cost, CostBreakdown):
        raise GateInputError(
            f"cost: se espera el CostBreakdown de #11 (`cost_breakdown`), no {type(cost).__name__}"
        )
    if not isinstance(params, GateParameters):
        raise GateInputError(f"params: se espera GateParameters, no {type(params).__name__}")
    if cost.nights != 0:  # regla 6: intradia puro
        raise GateInputError(
            f"cost.nights = {cost.nights}: la regla 6 no admite una posicion overnight; el coste "
            "tiene que llegar con nights = 0"
        )
    probability = _require_probability(prob_up_calibrated, field_name="prob_up_calibrated")
    move = _require_decimal(expected_move_pct, field_name="expected_move_pct")
    if move < 0:
        raise GateInputError(
            f"expected_move_pct: un movimiento esperado negativo no tiene sentido (llego "
            f"{_num(move)})"
        )
    basis = _require_text(expected_move_basis, field_name="expected_move_basis")
    capital = _require_decimal(capital_usd, field_name="capital_usd")
    if capital <= 0:
        raise GateInputError(
            "capital_usd: el capital tiene que ser positivo para poder dimensionar"
        )
    stop = _require_decimal(stop_pct, field_name="stop_pct")
    if stop <= 0:  # regla 7
        raise GateInputError(
            f"stop_pct: la regla 7 exige un stop definido antes de entrar y > 0 (llego "
            f"{_num(stop)})"
        )
    target = _optional_decimal(target_pct, field_name="target_pct")
    if target is not None and target <= 0:
        raise GateInputError(
            f"target_pct: un objetivo declarado tiene que ser > 0 (llego {_num(target)})"
        )
    if not isinstance(fomc_dates, Collection) or isinstance(fomc_dates, (str, bytes)):
        raise GateInputError(
            "fomc_dates: se espera una coleccion de fechas (conjunto declarado, #34), no "
            f"{type(fomc_dates).__name__}"
        )
    declared_fomc = frozenset(
        _require_iso_date(item, field_name="fomc_dates[]")
        for item in cast("Collection[object]", fomc_dates)
    )
    trades = _require_int(trades_today, field_name="trades_today", minimum=0)
    observation = _require_int(
        observation_sessions_remaining, field_name="observation_sessions_remaining", minimum=0
    )
    daily = _optional_decimal(daily_pnl_pct, field_name="daily_pnl_pct")
    weekly = _optional_decimal(weekly_pnl_pct, field_name="weekly_pnl_pct")
    monthly = _optional_decimal(monthly_pnl_pct, field_name="monthly_pnl_pct")
    flag = _require_flag(snapshot_ok, field_name="snapshot_ok")

    direction = Direction.LONG if probability >= DECISION_THRESHOLD else Direction.SHORT
    cost_pct = cost.c_declared_pct
    cost_total_pct = cost.c_total_pct
    favourable = _probability_for(direction, probability)
    return _Context(
        session=day,
        as_of=instant,
        today=now_day,
        direction=direction,
        prob_up_calibrated=probability,
        expected_move_pct=move,
        expected_move_basis=basis,
        cost=cost,
        cost_pct=cost_pct,
        cost_total_pct=cost_total_pct,
        ev_declared_pct=favourable * move - cost_pct,
        ev_net_pct=None if cost_total_pct is None else favourable * move - cost_total_pct,
        capital_usd=capital,
        snapshot_ok=flag,
        stop_pct=stop,
        target_pct=target,
        params=params,
        missing=_missing_parameters(params),
        trades_today=trades,
        daily_pnl_pct=daily,
        weekly_pnl_pct=weekly,
        monthly_pnl_pct=monthly,
        observation_sessions_remaining=observation,
        is_fomc_session=day in declared_fomc,
        is_half_session=calendar.is_half_day(day),
        fomc_dates_count=len(declared_fomc),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bloqueos de sesion: reglas 1, 3, 4, 5, 15, 17 y 18
# ─────────────────────────────────────────────────────────────────────────────
def _loss_blocker(
    *,
    rule: str,
    code: str,
    label: str,
    value: Decimal | None,
    limit: Decimal,
    ledger: _RuleLedger,
) -> dict[str, str] | None:
    """Reglas 3, 4 y 5: la perdida realizada contra su limite (el signo es el de §12)."""
    if value is None:
        ledger.mark(
            rule, _RULE_OK, f"sin {label} declarado en esta llamada: no hay perdida que comparar"
        )
        return None
    threshold = -limit
    if value <= threshold:
        detail = (
            f"{label} = {_num(value)} <= {_num(threshold)} (limite declarado {_num(limit)}): la "
            f"regla {rule} para la operativa"
        )
        ledger.block(rule, detail)
        return {"rule": rule, "code": code, "detail": detail}
    ledger.mark(rule, _RULE_OK, f"{label} = {_num(value)} > {_num(threshold)}: dentro del limite")
    return None


def _blockers(context: _Context, decided: _Decided) -> tuple[dict[str, str], ...]:
    """Todas las reglas de bloqueo de la sesion, en orden. No se corta en la primera (A9)."""
    ledger = context.ledger
    blockers: list[dict[str, str]] = []
    if context.trades_today >= 1:
        detail = (
            f"trades_today = {context.trades_today}: la regla 1 admite una sola operacion por "
            "sesion, sin excepciones"
        )
        ledger.block("1", detail)
        blockers.append({"rule": "1", "code": CODE_MAX_ONE_TRADE, "detail": detail})
    else:
        ledger.mark("1", _RULE_OK, "trades_today = 0: no se ha operado hoy")
    for loss in (
        _loss_blocker(
            rule="3",
            code=CODE_DAILY_LOSS,
            label="daily_pnl_pct",
            value=context.daily_pnl_pct,
            limit=decided.max_daily_loss_pct,
            ledger=ledger,
        ),
        _loss_blocker(
            rule="4",
            code=CODE_WEEKLY_LOSS,
            label="weekly_pnl_pct",
            value=context.weekly_pnl_pct,
            limit=decided.max_weekly_loss_pct,
            ledger=ledger,
        ),
        _loss_blocker(
            rule="5",
            code=CODE_MONTHLY_LOSS,
            label="monthly_pnl_pct",
            value=context.monthly_pnl_pct,
            limit=decided.max_monthly_loss_pct,
            ledger=ledger,
        ),
    ):
        if loss is not None:
            blockers.append(loss)
    if context.observation_sessions_remaining > 0:
        detail = (
            f"observation_sessions_remaining = {context.observation_sessions_remaining}: la regla "
            "15 exige terminar el modo observacion antes de volver a operar"
        )
        ledger.block("15", detail)
        blockers.append({"rule": "15", "code": CODE_OBSERVATION_MODE, "detail": detail})
    else:
        ledger.mark("15", _RULE_OK, "sin sesiones de observacion pendientes")
    if context.is_fomc_session:
        detail = (
            f"{context.session.isoformat()} esta en el conjunto de FOMC declarado: la regla 17 no "
            "opera en dias de FOMC (#34 ingiere el calendario)"
        )
        ledger.block("17", detail)
        blockers.append({"rule": "17", "code": CODE_FOMC_DAY, "detail": detail})
    else:
        ledger.mark("17", _RULE_OK, "la sesion no esta en el conjunto de FOMC declarado")
    if context.is_half_session:
        detail = (
            f"MarketCalendar.is_half_day({context.session.isoformat()}) = True: la regla 18 "
            "excluye las medias sesiones (el rango esperado cae en torno a un 45 %)"
        )
        ledger.block("18", detail)
        blockers.append({"rule": "18", "code": CODE_HALF_SESSION, "detail": detail})
    else:
        ledger.mark("18", _RULE_OK, "la sesion no es media sesion segun el calendario")
    return tuple(blockers)


# ─────────────────────────────────────────────────────────────────────────────
# Tier (regla 10) y reglas 16, 8, 9 y el sizing (regla 2)
# ─────────────────────────────────────────────────────────────────────────────
def _tier(context: _Context, decided: _Decided) -> Tier:
    """El tier derivado de §12: A si supera su multiplo del coste **y** su probabilidad minima.

    Sin EV neto (``c_total_pct`` es ``null``) no hay multiplo que comparar: el tier es ``C``,
    porque el A y el B se definen sobre el **EV neto** y un supuesto no los demuestra.
    """
    ev_net = context.ev_net_pct
    if ev_net is None:
        return TIER_C
    favourable = _probability_for(context.direction, context.prob_up_calibrated)
    if (
        ev_net > decided.tier_a_cost_multiple * context.cost_pct
        and favourable > decided.tier_a_min_probability
    ):
        return TIER_A
    if ev_net > decided.tier_b_cost_multiple * context.cost_pct:
        return TIER_B
    return TIER_C


def _sizing(
    *, capital_usd: Decimal, risk_per_trade_pct: Decimal, stop_pct: Decimal
) -> tuple[Decimal, Decimal]:
    """A5: el nocional sale del **riesgo** y de la distancia al stop, nunca del apalancamiento.

    ``notional_usd = capital x riesgo / distancia_al_stop`` y el apalancamiento es **derivado**
    (``notional / capital``), no una entrada: dimensionar desde el apalancamiento invertiria el
    orden y arriesgaria mas cuanto mas se apalanque la cuenta.
    """
    with localcontext() as decimal_context:
        decimal_context.prec = MONEY_PRECISION
        notional = (capital_usd * risk_per_trade_pct / stop_pct).quantize(
            NOTIONAL_QUANTUM, rounding=ROUND_HALF_UP
        )
        leverage = (notional / capital_usd).quantize(LEVERAGE_QUANTUM, rounding=ROUND_HALF_UP)
    return notional, leverage


def _trading_rules(
    context: _Context, decided: _Decided, tier: Tier
) -> tuple[tuple[dict[str, str], ...], Decimal | None, Decimal | None]:
    """Reglas 16, 8, 9 y 10 mas el *sizing* (regla 2): solo corre sin bloqueos de sesion.

    Devuelve los bloqueos de estas reglas y, si no hay ninguno, el nocional y el apalancamiento.
    """
    ledger = context.ledger
    blockers: list[dict[str, str]] = []
    target = context.target_pct
    if target is None:
        detail = (
            "target_pct = None: la regla 16 exige una orden bracket (objetivo y stop) y el "
            "objetivo no viene declarado; sin objetivo no hay cierre garantizado"
        )
        ledger.block("16", detail)
        blockers.append({"rule": "16", "code": CODE_BRACKET_TARGET_MISSING, "detail": detail})
    else:
        ledger.mark("16", _RULE_OK, "objetivo declarado: el bracket lleva las dos barreras")
        minimum = TARGET_MIN_COST_MULTIPLE * context.cost_pct
        if target < minimum:
            detail = (
                f"target_pct = {_num(target)} < {_num(minimum)} = 2 x coste declarado "
                f"({_num(context.cost_pct)}): la regla 8 no lo admite"
            )
            ledger.block("8", detail)
            blockers.append({"rule": "8", "code": CODE_TARGET_BELOW_COST, "detail": detail})
        else:
            ledger.mark(
                "8",
                _RULE_OK,
                f"target_pct = {_num(target)} >= {_num(minimum)} = 2 x coste declarado",
            )
    ev_net = context.ev_net_pct
    if ev_net is None:
        detail = (
            f"slippage.state = '{context.cost.slippage.state.value}': c_total_pct es null, el EV "
            "neto tambien y la regla 9 no se puede verificar (nunca se sustituye por 0). Medir el "
            "slippage es #62 y el tamano de R que lo convertiria en % del nocional es #60"
        )
        ledger.block("9", detail)
        blockers.append({"rule": "9", "code": CODE_EV_NET_NOT_COMPUTABLE, "detail": detail})
    elif ev_net <= decided.ev_threshold_pct:
        detail = (
            f"ev_net_pct = {_num(ev_net)} <= umbral declarado {_num(decided.ev_threshold_pct)}: "
            "la regla 9 exige un EV neto estrictamente mayor"
        )
        ledger.block("9", detail)
        blockers.append({"rule": "9", "code": CODE_EV_BELOW_THRESHOLD, "detail": detail})
    else:
        ledger.mark(
            "9",
            _RULE_OK,
            f"ev_net_pct = {_num(ev_net)} > umbral declarado {_num(decided.ev_threshold_pct)}",
        )
    if tier not in decided.authorized_tiers:
        detail = (
            f"tier {tier} no esta entre los tiers autorizados {decided.authorized_tiers}: la regla "
            "10 lo registra y devuelve NOTHING (no es un 'no se')"
        )
        ledger.block("10", detail)
        blockers.append({"rule": "10", "code": CODE_TIER_NOT_AUTHORIZED, "detail": detail})
    else:
        ledger.mark("10", _RULE_OK, f"tier {tier} autorizado a operar")
    if blockers:
        return tuple(blockers), None, None
    notional, leverage = _sizing(
        capital_usd=context.capital_usd,
        risk_per_trade_pct=decided.risk_per_trade_pct,
        stop_pct=context.stop_pct,
    )
    ledger.mark(
        "2",
        _RULE_OK,
        f"nocional {_num(notional)} USD = capital {_num(context.capital_usd)} x riesgo "
        f"{_num(decided.risk_per_trade_pct)} % / stop {_num(context.stop_pct)} %; el techo del 1 % "
        "lo declara #60 y la lectura de cartera es #28",
    )
    return (), notional, leverage


# ─────────────────────────────────────────────────────────────────────────────
# Constructores de la salida
# ─────────────────────────────────────────────────────────────────────────────
def _output(
    context: _Context,
    *,
    status: GateStatus,
    direction: Direction | None,
    tier: Tier,
    blockers: tuple[dict[str, str], ...] = (),
    notional_usd: Decimal | None = None,
    leverage_implied: Decimal | None = None,
) -> GateOutput:
    """Construye la salida y le pone su ``gate_sha256`` (el hash del payload sin esa clave)."""
    provisional = GateOutput(
        session=context.session,
        as_of=context.as_of,
        today=context.today,
        status=status,
        direction=direction,
        tier=tier,
        prob_up_calibrated=context.prob_up_calibrated,
        expected_move_pct=context.expected_move_pct,
        expected_move_basis=context.expected_move_basis,
        cost_pct=context.cost_pct,
        cost_total_pct=context.cost_total_pct,
        slippage_state=str(context.cost.slippage.state.value),
        ev_declared_pct=context.ev_declared_pct,
        ev_net_pct=context.ev_net_pct,
        stop_pct=context.stop_pct,
        target_pct=context.target_pct,
        stop_px=None,
        target_px=None,
        notional_usd=notional_usd,
        leverage_implied=leverage_implied,
        bracket_required=True,
        trades_today=context.trades_today,
        observation_sessions_remaining=context.observation_sessions_remaining,
        is_fomc_session=context.is_fomc_session,
        is_half_session=context.is_half_session,
        fomc_dates_count=context.fomc_dates_count,
        params=_parameters_payload(context.params),
        blockers=blockers,
        undecided=context.missing,
        rules=context.ledger.entries(),
        gate_sha256=GATE_HASH_PREFIX,
    )
    return provisional.model_copy(update={"gate_sha256": gate_sha256(provisional)})


def _nothing(context: _Context, *, tier: Tier, blockers: tuple[dict[str, str], ...]) -> GateOutput:
    """Una sesion evaluada sin operacion: es una **recomendacion**, no un "no se" (regla 14)."""
    return _output(
        context,
        status=GateStatus.RECOMMENDATION,
        direction=Direction.NOTHING,
        tier=tier,
        blockers=blockers,
    )


def _undecided_output(context: _Context) -> GateOutput:
    """Regla 14 / §11 bis: sin los parametros decididos no hay juicio, y eso no es ``NOTHING``."""
    context.ledger.mark(
        "14",
        _RULE_OK,
        "estado no_recommendation_undecided: se distingue de NOTHING y de los demas 'no se'",
    )
    return _output(
        context,
        status=GateStatus.NO_RECOMMENDATION_UNDECIDED,
        direction=None,
        tier=TIER_C,
    )


def _stale_output(context: _Context) -> GateOutput | None:
    """Regla 13: un ``as_of`` que no es de hoy no produce una recomendacion accionable."""
    if context.as_of.date() == context.today:
        context.ledger.mark(
            "13",
            _RULE_OK,
            f"as_of.date() = {context.as_of.date().isoformat()} es la fecha de hoy "
            f"({context.today.isoformat()})",
        )
        return None
    detail = (
        f"as_of = {context.as_of.isoformat()} no es de hoy ({context.today.isoformat()}): la "
        "regla 13 no emite recomendacion accionable con datos viejos; el modulo de la guardia "
        "completa es #40"
    )
    context.ledger.block("13", detail)
    return _output(
        context,
        status=GateStatus.NO_RECOMMENDATION_STALE_DATA,
        direction=None,
        tier=TIER_C,
        blockers=({"rule": "13", "code": CODE_AS_OF_NOT_TODAY, "detail": detail},),
    )


def _quality_output(context: _Context) -> GateOutput | None:
    """Regla 14: un snapshot que no pasa la validacion es un "no se", no un ``NOTHING``."""
    if context.snapshot_ok:
        context.ledger.mark("14", _RULE_OK, "snapshot_ok = True: el snapshot pasa la validacion")
        return None
    detail = (
        "snapshot_ok = False: fallo la validacion de datos del snapshot, asi que no hay juicio que "
        "hacer; los cinco estados no se fusionan con NOTHING y el registro es #40"
    )
    context.ledger.block("14", detail)
    return _output(
        context,
        status=GateStatus.NO_RECOMMENDATION_DATA_QUALITY,
        direction=None,
        tier=TIER_C,
        blockers=({"rule": "14", "code": CODE_SNAPSHOT_NOT_OK, "detail": detail},),
    )


# ─────────────────────────────────────────────────────────────────────────────
# El gate (A1): decide, bloquea o declara que no sabe
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_gate(
    *,
    session: date,
    as_of: datetime,
    today: date,
    calendar: MarketCalendar,
    prob_up_calibrated: float,
    expected_move_pct: Decimal,
    expected_move_basis: str,
    cost: CostBreakdown,
    capital_usd: Decimal,
    snapshot_ok: bool,
    stop_pct: Decimal,
    target_pct: Decimal | None,
    fomc_dates: Collection[date],
    params: GateParameters,
    trades_today: int = 0,
    daily_pnl_pct: Decimal | None = None,
    weekly_pnl_pct: Decimal | None = None,
    monthly_pnl_pct: Decimal | None = None,
    observation_sessions_remaining: int = 0,
) -> GateOutput:
    """Decide ``LONG``/``SHORT``/``NOTHING`` y publica stop, objetivo, nocional, tier y reglas.

    Todo es **keyword-only** y entra explicito: no hay reloj (``as_of`` y ``today`` llegan
    declarados), no hay red, no hay disco y el calendario llega ya construido. Mismos argumentos
    ⇒ misma salida **byte a byte** (A2).

    Orden de evaluacion (la precedencia declarada en el docstring del modulo):

    1. Frescura del dato (regla 13) y calidad del snapshot (regla 14) ⇒ estados "no se".
    2. Decisiones abiertas (#59/#60, ``GateParameters`` con ``None``) ⇒ ``undecided``.
    3. Bloqueos de sesion: reglas 1, 3, 4, 5, 15, 17 y 18.
    4. Reglas 16, 8, 9 y 10 y, si nada bloquea, el *sizing* (regla 2).

    Una entrada inadmisible (un porcentaje en ``float``, un coste con ``nights > 0``, un stop de
    0, un calendario que no es ``MarketCalendar``) lanza ``GateInputError``: nunca un resultado
    silencioso.
    """
    context = _context(
        session=session,
        as_of=as_of,
        today=today,
        calendar=calendar,
        prob_up_calibrated=prob_up_calibrated,
        expected_move_pct=expected_move_pct,
        expected_move_basis=expected_move_basis,
        cost=cost,
        capital_usd=capital_usd,
        snapshot_ok=snapshot_ok,
        stop_pct=stop_pct,
        target_pct=target_pct,
        fomc_dates=fomc_dates,
        params=params,
        trades_today=trades_today,
        daily_pnl_pct=daily_pnl_pct,
        weekly_pnl_pct=weekly_pnl_pct,
        monthly_pnl_pct=monthly_pnl_pct,
        observation_sessions_remaining=observation_sessions_remaining,
    )
    stale = _stale_output(context)
    if stale is not None:
        return stale
    quality = _quality_output(context)
    if quality is not None:
        return quality
    if context.missing:
        return _undecided_output(context)
    decided = _decided(context.params)
    blockers = _blockers(context, decided)
    tier = _tier(context, decided)
    if blockers:
        return _nothing(context, tier=tier, blockers=blockers)
    trading, notional, leverage = _trading_rules(context, decided, tier)
    if trading:
        return _nothing(context, tier=tier, blockers=trading)
    return _output(
        context,
        status=GateStatus.RECOMMENDATION,
        direction=context.direction,
        tier=tier,
        notional_usd=notional,
        leverage_implied=leverage,
    )
