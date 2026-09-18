"""Etiquetado tri-barrera (``target`` / ``stop`` / ``time``) — tarea #10.

La pregunta que responde este modulo: **dada la volatilidad prevista de la sesion y
un precio de entrada, que barrera se toca primero —la favorable, la adversa o el
cierre— y en las dos direcciones?**

Semantica de la etiqueta (``plan.md`` §4.2)
-------------------------------------------

- ``target``: se toca primero la barrera **favorable**.
- ``stop``: se toca primero la barrera **adversa**.
- ``time``: no se toca ninguna de las dos antes del cierre de sesion y la posicion
  sale al cierre.

La etiqueta **no** es «arriba / abajo»: es **favorable / adversa**. Con esa
definicion, ``LONG`` y ``SHORT`` son espejos explicitos: lo favorable para el largo
es ``entry * (1 + target_pct)`` (arriba) y para el corto ``entry * (1 - target_pct)``
(abajo); lo adverso, al contrario. El toque es **inclusivo** (``high >= barrera`` /
``low <= barrera``) y se compara con el precio de barrera **sin redondear**.

Cuando en la **misma barra** se tocan las dos barreras el orden es desconocido, asi
que la etiqueta es la **adversa** (``stop``) para las dos direcciones y el empate se
cuenta (``ties_in_bar``). Esa es la misma regla, conservadora, que se aplica al
respaldo diario (``fallback_ties``).

Barreras: proporcionales a la volatilidad prevista, nunca constantes
-------------------------------------------------------------------

``target_pct = stop_pct = k * sigma_t``, con ``sigma_t`` la raiz de la varianza
pronosticada por el **walk-forward de la tarea #7** (``analysis.volatility_forecast``)
para el candidato que su regla de seleccion elija. En este modulo no hay ninguna
sigma constante ni ninguna reimplementacion de HAR/GARCH: ``build_sample``,
``walk_forward``, ``select_candidate`` y ``scale_sigma_for_duration`` se consumen
tal cual. ``k`` sale de ``R_SIGMA_SCENARIOS``, que es un **escenario ilustrativo y no
una decision del propietario** (decision abierta 5 → #60); lo unico que se persiste
es ``k = 1.0``. Las barreras son simetricas: la asimetria queda fuera de alcance.

Media sesion (cierre a las 13:00 ET): el frame de #7 las excluye, asi que su sigma se
**arrastra** desde la ultima sesion completa etiquetada y se escala con
``scale_sigma_for_duration`` (raiz del tiempo). No hay ningun ``sqrt(3.5/6.5)``
literal aqui. Un dia sin *forecast* finito (el calentamiento de ``MIN_TRAIN``) queda
**sin etiquetar** con motivo ``no_forecast``: esta prohibido imputar sigma.

Orden de las barreras: intradia donde existe, respaldo diario declarado donde no
-------------------------------------------------------------------------------

Con barras de ``^GSPC`` de 5 minutos y cobertura ``>= MIN_INTRADAY_COVERAGE`` sobre
las barras esperadas de la sesion (``round(duracion_horas * 12)``: 78 en sesion
completa, 42 en media) se recorre la sesion en orden y decide la **primera** barra
que toca una barrera. En cualquier otro caso se usa el **respaldo diario**: solo la
favorable ⇒ ``target``, solo la adversa ⇒ ``stop``, ninguna ⇒ ``time`` y las dos a
la vez ⇒ ``stop``. El respaldo nunca elige la favorable cuando el orden es
desconocido, asi que el ``p_target`` de las sesiones con respaldo es una **cota
inferior**: el verdadero no puede ser menor. La cobertura real (hoy ~60 sesiones de
intradia frente a ~3.200 sesiones etiquetadas) se mide y se publica.

El precio de entrada es un parametro obligatorio
------------------------------------------------

``entry_px`` es un parametro **explicito y obligatorio** de la funcion pura: el
etiquetador nunca lee el ``open`` por su cuenta. Las fuentes permitidas estan en el
registro declarado ``ENTRY_PRICE_SOURCES``:

- ``session_open`` — el ``open`` de la subasta de apertura (09:30 ET): es la
  **decision del propietario del 2026-09-18** (registrada en la decision abierta 6 →
  **#61**) y **coincide** con lo que ya prescribe ``plan.md`` §4.1, que **no** se
  reescribe. Sigue siendo un **proxy declarado**: el precio usado es el del indice
  ``^GSPC``, **no** la cotizacion del ``SPX500:CFD`` (**#50**).
- ``t0_snapshot_0845_et`` — la **propuesta anterior del propietario, descartada el
  2026-09-18** (snapshot congelado de ``t0``, 08:45 ET). Sigue **declarada** a
  proposito: es la prueba de que **no hay *fallback* silencioso**. **No disponible**:
  el almacen no tiene ningun precio de las 08:45 ET (la primera barra intradia es de
  las 13:30 UTC, 09:30 ET) y a esa hora ni el indice ni el CFD cotizan, asi que el
  precio no es ejecutable. Con esta fuente el modulo falla con un error declarado
  (``state: unavailable`` + ``reason``) y **no escribe nada**; esta **prohibido**
  caer al ``open`` en silencio.

El anclaje se **declara** en el informe (``entry_price``) y no se parchea ``plan.md``.
Ahi va tambien ``auction_verification``, que comprueba **sobre el almacen** que el
``open`` diario que se usa es el *print* de la subasta de apertura: medida, no
supuesta (0,0 bp de diferencia en las 59 sesiones con intradia).

Puerta de Fase 0: se construye con ``fail``, y se declara
--------------------------------------------------------

La tarea #9 publico ``half_a: fail``, ``half_b: not_evaluable``, ``gate: fail`` y
``phase1_ready: false``. El **2026-09-18** el propietario decidio construir la Fase 1
igualmente (procedencia: *declaracion del usuario*). Este modulo lo registra en el
bloque ``phase0_context`` del informe (``gate: "fail"``, ``phase1_ready: false``,
``decided_on: "2026-09-18"``, ``source_issue: 9``) y **no** presenta la etiqueta como
una validacion de la estrategia: etiquetar no es validar.

Persistencia y determinismo
---------------------------

Las etiquetas van a ``derived.labels`` **solo** por la API del ``Store``
(``append`` para identidades nuevas, ``replace`` —recomputable— cuando el contenido
cambia: es la semantica de *supersede* de la capa ``derived``), con grano una fila
por ``(series_id, sesion)`` y las dos direcciones en columnas. ``as_of`` es el
**cierre de sesion en UTC** del ``MarketCalendar``, ``published_at = as_of`` (la
etiqueta solo se conoce al cerrar) y ``session`` es la fecha ET. Ninguna sesion cuyo
cierre sea posterior a ``now`` se escribe: se cuenta como
``unlabelled: session_not_closed``.

El etiquetado es una **funcion pura** de (sigma, precio de entrada, barreras, OHLC
diario y barras intradia): no lee el disco ni el ``open`` de la sesion siguiente
(``plan.md`` §12, regla 6: sin overnight). Dos ejecuciones con el mismo ``--now``
producen un JSON identico byte a byte y las mismas filas.

Limitaciones
------------

Las declaradas, con los numeros medidos, van en el artefacto (``limitations``): el
intradia solo cubre ~60 sesiones (``#50``, ``#57``), el precio de entrada es el
``open`` de la subasta de ``^GSPC`` —decision ya cerrada, con anclaje **proxy** del
CFD (``#50``)—, los numeros heredan el *look-ahead* de la muestra completa de #7
(``#63``) y la etiqueta no lleva spread ni
*slippage* por operacion (``#11``), la muestra usa el corte limpio de ``#52``, el
calentamiento de #7 queda sin etiqueta, las medias sesiones se etiquetan pero el gate
no las opera (``plan.md`` §12, regla 18) y ``k`` es ilustrativo (``#60``).

Unidades: **fraccion** en el calculo (sigma, ``target_pct``, ``stop_pct``, ``c`` y
los retornos) y **bp** (x 10^4) al informar, igual que ``analysis/drift.py`` y
``analysis/volatility_forecast.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final, cast

import duckdb
import numpy as np
import polars as pl
from loguru import logger

from cfdtrader.analysis.drift import clean_sample_cutoff, session_stale_open
from cfdtrader.analysis.volatility_forecast import (
    MIN_TRAIN,
    Selection,
    Verdict,
    build_sample,
    load_market,
    scale_sigma_for_duration,
    select_candidate,
    walk_forward,
)
from cfdtrader.analysis.volatility_forecast import (
    SERIES_ID as MARKET_SERIES_ID,
)
from cfdtrader.data.calendar import EASTERN, FULL_SESSION_HOURS, HALF_SESSION_HOURS, MarketCalendar
from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.store import ImmutableWriteError, Store, UnknownDatasetError, WriteOutcome

__all__ = [
    "AUCTION_VERIFICATION_TOLERANCE_BP",
    "BARS_PER_HOUR",
    "BP_PER_UNIT",
    "DEFAULT_ENTRY_PRICE_SOURCE",
    "ENTRY_PRICE_SOURCES",
    "EV_IDENTITY_TOLERANCE",
    "LABELS_DATASET",
    "LABEL_STOP",
    "LABEL_TARGET",
    "LABEL_TIME",
    "MIN_INTRADAY_COVERAGE",
    "PERSISTED_K_SIGMA",
    "ROUND_TRIP_SPREAD",
    "R_SIGMA_SCENARIOS",
    "BarrierLevels",
    "DailyBar",
    "EntryPriceSource",
    "EntryPriceUnavailableError",
    "IntradayBar",
    "LabelRow",
    "LabelsError",
    "LabelsRun",
    "SampleSession",
    "SelectionInconclusiveError",
    "SessionLabel",
    "auction_verification",
    "barrier_levels",
    "label_and_write",
    "label_history",
    "label_rows",
    "label_session",
    "main",
    "order_source_for",
    "phase0_context",
    "render_markdown",
    "report_payload",
    "resolve_candidate",
    "summarise_rows",
    "write_outcome_reason",
    "write_outputs",
]

#: Serie cuyo intradia se usa para ordenar las barreras. El CFD no tiene fuente (#50).
SERIES_ID: Final[str] = MARKET_SERIES_ID

#: Intervalo exigido: solo ``^GSPC`` de 5 minutos (A13). ``ES=F`` **no** se usa.
INTERVAL: Final[str] = "5m"

#: Serie intradia que se cuenta pero no se usa (A13, A17).
INTRADAY_UNUSED_SERIES: Final[str] = "ES=F"

#: Barras de 5 minutos esperadas por hora de sesion.
BARS_PER_HOUR: Final[int] = 12

#: Cobertura minima de barras intradia para ordenar con intradia en vez del respaldo.
MIN_INTRADAY_COVERAGE: Final[float] = 0.95

#: Escenarios de ``k`` en unidades de sigma. **Ilustrativo, no decision del
#: propietario** (decision abierta 5 → #60); ``plan.md`` §4.4 cita ``R = 0.5 / 1.0 /
#: 1.5 %`` solo como ejemplo.
R_SIGMA_SCENARIOS: Final[tuple[float, ...]] = (0.5, 1.0, 1.5)

#: Escenario que se persiste en ``derived.labels`` (A5).
PERSISTED_K_SIGMA: Final[float] = 1.0

#: Etiquetas posibles.
LABEL_TARGET: Final[str] = "target"
LABEL_STOP: Final[str] = "stop"
LABEL_TIME: Final[str] = "time"
LABELS: Final[tuple[str, ...]] = (LABEL_TARGET, LABEL_STOP, LABEL_TIME)

#: Direcciones que se etiquetan con la misma pasada de barreras.
DIRECTIONS: Final[tuple[str, ...]] = ("long", "short")

#: Fuentes de orden de barreras.
INTRADAY_ORDER_SOURCE: Final[str] = "intraday_5m"
FALLBACK_ORDER_SOURCE: Final[str] = "daily_ohlc_fallback"

#: Motivos de ``unlabelled``, en el orden de precedencia con que se decide.
REASON_CALENDAR_NO_SESSION: Final[str] = "calendar_no_session"
REASON_STALE_OPEN: Final[str] = "stale_open"
REASON_NULL_OHLC: Final[str] = "null_ohlc"
REASON_NO_FORECAST: Final[str] = "no_forecast"
REASON_SESSION_NOT_CLOSED: Final[str] = "session_not_closed"
UNLABELLED_REASONS: Final[tuple[str, ...]] = (
    REASON_CALENDAR_NO_SESSION,
    REASON_STALE_OPEN,
    REASON_NULL_OHLC,
    REASON_NO_FORECAST,
    REASON_SESSION_NOT_CLOSED,
)

#: De donde sale la sigma de cada fila.
CARRIER_WALK_FORWARD: Final[str] = "walk_forward"
CARRIER_PREVIOUS_LABELLED: Final[str] = "previous_labelled_session"

#: Diferencial declarado de ida y vuelta del CFD, intradia puro sin noche (A24).
ROUND_TRIP_SPREAD: Final[float] = 0.000042

#: Procedencia del diferencial: ``plan.md`` §3.3 y §4.4, cuantificado en #8.
ROUND_TRIP_SPREAD_PROVENANCE: Final[str] = (
    "plan.md §3.3 y §4.4 (0,42 $ sobre 10.000 $ de nocional = 0,0042 %) y #8 "
    "(auditoria de los costes declarados); intradia puro, sin financiacion"
)

#: Totales declarados de #8 para una tenencia de un dia completo. **Sensibilidad**:
#: nunca se fusionan con el diferencial intradia ni entre si.
COST_SENSITIVITY_SCENARIOS: Final[dict[str, float]] = {
    "short_one_day": 0.000024,
    "long_one_day": 0.000224,
}

#: Tolerancia de la identidad ``p_win * E[G] - (1 - p_win) * E[P] == media(ret)`` (A23).
EV_IDENTITY_TOLERANCE: Final[float] = 1e-12

#: bp: factor de presentacion (fraccion × 10^4).
BP_PER_UNIT: Final[float] = 10_000.0

#: Dataset y capa de las etiquetas (``tech_stack.md`` §12.4).
LABELS_LAYER: Final[str] = "derived"
LABELS_DATASET: Final[str] = "labels"

#: ``source`` declarado de las etiquetas: es un dataset derivado, no una fuente.
LABELS_SOURCE: Final[str] = "cfdtrader.models.labels"

#: Prefijo del informe, con la misma convencion que los otros artefactos.
REPORT_PREFIX: Final[str] = "triple_barrier"

#: Fuentes de precio de entrada permitidas.
ENTRY_PRICE_SESSION_OPEN: Final[str] = "session_open"
ENTRY_PRICE_T0_SNAPSHOT: Final[str] = "t0_snapshot_0845_et"
DEFAULT_ENTRY_PRICE_SOURCE: Final[str] = ENTRY_PRICE_SESSION_OPEN

#: Decision vigente del propietario sobre el precio de entrada (2026-09-18, A1).
#: Cierra la decision abierta 6, que se **registra** en `_docs/**` por la issue #61.
ENTRY_PRICE_DECISION_DATE: Final[str] = "2026-09-18"
ENTRY_PRICE_DECISION_PROVENANCE: Final[str] = "decision del propietario"
ENTRY_PRICE_OWNER_DECISION: Final[str] = (
    "el `open` de la subasta de apertura (09:30 ET), que es la fuente `session_open`"
)
#: Propuesta **anterior** del propietario, descartada el 2026-09-18 (A1, A4).
ENTRY_PRICE_PREVIOUS_DECISION: Final[str] = "t0 a las 08:45 ET (snapshot congelado)"
#: El precio usado es el del **indice**, no el del CFD: es un *proxy* declarado (A3).
ENTRY_PRICE_PROXY_OF: Final[str] = "SPX500:CFD"
ENTRY_PRICE_PROXY_ISSUE: Final[int] = 50

#: Tolerancia declarada de la verificacion de la subasta: el `open` diario y la primera
#: barra de las 09:30 ET deben coincidir dentro de **1 bp** (A6, procedencia #64). No
#: relaja ni sustituye el precio declarado: es solo el listón de la comprobacion.
AUCTION_VERIFICATION_TOLERANCE_BP: Final[float] = 1.0


class LabelsError(Exception):
    """Base de los errores del etiquetado tri-barrera."""


class SelectionInconclusiveError(LabelsError):
    """La regla de #7 no elige candidato: el modulo **no elige por su cuenta**."""


class EntryPriceUnavailableError(LabelsError):
    """El precio de entrada pedido no esta disponible en el almacen (A9).

    Lleva ``state`` y ``reason`` para que el fallo sea declarado y no una caida
    silenciosa al ``open`` de sesion.
    """

    def __init__(self, *, source: str, state: str, reason: str) -> None:
        super().__init__(f"el precio de entrada '{source}' no esta disponible ({state}): {reason}")
        self.source = source
        self.state = state
        self.reason = reason


@dataclass(frozen=True, slots=True)
class EntryPriceSource:
    """Fuente de precio de entrada declarada en el registro del modulo (A9)."""

    name: str
    state: str
    provenance: str
    reason: str | None
    is_proxy: bool
    tradable: bool
    diverges_from_owner_decision: bool
    spot_et: str


#: Registro de fuentes permitidas. Ampliarlo es una decision de #61, no de aqui.
ENTRY_PRICE_SOURCES: Final[dict[str, EntryPriceSource]] = {
    ENTRY_PRICE_SESSION_OPEN: EntryPriceSource(
        name=ENTRY_PRICE_SESSION_OPEN,
        state="available",
        provenance=(
            "decision del propietario, 2026-09-18: el `open` de la subasta de apertura "
            "(09:30 ET), que coincide con `plan.md` §4.1 y §4.2"
        ),
        reason=None,
        is_proxy=True,
        tradable=True,
        diverges_from_owner_decision=False,
        spot_et="09:30",
    ),
    ENTRY_PRICE_T0_SNAPSHOT: EntryPriceSource(
        name=ENTRY_PRICE_T0_SNAPSHOT,
        state="unavailable",
        provenance=(
            "propuesta anterior del propietario, descartada el 2026-09-18 (decision abierta 6, "
            "registrada por #61)"
        ),
        reason=(
            "el almacen no tiene ningun precio de las 08:45 ET: la primera barra intradia de "
            "^GSPC 5m es de las 13:30 UTC (09:30 ET) y a esa hora ni el indice ni el CFD "
            "cotizan, asi que el precio de t0 no es un precio ejecutable"
        ),
        is_proxy=False,
        tradable=False,
        diverges_from_owner_decision=True,
        spot_et="08:45",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Contexto declarado (A1) y divergencia del precio de entrada (A10)
# ─────────────────────────────────────────────────────────────────────────────
def phase0_context() -> dict[str, object]:
    """Decision del propietario del 2026-09-18: la Fase 1 se construye con la puerta en `fail`.

    Etiquetar **no** es validar la estrategia: la etiqueta dice que barrera se toco
    primero, no que la estrategia funcione. Este bloque va al informe y al docstring.
    """
    return {
        "gate": "fail",
        "phase1_ready": False,
        "half_a": "fail",
        "half_b": "not_evaluable",
        "owner_decision": (
            "construir la Fase 1 (arnes de backtest) con la puerta de Fase 0 en `fail`, "
            "aceptando que el veredicto pueda ser no pasar a Fase 1"
        ),
        "decided_on": "2026-09-18",
        "provenance": "declaracion del usuario",
        "source_issue": 9,
        "source_artifact": "data/derived/reports/phase0_report_2026-09-18.json",
        "consequence": (
            "la etiqueta tri-barrera es un artefacto de investigacion, no una validacion de la "
            "estrategia; el artefacto no puede presentar la puerta como superada"
        ),
    }


def _entry_price_block(
    *, source_used: str, evidence: Mapping[str, object], auction: Mapping[str, object]
) -> dict[str, object]:
    """Bloque ``entry_price``: decision vigente, *proxy* declarado y verificacion (A1-A7)."""
    spec = ENTRY_PRICE_SOURCES[source_used]
    return {
        "owner_decision": ENTRY_PRICE_OWNER_DECISION,
        "decided_on": ENTRY_PRICE_DECISION_DATE,
        "provenance": ENTRY_PRICE_DECISION_PROVENANCE,
        "previous_owner_decision": ENTRY_PRICE_PREVIOUS_DECISION,
        "previous_owner_decision_discarded_on": ENTRY_PRICE_DECISION_DATE,
        "previous_owner_decision_note": (
            "se conserva como rastro: fue la propuesta del propietario hasta el 2026-09-18 y "
            "sigue **declarada** en el registro de fuentes para que no haya *fallback* silencioso"
        ),
        "source_used": source_used,
        "default_source": DEFAULT_ENTRY_PRICE_SOURCE,
        "source_state": spec.state,
        "source_is_proxy": spec.is_proxy,
        "proxy_of": ENTRY_PRICE_PROXY_OF,
        "proxy_note": (
            f"el precio usado es el `open` diario de `{MARKET_SERIES_ID}` (el indice), un "
            f"**proxy declarado** del `{ENTRY_PRICE_PROXY_OF}`: no es la cotizacion del CFD"
        ),
        "source_tradable": spec.tradable,
        "diverges_from_owner_decision": spec.diverges_from_owner_decision,
        "diverges_from_owner_decision_reason": (
            "el anclaje usado (`session_open`) **es** la decision del propietario del "
            f"{ENTRY_PRICE_DECISION_DATE} y **coincide** con `plan.md` §4.1: no diverge de "
            "ninguna de las dos"
        ),
        "contradicts_plan_md_4_1": False,
        "contradicts_plan_md_4_1_reason": (
            "la decision del propietario del 2026-09-18 **hace coincidir** el anclaje con "
            "`plan.md` §4.1 (`open` de la subasta de apertura, 09:30 ET): el conflicto se "
            "resuelve **por decision**, no parcheando el documento"
        ),
        "plan_md_4_1_proposes": "el `open` de la subasta de apertura (09:30 ET), `plan.md` §4.1",
        "not_tradable": False,
        "not_tradable_reason": (
            "el instante de la subasta de apertura (09:30 ET) **si** es un instante en el que "
            "el CFD cotiza: es la apertura de la sesion regular"
        ),
        "follow_up_issue": ENTRY_PRICE_PROXY_ISSUE,
        "decision_issue": 61,
        "registry": {name: asdict(entry) for name, entry in sorted(ENTRY_PRICE_SOURCES.items())},
        "auction_verification": dict(auction),
        "evidence": dict(evidence),
        "declared": (
            "la coincidencia con `plan.md` §4.1 se **declara** y `plan.md` **no** se reescribe "
            "en esta tarea; la anotacion en los documentos es #61 y #65"
        ),
    }


def auction_verification(
    sessions: Sequence[SampleSession], *, tolerance_bp: float = AUCTION_VERIFICATION_TOLERANCE_BP
) -> dict[str, object]:
    """Comprueba que el precio usado es el *print* de la subasta de apertura (A6, A7).

    Compara, sesion a sesion, el `open` diario que usa el etiquetador
    (`raw.market_daily.open`) con el `open` de la **primera barra de 09:30 ET** del mismo
    dia, alli donde hay intradia con cobertura suficiente. Es una comprobacion de **solo
    lectura**: no cambia ninguna etiqueta, no excluye ninguna sesion y **no** introduce
    ninguna fuente de precio nueva.
    """
    compared = 0
    identical = 0
    max_abs_diff_bp: float | None = None
    mismatches: list[dict[str, object]] = []
    for item in sessions:
        if item.unlabelled_reason is not None or item.entry_px is None or item.open_utc is None:
            continue
        source, _coverage, _incomplete = order_source_for(
            observed_bars=len(item.intraday), expected_bars=item.expected_bars
        )
        if source != INTRADAY_ORDER_SOURCE:
            continue
        opening = next((bar for bar in item.intraday if bar.as_of == item.open_utc), None)
        if opening is None:
            continue
        difference_bp = abs(item.entry_px - opening.open) / opening.open * BP_PER_UNIT
        compared += 1
        max_abs_diff_bp = (
            difference_bp if max_abs_diff_bp is None else max(max_abs_diff_bp, difference_bp)
        )
        if difference_bp <= tolerance_bp:
            identical += 1
        else:
            mismatches.append(
                {
                    "session": item.session.isoformat(),
                    "daily_open": item.entry_px,
                    "auction_open": opening.open,
                    "diff_bp": difference_bp,
                }
            )
    if compared == 0:
        status = "not_evaluable"
        reason = (
            "ninguna sesion tiene intradia con cobertura suficiente para comparar el `open` "
            "diario con la primera barra de las 09:30 ET"
        )
    elif mismatches:
        status = "mismatch"
        reason = (
            f"{len(mismatches)} de {compared} sesiones comparadas superan la tolerancia "
            f"declarada de {AUCTION_VERIFICATION_TOLERANCE_BP} bp; ninguna etiqueta cambia por "
            "ello: la verificacion es de solo lectura"
        )
    else:
        status = "ok"
        reason = (
            f"las {compared} sesiones comparadas coinciden dentro de la tolerancia declarada "
            f"({AUCTION_VERIFICATION_TOLERANCE_BP} bp): el `open` diario **es** el *print* de la "
            "subasta de apertura"
        )
    return {
        "status": status,
        "reason": reason,
        "sessions_compared": compared,
        "identical": identical,
        "max_abs_diff_bp": max_abs_diff_bp,
        "mismatches": mismatches,
        "read_only": True,
        "read_only_note": (
            "la comprobacion no decide el precio ni descarta sesiones: el precio sigue siendo "
            "el declarado y con un desajuste sintetico las filas no cambian"
        ),
        "price_source": {
            "price_used": "raw.market_daily.open",
            "series_id": SERIES_ID,
            "reference": (
                "la primera barra intradia de las 09:30 ET de `raw.market_intraday` "
                f"(`{SERIES_ID}` a {INTERVAL})"
            ),
            "rule": (
                "solo las sesiones con cobertura >= "
                "`MIN_INTRADAY_COVERAGE` sobre las barras esperadas de la sesion"
            ),
        },
        "tolerance": {
            "name": "AUCTION_VERIFICATION_TOLERANCE_BP",
            "value_bp": AUCTION_VERIFICATION_TOLERANCE_BP,
            "unit": "bp",
            "provenance": "A6 del enunciado de #64 (tolerancia declarada, <= 1 bp)",
            "note": "por encima de esta diferencia la sesion se lista en `mismatches`",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Funcion pura: barreras y etiqueta
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class BarrierLevels:
    """Barreras de una sesion, en fraccion del precio de entrada (A4)."""

    entry_px: float
    target_pct: float
    stop_pct: float
    upper: float
    lower: float


@dataclass(frozen=True, slots=True)
class DailyBar:
    """OHLC diario reducido a lo que decide el respaldo."""

    high: float
    low: float
    close: float


@dataclass(frozen=True, slots=True)
class IntradayBar:
    """Barra intradia de 5 minutos dentro de la sesion."""

    as_of: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True, slots=True)
class SessionLabel:
    """Resultado del etiquetado de una sesion, en las dos direcciones."""

    label_long: str
    label_short: str
    target_pct: float
    stop_pct: float
    exit_long: float
    exit_short: float
    ret_long: float
    ret_short: float
    order_source: str
    bars_observed: int
    coverage: float
    intraday_incomplete: bool
    ties_in_bar: int
    fallback_ties: int
    decided_bar: int


def barrier_levels(*, entry_px: float, sigma: float, k_sigma: float) -> BarrierLevels:
    """Barreras simetricas ``entry * (1 +/- k * sigma)`` (A4).

    ``target_pct = stop_pct = k_sigma * sigma``: duplicar ``sigma`` duplica
    exactamente las dos barreras. No hay ningun multiplicador constante de ATR ni
    ningun ``target_pct`` fijo.
    """
    if entry_px <= 0.0:
        raise ValueError("el precio de entrada debe ser positivo")
    if sigma <= 0.0:
        raise ValueError("la sigma de la sesion debe ser positiva")
    if k_sigma <= 0.0:
        raise ValueError("k debe ser positivo")
    size = k_sigma * sigma
    return BarrierLevels(
        entry_px=entry_px,
        target_pct=size,
        stop_pct=size,
        upper=entry_px * (1.0 + size),
        lower=entry_px * (1.0 - size),
    )


def order_source_for(*, observed_bars: int, expected_bars: int) -> tuple[str, float, bool]:
    """Decide la fuente del orden: ``(fuente, cobertura, intradia_incompleto)`` (A13).

    ``intraday_5m`` solo si hay barras **y** la cobertura llega a
    ``MIN_INTRADAY_COVERAGE`` sobre ``expected_bars``; en cualquier otro caso,
    ``daily_ohlc_fallback``.
    """
    if expected_bars <= 0:
        return FALLBACK_ORDER_SOURCE, 0.0, True
    coverage = observed_bars / expected_bars
    if observed_bars <= 0 or coverage < MIN_INTRADAY_COVERAGE:
        return FALLBACK_ORDER_SOURCE, coverage, True
    return INTRADAY_ORDER_SOURCE, coverage, False


def _first_touch(
    bars: Sequence[tuple[float, float]], levels: BarrierLevels
) -> tuple[str, str, float | None, float | None, bool, int]:
    """Primera barra que toca una barrera: etiquetas, salidas, empate e indice (1-based).

    El toque es inclusivo y se compara con el precio de barrera sin redondear. Si la
    barra toca las dos barreras, el orden es desconocido: la etiqueta es la adversa
    para las dos direcciones (``stop``) y se devuelve ``both=True``.
    """
    for index, (high, low) in enumerate(bars, start=1):
        hit_upper = high >= levels.upper
        hit_lower = low <= levels.lower
        if hit_upper and hit_lower:
            return LABEL_STOP, LABEL_STOP, levels.lower, levels.upper, True, index
        if hit_upper:
            return LABEL_TARGET, LABEL_STOP, levels.upper, levels.upper, False, index
        if hit_lower:
            return LABEL_STOP, LABEL_TARGET, levels.lower, levels.lower, False, index
    return LABEL_TIME, LABEL_TIME, None, None, False, 0


def label_session(
    *,
    entry_px: float,
    sigma: float,
    k_sigma: float,
    daily: DailyBar,
    intraday: Sequence[IntradayBar] = (),
    expected_bars: int = 0,
) -> SessionLabel:
    """Etiqueta una sesion en las dos direcciones (funcion pura).

    ``entry_px`` es **obligatorio**: el etiquetador nunca lee el ``open`` por su
    cuenta (A9). Si la sesion tiene barras intradia con cobertura suficiente se
    ordena con ellas; si no, con el OHLC diario, de forma conservadora.

    Parameters
    ----------
    entry_px:
        Precio de entrada de la sesion, en unidades de precio.
    sigma:
        Volatilidad prevista de la sesion, en fraccion.
    k_sigma:
        Multiplo de sigma que fija las dos barreras.
    daily:
        OHLC de la sesion (respaldo y salida de las etiquetas ``time``).
    intraday:
        Barras de 5 minutos **dentro** de la sesion, en orden temporal.
    expected_bars:
        Barras esperadas de una sesion de esa duracion (78 completas, 42 medias).
    """
    levels = barrier_levels(entry_px=entry_px, sigma=sigma, k_sigma=k_sigma)
    source, coverage, incomplete = order_source_for(
        observed_bars=len(intraday), expected_bars=expected_bars
    )
    touches: list[tuple[float, float]] = (
        [(bar.high, bar.low) for bar in intraday]
        if source == INTRADAY_ORDER_SOURCE
        else [(daily.high, daily.low)]
    )
    label_long, label_short, exit_long, exit_short, both, decided_bar = _first_touch(
        touches, levels
    )
    if exit_long is None or exit_short is None:
        # Etiqueta `time`: la posicion sale al cierre de la sesion.
        exit_long = daily.close
        exit_short = daily.close
    is_intraday = source == INTRADAY_ORDER_SOURCE
    return SessionLabel(
        label_long=label_long,
        label_short=label_short,
        target_pct=levels.target_pct,
        stop_pct=levels.stop_pct,
        exit_long=exit_long,
        exit_short=exit_short,
        ret_long=(exit_long - entry_px) / entry_px,
        ret_short=(entry_px - exit_short) / entry_px,
        order_source=source,
        bars_observed=len(intraday),
        coverage=coverage,
        intraday_incomplete=incomplete,
        ties_in_bar=1 if both and is_intraday else 0,
        fallback_ties=1 if both and not is_intraday else 0,
        decided_bar=decided_bar,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Muestra etiquetable
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SampleSession:
    """Todo lo que necesita la funcion pura para etiquetar una sesion (o descartarla)."""

    session: date
    close_utc: datetime
    is_half_day: bool
    duration_hours: float
    entry_px: float | None
    daily: DailyBar | None
    sigma: float | None
    sigma_carrier: str | None
    intraday: tuple[IntradayBar, ...]
    expected_bars: int
    unlabelled_reason: str | None
    #: Apertura de la sesion en UTC: identifica la barra de las 09:30 ET (A6).
    open_utc: datetime | None = None


@dataclass(frozen=True, slots=True)
class LabelRow:
    """Fila etiquetada: grano ``(series_id, sesion)`` con las dos direcciones (A26)."""

    session: date
    close_utc: datetime
    is_half_day: bool
    sigma: float
    sigma_carrier: str
    k_sigma: float
    target_pct: float
    stop_pct: float
    entry_px: float
    order_source: str
    bars_observed: int
    expected_bars: int
    coverage: float
    intraday_incomplete: bool
    ties_in_bar: int
    fallback_ties: int
    decided_bar: int
    label_long: str
    label_short: str
    exit_long: float
    exit_short: float
    ret_long: float
    ret_short: float


def label_rows(sessions: Sequence[SampleSession], *, k_sigma: float) -> list[LabelRow]:
    """Etiqueta todas las sesiones etiquetables con ese ``k`` (funcion pura)."""
    rows: list[LabelRow] = []
    for item in sessions:
        if item.unlabelled_reason is not None:
            continue
        if item.daily is None or item.entry_px is None or item.sigma is None:
            continue
        label = label_session(
            entry_px=item.entry_px,
            sigma=item.sigma,
            k_sigma=k_sigma,
            daily=item.daily,
            intraday=item.intraday,
            expected_bars=item.expected_bars,
        )
        rows.append(
            LabelRow(
                session=item.session,
                close_utc=item.close_utc,
                is_half_day=item.is_half_day,
                sigma=item.sigma,
                sigma_carrier=item.sigma_carrier or CARRIER_WALK_FORWARD,
                k_sigma=k_sigma,
                target_pct=label.target_pct,
                stop_pct=label.stop_pct,
                entry_px=item.entry_px,
                order_source=label.order_source,
                bars_observed=label.bars_observed,
                expected_bars=item.expected_bars,
                coverage=label.coverage,
                intraday_incomplete=label.intraday_incomplete,
                ties_in_bar=label.ties_in_bar,
                fallback_ties=label.fallback_ties,
                decided_bar=label.decided_bar,
                label_long=label.label_long,
                label_short=label.label_short,
                exit_long=label.exit_long,
                exit_short=label.exit_short,
                ret_long=label.ret_long,
                ret_short=label.ret_short,
            )
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Intradia del almacen
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Intraday:
    """Barras intradia por sesion ET y cobertura medida (A17)."""

    by_session: dict[date, tuple[IntradayBar, ...]]
    coverage: dict[str, object]


def _intraday(store: Store) -> Intraday:
    """Lee ``raw.market_intraday``, agrupa por sesion ET y mide la cobertura.

    Que no haya intradia **no** aborta: el respaldo diario es el camino normal para
    casi todo el historico. La cobertura se declara con el numero de barras, las
    sesiones, la primera y la ultima barra y cuantas sesiones hay a cada hora.
    """
    query = (
        "SELECT series_id, interval, as_of, open, high, low, close "
        "FROM raw.market_intraday ORDER BY as_of"
    )
    try:
        frame = store.sql(query)
    except (UnknownDatasetError, duckdb.Error) as error:
        logger.warning("sin intradia en el almacen ({}): todo se ordena con respaldo diario", error)
        return Intraday(
            by_session={},
            coverage={
                "available": False,
                "reason": f"no hay dataset `raw.market_intraday` consultable: {error}",
            },
        )

    bars: dict[str, dict[date, list[IntradayBar]]] = {}
    null_bars = 0
    for row in frame.filter(pl.col("interval") == INTERVAL).iter_rows(named=True):
        values = [row["open"], row["high"], row["low"], row["close"]]
        if any(value is None for value in values):
            null_bars += 1
            continue
        moment = cast("datetime", row["as_of"])
        series = str(row["series_id"])
        bar = IntradayBar(
            as_of=moment,
            open=float(cast("float", row["open"])),
            high=float(cast("float", row["high"])),
            low=float(cast("float", row["low"])),
            close=float(cast("float", row["close"])),
        )
        bars.setdefault(series, {}).setdefault(moment.astimezone(EASTERN).date(), []).append(bar)

    used = bars.get(SERIES_ID, {})
    by_session = {
        day: tuple(sorted(group, key=lambda bar: bar.as_of)) for day, group in used.items()
    }
    all_bars = [bar for group in by_session.values() for bar in group]
    first = min(all_bars, key=lambda bar: bar.as_of) if all_bars else None
    last = max(all_bars, key=lambda bar: bar.as_of) if all_bars else None
    unused_sessions = len(bars.get(INTRADAY_UNUSED_SERIES, {}))
    coverage: dict[str, object] = {
        "available": bool(by_session),
        "series_id": SERIES_ID,
        "interval": INTERVAL,
        "bars": len(all_bars),
        "sessions": len(by_session),
        "null_bars": null_bars,
        "first_bar_utc": None if first is None else first.as_of.isoformat(),
        "last_bar_utc": None if last is None else last.as_of.isoformat(),
        "first_bar_et": None if first is None else first.as_of.astimezone(EASTERN).isoformat(),
        "expected_bars_full_session": round(FULL_SESSION_HOURS * BARS_PER_HOUR),
        "expected_bars_half_session": round(HALF_SESSION_HOURS * BARS_PER_HOUR),
        "min_intraday_coverage": MIN_INTRADAY_COVERAGE,
        "ordered_with_intraday": 0,
        "ordered_with_fallback": 0,
        "fallback_share": None,
        "unused_series": {
            INTRADAY_UNUSED_SERIES: {
                "sessions": unused_sessions,
                "reason": "A13: el orden de las barreras se decide solo con `^GSPC` 5m",
            }
        },
    }
    return Intraday(by_session=by_session, coverage=coverage)


def _stored_daily_sessions(store: Store, series_id: str) -> int:
    """Sesiones diarias almacenadas de esa serie (A17), tal cual estan en `raw`."""
    query = (
        "SELECT count(*) AS sessions FROM raw.market_daily "  # noqa: S608
        f"WHERE series_id = {_literal(series_id)}"
    )
    try:
        frame = store.sql(query)
    except (UnknownDatasetError, duckdb.Error):
        return 0
    if frame.height == 0:
        return 0
    return int(str(frame.get_column("sessions").to_list()[0]))


def _bars_at_0845_et(store: Store) -> int:
    """Barras del indice cuya hora ET es 08:45: la evidencia de A10 (hoy, ninguna)."""
    query = (
        "SELECT count(*) AS bars FROM raw.market_intraday "  # noqa: S608
        f"WHERE series_id = {_literal(SERIES_ID)} "
        "AND (as_of AT TIME ZONE 'America/New_York')::time = TIME '08:45:00'"
    )
    try:
        frame = store.sql(query)
    except (UnknownDatasetError, duckdb.Error):
        return 0
    if frame.height == 0:
        return 0
    return int(str(frame.get_column("bars").to_list()[0]))


def _literal(value: str) -> str:
    """Literal SQL seguro: las series vienen del registro declarado del proyecto."""
    return "'" + value.replace("'", "''") + "'"


# ─────────────────────────────────────────────────────────────────────────────
# Muestra: sesiones etiquetables y motivos de descarte
# ─────────────────────────────────────────────────────────────────────────────
def _sample_sessions(
    *,
    store: Store,
    calendar: MarketCalendar,
    now: datetime,
    sigma_by_session: Mapping[date, float],
    intraday: Intraday,
    series_id: str,
) -> tuple[list[SampleSession], dict[str, object]]:
    """Sesiones desde el corte limpio de #52, con su sigma y su motivo de descarte."""
    market = load_market(store, series_id=series_id)
    base = session_stale_open(market).filter(pl.col("prev_close").is_not_null())
    cutoff = clean_sample_cutoff(base)
    region = base if cutoff is None else base.filter(pl.col("session") >= cutoff)

    days = [cast("date", value) for value in region.get_column("session").to_list()]
    open_values = region.get_column("open").to_list()
    high_values = region.get_column("high").to_list()
    low_values = region.get_column("low").to_list()
    close_values = region.get_column("close").to_list()
    stale_values = region.get_column("open_stale").to_list()

    sessions: list[SampleSession] = []
    reasons: dict[str, list[date]] = {reason: [] for reason in UNLABELLED_REASONS}
    last_full_sigma: float | None = None
    for index, day in enumerate(days):
        info = calendar.session(day)
        daily = None
        has_ohlc = (
            high_values[index] is not None
            and low_values[index] is not None
            and close_values[index] is not None
        )
        if has_ohlc:
            daily = DailyBar(
                high=float(cast("float", high_values[index])),
                low=float(cast("float", low_values[index])),
                close=float(cast("float", close_values[index])),
            )
        entry_px = None if open_values[index] is None else float(cast("float", open_values[index]))
        expected_bars = round(info.duration_hours * BARS_PER_HOUR)
        sigma: float | None = None
        carrier: str | None = None
        reason: str | None = None
        if not info.is_session or info.close_utc is None:
            reason = REASON_CALENDAR_NO_SESSION
        elif bool(stale_values[index]):
            reason = REASON_STALE_OPEN
        elif daily is None:
            reason = REASON_NULL_OHLC
        elif info.is_half_day:
            if last_full_sigma is None:
                reason = REASON_NO_FORECAST
            else:
                sigma = scale_sigma_for_duration(
                    last_full_sigma, duration_hours=info.duration_hours
                )
                carrier = CARRIER_PREVIOUS_LABELLED
        else:
            sigma = sigma_by_session.get(day)
            if sigma is None:
                reason = REASON_NO_FORECAST
            else:
                carrier = CARRIER_WALK_FORWARD
        if reason is None and info.close_utc is not None and info.close_utc > now:
            reason = REASON_SESSION_NOT_CLOSED
        if reason is not None:
            reasons[reason].append(day)
        elif carrier == CARRIER_WALK_FORWARD and sigma is not None:
            # ``sigma_ref`` de una media sesion: la ultima sigma de sesion COMPLETA
            # etiquetada anterior (no la de una media sesion ya escalada).
            last_full_sigma = sigma
        sessions.append(
            SampleSession(
                session=day,
                close_utc=info.close_utc or datetime.combine(day, datetime.min.time(), tzinfo=UTC),
                is_half_day=info.is_half_day,
                duration_hours=info.duration_hours,
                entry_px=None if reason is not None else entry_px,
                daily=None if reason is not None else daily,
                sigma=None if reason is not None else sigma,
                sigma_carrier=None if reason is not None else carrier,
                intraday=(
                    ()
                    if info.open_utc is None or info.close_utc is None
                    else tuple(
                        bar
                        for bar in intraday.by_session.get(day, ())
                        # A13: solo las barras **dentro** de [open_utc, close_utc].
                        if info.open_utc <= bar.as_of <= info.close_utc
                    )
                ),
                expected_bars=expected_bars,
                unlabelled_reason=reason,
                open_utc=info.open_utc,
            )
        )

    detail = {
        reason: {
            "count": len(found),
            "first": None if not found else min(found).isoformat(),
            "last": None if not found else max(found).isoformat(),
        }
        for reason, found in reasons.items()
    }
    summary: dict[str, object] = {
        "region_sessions": region.height,
        "reasons": detail,
        "precedence": list(UNLABELLED_REASONS),
    }
    return sessions, summary


# ─────────────────────────────────────────────────────────────────────────────
# Ejecucion completa (sin escribir)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class LabelsRun:
    """Todo lo que produce una ejecucion del etiquetado, antes de escribir."""

    as_of: datetime
    series_id: str
    interval: str
    interval_hours: float
    k_sigma: float
    entry_price_source: str
    rows: tuple[LabelRow, ...]
    scenarios: dict[str, tuple[LabelRow, ...]]
    inputs: dict[str, object]
    sample: dict[str, object]
    summary: dict[str, object]
    cost: dict[str, object]
    forecast_sha256: str
    forecast_candidate: str
    selection_verdict: str
    limitations: tuple[str, ...]
    notes: tuple[str, ...]
    #: Resultado declarado de la regeneracion de ``derived.labels`` (A8):
    #: ``created`` / ``unchanged`` / ``superseded``. ``None`` mientras el run no se ha
    #: escrito, porque el etiquetado en si es una **funcion pura**.
    write_outcome: str | None = None
    #: ``version`` vigente del almacen tras la escritura: el almacen es su dueno.
    stored_version: int | None = None


#: Vocabulario declarado del resultado de la escritura (A8).
WRITE_OUTCOME_CREATED: Final[str] = "created"
WRITE_OUTCOME_UNCHANGED: Final[str] = "unchanged"
WRITE_OUTCOME_SUPERSEDED: Final[str] = "superseded"


def write_outcome_reason(outcome: str | None) -> str:
    """Motivo declarado del resultado de la escritura (A8). Nunca un valor mudo."""
    if outcome is None:
        return (
            "el etiquetado es una funcion pura y este run no se ha escrito en el almacen: "
            "no hay resultado de escritura que declarar"
        )
    if outcome == WRITE_OUTCOME_UNCHANGED:
        return (
            "las filas son **identicas** a las ya almacenadas (el anclaje ya era `session_open`): "
            "la regeneracion no cambia ningun dato, no se escribe nada y **no** se toca `version`"
        )
    if outcome == WRITE_OUTCOME_SUPERSEDED:
        return (
            "las identidades ya existian con otro contenido: se escribio una revision con "
            "`version` + 1 (*supersede* de la capa `derived`) y ningun fichero se borra"
        )
    return (
        "el almacen no tenia filas previas para estas identidades: se escribio su primera `version`"
    )


def _sigma_series(sessions: Sequence[SampleSession]) -> tuple[tuple[date, float], ...]:
    """Serie ``(sesion, sigma)`` realmente usada, para el hash de A28."""
    return tuple(
        (item.session, item.sigma)
        for item in sessions
        if item.unlabelled_reason is None and item.sigma is not None
    )


def _forecast_sha256(series: Sequence[tuple[date, float]]) -> str:
    """``sha256`` de la serie ``(sesion, sigma)`` usada: si cambia #7, el hash cambia."""
    payload = "\n".join(f"{day.isoformat()},{sigma!r}" for day, sigma in series)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_candidate(selection: Selection) -> str:
    """Candidato declarado a partir de la seleccion de #7 (A2).

    - si #7 elige candidato, se usa ese (es lo que manda su regla pre-registrada);
    - si el veredicto es ``no_better_than_naive``, el candidato declarado es ``rw``;
    - si el veredicto es ``inconclusive``, este modulo **no elige** por su cuenta:
      levanta :class:`SelectionInconclusiveError` y no se escribe nada.
    """
    if selection.selected is not None:
        return selection.selected
    if selection.verdict == Verdict.NO_BETTER_THAN_NAIVE.value:
        return "rw"
    raise SelectionInconclusiveError(
        "la regla de seleccion de #7 es `inconclusive` (los dos objetivos apuntan a "
        "candidatos distintos): este modulo no elige candidato por su cuenta y no escribe nada"
    )


def _median_bp(values: Sequence[float]) -> float | None:
    return None if not values else float(np.median(np.asarray(values, dtype=float))) * BP_PER_UNIT


def label_history(
    *,
    store: Store,
    now: datetime,
    k_sigma: float = PERSISTED_K_SIGMA,
    entry_price_source: str = DEFAULT_ENTRY_PRICE_SOURCE,
    series_id: str = SERIES_ID,
) -> LabelsRun:
    """Etiqueta el historico completo. **No escribe nada**: eso es ``write_outputs``.

    Consume la API publica de #7 (``build_sample``, ``walk_forward``,
    ``select_candidate``, ``scale_sigma_for_duration``) y la regla unica de la
    muestra limpia de #6/#7 (``session_stale_open`` y ``clean_sample_cutoff``).
    """
    if entry_price_source not in ENTRY_PRICE_SOURCES:
        raise ConfigurationError(
            f"fuente de precio de entrada desconocida: {entry_price_source!r}; "
            f"las declaradas son {sorted(ENTRY_PRICE_SOURCES)}"
        )
    spec = ENTRY_PRICE_SOURCES[entry_price_source]
    if spec.state != "available":
        raise EntryPriceUnavailableError(
            source=spec.name,
            state=spec.state,
            reason=spec.reason or "el almacen no tiene ese precio",
        )
    if k_sigma <= 0.0:
        raise ConfigurationError(f"`k` debe ser positivo, no {k_sigma!r}")

    frame, sample_summary = build_sample(store, series_id=series_id)
    walk = walk_forward(frame)
    selection = select_candidate(walk.candidates)
    candidate = resolve_candidate(selection)

    frame_sessions = [cast("date", value) for value in frame.get_column("session").to_list()]
    forecasts = walk.forecasts[candidate]
    first_index = walk.bounds[0][0] if walk.bounds else frame.height
    sigma_by_session: dict[date, float] = {}
    for index, day in enumerate(frame_sessions):
        if index < first_index:
            continue
        variance = float(forecasts[index])
        if not math.isfinite(variance) or variance <= 0.0:
            continue
        sigma_by_session[day] = math.sqrt(variance)

    calendar = MarketCalendar()
    intraday = _intraday(store)
    sessions, unlabelled = _sample_sessions(
        store=store,
        calendar=calendar,
        now=now,
        sigma_by_session=sigma_by_session,
        intraday=intraday,
        series_id=series_id,
    )
    rows = label_rows(sessions, k_sigma=k_sigma)

    scenarios: dict[str, tuple[LabelRow, ...]] = {}
    for scenario_k in R_SIGMA_SCENARIOS:
        scenarios[f"k_{scenario_k!r}"] = tuple(label_rows(sessions, k_sigma=scenario_k))
    if k_sigma not in R_SIGMA_SCENARIOS:
        scenarios[f"k_{k_sigma!r}"] = tuple(rows)

    summary: dict[str, object] = {}
    for scenario_k in (*R_SIGMA_SCENARIOS, k_sigma):
        key = f"k_{scenario_k!r}"
        summary[key] = scenario_summary(scenarios[key], k_sigma=scenario_k)

    sigma_series = _sigma_series(sessions)
    forecast_sha256 = _forecast_sha256(sigma_series)
    by_order = {
        INTRADAY_ORDER_SOURCE: sum(1 for row in rows if row.order_source == INTRADAY_ORDER_SOURCE),
        FALLBACK_ORDER_SOURCE: sum(1 for row in rows if row.order_source == FALLBACK_ORDER_SOURCE),
    }
    fallback_share = by_order[FALLBACK_ORDER_SOURCE] / len(rows) if rows else 0.0
    coverage = dict(intraday.coverage)
    coverage["ordered_with_intraday"] = by_order[INTRADAY_ORDER_SOURCE]
    coverage["ordered_with_fallback"] = by_order[FALLBACK_ORDER_SOURCE]
    coverage["fallback_share"] = fallback_share
    coverage["daily_series_id"] = series_id
    coverage["stored_daily_sessions"] = _stored_daily_sessions(store, series_id)
    coverage["daily_sessions_with_ohlc"] = int(str(sample_summary["sessions"]))
    coverage["daily_sessions"] = coverage["stored_daily_sessions"]
    coverage["bars_at_0845_et"] = _bars_at_0845_et(store)
    coverage["follow_up_issues"] = [50, 57]

    # A6/A7: verificacion del anclaje contra la primera barra de las 09:30 ET. Es de
    # **solo lectura**: no cambia ninguna etiqueta ni excluye ninguna sesion.
    auction = auction_verification(sessions)

    warmup_from = None if first_index == 0 else frame_sessions[0].isoformat()
    warmup_to = None if first_index == 0 else frame_sessions[first_index - 1].isoformat()
    sigmas = [sigma for _, sigma in sigma_series]
    frame_sigmas = list(sigma_by_session.values())
    inputs: dict[str, object] = {
        "series_id": series_id,
        "interval": INTERVAL,
        "k_sigma": k_sigma,
        "entry_price_source": entry_price_source,
        "daily_sessions": coverage["daily_sessions"],
        "intraday_coverage": coverage,
        "auction_verification": auction,
        "sigma": {
            "candidate": candidate,
            "selection_verdict": selection.verdict,
            "selection_selected": selection.selected,
            "selection_rule": list(selection.rule),
            "min_train": MIN_TRAIN,
            "refit_every": int(str(selection.constants["refit_every"])),
            "first_evaluated": None
            if not frame_sessions
            else frame_sessions[first_index].isoformat()
            if first_index < len(frame_sessions)
            else None,
            "evaluated_sessions": max(frame.height - first_index, 0),
            "warmup_sessions": first_index,
            "warmup_from": warmup_from,
            "warmup_to": warmup_to,
            "median_sigma_bp": _median_bp(frame_sigmas),
            "median_sigma_labelled_bp": _median_bp(sigmas),
            "median_sigma_bp_note": (
                "`median_sigma_bp` es la mediana de las sesiones con *forecast* del frame de #7 "
                "(comparable con su ancla); `median_sigma_labelled_bp` incluye ademas las medias "
                "sesiones con la sigma escalada"
            ),
            "sessions_with_forecast": len(sigma_by_session),
            "forecast_sha256": forecast_sha256,
            "no_imputation": (
                "las sesiones del frame sin *forecast* finito quedan `unlabelled: no_forecast`; "
                "esta prohibido imputar sigma 0, la media o la de otra sesion"
            ),
            "half_day_rule": (
                "una media sesion no esta en el frame de #7, asi que su sigma se arrastra desde la "
                "ultima sesion completa etiquetada y se escala con `scale_sigma_for_duration`"
            ),
        },
        "clean_sample": {
            **sample_summary,
            "definition": (
                "regla unica de #52: `session_stale_open` + `clean_sample_cutoff` de "
                "`analysis/drift.py`, importadas (no recalculadas con un corte a mano)"
            ),
        },
    }
    sample: dict[str, object] = {
        **unlabelled,
        "labelled": len(rows),
        "labelled_full_sessions": sum(1 for row in rows if not row.is_half_day),
        "labelled_half_sessions": sum(1 for row in rows if row.is_half_day),
        "by_order_source": by_order,
        "fallback_share": fallback_share,
        "bias": "conservative_lower_bound_on_p_target",
        "bias_sentence": (
            "el `p_target` de las sesiones con respaldo es una **cota inferior**: el verdadero no "
            "puede ser menor"
        ),
        "ties": {
            "in_bar": sum(row.ties_in_bar for row in rows),
            "fallback": sum(row.fallback_ties for row in rows),
        },
        "unlabelled_total": sum(
            cast("int", cast("dict[str, object]", detail)["count"])
            for detail in cast("dict[str, object]", unlabelled["reasons"]).values()
        ),
        "no_forecast_note": (
            "`no_forecast` incluye el calentamiento de #7 (las primeras sesiones del frame, sin "
            "*forecast* finito) y las medias sesiones anteriores a la primera sesion con "
            "*forecast*, que no tienen sigma que arrastrar"
        ),
    }

    limitations = _limitations(
        intraday_sessions=int(str(coverage["sessions"])),
        labelled=len(rows),
        fallback_share=fallback_share,
        warmup_sessions=first_index,
    )
    notes = (
        "unidades: **fraccion** en el calculo (sigma, `target_pct`, `stop_pct`, `c` y los "
        "retornos) y **bp** (x 10^4) al informar, como `analysis/drift.py`",
        f"solo se usa el intradia de `{SERIES_ID}` a {INTERVAL}: `{INTRADAY_UNUSED_SERIES}` no "
        "ordena barreras (A13; la fuente intradia y su compra son #50)",
        "cuando una barra toca las dos barreras el orden es desconocido y la etiqueta es la "
        "**adversa** en las dos direcciones: el sesgo es conservador por construccion",
        "`p*` y el veredicto de Fase 0 son de #9: aqui no se calculan",
        "el `open` repetido de Yahoo (#52) no se arregla aqui: se consume el corte que ya "
        "calcula la regla unica de la muestra limpia",
        "sin overnight (`plan.md` §12, regla 6): ninguna columna usa el `open` de la sesion "
        "siguiente y ninguna etiqueta de `t` cambia si se anade o se modifica `t+1`",
    )
    return LabelsRun(
        as_of=now,
        series_id=series_id,
        interval=INTERVAL,
        interval_hours=FULL_SESSION_HOURS,
        k_sigma=k_sigma,
        entry_price_source=entry_price_source,
        rows=tuple(rows),
        scenarios=scenarios,
        inputs=inputs,
        sample=sample,
        summary=summary,
        cost=_cost_block(),
        forecast_sha256=forecast_sha256,
        forecast_candidate=candidate,
        selection_verdict=selection.verdict,
        limitations=limitations,
        notes=notes,
    )


def _limitations(
    *, intraday_sessions: int, labelled: int, fallback_share: float, warmup_sessions: int
) -> tuple[str, ...]:
    """Limitaciones declaradas, con los numeros medidos (A33)."""
    return (
        f"solo {intraday_sessions} sesiones del almacen tienen intradia de `{SERIES_ID}` 5m, asi "
        f"que {fallback_share:.1%} de las {labelled} etiquetas vienen del respaldo diario "
        "**conservador**: `p_target` es una **cota inferior** (#50 fuente intradia; #57 RV "
        "intradia)",
        "el precio de entrada usado es el `open` de la subasta de apertura (09:30 ET) de "
        "`^GSPC`: la decision del propietario esta **cerrada** (2026-09-18) y **coincide** con "
        "`plan.md` §4.1, que **no** se reescribe; el precio sigue siendo el del **indice** y "
        "**no** el del **CFD** (#50; registro de la decision en #61), y los numeros publicados "
        "**heredan el *look-ahead* de la muestra completa de #7** (#63) y **no** son una "
        "validacion de la estrategia",
        "la etiqueta es de `^GSPC`, **no** del CFD, y **no** lleva spread ni *slippage* por "
        "operacion (el modelo de coste del motor es #11)",
        "la muestra usa el corte limpio de #52: el `open` diario de la fuente repite el cierre "
        "anterior en parte del historico",
        f"{warmup_sessions} sesiones de calentamiento quedan sin etiqueta (sin *forecast* finito: "
        f"`MIN_TRAIN = {MIN_TRAIN}` de #7)",
        "las medias sesiones se etiquetan, pero el gate no las opera (`plan.md` §12, regla 18)",
        f"`R = k * sigma` con `k` **ilustrativo** (decision abierta 5 → #60); solo se persiste "
        f"`k = {PERSISTED_K_SIGMA}` y las barreras son simetricas",
    )


def _cost_block() -> dict[str, object]:
    """Coste declarado del EV neto, con su procedencia (A24)."""
    return {
        "round_trip_spread": ROUND_TRIP_SPREAD,
        "round_trip_spread_pct": "0,0042 % (0,42 $ sobre 10.000 $ de nocional)",
        "provenance": ROUND_TRIP_SPREAD_PROVENANCE,
        "applies_to": "intradia puro, sin financiacion (el CFD solo cotiza 09:30-16:00 ET)",
        "basis": "ev_media (la media de los retornos realizados)",
        "identity_tolerance": EV_IDENTITY_TOLERANCE,
        "sensitivity_scenarios": COST_SENSITIVITY_SCENARIOS,
        "sensitivity_note": (
            "son **sensibilidad** (tenencia de un dia completo, totales declarados de #8) y "
            "**nunca** se fusionan con el diferencial intradia ni entre si"
        ),
        "engine_cost_model": (
            "el modelo de coste del motor es #11 (`src/cfdtrader/backtest/costs.py`); aqui solo se "
            "usa el diferencial intradia declarado para el EV"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Resumen por direccion (A21-A25)
# ─────────────────────────────────────────────────────────────────────────────
def _mean(values: Sequence[float]) -> float | None:
    """Media, o ``None`` si no hay valores: nunca un ``0`` por desconocido."""
    return None if not values else sum(values) / len(values)


def summarise_rows(
    rows: Sequence[LabelRow], *, direction: str, k_sigma: float
) -> dict[str, object]:
    """Etiquetas, ``p_win`` y EV de una direccion sobre esa muestra (A21-A25)."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direccion desconocida: {direction!r}")
    is_long = direction == "long"
    labels = [row.label_long if is_long else row.label_short for row in rows]
    returns = [row.ret_long if is_long else row.ret_short for row in rows]
    total = len(returns)
    counts = {label: labels.count(label) for label in LABELS}
    if total == 0:
        return {
            "direction": direction,
            "k_sigma": k_sigma,
            "sessions": 0,
            "counts": counts,
            "p_target": None,
            "p_stop": None,
            "p_time": None,
            "p_win": None,
            "e_gain": None,
            "e_loss": None,
            "ev_identity": None,
            "ev_media": None,
            "ev_identity_difference": None,
        }

    wins = [value for value in returns if value > 0.0]
    losses = [-value for value in returns if value <= 0.0]
    p_win = len(wins) / total
    e_gain = _mean(wins)
    e_loss = _mean(losses)
    ev_media = _mean(returns)
    ev_identity = (
        None if e_gain is None or e_loss is None else p_win * e_gain - (1.0 - p_win) * e_loss
    )
    difference = None if ev_identity is None or ev_media is None else abs(ev_identity - ev_media)
    ev_neto = None if ev_media is None else ev_media - ROUND_TRIP_SPREAD
    return {
        "direction": direction,
        "k_sigma": k_sigma,
        "sessions": total,
        "counts": counts,
        "p_target": counts[LABEL_TARGET] / total,
        "p_stop": counts[LABEL_STOP] / total,
        "p_time": counts[LABEL_TIME] / total,
        "p_target_note": (
            "`p_target` es la magnitud comparable con `p*` de `plan.md` §4.4 (bracket simetrico); "
            "el calculo de `p*` es de #9 y aqui no se implementa"
        ),
        "p_win": p_win,
        "p_win_note": (
            "`p_win >= p_target`: las salidas `time` positivas tambien cuentan como victoria "
            "(igualdad solo si ninguna salida `time` es positiva)"
        ),
        "e_gain": e_gain,
        "e_loss": e_loss,
        "ev_identity": ev_identity,
        "ev_media": ev_media,
        "ev_identity_difference": difference,
        "ev_identity_tolerance": EV_IDENTITY_TOLERANCE,
        "ev_neto": ev_neto,
        "ev_neto_basis": "ev_media",
        "c": ROUND_TRIP_SPREAD,
        "c_provenance": ROUND_TRIP_SPREAD_PROVENANCE,
        "bp": {
            "e_gain": None if e_gain is None else e_gain * BP_PER_UNIT,
            "e_loss": None if e_loss is None else e_loss * BP_PER_UNIT,
            "ev_identity": None if ev_identity is None else ev_identity * BP_PER_UNIT,
            "ev_media": None if ev_media is None else ev_media * BP_PER_UNIT,
            "ev_neto": None if ev_neto is None else ev_neto * BP_PER_UNIT,
            "c": ROUND_TRIP_SPREAD * BP_PER_UNIT,
            "sensitivity_one_day": (
                {}
                if ev_media is None
                else {
                    name: (ev_media - value) * BP_PER_UNIT
                    for name, value in sorted(COST_SENSITIVITY_SCENARIOS.items())
                }
            ),
        },
        "sensitivity_note": (
            "las variantes `short_one_day` / `long_one_day` son sensibilidad declarada de #8 "
            "(tenencia de un dia completo): no se suman al diferencial intradia ni entre si"
        ),
    }


def scenario_summary(rows: Sequence[LabelRow], *, k_sigma: float) -> dict[str, object]:
    """Resumen de un escenario ``k``: por direccion y en las dos muestras (A22, A25)."""
    full_only = [row for row in rows if not row.is_half_day]
    return {
        "k_sigma": k_sigma,
        "target_pct_note": (
            "`target_pct = stop_pct = k * sigma_t`; `R` en fraccion, con `R = 0,5 / 1,0 / 1,5 %` "
            "de `plan.md` §4.4 solo como ejemplo ilustrativo"
        ),
        "directions": {
            direction: {
                "all": summarise_rows(rows, direction=direction, k_sigma=k_sigma),
                "full_only": summarise_rows(full_only, direction=direction, k_sigma=k_sigma),
            }
            for direction in DIRECTIONS
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Persistencia (A26-A28)
# ─────────────────────────────────────────────────────────────────────────────
def record_for(row: LabelRow, *, run: LabelsRun, now: datetime) -> dict[str, object]:
    """Registro de ``derived.labels`` de una sesion (una fila, las dos direcciones)."""
    return {
        "source": LABELS_SOURCE,
        "series_id": run.series_id,
        "as_of": row.close_utc,
        "fetched_at": now,
        "published_at": row.close_utc,
        "session": row.session,
        "is_half_day": row.is_half_day,
        "sigma": row.sigma,
        "sigma_carrier": row.sigma_carrier,
        "k_sigma": row.k_sigma,
        "target_pct": row.target_pct,
        "stop_pct": row.stop_pct,
        "entry_px": row.entry_px,
        "entry_price_source": run.entry_price_source,
        "order_source": row.order_source,
        "bars_observed": row.bars_observed,
        "expected_bars": row.expected_bars,
        "coverage": row.coverage,
        "intraday_incomplete": row.intraday_incomplete,
        "ties_in_bar": row.ties_in_bar,
        "fallback_ties": row.fallback_ties,
        "decided_bar": row.decided_bar,
        "label_long": row.label_long,
        "label_short": row.label_short,
        "exit_long": row.exit_long,
        "exit_short": row.exit_short,
        "ret_long": row.ret_long,
        "ret_short": row.ret_short,
        "forecast_candidate": run.forecast_candidate,
        "selection_verdict": run.selection_verdict,
        "forecast_sha256": run.forecast_sha256,
    }


def persist_rows(store: Store, records: Sequence[dict[str, object]]) -> str:
    """Escribe las etiquetas en ``derived.labels`` por la API del ``Store``.

    ``append`` para identidades nuevas y ``replace`` (solo ``derived``, que es
    recomputable) cuando el contenido cambia: es la semantica de *supersede* de la
    capa, la vista sigue devolviendo una fila vigente por sesion y **ningun fichero
    se borra**. Una ejecucion identica es un no-op. Devuelve el resultado declarado
    (``created`` / ``unchanged`` / ``superseded``), que el informe publica (A8).
    """
    if not records:
        return WRITE_OUTCOME_UNCHANGED
    try:
        outcome = store.append(LABELS_LAYER, LABELS_DATASET, list(records))
    except ImmutableWriteError:
        logger.info(
            "las etiquetas cambian para identidades ya almacenadas: se escribe una revision "
            "(`replace` en `derived.labels`, supersede con `version` + 1)"
        )
        store.replace(LABELS_LAYER, LABELS_DATASET, list(records))
        return WRITE_OUTCOME_SUPERSEDED
    return WRITE_OUTCOME_UNCHANGED if outcome is WriteOutcome.UNCHANGED else WRITE_OUTCOME_CREATED


def _stored_version(store: Store) -> int | None:
    """``version`` vigente de ``derived.labels``, o ``None`` si no hay filas.

    Se lee con ``store.sql()`` (nunca con ``read_pit``, que responde «que sabiamos en
    T» y no sirve para esto). El almacen es el dueno del numero: aqui solo se publica.
    """
    try:
        frame = store.sql(
            f"SELECT max(version) AS version FROM {LABELS_LAYER}.{LABELS_DATASET}"  # noqa: S608
        )
    except (UnknownDatasetError, duckdb.Error):
        return None
    if frame.height == 0:
        return None
    value = frame.get_column("version").to_list()[0]
    return None if value is None else int(str(value))


@dataclass(frozen=True, slots=True)
class LabelsOutputs:
    """Resultado de escribir: que se hizo y donde."""

    outcome: str
    sessions: int
    json_path: Path
    markdown_path: Path
    #: El run **con** el resultado de la escritura ya declarado (A8).
    run: LabelsRun | None = None


def report_payload(run: LabelsRun) -> dict[str, object]:
    """Payload JSON del informe. **Determinista**: no lleva nada que varie entre ejecuciones."""
    coverage = cast("dict[str, object]", run.inputs["intraday_coverage"])
    return {
        "task": 10,
        "title": "Etiquetado tri-barrera (target / stop / time) para LONG y SHORT",
        "generated_at": run.as_of.isoformat(),
        "phase0_context": phase0_context(),
        "semantics": {
            "target": "se toca primero la barrera **favorable**",
            "stop": "se toca primero la barrera **adversa**",
            "time": "no se toca ninguna antes del cierre de sesion: la posicion sale al cierre",
            "not_up_down": ("la etiqueta **no** es «arriba/abajo», es «favorable/adversa»"),
            "long": {
                "favourable": "entry_px * (1 + target_pct) (arriba)",
                "adverse": "entry_px * (1 - stop_pct) (abajo)",
            },
            "short": {
                "favourable": "entry_px * (1 - target_pct) (abajo)",
                "adverse": "entry_px * (1 + stop_pct) (arriba)",
            },
            "touch": (
                "inclusivo (`high >= barrera` / `low <= barrera`) y comparado con el precio de "
                "barrera **sin redondear**"
            ),
            "tie_in_bar": (
                "si la misma barra toca las dos barreras, la etiqueta es la adversa (`stop`)"
                " en las dos direcciones; se cuenta en `ties_in_bar`"
            ),
            "tie_in_fallback": (
                "si el OHLC diario toca las dos barreras, la etiqueta es `stop`; se cuenta en "
                "`fallback_ties`"
            ),
            "no_overnight": (
                "ninguna etiqueta ni columna usa el `open` de la sesion siguiente (`plan.md` §12, "
                "regla 6); la barrera temporal es el cierre de sesion del calendario"
            ),
            "no_look_ahead": (
                "la etiqueta de `t` usa solo informacion de `t` y la sigma de `t`, que se produce "
                "con datos anteriores a `t`"
            ),
            "time_barrier": (
                "cierre de sesion de `MarketCalendar.close_utc`: 16:00 ET en sesion completa y "
                "13:00 ET en media sesion; ninguna hora fija CET/Madrid"
            ),
        },
        "units": {
            "calculation": "fraccion",
            "report": "bp (x 10^4) al informar; se publican las dos donde ayuda a leer",
            "bp_per_unit": BP_PER_UNIT,
        },
        "declared_constants": [
            {
                "name": "R_SIGMA_SCENARIOS",
                "value": list(R_SIGMA_SCENARIOS),
                "unit": "multiplos de sigma",
                "provenance": (
                    "escenario ilustrativo, no decision del propietario; " + DECISION_NOTE
                ),
                "note": "barreras simetricas: `target_pct = stop_pct = k * sigma_t`",
            },
            {
                "name": "PERSISTED_K_SIGMA",
                "value": PERSISTED_K_SIGMA,
                "unit": "multiplos de sigma",
                "provenance": "escenario que se persiste en esta tarea",
                "note": "solo se persiste este `k`; los otros se publican como escenario",
            },
            {
                "name": "MIN_INTRADAY_COVERAGE",
                "value": MIN_INTRADAY_COVERAGE,
                "unit": "fraccion de barras esperadas",
                "provenance": "A13 del enunciado de #10",
                "note": "por debajo se usa el respaldo diario declarado y conservador",
            },
            {
                "name": "BARS_PER_HOUR",
                "value": BARS_PER_HOUR,
                "unit": "barras por hora",
                "provenance": "barras de 5 minutos",
                "note": (
                    f"{round(FULL_SESSION_HOURS * BARS_PER_HOUR)} barras esperadas en sesion "
                    f"completa y {round(HALF_SESSION_HOURS * BARS_PER_HOUR)} en media sesion"
                ),
            },
            {
                "name": "ROUND_TRIP_SPREAD",
                "value": ROUND_TRIP_SPREAD,
                "unit": "fraccion del nocional",
                "provenance": ROUND_TRIP_SPREAD_PROVENANCE,
                "note": "diferencial de ida y vuelta, intradia puro; es el `c` del EV neto",
            },
            {
                "name": "COST_SENSITIVITY_SCENARIOS",
                "value": COST_SENSITIVITY_SCENARIOS,
                "unit": "fraccion del nocional",
                "provenance": "#8 (totales declarados para una tenencia de un dia completo)",
                "note": "sensibilidad: nunca se fusionan con el diferencial intradia",
            },
            {
                "name": "EV_IDENTITY_TOLERANCE",
                "value": EV_IDENTITY_TOLERANCE,
                "unit": "fraccion",
                "provenance": "tolerancia declarada de la identidad del EV (A23)",
                "note": "`p_win * E[G] - (1 - p_win) * E[P]` frente a la media de los retornos",
            },
            {
                "name": "LABELS_SOURCE",
                "value": LABELS_SOURCE,
                "unit": "texto",
                "provenance": "dataset derivado: no hay fuente externa",
                "note": f"`source` de las filas de `{LABELS_LAYER}.{LABELS_DATASET}`",
            },
            {
                "name": "INTERVAL",
                "value": INTERVAL,
                "unit": "texto",
                "provenance": "A13: solo se usa el intradia de `^GSPC` a 5 minutos",
                "note": f"`{INTRADAY_UNUSED_SERIES}` se cuenta pero no ordena barreras",
            },
        ],
        "entry_price": _entry_price_block(
            source_used=run.entry_price_source,
            evidence=coverage,
            auction=cast("dict[str, object]", run.inputs["auction_verification"]),
        ),
        "inputs": run.inputs,
        "sample": run.sample,
        "summary": run.summary,
        "cost": run.cost,
        "persistence": {
            "layer": LABELS_LAYER,
            "dataset": LABELS_DATASET,
            "source": LABELS_SOURCE,
            "series_id": run.series_id,
            "grain": "(series_id, session): una fila por sesion, las dos direcciones en columnas",
            "as_of": "cierre de sesion en UTC (`MarketCalendar.close_utc`)",
            "published_at": "igual a `as_of`: la etiqueta solo se conoce al cerrar la sesion",
            "session": "fecha ET de la sesion",
            "k_sigma_persisted": run.k_sigma,
            "sessions_persisted": len(run.rows),
            "supersede": (
                "una segunda ejecucion identica es un no-op (`unchanged`, mismos ficheros); un "
                "cambio de parametro (p. ej. `k`) escribe una revision con `version` + 1 y la "
                "vista sigue devolviendo **una** fila vigente por sesion; ningun fichero se borra"
            ),
            "outcome_note": (
                "el resultado de la escritura **si** se publica (`write_outcome`): dos "
                "ejecuciones que parten del mismo estado del almacen producen el mismo JSON "
                "byte a byte, y el resultado no se toca a mano"
            ),
            "write_outcome": run.write_outcome,
            "write_outcome_vocabulary": [
                WRITE_OUTCOME_CREATED,
                WRITE_OUTCOME_UNCHANGED,
                WRITE_OUTCOME_SUPERSEDED,
            ],
            "write_outcome_state": ("declared" if run.write_outcome is not None else "not_written"),
            "write_outcome_reason": write_outcome_reason(run.write_outcome),
            "rows": len(run.rows) if run.write_outcome is not None else None,
            "version": run.stored_version,
            "version_note": (
                "el `version` lo posee el almacen: una regeneracion identica **no** lo toca, y "
                "forzar un cambio de version a mano esta prohibido"
            ),
            "verification": (
                "la recomputacion se comprueba con `store.sql()` (y **no** con `read_pit`, cuya "
                "semantica es «que sabiamos en T» y no sirve para un estudio historico)"
            ),
        },
        "limitations": list(run.limitations),
        "notes": list(run.notes),
    }


#: Nota que acompana a los escenarios de ``k``: no son una decision del propietario.
DECISION_NOTE: Final[str] = "la decision de umbrales y tamano de `R` es la decision 5 → #60"


# ─────────────────────────────────────────────────────────────────────────────
# Informe legible
# ─────────────────────────────────────────────────────────────────────────────
def _block(value: object) -> dict[str, object]:
    """Sub-diccionario del payload, tipado para el modo estricto."""
    return cast("dict[str, object]", value)


def _number(value: object) -> float | None:
    """Numero del payload, o ``None`` si es nulo."""
    return None if value is None else float(cast("float", value))


def _bp(value: object) -> str:
    """Valor en fraccion como bp, o una raya si es desconocido."""
    number = _number(value)
    return "—" if number is None else f"{number * BP_PER_UNIT:.3f}"


def _bp_value(value: object) -> str:
    """Valor que **ya** viene en bp, formateado con una decimal."""
    number = _number(value)
    return "—" if number is None else f"{number:.1f} bp"


def _pct(value: object) -> str:
    """Valor en fraccion como porcentaje, o una raya si es desconocido."""
    number = _number(value)
    return "—" if number is None else f"{number:.2%}"


def render_markdown(run: LabelsRun) -> str:
    """Informe legible del etiquetado, empezando por lo que hay que saber."""
    context = phase0_context()
    sample = run.sample
    coverage = cast("dict[str, object]", run.inputs["intraday_coverage"])
    sigma = _block(run.inputs["sigma"])
    entry_price = _block(report_payload(run)["entry_price"])
    auction = _block(entry_price["auction_verification"])
    labelled = cast("int", sample["labelled"])
    full_sessions = cast("int", sample["labelled_full_sessions"])
    half_sessions = cast("int", sample["labelled_half_sessions"])
    unlabelled_total = cast("int", sample["unlabelled_total"])
    ordered = cast("dict[str, object]", sample["by_order_source"])
    ties = cast("dict[str, object]", sample["ties"])
    fallback_share = float(str(sample["fallback_share"]))
    ready_text = str(context["phase1_ready"]).lower()
    decided_on = entry_price["decided_on"]
    provenance = entry_price["provenance"]
    proxy_text = str(entry_price["source_is_proxy"]).lower()
    diverges_text = str(entry_price["diverges_from_owner_decision"]).lower()
    contradicts_text = str(entry_price["contradicts_plan_md_4_1"]).lower()
    tradable_text = str(entry_price["not_tradable"]).lower()

    unused = _block(_block(coverage["unused_series"])[INTRADAY_UNUSED_SERIES])
    clean = _block(run.inputs["clean_sample"])

    lines = [
        "# Etiquetado tri-barrera (tarea #10)",
        "",
        f"- **Serie:** `{run.series_id}` (intradia `{run.interval}`) — **etiqueta de indice**",
        "- **no del CFD**: las barreras se miden con el indice, no con el CFD (#50)",
        f"- **Calculado:** {run.as_of.isoformat()} · candidato de #7: `{run.forecast_candidate}` "
        f"(`{run.selection_verdict}`) · `k` persistido = {run.k_sigma}",
        f"- **Sesiones etiquetadas:** {labelled} ({full_sessions} completas, "
        f"{half_sessions} medias)",
        f"- **Orden:** intradia {ordered[INTRADAY_ORDER_SOURCE]} · respaldo diario "
        f"{ordered[FALLBACK_ORDER_SOURCE]} (`fallback_share` = {fallback_share:.3f})",
        f"- **Sin etiquetar:** {unlabelled_total} sesiones (motivos abajo)",
        "",
        "## Contexto de Fase 0: la puerta esta en `fail`",
        "",
        f"- `gate`: **`{context['gate']}`** · `phase1_ready`: **`{ready_text}`** "
        f"(mitad (a) `{context['half_a']}`, mitad (b) `{context['half_b']}`), segun la tarea #9.",
        f"- Decision del propietario del **{context['decided_on']}** "
        f"(procedencia: *{context['provenance']}*, issue de origen #{context['source_issue']}): "
        f"{context['owner_decision']}.",
        (
            "- **Esta etiqueta no es una validacion de la estrategia**: dice que barrera se toco "
            "primero, no que la estrategia gane dinero. La Fase 1 se construye con la puerta en "
            "`fail` y el artefacto no presenta la puerta como superada."
        ),
        "",
        "## Semantica de la etiqueta",
        "",
        "- `target` = se toca primero la barrera **favorable**; `stop` = se toca la barrera "
        "**adversa**; `time` = no se toca ninguna antes del cierre y la posicion sale al cierre.",
        "- La etiqueta **no** es «arriba/abajo», es «favorable/adversa». LONG: favorable arriba "
        "(`entry * (1 + target_pct)`), adversa abajo. SHORT, espejo: favorable abajo "
        "(`entry * (1 - target_pct)`), adversa arriba.",
        (
            "- El toque es **inclusivo** (`high >= barrera`, `low <= barrera`) y se compara con "
            "el precio de barrera **sin redondear**."
        ),
        "- **Empate dentro de la misma barra** ⇒ etiqueta **adversa** (`stop`) para las dos "
        f"direcciones (`ties_in_bar` = {ties['in_bar']}); en el respaldo diario, igual "
        f"(`fallback_ties` = {ties['fallback']}).",
        (
            "- Barrera temporal = cierre de sesion del `MarketCalendar`: **16:00 ET** en sesion "
            "completa y **13:00 ET** en media sesion. Sin overnight (`plan.md` §12, regla 6)."
        ),
        "",
        "## Barreras: proporcionales al *forecast* de #7",
        "",
        "- `target_pct = stop_pct = k * sigma_t`, con `sigma_t` la raiz de la varianza",
        f"pronosticada por el walk-forward de #7 (`{sigma['candidate']}`, veredicto "
        f"`{sigma['selection_verdict']}`, `min_train = {sigma['min_train']}`).",
        f"- `sigma` mediana del frame de #7: **{_bp_value(sigma['median_sigma_bp'])}** por sesion "
        f"({sigma['sessions_with_forecast']} sesiones con *forecast*); "
        f"**{_bp_value(sigma['median_sigma_labelled_bp'])}** contando las medias sesiones "
        "escaladas.",
        (
            f"- Escenarios de `k` publicados: `{list(R_SIGMA_SCENARIOS)}` — **ilustrativo, no "
            f"decision del propietario** (decision abierta 5 → #60); solo se persiste "
            f"`k = {PERSISTED_K_SIGMA}`."
        ),
        "- Las barreras son **simetricas**: la asimetria queda fuera de alcance. Los valores `R = "
        "0,5 / 1,0 / 1,5 %` de `plan.md` §4.4 se citan solo como ejemplo.",
        "",
        "## Precio de entrada: decision del propietario, cerrada el 2026-09-18",
        "",
        f"- El propietario **cerro** la cuestion el {decided_on} (procedencia: *{provenance}*): "
        f"**{entry_price['owner_decision']}**.",
        f"- El anclaje **coincide** con `plan.md` §4.1 ({entry_price['plan_md_4_1_proposes']}), "
        "que **no** se reescribe: la coincidencia se **declara**, no se parchea el documento "
        "(la anotacion en los documentos es #61 y #65).",
        f"- Rastro de lo sustituido: `previous_owner_decision` = "
        f"**{entry_price['previous_owner_decision']}**, descartada el "
        f"{entry_price['previous_owner_decision_discarded_on']}.",
        f"- **`diverges_from_owner_decision` = `{diverges_text}`** y "
        f"**`contradicts_plan_md_4_1` = `{contradicts_text}`**: "
        f"{entry_price['contradicts_plan_md_4_1_reason']}.",
        f"- **`not_tradable` = `{tradable_text}`**: {entry_price['not_tradable_reason']}.",
        f"- `source_used` = `{entry_price['source_used']}` (fuente por defecto: "
        f"`{entry_price['default_source']}`), con `source_is_proxy` = `{proxy_text}`.",
        f"- El precio usado es el del **indice** (`{run.series_id}`), **no** el del **CFD** "
        f"(`{entry_price['proxy_of']}`): {entry_price['proxy_note']} → "
        f"**#{entry_price['follow_up_issue']}**.",
        "- `t0_snapshot_0845_et` **sigue declarada** en el registro de fuentes con "
        "`state: unavailable`, y pedirla **falla con un motivo declarado** (codigo 2, sin "
        "escribir informe ni dataset): **no** hay *fallback* silencioso.",
        f"- Verificacion de la subasta (solo lectura): **{auction['sessions_compared']}** "
        f"sesiones comparadas, **{auction['identical']}** identicas, `max_abs_diff_bp` = "
        f"{auction['max_abs_diff_bp']} y `status` = `{auction['status']}` con la tolerancia "
        f"declarada de {_block(auction['tolerance'])['value_bp']} bp.",
        f"- Evidencia medida en el almacen: la primera barra intradia de `{run.series_id}` es "
        f"`{coverage['first_bar_utc']}` ({coverage['first_bar_et']}) y hay "
        f"**{coverage['bars_at_0845_et']}** barras a las 08:45 ET.",
        "- Los numeros publicados **heredan el *look-ahead* de la muestra completa de #7** "
        "(**#63**) y **no** son una validacion de la estrategia.",
        "",
        "## Fuente del orden y cobertura",
        "",
        f"- `intraday_5m` si la sesion tiene barras de `{run.series_id}` 5m dentro de "
        f"[`open_utc`, `close_utc`] y su cobertura es `>= {MIN_INTRADAY_COVERAGE}` sobre las "
        f"barras esperadas (`{coverage['expected_bars_full_session']}` en sesion completa, "
        f"`{coverage['expected_bars_half_session']}` en media); en cualquier otro caso "
        "`daily_ohlc_fallback` con `intraday_incomplete`.",
        f"- Cobertura real medida: **{coverage['sessions']}** sesiones con intradia de "
        f"`{coverage['series_id']}` 5m ({coverage['bars']} barras), frente a "
        f"**{coverage['daily_sessions']}** sesiones diarias almacenadas. "
        f"`{INTRADAY_UNUSED_SERIES}` se cuenta "
        f"({unused['sessions']} sesiones) pero "
        "**no** ordena barreras (#50).",
        f"- Sesiones ordenadas con intradia: **{coverage['ordered_with_intraday']}**; con respaldo "
        f"diario: **{coverage['ordered_with_fallback']}**.",
        f"- **Sesgo declarado:** `{sample['bias']}` — {sample['bias_sentence']}.",
        "- Consecuencia: el respaldo diario **nunca** elige la favorable cuando el orden es "
        "desconocido, asi que `p_target` **subestima** el valor real: el numero bueno sera "
        "mayor, nunca menor.",
        "",
        "## Muestra y sigma",
        "",
        f"- Muestra de #7: **{clean['clean_sessions']}** sesiones limpias desde `clean_from` = "
        f"`{clean['clean_from']}`, sobre {coverage['daily_sessions_with_ohlc']} sesiones diarias "
        f"con OHLC ({coverage['stored_daily_sessions']} almacenadas); **la misma** definicion que "
        "#6/#7 (importada, no recalculada).",
        f"- Calentamiento sin *forecast*: **{sigma['warmup_sessions']}** sesiones "
        f"({sigma['warmup_from']} → {sigma['warmup_to']}) quedan `unlabelled: no_forecast`; "
        "esta prohibido imputar sigma.",
        f"- Medias sesiones: se etiquetan con el cierre de las 13:00 ET y su sigma **arrastrada** "
        f"desde la ultima sesion completa etiquetada y escalada con `scale_sigma_for_duration` "
        f"({half_sessions} filas con `sigma_carrier = {CARRIER_PREVIOUS_LABELLED}`).",
        "- Motivos de descarte, en orden de precedencia: "
        + ", ".join(f"`{reason}`" for reason in UNLABELLED_REASONS)
        + ".",
        "",
        "| motivo | sesiones | primera | ultima |",
        "|---|---|---|---|",
    ]
    for reason in UNLABELLED_REASONS:
        detail = _block(_block(sample["reasons"])[reason])
        first = detail["first"] or "—"
        last = detail["last"] or "—"
        lines.append(f"| `{reason}` | {detail['count']} | {first} | {last} |")

    lines.extend(
        [
            "",
            "## Resumen por escenario y direccion",
            "",
            "`p_target` es la magnitud comparable con `p*` de `plan.md` §4.4 (bracket simetrico); "
            "**el calculo de `p*` es de #9 y aqui no se implementa**. `p_win` cuenta como victoria "
            "cualquier retorno positivo, incluidas las salidas `time` (`p_win >= p_target`). "
            "El EV usa `p_win`, `E[G]` y `E[P]` por las dos vias declaradas, y "
            f"`ev_neto = ev_media - c` con `c = {_bp(ROUND_TRIP_SPREAD)}` bp.",
            "",
        ]
    )
    for scenario in run.summary.values():
        block = _block(scenario)
        lines.extend(
            [
                f"### Escenario `k = {block['k_sigma']}`"
                + (" (persistido)" if _number(block["k_sigma"]) == run.k_sigma else ""),
                "",
                "| direccion | muestra | sesiones | `p_target` | `p_stop` | `p_time` | `p_win` | "
                "E[G] bp | E[P] bp | EV bruto bp | EV neto bp |",
                "|---|---|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for direction in DIRECTIONS:
            for sample_name in ("all", "full_only"):
                summary = _block(_block(_block(block["directions"])[direction])[sample_name])
                lines.append(
                    f"| `{direction}` | {'todas' if sample_name == 'all' else 'completas'} | "
                    f"{summary['sessions']} | {_pct(summary['p_target'])} | "
                    f"{_pct(summary['p_stop'])} | {_pct(summary['p_time'])} | "
                    f"{_pct(summary['p_win'])} | {_bp(summary['e_gain'])} | "
                    f"{_bp(summary['e_loss'])} | {_bp(summary['ev_media'])} | "
                    f"{_bp(summary['ev_neto'])} |"
                )
        differences: list[object] = [
            _block(_block(_block(block["directions"])[direction])["all"])["ev_identity_difference"]
            for direction in DIRECTIONS
        ]
        sensitivity = _block(_block(_block(_block(block["directions"])["long"])["all"])["bp"])[
            "sensitivity_one_day"
        ]
        lines.extend(
            [
                "",
                f"- Identidad del EV (`p_win * E[G] - (1 - p_win) * E[P]` frente a la media de los "
                f"retornos) dentro de `{EV_IDENTITY_TOLERANCE}`: diferencias maximas "
                + ", ".join(_bp(value) + " bp" for value in differences)
                + ".",
                "- Sensibilidad declarada de #8 (**nunca fusionada** con el diferencial intradia): "
                + ", ".join(
                    f"`{name}` = {_bp(value)} bp" for name, value in _block(sensitivity).items()
                )
                + ".",
                "",
            ]
        )

    persistence = _block(report_payload(run)["persistence"])
    lines.extend(
        [
            "## Persistencia",
            "",
            f"- `{persistence['layer']}.{persistence['dataset']}` por la API del `Store` "
            "(`append` / `replace`), nunca Parquet a mano. `source = "
            f"`{persistence['source']}``, `series_id = `{persistence['series_id']}``.",
            f"- Grano: {persistence['grain']}. `as_of` = {persistence['as_of']}; "
            f"{persistence['published_at']}.",
            f"- Filas de esta ejecucion: **{persistence['sessions_persisted']}** "
            f"(`k = {persistence['k_sigma_persisted']}`). {persistence['supersede']}.",
            f"- Verificacion: {persistence['verification']}.",
            "",
            "## Limitaciones (declaradas, no escondidas)",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in run.limitations)
    lines.extend(["", "## Notas", ""])
    lines.extend(f"- {item}" for item in run.notes)
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Escritura y CLI
# ─────────────────────────────────────────────────────────────────────────────
def write_outputs(run: LabelsRun, *, store: Store, reports_dir: Path) -> LabelsOutputs:
    """Persiste ``derived.labels`` y escribe el informe JSON + Markdown (A8, A26, A32)."""
    records = [record_for(row, run=run, now=run.as_of) for row in run.rows]
    outcome = persist_rows(store, records)
    written = replace(run, write_outcome=outcome, stored_version=_stored_version(store))
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{REPORT_PREFIX}_{run.as_of.date().isoformat()}"
    json_path = reports_dir / f"{stem}.json"
    markdown_path = reports_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(report_payload(written), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(written), encoding="utf-8")
    return LabelsOutputs(
        outcome=outcome,
        sessions=len(records),
        json_path=json_path,
        markdown_path=markdown_path,
        run=written,
    )


def label_and_write(
    *,
    data_root: Path,
    reports_dir: Path,
    now: datetime,
    k_sigma: float = PERSISTED_K_SIGMA,
    entry_price_source: str = DEFAULT_ENTRY_PRICE_SOURCE,
) -> tuple[LabelsRun, LabelsOutputs]:
    """Ejecuta el etiquetado y escribe dataset e informe. Nada mas toca el disco."""
    store = Store(data_root)
    run = label_history(
        store=store, now=now, k_sigma=k_sigma, entry_price_source=entry_price_source
    )
    outputs = write_outputs(run, store=store, reports_dir=reports_dir)
    return (outputs.run if outputs.run is not None else run), outputs


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del etiquetado tri-barrera.

    Codigos de salida: ``0`` = informe y dataset escritos; ``2`` = falta un dato, la
    seleccion de #7 es `inconclusive` o el precio de entrada pedido no esta
    disponible ⇒ **no se escribe nada** y el motivo sale por ``stderr``.
    """
    parser = argparse.ArgumentParser(
        prog="cfdtrader.models.labels",
        description="Etiquetado tri-barrera (target / stop / time) para LONG y SHORT",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="raiz del almacen")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="directorio de informes (por defecto <data-root>/derived/reports)",
    )
    parser.add_argument("--settings", type=Path, default=None, help="ruta de settings.yaml")
    parser.add_argument("--now", default=None, help="instante de referencia ISO (tests)")
    parser.add_argument(
        "--entry-price-source",
        default=DEFAULT_ENTRY_PRICE_SOURCE,
        choices=sorted(ENTRY_PRICE_SOURCES),
        help="fuente del precio de entrada (registro declarado)",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=PERSISTED_K_SIGMA,
        help=f"multiplo de sigma de las barreras (por defecto {PERSISTED_K_SIGMA})",
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.settings)
    except ConfigurationError as error:
        print(f"no se puede leer la configuracion: {error}", file=sys.stderr)
        return 2

    data_root = Path(args.data_root) if args.data_root is not None else Path(settings.data.root)
    reports_dir = (
        Path(args.reports_dir)
        if args.reports_dir is not None
        else data_root / "derived" / "reports"
    )
    try:
        run, outputs = label_and_write(
            data_root=data_root,
            reports_dir=reports_dir,
            now=_parse_now(args.now),
            k_sigma=args.k,
            entry_price_source=args.entry_price_source,
        )
    except (ConfigurationError, LabelsError) as error:
        print(f"no se puede etiquetar el historico: {error}", file=sys.stderr)
        return 2

    logger.info(
        "etiquetas: {} sesiones (k = {}), orden {} intradia / {} respaldo, escritura {}",
        outputs.sessions,
        run.k_sigma,
        cast("dict[str, object]", run.sample["by_order_source"])[INTRADAY_ORDER_SOURCE],
        cast("dict[str, object]", run.sample["by_order_source"])[FALLBACK_ORDER_SOURCE],
        outputs.outcome,
    )
    return 0


def _parse_now(value: str | None) -> datetime:
    """Instante de referencia: el de verdad, o el que fije el test."""
    if value is None:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
