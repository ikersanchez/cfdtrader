"""Adaptador ``Store`` -> ``SessionInput``, corrida de los seis baselines e informe (#69).

Esta es la ultima milla del arnes de Fase 1: el motor de #13 es puro y no toca disco, asi que
aqui vive **quien lee la historia real**. Reparto de capas (nada de esto se reimplementa):

- ``data.store`` lee el almacen (``store.sql()``, **nunca** la lectura *point-in-time*: su
  semantica es «que sabiamos en T» y con un ``fetched_at`` de hoy y un ``at`` historico
  devuelve vacio; A5 lo demuestra con los dos recuentos, no lo afirma);
- ``analysis.drift`` define la **muestra limpia** (regla unica de #52: se importa, no se
  copia, y por eso este modulo no contiene sus literales);
- ``data.calendar`` da la ventana de la sesion (aqui **no** hay ninguna hora ET literal);
- ``backtest.costs`` declara el coste y el *slippage* (se importan, no se escriben);
- ``backtest.splits``/``engine``/``baselines``/``metrics`` deciden, simulan y agregan;
- este modulo **orquesta y publica**, sin decidir nada.

Tres fronteras que el informe declara y **no** cruza:

1. **El reloj no entra.** El instante es un parametro (``--as-of``, obligatorio para
   escribir): ninguna ruta de este modulo consulta la hora de la maquina. El informe se llama
   con la fecha **UTC** declarada.
2. **Los estados del *slippage* no se fusionan.** El supuesto pesimista de #64 viaja como
   ``assumed`` con ``is_measurement = False``, asi que ninguna operacion cierra el total y
   ``pnl_net_pct`` es ``null`` en todas: un valor no medido **nunca** se escribe como ``0``.
   Por eso las metricas netas de #15 **no se pueden publicar** y el bloque ``net_metrics`` lo
   declara en vez de rellenarse; su publicacion es #18 y exige medir el *slippage* (#62).
3. **La tabla es de coste declarado, no una validacion.** ``gate: "fail"`` y
   ``phase1_ready: false`` viajan tal cual: los numeros del arnes **no** son un veredicto
   sobre la estrategia.

Determinismo: el ``report_sha256`` se calcula reutilizando ``canonical_text`` de #13 sobre el
payload (ya JSON puro), y dos procesos con el mismo ``--as-of`` producen el mismo hash y
ficheros identicos byte a byte. Es una puerta, no un adorno.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Final, cast

import duckdb
import polars as pl
from loguru import logger

from cfdtrader.analysis.drift import clean_sample, clean_sample_cutoff, session_stale_open
from cfdtrader.backtest.baselines import (
    BASELINE_IDS,
    RANDOM_MATCHED,
    run_baseline,
    run_random_matched,
)
from cfdtrader.backtest.costs import (
    CostModel,
    SlippageParameter,
    declared_cost_model,
    declared_slippage_assumption,
)
from cfdtrader.backtest.engine import (
    EXIT_SESSION_CLOSE,
    EXIT_STOP,
    EXIT_TARGET,
    STATUS_NO_TRADE,
    STATUS_SKIPPED,
    STATUS_TRADED,
    BacktestRun,
    Bar,
    SessionInput,
    SessionOutcome,
    canonical_text,
)
from cfdtrader.backtest.splits import SplitPlan, walk_forward_splits
from cfdtrader.data.calendar import MarketCalendar, load_calendar
from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.store import Store, UnknownDatasetError

__all__ = [
    "ADAPTER_DOES_NOT_DO",
    "FOLLOW_UPS",
    "INTRADAY_INTERVAL",
    "NOTIONAL_USD",
    "PHASE1_PLAN",
    "PRICE_PROXY_OF",
    "RANDOM_MATCHED_FREQUENCY",
    "RANDOM_MATCHED_SEED",
    "REPORT_HASH_FORMAT",
    "REPORT_PREFIX",
    "SERIES_ID",
    "BacktestReport",
    "BacktestReportError",
    "BaselineOutcome",
    "DuplicateSessionError",
    "History",
    "InsufficientSampleError",
    "InvalidAsOfError",
    "LabelHorizonError",
    "MissingAsOfError",
    "MissingDatasetError",
    "MissingPriceError",
    "PlanParams",
    "Universe",
    "analyse",
    "build_inputs",
    "build_split_plan",
    "label_horizon_sequence",
    "load_history",
    "main",
    "render_markdown",
    "run_all_baselines",
]

#: Serie analizada: el subyacente del CFD. El CFD no tiene fuente propia (#50).
SERIES_ID: Final[str] = "^GSPC"

#: Instrumento del que los precios son **proxy** mientras no haya fuente del CFD (#50).
PRICE_PROXY_OF: Final[str] = "SPX500:CFD"

#: Intervalo del unico intradia real disponible (5 minutos de ``^GSPC``).
INTRADAY_INTERVAL: Final[str] = "5m"

#: Nocional **plano y declarado**: la tabla de `plan.md` §3.3 via #8. No se deriva del
#: capital, del riesgo, de ``R`` ni del apalancamiento (#27, #60): eso no se decide aqui.
NOTIONAL_USD: Final[Decimal] = Decimal("10000")

NOTIONAL_PROVENANCE: Final[str] = (
    "tabla declarada de `plan.md` §3.3 (nocional de referencia de #8); plano e ilustrativo, "
    "no derivado del capital ni de `R` (#27, #60)"
)

#: Prefijo del informe: ``phase1_backtest_<AAAA-MM-DD>.{json,md}``.
REPORT_PREFIX: Final[str] = "phase1_backtest"

#: Las dos constantes de ``random_matched`` son explicitas, sin valor por defecto (#14).
RANDOM_MATCHED_FREQUENCY: Final[Fraction] = Fraction(1, 2)
RANDOM_MATCHED_SEED: Final[int] = 42

#: Horizonte de etiqueta declarado: 0 en todas las muestras (hecho de #10: ``t1`` cae en la
#: propia sesion). No es un parametro a inventar aqui (A15).
PHASE1_LABEL_HORIZON: Final[int] = 0

#: Motivos de salida, en orden estable, importados de #13 (A23).
EXIT_REASONS: Final[tuple[str, ...]] = (EXIT_TARGET, EXIT_STOP, EXIT_SESSION_CLOSE)

#: Estados del motor (#13), importados para no duplicar literales.
STATUSES: Final[tuple[str, ...]] = (STATUS_TRADED, STATUS_NO_TRADE, STATUS_SKIPPED)

#: Columns obligatorias del payload diario que forman el OHLC de una sesion (A9).
OHLC_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close")

#: Motivos de exclusion por sesion (A6): una sesion etiquetada que no llega al universo
#: **nunca** se rellena con un precio inventado, se declara con su motivo.
EXCLUSION_REASONS: Final[tuple[str, ...]] = (
    "no_daily_row",
    "missing_ohlc",
    "stale_open",
    "no_previous_session",
    "before_clean_cutoff",
)

EXCLUSION_TEXTS: Final[dict[str, str]] = {
    "no_daily_row": (
        "la sesion esta etiquetada en `derived.labels` pero no hay fila en `raw.market_daily`: "
        "no se inventa un precio (A6)"
    ),
    "missing_ohlc": (
        "la fila diaria existe pero le falta alguna de las cuatro columnas OHLC: no se rellena "
        "con un precio inventado (A6, A9)"
    ),
    "stale_open": (
        "la sesion quedo fuera de la muestra limpia por el `open` repetido de #52; la regla la "
        "define `analysis.drift` y aqui solo se aplica (A7, A8)"
    ),
    "no_previous_session": (
        "es la primera fila del almacen: sin sesion previa no hay `open_stale` que evaluar y la "
        "regla limpia de #52 la excluye; no se rellena con nada (A7, A8)"
    ),
    "before_clean_cutoff": (
        "la sesion es anterior al corte limpio de #52: queda fuera de la muestra limpia y por "
        "tanto del universo (A7, A8)"
    ),
}

#: Formato estable del ``report_sha256`` (A24).
REPORT_HASH_FORMAT: Final[str] = (
    "sha256(UTF-8) de cfdtrader.backtest.engine.canonical_text(payload), es decir "
    "json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False), **sin** la "
    "clave report_sha256 (un informe no se hashea a si mismo). El payload ya viaja como tipos "
    "JSON puros: Decimal como cadena decimal exacta (format(d, 'f')), fecha/hora en ISO-8601 y "
    "Fraction como 'numerador/denominador'; por eso #13 no necesita ningun tipo nuevo y **no** "
    "se escribe un segundo serializador ad hoc. float va via repr y nan/inf estan prohibidos"
)

#: Que **no** hace este modulo, legible por maquina (A33). Cada frontera con su issue.
ADAPTER_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_reimplementa_baselines",
        "issue": "#14",
        "statement": (
            "no reimplementa los seis baselines ni su API: los consume (`BASELINE_IDS`, "
            "`run_baseline`, `run_random_matched`) y no añade un septimo"
        ),
    },
    {
        "id": "no_publica_metricas_netas",
        "issue": "#15",
        "statement": (
            "no publica las metricas netas de #15 (Sharpe/Sortino, EV, drawdown, intervalos): "
            "`calculate_metrics` rechaza un `traded` sin `pnl_net_pct` y el supuesto de #64 no "
            "es una medicion; el bloque `net_metrics` lo declara no calculable"
        ),
    },
    {
        "id": "no_es_el_informe_de_fase_1",
        "issue": "#18",
        "statement": (
            "no es el informe de Fase 1 ni su puerta de salida: publica la tabla de coste "
            "declarado de los baselines, no un veredicto sobre la estrategia"
        ),
    },
    {
        "id": "no_decide_el_gate_ni_el_sizing",
        "issue": "#27",
        "statement": (
            "no implementa el gate de decision ni el *sizing*: por eso el camino intradia es "
            "inerte (ningun decididor declara `stop_px`/`target_px`) y el nocional es un plano"
        ),
    },
    {
        "id": "no_deriva_el_nocional",
        "issue": "#60",
        "statement": (
            "no deriva el tamaño de la posicion del capital, del riesgo, de `R` ni del "
            "apalancamiento: el nocional es la constante declarada de #8"
        ),
    },
    {
        "id": "no_tiene_intradia_real",
        "issue": "#50",
        "statement": (
            "no trae la fuente intradia real ni el `bid`/`ask` del SPX500:CFD: los precios son "
            "proxies de `^GSPC` y el diferencial es el declarado de #11"
        ),
    },
    {
        "id": "no_verifica_el_corte_de_financiacion",
        "issue": "#59",
        "statement": (
            "no verifica el corte de financiacion ni elige broker: `financing_cut` viaja como "
            "`None` y el informe declara que sigue sin verificar (asumir una hora esta prohibido)"
        ),
    },
    {
        "id": "no_mide_el_slippage",
        "issue": "#62",
        "statement": (
            "no mide el *slippage* real (10-15 ejecuciones en la apertura): publica el supuesto "
            "declarado de #64 sin cobrarlo, y por eso `pnl_net_pct` es `null`"
        ),
    },
    {
        "id": "no_arregla_la_sigma",
        "issue": "#63",
        "statement": (
            "no corrige el *look-ahead* heredado de la seleccion de candidato de #7: lo declara "
            "como limite"
        ),
    },
    {
        "id": "no_modela_el_diferencial_por_tramo",
        "issue": "#66",
        "statement": (
            "no modela el diferencial por tramo de sesion y por tamaño, ni el *gap* a traves "
            "del *stop*: cobra el diferencial declarado constante y simetrico"
        ),
    },
    {
        "id": "no_usa_cpcv",
        "issue": "#67",
        "statement": (
            "no implementa CPCV (el esquema donde la purga y el embargo trabajan de verdad): "
            "recorre el plan *walk-forward* puro de #12 con `label_horizon = 0`"
        ),
    },
    {
        "id": "no_reserva_holdout",
        "issue": "#68",
        "statement": (
            "no reserva ni toca el *holdout* final intocable: evalua las sesiones que le entrega "
            "el plan de #12"
        ),
    },
    {
        "id": "no_es_el_liston_b",
        "issue": "#70",
        "statement": (
            "no construye el liston B («siempre largo» aguantando la posicion con financiacion "
            "*overnight*, `close->close`): #13 no produce posiciones que crucen la noche"
        ),
    },
    {
        "id": "no_regulariza_el_backlog",
        "issue": "#67/#68/#69",
        "statement": (
            "no documenta en `_docs/tasks.md` que #67, #68 y #69 viven fuera del backlog: es una "
            "tarea de documentacion aparte, declarada aqui y **no** resuelta en esta tarea"
        ),
    },
)

#: Seguimientos abiertos que este modulo deja declarados (A33).
FOLLOW_UPS: Final[tuple[dict[str, str], ...]] = (
    {
        "issue": "#18",
        "topic": "informe de Fase 1 y su puerta de salida",
        "why": "publicara la tabla real cuando el *slippage* este medido y `R` decidido",
    },
    {
        "issue": "#27",
        "topic": "gate de decision, umbrales y *sizing*",
        "why": "de ahi sale el nocional real y las barreras que activan el camino intradia",
    },
    {
        "issue": "#50",
        "topic": "intradia real y `bid`/`ask` del SPX500:CFD",
        "why": "los precios seguiran siendo proxies hasta que exista la fuente del CFD",
    },
    {
        "issue": "#59",
        "topic": "broker definitivo y corte de financiacion",
        "why": "de ahi salen la comision real, el diferencial real y la hora de corte",
    },
    {
        "issue": "#60",
        "topic": "umbrales y tamaño de `R`",
        "why": "sin `R` decidido el supuesto de *slippage* no se puede cobrar",
    },
    {
        "issue": "#62",
        "topic": "medir el *slippage* real",
        "why": "es lo que convierte el supuesto de #64 en una medicion y desbloquea `pnl_net_pct`",
    },
    {
        "issue": "#63",
        "topic": "sigma con informacion del futuro",
        "why": "heredada de la seleccion de candidato de #7; el informe la declara sin corregirla",
    },
    {
        "issue": "#66",
        "topic": "diferencial por tramo y por tamaño",
        "why": "hoy se cobra el diferencial declarado constante y simetrico",
    },
    {
        "issue": "#67",
        "topic": "CPCV",
        "why": "el esquema donde la purga y el embargo hacen trabajo real",
    },
    {
        "issue": "#68",
        "topic": "*holdout* final intocable",
        "why": "`plan.md` §11.4 y §21 pregunta 10",
    },
    {
        "issue": "#70",
        "topic": "liston B",
        "why": "«siempre largo» con financiacion *overnight*: #13 no cruza la noche",
    },
    {
        "issue": "#67/#68/#69",
        "topic": "regularizacion del backlog",
        "why": "los tres viven solo como issues; regularizarlos es una tarea de documentacion",
    },
)

#: Limitaciones que el informe publica (A19-A21, A28). No se esconden.
LIMITATIONS: Final[tuple[str, ...]] = (
    '**La puerta de Fase 0 esta en `fail`** (`gate: "fail"`, `phase1_ready: false`) y **los '
    "numeros de este arnes no son una validacion de la estrategia**: la tabla es de recuentos y "
    'de retorno de **coste declarado**, y viaja etiquetada (`basis: "declared_cost"`, '
    "`is_validation: false`).",
    "**El *slippage* es un supuesto pesimista declarado** (#64), **no** una medicion: mientras "
    "no se mida (#62) y `R` no este decidido (#60), el total no se cierra y `pnl_net_pct` es "
    "`null` en todas las operaciones. Ningun valor no medido se escribe como `0`.",
    "**Las metricas netas de #15 no se publican** porque `calculate_metrics` rechaza un "
    "`traded` sin `pnl_net_pct` (regla «`null != 0`»): el bloque `net_metrics` lo declara "
    "`not_computable` con su seguimiento (#62, #60) en vez de rellenarse.",
    "**El corte de financiacion sigue sin verificar** (#8/#59): `financing_cut` viaja como "
    "`None` y asumir una hora de corte fija esta prohibido; la ventana de sesion la da el "
    "`MarketCalendar`, nunca una hora escrita a mano.",
    "**Los precios son *proxies*** de `^GSPC` y no del `SPX500:CFD` (#50), y la muestra de "
    "intradia real es corta: la mayoria de las sesiones usan el respaldo diario (#10, #57).",
    "**La sigma con *look-ahead*** heredada de la seleccion de candidato de #7 sigue sin "
    "arreglar (#63): las barreras de #10 pueden estar informadas por el futuro.",
    "**El *gap* a traves del *stop* no se modela** y el diferencial se cobra constante y "
    "simetrico: el ensanchamiento por tramo y por tamaño es #66.",
    "**La purga y el embargo son no-ops estructurales** con `label_horizon = 0` (#12): se "
    "publican con sus numeros en vez de presentarse como un filtro activo. El esquema donde si "
    "trabajan es CPCV (#67) y el *holdout* final intocable es #68.",
    '**El overlay LLM va deshabilitado por declaracion** (`llm_overlay: "disabled"`) y no hay '
    'scheduler (`scheduler: "none"`): el LLM no calcula ni decide en el camino critico '
    "(`tech_stack.md` §4.9).",
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados (A30): nunca un resultado silencioso
# ─────────────────────────────────────────────────────────────────────────────
class BacktestReportError(Exception):
    """Raiz de los errores del adaptador y del informe de Fase 1 (#69)."""


class MissingDatasetError(BacktestReportError):
    """Falta un dataset del almacen (``derived.labels`` o ``raw.market_daily``)."""


class InsufficientSampleError(BacktestReportError):
    """La muestra limpia resultante no alcanza el minimo de la regla compartida de #52."""


class DuplicateSessionError(BacktestReportError):
    """La secuencia de sesiones no es estrictamente creciente o trae duplicados (A12)."""


class LabelHorizonError(BacktestReportError):
    """El horizonte de etiqueta no es 0 (o su secuencia no casa con la muestra) (A15)."""


class MissingPriceError(BacktestReportError):
    """Un ``SessionInput`` sin ``open_px`` no se puede simular (A9, A30)."""


class MissingAsOfError(BacktestReportError):
    """Escribir el informe exige un instante declarado: el modulo no lee el reloj (A2, A3)."""


class InvalidAsOfError(BacktestReportError):
    """El instante declarado no es un ISO-8601 valido (A3)."""


# ─────────────────────────────────────────────────────────────────────────────
# Plan declarado (A14, A15)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PlanParams:
    """Los parametros **declarados** del plan *walk-forward* de la Fase 1 (A14).

    No se inventan: el horizonte es 0 (hecho de #10) y ``max_train_size = None`` significa
    expansivo. Cambiarlos cambia el ``plan_sha256`` y, con el, el ``report_sha256``.
    """

    n_splits: int = 10
    test_size: int = 50
    embargo_sessions: int = 5
    max_train_size: int | None = None
    label_horizon: int | tuple[int, ...] = PHASE1_LABEL_HORIZON


#: El plan de la Fase 1, tal cual lo declara la issue (A14).
PHASE1_PLAN: Final[PlanParams] = PlanParams()


def label_horizon_sequence(
    *, n_sessions: int, horizon: int | Sequence[int] = PHASE1_LABEL_HORIZON
) -> tuple[int, ...]:
    """Expande y **valida** el horizonte de etiqueta antes de tocar #12 (A15).

    Acepta un escalar (que se repite ``n_sessions`` veces) o una secuencia explicita, que
    entonces tiene que tener exactamente esa longitud. Cualquier valor distinto de 0 se
    rechaza con error tipado: el horizonte 0 es un hecho declarado de #10 (``t1`` cae en la
    propia sesion) y este adaptador no lo inventa.
    """
    if isinstance(horizon, int):
        values: tuple[int, ...] = (horizon,) * n_sessions
    else:
        values = tuple(horizon)
        if len(values) != n_sessions:
            raise LabelHorizonError(
                f"label_horizon trae {len(values)} entradas y la muestra tiene {n_sessions}: el "
                "horizonte de #10 es 0 en todas las muestras y no se inventa otro (A15)"
            )
    non_zero = sorted({value for value in values if value != 0})
    if non_zero:
        raise LabelHorizonError(
            f"label_horizon = {non_zero} no es admisible: el horizonte declarado de #10 es 0 "
            "(`t1` cae en la propia sesion) y el adaptador no acepta otro (A15)"
        )
    return values


# ─────────────────────────────────────────────────────────────────────────────
# Lectura de la historia (A5, A6, A7)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class History:
    """Lo que el adaptador lee del almacen, ya resuelto a sesiones ET.

    ``daily`` viene de ``session_stale_open`` (añade ``prev_close``, ``session`` ET y
    ``open_stale``) ordenado por sesion; ``labels`` trae una sesion por fila; ``intraday``
    trae las barras de 5 minutos con su sesion ET. Todo se lee con el motor de consulta del
    almacen.
    """

    series_id: str
    daily: pl.DataFrame
    labels: pl.DataFrame
    intraday: pl.DataFrame


def _literal(value: str) -> str:
    """Literal SQL seguro: los identificadores vienen del registro validado."""
    return "'" + value.replace("'", "''") + "'"


def _query(store: Store, query: str, *, dataset: str) -> pl.DataFrame:
    """Ejecuta una consulta del almacen traduciendo su ausencia a error tipado (A30)."""
    try:
        return store.sql(query)
    except (UnknownDatasetError, duckdb.Error) as error:
        raise MissingDatasetError(
            f"no se puede leer {dataset} en {store.root}: {error}. Ejecuta antes la ingesta o la "
            "tarea que produce ese dataset (A30)"
        ) from error


def load_history(store: Store, *, series_id: str = SERIES_ID) -> History:
    """Lee las sesiones diarias, las etiquetas y el intradia de esa serie (A5).

    Se usa el motor de consulta del almacen (**no** la lectura *point-in-time*): se quiere
    «que datos existen», que es el estado vigente del dataset, y no «que sabiamos en T». El
    test de A5 publica los dos recuentos sobre el mismo almacen para demostrarlo.

    Faltan ``derived.labels`` o ``raw.market_daily`` ⇒ error tipado. Falta
    ``raw.market_intraday`` ⇒ **no** es un error: todas las sesiones caen al respaldo diario,
    que es un resultado legitimo y declarado (A10).
    """
    daily_query = (
        "SELECT as_of, open, high, low, close FROM raw.market_daily "  # noqa: S608
        f"WHERE series_id = {_literal(series_id)} ORDER BY as_of"
    )
    daily_raw = _query(store, daily_query, dataset="raw.market_daily")
    if daily_raw.height == 0:
        raise MissingDatasetError(
            f"raw.market_daily no tiene ninguna sesion de {series_id!r}: el universo del "
            "adaptador se construye sobre el (A30)"
        )
    daily = session_stale_open(daily_raw).sort("session")

    labels_query = (
        "SELECT session FROM derived.labels "  # noqa: S608
        f"WHERE series_id = {_literal(series_id)} ORDER BY session"
    )
    labels = _query(store, labels_query, dataset="derived.labels")
    if labels.height == 0:
        raise MissingDatasetError(
            f"derived.labels no tiene ninguna sesion de {series_id!r}: sin etiquetas no hay "
            "universo que simular (A30)"
        )

    intraday_query = (
        "SELECT as_of, high, low FROM raw.market_intraday "  # noqa: S608
        f"WHERE series_id = {_literal(series_id)} AND interval = {_literal(INTRADAY_INTERVAL)} "
        "AND high IS NOT NULL AND low IS NOT NULL ORDER BY as_of"
    )
    try:
        intraday = store.sql(intraday_query)
    except (UnknownDatasetError, duckdb.Error):
        logger.warning(
            "no hay intradia de {} en {}: todas las sesiones usaran el respaldo diario (A10)",
            series_id,
            store.root,
        )
        intraday = pl.DataFrame(
            schema={
                "as_of": pl.Datetime("us", "UTC"),
                "high": pl.Float64(),
                "low": pl.Float64(),
            }
        )
    intraday = intraday.with_columns(
        pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("session")
    )
    return History(series_id=series_id, daily=daily, labels=labels, intraday=intraday)


# ─────────────────────────────────────────────────────────────────────────────
# Universo: derived.labels ∩ muestra limpia ∩ raw.market_daily (A6-A13)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Universe:
    """El universo del arnes: los ``SessionInput`` y los recuentos que los justifican.

    ``excluded`` lleva **un motivo por sesion** etiquetada que no llego al universo: una sesion
    sin OHLC completo se excluye declarandolo, nunca se rellena con un precio inventado (A6).
    """

    series_id: str
    inputs: tuple[SessionInput, ...]
    labelled_sessions: int
    labelled_clean_sessions: int
    clean_from: date | None
    clean_sessions: int
    raw_daily_rows: int
    rows_with_previous_session: int
    sessions_since_cutoff: int
    excluded_by_clean_rule: int
    stale_open_in_window: int
    excluded: tuple[dict[str, str], ...]
    intraday_sessions: int
    daily_fallback_sessions: int

    @property
    def first_session(self) -> date | None:
        """Primera sesion del universo, o ``None`` si esta vacio."""
        return self.inputs[0].session if self.inputs else None

    @property
    def last_session(self) -> date | None:
        """Ultima sesion del universo, o ``None`` si esta vacio."""
        return self.inputs[-1].session if self.inputs else None

    @property
    def no_forecast_sessions(self) -> int:
        """Sesiones limpias **sin** etiqueta: el calentamiento del *forecast* de #7 (A7)."""
        return self.clean_sessions - self.labelled_clean_sessions


def _require_unique_sessions(frame: pl.DataFrame, *, source: str) -> None:
    """Una sesion repetida en el origen es un error tipado **antes** del plan (A12)."""
    duplicates = frame.group_by("session").len().filter(pl.col("len") > 1).sort("session")
    if duplicates.height:
        first = duplicates.get_column("session").to_list()[0]
        raise DuplicateSessionError(
            f"{source}: hay {duplicates.height} sesiones repetidas; la primera es {first}. #12 y "
            "#13 exigen una secuencia estrictamente creciente y sin duplicados (A12)"
        )


def _require_strictly_increasing(sessions: Sequence[date], *, source: str) -> None:
    """La secuencia de sesiones tiene que ser creciente y sin repeticiones (A12)."""
    for index in range(1, len(sessions)):
        if sessions[index] <= sessions[index - 1]:
            raise DuplicateSessionError(
                f"{source}: la sesion {sessions[index].isoformat()} no es estrictamente posterior "
                f"a {sessions[index - 1].isoformat()}; el plan de #12 no se construye sobre una "
                "secuencia invalida (A12)"
            )


def _group_intraday(
    frame: pl.DataFrame,
) -> dict[date, list[tuple[datetime, float, float]]]:
    """Agrupa las barras por sesion ET, en orden ascendente, sin nulos (A10)."""
    grouped: dict[date, list[tuple[datetime, float, float]]] = {}
    if frame.height == 0:
        return grouped
    for row in frame.iter_rows(named=True):
        session = row["session"]
        moment = row["as_of"]
        high = row["high"]
        low = row["low"]
        if session is None or moment is None or high is None or low is None:
            continue
        grouped.setdefault(cast("date", session), []).append(
            (cast("datetime", moment), float(high), float(low))
        )
    return grouped


def _bars_for(
    session: date,
    grouped: Mapping[date, list[tuple[datetime, float, float]]],
    calendar: MarketCalendar,
) -> tuple[Bar, ...] | None:
    """El camino intradia de esa sesion, o ``None`` si no hay respaldo real (A10).

    La ventana sale **solo** del calendario (``open_utc``/``close_utc``): aqui no hay ninguna
    hora ET literal. Una barra fuera de la ventana no entra.
    """
    rows = grouped.get(session)
    if not rows:
        return None
    info = calendar.session(session)
    if info.open_utc is None or info.close_utc is None:
        return None
    bars = tuple(
        Bar(high_px=high, low_px=low)
        for moment, high, low in rows
        if info.open_utc <= moment <= info.close_utc
    )
    return bars or None


def _ohlc_complete(row: Mapping[str, object]) -> bool:
    """``True`` si las cuatro columnas OHLC tienen valor en la fila diaria (A9)."""
    return all(row.get(column) is not None for column in OHLC_COLUMNS)


def build_inputs(history: History, *, calendar: MarketCalendar) -> Universe:
    """Construye el universo declarado y contado a partir de la historia (A6, A9-A13).

    El universo es ``derived.labels`` **∩** muestra limpia de la regla compartida de #52
    **∩** ``raw.market_daily`` con OHLC completo. La regla de muestra limpia se **importa**
    (``session_stale_open``, ``clean_sample_cutoff``, ``clean_sample``): no se reimplementa y
    sus constantes no se copian aqui.

    Cada sesion etiquetada que no llega al universo se declara con su motivo; ninguna se
    descarta en silencio y ninguna se rellena con un precio inventado.
    """
    daily = history.daily
    labels = history.labels
    _require_unique_sessions(daily, source="raw.market_daily")
    _require_unique_sessions(labels, source="derived.labels")

    cutoff = clean_sample_cutoff(daily)
    clean = clean_sample(daily, cutoff=cutoff)
    if clean is None:
        raise InsufficientSampleError(
            "la muestra limpia de #52 no alcanza el minimo declarado (o no hay corte limpio): "
            "decidir con menos sesiones que eso no es decidir. El corte y la muestra salen de "
            "`cfdtrader.analysis.drift` (A8, A30)"
        )

    daily_by_session: dict[date, Mapping[str, object]] = {}
    for row in daily.iter_rows(named=True):
        session = row["session"]
        if session is None:
            continue
        daily_by_session[cast("date", session)] = cast("Mapping[str, object]", row)
    clean_sessions = {cast("date", value) for value in clean.get_column("session").to_list()}

    grouped = _group_intraday(history.intraday)
    inputs: list[SessionInput] = []
    excluded: list[dict[str, str]] = []
    labelled_clean = 0
    intraday_sessions = 0
    for session in cast("list[date]", labels.get_column("session").to_list()):
        row = daily_by_session.get(session)
        reason: str | None = None
        if row is None:
            reason = "no_daily_row"
        elif not _ohlc_complete(row):
            reason = "missing_ohlc"
        elif session not in clean_sessions:
            if cutoff is not None and session < cutoff:
                reason = "before_clean_cutoff"
            elif row.get("open_stale") is None:
                reason = "no_previous_session"
            else:
                reason = "stale_open"
        if reason is not None:
            excluded.append(
                {
                    "session": session.isoformat(),
                    "reason": reason,
                    "detail": EXCLUSION_TEXTS[reason],
                }
            )
            continue
        labelled_clean += 1
        bars = _bars_for(session, grouped, calendar)
        if bars is not None:
            intraday_sessions += 1
        inputs.append(
            SessionInput(
                session=session,
                open_px=None if row is None else cast("float", row["open"]),
                high_px=None if row is None else cast("float", row["high"]),
                low_px=None if row is None else cast("float", row["low"]),
                close_px=None if row is None else cast("float", row["close"]),
                context=None,
                bars=bars,
            )
        )

    sessions = [item.session for item in inputs]
    _require_strictly_increasing(sessions, source="el universo de SessionInput")

    window = daily.filter(pl.col("session") >= cutoff) if cutoff is not None else daily
    return Universe(
        series_id=history.series_id,
        inputs=tuple(inputs),
        labelled_sessions=labels.height,
        labelled_clean_sessions=labelled_clean,
        clean_from=cutoff,
        clean_sessions=clean.height,
        raw_daily_rows=daily.height,
        rows_with_previous_session=daily.filter(pl.col("prev_close").is_not_null()).height,
        sessions_since_cutoff=window.height,
        excluded_by_clean_rule=window.height - clean.height,
        stale_open_in_window=window.filter(pl.col("open_stale").fill_null(False)).height,
        excluded=tuple(excluded),
        intraday_sessions=intraday_sessions,
        daily_fallback_sessions=len(inputs) - intraday_sessions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plan y corrida (A14-A18, A22, A23)
# ─────────────────────────────────────────────────────────────────────────────
def build_split_plan(
    inputs: Sequence[SessionInput], *, params: PlanParams = PHASE1_PLAN
) -> SplitPlan:
    """Construye el plan declarado con ``walk_forward_splits`` de #12 (A14, A15).

    Valida antes de llamar: secuencia estrictamente creciente y sin duplicados (A12), precios
    presentes (A30) y horizonte 0 (A15). Despues **no captura nada**: si #12 rechaza el plan,
    su ``SplitsError``/``InsufficientSessionsError`` llega al llamante sin envolver (A30).
    """
    items = tuple(inputs)
    sessions = [item.session for item in items]
    _require_strictly_increasing(sessions, source="la secuencia de SessionInput")
    for item in items:
        if item.open_px is None:
            raise MissingPriceError(
                f"{item.session.isoformat()}: un SessionInput sin `open_px` no se puede simular; "
                "el universo del adaptador exige OHLC completo (A9, A30)"
            )
    horizon = label_horizon_sequence(n_sessions=len(sessions), horizon=params.label_horizon)
    return walk_forward_splits(
        sessions,
        label_horizon=horizon,
        n_splits=params.n_splits,
        test_size=params.test_size,
        embargo_sessions=params.embargo_sessions,
        max_train_size=params.max_train_size,
    )


@dataclass(frozen=True, slots=True)
class BaselineOutcome:
    """Una corrida del motor para un baseline, con las constantes que la definen (A16)."""

    baseline: str
    run: BacktestRun
    frequency: Fraction | None
    seed: int | None


def run_all_baselines(
    inputs: Sequence[SessionInput],
    *,
    split_plan: SplitPlan,
    cost_model: CostModel,
    slippage: SlippageParameter,
    notional_usd: Decimal = NOTIONAL_USD,
    baselines: Sequence[str] = BASELINE_IDS,
    frequency: Fraction = RANDOM_MATCHED_FREQUENCY,
    seed: int = RANDOM_MATCHED_SEED,
) -> tuple[BaselineOutcome, ...]:
    """Corre los seis baselines de #14 sobre la historia real, en su orden (A16, A17).

    Los cinco deterministas van por ``run_baseline``; ``random_matched`` por
    ``run_random_matched`` con ``frequency`` y ``seed`` **explicitas** (su API no las
    acepta por defecto). El nocional es el plano declarado, identico en los seis. El corte de
    financiacion se pasa como ``None`` **tal cual**: este modulo no asume ninguna hora (A29).

    No captura nada: un ``baseline`` desconocido lo rechaza #14 con su ``BaselinesError``.
    """
    outcomes: list[BaselineOutcome] = []
    for baseline in baselines:
        if baseline == RANDOM_MATCHED:
            run = run_random_matched(
                inputs,
                split_plan=split_plan,
                cost_model=cost_model,
                slippage=slippage,
                notional_usd=notional_usd,
                frequency=frequency,
                seed=seed,
                financing_cut=None,
            )
            outcomes.append(
                BaselineOutcome(baseline=baseline, run=run, frequency=frequency, seed=seed)
            )
            continue
        run = run_baseline(
            inputs,
            split_plan=split_plan,
            cost_model=cost_model,
            slippage=slippage,
            notional_usd=notional_usd,
            baseline=baseline,
            financing_cut=None,
        )
        outcomes.append(BaselineOutcome(baseline=baseline, run=run, frequency=None, seed=None))
    return tuple(outcomes)


# ─────────────────────────────────────────────────────────────────────────────
# Agregacion declarada de la tabla (A20, A22, A23)
# ─────────────────────────────────────────────────────────────────────────────
def _fraction_text(value: Fraction | None) -> str | None:
    """``Fraction`` como ``numerador/denominador``: texto exacto, sin float de por medio."""
    if value is None:
        return None
    return f"{value.numerator}/{value.denominator}"


def _exit_counts(sessions: Sequence[SessionOutcome]) -> dict[str, int]:
    """Recuento por motivo de salida, con orden estable (A20, A23)."""
    raw: dict[str, int] = {}
    for outcome in sessions:
        if outcome.exit_reason is None:
            continue
        raw[outcome.exit_reason] = raw.get(outcome.exit_reason, 0) + 1
    counts: dict[str, int] = {reason: raw.get(reason, 0) for reason in EXIT_REASONS}
    for reason in sorted(set(raw) - set(EXIT_REASONS)):
        counts[reason] = raw[reason]
    return counts


def _summary(values: Sequence[float]) -> dict[str, object]:
    """Media, mediana y suma de una serie de P&L declarados (A20).

    Sin operaciones no hay valor: los tres son ``null`` con su motivo, **nunca** ``0``. La
    suma usa ``math.fsum`` (exacta y determinista) para que el hash no dependa del orden.
    """
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "sum": None,
            "reason": "no hay operaciones: un valor no medido es null, nunca 0",
        }
    return {
        "n": len(values),
        "mean": math.fsum(values) / len(values),
        "median": float(statistics.median(values)),
        "sum": math.fsum(values),
    }


def _conservation(*, run: BacktestRun, evaluated: int) -> dict[str, object]:
    """Identidades de conservacion publicadas y verificadas (A22).

    ``n_sessions`` es el numero de sesiones **evaluadas** (las de *test* del plan), que es el
    sujeto de ``n_sessions == traded + no_trade + skipped == n_test``; ``n_inputs`` es el
    tamaño del universo, que es lo que el motor llama ``run.n_sessions``.
    """
    total = run.traded + run.no_trade + run.skipped
    return {
        "n_sessions": evaluated,
        "traded_plus_no_trade_plus_skipped": total,
        "equals_n_test": total == evaluated,
        "n_inputs": run.n_sessions,
        "n_test_plus_not_in_any_test": evaluated + run.not_in_any_test,
        "inputs_identity_holds": run.n_sessions == evaluated + run.not_in_any_test,
        "not_in_any_test": run.not_in_any_test,
        "holds": total == evaluated and run.n_sessions == evaluated + run.not_in_any_test,
        "statement": (
            "n_sessions == traded + no_trade + skipped == n_test y "
            "len(inputs) == n_test + not_in_any_test"
        ),
    }


def _baseline_row(outcome: BaselineOutcome, *, n_inputs: int) -> dict[str, object]:
    """Una fila de la tabla comparativa, con aritmetica **declarada** (A20, A23)."""
    run = outcome.run
    sessions = [session for fold in run.folds for session in fold.sessions]
    traded = [session for session in sessions if session.status == STATUS_TRADED]
    evaluated = len(sessions)
    declared = [
        session.pnl_declared_pct for session in traded if session.pnl_declared_pct is not None
    ]
    gross = [session.gross_pct for session in traded if session.gross_pct is not None]
    net_null = sum(1 for session in traded if session.pnl_net_pct is None)
    return {
        "baseline": outcome.baseline,
        "n_sessions": evaluated,
        "n_test": evaluated,
        "n_inputs": n_inputs,
        "not_in_any_test": run.not_in_any_test,
        "traded": run.traded,
        "no_trade": run.no_trade,
        "skipped": run.skipped,
        "trade_rate": (run.traded / evaluated) if evaluated else None,
        "exit_reason_counts": _exit_counts(sessions),
        "exit_session_equals_entry": all(
            session.exit_session == session.entry_session for session in traded
        ),
        "pnl_declared_pct": _summary(declared),
        "gross_pct": _summary(gross),
        "pnl_net_pct_null_trades": net_null,
        "pnl_net_reason": next(
            (session.pnl_net_reason for session in traded if session.pnl_net_reason), None
        ),
        "conservation": _conservation(run=run, evaluated=evaluated),
        "frequency": _fraction_text(outcome.frequency),
        "seed": outcome.seed,
        "run_sha256": run.run_sha256,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Serializacion (A24)
# ─────────────────────────────────────────────────────────────────────────────
def _jsonable(value: object, *, where: str) -> object:
    """Traduce un valor a tipo JSON puro, o falla con error tipado (A24).

    ``Decimal`` viaja como cadena decimal **exacta** (``format(d, 'f')``), las fechas como
    ISO-8601, los ``Fraction`` como texto y los enums por su valor. Un ``float`` no finito es
    un error: el JSON nunca lleva ``nan`` ni ``inf``, y **no** se escriben como ``0``.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BacktestReportError(f"{where}: no se publica `nan` ni `inf` en el informe (A24)")
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Fraction):
        return _fraction_text(value)
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        return {str(key): _jsonable(item, where=f"{where}.{key}") for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        sequence = cast("Sequence[object]", value)
        return [_jsonable(item, where=f"{where}[{index}]") for index, item in enumerate(sequence)]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise BacktestReportError(
        f"{where}: el informe solo admite tipos JSON, Decimal, Fraction, Enum y fechas; llego "
        f"{type(value).__name__} (A24)"
    )


def _digest(payload: Mapping[str, object]) -> str:
    """``report_sha256``: el sha256 del texto canonico de #13, sin la clave del hash (A24)."""
    return hashlib.sha256(canonical_text(payload).encode("utf-8")).hexdigest()


def _date_text(value: date | None) -> str | None:
    """Una sesion se publica en ISO-8601; ``None`` sigue siendo ``None``."""
    return None if value is None else value.isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# El informe (A2, A18-A28)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class BacktestReport:
    """El informe de la corrida: payload canonico, hash, y los objetos que lo produjeron.

    ``payload`` es el texto que se hashea (segun ``REPORT_HASH_FORMAT``) y **no** incluye
    ``report_sha256``: un informe no se hashea a si mismo. Los objetos (plan, corridas, modelo
    de coste, *slippage*) viajan al lado para que se puedan comprobar por igualdad sin
    reconstruirlos desde el JSON.
    """

    as_of: datetime
    report_date: date
    payload: dict[str, object]
    report_sha256: str
    cost_model: CostModel
    slippage: SlippageParameter
    split_plan: SplitPlan
    universe: Universe
    outcomes: tuple[BaselineOutcome, ...]

    @property
    def report_stem(self) -> str:
        """Nombre base del informe: ``phase1_backtest_<AAAA-MM-DD>``."""
        return f"{REPORT_PREFIX}_{self.report_date.isoformat()}"

    def json_text(self) -> str:
        """JSON determinista: mismas entradas y mismo ``as_of`` ⇒ mismo texto byte a byte."""
        published: dict[str, object] = {**self.payload, "report_sha256": self.report_sha256}
        return json.dumps(published, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    def write(self, directory: Path) -> tuple[Path, Path]:
        """Escribe el informe en JSON y Markdown. Solo escribe quien lo pida (el CLI)."""
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{self.report_stem}.json"
        markdown_path = directory / f"{self.report_stem}.md"
        json_path.write_text(self.json_text(), encoding="utf-8")
        markdown_path.write_text(render_markdown(self), encoding="utf-8")
        return json_path, markdown_path


def _universe_payload(universe: Universe) -> dict[str, object]:
    """El universo declarado y contado (A6, A7, A10, A11)."""
    counts: dict[str, int] = dict.fromkeys(EXCLUSION_REASONS, 0)
    for item in universe.excluded:
        counts[item["reason"]] = counts.get(item["reason"], 0) + 1
    return {
        "series_id": universe.series_id,
        "labelled_sessions": universe.labelled_sessions,
        "sessions": len(universe.inputs),
        "first_session": _date_text(universe.first_session),
        "last_session": _date_text(universe.last_session),
        "context": None,
        "sessions_with_intraday_path": universe.intraday_sessions,
        "sessions_with_daily_fallback": universe.daily_fallback_sessions,
        "excluded": [dict(item) for item in universe.excluded],
        "exclusion_counts": counts,
        "intraday_path_inert": True,
        "intraday_path_note": (
            "el camino intradia (A10) es **inerte** mientras el decididor no declare "
            "`stop_px`/`target_px` (eso es #27): los seis baselines no declaran barreras, asi que "
            "toda operacion sale por el cierre de la sesion (A23)"
        ),
        "note": (
            "el universo es `derived.labels` ∩ muestra limpia de #52 ∩ `raw.market_daily` con "
            "OHLC completo; `context` es None (la carga opaca es de #19) y una sesion sin OHLC "
            "completo se excluye con su motivo, nunca se rellena con un precio inventado (A6)"
        ),
    }


def _reconciliation_payload(universe: Universe) -> dict[str, object]:
    """La reconciliacion publicada, sin rellenar huecos (A7, A8)."""
    return {
        "raw_market_daily_rows": universe.raw_daily_rows,
        "rows_with_previous_session": universe.rows_with_previous_session,
        "clean_from": _date_text(universe.clean_from),
        "clean_sessions": universe.clean_sessions,
        "sessions_since_cutoff": universe.sessions_since_cutoff,
        "excluded_by_clean_rule": universe.excluded_by_clean_rule,
        "stale_open_in_window": universe.stale_open_in_window,
        "no_forecast": universe.no_forecast_sessions,
        "labelled": universe.labelled_clean_sessions,
        "labelled_sessions_in_labels": universe.labelled_sessions,
        "identity": (
            f"{universe.sessions_since_cutoff} = {universe.excluded_by_clean_rule} "
            f"(stale_open) + {universe.no_forecast_sessions} (no_forecast) + "
            f"{universe.labelled_clean_sessions} (etiquetadas)"
        ),
        "stale_open_note": (
            "`excluded_by_clean_rule` son las filas que la regla limpia de #52 quita dentro de la "
            "ventana: el artefacto del `open` repetido mas, si la ventana incluyera la primera "
            "fila del almacen, la que no tiene sesion previa; `stale_open_in_window` cuenta solo "
            "el artefacto estricto"
        ),
        "no_forecast_reason": (
            "las etiquetas empiezan donde termina el calentamiento del *forecast* de #7: la "
            "diferencia entre la muestra limpia de #52 y el universo etiquetado se declara con su "
            "motivo, nunca se resta en silencio"
        ),
        "clean_rule": (
            "el corte y la muestra salen de `cfdtrader.analysis.drift` "
            "(`session_stale_open`, `clean_sample_cutoff`, `clean_sample`): la regla de #52 se "
            "importa, no se copia (A8)"
        ),
    }


def _plan_payload(split_plan: SplitPlan, *, params: PlanParams) -> dict[str, object]:
    """El plan con su eco literal de #12 (A14, A15)."""
    return {
        "plan_sha256": split_plan.plan_sha256,
        "n_splits": params.n_splits,
        "test_size": params.test_size,
        "embargo_sessions": params.embargo_sessions,
        "max_train_size": params.max_train_size,
        "label_horizon": PHASE1_LABEL_HORIZON,
        "n_sessions": split_plan.n_sessions,
        "n_test": sum(len(fold.test) for fold in split_plan.folds),
        "not_in_any_test": len(split_plan.uncovered),
        "purge_total": split_plan.purge_total,
        "embargo_total": split_plan.embargo_total,
        "embargo_in_train_total": split_plan.embargo_in_train_total,
        "exclusions_are_no_op": split_plan.exclusions_are_no_op,
        "exclusions_note": (
            "con `label_horizon = 0` la purga y el embargo son **no-ops estructurales**: se "
            "publican con sus numeros, nunca como un filtro activo (A15)"
        ),
    }


def _slippage_payload(slippage: SlippageParameter) -> dict[str, object]:
    """El *slippage* con su estado **literal**: los tres estados no se fusionan (A19, A28).

    ``pct_of_r`` se publica como **ratio** sobre ``R`` (``0.2``), que es como lo declara la
    issue; el parametro declarado de #8/#64 lo guarda en porcentaje, asi que su valor literal
    viaja al lado, sin re-teclearlo.
    """
    block = cast("dict[str, object]", _jsonable(slippage.model_dump(), where="slippage"))
    declared_percent = slippage.pct_of_r
    ratio: str | None = None
    if declared_percent is not None:
        ratio = format(declared_percent / Decimal(100), "f")
    block["pct_of_r"] = ratio
    block["pct_of_r_declared_percent"] = (
        None if declared_percent is None else format(declared_percent, "f")
    )
    block["issue"] = "#62"
    block["note"] = (
        "el estado se copia literalmente de `declared_slippage_assumption()` (#64): "
        "`is_measurement = false` y `r_pct = null` mientras #60 no decida `R`. `pct_of_r` es el "
        "**ratio** sobre `R` (0,2 = 20 %); el parametro declarado lo guarda en porcentaje y su "
        "valor literal va en `pct_of_r_declared_percent`. `issue` = la medicion que cierra el "
        "hueco (#62); `follow_up_issue` del modelo apunta a `R` (#60)"
    )
    return block


def _limits_payload(*, model: CostModel, slippage: SlippageParameter) -> dict[str, object]:
    """El bloque de limites declarados, sin fusionar estados (A28)."""
    return {
        "gate": "fail",
        "phase1_ready": False,
        "is_validation": False,
        "statement": (
            "los numeros de este arnes **no son una validacion de la estrategia**: la puerta de "
            "Fase 0 sigue en `fail` (#9/#64) y la tabla es de coste declarado"
        ),
        "costs": {
            "state": "declared_not_measured",
            "model_name": model.name,
            "source": "cfdtrader.backtest.costs.declared_cost_model() (cifras de #8)",
        },
        "slippage": {
            "state": slippage.state.value,
            "is_measurement": slippage.is_measurement,
            "pct_of_r": (
                None if slippage.pct_of_r is None else format(slippage.pct_of_r / Decimal(100), "f")
            ),
            "r_pct": None if slippage.r_pct is None else format(slippage.r_pct, "f"),
            "issue": "#62",
        },
        "prices": {
            "series_id": SERIES_ID,
            "proxy_of": PRICE_PROXY_OF,
            "is_proxy": True,
            "issue": "#50",
        },
        "financing_cut": None,
        "financing_cut_verified": False,
        "financing_cut_issue": "#59",
        "net_metrics_state": "not_computable",
        "llm_overlay": "disabled",
        "scheduler": "none",
    }


NET_METRICS_REASON: Final[str] = (
    "`pnl_net_pct` es `null` en todas las operaciones: el supuesto de #64 no se puede cobrar sin "
    "`R` (#60) y medir el *slippage* es #62"
)

NET_METRICS: Final[dict[str, object]] = {
    "state": "not_computable",
    "reason": NET_METRICS_REASON,
    "where": "cfdtrader.backtest.metrics.calculate_metrics",
    "follow_up": ["#62", "#60"],
    "note": (
        "regla «`null != 0`» de #15: las metricas netas (Sharpe/Sortino, EV, drawdown e "
        "intervalos) **no** se publican porque su publicacion exige una medicion y es #18"
    ),
}


def _payload(
    *,
    as_of: datetime,
    universe: Universe,
    split_plan: SplitPlan,
    outcomes: Sequence[BaselineOutcome],
    model: CostModel,
    slippage: SlippageParameter,
) -> dict[str, object]:
    """El payload canonico del informe: tipos JSON puros y determinista (A24-A28)."""
    rows = [_baseline_row(outcome, n_inputs=len(universe.inputs)) for outcome in outcomes]
    raw: dict[str, object] = {
        "analysis": "cfdtrader.analysis.backtest_report",
        "task": "#69",
        "generated_at": as_of.isoformat(),
        "hash_format": REPORT_HASH_FORMAT,
        "gate": "fail",
        "phase1_ready": False,
        "is_validation": False,
        "llm_overlay": "disabled",
        "scheduler": "none",
        "universe": _universe_payload(universe),
        "reconciliation": _reconciliation_payload(universe),
        "plan": _plan_payload(split_plan, params=PHASE1_PLAN),
        "cost_model": model.model_dump(mode="json"),
        "slippage": _slippage_payload(slippage),
        "baselines": {
            "basis": "declared_cost",
            "is_validation": False,
            "notional_usd": format(NOTIONAL_USD, "f"),
            "notional_provenance": NOTIONAL_PROVENANCE,
            "baseline_order": [outcome.baseline for outcome in outcomes],
            "random_matched": {
                "frequency": _fraction_text(RANDOM_MATCHED_FREQUENCY),
                "seed": RANDOM_MATCHED_SEED,
                "n_test": rows[-1]["n_test"] if rows else None,
                "traded": rows[-1]["traded"] if rows else None,
                "note": "las dos constantes son explicitas y viajan al informe (A16)",
            },
            "note": (
                "las cifras se calculan con aritmetica declarada sobre `pnl_declared_pct` (coste "
                "declarado, sin el termino supuesto): **prohibido** presentarlas como rendimiento "
                "neto medido (A20)"
            ),
            "rows": rows,
        },
        "net_metrics": dict(NET_METRICS),
        "limits": _limits_payload(model=model, slippage=slippage),
        "limitations": list(LIMITATIONS),
        "does_not_do": [dict(item) for item in ADAPTER_DOES_NOT_DO],
        "follow_ups": [dict(item) for item in FOLLOW_UPS],
    }
    return cast("dict[str, object]", _jsonable(raw, where="payload"))


# ─────────────────────────────────────────────────────────────────────────────
# Informe en prosa (A2)
# ─────────────────────────────────────────────────────────────────────────────
def render_markdown(report: BacktestReport) -> str:
    """El informe en Markdown, determinista y sin cifras que no esten en el payload."""
    payload = report.payload
    universe = cast("dict[str, object]", payload["universe"])
    reconciliation = cast("dict[str, object]", payload["reconciliation"])
    plan = cast("dict[str, object]", payload["plan"])
    baselines = cast("dict[str, object]", payload["baselines"])
    rows = cast("list[dict[str, object]]", baselines["rows"])
    limits = cast("dict[str, object]", payload["limits"])
    net = cast("dict[str, object]", payload["net_metrics"])
    random_matched = cast("dict[str, object]", baselines["random_matched"])
    costs = cast("dict[str, object]", limits["costs"])
    slippage = cast("dict[str, object]", limits["slippage"])
    prices = cast("dict[str, object]", limits["prices"])
    follow_up = cast("list[str]", net["follow_up"])

    lines: list[str] = [
        f"# Informe del arnes de backtest (Fase 1) — `{universe['series_id']}`",
        "",
        f"Generado el `{payload['generated_at']}` (**declarado**, no leido del reloj). "
        f"`report_sha256 = {report.report_sha256}`.",
        "",
        f"**No es una validacion.** `gate = {limits['gate']}`, "
        f"`phase1_ready = {limits['phase1_ready']}`: {limits['statement']}.",
        "",
        "## Universo",
        "",
        f"- Sesiones etiquetadas (`derived.labels`): **{universe['labelled_sessions']}**.",
        f"- Universo simulado: **{universe['sessions']}** sesiones, "
        f"`{universe['first_session']}` → `{universe['last_session']}`.",
        f"- Con camino intradia real: **{universe['sessions_with_intraday_path']}**; con respaldo "
        f"diario: **{universe['sessions_with_daily_fallback']}**.",
        f"- Excluidas con motivo: **{len(cast('list[object]', universe['excluded']))}**.",
        "",
        "## Reconciliacion",
        "",
        f"- Filas diarias: **{reconciliation['raw_market_daily_rows']}** "
        f"({reconciliation['rows_with_previous_session']} con sesion previa).",
        f"- Corte limpio (#52, importado de `analysis.drift`): "
        f"`{reconciliation['clean_from']}` → **{reconciliation['clean_sessions']}** sesiones "
        "limpias.",
        f"- Identidad: {reconciliation['identity']}.",
        f"- {reconciliation['no_forecast_reason']}.",
        "",
        "## Plan de folds",
        "",
        f"- `plan_sha256 = {plan['plan_sha256']}`",
        f"- `n_splits = {plan['n_splits']}`, `test_size = {plan['test_size']}`, "
        f"`embargo_sessions = {plan['embargo_sessions']}`, "
        f"`max_train_size = {plan['max_train_size']}`, `label_horizon = {plan['label_horizon']}`",
        f"- Sesiones de *test*: **{plan['n_test']}**; fuera de todo *test*: "
        f"**{plan['not_in_any_test']}**.",
        f"- Purga y embargo: `purge_total = {plan['purge_total']}`, "
        f"`embargo_total = {plan['embargo_total']}`, "
        f"`embargo_in_train_total = {plan['embargo_in_train_total']}`, "
        f"`exclusions_are_no_op = {plan['exclusions_are_no_op']}`.",
        "",
        "## Tabla comparativa (`basis: declared_cost`, no validacion)",
        "",
        f"Nocional plano declarado: **{baselines['notional_usd']} USD** "
        f"({baselines['notional_provenance']}).",
        "",
        "| baseline | n_test | traded | no_trade | skipped | tasa | media `pnl_declared_pct` | "
        "mediana | suma | `session_close` | `run_sha256` |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        summary = cast("dict[str, object]", row["pnl_declared_pct"])
        exits = cast("dict[str, int]", row["exit_reason_counts"])
        lines.append(
            f"| `{row['baseline']}` | {row['n_test']} | {row['traded']} | {row['no_trade']} | "
            f"{row['skipped']} | {row['trade_rate']} | {summary['mean']} | {summary['median']} | "
            f"{summary['sum']} | {exits[EXIT_SESSION_CLOSE]} | `{row['run_sha256']}` |"
        )
    lines.extend(
        [
            "",
            f"- {baselines['note']}",
            f"- `random_matched`: `frequency = {random_matched['frequency']}`, "
            f"`seed = {random_matched['seed']}`, `n_test = {random_matched['n_test']}`, "
            f"`traded = {random_matched['traded']}`.",
            "",
            "## Metricas netas",
            "",
            f"- `{net['state']}`: {net['reason']}",
            f"- Donde: `{net['where']}`; seguimiento: {', '.join(follow_up)}.",
            "",
            "## Limites declarados",
            "",
            f"- Puerta: `gate = {limits['gate']}`, `phase1_ready = {limits['phase1_ready']}`, "
            f"`is_validation = {limits['is_validation']}`.",
            f"- Costes: `{costs['state']}`.",
            f"- *Slippage*: `{slippage['state']}` `is_measurement = {slippage['is_measurement']}`, "
            f"`pct_of_r = {slippage['pct_of_r']}`, `r_pct = {slippage['r_pct']}`, "
            f"issue {slippage['issue']}.",
            f"- Precios: `{prices['series_id']}` es *proxy* de `{prices['proxy_of']}` "
            f"(issue {prices['issue']}).",
            f"- Corte de financiacion: `{limits['financing_cut']}` sin verificar "
            f"(issue {limits['financing_cut_issue']}).",
            f"- `llm_overlay = {limits['llm_overlay']}`, `scheduler = {limits['scheduler']}`.",
            "",
            "## Limitaciones",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in cast("list[str]", payload["limitations"]))
    lines.extend(["", "## Que no hace este modulo", ""])
    lines.extend(
        f"- **{item['issue']}** · `{item['id']}`: {item['statement']}"
        for item in cast("list[dict[str, str]]", payload["does_not_do"])
    )
    lines.extend(["", "## Seguimientos", ""])
    lines.extend(
        f"- **{item['issue']}** · {item['topic']}: {item['why']}"
        for item in cast("list[dict[str, str]]", payload["follow_ups"])
    )
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Ejecucion (A2, A3)
# ─────────────────────────────────────────────────────────────────────────────
def _as_utc(value: datetime) -> datetime:
    """Normaliza el instante declarado a UTC; sin zona se interpreta como UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _calendar_years(daily: pl.DataFrame) -> tuple[int, ...]:
    """Años que el calendario necesita materializar, tomados del dato (nunca del reloj)."""
    sessions = daily.get_column("session")
    if sessions.is_empty():
        return ()
    first = sessions.min()
    last = sessions.max()
    if first is None or last is None:
        return ()
    return tuple(range(int(str(first)[:4]), int(str(last)[:4]) + 1))


def analyse(
    *, store: Store, reports_dir: Path, as_of: datetime, write: bool = True
) -> BacktestReport:
    """Lee la historia, corre los seis baselines y (por defecto) escribe el informe (A2).

    ``as_of`` es **obligatorio** y es el unico instante de la corrida: ninguna ruta consulta
    el reloj. ``write=False`` no escribe **nada**.
    """
    moment = _as_utc(as_of)
    history = load_history(store)
    calendar = load_calendar(years=_calendar_years(history.daily))
    universe = build_inputs(history, calendar=calendar)
    split_plan = build_split_plan(universe.inputs)
    model = declared_cost_model()
    slippage = declared_slippage_assumption()
    outcomes = run_all_baselines(
        universe.inputs,
        split_plan=split_plan,
        cost_model=model,
        slippage=slippage,
    )
    payload = _payload(
        as_of=moment,
        universe=universe,
        split_plan=split_plan,
        outcomes=outcomes,
        model=model,
        slippage=slippage,
    )
    report = BacktestReport(
        as_of=moment,
        report_date=moment.astimezone(UTC).date(),
        payload=payload,
        report_sha256=_digest(payload),
        cost_model=model,
        slippage=slippage,
        split_plan=split_plan,
        universe=universe,
        outcomes=outcomes,
    )
    if write:
        json_path, markdown_path = report.write(reports_dir)
        logger.info("informe del arnes de Fase 1: {} y {}", json_path, markdown_path)
    return report


def _parse_as_of(value: str | None) -> datetime:
    """El instante declarado de la CLI: **obligatorio**, y nunca tomado del reloj (A2, A3)."""
    if value is None:
        raise MissingAsOfError(
            "falta --as-of: escribir el informe exige un instante declarado (el modulo no lee el "
            "reloj) y sin el **no se escribe ningun fichero** (A3)"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise InvalidAsOfError(f"--as-of no es un ISO-8601 valido: {error} (A3)") from error
    return _as_utc(parsed)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del informe del arnes de Fase 1.

    Codigos de salida: ``0`` = informe escrito (aunque la puerta siga en `fail`, que es un
    resultado legitimo y declarado); ``2`` = falta o no es valido ``--as-of``, falta un dataset
    o la muestra no alcanza ⇒ **no se escribe nada** y el motivo sale por ``stderr``.
    """
    parser = argparse.ArgumentParser(
        prog="cfdtrader.analysis.backtest_report",
        description="Adaptador Store -> SessionInput, los seis baselines y su informe (Fase 1)",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="raíz del almacén")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="directorio de informes (por defecto <data-root>/derived/reports)",
    )
    parser.add_argument("--settings", type=Path, default=None, help="ruta de settings.yaml")
    parser.add_argument(
        "--as-of",
        default=None,
        help="instante declarado ISO-8601, obligatorio para escribir (el módulo no lee el reloj)",
    )
    args = parser.parse_args(argv)

    try:
        moment = _parse_as_of(cast("str | None", args.as_of))
        settings = load_settings(args.settings)
    except (BacktestReportError, ConfigurationError) as error:
        print(f"no se puede emitir el informe del arnes: {error}", file=sys.stderr)
        return 2

    data_root = Path(args.data_root) if args.data_root is not None else Path(settings.data.root)
    reports_dir = (
        Path(args.reports_dir)
        if args.reports_dir is not None
        else data_root / "derived" / "reports"
    )
    try:
        report = analyse(store=Store(data_root), reports_dir=reports_dir, as_of=moment, write=True)
    except BacktestReportError as error:
        print(f"no se puede emitir el informe del arnes: {error}", file=sys.stderr)
        return 2

    baselines = cast("dict[str, object]", report.payload["baselines"])
    rows = cast("list[dict[str, object]]", baselines["rows"])
    logger.info(
        "arnes de Fase 1: {} baselines sobre {} sesiones; report_sha256 = {}",
        len(rows),
        report.universe.labelled_clean_sessions,
        report.report_sha256,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
