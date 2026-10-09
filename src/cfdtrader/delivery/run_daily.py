"""Camino diario: la pista del dia, o el estado «sin recomendacion» (#110).

Este modulo es el *entrypoint* manual de **Carril A** (``plan.md`` §19.7, #109): un asistente
de decision que se ejecuta a mano, imprime la pista del dia y **no** coloca ninguna orden. La
valla de honestidad viaja en las cuatro salidas: **no hay edge demostrado**, la ejecucion es
manual y esto es apoyo a la decision, no una estrategia validada.

Flujo de manana (§13), no evaluacion a cierre
---------------------------------------------

``session`` es la **proxima sesion**: la fecha ET de ``--as-of``. Su estado de features es la
**ultima fila** de ``analysis.feature_frame.build_feature_matrix`` (el cierre de la sesion
anterior), que es el insumo *lag-1* de ``models.baseline.design_frame``. La sesion anterior
declarada es ``MarketCalendar.previous_session(session)``.

Modelo: las dos familias del registro, sin reajustar
----------------------------------------------------

El modelo vive en ``runs/<run_sha256>/model.json`` y la probabilidad es la del **ultimo fold**
(``folds[-1]``) aplicado a la fila nueva, seguida de su calibrador. **No se reajusta** y **no** se
consulta ``test_positions``: es una **extrapolacion** mas alla de la ventana de *test* del fold,
declarada aqui a proposito. La **familia** la decide el payload: sin ``library`` es la lineal de
``models.baseline`` (``((x - mean) / scale) @ coefficients + intercept`` y despues
``Calibration.calibrate``, o ``sigmoid`` si el metodo es ``"none"``); con
``library.name == 'lightgbm'`` es el *booster* de #111, que se **recarga del texto publicado**
(``Booster(model_str=...)``, sin `pickle`) y se puntua con ``raw_score=True``, con ese mismo
calibrador sobre el **margen**. Cualquier otro ``library`` es un ``UnsupportedModelError`` tipado.

Elegir el modelo: por identidad o por el registro (#111)
--------------------------------------------------------

``--model-run <run_sha256>`` selecciona la corrida por identidad (lo de #110/#112);
``--variant-id <variant_id>`` la resuelve contra el registro con ``load_registry`` de
``analysis.experiment_log`` (**importada**, nunca reimplementada), sin teclear un `run_sha256`.
Son alternativos: exactamente uno de los dos es obligatorio, y el criterio de seleccion y los
`variant_id` soportados se declaran abajo (``VARIANT_SELECTION_RULE``/``SUPPORTED_VARIANTS``),
importados de ``analysis.model_comparison`` para no tener una tabla paralela.

Los cuatro estados de §19.2 y la guardia de §8.4 (#40)
------------------------------------------------------

La salida es uno de los cuatro estados de ``plan.md`` §19.2 y el informe imprime su ``value``
literal: ``recommendation``, ``no_recommendation_stale_data``, ``no_recommendation_data_quality``
y ``error``. El quinto estado del gate, ``no_recommendation_undecided``, no aparece porque el
escenario declarado S1 cierra los once parametros (``scenario_parameters``).

La guardia de obsolescencia de ``tech_stack.md`` §8.4 vive en ``delivery.staleness`` y se
aplica **despues de construir la matriz**:

- **Frescura**, con ``session_guard``: el veredicto sale de comparar la ultima sesion del almacen
  con la anterior a la evaluada (``missing_previous_close`` y ``snapshot_ahead`` son
  ``no_recommendation_stale_data``; el segundo es el caso de un ``--as-of`` que no se corresponde
  con el almacen) y el **contador de observacion** de la regla 15 se deriva del diario
  (``execution_dates`` + ``observation_sessions_remaining``): la vuelta de una ausencia de mas de
  una semana deja 5 sesiones por revalidar, que el gate convierte en ``NOTHING`` con su bloqueo 15.
- **Clausura** (``MARKET_CLOSED``, #113): un dia de mercado cerrado (festivo de EE. UU. o fin de
  semana) **si** ejecuta el camino y emite fila: la sesion no es sesion, el gate lo justifica con
  la regla 19 y el informe trae ``bloqueo: 19:mercado_cerrado``. La rama "no se ejecuta" de §8.4
  fue #40; #113 implementa su alternativa (``NOTHING`` justificado) y por eso este modulo ya no
  imprime un aviso sin informe ni fila.

Despues, la fila del almacen tiene que traer las diez features y un ``garch_forecast`` positivo
(si no, ``no_recommendation_data_quality``). El informe imprime la **justificacion** de un
``NOTHING`` bloqueado (una linea ``bloqueo: <regla>:<codigo>`` por bloqueo, con los codigos del
gate) y, en modo observacion, el ``motivo:`` lleva el aviso de reincorporacion: sin eso, el
``NOTHING`` de una media sesion o el de la observacion serian indistinguibles de un ``NOTHING``
cualquiera.



Sin reloj y sin red
--------------------------------------

El modulo no consulta el reloj (``--as-of`` es obligatorio, ISO-8601 con zona), no importa
``yfinance``/``requests``/``urllib`` y no escribe en el almacen ni en el registro: solo lee el
almacen, el registro (``config.json``/``result.json`` de ``runs/``, a traves de
``load_registry``) y ``runs/<run_sha256>/model.json``, y escribe **solo** el diario de decisiones
que declara ``--journal-root`` (#112). Mismas entradas ⇒ misma salida byte a byte.

Diario de decisiones (#112)
---------------------------

Cada ejecucion registra una fila de ``journal.decisions`` (la capa de #39) para la sesion
evaluada, **en los cuatro estados** de §19.2 y con el ``report_text`` verbatim. La fila lleva
``git_commit`` (inyectado), ``features_version`` (``FeatureMatrix.matrix_sha256``) y
``model_version`` (el ``run_sha256``), mas ``prob_up_raw`` (el ``sigmoid(score)`` previo al
calibrador) y ``prob_up_calibrated``. La salida por pantalla **no** cambia respecto a #110.

El overlay del LLM no decide: §19.19 (#148)
-------------------------------------------

La recomendacion que se **publica** y se registra en ``journal.decisions`` se emite **sin** overlay
(``evaluate_gate(..., overlay=None)``): es la consecuencia que la puerta de Fase 3 declaro de
antemano (§19.9, *«el LLM queda solo como redactor»*). El overlay se **sigue calculando** para el
informe y el estado ``llm_overlay``, y su efecto se conserva **aparte** como la variante «con
overlay» en ``journal.agent_signals`` (``agent = "news"``, #149) para el registro prospectivo de la
Fase 4. La **regla 20** (veto ⇒ ``NOTHING``) y ``apply_overlay`` (±10 pp) siguen implementados y
probados en el gate; lo que cambia es que el camino diario **no** le pasa overlay.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final, cast

import duckdb
import numpy as np
from lightgbm import Booster
from lightgbm.basic import LightGBMError
from pydantic import ValidationError

from cfdtrader.agents import report as report_agent
from cfdtrader.agents.event_calendar import (
    EventCalendarInputError,
    EventCalendarSignal,
    calendar_signal,
)
from cfdtrader.agents.news import PROMPT_TEMPLATE_NAME, NewsAgent, NewsAgentError
from cfdtrader.analysis import premarket_gap
from cfdtrader.analysis.backtest_report import NOTIONAL_USD
from cfdtrader.analysis.experiment_log import (
    ExperimentLogError,
    RegistryEntry,
    load_registry,
)
from cfdtrader.analysis.feature_frame import (
    FeatureFrameError,
    FeatureMatrix,
    build_feature_matrix,
)
from cfdtrader.analysis.model_comparison import BASELINE_VARIANT_ID, VARIANT_ID
from cfdtrader.analysis.pipeline_report import (
    EXPECTED_MOVE_BASIS,
    GARCH_COLUMN,
    SCENARIO_R_PCT,
    SCENARIO_STOP_SIGMA_MULTIPLE,
    SCENARIO_TARGET_STOP_MULTIPLE,
    scenario_parameters,
)
from cfdtrader.backtest.costs import (
    Side,
    cost_breakdown,
    declared_cost_model,
    declared_slippage_assumption,
)
from cfdtrader.data.calendar import (
    EASTERN,
    MarketCalendar,
    fomc_dates_for,
    load_calendar,
    load_fomc_calendar,
    pending_reason,
)
from cfdtrader.data.earnings import EarningsEvent, earnings_on
from cfdtrader.data.macro import MacroPublication, publications_on
from cfdtrader.data.news import load_headlines
from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.store import Store
from cfdtrader.decision.gate import GateOutput, GateStatus, evaluate_gate
from cfdtrader.decision.overlay import (
    OverlayDecision,
    OverlayState,
    disabled_overlay,
    overlay_from_extraction,
)
from cfdtrader.delivery.staleness import (
    GuardVerdict,
    SessionGuard,
    execution_dates,
    session_guard,
)
from cfdtrader.features.store import FEATURE_VERSION_PREFIX
from cfdtrader.journal.decision_log import DecisionLogError, Journal, build_decision
from cfdtrader.llm.base import LLMClientConfig, LLMError, build_client
from cfdtrader.llm.budget import (
    BatchCounts,
    BudgetCaps,
    BudgetGuard,
    CallSequence,
    MeteredLLMClient,
    OverlayClient,
    prepare_batch,
)
from cfdtrader.llm.cache import ResponseCache
from cfdtrader.models.baseline import BASELINE_FEATURES
from cfdtrader.models.calibration import Calibration, sigmoid
from cfdtrader.orchestration.observability import RunObserver

__all__ = [
    "DeliveryError",
    "MissingModelError",
    "RegistrySelectionError",
    "UnsupportedModelError",
    "main",
    "predict",
    "render",
    "resolve_run",
]

#: Nombre del documento del modelo dentro de la corrida declarada.
MODEL_FILE: Final[str] = "model.json"

#: Etiqueta del escenario declarado que cierra los parametros del gate (S1).
SCENARIO_LABEL: Final[str] = "S1"

#: Nombre del agente del overlay de noticias en `journal.agent_signals` (§12.5). Bajo §19.19
#: (#148) la recomendacion **publicada** se emite sin overlay; su efecto se conserva en esa tabla
#: como la variante «con overlay» (#149).
NEWS_OVERLAY_AGENT: Final[str] = "news"

#: Valla de honestidad (#109): viaja en **las cuatro** salidas, sin excepcion.
HONESTY_FENCE: Final[tuple[str, ...]] = (
    "no hay edge demostrado",
    "ejecucion: manual (el sistema no coloca ordenes; las decide el operador)",
    "naturaleza: apoyo a la decision, no una estrategia validada",
    f"escenario declarado: {SCENARIO_LABEL} (scenario_parameters); el `R` = {SCENARIO_R_PCT} % "
    "lo decidio #60 y el broker quedo declarado en #59 (Revolut, cuenta CFD, tabla confirmada "
    "por su KID): las dos son **decisiones**, no mediciones",
    "coste: la regla 9 se decide sobre el coste **declarado** (§19.12); el EV neto bajo el "
    "supuesto de *slippage* (20 % de `R` = 20 bp, #64) se publica como **sensibilidad**, no como "
    "una medicion (#62)",
    "pendiente de medir: el *slippage* de ejecucion (#62) y el intradia y `bid`/`ask` reales del "
    "CFD (#107) — sin ellos no hay coste medido ni veredicto de edge",
)

#: Claves del bloque de la pista que el modelo lineal tiene que publicar en su ultimo fold.
_LINEAR_FOLD_KEYS: Final[tuple[str, ...]] = ("mean", "scale", "coefficients", "intercept")

#: ``features_version`` declarada cuando el almacen **no** pudo construir la matriz (estado
#: ``error``): una ausencia declarada, nunca un digest inventado.
FEATURES_VERSION_UNAVAILABLE: Final[str] = "unavailable"

#: Subdirectorio de la traza estructurada (#43) cuando no se pasa `--observability-root`:
#: ``<journal-root>/ops``. La guardia lee ``decisions/``, asi que ``ops/`` no la contamina.
OBSERVABILITY_DIRNAME: Final[str] = "ops"

#: La cache de respuestas del LLM vive dentro de `ops/` (§12.6, `ops.llm_cache`).
LLM_CACHE_DIRNAME: Final[str] = "llm_cache"

#: Las dos familias del payload: la lineal de #110 (sin `library`) y la LightGBM de #111.
_LINEAR_FAMILY: Final[str] = "linear"
_LIGHTGBM_FAMILY: Final[str] = "lightgbm"

#: Los `variant_id` soportados: se **importan** de `analysis.model_comparison`, la unica fuente de
#: verdad, para que no haya una tabla paralela que se desincronice.
SUPPORTED_VARIANTS: Final[tuple[str, ...]] = (BASELINE_VARIANT_ID, VARIANT_ID)

#: Criterio declarado de seleccion cuando el registro trae varias entradas del mismo `variant_id`.
VARIANT_SELECTION_RULE: Final[str] = (
    "entre las entradas del registro con ese `variant_id` gana la de mayor "
    "`sharpe_per_session` y, a igualdad, el `run_sha256` menor: determinista y sin depender "
    "del orden de lectura"
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados
# ─────────────────────────────────────────────────────────────────────────────
class DeliveryError(Exception):
    """Raiz de los errores del camino diario."""


class MissingModelError(DeliveryError):
    """No existe el ``model.json`` de la corrida declarada."""


class UnsupportedModelError(DeliveryError):
    """El documento no es ninguna de las familias soportadas (lineal o LightGBM)."""


class RegistrySelectionError(DeliveryError):
    """El `variant_id` declarado no se pudo resolver contra el registro de `runs/`."""


# ─────────────────────────────────────────────────────────────────────────────
# Prediccion: el ultimo fold aplicado a la fila nueva, con su calibrador
# ─────────────────────────────────────────────────────────────────────────────
def predict(model_path: Path, features: Mapping[str, float]) -> float:
    """Probabilidad **calibrada** de la fila nueva con el ultimo fold del modelo lineal.

    Lee ``model["folds"][-1]`` y devuelve la probabilidad de las dos familias soportadas: la
    lineal (``((x - mean) / scale) @ coefficients + intercept``) y la LightGBM (#111), en los dos
    casos pasada por el calibrador publicado (o por ``sigmoid`` si el metodo es ``"none"``). No
    reajusta nada y es determinista. Un documento ausente es :class:`MissingModelError`; uno que
    no sea ninguna de las dos familias, o que no declare exactamente las diez columnas de
    ``BASELINE_FEATURES``, es :class:`UnsupportedModelError`.
    """
    return _predictions(model_path, features)[1]


def _predictions(model_path: Path, features: Mapping[str, float]) -> tuple[float, float]:
    """``(prob_up_raw, prob_up_calibrated)`` del ultimo fold: las dos se persisten (#112).

    ``prob_up_raw`` es ``sigmoid(score)`` **antes** del calibrador (la probabilidad cruda de
    §19.3); ``prob_up_calibrated`` es la que consume el gate. Sin recalibrador (o con el metodo
    ``"none"``) coinciden.
    """
    document = _load_document(model_path)
    model = cast("Mapping[str, object]", document["model"])
    family = _family(model, model_path=model_path)
    if "features" not in model or "folds" not in model:
        raise UnsupportedModelError(
            f"{model_path}: el documento no declara `features`/`folds`: no es una familia soportada"
        )
    declared = tuple(str(name) for name in cast("Sequence[object]", model["features"]))
    if declared != BASELINE_FEATURES:
        raise UnsupportedModelError(
            f"{model_path}: el documento declara {declared!r} y el camino diario puntua "
            "exactamente las 10 columnas de `models.baseline.BASELINE_FEATURES`"
        )
    folds = cast("Sequence[object]", model["folds"])
    if not folds:
        raise UnsupportedModelError(f"{model_path}: el modelo no publica ningun fold")
    fold = cast("Mapping[str, object]", folds[-1])
    if family == _LIGHTGBM_FAMILY:
        return _lightgbm_probabilities(fold, features, declared, model_path=model_path)
    return _fold_probabilities(fold, features, declared, model_path=model_path)


def _family(model: Mapping[str, object], *, model_path: Path) -> str:
    """La familia del payload: la lineal de #110 (sin `library`) o la LightGBM de #111.

    Un `library` de otra forma (p. ej. un texto, como el que rechazaba #110) **no** es una familia
    soportada: se declara, no se intenta puntuar.
    """
    if "library" not in model:
        return _LINEAR_FAMILY
    block = model["library"]
    if isinstance(block, Mapping):
        declared_library = cast("Mapping[str, object]", block)
        if declared_library.get("name") == _LIGHTGBM_FAMILY:
            return _LIGHTGBM_FAMILY
    raise UnsupportedModelError(
        f"{model_path}: el documento publica un `library` que no es una familia soportada "
        "(`library.name == 'lightgbm'` para la de #111, o sin `library` para la lineal de #110)"
    )


def _lightgbm_probabilities(
    fold: Mapping[str, object],
    features: Mapping[str, float],
    declared: tuple[str, ...],
    *,
    model_path: Path,
) -> tuple[float, float]:
    """``(prob_up_raw, prob_up_calibrated)`` del ultimo fold de LightGBM, sin reajustar.

    El texto publicado del ``Booster`` (``model_to_string()``, **sin** `pickle`) se **recarga** con
    ``Booster(model_str=...)`` y la fila nueva se puntua con ``raw_score=True``: el margen, del que
    sale ``prob_up_raw = sigmoid(margen)``. El calibrador **publicado** del fold se aplica a ese
    mismo margen (la aritmetica de `models.lightgbm_model` / `analysis.model_comparison`). El
    artefacto **solo se lee**: no se reajusta ni se reescribe.
    """
    text = fold.get("booster_model")
    if not isinstance(text, str) or not text.strip():
        raise UnsupportedModelError(
            f"{model_path}: el ultimo fold no publica `booster_model` como texto: no es la "
            "familia LightGBM"
        )
    missing = [name for name in declared if name not in features]
    if missing:
        raise DeliveryError(
            f"la fila a predecir no trae las features declaradas: faltan {missing!r}"
        )
    row = np.asarray([[float(features[name]) for name in declared]], dtype=np.float64)
    try:
        booster = Booster(model_str=text)
        margin = float(np.asarray(booster.predict(row, raw_score=True)).ravel()[0])
    except LightGBMError as error:
        raise UnsupportedModelError(
            f"{model_path}: el `booster_model` del ultimo fold no puntua la fila nueva: {error}"
        ) from error
    raw = sigmoid(margin)
    block = fold.get("calibration")
    if block is None:
        return raw, raw
    calibrator = _calibration_from_payload(cast("Mapping[str, object]", block))
    if not calibrator.calibrated:
        return raw, raw
    value = calibrator.calibrate([margin])[0]
    return raw, (raw if value is None else float(value))


def _fold_probabilities(
    fold: Mapping[str, object],
    features: Mapping[str, float],
    declared: tuple[str, ...],
    *,
    model_path: Path,
) -> tuple[float, float]:
    """Las dos probabilidades del fold: cruda (``sigmoid``) y calibrada, sin reajustar."""
    for key in _LINEAR_FOLD_KEYS:
        if key not in fold:
            raise UnsupportedModelError(
                f"{model_path}: el ultimo fold no declara {key!r}: no es la familia lineal"
            )
    missing = [name for name in declared if name not in features]
    if missing:
        raise DeliveryError(
            f"la fila a predecir no trae las features declaradas: faltan {missing!r}"
        )
    score = _score(fold, features, declared, model_path=model_path)
    raw = sigmoid(score)
    block = fold.get("calibration")
    if block is None:
        return raw, raw
    calibrator = _calibration_from_payload(cast("Mapping[str, object]", block))
    if not calibrator.calibrated:
        return raw, raw
    value = calibrator.calibrate([score])[0]
    return raw, (raw if value is None else float(value))


def _score(
    fold: Mapping[str, object],
    features: Mapping[str, float],
    declared: tuple[str, ...],
    *,
    model_path: Path,
) -> float:
    """El score del fold: ``((x - mean) / scale) @ coefficients + intercept``, en ese orden."""
    mean = cast("Sequence[object]", fold["mean"])
    scale = cast("Sequence[object]", fold["scale"])
    coefficients = cast("Sequence[object]", fold["coefficients"])
    intercept = float(cast("float", fold["intercept"]))
    lengths = (len(mean), len(scale), len(coefficients), len(declared))
    if len(set(lengths)) != 1:
        raise UnsupportedModelError(
            f"{model_path}: el fold mezcla longitudes {lengths!r}: no es la familia lineal"
        )
    total = 0.0
    for name, centre, spread, coefficient in zip(declared, mean, scale, coefficients, strict=True):
        value = float(features[name])
        total += ((value - float(cast("float", centre))) / float(cast("float", spread))) * float(
            cast("float", coefficient)
        )
    return total + intercept


def _load_document(model_path: Path) -> Mapping[str, object]:
    """El ``model.json`` de esa corrida, o un error tipado (nunca un ``KeyError`` suelto)."""
    if not model_path.is_file():
        raise MissingModelError(f"no existe el modelo declarado: {model_path}")
    try:
        loaded = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UnsupportedModelError(f"`model.json` ilegible en {model_path}: {error}") from error
    if not isinstance(loaded, dict):
        raise UnsupportedModelError(
            f"{model_path}: no es un documento de modelo reconocible (falta el bloque `model`)"
        )
    document = cast("dict[str, object]", loaded)
    if not isinstance(document.get("model"), dict):
        raise UnsupportedModelError(
            f"{model_path}: no es un documento de modelo reconocible (falta el bloque `model`)"
        )
    return document


def _calibration_from_payload(block: Mapping[str, object]) -> Calibration:
    """Reconstruye el calibrador publicado desde su bloque JSON, sin reajustar (A7, #25)."""
    parameters = cast("Mapping[str, object]", block.get("parameters") or {})
    return Calibration(
        method=str(block["method"]),
        reason=None if block.get("reason") is None else str(block["reason"]),
        n_fit=int(cast("int", block["n_fit"])),
        n_calibration=int(cast("int", block["n_calibration"])),
        n_positives=int(cast("int", block["n_positives"])),
        calibration_positions=tuple(
            int(cast("int", value))
            for value in cast("Sequence[object]", block["calibration_positions"])
        ),
        purge_sessions=int(cast("int", block["purge_sessions"])),
        exclusions_are_no_op=bool(block["exclusions_are_no_op"]),
        coef=_optional_float(parameters.get("coef")),
        intercept=_optional_float(parameters.get("intercept")),
        mean=_optional_float(parameters.get("mean")),
        scale=_optional_float(parameters.get("scale")),
        thresholds=tuple(
            float(cast("float", value))
            for value in cast("Sequence[object]", parameters.get("thresholds") or ())
        ),
        values=tuple(
            float(cast("float", value))
            for value in cast("Sequence[object]", parameters.get("values") or ())
        ),
    )


def _optional_float(value: object) -> float | None:
    """Un parametro opcional del calibrador: ``None`` sigue siendo ``None``."""
    return None if value is None else float(cast("float", value))


# ─────────────────────────────────────────────────────────────────────────────
# Informe
# ─────────────────────────────────────────────────────────────────────────────
def _declared_fomc_dates(year: int) -> tuple[tuple[date, ...], str, bool]:
    """Los dias de FOMC del año, la nota que describe su procedencia y si el año esta declarado.

    Existe por una indistincion concreta: un conjunto vacio puede significar «hoy no hay FOMC» o
    «nadie ha declarado el calendario», y son cosas distintas (#114). Antes eran indistinguibles
    porque el camino diario pasaba un literal ``()``. Ahora el año ausente sale **en voz alta**.
    """
    try:
        config = load_fomc_calendar()
    except ConfigurationError as failure:
        return (), f"el calendario de FOMC declarado no se puede leer: {failure}", False
    if config is None:
        return (), "no hay calendario de FOMC declarado: la regla 17 no puede dispararse", False
    dates = fomc_dates_for(config, year)
    if dates is None:
        detail = (
            f"el año {year} no esta declarado en el calendario de FOMC ({config.source}): la regla "
            "17 no puede dispararse"
        )
        pending = pending_reason(config, year)
        if pending is not None:
            detail = (
                f"el año {year} no esta declarado en el calendario de FOMC "
                f"({config.source}) y esta declarado como pendiente desde "
                f"{config.pending[year].attempted_on.isoformat()}: {pending}: la regla 17 no puede "
                "dispararse"
            )
        return (), detail, False
    return dates, f"{config.source} (verificado {config.verified_on.isoformat()})", True


def _day_events(
    calendar: MarketCalendar, session: date, moment: datetime
) -> EventCalendarSignal | None:
    """La senal del calendario del dia, o ``None`` si no se puede calcular (#121).

    Es **informativa**: no bloquea nada por si misma, porque las reglas 18 y 19 del gate ya leen
    el **mismo** `MarketCalendar`. Si falla, el dia se emite **sin** la seccion de eventos en vez
    de caerse: es la degradacion gracil de §7.4 aplicada a una capa que no esta en el camino
    critico.
    """
    try:
        return calendar_signal(calendar, session, as_of=moment)
    except EventCalendarInputError as failure:
        print(f"no se puede calcular la senal del calendario: {failure}", file=sys.stderr)
        return None


def _day_publications(
    store: Store, session: date, moment: datetime
) -> tuple[MacroPublication, ...]:
    """Las publicaciones macro del dia, o ``()`` si no se pueden leer (#125).

    Es **informativa**: no bloquea nada ni cambia la direccion, el `tier` ni el
    `blocking_events`. Si el registro macro o el almacen no se pueden leer, el dia se emite
    **sin** su seccion en vez de caerse: es la degradacion gracil de §7.4 aplicada a una capa
    que no esta en el camino critico.
    """
    try:
        return publications_on(store=store, session=session, as_of=moment)
    except (ConfigurationError, duckdb.Error) as failure:
        print(f"no se pueden leer las publicaciones macro del dia: {failure}", file=sys.stderr)
        return ()


def _day_earnings(store: Store, session: date, moment: datetime) -> tuple[EarningsEvent, ...]:
    """Los resultados de mega-caps del dia, o ``()`` si no se pueden leer (#126).

    Se leen del almacen (``raw.earnings``), nunca de ``yfinance``: la ingesta es el CLI de
    ``data.earnings``. Si el almacen no se puede leer, el dia se emite **sin** su seccion.
    """
    try:
        return earnings_on(store, session=session, as_of=moment)
    except (ConfigurationError, duckdb.Error) as failure:
        print(f"no se pueden leer los resultados de mega-caps: {failure}", file=sys.stderr)
        return ()


def _earnings_blockers(earnings: Sequence[EarningsEvent]) -> tuple[str, ...]:
    """Los codigos de los resultados **confirmados** del dia (una fecha estimada no bloquea).

    Se calculan aqui, no en el gate: una fecha confirmada es un evento del calendario y su
    codigo viaja en `blocking_events` del diario sin que el gate gane una regla nueva (§19.8).
    """
    return tuple(f"earnings_confirmado:{event.symbol}" for event in earnings if event.blocking)


def _captured_fetcher(
    bars: Sequence[premarket_gap.PremarketBar],
) -> premarket_gap.BarsFetcher:
    """Un *fetcher* que devuelve las barras capturadas del ES (#141), sin red.

    El camino diario **no** descarga datos (lee el almacen): la captura de las 08:45 la hace el CLI
    ``cfdtrader.analysis.premarket_gap`` y aqui solo se leen sus barras.
    """

    def fetch(_series_id: str, *, as_of: datetime) -> Sequence[premarket_gap.PremarketBar]:
        del as_of
        return bars

    return fetch


def _report_facts(
    output: GateOutput,
    *,
    publications: Sequence[MacroPublication],
    earnings: Sequence[EarningsEvent],
    calendar_events: EventCalendarSignal | None,
) -> report_agent.ReportFacts:
    """Los **hechos** que el informe redacta: datos ya calculados, no numeros del LLM (#37)."""
    direction = "nothing" if output.direction is None else output.direction.value
    return report_agent.ReportFacts(
        trade_date=output.session,
        direction=direction,
        prob_up_calibrated=output.prob_up_calibrated,
        expected_move_pct=float(output.expected_move_pct),
        cost_pct=float(output.cost_pct),
        # `ev_net_pct` y `target_pct` son nullables en el gate; el stop **siempre** existe.
        ev_net_pct=None if output.ev_net_pct is None else float(output.ev_net_pct),
        stop_pct=float(output.stop_pct),
        target_pct=None if output.target_pct is None else float(output.target_pct),
        tier=output.tier or "",
        blocking_events=tuple(str(entry["code"]) for entry in output.blockers),
        day_events=() if calendar_events is None else tuple(_calendar_lines(calendar_events)),
        publications=tuple(_publication_lines(publications)),
        earnings=tuple(_earnings_lines(earnings)),
    )


def _compose_report(
    output: GateOutput,
    ops_root: Path,
    moment: datetime,
    *,
    publications: Sequence[MacroPublication],
    earnings: Sequence[EarningsEvent],
    calendar_events: EventCalendarSignal | None,
    sequence: CallSequence,
) -> tuple[report_agent.ReportDraft | None, Mapping[str, object]]:
    """El informe redactado por el modelo de mayor calidad, o ``None`` si no se puede (#37).

    **Nunca lanza.** Es una capa **opcional** por diseno (`tech_stack.md` §4.9): si falta la clave,
    el modelo, la plantilla o el presupuesto, devuelve ``None`` y el camino diario publica su
    informe determinista de siempre. Devuelve tambien los ``prompt_hashes`` que van al diario.

    El modelo **no** decide ni calcula: recibe los hechos ya calculados y solo redacta (con su
    contra-argumento). El guardia se consulta **antes** de llamar: con el overlay cortado no se paga
    una llamada para nada.
    """
    cache: ResponseCache | None = None
    metered: MeteredLLMClient | None = None
    draft: report_agent.ReportDraft | None = None
    prompt_hashes: dict[str, object] = {}
    try:
        settings = LLMClientConfig()
        model = settings.model_for("report")
        facts = _report_facts(
            output,
            publications=publications,
            earnings=earnings,
            calendar_events=calendar_events,
        )
        cache = ResponseCache(ops_root / LLM_CACHE_DIRNAME)
        metered = MeteredLLMClient(
            build_client(settings),
            cache=cache,
            guard=BudgetGuard(caps=BudgetCaps().caps()),
            as_of=moment,
            provider=settings.provider,
            purpose="report",
            journal=ops_root,
            sequence=sequence,
        )
        if metered.state != OverlayState.APPLIED:
            return None, {}
        agent = report_agent.ReportAgent(OverlayClient(metered), model=model)
        draft = agent.compose(facts)
        prompt_hashes = {report_agent.PROMPT_TEMPLATE_NAME: agent.prompt_hash}
    except (
        LLMError,
        report_agent.ReportAgentError,
        ValidationError,
        ConfigurationError,
    ) as failure:
        print(f"no se puede redactar el informe: {failure}", file=sys.stderr)
        return None, {}
    finally:
        if cache is not None:
            cache.close()
    return draft, prompt_hashes


def render(
    *,
    status: GateStatus,
    session: date | None,
    as_of: datetime,
    snapshot_session: date | None,
    model_source: str,
    message: str,
    output: GateOutput | None = None,
    calendar_events: EventCalendarSignal | None = None,
    publications: Sequence[MacroPublication] = (),
    earnings: Sequence[EarningsEvent] = (),
    report: report_agent.ReportDraft | None = None,
    premarket: Mapping[str, object] | None = None,
    notes: Sequence[str] = (),
) -> str:
    """El informe del dia: cabecera, pista (si la hay), bloqueos, motivo y valla (#109, #40).

    Los cuatro estados salen con el mismo formato y **todos** imprimen la valla: la salida no
    puede parecer una estrategia validada ni ocultar que la ejecucion es manual. Los bloqueos del
    gate se publican **solo** cuando los hay, para que un ``NOTHING`` justificado (media sesion,
    modo observacion) no sea indistinguible de un ``NOTHING`` cualquiera (#40).
    """
    lines = [
        f"estado: {status.value}",
        f"sesion: {_date_or_null(session)}",
        f"as_of: {as_of.isoformat()}",
        f"snapshot_sesion: {_date_or_null(snapshot_session)}",
        f"modelo: {model_source}",
    ]
    if output is not None:
        lines.extend(_recommendation_lines(output))
        lines.extend(_blocker_lines(output))
    if calendar_events is not None:
        lines.extend(_calendar_lines(calendar_events))
    lines.extend(_publication_lines(publications))
    lines.extend(_earnings_lines(earnings))
    if premarket is not None:
        lines.extend(_premarket_lines(premarket))
    if report is not None:
        lines.extend(_report_lines(report))
    lines.extend(f"nota: {note}" for note in notes)
    lines.append(f"motivo: {message}")
    lines.append("")
    lines.extend(HONESTY_FENCE)
    return "\n".join(lines) + "\n"


def _recommendation_lines(output: GateOutput) -> list[str]:
    """Las claves de la pista (criterio 8), con ``null`` donde el dato no existe."""
    direction = "null" if output.direction is None else output.direction.name
    return [
        f"direccion: {direction}",
        f"prob_calibrada: {output.prob_up_calibrated!r}",
        f"ev_declarado_pct: {_decimal_or_null(output.ev_declared_pct)}",
        f"ev_neto_pct: {_decimal_or_null(output.ev_net_pct)}",
        f"base_del_ev: {output.ev_basis}",
        f"ev_neto_es_sensibilidad: {str(output.ev_net_is_sensitivity).lower()}",
        f"stop_pct: {_decimal_or_null(output.stop_pct)}",
        f"objetivo_pct: {_decimal_or_null(output.target_pct)}",
        f"tier: {output.tier}",
        f"gate_sha256: {output.gate_sha256}",
    ]


def _blocker_lines(output: GateOutput) -> list[str]:
    """Una linea por bloqueo del gate (``regla:codigo``), en el orden en que los publica (#40).

    Es la justificacion de un ``NOTHING``: sin esta linea, un bloqueo por media sesion (regla 18)
    o por modo observacion (regla 15) se presentaria igual que un ``NOTHING`` sin motivo.
    """
    return [f"bloqueo: {entry['rule']}:{entry['code']}" for entry in output.blockers]


def _calendar_lines(signal: EventCalendarSignal) -> list[str]:
    """Los eventos del dia, **informativos incluidos** (#121).

    `blocking_events` lleva solo lo que bloquea; esta seccion publica el dia entero para que un
    OPEX o un roll del ES **se lean** y no se confundan con un bloqueo: son informativos por
    decision declarada (`plan.md` §19.8). El prefijo distingue las dos cosas.
    """
    return [
        f"{'evento_bloqueante' if event.blocking else 'evento'}: {event.kind.value} | {event.name}"
        for event in signal.events
    ]


def _publication_lines(publications: Sequence[MacroPublication]) -> list[str]:
    """Las publicaciones macro del dia, con su hora real y su disponibilidad (#125).

    Una publicacion **no bloquea**: va con prefijo propio (`publicacion_macro:`) para no
    confundirse con un `bloqueo:` del gate ni con un `evento:` del calendario. Cuando la serie
    no declara hora, se publica **declarando la ausencia**, nunca con una inventada; entonces la
    disponibilidad queda `no evaluable`.
    """
    lines: list[str] = []
    for publication in publications:
        if publication.release_time_et is None:
            when = "hora no declarada"
            available = "no evaluable"
        else:
            at_et = cast("datetime", publication.release_at_et)
            at_utc = cast("datetime", publication.release_at_utc)
            at_madrid = cast("datetime", publication.release_at_madrid)
            when = f"{at_et:%H:%M} ET ({at_utc:%H:%M} UTC / {at_madrid:%H:%M} Madrid)"
            available = "si" if publication.available_at_as_of else "no"
        lines.append(
            f"publicacion_macro: {publication.series_id} | {publication.name} | {when} | "
            f"disponible en el as_of: {available}"
        )
    return lines


def _earnings_lines(earnings: Sequence[EarningsEvent]) -> list[str]:
    """Los resultados de mega-caps del dia, con su momento y su certeza (#126).

    El **momento** (``bmo``/``amc``/``unknown``) se publica tal cual: cuando la fuente no lo da,
    se declara ``unknown`` en vez de asumir uno. Una fecha **estimada** se lee con
    ``bloquea: no``; solo una **confirmada** bloquea y su codigo va a ``blocking_events``.
    """
    return [
        f"resultado_mega_cap: {event.symbol} | {event.name} | {event.on.isoformat()} | "
        f"momento: {event.moment.value} | certeza: {event.certainty.value} | "
        f"bloquea: {'si' if event.blocking else 'no'}"
        for event in earnings
    ]


def _premarket_lines(payload: Mapping[str, object]) -> list[str]:
    """El gap de pre-mercado del ES (#141): **dato declarado** de la decision, no una feature.

    Los tres numeros van juntos y **nunca se suman**: el movimiento del futuro, el gap frente al
    indice (que incluye el *basis*) y el *basis* mismo.
    """

    def _show(name: str) -> str:
        value = payload[name]
        return "null" if value is None else f"{value} %"

    return [
        f"premarket_es: {payload['state']} — {payload['reason']}",
        f"  overnight_move_pct: {_show('overnight_move_pct')}",
        f"  gap_vs_index_pct: {_show('gap_vs_index_pct')}",
        f"  basis_pct: {_show('basis_pct')}",
        f"  nota: {payload['not_a_model_feature']}",
    ]


def _report_lines(draft: report_agent.ReportDraft) -> list[str]:
    """El informe redactado por el LLM: narrativa y contra-argumento (#37).

    Va con prefijo propio (`redaccion:`) y **nunca** reescribe los numeros: los hechos ya estan en
    las lineas de arriba. Es la unica parte del informe que puede faltar sin romper el pipeline.
    """
    lines = [f"redaccion: {draft.narrative}"]
    lines.extend(f"contra_argumento: {item}" for item in (*draft.bull_case, *draft.bear_case))
    lines.append(f"redaccion_modelo: {draft.model} | prompt: {draft.prompt_hash}")
    return lines


def _date_or_null(value: date | None) -> str:
    """Una fecha ISO, o ``null`` sin inventar un valor."""
    return "null" if value is None else value.isoformat()


def _decimal_or_null(value: Decimal | None) -> str:
    """Un ``Decimal`` exacto en forma decimal, o ``null`` (nunca un ``0`` de relleno)."""
    return "null" if value is None else format(value, "f")


# ─────────────────────────────────────────────────────────────────────────────
# La guardia de obsolescencia (§8.4), el motivo del informe y el movimiento esperado
# ─────────────────────────────────────────────────────────────────────────────
def _guard_message(guard: SessionGuard) -> str:
    """El ``motivo:`` del informe: el de la guardia (o el declarado) **mas** su aviso (#40).

    Con la guardia parada, el motivo es el suyo (la frescura). Siguiendo, el motivo es el declarado
    del escenario -para no cambiar el informe de #110 sin necesidad-. En los dos casos se le suma
    el **aviso de reincorporacion** cuando lo hay: la fila 3 de la tabla de §8.4 pide "sin
    recomendacion **+ aviso explicito de reincorporacion**", y la fila 6 pide que el modo
    observacion se muestre marcado ("no operar hasta revalidar") en el informe, no solo en el
    diario.
    """
    base = (
        guard.message
        if guard.blocks
        else f"pista evaluada con el escenario declarado {SCENARIO_LABEL} y coste declarado"
    )
    notice = guard.reincorporation_notice
    return base if notice is None else f"{base} | {notice}"


def _row_problem(row: Mapping[str, object]) -> str | None:
    """El problema de calidad de la fila evaluada, o ``None`` si se puede predecir (§8.4)."""
    missing = [name for name in BASELINE_FEATURES if row.get(name) is None]
    if missing:
        return f"la fila tiene features nulas: {', '.join(missing)}"
    variance = row.get(GARCH_COLUMN)
    if variance is None:
        return f"{GARCH_COLUMN} es null: sin sigma no hay movimiento, stop ni objetivo"
    value = float(cast("float", variance))
    if not math.isfinite(value) or value <= 0.0:
        return (
            f"{GARCH_COLUMN} no es positivo ({value!r}): sin sigma no hay movimiento, stop ni "
            "objetivo"
        )
    return None


def _last_row(matrix: FeatureMatrix) -> Mapping[str, object]:
    """La fila evaluada: la **ultima** del matrix (el estado al cierre de la anterior)."""
    return cast("Mapping[str, object]", matrix.frame.tail(1).row(0, named=True))


def _expected_move_pct(row: Mapping[str, object]) -> Decimal:
    """``sqrt(garch_forecast) * 100`` de la fila evaluada (``GARCH_COLUMN``, §13)."""
    variance = float(cast("float", row[GARCH_COLUMN]))
    return Decimal(str(math.sqrt(variance) * 100.0))


def _compute_overlay(
    data_root: Path,
    ops_root: Path,
    moment: datetime,
    *,
    sequence: CallSequence,
) -> tuple[OverlayDecision, Mapping[str, object], dict[str, int]]:
    """El overlay del dia: titulares del almacen -> `NewsAgent` -> `OverlayDecision`.

    **Nunca lanza.** El overlay es opcional por diseno (`tech_stack.md` §4.9): si falta la clave,
    el modelo, la plantilla o el presupuesto, devuelve el estado **desactivado** que corresponda y
    el camino diario sigue produciendo su recomendacion. Devuelve tambien el ``prompt_hashes`` que
    va al diario (``{}`` si no se llego a llamar al modelo).

    El guardian se consulta **antes** de invocar al agente: con el presupuesto agotado no se paga
    una llamada para nada. Los titulares se **leen del almacen** (la ingesta es el CLI de #30), no
    se descargan aqui: el camino diario no debe caer porque GDELT o un RSS esten lentos.

    La cache vive bajo ``ops_root`` —la raiz de observabilidad, no la del diario— porque es donde
    ``tech_stack.md`` §12.6 situa ``ops.llm_cache``, y porque ``--observability-root`` tiene que
    desviarla igual que desvia el ``run_log`` (#43).

    Devuelve tambien, como tercer elemento, los **conteos del lote** (#129): cuantos titulares se
    leyeron, cuantos colapso la deduplicacion, cuantos quedaron fuera de la ventana, cuantos se
    prepararon y cuantos llegaron de verdad al modelo. Sin ese conteo, §6.3.5 no se completa.
    """
    cache: ResponseCache | None = None
    metered: MeteredLLMClient | None = None
    counts = BatchCounts(read=0, duplicates=0, out_of_window=0, sent=0, truncated=0)
    try:
        # El conteo del lote se toma **antes** de construir la capa metrada, para poder declararlo
        # aunque la capa LLM este caida (#129). El **estado** del overlay no cambia de precedencia:
        # sigue declarandose con el mismo orden de siempre, y por eso el corte por «sin titulares»
        # va **despues** de construir el cliente. Un proveedor ausente sigue siendo `disabled_error`
        # y no «sin titulares en la ventana»: el journal registra que **no se pudo**, no que no
        # hiciera falta.
        prepared, counts = prepare_batch(load_headlines(Store(data_root), as_of=moment), now=moment)
        settings = LLMClientConfig()
        model = settings.model_for("extract")
        cache = ResponseCache(ops_root / LLM_CACHE_DIRNAME)
        metered = MeteredLLMClient(
            build_client(settings),
            cache=cache,
            guard=BudgetGuard(caps=BudgetCaps().caps()),
            as_of=moment,
            provider=settings.provider,
            purpose="extract",
            journal=ops_root,
            sequence=sequence,
        )
        if not prepared:
            return (
                OverlayDecision(
                    state=OverlayState.APPLIED, reasons=("sin titulares en la ventana",)
                ),
                {},
                _headline_counters(counts, sent=0),
            )
        extraction = NewsAgent(OverlayClient(metered), model=model).extract(prepared)
    except (LLMError, NewsAgentError, ValidationError, ConfigurationError) as failure:
        # `metered.state` habla el vocabulario de la capa de coste (#33) y `disabled_overlay` el de
        # la decision (#35). Los dos modulos declaran los MISMOS cinco estados de §12.5 pero **no**
        # comparten clase — el reparto de capas esta probado en los dos sentidos —, asi que aqui se
        # traduce **por valor**, que es el unico punto donde los dos vocabularios se cruzan.
        if metered is not None and metered.state != OverlayState.APPLIED:
            state = OverlayState(metered.state.value)
            reasons = metered.warnings
        else:
            state = OverlayState.DISABLED_ERROR
            reasons = ()
        return (
            disabled_overlay(state, reasons=(*reasons, str(failure))),
            {},
            _headline_counters(counts, sent=0),
        )
    finally:
        if cache is not None:
            cache.close()
    return (
        overlay_from_extraction(extraction),
        {PROMPT_TEMPLATE_NAME: extraction.prompt_hash},
        _headline_counters(counts, sent=counts.sent),
    )


def _headline_counters(counts: BatchCounts, *, sent: int) -> dict[str, int]:
    """Los conteos del lote que van al *manifest* de la sesion (tarea #129).

    Se publican **cinco** numeros y no cuatro porque ``prepared`` y ``sent`` **no** son lo mismo: el
    lote puede dejar titulares listos y que la capa LLM no llegue a llamar (sin clave, con el
    presupuesto agotado o con el proveedor caido). Confundirlos falsearia el **coste por titular**
    de #117 en la direccion contraria a la util: se dividiria el gasto entre titulares que nunca se
    enviaron.

    ``read == duplicates + out_of_window + prepared`` cierra por construccion
    (:class:`BatchCounts`); ``sent <= prepared``, y solo ``sent`` es «lo que costo dinero».
    """
    return {
        "headlines_read": counts.read,
        "headlines_duplicates": counts.duplicates,
        "headlines_out_of_window": counts.out_of_window,
        "headlines_prepared": counts.sent,
        "headlines_sent": sent,
    }


def _record(
    journal_root: Path,
    *,
    session: date,
    as_of: datetime,
    status: GateStatus,
    features_version: str,
    model_version: str,
    git_commit: str,
    report_text: str,
    output: GateOutput | None = None,
    prob_up_raw: float | None = None,
    overlay: OverlayDecision | None = None,
    prompt_hashes: Mapping[str, object] | None = None,
    extra_blockers: Sequence[str] = (),
) -> None:
    """Escribe la fila de ``journal.decisions`` con la capa de #39 (esquema cerrado, inmutable).

    ``report_text`` es el informe **tal cual se emitio** (el mismo texto que va a stdout/stderr),
    persistido verbatim (§19.3). ``prob_up_raw`` solo lo aporta el estado ``recommendation`` (el
    gate no lo calcula); en los estados "no se" y ``error`` va ``null``. ``overlay`` y
    ``prompt_hashes`` son lo que hizo el overlay del LLM ese dia (#35): sin overlay declarado, la
    columna va ``null`` y el mapa vacio, nunca un valor inventado. ``extra_blockers`` son los
    eventos del dia que bloquean y **no** son reglas del gate (p. ej. un resultado de mega-cap
    confirmado, #126): se anaden a los codigos que ya derivó el gate.
    """
    payload = build_decision(
        trade_date=session,
        as_of=as_of,
        features_version=features_version,
        model_version=model_version,
        git_commit=git_commit,
        report_text=report_text,
        status=status,
        output=output,
        prob_up_raw=prob_up_raw,
        llm_overlay=None if overlay is None else overlay.state.value,
        prompt_hashes=prompt_hashes,
    )
    if extra_blockers:
        payload["blocking_events"] = [
            *cast("list[str]", payload["blocking_events"]),
            *extra_blockers,
        ]
    Journal(journal_root).write("decisions", payload)


def _record_overlay_signal(
    journal_root: Path,
    *,
    session: date,
    overlay: OverlayDecision,
    output: GateOutput,
) -> None:
    """Registra la variante «con overlay» en `journal.agent_signals` (#149).

    Bajo §19.19 (#148) la recomendacion que se **publica** se emite **sin** overlay; esta fila
    conserva el efecto del overlay —estado, `veto`, su motivo y la probabilidad **ajustada**— para
    el registro prospectivo de la Fase 4 (§19.9). Solo se escribe cuando el overlay produce decision
    (`applied` o `veto`): en `disabled_*` no hay senal que registrar y basta `llm_overlay`.
    """
    if overlay.state not in (OverlayState.APPLIED, OverlayState.VETO):
        return
    veto = overlay.state is OverlayState.VETO
    Journal(journal_root).write(
        "agent_signals",
        {
            "trade_date": session.isoformat(),
            "agent": NEWS_OVERLAY_AGENT,
            "prob_up": output.prob_up_calibrated,
            "confidence": None,
            "veto": veto,
            "veto_reason": "; ".join(overlay.reasons) if veto else None,
            "evidence": {
                "state": overlay.state.value,
                "adjustment_pct": overlay.adjustment_pct,
                "direction": None if output.direction is None else output.direction.value,
                "status": output.status.value,
                "reasons": list(overlay.reasons),
                "prompt_hash": overlay.prompt_hash,
            },
        },
    )


def _record_or_report(
    journal_root: Path,
    *,
    session: date,
    as_of: datetime,
    status: GateStatus,
    features_version: str,
    model_version: str,
    git_commit: str,
    report_text: str,
    output: GateOutput | None = None,
    prob_up_raw: float | None = None,
    overlay: OverlayDecision | None = None,
    prompt_hashes: Mapping[str, object] | None = None,
    extra_blockers: Sequence[str] = (),
) -> int | None:
    """Registra la fila; si el diario no la acepta, publica el motivo y devuelve ``2``.

    Un diario no escribible (raiz invalida, o la identidad de la sesion ya registrada con **otro**
    contenido: ``JournalRewriteError``) convierte la ejecucion en un error visible: sin diario no
    hay auditoria (§19.1), asi que **no** se emite la pista en silencio.
    """
    try:
        _record(
            journal_root,
            session=session,
            as_of=as_of,
            status=status,
            features_version=features_version,
            model_version=model_version,
            git_commit=git_commit,
            report_text=report_text,
            output=output,
            prob_up_raw=prob_up_raw,
            overlay=overlay,
            prompt_hashes=prompt_hashes,
            extra_blockers=extra_blockers,
        )
    except DecisionLogError as error:
        print(f"no se puede registrar la decision en el diario: {error}", file=sys.stderr)
        return 2
    return None


def _fail_with_error(
    journal_root: Path,
    *,
    session: date,
    as_of: datetime,
    snapshot_session: date | None,
    model_source: str,
    features_version: str,
    model_version: str,
    git_commit: str,
    message: str,
) -> int:
    """Estado ``error``: registra la fila, imprime el informe por ``stderr`` y devuelve ``2``."""
    text = render(
        status=GateStatus.ERROR,
        session=session,
        as_of=as_of,
        snapshot_session=snapshot_session,
        model_source=model_source,
        message=message,
    )
    failure = _record_or_report(
        journal_root,
        session=session,
        as_of=as_of,
        status=GateStatus.ERROR,
        features_version=features_version,
        model_version=model_version,
        git_commit=git_commit,
        report_text=text,
    )
    if failure is not None:
        return failure
    print(text, file=sys.stderr)
    return 2


# ─────────────────────────────────────────────────────────────────────────────
# Registro: elegir el modelo por `variant_id` (#111)
# ─────────────────────────────────────────────────────────────────────────────
def resolve_run(*, runs_root: Path, variant_id: str) -> RegistryEntry:
    """Resuelve el `run_sha256` de una variante del registro, sin teclearlo (#111).

    El registro lo lee ``analysis.experiment_log.load_registry`` (**importada**, no
    reimplementada): este modulo no abre `config.json`/`result.json` ni calcula `registry_sha256`.
    Criterio declarado (``VARIANT_SELECTION_RULE``): entre las entradas con ese `variant_id` gana
    la de mayor `sharpe_per_session` y, a igualdad, el `run_sha256` menor. Un `variant_id` que no
    sea una de las familias soportadas, o que el registro no traiga, es error tipado.
    """
    if variant_id not in SUPPORTED_VARIANTS:
        raise RegistrySelectionError(
            f"`{variant_id}` no es un `variant_id` soportado por el camino diario "
            f"({list(SUPPORTED_VARIANTS)}): solo se puntuan las familias publicadas por "
            "`analysis.model_comparison`"
        )
    try:
        registry = load_registry(runs_root)
    except ExperimentLogError as error:
        raise RegistrySelectionError(
            f"no se pudo leer el registro de `{runs_root}`: {error}"
        ) from error
    candidates = [entry for entry in registry.entries if entry.variant_id == variant_id]
    if not candidates:
        published = sorted({entry.variant_id for entry in registry.entries})
        raise RegistrySelectionError(
            f"el registro de `{runs_root}` no trae ninguna variante `{variant_id}`: publica "
            f"{published!r} ({VARIANT_SELECTION_RULE})"
        )
    return min(candidates, key=lambda entry: (-entry.sharpe_per_session, entry.run_sha256))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_as_of(value: str | None) -> datetime:
    """El instante ISO-8601 **con zona horaria** declarado por el llamante (sin reloj interno)."""
    if value is None or not value.strip():
        raise DeliveryError("--as-of es obligatorio (ISO-8601 con zona horaria)")
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as error:
        raise DeliveryError(f"--as-of no es ISO-8601: {value!r}") from error
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise DeliveryError(f"--as-of tiene que declarar zona horaria: {value!r}")
    return moment


def main(argv: Sequence[str] | None = None) -> int:
    """Imprime la pista del dia o su estado «sin recomendacion»; devuelve el codigo de salida.

    Codigos: ``0`` = informe emitido (``recommendation``, ``no_recommendation_stale_data`` o
    ``no_recommendation_data_quality``; un dia de mercado cerrado es ``recommendation`` con
    ``NOTHING`` justificado, regla 19); ``2`` = ``--as-of``/``--model-run``/``--journal-root``/
    ``--git-commit`` ausentes o invalidos, fallo del pipeline (``error``) o diario no escribible,
    con el motivo por ``stderr`` y sin *traceback*. Cada ejecucion registra la fila del diario
    (§19.3) antes de devolver, tambien en un dia de mercado cerrado.

    Ademas escribe su **traza estructurada** (#43) bajo ``--observability-root`` (por defecto
    ``<journal-root>/ops``): ``<session>/run_log.jsonl`` (una linea por etapa) y
    ``<session>/manifest.json`` (versiones y hashes). Un fallo de una etapa queda en la traza
    con su *traceback* completo y no se pierde.
    """
    parser = argparse.ArgumentParser(
        prog="cfdtrader.delivery.run_daily",
        description="Pista diaria (features -> probabilidad calibrada -> gate -> informe)",
    )
    parser.add_argument(
        "--as-of", default=None, help="instante declarado ISO-8601 con zona (obligatorio)"
    )
    parser.add_argument(
        "--model-run",
        default=None,
        help="run_sha256 de la corrida de `runs/` (alternativo a --variant-id)",
    )
    parser.add_argument(
        "--variant-id",
        default=None,
        help="variant_id resuelto contra `--runs-root` (alternativo a --model-run)",
    )
    parser.add_argument(
        "--journal-root", default=None, help="raiz del diario de decisiones (obligatorio)"
    )
    parser.add_argument(
        "--git-commit",
        default=None,
        help="git rev-parse HEAD, inyectado por el llamante (obligatorio; el modulo no lee git)",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="raiz del almacen")
    parser.add_argument("--settings", type=Path, default=None, help="ruta de settings.yaml")
    parser.add_argument("--calendar", type=Path, default=None, help="ruta de calendar.yaml")
    parser.add_argument(
        "--runs-root", type=Path, default=Path("runs"), help="raiz del registro de experimentos"
    )
    parser.add_argument(
        "--observability-root",
        type=Path,
        default=None,
        help="raiz de la traza estructurada (run_log/manifest); por defecto `<journal-root>/ops`",
    )
    parser.add_argument(
        "--premarket-bars",
        type=Path,
        default=None,
        help="barras capturadas del pre-mercado del ES (#141): sin ellas, la seccion no aparece",
    )
    args = parser.parse_args(argv)

    try:
        moment = _parse_as_of(cast("str | None", args.as_of))
    except DeliveryError as error:
        print(f"no se puede emitir la pista diaria: {error}", file=sys.stderr)
        return 2
    model_run = cast("str | None", args.model_run)
    variant_id = cast("str | None", args.variant_id)
    declared_run = model_run if model_run is not None and model_run.strip() else None
    declared_variant = variant_id if variant_id is not None and variant_id.strip() else None
    if declared_run is not None and declared_variant is not None:
        print(
            "no se puede emitir la pista diaria: declara `--model-run` o `--variant-id`, no los "
            "dos",
            file=sys.stderr,
        )
        return 2
    if declared_run is None and declared_variant is None:
        print(
            "no se puede emitir la pista diaria: --model-run o --variant-id es obligatorio",
            file=sys.stderr,
        )
        return 2
    journal_root_arg = cast("str | None", args.journal_root)
    if journal_root_arg is None or not journal_root_arg.strip():
        print(
            "no se puede emitir la pista diaria: --journal-root es obligatorio (sin diario no "
            "hay auditoria, §19.1)",
            file=sys.stderr,
        )
        return 2
    git_commit = cast("str | None", args.git_commit)
    if git_commit is None or not git_commit.strip():
        print(
            "no se puede emitir la pista diaria: --git-commit es obligatorio (el modulo no lee "
            "git; p. ej. `git rev-parse HEAD`)",
            file=sys.stderr,
        )
        return 2

    journal_root = Path(journal_root_arg)
    # La traza estructurada (#43) vive bajo el diario por defecto. La guardia de obsolescencia
    # lee `decisions/` con `read_decisions`, asi que `ops/` no la contamina.
    observability_root_arg = cast("Path | None", args.observability_root)
    observability_root = (
        observability_root_arg
        if observability_root_arg is not None
        else journal_root / OBSERVABILITY_DIRNAME
    )
    session = moment.astimezone(EASTERN).date()
    observer = RunObserver(
        observability_root,
        run_id=session.isoformat(),
        as_of=moment,
        git_commit=git_commit,
    )
    with observer:
        return _deliver(
            observer,
            moment=moment,
            declared_run=declared_run,
            declared_variant=declared_variant,
            journal_root=journal_root,
            observability_root=observability_root,
            git_commit=git_commit,
            settings_path=cast("Path | None", args.settings),
            calendar_path=cast("Path | None", args.calendar),
            data_root_arg=cast("Path | None", args.data_root),
            runs_root=Path(args.runs_root),
            premarket_bars=cast("Path | None", args.premarket_bars),
        )


def _deliver(
    observer: RunObserver,
    *,
    moment: datetime,
    declared_run: str | None,
    declared_variant: str | None,
    journal_root: Path,
    observability_root: Path,
    git_commit: str,
    settings_path: Path | None,
    calendar_path: Path | None,
    data_root_arg: Path | None,
    runs_root: Path,
    premarket_bars: Path | None = None,
) -> int:
    """Ejecuta el pipeline de una sesion y devuelve su codigo de salida (ver ``main``).

    Los parametros ya vienen validados por ``main``. ``observer`` registra las etapas y, al
    salir, escribe el ``run_log`` y el ``manifest`` **tambien si una etapa lanza** (#43).
    """

    try:
        settings = load_settings(settings_path)
        calendar: MarketCalendar = load_calendar(calendar_path)
    except ConfigurationError as error:
        observer.record("config", ok=False, error=str(error))
        print(f"no se puede emitir la pista diaria: {error}", file=sys.stderr)
        return 2

    data_root = Path(data_root_arg) if data_root_arg is not None else Path(settings.data.root)
    # La ruta se conoce de antemano con `--model-run`; con `--variant-id` la decide el registro, y
    # hasta entonces el informe declara el selector sin resolver (nunca un digest inventado).
    model_path = (
        runs_root / declared_run / MODEL_FILE
        if declared_run is not None
        else runs_root / MODEL_FILE
    )
    model_source = (
        str(model_path)
        if declared_run is not None
        else f"variant_id {declared_variant!r} (sin resolver contra {runs_root})"
    )
    as_of_et = moment.astimezone(EASTERN)
    session = as_of_et.date()
    snapshot_session: date | None = None
    # La version del snapshot de features: se conoce en cuanto la matriz se construye; si no se
    # llega a construir (estado `error`), se declara la ausencia en vez de inventar un digest.
    features_version = FEATURES_VERSION_UNAVAILABLE
    # La probabilidad cruda solo existe si el modelo se pudo puntuar (estado `recommendation`).
    prob_up_raw: float | None = None
    # Los eventos del dia (informativos incluidos) se publican en el informe; si la senal del
    # calendario no se puede calcular, el informe sale **sin** su seccion en vez de caerse (#121).
    calendar_events: EventCalendarSignal | None = None
    # Las publicaciones macro del dia (informativas, sin bloqueo): parte (b) de #114 (#125).
    publications: tuple[MacroPublication, ...] = ()
    # Los resultados de mega-caps del dia: parte (c) de #114 (#126). Solo los **confirmados**
    # bloquean; su codigo entra en `blocking_events` sin que el gate gane una regla nueva (§19.8).
    earnings: tuple[EarningsEvent, ...] = ()
    # El conjunto de FOMC declarado y su procedencia (parte (a) de #114). Un año no declarado deja
    # el conjunto vacio **y una nota**: lo que no puede volver es el silencio del literal `()`.
    fomc_dates: tuple[date, ...] = ()
    fomc_note = ""
    fomc_declared = False
    # El `model_version` del diario: el `run_sha256` resuelto, o el selector declarado verbatim
    # mientras la resolucion contra el registro no lo haya devuelto (nunca un digest inventado).
    model_version = cast("str", declared_run if declared_run is not None else declared_variant)
    # El contador de llamadas del LLM es **uno** para toda la ejecucion: la extraccion de noticias
    # (#35) y la redaccion (#37) comparten el mismo instante, y con contadores independientes
    # producirian el mismo `call_id` —identidad del diario, que es inmutable— y se perderia la
    # segunda fila con `JournalRewriteError` (#147).
    sequence = CallSequence()

    try:
        if declared_variant is not None:
            with observer.stage("resolve-model"):
                entry = resolve_run(runs_root=runs_root, variant_id=declared_variant)
            model_version = entry.run_sha256
            model_path = runs_root / entry.run_sha256 / MODEL_FILE
            model_source = f"{model_path} (variant_id: {entry.variant_id})"
        store = Store(data_root)
        with observer.stage("features"):
            matrix = build_feature_matrix(store)
            features_version = FEATURE_VERSION_PREFIX + matrix.matrix_sha256
            snapshot_session = matrix.last_session
            guard = session_guard(
                as_of=moment,
                calendar=calendar,
                snapshot_session=snapshot_session,
                executions=execution_dates(journal_root),
            )
        observer.add_version("features_version", features_version)
        observer.add_version("model_version", model_version)
        # Las publicaciones macro y los resultados de mega-caps se leen del almacen; nunca de
        # la red (el camino diario no llama a `yfinance`).
        with observer.stage("macro-publications"):
            publications = _day_publications(store, session, moment)
            earnings = _day_earnings(store, session, moment)
        earnings_blockers = _earnings_blockers(earnings)
        # El gap de pre-mercado del ES (#141): un dato **declarado** de la decision, nunca una
        # feature. El camino diario **no** descarga: lee las barras que capturo el CLI de
        # `premarket_gap` a las 08:45 (`--premarket-bars`). Sin el fichero, la seccion no aparece.
        premarket: Mapping[str, object] | None = None
        if premarket_bars is not None:
            with observer.stage("premarket"):
                try:
                    captured = premarket_gap.load_bars(premarket_bars)
                    premarket = premarket_gap.analyse(
                        store=store,
                        as_of=moment,
                        fetcher=_captured_fetcher(captured),
                    ).payload()
                except premarket_gap.PremarketGapError as error:
                    print(f"no se puede medir el gap de pre-mercado: {error}", file=sys.stderr)
        if guard.blocks and guard.verdict is not GuardVerdict.MARKET_CLOSED:
            # Aqui solo llegan los dos "no se" de frescura. La clausura (#113, MARKET_CLOSED) **no**
            # para el camino: el gate la convierte en un `NOTHING` justificado (regla 19) que si se
            # registra, con el `observation_sessions_remaining` = 0 que la guardia ya derivo.
            text = render(
                status=GateStatus.NO_RECOMMENDATION_STALE_DATA,
                session=session,
                as_of=moment,
                snapshot_session=snapshot_session,
                model_source=model_source,
                message=_guard_message(guard),
                publications=publications,
                earnings=earnings,
                premarket=premarket,
            )
            with observer.stage("journal"):
                failure = _record_or_report(
                    journal_root,
                    session=session,
                    as_of=moment,
                    status=GateStatus.NO_RECOMMENDATION_STALE_DATA,
                    features_version=features_version,
                    model_version=model_version,
                    git_commit=git_commit,
                    report_text=text,
                    extra_blockers=earnings_blockers,
                )
            if failure is not None:
                return failure
            print(text)
            return 0
        row = _last_row(matrix)
        problem = _row_problem(row)
        if problem is not None:
            text = render(
                status=GateStatus.NO_RECOMMENDATION_DATA_QUALITY,
                session=session,
                as_of=moment,
                snapshot_session=snapshot_session,
                model_source=model_source,
                message=f"calidad de datos (tech_stack.md §8.4): {problem}; no se emite pista",
                publications=publications,
                earnings=earnings,
                premarket=premarket,
            )
            with observer.stage("journal"):
                failure = _record_or_report(
                    journal_root,
                    session=session,
                    as_of=moment,
                    status=GateStatus.NO_RECOMMENDATION_DATA_QUALITY,
                    features_version=features_version,
                    model_version=model_version,
                    git_commit=git_commit,
                    report_text=text,
                    extra_blockers=earnings_blockers,
                )
            if failure is not None:
                return failure
            print(text)
            return 0
        features = {name: float(cast("float", row[name])) for name in BASELINE_FEATURES}
        with observer.stage("predict"):
            prob_up_raw, probability = _predictions(model_path, features)
        move = _expected_move_pct(row)
        # #131: el supuesto de *slippage* se **cuantifica** con el `R` que el propietario decidio
        # en #60 (`SCENARIO_R_PCT`, la unica fuente del valor): 20 % de `R` = 20 bp. Sigue siendo
        # un supuesto —`assumed`, `is_measurement = false`—; lo que cambia es que el gate ya puede
        # verificar la regla 9 sobre el **coste declarado** (§19.12) y publicar el EV bajo el
        # supuesto como **sensibilidad** en vez de dejarlo en `null`.
        cost = cost_breakdown(
            model=declared_cost_model(),
            slippage=declared_slippage_assumption(SCENARIO_R_PCT),
            notional_usd=NOTIONAL_USD,
            side=Side.LONG,
            nights=0,
        )
        params = scenario_parameters(cost_pct=cost.c_declared_pct)
        stop_pct = SCENARIO_STOP_SIGMA_MULTIPLE * move
        with observer.stage("calendar"):
            calendar_events = _day_events(calendar, session, moment)
            fomc_dates, fomc_note, fomc_declared = _declared_fomc_dates(session.year)
            observer.add_version("fomc_calendar", fomc_note)
            if not fomc_declared:
                print(f"aviso: {fomc_note}", file=sys.stderr)
        if calendar_events is not None:
            observer.add_hash("calendar_sha256", calendar_events.signal_sha256)
        with observer.stage("overlay"):
            overlay, prompt_hashes, headline_counters = _compute_overlay(
                data_root, observability_root, moment, sequence=sequence
            )
        # El conteo del lote (#129) va al *manifest* de la sesion: es lo unico que permite publicar
        # el coste por titular y lo que la deduplicacion evito (#117, §6.3.5).
        for name, value in headline_counters.items():
            observer.add_counter(name, value)

        def _run_gate(declared_overlay: OverlayDecision | None) -> GateOutput:
            """El gate con ese overlay declarado; ``None`` = la recomendacion **sin** overlay."""
            return evaluate_gate(
                session=session,
                as_of=as_of_et,
                today=session,
                calendar=calendar,
                prob_up_calibrated=probability,
                expected_move_pct=move,
                expected_move_basis=EXPECTED_MOVE_BASIS,
                cost=cost,
                capital_usd=NOTIONAL_USD,
                snapshot_ok=True,
                stop_pct=stop_pct,
                target_pct=SCENARIO_TARGET_STOP_MULTIPLE * stop_pct,
                fomc_dates=fomc_dates,
                params=params,
                trades_today=0,
                daily_pnl_pct=None,
                weekly_pnl_pct=None,
                monthly_pnl_pct=None,
                observation_sessions_remaining=guard.observation_sessions_remaining,
                overlay=declared_overlay,
            )

        with observer.stage("gate"):
            # §19.19 (#148): la recomendacion que se **publica** se emite **sin** overlay. El
            # overlay se evalua aparte (`overlay_output`) y su efecto se conserva como la variante
            # «con overlay» en `agent_signals` (#149) para el registro prospectivo de §19.9.
            output = _run_gate(None)
            overlay_output = _run_gate(overlay)
    except (MissingModelError, UnsupportedModelError) as error:
        with observer.stage("journal"):
            return _fail_with_error(
                journal_root,
                session=session,
                as_of=moment,
                snapshot_session=snapshot_session,
                model_source=model_source,
                features_version=features_version,
                model_version=model_version,
                git_commit=git_commit,
                message=str(error),
            )
    except (DeliveryError, FeatureFrameError, ConfigurationError) as error:
        with observer.stage("journal"):
            return _fail_with_error(
                journal_root,
                session=session,
                as_of=moment,
                snapshot_session=snapshot_session,
                model_source=model_source,
                features_version=features_version,
                model_version=model_version,
                git_commit=git_commit,
                message=str(error),
            )

    with observer.stage("report"):
        report, report_hashes = _compose_report(
            output,
            observability_root,
            moment,
            publications=publications,
            earnings=earnings,
            calendar_events=calendar_events,
            sequence=sequence,
        )
    if report is not None:
        observer.add_hash("report_prompt_sha256", report.prompt_hash)
    observer.add_hash("gate_sha256", output.gate_sha256)
    text = render(
        status=output.status,
        session=session,
        as_of=moment,
        snapshot_session=snapshot_session,
        model_source=model_source,
        message=_guard_message(guard),
        output=output,
        calendar_events=calendar_events,
        publications=publications,
        earnings=earnings,
        report=report,
        premarket=premarket,
        notes=() if fomc_declared else (fomc_note,),
    )
    with observer.stage("journal"):
        failure = _record_or_report(
            journal_root,
            session=session,
            as_of=moment,
            status=output.status,
            features_version=features_version,
            model_version=model_version,
            git_commit=git_commit,
            report_text=text,
            output=output,
            prob_up_raw=prob_up_raw,
            overlay=overlay,
            prompt_hashes={**prompt_hashes, **report_hashes},
            extra_blockers=earnings_blockers,
        )
        if failure is None:
            # §19.19 (#148): la fila publicada va **sin** overlay; aqui se conserva la variante
            # «con overlay» (#149) para el registro prospectivo de la Fase 4 (§19.9).
            try:
                _record_overlay_signal(
                    journal_root, session=session, overlay=overlay, output=overlay_output
                )
            except DecisionLogError as error:
                print(
                    f"no se puede registrar la senal del overlay en el diario: {error}",
                    file=sys.stderr,
                )
                failure = 2
    if failure is not None:
        return failure
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
