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
consulta **en dos momentos**, porque no todos sus datos estan disponibles a la vez:

1. **Antes de tocar el almacen**, con ``market_closure``: si el mercado no abre ese dia (festivo
   de EE. UU. o fin de semana) imprime el aviso de ``closure_notice`` en stdout y **no ejecuta
   nada**: ni lee el almacen, ni puntua el modelo, ni emite informe, ni escribe fila. Es la rama
   "No se ejecuta" de §8.4: un mercado cerrado no es ``NOTHING`` (no hay sesion que evaluar) ni un
   "no se" (el calendario **si** sabe que no abre), asi que no se colapsa en ninguno de los cuatro
   estados.
2. **Despues de construir la matriz**, con ``session_guard``: el veredicto sale de comparar la
   ultima sesion del almacen con la anterior a la evaluada (``missing_previous_close`` y
   ``snapshot_ahead`` son ``no_recommendation_stale_data``; el segundo es el caso de un
   ``--as-of`` que no se corresponde con el almacen) y el **contador de observacion** de la regla
   15 se deriva del diario (``execution_dates`` + ``observation_sessions_remaining``): la vuelta de
   una ausencia de mas de una semana deja 5 sesiones por revalidar, que el gate convierte en
   ``NOTHING`` con su bloqueo 15.

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

import numpy as np
from lightgbm import Booster
from lightgbm.basic import LightGBMError

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
from cfdtrader.data.calendar import EASTERN, MarketCalendar, load_calendar
from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.store import Store
from cfdtrader.decision.gate import GateOutput, GateStatus, evaluate_gate
from cfdtrader.delivery.staleness import (
    SessionGuard,
    closure_notice,
    execution_dates,
    market_closure,
    session_guard,
)
from cfdtrader.features.store import FEATURE_VERSION_PREFIX
from cfdtrader.journal.decision_log import DecisionLogError, Journal, build_decision
from cfdtrader.models.baseline import BASELINE_FEATURES
from cfdtrader.models.calibration import Calibration, sigmoid

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

#: Valla de honestidad (#109): viaja en **las cuatro** salidas, sin excepcion.
HONESTY_FENCE: Final[tuple[str, ...]] = (
    "no hay edge demostrado",
    "ejecucion: manual (el sistema no coloca ordenes; las decide el operador)",
    "naturaleza: apoyo a la decision, no una estrategia validada",
    f"escenario declarado: {SCENARIO_LABEL} (scenario_parameters); #59 y #60 siguen OPEN",
)

#: Claves del bloque de la pista que el modelo lineal tiene que publicar en su ultimo fold.
_LINEAR_FOLD_KEYS: Final[tuple[str, ...]] = ("mean", "scale", "coefficients", "intercept")

#: ``features_version`` declarada cuando el almacen **no** pudo construir la matriz (estado
#: ``error``): una ausencia declarada, nunca un digest inventado.
FEATURES_VERSION_UNAVAILABLE: Final[str] = "unavailable"

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
def render(
    *,
    status: GateStatus,
    session: date | None,
    as_of: datetime,
    snapshot_session: date | None,
    model_source: str,
    message: str,
    output: GateOutput | None = None,
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
) -> None:
    """Escribe la fila de ``journal.decisions`` con la capa de #39 (esquema cerrado, inmutable).

    ``report_text`` es el informe **tal cual se emitio** (el mismo texto que va a stdout/stderr),
    persistido verbatim (§19.3). ``prob_up_raw`` solo lo aporta el estado ``recommendation`` (el
    gate no lo calcula); en los estados "no se" y ``error`` va ``null``.
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
    )
    Journal(journal_root).write("decisions", payload)


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
    ``no_recommendation_data_quality``) o mercado cerrado (la rama "no se ejecuta" de §8.4: aviso
    con su motivo, sin informe y sin fila); ``2`` = ``--as-of``/``--model-run``/``--journal-root``/
    ``--git-commit`` ausentes o invalidos, fallo del pipeline (``error``) o diario no escribible,
    con el motivo por ``stderr`` y sin *traceback*. Cada ejecucion registra la fila del diario
    (§19.3) antes de devolver, salvo cuando no hay sesion que registrar.
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

    try:
        settings = load_settings(args.settings)
        calendar: MarketCalendar = load_calendar(args.calendar)
    except ConfigurationError as error:
        print(f"no se puede emitir la pista diaria: {error}", file=sys.stderr)
        return 2

    data_root = Path(args.data_root) if args.data_root is not None else Path(settings.data.root)
    runs_root = Path(args.runs_root)
    journal_root = Path(journal_root_arg)
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
    # El `model_version` del diario: el `run_sha256` resuelto, o el selector declarado verbatim
    # mientras la resolucion contra el registro no lo haya devuelto (nunca un digest inventado).
    model_version = cast("str", declared_run if declared_run is not None else declared_variant)

    # La guardia de §8.4 empieza **antes de leer nada**: si ese dia el mercado americano no abre,
    # el camino diario "no se ejecuta" (ni almacen, ni registro, ni modelo, ni informe, ni fila).
    closure = market_closure(as_of=moment, calendar=calendar)
    if closure is not None:
        print(closure_notice(session=session, reason=closure))
        return 0

    try:
        if declared_variant is not None:
            entry = resolve_run(runs_root=runs_root, variant_id=declared_variant)
            model_version = entry.run_sha256
            model_path = runs_root / entry.run_sha256 / MODEL_FILE
            model_source = f"{model_path} (variant_id: {entry.variant_id})"
        matrix = build_feature_matrix(Store(data_root))
        features_version = FEATURE_VERSION_PREFIX + matrix.matrix_sha256
        snapshot_session = matrix.last_session
        guard = session_guard(
            as_of=moment,
            calendar=calendar,
            snapshot_session=snapshot_session,
            executions=execution_dates(journal_root),
        )
        if guard.blocks:
            # Aqui solo pueden llegar los dos "no se" de frescura: la clausura ya se comprobo
            # arriba, con el mismo `as_of` y el mismo calendario.
            text = render(
                status=GateStatus.NO_RECOMMENDATION_STALE_DATA,
                session=session,
                as_of=moment,
                snapshot_session=snapshot_session,
                model_source=model_source,
                message=_guard_message(guard),
            )
            failure = _record_or_report(
                journal_root,
                session=session,
                as_of=moment,
                status=GateStatus.NO_RECOMMENDATION_STALE_DATA,
                features_version=features_version,
                model_version=model_version,
                git_commit=git_commit,
                report_text=text,
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
            )
            failure = _record_or_report(
                journal_root,
                session=session,
                as_of=moment,
                status=GateStatus.NO_RECOMMENDATION_DATA_QUALITY,
                features_version=features_version,
                model_version=model_version,
                git_commit=git_commit,
                report_text=text,
            )
            if failure is not None:
                return failure
            print(text)
            return 0
        features = {name: float(cast("float", row[name])) for name in BASELINE_FEATURES}
        prob_up_raw, probability = _predictions(model_path, features)
        move = _expected_move_pct(row)
        cost = cost_breakdown(
            model=declared_cost_model(),
            slippage=declared_slippage_assumption(),
            notional_usd=NOTIONAL_USD,
            side=Side.LONG,
            nights=0,
        )
        params = scenario_parameters(cost_pct=cost.c_declared_pct)
        stop_pct = SCENARIO_STOP_SIGMA_MULTIPLE * move
        output = evaluate_gate(
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
            fomc_dates=(),
            params=params,
            trades_today=0,
            daily_pnl_pct=None,
            weekly_pnl_pct=None,
            monthly_pnl_pct=None,
            observation_sessions_remaining=guard.observation_sessions_remaining,
        )
    except (MissingModelError, UnsupportedModelError) as error:
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

    text = render(
        status=output.status,
        session=session,
        as_of=moment,
        snapshot_session=snapshot_session,
        model_source=model_source,
        message=_guard_message(guard),
        output=output,
    )
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
    )
    if failure is not None:
        return failure
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
