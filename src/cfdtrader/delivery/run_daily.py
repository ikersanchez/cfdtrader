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

Modelo: familia lineal, sin reajustar
-------------------------------------

El modelo vive en ``runs/<run_sha256>/model.json``. La probabilidad es la del **ultimo fold**
(``folds[-1]``) aplicado a la fila nueva, seguida de su calibrador: `((x - mean) / scale) @
coefficients + intercept`, y despues ``Calibration.calibrate`` (o ``sigmoid`` si el metodo es
``"none"``). **No se reajusta** y **no** se consulta ``test_positions``: es una
**extrapolacion** mas alla de la ventana de *test* del fold, declarada aqui a proposito. Cualquier
otra familia (p. ej. el *booster* de LightGBM, con ``library``/``booster_model``) es un
``UnsupportedModelError`` tipado: LightGBM en el camino diario es #111.

Los cuatro estados de §19.2 y la guardia de §8.4
------------------------------------------------

La salida es uno de los cuatro estados de ``plan.md`` §19.2 y el informe imprime su ``value``
literal: ``recommendation``, ``no_recommendation_stale_data``, ``no_recommendation_data_quality``
y ``error``. El quinto estado del gate, ``no_recommendation_undecided``, no aparece porque el
escenario declarado S1 cierra los once parametros (``scenario_parameters``). Antes de tocar el
modelo, la guardia de obsolescencia de ``tech_stack.md`` §8.4 exige que la ultima sesion del
almacen sea exactamente la anterior (si no, ``no_recommendation_stale_data``) y que la fila
tenga las diez features y un ``garch_forecast`` positivo (si no, el estado
``no_recommendation_data_quality``). El contador de ausencia, el modo observacion y los
festivos/medias sesiones completos siguen siendo **#40 (OPEN)**: aqui solo se consume el minimo
que impide que un dato viejo produzca una pista.

Sin reloj, sin red y sin escribir nada
--------------------------------------

El modulo no consulta el reloj (``--as-of`` es obligatorio, ISO-8601 con zona), no importa
``yfinance``/``requests``/``urllib`` y no escribe en el almacen ni en el registro: solo lee el
diario y ``runs/<run_sha256>/model.json``. Mismas entradas ⇒ misma salida byte a byte.
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

from cfdtrader.analysis.backtest_report import NOTIONAL_USD
from cfdtrader.analysis.feature_frame import (
    FeatureFrameError,
    FeatureMatrix,
    build_feature_matrix,
)
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
from cfdtrader.models.baseline import BASELINE_FEATURES
from cfdtrader.models.calibration import Calibration, sigmoid

__all__ = [
    "DeliveryError",
    "MissingModelError",
    "UnsupportedModelError",
    "main",
    "predict",
    "render",
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


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados
# ─────────────────────────────────────────────────────────────────────────────
class DeliveryError(Exception):
    """Raiz de los errores del camino diario."""


class MissingModelError(DeliveryError):
    """No existe el ``model.json`` de la corrida declarada."""


class UnsupportedModelError(DeliveryError):
    """El documento no es la familia lineal soportada (p. ej. LightGBM, #111)."""


# ─────────────────────────────────────────────────────────────────────────────
# Prediccion: el ultimo fold aplicado a la fila nueva, con su calibrador
# ─────────────────────────────────────────────────────────────────────────────
def predict(model_path: Path, features: Mapping[str, float]) -> float:
    """Probabilidad **calibrada** de la fila nueva con el ultimo fold del modelo lineal.

    Lee ``model["folds"][-1]`` y devuelve el score ``((x - mean) / scale) @ coefficients +
    intercept`` pasado por el calibrador publicado (o por ``sigmoid`` si el metodo es
    ``"none"``). No reajusta nada y es determinista. Un documento ausente es
    :class:`MissingModelError`; uno que no sea la familia lineal (LightGBM) o que no declare
    exactamente las diez columnas de ``BASELINE_FEATURES`` es :class:`UnsupportedModelError`.
    """
    document = _load_document(model_path)
    model = cast("Mapping[str, object]", document["model"])
    if "library" in model:
        raise UnsupportedModelError(
            f"{model_path}: el documento publica `library` (LightGBM); el camino diario solo "
            "soporta la familia lineal de `models.baseline` (#111)"
        )
    if "features" not in model or "folds" not in model:
        raise UnsupportedModelError(
            f"{model_path}: el documento no declara `features`/`folds`: no es la familia lineal"
        )
    declared = tuple(str(name) for name in cast("Sequence[object]", model["features"]))
    if declared != BASELINE_FEATURES:
        raise UnsupportedModelError(
            f"{model_path}: el documento declara {declared!r} y la familia lineal exige "
            "exactamente las 10 columnas de `models.baseline.BASELINE_FEATURES`"
        )
    folds = cast("Sequence[object]", model["folds"])
    if not folds:
        raise UnsupportedModelError(f"{model_path}: el modelo no publica ningun fold")
    fold = cast("Mapping[str, object]", folds[-1])
    return _calibrated_probability(fold, features, declared, model_path=model_path)


def _calibrated_probability(
    fold: Mapping[str, object],
    features: Mapping[str, float],
    declared: tuple[str, ...],
    *,
    model_path: Path,
) -> float:
    """El score del ultimo fold pasado por su calibrador (``sigmoid`` si no calibra)."""
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
    block = fold.get("calibration")
    if block is None:
        return sigmoid(score)
    calibrator = _calibration_from_payload(cast("Mapping[str, object]", block))
    if not calibrator.calibrated:
        return sigmoid(score)
    value = calibrator.calibrate([score])[0]
    return sigmoid(score) if value is None else float(value)


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
    """El informe del dia: cabecera, pista (si la hay), motivo y valla de honestidad (#109).

    Los cuatro estados salen con el mismo formato y **todos** imprimen la valla: la salida no
    puede parecer una estrategia validada ni ocultar que la ejecucion es manual.
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


def _date_or_null(value: date | None) -> str:
    """Una fecha ISO, o ``null`` sin inventar un valor."""
    return "null" if value is None else value.isoformat()


def _decimal_or_null(value: Decimal | None) -> str:
    """Un ``Decimal`` exacto en forma decimal, o ``null`` (nunca un ``0`` de relleno)."""
    return "null" if value is None else format(value, "f")


# ─────────────────────────────────────────────────────────────────────────────
# La guardia de obsolescencia (§8.4) y el movimiento esperado
# ─────────────────────────────────────────────────────────────────────────────
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
    ``no_recommendation_data_quality``); ``2`` = ``--as-of``/``--model-run`` ausentes o
    invalidos, o fallo del pipeline (``error``), con el motivo por ``stderr`` y sin *traceback*.
    """
    parser = argparse.ArgumentParser(
        prog="cfdtrader.delivery.run_daily",
        description="Pista diaria (features -> probabilidad calibrada -> gate -> informe)",
    )
    parser.add_argument(
        "--as-of", default=None, help="instante declarado ISO-8601 con zona (obligatorio)"
    )
    parser.add_argument(
        "--model-run", default=None, help="run_sha256 de la corrida de `runs/` (obligatorio)"
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
    if model_run is None or not model_run.strip():
        print(
            "no se puede emitir la pista diaria: --model-run es obligatorio",
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
    model_path = runs_root / model_run / MODEL_FILE
    model_source = str(model_path)
    as_of_et = moment.astimezone(EASTERN)
    session = as_of_et.date()
    snapshot_session: date | None = None

    try:
        previous = calendar.previous_session(session)
        matrix = build_feature_matrix(Store(data_root))
        snapshot_session = matrix.last_session
        if snapshot_session != previous:
            print(
                render(
                    status=GateStatus.NO_RECOMMENDATION_STALE_DATA,
                    session=session,
                    as_of=moment,
                    snapshot_session=snapshot_session,
                    model_source=model_source,
                    message=(
                        f"datos obsoletos (tech_stack.md §8.4): la ultima sesion del almacen es "
                        f"{snapshot_session.isoformat()} y la sesion anterior a "
                        f"{session.isoformat()} es {previous.isoformat()}; no se emite pista"
                    ),
                )
            )
            return 0
        row = _last_row(matrix)
        problem = _row_problem(row)
        if problem is not None:
            print(
                render(
                    status=GateStatus.NO_RECOMMENDATION_DATA_QUALITY,
                    session=session,
                    as_of=moment,
                    snapshot_session=snapshot_session,
                    model_source=model_source,
                    message=f"calidad de datos (tech_stack.md §8.4): {problem}; no se emite pista",
                )
            )
            return 0
        features = {name: float(cast("float", row[name])) for name in BASELINE_FEATURES}
        probability = predict(model_path, features)
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
            observation_sessions_remaining=0,
        )
    except (MissingModelError, UnsupportedModelError) as error:
        print(
            render(
                status=GateStatus.ERROR,
                session=session,
                as_of=moment,
                snapshot_session=snapshot_session,
                model_source=model_source,
                message=str(error),
            ),
            file=sys.stderr,
        )
        return 2
    except (DeliveryError, FeatureFrameError, ConfigurationError) as error:
        print(
            render(
                status=GateStatus.ERROR,
                session=session,
                as_of=moment,
                snapshot_session=snapshot_session,
                model_source=model_source,
                message=str(error),
            ),
            file=sys.stderr,
        )
        return 2

    print(
        render(
            status=output.status,
            session=session,
            as_of=moment,
            snapshot_session=snapshot_session,
            model_source=model_source,
            message=(
                f"pista evaluada con el escenario declarado {SCENARIO_LABEL} y coste declarado"
            ),
            output=output,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
