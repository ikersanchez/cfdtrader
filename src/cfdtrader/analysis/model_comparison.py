"""Comparacion **medida** de la familia lineal y la familia LightGBM, con regla pre-declarada (#26).

Cierra la pregunta que #24/#25 dejaron abierta: **que modelo es el definitivo de la Fase 2**.
No lo decide una preferencia, lo decide una regla escrita antes de mirar los numeros
(:data:`SELECTION_RULE`): se comparan, bajo el protocolo **exacto** de #24/#25, las cuatro
variantes del registro —lineal cruda y calibrada, LightGBM cruda y calibrada— y se publica
la comparacion **medida** con el Deflated Sharpe y el PBO de las variantes reales.

Lo que **no** reimplementa (A3): el universo y el plan (`analysis.backtest_report`, #69), la
matriz de las cinco familias (`analysis.feature_frame`, #24), el modelo lineal y su
calibracion (`models.baseline`, `models.calibration`, #24/#25), el LightGBM reducido
(`models.lightgbm_model`, #26), el motor (`backtest.engine`, #13), el coste declarado
(`backtest.costs`, #8/#11), las metricas de probabilidad (`backtest.metrics`, #15), el
registro y la correccion por intentos (`analysis.experiment_log`, #16). El plan, el umbral,
los bins y el coste se **importan**: el payload los publica, no los redeclara (A3).

Las series por sesion salen del **motor** (``run_walk_forward``, como en #24): heredan su
convencion, incluido el desajuste de unidades de #80, que se publica medido en su propio
bloque y **no** se arregla aqui (arreglarlo es #80, y `backtest/engine.py` no se toca).

Determinismo (A13): el ``run_sha256`` sale de la configuracion de #16, el ``model_sha256`` y
el ``report_sha256`` del canonicamente hasheable de #13, y el modulo **no consulta el reloj**
(``as_of`` entra por parametro y el CLI lo exige).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import polars as pl
from loguru import logger

from cfdtrader.analysis import backtest_report
from cfdtrader.analysis.backtest_report import (
    PHASE1_PLAN,
    SERIES_ID,
    Universe,
    _calendar_years,  # pyright: ignore[reportPrivateUsage]
    build_inputs,
    build_split_plan,
    declared_slippage_assumption,
    load_history,
)
from cfdtrader.analysis.baseline_report import (
    CALIBRATION_BINS,
    MODEL_FILE,
    _decider,  # pyright: ignore[reportPrivateUsage]
    _label_horizon,  # pyright: ignore[reportPrivateUsage]
    _require_alignment,  # pyright: ignore[reportPrivateUsage]
    _scored_inputs,  # pyright: ignore[reportPrivateUsage]
    split_assignments,
)
from cfdtrader.analysis.experiment_log import (
    DEFAULT_RUNS_ROOT,
    DSR_HALVES,
    PBO_HALVES,
    ExperimentConfig,
    ExperimentLogError,
    ExperimentRecord,
    ExperimentResult,
    Registry,
    RegistryEntry,
    _write_immutable,  # pyright: ignore[reportPrivateUsage]
    aggregate_verdict,
    deflate_block,
    load_registry,
    pbo_block,
    record_experiment,
    require_consistent_aggregate,
    require_trials_match_registry,
)
from cfdtrader.analysis.feature_frame import (
    FEATURES_LIMITATIONS,
    FeatureFrame,
    FeatureFrameError,
    build_feature_frame,
)
from cfdtrader.backtest.costs import CostModel, SlippageParameter, declared_cost_model
from cfdtrader.backtest.engine import (
    STATUS_TRADED,
    BacktestRun,
    canonical_text,
    run_walk_forward,
)
from cfdtrader.backtest.metrics import (
    LOG_LOSS_EPSILON,
    brier_score,
    log_loss,
    sharpe_ratio,
)
from cfdtrader.backtest.splits import SplitPlan
from cfdtrader.data.calendar import load_calendar
from cfdtrader.data.settings import ConfigurationError, Settings, load_settings
from cfdtrader.data.store import Store, WriteOutcome
from cfdtrader.models.baseline import (
    BASELINE_FEATURES,
    DECISION_THRESHOLD,
    SEED,
)
from cfdtrader.models.calibration import (
    CALIBRATION_HYPERPARAMETERS,
    Calibration,
    method_counts,
    sigmoid,
)
from cfdtrader.models.lightgbm_model import (
    LIGHTGBM_HYPERPARAMETERS,
    LightGBMModel,
    calibrated_probabilities,
    fit_lightgbm,
    probabilities,
)

__all__ = [
    "BASELINE_VARIANT_ID",
    "CALIBRATION_RULE",
    "CLI_NAME",
    "FAMILY_ORDER",
    "FROZEN_BASELINE",
    "FROZEN_TOLERANCE",
    "PBO_BLOCKS",
    "PRIMARY_METRIC",
    "REPORT_PREFIX",
    "SELECTION_RULE",
    "TIE_BREAKERS",
    "TIE_TOLERANCE",
    "VARIANT_ID",
    "FrozenBaselineMismatchError",
    "MissingDecisionError",
    "ModelComparisonError",
    "ModelComparisonReport",
    "analyse",
    "main",
    "model_sha256",
    "render_markdown",
    "require_matrix_matches_registry",
    "selection_block",
]

#: Identidad registrada de la familia nueva: **una** por familia, como #24/#25 (#26, decision 4).
VARIANT_ID: Final[str] = "lightgbm_gbdt_v1"

#: Identidad de la familia lineal ya registrada, para poder reconstruirla y para la regla.
BASELINE_VARIANT_ID: Final[str] = "baseline_logit_elasticnet_v1"

#: Prefijo del informe: ``model_comparison_<AAAA-MM-DD>.{json,md}``.
REPORT_PREFIX: Final[str] = "model_comparison"

#: Nombre del CLI, para los mensajes de ``stderr``.
CLI_NAME: Final[str] = "cfdtrader.analysis.model_comparison"

#: Metrica **primaria** de la seleccion, declarada antes de mirar los numeros (A11).
PRIMARY_METRIC: Final[str] = "brier_score"

#: Desempates, **en orden**: primero la log-loss, despues el orden de familia (A11).
TIE_BREAKERS: Final[tuple[str, ...]] = ("log_loss", "family_order")

#: Orden de familia (A11): a igualdad de metrica gana la familia **mas simple**.
FAMILY_ORDER: Final[tuple[str, ...]] = (BASELINE_VARIANT_ID, VARIANT_ID)

#: Tolerancia de empate (A11): por debajo de esto dos cifras son la **misma** cifra.
TIE_TOLERANCE: Final[float] = 1e-12

#: Regla de seleccion declarada, publicada literal en el payload (A11).
SELECTION_RULE: Final[str] = (
    "minimo de `brier_score` entre los candidatos evaluados; empates (|a - b| <= 1e-12) se "
    "rompen con `log_loss` y, si siguen empatados, con `family_order` "
    f"({list(FAMILY_ORDER)}: gana la familia mas simple) y, en ultimo termino, con el "
    "`run_sha256` menor. Un candidato sin metrica deja la seleccion en `not_evaluable`: no se "
    "elige por eliminacion"
)

#: La regla de calibracion, importada de #25 y publicada sin reescribirla (A3, A6).
CALIBRATION_RULE: Final[str] = (
    "Platt si `n_calibration < 500` e isotonica si no, evaluado **por fold** con el recuento "
    "medido; `none` es un estado publicado con su `reason` y ese fold pasa la probabilidad "
    "cruda. Las **dos** familias pasan por el mismo camino de `cfdtrader.models.calibration` "
    "(#25), con el mismo reparto por fold (A6)"
)

#: Bloques del PBO por CSCV sobre 500 sesiones.
#:
#: **No** es ``DEFAULT_BLOCKS`` (16) de #16 a proposito: ``blocks`` tiene que dividir al numero
#: de observaciones y 500 no es multiplo de 16. Con 10 bloques de 50 sesiones,
#: ``C(10, 5) = 252 <= MAX_COMBINATIONS``, asi que el CSCV se recorre **exhaustivo** y sin
#: semilla de muestreo.
PBO_BLOCKS: Final[int] = 10

#: Las cuatro cifras **congeladas** de la linea base de #24/#25, sobre las mismas 500 sesiones.
#:
#: No se copian al informe: se **verifican** contra la reconstruccion (A7). Una discrepancia
#: mayor que :data:`FROZEN_TOLERANCE` es error tipado, nunca una cifra publicada en silencio.
FROZEN_BASELINE: Final[dict[str, dict[str, float]]] = {
    "raw": {
        "n_traded": 356,
        "brier_score": 0.2511418117147099,
        "log_loss": 0.6953995223105118,
        "pnl_declared_sum": -1.4594798813611285,
    },
    "calibrated": {
        "n_traded": 404,
        "brier_score": 0.2548043292843032,
        "log_loss": 0.7672355544413263,
        "pnl_declared_sum": -1.663373389040144,
    },
}

#: Tolerancia de A7. Es ``1e-9`` y no ``0`` porque las cifras congeladas se publicaron con
#: ``repr`` de ``float`` y la reconstruccion vuelve a sumar los mismos terminos en el mismo
#: orden; el valor medido coincide en las cuatro cifras de las dos series.
FROZEN_TOLERANCE: Final[float] = 1e-9

#: Formato estable del ``report_sha256`` (A13).
REPORT_HASH_FORMAT: Final[str] = (
    "sha256(UTF-8) de cfdtrader.backtest.engine.canonical_text(payload), **sin** la clave "
    "report_sha256 (un informe no se hashea a si mismo). El payload es JSON puro y no lleva "
    "ninguna ruta absoluta: el directorio del registro se publica relativo (`runs/<sha>`) y de "
    "`--settings` solo viaja la raiz declarada en forma relativa, asi que el hash no depende de "
    "`--reports-dir` ni de `--runs-root` (A13)"
)

#: Formato estable del ``model_sha256`` (A5).
MODEL_HASH_FORMAT: Final[str] = (
    "sha256(UTF-8) de cfdtrader.backtest.engine.canonical_text(payload del modelo), donde el "
    "payload son la version de LightGBM, las 10 features, los hiperparametros, la semilla y, "
    "por fold, el **texto** del booster (`model_to_string()`), las posiciones de test, las "
    "probabilidades y los margenes publicados y su calibrador. Es **contenido**: el JSON basta "
    "para reproducir las predicciones sin `pickle`"
)

#: Motivo publicado cuando las metricas netas no se pueden calcular (A12).
NET_METRICS_REASON: Final[str] = (
    "`pnl_net_pct` es `null` en todas las operaciones: el *slippage* de #64 es un **supuesto** "
    "declarado (`state: assumed`) y sin `R` (#60) no se puede cobrar, asi que "
    "`cfdtrader.backtest.metrics.calculate_metrics` rechaza la corrida. Fabricar un *slippage* "
    "`measured` para poder publicar metricas netas esta **prohibido**"
)

#: El bloque de #80: el desajuste de unidades del motor, medido y **no** arreglado aqui.
UNIT_BUG_ISSUE: Final[str] = "#80"

#: Que **no** hace este modulo, legible por maquina. Cada frontera con su issue.
REPORT_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_barre_hiperparametros",
        "issue": "#82",
        "statement": (
            "no busca hiperparametros ni subconjunto de features: ajusta **una** familia "
            "LightGBM con la constante `LIGHTGBM_HYPERPARAMETERS` y no acepta banderas de "
            "ajuste. Otra variante exige una entrada nueva en `runs/` con otro `variant_id`, y "
            "la busqueda es #82"
        ),
    },
    {
        "id": "no_arregla_el_motor",
        "issue": "#80",
        "statement": (
            "**si** encuentra el desajuste de unidades de `backtest/engine.py` (`pnl_declared_pct` "
            "resta un porcentaje a una fraccion) y lo publica **medido** en `unit_bug_80`, con su "
            "magnitud y su efecto; **no** lo arregla: `backtest/engine.py` y `backtest/costs.py` "
            "no se tocan"
        ),
    },
    {
        "id": "no_publica_metricas_netas",
        "issue": "#62",
        "statement": (
            "no publica metricas netas: el supuesto de #64 deja `pnl_net_pct` en `null` y "
            "`calculate_metrics` rechaza la corrida a proposito; medir el *slippage* es #62"
        ),
    },
    {
        "id": "no_decide_el_umbral_economico",
        "issue": "#27",
        "statement": (
            "el umbral 0,5 es el del informe, no el del sistema: el umbral economico, el "
            "*sizing* y la puerta son #27, #60 y #29"
        ),
    },
    {
        "id": "no_es_el_backtest_de_fase_2",
        "issue": "#28",
        "statement": (
            "no corre el backtest completo ni los listones B y C: compara las cuatro variantes "
            "del registro en el espacio de probabilidad y de coste declarado"
        ),
    },
    {
        "id": "no_persiste_features",
        "issue": "#73",
        "statement": (
            "no escribe `derived.features_daily`: rearma la matriz leyendo el almacen en cada "
            "corrida; la persistencia es #73"
        ),
    },
    {
        "id": "no_toca_el_holdout",
        "issue": "#68",
        "statement": (
            "el calibrador se ajusta **dentro** del *train* de cada fold: el *holdout* final de "
            "§11.4 no se toca y su definicion es #68"
        ),
    },
)

#: Seguimientos declarados por el informe.
FOLLOW_UPS: Final[tuple[dict[str, str], ...]] = (
    {
        "issue": "#82",
        "topic": "busqueda de hiperparametros y de subconjunto de features",
        "why": (
            "la familia LightGBM se ajusta con una constante a priori; moverla exige una entrada "
            "nueva en `runs/` y un presupuesto declarado, que es #82"
        ),
    },
    {
        "issue": "#80",
        "topic": "desajuste de unidades de `pnl_declared_pct` en el motor",
        "why": (
            "el retorno declarado resta 100x el coste declarado; el orden entre variantes y las "
            "metricas de probabilidad no cambian, el Sharpe si"
        ),
    },
    {
        "issue": "#62",
        "topic": "medir el *slippage* real",
        "why": "mientras no exista, `pnl_net_pct` es `null` y las metricas netas no existen",
    },
    {
        "issue": "#60",
        "topic": "decidir `R` y el umbral economico",
        "why": "el supuesto de #64 se declara como porcentaje de `R`, que sigue sin decidirse",
    },
    {
        "issue": "#68",
        "topic": "holdout final intocable",
        "why": "la comparacion usa el plan de #69; el *holdout* de §11.4 sigue sin tocarse",
    },
    {
        "issue": "#78",
        "topic": "direccion corta",
        "why": "las dos familias se evaluan en la direccion **larga unica**; `ret_short` es #78",
    },
    {
        "issue": "#29",
        "topic": "puerta de Fase 2",
        "why": "este informe compara modelos; la puerta y el *sizing* son #27 y #29",
    },
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados
# ─────────────────────────────────────────────────────────────────────────────
class ModelComparisonError(Exception):
    """Raiz de los errores del informe de comparacion."""


class MissingAsOfError(ModelComparisonError):
    """Escribir el informe exige un instante declarado: el modulo no lee el reloj (A2)."""


class InvalidAsOfError(ModelComparisonError):
    """El instante declarado no es un ISO-8601 valido."""


class MissingDecisionError(ModelComparisonError):
    """Una sesion de *test* no trae decision: sin decision no hay probabilidad que comparar."""


class FrozenBaselineMismatchError(ModelComparisonError):
    """La linea base reconstruida **no** reproduce lo congelado por #24/#25 (A7)."""


class UnknownVariantError(ModelComparisonError):
    """Una entrada del registro no pertenece a ninguna familia reconstruible (A10)."""


class MissingModelDocumentError(ModelComparisonError):
    """Falta el `model.json` de una entrada del registro: no se puede reconstruir (A10)."""


class InconsistentObservationsError(ModelComparisonError):
    """El `n_observations` del registro no cuadra con la serie reconstruida (A10)."""


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades locales (el mismo contrato que #69/#24)
# ─────────────────────────────────────────────────────────────────────────────
def _as_utc(value: datetime) -> datetime:
    """Normaliza el instante declarado a UTC; sin zona se interpreta como UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _digest(payload: Mapping[str, object]) -> str:
    """sha256 del texto canonico de #13: la **unica** funcion de hash que se usa."""
    return hashlib.sha256(canonical_text(payload).encode("utf-8")).hexdigest()


def _json_text(document: Mapping[str, object]) -> str:
    """El mismo JSON determinista que #16/#24: sin `nan` ni `inf`, dos espacios, orden fijo."""
    return json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _series_digest(series: Sequence[float]) -> str:
    """sha256 de la serie por sesion, para poder casar columnas sin repetirlas."""
    return hashlib.sha256(canonical_text({"series": list(series)}).encode("utf-8")).hexdigest()


def model_sha256(model: LightGBMModel) -> str:
    """``model_sha256``: sha256 del payload del modelo LightGBM (texto del booster incluido)."""
    return _digest(model.to_payload())


def _date_text(value: date | None) -> str | None:
    """Una sesion se publica en ISO-8601; ``None`` sigue siendo ``None``."""
    return None if value is None else value.isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Candidatos: una fila por entrada del registro (A8, A10)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Variant:
    """Una variante del registro **reconstruida** y medida, con su serie por sesion (A8).

    ``series`` tiene una entrada por sesion de *test*, en orden, con ``0.0`` donde la variante
    no opera (plano no es perdida). ``deciding_probabilities`` son las probabilidades con las
    que **decide** esa variante (la cruda o la calibrada, segun su configuracion) y
    ``outcomes`` las etiquetas de esas mismas sesiones: con las dos se reproduce el Brier y la
    log-loss publicados sin recalcular nada.
    """

    run_sha256: str
    variant_id: str
    calibrated: bool
    source: str
    n_test: int
    n_traded: int
    brier_score: float
    log_loss_value: float
    pnl_declared_sum: float
    sharpe_per_session: float
    series: tuple[float, ...]
    deciding_probabilities: tuple[float, ...]
    outcomes: tuple[int, ...]
    fold_calibration: tuple[dict[str, object], ...]
    run: BacktestRun

    @property
    def zeros(self) -> int:
        """Sesiones de *test* en las que la variante no opera: ``n_test - n_traded``."""
        return self.n_test - self.n_traded

    def to_payload(self) -> dict[str, object]:
        """La fila del candidato como JSON puro, con sus 500 valores por sesion (A8, A12)."""
        return {
            "run_sha256": self.run_sha256,
            "variant_id": self.variant_id,
            "calibrated": self.calibrated,
            "source": self.source,
            "n_test": self.n_test,
            "n_traded": self.n_traded,
            "zeros": self.zeros,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss_value,
            "pnl_declared_pct_sum": self.pnl_declared_sum,
            "sharpe_per_session": self.sharpe_per_session,
            "sharpe_per_session_units": "por sesion, **solo las operadas** (`n_traded`)",
            "series_sha256": _series_digest(self.series),
            "series": list(self.series),
            "deciding_probabilities": list(self.deciding_probabilities),
            "outcomes": list(self.outcomes),
            "fold_calibration": [dict(item) for item in self.fold_calibration],
            "state": "evaluated",
        }


@dataclass(frozen=True, slots=True)
class NotEvaluable:
    """Una entrada del registro que **no** se pudo reconstruir: se declara, no se rellena (A10).

    La variante **no** se suelta: sigue contando en ``registry.n_trials`` y aparece con su
    ``run_sha256`` y su motivo, pero no aporta columna —y por tanto **no** aporta una columna de
    ceros— a la matriz.
    """

    run_sha256: str
    variant_id: str
    reason: str
    error: str

    def to_payload(self) -> dict[str, object]:
        """El bloque declarado, sin ningun numero inventado."""
        return {
            "run_sha256": self.run_sha256,
            "variant_id": self.variant_id,
            "state": "not_evaluable",
            "reason": self.reason,
            "error": self.error,
            "column": None,
            "note": (
                "no se crea una columna de ceros ni se suelta la variante: sigue contando en "
                "`registry.n_trials` y la matriz se declara incompleta (A10)"
            ),
        }


Candidate = Variant | NotEvaluable


def _float_tuple(value: object) -> tuple[float, ...]:
    """Una lista JSON de numeros como tupla de ``float`` (vacia si no viene)."""
    return tuple(float(cast("Any", item)) for item in cast("list[object]", value or []))


def _calibration_from_payload(block: Mapping[str, object]) -> Calibration:
    """Reconstruye el calibrador publicado por #25 desde su bloque JSON (A7).

    Se leen los parametros **publicados** (`coef`/`intercept`/`mean`/`scale` o
    `thresholds`/`values`): el calibrador no se reajusta, se vuelve a aplicar.
    """
    parameters = cast("Mapping[str, object]", block.get("parameters") or {})
    return Calibration(
        method=str(block["method"]),
        reason=None if block.get("reason") is None else str(block["reason"]),
        n_fit=int(cast("int", block["n_fit"])),
        n_calibration=int(cast("int", block["n_calibration"])),
        n_positives=int(cast("int", block["n_positives"])),
        calibration_positions=tuple(
            int(value) for value in cast("list[int]", block["calibration_positions"])
        ),
        purge_sessions=int(cast("int", block["purge_sessions"])),
        exclusions_are_no_op=bool(block["exclusions_are_no_op"]),
        coef=None if parameters.get("coef") is None else float(cast("float", parameters["coef"])),
        intercept=(
            None
            if parameters.get("intercept") is None
            else float(cast("float", parameters["intercept"]))
        ),
        mean=None if parameters.get("mean") is None else float(cast("float", parameters["mean"])),
        scale=(
            None if parameters.get("scale") is None else float(cast("float", parameters["scale"]))
        ),
        thresholds=_float_tuple(parameters.get("thresholds")),
        values=_float_tuple(parameters.get("values")),
    )


def _model_document(runs_root: Path, run_sha256: str) -> dict[str, object]:
    """El `model.json` de esa entrada del registro, o error tipado (A10)."""
    path = runs_root / run_sha256 / MODEL_FILE
    if not path.is_file():
        raise MissingModelDocumentError(
            f"`runs/{run_sha256}/model.json` no existe: la entrada no se puede reconstruir sin "
            "reajustar, y reajustarla seria otro experimento (A10)"
        )
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _baseline_deciding_probabilities(
    document: Mapping[str, object],
    *,
    matrix: np.ndarray[Any, np.dtype[np.float64]],
    calibrated: bool,
) -> tuple[float | None, ...]:
    """Reconstruye la probabilidad que **decide** en cada posicion, sin reajustar (A7).

    La aritmetica es la de #24/#25 sobre los parametros **publicados**:
    ``(x - mean) / scale @ coef + intercept`` y el enlace; y si el fold publica calibrador, se
    le vuelven a pasar esas mismas puntuaciones. Un fold sin calibrador pasa la cruda.
    """
    model = cast("Mapping[str, object]", document["model"])
    folds = cast("list[object]", model["folds"])
    out: list[float | None] = [None] * int(matrix.shape[0])
    for item in folds:
        fold = cast("Mapping[str, object]", item)
        positions = [int(value) for value in cast("list[int]", fold["test_positions"])]
        mean = np.asarray(cast("list[float]", fold["mean"]), dtype=np.float64)
        scale = np.asarray(cast("list[float]", fold["scale"]), dtype=np.float64)
        coefficients = np.asarray(cast("list[float]", fold["coefficients"]), dtype=np.float64)
        intercept = float(cast("float", fold["intercept"]))
        scores = ((matrix[positions, :] - mean) / scale) @ coefficients + intercept
        values: Sequence[float | None] = [None] * len(positions)
        if calibrated:
            calibration = _calibration_from_payload(
                cast("Mapping[str, object]", fold["calibration"])
            )
            values = calibration.calibrate([float(value) for value in scores])
        for position, score, value in zip(positions, scores, values, strict=True):
            out[position] = sigmoid(float(score)) if value is None else float(value)
    return tuple(out)


def _run_variant(
    *,
    decided: Sequence[float | None],
    universe: Universe,
    plan: SplitPlan,
    frame: FeatureFrame,
    cost_model: CostModel,
    slippage: SlippageParameter,
) -> tuple[BacktestRun, tuple[float, ...], tuple[float, ...], tuple[int, ...]]:
    """Corre el motor con esas probabilidades y devuelve la corrida y sus series (A8).

    Las probabilidades viajan en la carga opaca de la vista y el decider **solo** lee la vista
    (A11); el umbral es el declarado y no se mueve. La serie por sesion sale del motor: una
    entrada por sesion de *test*, ``0.0`` donde no opera.
    """
    scored = _scored_inputs(universe.inputs, decided)
    run = run_walk_forward(
        scored,
        split_plan=plan,
        cost_model=cost_model,
        slippage=slippage,
        decide_by_fold=tuple(_decider(fold.index) for fold in plan.folds),
        financing_cut=None,
    )
    labels = {
        cast("date", row["session"]): int(cast("int", row["y"]))
        for row in frame.design.frame.select("session", "y").iter_rows(named=True)
    }
    series: list[float] = []
    probabilities_ordered: list[float] = []
    outcomes: list[int] = []
    for fold in run.folds:
        for session in fold.sessions:
            decision = session.decision
            probability = None if decision is None else decision.probability
            if probability is None:
                raise MissingDecisionError(
                    f"la sesion {session.session.isoformat()} del fold {fold.index} no trae "
                    "probabilidad: el adaptador la dejo fuera de la vista, asi que no hay "
                    "metrica de probabilidad ni columna que publicar (A8)"
                )
            probabilities_ordered.append(probability)
            outcomes.append(labels[session.session])
            traded = session.status == STATUS_TRADED and session.pnl_declared_pct is not None
            series.append(float(cast("float", session.pnl_declared_pct)) if traded else 0.0)
    return run, tuple(series), tuple(probabilities_ordered), tuple(outcomes)


def _variant_from_run(
    *,
    run_sha256: str,
    variant_id: str,
    calibrated: bool,
    source: str,
    measured: tuple[BacktestRun, tuple[float, ...], tuple[float, ...], tuple[int, ...]],
    fold_calibration: Sequence[Mapping[str, object]],
) -> Variant:
    """Mide la variante con la aritmetica de #15 sobre las mismas 500 sesiones (A12)."""
    run, series, decided, outcomes = measured
    declared = [
        float(session.pnl_declared_pct)
        for fold in run.folds
        for session in fold.sessions
        if session.status == STATUS_TRADED and session.pnl_declared_pct is not None
    ]
    return Variant(
        run_sha256=run_sha256,
        variant_id=variant_id,
        calibrated=calibrated,
        source=source,
        n_test=len(series),
        n_traded=run.traded,
        brier_score=brier_score(list(decided), list(outcomes)),
        log_loss_value=log_loss(list(decided), list(outcomes)),
        pnl_declared_sum=math.fsum(declared),
        sharpe_per_session=sharpe_ratio(declared, annualization=1),
        series=series,
        deciding_probabilities=decided,
        outcomes=outcomes,
        fold_calibration=tuple(dict(item) for item in fold_calibration),
        run=run,
    )


def _require_frozen_match(variant: Variant) -> None:
    """La linea base reconstruida tiene que reproducir lo congelado (A7)."""
    reference = FROZEN_BASELINE["calibrated" if variant.calibrated else "raw"]
    observed = {
        "n_traded": float(variant.n_traded),
        "brier_score": variant.brier_score,
        "log_loss": variant.log_loss_value,
        "pnl_declared_sum": variant.pnl_declared_sum,
    }
    off = [
        f"{name}: reconstruido {value!r}, congelado {reference[name]!r}"
        for name, value in observed.items()
        if abs(value - reference[name]) > FROZEN_TOLERANCE
    ]
    if off:
        raise FrozenBaselineMismatchError(
            f"la variante `{variant.variant_id}` reconstruida desde "
            f"`runs/{variant.run_sha256}/model.json` no reproduce las cifras congeladas de "
            f"#24/#25 (tolerancia {FROZEN_TOLERANCE!r}): " + "; ".join(off)
        )


def reconstruct_baseline(
    entry: RegistryEntry,
    *,
    runs_root: Path,
    universe: Universe,
    frame: FeatureFrame,
    plan: SplitPlan,
    cost_model: CostModel,
    slippage: SlippageParameter,
) -> Variant:
    """Reconstruye una variante lineal desde su `model.json`, **sin reajustar** (A7).

    Lanza error tipado si el `variant_id` no es el de la familia lineal, si falta el
    `model.json` o si el `n_observations` del registro no cuadra con las operaciones de la
    serie reconstruida (A10).
    """
    if entry.variant_id != BASELINE_VARIANT_ID:
        raise UnknownVariantError(
            f"el `variant_id` {entry.variant_id!r} no es ninguna de las familias reconstruibles "
            f"({[BASELINE_VARIANT_ID, VARIANT_ID]}): no se rellena con una columna de ceros ni "
            "se reajusta (A10)"
        )
    document = _model_document(runs_root, entry.run_sha256)
    model = cast("Mapping[str, object]", document["model"])
    folds = cast("list[object]", model["folds"])
    first = cast("Mapping[str, object]", folds[0])
    calibrated = "calibration" in first
    if list(cast("list[str]", model["features"])) != list(BASELINE_FEATURES):
        raise UnknownVariantError(
            f"`runs/{entry.run_sha256}/model.json` declara otras features que las 10 de #24: la "
            "serie no es comparable en la misma matriz de diseno (A10)"
        )
    matrix = np.asarray(
        frame.design.frame.select(list(BASELINE_FEATURES)).cast(pl.Float64).to_numpy(),
        dtype=np.float64,
    )
    decided = _baseline_deciding_probabilities(document, matrix=matrix, calibrated=calibrated)
    measured = _run_variant(
        decided=decided,
        universe=universe,
        plan=plan,
        frame=frame,
        cost_model=cost_model,
        slippage=slippage,
    )
    fold_calibration = (
        [
            cast("Mapping[str, object]", cast("Mapping[str, object]", item)["calibration"])
            for item in folds
        ]
        if calibrated
        else []
    )
    variant = _variant_from_run(
        run_sha256=entry.run_sha256,
        variant_id=entry.variant_id,
        calibrated=calibrated,
        source="model_json_reconstruction",
        measured=measured,
        fold_calibration=fold_calibration,
    )
    if variant.n_traded != entry.n_observations:
        raise InconsistentObservationsError(
            f"`runs/{entry.run_sha256}/result.json` declara `n_observations = "
            f"{entry.n_observations}` y la serie reconstruida opera {variant.n_traded} sesiones: "
            "la entrada no se puede emparejar por `variant_id` + `n_observations` y su columna "
            "no se publica (A10)"
        )
    _require_frozen_match(variant)
    return variant


def _lightgbm_fold_calibration(model: LightGBMModel) -> list[dict[str, object]]:
    """El calibrador de cada fold LightGBM, con el mismo vocabulario que #25 (A6)."""
    return [
        {
            "index": fold.index,
            "method": fold.calibration.method,
            "reason": fold.calibration.reason,
            "n_calibration": fold.calibration.n_calibration,
            "n_positives": fold.calibration.n_positives,
            "purge_sessions": fold.calibration.purge_sessions,
            "exclusions_are_no_op": fold.calibration.exclusions_are_no_op,
        }
        for fold in model.folds
    ]


def _candidates(
    *,
    registry: Registry,
    runs_root: Path,
    known: Mapping[str, Variant],
    universe: Universe,
    frame: FeatureFrame,
    plan: SplitPlan,
    cost_model: CostModel,
    slippage: SlippageParameter,
) -> tuple[Candidate, ...]:
    """Una fila por entrada del registro, en su orden, con su estado **declarado** (A10).

    Ninguna variante se rellena y ninguna se suelta: si no se puede medir, se publica
    `not_evaluable` con el motivo y su `run_sha256`, y no aporta columna.
    """
    out: list[Candidate] = []
    for entry in registry.entries:
        if entry.run_sha256 in known:
            variant = known[entry.run_sha256]
            if variant.n_traded != entry.n_observations:
                raise InconsistentObservationsError(
                    f"la variante LightGBM `{entry.run_sha256}` opera {variant.n_traded} sesiones "
                    f"y su `result.json` declara {entry.n_observations}"
                )
            out.append(variant)
            continue
        try:
            out.append(
                reconstruct_baseline(
                    entry,
                    runs_root=runs_root,
                    universe=universe,
                    frame=frame,
                    plan=plan,
                    cost_model=cost_model,
                    slippage=slippage,
                )
            )
        except ModelComparisonError as error:
            out.append(
                NotEvaluable(
                    run_sha256=entry.run_sha256,
                    variant_id=entry.variant_id,
                    reason=str(error),
                    error=type(error).__name__,
                )
            )
    return tuple(out)


def _evaluated(candidates: Sequence[Candidate]) -> tuple[Variant, ...]:
    """Las variantes medidas, en el orden del registro."""
    return tuple(item for item in candidates if isinstance(item, Variant))


# ─────────────────────────────────────────────────────────────────────────────
# La regla de seleccion, pre-declarada (A11)
# ─────────────────────────────────────────────────────────────────────────────
def _family_rank(variant_id: str) -> int:
    """Posicion de la familia en :data:`FAMILY_ORDER`; las desconocidas van al final."""
    order = list(FAMILY_ORDER)
    return order.index(variant_id) if variant_id in order else len(FAMILY_ORDER)


def _compare(one: Variant, other: Variant) -> int:
    """Orden total declarado: Brier, log-loss, familia y, en ultimo termino, `run_sha256`."""
    if abs(one.brier_score - other.brier_score) > TIE_TOLERANCE:
        return -1 if one.brier_score < other.brier_score else 1
    if abs(one.log_loss_value - other.log_loss_value) > TIE_TOLERANCE:
        return -1 if one.log_loss_value < other.log_loss_value else 1
    ranks = (_family_rank(one.variant_id), _family_rank(other.variant_id))
    if ranks[0] != ranks[1]:
        return -1 if ranks[0] < ranks[1] else 1
    if one.run_sha256 == other.run_sha256:
        return 0
    return -1 if one.run_sha256 < other.run_sha256 else 1


def selection_block(candidates: Sequence[Candidate]) -> dict[str, object]:
    """La eleccion del modelo definitivo segun la regla declarada (A11).

    Un candidato sin metrica deja la seleccion en ``not_evaluable`` **sin** `selected`: no se
    elige por eliminacion ni se cablea el ganador.
    """
    evaluated = _evaluated(candidates)
    blockers = [
        {
            "run_sha256": item.run_sha256,
            "variant_id": item.variant_id,
            "reason": item.reason,
        }
        for item in candidates
        if isinstance(item, NotEvaluable)
    ]
    rows = [
        {
            "run_sha256": item.run_sha256,
            "variant_id": item.variant_id,
            "calibrated": item.calibrated,
            "brier_score": item.brier_score,
            "log_loss": item.log_loss_value,
            "n_traded": item.n_traded,
            "family_rank": _family_rank(item.variant_id),
        }
        for item in evaluated
    ]
    base: dict[str, object] = {
        "rule": SELECTION_RULE,
        "primary_metric": PRIMARY_METRIC,
        "tie_breakers": list(TIE_BREAKERS),
        "family_order": list(FAMILY_ORDER),
        "tie_tolerance": TIE_TOLERANCE,
        "is_validation": False,
        "candidates": rows,
        "blockers": blockers,
        "n_candidates": len(rows),
        "n_blockers": len(blockers),
    }
    if blockers or not evaluated:
        return {
            **base,
            "state": "not_evaluable",
            "selected": None,
            "reason": (
                "hay candidatos sin metrica: la seleccion no se resuelve por eliminacion ni se "
                "rellena con el mejor de los que quedan (A11)"
            ),
        }
    best = min(evaluated, key=functools.cmp_to_key(_compare))
    return {
        **base,
        "state": "selected",
        "selected": {
            "run_sha256": best.run_sha256,
            "variant_id": best.variant_id,
            "calibrated": best.calibrated,
            "brier_score": best.brier_score,
            "log_loss": best.log_loss_value,
            "n_traded": best.n_traded,
            "runs_directory": f"{DEFAULT_RUNS_ROOT}/{best.run_sha256}",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Matriz, DSR y PBO de las variantes **reales** (A8, A9)
# ─────────────────────────────────────────────────────────────────────────────
def require_matrix_matches_registry(*, registry: Registry, n_columns: int) -> None:
    """La matriz tiene que traer una columna por variante del registro (A9).

    Es la puerta que hace imposible deflactar por un numero de intentos «estimado»: si el
    registro tiene 4 entradas y la matriz 3 columnas, ``TrialsMismatchError``.
    """
    require_trials_match_registry(
        n_trials=n_columns, sr_variance=registry.sr_variance, registry=registry
    )


def _series_matrix(evaluated: Sequence[Variant]) -> list[list[float]]:
    """La matriz `T x N` del CSCV: **una fila por sesion** y una columna por variante (A8).

    El PBO recibe una matriz de retornos con las **observaciones en las filas** y las variantes
    en las columnas: pasarle las columnas como filas seria medir otra cosa.
    """
    if not evaluated:
        return []
    n_observations = len(evaluated[0].series)
    return [[item.series[index] for item in evaluated] for index in range(n_observations)]


def _matrix_block(evaluated: Sequence[Variant], *, registry: Registry) -> dict[str, object]:
    """La matriz rectangular de las variantes medidas: 500 filas por columna (A8)."""
    sizes = sorted({item.n_test for item in evaluated})
    return {
        "n_observations": sizes[0] if len(sizes) == 1 else None,
        "n_variants": len(evaluated),
        "registry_n_trials": registry.n_trials,
        "matrix_matches_registry": len(evaluated) == registry.n_trials,
        "blocks": PBO_BLOCKS,
        "units": "per_session",
        "columns": [
            {
                "index": index,
                "run_sha256": item.run_sha256,
                "variant_id": item.variant_id,
                "calibrated": item.calibrated,
                "zeros": item.zeros,
                "series_sha256": _series_digest(item.series),
            }
            for index, item in enumerate(evaluated)
        ],
        "zero_rule": (
            "`0.0` donde la variante **no opera**: plano no es perdida, y un `0.0` es el retorno "
            "de una sesion sin posicion. Los ceros de cada columna son `n_test - n_traded`"
        ),
        "blocks_rule": (
            f"el CSCV necesita que `blocks` divida al numero de observaciones: 500 no es multiplo "
            f"de `DEFAULT_BLOCKS` (16) de #16, asi que se usan {PBO_BLOCKS} bloques de 50 "
            f"sesiones (C({PBO_BLOCKS}, {PBO_BLOCKS // 2}) = 252 <= MAX_COMBINATIONS, "
            "exhaustivo y sin semilla de muestreo)"
        ),
        "note": (
            "cada serie sale del **motor** (`run_walk_forward`, como en #24): hereda su "
            "convencion, incluido el desajuste de unidades de #80, que se publica en "
            "`unit_bug_80` y no se arregla aqui"
        ),
    }


def _verdict_block(
    *,
    evaluated: Sequence[Variant],
    registry: Registry,
    selection: Mapping[str, object],
    candidates: Sequence[Candidate],
) -> dict[str, object]:
    """DSR de la variante seleccionada y PBO de la matriz, con el agregado de #9 (A9, A10).

    Si la matriz no trae una columna por variante del registro, los dos calculos se declaran
    ``not_evaluable``: no se deflacta con un `n_trials` que no sea el de las columnas.
    """
    complete = len(evaluated) == registry.n_trials
    if not complete:
        reason = (
            f"la matriz trae {len(evaluated)} columnas y el registro {registry.n_trials} "
            "experimentos: hay variantes que no se pudieron medir, asi que no hay `n_trials` con "
            "el que deflactar ni matriz completa que someter al CSCV (A10)"
        )
        return {
            "state": "not_evaluable",
            "reason": reason,
            "blockers": [item.run_sha256 for item in candidates if isinstance(item, NotEvaluable)],
            "deflated_sharpe_ratio": {
                "state": "not_evaluable",
                "calculation": "deflated_sharpe_ratio",
                "verdict": "not_evaluable",
                "reason": reason,
            },
            "probability_of_backtest_overfitting": {
                "state": "not_evaluable",
                "calculation": "probability_of_backtest_overfitting",
                "verdict": "not_evaluable",
                "reason": reason,
            },
            "gate": _gate_block(reason=reason),
        }
    require_matrix_matches_registry(registry=registry, n_columns=len(evaluated))

    selected_digest = cast("Mapping[str, object] | None", selection.get("selected"))
    dsr: dict[str, object]
    if selected_digest is None:
        dsr = {
            "state": "not_evaluable",
            "calculation": "deflated_sharpe_ratio",
            "verdict": "not_evaluable",
            "reason": "no hay variante seleccionada sobre la que calcular el DSR (A11)",
        }
    else:
        chosen = next(
            item for item in evaluated if item.run_sha256 == str(selected_digest["run_sha256"])
        )
        dsr = deflate_block(returns=chosen.series, registry=registry)
        dsr = {
            **dsr,
            "selected_run_sha256": chosen.run_sha256,
            "selected_variant_id": chosen.variant_id,
            "registry_sharpe_per_session": chosen.sharpe_per_session,
            "series": (
                "el Sharpe del DSR sale de las **500** sesiones de test con `0.0` donde no opera "
                "y el `sharpe_per_session` del registro de las **solo operadas** "
                f"({chosen.n_traded}): son `per_session` los dos, pero **no miden lo mismo**"
            ),
        }
    pbo = pbo_block(returns_matrix=_series_matrix(evaluated), blocks=PBO_BLOCKS)
    return {
        "state": "evaluated",
        "deflated_sharpe_ratio": dsr,
        "probability_of_backtest_overfitting": pbo,
        "gate": _gate_block(
            reason=None,
            dsr_verdict=str(dsr["verdict"]),
            pbo_verdict=str(pbo["verdict"]),
        ),
    }


def _gate_block(
    *,
    reason: str | None,
    dsr_verdict: str = "not_evaluable",
    pbo_verdict: str = "not_evaluable",
) -> dict[str, object]:
    """El agregado de los dos veredictos, con la regla importada de #9 y comprobada (A10)."""
    gate = aggregate_verdict(dsr_verdict=dsr_verdict, pbo_verdict=pbo_verdict)
    require_consistent_aggregate(
        gate=gate, dsr_half=DSR_HALVES[dsr_verdict], pbo_half=PBO_HALVES[pbo_verdict]
    )
    return {
        "aggregate": gate,
        "halves": {
            "deflated_sharpe_ratio": DSR_HALVES[dsr_verdict],
            "probability_of_backtest_overfitting": PBO_HALVES[pbo_verdict],
        },
        "verdicts": {
            "deflated_sharpe_ratio": dsr_verdict,
            "probability_of_backtest_overfitting": pbo_verdict,
        },
        "aggregation_source": "cfdtrader.analysis.phase0_report.aggregate_gate (#9)",
        "reason": reason,
        "note": (
            "`not_evaluable` nunca se convierte en `pass`: el agregado solo aprueba con las dos "
            "mitades aprobadas (A10). Es una correccion por intentos, **no** una validacion de "
            "la estrategia"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# El bloque de #80 y el bloque de coste declarado (A12)
# ─────────────────────────────────────────────────────────────────────────────
def _unit_bug_block(variant: Variant | None) -> dict[str, object]:
    """El desajuste de unidades del motor, **medido** en esta corrida (A12).

    `engine.py` resta `c_declared_pct` (porcentaje del nocional) a `gross_pct` (fraccion): el
    termino que se resta es 100x el correcto. La constante por operacion se **mide** aqui
    (``gross - declared``) y se compara con el termino correcto (``c_declared_pct / 100``).
    """
    traded = (
        [
            session
            for fold in variant.run.folds
            for session in fold.sessions
            if session.status == STATUS_TRADED and session.gross_pct is not None
        ]
        if variant is not None
        else []
    )
    differences: list[float] = []
    correct: list[float] = []
    for session in traded:
        declared = session.pnl_declared_pct
        if declared is None:
            continue
        differences.append(float(cast("float", session.gross_pct)) - float(declared))
        if session.cost is not None:
            correct.append(float(session.cost.c_declared_pct) / 100.0)
    observed = math.fsum(differences) / len(differences) if differences else None
    correct_term = math.fsum(correct) / len(correct) if correct else None
    return {
        "issue": UNIT_BUG_ISSUE,
        "observed_on": None if variant is None else variant.run_sha256,
        "n_operations": len(traded),
        "observed_per_operation": observed,
        "correct_term_per_operation": correct_term,
        "difference_per_operation": (
            None if observed is None or correct_term is None else observed - correct_term
        ),
        "constants": {
            "c_declared_pct": "0.0042 (% del nocional, lo que el motor resta)",
            "c_fraction_of_notional": "0.000042 (fraccion, el termino correcto)",
            "identity": "0.0042 - 0.000042 = 0.004158",
        },
        "statement": (
            "`backtest/engine.py` publica `pnl_declared_pct = gross_pct - c_declared_pct`, con "
            "`gross_pct = close / open - 1` (**fraccion**) y `c_declared_pct` en **porcentaje** "
            "del nocional: se resta 100x el coste declarado, una constante por operacion"
        ),
        "affects": (
            "la media y la suma de `pnl_declared_pct` de **todas** las filas que operan y, con "
            "ellas, el Sharpe deflactado: el desplazamiento es constante y del mismo signo en "
            "todas las variantes"
        ),
        "does_not_affect": (
            "el **orden** entre variantes (todas restan la misma constante), las metricas de "
            "**probabilidad** (Brier, log-loss, curva) ni el PBO, que reordena columnas de la "
            "misma matriz desplazada"
        ),
        "fixed_here": False,
        "follow_up": [UNIT_BUG_ISSUE],
        "note": (
            "este informe **no** arregla el motor: `backtest/engine.py` y `backtest/costs.py` no "
            "se tocan (A12). El bloque esta medido en esta corrida, no copiado"
        ),
    }


def _declared_cost_block(evaluated: Sequence[Variant], *, model: CostModel) -> dict[str, object]:
    """El coste declarado de la corrida, con la base y el estado del *slippage* (A12)."""
    return {
        "basis": "declared_cost",
        "is_validation": False,
        "model": {
            "name": model.name,
            "spread_entry_pct": format(model.spread_entry_pct, "f"),
            "spread_exit_pct": format(model.spread_exit_pct, "f"),
            "carry_long_pct_per_night": format(model.carry_long_pct_per_night, "f"),
            "carry_short_pct_per_night": format(model.carry_short_pct_per_night, "f"),
            "fx_pct": format(model.fx_pct, "f"),
            "commission_pct": format(model.commission_pct, "f"),
            "provenance": "`declared_cost_model()` de #8, importado: el modulo no lo redeclara",
        },
        "slippage": {
            "state": "assumed",
            "is_measurement": False,
            "note": (
                "el *slippage* es el **supuesto** pesimista de #64 (`assumed`), nunca medido: "
                "medirlo es #62. No se fabrica un *slippage* `measured` para publicar metricas "
                "netas (A12)"
            ),
        },
        "net_metrics": {
            "state": "not_computable",
            "reason": NET_METRICS_REASON,
            "where": "cfdtrader.backtest.metrics.calculate_metrics",
            "follow_up": ["#62", "#60"],
        },
        "n_variants": len(evaluated),
        "sources": (
            "el coste del motor y el reparto evaluado salen de `declared_cost_model()` y de "
            "`analysis.backtest_report.declared_slippage_assumption()` (#69): el modulo no "
            "declara ningun literal de coste (A3)"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# La tabla comparativa de cuatro filas (A12)
# ─────────────────────────────────────────────────────────────────────────────
def _comparison_block(evaluated: Sequence[Variant]) -> dict[str, object]:
    """Las cuatro filas con Brier, log-loss y operadas, y su delta contra la cruda de #24 (A12).

    La referencia del delta es la variante lineal **cruda** de #24 (`408fead5`, la primera
    entrada del registro de la familia lineal): el delta se **mide**, no se copia.
    """
    reference = next(
        (
            item
            for item in evaluated
            if item.variant_id == BASELINE_VARIANT_ID and not item.calibrated
        ),
        None,
    )
    rows: list[dict[str, object]] = []
    for item in evaluated:
        row: dict[str, object] = {
            "run_sha256": item.run_sha256,
            "variant_id": item.variant_id,
            "calibrated": item.calibrated,
            "source": item.source,
            "n_traded": item.n_traded,
            "zeros": item.zeros,
            "brier_score": item.brier_score,
            "log_loss": item.log_loss_value,
            "pnl_declared_pct_sum": item.pnl_declared_sum,
            "sharpe_per_session": item.sharpe_per_session,
            "basis": "declared_cost",
            "is_validation": False,
        }
        if reference is None:
            row["delta_vs_baseline_raw"] = None
        else:
            row["delta_vs_baseline_raw"] = {
                "reference_run_sha256": reference.run_sha256,
                "reference_variant_id": reference.variant_id,
                "rule": (
                    "`esta variante - linea base cruda de #24`; **negativo** = mejor Brier/log-loss"
                ),
                "brier_score": item.brier_score - reference.brier_score,
                "log_loss": item.log_loss_value - reference.log_loss_value,
                "n_traded": item.n_traded - reference.n_traded,
            }
        rows.append(row)
    return {
        "basis": "declared_cost",
        "is_validation": False,
        "n_rows": len(rows),
        "reference": (
            None
            if reference is None
            else {
                "run_sha256": reference.run_sha256,
                "variant_id": reference.variant_id,
                "brier_score": reference.brier_score,
                "log_loss": reference.log_loss_value,
                "n_traded": reference.n_traded,
                "note": "la linea base **cruda** de #24, medida en esta corrida",
            }
        ),
        "rows": rows,
        "note": (
            "las filas se calculan **en el mismo proceso**, sobre las mismas 500 sesiones de "
            "test, el mismo `SplitPlan`, el mismo coste declarado y el mismo umbral 0,5: lo unico "
            "que cambia es la probabilidad que decide. Ninguna cifra se copia de los informes "
            "congelados de #24/#25 (A12)"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# El informe
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ModelComparisonReport:
    """El informe: payload canonico, hash y los objetos que lo produjeron.

    ``payload`` es el texto que se hashea y **no** incluye ``report_sha256``: un informe no se
    hashea a si mismo.
    """

    as_of: datetime
    report_date: date
    payload: dict[str, object]
    report_sha256: str
    universe: Universe
    features: FeatureFrame
    split_plan: SplitPlan
    lightgbm: LightGBMModel
    candidates: tuple[Candidate, ...]
    registry: Registry
    records: tuple[ExperimentRecord, ...]
    model_digests: Mapping[str, str]
    outcomes: Mapping[str, WriteOutcome | None]
    selection: Mapping[str, object]

    @property
    def report_stem(self) -> str:
        """Nombre base del informe: ``model_comparison_<AAAA-MM-DD>``."""
        return f"{REPORT_PREFIX}_{self.report_date.isoformat()}"

    @property
    def evaluated(self) -> tuple[Variant, ...]:
        """Las variantes medidas, en el orden del registro."""
        return _evaluated(self.candidates)

    def json_text(self) -> str:
        """JSON determinista: mismas entradas y mismo ``as_of`` ⇒ mismo texto byte a byte."""
        published: dict[str, object] = {**self.payload, "report_sha256": self.report_sha256}
        return _json_text(published)

    def write(self, directory: Path) -> tuple[Path, Path]:
        """Escribe el informe en JSON y Markdown. Solo escribe quien lo pida (el CLI)."""
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{self.report_stem}.json"
        markdown_path = directory / f"{self.report_stem}.md"
        json_path.write_text(self.json_text(), encoding="utf-8")
        markdown_path.write_text(render_markdown(self), encoding="utf-8")
        return json_path, markdown_path


def _settings_block(settings: Settings) -> dict[str, object]:
    """La raiz declarada de `--settings`, **sin** rutas absolutas en el payload (A13)."""
    declared = str(settings.data.root)
    return {
        "data_root_declared": declared if not Path(declared).is_absolute() else "<absoluta>",
        "note": (
            "de `settings` solo viaja la raiz declarada y en forma relativa: una ruta absoluta "
            "depende de la maquina y no puede entrar en un payload que se hashea (A13)"
        ),
    }


def _protocol_block(
    *,
    plan: SplitPlan,
    universe: Universe,
    frame: FeatureFrame,
    cost_model: CostModel,
    slippage: SlippageParameter,
) -> dict[str, object]:
    """El protocolo **importado**: plan, features, umbral, bins y coste (A3)."""
    return {
        "source": (
            "el universo y el plan se importan de `analysis.backtest_report` (#69); la matriz de "
            "las cinco familias, de `analysis.feature_frame` (#24); las 10 features, el umbral y "
            "la semilla, de `models.baseline` (#24); los bins, de "
            "`analysis.baseline_report.CALIBRATION_BINS`; el coste y el *slippage*, de "
            "`backtest.costs` (#8/#11). El modulo **no** redeclara ninguno (A3)"
        ),
        "plan_sha256": plan.plan_sha256,
        "plan_params": {
            "n_splits": PHASE1_PLAN.n_splits,
            "test_size": PHASE1_PLAN.test_size,
            "embargo_sessions": PHASE1_PLAN.embargo_sessions,
            "max_train_size": PHASE1_PLAN.max_train_size,
            "label_horizon": PHASE1_PLAN.label_horizon,
        },
        "n_folds": len(plan.folds),
        "n_test": sum(len(fold.test) for fold in plan.folds),
        "n_sessions": plan.n_sessions,
        "not_in_any_test": len(plan.uncovered),
        "purge_total": plan.purge_total,
        "embargo_total": plan.embargo_total,
        "exclusions_are_no_op": plan.exclusions_are_no_op,
        "features": list(BASELINE_FEATURES),
        "n_features": len(BASELINE_FEATURES),
        "decision_threshold": DECISION_THRESHOLD,
        "decision_rule": "`Direction.LONG` si `p >= 0,5`; `Direction.NOTHING` si no",
        "calibration_bins": CALIBRATION_BINS,
        "calibration_rule": CALIBRATION_RULE,
        "calibration_constants": dict(CALIBRATION_HYPERPARAMETERS),
        "cost": {
            "basis": "declared_cost",
            "is_validation": False,
            "model_name": cost_model.name,
            "slippage_state": str(getattr(slippage, "state", "assumed")),
        },
        "cost_basis": "declared_cost",
        "seed": SEED,
        "lightgbm_hyperparameters": dict(LIGHTGBM_HYPERPARAMETERS),
        "series_id": SERIES_ID,
        "universe": {
            "n_sessions": len(universe.inputs),
            "first_session": _date_text(universe.first_session),
            "last_session": _date_text(universe.last_session),
            "n_half_days": frame.n_half_days,
            "n_nulls_in_features": frame.n_nulls_in_features,
        },
        "matrix_sha256": frame.matrix.matrix_sha256,
        "feature_spec_sha256": dict(frame.matrix.feature_spec_sha256),
        "feature_code_version": frame.matrix.feature_code_version,
    }


def _frozen_block(evaluated: Sequence[Variant]) -> dict[str, object]:
    """La verificacion de la linea base contra lo congelado de #24/#25 (A7)."""
    rows: list[dict[str, object]] = []
    for item in evaluated:
        if item.variant_id != BASELINE_VARIANT_ID:
            continue
        reference = FROZEN_BASELINE["calibrated" if item.calibrated else "raw"]
        rows.append(
            {
                "run_sha256": item.run_sha256,
                "calibrated": item.calibrated,
                "n_traded": item.n_traded,
                "brier_score": item.brier_score,
                "log_loss": item.log_loss_value,
                "pnl_declared_sum": item.pnl_declared_sum,
                "pnl_declared_pct_sum": item.pnl_declared_sum,
                "reference": dict(reference),
                "tolerance": FROZEN_TOLERANCE,
            }
        )
    return {
        "source": (
            "#24 (cruda) y #25 (calibrada): la serie se reconstruye desde "
            "`runs/<sha>/model.json` con `numpy` y el almacen, **sin reajustar**"
        ),
        "tolerance": FROZEN_TOLERANCE,
        "rows": rows,
        "verified": len(rows) == 2,
        "rule": (
            "una discrepancia mayor que la tolerancia es **error tipado** "
            "(`FrozenBaselineMismatchError`), nunca una cifra publicada en silencio (A7)"
        ),
    }


def _calibration_parity_block(evaluated: Sequence[Variant]) -> dict[str, object]:
    """La paridad de calibracion fold a fold entre las dos familias (A6).

    Se comparan las variantes **calibradas** de cada familia: la cruda de #24 no publica
    bloque de calibracion (su `model.json` no lo trae) y la cruda de LightGBM tampoco, porque
    no es la que decide. Lo que A6 exige es que las dos familias repartan el *train* de la
    **misma** forma.
    """
    families: dict[str, list[dict[str, object]]] = {}
    for item in evaluated:
        if item.calibrated and item.fold_calibration:
            families.setdefault(item.variant_id, list(item.fold_calibration))
    per_family: dict[str, dict[str, object]] = {
        family: {
            "n_calibration": [int(cast("int", row["n_calibration"])) for row in rows],
            "methods": method_counts([str(row["method"]) for row in rows]),
            "methods_per_fold": [str(row["method"]) for row in rows],
            "run_sha256": None,
        }
        for family, rows in families.items()
    }
    for item in evaluated:
        if item.calibrated and item.variant_id in per_family:
            per_family[item.variant_id]["run_sha256"] = item.run_sha256
    values = list(per_family.values())
    identical = len(values) >= 2 and all(
        entry["n_calibration"] == values[0]["n_calibration"]
        and entry["methods"] == values[0]["methods"]
        for entry in values
    )
    return {
        "per_family": per_family,
        "families": sorted(per_family),
        "identical": identical,
        "n_families_compared": len(values),
        "rule": CALIBRATION_RULE,
        "note": (
            "las dos familias reparten el *train* de cada fold con el **mismo** "
            "`split_train_for_calibration` y eligen metodo con la **misma** "
            "`select_method`: por eso los `n_calibration` y el histograma coinciden fold a fold "
            "(A6)"
        ),
    }


def _payload(
    *,
    as_of: datetime,
    settings: Settings,
    universe: Universe,
    frame: FeatureFrame,
    plan: SplitPlan,
    lightgbm: LightGBMModel,
    candidates: Sequence[Candidate],
    registry: Registry,
    records: Sequence[ExperimentRecord],
    model_digests: Mapping[str, str],
    selection: Mapping[str, object],
    cost_model: CostModel,
    slippage: SlippageParameter,
) -> dict[str, object]:
    """El payload canonico del informe: tipos JSON puros y determinista (A13)."""
    evaluated = _evaluated(candidates)
    verdict = _verdict_block(
        evaluated=evaluated, registry=registry, selection=selection, candidates=candidates
    )
    selected = cast("Mapping[str, object] | None", selection.get("selected"))
    unit_bug_variant = next(
        (
            item
            for item in evaluated
            if selected is not None and item.run_sha256 == str(selected["run_sha256"])
        ),
        next(iter(evaluated), None),
    )
    unit_bug = _unit_bug_block(unit_bug_variant)
    library = cast("Mapping[str, object]", lightgbm.to_payload()["library"])
    raw: dict[str, object] = {
        "analysis": "cfdtrader.analysis.model_comparison",
        "task": "#26",
        "title": (
            "Comparacion medida de la familia lineal y la familia LightGBM, con regla pre-declarada"
        ),
        "variant_id": VARIANT_ID,
        "generated_at": as_of.isoformat(),
        "report_date": as_of.date().isoformat(),
        "hash_format": REPORT_HASH_FORMAT,
        "basis": "declared_cost",
        "is_validation": False,
        "gate": "fail",
        "phase1_ready": False,
        "llm_overlay": "disabled",
        "scheduler": "none",
        "clock": {
            "as_of": as_of.isoformat(),
            "rule": (
                "el modulo no lee el reloj: el instante entra por `--as-of` (obligatorio para "
                "escribir) y `generated_at = as_of`; no hay `datetime.now`, `utcnow`, "
                "`date.today` ni `time.time` en el fuente (A2)"
            ),
        },
        "settings": _settings_block(settings),
        "protocol": _protocol_block(
            plan=plan, universe=universe, frame=frame, cost_model=cost_model, slippage=slippage
        ),
        "frozen_baseline": _frozen_block(evaluated),
        "calibration_parity": _calibration_parity_block(evaluated),
        "model": {
            "library": dict(library),
            "hyperparameters": dict(lightgbm.hyperparameters),
            "features": list(lightgbm.features),
            "seed": lightgbm.seed,
            "n_folds": len(lightgbm.folds),
            "n_trees_per_fold": [fold.n_trees for fold in lightgbm.folds],
            "folds": [
                {
                    "index": fold.index,
                    "n_train": fold.n_train,
                    "n_test": fold.n_test,
                    "train_first_session": fold.train_first_session.isoformat(),
                    "train_last_session": fold.train_last_session.isoformat(),
                    "train_positives": fold.train_positives,
                    "train_base_rate": fold.train_base_rate,
                    "n_trees": fold.n_trees,
                    "booster_model_sha256": hashlib.sha256(
                        fold.booster_model.encode("utf-8")
                    ).hexdigest(),
                    "calibration": fold.calibration.to_payload(),
                }
                for fold in lightgbm.folds
            ],
            "hash_format": MODEL_HASH_FORMAT,
            "note": (
                "el **texto** del booster vive en `runs/<sha>/model.json` (es lo que lo hace "
                "reproducible sin `pickle`); en el informe se publica su sha256 por fold para no "
                "duplicar ~1 MB de texto"
            ),
        },
        "model_sha256": dict(model_digests),
        "registry": {
            "runs_directory": DEFAULT_RUNS_ROOT,
            "n_trials": registry.n_trials,
            "sr_variance": None if registry.n_trials < 2 else registry.sr_variance,
            "registry_sha256": registry.registry_sha256,
            "variant_ids": list(registry.variant_ids),
            "entries": [entry.to_payload() for entry in registry.entries],
            "new_entries": [
                {
                    "run_sha256": record.run_sha256,
                    "variant_id": record.config.variant_id,
                    "calibrated": "calibration_fraction" in record.config.hyperparameters,
                    "model_sha256": model_digests[record.run_sha256],
                }
                for record in records
            ],
            "rule": (
                "`n_trials` y `sr_variance` se derivan del registro, y las dos entradas nuevas "
                "son la cruda y la calibrada de `lightgbm_gbdt_v1`: **no** se re-registra la "
                "linea base, que ya estaba en `runs/` (A9). El **resultado de la escritura** "
                "(`created`/`unchanged`) **no** entra en el payload: un informe seco y uno "
                "escrito tienen que hashear igual (A13)"
            ),
        },
        "variants": [item.to_payload() for item in candidates],
        "not_evaluable": [
            item.to_payload() for item in candidates if isinstance(item, NotEvaluable)
        ],
        "matrix": _matrix_block(evaluated, registry=registry),
        "deflated_sharpe_ratio": verdict["deflated_sharpe_ratio"],
        "probability_of_backtest_overfitting": verdict["probability_of_backtest_overfitting"],
        "verdict": {
            "state": verdict["state"],
            "gate": verdict["gate"],
            "sharpe_units": {
                "dsr_series": "500 sesiones de test, `0.0` donde no opera",
                "registry_series": "solo las operadas (`n_traded`)",
                "note": (
                    "el Sharpe del DSR **no** es el `sharpe_per_session` del registro: los dos se "
                    "publican y los dos son `per_session`, pero el del DSR incluye los ceros de "
                    "las sesiones planas (A8)"
                ),
            },
        },
        "selection": dict(selection),
        "comparison": _comparison_block(evaluated),
        "declared_cost": _declared_cost_block(evaluated, model=cost_model),
        "net_metrics": {
            "state": "not_computable",
            "reason": NET_METRICS_REASON,
            "where": "cfdtrader.backtest.metrics.calculate_metrics",
            "follow_up": ["#62", "#60"],
        },
        "unit_bug_80": unit_bug,
        "calibration_note": {
            "bins": CALIBRATION_BINS,
            "log_loss_epsilon": LOG_LOSS_EPSILON,
            "note": (
                "la curva de fiabilidad de cada variante se reproduce desde "
                "`variants[].deciding_probabilities` y `variants[].outcomes` con los "
                "`calibration_bins` declarados, y la `log_loss` va recortada en "
                "`[epsilon, 1 - epsilon]` como #15: sin el recorte una probabilidad saturada "
                "daria `log(0)`"
            ),
        },
        "limits": {
            "gate": "fail",
            "phase1_ready": False,
            "is_validation": False,
            "statement": (
                "los numeros de este informe comparan modelos; **no** validan la estrategia: la "
                "puerta de Fase 0 sigue en `fail` (#9/#64/#18), el *slippage* es un supuesto (#62) "
                "y el umbral economico no esta decidido (#60)"
            ),
            "net_metrics_state": "not_computable",
            "costs": "declared_not_measured",
            "slippage_state": "assumed (#64), nunca medido",
            "threshold": DECISION_THRESHOLD,
            "threshold_issue": "#27",
            "direction": "larga unica (#78 cubre la corta)",
        },
        "features_limitations": list(FEATURES_LIMITATIONS),
        "limitations": list(LIMITATIONS),
        "does_not_do": [dict(item) for item in REPORT_DOES_NOT_DO],
        "follow_ups": [dict(item) for item in FOLLOW_UPS],
    }
    return raw


#: Limites declarados del informe.
LIMITATIONS: Final[tuple[str, ...]] = (
    "la comparacion es en el espacio de **probabilidad** y de coste **declarado**: el "
    "*slippage* de #64 es un supuesto (`pnl_net_pct` nulo en el 100 % de las operaciones), asi "
    "que ninguna cifra economica es una validacion (#62, #60)",
    "el `pnl_declared_pct` con el que se mide el Sharpe arrastra el desajuste de unidades de "
    "#80: el **orden** entre variantes y las metricas de probabilidad no cambian, el Sharpe si",
    "la direccion es la **larga unica** (`y = 1{ret_long > 0}`): la pata corta es #78",
    "la familia LightGBM se ajusta con una constante **a priori** (`LIGHTGBM_HYPERPARAMETERS`) y "
    "sin busqueda: mejorarla exige un presupuesto declarado y es #82",
    "el PBO se calcula con 10 bloques y la sensibilidad al numero de bloques no se publica",
    "el camino intradia es inerte (ningun decididor declara barreras): toda operacion sale por el "
    "cierre de la sesion (#27)",
)


# ─────────────────────────────────────────────────────────────────────────────
# Informe en prosa (A12)
# ─────────────────────────────────────────────────────────────────────────────
def _number(value: object, *, digits: int = 6) -> str:
    """Un numero del payload en texto legible, o ``null`` si no se pudo medir."""
    if value is None:
        return "`null`"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: ModelComparisonReport) -> str:
    """El informe en Markdown, determinista y sin cifras que no esten en el payload."""
    payload = report.payload
    protocol = cast("dict[str, object]", payload["protocol"])
    universe = cast("dict[str, object]", protocol["universe"])
    parity = cast("dict[str, object]", payload["calibration_parity"])
    per_family = cast("dict[str, dict[str, object]]", parity["per_family"])
    registry = cast("dict[str, object]", payload["registry"])
    matrix = cast("dict[str, object]", payload["matrix"])
    comparison = cast("dict[str, object]", payload["comparison"])
    rows = cast("list[dict[str, object]]", comparison["rows"])
    selection = cast("dict[str, object]", payload["selection"])
    selected = cast("dict[str, object] | None", selection.get("selected"))
    dsr = cast("dict[str, object]", payload["deflated_sharpe_ratio"])
    pbo = cast("dict[str, object]", payload["probability_of_backtest_overfitting"])
    verdict = cast("dict[str, object]", payload["verdict"])
    gate = cast("dict[str, object]", verdict["gate"])
    unit_bug = cast("dict[str, object]", payload["unit_bug_80"])
    cost = cast("dict[str, object]", payload["declared_cost"])
    net = cast("dict[str, object]", payload["net_metrics"])
    frozen = cast("dict[str, object]", payload["frozen_baseline"])
    frozen_rows = cast("list[dict[str, object]]", frozen["rows"])
    model = cast("dict[str, object]", payload["model"])
    library = cast("Mapping[str, str]", model["library"])
    library_name = library["name"]
    library_version = library["version"]
    hyperparameters_text = json.dumps(model["hyperparameters"], ensure_ascii=False, sort_keys=True)
    constants = cast("dict[str, str]", unit_bug["constants"])

    lines: list[str] = [
        "# Comparacion de modelos — familia lineal frente a LightGBM",
        "",
        f"Variante nueva `{payload['variant_id']}` (LightGBM reducido, `min_child_samples` alto). "
        f"Generado el `{payload['generated_at']}` (**declarado**, no leido del reloj). "
        f"`report_sha256 = {report.report_sha256}`.",
        "",
        f"**No es una validacion.** `gate = {payload['gate']}`, "
        f"`phase1_ready = {payload['phase1_ready']}`.",
        "",
        "## Protocolo (importado, no redeclarado)",
        "",
        f"- `plan_sha256 = {protocol['plan_sha256']}` con **{protocol['n_folds']}** folds y "
        f"**{protocol['n_test']}** sesiones de test ({protocol['n_sessions']} en la muestra).",
        f"- Features: las **{protocol['n_features']}** declaradas "
        f"({', '.join(f'`{name}`' for name in cast('list[str]', protocol['features']))}).",
        f"- Umbral de decision **{protocol['decision_threshold']}**; bins de calibracion "
        f"**{protocol['calibration_bins']}**; coste `{protocol['cost_basis']}`.",
        f"- Universo: **{universe['n_sessions']}** sesiones, `{universe['first_session']}` → "
        f"`{universe['last_session']}`; nulos en las 10 features: "
        f"**{universe['n_nulls_in_features']}**.",
        "",
        "## Linea base congelada (#24/#25) verificada",
        "",
        "| variante | operadas | Brier | log-loss | suma declarada | verificada |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in frozen_rows:
        reference = cast("dict[str, object]", row["reference"])
        verified = abs(
            float(cast("float", row["brier_score"]))
            - float(cast("float", reference["brier_score"]))
        ) <= float(cast("float", frozen["tolerance"]))
        lines.append(
            f"| `{row['run_sha256']}` ({'calibrada' if row['calibrated'] else 'cruda'}) | "
            f"{row['n_traded']} | {_number(row['brier_score'])} | {_number(row['log_loss'])} | "
            f"{_number(row['pnl_declared_sum'])} | {verified} |"
        )
    lines.extend(
        [
            f"- Tolerancia de A7: `{frozen['tolerance']}`; {frozen['rule']}",
            "",
            "## Paridad de calibracion (fold a fold)",
            "",
        ]
    )
    for family in sorted(per_family):
        entry = per_family[family]
        lines.append(
            f"- `{family}`: `n_calibration = {entry['n_calibration']}`, "
            f"metodos = `{entry['methods']}`, por fold = `{entry['methods_per_fold']}`."
        )
    lines.extend(
        [
            f"- Identicas fold a fold: **{parity['identical']}**. {parity['note']}",
            "",
            "## Tabla comparativa (4 filas, las mismas 500 sesiones)",
            "",
            "| variante | calibrada | operadas | ceros | Brier | log-loss | Δ Brier vs cruda | "
            "Δ log-loss vs cruda | Δ operadas |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        delta = cast("dict[str, object] | None", row["delta_vs_baseline_raw"])
        lines.append(
            f"| `{row['variant_id']}` | {row['calibrated']} | {row['n_traded']} | "
            f"{row['zeros']} | {_number(row['brier_score'])} | {_number(row['log_loss'])} | "
            f"{_number(None if delta is None else delta['brier_score'])} | "
            f"{_number(None if delta is None else delta['log_loss'])} | "
            f"{_number(None if delta is None else delta['n_traded'])} |"
        )
    lines.extend(
        [
            "",
            f"- {comparison['note']}",
            "",
            "## Regla de seleccion (pre-declarada)",
            "",
            f"- `primary_metric = {selection['primary_metric']}`, "
            f"`tie_breakers = {selection['tie_breakers']}`, "
            f"`family_order = {selection['family_order']}`, "
            f"`tie_tolerance = {selection['tie_tolerance']}`.",
            f"- {selection['rule']}",
            f"- **Estado: `{selection['state']}`.**",
        ]
    )
    if selected is None:
        lines.append(f"- Sin seleccion: {selection.get('reason')}")
    else:
        kind = "calibrada" if selected["calibrated"] else "cruda"
        lines.append(
            f"- Seleccionada: `{selected['variant_id']}` (`{kind}`, "
            f"`{selected['run_sha256']}`) con Brier {_number(selected['brier_score'])}, "
            f"log-loss {_number(selected['log_loss'])} y {selected['n_traded']} operadas."
        )
    header = "| candidato | calibrada | Brier | log-loss | operadas |"
    lines.extend(["", header, "| --- | --- | --- | --- | --- |"])
    for candidate in cast("list[dict[str, object]]", selection["candidates"]):
        lines.append(
            f"| `{candidate['variant_id']}` | {candidate['calibrated']} | "
            f"{_number(candidate['brier_score'])} | {_number(candidate['log_loss'])} | "
            f"{candidate['n_traded']} |"
        )
    blockers = cast("list[dict[str, object]]", selection["blockers"])
    if blockers:
        lines.extend(["", "- **Vetos** (candidato sin metrica):"])
        lines.extend(
            f"  - `{item['run_sha256']}` (`{item['variant_id']}`): {item['reason']}"
            for item in blockers
        )
    lines.extend(
        [
            "",
            "## Correccion por intentos (DSR y PBO de las variantes reales)",
            "",
            f"- Registro: **{registry['n_trials']}** intentos, "
            f"`V[SR] = {_number(registry['sr_variance'])}`, "
            f"`registry_sha256 = {registry['registry_sha256']}`.",
            f"- Matriz: **{matrix['n_observations']}** filas × **{matrix['n_variants']}** "
            f"columnas ({matrix['blocks']} bloques), "
            f"`matrix_matches_registry = {matrix['matrix_matches_registry']}`.",
            f"- DSR: `{_number(dsr.get('dsr'))}` (`{dsr.get('verdict')}`) con "
            f"`sr_observed = {_number(dsr.get('sr_observed'))}`, "
            f"`sr0_expected_max = {_number(dsr.get('sr0_expected_max'))}`, "
            f"`deflation = {dsr.get('deflation')}`.",
            f"- `registry_sharpe_per_session` de la seleccionada: "
            f"`{_number(dsr.get('registry_sharpe_per_session'))}`. "
            f"{cast('dict[str, object]', verdict['sharpe_units'])['note']}",
            f"- PBO: `{_number(pbo.get('pbo'))}` (`{pbo.get('verdict')}`, "
            f"`pbo_max = {pbo.get('pbo_max')}`), metodo `{pbo.get('method')}`, "
            f"`n_combinations_drawn = {pbo.get('n_combinations_drawn')}`.",
            f"- Agregado: **`{gate['aggregate']}`** — {gate['note']}",
            "",
            "## Coste declarado y metricas netas",
            "",
            f"- `basis = {cost['basis']}`, `is_validation = {cost['is_validation']}`; "
            f"*slippage* `{cast('dict[str, object]', cost['slippage'])['state']}`.",
            f"- Metricas netas: `{net['state']}` — {net['reason']}",
            "",
            "## Desajuste de unidades del motor (#80, medido y **no** arreglado)",
            "",
            f"- Operaciones medidas: **{unit_bug['n_operations']}** (`{unit_bug['observed_on']}`).",
            f"- Constante observada por operacion: "
            f"`{_number(unit_bug['observed_per_operation'])}`; termino correcto: "
            f"`{_number(unit_bug['correct_term_per_operation'])}`; diferencia: "
            f"`{_number(unit_bug['difference_per_operation'])}`.",
            f"- Identidad declarada: {constants['identity']}.",
            f"- {unit_bug['statement']}",
            f"- Afecta a: {unit_bug['affects']}",
            f"- **No** afecta a: {unit_bug['does_not_affect']}",
            f"- Seguimiento: {', '.join(cast('list[str]', unit_bug['follow_up']))}.",
            "",
            "## Modelo LightGBM ajustado",
            "",
            f"- Libreria: `{library_name} {library_version}`; "
            f"**{model['n_folds']}** folds; arboles por fold: `{model['n_trees_per_fold']}`.",
            f"- Hiperparametros: `{hyperparameters_text}`.",
            f"- {model['note']}",
            "",
            "## Limitaciones",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in cast("list[str]", payload["limitations"]))
    lines.append("")
    lines.extend(f"- {item}" for item in cast("list[str]", payload["features_limitations"]))
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
# Configuracion registrada y ejecucion (A2, A9, A13)
# ─────────────────────────────────────────────────────────────────────────────
def _registered_hyperparameters(*, calibrated: bool) -> dict[str, object]:
    """Los hiperparametros registrados de cada variante LightGBM (A9).

    La cruda registra **solo** los de LightGBM; la calibrada anade las tres constantes de
    calibracion, que es la convencion de #25: asi las dos tienen `run_sha256` distinto con el
    **mismo** `variant_id`, que es lo que exige «un id por familia» (#26, decision 4).
    """
    if calibrated:
        return {**LIGHTGBM_HYPERPARAMETERS, **CALIBRATION_HYPERPARAMETERS}
    return dict(LIGHTGBM_HYPERPARAMETERS)


def _configuration(*, frame: FeatureFrame, plan: SplitPlan, calibrated: bool) -> ExperimentConfig:
    """La configuracion registrada de una variante LightGBM: identidad del experimento (A9)."""
    return ExperimentConfig(
        variant_id=VARIANT_ID,
        features=BASELINE_FEATURES,
        hyperparameters=_registered_hyperparameters(calibrated=calibrated),
        seed=SEED,
        series_id=SERIES_ID,
        window={
            "first_session": frame.first_session.isoformat(),
            "last_session": frame.last_session.isoformat(),
            "n_sessions": frame.n_design_rows,
            "n_positives": frame.n_positives,
            "design_lag_sessions": frame.design_lag_sessions,
            "plan_sha256": plan.plan_sha256,
            "matrix_sha256": frame.matrix.matrix_sha256,
            "feature_spec_sha256": dict(frame.matrix.feature_spec_sha256),
            "feature_code_version": frame.matrix.feature_code_version,
        },
    )


def _model_text(
    *,
    record: ExperimentRecord,
    payload: Mapping[str, object],
    digest: str,
    calibrated: bool,
) -> str:
    """El ``model.json``: cuarto artefacto del registro, JSON puro y sin `pickle` (A5)."""
    return _json_text(
        {
            "run_sha256": record.run_sha256,
            "model_sha256": digest,
            "hash_format": MODEL_HASH_FORMAT,
            "calibrated": calibrated,
            "model": dict(payload),
            "note": (
                "el **texto** del booster (`model_to_string()`) por fold y las probabilidades y "
                "margenes de *test* publicados: `load_registry` de #16 solo lee `config.json` y "
                "`result.json`, asi que este cuarto fichero no altera el registro, y no hay "
                "`pickle` (A5). El bloque `calibration` solo aparece en la variante calibrada: "
                "es la convencion de #24/#25 que usa la reconstruccion para saber cual es cual"
            ),
        }
    )


def _model_payload(model: LightGBMModel, *, calibrated: bool) -> dict[str, object]:
    """El payload del modelo publicado por esa variante: con o sin bloque de calibracion."""
    payload = model.to_payload()
    return payload if calibrated else _without_calibration(payload)


def _without_calibration(payload: Mapping[str, object]) -> dict[str, object]:
    """El payload del modelo **sin** el bloque de calibracion: la variante cruda de #26.

    Es la convencion de #24/#25: el `model.json` con bloque `calibration` es la **calibrada** y
    el que no lo trae es la **cruda**. Asi la reconstruccion sabe cual es cual sin mirar la
    configuracion.
    """
    folds = cast("list[object]", payload["folds"])
    return {
        **payload,
        "folds": [
            {
                key: value
                for key, value in cast("Mapping[str, object]", item).items()
                if key != "calibration"
            }
            for item in folds
        ],
    }


def analyse(
    *,
    store: Store,
    reports_dir: Path,
    runs_root: Path,
    settings: Settings,
    as_of: datetime,
    write: bool = True,
) -> ModelComparisonReport:
    """Ajusta LightGBM, reconstruye la linea base, registra y escribe el informe (A2, A13).

    ``as_of`` es **obligatorio** y es el unico instante de la corrida: ninguna ruta consulta el
    reloj. ``write=False`` no escribe **nada** (ni el informe ni las carpetas del registro).
    """
    moment = _as_utc(as_of)
    history = load_history(store, series_id=SERIES_ID)
    calendar = load_calendar(years=_calendar_years(history.daily))
    universe = build_inputs(history, calendar=calendar)
    frame = build_feature_frame(store, series_id=SERIES_ID)
    _require_alignment(universe, frame)
    plan = build_split_plan(universe.inputs, params=PHASE1_PLAN)
    horizon = _label_horizon(plan)
    cost_model = declared_cost_model()
    slippage: SlippageParameter = declared_slippage_assumption()

    lightgbm = fit_lightgbm(
        frame.design,
        splits=split_assignments(plan),
        hyperparameters=LIGHTGBM_HYPERPARAMETERS,
        seed=SEED,
        label_horizon=horizon,
    )
    raw = _variant_from_run(
        run_sha256="",
        variant_id=VARIANT_ID,
        calibrated=False,
        source="lightgbm_engine_run",
        measured=_run_variant(
            decided=probabilities(lightgbm, frame.design.frame),
            universe=universe,
            plan=plan,
            frame=frame,
            cost_model=cost_model,
            slippage=slippage,
        ),
        fold_calibration=(),
    )
    calibrated_variant = _variant_from_run(
        run_sha256="",
        variant_id=VARIANT_ID,
        calibrated=True,
        source="lightgbm_engine_run",
        measured=_run_variant(
            decided=calibrated_probabilities(lightgbm, frame.design.frame),
            universe=universe,
            plan=plan,
            frame=frame,
            cost_model=cost_model,
            slippage=slippage,
        ),
        fold_calibration=_lightgbm_fold_calibration(lightgbm),
    )

    records: list[ExperimentRecord] = []
    known: dict[str, Variant] = {}
    outcomes: dict[str, WriteOutcome | None] = {}
    digests: dict[str, str] = {}
    for variant, is_calibrated in ((raw, False), (calibrated_variant, True)):
        config = _configuration(frame=frame, plan=plan, calibrated=is_calibrated)
        record = record_experiment(
            runs_root=runs_root,
            config=config,
            result=ExperimentResult(
                sharpe_per_session=variant.sharpe_per_session,
                n_observations=variant.n_traded,
            ),
            as_of=moment,
            write=write,
        )
        payload_variant = _model_payload(lightgbm, calibrated=is_calibrated)
        digest = _digest(payload_variant)
        model_path = record.directory / MODEL_FILE
        outcome = (
            _write_immutable(
                model_path,
                _model_text(
                    record=record, payload=payload_variant, digest=digest, calibrated=is_calibrated
                ),
            )
            if write
            else WriteOutcome.UNCHANGED
        )
        outcomes[record.run_sha256] = outcome
        digests[record.run_sha256] = digest
        known[record.run_sha256] = dataclasses.replace(
            variant, run_sha256=record.run_sha256, calibrated=is_calibrated
        )
        records.append(record)

    registry = load_registry(runs_root, extra=tuple(records))
    candidates = _candidates(
        registry=registry,
        runs_root=runs_root,
        known=known,
        universe=universe,
        frame=frame,
        plan=plan,
        cost_model=cost_model,
        slippage=slippage,
    )
    selection = selection_block(candidates)
    payload = _payload(
        as_of=moment,
        settings=settings,
        universe=universe,
        frame=frame,
        plan=plan,
        lightgbm=lightgbm,
        candidates=candidates,
        registry=registry,
        records=tuple(records),
        model_digests=digests,
        selection=selection,
        cost_model=cost_model,
        slippage=slippage,
    )
    report = ModelComparisonReport(
        as_of=moment,
        report_date=moment.date(),
        payload=payload,
        report_sha256=_digest(payload),
        universe=universe,
        features=frame,
        split_plan=plan,
        lightgbm=lightgbm,
        candidates=candidates,
        registry=registry,
        records=tuple(records),
        model_digests=digests,
        outcomes=outcomes,
        selection=selection,
    )
    if write:
        json_path, markdown_path = report.write(reports_dir)
        logger.info("informe de comparacion: {} y {}", json_path, markdown_path)
    return report


def _parse_as_of(value: str | None) -> datetime:
    """El instante declarado de la CLI: **obligatorio**, y nunca tomado del reloj (A2)."""
    if value is None:
        raise MissingAsOfError(
            "falta --as-of: escribir el informe exige un instante declarado (el modulo no lee el "
            "reloj) y sin el **no se escribe ningun fichero** (A2)"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise InvalidAsOfError(f"--as-of no es un ISO-8601 valido: {error} (A2)") from error
    return _as_utc(parsed)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del informe de comparacion.

    Codigos de salida: ``0`` = informe escrito (aunque la puerta siga en `fail`, que es un
    resultado legitimo y declarado); ``2`` = falta o no es valido ``--as-of``, falta un dataset
    o el registro no da `n_trials` ⇒ **no se escribe nada** y el motivo sale por ``stderr``.

    El CLI **no** acepta `--n-trials`, `--pbo-max`, `--confidence-level` ni banderas de ajuste:
    los intentos se derivan del registro y los hiperparametros son la constante declarada (A4, A9).
    """
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description=(
            "Comparacion medida de la familia lineal y la familia LightGBM, con la regla de "
            "seleccion pre-declarada y el DSR/PBO de las variantes del registro (#26)"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=None, help="raiz del almacen")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="directorio de informes (por defecto <data-root>/derived/reports)",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="raiz del registro de experimentos (por defecto ./runs)",
    )
    parser.add_argument("--settings", type=Path, default=None, help="ruta de settings.yaml")
    parser.add_argument(
        "--as-of",
        default=None,
        help="instante declarado ISO-8601, obligatorio para escribir (el modulo no lee el reloj)",
    )
    args = parser.parse_args(argv)

    try:
        moment = _parse_as_of(cast("str | None", args.as_of))
        settings = load_settings(args.settings)
    except (ModelComparisonError, ConfigurationError) as error:
        print(f"no se puede emitir la comparacion de modelos: {error}", file=sys.stderr)
        return 2

    data_root = Path(args.data_root) if args.data_root is not None else Path(settings.data.root)
    reports_dir = (
        Path(args.reports_dir)
        if args.reports_dir is not None
        else data_root / "derived" / "reports"
    )
    runs_root = Path(args.runs_root) if args.runs_root is not None else Path(DEFAULT_RUNS_ROOT)
    try:
        report = analyse(
            store=Store(data_root),
            reports_dir=reports_dir,
            runs_root=runs_root,
            settings=settings,
            as_of=moment,
            write=True,
        )
    except (
        ModelComparisonError,
        FeatureFrameError,
        ExperimentLogError,
        backtest_report.BacktestReportError,
    ) as error:
        print(f"no se puede emitir la comparacion de modelos: {error}", file=sys.stderr)
        return 2

    evaluated = report.evaluated
    logger.info(
        "comparacion: {} variantes medidas de {} del registro; seleccion {}; report_sha256 = {}",
        len(evaluated),
        report.registry.n_trials,
        report.selection.get("state"),
        report.report_sha256,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
