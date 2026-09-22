"""Modelo baseline con *purged CV* y calibracion: orquestacion, registro, informe y CLI (#24, #25).

Encadena las piezas ya construidas, sin reimplementar ninguna:

``analysis.backtest_report`` (#69)
    el adaptador ``Store -> SessionInput`` (``load_history``, ``build_inputs``), el plan
    declarado (``build_split_plan``, ``PHASE1_PLAN``), el nocional y los seis baselines
    (``run_all_baselines``).
``analysis.feature_frame`` (#24)
    la matriz de las cinco familias (#19-#23) y el corrimiento de diseno.
``models.baseline`` (#24)
    la logistica con elastic net, fold a fold.
``models.calibration`` (#25)
    el reparto del *train* con purga, la regla del metodo por fold y el calibrador
    serializable sin `pickle`.
``backtest.engine`` (#13), ``backtest.costs`` (#8/#11), ``backtest.metrics`` (#15),
``analysis.experiment_log`` (#16)
    el motor, el coste declarado, las metricas elementales y el registro.

Lo que **si** hace con las probabilidades: las **calibra** dentro del *train* de cada fold,
con la regla Platt/Isotonica (#25), y publica la comparacion **medida** cruda-frente-a-calibrada
en las mismas 500 sesiones de *test*.

Lo que **no** hace: no barre hiperparametros ni compara variantes (#26), no
decide el umbral economico ni el *sizing* (#27), no corre el backtest completo de Fase 2 (#28)
y no escribe ``derived.features_daily`` (#73).

Determinismo (A8): el ``run_sha256`` sale de la configuracion de #16 (contenido, nunca ruta ni
instante), el ``model_sha256`` del canonicamente hasheable de #13 y el ``report_sha256``
tambien. El modulo **no consulta el reloj**: ``as_of`` entra por parametro y el CLI lo exige.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final, cast

from loguru import logger

from cfdtrader.analysis import backtest_report
from cfdtrader.analysis.backtest_report import (
    BaselineOutcome,
    _baseline_row,  # pyright: ignore[reportPrivateUsage]
    _calendar_years,  # pyright: ignore[reportPrivateUsage]
)
from cfdtrader.analysis.experiment_log import (
    DEFAULT_RUNS_ROOT,
    ExperimentConfig,
    ExperimentLogError,
    ExperimentRecord,
    ExperimentResult,
    Registry,
    _write_immutable,  # pyright: ignore[reportPrivateUsage]
    load_registry,
    record_experiment,
)
from cfdtrader.analysis.feature_frame import (
    FEATURES_LIMITATIONS,
    FeatureFrame,
    FeatureFrameError,
    build_feature_frame,
)
from cfdtrader.backtest.costs import SlippageParameter, declared_cost_model
from cfdtrader.backtest.engine import (
    STATUS_TRADED,
    BacktestRun,
    Decision,
    DecisionError,
    DecisionFn,
    Direction,
    SessionInput,
    SessionOutcome,
    SessionView,
    canonical_text,
    run_walk_forward,
)
from cfdtrader.backtest.metrics import (
    LOG_LOSS_EPSILON,
    brier_score,
    calibration_curve,
    log_loss,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)
from cfdtrader.backtest.splits import SplitPlan
from cfdtrader.data.calendar import load_calendar
from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.store import Store, WriteOutcome
from cfdtrader.features import store as feature_store
from cfdtrader.models.baseline import (
    BASELINE_FEATURES,
    DECISION_THRESHOLD,
    DESIGN_LAG_SESSIONS,
    HYPERPARAMETERS,
    SEED,
    BaselineModel,
    FoldFit,
    SplitAssignment,
    calibrated_probabilities,
    fit_baseline,
    long_signal,
    probabilities,
)
from cfdtrader.models.calibration import (
    CALIBRATION_HYPERPARAMETERS,
    METHOD_NONE,
    method_counts,
)

__all__ = [
    "CALIBRATION_BINS",
    "CLI_NAME",
    "LONG_REASON",
    "MODEL_FILE",
    "MODEL_HASH_FORMAT",
    "NO_TRADE_REASON",
    "REPORT_HASH_FORMAT",
    "REPORT_PREFIX",
    "VARIANT_ID",
    "BaselineReport",
    "BaselineReportError",
    "NoTradesError",
    "analyse",
    "main",
    "model_sha256",
    "render_markdown",
]

#: Identidad de la variante registrada: **una** variante, sin barridos (#26).
VARIANT_ID: Final[str] = "baseline_logit_elasticnet_v1"

#: Prefijo del informe: ``baseline_<AAAA-MM-DD>.{json,md}``.
REPORT_PREFIX: Final[str] = "baseline"

#: Nombre del CLI, para los mensajes de ``stderr``.
CLI_NAME: Final[str] = "cfdtrader.analysis.baseline_report"

#: Cuarto artefacto de la carpeta del registro: JSON, **sin** ``pickle`` (A12).
MODEL_FILE: Final[str] = "model.json"

#: Bins de la curva de fiabilidad (A9): 5, declarados.
CALIBRATION_BINS: Final[int] = 5

#: Lo que viaja a la configuracion registrada de #16: los hiperparametros **fijos** del
#: estimador y las **tres** constantes de calibracion (A9). Cambiar cualquiera de ellas cambia
#: el `run_sha256` del experimento, y por eso se registran juntas.
REGISTERED_HYPERPARAMETERS: Final[dict[str, object]] = {
    **HYPERPARAMETERS,
    **CALIBRATION_HYPERPARAMETERS,
}

#: Motivos del decider (A11): distinguibles en la tabla de operaciones.
LONG_REASON: Final[str] = (
    "probabilidad calibrada del modelo >= umbral declarado 0,5 (A11): se declara largo en la "
    "direccion larga unica"
)

NO_TRADE_REASON: Final[str] = (
    "probabilidad calibrada del modelo < umbral declarado 0,5 (A11): no se opera; el umbral "
    "economico y el *sizing* son #27 y #60, no este"
)

#: Formato estable del ``report_sha256`` (A13).
REPORT_HASH_FORMAT: Final[str] = (
    "sha256(UTF-8) de cfdtrader.backtest.engine.canonical_text(payload) **sin** la clave "
    "report_sha256: un informe no se hashea a si mismo. El payload es JSON puro (Decimal como "
    "cadena decimal exacta, fechas en ISO-8601) y no lleva ni la ruta del informe ni ningun "
    "instante distinto del `as_of` declarado: por eso el hash no depende de `--reports-dir` (A8)"
)

#: Formato estable del ``model_sha256`` (A12).
MODEL_HASH_FORMAT: Final[str] = (
    "sha256(UTF-8) de cfdtrader.backtest.engine.canonical_text(payload del modelo), donde el "
    "payload son las features, los hiperparametros, la semilla y, por fold, los coeficientes, el "
    "intercepto y el escalado. Es **contenido**: no lleva ni la ruta ni el instante, asi que dos "
    "corridas distintas dan el mismo ``model_sha256`` y el JSON es suficiente para reproducir "
    "las predicciones sin `pickle`"
)

#: Motivo publicado cuando las metricas netas no se pueden calcular (A10).
NET_METRICS_REASON: Final[str] = (
    "`pnl_net_pct` es `null` en todas las operaciones: el supuesto de #64 no se puede cobrar sin "
    "`R` (#60) y medir el *slippage* es #62, asi que "
    "`cfdtrader.backtest.metrics.calculate_metrics` rechaza la corrida con `MetricsInputError`. "
    "Las cifras publicadas son de coste **declarado**"
)

#: Que **no** hace este modulo, legible por maquina. Cada frontera con su issue.
REPORT_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "calibra_en_el_train",
        "issue": "#26",
        "statement": (
            "**si** calibra las probabilidades: el calibrador (Platt/isotonica) se ajusta con la "
            "cola purgada del train de cada fold y la probabilidad calibrada es la que decide; "
            "comparar modelos, barrer hiperparametros y el DSR/PBO son #26"
        ),
    },
    {
        "id": "no_compara_variantes",
        "issue": "#26",
        "statement": (
            "registra **una** variante y no barre hiperparametros ni compara modelos (LightGBM, "
            "DSR/PBO) : eso es #26"
        ),
    },
    {
        "id": "no_decide_el_umbral_economico",
        "issue": "#27",
        "statement": (
            "el 0,5 de A11 es el umbral del **informe**, no el del sistema: el umbral economico, "
            "el dimensionado y el gate son #27 y #60"
        ),
    },
    {
        "id": "no_es_el_backtest_de_fase_2",
        "issue": "#28",
        "statement": (
            "no corre el backtest completo ni los listones B y C: publica la comparacion contra "
            "los seis baselines de #14 sobre la **misma** muestra"
        ),
    },
    {
        "id": "no_publica_metricas_netas",
        "issue": "#62",
        "statement": (
            "no publica metricas netas: el supuesto de #64 deja `pnl_net_pct` en `null` y "
            "`calculate_metrics` rechaza la corrida a proposito (se demuestra con un test)"
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
            "el calibrador se ajusta **dentro** del train de cada fold: el *holdout* final de "
            "§11.4 no se toca y su definicion es #68"
        ),
    },
)

#: Seguimientos declarados por el informe (A10).
FOLLOW_UPS: Final[tuple[dict[str, str], ...]] = (
    {
        "issue": "#62",
        "topic": "medir el *slippage* real",
        "why": (
            "mientras no exista, `pnl_net_pct` es `null` y las metricas netas no se pueden calcular"
        ),
    },
    {
        "issue": "#60",
        "topic": "decidir `R` y el umbral economico",
        "why": "el supuesto de #64 se declara como porcentaje de `R`, que sigue sin decidirse",
    },
    {
        "issue": "#26",
        "topic": "comparar variantes y el DSR/PBO",
        "why": (
            "aqui se calibra **una** variante con **una** semilla, fold a fold; comparar modelos "
            "(LightGBM) y corregir por intentos es #26"
        ),
    },
    {
        "issue": "#28",
        "topic": "backtest completo y listones B y C",
        "why": (
            "aqui solo se compara contra los seis baselines de #14, en el espacio de "
            "probabilidad y de coste declarado"
        ),
    },
    {
        "issue": "#68",
        "topic": "holdout final intocable",
        "why": (
            "el calibrador se ajusta con la cola del train de cada fold; el *holdout* de §11.4 "
            "sigue sin tocarse y su definicion es #68"
        ),
    },
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados
# ─────────────────────────────────────────────────────────────────────────────
class BaselineReportError(Exception):
    """Raiz de los errores del informe del baseline."""


class MissingAsOfError(BaselineReportError):
    """Escribir el informe exige un instante declarado: el modulo no lee el reloj (A13)."""


class InvalidAsOfError(BaselineReportError):
    """El instante declarado no es un ISO-8601 valido."""


class MisalignedUniverseError(BaselineReportError):
    """El frame de diseno y el universo del arnes no hablan de las mismas sesiones (A3)."""


class NoTradesError(BaselineReportError):
    """El modelo no opera ninguna sesion de *test*: sin operaciones no hay experimento (A10)."""


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades locales (mismo contrato que #13/#16/#69, sin importar sus privados)
# ─────────────────────────────────────────────────────────────────────────────
def _as_utc(value: datetime) -> datetime:
    """Normaliza el instante declarado a UTC; sin zona se interpreta como UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _digest(payload: Mapping[str, object]) -> str:
    """sha256 del texto canonico de #13: la **unica** funcion de hash que se usa."""
    return hashlib.sha256(canonical_text(payload).encode("utf-8")).hexdigest()


def model_sha256(model: BaselineModel) -> str:
    """``model_sha256``: sha256 del payload del modelo (coeficientes, intercepto y escalado)."""
    return _digest(model.to_payload())


def _date_text(value: date | None) -> str | None:
    """Una sesion se publica en ISO-8601; ``None`` sigue siendo ``None``."""
    return None if value is None else value.isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# El decider declarado (A11)
# ─────────────────────────────────────────────────────────────────────────────
def _view_probability(view: SessionView, *, fold_index: int) -> float:
    """La probabilidad que el adaptador dejo en la vista, con su error tipado si no esta.

    El decider **solo** lee la ``SessionView`` (A11): la probabilidad viaja en su carga opaca
    ``context``, que es el unico canal que el motor ofrece sin ensenar el futuro de la sesion.
    """
    value = view.context
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionError(
            f"{view.session.isoformat()}: la vista del fold {fold_index} no trae una probabilidad "
            f"numerica en `context` ({type(value).__name__}): el adaptador la dejo fuera"
        )
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise DecisionError(
            f"{view.session.isoformat()}: la probabilidad {probability!r} del fold {fold_index} "
            "no esta entre 0 y 1"
        )
    return probability


def _decider(fold_index: int) -> DecisionFn:
    """Una decision por sesion: ``LONG`` si ``p_cal >= 0,5`` y ``NOTHING`` si no (A7/A11).

    La probabilidad que decide es la **calibrada** (viaja en la carga opaca de la vista); la
    cruda se sigue publicando, pero no decide. El umbral no se mueve (A11).
    """

    def decide(view: SessionView) -> Decision:
        probability = _view_probability(view, fold_index=fold_index)
        if not probability >= DECISION_THRESHOLD:
            return Decision(
                direction=Direction.NOTHING,
                reason=NO_TRADE_REASON,
                probability=probability,
            )
        return Decision(
            direction=Direction.LONG,
            reason=LONG_REASON,
            notional_usd=backtest_report.NOTIONAL_USD,
            probability=probability,
        )

    return decide


def _scored_inputs(
    inputs: Sequence[SessionInput], predicted: Sequence[float | None]
) -> tuple[SessionInput, ...]:
    """Los ``SessionInput`` con la probabilidad **calibrada** de su fold en la carga opaca."""
    if len(inputs) != len(predicted):
        raise MisalignedUniverseError(
            f"hay {len(inputs)} sesiones y {len(predicted)} probabilidades: el frame de diseno y "
            "el universo del arnes tienen que venir alineados, sesion a sesion y en orden (A3)"
        )
    return tuple(
        dataclasses.replace(item, context=probability)
        for item, probability in zip(inputs, predicted, strict=True)
    )


def _require_alignment(universe: backtest_report.Universe, frame: FeatureFrame) -> None:
    """Las sesiones del universo tienen que ser, una a una y en orden, las del diseno (A3).

    Sin esta comprobacion, el modelo podria estar prediciendo una sesion y el motor simulando
    otra: los dos frames se construyen por caminos distintos y el orden es parte del contrato.
    """
    sessions = tuple(item.session for item in universe.inputs)
    if sessions != frame.design.sessions:
        difference = next(
            (
                (index, left.isoformat(), right.isoformat())
                for index, (left, right) in enumerate(
                    zip(sessions, frame.design.sessions, strict=False)
                )
                if left != right
            ),
            None,
        )
        raise MisalignedUniverseError(
            f"el universo del arnes trae {len(sessions)} sesiones y el diseno "
            f"{frame.design.n_sessions}, y no coinciden una a una: primera discrepancia "
            f"{difference}. Las dos listas se construyen por caminos distintos (`build_inputs` y "
            "la matriz de features) y el orden es parte del contrato (A3)"
        )


def split_assignments(plan: SplitPlan) -> tuple[SplitAssignment, ...]:
    """Traduce el plan de #12 a posiciones de #24, sin copiar ninguna regla del plan."""
    return tuple(
        SplitAssignment(index=fold.index, train=fold.train, test=fold.test) for fold in plan.folds
    )


# ─────────────────────────────────────────────────────────────────────────────
# El informe (A2-A14)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class BaselineReport:
    """El informe del baseline: payload canonico, hash y los objetos que lo produjeron.

    ``payload`` es el texto que se hashea (segun :data:`REPORT_HASH_FORMAT`) y **no** incluye
    ``report_sha256``: un informe no se hashea a si mismo. Los objetos viajan al lado para que
    se puedan comprobar por igualdad sin reconstruirlos desde el JSON.
    """

    as_of: datetime
    report_date: date
    payload: dict[str, object]
    report_sha256: str
    universe: backtest_report.Universe
    features: FeatureFrame
    split_plan: SplitPlan
    model: BaselineModel
    run: BacktestRun
    baselines: tuple[BaselineOutcome, ...]
    record: ExperimentRecord
    registry: Registry
    model_path: Path
    model_outcome: WriteOutcome | None
    config: ExperimentConfig
    result: ExperimentResult
    probabilities: tuple[float | None, ...]
    calibrated: tuple[float | None, ...]

    @property
    def report_stem(self) -> str:
        """Nombre base del informe: ``baseline_<AAAA-MM-DD>``."""
        return f"{REPORT_PREFIX}_{self.report_date.isoformat()}"

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


def _json_text(published: Mapping[str, object]) -> str:
    """El mismo JSON determinista que #16/#69: sin `nan` ni `inf`, dos espacios, orden fijo."""
    return json.dumps(published, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


#: La regla del metodo, en el vocabulario de la issue (A4).
CALIBRATION_RULE: Final[str] = (
    "Platt si `n_calibration < 500` e isotonica si no, evaluado **por fold** con el recuento "
    "medido (`PLATT_MAX_CALIBRATION_SESSIONS`); `none` es un estado publicado con su `reason`, "
    "nunca una calibracion mala: ese fold pasa su probabilidad cruda"
)

#: Motivo publicado cuando ningun fold llega a calibrar.
CALIBRATION_NONE_REASON: Final[str] = (
    "ningun fold publico calibrador: los repartos son demasiado cortos, de una sola clase o "
    "invierten el orden, y cada uno escribe su `reason`. La decision se toma con la cruda"
)


def _probability_side(
    *,
    label: str,
    probabilities: Sequence[float],
    outcomes: Sequence[int],
) -> dict[str, object]:
    """Un lado de la comparacion (A6): Brier, log-loss, operadas y su curva de 5 bins.

    Las dos orillas se calculan con las **mismas** funciones de #15 y sobre las **mismas**
    sesiones de *test*: lo unico que cambia es la probabilidad (cruda o calibrada).

    El recorte del log-loss se **publica** (``log_loss_epsilon``): la isotonica satura y la
    probabilidad calibrada llega a ``0,0`` y a ``1,0`` exactos, asi que la cifra esta recortada
    y no se tiene que leer como si no lo estuviera. ``n_at_zero``/``n_at_one`` publican cuantas
    sesiones caen en cada extremo: un artefacto de la interpolacion, no una creencia.
    """
    return {
        "label": label,
        "n_test": len(probabilities),
        "n_positives": int(sum(outcomes)),
        "mean_probability": sum(probabilities) / len(probabilities),
        "brier_score": brier_score(probabilities, outcomes),
        "log_loss": log_loss(probabilities, outcomes),
        "log_loss_epsilon": LOG_LOSS_EPSILON,
        "log_loss_note": (
            "`log_loss` recorta cada probabilidad en `[epsilon, 1 - epsilon]`, como #15: sin el "
            "recorte una probabilidad saturada daria `log(0)` y la cifra no existiria"
        ),
        "n_at_zero": sum(1 for value in probabilities if value == 0.0),
        "n_at_one": sum(1 for value in probabilities if value == 1.0),
        "n_traded": sum(1 for value in probabilities if long_signal(value)),
        "bins": CALIBRATION_BINS,
        "curve": [
            dataclasses.asdict(item)
            for item in calibration_curve(probabilities, outcomes, n_bins=CALIBRATION_BINS)
        ],
    }


def _calibration_method(methods: Sequence[str]) -> str:
    """El metodo publicado del informe: `none`, uno solo o `mixed` cuando conviven (A4)."""
    distinct = sorted({value for value in methods if value != METHOD_NONE})
    if not distinct:
        return METHOD_NONE
    if len(distinct) == 1:
        return distinct[0]
    return "mixed"


def _calibration_per_fold(folds: Sequence[FoldFit]) -> list[dict[str, object]]:
    """El calibrador de cada fold, en orden, sin promediar metodos distintos (A4)."""
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
        for fold in folds
    ]


def _probability_block(
    *,
    raw: Sequence[float],
    calibrated: Sequence[float],
    outcomes: Sequence[int],
    references: Sequence[float],
    folds: Sequence[FoldFit],
) -> dict[str, object]:
    """La comparacion **medida** cruda-frente-a-calibrada sobre las mismas sesiones (A6).

    Las dos veredas salen de la misma aritmetica de #15 y de las mismas 500 sesiones de *test*:
    `before` es la probabilidad cruda y `after` la calibrada, la que decide. `delta` va **con
    signo** (`before - after`, positivo = mejora) y un `delta` que no mejora se publica tal
    cual: no hay umbral inventado ni se cambia el 0,5 para que mejore.
    """
    before = _probability_side(label="raw", probabilities=raw, outcomes=outcomes)
    after = _probability_side(label="calibrated", probabilities=calibrated, outcomes=outcomes)
    brier_before = float(cast("float", before["brier_score"]))
    brier_after = float(cast("float", after["brier_score"]))
    loss_before = float(cast("float", before["log_loss"]))
    loss_after = float(cast("float", after["log_loss"]))
    traded_before = int(cast("int", before["n_traded"]))
    traded_after = int(cast("int", after["n_traded"]))
    calibrations = [fold.calibration for fold in folds]
    methods = [item.method for item in calibrations]
    counts = method_counts(methods)
    n_calibrated = sum(1 for item in calibrations if item.calibrated)
    base_rate_brier = brier_score(references, outcomes)
    base_rate_loss = log_loss(references, outcomes)
    always_brier = brier_score([1.0] * len(outcomes), outcomes)
    always_loss = log_loss([1.0] * len(outcomes), outcomes)
    return {
        "n_test": len(calibrated),
        "n_positives": int(sum(outcomes)),
        "brier_score": brier_after,
        "log_loss": loss_after,
        "headline": (
            "`brier_score`/`log_loss` de la raiz son los de `after`: la probabilidad calibrada "
            "es la publicada y la que decide, y los `before` se publican a su lado para que la "
            "comparacion sea **medida** (A6)"
        ),
        "before": before,
        "after": after,
        "delta": {
            "rule": "`before - after`: **positivo** significa que la calibrada mejora",
            "brier_score": brier_before - brier_after,
            "log_loss": loss_before - loss_after,
            "n_traded": traded_before - traded_after,
            "improves": brier_before - brier_after > 0.0 and loss_before - loss_after > 0.0,
            "note": (
                "ninguna mejora se afirma sin esta resta: un `improves: false` se publica tal "
                "cual, como el resultado negativo de #24"
            ),
        },
        "saturation": {
            "n_at_zero": after["n_at_zero"],
            "n_at_one": after["n_at_one"],
            "n_at_boundary": cast("int", after["n_at_zero"]) + cast("int", after["n_at_one"]),
            "rule": (
                "sesiones de *test* con probabilidad calibrada **exactamente** 0,0 o 1,0: la "
                "isotonica es constante a trozos y satura en los extremos"
            ),
            "note": (
                "un 0,0 o un 1,0 exactos son un **artefacto de saturacion**, no una creencia: se "
                "publican para que nadie los lea como una probabilidad calibrada. El EV y el "
                "*sizing* de #27 no pueden consumir una probabilidad saturada como si fuera una "
                "creencia, y por eso el recuento viaja al informe en vez de esconderse tras el "
                "recorte del log-loss"
            ),
            "follow_up": ["#27"],
        },
        "calibration": {
            "bins": CALIBRATION_BINS,
            "calibrated": n_calibrated > 0,
            "fully_calibrated": n_calibrated == len(folds),
            "method": _calibration_method(methods),
            "methods": counts,
            "n_folds": len(folds),
            "n_folds_calibrated": n_calibrated,
            "rule": CALIBRATION_RULE,
            "per_fold": _calibration_per_fold(folds),
            "curve": [dict(item) for item in cast("list[dict[str, object]]", after["curve"])],
            "note": (
                "`curve` es la **calibrada** (`after`) y el histograma `methods` cuenta folds; "
                "no se promedian metodos distintos dentro de un fold"
            ),
        },
        "references": {
            "base_rate": {
                "rule": "tasa base del **train** de cada fold, aplicada a las sesiones de su test",
                "brier_score": base_rate_brier,
                "log_loss": base_rate_loss,
                "mean_probability": sum(references) / len(references),
            },
            "always_long": {
                "rule": "probabilidad declarada 1,0 en las **mismas** sesiones de test (A9)",
                "brier_score": always_brier,
                "log_loss": always_loss,
                "mean_probability": 1.0,
            },
        },
        "versus_references": {
            "beats_base_rate_brier": brier_after < base_rate_brier,
            "beats_base_rate_log_loss": loss_after < base_rate_loss,
            "beats_always_long_brier": brier_after < always_brier,
            "beats_always_long_log_loss": loss_after < always_loss,
            "note": (
                "**medido**, no afirmado: son las cuatro comparaciones de la probabilidad "
                "**calibrada** contra las dos referencias, sobre las mismas 500 sesiones. Un "
                "`false` se publica tal cual —el modelo baseline no tiene por que batir su "
                "referencia— y no se maquilla ni se cambia el umbral para que bata (A11)"
            ),
        },
        "note": (
            "las dos veredas se calculan sobre las mismas sesiones de *test*: el modelo crudo, "
            "el calibrado, la tasa base del train del fold y `always_long`. Son metricas de "
            "**probabilidad**, no de rentabilidad: la comparacion en el espacio de coste "
            "declarado esta en `comparison`"
        ),
    }


def _declared_cost_block(outcomes: Sequence[SessionOutcome]) -> dict[str, object]:
    """La serie de coste declarado y sus metricas elementales de #15 (A10).

    La serie es ``pnl_declared_pct`` de las sesiones ``traded``, en orden de sesion. Sin
    operaciones no hay metrica que publicar: los valores son ``null`` con su motivo, **nunca**
    ``0``.
    """
    traded = [item for item in outcomes if item.status == STATUS_TRADED]
    series = [item.pnl_declared_pct for item in traded if item.pnl_declared_pct is not None]
    block: dict[str, object] = {
        "basis": "declared_cost",
        "is_validation": False,
        "units": "tanto por uno del nocional (pnl_declared_pct)",
        "n_traded": len(traded),
        "n_observations": len(series),
        "net_metrics": {
            "state": "not_computable",
            "reason": NET_METRICS_REASON,
            "where": "cfdtrader.backtest.metrics.calculate_metrics",
            "follow_up": ["#62", "#60"],
        },
    }
    if not series:
        block.update(
            {
                "mean": None,
                "median": None,
                "sum": None,
                "sharpe_ratio": None,
                "sortino_ratio": None,
                "max_drawdown": None,
                "profit_factor": None,
                "reason": "no hay operaciones: un valor no medido es null, nunca 0",
            }
        )
        return block
    block.update(
        {
            "mean": sum(series) / len(series),
            "median": statistics.median(series),
            "sum": sum(series),
            "sharpe_ratio": sharpe_ratio(series, annualization=1),
            "sortino_ratio": sortino_ratio(series, annualization=1),
            "max_drawdown": max_drawdown(series),
            "profit_factor": profit_factor(series),
            "annualization": 1,
            "sharpe_units": "por sesion (annualization=1): anualizar seria inventar la frecuencia",
        }
    )
    return block


def _probability_series(
    frame: FeatureFrame,
    run: BacktestRun,
    model: BaselineModel,
    raw: Sequence[float | None],
) -> tuple[list[float], list[float], list[int], list[float]]:
    """Cruda, calibrada, etiqueta y tasa base por sesion de *test*, en orden de sesion (A6).

    La calibrada es la que viaja en la decision (A7), asi que se recorre el **plan** (fold a
    fold, en orden de sesion) y no un diccionario: el orden es parte del resultado cuando la
    metrica depende de la secuencia (el drawdown de A10). La cruda sale del mismo contrato de
    `probabilities`, casada por sesion, para que las dos veredas hablen de **las mismas** 500.
    """
    labelled = frame.design.frame.select("session", "y")
    labels = {
        cast("date", row["session"]): int(cast("int", row["y"]))
        for row in labelled.iter_rows(named=True)
    }
    raw_by_session: dict[date, float | None] = dict(zip(frame.design.sessions, raw, strict=True))
    base_rate = {item.index: item.train_base_rate for item in model.folds}
    raw_series: list[float] = []
    calibrated_series: list[float] = []
    outcomes: list[int] = []
    references: list[float] = []
    for fold in run.folds:
        for session in fold.sessions:
            decision = session.decision
            if decision is None or decision.probability is None:
                continue
            raw_probability = raw_by_session[session.session]
            if raw_probability is None:
                raise BaselineReportError(
                    f"la sesion {session.session.isoformat()} es del *test* del fold "
                    f"{fold.index} y no tiene probabilidad cruda: la cruda y la calibrada se "
                    "publican sobre **las mismas** sesiones (A6)"
                )
            raw_series.append(raw_probability)
            calibrated_series.append(decision.probability)
            outcomes.append(labels[session.session])
            references.append(base_rate[fold.index])
    return raw_series, calibrated_series, outcomes, references


def _fold_tables(model: BaselineModel) -> list[dict[str, object]]:
    """Los folds como JSON puro: escalado y coeficientes **por fold** (A7, A12)."""
    return [fold.to_payload() for fold in model.folds]


def _universe_block(universe: backtest_report.Universe, frame: FeatureFrame) -> dict[str, object]:
    """El universo y el objetivo, contados (A3)."""
    return {
        "series_id": universe.series_id,
        "n_sessions": len(universe.inputs),
        "first_session": _date_text(universe.first_session),
        "last_session": _date_text(universe.last_session),
        "n_nulls_in_features": frame.n_nulls_in_features,
        "n_labels": frame.design.n_labels,
        "n_half_days": frame.n_half_days,
        "n_excluded_from_universe": len(universe.excluded),
        "note": (
            "el universo es `derived.labels` ∩ muestra limpia de #52 ∩ `raw.market_daily` con "
            "OHLC completo, importado de #69 (`build_inputs`): aqui no se redefine (A3)"
        ),
    }


def _features_block(frame: FeatureFrame) -> dict[str, object]:
    """Las 10 features declaradas, con su familia y su digest (A4)."""
    families: dict[str, str] = {}
    for feature_set, catalog in feature_store.CATALOG_BY_FEATURE_SET.items():
        for entry in catalog:
            families.setdefault(entry.name, feature_set)
    return {
        "names": list(frame.feature_columns),
        "n_features": len(frame.feature_columns),
        "family_of": {name: families.get(name, "unknown") for name in frame.feature_columns},
        "families_represented": sorted(
            {families.get(name, "unknown") for name in frame.feature_columns}
        ),
        "in_catalog": all(
            name in feature_store.ALL_FEATURE_COLUMNS for name in frame.feature_columns
        ),
        "selection": (
            "lista fijada **a priori**: sin seleccion guiada por datos, luego sin *leakage* por "
            "construccion. Fuera: `sessions_to_opex` (17 nulos de cola, #79), `pendiente_2s10s` y "
            "`pendiente_2s10s_chg_5` (dependencia lineal exacta) y el resto de cada bloque "
            "redundante entre si"
        ),
    }


def _features_vs_sample_block(frame: FeatureFrame) -> dict[str, object]:
    """El tamano muestral frente al numero de features, con la cita de §9 (A5)."""
    n_features = len(frame.feature_columns)
    n_positives = frame.n_positives
    return {
        "n_sessions": frame.n_design_rows,
        "n_features": n_features,
        "n_parameters": n_features + 1,
        "n_positives": n_positives,
        "positives_per_feature": n_positives / n_features,
        "plan_reference": (
            "`plan.md` §9: «con ~250-375 operaciones utiles, mas de ~10-15 features es "
            "temerario»; aqui hay 10 features y " + str(n_positives) + " sesiones positivas"
        ),
    }


def _design_block(frame: FeatureFrame) -> dict[str, object]:
    """El corrimiento de disponibilidad y los digests de la matriz (A2, A8)."""
    return {
        "design_lag_sessions": frame.design_lag_sessions,
        "n_shifted_rows": frame.design.n_shifted_rows,
        "n_nulls_in_design": frame.design.n_nulls_in_features,
        "rule": (
            "la fila de diseno de `t` es la fila **completa** de features de `t-1`, la sesion "
            "anterior del diario (`DESIGN_LAG_SESSIONS = 1`). Ninguna columna se filtra por "
            "`required_as_of`: `atr_norm` declara «cierre de `t`» en `volatility_v1` y «cierre de "
            "`t-1`» en `technical_v1` (#72), asi que un filtro por columna no tendria respuesta "
            "unica (A2)"
        ),
        "matrix_sessions": frame.matrix.n_sessions,
        "matrix_columns": frame.matrix.n_columns,
        "matrix_sha256": frame.matrix.matrix_sha256,
        "feature_code_version": frame.matrix.feature_code_version,
        "feature_spec_sha256": dict(frame.matrix.feature_spec_sha256),
        "duplicated_columns": list(frame.matrix.duplicated_columns),
        "missing_series": list(frame.matrix.missing_series),
    }


def _plan_block(plan: SplitPlan) -> dict[str, object]:
    """El plan *walk-forward* con su eco de #12 y el eco de los no-ops (A6)."""
    params = backtest_report.PHASE1_PLAN
    return {
        "plan_sha256": plan.plan_sha256,
        "source": (
            "`cfdtrader.analysis.backtest_report.build_split_plan(inputs, "
            "params=PHASE1_PLAN)`: el plan de #69 se **importa**, con los 500 test exactamente "
            "iguales y el mismo coste (A6). Este modulo no construye `PlanParams`"
        ),
        "n_splits": params.n_splits,
        "test_size": params.test_size,
        "embargo_sessions": params.embargo_sessions,
        "max_train_size": params.max_train_size,
        "label_horizon": plan.inputs.get("label_horizon"),
        "n_sessions": plan.n_sessions,
        "n_test": sum(len(fold.test) for fold in plan.folds),
        "not_in_any_test": len(plan.uncovered),
        "purge_total": plan.purge_total,
        "embargo_total": plan.embargo_total,
        "embargo_in_train_total": plan.embargo_in_train_total,
        "exclusions_are_no_op": plan.exclusions_are_no_op,
        "exclusions_note": (
            "con `label_horizon = 0` la purga y el embargo son **no-ops estructurales**: se "
            "publican con sus numeros, nunca como un filtro activo (A6)"
        ),
    }


def _registry_block(
    record: ExperimentRecord,
    registry: Registry,
    *,
    model_digest: str,
    model_path: Path,
) -> dict[str, object]:
    """El registro de #16 y el cuarto artefacto (A12).

    La carpeta se publica **relativa** (`runs/<run_sha256>/`, la convencion de #16) y nunca
    como ruta absoluta: la ubicacion de `--runs-root` no forma parte del contenido del informe,
    asi que no puede cambiar su hash (A8). El resultado de la escritura (`created`/`unchanged`)
    tampoco entra: un informe seco y uno escrito tienen que hashear **igual**.
    """
    return {
        "runs_directory": f"{DEFAULT_RUNS_ROOT}/{record.run_sha256}",
        "run_sha256": record.run_sha256,
        "model_file": model_path.name,
        "model_sha256": model_digest,
        "registry_entries": len(registry.entries),
        "registry_sha256": registry.registry_sha256,
        "note": (
            "`record_experiment` de #16 escribe `config.json`, `result.json` y `summary.md` bajo "
            "`runs/<run_sha256>/`; el cuarto artefacto es `model.json` (JSON puro, **sin** "
            "`pickle`) y no rompe el registro, que solo lee los dos primeros (A12)"
        ),
    }


def _comparison_row(
    outcome: BaselineOutcome,
    *,
    n_inputs: int,
    plan_sha256: str,
    strategy: str,
    source: str,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Una fila de la tabla comparativa, con la aritmetica declarada de #69 reutilizada (A14).

    La fila de cada baseline la produce ``_baseline_row`` de #69: es la unica forma de
    garantizar que las siete filas salen de la **misma** aritmetica y el mismo contrato.
    """
    row: dict[str, object] = {
        **_baseline_row(outcome, n_inputs=n_inputs),
        "strategy": strategy,
        "source": source,
        "plan_sha256": plan_sha256,
    }
    if extra is not None:
        row.update(extra)
    return row


def _comparison_block(
    *,
    model_run: BacktestRun,
    baselines: Sequence[BaselineOutcome],
    universe: backtest_report.Universe,
    plan: SplitPlan,
    probability_block: Mapping[str, object],
) -> dict[str, object]:
    """La tabla de las siete filas: el modelo y los seis baselines, misma muestra (A14)."""
    rows = [
        _comparison_row(
            BaselineOutcome(baseline=VARIANT_ID, run=model_run, frequency=None, seed=SEED),
            n_inputs=len(universe.inputs),
            plan_sha256=plan.plan_sha256,
            strategy=VARIANT_ID,
            source="model",
            extra={
                "decision_threshold": DECISION_THRESHOLD,
                "brier_score": probability_block["brier_score"],
                "log_loss": probability_block["log_loss"],
            },
        )
    ]
    rows.extend(
        _comparison_row(
            outcome,
            n_inputs=len(universe.inputs),
            plan_sha256=plan.plan_sha256,
            strategy=outcome.baseline,
            source="baseline",
        )
        for outcome in baselines
    )
    return {
        "basis": "declared_cost",
        "is_validation": False,
        "notional_usd": format(backtest_report.NOTIONAL_USD, "f"),
        "notional_provenance": backtest_report.NOTIONAL_PROVENANCE,
        "plan_sha256": plan.plan_sha256,
        "n_inputs": len(universe.inputs),
        "n_test": sum(len(fold.test) for fold in plan.folds),
        "rows": rows,
        "note": (
            "las siete filas se calculan **en el mismo proceso**, sobre los mismos `inputs`, el "
            "mismo `SplitPlan`, el mismo coste declarado y el mismo nocional importados de #69 "
            "(A14). Los seis `BASELINE_IDS` de #14 estan presentes"
        ),
    }


def _limits_block() -> dict[str, object]:
    """Los limites declarados: el informe no es una validacion."""
    return {
        "gate": "fail",
        "phase1_ready": False,
        "is_validation": False,
        "statement": (
            "los numeros de este informe **no** validan la estrategia: la puerta de Fase 0 sigue "
            "en `fail` (#9/#64/#18) y el informe es de coste declarado sobre un modelo baseline"
        ),
        "net_metrics_state": "not_computable",
        "costs": "declared_not_measured",
        "slippage_state": "assumed (#64), nunca medido",
        "financing_cut": "sin verificar (#59)",
        "prices": "proxy de ^GSPC (#50)",
        "threshold": DECISION_THRESHOLD,
        "threshold_issue": "#27 (umbral economico y sizing)",
        "llm_overlay": "disabled",
        "scheduler": "none",
    }


def _payload(
    *,
    as_of: datetime,
    universe: backtest_report.Universe,
    frame: FeatureFrame,
    plan: SplitPlan,
    model: BaselineModel,
    model_run: BacktestRun,
    baselines: Sequence[BaselineOutcome],
    probability_series: tuple[list[float], list[float], list[int], list[float]],
    model_digest: str,
    registry: Mapping[str, object],
    declared_cost: Mapping[str, object],
) -> dict[str, object]:
    """El payload canonico del informe: tipos JSON puros y determinista (A8, A13)."""
    raw_series, calibrated_series, outcomes, references = probability_series
    probability_block = _probability_block(
        raw=raw_series,
        calibrated=calibrated_series,
        outcomes=outcomes,
        references=references,
        folds=model.folds,
    )
    raw: dict[str, object] = {
        "analysis": "cfdtrader.analysis.baseline_report",
        "task": "#25",
        "variant_id": VARIANT_ID,
        "generated_at": as_of.isoformat(),
        "hash_format": REPORT_HASH_FORMAT,
        "gate": "fail",
        "phase1_ready": False,
        "is_validation": False,
        "llm_overlay": "disabled",
        "scheduler": "none",
        "universe": _universe_block(universe, frame),
        "target": {
            "rule": "ret_long > 0",
            "n_sessions": frame.n_design_rows,
            "n_positives": frame.n_positives,
            "n_negatives": frame.n_design_rows - frame.n_positives,
            "positive_rate": frame.n_positives / frame.n_design_rows,
            "direction": "long",
            "note": (
                "la direccion es la **larga unica**: `y = 1{ret_long > 0}` de `derived.labels`. "
                'No se usa `label_long == "target"`: el motor de #13 realiza el cierre de sesion '
                "(100 % `exit_reason = session_close` en #69), no las barreras, asi que etiquetar "
                "por barrera entrenaria para una salida que el arnes nunca toma. La pata corta es "
                "#78"
            ),
        },
        "features": _features_block(frame),
        "features_vs_sample": _features_vs_sample_block(frame),
        "design": _design_block(frame),
        "plan": _plan_block(plan),
        "hyperparameters": dict(model.hyperparameters),
        "hyperparameters_note": (
            "fijos **a priori**, sin busqueda ni barrido (A7). `penalty: elasticnet` es la "
            "ortografia declarada; en scikit-learn 1.9.1 un `penalty` explicito esta deprecado y "
            "el mismo estimador se pide con `0 < l1_ratio < 1`, que es como se pasa. `max_iter` y "
            "`tol` son cotas declaradas: la convergencia se publica por fold"
        ),
        "seed": model.seed,
        "calibration": {
            "constants": dict(CALIBRATION_HYPERPARAMETERS),
            "rule": CALIBRATION_RULE,
            "where": (
                "la cola purgada del *train* de cada fold (`cfdtrader.models.calibration`, #25); "
                "el *test* no entra ni en el estimador ni en el calibrador"
            ),
            "decides": (
                "la decision usa la probabilidad **calibrada** (`p_cal >= 0,5`); la cruda se "
                "publica en `probability_metrics.before` (A7)"
            ),
            "none_reason": CALIBRATION_NONE_REASON,
        },
        "folds": _fold_tables(model),
        "probability_metrics": probability_block,
        "decision": {
            "threshold": DECISION_THRESHOLD,
            "rule": "`Direction.LONG` si `p_cal >= 0,5`; `Direction.NOTHING` si no (A7/A11)",
            "n_traded": model_run.traded,
            "n_traded_raw": cast("dict[str, object]", probability_block["before"])["n_traded"],
            "n_no_trade": model_run.no_trade,
            "n_skipped": model_run.skipped,
            "trade_rate": model_run.traded / len(calibrated_series) if calibrated_series else None,
            "decider_reads": (
                "solo la `SessionView`: la probabilidad viaja en su carga opaca, y la vista no "
                "expone `high`/`low`/`close` de la sesion en curso (A11)"
            ),
        },
        "declared_cost": dict(declared_cost),
        "registry": dict(registry),
        "model_sha256": model_digest,
        "model_hash_format": MODEL_HASH_FORMAT,
        "comparison": _comparison_block(
            model_run=model_run,
            baselines=baselines,
            universe=universe,
            plan=plan,
            probability_block=probability_block,
        ),
        "net_metrics": {
            "state": "not_computable",
            "reason": NET_METRICS_REASON,
            "where": "cfdtrader.backtest.metrics.calculate_metrics",
            "follow_up": ["#62", "#60"],
            "demonstrated_by": (
                "`tests/test_baseline_report.py` demuestra que `calculate_metrics` lanza "
                "`MetricsInputError` sobre estas mismas `SessionOutcome` (A10)"
            ),
        },
        "limits": _limits_block(),
        "features_limitations": list(FEATURES_LIMITATIONS),
        "limitations": list(LIMITATIONS),
        "does_not_do": [dict(item) for item in REPORT_DOES_NOT_DO],
        "follow_ups": [dict(item) for item in FOLLOW_UPS],
    }
    return raw


#: Limites declarados del informe.
LIMITATIONS: Final[tuple[str, ...]] = (
    "una sola variante y una sola semilla: no hay barrido de hiperparametros ni comparacion "
    "entre modelos (#26), asi que el DSR/PBO de #16 no se calcula aqui",
    "las probabilidades **si** estan calibradas (Platt/isotonica, ajustado con la cola purgada "
    "del train de cada fold): el calibrador no ve el *test*, pero sus puntuaciones son las del "
    "estimador de #24, que si vio el train entero — es el precio de no mover el modelo crudo "
    "(#26 puede refitearlo con el bloque de ajuste)",
    "la serie economica es de coste **declarado** (`pnl_declared_pct`): no incluye el supuesto "
    "de *slippage* de #64 porque no se puede cobrar sin `R` (#60)",
    "el camino intradia es inerte (ningun decididor declara barreras): toda operacion sale por el "
    "cierre de la sesion (#27)",
    "`sessions_to_opex` se queda fuera por sus 17 nulos de cola (#79), y `pendiente_2s10s` y su "
    "`_chg_5` por dependencia lineal exacta",
)


# ─────────────────────────────────────────────────────────────────────────────
# Informe en prosa (A13)
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


def render_markdown(report: BaselineReport) -> str:
    """El informe en Markdown, determinista y sin cifras que no esten en el payload."""
    payload = report.payload
    universe = cast("dict[str, object]", payload["universe"])
    target = cast("dict[str, object]", payload["target"])
    features = cast("dict[str, object]", payload["features"])
    vs_sample = cast("dict[str, object]", payload["features_vs_sample"])
    design = cast("dict[str, object]", payload["design"])
    plan = cast("dict[str, object]", payload["plan"])
    probability = cast("dict[str, object]", payload["probability_metrics"])
    calibration = cast("dict[str, object]", probability["calibration"])
    before = cast("dict[str, object]", probability["before"])
    after = cast("dict[str, object]", probability["after"])
    delta = cast("dict[str, object]", probability["delta"])
    methods = cast("dict[str, int]", calibration["methods"])
    per_fold = cast("list[dict[str, object]]", calibration["per_fold"])
    saturation = cast("dict[str, object]", probability["saturation"])
    references = cast("dict[str, object]", probability["references"])
    base_rate = cast("dict[str, object]", references["base_rate"])
    always_long = cast("dict[str, object]", references["always_long"])
    versus = cast("dict[str, object]", probability["versus_references"])
    decision = cast("dict[str, object]", payload["decision"])
    cost = cast("dict[str, object]", payload["declared_cost"])
    registry = cast("dict[str, object]", payload["registry"])
    comparison = cast("dict[str, object]", payload["comparison"])
    rows = cast("list[dict[str, object]]", comparison["rows"])
    net = cast("dict[str, object]", payload["net_metrics"])
    folds = cast("list[dict[str, object]]", payload["folds"])

    lines: list[str] = [
        f"# Modelo baseline con *purged CV* — `{universe['series_id']}`",
        "",
        f"Variante `{payload['variant_id']}` (logistica elastic net + calibracion "
        f"Platt/isotonica dentro del *train*). Generado el "
        f"`{payload['generated_at']}` (**declarado**, no leido del reloj). "
        f"`report_sha256 = {report.report_sha256}`.",
        "",
        f"**No es una validacion.** `gate = {payload['gate']}`, "
        f"`phase1_ready = {payload['phase1_ready']}`.",
        "",
        "## Universo y objetivo",
        "",
        f"- Sesiones: **{universe['n_sessions']}**, `{universe['first_session']}` → "
        f"`{universe['last_session']}`; nulos en las 10 features: "
        f"**{universe['n_nulls_in_features']}**.",
        f"- Sesiones con media sesion (`is_half_day`): {universe['n_half_days']}.",
        f"- Objetivo: `{target['rule']}` ⇒ **{target['n_positives']}** positivos de "
        f"{target['n_sessions']} ({_number(target['positive_rate'])}), direccion "
        f"`{target['direction']}`.",
        "",
        "## Features frente al tamano muestral",
        "",
        f"- `n_sessions = {vs_sample['n_sessions']}`,"
        f" `n_features = {vs_sample['n_features']}`,"
        f" `n_parameters = {vs_sample['n_parameters']}`,"
        f" `n_positives = {vs_sample['n_positives']}`,"
        f" `positives_per_feature = {_number(vs_sample['positives_per_feature'])}`.",
        f"- `{vs_sample['plan_reference']}`.",
        f"- Features: {', '.join(f'`{name}`' for name in cast('list[str]', features['names']))}.",
        "- Familias representadas: "
        f"{', '.join(cast('list[str]', features['families_represented']))}.",
        "",
        "## Corrimiento de disponibilidad (A2)",
        "",
        f"- `design_lag_sessions = {design['design_lag_sessions']}`, "
        f"`n_shifted_rows = {design['n_shifted_rows']}`, "
        f"`n_nulls_in_design = {design['n_nulls_in_design']}`.",
        f"- `matrix_sha256 = {design['matrix_sha256']}` "
        f"({design['matrix_sessions']} sesiones × {design['matrix_columns']} columnas).",
        f"- Duplicadas comprobadas e iguales: {design['duplicated_columns']}.",
        f"- {design['rule']}",
        "",
        "## Plan de folds (importado de #69)",
        "",
        f"- `plan_sha256 = {plan['plan_sha256']}`",
        f"- `n_splits = {plan['n_splits']}`, `test_size = {plan['test_size']}`, "
        f"`embargo_sessions = {plan['embargo_sessions']}`, "
        f"`max_train_size = {plan['max_train_size']}`",
        f"- Sesiones de *test*: **{plan['n_test']}**; fuera de todo *test*: "
        f"**{plan['not_in_any_test']}**.",
        f"- Purga y embargo: `purge_total = {plan['purge_total']}`, "
        f"`embargo_total = {plan['embargo_total']}`, "
        f"`embargo_in_train_total = {plan['embargo_in_train_total']}`, "
        f"`exclusions_are_no_op = {plan['exclusions_are_no_op']}`. {plan['exclusions_note']}",
        "",
        "## Hiperparametros (fijos a priori)",
        "",
        "```json",
        __import__("json").dumps(payload["hyperparameters"], ensure_ascii=False, sort_keys=True),
        "```",
        "",
        f"- {payload['hyperparameters_note']}",
        "",
        "## Folds ajustados",
        "",
        "| fold | train | test | positivos train | tasa base | iter | convergido | intercepto "
        "| calibracion | n_cal |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for fold, entry in zip(folds, per_fold, strict=True):
        lines.append(
            f"| {fold['index']} | {fold['n_train']} ({fold['train_first_session']} → "
            f"{fold['train_last_session']}) | {fold['n_test']} | {fold['train_positives']} | "
            f"{_number(fold['train_base_rate'])} | {fold['n_iter']} | {fold['converged']} | "
            f"{_number(fold['intercept'])} | {entry['method']} | {entry['n_calibration']} |"
        )
    lines.extend(
        [
            "",
            "- El escalado (`mean`, `scale`) y los coeficientes de **cada** fold viajan al "
            "informe y a `model.json`: se ajustan **solo** con el train de ese fold (A7).",
            "- El calibrador de cada fold se ajusta con la **cola** purgada de ese mismo train "
            "(`calibration_positions`); el *test* no entra ni en el estimador ni en el "
            "calibrador (A2/A3).",
            "",
            "## Metricas de probabilidad (cruda frente a calibrada)",
            "",
            f"- Sesiones de test: **{probability['n_test']}** "
            f"({probability['n_positives']} positivos), las mismas en las dos veredas.",
            f"- Cruda (`before`): `brier_score = {_number(before['brier_score'])}`, "
            f"`log_loss = {_number(before['log_loss'])}`, `n_traded = {before['n_traded']}`, "
            f"`mean_probability = {_number(before['mean_probability'])}`.",
            f"- Calibrada (`after`, la que decide): "
            f"`brier_score = {_number(after['brier_score'])}`, "
            f"`log_loss = {_number(after['log_loss'])}`, `n_traded = {after['n_traded']}`, "
            f"`mean_probability = {_number(after['mean_probability'])}`.",
            f"- `delta` ({delta['rule']}): Brier `{_number(delta['brier_score'])}`, log-loss "
            f"`{_number(delta['log_loss'])}`, operadas `{delta['n_traded']}`; "
            f"`improves = {delta['improves']}`. {delta['note']}",
            f"- Recorte declarado del log-loss: "
            f"`log_loss_epsilon = {_number(after['log_loss_epsilon'])}`. "
            f"{after['log_loss_note']}",
            f"- Saturacion (artefacto, no creencia): `n_at_zero = {saturation['n_at_zero']}`, "
            f"`n_at_one = {saturation['n_at_one']}` de {probability['n_test']} sesiones. "
            f"{saturation['note']} Seguimiento: "
            f"{', '.join(cast('list[str]', saturation['follow_up']))}.",
            f"- Calibradores por fold: {calibration['method']} "
            f"(`platt = {methods['platt']}`, `isotonic = {methods['isotonic']}`, "
            f"`none = {methods['none']}`; `n_folds_calibrated = "
            f"{calibration['n_folds_calibrated']}` de {calibration['n_folds']}). "
            f"{calibration['rule']}",
            f"- Referencia `base_rate` (tasa base del train de cada fold): "
            f"`brier_score = {_number(base_rate['brier_score'])}`, "
            f"`log_loss = {_number(base_rate['log_loss'])}`, "
            f"`mean_probability = {_number(base_rate['mean_probability'])}`.",
            f"- Referencia `always_long` (`p = 1,0`): "
            f"`brier_score = {_number(always_long['brier_score'])}`, "
            f"`log_loss = {_number(always_long['log_loss'])}`.",
            f"- La probabilidad **calibrada** **bate** al `base_rate`: en Brier "
            f"`{versus['beats_base_rate_brier']}`, en log-loss "
            f"`{versus['beats_base_rate_log_loss']}`; y a `always_long`: en Brier "
            f"`{versus['beats_always_long_brier']}`, en log-loss "
            f"`{versus['beats_always_long_log_loss']}`. {versus['note']}",
            f"- Curva de fiabilidad con **{calibration['bins']}** bins: "
            f"`calibrated = {calibration['calibrated']}`, `method = {calibration['method']}`.",
            "",
        ]
    )
    lines.extend(
        f"- Bin [{_number(row['lower'])}, {_number(row['upper'])}): "
        f"p = {_number(row['predicted_probability'])}, "
        f"observado = {_number(row['observed_frequency'])}, n = {row['count']}"
        for row in cast("list[dict[str, object]]", calibration["curve"])
    )
    lines.extend(
        [
            "",
            "## Decision declarada (A11)",
            "",
            f"- `DECISION_THRESHOLD = {decision['threshold']}`; {decision['rule']}",
            f"- Operadas: **{decision['n_traded']}** de {probability['n_test']} "
            f"(`no_trade = {decision['n_no_trade']}`, `skipped = {decision['n_skipped']}`, "
            f"tasa = {_number(decision['trade_rate'])}).",
            "",
            "## Coste declarado y rechazo de las metricas netas (A10)",
            "",
            f"- `basis = {cost['basis']}`, `is_validation = {cost['is_validation']}`; serie = "
            f"`pnl_declared_pct` de las **{cost['n_traded']}** sesiones operadas.",
            f"- `mean = {_number(cost['mean'])}`, `median = {_number(cost['median'])}`, "
            f"`sum = {_number(cost['sum'])}`.",
            f"- `sharpe_ratio(annualization=1) = {_number(cost['sharpe_ratio'])}`, "
            f"`sortino_ratio(annualization=1) = {_number(cost['sortino_ratio'])}`.",
            f"- `max_drawdown = {_number(cost['max_drawdown'])}`, "
            f"`profit_factor = {_number(cost['profit_factor'])}`.",
            f"- Metricas netas: `{net['state']}` — {net['reason']}",
            f"- Donde: `{net['where']}`; seguimiento: "
            f"{', '.join(cast('list[str]', net['follow_up']))}.",
            "",
            "## Registro",
            "",
            f"- `run_sha256 = {registry['run_sha256']}` en `{registry['runs_directory']}`.",
            f"- `model_sha256 = {registry['model_sha256']}` en `{registry['model_file']}`.",
            f"- Entradas del registro: {registry['registry_entries']}.",
            "",
            "## Comparacion contra los baselines (misma muestra)",
            "",
            f"Nocional plano declarado: **{comparison['notional_usd']} USD**. "
            f"`n_inputs = {comparison['n_inputs']}`, `n_test = {comparison['n_test']}`, "
            f"`plan_sha256 = {comparison['plan_sha256']}` en las siete filas.",
            "",
            "| fila | fuente | n_test | operadas | no_trade | skipped | media | mediana | suma | "
            "Brier | log-loss |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        summary = cast("dict[str, object]", row["pnl_declared_pct"])
        lines.append(
            f"| `{row['strategy']}` | {row['source']} | {row['n_test']} | {row['traded']} | "
            f"{row['no_trade']} | {row['skipped']} | {_number(summary['mean'])} | "
            f"{_number(summary['median'])} | {_number(summary['sum'])} | "
            f"{_number(row.get('brier_score'))} | {_number(row.get('log_loss'))} |"
        )
    lines.extend(
        [
            "",
            f"- {comparison['note']}",
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
# Ejecucion (A13)
# ─────────────────────────────────────────────────────────────────────────────
def _configuration(store: Store, *, frame: FeatureFrame, plan: SplitPlan) -> ExperimentConfig:
    """La configuracion registrada de la variante: features, hiperparametros, semilla y fuente.

    Los hiperparametros registrados incluyen las **tres** constantes de calibracion (A9):
    cambiarlas cambia el `run_sha256` del experimento, como cualquier otra decision declarada.
    """
    return ExperimentConfig(
        variant_id=VARIANT_ID,
        features=BASELINE_FEATURES,
        hyperparameters=dict(REGISTERED_HYPERPARAMETERS),
        seed=SEED,
        series_id=backtest_report.SERIES_ID,
        window={
            "first_session": frame.first_session.isoformat(),
            "last_session": frame.last_session.isoformat(),
            "n_sessions": frame.n_design_rows,
            "n_positives": frame.n_positives,
            "design_lag_sessions": DESIGN_LAG_SESSIONS,
            "plan_sha256": plan.plan_sha256,
            "matrix_sha256": frame.matrix.matrix_sha256,
            "feature_spec_sha256": dict(frame.matrix.feature_spec_sha256),
            "feature_code_version": frame.matrix.feature_code_version,
        },
    )


def _analyse_feature_frame(store: Store) -> FeatureFrame:
    """Un solo acceso al almacen para la matriz de features."""
    return build_feature_frame(store, series_id=backtest_report.SERIES_ID)


def _label_horizon(plan: SplitPlan) -> tuple[int, ...]:
    """El horizonte por posicion que publica el plan de #12: lo que purga la calibracion (A2)."""
    value = plan.inputs.get("label_horizon")
    if not isinstance(value, (tuple, list)):
        raise BaselineReportError(
            "el plan de #12 no publica `label_horizon` por posicion: sin el no se puede purgar "
            "la cola del train que calibra (A2)"
        )
    return tuple(int(cast("int", item)) for item in cast("Sequence[object]", value))


def analyse(
    *,
    store: Store,
    reports_dir: Path,
    runs_root: Path,
    as_of: datetime,
    write: bool = True,
) -> BaselineReport:
    """Entrena, evalua, registra y (por defecto) escribe el informe del baseline (A13).

    ``as_of`` es **obligatorio** y es el unico instante de la corrida: ninguna ruta consulta el
    reloj. ``write=False`` no escribe **nada** (ni el informe ni la carpeta del registro).
    """
    moment = _as_utc(as_of)
    history = backtest_report.load_history(store, series_id=backtest_report.SERIES_ID)
    calendar = load_calendar(years=_calendar_years(history.daily))
    universe = backtest_report.build_inputs(history, calendar=calendar)
    frame = _analyse_feature_frame(store)
    _require_alignment(universe, frame)
    plan = backtest_report.build_split_plan(universe.inputs, params=backtest_report.PHASE1_PLAN)
    model = fit_baseline(
        frame.design,
        splits=split_assignments(plan),
        hyperparameters=REGISTERED_HYPERPARAMETERS,
        label_horizon=_label_horizon(plan),
    )
    predicted = probabilities(model, frame.design.frame)
    calibrated = calibrated_probabilities(model, frame.design.frame)
    inputs = _scored_inputs(universe.inputs, calibrated)

    model_cost = declared_cost_model()
    slippage: SlippageParameter = backtest_report.declared_slippage_assumption()
    run = run_walk_forward(
        inputs,
        split_plan=plan,
        cost_model=model_cost,
        slippage=slippage,
        decide_by_fold=tuple(_decider(fold.index) for fold in plan.folds),
        financing_cut=None,
    )
    baselines = backtest_report.run_all_baselines(
        universe.inputs, split_plan=plan, cost_model=model_cost, slippage=slippage
    )
    declared_cost = _declared_cost_block(
        [session for fold in run.folds for session in fold.sessions]
    )
    if run.traded == 0:
        raise NoTradesError(
            "el modelo no opera ninguna sesion de *test*: sin operaciones no hay experimento que "
            "registrar ni serie de coste declarado que publicar (A10). El umbral de A11 no se "
            "mueve para forzar operaciones"
        )

    probability_series = _probability_series(frame, run, model, predicted)
    config = _configuration(store, frame=frame, plan=plan)
    result = ExperimentResult(
        sharpe_per_session=sharpe_ratio(
            [
                session.pnl_declared_pct
                for fold in run.folds
                for session in fold.sessions
                if session.status == STATUS_TRADED and session.pnl_declared_pct is not None
            ],
            annualization=1,
        ),
        n_observations=run.traded,
    )
    record = record_experiment(
        runs_root=runs_root, config=config, result=result, as_of=moment, write=write
    )
    model_digest = model_sha256(model)
    registry = load_registry(record.directory.parent, extra=(record,))
    model_path = record.directory / MODEL_FILE
    outcome = (
        _write_immutable(model_path, _model_text(record=record, model=model, digest=model_digest))
        if write
        else WriteOutcome.UNCHANGED
    )

    payload = _payload(
        as_of=moment,
        universe=universe,
        frame=frame,
        plan=plan,
        model=model,
        model_run=run,
        baselines=baselines,
        probability_series=probability_series,
        model_digest=model_digest,
        registry=_registry_block(
            record, registry, model_digest=model_digest, model_path=model_path
        ),
        declared_cost=declared_cost,
    )
    report = BaselineReport(
        as_of=moment,
        report_date=moment.date(),
        payload=payload,
        report_sha256=_digest(payload),
        universe=universe,
        features=frame,
        split_plan=plan,
        model=model,
        run=run,
        baselines=baselines,
        record=record,
        registry=registry,
        model_path=model_path,
        model_outcome=outcome if write else None,
        config=config,
        result=result,
        probabilities=predicted,
        calibrated=calibrated,
    )
    if write:
        json_path, markdown_path = report.write(reports_dir)
        logger.info("informe del baseline: {} y {}", json_path, markdown_path)
    return report


def _model_text(*, record: ExperimentRecord, model: BaselineModel, digest: str) -> str:
    """El ``model.json``: cuarto artefacto del registro, JSON puro y sin `pickle` (A12)."""
    return _json_text(
        {
            "run_sha256": record.run_sha256,
            "model_sha256": digest,
            "hash_format": MODEL_HASH_FORMAT,
            "model": model.to_payload(),
            "note": (
                "coeficientes, intercepto y escalado **por fold**, en JSON: `load_registry` de #16 "
                "solo lee `config.json` y `result.json`, asi que este cuarto fichero no altera el "
                "registro, y no hay `pickle`"
            ),
        }
    )


def _parse_as_of(value: str | None) -> datetime:
    """El instante declarado de la CLI: **obligatorio**, y nunca tomado del reloj (A13)."""
    if value is None:
        raise MissingAsOfError(
            "falta --as-of: escribir el informe exige un instante declarado (el modulo no lee el "
            "reloj) y sin el **no se escribe ningun fichero** (A13)"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise InvalidAsOfError(f"--as-of no es un ISO-8601 valido: {error} (A13)") from error
    return _as_utc(parsed)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del informe del baseline.

    Codigos de salida: ``0`` = informe escrito (aunque la puerta siga en `fail`, que es un
    resultado legitimo y declarado); ``2`` = falta o no es valido ``--as-of``, falta un dataset,
    la muestra no alcanza o el modelo no opera ⇒ **no se escribe nada** y el motivo sale por
    ``stderr``.
    """
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description=(
            "Modelo baseline con *purged CV*: logistica con elastic net sobre las cinco familias, "
            "informe determinista y registro (#24)"
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
    except (BaselineReportError, ConfigurationError) as error:
        print(f"no se puede emitir el informe del baseline: {error}", file=sys.stderr)
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
            as_of=moment,
            write=True,
        )
    except (
        BaselineReportError,
        FeatureFrameError,
        ExperimentLogError,
        backtest_report.BacktestReportError,
    ) as error:
        print(f"no se puede emitir el informe del baseline: {error}", file=sys.stderr)
        return 2

    logger.info(
        "baseline: {} operadas de {} sesiones de test; report_sha256 = {}, run_sha256 = {}",
        report.run.traded,
        sum(len(fold.test) for fold in report.split_plan.folds),
        report.report_sha256,
        report.record.run_sha256,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
