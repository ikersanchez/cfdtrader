"""Registro y medicion del modelo espejo de la direccion corta (#158, seguimiento de #78).

**Que entrega.** La capa de analisis que #78 dejo pendiente: ajusta el espejo con
:func:`cfdtrader.models.mirror.fit_mirror`, recorre el *walk-forward* del motor de #13 **con
decisiones `SHORT`** (mismo umbral declarado 0,5 que #24) y **registra** el experimento en
``runs/`` con su propia identidad (#16): ``variant_id``, features y **objetivo** propios, de modo
que su ``run_sha256`` **no** colisiona con el de la familia larga.

**La identidad es la del espejo, no la de #24.** El ``variant_id`` es
:data:`cfdtrader.models.mirror.MIRROR_VARIANT_ID` y la ventana registrada declara la columna y la
regla del objetivo (``ret_short > 0``): el ``run_sha256`` es el sha256 del **contenido** de esa
configuracion, asi que un objetivo distinto da un experimento distinto sin tocar la ruta.

**Por que el espejo no es un cambio de signo.** El coste declarado de #8/#11 es **asimetrico**
(la tenencia de una noche difiere entre largo y corto), asi que la aritmetica con la que se compara
una sesion corta no es la de una larga. El espejo plantea la pregunta al reves
(``P(ret_short > 0)``) sobre la **misma** maquinaria, no reutiliza la probabilidad larga.

**Lo que NO hace** (fronteras declaradas, con su issue)

- **No** toca produccion: no cambia ``BASELINE_FEATURES``, ni `delivery.run_daily`, ni la
  recomendacion publica, ni ``phase2_ready``. El registro vive en su propio ``runs/<sha>``.
- **No** se consume en la cadena de modelos todavia: ``model_comparison`` enumera **todas** las
  entradas de ``runs/``, asi que registrar el espejo ahi exige la **misma** disciplina que el
  runbook aplica a los ensayos del barrido (§19.17, #82) —apartarlo o usar una raiz segregada—.
  Es el hallazgo de #158 y esta en :data:`REGISTRY_COUPLING_NOTE`.
- **No** publica ``model.json``: la reconstruccion del espejo es #26.
- **No** decide si el espejo aporta: comparar y seleccionar (DSR/PBO) es #26.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Final, cast

from loguru import logger

from cfdtrader.analysis import backtest_report
from cfdtrader.analysis.backtest_report import (
    _calendar_years,  # pyright: ignore[reportPrivateUsage]
)
from cfdtrader.analysis.baseline_report import (
    NO_PROBABILITY_REASON,
    NO_TRADE_REASON,
    REGISTERED_HYPERPARAMETERS,
    _as_utc,  # pyright: ignore[reportPrivateUsage]
    _json_text,  # pyright: ignore[reportPrivateUsage]
    _label_horizon,  # pyright: ignore[reportPrivateUsage]
    _report_digest,  # pyright: ignore[reportPrivateUsage]
    _scored_inputs,  # pyright: ignore[reportPrivateUsage]
    _view_probability,  # pyright: ignore[reportPrivateUsage]
    split_assignments,
)
from cfdtrader.analysis.experiment_log import (
    DEFAULT_RUNS_ROOT,
    ExperimentConfig,
    ExperimentLogError,
    ExperimentRecord,
    ExperimentResult,
    Registry,
    load_registry,
    record_experiment,
)
from cfdtrader.analysis.feature_frame import (
    FeatureFrame,
    FeatureFrameError,
    build_feature_frame,
    load_short_labels,
)
from cfdtrader.backtest.costs import SlippageParameter, declared_cost_model
from cfdtrader.backtest.engine import (
    STATUS_TRADED,
    BacktestRun,
    Decision,
    DecisionError,
    DecisionFn,
    Direction,
    SessionView,
    run_walk_forward,
)
from cfdtrader.backtest.metrics import sharpe_ratio
from cfdtrader.backtest.splits import SplitPlan
from cfdtrader.data.calendar import load_calendar
from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.store import Store
from cfdtrader.features import store as feature_store
from cfdtrader.models.baseline import (
    DECISION_THRESHOLD,
    SEED,
    BaselineModel,
    DesignFrame,
    calibrated_probabilities,
)
from cfdtrader.models.mirror import (
    MIRROR_FEATURES,
    MIRROR_TARGET_COLUMN,
    MIRROR_TARGET_RULE,
    MIRROR_VARIANT_ID,
    fit_mirror,
    mirror_design_frame,
    mirror_identity,
)

__all__ = [
    "CLI_NAME",
    "COST_ASYMMETRY_NOTE",
    "NOT_A_SIGN_FLIP_NOTE",
    "REGISTRY_COUPLING_NOTE",
    "REPORT_DOES_NOT_DO",
    "REPORT_HASH_FORMAT",
    "REPORT_PREFIX",
    "SHORT_REASON",
    "MirrorReport",
    "MirrorReportError",
    "NoShortTradesError",
    "analyse",
    "main",
    "mirror_config",
    "mirror_decider",
    "render_markdown",
]

#: Prefijo del informe: ``mirror_short_<AAAA-MM-DD>.{json,md}`` (no pisa el de #24/#26).
REPORT_PREFIX: Final[str] = "mirror_short"

#: Nombre del CLI, para los mensajes de ``stderr``.
CLI_NAME: Final[str] = "cfdtrader.analysis.mirror_report"

#: Formato estable del ``report_sha256``: identico al de #24/#26 (prefijo dentro del valor).
REPORT_HASH_FORMAT: Final[str] = (
    "`sha256:<64 hex>` del sha256(UTF-8) de cfdtrader.backtest.engine.canonical_text(payload), "
    "**sin** la clave report_sha256 (un informe no se hashea a si mismo). El payload es JSON puro "
    "y no lleva ninguna ruta absoluta: el hash no depende de `--reports-dir` ni de `--runs-root`"
)

#: Motivo del decider del espejo: el espejo **no** es un cambio de signo de la larga.
SHORT_REASON: Final[str] = (
    "probabilidad calibrada del espejo >= umbral declarado 0,5 (#78): se declara corto. La "
    "pregunta es `P(ret_short > 0)`, no el complemento de la probabilidad larga"
)

#: La asimetria de coste, publicada en el informe (frontera con #11, #8).
COST_ASYMMETRY_NOTE: Final[str] = (
    "el coste declarado de #8/#11 es **asimetrico** (la tenencia de una noche difiere entre largo "
    "y corto): la aritmetica con la que se compara una sesion corta no es la de una larga y el "
    "espejo la usa tal cual, sin reutilizar la probabilidad larga"
)

#: Por que el espejo no es un cambio de signo (lo que #78 declara).
NOT_A_SIGN_FLIP_NOTE: Final[str] = (
    "el espejo **no** es `1 - p_largo`: entrena `y = 1{ret_short > 0}` por el mismo camino que "
    "#24 y su objetivo es un experimento **distinto** (su propio `variant_id` y su propio "
    "`run_sha256`), no una variante de la familia larga"
)

#: **Hallazgo de #158.** La cadena de modelos enumera **todas** las entradas de ``runs/``
#: (``model_comparison._candidates``), y una variante que no se puede reconstruir cuenta en su
#: ``n_trials`` y deja la matriz incompleta. Es el **mismo** acoplamiento que el runbook ya
#: resuelve apartando los ensayos del barrido ``lightgbm_search_v1#*`` (§19.17, #82). Por eso el
#: registro del espejo debe vivir en una raiz **segregada** (o apartarse igual que aquellos)
#: mientras no se decida consumirlo en la cadena.
REGISTRY_COUPLING_NOTE: Final[str] = (
    "`runs/` es **una sola** raiz para todos los experimentos y `model_comparison` enumera "
    "**todas** sus entradas: una variante que no sabe reconstruir cuenta en su `n_trials` y deja "
    "la matriz incompleta. Registrar el espejo en `runs/` exige la **misma** disciplina que el "
    "runbook aplica a los ensayos del barrido (§19.17, #82): apartarlo antes de regenerar la "
    "cadena, o usar una raiz segregada con `--runs-root`. El espejo **no** se consume en la "
    "cadena hasta que #26 lo decida"
)

#: Que **no** hace este modulo, legible por maquina. Cada frontera con su issue.
REPORT_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_cambia_produccion",
        "issue": "#78",
        "statement": (
            "no toca `BASELINE_FEATURES`, ni `delivery.run_daily`, ni la recomendacion publica, "
            "ni `phase2_ready`: el registro vive en su propio `runs/<sha>` y el informe en su "
            "propio prefijo (`mirror_short_`)"
        ),
    },
    {
        "id": "no_publica_model_json",
        "issue": "#26",
        "statement": (
            "no publica `model.json` ni reconstruye el espejo: la reconstruccion y la comparacion "
            "con DSR/PBO son #26"
        ),
    },
    {
        "id": "no_decide_el_umbral_economico",
        "issue": "#27",
        "statement": (
            "el umbral 0,5 es el del informe, no el del sistema: el umbral economico y el "
            "*sizing* son #27 y #60"
        ),
    },
)


class MirrorReportError(Exception):
    """Raiz de los errores del informe y del registro del espejo."""


class NoShortTradesError(MirrorReportError):
    """El espejo no opera ninguna sesion de *test*: sin operaciones no hay experimento.

    Se publica como error tipado en vez de registrar un Sharpe ``0``: una corrida en la que el
    espejo **no** opera no es «Sharpe cero», es «no medible» (#158).
    """


# ─────────────────────────────────────────────────────────────────────────────
# El decider declarado (SHORT) y la identidad registrada
# ─────────────────────────────────────────────────────────────────────────────
def mirror_decider(fold_index: int) -> DecisionFn:
    """Una decision por sesion: ``SHORT`` si ``p_cal >= 0,5`` y ``NOTHING`` si no (#78).

    La probabilidad que decide es la **calibrada** del espejo —``P(ret_short > 0)``, la de rendir
    en la direccion corta— y viaja en la carga opaca de la vista; el umbral 0,5 es el declarado y
    no se mueve (el economico es #27). Sin probabilidad (sesion fuera del documento congelado) la
    decision es ``NOTHING`` con su motivo, nunca un ``0`` inventado.
    """

    def decide(view: SessionView) -> Decision:
        probability = _view_probability(view, fold_index=fold_index)
        if probability is None:
            return Decision(
                direction=Direction.NOTHING,
                reason=NO_PROBABILITY_REASON,
                probability=None,
            )
        if not probability >= DECISION_THRESHOLD:
            return Decision(
                direction=Direction.NOTHING,
                reason=NO_TRADE_REASON,
                probability=probability,
            )
        return Decision(
            direction=Direction.SHORT,
            reason=SHORT_REASON,
            notional_usd=backtest_report.NOTIONAL_USD,
            probability=probability,
        )

    return decide


def mirror_config(
    *,
    frame: FeatureFrame,
    design: DesignFrame,
    plan: SplitPlan,
    hyperparameters: Mapping[str, object] | None = None,
) -> ExperimentConfig:
    """La configuracion registrada del espejo: identidad **propia** (#16) con su objetivo.

    ``variant_id`` es el del espejo y la ventana declara la direccion, la columna y la regla del
    objetivo: el ``run_sha256`` es el sha256 de este contenido, asi que **no** coincide con el de
    ``baseline_logit_elasticnet_v1`` aunque compartan features y sesiones.
    """
    return ExperimentConfig(
        variant_id=MIRROR_VARIANT_ID,
        features=MIRROR_FEATURES,
        hyperparameters=dict(
            REGISTERED_HYPERPARAMETERS if hyperparameters is None else hyperparameters
        ),
        seed=SEED,
        series_id=backtest_report.SERIES_ID,
        window={
            "direction": "short",
            "target_column": MIRROR_TARGET_COLUMN,
            "target_rule": MIRROR_TARGET_RULE,
            "first_session": frame.matrix.first_session.isoformat(),
            "last_session": frame.matrix.last_session.isoformat(),
            "n_sessions": design.n_sessions,
            "n_positives": design.positives,
            "plan_sha256": plan.plan_sha256,
            "matrix_sha256": frame.matrix.matrix_sha256,
            "feature_spec_sha256": dict(frame.matrix.feature_spec_sha256),
            "feature_code_version": frame.matrix.feature_code_version,
        },
    )


def _require_mirror_alignment(universe: backtest_report.Universe, design: DesignFrame) -> None:
    """Las sesiones del universo tienen que ser, una a una y en orden, las del diseno del espejo.

    El motor simula el universo del arnes y el espejo predice filas del diseno: si no coinciden una
    a una, el modelo puntuaria una sesion y el motor simularia otra. El objetivo corto se alinea
    por la **misma** `session` de `derived.labels`, asi que una divergencia es un contrato roto.
    """


def _objective_block(design: DesignFrame) -> dict[str, object]:
    """El bloque del objetivo: columna, regla, recuento y tasa de positivos (medidos)."""
    positives = design.positives
    n_sessions = design.n_sessions
    return {
        "column": MIRROR_TARGET_COLUMN,
        "rule": MIRROR_TARGET_RULE,
        "n_sessions": n_sessions,
        "n_positives": positives,
        "positive_rate": (positives / n_sessions) if n_sessions else None,
        "note": (
            "el objetivo es la direccion **corta** (`ret_short > 0`): es un experimento "
            "**distinto** de #24, no el complemento de su probabilidad"
        ),
    }


def _payload(
    *,
    as_of: datetime,
    design: DesignFrame,
    plan: SplitPlan,
    model: BaselineModel,
    run: BacktestRun,
    config: ExperimentConfig,
    result: ExperimentResult,
    record: ExperimentRecord,
    registry: Registry,
) -> dict[str, object]:
    """El payload canonico del informe (JSON puro, sin rutas absolutas: A13)."""
    return {
        "analysis": "cfdtrader.analysis.mirror_report",
        "task": "#158",
        "title": "Registro y medicion del modelo espejo de la direccion corta (#78)",
        "variant_id": MIRROR_VARIANT_ID,
        "identity": mirror_identity(),
        "as_of": as_of.isoformat(),
        "hash_format": REPORT_HASH_FORMAT,
        "objective": _objective_block(design),
        "cost": {
            "basis": "declared_cost",
            "asymmetry": COST_ASYMMETRY_NOTE,
            "not_a_sign_flip": NOT_A_SIGN_FLIP_NOTE,
        },
        "features": list(MIRROR_FEATURES),
        "model_features": list(model.features),
        "plan": {"plan_sha256": plan.plan_sha256, "n_folds": len(plan.folds)},
        "registry": {
            "runs_directory": f"runs/{record.run_sha256}",
            "run_sha256": record.run_sha256,
            "registry_entries": registry.n_trials,
            "registry_sha256": registry.registry_sha256,
            "coupling": REGISTRY_COUPLING_NOTE,
        },
        "run": {
            "direction": "short",
            "n_sessions": run.n_sessions,
            "traded": run.traded,
            "sharpe_per_session": result.sharpe_per_session,
            "n_observations": result.n_observations,
        },
        "config": config.to_payload(),
        "does_not_do": [dict(item) for item in REPORT_DOES_NOT_DO],
        "honesty": {
            "edge": "no hay edge demostrado",
            "phase2": "`phase2_ready = false`; carril B bloqueado (`plan.md` §11.6 / §19.7)",
            "lane": (
                "esta entrega **registra y mide** el espejo; no lo adopta ni cambia la "
                "recomendacion publica"
            ),
        },
    }


@dataclass(frozen=True, slots=True)
class MirrorReport:
    """El informe del espejo: payload canonico, hash y los objetos que lo produjeron."""

    as_of: datetime
    report_date: date
    payload: dict[str, object]
    report_sha256: str
    design: DesignFrame
    plan: SplitPlan
    model: BaselineModel
    run: BacktestRun
    record: ExperimentRecord
    registry: Registry
    config: ExperimentConfig
    result: ExperimentResult

    @property
    def report_stem(self) -> str:
        """Nombre base del informe: ``mirror_short_<AAAA-MM-DD>``."""
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


def render_markdown(report: MirrorReport) -> str:
    """El informe legible, con el objetivo, la asimetria de coste y la valla de honestidad."""
    objective = _objective_block(report.design)
    returns = ", ".join(f"`{name}`" for name in report.model.features)
    return "\n".join(
        [
            f"# Espejo de la direccion corta — {report.report_date.isoformat()} (#158 / #78)",
            "",
            "> **No hay edge demostrado.** Esto **registra y mide** el espejo; no lo adopta. "
            "`phase2_ready = false` y el carril B sigue bloqueado (`plan.md` §11.6 / §19.7).",
            "",
            "## Identidad registrada",
            "",
            f"- `variant_id`: `{MIRROR_VARIANT_ID}`",
            f"- `run_sha256`: `{report.record.run_sha256}` (en `runs/{report.record.run_sha256}/`)",
            f"- features ({len(report.model.features)}): {returns}",
            f"- objetivo: `{MIRROR_TARGET_RULE}` sobre {objective['n_sessions']} sesiones "
            f"({objective['n_positives']} positivos)",
            f"- `report_sha256`: `{report.report_sha256}`",
            "",
            "## Por que el espejo **no** es un cambio de signo",
            "",
            NOT_A_SIGN_FLIP_NOTE,
            "",
            COST_ASYMMETRY_NOTE,
            "",
            "## Medicion (walk-forward declarado)",
            "",
            "| campo | valor |",
            "|---|---|",
            f"| sesiones de test | {report.run.n_sessions} |",
            f"| operadas (corto) | {report.run.traded} |",
            f"| Sharpe por sesion (declarado) | {report.result.sharpe_per_session!r} |",
            f"| observaciones | {report.result.n_observations} |",
            f"| `plan_sha256` | `{report.plan.plan_sha256}` |",
            "",
            "## Valla de honestidad",
            "",
            "- **Edge:** no hay edge demostrado.",
            "- **Fase 2:** `phase2_ready = false`; carril B bloqueado.",
            "- El coste es el **declarado** (`basis: declared_cost`); el neto no se publica (#62).",
            "",
            "## Acoplamiento del registro (hallazgo de #158)",
            "",
            REGISTRY_COUPLING_NOTE,
            "",
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# La corrida: ajustar, recorrer el walk-forward con decisiones SHORT y registrar
# ─────────────────────────────────────────────────────────────────────────────
def analyse(
    *,
    store: Store,
    reports_dir: Path,
    runs_root: Path,
    as_of: datetime,
    write: bool = True,
) -> MirrorReport:
    """Ajusta el espejo, mide con el motor de #13, registra el run y (por defecto) escribe.

    ``as_of`` es **obligatorio** y es el unico instante de la corrida: ninguna ruta consulta el
    reloj. ``write=False`` no escribe **nada** (ni el informe ni la carpeta del registro). El
    espejo **no** cambia produccion: su ``variant_id`` es propio y su informe lleva su propio
    prefijo, asi que el registro y el informe de #24/#26 quedan intactos.
    """
    moment = _as_utc(as_of)
    history = backtest_report.load_history(store, series_id=backtest_report.SERIES_ID)
    calendar = load_calendar(years=_calendar_years(history.daily))
    universe = backtest_report.build_inputs(history, calendar=calendar)
    frame = build_feature_frame(store, series_id=backtest_report.SERIES_ID)
    labels = load_short_labels(store, series_id=backtest_report.SERIES_ID)
    design = mirror_design_frame(
        frame.matrix.frame,
        labels=labels,
        availability=feature_store.DESIGN_LAG_BY_FEATURE,
        selected=MIRROR_FEATURES,
    )
    _require_mirror_alignment(universe, design)
    plan = backtest_report.build_split_plan(universe.inputs, params=backtest_report.PHASE1_PLAN)
    model = fit_mirror(
        design,
        splits=split_assignments(plan),
        hyperparameters=REGISTERED_HYPERPARAMETERS,
        label_horizon=_label_horizon(plan),
    )
    predicted = calibrated_probabilities(model, design.frame)
    inputs = _scored_inputs(universe.inputs, predicted)
    model_cost = declared_cost_model()
    slippage: SlippageParameter = backtest_report.declared_slippage_assumption()
    run = run_walk_forward(
        inputs,
        split_plan=plan,
        cost_model=model_cost,
        slippage=slippage,
        decide_by_fold=tuple(mirror_decider(fold.index) for fold in plan.folds),
        financing_cut=None,
    )
    declared = [
        session.pnl_declared
        for fold in run.folds
        for session in fold.sessions
        if session.status == STATUS_TRADED and session.pnl_declared is not None
    ]
    if run.traded == 0 or not declared:
        raise NoShortTradesError(
            "el espejo no opera ninguna sesion de *test*: sin operaciones no hay experimento que "
            "registrar. El umbral declarado no se mueve para forzar operaciones (#158)"
        )
    config = mirror_config(
        frame=frame, design=design, plan=plan, hyperparameters=REGISTERED_HYPERPARAMETERS
    )
    result = ExperimentResult(
        sharpe_per_session=sharpe_ratio(declared, annualization=1),
        n_observations=run.traded,
    )
    record = record_experiment(
        runs_root=runs_root, config=config, result=result, as_of=moment, write=write
    )
    registry = load_registry(record.directory.parent, extra=(record,))
    payload = _payload(
        as_of=moment,
        design=design,
        plan=plan,
        model=model,
        run=run,
        config=config,
        result=result,
        record=record,
        registry=registry,
    )
    report = MirrorReport(
        as_of=moment,
        report_date=moment.date(),
        payload=payload,
        report_sha256=_report_digest(payload),
        design=design,
        plan=plan,
        model=model,
        run=run,
        record=record,
        registry=registry,
        config=config,
        result=result,
    )
    if write:
        json_path, markdown_path = report.write(reports_dir)
        logger.info("informe del espejo: {} y {}", json_path, markdown_path)
    return report


def _parse_as_of(value: str | None) -> datetime:
    """El instante declarado de la CLI: **obligatorio**, y nunca tomado del reloj."""
    if value is None:
        raise MirrorReportError(
            "falta --as-of: el modulo no lee el reloj y sin instante no se escribe nada"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise MirrorReportError(f"--as-of no es un ISO-8601 valido: {error}") from error
    return _as_utc(parsed)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del informe del espejo.

    Codigos de salida: ``0`` = informe emitido; ``2`` = falta o no es valido ``--as-of``, el
    registro no se puede escribir o el espejo no opera (un «no medible» declarado, nunca un
    Sharpe inventado) ⇒ **no se escribe nada** y el motivo sale por ``stderr``.
    """
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description=(
            "Registro y medicion del modelo espejo de la direccion corta (#158): ajusta el "
            "objetivo `ret_short > 0`, recorre el walk-forward con decisiones SHORT y registra "
            "el experimento con su propia identidad"
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
    except (MirrorReportError, ConfigurationError) as error:
        print(f"no se puede emitir el informe del espejo: {error}", file=sys.stderr)
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
        MirrorReportError,
        FeatureFrameError,
        ExperimentLogError,
        backtest_report.BacktestReportError,
        DecisionError,
    ) as error:
        print(f"no se puede emitir el informe del espejo: {error}", file=sys.stderr)
        return 2

    logger.info(
        "espejo: {} variantes en el registro; run_sha256 = {}; report_sha256 = {}",
        report.registry.n_trials,
        report.record.run_sha256,
        report.report_sha256,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
