"""Pipeline completo (features -> probabilidad calibrada -> gate -> sizing) en el motor (#28).

Esta es la ultima milla del nucleo cuantitativo de la Fase 2: el modelo de #24/#25 y el gate de
#27 ya existen y estan congelados, pero nadie los habia cableado al motor de #13 sobre los folds
del plan oficial. Aqui se cablean **sin duplicar** nada:

- la historia, el universo, el plan y los seis baselines se consumen de
  ``analysis.backtest_report`` (#69): este modulo **no** vuelve a leer el almacen ni a construir el
  plan de folds;
- la matriz de features, las etiquetas y el frame de diseno vienen de ``analysis.feature_frame``
  (#24), y ``fit_baseline``/``calibrated_probabilities`` deciden con la probabilidad **calibrada**;
- el gate de #27 se evalua **tal cual** (``evaluate_gate`` con las 19 entradas declaradas) y la
  conversion a ``Decision`` la hace ``to_engine_decision`` (el ``open`` de la subasta, #64);
- las metricas son los **helpers exportados** de #15: ``calculate_metrics`` **lanza** con
  ``pnl_net_pct = null``, que es el 100 % de los casos mientras el *slippage* siga supuesto.

Tres brazos declarados (A4-A6):

1. ``oficial``: ``GateParameters()`` sin ningun campo declarado. El gate devuelve
   ``no_recommendation_undecided`` en las ``n_test`` sesiones y **no opera ninguna**.
2. ``escenario`` (S1): los once parametros declarados con su procedencia. Con #64 el *slippage*
   sigue en ``assumed`` y ``c_total_pct`` es ``null``, asi que el tier es **C** por construccion y
   bloquean las reglas 9 (``ev_neto_no_calculable``) y 10 (``tier_no_autorizado``): tampoco opera
   ninguna sesion, y **se mide**, no se afirma.
3. ``coste_declarado``: **no es una validacion**. Es la unica forma de saber si el pipeline
   operaria alguna sesion sin esperar a #60/#62: se re-deriva la decision **declarada** sobre
   ``ev_declared_pct`` (coste declarado, sin el termino supuesto) con la regla literal que el
   propio informe publica. Viaja con ``is_validation = false`` y con el recuento de sesiones en que
   discrepa de los dos brazos del gate.

Que **no** hace, y se declara en el payload en vez de rellenarse:

- ninguna metrica neta: ``net_metrics = not_computable`` con motivo y seguimientos #62 y #60;
- las unidades del motor las declara **#80** (fraccion del nocional): el retorno de coste
  declarado se calcula **aqui**, en %, y el AST de este modulo **no** lee el atributo del P&L
  declarado del motor, lo re-deriva;
- el liston B es una **serie de referencia declarada** (cierre a cierre menos la financiacion
  declarada por noche, importada de la tabla de costes): el liston B de primera clase, con
  posiciones overnight por el motor, es #70.

Sin reloj (``--as-of`` es obligatorio para escribir), sin red, sin escrituras fuera de
``--reports-dir`` y determinista byte a byte: ``report_sha256`` es el sha256 del texto canonico de
#13 sobre el payload sin la clave del hash, con el prefijo ``sha256:``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path
from typing import Final, cast

import polars as pl
from loguru import logger

from cfdtrader.analysis.backtest_report import (
    NOTIONAL_USD,
    PHASE1_PLAN,
    PRICE_PROXY_OF,
    RANDOM_MATCHED_FREQUENCY,
    RANDOM_MATCHED_SEED,
    SERIES_ID,
    BacktestReportError,
    BaselineOutcome,
    History,
    Universe,
    build_inputs,
    build_split_plan,
    label_horizon_sequence,
    load_history,
    run_all_baselines,
)
from cfdtrader.analysis.feature_frame import FeatureFrame, build_feature_frame
from cfdtrader.backtest.baselines import BASELINE_IDS, NO_TRADE
from cfdtrader.backtest.costs import (
    CostBreakdown,
    CostModel,
    Side,
    SlippageParameter,
    cost_breakdown,
    declared_cost_model,
    declared_slippage_assumption,
)
from cfdtrader.backtest.engine import (
    STATUS_NO_TRADE,
    STATUS_SKIPPED,
    STATUS_TRADED,
    BacktestRun,
    Decision,
    DecisionFn,
    Direction,
    SessionInput,
    SessionOutcome,
    SessionView,
    canonical_text,
    run_walk_forward,
)
from cfdtrader.backtest.metrics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_confidence_interval,
    drawdown_metrics,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)
from cfdtrader.backtest.splits import SplitPlan
from cfdtrader.data.calendar import MarketCalendar, load_calendar
from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.store import Store
from cfdtrader.decision.gate import (
    DECISION_THRESHOLD as GATE_DECISION_THRESHOLD,
)
from cfdtrader.decision.gate import (
    TARGET_MIN_COST_MULTIPLE,
    TIER_A,
    TIER_B,
    TIER_C,
    GateOutput,
    GateParameters,
    evaluate_gate,
    to_engine_decision,
)
from cfdtrader.models.baseline import (
    BASELINE_FEATURES,
    DESIGN_LAG_SESSIONS,
    BaselineModel,
    SplitAssignment,
    calibrated_probabilities,
    fit_baseline,
)
from cfdtrader.models.calibration import method_counts

__all__ = [
    "ANALYSIS",
    "ARM_COSTE_DECLARADO",
    "ARM_ESCENARIO",
    "ARM_NAMES",
    "ARM_OFICIAL",
    "BASIS_DECLARED_COST",
    "HASH_PREFIX",
    "METRIC_NAMES",
    "NO_INTERVAL_METRICS",
    "REPORT_HASH_FORMAT",
    "REPORT_PREFIX",
    "RULE_11_BAND",
    "SCENARIO_ID",
    "SERIES_UNITS",
    "SESSION_RULES",
    "TASK",
    "TEMPORAL_MAPPING",
    "AlignmentError",
    "ArmRun",
    "InvalidAsOfError",
    "MissingAsOfError",
    "PipelineReport",
    "PipelineReportError",
    "TableRow",
    "analyse",
    "main",
    "render_markdown",
    "scenario_parameters",
]

#: Identidad del informe: quien lo emite y que tarea lo pide.
ANALYSIS: Final[str] = "cfdtrader.analysis.pipeline_report"
TASK: Final[str] = "#28"
REPORT_PREFIX: Final[str] = "pipeline_backtest"
HASH_PREFIX: Final[str] = "sha256:"
REPORT_HASH_FORMAT: Final[str] = (
    "sha256:<64 hex> del texto canonico (``canonical_text`` de #13) del payload **sin** la clave "
    "``report_sha256``. El prefijo viaja dentro del valor: un digest desnudo lo bloquea "
    "``detect-secrets``"
)
SERIES_UNITS: Final[str] = "% del nocional (puntos porcentuales), una entrada por sesion de *test*"

#: Los tres brazos declarados (A4). El orden es fijo: se publica tal cual, nunca el de un ``set``.
ARM_OFICIAL: Final[str] = "oficial"
ARM_ESCENARIO: Final[str] = "escenario"
ARM_COSTE_DECLARADO: Final[str] = "coste_declarado"
ARM_NAMES: Final[tuple[str, ...]] = (ARM_OFICIAL, ARM_ESCENARIO, ARM_COSTE_DECLARADO)

#: La base de todo lo que se publica aqui: coste **declarado**, nunca medido (A9).
BASIS_DECLARED_COST: Final[str] = "declared_cost"

#: Escenario declarado S1 (tabla de la issue). El broker es un **centinela**: #59 sigue abierta.
SCENARIO_ID: Final[str] = "S1"
SCENARIO_BROKER: Final[str] = "escenario:sin-decidir-#59"

#: Procedencia declarada de cada parametro de S1, en el orden de ``PARAMETER_ISSUES`` del gate.
PROVENANCE: Final[dict[str, str]] = {
    "broker": "centinela declarado: el broker real es #59",
    "risk_per_trade_pct": "§12 regla 2 (techo <= 1 %) y §4.6 (R = 0,5/1,0/1,5 %)",
    "ev_threshold_pct": "§12 regla 9 («p. ej. > 2c»): aqui, 2 x el coste declarado de la tabla",
    "max_daily_loss_pct": "§12 regla 3 (perdida diaria maxima)",
    "max_weekly_loss_pct": "§12 regla 4 (perdida semanal maxima)",
    "max_monthly_loss_pct": "§12 regla 5 (perdida mensual maxima)",
    "r_pct": "§4.6 y §12: tamano de R en %; el gate no lo cablea (#60)",
    "tier_a_cost_multiple": "§12 regla 10, tier A (3c en el enunciado)",
    "tier_b_cost_multiple": "§12 regla 10, tier B (2c en el enunciado)",
    "tier_a_min_probability": "§12 regla 10, probabilidad calibrada minima del tier A",
    "authorized_tiers": "§12 regla 10: al principio solo el tier A",
}

#: Multiplos de S1 sobre sigma: ``stop = 1,0 x sigma`` y ``target = 2 x stop`` (reglas 7 y 8).
SCENARIO_STOP_SIGMA_MULTIPLE: Final[Decimal] = Decimal("1")
SCENARIO_TARGET_STOP_MULTIPLE: Final[Decimal] = Decimal("2")

#: La columna de ``regime_v1`` de la que sale el movimiento esperado y su base declarada.
GARCH_COLUMN: Final[str] = "garch_forecast"
EXPECTED_MOVE_BASIS: Final[str] = "garch_forecast_sigma_1s"

#: Reglas **de sesion** del gate: si alguna bloquea, el brazo declarado tampoco opera (A6).
SESSION_RULES: Final[tuple[str, ...]] = ("13", "14", "1", "3", "4", "5", "15", "17", "18")

#: Regla 11 (§12): el sistema debe operar como maximo el 10-30 % de los dias. Se **mide**.
RULE_11: Final[str] = "11"
RULE_11_BAND: Final[tuple[float, float]] = (0.10, 0.30)

#: Mapeo temporal declarado (A11): el gate no tiene reloj, el informe lo publica.
TEMPORAL_MAPPING: Final[dict[str, str]] = {
    "today": "la propia sesion evaluada (`today = session`)",
    "as_of": "el instante UTC de esa misma sesion, leido del almacen (`as_of` del diario)",
    "entry_px": "el `open` de la subasta de apertura de la sesion (#64)",
    "probability": (
        "la probabilidad **calibrada** del fold cuyo test contiene la sesion (#24/#25): fuera de "
        "todo test no hay prediccion honesta y la sesion no se decide"
    ),
    "expected_move_pct": (
        "raiz del `garch_forecast` de `regime_v1` de la sesion evaluada, en %: el pronostico de la "
        "sesion `t` hecho con los retornos hasta `t-1` (features/regime.py), luego conocido antes "
        "de su `open`; lo declara S1 y el gate no lo deriva"
    ),
    "features": (
        "las 10 features de diseno de la sesion `t` son la fila `t-1` del diario "
        "(`DESIGN_LAG_SESSIONS` = 1, #24)"
    ),
    "clock": "ninguna ruta de este modulo consulta el reloj: `as_of` es un parametro",
}

#: Issues que cierran los parametros sin decidir del brazo oficial (informativo).
UNDECIDED_ISSUES: Final[tuple[str, ...]] = ("#59", "#60")

#: Metrics publicadas **sin** intervalo bootstrap, con su motivo declarado en el bloque (A8).
NO_INTERVAL_METRICS: Final[tuple[str, ...]] = ("profit_factor",)

#: Las dos denominaciones de una tasa de acierto (#92). La `hit_rate` de #28 se estima sobre la
#: **serie de riesgo**: una sesion `no_trade` entra como 0 exacto y diluye la tasa, luego es una
#: tasa **por sesion**. La `p_win` de plan.md §11.6 es **por operacion**: la tasa comparable con
#: ella es `hit_rate_per_trade`. Las dos se publican, cada una con su denominacion declarada.
DENOMINATOR_SESSION: Final[str] = "session"
DENOMINATOR_TRADE: Final[str] = "trade"

#: Las once metricas del informe, en orden estable (y el orden del que salen las semillas).
#: `hit_rate_per_trade` va la **ultima** (posicion 11) para que las diez semillas historicas
#: (43-52) no se muevan: la suya es `_derived_seed(offset=11)` = 53 (#92).
METRIC_NAMES: Final[tuple[str, ...]] = (
    "mean_return_pct",
    "hit_rate",
    "sharpe",
    "sortino",
    "max_drawdown_pct",
    "profit_factor",
    "benchmark_return_pct",
    "excess_return_pct",
    "beta",
    "alpha_pct",
    "hit_rate_per_trade",
)

#: Cuantizacion del nocional: el mismo quantum declarado del gate (#27), que no es publico.
NOTIONAL_QUANTUM: Final[Decimal] = Decimal("0.01")
MONEY_PRECISION: Final[int] = 50

#: Motivo de ``net_metrics``: los estados del *slippage* no se fusionan (A9).
NET_METRICS_REASON: Final[str] = (
    "`pnl_net_pct` es `null` en todas las operaciones: el supuesto de #64 (estado `assumed`, "
    "`is_measurement = false`) no se puede cobrar sin `R` (#60) y medir el *slippage* es #62. "
    "Ninguna metrica neta (Sharpe, Sortino, EV, IC) se fabrica sobre la base neta"
)

#: El informe declara lo que no hace, con la issue que lo cierra (A9, A10).
DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "net_metrics",
        "issue": "#62",
        "statement": (
            "no publica metricas netas: sin *slippage* medido no hay `c_total_pct` y "
            "`calculate_metrics` **lanza** (la tabla es de coste declarado)"
        ),
    },
    {
        "id": "liston_b_first_class",
        "issue": "#70",
        "statement": (
            "el liston B es una **serie de referencia declarada** (cierre a cierre menos la "
            "financiacion declarada por noche); el liston B de primera clase, con posiciones "
            "overnight por el motor, es #70"
        ),
    },
    {
        "id": "sensitivity_sweep",
        "issue": "#86",
        "statement": (
            "declara **un** escenario (S1); el barrido de sensibilidad de los once parametros del "
            "gate es #86"
        ),
    },
    {
        "id": "kill_verdict",
        "issue": "#29",
        "statement": (
            "no emite veredicto de Fase 2: DSR, PBO y la tabla de *kill* de §11.6 son #29"
        ),
    },
    {
        "id": "financing_cut",
        "issue": "#87",
        "statement": (
            "el liston B cobra la financiacion declarada por noche **sin** el instante exacto de "
            "corte (plan.md §21, pregunta 3): se declara, no se asume"
        ),
    },
    {
        "id": "portfolio_rules",
        "issue": "#83",
        "statement": (
            "las reglas 3, 4 y 5 llegan con `None`: la contabilidad de cartera del *kill switch* "
            "es #83"
        ),
    },
)

#: Seguimientos vivos que este informe crea o alimenta.
FOLLOW_UPS: Final[tuple[dict[str, str], ...]] = (
    {
        "issue": "#62",
        "topic": "medir el *slippage*",
        "why": "mientras siga supuesto, ninguna metrica neta es publicable",
    },
    {
        "issue": "#60",
        "topic": "R y umbrales del gate",
        "why": "sin `R` el supuesto de *slippage* no tiene % del nocional y el tier A no se decide",
    },
)


class PipelineReportError(Exception):
    """Raiz de los errores del informe del pipeline."""


class MissingAsOfError(PipelineReportError):
    """Escribir el informe exige un instante declarado: el modulo no lee el reloj (A1, A2)."""


class InvalidAsOfError(PipelineReportError):
    """El instante declarado no es un ISO-8601 valido (A2)."""


class AlignmentError(PipelineReportError):
    """El frame de diseno y el universo del motor no son la misma secuencia de sesiones (A4)."""


# ─────────────────────────────────────────────────────────────────────────────
# Contratos internos
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SessionBundle:
    """Lo que la funcion de decision ve por sesion, colgado de ``SessionView.context``.

    El motor **no lee** ``context``: es la carga opaca del decididor. Aqui lleva la probabilidad
    calibrada y las dos salidas del gate que el brazo necesita, ya evaluadas.
    """

    probability: float
    oficial: GateOutput
    escenario: GateOutput


@dataclass
class ArmLedger:
    """El recuento **medido** de un brazo, en el orden en que el motor recorre las sesiones.

    Es estado local de una corrida: la funcion de decision no toca disco, ni red, ni reloj, y el
    motor sigue siendo puro. Los recuentos se publican ordenados por clave, asi que el hash no
    depende del orden de iteracion de un ``dict``.
    """

    status_counts: dict[str, int] = field(default_factory=dict[str, int])
    blocker_code_counts: dict[str, int] = field(default_factory=dict[str, int])
    rejections: dict[str, int] = field(default_factory=dict[str, int])
    traded: int = 0
    without_expected_move: int = 0

    def observe(self, output: GateOutput) -> None:
        """Anota el estado del gate y cada uno de sus codigos de bloqueo."""
        status = output.status.value
        self.status_counts[status] = self.status_counts.get(status, 0) + 1
        for blocker in output.blockers:
            code = blocker["code"]
            self.blocker_code_counts[code] = self.blocker_code_counts.get(code, 0) + 1

    def reject(self, reason: str) -> None:
        """Anota que el brazo declarado **no** opero esa sesion y por que (A6)."""
        self.rejections[reason] = self.rejections.get(reason, 0) + 1


@dataclass(frozen=True, slots=True)
class ArmRun:
    """Una corrida del motor para un brazo, con su parametrizacion y su salida del gate."""

    name: str
    run: BacktestRun
    params: GateParameters
    outputs: Mapping[date, GateOutput]
    ledger: ArmLedger

    @property
    def n_test(self) -> int:
        """Sesiones de *test* evaluadas por el brazo."""
        return self.run.traded + self.run.no_trade + self.run.skipped


@dataclass(frozen=True, slots=True)
class TableRow:
    """Una fila de la tabla unica: seis baselines y tres listones (A7).

    ``series_pct`` es la serie declarada de la fila en **puntos porcentuales** (una entrada por
    sesion de *test*, en orden), derivada aqui en unidades coherentes (A10) o tomada del diario
    para los listones B y C. ``traded_series_pct`` es la misma derivacion pero **solo** con las
    sesiones operadas: es la serie de la tasa de acierto **por operacion** (#92). Las metricas se
    calculan sobre esa misma serie dividida por 100 (la convencion decimal de #15) y cada una
    viaja con su intervalo bootstrap (A8).
    """

    row: str
    kind: str
    is_invertible: bool
    provenance: str
    series_pct: tuple[float, ...]
    traded_series_pct: tuple[float, ...]
    n_test: int
    traded: int
    no_trade: int
    skipped: int
    run_sha256: str | None
    exit_reason_counts: Mapping[str, int]
    frequency: str | None
    seed: int | None


@dataclass(frozen=True, slots=True)
class PipelineReport:
    """El informe: payload canonico, hash y los objetos que lo produjeron.

    ``payload`` es lo que se hashea (sin ``report_sha256``) y los objetos viajan al lado para que
    los criterios se puedan comprobar por igualdad sin reconstruirlos desde el JSON.
    """

    as_of: datetime
    report_date: date
    payload: dict[str, object]
    report_sha256: str
    reports_dir: Path
    history: History
    universe: Universe
    split_plan: SplitPlan
    features: FeatureFrame
    model: BaselineModel
    cost_model: CostModel
    slippage: SlippageParameter
    cost: CostBreakdown
    arms: tuple[ArmRun, ...]
    baselines: tuple[BaselineOutcome, ...]
    rows: tuple[TableRow, ...]
    test_sessions: tuple[date, ...]

    @property
    def report_stem(self) -> str:
        """Nombre base del informe: ``pipeline_backtest_<AAAA-MM-DD>``."""
        return f"{REPORT_PREFIX}_{self.report_date.isoformat()}"

    def json_text(self) -> str:
        """JSON determinista: mismo ``as_of`` ⇒ mismo texto byte a byte (A2, A3)."""
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

    def arm(self, name: str) -> ArmRun:
        """El brazo con ese nombre, o error tipado: nunca ``None`` silencioso."""
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise PipelineReportError(
            f"el informe no trae el brazo {name!r}: los declarados son {ARM_NAMES} (A4)"
        )

    def row(self, name: str) -> TableRow:
        """La fila con ese nombre, o error tipado."""
        for row in self.rows:
            if row.row == name:
                return row
        raise PipelineReportError(
            f"el informe no trae la fila {name!r}: los seis baselines son {BASELINE_IDS} (A7)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades exactas y serializacion
# ─────────────────────────────────────────────────────────────────────────────
def _as_utc(value: datetime) -> datetime:
    """Normaliza el instante declarado a UTC; sin zona se interpreta como UTC (A2)."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _num(value: Decimal) -> str:
    """``Decimal`` -> cadena decimal exacta, sin notacion cientifica."""
    return format(value, "f")


def _plain(value: object, *, where: str) -> object:
    """Traduce a tipo JSON puro, o falla con error tipado: el JSON nunca lleva ``nan``/``inf``."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PipelineReportError(f"{where}: no se publica `nan` ni `inf` en el informe (A2)")
        return value
    if isinstance(value, Decimal):
        return _num(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        return {str(key): _plain(item, where=f"{where}.{key}") for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        sequence = cast("Sequence[object]", value)
        return [_plain(item, where=f"{where}[{index}]") for index, item in enumerate(sequence)]
    raise PipelineReportError(
        f"{where}: el informe solo admite tipos JSON, Decimal y secuencias; llego "
        f"{type(value).__name__} (A2)"
    )


def _digest(payload: Mapping[str, object]) -> str:
    """``report_sha256`` con el prefijo ``sha256:`` (A3: nunca un digest desnudo)."""
    text = canonical_text(payload)
    return f"{HASH_PREFIX}{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _fraction_text(numerator: int, denominator: int) -> str:
    """Fraccion exacta como ``numerador/denominador`` (nunca un float redondeado)."""
    return f"{numerator}/{denominator}"


def _derived_seed(*, offset: int) -> int:
    """Una semilla derivada de la declarada, **validada** dentro del rango del generador (A8).

    ``numpy.random.RandomState`` exige ``[0, 2**32)``: la reduccion modulo el limite hace la cota
    explicita y es un no-op para toda semilla interior (con la declarada salen 43, 44, ...).
    """
    if offset < 0:
        raise PipelineReportError(
            f"el desplazamiento {offset} no vale para derivar una semilla: tiene que ser >= 0 (A8)"
        )
    seed = (DEFAULT_BOOTSTRAP_SEED + offset) % (2**32)
    if not 0 <= seed <= 2**32 - 1:
        raise PipelineReportError(
            f"la semilla derivada {seed} se sale de [0, 2**32 - 1]: el generador declarado no la "
            "admite (A8)"
        )
    return seed


def _seed_by_metric() -> dict[str, int]:
    """La semilla de cada metrica: derivada de la declarada por su posicion en ``METRIC_NAMES``.

    Depende **solo** de la metrica: la misma serie con la misma metrica da el mismo intervalo en
    cualquier fila o brazo, y eso se puede comprobar sin conocer el orden de la tabla.
    """
    return {name: _derived_seed(offset=index + 1) for index, name in enumerate(METRIC_NAMES)}


def _calendar_years(daily: pl.DataFrame) -> tuple[int, ...]:
    """Anos que el calendario necesita materializar, tomados del dato (nunca del reloj)."""
    sessions = daily.get_column("session")
    first = cast("date", sessions.min())
    last = cast("date", sessions.max())
    return tuple(range(first.year, last.year + 1))


# ─────────────────────────────────────────────────────────────────────────────
# Alineacion y lecturas derivadas del almacen
# ─────────────────────────────────────────────────────────────────────────────
def _require_alignment(universe: Universe, features: FeatureFrame) -> None:
    """El frame de diseno y el universo del motor tienen que ser **la misma** secuencia (A4).

    Si no lo fueran, la traduccion de posiciones del plan a posiciones del diseno seria una
    afirmacion no medida: se comprueba, sesion a sesion, y se falla con error tipado.
    """
    engine_sessions = [item.session for item in universe.inputs]
    design_sessions = list(features.design.sessions)
    if engine_sessions != design_sessions:
        raise AlignmentError(
            f"el frame de diseno trae {len(design_sessions)} sesiones y el universo del motor "
            f"{len(engine_sessions)}, y no son la misma secuencia (A4): las posiciones del plan de "
            "#12 no se pueden traducir a posiciones del diseno sin medirlo"
        )


def _session_instants(daily: pl.DataFrame) -> dict[date, datetime]:
    """El instante UTC declarado de cada sesion, leido del almacen (A11): nunca del reloj."""
    instants: dict[date, datetime] = {}
    for row in daily.select("session", "as_of").iter_rows(named=True):
        session = row["session"]
        moment = row["as_of"]
        if session is None or moment is None:
            continue
        instants[cast("date", session)] = cast("datetime", moment)
    return instants


def _daily_by_session(daily: pl.DataFrame) -> dict[date, Mapping[str, object]]:
    """El diario indexado por sesion, para leer cierre y cierre previo de los listones B y C."""
    out: dict[date, Mapping[str, object]] = {}
    for row in daily.iter_rows(named=True):
        session = row["session"]
        if session is None:
            continue
        out[cast("date", session)] = cast("Mapping[str, object]", row)
    return out


def _expected_move_pct(matrix: pl.DataFrame) -> dict[date, Decimal]:
    """El movimiento esperado declarado de cada sesion: ``sqrt(garch_forecast)``, en % (A11).

    ``garch_forecast`` de ``regime_v1`` es el pronostico de **la sesion ``t``** hecho con los
    retornos hasta ``t-1`` (``features/regime.py``), asi que se conoce antes de su ``open``. Un
    ``null`` **no** se sustituye por 0: la sesion se queda sin movimiento declarado y se cuenta.
    """
    if GARCH_COLUMN not in matrix.columns:
        raise PipelineReportError(
            f"la matriz de features no trae {GARCH_COLUMN!r}: sin el no hay `expected_move_pct` "
            "declarado y el gate no se puede evaluar (A11)"
        )
    moves: dict[date, Decimal] = {}
    for row in matrix.select("session", GARCH_COLUMN).iter_rows(named=True):
        session = row["session"]
        variance = row[GARCH_COLUMN]
        if session is None or variance is None:
            continue
        value = float(cast("float", variance))
        if not math.isfinite(value) or value < 0.0:
            continue
        moves[cast("date", session)] = Decimal(str(math.sqrt(value) * 100.0))
    return moves


# ─────────────────────────────────────────────────────────────────────────────
# Modelo: traduccion del plan, ajuste y probabilidad calibrada
# ─────────────────────────────────────────────────────────────────────────────
def _split_assignments(split_plan: SplitPlan, *, n_design: int) -> tuple[SplitAssignment, ...]:
    """Traduce el plan de #12 a ``SplitAssignment`` de #24: **posiciones del frame de diseno**."""
    assignments: list[SplitAssignment] = []
    for fold in split_plan.folds:
        for position in fold.test:
            if not 0 <= position < n_design:
                raise AlignmentError(
                    f"el fold {fold.index} trae la posicion {position} y el diseno tiene "
                    f"{n_design} filas (A4)"
                )
        assignments.append(
            SplitAssignment(index=fold.index, train=tuple(fold.train), test=tuple(fold.test))
        )
    return tuple(assignments)


def _test_positions(split_plan: SplitPlan) -> tuple[int, ...]:
    """Las posiciones de *test* del plan, en orden y sin repetir (el plan no solapa folds)."""
    return tuple(sorted({position for fold in split_plan.folds for position in fold.test}))


def _probability_by_session(
    model: BaselineModel,
    frame: pl.DataFrame,
    *,
    positions: Sequence[int],
    sessions: Sequence[date],
) -> dict[date, float]:
    """La probabilidad **calibrada** de cada sesion de *test*, indexada por sesion (A4).

    Fuera de todo *test* ``calibrated_probabilities`` devuelve ``None`` y aqui es un error tipado:
    una sesion que el plan evalua sin probabilidad honesta no se puede decidir.
    """
    values = calibrated_probabilities(model, frame)
    out: dict[date, float] = {}
    for position in positions:
        value = values[position]
        if value is None:
            raise PipelineReportError(
                f"{sessions[position].isoformat()}: no hay probabilidad calibrada para una sesion "
                "que el plan evalua (el modelo solo predice dentro de un *test*, #24) (A4)"
            )
        out[sessions[position]] = float(value)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Escenario S1, tier re-derivado y regla declarada del brazo de coste declarado (A6)
# ─────────────────────────────────────────────────────────────────────────────
def scenario_parameters(*, cost_pct: Decimal) -> GateParameters:
    """Los once parametros de S1, **declarados** y con procedencia publicada.

    ``ev_threshold_pct`` no es un literal: es ``TARGET_MIN_COST_MULTIPLE`` (la constante de la
    regla 8, importada del gate) por el coste declarado de ida y vuelta de la tabla de #8, que es
    la lectura literal de «p. ej. > 2c». Los demas valores son los que declara la tabla de S1.
    """
    return GateParameters(
        broker=SCENARIO_BROKER,
        risk_per_trade_pct=Decimal("1"),
        ev_threshold_pct=TARGET_MIN_COST_MULTIPLE * cost_pct,
        max_daily_loss_pct=Decimal("2"),
        max_weekly_loss_pct=Decimal("5"),
        max_monthly_loss_pct=Decimal("10"),
        r_pct=Decimal("1"),
        tier_a_cost_multiple=Decimal("3"),
        tier_b_cost_multiple=Decimal("2"),
        tier_a_min_probability=Decimal("0.58"),
        authorized_tiers=(TIER_A,),
    )


def _favourable_probability(output: GateOutput) -> Decimal:
    """La probabilidad a favor de la direccion derivada: ``p`` en largo y ``1 - p`` en corto."""
    probability = output.prob_up_calibrated
    value = probability if output.direction is Direction.LONG else 1.0 - probability
    return Decimal(str(value))


def _declared_tier(output: GateOutput, params: GateParameters) -> str:
    """Re-deriva el tier A/B/C sobre ``ev_declared_pct`` con los multiplos y ``p`` de S1 (A6).

    Es la unica diferencia con el tier del gate: alli el multiplo se compara contra el **EV
    neto**, que con #64 es ``null`` y da siempre C; aqui, contra el EV **declarado**, que es la
    unica cifra que existe sin medir el *slippage*.
    """
    ev_declared = output.ev_declared_pct
    if ev_declared is None:
        return TIER_C
    multiple_a = cast("Decimal", params.tier_a_cost_multiple)
    multiple_b = cast("Decimal", params.tier_b_cost_multiple)
    minimum_a = cast("Decimal", params.tier_a_min_probability)
    if ev_declared > multiple_a * output.cost_pct and _favourable_probability(output) > minimum_a:
        return TIER_A
    if ev_declared > multiple_b * output.cost_pct:
        return TIER_B
    return TIER_C


def _notional_usd(
    *, capital_usd: Decimal, risk_per_trade_pct: Decimal, stop_pct: Decimal
) -> Decimal:
    """``capital x riesgo / stop``, con el mismo quantum declarado que el gate (A6, regla 2)."""
    with localcontext() as decimal_context:
        decimal_context.prec = MONEY_PRECISION
        return (capital_usd * risk_per_trade_pct / stop_pct).quantize(
            NOTIONAL_QUANTUM, rounding=ROUND_HALF_UP
        )


def _barrier_prices(
    *, direction: Direction, entry_px: float | None, stop_pct: Decimal, target_pct: Decimal | None
) -> tuple[float | None, float | None]:
    """Las dos barreras en precio, o ninguna: la **misma** conversion declarada del gate (#27).

    El gate no puede contestar una direccion en este brazo (devuelve ``NOTHING`` por las reglas 9
    y 10), asi que la geometria se reproduce aqui con el ``open`` de #64 como ``entry_px``:
    ``stop < open < target`` en largo y su espejo en corto, por construccion.
    """
    if entry_px is None or target_pct is None:
        return None, None
    stop_fraction = float(stop_pct) / 100.0
    target_fraction = float(target_pct) / 100.0
    if direction is Direction.LONG:
        return entry_px * (1.0 - stop_fraction), entry_px * (1.0 + target_fraction)
    return entry_px * (1.0 + stop_fraction), entry_px * (1.0 - target_fraction)


def _declared_cost_decision(
    *,
    output: GateOutput,
    params: GateParameters,
    capital_usd: Decimal,
    entry_px: float | None,
    ledger: ArmLedger,
) -> Decision:
    """La regla **literal** del brazo de coste declarado (A6), sesion a sesion.

    Se opera sii (i) ningun *blocker* del gate trae una regla de sesion (13, 14, 1, 3, 4, 5, 15,
    17, 18), (ii) ``ev_declared_pct > ev_threshold_pct``, (iii) ``target_pct >= 2 x
    c_declared_pct`` y (iv) el tier re-derivado sobre el EV declarado esta autorizado. Entonces la
    direccion sale de ``p_up_calibrated >= DECISION_THRESHOLD`` (constante **del gate**), el
    nocional de ``capital x riesgo / stop`` y los precios del ``open`` de la subasta (#64).
    """
    blocked = tuple(
        blocker["rule"] for blocker in output.blockers if blocker["rule"] in SESSION_RULES
    )
    if blocked:
        ledger.reject("gate_session_rule")
        return Decision(
            direction=Direction.NOTHING,
            reason=f"arm={ARM_COSTE_DECLARADO}; blocked_by_session_rules={','.join(blocked)}",
            probability=output.prob_up_calibrated,
        )
    threshold = cast("Decimal", params.ev_threshold_pct)
    ev_declared = output.ev_declared_pct
    if ev_declared is None or ev_declared <= threshold:
        ledger.reject("ev_declared_not_above_threshold")
        shown = "null" if ev_declared is None else _num(ev_declared)
        return Decision(
            direction=Direction.NOTHING,
            reason=(
                f"arm={ARM_COSTE_DECLARADO}; ev_declared_pct={shown} <= "
                f"ev_threshold_pct={_num(threshold)}"
            ),
            probability=output.prob_up_calibrated,
        )
    minimum_target = TARGET_MIN_COST_MULTIPLE * output.cost_pct
    if output.target_pct is None or output.target_pct < minimum_target:
        ledger.reject("target_below_cost_multiple")
        shown_target = "null" if output.target_pct is None else _num(output.target_pct)
        return Decision(
            direction=Direction.NOTHING,
            reason=(
                f"arm={ARM_COSTE_DECLARADO}; target_pct={shown_target} < {_num(minimum_target)}"
            ),
            probability=output.prob_up_calibrated,
        )
    tier = _declared_tier(output, params)
    authorized = cast("tuple[str, ...]", params.authorized_tiers)
    if tier not in authorized:
        ledger.reject("tier_not_authorized")
        return Decision(
            direction=Direction.NOTHING,
            reason=f"arm={ARM_COSTE_DECLARADO}; tier={tier} not in {authorized}",
            probability=output.prob_up_calibrated,
        )
    direction = (
        Direction.LONG if output.prob_up_calibrated >= GATE_DECISION_THRESHOLD else Direction.SHORT
    )
    notional = _notional_usd(
        capital_usd=capital_usd,
        risk_per_trade_pct=cast("Decimal", params.risk_per_trade_pct),
        stop_pct=output.stop_pct,
    )
    stop_px, target_px = _barrier_prices(
        direction=direction,
        entry_px=entry_px,
        stop_pct=output.stop_pct,
        target_pct=output.target_pct,
    )
    ledger.traded += 1
    return Decision(
        direction=direction,
        reason=(
            f"arm={ARM_COSTE_DECLARADO}; basis={BASIS_DECLARED_COST}; tier={tier}; "
            f"ev_declared_pct={_num(ev_declared)} > {_num(threshold)}"
        ),
        stop_px=stop_px,
        target_px=target_px,
        notional_usd=notional,
        probability=output.prob_up_calibrated,
    )


# ─────────────────────────────────────────────────────────────────────────────
# El gate por sesion y la corrida de los tres brazos
# ─────────────────────────────────────────────────────────────────────────────
def _gate_outputs(
    *,
    sessions: Sequence[date],
    probabilities: Mapping[date, float],
    instants: Mapping[date, datetime],
    moves: Mapping[date, Decimal],
    calendar: MarketCalendar,
    cost: CostBreakdown,
    params_oficial: GateParameters,
    params_escenario: GateParameters,
) -> tuple[dict[date, GateOutput], dict[date, GateOutput]]:
    """Evalua el gate **una vez** por sesion y por brazo de gate (A4-A6, A11).

    ``as_of`` es el instante declarado del almacen de esa misma sesion y ``today`` la propia
    sesion: la regla 13 (frescura) sale ``pass`` por construccion y se **publica** medida. Las
    sesiones sin ``garch_forecast`` no se inventan: no llegan aqui y el decididor las declara.
    """
    oficial: dict[date, GateOutput] = {}
    escenario: dict[date, GateOutput] = {}
    for session in sessions:
        move = moves[session]
        stop_pct = SCENARIO_STOP_SIGMA_MULTIPLE * move
        target_pct = SCENARIO_TARGET_STOP_MULTIPLE * stop_pct
        oficial[session] = _evaluate_one(
            session=session,
            instant=instants[session],
            calendar=calendar,
            probability=probabilities[session],
            move=move,
            stop_pct=stop_pct,
            target_pct=target_pct,
            cost=cost,
            params=params_oficial,
        )
        escenario[session] = _evaluate_one(
            session=session,
            instant=instants[session],
            calendar=calendar,
            probability=probabilities[session],
            move=move,
            stop_pct=stop_pct,
            target_pct=target_pct,
            cost=cost,
            params=params_escenario,
        )
    return oficial, escenario


def _evaluate_one(
    *,
    session: date,
    instant: datetime,
    calendar: MarketCalendar,
    probability: float,
    move: Decimal,
    stop_pct: Decimal,
    target_pct: Decimal,
    cost: CostBreakdown,
    params: GateParameters,
) -> GateOutput:
    """Una llamada al gate de #27 con las 19 entradas declaradas del mapeo temporal (A11)."""
    return evaluate_gate(
        session=session,
        as_of=instant,
        today=session,
        calendar=calendar,
        prob_up_calibrated=probability,
        expected_move_pct=move,
        expected_move_basis=EXPECTED_MOVE_BASIS,
        cost=cost,
        capital_usd=NOTIONAL_USD,
        snapshot_ok=True,
        stop_pct=stop_pct,
        target_pct=target_pct,
        fomc_dates=(),
        params=params,
        trades_today=0,
        daily_pnl_pct=None,
        weekly_pnl_pct=None,
        monthly_pnl_pct=None,
        observation_sessions_remaining=0,
    )


def _bundle(view: SessionView) -> SessionBundle:
    """La carga opaca de la vista: si no es la declarada, el error lo dice con su tipo."""
    context = view.context
    if not isinstance(context, SessionBundle):
        raise PipelineReportError(
            f"{view.session.isoformat()}: la funcion de decision espera un `SessionBundle` en "
            f"`context` y llego {type(context).__name__} (el motor no lee `context`)"
        )
    return context


def _nothing_for_missing_move(view: SessionView, ledger: ArmLedger) -> Decision:
    """Sesion sin ``garch_forecast``: no hay movimiento declarado y **no se inventa** (A11)."""
    ledger.without_expected_move += 1
    return Decision(
        direction=Direction.NOTHING,
        reason=(
            f"arm_missing_expected_move; session={view.session.isoformat()}: {GARCH_COLUMN} es "
            "null y S1 no declara un movimiento por defecto"
        ),
    )


def _deciders(
    *,
    name: str,
    params: GateParameters,
    ledger: ArmLedger,
    capital_usd: Decimal,
    n_folds: int,
) -> tuple[DecisionFn, ...]:
    """Una funcion de decision **por fold** para ese brazo, todas leyendo ``view.context`` (A4)."""

    def decide(view: SessionView) -> Decision:
        if view.context is None:
            return _nothing_for_missing_move(view, ledger)
        bundle = _bundle(view)
        if name == ARM_COSTE_DECLARADO:
            output = bundle.escenario
            ledger.observe(output)
            return _declared_cost_decision(
                output=output,
                params=params,
                capital_usd=capital_usd,
                entry_px=view.open_px,
                ledger=ledger,
            )
        output = bundle.oficial if name == ARM_OFICIAL else bundle.escenario
        ledger.observe(output)
        return to_engine_decision(output, entry_px=view.open_px)

    return tuple(decide for _ in range(n_folds))


def _run_arms(
    *,
    inputs: Sequence[SessionInput],
    split_plan: SplitPlan,
    cost_model: CostModel,
    slippage: SlippageParameter,
    outputs_oficial: Mapping[date, GateOutput],
    outputs_escenario: Mapping[date, GateOutput],
    params_oficial: GateParameters,
    params_escenario: GateParameters,
    capital_usd: Decimal,
) -> tuple[ArmRun, ...]:
    """Corre los tres brazos por el motor con sus deciders declarados (A4-A6)."""
    plan_by_name: tuple[tuple[str, Mapping[date, GateOutput], GateParameters], ...] = (
        (ARM_OFICIAL, outputs_oficial, params_oficial),
        (ARM_ESCENARIO, outputs_escenario, params_escenario),
        (ARM_COSTE_DECLARADO, outputs_escenario, params_escenario),
    )
    arms: list[ArmRun] = []
    for name, outputs, params in plan_by_name:
        ledger = ArmLedger()
        run = run_walk_forward(
            inputs,
            split_plan=split_plan,
            cost_model=cost_model,
            slippage=slippage,
            decide_by_fold=_deciders(
                name=name,
                params=params,
                ledger=ledger,
                capital_usd=capital_usd,
                n_folds=len(split_plan.folds),
            ),
            financing_cut=None,
        )
        arms.append(ArmRun(name=name, run=run, params=params, outputs=outputs, ledger=ledger))
    return tuple(arms)


def _with_context(
    inputs: Sequence[SessionInput],
    *,
    probabilities: Mapping[date, float],
    outputs_oficial: Mapping[date, GateOutput],
    outputs_escenario: Mapping[date, GateOutput],
) -> tuple[SessionInput, ...]:
    """Cuelga la carga opaca del decididor en la sesion que el plan evalua (A4).

    Una sesion sin salida del gate (sin ``garch_forecast``) viaja con ``context = None`` y el
    decididor la declara, en vez de romperse.
    """
    prepared: list[SessionInput] = []
    for item in inputs:
        session = item.session
        if session in outputs_oficial:
            prepared.append(
                replace(
                    item,
                    context=SessionBundle(
                        probability=probabilities[session],
                        oficial=outputs_oficial[session],
                        escenario=outputs_escenario[session],
                    ),
                )
            )
        else:
            prepared.append(replace(item, context=None))
    return tuple(prepared)


# ─────────────────────────────────────────────────────────────────────────────
# Series declaradas y metricas (A7, A8, A10, A13)
# ─────────────────────────────────────────────────────────────────────────────
def _sessions_of_run(run: BacktestRun) -> tuple[SessionOutcome, ...]:
    """Las sesiones de *test* de la corrida, en orden (folds y, dentro, sesion a sesion)."""
    return tuple(outcome for fold in run.folds for outcome in fold.sessions)


def _declared_return_pct(outcome: SessionOutcome) -> float:
    """El retorno de coste declarado, en **unidades coherentes** y dentro del informe (A10).

    ``100 x gross_pct - c_declared_pct``: ``gross_pct`` llega como **fraccion** (la unidad que
    declara #80) y ``c_declared_pct`` en **%**, asi que el producto por 100 los pone en la misma
    unidad. El informe **no** lee el P&L declarado del motor: lo re-deriva aqui, en %.
    """
    cost = outcome.cost
    if outcome.gross_pct is None or cost is None:
        raise PipelineReportError(
            f"{outcome.session.isoformat()}: una operacion sin `gross_pct` o sin `CostBreakdown` "
            "no tiene retorno declarado que publicar (A10)"
        )
    return 100.0 * outcome.gross_pct - float(cost.c_declared_pct)


def _series_of_run(run: BacktestRun) -> tuple[float, ...]:
    """La serie declarada de una corrida: cero en ``no_trade`` y fuera los ``skipped`` (A10).

    La convencion es la de #15 (``_risk_returns``): una sesion sin operacion **diluye** la serie
    con un cero exacto (no se opero, luego el retorno es 0 %) y una sesion saltada no entra.
    """
    values: list[float] = []
    for outcome in _sessions_of_run(run):
        if outcome.status == STATUS_SKIPPED:
            continue
        values.append(0.0 if outcome.status == STATUS_NO_TRADE else _declared_return_pct(outcome))
    return tuple(values)


def _declared_series_payload(series_pct: Sequence[float]) -> dict[str, object]:
    """El bloque publicado de la serie declarada de un brazo, con su digest autoconsistente (A10).

    ``units``/``n``/``series_pct`` son el cuerpo y ``series_sha256`` viaja al lado: es el sha256
    del ``canonical_text`` (#13) de ese cuerpo **sin** su propia clave, con el prefijo
    ``sha256:`` (nunca un digest desnudo). El digest se puede recomputar desde el JSON en disco,
    sin fijar ningun literal: un artefacto regenerable no se ancla a un dorado.

    La serie es **la misma** con la que el informe calcula las metricas del brazo: sale de
    ``_series_of_run`` (cero exacto en ``no_trade``, fuera las ``skipped``), no de una segunda
    derivacion.
    """
    body: dict[str, object] = {
        "units": SERIES_UNITS,
        "n": len(series_pct),
        "series_pct": [float(value) for value in series_pct],
    }
    plain = cast("Mapping[str, object]", _plain(body, where="declared_series"))
    digest = hashlib.sha256(canonical_text(plain).encode("utf-8")).hexdigest()
    return {**body, "series_sha256": f"{HASH_PREFIX}{digest}"}


def _traded_series_pct(run: BacktestRun) -> tuple[float, ...]:
    """La serie declarada de **solo** las sesiones operadas, una entrada por operacion (#92).

    Es la serie de la tasa de acierto **por operacion**: los `no_trade` no entran (no hubo
    operacion que contar) y las sesiones `skipped` tampoco. Se re-deriva con
    `_declared_return_pct`, como la serie de riesgo, y **no** se lee el P&L declarado del motor
    (A10).
    """
    return tuple(
        _declared_return_pct(outcome)
        for outcome in _sessions_of_run(run)
        if outcome.status == STATUS_TRADED
    )


def _exit_counts(run: BacktestRun) -> dict[str, int]:
    """Recuento por motivo de salida, en orden estable (los ceros medidos viajan)."""
    raw: dict[str, int] = {}
    for outcome in _sessions_of_run(run):
        if outcome.exit_reason is None:
            continue
        raw[outcome.exit_reason] = raw.get(outcome.exit_reason, 0) + 1
    return {reason: raw[reason] for reason in sorted(raw)}


def _close_to_close_pct(session: date, daily: Mapping[date, Mapping[str, object]]) -> float:
    """El retorno cierre a cierre de esa sesion, en %, leido del diario del almacen (A7)."""
    row = daily.get(session)
    if row is None or row.get("close") is None or row.get("prev_close") is None:
        raise PipelineReportError(
            f"{session.isoformat()}: el liston B/C exige cierre y cierre previo en el diario (A7)"
        )
    close = float(cast("float", row["close"]))
    previous = float(cast("float", row["prev_close"]))
    return 100.0 * (close / previous - 1.0)


def _decimal_series(series_pct: Sequence[float]) -> tuple[float, ...]:
    """La misma serie en la unidad decimal que esperan los helpers de #15 (``0,01`` = 1 %)."""
    return tuple(value / 100.0 for value in series_pct)


def _mean(series: Sequence[float]) -> float:
    """Media aritmetica de la serie (vacia ⇒ 0, para no dividir por cero)."""
    return math.fsum(series) / len(series) if series else 0.0


def _metric_block(
    *,
    estimate: float | None,
    lower: float | None,
    upper: float | None,
    basis: str,
    n: int,
    seed: int,
    reason: str | None = None,
) -> dict[str, object]:
    """Una metrica con su **etiqueta** completa (A8): sin ella no hay intervalo que publicar."""
    block: dict[str, object] = {
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "basis": basis,
        "n": n,
        "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
        "n_bootstrap": DEFAULT_BOOTSTRAP_SAMPLES,
        "seed": seed,
    }
    if reason is not None:
        block["reason"] = reason
    return block


def _measure(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    scale: float,
    basis: str,
    seed: int,
) -> dict[str, object]:
    """La metrica y su intervalo bootstrap con el helper exportado de #15 (A8)."""
    interval = bootstrap_confidence_interval(
        values,
        statistic,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
        n_bootstrap=DEFAULT_BOOTSTRAP_SAMPLES,
        seed=seed,
    )
    return _metric_block(
        estimate=interval.estimate * scale,
        lower=interval.lower * scale,
        upper=interval.upper * scale,
        basis=basis,
        n=len(values),
        seed=seed,
    )


def _beta(series: Sequence[float], benchmark: Sequence[float]) -> float:
    """La beta de la serie contra el benchmark **alineado sesion a sesion** (A13, formula de #15).

    Con varianza del benchmark exactamente 0 no hay beta que estimar: se publica ``nan`` y el
    bloque se declara sin valor, nunca una beta inventada (la misma guarda que #15).
    """
    mean_benchmark = _mean(benchmark)
    mean_series = _mean(series)
    variance = math.fsum((value - mean_benchmark) ** 2 for value in benchmark)
    if variance == 0.0 or not series:
        return math.nan
    covariance = math.fsum(
        (value - mean_series) * (reference - mean_benchmark)
        for value, reference in zip(series, benchmark, strict=True)
    )
    return covariance / variance


def _alpha(series: Sequence[float], benchmark: Sequence[float]) -> float:
    """El alfa de Jensen de la serie contra el benchmark, en decimales (A13)."""
    beta = _beta(series, benchmark)
    if not math.isfinite(beta):
        return math.nan
    return _mean(series) - beta * _mean(benchmark)


def _metric_statistic(name: str, benchmark: Sequence[float]) -> Callable[[Sequence[float]], float]:
    """El estadistico de cada metrica, sobre la serie **decimal** (la convencion de #15).

    Para ``beta`` y ``alpha_pct`` el cierre es sobre el benchmark fijo: la pareja sesion a sesion
    la fija el plan, nunca el remuestreo.
    """
    if name == "mean_return_pct":
        return _mean
    if name in ("hit_rate", "hit_rate_per_trade"):
        return lambda sample: sum(1 for value in sample if value > 0.0) / max(len(sample), 1)
    if name == "sharpe":
        return sharpe_ratio
    if name == "sortino":
        return sortino_ratio
    if name == "max_drawdown_pct":
        return lambda sample: drawdown_metrics(sample)[0]
    if name == "profit_factor":
        return lambda sample: profit_factor(sample) or 0.0
    if name == "benchmark_return_pct":
        return lambda sample: math.prod(1.0 + value for value in sample) - 1.0
    if name == "beta":
        return lambda sample: _beta(sample, benchmark)
    if name == "alpha_pct":
        return lambda sample: _alpha(sample, benchmark)
    return lambda sample: _mean(sample) - _mean(benchmark)


def _row_metrics(
    *,
    series_pct: Sequence[float],
    traded_series_pct: Sequence[float],
    benchmark_pct: Sequence[float],
    basis: str,
    cache: dict[tuple[str, tuple[float, ...]], dict[str, object]],
) -> dict[str, object]:
    """Las once metricas declaradas de una fila, cada una con su intervalo bootstrap (A8).

    El remuestreo es el helper exportado de #15. Para ``beta``, ``alpha_pct`` y
    ``excess_return_pct`` la serie remuestreada es la de la estrategia y el benchmark queda **fijo**
    en su valor observado: la pareja sesion a sesion la fija el plan, nunca el remuestreo, y el
    helper solo admite una serie (se declara en ``bootstrap_note``). ``excess_return_pct`` no gasta
    otro remuestreo: su intervalo es el de la media desplazado por la media del benchmark, que es
    una constante.

    ``hit_rate_per_trade`` (#92) es la excepcion: su serie es ``traded_series_pct`` (solo las
    sesiones operadas) y no la serie de riesgo, porque su denominacion es **por operacion**. Las
    dos tasas se publican con su ``denominator``, su ``denominator_note``, su ``n_wins`` y su
    ``wins_fraction``: ``estimate == n_wins / n``.
    """
    values = _decimal_series(series_pct)
    traded_values = _decimal_series(traded_series_pct)
    benchmark = _decimal_series(benchmark_pct)
    n = len(values)
    n_trades = len(traded_values)
    n_wins_session = sum(1 for value in values if value > 0.0)
    n_wins_trade = sum(1 for value in traded_values if value > 0.0)
    seeds = _seed_by_metric()
    scales: dict[str, float] = {
        "mean_return_pct": 100.0,
        "hit_rate": 1.0,
        "sharpe": 1.0,
        "sortino": 1.0,
        "max_drawdown_pct": 100.0,
        "profit_factor": 1.0,
        "benchmark_return_pct": 100.0,
        "excess_return_pct": 100.0,
        "beta": 1.0,
        "alpha_pct": 100.0,
        "hit_rate_per_trade": 1.0,
    }
    blocks: dict[str, object] = {}
    for name in METRIC_NAMES:
        if name in NO_INTERVAL_METRICS:
            blocks[name] = _metric_block(
                estimate=profit_factor(values),
                lower=None,
                upper=None,
                basis=basis,
                n=n,
                seed=seeds[name],
                reason=(
                    "el *profit factor* no se publica con intervalo: no todos los remuestreos "
                    "tienen valor (una muestra sin perdidas no tiene *profit factor*) y no se "
                    "sustituye por 0 (null != 0)"
                ),
            )
            continue
        if name == "excess_return_pct":
            continue
        if name == "hit_rate_per_trade":
            if not traded_values:
                blocks[name] = _metric_block(
                    estimate=None,
                    lower=None,
                    upper=None,
                    basis=basis,
                    n=0,
                    seed=seeds[name],
                    reason=(
                        "ninguna sesion opero: la tasa de acierto por operacion no existe y se "
                        "publica `null`, nunca `0`; sin operaciones no hay intervalo que estimar"
                    ),
                )
            else:
                key = (name, tuple(traded_values))
                if key not in cache:
                    cache[key] = _measure(
                        traded_values,
                        _metric_statistic(name, benchmark),
                        scale=scales[name],
                        basis=basis,
                        seed=seeds[name],
                    )
                blocks[name] = dict(cache[key])
            continue
        source = benchmark if name == "benchmark_return_pct" else values
        key = (name, tuple(source))
        if key not in cache:
            cache[key] = _measure(
                source,
                _metric_statistic(name, benchmark),
                scale=scales[name],
                basis=basis,
                seed=seeds[name],
            )
        blocks[name] = dict(cache[key])
    _add_excess_metric(
        blocks,
        mean_block=cast("dict[str, object]", blocks["mean_return_pct"]),
        benchmark=benchmark,
        n=n,
        basis=basis,
        seed=seeds["excess_return_pct"],
    )
    blocks["hit_rate"] = _annotate_rate(
        cast("dict[str, object]", blocks["hit_rate"]),
        denominator=DENOMINATOR_SESSION,
        note=(
            "denominacion **por sesion**: entran todas las sesiones de *test* que no estan "
            "`skipped`; una sesion `no_trade` entra como 0 exacto (no se opero, el retorno es "
            "0 %) y diluye la tasa. Las sesiones `skipped` **no** entran. La `p_win` de "
            "plan.md §11.6 es por operacion: la tasa comparable con ella es `hit_rate_per_trade` "
            "(#92)"
        ),
        n_wins=n_wins_session,
        n_trades=n_trades,
    )
    blocks["hit_rate_per_trade"] = _annotate_rate(
        cast("dict[str, object]", blocks["hit_rate_per_trade"]),
        denominator=DENOMINATOR_TRADE,
        note=(
            "denominacion **por operacion**: entran solo las sesiones `traded`, una por operacion "
            "(los `no_trade` entran como 0 exacto en la tasa por sesion de al lado, pero aqui no "
            "cuentan; las `skipped` no entran en ninguna de las dos). Es la `p_win` de plan.md "
            "§11.6; la `hit_rate` por sesion diluye con los `no_trade` y no es comparable (#92)"
        ),
        n_wins=n_wins_trade,
        n_trades=n_trades,
    )
    blocks["metric_names"] = list(METRIC_NAMES)
    blocks["bootstrap_note"] = (
        "cada intervalo usa `bootstrap_confidence_interval` (#15) con "
        f"`n_bootstrap = {DEFAULT_BOOTSTRAP_SAMPLES}`, `confidence_level = "
        f"{DEFAULT_CONFIDENCE_LEVEL}` y una semilla derivada de la declarada "
        f"({DEFAULT_BOOTSTRAP_SEED}) por la posicion de la metrica, acotada a [0, 2**32 - 1]. La "
        "serie remuestreada es la de la fila; para `beta`, `alpha_pct` y `excess_return_pct` el "
        "benchmark queda fijo en su valor observado, porque la pareja sesion a sesion la fija el "
        "plan"
    )
    return blocks


def _annotate_rate(
    block: dict[str, object],
    *,
    denominator: str,
    note: str,
    n_wins: int,
    n_trades: int,
) -> dict[str, object]:
    """Declara la denominacion de una tasa de acierto y su recuento de aciertos (#92).

    ``n`` es el tamano de la serie sobre la que se estima (sesiones de la serie de riesgo o
    operaciones) y ``wins_fraction`` su fraccion exacta, de modo que ``estimate == n_wins / n``.
    ``n_trades`` viaja en las dos tasas para poder comprobar la denominacion sin reconstruir la
    corrida.
    """
    n = cast("int", block["n"])
    block["denominator"] = denominator
    block["denominator_note"] = note
    block["n_wins"] = n_wins
    block["n_trades"] = n_trades
    block["wins_fraction"] = _fraction_text(n_wins, n)
    return block


def _add_excess_metric(
    blocks: dict[str, object],
    *,
    mean_block: Mapping[str, object],
    benchmark: Sequence[float],
    n: int,
    basis: str,
    seed: int,
) -> None:
    """El retorno en exceso: la media declarada menos la media del benchmark, con su intervalo."""
    estimate = mean_block["estimate"]
    if estimate is None:
        blocks["excess_return_pct"] = _metric_block(
            estimate=None,
            lower=None,
            upper=None,
            basis=basis,
            n=n,
            seed=seed,
            reason="la serie no tiene ninguna sesion: no hay metrica que estimar",
        )
        return
    shift = _mean(benchmark) * 100.0
    blocks["excess_return_pct"] = _metric_block(
        estimate=cast("float", estimate) - shift,
        lower=cast("float", mean_block["lower"]) - shift,
        upper=cast("float", mean_block["upper"]) - shift,
        basis=basis,
        n=n,
        seed=seed,
        reason=(
            "el intervalo es el de `mean_return_pct` desplazado por la media del benchmark, que es "
            "una constante: no se gasta otro remuestreo"
        ),
    )


def _table_rows(
    *,
    baselines: Sequence[BaselineOutcome],
    sessions: Sequence[date],
    daily: Mapping[date, Mapping[str, object]],
    carry_pct: float,
) -> tuple[TableRow, ...]:
    """Las nueve filas de la tabla unica: seis baselines y los tres listones A/B/C (A7).

    El liston A **es** la fila ``always_long`` (mismo ``run_sha256`` y misma serie: es el mismo
    motor con el mismo sesgo), el B es la referencia declarada cierre a cierre menos la financiacion
    por noche de la tabla de costes y el C es el indice puro, que **no** es invertible y por eso no
    cuenta entre los seis baselines.
    """
    rows: list[TableRow] = []
    for outcome in baselines:
        run = outcome.run
        rows.append(
            TableRow(
                row=outcome.baseline,
                kind="baseline",
                is_invertible=True,
                provenance=(
                    "baseline de #14 vía `run_all_baselines` (#69) con el nocional plano "
                    "declarado; retorno derivado aquí con `100 x gross_pct - c_declared_pct` (A10)"
                ),
                series_pct=_series_of_run(run),
                traded_series_pct=_traded_series_pct(run),
                n_test=run.traded + run.no_trade + run.skipped,
                traded=run.traded,
                no_trade=run.no_trade,
                skipped=run.skipped,
                run_sha256=run.run_sha256,
                exit_reason_counts=_exit_counts(run),
                frequency=(
                    None
                    if outcome.frequency is None
                    else _fraction_text(outcome.frequency.numerator, outcome.frequency.denominator)
                ),
                seed=outcome.seed,
            )
        )
    always_long = next(row for row in rows if row.row == "always_long")
    close_to_close = tuple(_close_to_close_pct(session, daily) for session in sessions)
    rows.insert(
        len(baselines),
        TableRow(
            row="liston_a",
            kind="liston_a",
            is_invertible=True,
            provenance=(
                "la fila `always_long` (open->close) de esta misma tabla: mismo motor, misma serie "
                "y mismo `run_sha256`; el baseline obligatorio de plan.md §11.2"
            ),
            series_pct=always_long.series_pct,
            traded_series_pct=always_long.traded_series_pct,
            n_test=always_long.n_test,
            traded=always_long.traded,
            no_trade=always_long.no_trade,
            skipped=always_long.skipped,
            run_sha256=always_long.run_sha256,
            exit_reason_counts=always_long.exit_reason_counts,
            frequency=always_long.frequency,
            seed=always_long.seed,
        ),
    )
    rows.append(
        TableRow(
            row="liston_b",
            kind="liston_b",
            is_invertible=True,
            provenance=(
                "referencia declarada: cierre a cierre de `^GSPC` en % menos la tenencia en largo "
                "por noche **importada** de la tabla de costes (#8); el diferencial de entrada y "
                "salida no se cobra aquí (el liston B de primera clase, con posiciones overnight "
                "por el motor, es #70) y el instante de corte de la financiación es #87"
            ),
            series_pct=tuple(value - carry_pct for value in close_to_close),
            traded_series_pct=tuple(value - carry_pct for value in close_to_close),
            n_test=len(sessions),
            traded=len(sessions),
            no_trade=0,
            skipped=0,
            run_sha256=None,
            exit_reason_counts={},
            frequency=None,
            seed=None,
        )
    )
    rows.append(
        TableRow(
            row="liston_c",
            kind="liston_c",
            is_invertible=False,
            provenance=(
                "referencia, **no** un baseline: el índice puro `^GSPC` cierre a cierre en %, que "
                "nadie puede comprar (plan.md §11.2); viaja como `reference` y no cuenta entre los "
                "seis baselines"
            ),
            series_pct=close_to_close,
            traded_series_pct=close_to_close,
            n_test=len(sessions),
            traded=len(sessions),
            no_trade=0,
            skipped=0,
            run_sha256=None,
            exit_reason_counts={},
            frequency=None,
            seed=None,
        )
    )
    return tuple(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Payload
# ─────────────────────────────────────────────────────────────────────────────
def _counts_by(values: Collection[object] | Sequence[object]) -> dict[str, int]:
    """Recuento por clave, en orden alfabetico: el hash no depende del orden de un ``dict``."""
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _universe_payload(universe: Universe, features: FeatureFrame) -> dict[str, object]:
    """El universo, contado y con su identidad (A4)."""
    return {
        "series_id": universe.series_id,
        "sessions": len(universe.inputs),
        "labelled_sessions": universe.labelled_sessions,
        "first_session": None
        if universe.first_session is None
        else universe.first_session.isoformat(),
        "last_session": None
        if universe.last_session is None
        else universe.last_session.isoformat(),
        "clean_from": None if universe.clean_from is None else universe.clean_from.isoformat(),
        "n_design_rows": features.n_design_rows,
        "n_positives": features.n_positives,
        "positive_share": features.n_positives / features.n_design_rows,
        "n_nulls_in_features": features.n_nulls_in_features,
        "excluded": len(universe.excluded),
        "exclusion_counts": _counts_by([item["reason"] for item in universe.excluded]),
        "identity": (
            "el universo del motor y el frame de diseno son la misma secuencia de sesiones, "
            f"comprobada sesión a sesión: {len(universe.inputs)} = {features.n_design_rows} (A4)"
        ),
        "context": (
            "`SessionView.context` lleva un `SessionBundle` (probabilidad calibrada y las dos "
            "salidas del gate): el motor **no lee** `context`, que es la carga opaca del decididor"
        ),
    }


def _plan_payload(split_plan: SplitPlan) -> dict[str, object]:
    """El plan de #12, con su eco (A4)."""
    return {
        "plan_sha256": split_plan.plan_sha256,
        "n_sessions": split_plan.n_sessions,
        "n_folds": len(split_plan.folds),
        "n_test": sum(len(fold.test) for fold in split_plan.folds),
        "not_in_any_test": len(split_plan.uncovered),
        "purge_total": split_plan.purge_total,
        "embargo_total": split_plan.embargo_total,
        "embargo_in_train_total": split_plan.embargo_in_train_total,
        "exclusions_are_no_op": split_plan.exclusions_are_no_op,
        "n_splits": PHASE1_PLAN.n_splits,
        "test_size": PHASE1_PLAN.test_size,
        "embargo_sessions": PHASE1_PLAN.embargo_sessions,
        "max_train_size": PHASE1_PLAN.max_train_size,
        "folds": [
            {
                "index": fold.index,
                "test_start": fold.test_start,
                "test_stop": fold.test_stop,
                "n_train": len(fold.train),
                "n_test": len(fold.test),
            }
            for fold in split_plan.folds
        ],
    }


def _model_payload(
    model: BaselineModel,
    *,
    features: FeatureFrame,
    probabilities: Mapping[date, float],
    moves: Mapping[date, Decimal],
    test_sessions: Sequence[date],
) -> dict[str, object]:
    """El modelo ajustado y la cobertura del escenario, sin re-serializar los folds enteros."""
    digest = hashlib.sha256(canonical_text(model.to_payload()).encode("utf-8")).hexdigest()
    methods = [fold.calibration.method for fold in model.folds]
    probability_text = "|".join(
        f"{session.isoformat()}:{probabilities[session]!r}"
        for session in test_sessions
        if session in probabilities
    )
    return {
        "features": list(BASELINE_FEATURES),
        "seed": model.seed,
        "design_lag_sessions": DESIGN_LAG_SESSIONS,
        "decision_threshold": GATE_DECISION_THRESHOLD,
        "n_folds": len(model.folds),
        "calibration_method_counts": method_counts(methods),
        "folds": [
            {
                "index": fold.index,
                "n_train": fold.n_train,
                "n_test": fold.n_test,
                "n_iter": fold.n_iter,
                "converged": fold.converged,
                "train_base_rate": fold.train_base_rate,
                "calibration_method": fold.calibration.method,
                "n_calibration": fold.calibration.n_calibration,
            }
            for fold in model.folds
        ],
        "model_sha256": f"{HASH_PREFIX}{digest}",
        "probabilities": {
            "n": len(probabilities),
            "n_test_with_probability": sum(
                1 for session in test_sessions if session in probabilities
            ),
            "digest": f"{HASH_PREFIX}"
            f"{hashlib.sha256(probability_text.encode('utf-8')).hexdigest()}",
            "note": (
                "la probabilidad que decide es la **calibrada** de #25; `digest` fija su valor "
                "sesión a sesión para poder comparar dos corridas sin publicar 500 numeros"
            ),
        },
        "expected_move": {
            "basis": EXPECTED_MOVE_BASIS,
            "column": GARCH_COLUMN,
            "n_test_with_move": sum(1 for session in test_sessions if session in moves),
            "n_test_without_move": sum(1 for session in test_sessions if session not in moves),
            "note": (
                "un `garch_forecast` nulo **no** se sustituye por 0: la sesión se queda sin "
                "movimiento declarado y el decididor la declara sin operar (A5, A11)"
            ),
        },
        "matrix_sha256": features.matrix.matrix_sha256,
    }


def _scenario_payload(params: GateParameters, *, cost: CostBreakdown) -> dict[str, object]:
    """Los once parámetros de S1 con su procedencia y las cifras derivadas (A6)."""
    declared: dict[str, object] = {}
    for name, provenance in PROVENANCE.items():
        value = getattr(params, name)
        declared[name] = {
            "value": None if value is None else _plain(value, where=f"scenario.{name}"),
            "provenance": provenance,
        }
    return {
        "id": SCENARIO_ID,
        "declared_fields": sorted(params.model_fields_set),
        "n_declared_fields": len(params.model_fields_set),
        "parameters": declared,
        "ev_threshold_derivation": (
            f"ev_threshold_pct = TARGET_MIN_COST_MULTIPLE ({_num(TARGET_MIN_COST_MULTIPLE)}) x "
            f"c_declared_pct ({_num(cost.c_declared_pct)}) = "
            f"{_num(TARGET_MIN_COST_MULTIPLE * cost.c_declared_pct)}"
        ),
        "cost_basis_pct": _num(cost.c_declared_pct),
        "cost_provenance": (
            "tabla declarada de #8 vía `declared_cost_model()` y `cost_breakdown` con `nights = 0` "
            "(intradia puro, regla 6) y el nocional plano declarado: el % no depende del nocional"
        ),
        "expected_move_basis": EXPECTED_MOVE_BASIS,
        "stop_target_rule": (
            f"stop_pct = {_num(SCENARIO_STOP_SIGMA_MULTIPLE)} x sigma y target_pct = "
            f"{_num(SCENARIO_TARGET_STOP_MULTIPLE)} x stop_pct (reglas 7 y 8 de §12)"
        ),
    }


def _rule_outcomes(outputs: Mapping[date, GateOutput]) -> dict[str, dict[str, int]]:
    """Resultado de las 18 reglas, agregado: ``{regla: {resultado: recuento}}`` (A11)."""
    aggregated: dict[str, dict[str, int]] = {}
    for output in outputs.values():
        for entry in output.rules:
            per_rule = aggregated.setdefault(entry["rule"], {})
            outcome = entry["outcome"]
            per_rule[outcome] = per_rule.get(outcome, 0) + 1
    return {
        rule: dict(sorted(counts.items()))
        for rule, counts in sorted(aggregated.items(), key=lambda item: int(item[0]))
    }


def _arm_payload(
    arm: ArmRun,
    *,
    metrics: Mapping[str, object],
    mismatches: int,
    declared_series_zero: bool,
    n_test_without_move: int,
) -> dict[str, object]:
    """Un brazo, con la identidad que exige A4 y las reglas del gate medidas (A5, A6, A12)."""
    trades = [outcome for outcome in _sessions_of_run(arm.run) if outcome.status == STATUS_TRADED]
    declared_cost_sum = math.fsum(
        float(cast("CostBreakdown", outcome.cost).c_declared_pct) for outcome in trades
    )
    return {
        "name": arm.name,
        "n_test": arm.n_test,
        "traded": arm.run.traded,
        "no_trade": arm.run.no_trade,
        "skipped": arm.run.skipped,
        "basis": BASIS_DECLARED_COST,
        "is_validation": False,
        "run_sha256": arm.run.run_sha256,
        "plan_sha256": arm.run.plan_sha256,
        "declared_return_series_all_zero": declared_series_zero,
        "declared_series": _declared_series_payload(_series_of_run(arm.run)),
        "metrics": dict(metrics),
        "gate": {
            "params_declared": sorted(arm.params.model_fields_set),
            "n_params_declared": len(arm.params.model_fields_set),
            "status_counts": dict(sorted(arm.ledger.status_counts.items())),
            "blocker_code_counts": dict(sorted(arm.ledger.blocker_code_counts.items())),
            "rule_outcomes": _rule_outcomes(arm.outputs),
            "undecided": sorted(
                {
                    entry["parameter"]
                    for output in arm.outputs.values()
                    for entry in output.undecided
                }
            ),
            "undecided_issues": list(UNDECIDED_ISSUES),
        },
        "exact_aggregates": {
            "n_test": arm.n_test,
            "traded": arm.run.traded,
            "declared_cost_sum_pct": declared_cost_sum,
            "n_test_without_expected_move": n_test_without_move,
            "note": (
                "cifras exactas acumuladas de la tabla de costes y del plan: recuentos y sumas, no "
                "estimaciones con intervalo"
            ),
        },
        "declared_rule": (
            "se opera sii (i) ningun blocker del gate trae una regla de sesion "
            f"({', '.join(SESSION_RULES)}), (ii) ev_declared_pct > ev_threshold_pct, "
            f"(iii) target_pct >= {_num(TARGET_MIN_COST_MULTIPLE)} x c_declared_pct y (iv) el tier "
            "re-derivado A/B/C sobre ev_declared_pct con los multiplos y la probabilidad de S1 "
            "esta en authorized_tiers; direccion p_up_calibrated >= DECISION_THRESHOLD (constante "
            "del gate), notional = capital_usd x risk_per_trade_pct / stop_pct y precios desde "
            "entry_px = open (#64)"
            if arm.name == ARM_COSTE_DECLARADO
            else (
                "el gate decide tal cual: direccion, tier y, si opera, nocional y barreras; "
                "`to_engine_decision` convierte los % en precios con el open de #64"
            )
        ),
        "rejections": dict(sorted(arm.ledger.rejections.items())),
        "mismatches_with_other_arms": mismatches,
        "mismatch_note": (
            "sesiones de *test* en que la direccion de este brazo difiere de la de los otros dos "
            "brazos del gate, medida sobre los informes del motor (A6)"
        ),
    }


def _first_reason(run: BacktestRun) -> str | None:
    """El ``reason`` de la primera sesion de la corrida (viaja en el informe del motor)."""
    sessions = _sessions_of_run(run)
    return None if not sessions else sessions[0].reason


def _null_arms_payload(
    *,
    arms: Sequence[ArmRun],
    baselines: Sequence[BaselineOutcome],
    zero_series: Mapping[str, bool],
) -> dict[str, object]:
    """Los dos brazos del gate, medidos: cero operaciones y su hash junto al de ``no_trade``.

    Es A5: la identidad con ``no_trade`` no se afirma, se mide con el ``run_sha256`` y el
    ``reason`` que viaja dentro del informe del motor.
    """
    by_name = {arm.name: arm for arm in arms}
    no_trade = next(outcome.run for outcome in baselines if outcome.baseline == NO_TRADE)
    reasons: dict[str, str | None] = {NO_TRADE: _first_reason(no_trade)}
    for name in (ARM_OFICIAL, ARM_ESCENARIO):
        reasons[name] = _first_reason(by_name[name].run)
    return {
        "arms": [ARM_OFICIAL, ARM_ESCENARIO],
        "traded": {name: by_name[name].run.traded for name in (ARM_OFICIAL, ARM_ESCENARIO)},
        "all_no_trade": all(
            by_name[name].run.no_trade == by_name[name].n_test
            for name in (ARM_OFICIAL, ARM_ESCENARIO)
        ),
        "declared_return_series_all_zero": {
            name: zero_series[name] for name in (ARM_OFICIAL, ARM_ESCENARIO)
        },
        "run_sha256": {
            NO_TRADE: no_trade.run_sha256,
            ARM_OFICIAL: by_name[ARM_OFICIAL].run.run_sha256,
            ARM_ESCENARIO: by_name[ARM_ESCENARIO].run.run_sha256,
        },
        "equals_no_trade": {
            ARM_OFICIAL: by_name[ARM_OFICIAL].run.run_sha256 == no_trade.run_sha256,
            ARM_ESCENARIO: by_name[ARM_ESCENARIO].run.run_sha256 == no_trade.run_sha256,
        },
        "status_counts": {
            name: dict(sorted(by_name[name].ledger.status_counts.items()))
            for name in (ARM_OFICIAL, ARM_ESCENARIO)
        },
        "blocker_code_counts": dict(
            sorted(by_name[ARM_ESCENARIO].ledger.blocker_code_counts.items())
        ),
        "decision_reason": reasons,
        "measured_on": "run_sha256 del informe canonico del motor (el `reason` viaja dentro)",
        "why": (
            "los tres hashes no coinciden con el de `no_trade` y el motivo se mide, no se afirma: "
            "el `reason` de la decision viaja en el informe del motor y los tres son distintos "
            f"(`no_trade`: {reasons[NO_TRADE]!r}; `{ARM_OFICIAL}`: {reasons[ARM_OFICIAL]!r}; "
            f"`{ARM_ESCENARIO}`: {reasons[ARM_ESCENARIO]!r})"
        ),
        "note": (
            "el brazo oficial devuelve `no_recommendation_undecided` en las `n_test` sesiones: sin "
            "los once parametros declarados el gate no emite juicio, y eso **no** es `NOTHING`"
        ),
    }


def _traded_indicator(run: BacktestRun) -> tuple[float, ...]:
    """La serie 1/0 de sesiones operadas, para medir la cuota de la regla 11 con intervalo (A12)."""
    return tuple(
        1.0 if outcome.status == STATUS_TRADED else 0.0 for outcome in _sessions_of_run(run)
    )


def _rule_11_payload(
    arms: Sequence[ArmRun], *, cache: dict[tuple[str, tuple[float, ...]], dict[str, object]]
) -> dict[str, object]:
    """La regla 11 **medida**: cuota de sesiones operadas por brazo, con banda y veredicto (A12)."""
    seeds = _seed_by_metric()
    out: dict[str, object] = {}
    for arm in arms:
        n_test = arm.n_test
        share = arm.run.traded / n_test if n_test else 0.0
        if share > RULE_11_BAND[1]:
            verdict = "above"
        elif share < RULE_11_BAND[0]:
            verdict = "below"
        else:
            verdict = "inside"
        indicator = _traded_indicator(arm.run)
        key = ("rule_11_share", indicator)
        if indicator and key not in cache:
            cache[key] = _measure(
                indicator, _mean, scale=1.0, basis=BASIS_DECLARED_COST, seed=seeds["hit_rate"]
            )
        out[arm.name] = {
            "traded": arm.run.traded,
            "n_test": n_test,
            "share_fraction": _fraction_text(arm.run.traded, n_test),
            "share": share,
            "share_pct": share * 100.0,
            "share_interval": dict(cache[key]) if indicator else None,
            "declared_band": [RULE_11_BAND[0], RULE_11_BAND[1]],
            "verdict": verdict,
            "measured": True,
        }
    return {
        "rule": RULE_11,
        "statement": "regla 11 de §12: el sistema debe operar como maximo el 10-30 % de los dias",
        "band": [RULE_11_BAND[0], RULE_11_BAND[1]],
        "arms": out,
        "note": (
            "la cuota se **mide** sesión a sesión sobre las `n_test` del plan; quedarse por debajo "
            "de la banda no incumple la regla (que es un techo), es un resultado"
        ),
    }


def _row_payload(row: TableRow, *, metrics: Mapping[str, object]) -> dict[str, object]:
    """Una fila de la tabla con su identidad, sus metricas y sus cifras exactas (A7, A8)."""
    return {
        "row": row.row,
        "kind": row.kind,
        "is_invertible": row.is_invertible,
        "provenance": row.provenance,
        "basis": BASIS_DECLARED_COST,
        "is_validation": False,
        "n_test": row.n_test,
        "traded": row.traded,
        "no_trade": row.no_trade,
        "skipped": row.skipped,
        "run_sha256": row.run_sha256,
        "exit_reason_counts": dict(row.exit_reason_counts),
        "frequency": row.frequency,
        "seed": row.seed,
        "series_units": SERIES_UNITS,
        "n_series": len(row.series_pct),
        "sum_return_pct": math.fsum(row.series_pct),
        "n_non_zero_returns": sum(1 for value in row.series_pct if value != 0.0),
        "rotation": {
            "traded": row.traded,
            "n_test": row.n_test,
            "share": (row.traded / row.n_test) if row.n_test else 0.0,
            "note": "recuento exacto, sin intervalo: no es una estimacion de una distribucion",
        },
        "metrics": dict(metrics),
    }


def _estimate(metrics: Mapping[str, object], name: str) -> float | None:
    """La estimacion de esa metrica, o ``None`` si no esta o no tiene valor."""
    block = metrics.get(name)
    if not isinstance(block, Mapping):
        return None
    value = cast("Mapping[str, object]", block).get("estimate")
    return None if value is None else cast("float", value)


def _comparison_payload(
    *,
    rows: Sequence[TableRow],
    metrics_by_name: Mapping[str, Mapping[str, object]],
    benchmark: tuple[float, ...],
) -> dict[str, object]:
    """El brazo declarado contra las nueve filas y la frase que dice quien carga el resultado (A13).

    La atribucion se **mide**: se compara el retorno medio que aporta el benchmark (``beta x media
    del benchmark``) con el alfa de Jensen de la misma fila y se publica cual de los dos pesa mas.
    La frase no afirma nada que no salga de esos numeros.

    **Convencion de unidades (una sola, A13):** ``benchmark`` ya llega en **%**
    (``_close_to_close_pct`` = ``100 x (close / prev - 1)``), ``beta`` es adimensional y
    ``alpha_pct`` sale de ``_row_metrics`` ya en %. Todo lo que este bloque publica va en %, asi
    que **no se vuelve a multiplicar por 100**: hacerlo era el error de unidades 100x de A13, que
    dejaba ``benchmark_mean_pct`` en 100x la media por sesion y rompia la identidad de Jensen.
    """
    deltas: dict[str, object] = {}
    declared_metrics = metrics_by_name.get(ARM_COSTE_DECLARADO, {})
    declared_mean = _estimate(declared_metrics, "mean_return_pct")
    declared_sharpe = _estimate(declared_metrics, "sharpe")
    for row in rows:
        metrics = metrics_by_name.get(row.row, {})
        mean = _estimate(metrics, "mean_return_pct")
        sharpe = _estimate(metrics, "sharpe")
        deltas[row.row] = {
            "delta_mean_return_pct": (
                None if declared_mean is None or mean is None else declared_mean - mean
            ),
            "delta_sharpe": (
                None if declared_sharpe is None or sharpe is None else declared_sharpe - sharpe
            ),
            "note": "diferencia del brazo de coste declarado menos la fila (medida, no afirmada)",
        }
    benchmark_mean_pct = _mean(benchmark)
    beta = _estimate(declared_metrics, "beta")
    alpha = _estimate(declared_metrics, "alpha_pct")
    benchmark_contribution_pct = None if beta is None else beta * benchmark_mean_pct
    if benchmark_contribution_pct is None or alpha is None:
        loader = "no_atribuible"
        statement = (
            "no hay atribucion: falta la beta o el alfa del brazo de coste declarado (benchmark de "
            "varianza cero o serie vacia)"
        )
    else:
        loader = "beta" if abs(benchmark_contribution_pct) > abs(alpha) else "alpha"
        statement = (
            f"el resultado del brazo de coste declarado lo carga el **{loader}**: aportacion "
            f"del benchmark = beta x media del benchmark = {benchmark_contribution_pct:.6f} % "
            f"frente a alfa de Jensen = {alpha:.6f} % (separados contra `{SERIES_ID}`)"
        )
    return {
        "deltas": deltas,
        "attribution": {
            "loader": loader,
            "benchmark_contribution_pct": benchmark_contribution_pct,
            "alpha_pct": alpha,
            "beta": beta,
            "benchmark_mean_pct": benchmark_mean_pct,
            "statement": statement,
            "series": "series de riesgo alineadas sesion a sesion contra el indice de referencia",
        },
    }


def _payload(
    *,
    as_of: datetime,
    universe: Universe,
    features: FeatureFrame,
    model: BaselineModel,
    split_plan: SplitPlan,
    arms: Sequence[ArmRun],
    baselines: Sequence[BaselineOutcome],
    rows: Sequence[TableRow],
    metrics_by_name: Mapping[str, Mapping[str, object]],
    benchmark: tuple[float, ...],
    mismatches: Mapping[str, int],
    zero_series: Mapping[str, bool],
    cost: CostBreakdown,
    params_escenario: GateParameters,
    runs: Mapping[str, BacktestRun],
    probabilities: Mapping[date, float],
    moves: Mapping[date, Decimal],
    test_sessions: Sequence[date],
    cache: dict[tuple[str, tuple[float, ...]], dict[str, object]],
) -> dict[str, object]:
    """El payload canonico: tipos JSON puros y determinista (A2, A3)."""
    n_test_without_move = sum(1 for session in test_sessions if session not in moves)
    raw: dict[str, object] = {
        "analysis": ANALYSIS,
        "task": TASK,
        "generated_at": as_of.isoformat(),
        "hash_format": REPORT_HASH_FORMAT,
        "basis": BASIS_DECLARED_COST,
        "is_validation": False,
        "series_id": SERIES_ID,
        "renamed_series": {"proxy_of": PRICE_PROXY_OF, "issue": "#50"},
        "universe": _universe_payload(universe, features),
        "plan": _plan_payload(split_plan),
        "model": _model_payload(
            model,
            features=features,
            probabilities=probabilities,
            moves=moves,
            test_sessions=test_sessions,
        ),
        "temporal_mapping": dict(TEMPORAL_MAPPING),
        "scenario": _scenario_payload(params_escenario, cost=cost),
        "arms": {
            arm.name: _arm_payload(
                arm,
                metrics=metrics_by_name.get(arm.name, {}),
                mismatches=mismatches.get(arm.name, 0),
                declared_series_zero=zero_series.get(arm.name, True),
                n_test_without_move=n_test_without_move,
            )
            for arm in arms
        },
        "null_arms": _null_arms_payload(arms=arms, baselines=baselines, zero_series=zero_series),
        "rule_11": _rule_11_payload(arms, cache=cache),
        "table": {
            "rows": [_row_payload(row, metrics=metrics_by_name.get(row.row, {})) for row in rows],
            "kinds": ["baseline", "liston_a", "liston_b", "liston_c"],
            "note": (
                "los seis baselines de #14 corren por `run_all_baselines` (#69) y los tres "
                "listones son la referencia declarada de plan.md §11.2: A **es** la fila "
                "`always_long`, B es la serie cierre a cierre menos la financiacion por noche "
                "importada de la tabla de costes y C es el indice puro, que no es invertible y "
                "**no** cuenta entre los seis baselines. Cada fila lleva sus metricas con "
                "intervalo bootstrap en su propio bloque `metrics`"
            ),
        },
        "arm_comparison": _comparison_payload(
            rows=rows, metrics_by_name=metrics_by_name, benchmark=benchmark
        ),
        "net_metrics": {
            "state": "not_computable",
            "reason": NET_METRICS_REASON,
            "where": "cfdtrader.backtest.metrics.calculate_metrics",
            "follow_ups": ["#62", "#60"],
        },
        "limits": {
            "phase0_gate": "fail",
            "phase0_gate_source": "#9/#64 (heredado: no se re-evalua aqui)",
            "verdict": "not_evaluated",
            "verdict_issue": "#29",
            "basis": BASIS_DECLARED_COST,
            "is_validation": False,
            "slippage_state": cost.slippage.state.value,
            "slippage_is_measurement": cost.slippage.is_measurement,
            "financing_cut": None,
            "financing_cut_issue": "#87",
            "llm_overlay": "disabled",
            "scheduler": "none",
        },
        "limitations": [
            "el motor es puro y no lee `context`: el decididor recibe la carga opaca por "
            "sesion y no ve altos, bajos ni cierres de la sesion en curso",
            "el *slippage* sigue **supuesto** (#64): ninguna cifra de este informe incluye un "
            "termino medido de ejecucion",
            "el liston B de primera clase (posiciones overnight por el motor) es #70 y el instante "
            "de corte de la financiacion es #87",
            "las reglas de sesion 1, 3, 4, 5, 15, 17 y 18 se evaluan con los valores "
            "declarados del mapeo (sin cartera, sin calendario de FOMC y sin modo observacion): "
            "su efecto se publica, no se asume",
        ],
        "does_not_do": [dict(item) for item in DOES_NOT_DO],
        "follow_ups": [dict(item) for item in FOLLOW_UPS],
        "engine_runs": {
            name: {
                "run_sha256": run.run_sha256,
                "traded": run.traded,
                "no_trade": run.no_trade,
                "skipped": run.skipped,
                "not_in_any_test": run.not_in_any_test,
            }
            for name, run in sorted(runs.items())
        },
    }
    return cast("dict[str, object]", _plain(raw, where="payload"))


# ─────────────────────────────────────────────────────────────────────────────
# Informe en prosa (A2)
# ─────────────────────────────────────────────────────────────────────────────
def _metric_cell(metrics: Mapping[str, object], name: str) -> str:
    """La celda de la tabla: la estimacion, o ``null`` cuando no hay valor (nunca un 0 falso)."""
    value = _estimate(metrics, name)
    return "null" if value is None else f"{value:.6f}"


def render_markdown(report: PipelineReport) -> str:
    """El informe en Markdown, determinista y sin cifras que no esten en el payload (A2)."""
    payload = report.payload
    arms = cast("dict[str, dict[str, object]]", payload["arms"])
    null_arms = cast("dict[str, object]", payload["null_arms"])
    rule_11 = cast("dict[str, object]", payload["rule_11"])
    table = cast("dict[str, object]", payload["table"])
    rows = cast("list[dict[str, object]]", table["rows"])
    net = cast("dict[str, object]", payload["net_metrics"])
    scenario = cast("dict[str, object]", payload["scenario"])
    plan = cast("dict[str, object]", payload["plan"])
    universe = cast("dict[str, object]", payload["universe"])
    comparison = cast("dict[str, object]", payload["arm_comparison"])
    attribution = cast("dict[str, object]", comparison["attribution"])
    declared_metrics = cast("dict[str, object]", arms[ARM_COSTE_DECLARADO]["metrics"])
    session_rate = cast("dict[str, object]", declared_metrics["hit_rate"])
    trade_rate = cast("dict[str, object]", declared_metrics["hit_rate_per_trade"])

    lines: list[str] = [
        f"# Pipeline completo sobre el motor *walk-forward* - `{universe['series_id']}`",
        "",
        f"Generado el `{payload['generated_at']}` (**declarado**, nunca leido del reloj). "
        f"`report_sha256 = {report.report_sha256}`.",
        "",
        f"**No es una validacion.** `basis = {payload['basis']}`, "
        f"`is_validation = {payload['is_validation']}`: la tabla es de coste declarado y la puerta "
        "de Fase 0 sigue en `fail` (#9/#64). El veredicto de Fase 2 es #29.",
        "",
        "## Universo y plan",
        "",
        f"- Sesiones: **{universe['sessions']}** (`{universe['first_session']}` -> "
        f"`{universe['last_session']}`), {universe['n_positives']} positivas "
        f"({cast('float', universe['positive_share']) * 100:.3f} %).",
        f"- Plan oficial: **{plan['n_test']}** sesiones de *test* en {plan['n_folds']} folds, "
        f"{plan['not_in_any_test']} fuera de todo *test*; `plan_sha256 = {plan['plan_sha256']}`.",
        "",
        "## Mapeo temporal declarado",
        "",
    ]
    for key, value in TEMPORAL_MAPPING.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            f"## Escenario {scenario['id']}",
            "",
            f"- Campos declarados: {scenario['n_declared_fields']} "
            f"({', '.join(cast('list[str]', scenario['declared_fields']))}).",
            f"- {scenario['ev_threshold_derivation']}.",
            f"- {scenario['stop_target_rule']}.",
            "",
            "## Los tres brazos",
            "",
            "| brazo | n_test | operadas | no_trade | skipped | basis | is_validation | cuota "
            "(regla 11) | veredicto | `run_sha256` |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    rule_11_arms = cast("dict[str, dict[str, object]]", rule_11["arms"])
    for name in ARM_NAMES:
        arm = arms[name]
        share = rule_11_arms[name]
        lines.append(
            f"| `{name}` | {arm['n_test']} | {arm['traded']} | {arm['no_trade']} | "
            f"{arm['skipped']} | `{arm['basis']}` | {arm['is_validation']} | "
            f"{share['share_fraction']} ({cast('float', share['share_pct']):.3f} %) | "
            f"`{share['verdict']}` | `{arm['run_sha256']}` |"
        )
    lines.extend(
        [
            "",
            f"- Los dos brazos del gate operan **{null_arms['traded']}** sesiones; serie declarada "
            f"toda a cero: {null_arms['declared_return_series_all_zero']}.",
            f"- `run_sha256` de los tres informes: {null_arms['run_sha256']}; coinciden: "
            f"{null_arms['equals_no_trade']}. {null_arms['why']}",
            f"- Recuento por `code` de bloqueo del brazo de escenario: "
            f"{null_arms['blocker_code_counts']}.",
            "",
            f"- Tasa de acierto de `{ARM_COSTE_DECLARADO}`, con su denominacion declarada: "
            f"`hit_rate` **por sesion** (`{session_rate['wins_fraction']}`, n = "
            f"{session_rate['n']}) frente a `hit_rate_per_trade` **por operacion** "
            f"(`{trade_rate['wins_fraction']}`, n = {trade_rate['n']}): los `no_trade` "
            "diluyen la primera y no entran en la segunda, y las sesiones `skipped` no entran "
            "en ninguna de las dos (#92).",
            "",
            "## Atribucion: alpha contra beta",
            "",
            f"- {attribution['statement']}",
            "",
            "## Tabla unica (seis baselines y los tres listones)",
            "",
            "| fila | kind | invertible | n_test | operadas | media % | Sharpe | Sortino | dd max "
            "% | profit factor | beta | alpha % |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        metrics = cast("dict[str, object]", row["metrics"])
        lines.append(
            f"| `{row['row']}` | `{row['kind']}` | {row['is_invertible']} | {row['n_test']} | "
            f"{row['traded']} | {_metric_cell(metrics, 'mean_return_pct')} | "
            f"{_metric_cell(metrics, 'sharpe')} | {_metric_cell(metrics, 'sortino')} | "
            f"{_metric_cell(metrics, 'max_drawdown_pct')} | "
            f"{_metric_cell(metrics, 'profit_factor')} | {_metric_cell(metrics, 'beta')} | "
            f"{_metric_cell(metrics, 'alpha_pct')} |"
        )
    lines.extend(
        [
            "",
            f"- {table['note']}",
            "",
            "## Metricas netas",
            "",
            f"- `{net['state']}`: {net['reason']}",
            f"- Seguimientos: {', '.join(cast('list[str]', net['follow_ups']))}.",
            "",
            "## Limitaciones y seguimientos",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in cast("list[str]", payload["limitations"]))
    lines.append("")
    lines.extend(
        f"- **{item['issue']}** - `{item['id']}`: {item['statement']}"
        for item in cast("list[dict[str, str]]", payload["does_not_do"])
    )
    lines.append("")
    lines.extend(
        f"- **{item['issue']}** - {item['topic']}: {item['why']}"
        for item in cast("list[dict[str, str]]", payload["follow_ups"])
    )
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Ejecucion (A1, A2)
# ─────────────────────────────────────────────────────────────────────────────
def _mismatches(arms: Sequence[ArmRun]) -> dict[str, int]:
    """Sesiones en que la direccion de cada brazo difiere de la de los otros dos (A6)."""
    directions: dict[str, dict[date, str]] = {
        arm.name: {
            outcome.session: (
                "nothing" if outcome.decision is None else outcome.decision.direction.value
            )
            for outcome in _sessions_of_run(arm.run)
        }
        for arm in arms
    }
    out: dict[str, int] = {}
    for arm in arms:
        others = [name for name in ARM_NAMES if name != arm.name]
        out[arm.name] = sum(
            1
            for session, direction in directions[arm.name].items()
            if any(directions[other].get(session) != direction for other in others)
        )
    return out


def _build_rows(
    *,
    arms: Sequence[ArmRun],
    rows: Sequence[TableRow],
    benchmark: tuple[float, ...],
    cache: dict[tuple[str, tuple[float, ...]], dict[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, bool]]:
    """Las metricas de las nueve filas y de los tres brazos, con la cache de series compartida."""
    metrics_by_name: dict[str, dict[str, object]] = {}
    zero_series: dict[str, bool] = {}
    for row in rows:
        metrics_by_name[row.row] = _row_metrics(
            series_pct=row.series_pct,
            traded_series_pct=row.traded_series_pct,
            benchmark_pct=benchmark,
            basis=BASIS_DECLARED_COST,
            cache=cache,
        )
        zero_series[row.row] = all(value == 0.0 for value in row.series_pct)
    for arm in arms:
        series = _series_of_run(arm.run)
        metrics_by_name[arm.name] = _row_metrics(
            series_pct=series,
            traded_series_pct=_traded_series_pct(arm.run),
            benchmark_pct=benchmark,
            basis=BASIS_DECLARED_COST,
            cache=cache,
        )
        zero_series[arm.name] = all(value == 0.0 for value in series)
    return metrics_by_name, zero_series


def analyse(
    *, store: Store, reports_dir: Path, as_of: datetime, write: bool = True
) -> PipelineReport:
    """Cablea el pipeline, corre los tres brazos y (por defecto) escribe el informe (A1, A2).

    ``as_of`` es **obligatorio** y es el unico instante de la corrida: ninguna ruta consulta el
    reloj. ``write = False`` no escribe **nada**.
    """
    moment = _as_utc(as_of)
    history = load_history(store)
    calendar = load_calendar(years=_calendar_years(history.daily))
    universe = build_inputs(history, calendar=calendar)
    split_plan = build_split_plan(universe.inputs)
    features = build_feature_frame(store, series_id=SERIES_ID)
    _require_alignment(universe, features)

    cost_model = declared_cost_model()
    slippage = declared_slippage_assumption()
    cost = cost_breakdown(
        model=cost_model, slippage=slippage, notional_usd=NOTIONAL_USD, side=Side.LONG, nights=0
    )
    params_oficial = GateParameters()
    params_escenario = scenario_parameters(cost_pct=cost.c_declared_pct)

    sessions = tuple(item.session for item in universe.inputs)
    positions = _test_positions(split_plan)
    test_sessions = tuple(sessions[position] for position in positions)
    fitted = fit_baseline(
        features.design,
        splits=_split_assignments(split_plan, n_design=features.n_design_rows),
        label_horizon=label_horizon_sequence(n_sessions=features.n_design_rows),
    )
    probabilities = _probability_by_session(
        fitted, features.design.frame, positions=positions, sessions=sessions
    )
    moves = _expected_move_pct(features.matrix.frame)
    instants = _session_instants(history.daily)
    gated_sessions = tuple(session for session in test_sessions if session in moves)
    outputs_oficial, outputs_escenario = _gate_outputs(
        sessions=gated_sessions,
        probabilities=probabilities,
        instants=instants,
        moves=moves,
        calendar=calendar,
        cost=cost,
        params_oficial=params_oficial,
        params_escenario=params_escenario,
    )
    arms = _run_arms(
        inputs=_with_context(
            universe.inputs,
            probabilities=probabilities,
            outputs_oficial=outputs_oficial,
            outputs_escenario=outputs_escenario,
        ),
        split_plan=split_plan,
        cost_model=cost_model,
        slippage=slippage,
        outputs_oficial=outputs_oficial,
        outputs_escenario=outputs_escenario,
        params_oficial=params_oficial,
        params_escenario=params_escenario,
        capital_usd=NOTIONAL_USD,
    )
    baselines = run_all_baselines(
        universe.inputs,
        split_plan=split_plan,
        cost_model=cost_model,
        slippage=slippage,
        notional_usd=NOTIONAL_USD,
        frequency=RANDOM_MATCHED_FREQUENCY,
        seed=RANDOM_MATCHED_SEED,
    )
    runs: dict[str, BacktestRun] = {
        **{outcome.baseline: outcome.run for outcome in baselines},
        **{arm.name: arm.run for arm in arms},
    }
    daily = _daily_by_session(history.daily)
    carry_pct = float(cost_model.carry_long_pct_per_night)
    benchmark = tuple(_close_to_close_pct(session, daily) for session in test_sessions)
    rows = _table_rows(
        baselines=baselines, sessions=test_sessions, daily=daily, carry_pct=carry_pct
    )
    cache: dict[tuple[str, tuple[float, ...]], dict[str, object]] = {}
    metrics_by_name, zero_series = _build_rows(
        arms=arms, rows=rows, benchmark=benchmark, cache=cache
    )
    payload = _payload(
        as_of=moment,
        universe=universe,
        features=features,
        model=fitted,
        split_plan=split_plan,
        arms=arms,
        baselines=baselines,
        rows=rows,
        metrics_by_name=metrics_by_name,
        benchmark=benchmark,
        mismatches=_mismatches(arms),
        zero_series=zero_series,
        cost=cost,
        params_escenario=params_escenario,
        runs=runs,
        probabilities=probabilities,
        moves=moves,
        test_sessions=test_sessions,
        cache=cache,
    )
    report = PipelineReport(
        as_of=moment,
        report_date=moment.astimezone(UTC).date(),
        payload=payload,
        report_sha256=_digest(payload),
        reports_dir=reports_dir,
        history=history,
        universe=universe,
        split_plan=split_plan,
        features=features,
        model=fitted,
        cost_model=cost_model,
        slippage=slippage,
        cost=cost,
        arms=arms,
        baselines=baselines,
        rows=rows,
        test_sessions=test_sessions,
    )
    if write:
        json_path, markdown_path = report.write(reports_dir)
        logger.info("informe del pipeline de #28: {} y {}", json_path, markdown_path)
    return report


def _parse_as_of(value: str | None) -> datetime:
    """El instante declarado de la CLI: **obligatorio**, y nunca tomado del reloj (A1, A2)."""
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
    """Punto de entrada del informe del pipeline completo.

    Codigos de salida: ``0`` = informe escrito (aunque el veredicto siga pendiente, que es un
    resultado legitimo y declarado); ``2`` = falta o no es valido ``--as-of``, falta un dataset o
    la muestra no alcanza: **no se escribe nada** y el motivo sale por ``stderr``.
    """
    parser = argparse.ArgumentParser(
        prog="cfdtrader.analysis.pipeline_report",
        description="Pipeline completo (features -> probabilidad -> gate -> sizing) en el motor",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="raiz del almacen")
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
        help="instante declarado ISO-8601, obligatorio para escribir (el modulo no lee el reloj)",
    )
    args = parser.parse_args(argv)

    try:
        moment = _parse_as_of(cast("str | None", args.as_of))
        settings = load_settings(args.settings)
    except (PipelineReportError, ConfigurationError) as error:
        print(f"no se puede emitir el informe del pipeline: {error}", file=sys.stderr)
        return 2

    data_root = Path(args.data_root) if args.data_root is not None else Path(settings.data.root)
    reports_dir = (
        Path(args.reports_dir)
        if args.reports_dir is not None
        else data_root / "derived" / "reports"
    )
    try:
        report = analyse(store=Store(data_root), reports_dir=reports_dir, as_of=moment, write=True)
    except (PipelineReportError, BacktestReportError) as error:
        print(f"no se puede emitir el informe del pipeline: {error}", file=sys.stderr)
        return 2

    table = cast("dict[str, object]", report.payload["table"])
    logger.info(
        "pipeline de #28: {} filas de tabla y {} brazos; report_sha256 = {}",
        len(cast("list[object]", table["rows"])),
        len(report.arms),
        report.report_sha256,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
