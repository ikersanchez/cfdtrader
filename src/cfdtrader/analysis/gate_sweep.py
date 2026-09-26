"""Barrido de sensibilidad del gate sobre el brazo de coste declarado de #28 (#86).

Mide la **sensibilidad** del brazo de coste declarado de ``pipeline_report`` a los once
parametros declarados del gate, sobre una rejilla **declarada y acotada** de diez celdas, y
publica por celda el recuento de operaciones y el retorno de coste declarado con su intervalo
bootstrap. La disciplina es la de #28: ``basis = declared_cost``, ``is_validation = false``,
hashes reproducibles y **ninguna metrica de base medida** (el *slippage* medido es #88, que
depende de #62).

La rejilla no varia los once campos a ciegas: ocho de ellos **no mueven** la serie declarada
(los motivos estan medidos y se publican en ``inert_parameters``), asi que el barrido varia los
que si la mueven y anade dos **controles de invariancia** (``inv_risk2`` e ``inv_thr32``) que lo
demuestran con el mismo ``series_sha256`` que la celda de referencia ``s1``.

Como se **reutiliza** #28 en vez de re-derivarlo:

- las entradas compartidas (historia, calendario, universo, plan, features, modelo, probabilidad
  calibrada, movimientos e instantes) se calculan **una sola vez** y la rejilla las itera;
- el gate por sesion sale de ``_gate_outputs`` de #28, el contexto opaco de ``_with_context``,
  los decididores de ``_deciders`` y la serie declarada de ``_series_of_run``: **se importan, no
  se copian**;
- la metrica con intervalo es ``mean_return_pct`` con la **misma** configuracion de #28 (semilla
  43, ``n_bootstrap`` 10000 y la serie en decimales), asi que la celda ``s1`` reproduce
  ``arms.coste_declarado`` del informe de #28.

Sin reloj (``--as-of`` es obligatorio), sin red, sin escrituras fuera de ``--reports-dir`` y
determinista byte a byte: ``report_sha256`` es el sha256 del texto canonico de #13 sobre el
payload sin la clave del hash, con el prefijo ``sha256:``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final, cast

from loguru import logger

from cfdtrader.analysis.backtest_report import (
    NOTIONAL_USD,
    SERIES_ID,
    History,
    Universe,
    _calendar_years,  # pyright: ignore[reportPrivateUsage]
    build_inputs,
    build_split_plan,
    label_horizon_sequence,
    load_history,
)
from cfdtrader.analysis.feature_frame import FeatureFrame, build_feature_frame
from cfdtrader.analysis.pipeline_report import (
    ARM_COSTE_DECLARADO,
    BASIS_DECLARED_COST,
    HASH_PREFIX,
    SCENARIO_BROKER,
    SERIES_UNITS,
    ArmLedger,
    PipelineReportError,
    _as_utc,  # pyright: ignore[reportPrivateUsage]
    _deciders,  # pyright: ignore[reportPrivateUsage]
    _decimal_series,  # pyright: ignore[reportPrivateUsage]
    _declared_series_payload,  # pyright: ignore[reportPrivateUsage]
    _expected_move_pct,  # pyright: ignore[reportPrivateUsage]
    _gate_outputs,  # pyright: ignore[reportPrivateUsage]
    _mean,  # pyright: ignore[reportPrivateUsage]
    _measure,  # pyright: ignore[reportPrivateUsage]
    _metric_block,  # pyright: ignore[reportPrivateUsage]
    _num,  # pyright: ignore[reportPrivateUsage]
    _parse_as_of,  # pyright: ignore[reportPrivateUsage]
    _plain,  # pyright: ignore[reportPrivateUsage]
    _probability_by_session,  # pyright: ignore[reportPrivateUsage]
    _require_alignment,  # pyright: ignore[reportPrivateUsage]
    _seed_by_metric,  # pyright: ignore[reportPrivateUsage]
    _series_of_run,  # pyright: ignore[reportPrivateUsage]
    _session_instants,  # pyright: ignore[reportPrivateUsage]
    _sessions_of_run,  # pyright: ignore[reportPrivateUsage]
    _split_assignments,  # pyright: ignore[reportPrivateUsage]
    _test_positions,  # pyright: ignore[reportPrivateUsage]
    _with_context,  # pyright: ignore[reportPrivateUsage]
    scenario_parameters,
)
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
    STATUS_TRADED,
    BacktestRun,
    canonical_text,
    run_walk_forward,
)
from cfdtrader.backtest.splits import SplitPlan
from cfdtrader.data.calendar import MarketCalendar, load_calendar
from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.store import Store
from cfdtrader.decision.gate import GateParameters
from cfdtrader.models.baseline import fit_baseline

__all__ = [
    "ANALYSIS",
    "FIXED_PARAMETERS",
    "GATE_PARAMETER_NAMES",
    "GRID_CELLS",
    "HASH_PREFIX",
    "INERT_PARAMETERS",
    "INVARIANCE_CONTROLS",
    "REFERENCE_CELL",
    "REPORT_PREFIX",
    "TASK",
    "CellResult",
    "GateSweepError",
    "GateSweepReport",
    "GridCell",
    "SharedInputs",
    "analyse",
    "build_shared_inputs",
    "cell_parameters",
    "main",
    "render_markdown",
    "sweep_cell",
]

#: Identidad del informe: quien lo emite y que tarea lo pide.
ANALYSIS: Final[str] = "cfdtrader.analysis.gate_sweep"
TASK: Final[str] = "#86"
REPORT_PREFIX: Final[str] = "gate_sweep"
HASH_FORMAT: Final[str] = (
    "sha256:<64 hex> del texto canonico (``canonical_text`` de #13) del payload **sin** la clave "
    "``report_sha256``. El prefijo viaja dentro del valor: un digest desnudo lo bloquea "
    "``detect-secrets``"
)

#: Los once campos declarados del gate, en el orden de la tabla de la issue.
GATE_PARAMETER_NAMES: Final[tuple[str, ...]] = (
    "broker",
    "risk_per_trade_pct",
    "ev_threshold_pct",
    "max_daily_loss_pct",
    "max_weekly_loss_pct",
    "max_monthly_loss_pct",
    "r_pct",
    "tier_a_cost_multiple",
    "tier_b_cost_multiple",
    "tier_a_min_probability",
    "authorized_tiers",
)

#: La unica metrica con intervalo que publica cada celda: el techo de tiempo manda (una sola).
METRIC_NAME: Final[str] = "mean_return_pct"
#: Cuantizacion declarada de la metrica: la serie viaja en % y el helper de #15 va en decimales.
METRIC_SCALE: Final[float] = 100.0
#: Semilla de esa metrica: la **misma** de #28 (`DEFAULT_BOOTSTRAP_SEED + posicion + 1` = 43).
METRIC_SEED: Final[int] = _seed_by_metric()[METRIC_NAME]

#: Los cinco campos **fijos** de la rejilla: no se barren, se declaran una vez.
FIXED_PARAMETERS: Final[dict[str, str]] = {
    "broker": SCENARIO_BROKER,
    "max_daily_loss_pct": "2",
    "max_weekly_loss_pct": "5",
    "max_monthly_loss_pct": "10",
    "r_pct": "1",
}

#: La celda de referencia: reproduce ``arms.coste_declarado`` del informe de #28.
REFERENCE_CELL: Final[str] = "s1"
#: Los dos controles de invariancia: misma serie declarada que la referencia.
INVARIANCE_CONTROLS: Final[tuple[str, ...]] = ("inv_risk2", "inv_thr32")

#: Motivo declarado de un intervalo que **no existe**: nunca se sustituye por 0.
ZERO_TRADES_REASON: Final[str] = (
    "ninguna sesion de la celda opero: sin operaciones no hay retorno de coste declarado que "
    "estimar, asi que la estimacion, el limite inferior y el superior se publican `null` "
    "(nunca `0`) y no hay intervalo bootstrap que calcular"
)

#: Los ocho campos que **no mueven** la serie declarada de la base S1, con su motivo medido.
INERT_PARAMETERS: Final[tuple[dict[str, str], ...]] = (
    {
        "field": "broker",
        "reason": (
            "la regla declarada del brazo no lo lee: es un centinela de la decision abierta 4 "
            "(#59) y se publica tal cual en toda la rejilla"
        ),
    },
    {
        "field": "ev_threshold_pct",
        "reason": (
            "`ev_declared_pct / c_declared_pct` va de 62.846 a 539.5 en las 500 sesiones, luego "
            "2 x c y 32 x c se quedan siempre por debajo del minimo: cero rechazos por umbral de "
            "EV en la rejilla (el control `inv_thr32` lo comprueba)"
        ),
    },
    {
        "field": "max_daily_loss_pct",
        "reason": (
            "el pipeline llama al gate con `daily_pnl_pct = None`: la regla 3 no tiene cifra de "
            "cartera que comparar y no puede bloquear (la contabilidad es #83)"
        ),
    },
    {
        "field": "max_weekly_loss_pct",
        "reason": (
            "el pipeline llama al gate con `weekly_pnl_pct = None`: la regla 4 no tiene cifra de "
            "cartera que comparar y no puede bloquear (la contabilidad es #83)"
        ),
    },
    {
        "field": "max_monthly_loss_pct",
        "reason": (
            "el pipeline llama al gate con `monthly_pnl_pct = None`: la regla 5 no tiene cifra de "
            "cartera que comparar y no puede bloquear (la contabilidad es #83)"
        ),
    },
    {
        "field": "r_pct",
        "reason": (
            "declara el supuesto de *slippage* de la decision abierta 5 (#60); la base declarada "
            "no lo usa: `gross_pct` sale del motor y `c_declared_pct` de la tabla de #8"
        ),
    },
    {
        "field": "risk_per_trade_pct",
        "reason": (
            "solo dimensiona el nocional (`capital x riesgo / stop`) y `gross_pct = exit/entry - "
            "1` es independiente del tamano: el control `inv_risk2` lo demuestra con el mismo "
            "`series_sha256` que `s1`"
        ),
    },
    {
        "field": "tier_b_cost_multiple",
        "reason": (
            "en la base declarada S1 (`authorized_tiers = (A,)`) el tier B no se autoriza nunca y "
            "el multiplo no llega a compararse; solo interviene cuando el eje "
            "`authorized_tiers` autoriza B (celda `tiers_ab_multb100`)"
        ),
    },
)

#: El texto que declara que la inercia se mide **sobre la base declarada**, no en abstracto.
INERTNESS_SCOPE: Final[str] = (
    "estos ocho campos no mueven la serie declarada de la base S1; los dos controles de "
    "invariancia (`inv_risk2` e `inv_thr32`) lo miden con el mismo `series_sha256` que `s1`. "
    "Los ejes que si la mueven en la rejilla son `tier_a_min_probability`, `authorized_tiers` y "
    "los multiplos **por encima del suelo** medido (62.846 x c)"
)


class GateSweepError(PipelineReportError):
    """Error tipado del barrido: una celda pedida que no existe, o un payload no publicable."""


# ─────────────────────────────────────────────────────────────────────────────
# La rejilla declarada (A2, A12)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class GridCell:
    """Una celda declarada de la rejilla: los ejes que se barren y los que se dejan fijos.

    ``ev_threshold_multiple`` es el unico eje **relativo**: el umbral de EV se declara como
    multiplo del coste declarado de ida y vuelta (``2*c`` / ``32*c``), porque el valor absoluto
    depende de una tabla de costes regenerable y el literal estable es el multiplo.
    """

    cell_id: str
    tier_a_min_probability: Decimal
    authorized_tiers: tuple[str, ...]
    tier_a_cost_multiple: Decimal
    tier_b_cost_multiple: Decimal
    ev_threshold_multiple: Decimal
    risk_per_trade_pct: Decimal


#: La rejilla **declarada** de #86, en el orden de la tabla de la issue (10 celdas).
GRID_CELLS: Final[tuple[GridCell, ...]] = (
    GridCell("s1", Decimal("0.58"), ("A",), Decimal("3"), Decimal("2"), Decimal("2"), Decimal("1")),
    GridCell(
        "pmin_050", Decimal("0.50"), ("A",), Decimal("3"), Decimal("2"), Decimal("2"), Decimal("1")
    ),
    GridCell(
        "pmin_055", Decimal("0.55"), ("A",), Decimal("3"), Decimal("2"), Decimal("2"), Decimal("1")
    ),
    GridCell(
        "pmin_062", Decimal("0.62"), ("A",), Decimal("3"), Decimal("2"), Decimal("2"), Decimal("1")
    ),
    GridCell(
        "pmin_070", Decimal("0.70"), ("A",), Decimal("3"), Decimal("2"), Decimal("2"), Decimal("1")
    ),
    GridCell(
        "tiers_ab",
        Decimal("0.58"),
        ("A", "B"),
        Decimal("3"),
        Decimal("2"),
        Decimal("2"),
        Decimal("1"),
    ),
    GridCell(
        "tiers_ab_multb100",
        Decimal("0.58"),
        ("A", "B"),
        Decimal("3"),
        Decimal("100"),
        Decimal("2"),
        Decimal("1"),
    ),
    GridCell(
        "mult_a_100",
        Decimal("0.58"),
        ("A",),
        Decimal("100"),
        Decimal("2"),
        Decimal("2"),
        Decimal("1"),
    ),
    GridCell(
        "inv_risk2", Decimal("0.58"), ("A",), Decimal("3"), Decimal("2"), Decimal("2"), Decimal("2")
    ),
    GridCell(
        "inv_thr32",
        Decimal("0.58"),
        ("A",),
        Decimal("3"),
        Decimal("2"),
        Decimal("32"),
        Decimal("1"),
    ),
)


def _multiple_text(multiple: Decimal) -> str:
    """El multiplo del coste como texto estable: ``2*c``, ``32*c`` (nunca el valor absoluto)."""
    return f"{_num(multiple)}*c"


def _distinct(values: Sequence[object]) -> list[object]:
    """Valores distintos en orden de primera aparicion: un ``set`` no tiene orden publicado."""
    out: list[object] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _grid_axes() -> dict[str, list[object]]:
    """Los ejes de la rejilla, derivados de las celdas: el orden es el de la tabla."""
    return {
        "tier_a_min_probability": _distinct(
            [_num(cell.tier_a_min_probability) for cell in GRID_CELLS]
        ),
        "authorized_tiers": _distinct([list(cell.authorized_tiers) for cell in GRID_CELLS]),
        "tier_a_cost_multiple": _distinct([_num(cell.tier_a_cost_multiple) for cell in GRID_CELLS]),
        "tier_b_cost_multiple": _distinct([_num(cell.tier_b_cost_multiple) for cell in GRID_CELLS]),
        "ev_threshold_pct": _distinct(
            [_multiple_text(cell.ev_threshold_multiple) for cell in GRID_CELLS]
        ),
        "risk_per_trade_pct": _distinct([_num(cell.risk_per_trade_pct) for cell in GRID_CELLS]),
    }


def _grid_cell_payload(cell: GridCell) -> dict[str, object]:
    """La celda **verbatim**: ``cell_id`` y los once campos del gate, sin cifras derivadas.

    ``ev_threshold_pct`` viaja como multiplo del coste declarado (``2*c``): es el literal estable
    de la tabla de la issue y no depende de una tabla de costes regenerable.
    """
    return {
        "cell_id": cell.cell_id,
        "broker": FIXED_PARAMETERS["broker"],
        "risk_per_trade_pct": _num(cell.risk_per_trade_pct),
        "ev_threshold_pct": _multiple_text(cell.ev_threshold_multiple),
        "max_daily_loss_pct": FIXED_PARAMETERS["max_daily_loss_pct"],
        "max_weekly_loss_pct": FIXED_PARAMETERS["max_weekly_loss_pct"],
        "max_monthly_loss_pct": FIXED_PARAMETERS["max_monthly_loss_pct"],
        "r_pct": FIXED_PARAMETERS["r_pct"],
        "tier_a_cost_multiple": _num(cell.tier_a_cost_multiple),
        "tier_b_cost_multiple": _num(cell.tier_b_cost_multiple),
        "tier_a_min_probability": _num(cell.tier_a_min_probability),
        "authorized_tiers": list(cell.authorized_tiers),
    }


def cell_parameters(cell: GridCell, *, cost: CostBreakdown) -> GateParameters:
    """Los once parametros **resueltos** de una celda, como deltas sobre la base declarada S1.

    La base es ``scenario_parameters`` de #28 (los once campos de S1), asi que la celda ``s1`` es
    **identica** al escenario publicado por #28 y las demas solo cambian los ejes barridos. El
    unico eje **relativo** es ``ev_threshold_pct``: se pasa a valor absoluto con el coste
    declarado de ida y vuelta de la tabla de #8 (``2*c`` / ``32*c``).
    """
    base = scenario_parameters(cost_pct=cost.c_declared_pct)
    return base.model_copy(
        update={
            "risk_per_trade_pct": cell.risk_per_trade_pct,
            "ev_threshold_pct": cell.ev_threshold_multiple * cost.c_declared_pct,
            "tier_a_cost_multiple": cell.tier_a_cost_multiple,
            "tier_b_cost_multiple": cell.tier_b_cost_multiple,
            "tier_a_min_probability": cell.tier_a_min_probability,
            "authorized_tiers": cell.authorized_tiers,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entradas compartidas y barrido de la rejilla (A3, A4)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SharedInputs:
    """Las entradas compartidas del barrido: se calculan **una vez** y la rejilla las itera.

    ``analyse`` de #28 se re-derivaria por celda (114 s cada una); aqui solo se re-ejecutan las
    dos piezas que **si** dependen de los parametros (el gate por sesion y la corrida del motor)
    y el resto viaja compartido.
    """

    history: History
    calendar: MarketCalendar
    universe: Universe
    split_plan: SplitPlan
    features: FeatureFrame
    cost_model: CostModel
    slippage: SlippageParameter
    cost: CostBreakdown
    params_oficial: GateParameters
    test_sessions: tuple[date, ...]
    probabilities: Mapping[date, float]
    moves: Mapping[date, Decimal]
    instants: Mapping[date, datetime]
    gated_sessions: tuple[date, ...]

    @property
    def n_test(self) -> int:
        """Sesiones de *test* del plan oficial."""
        return len(self.test_sessions)

    @property
    def n_test_without_expected_move(self) -> int:
        """Sesiones de *test* sin ``garch_forecast``: no se les inventa un movimiento (A11)."""
        return self.n_test - len(self.gated_sessions)


def build_shared_inputs(*, store: Store) -> SharedInputs:
    """Cablea las entradas comunes del barrido **una sola vez** (historia -> probabilidad).

    Es el preambulo de ``pipeline_report.analyse`` sin los tres brazos ni los baselines: lo que
    no cambia entre celdas se calcula aqui y lo que si cambia lo itera ``sweep_cell``.
    """
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
    return SharedInputs(
        history=history,
        calendar=calendar,
        universe=universe,
        split_plan=split_plan,
        features=features,
        cost_model=cost_model,
        slippage=slippage,
        cost=cost,
        params_oficial=GateParameters(),
        test_sessions=test_sessions,
        probabilities=probabilities,
        moves=moves,
        instants=instants,
        gated_sessions=tuple(session for session in test_sessions if session in moves),
    )


@dataclass(frozen=True, slots=True)
class CellResult:
    """El resultado medido de una celda: la corrida, su serie declarada y su metrica."""

    cell: GridCell
    params: GateParameters
    run: BacktestRun
    ledger: ArmLedger
    series_pct: tuple[float, ...]
    mean_return_pct: Mapping[str, object]

    @property
    def n_test(self) -> int:
        """Sesiones de *test* evaluadas por la celda."""
        return self.run.traded + self.run.no_trade + self.run.skipped


def _direction_counts(run: BacktestRun) -> dict[str, int]:
    """``{long, short}`` contado sobre las sesiones operadas: suma exacta = ``traded`` (A9)."""
    counts = {"long": 0, "short": 0}
    for outcome in _sessions_of_run(run):
        if outcome.status != STATUS_TRADED:
            continue
        decision = outcome.decision
        if decision is None:
            raise GateSweepError(
                f"{outcome.session.isoformat()}: una sesion operada sin decision no tiene "
                "direccion que publicar (A9)"
            )
        counts[decision.direction.value] += 1
    return counts


def _declared_cost_sum_pct(run: BacktestRun) -> float:
    """Suma **exacta** del coste declarado de las sesiones operadas, sin intervalo (A4)."""
    return math.fsum(
        float(cast("CostBreakdown", outcome.cost).c_declared_pct)
        for outcome in _sessions_of_run(run)
        if outcome.status == STATUS_TRADED
    )


def _mean_return_block(series_pct: Sequence[float], *, traded: int) -> dict[str, object]:
    """La unica metrica con intervalo de la celda, con la **misma** configuracion que #28 (A5).

    Sin operaciones no se fabrica un cero: la estimacion y los dos limites se publican ``null``
    con su motivo (A8), porque un intervalo bootstrap sobre una serie vacia no existe.
    """
    if traded == 0:
        return _metric_block(
            estimate=None,
            lower=None,
            upper=None,
            basis=BASIS_DECLARED_COST,
            n=len(series_pct),
            seed=METRIC_SEED,
            reason=ZERO_TRADES_REASON,
        )
    return _measure(
        _decimal_series(series_pct),
        _mean,
        scale=METRIC_SCALE,
        basis=BASIS_DECLARED_COST,
        seed=METRIC_SEED,
    )


def sweep_cell(shared: SharedInputs, cell: GridCell) -> CellResult:
    """Evalua **una** celda: el gate de #28 con sus parametros, el motor y su metrica (A4, A5)."""
    params = cell_parameters(cell, cost=shared.cost)
    outputs_oficial, outputs_escenario = _gate_outputs(
        sessions=shared.gated_sessions,
        probabilities=shared.probabilities,
        instants=shared.instants,
        moves=shared.moves,
        calendar=shared.calendar,
        cost=shared.cost,
        params_oficial=shared.params_oficial,
        params_escenario=params,
    )
    ledger = ArmLedger()
    run = run_walk_forward(
        _with_context(
            shared.universe.inputs,
            probabilities=shared.probabilities,
            outputs_oficial=outputs_oficial,
            outputs_escenario=outputs_escenario,
        ),
        split_plan=shared.split_plan,
        cost_model=shared.cost_model,
        slippage=shared.slippage,
        decide_by_fold=_deciders(
            name=ARM_COSTE_DECLARADO,
            params=params,
            ledger=ledger,
            capital_usd=NOTIONAL_USD,
            n_folds=len(shared.split_plan.folds),
        ),
        financing_cut=None,
    )
    series_pct = _series_of_run(run)
    return CellResult(
        cell=cell,
        params=params,
        run=run,
        ledger=ledger,
        series_pct=series_pct,
        mean_return_pct=_mean_return_block(series_pct, traded=run.traded),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Payload y Markdown (A4, A6, A11, A13)
# ─────────────────────────────────────────────────────────────────────────────
def _parameters_payload(params: GateParameters) -> dict[str, object]:
    """Los once parametros resueltos de una celda, como tipos JSON puros."""
    return {
        name: _plain(getattr(params, name), where=f"cell.parameters.{name}")
        for name in GATE_PARAMETER_NAMES
    }


def _cell_payload(result: CellResult, *, shared: SharedInputs) -> dict[str, object]:
    """El bloque publicado de una celda: contrato de sesion, serie declarada y su metrica.

    ``exact_aggregates`` son los recuentos y la suma exacta del coste declarado; el intervalo
    bootstrap vive aparte, en ``mean_return_pct``, porque es una estimacion y no una cifra exacta.
    """
    run = result.run
    declared_cost_sum = _declared_cost_sum_pct(run)
    return {
        "cell_id": result.cell.cell_id,
        "parameters": _parameters_payload(result.params),
        "basis": BASIS_DECLARED_COST,
        "is_validation": False,
        "run_sha256": run.run_sha256,
        "plan_sha256": run.plan_sha256,
        "n_test": result.n_test,
        "traded": run.traded,
        "no_trade": run.no_trade,
        "skipped": run.skipped,
        "direction_counts": _direction_counts(run),
        "declared_cost_sum_pct": declared_cost_sum,
        "exact_aggregates": {
            "n_test": result.n_test,
            "traded": run.traded,
            "declared_cost_sum_pct": declared_cost_sum,
            "n_test_without_expected_move": shared.n_test_without_expected_move,
            "note": (
                "cifras exactas acumuladas del plan y de la tabla de costes declarada: recuentos y "
                "una suma, no estimaciones con intervalo"
            ),
        },
        "declared_series": _declared_series_payload(result.series_pct),
        METRIC_NAME: dict(result.mean_return_pct),
        "rejections": dict(sorted(result.ledger.rejections.items())),
        "ledger": {
            "traded": result.ledger.traded,
            "without_expected_move": result.ledger.without_expected_move,
        },
    }


def _payload(
    *,
    as_of: datetime,
    shared: SharedInputs,
    results: Sequence[CellResult],
) -> dict[str, object]:
    """El payload canonico del barrido: tipos JSON puros y determinista (A2, A11)."""
    payloads = {result.cell.cell_id: _cell_payload(result, shared=shared) for result in results}
    reference = payloads[REFERENCE_CELL]
    reference_sha = cast("Mapping[str, object]", reference["declared_series"])["series_sha256"]
    invariance = {
        control: cast("Mapping[str, object]", payloads[control]["declared_series"])["series_sha256"]
        == reference_sha
        for control in INVARIANCE_CONTROLS
    }
    raw: dict[str, object] = {
        "analysis": ANALYSIS,
        "task": TASK,
        "as_of": as_of.isoformat(),
        "generated_at": as_of.isoformat(),
        "hash_format": HASH_FORMAT,
        "basis": BASIS_DECLARED_COST,
        "is_validation": False,
        "series_id": SERIES_ID,
        "units": SERIES_UNITS,
        "grid": {
            "axes": _grid_axes(),
            "fixed": dict(FIXED_PARAMETERS),
            "cells": [_grid_cell_payload(cell) for cell in GRID_CELLS],
            "n_cells": len(GRID_CELLS),
            "note": (
                "rejilla **declarada** y acotada: `ev_threshold_pct` viaja como multiplo del coste "
                "declarado de ida y vuelta (2*c / 32*c) porque su valor absoluto depende de la "
                "tabla de costes regenerable; ningun parametro fuera de `grid.cells` se barre"
            ),
        },
        "inert_parameters": [dict(item) for item in INERT_PARAMETERS],
        "inertness_scope": INERTNESS_SCOPE,
        "cells": [payloads[result.cell.cell_id] for result in results],
        "invariance": {
            "reference": REFERENCE_CELL,
            "controls": list(INVARIANCE_CONTROLS),
            "measured_on": "declared_series.series_sha256",
            "holds": invariance,
            "note": (
                "los dos controles publican la **misma** serie declarada que la referencia: el "
                "nocional solo dimensiona y el umbral de EV declarado no bindea"
            ),
        },
        "checks": {
            "n_cells": len(results),
            "distinct_traded_counts": len({result.run.traded for result in results}),
            "session_contract": (
                "por celda `traded + no_trade + skipped == n_test` y "
                "`direction_counts.long + direction_counts.short == traded`"
            ),
            "bootstrapped_metrics_per_cell": [METRIC_NAME],
            "metric_seed": METRIC_SEED,
            "basis_values": [BASIS_DECLARED_COST],
        },
        "finding": {
            "id": "favourable_probability",
            "issue": "#98",
            "cell": REFERENCE_CELL,
            "statement": (
                f"en `{REFERENCE_CELL}` todas las operaciones son `short`: el gate de `escenario` "
                "devuelve `direction = nothing` en las 500 sesiones (reglas 9 y 10) y "
                "`_favourable_probability` deriva el tier sobre `1 - p`. La convencion se publica "
                "medida, no se cambia en este barrido (#98)"
            ),
        },
        "limits": {
            "basis": BASIS_DECLARED_COST,
            "is_validation": False,
            "phase0_gate": "fail",
            "phase0_gate_source": "#9/#64 (heredado: no se re-evalua aqui)",
            "verdict": "not_evaluated",
            "verdict_issue": "#29",
            "slippage_state": shared.cost.slippage.state.value,
            "slippage_is_measurement": shared.cost.slippage.is_measurement,
            "decisions_open": ["#59", "#60"],
        },
        "limitations": [
            "el barrido mide la sensibilidad de un **supuesto declarado**: no resuelve la decision "
            "del broker (#59) ni la de umbrales, R y tamano (#60)",
            "el *slippage* sigue supuesto (#64): ninguna cifra de este informe lleva un termino "
            "medido de ejecucion (medirlo es #62 y publicar entonces la base medida es #88)",
            "cada celda publica **una sola** metrica con intervalo por el techo de tiempo: el "
            "resto de metricas de #28 no se re-estiman aqui",
            "las reglas de sesion del gate se evaluan con los valores declarados del pipeline "
            "(`daily/weekly/monthly_pnl_pct = None`): su efecto se mide, no se asume",
        ],
        "does_not_do": [
            {
                "issue": "#59",
                "id": "broker",
                "statement": "no decide el broker: lo deja como el centinela declarado de #28",
            },
            {
                "issue": "#60",
                "id": "thresholds",
                "statement": "no decide umbrales, R ni tamano: mide su sensibilidad",
            },
            {
                "issue": "#82",
                "id": "model_hyperparameters",
                "statement": "el barrido es del gate, no de los hiperparametros del modelo",
            },
            {
                "issue": "#88",
                "id": "measured_slippage_metrics",
                "statement": (
                    "no publica cifras con *slippage* medido: la base es la declarada y #88 esta "
                    "bloqueado en #62"
                ),
            },
            {
                "issue": "#98",
                "id": "favourable_probability_convention",
                "statement": (
                    "no cambia la convencion de `_favourable_probability`: la publica medida"
                ),
            },
            {
                "issue": "#97",
                "id": "stale_runs",
                "statement": "no regenera `baseline_2026-09-20` ni `runs/408fead...`",
            },
        ],
    }
    return cast("dict[str, object]", _plain(raw, where="payload"))


def _metric_text(block: Mapping[str, object]) -> tuple[str, str, str]:
    """Las tres celdas de la metrica en Markdown: ``null`` cuando no hay valor, nunca un 0."""

    def one(key: str) -> str:
        value = block.get(key)
        return "null" if value is None else f"{float(cast('float', value)):.6f}"

    return one("estimate"), one("lower"), one("upper")


def render_markdown(report: GateSweepReport) -> str:
    """El informe del barrido en Markdown, determinista y sin cifras fuera del payload (A13)."""
    payload = report.payload
    grid = cast("Mapping[str, object]", payload["grid"])
    cells = cast("list[dict[str, object]]", payload["cells"])
    invariance = cast("Mapping[str, object]", payload["invariance"])
    finding = cast("Mapping[str, object]", payload["finding"])

    lines: list[str] = [
        f"# Barrido de sensibilidad del gate - `{payload['series_id']}`",
        "",
        f"Generado el `{payload['generated_at']}` (**declarado**, nunca leido del reloj). "
        f"`report_sha256 = {report.report_sha256}`.",
        "",
        f"**No es una validacion.** `basis = {payload['basis']}`, "
        f"`is_validation = {payload['is_validation']}`: la rejilla mide la sensibilidad de un "
        "supuesto declarado y la base sigue siendo el coste declarado de #28. El veredicto de "
        "Fase 2 es #29.",
        "",
        "## Rejilla declarada",
        "",
        f"`{grid['n_cells']}` celdas. Campos fijos: `{grid['fixed']}`.",
        "",
        "| celda | `tier_a_min_probability` | `authorized_tiers` | `tier_a_cost_multiple` | "
        "`tier_b_cost_multiple` | `ev_threshold_pct` | `risk_per_trade_pct` |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in cast("list[dict[str, object]]", grid["cells"]):
        lines.append(
            f"| `{cell['cell_id']}` | {cell['tier_a_min_probability']} | "
            f"{cell['authorized_tiers']} | {cell['tier_a_cost_multiple']} | "
            f"{cell['tier_b_cost_multiple']} | `{cell['ev_threshold_pct']}` | "
            f"{cell['risk_per_trade_pct']} |"
        )
    lines.extend(
        [
            "",
            "## Resultados por celda",
            "",
            "| celda | operadas | no_trade | skipped | long | short | coste declarado sum % | "
            "media % | IC95 inferior | IC95 superior |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for cell in cells:
        block = cast("Mapping[str, object]", cell[METRIC_NAME])
        estimate, lower, upper = _metric_text(block)
        counts = cast("Mapping[str, object]", cell["direction_counts"])
        lines.append(
            f"| `{cell['cell_id']}` | {cell['traded']} | {cell['no_trade']} | {cell['skipped']} | "
            f"{counts['long']} | {counts['short']} | "
            f"{cast('float', cell['declared_cost_sum_pct']):.6f} | {estimate} | {lower} | "
            f"{upper} |"
        )
    lines.extend(
        [
            "",
            "## Invariancia",
            "",
            f"- Referencia `{invariance['reference']}`, controles "
            f"{invariance['controls']}, medido sobre `{invariance['measured_on']}`: "
            f"{invariance['holds']}.",
            f"- {invariance['note']}.",
            "",
            "## Campos inertes",
            "",
            f"- {payload['inertness_scope']}.",
        ]
    )
    for item in cast("list[dict[str, str]]", payload["inert_parameters"]):
        lines.append(f"- `{item['field']}`: {item['reason']}")
    lines.extend(
        [
            "",
            "## Hallazgo declarado",
            "",
            f"- **{finding['issue']}** - `{finding['id']}`: {finding['statement']}.",
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
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Ejecucion (A1, A3)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class GateSweepReport:
    """El informe del barrido: payload canonico, hash y los objetos que lo produjeron."""

    as_of: datetime
    report_date: date
    payload: dict[str, object]
    report_sha256: str
    reports_dir: Path
    shared: SharedInputs
    results: tuple[CellResult, ...]

    @property
    def report_stem(self) -> str:
        """Nombre base del informe: ``gate_sweep_<AAAA-MM-DD>``."""
        return f"{REPORT_PREFIX}_{self.report_date.isoformat()}"

    def json_text(self) -> str:
        """JSON determinista: mismo ``as_of`` ⇒ mismo texto byte a byte (A3)."""
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

    def cell(self, cell_id: str) -> CellResult:
        """La celda con ese identificador, o error tipado: nunca ``None`` silencioso."""
        for result in self.results:
            if result.cell.cell_id == cell_id:
                return result
        raise GateSweepError(
            f"el barrido no trae la celda {cell_id!r}: las declaradas son "
            f"{[cell.cell_id for cell in GRID_CELLS]} (A2)"
        )


def _digest(payload: Mapping[str, object]) -> str:
    """``report_sha256`` con el prefijo ``sha256:`` (A11: nunca un digest desnudo)."""
    return f"{HASH_PREFIX}{hashlib.sha256(canonical_text(payload).encode('utf-8')).hexdigest()}"


def analyse(
    *, store: Store, reports_dir: Path, as_of: datetime, write: bool = True
) -> GateSweepReport:
    """Corre la rejilla declarada sobre las entradas compartidas y (si se pide) escribe (A1, A3).

    ``as_of`` es **obligatorio** y es el unico instante de la corrida: ninguna ruta consulta el
    reloj. ``write = False`` no escribe **nada**.
    """
    moment = _as_utc(as_of)
    shared = build_shared_inputs(store=store)
    results = tuple(sweep_cell(shared, cell) for cell in GRID_CELLS)
    payload = _payload(as_of=moment, shared=shared, results=results)
    report = GateSweepReport(
        as_of=moment,
        report_date=moment.astimezone(UTC).date(),
        payload=payload,
        report_sha256=_digest(payload),
        reports_dir=reports_dir,
        shared=shared,
        results=results,
    )
    if write:
        json_path, markdown_path = report.write(reports_dir)
        logger.info("barrido del gate de #86: {} y {}", json_path, markdown_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del barrido de sensibilidad del gate.

    Codigos de salida: ``0`` = informe escrito; ``2`` = falta o no es valido ``--as-of``, falta un
    dataset o la muestra no alcanza: **no se escribe nada** y el motivo sale por ``stderr``.
    """
    parser = argparse.ArgumentParser(
        prog="cfdtrader.analysis.gate_sweep",
        description="Barrido de sensibilidad del gate sobre el brazo de coste declarado de #28",
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
        print(f"no se puede emitir el barrido del gate: {error}", file=sys.stderr)
        return 2

    data_root = Path(args.data_root) if args.data_root is not None else Path(settings.data.root)
    reports_dir = (
        Path(args.reports_dir)
        if args.reports_dir is not None
        else data_root / "derived" / "reports"
    )
    try:
        report = analyse(store=Store(data_root), reports_dir=reports_dir, as_of=moment, write=True)
    except (PipelineReportError, ConfigurationError) as error:
        print(f"no se puede emitir el barrido del gate: {error}", file=sys.stderr)
        return 2

    logger.info(
        "barrido de #86: {} celdas; report_sha256 = {}",
        len(report.results),
        report.report_sha256,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
