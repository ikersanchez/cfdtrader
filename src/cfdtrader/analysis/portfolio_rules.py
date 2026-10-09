"""Acumulacion del P&L realizado para el *kill switch* de cartera (tarea #83).

**Que entrega.** Las tres cifras de cartera que el gate de #27 recibe como argumentos explicitos
—``daily_pnl_pct``, ``weekly_pnl_pct`` y ``monthly_pnl_pct``— acumuladas a partir de las
**operaciones cerradas**, con la convencion de ventana **declarada** y **point-in-time**: para
decidir la sesion ``t`` solo se suman las operaciones cuyo ``trade_date`` es **estrictamente
anterior** a ``t`` (la sesion que se decide no ha cerrado todavia, asi que su resultado no existe).

**Por que existe.** Las reglas 3, 4 y 5 de ``plan.md`` §12 (perdida diaria -2 %, semanal -5 % y
mensual -10 %) se evaluan en el gate, que es una funcion **pura**: compara el acumulado contra su
umbral, pero **no acumula nada**. Hasta #83 el camino diario llamaba al gate con los tres valores a
``None`` y ``analysis/gate_sweep.py`` lo publicaba como parametro inerte: el *kill switch* era una
valla que no podia dispararse. Este modulo cierra ese hueco.

**Donde vive el estado.** En el **diario** (``journal.trades``, §12.5), que es *append-only* e
inmutable (§19.1): la acumulacion se **recomputa** de el. No hay contador en memoria ni fichero de
estado propio, porque un segundo sitio con la misma cifra seria una **segunda fuente de verdad**
que puede divergir; el diario es, por diseno, el dato irreversible del sistema. El estado persistido
**es** la tabla de operaciones cerradas; el acumulado es una funcion pura suya.

**De donde salen las operaciones cerradas.** Dos vias, declaradas:

- ``journal.trades`` — la operacion **real** (#47). En observacion esa tabla esta **vacia** y eso es
  lo correcto, no un olvido (§19.11); el estado es entonces ``sin_historial`` y el gate recibe
  ``None``, ``None``, ``None``: **no cambia nada** hasta que exista una operacion real.
- **Recomputacion** desde ``journal.decisions`` mas el almacen de mercado — la **misma** maquina de
  #45 (``analysis/paper_trading.py``, reutilizada **por import**: definir una segunda aritmetica de
  coste seria un segundo coste, §19.14). Es la via del *backtest* (#28).

**La trampa de unidades (``%`` del nocional frente a ``%`` del capital), y como se resuelve.** El
limite de §12 es **% del capital**; el retorno de una operacion se publica **% del nocional**. El
nocional sale del riesgo —``notional = capital x riesgo / stop`` (#27, A5)—, asi que una salida en
el stop pierde ``stop_pct`` % del nocional, que son **``riesgo`` % del capital** (1 %). La
conversion es **explicita** (``capital_pct = nocional_pct x apalancamiento``) y el factor se
publica operacion a operacion, en vez de asumir que vale 1 (cierto solo si ``stop_pct == 1 %``).

**Convencion de ventana, del signo y del redondeo.** Las tres ventanas son de **calendario**, no de
sesiones: el dia natural **ET** (``America/New_York``), la semana **ISO-8601** (lunes→domingo) y el
mes natural. La del signo es la de §12: una perdida es **negativa** y el bloqueo salta con
``valor <= -limite`` (lo aplica el gate, no este modulo). La suma es **exacta** en ``Decimal``: sin
``float`` en el nucleo, para que dos procesos con ``PYTHONHASHSEED`` distinto den la misma cifra.

**Que NO hace este modulo** (fronteras declaradas, con su issue)

- **No** decide los umbrales: son la decision 5 de ``tech_stack.md`` §11 bis, cerrada por **#60** y
  registrada en ``plan.md`` §19.12; aqui solo se **copian** con su procedencia y un test comprueba
  que coinciden con los que el escenario S1 sirve al gate (``scenario_parameters``).
- **No** es #27: no evalua las reglas ni decide. Publica tres cifras; el gate las recibe.
- **No** es #28: no cablea la acumulacion en el bucle de sesiones del *backtest*; le deja la funcion
  pura (:func:`walk`) y la via de recomputacion.
- **No** es #47/#84: no escribe ``journal.trades``, no coloca ordenes y no cierra posiciones.
- **No** valida el ``trade_date`` contra el calendario de sesiones: el contrato de escritura es de
  #47 (un dia no sesion es un problema de su camino, no una ventana de este).

Sin reloj (la sesion entra por ``--session`` o por ``--as-of``) y sin red; salida **determinista
byte a byte**: ``report_sha256`` es el sha256 del texto canonico de #13 sobre el payload sin esa
clave.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Final, cast

from cfdtrader.analysis.backtest_report import NOTIONAL_USD
from cfdtrader.analysis.paper_trading import (
    DecisionOutcome,
    _as_date,  # pyright: ignore[reportPrivateUsage]
    _daily_by_session,  # pyright: ignore[reportPrivateUsage]
    _intraday_by_session,  # pyright: ignore[reportPrivateUsage]
    _outcome,  # pyright: ignore[reportPrivateUsage]
    _recommendations,  # pyright: ignore[reportPrivateUsage]
)
from cfdtrader.backtest.engine import canonical_text
from cfdtrader.data.store import Store
from cfdtrader.journal.decision_log import Journal, read_decisions, read_table

__all__ = [
    "CAPITAL_PROVENANCE",
    "CAPITAL_USD",
    "DAY_TIMEZONE",
    "LIMITS",
    "MODULE",
    "REPORT_PREFIX",
    "SHA256_PREFIX",
    "SOURCE_RECOMPUTED",
    "SOURCE_TRADES",
    "STATE_IN_PROGRESS",
    "STATE_NO_HISTORY",
    "STATE_UNAVAILABLE",
    "TASK",
    "WINDOWS",
    "ClosedTrade",
    "LossLimits",
    "PortfolioRulesError",
    "PortfolioRulesReport",
    "WindowTotals",
    "accumulate",
    "analyse",
    "assess",
    "closed_trades_from_journal",
    "main",
    "recompute_closed_trades",
    "render_markdown",
    "walk",
    "write_report",
]

MODULE: Final[str] = "cfdtrader.analysis.portfolio_rules"
TASK: Final[str] = "#83"
ANALYSIS: Final[str] = MODULE
REPORT_PREFIX: Final[str] = "portfolio_rules"

#: Prefijo obligatorio de los digests del repositorio (detect-secrets: nunca un hex desnudo).
SHA256_PREFIX: Final[str] = "sha256:"

#: Zona de la convencion de dia natural: el dia de la sesion es el dia en **ET** (§12, §13).
DAY_TIMEZONE: Final[str] = "America/New_York"

#: Los tres estados de la acumulacion. ``sin_historial`` no es un cero: es una ausencia declarada.
STATE_IN_PROGRESS: Final[str] = "en_curso"
STATE_NO_HISTORY: Final[str] = "sin_historial"
STATE_UNAVAILABLE: Final[str] = "unavailable"

#: Las dos vias de operaciones cerradas: la real (#47) y la recomputada (la maquina de #45).
SOURCE_TRADES: Final[str] = "trades"
SOURCE_RECOMPUTED: Final[str] = "recomputed"

#: El capital de referencia: el mismo que el camino diario pasa al gate como ``capital_usd``.
CAPITAL_USD: Final[Decimal] = NOTIONAL_USD
CAPITAL_PROVENANCE: Final[str] = (
    "`analysis/backtest_report.NOTIONAL_USD` (tabla declarada de `plan.md` §3.3 vía #8): es el "
    "`capital_usd` que el camino diario pasa al gate, y por tanto la base de la conversion de "
    "`%` del nocional a `%` del capital"
)

#: Precision de la suma interna. El valor **no** se redondea en ningun punto —ni al publicarlo ni al
#: pasarlo al gate—: redondear una cifra que decide podria esconder un incumplimiento de 0,0001.
SUM_PRECISION: Final[int] = 34


class PortfolioRulesError(Exception):
    """Raiz de los errores del modulo (argumentos invalidos, diario o almacen ilegibles)."""


@dataclass(frozen=True, slots=True)
class LossLimits:
    """Los tres umbrales de §12, **copiados** de la decision de #60 (no se deciden aqui)."""

    max_daily_loss_pct: Decimal
    max_weekly_loss_pct: Decimal
    max_monthly_loss_pct: Decimal


#: Los umbrales decididos el 2026-10-04 (decision abierta 5, tarea #60) y registrados en §19.12.
#: ``tests/test_portfolio_rules.py`` comprueba que coinciden con los que ``scenario_parameters``
#: (el escenario S1) sirve al gate: la copia se **verifica**, no se confia.
LIMITS: Final[LossLimits] = LossLimits(
    max_daily_loss_pct=Decimal("2"),
    max_weekly_loss_pct=Decimal("5"),
    max_monthly_loss_pct=Decimal("10"),
)

#: Las tres ventanas de §12 reglas 3, 4 y 5: su cubo, su rearme y su umbral. Se publican enteras
#: porque la convencion **es** parte del entregable: elegirla despues de ver el resultado seria
#: elegirla a conveniencia (§19.11).
WINDOWS: Final[tuple[dict[str, str], ...]] = (
    {
        "rule": "3",
        "field": "daily_pnl_pct",
        "label": "perdida diaria",
        "bucket": "dia natural ET (`America/New_York`) del `trade_date`",
        "rearm": "al empezar el siguiente dia natural ET",
        "limit_pct": str(LIMITS.max_daily_loss_pct),
        "issue": "#60",
        "note": (
            "la ventana es el dia **de la sesion**; como la decision es anterior al cierre de la "
            "sesion y la regla 1 admite una sola operacion, su cubo va vacio por construccion y la "
            "cifra es 0 con su recuento. Protege el caso de la regla 12 (nunca ampliar una "
            "perdedora) y cualquier re-evaluacion intradia, no una decision al alza"
        ),
    },
    {
        "rule": "4",
        "field": "weekly_pnl_pct",
        "label": "perdida semanal",
        "bucket": "semana ISO-8601 (lunes a domingo) del `trade_date`",
        "rearm": "al empezar la siguiente semana ISO (lunes 00:00 ET)",
        "limit_pct": str(LIMITS.max_weekly_loss_pct),
        "issue": "#60",
        "note": (
            "acumula las sesiones **cerradas** de la semana ISO en curso; es la primera ventana "
            "que puede dispararse de verdad en una sola sesion de decision"
        ),
    },
    {
        "rule": "5",
        "field": "monthly_pnl_pct",
        "label": "perdida mensual",
        "bucket": "mes natural del `trade_date`",
        "rearm": "al empezar el siguiente mes natural (dia 1, 00:00 ET)",
        "limit_pct": str(LIMITS.max_monthly_loss_pct),
        "issue": "#60",
        "note": "acumula las sesiones **cerradas** del mes natural en curso",
    },
)


# ─────────────────────────────────────────────────────────────────────────────
# Operaciones cerradas, ventanas y utilidades exactas
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ClosedTrade:
    """Una operacion **cerrada**: lo unico que suma el *kill switch*.

    ``capital_pct`` es el retorno neto en ``%`` del **capital** —ya convertido desde ``%`` del
    nocional con el apalancamiento de la operacion—; ``notional_pct`` y ``leverage`` se conservan
    para poder auditar esa conversion.
    """

    trade_date: date
    capital_pct: Decimal
    notional_pct: Decimal
    leverage: Decimal
    exit_reason: str | None = None
    source: str = "journal.trades"


@dataclass(frozen=True, slots=True)
class TradeLedger:
    """Las operaciones legibles de una fuente, mas las que **no** se pudieron convertir.

    Una fila que no se puede convertir a ``%`` del capital no se descarta en silencio ni se rellena
    con un cero: se publica en ``skipped`` con su motivo.
    """

    trades: tuple[ClosedTrade, ...]
    skipped: tuple[dict[str, str], ...]
    source: str


@dataclass(frozen=True, slots=True)
class WindowTotals:
    """Las tres cifras de cartera **al empezar** la sesion, con su recuento y su procedencia."""

    session: date
    state: str
    reason: str
    capital_usd: Decimal
    limits: LossLimits
    daily_pnl_pct: Decimal | None
    weekly_pnl_pct: Decimal | None
    monthly_pnl_pct: Decimal | None
    daily_trades: int
    weekly_trades: int
    monthly_trades: int
    history_trades: int
    considered_trades: int
    last_closed: date | None
    source: str

    def for_gate(self) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """Los tres argumentos que el gate de #27 espera, en su orden (``None`` = sin historial)."""
        return (self.daily_pnl_pct, self.weekly_pnl_pct, self.monthly_pnl_pct)


@dataclass(frozen=True, slots=True)
class PortfolioRulesReport:
    """Informe publicado (``.json`` + ``.md``)."""

    session: date
    as_of: datetime
    payload: dict[str, object]
    report_sha256: str


def _dec(value: object) -> Decimal:
    """Un porcentaje **exacto**: un ``float`` se lee por su ``repr`` (nunca por su binario)."""
    if isinstance(value, bool):
        raise PortfolioRulesError("un booleano no es un porcentaje")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(repr(value))
    if isinstance(value, str):
        try:
            return Decimal(value)
        except InvalidOperation as error:
            raise PortfolioRulesError(f"no es un porcentaje legible: {value!r}") from error
    raise PortfolioRulesError(f"no es un porcentaje: {type(value).__name__}")


def _optional_dec(value: object) -> Decimal | None:
    """Un porcentaje opcional del diario: ``None`` sigue siendo ``None`` (nunca un cero)."""
    if value is None:
        return None
    try:
        return _dec(value)
    except PortfolioRulesError:
        return None


def _iso_week(moment: date) -> tuple[int, int]:
    """El par ``(ano ISO, semana ISO)`` de la fecha: la semana de §12 regla 4 es lunes a domingo."""
    calendar_week = moment.isocalendar()
    return (calendar_week.year, calendar_week.week)


def _month(moment: date) -> tuple[int, int]:
    """El par ``(ano, mes)`` de la fecha: el mes de §12 regla 5 es el natural."""
    return (moment.year, moment.month)


def _sum(values: Iterable[Decimal]) -> Decimal:
    """Suma exacta con la precision declarada: un contexto propio, no el del proceso."""
    with localcontext() as context:
        context.prec = SUM_PRECISION
        total = Decimal(0)
        for value in values:
            total += value
        return total


# ─────────────────────────────────────────────────────────────────────────────
# El nucleo: la acumulacion por ventana (pura, sin reloj, sin red, sin disco)
# ─────────────────────────────────────────────────────────────────────────────
def _require_session(value: object) -> date:
    """La sesion es una fecha de calendario: un ``datetime`` no cuela (arrastraria una hora)."""
    if isinstance(value, datetime) or not isinstance(value, date):
        raise PortfolioRulesError(f"`session` tiene que ser una fecha, no {type(value).__name__}")
    return value


def accumulate(
    trades: Iterable[ClosedTrade],
    *,
    session: date,
    capital_usd: Decimal = CAPITAL_USD,
    limits: LossLimits = LIMITS,
    source: str = "journal.trades",
) -> WindowTotals:
    """Las tres ventanas de §12 **al empezar** la sesion ``session`` (point-in-time).

    Solo entran las operaciones con ``trade_date`` **estrictamente anterior** a ``session``: la
    sesion que se decide no ha cerrado todavia, asi que su resultado no existe y usarlo seria
    *look-ahead*. El cubo es el de la **fecha de cierre** de la operacion, no el de la sesion, de
    modo que una perdida de la semana pasada no se arrastra a la ventana semanal de esta.

    Con el diario **sin** operaciones cerradas el estado es ``sin_historial`` y las tres cifras son
    ``None`` (**nunca** ``0``): no es que no haya perdida, es que no hay cartera que medir todavia.
    """
    session = _require_session(session)
    history = tuple(sorted(trades, key=lambda trade: trade.trade_date))
    considered = tuple(trade for trade in history if trade.trade_date < session)
    last_closed = history[-1].trade_date if history else None
    if not history:
        return WindowTotals(
            session=session,
            state=STATE_NO_HISTORY,
            reason=(
                "el diario no tiene ninguna operacion cerrada: la contabilidad empieza con la "
                "primera operacion real (#47), y hasta entonces el gate recibe `None` en las tres "
                "reglas, no un cero"
            ),
            capital_usd=capital_usd,
            limits=limits,
            daily_pnl_pct=None,
            weekly_pnl_pct=None,
            monthly_pnl_pct=None,
            daily_trades=0,
            weekly_trades=0,
            monthly_trades=0,
            history_trades=0,
            considered_trades=0,
            last_closed=None,
            source=source,
        )
    day_values = [trade.capital_pct for trade in considered if trade.trade_date == session]
    week_values = [
        trade.capital_pct
        for trade in considered
        if _iso_week(trade.trade_date) == _iso_week(session)
    ]
    month_values = [
        trade.capital_pct for trade in considered if _month(trade.trade_date) == _month(session)
    ]
    return WindowTotals(
        session=session,
        state=STATE_IN_PROGRESS,
        reason=(
            "acumulado de las operaciones cerradas con `trade_date` anterior a la sesion; los "
            "cubos son de calendario (dia natural ET, semana ISO-8601 y mes natural)"
        ),
        capital_usd=capital_usd,
        limits=limits,
        daily_pnl_pct=_sum(day_values),
        weekly_pnl_pct=_sum(week_values),
        monthly_pnl_pct=_sum(month_values),
        daily_trades=len(day_values),
        weekly_trades=len(week_values),
        monthly_trades=len(month_values),
        history_trades=len(history),
        considered_trades=len(considered),
        last_closed=last_closed,
        source=source,
    )


def assess(totals: WindowTotals) -> tuple[dict[str, object], ...]:
    """El veredicto **informativo** de cada ventana contra su umbral (quien bloquea es el gate).

    Se publica para que el operador vea la valla antes de que el gate la aplique, con el signo de
    §12: la perdida es negativa y el incumplimiento es ``valor <= -limite``.
    """
    limits = {
        "3": totals.limits.max_daily_loss_pct,
        "4": totals.limits.max_weekly_loss_pct,
        "5": totals.limits.max_monthly_loss_pct,
    }
    values: dict[str, Decimal | None] = {
        "3": totals.daily_pnl_pct,
        "4": totals.weekly_pnl_pct,
        "5": totals.monthly_pnl_pct,
    }
    counts = {"3": totals.daily_trades, "4": totals.weekly_trades, "5": totals.monthly_trades}
    verdicts: list[dict[str, object]] = []
    for window in WINDOWS:
        rule = window["rule"]
        limit = limits[rule]
        value = values[rule]
        verdicts.append(
            {
                "rule": rule,
                "field": window["field"],
                "label": window["label"],
                "value_pct": None if value is None else str(value),
                "limit_pct": str(limit),
                "threshold_pct": str(-limit),
                "closed_trades": counts[rule],
                "breached": None if value is None else value <= -limit,
                "verdict": (
                    "sin_historial"
                    if value is None
                    else ("incumplida" if value <= -limit else "dentro_del_limite")
                ),
            }
        )
    return tuple(verdicts)


def walk(
    trades: Iterable[ClosedTrade],
    sessions: Iterable[date],
    *,
    capital_usd: Decimal = CAPITAL_USD,
    limits: LossLimits = LIMITS,
    source: str = "journal.trades",
) -> tuple[WindowTotals, ...]:
    """La acumulacion de **cada** sesion de la serie, en orden: la via del *backtest* (#28).

    Es la misma funcion pura aplicada sesion a sesion, de modo que #28 no tiene que reimplementar
    la ventana: recorre las sesiones y pasa el resultado a ``evaluate_gate``.
    """
    ordered = tuple(sorted(trades, key=lambda trade: trade.trade_date))
    return tuple(
        accumulate(ordered, session=session, capital_usd=capital_usd, limits=limits, source=source)
        for session in sessions
    )


# ─────────────────────────────────────────────────────────────────────────────
# Las dos fuentes de operaciones cerradas
# ─────────────────────────────────────────────────────────────────────────────
def _capital_pct(*, pnl_pct: Decimal, notional: Decimal, capital_usd: Decimal) -> Decimal:
    """De ``%`` del nocional a ``%`` del capital: la conversion declarada, en un solo sitio."""
    with localcontext() as context:
        context.prec = SUM_PRECISION
        return pnl_pct * notional / capital_usd


def closed_trades_from_journal(
    journal_root: Path | str, *, capital_usd: Decimal = CAPITAL_USD
) -> TradeLedger:
    """Las operaciones **reales** de ``journal.trades`` (§12.5), la via de #47.

    En observacion la tabla esta **vacia** y eso es lo correcto (§19.11); entonces el ledger sale
    vacio, sin filas inventadas. ``pnl_pct`` se lee en ``%`` del **nocional** —la misma magnitud que
    el retorno que publica #45— y se convierte con ``notional / capital``: el contrato de escritura
    de la tabla es de #47, y si algun dia escribe en ``%`` del capital la conversion se declara
    alli, no se adivina aqui.
    """
    rows = read_table(Journal(Path(journal_root)), "trades")
    trades: list[ClosedTrade] = []
    skipped: list[dict[str, str]] = []
    for row in rows:
        trade_date = _as_date(row.get("trade_date"))
        pnl_pct = _optional_dec(row.get("pnl_pct"))
        notional = _optional_dec(row.get("notional"))
        if trade_date is None:
            skipped.append(
                {"trade_date": str(row.get("trade_date")), "reason": "sin fecha legible"}
            )
            continue
        if pnl_pct is None or notional is None or notional <= 0 or capital_usd <= 0:
            skipped.append(
                {
                    "trade_date": trade_date.isoformat(),
                    "reason": (
                        "sin `pnl_pct` y `notional` utilizables: no hay con que convertir a `%` "
                        "del capital y no se rellena con un cero"
                    ),
                }
            )
            continue
        leverage = notional / capital_usd
        trades.append(
            ClosedTrade(
                trade_date=trade_date,
                capital_pct=_capital_pct(
                    pnl_pct=pnl_pct, notional=notional, capital_usd=capital_usd
                ),
                notional_pct=pnl_pct,
                leverage=leverage,
                exit_reason=None if row.get("exit_reason") is None else str(row["exit_reason"]),
                source="journal.trades",
            )
        )
    return TradeLedger(trades=tuple(trades), skipped=tuple(skipped), source="journal.trades")


def recompute_closed_trades(
    store: Store, journal_root: Path | str, *, capital_usd: Decimal = CAPITAL_USD
) -> TradeLedger:
    """Las operaciones **recomputadas** desde ``journal.decisions`` mas el almacen (via #28).

    Reutiliza **por import** la maquina de #45 (`analysis/paper_trading.py`) en vez de reimplementar
    el resultado y su coste: una segunda aritmetica de coste seria definir un segundo coste, que es
    justo lo que §19.14 prohibe. El apalancamiento de cada sesion lo publica el gate en la fila de
    ``journal.decisions`` (``leverage_implied``) y es lo que convierte el retorno de ``%`` del
    nocional a ``%`` del capital.
    """
    decisions = read_decisions(Journal(Path(journal_root)))
    emitted, _counts = _recommendations(decisions)
    daily = _daily_by_session(store)
    intraday = _intraday_by_session(store)
    trades: list[ClosedTrade] = []
    skipped: list[dict[str, str]] = []
    for row in emitted:
        outcome: DecisionOutcome | None = _outcome(row, daily=daily, intraday=intraday)
        session = _as_date(row.get("trade_date"))
        label = "?" if session is None else session.isoformat()
        if outcome is None:
            skipped.append(
                {
                    "trade_date": label,
                    "reason": (
                        "el almacen no tiene la sesion (o le faltan `stop_pct`/`target_pct`/"
                        "`cost_pct`): el resultado no se puede recomputar"
                    ),
                }
            )
            continue
        leverage = _optional_dec(row.get("leverage_implied"))
        if leverage is None or leverage <= 0:
            skipped.append(
                {
                    "trade_date": label,
                    "reason": (
                        "la fila de `journal.decisions` no trae `leverage_implied`: sin el no se "
                        "puede convertir el retorno de `%` del nocional a `%` del capital"
                    ),
                }
            )
            continue
        notional_pct = _dec(outcome.net_return_pct)
        trades.append(
            ClosedTrade(
                trade_date=outcome.trade_date,
                capital_pct=notional_pct * leverage,
                notional_pct=notional_pct,
                leverage=leverage,
                exit_reason=outcome.exit_reason,
                source="journal.decisions+almacen",
            )
        )
    return TradeLedger(
        trades=tuple(trades), skipped=tuple(skipped), source="journal.decisions+almacen"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Analisis
# ─────────────────────────────────────────────────────────────────────────────
def _trade_payload(trade: ClosedTrade) -> dict[str, object]:
    """Una operacion cerrada, con la conversion visible para poder auditarla."""
    return {
        "trade_date": trade.trade_date.isoformat(),
        "notional_pct": str(trade.notional_pct),
        "leverage": str(trade.leverage),
        "capital_pct": str(trade.capital_pct),
        "exit_reason": trade.exit_reason,
        "source": trade.source,
    }


def _totals_payload(totals: WindowTotals) -> dict[str, object]:
    """El bloque de acumulacion: las tres cifras, su recuento y su procedencia."""
    return {
        "session": totals.session.isoformat(),
        "state": totals.state,
        "reason": totals.reason,
        "day_bucket": "dia natural ET (`America/New_York`)",
        "week_bucket": "semana ISO-8601 (lunes a domingo)",
        "month_bucket": "mes natural",
        "as_of_rule": (
            "solo entran las operaciones con `trade_date` **estrictamente anterior** a la sesion: "
            "la sesion que se decide no ha cerrado todavia"
        ),
        "daily_pnl_pct": None if totals.daily_pnl_pct is None else str(totals.daily_pnl_pct),
        "weekly_pnl_pct": None if totals.weekly_pnl_pct is None else str(totals.weekly_pnl_pct),
        "monthly_pnl_pct": None if totals.monthly_pnl_pct is None else str(totals.monthly_pnl_pct),
        "daily_trades": totals.daily_trades,
        "weekly_trades": totals.weekly_trades,
        "monthly_trades": totals.monthly_trades,
        "history_trades": totals.history_trades,
        "considered_trades": totals.considered_trades,
        "last_closed": None if totals.last_closed is None else totals.last_closed.isoformat(),
        "source": totals.source,
    }


def _limits_payload(limits: LossLimits) -> dict[str, object]:
    """Los tres umbrales con su procedencia (se **copian** de #60; no se deciden aqui)."""
    return {
        "max_daily_loss_pct": str(limits.max_daily_loss_pct),
        "max_weekly_loss_pct": str(limits.max_weekly_loss_pct),
        "max_monthly_loss_pct": str(limits.max_monthly_loss_pct),
        "issue": "#60",
        "note": (
            "los umbrales los decide el propietario (decision abierta 5, #60; `plan.md` §19.12) y "
            "el gate los lee de `GateParameters`: aqui se **copian** para publicar la valla, y el "
            "test los coteja con el escenario S1"
        ),
    }


def _payload(
    *,
    as_of: datetime,
    totals: WindowTotals,
    ledger: TradeLedger,
    limits: LossLimits,
) -> dict[str, object]:
    """El payload publicable (sin `report_sha256`; lo anade :func:`analyse`)."""
    daily, weekly, monthly = totals.for_gate()
    return {
        "analysis": ANALYSIS,
        "task": TASK,
        "generated_at": as_of.isoformat(),
        "phase": "Fase 4 — operacion (carril A, ejecucion manual)",
        "basis": "declared_cost",
        "is_measurement": False,
        "is_validation": False,
        "session": totals.session.isoformat(),
        "window_convention": (
            "ventanas de **calendario** (dia natural ET, semana ISO-8601, mes natural) sobre la "
            "fecha de cierre de cada operacion; el rearme lo declara `rearm` en cada ventana"
        ),
        "capital": {"usd": str(CAPITAL_USD), "provenance": CAPITAL_PROVENANCE},
        "limits": _limits_payload(limits),
        "windows": [dict(window) for window in WINDOWS],
        "source": {
            "kind": ledger.source,
            "note": (
                "`journal.trades` es la operacion real (#47) y en observacion esta vacia (§19.11); "
                "la otra via recomputa el resultado con la maquina de #45 (`paper_trading`) desde "
                "`journal.decisions` mas el almacen"
            ),
            "skipped": [dict(entry) for entry in ledger.skipped],
        },
        "accumulation": _totals_payload(totals),
        "trades": [_trade_payload(trade) for trade in ledger.trades],
        "assessment": [dict(verdict) for verdict in assess(totals)],
        "gate_input": {
            "daily_pnl_pct": None if daily is None else str(daily),
            "weekly_pnl_pct": None if weekly is None else str(weekly),
            "monthly_pnl_pct": None if monthly is None else str(monthly),
            "note": (
                "es lo que el camino diario pasa a `evaluate_gate`: con `sin_historial` van los "
                "tres a `None` y el gate declara «sin ... declarado en esta llamada», nunca un 0"
            ),
        },
        "does_not_do": [
            {
                "id": "no_decide_umbrales",
                "statement": "los umbrales son de #60 (§19.12); aqui se copian y se cotejan",
            },
            {
                "id": "no_evalua_las_reglas",
                "statement": (
                    "quien bloquea es el gate (#27); `assessment` es informativo y reproduce su "
                    "misma comparacion"
                ),
            },
            {
                "id": "no_cablea_el_backtest",
                "statement": (
                    "la acumulacion sesion a sesion del motor es #28: aqui se le deja la funcion "
                    "pura `walk`"
                ),
            },
            {
                "id": "no_escribe_trades",
                "statement": (
                    "no escribe `journal.trades` ni cierra posiciones: el contrato de escritura es "
                    "#47/#84"
                ),
            },
        ],
        "follow_ups": [
            {"issue": "#47", "topic": "operacion real: escribe `journal.trades` y usa esta valla"},
            {
                "issue": "#28",
                "topic": "cablear `walk` en el bucle de sesiones del backtest (kill switch medido)",
            },
            {"issue": "#62", "topic": "el coste de la recomputacion sigue siendo el declarado"},
            {
                "issue": "#84",
                "topic": "el *bracket* del broker y el incumplimiento de las 16:00 ET",
            },
        ],
        "honesty": {
            "edge": "no demostrado",
            "phase2": "`not_evaluable`/`fail`; `phase2_ready = false`",
            "lane": "carril A (asistente de decision, ejecucion manual)",
            "note": (
                "la valla de cartera **no** es una afirmacion de *edge*: hace verificables las "
                "reglas 3, 4 y 5 de §12. `§11.6` **no** se altera y el carril B sigue bloqueado"
            ),
        },
    }


def analyse(
    *,
    session: date,
    as_of: datetime,
    journal_root: Path | str,
    store: Store | None = None,
    source: str = SOURCE_TRADES,
    capital_usd: Decimal = CAPITAL_USD,
    limits: LossLimits = LIMITS,
    reports_dir: Path | str | None = None,
    write: bool = True,
) -> PortfolioRulesReport:
    """Acumula el P&L realizado **al empezar** la sesion ``session`` y publica el informe.

    ``source = "trades"`` lee ``journal.trades`` (la operacion real, #47); ``source = "recomputed"``
    recomputa el resultado con la maquina de #45 y **exige** ``store``.
    """
    if source == SOURCE_TRADES:
        ledger = closed_trades_from_journal(journal_root, capital_usd=capital_usd)
    elif source == SOURCE_RECOMPUTED:
        if store is None:
            raise PortfolioRulesError(
                "la via `recomputed` necesita `store` (el almacen de mercado)"
            )
        ledger = recompute_closed_trades(store, journal_root, capital_usd=capital_usd)
    else:
        raise PortfolioRulesError(f"via desconocida: {source!r} (usa `trades` o `recomputed`)")
    totals = accumulate(
        ledger.trades, session=session, capital_usd=capital_usd, limits=limits, source=ledger.source
    )
    body = _payload(as_of=as_of, totals=totals, ledger=ledger, limits=limits)
    report_sha256 = SHA256_PREFIX + hashlib.sha256(canonical_text(body).encode("utf-8")).hexdigest()
    report = PortfolioRulesReport(
        session=session,
        as_of=as_of,
        payload={**body, "report_sha256": report_sha256},
        report_sha256=report_sha256,
    )
    if write:
        if reports_dir is None:
            raise PortfolioRulesError("`--reports-dir` es obligatorio para escribir")
        write_report(report, Path(reports_dir))
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Publicacion
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(value: object) -> str:
    """Una cifra, o ``null``: un valor ausente **nunca** se presenta como ``0``."""
    return "`null` (sin historial)" if value is None else f"**{value}** %"


def _conversion_lines(payload: Mapping[str, object]) -> list[str]:
    """Las lineas de la conversion de unidades: la trampa, escrita donde se lee."""
    capital = cast("Mapping[str, object]", payload["capital"])
    return [
        "## Conversion de unidades (`%` del nocional a `%` del capital)",
        "",
        f"- {capital['provenance']}",
        "- `capital_pct = notional_pct x apalancamiento`, con `apalancamiento = nocional / "
        "capital`: el nocional sale del riesgo (`notional = capital x riesgo / stop`, #27 A5), asi "
        "que una salida en el stop pierde `stop_pct` % del nocional, que son el **1 %** del "
        "capital de la regla 2",
        "",
    ]


def render_markdown(report: PortfolioRulesReport) -> str:
    """El informe en prosa: la valla, su convencion, la conversion y la procedencia."""
    payload = report.payload
    accumulation = cast("Mapping[str, object]", payload["accumulation"])
    limits = cast("Mapping[str, object]", payload["limits"])
    source_block = cast("Mapping[str, object]", payload["source"])
    skipped = cast("list[dict[str, object]]", source_block["skipped"])
    trades = cast("list[dict[str, object]]", payload["trades"])
    assessment = cast("list[dict[str, object]]", payload["assessment"])
    windows = cast("list[dict[str, object]]", payload["windows"])
    capital = cast("Mapping[str, object]", payload["capital"])
    honesty = cast("Mapping[str, object]", payload["honesty"])
    lines: list[str] = [
        f"# Valla de cartera del *kill switch* (§12 reglas 3, 4 y 5) — {payload['session']}",
        "",
        f"- **Generado:** `{payload['generated_at']}` · **via:** `{source_block['kind']}` · "
        f"**¿medicion?** `{payload['is_measurement']}`",
        f"- **Estado de la acumulacion:** **`{accumulation['state']}`** — {accumulation['reason']}",
        f"- **Capital de referencia:** `{capital['usd']}` USD "
        "(la base de la conversion a `%` del capital)",
        "",
        "## Las tres ventanas (acumuladas antes de decidir)",
        "",
        "| Regla | Campo | Valor | Limite | Cierre | Cerradas |",
        "|---|---|---|---|---|---|",
        f"| 3 | `daily_pnl_pct` | {_fmt(accumulation['daily_pnl_pct'])} | "
        f"{limits['max_daily_loss_pct']} % | {accumulation['day_bucket']} | "
        f"{accumulation['daily_trades']} |",
        f"| 4 | `weekly_pnl_pct` | {_fmt(accumulation['weekly_pnl_pct'])} | "
        f"{limits['max_weekly_loss_pct']} % | {accumulation['week_bucket']} | "
        f"{accumulation['weekly_trades']} |",
        f"| 5 | `monthly_pnl_pct` | {_fmt(accumulation['monthly_pnl_pct'])} | "
        f"{limits['max_monthly_loss_pct']} % | {accumulation['month_bucket']} | "
        f"{accumulation['monthly_trades']} |",
        "",
        f"- {accumulation['as_of_rule']}",
        f"- **Operaciones cerradas en el diario:** {accumulation['history_trades']} · "
        f"**consideradas (anteriores a la sesion):** {accumulation['considered_trades']} · "
        f"**ultima:** {accumulation['last_closed'] or '`null`'}",
        "",
        "## Veredicto informativo (quien bloquea es el gate)",
        "",
        "| Regla | Valor | Umbral | Veredicto |",
        "|---|---|---|---|",
        *(
            f"| {item['rule']} | "
            f"{item['value_pct'] if item['value_pct'] is not None else '`null`'} "
            f"| {item['threshold_pct']} | **`{item['verdict']}`** |"
            for item in assessment
        ),
        "",
    ]
    lines += _conversion_lines(payload)
    lines += [
        "## Rearme declarado",
        "",
        *(
            f"- **`{window['field']}`** ({window['label']}): cierre = {window['bucket']}; rearme = "
            f"{window['rearm']} (umbral de {window['issue']})"
            for window in windows
        ),
        "",
    ]
    if trades:
        lines += [
            "## Operaciones cerradas consideradas",
            "",
            "| trade_date | `notional_pct` | apalancamiento | `capital_pct` | salida |",
            "|---|---|---|---|---|",
            *(
                f"| {trade['trade_date']} | {trade['notional_pct']} | {trade['leverage']} | "
                f"{trade['capital_pct']} | {trade['exit_reason'] or '`null`'} |"
                for trade in trades
            ),
            "",
        ]
    else:
        lines += [
            "## Operaciones cerradas consideradas",
            "",
            "_Ninguna: el diario no tiene operaciones cerradas (la operacion real es #47 y en "
            "observacion `journal.trades` queda vacio, §19.11)._",
            "",
        ]
    if skipped:
        lines += [
            "## Filas no convertibles (declaradas, nunca rellenadas con 0)",
            "",
            *(f"- `{entry['trade_date']}`: {entry['reason']}" for entry in skipped),
            "",
        ]
    lines += [
        "## Valla de honestidad",
        "",
        f"- **Edge:** {honesty['edge']} · **Fase 2:** {honesty['phase2']}",
        f"- **Carril:** {honesty['lane']}",
        f"- {honesty['note']}",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_report(report: PortfolioRulesReport, reports_dir: Path) -> tuple[Path, Path]:
    """Escribe el par ``.json``/``.md`` y devuelve sus rutas."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{REPORT_PREFIX}_{report.session.isoformat()}"
    json_path = reports_dir / f"{stem}.json"
    markdown_path = reports_dir / f"{stem}.md"
    json_path.write_text(canonical_text(report.payload), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_session(value: str | None) -> date:
    """La sesion declarada, en ISO-8601 (una fecha; el modulo no lee el reloj)."""
    if value is None or not value.strip():
        raise PortfolioRulesError("`--session` es obligatorio (ISO-8601, `AAAA-MM-DD`)")
    try:
        return date.fromisoformat(value.strip())
    except ValueError as error:
        raise PortfolioRulesError(f"`--session` no es una fecha ISO-8601: {value!r}") from error


def _parse_as_of(value: str | None) -> datetime:
    """El instante declarado, en ISO-8601 **con zona** (el modulo no lee el reloj)."""
    if value is None or not value.strip():
        raise PortfolioRulesError("`--as-of` es obligatorio (ISO-8601 con zona horaria)")
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as error:
        raise PortfolioRulesError(f"`--as-of` no es ISO-8601: {value!r}") from error
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise PortfolioRulesError(f"`--as-of` tiene que declarar zona horaria: {value!r}")
    return moment


def main(argv: Sequence[str] | None = None) -> int:
    """Acumula la valla de cartera y publica el informe (#83).

    Codigos de salida: ``0`` = informe emitido; ``2`` = argumentos invalidos o diario/almacen
    ilegibles, con el motivo por ``stderr``.

    ``--source trades`` (por defecto) lee ``journal.trades``, la operacion **real** (#47);
    ``--source recomputed`` recomputa el resultado con la maquina de #45 y exige ``--data-root``.
    """
    parser = argparse.ArgumentParser(
        prog=MODULE, description="Valla de cartera del kill switch (§12 reglas 3, 4 y 5)"
    )
    parser.add_argument("--journal-root", required=True, help="raiz del diario (#39)")
    parser.add_argument("--data-root", type=Path, default=None, help="raiz del almacen")
    parser.add_argument("--session", default=None, help="sesion declarada ISO-8601 (AAAA-MM-DD)")
    parser.add_argument("--as-of", default=None, help="instante declarado ISO-8601 con zona")
    parser.add_argument(
        "--source",
        default=SOURCE_TRADES,
        choices=(SOURCE_TRADES, SOURCE_RECOMPUTED),
        help="`trades` (operacion real, #47) o `recomputed` (maquina de #45)",
    )
    parser.add_argument("--reports-dir", type=Path, default=None, help="directorio de informes")
    args = parser.parse_args(argv)

    try:
        as_of = _parse_as_of(cast("str | None", args.as_of))
        session = _parse_session(cast("str | None", args.session))
    except PortfolioRulesError as error:
        print(f"no se puede acumular la valla de cartera: {error}", file=sys.stderr)
        return 2

    journal_root = Path(cast("str", args.journal_root))
    source = cast("str", args.source)
    data_root = cast("Path | None", args.data_root)
    store: Store | None = None
    if source == SOURCE_RECOMPUTED:
        if data_root is None:
            print(
                "no se puede acumular la valla de cartera: `--data-root` es obligatorio con "
                "`--source recomputed`",
                file=sys.stderr,
            )
            return 2
        store = Store(data_root)
    try:
        report = analyse(
            session=session,
            as_of=as_of,
            journal_root=journal_root,
            store=store,
            source=source,
            reports_dir=cast("Path | None", args.reports_dir),
            write=args.reports_dir is not None,
        )
    except (PortfolioRulesError, InvalidOperation) as error:
        print(f"no se puede acumular la valla de cartera: {error}", file=sys.stderr)
        return 2

    accumulation = cast("Mapping[str, object]", report.payload["accumulation"])
    if args.reports_dir is not None:
        json_path, markdown_path = write_report(report, cast("Path", args.reports_dir))
        print(f"informe de la valla: {json_path} y {markdown_path}")
    print(render_markdown(report))
    print(
        f"estado: {accumulation['state']} — dia: {accumulation['daily_pnl_pct']}, "
        f"semana: {accumulation['weekly_pnl_pct']}, mes: {accumulation['monthly_pnl_pct']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
