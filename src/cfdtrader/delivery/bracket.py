"""Orden *bracket* contra el broker real y registro de la ejecucion (tarea #84).

**Que entrega.** El mecanismo operativo de la regla 16 de ``plan.md`` §12 —«el cierre a las 16:00 ET
no es una intencion, es una obligacion»— con el broker **ya declarado** (#59, Revolut, cuenta de
CFD): el **billete** de ejecucion de una sesion (entrada, stop, objetivo, nocional y el orden exacto
de los pasos con su ancla temporal), el **protocolo declarado** de fallo (que se hace si el broker
no acepta el *bracket* o acepta una sola pata) y el **registro** de la ejecucion real en
``journal.trades`` (§12.5), incluido el incumplimiento de las 16:00 ET.

**La conversion de ``%`` a precio no se reimplementa.** El gate ya traduce su geometria con
``GateOutput.to_engine_decision(entry_px=...)`` (#27, A23): alli ``stop_pct``/``target_pct`` se
convierten en ``stop_px``/``target_px`` con el ``open`` de la subasta y la geometria
(``stop < open < target`` en largo y la espejo en corto) se cumple **por construccion**. Este modulo
**consume** esa conversion; el gate ya declaraba que no tener precio era suyo (``stop_px = None``,
«la conversion, #84»).

**Sin reloj, sin red y sin automatismo.** La ejecucion es **manual y a demanda** (§13): no hay
*scheduler*, ni alarma, ni llamada al broker. El modulo no habla con Revolut, no lee el reloj y no
coloca nada: compone el billete, lo imprime y —cuando el operador le da los datos **reales**—
escribe la fila del diario. Que la verificacion de las 15:45 ET no tenga alarma es una **decision
de diseno consciente** (§12 regla 16), no una carencia que este modulo deba tapar.

**Unidades del registro (contrato con #83).** ``journal.trades.pnl_pct`` se escribe en ``%`` del
**nocional** y **neto** —el recorrido de precio de la entrada a la salida, con signo segun la
direccion, **menos** el coste efectivo del viaje— y ``costs_pct`` publica el coste ya restado. La
valla de cartera (`analysis/portfolio_rules.py`, #83) lee esa magnitud y la convierte a ``%`` del
capital con el apalancamiento: escrito asi, la regla 3/4/5 consume la cifra sin una segunda resta de
costes. ``closed_by_close`` es el indicador de la regla 16: **true** si la posicion **no** paso la
noche (la cerro el *bracket* o la salida de las 16:00 ET) y **false** si la paso (incumplimiento,
que se registra y se revisa).

**Que NO hace este modulo** (fronteras declaradas, con su issue)

- **No** decide: la direccion, el stop, el objetivo y el nocional vienen del gate (#27/#60); aqui no
  se recalcula la geometria ni se inventa un precio de entrada.
- **No** habla con el broker: no hay API de Revolut en el proyecto. Si algun dia la hubiera, el
  billete es lo que cambiaria de forma, no la decision.
- **No** es #47: no activa la produccion ni fija el tamano minimo; es el mecanismo que #47 usa.
- **No** es #83: no acumula P&L; escribe la fila que #83 lee.
- **No** mide el *slippage* (#62): registra los costes **efectivos** que el operador lee del broker,
  sin convertirlos en una medicion del supuesto declarado.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Final, cast

from cfdtrader.analysis.paper_trading import EXIT_CLOSE, EXIT_STOP, EXIT_TARGET
from cfdtrader.analysis.portfolio_rules import CAPITAL_USD
from cfdtrader.backtest.engine import Direction
from cfdtrader.decision.gate import GateOutput, GateStatus
from cfdtrader.journal.decision_log import (
    DecisionLogError,
    Journal,
    read_decision,
    read_record,
)

__all__ = [
    "ANCHORS",
    "BRACKET_LEGS",
    "EXIT_REASONS",
    "INCIDENTS",
    "MODULE",
    "REPORT_DOES_NOT_DO",
    "TASK",
    "BracketError",
    "BracketTicket",
    "ExecutionFacts",
    "execution_pnl_pct",
    "main",
    "read_trade",
    "record_trade",
    "render_ticket",
    "ticket",
]

MODULE: Final[str] = "cfdtrader.delivery.bracket"
TASK: Final[str] = "#84"

#: Las dos patas del *bracket* (stop y objetivo): la regla 16 las exige **las dos o ninguna**.
BRACKET_LEGS: Final[tuple[str, ...]] = ("stop", "objetivo")

#: El vocabulario de ``exit_reason`` de ``journal.trades``: el **mismo** que #45 usa al recomputar
#: (`analysis/paper_trading.py`), para que las dos vias hablen de lo mismo sin un segundo idioma.
EXIT_REASONS: Final[tuple[str, ...]] = (EXIT_TARGET, EXIT_STOP, EXIT_CLOSE)


#: El orden de los pasos con su ancla temporal (`plan.md` §13, todo en ET). La ejecucion es manual:
#: esto describe **que** se hace y **cuando**, no dispara nada.
ANCHORS: Final[tuple[dict[str, str], ...]] = (
    {
        "et": "09:00",
        "madrid": "15:00",
        "step": "informe",
        "action": "llega la pista: direccion, nocional, stop y objetivo en %",
    },
    {
        "et": "09:20-09:30",
        "madrid": "15:20-15:30",
        "step": "deadline",
        "action": (
            "decision del operador (**tu decides**): si se opera, la entrada tiene que estar "
            "puesta **antes** de la subasta. Anular la recomendacion se registra con motivo (§13.2)"
        ),
    },
    {
        "et": "09:30",
        "madrid": "15:30",
        "step": "entrada",
        "action": "entrada **en la subasta de apertura**; se anota el precio de relleno real",
    },
    {
        "et": "09:30 (inmediatamente)",
        "madrid": "15:30 (inmediatamente)",
        "step": "bracket",
        "action": (
            "con el relleno delante, colocar **las dos patas** (stop y objetivo) en el broker: las "
            "dos o ninguna, porque una sola deja la posicion sin red (`una_sola_pata`)"
        ),
    },
    {
        "et": "15:45",
        "madrid": "21:45",
        "step": "verificacion",
        "action": (
            "comprobacion **manual**: sigue abierta y con las dos patas en pie. **No hay alarma "
            "del sistema**: la responsabilidad es del operador (§12 regla 16, por diseno)"
        ),
    },
    {
        "et": "16:00",
        "madrid": "22:00",
        "step": "cierre",
        "action": (
            "si el *bracket* no ha saltado, **cerrar a mano**. Dejar pasar la noche es el "
            "incumplimiento `paso_la_noche`: gap mas financiacion (§12 regla 16)"
        ),
    },
    {
        "et": "16:15",
        "madrid": "22:15",
        "step": "registro",
        "action": (
            "escribir la fila real: precios, coste efectivo, motivo de salida y si paso la noche"
        ),
    },
)

#: El protocolo declarado de fallo: cada modo dice **que se hace** y **que se registra y donde**.
INCIDENTS: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "broker_rechaza_el_bracket",
        "when": "el broker no acepta la orden *bracket* al entrar",
        "action": "**no se opera**: sin cierre garantizado no hay intradia puro (§12 regla 16)",
        "record": (
            "no se escribe `journal.trades` (no hubo operacion): la incidencia va en la traza "
            "de la sesion (`ops.run_log`, #43) y en el diario del operador"
        ),
    },
    {
        "id": "una_sola_pata",
        "when": "el broker acepta solo una de las dos patas (stop u objetivo)",
        "action": (
            "**cerrar de inmediato** por lo que acepto el broker y no insistir esa sesion: la "
            "posicion se quedo sin red"
        ),
        "record": (
            "se escribe `journal.trades` con el precio de salida **real** y su `exit_reason`: la "
            "incidencia no se maquilla, la cierra el dato"
        ),
    },
    {
        "id": "sin_relleno",
        "when": "la subasta abre sin rellenar la entrada",
        "action": "**no se opera**: no hay posicion que cubrir",
        "record": "no hay fila de `journal.trades`: no existio operacion",
    },
    {
        "id": "paso_la_noche",
        "when": "a las 16:00 ET la posicion sigue abierta (no se cerro a mano ni salto el bracket)",
        "action": (
            "cerrar en cuanto abra el mercado siguiente, asumiendo el *gap* y la financiacion "
            "(§12 regla 16)"
        ),
        "record": (
            "`journal.trades` con `closed_by_close = false` y el coste efectivo **con** la "
            "financiacion: la noche se registra, no se redondea a cero"
        ),
    },
    {
        "id": "salto_el_bracket",
        "when": "el objetivo o el stop saltaron antes de las 16:00 ET: la posicion ya esta cerrada",
        "action": "nada: el *bracket* hizo su trabajo",
        "record": (
            "`journal.trades` con `exit_reason` = `stop` o `target` y `closed_by_close = true` "
            "(no paso la noche)"
        ),
    },
)

#: Lo que este modulo **no** hace, con su issue.
REPORT_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_habla_con_el_broker",
        "statement": (
            "no hay API del broker en el proyecto: compone el billete y registra lo que el "
            "operador lee, no coloca nada"
        ),
    },
    {
        "id": "no_decide_la_geometria",
        "statement": (
            "direccion, stop, objetivo y nocional vienen del gate (#27/#60); aqui no se recalcula "
            "ni se inventa un precio de entrada"
        ),
    },
)


class BracketError(Exception):
    """Raiz de los errores del mecanismo del *bracket* (billete o registro invalidos)."""


def _barrier_prices(
    *, direction: str, stop_pct: Decimal, target_pct: Decimal, entry_px: float
) -> tuple[float, float]:
    """``stop_px`` y ``target_px`` desde ``%`` y el relleno: la misma geometria de #27 (A23).

    Es la aritmetica que el gate ya aplica en ``to_engine_decision(entry_px=...)`` para su propio
    ``Decision``. Aqui se repite **porque el billete tiene que existir sin un ``GateOutput``** (el
    operador lo saca de la fila del diario, que no guarda el objeto entero), y para que la
    repeticion no derive en dos aritmeticas hay un test que la **coteja** con la del gate: si #27
    cambia su convencion, ese test cae.
    """
    stop_fraction = float(stop_pct) / 100.0
    target_fraction = float(target_pct) / 100.0
    if direction == Direction.LONG.value:
        return entry_px * (1.0 - stop_fraction), entry_px * (1.0 + target_fraction)
    if direction == Direction.SHORT.value:
        return entry_px * (1.0 + stop_fraction), entry_px * (1.0 - target_fraction)
    raise BracketError(f"`direction` tiene que ser long o short: {direction!r}")


@dataclass(frozen=True, slots=True)
class BracketTicket:
    """El billete de ejecucion de una sesion: la geometria del gate, en ``%`` y en precio."""

    session: date
    direction: str
    tier: str
    notional_usd: Decimal
    leverage_implied: Decimal | None
    stop_pct: Decimal
    target_pct: Decimal
    entry_px: float | None
    stop_px: float | None
    target_px: float | None
    gate_sha256: str | None
    source: str

    def payload(self) -> dict[str, object]:
        """El billete como *mapping* (lo que se imprime y lo que se audita mas tarde)."""
        return {
            "task": TASK,
            "session": self.session.isoformat(),
            "direction": self.direction,
            "tier": self.tier,
            "notional_usd": str(self.notional_usd),
            "leverage_implied": (
                None if self.leverage_implied is None else str(self.leverage_implied)
            ),
            "stop_pct": str(self.stop_pct),
            "target_pct": str(self.target_pct),
            "entry_px": self.entry_px,
            "stop_px": self.stop_px,
            "target_px": self.target_px,
            "gate_sha256": self.gate_sha256,
            "gate_sha256_note": (
                None
                if self.gate_sha256 is not None
                else (
                    "el esquema de `journal.decisions` no guarda `gate_sha256`: si el billete sale "
                    "de la fila del diario, el digest vive en `ops/<sesion>/manifest.json`"
                )
            ),
            "source": self.source,
            "legs": list(BRACKET_LEGS),
        }


def ticket(
    *,
    session: date,
    direction: str,
    tier: str,
    notional_usd: Decimal,
    stop_pct: Decimal,
    target_pct: Decimal | None,
    leverage_implied: Decimal | None = None,
    entry_px: float | None = None,
    gate_sha256: str | None = None,
    source: str = "declarado",
) -> BracketTicket:
    """Compone el billete de la sesion; **sin objetivo no hay billete** (regla 16).

    La regla 16 exige el cierre con **las dos patas** (stop y objetivo): si el objetivo no viene
    declarado, el protocolo declarado es **no operar** y aqui se lanza el error tipado en vez de
    fabricar una pata. Sin ``entry_px`` los dos precios van a ``None``: la entrada es el relleno de
    la subasta y **no se inventa**, asi que el billete publica los ``%`` y la instruccion de
    multiplicarlos por el relleno.
    """
    if direction not in (Direction.LONG.value, Direction.SHORT.value):
        raise BracketError(
            f"la regla 16 solo tiene billete para una direccion operada; llego {direction!r}"
        )
    if target_pct is None:
        raise BracketError(
            "la regla 16 exige objetivo y stop (las dos o ninguna): sin objetivo, **no se opera** "
            "(bloqueo `bracket_sin_objetivo` del gate)"
        )
    if notional_usd <= 0:
        raise BracketError(f"`notional_usd` tiene que ser positivo: {notional_usd!r}")
    if stop_pct <= 0:
        raise BracketError(f"`stop_pct` tiene que ser positivo: {stop_pct!r}")
    if entry_px is None or entry_px <= 0.0:
        stop_px, target_px = None, None
    else:
        stop_px, target_px = _barrier_prices(
            direction=direction, stop_pct=stop_pct, target_pct=target_pct, entry_px=entry_px
        )
    return BracketTicket(
        session=session,
        direction=direction,
        tier=tier,
        notional_usd=notional_usd,
        leverage_implied=leverage_implied,
        stop_pct=stop_pct,
        target_pct=target_pct,
        entry_px=entry_px,
        stop_px=stop_px,
        target_px=target_px,
        gate_sha256=gate_sha256,
        source=source,
    )


def ticket_from_output(output: GateOutput, *, entry_px: float | None = None) -> BracketTicket:
    """El billete a partir de la salida del gate (el camino **en proceso**, para #47).

    Solo vale para una recomendacion direccional operada: en cualquier otro estado —``NOTHING``,
    ``no_recommendation_*`` o ``error``— el protocolo es **no operar**, y aqui se declara el error
    tipado en vez de fabricar un billete.
    """
    direction = output.direction
    if (
        output.status is not GateStatus.RECOMMENDATION
        or direction is None
        or direction is Direction.NOTHING
    ):
        raise BracketError(
            "la salida del gate no es una recomendacion direccional "
            f"(status={output.status.value}): el protocolo declarado es **no operar**"
        )
    notional = output.notional_usd
    if not isinstance(notional, Decimal) or notional <= 0:
        raise BracketError(
            "la salida direccional del gate tiene que traer `notional_usd` positivo (A19); llego "
            f"{notional!r}"
        )
    return ticket(
        session=output.session,
        direction=direction.value,
        tier=output.tier,
        notional_usd=notional,
        stop_pct=output.stop_pct,
        target_pct=output.target_pct,
        leverage_implied=output.leverage_implied,
        entry_px=entry_px,
        gate_sha256=output.gate_sha256,
        source="gate_sha256",
    )


@dataclass(frozen=True, slots=True)
class ExecutionFacts:
    """Los datos **reales** de una sesion operada, tal como los lee el operador del broker."""

    session: date
    direction: str
    entry_px: float
    exit_px: float
    notional: Decimal
    costs_pct: Decimal
    exit_reason: str
    entry_time: datetime
    exit_time: datetime
    closed_by_close: bool

    def payload(self) -> dict[str, object]:
        """La fila de ``journal.trades`` (§12.5): las **diez** columnas del esquema cerrado."""
        return {
            "trade_date": self.session.isoformat(),
            "entry_price": self.entry_px,
            "exit_price": self.exit_px,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "notional": float(self.notional),
            "pnl_pct": float(execution_pnl_pct(self)),
            "costs_pct": float(self.costs_pct),
            "exit_reason": self.exit_reason,
            "closed_by_close": self.closed_by_close,
        }


def execution_pnl_pct(facts: ExecutionFacts) -> Decimal:
    """El retorno **neto** en ``%`` del **nocional**: recorrido de precio menos el coste efectivo.

    Es exactamente la magnitud que la valla de cartera (#83) lee de ``journal.trades.pnl_pct`` y
    convierte a ``%`` del capital. Se calcula **aqui**, en un solo sitio, para que no exista un
    segundo ``pnl_pct`` escrito a mano.
    """
    if facts.entry_px <= 0.0:
        raise BracketError(f"`entry_px` tiene que ser positivo: {facts.entry_px!r}")
    if facts.direction == Direction.LONG.value:
        gross = (facts.exit_px - facts.entry_px) / facts.entry_px
    elif facts.direction == Direction.SHORT.value:
        gross = (facts.entry_px - facts.exit_px) / facts.entry_px
    else:
        raise BracketError(f"`direction` tiene que ser long o short: {facts.direction!r}")
    return Decimal("100") * Decimal(repr(gross)) - facts.costs_pct


def render_ticket(billete: BracketTicket) -> str:
    """El billete en texto: la geometria, el orden de los pasos y el protocolo de fallo."""
    payload = billete.payload()
    prices = (
        "no publicado: la entrada es el relleno de la subasta y no se inventa"
        if payload["entry_px"] is None
        else (
            f"relleno {payload['entry_px']} -> stop **{payload['stop_px']}** / "
            f"objetivo **{payload['target_px']}**"
        )
    )
    lines: list[str] = [
        f"# Billete de ejecucion (#84) — {payload['session']}",
        "",
        f"- **Direccion:** `{payload['direction']}` · **tier:** `{payload['tier']}` · "
        f"**nocional:** {payload['notional_usd']} USD "
        f"(apalancamiento `{payload['leverage_implied']}`)",
        f"- **Stop:** {payload['stop_pct']} % · **objetivo:** {payload['target_pct']} % "
        "(las dos patas: regla 16)",
        f"- **Precios:** {prices}",
        f"- **Origen de la geometria:** `{payload['source']}` · "
        f"**`gate_sha256`:** {payload['gate_sha256'] or '`null` (ver nota)'}",
        "",
        "## Pasos (todo en ET; la ejecucion es manual)",
        "",
    ]
    lines.extend(
        f"{index}. **{anchor['et']}** ({anchor['madrid']} Madrid) — `{anchor['step']}`: "
        f"{anchor['action']}"
        for index, anchor in enumerate(ANCHORS, start=1)
    )
    lines += [
        "",
        "## Protocolo de fallo (declarado)",
        "",
        "| Incidencia | Cuando | Que se hace | Que se registra |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| `{incident['id']}` | {incident['when']} | {incident['action']} | {incident['record']} |"
        for incident in INCIDENTS
    )
    lines += ["", "## Lo que este modulo no hace", ""]
    lines.extend(f"- {entry['statement']}" for entry in REPORT_DOES_NOT_DO)
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Registro de la ejecucion real en `journal.trades` (§12.5)
# ─────────────────────────────────────────────────────────────────────────────
def record_trade(journal_root: Path | str, facts: ExecutionFacts) -> Path:
    """Escribe la fila **real** de la sesion y devuelve su ruta.

    El diario es *append-only* e inmutable (§19.1): si la sesion ya tiene fila con **otro**
    contenido, el error tipado del diario (`JournalRewriteError`) sube tal cual, sin sobrescribir
    nada. La validacion de aqui es de forma —motivo de salida del vocabulario, direccion y tiempos
    coherentes— y **no** re-deriva la decision: los numeros los da el operador.
    """
    if facts.direction not in (Direction.LONG.value, Direction.SHORT.value):
        raise BracketError(f"`direction` tiene que ser long o short: {facts.direction!r}")
    if facts.exit_reason not in EXIT_REASONS:
        raise BracketError(
            f"`exit_reason` tiene que ser uno de {EXIT_REASONS}: llego {facts.exit_reason!r}"
        )
    if facts.costs_pct < 0:
        raise BracketError(f"`costs_pct` no puede ser negativo: {facts.costs_pct!r}")
    if facts.exit_time < facts.entry_time:
        raise BracketError("`exit_time` no puede ser anterior a `entry_time`")
    journal = Journal(Path(journal_root))
    try:
        journal.write("trades", facts.payload())
    except DecisionLogError as error:
        raise BracketError(f"no se puede registrar la operacion en el diario: {error}") from error
    return journal.path("trades", facts.session.isoformat())


def read_trade(journal_root: Path | str, session: date) -> dict[str, object]:
    """La fila del diario de esa sesion (payload sin el digest), o error tipado si no existe."""
    try:
        return read_record(Journal(Path(journal_root)), "trades", session.isoformat())
    except DecisionLogError as error:
        raise BracketError(str(error)) from error


def _dec(value: object, *, field_name: str) -> Decimal:
    """Un porcentaje del diario como ``Decimal`` exacto (nunca por el binario del ``float``)."""
    if value is None:
        raise BracketError(f"la fila del diario no trae `{field_name}`: sin el no hay billete")
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise BracketError(f"`{field_name}` no es un numero legible: {value!r}") from error


def _geometry_from_journal(session: date, row: dict[str, object]) -> dict[str, object]:
    """La geometria declarada de la fila de ``journal.decisions``, sin inventar campos que no trae.

    El nocional no lo guarda el esquema de decisiones: sale del apalancamiento que el gate publico
    (`leverage_implied`) por el capital declarado, que es la misma relacion que el gate aplico
    (`notional = capital x riesgo / stop`, #27 A5). ``--notional`` lo puede sobrescribir.
    """
    leverage = row.get("leverage_implied")
    leverage_dec = None if leverage is None else _dec(leverage, field_name="leverage_implied")
    notional = CAPITAL_USD * leverage_dec if leverage_dec is not None else CAPITAL_USD
    target = row.get("target_pct")
    return {
        "session": session,
        "direction": None if row.get("direction") is None else str(row["direction"]),
        "tier": "?" if row.get("tier") is None else str(row["tier"]),
        "notional_usd": notional,
        "leverage_implied": leverage_dec,
        "stop_pct": _dec(row.get("stop_pct"), field_name="stop_pct"),
        "target_pct": None if target is None else _dec(target, field_name="target_pct"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: Sequence[str] | None = None) -> int:
    """Imprime el billete de la sesion o registra su ejecucion real (#84).

    Codigos de salida: ``0`` = billete emitido o fila registrada; ``2`` = argumentos invalidos,
    diario ilegible o una sesion **sin recomendacion direccional** (el protocolo declarado es
    *no operar*), con el motivo por ``stderr``.
    """
    parser = argparse.ArgumentParser(
        prog=MODULE, description="Billete de ejecucion del bracket y registro de la operacion"
    )
    parser.add_argument("--journal-root", required=True, help="raiz del diario (#39)")
    parser.add_argument("--session", required=True, help="sesion declarada ISO-8601 (AAAA-MM-DD)")
    parser.add_argument(
        "--entry-px",
        type=float,
        default=None,
        help="precio de relleno declarado (el de la subasta)",
    )
    parser.add_argument(
        "--notional", default=None, help="nocional declarado en USD (si no, el del gate)"
    )
    parser.add_argument(
        "--direction", default=None, help="direccion real (por defecto, la de la fila)"
    )
    parser.add_argument(
        "--record", action="store_true", help="registra la ejecucion en `journal.trades`"
    )
    parser.add_argument("--exit-px", type=float, default=None, help="precio de salida real")
    parser.add_argument(
        "--exit-reason", default=None, choices=EXIT_REASONS, help="motivo de salida real"
    )
    parser.add_argument("--costs-pct", default=None, help="coste efectivo de ida y vuelta, en %")
    parser.add_argument("--entry-time", default=None, help="instante de entrada ISO-8601")
    parser.add_argument("--exit-time", default=None, help="instante de salida ISO-8601")
    parser.add_argument(
        "--overnight",
        action="store_true",
        help="la posicion paso la noche: `closed_by_close = false` (incumplimiento, regla 16)",
    )
    args = parser.parse_args(argv)

    try:
        session = date.fromisoformat(cast("str", args.session))
    except ValueError:
        print(f"error: `--session` no es ISO-8601: {args.session!r}", file=sys.stderr)
        return 2
    journal_root = Path(cast("str", args.journal_root))
    try:
        row = read_decision(journal_root, session)
        geometry = _geometry_from_journal(session, row)
    except (DecisionLogError, BracketError) as error:
        print(f"error: no se puede leer la sesion del diario: {error}", file=sys.stderr)
        return 2
    if args.notional is not None:
        try:
            geometry["notional_usd"] = _dec(args.notional, field_name="--notional")
        except BracketError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    direction = cast("str | None", args.direction if args.direction else geometry["direction"])

    if args.record:
        return _record_from_cli(
            args, geometry=geometry, direction=direction, journal_root=journal_root
        )

    try:
        billete = ticket(
            session=session,
            direction=cast("str", direction) or "",
            tier=cast("str", geometry["tier"]),
            notional_usd=cast("Decimal", geometry["notional_usd"]),
            stop_pct=cast("Decimal", geometry["stop_pct"]),
            target_pct=cast("Decimal | None", geometry["target_pct"]),
            leverage_implied=cast("Decimal | None", geometry["leverage_implied"]),
            entry_px=cast("float | None", args.entry_px),
            source=f"journal.decisions/{session.isoformat()}.json",
        )
    except BracketError as error:
        print(f"no se opera: {error}", file=sys.stderr)
        return 2
    print(render_ticket(billete))
    return 0


def _record_from_cli(
    args: argparse.Namespace,
    *,
    geometry: dict[str, object],
    direction: str | None,
    journal_root: Path,
) -> int:
    """El registro de la ejecucion real: cada dato lo declara el operador y ninguno se inventa."""
    missing = [
        name
        for name, value in (
            ("--entry-px", args.entry_px),
            ("--exit-px", args.exit_px),
            ("--exit-reason", args.exit_reason),
            ("--costs-pct", args.costs_pct),
            ("--entry-time", args.entry_time),
            ("--exit-time", args.exit_time),
        )
        if value is None
    ]
    if direction is None:
        missing.append("--direction (la fila no trae direccion)")
    if missing:
        print(f"error: faltan datos reales: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        facts = ExecutionFacts(
            session=cast("date", geometry["session"]),
            direction=cast("str", direction),
            entry_px=float(cast("float", args.entry_px)),
            exit_px=float(cast("float", args.exit_px)),
            notional=cast("Decimal", geometry["notional_usd"]),
            costs_pct=_dec(args.costs_pct, field_name="--costs-pct"),
            exit_reason=cast("str", args.exit_reason),
            entry_time=datetime.fromisoformat(cast("str", args.entry_time)),
            exit_time=datetime.fromisoformat(cast("str", args.exit_time)),
            closed_by_close=not args.overnight,
        )
        path = record_trade(journal_root, facts)
    except (BracketError, ValueError) as error:
        print(f"error: no se puede registrar la operacion: {error}", file=sys.stderr)
        return 2
    if geometry["direction"] is not None and geometry["direction"] != direction:
        print(
            f"aviso: la direccion real ({direction}) no es la registrada en la pista "
            f"({geometry['direction']}): la anulacion va aparte (#40, `journal.overrides`)",
            file=sys.stderr,
        )
    print(f"operacion registrada: {path}")
    print(
        f"pnl_pct (neto, % del nocional): {execution_pnl_pct(facts)} · "
        f"closed_by_close: {facts.closed_by_close}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
