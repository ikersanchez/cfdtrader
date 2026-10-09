"""Puesta en produccion con tamano minimo (tarea #47).

**Que entrega.** El **procedimiento de operacion manual** de un dia completo, declarado y
verificable: las reglas de la puesta en produccion, la politica de **tamano minimo**, la **tarjeta
de operacion** del dia (que se hace y a que hora) y los **comandos exactos** que la ejecutan. Es el
ultimo eslabon del carril A: el sistema lleva desde #39 registrando decisiones y desde #45
observando, y aqui se declara **como se opera con dinero real** sin arriesgar mas de lo que el
sistema ha demostrado.

**Lo que la puesta en produccion NO cambia.** El veredicto medido de la Fase 2 es ``fail`` sobre la
base neta con el coste declarado (`plan.md` §19.13) y `phase2_ready = false`: **no hay edge
demostrado** y el valor esperado neto medido es **negativo**. Poner en produccion con tamano minimo
**no** es una afirmacion de *edge* ni reabre `§11.6`: es operar **por debajo** de lo que el sistema
ha demostrado, con las reglas duras activas, para poder auditar la ejecucion real. El tamano minimo
es la unica escala que corresponde a esa evidencia.

**Las piezas, declaradas.**

- **Solo tier A.** La regla 10 de §12: los tiers B y C se registran y **no** se operan. El gate ya
  autoriza unicamente el tier A; esta puesta lo **verifica** y no lo suaviza.
- **Tamano minimo.** El nocional del gate es un **techo** —sale del riesgo del 1 % del capital,
  regla 2—, nunca un objetivo: se opera el **menor tamano admisible** del broker y **nunca** por
  encima de ese techo. Si el minimo del broker ya lo supera, **no se opera**.
- **La valla de cartera esta activa.** Las reglas 3, 4 y 5 (perdida -2 % diaria, -5 % semanal,
  -10 % mensual) se evaluan con el P&L realizado que acumula #83 desde ``journal.trades``.
- **El registro es el de #84.** Precios y costes **efectivos** en ``journal.trades`` (§12.5), con
  las diez columnas del esquema cerrado y `closed_by_close = false` si se incumplio el cierre.
- **Sin overnight y sin automatismo.** El cierre a las 16:00 ET es una obligacion (§12 regla 16);
  la ejecucion es manual y a demanda: **no** hay *scheduler*, ni alarma, ni API del broker.
- **Una sola operacion por sesion** (regla 1) y **sin ampliar perdedoras** (regla 12).

**El dia, probado.** `tests/test_run_daily.py` ensaya el dia completo sobre un almacen y un diario
sinteticos: la pista, el billete, el registro real, la valla leyendolo y el bloqueo del dia
siguiente cuando la perdida semanal acumulada alcanza su limite. Ese ensayo **es** el artefacto de
aceptacion de la tarea.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Final, cast

from cfdtrader.backtest.engine import Direction
from cfdtrader.decision.gate import GateStatus
from cfdtrader.delivery.bracket import MODULE as BRACKET_MODULE
from cfdtrader.delivery.bracket import (
    BracketError,
    render_ticket,
    ticket_from_decision_row,
)
from cfdtrader.journal.decision_log import DecisionLogError, read_decision

__all__ = [
    "AUTHORIZED_TIERS",
    "CHECKLIST",
    "COMMANDS",
    "HONESTY_FENCE",
    "MODULE",
    "PRODUCTION_RULES",
    "SIZE_POLICY",
    "TASK",
    "OperatingCard",
    "ProductionError",
    "commands_for",
    "main",
    "operating_card",
    "render_card",
]

MODULE: Final[str] = "cfdtrader.delivery.production"
TASK: Final[str] = "#47"

#: El unico tier autorizado a operar (regla 10 de §12; el techo lo declara #60).
AUTHORIZED_TIERS: Final[tuple[str, ...]] = ("A",)

#: La politica de tamano de la puesta en produccion: minimo del broker, techo del gate.
SIZE_POLICY: Final[str] = (
    "se opera el **menor tamano admisible** del broker y **nunca** por encima del nocional que "
    "publica el gate (techo: 1 % del capital en el peor caso, regla 2). Si el minimo del broker "
    "supera ese techo, **no se opera**"
)


#: Las reglas de la puesta en produccion, con la regla que las respalda.
PRODUCTION_RULES: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "solo_tier_a",
        "rule": "§12 regla 10",
        "statement": "solo se opera una senal de tier A; B y C se registran y no se operan",
    },
    {
        "id": "tamano_minimo",
        "rule": "§12 regla 2",
        "statement": SIZE_POLICY,
    },
    {
        "id": "valla_de_cartera",
        "rule": "§12 reglas 3, 4 y 5",
        "statement": (
            "el P&L realizado se acumula con `analysis/portfolio_rules` (#83) y el gate bloquea la "
            "sesion en cuanto una ventana alcanza su limite"
        ),
    },
    {
        "id": "cierre_obligatorio",
        "rule": "§12 reglas 6 y 16",
        "statement": (
            "sin overnight: si la posicion sigue abierta a las 16:00 ET se cierra al abrir la "
            "sesion siguiente y se registra con `closed_by_close = false` (#84)"
        ),
    },
    {
        "id": "una_operacion",
        "rule": "§12 regla 1",
        "statement": "una sola operacion por sesion, sin excepciones",
    },
    {
        "id": "sin_automatismo",
        "rule": "§13",
        "statement": (
            "ejecucion manual y a demanda: no hay *scheduler*, ni alarma, ni API del broker; la "
            "verificacion de las 15:45 ET la garantiza el operador"
        ),
    },
)

#: El dia, paso a paso, con su ancla en ET (`plan.md` §13).
CHECKLIST: Final[tuple[str, ...]] = (
    "08:00 — ingesta y regeneracion en orden (`_docs/runbook.md` §1 y §2)",
    "08:45 — captura del pre-mercado del ES (#141) y camino diario (#110): la pista",
    "09:00 — se lee la pista: direccion, tier, nocional, stop y objetivo; sin tier A no se opera",
    "09:20-09:30 — decision del operador: se coloca la entrada o se anula con motivo (§13.2)",
    "09:30 — entrada en la subasta; con el relleno, el billete da los dos precios (#84)",
    "09:30 — inmediatamente: las **dos** patas del *bracket* en el broker (las dos o ninguna)",
    "15:45 — verificacion **manual**: sigue abierta y con las dos patas en pie (no hay alarma)",
    "16:00 — cierre obligatorio: si no ha saltado el *bracket*, se cierra a mano",
    "16:15 — registro real en `journal.trades`: precios, coste efectivo y motivo de salida (#84)",
    "16:30 — la valla de cartera se comprueba con #83; el `manifest` declara las cerradas",
)

#: Los comandos del dia, declarados una sola vez (el informe diario imprime los tres primeros).
COMMANDS: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "camino_diario",
        "when": "08:45 ET",
        "command": (
            "uv run python -m cfdtrader.delivery.run_daily --as-of <iso> --variant-id <id> "
            "--journal-root journal --runs-root runs --git-commit <sha> "
            "--premarket-bars <fichero>"
        ),
    },
    {
        "id": "billete",
        "when": "09:30 ET (con el relleno)",
        "command": (
            f"uv run python -m {BRACKET_MODULE} --journal-root journal --session <sesion> "
            "--entry-px <relleno>"
        ),
    },
    {
        "id": "registro",
        "when": "16:15 ET (con los datos reales)",
        "command": (
            f"uv run python -m {BRACKET_MODULE} --journal-root journal --session <sesion> --record "
            "--entry-px <relleno> --exit-px <salida> --exit-reason <stop|target|close> "
            "--costs-pct <efectivo> --entry-time <iso> --exit-time <iso>"
        ),
    },
    {
        "id": "valla",
        "when": "16:30 ET",
        "command": (
            "uv run python -m cfdtrader.analysis.portfolio_rules --journal-root journal "
            "--session <sesion> --as-of <iso> --reports-dir data/derived/reports"
        ),
    },
    {
        "id": "veredicto_paper",
        "when": "al cerrar la ventana (>= 30 sesiones)",
        "command": (
            "uv run python -m cfdtrader.analysis.paper_trading --data-root data "
            "--journal-root journal --reference-artifact <pipeline_backtest>.json --as-of <iso>"
        ),
    },
)

#: La valla de honestidad, la misma que viaja en el informe diario.
HONESTY_FENCE: Final[tuple[str, ...]] = (
    "no hay edge demostrado: Fase 2 `fail` en la base neta (§19.13), `phase2_ready = false`",
    "el valor esperado neto **medido** sale negativo: operar de verdad con tamano minimo es una "
    "decision del propietario, no una conclusion del sistema",
    "carril A (apoyo a la decision, ejecucion manual): el carril B sigue bloqueado (§11.6/§19.7)",
    "el coste es el **declarado** de §3.3: medir el *slippage* real es #62",
)


class ProductionError(Exception):
    """Raiz de los errores de la puesta en produccion (tarjeta o argumentos invalidos)."""


@dataclass(frozen=True, slots=True)
class OperatingCard:
    """La tarjeta de una sesion: se opera o no, con que geometria y con que comandos."""

    session: date
    operable: bool
    reason: str
    direction: str | None
    tier: str | None
    notional_usd: Decimal | None
    ticket: dict[str, object] | None
    commands: tuple[dict[str, str], ...]

    def payload(self) -> dict[str, object]:
        """La tarjeta como *mapping* JSON-serializable."""
        return {
            "task": TASK,
            "session": self.session.isoformat(),
            "operable": self.operable,
            "reason": self.reason,
            "direction": self.direction,
            "tier": self.tier,
            "notional_usd": None if self.notional_usd is None else str(self.notional_usd),
            "authorized_tiers": list(AUTHORIZED_TIERS),
            "size_policy": SIZE_POLICY,
            "ticket": self.ticket,
            "commands": [dict(command) for command in self.commands],
            "rules": [dict(rule) for rule in PRODUCTION_RULES],
            "checklist": list(CHECKLIST),
            "honesty": list(HONESTY_FENCE),
        }


def commands_for(
    *, journal_root: str = "journal", session: date | None = None
) -> tuple[dict[str, str], ...]:
    """Los comandos del dia con la sesion y la raiz del diario ya puestas."""
    label = "<sesion>" if session is None else session.isoformat()
    return tuple(
        {
            **command,
            "command": command["command"]
            .replace("<sesion>", label)
            .replace("journal", journal_root),
        }
        for command in COMMANDS
    )


def operating_card(journal_root: Path | str, session: date) -> OperatingCard:
    """La tarjeta de la sesion: lee la pista del diario y decide si se opera, **sin inventar nada**.

    Se opera sii hay una fila de ``journal.decisions`` con estado ``recommendation``, direccion
    operada y tier autorizado. Cualquier otra cosa —sin fila, ``nothing``, un "no se" o un tier B/C—
    es **no operar**, con su motivo: el protocolo declarado de la regla 10.
    """
    commands = commands_for(journal_root=str(journal_root), session=session)
    tickets: dict[str, object] | None = None
    try:
        row = read_decision(journal_root, session)
    except DecisionLogError as error:
        return OperatingCard(
            session=session,
            operable=False,
            reason=f"sin pista registrada para la sesion: {error}",
            direction=None,
            tier=None,
            notional_usd=None,
            ticket=None,
            commands=commands,
        )
    direction = None if row.get("direction") is None else str(row["direction"])
    tier = None if row.get("tier") is None else str(row["tier"])
    status = None if row.get("status") is None else str(row["status"])
    notional = None
    blockers = _why_not_operable(status=status, direction=direction, tier=tier)
    if blockers:
        return OperatingCard(
            session=session,
            operable=False,
            reason=blockers[0],
            direction=direction,
            tier=tier,
            notional_usd=None,
            ticket=None,
            commands=commands,
        )
    try:
        billete = ticket_from_decision_row(session, row)
    except BracketError as error:
        return OperatingCard(
            session=session,
            operable=False,
            reason=f"sin billete: {error}",
            direction=direction,
            tier=tier,
            notional_usd=None,
            ticket=None,
            commands=commands,
        )
    notional = billete.notional_usd
    tickets = billete.payload()
    tickets["rendered"] = render_ticket(billete)
    return OperatingCard(
        session=session,
        operable=True,
        reason=(
            "pista direccional de tier A: se opera el **menor tamano admisible** del broker, nunca "
            "por encima del nocional del gate"
        ),
        direction=direction,
        tier=tier,
        notional_usd=notional,
        ticket=tickets,
        commands=commands,
    )


def _why_not_operable(*, status: str | None, direction: str | None, tier: str | None) -> list[str]:
    """Los motivos declarados por los que la sesion **no** se opera (lista vacia = se opera)."""
    if status != GateStatus.RECOMMENDATION.value:
        return [f"el estado del diario es `{status}`: solo se opera `recommendation`"]
    if direction not in (Direction.LONG.value, Direction.SHORT.value):
        return [f"la direccion del diario es `{direction}`: no hay nada que operar"]
    if tier not in AUTHORIZED_TIERS:
        return [f"el tier es `{tier}` y solo se opera {AUTHORIZED_TIERS} (regla 10)"]
    return []


def render_card(card: OperatingCard) -> str:
    """La tarjeta en texto: se opera o no, la geometria, los pasos y los comandos del dia."""
    lines: list[str] = [
        f"# Tarjeta de operacion (#47) — {card.session.isoformat()}",
        "",
        f"- **Se opera:** {'**si**' if card.operable else '**no**'} — {card.reason}",
    ]
    if card.operable:
        lines += [
            f"- **Direccion:** `{card.direction}` · **tier:** `{card.tier}` · "
            f"**techo de nocional:** {card.notional_usd} USD",
            f"- **Tamano:** {SIZE_POLICY}",
        ]
    lines += ["", "## Reglas de la puesta en produccion", ""]
    lines.extend(
        f"- **{rule['id']}** ({rule['rule']}): {rule['statement']}" for rule in PRODUCTION_RULES
    )
    lines += ["", "## El dia, paso a paso (ET)", ""]
    lines.extend(f"{index}. {step}" for index, step in enumerate(CHECKLIST, start=1))
    lines += ["", "## Comandos", ""]
    lines.extend(f"- **{command['when']}:** `{command['command']}`" for command in card.commands)
    lines += ["", "## Valla de honestidad", ""]
    lines.extend(f"- {note}" for note in HONESTY_FENCE)
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Imprime la tarjeta de operacion de una sesion (#47).

    Codigos de salida: ``0`` = tarjeta emitida (se opere o no: «no operar» es un resultado legitimo
    y con su motivo); ``2`` = argumentos invalidos, con el motivo por ``stderr``.
    """
    parser = argparse.ArgumentParser(
        prog=MODULE, description="Tarjeta de operacion de un dia de la puesta en produccion"
    )
    parser.add_argument("--journal-root", required=True, help="raiz del diario (#39)")
    parser.add_argument("--session", required=True, help="sesion declarada ISO-8601 (AAAA-MM-DD)")
    args = parser.parse_args(argv)

    try:
        session = date.fromisoformat(cast("str", args.session))
    except ValueError:
        print(f"error: `--session` no es ISO-8601: {args.session!r}", file=sys.stderr)
        return 2
    card = operating_card(Path(cast("str", args.journal_root)), session)
    print(render_card(card))
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
