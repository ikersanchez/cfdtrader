"""Atribucion por agente de la observacion de la Fase 4 (tarea #162).

**Que entrega.** Rellena ``journal.attribution`` (``tech_stack.md`` §12.5) con una fila por
agente y sesion —``trade_date``, ``agent``, ``would_have_won`` y ``evidence``— a partir de dos
fuentes que ya existen y no se reimplementan:

- el **tilt** que publica el overlay de noticias en ``journal.agent_signals.evidence.tilt``
  (#161): la aritmetica declarada de ``ADJUSTMENT_BY_MAGNITUDE`` sobre el lote, **sin** autoridad
  sobre la decision (§19.19);
- la **serie de paper** de #45 (``analysis.paper_trading``): el resultado de cada recomendacion
  direccional, recomputado desde ``journal.decisions`` mas el almacen con el coste declarado de
  §3.3. Se reutiliza **por import** la misma maquina que #45 y #83: una segunda aritmetica de
  coste o de barreras seria un segundo coste, justo lo que §19.14 prohibe.

**Que significa ``would_have_won``.** El post-mortem de §19.4 pregunta *que agente habria acertado
si el sistema lo hubiera escuchado solo a el*. Para el agente de noticias el juicio es el lado que
declara su tilt (``long``/``short``) frente al resultado de la **sesion** (``open -> close`` de
``raw.market_daily``, el tramo que el sistema captura, §1.1.a). Un tilt ``neutral`` no hace
afirmacion direccional y su ``would_have_won`` queda ``None``: no se inventa un aprobado ni un
suspenso. En ``evidence`` viajan el tilt **verbatim** y el resultado de la sesion; el resultado
del *paper* se publica aparte, como contexto, y **no** define el juicio del agente.

**Que NO hace.** No cambia el gate, ni ``journal.decisions``, ni los prompts, ni da autoridad al
tilt (§19.19). No hace *backfill* de filas historicas: el diario es inmutable (§19.1) y una
identidad ``(trade_date, agent)`` ya escrita **no se toca**; solo se escriben las que faltan. En
particular, las filas de ``agent_signals`` anteriores a #161 no traen ``tilt`` y se declaran
omitidas, no se reescriben.

Sin reloj (las sesiones salen del diario y del almacen) y sin red. La escritura usa la capa de
#39, con esquema cerrado, digest autoconsistente y *append-only* por identidad.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, cast

import duckdb

from cfdtrader.analysis.paper_trading import (
    NEWS_AGENT,
    DecisionOutcome,
    _as_date,  # pyright: ignore[reportPrivateUsage]
    _daily_by_session,  # pyright: ignore[reportPrivateUsage]
    _intraday_by_session,  # pyright: ignore[reportPrivateUsage]
    _outcome,  # pyright: ignore[reportPrivateUsage]
    _recommendations,  # pyright: ignore[reportPrivateUsage]
)
from cfdtrader.data.store import Store, UnknownDatasetError, WriteOutcome
from cfdtrader.journal.decision_log import (
    DecisionLogError,
    Journal,
    read_decisions,
    read_table,
)

__all__ = [
    "AGENT",
    "DOWN",
    "FLAT",
    "LONG",
    "MODULE",
    "NEUTRAL",
    "SHORT",
    "TASK",
    "UP",
    "AttributionError",
    "AttributionReport",
    "AttributionRow",
    "InvalidTiltError",
    "MissingMarketDataError",
    "attribute_sessions",
    "build_attribution",
    "main",
    "record_attribution",
]

MODULE: Final[str] = "cfdtrader.journal.attribution"
TASK: Final[str] = "#162"

#: El agente que publica el tilt del lote (#161); la misma constante que lee #150.
AGENT: Final[str] = NEWS_AGENT

LONG: Final[str] = "long"
SHORT: Final[str] = "short"
NEUTRAL: Final[str] = "neutral"

UP: Final[str] = "up"
DOWN: Final[str] = "down"
FLAT: Final[str] = "flat"


class AttributionError(Exception):
    """Raiz de los errores de la atribucion por agente."""


class InvalidTiltError(AttributionError):
    """El tilt publicado no trae un lado valido (``long``/``short``/``neutral``)."""


class MissingMarketDataError(AttributionError):
    """El almacen no tiene la serie diaria con la que recomputar el resultado de la sesion."""


@dataclass(frozen=True, slots=True)
class AttributionRow:
    """Una fila de ``journal.attribution``: el juicio del agente para una sesion."""

    trade_date: date
    agent: str
    would_have_won: bool | None
    evidence: dict[str, object]

    def payload(self) -> dict[str, object]:
        """La fila con las cuatro columnas del esquema cerrado de #39."""
        return {
            "trade_date": self.trade_date.isoformat(),
            "agent": self.agent,
            "would_have_won": self.would_have_won,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class AttributionReport:
    """Resultado de rellenar la atribucion: lo calculado, lo escrito y lo omitido, con motivo."""

    rows: tuple[AttributionRow, ...]
    written: tuple[AttributionRow, ...]
    existing: tuple[AttributionRow, ...]
    skipped: tuple[dict[str, str], ...]


def _session_result(session: date, bar: Mapping[str, float]) -> tuple[str, float]:
    """Resultado de la sesion (``open -> close``) en ``%`` y su direccion declarada."""
    open_px = float(bar["open"])
    close_px = float(bar["close"])
    if open_px <= 0.0:
        raise AttributionError(
            f"{session.isoformat()}: el `open` diario no es positivo ({open_px!r}); sin el no hay "
            "resultado de sesion que atribuir"
        )
    return_pct = 100.0 * (close_px / open_px - 1.0)
    if return_pct > 0.0:
        direction = UP
    elif return_pct < 0.0:
        direction = DOWN
    else:
        direction = FLAT
    return direction, return_pct


def _would_have_won(side: str, direction: str) -> bool | None:
    """Si el lado que declaro el agente habria acertado en esa sesion.

    ``neutral`` no es una apuesta direccional: devuelve ``None``. Un cierre exactamente plano no
    da la razon a ninguno de los dos lados (``False``), que es lo honesto.
    """
    if side == NEUTRAL:
        return None
    return (side == LONG and direction == UP) or (side == SHORT and direction == DOWN)


def _evidence(
    *,
    tilt: Mapping[str, object],
    side: str,
    direction: str,
    return_pct: float,
    bar: Mapping[str, float],
    outcome: DecisionOutcome,
) -> dict[str, object]:
    """La evidencia publicada: el tilt **verbatim** y el resultado de la sesion."""
    return {
        "tilt": dict(tilt),
        "side": side,
        "session": {
            "open": float(bar["open"]),
            "close": float(bar["close"]),
            "return_pct": return_pct,
            "direction": direction,
            "basis": "open -> close de la sesion ET en raw.market_daily (drift intraday, #6)",
        },
        "paper": {
            "direction": outcome.direction,
            "exit_reason": outcome.exit_reason,
            "net_return_pct": outcome.net_return_pct,
        },
        "note": (
            "el tilt describe el lote (aritmetica de ADJUSTMENT_BY_MAGNITUDE, #161) y no tiene "
            "autoridad sobre la decision (§19.19): esta fila es post-mortem y no cambia el gate, "
            "`journal.decisions` ni los prompts (#162)"
        ),
    }


def attribute_sessions(
    *,
    outcomes: Mapping[date, DecisionOutcome],
    daily: Mapping[date, Mapping[str, float]],
    tilts: Mapping[date, Mapping[str, object]],
    agent: str = AGENT,
) -> tuple[AttributionRow, ...]:
    """Filas de atribucion, ordenadas por sesion, a partir de datos ya leidos.

    Solo entran las sesiones con **las dos** piezas: un tilt del agente y un resultado en la
    serie de paper. Una sesion sin resultado recomputable no se rellena (no hay backfill), y una
    senal sin tilt tampoco: la ausencia se declara en :func:`record_attribution`, no se inventa.
    """
    rows: list[AttributionRow] = []
    for session in sorted(tilts):
        tilt = tilts[session]
        outcome = outcomes.get(session)
        bar = daily.get(session)
        if outcome is None or bar is None:
            continue
        side = tilt.get("side")
        if not isinstance(side, str) or side not in (LONG, SHORT, NEUTRAL):
            raise InvalidTiltError(
                f"{session.isoformat()}: el tilt no declara un lado valido en `side` ({side!r}); "
                f"se esperaba uno de {[LONG, SHORT, NEUTRAL]}"
            )
        direction, return_pct = _session_result(session, bar)
        rows.append(
            AttributionRow(
                trade_date=session,
                agent=agent,
                would_have_won=_would_have_won(side, direction),
                evidence=_evidence(
                    tilt=tilt,
                    side=side,
                    direction=direction,
                    return_pct=return_pct,
                    bar=bar,
                    outcome=outcome,
                ),
            )
        )
    return tuple(rows)


def _paper_series(
    store: Store, journal_root: Path
) -> tuple[dict[date, DecisionOutcome], dict[date, dict[str, float]]]:
    """La serie de paper de #45 y el OHLC diario con el que se mide la sesion."""
    try:
        daily = _daily_by_session(store)
    except (UnknownDatasetError, duckdb.Error) as error:
        raise MissingMarketDataError(
            f"no se puede leer `raw.market_daily` en {store.root}: {error}. La atribucion "
            "recomputa el resultado de la sesion y no lo inventa"
        ) from error
    intraday = _intraday_by_session(store)
    decisions = read_decisions(Journal(journal_root))
    emitted, _counts = _recommendations(decisions)
    outcomes: dict[date, DecisionOutcome] = {}
    for row in emitted:
        outcome = _outcome(row, daily=daily, intraday=intraday)
        if outcome is not None:
            outcomes[outcome.trade_date] = outcome
    return outcomes, daily


def _news_signals(journal_root: Path) -> dict[date, Mapping[str, object] | None]:
    """Sesiones con senal del agente de noticias; ``None`` cuando la fila no trae tilt."""
    signals: dict[date, Mapping[str, object] | None] = {}
    for row in read_table(Journal(journal_root), "agent_signals"):
        if str(row.get("agent")) != AGENT:
            continue
        session = _as_date(row.get("trade_date"))
        if session is None:
            continue
        evidence = row.get("evidence")
        tilt: Mapping[str, object] | None = None
        if isinstance(evidence, Mapping):
            candidate = cast("Mapping[str, object]", evidence).get("tilt")
            if isinstance(candidate, Mapping):
                tilt = cast("Mapping[str, object]", candidate)
        signals[session] = tilt
    return signals


def _skipped(
    *,
    signals: Mapping[date, Mapping[str, object] | None],
    tilts: Mapping[date, Mapping[str, object]],
    outcomes: Mapping[date, DecisionOutcome],
) -> tuple[dict[str, str], ...]:
    """Las sesiones que **no** producen fila, cada una con su motivo declarado."""
    entries: list[dict[str, str]] = []
    for session in sorted(set(tilts) - set(outcomes)):
        entries.append(
            {
                "trade_date": session.isoformat(),
                "reason": (
                    "sin serie de paper: la sesion no tiene una recomendacion direccional "
                    "recomputable desde `journal.decisions` + almacen (#45)"
                ),
            }
        )
    for session in sorted({day for day, tilt in signals.items() if tilt is None} & set(outcomes)):
        entries.append(
            {
                "trade_date": session.isoformat(),
                "reason": (
                    "la senal del agente no trae `tilt`: fila anterior a #161 o overlay "
                    "desactivado; no se reescribe (el diario es inmutable)"
                ),
            }
        )
    for session in sorted(set(outcomes) - set(signals)):
        entries.append(
            {
                "trade_date": session.isoformat(),
                "reason": "no hay senal del agente `news` para esa sesion",
            }
        )
    entries.sort(key=lambda entry: (entry["trade_date"], entry["reason"]))
    return tuple(entries)


def _collect(
    *, store: Store, journal_root: Path, agent: str
) -> tuple[tuple[AttributionRow, ...], tuple[dict[str, str], ...]]:
    """Calcula las filas y las omisiones, sin escribir nada."""
    outcomes, daily = _paper_series(store, journal_root)
    signals = _news_signals(journal_root)
    tilts = {day: tilt for day, tilt in signals.items() if tilt is not None}
    rows = attribute_sessions(outcomes=outcomes, daily=daily, tilts=tilts, agent=agent)
    return rows, _skipped(signals=signals, tilts=tilts, outcomes=outcomes)


def build_attribution(
    *, store: Store, journal_root: Path | str, agent: str = AGENT
) -> tuple[AttributionRow, ...]:
    """Las filas que **faltan** por escribir, sin tocar el diario."""
    rows, _skipped_entries = _collect(store=store, journal_root=Path(journal_root), agent=agent)
    return rows


def record_attribution(
    *,
    store: Store,
    journal_root: Path | str,
    write: bool = True,
    agent: str = AGENT,
) -> AttributionReport:
    """Rellena ``journal.attribution`` con las filas nuevas y **no** reescribe ninguna existente.

    La identidad es ``(trade_date, agent)``: si ya esta en el diario se deja intacta, aunque su
    contenido difiera. Es la frontera declarada de #162: el diario es *append-only* y el
    post-mortem no se re-baseliniza hacia atras.
    """
    root = Path(journal_root)
    rows, skipped = _collect(store=store, journal_root=root, agent=agent)
    journal = Journal(root)
    existing_sessions: set[date] = set()
    for record in read_table(journal, "attribution"):
        if str(record.get("agent")) != agent:
            continue
        session = _as_date(record.get("trade_date"))
        if session is not None:
            existing_sessions.add(session)
    written: list[AttributionRow] = []
    existing: list[AttributionRow] = []
    for row in rows:
        if row.trade_date in existing_sessions:
            existing.append(row)
            continue
        if not write:
            continue
        outcome = journal.write("attribution", row.payload())
        if outcome is WriteOutcome.CREATED:
            written.append(row)
        else:
            existing.append(row)
    return AttributionReport(
        rows=rows,
        written=tuple(written),
        existing=tuple(existing),
        skipped=skipped,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI manual: rellena la atribucion por agente cuando exista la serie de paper."""
    parser = argparse.ArgumentParser(
        description="Atribucion por agente de la Fase 4 (tarea #162): tilt (#161) + paper (#45)."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="raiz del almacen")
    parser.add_argument("--journal-root", type=Path, required=True, help="raiz del diario (#39)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="calcula las filas y las omisiones, pero no escribe en el diario",
    )
    args = parser.parse_args(argv)
    try:
        report = record_attribution(
            store=Store(args.data_root),
            journal_root=args.journal_root,
            write=not args.dry_run,
        )
    except (AttributionError, DecisionLogError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        f"atribucion {AGENT}: {len(report.rows)} filas candidatas, "
        f"{len(report.written)} escritas, {len(report.existing)} ya en el diario, "
        f"{len(report.skipped)} omitidas"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    raise SystemExit(main())
