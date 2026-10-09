"""Informe de *paper trading* de la Fase 4 (tarea #45).

Arranca la observacion de la Fase 4 —carril A de ``plan.md`` §19.7, ejecucion **manual**— y entrega
el **arnes que la puerta de §16 declara**: el resultado de cada recomendacion **se recomputa**, no
se lee, desde lo persistido —la fila de ``journal.decisions`` (``trade_date``, ``direction``,
``stop_pct``, ``target_pct``, ``cost_pct``) mas el almacen de mercado de esa sesion— porque
``journal.trades`` (§12.5) esta reservado a la operacion **real** (#47) y en observacion queda
**vacio**.

La puerta se aplica **al pie de la letra** (§16, pre-registrada en #130, no se mueve):

- **Estadistico**: media del retorno neto por sesion de las recomendaciones **emitidas**
  (``status = recommendation`` y ``direction ∈ {long, short}``); las de ``nothing`` y las "no se"
  **no** entran y su recuento se **publica**.
- **Referencia**: media y **sigma del backtest**, leidas del artefacto de #28
  (``arms.<brazo>.declared_series``): la observacion decide sobre el **coste declarado** (§19.12),
  asi que la serie comparable es la **declarada**. La sigma **no** se recalcula con la muestra del
  *paper*.
- **Regla**: hay divergencia si ``|media_paper - media_backtest| > 2 * sigma_backtest / sqrt(N)``.
- **Muestra minima**: con ``N < 30`` el veredicto es **``not_evaluable``**, nunca un aprobado por
  silencio.
- **Coste**: el **declarado** de §3.3 (tarea #11), el mismo que el gate cobro en la decision; **no**
  se define un segundo coste.

**Comparacion «con / sin overlay» (#150, §19.9).** La serie que la puerta mide es la **publicada**,
que bajo §19.19 se emite **sin** overlay. El informe publica **aparte** la variante **«con
overlay»** —leida de `journal.agent_signals` (`agent = "news"`, la escribe `run_daily`)—: las
**mismas** sesiones, con la direccion que el gate habria dado con el veto (regla 20 ⇒ `nothing`) y
el ajuste ±10 pp del `NewsAgent`, mas las sesiones donde discrepan. Es **informativa**: no es una
segunda puerta.

**Valla de honestidad** (viaja en el informe): esto **no** es una afirmacion de *edge*; la Fase 2
sigue `not_evaluable`/`fail` con `phase2_ready = false`, el carril B sigue bloqueado, la ejecucion
es manual y `§11.6` **no** se altera.

Sin reloj (el instante entra por ``--as-of``) y sin red; salida **determinista byte a byte**:
``report_sha256`` es el sha256 del texto canonico de #13 sobre el payload sin la clave del hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Final, cast

import duckdb
import polars as pl

from cfdtrader.backtest.engine import canonical_text
from cfdtrader.data.calendar import EASTERN
from cfdtrader.data.store import Store, UnknownDatasetError
from cfdtrader.journal.decision_log import Journal, read_decisions

__all__ = [
    "MIN_SESSIONS",
    "MODULE",
    "REFERENCE_ARM",
    "REPORT_PREFIX",
    "SIGMA_MULTIPLE",
    "TASK",
    "PaperTradingError",
    "PaperTradingReport",
    "analyse",
    "main",
    "render_markdown",
    "write_report",
]

MODULE: Final[str] = "cfdtrader.analysis.paper_trading"
TASK: Final[str] = "#45"
ANALYSIS: Final[str] = "cfdtrader.analysis.paper_trading"
SERIES_ID: Final[str] = "^GSPC"
INTERVAL: Final[str] = "5m"
REPORT_PREFIX: Final[str] = "paper_trading"
SHA256_PREFIX: Final[str] = "sha256:"

#: Brazo del backtest de referencia (la observacion decide sobre coste declarado, §19.12).
REFERENCE_ARM: Final[str] = "coste_declarado"
#: Muestra minima de §16: por debajo, `not_evaluable`.
MIN_SESSIONS: Final[int] = 30
#: Regla de §16: ``|media_paper - media_backtest| > SIGMA_MULTIPLE * sigma / sqrt(N)``.
SIGMA_MULTIPLE: Final[float] = 2.0

LONG: Final[str] = "long"
SHORT: Final[str] = "short"
STATUS_RECOMMENDATION: Final[str] = "recommendation"
EXIT_TARGET: Final[str] = "target"
EXIT_STOP: Final[str] = "stop"
EXIT_CLOSE: Final[str] = "close"

STATE_DIVERGES: Final[str] = "diverges"
STATE_IN_RANGE: Final[str] = "in_range"
STATE_NOT_EVALUABLE: Final[str] = "not_evaluable"

#: Agente del overlay de noticias en `journal.agent_signals` (§12.5). Lo escribe el camino diario
#: (`run_daily.NEWS_OVERLAY_AGENT`) como la variante «con overlay» (§19.19, #149); aqui se **lee**
#: para comparar «con / sin overlay» (§19.9, #150).
NEWS_AGENT: Final[str] = "news"


class PaperTradingError(Exception):
    """Raiz de los errores del informe de *paper trading*."""


class MissingReferenceArtifactError(PaperTradingError):
    """Falta el artefacto de #28 que declara la referencia (media y sigma del backtest)."""


class MalformedReferenceError(PaperTradingError):
    """El artefacto de referencia no trae la serie declarada del brazo de referencia."""


@dataclass(frozen=True, slots=True)
class DecisionOutcome:
    """Resultado **recomputado** de una recomendacion, con el coste declarado de #11."""

    trade_date: date
    direction: str
    entry_px: float
    stop_pct: float
    target_pct: float
    exit_px: float
    exit_reason: str
    gross_return_pct: float
    cost_pct: float
    net_return_pct: float


@dataclass(frozen=True, slots=True)
class PaperTradingReport:
    """Informe publicado (`.json` + `.md`)."""

    as_of: datetime
    report_date: str
    payload: dict[str, object]
    report_sha256: str


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def _literal(value: str) -> str:
    """Literal SQL seguro: la serie viene del registro declarado del proyecto."""
    return "'" + value.replace("'", "''") + "'"


def _as_float(value: object) -> float | None:
    """Convierte un numero del diario (Decimal-como-cadena, float o None) a ``float``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(Decimal(value))
        except (InvalidOperation, ValueError):
            return None
    return None


def _as_date(value: object) -> date | None:
    """Normaliza el ``trade_date`` del diario (``date`` o ISO-8601 en texto) a ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_std(values: Sequence[float]) -> float:
    """Desviacion tipica muestral (``ddof = 1``); ``0.0`` si la serie tiene menos de dos puntos."""
    if len(values) < 2:
        return 0.0
    average = _mean(values)
    variance = math.fsum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _recommendations(
    decisions: Sequence[Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], dict[str, int]]:
    """Filas con recomendacion direccional, y los recuentos de §16.

    Entran las filas con ``status = recommendation`` y ``direction ∈ {long, short}``; las de
    ``nothing`` y las de los estados "no se" se cuentan **aparte** (no entran en la media).
    """
    emitted: list[Mapping[str, object]] = []
    counts = {"recommendation": 0, "nothing": 0, "no_recommendation": 0, "error": 0}
    for row in decisions:
        status = str(row.get("status", ""))
        direction = row.get("direction")
        if status != STATUS_RECOMMENDATION:
            counts["error" if status == "error" else "no_recommendation"] += 1
            continue
        counts["recommendation"] += 1
        if direction in (LONG, SHORT):
            emitted.append(row)
        else:
            counts["nothing"] += 1
    return emitted, counts


def _overlay_directions(journal_root: Path | str) -> dict[date, str]:
    """La direccion de la variante «con overlay» por sesion (#150), leida de `agent_signals`.

    El camino diario escribe una fila con ``agent = NEWS_AGENT`` (§19.19, #149): la direccion que el
    gate habria dado **con** overlay viaja en ``evidence`` (``nothing`` si veto). Una fila sin
    ``evidence`` o sin direccion se **ignora**: no se inventa una variante que no se registro.
    """
    directions: dict[date, str] = {}
    for row in Journal(Path(journal_root)).read_table("agent_signals"):
        if str(row.get("agent")) != NEWS_AGENT:
            continue
        trade_date = _as_date(row.get("trade_date"))
        evidence = row.get("evidence")
        if trade_date is None or not isinstance(evidence, Mapping):
            continue
        direction = cast("Mapping[str, object]", evidence).get("direction")
        if isinstance(direction, str):
            directions[trade_date] = direction
    return directions


def _daily_by_session(store: Store) -> dict[date, dict[str, float]]:
    """OHLC diario por sesion ET de la serie del CFD (``store.sql``, no ``read_pit``)."""
    query = (
        "SELECT as_of, open, high, low, close FROM raw.market_daily "  # noqa: S608
        f"WHERE series_id = {_literal(SERIES_ID)} "
        "AND open IS NOT NULL AND close IS NOT NULL ORDER BY as_of"
    )
    frame = (
        store.sql(query)
        .with_columns(
            pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("session")
        )
        .sort("as_of")
        .unique(subset=["session"], keep="first", maintain_order=True)
    )
    daily: dict[date, dict[str, float]] = {}
    for row in frame.iter_rows(named=True):
        daily[cast("date", row["session"])] = {
            "open": float(cast("float", row["open"])),
            "high": float(cast("float", row["high"])),
            "low": float(cast("float", row["low"])),
            "close": float(cast("float", row["close"])),
        }
    return daily


def _intraday_by_session(store: Store) -> dict[date, list[tuple[float, float]]]:
    """Camino intradia (``high``, ``low``) por sesion ET; si falta, hay respaldo diario."""
    query = (
        "SELECT as_of, high, low FROM raw.market_intraday "  # noqa: S608
        f"WHERE series_id = {_literal(SERIES_ID)} AND interval = {_literal(INTERVAL)} "
        "AND high IS NOT NULL AND low IS NOT NULL ORDER BY as_of"
    )
    try:
        frame = store.sql(query)
    except (UnknownDatasetError, duckdb.Error):
        return {}
    bars: dict[date, list[tuple[float, float]]] = {}
    for row in frame.iter_rows(named=True):
        moment = cast("datetime", row["as_of"]).astimezone(EASTERN).date()
        bars.setdefault(moment, []).append(
            (float(cast("float", row["high"])), float(cast("float", row["low"])))
        )
    return bars


def _resolve(
    *, direction: str, bars: Sequence[tuple[float, float]], upper: float, lower: float, close: float
) -> tuple[str, float]:
    """Orden de las barreras del **decisor** para la direccion observada (asimetricas, #60).

    ``upper`` es la barrera de **arriba** (a favor del largo, en contra del corto) y ``lower`` la de
    abajo. Si una barra toca las dos, el orden es desconocido y se resuelve de forma **adversa** (el
    stop), igual que el etiquetador de #10. Sin toque, la posicion sale al cierre de la sesion.
    """
    for high, low in bars:
        hit_upper = high >= upper
        hit_lower = low <= lower
        if direction == LONG:
            if hit_upper and hit_lower:
                return EXIT_STOP, lower
            if hit_upper:
                return EXIT_TARGET, upper
            if hit_lower:
                return EXIT_STOP, lower
        elif hit_upper and hit_lower:
            return EXIT_STOP, upper
        elif hit_lower:
            return EXIT_TARGET, lower
        elif hit_upper:
            return EXIT_STOP, upper
    return EXIT_CLOSE, close


def _outcome(
    row: Mapping[str, object],
    *,
    daily: Mapping[date, dict[str, float]],
    intraday: Mapping[date, Sequence[tuple[float, float]]],
) -> DecisionOutcome | None:
    """Recomputa el resultado de una recomendacion, o ``None`` si faltan datos de la sesion."""
    raw_date = row.get("trade_date")
    direction = row.get("direction")
    trade_date = _as_date(raw_date)
    if trade_date is None or direction not in (LONG, SHORT):
        return None
    bar = daily.get(trade_date)
    if bar is None:
        return None
    entry_px = bar["open"]
    stop_pct = _as_float(row.get("stop_pct"))
    target_pct = _as_float(row.get("target_pct"))
    cost_pct = _as_float(row.get("cost_pct"))
    if entry_px <= 0.0 or stop_pct is None or target_pct is None or cost_pct is None:
        return None
    if direction == LONG:
        upper, lower = entry_px * (1.0 + target_pct), entry_px * (1.0 - stop_pct)
    else:
        upper, lower = entry_px * (1.0 + stop_pct), entry_px * (1.0 - target_pct)
    bars = intraday.get(trade_date)
    touches: Sequence[tuple[float, float]] = bars if bars else [(bar["high"], bar["low"])]
    reason, exit_px = _resolve(
        direction=direction, bars=touches, upper=upper, lower=lower, close=bar["close"]
    )
    if direction == LONG:
        gross = (exit_px - entry_px) / entry_px
    else:
        gross = (entry_px - exit_px) / entry_px
    return DecisionOutcome(
        trade_date=trade_date,
        direction=direction,
        entry_px=entry_px,
        stop_pct=stop_pct,
        target_pct=target_pct,
        exit_px=exit_px,
        exit_reason=reason,
        gross_return_pct=100.0 * gross,
        cost_pct=100.0 * cost_pct,
        net_return_pct=100.0 * gross - 100.0 * cost_pct,
    )


def _reference_series(payload: Mapping[str, object]) -> list[float]:
    """Serie **declarada** del brazo de referencia del artefacto de #28, en ``%`` del nocional."""
    arms = payload.get("arms")
    if not isinstance(arms, Mapping):
        raise MalformedReferenceError("el artefacto de referencia no trae `arms`")
    arm = cast("Mapping[str, object]", arms).get(REFERENCE_ARM)
    if not isinstance(arm, Mapping):
        raise MalformedReferenceError(f"el artefacto no trae el brazo `{REFERENCE_ARM}`")
    series = cast("Mapping[str, object]", arm).get("declared_series")
    if not isinstance(series, Mapping):
        raise MalformedReferenceError(f"`arms.{REFERENCE_ARM}` no trae `declared_series`")
    values = cast("Mapping[str, object]", series).get("series_pct")
    if not isinstance(values, list):
        raise MalformedReferenceError("`declared_series` no trae `series_pct`")
    return [float(cast("float", value)) for value in cast("list[object]", values)]


def _reference_block(reference_path: Path) -> dict[str, object]:
    """Bloque de referencia de §16: media y sigma de la serie **declarada** del backtest."""
    if not reference_path.is_file():
        raise MissingReferenceArtifactError(
            f"no existe el artefacto de referencia del backtest: {reference_path}"
        )
    payload = cast("Mapping[str, object]", json.loads(reference_path.read_text(encoding="utf-8")))
    series = _reference_series(payload)
    return {
        "artifact": reference_path.name,
        "arm": REFERENCE_ARM,
        "series": f"arms.{REFERENCE_ARM}.declared_series.series_pct",
        "units": "% del nocional (puntos porcentuales), una entrada por sesion de test",
        "n": len(series),
        "mean_pct": _mean(series) if series else None,
        "sigma_pct": _sample_std(series) if series else None,
        "note": (
            "la sigma **no** se recalcula con la muestra del *paper*: hacerlo seria redefinir la "
            "puerta a posteriori (§16)"
        ),
    }


def _verdict(
    *, n_sessions: int, mean_paper_pct: float | None, reference: Mapping[str, object]
) -> dict[str, object]:
    """La regla de §16, al pie de la letra."""
    mean_ref = reference.get("mean_pct")
    sigma_ref = reference.get("sigma_pct")
    block: dict[str, object] = {
        "n_sessions": n_sessions,
        "mean_paper_pct": mean_paper_pct,
        "mean_backtest_pct": mean_ref,
        "sigma_backtest_pct": sigma_ref,
        "sigma_multiple": SIGMA_MULTIPLE,
        "min_sessions": MIN_SESSIONS,
        "threshold_pct": None,
        "gap_pct": None,
        "diverges": None,
    }
    if n_sessions < MIN_SESSIONS or mean_paper_pct is None:
        block["state"] = STATE_NOT_EVALUABLE
        block["reason"] = (
            f"`N = {n_sessions} < {MIN_SESSIONS}`: muestra por debajo del minimo de §16, "
            "`not_evaluable`, nunca un aprobado por silencio"
        )
        return block
    if not isinstance(mean_ref, (int, float)) or not isinstance(sigma_ref, (int, float)):
        block["state"] = STATE_NOT_EVALUABLE
        block["reason"] = "la referencia no declara media y sigma del backtest (revisar #108)"
        return block
    threshold = SIGMA_MULTIPLE * float(sigma_ref) / math.sqrt(n_sessions)
    gap = abs(mean_paper_pct - float(mean_ref))
    diverges = gap > threshold
    block["threshold_pct"] = threshold
    block["gap_pct"] = gap
    block["diverges"] = diverges
    block["state"] = STATE_DIVERGES if diverges else STATE_IN_RANGE
    block["reason"] = (
        f"`|{mean_paper_pct!r} - {float(mean_ref)!r}| = {gap!r} "
        f"{'>' if diverges else '<='} {SIGMA_MULTIPLE} * {float(sigma_ref)!r} / sqrt({n_sessions}) "
        f"= {threshold!r}`: "
        + ("divergencia (obliga a auditar antes de operar)" if diverges else "sin divergencia")
    )
    return block


def _outcome_payload(outcome: DecisionOutcome) -> dict[str, object]:
    """La fila publicada de una recomendacion: lo recomputado, sesion a sesion."""
    return {
        "trade_date": outcome.trade_date.isoformat(),
        "direction": outcome.direction,
        "entry_px": outcome.entry_px,
        "stop_pct": outcome.stop_pct,
        "target_pct": outcome.target_pct,
        "exit_px": outcome.exit_px,
        "exit_reason": outcome.exit_reason,
        "gross_return_pct": outcome.gross_return_pct,
        "cost_pct": outcome.cost_pct,
        "net_return_pct": outcome.net_return_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analisis
# ─────────────────────────────────────────────────────────────────────────────
def _payload(
    *,
    as_of: datetime,
    reference: Mapping[str, object],
    counts: Mapping[str, int],
    sessions: Sequence[dict[str, object]],
    missing: int,
    mean_paper_pct: float | None,
    verdict: Mapping[str, object],
    with_overlay: Mapping[str, object],
) -> dict[str, object]:
    """El payload publicable (sin `report_sha256`; lo anade `analyse`)."""
    return {
        "analysis": ANALYSIS,
        "task": TASK,
        "generated_at": as_of.isoformat(),
        "phase": "Fase 4 — Operacion (carril A, ejecucion manual)",
        "basis": "declared_cost",
        "is_measurement": False,
        "is_validation": False,
        "gate_rule": (
            "§16: |media_paper - media_backtest| > 2 * sigma_backtest / sqrt(N); "
            f"N < {MIN_SESSIONS} => not_evaluable"
        ),
        "series_id": SERIES_ID,
        "cost": {
            "basis": "declarado (§3.3, #11)",
            "note": (
                "el resultado se recomputa con el **coste declarado**, el mismo que el gate "
                "cobro en la decision; no se define un segundo coste"
            ),
        },
        "reference": dict(reference),
        "sample": {
            "n_sessions": len(sessions),
            "counts": dict(counts),
            "missing_outcomes": missing,
            "note": (
                "solo entran las recomendaciones direccionales (`recommendation` y "
                "`direction ∈ {long, short}`); las de `nothing` y las sin recomendacion se cuentan "
                "aparte y **no** entran en la media (§16)"
            ),
        },
        "sessions": list(sessions),
        "metrics": {"mean_net_return_pct": mean_paper_pct, "verdict": dict(verdict)},
        "series_published": {
            "basis": "sin_overlay",
            "note": (
                "la recomendacion **publicada** se emite **sin** overlay (§19.19); la puerta de "
                "§16 decide sobre esta serie"
            ),
        },
        "with_overlay": dict(with_overlay),
        "honesty": {
            "edge": "no demostrado",
            "phase2": "`not_evaluable`/`fail`; `phase2_ready = false`",
            "lane": "carril A (asistente de decision, ejecucion manual)",
            "note": (
                "esto **no** es una afirmacion de *edge* ni una validacion: la observacion de la "
                "Fase 4. `§11.6` **no** se altera y el carril B sigue bloqueado"
            ),
        },
        "does_not_do": [
            {
                "id": "no_opera_ni_registra_trades",
                "statement": (
                    "no coloca ninguna orden ni escribe `journal.trades`: es #47; en observacion "
                    "el resultado se **recomputa** desde `journal.decisions` + almacen"
                ),
            },
            {
                "id": "no_mide_slippage",
                "statement": "el coste es el declarado de §3.3 (#11); medir el *slippage* es #62",
            },
            {
                "id": "no_recalcula_la_sigma",
                "statement": (
                    "la sigma de la regla es la del **backtest** (§16); recalcularla con la "
                    "muestra del *paper* redefiniria la puerta a posteriori"
                ),
            },
        ],
        "follow_ups": [
            {"issue": "#62", "topic": "medir el *slippage* real (hoy el coste es el declarado)"},
            {"issue": "#47", "topic": "operacion real: `journal.trades` y el cierre de sesion"},
            {"issue": "#70", "topic": "liston B de primera clase (referencia de comparacion)"},
        ],
    }


def analyse(
    *,
    store: Store,
    journal_root: Path | str,
    reference_artifact: Path | str,
    as_of: datetime,
    reports_dir: Path | str | None = None,
    write: bool = True,
) -> PaperTradingReport:
    """Recomputa la observacion de la Fase 4 y, si ``write``, publica el informe."""
    decisions = read_decisions(Journal(Path(journal_root)))
    emitted, counts = _recommendations(decisions)
    daily = _daily_by_session(store)
    intraday = _intraday_by_session(store)
    outcomes = [
        outcome
        for outcome in (_outcome(row, daily=daily, intraday=intraday) for row in emitted)
        if outcome is not None
    ]
    missing = len(emitted) - len(outcomes)
    mean_paper = _mean([item.net_return_pct for item in outcomes]) if outcomes else None
    # #150: la variante «con overlay» (§19.9). Se recomputa con la **misma** maquina que la serie
    # publicada, cambiando solo la direccion por la que el gate habria dado **con** overlay.
    directions = _overlay_directions(journal_root)
    overlay_rows: list[Mapping[str, object]] = []
    overrides: list[dict[str, object]] = []
    for row in emitted:
        session = _as_date(row.get("trade_date"))
        if session is None:
            continue
        published = str(row.get("direction", ""))
        # Sin senal del overlay (p. ej. `disabled_*`) la variante **coincide** con la publicada: asi
        # las dos series cubren las mismas sesiones y son comparables (§19.9).
        variant = directions.get(session, published)
        overlay_rows.append({**row, "direction": variant})
        if variant != published:
            overrides.append(
                {"trade_date": session.isoformat(), "published": published, "with_overlay": variant}
            )
    overlay_outcomes = [
        outcome
        for outcome in (_outcome(row, daily=daily, intraday=intraday) for row in overlay_rows)
        if outcome is not None
    ]
    mean_overlay = (
        _mean([item.net_return_pct for item in overlay_outcomes]) if overlay_outcomes else None
    )
    with_overlay: dict[str, object] = {
        "series": "con_overlay",
        "n_sessions": len(overlay_outcomes),
        "mean_net_return_pct": mean_overlay,
        "sessions": [_outcome_payload(item) for item in overlay_outcomes],
        "overrides": overrides,
        "note": (
            "variante «con overlay» (§19.9): el **veto** (regla 20 ⇒ `nothing`) y el ajuste "
            "±10 pp del `NewsAgent`. Es **informativa**: la puerta de §16 decide sobre la serie "
            "publicada, que es la **sin overlay** (§19.19)"
        ),
    }
    reference = _reference_block(Path(reference_artifact))
    verdict = _verdict(n_sessions=len(outcomes), mean_paper_pct=mean_paper, reference=reference)
    body = _payload(
        as_of=as_of,
        reference=reference,
        counts=counts,
        sessions=[_outcome_payload(item) for item in outcomes],
        missing=missing,
        mean_paper_pct=mean_paper,
        verdict=verdict,
        with_overlay=with_overlay,
    )
    report_sha256 = SHA256_PREFIX + hashlib.sha256(canonical_text(body).encode("utf-8")).hexdigest()
    payload = {**body, "report_sha256": report_sha256}
    report = PaperTradingReport(
        as_of=as_of,
        report_date=as_of.date().isoformat(),
        payload=payload,
        report_sha256=report_sha256,
    )
    if write:
        if reports_dir is None:
            raise PaperTradingError("`--reports-dir` es obligatorio para escribir")
        write_report(report, Path(reports_dir))
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Publicacion
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(value: object) -> str:
    """Numero con 4 decimales, o ``null`` si no es un numero (nunca se inventa un ``0``)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    return "null"


def render_markdown(report: PaperTradingReport) -> str:
    """Informe en Markdown: la puerta, la muestra y la valla de honestidad."""
    payload = report.payload
    metrics = cast("Mapping[str, object]", payload["metrics"])
    verdict = cast("Mapping[str, object]", metrics["verdict"])
    reference = cast("Mapping[str, object]", payload["reference"])
    sample = cast("Mapping[str, object]", payload["sample"])
    counts = cast("Mapping[str, object]", sample["counts"])
    sessions = cast("list[dict[str, object]]", payload["sessions"])
    honesty = cast("Mapping[str, object]", payload["honesty"])
    order = (
        "trade_date",
        "direction",
        "entry_px",
        "stop_pct",
        "target_pct",
        "exit_px",
        "exit_reason",
        "gross_return_pct",
        "cost_pct",
        "net_return_pct",
    )
    lines = [
        f"# Paper trading de la Fase 4 (tarea #45) — {report.report_date}",
        "",
        f"- **Generado:** `{payload['generated_at']}` · **base:** `{payload['basis']}` · "
        f"**¿medicion?** `{payload['is_measurement']}`",
        f"- **Carril:** {payload['phase']}",
        "",
        "## Puerta de salida de la Fase 4 (§16)",
        "",
        f"- **N (sesiones con recomendacion):** {verdict['n_sessions']} "
        f"(minimo **{verdict['min_sessions']}**)",
        f"- **Media paper:** {_fmt(verdict['mean_paper_pct'])} % por sesion",
        f"- **Media backtest:** {_fmt(verdict['mean_backtest_pct'])} % por sesion",
        f"- **Sigma backtest:** {_fmt(verdict['sigma_backtest_pct'])} %",
        f"- **Umbral `2*sigma/sqrt(N)`:** {_fmt(verdict['threshold_pct'])} %",
        f"- **Estado:** **`{verdict['state']}`**",
        "",
        f"> {verdict['reason']}",
        "",
        "## Referencia",
        "",
        f"- **Artefacto:** `{reference['artifact']}` · **brazo:** `{reference['arm']}`",
        f"- **Serie:** `{reference['series']}` ({reference['units']}, n = {reference['n']})",
        f"- {reference['note']}",
        "",
        "## Muestra",
        "",
        f"- Sesiones con recomendacion: **{sample['n_sessions']}** · sin datos de mercado: "
        f"{sample['missing_outcomes']}",
        f"- Recuentos: `recommendation` = {counts['recommendation']}, "
        f"`nothing` = {counts['nothing']}, sin recomendacion = {counts['no_recommendation']}, "
        f"`error` = {counts['error']}",
        "",
    ]
    if sessions:
        lines += [
            "| " + " | ".join(order) + " |",
            "|" + "---|" * len(order),
            *("| " + " | ".join(str(item[key]) for key in order) + " |" for item in sessions),
            "",
        ]
    else:
        lines += [
            "_Sin recomendaciones direccionales todavia: el reloj de observacion no ha empezado a "
            "producir muestra._",
            "",
        ]
    overlay_block = cast("Mapping[str, object]", payload["with_overlay"])
    overrides = cast("list[dict[str, object]]", overlay_block["overrides"])
    lines += [
        "## Serie «con overlay» (§19.9, #150)",
        "",
        f"- **Sesiones direccionales con overlay:** {overlay_block['n_sessions']} · "
        f"**media:** {_fmt(overlay_block['mean_net_return_pct'])} % por sesion",
        f"- **Sesiones donde el overlay cambia la direccion publicada:** {len(overrides)}",
        f"- {overlay_block['note']}",
        "",
    ]
    if overrides:
        lines += [
            "| trade_date | publicada (sin overlay) | con overlay |",
            "|---|---|---|",
            *(
                f"| {item['trade_date']} | {item['published']} | {item['with_overlay']} |"
                for item in overrides
            ),
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


def write_report(report: PaperTradingReport, reports_dir: Path) -> tuple[Path, Path]:
    """Escribe el par ``.json``/``.md`` y devuelve sus rutas."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{REPORT_PREFIX}_{report.report_date}"
    json_path = reports_dir / f"{stem}.json"
    md_path = reports_dir / f"{stem}.md"
    json_path.write_text(canonical_text(report.payload), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: Sequence[str] | None = None) -> int:
    """CLI manual: recomputa la observacion y publica el informe."""
    parser = argparse.ArgumentParser(
        description="Informe de paper trading de la Fase 4 (tarea #45): la puerta de §16."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="raiz del almacen")
    parser.add_argument("--journal-root", type=Path, required=True, help="raiz del diario (#39)")
    parser.add_argument(
        "--reference-artifact",
        type=Path,
        required=True,
        help="informe del pipeline (#28) que declara la media y la sigma del backtest",
    )
    parser.add_argument("--reports-dir", type=Path, default=None, help="directorio de informes")
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="instante declarado ISO-8601, obligatorio para escribir",
    )
    args = parser.parse_args(argv)
    if args.as_of is None:
        print("error: `--as-of` es obligatorio (el modulo no lee el reloj)", file=sys.stderr)
        return 2
    try:
        as_of = datetime.fromisoformat(args.as_of)
    except ValueError:
        print(f"error: `--as-of` no es ISO-8601: {args.as_of!r}", file=sys.stderr)
        return 2
    try:
        report = analyse(
            store=Store(args.data_root),
            journal_root=args.journal_root,
            reference_artifact=args.reference_artifact,
            as_of=as_of,
            reports_dir=args.reports_dir,
            write=args.reports_dir is not None,
        )
    except PaperTradingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    metrics = cast("Mapping[str, object]", report.payload["metrics"])
    verdict = cast("Mapping[str, object]", metrics["verdict"])
    print(
        f"paper trading {report.report_date}: N = {verdict['n_sessions']}, "
        f"estado = {verdict['state']}, report_sha256 = {report.report_sha256[7:][:11]}…"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
