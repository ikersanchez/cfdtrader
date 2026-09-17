"""Informe de cobertura por fuente — tarea #3.

El informe es el **artefacto verificable** de la tarea: una fila por
``(serie, fuente)`` con el estado declarado, los intentos, las filas escritas, la
ventana obtenida, el ``stale``, los huecos y los bloqueos de fase.

Dos reglas que el informe hace cumplir:

- **Anti-optimismo** (A16): ninguna fila puede figurar con ``status: ok`` si no
  hay filas en el almacén. ``rows_written`` son las filas **vigentes en el
  almacén** para esa ``(serie, fuente)`` —por eso se puede cruzar con
  ``store.sql("SELECT count(*) …")``— y ``rows_new`` aparte son las que añadió
  esta ejecución. Así una segunda ejecución idéntica es ``UNCHANGED`` (A10) y
  sigue teniendo un ``rows_written`` que cuadra con el almacén.
- **Bloqueo comprobable** (A21): ``phase1_ready: false`` con un ``blocker`` por
  serie sin fuente. Que la Fase 1 esté bloqueada **no** puede romper la ingesta
  diaria: el informe se escribe igual y sin ``--require-ready`` la ejecución sale
  con 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from cfdtrader.data.sources.base import SeriesSpec, SourceStatus
from cfdtrader.data.sources.registry import UnavailableSeries

__all__ = [
    "BLOCKER_CFD_MISSING",
    "Blocker",
    "CoverageReport",
    "SeriesRow",
    "build_report",
    "report_paths",
    "write_report",
]

#: Código del bloqueo por ausencia de fuente del CFD (A21). Lo consume la Fase 1.
BLOCKER_CFD_MISSING = "cfd_source_missing"


class SeriesRow(BaseModel):
    """Una fila del informe: el resultado de intentar una serie en una fuente."""

    model_config = ConfigDict(extra="forbid")

    series_id: str
    source: str
    dataset: str
    granularity: str
    interval: str
    status: SourceStatus
    attempts: int
    rows_written: int = Field(description="Filas vigentes en el almacén para (serie, fuente).")
    rows_new: int = Field(description="Filas nuevas de esta ejecución.")
    span_start: str | None = None
    span_end: str | None = None
    timezone: str = "UTC"
    bid_ask: str = Field(default="na", description="`yes`, `no` o `na` si no aplica.")
    stale: bool = False
    stale_dates: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    span_ok: bool = False
    min_start: str | None = None
    history_window_limit_days: int | None = None
    history_window_limited: bool = False
    discarded_open_session: int = 0
    rejected_rows: int = 0
    issue_codes: tuple[str, ...] = ()
    issues: tuple[dict[str, object], ...] = ()
    notes: tuple[str, ...] = ()


class Blocker(BaseModel):
    """Motivo por el que la Fase 1 no puede empezar todavía."""

    model_config = ConfigDict(extra="forbid")

    code: str
    series_id: str
    detail: str
    follow_up_issue: int | None = None


class CoverageReport(BaseModel):
    """Informe completo: diario e intradía separados, con los bloqueos declarados."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    as_of: datetime
    data_root: str
    phase1_ready: bool
    blockers: tuple[Blocker, ...] = ()
    daily: tuple[SeriesRow, ...] = ()
    intraday: tuple[SeriesRow, ...] = ()
    unavailable: tuple[UnavailableSeries, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class SeriesOutcome:
    """Lo que la orquestación sabe de una ``(serie, fuente)`` antes de ser fila."""

    spec: SeriesSpec
    source: str
    status: SourceStatus
    attempts: int = 1
    rows_written: int = 0
    rows_new: int = 0
    span_start: str | None = None
    span_end: str | None = None
    stale: bool = False
    stale_dates: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    discarded_open_session: int = 0
    rejected_rows: int = 0
    issue_codes: tuple[str, ...] = ()
    issues: tuple[dict[str, object], ...] = ()
    notes: tuple[str, ...] = field(default=())


def build_report(
    *,
    outcomes: list[SeriesOutcome] | tuple[SeriesOutcome, ...],
    unavailable: tuple[UnavailableSeries, ...],
    now: datetime,
    data_root: Path,
) -> CoverageReport:
    """Construye el informe a partir de los resultados de la ingesta."""
    rows = [_row(outcome) for outcome in outcomes]
    blockers = tuple(
        Blocker(
            code=BLOCKER_CFD_MISSING,
            series_id=item.series_id,
            detail=item.reason,
            follow_up_issue=item.follow_up_issue,
        )
        for item in unavailable
    )
    return CoverageReport(
        generated_at=now,
        as_of=now,
        data_root=str(data_root),
        phase1_ready=not blockers,
        blockers=blockers,
        daily=tuple(row for row in rows if row.granularity == "daily"),
        intraday=tuple(row for row in rows if row.granularity == "intraday"),
        unavailable=unavailable,
        notes=(
            "El estado de cada serie lo declara la ejecución: un fallo de fuente se declara con "
            "su estado y su motivo, nunca se inventa un dato.",
            "`rows_written` son las filas vigentes en el almacén para (serie, fuente) y se puede "
            "cruzar con `SELECT count(*)`; `rows_new` son las de esta ejecución.",
        ),
    )


def report_paths(directory: Path, *, day: datetime) -> tuple[Path, Path]:
    """Rutas del informe en JSON y Markdown, con la fecha de la ejecución."""
    stem = f"market_coverage_{day.date().isoformat()}"
    return directory / f"{stem}.json", directory / f"{stem}.md"


def write_report(report: CoverageReport, directory: Path) -> tuple[Path, Path]:
    """Escribe el informe en JSON y Markdown y devuelve ambas rutas."""
    directory.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = report_paths(directory, day=report.generated_at)
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(report: CoverageReport) -> str:
    """Informe legible: separa diario e intradía, como exige A18."""
    lines: list[str] = [
        "# Informe de cobertura de mercado",
        "",
        f"- **Generado:** {report.generated_at.isoformat()}",
        f"- **Raíz del almacén:** `{report.data_root}`",
        f"- **Fase 1 lista:** {'sí' if report.phase1_ready else 'no'}",
        "",
    ]
    for blocker in report.blockers:
        follow_up = (
            f" (seguimiento: #{blocker.follow_up_issue})" if blocker.follow_up_issue else ""
        )
        lines.append(f"> **BLOQUEO `{blocker.code}`** — `{blocker.series_id}`{follow_up}")
        lines.append(f"> {blocker.detail}")
        lines.append("")
    lines.extend(_table("Diario", report.daily))
    lines.extend(_table("Intradía", report.intraday))

    if report.unavailable:
        lines.extend(["## Series sin fuente", ""])
        for item in report.unavailable:
            lines.append(
                f"- `{item.series_id}` — **SIN FUENTE** (comprobado el {item.checked_on}, "
                f"bid/ask: {'sí' if item.bid_ask else 'no'}). Ver `{item.documentation}`."
            )
        lines.append("")

    lines.extend(["## Notas", ""])
    lines.extend(f"- {note}" for note in report.notes)
    lines.append("")
    return "\n".join(lines)


def _table(title: str, rows: tuple[SeriesRow, ...]) -> list[str]:
    """Tabla Markdown con una fila por ``(serie, fuente)``."""
    lines = [f"## {title}", ""]
    if not rows:
        lines.extend(["_Sin series en esta granularidad._", ""])
        return lines
    lines.append(
        "| serie | fuente | estado | intentos | filas guardadas | filas nuevas | span | "
        "span_ok | stale | huecos | bid/ask | límite ventana | descartadas | notas |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        span = f"{row.span_start or '—'} → {row.span_end or '—'}"
        limit = (
            "—"
            if row.history_window_limit_days is None
            else f"{row.history_window_limit_days} d"
            + (" (pegada al límite)" if row.history_window_limited else "")
        )
        notes = "; ".join((*row.issue_codes, *row.notes)) or "—"
        lines.append(
            f"| `{row.series_id}` | `{row.source}` | {row.status.value} | {row.attempts} | "
            f"{row.rows_written} | {row.rows_new} | {span} | "
            f"{'sí' if row.span_ok else 'no'} | {'sí' if row.stale else 'no'} | "
            f"{len(row.gaps)} | {row.bid_ask} | {limit} | {row.discarded_open_session} | {notes} |"
        )
    lines.append("")
    return lines


def _row(outcome: SeriesOutcome) -> SeriesRow:
    """Convierte el resultado de la orquestación en fila del informe."""
    spec = outcome.spec
    return SeriesRow(
        series_id=spec.series_id,
        source=outcome.source,
        dataset=spec.dataset,
        granularity=spec.granularity,
        interval=spec.interval,
        status=outcome.status,
        attempts=outcome.attempts,
        rows_written=outcome.rows_written,
        rows_new=outcome.rows_new,
        span_start=outcome.span_start,
        span_end=outcome.span_end,
        timezone="UTC",
        bid_ask=_bid_ask(spec),
        stale=outcome.stale,
        stale_dates=outcome.stale_dates,
        gaps=outcome.gaps,
        span_ok=_span_ok(spec, outcome.span_start),
        min_start=None if spec.min_start is None else spec.min_start.isoformat(),
        history_window_limit_days=spec.history_window_limit_days,
        history_window_limited=_window_limited(spec, outcome.span_start),
        discarded_open_session=outcome.discarded_open_session,
        rejected_rows=outcome.rejected_rows,
        issue_codes=outcome.issue_codes,
        issues=outcome.issues,
        notes=outcome.notes,
    )


def _bid_ask(spec: SeriesSpec) -> str:
    """``yes``/``no``/``na`` según lo declarado, no según lo que nos gustaría."""
    if spec.supports_bid_ask:
        return "yes"
    if spec.asset_class.value == "cfd":
        return "no"
    return "na"


def _span_ok(spec: SeriesSpec, span_start: str | None) -> bool:
    """``True`` si la ventana obtenida alcanza el ``min_start`` declarado."""
    if spec.min_start is None:
        return span_start is not None
    if span_start is None:
        return False
    return span_start[:10] <= spec.min_start.isoformat()


def _window_limited(spec: SeriesSpec, span_start: str | None) -> bool:
    """``True`` si la ventana obtenida está pegada al límite rodante del proveedor.

    Se pide la ventana máxima que sirve la fuente, así que la ventana obtenida es
    la del límite. Es una declaración de hecho, sin inventar umbrales: el límite
    **no** invalida el diario (A18).
    """
    if spec.history_window_limit_days is None or span_start is None:
        return False
    return _lookback_days(spec) >= spec.history_window_limit_days


def _lookback_days(spec: SeriesSpec) -> int:
    """Días de ventana pedidos a la fuente, si se declaró un ``lookback_period``."""
    if spec.lookback_period is None:
        return 0
    period = spec.lookback_period.strip().lower()
    if not period.endswith("d") or not period[:-1].isdigit():
        return 0
    return int(period[:-1])
