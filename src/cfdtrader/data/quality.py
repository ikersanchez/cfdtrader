"""Validaciones de calidad de datos (``plan.md`` §8.4) — tarea #3.

Cada validación devuelve **la lista de fechas afectadas**, no un contador suelto:
un «hay 3 problemas» no permite investigar nada, un «estas tres fechas» sí.

Dos reglas de diseño que atraviesan el módulo:

- **Todas las validaciones reciben ``now``**; ninguna llama a ``datetime.now()``
  por dentro (A13). Dos llamadas con el mismo ``now`` dan el mismo veredicto.
- Huecos y festivos se reportan **de forma descriptiva, no autoritativa**: el
  informe dice «N sesiones sin dato entre X e Y» y **no** afirma «es festivo».
  El calendario de sesiones es la tarea #4 y este módulo no lo importa.

Las comprobaciones se aplican en este orden: las que **rechazan** filas (nunca
se escribe un dato dudoso) y luego las que **marcan** el conjunto (``stale``,
huecos).

Coherencia entre fuentes: :func:`compare_sources` solo compara **la misma serie
lógica** (``^GSPC`` por ``yfinance`` frente a ``^GSPC`` por Stooq). Comparar
``SPY`` o ``ES=F`` con ``^GSPC`` está prohibido: son instrumentos distintos con
niveles distintos y el informe lo dice así.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Final

import polars as pl

from cfdtrader.data.sources.base import AssetClass, SeriesSpec
from cfdtrader.data.sources.frames import SESSION_CLOSE_ET, SESSION_OPEN_ET

__all__ = [
    "STALE_RUN",
    "CROSS_SOURCE_TOLERANCE_BP",
    "QualityIssue",
    "QualityReport",
    "compare_sources",
    "validate",
]

#: Sesiones consecutivas con el mismo ``close`` a partir de las cuales se marca *stale*.
STALE_RUN: Final[int] = 5

#: Diferencia relativa máxima tolerada entre dos fuentes para la misma serie lógica,
#: en puntos básicos (A12-h).
CROSS_SOURCE_TOLERANCE_BP: Final[float] = 10.0

#: Códigos de problema. Son estables: el informe los cita tal cual.
CODE_DUPLICATE = "duplicate"
CODE_NULL_OHLC = "null_ohlc"
CODE_OHLC_INCOHERENT = "ohlc_incoherent"
CODE_NEGATIVE_VOLUME = "negative_volume"
CODE_NULL_VOLUME = "null_volume"
CODE_OUT_OF_WINDOW = "out_of_window"
CODE_STALE = "stale"
CODE_GAP = "gap"
CODE_CROSS_SOURCE = "cross_source"


@dataclass(frozen=True, slots=True)
class QualityIssue:
    """Un problema de calidad con sus fechas exactas."""

    code: str
    series_id: str
    dates: tuple[str, ...]
    rows: int
    detail: str


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Resultado de validar una respuesta: qué se acepta y qué se rechaza, y por qué."""

    series_id: str
    source: str
    accepted: pl.DataFrame
    rejected_rows: int
    issues: tuple[QualityIssue, ...] = ()
    stale: bool = False
    stale_dates: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    checked_at: datetime | None = None
    notes: tuple[str, ...] = field(default=())

    @property
    def issue_codes(self) -> tuple[str, ...]:
        """Códigos presentes, sin repetir y en orden de aparición."""
        seen: list[str] = []
        for issue in self.issues:
            if issue.code not in seen:
                seen.append(issue.code)
        return tuple(seen)

    def issue(self, code: str) -> QualityIssue | None:
        """El problema con ese código, si lo hay."""
        for issue in self.issues:
            if issue.code == code:
                return issue
        return None


def validate(
    frame: pl.DataFrame,
    *,
    spec: SeriesSpec,
    source: str,
    now: datetime,
) -> QualityReport:
    """Valida una respuesta de fuente y devuelve las filas aceptadas y los problemas.

    Las filas con ``as_of`` posterior a ``now`` (sesión o barra todavía en
    formación) no se validan aquí: las descarta la orquestación antes de llegar.
    """
    issues: list[QualityIssue] = []
    frame = frame.filter(pl.col("as_of") <= pl.lit(now))
    if frame.height == 0:
        return QualityReport(
            series_id=spec.series_id,
            source=source,
            accepted=frame,
            rejected_rows=0,
            checked_at=now,
        )

    reject = pl.lit(value=False)
    rejected_rows = 0

    # (a) Duplicados dentro de la misma respuesta: si una identidad aparece dos
    # veces, no se puede saber cuál es la buena, así que se rechazan las dos.
    duplicated = (
        frame.group_by("as_of").len().filter(pl.col("len") > 1).sort("as_of").get_column("as_of")
    )
    duplicate_times = _instant_strings(duplicated)
    if duplicate_times:
        # `implode()` convierte la columna en una lista: `is_in` de un elemento
        # contra esa lista, sin la ambigüedad de serie contra serie que polars
        # marca como obsoleta.
        duplicate_mask = pl.col("as_of").is_in(duplicated.implode())
        rows = int(frame.filter(duplicate_mask).height)
        rejected_rows += rows
        reject = reject | duplicate_mask
        issues.append(
            QualityIssue(
                code=CODE_DUPLICATE,
                series_id=spec.series_id,
                dates=duplicate_times,
                rows=rows,
                detail=(
                    f"{rows} filas repiten una identidad (series_id, as_of) dentro de la misma "
                    "respuesta: se rechazan todas, no se escribe ninguna"
                ),
            )
        )

    # (b) OHLC nulo.
    null_ohlc = (
        pl.col("open").is_null()
        | pl.col("high").is_null()
        | pl.col("low").is_null()
        | pl.col("close").is_null()
    )
    issues, rejected_rows, reject = _apply_rejection(
        frame,
        mask=null_ohlc,
        code=CODE_NULL_OHLC,
        detail="OHLC incompleto: falta al menos uno de open/high/low/close",
        spec=spec,
        issues=issues,
        rejected_rows=rejected_rows,
        reject=reject,
    )

    # (c) Coherencia OHLC.
    incoherent = pl.col("high") < pl.max_horizontal("open", "close")
    incoherent = incoherent | (pl.col("low") > pl.min_horizontal("open", "close"))
    issues, rejected_rows, reject = _apply_rejection(
        frame,
        mask=incoherent,
        code=CODE_OHLC_INCOHERENT,
        detail="high < max(open, close) o low > min(open, close)",
        spec=spec,
        issues=issues,
        rejected_rows=rejected_rows,
        reject=reject,
    )

    # (d) Volumen negativo; y volumen nulo admitido solo para índices.
    negative = pl.col("volume").is_not_null() & (pl.col("volume") < 0)
    issues, rejected_rows, reject = _apply_rejection(
        frame,
        mask=negative,
        code=CODE_NEGATIVE_VOLUME,
        detail="volume < 0",
        spec=spec,
        issues=issues,
        rejected_rows=rejected_rows,
        reject=reject,
    )
    if spec.asset_class is not AssetClass.INDEX and spec.volume_expected:
        null_volume = pl.col("volume").is_null()
        issues, rejected_rows, reject = _apply_rejection(
            frame,
            mask=null_volume,
            code=CODE_NULL_VOLUME,
            detail=(
                "volume nulo en una serie que debería publicarlo: solo `asset_class: index` "
                "(o una serie declarada con `volume_expected: false`) puede no traer volumen"
            ),
            spec=spec,
            issues=issues,
            rejected_rows=rejected_rows,
            reject=reject,
        )

    # (e) Barra fuera de la ventana 09:30–16:00 ET.
    if spec.granularity == "intraday":
        local = pl.col("as_of").dt.convert_time_zone("America/New_York")
        outside = (local.dt.time() < SESSION_OPEN_ET) | (local.dt.time() > SESSION_CLOSE_ET)
        issues, rejected_rows, reject = _apply_rejection(
            frame,
            mask=outside,
            code=CODE_OUT_OF_WINDOW,
            detail="barra fuera de la ventana 09:30–16:00 ET",
            spec=spec,
            issues=issues,
            rejected_rows=rejected_rows,
            reject=reject,
        )

    accepted = frame.filter(~reject).sort("as_of")

    # (f) `close` idéntico en sesiones consecutivas: se marca, no se rechaza.
    stale_dates = _stale_dates(accepted) if spec.granularity == "daily" else ()
    if stale_dates:
        issues.append(
            QualityIssue(
                code=CODE_STALE,
                series_id=spec.series_id,
                dates=stale_dates,
                rows=len(stale_dates),
                detail=(
                    f"close idéntico en {STALE_RUN} o más sesiones consecutivas "
                    "(posible dato congelado de la fuente)"
                ),
            )
        )

    # (g) Huecos: días laborables sin dato entre la primera y la última fecha.
    gaps: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    if spec.granularity == "daily":
        gaps = _gaps(accepted)
        if gaps:
            issues.append(
                QualityIssue(
                    code=CODE_GAP,
                    series_id=spec.series_id,
                    dates=gaps,
                    rows=len(gaps),
                    detail=(
                        f"{len(gaps)} días laborables sin dato entre la primera y la última "
                        "fecha de esta fuente. Es descriptivo: no se afirma que sean festivos "
                        "(el calendario de sesiones es la tarea #4)"
                    ),
                )
            )
        notes = ("los huecos se calculan sobre días laborables; no se valida el calendario",)

    return QualityReport(
        series_id=spec.series_id,
        source=source,
        accepted=accepted,
        rejected_rows=rejected_rows,
        issues=tuple(issues),
        stale=bool(stale_dates),
        stale_dates=stale_dates,
        gaps=gaps,
        checked_at=now,
        notes=notes,
    )


def compare_sources(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    series_id: str,
    left_source: str,
    right_source: str,
    tolerance_bp: float = CROSS_SOURCE_TOLERANCE_BP,
) -> QualityIssue | None:
    """Compara el **mismo** ``series_id`` entre dos fuentes (A12-h).

    Solo compara la misma serie lógica. Comparar ``^GSPC`` con ``SPY`` o
    ``ES=F`` está prohibido: son instrumentos distintos, con niveles distintos y
    con bases distintas, y una diferencia entre ellos no es un problema de
    calidad sino una diferencia real de instrumento.

    Returns
    -------
    QualityIssue | None
        El problema de coherencia, o ``None`` si la diferencia cabe en la
        tolerancia o si no hay fechas comunes.
    """
    if left.height == 0 or right.height == 0:
        return None
    joined = left.select("as_of", pl.col("close").alias("left")).join(
        right.select("as_of", pl.col("close").alias("right")),
        on="as_of",
        how="inner",
    )
    if joined.height == 0:
        return None
    difference_bp = (
        (pl.col("left") - pl.col("right")).abs() / pl.col("right").abs() * 10_000.0
    ).alias("difference_bp")
    flagged = (
        joined.with_columns(difference_bp)
        .filter(pl.col("difference_bp") > tolerance_bp)
        .sort("as_of")
    )
    if flagged.height == 0:
        return None
    return QualityIssue(
        code=CODE_CROSS_SOURCE,
        series_id=series_id,
        dates=_instant_strings(flagged.get_column("as_of")),
        rows=flagged.height,
        detail=(
            f"{left_source} y {right_source} difieren más de {tolerance_bp} pb en el cierre de "
            f"{flagged.height} fechas comunes de la misma serie {series_id}"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internos
# ─────────────────────────────────────────────────────────────────────────────
def _apply_rejection(
    frame: pl.DataFrame,
    *,
    mask: pl.Expr,
    code: str,
    detail: str,
    spec: SeriesSpec,
    issues: list[QualityIssue],
    rejected_rows: int,
    reject: pl.Expr,
) -> tuple[list[QualityIssue], int, pl.Expr]:
    """Registra un problema y acumula su máscara de rechazo, si hay filas afectadas."""
    affected = frame.filter(mask & ~reject)
    if affected.height == 0:
        return issues, rejected_rows, reject
    dates = _instant_strings(affected.get_column("as_of"))
    issues.append(
        QualityIssue(
            code=code,
            series_id=spec.series_id,
            dates=dates,
            rows=affected.height,
            detail=detail,
        )
    )
    return issues, rejected_rows + affected.height, reject | mask


def _instant_strings(column: pl.Series) -> tuple[str, ...]:
    """Instantes como texto ISO, para que el informe sea legible y comparable."""
    values: tuple[str, ...] = tuple(str(value) for value in column.to_list())
    return values


def _stale_dates(frame: pl.DataFrame) -> tuple[str, ...]:
    """Fechas que participan en una racha de ``close`` idéntico de longitud >= ``STALE_RUN``."""
    if frame.height < STALE_RUN:
        return ()

    local = frame.with_columns(
        pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("session")
    ).sort("session")
    sessions: list[date] = []
    closes: list[float | None] = []
    for row in local.iter_rows(named=True):
        session = row["session"]
        if isinstance(session, datetime):
            session = session.date()
        if not isinstance(session, date):
            continue
        close = row["close"]
        sessions.append(session)
        closes.append(float(close) if isinstance(close, (int, float)) else None)

    stale: list[str] = []
    run_start = 0
    for index in range(1, len(sessions) + 1):
        continue_run = (
            index < len(sessions)
            and closes[index] is not None
            and closes[index] == closes[run_start]
            and (sessions[index] - sessions[index - 1]).days <= 4
        )
        if continue_run:
            continue
        run_length = index - run_start
        if run_length >= STALE_RUN and closes[run_start] is not None:
            stale.extend(session.isoformat() for session in sessions[run_start:index])
        run_start = index
    return tuple(stale)


def _gaps(frame: pl.DataFrame) -> tuple[str, ...]:
    """Días laborables sin dato entre la primera y la última fecha de la muestra."""
    if frame.height < 2:
        return ()
    sessions = sorted(
        {
            value.date() if isinstance(value, datetime) else value
            for value in frame.with_columns(
                pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("session")
            )
            .get_column("session")
            .to_list()
            if isinstance(value, (date, datetime))
        }
    )
    if len(sessions) < 2:
        return ()
    present = set(sessions)
    cursor = sessions[0]
    missing: list[str] = []
    while cursor <= sessions[-1]:
        if cursor.weekday() < 5 and cursor not in present:
            missing.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return tuple(missing)
