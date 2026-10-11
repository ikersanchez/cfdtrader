"""Candidatas historicas de ES y S&P 500 para el estudio offline de #156.

Este modulo es independiente de ``commodities_v1`` y no persiste features. Recibe las barras
locales de ``raw.market_daily``, alinea por las sesiones de ``^GSPC`` y publica valores nulos
cuando falta una entrada o el precio no es valido. No consulta ninguna fuente ni lee el reloj.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Final, cast

import polars as pl

from cfdtrader.features.store import CatalogEntry

__all__ = [
    "HISTORICAL_MARKET_CANDIDATE_CATALOG",
    "HISTORICAL_MARKET_CANDIDATE_COLUMNS",
    "HISTORICAL_MARKET_CANDIDATE_SET",
    "HISTORICAL_MARKET_DESIGN_LAG_BY_FEATURE",
    "HISTORICAL_MARKET_INPUT_SERIES",
    "REFERENCE_SNAPSHOT_DATE",
    "HistoricalMarketCandidates",
    "HistoricalMarketDataError",
    "build_historical_market_candidates",
]

HISTORICAL_MARKET_CANDIDATE_SET: Final[str] = "historical_market_v1"
HISTORICAL_MARKET_INPUT_SERIES: Final[tuple[str, ...]] = ("ES=F", "^GSPC")
REFERENCE_SNAPSHOT_DATE: Final[date] = date(2026, 10, 9)

HISTORICAL_MARKET_CANDIDATE_CATALOG: Final[tuple[CatalogEntry, ...]] = (
    CatalogEntry(
        "es_basis",
        "ES=F_close(t−1) / ^GSPC_close(t−1) − 1",  # noqa: RUF001
        1,
        "raw.market_daily",
        "cierre de la sesion t-1",
        0,
    ),
    CatalogEntry(
        "spx_overnight_ret_1",
        "ln(^GSPC_open(t−1) / ^GSPC_close(t−2))",  # noqa: RUF001
        2,
        "raw.market_daily",
        "apertura de t-1 y cierre de t-2",
        0,
    ),
)
HISTORICAL_MARKET_CANDIDATE_COLUMNS: Final[tuple[str, ...]] = tuple(
    entry.name for entry in HISTORICAL_MARKET_CANDIDATE_CATALOG
)
HISTORICAL_MARKET_DESIGN_LAG_BY_FEATURE: Final[dict[str, int]] = {
    entry.name: entry.design_lag for entry in HISTORICAL_MARKET_CANDIDATE_CATALOG
}


class HistoricalMarketDataError(ValueError):
    """Las barras locales no permiten una alineacion historica inequivoca."""


@dataclass(frozen=True, slots=True)
class HistoricalMarketCandidates:
    """Matriz candidata y auditoria de cobertura del snapshot de referencia."""

    frame: pl.DataFrame
    audit: Mapping[str, object]


def _valid_price(value: object) -> float | None:
    """Convierte un precio estrictamente positivo y finito; el resto se conserva como nulo."""
    if value is None or isinstance(value, bool):
        return None
    try:
        price = float(cast("float", value))
    except (TypeError, ValueError, OverflowError):
        return None
    return price if math.isfinite(price) and price > 0.0 else None


def _input_rows(bars: pl.DataFrame) -> dict[str, dict[date, dict[str, object]]]:
    """Valida esquema, sesiones unicas y separa las dos series declaradas."""
    required = {"series_id", "session", "open", "close"}
    missing = sorted(required - set(bars.columns))
    if missing:
        raise HistoricalMarketDataError(f"faltan columnas de entrada: {missing}")

    rows: dict[str, dict[date, dict[str, object]]] = {
        series_id: {} for series_id in HISTORICAL_MARKET_INPUT_SERIES
    }
    for row in bars.iter_rows(named=True):
        series_id = row["series_id"]
        if series_id not in rows:
            continue
        session = row["session"]
        if not isinstance(session, date):
            raise HistoricalMarketDataError(
                f"la serie {series_id!r} contiene una sesion ausente o no-fecha: {session!r}"
            )
        session_rows = rows[cast("str", series_id)]
        if session in session_rows:
            raise HistoricalMarketDataError(
                f"sesion duplicada para {series_id!r}: {session.isoformat()}"
            )
        session_rows[session] = row
    return rows


def _audit_series(
    bars: pl.DataFrame,
    rows: Mapping[str, Mapping[date, Mapping[str, object]]],
    *,
    reference_snapshot: date,
) -> dict[str, object]:
    """Resume filas, sesiones, fuentes y precios invalidos hasta el snapshot fijado."""
    result: dict[str, object] = {}
    for series_id in HISTORICAL_MARKET_INPUT_SERIES:
        sessions = sorted(session for session in rows[series_id] if session <= reference_snapshot)
        source_values: list[str] = []
        if "source" in bars.columns:
            source_values = sorted(
                {
                    str(row["source"])
                    for row in bars.filter(
                        (pl.col("series_id") == series_id)
                        & (pl.col("session") <= reference_snapshot)
                    ).iter_rows(named=True)
                    if row["source"] is not None
                }
            )
        invalid_open = sum(
            _valid_price(rows[series_id][session]["open"]) is None for session in sessions
        )
        invalid_close = sum(
            _valid_price(rows[series_id][session]["close"]) is None for session in sessions
        )
        result[series_id] = {
            "input_rows": len(sessions),
            "unique_sessions": len(sessions),
            "duplicate_sessions": 0,
            "sessions_unique": True,
            "first_session": sessions[0].isoformat() if sessions else None,
            "last_session": sessions[-1].isoformat() if sessions else None,
            "sources": source_values,
            "invalid_open_rows": invalid_open,
            "invalid_close_rows": invalid_close,
        }
    return result


def _audit_candidates(frame: pl.DataFrame, *, reference_snapshot: date) -> dict[str, object]:
    """Cuadra filas validas con las ausentes/invalidas para cada candidata."""
    snapshot = frame.filter(pl.col("session") <= reference_snapshot)
    result: dict[str, object] = {}
    for column in HISTORICAL_MARKET_CANDIDATE_COLUMNS:
        total = snapshot.height
        valid = int(snapshot.get_column(column).is_not_null().sum())
        missing_or_invalid = total - valid
        result[column] = {
            "rows": total,
            "valid_rows": valid,
            "missing_or_invalid_rows": missing_or_invalid,
            "reconciles": valid + missing_or_invalid == total,
        }
    return result


def build_historical_market_candidates(
    bars: pl.DataFrame, *, reference_snapshot: date = REFERENCE_SNAPSHOT_DATE
) -> HistoricalMarketCandidates:
    """Calcula las dos candidatas hacia la fila ``t`` en el calendario de sesiones de ``^GSPC``.

    Para una fila ``t``, ``t-1`` y ``t-2`` son posiciones previas del calendario real de ``^GSPC``.
    Una sesion ausente en ES, o cualquier precio requerido nulo, no positivo o no finito, produce
    ``null``. Una sesion duplicada no se resuelve por fuente ni por orden: aborta la medicion.
    """
    rows = _input_rows(bars)
    spx = rows["^GSPC"]
    es = rows["ES=F"]
    sessions = sorted(spx)
    basis_values: list[float | None] = []
    overnight_values: list[float | None] = []

    for index, _session in enumerate(sessions):
        basis: float | None = None
        overnight: float | None = None
        if index >= 1:
            previous = sessions[index - 1]
            es_close = _valid_price(es.get(previous, {}).get("close"))
            spx_close = _valid_price(spx[previous]["close"])
            if es_close is not None and spx_close is not None:
                value = es_close / spx_close - 1.0
                basis = value if math.isfinite(value) else None
        if index >= 2:
            previous = sessions[index - 1]
            two_sessions_back = sessions[index - 2]
            spx_open = _valid_price(spx[previous]["open"])
            earlier_close = _valid_price(spx[two_sessions_back]["close"])
            if spx_open is not None and earlier_close is not None:
                ratio = spx_open / earlier_close
                if math.isfinite(ratio) and ratio > 0.0:
                    overnight = math.log(ratio)
        basis_values.append(basis)
        overnight_values.append(overnight)

    frame = pl.DataFrame(
        {
            "session": pl.Series(sessions, dtype=pl.Date()),
            "es_basis": pl.Series(basis_values, dtype=pl.Float64()),
            "spx_overnight_ret_1": pl.Series(overnight_values, dtype=pl.Float64()),
        }
    )
    audit: dict[str, object] = {
        "reference_snapshot_date": reference_snapshot.isoformat(),
        "input_series": _audit_series(bars, rows, reference_snapshot=reference_snapshot),
        "candidate_rows": _audit_candidates(frame, reference_snapshot=reference_snapshot),
    }
    return HistoricalMarketCandidates(frame=frame, audit=audit)
