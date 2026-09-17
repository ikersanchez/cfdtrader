"""Adaptador de Stooq (respaldo declarado del histórico diario) — tarea #3.

Stooq sirve CSV directo en ``https://stooq.com/q/d/l/?s=<símbolo>&i=d`` y su
histórico diario es más largo y más estable que el de Yahoo, así que
``tech_stack.md`` §4.5 lo declara respaldo del diario.

⚠️ **Hoy no es usable programáticamente** (comprobado el 2026-09-17): responde
``200`` con un *challenge* JavaScript (``POST /__verify``) en lugar de CSV. Ese
caso termina en :class:`SourceBlockedError` con el ``Content-Type`` y los
primeros bytes en el mensaje, que es exactamente lo que exige A4(ii). El
*challenge* **no** genera issue de seguimiento: se documenta en
``_docs/data_sources.md`` y el respaldo de facto del diario es ``yfinance`` con
sus dos configuraciones de descarga (A3).
"""

from __future__ import annotations

from datetime import date
from io import StringIO

import polars as pl

from cfdtrader.data.sources.base import (
    FetchRequest,
    FetchResult,
    SeriesSpec,
    SourceAdapter,
    SourceBlockedError,
    SourceError,
    SourceRateLimitedError,
    SourceStatus,
    SourceUnavailableError,
)
from cfdtrader.data.sources.frames import empty_canonical_frame, session_close_utc
from cfdtrader.data.sources.http import CachedHttpClient

__all__ = ["STOOQ_DAILY_URL", "StooqAdapter", "stooq_symbols"]

#: Endpoint del histórico diario en CSV.
STOOQ_DAILY_URL = "https://stooq.com/q/d/l/"

#: Traducción de símbolo propio → símbolo de Stooq.
#:
#: **Nunca** aparece aquí ``SPX500:CFD``: mapearlo a ``^spx`` sería sustituir el
#: instrumento del proyecto por el índice. El test
#: ``test_cfd_has_no_alias_to_index_or_future`` falla si alguien lo introduce.
_SYMBOLS: dict[str, str] = {
    "^GSPC": "^spx",
}


def stooq_symbols() -> dict[str, str]:
    """Mapa de símbolos propio → Stooq, para que el registro pueda auditarlo."""
    return dict(_SYMBOLS)


class StooqAdapter(SourceAdapter):
    """Histórico diario de respaldo, con el cliente HTTP inyectado."""

    name = "stooq"

    def __init__(self, client: CachedHttpClient) -> None:
        self._client = client

    def fetch(self, request: FetchRequest) -> FetchResult:
        """Descarga y normaliza el CSV diario, o declara por qué no pudo."""
        spec = request.spec
        if spec.granularity != "daily":
            return _failed(
                spec,
                SourceStatus.UNAVAILABLE,
                f"stooq solo se usa como respaldo diario, no para {spec.granularity}",
            )
        symbol = _SYMBOLS.get(spec.series_id)
        if symbol is None:
            return _failed(
                spec,
                SourceStatus.UNAVAILABLE,
                f"no hay símbolo de Stooq declarado para {spec.series_id!r}",
            )

        try:
            cached = self._client.get(
                STOOQ_DAILY_URL,
                params={"s": symbol, "i": "d"},
                now=request.now,
            )
            frame = _parse_csv(cached.text, spec=spec)
        except SourceError as error:
            return FetchResult(
                spec=spec,
                source=self.name,
                status=_status_for(error),
                frame=empty_canonical_frame(),
                attempts=error.attempts,
                error=str(error),
            )

        if frame.height == 0:
            return _failed(spec, SourceStatus.UNAVAILABLE, "el CSV de Stooq no trae filas")
        return FetchResult(
            spec=spec,
            source=self.name,
            status=SourceStatus.OK,
            frame=frame,
            attempts=cached.attempts,
            notes=(f"cache={'sí' if cached.from_cache else 'no'}",),
        )


def _failed(spec: SeriesSpec, status: SourceStatus, message: str) -> FetchResult:
    """Resultado fallido con el frame canónico vacío: nunca filas sin estado ``ok``."""
    return FetchResult(
        spec=spec,
        source=StooqAdapter.name,
        status=status,
        frame=empty_canonical_frame(),
        attempts=1,
        error=message,
    )


def _status_for(error: SourceError) -> SourceStatus:
    """Estado declarado que corresponde al error del cliente HTTP."""
    if isinstance(error, SourceBlockedError):
        return SourceStatus.BLOCKED
    if isinstance(error, SourceRateLimitedError):
        return SourceStatus.RATE_LIMITED
    if isinstance(error, SourceUnavailableError):
        return SourceStatus.UNAVAILABLE
    return SourceStatus.ERROR


def _parse_csv(text: str, *, spec: SeriesSpec) -> pl.DataFrame:
    """Convierte el CSV de Stooq en el frame canónico, en UTC.

    Stooq no publica ``Adj Close`` ni ``bid``/``ask``: esas columnas salen
    nulas. Rellenarlas con el ``close`` sería inventar un dato.
    """
    try:
        raw = pl.read_csv(
            StringIO(text), columns=["Date", "Open", "High", "Low", "Close", "Volume"]
        )
    except (pl.exceptions.PolarsError, KeyError) as error:
        raise SourceUnavailableError(
            f"el CSV de Stooq no se pudo parsear: {error}",
            source=StooqAdapter.name,
            series_id=spec.series_id,
        ) from error

    days: list[date] = []
    rows: list[dict[str, object]] = []
    for row in raw.iter_rows(named=True):
        day = _as_date(row.get("Date"))
        if day is None:
            continue
        days.append(day)
        rows.append(row)

    if not days:
        return empty_canonical_frame()

    as_of = [session_close_utc(day) for day in days]
    return pl.DataFrame(
        {
            "as_of": as_of,
            "open": [row.get("Open") for row in rows],
            "high": [row.get("High") for row in rows],
            "low": [row.get("Low") for row in rows],
            "close": [row.get("Close") for row in rows],
            "volume": [float(str(row.get("Volume") or 0.0)) for row in rows],
            "adj_close": [None] * len(rows),
            "bid": [None] * len(rows),
            "ask": [None] * len(rows),
        },
        schema_overrides={
            "as_of": pl.Datetime("us", "UTC"),
            "open": pl.Float64(),
            "high": pl.Float64(),
            "low": pl.Float64(),
            "close": pl.Float64(),
            "volume": pl.Float64(),
            "adj_close": pl.Float64(),
            "bid": pl.Float64(),
            "ask": pl.Float64(),
        },
    ).sort("as_of")


def _as_date(value: object) -> date | None:
    """Fecha de la columna ``Date`` del CSV."""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None
