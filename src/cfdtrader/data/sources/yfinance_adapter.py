"""Adaptador de Yahoo Finance (``yfinance``) — tarea #3.

Es **el único fichero del proyecto** que importa ``yfinance`` (A1): ninguna otra
parte del código habla con Yahoo. ``yfinance`` es un *scraper* no oficial, así
que vive detrás de esta interfaz propia, con reintentos, normalización y estado
declarado (``tech_stack.md`` §4.5).

Dos configuraciones de descarga, que es el "respaldo de facto" del diario (A3):

1. ``yf.download`` — la vía normal.
2. ``yf.Ticker(...).history(...)`` — la segunda, cuando la primera falla o
   devuelve vacío. No es una fuente nueva: es la misma fuente, otra puerta.

Ninguna de las dos publica ``bid``/``ask``: esas columnas salen nulas, porque
está prohibido rellenarlas con otra cosa (A8).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from typing import cast

import pandas as pd
import polars as pl
import yfinance as yf
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from cfdtrader.data.sources.base import (
    FetchRequest,
    FetchResult,
    SeriesSpec,
    SourceAdapter,
    SourceError,
    SourceRateLimitedError,
    SourceStatus,
    SourceUnavailableError,
)
from cfdtrader.data.sources.frames import empty_canonical_frame, session_close_utc, to_utc

__all__ = ["YFinanceAdapter"]

#: Nombres canónicos y sus equivalentes en la respuesta de Yahoo (en minúsculas).
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "open": ("open",),
    "high": ("high",),
    "low": ("low",),
    "close": ("close",),
    "volume": ("volume",),
    "adj_close": ("adj close", "adj_close", "adjclose"),
}

#: Estados finales según el error que agotó la cadena de intentos.
_STATUS_BY_ERROR: tuple[tuple[type[SourceError], SourceStatus], ...] = (
    (SourceRateLimitedError, SourceStatus.RATE_LIMITED),
    (SourceUnavailableError, SourceStatus.UNAVAILABLE),
)

_STRATEGY = Callable[[FetchRequest], object]


class YFinanceAdapter(SourceAdapter):
    """Descarga índice, futuros, ETFs, divisas y materias primas desde Yahoo.

    Parameters
    ----------
    max_attempts:
        Reintentos por estrategia ante error de red o límite de peticiones. El
        mínimo es 5 (A4).
    backoff_seconds:
        Base del backoff exponencial. ``0`` en tests para no dormir.
    """

    name = "yfinance"

    def __init__(self, *, max_attempts: int = 5, backoff_seconds: float = 1.0) -> None:
        if max_attempts < 5:
            raise ValueError("A4 exige cinco intentos como mínimo ante 429 o error de red")
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._attempts_used = 0

    # ── Interfaz ─────────────────────────────────────────────────────────────
    def fetch(self, request: FetchRequest) -> FetchResult:
        """Intenta las dos configuraciones de descarga y declara el resultado."""
        spec = request.spec
        attempts = 0
        notes: list[str] = []
        last_error: SourceError | None = None

        for label, strategy in self._strategies(spec):
            try:
                raw, used = self._call_with_retries(strategy, request)
                attempts += used
                frame = _normalize(raw, spec=spec)
            except SourceError as error:
                attempts += error.attempts
                last_error = error
                notes.append(f"{label}: {type(error).__name__}: {error}")
                continue

            if frame.height:
                return FetchResult(
                    spec=spec,
                    source=self.name,
                    status=SourceStatus.OK,
                    frame=frame,
                    attempts=max(attempts, 1),
                    notes=tuple(notes),
                )
            notes.append(f"{label}: respuesta sin filas")
            last_error = SourceUnavailableError(
                f"{spec.series_id}: {label} no devolvió ninguna fila",
                source=self.name,
                series_id=spec.series_id,
                attempts=max(used, 1),
            )

        status = _status_for(last_error)
        return FetchResult(
            spec=spec,
            source=self.name,
            status=status,
            frame=empty_canonical_frame(),
            attempts=max(attempts, 1),
            notes=tuple(notes),
            error=str(last_error) if last_error is not None else "sin datos de la fuente",
        )

    # ── Estrategias de descarga ──────────────────────────────────────────────
    def _strategies(self, spec: SeriesSpec) -> list[tuple[str, _STRATEGY]]:
        """Las dos configuraciones de descarga, en orden de intento."""
        if spec.granularity == "daily":
            return [
                ("yf.download", self._download_daily),
                ("Ticker.history", self._history_daily),
            ]
        return [
            ("yf.download", self._download_intraday),
            ("Ticker.history", self._history_intraday),
        ]

    def _download_daily(self, request: FetchRequest) -> object:
        """Vía normal para barras diarias."""
        start = request.start or request.spec.min_start
        # `yfinance` no publica firmas completas: el límite de la librería se aísla aquí.
        return yf.download(  # pyright: ignore
            tickers=request.spec.series_id,
            start=None if start is None else start.isoformat(),
            end=None if request.end is None else request.end.isoformat(),
            interval=request.spec.interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

    def _history_daily(self, request: FetchRequest) -> object:
        """Segunda vía para barras diarias."""
        start = request.start or request.spec.min_start
        return yf.Ticker(request.spec.series_id).history(  # pyright: ignore
            start=None if start is None else start.isoformat(),
            end=None if request.end is None else request.end.isoformat(),
            interval=request.spec.interval,
            auto_adjust=False,
        )

    def _download_intraday(self, request: FetchRequest) -> object:
        """Vía normal para barras intradía, con la ventana rodante declarada."""
        period = request.spec.lookback_period or "5d"
        return yf.download(  # pyright: ignore
            tickers=request.spec.series_id,
            period=period,
            interval=request.spec.interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

    def _history_intraday(self, request: FetchRequest) -> object:
        """Segunda vía para barras intradía."""
        period = request.spec.lookback_period or "5d"
        return yf.Ticker(request.spec.series_id).history(  # pyright: ignore
            period=period,
            interval=request.spec.interval,
            auto_adjust=False,
        )

    # ── Reintentos ───────────────────────────────────────────────────────────
    def _call_with_retries(self, strategy: _STRATEGY, request: FetchRequest) -> tuple[object, int]:
        """Ejecuta una estrategia con backoff; al agotar, tipa el fallo."""
        retrying = Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(
                multiplier=self._backoff_seconds,
                min=self._backoff_seconds,
                max=max(self._backoff_seconds, 8.0),
            ),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        try:
            for attempt in retrying:
                with attempt:
                    self._attempts_used = attempt.retry_state.attempt_number
                    return strategy(request), self._attempts_used
        except Exception as error:
            raise _as_source_error(
                error,
                source=self.name,
                series_id=request.spec.series_id,
                attempts=self._attempts_used,
            ) from error
        raise SourceUnavailableError(  # pragma: no cover - defensivo
            f"{request.spec.series_id}: la estrategia no devolvió nada",
            source=self.name,
            series_id=request.spec.series_id,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Traducción de errores
# ─────────────────────────────────────────────────────────────────────────────
def _as_source_error(
    error: Exception, *, source: str, series_id: str, attempts: int
) -> SourceError:
    """Traduce cualquier fallo de ``yfinance`` a un error tipado del proyecto."""
    message = f"{series_id}: {type(error).__name__}: {error}"
    lowered = message.lower()
    if "429" in lowered or "too many requests" in lowered or "rate limit" in lowered:
        return SourceRateLimitedError(
            message, source=source, series_id=series_id, attempts=attempts
        )
    return SourceUnavailableError(message, source=source, series_id=series_id, attempts=attempts)


def _status_for(error: SourceError | None) -> SourceStatus:
    """Estado declarado que corresponde al último error de la cadena."""
    if error is None:
        return SourceStatus.UNAVAILABLE
    for error_type, status in _STATUS_BY_ERROR:
        if isinstance(error, error_type):
            return status
    return SourceStatus.ERROR


# ─────────────────────────────────────────────────────────────────────────────
# Normalización
# ─────────────────────────────────────────────────────────────────────────────
def _normalize(raw: object, *, spec: SeriesSpec) -> pl.DataFrame:
    """Convierte la respuesta de Yahoo en el frame canónico, en UTC."""
    if not isinstance(raw, pd.DataFrame):
        raise SourceUnavailableError(
            f"{spec.series_id}: yfinance devolvió {type(raw).__name__}, no un DataFrame",
            source=YFinanceAdapter.name,
            series_id=spec.series_id,
        )
    if raw.empty:
        return empty_canonical_frame()

    columns = _column_map(_column_names(raw))
    missing = {"open", "high", "low", "close"} - set(columns)
    if missing:
        raise SourceUnavailableError(
            f"{spec.series_id}: la respuesta no trae las columnas {sorted(missing)}",
            source=YFinanceAdapter.name,
            series_id=spec.series_id,
        )

    timestamps = _index_values(raw)
    records = _row_dicts(raw)
    if len(timestamps) != len(records):
        raise SourceUnavailableError(
            f"{spec.series_id}: índice y filas no cuadran ({len(timestamps)} vs {len(records)})",
            source=YFinanceAdapter.name,
            series_id=spec.series_id,
        )

    is_daily = spec.granularity == "daily"
    payload = ("open", "high", "low", "close", "volume", "adj_close")
    data: dict[str, list[object]] = {name: [] for name in (*payload, "as_of", "bid", "ask")}

    for position, timestamp in enumerate(timestamps):
        as_of = _as_of(timestamp, daily=is_daily)
        if as_of is None:
            continue
        row = _as_mapping(records[position])
        if row is None:
            continue
        data["as_of"].append(as_of)
        for name in payload:
            key = columns.get(name)
            data[name].append(None if key is None else _number(row.get(key)))
        # Yahoo no publica bid/ask de esta serie: nulo, nunca inventado (A8/A19).
        data["bid"].append(None)
        data["ask"].append(None)

    return _frame(data, interval=None if is_daily else spec.interval)


def _as_of(value: object, *, daily: bool) -> datetime | None:
    """Instante del dato: cierre de sesión si es diaria, cierre de barra si no."""
    if daily:
        day = _day_of(value)
        return None if day is None else session_close_utc(day)
    return to_utc(value)


def _day_of(value: object) -> date | None:
    """Fecha de una marca temporal de Yahoo (o ``None`` si no es interpretable)."""
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


def _column_map(names: list[object]) -> dict[str, object]:
    """Mapea nombre canónico → **clave real** de la respuesta.

    La clave se devuelve sin aplanar a propósito: cuando ``yfinance`` responde
    con columnas jerárquicas (``("Close", "^GSPC")``) las filas de
    ``to_dict(orient="records")`` vienen indexadas por esa tupla, no por
    ``"Close"``. Aplanarla aquí perdería la fila entera.
    """
    resolved: dict[str, object] = {}
    for raw_name in names:
        plain = _plain_name(raw_name)
        if plain is None:
            continue
        lowered = plain.strip().lower()
        for canonical, aliases in _COLUMN_ALIASES.items():
            if lowered in aliases and canonical not in resolved:
                resolved[canonical] = raw_name
    return resolved


def _plain_name(raw_name: object) -> str | None:
    """Nombre de columna aplanado: descarta el nivel de ticker si es jerárquico."""
    if isinstance(raw_name, str):
        return raw_name
    if isinstance(raw_name, tuple) and raw_name:
        first: object = raw_name[0]  # pyright: ignore
        return first if isinstance(first, str) else None
    return None


def _column_names(frame: pd.DataFrame) -> list[object]:
    """Nombres de columna de la respuesta, planos o jerárquicos (``MultiIndex``)."""
    return list(frame.columns)  # pyright: ignore


def _index_values(frame: pd.DataFrame) -> list[object]:
    """Marcas temporales del índice, tal y como las devuelve la fuente."""
    return list(frame.index)  # pyright: ignore


def _row_dicts(frame: pd.DataFrame) -> list[dict[object, object]]:
    """Filas como mappings, para no depender de la API de ``pandas`` por celda."""
    return cast("list[dict[object, object]]", frame.to_dict(orient="records"))  # pyright: ignore


def _as_mapping(record: object) -> dict[object, object] | None:
    """Fila de ``to_dict`` como mapping, conservando las claves originales."""
    if not isinstance(record, dict):
        return None
    return cast("dict[object, object]", record)


def _number(value: object) -> float | None:
    """Número de la fuente, o ``None`` si es nulo, ``NaN`` o no numérico."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return None if number != number else number
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        return None if number != number else number
    return None


def _frame(data: dict[str, list[object]], *, interval: str | None) -> pl.DataFrame:
    """Construye el frame canónico con el esquema explícito y ordenado por ``as_of``."""
    if not data["as_of"]:
        return empty_canonical_frame()
    names = ["as_of", "open", "high", "low", "close", "volume", "adj_close", "bid", "ask"]
    frame = pl.DataFrame({name: data[name] for name in names})
    frame = frame.with_columns(
        pl.col("as_of").cast(pl.Datetime("us", "UTC")),
        *(
            pl.col(name).cast(pl.Float64())
            for name in ("open", "high", "low", "close", "volume", "adj_close", "bid", "ask")
        ),
    )
    if interval is not None:
        frame = frame.with_columns(pl.lit(interval).alias("interval"))
    return frame.sort("as_of")
