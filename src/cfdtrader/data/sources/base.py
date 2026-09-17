"""Interfaz común de los adaptadores de fuente (tarea #3).

Contrato que cumplen **todos** los adaptadores:

- Una fuente frágil nunca se usa directamente: cada una vive detrás de su propio
  adaptador, que valida, normaliza y declara el estado del dato
  (``_docs/tech_stack.md`` §3.3.3 y §4.5).
- El cliente HTTP se **inyecta**: los adaptadores no construyen su propio
  ``httpx.Client``. Así los tests usan ``httpx.MockTransport`` y no abren red.
- Un fallo de la fuente nunca es una excepción cruda de red: se traduce a un
  error tipado (:class:`SourceRateLimitedError`, :class:`SourceBlockedError`…)
  y, en el nivel de orquestación, a un :class:`SourceStatus` declarado.
- El frame que sale de un adaptador tiene **siempre** las mismas columnas
  (:data:`CANONICAL_COLUMNS`), en UTC, con ``as_of`` como instante del barrido o
  del cierre de sesión (nunca medianoche). El adaptador no decide el esquema del
  almacén: eso lo hace :mod:`cfdtrader.data.market`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

import polars as pl

__all__ = [
    "CANONICAL_COLUMNS",
    "AssetClass",
    "FetchRequest",
    "FetchResult",
    "SeriesSpec",
    "SourceAdapter",
    "SourceBlockedError",
    "SourceError",
    "SourceRateLimitedError",
    "SourceStatus",
    "SourceUnavailableError",
]

#: Columnas del frame canónico que devuelve cualquier adaptador. ``adj_close``,
#: ``bid`` y ``ask`` pueden ser nulas (la fuente no las publica); el resto son
#: obligatorias para una fila que se vaya a escribir.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "as_of",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
    "bid",
    "ask",
)


class AssetClass(StrEnum):
    """Naturaleza del instrumento. Decide reglas de calidad (p. ej. volumen 0)."""

    INDEX = "index"
    FUTURE = "future"
    ETF = "etf"
    SECTOR = "sector"
    FX = "fx"
    COMMODITY = "commodity"
    RATE = "rate"
    CFD = "cfd"
    MACRO = "macro"


class SourceStatus(StrEnum):
    """Resultado declarado de un intento de ingesta. Nunca se adivina el estado."""

    OK = "ok"
    """La fuente entregó datos utilizables."""

    RATE_LIMITED = "rate_limited"
    """Se agotaron los reintentos ante ``429`` o errores de red."""

    BLOCKED = "blocked"
    """La respuesta no es el formato esperado (HTML, *challenge* JS, captcha)."""

    UNAVAILABLE = "unavailable"
    """La fuente respondió pero sin datos para la serie (o no hay fuente conocida)."""

    ERROR = "error"
    """Fallo no clasificable en los anteriores. Se declara, no se inventa un dato."""


class SourceError(Exception):
    """Base de los errores de fuente. Siempre lleva la fuente y los intentos."""

    def __init__(
        self,
        message: str,
        *,
        source: str,
        series_id: str | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.source = source
        self.series_id = series_id
        self.attempts = attempts


class SourceRateLimitedError(SourceError):
    """``429`` o error de red tras agotar los reintentos con backoff."""


class SourceBlockedError(SourceError):
    """La fuente devolvió algo que no es CSV/JSON (bloqueo, *challenge*, captcha)."""


class SourceUnavailableError(SourceError):
    """La fuente es alcanzable pero no entrega datos para esa serie."""


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    """Declaración de una serie a ingestar.

    Attributes
    ----------
    series_id:
        Símbolo en la fuente (``^GSPC``) o identificador lógico (``SPX500:CFD``).
    dataset:
        Dataset destino en el almacén (``market_daily``, ``market_intraday``,
        ``sectors``…).
    asset_class:
        Naturaleza del instrumento; gobierna reglas de calidad.
    granularity:
        ``daily`` o ``intraday``. El informe separa por esto (A18).
    interval:
        Cadencia de la barra tal y como la nombra la fuente (``1d``, ``5m``).
    primary:
        Fuente que se intenta primero.
    fallbacks:
        Fuentes declaradas de respaldo, en orden. **El respaldo no sustituye en
        silencio**: la fila guardada lleva el ``source`` real.
    min_start:
        Fecha mínima que la serie debería cubrir. ``span_ok`` la compara con lo
        obtenido de verdad; si no se alcanza, se declara con las fechas reales.
    lookback_period:
        Ventana a pedir a la fuente cuando es rodante (``60d`` en intradía).
    history_window_limit_days:
        Límite rodante declarado del proveedor, si lo tiene (``plan.md`` §8.1).
    supports_bid_ask:
        ``True`` solo si la fuente publica bid/ask de forma verificada.
    volume_expected:
        ``False`` para índices, donde el volumen suele llegar a 0 o nulo.
    """

    series_id: str
    dataset: str
    asset_class: AssetClass
    granularity: str
    interval: str
    primary: str
    fallbacks: tuple[str, ...] = ()
    min_start: date | None = None
    lookback_period: str | None = None
    history_window_limit_days: int | None = None
    supports_bid_ask: bool = False
    volume_expected: bool = True

    @property
    def sources(self) -> tuple[str, ...]:
        """Fuente primaria y respaldos, en orden de intento."""
        return (self.primary, *self.fallbacks)


@dataclass(frozen=True, slots=True)
class FetchRequest:
    """Lo que se le pide a un adaptador, con el instante de referencia explícito.

    ``now`` se pasa siempre desde fuera: ninguna validación ni adaptador llama a
    ``datetime.now()`` por dentro (A13), para que dos ejecuciones con el mismo
    ``now`` sean idénticas.
    """

    spec: SeriesSpec
    now: datetime
    start: date | None = None
    end: date | None = None

    def __post_init__(self) -> None:
        if self.now.tzinfo is None or self.now.utcoffset() is None:
            raise ValueError("'now' debe ser un datetime con zona horaria (UTC)")


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Resultado de un intento de fetch, con estado declarado y trazabilidad."""

    spec: SeriesSpec
    source: str
    status: SourceStatus
    frame: pl.DataFrame
    attempts: int = 1
    notes: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        """``True`` solo si el estado es ``ok``."""
        return self.status is SourceStatus.OK

    def __post_init__(self) -> None:
        if self.status is SourceStatus.OK and self.error is not None:
            raise ValueError("un resultado 'ok' no puede llevar 'error'")
        if self.status is not SourceStatus.OK and self.frame.height:
            raise ValueError("un resultado no-'ok' no puede traer filas: no se inventan datos")


class SourceAdapter(ABC):
    """Interfaz de un adaptador de fuente.

    Implementaciones: :mod:`cfdtrader.data.sources.yfinance_adapter` (índice,
    futuros, ETFs, divisas, materias primas) y
    :mod:`cfdtrader.data.sources.stooq_adapter` (histórico diario de respaldo).
    """

    #: Nombre de la fuente tal y como se escribe en la columna ``source``.
    name: str

    @abstractmethod
    def fetch(self, request: FetchRequest) -> FetchResult:
        """Descarga una serie y la devuelve normalizada, o declara por qué no pudo.

        Nunca levanta una excepción de red sin tipar: o devuelve un
        :class:`FetchResult` con estado, o un :class:`SourceError` tipado.
        """
        raise NotImplementedError
