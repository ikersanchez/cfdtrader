"""Tests de las fuentes de datos: registro, HTTP, adaptadores (tarea #3).

Ningún test abre red: el cliente HTTP se inyecta con ``httpx.MockTransport`` y
``yfinance`` se sustituye con ``monkeypatch`` (A2). Todo escribe bajo
``tmp_path``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
import polars as pl
import pytest
import yfinance as yf

from cfdtrader.data.sources.base import (
    AssetClass,
    FetchRequest,
    FetchResult,
    SeriesSpec,
    SourceBlockedError,
    SourceRateLimitedError,
    SourceStatus,
)
from cfdtrader.data.sources.http import CachedHttpClient
from cfdtrader.data.sources.registry import cfd_substitutions, load_registry
from cfdtrader.data.sources.stooq_adapter import StooqAdapter, stooq_symbols
from cfdtrader.data.sources.yfinance_adapter import YFinanceAdapter

FIXTURES = Path(__file__).parent / "fixtures"

NOW = datetime(2024, 6, 10, 18, 0, tzinfo=UTC)

#: Manejador de ``httpx.MockTransport``.
Handler = Callable[[httpx.Request], httpx.Response]

CSV_BODY = (
    b"Date,Open,High,Low,Close,Volume\n"
    b"2024-06-07,5350.0,5360.0,5340.0,5355.0,1000\n"
    b"2024-06-10,5360.0,5370.0,5350.0,5365.0,1100\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def _spec(series_id: str, dataset: str) -> SeriesSpec:
    """Especificación de serie tomada del registro real (no una copia a mano)."""
    registry = load_registry()
    for spec in registry.series:
        if spec.series_id == series_id and spec.dataset == dataset:
            return spec
    raise AssertionError(f"{series_id!r} no está declarada en {dataset}")


def _client(handler: Handler, *, cache_root: Path | None = None) -> CachedHttpClient:
    """Cliente con transporte simulado y sin backoff (los tests no duermen)."""
    return CachedHttpClient(
        source="stooq",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        cache_root=cache_root,
        backoff_seconds=0.0,
    )


def _returning(frame: pd.DataFrame) -> Callable[..., pd.DataFrame]:
    """Doble de una función de yfinance que devuelve siempre ese frame."""

    def fake(*args: object, **kwargs: object) -> pd.DataFrame:
        return frame

    return fake


def _raising(message: str, log: list[str]) -> Callable[..., pd.DataFrame]:
    """Doble que falla siempre, contando las llamadas."""

    def fake(*args: object, **kwargs: object) -> pd.DataFrame:
        log.append(message)
        raise RuntimeError(message)

    return fake


def _ticker_returning(frame: pd.DataFrame, log: list[str]) -> type:
    """Doble de ``yf.Ticker`` cuya ``history`` devuelve siempre ese frame."""

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **kwargs: object) -> pd.DataFrame:
            log.append("history")
            return frame

    return FakeTicker


def _ticker_raising(log: list[str]) -> type:
    """Doble de ``yf.Ticker`` cuya ``history`` falla siempre."""

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **kwargs: object) -> pd.DataFrame:
            log.append("history")
            raise RuntimeError("429 Too Many Requests")

    return FakeTicker


def _daily_frame() -> pd.DataFrame:
    """Respuesta diaria de Yahoo, con las dos fechas del criterio A9."""
    return pd.DataFrame(
        {
            "Open": [5150.0, 4780.0],
            "High": [5170.0, 4800.0],
            "Low": [5140.0, 4770.0],
            "Close": [5160.0, 4790.0],
            "Adj Close": [5160.0, 4790.0],
            "Volume": [1_000_000.0, 900_000.0],
        },
        index=pd.DatetimeIndex(["2024-03-15", "2024-01-15"]),
    )


def _intraday_frame(day: str) -> pd.DataFrame:
    """Respuesta de barras de 5m con la apertura y el cierre de sesión."""
    index = pd.DatetimeIndex([f"{day} 09:30", f"{day} 16:00"], tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": [5300.0, 5350.0],
            "High": [5310.0, 5360.0],
            "Low": [5290.0, 5340.0],
            "Close": [5305.0, 5355.0],
            "Volume": [5000.0, 6000.0],
        },
        index=index,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registro (A3, A6, A19)
# ─────────────────────────────────────────────────────────────────────────────
def test_registry_declares_every_required_series() -> None:
    """A6: diario del índice, futuro, ETF, VIX, los 11 sectores, contexto e intradía 5m."""
    registry = load_registry()
    daily = {spec.series_id for spec in registry.series if spec.granularity == "daily"}
    intraday = {spec.series_id for spec in registry.series if spec.granularity == "intraday"}
    sectors = {spec.series_id for spec in registry.series if spec.asset_class is AssetClass.SECTOR}

    assert {"^GSPC", "ES=F", "SPY", "^VIX"} <= daily
    assert sectors == {
        "XLK",
        "XLF",
        "XLE",
        "XLV",
        "XLY",
        "XLP",
        "XLI",
        "XLU",
        "XLB",
        "XLRE",
        "XLC",
    }
    assert {"^GDAXI", "^FTSE", "^STOXX50E", "^N225", "^HSI"} <= daily
    assert {"DX-Y.NYB", "EURUSD=X", "BZ=F", "CL=F", "GC=F"} <= daily
    assert intraday == {"^GSPC", "ES=F"}
    assert all(spec.interval == "5m" for spec in registry.series if spec.granularity == "intraday")
    assert all(
        spec.min_start is not None for spec in registry.series if spec.granularity == "daily"
    )
    assert _spec("^GSPC", "market_daily").min_start == date(2005, 1, 1)


def test_cfd_has_no_alias_to_index_or_future() -> None:
    """A19: `SPX500:CFD` no se mapea a `^GSPC`, `ES=F` ni `SPY`, en ningún sitio."""
    registry = load_registry()

    assert cfd_substitutions(registry, {"stooq": stooq_symbols(), "yfinance": {}}) == []
    assert all(spec.series_id != "SPX500:CFD" for spec in registry.series)
    assert all(spec.asset_class is not AssetClass.CFD for spec in registry.series)

    cfd = {item.series_id: item for item in registry.unavailable}["SPX500:CFD"]
    assert cfd.status == "unavailable"
    assert cfd.bid_ask is False
    assert cfd.reason and cfd.checked_on == date(2026, 9, 17)
    assert cfd.follow_up_issue == 50

    # Si alguien introduce el alias, el guardián lo dice con nombre y apellido.
    broken = cfd_substitutions(registry, {"stooq": {**stooq_symbols(), "SPX500:CFD": "^spx"}})
    assert broken and "^spx" in broken[0]


def test_stooq_has_no_symbol_for_the_cfd() -> None:
    """Ni el mapa de símbolos del adaptador de respaldo puede nombrar el CFD."""
    assert "SPX500:CFD" not in stooq_symbols()


# ─────────────────────────────────────────────────────────────────────────────
# Cliente HTTP (A4, A5)
# ─────────────────────────────────────────────────────────────────────────────
def test_rate_limit_retries_five_times_then_typed_error() -> None:
    """A4(i): cinco peticiones ante 429 y excepción tipada, no un error HTTP crudo."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, json={"error": "slow down"})

    client = _client(handler)
    with pytest.raises(SourceRateLimitedError) as failure:
        client.get("https://stooq.com/q/d/l/", params={"s": "^spx", "i": "d"})

    assert calls == 5
    assert failure.value.attempts == 5
    assert failure.value.source == "stooq"


def test_js_challenge_is_blocked_with_evidence() -> None:
    """A4(ii): un *challenge* HTML es `SourceBlockedError`, con content-type y bytes."""
    body = (FIXTURES / "stooq" / "js_challenge.html").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/html; charset=utf-8"}, content=body
        )

    client = _client(handler)
    with pytest.raises(SourceBlockedError) as failure:
        client.get("https://stooq.com/q/d/l/")

    message = str(failure.value)
    assert "text/html" in message
    assert "DOCTYPE" in message


def test_cache_avoids_the_second_request(tmp_path: Path) -> None:
    """A5: con la caché vigente no se repite la petición."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, headers={"content-type": "text/csv"}, content=CSV_BODY)

    cache_root = tmp_path / "cache"
    first = _client(handler, cache_root=cache_root).get(
        "https://stooq.com/q/d/l/?s=%5Espx&i=d", now=NOW
    )
    assert first.from_cache is False
    assert calls == 1

    def explode(request: httpx.Request) -> httpx.Response:  # pragma: no cover - no debe llamarse
        raise AssertionError("la caché vigente no debe repetir la petición")

    second = _client(explode, cache_root=cache_root).get(
        "https://stooq.com/q/d/l/?s=%5Espx&i=d", now=NOW
    )
    assert second.from_cache is True
    assert calls == 1
    assert second.text.startswith("Date,Open")


# ─────────────────────────────────────────────────────────────────────────────
# Adaptador de yfinance (A1, A2, A9)
# ─────────────────────────────────────────────────────────────────────────────
def test_daily_bar_is_anchored_to_the_session_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """A9: la barra diaria se ancla al cierre de sesión, nunca a medianoche."""
    monkeypatch.setattr(yf, "download", _returning(_daily_frame()))

    spec = _spec("^GSPC", "market_daily")
    result = YFinanceAdapter().fetch(FetchRequest(spec=spec, now=NOW, start=date(2024, 1, 1)))

    assert result.status is SourceStatus.OK
    as_of = result.frame.get_column("as_of").to_list()
    assert as_of == [
        datetime(2024, 1, 15, 21, 0, tzinfo=UTC),  # EST
        datetime(2024, 3, 15, 20, 0, tzinfo=UTC),  # EDT
    ]
    assert result.frame.get_column("bid").to_list() == [None, None]
    assert result.frame.get_column("ask").to_list() == [None, None]
    assert result.frame.get_column("adj_close").to_list() == [4790.0, 5160.0]


@pytest.mark.parametrize(
    ("day", "expected_open", "expected_close"),
    [
        (
            "2024-06-10",
            datetime(2024, 6, 10, 13, 30, tzinfo=UTC),
            datetime(2024, 6, 10, 20, 0, tzinfo=UTC),
        ),
        (
            "2024-01-10",
            datetime(2024, 1, 10, 14, 30, tzinfo=UTC),
            datetime(2024, 1, 10, 21, 0, tzinfo=UTC),
        ),
    ],
)
def test_intraday_bar_is_anchored_to_the_bar_close(
    monkeypatch: pytest.MonkeyPatch, day: str, expected_open: datetime, expected_close: datetime
) -> None:
    """A9: 09:30 y 16:00 ET se guardan como 13:30/20:00 UTC en EDT y 14:30/21:00 en EST."""
    monkeypatch.setattr(yf, "download", _returning(_intraday_frame(day)))

    spec = _spec("^GSPC", "market_intraday")
    result = YFinanceAdapter().fetch(FetchRequest(spec=spec, now=NOW))

    assert result.status is SourceStatus.OK
    assert result.frame.get_column("as_of").to_list() == [expected_open, expected_close]
    assert result.frame.get_column("interval").unique().to_list() == ["5m"]


def test_empty_answer_is_declared_unavailable_and_tries_the_second_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A3: las dos configuraciones de descarga; un vacío se declara, no se inventa."""
    calls: list[str] = []
    monkeypatch.setattr(yf, "download", _returning(pd.DataFrame()))
    monkeypatch.setattr(yf, "Ticker", _ticker_returning(pd.DataFrame(), calls))

    spec = _spec("^GSPC", "market_daily")
    result = YFinanceAdapter().fetch(FetchRequest(spec=spec, now=NOW))

    assert result.status is SourceStatus.UNAVAILABLE
    assert result.frame.height == 0
    assert result.error is not None
    # Las dos configuraciones se intentaron: `download` (vacío) y `Ticker.history`.
    assert calls == ["history"]
    assert "yf.download" in result.notes[0]
    assert any("Ticker.history" in note for note in result.notes)


def test_network_failure_is_retried_and_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A4: cinco intentos por configuración ante límite de peticiones, y estado tipado."""
    log: list[str] = []
    monkeypatch.setattr(yf, "download", _raising("429 Too Many Requests", log))
    monkeypatch.setattr(yf, "Ticker", _ticker_raising(log))

    spec = _spec("^GSPC", "market_daily")
    result = YFinanceAdapter(backoff_seconds=0.0).fetch(FetchRequest(spec=spec, now=NOW))

    assert result.status is SourceStatus.RATE_LIMITED
    assert result.frame.height == 0
    assert len(log) == 10  # cinco por cada una de las dos configuraciones
    assert result.attempts == 10


def test_a_frame_without_ohlc_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Una respuesta que no trae OHLC no se puede normalizar: se declara."""
    broken = pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex(["2024-06-07"]))
    monkeypatch.setattr(yf, "download", _returning(broken))
    monkeypatch.setattr(yf, "Ticker", _ticker_returning(broken, []))

    spec = _spec("^GSPC", "market_daily")
    result = YFinanceAdapter().fetch(FetchRequest(spec=spec, now=NOW))

    assert result.status is SourceStatus.UNAVAILABLE
    assert result.frame.height == 0


# ─────────────────────────────────────────────────────────────────────────────
# Adaptador de Stooq (A3, A4)
# ─────────────────────────────────────────────────────────────────────────────
def test_stooq_parses_csv_and_leaves_bid_ask_null() -> None:
    """El CSV de respaldo se normaliza: cierre de sesión en UTC y bid/ask nulos."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["s"] == "^spx"
        return httpx.Response(200, headers={"content-type": "text/csv"}, content=CSV_BODY)

    spec = _spec("^GSPC", "market_daily")
    result = StooqAdapter(_client(handler)).fetch(FetchRequest(spec=spec, now=NOW))

    assert result.status is SourceStatus.OK
    assert result.source == "stooq"
    assert result.frame.get_column("as_of").to_list() == [
        datetime(2024, 6, 7, 20, 0, tzinfo=UTC),
        datetime(2024, 6, 10, 20, 0, tzinfo=UTC),
    ]
    assert result.frame.get_column("bid").to_list() == [None, None]
    assert result.frame.get_column("adj_close").to_list() == [None, None]


def test_stooq_challenge_ends_as_blocked_status() -> None:
    """El *challenge* se declara como estado `blocked`, con el motivo."""
    body = (FIXTURES / "stooq" / "js_challenge.html").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=body)

    spec = _spec("^GSPC", "market_daily")
    result = StooqAdapter(_client(handler)).fetch(FetchRequest(spec=spec, now=NOW))

    assert result.status is SourceStatus.BLOCKED
    assert result.frame.height == 0
    assert result.error is not None and "text/html" in result.error


def test_stooq_only_serves_the_daily_series() -> None:
    """Stooq es respaldo del diario: para intradía declara el estado, no lo intenta."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no debe pedirse nada a Stooq para intradía")

    spec = _spec("^GSPC", "market_intraday")
    result = StooqAdapter(_client(handler)).fetch(FetchRequest(spec=spec, now=NOW))
    assert result.status is SourceStatus.UNAVAILABLE
    assert result.frame.height == 0


def test_fetch_result_refuses_rows_without_ok_status() -> None:
    """Invariante: un estado que no es `ok` no puede traer filas (no se inventan datos)."""
    spec = _spec("^GSPC", "market_daily")
    with pytest.raises(ValueError, match="no puede traer filas"):
        FetchResult(
            spec=spec,
            source="yfinance",
            status=SourceStatus.UNAVAILABLE,
            frame=pl.DataFrame({"as_of": [NOW]}),
        )
