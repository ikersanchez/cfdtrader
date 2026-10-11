"""Tests de la fuente Dukascopy (``.bi5``) y de su **contrato de cortesia** (tarea #153/#150).

Ningun test abre red: el cliente se inyecta con ``httpx.MockTransport`` y el sueno tambien, asi que
el ritmo (una peticion por sesion, un turno entre peticiones, un tope por ejecucion) se comprueba
sin
esperar 30 s ni pedirle nada a nadie. El ``.bi5`` de los tests lo **genera este fichero**: en el
repositorio no entra dato de terceros.
"""

from __future__ import annotations

import json
import lzma
import struct
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from cfdtrader.analysis import premarket_gap
from cfdtrader.data.sources import dukascopy
from cfdtrader.data.sources.base import (
    SourceBlockedError,
    SourceRateLimitedError,
    SourceUnavailableError,
)
from cfdtrader.data.sources.http import CachedHttpClient

#: Manejador de ``httpx.MockTransport``.
Handler = Callable[[httpx.Request], httpx.Response]

NOW = datetime(2026, 10, 11, 23, 0, tzinfo=UTC)

#: Sesion de verano: las 08:45 ET son las 12:45 UTC (EDT).
SUMMER_DAY = date(2024, 6, 3)
#: Sesion de invierno: las 08:45 ET son las 13:45 UTC (EST) — la hora del fichero **cambia**.
WINTER_DAY = date(2024, 1, 3)


def bi5(records: list[tuple[int, float, float]]) -> bytes:
    """Genera un ``.bi5`` **de test** (LZMA *alone*) con los ticks ``(ms, ask, bid)`` dados."""
    raw = b"".join(
        struct.pack(
            ">IIIff",
            milliseconds,
            round(ask * dukascopy.PRICE_SCALE),
            round(bid * dukascopy.PRICE_SCALE),
            1.0,
            1.0,
        )
        for milliseconds, ask, bid in records
    )
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


class Clock:
    """Reloj y sueno inyectados: el contrato de cortesia se comprueba sin esperar de verdad."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(round(seconds, 6))
        self.now += seconds


class Harness:
    """Cliente de Dukascopy con reloj simulado, que ademas apunta las URLs pedidas."""

    def __init__(
        self,
        handler: Handler,
        *,
        raw_dir: Path | None = None,
        min_interval_seconds: float = dukascopy.MIN_INTERVAL_SECONDS,
        max_requests: int | None = dukascopy.DEFAULT_MAX_REQUESTS,
    ) -> None:
        self.asked: list[str] = []
        self.clock = Clock()

        def counting(request: httpx.Request) -> httpx.Response:
            self.asked.append(str(request.url))
            return handler(request)

        self.client = dukascopy.DukascopyClient(
            CachedHttpClient(
                source=dukascopy.SOURCE,
                client=httpx.Client(transport=httpx.MockTransport(counting)),
                backoff_seconds=0.0,
            ),
            raw_dir=raw_dir,
            min_interval_seconds=min_interval_seconds,
            max_requests=max_requests,
            sleep=self.clock.sleep,
            monotonic=self.clock.monotonic,
        )


def ticks_response(records: list[tuple[int, float, float]]) -> httpx.Response:
    """Respuesta ``200`` con un ``.bi5`` de test y el content-type real de la fuente."""
    return httpx.Response(
        200,
        headers={"content-type": "application/octet-stream"},
        content=bi5(records),
    )


# ─────────────────────────────────────────────────────────────────────────────
# El formato: la ruta con el mes a cero, los precios y el recorte point-in-time
# ─────────────────────────────────────────────────────────────────────────────
def test_the_month_in_the_url_is_zero_based() -> None:
    """La trampa de este feed: ``/2024/05/03/`` es el **3 de junio** de 2024."""
    assert dukascopy.hour_url(SUMMER_DAY, 12) == (
        "https://datafeed.dukascopy.com/datafeed/USA500IDXUSD/2024/05/03/12h_ticks.bi5"
    )
    assert dukascopy.hour_url(WINTER_DAY, 13).endswith("/2024/00/03/13h_ticks.bi5")


def test_the_prices_are_scaled_and_the_ticks_ordered() -> None:
    """Los enteros del fichero son precios con 3 decimales, y salen ordenados por instante."""
    hour_start = datetime(2024, 6, 3, 12, tzinfo=UTC)
    payload = bi5([(60_000, 5284.269, 5284.2), (0, 5284.0, 5283.9)])

    ticks = dukascopy.decode_ticks(payload, hour_start=hour_start)

    assert [tick.instant for tick in ticks] == [
        datetime(2024, 6, 3, 12, 0, tzinfo=UTC),
        datetime(2024, 6, 3, 12, 1, tzinfo=UTC),
    ]
    assert ticks[0].ask == pytest.approx(5284.0)
    assert ticks[0].bid == pytest.approx(5283.9)
    assert ticks[1].ask == pytest.approx(5284.269)


def test_an_empty_hour_is_a_declared_absence_and_not_an_error() -> None:
    """Sin cuerpo no hay ticks: se declara la ausencia, no se inventa un valor."""
    assert dukascopy.decode_ticks(b"", hour_start=datetime(2024, 6, 3, 12, tzinfo=UTC)) == ()


def test_a_payload_that_is_not_a_multiple_of_a_tick_fails_typed() -> None:
    """Si el cuerpo no encaja con el formato, se declara en vez de leer basura."""
    broken = lzma.compress(b"\x01\x02\x03", format=lzma.FORMAT_ALONE)

    with pytest.raises(SourceUnavailableError):
        dukascopy.decode_ticks(broken, hour_start=datetime(2024, 6, 3, 12, tzinfo=UTC))


def test_the_snapshot_instant_follows_the_dst() -> None:
    """08:45 ET son 12:45 UTC en verano y 13:45 UTC en invierno: lo resuelve ``zoneinfo``."""
    assert dukascopy.snapshot_instant(SUMMER_DAY) == datetime(2024, 6, 3, 12, 45, tzinfo=UTC)
    assert dukascopy.snapshot_instant(WINTER_DAY) == datetime(2024, 1, 3, 13, 45, tzinfo=UTC)


def test_the_premarket_tick_ignores_whatever_comes_after_the_snapshot() -> None:
    """Point-in-time: un tick posterior a las 08:45 ET seria *look-ahead* y no se usa."""
    snapshot = dukascopy.snapshot_instant(SUMMER_DAY)
    before = dukascopy.Tick(instant=snapshot, ask=100.0, bid=99.0)
    after = dukascopy.Tick(instant=snapshot.replace(minute=46), ask=200.0, bid=199.0)

    assert dukascopy.premarket_tick([before, after], snapshot) is before
    assert dukascopy.premarket_tick([after], snapshot) is None
    assert dukascopy.premarket_tick([], snapshot) is None


# ─────────────────────────────────────────────────────────────────────────────
# El contrato de cortesia: una peticion por sesion, con turno, tope y archivo
# ─────────────────────────────────────────────────────────────────────────────
def test_one_request_per_session_and_the_hour_follows_the_dst() -> None:
    """**Una** peticion por sesion, y es la hora UTC que contiene las 08:45 ET (12h o 13h)."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return ticks_response([(0, 100.0, 99.0)])

    summer = Harness(handler)
    tick, note = summer.client.session_tick(SUMMER_DAY, now=NOW)
    assert tick is not None
    assert tick.bid == pytest.approx(99.0) and note == ""
    assert len(seen) == 1 and seen[0].endswith("/2024/05/03/12h_ticks.bi5")

    seen.clear()
    winter = Harness(handler)
    winter.client.session_tick(WINTER_DAY, now=NOW)
    assert len(seen) == 1 and seen[0].endswith("/2024/00/03/13h_ticks.bi5")


def test_the_turn_is_respected_between_two_network_requests() -> None:
    """Dos peticiones de red seguidas se separan por el turno, sea cual sea su valor por defecto."""
    harness = Harness(lambda request: ticks_response([(0, 100.0, 99.0)]))

    harness.client.hour_ticks(SUMMER_DAY, 12, now=NOW)
    harness.client.hour_ticks(SUMMER_DAY, 13, now=NOW)

    assert harness.client.requests == 2
    assert harness.clock.slept == [dukascopy.MIN_INTERVAL_SECONDS]


def test_the_default_turn_is_the_fastest_rate_adopted() -> None:
    """El turno por defecto es un numero **declarado**, no uno que aparezca por accidente.

    Se fijo en 1 s el 2026-10-11 (decision del propietario): el backfill de ~3 000 sesiones pasa de
    ~25 h a ~50 min. Si alguien lo sube o lo baja, este test lo dice.
    """
    assert dukascopy.MIN_INTERVAL_SECONDS == 1.0
    assert f"{dukascopy.MIN_INTERVAL_SECONDS:.0f}s" in dukascopy.USER_AGENT
    assert dukascopy.DEFAULT_MAX_REQUESTS >= 3000


def test_the_local_archive_avoids_the_network_and_the_wait(tmp_path: Path) -> None:
    """Lo ya bajado **no se vuelve a pedir**: ni una peticion, ni un segundo de turno."""
    first = Harness(lambda request: ticks_response([(0, 100.0, 99.0)]), raw_dir=tmp_path)
    first.client.hour_ticks(SUMMER_DAY, 12, now=NOW)

    def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no debe pedirse nada: la hora ya esta en el archivo local")

    again = Harness(forbidden, raw_dir=tmp_path)
    ticks = again.client.hour_ticks(SUMMER_DAY, 12, now=NOW)

    assert len(ticks) == 1 and ticks[0].bid == pytest.approx(99.0)
    assert again.client.requests == 0
    assert again.client.from_archive == 1
    assert again.clock.slept == []


def test_the_budget_stops_the_run_with_a_typed_error() -> None:
    """Agotado el tope, el run **para** (y se retoma despues): no se convierte en un barrido."""
    harness = Harness(
        lambda request: ticks_response([(0, 100.0, 99.0)]), min_interval_seconds=0.0, max_requests=1
    )

    harness.client.hour_ticks(SUMMER_DAY, 12, now=NOW)
    with pytest.raises(dukascopy.DukascopyBudgetExceededError):
        harness.client.hour_ticks(SUMMER_DAY, 13, now=NOW)


def test_a_rate_limited_source_stops_the_run_instead_of_hammering() -> None:
    """Un ``429``/``503`` es la senal de ir mas despacio: el run **aborta**, no insiste."""
    calls: list[str] = []

    def limited(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(503, content=b"")

    harness = Harness(limited, min_interval_seconds=0.0)

    with pytest.raises(dukascopy.DukascopyError, match="limite de peticiones") as failure:
        harness.client.hour_ticks(SUMMER_DAY, 12, now=NOW)

    assert isinstance(failure.value.__cause__, SourceRateLimitedError)
    # Los cinco intentos del cliente (A4) y **ni uno mas**: no hay bucle propio.
    assert len(calls) == 5


def test_an_empty_response_is_archived_as_a_declared_absence(tmp_path: Path) -> None:
    """Una hora sin datos se declara (no es un ``0``) y ademas se archiva, para no repreguntar."""
    empty = httpx.Response(200, headers={"content-type": "application/octet-stream"}, content=b"")
    harness = Harness(lambda request: empty, raw_dir=tmp_path)

    assert harness.client.hour_ticks(SUMMER_DAY, 12, now=NOW) == ()
    marker = tmp_path / dukascopy.INSTRUMENT / "2024" / "06" / "03" / "12h_ticks.empty"
    assert marker.is_file()

    def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError("la ausencia ya esta archivada: no se repregunta")

    again = Harness(forbidden, raw_dir=tmp_path)
    assert again.client.hour_ticks(SUMMER_DAY, 12, now=NOW) == ()
    assert again.client.requests == 0


def test_the_snapshot_falls_back_one_hour_and_says_so() -> None:
    """Si la hora del snapshot no trae tick, se prueba **una** hora antes y se declara."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "/12h_ticks.bi5" in str(request.url):
            return httpx.Response(
                200, headers={"content-type": "application/octet-stream"}, content=b""
            )
        return ticks_response([(0, 100.0, 99.0)])

    harness = Harness(handler)
    tick, note = harness.client.session_tick(SUMMER_DAY, now=NOW)

    assert tick is not None
    assert "hora anterior" in note
    assert harness.client.requests == 2


def test_the_user_agent_identifies_the_project() -> None:
    """El proveedor tiene que poder saber quien pide, para poder decir que pare."""
    agents: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        agents.append(request.headers.get("user-agent", ""))
        return ticks_response([(0, 100.0, 99.0)])

    Harness(handler).client.hour_ticks(SUMMER_DAY, 12, now=NOW)

    assert agents and "cfdtrader" in agents[0]


# ─────────────────────────────────────────────────────────────────────────────
# El formato de salida y el CLI (los dos, sin red)
# ─────────────────────────────────────────────────────────────────────────────
def test_the_bars_round_trip_through_the_141_format(tmp_path: Path) -> None:
    """Las barras salen con el formato de #141: quien re-mida no necesita conversion."""
    target = tmp_path / "bars.json"
    dukascopy.write_bars([(datetime(2024, 6, 3, 12, 45, tzinfo=UTC), 5284.269)], target)

    bars = premarket_gap.load_bars(target)

    assert len(bars) == 1
    assert bars[0].instant == datetime(2024, 6, 3, 12, 45, tzinfo=UTC)
    assert bars[0].price == pytest.approx(5284.269)
    assert json.loads(target.read_text(encoding="utf-8"))[0]["price"] == 5284.269


def test_business_days_skip_the_weekend_and_reject_an_inverted_range() -> None:
    """De lunes a viernes; un rango invertido es un error de argumentos, no una lista vacia."""
    assert dukascopy.business_days(date(2024, 6, 1), date(2024, 6, 4)) == (
        date(2024, 6, 3),
        date(2024, 6, 4),
    )
    with pytest.raises(dukascopy.DukascopyError):
        dukascopy.business_days(date(2024, 6, 10), date(2024, 6, 4))


def test_the_cli_rejects_bad_arguments_before_any_request(tmp_path: Path) -> None:
    """Los argumentos se validan **antes** de construir nada que pueda pedir: codigo 2."""
    base = [
        "--raw-dir",
        str(tmp_path / "raw"),
        "--bars-out",
        str(tmp_path / "bars.json"),
        "--end",
        "2024-06-04",
    ]

    assert (
        dukascopy.main([*base, "--start", "no-es-fecha", "--now", "2026-10-11T23:00:00+00:00"]) == 2
    )
    assert dukascopy.main([*base, "--start", "2024-06-10", "--now", "2026-10-11T23:00:00"]) == 2
    assert (
        dukascopy.main([*base, "--start", "2024-06-10", "--now", "2026-10-11T23:00:00+00:00"]) == 2
    )
    assert not (tmp_path / "bars.json").exists()


def test_the_octet_stream_of_the_feed_is_an_accepted_content_type() -> None:
    """Regresion: el ``.bi5`` llega como ``application/octet-stream`` y **no** es un bloqueo."""

    def handler(request: httpx.Request) -> httpx.Response:
        return ticks_response([(0, 100.0, 99.0)])

    client = CachedHttpClient(
        source=dukascopy.SOURCE,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        backoff_seconds=0.0,
    )
    response = client.get(dukascopy.hour_url(SUMMER_DAY, 12), now=NOW)

    ticks = dukascopy.decode_ticks(
        response.response.content, hour_start=datetime(2024, 6, 3, 12, tzinfo=UTC)
    )
    assert len(ticks) == 1


def test_a_blocked_page_is_not_read_as_ticks() -> None:
    """Si la fuente contesta una pagina en vez del fichero, se declara el bloqueo, no se adivina."""
    html = httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>no</html>")
    harness = Harness(lambda request: html, min_interval_seconds=0.0)

    with pytest.raises(SourceBlockedError):
        harness.client.hour_ticks(SUMMER_DAY, 12, now=NOW)


# ─────────────────────────────────────────────────────────────────────────────
# Los argumentos que protegen el ritmo, y el camino completo del CLI
# ─────────────────────────────────────────────────────────────────────────────
def test_the_rate_limits_cannot_be_switched_off_by_argument() -> None:
    """El ritmo es la mitigacion: un turno negativo o un tope de cero son errores de argumentos."""
    client = CachedHttpClient(
        source=dukascopy.SOURCE,
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))),
    )
    with pytest.raises(dukascopy.DukascopyError, match="intervalo minimo"):
        dukascopy.DukascopyClient(client, min_interval_seconds=-1.0)
    with pytest.raises(dukascopy.DukascopyError, match="tope de peticiones"):
        dukascopy.DukascopyClient(client, max_requests=0)
    with pytest.raises(dukascopy.DukascopyError, match="entre 0 y 23"):
        dukascopy.hour_url(SUMMER_DAY, 24)


def test_the_cli_builds_bars_and_declares_the_absences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """El camino completo del CLI, con el transporte simulado: barras, ausencia y nota."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/03/12h_ticks.bi5" in url:  # lunes: el tick del snapshot
            return ticks_response([(0, 100.0, 99.0)])
        if "/05/11h_ticks.bi5" in url:  # miercoles: la hora del snapshot viene vacia
            return ticks_response([(0, 100.0, 99.0)])
        return httpx.Response(
            200, headers={"content-type": "application/octet-stream"}, content=b""
        )

    def fake_client(*, source: str, cache_root: Path | None = None) -> CachedHttpClient:
        return CachedHttpClient(
            source=source,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            cache_root=cache_root,
            backoff_seconds=0.0,
        )

    monkeypatch.setattr(dukascopy, "CachedHttpClient", fake_client)
    bars_out = tmp_path / "bars.json"
    code = dukascopy.main(
        [
            "--raw-dir",
            str(tmp_path / "raw"),
            "--bars-out",
            str(bars_out),
            "--start",
            "2024-06-03",
            "--end",
            "2024-06-05",
            "--now",
            "2026-10-11T23:00:00+00:00",
            "--min-interval",
            "0",
            "--max-requests",
            "0",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert code == 0
    assert summary["sessions_asked"] == 3
    assert summary["bars"] == 2  # lunes y miercoles (por la hora anterior)
    assert summary["absences"] == 1  # martes: sin ningun tick
    assert summary["notes"] == 1  # miercoles: se fue a la hora anterior, y se dice
    assert (
        summary["requests"] == 5
    )  # 1 (lunes) + 2 (martes, sin datos en ninguna hora) + 2 (miercoles)
    assert len(premarket_gap.load_bars(bars_out)) == 2


def test_the_cli_stops_with_code_3_when_the_source_asks_to_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Un limite de peticiones corta el backfill: codigo 3 y lo bajado queda archivado."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"")

    def fake_client(*, source: str, cache_root: Path | None = None) -> CachedHttpClient:
        return CachedHttpClient(
            source=source,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            backoff_seconds=0.0,
        )

    monkeypatch.setattr(dukascopy, "CachedHttpClient", fake_client)
    bars_out = tmp_path / "bars.json"
    code = dukascopy.main(
        [
            "--raw-dir",
            str(tmp_path / "raw"),
            "--bars-out",
            str(bars_out),
            "--start",
            "2024-06-03",
            "--end",
            "2024-06-03",
            "--now",
            "2026-10-11T23:00:00+00:00",
            "--min-interval",
            "0",
        ]
    )

    assert code == 3
    assert "se detiene el backfill" in capsys.readouterr().err
    assert not bars_out.exists()  # la verdad durable es el archivo local, no un informe a medias
