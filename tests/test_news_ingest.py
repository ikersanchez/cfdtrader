"""Tests de la ingesta de noticias (`#30`): adaptadores, deduplicación y point-in-time.

El artefacto verificable es que ``raw.news_headlines`` queda poblado con **titular, URL,
`published_at` y hash —nunca el cuerpo—** y que **nada con `published_at` posterior al instante
de ejecución se guarda**. El parseo se prueba con payloads de GDELT y RSS fijos; no hay red.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, cast

import httpx
import pytest

from cfdtrader.data import news
from cfdtrader.data.news import deduplicate, existing_headlines, ingest, load_headlines
from cfdtrader.data.settings import ConfigurationError
from cfdtrader.data.sources.http import CachedHttpClient, CachedResponse
from cfdtrader.data.sources.news import (
    GdeltAdapter,
    Headline,
    NewsAdapterError,
    RssAdapter,
    headline_hash,
    normalize_title,
    parse_gdelt_articles,
    parse_rss_feed,
)
from cfdtrader.data.store import Store

NOW: Final[datetime] = datetime(2026, 10, 3, 12, 45, tzinfo=UTC)


def _headline(title: str, *, offset_minutes: int = 0) -> Headline:
    return Headline(
        source="rss",
        feed="reuters",
        title=title,
        url=f"https://example.invalid/{headline_hash(title)[-8:]}",
        published_at=NOW + timedelta(minutes=offset_minutes),
    )


# ─────────────────────────────────────────────────────────────────────────────
# A1 · Normalización y hash (deduplicación barata, sin *embeddings*)
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_normalized_title_and_hash_are_stable() -> None:
    """Minúsculas, sin acentos y sin puntuación; el hash ignora caja y espacios."""
    assert normalize_title("  ¡S&P 500 SUBE, 2,5%! ") == "s p 500 sube 2 5"
    assert normalize_title("S&P 500 sube 2,5%") == normalize_title("s&p 500 SUBE 2.5%")
    assert headline_hash("Hola, mundo") == headline_hash("  hola mundo  ")
    assert headline_hash("x").startswith("sha256:")


# ─────────────────────────────────────────────────────────────────────────────
# A2/A3 · Parseo de los payloads (puro, sin red)
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_gdelt_articles_are_parsed_and_incomplete_rows_are_dropped() -> None:
    """Solo las filas con título, URL y fecha válidos salen; el resto se descarta."""
    payload: dict[str, object] = {
        "articles": [
            {"title": "Fed holds rates", "url": "https://a", "seendate": "20261003T083000Z"},
            {"title": "", "url": "https://b", "seendate": "20261003T083000Z"},
            {"title": "No date", "url": "https://c"},
            {"url": "https://d", "seendate": "20261003T083000Z"},
        ]
    }

    headlines = parse_gdelt_articles(payload, feed="sp500")

    assert [headline.title for headline in headlines] == ["Fed holds rates"]
    assert headlines[0].source == "gdelt"
    assert headlines[0].feed == "sp500"
    assert headlines[0].published_at == datetime(2026, 10, 3, 8, 30, tzinfo=UTC)


def test_a2_a_payload_without_articles_is_a_typed_error() -> None:
    with pytest.raises(NewsAdapterError, match="articles"):
        parse_gdelt_articles({}, feed="query")


_RSS: Final[bytes] = (
    b'<?xml version="1.0"?><rss version="2.0"><channel><title>Feed</title>'
    b"<item><title>Stocks rally</title><link>https://x/1</link>"
    b"<pubDate>Fri, 03 Oct 2026 08:30:00 GMT</pubDate></item>"
    b"<item><title>Sin fecha</title><link>https://x/2</link></item>"
    b"</channel></rss>"
)


def test_a3_rss_items_are_parsed_from_the_feed() -> None:
    """Un RSS deja título, enlace y fecha; el ítem sin fecha se descarta."""
    headlines = parse_rss_feed(_RSS, feed="reuters")

    assert [headline.title for headline in headlines] == ["Stocks rally"]
    assert headlines[0].source == "rss"
    assert headlines[0].url == "https://x/1"
    assert headlines[0].published_at == datetime(2026, 10, 3, 8, 30, tzinfo=UTC)


# ─────────────────────────────────────────────────────────────────────────────
# A4 · Deduplicación por hash exacto y por similitud difusa
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_deduplicate_by_hash_and_fuzzy_similarity() -> None:
    """La misma noticia con otra caja es hash; con una errata, difusa. Un titular distinto pasa."""
    original = _headline("Fed holds rates steady")
    same_hash = _headline("FED HOLDS RATES STEADY!")
    fuzzy = _headline("Fed holds rate steady")
    distinct = _headline("Oil jumps on supply fears")

    new, duplicates = deduplicate([original, same_hash, fuzzy, distinct])

    assert [headline.title for headline in new] == [
        "Fed holds rates steady",
        "Oil jumps on supply fears",
    ]
    assert [headline.title for headline in duplicates] == [
        "FED HOLDS RATES STEADY!",
        "Fed holds rate steady",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# A5 · Point-in-time: nada del futuro se guarda
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_a_headline_from_the_future_is_never_stored(tmp_path: Path) -> None:
    """Un titular con `published_at` posterior a `now` se descarta y no llega al almacén."""
    store = Store(tmp_path)
    past = _headline("Noticia de ayer", offset_minutes=-30)
    future = _headline("Noticia del futuro", offset_minutes=30)

    report = ingest(store=store, headlines=[past, future], now=NOW)

    assert report.fetched == 2
    assert report.discarded_future == 1
    assert report.new == 1
    frame = store.sql("SELECT title, headline_hash FROM raw.news_headlines")
    assert set(frame.get_column("title").to_list()) == {"Noticia de ayer"}
    assert headline_hash("Noticia del futuro") not in set(
        frame.get_column("headline_hash").to_list()
    )


# ─────────────────────────────────────────────────────────────────────────────
# A6 · El esquema queda poblado sin el cuerpo del artículo (§12.7)
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_the_schema_is_populated_without_the_article_body(tmp_path: Path) -> None:
    """`raw.news_headlines` trae titular, URL y hash, y **ninguna** columna con el cuerpo."""
    store = Store(tmp_path)

    report = ingest(store=store, headlines=[_headline("Titular uno")], now=NOW)

    assert report.new == 1
    assert report.outcome == "created"
    frame = store.sql("SELECT * FROM raw.news_headlines")
    columns = set(frame.columns)
    assert {
        "title",
        "url",
        "headline_hash",
        "published_at",
        "as_of",
        "source",
        "series_id",
    } <= columns
    for forbidden in ("body", "content", "text", "summary", "article"):
        assert forbidden not in columns


# ─────────────────────────────────────────────────────────────────────────────
# A7 · Idempotencia: reingestar lo mismo no escribe nada
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_reingesting_the_same_headline_is_idempotent(tmp_path: Path) -> None:
    """Con el historial por delante, el mismo titular es duplicado y el almacén no cambia."""
    store = Store(tmp_path)
    first = ingest(store=store, headlines=[_headline("Titular uno")], now=NOW)
    assert first.new == 1

    known_hashes, known_titles = existing_headlines(store)
    assert known_hashes == frozenset({headline_hash("Titular uno")})
    second = ingest(
        store=store,
        headlines=[_headline("Titular uno")],
        now=NOW,
        known_hashes=known_hashes,
        known_titles=known_titles,
    )

    assert second.new == 0
    assert second.discarded_duplicate == 1
    assert second.outcome == "unchanged"
    frame = store.sql("SELECT title FROM raw.news_headlines")
    assert len(frame) == 1


# ─────────────────────────────────────────────────────────────────────────────
# A8 · CLI: `--as-of` y al menos una fuente
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_the_cli_requires_as_of_and_a_source(capsys: pytest.CaptureFixture[str]) -> None:
    """Sin `--as-of` o sin fuentes, la ingesta no arranca (código 2, motivo por stderr)."""
    assert news.main([]) == 2
    assert "as-of" in capsys.readouterr().err

    assert news.main(["--as-of", NOW.isoformat()]) == 2
    assert "al menos un" in capsys.readouterr().err


# ─────────────────────────────────────────────────────────────────────────────
# A9 · Adaptadores sobre un cliente HTTP inyectado (sin red)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# A13 (#35) · El lector del almacen: point-in-time y sin sorpresas
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_load_headlines_reads_the_store_and_drops_the_future(tmp_path: Path) -> None:
    """El lado de lectura de #30: lo guardado vuelve, y nada posterior a `as_of` entra."""
    store = Store(tmp_path)
    past = _headline("Titular del pasado", offset_minutes=-10)
    future = _headline("Titular del futuro", offset_minutes=+10)
    ingest(store=store, headlines=[past, future], now=NOW)

    loaded = load_headlines(store, as_of=NOW)

    assert [item.title for item in loaded] == ["Titular del pasado"]
    assert loaded[0].feed == past.feed
    assert loaded[0].published_at == past.published_at
    assert loaded[0].url == past.url


def test_a13_load_headlines_respects_the_window_and_an_empty_store(tmp_path: Path) -> None:
    store = Store(tmp_path)
    ingest(
        store=store,
        headlines=[_headline("De hace dos dias", offset_minutes=-60 * 48)],
        now=NOW,
    )

    assert load_headlines(store, as_of=NOW) == (), "fuera de la ventana de 24 h"
    assert load_headlines(store, as_of=NOW, window_hours=72) != ()
    assert load_headlines(Store(tmp_path / "vacio"), as_of=NOW) == (), "sin dataset, sin noticias"

    with pytest.raises(ConfigurationError, match="zona"):
        load_headlines(store, as_of=datetime(2026, 10, 3, 12, 45))


def _cached(*, text: str = "", content: bytes | None = None) -> CachedResponse:
    request = httpx.Request("GET", "https://example.invalid/feed")
    response = (
        httpx.Response(200, content=content, request=request)
        if content is not None
        else httpx.Response(200, text=text, request=request)
    )
    return CachedResponse(response=response, attempts=1, from_cache=False)


class _FakeClient:
    """Cliente HTTP de mentira: devuelve una respuesta pautada y registra las llamadas."""

    def __init__(self, response: CachedResponse) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, object] | None = None,
        now: datetime | None = None,
        headers: dict[str, str] | None = None,
    ) -> CachedResponse:
        self.calls.append({"url": url, "params": params})
        return self._response


def test_a9_the_gdelt_adapter_fetches_and_parses() -> None:
    """El adaptador de GDELT pide la consulta y traduce el JSON a titulares."""
    payload = json.dumps(
        {"articles": [{"title": "Fed holds", "url": "https://a", "seendate": "20261003T083000Z"}]}
    )
    client = _FakeClient(_cached(text=payload))

    headlines = GdeltAdapter(cast("CachedHttpClient", client)).fetch(query="sp500", now=NOW)

    assert [headline.title for headline in headlines] == ["Fed holds"]
    params = cast("dict[str, object]", client.calls[0]["params"])
    assert params["query"] == "sp500"


def test_a9_a_non_json_gdelt_payload_is_a_typed_error() -> None:
    """Un payload que no es JSON (p. ej. un *challenge*) es error tipado, no un `ValueError`."""
    client = _FakeClient(_cached(text="<html>challenge</html>"))

    with pytest.raises(NewsAdapterError, match="JSON"):
        GdeltAdapter(cast("CachedHttpClient", client)).fetch(query="q", now=NOW)


def test_a9_the_rss_adapter_fetches_and_parses() -> None:
    """El adaptador de RSS parsea el canal servido por el cliente inyectado."""
    client = _FakeClient(_cached(content=_RSS))

    headlines = RssAdapter(cast("CachedHttpClient", client)).fetch(
        feed_url="https://example.invalid/feed", label="reuters", now=NOW
    )

    assert [headline.title for headline in headlines] == ["Stocks rally"]


# ─────────────────────────────────────────────────────────────────────────────
# A10 · El CLI rechaza un `--as-of` inválido y un `--feed` mal formado
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_the_cli_rejects_a_bad_as_of_and_a_bad_feed(capsys: pytest.CaptureFixture[str]) -> None:
    assert news.main(["--as-of", "no-iso", "--query", "q"]) == 2
    assert "ISO-8601" in capsys.readouterr().err

    assert news.main(["--as-of", "2026-10-03T12:45:00", "--query", "q"]) == 2
    assert "zona" in capsys.readouterr().err

    assert news.main(["--as-of", NOW.isoformat(), "--feed", "sin-igual"]) == 2
    assert "etiqueta=url" in capsys.readouterr().err


# ─────────────────────────────────────────────────────────────────────────────
# A11 · El CLI cablea los adaptadores y escribe (con los adaptadores simulados)
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_main_wires_the_adapters_and_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """El `main` real recoge de GDELT + RSS (simulados) y deja el almacén poblado."""

    class _Context:
        def __enter__(self) -> _Context:
            return self

        def __exit__(self, *arguments: object) -> None:
            return None

    class _Gdelt:
        def __init__(self, client: object) -> None: ...

        def fetch(self, *, query: str, now: datetime) -> list[Headline]:
            return [_headline("Titular GDELT", offset_minutes=-1)]

    class _Rss:
        def __init__(self, client: object) -> None: ...

        def fetch(self, *, feed_url: str, label: str, now: datetime) -> list[Headline]:
            return [_headline("Titular RSS", offset_minutes=-2)]

    def _client_factory(**_kwargs: object) -> _Context:
        return _Context()

    monkeypatch.setattr(news, "CachedHttpClient", _client_factory)
    monkeypatch.setattr(news, "GdeltAdapter", _Gdelt)
    monkeypatch.setattr(news, "RssAdapter", _Rss)

    code = news.main(
        [
            "--as-of",
            NOW.isoformat(),
            "--data-root",
            str(tmp_path),
            "--query",
            "sp500",
            "--feed",
            "reuters=https://example.invalid/feed",
        ]
    )

    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["new"] == 2
    frame = Store(tmp_path).sql("SELECT title FROM raw.news_headlines")
    assert set(frame.get_column("title").to_list()) == {"Titular GDELT", "Titular RSS"}


# ─────────────────────────────────────────────────────────────────────────────
# A12 · Entradas incoherentes: `now` sin zona y filas de GDELT corruptas
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_ingest_requires_an_aware_now(tmp_path: Path) -> None:
    """Un `now` sin zona no identifica un instante: es error tipado, no un valor por defecto."""
    with pytest.raises(ConfigurationError, match="zona"):
        ingest(store=Store(tmp_path), headlines=[], now=datetime(2026, 10, 3, 12, 45))


def test_a12_gdelt_rows_without_a_valid_date_are_dropped() -> None:
    """Una fila que no es un objeto, o con fecha inválida, se descarta sin romper el parseo."""
    payload: dict[str, object] = {
        "articles": [
            "no-es-un-objeto",
            {"title": "Fecha mala", "url": "https://a", "seendate": "ayer"},
            {"title": "Sin fecha", "url": "https://b"},
        ]
    }

    assert parse_gdelt_articles(payload, feed="q") == []
