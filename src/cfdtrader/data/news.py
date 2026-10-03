"""Ingesta de titulares de noticias (`tech_stack.md` §12.4 y §12.7) — tarea #30.

Guarda en ``raw.news_headlines`` **titular + URL + `published_at` + hash**, nunca el cuerpo del
artículo (§12.7: almacenarlo puede infringir los términos de uso de las agencias, no es solo una
cuestión de disco).

- **Point-in-time**: nada con ``published_at`` **posterior** al instante de la ejecución se guarda.
- **Deduplicación barata** (§4.9): hash del titular normalizado **más** similitud difusa, sin
  *embeddings* (ni coste ni proveedor extra).
- El almacén es **inmutable**: reingestar el mismo titular es idempotente (``UNCHANGED``).
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

from pydantic import BaseModel, ConfigDict

from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.sources.http import CachedHttpClient
from cfdtrader.data.sources.news import (
    GdeltAdapter,
    Headline,
    RssAdapter,
    headline_hash,
    normalize_title,
)
from cfdtrader.data.store import Store

__all__ = [
    "DATASET",
    "DEFAULT_FUZZY_THRESHOLD",
    "NEWS_VERSION",
    "NewsReport",
    "deduplicate",
    "existing_headlines",
    "headline_records",
    "ingest",
    "main",
]

#: Dataset del almacén (la capa ``raw`` lo convierte en la vista ``raw.news_headlines``).
DATASET: Final[str] = "news_headlines"

#: Versión de la fila: la primera revisión de un titular.
NEWS_VERSION: Final[int] = 1

#: Umbral de similitud difusa: por encima, dos titulares son "la misma noticia" (§4.9).
DEFAULT_FUZZY_THRESHOLD: Final[float] = 0.9


class NewsReport(BaseModel):
    """Resumen declarado de una ingesta: qué se recogió, qué se descartó y qué se escribió."""

    model_config = ConfigDict(extra="forbid")

    as_of: str
    fetched: int
    discarded_future: int
    discarded_duplicate: int
    new: int
    outcome: str
    headline_hashes: tuple[str, ...] = ()


def _too_similar(candidate: str, existing: Sequence[str], threshold: float) -> bool:
    """``True`` si algún titular ya visto se parece lo bastante (``difflib``, sin *embeddings*)."""
    return any(
        difflib.SequenceMatcher(None, candidate, other).ratio() >= threshold for other in existing
    )


def deduplicate(
    headlines: Sequence[Headline],
    *,
    known_hashes: frozenset[str] = frozenset(),
    known_titles: Sequence[str] = (),
    threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> tuple[list[Headline], list[Headline]]:
    """Separa ``(nuevos, duplicados)`` por hash normalizado **exacto** y por similitud difusa."""
    seen_hashes = set(known_hashes)
    seen_titles = [normalize_title(title) for title in known_titles]
    fresh: list[Headline] = []
    duplicates: list[Headline] = []
    for headline in headlines:
        normalized = normalize_title(headline.title)
        if headline_hash(headline.title) in seen_hashes or _too_similar(
            normalized, seen_titles, threshold
        ):
            duplicates.append(headline)
            continue
        fresh.append(headline)
        seen_hashes.add(headline_hash(headline.title))
        seen_titles.append(normalized)
    return fresh, duplicates


def headline_records(
    headlines: Iterable[Headline], *, fetched_at: datetime, version: int = NEWS_VERSION
) -> list[dict[str, object]]:
    """Las filas del almacén: sobre obligatorio + ``title``, ``url`` y ``headline_hash``."""
    return [
        {
            "source": headline.source,
            "series_id": headline.feed,
            "as_of": headline.published_at,
            "fetched_at": fetched_at,
            "published_at": headline.published_at,
            "version": version,
            "title": headline.title,
            "url": headline.url,
            "headline_hash": headline_hash(headline.title),
        }
        for headline in headlines
    ]


def existing_headlines(store: Store) -> tuple[frozenset[str], tuple[str, ...]]:
    """Los ``(hashes, títulos)`` ya guardados, para deduplicar **entre ejecuciones**."""
    if DATASET not in store.datasets("raw"):
        return frozenset(), ()
    frame = store.sql("SELECT title, headline_hash FROM raw.news_headlines")
    hashes = frozenset(str(value) for value in frame.get_column("headline_hash").to_list())
    titles = tuple(str(value) for value in frame.get_column("title").to_list())
    return hashes, titles


def ingest(
    *,
    store: Store,
    headlines: Sequence[Headline],
    now: datetime,
    known_hashes: frozenset[str] = frozenset(),
    known_titles: Sequence[str] = (),
    threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> NewsReport:
    """Filtra lo posterior a ``now``, deduplica, escribe ``raw.news_headlines`` y resume.

    Lo que llega con ``published_at > now`` **nunca** se guarda: sería un dato del futuro.
    """
    if now.utcoffset() is None:
        raise ConfigurationError("now: se espera un datetime con zona (TZ-aware)")

    fresh = [headline for headline in headlines if headline.published_at <= now]
    discarded_future = len(headlines) - len(fresh)
    new, duplicates = deduplicate(
        fresh, known_hashes=known_hashes, known_titles=known_titles, threshold=threshold
    )
    outcome = "unchanged"
    if new:
        outcome = store.append("raw", DATASET, headline_records(new, fetched_at=now)).value
    return NewsReport(
        as_of=now.astimezone(UTC).isoformat(),
        fetched=len(headlines),
        discarded_future=discarded_future,
        discarded_duplicate=len(duplicates),
        new=len(new),
        outcome=outcome,
        headline_hashes=tuple(headline_hash(headline.title) for headline in new),
    )


def _parse_as_of(value: str | None) -> datetime:
    """El instante declarado, ISO-8601 y con zona; sin reloj interno."""
    if value is None or not value.strip():
        raise ConfigurationError("falta --as-of: el instante declarado (ISO-8601 con zona)")
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as error:
        raise ConfigurationError(f"--as-of no es ISO-8601: {error}") from error
    if moment.utcoffset() is None:
        raise ConfigurationError("--as-of necesita zona horaria (p. ej. 2026-10-03T12:45:00+00:00)")
    return moment.astimezone(UTC)


def _parse_feed(spec: str) -> tuple[str, str]:
    """``etiqueta=url`` → ``(etiqueta, url)``; sin la etiqueta el spec no vale."""
    label, separator, url = spec.partition("=")
    if not separator or not label.strip() or not url.strip():
        raise ConfigurationError(f"--feed espera `etiqueta=url`, no {spec!r}")
    return label.strip(), url.strip()


def main(argv: Sequence[str] | None = None) -> int:
    """Ingesta manual: recoge GDELT + RSS, deduplica y escribe; imprime el informe (JSON).

    Códigos: ``0`` = ingesta emitida; ``2`` = falta ``--as-of``, no hay fuentes declaradas o la
    configuración es inválida (con el motivo por ``stderr``).
    """
    parser = argparse.ArgumentParser(
        prog="cfdtrader.data.news",
        description="Ingesta de titulares (GDELT + RSS) a raw.news_headlines",
    )
    parser.add_argument(
        "--as-of", default=None, help="instante declarado ISO-8601 con zona (obligatorio)"
    )
    parser.add_argument("--data-root", type=Path, default=None, help="raíz del almacén")
    parser.add_argument("--settings", type=Path, default=None, help="ruta de settings.yaml")
    parser.add_argument("--query", action="append", default=[], help="consulta GDELT (repetible)")
    parser.add_argument(
        "--feed", action="append", default=[], help="RSS `etiqueta=url` (repetible)"
    )
    args = parser.parse_args(argv)

    try:
        moment = _parse_as_of(cast("str | None", args.as_of))
        settings = load_settings(args.settings)
        queries = [str(query) for query in cast("list[object]", args.query)]
        feeds = [_parse_feed(str(feed)) for feed in cast("list[object]", args.feed)]
    except ConfigurationError as error:
        print(f"no se pueden ingestar titulares: {error}", file=sys.stderr)
        return 2
    if not queries and not feeds:
        print(
            "no se pueden ingestar titulares: declara al menos un --query o un --feed",
            file=sys.stderr,
        )
        return 2

    data_root = Path(args.data_root) if args.data_root is not None else Path(settings.data.root)
    store = Store(data_root)
    with CachedHttpClient(source="news", cache_root=data_root / "cache") as client:
        gdelt = GdeltAdapter(client)
        rss = RssAdapter(client)
        headlines: list[Headline] = []
        for query in queries:
            headlines.extend(gdelt.fetch(query=query, now=moment))
        for label, url in feeds:
            headlines.extend(rss.fetch(feed_url=url, label=label, now=moment))

    known_hashes, known_titles = existing_headlines(store)
    report = ingest(
        store=store,
        headlines=headlines,
        now=moment,
        known_hashes=known_hashes,
        known_titles=known_titles,
    )
    print(json.dumps(report.model_dump(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
