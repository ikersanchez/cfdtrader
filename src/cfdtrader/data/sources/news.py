"""Adaptadores de noticias: GDELT y RSS (`tech_stack.md` §4.9, §12.4) — tarea #30.

Recogen **titulares** con su hora real de publicación. Guardan **titular, URL,
`published_at` y hash; nunca el cuerpo del artículo** (§12.7): almacenar el texto íntegro
puede infringir los términos de uso de las agencias, no solo ocupar disco.

- :func:`normalize_title` y :func:`headline_hash` son la **deduplicación barata** de §4.9:
  minúsculas, sin acentos y sin puntuación, más un hash. **Sin *embeddings***.
- :func:`parse_gdelt_articles` y :func:`parse_rss_feed` son **puros** (entra un payload, sale
  una lista de titulares): el parseo se prueba sin red.
- :class:`GdeltAdapter` y :class:`RssAdapter` reciben el cliente HTTP **inyectado** (con caché
  y reintentos, ``tenacity``): los adaptadores no abren red por su cuenta.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Final, cast

import feedparser
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cfdtrader.data.sources.http import CachedHttpClient

__all__ = [
    "GDELT_ENDPOINT",
    "GDELT_MAX_RECORDS",
    "HASH_PREFIX",
    "RSS_USER_AGENT",
    "GdeltAdapter",
    "Headline",
    "NewsAdapterError",
    "RssAdapter",
    "headline_hash",
    "normalize_title",
    "parse_gdelt_articles",
    "parse_rss_feed",
]

#: Endpoint de la API de documentos de GDELT (uso libre, ``tech_stack.md`` §4.5).
GDELT_ENDPOINT: Final[str] = "https://api.gdeltproject.org/api/v2/doc/doc"

#: Tope de artículos por consulta (el proveedor limita a 250).
GDELT_MAX_RECORDS: Final[int] = 250

#: Prefijo obligatorio de los digests (``detect-secrets``: nunca un hex desnudo).
HASH_PREFIX: Final[str] = "sha256:"

#: Formato de la fecha de GDELT en la lista de artículos.
_GDELT_DATE_FORMAT: Final[str] = "%Y%m%dT%H%M%SZ"

#: User-Agent de navegador para los RSS. Varios medios —`cnbc.com`, `nasdaq.com`—
#: devuelven «Access Denied» (403) o cortan la conexión a un cliente sin UA
#: reconocible, aunque sirvan el mismo feed a un navegador (#142).
RSS_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class NewsAdapterError(Exception):
    """La fuente de noticias no entregó un payload reconocible."""


class Headline(BaseModel):
    """Un titular con su procedencia y su hora de publicación (UTC, con zona)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(description="'gdelt' o 'rss'")
    feed: str = Field(min_length=1, description="etiqueta de la fuente (consulta o medio)")
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    published_at: datetime = Field(description="instante de publicación, con zona (UTC)")

    @field_validator("published_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("published_at: se espera un datetime con zona (TZ-aware)")
        return value.astimezone(UTC)


def normalize_title(title: str) -> str:
    """Titular normalizado: minúsculas, sin acentos, sin puntuación, espacios colapsados (§4.9)."""
    decomposed = unicodedata.normalize("NFKD", title.casefold())
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    cleaned = "".join(
        char if (char.isalnum() or char.isspace()) else " " for char in without_accents
    )
    return " ".join(cleaned.split())


def headline_hash(title: str) -> str:
    """``sha256:<hex>`` del titular normalizado: la clave de deduplicación exacta."""
    digest = hashlib.sha256(normalize_title(title).encode("utf-8")).hexdigest()
    return f"{HASH_PREFIX}{digest}"


def _gdelt_datetime(value: str) -> datetime | None:
    """Fecha de GDELT (``20240102T153000Z``) en UTC, o ``None`` si no encaja."""
    try:
        return datetime.strptime(value, _GDELT_DATE_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None


def parse_gdelt_articles(payload: Mapping[str, object], *, feed: str) -> list[Headline]:
    """Titulares de la respuesta *ArtList* de GDELT; las filas incompletas se descartan."""
    articles = payload.get("articles")
    if not isinstance(articles, list):
        raise NewsAdapterError("payload de GDELT sin la lista `articles`")
    headlines: list[Headline] = []
    for article in cast("list[object]", articles):
        if not isinstance(article, dict):
            continue
        item = cast("Mapping[str, object]", article)
        title = item.get("title")
        url = item.get("url")
        seen = item.get("seendate")
        if not (isinstance(title, str) and isinstance(url, str) and isinstance(seen, str)):
            continue
        published = _gdelt_datetime(seen)
        if published is None or not title.strip():
            continue
        headlines.append(
            Headline(source="gdelt", feed=feed, title=title, url=url, published_at=published)
        )
    return headlines


def _entry_datetime(entry: Any) -> datetime | None:
    """La fecha de un *entry* de feedparser (``published_parsed``), en UTC."""
    parsed_time = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed_time is None:
        return None
    try:
        return datetime(*parsed_time[:6], tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def _feedparser_entries(payload: bytes) -> list[Any]:
    """Los *entries* de feedparser, aislados en un único punto de su falta de *stubs*."""
    parsed = cast("Any", feedparser.parse(payload))  # pyright: ignore[reportUnknownMemberType]
    return cast("list[Any]", parsed.entries)


def parse_rss_feed(payload: bytes, *, feed: str) -> list[Headline]:
    """Titulares de un RSS: título, enlace y fecha de publicación; sin el cuerpo (§12.7)."""
    headlines: list[Headline] = []
    for entry in _feedparser_entries(payload):
        title = entry.get("title")
        link = entry.get("link")
        if not (
            isinstance(title, str) and title.strip() and isinstance(link, str) and link.strip()
        ):
            continue
        published = _entry_datetime(entry)
        if published is None:
            continue
        headlines.append(
            Headline(source="rss", feed=feed, title=title.strip(), url=link, published_at=published)
        )
    return headlines


class GdeltAdapter:
    """Adaptador de GDELT sobre el cliente HTTP inyectado (caché y reintentos incluidos)."""

    name: Final[str] = "gdelt"

    def __init__(self, client: CachedHttpClient) -> None:
        self._client = client

    def fetch(self, *, query: str, now: datetime) -> list[Headline]:
        """Titulares de esa consulta a GDELT, con la fecha de referencia ``now``."""
        response = self._client.get(
            GDELT_ENDPOINT,
            params={
                "query": query,
                "mode": "ArtList",
                "format": "json",
                "maxrecords": GDELT_MAX_RECORDS,
            },
            now=now,
        )
        try:
            payload = json.loads(response.response.text)
        except json.JSONDecodeError as error:
            raise NewsAdapterError(f"GDELT: el payload no es JSON ({error})") from error
        if not isinstance(payload, dict):
            raise NewsAdapterError("GDELT: se esperaba un objeto JSON en la raíz")
        return parse_gdelt_articles(cast("Mapping[str, object]", payload), feed=query)


class RssAdapter:
    """Adaptador de RSS sobre el cliente HTTP inyectado (``feedparser`` tras la caché)."""

    name: Final[str] = "rss"

    def __init__(self, client: CachedHttpClient) -> None:
        self._client = client

    def fetch(self, *, feed_url: str, label: str, now: datetime) -> list[Headline]:
        """Titulares de ese canal RSS, con la fecha de referencia ``now``.

        Envía un ``User-Agent`` de navegador: sin él, varios medios contestan
        «Access Denied» aunque sirvan el feed a un navegador (#142).
        """
        response = self._client.get(feed_url, now=now, headers={"User-Agent": RSS_USER_AGENT})
        return parse_rss_feed(response.response.content, feed=label)
