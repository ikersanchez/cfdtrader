"""Cliente HTTP con reintentos, caché en disco y bloqueo tipado (tarea #3).

``_docs/tech_stack.md`` §4.5 exige tres mitigaciones para las fuentes gratuitas:

1. **Reintentos** con backoff exponencial (``tenacity``) ante ``429`` y errores
   de red. Al agotarlos, el error es tipado (:class:`SourceRateLimitedError`),
   nunca un error HTTP crudo.
2. **Caché en disco** de toda respuesta cruda, para no repetir peticiones.
3. **Detección de bloqueo**: si la respuesta no es el formato esperado (HTML,
   *challenge* JavaScript, captcha) se levanta :class:`SourceBlockedError` con el
   ``Content-Type`` y los primeros bytes en el mensaje.

El cliente ``httpx.Client`` se **inyecta** (A2): los tests usan
``httpx.MockTransport`` y no abren red.

La caché vive en ``<raíz>/cache/<fuente>/<AAAA-MM-DD>/`` (dentro de ``/data/``,
ya ignorado por git). Su **política de poda** no se decide aquí: es la #44.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from cfdtrader.data.sources.base import (
    SourceBlockedError,
    SourceRateLimitedError,
    SourceUnavailableError,
)

__all__ = [
    "ACCEPTED_CONTENT_TYPES",
    "LOCK_DETECTOR",
    "CachedHttpClient",
    "CachedResponse",
]

#: Content-types que se aceptan como "formato de datos". Cualquier otra cosa se
#: considera bloqueo salvo que el adaptador declare lo contrario.
#: ``text/plain`` se admite porque Stooq sirve CSV con ese tipo (cuando no bloquea).
ACCEPTED_CONTENT_TYPES: tuple[str, ...] = (
    "text/csv",
    "text/plain",
    "application/json",
    "application/csv",
    "text/json",
)

#: Firmas de bloqueo conocidas: HTML, doctype, XML/HTML y el *challenge* de Stooq.
LOCK_DETECTOR: tuple[bytes, ...] = (
    b"<!DOCTYPE",
    b"<!doctype",
    b"<html",
    b"<HTML",
    b"<?xml",
    b"<script",
    b"__verify",
)

#: Cabeceras que describen el **transporte** y no el contenido. El cuerpo que se
#: guarda en la caché ya está descomprimido, así que `content-encoding` no se
#: conserva: reenviarla haría que `httpx` descomprimiera dos veces.
_TRANSPORT_HEADERS: frozenset[str] = frozenset({"content-encoding", "content-length"})


class _RetryableError(Exception):
    """Fallo transitorio (``429`` o red): se reintenta y, al agotar, se tipa."""

    def __init__(self, message: str, *, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass(frozen=True, slots=True)
class CachedResponse:
    """Respuesta (posiblemente cacheada) con trazabilidad del coste de la petición."""

    response: httpx.Response
    attempts: int
    from_cache: bool

    @property
    def text(self) -> str:
        """Cuerpo como texto."""
        return self.response.text


class CachedHttpClient:
    """``httpx.Client`` con reintentos, caché en disco y errores tipados.

    Parameters
    ----------
    client:
        Cliente inyectado. Si es ``None`` se construye uno propio (uso real).
    source:
        Nombre de la fuente (``stooq``, ``fred``): decide el directorio de caché
        y aparece en los errores.
    cache_root:
        ``<raíz>/cache`` si hay caché de disco, o ``None`` para desactivarla
        (los tests de reintentos la desactivan para que el contador sea exacto).
    cache_ttl_days:
        Vigencia de la caché. ``1`` significa "vale para el mismo día UTC".
    max_attempts:
        Intentos totales ante fallo transitorio. El mínimo es 5 (A4).
    backoff_seconds:
        Base del backoff exponencial. ``0`` en tests para no dormir.
    """

    def __init__(
        self,
        *,
        source: str,
        client: httpx.Client | None = None,
        cache_root: Path | None = None,
        cache_ttl_days: int = 1,
        max_attempts: int = 5,
        backoff_seconds: float = 1.0,
    ) -> None:
        if max_attempts < 5:
            raise ValueError("A4 exige cinco intentos como mínimo ante 429 o error de red")
        self._source = source
        self._client = client if client is not None else httpx.Client(timeout=30.0)
        self._owns_client = client is None
        self._cache_root = cache_root
        self._cache_ttl_days = cache_ttl_days
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds

    # ── Ciclo de vida ────────────────────────────────────────────────────────
    def close(self) -> None:
        """Cierra el cliente solo si lo construyó este objeto."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> CachedHttpClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── GET ──────────────────────────────────────────────────────────────────
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str | int] | None = None,
        now: datetime | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> CachedResponse:
        """GET con caché vigente y reintentos.

        Raises
        ------
        SourceRateLimitedError
            Se agotaron los reintentos ante ``429`` o error de red.
        SourceBlockedError
            La respuesta no es un formato de datos (HTML, *challenge*).
        """
        instant = (now or datetime.now(UTC)).astimezone(UTC)
        key = _cache_key(url, params)
        cached = self._read_cache(key, instant)
        if cached is not None:
            response, attempts = cached
            self._raise_if_blocked(response, attempts=attempts)
            return CachedResponse(response=response, attempts=attempts, from_cache=True)

        response = self._get_with_retries(url, params=params, headers=headers)
        attempts = self._attempts_used
        self._write_cache(key, instant, response)
        self._raise_if_blocked(response, attempts=attempts)
        return CachedResponse(response=response, attempts=attempts, from_cache=False)

    # ── Internos ─────────────────────────────────────────────────────────────
    _attempts_used: int = 0

    def _get_with_retries(
        self,
        url: str,
        *,
        params: Mapping[str, str | int] | None,
        headers: Mapping[str, str] | None,
    ) -> httpx.Response:
        """Realiza la petición con backoff exponencial y traduce el fallo final."""
        retrying = Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(
                multiplier=self._backoff_seconds,
                min=self._backoff_seconds,
                max=max(self._backoff_seconds, 8.0),
            ),
            retry=retry_if_exception_type(_RetryableError),
            reraise=True,
        )
        try:
            for attempt in retrying:
                with attempt:
                    self._attempts_used = attempt.retry_state.attempt_number
                    return self._send(url, params=params, headers=headers)
        except _RetryableError as failure:
            raise SourceRateLimitedError(
                str(failure), source=self._source, attempts=failure.attempts
            ) from failure
        except httpx.HTTPError as error:
            raise SourceUnavailableError(
                f"error HTTP no recuperable en {url}: {error}",
                source=self._source,
                attempts=self._attempts_used,
            ) from error
        raise SourceUnavailableError(  # pragma: no cover - defensivo
            f"no se pudo completar la petición a {url}", source=self._source
        )

    def _send(
        self,
        url: str,
        *,
        params: Mapping[str, str | int] | None,
        headers: Mapping[str, str] | None,
    ) -> httpx.Response:
        """Una petición. Los fallos transitorios se convierten en ``_RetryableError``."""
        try:
            response = self._client.get(url, params=params, headers=headers)
        except httpx.HTTPError as error:
            raise _RetryableError(
                f"error de red pidiendo {url}: {error}", attempts=self._attempts_used
            ) from error
        if response.status_code == 429 or response.status_code >= 500:
            raise _RetryableError(
                f"{url} respondió {response.status_code}", attempts=self._attempts_used
            )
        return response

    def _raise_if_blocked(self, response: httpx.Response, *, attempts: int) -> None:
        """HTML, *challenge* o content-type inesperado ⇒ ``SourceBlockedError``."""
        body = response.content
        content_type = response.headers.get("content-type", "")
        signature = _lock_signature(body)
        if signature is not None:
            raise SourceBlockedError(
                f"{self._source} bloqueó la petición: Content-Type={content_type!r}, "
                f"primeros bytes={body[:60]!r}",
                source=self._source,
                attempts=attempts,
            )
        if not _content_type_accepted(content_type) and body:
            raise SourceBlockedError(
                f"respuesta con formato inesperado: Content-Type={content_type!r}, "
                f"primeros bytes={body[:60]!r}",
                source=self._source,
                attempts=attempts,
            )
        if not body:
            raise SourceUnavailableError(
                f"respuesta vacía de {self._source}",
                source=self._source,
                attempts=attempts,
            )

    # ── Caché ────────────────────────────────────────────────────────────────
    def _cache_dir(self, day: date) -> Path | None:
        if self._cache_root is None:
            return None
        return self._cache_root / self._source / day.isoformat()

    def _read_cache(self, key: str, now: datetime) -> tuple[httpx.Response, int] | None:
        """Respuesta cacheada si sigue vigente (mismo día UTC o dentro del TTL)."""
        directory = self._cache_dir(now.date())
        if directory is None:
            return None
        path = directory / f"{key}.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        stored_day = date.fromisoformat(str(payload["day"]))
        age = (now.date() - stored_day).days
        if age > self._cache_ttl_days:
            return None
        response = httpx.Response(
            status_code=int(payload["status_code"]),
            # El cuerpo guardado ya está descomprimido: `content-encoding` se
            # descarta **también al leer**, para que una entrada escrita por una
            # versión anterior no rompa la lectura.
            headers={
                str(name): str(value)
                for name, value in payload["headers"].items()
                if str(name).lower() not in _TRANSPORT_HEADERS
            },
            content=base64.b64decode(str(payload["body_b64"])),
        )
        return response, int(payload["attempts"])

    def _write_cache(self, key: str, now: datetime, response: httpx.Response) -> None:
        directory = self._cache_dir(now.date())
        if directory is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "source": self._source,
            "day": now.date().isoformat(),
            "status_code": response.status_code,
            # ⚠️ `response.content` ya viene **descomprimido**: si se guarda
            # `content-encoding` tal cual, al releer la caché `httpx` intenta
            # descomprimir otra vez y revienta con `DecodingError`. Se guarda el
            # `Content-Type` y lo demás, pero no la codificación del transporte.
            "headers": {
                name: value
                for name, value in response.headers.items()
                if name.lower() not in _TRANSPORT_HEADERS
            },
            "body_b64": base64.b64encode(response.content).decode("ascii"),
            "attempts": self._attempts_used,
        }
        path = directory / f"{key}.json"
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)


def _cache_key(url: str, params: Mapping[str, str | int] | None) -> str:
    """Clave estable de la caché: URL y parámetros en orden canónico."""
    canonical = json.dumps(
        {"url": url, "params": dict(sorted((params or {}).items()))},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _lock_signature(body: bytes) -> bytes | None:
    """Firma de bloqueo presente en los primeros bytes, si la hay."""
    head = body[:512]
    for marker in LOCK_DETECTOR:
        if marker in head:
            return marker
    return None


def _content_type_accepted(content_type: str) -> bool:
    """``True`` si el ``Content-Type`` declara un formato de datos aceptado."""
    normalized = content_type.split(";")[0].strip().lower()
    if not normalized:
        return False
    return any(normalized == accepted for accepted in ACCEPTED_CONTENT_TYPES)
