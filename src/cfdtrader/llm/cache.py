"""Caché en disco de respuestas del LLM (`tech_stack.md` §4.9, §6.3 y §12.6) — tarea #33.

La pregunta que responde este módulo: **¿cómo no pagar dos veces por la misma pregunta?**

- La clave es el ``hash(prompt + modelo + inputs)`` que pide §4.9, y es **determinista entre
  procesos**: dos ejecuciones con ``PYTHONHASHSEED`` distinto producen la misma clave.
- Un acierto de caché **no es una llamada**: no gasta presupuesto ni cuenta como fallo.
- Un **fallo no se cachea**. Guardar un error convertiría un problema transitorio del proveedor en
  un fallo permanente hasta que alguien purgase la caché.
- En la caché entra **la respuesta y nada más**. Ni claves de API, ni posiciones, ni capital: §3.2.b
  y §12.8 lo exigen **por diseño**, así que aquí no hay nada que «filtrar» después.

El almacén lo pone ``diskcache``; la política de poda y retención es de #44, no de este módulo.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final, cast

from diskcache import Cache
from pydantic import ValidationError

from cfdtrader.data.sources.news import HASH_PREFIX
from cfdtrader.llm.base import LLMClient, LLMRequest, LLMResponse

__all__ = [
    "DEFAULT_SIZE_LIMIT",
    "CachedLLMClient",
    "ResponseCache",
    "cache_key",
    "request_key",
]

#: Tope de disco por defecto de la caché (~150-250 MB es el presupuesto de §12.7).
DEFAULT_SIZE_LIMIT: Final[int] = 200 * 1024 * 1024


def cache_key(*, prompt: str, model: str, inputs: str) -> str:
    """``sha256:<hex>`` de ``hash(prompt + modelo + inputs)``, estable entre procesos.

    El JSON se serializa con claves ordenadas y sin espacios: el mismo contenido da el mismo texto
    con independencia de ``PYTHONHASHSEED`` y del orden de insercion del diccionario.
    """
    canonical = json.dumps(
        {"inputs": inputs, "model": model, "prompt": prompt},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{HASH_PREFIX}{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


class ResponseCache:
    """La caché en disco: un valor por clave, sin interpretar su contenido."""

    def __init__(self, root: Path | str, *, size_limit: int = DEFAULT_SIZE_LIMIT) -> None:
        self._root = Path(str(root))
        self._cache = Cache(str(self._root), size_limit=size_limit)

    @property
    def root(self) -> Path:
        """El directorio donde vive la caché."""
        return self._root

    def get(self, key: str) -> LLMResponse | None:
        """La respuesta guardada, o ``None``. Una entrada ilegible se descarta, no se propaga."""
        # `diskcache` no publica stubs: el valor que devuelve es opaco por definicion.
        stored = cast("object", self._cache.get(key))  # pyright: ignore[reportUnknownMemberType]
        if not isinstance(stored, str):
            return None
        try:
            return LLMResponse.model_validate_json(stored)
        except ValidationError:
            return None

    def put(self, key: str, response: LLMResponse) -> None:
        """Guarda **solo** los campos de la respuesta: es lo unico que puede vivir aqui."""
        self._cache.set(key, response.model_dump_json())  # pyright: ignore[reportUnknownMemberType]

    def close(self) -> None:
        """Cierra el almacen (en pruebas, para poder medir los bytes escritos)."""
        self._cache.close()

    def __enter__(self) -> ResponseCache:
        return self

    def __exit__(self, *arguments: object) -> None:
        self.close()


class CachedLLMClient:
    """Cliente decorador: responde de la caché cuando puede y solo entonces llama al proveedor."""

    def __init__(self, client: LLMClient, cache: ResponseCache) -> None:
        self._client = client
        self._cache = cache
        self._last_cache_hit = False

    @property
    def last_cache_hit(self) -> bool:
        """Si la ultima peticion se resolvio con la cache (no es una llamada al proveedor)."""
        return self._last_cache_hit

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Devuelve la respuesta cacheada o pide una nueva; un ``LLMError`` **no** se cachea."""
        key = request_key(request)
        hit = self._cache.get(key)
        if hit is not None:
            self._last_cache_hit = True
            return hit
        self._last_cache_hit = False
        response = self._client.complete(request)
        self._cache.put(key, response)
        return response


def request_key(request: LLMRequest) -> str:
    """La clave de esa peticion concreta: ``hash(prompt + modelo + inputs)`` (§4.9)."""
    return cache_key(
        prompt=_prompt_text(request), model=request.model, inputs=_inputs_text(request)
    )


def _prompt_text(request: LLMRequest) -> str:
    """El *system prompt* de la peticion: la parte fija que se repite cada dia (§6.3, palanca 5)."""
    return "\n".join(message.content for message in request.messages if message.role == "system")


def _inputs_text(request: LLMRequest) -> str:
    """El resto de la conversacion: lo que varia de una ejecucion a otra."""
    return "\n".join(message.content for message in request.messages if message.role != "system")
