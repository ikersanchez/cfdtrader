"""Tests de la caché en disco del LLM (tarea #33): B2-B5.

Se prueba sin proveedor y sin red: el cliente entra simulado. Lo que se blinda:

- La clave es **determinista entre procesos** (B2), no «estable en esta máquina».
- Un acierto **no es una llamada**: el contador del cliente no sube (B3).
- Un **fallo no se cachea** (B4): si no, un error transitorio se vuelve permanente.
- En la caché **no entra nada más** que la respuesta (B5).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

from cfdtrader.llm.base import ChatMessage, LLMError, LLMRequest, LLMResponse
from cfdtrader.llm.cache import CachedLLMClient, ResponseCache, cache_key, request_key

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
KEY_PATTERN: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")

#: Clave ficticia: sirve para comprobar que **no** acaba en la caché.
FAKE_KEY: Final[str] = "sk-cfd-fake-000000000000"  # pragma: allowlist secret


def _request(prompt: str = "Eres un extractor", inputs: str = "un titular") -> LLMRequest:
    return LLMRequest(
        model="deepseek-chat",
        messages=(
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=inputs),
        ),
    )


def _response(content: str = '{"events": []}') -> LLMResponse:
    return LLMResponse(
        content=content,
        model="deepseek-chat",
        system_fingerprint="fp-1",
        prompt_tokens=120,
        completion_tokens=40,
    )


class _CountingClient:
    """Cliente simulado que cuenta llamadas; cada efecto puede ser una respuesta o un fallo."""

    def __init__(self, *effects: LLMResponse | LLMError) -> None:
        self._effects = list(effects)
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        effect = self._effects[min(self.calls - 1, len(self._effects) - 1)]
        if isinstance(effect, LLMError):
            raise effect
        return effect


def _bytes_under(root: Path) -> bytes:
    return b"".join(path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file())


def _key_in_subprocess(seed: str, *, prompt: str, model: str, inputs: str) -> str:
    code = (
        "from cfdtrader.llm.cache import cache_key;"
        f"print(cache_key(prompt={prompt!r}, model={model!r}, inputs={inputs!r}))"
    )
    completed = subprocess.run(  # noqa: S603 - el ejecutable es el interprete de la sesion
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": seed},
    )
    return completed.stdout.strip()


# ─────────────────────────────────────────────────────────────────────────────
# B1 · Los ficheros de la tarea existen
# ─────────────────────────────────────────────────────────────────────────────
def test_b1_the_modules_exist() -> None:
    assert (REPO_ROOT / "src" / "cfdtrader" / "llm" / "cache.py").is_file()
    assert (REPO_ROOT / "src" / "cfdtrader" / "llm" / "budget.py").is_file()
    assert Path(__file__).resolve().is_file()


# ─────────────────────────────────────────────────────────────────────────────
# B2 · La clave: formato, sensibilidad y determinismo entre procesos
# ─────────────────────────────────────────────────────────────────────────────
def test_b2_the_cache_key_is_deterministic_and_sensitive() -> None:
    base = {"prompt": "eres un extractor", "model": "deepseek-chat", "inputs": "titular 1"}
    key = cache_key(**base)

    assert KEY_PATTERN.match(key)
    assert cache_key(**base) == key, "el mismo contenido da la misma clave"
    for component in ("prompt", "model", "inputs"):
        assert cache_key(**(base | {component: "otro"})) != key, component

    assert {_key_in_subprocess(seed, **base) for seed in ("0", "1")} == {key}
    # Ningun digest literal: la clave se recomputa, no se copia.


def test_b2_the_request_key_separates_prompt_from_inputs() -> None:
    request = _request(prompt="instrucciones", inputs="lote de hoy")
    assert request_key(request) == cache_key(
        prompt="instrucciones", model="deepseek-chat", inputs="lote de hoy"
    )
    assert request_key(_request(prompt="otras", inputs="lote de hoy")) != request_key(request)


# ─────────────────────────────────────────────────────────────────────────────
# B3 · Un acierto no es una llamada
# ─────────────────────────────────────────────────────────────────────────────
def test_b3_a_hit_does_not_call_the_provider(tmp_path: Path) -> None:
    client = _CountingClient(_response())
    with ResponseCache(tmp_path / "cache") as cache:
        cached = CachedLLMClient(client, cache)

        first = cached.complete(_request())
        assert cached.last_cache_hit is False
        assert client.calls == 1

        second = cached.complete(_request())
        assert cached.last_cache_hit is True
        assert client.calls == 1, "un acierto no puede volver a llamar al proveedor"
        assert second == first


# ─────────────────────────────────────────────────────────────────────────────
# B4 · Un fallo no se cachea
# ─────────────────────────────────────────────────────────────────────────────
def test_b4_a_failure_is_not_cached(tmp_path: Path) -> None:
    client = _CountingClient(LLMError("el proveedor dijo que no"), _response("vale"))

    with ResponseCache(tmp_path / "cache") as cache:
        cached = CachedLLMClient(client, cache)

        with pytest.raises(LLMError):
            cached.complete(_request())

        assert cached.complete(_request()).content == "vale"
        assert client.calls == 2, "el reintento tiene que volver a preguntar"


# ─────────────────────────────────────────────────────────────────────────────
# B5 · En la cache no entra nada mas que la respuesta
# ─────────────────────────────────────────────────────────────────────────────
def test_b5_the_cache_stores_the_response_and_nothing_else(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    with ResponseCache(root) as cache:
        CachedLLMClient(_CountingClient(_response()), cache).complete(_request())

    blob = _bytes_under(root)
    assert blob, "la cache tiene que haber escrito algo"
    assert FAKE_KEY.encode() not in blob
    for forbidden in (b"notional", b"capital", b"api_key", b"position"):
        assert forbidden not in blob.lower()
