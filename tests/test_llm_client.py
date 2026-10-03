"""Tests del cliente LLM (`#31`): protocolo, configuración y proveedor.

El artefacto verificable es que **`agents/` no se acopla al proveedor**: el protocolo
(`cfdtrader.llm.base`) no importa el SDK y un **proveedor simulado** lo satisface. La
implementación del SDK se prueba con un **doble** inyectado, sin red ni clave real.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final, cast

import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from pydantic import ValidationError

from cfdtrader.llm.base import (
    DEFAULT_BASE_URLS,
    PROVIDERS,
    ChatMessage,
    LLMClient,
    LLMClientConfig,
    LLMConfigError,
    LLMError,
    LLMRequest,
    LLMResponse,
    build_client,
)
from cfdtrader.llm.openai_client import OpenAIClient

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
BASE_MODULE: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "llm" / "base.py"
INIT_MODULE: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "llm" / "__init__.py"
AGENTS_DIR: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "agents"


def _user_message(content: str = "hola") -> ChatMessage:
    return ChatMessage(role="user", content=content)


def _config(**fields: object) -> LLMClientConfig:
    """Config tipada **sin** leer el ``.env`` real: aísla las pruebas del entorno del repo."""
    return LLMClientConfig(_env_file=None, **fields)  # pyright: ignore[reportCallIssue]


# ─────────────────────────────────────────────────────────────────────────────
# A1/A2 · El acoplamiento al proveedor, fuera de `agents/` y del protocolo
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_protocol_module_does_not_import_the_sdk() -> None:
    """El protocolo no importa el SDK: `agents/` puede depender de él sin proveedor."""
    source = BASE_MODULE.read_text(encoding="utf-8")

    assert "import openai" not in source
    assert "from openai" not in source
    # El paquete tampoco importa nada al cargarse: no arrastra la implementación del SDK.
    tree = ast.parse(INIT_MODULE.read_text(encoding="utf-8"))
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
    ]
    assert imports == []


def test_a2_agents_do_not_import_the_provider() -> None:
    """Ningún módulo de `agents/` importa el SDK: viaja detrás de `LLMClient` (§4.9)."""
    for path in sorted(AGENTS_DIR.glob("*.py")):
        assert "openai" not in path.read_text(encoding="utf-8"), path


# ─────────────────────────────────────────────────────────────────────────────
# A3 · Configuración tipada (entorno y `.env`)
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_the_config_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """La config tipada lee `LLM_*` del entorno (con `env_prefix`) y resuelve la `base_url`."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "clave-de-prueba")
    config = _config()

    assert config.provider == "openai"
    assert config.api_key == "clave-de-prueba"  # pragma: allowlist secret
    assert config.resolved_base_url() == DEFAULT_BASE_URLS["openai"]
    assert set(PROVIDERS) == {"openai", "deepseek"}


def test_a3_the_config_resolves_the_model_and_the_base_url() -> None:
    """La `base_url` declarada gana; el modelo con versión concreta se exige por propósito."""
    config = _config(
        provider="deepseek",
        api_key="k",
        base_url="https://example.invalid/v1",
        model_extract="deepseek-chat-2026-01",
    )

    assert config.resolved_base_url() == "https://example.invalid/v1"
    assert config.model_for("extract") == "deepseek-chat-2026-01"
    with pytest.raises(LLMConfigError, match="modelo de 'report'"):
        config.model_for("report")
    with pytest.raises(LLMConfigError, match="proposito desconocido"):
        config.model_for("otro")


# ─────────────────────────────────────────────────────────────────────────────
# A4 · El protocolo acepta un proveedor simulado
# ─────────────────────────────────────────────────────────────────────────────
class SimulatedClient:
    """Un proveedor de mentira: satisface `LLMClient` sin SDK, red ni clave."""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            content='{"eventos": []}', model=request.model, system_fingerprint="fp-sim"
        )


def test_a4_a_simulated_provider_satisfies_the_protocol() -> None:
    """Un cliente simulado es un `LLMClient` y devuelve la respuesta tipada."""
    client = SimulatedClient()
    assert isinstance(client, LLMClient)

    request = LLMRequest(model="modelo-sim", messages=(_user_message(),))
    response = client.complete(request)

    assert isinstance(response, LLMResponse)
    assert response.model == "modelo-sim"
    assert response.system_fingerprint == "fp-sim"
    assert client.requests == [request]


# ─────────────────────────────────────────────────────────────────────────────
# A5 · La fábrica: proveedor declarado y clave presente
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_the_factory_rejects_unknown_providers_and_missing_keys() -> None:
    """Un proveedor fuera de la lista o la ausencia de clave son error tipado, nunca ImportError."""
    with pytest.raises(LLMConfigError, match="proveedor"):
        build_client(_config(provider="acme", api_key="k"))
    with pytest.raises(LLMConfigError, match="LLM_API_KEY"):
        build_client(_config(provider="openai", api_key="   "))


def test_a5_the_factory_builds_the_provider_client() -> None:
    """Con proveedor y clave válidos, la fábrica construye la implementación del SDK."""
    client = build_client(_config(provider="deepseek", api_key="k"))

    assert isinstance(client, OpenAIClient)


# ─────────────────────────────────────────────────────────────────────────────
# A6 · Contrato de los modelos tipados
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_requests_and_responses_are_frozen_and_validated() -> None:
    """La petición y la respuesta son inmutables, de esquema cerrado y con valores válidos."""
    assert LLMRequest.model_config.get("frozen") is True
    assert LLMResponse.model_config.get("frozen") is True

    request = LLMRequest(model="modelo-1", messages=(_user_message(),))
    assert request.temperature == 0.0  # el LLM no debe ser creativo (§7.4)

    with pytest.raises(ValidationError):
        LLMRequest(model="modelo-1", messages=())
    with pytest.raises(ValidationError):
        LLMRequest(model="", messages=(_user_message(),))
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")
    with pytest.raises(ValidationError):
        request.model = "otro"
    with pytest.raises(ValidationError):
        LLMResponse.model_validate({"content": "x", "model": "m", "extra": 1})


# ─────────────────────────────────────────────────────────────────────────────
# A7 · La implementación del SDK, con un doble inyectado (sin red)
# ─────────────────────────────────────────────────────────────────────────────
class _FakeCompletions:
    """Doble de `chat.completions`: registra la llamada y devuelve una respuesta pautada."""

    def __init__(self, *, completion: ChatCompletion | None, error: Exception | None) -> None:
        self._completion = completion
        self._error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> ChatCompletion:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        assert self._completion is not None
        return self._completion


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeSDK:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _FakeChat(completions)


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="1",
        object="chat.completion",
        created=0,
        model="modelo-1",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="hola"),
            )
        ],
        system_fingerprint="fp-1",
        usage=CompletionUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )


def _client(completions: _FakeCompletions) -> OpenAIClient:
    sdk = cast("OpenAI", _FakeSDK(completions))
    return OpenAIClient(api_key="k", base_url="https://example.invalid", sdk=sdk)


def test_a7_the_client_maps_the_sdk_response_and_registers_the_identity() -> None:
    """La respuesta del SDK se traduce a `LLMResponse` con su modelo, fingerprint y tokens."""
    completions = _FakeCompletions(completion=_completion(), error=None)
    response = _client(completions).complete(
        LLMRequest(model="modelo-1", messages=(_user_message(),))
    )

    assert response.content == "hola"
    assert response.model == "modelo-1"
    assert response.system_fingerprint == "fp-1"
    assert response.prompt_tokens == 3 and response.completion_tokens == 2

    called = completions.calls[0]
    assert called["model"] == "modelo-1"
    assert called["temperature"] == 0.0
    assert called["messages"] == [{"role": "user", "content": "hola"}]


def test_a7_a_provider_failure_is_wrapped_in_a_typed_error() -> None:
    """Un fallo del SDK no escapa crudo: se envuelve en `LLMError` (el overlay nunca bloquea)."""
    completions = _FakeCompletions(completion=None, error=RuntimeError("boom"))

    with pytest.raises(LLMError, match="proveedor"):
        _client(completions).complete(LLMRequest(model="m", messages=(_user_message(),)))


def test_a7_the_client_requires_a_key_and_a_base_url() -> None:
    """Construir sin clave o sin `base_url` es error tipado, no un fallo a mitad de la llamada."""
    with pytest.raises(LLMError, match="clave"):
        OpenAIClient(api_key="  ", base_url="https://example.invalid")
    with pytest.raises(LLMError, match="base_url"):
        OpenAIClient(api_key="k", base_url="")


def test_a8_the_optional_parameters_are_forwarded_when_declared() -> None:
    """`seed`, `response_format` y `max_tokens` se reenvían al SDK solo cuando se declaran."""
    completions = _FakeCompletions(completion=_completion(), error=None)
    _client(completions).complete(
        LLMRequest(
            model="modelo-1",
            messages=(_user_message(),),
            seed=7,
            response_format={"type": "json_object"},
            max_tokens=64,
        )
    )

    called = completions.calls[0]
    assert called["seed"] == 7
    assert called["response_format"] == {"type": "json_object"}
    assert called["max_tokens"] == 64


def test_a9_a_missing_usage_or_no_choices_is_handled() -> None:
    """Un `usage` ausente deja los tokens a `None`; una respuesta sin opciones es error tipado."""
    without_usage = ChatCompletion(
        id="1",
        object="chat.completion",
        created=0,
        model="m",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="x"),
            )
        ],
        system_fingerprint=None,
        usage=None,
    )
    response = _client(_FakeCompletions(completion=without_usage, error=None)).complete(
        LLMRequest(model="m", messages=(_user_message(),))
    )
    assert response.prompt_tokens is None and response.completion_tokens is None
    assert response.system_fingerprint is None

    empty = ChatCompletion(
        id="1",
        object="chat.completion",
        created=0,
        model="m",
        choices=[],
        system_fingerprint=None,
        usage=None,
    )
    with pytest.raises(LLMError, match="ninguna opción"):
        _client(_FakeCompletions(completion=empty, error=None)).complete(
            LLMRequest(model="m", messages=(_user_message(),))
        )


def test_a9_resolved_base_url_rejects_an_unknown_provider() -> None:
    """Sin `base_url` declarada y con un proveedor desconocido, resolver la URL es error tipado."""
    with pytest.raises(LLMConfigError, match="proveedor"):
        _config(provider="acme").resolved_base_url()
