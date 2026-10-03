"""Implementación del `LLMClient` sobre el SDK `openai` (`tech_stack.md` §4.9) — tarea #31.

**Único** módulo que importa el SDK. Un solo cliente cubre OpenAI y DeepSeek: solo cambia
``base_url``. No decide ni valida esquemas (eso es de los agentes, #32): traduce la respuesta
del proveedor a :class:`~cfdtrader.llm.base.LLMResponse` y envuelve cualquier fallo en
:class:`~cfdtrader.llm.base.LLMError`, de forma que el overlay nunca bloquee el pipeline.
"""

from __future__ import annotations

from typing import Final, cast

from openai import OpenAI
from openai.types.chat import ChatCompletion

from cfdtrader.llm.base import LLMError, LLMRequest, LLMResponse

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "OpenAIClient",
]

#: Tiempo máximo de una llamada (segundos). El overlay es opcional: si tarda, el pipeline sigue.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 60.0


class OpenAIClient:
    """Cliente sobre Chat Completions del SDK ``openai`` (sirve OpenAI y DeepSeek).

    ``sdk`` permite inyectar un cliente ya construido (o un doble en las pruebas): la
    construcción normal cubre OpenAI y DeepSeek cambiando solo ``base_url``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        sdk: OpenAI | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise LLMError("OpenAIClient: falta la clave de API")
        if not base_url or not base_url.strip():
            raise LLMError("OpenAIClient: falta la `base_url`")
        self._sdk = (
            sdk if sdk is not None else OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        )

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Envía la petición al proveedor y devuelve la respuesta tipada (o :class:`LLMError`)."""
        payload: dict[str, object] = {
            "model": request.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
            "temperature": request.temperature,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        try:
            completion = cast(
                "ChatCompletion",
                self._sdk.chat.completions.create(**payload),  # pyright: ignore[reportArgumentType, reportCallIssue]
            )
        except Exception as error:  # el SDK lanza sus propias excepciones
            raise LLMError(f"falló la llamada al proveedor: {error}") from error
        if not completion.choices:
            raise LLMError("el proveedor no devolvió ninguna opción")
        choice = completion.choices[0]
        usage = completion.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=completion.model,
            system_fingerprint=completion.system_fingerprint,
            prompt_tokens=None if usage is None else usage.prompt_tokens,
            completion_tokens=None if usage is None else usage.completion_tokens,
        )
