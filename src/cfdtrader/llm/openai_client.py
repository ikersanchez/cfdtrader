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
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT_SECONDS",
    "OpenAIClient",
]

#: Tiempo máximo de **un intento** (segundos). Con ``60`` una extracción de noticias lenta se
#: agotaba: el SDK reintentaba (2 veces por defecto) y el overlay moría con ``3 x 60 s``. El
#: overlay es opcional y no debe tumbar el pipeline, pero tampoco rendirse por impaciencia.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 180.0

#: Reintentos **por debajo** de los 2 del SDK: un único reintento basta y acota el peor caso a
#: ``(1 + retries) x timeout`` (360 s), en vez de multiplicarlo por tres.
DEFAULT_MAX_RETRIES: Final[int] = 1


class OpenAIClient:
    """Cliente sobre Chat Completions del SDK ``openai`` (sirve OpenAI y DeepSeek).

    ``sdk`` permite inyectar un cliente ya construido (o un doble en las pruebas): la
    construcción normal cubre OpenAI y DeepSeek cambiando solo ``base_url``. ``timeout`` y
    ``max_retries`` son declarables (``LLM_TIMEOUT_SECONDS`` / ``LLM_MAX_RETRIES``): el overlay de
    noticias es opcional, así que una llamada lenta no debe tumbar el camino diario, pero tampoco
    debe rendirse antes de tiempo.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        sdk: OpenAI | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise LLMError("OpenAIClient: falta la clave de API")
        if not base_url or not base_url.strip():
            raise LLMError("OpenAIClient: falta la `base_url`")
        if timeout <= 0:
            raise LLMError(f"OpenAIClient: `timeout` debe ser > 0 segundos, no {timeout!r}")
        if max_retries < 0:
            raise LLMError(f"OpenAIClient: `max_retries` debe ser >= 0, no {max_retries!r}")
        self._sdk = (
            sdk
            if sdk is not None
            else OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
            )
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
