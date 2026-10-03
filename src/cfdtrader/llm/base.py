"""Protocolo interno del cliente LLM (`tech_stack.md` §4.9) — tarea #31.

La pregunta que responde: **¿cómo pide el sistema una extracción o una redacción al LLM
sin acoplarse a un proveedor?**

- :class:`LLMClient` es el **protocolo** que consumen los agentes: un solo método ``complete``.
- :class:`LLMRequest` / :class:`LLMResponse` son los datos tipados que cruzan esa frontera.
- :class:`LLMClientConfig` es la configuración tipada (``pydantic-settings``), leída del
  entorno y del ``.env``.
- :func:`build_client` es la **fábrica**: importa **perezosamente** la implementación del SDK.

Garantías (§4.9 y §7.4):

- **OpenAI y DeepSeek con un solo cliente**: ambos comparten el SDK ``openai`` y solo cambia
  ``base_url``. La implementación vive en :mod:`cfdtrader.llm.openai_client`.
- **``temperature = 0`` por defecto** y ``seed`` *best effort*: nada es determinista al 100 %,
  así que además se registra ``model`` y ``system_fingerprint`` en cada respuesta.
- **Este módulo no importa el SDK**: ``agents/`` puede depender de ``llm.base`` sin arrastrar
  el proveedor (la prueba lo comprueba).
"""

from __future__ import annotations

from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from cfdtrader.data.settings import REPO_ROOT

__all__ = [
    "DEFAULT_BASE_URLS",
    "DEFAULT_TEMPERATURE",
    "PROVIDERS",
    "ChatMessage",
    "LLMClient",
    "LLMClientConfig",
    "LLMConfigError",
    "LLMError",
    "LLMRequest",
    "LLMResponse",
    "Role",
    "build_client",
]

#: Roles admitidos en la conversación (subconjunto de Chat Completions).
Role = Literal["system", "user", "assistant"]

#: Proveedores soportados: los dos que comparten el SDK ``openai`` (§4.9).
PROVIDERS: Final[tuple[str, ...]] = ("openai", "deepseek")

#: ``base_url`` por defecto de cada proveedor (solo cambia el *endpoint*, §4.9).
DEFAULT_BASE_URLS: Final[dict[str, str]] = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
}

#: ``temperature`` fija: el LLM no debe ser creativo (§7.4).
DEFAULT_TEMPERATURE: Final[float] = 0.0


class LLMError(Exception):
    """Raíz de los errores de la capa LLM."""


class LLMConfigError(LLMError):
    """La configuración del cliente no es válida (proveedor desconocido, falta la clave)."""


class ChatMessage(BaseModel):
    """Un mensaje de la conversación, con su rol."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    content: str = Field(min_length=1)


class LLMRequest(BaseModel):
    """Una petición tipada al cliente: modelo, mensajes y parámetros de muestreo."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1, description="identificador con versión concreta, nunca alias")
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    seed: int | None = Field(default=None, description="best effort; no todos los proveedores")
    response_format: dict[str, object] | None = Field(
        default=None, description="por ejemplo {'type': 'json_object'} (modo JSON, no esquema)"
    )
    max_tokens: int | None = Field(default=None, ge=1)


class LLMResponse(BaseModel):
    """La respuesta tipada: texto y la identidad del modelo que lo produjo."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str
    model: str
    system_fingerprint: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@runtime_checkable
class LLMClient(Protocol):
    """El contrato que consumen los agentes: una implementación por proveedor (§4.9)."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Envía la petición y devuelve la respuesta tipada; un fallo es :class:`LLMError`."""
        ...  # pragma: no cover - el cuerpo de un Protocol no se ejecuta


class LLMClientConfig(BaseSettings):
    """Configuración tipada del cliente, del entorno y del ``.env`` (`tech_stack.md` §4.2)."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: str = Field(default="deepseek", description="openai | deepseek")
    api_key: str | None = None
    base_url: str | None = None
    model_extract: str | None = None
    model_report: str | None = None
    daily_budget_eur: str | None = None

    def resolved_base_url(self) -> str:
        """La ``base_url`` declarada o, si falta, la del proveedor por defecto (§4.9)."""
        if self.base_url is not None and self.base_url.strip():
            return self.base_url
        if self.provider not in DEFAULT_BASE_URLS:
            raise LLMConfigError(
                f"proveedor no soportado: {self.provider!r}; se espera uno de {PROVIDERS}"
            )
        return DEFAULT_BASE_URLS[self.provider]

    def model_for(self, purpose: str) -> str:
        """El modelo declarado para ese propósito (``extract`` o ``report``), nunca un alias."""
        if purpose == "extract":
            chosen = self.model_extract
        elif purpose == "report":
            chosen = self.model_report
        else:
            raise LLMConfigError(
                f"proposito desconocido: {purpose!r}; se espera 'extract' o 'report'"
            )
        if chosen is None or not chosen.strip():
            raise LLMConfigError(
                f"falta el modelo de {purpose!r}: declara LLM_MODEL_{purpose.upper()} en el .env"
            )
        return chosen


def build_client(config: LLMClientConfig) -> LLMClient:
    """Construye el cliente del proveedor declarado (importa el SDK **de forma perezosa**).

    Un proveedor desconocido o la ausencia de clave son :class:`LLMConfigError`, nunca un
    ``ImportError`` ni un fallo a mitad del pipeline.
    """
    if config.provider not in PROVIDERS:
        raise LLMConfigError(
            f"proveedor no soportado: {config.provider!r}; se espera uno de {PROVIDERS}"
        )
    key = config.api_key
    if key is None or not key.strip():
        raise LLMConfigError("falta LLM_API_KEY: sin clave no hay capa LLM (ver .env.example)")
    # Import perezoso: el SDK solo entra por aquí, nunca al importar `llm.base`.
    from cfdtrader.llm.openai_client import OpenAIClient

    return OpenAIClient(api_key=key, base_url=config.resolved_base_url())
