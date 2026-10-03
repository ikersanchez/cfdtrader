"""Agente de noticias: de titulares a eventos tipados — tarea #32 (`tech_stack.md` §4.9).

La pregunta que responde este módulo: **¿cómo se convierte un lote de titulares en eventos que no
puedan envenenar la decisión?**

Tres garantías, y ninguna es opcional:

- **Ningún evento que no valide sale de aquí.** El proveedor **no** valida por nosotros: DeepSeek
  garantiza que la salida *es* JSON, no que cumpla el esquema (§4.9, aviso 1). Por eso se valida
  siempre con Pydantic, se reintenta con el error de validación **dentro del mensaje** y, agotados
  los intentos, el lote se descarta. El texto crudo del modelo no cruza esta frontera.
- **Ningún evento inventado sale de aquí.** Un `headline_hash` que no pertenezca al lote de entrada
  se descarta: el modelo no puede referenciar titulares que no se le dieron.
- **El prompt vive en git, y se persiste su hash.** La plantilla
  (``prompts/news_extract.j2``) está versionada en el repositorio y solo se registra su
  ``sha256`` (§12.8). Nada de prompts en la base de datos.

Por qué el vocabulario es el que es:

- ``sentiment`` **no** es la dirección de la operación. ``Direction``
  (``long``/``short``/``nothing``) ya existe en el repositorio (``backtest.engine``,
  ``decision.gate``) y este módulo **no la importa**: el LLM no decide la dirección (§4.9).
- ``magnitude`` es **ordinal** (``low``/``medium``/``high``), nunca un número: el LLM **no calcula
  números**. El mapeo a puntos porcentuales es del overlay, en el gate (#35).

El proveedor entra **inyectado** por :class:`cfdtrader.llm.base.LLMClient`: este módulo no
importa el SDK y no sabe qué hay detrás. No lee el reloj y no abre la red.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateError,
    select_autoescape,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cfdtrader.data.sources.news import HASH_PREFIX, Headline, headline_hash
from cfdtrader.llm.base import ChatMessage, LLMClient, LLMRequest

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "HEADLINE_HASH_FORMAT",
    "HEADLINE_PROMPT_FIELDS",
    "PROMPTS_DIRNAME",
    "PROMPT_TEMPLATE_NAME",
    "EventType",
    "Horizon",
    "Magnitude",
    "NewsAgent",
    "NewsAgentError",
    "NewsEvent",
    "NewsExtraction",
    "PromptTemplateError",
    "Sentiment",
    "headline_context",
    "prompt_hash",
]

#: Nombre de la plantilla del prompt (versionada en git, ``§12.8``).
PROMPT_TEMPLATE_NAME: Final[str] = "news_extract.j2"

#: Subdirectorio de las plantillas dentro del paquete ``agents``.
PROMPTS_DIRNAME: Final[str] = "prompts"

#: Directorio por defecto de las plantillas: junto a este modulo.
DEFAULT_TEMPLATE_DIR: Final[Path] = Path(__file__).resolve().parent / PROMPTS_DIRNAME

#: Intentos maximos antes de descartar un lote (``tech_stack.md`` §4.9).
DEFAULT_MAX_ATTEMPTS: Final[int] = 3

#: Contexto que se envia al modelo, por titular. **Solo** titulo y procedencia: nada de posiciones,
#: capital, precios ni salidas del modelo (``tech_stack.md`` §3.2 y §12.7).
HEADLINE_PROMPT_FIELDS: Final[tuple[str, ...]] = (
    "title",
    "headline_hash",
    "feed",
    "published_at",
)

#: Formato del ``headline_hash``: el mismo que ``data/sources/news.py``
#: (reutilizado por import, nunca redefinido).
HEADLINE_HASH_FORMAT: Final[str] = rf"^{re.escape(HASH_PREFIX)}[0-9a-f]{{64}}$"

#: Modo JSON del proveedor: garantiza JSON valido, **no** el esquema (§4.9, aviso 1).
RESPONSE_FORMAT: Final[dict[str, object]] = {"type": "json_object"}


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados
# ─────────────────────────────────────────────────────────────────────────────
class NewsAgentError(Exception):
    """Raiz de los errores del agente de noticias."""


class PromptTemplateError(NewsAgentError):
    """La plantilla del prompt falta o no se puede leer/renderizar."""


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulario tipado del evento
# ─────────────────────────────────────────────────────────────────────────────
class EventType(StrEnum):
    """De que va el evento."""

    MONETARY_POLICY = "monetary_policy"
    INFLATION = "inflation"
    EMPLOYMENT = "employment"
    EARNINGS = "earnings"
    GEOPOLITICS = "geopolitics"
    MARKET_STRUCTURE = "market_structure"
    OTHER = "other"


class Sentiment(StrEnum):
    """Efecto del evento sobre el indice. **No** es la direccion de la operacion (``Direction``)."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Magnitude(StrEnum):
    """Intensidad **ordinal**: el LLM no calcula numeros (§4.9)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Horizon(StrEnum):
    """Ventana en la que se espera que el evento pese."""

    INTRADAY = "intraday"
    DAYS = "days"


class NewsEvent(BaseModel):
    """Un evento extraido de **un** titular del lote, validado contra un esquema cerrado."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    headline_hash: str = Field(
        pattern=HEADLINE_HASH_FORMAT, description="hash de un titular del lote, con prefijo sha256:"
    )
    event_type: EventType
    sentiment: Sentiment
    magnitude: Magnitude
    confidence: float = Field(ge=0.0, le=1.0)
    horizon: Horizon
    rationale: str = Field(min_length=1, description="una frase; la evidencia legible")


class NewsExtraction(BaseModel):
    """El resultado completo de un lote: lo que se acepto y lo que se descarto, con su traza."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    events: tuple[NewsEvent, ...] = ()
    discarded_invalid: int = Field(default=0, ge=0, description="candidatos que no validaron")
    discarded_unknown: int = Field(default=0, ge=0, description="hashes que no son del lote")
    attempts: int = Field(default=0, ge=0, description="peticiones enviadas al cliente")
    prompt_hash: str = Field(description="sha256 del contenido de la plantilla en git")
    model: str = Field(min_length=1)
    system_fingerprint: str | None = None


def headline_context(headlines: Sequence[Headline]) -> tuple[dict[str, str], ...]:
    """El contexto que se envia al modelo: por titular, **solo** las claves declaradas.

    Las claves son exactamente las de ``HEADLINE_PROMPT_FIELDS``: título y procedencia, nada más.
    """
    return tuple(
        {
            "title": headline.title,
            "headline_hash": headline_hash(headline.title),
            "feed": headline.feed,
            "published_at": headline.published_at.isoformat(),
        }
        for headline in headlines
    )


def prompt_hash(template_path: Path | str) -> str:
    """``sha256:<hex>`` del **contenido de la plantilla**, que vive en git (§12.8)."""
    content = Path(template_path).read_bytes()
    return f"{HASH_PREFIX}{hashlib.sha256(content).hexdigest()}"


# ─────────────────────────────────────────────────────────────────────────────
# Interpretacion de la salida del modelo
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _Attempt:
    """Lo que se pudo sacar de **una** respuesta: lo aceptado, lo rechazado y por que.

    ``error is None`` significa "esta respuesta vale": o trae eventos validos, o dice
    explicitamente que no hay ninguno (``{"events": []}``). Con error, se reintenta.
    """

    events: tuple[NewsEvent, ...]
    invalid: int
    error: str | None


def _interpret(content: str) -> _Attempt:
    """Parsea y valida una respuesta cruda del modelo, sin dejar salir texto sin validar."""
    try:
        payload = cast("object", json.loads(content))
    except json.JSONDecodeError as failure:
        return _Attempt(events=(), invalid=1, error=f"la salida no es JSON valido: {failure}")
    if not isinstance(payload, dict):
        return _Attempt(events=(), invalid=1, error="se esperaba un objeto JSON en la raiz")
    raw_events = cast("Mapping[str, object]", payload).get("events")
    if not isinstance(raw_events, list):
        return _Attempt(events=(), invalid=1, error="falta la lista 'events' en la respuesta")
    events: list[NewsEvent] = []
    problems: list[str] = []
    for index, item in enumerate(cast("list[object]", raw_events)):
        try:
            events.append(NewsEvent.model_validate(item))
        except ValidationError as failure:
            problems.append(f"events[{index}]: {failure}")
    if events:
        return _Attempt(events=tuple(events), invalid=len(problems), error=None)
    if problems:
        return _Attempt(events=(), invalid=len(problems), error="\n".join(problems))
    return _Attempt(events=(), invalid=0, error=None)


class NewsAgent:
    """Convierte un lote de :class:`Headline` en :class:`NewsEvent` validados (§4.9)."""

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        template_dir: Path | str | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        if not model.strip():
            raise NewsAgentError("model: se espera un identificador con version, nunca un alias")
        if max_attempts < 1:
            raise NewsAgentError(f"max_attempts: se espera >= 1, no {max_attempts}")
        self._client = client
        self._model = model
        self._max_attempts = max_attempts
        self._template_dir = (
            DEFAULT_TEMPLATE_DIR if template_dir is None else Path(str(template_dir))
        )
        self._template_path = self._template_dir / PROMPT_TEMPLATE_NAME
        if not self._template_path.is_file():
            raise PromptTemplateError(f"falta la plantilla del prompt: {self._template_path}")
        environment = Environment(
            loader=FileSystemLoader(str(self._template_dir)),
            undefined=StrictUndefined,
            # El prompt es texto plano, no HTML: `autoescape` solo corromperia las comillas y el
            # JSON. Se usa `select_autoescape(default=False)` —y no `autoescape=False`— para no
            # disparar el aviso S701 sin mentir sobre lo que hace.
            autoescape=select_autoescape(
                enabled_extensions=(), default=False, default_for_string=False
            ),
            keep_trailing_newline=True,
        )
        try:
            self._template = environment.get_template(PROMPT_TEMPLATE_NAME)
        except TemplateError as failure:
            raise PromptTemplateError(f"{self._template_path}: {failure}") from failure
        self._prompt_hash = prompt_hash(self._template_path)

    @property
    def model(self) -> str:
        """El identificador del modelo que se declara en cada peticion."""
        return self._model

    @property
    def max_attempts(self) -> int:
        """Intentos maximos por lote antes de descartarlo."""
        return self._max_attempts

    @property
    def template_path(self) -> Path:
        """La ruta de la plantilla del prompt (la que se hashea)."""
        return self._template_path

    @property
    def prompt_hash(self) -> str:
        """El ``sha256`` del contenido de la plantilla, calculado al construir el agente."""
        return self._prompt_hash

    def render(self, headlines: Sequence[Headline]) -> tuple[str, str]:
        """``(mensaje de sistema, mensaje de lote)`` ya renderizados; util para inspeccionarlos."""
        return (
            self._render(part="instructions"),
            self._render(part="batch", headlines=headline_context(headlines)),
        )

    def extract(self, headlines: Sequence[Headline]) -> NewsExtraction:
        """Extrae los eventos del lote, reintentando con el error de validacion en el mensaje.

        Un evento que no valida **nunca** sale de aqui: se reintenta y, agotados los intentos,
        el lote se descarta y queda contado. Un evento que referencia un titular que no esta
        en el lote tambien se descarta.
        """
        allowed = frozenset(headline_hash(headline.title) for headline in headlines)
        system, batch = self.render(headlines)
        base = (
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=batch),
        )
        messages = base
        attempts = 0
        discarded_invalid = 0
        model = self._model
        fingerprint: str | None = None
        while attempts < self._max_attempts:
            attempts += 1
            response = self._client.complete(self._request(messages))
            model = response.model
            fingerprint = response.system_fingerprint
            attempt = _interpret(response.content)
            discarded_invalid += attempt.invalid
            if attempt.error is None:
                kept = tuple(event for event in attempt.events if event.headline_hash in allowed)
                return self._outcome(
                    kept,
                    discarded_invalid=discarded_invalid,
                    discarded_unknown=len(attempt.events) - len(kept),
                    attempts=attempts,
                    model=model,
                    fingerprint=fingerprint,
                )
            messages = (*base, *self._correction(response.content, attempt.error))
        return self._outcome(
            (),
            discarded_invalid=discarded_invalid,
            discarded_unknown=0,
            attempts=attempts,
            model=model,
            fingerprint=fingerprint,
        )

    def _outcome(
        self,
        events: tuple[NewsEvent, ...],
        *,
        discarded_invalid: int,
        discarded_unknown: int,
        attempts: int,
        model: str,
        fingerprint: str | None,
    ) -> NewsExtraction:
        return NewsExtraction(
            events=events,
            discarded_invalid=discarded_invalid,
            discarded_unknown=discarded_unknown,
            attempts=attempts,
            prompt_hash=self._prompt_hash,
            model=model,
            system_fingerprint=fingerprint,
        )

    def _request(self, messages: tuple[ChatMessage, ...]) -> LLMRequest:
        return LLMRequest(
            model=self._model, messages=messages, response_format=dict(RESPONSE_FORMAT)
        )

    def _correction(self, content: str, detail: str) -> tuple[ChatMessage, ChatMessage]:
        """El par (respuesta invalida, correccion) que se anyade antes del siguiente intento."""
        return (
            ChatMessage(role="assistant", content=content),
            ChatMessage(role="user", content=self._render(part="correction", detail=detail)),
        )

    def _render(
        self,
        *,
        part: str,
        headlines: tuple[dict[str, str], ...] = (),
        detail: str = "",
    ) -> str:
        """Renderiza una parte de la plantilla; un fallo de plantilla es error tipado."""
        try:
            return self._template.render(part=part, headlines=headlines, error=detail)
        except TemplateError as failure:
            raise PromptTemplateError(f"{self._template_path}: {failure}") from failure
