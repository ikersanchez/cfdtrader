"""Informe diario narrativo con el modelo de mayor calidad — tarea #37.

La pregunta que responde este módulo: **¿cómo se redacta el informe del día —y su
contra-argumento— sin que el LLM calcule ni decida nada?**

El LLM **extrae, veta y redacta** (`tech_stack.md` §4.9); no calcula números ni decide la
dirección. Este agente consume **hechos ya calculados** (:class:`ReportFacts`: dirección,
probabilidad calibrada, EV neto, stop, objetivo, tier y los eventos del día) y devuelve el
informe redactado **más el contra-argumento** del abogado del diablo (§7.1). Nada del texto del
modelo cruza la frontera sin validar contra un esquema cerrado.

Tres garantías, calcadas de :mod:`cfdtrader.agents.news`:

- **Ninguna salida que no valide sale de aquí.** Se valida con Pydantic y se reintenta con el
  error de validación **dentro del mensaje** (§4.9, aviso 1: el proveedor no valida el esquema).
- **El prompt vive en git.** La plantilla (``prompts/daily_report.j2``) está versionada y solo se
  persiste su ``sha256`` (§12.8).
- **El proveedor entra inyectado.** El módulo no importa el SDK, no lee el reloj y no abre red.

El agente **no** sustituye al informe determinista de ``delivery/run_daily.py``: lo enriquece.
Una redacción que no valide se descarta y el camino diario publica su informe de siempre.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Final, cast

from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    Template,
    TemplateError,
    select_autoescape,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cfdtrader.llm.base import ChatMessage, LLMClient, LLMRequest

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DIRECTIONS",
    "PROMPTS_DIRNAME",
    "PROMPT_TEMPLATE_NAME",
    "PromptTemplateError",
    "ReportAgent",
    "ReportAgentError",
    "ReportDraft",
    "ReportFacts",
    "prompt_hash",
]

#: Nombre de la plantilla del prompt (versionada en git, ``§12.8``).
PROMPT_TEMPLATE_NAME: Final[str] = "daily_report.j2"

#: Subdirectorio de las plantillas dentro del paquete ``agents``.
PROMPTS_DIRNAME: Final[str] = "prompts"

#: Directorio por defecto de las plantillas: junto a este modulo.
DEFAULT_TEMPLATE_DIR: Final[Path] = Path(__file__).resolve().parent / PROMPTS_DIRNAME

#: Intentos maximos antes de descartar la redaccion (``tech_stack.md`` §4.9).
DEFAULT_MAX_ATTEMPTS: Final[int] = 3

#: Direcciones admitidas (las mismas de ``backtest.engine``; no se redefinen, se declaran aqui
#: como cadenas para que el modulo no dependa del motor de backtest).
DIRECTIONS: Final[tuple[str, ...]] = ("long", "short", "nothing")

#: Modo JSON del proveedor: garantiza JSON valido, **no** el esquema (§4.9, aviso 1).
RESPONSE_FORMAT: Final[dict[str, object]] = {"type": "json_object"}


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados
# ─────────────────────────────────────────────────────────────────────────────
class ReportAgentError(Exception):
    """Raiz de los errores del agente del informe."""


class PromptTemplateError(ReportAgentError):
    """La plantilla del prompt falta o no se puede leer/renderizar."""


# ─────────────────────────────────────────────────────────────────────────────
# Contratos de datos
# ─────────────────────────────────────────────────────────────────────────────
class ReportFacts(BaseModel):
    """Los **hechos** que el informe debe redactar: ya calculados, congelados y trazables.

    Solo datos del dia: no lleva posiciones, capital ni claves. Todo lo que entra aqui lo produjo
    el propio sistema (gate, calendario, publicaciones macro, resultados de mega-caps).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trade_date: date
    direction: str = Field(min_length=1, description="long | short | nothing")
    prob_up_calibrated: float = Field(ge=0.0, le=1.0)
    expected_move_pct: float = Field(ge=0.0)
    cost_pct: float = Field(ge=0.0)
    ev_net_pct: float | None = None
    stop_pct: float | None = None
    target_pct: float | None = None
    tier: str = ""
    blocking_events: tuple[str, ...] = ()
    day_events: tuple[str, ...] = ()
    publications: tuple[str, ...] = ()
    earnings: tuple[str, ...] = ()


class ReportDraft(BaseModel):
    """El informe redactado: narrativa, los dos casos y la identidad del modelo que lo escribio."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    narrative: str = Field(min_length=1)
    bull_case: tuple[str, ...] = ()
    bear_case: tuple[str, ...] = ()
    prompt_hash: str = Field(description="sha256 del contenido de la plantilla en git")
    model: str = Field(min_length=1)
    system_fingerprint: str | None = None


#: Prefijo obligatorio de los digests del repositorio (detect-secrets: nunca un hex desnudo).
SHA256_PREFIX: Final[str] = "sha256:"


def prompt_hash(template_path: Path) -> str:
    """``sha256:`` + sha256 del **contenido** de la plantilla en git (§12.8)."""
    digest = hashlib.sha256(Path(template_path).read_bytes()).hexdigest()
    return f"{SHA256_PREFIX}{digest}"


class _Draft(BaseModel):
    """El esquema **cerrado** de la respuesta del modelo (lo unico que se acepta)."""

    model_config = ConfigDict(extra="forbid")

    narrative: str = Field(min_length=1)
    bull_case: tuple[str, ...] = ()
    bear_case: tuple[str, ...] = ()


def _require_counter_argument(draft: _Draft, *, direction: str) -> None:
    """El informe **tiene** que traer el contra-argumento del abogado del diablo (§7.1)."""
    if direction == "long" and not draft.bear_case:
        raise ValueError("falta el contra-argumento (`bear_case`) para una direccion long")
    if direction == "short" and not draft.bull_case:
        raise ValueError("falta el contra-argumento (`bull_case`) para una direccion short")
    if direction == "nothing" and not (draft.bull_case or draft.bear_case):
        raise ValueError("falta el contra-argumento para un dia sin operacion")


def _interpret(content: str, *, direction: str) -> _Draft:
    """Interpreta y valida la respuesta del modelo; cualquier fallo es la senal de reintento."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"la respuesta no es JSON valido: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("la respuesta no es un objeto JSON")
    draft = _Draft.model_validate(cast("dict[str, object]", payload))
    _require_counter_argument(draft, direction=direction)
    return draft


class ReportAgent:
    """Redacta el informe del dia con el modelo de mayor calidad (#37).

    Parameters
    ----------
    client:
        El proveedor, **inyectado** como :class:`cfdtrader.llm.base.LLMClient`. Este modulo no
        importa el SDK ni sabe que hay detras.
    model:
        Identificador con version concreta (``LLMClientConfig.model_for("report")``), nunca un
        alias.
    template_dir / template_name:
        La plantilla Jinja2, versionada en el repositorio.
    max_attempts:
        Reintentos ante una salida que no valide (§4.9, aviso 1).
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        template_dir: Path = DEFAULT_TEMPLATE_DIR,
        template_name: str = PROMPT_TEMPLATE_NAME,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self._client = client
        self._model = model
        self._max_attempts = max_attempts
        self._template_path = Path(template_dir) / template_name
        self._template = _load_template(Path(template_dir), template_name)
        self._prompt_hash = prompt_hash(self._template_path)

    @property
    def prompt_hash(self) -> str:
        """El ``sha256`` de la plantilla (lo que va al diario y al ``manifest``)."""
        return self._prompt_hash

    @property
    def template_path(self) -> Path:
        """La ruta de la plantilla versionada."""
        return self._template_path

    def compose(self, facts: ReportFacts) -> ReportDraft:
        """Redacta el informe y su contra-argumento, o falla con error tipado.

        El modelo **no** escribe numeros: los hechos entran ya calculados. Una salida que no
        valide el esquema se reintenta con el error en el mensaje; agotados los intentos, es
        :class:`ReportAgentError` y el camino diario publica su informe determinista.
        """
        system = self._render(part="system")
        user = self._render(part="user", facts=facts)
        base = (
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        )
        messages = base
        attempts = 0
        model = self._model
        fingerprint: str | None = None
        while attempts < self._max_attempts:
            attempts += 1
            response = self._client.complete(self._request(messages))
            model = response.model
            fingerprint = response.system_fingerprint
            try:
                draft = _interpret(response.content, direction=facts.direction)
            except (ValueError, ValidationError) as error:
                messages = (*base, *self._correction(response.content, str(error)))
                continue
            return ReportDraft(
                narrative=draft.narrative,
                bull_case=draft.bull_case,
                bear_case=draft.bear_case,
                prompt_hash=self._prompt_hash,
                model=model,
                system_fingerprint=fingerprint,
            )
        raise ReportAgentError(
            f"la redaccion no valido el esquema tras {attempts} intentos: el informe se descarta"
        )

    def _request(self, messages: tuple[ChatMessage, ...]) -> LLMRequest:
        return LLMRequest(
            model=self._model, messages=messages, response_format=dict(RESPONSE_FORMAT)
        )

    def _correction(self, content: str, detail: str) -> tuple[ChatMessage, ChatMessage]:
        """El par (respuesta invalida, correccion) que se anyade antes del siguiente intento."""
        return (
            ChatMessage(role="assistant", content=content),
            ChatMessage(role="user", content=self._render(part="correction", error=detail)),
        )

    def _render(self, *, part: str, facts: ReportFacts | None = None, error: str = "") -> str:
        """Renderiza una parte de la plantilla; un fallo de plantilla es error tipado."""
        try:
            return self._template.render(part=part, facts=facts, error=error)
        except TemplateError as failure:
            raise PromptTemplateError(f"{self._template_path}: {failure}") from failure


def _load_template(template_dir: Path, template_name: str) -> Template:
    """Carga la plantilla del directorio declarado; un fallo es :class:`PromptTemplateError`."""
    try:
        environment = Environment(
            loader=FileSystemLoader(str(template_dir)),
            undefined=StrictUndefined,
            autoescape=select_autoescape(enabled_extensions=()),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return environment.get_template(template_name)
    except TemplateError as failure:
        raise PromptTemplateError(
            f"no se puede cargar la plantilla {template_name!r}: {failure}"
        ) from failure
