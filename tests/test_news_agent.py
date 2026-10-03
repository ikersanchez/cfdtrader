"""Tests del `NewsAgent` con salida estructurada (tarea #32): A1-A15.

El agente se prueba **sin red y sin proveedor**: el `LLMClient` entra inyectado y simulado. Lo que
se blinda aqui es lo que no puede fallar en silencio:

- Un evento que no valida **no sale** del agente, ni siquiera despues de N intentos (A7-A9).
- Un evento que referencia un titular que no se le dio **se descarta** (A6).
- El prompt vive en git y solo se persiste su hash (A1, A11), sin fijar ningun digest literal.
- El agente **no conoce el proveedor** ni la direccion de la operacion (A3, A12).
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import pytest
from pydantic import ValidationError

from cfdtrader.agents import news as news_module
from cfdtrader.agents.news import (
    DEFAULT_MAX_ATTEMPTS,
    HEADLINE_PROMPT_FIELDS,
    PROMPT_TEMPLATE_NAME,
    EventType,
    Horizon,
    Magnitude,
    NewsAgent,
    NewsAgentError,
    NewsEvent,
    NewsExtraction,
    PromptTemplateError,
    Sentiment,
    headline_context,
    prompt_hash,
)
from cfdtrader.data.sources.news import Headline, headline_hash
from cfdtrader.llm.base import LLMRequest, LLMResponse

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
NEWS_MODULE: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "agents" / "news.py"
PROMPTS_DIR: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "agents" / "prompts"
TEMPLATE: Final[Path] = PROMPTS_DIR / PROMPT_TEMPLATE_NAME
HASH_PATTERN: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")

NOW: Final[datetime] = datetime(2026, 10, 3, 12, 45, tzinfo=UTC)


def _headline(title: str, *, feed: str = "reuters", minutes: int = -5) -> Headline:
    slug = "-".join(title.lower().split())
    return Headline(
        source="rss",
        feed=feed,
        title=title,
        url=f"https://example.invalid/{slug}",
        published_at=NOW + timedelta(minutes=minutes),
    )


HEADLINES: Final[tuple[Headline, ...]] = (
    _headline("La Fed mantiene los tipos sin cambios"),
    _headline("El IPC de septiembre sube mas de lo esperado", feed="bloomberg"),
    _headline("Nvidia presenta sus resultados trimestrales", minutes=-30),
)


def _fields(title: str, **overrides: Any) -> dict[str, Any]:
    """Los campos de un evento valido para ese titular, con lo que se quiera sobrescribir."""
    payload: dict[str, Any] = {
        "headline_hash": headline_hash(title),
        "event_type": "monetary_policy",
        "sentiment": "bearish",
        "magnitude": "high",
        "confidence": 0.7,
        "horizon": "intraday",
        "rationale": "El titular pesa sobre el indice.",
    }
    payload.update(overrides)
    return payload


def _response(*items: object) -> str:
    """Una respuesta del modelo con esos candidatos de evento."""
    return json.dumps({"events": list(items)})


class _ScriptedClient:
    """Cliente LLM de mentira: devuelve respuestas pautadas y cuenta las peticiones."""

    def __init__(
        self,
        *contents: str,
        model: str = "deepseek-chat",
        fingerprint: str | None = "fp-2026-10-03",
    ) -> None:
        self._contents = list(contents)
        self._model = model
        self._fingerprint = fingerprint
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._contents) - 1)
        return LLMResponse(
            content=self._contents[index],
            model=self._model,
            system_fingerprint=self._fingerprint,
            prompt_tokens=120,
            completion_tokens=40,
        )


def _agent(
    client: _ScriptedClient,
    *,
    template_dir: Path | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> NewsAgent:
    return NewsAgent(
        client, model="deepseek-chat", template_dir=template_dir, max_attempts=max_attempts
    )


def _imported_modules(tree: ast.Module) -> set[str]:
    """Los modulos importados por un AST, tanto en `import x` como en `from x import y`."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _imported_names(tree: ast.Module) -> set[str]:
    """Los nombres ligados por `import ... as x` y por `from ... import x as y`."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


# ─────────────────────────────────────────────────────────────────────────────
# A1 · Los ficheros existen y la plantilla vive en git, no en la base de datos
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_files_exist_and_the_prompt_lives_in_git() -> None:
    assert NEWS_MODULE.is_file()
    assert TEMPLATE.is_file()
    listed = subprocess.run(  # noqa: S603 - sin shell, comando fijo
        [  # noqa: S607 - 'git' del PATH, es una comprobacion de repositorio local
            "git",
            "ls-files",
            "--error-unmatch",
            str(TEMPLATE.relative_to(REPO_ROOT)),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert listed.returncode == 0, f"la plantilla no esta versionada en git: {listed.stderr}"


# ─────────────────────────────────────────────────────────────────────────────
# A2 · Esquema cerrado y congelado del evento
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_event_schema_is_closed_and_frozen() -> None:
    assert set(NewsEvent.model_fields) == {
        "headline_hash",
        "event_type",
        "sentiment",
        "magnitude",
        "confidence",
        "horizon",
        "rationale",
    }
    assert NewsEvent.model_config.get("extra") == "forbid"
    assert NewsEvent.model_config.get("frozen") is True

    with pytest.raises(ValidationError):
        NewsEvent.model_validate(_fields(HEADLINES[0].title) | {"campo_de_mas": 1})

    event = NewsEvent.model_validate(_fields(HEADLINES[0].title))
    with pytest.raises(ValidationError):
        event.rationale = "otra frase"  # type: ignore[misc]  # congelado a proposito


# ─────────────────────────────────────────────────────────────────────────────
# A3 · `sentiment` no es `Direction`: el LLM no decide la direccion de la operacion
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_sentiment_does_not_collide_with_direction() -> None:
    tree = ast.parse(NEWS_MODULE.read_text(encoding="utf-8"))

    assert "Direction" not in _imported_names(tree)
    assert not any(module.endswith("backtest.engine") for module in _imported_modules(tree))
    assert "Direction" not in news_module.__all__
    assert {member.value for member in Sentiment} == {"bullish", "bearish", "neutral"}


# ─────────────────────────────────────────────────────────────────────────────
# A4 · El vocabulario es tipado y `magnitude` es ordinal, nunca un numero
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_the_event_vocabulary_is_typed_and_ordinal() -> None:
    for enum_type in (EventType, Sentiment, Magnitude, Horizon):
        assert issubclass(enum_type, StrEnum)
    assert {member.value for member in EventType} == {
        "monetary_policy",
        "inflation",
        "employment",
        "earnings",
        "geopolitics",
        "market_structure",
        "other",
    }
    assert {member.value for member in Magnitude} == {"low", "medium", "high"}
    assert {member.value for member in Horizon} == {"intraday", "days"}

    with pytest.raises(ValidationError):
        NewsEvent.model_validate(_fields(HEADLINES[0].title, magnitude=3))


# ─────────────────────────────────────────────────────────────────────────────
# A5 · El formato del hash es el del repositorio, no uno parecido
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_the_headline_hash_keeps_the_prefix_and_the_length() -> None:
    good = headline_hash(HEADLINES[0].title)
    assert HASH_PATTERN.match(good)

    with pytest.raises(ValidationError):
        NewsEvent.model_validate(
            _fields(HEADLINES[0].title, headline_hash=good.removeprefix("sha256:"))
        )
    with pytest.raises(ValidationError):
        NewsEvent.model_validate(_fields(HEADLINES[0].title, headline_hash="sha256:abc"))
    with pytest.raises(ValidationError):
        NewsEvent.model_validate(_fields(HEADLINES[0].title, headline_hash=""))


# ─────────────────────────────────────────────────────────────────────────────
# A6 · Un evento inventado se descarta: el modelo no referencia lo que no vio
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_an_event_for_an_unknown_headline_is_discarded() -> None:
    foreign = headline_hash("Un titular que nunca se le dio al modelo")
    client = _ScriptedClient(_response(_fields("x", headline_hash=foreign)))

    result = _agent(client).extract(HEADLINES)

    assert result.events == ()
    assert result.discarded_unknown == 1
    assert result.attempts == 1


# ─────────────────────────────────────────────────────────────────────────────
# A7 · El reintento lleva el error de validacion dentro del mensaje
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_the_retry_carries_the_validation_error_in_the_message() -> None:
    broken = _response(_fields(HEADLINES[0].title, sentiment="muy_bajista"))
    client = _ScriptedClient(broken, _response(_fields(HEADLINES[0].title)))

    result = _agent(client).extract(HEADLINES)

    assert [event.headline_hash for event in result.events] == [headline_hash(HEADLINES[0].title)]
    assert result.attempts == 2
    assert len(client.requests) == 2

    second = "\n".join(message.content for message in client.requests[1].messages)
    assert "Input should be" in second, "el error de validacion tiene que viajar en el mensaje"
    assert broken in second, "la respuesta cruda se reenvia para que el modelo la corrija"


# ─────────────────────────────────────────────────────────────────────────────
# A8 · Agotados los intentos se descarta; `extract` no lanza
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_nothing_valid_after_n_attempts_is_discarded_not_raised() -> None:
    client = _ScriptedClient("no soy json")

    result = _agent(client).extract(HEADLINES)

    assert result.events == ()
    assert result.discarded_invalid > 0
    assert result.attempts == DEFAULT_MAX_ATTEMPTS
    assert len(client.requests) == DEFAULT_MAX_ATTEMPTS

    short = _ScriptedClient("tampoco soy json")
    assert _agent(short, max_attempts=2).extract(HEADLINES).attempts == 2
    assert len(short.requests) == 2


# ─────────────────────────────────────────────────────────────────────────────
# A9 · Mezcla de valido e invalido: solo sale el valido
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_a_mixed_batch_keeps_only_the_valid_event() -> None:
    mixed = _response(
        _fields(HEADLINES[0].title),
        _fields(HEADLINES[1].title, confidence=7.0),
    )
    client = _ScriptedClient(mixed)

    result = _agent(client).extract(HEADLINES)

    assert [event.headline_hash for event in result.events] == [headline_hash(HEADLINES[0].title)]
    assert result.discarded_invalid == 1
    assert result.attempts == 1
    assert all(isinstance(event, NewsEvent) for event in result.events)


# ─────────────────────────────────────────────────────────────────────────────
# A10 · Batching: un lote, una llamada (palanca 1 de §6.3)
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_one_call_per_batch() -> None:
    client = _ScriptedClient(_response(*[_fields(headline.title) for headline in HEADLINES]))

    result = _agent(client).extract(HEADLINES)

    assert len(client.requests) == 1
    assert len(result.events) == len(HEADLINES)


# ─────────────────────────────────────────────────────────────────────────────
# A11 · El hash del prompt: formato, autoconsistencia y determinismo
# ─────────────────────────────────────────────────────────────────────────────
def _hash_in_subprocess(seed: str) -> str:
    """El `prompt_hash` calculado en **otro** proceso, con ese `PYTHONHASHSEED`."""
    code = f"from cfdtrader.agents.news import prompt_hash; print(prompt_hash({str(TEMPLATE)!r}))"
    completed = subprocess.run(  # noqa: S603 - el ejecutable es el interprete de la sesion
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": seed},
    )
    return completed.stdout.strip()


def test_a11_the_prompt_hash_is_consistent_and_deterministic(tmp_path: Path) -> None:
    agent = _agent(_ScriptedClient("{}"))

    assert HASH_PATTERN.match(agent.prompt_hash)
    assert agent.prompt_hash == prompt_hash(TEMPLATE), "autoconsistencia con la plantilla de git"

    edited = tmp_path / PROMPT_TEMPLATE_NAME
    edited.write_bytes(TEMPLATE.read_bytes() + b"\n{# un cambio cualquiera #}\n")
    assert prompt_hash(edited) != agent.prompt_hash, "el hash sigue al contenido"

    assert {_hash_in_subprocess("0"), _hash_in_subprocess("1")} == {agent.prompt_hash}
    # Ningun digest literal en la prueba: la plantilla es un artefacto vivo.


# ─────────────────────────────────────────────────────────────────────────────
# A12 · El agente no conoce el proveedor
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_the_agent_does_not_know_the_provider() -> None:
    modules = _imported_modules(ast.parse(NEWS_MODULE.read_text(encoding="utf-8")))

    assert "openai" not in modules
    assert not any(module.endswith("llm.openai_client") for module in modules)

    code = "import sys, cfdtrader.agents.news; print('openai' in sys.modules)"
    completed = subprocess.run(  # noqa: S603 - el ejecutable es el interprete de la sesion
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "False", "cargar el agente no arrastra el SDK"


# ─────────────────────────────────────────────────────────────────────────────
# A13 · Solo titulares publicos: nada de posiciones, capital ni claves
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_only_public_headlines_are_sent() -> None:
    context = headline_context(HEADLINES)
    assert context
    assert all(set(item) == set(HEADLINE_PROMPT_FIELDS) for item in context)
    assert [item["title"] for item in context] == [headline.title for headline in HEADLINES]

    system, batch = _agent(_ScriptedClient("{}")).render(HEADLINES)
    for headline in HEADLINES:
        assert headline.title in batch
        assert headline.feed in batch

    forbidden = (
        "notional",
        "capital",
        "equity",
        "position",
        "api_key",
        "prob_up",
        "stop_pct",
        "size_",
        "leverage",
    )
    rendered = f"{system}\n{batch}".lower()
    assert [token for token in forbidden if token in rendered] == []


# ─────────────────────────────────────────────────────────────────────────────
# A14 · Determinismo con el cliente simulado
# ─────────────────────────────────────────────────────────────────────────────
def test_a14_the_same_batch_gives_the_same_extraction() -> None:
    content = _response(*[_fields(headline.title) for headline in HEADLINES])

    first = _agent(_ScriptedClient(content)).extract(HEADLINES)
    second = _agent(_ScriptedClient(content)).extract(HEADLINES)

    assert isinstance(first, NewsExtraction)
    assert first.model_dump() == second.model_dump()
    assert first.model == "deepseek-chat"
    assert first.system_fingerprint == "fp-2026-10-03"


# ─────────────────────────────────────────────────────────────────────────────
# Bordes: errores tipados y sobres de respuesta malformados
# ─────────────────────────────────────────────────────────────────────────────
def test_the_configuration_errors_are_typed(tmp_path: Path) -> None:
    with pytest.raises(PromptTemplateError):
        _agent(_ScriptedClient("{}"), template_dir=tmp_path)

    with pytest.raises(NewsAgentError):
        NewsAgent(_ScriptedClient("{}"), model="   ")
    with pytest.raises(NewsAgentError):
        _agent(_ScriptedClient("{}"), max_attempts=0)

    agent = _agent(_ScriptedClient("{}"))
    assert agent.model == "deepseek-chat"
    assert agent.max_attempts == DEFAULT_MAX_ATTEMPTS
    assert agent.template_path == TEMPLATE
    assert agent.prompt_hash == prompt_hash(TEMPLATE)


def test_an_empty_batch_is_accepted_and_a_broken_envelope_is_not() -> None:
    empty = _agent(_ScriptedClient('{"events": []}')).extract(HEADLINES)
    assert empty.events == ()
    assert empty.attempts == 1
    assert empty.discarded_invalid == 0

    for payload in ("[]", '{"otros": []}', "{}"):
        result = _agent(_ScriptedClient(payload)).extract(HEADLINES)
        assert result.events == ()
        assert result.attempts == DEFAULT_MAX_ATTEMPTS
