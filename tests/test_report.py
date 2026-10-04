"""Tests del informe diario narrativo (#37).

Ninguna prueba abre red: el ``LLMClient`` se simula. El artefacto verificable es el informe
redactado —con su contra-argumento— validado contra un esquema cerrado, y su ``prompt_hash``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from cfdtrader.agents import report as report_module
from cfdtrader.agents.report import (
    PROMPT_TEMPLATE_NAME,
    PromptTemplateError,
    ReportAgent,
    ReportAgentError,
    ReportFacts,
    prompt_hash,
)
from cfdtrader.llm.base import LLMRequest, LLMResponse

#: Los hechos de un dia cualquiera, ya calculados (el agente no calcula numeros).
FACTS = ReportFacts(
    trade_date=date(2026, 10, 2),
    direction="long",
    prob_up_calibrated=0.57,
    expected_move_pct=1.2,
    cost_pct=0.0042,
    ev_net_pct=0.31,
    stop_pct=0.6,
    target_pct=0.9,
    tier="B",
    blocking_events=("dia_de_fomc",),
    day_events=("evento: opex | vencimiento mensual de opciones",),
    publications=("publicacion_macro: CPIAUCSL | CPI",),
    earnings=("resultado_mega_cap: NVDA | NVIDIA",),
)


class _FakeClient:
    """Un cliente simulado: devuelve las respuestas encoladas y registra las peticiones."""

    def __init__(self, *contents: str) -> None:
        self._contents = list(contents)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        content = self._contents.pop(0)
        return LLMResponse(content=content, model="fake-model-v1", system_fingerprint="fp-abc")


def _valid(**overrides: object) -> str:
    payload: dict[str, object] = {
        "narrative": "El sistema no ve edge demostrado; la pista es apoyo a la decision.",
        "bull_case": ["el soporte aguanta", "el VIX cede"],
        "bear_case": ["la subasta falla", "el dato macro decepciona"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_37_the_report_includes_the_counter_argument_and_the_identity() -> None:
    """El informe trae narrativa, contra-argumento y la identidad del modelo que lo escribio."""
    client = _FakeClient(_valid())

    draft = ReportAgent(client, model="report-model-v2").compose(FACTS)

    assert draft.narrative.startswith("El sistema no ve edge")
    assert draft.bull_case and draft.bear_case
    assert draft.model == "fake-model-v1"
    assert draft.system_fingerprint == "fp-abc"
    assert draft.prompt_hash.startswith("sha256:")
    assert len(draft.prompt_hash) == len("sha256:") + 64


def test_37_the_facts_travel_in_the_prompt_without_inventing_numbers() -> None:
    """Los hechos ya calculados viajan al modelo: el LLM no calcula, redacta."""
    client = _FakeClient(_valid())

    ReportAgent(client, model="m").compose(FACTS)

    user = client.requests[0].messages[1].content
    assert "long" in user
    assert "0.57" in user
    assert "dia_de_fomc" in user
    assert "CPIAUCSL" in user and "NVDA" in user
    # El modelo declarado es el de mayor calidad, con version concreta.
    assert client.requests[0].model == "m"
    assert client.requests[0].response_format == {"type": "json_object"}


def test_37_a_missing_counter_argument_is_retried_and_then_discarded() -> None:
    """Sin contra-argumento el informe no vale: reintenta con el error y, agotado, se descarta."""
    client = _FakeClient(_valid(bear_case=[]), _valid(bear_case=[]), _valid(bear_case=[]))

    with pytest.raises(ReportAgentError, match="esquema"):
        ReportAgent(client, model="m", max_attempts=3).compose(FACTS)

    assert len(client.requests) == 3, "los tres intentos se enviaron"


def test_37_an_invalid_response_is_retried_with_the_error_in_the_message() -> None:
    """La primera salida no valida; la segunda si. El error viaja en la correccion."""
    client = _FakeClient("{no es json", _valid())

    draft = ReportAgent(client, model="m").compose(FACTS)

    assert draft.narrative
    assert len(client.requests) == 2
    correction = client.requests[1].messages[-1].content
    assert "validacion" in correction.lower() or "error" in correction.lower()


def test_37_an_extra_key_is_a_closed_schema_failure() -> None:
    """El esquema es cerrado: una clave de mas no vale y se reintenta."""
    with pytest.raises(ReportAgentError):
        ReportAgent(
            _FakeClient(_valid(extra=1), _valid(extra=1), _valid(extra=1)), model="m"
        ).compose(FACTS)


def test_37_the_prompt_hash_is_the_template_digest() -> None:
    """``prompt_hash`` es el sha256 del **contenido** de la plantilla versionada."""
    agent = ReportAgent(_FakeClient(_valid()), model="m")

    expected = "sha256:" + hashlib.sha256(agent.template_path.read_bytes()).hexdigest()

    assert agent.prompt_hash == expected
    assert prompt_hash(agent.template_path) == expected
    assert PROMPT_TEMPLATE_NAME == "daily_report.j2"


def test_37_a_missing_template_is_a_typed_error(tmp_path: Path) -> None:
    """Un directorio sin la plantilla es error tipado, nunca un fallo a mitad del pipeline."""
    with pytest.raises(PromptTemplateError):
        ReportAgent(_FakeClient(_valid()), model="m", template_dir=tmp_path)


def test_37_the_module_has_no_clock_and_no_network() -> None:
    """El modulo no consulta el reloj, no abre red y no importa el SDK del proveedor."""
    source = Path(report_module.__file__).read_text(encoding="utf-8")
    for forbidden in ("datetime.now", "date.today", "time.time"):
        assert forbidden not in source, forbidden
    for network in ("import yfinance", "import requests", "import urllib", "import openai"):
        assert network not in source, network
