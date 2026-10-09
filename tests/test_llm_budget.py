"""Tests de topes, registro y palancas de coste del LLM (tarea #33): B6-B15.

Sin red y sin proveedor. Lo que se blinda aquí:

- Cada llamada deja **una** fila en `ops.llm_calls` con el esquema cerrado de #39 (B6, B7).
- Superar cualquier tope **no lanza**: desactiva y el sistema sigue (B9), que es la regla dura de
  §6.3. El corte mensual además avisa (B10).
- Los tres estados de corte son los de §12.5 y no se inventa vocabulario (B6, B11).
- Las palancas de §6.3 se implementan **reutilizando** lo que ya existe (B13).

El end-to-end («la recomendación del día es la misma con y sin presupuesto») es de #35, que es quien
integra el overlay en el gate; aquí se prueba la capa que #33 posee.
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import pytest

from cfdtrader.agents.news import NewsAgent
from cfdtrader.data.news import deduplicate as news_deduplicate
from cfdtrader.data.sources.news import Headline
from cfdtrader.data.store import WriteOutcome
from cfdtrader.journal.decision_log import (
    LLM_OVERLAYS,
    TABLE_COLUMNS,
    ClosedSchemaError,
    Journal,
)
from cfdtrader.llm.base import ChatMessage, LLMClientConfig, LLMError, LLMRequest, LLMResponse
from cfdtrader.llm.budget import (
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
    KNOWN_SYSTEM_FINGERPRINTS,
    MONTHLY_WARNING,
    PRICES_EUR_PER_MTOKENS,
    PRICES_VERIFIED_ON,
    BatchCounts,
    BudgetCaps,
    BudgetGuard,
    CallSequence,
    Caps,
    LLMCall,
    MeteredLLMClient,
    OverlayClient,
    OverlayState,
    estimate_cost,
    fingerprint_warning,
    prepare_batch,
    prepare_headlines,
    record_call,
)
from cfdtrader.llm.cache import CachedLLMClient, ResponseCache

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
NOW: Final[datetime] = datetime(2026, 10, 3, 12, 45, tzinfo=UTC)


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


class _FakeClock:
    """Reloj monótono de mentira: el tiempo se agota sin dormir en el test."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _headline(title: str, *, minutes: int = -5) -> Headline:
    return Headline(
        source="rss",
        feed="qa",
        title=title,
        url=f"https://example.invalid/{abs(len(title))}",
        published_at=NOW + timedelta(minutes=minutes),
    )


# ─────────────────────────────────────────────────────────────────────────────
# B6 · El registro en `ops.llm_calls` usa el esquema que declaro #39
# ─────────────────────────────────────────────────────────────────────────────
def test_b6_the_row_uses_the_schema_declared_by_39(tmp_path: Path) -> None:
    call = LLMCall(
        call_id="20261003T124500Z-0001",
        as_of=NOW,
        provider="deepseek",
        model="deepseek-chat",
        purpose="extract",
        tokens_in=120,
        tokens_out=40,
        cache_hit=False,
        cost_estimate=None,
        latency_ms=7,
        ok=True,
    )
    assert set(call.payload()) == set(TABLE_COLUMNS["llm_calls"])

    handle = Journal(tmp_path)
    assert record_call(handle, call) == WriteOutcome.CREATED
    assert (tmp_path / "llm_calls" / "20261003T124500Z-0001.json").is_file()

    with pytest.raises(ClosedSchemaError):
        handle.write("llm_calls", {**call.payload(), "columna_de_mas": 1})


def test_b6_the_overlay_vocabulary_is_the_one_39_declared() -> None:
    assert {member.value for member in OverlayState} == set(LLM_OVERLAYS)


# ─────────────────────────────────────────────────────────────────────────────
# B7 · La identidad de la fila es unica por llamada
# ─────────────────────────────────────────────────────────────────────────────
def test_b7_two_calls_in_the_same_instant_are_two_rows(tmp_path: Path) -> None:
    handle = Journal(tmp_path)
    first = LLMCall(call_id="20261003T124500Z-0001", as_of=NOW, provider="p", model="m")
    second = LLMCall(call_id="20261003T124500Z-0002", as_of=NOW, provider="p", model="m")

    assert record_call(handle, first) == WriteOutcome.CREATED
    assert record_call(handle, second) == WriteOutcome.CREATED
    assert len(list((tmp_path / "llm_calls").glob("*.json"))) == 2
    assert record_call(handle, second) == WriteOutcome.UNCHANGED


def test_b7_the_metered_client_builds_a_unique_id_per_call(tmp_path: Path) -> None:
    guard = BudgetGuard()
    with ResponseCache(tmp_path / "cache") as cache:
        metered = MeteredLLMClient(
            _CountingClient(_response(), _response("otra")),
            cache=cache,
            guard=guard,
            as_of=NOW,
            provider="deepseek",
            journal=tmp_path,
        )
        metered.call(_request(inputs="lote A"))
        metered.call(_request(inputs="lote B"))

    rows = sorted(path.name for path in (tmp_path / "llm_calls").glob("*.json"))
    assert rows == ["20261003T124500Z-0001.json", "20261003T124500Z-0002.json"]


def test_b7_two_clients_can_share_one_call_sequence(tmp_path: Path) -> None:
    """#147: dos clientes del **mismo** instante comparten contador y no colisionan de identidad.

    Es el caso del camino diario: la extraccion de noticias (#35) y la redaccion (#37) usan el
    mismo ``as_of``. Con contadores independientes, la primera llamada de cada uno pediria
    ``…-0001`` y el diario —inmutable por identidad— rechazaria la segunda fila. Compartir el
    contador da a cada llamada de la ejecucion un numero distinto y conserva el proposito.
    """
    sequence = CallSequence()
    with (
        ResponseCache(tmp_path / "cache-extract") as cache_extract,
        ResponseCache(tmp_path / "cache-report") as cache_report,
    ):
        extract = MeteredLLMClient(
            _CountingClient(_response()),
            cache=cache_extract,
            guard=BudgetGuard(),
            as_of=NOW,
            provider="deepseek",
            purpose="extract",
            journal=tmp_path,
            sequence=sequence,
        )
        report = MeteredLLMClient(
            _CountingClient(_response('{"narrative": "informe"}')),
            cache=cache_report,
            guard=BudgetGuard(),
            as_of=NOW,
            provider="deepseek",
            purpose="report",
            journal=tmp_path,
            sequence=sequence,
        )
        extract.call(_request(inputs="lote de titulares"))
        report.call(_request(inputs="hechos del dia"))

    files = sorted((tmp_path / "llm_calls").glob("*.json"))
    assert [path.name for path in files] == [
        "20261003T124500Z-0001.json",
        "20261003T124500Z-0002.json",
    ]
    purposes = {json.loads(path.read_text(encoding="utf-8"))["purpose"] for path in files}
    assert purposes == {"extract", "report"}, "cada llamada conserva su proposito"


def test_b7_a_cache_hit_row_declares_no_tokens(tmp_path: Path) -> None:
    """#119: la fila y el guardian dicen lo mismo, y antes de este criterio no lo decian.

    La segunda llamada es identica, asi que la absorbe la cache. Su fila declara
    ``cache_hit: true`` y **ningun** token: no se gasto ninguno. Copiar los tokens de la respuesta
    cacheada inflaba el agregado de ``ops.llm_calls`` justo en la direccion contraria a la util,
    porque cuanto mejor funciona la cache mas mentiria el informe mensual.
    """
    guard = BudgetGuard()
    with ResponseCache(tmp_path / "cache") as cache:
        metered = MeteredLLMClient(
            _CountingClient(_response()),
            cache=cache,
            guard=guard,
            as_of=NOW,
            provider="deepseek",
            journal=tmp_path,
        )
        metered.call(_request())
        metered.call(_request())

    handle = Journal(tmp_path)
    real = handle.read("llm_calls", "20261003T124500Z-0001")
    cached = handle.read("llm_calls", "20261003T124500Z-0002")

    assert real["cache_hit"] is False
    assert (real["tokens_in"], real["tokens_out"]) == (120, 40)
    assert cached["cache_hit"] is True
    assert (cached["tokens_in"], cached["tokens_out"]) == (None, None)
    assert guard.tokens_used == 160, "el guardian cuenta los tokens una sola vez"


# ─────────────────────────────────────────────────────────────────────────────
# B8 · Los seis topes de §6.3.4, declarados y configurables por entorno
# ─────────────────────────────────────────────────────────────────────────────
def test_b8_the_six_caps_are_declared_and_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = Caps()
    assert base.max_consecutive_failures == DEFAULT_MAX_CONSECUTIVE_FAILURES == 3
    for name in (
        "max_tokens_per_run",
        "max_calls_per_run",
        "daily_budget_eur",
        "monthly_budget_eur",
        "max_seconds",
    ):
        assert getattr(base, name) is None, f"{name} no debe traer un tope inventado"

    monkeypatch.setenv("LLM_MAX_CALLS_PER_RUN", "5")
    monkeypatch.setenv("LLM_DAILY_BUDGET_EUR", "1.5")
    monkeypatch.setenv("LLM_MAX_SECONDS", "120")
    caps = BudgetCaps(_env_file=None).caps()  # pyright: ignore[reportCallIssue]

    assert caps.max_calls_per_run == 5
    assert caps.daily_budget_eur == 1.5
    assert caps.max_seconds == 120.0


def test_b8_an_empty_variable_means_no_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """`.env.example` trae los topes en blanco: vacio es «sin tope», no un error de parseo.

    Lo cazo el cableado de #35: la prueba original construia los topes con `_env_file=None`, asi
    que nunca veia el `.env` que el repo distribuye.
    """
    monkeypatch.setenv("LLM_DAILY_BUDGET_EUR", "")
    monkeypatch.setenv("LLM_MAX_CALLS_PER_RUN", "")
    monkeypatch.setenv("LLM_MAX_SECONDS", "")
    monkeypatch.setenv("LLM_MONTHLY_BUDGET_EUR", "")

    caps = BudgetCaps(_env_file=None).caps()  # pyright: ignore[reportCallIssue]

    assert caps.daily_budget_eur is None
    assert caps.max_calls_per_run is None
    assert caps.max_seconds is None
    assert caps.monthly_budget_eur is None


# ─────────────────────────────────────────────────────────────────────────────
# B9 · Superar un tope desactiva; nunca lanza
# ─────────────────────────────────────────────────────────────────────────────
def test_b9_a_spent_budget_disables_and_never_raises(tmp_path: Path) -> None:
    client = _CountingClient(_response())
    guard = BudgetGuard(caps=Caps(max_calls_per_run=0))

    with ResponseCache(tmp_path / "cache") as cache:
        metered = MeteredLLMClient(client, cache=cache, guard=guard, as_of=NOW, provider="deepseek")
        assert metered.call(_request()) is None, "un tope agotado no puede lanzar"
        assert metered.call(_request()) is None

    assert metered.state is OverlayState.DISABLED_BUDGET
    assert client.calls == 0, "ni una llamada al proveedor con el presupuesto agotado"
    assert metered.warnings


def test_b9_the_cost_layer_does_not_touch_the_gate() -> None:
    source = (REPO_ROOT / "src" / "cfdtrader" / "llm" / "budget.py").read_text(encoding="utf-8")
    modules = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not [
        name for name in modules if name.startswith(("cfdtrader.decision", "cfdtrader.delivery"))
    ]


# ─────────────────────────────────────────────────────────────────────────────
# B10 · El corte mensual avisa; el diario no
# ─────────────────────────────────────────────────────────────────────────────
def test_b10_the_monthly_cut_warns_and_the_daily_one_does_not(tmp_path: Path) -> None:
    seen: list[str] = []
    guard = BudgetGuard(caps=Caps(monthly_budget_eur=0.001))
    guard.record(cache_hit=False, cost_eur=1.0)

    with ResponseCache(tmp_path / "cache") as cache:
        metered = MeteredLLMClient(
            _CountingClient(_response()),
            cache=cache,
            guard=guard,
            as_of=NOW,
            provider="p",
            warning_sink=seen.append,
        )
        assert metered.call(_request()) is None

    assert metered.state is OverlayState.DISABLED_BUDGET
    assert MONTHLY_WARNING in metered.warnings
    assert MONTHLY_WARNING in seen, "el aviso tiene que salir hacia fuera"

    daily = BudgetGuard(caps=Caps(daily_budget_eur=0.0))
    decision = daily.check()
    assert decision.state is OverlayState.DISABLED_BUDGET
    assert decision.warnings
    assert MONTHLY_WARNING not in decision.warnings


# ─────────────────────────────────────────────────────────────────────────────
# B11 · Fallos consecutivos y tiempo: dos estados distintos
# ─────────────────────────────────────────────────────────────────────────────
def test_b11_three_consecutive_failures_disable_with_disabled_error(tmp_path: Path) -> None:
    client = _CountingClient(*([LLMError("boom")] * 5))
    guard = BudgetGuard(caps=Caps(max_consecutive_failures=3))

    with ResponseCache(tmp_path / "cache") as cache:
        metered = MeteredLLMClient(client, cache=cache, guard=guard, as_of=NOW, provider="p")
        for _ in range(3):
            with pytest.raises(LLMError):
                metered.call(_request())
        assert client.calls == 3

        assert metered.call(_request()) is None, "el cuarto intento ya no debe salir"

    assert metered.state is OverlayState.DISABLED_ERROR
    assert client.calls == 3


def test_b11_a_cache_hit_does_not_reset_the_failure_counter() -> None:
    guard = BudgetGuard(caps=Caps(max_consecutive_failures=3))
    guard.record(cache_hit=False, ok=False)
    guard.record(cache_hit=False, ok=False)
    assert guard.consecutive_failures == 2

    guard.record(cache_hit=True)

    assert guard.consecutive_failures == 2, "un acierto no es una llamada"
    assert guard.calls_made == 2


def test_b11_an_exhausted_clock_is_a_timeout_and_not_an_error() -> None:
    clock = _FakeClock()
    guard = BudgetGuard(caps=Caps(max_seconds=1.0), clock=clock)
    assert guard.check().state is OverlayState.APPLIED

    clock.advance(1.0)
    decision = guard.check()

    assert decision.state is OverlayState.DISABLED_TIMEOUT
    assert decision.state is not OverlayState.DISABLED_ERROR


# ─────────────────────────────────────────────────────────────────────────────
# El coste: sin tarifa declarada no se inventa un cero
# ─────────────────────────────────────────────────────────────────────────────
def test_the_cost_is_none_without_a_declared_price() -> None:
    assert estimate_cost(model="deepseek-chat", tokens_in=1000, tokens_out=500) is None

    prices = {"deepseek-chat": (0.5, 1.5)}
    assert estimate_cost(
        model="deepseek-chat", tokens_in=1_000_000, tokens_out=1_000_000, prices=prices
    ) == pytest.approx(2.0)
    assert estimate_cost(model="otro", tokens_in=1, tokens_out=1, prices=prices) is None


def test_a12_the_overlay_client_bridge_is_public(tmp_path: Path) -> None:
    """El puente que el agente necesita: `call()` de la capa metrada -> `complete()` del agente.

    Antes de #35 este adaptador no existia y hubo que escribirlo en un script de un solo uso para
    poder ejecutar la cadena de punta a punta.
    """
    with ResponseCache(tmp_path / "cache") as cache:
        bridge = OverlayClient(
            MeteredLLMClient(
                _CountingClient(_response("vale")),
                cache=cache,
                guard=BudgetGuard(),
                as_of=NOW,
                provider="deepseek",
            )
        )
        assert bridge.state is OverlayState.APPLIED
        assert bridge.complete(_request()).content == "vale"
        assert bridge.last_cache_hit is False

        stopped = OverlayClient(
            MeteredLLMClient(
                _CountingClient(_response()),
                cache=cache,
                guard=BudgetGuard(caps=Caps(max_calls_per_run=0)),
                as_of=NOW,
                provider="deepseek",
            )
        )
        with pytest.raises(LLMError, match="no puede responder"):
            stopped.complete(_request(inputs="otra peticion"))
        assert stopped.state is OverlayState.DISABLED_BUDGET


# ─────────────────────────────────────────────────────────────────────────────
# B12 · Palancas 1, 2 y 4 de §6.3
# ─────────────────────────────────────────────────────────────────────────────
def test_b12_the_cheap_model_comes_from_the_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MODEL_EXTRACT", "modelo-barato-2026-01")
    config = LLMClientConfig(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert config.model_for("extract") == "modelo-barato-2026-01"


def test_b12_one_call_per_batch_and_the_cache_absorbs_the_second(tmp_path: Path) -> None:
    headlines = tuple(_headline(f"Titular numero {index}") for index in range(3))
    client = _CountingClient(_response())

    with ResponseCache(tmp_path / "cache") as cache:
        agent = NewsAgent(CachedLLMClient(client, cache), model="deepseek-chat")
        first = agent.extract(headlines)
        second = agent.extract(headlines)

    assert first.attempts == 1
    assert second.attempts == 1
    assert client.calls == 1, "tres titulares son una llamada, y la repeticion ninguna"


# ─────────────────────────────────────────────────────────────────────────────
# #118 · La tarifa declarada, el tope en euros y la huella del modelo
# ─────────────────────────────────────────────────────────────────────────────
#: La huella declarada, **sin** volver a escribir el literal: la fuente es el mapa del modulo.
_DECLARED_FINGERPRINT: Final[str] = KNOWN_SYSTEM_FINGERPRINTS["deepseek-flash"]


def _priced(
    model: str = "deepseek-flash",
    *,
    fingerprint: str = _DECLARED_FINGERPRINT,
) -> LLMResponse:
    """Una respuesta con el modelo que el proveedor **resuelve de verdad**, no el que se pide."""
    return LLMResponse(
        content='{"events": []}',
        model=model,
        system_fingerprint=fingerprint,
        prompt_tokens=814,
        completion_tokens=1730,
    )


def test_118_the_price_table_is_by_concrete_id_and_declares_its_date() -> None:
    """Sin identificadores de version concreta, la tabla es la unica cifra verificable."""
    assert set(PRICES_EUR_PER_MTOKENS) == {"deepseek-flash", "deepseek-v4-pro"}
    for model, (entrada, salida) in PRICES_EUR_PER_MTOKENS.items():
        assert entrada > 0 and salida > 0, model
        assert salida > entrada, f"{model}: la salida es la parte cara y debe costar mas"
    assert PRICES_VERIFIED_ON == "2026-10-03"
    assert set(KNOWN_SYSTEM_FINGERPRINTS) == {"deepseek-flash"}
    declared = KNOWN_SYSTEM_FINGERPRINTS["deepseek-flash"]
    assert len(declared) == 32 and all(char in "0123456789abcdef" for char in declared)


def test_118_the_declared_tariff_turns_tokens_into_euros() -> None:
    """Con tarifa, el coste deja de ser ``null``: es el numero que #117 podra agregar."""
    cost = estimate_cost(model="deepseek-flash", tokens_in=814, tokens_out=1730)
    assert cost is not None
    assert cost == pytest.approx((814 * 0.133630 + 1730 * 0.534521) / 1_000_000)
    assert cost > 0
    assert estimate_cost(model="modelo-sin-tarifa", tokens_in=814, tokens_out=1730) is None
    assert estimate_cost(model="deepseek-flash", tokens_in=None, tokens_out=None) is None


def test_118_the_eur_cap_bites_and_the_pipeline_never_blocks(tmp_path: Path) -> None:
    """El B9 de #33 en euros: superar el tope diario desactiva el overlay, **no** el pipeline."""
    guard = BudgetGuard(caps=Caps(daily_budget_eur=0.0001))
    with ResponseCache(tmp_path / "cache") as cache:
        metered = MeteredLLMClient(
            _CountingClient(_priced(), _priced()),
            cache=cache,
            guard=guard,
            as_of=NOW,
            provider="deepseek",
            journal=tmp_path,
        )
        assert metered.call(_request()) is not None
        assert guard.daily_spend_eur > 0.0, "hay tarifa declarada, asi que el gasto se acumula"
        assert metered.call(_request(inputs="otro lote")) is None, "no lanza: devuelve None"
        assert metered.state is OverlayState.DISABLED_BUDGET

    assert any("gasto diario" in warning for warning in metered.warnings)


def test_118_a_fingerprint_change_warns_and_a_known_one_stays_silent() -> None:
    """El alias es movil: lo unico verificable es la huella, y solo se avisa de lo conocido."""
    assert fingerprint_warning(model="deepseek-flash", fingerprint=_DECLARED_FINGERPRINT) is None
    assert fingerprint_warning(model="deepseek-flash", fingerprint=None) is None
    assert fingerprint_warning(model="modelo-no-observado", fingerprint="ffff") is None
    warning = fingerprint_warning(model="deepseek-flash", fingerprint="ffff")
    assert warning is not None
    assert "system_fingerprint" in warning


def test_118_the_fingerprint_change_reaches_the_warning_sink(tmp_path: Path) -> None:
    seen: list[str] = []
    with ResponseCache(tmp_path / "cache") as cache:
        metered = MeteredLLMClient(
            _CountingClient(_priced(fingerprint="ffffffffffff")),
            cache=cache,
            guard=BudgetGuard(),
            as_of=NOW,
            provider="deepseek",
            journal=tmp_path,
            warning_sink=seen.append,
        )
        assert metered.call(_request()) is not None

    assert seen, "un cambio de huella tiene que llegar al aviso, no quedarse en el codigo"
    assert "system_fingerprint" in seen[0]


# ─────────────────────────────────────────────────────────────────────────────
# B13 · La palanca 3 reutiliza la deduplicacion de #30
# ─────────────────────────────────────────────────────────────────────────────
def test_b13_the_dedup_is_reused_and_not_reimplemented() -> None:
    from cfdtrader.llm import budget as budget_module

    assert budget_module.deduplicate is news_deduplicate


def test_b13_near_duplicate_headlines_collapse_before_the_call() -> None:
    batch = (
        _headline("La Fed sube los tipos"),
        _headline("¡La FED sube los tipos!"),
        _headline("Nvidia presenta resultados"),
    )
    prepared = prepare_headlines(batch, now=NOW)
    assert len(prepared) == 2


# ─────────────────────────────────────────────────────────────────────────────
# B14 · Palancas 6 y 7: truncado y ventana temporal
# ─────────────────────────────────────────────────────────────────────────────
def test_b14_long_titles_are_truncated_and_stale_ones_dropped() -> None:
    long_title = "x" * 500
    prepared = prepare_headlines(
        (
            _headline(long_title, minutes=-1),
            _headline("noticia vieja", minutes=-60 * 48),
            _headline("noticia del futuro", minutes=5),
        ),
        now=NOW,
        window_hours=24,
        max_chars=100,
    )

    assert len(prepared) == 1
    assert prepared[0].title == long_title[:100]
    assert len(prepared[0].title) == 100


# ─────────────────────────────────────────────────────────────────────────────
# #129 · El conteo del lote: lo que antes se calculaba y se tiraba
# ─────────────────────────────────────────────────────────────────────────────
def test_b2_prepare_headlines_keeps_its_signature_and_delegates() -> None:
    """`prepare_headlines` no cambia para quien ya la usa: mismo retorno, mismo lote."""
    batch = (_headline("La Fed sube los tipos"), _headline("Nvidia presenta resultados"))
    assert prepare_headlines(batch, now=NOW) == prepare_batch(batch, now=NOW)[0]


def test_b3_the_counts_are_declared_over_a_known_batch() -> None:
    """Un lote del que se sabe la respuesta exacta: 2 duplicados, 1 fuera, 1 truncado."""
    batch = (
        _headline("La Fed sube los tipos"),
        _headline("¡La FED sube los tipos!"),
        _headline("Nvidia presenta resultados"),
        _headline("titular del mes pasado", minutes=-60 * 24 * 30),
        _headline("x" * 500),
    )
    prepared, counts = prepare_batch(batch, now=NOW, window_hours=24, max_chars=100)

    assert counts.read == 5
    assert counts.duplicates == 1, "los dos titulares casi identicos colapsan"
    assert counts.out_of_window == 1, "el del mes pasado queda fuera de la ventana"
    assert counts.sent == 3
    assert counts.truncated == 1, "el titular de 500 caracteres llega truncado"
    assert len(prepared) == counts.sent


def test_b4_the_identity_closes_over_the_real_batch() -> None:
    """⭐ El conteo **no** puede tener huecos: todo lo leido acaba en una de las tres categorias."""
    batch = (
        _headline("a"),
        _headline("a"),
        _headline("b"),
        _headline("c", minutes=600),
        _headline("d"),
    )
    _, counts = prepare_batch(batch, now=NOW, window_hours=1)
    assert counts.read == counts.duplicates + counts.out_of_window + counts.sent


def test_b4_an_incoherent_count_cannot_be_constructed() -> None:
    """La identidad se exige **al construir**: un conteo incompleto no llega a existir."""
    with pytest.raises(ValueError, match="no cierra"):
        BatchCounts(read=5, duplicates=1, out_of_window=0, sent=3, truncated=0)


@pytest.mark.parametrize("field", ["read", "duplicates", "out_of_window", "sent", "truncated"])
def test_b4_a_negative_or_boolean_count_is_rejected(field: str) -> None:
    """Un `bool` es un `int` disfrazado y un negativo falsearia el informe: los dos se cortan."""
    base = {"read": 2, "duplicates": 0, "out_of_window": 0, "sent": 2, "truncated": 0}
    with pytest.raises(ValueError, match=f"BatchCounts.{field}"):
        BatchCounts(**{**base, field: True if field == "truncated" else -1})


def test_b10_the_duplicate_count_comes_from_the_dedup_of_30() -> None:
    """El conteo **reutiliza** lo que #30 ya devolvia; no se recuenta por hash aqui."""
    batch = (
        _headline("La Fed sube los tipos"),
        _headline("La Fed sube los tipos"),
        _headline("Nvidia presenta resultados"),
    )
    _, duplicates = news_deduplicate(batch)
    _, counts = prepare_batch(batch, now=NOW)
    assert counts.duplicates == len(duplicates) == 1
