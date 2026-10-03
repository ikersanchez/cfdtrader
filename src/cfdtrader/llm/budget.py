"""Presupuesto, topes duros y registro de llamadas del LLM (§6.3, §12.5, §12.6) — tarea #33.

La pregunta que responde este módulo: **¿cómo se acota un gasto sin que el gasto pueda parar el
sistema?**

La regla dura de §6.3 es que superar cualquier tope **desactiva el overlay y el pipeline sigue**.
Aqui toma una forma concreta y comprobable: la capa metrada **nunca lanza** por presupuesto;
devuelve «no hay respuesta» y el estado del overlay, y el resto del sistema sigue su camino.

El vocabulario de los estados es el de §12.5 y **no se inventa**: ``disabled_budget`` (topes),
``disabled_error`` (fallos consecutivos) y ``disabled_timeout`` (tiempo agotado). ``applied`` y
``veto`` los usara el overlay (#35).

Cada llamada queda registrada en ``ops.llm_calls`` con las **12 columnas que ya declaro #39**
(``journal.TABLE_COLUMNS``): este modulo las **consume**, no las reimplementa.

Las siete palancas de §6.3, y donde vive cada una: (1) *batching* lo garantiza el agente (#32, A10);
(2) la cache, en :mod:`cfdtrader.llm.cache`; (3) la deduplicacion previa **reutiliza**
``data.news.deduplicate``; (4) el modelo economico sale de la configuracion; (6) el truncado y
(7) la ventana temporal, en :func:`prepare_headlines`. La (5), la cache de contexto del proveedor,
se partio a #116 porque depende del proveedor y hoy no se puede medir.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Final

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from cfdtrader.data.news import DEFAULT_WINDOW_HOURS, deduplicate
from cfdtrader.data.settings import REPO_ROOT
from cfdtrader.data.sources.news import Headline
from cfdtrader.journal.decision_log import Journal, WriteOutcome
from cfdtrader.llm.base import LLMClient, LLMError, LLMRequest, LLMResponse
from cfdtrader.llm.cache import ResponseCache, request_key

__all__ = [
    "DEFAULT_MAX_CONSECUTIVE_FAILURES",
    "DEFAULT_TRUNCATE_CHARS",
    "DEFAULT_WINDOW_HOURS",
    "MONTHLY_WARNING",
    "PRICES_EUR_PER_MTOKENS",
    "BudgetCaps",
    "BudgetDecision",
    "BudgetGuard",
    "Caps",
    "LLMCall",
    "MeteredLLMClient",
    "OverlayClient",
    "OverlayState",
    "estimate_cost",
    "prepare_headlines",
    "record_call",
]

#: Fallos consecutivos antes de desactivar el overlay ese dia (§6.3.4).
DEFAULT_MAX_CONSECUTIVE_FAILURES: Final[int] = 3

#: Longitud maxima del titular que se envia al modelo (palanca 6 de §6.3).
DEFAULT_TRUNCATE_CHARS: Final[int] = 280

#: Ventana de noticias que se envia al modelo (palanca 7 de §6.3): se declara con el dato, en
#: `data.news`, y se reexporta aqui para no tener dos numeros que puedan divergir.

#: Precios por millon de tokens, ``(entrada, salida)`` en euros. **Vacio a proposito**: §6.3 avisa
#: de que las tarifas cambian y hay que verificarlas, asi que no se inventa ninguna. Sin precio
#: declarado, el coste de una llamada es ``None`` — que es la verdad — y nunca ``0,0``.
PRICES_EUR_PER_MTOKENS: Final[dict[str, tuple[float, float]]] = {}


#: Los cinco estados del overlay de §12.5. Vive **también** en :mod:`cfdtrader.decision.overlay`,
#: y no es un descuido: el reparto de capas está **probado en los dos sentidos** — #33 comprueba
#: que esta capa de coste no importa `cfdtrader.decision`, y #35 comprueba que la decisión no
#: importa `cfdtrader.llm`. Compartir el enum obligaría a romper una de las dos. El origen de
#: verdad es ``journal.decision_log.LLM_OVERLAYS`` (#39) y **cada** módulo tiene su prueba que
#: compara su vocabulario con él: una sexta variante hace fallar las dos.
class OverlayState(StrEnum):
    """Los cinco estados de §12.5, declarados por #39. Aqui no se inventa ninguno."""

    APPLIED = "applied"
    VETO = "veto"
    DISABLED_BUDGET = "disabled_budget"
    DISABLED_ERROR = "disabled_error"
    DISABLED_TIMEOUT = "disabled_timeout"


@dataclass(frozen=True)
class Caps:
    """Los seis topes duros de §6.3.4. ``None`` significa «sin tope declarado»."""

    max_tokens_per_run: int | None = None
    max_calls_per_run: int | None = None
    daily_budget_eur: float | None = None
    monthly_budget_eur: float | None = None
    max_seconds: float | None = None
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES


class BudgetCaps(BaseSettings):
    """Los topes, leidos del entorno (``LLM_*``) y del ``.env`` (`tech_stack.md` §4.2).

    Viven aqui y no en :class:`cfdtrader.llm.base.LLMClientConfig` porque son presupuesto, no
    identidad del cliente: el cliente puede estar bien configurado y el presupuesto agotado.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_tokens_per_run: int | None = Field(default=None, ge=1)
    max_calls_per_run: int | None = Field(default=None, ge=0)
    daily_budget_eur: float | None = Field(default=None, ge=0.0)
    monthly_budget_eur: float | None = Field(default=None, ge=0.0)
    max_seconds: float | None = Field(default=None, gt=0.0)
    max_consecutive_failures: int = Field(default=DEFAULT_MAX_CONSECUTIVE_FAILURES, ge=1)

    @field_validator(
        "max_tokens_per_run",
        "max_calls_per_run",
        "daily_budget_eur",
        "monthly_budget_eur",
        "max_seconds",
        mode="before",
    )
    @classmethod
    def _blank_means_no_cap(cls, value: object) -> object:
        """Una variable declarada y **vacia** significa «sin tope», no un error de parseo.

        ``.env.example`` trae ``LLM_DAILY_BUDGET_EUR=`` en blanco, y ``pydantic-settings``
        intentaria parsear ``''`` como numero y fallaria: el overlay quedaria desactivado **en
        silencio** por un fichero de configuracion que es correcto. Lo cazo el cableado de #35,
        porque la prueba de #33 construia estos topes con ``_env_file=None``.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def caps(self) -> Caps:
        """Los topes como valor inmutable, que es lo que consume el guardian."""
        return Caps(
            max_tokens_per_run=self.max_tokens_per_run,
            max_calls_per_run=self.max_calls_per_run,
            daily_budget_eur=self.daily_budget_eur,
            monthly_budget_eur=self.monthly_budget_eur,
            max_seconds=self.max_seconds,
            max_consecutive_failures=self.max_consecutive_failures,
        )


def estimate_cost(
    *,
    model: str,
    tokens_in: int | None,
    tokens_out: int | None,
    prices: dict[str, tuple[float, float]] | None = None,
) -> float | None:
    """El coste estimado en euros, o ``None`` si no hay tarifa declarada para ese modelo.

    ``None`` es una respuesta legitima: no se conoce el precio, y escribir ``0,0`` seria afirmar
    que salio gratis. La tabla de precios la declara el propietario y hay que revisarla (§6.3).
    """
    table = PRICES_EUR_PER_MTOKENS if prices is None else prices
    tariff = table.get(model)
    if tariff is None:
        return None
    if tokens_in is None and tokens_out is None:
        return None
    input_price, output_price = tariff
    spend = (tokens_in or 0) * input_price + (tokens_out or 0) * output_price
    return spend / 1_000_000


#: Aviso especifico de §6.3.4 para el tope **mensual** (los demas cortes no lo llevan).
MONTHLY_WARNING: Final[str] = (
    "aviso: el gasto mensual acumulado supera el tope declarado; revisar el informe de coste"
)


@dataclass(frozen=True)
class BudgetDecision:
    """Si se puede llamar al proveedor, y con que motivo si no."""

    state: OverlayState
    warnings: tuple[str, ...] = ()


class BudgetGuard:
    """Cuenta lo gastado y decide si se puede seguir. **Nunca lanza**: decide.

    Un acierto de cache **no es una llamada**: no suma al contador de llamadas, no gasta tokens ni
    euros, y **no reinicia** el contador de fallos consecutivos. Si lo reiniciase, un *hit* podria
    mantener vivo un overlay que ya habia fallado tres veces.
    """

    def __init__(
        self, *, caps: Caps | None = None, clock: Callable[[], float] | None = None
    ) -> None:
        self._caps = Caps() if caps is None else caps
        self._clock = time.monotonic if clock is None else clock
        self._started = self._clock()
        self.tokens_used = 0
        self.calls_made = 0
        self.daily_spend_eur = 0.0
        self.monthly_spend_eur = 0.0
        self.consecutive_failures = 0

    @property
    def caps(self) -> Caps:
        """Los topes declarados."""
        return self._caps

    @property
    def elapsed_seconds(self) -> float:
        """Lo que lleva vivo el guardian, segun el reloj **inyectado**."""
        return self._clock() - self._started

    def check(self) -> BudgetDecision:
        """El estado del overlay ahora mismo. Precedencia: error, tiempo, topes."""
        if self.consecutive_failures >= self._caps.max_consecutive_failures:
            return BudgetDecision(
                OverlayState.DISABLED_ERROR,
                (f"{self.consecutive_failures} fallos consecutivos de la API",),
            )
        if self._caps.max_seconds is not None and self.elapsed_seconds >= self._caps.max_seconds:
            return BudgetDecision(
                OverlayState.DISABLED_TIMEOUT,
                (f"la capa LLM lleva {self.elapsed_seconds:.3f} s de {self._caps.max_seconds} s",),
            )
        reasons = self._exceeded()
        if reasons:
            warnings = reasons
            if self._monthly_exceeded():
                warnings = (*reasons, MONTHLY_WARNING)
            return BudgetDecision(OverlayState.DISABLED_BUDGET, warnings)
        return BudgetDecision(OverlayState.APPLIED)

    def record(
        self,
        *,
        cache_hit: bool,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cost_eur: float | None = None,
        ok: bool = True,
    ) -> None:
        """Registra el resultado de una peticion. Un acierto de cache no toca ningun contador."""
        if cache_hit:
            return
        self.calls_made += 1
        self.tokens_used += (tokens_in or 0) + (tokens_out or 0)
        if cost_eur is not None:
            self.daily_spend_eur += cost_eur
            self.monthly_spend_eur += cost_eur
        self.consecutive_failures = 0 if ok else self.consecutive_failures + 1

    def _exceeded(self) -> tuple[str, ...]:
        """Los topes superados, uno por linea, en el orden de §6.3.4."""
        caps = self._caps
        reasons: list[str] = []
        if caps.max_tokens_per_run is not None and self.tokens_used >= caps.max_tokens_per_run:
            reasons.append(
                f"tokens de entrada/salida: {self.tokens_used} >= {caps.max_tokens_per_run}"
            )
        if caps.max_calls_per_run is not None and self.calls_made >= caps.max_calls_per_run:
            reasons.append(f"llamadas: {self.calls_made} >= {caps.max_calls_per_run}")
        if caps.daily_budget_eur is not None and self.daily_spend_eur >= caps.daily_budget_eur:
            reasons.append(
                f"gasto diario: {self.daily_spend_eur:.6f} EUR >= {caps.daily_budget_eur}"
            )
        if self._monthly_exceeded():
            reasons.append(
                f"gasto mensual: {self.monthly_spend_eur:.6f} EUR >= {caps.monthly_budget_eur}"
            )
        return tuple(reasons)

    def _monthly_exceeded(self) -> bool:
        cap = self._caps.monthly_budget_eur
        return cap is not None and self.monthly_spend_eur >= cap


# ─────────────────────────────────────────────────────────────────────────────
# Registro en `ops.llm_calls` (esquema cerrado de #39)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LLMCall:
    """Una fila de ``ops.llm_calls``: las **12 columnas** que declaro #39, ni una mas."""

    call_id: str
    as_of: datetime
    provider: str
    model: str
    system_fingerprint: str | None = None
    purpose: str = "extract"
    tokens_in: int | None = None
    tokens_out: int | None = None
    cache_hit: bool = False
    cost_estimate: float | None = None
    latency_ms: int | None = None
    ok: bool = True

    def payload(self) -> dict[str, object]:
        """El payload con las columnas exactas de ``TABLE_COLUMNS["llm_calls"]``."""
        return {
            "call_id": self.call_id,
            "as_of": self.as_of,
            "provider": self.provider,
            "model": self.model,
            "system_fingerprint": self.system_fingerprint,
            "purpose": self.purpose,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cache_hit": self.cache_hit,
            "cost_estimate": self.cost_estimate,
            "latency_ms": self.latency_ms,
            "ok": self.ok,
        }


# ─────────────────────────────────────────────────────────────────────────────
# El puente hacia el agente (#35)
# ─────────────────────────────────────────────────────────────────────────────
class OverlayClient:
    """Adapta la capa metrada al contrato ``LLMClient`` que consume el agente (#32).

    Existe porque los dos contratos son **distintos a propósito**: ``NewsAgent`` pide ``complete``
    (siempre hay respuesta o hay error), y la capa metrada expone :meth:`MeteredLLMClient.call`,
    que ademas puede devolver ``None`` cuando el overlay esta cortado (presupuesto, fallos, tiempo).
    Ese ``None`` es una **decision**, no un fallo, y aqui se traduce a :class:`LLMError` para que el
    agente no tenga que saber de presupuestos.

    El camino diario **consulta el estado antes** de invocar al agente, asi que en uso correcto
    nunca se llega aqui con el overlay cortado; si se llega, es un fallo del llamante y prefiero un
    error tipado a un dato inventado.
    """

    def __init__(self, metered: MeteredLLMClient) -> None:
        self._metered = metered

    @property
    def state(self) -> OverlayState:
        """El estado del overlay, tal cual lo dejo la ultima comprobacion."""
        return self._metered.state

    @property
    def last_cache_hit(self) -> bool:
        """Si la ultima peticion se resolvio con la cache."""
        return self._metered.last_cache_hit

    def complete(self, request: LLMRequest) -> LLMResponse:
        """La respuesta del agente, o ``LLMError`` si el overlay esta cortado."""
        response = self._metered.call(request)
        if response is None:
            raise LLMError(
                f"el overlay no puede responder (estado {self._metered.state.value}); el llamante "
                "debe haber consultado el estado antes de invocar al agente"
            )
        return response


def record_call(journal: Journal | Path | str, call: LLMCall) -> WriteOutcome:
    """Escribe la fila en ``ops.llm_calls``: esquema cerrado e inmutable, el de #39."""
    handle = journal if isinstance(journal, Journal) else Journal(Path(str(journal)))
    return handle.write("llm_calls", call.payload())


# ─────────────────────────────────────────────────────────────────────────────
# Palancas 3, 6 y 7 de §6.3
# ─────────────────────────────────────────────────────────────────────────────
def prepare_headlines(
    headlines: Sequence[Headline],
    *,
    now: datetime,
    known_hashes: frozenset[str] = frozenset(),
    known_titles: Sequence[str] = (),
    window_hours: int = DEFAULT_WINDOW_HOURS,
    max_chars: int = DEFAULT_TRUNCATE_CHARS,
) -> tuple[Headline, ...]:
    """Deja el lote listo para gastar lo minimo: deduplicar, acotar la ventana y truncar.

    - **Palanca 3**: la deduplicacion **reutiliza** ``data.news.deduplicate`` (la de #30, prueba
      incluida). No se reescribe la similitud difusa aqui.
    - **Palanca 7**: fuera el titular anterior a ``now - window_hours`` y tambien el posterior a
      ``now`` (point-in-time, la misma regla que #30).
    - **Palanca 6**: el titulo que se envia se trunca a ``max_chars``. Se trunca el **texto**, no la
      fila del almacen: ``raw.news_headlines`` no se toca.

    ⚠️ Truncar cambia el ``headline_hash`` del titular resultante. Es deliberado y coherente: aguas
    abajo se usa el hash del titular que **realmente se envio al modelo**, que es el unico que el
    modelo puede citar.
    """
    fresh, _ = deduplicate(headlines, known_hashes=known_hashes, known_titles=known_titles)
    floor = now - timedelta(hours=window_hours)
    prepared: list[Headline] = []
    for headline in fresh:
        if not floor <= headline.published_at <= now:
            continue
        if len(headline.title) <= max_chars:
            prepared.append(headline)
        else:
            prepared.append(headline.model_copy(update={"title": headline.title[:max_chars]}))
    return tuple(prepared)


# ─────────────────────────────────────────────────────────────────────────────
# La capa metrada
# ─────────────────────────────────────────────────────────────────────────────
class MeteredLLMClient:
    """Capa metrada sobre un proveedor: cache, topes y registro.

    **Nunca lanza por presupuesto.** Cuando un tope, el tiempo o los fallos consecutivos cortan,
    :meth:`call` devuelve ``None`` y deja el estado en :attr:`state`. Es la regla dura de §6.3:
    superar un tope desactiva el overlay y el sistema sigue; una excepcion en el camino critico
    haria justo lo contrario.

    Un ``LLMError`` del proveedor **si** se propaga (el llamante decide), pero queda contado en
    ``ops.llm_calls`` y en el contador de fallos consecutivos, que es lo que desactiva el overlay.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        cache: ResponseCache,
        guard: BudgetGuard,
        as_of: datetime,
        provider: str,
        purpose: str = "extract",
        journal: Journal | Path | str | None = None,
        prices: dict[str, tuple[float, float]] | None = None,
        warning_sink: Callable[[str], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._client = client
        self._cache = cache
        self._guard = guard
        self._as_of = as_of
        self._provider = provider
        self._purpose = purpose
        self._journal = journal
        self._prices = prices
        self._warning_sink = warning_sink
        self._clock = time.monotonic if clock is None else clock
        self._sequence = 0
        self._decision = BudgetDecision(OverlayState.APPLIED)
        self._last_cache_hit = False
        self.rows_written = 0

    @property
    def state(self) -> OverlayState:
        """El estado del overlay segun la ultima comprobacion."""
        return self._decision.state

    @property
    def warnings(self) -> tuple[str, ...]:
        """Los motivos del corte (y el aviso mensual, si toca)."""
        return self._decision.warnings

    @property
    def last_cache_hit(self) -> bool:
        """Si la ultima peticion se resolvio con la cache."""
        return self._last_cache_hit

    def call(self, request: LLMRequest) -> LLMResponse | None:
        """La respuesta, o ``None`` si la capa esta desactivada. Nunca lanza por presupuesto."""
        decision = self._guard.check()
        self._decision = decision
        if decision.state is not OverlayState.APPLIED:
            self._warn(decision.warnings)
            return None

        self._sequence += 1
        call_id = self._call_id(self._sequence)
        started = self._clock()
        key = request_key(request)

        hit = self._cache.get(key)
        if hit is not None:
            self._last_cache_hit = True
            self._guard.record(cache_hit=True)
            self._write(
                call_id,
                model=hit.model,
                fingerprint=hit.system_fingerprint,
                tokens_in=hit.prompt_tokens,
                tokens_out=hit.completion_tokens,
                cache_hit=True,
                cost=None,
                ok=True,
                started=started,
            )
            return hit

        self._last_cache_hit = False
        try:
            response = self._client.complete(request)
        except LLMError:
            self._guard.record(cache_hit=False, ok=False)
            self._write(
                call_id,
                model=request.model,
                fingerprint=None,
                tokens_in=None,
                tokens_out=None,
                cache_hit=False,
                cost=None,
                ok=False,
                started=started,
            )
            raise
        cost = estimate_cost(
            model=response.model,
            tokens_in=response.prompt_tokens,
            tokens_out=response.completion_tokens,
            prices=self._prices,
        )
        self._guard.record(
            cache_hit=False,
            tokens_in=response.prompt_tokens,
            tokens_out=response.completion_tokens,
            cost_eur=cost,
            ok=True,
        )
        self._cache.put(key, response)
        self._write(
            call_id,
            model=response.model,
            fingerprint=response.system_fingerprint,
            tokens_in=response.prompt_tokens,
            tokens_out=response.completion_tokens,
            cache_hit=False,
            cost=cost,
            ok=True,
            started=started,
        )
        return response

    def _call_id(self, sequence: int) -> str:
        """La identidad de la fila: el instante declarado mas la secuencia de la ejecucion.

        Dos llamadas del **mismo** instante no pueden compartir identidad (perderiamos una fila),
        asi que la secuencia forma parte del identificador. El formato es un nombre de fichero
        limpio, sin separadores que #39 rechazaria.
        """
        stamp = self._as_of.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}-{sequence:04d}"

    def _warn(self, warnings: tuple[str, ...]) -> None:
        if self._warning_sink is None:
            return
        for warning in warnings:
            self._warning_sink(warning)

    def _write(
        self,
        call_id: str,
        *,
        model: str,
        fingerprint: str | None,
        tokens_in: int | None,
        tokens_out: int | None,
        cache_hit: bool,
        cost: float | None,
        ok: bool,
        started: float,
    ) -> None:
        """Deja la fila en ``ops.llm_calls``. Sin diario configurado, solo cuenta la fila."""
        self.rows_written += 1
        if self._journal is None:
            return
        record_call(
            self._journal,
            LLMCall(
                call_id=call_id,
                as_of=self._as_of,
                provider=self._provider,
                model=model,
                system_fingerprint=fingerprint,
                purpose=self._purpose,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cache_hit=cache_hit,
                cost_estimate=cost,
                latency_ms=round((self._clock() - started) * 1000),
                ok=ok,
            ),
        )
