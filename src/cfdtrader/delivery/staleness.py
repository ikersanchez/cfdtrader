"""Guardia de obsolescencia del camino diario (`tech_stack.md` §8.4, tarea #40).

Una recomendacion calculada con datos de hace tres sesiones es **peor que ninguna**: se presenta
con el mismo formato y la misma confianza que una buena. Sin *scheduler* (``tech_stack.md`` §8.1)
nadie ejecuta el pipeline por ti, asi que el riesgo **aumenta**: basta con no haberlo lanzado. La
guardia de §8.4 es obligatoria y este modulo es su implementacion en el camino diario: la
comprobacion de "¿es dia de sesion valido?" y de "¿corresponden los datos a la sesion evaluada?".

Las tres preguntas que responde, y quien las responde:

- **¿Es dia de sesion?** :func:`market_closure` reutiliza ``MarketCalendar`` (el modulo de #4) y
  devuelve el motivo declarado (``festivo: ...``, ``fin de semana``) o ``None`` si el mercado
  abre. **No re-deriva festivos, medias sesiones ni horarios**: la clausura la decide el
  calendario y, desde #113, el camino diario la convierte en un ``NOTHING`` justificado (la
  regla 19 del gate), no en un "no se ejecuta".
- **¿Corresponden los datos a la sesion?** :func:`session_guard` compara la ultima sesion del
  almacen con la anterior a la evaluada y publica un :class:`GuardVerdict`: falta el cierre de la
  sesion anterior (§8.4 fila 2), el almacen va por delante de la declaracion, o se puede seguir.
  Los dos fallos son **"no se"** (``no_recommendation_stale_data``), nunca ``NOTHING``.
- **¿Hay que reincorporarse?** :func:`observation_sessions_remaining` cuenta las 5 sesiones de
  observacion de la regla 15 **derivadas del diario**: el historial de ejecuciones es el unico
  registro de la ausencia, y no se persiste ningun contador nuevo.

Los cuatro estados de ``plan.md`` §19.2 y donde entra la clausura (#113)
------------------------------------------------------------------------

El registro distingue ``recommendation`` (con ``LONG``/``SHORT``/``NOTHING``),
``no_recommendation_stale_data``, ``no_recommendation_data_quality`` y ``error``. Un dia de
mercado cerrado **no es un "no se"**: no hay sesion que evaluar y el calendario *si* sabe que no
abre. #40 implemento la rama **primera** de §8.4 ("no se ejecuta": un aviso sin informe ni fila);
**#113 implementa su alternativa** (la del parentesis: "o se ejecuta y devuelve ``NOTHING``
justificado"). Por eso este modulo ya **no** publica un aviso de clausura (``closure_notice`` era
de #40): :func:`market_closure` sigue dando el motivo, y el gate lo convierte en un ``NOTHING``
con la regla 19 que si se registra en el diario.

Sin reloj, sin red y sin escritura
----------------------------------

El instante entra declarado (``as_of``, ISO-8601 con zona) y el modulo no lo consulta: no hay
``datetime.now``. No importa ``yfinance``/``requests``/``urllib``, no lanza procesos y **no
escribe nada**. Del disco solo lee el diario de decisiones, y lo hace por el lector de #39
(``read_decisions``, **importado**, nunca reabriendo ``<root>/decisions/*.json`` a mano).

Reuso por import: ``MarketCalendar``/``EASTERN`` de ``data.calendar`` y
``Journal``/``read_decisions`` de ``journal.decision_log`` (la capa de #39). Ninguno se
reimplementa. El **camino diario** (``delivery.run_daily``) es quien mapea el veredicto al
``GateStatus`` de ``decision.gate``: la guardia no necesita el gate para decidir la frescura ni la
clausura, y por eso no lo importa.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Final

from cfdtrader.data.calendar import EASTERN, MarketCalendar
from cfdtrader.journal.decision_log import Journal, read_decisions

__all__ = [
    "ABSENCE_DAYS",
    "OBSERVATION_SESSIONS",
    "GuardVerdict",
    "SessionGuard",
    "execution_dates",
    "market_closure",
    "observation_sessions_remaining",
    "session_guard",
]

#: Dias naturales sin ejecucion a partir de los cuales la vuelta es una **reincorporacion**:
#: "mas de una semana" (``plan.md`` §12 regla 15 y ``tech_stack.md`` §8.4). El "N dias naturales"
#: de la tercera fila de la tabla de §8.4 se declara **con este mismo valor**, para que las dos
#: filas no puedan contradecirse.
ABSENCE_DAYS: Final[int] = 7

#: Sesiones de observacion tras la reincorporacion (``plan.md`` §12 regla 15).
OBSERVATION_SESSIONS: Final[int] = 5

#: Motivo de ultimo recurso si un calendario no nombrase la clausura (nunca se inventa el dato).
CLOSED_REASON: Final[str] = "mercado cerrado"


class GuardVerdict(StrEnum):
    """El veredicto de la guardia sobre la sesion declarada.

    ``PROCEED`` no significa "hay pista": significa que la guardia **no** la prohibe y el gate
    decide despues (con sus reglas 1, 3, 4, 5, 15, 17, 18 y 19). ``MISSING_PREVIOUS_CLOSE`` y
    ``SNAPSHOT_AHEAD`` son paradas "no se" (``no_recommendation_stale_data``); ``MARKET_CLOSED``
    no para el camino: el gate la convierte en ``NOTHING`` justificado (regla 19, #113).
    """

    PROCEED = "proceed"
    MARKET_CLOSED = "market_closed"
    MISSING_PREVIOUS_CLOSE = "missing_previous_close"
    SNAPSHOT_AHEAD = "snapshot_ahead"


# ─────────────────────────────────────────────────────────────────────────────
# Clausura: la comprobacion de "¿es dia de sesion valido?" (§8.4, §8.1)
# ─────────────────────────────────────────────────────────────────────────────
def market_closure(*, as_of: datetime, calendar: MarketCalendar) -> str | None:
    """El motivo por el que esa fecha **no** tiene sesion, o ``None`` si el mercado abre.

    La fecha es la de ``as_of`` en la hora de referencia interna (``America/New_York``): anclar
    a hora local desplazaria la sesion dos veces al ano (``plan.md`` §8.3). El motivo lo da el
    propio calendario (``festivo: <nombre>`` / ``fin de semana``): este modulo no tiene ninguna
    tabla de festivos ni de medias sesiones.
    """
    info = calendar.session(as_of.astimezone(EASTERN).date())
    if info.is_session:
        return None
    return info.reason or CLOSED_REASON


# ─────────────────────────────────────────────────────────────────────────────
# Reincorporacion: el contador de la regla 15, derivado del diario
# ─────────────────────────────────────────────────────────────────────────────
def execution_dates(journal_root: Journal | Path | str) -> tuple[date, ...]:
    """Las sesiones ya registradas en ``journal.decisions``, ordenadas y sin repetir.

    Se leen con ``read_decisions`` (importada de #39): este modulo no reabre los ficheros del
    diario ni necesita su esquema completo, solo la identidad de cada fila (``trade_date``). Un
    diario que todavia no existe devuelve una tupla vacia (sin historial, sin ausencia).
    """
    records = read_decisions(journal_root)
    return tuple(sorted({date.fromisoformat(str(record["trade_date"])) for record in records}))


def observation_sessions_remaining(
    *,
    executions: Sequence[date],
    calendar: MarketCalendar,
    session: date,
    absence_days: int = ABSENCE_DAYS,
    observation_sessions: int = OBSERVATION_SESSIONS,
) -> int:
    """Las sesiones de observacion que quedan (regla 15), derivadas del historial de ejecuciones.

    La secuencia es el historial **mas la sesion evaluada**: la vuelta de una ausencia ocurre
    precisamente en la fecha que se esta evaluando, que todavia **no** tiene fila en el diario
    (si no, el contador moriria en la primera vuelta). La ventana la arranca el **ultimo** hueco
    de mas de ``absence_days`` dias naturales, y se consume en sesiones de calendario: un dia de
    mercado cerrado no la gasta.
    """
    sequence = sorted({day for day in executions if day < session} | {session})
    start: date | None = None
    for previous, current in pairwise(sequence):
        if (current - previous).days > absence_days:
            start = current
    if start is None:
        return 0
    elapsed = len(calendar.sessions(start, session))
    return max(0, observation_sessions - elapsed + 1)


# ─────────────────────────────────────────────────────────────────────────────
# El veredicto: la sesion evaluada, su trazabilidad y lo que se puede hacer con ella
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SessionGuard:
    """Lo que la guardia sabe de la sesion evaluada, con su motivo y su trazabilidad.

    ``verdict = PROCEED`` **no** autoriza a operar: solo declara que el guardia no prohibe
    continuar, y el gate aplica despues sus reglas (1, 3, 4, 5, 15, 17 y 18). ``message`` es el
    motivo que el informe presenta **verbatim** en su linea ``motivo:``.
    """

    verdict: GuardVerdict
    session: date
    as_of: datetime
    previous_session: date
    snapshot_session: date | None
    is_session: bool
    is_half_session: bool
    closure_reason: str | None
    message: str
    last_execution: date | None
    absence_days: int | None
    observation_sessions_remaining: int
    reincorporation_notice: str | None

    @property
    def blocks(self) -> bool:
        """``True`` si la guardia para el camino diario: cualquier veredicto que no sea seguir."""
        return self.verdict is not GuardVerdict.PROCEED


def _reincorporation_notice(remaining: int) -> str | None:
    """El aviso de reincorporacion de §8.4, con las sesiones que quedan por revalidar."""
    if remaining <= 0:
        return None
    return (
        f"reincorporacion (tech_stack.md §8.4): modo observacion, {remaining} de "
        f"{OBSERVATION_SESSIONS} sesiones por revalidar antes de volver a operar"
    )


def session_guard(
    *,
    as_of: datetime,
    calendar: MarketCalendar,
    snapshot_session: date | None,
    executions: Sequence[date] = (),
    absence_days: int = ABSENCE_DAYS,
    observation_sessions: int = OBSERVATION_SESSIONS,
) -> SessionGuard:
    """Clasifica la sesion declarada: se puede seguir, falta el cierre anterior, o no hay sesion.

    ``snapshot_session`` es la **ultima** sesion del almacen (``FeatureMatrix.last_session``);
    ``executions`` el historial del diario (``execution_dates``). Ninguna de las dos se lee aqui:
    entran declaradas, que es lo que hace que el modulo sea una funcion pura y determinista.

    Precedencia: la clausura (no hay nada que evaluar) manda sobre la frescura, y la frescura
    sobre la reincorporacion. Un dia de mercado cerrado **no** gasta sesiones de observacion.
    """
    day = as_of.astimezone(EASTERN).date()
    info = calendar.session(day)
    previous = calendar.previous_session(day)
    closure = market_closure(as_of=as_of, calendar=calendar)
    history = sorted({item for item in executions if item < day})
    last_execution = history[-1] if history else None
    absence = None if last_execution is None else (day - last_execution).days

    if closure is not None:
        verdict = GuardVerdict.MARKET_CLOSED
        message = (
            f"el mercado americano no abre el {day.isoformat()} ({closure}): la guardia de "
            "obsolescencia (tech_stack.md §8.4) no emite ninguna recomendacion"
        )
    elif snapshot_session is None:
        verdict = GuardVerdict.MISSING_PREVIOUS_CLOSE
        message = (
            f"el cierre de la sesion anterior ({previous.isoformat()}) no esta en el almacen: sin "
            "ese cierre no se puede calcular el gap ni emitir pista (tech_stack.md §8.4)"
        )
    elif snapshot_session < previous:
        verdict = GuardVerdict.MISSING_PREVIOUS_CLOSE
        message = (
            f"datos obsoletos (tech_stack.md §8.4): la ultima sesion del almacen es "
            f"{snapshot_session.isoformat()} y la sesion anterior a {day.isoformat()} es "
            f"{previous.isoformat()}; no se emite pista"
        )
    elif snapshot_session > previous:
        verdict = GuardVerdict.SNAPSHOT_AHEAD
        message = (
            f"el almacen va por delante de la declaracion: su ultima sesion es "
            f"{snapshot_session.isoformat()} y la sesion anterior a {day.isoformat()} es "
            f"{previous.isoformat()}; `--as-of` no se corresponde con el almacen "
            "(tech_stack.md §8.4)"
        )
    else:
        verdict = GuardVerdict.PROCEED
        message = (
            f"la ultima sesion del almacen ({snapshot_session.isoformat()}) es la anterior a "
            f"{day.isoformat()}: los datos corresponden a la sesion evaluada"
        )

    closed = closure is not None
    remaining = (
        0
        if closed
        else observation_sessions_remaining(
            executions=executions,
            calendar=calendar,
            session=day,
            absence_days=absence_days,
            observation_sessions=observation_sessions,
        )
    )
    return SessionGuard(
        verdict=verdict,
        session=day,
        as_of=as_of,
        previous_session=previous,
        snapshot_session=snapshot_session,
        is_session=info.is_session,
        is_half_session=info.is_half_day,
        closure_reason=closure,
        message=message,
        last_execution=last_execution,
        absence_days=absence,
        observation_sessions_remaining=remaining,
        reincorporation_notice=_reincorporation_notice(remaining),
    )
