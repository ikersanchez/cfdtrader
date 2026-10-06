"""Calendario de eventos de la sesión — tarea #34 (núcleo derivable).

La pregunta que responde este módulo: **¿qué eventos de calendario caen en la
sesión evaluada y cuáles bloquean la operación?**

Núcleo derivable (este corte)
-----------------------------
Todo lo que se deduce de reglas y del calendario del proyecto, **sin red, sin
dependencias nuevas y sin inventar ninguna fecha**:

- **Festivo / fin de semana** (mercado cerrado): se reutiliza
  ``MarketCalendar.session(day).reason`` —la **misma** fuente que la regla 19 del
  gate—. Bloquea.
- **Media sesión** (cierre a las 13:00 ET): ``MarketCalendar.is_half_day``, la
  misma definición que la regla 18 del gate. Bloquea.
- **OPEX**: el vencimiento mensual de opciones, el tercer viernes del mes
  **rodado a la sesión inmediatamente anterior** cuando el tercer viernes no es
  sesión. La regla vive **una sola vez**, en
  ``MarketCalendar.opex_dates``/``is_opex``.
- **Triple *witching***: el OPEX de un mes trimestral (mar/jun/sep/dic).
- **Roll trimestral del futuro ES**: ``MarketCalendar.is_es_roll``, que es el
  subconjunto trimestral de la OPEX **ya rodada** a sesión.

Frontera declarada (por qué esto **no** es la tarea #34 completa)
----------------------------------------------------------------
Los **días de FOMC**, las **publicaciones macro** (CPI/PCE/NFP/ISM) y los
**resultados de mega-caps** no se deducen de ninguna regla: necesitan una fuente
externa (calendario público de la Fed, *release dates* de FRED/ALFRED, calendario
de resultados). Ese corte se parte a un seguimiento y aquí **no se inventa
ninguna fecha**: el conjunto de FOMC sigue entrando al gate como parámetro
(``plan.md`` §12, regla 17; ``decision/gate.py``). Hasta entonces este agente
publica **solo** los eventos derivables.

Regla única de vencimientos
---------------------------
El OPEX y el roll del ES los marca **``MarketCalendar``** (``opex_dates`` /
``is_opex`` / ``es_roll_dates`` / ``is_es_roll``): este agente **no** vuelve a
derivar el tercer viernes. Antes había dos copias de la regla —una aquí y otra
en ``features/regime.py::_opex_session``—; la local desapareció con la #77, que
además corrigió el caso en el que el tercer viernes es festivo (``2026-06-19``,
Juneteenth). ``features/regime.py`` **sigue** derivando el vencimiento del propio
frame de sesiones porque esa familia no consulta el calendario (*declarado* en
#23 y #79): es la misma regla sobre dos entradas distintas, no dos definiciones.

El módulo es **puro respecto al reloj**: recibe ``as_of`` explícito y un
``MarketCalendar`` ya construido; no lee la red ni el sistema de ficheros.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from cfdtrader.backtest.engine import canonical_text
from cfdtrader.data.calendar import MarketCalendar

__all__ = [
    "BLOCKING_KINDS",
    "CALENDAR_HASH_FORMAT",
    "CALENDAR_HASH_PREFIX",
    "QUARTER_MONTHS",
    "CalendarEvent",
    "EventCalendarInputError",
    "EventCalendarSignal",
    "EventKind",
    "calendar_signal",
    "signal_sha256",
]

#: Formato estable de ``signal_sha256``. El prefijo viaja dentro del valor: un
#: sha256 desnudo (64 hex) es lo que ``detect-secrets`` marca y bloquearía el commit.
CALENDAR_HASH_PREFIX: Final[str] = "sha256:"
CALENDAR_HASH_FORMAT: Final[str] = (
    "sha256(UTF-8) de json.dumps(payload, sort_keys=True, separators=(',', ':'), "
    "ensure_ascii=False) del payload **sin** la clave signal_sha256"
)

#: Meses con vencimiento trimestral: triple *witching* y roll del futuro ES.
QUARTER_MONTHS: Final[tuple[int, ...]] = (3, 6, 9, 12)


class EventCalendarInputError(Exception):
    """Una entrada del agente no es válida (fecha, instante o calendario)."""


class EventKind(StrEnum):
    """Tipos de evento que el núcleo derivable sabe reconocer."""

    MARKET_CLOSED = "market_closed"
    HALF_SESSION = "half_session"
    OPEX = "opex"
    TRIPLE_WITCHING = "triple_witching"
    ES_ROLL = "es_roll"


#: Eventos que el proyecto trata como **bloqueo duro** por defecto: el mercado
#: cerrado (regla 19) y la media sesión (regla 18). OPEX, triple *witching* y roll
#: del ES son **informativos**: no son regla dura de ``plan.md`` §12.
BLOCKING_KINDS: Final[frozenset[EventKind]] = frozenset(
    {EventKind.MARKET_CLOSED, EventKind.HALF_SESSION}
)


class CalendarEvent(BaseModel):
    """Un evento del calendario de la sesión evaluada."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EventKind = Field(description="tipo de evento del núcleo derivable")
    name: str = Field(
        description="descripción legible; en cierre y media sesión, el `reason` del calendario"
    )
    at_utc: datetime | None = Field(
        default=None,
        description="instante del evento en UTC; None si ocupa la sesión entera (OPEX, roll…)",
    )
    blocking: bool = Field(description="True si es un bloqueo duro por defecto (reglas 18/19)")


class EventCalendarSignal(BaseModel):
    """Señal tipada del calendario de eventos para una sesión (`plan.md` §7.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session: date = Field(description="sesión evaluada (ET)")
    as_of: datetime = Field(
        description="instante de la decisión: entrada explícita, el agente no tiene reloj"
    )
    is_session: bool = Field(description="False si el mercado está cerrado ese día")
    session_offset_hours: int = Field(
        description="horas con Madrid ese día: 6 normalmente, 5 en las ventanas DST"
    )
    events: tuple[CalendarEvent, ...] = Field(
        default=(), description="eventos del día, en orden de emisión"
    )
    blocking: tuple[str, ...] = Field(
        default=(),
        description="códigos de los eventos que bloquean (subconjunto de `events`)",
    )
    signal_sha256: str = Field(
        description=f"{CALENDAR_HASH_PREFIX}<64 hex>; ver CALENDAR_HASH_FORMAT"
    )


def _require_date(value: object, *, field_name: str) -> date:
    """Una fecha de calendario: un ``datetime`` no cuela como día (arrastraría una hora)."""
    if isinstance(value, datetime) or not isinstance(value, date):
        raise EventCalendarInputError(
            f"{field_name}: se espera una `datetime.date`, no {type(value).__name__}"
        )
    return value


def _require_aware(value: object, *, field_name: str) -> datetime:
    """Un instante con zona: un ``datetime`` sin TZ no identifica un momento."""
    if not isinstance(value, datetime):
        raise EventCalendarInputError(
            f"{field_name}: se espera un `datetime`, no {type(value).__name__}"
        )
    if value.utcoffset() is None:
        raise EventCalendarInputError(f"{field_name}: se espera un `datetime` con zona (TZ-aware)")
    return value


def _is_opex_session(calendar: MarketCalendar, day: date) -> bool:
    """``True`` si ``day`` es la **sesión** del vencimiento mensual de opciones.

    Delegado en ``MarketCalendar.is_opex`` (issue #77): el tercer viernes del mes
    **rodado** a la sesión inmediatamente anterior cuando no es sesión (Viernes
    Santo, Juneteenth) es una **sola** definición, y vive en el calendario.
    """
    return calendar.is_opex(day)


def _events_for(calendar: MarketCalendar, day: date) -> tuple[CalendarEvent, ...]:
    """Eventos derivables del día: cierre → media sesión → OPEX → *witching* → roll."""
    info = calendar.session(day)
    if not info.is_session:
        return (
            CalendarEvent(
                kind=EventKind.MARKET_CLOSED,
                name=info.reason or "mercado cerrado",
                blocking=True,
            ),
        )

    events: list[CalendarEvent] = []
    if info.is_half_day:
        events.append(
            CalendarEvent(
                kind=EventKind.HALF_SESSION,
                name=info.reason or "media sesión: cierre a las 13:00 ET",
                at_utc=info.close_utc,
                blocking=True,
            )
        )
    if _is_opex_session(calendar, day):
        events.append(
            CalendarEvent(
                kind=EventKind.OPEX,
                name="OPEX: vencimiento mensual de opciones",
                blocking=False,
            )
        )
        if day.month in QUARTER_MONTHS:
            events.append(
                CalendarEvent(
                    kind=EventKind.TRIPLE_WITCHING,
                    name="triple witching: vencimiento trimestral",
                    blocking=False,
                )
            )
    if calendar.is_es_roll(day):
        events.append(
            CalendarEvent(
                kind=EventKind.ES_ROLL,
                name="roll trimestral del futuro ES",
                blocking=False,
            )
        )
    return tuple(events)


def calendar_signal(calendar: object, session: object, *, as_of: object) -> EventCalendarSignal:
    """Señal del calendario de eventos para ``session`` en el instante ``as_of``.

    Parameters
    ----------
    calendar:
        ``MarketCalendar`` ya construido (la **única** fuente de festivos, medias
        sesiones, DST y roll del ES; no se re-deriva nada).
    session:
        Sesión evaluada (ET). Puede ser un día sin sesión: entonces la señal sale
        con un único evento ``market_closed`` que bloquea.
    as_of:
        Instante de la decisión, con zona. Es una **entrada**: el módulo no lee el reloj.

    Los parámetros van anotados como ``object`` a propósito: el contrato se comprueba aquí
    con errores tipados, de forma que un llamante sin tipar reciba un
    ``EventCalendarInputError`` claro y nunca un ``AttributeError`` suelto.
    """
    if not isinstance(calendar, MarketCalendar):
        raise EventCalendarInputError(
            f"calendar: se espera un MarketCalendar ya construido, no {type(calendar).__name__}"
        )
    day = _require_date(session, field_name="session")
    instant = _require_aware(as_of, field_name="as_of")

    events = _events_for(calendar, day)
    blocking = tuple(event.kind.value for event in events if event.blocking)
    provisional = EventCalendarSignal(
        session=day,
        as_of=instant,
        is_session=calendar.is_session(day),
        session_offset_hours=calendar.session_offset_hours(day),
        events=events,
        blocking=blocking,
        signal_sha256=CALENDAR_HASH_PREFIX,
    )
    return provisional.model_copy(update={"signal_sha256": signal_sha256(provisional)})


def _json_payload(signal: EventCalendarSignal) -> dict[str, object]:
    """Payload JSON puro de la señal, **sin** la clave del hash.

    Una salida no se hashea a sí misma (misma regla que ``gate_sha256``).
    """
    return {
        "session": signal.session.isoformat(),
        "as_of": signal.as_of.astimezone(UTC).isoformat(),
        "is_session": signal.is_session,
        "session_offset_hours": signal.session_offset_hours,
        "events": [
            {
                "kind": event.kind.value,
                "name": event.name,
                "at_utc": (
                    None if event.at_utc is None else event.at_utc.astimezone(UTC).isoformat()
                ),
                "blocking": event.blocking,
            }
            for event in signal.events
        ],
        "blocking": list(signal.blocking),
    }


def signal_sha256(signal: EventCalendarSignal) -> str:
    """``signal_sha256`` del payload **sin** la clave del hash (``CALENDAR_HASH_FORMAT``)."""
    digest = hashlib.sha256(canonical_text(_json_payload(signal)).encode("utf-8")).hexdigest()
    return f"{CALENDAR_HASH_PREFIX}{digest}"
