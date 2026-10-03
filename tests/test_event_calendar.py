"""Tests del calendario de eventos (`#34`, núcleo derivable).

El artefacto verificable es la **señal tipada** que responde: «¿qué eventos de
calendario caen en esta sesión y cuáles bloquean?». Todo se prueba con casos
declarados a mano sobre un `MarketCalendar` real: festivos, fin de semana, media
sesión, OPEX (nominal y rodado), triple *witching*, roll del ES y las dos
ventanas de DST.

Frontera declarada: FOMC, publicaciones macro y resultados de mega-caps **no**
están aquí (necesitan fuente externa); el módulo lo dice y estas pruebas fijan
que no los inventa.
"""

from __future__ import annotations

import inspect
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from cfdtrader.agents.event_calendar import (
    BLOCKING_KINDS,
    CALENDAR_HASH_PREFIX,
    QUARTER_MONTHS,
    CalendarEvent,
    EventCalendarInputError,
    EventCalendarSignal,
    EventKind,
    calendar_signal,
    signal_sha256,
)
from cfdtrader.data.calendar import EASTERN, CalendarConfig, MarketCalendar

#: Años materializados en los tests: rango amplio y explícito, sin depender del reloj.
YEARS: Final[tuple[int, ...]] = tuple(range(2018, 2031))

#: Instante de decisión declarado. El agente no lo relaciona con la sesión: es solo su `as_of`.
AS_OF: Final[datetime] = datetime(2026, 6, 18, 12, 45, tzinfo=UTC)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODULE: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "agents" / "event_calendar.py"


@pytest.fixture
def calendar() -> MarketCalendar:
    """Calendario con los años de test materializados."""
    return MarketCalendar(CalendarConfig(), years=YEARS)


def _kinds(signal: EventCalendarSignal) -> tuple[str, ...]:
    """Los `kind` de los eventos de la señal, en orden."""
    return tuple(event.kind.value for event in signal.events)


# ─────────────────────────────────────────────────────────────────────────────
# A1 · Módulo, contrato y ausencia de reloj
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_signal_function_takes_as_of_as_a_keyword_and_has_no_clock() -> None:
    """`calendar_signal` exige `as_of` (keyword-only) y el módulo no lee el reloj."""
    parameters = inspect.signature(calendar_signal).parameters

    assert parameters["as_of"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["as_of"].default is inspect.Parameter.empty
    source = MODULE.read_text(encoding="utf-8")
    assert "datetime.now" not in source and "date.today" not in source


def test_a1_the_module_does_not_talk_to_the_network() -> None:
    """El núcleo derivable no importa ninguna fuente de red."""
    source = MODULE.read_text(encoding="utf-8")
    for forbidden in ("import yfinance", "import requests", "import urllib", "import httpx"):
        assert forbidden not in source


# ─────────────────────────────────────────────────────────────────────────────
# A2/A3 · Mercado cerrado (regla 19): festivos y fin de semana
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_a_us_holiday_is_a_blocking_market_closed_event(calendar: MarketCalendar) -> None:
    """Un festivo de EE. UU. es `market_closed`, bloquea y trae el motivo del calendario."""
    signal = calendar_signal(calendar, date(2024, 12, 25), as_of=AS_OF)

    assert signal.is_session is False
    assert _kinds(signal) == ("market_closed",)
    assert signal.blocking == ("market_closed",)
    assert signal.events[0].kind in BLOCKING_KINDS
    assert signal.events[0].name == calendar.session(date(2024, 12, 25)).reason


def test_a3_a_weekend_is_a_blocking_market_closed_event(calendar: MarketCalendar) -> None:
    """Un fin de semana sale como `market_closed` con el motivo «fin de semana»."""
    signal = calendar_signal(calendar, date(2024, 6, 8), as_of=AS_OF)

    assert signal.is_session is False
    assert _kinds(signal) == ("market_closed",)
    assert signal.events[0].name == "fin de semana"


# ─────────────────────────────────────────────────────────────────────────────
# A4 · Media sesión (regla 18)
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_a_half_session_blocks_and_carries_the_13_00_et_close(calendar: MarketCalendar) -> None:
    """El día después de Acción de Gracias bloquea y trae el cierre de las 13:00 ET en UTC."""
    day = date(2026, 11, 27)
    signal = calendar_signal(calendar, day, as_of=AS_OF)

    assert signal.is_session is True
    assert _kinds(signal) == ("half_session",)
    assert signal.blocking == ("half_session",)
    event = signal.events[0]
    assert event.kind is EventKind.HALF_SESSION
    assert event.blocking is True
    assert event.at_utc == calendar.session(day).close_utc
    assert event.at_utc is not None
    assert event.at_utc.astimezone(EASTERN).hour == 13
    assert event.at_utc.astimezone(EASTERN).minute == 0


# ─────────────────────────────────────────────────────────────────────────────
# A5/A6/A7 · OPEX y triple witching (informativos, no bloquean)
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_a_quarterly_opex_is_opex_and_triple_witching_without_blocking(
    calendar: MarketCalendar,
) -> None:
    """El tercer viernes trimestral (2024-06-21) es OPEX, triple *witching* y roll del ES."""
    signal = calendar_signal(calendar, date(2024, 6, 21), as_of=AS_OF)

    assert _kinds(signal) == ("opex", "triple_witching", "es_roll")
    assert signal.blocking == ()
    assert all(event.blocking is False for event in signal.events)


def test_a6_a_non_quarterly_opex_is_opex_but_not_triple_witching(
    calendar: MarketCalendar,
) -> None:
    """El tercer viernes de julio (2024-07-19) es OPEX pero no triple *witching*."""
    signal = calendar_signal(calendar, date(2024, 7, 19), as_of=AS_OF)

    assert "opex" in _kinds(signal)
    assert "triple_witching" not in _kinds(signal)


def test_a7_the_opex_rolls_back_when_the_third_friday_is_a_holiday(
    calendar: MarketCalendar,
) -> None:
    """En 2026 el tercer viernes de junio (19, Juneteenth) es festivo: el OPEX cae en la previa."""
    june = calendar_signal(calendar, date(2026, 6, 18), as_of=AS_OF)
    holiday = calendar_signal(calendar, date(2026, 6, 19), as_of=AS_OF)

    assert calendar.is_session(date(2026, 6, 19)) is False
    assert _kinds(june) == ("opex", "triple_witching")
    assert _kinds(holiday) == ("market_closed",)
    # El 18 no es el tercer viernes nominal: la rodadura exige contar desde el calendario.
    assert calendar.is_session(date(2026, 6, 18)) is True


# ─────────────────────────────────────────────────────────────────────────────
# A8 · Roll trimestral del futuro ES
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_the_es_roll_is_marked_on_a_normal_quarterly_friday(
    calendar: MarketCalendar,
) -> None:
    """En un año normal, el roll del ES cae en el tercer viernes trimestral (2024-06-21)."""
    signal = calendar_signal(calendar, date(2024, 6, 21), as_of=AS_OF)

    assert "es_roll" in _kinds(signal)
    assert calendar.es_roll_dates(2024) == (
        date(2024, 3, 15),
        date(2024, 6, 21),
        date(2024, 9, 20),
        date(2024, 12, 20),
    )
    assert tuple(day.month for day in calendar.es_roll_dates(2024)) == QUARTER_MONTHS


# ─────────────────────────────────────────────────────────────────────────────
# A9 · Ventanas de DST: el desfase ET↔Madrid (solo presentación)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("day", [date(2024, 3, 15), date(2024, 10, 29)])
def test_a9_a_dst_mismatch_day_carries_the_five_hour_offset(
    calendar: MarketCalendar, day: date
) -> None:
    """En las dos ventanas de desfase con Europa el mercado abre a 5 h de Madrid, no a 6."""
    signal = calendar_signal(calendar, day, as_of=AS_OF)

    assert calendar.has_dst_mismatch(day) is True
    assert signal.session_offset_hours == 5


def test_a9_outside_the_windows_the_offset_is_six(calendar: MarketCalendar) -> None:
    """Fuera de las ventanas el desfase con Madrid es de 6 h."""
    signal = calendar_signal(calendar, date(2024, 6, 21), as_of=AS_OF)

    assert calendar.has_dst_mismatch(date(2024, 6, 21)) is False
    assert signal.session_offset_hours == 6


# ─────────────────────────────────────────────────────────────────────────────
# A10 · Determinismo y hash de la señal
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_the_signal_is_deterministic_and_hashes_its_own_payload(
    calendar: MarketCalendar,
) -> None:
    """Mismo input ⇒ misma señal y mismo hash; otra sesión ⇒ otro hash; el hash es idempotente."""
    first = calendar_signal(calendar, date(2024, 6, 21), as_of=AS_OF)
    second = calendar_signal(calendar, date(2024, 6, 21), as_of=AS_OF)
    other = calendar_signal(calendar, date(2024, 6, 20), as_of=AS_OF)

    assert first == second
    assert first.signal_sha256 == second.signal_sha256
    assert first.signal_sha256.startswith(CALENDAR_HASH_PREFIX)
    assert len(first.signal_sha256) == len(CALENDAR_HASH_PREFIX) + 64
    assert first.signal_sha256 != other.signal_sha256
    assert signal_sha256(first) == first.signal_sha256


# ─────────────────────────────────────────────────────────────────────────────
# A11 · Entradas inválidas: errores tipados, nunca un AttributeError suelto
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "bad_session",
    [datetime(2024, 6, 21, 0, 0, tzinfo=UTC), "2024-06-21", 20240621],
)
def test_a11_a_non_date_session_is_a_typed_error(
    calendar: MarketCalendar, bad_session: object
) -> None:
    """La sesión tiene que ser un `date` de calendario, no un `datetime` ni una cadena."""
    with pytest.raises(EventCalendarInputError, match="session"):
        calendar_signal(calendar, bad_session, as_of=AS_OF)


@pytest.mark.parametrize(
    "bad_as_of",
    [datetime(2024, 6, 21, 12, 45), "2024-06-21T12:45:00+00:00"],
)
def test_a11_an_as_of_without_a_zone_is_a_typed_error(
    calendar: MarketCalendar, bad_as_of: object
) -> None:
    """Un `as_of` que no sea un `datetime` con zona no identifica un instante y es error tipado."""
    with pytest.raises(EventCalendarInputError, match="as_of"):
        calendar_signal(calendar, date(2024, 6, 21), as_of=bad_as_of)


def test_a11_a_non_calendar_is_a_typed_error() -> None:
    """El calendario tiene que ser un `MarketCalendar` ya construido."""
    with pytest.raises(EventCalendarInputError, match="calendar"):
        calendar_signal("no-calendario", date(2024, 6, 21), as_of=AS_OF)


# ─────────────────────────────────────────────────────────────────────────────
# A12 · Frontera heredada de #77 (roll del ES nominal) — declarada, no corregida
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_the_nominal_es_roll_limitation_is_inherited_not_fixed(
    calendar: MarketCalendar,
) -> None:
    """`is_es_roll` marca el tercer viernes nominal (2026-06-19, festivo): el agente lo hereda.

    La sesión rodada (2026-06-18) no trae `es_roll`, porque `MarketCalendar` marca el nominal
    aunque el mercado esté cerrado (issue #77). Aquí se **documenta**, no se corrige.
    """
    assert calendar.is_es_roll(date(2026, 6, 19)) is True
    assert calendar.is_es_roll(date(2026, 6, 18)) is False
    assert "es_roll" not in _kinds(calendar_signal(calendar, date(2026, 6, 18), as_of=AS_OF))
    assert "es_roll" not in _kinds(calendar_signal(calendar, date(2026, 6, 19), as_of=AS_OF))


# ─────────────────────────────────────────────────────────────────────────────
# A13 · Contrato congelado de los modelos
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_the_signal_and_the_event_are_frozen_and_forbid_extra_fields(
    calendar: MarketCalendar,
) -> None:
    """La señal y cada evento son inmutables y no admiten campos inventados."""
    signal = calendar_signal(calendar, date(2024, 6, 21), as_of=AS_OF)

    assert EventCalendarSignal.model_config.get("frozen") is True
    assert CalendarEvent.model_config.get("frozen") is True
    with pytest.raises(ValidationError):
        CalendarEvent.model_validate({"kind": "opex", "name": "x", "blocking": False, "extra": 1})
    with pytest.raises(ValidationError):
        signal.session = date(2024, 1, 1)
