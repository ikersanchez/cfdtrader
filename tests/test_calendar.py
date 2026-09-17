"""Tests del calendario de sesiones, festivos, medias sesiones y DST (tarea #4).

El artefacto verificable son las respuestas a: «¿es sesión?», «¿cuántas horas
dura?» y «¿en qué ventana anual el horario de Madrid se adelanta una hora?».
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from cfdtrader.data.calendar import (
    HALF_SESSION_CLOSE_ET,
    SESSION_CLOSE_ET,
    SESSION_OPEN_ET,
    CalendarConfig,
    MarketCalendar,
    SessionInfo,
    load_calendar,
)
from cfdtrader.data.settings import ConfigurationError

#: Años materializados en los tests: rango amplio y explícito, sin depender del reloj.
YEARS = tuple(range(2018, 2031))


@pytest.fixture
def calendar() -> MarketCalendar:
    """Calendario con los años de test materializados."""
    return MarketCalendar(CalendarConfig(), years=YEARS)


def _time_in(zone: str, *, y: int, m: int, d: int, hour: int, minute: int = 0) -> datetime:
    """Instante construido en esa zona, para comparar sin ambigüedad."""
    return datetime(y, m, d, hour, minute, tzinfo=ZoneInfo(zone))


# ─────────────────────────────────────────────────────────────────────────────
# Festivos (A: `plan.md` §8.3)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("day", "name"),
    [
        (date(2024, 1, 1), "New Year"),
        (date(2024, 1, 15), "Martin Luther King"),
        (date(2024, 2, 19), "Washington"),
        (date(2024, 3, 29), "Good Friday"),
        (date(2024, 5, 27), "Memorial"),
        (date(2024, 6, 19), "Juneteenth"),
        (date(2024, 7, 4), "Independence"),
        (date(2024, 9, 2), "Labor"),
        (date(2024, 11, 28), "Thanksgiving"),
        (date(2024, 12, 25), "Christmas"),
    ],
)
def test_known_us_holidays_are_not_sessions(calendar: MarketCalendar, day: date, name: str) -> None:
    """Los diez festivos que enumera `plan.md` §8.3 cierran el mercado."""
    info = calendar.session(day)

    assert info.is_session is False
    assert info.duration_hours == 0.0
    assert info.open_utc is None and info.close_utc is None
    assert info.reason is not None and "festivo" in info.reason


def test_the_observed_holiday_rule_wins_over_the_half_day_rule(calendar: MarketCalendar) -> None:
    """El 2026-07-03 es el festivo *observado* del 4 de julio: cierra, no media sesión."""
    info = calendar.session(date(2026, 7, 3))

    assert info.is_session is False
    assert info.is_half_day is False
    assert info.reason is not None and "festivo" in info.reason


def test_spanish_holidays_do_not_close_the_american_market(calendar: MarketCalendar) -> None:
    """Los festivos españoles no son festivos del mercado objetivo (`plan.md` §8.3)."""
    for day in (date(2024, 5, 1), date(2025, 1, 6), date(2024, 10, 12)):
        info = calendar.session(day)
        if day.weekday() < 5:
            assert info.is_session is True, f"{day.isoformat()} debería ser sesión"


def test_weekends_are_not_sessions(calendar: MarketCalendar) -> None:
    """Un fin de semana no es sesión, y el motivo lo dice."""
    info = calendar.session(date(2024, 6, 8))  # sábado
    assert info.is_session is False
    assert info.reason == "fin de semana"


# ─────────────────────────────────────────────────────────────────────────────
# Medias sesiones
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "day",
    [
        date(2024, 11, 29),  # día después de Acción de Gracias
        date(2025, 11, 28),
        date(2024, 12, 24),  # víspera de Navidad
        date(2024, 7, 3),  # víspera del Día de la Independencia
        date(2023, 7, 3),
    ],
)
def test_known_half_days_close_at_thirteen_et(calendar: MarketCalendar, day: date) -> None:
    """Las medias sesiones conocidas cierran a las 13:00 ET y duran 3,5 h."""
    info = calendar.session(day)

    assert info.is_session is True
    assert info.is_half_day is True
    assert info.duration_hours == 3.5
    assert info.close_et is not None
    assert info.close_et.time().replace(tzinfo=None) == HALF_SESSION_CLOSE_ET
    assert info.reason is not None and "media sesión" in info.reason


def test_a_full_session_is_six_and_a_half_hours(calendar: MarketCalendar) -> None:
    """Una sesión completa dura 6,5 h y cierra a las 16:00 ET."""
    info = calendar.session(date(2024, 6, 10))

    assert info.is_session is True
    assert info.is_half_day is False
    assert info.duration_hours == 6.5
    assert info.open_et is not None and info.open_et.time().replace(tzinfo=None) == SESSION_OPEN_ET
    assert info.close_et is not None
    assert info.close_et.time().replace(tzinfo=None) == SESSION_CLOSE_ET


def test_extra_half_days_can_be_declared_in_the_config() -> None:
    """Un cierre anticipado anunciado fuera de la regla se declara con fecha explícita."""
    config = CalendarConfig(
        extra_holidays=(date(2025, 1, 9),),  # luto nacional
        extra_half_days=(date(2025, 12, 31),),
    )
    calendar = MarketCalendar(config, years=YEARS)

    assert calendar.session(date(2025, 1, 9)).is_session is False
    assert calendar.session(date(2025, 12, 31)).is_half_day is True


# ─────────────────────────────────────────────────────────────────────────────
# Instantes UTC y hora de Madrid (solo presentación)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("day", "open_utc", "close_utc"),
    [
        # EDT: 09:30 ET = 13:30 UTC y 16:00 ET = 20:00 UTC
        (
            date(2024, 6, 10),
            datetime(2024, 6, 10, 13, 30, tzinfo=UTC),
            datetime(2024, 6, 10, 20, 0, tzinfo=UTC),
        ),
        # EST: 09:30 ET = 14:30 UTC y 16:00 ET = 21:00 UTC
        (
            date(2024, 1, 10),
            datetime(2024, 1, 10, 14, 30, tzinfo=UTC),
            datetime(2024, 1, 10, 21, 0, tzinfo=UTC),
        ),
    ],
)
def test_session_bounds_in_utc_never_use_a_fixed_offset(
    calendar: MarketCalendar, day: date, open_utc: datetime, close_utc: datetime
) -> None:
    """La conversión es con `zoneinfo`: 13:30/20:00 en EDT y 14:30/21:00 en EST."""
    assert calendar.open_utc(day) == open_utc
    assert calendar.close_utc(day) == close_utc
    assert calendar.bounds_utc(day) == (open_utc, close_utc)


def test_madrid_is_only_presentation(calendar: MarketCalendar) -> None:
    """La hora de Madrid es una vista del mismo instante, no otra fuente de verdad."""
    open_utc, close_utc = calendar.bounds_utc(date(2024, 6, 10))
    open_madrid, close_madrid = calendar.madrid_bounds(date(2024, 6, 10))

    assert open_madrid == _time_in("Europe/Madrid", y=2024, m=6, d=10, hour=15, minute=30)
    assert close_madrid == _time_in("Europe/Madrid", y=2024, m=6, d=10, hour=22, minute=0)
    # El instante es el mismo: solo cambia cómo se escribe.
    assert open_madrid.timestamp() == open_utc.timestamp()
    assert close_madrid.timestamp() == close_utc.timestamp()


def test_a_half_day_in_madrid_closes_at_nineteen(calendar: MarketCalendar) -> None:
    """Una media sesión cierra a las 19:00 de Madrid (`plan.md` §8.3)."""
    _, close_madrid = calendar.madrid_bounds(date(2024, 11, 29))
    assert close_madrid.hour == 19


# ─────────────────────────────────────────────────────────────────────────────
# DST: las dos ventanas anuales de desfase
# ─────────────────────────────────────────────────────────────────────────────
def test_us_and_european_dst_transitions_for_2024(calendar: MarketCalendar) -> None:
    """EE. UU. cambia el 10 de marzo y el 3 de noviembre; Europa, el 31/03 y el 27/10."""
    assert calendar.us_dst_transitions(2024) == (date(2024, 3, 10), date(2024, 11, 3))
    assert calendar.eu_dst_transitions(2024) == (date(2024, 3, 31), date(2024, 10, 27))


def test_the_transitions_match_the_real_timezone_database(calendar: MarketCalendar) -> None:
    """Las fechas calculadas son las que aplica de verdad `zoneinfo`."""
    for year in range(2020, 2027):
        us_spring, us_fall = calendar.us_dst_transitions(year)
        eu_spring, eu_fall = calendar.eu_dst_transitions(year)

        assert _offset("America/New_York", us_spring - timedelta(days=1)) == timedelta(hours=-5)
        assert _offset("America/New_York", us_spring) == timedelta(hours=-4)
        assert _offset("America/New_York", us_fall) == timedelta(hours=-5)
        assert _offset("Europe/Madrid", eu_spring - timedelta(days=1)) == timedelta(hours=1)
        assert _offset("Europe/Madrid", eu_spring) == timedelta(hours=2)
        assert _offset("Europe/Madrid", eu_fall) == timedelta(hours=1)


def test_the_two_mismatch_windows_of_2024(calendar: MarketCalendar) -> None:
    """Las dos ventanas: del 10 al 31 de marzo, y del 27 de octubre al 3 de noviembre."""
    assert calendar.dst_mismatch_windows(2024) == (
        (date(2024, 3, 10), date(2024, 3, 31)),
        (date(2024, 10, 27), date(2024, 11, 3)),
    )


def test_during_the_mismatch_the_market_opens_an_hour_earlier_in_madrid(
    calendar: MarketCalendar,
) -> None:
    """En la ventana de marzo el mercado abre a las 14:30 de Madrid en vez de a las 15:30."""
    outside = calendar.madrid_bounds(date(2024, 3, 8))[0]
    inside = calendar.madrid_bounds(date(2024, 3, 18))[0]

    assert (outside.hour, outside.minute) == (15, 30)
    assert (inside.hour, inside.minute) == (14, 30)
    assert calendar.session_offset_hours(date(2024, 3, 8)) == 6
    assert calendar.session_offset_hours(date(2024, 3, 18)) == 5
    assert calendar.has_dst_mismatch(date(2024, 3, 18)) is True
    assert calendar.has_dst_mismatch(date(2024, 3, 8)) is False


def test_every_year_has_exactly_two_mismatch_windows(calendar: MarketCalendar) -> None:
    """La forma del calendario es estable: dos ventanas cada año, la primera en marzo."""
    for year in range(2019, 2031):
        windows = calendar.dst_mismatch_windows(year)
        assert len(windows) == 2
        assert windows[0][0].month == 3
        assert all(start < end for start, end in windows)
        assert len(calendar.dst_mismatch_days(year)) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Roll del futuro ES y enumeración de sesiones
# ─────────────────────────────────────────────────────────────────────────────
def test_es_roll_happens_four_times_a_year(calendar: MarketCalendar) -> None:
    """El futuro ES vence el tercer viernes de marzo, junio, septiembre y diciembre."""
    assert calendar.es_roll_dates(2024) == (
        date(2024, 3, 15),
        date(2024, 6, 21),
        date(2024, 9, 20),
        date(2024, 12, 20),
    )
    assert calendar.is_es_roll(date(2024, 6, 21)) is True
    assert calendar.is_es_roll(date(2024, 6, 20)) is False


def test_sessions_between_skips_weekends_and_holidays(calendar: MarketCalendar) -> None:
    """De lunes a viernes de una semana con festivo salen cuatro sesiones, no cinco."""
    found = calendar.sessions(date(2024, 6, 17), date(2024, 6, 21))
    days = [info.day for info in found]

    assert days == [
        date(2024, 6, 17),
        date(2024, 6, 18),
        # 19 de junio: Juneteenth
        date(2024, 6, 20),
        date(2024, 6, 21),
    ]
    assert all(isinstance(info, SessionInfo) for info in found)


def test_previous_and_next_session_skip_the_weekend(calendar: MarketCalendar) -> None:
    """De lunes a viernes anteriores/siguientes, saltando el fin de semana."""
    assert calendar.previous_session(date(2024, 6, 10)) == date(2024, 6, 7)
    assert calendar.next_session(date(2024, 6, 7)) == date(2024, 6, 10)


def test_bounds_of_a_non_session_raise(calendar: MarketCalendar) -> None:
    """Pedir la ventana de un día sin sesión es un error, no un valor inventado."""
    with pytest.raises(ValueError, match="no es sesión"):
        calendar.bounds_utc(date(2024, 12, 25))


# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────
def test_calendar_yaml_is_valid_and_declares_the_reference_timezone() -> None:
    """El fichero de configuración se valida al cargarlo y ancla la referencia en ET."""
    calendar = load_calendar(years=YEARS)
    assert calendar.config.reference_timezone == "America/New_York"
    assert calendar.config.presentation_timezone == "Europe/Madrid"
    assert calendar.config.extra_holidays == ()
    assert calendar.config.extra_half_days == ()


def test_a_broken_calendar_yaml_fails_at_load(tmp_path: Path) -> None:
    """Un fichero mal formado falla al arrancar, no a mitad del pipeline."""
    broken = tmp_path / "calendar.yaml"
    broken.write_text("extra_half_days: [no-es-una-fecha]\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="calendario inválido"):
        load_calendar(broken, years=YEARS)


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    """La configuración no admite claves inventadas: un error de config es un error."""
    broken = tmp_path / "calendar.yaml"
    broken.write_text("festivos_espanoles: [2024-01-06]\n", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_calendar(broken, years=YEARS)


def _offset(zone: str, day: date) -> timedelta:
    """Desplazamiento UTC de esa zona a mediodía de ese día."""
    noon = datetime.combine(day, time(12, 0), tzinfo=ZoneInfo(zone))
    return noon.utcoffset() or timedelta()
