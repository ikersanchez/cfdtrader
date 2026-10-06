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


def test_opex_rolls_the_third_friday_back_to_a_session() -> None:
    """El vencimiento mensual es el tercer viernes **rodado** a sesión (issue #77).

    Seis terceros viernes de mar/jun/sep/dic entre 2005 y 2026 no son sesión
    (Viernes Santo o Juneteenth): el vencimiento real es la sesión anterior. El
    peor caso es ``2026-06-19``, festivo y vencimiento de junio a la vez.
    """
    calendar = MarketCalendar(CalendarConfig(), years=tuple(range(2005, 2027)))

    assert len(calendar.opex_dates(2026)) == 12
    assert tuple(day.month for day in calendar.opex_dates(2026)) == tuple(range(1, 13))
    assert all(calendar.is_session(day) for day in calendar.opex_dates(2026))

    for nominal, rolled in (
        (date(2008, 3, 21), date(2008, 3, 20)),
        (date(2014, 4, 18), date(2014, 4, 17)),
        (date(2019, 4, 19), date(2019, 4, 18)),
        (date(2022, 4, 15), date(2022, 4, 14)),
        (date(2025, 4, 18), date(2025, 4, 17)),
        (date(2026, 6, 19), date(2026, 6, 18)),
    ):
        assert calendar.is_session(nominal) is False, f"{nominal} debería estar cerrado"
        assert calendar.is_opex(nominal) is False, "un día cerrado no es la sesión OPEX"
        assert calendar.is_opex(rolled) is True

    # el roll trimestral es el subconjunto trimestral de la OPEX, ya rodada
    assert calendar.es_roll_dates(2026) == (
        date(2026, 3, 20),
        date(2026, 6, 18),
        date(2026, 9, 18),
        date(2026, 12, 18),
    )
    assert calendar.is_es_roll(date(2026, 6, 18)) is True
    assert calendar.is_es_roll(date(2026, 6, 19)) is False
    assert all(day in calendar.opex_dates(day.year) for day in calendar.es_roll_dates(2026))

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


# ─────────────────────────────────────────────────────────────────────────────
# #124 · El calendario de FOMC declarado: fuente, fecha y año ausente
# ─────────────────────────────────────────────────────────────────────────────
def test_124_the_declared_calendar_declares_its_source_date_and_meetings() -> None:
    """Sin procedencia ni fecha, un calendario declarado no es auditable."""
    from cfdtrader.data.calendar import DEFAULT_FOMC_CALENDAR_PATH, load_fomc_calendar

    config = load_fomc_calendar()
    assert config is not None, f"falta el artefacto {DEFAULT_FOMC_CALENDAR_PATH}"
    assert config.source.startswith("https://www.federalreserve.gov/")
    assert config.verified_on == date(2026, 10, 4)
    assert config.tentative_note.strip(), "la fuente avisa de que las fechas son provisionales"
    assert len(config.meetings[2027]) == 16, "las ocho reuniones de 2027, con sus dos dias"
    assert len(config.meetings[2026]) == 16, "y las ocho de 2026, recuperadas en #132"


def test_124_an_undeclared_year_is_none_and_not_an_empty_tuple() -> None:
    """«Nadie lo ha declarado» y «declarado, y no hay ninguna» no son lo mismo."""
    from cfdtrader.data.calendar import fomc_dates_for, load_fomc_calendar

    config = load_fomc_calendar()
    assert config is not None
    assert fomc_dates_for(config, 2030) is None, "2030 no esta declarado y no se inventa"
    declared = fomc_dates_for(config, 2027)
    assert declared is not None
    assert list(declared) == sorted(set(declared)), "sin repetidas y en orden"


def test_124_a_missing_artifact_is_not_declared_and_a_broken_one_raises(tmp_path: Path) -> None:
    """``None`` si no existe; un fichero mal formado falla al arrancar (``tech_stack.md`` §4.2)."""
    from cfdtrader.data.calendar import load_fomc_calendar

    assert load_fomc_calendar(tmp_path / "no-existe.yaml") is None

    unknown = tmp_path / "desconocido.yaml"
    unknown.write_text(
        "version: 1\nsource: https://x\nverified_on: 2026-10-03\n"
        "tentative_note: n\ncampo_de_mas: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError):
        load_fomc_calendar(unknown)


def test_124_a_year_declared_without_meetings_is_rejected(tmp_path: Path) -> None:
    """Declarar el año vacío afirmaría «se sabe, y no hay ninguna»: la indistinción a eliminar."""
    from cfdtrader.data.calendar import load_fomc_calendar

    empty = tmp_path / "vacio.yaml"
    empty.write_text(
        "version: 1\nsource: https://x\nverified_on: 2026-10-03\ntentative_note: n\n"
        "meetings:\n  2026: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="sin ninguna reunion"):
        load_fomc_calendar(empty)


# ─────────────────────────────────────────────────────────────────────────────
# #132 · El año que faltaba: recuperado de la fuente y declarado
# ─────────────────────────────────────────────────────────────────────────────
def test_132_2026_is_now_declared_with_its_sixteen_days() -> None:
    """El año que quedaba pendiente está declarado, con sus ocho reuniones y sus dos días (#132)."""
    from cfdtrader.data.calendar import fomc_dates_for, load_fomc_calendar, pending_reason

    config = load_fomc_calendar()
    assert config is not None
    declared = fomc_dates_for(config, 2026)
    assert declared is not None, "2026 sigue siendo «nadie lo ha declarado»: no es lo que toca"
    assert len(declared) == 16, "las ocho reuniones, con sus dos dias"
    assert list(declared) == sorted(set(declared)), "sin repetidas y en orden"
    assert declared[0] == date(2026, 1, 27) and declared[-1] == date(2026, 12, 9)
    assert pending_reason(config, 2026) is None, "no puede estar declarado y pendiente a la vez"
    assert config.pending == {}, "y no queda ningun año pendiente"


# ─────────────────────────────────────────────────────────────────────────────
# #132 · El parser de la fuente: leerla sin que se cuele nada
# ─────────────────────────────────────────────────────────────────────────────
_SOURCE_FIXTURE = """
<h4>2029 FOMC Meetings</h4>
January 30-31 Statement: PDF | HTML Minutes: PDF | HTML (Released February 21, 2029)
March 20-21* Statement: PDF | HTML
October 30 - November 1 Statement: PDF | HTML
December 11-12* Statement: PDF | HTML
Note: A two-day meeting is scheduled for January 29-30, 2030.
<h4>2028 FOMC Meetings</h4>
January 25-26
"""


def test_132_the_parser_reads_the_announced_year_and_ignores_the_noise() -> None:
    """Ni la fecha de publicacion de las actas ni la nota del año siguiente son dias de reunion."""
    from cfdtrader.data.calendar import parse_declared_meetings

    days = parse_declared_meetings(_SOURCE_FIXTURE, 2029)
    assert [day.isoformat() for day in days] == [
        "2029-01-30",
        "2029-01-31",
        "2029-03-20",
        "2029-03-21",
        "2029-10-30",
        "2029-11-01",
        "2029-12-11",
        "2029-12-12",
    ]
    assert date(2029, 2, 21) not in days, "la publicacion de las actas no es una reunion"
    assert not any(day.year == 2030 for day in days), (
        "la nota nombra el año **siguiente**: no cuela"
    )
    assert parse_declared_meetings(_SOURCE_FIXTURE, 2028) == (
        date(2028, 1, 25),
        date(2028, 1, 26),
    ), "cada bloque se lee por separado"


def test_132_an_absent_year_is_empty_and_never_an_invention() -> None:
    """`()` es «ese año no aparece en el texto», no «ese año no tiene reuniones»."""
    from cfdtrader.data.calendar import parse_declared_meetings

    assert parse_declared_meetings(_SOURCE_FIXTURE, 2031) == ()
    assert parse_declared_meetings("", 2026) == ()
    assert parse_declared_meetings("sin años aqui", 2026) == ()


def test_132_the_parser_reproduces_the_declared_years() -> None:
    """El parser (sobre el maquetado real que se copio) da lo mismo que el artefacto declarado.

    El bloque de abajo es el de 2026 tal y como lo sirve la fuente —descargado el 2026-10-04—, con
    su ruido (actas, notas, materiales). Si el parser se desvia, esto lo dice sin tocar la red.
    """
    from cfdtrader.data.calendar import fomc_dates_for, load_fomc_calendar, parse_declared_meetings

    source_2026 = """
    2026 FOMC Meetings January 27-28 Statement: PDF (Released February 18, 2026)
    March 17-18* Statement: PDF (Released April 08, 2026)
    April 28-29 Statement: PDF (Released May 20, 2026)
    June 16-17* Statement: PDF (Released July 08, 2026)
    July 28-29 Statement: PDF (Released August 19, 2026)
    September 15-16* Statement: PDF (Released October 07, 2026)
    October 27-28 Statement: PDF
    December 8-9* Statement: PDF
    Note: A two-day meeting is scheduled for January 26-27, 2027.
    """

    config = load_fomc_calendar()
    assert config is not None
    assert parse_declared_meetings(source_2026, 2026) == fomc_dates_for(config, 2026)


def test_132_a_year_declared_and_pending_at_once_is_rejected(tmp_path: Path) -> None:
    """Las dos cosas a la vez harían mentir al aviso según cuál se leyera primero."""
    from cfdtrader.data.calendar import load_fomc_calendar

    contradictory = tmp_path / "contradictorio.yaml"
    contradictory.write_text(
        "version: 1\nsource: https://x\nverified_on: 2026-10-03\ntentative_note: n\n"
        "meetings:\n  2026:\n    - 2026-01-27\n"
        "pending:\n  2026:\n    reason: porque si\n    attempted_on: 2026-10-04\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="pendientes"):
        load_fomc_calendar(contradictory)


def test_132_a_pending_entry_without_a_reason_is_rejected(tmp_path: Path) -> None:
    """Una entrada pendiente sin motivo es un hueco mudo: la indistincion que #114 quito."""
    from cfdtrader.data.calendar import load_fomc_calendar

    muted = tmp_path / "mudo.yaml"
    muted.write_text(
        "version: 1\nsource: https://x\nverified_on: 2026-10-03\ntentative_note: n\n"
        "pending:\n  2026:\n    reason: ''\n    attempted_on: 2026-10-04\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError):
        load_fomc_calendar(muted)
