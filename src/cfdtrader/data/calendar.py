"""Calendario de sesiones, festivos y DST (`plan.md` §8.3) — tarea #4.

La pregunta que responde este módulo: **¿esta fecha es sesión válida del mercado
americano y cuántas horas dura?**

Decisiones que implementa:

- **Festivos de Estados Unidos**, no de España: los españoles solo afectan a la
  disponibilidad del operador, no al mercado (``plan.md`` §8.3). La lista la da
  la librería ``holidays`` con el calendario **NYSE**, que además aplica las
  reglas de traslado (el 2026-07-03 es el festivo observado del 4 de julio).
- **Medias sesiones** con cierre a las **13:00 ET** (3,5 h en vez de 6,5 h): el
  día después de Acción de Gracias, el 24 de diciembre y el 3 de julio, siempre
  que sean día laborable y no sean ya festivo. Se pueden añadir **excepciones
  anunciadas** desde ``config/calendar.yaml`` sin tocar el código.
- **Hora de referencia interna: ``America/New_York``**, siempre. ``Europe/Madrid``
  es **solo presentación**. Todo se guarda en UTC y la conversión es con
  ``zoneinfo``, nunca con un offset fijo: eso es lo que hace que las dos ventanas
  anuales de desfase entre el DST americano y el europeo salgan bien.
- El **roll del futuro ES** (cuatro veces al año) se marca como evento, para que
  no se confunda con un movimiento de mercado.

Lo que este módulo **no** decide: qué hacer con una media sesión (el
``NOTHING`` por defecto de las medias sesiones y de los días de FOMC es una
regla del *gate*, no del calendario).
"""

from __future__ import annotations

import calendar as _calendar
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, cast
from zoneinfo import ZoneInfo

import holidays
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from cfdtrader.data.settings import ConfigurationError

__all__ = [
    "DEFAULT_CALENDAR_PATH",
    "DEFAULT_FOMC_CALENDAR_PATH",
    "EASTERN",
    "HALF_SESSION_CLOSE_ET",
    "MADRID",
    "SESSION_CLOSE_ET",
    "SESSION_OPEN_ET",
    "SOURCE_MONTHS",
    "CalendarConfig",
    "FomcCalendarConfig",
    "MarketCalendar",
    "PendingYear",
    "SessionInfo",
    "fomc_dates_for",
    "load_calendar",
    "load_fomc_calendar",
    "parse_declared_meetings",
    "pending_reason",
]

#: Zona de referencia interna del proyecto: **nunca** una hora local fija.
EASTERN = ZoneInfo("America/New_York")

#: Zona de presentación (``plan.md`` §8.3). No decide nada.
MADRID = ZoneInfo("Europe/Madrid")

#: Subasta de apertura y cierre de una sesión completa.
SESSION_OPEN_ET = time(9, 30)
SESSION_CLOSE_ET = time(16, 0)

#: Cierre de una media sesión: 13:00 ET, tres horas y media de sesión.
HALF_SESSION_CLOSE_ET = time(13, 0)

DEFAULT_CALENDAR_PATH = Path(__file__).resolve().parents[3] / "config" / "calendar.yaml"

#: Duración en horas de una sesión completa y de una media sesión.
FULL_SESSION_HOURS = 6.5
HALF_SESSION_HOURS = 3.5


class CalendarConfig(BaseModel):
    """Contenido de ``config/calendar.yaml``.

    Las listas de excepciones existen para lo que ninguna regla puede deducir:
    cierres por luto nacional o medias sesiones anunciadas a última hora. Se
    declaran con fecha explícita, que es lo único auditable.
    """

    model_config = ConfigDict(extra="forbid")

    reference_timezone: str = "America/New_York"
    presentation_timezone: str = "Europe/Madrid"
    extra_holidays: tuple[date, ...] = ()
    extra_half_days: tuple[date, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """Todo lo que hay que saber de una fecha concreta."""

    day: date
    is_session: bool
    is_half_day: bool
    open_utc: datetime | None
    close_utc: datetime | None
    duration_hours: float
    reason: str | None = None

    @property
    def close_et(self) -> datetime | None:
        """Cierre en hora de referencia interna (ET)."""
        return None if self.close_utc is None else self.close_utc.astimezone(EASTERN)

    @property
    def open_et(self) -> datetime | None:
        """Apertura en hora de referencia interna (ET)."""
        return None if self.open_utc is None else self.open_utc.astimezone(EASTERN)


class MarketCalendar:
    """Calendario del mercado americano, con la hora de referencia en ET.

    Parameters
    ----------
    config:
        Excepciones declaradas en ``config/calendar.yaml``.
    years:
        Años que se materializan del calendario de festivos. Por defecto, un
        rango amplio alrededor del año en curso: ``holidays`` genera perezosamente
        y consultar un año no materializado sería una respuesta silenciosamente
        equivocada, que es justo lo que este módulo existe para evitar.
    """

    def __init__(
        self, config: CalendarConfig | None = None, *, years: tuple[int, ...] | None = None
    ) -> None:
        self._config = config or CalendarConfig()
        current = datetime.now(UTC).year
        self._years = years or tuple(range(current - 30, current + 11))
        self._holidays = _nyse_holidays(self._years)
        self._extra_holidays = frozenset(self._config.extra_holidays)
        self._extra_half_days = frozenset(self._config.extra_half_days)

    # ── Configuración ────────────────────────────────────────────────────────
    @property
    def config(self) -> CalendarConfig:
        """Configuración con la que se construyó el calendario."""
        return self._config

    # ── Sesiones ─────────────────────────────────────────────────────────────
    def is_session(self, day: date) -> bool:
        """``True`` si el mercado americano abre ese día."""
        return self.session(day).is_session

    def is_half_day(self, day: date) -> bool:
        """``True`` si es sesión y cierra a las 13:00 ET."""
        return self.session(day).is_half_day

    def close_time_et(self, day: date) -> time:
        """Hora de cierre en ET: 13:00 en media sesión, 16:00 en sesión completa."""
        return HALF_SESSION_CLOSE_ET if self.is_half_day(day) else SESSION_CLOSE_ET

    def session(self, day: date) -> SessionInfo:
        """Información completa de una fecha, sea sesión o no."""
        holiday_name = self._holiday_name(day)
        if holiday_name is not None:
            return SessionInfo(
                day=day,
                is_session=False,
                is_half_day=False,
                open_utc=None,
                close_utc=None,
                duration_hours=0.0,
                reason=f"festivo: {holiday_name}",
            )
        if day.weekday() >= 5:
            return SessionInfo(
                day=day,
                is_session=False,
                is_half_day=False,
                open_utc=None,
                close_utc=None,
                duration_hours=0.0,
                reason="fin de semana",
            )

        half = self._is_half_day(day)
        reason = "media sesión: cierre a las 13:00 ET" if half else None
        return SessionInfo(
            day=day,
            is_session=True,
            is_half_day=half,
            open_utc=self.to_utc(day, SESSION_OPEN_ET),
            close_utc=self.to_utc(day, HALF_SESSION_CLOSE_ET if half else SESSION_CLOSE_ET),
            duration_hours=HALF_SESSION_HOURS if half else FULL_SESSION_HOURS,
            reason=reason,
        )

    def sessions(self, start: date, end: date) -> list[SessionInfo]:
        """Sesiones entre dos fechas, ambas incluidas."""
        if end < start:
            raise ValueError("'end' no puede ser anterior a 'start'")
        found: list[SessionInfo] = []
        cursor = start
        while cursor <= end:
            info = self.session(cursor)
            if info.is_session:
                found.append(info)
            cursor += timedelta(days=1)
        return found

    def previous_session(self, day: date) -> date:
        """Sesión anterior a esa fecha (la fecha anterior más cercana que abre)."""
        cursor = day - timedelta(days=1)
        for _ in range(400):
            if self.is_session(cursor):
                return cursor
            cursor -= timedelta(days=1)
        raise ValueError(f"no se encontró sesión anterior a {day.isoformat()}")  # pragma: no cover

    def next_session(self, day: date) -> date:
        """Sesión siguiente a esa fecha."""
        cursor = day + timedelta(days=1)
        for _ in range(400):
            if self.is_session(cursor):
                return cursor
            cursor += timedelta(days=1)
        raise ValueError(f"no se encontró sesión siguiente a {day.isoformat()}")  # pragma: no cover

    # ── Instantes UTC ────────────────────────────────────────────────────────
    def open_utc(self, day: date) -> datetime:
        """Apertura de la subasta (09:30 ET) en UTC."""
        return self.to_utc(day, SESSION_OPEN_ET)

    def close_utc(self, day: date) -> datetime:
        """Cierre de sesión en UTC: 13:00 ET en media sesión, 16:00 ET si no."""
        return self.to_utc(day, self.close_time_et(day))

    def bounds_utc(self, day: date) -> tuple[datetime, datetime]:
        """Apertura y cierre de esa sesión, en UTC."""
        info = self.session(day)
        if not info.is_session or info.open_utc is None or info.close_utc is None:
            raise ValueError(f"{day.isoformat()} no es sesión: {info.reason}")
        return info.open_utc, info.close_utc

    @staticmethod
    def to_utc(day: date, at_et: time) -> datetime:
        """Hora ET de ese día en UTC. Con ``zoneinfo``, nunca con un offset fijo."""
        return datetime.combine(day, at_et, tzinfo=EASTERN).astimezone(UTC)

    @staticmethod
    def to_et(instant: datetime) -> datetime:
        """El mismo instante en la hora de referencia interna."""
        return instant.astimezone(EASTERN)

    @staticmethod
    def to_madrid(instant: datetime) -> datetime:
        """El mismo instante en hora de Madrid, **solo para presentar**."""
        return instant.astimezone(MADRID)

    def madrid_bounds(self, day: date) -> tuple[datetime, datetime]:
        """Apertura y cierre de la sesión en hora de Madrid (presentación)."""
        open_utc, close_utc = self.bounds_utc(day)
        return self.to_madrid(open_utc), self.to_madrid(close_utc)

    def session_offset_hours(self, day: date) -> int:
        """Horas de diferencia con Madrid ese día: 6 h normalmente, 5 h en las
        ventanas de desfase."""
        instant = self.open_utc(day)
        et_offset = instant.astimezone(EASTERN).utcoffset() or timedelta()
        madrid_offset = instant.astimezone(MADRID).utcoffset() or timedelta()
        return int((madrid_offset - et_offset).total_seconds() // 3600)

    # ── DST ──────────────────────────────────────────────────────────────────
    def us_dst_transitions(self, year: int) -> tuple[date, date]:
        """EE. UU.: entra en horario de verano el 2.º domingo de marzo y sale el 1.º
        de noviembre."""
        return _nth_weekday(year, 3, 6, 2), _nth_weekday(year, 11, 6, 1)

    def eu_dst_transitions(self, year: int) -> tuple[date, date]:
        """Europa: entra el último domingo de marzo y sale el último domingo de octubre."""
        return _last_weekday(year, 3, 6), _last_weekday(year, 10, 6)

    def dst_mismatch_windows(self, year: int) -> tuple[tuple[date, date], ...]:
        """Las **dos ventanas anuales de desfase** entre el DST americano y el europeo.

        - En marzo, EE. UU. se adelanta antes que Europa.
        - A finales de octubre, Europa se retrasa antes que EE. UU.

        En esas ventanas el mercado abre a las **14:30** de Madrid en vez de a las
        15:30. Cualquier hora fija escrita a mano queda mal esos días, y por eso
        el proyecto trabaja siempre en ``America/New_York`` y usa Madrid solo
        para presentar. No hay scheduler: el anclaje lo garantiza este módulo,
        no el sistema operativo (``_docs/tech_stack.md`` §4.11).
        """
        us_spring, us_fall = self.us_dst_transitions(year)
        eu_spring, eu_fall = self.eu_dst_transitions(year)
        windows: list[tuple[date, date]] = []
        if us_spring <= eu_spring:
            windows.append((us_spring, eu_spring))
        if eu_fall <= us_fall:
            windows.append((eu_fall, us_fall))
        return tuple(windows)

    def dst_mismatch_days(self, year: int) -> tuple[date, ...]:
        """Todos los días de las dos ventanas de desfase, sin repetir y en orden."""
        days: list[date] = []
        for start, end in self.dst_mismatch_windows(year):
            cursor = start
            while cursor < end:
                if cursor not in days:
                    days.append(cursor)
                cursor += timedelta(days=1)
        return tuple(sorted(days))

    def has_dst_mismatch(self, day: date) -> bool:
        """``True`` si ese día cae en una de las dos ventanas de desfase."""
        return day in self.dst_mismatch_days(day.year)

    # ── Roll del futuro ES ───────────────────────────────────────────────────
    def es_roll_dates(self, year: int) -> tuple[date, ...]:
        """Los cuatro *rolls* trimestrales del futuro ES (tercer viernes de mar/jun/sep/dic).

        Se marcan como evento para que un cambio de contrato no se confunda con
        un movimiento de mercado (``plan.md`` §8.3).
        """
        return tuple(_nth_weekday(year, month, 4, 3) for month in (3, 6, 9, 12))

    def is_es_roll(self, day: date) -> bool:
        """``True`` si ese día vence el futuro ES (roll trimestral)."""
        return day in self.es_roll_dates(day.year)

    # ── Internos ─────────────────────────────────────────────────────────────
    def _holiday_name(self, day: date) -> str | None:
        if day in self._extra_holidays:
            return "declarado en config/calendar.yaml"
        return _lookup_holiday(self._years, day)

    def _is_half_day(self, day: date) -> bool:
        if day in self._extra_half_days:
            return True
        if day.weekday() >= 5:
            return False
        # Día después de Acción de Gracias (4.º jueves de noviembre).
        thanksgiving = _nth_weekday(day.year, 11, 3, 4)
        if day == thanksgiving + timedelta(days=1):
            return True
        # Víspera de Navidad, si no es festivo ni fin de semana.
        if (day.month, day.day) == (12, 24):
            return True
        # Víspera del Día de la Independencia, si no es festivo ni fin de semana.
        return (day.month, day.day) == (7, 3)


@lru_cache(maxsize=8)
def _lookup_holiday(years: tuple[int, ...], day: date) -> str | None:
    """Nombre del festivo NYSE de ese día, o ``None`` si no lo es."""
    return _nyse_holidays(years).get(day)


@lru_cache(maxsize=4)
def _nyse_holidays(years: tuple[int, ...]) -> Any:
    """Calendario de festivos NYSE materializado.

    ``holidays`` no publica anotaciones de tipo completas, así que el límite de
    la librería se aísla aquí: el resto del módulo trabaja con ``date`` y ``str``.
    """
    return holidays.NYSE(years=list(years))  # pyright: ignore


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """El n-ésimo ``weekday`` (0=lunes) del mes."""
    if n < 1:
        raise ValueError("'n' empieza en 1")
    occurrences = [
        day
        for day in _calendar.Calendar().itermonthdates(year, month)
        if day.month == month and day.weekday() == weekday
    ]
    return occurrences[n - 1]


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """El último ``weekday`` del mes."""
    occurrences = [
        day
        for day in _calendar.Calendar().itermonthdates(year, month)
        if day.month == month and day.weekday() == weekday
    ]
    return occurrences[-1]


def load_calendar(
    path: Path | str | None = None, *, years: tuple[int, ...] | None = None
) -> MarketCalendar:
    """Carga ``config/calendar.yaml`` y devuelve el calendario listo para usar.

    Un fichero mal formado falla **al arrancar**, con el motivo, y no a mitad del
    pipeline (``tech_stack.md`` §4.2).
    """
    target = Path(path) if path is not None else DEFAULT_CALENDAR_PATH
    config = CalendarConfig()
    if target.is_file():
        try:
            loaded: object = yaml.safe_load(target.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ConfigurationError(f"YAML inválido en {target}: {error}") from error
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ConfigurationError(f"{target} debe contener un mapping en la raíz")
            try:
                config = CalendarConfig.model_validate(cast("dict[str, object]", loaded))
            except ValidationError as error:
                raise ConfigurationError(f"calendario inválido en {target}: {error}") from error
    return MarketCalendar(config, years=years)


# ─────────────────────────────────────────────────────────────────────────────
# El calendario de FOMC declarado (parte (a) de #114)
# ─────────────────────────────────────────────────────────────────────────────
#: Ruta por defecto del calendario de reuniones del FOMC declarado.
DEFAULT_FOMC_CALENDAR_PATH = Path(__file__).resolve().parents[3] / "config" / "fomc_calendar.yaml"


class PendingYear(BaseModel):
    """Un año que se **intentó** recuperar y no se pudo declarar (tarea #132).

    Existe para que el hueco sea **dato**, no un comentario: un lector —o el aviso del camino
    diario— puede nombrarlo, y la fecha del intento queda registrada. Declarar aquí una fecha
    inventada sería justo lo que el artefacto existe para impedir.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str = Field(min_length=1, description="por que el año no esta declarado")
    attempted_on: date = Field(description="fecha del ultimo intento de recuperacion")


class FomcCalendarConfig(BaseModel):
    """Contenido de ``config/fomc_calendar.yaml``.

    A diferencia de una noticia, este calendario se **publica con antelacion**: usarlo para
    bloquear un dia futuro **no es look-ahead**. Lo que hay que declarar es la procedencia y la
    fecha en que se consulto, porque la propia Fed avisa de que las fechas son **provisionales**
    hasta la reunion inmediatamente anterior.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(ge=1, description="version del esquema del fichero")
    source: str = Field(min_length=1, description="URL de procedencia del calendario")
    verified_on: date = Field(description="fecha en que se consulto la fuente")
    tentative_note: str = Field(min_length=1, description="aviso de la fuente sobre las fechas")
    #: Días de reunión por año; un año declarado lleva al menos una fecha.
    meetings: dict[int, tuple[date, ...]] = {}
    #: Años cuyo calendario **no** se ha podido declarar, con su motivo y la fecha del intento.
    pending: dict[int, PendingYear] = {}

    @field_validator("meetings")
    @classmethod
    def _reject_empty_years(cls, value: dict[int, tuple[date, ...]]) -> dict[int, tuple[date, ...]]:
        """Un año declarado con la lista vacia seria «declarado sin reuniones»: no existe.

        Es la distincion que sostiene toda la tarea: si no se sabe, **no se declara el año**, y el
        camino diario lo avisa. Declararlo vacio seria decir que se sabe y que no hay ninguna.
        """
        empty = sorted(year for year, days in value.items() if not days)
        if empty:
            raise ValueError(
                f"los años {empty} estan declarados sin ninguna reunion: si no se sabe, no se "
                "declara el año en vez de declararlo vacio"
            )
        return value

    @model_validator(mode="after")
    def _reject_a_year_that_is_declared_and_pending_at_once(self) -> FomcCalendarConfig:
        """Un año no puede estar a la vez declarado y pendiente: sería una contradicción silenciosa.

        Si `meetings` trae sus fechas, ya no hay nada pendiente; y si está pendiente, no hay fechas.
        Dejar las dos cosas convivir haría que el aviso del camino diario mintiera en una dirección
        u otra según cuál se leyera primero.
        """
        both = sorted(set(self.meetings) & set(self.pending))
        if both:
            raise ValueError(
                f"los años {both} estan declarados en `meetings` y ademas como pendientes: son "
                "excluyentes"
            )
        return self


def load_fomc_calendar(path: Path | str | None = None) -> FomcCalendarConfig | None:
    """Carga ``config/fomc_calendar.yaml``.

    Devuelve ``None`` cuando el artefacto **no existe**: entonces el calendario no esta declarado,
    que es informacion **distinta** de «hoy no hay FOMC», y el llamante tiene que poder
    distinguirla. Un fichero mal formado falla al arrancar, con el motivo (``tech_stack.md`` §4.2).
    """
    target = Path(path) if path is not None else DEFAULT_FOMC_CALENDAR_PATH
    if not target.is_file():
        return None
    try:
        loaded: object = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"YAML inválido en {target}: {error}") from error
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"{target} debe contener un mapping en la raíz")
    try:
        return FomcCalendarConfig.model_validate(cast("dict[str, object]", loaded))
    except ValidationError as error:
        raise ConfigurationError(f"calendario de FOMC inválido en {target}: {error}") from error


def fomc_dates_for(config: FomcCalendarConfig, year: int) -> tuple[date, ...] | None:
    """Los dias de FOMC declarados para ese año, o ``None`` si el **año no esta declarado**.

    ``None`` **no** es una lista vacia: es «nadie ha declarado el calendario de ese año». Confundir
    las dos cosas es exactamente lo que hacia el literal ``fomc_dates=()`` del camino diario, que
    hacia indistinguible «hoy no hay FOMC» de «la regla 17 no puede dispararse» (#114).
    """
    return config.meetings.get(year)


#: Los doce meses tal y como los publica la fuente (en orden), para leer su calendario.
SOURCE_MONTHS: Final[tuple[str, ...]] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

#: Una reunion: `Mes D-D` o `Mes D - Mes D` (a caballo entre dos meses, p. ej. `October 31 -
#: November 1`). **El rango es obligatorio a proposito**: exigirlo evita confundir la fecha de
#: publicacion de las actas —`(Released February 18, 2026)`— con un dia de reunion.
_MEETING_RANGE: Final[re.Pattern[str]] = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2})\s*[-\u2013]\s*(?:(" + "|".join(SOURCE_MONTHS) + r")\s+)?(\d{1,2})"
)


def parse_declared_meetings(text: str, year: int) -> tuple[date, ...]:
    """Los dias de reunion de ``year`` **leidos del texto** de la fuente (tarea #132).

    Funcion **pura**: recibe el texto y devuelve las fechas; no abre red, ni ficheros, ni el reloj.
    Existe porque leer esa pagina tiene una trampa **medida** —le paso a #124 con 2026 y volvio a
    pasar en #132—: los lectores que la sirven troceada **truncan el centro** y la seccion del año
    buscado cae justo en el hueco. La receta que si funciona es descargarla **entera** y parsearla
    aqui, con `urllib` del sistema (sin lector de por medio):

    .. code-block:: bash

        uv run python -c "import urllib.request as u; print(u.urlopen(u.Request('https://www.
        federalreserve.gov/monetarypolicy/fomccalendars.htm',
        headers={'User-Agent': 'cfdtrader'})).read().decode('utf-8', 'replace'))" > /tmp/fomc.html

    Una lista vacia significa «ese año **no aparece** en el texto», nunca «ese año no tiene
    reuniones»: esa distincion es la que sostiene el aviso del camino diario.

    **Alcance, y es un limite real:** este parser lee el maquetado de los años **anunciados**
    (`Mes D-D`, con los dos dias en el mismo bloque), que es donde vive el calendario
    **prospectivo** —lo unico que se declara: 2026, 2027 y los que vengan—. Los años **historicos**
    de esa misma pagina usan otro maquetado (mes y dias en celdas separadas, y reuniones a caballo
    entre meses como `Oct/Nov 31-1`) y **no** quedan cubiertos: no se declaran, asi que no hacen
    falta. Si algun dia hay que leerlos, es otra tarea y con su propia prueba.
    """
    start = text.find(f"{year} FOMC Meetings")
    if start < 0:
        return ()
    block = text[start + len(f"{year} FOMC Meetings") :]
    following = re.search(r"\b\d{4} FOMC Meetings\b", block)
    # La fuente cierra cada año con una **nota** que puede nombrar la reunion del año **siguiente**
    # («Note: A two-day meeting is scheduled for January 25-26, 2028»): sin cortarla, esa fecha se
    # colaria como si fuera de este año. Se corta por el primero de los dos finales.
    note = block.find("Note:")
    stops = [
        index
        for index in (following.start() if following is not None else None, note)
        if index is not None and index >= 0
    ]
    if stops:
        block = block[: min(stops)]
    plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", block))

    days: list[date] = []
    for month_name, first, next_month, second in _MEETING_RANGE.findall(plain):
        month = SOURCE_MONTHS.index(month_name) + 1
        days.append(date(year, month, int(first)))
        month_2 = SOURCE_MONTHS.index(next_month) + 1 if next_month else month
        days.append(date(year + 1 if month_2 < month else year, month_2, int(second)))
    return tuple(sorted(set(days)))


def pending_reason(config: FomcCalendarConfig, year: int) -> str | None:
    """Por que ese año **no** se ha podido declarar, o ``None`` si no esta declarado como pendiente.

    Es la respuesta a «el año no esta declarado, ¿y por que?» (tarea #132): sin esto, el aviso del
    camino diario dice que falta el calendario y deja al lector sin saber si nadie lo ha intentado
    o si se intento y la fuente no lo dio.
    """
    pending = config.pending.get(year)
    return None if pending is None else pending.reason
