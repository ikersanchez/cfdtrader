"""Normalización temporal y de esquema compartida por los adaptadores.

La regla de diseño (`plan.md` §8.3) es que **la hora de referencia interna es
``America/New_York``** y todo se guarda en UTC. Este módulo es el único sitio
donde se hace esa conversión, para que no haya dos versiones ligeramente
distintas de la misma regla en cada adaptador.

Nada de offsets fijos: la conversión usa ``zoneinfo``, así que las dos ventanas
anuales de desfase entre el DST americano y el europeo salen bien por
construcción (la tarea #4 construye el calendario de sesiones encima de esto).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

import polars as pl

from cfdtrader.data.sources.base import CANONICAL_COLUMNS

__all__ = [
    "CANONICAL_DTYPES",
    "EASTERN",
    "MADRID",
    "empty_canonical_frame",
    "session_close_utc",
    "session_open_utc",
    "to_utc",
]

#: Zona de referencia interna del proyecto. Nunca una hora local fija.
EASTERN = ZoneInfo("America/New_York")

#: Zona **solo de presentación** (``plan.md`` §8.3).
MADRID = ZoneInfo("Europe/Madrid")

#: Hora de cierre de una sesión completa en ET, y de apertura (subasta).
SESSION_OPEN_ET = time(9, 30)
SESSION_CLOSE_ET = time(16, 0)

#: Esquema canónico del frame que devuelve un adaptador.
CANONICAL_DTYPES: dict[str, pl.DataType] = {
    "as_of": pl.Datetime("us", "UTC"),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "adj_close": pl.Float64(),
    "bid": pl.Float64(),
    "ask": pl.Float64(),
}


def empty_canonical_frame() -> pl.DataFrame:
    """Frame canónico sin filas, con el esquema correcto.

    Un resultado no-``ok`` **no puede traer filas** (no se inventan datos), así
    que el frame que lo acompaña es siempre este.
    """
    return pl.DataFrame(schema={name: CANONICAL_DTYPES[name] for name in CANONICAL_COLUMNS})


def to_utc(value: object) -> datetime | None:
    """Convierte una marca temporal de fuente a UTC, o ``None`` si no es válida.

    - Un ``datetime`` con zona se respeta.
    - Un ``datetime`` sin zona se interpreta como hora ET: es la convención de
      la fuente, no una invención nuestra.
    - Una ``date`` se interpreta como las 00:00 ET de ese día (la usa el que
      necesita el instante, no el cierre de sesión).
    """
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=EASTERN).astimezone(UTC)
        return value.astimezone(UTC)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=EASTERN).astimezone(UTC)
    return None


def session_close_utc(day: date) -> datetime:
    """Cierre de sesión de ``day`` en UTC (16:00 ET).

    Una barra diaria se ancla al cierre de sesión, **no** a medianoche (A9): el
    ``2024-03-15`` (EDT) es ``2024-03-15 20:00 UTC`` y el ``2024-01-15`` (EST)
    es ``2024-01-15 21:00 UTC``.

    Las medias sesiones (cierre a las 13:00 ET) las conoce el calendario de la
    tarea #4; #3 no las adivina, y una sesión que aún no ha cerrado se descarta
    en lugar de escribirse a medias (A11).
    """
    return datetime.combine(day, SESSION_CLOSE_ET, tzinfo=EASTERN).astimezone(UTC)


def session_open_utc(day: date) -> datetime:
    """Apertura de la subasta de ``day`` en UTC (09:30 ET)."""
    return datetime.combine(day, SESSION_OPEN_ET, tzinfo=EASTERN).astimezone(UTC)
