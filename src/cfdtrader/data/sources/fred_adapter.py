"""Adaptador de FRED/ALFRED — fuente macro primaria (tarea #5).

``tech_stack.md`` §4.5 hace de FRED la **fuente primaria del proyecto** para
tipos, inflación y empleo. Se consume con ``httpx`` (cliente inyectado, con
reintentos y caché de ``sources/http.py``) y no con ``fredapi``: una dependencia
menos y el control del *point-in-time* en nuestras manos.

**Point-in-time (lo único que de verdad importa aquí).** Un dato macro no existe
hasta que se publica, y ese instante no es el día al que se refiere:

- ``as_of`` es el **periodo observado** (el día del fed funds, el mes del CPI).
- ``published_at`` es el **instante de publicación** en UTC, calculado con la
  hora oficial de publicación en ET y la fecha de la *vintage* que devuelve
  ALFRED (``realtime_start``):

  · series de calendario de publicación (BLS/BEA: CPI, PCE, NFP) ⇒
    ``realtime_start`` a la hora declarada, típicamente **08:30 ET**;
  · series determinadas por el mercado (Treasury, fed funds efectivo) ⇒
    ``as_of`` más un desplazamiento declarado, a la hora declarada, porque no
    hay una rueda de prensa: el valor se conoce cuando el mercado cierra.

  Cuando no se puede determinar ese instante, ``published_at`` queda **NULL** (y
  el almacén cae a ``fetched_at``). Nunca se rellena con ``fetched_at`` haciendo
  pasar la descarga por publicación: eso destruiría la garantía point-in-time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, cast

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from cfdtrader.data.sources.base import (
    SourceBlockedError,
    SourceError,
    SourceRateLimitedError,
    SourceStatus,
    SourceUnavailableError,
)
from cfdtrader.data.sources.frames import EASTERN
from cfdtrader.data.sources.http import CachedHttpClient

__all__ = [
    "FRED_OBSERVATIONS_URL",
    "FredAdapter",
    "MacroFetchResult",
    "MacroSeriesRegistry",
    "MacroSeriesSpec",
]

#: Endpoint de observaciones. ALFRED (los *vintages*) es el mismo endpoint con
#: ``realtime_start``/``realtime_end``: por eso la selección de *vintage* está
#: soportada de serie y no como un caso aparte.
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

#: Valor con el que FRED marca un dato ausente. No es un 0 ni un nulo: es «no hay dato».
FRED_MISSING = "."

#: Extremos de la ventana de *vintages*: desde el principio de los datos hasta hoy.
FIRST_VINTAGE = date(1900, 1, 1)
LAST_VINTAGE = "9999-12-31"


class MacroSeriesSpec(BaseModel):
    """Declaración de una serie macro: qué es, cuándo se publica y qué se exige de ella."""

    model_config = ConfigDict(extra="forbid")

    series_id: str
    name: str
    dataset: str = "macro"
    unit: str = ""
    seasonal_adjustment: str | None = None
    #: Hora oficial de publicación en ``America/New_York`` (08:30 en BLS/BEA).
    release_time_et: time = time(8, 30)
    #: ``True`` si la fecha de publicación la da la *vintage* de ALFRED
    #: (BLS/BEA). ``False`` si el valor se conoce el día al que se refiere
    #: (series de mercado), con el desplazamiento de abajo.
    publication_from_realtime_start: bool = False
    #: Días naturales que se suman a ``as_of`` cuando la publicación no viene de
    #: la *vintage* (el fed funds efectivo de un día se publica al siguiente).
    publication_offset_days: int = 0
    min_start: date | None = None
    notes: str | None = None

    @property
    def release_hour_et(self) -> time:
        """Hora de publicación en ET, tal y como se declaró."""
        return self.release_time_et


class MacroSeriesRegistry(BaseModel):
    """Contenido de ``config/macro_series.yaml``."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    dataset: str = "macro"
    series: tuple[MacroSeriesSpec, ...] = ()
    #: Contexto europeo opcional (ECB SDW, Eurostat): no implementado y declarado
    #: como tal, para que no parezca olvidado (``tech_stack.md`` §4.5).
    european_context: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class MacroFetchResult:
    """Resultado de intentar una serie macro, con estado declarado.

    Frame: ``as_of`` (``Date``), ``value`` (``Float64``) y ``published_at``
    (``Datetime`` UTC, puede ser nulo). Un estado que no sea ``ok`` **nunca**
    trae filas.
    """

    spec: MacroSeriesSpec
    source: str
    status: SourceStatus
    frame: pl.DataFrame
    attempts: int = 1
    notes: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        """``True`` solo si el estado es ``ok``."""
        return self.status is SourceStatus.OK

    def __post_init__(self) -> None:
        if self.status is SourceStatus.OK and self.error is not None:
            raise ValueError("un resultado 'ok' no puede llevar 'error'")
        if self.status is not SourceStatus.OK and self.frame.height:
            raise ValueError("un resultado no-'ok' no puede traer filas: no se inventan datos")


class FredAdapter:
    """Cliente de FRED con el cliente HTTP **inyectado**.

    ``api_key=None`` no es un fallo del adaptador: es la declaración de que no
    hay clave. Cada serie termina con estado ``unavailable`` y su motivo, en vez
    de intentar una llamada que FRED rechazará sin decir por qué.
    """

    name = "fred"

    def __init__(self, client: CachedHttpClient, *, api_key: str | None) -> None:
        self._client = client
        self._api_key = api_key

    def fetch(
        self, spec: MacroSeriesSpec, *, now: datetime, realtime: date | None = None
    ) -> MacroFetchResult:
        """Descarga una serie y la normaliza a ``(as_of, value, published_at)``.

        Parameters
        ----------
        spec:
            Serie declarada en ``config/macro_series.yaml``.
        now:
            Instante de referencia. Se pasa desde fuera: el adaptador no lee el reloj.
        realtime:
            Fecha de la *vintage* a pedir (point-in-time estricto en ALFRED). Si
            es ``None`` se pide la última disponible.
        """
        if not self._api_key:
            return _failed(
                spec,
                SourceStatus.UNAVAILABLE,
                "falta FRED_API_KEY: sin clave no hay serie macro (ver .env.example)",
            )

        params: dict[str, str | int] = {
            "series_id": spec.series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": "asc",
        }
        if spec.min_start is not None:
            params["observation_start"] = spec.min_start.isoformat()
        if realtime is not None:
            params["realtime_start"] = realtime.isoformat()
            params["realtime_end"] = realtime.isoformat()
        elif spec.publication_from_realtime_start:
            # ⚠️ Sin esto, FRED devuelve la **última** vintage y cada observación
            # trae `realtime_start` = el día de la consulta: las 260 observaciones
            # de CPI desde 2005 quedarían publicadas «hoy» y el *point-in-time*
            # sería decorativo. Pidiendo todas las vintages, `realtime_start` de
            # cada observación es la fecha real de su primer comunicado
            # (comprobado contra la API el 2026-09-17: 1.367 filas y 0,1 MB para
            # CPIAUCSL desde 2005, con el primer dato de 2005-01-01 publicado el
            # 2005-02-23).
            params["realtime_start"] = (spec.min_start or FIRST_VINTAGE).isoformat()
            params["realtime_end"] = LAST_VINTAGE

        try:
            cached = self._client.get(FRED_OBSERVATIONS_URL, params=params, now=now)
        except SourceError as error:
            return _failed(spec, _status_for(error), str(error), attempts=error.attempts)

        payload = _as_json(cached.response.json())
        if payload is None:
            return _failed(spec, SourceStatus.BLOCKED, "FRED no devolvió un objeto JSON")

        fred_error = payload.get("error_message")
        if fred_error is not None:
            message = f"FRED rechazó la petición: {fred_error}"
            status = (
                SourceStatus.RATE_LIMITED
                if "rate" in str(fred_error).lower()
                else SourceStatus.ERROR
            )
            return _failed(spec, status, message, attempts=cached.attempts)

        frame, vintages = _normalize(payload, spec=spec)
        if frame.height == 0:
            return _failed(
                spec, SourceStatus.UNAVAILABLE, f"{spec.series_id}: FRED no trae observaciones"
            )

        notes = [f"cache={'sí' if cached.from_cache else 'no'}"]
        if spec.publication_from_realtime_start:
            notes.append(
                f"vintages desde {params['realtime_start']}: {vintages} filas leídas, "
                f"se guarda la primera publicación de cada observación"
            )
        vintage = payload.get("realtime_start")
        if vintage is not None:
            notes.append(f"vintage={vintage}")
        return MacroFetchResult(
            spec=spec,
            source=self.name,
            status=SourceStatus.OK,
            frame=frame,
            attempts=cached.attempts,
            notes=tuple(notes),
        )


def _failed(
    spec: MacroSeriesSpec,
    status: SourceStatus,
    message: str,
    *,
    attempts: int = 1,
) -> MacroFetchResult:
    """Resultado fallido con su motivo y el frame declarado vacío."""
    return MacroFetchResult(
        spec=spec,
        source=FredAdapter.name,
        status=status,
        frame=_empty_frame(),
        attempts=attempts,
        error=message,
    )


def _status_for(error: SourceError) -> SourceStatus:
    """Estado declarado que corresponde al error del cliente HTTP."""
    if isinstance(error, SourceRateLimitedError):
        return SourceStatus.RATE_LIMITED
    if isinstance(error, SourceBlockedError):
        return SourceStatus.BLOCKED
    if isinstance(error, SourceUnavailableError):
        return SourceStatus.UNAVAILABLE
    return SourceStatus.ERROR


def _empty_frame() -> pl.DataFrame:
    """Frame macro sin filas, con el esquema correcto."""
    return pl.DataFrame(
        schema={
            "as_of": pl.Date(),
            "value": pl.Float64(),
            "published_at": pl.Datetime("us", "UTC"),
        }
    )


def _as_json(payload: object) -> dict[str, Any] | None:
    """Payload como mapping tipado, o ``None`` si no lo es."""
    if not isinstance(payload, dict):
        return None
    return cast("dict[str, Any]", payload)


def _normalize(payload: dict[str, Any], *, spec: MacroSeriesSpec) -> tuple[pl.DataFrame, int]:
    """Observaciones de FRED → ``(as_of, value, published_at)`` en UTC.

    Cuando la respuesta trae **varias vintages** de la misma observación (series
    del BLS/BEA), se guarda la **primera**: el valor que movió el mercado es el
    primer publicado, y su instante de publicación es el del primer comunicado.
    Las revisiones posteriores son otro dato y no se mezclan aquí (si algún día
    hacen falta, se piden por vintage y se escriben como revisión).

    Returns
    -------
    tuple[pl.DataFrame, int]
        El frame normalizado y cuántas filas de vintage se leyeron.
    """
    observations = payload.get("observations")
    if not isinstance(observations, list):
        return _empty_frame(), 0

    # Fecha observada → (primera vintage con valor, valor de esa vintage).
    first: dict[date, tuple[str, float | None]] = {}
    vintages = 0
    for item in cast("list[object]", observations):
        if not isinstance(item, dict):
            continue
        observation = cast("dict[object, object]", item)
        day = _as_date(observation.get("date"))
        if day is None:
            continue
        vintages += 1
        value = _as_number(observation.get("value"))
        realtime = observation.get("realtime_start")
        release = realtime if isinstance(realtime, str) else ""
        previous = first.get(day)
        # Se prefiere la vintage más antigua **con valor**: una vintage con `.`
        # no es una publicación, es un hueco.
        if (
            previous is None
            or (previous[1] is None and value is not None)
            or (value is not None and previous[1] is not None and release and release < previous[0])
        ):
            first[day] = (release, value)

    rows: list[dict[str, object]] = []
    for day, (release, value) in sorted(first.items()):
        rows.append(
            {
                "as_of": day,
                # NULL = no hay dato. No se interpola ni se arrastra el valor anterior.
                "value": value,
                "published_at": _published_at(spec, day=day, realtime_start=release),
            }
        )
    if not rows:
        return _empty_frame(), vintages
    return (
        pl.DataFrame(
            rows,
            schema_overrides={
                "as_of": pl.Date(),
                "value": pl.Float64(),
                "published_at": pl.Datetime("us", "UTC"),
            },
        ).sort("as_of"),
        vintages,
    )


def _published_at(spec: MacroSeriesSpec, *, day: date, realtime_start: object) -> datetime | None:
    """Instante de publicación en UTC, o ``None`` si no se puede determinar.

    Es la pieza que hace que el *point-in-time* sea real: una serie de las 08:30
    ET **no** es visible en el snapshot de las 08:45 ET del día anterior.
    """
    if spec.publication_from_realtime_start:
        release_day = _as_date(realtime_start)
        if release_day is None:
            return None
        return _et_to_utc(release_day, spec.release_time_et)
    return _et_to_utc(day + timedelta(days=spec.publication_offset_days), spec.release_time_et)


def _et_to_utc(day: date, at_et: time) -> datetime:
    """Hora ET de ese día en UTC, con ``zoneinfo`` y **nunca** con un offset fijo."""
    return datetime.combine(day, at_et, tzinfo=EASTERN).astimezone(UTC)


def _as_date(value: object) -> date | None:
    """Fecha de un campo de FRED (``AAAA-MM-DD``)."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _as_number(value: object) -> float | None:
    """Valor numérico, o ``None`` para el marcador de dato ausente de FRED."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return None if number != number else number
    if isinstance(value, str):
        text = value.strip()
        if not text or text == FRED_MISSING:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return None if number != number else number
    return None
