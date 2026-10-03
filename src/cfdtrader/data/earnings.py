"""Resultados de las mega-caps: ingesta a ``raw.earnings`` y lectura para el diario — #126.

Es la **parte (c)** de la división de #114, la más delicada de las tres. ``plan.md`` §7.1 dice
que una sola mega-cap publicando puede mover el índice más que cualquier dato macro; §17
prohíbe inventar fechas; y el gate **no puede bloquear por una conjetura**. De ahí la regla que
gobierna todo el módulo: **una fecha estimada no bloquea**.

El corte es el de #30 para las noticias:

- **Ingesta** (``ingest``/``main``): un adaptador sobre ``yfinance`` (``YFinanceEarningsAdapter``)
  obtiene las fechas por emisor, se fusionan con las **confirmadas** que declara
  ``config/mega_caps.yaml`` y se escriben en ``raw.earnings`` como *observaciones*: ``as_of`` es
  el instante de la observación y la **fecha del evento** viaja en el payload (una fecha futura
  no puede ser el ``as_of`` del almacén, que exige ``fetched_at >= as_of``).
- **Lectura** (``earnings_on``): el camino diario lee el almacén, nunca llama a ``yfinance``.
  Aplica el mismo *point-in-time* que las noticias: nada observado **después** del ``as_of``
  entra.

**Certeza.** Cada evento declara ``certainty`` (``confirmed`` | ``estimated``) y su
``observed_at``. El valor por defecto de una fecha sin confirmar es **``estimated``**, nunca
``confirmed``; una estimada **no bloquea** (no entra en ``blocking_events`` ni cambia la
dirección ni el ``tier``). Una confirmada sí puede bloquear.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol, cast

import duckdb
import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cfdtrader.data.settings import REPO_ROOT, ConfigurationError
from cfdtrader.data.store import Store, UnknownDatasetError

__all__ = [
    "DATASET",
    "DEFAULT_MEGA_CAPS_PATH",
    "EarningsCertainty",
    "EarningsEvent",
    "EarningsMoment",
    "EarningsReport",
    "EarningsSource",
    "MegaCapIssuer",
    "MegaCapsConfig",
    "RawEarnings",
    "earnings_on",
    "ingest",
    "load_mega_caps",
    "main",
]

#: Dataset del almacén (la capa ``raw`` lo convierte en la vista ``raw.earnings``).
DATASET: Final[str] = "earnings"

#: ``<repo>/config/mega_caps.yaml`` — la lista declarada de emisores.
DEFAULT_MEGA_CAPS_PATH: Final[Path] = REPO_ROOT / "config" / "mega_caps.yaml"


class EarningsCertainty(StrEnum):
    """Si la fecha del resultado está confirmada por la compañía o es una estimación."""

    CONFIRMED = "confirmed"
    ESTIMATED = "estimated"


class EarningsMoment(StrEnum):
    """Momento del día en que publica: antes de apertura, tras el cierre, o sin declarar."""

    BMO = "bmo"
    AMC = "amc"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RawEarnings:
    """Una fecha de resultados tal y como la devuelve una fuente, sin interpretar."""

    symbol: str
    on: date
    moment: EarningsMoment
    observed_at: datetime


class EarningsSource(Protocol):
    """Lo que el ingestor necesita de una fuente de resultados (``yfinance`` la implementa)."""

    name: str

    def earnings_dates(self, symbol: str, *, now: datetime) -> tuple[RawEarnings, ...]:
        """Las fechas de resultados conocidas para ``symbol`` en el instante ``now``."""
        ...


class ConfirmedEarnings(BaseModel):
    """Una fecha de resultados **confirmada** por el propietario, con su momento."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    date: date
    moment: EarningsMoment = EarningsMoment.UNKNOWN


class MegaCapIssuer(BaseModel):
    """Un emisor declarado: su ticker de Yahoo, su nombre y sus fechas confirmadas."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    name: str = Field(min_length=1)
    confirmed_earnings: tuple[ConfirmedEarnings, ...] = ()


class MegaCapsConfig(BaseModel):
    """Contenido de ``config/mega_caps.yaml``: la lista **declarada** de emisores."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(ge=1)
    dataset: str = DATASET
    source: str = Field(min_length=1)
    declared_on: date = Field(description="fecha en que se declaró la lista de emisores")
    issuers: tuple[MegaCapIssuer, ...] = ()


class EarningsEvent(BaseModel):
    """Un resultado de mega-cap del día, con su certeza y su instante de observación."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    name: str
    on: date
    moment: EarningsMoment
    certainty: EarningsCertainty
    observed_at: datetime

    @property
    def blocking(self) -> bool:
        """``True`` sólo si la fecha está **confirmada**: una estimada nunca bloquea."""
        return self.certainty is EarningsCertainty.CONFIRMED


class EarningsReport(BaseModel):
    """Resumen de una ingesta: cuántas observaciones se escribieron."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    data_root: str
    events: tuple[EarningsEvent, ...] = ()


def load_mega_caps(path: Path | str | None = None) -> MegaCapsConfig:
    """Carga y valida ``config/mega_caps.yaml``."""
    target = Path(path) if path is not None else DEFAULT_MEGA_CAPS_PATH
    if not target.is_file():
        raise ConfigurationError(f"no existe la lista declarada de mega-caps: {target}")
    try:
        loaded: object = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"YAML inválido en {target}: {error}") from error
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"{target} debe contener un mapping en la raíz")
    try:
        return MegaCapsConfig.model_validate(cast("dict[str, object]", loaded))
    except ValidationError as error:
        raise ConfigurationError(f"lista de mega-caps inválida en {target}: {error}") from error


def _event(
    issuer: MegaCapIssuer,
    *,
    on: date,
    moment: EarningsMoment,
    certainty: EarningsCertainty,
    observed_at: datetime,
) -> EarningsEvent:
    """El evento del emisor con la certeza declarada."""
    return EarningsEvent(
        symbol=issuer.symbol,
        name=issuer.name,
        on=on,
        moment=moment,
        certainty=certainty,
        observed_at=observed_at,
    )


def events_for(
    config: MegaCapsConfig, *, source: EarningsSource, now: datetime
) -> tuple[EarningsEvent, ...]:
    """Fusiona lo que devuelve la fuente con las fechas confirmadas del artefacto.

    Lo que devuelve la fuente es **estimado** salvo que su fecha esté declarada como confirmada
    en ``config/mega_caps.yaml``. Una fecha confirmada que la fuente no devuelva se añade igual:
    el artefacto declarado manda sobre la estimación. ``observed_at`` nunca es posterior a
    ``now`` (el almacén lo exige).
    """
    events: list[EarningsEvent] = []
    for issuer in config.issuers:
        declared = {item.date: item.moment for item in issuer.confirmed_earnings}
        by_date: dict[date, EarningsEvent] = {}
        for raw in source.earnings_dates(issuer.symbol, now=now):
            observed = min(raw.observed_at, now)
            if raw.on in declared:
                moment = declared[raw.on]
                by_date[raw.on] = _event(
                    issuer,
                    on=raw.on,
                    moment=raw.moment if moment is EarningsMoment.UNKNOWN else moment,
                    certainty=EarningsCertainty.CONFIRMED,
                    observed_at=observed,
                )
            else:
                by_date[raw.on] = _event(
                    issuer,
                    on=raw.on,
                    moment=raw.moment,
                    certainty=EarningsCertainty.ESTIMATED,
                    observed_at=observed,
                )
        for on, moment in declared.items():
            by_date.setdefault(
                on,
                _event(
                    issuer,
                    on=on,
                    moment=moment,
                    certainty=EarningsCertainty.CONFIRMED,
                    observed_at=now,
                ),
            )
        events.extend(by_date.values())
    return tuple(sorted(events, key=lambda event: (event.on, event.symbol)))


def ingest(
    *,
    config: MegaCapsConfig,
    store: Store,
    source: EarningsSource,
    now: datetime,
) -> EarningsReport:
    """Observa las fechas de cada emisor y las escribe en ``raw.earnings``."""
    events = events_for(config, source=source, now=now)
    records: list[dict[str, object]] = [
        {
            "source": source.name,
            # `series_id` = `símbolo@fecha`: la identidad del almacén es
            # `(source, series_id, as_of)` y un emisor tiene varios eventos por corrida.
            "series_id": f"{event.symbol}@{event.on.isoformat()}",
            # `as_of` es la **observación** (una fecha futura no puede ser as_of del almacén,
            # que exige `fetched_at >= as_of`); la fecha del evento viaja en `event_date`.
            "as_of": now,
            "fetched_at": now,
            "published_at": event.observed_at,
            "name": event.name,
            "event_date": event.on,
            "moment": event.moment.value,
            "certainty": event.certainty.value,
            "observed_at": event.observed_at,
        }
        for event in events
    ]
    if records:
        store.append("raw", DATASET, records)
    return EarningsReport(generated_at=now, data_root=str(store.root), events=events)


def earnings_on(store: Store, *, session: date, as_of: datetime) -> tuple[EarningsEvent, ...]:
    """Los resultados de mega-caps del día ``session``, de solo lectura y *point-in-time*.

    Una sola fila por emisor: la **última** observación cuyo ``observed_at`` no es posterior al
    ``as_of`` (la estimación vigente ese día). Un almacén sin ``raw.earnings`` devuelve ``()``:
    no tener resultados no es un error.
    """
    if as_of.utcoffset() is None:
        raise ConfigurationError("as_of: se espera un datetime con zona (TZ-aware)")
    if DATASET not in store.datasets("raw"):
        return ()
    moment = as_of.astimezone(UTC)
    query = (
        "SELECT series_id, name, certainty, moment, event_date, observed_at FROM raw.earnings "  # noqa: S608
        f"WHERE event_date = DATE '{session.isoformat()}' "
        f"AND observed_at <= CAST('{moment.isoformat()}' AS TIMESTAMPTZ) "
        "ORDER BY observed_at"
    )
    try:
        frame = store.sql(query)
    except (UnknownDatasetError, duckdb.Error):
        return ()
    latest: dict[str, EarningsEvent] = {}
    for row in frame.iter_rows(named=True):
        series_id = str(row["series_id"])
        # `series_id` es `símbolo@fecha`: el símbolo es lo que va antes de la arroba.
        symbol = series_id.split("@", 1)[0]
        event = EarningsEvent(
            symbol=symbol,
            name=str(row["name"]),
            on=cast("date", row["event_date"]),
            moment=EarningsMoment(str(row["moment"])),
            certainty=EarningsCertainty(str(row["certainty"])),
            observed_at=cast("datetime", row["observed_at"]),
        )
        # El resultado viene ordenado por `observed_at`: la última gana.
        latest[event.symbol] = event
    return tuple(latest[symbol] for symbol in sorted(latest))


def main(argv: Sequence[str] | None = None) -> int:
    """Ingesta manual de resultados de mega-caps (``yfinance``) a ``raw.earnings``.

    Códigos: ``0`` = ingesta emitida; ``2`` = configuración o ``--now`` inválidos.
    """
    parser = argparse.ArgumentParser(
        prog="cfdtrader.data.earnings",
        description="Ingesta de resultados de mega-caps (yfinance) a raw.earnings",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="raíz del almacén")
    parser.add_argument("--mega-caps", type=Path, default=None, help="ruta de mega_caps.yaml")
    parser.add_argument("--now", required=True, help="instante declarado ISO-8601 con zona")
    args = parser.parse_args(argv)

    try:
        config = load_mega_caps(args.mega_caps)
        now = _parse_now(cast("str", args.now))
    except ConfigurationError as error:
        print(f"no se pueden ingestar los resultados: {error}", file=sys.stderr)
        return 2

    # Import perezoso: `yfinance` vive aislado en el adaptador y **no** entra en la lectura.
    from cfdtrader.data.sources.yfinance_adapter import YFinanceEarningsAdapter

    report = ingest(
        config=config,
        store=Store(Path(args.data_root)),
        source=YFinanceEarningsAdapter(),
        now=now,
    )
    logger.info("resultados de mega-caps: {} observaciones", len(report.events))
    return 0


def _parse_now(value: str) -> datetime:
    """Instante de referencia: el declarado, con zona, en UTC."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ConfigurationError(f"--now no es ISO-8601: {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ConfigurationError(f"--now necesita zona horaria: {value!r}")
    return parsed.astimezone(UTC)


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
