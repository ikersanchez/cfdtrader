"""Registro de series y fuentes — tarea #3.

Lee ``config/data_sources.yaml`` y lo valida con Pydantic. La declaración es
**por serie**: ``series_id``, ``dataset``, ``asset_class``, granularidad, fuente
primaria, respaldos y ``min_start``.

Este módulo es también donde se hace cumplir la regla de oro de la tarea:
``SPX500:CFD`` **no se sustituye** por ``^GSPC``, ``ES=F`` ni ``SPY``
(:func:`cfd_substitutions`). Un alias silencioso aquí falsearía toda la Fase 1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cfdtrader.data.settings import DEFAULT_DATA_SOURCES_PATH, ConfigurationError
from cfdtrader.data.sources.base import AssetClass, SeriesSpec

__all__ = [
    "CFD_SERIES_ID",
    "SeriesRegistry",
    "UnavailableSeries",
    "cfd_substitutions",
    "load_registry",
    "series_by_dataset",
    "series_for_source",
    "unavailable_ids",
]

#: Instrumento del proyecto. Es el único identificador que no se puede sustituir.
CFD_SERIES_ID = "SPX500:CFD"


class UnavailableSeries(BaseModel):
    """Serie declarada **sin fuente**, con la evidencia de la comprobación.

    No es un ``TODO``: es una declaración explícita que el informe de cobertura
    convierte en un bloqueo comprobable de la Fase 1 (A19, A21).
    """

    model_config = ConfigDict(extra="forbid")

    series_id: str
    status: str
    bid_ask: bool
    reason: str
    checked_on: date
    follow_up_issue: int | None = None
    documentation: str | None = None


class SeriesRegistry(BaseModel):
    """Contenido de ``config/data_sources.yaml``."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    series: tuple[SeriesSpec, ...] = ()
    unavailable: tuple[UnavailableSeries, ...] = ()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"no existe el registro de fuentes: {path}")
    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"YAML inválido en {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"{path} debe contener un mapping en la raíz")
    raw = cast("dict[object, object]", loaded)
    return {str(key): value for key, value in raw.items()}


def load_registry(path: Path | str | None = None) -> SeriesRegistry:
    """Carga y valida el registro de series.

    Raises
    ------
    ConfigurationError
        Si el fichero no existe, es YAML inválido o incumple el esquema.
    """
    target = Path(path) if path is not None else DEFAULT_DATA_SOURCES_PATH
    try:
        registry = SeriesRegistry.model_validate(_read_yaml(target))
    except ValidationError as error:
        raise ConfigurationError(f"registro de fuentes inválido en {target}: {error}") from error
    _check_registry(registry, target=target)
    return registry


def _check_registry(registry: SeriesRegistry, *, target: Path) -> None:
    """Invariantes del registro que el esquema de Pydantic no puede expresar."""
    identities = [(spec.dataset, spec.series_id, spec.interval) for spec in registry.series]
    if len(set(identities)) != len(identities):
        raise ConfigurationError(
            f"{target}: hay una serie repetida (mismo dataset, series_id e intervalo)"
        )
    unavailable_ids = {item.series_id for item in registry.unavailable}
    for spec in registry.series:
        if spec.series_id in unavailable_ids:
            raise ConfigurationError(
                f"{target}: {spec.series_id!r} está declarada a la vez como descargable y como "
                "sin fuente; una serie sin fuente no puede intentar descargarse"
            )
        if spec.asset_class is AssetClass.CFD:
            raise ConfigurationError(
                f"{target}: {spec.series_id!r} está marcada como CFD. El CFD del proyecto no "
                "tiene fuente: no se sustituye por el índice ni por el futuro"
            )


def series_for_source(registry: SeriesRegistry, source: str) -> list[SeriesSpec]:
    """Series en las que esa fuente es primaria o respaldo."""
    return [spec for spec in registry.series if source in spec.sources]


def series_by_dataset(registry: SeriesRegistry, dataset: str) -> list[SeriesSpec]:
    """Series que escriben en un dataset del almacén."""
    return [spec for spec in registry.series if spec.dataset == dataset]


def cfd_substitutions(
    registry: SeriesRegistry,
    alias_maps: Mapping[str, Mapping[str, str]] | None = None,
) -> list[str]:
    """Sustituciones prohibidas del CFD, en forma de lista de motivos.

    Devuelve lista vacía si el CFD no se sustituye por nada —que es el estado
    correcto hoy, y el que el test ``test_cfd_has_no_alias_to_index_or_future``
    exige. Cualquier alias que alguien introduzca (en el registro o en el mapa
    de símbolos de un adaptador) aparece aquí.

    Parameters
    ----------
    registry:
        Registro cargado de ``config/data_sources.yaml``.
    alias_maps:
        Mapas de símbolos declarados por los adaptadores, como
        ``{"stooq": {"^GSPC": "^spx"}}``.
    """
    problems: list[str] = []
    for spec in registry.series:
        if spec.series_id == CFD_SERIES_ID:
            problems.append(
                f"{CFD_SERIES_ID} figura como serie descargable de la fuente {spec.primary!r}"
            )
    for source, mapping in (alias_maps or {}).items():
        target = mapping.get(CFD_SERIES_ID)
        if target is not None:
            problems.append(f"{source} mapea {CFD_SERIES_ID} a {target!r}")
    return problems


def unavailable_ids(registry: SeriesRegistry) -> Sequence[str]:
    """Identificadores declarados sin fuente."""
    return tuple(item.series_id for item in registry.unavailable)
