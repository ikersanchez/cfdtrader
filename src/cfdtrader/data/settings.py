"""Configuración tipada del pipeline de datos (``tech_stack.md`` §4.2).

Regla del proyecto: **un error de configuración debe fallar al arrancar, no a
mitad del pipeline**. Por eso el YAML se valida con Pydantic antes de tocar la
red o el almacén.

En Fase 0 solo se cierra ``data.root``: la raíz del almacén. El resto de valores
de ``config/settings.yaml`` (bróker, tier, umbrales, capital) son decisiones
abiertas (`tech_stack.md` §11 bis) y no se inventan aquí.

La raíz es configuración, no una constante del código: ``store.py`` recibe la
raíz por parámetro y este módulo es quien la cablea.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = [
    "DEFAULT_DATA_SOURCES_PATH",
    "DEFAULT_MACRO_SERIES_PATH",
    "DEFAULT_SETTINGS_PATH",
    "REPO_ROOT",
    "ConfigurationError",
    "DataSettings",
    "Settings",
    "load_settings",
]

#: ``<repo>/config/settings.yaml``, calculado desde la posición de este módulo.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"

#: ``<repo>/config/data_sources.yaml`` — declaración de series y fuentes (#3).
DEFAULT_DATA_SOURCES_PATH = REPO_ROOT / "config" / "data_sources.yaml"

#: ``<repo>/config/macro_series.yaml`` — series macro de FRED (#5).
DEFAULT_MACRO_SERIES_PATH = REPO_ROOT / "config" / "macro_series.yaml"


class ConfigurationError(Exception):
    """La configuración no es válida: se falla al arrancar, con el motivo."""


class DataSettings(BaseModel):
    """Bloque ``data`` de ``config/settings.yaml``."""

    model_config = ConfigDict(extra="forbid")

    root: Path = Field(default=Path("data"))
    """Raíz del almacén. Relativa al directorio de trabajo si no es absoluta."""

    def resolve(self, *, base: Path | None = None) -> Path:
        """Ruta absoluta de la raíz del almacén (relativa a ``base`` si hace falta)."""
        if self.root.is_absolute():
            return self.root
        return (base or Path.cwd()) / self.root


class Settings(BaseModel):
    """Configuración completa. Los bloques no cerrados todavía se ignoran a propósito."""

    model_config = ConfigDict(extra="ignore")

    data: DataSettings = Field(default_factory=DataSettings)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"no existe el fichero de configuración: {path}")
    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"YAML inválido en {path}: {error}") from error
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigurationError(
            f"{path} debe contener un mapping en la raíz, no {type(loaded).__name__}"
        )
    raw = cast("dict[object, object]", loaded)
    return {str(key): value for key, value in raw.items()}


def load_settings(path: Path | str | None = None) -> Settings:
    """Carga y valida ``config/settings.yaml``.

    Raises
    ------
    ConfigurationError
        Si el fichero no existe, no es YAML válido o no cumple el esquema.
    """
    target = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    try:
        return Settings.model_validate(_read_yaml(target))
    except ValidationError as error:
        raise ConfigurationError(f"configuración inválida en {target}: {error}") from error
