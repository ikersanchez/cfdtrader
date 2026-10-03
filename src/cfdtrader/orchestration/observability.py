"""Observabilidad del pipeline: ``run_log`` en JSONL y ``manifest.json`` (tarea #43).

El objetivo: que **cualquier fallo quede registrado y no se pierda** al ejecutar el
pipeline a mano (``tech_stack.md`` §4.13). No hay alertas, ni *heartbeat*, ni envío a
terceros: la señal de que algo fue mal es el código de salida y la traza de la ejecución.

Qué escribe, por ejecución
--------------------------
- ``<root>/<run_id>/run_log.jsonl``: una línea JSON por **etapa**, con sus columnas
  ``run_id``, ``as_of``, ``stage``, ``duration_ms``, ``ok`` y ``error``. Son las mismas
  columnas que declara ``ops.run_log`` en el esquema del diario (§12.6): se **importan**
  de ``journal.decision_log.TABLE_COLUMNS`` para no tener una segunda definición.
- ``<root>/<run_id>/manifest.json``: la identidad de la ejecución, con **versiones** y
  **hashes** declarados y el resultado global (``ok``).

Determinismo y reloj
--------------------
El módulo **no lee la red** y no decide nada por sí solo: el ``run_id`` y el ``as_of``
son **entradas** (el llamante los declara). Las **duraciones** de cada etapa sí necesitan
medir el tiempo, y por eso el cronómetro se **inyecta** (``timer``): en producción es
``time.perf_counter``; en los tests, un reloj de mentira que fija las duraciones.

Un fallo no se pierde
---------------------
``RunObserver`` es un *context manager*: al salir —**también si una etapa lanza**— escribe
el ``run_log`` y el ``manifest``. El contexto ``stage`` captura la **traza completa**
(``traceback.format_exc``) en la columna ``error`` y **vuelve a lanzar**, de modo que el
pipeline falla a la vista y la ejecución queda documentada.

Frontera declarada
------------------
Esta tarea entrega el **módulo** y sus pruebas; **cablearlo** en ``delivery/run_daily.py``
(que hoy no escribe traza estructurada) es un paso aparte, con su propia guarda de diff.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from cfdtrader.backtest.engine import canonical_text
from cfdtrader.journal.decision_log import TABLE_COLUMNS

__all__ = [
    "MANIFEST_FILENAME",
    "RUN_LOG_COLUMNS",
    "RUN_LOG_FILENAME",
    "Manifest",
    "ObservabilityError",
    "RunObserver",
    "StageRecord",
]

#: Nombre del fichero de traza (una linea JSON por etapa).
RUN_LOG_FILENAME: Final[str] = "run_log.jsonl"

#: Nombre del fichero de identidad de la ejecucion (hashes y versiones).
MANIFEST_FILENAME: Final[str] = "manifest.json"

#: Columnas del ``run_log``: las **mismas** que declara ``ops.run_log`` (§12.6). Se importan
#: del esquema del diario para que no exista una segunda definicion.
RUN_LOG_COLUMNS: Final[tuple[str, ...]] = tuple(TABLE_COLUMNS["run_log"])


class ObservabilityError(Exception):
    """Una entrada o una ruta del observador no es válida."""


class StageRecord(BaseModel):
    """Una etapa de la ejecución: columnas exactas de ``ops.run_log`` (§12.6)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(description="identidad de la ejecución (entrada del llamante)")
    as_of: datetime = Field(description="instante declarado de la ejecución, con zona")
    stage: str = Field(description="nombre de la etapa")
    duration_ms: int = Field(ge=0, description="duración medida por el cronómetro inyectado")
    ok: bool = Field(description="True si la etapa terminó sin excepción")
    error: str | None = Field(
        default=None, description="traza completa (traceback) si la etapa falló; None si no"
    )


class Manifest(BaseModel):
    """Identidad de la ejecución: versiones, hashes y resultado global (§4.13)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(description="identidad de la ejecución")
    as_of: datetime = Field(description="instante declarado, con zona")
    git_commit: str = Field(description="commit declarado por el llamante; el módulo no lee git")
    versions: dict[str, str] = Field(
        default_factory=dict, description="versiones (features_version, model_version, …)"
    )
    hashes: dict[str, str] = Field(
        default_factory=dict, description="digests con prefijo sha256: (gate_sha256, …)"
    )
    stages: tuple[str, ...] = Field(default=(), description="etapas registradas, en orden")
    ok: bool = Field(description="False si alguna etapa o el cierre fallaron")


# ─────────────────────────────────────────────────────────────────────────────
# Validación y payloads
# ─────────────────────────────────────────────────────────────────────────────
def _require_text(value: object, *, field_name: str) -> str:
    """Un texto no vacío: una identidad o una etiqueta vacía no dice nada."""
    if not isinstance(value, str) or not value.strip():
        raise ObservabilityError(f"{field_name}: se espera un texto no vacío, no {value!r}")
    return value


def _require_aware(value: object, *, field_name: str) -> datetime:
    """Un instante con zona: sin TZ no identifica un momento."""
    if not isinstance(value, datetime):
        raise ObservabilityError(
            f"{field_name}: se espera un `datetime`, no {type(value).__name__}"
        )
    if value.utcoffset() is None:
        raise ObservabilityError(f"{field_name}: se espera un `datetime` con zona (TZ-aware)")
    return value


def _iso(instant: datetime) -> str:
    """El instante en UTC, ISO-8601 (el almacenamiento es siempre UTC, ``plan.md`` §8.3)."""
    return instant.astimezone(UTC).isoformat()


def _stage_payload(record: StageRecord) -> dict[str, object]:
    """Payload JSON puro de una etapa, con las columnas de ``ops.run_log``."""
    return {
        "run_id": record.run_id,
        "as_of": _iso(record.as_of),
        "stage": record.stage,
        "duration_ms": record.duration_ms,
        "ok": record.ok,
        "error": record.error,
    }


def _manifest_payload(manifest: Manifest) -> dict[str, object]:
    """Payload JSON puro del *manifest*."""
    return {
        "run_id": manifest.run_id,
        "as_of": _iso(manifest.as_of),
        "git_commit": manifest.git_commit,
        "versions": dict(manifest.versions),
        "hashes": dict(manifest.hashes),
        "stages": list(manifest.stages),
        "ok": manifest.ok,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Observador de una ejecución
# ─────────────────────────────────────────────────────────────────────────────
class RunObserver:
    """Traza estructurada de una ejecución: etapas, duraciones, errores, hashes y versiones.

    Se usa como *context manager*; al salir escribe el ``run_log`` y el ``manifest``, **también
    si una etapa lanza**. El cronómetro se inyecta (``timer``) para que las duraciones sean
    deterministas en los tests y reales en producción.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        run_id: str,
        as_of: datetime,
        git_commit: str,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._root = Path(str(root))
        self._run_id = _require_text(run_id, field_name="run_id")
        self._as_of = _require_aware(as_of, field_name="as_of")
        self._git_commit = _require_text(git_commit, field_name="git_commit")
        if not callable(timer):
            raise ObservabilityError(f"timer: se espera un callable, no {type(timer).__name__}")
        self._timer = timer
        self._records: list[StageRecord] = []
        self._versions: dict[str, str] = {}
        self._hashes: dict[str, str] = {}
        self._ok = True

    @property
    def run_id(self) -> str:
        """Identidad declarada de la ejecución."""
        return self._run_id

    @property
    def directory(self) -> Path:
        """Directorio de la ejecución: ``<root>/<run_id>``."""
        return self._root / self._run_id

    @property
    def records(self) -> tuple[StageRecord, ...]:
        """Las etapas registradas, en orden."""
        return tuple(self._records)

    def add_version(self, name: str, value: str) -> None:
        """Declara una versión (``features_version``, ``model_version``, …) para el *manifest*."""
        self._versions[_require_text(name, field_name="version")] = _require_text(
            value, field_name=f"version {name!r}"
        )

    def add_hash(self, name: str, value: str) -> None:
        """Declara un digest (``gate_sha256``, …) para el *manifest*."""
        self._hashes[_require_text(name, field_name="hash")] = _require_text(
            value, field_name=f"hash {name!r}"
        )

    def record(
        self, stage: str, *, duration_ms: int = 0, ok: bool = True, error: str | None = None
    ) -> StageRecord:
        """Añade una etapa a la traza y devuelve la fila escrita."""
        if isinstance(duration_ms, bool) or duration_ms < 0:
            raise ObservabilityError(f"duration_ms: se espera un entero >= 0, no {duration_ms!r}")
        entry = StageRecord(
            run_id=self._run_id,
            as_of=self._as_of,
            stage=_require_text(stage, field_name="stage"),
            duration_ms=duration_ms,
            ok=bool(ok),
            error=error,
        )
        self._records.append(entry)
        if not entry.ok:
            self._ok = False
        return entry

    def fail(self, error: str) -> None:
        """Marca la ejecución como fallida sin registrar una etapa (para cierres externos)."""
        self._ok = False
        _require_text(error, field_name="error")

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Mide una etapa; si lanza, guarda su **traza completa** y **vuelve a lanzar**."""
        start = self._timer()
        try:
            yield
        except Exception:
            self.record(
                name,
                duration_ms=self._elapsed(start),
                ok=False,
                error=traceback.format_exc(),
            )
            raise
        else:
            self.record(name, duration_ms=self._elapsed(start), ok=True)

    def _elapsed(self, start: float) -> int:
        return max(0, round((self._timer() - start) * 1000))

    def manifest(self) -> Manifest:
        """El *manifest* de la ejecución con lo acumulado hasta ahora."""
        return Manifest(
            run_id=self._run_id,
            as_of=self._as_of,
            git_commit=self._git_commit,
            versions=dict(self._versions),
            hashes=dict(self._hashes),
            stages=tuple(entry.stage for entry in self._records),
            ok=self._ok,
        )

    def finalize(self) -> tuple[Path, Path]:
        """Escribe ``run_log.jsonl`` y ``manifest.json``; devuelve sus rutas."""
        directory = self.directory
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / RUN_LOG_FILENAME
        manifest_path = directory / MANIFEST_FILENAME
        log_path.write_text(
            "".join(canonical_text(_stage_payload(entry)) + "\n" for entry in self._records),
            encoding="utf-8",
        )
        manifest_path.write_text(
            canonical_text(_manifest_payload(self.manifest())) + "\n", encoding="utf-8"
        )
        return log_path, manifest_path

    def __enter__(self) -> RunObserver:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        if exc is not None:
            self._ok = False
        self.finalize()
