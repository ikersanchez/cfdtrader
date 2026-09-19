"""Registro de experimentos en `runs/<hash>/` e informe de sobreajuste (tarea #16).

`plan.md` §11.4 lo dice sin matices: «**Registro de todos los experimentos**: cada variante
probada se anota (features, hiperparametros, resultado). **Innegociable**». Ese registro es
lo que da sentido al DSR: el numero de variantes probadas se **deriva** de el, asi que
ninguna funcion publica, ningun campo del payload y ninguna opcion del CLI permiten fijarlo
a mano (`--n-trials` no existe: un valor «estimado» es error tipado). Este modulo es el
unico que **escribe**: el calculo puro vive en `cfdtrader.backtest.overfitting`, el mismo
reparto que dejaron #69 y #18 (nucleo puro en `backtest/`, disco en `analysis/`).

Que se escribe, por experimento, en `runs/<sha256>/`:

- `config.json`: la configuracion (features, hiperparametros, semilla, serie y ventana).
- `result.json`: el resultado registrado, con el Sharpe **por sesion** y el bloque
  `net_metrics` no computable de #15 (A23).
- `summary.md`: el resumen legible, que cita el hash.

La **identidad es el contenido de la configuracion**: ``run_sha256 =
sha256(canonical_text(config))``, el canonico de #13, con `EXPERIMENT_HASH_FORMAT`
documentado. **Ni la ruta ni el instante entran en la identidad** — igual que el almacen de
#2 deja `version` y `fetched_at` fuera del contenido: la misma configuracion registrada con
dos `as_of` distintos da el mismo hash, y por eso el instante **no** se escribe dentro de
los tres ficheros (el informe de sobreajuste, que si lleva `generated_at`, lo publica).
Reescribir una identidad con **otro** contenido es `ExperimentRewriteError`, nunca una
sobrescritura silenciosa; reescribir el contenido **identico** es un no-op que devuelve el
vocabulario de #2 (`WriteOutcome.created` / `WriteOutcome.unchanged`, **importado** de
`cfdtrader.data.store`, no un enum paralelo). Como el resultado es inmutable igual, el
registro es de paso un **detector de no-determinismo**.

`runs/` esta gitignorado (`.gitignore` linea 55): los artefactos **no** se commitean. El
registro **no** usa el `Store` ni `read_pit` (el almacen es para datos de mercado, no para
experimentos): son ficheros JSON y Markdown. La divergencia con `ops.backtest_runs` de
`tech_stack.md` §12 queda declarada en el payload, con su seguimiento (#65 documenta, #44
retencion). **Sin reloj**: el instante entra por `--as-of`, obligatorio para escribir.

Los dos veredictos (DSR y PBO) se agregan con `aggregate_gate` de #9, **importado**, nunca
reimplementado: `fail` si alguno falla, `not_evaluable` si ninguno falla y alguno no es
evaluable, y `pass` **solo** si los dos son `pass`. Un agregado que no sea el de esa regla
es error de consistencia.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final, cast

from loguru import logger

from cfdtrader.analysis.phase0_report import (
    GATE_AGGREGATION_RULE,
    HalfResult,
    aggregate_gate,
)
from cfdtrader.backtest.engine import canonical_text
from cfdtrader.backtest.metrics import sharpe_ratio
from cfdtrader.backtest.overfitting import (
    DEFAULT_BLOCKS,
    DSR_CONFIDENCE_LEVEL,
    FOLLOW_UPS,
    MAX_COMBINATIONS,
    NOISE_SEED,
    OVERFITTING_DOES_NOT_DO,
    PBO_MAX,
    PER_SESSION,
    SIGNAL_MEAN,
    SIGNAL_SEED,
    SYNTHETIC_OBSERVATIONS,
    SYNTHETIC_SIGMA,
    SYNTHETIC_VARIANTS,
    THRESHOLD_RULE,
    VERDICT_NOT_EVALUABLE,
    DegenerateMatrixError,
    DegenerateSeriesError,
    InsufficientObservationsError,
    InsufficientTrialsError,
    InsufficientVariantsError,
    InvalidMatrixShapeError,
    InvalidVarianceError,
    OverfittingError,
    deflated_sharpe_ratio,
    noise_matrix,
    probability_of_backtest_overfitting,
    select_variant,
    signal_matrix,
    variant_sharpe_variance,
)
from cfdtrader.data.store import WriteOutcome

__all__ = [
    "BASIS_DECLARED_COST",
    "CLI_NAME",
    "CONFIG_FILE",
    "CSCV_VERSUS_CPCV",
    "DEFAULT_REPORTS_DIR",
    "DEFAULT_RUNS_ROOT",
    "EXPERIMENT_HASH_FORMAT",
    "EXPERIMENT_NAMES",
    "HOLDOUT_BOUNDARY",
    "NET_METRICS",
    "NET_METRICS_REASON",
    "NOISE_SIGNAL_RULE",
    "REPORT_HASH_FORMAT",
    "REPORT_PREFIX",
    "RESULT_FILE",
    "SELECTION_RULE",
    "STORE_DIVERGENCE",
    "SUMMARY_FILE",
    "SYNTHETIC_SERIES",
    "EmptyRegistryError",
    "ExperimentConfig",
    "ExperimentLogError",
    "ExperimentRecord",
    "ExperimentResult",
    "ExperimentRewriteError",
    "InvalidAsOfError",
    "MissingAsOfError",
    "OverfittingReport",
    "Registry",
    "RegistryEntry",
    "RegistryIntegrityError",
    "TrialsMismatchError",
    "VerdictConsistencyError",
    "aggregate_verdict",
    "analyse",
    "deflate_block",
    "load_registry",
    "main",
    "pbo_block",
    "record_experiment",
    "render_markdown",
    "require_consistent_aggregate",
    "require_trials_match_registry",
    "run_sha256",
]

#: Nombre del programa en la CLI (`python -m cfdtrader.analysis.experiment_log`).
CLI_NAME: Final[str] = "cfdtrader.analysis.experiment_log"

#: Los tres ficheros de un experimento, con nombre fijo.
CONFIG_FILE: Final[str] = "config.json"
RESULT_FILE: Final[str] = "result.json"
SUMMARY_FILE: Final[str] = "summary.md"

#: Raiz y destino declarados por defecto del CLI (A31).
DEFAULT_RUNS_ROOT: Final[str] = "runs"
DEFAULT_REPORTS_DIR: Final[str] = "data/derived/reports"

#: Prefijo del informe: ``overfitting_<AAAA-MM-DD>.{json,md}``.
REPORT_PREFIX: Final[str] = "overfitting"

#: Serie declarada de la validacion sintetica: no es una serie de mercado.
SYNTHETIC_SERIES: Final[str] = "SYNTHETIC"

#: Los dos experimentos sinteticos, en orden estable.
EXPERIMENT_NAMES: Final[tuple[str, ...]] = ("noise", "signal")

#: Base declarada del retorno registrado: coste **declarado**, no medido (A23).
BASIS_DECLARED_COST: Final[str] = "declared_cost"

#: Formato estable del ``run_sha256`` (A12): el canonico de #13 sobre la configuracion.
EXPERIMENT_HASH_FORMAT: Final[str] = (
    "sha256(UTF-8) de cfdtrader.backtest.engine.canonical_text(config), es decir "
    "json.dumps(config, sort_keys=True, separators=(',', ':'), ensure_ascii=False) con "
    "Decimal como cadena exacta y float via repr; el texto que se hashea **no** incluye la "
    "clave run_sha256, **ni la ruta ni el instante**: la identidad es el contenido de la "
    "configuracion, igual que el almacen de #2 deja version y fetched_at fuera del contenido"
)

#: Formato estable del ``registry_sha256``: la lista ordenada de entradas del registro.
REGISTRY_HASH_FORMAT: Final[str] = (
    "sha256(UTF-8) de canonical_text({'entries': [...]}) con las entradas **ordenadas por "
    "run_sha256**: la huella del registro que produjo el `n_trials` declarado"
)

#: Formato estable del ``report_sha256`` del informe (misma convencion que #18/#69).
REPORT_HASH_FORMAT: Final[str] = (
    "sha256(UTF-8) de cfdtrader.backtest.engine.canonical_text(payload), **sin** la clave "
    "report_sha256 (un informe no se hashea a si mismo); el payload va como tipos JSON puros, "
    "float via repr y nan/inf prohibidos"
)

#: Regla de seleccion declarada (la del nucleo, publicada para poder auditar cada fila).
SELECTION_RULE: Final[str] = (
    "la variante publicada es la de mayor Sharpe por sesion sobre la muestra completa "
    "(`sharpe_ratio(..., annualization=1)` de #15), con los empates al indice de columna mas "
    "bajo"
)

#: Regla de oro de la validacion sintetica (A16, A17).
NOISE_SIGNAL_RULE: Final[str] = (
    "el **ruido puro** no puede aprobar (su DSR sale `not_significant` y su PBO alto) y la "
    "**senal plantada** debe detectarse (`significant` y `pbo < 0.20`); si el ruido pasa, el "
    "modulo miente"
)

#: Por que el retorno registrado es declarado y no neto (A23).
NET_METRICS_REASON: Final[str] = (
    "`pnl_net_pct` es `null` en todas las operaciones: el *slippage* de #64 es un **supuesto** "
    "declarado (`state: assumed`) y sin `R` (#60) no se puede cobrar, asi que "
    "`cfdtrader.backtest.metrics.calculate_metrics` **rechaza** la corrida en vez de publicar "
    "una metrica neta construida sobre el retorno declarado"
)

#: El hueco declarado de las metricas netas, con su procedencia y su seguimiento (A23).
NET_METRICS: Final[dict[str, object]] = {
    "state": "not_computable",
    "reason": NET_METRICS_REASON,
    "where": "cfdtrader.backtest.metrics.calculate_metrics",
    "follow_up": ["#62", "#60"],
    "note": (
        "regla «`null != 0`» de #15: el Sharpe que se deflacta y el PBO que se mide son de "
        "**retorno declarado**; ningun campo de este informe afirma que la estrategia este "
        "validada"
    ),
}

#: Frontera con #67: el CSCV del PBO **no** es el CPCV (A33).
CSCV_VERSUS_CPCV: Final[str] = (
    "el CSCV del PBO recombina bloques de la **matriz de retornos de las variantes** y no "
    "construye un `SplitPlan` ni purga ni embargo nada: el esquema de validacion del modelo "
    "(particiones, purga y embargo) es #67. `cfdtrader.backtest.overfitting` no importa "
    "`cfdtrader.backtest.splits`"
)

#: Frontera con #68: el *holdout* final no se reserva ni se mira (A34).
HOLDOUT_BOUNDARY: Final[str] = (
    "este modulo no reserva ni lee el periodo final intocable (`plan.md` §11.4 linea 715 y §21 "
    "pregunta 10): no hay ninguna constante de «ultimos meses» ni muestra reservada, y el "
    "*holdout* es #68"
)

#: Divergencia declarada con el almacen operativo (A32).
STORE_DIVERGENCE: Final[str] = (
    "el registro **no** usa el `Store` ni `read_pit`: `runs/<hash>/` son ficheros JSON/MD y el "
    "almacen es para datos de mercado. `tech_stack.md` §12 preve `ops.backtest_runs`; la "
    "reconciliacion entre ambos (y la retencion de `runs/`) queda declarada, no resuelta: la "
    "documenta #65 y la poda es #44"
)

#: Traduccion de los veredictos del nucleo al vocabulario de #9, que agrega.
DSR_HALVES: Final[dict[str, str]] = {
    "significant": HalfResult.PASS.value,
    "not_significant": HalfResult.FAIL.value,
    VERDICT_NOT_EVALUABLE: HalfResult.NOT_EVALUABLE.value,
}

#: Traduccion del veredicto del PBO: `not_detected` es el que aprueba (`pbo < PBO_MAX`).
PBO_HALVES: Final[dict[str, str]] = {
    "not_detected": HalfResult.PASS.value,
    "detected": HalfResult.FAIL.value,
    VERDICT_NOT_EVALUABLE: HalfResult.NOT_EVALUABLE.value,
}

#: Consecuencia declarada de un calculo no evaluable, con su seguimiento.
NOT_EVALUABLE_FOLLOW_UP: Final[tuple[str, ...]] = ("#62", "#60", "#68")

#: Registro vacio o sin la varianza de las variantes: se declara, nunca se inventa un numero.
REGISTRY_NOT_EVALUABLE: Final[str] = (
    "sin la varianza de los Sharpe de las variantes (`V[SR]`) no hay `SR0` que deflactar: hace "
    "falta un registro con al menos **dos** variantes probadas (A22)"
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados (A13, A15, A19, A24)
# ─────────────────────────────────────────────────────────────────────────────
class ExperimentLogError(Exception):
    """Raiz de los errores del registro de experimentos y de su informe."""


class ExperimentRewriteError(ExperimentLogError):
    """Reescritura de una identidad ya registrada con otro contenido: nunca sobrescritura."""


class EmptyRegistryError(ExperimentLogError):
    """El registro no tiene ninguna entrada: no se inventa un `n_trials`."""


class RegistryIntegrityError(ExperimentLogError):
    """Una entrada del registro no es coherente con su propia identidad (o esta corrupta)."""


class TrialsMismatchError(ExperimentLogError):
    """Un `n_trials`/`sr_variance` suelto no coincide con el que deriva el registro."""


class MissingAsOfError(ExperimentLogError):
    """Falta `--as-of`: el modulo no lee el reloj y sin instante no se escribe nada."""


class InvalidAsOfError(ExperimentLogError):
    """`--as-of` no es un instante ISO-8601 valido."""


class VerdictConsistencyError(ExperimentLogError):
    """El agregado declarado no es el de la regla de #9, o aprueba con una mitad no evaluable."""


# ─────────────────────────────────────────────────────────────────────────────
# Tipos JSON puros (convencion de #13/#18): Decimal como cadena, sin nan ni inf
# ─────────────────────────────────────────────────────────────────────────────
def _jsonable(value: object, *, where: str) -> object:
    """Traduce un valor a tipo JSON puro, o falla con error tipado (A20).

    ``float`` no finito es error (nunca ``0``), las fechas y los ``Decimal`` se convierten en
    cadena (ISO-8601 y decimal exacta) y cualquier otro tipo es error: el JSON del registro
    no admite sorpresas.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExperimentLogError(f"{where}: no se registra `nan` ni `inf` (A20)")
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        return {str(key): _jsonable(item, where=f"{where}.{key}") for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        sequence = cast("Sequence[object]", value)
        return [_jsonable(item, where=f"{where}[{index}]") for index, item in enumerate(sequence)]
    raise ExperimentLogError(
        f"{where}: el registro solo admite tipos JSON, `Decimal` y secuencias; llego "
        f"{type(value).__name__} (A20)"
    )


def _json_object(value: object, *, where: str) -> dict[str, object]:
    """Como :func:`_jsonable`, exigiendo un objeto JSON: una configuracion es un objeto."""
    translated = _jsonable(value, where=where)
    if not isinstance(translated, dict):
        raise ExperimentLogError(f"{where}: se esperaba un objeto JSON (A20)")
    return cast("dict[str, object]", translated)


def _json_file(document: Mapping[str, object]) -> str:
    """Texto de un fichero del registro: determinista, legible y con ``allow_nan=False``."""
    return json.dumps(dict(document), ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _digest(document: Mapping[str, object]) -> str:
    """sha256 del texto canonico de #13: la unica funcion de hash que usa el modulo."""
    return hashlib.sha256(canonical_text(document).encode("utf-8")).hexdigest()


def _as_utc(value: datetime) -> datetime:
    """Normaliza el instante declarado a UTC; sin zona se interpreta como UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


# ─────────────────────────────────────────────────────────────────────────────
# La configuracion, el resultado y su identidad (A11, A12)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """La configuracion registrada de una variante (A11): identidad del experimento.

    Contiene lo que `plan.md` §11.4 exige anotar por variante: **features**,
    **hiperparametros**, **semilla**, **serie** y **ventana de datos**. Todos los campos son
    JSON puros, y ni la ruta ni el instante forman parte de ellos (A12).
    """

    variant_id: str
    features: tuple[str, ...]
    hyperparameters: Mapping[str, object]
    seed: int
    series_id: str
    window: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        """La configuracion como objeto JSON puro: es **el** texto que se hashea (A12)."""
        return {
            "variant_id": self.variant_id,
            "features": list(self.features),
            "hyperparameters": _json_object(self.hyperparameters, where="hyperparameters"),
            "seed": self.seed,
            "series_id": self.series_id,
            "window": _json_object(self.window, where="window"),
        }


def run_sha256(config: ExperimentConfig) -> str:
    """`run_sha256`: el sha256 del contenido canonico de la configuracion (A12)."""
    return _digest(config.to_payload())


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """El resultado registrado de una variante: el Sharpe **por sesion** y su base (A23).

    Con el *slippage* supuesto de #64 el Sharpe neto no existe: lo que se registra es el
    Sharpe del retorno **declarado** (`basis: "declared_cost"`, `is_validation: false`) y el
    bloque ``net_metrics`` de #15, que declara por que no se publica. Sustituir el retorno
    declarado por el neto, o el supuesto por una medicion, esta prohibido.
    """

    sharpe_per_session: float
    n_observations: int
    basis: str = BASIS_DECLARED_COST
    is_validation: bool = False

    def to_payload(self) -> dict[str, object]:
        """El resultado como objeto JSON puro, con el motivo del `null` y su seguimiento."""
        if not math.isfinite(self.sharpe_per_session):
            raise ExperimentLogError("el Sharpe registrado debe ser finito (A20)")
        return {
            "sharpe_per_session": self.sharpe_per_session,
            "n_observations": self.n_observations,
            "units": PER_SESSION,
            "annualization_used": 1,
            "basis": self.basis,
            "is_validation": self.is_validation,
            "net_metrics": dict(NET_METRICS),
        }


def _write_immutable(path: Path, text: str) -> WriteOutcome:
    """Escribe si el fichero no existe; contenido identico es no-op; otro, error (A13)."""
    encoded = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() == encoded:
            return WriteOutcome.UNCHANGED
        raise ExperimentRewriteError(
            f"{path} ya existe con **otro** contenido: la identidad del experimento es su "
            "contenido, asi que reescribirla con otro resultado es error tipado, nunca una "
            "sobrescritura silenciosa (A13, A14)"
        )
    path.write_bytes(encoded)
    return WriteOutcome.CREATED


def render_summary(*, digest: str, config: ExperimentConfig, result: ExperimentResult) -> str:
    """El `summary.md` legible: cita el hash y no duplica el JSON (A11)."""
    hyperparameters = cast("dict[str, object]", config.to_payload()["hyperparameters"])
    window = cast("dict[str, object]", config.to_payload()["window"])
    lines = [
        f"# Experimento `{config.variant_id}`",
        "",
        f"- `run_sha256`: `{digest}`",
        f"- directorio: `runs/{digest}/` (gitignorado: no se commitea)",
        f"- serie: `{config.series_id}`",
        f"- semilla: `{config.seed}`",
        f"- ventana: {json.dumps(window, ensure_ascii=False, sort_keys=True)}",
        f"- features: {', '.join(f'`{feature}`' for feature in config.features) or 'ninguna'}",
        f"- hiperparametros: {json.dumps(hyperparameters, ensure_ascii=False, sort_keys=True)}",
        "",
        "## Resultado",
        "",
        "| campo | valor |",
        "|---|---|",
        f"| Sharpe por sesion | {result.sharpe_per_session!r} |",
        f"| observaciones | {result.n_observations} |",
        f"| base del retorno | `{result.basis}` |",
        f"| `is_validation` | `{str(result.is_validation).lower()}` |",
        f"| metricas netas | `{NET_METRICS['state']}` — {NET_METRICS_REASON} |",
        "",
        "El detalle completo esta en `config.json` y `result.json` de este mismo directorio;",
        "este resumen no lo duplica.",
        "",
    ]
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """Un experimento registrado (o listo para registrar, si ``written`` es ``False``)."""

    as_of: datetime
    run_sha256: str
    directory: Path
    config: ExperimentConfig
    result: ExperimentResult
    outcome: WriteOutcome | None
    written: bool

    def to_entry(self) -> RegistryEntry:
        """La entrada de registro que este experimento aporta."""
        return RegistryEntry(
            run_sha256=self.run_sha256,
            variant_id=self.config.variant_id,
            sharpe_per_session=self.result.sharpe_per_session,
            n_observations=self.result.n_observations,
        )


def record_experiment(
    *,
    runs_root: Path,
    config: ExperimentConfig,
    result: ExperimentResult,
    as_of: datetime,
    write: bool = True,
) -> ExperimentRecord:
    """Registra un experimento bajo `runs/<sha256>/` y devuelve su registro (A11-A14).

    ``write=False`` **no escribe nada** (ni directorios): el CLI lo usa para `--dry-run`.
    El instante entra por parametro explicito y **no** forma parte de la identidad ni de los
    tres ficheros: el mismo experimento registrado con dos `as_of` distintos es el mismo
    experimento y sus bytes no cambian (A12, A14).
    """
    moment = _as_utc(as_of)
    digest = run_sha256(config)
    directory = runs_root / digest
    documents = {
        CONFIG_FILE: _json_file(
            {
                "run_sha256": digest,
                "hash_format": EXPERIMENT_HASH_FORMAT,
                "config": config.to_payload(),
            }
        ),
        RESULT_FILE: _json_file(
            {
                "run_sha256": digest,
                "hash_format": EXPERIMENT_HASH_FORMAT,
                "result": result.to_payload(),
            }
        ),
        SUMMARY_FILE: render_summary(digest=digest, config=config, result=result),
    }
    if not write:
        return ExperimentRecord(
            as_of=moment,
            run_sha256=digest,
            directory=directory,
            config=config,
            result=result,
            outcome=None,
            written=False,
        )
    directory.mkdir(parents=True, exist_ok=True)
    outcomes = tuple(_write_immutable(directory / name, text) for name, text in documents.items())
    outcome = WriteOutcome.CREATED if WriteOutcome.CREATED in outcomes else WriteOutcome.UNCHANGED
    return ExperimentRecord(
        as_of=moment,
        run_sha256=digest,
        directory=directory,
        config=config,
        result=result,
        outcome=outcome,
        written=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# El registro: de donde salen `n_trials` y `sr_variance` (A10, A12, A22)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """Una variante probada, tal como la ve el registro."""

    run_sha256: str
    variant_id: str
    sharpe_per_session: float
    n_observations: int

    def to_payload(self) -> dict[str, object]:
        """La entrada como objeto JSON puro: es lo que se hashea para `registry_sha256`."""
        return {
            "run_sha256": self.run_sha256,
            "variant_id": self.variant_id,
            "sharpe_per_session": self.sharpe_per_session,
            "n_observations": self.n_observations,
        }


@dataclass(frozen=True, slots=True)
class Registry:
    """El registro de experimentos y su huella: **la** fuente de `n_trials` (A10)."""

    entries: tuple[RegistryEntry, ...]
    registry_sha256: str

    @property
    def n_trials(self) -> int:
        """Numero **real** de variantes probadas: no se inventa ni se inyecta."""
        return len(self.entries)

    @property
    def sr_variance(self) -> float:
        """`V[SR]` muestral de los Sharpe del registro (`ddof = 1`); falla con una sola."""
        return variant_sharpe_variance([entry.sharpe_per_session for entry in self.entries])

    @property
    def variant_ids(self) -> tuple[str, ...]:
        """Los `variant_id` que produjeron el `n_trials` publicado, ordenados."""
        return tuple(entry.variant_id for entry in self.entries)

    def to_payload(self) -> dict[str, object]:
        """El bloque del registro: `n_trials`, `sr_variance`, la huella y las entradas."""
        sample = [entry.sharpe_per_session for entry in self.entries]
        variance: float | None
        try:
            variance = self.sr_variance
        except InsufficientTrialsError:
            variance = None
        return {
            "n_trials": self.n_trials,
            "sr_variance": variance,
            "sr_variance_state": "not_evaluable" if variance is None else "evaluated",
            "registry_sha256": self.registry_sha256,
            "variant_ids": list(self.variant_ids),
            "sharpe_per_session_max": max(sample),
            "sharpe_per_session_min": min(sample),
            "hash_format": REGISTRY_HASH_FORMAT,
            "entries": [entry.to_payload() for entry in self.entries],
            "rule": (
                "`n_trials` y `sr_variance` se derivan del registro; no hay `--n-trials` ni "
                "campo del payload que los fije a mano (A10)"
            ),
            "reason": REGISTRY_NOT_EVALUABLE if variance is None else None,
            "follow_up": list(NOT_EVALUABLE_FOLLOW_UP) if variance is None else [],
        }


def _entry_from_disk(runs_root: Path, digest: str) -> RegistryEntry:
    """Lee una entrada del registro y comprueba que es coherente con su identidad (A12)."""
    config_path = runs_root / digest / CONFIG_FILE
    result_path = runs_root / digest / RESULT_FILE
    for path in (config_path, result_path):
        if not path.is_file():
            raise RegistryIntegrityError(
                f"{path} no existe: `runs/{digest}/` tiene que traer `config.json` y "
                "`result.json` (A11)"
            )
    try:
        config_document = cast("dict[str, object]", json.loads(config_path.read_text("utf-8")))
        result_document = cast("dict[str, object]", json.loads(result_path.read_text("utf-8")))
    except json.JSONDecodeError as error:
        raise RegistryIntegrityError(f"{config_path}: JSON no valido ({error})") from error
    stored = config_document.get("run_sha256")
    recomputed = _digest(cast("Mapping[str, object]", config_document.get("config") or {}))
    if stored != digest or recomputed != digest:
        raise RegistryIntegrityError(
            f"`runs/{digest}/config.json` no es coherente con su identidad (declara {stored!r} y "
            f"su contenido hashea a {recomputed}): el registro es tambien un detector de "
            "no-determinismo (A12)"
        )
    if result_document.get("run_sha256") not in {None, digest}:
        raise RegistryIntegrityError(
            f"`runs/{digest}/result.json` declara otra identidad "
            f"({result_document.get('run_sha256')!r}) (A14)"
        )
    result_block = cast("Mapping[str, object]", result_document.get("result") or {})
    sharpe = result_block.get("sharpe_per_session")
    observations = result_block.get("n_observations")
    if not isinstance(sharpe, (int, float)) or isinstance(sharpe, bool):
        raise RegistryIntegrityError(f"`runs/{digest}/result.json` no publica un Sharpe numerico")
    if not isinstance(observations, int) or isinstance(observations, bool):
        raise RegistryIntegrityError(
            f"`runs/{digest}/result.json` no publica `n_observations` entero"
        )
    config_block = cast("Mapping[str, object]", config_document.get("config") or {})
    variant_id = config_block.get("variant_id")
    if not isinstance(variant_id, str):
        raise RegistryIntegrityError(f"`runs/{digest}/config.json` no declara `variant_id`")
    return RegistryEntry(
        run_sha256=digest,
        variant_id=variant_id,
        sharpe_per_session=float(sharpe),
        n_observations=observations,
    )


def load_registry(runs_root: Path, *, extra: Sequence[ExperimentRecord] = ()) -> Registry:
    """Carga el registro de `runs_root` y le anade los experimentos de esta corrida (A10).

    La union se deduplica por `run_sha256` y se ordena por esa misma identidad, asi que el
    resultado **no** depende del orden de lectura ni de la ruta: la misma lista de
    experimentos da el mismo `registry_sha256` y el mismo `n_trials`. Registro vacio es error
    tipado, nunca un `n_trials` inventado.
    """
    found: dict[str, RegistryEntry] = {}
    if runs_root.is_dir():
        for child in sorted(runs_root.iterdir(), key=lambda path: path.name):
            if child.is_dir() and not child.name.startswith("."):
                found[child.name] = _entry_from_disk(runs_root, child.name)
    for record in extra:
        found.setdefault(record.run_sha256, record.to_entry())
    if not found:
        raise EmptyRegistryError(
            f"el registro de `{runs_root}` esta vacio: sin variantes probadas no hay `n_trials` "
            "ni `V[SR]`, y no se inventan (A10, A22)"
        )
    entries = tuple(found[digest] for digest in sorted(found))
    return Registry(
        entries=entries,
        registry_sha256=_digest({"entries": [entry.to_payload() for entry in entries]}),
    )


def require_trials_match_registry(*, n_trials: int, sr_variance: float, registry: Registry) -> None:
    """Exige que `n_trials`/`sr_variance` sean los que deriva el registro (A19).

    Es la puerta que hace imposible inyectar un numero de variantes «estimado»: quien traiga
    un `n_trials` propio tiene que ser **el mismo** que el del registro, o es error tipado.
    """
    try:
        derived_variance = registry.sr_variance
    except InsufficientTrialsError as error:
        raise TrialsMismatchError(
            f"el registro no permite derivar `V[SR]`: {error} (A19, A22)"
        ) from error
    if n_trials != registry.n_trials or sr_variance != derived_variance:
        raise TrialsMismatchError(
            f"`n_trials`/`sr_variance` no son los del registro: llego ({n_trials}, "
            f"{sr_variance!r}) y el registro deriva ({registry.n_trials}, {derived_variance!r}). "
            "El numero de variantes no se inventa (A10, A19)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Los dos calculos y su agregado (A4, A18, A22)
# ─────────────────────────────────────────────────────────────────────────────
def _not_evaluable(*, calculation: str, reason: str) -> dict[str, object]:
    """Bloque declarado de un calculo no evaluable: motivo y seguimiento, ningun numero."""
    return {
        "state": VERDICT_NOT_EVALUABLE,
        "calculation": calculation,
        "verdict": VERDICT_NOT_EVALUABLE,
        "reason": reason,
        "follow_up": list(NOT_EVALUABLE_FOLLOW_UP),
        "note": (
            "no se rellena el hueco con un numero ni se convierte en un «pasa» condicional: "
            "`not_evaluable` nunca se degrada a `pass` (A22)"
        ),
    }


def deflate_block(
    *,
    returns: Sequence[float],
    registry: Registry,
    confidence_level: float = DSR_CONFIDENCE_LEVEL,
) -> dict[str, object]:
    """Bloque del DSR de una variante, con `n_trials` y `sr_variance` **del registro** (A10).

    Los dos parametros no se reciben sueltos: se derivan del registro, y un registro sin
    varianza estimable no produce un numero, produce ``not_evaluable`` con su motivo (A22).
    """
    try:
        trials = registry.n_trials
        variance = registry.sr_variance
        result = deflated_sharpe_ratio(
            returns,
            n_trials=trials,
            sr_variance=variance,
            confidence_level=confidence_level,
        )
    except (
        InsufficientTrialsError,
        InvalidVarianceError,
        DegenerateSeriesError,
        InsufficientObservationsError,
    ) as error:
        return _not_evaluable(calculation="deflated_sharpe_ratio", reason=str(error))
    return {"state": "evaluated", "calculation": "deflated_sharpe_ratio", **result.to_payload()}


def pbo_block(
    *,
    returns_matrix: Sequence[Sequence[float]],
    blocks: int = DEFAULT_BLOCKS,
    sampling_seed: int | None = None,
) -> dict[str, object]:
    """Bloque del PBO por CSCV de una matriz de variantes (A6, A7, A22).

    Un caso degenerado con datos validos (una sola variante, matriz constante) se declara
    ``not_evaluable`` con su motivo en vez de rellenarse; un presupuesto combinatorio
    excedido sin semilla si es error tipado y **no** se captura aqui.
    """
    try:
        result = probability_of_backtest_overfitting(
            returns_matrix, blocks=blocks, sampling_seed=sampling_seed
        )
    except (
        InsufficientVariantsError,
        DegenerateMatrixError,
        InvalidMatrixShapeError,
        InsufficientObservationsError,
    ) as error:
        return _not_evaluable(calculation="probability_of_backtest_overfitting", reason=str(error))
    return {
        "state": "evaluated",
        "calculation": "probability_of_backtest_overfitting",
        **result.to_payload(),
    }


def require_consistent_aggregate(*, gate: str, dsr_half: str, pbo_half: str) -> None:
    """Comprueba que el agregado es **el de la regla de #9**, no otro (A18).

    Dos condiciones: el valor tiene que coincidir con `aggregate_gate` y un ``pass`` exige que
    las dos mitades sean ``pass``. Un agregado que apruebe con una mitad no evaluable es un
    error de consistencia, nunca un resultado valido.
    """
    expected = str(aggregate_gate(dsr_half, pbo_half))
    if expected == HalfResult.PASS.value and (
        dsr_half != HalfResult.PASS.value or pbo_half != HalfResult.PASS.value
    ):
        raise VerdictConsistencyError(
            f"un agregado `pass` exige las dos mitades `pass`; llegaron ({dsr_half}, {pbo_half}) "
            "(A18)"
        )
    if gate != expected:
        raise VerdictConsistencyError(
            f"el agregado declarado ({gate!r}) no es el de `aggregate_gate` de #9 ({expected!r}) "
            f"para las mitades ({dsr_half}, {pbo_half}): {GATE_AGGREGATION_RULE} (A18)"
        )


def aggregate_verdict(*, dsr_verdict: str, pbo_verdict: str) -> str:
    """Agrega los dos veredictos con la regla importada de #9 y la comprueba (A18)."""
    try:
        dsr_half = DSR_HALVES[dsr_verdict]
        pbo_half = PBO_HALVES[pbo_verdict]
    except KeyError as error:
        raise VerdictConsistencyError(
            f"veredicto fuera del vocabulario declarado: {error}; se admite "
            f"{sorted(set(DSR_HALVES) | set(PBO_HALVES))}"
        ) from error
    gate = str(aggregate_gate(dsr_half, pbo_half))
    require_consistent_aggregate(gate=gate, dsr_half=dsr_half, pbo_half=pbo_half)
    return gate


def _gate_block(*, dsr_verdict: str, pbo_verdict: str) -> dict[str, object]:
    """El agregado publicado, con sus mitades traducidas y la regla citada."""
    gate = aggregate_verdict(dsr_verdict=dsr_verdict, pbo_verdict=pbo_verdict)
    return {
        "aggregate": gate,
        "halves": {
            "deflated_sharpe_ratio": DSR_HALVES[dsr_verdict],
            "probability_of_backtest_overfitting": PBO_HALVES[pbo_verdict],
        },
        "verdicts": {
            "deflated_sharpe_ratio": dsr_verdict,
            "probability_of_backtest_overfitting": pbo_verdict,
        },
        "aggregation_rule": GATE_AGGREGATION_RULE,
        "aggregation_source": "cfdtrader.analysis.phase0_report.aggregate_gate (#9)",
        "note": (
            "`not_evaluable` nunca se convierte en `pass`: el agregado solo aprueba con las dos "
            "mitades aprobadas (A18)"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validacion sintetica: ruido puro y senal plantada (A16, A17)
# ─────────────────────────────────────────────────────────────────────────────
def _synthetic_matrix(name: str) -> tuple[tuple[float, ...], ...]:
    """La matriz del experimento sintetico declarado: `noise` o `signal` (A16, A17)."""
    if name == "noise":
        return noise_matrix()
    if name == "signal":
        return signal_matrix()
    raise ExperimentLogError(f"experimento sintetico desconocido: {name!r}")


def _synthetic_config(
    *, name: str, index: int, seed: int, n_observations: int, effect: float
) -> ExperimentConfig:
    """La configuracion registrada de una variante sintetica: todo declarado y reproducible."""
    return ExperimentConfig(
        variant_id=f"{name}-{index:02d}",
        features=("synthetic_random_draw",),
        hyperparameters={
            "kind": name,
            "generator": "numpy.random.RandomState",
            "sigma": SYNTHETIC_SIGMA,
            "effect": effect if index == 0 else 0.0,
            "observations": n_observations,
        },
        seed=seed,
        series_id=SYNTHETIC_SERIES,
        window={"kind": "index", "start": 0, "stop": n_observations},
    )


def _synthetic_records(
    *, runs_root: Path, as_of: datetime, write: bool
) -> dict[str, tuple[ExperimentRecord, ...]]:
    """Registra las variantes de los dos experimentos sinteticos (A11, A16, A17).

    Cada variante probada se anota, que es el punto de `plan.md` §11.4: el registro de ruido
    y el de senal son los que hacen que `n_trials` sea un hecho y no una declaracion de
    intenciones. Con ``write=False`` no se escribe nada: los registros quedan en memoria.
    """
    seeds = {"noise": (NOISE_SEED, 0.0), "signal": (SIGNAL_SEED, SIGNAL_MEAN)}
    built: dict[str, tuple[ExperimentRecord, ...]] = {}
    for name in EXPERIMENT_NAMES:
        matrix = _synthetic_matrix(name)
        seed, effect = seeds[name]
        n_observations = len(matrix)
        records: list[ExperimentRecord] = []
        for index in range(len(matrix[0])):
            column = tuple(row[index] for row in matrix)
            records.append(
                record_experiment(
                    runs_root=runs_root,
                    config=_synthetic_config(
                        name=name,
                        index=index,
                        seed=seed,
                        n_observations=n_observations,
                        effect=effect,
                    ),
                    result=ExperimentResult(
                        sharpe_per_session=sharpe_ratio(column, annualization=1),
                        n_observations=n_observations,
                    ),
                    as_of=as_of,
                    write=write,
                )
            )
        built[name] = tuple(records)
    return built


@dataclass(frozen=True, slots=True)
class ExperimentOutcome:
    """Un experimento sintetico evaluado: su matriz, sus filas y sus dos calculos."""

    name: str
    matrix: tuple[tuple[float, ...], ...]
    records: tuple[ExperimentRecord, ...]
    selected_index: int
    rows: tuple[dict[str, object], ...]
    dsr: dict[str, object]
    pbo: dict[str, object]
    gate: dict[str, object]

    @property
    def selected(self) -> ExperimentRecord:
        """La variante seleccionada por la regla declarada."""
        return self.records[self.selected_index]

    @property
    def verdict(self) -> str:
        """El veredicto del bloque DSR de la variante seleccionada."""
        return str(self.dsr["verdict"])

    def to_payload(self) -> dict[str, object]:
        """El bloque del experimento: la matriz declarada, el seleccionado y sus veredictos."""
        blocks = self.pbo.get("blocks")
        return {
            "experiment": self.name,
            "matrix": {
                "n_observations": len(self.matrix),
                "n_variants": len(self.matrix[0]),
                "blocks": blocks if isinstance(blocks, int) else None,
                "units": PER_SESSION,
            },
            "selection": {
                "rule": SELECTION_RULE,
                "variant_id": self.selected.config.variant_id,
                "run_sha256": self.selected.run_sha256,
                "runs_directory": f"{DEFAULT_RUNS_ROOT}/{self.selected.run_sha256}",
                "index": self.selected_index,
                "sharpe_per_session": self.selected.result.sharpe_per_session,
            },
            "deflated_sharpe_ratio": self.dsr,
            "probability_of_backtest_overfitting": self.pbo,
            "gate": self.gate,
            "variants": [dict(row) for row in self.rows],
        }


def _experiment_outcome(
    *,
    name: str,
    matrix: tuple[tuple[float, ...], ...],
    records: tuple[ExperimentRecord, ...],
    registry: Registry,
    blocks: int,
) -> ExperimentOutcome:
    """Evalua un experimento: fila por variante, DSR de la seleccionada y PBO de la matriz.

    El DSR de **cada** variante usa el `n_trials` y la `sr_variance` del registro (A10); el
    PBO es de la matriz del experimento. El bloque de la variante **seleccionada** es el que
    agrega con el PBO (A18).
    """
    if len(records) != len(matrix[0]):
        raise ExperimentLogError(
            f"el experimento {name!r} tiene {len(matrix[0])} columnas y {len(records)} registros"
        )
    selected_index = select_variant(matrix)
    selected_dsr = deflate_block(
        returns=tuple(row[selected_index] for row in matrix), registry=registry
    )
    pbo = pbo_block(returns_matrix=matrix, blocks=blocks)
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        column = tuple(row[index] for row in matrix)
        dsr = deflate_block(returns=column, registry=registry)
        rows.append(
            {
                "variant_id": record.config.variant_id,
                "runs_directory": f"{DEFAULT_RUNS_ROOT}/{record.run_sha256}",
                "run_sha256": record.run_sha256,
                "sharpe_per_session": record.result.sharpe_per_session,
                "n_observations": record.result.n_observations,
                "registry_sharpe_matches_matrix": dsr.get("sr_observed")
                in {None, record.result.sharpe_per_session},
                "dsr": dsr.get("dsr"),
                "sr0_expected_max": dsr.get("sr0_expected_max"),
                "deflation": dsr.get("deflation"),
                "dsr_verdict": dsr["verdict"],
                "pbo": pbo.get("pbo"),
                "pbo_verdict": pbo["verdict"],
                "verdict": dsr["verdict"],
            }
        )
    return ExperimentOutcome(
        name=name,
        matrix=matrix,
        records=records,
        selected_index=selected_index,
        rows=tuple(rows),
        dsr=selected_dsr,
        pbo=pbo,
        gate=_gate_block(dsr_verdict=str(selected_dsr["verdict"]), pbo_verdict=str(pbo["verdict"])),
    )


# ─────────────────────────────────────────────────────────────────────────────
# El informe: un payload canonico, dos ficheros (A20, A21, A31, A35)
# ─────────────────────────────────────────────────────────────────────────────
def _payload(
    *,
    as_of: datetime,
    registry: Registry,
    outcomes: Sequence[ExperimentOutcome],
) -> dict[str, object]:
    """El payload canonico del informe: tipos JSON puros, unidades declaradas y sin reloj."""
    by_name = {outcome.name: outcome for outcome in outcomes}
    noise, signal = by_name["noise"], by_name["signal"]
    criteria_met = noise.gate["aggregate"] != HalfResult.PASS.value and (
        signal.gate["aggregate"] == HalfResult.PASS.value
    )
    raw: dict[str, object] = {
        "task": "#16",
        "analysis": "cfdtrader.analysis.experiment_log",
        "title": "Correccion por sobreajuste: Deflated Sharpe Ratio y PBO sobre el registro",
        "generated_at": as_of.isoformat(),
        "report_date": as_of.date().isoformat(),
        "hash_format": REPORT_HASH_FORMAT,
        "units": PER_SESSION,
        "basis": BASIS_DECLARED_COST,
        "is_validation": False,
        "clock": {
            "as_of": as_of.isoformat(),
            "rule": (
                "el modulo no lee el reloj: el instante entra por `--as-of` (obligatorio para "
                "escribir) y `generated_at = as_of`; no hay `datetime.now`, `utcnow`, "
                "`date.today` ni `time.time` en el fuente (A15)"
            ),
        },
        "constants": {
            "dsr_confidence_level": DSR_CONFIDENCE_LEVEL,
            "pbo_max": PBO_MAX,
            "default_blocks": DEFAULT_BLOCKS,
            "max_combinations": MAX_COMBINATIONS,
            "threshold_rule": THRESHOLD_RULE,
            "provenance": (
                "`plan.md` §11.4 («PBO < 20 %», Deflated Sharpe con el numero real de pruebas) y "
                "§11.6 («Se fijan antes de ver resultados. No se mueven»): constantes del modulo, "
                "sin opcion de CLI que las mueva (A21)"
            ),
        },
        "registry": registry.to_payload(),
        "synthetic_validation": {
            "rule": NOISE_SIGNAL_RULE,
            "noise": {
                "gate": noise.gate["aggregate"],
                "dsr_verdict": noise.verdict,
                "pbo_verdict": noise.pbo["verdict"],
                "dsr": noise.dsr.get("dsr"),
                "sr0_expected_max": noise.dsr.get("sr0_expected_max"),
                "pbo": noise.pbo.get("pbo"),
                "expected": "no `pass`: el ruido puro no puede aprobar",
            },
            "signal": {
                "gate": signal.gate["aggregate"],
                "dsr_verdict": signal.verdict,
                "pbo_verdict": signal.pbo["verdict"],
                "dsr": signal.dsr.get("dsr"),
                "sr0_expected_max": signal.dsr.get("sr0_expected_max"),
                "pbo": signal.pbo.get("pbo"),
                "expected": "`pass`: la senal plantada tiene que detectarse",
            },
            "criteria_met": criteria_met,
            "seed_provenance": {
                "noise_seed": NOISE_SEED,
                "signal_seed": SIGNAL_SEED,
                "generator": "numpy.random.RandomState",
                "observations": SYNTHETIC_OBSERVATIONS,
                "variants": SYNTHETIC_VARIANTS,
                "sigma": SYNTHETIC_SIGMA,
                "signal_mean": SIGNAL_MEAN,
                "note": (
                    "semillas y tamano del efecto son constantes con nombre; el generador tiene "
                    "el *stream* congelado entre versiones (la eleccion de #15)"
                ),
            },
        },
        "experiments": [outcome.to_payload() for outcome in outcomes],
        "net_metrics": dict(NET_METRICS),
        "boundaries": {
            "cscv_versus_cpcv": CSCV_VERSUS_CPCV,
            "final_holdout": HOLDOUT_BOUNDARY,
            "store_divergence": STORE_DIVERGENCE,
            "overfitting_does_not_do": [dict(entry) for entry in OVERFITTING_DOES_NOT_DO],
            "follow_ups": [dict(entry) for entry in FOLLOW_UPS],
        },
        "limitations": [
            (
                "el registro crece con una carpeta por identidad y hoy nada lo poda: la "
                "retencion de `ops.*` (y de `runs/`) es #44"
            ),
            (
                "el PBO y el DSR se calculan sobre **retorno declarado**: mientras el "
                "*slippage* sea un supuesto (#62) y `R` no este decidido (#60), esto es una "
                "correccion por sobreajuste, no una validacion de la estrategia"
            ),
            (
                "la rejilla de bloques es una sola (`blocks = 16`): la sensibilidad del PBO al "
                "numero de bloques no se publica aqui"
            ),
            (
                "las series son sinteticas y las variantes de la validacion no son modelos "
                "entrenados: el uso con variantes **reales** es Fase 2 (#28, #29)"
            ),
        ],
    }
    return _json_object(raw, where="payload")


@dataclass(frozen=True, slots=True)
class OverfittingReport:
    """El informe: payload canonico, su hash y los objetos que lo produjeron."""

    as_of: datetime
    report_date: date
    payload: dict[str, object]
    report_sha256: str
    registry: Registry
    experiments: tuple[ExperimentOutcome, ...]

    @property
    def report_stem(self) -> str:
        """Nombre base del informe: ``overfitting_<AAAA-MM-DD>``."""
        return f"{REPORT_PREFIX}_{self.report_date.isoformat()}"

    def json_text(self) -> str:
        """JSON determinista: mismo registro y mismo ``as_of`` ⇒ mismo texto byte a byte."""
        published: dict[str, object] = {**self.payload, "report_sha256": self.report_sha256}
        return json.dumps(published, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    def write(self, directory: Path) -> tuple[Path, Path]:
        """Escribe el informe en JSON y Markdown. Solo escribe quien lo pida (el CLI)."""
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{self.report_stem}.json"
        markdown_path = directory / f"{self.report_stem}.md"
        json_path.write_text(self.json_text(), encoding="utf-8")
        markdown_path.write_text(render_markdown(self), encoding="utf-8")
        return json_path, markdown_path


def _table_row(cells: Sequence[object]) -> str:
    """Una fila de tabla Markdown con los textos ya formateados."""
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _number(value: object) -> str:
    """Un numero del payload para el informe legible; un hueco sigue siendo un hueco."""
    return "`null`" if value is None else repr(value)


def render_markdown(report: OverfittingReport) -> str:
    """El informe legible, construido **del mismo payload** que el JSON (A35).

    Cita el directorio `runs/<hash>/` de cada variante en vez de duplicar el JSON, y publica
    una fila por variante con el Sharpe por sesion, el DSR, el ``SR0``, el PBO del experimento
    y su veredicto explicito.
    """
    payload = report.payload
    validation = cast("dict[str, object]", payload["synthetic_validation"])
    registry = cast("dict[str, object]", payload["registry"])
    experiments = cast("list[dict[str, object]]", payload["experiments"])
    lines = [
        "# Correccion por sobreajuste (tarea #16)",
        "",
        f"- `report_sha256`: `{report.report_sha256}`",
        f"- `generated_at`: `{payload['generated_at']}`",
        f"- unidades: `{payload['units']}` (por sesion, nunca anualizadas)",
        f"- base del retorno: `{payload['basis']}` · `is_validation` = "
        f"`{str(payload['is_validation']).lower()}`",
        f"- registro: **{registry['n_trials']}** variantes probadas · "
        f"`registry_sha256` = `{registry['registry_sha256']}`",
        "",
        "## Constantes pre-registradas",
        "",
        "| constante | valor |",
        "|---|---|",
    ]
    constants = cast("dict[str, object]", payload["constants"])
    for key in ("dsr_confidence_level", "pbo_max", "default_blocks", "max_combinations"):
        lines.append(_table_row((f"`{key}`", _number(constants[key]))))
    lines.extend(
        [
            "",
            f"> {constants['threshold_rule']}",
            "",
            "## Validacion sintetica",
            "",
            f"> {validation['rule']}",
            "",
            "| experimento | DSR | `SR0` | veredicto DSR | PBO | veredicto PBO | agregado "
            "| esperado |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for name in EXPERIMENT_NAMES:
        block = cast("dict[str, object]", validation[name])
        lines.append(
            _table_row(
                (
                    f"`{name}`",
                    _number(block["dsr"]),
                    _number(block["sr0_expected_max"]),
                    f"`{block['dsr_verdict']}`",
                    _number(block["pbo"]),
                    f"`{block['pbo_verdict']}`",
                    f"`{block['gate']}`",
                    str(block["expected"]),
                )
            )
        )
    lines.append("")
    lines.append(f"- criterios de oro cumplidos: `{str(validation['criteria_met']).lower()}`")
    lines.extend(["", "## Variantes probadas", ""])
    for experiment in experiments:
        name = experiment["experiment"]
        matrix = cast("dict[str, object]", experiment["matrix"])
        selected = cast("dict[str, object]", experiment["selection"])
        rows = cast("list[dict[str, object]]", experiment["variants"])
        pbo = cast("dict[str, object]", experiment["probability_of_backtest_overfitting"])
        lines.extend(
            [
                f"### `{name}` — {matrix['n_observations']} observaciones x "
                f"{matrix['n_variants']} variantes · bloques = {matrix['blocks']}",
                "",
                f"- seleccionada: `{selected['variant_id']}` en "
                f"`{selected['runs_directory']}` ({SELECTION_RULE})",
                f"- PBO del experimento: {_number(pbo['pbo'])} (`{pbo['verdict']}`)",
                "",
                "| variante | `run_sha256` | Sharpe/sesion | DSR | `SR0` | PBO | veredicto |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for row in rows:
            lines.append(
                _table_row(
                    (
                        f"`{row['variant_id']}`",
                        f"`{row['runs_directory']}`",
                        _number(row["sharpe_per_session"]),
                        _number(row["dsr"]),
                        _number(row["sr0_expected_max"]),
                        _number(row["pbo"]),
                        f"`{row['verdict']}`",
                    )
                )
            )
        lines.append("")
    lines.extend(["## Agregado", ""])
    for experiment in experiments:
        gate = cast("dict[str, object]", experiment["gate"])
        halves = cast("dict[str, object]", gate["halves"])
        lines.extend(
            [
                f"- `{experiment['experiment']}`: **`{gate['aggregate']}`** "
                f"(DSR `{halves['deflated_sharpe_ratio']}`, PBO "
                f"`{halves['probability_of_backtest_overfitting']}`)",
            ]
        )
    net = cast("dict[str, object]", payload["net_metrics"])
    notes = payload["limitations"]
    boundaries = cast("dict[str, object]", payload["boundaries"])
    lines.extend(
        [
            "",
            "## Metricas netas y fronteras declaradas",
            "",
            f"- metricas netas: `{net['state']}` — {net['reason']} (seguimiento "
            f"{', '.join(cast('list[str]', net['follow_up']))})",
            f"- CSCV frente a CPCV: {boundaries['cscv_versus_cpcv']}",
            f"- *holdout* final: {boundaries['final_holdout']}",
            f"- almacen: {boundaries['store_divergence']}",
            "",
            "### Limitaciones",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in cast("list[str]", notes))
    lines.extend(["", "### Seguimientos", ""])
    lines.extend(
        f"- `{entry['issue']}` ({entry['id']}): {entry['reason']}"
        for entry in cast("list[dict[str, str]]", boundaries["follow_ups"])
    )
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Ejecucion (A31)
# ─────────────────────────────────────────────────────────────────────────────
def analyse(
    *, runs_root: Path, reports_dir: Path, as_of: datetime, write: bool = True
) -> OverfittingReport:
    """Registra la validacion sintetica, deriva `n_trials` del registro y emite el informe.

    ``as_of`` es **obligatorio** y es el unico instante de la corrida: ninguna ruta consulta
    el reloj. ``write=False`` no escribe **nada**: ni `runs/`, ni el informe.
    """
    moment = _as_utc(as_of)
    matrices = {name: _synthetic_matrix(name) for name in EXPERIMENT_NAMES}
    records = _synthetic_records(runs_root=runs_root, as_of=moment, write=write)
    flattened = tuple(record for name in EXPERIMENT_NAMES for record in records[name])
    registry = load_registry(runs_root, extra=flattened)
    outcomes = tuple(
        _experiment_outcome(
            name=name,
            matrix=matrices[name],
            records=records[name],
            registry=registry,
            blocks=DEFAULT_BLOCKS,
        )
        for name in EXPERIMENT_NAMES
    )
    payload = _payload(as_of=moment, registry=registry, outcomes=outcomes)
    report = OverfittingReport(
        as_of=moment,
        report_date=moment.astimezone(UTC).date(),
        payload=payload,
        report_sha256=_digest(payload),
        registry=registry,
        experiments=outcomes,
    )
    if write:
        json_path, markdown_path = report.write(reports_dir)
        logger.info("informe de sobreajuste: {} y {}", json_path, markdown_path)
    return report


def _parse_as_of(value: str | None) -> datetime:
    """El instante declarado de la CLI: **obligatorio**, y nunca tomado del reloj (A15)."""
    if value is None:
        raise MissingAsOfError(
            "falta --as-of: registrar experimentos y escribir el informe exige un instante "
            "declarado (el modulo no lee el reloj) y sin el **no se escribe ningun fichero**"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise InvalidAsOfError(f"--as-of no es un ISO-8601 valido: {error}") from error
    return _as_utc(parsed)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del informe de sobreajuste.

    Codigos de salida: ``0`` = registro e informe escritos (aunque el veredicto sea `fail`,
    que es un resultado legitimo y declarado); ``2`` = falta o no es valido ``--as-of``, o el
    registro no permite calcular ⇒ **no se escribe nada** y el motivo sale por ``stderr``.
    """
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description=(
            "Registro de experimentos en runs/<hash>/ y correccion por sobreajuste "
            "(DSR + PBO) sobre la validacion sintetica"
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help=f"raiz del registro de experimentos (por defecto {DEFAULT_RUNS_ROOT}/)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help=f"directorio de informes (por defecto {DEFAULT_REPORTS_DIR})",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="instante declarado ISO-8601, obligatorio para escribir (el modulo no lee el reloj)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="calcula la validacion sin escribir nada: ni runs/, ni informe",
    )
    args = parser.parse_args(argv)

    try:
        moment = _parse_as_of(cast("str | None", args.as_of))
    except ExperimentLogError as error:
        print(f"no se puede emitir el informe de sobreajuste: {error}", file=sys.stderr)
        return 2
    runs_root = Path(args.runs_root) if args.runs_root is not None else Path(DEFAULT_RUNS_ROOT)
    reports_dir = (
        Path(args.reports_dir) if args.reports_dir is not None else Path(DEFAULT_REPORTS_DIR)
    )
    try:
        report = analyse(
            runs_root=runs_root,
            reports_dir=reports_dir,
            as_of=moment,
            write=not args.dry_run,
        )
    except (ExperimentLogError, OverfittingError) as error:
        print(f"no se puede emitir el informe de sobreajuste: {error}", file=sys.stderr)
        return 2

    validation = cast("dict[str, object]", report.payload["synthetic_validation"])
    logger.info(
        "sobreajuste: {} variantes en el registro; ruido = {} y senal = {}; report_sha256 = {}",
        report.registry.n_trials,
        cast("dict[str, object]", validation["noise"])["gate"],
        cast("dict[str, object]", validation["signal"])["gate"],
        report.report_sha256,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
