"""Informe de Fase 1 y puerta de salida (`tasks.md`, tarea 18) — tarea #18.

Consume el artefacto que dejo **#69** (`data/derived/reports/phase1_backtest_<AAAA-MM-DD>.json`,
con los seis baselines de #14 ya corridos sobre la historia real), **audita el arnes con
evidencia propia** y publica el **veredicto de la puerta de salida de la Fase 1**.

Que **hace** este modulo:

- **consume** el artefacto de #69 (ruta relativa y `sha256` del fichero) y **copia** su tabla
  de baselines: no la recalcula ni la reescribe (A6);
- **audita** el arnes produciendo la evidencia aqui mismo: determinismo del motor re-ejecutado
  (`harness_audit.determinism`), no-*look-ahead* observable mutando una sesion posterior
  (`harness_audit.lookahead`), identidades de recuento (`harness_audit.conservation`), coste
  declarado reproducido por importacion (`harness_audit.costs`) y purga/embargo de #12
  publicados tal cual con `label_horizon = 0` declarado **no-op** (`harness_audit.purge_embargo`);
- **aplica** la puerta de forma **mecanica** reutilizando `aggregate_gate` de #9 y publica un
  veredicto **parcial**: la mitad «es determinista el motor?» se **mide** hoy; la mitad «bate
  *siempre largo* a todos los baselines de forma significativa?» es hoy **`not_evaluable`**,
  porque `pnl_net_pct` es `null` en el **100 %** de las operaciones (supuesto de #64) y
  `calculate_metrics` de #15 lanza `MetricsInputError`.

Que **no** hace, y por tanto no puede inventar:

- **no** vuelve a correr los baselines sobre la historia (eso es #69, cerrada): la tabla viaja
  **copiada** y marcada con su procedencia (`evidence: "artifact"`);
- **no** fabrica un *slippage* que cierre el total, **no** escribe un valor no medido como `0`
  (regla «`null != 0`») y **no** llama «validacion» a una tabla de coste declarado (A18);
- **no** publica ninguna metrica neta (Sharpe, Sortino, valor esperado, drawdown, intervalos):
  se declara `not_computable` con su motivo, su `where` y su `follow_up` (A31);
- **no** escribe *sizing* ni deriva el nocional del capital, de `R` o del apalancamiento (A33).

Un `not_evaluable` **nunca** se convierte en un `pass`: convertirlo en un «adelante» es el error
mas caro que puede cometer este informe, y la agregacion de #9 lo hace imposible por
construccion.

**Reloj prohibido** (A2): ninguna ruta consulta el reloj del sistema; el instante entra por
`--as-of`, obligatorio para escribir, que sale con codigo 2 y sin tocar disco cuando falta o no
es ISO-8601 (A3). A diferencia de :mod:`cfdtrader.analysis.phase0_report` —cuya CLI acepta
`--now` y recurre al reloj del sistema cuando no se le pasa—, aqui el instante declarado es la
unica fuente de tiempo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, cast

import polars as pl
from loguru import logger

from cfdtrader.analysis import backtest_report
from cfdtrader.analysis.phase0_report import (
    FILE_SELECTION_RULE,
    GATE_AGGREGATION_RULE,
    RECOMMENDATION_CONSISTENCY_RULE,
    AmbiguousArtifactError,
    ArtifactClass,
    BlockerCode,
    GateVerdict,
    HalfResult,
    MissingArtifactError,
    Recommendation,
    aggregate_gate,
    recommendation_is_consistent,
    select_artifact,
)
from cfdtrader.backtest.costs import (
    DECLARED_OVERNIGHT_REASON,
    UNIT_BP,
    UNIT_PCT,
    UNIT_PCT_PER_NIGHT,
    UNIT_USD,
    CostBreakdown,
    CostModel,
    Side,
    SlippageParameter,
    cost_breakdown,
    declared_cost_model,
    declared_slippage_assumption,
)
from cfdtrader.backtest.engine import canonical_text
from cfdtrader.backtest.metrics import MetricsInputError, calculate_metrics
from cfdtrader.data.calendar import load_calendar
from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.store import Store

__all__ = [
    "CLOCK_DIVERGENCE",
    "EVIDENCE_ARTIFACT",
    "EVIDENCE_RERUN",
    "FOLLOW_UPS",
    "GATE_AGGREGATION_SOURCE",
    "HALF_ALWAYS_LONG",
    "HALF_DETERMINISM",
    "HOW_TO_CLOSE",
    "INPUT_CLASS",
    "INPUT_PATTERN",
    "NOTIONAL_PROVENANCE",
    "REPORT_DOES_NOT_DO",
    "REPORT_HASH_FORMAT",
    "REPORT_PREFIX",
    "AmbiguousInputArtifactError",
    "AuditInvariantError",
    "ConservationError",
    "InputArtifact",
    "InvalidAsOfError",
    "MalformedInputArtifactError",
    "MissingAsOfError",
    "MissingInputArtifactError",
    "MissingInputFieldError",
    "Phase1Report",
    "Phase1ReportError",
    "VerdictError",
    "analyse",
    "audit_determinism",
    "audit_lookahead",
    "conservation_audit",
    "costs_audit",
    "load_input_artifact",
    "main",
    "mechanical_gate",
    "render_markdown",
]

#: Prefijo del informe: ``phase1_report_<AAAA-MM-DD>.{json,md}``.
REPORT_PREFIX: Final[str] = "phase1_report"

#: Prefijo del artefacto **consumido** (el de #69). Se lee de #69 para no copiarlo.
INPUT_PATTERN: Final[str] = f"{backtest_report.REPORT_PREFIX}_*.json"

#: La clase de artefacto de entrada, con el mismo contrato que las de #9 (A5, A27).
INPUT_CLASS: Final[ArtifactClass] = ArtifactClass(
    kind=backtest_report.REPORT_PREFIX,
    pattern=INPUT_PATTERN,
    description="informe del arnes de Fase 1 (#69) con los seis baselines ya corridos",
)

#: Procedencia de un bloque: **producido/medido en esta corrida**.
EVIDENCE_RERUN: Final[str] = "re-run"

#: Procedencia de un bloque: **copiado del artefacto de #69**, no medido aqui.
EVIDENCE_ARTIFACT: Final[str] = "artifact"

#: Nombres de las dos mitades de la puerta (A12, A13).
HALF_DETERMINISM: Final[str] = "determinism"
HALF_ALWAYS_LONG: Final[str] = "always_long"

#: De donde sale la agregacion mecanica publicada (A14): se **importa**, no se reescribe.
GATE_AGGREGATION_SOURCE: Final[str] = (
    "cfdtrader.analysis.phase0_report.aggregate_gate (regla de #9, reutilizada tal cual)"
)

#: Procedencia del nocional declarado, leida de #69 (nunca derivada de `R` ni del capital).
NOTIONAL_PROVENANCE: Final[str] = (
    "nocional plano declarado de #69 (`NOTIONAL_USD`); no se deriva del capital, de `R` ni del "
    "apalancamiento (#27/#60) y este modulo no calcula *sizing* (A33)"
)

#: Formato estable del ``report_sha256`` (A20).
REPORT_HASH_FORMAT: Final[str] = (
    "sha256(UTF-8) de cfdtrader.backtest.engine.canonical_text(payload) —el canonico de #13, no "
    "un serializador propio—, es decir json.dumps(payload, sort_keys=True, "
    "separators=(',', ':'), ensure_ascii=False), **sin** la clave report_sha256 (un informe no "
    "se hashea a si mismo). Decimal viaja como cadena decimal exacta (format(d, 'f')) y float "
    "via repr; nan e inf estan prohibidos"
)

#: Divergencia declarada con la CLI de #9 (A3), **sin** nombrar el reloj literal que A2 prohibe.
CLOCK_DIVERGENCE: Final[str] = (
    "la CLI de `cfdtrader.analysis.phase0_report` acepta `--now` y, cuando no se le pasa, "
    "recurre al reloj del sistema; aqui eso esta **prohibido en todas las rutas**: el instante "
    "declarado (`--as-of`) es obligatorio para escribir y sin el la CLI sale con 2 sin tocar "
    "disco (A2, A3)"
)

#: Factor con el que se muta la sesion posterior de la auditoria de no-*look-ahead* (A11). No
#: es una cifra de negocio: solo tiene que ser distinta de 1 para que la mutacion sea observable.
MUTATION_FACTOR: Final[float] = 1.5

#: Columnas que se mutan en la sesion `j`: **nunca** `open` (eso cambiaria la membresia de la
#: muestra limpia de #52) ni las columnas derivadas (`prev_close`, `open_stale`), que son el
#: contrato que la regla limpia ya dejo calculado (A11).
MUTATED_COLUMNS: Final[tuple[str, ...]] = ("close", "high", "low")

#: Cuanto se publica del detalle de sesiones cuyo resultado cambio con la mutacion (A11): la
#: lista completa no aporta y crece con la muestra; el recuento si.
CHANGED_SAMPLE_LIMIT: Final[int] = 12


# ─────────────────────────────────────────────────────────────────────────────
# Errores declarados: un hueco nunca se rellena con un valor por defecto (A23)
# ─────────────────────────────────────────────────────────────────────────────
class Phase1ReportError(Exception):
    """Error declarado del informe de Fase 1: nunca se rellena el hueco."""


class MissingInputArtifactError(Phase1ReportError):
    """No hay artefacto de entrada de #69 que consumir."""


class AmbiguousInputArtifactError(Phase1ReportError):
    """Dos o mas artefactos de #69 resuelven a la misma fecha: no se elige a dedo."""


class MalformedInputArtifactError(Phase1ReportError):
    """El artefacto de entrada no es JSON legible o no es un objeto."""


class MissingInputFieldError(Phase1ReportError):
    """El artefacto no trae un campo obligatorio, o lo trae con un tipo inesperado."""


class ConservationError(Phase1ReportError):
    """Una identidad de conservacion no cuadra: una sesion no puede desaparecer (A10)."""


class AuditInvariantError(Phase1ReportError):
    """Una invariante declarada de la auditoria no se cumple (coste, plan o mutacion)."""


class MissingAsOfError(Phase1ReportError):
    """Falta el instante declarado: escribir el informe lo exige (A3)."""


class InvalidAsOfError(Phase1ReportError):
    """El instante declarado no es ISO-8601 (A3)."""


class VerdictError(Phase1ReportError):
    """El veredicto pedido rompe una regla declarada (vocabulario o consistencia)."""


# ─────────────────────────────────────────────────────────────────────────────
# Lectura tipada del artefacto: un campo que falta es un error, nunca un cero
# ─────────────────────────────────────────────────────────────────────────────
def _mapping(node: object, *, where: str) -> dict[str, object]:
    """Vista tipada de un nodo que tiene que ser un objeto JSON."""
    if not isinstance(node, dict):
        raise MissingInputFieldError(
            f"{where}: se espera un objeto JSON, no {type(node).__name__} (A23)"
        )
    return cast("dict[str, object]", node)


def _require(node: object, *keys: str) -> object:
    """Valor anidado obligatorio del artefacto; su ausencia es un error tipado (A23)."""
    current: object = node
    path = ".".join(keys)
    for index, key in enumerate(keys):
        mapping = _mapping(current, where=".".join(keys[:index]) or "el artefacto")
        if key not in mapping:
            raise MissingInputFieldError(
                f"falta el campo obligatorio `{path}` en el artefacto de #69: no se rellena con "
                "un valor por defecto (A23)"
            )
        current = mapping[key]
    return current


def _require_str(node: object, *keys: str) -> str:
    """Campo obligatorio de tipo cadena."""
    value = _require(node, *keys)
    if not isinstance(value, str):
        raise MissingInputFieldError(
            f"`{'.'.join(keys)}` deberia ser una cadena, no {type(value).__name__} (A23)"
        )
    return value


def _require_int(node: object, *keys: str) -> int:
    """Campo obligatorio de tipo entero (los `bool` no cuentan como enteros)."""
    value = _require(node, *keys)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MissingInputFieldError(
            f"`{'.'.join(keys)}` deberia ser un entero, no {type(value).__name__} (A23)"
        )
    return value


def _require_bool(node: object, *keys: str) -> bool:
    """Campo obligatorio de tipo booleano."""
    value = _require(node, *keys)
    if not isinstance(value, bool):
        raise MissingInputFieldError(
            f"`{'.'.join(keys)}` deberia ser booleano, no {type(value).__name__} (A23)"
        )
    return value


def _require_list(node: object, *keys: str) -> list[object]:
    """Campo obligatorio de tipo lista."""
    value = _require(node, *keys)
    if not isinstance(value, list):
        raise MissingInputFieldError(
            f"`{'.'.join(keys)}` deberia ser una lista, no {type(value).__name__} (A23)"
        )
    return cast("list[object]", value)


def _require_rows(node: object, *keys: str) -> list[dict[str, object]]:
    """Lista obligatoria de objetos JSON."""
    raw = _require_list(node, *keys)
    rows: list[dict[str, object]] = []
    for index, item in enumerate(raw):
        rows.append(_mapping(item, where=f"{'.'.join(keys)}[{index}]"))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# El artefacto de entrada (#69), seleccionado con el contrato de #9 (A5, A6)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class InputArtifact:
    """El artefacto consumido, con su procedencia y sus huellas."""

    kind: str
    path: Path
    relative_path: str
    artifact_date: str
    sha256: str
    payload: dict[str, object]
    json_bytes: bytes
    markdown_bytes: bytes

    @property
    def generated_at(self) -> str:
        """Instante declarado del artefacto de #69."""
        return _require_str(self.payload, "generated_at")

    @property
    def report_sha256(self) -> str:
        """`report_sha256` que declara el artefacto de #69."""
        return _require_str(self.payload, "report_sha256")

    def provenance(self) -> dict[str, object]:
        """Bloque de procedencia que publica el informe (A6, A32)."""
        return {
            "kind": self.kind,
            "path": self.relative_path,
            "sha256": self.sha256,
            "sha256_of": "fichero .json",
            "artifact_date": self.artifact_date,
            "report_sha256": self.report_sha256,
            "generated_at": self.generated_at,
            "evidence": EVIDENCE_ARTIFACT,
            "selection_rule": FILE_SELECTION_RULE,
            "note": (
                "el artefacto de #69 se **consume**: se publica su tabla copiada y su huella, y "
                "no se recalcula ni se reescribe (A6)"
            ),
        }


def _relative_path(path: Path, *, store: Store) -> str:
    """Ruta del artefacto relativa a la raiz del almacen cuando cuelga de ella (A6)."""
    root = store.root
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_input_artifact(reports_dir: Path, *, store: Store) -> InputArtifact:
    """Selecciona, lee y valida el artefacto de #69. Solo lee: no escribe nada (A5, A21).

    La seleccion **reutiliza** :func:`cfdtrader.analysis.phase0_report.select_artifact`: cero
    candidatos y ambiguedad son errores tipados propios, y nunca se elige un fichero a dedo.
    """
    try:
        path = select_artifact(reports_dir, INPUT_CLASS)
    except MissingArtifactError as error:
        raise MissingInputArtifactError(f"no hay artefacto de #69 que consumir: {error}") from error
    except AmbiguousArtifactError as error:
        raise AmbiguousInputArtifactError(
            f"hay mas de un artefacto de #69 con la misma fecha: {error}"
        ) from error

    raw = path.read_bytes()
    try:
        loaded: Any = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MalformedInputArtifactError(
            f"el artefacto {path.name} no es JSON legible: {error} (A23)"
        ) from error
    if not isinstance(loaded, dict):
        raise MalformedInputArtifactError(
            f"el artefacto {path.name} no es un objeto JSON: el informe consume objetos con "
            "campos (A23)"
        )

    markdown_path = path.with_suffix(".md")
    if not markdown_path.is_file():
        raise MissingInputArtifactError(
            f"falta el companero {markdown_path.name} del artefacto {path.name}: el entregable de "
            "#69 es un par `.json`/`.md` y la auditoria compara los dos byte a byte (A6, A7)"
        )

    stem = path.stem
    prefix = f"{backtest_report.REPORT_PREFIX}_"
    return InputArtifact(
        kind=INPUT_CLASS.kind,
        path=path,
        relative_path=_relative_path(path, store=store),
        artifact_date=stem[len(prefix) :] if stem.startswith(prefix) else stem,
        sha256=hashlib.sha256(raw).hexdigest(),
        payload=cast("dict[str, object]", loaded),
        json_bytes=raw,
        markdown_bytes=markdown_path.read_bytes(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades deterministas
# ─────────────────────────────────────────────────────────────────────────────
def _as_utc(value: datetime) -> datetime:
    """Normaliza el instante declarado a UTC; sin zona se interpreta como UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _sha256_bytes(raw: bytes) -> str:
    """sha256 de unos bytes, en hexadecimal."""
    return hashlib.sha256(raw).hexdigest()


def _dec(value: Decimal, quantum: str) -> str:
    """``Decimal`` como cadena exacta con el numero de decimales pedido, sin cero negativo."""
    step = Decimal(quantum)
    return format((abs(value) if value == 0 else value).quantize(step), "f")


def _digest(payload: Mapping[str, object]) -> str:
    """``report_sha256``: el sha256 del texto canonico de #13, sin la clave del hash (A20)."""
    return _sha256_bytes(canonical_text(payload).encode("utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Auditoria: determinismo del motor (A7, A12)
# ─────────────────────────────────────────────────────────────────────────────
def audit_determinism(
    *, expected: Mapping[str, str], observed: Mapping[str, str], command: str, as_of: str
) -> dict[str, object]:
    """Compara las huellas del artefacto con las de la re-ejecucion (A7).

    Funcion **pura**: entra el mapa de huellas esperado (del artefacto de #69) y el observado
    (de la re-ejecucion del motor en esta corrida) y sale el bloque publicado. Que sea pura
    permite forzar un desacuerdo en las pruebas y comprobar que el estado se vuelve `fail`.
    """
    matches: dict[str, bool] = {}
    for key in sorted(expected):
        matches[key] = expected[key] == observed.get(key)
    reproduced = bool(matches) and all(matches.values()) and set(matches) == set(observed)
    return {
        "state": str(HalfResult.PASS) if reproduced else str(HalfResult.FAIL),
        "evidence": EVIDENCE_RERUN,
        "command": command,
        "as_of": as_of,
        "expected": dict(expected),
        "observed": dict(observed),
        "matches": matches,
        "reproduced_byte_for_byte": reproduced,
        "note": (
            "el motor se re-ejecuta aqui via `cfdtrader.analysis.backtest_report.analyse` con el "
            "mismo instante declarado que el artefacto y se comparan `report_sha256` y los bytes "
            "de `.json`/`.md`; la reproduccion en **procesos nuevos** con `PYTHONHASHSEED` "
            "0/1/random y en segunda pasada la aporta la suite de pruebas (A7, A19)"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Auditoria: no-*look-ahead* observable (A11)
# ─────────────────────────────────────────────────────────────────────────────
def _mutate_session(daily: pl.DataFrame, *, session: date, factor: float) -> pl.DataFrame:
    """Multiplica los precios de la sesion `j`, **sin** tocar `open` ni lo ya derivado."""
    hit = pl.col("session") == session
    return daily.with_columns(
        *[
            pl.when(hit).then(pl.col(column) * factor).otherwise(pl.col(column)).alias(column)
            for column in MUTATED_COLUMNS
        ]
    )


def _calendar_years(daily: pl.DataFrame) -> tuple[int, ...]:
    """Anos que necesita el calendario, tomados del dato (nunca del reloj)."""
    sessions = daily.get_column("session")
    first = cast("date", sessions.min())
    last = cast("date", sessions.max())
    return tuple(range(first.year, last.year + 1))


def _outcome_pairs(
    base: backtest_report.BacktestReport, mutated: Sequence[backtest_report.BaselineOutcome]
) -> list[tuple[backtest_report.BaselineOutcome, backtest_report.BaselineOutcome]]:
    """Empareja las corridas de cada baseline, en el orden declarado."""
    if len(base.outcomes) != len(mutated):
        raise AuditInvariantError(
            f"la corrida mutada trae {len(mutated)} baselines y la de referencia "
            f"{len(base.outcomes)}: la comparacion de A11 exige los mismos (A23)"
        )
    return list(zip(base.outcomes, mutated, strict=True))


def audit_lookahead(*, store: Store, base: backtest_report.BacktestReport) -> dict[str, object]:
    """Muta una sesion posterior y comprueba que **nada** anterior cambia (A11).

    Se re-corre el motor sobre una copia **en memoria** de la historia (jamas sobre el `data/`
    del repositorio y sin escribir un solo fichero) y se comparan, por igualdad, los resultados
    de las sesiones anteriores a `j` en los seis baselines. La mutacion se elige dentro de un
    bloque de test y no toca `open` ni las columnas derivadas de la regla limpia de #52, de modo
    que la membresia del universo no cambia: solo los precios de `j`.
    """
    history = backtest_report.load_history(store)
    calendar = load_calendar(years=_calendar_years(history.daily))

    test_positions = sorted({position for fold in base.split_plan.folds for position in fold.test})
    if not test_positions:
        raise AuditInvariantError(
            "el plan no tiene ninguna sesion de test: sin test no hay decision que auditar (A11)"
        )
    position = test_positions[len(test_positions) // 2]
    session = base.universe.inputs[position].session

    mutated_history = backtest_report.History(
        series_id=history.series_id,
        daily=_mutate_session(history.daily, session=session, factor=MUTATION_FACTOR),
        labels=history.labels,
        intraday=history.intraday,
    )
    mutated_universe = backtest_report.build_inputs(mutated_history, calendar=calendar)
    if len(mutated_universe.inputs) != len(base.universe.inputs):
        raise AuditInvariantError(
            "mutar los precios de una sesion no puede cambiar la membresia del universo: la "
            "mutacion de A11 toca `close`/`high`/`low`, nunca `open` (A11, A23)"
        )
    mutated_plan = backtest_report.build_split_plan(mutated_universe.inputs)
    mutated_outcomes = backtest_report.run_all_baselines(
        mutated_universe.inputs,
        split_plan=mutated_plan,
        cost_model=base.cost_model,
        slippage=base.slippage,
    )

    violations: list[dict[str, object]] = []
    changed: list[dict[str, object]] = []
    compared = 0
    for reference, hostile in _outcome_pairs(base, mutated_outcomes):
        if len(reference.run.folds) != len(hostile.run.folds):
            raise AuditInvariantError(
                f"{reference.baseline}: la corrida mutada tiene otro numero de bloques; la "
                "comparacion de A11 exige el mismo plan (A23)"
            )
        for base_fold, mutated_fold in zip(reference.run.folds, hostile.run.folds, strict=True):
            if len(base_fold.sessions) != len(mutated_fold.sessions):
                raise AuditInvariantError(
                    f"{reference.baseline}: la corrida mutada tiene otro numero de sesiones en un "
                    "bloque; la comparacion de A11 exige la misma rejilla (A23)"
                )
            for left, right in zip(base_fold.sessions, mutated_fold.sessions, strict=True):
                compared += 1
                if left == right:
                    continue
                entry: dict[str, object] = {
                    "baseline": reference.baseline,
                    "session": left.session.isoformat(),
                }
                if left.session < session:
                    violations.append(entry)
                elif len(changed) < CHANGED_SAMPLE_LIMIT:
                    changed.append(entry)

    return {
        "state": str(HalfResult.PASS) if not violations else str(HalfResult.FAIL),
        "evidence": EVIDENCE_RERUN,
        "mutation": {
            "session": session.isoformat(),
            "position": position,
            "factor": MUTATION_FACTOR,
            "columns": list(MUTATED_COLUMNS),
            "open_untouched": True,
        },
        "compared_outcomes": compared,
        "changed_count": len(changed),
        "changed_sample": changed,
        "violations": violations,
        "note": (
            "se muta **una sesion posterior** y se re-corre el motor sobre una copia en memoria "
            "de la historia: ningun resultado de una sesion anterior puede cambiar (A11). La "
            "mutacion es observable (al menos un resultado cambia), asi que la comprobacion no "
            "es vacua"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Auditoria: identidades de conservacion (A10)
# ─────────────────────────────────────────────────────────────────────────────
def _artifact_row_by_baseline(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    """Las filas de la tabla del artefacto, indexadas por baseline."""
    rows = _require_rows(payload, "baselines", "rows")
    indexed: dict[str, dict[str, object]] = {}
    for row in rows:
        indexed[_require_str(row, "baseline")] = row
    return indexed


def conservation_audit(
    *, rerun: backtest_report.BacktestReport, artifact: InputArtifact
) -> dict[str, object]:
    """Verifica las identidades de recuento y las cruza con el artefacto de #69 (A10).

    Falla con error tipado si alguna identidad no cuadra, incluida la discrepancia entre lo
    medido en esta corrida y lo publicado por #69: una sesion del universo no puede
    desaparecer sin aparecer en un recuento.
    """
    plan = rerun.split_plan
    n_test = sum(len(fold.test) for fold in plan.folds)
    not_in_any_test = len(plan.uncovered)
    n_sessions = len(rerun.universe.inputs)
    if n_sessions != n_test + not_in_any_test:
        raise ConservationError(
            f"la identidad del universo no cuadra: {n_sessions} sesiones frente a {n_test} de "
            f"test + {not_in_any_test} fuera de todo test (A10)"
        )

    artifact_rows = _artifact_row_by_baseline(artifact.payload)
    published: list[dict[str, object]] = []
    for outcome in rerun.outcomes:
        run = outcome.run
        total = run.traded + run.no_trade + run.skipped
        if total != n_test:
            raise ConservationError(
                f"{outcome.baseline}: traded + no_trade + skipped = {total} y el test tiene "
                f"{n_test} sesiones (A10)"
            )
        if run.not_in_any_test != not_in_any_test:
            raise ConservationError(
                f"{outcome.baseline}: declara {run.not_in_any_test} sesiones fuera de todo test y "
                f"el plan tiene {not_in_any_test} (A10)"
            )
        row = artifact_rows.get(outcome.baseline)
        if row is None:
            raise ConservationError(
                f"el artefacto de #69 no publica ninguna fila para {outcome.baseline!r}: la "
                "tabla del informe es la suya y no puede faltar ninguna (A6, A10)"
            )
        published.append(
            {
                "baseline": outcome.baseline,
                "n_test": n_test,
                "traded": run.traded,
                "no_trade": run.no_trade,
                "skipped": run.skipped,
                "total": total,
                "equals_n_test": True,
                "not_in_any_test": not_in_any_test,
            }
        )
        _cross_check_row(baseline=outcome.baseline, row=row, run_total=total, n_test=n_test)

    if _require_int(artifact.payload, "universe", "sessions") != n_sessions:
        raise ConservationError(
            "el artefacto de #69 declara un universo de "
            f"{_require_int(artifact.payload, 'universe', 'sessions')} sesiones y la corrida de "
            f"esta auditoria mide {n_sessions} (A10)"
        )

    return {
        "state": str(HalfResult.PASS),
        "evidence": EVIDENCE_RERUN,
        "identity": f"{n_sessions} = {n_test} (test) + {not_in_any_test} (fuera de todo test)",
        "per_baseline_identity": "traded + no_trade + skipped = n_test",
        "not_in_any_test": not_in_any_test,
        "n_test": n_test,
        "n_sessions": n_sessions,
        "rows": published,
        "note": (
            "las identidades se miden sobre la re-ejecucion del motor y se **cruzan** con la "
            "tabla del artefacto de #69: si discrepan, el informe falla en vez de publicar un "
            "recuento que no cuadra (A10)"
        ),
    }


def _cross_check_row(*, baseline: str, row: dict[str, object], run_total: int, n_test: int) -> None:
    """Comprueba que la fila del artefacto de #69 coincide con la corrida de esta auditoria."""
    traded = _require_int(row, "traded")
    no_trade = _require_int(row, "no_trade")
    skipped = _require_int(row, "skipped")
    published_n_test = _require_int(row, "n_test")
    if traded + no_trade + skipped != published_n_test:
        raise ConservationError(
            f"{baseline}: la fila publicada por #69 no cuadra por si misma "
            f"({traded} + {no_trade} + {skipped} != {published_n_test}) (A10)"
        )
    if published_n_test != n_test or traded + no_trade + skipped != run_total:
        raise ConservationError(
            f"{baseline}: la tabla de #69 y la corrida de esta auditoria no coinciden en los "
            "recuentos; la tabla se copia, asi que una discrepancia es un error, no un matiz "
            "(A6, A10)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Auditoria: coste declarado reproducido exacto (A9)
# ─────────────────────────────────────────────────────────────────────────────
def _round_trip(
    *, model: CostModel, slippage: SlippageParameter, notional_usd: Decimal, side: Side
) -> CostBreakdown:
    """Coste de una ida y vuelta con **una** noche, calculado por el motor de #11."""
    return cost_breakdown(
        model=model,
        slippage=slippage,
        notional_usd=notional_usd,
        side=side,
        nights=1,
        overnight_reason=DECLARED_OVERNIGHT_REASON,
    )


def costs_audit() -> dict[str, object]:
    """Reproduce la tabla declarada importando el modelo de #8/#11 y comprueba su aritmetica.

    Las cifras **no** se copian aqui: se importan y se recalculan con `cost_breakdown` de #11.
    Las invariantes que se comprueban son internas (cada importe cuadra con su porcentaje y su
    nocional, la ida y vuelta es el diferencial mas una noche, y la tenencia del corto es
    negativa mientras la del largo es positiva), nunca una re-copia de la tabla.
    """
    model = declared_cost_model()
    slippage = declared_slippage_assumption()
    notional_usd = backtest_report.NOTIONAL_USD
    short = cost_breakdown(
        model=model, slippage=slippage, notional_usd=notional_usd, side=Side.SHORT, nights=0
    )
    long = cost_breakdown(
        model=model, slippage=slippage, notional_usd=notional_usd, side=Side.LONG, nights=0
    )
    short_round = _round_trip(
        model=model, slippage=slippage, notional_usd=notional_usd, side=Side.SHORT
    )
    long_round = _round_trip(
        model=model, slippage=slippage, notional_usd=notional_usd, side=Side.LONG
    )

    checks: list[dict[str, object]] = [
        _check(
            name="spread_usd_vs_pct",
            ok=short.spread_usd == short.spread_pct / Decimal(100) * notional_usd,
            detail="el diferencial en $ cuadra con su % y el nocional declarado",
        ),
        _check(
            name="carry_short_sign",
            ok=short_round.carry_usd < 0,
            detail="la tenencia del corto es negativa (el CFD paga al llevarlo corto)",
        ),
        _check(
            name="carry_long_sign",
            ok=long_round.carry_usd > 0,
            detail="la tenencia del largo es positiva (el CFD cobra por llevarlo largo)",
        ),
        _check(
            name="carry_usd_vs_pct_short",
            ok=short_round.carry_usd == short_round.carry_pct / Decimal(100) * notional_usd,
            detail="la tenencia del corto en $ cuadra con su % y el nocional",
        ),
        _check(
            name="carry_usd_vs_pct_long",
            ok=long_round.carry_usd == long_round.carry_pct / Decimal(100) * notional_usd,
            detail="la tenencia del largo en $ cuadra con su % y el nocional",
        ),
        _check(
            name="round_trip_short",
            ok=short_round.c_declared_usd == short.spread_usd + short_round.carry_usd,
            detail="la ida y vuelta del corto es el diferencial mas una noche de tenencia",
        ),
        _check(
            name="round_trip_long",
            ok=long_round.c_declared_usd == long.spread_usd + long_round.carry_usd,
            detail="la ida y vuelta del largo es el diferencial mas una noche de tenencia",
        ),
        _check(
            name="fx_is_declared_zero",
            ok=short.fx_pct == 0 and model.fx_source is not None and model.fx_reason is not None,
            detail="el coste de divisa es 0 **con procedencia y motivo**, no un cero mudo",
        ),
        _check(
            name="slippage_total_stays_null",
            ok=short_round.c_total_usd is None and len(short_round.nulls) > 0,
            detail="el total con el supuesto de #64 no se cierra: «`null != 0`» (A18)",
        ),
    ]
    failed = [entry["name"] for entry in checks if entry["ok"] is not True]
    if failed:
        raise AuditInvariantError(
            f"la reproduccion del coste declarado no cuadra en {failed}: la tabla de #8/#11 "
            "cambio o se dejo de importar (A9)"
        )

    return {
        "state": str(HalfResult.PASS),
        "evidence": EVIDENCE_RERUN,
        "source": (
            "cfdtrader.backtest.costs.declared_cost_model() y "
            "declared_slippage_assumption() (cifras de #8, reproducidas por #11)"
        ),
        "model_name": model.name,
        "notional_usd": format(notional_usd, "f"),
        "units": {
            "usd": UNIT_USD,
            "pct": UNIT_PCT,
            "pct_per_night": UNIT_PCT_PER_NIGHT,
            "bp": UNIT_BP,
        },
        "spread": {
            "usd": _dec(short.spread_usd, "0.01"),
            "pct": _dec(short.spread_pct, "0.0001"),
            "entry_pct": _dec(model.spread_entry_pct, "0.0001"),
            "exit_pct": _dec(model.spread_exit_pct, "0.0001"),
            "entry_source": model.spread_entry_source,
            "exit_source": model.spread_exit_source,
        },
        "carry_per_night": {
            Side.SHORT.value: {
                "usd": _dec(short_round.carry_usd, "0.01"),
                "pct": _dec(short_round.carry_pct, "0.0001"),
            },
            Side.LONG.value: {
                "usd": _dec(long_round.carry_usd, "0.01"),
                "pct": _dec(long_round.carry_pct, "0.0001"),
            },
            "source": model.carry_source,
        },
        "fx": {
            "usd": _dec(short.fx_usd, "0.01"),
            "pct": _dec(short.fx_pct, "0.0001"),
            "state": model.fx_state.value,
            "source": model.fx_source,
            "reason": model.fx_reason,
        },
        "commission": {
            "usd": _dec(short.commission_usd, "0.01"),
            "pct": _dec(short.commission_pct, "0.0001"),
            "state": model.commission_state.value,
            "source": model.commission_source,
            "reason": model.commission_reason,
        },
        "round_trip_one_night": {
            Side.SHORT.value: {
                "usd": _dec(short_round.c_declared_usd, "0.01"),
                "pct": _dec(short_round.c_declared_pct, "0.0001"),
                "nights": short_round.nights,
            },
            Side.LONG.value: {
                "usd": _dec(long_round.c_declared_usd, "0.01"),
                "pct": _dec(long_round.c_declared_pct, "0.0001"),
                "nights": long_round.nights,
            },
        },
        "slippage": {
            "state": slippage.state.value,
            "is_measurement": slippage.is_measurement,
            "pct_of_r": _pct_of_r_ratio(slippage),
            "pct_of_r_declared_percent": _pct_of_r_declared_percent(slippage),
            "r_pct": None if slippage.r_pct is None else _dec(slippage.r_pct, "0.0001"),
            "source": slippage.source,
            "issue": "#62",
        },
        "checks": checks,
        "note": (
            "los valores declarados se **reproducen por importacion y recalculo** (nunca "
            "copiando literales): la tabla es de coste declarado, **no** una validacion de la "
            "estrategia, y el supuesto de #64 no cierra el total (A9, A18)"
        ),
    }


def _check(*, name: str, ok: bool, detail: str) -> dict[str, object]:
    """Una comprobacion de la auditoria de coste, con su resultado inseparable de su detalle."""
    return {"name": name, "ok": ok, "detail": detail}


def _pct_of_r_ratio(slippage: SlippageParameter) -> str | None:
    """El supuesto como **ratio** sobre `R` (0,2 = 20 %), o `null` si no hay supuesto."""
    if slippage.pct_of_r is None:
        return None
    return _dec(slippage.pct_of_r / Decimal(100), "0.0001")


def _pct_of_r_declared_percent(slippage: SlippageParameter) -> str | None:
    """El valor literal del parametro declarado de #8/#64, sin re-teclearlo."""
    if slippage.pct_of_r is None:
        return None
    return _dec(slippage.pct_of_r, "0.01")


# ─────────────────────────────────────────────────────────────────────────────
# Auditoria: purga y embargo tal cual los da #12, con `h = 0` declarado no-op (A8)
# ─────────────────────────────────────────────────────────────────────────────
def purge_embargo_block(
    *, artifact: InputArtifact, rerun: backtest_report.BacktestReport
) -> dict[str, object]:
    """Publica los numeros de #12 y los declara **no-ops estructurales** si lo son (A8)."""
    payload = artifact.payload
    values: dict[str, int] = {
        "purge_total": _require_int(payload, "plan", "purge_total"),
        "embargo_total": _require_int(payload, "plan", "embargo_total"),
        "embargo_in_train_total": _require_int(payload, "plan", "embargo_in_train_total"),
    }
    label_horizon = _require_int(payload, "plan", "label_horizon")
    declared = _require_bool(payload, "plan", "exclusions_are_no_op")

    from_rerun = {
        "purge_total": rerun.split_plan.purge_total,
        "embargo_total": rerun.split_plan.embargo_total,
        "embargo_in_train_total": rerun.split_plan.embargo_in_train_total,
    }
    if values != from_rerun:
        raise ConservationError(
            "los recuentos de purga/embargo publicados por #69 no coinciden con la re-ejecucion "
            f"de esta auditoria ({values} frente a {from_rerun}) (A8, A10)"
        )
    if declared != rerun.split_plan.exclusions_are_no_op:
        raise ConservationError(
            "el flag `exclusions_are_no_op` publicado por #69 no coincide con la re-ejecucion "
            "(A8, A10)"
        )

    if declared:
        note = (
            f"con `label_horizon = {label_horizon}` la purga y el embargo son **no-ops "
            "estructurales** de #12: se publican con sus numeros, nunca como un filtro activo"
        )
    else:
        note = (
            "las exclusiones de #12 **si** quitaron muestra del train con este plan: se publican "
            "tal cual, sin presentarlas como un no-op"
        )
    return {
        **values,
        "label_horizon": label_horizon,
        "exclusions_are_no_op": declared,
        "structural_no_op": declared,
        "presented_as_active_filter": False,
        "evidence": EVIDENCE_ARTIFACT,
        "note": note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Auditoria: rechazo de las metricas netas (A13, A31)
# ─────────────────────────────────────────────────────────────────────────────
def net_metrics_rejection(*, rerun: backtest_report.BacktestReport) -> dict[str, object]:
    """**Demuestra** el rechazo de #15 con la corrida real de `always_long` (A13).

    No se publica ninguna metrica neta: se comprueba que `calculate_metrics` lanza su error
    tipado nombrando `pnl_net_pct`, que es la razon por la que la mitad (b) es `not_evaluable`.
    """
    outcome = next(
        (item for item in rerun.outcomes if item.baseline == HALF_ALWAYS_LONG),
        None,
    )
    if outcome is None:
        raise AuditInvariantError(
            f"la corrida no trae el baseline {HALF_ALWAYS_LONG!r}: sin el no se puede demostrar "
            "el rechazo de las metricas netas (A13)"
        )
    try:
        calculate_metrics(outcome.run)
    except MetricsInputError as error:
        return {
            "state": "rejected",
            "evidence": EVIDENCE_RERUN,
            "baseline": HALF_ALWAYS_LONG,
            "where": "cfdtrader.backtest.metrics.calculate_metrics",
            "error": type(error).__name__,
            "message": str(error),
            "mentions_pnl_net_pct": "pnl_net_pct" in str(error),
            "note": (
                "el rechazo se **mide** ejecutando la agregacion de #15 sobre la corrida real: "
                "mientras `pnl_net_pct` sea `null` no hay metrica neta que publicar (A13, A31)"
            ),
        }
    except Exception as error:  # se declara, no se traga
        raise AuditInvariantError(
            "la agregacion de #15 fallo con un error distinto del declarado "
            f"({type(error).__name__}): la mitad (b) de la puerta ya no es lo que dice la issue "
            "(A13)"
        ) from error
    raise AuditInvariantError(
        "la agregacion de #15 **no** rechazo la corrida de la Fase 1: si hubiera metricas netas "
        "publicables, la mitad (b) dejaria de ser `not_evaluable` y el informe tendria que "
        "compararlas, no declararlas ausentes (A13)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# La puerta: mitad (a) medida, mitad (b) no evaluable, agregado mecanico (A12-A15, A29)
# ─────────────────────────────────────────────────────────────────────────────
def mechanical_gate(*, half_a: str, half_b: str, claimed: str | None = None) -> GateVerdict:
    """Aplica la agregacion de #9 y rechaza un agregado **afirmado** que no cuadre (A14, A29).

    Reutiliza `aggregate_gate` de #9: `fail` si alguna mitad es `fail`, `not_evaluable` si
    ninguna es `fail` y al menos una no es `pass`, y `pass` **solo** si las dos son `pass`. Un
    `not_evaluable` nunca se convierte en `pass`, y afirmar un agregado distinto del que sale de
    la regla es un error de consistencia.
    """
    computed = aggregate_gate(half_a, half_b)
    if claimed is not None and claimed != str(computed):
        raise VerdictError(
            f"el agregado afirmado ({claimed!r}) no es el que sale de la regla de #9 "
            f"({str(computed)!r}): {GATE_AGGREGATION_RULE} (A14, A29)"
        )
    return computed


def _verdict(gate: GateVerdict) -> Recommendation:
    """Recomendacion declarada para cada veredicto, **sujeta** a la regla de consistencia."""
    mapping: dict[GateVerdict, Recommendation] = {
        GateVerdict.PASS: Recommendation.CONTINUE,
        GateVerdict.NOT_EVALUABLE: Recommendation.REFRAME,
        GateVerdict.FAIL: Recommendation.STOP,
    }
    choice = mapping[gate]
    if not recommendation_is_consistent(gate, choice):
        raise VerdictError(
            f"la recomendacion {choice.value!r} no es coherente con el veredicto {gate.value!r}: "
            f"{RECOMMENDATION_CONSISTENCY_RULE} (A29)"
        )
    return choice


# ─────────────────────────────────────────────────────────────────────────────
# Fronteras legibles por maquina (A26) y textos declarados
# ─────────────────────────────────────────────────────────────────────────────
REPORT_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_recalcula_la_tabla",
        "issue": "#69",
        "statement": (
            "no vuelve a correr los seis baselines sobre la historia: consume el artefacto de #69 "
            "y copia su tabla, marcada con `evidence: artifact`"
        ),
    },
    {
        "id": "no_reescribe_el_artefacto_de_entrada",
        "issue": "#69",
        "statement": (
            "no reescribe `phase1_backtest_*.json` ni su `.md`: la huella del par es identica "
            "antes y despues de la corrida"
        ),
    },
    {
        "id": "no_escribe_la_suite_de_integridad",
        "issue": "#17",
        "statement": (
            "no escribe la suite de integridad del arnes (no-*look-ahead* sobre *features*, "
            "*golden dataset*, costes a mano con CI): aqui los cuatro tests criticos se **usan** "
            "como evidencia de auditoria de la Fase 1"
        ),
    },
    {
        "id": "no_calcula_pbo_ni_dsr",
        "issue": "#16",
        "statement": (
            "no calcula la probabilidad de sobreajuste ni el Sharpe deflactado, ni registra "
            "experimentos en `runs/<hash>/`"
        ),
    },
    {
        "id": "no_mide_el_slippage",
        "issue": "#62",
        "statement": (
            "no mide el *slippage* real: declara el supuesto de #64 (`state: assumed`, "
            "`is_measurement: false`) y **no** lo convierte en medicion para poder publicar "
            "metricas"
        ),
    },
    {
        "id": "no_publica_metricas_netas",
        "issue": "#15",
        "statement": (
            "no publica Sharpe, Sortino, valor esperado, drawdown ni intervalos: `net_metrics` "
            "viaja como `not_computable` con su motivo y su `follow_up`"
        ),
    },
    {
        "id": "no_deriva_el_nocional",
        "issue": "#60",
        "statement": (
            "no calcula *sizing*: el nocional es el plano declarado de #69, nunca derivado del "
            "capital, de `R` ni del apalancamiento"
        ),
    },
    {
        "id": "no_produce_el_liston_b",
        "issue": "#70",
        "statement": (
            "no produce el liston B (`close->close` con financiacion *overnight*): el motor de "
            "#13 no puede mantener una posicion que cruce la noche (regla 6)"
        ),
    },
    {
        "id": "no_construye_la_fase_2",
        "issue": "#28",
        "statement": (
            "no construye el *feature store* ni los modelos de Fase 2: la puerta de salida no es "
            "el inicio de la fase siguiente"
        ),
    },
    {
        "id": "no_verifica_el_corte_de_financiacion",
        "issue": "#59",
        "statement": (
            "no asume ninguna hora de corte de financiacion: viaja como `None` y sin verificar, "
            "porque fijarla es la pregunta abierta del broker"
        ),
    },
    {
        "id": "no_toca_el_motor",
        "issue": "#13",
        "statement": (
            "no modifica ni envuelve `src/cfdtrader/backtest/**`: lo reutiliza a traves de "
            "`backtest_report` y `baselines` y no captura sus errores"
        ),
    },
)

FOLLOW_UPS: Final[tuple[dict[str, str], ...]] = (
    {
        "issue": "#17",
        "topic": "suite de integridad del arnes de Fase 1",
        "why": (
            "el *golden dataset* con hash congelado, el no-*look-ahead* sobre *features* y los "
            "costes a mano con CI merecen una suite propia; aqui solo se usan como evidencia"
        ),
    },
    {
        "issue": "#16",
        "topic": "PBO y Sharpe deflactado",
        "why": (
            "la probabilidad de sobreajuste y el DSR son la medicion que hoy falta para juzgar "
            "la seleccion de modelos, no el arnes"
        ),
    },
    {
        "issue": "#62",
        "topic": "medicion del *slippage* real",
        "why": (
            "medir el *slippage* (10-15 ejecuciones en la apertura) es lo que convierte el "
            "supuesto de #64 en un numero y desbloquea la mitad (b) de la puerta"
        ),
    },
    {
        "issue": "#60",
        "topic": "tamano de `R` y umbrales",
        "why": (
            "sin `R` decidido el supuesto de #64 no se puede cobrar y no hay comparacion neta "
            "contra *siempre largo*"
        ),
    },
    {
        "issue": "#70",
        "topic": "liston B (*siempre largo* con financiacion *overnight*)",
        "why": (
            "el liston B es la comparacion honesta contra *mantener la posicion*, y #13 no puede "
            "producirla porque la regla 6 prohibe cruzar la noche"
        ),
    },
    {
        "issue": "#50",
        "topic": "fuente del `SPX500:CFD`",
        "why": (
            "los precios son *proxy* de `^GSPC` mientras no haya fuente del CFD; el signo de las "
            "conclusiones se traslada, la magnitud no"
        ),
    },
    {
        "issue": "#51",
        "topic": "`bid`/`ask` del CFD",
        "why": "el diferencial real del CFD necesita una fuente de `bid`/`ask` que hoy no existe",
    },
    {
        "issue": "#28",
        "topic": "Fase 2: *feature store* y modelos",
        "why": "la construccion de la Fase 2 arranca con el veredicto de esta puerta",
    },
    {
        "issue": "#29",
        "topic": "Fase 2: puerta de salida de modelos",
        "why": "el gate de modelos repite este patron de veredicto parcial y evidencia propia",
    },
    {
        "issue": "#65",
        "topic": "documentacion de las decisiones del propietario",
        "why": (
            "las decisiones de #64 y las de #67/#68/#69 viven hoy solo en las issues; `plan.md` "
            "y `tech_stack.md` las recogen en #65"
        ),
    },
    {
        "issue": "#67",
        "topic": "regularizacion del backlog: CPCV",
        "why": (
            "#67, #68 y #69 no estan en `_docs/tasks.md` (viven solo como issues); alinearlos es "
            "una tarea de documentacion aparte, no se crea issue nueva aqui"
        ),
    },
    {
        "issue": "#68",
        "topic": "regularizacion del backlog: *holdout* final intocable",
        "why": "misma razon que #67: el backlog documentado no conoce el *holdout* final",
    },
)

HOW_TO_CLOSE: Final[tuple[dict[str, str], ...]] = (
    {
        "issue": "#62",
        "reason": (
            "medir el *slippage* real con ejecuciones en la apertura: es lo que convierte el "
            "supuesto de #64 en una medicion y permite calcular `pnl_net_pct`"
        ),
    },
    {
        "issue": "#70",
        "reason": (
            "construir el liston B (*siempre largo* `close->close` con financiacion *overnight*): "
            "sin el, «bate *siempre largo*» no tiene el termino de comparacion que pide la mitad "
            "(b), y #13 no puede producirlo (regla 6)"
        ),
    },
    {
        "issue": "#60",
        "reason": (
            "decidir el tamano de `R` y los umbrales: sin `R` el supuesto de #64 no se puede "
            "cobrar y `calculate_metrics` sigue rechazando la corrida"
        ),
    },
)


# ─────────────────────────────────────────────────────────────────────────────
# El informe
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Phase1Report:
    """El informe de Fase 1: payload canonico, hash, el artefacto consumido y la re-ejecucion."""

    as_of: datetime
    report_date: date
    payload: dict[str, object]
    report_sha256: str
    input_artifact: InputArtifact
    rerun: backtest_report.BacktestReport

    @property
    def report_stem(self) -> str:
        """Nombre base del informe: ``phase1_report_<AAAA-MM-DD>``."""
        return f"{REPORT_PREFIX}_{self.report_date.isoformat()}"

    def json_text(self) -> str:
        """JSON determinista: mismas entradas y mismo `as_of` ⇒ mismo texto byte a byte (A19)."""
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


def _baselines_table(*, artifact: InputArtifact) -> dict[str, object]:
    """La tabla de baselines **copiada** del artefacto de #69, con su procedencia (A6, A32)."""
    payload = artifact.payload
    rows = _require_rows(payload, "baselines", "rows")
    summary: list[dict[str, object]] = []
    for row in rows:
        summary.append(
            {
                "baseline": _require_str(row, "baseline"),
                "n_test": _require_int(row, "n_test"),
                "traded": _require_int(row, "traded"),
                "no_trade": _require_int(row, "no_trade"),
                "skipped": _require_int(row, "skipped"),
                "not_in_any_test": _require_int(row, "not_in_any_test"),
                "run_sha256": _require_str(row, "run_sha256"),
            }
        )
    return {
        "evidence": EVIDENCE_ARTIFACT,
        "source_artifact": artifact.relative_path,
        "basis": _require_str(payload, "baselines", "basis"),
        "is_validation": _require_bool(payload, "baselines", "is_validation"),
        "notional_usd": _require_str(payload, "baselines", "notional_usd"),
        "notional_provenance": NOTIONAL_PROVENANCE,
        "baseline_order": [entry["baseline"] for entry in summary],
        "random_matched": deepcopy(_require(payload, "baselines", "random_matched")),
        "rows": summary,
        "note": (
            "las cifras son de **coste declarado** sobre el retorno declarado de #69: la tabla se "
            "copia literalmente, no se recalcula, y **no** es una validacion de la estrategia "
            "(A6, A18)"
        ),
    }


def _limits_block(*, artifact: InputArtifact) -> dict[str, object]:
    """El bloque de limites heredado del artefacto de #69, **sin fusionar** estados (A17)."""
    inherited = deepcopy(_mapping(_require(artifact.payload, "limits"), where="limits"))
    inherited["evidence"] = EVIDENCE_ARTIFACT
    inherited["source_artifact"] = artifact.relative_path
    inherited["note"] = (
        "los limites se **heredan** del artefacto de #69 sin edulcorar ni fusionar: la puerta de "
        "Fase 0 sigue en `fail`, el *slippage* sigue siendo un supuesto, los precios siguen siendo "
        "*proxy*, el corte de financiacion sigue sin verificar y las metricas netas siguen siendo "
        "`not_computable` (A17)"
    )
    return inherited


def _net_metrics_block(*, artifact: InputArtifact) -> dict[str, object]:
    """Las metricas netas declaradas **no calculables**, con su motivo y su seguimiento (A31)."""
    inherited = deepcopy(_mapping(_require(artifact.payload, "net_metrics"), where="net_metrics"))
    inherited["evidence"] = EVIDENCE_ARTIFACT
    inherited["source_artifact"] = artifact.relative_path
    inherited["published_metrics"] = []
    inherited["note"] = (
        "el informe **no** publica Sharpe, Sortino, valor esperado, drawdown ni intervalos: se "
        "declara `not_computable` en vez de rellenar el hueco (A31)"
    )
    return inherited


def _blockers(*, gate: GateVerdict) -> list[dict[str, object]]:
    """Los bloqueos buscables por maquina, con el vocabulario de #9 donde encaja (A30)."""
    codes: tuple[tuple[BlockerCode | str, tuple[str, ...], str], ...] = (
        (
            BlockerCode.SLIPPAGE_ASSUMED_NOT_MEASURED,
            ("#62",),
            "el *slippage* es un supuesto declarado de #64, no una medicion; sin medirlo no hay "
            "`pnl_net_pct` y la mitad (b) no se puede evaluar",
        ),
        (
            BlockerCode.R_UNDECIDED,
            ("#60",),
            "el tamano de `R` sigue sin decidir: el supuesto de #64 no se puede cobrar sin el",
        ),
        (
            BlockerCode.FINANCING_CUT_UNVERIFIED,
            ("#59",),
            "el corte de financiacion sigue sin verificar: se declara `None` y no se asume "
            "ninguna hora",
        ),
        (
            BlockerCode.BROKER_UNDECIDED,
            ("#59",),
            "el broker definitivo sigue sin decidir: diferencial, comision y corte son los "
            "declarados, no los del broker real",
        ),
        (
            "net_metrics_not_computable",
            ("#62", "#60"),
            "`calculate_metrics` de #15 rechaza la corrida porque `pnl_net_pct` es `null`: no "
            "hay ninguna metrica neta que publicar",
        ),
        (
            "always_long_not_evaluable",
            ("#70", "#62"),
            "sin el liston B y sin metrica neta no se puede afirmar ni negar que *siempre largo* "
            "bata a los baselines de forma significativa",
        ),
    )
    blockers: list[dict[str, object]] = []
    for code, issues, reason in codes:
        blockers.append({"code": str(code), "issues": list(issues), "reason": reason})
    if gate is not GateVerdict.PASS:
        blockers.append(
            {
                "code": "gate_not_passed",
                "issues": ["#18"],
                "reason": (
                    f"el veredicto agregado de la puerta es {gate.value!r}: ni un `fail` ni un "
                    "`not_evaluable` autorizan a seguir como si la puerta estuviera superada"
                ),
            }
        )
    return blockers


def _gate_halves(
    *, determinism: dict[str, object], rejection: dict[str, object]
) -> dict[str, object]:
    """Las dos mitades de la puerta, cada una con su evidencia (A12, A13)."""
    return {
        HALF_DETERMINISM: {
            "state": determinism["state"],
            "evidence": EVIDENCE_RERUN,
            "block": f"harness_audit.{HALF_DETERMINISM}",
            "reason": (
                "el motor se re-ejecuta con el mismo instante declarado y reproduce "
                "`report_sha256` y los bytes de `.json`/`.md` del artefacto de #69"
            ),
            "observed": deepcopy(determinism["observed"]),
        },
        HALF_ALWAYS_LONG: {
            "state": str(HalfResult.NOT_EVALUABLE),
            "evidence": EVIDENCE_ARTIFACT,
            "block": "harness_audit.net_metrics",
            "reason": (
                "`pnl_net_pct` es `null` en el 100 % de las operaciones (supuesto de #64) y "
                "`calculate_metrics` de #15 lanza `MetricsInputError`: no hay ninguna metrica "
                "neta con la que comparar *siempre largo* contra los demas baselines"
            ),
            "demonstration": {
                "where": rejection["where"],
                "error": rejection["error"],
                "mentions_pnl_net_pct": rejection["mentions_pnl_net_pct"],
            },
            "comparison_performed": False,
        },
    }


def _partial_verdict(*, half_a: str, half_b: str, gate: GateVerdict) -> str:
    """Prosa corta que dice **cual** de las dos mitades esta medida y cual no (A15)."""
    if half_a == str(HalfResult.NOT_EVALUABLE):
        state_text = "no evaluable hoy"
    else:
        state_text = "medida hoy con evidencia propia"
    return (
        f"Mitad (a) determinismo del motor: {state_text} ({half_a}). Mitad (b) *siempre largo* "
        f"contra los seis baselines: no evaluable en terminos netos ({half_b}), porque el "
        "`slippage` es un supuesto declarado y no una medicion. Agregado: "
        f"{gate.value} ⇒ la puerta de salida de la Fase 1 **no** se declara superada."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Payload y ejecucion
# ─────────────────────────────────────────────────────────────────────────────
def _payload(
    *,
    as_of: datetime,
    artifact: InputArtifact,
    determinism: dict[str, object],
    lookahead: dict[str, object],
    conservation: dict[str, object],
    costs: dict[str, object],
    purge_embargo: dict[str, object],
    rejection: dict[str, object],
) -> dict[str, object]:
    """El payload canonico del informe: tres verdades separadas y tipos JSON puros (A34)."""
    halves = _gate_halves(determinism=determinism, rejection=rejection)
    half_a = str(cast("dict[str, object]", halves[HALF_DETERMINISM])["state"])
    half_b = str(cast("dict[str, object]", halves[HALF_ALWAYS_LONG])["state"])
    gate = mechanical_gate(half_a=half_a, half_b=half_b)
    recommendation = _verdict(gate)
    blockers = _blockers(gate=gate)
    inherited_gate = _require_str(artifact.payload, "limits", "gate")
    if inherited_gate not in {member.value for member in GateVerdict}:
        raise MissingInputFieldError(
            f"`limits.gate` del artefacto de #69 trae {inherited_gate!r}, fuera del vocabulario "
            "declarado (A17, A23)"
        )
    phase1_ready = (
        gate is GateVerdict.PASS and inherited_gate == str(GateVerdict.PASS) and not blockers
    )

    payload: dict[str, object] = {
        "task": "#18",
        "artifact": REPORT_PREFIX,
        "title": "Informe de Fase 1 y puerta de salida",
        "generated_at": as_of.isoformat(),
        "report_date": as_of.date().isoformat(),
        "hash_format": REPORT_HASH_FORMAT,
        "phase1_ready": phase1_ready,
        "is_validation": False,
        "clock": {"as_of": as_of.isoformat(), "divergence_with_phase0": CLOCK_DIVERGENCE},
        "input_selection": {
            "class": {
                "kind": INPUT_CLASS.kind,
                "pattern": INPUT_CLASS.pattern,
                "description": INPUT_CLASS.description,
            },
            "rule": FILE_SELECTION_RULE,
            "reused_from": "cfdtrader.analysis.phase0_report.select_artifact (#9)",
            "note": (
                "cero candidatos y ambiguedad son errores tipados propios: nunca se elige un "
                "artefacto a dedo ni en silencio (A5)"
            ),
        },
        "source_artifact": artifact.provenance(),
        "baselines": _baselines_table(artifact=artifact),
        "harness_audit": {
            "determinism": determinism,
            "lookahead": lookahead,
            "conservation": conservation,
            "costs": costs,
            "purge_embargo": purge_embargo,
            "net_metrics": rejection,
        },
        "gate_halves": halves,
        "gate": {
            "aggregate": gate.value,
            "halves": [HALF_DETERMINISM, HALF_ALWAYS_LONG],
            "aggregation_rule": GATE_AGGREGATION_RULE,
            "aggregation_source": GATE_AGGREGATION_SOURCE,
            "inherited_phase0_gate": inherited_gate,
            "phase1_ready": phase1_ready,
            "is_validation": False,
            "partial_verdict": _partial_verdict(half_a=half_a, half_b=half_b, gate=gate),
            "statement": (
                "la puerta de salida de la Fase 1 no se declara superada: el veredicto es parcial "
                "y la mitad que decidiria la continuidad no es evaluable hoy"
            ),
            "what_is_verified": (
                "el determinismo del motor y la auditoria del arnes (purga/embargo, coste "
                "declarado, identidades de recuento y no-*look-ahead* observable)"
            ),
            "what_is_not_evaluable": (
                "la comparacion neta de *siempre largo* contra los seis baselines: exige "
                "`pnl_net_pct`, que hoy es `null`"
            ),
        },
        "limits": _limits_block(artifact=artifact),
        "net_metrics": _net_metrics_block(artifact=artifact),
        "how_to_close": [dict(entry) for entry in HOW_TO_CLOSE],
        "recommendation": {
            "value": recommendation.value,
            "rule": RECOMMENDATION_CONSISTENCY_RULE,
            "gate": gate.value,
            "consistent": recommendation_is_consistent(gate, recommendation),
        },
        "blockers": blockers,
        "limitations": [
            "el informe **no** es una validacion de la estrategia: la puerta de Fase 0 sigue en "
            "`fail` y la tabla es de coste declarado (`basis: declared_cost`)",
            "la mitad (b) es `not_evaluable`: no se ha medido el *slippage* ni decidido `R`, "
            "asi que no hay comparacion neta que hacer",
            "los precios son *proxy* de `^GSPC` mientras no haya fuente del `SPX500:CFD` (#50) y "
            "el corte de financiacion sigue sin verificar (#59)",
            "el no-*look-ahead* se audita mutando una sesion posterior y re-corriendo el motor "
            "sobre una copia en memoria; la suite de integridad completa es #17",
            "el liston B de *siempre largo* con financiacion *overnight* (#70) no existe todavia: "
            "el motor de #13 no puede cruzar la noche",
        ],
        "does_not_do": [dict(entry) for entry in REPORT_DOES_NOT_DO],
        "follow_ups": [dict(entry) for entry in FOLLOW_UPS],
        "notes": [
            "`source_artifact`, `harness_audit` y `gate` viajan **separados**: lo copiado de #69, "
            "lo medido aqui y el veredicto no se fusionan en un unico «resultado» (A34)",
            "cada bloque declara su procedencia (`evidence: re-run` si se produjo en esta corrida, "
            "`artifact` si se copio del artefacto de #69) (A32)",
            "el informe declara `llm_overlay: disabled` y `scheduler: none`, y no importa ningun "
            "cliente de red ni el paquete de agentes (A22)",
        ],
    }
    return payload


def analyse(
    *, store: Store, reports_dir: Path, as_of: datetime, write: bool = True
) -> Phase1Report:
    """Consume el artefacto de #69, audita el arnes y (por defecto) escribe el informe (A2).

    ``as_of`` es **obligatorio** y es el unico instante de la corrida: ninguna ruta consulta el
    reloj. ``write=False`` no escribe **nada** y la auditoria de no-*look-ahead* corre sobre una
    copia en memoria de la historia, nunca sobre el `data/` del repositorio (A11, A21).
    """
    moment = _as_utc(as_of)
    artifact = load_input_artifact(reports_dir, store=store)
    artifact_moment = _artifact_moment(artifact)

    rerun = backtest_report.analyse(
        store=store, reports_dir=reports_dir, as_of=artifact_moment, write=False
    )

    command = f"python -m cfdtrader.analysis.backtest_report --as-of {artifact_moment.isoformat()}"
    determinism = audit_determinism(
        expected={
            "report_sha256": artifact.report_sha256,
            "json_sha256": artifact.sha256,
            "md_sha256": _sha256_bytes(artifact.markdown_bytes),
        },
        observed={
            "report_sha256": rerun.report_sha256,
            "json_sha256": _sha256_bytes(rerun.json_text().encode("utf-8")),
            "md_sha256": _sha256_bytes(backtest_report.render_markdown(rerun).encode("utf-8")),
        },
        command=command,
        as_of=artifact_moment.isoformat(),
    )
    lookahead = audit_lookahead(store=store, base=rerun)
    conservation = conservation_audit(rerun=rerun, artifact=artifact)
    costs = costs_audit()
    purge_embargo = purge_embargo_block(artifact=artifact, rerun=rerun)
    rejection = net_metrics_rejection(rerun=rerun)

    payload = _payload(
        as_of=moment,
        artifact=artifact,
        determinism=determinism,
        lookahead=lookahead,
        conservation=conservation,
        costs=costs,
        purge_embargo=purge_embargo,
        rejection=rejection,
    )
    report = Phase1Report(
        as_of=moment,
        report_date=moment.date(),
        payload=payload,
        report_sha256=_digest(payload),
        input_artifact=artifact,
        rerun=rerun,
    )
    if write:
        json_path, markdown_path = report.write(reports_dir)
        logger.info("informe de Fase 1: {} y {}", json_path, markdown_path)
    return report


def _artifact_moment(artifact: InputArtifact) -> datetime:
    """El instante declarado del artefacto de #69: con el se re-ejecuta el motor (A7)."""
    text = artifact.generated_at
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise MalformedInputArtifactError(
            f"`generated_at` del artefacto de #69 no es ISO-8601 ({text!r}): sin instante "
            "declarado no se puede re-ejecutar ni comparar (A7, A23)"
        ) from error
    return _as_utc(parsed)


# ─────────────────────────────────────────────────────────────────────────────
# Informe en prosa (A28)
# ─────────────────────────────────────────────────────────────────────────────
def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Tabla Markdown determinista."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_markdown(report: Phase1Report) -> str:
    """El informe en Markdown: resume la puerta, los seis baselines y los limites (A28)."""
    payload = report.payload
    source = _mapping(payload["source_artifact"], where="source_artifact")
    baselines = _mapping(payload["baselines"], where="baselines")
    halves = _mapping(payload["gate_halves"], where="gate_halves")
    gate = _mapping(payload["gate"], where="gate")
    audit = _mapping(payload["harness_audit"], where="harness_audit")
    limits = _mapping(payload["limits"], where="limits")
    recommendation = _mapping(payload["recommendation"], where="recommendation")

    lines: list[str] = [
        "# Informe de Fase 1 y puerta de salida",
        "",
        f"- **Tarea**: {payload['task']}",
        f"- **Instante declarado (`as_of`)**: `{payload['generated_at']}`",
        f"- **Artefacto consumido**: `{source['path']}` (sha256 `{source['sha256']}`)",
        f"- **`report_sha256` del artefacto de #69**: `{source['report_sha256']}`",
        f"- **`report_sha256` de este informe**: `{report.report_sha256}`",
        f"- **`phase1_ready`**: `{str(payload['phase1_ready']).lower()}` · "
        f"**`is_validation`**: `{str(payload['is_validation']).lower()}`",
        "",
        "> **Este informe no es una validacion de la estrategia.** La puerta de Fase 0 sigue en "
        "`fail` y la tabla es de **coste declarado**, no de rendimiento neto medido.",
        "",
        "## La puerta de salida",
        "",
    ]
    rows: list[list[str]] = []
    for name in cast("list[str]", gate["halves"]):
        half = _mapping(halves[name], where=name)
        rows.append([name, f"`{half['state']}`", f"`{half['evidence']}`", str(half["reason"])])
    rows.append(
        [
            "**agregado**",
            f"**`{gate['aggregate']}`**",
            "mecanico",
            f"regla de #9: {gate['aggregation_rule']}",
        ]
    )
    lines += _table(["mitad", "estado", "procedencia", "motivo"], rows)
    lines += [
        "",
        f"{gate['partial_verdict']}",
        "",
        f"- Verificado hoy: {gate['what_is_verified']}.",
        f"- No evaluable hoy: {gate['what_is_not_evaluable']}.",
        f"- Recomendacion: `{recommendation['value']}` (regla: {recommendation['rule']}).",
        "",
        "## Auditoria del arnes",
        "",
    ]
    audit_rows: list[list[str]] = []
    for name in ("determinism", "lookahead", "conservation", "costs", "net_metrics"):
        block = _mapping(audit[name], where=name)
        audit_rows.append(
            [f"`{name}`", f"`{block['state']}`", f"`{block['evidence']}`", str(block["note"])]
        )
    purification = _mapping(audit["purge_embargo"], where="purge_embargo")
    audit_rows.append(
        [
            "`purge_embargo`",
            f"`{'no-op' if purification['structural_no_op'] else 'activo'}`",
            f"`{purification['evidence']}`",
            str(purification["note"]),
        ]
    )
    lines += _table(["comprobacion", "estado", "procedencia", "nota"], audit_rows)
    lines += [
        "",
        "### Purga y embargo, tal cual los da #12",
        "",
        f"- `purge_total` = {purification['purge_total']}",
        f"- `embargo_total` = {purification['embargo_total']}",
        f"- `embargo_in_train_total` = {purification['embargo_in_train_total']}",
        f"- `exclusions_are_no_op` = `{str(purification['exclusions_are_no_op']).lower()}`",
        f"- `label_horizon` = {purification['label_horizon']}",
        "",
        "## Los seis baselines (copiado del artefacto de #69)",
        "",
    ]
    baseline_rows = [
        [
            f"`{row['baseline']}`",
            str(row["traded"]),
            str(row["no_trade"]),
            str(row["skipped"]),
            str(row["n_test"]),
            f"`{row['run_sha256']}`",
        ]
        for row in cast("list[dict[str, object]]", baselines["rows"])
    ]
    lines += _table(
        ["baseline", "traded", "no_trade", "skipped", "n_test", "run_sha256"], baseline_rows
    )
    lines += [
        "",
        f"Tabla copiada de `{source['path']}` (procedencia `{baselines['evidence']}`); el detalle "
        "completo —reconciliacion del universo, plan, modelo de coste y limites— esta en ese "
        "artefacto, que este informe **no** duplica.",
        "",
        "## Limites heredados (sin fusionar)",
        "",
        f"- Puerta de Fase 0: `{limits['gate']}` · `phase1_ready`: "
        f"`{str(limits['phase1_ready']).lower()}`",
        f"- Costes: `{_mapping(limits['costs'], where='costs')['state']}` · *slippage*: "
        f"`{_mapping(limits['slippage'], where='slippage')['state']}` "
        f"(`is_measurement`: "
        f"`{str(_mapping(limits['slippage'], where='slippage')['is_measurement']).lower()}`)",
        f"- Precios: `{_mapping(limits['prices'], where='prices')['series_id']}` como *proxy* de "
        f"`{_mapping(limits['prices'], where='prices')['proxy_of']}`",
        f"- Corte de financiacion: `{limits['financing_cut']}` "
        f"(verificado: `{str(limits['financing_cut_verified']).lower()}`, "
        f"{limits['financing_cut_issue']})",
        f"- Metricas netas: `{limits['net_metrics_state']}` · LLM: `{limits['llm_overlay']}` · "
        f"scheduler: `{limits['scheduler']}`",
        "",
        "## Como se cierra la mitad no evaluable",
        "",
    ]
    lines += _table(
        ["issue", "que hace falta"],
        [[f"`{entry['issue']}`", str(entry["reason"])] for entry in _each(payload["how_to_close"])],
    )
    lines += ["", "## Bloqueos", ""]
    lines += _table(
        ["codigo", "issues", "motivo"],
        [
            [
                f"`{entry['code']}`",
                ", ".join(str(item) for item in cast("list[object]", entry["issues"])),
                str(entry["reason"]),
            ]
            for entry in _each(payload["blockers"])
        ],
    )
    lines += ["", "## Limites declarados de este informe", ""]
    lines += [f"- {item}" for item in cast("list[str]", payload["limitations"])]
    lines += ["", "## Que no hace este informe", ""]
    lines += [
        f"- **{entry['id']}** ({entry['issue']}): {entry['statement']}"
        for entry in _each(payload["does_not_do"])
    ]
    lines += [""]
    return "\n".join(lines)


def _each(node: object) -> list[dict[str, object]]:
    """Una lista de objetos JSON ya tipada."""
    raw = cast("list[object]", node)
    return [_mapping(item, where="lista") for item in raw]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_as_of(value: str | None) -> datetime:
    """El instante declarado de la CLI: **obligatorio**, y nunca tomado del reloj (A2, A3)."""
    if value is None:
        raise MissingAsOfError(
            "falta --as-of: escribir el informe exige un instante declarado (el modulo no lee el "
            "reloj) y sin el **no se escribe ningun fichero** (A3)"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise InvalidAsOfError(f"--as-of no es un ISO-8601 valido: {error} (A3)") from error
    return _as_utc(parsed)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del informe de Fase 1.

    Codigos de salida: ``0`` = informe escrito (aunque la puerta **no** se supere, que es un
    resultado legitimo y declarado); ``2`` = falta o no es valido ``--as-of``, falta el artefacto
    de #69, hay ambiguedad, el artefacto no es legible o una identidad no cuadra ⇒ **no se
    escribe nada** y el motivo sale por ``stderr``.
    """
    parser = argparse.ArgumentParser(
        prog="cfdtrader.analysis.phase1_report",
        description="Informe de Fase 1 y puerta de salida (auditoria del arnes + veredicto)",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="raiz del almacen")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="directorio de informes (por defecto <data-root>/derived/reports)",
    )
    parser.add_argument("--settings", type=Path, default=None, help="ruta de settings.yaml")
    parser.add_argument(
        "--as-of",
        default=None,
        help="instante declarado ISO-8601, obligatorio para escribir (el modulo no lee el reloj)",
    )
    args = parser.parse_args(argv)

    try:
        moment = _parse_as_of(cast("str | None", args.as_of))
        settings = load_settings(args.settings)
    except (Phase1ReportError, ConfigurationError) as error:
        print(f"no se puede emitir el informe de Fase 1: {error}", file=sys.stderr)
        return 2

    data_root = Path(args.data_root) if args.data_root is not None else Path(settings.data.root)
    reports_dir = (
        Path(args.reports_dir)
        if args.reports_dir is not None
        else data_root / "derived" / "reports"
    )
    try:
        report = analyse(store=Store(data_root), reports_dir=reports_dir, as_of=moment, write=True)
    except (Phase1ReportError, backtest_report.BacktestReportError) as error:
        print(f"no se puede emitir el informe de Fase 1: {error}", file=sys.stderr)
        return 2

    gate = cast("dict[str, object]", report.payload["gate"])
    logger.info(
        "Fase 1: mitad (a) {}, mitad (b) {}, agregado {}; phase1_ready {}; report_sha256 = {}",
        cast("dict[str, object]", report.payload["gate_halves"])["determinism"],
        cast("dict[str, object]", report.payload["gate_halves"])["always_long"],
        gate["aggregate"],
        report.payload["phase1_ready"],
        report.report_sha256,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
