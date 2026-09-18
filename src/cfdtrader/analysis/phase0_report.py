"""Informe determinista de Fase 0 y decisión de continuidad (`tasks.md`, tarea 9) — tarea #9.

Consume los **tres artefactos ya producidos** —drift (#6), volatilidad (#7) y costes
(#8)— desde `data/derived/reports/`, aplica **mecánicamente** la doble puerta de salida
de la Fase 0, calcula ``p* = (R + c) / 2R`` con el coste **declarado** y deja por escrito
y razonada la decisión de **continuar, reencuadrar o parar**.

Qué **decide** este módulo:

- la mitad **(a)** de la puerta, leyendo ``verdict`` y ``phase0_gate`` del artefacto del
  drift con la tabla de correspondencia declarada en :data:`HALF_A_MAPPING` (nunca con la
  prosa del informe);
- la mitad **(b)**, leyendo ``phase0_gate_b`` del artefacto de costes;
- el veredicto agregado, con la regla declarada en :data:`GATE_AGGREGATION_RULE`;
- la recomendación, sujeta a :data:`RECOMMENDATION_CONSISTENCY_RULE`, y ``phase1_ready``
  con su lista de ``blockers`` legible por máquina.

Qué **no** decide, y por tanto no puede inventar:

- el **modelo de coste del motor de backtest**: es #11
  (``src/cfdtrader/backtest/costs.py``); aquí solo se reproduce la tabla declarada de #8;
- el **bróker definitivo** (decisión abierta 4 -> #59), los **umbrales** —incluido el
  tamaño de ``R``— (decisión abierta 5 -> #60) y el **precio de entrada exacto**
  (decisión abierta 6 -> #61). Se publican en ``open_decisions`` con
  ``state: "unresolved"``. En particular, ``R`` sin decidir deja la mitad (b) en
  ``not_evaluable`` **aunque el *slippage* llegue a medirse**;
- la medición del ***slippage*** real (10-15 ejecuciones a la apertura -> #62), la fuente
  de bid/ask del CFD (-> #50) y el arreglo del `open` repetido del índice (-> #52).

Un ``not_evaluable`` **nunca** es un ``pass``: convertirlo en un «adelante» es el error más
caro que puede cometer este informe, y :func:`aggregate_gate` lo hace imposible por
construcción.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, cast

from loguru import logger

from cfdtrader.data.settings import ConfigurationError, load_settings

__all__ = [
    "ARTIFACT_CLASSES",
    "DECLARED_CONSTANTS",
    "FILE_SELECTION_RULE",
    "GATE_AGGREGATION_RULE",
    "HALF_A_MAPPING",
    "MEASURE_KEYS",
    "OPEN_DECISIONS",
    "RECOMMENDATION_CONSISTENCY_RULE",
    "WHAT_WOULD_CHANGE_THE_VERDICT",
    "AmbiguousArtifactError",
    "Artifact",
    "ArtifactClass",
    "GateVerdict",
    "HalfResult",
    "InputConflictError",
    "InputShapeError",
    "Phase0Inputs",
    "Phase0Report",
    "Phase0ReportError",
    "Recommendation",
    "aggregate_gate",
    "analyse",
    "artifact_from_payload",
    "confidence_interval",
    "consolidate",
    "declared_value",
    "half_a_state",
    "load_inputs",
    "main",
    "p_star",
    "recommendation_is_consistent",
    "render_markdown",
    "required_operations",
    "select_artifact",
    "threshold_breached",
]

# ─────────────────────────────────────────────────────────────────────────────
# Regla de selección de fichero (A1) — declarada, no implícita
# ─────────────────────────────────────────────────────────────────────────────
#: Cómo se elige el artefacto de cada clase cuando no se pasan rutas explícitas.
#: La fecha se lee del **sufijo del nombre**, nunca de `mtime` ni del contenido.
FILE_SELECTION_RULE: Final[str] = (
    "de los ficheros de esa clase en el directorio de informes, el de fecha más reciente "
    "del sufijo `<nombre>_<AAAA-MM-DD>.json`; si dos ficheros de la misma clase resuelven a "
    "la misma fecha, es una ambigüedad y se aborta con código 2"
)

#: Formato de la fecha del sufijo del nombre del artefacto.
ARTIFACT_DATE_FORMAT: Final[str] = "%Y-%m-%d"

#: Nivel de significación con el que se lee la significación de los tramos.
SIGNIFICANCE_LEVEL: Final[Decimal] = Decimal("0.05")

#: Fórmula de break-even que se aplica (`plan.md` §4.4).
P_STAR_FORMULA: Final[str] = "p* = (R + c) / 2R"

#: Regla de agregación de las dos mitades (`tasks.md`, nota de la Fase 0).
GATE_AGGREGATION_RULE: Final[str] = (
    "`fail` si **alguna** mitad es `fail`; `not_evaluable` si ninguna es `fail` y al menos "
    "una es `not_evaluable`; `pass` **solo** si las dos son `pass`"
)

#: Regla de consistencia entre veredicto y recomendación (A26).
RECOMMENDATION_CONSISTENCY_RULE: Final[str] = (
    "si el veredicto agregado no es `pass`, la recomendación **no puede** ser `continue`"
)

#: Tabla de correspondencia declarada de la mitad (a) (A9). Cualquier otro par es `not_evaluable`.
HALF_A_MAPPING: Final[tuple[tuple[str, str, str], ...]] = (
    ("overnight", "fail", "fail"),
    ("intraday", "pass", "pass"),
)

#: Qué se hace con un par (`verdict`, `phase0_gate`) que no esté en la tabla.
HALF_A_FALLBACK: Final[str] = "cualquier otro par (`verdict`, `phase0_gate`) ⇒ `not_evaluable`"

#: Las tres medidas de #8, que se publican **por separado y nunca sumadas** (A17).
MEASURE_KEYS: Final[tuple[str, ...]] = (
    "spread_cotizado",
    "tracking_difference",
    "slippage_ejecucion",
)

#: Las dos formulaciones de la puerta (a) que conviven en la documentación (A11).
GATE_A_WORDINGS: Final[tuple[dict[str, str], ...]] = (
    {
        "source": "`tasks.md`, nota de la Fase 0 y descripción de la tarea 9",
        "wording": (
            "si el drift se concentra en `close→open` en vez de en `open→close`, no se pasa "
            "a Fase 1"
        ),
    },
    {
        "source": "`plan.md` §1.1.a",
        "wording": (
            "si el `open→close` tiene drift negativo o nulo mientras el nocturno lo tiene "
            "positivo, la estrategia debe rehacerse o abandonarse"
        ),
    },
)

#: Cuál de las dos redacciones se aplica con estos artefactos (A11). El «por qué» se
#: compone con los números del artefacto, nunca con literales.
GATE_A_WORDING_APPLIED: Final[str] = "`plan.md` §1.1.a"

#: Limitaciones de la mitad (a), con su issue cuando la tiene (A13).
GATE_A_LIMITATIONS: Final[tuple[dict[str, str], ...]] = (
    {
        "limitation": (
            "la medición es sobre `^GSPC`, el subyacente, y **no** sobre el `SPX500:CFD`: el "
            "signo de la conclusión se traslada, la magnitud no (diferencial y financiación)"
        ),
        "issue": "#50",
    },
    {
        "limitation": (
            "el `open` del índice está degradado antes del corte (`open` = cierre anterior "
            "repetido): la contaminación se declara año a año y el veredicto se calcula sobre "
            "la muestra limpia"
        ),
        "issue": "#52",
    },
    {
        "limitation": (
            "el tramo nocturno del **sistema** es cero por construcción (opera intradía puro, "
            "sin overnight): esto describe dónde está el retorno del índice, **no** el P&L del "
            "sistema"
        ),
        "issue": "",
    },
    {
        "limitation": (
            "el precio de entrada real es del CFD y está sin decidir (subasta frente a unos "
            "minutos después)"
        ),
        "issue": "#61",
    },
    {
        "limitation": (
            "el `fail` se apoya en la **asimetría de significación por tramo** y en la condición "
            "pre-registrada, **no** en una diferencia demostrada entre los dos tramos"
        ),
        "issue": "",
    },
)

#: Decisiones abiertas que este informe declara sin resolver (A4).
OPEN_DECISIONS: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "4",
        "name": "Bróker definitivo",
        "state": "unresolved",
        "issue": "#59",
        "missing_information": (
            "medición del *spread* real y del instante de corte de la financiación en el bróker "
            "definitivo (`plan.md` §8.5)"
        ),
        "depends": (
            "la procedencia de la tabla de costes declarada —y con ella el `p*` de este "
            "informe— y el corte de financiación, que hoy sale sin verificar (#59)"
        ),
    },
    {
        "id": "5",
        "name": (
            "Umbrales concretos: EV mínimo, riesgo por operación, pérdida máxima diaria y "
            "tamaño de `R`"
        ),
        "state": "unresolved",
        "issue": "#60",
        "missing_information": (
            "el valor de `R` (amplitud del bracket) y el resto de umbrales, que solo puede fijar "
            "el propietario"
        ),
        "depends": (
            "la mitad (b) **entera**: el umbral se aplica como `slippage_pct / R_pct` y sin `R` "
            "no hay comparación posible (#60). Los escenarios de `p*` son escenarios declarados "
            "de `plan.md` §4.4, no una decisión del propietario"
        ),
    },
    {
        "id": "6",
        "name": (
            "Horizonte y precio de entrada exactos (`open` de la subasta frente a unos minutos "
            "después)"
        ),
        "state": "unresolved",
        "issue": "#61",
        "missing_information": (
            "si la entrada es el `open` de la subasta o la ejecución real unos minutos después "
            "(`plan.md` §4.1 y §21)"
        ),
        "depends": (
            "la interpretación del *slippage* medido —y por tanto la mitad (b)— y el horizonte "
            "del etiquetado tri-barra de la Fase 1 (#61)"
        ),
    },
)

#: Constantes declaradas (nombre, valor y procedencia) que **sí** son literales del
#: proyecto (A2). Las que provienen de un artefacto se añaden en tiempo de consolidación.
DECLARED_CONSTANTS: Final[tuple[dict[str, str], ...]] = (
    {
        "name": "r_scenarios_pct",
        "value": "0.5 / 1.0 / 1.5",
        "unit": "% de amplitud del bracket",
        "provenance": "`plan.md` §4.4 (tabla de `p*`)",
        "note": (
            "**escenario declarado, no decisión del propietario**: el tamaño de `R` es la "
            "decisión abierta 5 (-> #60)"
        ),
    },
    {
        "name": "z_alpha_two_sided",
        "value": "1.96",
        "unit": "adimensional",
        "provenance": "`plan.md` §4.5",
        "note": "95 % de confianza (dos colas)",
    },
    {
        "name": "z_beta",
        "value": "0.84",
        "unit": "adimensional",
        "provenance": "`plan.md` §4.5",
        "note": "80 % de potencia",
    },
    {
        "name": "edge_to_detect",
        "value": "0.52",
        "unit": "probabilidad",
        "provenance": "`plan.md` §4.5",
        "note": "edge real que se quiere distinguir del azar",
    },
    {
        "name": "null_probability",
        "value": "0.50",
        "unit": "probabilidad",
        "provenance": "`plan.md` §4.5",
        "note": "hipótesis nula del contraste de potencia",
    },
    {
        "name": "sample_operations",
        "value": "300",
        "unit": "operaciones",
        "provenance": "`plan.md` §4.5",
        "note": "tamaño de muestra cuyo intervalo de confianza se publica",
    },
    {
        "name": "sessions_per_year",
        "value": "250",
        "unit": "sesiones/año",
        "provenance": "`plan.md` §4.5",
        "note": "ritmo declarado para convertir operaciones en años",
    },
    {
        "name": "significance_level",
        "value": "0.05",
        "unit": "p-valor",
        "provenance": "`plan.md` §4.5 y §8.5 (95 % de confianza)",
        "note": "con este listón se lee la significación de cada tramo del artefacto del drift",
    },
    {
        "name": "p_star_decimals",
        "value": "0.01",
        "unit": "% de probabilidad",
        "provenance": "`plan.md` §4.4",
        "note": "dos decimales de porcentaje, la convención de la tabla declarada",
    },
    {
        "name": "interval_decimals",
        "value": "0.001",
        "unit": "probabilidad",
        "provenance": "`plan.md` §4.5",
        "note": (
            "los extremos del intervalo se publican **truncados** (hacia cero) a tres decimales, "
            "que es la convención con la que `plan.md` §4.5 publica `[0,463; 0,576]`"
        ),
    },
    {
        "name": "derogated_p_star_gate",
        "value": "p* > 60 % ⇒ parar",
        "unit": "% de probabilidad",
        "provenance": "`tasks.md`, nota de la Fase 0, y `plan.md` §16 (`plan.md` §4.6)",
        "note": "**derogada**: con el coste declarado, cualquier modelo sesgado la superaría",
    },
    {
        "name": "slippage_dominance_declared",
        "value": "El *slippage* de 20 pb pesa ~50× el diferencial declarado",
        "unit": "pb frente a pb",
        "provenance": "`plan.md` §4.4",
        "note": (
            "es una afirmación del documento, no una medición; el diferencial que se cita sale "
            "del artefacto de #8"
        ),
    },
)

#: Condiciones que cambiarían el veredicto (A28): comprobables, con mitad, dueño y enlace.
WHAT_WOULD_CHANGE_THE_VERDICT: Final[tuple[dict[str, str], ...]] = (
    {
        "condition": (
            "medir el drift sobre el **CFD** con el `open` real del bróker, en lugar del índice "
            "y de una subasta reconstruida"
        ),
        "half": "a",
        "owner": "#9 (dueño de este informe) con #6 (estudio del drift)",
        "issues": "#50, #52",
    },
    {
        "condition": (
            "medir el ***slippage*** de ejecución (10-15 ejecuciones a la apertura) **y** cerrar "
            "la decisión 5 para tener `R`: solo entonces la mitad (b) es evaluable"
        ),
        "half": "b",
        "owner": "#62 (medición) y #60 (decisión de `R`)",
        "issues": "#62, #60",
    },
    {
        "condition": (
            "una muestra limpia más larga en la que el tramo intradía tenga drift propio "
            "significativo (hoy la significación está del lado nocturno)"
        ),
        "half": "a",
        "owner": "#6 (estudio del drift) y #52 (fuente)",
        "issues": "#52",
    },
)

#: Limitaciones que afectan a las dos mitades.
CROSS_LIMITATIONS: Final[tuple[dict[str, str], ...]] = (
    {
        "statement": (
            "el sistema **no** será validable por resultado en un plazo razonable: con la "
            "muestra disponible, un acierto del 52 % es indistinguible del azar (`plan.md` §4.6)"
        ),
        "half": "ambas",
        "issue": "",
    },
    {
        "statement": (
            "la consolidación depende de que los tres artefactos se hayan generado con el "
            "mismo corte de muestra limpia; si divergen, el informe aborta en vez de elegir uno"
        ),
        "half": "ambas",
        "issue": "#52",
    },
)


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulario del veredicto
# ─────────────────────────────────────────────────────────────────────────────
class HalfResult(StrEnum):
    """Estado de una mitad de la puerta. `not_evaluable` **no** es un aprobado."""

    # `pass` es el vocabulario declarado del veredicto (`tasks.md`, tarea 9), no una
    # credencial: S105 lo confunde con un nombre de contraseña.
    PASS = "pass"  # noqa: S105
    FAIL = "fail"
    NOT_EVALUABLE = "not_evaluable"


class GateVerdict(StrEnum):
    """Veredicto agregado de la doble puerta."""

    PASS = "pass"  # noqa: S105
    FAIL = "fail"
    NOT_EVALUABLE = "not_evaluable"


class Recommendation(StrEnum):
    """Recomendación de continuidad, sujeta a la regla de consistencia."""

    CONTINUE = "continue"
    REFRAME = "reframe"
    STOP = "stop"


class BlockerCode(StrEnum):
    """Códigos de ``blockers`` buscables por máquina (A24)."""

    DRIFT_OVERNIGHT = "drift_overnight"
    DRIFT_NOT_EVALUABLE = "drift_not_evaluable"
    SLIPPAGE_UNMEASURED = "slippage_unmeasured"
    R_UNDECIDED = "r_undecided"
    BROKER_UNDECIDED = "broker_undecided"
    FINANCING_CUT_UNVERIFIED = "financing_cut_unverified"
    ENTRY_PRICE_UNDECIDED = "entry_price_undecided"


# ─────────────────────────────────────────────────────────────────────────────
# Errores declarados: el CLI sale con 2, el motivo por `stderr` y sin informe
# ─────────────────────────────────────────────────────────────────────────────
class Phase0ReportError(Exception):
    """Error declarado del informe de Fase 0: nunca se rellena el hueco con un valor."""


class MissingArtifactError(Phase0ReportError):
    """Falta un artefacto de entrada."""


class AmbiguousArtifactError(Phase0ReportError):
    """Dos artefactos de la misma clase resuelven a la misma fecha."""


class InputConflictError(Phase0ReportError):
    """Los artefactos no son coherentes entre sí.

    El motivo viaja como un objeto ``input_conflict`` con el detalle: cuando los artefactos
    se contradicen **no se escribe informe**, así que el conflicto declarado solo puede
    publicarse por el canal de error (A6).
    """

    def __init__(self, detail: str, *, conflict: dict[str, Any]) -> None:
        emitted: dict[str, Any] = {"input_conflict": {"kind": "input_conflict", **conflict}}
        super().__init__(f"{detail} — {json.dumps(emitted, ensure_ascii=False)}")
        self.conflict: dict[str, Any] = emitted["input_conflict"]


class InputShapeError(Phase0ReportError):
    """Un artefacto no trae un campo obligatorio, o lo trae con un tipo inesperado."""


class VerdictError(Phase0ReportError):
    """El veredicto pedido rompe una regla declarada (vocabulario o consistencia)."""


# ─────────────────────────────────────────────────────────────────────────────
# Entradas: los tres artefactos, leídos de disco o construidos en memoria
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ArtifactClass:
    """Una de las tres clases de artefacto que consume el informe."""

    kind: str
    pattern: str
    description: str


ARTIFACT_CLASSES: Final[tuple[ArtifactClass, ...]] = (
    ArtifactClass("drift", "drift_decomposition_*.json", "descomposición del drift (#6)"),
    ArtifactClass("volatility", "volatility_forecast_*.json", "volatilidad y forecast (#7)"),
    ArtifactClass("costs", "cost_audit_*.json", "auditoría de costes declarados (#8)"),
)

#: Claves donde un artefacto declara su instante de referencia.
_AS_OF_KEYS: Final[tuple[str, ...]] = ("as_of", "as_of_utc")


@dataclass(frozen=True, slots=True)
class Artifact:
    """Un artefacto de entrada, con su procedencia y su huella."""

    kind: str
    path: str
    date: str
    sha256: str
    sha256_of: str
    payload: dict[str, Any]

    @property
    def as_of(self) -> str | None:
        """`as_of` que declara el artefacto, si lo trae."""
        for key in _AS_OF_KEYS:
            value = self.payload.get(key)
            if isinstance(value, str):
                return value
        return None

    @property
    def source(self) -> str | None:
        """`source` que declara el artefacto, si lo trae."""
        value = self.payload.get("source")
        return value if isinstance(value, str) else None

    @property
    def series_id(self) -> str | None:
        """`series_id` que declara el artefacto, si lo trae."""
        value = self.payload.get("series_id")
        return value if isinstance(value, str) else None

    def provenance(self) -> dict[str, Any]:
        """Bloque de procedencia que publica el informe (A3)."""
        return {
            "kind": self.kind,
            "path": self.path,
            "artifact_date": self.date,
            "as_of": self.as_of,
            "source": self.source,
            "series_id": self.series_id,
            "sha256": self.sha256,
            "sha256_of": self.sha256_of,
        }


@dataclass(frozen=True, slots=True)
class Phase0Inputs:
    """Las tres entradas del informe, ya cargadas: la frontera entre disco y cálculo."""

    drift: Artifact
    volatility: Artifact
    costs: Artifact

    @classmethod
    def from_payloads(
        cls,
        *,
        drift: dict[str, Any],
        volatility: dict[str, Any],
        costs: dict[str, Any],
    ) -> Phase0Inputs:
        """Construye las entradas **en memoria**, sin tocar disco (A31)."""
        return cls(
            drift=artifact_from_payload("drift", drift),
            volatility=artifact_from_payload("volatility", volatility),
            costs=artifact_from_payload("costs", costs),
        )


def artifact_from_payload(
    kind: str,
    payload: dict[str, Any],
    *,
    path: str = "<memoria>",
    artifact_date: str | None = None,
) -> Artifact:
    """Artefacto construido desde un *payload* en memoria (consolidación pura y pruebas).

    La huella es la del texto JSON canónico. Para un artefacto leído de disco la huella es
    la del **fichero**, como corresponde a la procedencia que publica el informe.
    """
    if artifact_date is None:
        as_of = _as_of_of(payload)
        artifact_date = as_of[:10] if as_of is not None else "sin-fecha"
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return Artifact(
        kind=kind,
        path=f"{path}:{kind}.json",
        date=artifact_date,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        sha256_of="payload",
        payload=payload,
    )


def _as_of_of(payload: Mapping[str, Any]) -> str | None:
    """`as_of` (o `as_of_utc`) de un *payload*, si es una cadena."""
    for key in _AS_OF_KEYS:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def _artifact_date_of(name: str, klass: ArtifactClass) -> date | None:
    """Fecha del sufijo del nombre, o `None` si el nombre no es de esa clase."""
    prefix = klass.pattern.removesuffix("*.json")
    suffix = ".json"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    raw = name[len(prefix) : -len(suffix)]
    try:
        return datetime.strptime(raw, ARTIFACT_DATE_FORMAT).date()
    except ValueError:
        return None


def select_artifact(directory: Path, klass: ArtifactClass) -> Path:
    """Elige el artefacto de esa clase aplicando :data:`FILE_SELECTION_RULE`.

    Ausencia y ambigüedad son errores declarados: nunca se rellena el hueco.
    """
    if not directory.is_dir():
        raise MissingArtifactError(
            f"no existe el directorio de informes {directory}: {FILE_SELECTION_RULE}"
        )
    candidates: list[tuple[date, Path]] = []
    for path in sorted(directory.glob(klass.pattern)):
        if not path.is_file():
            continue
        day = _artifact_date_of(path.name, klass)
        if day is None:
            continue
        candidates.append((day, path))
    if not candidates:
        raise MissingArtifactError(
            f"falta el artefacto de {klass.description} en {directory} "
            f"(patrón `{klass.pattern}`): {FILE_SELECTION_RULE}. No se rellena con un valor "
            "por defecto."
        )
    latest = max(day for day, _ in candidates)
    matches = [path for day, path in candidates if day == latest]
    if len(matches) > 1:
        names = ", ".join(sorted(path.name for path in matches))
        raise AmbiguousArtifactError(
            f"hay {len(matches)} artefactos de {klass.description} con la misma fecha "
            f"({latest.isoformat()}): {names}. Ambigüedad declarada: no se elige ninguno."
        )
    return matches[0]


def _read_artifact(path: Path, klass: ArtifactClass, day: date) -> Artifact:
    """Lee y valida un artefacto de disco."""
    raw = path.read_bytes()
    try:
        loaded: Any = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InputShapeError(f"el artefacto {path} no es JSON legible: {error}") from error
    if not isinstance(loaded, dict):
        raise InputShapeError(
            f"el artefacto {path} no es un objeto JSON: el informe consume objetos con campos."
        )
    return Artifact(
        kind=klass.kind,
        path=str(path),
        date=day.isoformat(),
        sha256=hashlib.sha256(raw).hexdigest(),
        sha256_of="fichero",
        payload=cast("dict[str, Any]", loaded),
    )


def load_inputs(reports_dir: Path) -> Phase0Inputs:
    """Selecciona, lee y valida los tres artefactos. Solo lee: no escribe nada."""
    artifacts: dict[str, Artifact] = {}
    for klass in ARTIFACT_CLASSES:
        path = select_artifact(reports_dir, klass)
        day = _artifact_date_of(path.name, klass)
        if day is None:  # pragma: no cover - `select_artifact` ya lo garantiza
            raise InputShapeError(f"el nombre de {path} no lleva fecha legible")
        artifacts[klass.kind] = _read_artifact(path, klass, day)
    return Phase0Inputs(
        drift=artifacts["drift"],
        volatility=artifacts["volatility"],
        costs=artifacts["costs"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aritmética exacta (A19 y A21): `Decimal`, verificable a mano
# ─────────────────────────────────────────────────────────────────────────────
def p_star(r_pct: Decimal, c_pct: Decimal) -> Decimal:
    """`(R + c) / 2R` con aritmética **exacta**. Entran y salen porcentajes.

    Devuelve la **probabilidad** (fracción), no el porcentaje.
    """
    if r_pct <= 0:
        raise VerdictError(f"`R` debe ser positivo para calcular `p*`: {r_pct}")
    hundred = Decimal(100)
    r = r_pct / hundred
    c = c_pct / hundred
    return (r + c) / (2 * r)


def required_operations(
    *,
    edge: Decimal,
    null_probability: Decimal,
    z_alpha: Decimal,
    z_beta: Decimal,
) -> Decimal:
    """Operaciones necesarias para distinguir `edge` de `null_probability` (`plan.md` §4.5)."""
    numerator = (z_alpha + z_beta) ** 2 * edge * (1 - edge)
    return numerator / (edge - null_probability) ** 2


def confidence_interval(
    *, observed: Decimal, operations: int, z: Decimal, decimals: Decimal
) -> tuple[Decimal, Decimal]:
    """Intervalo de confianza de Wald de una proporción, con los extremos **truncados**.

    El truncado (hacia cero) es la convención declarada en :data:`DECLARED_CONSTANTS`: es la
    que reproduce los extremos que publica `plan.md` §4.5.
    """
    if operations <= 0:
        raise VerdictError(f"el número de operaciones debe ser positivo: {operations}")
    standard_error = (observed * (1 - observed) / Decimal(operations)).sqrt()
    half_width = z * standard_error
    return (
        (observed - half_width).quantize(decimals, rounding=ROUND_DOWN),
        (observed + half_width).quantize(decimals, rounding=ROUND_DOWN),
    )


def threshold_breached(
    *, slippage_pct: Decimal, r_pct: Decimal, threshold_pct_of_r: Decimal
) -> bool:
    """¿Incumple el *slippage* el umbral declarado, **en unidades comparables**? (A16)

    La comparación es `slippage_pct / R_pct` frente al umbral declarado, no una resta de
    magnitudes distintas. Se incumple si lo supera **estrictamente**.
    """
    if r_pct <= 0:
        raise VerdictError(f"`R` debe ser positivo para aplicar el umbral: {r_pct}")
    return slippage_pct / r_pct * Decimal(100) > threshold_pct_of_r


def _dec_str(value: Decimal) -> str:
    """Cadena decimal exacta y sin notación científica."""
    return format(value, "f")


def _trim(value: Decimal) -> Decimal:
    """Sin ceros finales en la parte decimal (la división los arrastra del contexto)."""
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return Decimal(text)


def _pct_str(value: Decimal, decimals: Decimal) -> str:
    """Porcentaje con la convención declarada de decimales."""
    return _dec_str(value.quantize(decimals))


def declared_value(name: str) -> str:
    """Valor de una constante declarada del proyecto, por nombre."""
    for entry in DECLARED_CONSTANTS:
        if entry["name"] == name:
            return entry["value"]
    raise VerdictError(f"no hay ninguna constante declarada llamada `{name}`")


def _r_scenario_values() -> list[str]:
    """Los escenarios de `R` declarados, leídos de :data:`DECLARED_CONSTANTS`."""
    return [part.strip() for part in declared_value("r_scenarios_pct").split("/")]


# ─────────────────────────────────────────────────────────────────────────────
# Veredicto: mitad (a), mitad (b) y agregación
# ─────────────────────────────────────────────────────────────────────────────
def half_a_state(verdict: str, phase0_gate: str) -> HalfResult:
    """Aplica la tabla de correspondencia declarada de la mitad (a) (A9).

    El veredicto **no** se deriva de la prosa del artefacto: se leen `verdict` y
    `phase0_gate` y se traducen con :data:`HALF_A_MAPPING`.
    """
    for artifact_verdict, artifact_gate, state in HALF_A_MAPPING:
        if verdict == artifact_verdict and phase0_gate == artifact_gate:
            return HalfResult(state)
    return HalfResult.NOT_EVALUABLE


def aggregate_gate(half_a: str, half_b: str) -> GateVerdict:
    """Aplica :data:`GATE_AGGREGATION_RULE`. `not_evaluable` nunca se convierte en `pass`."""
    vocabulary = {member.value for member in HalfResult}
    for half in (half_a, half_b):
        if half not in vocabulary:
            raise VerdictError(
                f"estado de mitad fuera del vocabulario declarado: {half!r}; "
                f"se admite {sorted(vocabulary)}"
            )
    if half_a == HalfResult.FAIL or half_b == HalfResult.FAIL:
        return GateVerdict.FAIL
    if half_a != HalfResult.PASS or half_b != HalfResult.PASS:
        return GateVerdict.NOT_EVALUABLE
    return GateVerdict.PASS


def recommendation_is_consistent(gate: GateVerdict, recommendation: Recommendation) -> bool:
    """Regla de consistencia declarada (A26): con `gate != pass` no se recomienda `continue`."""
    if gate is GateVerdict.PASS:
        return True
    return recommendation is not Recommendation.CONTINUE


# ─────────────────────────────────────────────────────────────────────────────
# Lectura tipada de los artefactos: un campo que falta es un error, no un cero
# ─────────────────────────────────────────────────────────────────────────────
_MISSING: Final[object] = object()


def _get(node: Any, *keys: str, default: Any = _MISSING) -> Any:
    """Valor anidado de un JSON ya cargado, o `default` si el camino no existe."""
    current: Any = node
    for key in keys:
        if not isinstance(current, dict):
            return default
        mapping = cast("dict[str, Any]", current)
        if key not in mapping:
            return default
        current = mapping[key]
    return current


def _require(node: Any, *keys: str) -> Any:
    """Campo obligatorio del artefacto. Ausente o nulo ⇒ error declarado."""
    value = _get(node, *keys)
    if value is _MISSING or value is None:
        raise InputShapeError(
            f"el artefacto no trae el campo obligatorio `{'.'.join(keys)}`: el informe no lo "
            "rellena con un valor por defecto."
        )
    return value


def _require_str(node: Any, *keys: str) -> str:
    """Campo obligatorio de tipo cadena."""
    value = _require(node, *keys)
    if not isinstance(value, str):
        raise InputShapeError(f"el campo `{'.'.join(keys)}` debería ser una cadena: {value!r}")
    return value


def _require_decimal(node: Any, *keys: str) -> Decimal:
    """Campo obligatorio que es un importe o un porcentaje declarado."""
    value = _require(node, *keys)
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise InputShapeError(
            f"el campo `{'.'.join(keys)}` debería ser un número o una cadena decimal: {value!r}"
        )
    try:
        return Decimal(str(value))
    except ArithmeticError as error:  # pragma: no cover - rama defensiva
        raise InputShapeError(
            f"el campo `{'.'.join(keys)}` no es un decimal válido: {value!r}"
        ) from error


def _require_float(node: Any, *keys: str) -> float:
    """Campo obligatorio numérico."""
    value = _require(node, *keys)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputShapeError(f"el campo `{'.'.join(keys)}` debería ser numérico: {value!r}")
    return float(value)


def _require_int(node: Any, *keys: str) -> int:
    """Campo obligatorio entero."""
    value = _require(node, *keys)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputShapeError(f"el campo `{'.'.join(keys)}` debería ser entero: {value!r}")
    return int(value)


def _require_list(node: Any, *keys: str) -> list[Any]:
    """Campo obligatorio de tipo lista."""
    value = _require(node, *keys)
    if not isinstance(value, list):
        raise InputShapeError(f"el campo `{'.'.join(keys)}` debería ser una lista: {value!r}")
    return cast("list[Any]", value)


def _require_mapping(node: Any, *keys: str) -> dict[str, Any]:
    """Campo obligatorio de tipo objeto, copiado para que nadie lo mute por referencia."""
    value = _require(node, *keys)
    if not isinstance(value, dict):
        raise InputShapeError(f"el campo `{'.'.join(keys)}` debería ser un objeto: {value!r}")
    return deepcopy(cast("dict[str, Any]", value))


# ─────────────────────────────────────────────────────────────────────────────
# Bloques del informe
# ─────────────────────────────────────────────────────────────────────────────
def _sample_consistency(inputs: Phase0Inputs) -> dict[str, Any]:
    """Una sola muestra limpia (A6): cortes que coinciden y misma definición."""
    drift_cut = _require_str(inputs.drift.payload, "clean_from")
    volatility_cut = _require_str(inputs.volatility.payload, "sample", "clean_from")
    drift_sessions = _require_int(inputs.drift.payload, "clean_sessions")
    volatility_sessions = _require_int(inputs.volatility.payload, "sample", "clean_sessions")
    drift_stale = _require_float(inputs.drift.payload, "stale_open_share")
    volatility_stale = _require_float(inputs.volatility.payload, "sample", "stale_open_share")
    if drift_cut != volatility_cut:
        raise InputConflictError(
            "los dos artefactos que deciden sobre la muestra limpia no coinciden en el corte "
            f"(`clean_from`): drift = {drift_cut}, volatilidad = {volatility_cut}. No se "
            "promedia ni se elige uno: el corte debe salir de la misma definición "
            "(`cfdtrader.analysis.drift.clean_sample_cutoff`).",
            conflict={
                "field": "clean_from",
                "drift": drift_cut,
                "volatility": volatility_cut,
                "shared_definition": "cfdtrader.analysis.drift.clean_sample_cutoff",
                "action": "código 2 y ningún informe escrito: no se promedia ni se elige uno",
            },
        )
    return {
        "clean_from": drift_cut,
        "definition": (
            "el corte de la muestra limpia se calcula una sola vez, en "
            "`cfdtrader.analysis.drift.clean_sample_cutoff` (primer 1 de enero desde el que "
            "**todos** los años posteriores cumplen la tolerancia de `open` repetido, #52), y "
            "`analysis/volatility_forecast.py` la **importa** en lugar de reimplementarla"
        ),
        "drift_clean_from": drift_cut,
        "volatility_clean_from": volatility_cut,
        "consistent": True,
        "drift_clean_sessions": drift_sessions,
        "volatility_clean_sessions": volatility_sessions,
        "clean_sessions_match": drift_sessions == volatility_sessions,
        "stale_open_share_match": drift_stale == volatility_stale,
    }


def _segment_rows(payload: Any, key: str) -> list[dict[str, Any]]:
    """Tramos del artefacto, con la significación recalculada con el listón declarado."""
    rows: list[dict[str, Any]] = []
    for item in _require_list(payload, key):
        p_value = _require_float(item, "p_value")
        rows.append(
            {
                "name": _require_str(item, "name"),
                "sessions": _require_int(item, "sessions"),
                "mean_bp": _require_float(item, "mean_bp"),
                "median_bp": _require_float(item, "median_bp"),
                "std_bp": _require_float(item, "std_bp"),
                "t_stat": _require_float(item, "t_stat"),
                "p_value": p_value,
                "hit_rate": _require_float(item, "hit_rate"),
                "significant": Decimal(str(p_value)) < SIGNIFICANCE_LEVEL,
            }
        )
    return rows


def _gate_a(inputs: Phase0Inputs) -> dict[str, Any]:
    """Mitad (a): el drift. Se resuelve mecánicamente desde el artefacto (A9-A13)."""
    payload = inputs.drift.payload
    verdict = _require_str(payload, "verdict")
    artifact_gate = _require_str(payload, "phase0_gate")
    state = half_a_state(verdict, artifact_gate)
    percent = SIGNIFICANCE_LEVEL * 100

    rows = _segment_rows(payload, "clean_segments")
    by_name = {str(row["name"]): row for row in rows}
    if "intraday" not in by_name or "overnight" not in by_name:
        raise InputShapeError(
            "el artefacto del drift no trae los tramos `intraday` y `overnight` de la muestra "
            "limpia: sin ellos no se puede publicar el detalle que sostiene la mitad (a)."
        )
    intraday = by_name["intraday"]
    overnight = by_name["overnight"]
    difference: dict[str, Any] = {
        "mean_difference_bp": _require_float(payload, "clean_difference", "mean_difference_bp"),
        "t_stat": _require_float(payload, "clean_difference", "t_stat"),
        "p_value": _require_float(payload, "clean_difference", "p_value"),
    }
    difference["significant"] = Decimal(str(difference["p_value"])) < SIGNIFICANCE_LEVEL
    difference["statement"] = (
        "La diferencia emparejada `intraday - overnight` **no es significativa** al "
        f"{percent:.0f} %: el `fail` no se apoya en una diferencia demostrada entre los dos "
        "tramos, sino en la asimetría de significación por tramo y en la condición "
        "pre-registrada"
    )
    why = (
        f"La sesión tiene media {intraday['mean_bp']:+.2f} pb con p = {intraday['p_value']:.4f} "
        f"(no distinguible de cero al {percent:.0f} %), que es la lectura estadística de "
        f"«nulo»; el tramo nocturno tiene media {overnight['mean_bp']:+.2f} pb con "
        f"p = {overnight['p_value']:.4f} y **sí** es significativo. La «concentración» estricta "
        f"de `tasks.md` **no** está demostrada: la diferencia emparejada es "
        f"{difference['mean_difference_bp']:+.2f} pb con p = {difference['p_value']:.4f}. Con "
        "la redacción de `plan.md` §1.1.a el `fail` se cumple, porque el `open→close` es "
        "estadísticamente nulo mientras el nocturno es positivo."
    )
    return {
        "half": "a",
        "title": "El drift",
        "criterion": (
            "el drift debe estar en `open→close`; si se concentra en `close→open`, la "
            "estrategia intradía opera la peor parte del día"
        ),
        "state": str(state),
        "mapping": {
            "rule": "se leen `verdict` y `phase0_gate` del artefacto del drift",
            "pairs": [
                {"verdict": item_verdict, "phase0_gate": item_gate, "state": item_state}
                for item_verdict, item_gate, item_state in HALF_A_MAPPING
            ],
            "fallback": HALF_A_FALLBACK,
        },
        "artifact": {"verdict": verdict, "phase0_gate": artifact_gate},
        "clean_sample": {
            "clean_from": _require_str(payload, "clean_from"),
            "clean_sessions": _require_int(payload, "clean_sessions"),
            "first_session": _require_str(payload, "first_session"),
            "last_session": _require_str(payload, "last_session"),
            "sessions": _require_int(payload, "sessions"),
            "segments": rows,
        },
        "base_rate": {
            "sessions": _require_int(payload, "base_rate", "sessions"),
            "up_share": _require_float(payload, "base_rate", "up_share"),
            "abs_move_median_bp_full_sample": _require_float(
                payload, "base_rate", "abs_move_median_bp"
            ),
        },
        "open_quality": {
            "state": _require_str(payload, "open_quality"),
            "stale_open_share": _require_float(payload, "stale_open_share"),
        },
        "paired_difference": difference,
        "wording_discrepancy": {
            "declared": [dict(item) for item in GATE_A_WORDINGS],
            "applies": GATE_A_WORDING_APPLIED,
            "why": why,
            "consequence": (
                "Con la redacción estricta de `tasks.md` la concentración **no** está "
                "demostrada; con la de `plan.md` §1.1.a el `fail` **sí** se cumple. No se elige "
                "en silencio: se declara cuál aplica y por qué."
            ),
        },
        "limitations": [dict(item) for item in GATE_A_LIMITATIONS],
        "artifact_limitations": [str(item) for item in _require_list(payload, "limitations")],
    }


def _gate_b(
    inputs: Phase0Inputs, *, threshold_pct_of_r: Decimal, spread_usd: Decimal
) -> dict[str, Any]:
    """Mitad (b): el *slippage*. Nunca se presenta como aprobado condicional (A14-A17)."""
    payload = inputs.costs.payload
    evaluable = _require(payload, "phase0_gate_b", "evaluable")
    artifact_reason = _require_str(payload, "phase0_gate_b", "reason")
    slippage_state = _require_str(payload, "slippage_ejecucion", "state")
    measured_pct = _get(payload, "slippage_ejecucion", "value_pct")
    slippage_pct = (
        Decimal(str(measured_pct)) if isinstance(measured_pct, (str, int, float)) else None
    )
    if evaluable is not True:
        reason = (
            "El motivo está en el artefacto de costes: "
            f"{artifact_reason}. La mitad (b) es `not_evaluable`: no es un aprobado condicional "
            "ni un pendiente favorable."
        )
    else:
        reason = (
            "El *slippage* de ejecución **sí** está medido en el artefacto, pero `R` no está "
            "decidido (decisión abierta 5, #60): el umbral se aplica como `slippage_pct / "
            "R_pct` frente al umbral declarado y, sin `R`, no hay comparación posible; la "
            "mitad (b) sigue siendo `not_evaluable`, no un aprobado condicional."
        )
    return {
        "half": "b",
        "title": "El *slippage*",
        "criterion": (
            "el *slippage* sistemático no debe superar el umbral declarado sobre `R`; con un "
            "diferencial mínimo, el *slippage* es el coste dominante"
        ),
        # `R` no lo declara este módulo (A5): sin `R` la mitad (b) no puede ser `pass` ni
        # `fail`, solo `not_evaluable`. Por eso el estado es una constante y no una rama.
        "state": str(HalfResult.NOT_EVALUABLE),
        "artifact": {
            "evaluable": bool(evaluable),
            "criterion": _require_str(payload, "phase0_gate_b", "criterion"),
            "reason": artifact_reason,
            "verdict_owner": _require_str(payload, "phase0_gate_b", "verdict_owner"),
            "threshold_pct_of_r": _dec_str(threshold_pct_of_r),
        },
        "reason": reason,
        "not_presented_as": (
            "No se presenta como «aprobado condicional» ni como «pendiente favorable»: es "
            "`not_evaluable`, y **no evaluable no es un aprobado**."
        ),
        "slippage": {
            "state": slippage_state,
            "value_pct": None if slippage_pct is None else _dec_str(slippage_pct),
            "observations": _require_int(payload, "slippage_ejecucion", "observations"),
            "reason": _require_str(payload, "slippage_ejecucion", "reason"),
            "how_to_fill": _require_str(payload, "slippage_ejecucion", "how_to_fill"),
        },
        "dominance": {
            "statement": (
                f"{declared_value('slippage_dominance_declared')} (`plan.md` §4.4), sobre un "
                f"diferencial declarado de `{_dec_str(spread_usd)}` $ sobre el nocional de "
                "referencia: por eso el *slippage* es el término dominante del coste y el único "
                "no declarado."
            ),
            "provenance": "`plan.md` §4.4 y la tabla declarada del artefacto de #8",
        },
        "threshold_in_comparable_units": {
            "rule": "`slippage_pct / R_pct` frente al umbral declarado (no una resta)",
            "threshold_pct_of_r": _dec_str(threshold_pct_of_r),
            "applied": False,
            "why_not_applied": (
                "`R` es la decisión abierta 5 (#60): sin un valor declarado no se aplica el "
                "umbral a ninguna medida, ni siquiera si estuviera medida."
            ),
            "absolute_threshold_pct_per_scenario": [
                {
                    "r_pct": r_text,
                    "r_label": "escenario declarado, no decisión del propietario (-> #60)",
                    "threshold_pct": _pct_str(
                        threshold_pct_of_r * Decimal(r_text) / Decimal(100), Decimal("0.01")
                    ),
                }
                for r_text in _r_scenario_values()
            ],
        },
        "financing_cut": {
            "state": _require_str(payload, "financing_cut", "state"),
            "value_et": _get(payload, "financing_cut", "value_et"),
            "reason": _require_str(payload, "financing_cut", "reason"),
            "consequence": (
                "el corte de financiación sigue sin verificar, así que **no** se puede afirmar "
                "que la tenencia intradía sea 0: si el corte cae antes del cierre, el intradía "
                "puro paga tenencia igualmente"
            ),
            "issue": "#59",
        },
    }


def _spread_round_trip(payload: Any) -> tuple[Decimal, Decimal]:
    """La fila del diferencial (la única sin `direction`): es el coste del intradía puro."""
    for row in _require_list(payload, "declared_table", "rows"):
        if _get(row, "direction") is None:
            return _require_decimal(row, "amount", "pct"), _require_decimal(row, "amount", "usd")
    raise InputShapeError(
        "la tabla declarada del artefacto de costes no trae ninguna fila sin `direction`: no "
        "se puede identificar el diferencial de ida y vuelta."
    )


def _declared_costs(payload: Any) -> dict[str, Any]:
    """Reproduce la tabla declarada de #8 **tal cual**, sin recalcular ni redondear (A18).

    Las tres medidas de #8 se copian **por separado**: se publican al lado, nunca sumadas.
    """
    declared_table = _require_mapping(payload, "declared_table")
    fx = _require_mapping(payload, "fx_cost")
    spread_pct, spread_usd = _spread_round_trip(payload)
    short_pct = _require_decimal(payload, "declared_table", "round_trip", "short", "amount", "pct")
    short_usd = _require_decimal(payload, "declared_table", "round_trip", "short", "amount", "usd")
    long_pct = _require_decimal(payload, "declared_table", "round_trip", "long", "amount", "pct")
    long_usd = _require_decimal(payload, "declared_table", "round_trip", "long", "amount", "usd")
    if _dec_str(_require_decimal(fx, "value_pct")) == "0":
        # Un cero solo vale con motivo declarado: es la convención de #8 (A24 de aquella tarea).
        _require_str(fx, "reason")
    source = declared_table.get("source")
    return {
        "source": source if isinstance(source, str) else "artefacto de #8 (`declared_table`)",
        "declared_table": declared_table,
        "fx_cost": fx,
        "round_trip": {
            "holding_nights": _require_int(
                payload, "declared_table", "round_trip", "holding_nights"
            ),
            "short_pct": _dec_str(short_pct),
            "short_usd": _dec_str(short_usd),
            "short_label": "una noche en corto",
            "long_pct": _dec_str(long_pct),
            "long_usd": _dec_str(long_usd),
            "long_label": "una noche en largo",
            "spread_pct": _dec_str(spread_pct),
            "spread_usd": _dec_str(spread_usd),
        },
        "measures": {key: _require_mapping(payload, key) for key in MEASURE_KEYS},
        "measures_rule": (
            "Las tres medidas (`" + "`, `".join(MEASURE_KEYS) + "`) se publican **por separado y "
            "nunca sumadas**: sumarlas daría un «coste total» que nadie ha medido"
        ),
        "annualisation": {
            "consistent": _require(
                payload, "declared_table", "annualisation", "annualisation_consistent"
            ),
            "relative_difference": _dec_str(
                _require_decimal(payload, "declared_table", "annualisation", "relative_difference")
            ),
            "tolerance": _dec_str(
                _require_decimal(payload, "declared_table", "annualisation", "tolerance")
            ),
            "note": _require_str(payload, "declared_table", "annualisation", "note"),
        },
    }


def _p_star_block(
    *, spread_pct: Decimal, spread_usd: Decimal, short_pct: Decimal, long_pct: Decimal
) -> dict[str, Any]:
    """El bloque `p*` con la aritmética exacta y los escenarios declarados (A19)."""
    decimals = Decimal(declared_value("p_star_decimals"))
    cost_rows: tuple[tuple[str, Decimal], ...] = (
        ("intradía puro sin noche (el diferencial declarado)", spread_pct),
        ("corto con una noche", short_pct),
        ("largo con una noche mal cerrada", long_pct),
    )
    rows = [
        {
            "c_pct": _dec_str(c_pct),
            "c_label": label,
            "r_pct": r_text,
            "r_label": "escenario declarado, no decisión del propietario (-> #60)",
            "p_star_fraction": _dec_str(p_star(Decimal(r_text), c_pct)),
            "p_star_pct": _pct_str(p_star(Decimal(r_text), c_pct) * Decimal(100), decimals),
        }
        for label, c_pct in cost_rows
        for r_text in _r_scenario_values()
    ]
    bad_night_r = "1.0"
    bad_night = p_star(Decimal(bad_night_r), long_pct) * Decimal(100)
    return {
        "formula": P_STAR_FORMULA,
        "arithmetic": (
            "aritmética **exacta** con `decimal.Decimal`: los importes entran como cadenas del "
            "artefacto de costes y los escenarios de `R` como literales declarados de "
            "`plan.md` §4.4. No se recalcula ni se redondea la tabla declarada de #8."
        ),
        "cost_used_pct": _dec_str(spread_pct),
        "cost_used_label": (
            "coste de ida y vuelta **declarado** por #8 sin noche (el diferencial): es el que "
            "corresponde al intradía puro"
        ),
        "r_scenarios_source": (
            "`plan.md` §4.4 — **escenario declarado, no decisión del propietario**: el tamaño de "
            "`R` es la decisión abierta 5 (-> #60)"
        ),
        "rows": rows,
        "bad_night_scenario": {
            "r_pct": bad_night_r,
            "c_pct": _dec_str(long_pct),
            "label": "una noche mal cerrada (largo), con el `R` del escenario del 1 %",
            "p_star_pct": _pct_str(bad_night, decimals),
        },
        "viability": {
            "statement": (
                "**un `p*` bajo ya no es criterio de viabilidad** (`plan.md` §4.6), y este "
                "informe no lo usa como tal: con el coste declarado, casi cualquier sesgo mínimo "
                "es rentable, así que la barrera deja de ser económica y pasa a ser estadística."
            ),
            "reason": (
                f"el coste de ida y vuelta declarado para el intradía puro es "
                f"`{_dec_str(spread_pct)}` % (`{_dec_str(spread_usd)}` $ sobre el nocional de "
                "referencia); con una `c` tan pequeña, el término dominante del coste deja de "
                "ser el diferencial y pasa a ser el *slippage*."
            ),
            "derogated_gate": {
                "criterion": declared_value("derogated_p_star_gate"),
                "status": "derogada",
                "provenance": "`tasks.md`, nota de la Fase 0, y `plan.md` §16 (`plan.md` §4.6)",
                "why": (
                    "con los costes reales, cualquier modelo sesgado la superaría; está "
                    "prohibido usar un `p*` bajo como señal de viabilidad"
                ),
            },
            "forbidden_use": "usar un `p*` bajo como señal de viabilidad",
        },
    }


def _statistics(*, session_drift_bp: float) -> dict[str, Any]:
    """Consecuencia estadística, con la aritmética derivada de parámetros declarados (A21)."""
    edge = Decimal(declared_value("edge_to_detect"))
    null_probability = Decimal(declared_value("null_probability"))
    z_alpha = Decimal(declared_value("z_alpha_two_sided"))
    z_beta = Decimal(declared_value("z_beta"))
    sessions_per_year = Decimal(declared_value("sessions_per_year"))
    sample_operations = int(declared_value("sample_operations"))
    interval_decimals = Decimal(declared_value("interval_decimals"))

    operations_exact = required_operations(
        edge=edge, null_probability=null_probability, z_alpha=z_alpha, z_beta=z_beta
    )
    operations = int(operations_exact)
    years = Decimal(operations) / sessions_per_year
    years_approx = int(years.to_integral_value(rounding=ROUND_HALF_UP))
    low, high = confidence_interval(
        observed=edge, operations=sample_operations, z=z_alpha, decimals=interval_decimals
    )
    edge_pct = _dec_str(edge * Decimal(100))
    null_pct = _dec_str(null_probability * Decimal(100))
    return {
        "arithmetic": (
            "`n = (z_alpha/2 + z_beta)^2 · p(1-p) / (p - p0)^2` con los `z` declarados en "
            "`plan.md` §4.5 y `p` y `p0` declarados; el intervalo es el de Wald con la misma "
            "`z`, con los extremos truncados a la convención declarada."
        ),
        "parameters": {
            "edge": _dec_str(edge),
            "null_probability": _dec_str(null_probability),
            "z_alpha_two_sided": _dec_str(z_alpha),
            "z_beta": _dec_str(z_beta),
            "sessions_per_year": _dec_str(sessions_per_year),
            "sample_operations": str(sample_operations),
            "interval_decimals": _dec_str(interval_decimals),
        },
        "operations_exact": _dec_str(_trim(operations_exact)),
        "operations": operations,
        "years": _dec_str(years.quantize(Decimal("0.1"))),
        "years_approx": years_approx,
        "interval_95": {"low": _dec_str(low), "high": _dec_str(high)},
        "interval_contains_null": low < null_probability < high,
        "statement": (
            f"Para distinguir un edge del {edge_pct} % frente al {null_pct} % (95 % / 80 %) "
            f"hacen falta **{operations} operaciones** (≈{years_approx} años al ritmo declarado "
            f"de {_dec_str(sessions_per_year)} sesiones/año), y con {sample_operations} "
            f"operaciones el IC 95 % `[{_dec_str(low)}; {_dec_str(high)}]` contiene el "
            f"{null_pct} ⇒ un acierto observado del {edge_pct} % es **indistinguible del azar**."
        ),
        "conclusion": (
            "**El sistema no será validable por resultado en un plazo razonable**; si eso no es "
            "aceptable, hay que replantear el proyecto (`plan.md` §4.6)"
        ),
        "session_drift_mean_bp": session_drift_bp,
    }


def _volatility_anchor(inputs: Phase0Inputs, *, session_drift_bp: float) -> dict[str, Any]:
    """Conexión con #7 (A22): el anclaje de volatilidad y la relación señal/ruido."""
    payload = inputs.volatility.payload
    sigma_bp = _require_float(payload, "anchors", "median_forecast_sigma_bp")
    abs_move_bp = _require_float(payload, "anchors", "median_abs_open_close_bp")
    ratio = abs(sigma_bp / session_drift_bp) if session_drift_bp != 0 else None
    ratio_text = (
        "no definida, porque el drift de sesión medido es 0" if ratio is None else f"{ratio:.1f}x"
    )
    return {
        "selection": {
            "selected": _require_str(payload, "selection", "selected"),
            "verdict": _require_str(payload, "selection", "verdict"),
            "metric": _require_str(payload, "selection", "metric"),
        },
        "anchors": {
            "used_candidate": _require_str(payload, "anchors", "used_candidate"),
            "median_forecast_sigma_bp": sigma_bp,
            "median_abs_open_close_bp": abs_move_bp,
            "absolute_move_sessions": _require_int(payload, "anchors", "absolute_move_sessions"),
            "evaluated_sessions": _require_int(payload, "features", "evaluated_sessions"),
        },
        "session_drift_mean_bp": session_drift_bp,
        "signal_to_noise_ratio": None if ratio is None else round(ratio, 1),
        "statement": (
            f"El drift de sesión medido ({session_drift_bp:+.2f} pb) es **mucho menor** que la "
            f"sigma pronosticada ({sigma_bp:.2f} bp): la relación señal/ruido ({ratio_text}) es "
            "la que sostiene `plan.md` §4.6.3 — lo verificable está en la volatilidad y en el "
            "proceso, no en la dirección."
        ),
        "forbidden_use": (
            "presentar la selección del modelo de volatilidad como evidencia de viabilidad "
            "direccional"
        ),
    }


def _blockers(
    *, half_a: dict[str, Any], half_b: dict[str, Any], threshold_pct_of_r: Decimal
) -> list[dict[str, Any]]:
    """Motivos por los que la Fase 1 no está lista, buscables por máquina (A24)."""
    blockers: list[dict[str, Any]] = []
    verdict = half_a["artifact"]["verdict"]
    if half_a["state"] == HalfResult.FAIL and verdict == "overnight":
        blockers.append(
            {
                "code": str(BlockerCode.DRIFT_OVERNIGHT),
                "half": "a",
                "reason": (
                    "el artefacto del drift concluye `overnight` con `phase0_gate = fail`: el "
                    "retorno del índice se concentra fuera de la sesión"
                ),
                "issues": ["#6", "#52"],
            }
        )
    else:
        blockers.append(
            {
                "code": str(BlockerCode.DRIFT_NOT_EVALUABLE),
                "half": "a",
                "reason": (
                    f"el artefacto del drift trae `verdict = {verdict}` con `phase0_gate = "
                    f"{half_a['artifact']['phase0_gate']}`, que no está en la tabla de "
                    "correspondencia declarada"
                ),
                "issues": ["#6"],
            }
        )
    if half_b["slippage"]["state"] == "unmeasured":
        blockers.append(
            {
                "code": str(BlockerCode.SLIPPAGE_UNMEASURED),
                "half": "b",
                "reason": (
                    "el *slippage* de ejecución está sin medir: no existe ninguna ejecución real "
                    "y el artefacto no emite ningún valor de relleno"
                ),
                "issues": ["#62"],
            }
        )
    blockers.append(
        {
            "code": str(BlockerCode.R_UNDECIDED),
            "half": "b",
            "reason": (
                "`R` (amplitud del bracket) es la decisión abierta 5 y no está decidido: el "
                f"umbral del {_dec_str(threshold_pct_of_r)} % sobre `R` no se puede aplicar, ni "
                "siquiera con el *slippage* medido"
            ),
            "issues": ["#60"],
        }
    )
    if half_b["financing_cut"]["state"] != "verified":
        blockers.append(
            {
                "code": str(BlockerCode.FINANCING_CUT_UNVERIFIED),
                "half": "b",
                "reason": (
                    "el instante de corte de la financiación sigue sin verificar: no se puede "
                    "afirmar que la tenencia intradía sea 0"
                ),
                "issues": ["#59"],
            }
        )
    blockers.append(
        {
            "code": str(BlockerCode.BROKER_UNDECIDED),
            "half": "b",
            "reason": (
                "el bróker definitivo es la decisión abierta 4 (#59): la procedencia de la tabla "
                "de costes —y con ella el `p*`— no está cerrada"
            ),
            "issues": ["#59"],
        }
    )
    blockers.append(
        {
            "code": str(BlockerCode.ENTRY_PRICE_UNDECIDED),
            "half": "b",
            "reason": (
                "el precio de entrada exacto es la decisión abierta 6 (#61): sin él no se puede "
                "interpretar el *slippage* medido ni cerrar la mitad (b)"
            ),
            "issues": ["#61"],
        }
    )
    return blockers


def _recommendation(
    *,
    gate: GateVerdict,
    half_a: dict[str, Any],
    half_b: dict[str, Any],
    statistics: dict[str, Any],
) -> dict[str, Any]:
    """Recomendación razonada con la aritmética, sujeta a la regla de consistencia (A26)."""
    if gate is GateVerdict.PASS:
        value = Recommendation.CONTINUE
        reason = "las dos mitades son `pass`: la puerta de la Fase 0 está superada."
    else:
        value = Recommendation.REFRAME
        verdict = half_a["artifact"]["verdict"]
        intraday = next(
            row for row in half_a["clean_sample"]["segments"] if row["name"] == "intraday"
        )
        difference = half_a["paired_difference"]
        reason = (
            f"La mitad (a) es `{half_a['state']}`: el artefacto del drift concluye `{verdict}` "
            f"(sesión {intraday['mean_bp']:+.2f} pb, p = {intraday['p_value']:.4f}; diferencia "
            f"emparejada {difference['mean_difference_bp']:+.2f} pb, "
            f"p = {difference['p_value']:.4f}, no significativa). La mitad (b) es "
            f"`{half_b['state']}`. {half_b['reason']} El veredicto agregado es `{gate}` "
            f"({GATE_AGGREGATION_RULE}), de modo que `continue` queda **prohibido** "
            f"({RECOMMENDATION_CONSISTENCY_RULE}). **`reframe`** es la recomendación porque la "
            "estrategia intradía larga pasiva no se sostiene con esta evidencia, pero la "
            "evidencia es una asimetría de significación medida sobre el índice, no una "
            "diferencia demostrada ni una medición del CFD: eso es un reencuadre, no una "
            "parada. `stop` solo se justificaría si el propietario aceptase la puerta (a) como "
            "vinculante e irreversible. Y el sistema no es validable por resultado: "
            f"{statistics['operations']} operaciones hacen falta para un edge del "
            f"{statistics['parameters']['edge']} % frente al "
            f"{statistics['parameters']['null_probability']} %."
        )
    if not recommendation_is_consistent(gate, value):
        raise VerdictError(
            f"la recomendación `{value}` rompe la regla declarada: "
            f"{RECOMMENDATION_CONSISTENCY_RULE}"
        )
    return {
        "value": str(value),
        "rule": RECOMMENDATION_CONSISTENCY_RULE,
        "reason": reason,
        "options": [
            {
                "option": str(Recommendation.CONTINUE),
                "requires": (
                    "que las dos mitades sean `pass`: hoy la (a) es `fail` y la (b) es "
                    "`not_evaluable`, así que exigiría datos que no existen (drift del CFD y "
                    "*slippage* medidos, y `R` decidido)"
                ),
                "allowed": gate is GateVerdict.PASS,
            },
            {
                "option": str(Recommendation.REFRAME),
                "requires": (
                    "rehacer el planteamiento antes de construir el arnés: medir el drift sobre "
                    "el CFD, medir el *slippage* y cerrar `R`; y aceptar por escrito que la "
                    "validación por resultado no llegará en plazo razonable"
                ),
                "allowed": True,
            },
            {
                "option": str(Recommendation.STOP),
                "requires": (
                    "tratar la puerta (a) como vinculante e irreversible: el artefacto declara "
                    "`overnight` y la condición pre-registrada se cumple con la redacción de "
                    "`plan.md` §1.1.a"
                ),
                "allowed": True,
            },
        ],
    }


def _limitations(
    *, gate_a: dict[str, Any], gate_b: dict[str, Any], sample: dict[str, Any]
) -> list[dict[str, str]]:
    """Limitaciones declaradas, con la mitad a la que afectan y su enlace (A13, A17)."""
    items: list[dict[str, str]] = [
        {"statement": entry["limitation"], "half": "a", "issue": entry["issue"]}
        for entry in gate_a["limitations"]
    ]
    items.append(
        {
            "statement": (
                "el *slippage* de ejecución no está medido (`slippage_ejecucion.state = "
                f"{gate_b['slippage']['state']}`): no hay ninguna ejecución real y no se emite "
                "ningún valor de relleno"
            ),
            "half": "b",
            "issue": "#62",
        }
    )
    items.append(
        {
            "statement": (
                "`R` no está decidido (decisión abierta 5, #60): el umbral sobre `R` no se puede "
                "aplicar a ninguna medida, ni siquiera con el *slippage* medido"
            ),
            "half": "b",
            "issue": "#60",
        }
    )
    items.append(
        {
            "statement": gate_b["financing_cut"]["consequence"],
            "half": "b",
            "issue": gate_b["financing_cut"]["issue"],
        }
    )
    items.append(
        {
            "statement": (
                "las dos anclas de `|open→close|` **no son el mismo número**: la del drift es "
                "sobre la muestra completa (contaminada por el artefacto de #52) y la de "
                "volatilidad sobre su ventana evaluada limpia"
            ),
            "half": "a",
            "issue": "#52",
        }
    )
    items.append(
        {
            "statement": gate_a["wording_discrepancy"]["consequence"],
            "half": "a",
            "issue": "",
        }
    )
    items.append(
        {
            "statement": (
                "la muestra limpia compartida es "
                f"`clean_from = {sample['clean_from']}`; si los dos artefactos divergieran, el "
                "informe abortaría en vez de elegir"
            ),
            "half": "ambas",
            "issue": "#52",
        }
    )
    items.extend(dict(item) for item in CROSS_LIMITATIONS)
    return items


def _notes() -> list[str]:
    """Notas de método y de frontera con otras tareas."""
    return [
        "Este informe **consolida artefactos**; no reejecuta los estudios de #6, #7 ni #8, y no "
        "arregla sus fuentes: un defecto de un artefacto se comenta en su issue.",
        "El **modelo de coste del motor de backtest** es #11 "
        "(`src/cfdtrader/backtest/costs.py`): aquí solo se reproduce la tabla declarada de #8.",
        "El **bróker definitivo** es la decisión abierta 4 (#59): la tabla de costes es de un "
        "bróker concreto y su procedencia no está cerrada.",
        "Se consumen **solo los `.json`**: la prosa `.md` de los otros informes no se lee.",
        "`not_evaluable` **nunca** se convierte en `pass`: la agregación está declarada y hay una "
        "prueba por combinación.",
        "Todo instante se publica con zona horaria explícita (UTC) y ninguna hora fija de "
        "Madrid/CET.",
        "Determinismo: el informe depende solo de `--now` y de los artefactos, nunca de la hora "
        "de ejecución ni del orden del sistema de ficheros.",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Consolidación: función pura (entran los tres artefactos, sale el informe)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Phase0Report:
    """El informe consolidado. El JSON es el artefacto; el Markdown lo presenta."""

    as_of: datetime
    report_date: date
    payload: dict[str, Any]

    @property
    def report_stem(self) -> str:
        """Nombre base del informe: ``phase0_report_<AAAA-MM-DD>``."""
        return f"phase0_report_{self.report_date.isoformat()}"

    def json_text(self) -> str:
        """JSON determinista: mismas entradas y mismo `--now` ⇒ mismo texto byte a byte (A30)."""
        return json.dumps(self.payload, ensure_ascii=False, indent=2) + "\n"

    def write(self, directory: Path) -> tuple[Path, Path]:
        """Escribe el informe en JSON y Markdown. Solo escribe quien lo pida (el CLI)."""
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{self.report_stem}.json"
        markdown_path = directory / f"{self.report_stem}.md"
        json_path.write_text(self.json_text(), encoding="utf-8")
        markdown_path.write_text(render_markdown(self), encoding="utf-8")
        return json_path, markdown_path


def consolidate(inputs: Phase0Inputs, *, now: datetime) -> Phase0Report:
    """Consolida el informe a partir de los tres artefactos **ya cargados**. Función pura.

    No lee ni escribe disco: entra :class:`Phase0Inputs` y sale :class:`Phase0Report`, lo que
    permite probar la consolidación en memoria (A31) y reproducirla desde artefactos
    concretos, porque el informe declara ruta y `sha256` de cada entrada (A3).
    """
    costs_payload = inputs.costs.payload
    sample = _sample_consistency(inputs)
    declared_costs = _declared_costs(costs_payload)
    spread_pct = Decimal(declared_costs["round_trip"]["spread_pct"])
    spread_usd = Decimal(declared_costs["round_trip"]["spread_usd"])
    short_pct = Decimal(declared_costs["round_trip"]["short_pct"])
    long_pct = Decimal(declared_costs["round_trip"]["long_pct"])
    threshold_pct_of_r = _require_decimal(costs_payload, "phase0_gate_b", "threshold_pct_of_r")

    gate_a = _gate_a(inputs)
    gate_b = _gate_b(inputs, threshold_pct_of_r=threshold_pct_of_r, spread_usd=spread_usd)
    half_a = HalfResult(gate_a["state"])
    half_b = HalfResult(gate_b["state"])
    gate = aggregate_gate(str(half_a), str(half_b))

    intraday = next(row for row in gate_a["clean_sample"]["segments"] if row["name"] == "intraday")
    session_drift_bp = float(intraday["mean_bp"])
    statistics = _statistics(session_drift_bp=session_drift_bp)
    volatility_anchor = _volatility_anchor(inputs, session_drift_bp=session_drift_bp)
    blockers = _blockers(half_a=gate_a, half_b=gate_b, threshold_pct_of_r=threshold_pct_of_r)
    recommendation = _recommendation(gate=gate, half_a=gate_a, half_b=gate_b, statistics=statistics)
    phase1_ready = gate is GateVerdict.PASS and not blockers

    if gate is GateVerdict.FAIL:
        statement = "no se pasa a Fase 1"
    elif gate is GateVerdict.NOT_EVALUABLE:
        statement = (
            "no hay base para pasar a Fase 1: la puerta no es evaluable con estos artefactos"
        )
    else:
        statement = "se pasa a Fase 1"

    payload: dict[str, Any] = {
        "task": "#9",
        "artifact": "phase0_report",
        "title": "Informe de Fase 0 y decisión de continuidad",
        "as_of_utc": now.astimezone(UTC).isoformat(),
        "report_date": now.astimezone(UTC).date().isoformat(),
        "phases": {
            "phase": "Fase 0 — medir la realidad",
            "gate": "doble puerta de salida (`tasks.md`, tarea 9)",
            "phase1_ready": phase1_ready,
        },
        "selection_rule": {
            "rule": FILE_SELECTION_RULE,
            "classes": [asdict(klass) for klass in ARTIFACT_CLASSES],
            "explicit_paths_supported": (
                "`Phase0Inputs` y `consolidate` permiten entrar los tres artefactos ya cargados: "
                "esa es la vía de las rutas explícitas"
            ),
        },
        "inputs": {
            "drift": inputs.drift.provenance(),
            "volatility": inputs.volatility.provenance(),
            "costs": inputs.costs.provenance(),
        },
        "sample_consistency": sample,
        "declared_constants": _published_constants(
            threshold_pct_of_r=threshold_pct_of_r, spread_pct=spread_pct, spread_usd=spread_usd
        ),
        "open_decisions": [dict(decision) for decision in OPEN_DECISIONS],
        "gate_a": gate_a,
        "gate_b": gate_b,
        "declared_costs": declared_costs,
        "p_star": _p_star_block(
            spread_pct=spread_pct, spread_usd=spread_usd, short_pct=short_pct, long_pct=long_pct
        ),
        "statistics": statistics,
        "volatility_anchor": volatility_anchor,
        "verdict": {
            "half_a": str(half_a),
            "half_b": str(half_b),
            "gate": str(gate),
            "half_vocabulary": [member.value for member in HalfResult],
            "aggregation_rule": GATE_AGGREGATION_RULE,
            "phase1_ready": phase1_ready,
            "statement": statement,
            "blockers": blockers,
        },
        "recommendation": recommendation,
        "what_would_change_the_verdict": [
            dict(condition) for condition in WHAT_WOULD_CHANGE_THE_VERDICT
        ],
        "limitations": _limitations(gate_a=gate_a, gate_b=gate_b, sample=sample),
        "notes": _notes(),
    }
    if gate is not GateVerdict.PASS:
        payload["no_basis_for_continuation"] = {
            "statement": (
                "**No hay base para recomendar «continuar»**: la puerta agregada no es `pass`, "
                "y ni un `fail` ni un `not_evaluable` autorizan a seguir como si nada"
            ),
            "missing": [
                {
                    "what": "medir el drift sobre el CFD con el `open` real (mitad a)",
                    "issue": "#50 y #52",
                },
                {
                    "what": "medir el *slippage* de ejecución y cerrar `R` (mitad b)",
                    "issue": "#62 y #60",
                },
            ],
        }
    return Phase0Report(
        as_of=now.astimezone(UTC), report_date=now.astimezone(UTC).date(), payload=payload
    )


def _published_constants(
    *, threshold_pct_of_r: Decimal, spread_pct: Decimal, spread_usd: Decimal
) -> list[dict[str, str]]:
    """Constantes declaradas con nombre, valor y procedencia (A2)."""
    published = [dict(entry) for entry in DECLARED_CONSTANTS]
    published.append(
        {
            "name": "phase0_gate_b_threshold_pct_of_r",
            "value": _dec_str(threshold_pct_of_r),
            "unit": "% de R",
            "provenance": (
                "artefacto de #8 (`phase0_gate_b.threshold_pct_of_r`), que cita `tasks.md` "
                "tarea 9 (puerta b) y `plan.md` §16"
            ),
            "note": "umbral declarado, no medido: se aplica como `slippage_pct / R_pct`",
        }
    )
    published.append(
        {
            "name": "declared_spread_round_trip_pct",
            "value": _dec_str(spread_pct),
            "unit": "% del nocional",
            "provenance": "artefacto de #8 (`declared_table`, tabla declarada del bróker)",
            "note": "es la `c` del intradía puro en el bloque `p*`; no se recalcula",
        }
    )
    published.append(
        {
            "name": "declared_spread_round_trip_usd",
            "value": _dec_str(spread_usd),
            "unit": "$ sobre el nocional de referencia",
            "provenance": "artefacto de #8 (`declared_table`, tabla declarada del bróker)",
            "note": "importe declarado del diferencial, tal cual; no se recalcula",
        }
    )
    return published


# ─────────────────────────────────────────────────────────────────────────────
# Presentación
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_bp(value: float) -> str:
    """pb con signo y dos decimales."""
    return f"{value:+.2f}"


def _optional_pct(value: object) -> str:
    """Porcentaje o `null` explícito: nunca un cero de relleno."""
    return "`null`" if value is None else str(value)


def render_markdown(report: Phase0Report) -> str:
    """Informe legible. Los estados y las cifras clave son los mismos que los del JSON."""
    payload = report.payload
    verdict = payload["verdict"]
    recommendation = payload["recommendation"]
    gate_a = payload["gate_a"]
    gate_b = payload["gate_b"]
    p_star_block = payload["p_star"]
    statistics = payload["statistics"]
    anchor = payload["volatility_anchor"]
    costs = payload["declared_costs"]

    lines: list[str] = [
        "# Informe de Fase 0 y decisión de continuidad",
        "",
        f"- **Artefacto:** `{report.report_stem}.json` (esta prosa es su presentación)",
        f"- **Calculado:** `{payload['as_of_utc']}`",
        f"- **Puerta:** {payload['phases']['gate']}",
        f"- **Veredicto:** mitad (a) = `{verdict['half_a']}`, mitad (b) = `{verdict['half_b']}`, "
        f"agregado = `{verdict['gate']}`",
        f"- **`phase1_ready`:** `{str(verdict['phase1_ready']).lower()}` — "
        f"**{verdict['statement']}**",
        f"- **Recomendación:** `{recommendation['value']}`",
        "",
        "## Estado de la doble puerta",
        "",
        "| mitad | criterio | estado | por qué |",
        "|---|---|---|---|",
        f"| **(a) el drift** | {gate_a['criterion']} | `{gate_a['state']}` | artefacto: "
        f"`verdict = {gate_a['artifact']['verdict']}`, "
        f"`phase0_gate = {gate_a['artifact']['phase0_gate']}` |",
        f"| **(b) el *slippage*** | {gate_b['criterion']} | `{gate_b['state']}` | "
        f"{gate_b['reason']} |",
        "",
        f"**Regla de agregación declarada:** {verdict['aggregation_rule']}.",
        f"**Vocabulario:** {', '.join('`' + item + '`' for item in verdict['half_vocabulary'])}. "
        "Un `not_evaluable` **no** es un aprobado.",
        "",
        "### `blockers` (consumible sin leer prosa)",
        "",
        "| código | mitad | motivo | issues |",
        "|---|---|---|---|",
    ]
    for blocker in verdict["blockers"]:
        lines.append(
            f"| `{blocker['code']}` | {blocker['half']} | {blocker['reason']} | "
            f"{', '.join(blocker['issues'])} |"
        )

    lines.extend(
        [
            "",
            "## Puerta (a) — el drift",
            "",
            "Se resuelve leyendo `verdict` y `phase0_gate` del artefacto del drift y aplicando la "
            f"tabla declarada: {gate_a['mapping']['fallback']}.",
            "",
            f"Muestra limpia: `clean_from = {gate_a['clean_sample']['clean_from']}`, "
            f"{gate_a['clean_sample']['clean_sessions']} sesiones de "
            f"{gate_a['clean_sample']['sessions']} almacenadas. Calidad del `open`: "
            f"`{gate_a['open_quality']['state']}` "
            f"({gate_a['open_quality']['stale_open_share']:.1%} con el `open` repetido del cierre "
            "anterior).",
            "",
            "| tramo (muestra limpia) | sesiones | media (pb) | mediana (pb) | p | significativo |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in gate_a["clean_sample"]["segments"]:
        lines.append(
            f"| `{row['name']}` | {row['sessions']} | {_fmt_bp(row['mean_bp'])} | "
            f"{_fmt_bp(row['median_bp'])} | {row['p_value']:.4f} | "
            f"{'sí' if row['significant'] else 'no'} |"
        )
    difference = gate_a["paired_difference"]
    lines.extend(
        [
            "",
            f"Diferencia emparejada `intraday - overnight`: "
            f"{_fmt_bp(difference['mean_difference_bp'])} pb "
            f"(t = {difference['t_stat']:+.2f}, p = {difference['p_value']:.4f}). "
            f"{difference['statement']}.",
            "",
            "### Las dos anclas de `|open→close|` no se confunden",
            "",
            "| ancla | valor (pb) | ventana | sesiones |",
            "|---|---|---|---|",
            f"| mediana de `|open→close|` del drift | "
            f"{gate_a['base_rate']['abs_move_median_bp_full_sample']:.2f} | muestra **completa** "
            f"(contaminada por el artefacto de #52) | {gate_a['base_rate']['sessions']} |",
            f"| mediana de `|open→close|` de la volatilidad | "
            f"{anchor['anchors']['median_abs_open_close_bp']:.2f} | ventana **evaluada limpia** | "
            f"{anchor['anchors']['absolute_move_sessions']} |",
            "",
            "Son **dos números distintos con dos ventanas distintas**: publicar «la» mediana de "
            "`|open→close|` sin ventana está prohibido en este informe.",
            "",
            "### Discrepancia de redacción de la puerta (a)",
            "",
        ]
    )
    for wording in gate_a["wording_discrepancy"]["declared"]:
        lines.append(f"- {wording['source']}: «{wording['wording']}»")
    lines.extend(
        [
            "",
            f"**Aplica:** {gate_a['wording_discrepancy']['applies']}. "
            f"{gate_a['wording_discrepancy']['why']}",
            "",
            gate_a["wording_discrepancy"]["consequence"],
            "",
            "## Puerta (b) — el *slippage*",
            "",
            f"`phase0_gate_b.evaluable = {str(gate_b['artifact']['evaluable']).lower()}`. "
            f"La mitad (b) es `{gate_b['state']}`. {gate_b['reason']}",
            "",
            gate_b["not_presented_as"],
            "",
            "### Las tres medidas de #8, por separado y nunca sumadas",
            "",
            "| medida | estado | valor (%) | observaciones | motivo |",
            "|---|---|---|---|---|",
        ]
    )
    for key in MEASURE_KEYS:
        block = costs["measures"][key]
        lines.append(
            f"| `{key}` | `{block['state']}` | {_optional_pct(block['value_pct'])} | "
            f"{block['observations']} | {block['reason']} |"
        )
    lines.extend(
        [
            "",
            f"{costs['measures_rule']}.",
            "",
            gate_b["dominance"]["statement"],
            "",
            f"**Corte de financiación:** `{gate_b['financing_cut']['state']}` — "
            f"{gate_b['financing_cut']['consequence']} (issue {gate_b['financing_cut']['issue']}).",
            "",
            "### Umbral, en unidades comparables",
            "",
            f"{gate_b['threshold_in_comparable_units']['rule']}. Umbral declarado: "
            f"`{gate_b['threshold_in_comparable_units']['threshold_pct_of_r']}` % de `R`. "
            f"{gate_b['threshold_in_comparable_units']['why_not_applied']}",
            "",
            "| `R` (escenario declarado) | umbral absoluto (%) |",
            "|---|---|",
        ]
    )
    for row in gate_b["threshold_in_comparable_units"]["absolute_threshold_pct_per_scenario"]:
        lines.append(f"| {row['r_pct']} % | {row['threshold_pct']} % |")

    lines.extend(
        [
            "",
            "## Coste declarado y `p*`",
            "",
            "Se reproduce la tabla declarada de #8 **tal cual** (los importes, como cadenas "
            "decimales exactas, sin recalcular ni redondear). Diferencial de ida y vuelta: "
            f"`{costs['round_trip']['spread_pct']}` % "
            f"(`{costs['round_trip']['spread_usd']}` $ sobre el nocional de referencia); tenencia "
            f"por noche: `{costs['round_trip']['short_pct']}` % en corto y "
            f"`{costs['round_trip']['long_pct']}` % en largo; cambio de divisa: "
            f"`{costs['fx_cost']['value_pct']}` % ({costs['fx_cost']['reason']}); coherencia de "
            f"las cifras anualizadas declaradas: "
            f"`{str(costs['annualisation']['consistent']).lower()}`.",
            "",
            f"**{p_star_block['formula']}**, con {p_star_block['arithmetic']}",
            "",
            "| `c` (declarado) | `R` (escenario) | `p*` |",
            "|---|---|---|",
        ]
    )
    for row in p_star_block["rows"]:
        lines.append(
            f"| {row['c_label']} (`{row['c_pct']}` %) | {row['r_pct']} % | "
            f"**{row['p_star_pct']} %** |"
        )
    bad_night = p_star_block["bad_night_scenario"]
    lines.extend(
        [
            "",
            f"Escenario de «una noche mal cerrada» ({bad_night['label']}, "
            f"`c = {bad_night['c_pct']}` %, `R = {bad_night['r_pct']}` %): "
            f"**{bad_night['p_star_pct']} %**.",
            "",
            f"{p_star_block['viability']['statement']} {p_star_block['viability']['reason']}",
            "",
            f"Puerta antigua «{p_star_block['viability']['derogated_gate']['criterion']}»: "
            f"**{p_star_block['viability']['derogated_gate']['status']}** "
            f"({p_star_block['viability']['derogated_gate']['provenance']}): "
            f"{p_star_block['viability']['derogated_gate']['why']}.",
            "",
            "## Consecuencia estadística",
            "",
            statistics["statement"],
            "",
            f"{statistics['conclusion']}.",
            "",
            "## Anclaje de volatilidad (#7)",
            "",
            "El artefacto de volatilidad **sí selecciona**: "
            f"`selected = {anchor['selection']['selected']}` "
            f"(`verdict = {anchor['selection']['verdict']}`, "
            f"métrica `{anchor['selection']['metric']}`). Anclaje: sigma mediana del *forecast* "
            f"**{anchor['anchors']['median_forecast_sigma_bp']:.2f} bp** frente a "
            f"`|open→close|` mediano "
            f"**{anchor['anchors']['median_abs_open_close_bp']:.2f} bp**. {anchor['statement']}",
            "",
            f"Prohibido: {anchor['forbidden_use']}.",
            "",
            "## Recomendación",
            "",
            f"**`{recommendation['value']}`** — {recommendation['reason']}",
            "",
            "### Qué exigiría cada opción",
            "",
            "| opción | qué exigiría | permitida con este veredicto |",
            "|---|---|---|",
        ]
    )
    for option in recommendation["options"]:
        lines.append(
            f"| `{option['option']}` | {option['requires']} | "
            f"{'sí' if option['allowed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Qué cambiaría el veredicto",
            "",
            "| condición (comprobable) | mitad | dueño | issues |",
            "|---|---|---|---|",
        ]
    )
    for condition in payload["what_would_change_the_verdict"]:
        lines.append(
            f"| {condition['condition']} | {condition['half']} | {condition['owner']} | "
            f"{condition['issues']} |"
        )
    lines.extend(
        [
            "",
            "## Decisiones abiertas (declaradas, no inventadas)",
            "",
            "| # | decisión | estado | información que falta | qué depende de ella | issue |",
            "|---|---|---|---|---|---|",
        ]
    )
    for decision in payload["open_decisions"]:
        lines.append(
            f"| {decision['id']} | {decision['name']} | `{decision['state']}` | "
            f"{decision['missing_information']} | {decision['depends']} | {decision['issue']} |"
        )
    lines.extend(["", "## Limitaciones", ""])
    for item in payload["limitations"]:
        issue = f" (issue {item['issue']})" if item["issue"] else ""
        lines.append(f"- **mitad ({item['half']})**: {item['statement']}{issue}")
    lines.extend(["", "## Constantes declaradas (nombre, valor y procedencia)", ""])
    for constant in payload["declared_constants"]:
        lines.append(
            f"- `{constant['name']}` = `{constant['value']}` {constant['unit']} — "
            f"{constant['provenance']}. {constant['note']}"
        )
    lines.extend(["", "## Entradas (ruta y `sha256`)", ""])
    for kind, entry in payload["inputs"].items():
        lines.append(
            f"- `{kind}`: `{entry['path']}` — `as_of = {entry['as_of']}`, "
            f"`source = {entry['source']}`, `series_id = {entry['series_id']}`, "
            f"`sha256 = {entry['sha256']}` ({entry['sha256_of']})"
        )
    lines.extend(["", "## Notas", ""])
    lines.extend(f"- {note}" for note in payload["notes"])
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución
# ─────────────────────────────────────────────────────────────────────────────
def analyse(*, reports_dir: Path, now: datetime, write: bool = True) -> Phase0Report:
    """Carga los artefactos, consolida el informe y (por defecto) lo escribe."""
    inputs = load_inputs(reports_dir)
    report = consolidate(inputs, now=now)
    if write:
        json_path, markdown_path = report.write(reports_dir)
        logger.info("informe de Fase 0: {} y {}", json_path, markdown_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del informe de Fase 0.

    Códigos de salida: ``0`` = informe escrito (aunque el veredicto sea `fail` o
    `not_evaluable`, que es un resultado legítimo); ``2`` = falta un artefacto, hay
    ambigüedad, los artefactos se contradicen o falta un campo obligatorio ⇒ **no se escribe
    informe** y el motivo sale por ``stderr``.
    """
    parser = argparse.ArgumentParser(
        prog="cfdtrader.analysis.phase0_report",
        description="Informe determinista de Fase 0 y decisión de continuidad",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="raíz del almacén")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="directorio de informes (por defecto <data-root>/derived/reports)",
    )
    parser.add_argument("--settings", type=Path, default=None, help="ruta de settings.yaml")
    parser.add_argument("--now", default=None, help="instante de referencia ISO (tests)")
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.settings)
    except ConfigurationError as error:
        print(f"no se puede leer la configuración: {error}", file=sys.stderr)
        return 2

    data_root = Path(args.data_root) if args.data_root is not None else Path(settings.data.root)
    reports_dir = (
        Path(args.reports_dir)
        if args.reports_dir is not None
        else data_root / "derived" / "reports"
    )
    now = _parse_now(args.now)
    try:
        report = analyse(reports_dir=reports_dir, now=now)
    except Phase0ReportError as error:
        print(f"no se puede emitir el informe de Fase 0: {error}", file=sys.stderr)
        return 2

    verdict = report.payload["verdict"]
    logger.info(
        "Fase 0: mitad (a) {}, mitad (b) {}, agregado {}; phase1_ready {}; recomendación {}",
        verdict["half_a"],
        verdict["half_b"],
        verdict["gate"],
        verdict["phase1_ready"],
        report.payload["recommendation"]["value"],
    )
    return 0


def _parse_now(value: str | None) -> datetime:
    """Instante de referencia: el de verdad, o el que fije el test."""
    if value is None:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
