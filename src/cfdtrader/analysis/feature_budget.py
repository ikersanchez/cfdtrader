"""Presupuesto de features de #24 y re-medida de la Fase 2 con las candidatas (#146).

Cierra el reparto de #143: aquella tarea decidio **que** series de commodities/FX entran como
features y materializo la familia ``commodities_v1`` (cinco columnas candidatas); decidir si esas
candidatas se **quedan** en el conjunto del modelo y si el conjunto **mejora, empata o empeora**
la Fase 2 es esta tarea. El contrato de #24 son **10** columnas (``BASELINE_FEATURES``) y el
limite declarado **10-15** (`plan.md` §9): meter candidatas exige decidir si el conjunto se
**amplia** o se **intercambia**, y esa decision no se puede tomar sin medirla.

Que hace este modulo, en orden:

1. **Pre-registra la regla** (:data:`BUDGET_RULE`, :data:`INTERCHANGE_RULE`, :data:`VERDICT_RULE`)
   y el **espacio** de conjuntos (:func:`planned_sets`), **antes** de mirar ningun resultado.
2. **Re-mide la Fase 2** con el protocolo declarado: el plan de particiones de #12 (purga y
   embargo), el modelo de #24/#25 con su calibrador, el motor de #13 con el coste declarado de
   #11 y, sobre la matriz de retornos por sesion, el **DSR** y el **PBO** de #16
   (:mod:`cfdtrader.backtest.overfitting`).
3. Publica el veredicto **`mejora` / `empata` / `empeora` / `not_evaluable`** y la decision de
   presupuesto (**amplia** / **intercambia** / **mantiene_control**) con su identidad
   (``matrix_sha256``, ``feature_spec_sha256``, ``features_version``).

Vallas de honestidad (declaradas de antemano):

- Que el crudo «pese» economicamente **no** demuestra que su feature mejore el modelo fuera de
  muestra. Este trabajo puede terminar en **`empeora`** o en **`not_evaluable`** y eso es un
  resultado valido; lo que no es valido es declarar mejora sin medirla.
- Sigue siendo **carril B** (`plan.md` §11.6 / §19.7): **no** desbloquea produccion ni cambia
  ``phase2_ready``. La constante de produccion ``BASELINE_FEATURES`` **no** se toca aqui: el
  gancho es opt-in (`analysis.feature_frame.build_feature_frame(features=...)`) y la adopcion,
  si el veredicto la pide, es un cambio de constante con su propio registro.
- ``not_evaluable`` **nunca** se escribe como aprobado; un valor no medido **nunca** se escribe
  como ``0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final, Protocol, cast

import polars as pl
from loguru import logger

from cfdtrader.analysis import backtest_report
from cfdtrader.analysis.backtest_report import (
    _calendar_years,  # pyright: ignore[reportPrivateUsage]
)
from cfdtrader.analysis.baseline_report import (
    REGISTERED_HYPERPARAMETERS,
    _decider,  # pyright: ignore[reportPrivateUsage]
    _label_horizon,  # pyright: ignore[reportPrivateUsage]
    _require_alignment,  # pyright: ignore[reportPrivateUsage]
    _scored_inputs,  # pyright: ignore[reportPrivateUsage]
    split_assignments,
)
from cfdtrader.analysis.feature_frame import (
    FeatureFrame,
    FeatureFrameError,
    build_feature_frame,
)
from cfdtrader.backtest.costs import declared_cost_model, declared_slippage_assumption
from cfdtrader.backtest.engine import (
    STATUS_TRADED,
    BacktestRun,
    DecisionError,
    canonical_text,
    run_walk_forward,
)
from cfdtrader.backtest.metrics import brier_score, log_loss, sharpe_ratio
from cfdtrader.backtest.overfitting import (
    DEFAULT_BLOCKS,
    PBO_MAX,
    VERDICT_NOT_DETECTED,
    VERDICT_SIGNIFICANT,
    OverfittingError,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    select_variant,
    variant_sharpe_variance,
)
from cfdtrader.data.calendar import load_calendar
from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.store import Store
from cfdtrader.features import store as feature_store
from cfdtrader.models.baseline import (
    BASELINE_FEATURES,
    DesignFrame,
    InvalidDesignFrameError,
    UnknownFeatureError,
    calibrated_probabilities,
    design_frame,
    fit_baseline,
)

__all__ = [
    "BUDGET_MAX",
    "BUDGET_MIN",
    "BUDGET_RULE",
    "CANDIDATE_FEATURES",
    "CANDIDATE_SET",
    "CLI_NAME",
    "CONTROL_FEATURES",
    "CONTROL_SET",
    "DECISION_ADOPT",
    "DECISION_KEEP",
    "DECISION_SWAP",
    "EXPANDED_SET",
    "GATE_RULE",
    "INTERCHANGE_RULE",
    "REPORT_PREFIX",
    "SWAP_PREFIX",
    "TOLERANCE_STANDARD_ERRORS",
    "VERDICT_IMPROVES",
    "VERDICT_NOT_EVALUABLE",
    "VERDICT_RULE",
    "VERDICT_TIES",
    "VERDICT_WORSENS",
    "BudgetError",
    "BudgetReport",
    "BudgetSet",
    "FeatureSet",
    "SetMeasurement",
    "Verdict",
    "classify_verdict",
    "decide_budget",
    "measure",
    "planned_sets",
]

#: Nombre del CLI, para los mensajes de ``stderr``.
CLI_NAME: Final[str] = "cfdtrader.analysis.feature_budget"

#: Modulo que produce el informe (viaja en el payload).
ANALYSIS: Final[str] = "cfdtrader.analysis.feature_budget"

#: Prefijo del informe: ``feature_budget_<AAAA-MM-DD>.{json,md}``.
REPORT_PREFIX: Final[str] = "feature_budget"

#: Directorio de informes por defecto, relativo a la raiz del almacen.
DEFAULT_REPORTS_DIR: Final[str] = "derived/reports"

#: El contrato de #24: **10** columnas, y el limite declarado **10-15** (`plan.md` §9).
BUDGET_MIN: Final[int] = 10
BUDGET_MAX: Final[int] = 15

#: Nombre del conjunto de **control**: las 10 de ``BASELINE_FEATURES`` (#24).
CONTROL_SET: Final[str] = "control_v1"

#: Nombre del conjunto **ampliado**: control + todas las candidatas de #143.
EXPANDED_SET: Final[str] = "expanded_v1"

#: Prefijo del conjunto de **intercambio** (mismo tamano que el control).
SWAP_PREFIX: Final[str] = "swap"

#: Valla del veredicto: cuantos errores estandar pareados separan `mejora`/`empata`/`empeora`.
TOLERANCE_STANDARD_ERRORS: Final[float] = 1.0

#: Los cuatro veredictos de la re-medida. `not_evaluable` es un resultado **valido**.
VERDICT_IMPROVES: Final[str] = "mejora"
VERDICT_TIES: Final[str] = "empata"
VERDICT_WORSENS: Final[str] = "empeora"
VERDICT_NOT_EVALUABLE: Final[str] = "not_evaluable"

#: Las tres decisiones de presupuesto.
DECISION_ADOPT: Final[str] = "amplia"
DECISION_SWAP: Final[str] = "intercambia"
DECISION_KEEP: Final[str] = "mantiene_control"

#: Las 10 de control y las 5 candidatas de #143, leidas del catalogo (nunca recopiadas).
CONTROL_FEATURES: Final[tuple[str, ...]] = BASELINE_FEATURES
CANDIDATE_FEATURES: Final[tuple[str, ...]] = feature_store.COMMODITIES_FEATURE_COLUMNS
CANDIDATE_SET: Final[str] = feature_store.COMMODITIES_FEATURE_SET


class BudgetError(Exception):
    """Fallo de la re-medida del presupuesto de features."""


#: Regla del **presupuesto**: que se prueba y como se decide, declarada antes de ver resultados.
BUDGET_RULE: Final[str] = (
    "El conjunto del modelo es el de **control** (las 10 de `BASELINE_FEATURES`, #24). Se mide "
    "frente a los conjuntos candidatos del **espacio declarado** —el **ampliado** (control + las "
    "5 candidatas de `commodities_v1`, 15 columnas, dentro del limite 10-15 de `plan.md` §9) y los "
    "**intercambios** (mismo tamano 10)— con el protocolo de la Fase 2 (plan de #12 con purga y "
    "embargo, modelo de #24/#25 con su calibrador, motor de #13 con el coste declarado). La "
    "decision solo puede ser `amplia` (adopta el ampliado), `intercambia` (adopta un swap) o "
    "`mantiene_control`. Ningun conjunto fuera de este espacio se prueba en esta tarea."
)

#: Regla del **intercambio**: como se elige la columna de control que sale, declarada a priori.
INTERCHANGE_RULE: Final[str] = (
    "En un conjunto de **intercambio**, por cada candidata entra **una** columna del control y "
    "sale **la mas redundante con ella**: la de mayor `|r|` de Pearson sobre el frame de features "
    "(las filas completas), con los empates resueltos por el **orden declarado** del control "
    "(gana la de indice mas bajo). El resultado es un conjunto del **mismo tamano** que el control "
    "(10): no se amplia el presupuesto, se cambia una columna. La redundancia se mide **antes** de "
    "ver el modelo, sobre las features, no sobre el resultado."
)

#: Regla del **veredicto**: como se pasa de los numeros a `mejora`/`empata`/`empeora`.
VERDICT_RULE: Final[str] = (
    "El veredicto compara el **mejor candidato** con el **control** por la serie pareada de "
    "retornos netos por sesion (`d_t = r_candidato - r_control`) sobre las sesiones de test "
    "comunes: con `media(d)` y `se = desv_tipica(d) / sqrt(n)`, es `mejora` si "
    f"`media(d) > {TOLERANCE_STANDARD_ERRORS:g} * se`, `empeora` si "
    f"`media(d) < -{TOLERANCE_STANDARD_ERRORS:g} * se` y `empata` en otro caso. `not_evaluable` "
    "cuando el control o ningun candidato opera, o cuando el DSR/PBO no es estimable. El veredicto "
    "es **descriptivo**; la significacion la llevan el DSR y el PBO, que se publican al lado."
)

#: Regla de la **decision**: la mejora descriptiva **no** basta si el sobreajuste la desmiente.
GATE_RULE: Final[str] = (
    "La decision solo **adopta** un conjunto candidato si el veredicto es `mejora` **y** el DSR "
    "sale `significant` **y** el PBO no esta `detected` (`PBO <= PBO_MAX`). Una mejora descriptiva "
    "sin el respaldo de la maquinaria de sobreajuste **no** mueve produccion: la carga de la "
    "prueba la lleva quien afirma la mejora, no quien la niega."
)


class BudgetSet(Protocol):
    """Lo minimo que `decide_budget` necesita de un conjunto: su nombre y su tipo.

    Lo satisfacen estructuralmente :class:`FeatureSet` (el espacio pre-registrado) y
    :class:`SetMeasurement` (la medida), sin acoplar la decision a ninguno de los dos.
    """

    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> str: ...


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Un conjunto de features **nombrado y con procedencia**, pre-registrado (A1)."""

    name: str
    kind: str
    columns: tuple[str, ...]
    source: str
    provenance: str

    @property
    def n_features(self) -> int:
        """Cuantas columnas entran en el diseno de este conjunto."""
        return len(self.columns)

    def to_payload(self) -> dict[str, object]:
        """El conjunto como JSON puro: nombre, tipo, columnas, tamano y procedencia."""
        return {
            "name": self.name,
            "kind": self.kind,
            "columns": list(self.columns),
            "n_features": self.n_features,
            "source": self.source,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class SetMeasurement:
    """La medida de un conjunto: Sharpe OOS, Brier, log-loss e identidad (A2)."""

    name: str
    kind: str
    columns: tuple[str, ...]
    usable: bool
    n_traded: int
    n_test_sessions: int
    sharpe: float | None
    brier: float | None
    log_loss: float | None
    matrix_sha256: str
    features_version: str
    feature_spec_sha256: Mapping[str, str]
    feature_code_version: int
    error: str | None = None

    @property
    def n_features(self) -> int:
        """Cuantas columnas tiene el conjunto medido."""
        return len(self.columns)

    def to_payload(self) -> dict[str, object]:
        """La medida como JSON puro; `None` sigue siendo `None`, nunca `0`."""
        return {
            "name": self.name,
            "kind": self.kind,
            "columns": list(self.columns),
            "n_features": len(self.columns),
            "usable": self.usable,
            "n_traded": self.n_traded,
            "n_test_sessions": self.n_test_sessions,
            "sharpe": self.sharpe,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "matrix_sha256": self.matrix_sha256,
            "features_version": self.features_version,
            "feature_spec_sha256": dict(self.feature_spec_sha256),
            "feature_code_version": self.feature_code_version,
            "error": self.error,
        }


# ─────────────────────────────────────────────────────────────────────────────
# El espacio pre-registrado (A1, A3, A4)
# ─────────────────────────────────────────────────────────────────────────────
def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    """`r` de Pearson sobre las filas completas; una serie constante aporta `0.0`.

    Una columna constante no tiene correlacion **estimable**: se publica `0.0` (no elige el
    intercambio) en vez de un `nan` que romperia el orden declarado.
    """
    if len(left) != len(right):
        raise BudgetError(f"las dos series tienen que medir lo mismo: {len(left)} != {len(right)}")
    if len(left) < 2:
        return 0.0
    mean_left = math.fsum(left) / len(left)
    mean_right = math.fsum(right) / len(right)
    numerator = math.fsum(
        (a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True)
    )
    spread_left = math.sqrt(math.fsum((a - mean_left) ** 2 for a in left))
    spread_right = math.sqrt(math.fsum((b - mean_right) ** 2 for b in right))
    if spread_left == 0.0 or spread_right == 0.0:
        return 0.0
    return numerator / (spread_left * spread_right)


def _complete_pair(
    matrix: pl.DataFrame, left: str, right: str
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Las dos columnas como tuplas de `float`, con las filas incompletas descartadas."""
    if left not in matrix.columns or right not in matrix.columns:
        raise BudgetError(
            f"la matriz de features no trae `{left}` o `{right}`: el espacio del presupuesto solo "
            "puede usar columnas del catalogo de #73"
        )
    pair = matrix.select(pl.col(left), pl.col(right)).drop_nulls()
    return (
        tuple(float(value) for value in pair.get_column(left).cast(pl.Float64).to_list()),
        tuple(float(value) for value in pair.get_column(right).cast(pl.Float64).to_list()),
    )


def redundant_control_feature(
    matrix: pl.DataFrame,
    candidate: str,
    *,
    control: Sequence[str] = CONTROL_FEATURES,
) -> str:
    """La columna de control **mas redundante** con la candidata (regla del intercambio).

    Es la de mayor `|r|` de Pearson sobre el frame de features; los empates se resuelven por el
    **orden declarado** del control (gana la de indice mas bajo), nunca por el orden de iteracion
    de un contenedor sin orden.
    """
    best = control[0]
    best_score = -1.0
    for name in control:
        left, right = _complete_pair(matrix, name, candidate)
        score = abs(_pearson(left, right))
        if score > best_score:
            best, best_score = name, score
    return best


def planned_sets(
    matrix: pl.DataFrame,
    *,
    control: Sequence[str] = CONTROL_FEATURES,
    candidates: Sequence[str] = CANDIDATE_FEATURES,
) -> tuple[FeatureSet, ...]:
    """El **espacio pre-registrado**: control, ampliado y un intercambio por candidata (A1).

    El orden es el declarado y **determinista**: control primero, ampliado despues y, por ultimo,
    los intercambios en el orden de :data:`CANDIDATE_FEATURES`. Las columnas del ampliado son la
    concatenacion literal de control y candidatas. El intercambio de una candidata **no** puede
    reutilizar una columna que ya haya salido: sale la siguiente mas redundante disponible.
    """
    control_tuple = tuple(control)
    candidates_tuple = tuple(candidates)
    if not control_tuple or not candidates_tuple:
        raise BudgetError(
            "el espacio del presupuesto necesita control y candidatas no vacios: sin candidatas "
            "no hay nada que medir (A1)"
        )
    overlap = sorted(set(control_tuple) & set(candidates_tuple))
    if overlap:
        raise BudgetError(
            f"control y candidatas no pueden solaparse: {overlap} (la procedencia seria ambigua)"
        )
    expanded = FeatureSet(
        name=EXPANDED_SET,
        kind="expanded",
        columns=(*control_tuple, *candidates_tuple),
        source="BASELINE_FEATURES + commodities_v1 (control + candidatas)",
        provenance=(
            "control + las 5 candidatas de `commodities_v1` (#143); 15 columnas, el limite "
            "declarado de `plan.md` §9"
        ),
    )
    sets: list[FeatureSet] = [
        FeatureSet(
            name=CONTROL_SET,
            kind="control",
            columns=control_tuple,
            source="cfdtrader.models.baseline.BASELINE_FEATURES",
            provenance="las 10 de control de #24, sin tocar: es la referencia de la comparacion",
        ),
        expanded,
    ]
    used: set[str] = set()
    for candidate in candidates_tuple:
        removable = tuple(name for name in control_tuple if name not in used)
        if not removable:
            raise BudgetError(
                "se agotaron las columnas de control que se pueden intercambiar: el espacio pide "
                "mas intercambios que columnas tiene el control"
            )
        dropped = redundant_control_feature(matrix, candidate, control=removable)
        used.add(dropped)
        columns = tuple(candidate if name == dropped else name for name in control_tuple)
        sets.append(
            FeatureSet(
                name=f"{SWAP_PREFIX}_{candidate}",
                kind="swap",
                columns=columns,
                source=(f"BASELINE_FEATURES - `{dropped}` + `{candidate}` (regla del intercambio)"),
                provenance=(
                    f"sale `{dropped}` (la mas redundante con `{candidate}` por |r| de Pearson) y "
                    f"entra `{candidate}`; mismo tamano que el control (10)"
                ),
            )
        )
    sizes = sorted({item.n_features for item in sets})
    if min(sizes) < BUDGET_MIN or max(sizes) > BUDGET_MAX:
        raise BudgetError(
            f"algun conjunto del espacio cae fuera del presupuesto declarado [{BUDGET_MIN}, "
            f"{BUDGET_MAX}]: tamanos {sizes} (A1)"
        )
    return tuple(sets)


# ─────────────────────────────────────────────────────────────────────────────
# El veredicto y la decision (A5, A6, A7)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Verdict:
    """El veredicto pareado: la etiqueta, la media de la diferencia y su error estandar."""

    verdict: str
    mean_difference: float | None
    standard_error: float | None
    n_sessions: int

    def to_payload(self) -> dict[str, object]:
        """El veredicto como JSON puro, con la regla y la valla declaradas."""
        return {
            "verdict": self.verdict,
            "mean_return_difference": self.mean_difference,
            "standard_error": self.standard_error,
            "n_sessions": self.n_sessions,
            "tolerance_standard_errors": TOLERANCE_STANDARD_ERRORS,
            "rule": VERDICT_RULE,
        }


def _paired_moments(control: Sequence[float], candidate: Sequence[float]) -> tuple[float, float]:
    """`(media(d), se(d))` de la serie pareada `d_t = candidato_t - control_t`."""
    if len(control) != len(candidate):
        raise BudgetError(
            "las dos series tienen que tener las mismas sesiones: "
            f"{len(control)} != {len(candidate)}"
        )
    if len(control) < 2:
        raise BudgetError(
            "un veredicto necesita al menos dos sesiones pareadas: `se` no existe con una (A5)"
        )
    differences = [a - b for a, b in zip(candidate, control, strict=True)]
    mean = math.fsum(differences) / len(differences)
    standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    return mean, standard_error


def classify_verdict(
    control: Sequence[float],
    candidate: Sequence[float],
    *,
    tolerance: float = TOLERANCE_STANDARD_ERRORS,
) -> Verdict:
    """Clasifica el candidato frente al control con la regla pareada **pre-registrada** (A6).

    `mejora` si `media(d) > tolerance * se`, `empeora` si `media(d) < -tolerance * se` y `empata`
    en otro caso. Es **descriptivo**; la significacion la llevan el DSR y el PBO publicados al
    lado. La funcion es pura: dos series entran, una etiqueta sale.
    """
    mean, standard_error = _paired_moments(control, candidate)
    if mean > tolerance * standard_error:
        verdict = VERDICT_IMPROVES
    elif mean < -tolerance * standard_error:
        verdict = VERDICT_WORSENS
    else:
        verdict = VERDICT_TIES
    return Verdict(
        verdict=verdict,
        mean_difference=mean,
        standard_error=standard_error,
        n_sessions=len(control),
    )


def _significance(dsr: Mapping[str, object], pbo: Mapping[str, object]) -> tuple[bool, str]:
    """La **puerta de significacion**: DSR `significant` y PBO no `detected` (GATE_RULE).

    Un DSR o un PBO no evaluables **no** son un aprobado: sin maquinaria de sobreajuste no se
    adopta, y se publica el motivo.
    """
    if dsr.get("state") != "evaluated" or pbo.get("state") != "evaluated":
        return False, "DSR o PBO no evaluables: sin maquinaria de sobreajuste no se adopta"
    reasons: list[str] = []
    if dsr.get("verdict") != VERDICT_SIGNIFICANT:
        reasons.append(f"DSR = {dsr.get('dsr')} no es `{VERDICT_SIGNIFICANT}`")
    if pbo.get("verdict") != VERDICT_NOT_DETECTED:
        reasons.append(f"PBO = {pbo.get('pbo')} es `detected` (> {PBO_MAX})")
    if reasons:
        return False, "; ".join(reasons)
    return True, f"DSR `{VERDICT_SIGNIFICANT}` y PBO `{VERDICT_NOT_DETECTED}`"


def decide_budget(
    verdict: str, best: BudgetSet | None, *, significance_ok: bool
) -> tuple[str, str]:
    """La decision de presupuesto a partir del veredicto, del ganador y de la puerta (A7).

    Solo un veredicto `mejora` **con** la puerta de significacion superada mueve la aguja: adopta
    el ampliado (`amplia`) o un intercambio (`intercambia`). `empata`, `empeora`, `not_evaluable`
    y una mejora sin respaldo de DSR/PBO **mantienen** el control: la carga de la prueba la lleva
    quien afirma la mejora, no quien la niega.
    """
    if verdict != VERDICT_IMPROVES or not significance_ok or best is None:
        return DECISION_KEEP, CONTROL_SET
    if best.kind == "expanded":
        return DECISION_ADOPT, best.name
    if best.kind == "swap":
        return DECISION_SWAP, best.name
    return DECISION_KEEP, CONTROL_SET


# ─────────────────────────────────────────────────────────────────────────────
# La re-medida (A2, A8, A9, A10)
# ─────────────────────────────────────────────────────────────────────────────
def _as_utc(value: datetime) -> datetime:
    """El instante, en UTC y con zona: el modulo **no** lee el reloj."""
    if value.tzinfo is None:
        raise BudgetError("`as_of` tiene que traer zona horaria: el modulo no lee el reloj")
    return value.astimezone(UTC)


def _features_version(columns: Sequence[str], *, matrix_sha256: str, plan_sha256: str) -> str:
    """La identidad del **diseno** de un conjunto: features + matriz + plan, hasheados.

    Es la version que pide #146: la matriz completa (57 columnas) y su spec no cambian al elegir
    un subconjunto, asi que lo que se registra es la **lista elegida** junto a las dos identidades
    que no dependen de ella. Mismo conjunto y mismos datos ⇒ mismo digest, en cualquier proceso.
    """
    payload: dict[str, object] = {
        "features": list(columns),
        "matrix_sha256": matrix_sha256,
        "plan_sha256": plan_sha256,
    }
    return "sha256:" + hashlib.sha256(canonical_text(payload).encode("utf-8")).hexdigest()


def _returns_by_session(run: BacktestRun) -> dict[date, float]:
    """Retorno neto declarado por sesion de *test*: el P&L si se opero, `0.0` si no.

    Una sesion `no_trade` **no** es un dato ausente: es una posicion de cero, y asi entra en la
    matriz del PBO. Un retorno no medido (`pnl_declared is None` con estado `traded`) **no** se
    inventa: se cuenta como `0.0` solo si el estado no es `traded`.
    """
    returns: dict[date, float] = {}
    for fold in run.folds:
        for session in fold.sessions:
            value = (
                float(session.pnl_declared)
                if session.status == STATUS_TRADED and session.pnl_declared is not None
                else 0.0
            )
            returns[session.session] = value
    return returns


def _blocks_for(n_observations: int) -> int | None:
    """El mayor `S` par, `>= 2` y `<= DEFAULT_BLOCKS` que divide a `T`, o `None`.

    `probability_of_backtest_overfitting` exige que `S` divida a `T`; con `T = 500` y
    `DEFAULT_BLOCKS = 16` el mayor divisor valido es `10`. Sin ningun divisor valido el PBO **no**
    es estimable y se declara, nunca se rellena.
    """
    for blocks in range(min(DEFAULT_BLOCKS, n_observations), 1, -1):
        if blocks % 2 == 0 and n_observations % blocks == 0:
            return blocks
    return None


def _unusable(feature_set: FeatureSet, *, frame: FeatureFrame, error: str) -> SetMeasurement:
    """La medida de un conjunto que no se pudo ajustar: `usable = false` con su motivo (A2)."""
    return SetMeasurement(
        name=feature_set.name,
        kind=feature_set.kind,
        columns=feature_set.columns,
        usable=False,
        n_traded=0,
        n_test_sessions=0,
        sharpe=None,
        brier=None,
        log_loss=None,
        matrix_sha256=frame.matrix.matrix_sha256,
        features_version=_features_version(
            feature_set.columns,
            matrix_sha256=frame.matrix.matrix_sha256,
            plan_sha256="",
        ),
        feature_spec_sha256=frame.matrix.feature_spec_sha256,
        feature_code_version=frame.matrix.feature_code_version,
        error=error,
    )


def _best_candidate(candidates: Sequence[SetMeasurement]) -> SetMeasurement | None:
    """El candidato usable de mayor Sharpe OOS; empates al **orden declarado** (indice mas bajo)."""
    best: SetMeasurement | None = None
    for measurement in candidates:
        if not measurement.usable or measurement.sharpe is None:
            continue
        if best is None or measurement.sharpe > cast("float", best.sharpe):
            best = measurement
    return best


def _usable_matrix(
    usable: Sequence[SetMeasurement], returns_by_name: Mapping[str, Mapping[date, float]]
) -> tuple[tuple[date, ...], Sequence[Sequence[float]]] | None:
    """La matriz `T x N` de retornos de los conjuntos usables, o `None` si no es estimable."""
    if len(usable) < 2:
        return None
    sessions = tuple(sorted(returns_by_name[usable[0].name]))
    if not sessions:
        return None
    matrix = [
        [returns_by_name[measurement.name][session] for measurement in usable]
        for session in sessions
    ]
    return sessions, matrix


def _dsr_block(
    usable: Sequence[SetMeasurement], returns_by_name: Mapping[str, Mapping[date, float]]
) -> dict[str, object]:
    """El DSR de la variante de mayor Sharpe, con `n_trials` = conjuntos probados (A9)."""
    built = _usable_matrix(usable, returns_by_name)
    if built is None:
        return {
            "state": "not_evaluable",
            "calculation": "deflated_sharpe_ratio",
            "reason": "se necesitan al menos dos conjuntos usables para estimar V[SR]",
        }
    _, matrix = built
    try:
        variance = variant_sharpe_variance(
            [cast("float", measurement.sharpe) for measurement in usable]
        )
        index = select_variant(matrix)
        selected = [row[index] for row in matrix]
        result = deflated_sharpe_ratio(selected, n_trials=len(usable), sr_variance=variance)
    except OverfittingError as error:
        return {
            "state": "not_evaluable",
            "calculation": "deflated_sharpe_ratio",
            "reason": str(error),
        }
    return {
        "state": "evaluated",
        "calculation": "deflated_sharpe_ratio",
        "selected_set": usable[index].name,
        **result.to_payload(),
    }


def _pbo_block(
    usable: Sequence[SetMeasurement], returns_by_name: Mapping[str, Mapping[date, float]]
) -> dict[str, object]:
    """El PBO por CSCV sobre la matriz de conjuntos, con `PBO_MAX` declarado (A9)."""
    built = _usable_matrix(usable, returns_by_name)
    if built is None:
        return {
            "state": "not_evaluable",
            "calculation": "probability_of_backtest_overfitting",
            "reason": "se necesitan al menos dos conjuntos usables",
        }
    sessions, matrix = built
    blocks = _blocks_for(len(sessions))
    if blocks is None:
        return {
            "state": "not_evaluable",
            "calculation": "probability_of_backtest_overfitting",
            "reason": (
                f"ningun numero de bloques par (>= 2 y <= {DEFAULT_BLOCKS}) divide a las "
                f"{len(sessions)} sesiones de test"
            ),
        }
    try:
        result = probability_of_backtest_overfitting(matrix, blocks=blocks)
    except OverfittingError as error:
        return {
            "state": "not_evaluable",
            "calculation": "probability_of_backtest_overfitting",
            "reason": str(error),
        }
    return {
        "state": "evaluated",
        "calculation": "probability_of_backtest_overfitting",
        "pbo_max": PBO_MAX,
        **result.to_payload(),
    }


def _complete_sessions(design: DesignFrame) -> tuple[date, ...]:
    """Las sesiones donde **todas** las columnas del diseno estan presentes (casos completos).

    El modelo no acepta `NaN` y estas features **no** se imputan (`plan.md` §9): un nulo se
    publica. La re-medida usa por eso una **ventana comun de casos completos** para todos los
    conjuntos, de modo que las series pareadas comparten sesiones. El unico hueco medido son las
    dos sesiones del WTI negativo del 2020-04 (`oil_ret_*` declara `null`, #143).
    """
    complete = design.frame.filter(
        pl.all_horizontal([pl.col(name).is_not_null() for name in design.features])
    )
    return tuple(cast("list[date]", complete.get_column("session").to_list()))


def _restrict_design(design: DesignFrame, sessions: Sequence[date]) -> DesignFrame:
    """El diseno recortado a esas sesiones: se filtran **filas del diseno**, nunca la matriz.

    Filtrar la matriz moveria los vecinos y el corrimiento de `t-1` dejaria de ser `t-1`. Aqui se
    parte del diseno ya construido (con el `shift` correcto) y solo se quitan filas, de modo que la
    disponibilidad temporal sigue intacta.
    """
    kept = design.frame.filter(pl.col("session").is_in(list(sessions)))
    nulls = sum(kept.get_column(name).null_count() for name in design.features)
    return DesignFrame(
        frame=kept,
        n_sessions=kept.height,
        n_labels=design.n_labels,
        n_shifted_rows=design.n_shifted_rows,
        n_nulls_in_features=nulls,
        design_lag_sessions=design.design_lag_sessions,
        features=design.features,
    )


def measure(
    *,
    store: Store,
    as_of: datetime,
    sets: Sequence[FeatureSet] | None = None,
) -> BudgetReport:
    """Re-mide la Fase 2 para el espacio pre-registrado y devuelve el informe (A8, A10).

    El protocolo es el de la Fase 2, **importado** y no reimplementado: el plan de particiones de
    #12 (purga y embargo), el modelo de #24/#25 con su calibrador, el motor de #13 con el coste
    declarado de #11. La unica cosa que cambia entre conjuntos es **la lista de columnas** del
    diseno. La matriz completa de features (las 57 del catalogo) se construye **una sola vez**.
    """
    moment = _as_utc(as_of)
    history = backtest_report.load_history(store, series_id=backtest_report.SERIES_ID)
    calendar = load_calendar(years=_calendar_years(history.daily))
    universe = backtest_report.build_inputs(history, calendar=calendar)
    frame = build_feature_frame(store, series_id=backtest_report.SERIES_ID)
    _require_alignment(universe, frame)
    planned = tuple(sets) if sets is not None else planned_sets(frame.matrix.frame)
    # Ventana **comun de casos completos**: el modelo no acepta `NaN` y las candidatas traen los
    # nulos del WTI negativo del 2020-04. Todos los conjuntos se miden en las MISMAS sesiones.
    union_columns = tuple(dict.fromkeys((*CONTROL_FEATURES, *CANDIDATE_FEATURES)))
    union_design = design_frame(frame.matrix.frame, labels=frame.labels, selected=union_columns)
    keep = _complete_sessions(union_design)
    keep_set = set(keep)
    inputs = tuple(item for item in universe.inputs if item.session in keep_set)
    plan = backtest_report.build_split_plan(inputs, params=backtest_report.PHASE1_PLAN)
    assignments = split_assignments(plan)
    horizon = _label_horizon(plan)
    cost_model = declared_cost_model()
    slippage = declared_slippage_assumption()
    deciders = tuple(_decider(fold.index) for fold in plan.folds)

    measurements: list[SetMeasurement] = []
    returns_by_name: dict[str, dict[date, float]] = {}
    for feature_set in planned:
        try:
            full_design = design_frame(
                frame.matrix.frame, labels=frame.labels, selected=feature_set.columns
            )
            design = _restrict_design(full_design, keep)
            model = fit_baseline(
                design,
                splits=assignments,
                hyperparameters=REGISTERED_HYPERPARAMETERS,
                label_horizon=horizon,
            )
            decided = calibrated_probabilities(model, design.frame)
            run = run_walk_forward(
                _scored_inputs(inputs, decided),
                split_plan=plan,
                cost_model=cost_model,
                slippage=slippage,
                decide_by_fold=deciders,
                financing_cut=None,
            )
        except (UnknownFeatureError, InvalidDesignFrameError, DecisionError) as error:
            measurements.append(_unusable(feature_set, frame=frame, error=str(error)))
            returns_by_name[feature_set.name] = {}
            continue
        returns = _returns_by_session(run)
        returns_by_name[feature_set.name] = returns
        usable = run.traded > 0
        series = [returns[session] for session in sorted(returns)]
        outcomes = [int(value) for value in design.frame.get_column("y").to_list()]
        probabilities: list[float] = []
        observed: list[int] = []
        for value, outcome in zip(decided, outcomes, strict=True):
            if value is not None:
                probabilities.append(float(value))
                observed.append(outcome)
        measurements.append(
            SetMeasurement(
                name=feature_set.name,
                kind=feature_set.kind,
                columns=feature_set.columns,
                usable=usable,
                n_traded=run.traded,
                n_test_sessions=len(returns),
                sharpe=sharpe_ratio(series, annualization=1) if usable and series else None,
                brier=brier_score(probabilities, observed) if probabilities else None,
                log_loss=log_loss(probabilities, observed) if probabilities else None,
                matrix_sha256=frame.matrix.matrix_sha256,
                features_version=_features_version(
                    feature_set.columns,
                    matrix_sha256=frame.matrix.matrix_sha256,
                    plan_sha256=plan.plan_sha256,
                ),
                feature_spec_sha256=frame.matrix.feature_spec_sha256,
                feature_code_version=frame.matrix.feature_code_version,
                error=None,
            )
        )

    control = next((item for item in measurements if item.name == CONTROL_SET), None)
    candidates = [item for item in measurements if item.name != CONTROL_SET]
    usable_measurements = [item for item in measurements if item.usable]
    if control is None or not control.usable:
        verdict = Verdict(VERDICT_NOT_EVALUABLE, None, None, 0)
        best: SetMeasurement | None = None
    else:
        best = _best_candidate(candidates)
        if best is None:
            verdict = Verdict(VERDICT_NOT_EVALUABLE, None, None, 0)
        else:
            sessions = sorted(returns_by_name[CONTROL_SET])
            control_series = [returns_by_name[CONTROL_SET][session] for session in sessions]
            best_series = [returns_by_name[best.name][session] for session in sessions]
            verdict = classify_verdict(control_series, best_series)
    dsr = _dsr_block(usable_measurements, returns_by_name)
    pbo = _pbo_block(usable_measurements, returns_by_name)
    significance_ok, gate_reason = _significance(dsr, pbo)
    decision, adopted = decide_budget(verdict.verdict, best, significance_ok=significance_ok)
    chosen = next(item for item in measurements if item.name == adopted)
    payload = _payload(
        as_of=moment,
        frame=frame,
        window=_restrict_design(union_design, keep),
        plan_sha256=plan.plan_sha256,
        planned=planned,
        measurements=measurements,
        verdict=verdict,
        best=best,
        decision=decision,
        adopted=chosen,
        dsr=dsr,
        pbo=pbo,
        gate_ok=significance_ok,
        gate_reason=gate_reason,
    )
    return BudgetReport(
        as_of=moment,
        report_date=moment.date(),
        payload=payload,
        report_sha256=_digest(payload),
        sets=tuple(measurements),
        verdict=verdict,
        decision=decision,
        adopted=chosen,
    )


# ─────────────────────────────────────────────────────────────────────────────
# El informe (A11): payload canonico, digest y Markdown
# ─────────────────────────────────────────────────────────────────────────────
def _digest(payload: Mapping[str, object]) -> str:
    """``report_sha256``: sha256 del texto canonico del payload (sin el propio digest)."""
    return "sha256:" + hashlib.sha256(canonical_text(payload).encode("utf-8")).hexdigest()


def _payload(
    *,
    as_of: datetime,
    frame: FeatureFrame,
    window: DesignFrame,
    plan_sha256: str,
    planned: Sequence[FeatureSet],
    measurements: Sequence[SetMeasurement],
    verdict: Verdict,
    best: SetMeasurement | None,
    decision: str,
    adopted: SetMeasurement,
    dsr: Mapping[str, object],
    pbo: Mapping[str, object],
    gate_ok: bool,
    gate_reason: str,
) -> dict[str, object]:
    """El payload del informe: reglas, espacio, medidas, DSR/PBO, decision e identidad (A11)."""
    sessions = window.sessions
    return {
        "analysis": ANALYSIS,
        "kind": "feature_budget",
        "title": "Presupuesto de features de #24 y re-medida de la Fase 2 (#146)",
        "as_of": as_of.isoformat(),
        "budget": {
            "min": BUDGET_MIN,
            "max": BUDGET_MAX,
            "rule": BUDGET_RULE,
            "interchange_rule": INTERCHANGE_RULE,
            "verdict_rule": VERDICT_RULE,
            "gate_rule": GATE_RULE,
            "tolerance_standard_errors": TOLERANCE_STANDARD_ERRORS,
            "control": {
                "name": CONTROL_SET,
                "columns": list(CONTROL_FEATURES),
                "source": "cfdtrader.models.baseline.BASELINE_FEATURES",
            },
            "candidates": {
                "set": CANDIDATE_SET,
                "columns": list(CANDIDATE_FEATURES),
                "source": "cfdtrader.features.store.COMMODITIES_FEATURE_COLUMNS",
            },
        },
        "universe": {
            "series_id": frame.series_id,
            "first_session": sessions[0].isoformat(),
            "last_session": sessions[-1].isoformat(),
            "n_design_sessions": window.n_sessions,
            "n_positives": window.positives,
            "n_matrix_columns": frame.matrix.n_columns,
            "n_nulls_in_window": window.n_nulls_in_features,
            "window_rule": (
                "ventana **comun de casos completos** de la union control+candidatas: el modelo no "
                "acepta `NaN` y las features no se imputan (`plan.md` §9). Todos los conjuntos se "
                "miden en las mismas sesiones, de modo que las series pareadas son comparables"
            ),
        },
        "planned_sets": [item.to_payload() for item in planned],
        "sets": [item.to_payload() for item in measurements],
        "comparison": {
            "control": CONTROL_SET,
            "best_candidate": None if best is None else best.name,
            "best_kind": None if best is None else best.kind,
            **verdict.to_payload(),
        },
        "deflated_sharpe_ratio": dict(dsr),
        "probability_of_backtest_overfitting": dict(pbo),
        "decision": {
            "decision": decision,
            "adopted_set": adopted.name,
            "n_features": adopted.n_features,
            "features": list(adopted.columns),
            "production_changes": decision != DECISION_KEEP,
            "gate": {
                "passed": gate_ok,
                "reason": gate_reason,
                "rule": GATE_RULE,
            },
            "phase2_ready": False,
            "lane": "B",
            "note": (
                "carril B (`plan.md` §11.6 / §19.7): esta decision **no** desbloquea produccion "
                "ni cambia `phase2_ready`. La constante de produccion `BASELINE_FEATURES` solo "
                "cambia, si el veredicto la adopta, con su propio registro"
            ),
        },
        "identity": {
            "matrix_sha256": frame.matrix.matrix_sha256,
            "feature_spec_sha256": dict(frame.matrix.feature_spec_sha256),
            "feature_code_version": frame.matrix.feature_code_version,
            "plan_sha256": plan_sha256,
            "adopted_features_version": adopted.features_version,
        },
        "evidence": {
            "protocol": (
                "plan de particiones de #12 (purga y embargo), modelo de #24/#25 con calibrador, "
                "motor de #13 con el coste declarado de #11; matriz de features construida una vez"
            ),
            "dsr_n_trials_rule": "el numero de conjuntos **probados** en el espacio pre-registrado",
            "pbo_blocks": pbo.get("blocks"),
            "honesty": (
                "que el crudo «pese» economicamente no demuestra que su feature mejore el modelo "
                "fuera de muestra: `empeora` y `not_evaluable` son resultados validos"
            ),
        },
    }


@dataclass(frozen=True, slots=True)
class BudgetReport:
    """El informe de la re-medida: payload canonico, digest y los objetos que lo produjeron."""

    as_of: datetime
    report_date: date
    payload: dict[str, object]
    report_sha256: str
    sets: tuple[SetMeasurement, ...]
    verdict: Verdict
    decision: str
    adopted: SetMeasurement

    @property
    def report_stem(self) -> str:
        """Nombre base del informe: ``feature_budget_<AAAA-MM-DD>``."""
        return f"{REPORT_PREFIX}_{self.report_date.isoformat()}"

    def json_text(self) -> str:
        """JSON determinista: mismas entradas y mismo `as_of` ⇒ mismo texto byte a byte."""
        published: dict[str, object] = {**self.payload, "report_sha256": self.report_sha256}
        return json.dumps(published, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    def write(self, directory: Path) -> tuple[Path, Path]:
        """Escribe el informe en JSON y Markdown. Solo escribe quien lo pida (el CLI)."""
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{self.report_stem}.json"
        markdown_path = directory / f"{self.report_stem}.md"
        json_path.write_text(self.json_text(), encoding="utf-8")
        markdown_path.write_text(_render_markdown(self), encoding="utf-8")
        return json_path, markdown_path


def _number(value: object, *, digits: int = 6) -> str:
    """Un numero con signo y cifras fijas; ``None`` se publica como «no medido»."""
    if value is None:
        return "no medido"
    return f"{float(cast('float', value)):+.{digits}f}"


def _render_markdown(report: BudgetReport) -> str:
    """El informe legible: reglas, tabla de conjuntos, DSR/PBO y la decision con su identidad."""
    payload = report.payload
    decision = cast("Mapping[str, object]", payload["decision"])
    comparison = cast("Mapping[str, object]", payload["comparison"])
    identity = cast("Mapping[str, object]", payload["identity"])
    gate = cast("Mapping[str, object]", decision["gate"])
    dsr = cast("Mapping[str, object]", payload["deflated_sharpe_ratio"])
    pbo = cast("Mapping[str, object]", payload["probability_of_backtest_overfitting"])
    lines: list[str] = [
        "# Presupuesto de features de #24 y re-medida de la Fase 2 (#146)",
        "",
        f"> **Fecha:** {report.report_date.isoformat()} · **Veredicto:** "
        f"**{report.verdict.verdict}** · **Decision:** **{report.decision}** · "
        "carril B (`plan.md` §11.6 / §19.7): no cambia `phase2_ready`.",
        "",
        "## Tabla de conjuntos",
        "",
        "| Conjunto | Tipo | n | Operadas | Sharpe OOS | Brier | Log-loss |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report.sets:
        lines.append(
            f"| `{item.name}` | {item.kind} | {len(item.columns)} | {item.n_traded} | "
            f"{_number(item.sharpe)} | {_number(item.brier)} | {_number(item.log_loss)} |"
        )
    lines.extend(
        [
            "",
            "## Veredicto",
            "",
            f"- Mejor candidato: `{comparison['best_candidate']}`.",
            f"- Diferencia media por sesion: {_number(comparison['mean_return_difference'])}; "
            f"error estandar {_number(comparison['standard_error'])}.",
            f"- Regla: {comparison['rule']}",
            "",
            "## Sobreajuste",
            "",
            f"- DSR: `{dsr.get('verdict')}` (dsr = {dsr.get('dsr')}, "
            f"n_trials = {dsr.get('n_trials')}, selected set = `{dsr.get('selected_set')}`).",
            f"- PBO: `{pbo.get('verdict')}` (pbo = {pbo.get('pbo')}, blocks = {pbo.get('blocks')}, "
            f"pbo_max = {pbo.get('pbo_max')}).",
            "",
            "## Decision e identidad",
            "",
            f"- Decision: **{decision['decision']}** sobre `{decision['adopted_set']}` "
            f"({decision['n_features']} features).",
            f"- Puerta de significacion: {'superada' if gate['passed'] else 'NO superada'} "
            f"({gate['reason']}).",
            f"- `matrix_sha256 = {identity['matrix_sha256']}`",
            f"- `adopted_features_version = {identity['adopted_features_version']}`",
            f"- `report_sha256 = {report.report_sha256}`",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_as_of(value: str | None) -> datetime:
    """`--as-of` obligatorio: ISO-8601 **con** zona, el unico instante de la corrida."""
    if value is None:
        raise BudgetError("falta `--as-of`: el modulo no lee el reloj (ISO-8601 con zona)")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise BudgetError(f"`--as-of` no es ISO-8601 valido: {value!r}") from error
    return _as_utc(parsed)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada: ``0`` informe escrito, ``2`` falta un argumento o la re-medida fallo."""
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description=(
            "Re-mide la Fase 2 para el presupuesto de features de #24 con las candidatas de "
            "commodities/FX (#143) y publica el veredicto mejora/empata/empeora (#146)"
        ),
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
        help="instante declarado ISO-8601 con zona, obligatorio (el modulo no lee el reloj)",
    )
    args = parser.parse_args(argv)

    try:
        moment = _parse_as_of(cast("str | None", args.as_of))
        settings = load_settings(args.settings)
    except (BudgetError, ConfigurationError) as error:
        print(f"no se puede emitir el presupuesto de features: {error}", file=sys.stderr)
        return 2

    data_root = Path(args.data_root) if args.data_root is not None else Path(settings.data.root)
    reports_dir = (
        Path(args.reports_dir) if args.reports_dir is not None else data_root / DEFAULT_REPORTS_DIR
    )
    try:
        report = measure(store=Store(data_root), as_of=moment)
    except (BudgetError, FeatureFrameError) as error:
        print(f"no se puede emitir el presupuesto de features: {error}", file=sys.stderr)
        return 2

    json_path, _ = report.write(reports_dir)
    logger.info(
        "presupuesto de features: veredicto {} decision {} sobre {}; informe {}",
        report.verdict.verdict,
        report.decision,
        report.adopted.name,
        json_path,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    raise SystemExit(main())
