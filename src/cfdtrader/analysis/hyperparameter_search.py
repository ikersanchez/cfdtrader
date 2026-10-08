"""Barrido **pre-registrado** de la familia LightGBM: hiperparametros y subconjuntos (#82).

#26 ajusto **una** familia LightGBM con la constante ``LIGHTGBM_HYPERPARAMETERS`` y las **10**
features de ``BASELINE_FEATURES``, y dejo el ajuste de cualquiera de las dos cosas fuera de su
alcance (seguimiento **#82**). Mover un eje de la constante o cambiar el subconjunto de features
exige un presupuesto **declarado antes de ver resultados**, y ese presupuesto es ``plan.md``
§19.17 (T26b, decision del PM del 2026-10-08). Este modulo lo **implementa**: no elige el espacio
ni lo amplia.

Por que el espacio es un literal del modulo y no una bandera (A1, A11): el ``n_trials`` del DSR y
el PBO por CSCV se **derivan del registro** (`analysis.experiment_log`, #16). Buscar sin registrar
infla el Sharpe esperado y el informe mentiria sin decirlo, asi que :data:`SEARCH_SPACE` son las
**10** (`BUDGET`) variantes de §19.17 —6 de hiperparametros, cada una moviendo **un solo eje** de
la constante, y 4 subconjuntos de features sobre las **52** columnas del catalogo de #73— y cada
variante **intentada** entra en `runs/` con su propia identidad, incluso si no se puede medir
(`not_evaluable` con su motivo): el DSR deflacta por **intentos**, no por exitos.

El registro de #16 no se toca: ``ExperimentConfig`` ya lleva ``features``, ``hyperparameters``,
``seed``, ``series_id`` y ``window``, asi que las 10 entradas nuevas se registran con el, y
4 + 10 = **14** `n_trials` es el numero con el que se deflacta cada variante. Las 4 entradas de
#24/#25/#26 **no** se borran, no se reescriben y no se re-registran: se **reconstruyen** desde su
``model.json`` (la lineal con la aritmetica publicada y la de LightGBM recargando su
``booster_model``) y aportan su columna a la matriz del PBO.

Lo que se **reutiliza** y no se reimplementa: el universo y el plan (`analysis.backtest_report`,
#69), la matriz de las cinco familias (`analysis.feature_frame`, #24/#73), la regla de corrimiento
de diseno (`models.baseline.design_frame`, #24), el ajuste y la calibracion
(`models.lightgbm_model`, `models.calibration`, #26/#25), el motor (`backtest.engine`), el coste
declarado y las metricas (`backtest.costs`, `backtest.metrics`), el registro, el DSR y el PBO
(`analysis.experiment_log`, #16) y las piezas de informe de #26 (`analysis.model_comparison`).

Vallas de honestidad (A9): el defecto de unidades de #80 desplazaba el Sharpe de **todas** las
variantes en la misma constante por operacion —``0,0042 - 0,000042 = 0,004158``— y **no** el orden
de las medias ni las metricas de probabilidad; el bloque ``unit_bug_80`` lo publica medido. El
protocolo, la regla de seleccion y las 10 features de control de #26 **no** se tocan: si el
barrido cambiaria la familia ganadora, el informe lo declara (``family_changed``) y abre
seguimiento a **#27/#28**, en vez de sustituir el modelo en silencio. ``§11.6``, ``§19.6`` y
``§19.7`` **no** se tocan, `gate` sigue en `fail` y `phase2_ready` sigue `false`.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import polars as pl
from lightgbm import Booster
from loguru import logger

from cfdtrader.analysis import backtest_report, regeneration_delta
from cfdtrader.analysis.backtest_report import (
    PHASE1_PLAN,
    SERIES_ID,
    Universe,
    _calendar_years,  # pyright: ignore[reportPrivateUsage]
    build_inputs,
    build_split_plan,
    declared_slippage_assumption,
    load_history,
)
from cfdtrader.analysis.baseline_report import (
    CALIBRATION_BINS,
    MODEL_FILE,
    _label_horizon,  # pyright: ignore[reportPrivateUsage]
    _require_alignment,  # pyright: ignore[reportPrivateUsage]
    split_assignments,
)
from cfdtrader.analysis.experiment_log import (
    DEFAULT_RUNS_ROOT,
    DSR_HALVES,
    PBO_HALVES,
    ExperimentConfig,
    ExperimentLogError,
    ExperimentRecord,
    ExperimentResult,
    Registry,
    RegistryEntry,
    TrialsMismatchError,
    WriteOutcome,
    _write_immutable,  # pyright: ignore[reportPrivateUsage]
    aggregate_verdict,
    deflate_block,
    load_registry,
    pbo_block,
    record_experiment,
    require_consistent_aggregate,
    require_trials_match_registry,
)
from cfdtrader.analysis.feature_frame import (
    FEATURES_LIMITATIONS,
    FeatureFrame,
    FeatureFrameError,
    build_feature_frame,
)
from cfdtrader.analysis.model_comparison import (
    BASELINE_VARIANT_ID,
    CALIBRATION_RULE,
    NET_METRICS_REASON,
    SELECTION_RULE,
    TIE_TOLERANCE,
    Candidate,
    InconsistentObservationsError,
    ModelComparisonError,
    NotEvaluable,
    ReconstructionMismatchError,
    UnknownVariantError,
    Variant,
    _calibration_from_payload,  # pyright: ignore[reportPrivateUsage]
    _date_text,  # pyright: ignore[reportPrivateUsage]
    _declared_cost_block,  # pyright: ignore[reportPrivateUsage]
    _digest,  # pyright: ignore[reportPrivateUsage]
    _evaluated,  # pyright: ignore[reportPrivateUsage]
    _json_text,  # pyright: ignore[reportPrivateUsage]
    _lightgbm_fold_calibration,  # pyright: ignore[reportPrivateUsage]
    _matrix_block,  # pyright: ignore[reportPrivateUsage]
    _model_document,  # pyright: ignore[reportPrivateUsage]
    _number,  # pyright: ignore[reportPrivateUsage]
    _require_current_window,  # pyright: ignore[reportPrivateUsage]
    _require_reconstruction_matches_run,  # pyright: ignore[reportPrivateUsage]
    _run_variant,  # pyright: ignore[reportPrivateUsage]
    _series_matrix,  # pyright: ignore[reportPrivateUsage]
    _unit_bug_block,  # pyright: ignore[reportPrivateUsage]
    _variant_from_run,  # pyright: ignore[reportPrivateUsage]
    reconstruct_baseline,
    selection_block,
)
from cfdtrader.analysis.model_comparison import (
    MODEL_HASH_FORMAT as LIGHTGBM_MODEL_HASH_FORMAT,
)
from cfdtrader.analysis.model_comparison import (
    VARIANT_ID as LIGHTGBM_VARIANT_ID,
)
from cfdtrader.backtest.costs import CostModel, SlippageParameter, declared_cost_model
from cfdtrader.backtest.metrics import LOG_LOSS_EPSILON
from cfdtrader.backtest.splits import SplitPlan
from cfdtrader.data.calendar import load_calendar
from cfdtrader.data.settings import ConfigurationError, Settings, load_settings
from cfdtrader.data.store import Store
from cfdtrader.features import store as feature_store
from cfdtrader.models.baseline import (
    BASELINE_FEATURES,
    DECISION_THRESHOLD,
    SEED,
    DesignFrame,
    UnknownFeatureError,
    _matrix,  # pyright: ignore[reportPrivateUsage]
    design_frame,
)
from cfdtrader.models.calibration import sigmoid
from cfdtrader.models.lightgbm_model import (
    LIGHTGBM_HYPERPARAMETERS,
    RELOAD_TOLERANCE,
    LightGBMError,
    LightGBMModel,
    calibrated_probabilities,
    fit_lightgbm,
)

__all__ = [
    "AXES",
    "BASELINE_VARIANT_ID",
    "BUDGET",
    "CLI_NAME",
    "FEATURE_SETS",
    "PBO_BLOCKS",
    "PBO_MAX",
    "REGISTERED_TRIALS",
    "REPORT_PREFIX",
    "SEARCH_SPACE",
    "SPACE_SOURCE",
    "SWEEP_VERDICT_RULE",
    "TOTAL_TRIALS",
    "VARIANT_ID",
    "Axis",
    "FeatureSet",
    "HyperparameterSearchError",
    "InvalidAsOfError",
    "MissingAsOfError",
    "SearchReport",
    "SearchSpaceError",
    "SearchVariant",
    "SweepRow",
    "VariantNotEvaluableError",
    "analyse",
    "main",
    "reconstruct_lightgbm",
    "render_markdown",
    "space_audit",
]

#: Nombre del CLI, para los mensajes de ``stderr``.
CLI_NAME: Final[str] = "cfdtrader.analysis.hyperparameter_search"

#: Prefijo de identidad de la familia del barrido: cada variante anade ``#hN`` / ``#fN`` (A3).
VARIANT_ID: Final[str] = "lightgbm_search_v1"

#: Prefijo del informe: ``lightgbm_search_<AAAA-MM-DD>.{json,md}``.
REPORT_PREFIX: Final[str] = "lightgbm_search"

#: Presupuesto declarado de §19.17: **10** variantes nuevas (6 + 4), contando intentos (A1, A7).
BUDGET: Final[int] = 10

#: Las 4 entradas que #24/#25/#26 **ya** registraron: cuentan en `n_trials`, no se re-registran.
REGISTERED_TRIALS: Final[int] = 4

#: `n_trials` del registro con las 10 nuevas: el numero con el que se deflacta cada variante (A5).
TOTAL_TRIALS: Final[int] = REGISTERED_TRIALS + BUDGET

#: Bloques del PBO por CSCV: los mismos 10 de #26 (500 observaciones no es multiplo de 16).
PBO_BLOCKS: Final[int] = 10

#: Objetivo declarado del PBO (§19.17): por encima de esto el barrido no aprueba (A6).
PBO_MAX: Final[float] = 0.20

#: De donde sale el espacio: la decision del PM, no una bandera del CLI (A11).
SPACE_SOURCE: Final[str] = (
    "`plan.md` §19.17 (T26b, decision del PM del 2026-10-08): el **espacio** —6 variantes de "
    "hiperparametros moviendo **un** eje cada una y 4 subconjuntos de features sobre las 52 "
    "columnas del catalogo de #73— y el **presupuesto** (`BUDGET = 10` variantes nuevas ⇒ 14 "
    "`n_trials` con las 4 ya registradas) se declaran **antes** de ver resultados. Ampliarlo "
    "despues de mirar exige una decision nueva (§19.18), nunca una bandera"
)

#: Regla del veredicto del barrido (A6): nunca `pass`, con su motivo escrito.
SWEEP_VERDICT_RULE: Final[str] = (
    "`not_evaluable` si el DSR o el PBO no se pueden calcular (matriz incompleta, registro sin "
    f"`V[SR]`); `fail` en cualquier otro caso, **incluido** el de las dos mitades aprobadas: un "
    "barrido pre-registrado produce **hipotesis** y ni el DSR ni el PBO convierten una hipotesis "
    f"en una validacion. Objetivo declarado: `PBO <= {PBO_MAX}` con el DSR significativo"
)

#: Formato estable del `report_sha256` (A10). El prefijo viaja dentro del valor: un digest
#: desnudo lo bloquea `detect-secrets` (convencion de #93/#26).
REPORT_HASH_FORMAT: Final[str] = (
    "`sha256:<64 hex>` del sha256(UTF-8) de cfdtrader.backtest.engine.canonical_text(payload), "
    "**sin** la clave report_sha256 (un informe no se hashea a si mismo). El payload es JSON puro "
    "y no lleva ninguna ruta: de `--settings` solo viaja la raiz declarada en forma relativa y el "
    "registro se publica como `runs/<sha>`, asi que el hash no depende de `--reports-dir` ni de "
    "`--runs-root` (A10)"
)

#: Que **no** hace este modulo, legible por maquina. Cada frontera con su issue.
REPORT_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_amplia_el_espacio",
        "issue": "#82",
        "statement": (
            "no amplia el espacio ni el presupuesto despues de mirar: son los literales de "
            "§19.17, el CLI no acepta banderas que los muevan (ni `--n-trials` ni `--budget`) y "
            "ampliarlos exige una decision nueva (§19.18)"
        ),
    },
    {
        "id": "no_reabre_26",
        "issue": "#26",
        "statement": (
            "no re-elege la familia ni reabre #26: su protocolo (mismo plan de folds, el umbral "
            "0,5 y el coste declarado), su regla de seleccion y sus 10 features de control se "
            "**usan**, no se cambian; si el barrido cambiaria la familia ganadora, el informe lo "
            "declara y abre seguimiento a #27/#28"
        ),
    },
    {
        "id": "no_publica_metricas_netas",
        "issue": "#62",
        "statement": (
            "no publica metricas netas: el supuesto de #64 deja `pnl_net_pct` en `null` y "
            "`calculate_metrics` rechaza la corrida a proposito; medir el *slippage* es #62"
        ),
    },
    {
        "id": "no_decide_el_umbral_economico",
        "issue": "#27",
        "statement": (
            "el umbral 0,5 es el del informe, no el del sistema: el umbral economico, el "
            "*sizing* y la puerta son #27, #60 y #29"
        ),
    },
    {
        "id": "no_usa_cpcv",
        "issue": "#67",
        "statement": (
            "el PBO es el CSCV de #16 sobre la matriz de retornos de las variantes: **no** es el "
            "CPCV, que construye su propio esquema de particiones, purga y embargo (#67)"
        ),
    },
    {
        "id": "no_toca_el_holdout",
        "issue": "#68",
        "statement": (
            "no reserva ni mira el periodo final intocable: el plan es el de #69 y el *holdout* "
            "de §11.4 es #68"
        ),
    },
    {
        "id": "no_cambia_el_motor",
        "issue": "#80",
        "statement": (
            "no cambia el motor ni el coste: `backtest/engine.py` y `backtest/costs.py` quedan "
            "fuera del diff de esta entrega (el segundo tramo de #80 y el spread por tramo de "
            "#66 son de otras tareas)"
        ),
    },
    {
        "id": "no_poda_el_registro",
        "issue": "#44",
        "statement": (
            "no borra, reescribe ni poda entradas de `runs/`: las 4 de #24/#25/#26 quedan "
            "intactas y la retencion del registro es #44"
        ),
    },
    {
        "id": "no_sustituye_el_modelo",
        "issue": "#28",
        "statement": (
            "no corre el backtest de Fase 2 ni sustituye el modelo publicado: mide el barrido y "
            "publica sus numeros **deflactados**"
        ),
    },
)

#: Seguimientos declarados por el informe.
FOLLOW_UPS: Final[tuple[dict[str, str], ...]] = (
    {
        "issue": "#27",
        "topic": "umbral economico y sizing",
        "why": (
            "si el barrido cambiaria la familia ganadora, el siguiente paso es #27/#28, nunca "
            "sustituir el modelo en silencio"
        ),
    },
    {
        "issue": "#28",
        "topic": "backtest de Fase 2",
        "why": "este informe mide el barrido; el backtest completo con listones B y C es #28",
    },
    {
        "issue": "#62",
        "topic": "medir el *slippage* real",
        "why": "mientras no exista, `pnl_net_pct` es `null` y las metricas netas no existen",
    },
    {
        "issue": "#60",
        "topic": "decidir `R` y el umbral economico",
        "why": "el supuesto de #64 se declara como porcentaje de `R`, que sigue sin decidirse",
    },
    {
        "issue": "#78",
        "topic": "direccion corta",
        "why": "el barrido se evalua en la direccion **larga unica**; `ret_short` es #78",
    },
    {
        "issue": "#67",
        "topic": "CPCV",
        "why": "el esquema de validacion del modelo (particiones, purga y embargo) es #67",
    },
)

#: Seguimiento que abre el barrido **solo** si cambiaria la familia ganadora de #26 (A9).
FAMILY_CHANGED_FOLLOW_UPS: Final[tuple[str, ...]] = ("#27", "#28")

#: Vallas de honestidad que el informe publica con las secciones de `plan.md` intactas (A9).
HONESTY_WALLS: Final[tuple[str, ...]] = (
    "el defecto de unidades de #80 —el motor restaba un porcentaje a una fraccion— desplazaba el "
    "Sharpe de **todas** las variantes en la misma constante por operacion "
    "(`0,0042 - 0,000042 = 0,004158`): no cambia el orden de las medias entre variantes ni las "
    "metricas de probabilidad (Brier, log-loss, curva) ni el PBO, si el Sharpe; el bloque "
    "`unit_bug_80` lo publica **medido** en esta corrida",
    "el protocolo, la regla de seleccion y las 10 features de control de #26 **no** se tocan: si "
    "el barrido cambiaria la familia ganadora, el informe lo declara (`family_changed`) y abre "
    "seguimiento a #27/#28 en vez de sustituir el modelo en silencio",
    "`plan.md` §11.6, §19.6 y §19.7 **no** se tocan: `gate` sigue en `fail`, `phase1_ready` y "
    "`phase2_ready` siguen en `false`, y este informe **no** valida la estrategia",
)

#: Limites declarados del informe.
LIMITATIONS: Final[tuple[str, ...]] = (
    "el barrido es **declarado y acotado**: las 10 variantes de §19.17 y ninguna mas; ampliarlo "
    "exige una decision nueva (§19.18) y volver a correrlo entero",
    "la comparacion es en el espacio de **probabilidad** y de coste **declarado**: el *slippage* "
    "de #64 es un supuesto (`pnl_net_pct` nulo en el 100 % de las operaciones), asi que ninguna "
    "cifra economica es una validacion (#62, #60)",
    "las series de retorno salen del motor en la **fraccion** del nocional que declara #80: el "
    "Sharpe se mide sobre la serie corregida, con el coste declarado en su unidad",
    "la direccion es la **larga unica** (`y = 1{ret_long > 0}`): la pata corta es #78",
    "el PBO se calcula con 10 bloques y la sensibilidad al numero de bloques no se publica",
    "un intento que no se puede medir se registra con el resultado nulo **declarado** "
    "(`sharpe_per_session = 0.0`, `n_observations = 0`): el `0.0` no es una medicion, pero entra "
    "en el `V[SR]` del registro porque #16 no admite una entrada sin Sharpe",
    "el camino intradia es inerte (ningun decididor declara barreras): toda operacion sale por el "
    "cierre de la sesion (#27)",
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores tipados
# ─────────────────────────────────────────────────────────────────────────────
class HyperparameterSearchError(Exception):
    """Raiz de los errores del barrido."""


class SearchSpaceError(HyperparameterSearchError):
    """El espacio enumerado no es el pre-registrado en §19.17 (A1)."""


class MissingAsOfError(HyperparameterSearchError):
    """Escribir el informe exige un instante declarado: el modulo no lee el reloj (A11)."""


class InvalidAsOfError(HyperparameterSearchError):
    """El instante declarado no es un ISO-8601 valido."""


class VariantNotEvaluableError(HyperparameterSearchError):
    """La variante **se intento** y no se puede medir: se registra con su motivo (A7)."""


# ─────────────────────────────────────────────────────────────────────────────
# El espacio pre-registrado (§19.17): 6 ejes + 4 subconjuntos = BUDGET (A1)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Axis:
    """Un eje de ``LIGHTGBM_HYPERPARAMETERS`` y el valor declarado que prueba una variante.

    ``control`` tiene que ser **el valor de la constante** y ``value`` el de §19.17: la variante
    mueve **un solo** eje y deja los demas exactamente como estan (A1).
    """

    name: str
    control: object
    value: object
    provenance: str


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Un subconjunto **nombrado y con procedencia** del catalogo de features de #73 (A1).

    ``columns`` son las columnas del catalogo que entran en el modelo, en su orden. El conjunto
    ``control`` es el de #26 (las 10 de ``BASELINE_FEATURES``) y los otros tres son familias
    declaradas del catalogo: la lista sale del **modulo** que las define, no de una copia.
    """

    name: str
    columns: tuple[str, ...]
    source: str
    provenance: str


#: Los **6** ejes de §19.17, cada uno con su valor de control y su valor de busqueda.
AXES: Final[tuple[Axis, ...]] = (
    Axis(
        name="n_estimators",
        control=200,
        value=400,
        provenance="§19.17: `n_estimators` 200→400 (mas arboles, mismo aprendizaje)",
    ),
    Axis(
        name="learning_rate",
        control=0.05,
        value=0.02,
        provenance="§19.17: `learning_rate` 0,05→0,02 (aprendizaje mas lento)",
    ),
    Axis(
        name="num_leaves",
        control=4,
        value=8,
        provenance="§19.17: `num_leaves` 4→8 (mas capacidad por arbol)",
    ),
    Axis(
        name="max_depth",
        control=2,
        value=3,
        provenance="§19.17: `max_depth` 2→3 (un corte mas de profundidad)",
    ),
    Axis(
        name="min_child_samples",
        control=200,
        value=100,
        provenance="§19.17: `min_child_samples` 200→100 (hojas menos conservadoras)",
    ),
    Axis(
        name="reg_lambda",
        control=0.0,
        value=1.0,
        provenance="§19.17: `reg_lambda` 0,0→1,0 (regularizacion L2 declarada)",
    ),
)

#: El nombre del subconjunto de **control**: las 10 features de #26 (A1, A2).
CONTROL_FEATURE_SET: Final[str] = "control"

#: Los **4** subconjuntos de §19.17, sobre las 52 columnas del catalogo de #73.
FEATURE_SETS: Final[tuple[FeatureSet, ...]] = (
    FeatureSet(
        name=CONTROL_FEATURE_SET,
        columns=BASELINE_FEATURES,
        source="cfdtrader.models.baseline.BASELINE_FEATURES",
        provenance="§19.17: las 10 features **de control** de #26 (`BASELINE_FEATURES`), sin tocar",
    ),
    FeatureSet(
        name="volatility_vix",
        columns=feature_store.FEATURE_COLUMNS,
        source="cfdtrader.features.store.FEATURE_COLUMNS",
        provenance=(
            "§19.17: la familia de volatilidad y VIX (#7/#23) —`har_*`, `atr_norm`, `vix_*`— "
            "tal como la declara el catalogo de #19"
        ),
    ),
    FeatureSet(
        name="technical",
        columns=feature_store.TECHNICAL_FEATURE_COLUMNS,
        source="cfdtrader.features.store.TECHNICAL_FEATURE_COLUMNS",
        provenance="§19.17: la familia tecnica (#20) —retornos, distancia a la media, RSI, rango—",
    ),
    FeatureSet(
        name="macro_regime",
        columns=(*feature_store.MACRO_FEATURE_COLUMNS, *feature_store.REGIME_FEATURE_COLUMNS),
        source=(
            "cfdtrader.features.store.MACRO_FEATURE_COLUMNS + "
            "cfdtrader.features.store.REGIME_FEATURE_COLUMNS"
        ),
        provenance=(
            "§19.17: la familia macro (#22) —tipos, inflacion, dolar— junto a la de regimen "
            "(#23) —volatilidad pronosticada, eficiencia, calendario de vencimientos—"
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class SearchVariant:
    """Una variante del espacio pre-registrado: hiperparametros **o** subconjunto (A1, A3).

    ``name`` es el sufijo estable del ``variant_id`` (``h0``…``h5`` / ``f0``…``f3``) y ``index``
    el orden de la enumeracion: nombrar y ordenar son parte del pre-registro, no un detalle de
    implementacion, porque de ellos sale la identidad registrada de cada intento.
    """

    index: int
    name: str
    kind: str
    axis: str | None
    feature_set: str
    features: tuple[str, ...]
    hyperparameters: Mapping[str, object]
    provenance: str

    @property
    def variant_id(self) -> str:
        """La identidad registrada de la variante: ``lightgbm_search_v1#h0`` / ``…#f3`` (A3)."""
        return f"{VARIANT_ID}#{self.name}"

    def to_payload(self) -> dict[str, object]:
        """La variante como JSON puro, con lo que la distingue y su procedencia."""
        return {
            "index": self.index,
            "name": self.name,
            "variant_id": self.variant_id,
            "kind": self.kind,
            "axis": self.axis,
            "feature_set": self.feature_set,
            "features": list(self.features),
            "n_features": len(self.features),
            "hyperparameters": {**self.hyperparameters},
            "provenance": self.provenance,
        }


def _hyperparameter_variants() -> tuple[SearchVariant, ...]:
    """Las **6** variantes de hiperparametros: la constante con **un** eje movido (A1)."""
    out: list[SearchVariant] = []
    for index, axis in enumerate(AXES):
        parameters = {**LIGHTGBM_HYPERPARAMETERS, axis.name: axis.value}
        out.append(
            SearchVariant(
                index=index,
                name=f"h{index}",
                kind="hyperparameters",
                axis=axis.name,
                feature_set=CONTROL_FEATURE_SET,
                features=BASELINE_FEATURES,
                hyperparameters=parameters,
                provenance=axis.provenance,
            )
        )
    return tuple(out)


def _feature_variants() -> tuple[SearchVariant, ...]:
    """Las **4** variantes de subconjunto: la constante, con **otra** lista de features (A1)."""
    base = len(AXES)
    return tuple(
        SearchVariant(
            index=base + position,
            name=f"f{position}",
            kind="feature_set",
            axis=None,
            feature_set=conjunto.name,
            features=conjunto.columns,
            hyperparameters=dict(LIGHTGBM_HYPERPARAMETERS),
            provenance=conjunto.provenance,
        )
        for position, conjunto in enumerate(FEATURE_SETS)
    )


#: El espacio: la enumeracion **literal** de §19.17, en un orden estable (A1).
SEARCH_SPACE: Final[tuple[SearchVariant, ...]] = (*_hyperparameter_variants(), *_feature_variants())


def moved_axes(variant: SearchVariant) -> tuple[str, ...]:
    """Los ejes de ``LIGHTGBM_HYPERPARAMETERS`` que esa variante **mueve**, en orden estable."""
    names = sorted(set(LIGHTGBM_HYPERPARAMETERS) | set(variant.hyperparameters))
    return tuple(
        name
        for name in names
        if variant.hyperparameters.get(name) != LIGHTGBM_HYPERPARAMETERS.get(name)
    )


def require_known_features(columns: Sequence[str]) -> tuple[str, ...]:
    """Las columnas tienen que estar en el catalogo de #73, y sin repetir (A1, A2).

    Un nombre que no este en :data:`cfdtrader.features.store.ALL_FEATURE_COLUMNS` es
    :class:`cfdtrader.models.baseline.UnknownFeatureError`: el subconjunto elige del catalogo
    declarado, nunca de una lista escrita a mano.
    """
    values = tuple(columns)
    if not values:
        raise UnknownFeatureError(
            "un subconjunto de features vacio no es un subconjunto: el catalogo de #73 es el "
            "unico universo del que se elige (A2)"
        )
    unknown = sorted({name for name in values if name not in feature_store.ALL_FEATURE_COLUMNS})
    if unknown:
        raise UnknownFeatureError(
            f"las columnas {unknown} no estan en el catalogo de #73 "
            f"({len(feature_store.ALL_FEATURE_COLUMNS)} columnas): un subconjunto solo puede "
            "elegir del catalogo declarado (A2)"
        )
    if len(set(values)) != len(values):
        raise UnknownFeatureError(
            f"el subconjunto repite columnas ({list(values)}): contar dos veces una columna "
            "cambiaria el modelo sin declararlo (A2)"
        )
    return values


def space_audit() -> dict[str, object]:
    """El espacio pre-registrado, **comprobado** contra la constante y el catalogo (A1).

    Se ejecuta en cada corrida: si el numero de variantes no es ``BUDGET``, si el control de un
    eje ya no es el de ``LIGHTGBM_HYPERPARAMETERS``, si una variante de hiperparametros no mueve
    **exactamente un** eje o si un subconjunto sale del catalogo, es :class:`SearchSpaceError` y
    no se escribe ningun informe.
    """
    if len(SEARCH_SPACE) != BUDGET:
        raise SearchSpaceError(
            f"el espacio enumera {len(SEARCH_SPACE)} variantes y el presupuesto declarado de "
            f"§19.17 es {BUDGET}: el espacio es un pre-registro y no se ajusta a mano (A1)"
        )
    names = [variant.name for variant in SEARCH_SPACE]
    if len(set(names)) != len(names):
        raise SearchSpaceError(
            f"el espacio repite nombres de variante ({names}): cada intento tiene que ser una "
            "variante distinta del pre-registro, no dos veces la misma (A1)"
        )
    if [variant.index for variant in SEARCH_SPACE] != list(range(len(SEARCH_SPACE))):
        raise SearchSpaceError(
            "los indices del espacio no son 0..n-1 en orden: la enumeracion es parte del "
            "pre-registro (A1)"
        )
    for axis in AXES:
        current = LIGHTGBM_HYPERPARAMETERS.get(axis.name)
        if current != axis.control:
            raise SearchSpaceError(
                f"el eje `{axis.name}` declara control {axis.control!r} y la constante "
                f"`LIGHTGBM_HYPERPARAMETERS` trae {current!r}: el control no se mueve de sitio (A1)"
            )
    rows: list[dict[str, object]] = []
    for variant in SEARCH_SPACE:
        features = require_known_features(variant.features)
        moved = moved_axes(variant)
        if variant.kind == "hyperparameters":
            if len(moved) != 1 or variant.axis not in moved:
                raise SearchSpaceError(
                    f"la variante `{variant.name}` mueve {list(moved)} y tiene que mover **un "
                    f"solo** eje ({variant.axis!r}) de `LIGHTGBM_HYPERPARAMETERS` (A1)"
                )
            if features != BASELINE_FEATURES:
                raise SearchSpaceError(
                    f"la variante `{variant.name}` mueve hiperparametros y tiene que usar las "
                    "features de control de #26: mover las dos cosas a la vez mediria dos efectos "
                    "en un intento (A1)"
                )
        else:
            if moved or variant.axis is not None:
                raise SearchSpaceError(
                    f"la variante `{variant.name}` es de subconjunto y no puede mover ningun eje "
                    f"de `LIGHTGBM_HYPERPARAMETERS`: mueve {list(moved)} (A1)"
                )
            declared = next(item for item in FEATURE_SETS if item.name == variant.feature_set)
            if declared.columns != features:
                raise SearchSpaceError(
                    f"el subconjunto `{variant.feature_set}` de la variante `{variant.name}` no "
                    "coincide con el declarado en §19.17 (A1)"
                )
        rows.append(
            {
                "name": variant.name,
                "variant_id": variant.variant_id,
                "kind": variant.kind,
                "axis": variant.axis,
                "moved_axes": list(moved),
                "feature_set": variant.feature_set,
                "n_features": len(features),
                "features": list(features),
                "provenance": variant.provenance,
            }
        )
    return {
        "source": SPACE_SOURCE,
        "budget": BUDGET,
        "n_variants": len(SEARCH_SPACE),
        "registered_trials": REGISTERED_TRIALS,
        "total_trials": TOTAL_TRIALS,
        "n_hyperparameter_variants": len(AXES),
        "n_feature_variants": len(FEATURE_SETS),
        "catalog_columns": len(feature_store.ALL_FEATURE_COLUMNS),
        "axes": [
            {
                "name": axis.name,
                "control": axis.control,
                "value": axis.value,
                "provenance": axis.provenance,
            }
            for axis in AXES
        ],
        "feature_sets": [
            {
                "name": conjunto.name,
                "source": conjunto.source,
                "columns": list(conjunto.columns),
                "n_columns": len(conjunto.columns),
                "provenance": conjunto.provenance,
            }
            for conjunto in FEATURE_SETS
        ],
        "variants": rows,
        "axis_notes": {
            "num_leaves": (
                "con `max_depth = 2` (el control de #26) un arbol tiene como mucho 4 hojas, asi "
                "que subir `num_leaves` a 8 **no cambia** el modelo: `h2` mide lo mismo que la de "
                "control y las dos filas salen identicas. El espacio es el de "
                "§19.17 y se respeta **tal cual**; lo que se declara es que ese eje es inerte en "
                "esta familia, no que se haya ampliado"
            ),
        },
        "rule": (
            "el espacio y el presupuesto son literales pre-registrados (§19.17): una variante de "
            "hiperparametros mueve **un solo** eje de `LIGHTGBM_HYPERPARAMETERS` y todo "
            "subconjunto es subconjunto de las 52 columnas del catalogo de #73. Ampliarlo exige "
            "una decision nueva (§19.18), nunca una bandera (A1)"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Intentar una variante: diseno, ajuste y medida (A2, A3, A7)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SweepRow:
    """Una variante **intentada**: su identidad registrada, su diseno y su medida (A3, A7).

    ``measured`` es ``None`` cuando la variante no se pudo medir, y entonces ``reason`` y
    ``error`` dicen por que. La fila existe **siempre**: el intento se registra, y lo que no se
    hace es fabricar una columna de ceros para tapar el hueco.
    """

    variant: SearchVariant
    run_sha256: str
    state: str
    reason: str | None
    error: str | None
    n_sessions: int
    n_labels: int
    n_shifted_rows: int
    n_nulls_in_features: int
    model_sha256: str | None
    measured: Variant | None

    @property
    def evaluated(self) -> bool:
        """La variante tiene medida."""
        return self.measured is not None

    def to_payload(self) -> dict[str, object]:
        """La fila como JSON puro, con su estado **declarado** y sin cifras inventadas."""
        row: dict[str, object] = {
            **self.variant.to_payload(),
            "run_sha256": self.run_sha256,
            "state": self.state,
            "reason": self.reason,
            "error": self.error,
            "design": {
                "n_sessions": self.n_sessions,
                "n_labels": self.n_labels,
                "n_shifted_rows": self.n_shifted_rows,
                "n_nulls_in_features": self.n_nulls_in_features,
            },
            "model_sha256": self.model_sha256,
            "column": None if self.measured is None else self.run_sha256,
        }
        if self.measured is None:
            row.update(
                {
                    "n_traded": None,
                    "zeros": None,
                    "brier_score": None,
                    "log_loss": None,
                    "pnl_declared_pct_sum": None,
                    "sharpe_per_session": None,
                    "deciding_probabilities": None,
                    "outcomes": None,
                    "note": (
                        "la variante se **intento** y entra en el registro (`n_trials` deflacta "
                        "por intentos), pero **no** aporta columna a la matriz: no se rellena con "
                        "una columna de ceros (A7)"
                    ),
                }
            )
            return row
        measured = self.measured
        row.update(
            {
                "n_traded": measured.n_traded,
                "zeros": measured.zeros,
                "brier_score": measured.brier_score,
                "log_loss": measured.log_loss_value,
                "pnl_declared_pct_sum": measured.pnl_declared_sum,
                "sharpe_per_session": measured.sharpe_per_session,
                "deciding_probabilities": list(measured.deciding_probabilities),
                "outcomes": list(measured.outcomes),
                "series": list(measured.series),
                "source": measured.source,
                "calibrated": measured.calibrated,
                "state": "evaluated",
            }
        )
        return row


def design_for(
    variant: SearchVariant, *, features: pl.DataFrame, labels: pl.DataFrame
) -> DesignFrame:
    """La matriz de diseno de #24 para ese subconjunto, con la **misma** regla de lag (A2).

    Se llama al constructor de ``models.baseline`` con la lista de columnas de la variante: el
    corrimiento de una sesion, el conteo de nulos y el de sesiones son el **mismo** codigo que
    construye la matriz de #26, asi que el control sale identico por construccion.
    """
    return design_frame(features, labels=labels, selected=require_known_features(variant.features))


def fit_search_variant(
    variant: SearchVariant,
    *,
    features: pl.DataFrame,
    labels: pl.DataFrame,
    plan: SplitPlan,
    horizon: Sequence[int],
) -> tuple[DesignFrame, LightGBMModel]:
    """Ajusta la variante fold a fold con los hiperparametros **suyos** (A2, A7).

    Un subconjunto que deja nulos de diseno es :class:`VariantNotEvaluableError` (no se imputa) y
    un ajuste no determinista es ``LightGBMDeterminismError``; las dos cosas **se registran** como
    intento, con su motivo.
    """
    design = design_for(variant, features=features, labels=labels)
    if design.n_nulls_in_features:
        raise VariantNotEvaluableError(
            f"el subconjunto `{variant.feature_set}` deja {design.n_nulls_in_features} nulos de "
            "diseno: no se imputa —un nulo rellenado seria un dato inventado— y la variante se "
            "declara `not_evaluable` con su motivo (A2)"
        )
    model = fit_lightgbm(
        design,
        splits=split_assignments(plan),
        hyperparameters=variant.hyperparameters,
        seed=SEED,
        label_horizon=horizon,
    )
    return design, model


def measure_variant(
    variant: SearchVariant,
    model: LightGBMModel,
    design: DesignFrame,
    *,
    universe: Universe,
    plan: SplitPlan,
    frame: FeatureFrame,
    cost_model: CostModel,
    slippage: SlippageParameter,
) -> Variant:
    """Mide la variante con las probabilidades **calibradas** de #25 y las series del motor (A8).

    La probabilidad que decide es la calibrada —el protocolo que #25/#26 fijaron para decidir— y
    el umbral es el declarado (0,5). El ``run_sha256`` se rellena **despues** del registro: la
    identidad sale del contenido de la configuracion, no de esta medida.
    """
    decided = calibrated_probabilities(model, design.frame)
    measured = _run_variant(
        decided=decided,
        universe=universe,
        plan=plan,
        frame=frame,
        cost_model=cost_model,
        slippage=slippage,
    )
    return _variant_from_run(
        run_sha256="",
        variant_id=variant.variant_id,
        calibrated=True,
        source="lightgbm_search_fit",
        measured=measured,
        fold_calibration=_lightgbm_fold_calibration(model),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Las 4 entradas de #24/#25/#26: reconstruidas, nunca re-registradas (A4)
# ─────────────────────────────────────────────────────────────────────────────
def _reload_fold_calibration(
    folds: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """El calibrador **publicado** de cada fold, con el vocabulario de #25 (A4)."""
    out: list[dict[str, object]] = []
    for fold in folds:
        block = cast("Mapping[str, object]", fold["calibration"])
        out.append(
            {
                "index": fold["index"],
                "method": block["method"],
                "reason": block["reason"],
                "n_calibration": block["n_calibration"],
                "n_positives": block["n_positives"],
                "purge_sessions": block["purge_sessions"],
                "exclusions_are_no_op": block["exclusions_are_no_op"],
            }
        )
    return tuple(out)


def _lightgbm_deciding_probabilities(
    document: Mapping[str, object],
    *,
    matrix: np.ndarray[Any, np.dtype[np.float64]],
    calibrated: bool,
) -> tuple[tuple[float | None, ...], float]:
    """Las probabilidades de una variante LightGBM **recargada** de su `model.json`.

    Devuelve las probabilidades por sesion y la discrepancia **medida** maxima frente a lo que el
    propio `model.json` publica (`test_probabilities`): recargar el texto del booster tiene que
    reproducir lo publicado, y si no, es error tipado en vez de una cifra en silencio.
    """
    model = cast("Mapping[str, object]", document["model"])
    folds = cast("list[object]", model["folds"])
    out: list[float | None] = [None] * int(matrix.shape[0])
    worst = 0.0
    for item in folds:
        fold = cast("Mapping[str, object]", item)
        positions = [int(value) for value in cast("list[int]", fold["test_positions"])]
        booster = Booster(model_str=str(fold["booster_model"]))
        margins = [
            float(value)
            for value in np.asarray(
                cast("Any", booster.predict(matrix[positions, :], raw_score=True))
            )
        ]
        reloaded = [sigmoid(value) for value in margins]
        published = [float(value) for value in cast("list[float]", fold["test_probabilities"])]
        worst = max(
            worst,
            max(
                (abs(one - other) for one, other in zip(reloaded, published, strict=True)),
                default=0.0,
            ),
        )
        values: Sequence[float | None] = reloaded
        if calibrated:
            calibration = _calibration_from_payload(
                cast("Mapping[str, object]", fold["calibration"])
            )
            values = calibration.calibrate(margins)
        for position, value in zip(positions, values, strict=True):
            out[position] = None if value is None else float(value)
    return tuple(out), worst


def reconstruct_lightgbm(
    entry: RegistryEntry,
    *,
    runs_root: Path,
    universe: Universe,
    frame: FeatureFrame,
    plan: SplitPlan,
    cost_model: CostModel,
    slippage: SlippageParameter,
) -> Variant:
    """Reconstruye una variante LightGBM desde su `model.json`, **sin reajustar** (A4).

    Es la mitad que #26 no necesitaba —alli la familia se medía en el mismo proceso que la
    ajustaba— y aqui si: las entradas de #26 aportan su columna a la matriz del PBO. El
    ``booster_model`` publicado se recarga, sus probabilidades se comprueban contra las que el
    propio `model.json` publica (``RELOAD_TOLERANCE``) y el resultado se verifica contra los dos
    numeros de su `result.json` (``n_observations`` y ``sharpe_per_session``), nunca contra un
    literal que caduque con la siguiente ingesta (A4, #136).
    """
    if entry.variant_id != LIGHTGBM_VARIANT_ID:
        raise UnknownVariantError(
            f"el `variant_id` {entry.variant_id!r} no es la familia LightGBM de #26 ni la linea "
            f"base de #24 ({[BASELINE_VARIANT_ID, LIGHTGBM_VARIANT_ID]}): no se rellena con una "
            "columna de ceros ni se reajusta (A4)"
        )
    _require_current_window(entry, runs_root=runs_root, frame=frame, plan=plan)
    document = _model_document(runs_root, entry.run_sha256)
    model = cast("Mapping[str, object]", document["model"])
    if list(cast("list[str]", model["features"])) != list(BASELINE_FEATURES):
        raise UnknownVariantError(
            f"`runs/{entry.run_sha256}/model.json` declara otras features que las 10 de control: "
            "su columna no es comparable en la misma matriz de diseno (A4)"
        )
    folds = [cast("Mapping[str, object]", item) for item in cast("list[object]", model["folds"])]
    calibrated = "calibration" in folds[0]
    matrix = _matrix(frame.design.frame)
    decided, worst = _lightgbm_deciding_probabilities(
        document, matrix=matrix, calibrated=calibrated
    )
    if worst > RELOAD_TOLERANCE:
        raise ReconstructionMismatchError(
            f"el `booster_model` de `runs/{entry.run_sha256}/model.json` recargado no reproduce "
            f"las probabilidades que el propio fichero publica: discrepancia maxima {worst!r} > "
            f"{RELOAD_TOLERANCE!r} (A4)"
        )
    measured = _run_variant(
        decided=decided,
        universe=universe,
        plan=plan,
        frame=frame,
        cost_model=cost_model,
        slippage=slippage,
    )
    variant = _variant_from_run(
        run_sha256=entry.run_sha256,
        variant_id=entry.variant_id,
        calibrated=calibrated,
        source="booster_reload",
        measured=measured,
        fold_calibration=_reload_fold_calibration(folds) if calibrated else (),
    )
    if variant.n_traded != entry.n_observations:
        raise InconsistentObservationsError(
            f"`runs/{entry.run_sha256}/result.json` declara `n_observations = "
            f"{entry.n_observations}` y la recarga opera {variant.n_traded} sesiones: la entrada "
            "no se puede emparejar y su columna no se publica (A4)"
        )
    _require_reconstruction_matches_run(variant, entry)
    return variant


# ─────────────────────────────────────────────────────────────────────────────
# El registro: 14 intentos y la matriz completa (A3, A4, A5)
# ─────────────────────────────────────────────────────────────────────────────
def variant_config(
    variant: SearchVariant, *, frame: FeatureFrame, plan: SplitPlan
) -> ExperimentConfig:
    """La configuracion registrada de la variante: **sus** features y **sus** hiperparametros (A3).

    Es la identidad de #16 —``variant_id``, ``features``, ``hyperparameters``, ``seed``,
    ``series_id`` y la ``window`` de §11.4— sin inventar campos: cambiar una sola de esas cosas
    cambia su ``run_sha256`` y solo el suyo.
    """
    return ExperimentConfig(
        variant_id=variant.variant_id,
        features=variant.features,
        hyperparameters={**variant.hyperparameters},
        seed=SEED,
        series_id=SERIES_ID,
        window={
            "first_session": frame.first_session.isoformat(),
            "last_session": frame.last_session.isoformat(),
            "n_sessions": frame.n_design_rows,
            "n_positives": frame.n_positives,
            "design_lag_sessions": frame.design_lag_sessions,
            "plan_sha256": plan.plan_sha256,
            "matrix_sha256": frame.matrix.matrix_sha256,
            "feature_spec_sha256": dict(frame.matrix.feature_spec_sha256),
            "feature_code_version": frame.matrix.feature_code_version,
        },
    )


def _model_text(
    *,
    record: ExperimentRecord,
    payload: Mapping[str, object],
    digest: str,
) -> str:
    """El ``model.json`` de una variante medida: JSON puro, sin `pickle` (A3).

    Es el cuarto fichero del directorio del experimento. `load_registry` de #16 solo lee
    `config.json` y `result.json`, asi que no altera el registro, y es lo que hace que la entrada
    sea **reconstruible** sin reajustar.
    """
    return _json_text(
        {
            "run_sha256": record.run_sha256,
            "model_sha256": digest,
            "hash_format": LIGHTGBM_MODEL_HASH_FORMAT,
            "calibrated": True,
            "model": dict(payload),
            "note": (
                "el **texto** del booster por fold y las probabilidades y margenes de *test* "
                "publicados: `load_registry` de #16 solo lee `config.json` y `result.json`, y "
                "este cuarto fichero es lo que permite **reconstruir** la variante sin reajustar "
                "(`analysis.hyperparameter_search.reconstruct_lightgbm`)"
            ),
        }
    )


def _candidates_from_registry(
    *,
    registry: Registry,
    known: Mapping[str, Variant],
    runs_root: Path,
    universe: Universe,
    frame: FeatureFrame,
    plan: SplitPlan,
    cost_model: CostModel,
    slippage: SlippageParameter,
) -> tuple[Candidate, ...]:
    """Una fila por entrada del registro, en su orden, con su estado **declarado** (A4, A7).

    Las 10 del barrido vienen medidas (`known`); las 4 de #24/#25/#26 se **reconstruyen**. Ninguna
    variante se rellena: si una entrada no se puede reconstruir, se publica ``not_evaluable`` con
    el motivo y su ``run_sha256``, y no aporta columna a la matriz del PBO.
    """
    out: list[Candidate] = []
    for entry in registry.entries:
        if entry.run_sha256 in known:
            variant = known[entry.run_sha256]
            if variant.n_traded != entry.n_observations:
                raise InconsistentObservationsError(
                    f"la variante `{variant.variant_id}` opera {variant.n_traded} sesiones y su "
                    f"`result.json` declara {entry.n_observations}"
                )
            out.append(variant)
            continue
        try:
            if entry.variant_id == BASELINE_VARIANT_ID:
                out.append(
                    reconstruct_baseline(
                        entry,
                        runs_root=runs_root,
                        universe=universe,
                        frame=frame,
                        plan=plan,
                        cost_model=cost_model,
                        slippage=slippage,
                    )
                )
            else:
                out.append(
                    reconstruct_lightgbm(
                        entry,
                        runs_root=runs_root,
                        universe=universe,
                        frame=frame,
                        plan=plan,
                        cost_model=cost_model,
                        slippage=slippage,
                    )
                )
        except ModelComparisonError as error:
            out.append(
                NotEvaluable(
                    run_sha256=entry.run_sha256,
                    variant_id=entry.variant_id,
                    reason=str(error),
                    error=type(error).__name__,
                )
            )
    return tuple(out)


def sweep_verdict(*, gate: str, pbo: Mapping[str, object]) -> str:
    """El veredicto del barrido: ``fail`` o ``not_evaluable``, **nunca** ``pass`` (A6).

    Con el PBO por encima de :data:`PBO_MAX` no hay aprobado, y con un calculo no evaluable se
    declara ``not_evaluable`` con su motivo. Cuando las dos mitades del agregado de #9 aprueban el
    veredicto **sigue** siendo ``fail``: un barrido pre-registrado produce hipotesis, no una
    validacion (``§11.6``/``§19.6``/``§19.7`` intactos, `phase2_ready = false`).
    """
    if gate == "not_evaluable" or pbo.get("state") != "evaluated":
        return "not_evaluable"
    return "fail"


def _beats_reference(best: Variant, reference: Variant | None) -> bool:
    """Si el mejor del barrido gana a la familia de #26 con la regla **declarada** (A9).

    La comparacion es la de #26 —Brier, luego log-loss— y el empate **no** cambia la familia: solo
    una mejora estricta la cambia, y entonces el informe abre seguimiento a #27/#28.
    """
    if reference is None:
        return True
    if abs(best.brier_score - reference.brier_score) > TIE_TOLERANCE:
        return best.brier_score < reference.brier_score
    if abs(best.log_loss_value - reference.log_loss_value) > TIE_TOLERANCE:
        return best.log_loss_value < reference.log_loss_value
    return False


# ─────────────────────────────────────────────────────────────────────────────
# El informe
# ─────────────────────────────────────────────────────────────────────────────
def _selected_of(block: Mapping[str, object]) -> str | None:
    """El `run_sha256` seleccionado por una regla de #26, o ``None`` si no se resolvio."""
    selected = block.get("selected")
    if not isinstance(selected, Mapping):
        return None
    return str(cast("Mapping[str, object]", selected)["run_sha256"])


def _pick(variants: Sequence[Variant], digest: str | None) -> Variant | None:
    """La variante con ese `run_sha256`, o ``None``."""
    if digest is None:
        return None
    return next((item for item in variants if item.run_sha256 == digest), None)


def _sweep_block(
    *,
    evaluated: Sequence[Variant],
    registry: Registry,
    rows: Sequence[SweepRow],
    best: Variant | None,
    reference_winner: Variant | None,
    dsr: Mapping[str, object],
    pbo: Mapping[str, object],
) -> dict[str, object]:
    """El bloque del barrido: matriz completa, DSR del mejor, PBO y veredicto (A5, A6, A9).

    El mejor del barrido **nunca** sale sin su numero deflactado: el bloque lleva su DSR y el PBO
    de la matriz al lado, y el veredicto se calcula con los dos.
    """
    complete = len(evaluated) == registry.n_trials
    blocking = [item.run_sha256 for item in rows if not item.evaluated]
    if not complete:
        gate = aggregate_verdict(dsr_verdict="not_evaluable", pbo_verdict="not_evaluable")
        return {
            "state": "not_evaluable",
            "verdict": sweep_verdict(gate=gate, pbo={"state": "not_evaluable"}),
            "gate": gate,
            "halves": {
                "deflated_sharpe_ratio": DSR_HALVES["not_evaluable"],
                "probability_of_backtest_overfitting": PBO_HALVES["not_evaluable"],
            },
            "rule": SWEEP_VERDICT_RULE,
            "reason": (
                f"la matriz trae {len(evaluated)} columnas y el registro {registry.n_trials} "
                "intentos: hay variantes intentadas que no se pudieron medir, asi que no hay "
                "`n_trials` con el que deflactar ni matriz completa que someter al CSCV (A5, A7)"
            ),
            "blockers": blocking,
            "n_evaluated": len(evaluated),
            "n_trials": registry.n_trials,
            "n_sweep_variants": len(rows),
            "n_sweep_evaluated": sum(1 for item in rows if item.evaluated),
            "best": None,
            "dsr": dict(dsr),
            "pbo": dict(pbo),
            "pbo_max": PBO_MAX,
            "family_changed": False,
            "follow_ups": [],
            "family_statement": (
                "no hay veredicto de familia: sin matriz completa no se deflacta, y el informe no "
                "declara una mejora que no puede corregir por intentos (A5, A6)"
            ),
        }
    gate = aggregate_verdict(dsr_verdict=str(dsr["verdict"]), pbo_verdict=str(pbo["verdict"]))
    require_consistent_aggregate(
        gate=gate,
        dsr_half=DSR_HALVES[str(dsr["verdict"])],
        pbo_half=PBO_HALVES[str(pbo["verdict"])],
    )
    verdict = sweep_verdict(gate=gate, pbo=pbo)
    family_changed = best is not None and _beats_reference(best, reference_winner)
    return {
        "state": "evaluated",
        "verdict": verdict,
        "gate": gate,
        "halves": {
            "deflated_sharpe_ratio": DSR_HALVES[str(dsr["verdict"])],
            "probability_of_backtest_overfitting": PBO_HALVES[str(pbo["verdict"])],
        },
        "rule": SWEEP_VERDICT_RULE,
        "reason": None,
        "blockers": blocking,
        "n_evaluated": len(evaluated),
        "n_trials": registry.n_trials,
        "n_sweep_variants": len(rows),
        "n_sweep_evaluated": sum(1 for item in rows if item.evaluated),
        "best": (
            None
            if best is None
            else {
                "index": next(item.variant.index for item in rows if item.measured is best),
                "variant_id": best.variant_id,
                "run_sha256": best.run_sha256,
                "brier_score": best.brier_score,
                "log_loss": best.log_loss_value,
                "n_traded": best.n_traded,
                "sharpe_per_session": best.sharpe_per_session,
                "deflated_sharpe_ratio": dict(dsr),
                "note": (
                    "el mejor del barrido es una **hipotesis** medida bajo el protocolo de #26: "
                    "sale con su DSR y con el PBO de la matriz, que es la condicion para que el "
                    "numero signifique algo (A6)"
                ),
            }
        ),
        "pbo_max": PBO_MAX,
        "pbo_within_max": (
            bool(float(cast("float", pbo["pbo"])) <= PBO_MAX)
            if pbo.get("state") == "evaluated"
            else None
        ),
        "dsr": dict(dsr),
        "pbo": dict(pbo),
        "family_changed": family_changed,
        "follow_ups": list(FAMILY_CHANGED_FOLLOW_UPS) if family_changed else [],
        "family_statement": (
            "el barrido **cambiaria** la familia ganadora de #26: el mejor del barrido gana con "
            "la regla declarada, asi que el siguiente paso es #27/#28, nunca sustituir el modelo "
            "en silencio"
            if family_changed
            else "el barrido **no** cambia la familia ganadora de #26 bajo la regla declarada: el "
            "modelo publicado sigue siendo el de #26"
        ),
    }


def _delta_block(measured: Variant, reference: Variant, *, name: str) -> dict[str, object]:
    """La diferencia de una fila frente a una referencia **medida en esta corrida** (A8)."""
    return {
        "reference": name,
        "reference_run_sha256": reference.run_sha256,
        "reference_variant_id": reference.variant_id,
        "rule": "`esta variante - referencia`; **negativo** = mejor Brier/log-loss",
        "brier_score": measured.brier_score - reference.brier_score,
        "log_loss": measured.log_loss_value - reference.log_loss_value,
        "n_traded": measured.n_traded - reference.n_traded,
    }


def _reference_payload(reference: Variant | None) -> dict[str, object] | None:
    """La referencia de la tabla como bloque JSON puro, o ``None`` si no se pudo medir."""
    if reference is None:
        return None
    return {
        "run_sha256": reference.run_sha256,
        "variant_id": reference.variant_id,
        "calibrated": reference.calibrated,
        "brier_score": reference.brier_score,
        "log_loss": reference.log_loss_value,
        "n_traded": reference.n_traded,
        "sharpe_per_session": reference.sharpe_per_session,
        "source": reference.source,
    }


def _row_dsr(row: SweepRow, *, registry: Registry) -> dict[str, object] | None:
    """El DSR **de esa variante**, con el `n_trials` y el `V[SR]` del registro (A6)."""
    if row.measured is None:
        return None
    return deflate_block(returns=row.measured.series, registry=registry)


def _sweep_row_payloads(rows: Sequence[SweepRow], *, registry: Registry) -> list[dict[str, object]]:
    """Las filas del barrido con su serie y **su** DSR: ninguna queda sin deflactar (A6)."""
    out: list[dict[str, object]] = []
    for row in rows:
        payload = row.to_payload()
        payload["deflated_sharpe_ratio"] = _row_dsr(row, registry=registry)
        out.append(payload)
    return out


def _comparison_block(
    rows: Sequence[SweepRow],
    *,
    registry: Registry,
    reference_raw: Variant | None,
    reference_winner: Variant | None,
) -> dict[str, object]:
    """Las filas del barrido con su Brier, su log-loss, sus operadas y sus dos deltas (A8).

    La referencia de la izquierda es la linea base **cruda** de #24 y la de la derecha la
    **ganadora de #26** (la regla de #26 aplicada a sus cuatro candidatos). Las dos se **miden**
    en el mismo proceso: ninguna cifra se copia de los informes congelados de #24/#25/#26.
    """
    published: list[dict[str, object]] = []
    for row in rows:
        entry: dict[str, object] = {
            **row.variant.to_payload(),
            "run_sha256": row.run_sha256,
            "state": row.state,
            "reason": row.reason,
            "n_nulls_in_features": row.n_nulls_in_features,
            "basis": "declared_cost",
            "is_validation": False,
            "deflated_sharpe_ratio": _row_dsr(row, registry=registry),
        }
        measured = row.measured
        if measured is None:
            entry.update(
                {
                    "brier_score": None,
                    "log_loss": None,
                    "n_traded": None,
                    "zeros": None,
                    "pnl_declared_pct_sum": None,
                    "sharpe_per_session": None,
                    "delta_vs_baseline_raw": None,
                    "delta_vs_winner_26": None,
                }
            )
            published.append(entry)
            continue
        entry.update(
            {
                "brier_score": measured.brier_score,
                "log_loss": measured.log_loss_value,
                "n_traded": measured.n_traded,
                "zeros": measured.zeros,
                "pnl_declared_pct_sum": measured.pnl_declared_sum,
                "sharpe_per_session": measured.sharpe_per_session,
                "delta_vs_baseline_raw": (
                    None
                    if reference_raw is None
                    else _delta_block(measured, reference_raw, name="la cruda de #24")
                ),
                "delta_vs_winner_26": (
                    None
                    if reference_winner is None
                    else _delta_block(measured, reference_winner, name="la ganadora de #26")
                ),
            }
        )
        published.append(entry)
    return {
        "basis": "declared_cost",
        "is_validation": False,
        "n_rows": len(published),
        "n_evaluated": sum(1 for row in rows if row.evaluated),
        "reference_raw": _reference_payload(reference_raw),
        "reference_winner_26": _reference_payload(reference_winner),
        "primary_metric": "brier_score",
        "selection_rule": SELECTION_RULE,
        "rows": published,
        "note": (
            "las filas se calculan **en el mismo proceso**, sobre las mismas 500 sesiones de test, "
            "el mismo `SplitPlan` de #69, el mismo coste declarado y el mismo umbral 0,5: lo unico "
            "que cambia entre variantes es la lista de features y/o un eje de "
            "`LIGHTGBM_HYPERPARAMETERS`. Ninguna cifra se copia de los informes congelados de "
            "#24/#25/#26 (A8)"
        ),
    }


def _unit_bug_search_block(variant: Variant | None, *, model: CostModel) -> dict[str, object]:
    """El bloque de #80: la constante **medida** y el desplazamiento declarado por operacion (A9).

    La constante que el motor restaba mal (`c_declared_pct` en lugar de su fraccion) se publica
    como numero declarado —`0,0042 - 0,000042 = 0,004158` por operacion— y el bloque de #26 la
    **mide** en esta corrida: con el motor corregido, la diferencia observada es 0.
    """
    measured = dict(_unit_bug_block(variant))
    # El diferencial declarado por operacion es el **viaje completo** (`plan.md` §3.3: 0,0021 %
    # por lado ⇒ 0,0042 %), que es lo que el motor carga en cada operacion.
    round_trip = model.spread_entry_pct + model.spread_exit_pct
    declared = float(round_trip)
    fraction = declared / 100.0
    return {
        **measured,
        "displacement": {
            "declared_pct": format(round_trip, "f"),
            "correct_fraction": format(round_trip / 100, "f"),
            "per_operation": declared - fraction,
            "identity": (
                "`c_declared_pct - c_fraction_of_notional` = 0,0042 - 0,000042 = 0,004158 "
                "(el diferencial del viaje completo: 0,0021 % por lado)"
            ),
            "state": "fixed_in_#80",
            "affects": (
                "la media y la suma de `pnl_declared` de **todas** las variantes y, con ellas, el "
                "Sharpe"
            ),
            "does_not_affect": (
                "el **orden** entre variantes, las metricas de **probabilidad** (Brier, log-loss, "
                "curva de fiabilidad) ni el PBO, que no leen el P&L"
            ),
            "note": (
                "el desplazamiento es **medido** en esta corrida: "
                "`observed_per_operation - correct_term_per_operation` da 0 cuando el motor resta "
                "`c_fraction_of_notional`"
            ),
        },
    }


def _settings_block(settings: Settings) -> dict[str, object]:
    """La raiz declarada de `--settings`, **sin** rutas absolutas en el payload (A10)."""
    declared = str(settings.data.root)
    return {
        "data_root_declared": declared if not Path(declared).is_absolute() else "<absoluta>",
        "note": (
            "de `settings` solo viaja la raiz declarada y en forma relativa: una ruta absoluta "
            "depende de la maquina y no puede entrar en un payload que se hashea (A10)"
        ),
    }


def _protocol_block(
    *,
    plan: SplitPlan,
    universe: Universe,
    frame: FeatureFrame,
    cost_model: CostModel,
    slippage: SlippageParameter,
) -> dict[str, object]:
    """El protocolo **importado** de #24/#25/#26/#69: plan, features, umbral, bins y coste (A2)."""
    return {
        "source": (
            "el universo y el plan se importan de `analysis.backtest_report` (#69); la matriz de "
            "las cinco familias, de `analysis.feature_frame` (#24/#73); las 10 de control, "
            "el umbral y la semilla, de `models.baseline` (#24); los bins, de "
            "`analysis.baseline_report.CALIBRATION_BINS`; el ajuste y la calibracion, de "
            "`models.lightgbm_model` y `models.calibration` (#26/#25); el coste y el *slippage*, "
            "`backtest.costs` (#8/#11). El modulo **no** redeclara ninguno (A2)"
        ),
        "plan_sha256": plan.plan_sha256,
        "plan_params": {
            "n_splits": PHASE1_PLAN.n_splits,
            "test_size": PHASE1_PLAN.test_size,
            "embargo_sessions": PHASE1_PLAN.embargo_sessions,
            "max_train_size": PHASE1_PLAN.max_train_size,
            "label_horizon": PHASE1_PLAN.label_horizon,
        },
        "n_folds": len(plan.folds),
        "n_test": sum(len(fold.test) for fold in plan.folds),
        "n_sessions": len(universe.inputs),
        "n_features": len(BASELINE_FEATURES),
        "features": list(BASELINE_FEATURES),
        "feature_set": CONTROL_FEATURE_SET,
        "decision_threshold": DECISION_THRESHOLD,
        "decision_rule": "`Direction.LONG` si `p >= 0,5`; `Direction.NOTHING` si no",
        "calibration_bins": CALIBRATION_BINS,
        "calibration_rule": CALIBRATION_RULE,
        "control_hyperparameters": dict(LIGHTGBM_HYPERPARAMETERS),
        "cost": {
            "basis": "declared_cost",
            "is_validation": False,
            "model_name": cost_model.name,
            "spread_entry_pct": format(cost_model.spread_entry_pct, "f"),
            "slippage_state": str(getattr(slippage, "state", "assumed")),
        },
        "cost_basis": "declared_cost",
        "seed": SEED,
        "series_id": SERIES_ID,
        "universe": {
            "n_sessions": len(universe.inputs),
            "first_session": _date_text(universe.first_session),
            "last_session": _date_text(universe.last_session),
            "n_half_days": frame.n_half_days,
            "n_nulls_in_features": frame.n_nulls_in_features,
        },
        "matrix_sha256": frame.matrix.matrix_sha256,
        "feature_spec_sha256": dict(frame.matrix.feature_spec_sha256),
        "feature_code_version": frame.matrix.feature_code_version,
    }


def _registry_block(
    *,
    registry: Registry,
    records: Sequence[ExperimentRecord],
    model_digests: Mapping[str, str],
) -> dict[str, object]:
    """El bloque del registro: `n_trials`, `sr_variance` y las 10 entradas nuevas (A3, A5)."""
    return {
        "runs_directory": DEFAULT_RUNS_ROOT,
        "n_trials": registry.n_trials,
        "sr_variance": None if registry.n_trials < 2 else registry.sr_variance,
        "registry_sha256": registry.registry_sha256,
        "variant_ids": list(registry.variant_ids),
        "entries": [entry.to_payload() for entry in registry.entries],
        "new_entries": [
            {
                "run_sha256": record.run_sha256,
                "variant_id": record.config.variant_id,
                "features": list(record.config.features),
                "n_features": len(record.config.features),
                "model_sha256": model_digests.get(record.run_sha256),
            }
            for record in records
        ],
        "n_new": len(records),
        "registered_before": registry.n_trials - len(records),
        "rule": (
            f"`n_trials` y `sr_variance` se derivan del registro: {len(records)} intentos de "
            f"§19.17 mas los {registry.n_trials - len(records)} que #24/#25/#26 ya habian "
            "registrado, que **no** se re-registran. El **resultado de la escritura** "
            "(`created`/`unchanged`) **no** entra en el payload: un informe seco y uno escrito "
            "tienen que hashear igual (A5, A10)"
        ),
    }


def _dsr_for_best(best: Variant | None, *, registry: Registry) -> dict[str, object]:
    """El DSR del mejor del barrido, con el `n_trials` y el `V[SR]` **del registro** (A6).

    Sin mejor medido no hay serie que deflactar y el bloque se declara: nunca se publica un Sharpe
    de barrido sin su numero deflactado.
    """
    if best is None:
        return {
            "state": "not_evaluable",
            "calculation": "deflated_sharpe_ratio",
            "verdict": "not_evaluable",
            "reason": (
                "el barrido no tiene mejor medido: sin variante seleccionada no hay serie que "
                "deflactar (A6)"
            ),
        }
    return {
        **deflate_block(returns=best.series, registry=registry),
        "selected_run_sha256": best.run_sha256,
        "selected_variant_id": best.variant_id,
        "registry_sharpe_per_session": best.sharpe_per_session,
        "series": (
            "el Sharpe del DSR sale de las **500** sesiones de test con `0.0` donde no opera y el "
            f"`sharpe_per_session` del registro, de las **solo operadas** ({best.n_traded}): son "
            "`per_session` los dos, pero **no miden lo mismo**"
        ),
    }


def _payload(
    *,
    as_of: datetime,
    settings: Settings,
    universe: Universe,
    frame: FeatureFrame,
    plan: SplitPlan,
    rows: Sequence[SweepRow],
    candidates: Sequence[Candidate],
    registry: Registry,
    records: Sequence[ExperimentRecord],
    model_digests: Mapping[str, str],
    space: Mapping[str, object],
    sweep_selection: Mapping[str, object],
    best: Variant | None,
    reference_26: Mapping[str, object],
    reference_raw: Variant | None,
    reference_winner: Variant | None,
    cost_model: CostModel,
    slippage: SlippageParameter,
) -> dict[str, object]:
    """El payload canonico del informe: tipos JSON puros y determinista (A10)."""
    evaluated = _evaluated(candidates)
    matrix = _matrix_block(evaluated, registry=registry)
    matrix_gate = matrix_matches_registry(registry=registry, n_columns=len(evaluated))
    dsr = _dsr_for_best(best, registry=registry)
    pbo = pbo_block(returns_matrix=_series_matrix(evaluated), blocks=PBO_BLOCKS)
    sweep = _sweep_block(
        evaluated=evaluated,
        registry=registry,
        rows=rows,
        best=best,
        reference_winner=reference_winner,
        dsr=dsr,
        pbo=pbo,
    )
    unit_bug_variant = best if best is not None else next(iter(evaluated), None)
    comparison = _comparison_block(
        rows, registry=registry, reference_raw=reference_raw, reference_winner=reference_winner
    )
    raw: dict[str, object] = {
        "analysis": "cfdtrader.analysis.hyperparameter_search",
        "task": "#82",
        "title": (
            "Barrido pre-registrado de la familia LightGBM: hiperparametros y subconjuntos de "
            "features, con presupuesto y registro"
        ),
        "variant_id": VARIANT_ID,
        "generated_at": as_of.isoformat(),
        "report_date": as_of.date().isoformat(),
        "hash_format": REPORT_HASH_FORMAT,
        "basis": "declared_cost",
        "is_validation": False,
        "gate": "fail",
        "phase1_ready": False,
        "phase2_ready": False,
        "llm_overlay": "disabled",
        "scheduler": "none",
        "clock": {
            "as_of": as_of.isoformat(),
            "rule": (
                "el modulo no lee el reloj: el instante entra por `--as-of` (obligatorio para "
                "escribir) y `generated_at = as_of`; no hay `datetime.now`, `utcnow`, "
                "`date.today` ni `time.time` en el fuente (A11)"
            ),
        },
        "settings": _settings_block(settings),
        "space": dict(space),
        "protocol": _protocol_block(
            plan=plan, universe=universe, frame=frame, cost_model=cost_model, slippage=slippage
        ),
        "registry": _registry_block(
            registry=registry, records=records, model_digests=model_digests
        ),
        "sweep": sweep,
        "selection": dict(sweep_selection),
        "reference_26": dict(reference_26),
        "matrix": matrix,
        "matrix_gate": {
            "matches": matrix_gate,
            "n_columns": len(evaluated),
            "n_trials": registry.n_trials,
            "rule": (
                "`require_trials_match_registry` de #16, llamado con el numero de columnas de la "
                "matriz: una matriz incompleta (13 columnas de 14 intentos, por ejemplo) es una "
                "discrepancia **declarada** con `TrialsMismatchError`, no una cifra publicada. Lo "
                "que la puerta impide —y este informe nunca hace— es **deflactar** con un "
                "`n_trials` que no sea el del registro (A5)"
            ),
        },
        "deflated_sharpe_ratio": dsr,
        "probability_of_backtest_overfitting": pbo,
        "comparison": comparison,
        "declared_cost": _declared_cost_block(evaluated, model=cost_model),
        "net_metrics": {
            "state": "not_computable",
            "reason": NET_METRICS_REASON,
            "where": "cfdtrader.backtest.metrics.calculate_metrics",
            "follow_up": ["#62", "#60"],
        },
        "unit_bug_80": _unit_bug_search_block(unit_bug_variant, model=cost_model),
        "calibration_note": {
            "bins": CALIBRATION_BINS,
            "log_loss_epsilon": LOG_LOSS_EPSILON,
            "note": (
                "la curva de fiabilidad de cada variante se reproduce desde las probabilidades y "
                "las etiquetas que publica `sweep_rows[]` (`deciding_probabilities` y `outcomes`) "
                "con los `calibration_bins` declarados, y la `log_loss` va recortada en "
                "`[epsilon, 1 - epsilon]` como #15"
            ),
        },
        "honesty_walls": list(HONESTY_WALLS),
        "limits": {
            "gate": "fail",
            "phase1_ready": False,
            "phase2_ready": False,
            "is_validation": False,
            "statement": (
                "los numeros de este informe miden un barrido; **no** validan la estrategia: la "
                "puerta de Fase 0 sigue en `fail` (#9/#64/#18), el *slippage* es un supuesto (#62) "
                "y el umbral economico no esta decidido (#60)"
            ),
            "net_metrics_state": "not_computable",
            "costs": "declared_not_measured",
            "slippage_state": "assumed (#64), nunca medido",
            "threshold": DECISION_THRESHOLD,
            "threshold_issue": "#27",
            "direction": "larga unica (#78 cubre la corta)",
            "plan_sections_untouched": ["§11.6", "§19.6", "§19.7"],
        },
        "features_limitations": list(FEATURES_LIMITATIONS),
        "limitations": list(LIMITATIONS),
        "does_not_do": [dict(item) for item in REPORT_DOES_NOT_DO],
        "follow_ups": [dict(item) for item in FOLLOW_UPS],
        "sweep_rows": _sweep_row_payloads(rows, registry=registry),
    }
    return raw


@dataclass(frozen=True, slots=True)
class SearchReport:
    """El informe: payload canonico, hash y los objetos que lo produjeron.

    ``payload`` es el texto que se hashea y **no** incluye ``report_sha256``: un informe no se
    hashea a si mismo.
    """

    as_of: datetime
    report_date: date
    payload: dict[str, object]
    report_sha256: str
    universe: Universe
    features: FeatureFrame
    split_plan: SplitPlan
    rows: tuple[SweepRow, ...]
    candidates: tuple[Candidate, ...]
    registry: Registry
    records: tuple[ExperimentRecord, ...]
    model_digests: Mapping[str, str]
    outcomes: Mapping[str, WriteOutcome | None]
    space: Mapping[str, object]
    selection: Mapping[str, object]

    @property
    def report_stem(self) -> str:
        """Nombre base del informe: ``lightgbm_search_<AAAA-MM-DD>``."""
        return f"{REPORT_PREFIX}_{self.report_date.isoformat()}"

    @property
    def evaluated(self) -> tuple[Variant, ...]:
        """Las variantes medidas del registro, en su orden."""
        return _evaluated(self.candidates)

    @property
    def sweep(self) -> Mapping[str, object]:
        """El bloque del barrido del payload (veredicto, mejor, `family_changed`)."""
        return cast("Mapping[str, object]", self.payload["sweep"])

    def json_text(self) -> str:
        """JSON determinista: mismas entradas y mismo ``as_of`` ⇒ mismo texto byte a byte."""
        published: dict[str, object] = {**self.payload, "report_sha256": self.report_sha256}
        return _json_text(published)

    def write(self, directory: Path) -> tuple[Path, Path]:
        """Escribe el informe en JSON y Markdown. Solo escribe quien lo pida (el CLI)."""
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{self.report_stem}.json"
        markdown_path = directory / f"{self.report_stem}.md"
        json_path.write_text(self.json_text(), encoding="utf-8")
        markdown_path.write_text(render_markdown(self), encoding="utf-8")
        return json_path, markdown_path


def _wide(value: object) -> dict[str, object]:
    """Un bloque anidado del payload, ya tipado (sin inventar claves)."""
    return cast("dict[str, object]", value)


def _axis_label(row: Mapping[str, object]) -> str:
    """La etiqueta estable de una fila: su eje (de hiperparametros) o su subconjunto."""
    if row["kind"] == "hyperparameters":
        return f"`{row['axis']}`"
    return f"`{row['feature_set']}`"


def _reference_line(reference: object) -> str:
    """La linea de una referencia de la tabla, o su ausencia declarada."""
    if not isinstance(reference, Mapping):
        return "no medida en esta corrida (`null`), asi que su delta no se publica"
    item = cast("Mapping[str, object]", reference)
    return (
        f"`{item['variant_id']}` (`{item['run_sha256']}`) con Brier "
        f"{_number(item['brier_score'])}, log-loss {_number(item['log_loss'])} y "
        f"{item['n_traded']} operadas"
    )


def _row_dsr_value(row: Mapping[str, object]) -> object:
    """El DSR publicado de una fila, o ``None`` si la variante no se midio."""
    block = row.get("deflated_sharpe_ratio")
    if not isinstance(block, Mapping):
        return None
    return cast("Mapping[str, object]", block).get("dsr")


def render_markdown(report: SearchReport) -> str:
    """El informe en Markdown, determinista y sin cifras que no esten en el payload."""
    payload = report.payload
    space = _wide(payload["space"])
    registry = _wide(payload["registry"])
    sweep = _wide(payload["sweep"])
    selection = _wide(payload["selection"])
    reference = _wide(payload["reference_26"])
    comparison = _wide(payload["comparison"])
    rows = cast("list[dict[str, object]]", comparison["rows"])
    reference_selected = reference.get("selected")

    lines: list[str] = [
        "# Barrido pre-registrado de la familia LightGBM (#82)",
        "",
        f"Variante `{payload['variant_id']}`: **{space['n_variants']}** variantes de §19.17, cada "
        f"una con su entrada en `runs/`. Generado el `{payload['generated_at']}` (**declarado**, "
        f"no leido del reloj). `report_sha256 = {report.report_sha256}`.",
        "",
        f"**No es una validacion.** `gate = {payload['gate']}`, "
        f"`phase2_ready = {payload['phase2_ready']}`.",
        "",
        "## Espacio pre-registrado (§19.17, decision del PM)",
        "",
        f"- {space['source']}",
        f"- Presupuesto: **{space['budget']}** variantes nuevas ⇒ **{space['total_trials']}** "
        f"`n_trials` contando las **{space['registered_trials']}** de #24/#25/#26; catalogo: "
        f"**{space['catalog_columns']}** columnas.",
        f"- {space['rule']}",
        "",
        "| variante | tipo | eje | control | barrido | features | procedencia |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    axes = cast("list[dict[str, object]]", space["axes"])
    for row in cast("list[dict[str, object]]", space["variants"]):
        axis = next((item for item in axes if item["name"] == row["axis"]), None)
        lines.append(
            f"| `{row['variant_id']}` | {row['kind']} | {_axis_label(row)} | "
            f"{_number(None if axis is None else axis['control'])} | "
            f"{_number(None if axis is None else axis['value'])} | {row['n_features']} | "
            f"{row['provenance']} |"
        )
    lines.extend(
        [
            "",
            "## Subconjuntos de features (del catalogo de #73)",
            "",
            "| conjunto | columnas | origen | procedencia |",
            "| --- | --- | --- | --- |",
        ]
    )
    for conjunto in cast("list[dict[str, object]]", space["feature_sets"]):
        columns = cast("list[str]", conjunto["columns"])
        lines.append(
            f"| `{conjunto['name']}` | {conjunto['n_columns']} | `{conjunto['source']}` | "
            f"{conjunto['provenance']} |"
        )
        lines.append(f"  - `{'`, `'.join(columns)}`")
    lines.extend(
        [
            "",
            "## Registro (los intentos con los que se deflacta)",
            "",
            f"- `n_trials = {registry['n_trials']}`, "
            f"`V[SR] = {_number(registry['sr_variance'])}`, "
            f"`registry_sha256 = {registry['registry_sha256']}`: **{registry['n_new']}** entradas "
            f"nuevas mas **{registry['registered_before']}** de #24/#25/#26.",
            f"- {registry['rule']}",
            "",
            "## Tabla del barrido (las 10 variantes, las mismas 500 sesiones de test)",
            "",
            "| # | variante | eje / subconjunto | features | nulos | estado | operadas | Brier | "
            "log-loss | Sharpe | DSR | Δ Brier vs cruda #24 | Δ Brier vs ganadora #26 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        delta_raw = cast("dict[str, object] | None", row["delta_vs_baseline_raw"])
        delta_winner = cast("dict[str, object] | None", row["delta_vs_winner_26"])
        lines.append(
            f"| {row['index']} | `{row['variant_id']}` | {_axis_label(row)} | {row['n_features']} "
            f"| {row['n_nulls_in_features']} | `{row['state']}` | {_number(row['n_traded'])} | "
            f"{_number(row['brier_score'])} | {_number(row['log_loss'])} | "
            f"{_number(row['sharpe_per_session'])} | {_number(_row_dsr_value(row))} | "
            f"{_number(None if delta_raw is None else delta_raw['brier_score'])} | "
            f"{_number(None if delta_winner is None else delta_winner['brier_score'])} |"
        )
    lines.extend(
        [
            f"- {comparison['note']}",
            f"- Referencia izquierda: {_reference_line(comparison.get('reference_raw'))}",
            f"- Referencia derecha: {_reference_line(comparison.get('reference_winner_26'))}",
            "",
            "## La familia de #26: se usa, no se reabre",
            "",
            f"- Regla de seleccion (la de #26, publicada literal): {SELECTION_RULE}",
            f"- `primary_metric = {selection['primary_metric']}`, "
            f"`tie_breakers = {selection['tie_breakers']}`, "
            f"`tie_tolerance = {selection['tie_tolerance']}`; estado del barrido: "
            f"`{selection['state']}`.",
        ]
    )
    if isinstance(reference_selected, Mapping):
        selected = cast("Mapping[str, object]", reference_selected)
        lines.append(
            f"- **Ganadora de #26** (recalculada aqui sobre sus cuatro candidatos): "
            f"`{selected['variant_id']}` (`{selected['run_sha256']}`) con Brier "
            f"{_number(selected['brier_score'])}, log-loss {_number(selected['log_loss'])} y "
            f"{selected['n_traded']} operadas."
        )
    else:
        lines.append(
            "- Sin ganadora de #26 reconstruible: la referencia del delta se declara `null`."
        )
    return _markdown_tail(lines, payload=payload, sweep=sweep)


def _markdown_tail(
    lines: list[str], *, payload: Mapping[str, object], sweep: Mapping[str, object]
) -> str:
    """El cierre del `.md`: mejor del barrido, correccion por intentos, vallas y seguimientos."""
    matrix = _wide(payload["matrix"])
    dsr = _wide(sweep["dsr"])
    pbo = _wide(sweep["pbo"])
    cost = _wide(payload["declared_cost"])
    net = _wide(payload["net_metrics"])
    bug = _wide(payload["unit_bug_80"])
    displacement = _wide(bug["displacement"])
    best = sweep.get("best")
    lines.extend(["", "## Mejor del barrido y correccion por intentos (DSR y PBO)", ""])
    if isinstance(best, Mapping):
        winner = cast("Mapping[str, object]", best)
        lines.append(
            f"- **Mejor del barrido**: `{winner['variant_id']}` (`{winner['run_sha256']}`) con "
            f"Brier {_number(winner['brier_score'])}, log-loss {_number(winner['log_loss'])}, "
            f"{winner['n_traded']} operadas y Sharpe "
            f"{_number(winner['sharpe_per_session'])}; `family_changed = "
            f"{sweep['family_changed']}`."
        )
    else:
        lines.append(f"- **Sin mejor medido**: `{sweep['state']}` — {sweep.get('reason')}")
    follow_ups = cast("list[str]", sweep["follow_ups"])
    lines.extend(
        [
            f"- {sweep['family_statement']}"
            + (f" Seguimiento: {', '.join(follow_ups)}." if follow_ups else ""),
            f"- Matriz: **{matrix['n_observations']}** filas × **{matrix['n_variants']}** "
            f"columnas ({matrix['blocks']} bloques), "
            f"`matrix_matches_registry = {matrix['matrix_matches_registry']}` (registro: "
            f"{matrix['registry_n_trials']}); variantes del barrido evaluadas: "
            f"**{sweep['n_sweep_evaluated']}** de {sweep['n_sweep_variants']}.",
            f"- DSR del mejor: `{_number(dsr.get('dsr'))}` (`{dsr.get('verdict')}`) con "
            f"`sr_observed = {_number(dsr.get('sr_observed'))}`, "
            f"`sr0_expected_max = {_number(dsr.get('sr0_expected_max'))}`, "
            f"`deflation = {dsr.get('deflation')}`, `n_trials = {dsr.get('n_trials')}`, "
            f"`sr_variance = {_number(dsr.get('sr_variance'))}`.",
            f"- PBO de la matriz: `{_number(pbo.get('pbo'))}` (`{pbo.get('verdict')}`) contra "
            f"`pbo_max = {sweep['pbo_max']}`, metodo `{pbo.get('method')}`, "
            f"`n_combinations_drawn = {pbo.get('n_combinations_drawn')}`.",
            f"- Agregado de #9: **`{sweep['gate']}`** (mitades `{sweep['halves']}`).",
            f"- {sweep['rule']}",
            f"- **Veredicto del barrido: `{sweep['verdict']}`** — `pass` no existe aqui: el "
            "barrido produce hipotesis y la puerta de la estrategia sigue en `fail`.",
        ]
    )
    blockers = cast("list[str]", sweep["blockers"])
    if blockers:
        lines.extend(
            ["", "- **Vetos** (variante intentada sin columna):"]
            + [f"  - `{item}`" for item in blockers]
        )
    lines.extend(
        [
            "",
            "## Coste declarado y metricas netas",
            "",
            f"- `basis = {cost['basis']}`, `is_validation = {cost['is_validation']}`; "
            f"modelo: `{cost['model']}`.",
            f"- `net_metrics`: `{net['state']}` — {net['reason']}",
            f"- Seguimiento: `{net['follow_up']}`.",
            "",
            "## Unidades del motor (#80, arreglado y medido)",
            "",
            f"- Observado por operacion: `{_number(bug['observed_per_operation'])}`; termino "
            f"correcto: `{_number(bug['correct_term_per_operation'])}`; diferencia: "
            f"`{_number(bug['difference_per_operation'])}` sobre `{bug['n_operations']}` "
            "operaciones.",
            f"- Desplazamiento declarado por operacion: "
            f"`{_number(displacement['per_operation'])}` ({displacement['identity']}), estado "
            f"`{displacement['state']}`.",
            f"- Afecta a: {displacement['affects']}",
            f"- **No** afecta a: {displacement['does_not_affect']}",
            f"- Unidad declarada: {bug['unit']}. {bug['resolution']}",
            "",
            "## Vallas de honestidad",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in cast("list[str]", payload["honesty_walls"]))
    lines.extend(["", "## Limitaciones", ""])
    lines.extend(f"- {item}" for item in cast("list[str]", payload["limitations"]))
    lines.append("")
    lines.extend(f"- {item}" for item in cast("list[str]", payload["features_limitations"]))
    lines.extend(["", "## Que no hace este modulo", ""])
    lines.extend(
        f"- **{item['issue']}** · `{item['id']}`: {item['statement']}"
        for item in cast("list[dict[str, str]]", payload["does_not_do"])
    )
    lines.extend(["", "## Seguimientos", ""])
    lines.extend(
        f"- **{item['issue']}** · {item['topic']}: {item['why']}"
        for item in cast("list[dict[str, str]]", payload["follow_ups"])
    )
    lines.append("")
    return "\n".join(lines)


def matrix_matches_registry(*, registry: Registry, n_columns: int) -> bool:
    """La puerta de #16: `n_trials` y matriz tienen que ser **el mismo** numero (A5).

    Llama a :func:`cfdtrader.analysis.experiment_log.require_trials_match_registry` —la puerta que
    hace imposible inyectar un `n_trials` estimado— y **declara** el resultado en vez de tumbar el
    barrido: un intento que no se pudo medir deja la matriz incompleta (13 columnas de 14 intentos)
    y eso es un estado que se publica. Lo que esa puerta impide, y el barrido nunca hace, es
    **deflactar** con un numero que no sea el del registro.
    """
    try:
        require_trials_match_registry(
            n_trials=n_columns, sr_variance=registry.sr_variance, registry=registry
        )
    except (TrialsMismatchError, ExperimentLogError):
        return False
    return True


def _selection_of_sweep(
    rows: Sequence[SweepRow],
) -> tuple[dict[str, object], Variant | None]:
    """El mejor del barrido con la regla **de #26**, entre las variantes medidas (A6)."""
    measured = tuple(row.measured for row in rows if row.measured is not None)
    block = selection_block(measured)
    return block, _pick(measured, _selected_of(block))


def _attempt(
    variant: SearchVariant,
    *,
    frame: FeatureFrame,
    labels: pl.DataFrame,
    plan: SplitPlan,
    horizon: Sequence[int],
    universe: Universe,
    cost_model: CostModel,
    slippage: SlippageParameter,
) -> tuple[DesignFrame, LightGBMModel | None, Variant | None, str | None, str | None]:
    """Intenta **una** variante: su diseno, su ajuste y su medida, o el motivo de no tenerlos (A7).

    Devuelve siempre el diseno (con su `n_nulls_in_features`, que se publica aunque la variante no
    se pueda medir) y, si se pudo, el modelo y la medida del motor.
    """
    design = design_for(variant, features=frame.matrix.frame, labels=labels)
    try:
        _, model = fit_search_variant(
            variant,
            features=frame.matrix.frame,
            labels=labels,
            plan=plan,
            horizon=horizon,
        )
    except (VariantNotEvaluableError, LightGBMError, HyperparameterSearchError) as failure:
        return design, None, None, str(failure), type(failure).__name__
    measured = measure_variant(
        variant,
        model,
        design,
        universe=universe,
        plan=plan,
        frame=frame,
        cost_model=cost_model,
        slippage=slippage,
    )
    return design, model, measured, None, None


def _record_and_write(
    variant: SearchVariant,
    *,
    design: DesignFrame,
    model: LightGBMModel | None,
    measured: Variant | None,
    reason: str | None,
    error: str | None,
    frame: FeatureFrame,
    plan: SplitPlan,
    runs_root: Path,
    as_of: datetime,
    write: bool,
    outcomes: dict[str, WriteOutcome | None],
    digests: dict[str, str],
) -> tuple[SweepRow, ExperimentRecord]:
    """Registra el intento —se pueda medir o no— y escribe su ``model.json`` si lo hay (A3, A7).

    Un intento sin medida se registra con el resultado nulo **declarado** (`0.0` / `0`) y con su
    motivo en la fila del informe: no se inventa un Sharpe y **no** se le da columna a la matriz.
    """
    record = record_experiment(
        runs_root=runs_root,
        config=variant_config(variant, frame=frame, plan=plan),
        result=ExperimentResult(
            sharpe_per_session=0.0 if measured is None else measured.sharpe_per_session,
            n_observations=0 if measured is None else measured.n_traded,
        ),
        as_of=as_of,
        write=write,
    )
    digest: str | None = None
    if model is not None and measured is not None:
        model_payload = model.to_payload()
        digest = _digest(model_payload)
        outcomes[record.run_sha256] = (
            _write_immutable(
                record.directory / MODEL_FILE,
                _model_text(record=record, payload=model_payload, digest=digest),
            )
            if write
            else WriteOutcome.UNCHANGED
        )
        digests[record.run_sha256] = digest
        measured = dataclasses.replace(measured, run_sha256=record.run_sha256)
    row = SweepRow(
        variant=variant,
        run_sha256=record.run_sha256,
        state="evaluated" if measured is not None else "not_evaluable",
        reason=reason,
        error=error,
        n_sessions=design.n_sessions,
        n_labels=design.n_labels,
        n_shifted_rows=design.n_shifted_rows,
        n_nulls_in_features=design.n_nulls_in_features,
        model_sha256=digest,
        measured=measured,
    )
    return row, record


def analyse(
    *,
    store: Store,
    reports_dir: Path,
    runs_root: Path,
    settings: Settings,
    as_of: datetime,
    write: bool = True,
) -> SearchReport:
    """Corre las 10 variantes de §19.17, las registra y escribe el informe (A3, A7, A10).

    ``as_of`` es **obligatorio** y es el unico instante de la corrida: ninguna ruta consulta el
    reloj. ``write=False`` (el ``--dry-run`` del CLI) no escribe **nada**: ni el informe ni las
    carpetas del registro.
    """
    moment = _as_utc(as_of)
    audit = space_audit()
    history = load_history(store, series_id=SERIES_ID)
    calendar = load_calendar(years=_calendar_years(history.daily))
    universe = build_inputs(history, calendar=calendar)
    frame = build_feature_frame(store, series_id=SERIES_ID)
    _require_alignment(universe, frame)
    plan = build_split_plan(universe.inputs, params=PHASE1_PLAN)
    horizon = _label_horizon(plan)
    cost_model = declared_cost_model()
    slippage: SlippageParameter = declared_slippage_assumption()

    rows: list[SweepRow] = []
    records: list[ExperimentRecord] = []
    known: dict[str, Variant] = {}
    outcomes: dict[str, WriteOutcome | None] = {}
    digests: dict[str, str] = {}
    for variant in SEARCH_SPACE:
        design, model, measured, reason, error = _attempt(
            variant,
            frame=frame,
            labels=frame.labels,
            plan=plan,
            horizon=horizon,
            universe=universe,
            cost_model=cost_model,
            slippage=slippage,
        )
        row, record = _record_and_write(
            variant,
            design=design,
            model=model,
            measured=measured,
            reason=reason,
            error=error,
            frame=frame,
            plan=plan,
            runs_root=runs_root,
            as_of=moment,
            write=write,
            outcomes=outcomes,
            digests=digests,
        )
        if row.measured is not None:
            known[row.run_sha256] = row.measured
        rows.append(row)
        records.append(record)

    registry = load_registry(runs_root, extra=tuple(records))
    candidates = _candidates_from_registry(
        registry=registry,
        known=known,
        runs_root=runs_root,
        universe=universe,
        frame=frame,
        plan=plan,
        cost_model=cost_model,
        slippage=slippage,
    )
    evaluated = _evaluated(candidates)
    sweep_selection, best = _selection_of_sweep(rows)
    old_families = {BASELINE_VARIANT_ID, LIGHTGBM_VARIANT_ID}
    reference_26 = selection_block(
        tuple(
            item
            for item in candidates
            if isinstance(item, Variant) and item.variant_id in old_families
        )
    )
    reference_winner = _pick(
        tuple(item for item in evaluated if item.variant_id in old_families),
        _selected_of(reference_26),
    )
    reference_raw = next(
        (
            item
            for item in evaluated
            if item.variant_id == BASELINE_VARIANT_ID and not item.calibrated
        ),
        None,
    )
    payload = _payload(
        as_of=moment,
        settings=settings,
        universe=universe,
        frame=frame,
        plan=plan,
        rows=rows,
        candidates=candidates,
        registry=registry,
        records=tuple(records),
        model_digests=digests,
        space=audit,
        sweep_selection=sweep_selection,
        best=best,
        reference_26=reference_26,
        reference_raw=reference_raw,
        reference_winner=reference_winner,
        cost_model=cost_model,
        slippage=slippage,
    )
    report = SearchReport(
        as_of=moment,
        report_date=moment.date(),
        payload=payload,
        report_sha256=regeneration_delta.HASH_PREFIX + _digest(payload),
        universe=universe,
        features=frame,
        split_plan=plan,
        rows=tuple(rows),
        candidates=candidates,
        registry=registry,
        records=tuple(records),
        model_digests=digests,
        outcomes=outcomes,
        space=audit,
        selection=sweep_selection,
    )
    if write:
        json_path, markdown_path = report.write(reports_dir)
        logger.info("informe del barrido: {} y {}", json_path, markdown_path)
    return report


def _as_utc(value: datetime) -> datetime:
    """Normaliza el instante declarado a UTC; sin zona se interpreta como UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_as_of(value: str | None) -> datetime:
    """El instante declarado de la CLI: **obligatorio**, y nunca tomado del reloj (A11)."""
    if value is None:
        raise MissingAsOfError(
            "falta --as-of: escribir el informe exige un instante declarado (el modulo no lee el "
            "reloj) y sin el **no se escribe ningun fichero** (A11)"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise InvalidAsOfError(
            f"--as-of no es un instante ISO-8601 valido ({value!r}): {error}"
        ) from error
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    """El CLI del barrido: ``0`` = informe emitido (el veredicto puede ser `fail`), ``2`` = nada.

    Acepta ``--data-root``, ``--reports-dir``, ``--runs-root``, ``--settings``, ``--as-of`` y
    ``--dry-run``. **No** acepta banderas de espacio ni de presupuesto (`--n-trials`, `--budget`,
    hiperparametros sueltos): el espacio es §19.17 y ampliarlo exige una decision nueva (A11).
    """
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description=(
            "Barrido pre-registrado de la familia LightGBM (hiperparametros y subconjuntos de "
            "features) con presupuesto, registro y correccion por intentos (#82)"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=None, help="raiz del almacen")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="directorio de informes (por defecto <data-root>/derived/reports)",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="raiz del registro de experimentos (por defecto ./runs)",
    )
    parser.add_argument("--settings", type=Path, default=None, help="ruta de settings.yaml")
    parser.add_argument(
        "--as-of",
        default=None,
        help="instante declarado ISO-8601, obligatorio para escribir (el modulo no lee el reloj)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="no escribe nada (ni el informe ni las carpetas del registro)",
    )
    args = parser.parse_args(argv)

    try:
        moment = _parse_as_of(cast("str | None", args.as_of))
        settings = load_settings(args.settings)
    except (HyperparameterSearchError, ConfigurationError) as error:
        print(f"no se puede emitir el barrido: {error}", file=sys.stderr)
        return 2

    data_root = Path(args.data_root) if args.data_root is not None else Path(settings.data.root)
    reports_dir = (
        Path(args.reports_dir)
        if args.reports_dir is not None
        else data_root / "derived" / "reports"
    )
    runs_root = Path(args.runs_root) if args.runs_root is not None else Path(DEFAULT_RUNS_ROOT)
    try:
        report = analyse(
            store=Store(data_root),
            reports_dir=reports_dir,
            runs_root=runs_root,
            settings=settings,
            as_of=moment,
            write=not bool(args.dry_run),
        )
    except (
        HyperparameterSearchError,
        FeatureFrameError,
        ExperimentLogError,
        ModelComparisonError,
        LightGBMError,
        backtest_report.BacktestReportError,
    ) as error:
        print(f"no se puede emitir el barrido: {error}", file=sys.stderr)
        return 2

    sweep = report.sweep
    logger.info(
        "barrido: {} variantes intentadas de {} del registro, {} medidas; veredicto {}; "
        "report_sha256 = {}",
        len(report.rows),
        report.registry.n_trials,
        sum(1 for row in report.rows if row.evaluated),
        sweep["verdict"],
        report.report_sha256,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
