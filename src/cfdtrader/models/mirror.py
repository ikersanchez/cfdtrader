"""Modelo espejo para la direccion **corta**: `ret_short` y su propia identidad de registro (#78).

#24 entrena **una sola** direccion (la larga), con ``y = 1{ret_long > 0}``. La direccion corta
—``ret_short`` del catalogo de ``derived.labels`` (#10)— queda fuera y este modulo la cierra.

**Lo que hace.** Reutiliza **por import** la maquinaria de ``models.baseline`` (el frame de
diseno, el ajuste con elastic net fold a fold y el calibrador de #25) cambiando **solo** la
columna de objetivo: ``y = 1{ret_short > 0}``. No hay una segunda formula de etiqueta, ni un
segundo codigo de ajuste, ni un segundo camino de calibracion; el espejo **no** es un cambio de
signo —el coste declarado de #8/#11 es asimetrico— sino la misma pregunta planteada al reves.

**Su propia identidad.** ``MIRROR_VARIANT_ID`` es la del experimento en ``runs/`` (#16: la
identidad es el **contenido** de la configuracion, no la ruta ni el instante), ``MIRROR_FEATURES``
es su lista declarada —las **mismas** 10 de ``BASELINE_FEATURES``, que es la razon de que el
espejo no necesite un catalogo nuevo— y :func:`mirror_identity` publica las dos cosas mas el
objetivo y la semilla. Un ``run_sha256`` distinto del de #24 sale por construccion, porque el
``variant_id`` viaja en la configuracion.

**Lo que NO hace** (fronteras declaradas, con su issue)

- **No** decide el umbral economico ni el *sizing*: el umbral 0,5 es el del informe y los del
  sistema son #27 y #60, igual que en #24.
- **No** escribe nada: ni el registro de ``runs/`` (#16) ni un informe. La fila del registro y la
  medicion con el motor de #13 son la capa de analisis, que importa este modulo.
- **No** compara ni selecciona: el DSR/PBO con el que se decide si el espejo aporta es #26.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final

import polars as pl

from cfdtrader.models.baseline import (
    BASELINE_FEATURES,
    DECISION_THRESHOLD,
    HYPERPARAMETERS,
    SEED,
    BaselineError,
    BaselineModel,
    DesignFrame,
    SplitAssignment,
    design_frame,
    fit_baseline,
)

__all__ = [
    "MIRROR_DOES_NOT_DO",
    "MIRROR_FEATURES",
    "MIRROR_HYPERPARAMETERS",
    "MIRROR_TARGET_COLUMN",
    "MIRROR_TARGET_RULE",
    "MIRROR_VARIANT_ID",
    "MirrorError",
    "fit_mirror",
    "mirror_design_frame",
    "mirror_identity",
    "short_signal",
]

#: Identidad registrada del espejo (A11 de #78): **una** variante, sin barridos. Es distinta de
#: ``baseline_logit_elasticnet_v1`` (#24) **por el objetivo**, y es lo que hace que su
#: ``run_sha256`` sea otro: la identidad del experimento de #16 es el contenido de su config.
MIRROR_VARIANT_ID: Final[str] = "baseline_logit_elasticnet_short_v1"

#: La columna de etiqueta que produce ``y``: la direccion **corta** (#10).
MIRROR_TARGET_COLUMN: Final[str] = "ret_short"

#: El objetivo, en prosa, tal y como lo publica el informe: espejo exacto de ``ret_long > 0``.
MIRROR_TARGET_RULE: Final[str] = "ret_short > 0"

#: Su lista de features, declarada. Son las **mismas** 10 de #24 a proposito: el espejo duplica
#: el numero de parametros sobre las mismas 2 687 sesiones y las mismas features (la declaracion
#: de #78), asi que un catalogo distinto seria otro experimento, no este.
MIRROR_FEATURES: Final[tuple[str, ...]] = BASELINE_FEATURES

#: Hiperparametros declarados del espejo: los 8 de #24, sin barrido (A7). Una variante con otros
#: valores se **pasa por argumento** a :func:`fit_mirror` y se registra como otro experimento.
MIRROR_HYPERPARAMETERS: Final[dict[str, object]] = dict(HYPERPARAMETERS)

#: Que **no** hace este modulo, legible por maquina. Cada frontera con su issue.
MIRROR_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_escribe_el_registro",
        "issue": "#16",
        "statement": (
            "no escribe `runs/` ni un informe: solo ajusta el modelo y declara su identidad; la "
            "fila del registro la escribe la capa de analisis con `record_experiment`"
        ),
    },
    {
        "id": "no_compara_ni_selecciona",
        "issue": "#26",
        "statement": (
            "no compara el espejo con la familia larga ni decide si aporta: el DSR/PBO de la "
            "seleccion es #26"
        ),
    },
    {
        "id": "no_decide_el_umbral_economico",
        "issue": "#27",
        "statement": (
            "el umbral 0,5 es el del informe, no el del sistema: el umbral economico y el "
            "*sizing* son #27 y #60"
        ),
    },
)


class MirrorError(BaselineError):
    """Se intento ajustar el espejo con un frame de diseno que **no** es el de la direccion corta.

    El espejo solo acepta un :class:`DesignFrame` cuyo objetivo sea ``ret_short`` (#78): pasarle
    el frame de #24 (``ret_long``) seria entrenar la direccion larga llamandola corta.
    """


def mirror_identity() -> dict[str, object]:
    """La identidad declarada del espejo: lo que viaja a la configuracion registrada de #16.

    Publica el ``variant_id``, la direccion, el objetivo (columna y regla), las features, los
    hiperparametros y la semilla. Todos son JSON puros: la identidad del experimento es su
    contenido, nunca la ruta ni el instante (A12 de #16).
    """
    return {
        "variant_id": MIRROR_VARIANT_ID,
        "direction": "short",
        "target_column": MIRROR_TARGET_COLUMN,
        "target_rule": MIRROR_TARGET_RULE,
        "features": list(MIRROR_FEATURES),
        "hyperparameters": dict(MIRROR_HYPERPARAMETERS),
        "seed": SEED,
        "decision_threshold": DECISION_THRESHOLD,
        "does_not_do": [dict(item) for item in MIRROR_DOES_NOT_DO],
    }


def mirror_design_frame(
    features: pl.DataFrame,
    *,
    labels: pl.DataFrame,
    availability: Mapping[str, int],
    selected: Sequence[str] = MIRROR_FEATURES,
) -> DesignFrame:
    """La matriz de diseno del espejo: mismas features y objetivo ``y = 1{ret_short > 0}`` (#78).

    Es un envoltorio **fino** sobre :func:`cfdtrader.models.baseline.design_frame`: la regla de
    disponibilidad por columna (#147), el corrimiento de una sesion y el conteo de nulos son
    exactamente los de #24. Lo unico que cambia es la columna de objetivo, y por eso el frame
    resultante declara ``target_column = "ret_short"``.
    """
    return design_frame(
        features,
        labels=labels,
        selected=selected,
        availability=availability,
        target_column=MIRROR_TARGET_COLUMN,
    )


def fit_mirror(
    design: DesignFrame,
    *,
    splits: Sequence[SplitAssignment],
    hyperparameters: Mapping[str, object] | None = None,
    seed: int = SEED,
    label_horizon: Sequence[int] | None = None,
) -> BaselineModel:
    """Ajusta el espejo corto con la **misma** maquinaria de #24 (importada, no reimplementada).

    Se exige que ``design`` declare el objetivo corto (:data:`MIRROR_TARGET_COLUMN`); con el
    frame de #24 (``ret_long``) se lanza :class:`MirrorError`, nunca un ajuste silencioso de la
    direccion equivocada. La firma es la de :func:`cfdtrader.models.baseline.fit_baseline`: un
    fold por :class:`SplitAssignment`, escalado y calibrador ajustados **solo** con el train de
    ese fold (sin *look-ahead*).
    """
    if design.target_column != MIRROR_TARGET_COLUMN:
        raise MirrorError(
            f"el espejo corto exige un diseno con objetivo {MIRROR_TARGET_COLUMN!r}, y llego "
            f"{design.target_column!r}: ajustar la direccion larga llamandola corta seria un "
            "espejo falso (A11 de #78)"
        )
    return fit_baseline(
        design,
        splits=splits,
        hyperparameters=MIRROR_HYPERPARAMETERS if hyperparameters is None else hyperparameters,
        seed=seed,
        label_horizon=label_horizon,
    )


def short_signal(probability: float) -> bool:
    """La decision declarada del espejo (A11): ``True`` (corto) si ``p >= DECISION_THRESHOLD``.

    Es la misma regla declarada que ``long_signal`` de #24 —el umbral 0,5, que aqui lee
    ``P(la sesion corta rinde)``— y no se mueve: el umbral economico es #27.
    """
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise BaselineError(
            f"una probabilidad de {probability!r} no es admisible: el decider solo lee una "
            "probabilidad entre 0 y 1"
        )
    return probability >= DECISION_THRESHOLD
