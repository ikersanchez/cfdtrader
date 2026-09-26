"""Veredicto robusto de Fase 2 por dominancia de la base declarada (#93, tarea T29c).

Este modulo es una **puerta de salida** que no toca el motor ni publica mediciones: emite un
veredicto **robusto hacia abajo** sobre la **base declarada** del brazo ``coste_declarado`` de
#28. El argumento es de dominancia y no necesita ni #60 (decidir ``R``) ni #62 (medir el
*slippage*):

- el *slippage* de ejecucion es **>= 0**, asi que la base **neta** esta dominada sesion a sesion
  por la base **declarada** (coste declarado, sin el termino supuesto): es el escenario
  admisible **mas favorable**;
- por tanto, si la base declarada **no** cruza la fila principal, **ningun** escenario admisible
  la cruza. El veredicto se publica como **supuesto** (``is_measurement: false``,
  ``is_validation: false``, ``basis: declared_cost``) y **nunca** como metrica neta.

La fila principal de ``plan.md`` §11.6 es un **o** (el IC de la tasa de acierto **o** el IC del
Sharpe). Como la ``hit_rate`` del artefacto de #28 es por **sesion** y no por operacion (#92), la
decision se sostiene en la **mitad del Sharpe**: la tasa de acierto se **publica**, no decide.

Que **hace**:

- **barre** un escenario declarado de *slippage* (``scenario.slippage_grid_bp``: ``0`` y los tres
  percentiles declarados de #8 —mediana, p90 y maximo de 59 sesiones—) sobre la serie declarada
  del brazo base, en **bp del nocional de ida y vuelta** restados una vez por **sesion operada**
  (``1 bp = 0,01 %``); las sesiones sin operacion quedan en ``0`` exacto y **no** se tocan (A8);
- por cada celda publica ``hit_rate`` y ``sharpe`` (``estimate``/``lower``/``upper``), su
  ``state``, su ``code``, ``sharpe_excludes_zero_above``, ``sharpe_below`` y ``crosses`` (A10);
- **reproduce exactamente** (igualdad de ``float``) la celda de ``0`` bp del artefacto de #28:
  estimacion, ``lower`` y ``upper`` de la tasa de acierto y del Sharpe, con la **misma semilla
  declarada** de #15/#28 (A9). Si la reconstruccion no reproduce el artefacto, el modulo **falla
  con error tipado** en vez de publicar una serie distinta con el mismo nombre;
- **lee** la tabla de §11.6 de ``_docs/plan.md`` en tiempo de ejecucion (con su ``sha256`` y sus
  9 filas), evalua las ocho filas restantes con #29 y **sustituye la fila principal** por la de
  dominancia, y agrega con ``gate_block``/``aggregate_gate`` de #9 y ``resolve_verdict`` (A16);
- **deriva** ``p*`` por escenario declarado de ``R`` (reutilizando #9) y usa el **vinculante**
  (el mas exigente) para la mitad de la tasa de acierto: ningun umbral se cablea (A15);
- publica un ``dominance_check`` que verifica que la media y la tasa de acierto **no crecen** con
  el *slippage*: si crecieran, el escenario estaria mal declarado y se lanza error tipado (A17).

Que **no** hace, y por tanto no puede inventar:

- **no** mide el *slippage*: la rejilla es una **cota superior declarada** derivada de #8, no una
  medicion (medirlo son 10-15 ejecuciones reales → #62);
- **no** decide ``R`` (#60) ni publica metricas netas: ``net_metrics`` sale con el estado que
  declara #28 (``not_computable``) y **nunca** ``computed``;
- **no** convierte un ``not_evaluable`` en un aprobado: el veredicto por dominancia solo puede
  ser ``fail`` o ``not_evaluable``, y ``pass``/``continue`` no se publican nunca (A12);
- **no** sustituye el veredicto real sobre la base neta: eso es #88;
- **no** arregla que la ``hit_rate`` de #28 sea por sesion y no por operacion (#92);
- **no** publica la serie declarada por sesion en el artefacto de #28: mientras siga sin
  publicarse (→ #94), este modulo **re-deriva** el pipeline de #28 con su API publica
  (``analyse(..., write=False)``) y **solo lee** sus artefactos.

**Reloj prohibido** (A2): ninguna ruta consulta el reloj del sistema; el instante entra por
``--as-of``, obligatorio para escribir, que sale con codigo 2 y sin tocar disco cuando falta o no
es ISO-8601. **Red prohibida** y sin escrituras fuera de ``--reports-dir``. Determinista byte a
byte: ``report_sha256`` es el sha256 del texto canonico de #13 sobre el payload sin la clave del
hash, con el prefijo ``sha256:``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final, cast

from loguru import logger

from cfdtrader.analysis import pipeline_report, regeneration_delta
from cfdtrader.analysis.backtest_report import BacktestReportError
from cfdtrader.analysis.cost_audit import slippage_assumption_block
from cfdtrader.analysis.phase0_report import HalfResult
from cfdtrader.analysis.phase2_report import (
    HASH_PREFIX,
    MODEL_CLASS,
    PIPELINE_CLASS,
    InputArtifact,
    InvalidAsOfError,
    MissingAsOfError,
    Phase2ReportError,
    evaluate_criteria,
    gate_block,
    load_input_artifact,
    load_kill_table,
    p_star_block,
    resolve_verdict,
)
from cfdtrader.analysis.pipeline_report import (
    ARM_COSTE_DECLARADO,
    ARM_ESCENARIO,
    ARM_OFICIAL,
    METRIC_NAMES,
    PipelineReportError,
)
from cfdtrader.backtest.engine import (
    STATUS_NO_TRADE,
    STATUS_SKIPPED,
    STATUS_TRADED,
    BacktestRun,
    canonical_text,
)
from cfdtrader.backtest.metrics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_confidence_interval,
    sharpe_ratio,
)
from cfdtrader.data.store import Store

__all__ = [
    "ANALYSIS",
    "BASE_ARM",
    "BASIS_DECLARED_COST",
    "BP_TO_PCT",
    "CELL_STATE_RULE",
    "CODE_BASE_CROSSES",
    "CODE_DEGENERATE",
    "CODE_FAIL",
    "CODE_INCONCLUSIVE",
    "CODE_SCENARIO_CROSSES",
    "DOMINANCE_RULE",
    "HASH_PREFIX",
    "NET_UNMEASURED_RULE",
    "REPORT_HASH_FORMAT",
    "REPORT_PREFIX",
    "SEED_DERIVATION",
    "SHARPE_HALF_RULE",
    "TASK",
    "TITLE",
    "VERDICT_RULE",
    "DeclaredSeries",
    "Dominance",
    "DominanceReport",
    "DominanceViolationError",
    "InvalidSlippageGridError",
    "Phase2DominanceError",
    "ReproductionMismatchError",
    "analyse",
    "compute_dominance",
    "declared_series_of",
    "main",
    "render_markdown",
    "slippage_levels",
]

#: Identidad del informe: quien lo emite y que tarea lo pide.
ANALYSIS: Final[str] = "cfdtrader.analysis.phase2_dominance"
TASK: Final[str] = "#93"
TITLE: Final[str] = "Veredicto robusto de Fase 2 por dominancia de la base declarada"

#: Prefijo del informe: ``phase2_dominance_<AAAA-MM-DD>.{json,md}``.
REPORT_PREFIX: Final[str] = "phase2_dominance"

#: Formato estable del ``report_sha256`` (A4).
REPORT_HASH_FORMAT: Final[str] = (
    "sha256:<64 hex> del texto canonico (``canonical_text`` de #13) del payload **sin** la "
    "clave ``report_sha256``. El prefijo viaja dentro del valor: un digest desnudo lo bloquea "
    "`detect-secrets`"
)

#: La base de todo lo que se publica aqui: coste **declarado**, nunca medido (A5).
BASIS_DECLARED_COST: Final[str] = "declared_cost"

#: El brazo base declarado: el escenario admisible **mas favorable** (A6).
BASE_ARM: Final[str] = ARM_COSTE_DECLARADO

#: Conversion declarada del escenario (A8): ``1 bp = 0,01 %`` del nocional de ida y vuelta.
BP_TO_PCT: Final[float] = 0.01

#: De donde sale la semilla de cada metrica en #28: ``DEFAULT_BOOTSTRAP_SEED`` + su posicion en
#: ``METRIC_NAMES`` (la declarada de #15 mas la posicion) acotada al rango del generador.
SEED_DERIVATION: Final[str] = (
    "cada intervalo usa `bootstrap_confidence_interval` (#15) con la semilla derivada por #28: "
    "`(DEFAULT_BOOTSTRAP_SEED + posicion de la metrica en METRIC_NAMES + 1) % 2**32`, que es "
    "exactamente el mismo flujo del generador declarado"
)

# ─────────────────────────────────────────────────────────────────────────────
# Textos declarados: la regla de dominancia y sus limites
# ─────────────────────────────────────────────────────────────────────────────
#: La regla de dominancia que sostiene todo el informe (A12).
DOMINANCE_RULE: Final[str] = (
    "el *slippage* de ejecucion es >= 0, asi que la base **neta** esta dominada sesion a sesion "
    "por la base **declarada** (el escenario admisible mas favorable); si la base declarada no "
    "cruza la fila principal, ningun escenario admisible la cruza"
)

#: Por que decide el Sharpe y no la tasa de acierto (A11 y #92).
SHARPE_HALF_RULE: Final[str] = (
    "la fila principal es un `o` (IC de la tasa de acierto **o** IC del Sharpe); como la "
    "`hit_rate` del artefacto de #28 es por **sesion** y no por operacion (#92), la decision se "
    "sostiene en la **mitad del Sharpe** y la tasa de acierto solo se **publica**"
)

#: Regla por celda (A11), en una frase publicada.
CELL_STATE_RULE: Final[str] = (
    "por celda: `crosses` sii `sharpe.lower > 0`; `fail` sii `sharpe.upper < 0`; en otro caso "
    "`not_evaluable`. Un intervalo que excluye el 0 **por abajo** es `fail`, **nunca** un `pass`"
)

#: Regla del veredicto (A12, A13).
VERDICT_RULE: Final[str] = (
    "`fail` sii la celda de 0 bp es `fail` **y** ninguna celda cruza; si **alguna** celda cruza, "
    "el veredicto es `not_evaluable`; `pass` y `continue` no se publican nunca"
)

#: Etiquetado obligatorio (A5).
NET_UNMEASURED_RULE: Final[str] = (
    "un supuesto nunca se publica como medicion: este informe lleva `is_measurement: false`, "
    "`is_validation: false` y `basis: declared_cost`, y `phase2_ready` es `false` siempre"
)

#: Codigos del veredicto por dominancia (A12, A13, A14), legibles por maquina.
CODE_DEGENERATE: Final[str] = "declared_base_degenerate"
CODE_BASE_CROSSES: Final[str] = "declared_base_crosses_net_not_measured"
CODE_SCENARIO_CROSSES: Final[str] = "slippage_cell_crosses_within_declared_bound"
CODE_FAIL: Final[str] = "declared_base_fails_within_declared_bound"
CODE_INCONCLUSIVE: Final[str] = "declared_base_inconclusive_within_declared_bound"

#: Codigos por celda: la base declarada y los escenarios con *slippage*.
CODE_CELL_BASE_CROSSES: Final[str] = "declared_base_sharpe_excludes_zero_above"
CODE_CELL_BASE_FAIL: Final[str] = "declared_base_sharpe_below_zero"
CODE_CELL_BASE_FLAT: Final[str] = "declared_base_sharpe_interval_contains_zero"
CODE_CELL_CROSSES: Final[str] = "scenario_sharpe_excludes_zero_above"
CODE_CELL_FAIL: Final[str] = "scenario_sharpe_below_zero"
CODE_CELL_FLAT: Final[str] = "scenario_sharpe_interval_contains_zero"

#: Etiquetas del barrido, en el orden declarado (A7): ``0``, mediana, p90 y maximo.
GRID_LABELS: Final[tuple[tuple[str, str], ...]] = (
    ("sin_slippage", "0"),
    ("mediana", "median_bp"),
    ("p90", "p90_bp"),
    ("maximo", "max_bp"),
)

#: Claves de la evidencia declarada de #8 que dan los tres percentiles de la rejilla.
EVIDENCE_ISSUE: Final[str] = "#8"
EVIDENCE_SOURCE: Final[str] = (
    "cfdtrader.analysis.cost_audit.slippage_assumption_block().measured_evidence (#8, #9)"
)

#: Limites declarados del informe.
REPORT_LIMITATIONS: Final[tuple[str, ...]] = (
    "la rejilla de *slippage* es una **cota superior declarada** sobre 59 sesiones de `^GSPC` "
    "de 5 minutos (mediana, p90 y maximo de #8), **no** una medicion del *slippage* de "
    "ejecucion: medirlo exige 10-15 ejecuciones reales (#62)",
    "la base declarada **no** es la base neta: el *slippage* supuesto no se puede cobrar sin `R` "
    "(#60) y sin medicion (#62), asi que este informe emite un veredicto por **dominancia** y "
    "**no** sustituye al de #88",
    "la `hit_rate` del artefacto de #28 es por **sesion** y no por operacion (#92): decide la "
    "mitad del Sharpe y la tasa de acierto se publica con esa salvedad",
    "el artefacto de #28 **no** publica la serie declarada por sesion, asi que este informe "
    "re-deriva el pipeline con su API publica (`write=False`): publicar esa serie es #94",
)

#: Lo que este informe no hace, con la issue que lo cierra.
REPORT_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_mide_el_slippage",
        "issue": "#62",
        "statement": (
            "no mide el *slippage*: usa una cota superior declarada (los tres percentiles de 59 "
            "sesiones de #8) y la etiqueta `is_measurement: false`"
        ),
    },
    {
        "id": "no_decide_r",
        "issue": "#60",
        "statement": (
            "no decide `R`: publica `p*` por cada escenario declarado y usa el mas exigente, sin "
            "cablear ningun umbral"
        ),
    },
    {
        "id": "no_emite_el_veredicto_de_la_base_neta",
        "issue": "#88",
        "statement": (
            "no sustituye el veredicto sobre la base neta: cuando #60 y #62 cierren, el veredicto "
            "se reemite"
        ),
    },
    {
        "id": "no_arregla_la_hit_rate_por_sesion",
        "issue": "#92",
        "statement": (
            "no convierte la `hit_rate` por sesion del artefacto de #28 en una tasa por "
            "operacion: solo se publica, y la decision va por el Sharpe"
        ),
    },
    {
        "id": "no_publica_la_serie_declarada",
        "issue": "#94",
        "statement": (
            "no publica la serie declarada por sesion en el artefacto de #28: la re-deriva aqui, "
            "en solo lectura, mientras #94 no la publique"
        ),
    },
)

#: Seguimientos declarados.
FOLLOW_UPS: Final[tuple[dict[str, str], ...]] = (
    {
        "issue": "#88",
        "topic": "reemitir el veredicto con la base neta",
        "why": "es el unico veredicto que sustituye a este",
    },
    {
        "issue": "#94",
        "topic": "publicar la serie declarada por sesion en #28",
        "why": "hoy este informe re-deriva el pipeline (minutos por corrida) porque el artefacto "
        "no la publica",
    },
    {
        "issue": "#62",
        "topic": "medir el *slippage*",
        "why": "la rejilla es una cota superior declarada; sin medicion no hay base neta",
    },
    {
        "issue": "#60",
        "topic": "decidir `R`",
        "why": "`R` decide `p*`; aqui se publica por escenarios y se usa el mas exigente",
    },
    {
        "issue": "#92",
        "topic": "tasa de acierto por operacion",
        "why": "la `hit_rate` de #28 es por sesion, asi que la decision va por el Sharpe",
    },
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores declarados: un hueco nunca se rellena con un valor por defecto
# ─────────────────────────────────────────────────────────────────────────────
class Phase2DominanceError(Exception):
    """Error declarado del veredicto por dominancia: nunca se rellena el hueco."""


class InvalidSlippageGridError(Phase2DominanceError):
    """La rejilla declarada de *slippage* no es ascendente o no arranca en ``0`` exacto (A7)."""


class ReproductionMismatchError(Phase2DominanceError):
    """La celda de ``0`` bp no reproduce el artefacto de #28: no se publica una serie distinta."""


class DominanceViolationError(Phase2DominanceError):
    """La media o la tasa de acierto **crecen** con el *slippage*: el escenario esta mal (A17)."""


# ─────────────────────────────────────────────────────────────────────────────
# La serie declarada del brazo base (A6, A8) y la rejilla declarada (A7)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class DeclaredSeries:
    """La serie declarada del brazo base: ``%`` por sesion de *test* y mascara de operadas.

    Una sesion sin operacion **diluye** la serie con un ``0`` exacto y una sesion saltada no
    entra (la convencion de #15/#28). La mascara dice en que sesiones hubo operacion: solo esas
    sesiones pagan el *slippage* del escenario (A8).
    """

    values_pct: tuple[float, ...]
    traded: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.values_pct) != len(self.traded):
            raise Phase2DominanceError(
                "la serie declarada y su mascara de operadas tienen que tener la misma longitud "
                f"({len(self.values_pct)} != {len(self.traded)})"
            )

    @property
    def n_sessions(self) -> int:
        """Sesiones de *test* que entran en la serie."""
        return len(self.values_pct)

    @property
    def n_traded(self) -> int:
        """Sesiones operadas: las unicas que pagan el *slippage* del escenario."""
        return sum(1 for flag in self.traded if flag)

    @property
    def is_degenerate(self) -> bool:
        """Serie degenerada: sin sesiones o con todos los retornos exactamente ``0`` (A14)."""
        return not self.values_pct or all(value == 0.0 for value in self.values_pct)

    def shifted(self, *, slippage_bp: float) -> tuple[float, ...]:
        """La serie con el *slippage* restado una vez por sesion operada (A8).

        ``0`` bp es la **identidad** exacta: no se toca ningun valor, ni las sesiones sin
        operacion (que quedan en ``0`` exacto).
        """
        if slippage_bp == 0.0:
            return self.values_pct
        cost_pct = slippage_bp * BP_TO_PCT
        return tuple(
            value - cost_pct if flag else value
            for value, flag in zip(self.values_pct, self.traded, strict=True)
        )

    def digest(self) -> str:
        """``sha256:`` del texto canonico de la serie y su mascara: la huella de la base."""
        body = canonical_text(
            {
                "values_pct": list(self.values_pct),
                "traded": list(self.traded),
            }
        )
        return HASH_PREFIX + hashlib.sha256(body.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, object]:
        """Resumen publicado de la serie: tamano, operadas y huella, nunca la serie entera."""
        return {
            "n_sessions": self.n_sessions,
            "n_traded": self.n_traded,
            "n_no_trade": self.n_sessions - self.n_traded,
            "all_zero": self.is_degenerate,
            "sha256": self.digest(),
            "sha256_of": "texto canonico de `values_pct` + `traded`",
            "unit": "% del nocional (puntos porcentuales), una entrada por sesion de *test*",
        }


def _evidence_block() -> dict[str, object]:
    """La evidencia declarada de #8 que acota el orden de magnitud del *slippage* (A7)."""
    block = slippage_assumption_block()
    evidence = cast("dict[str, object]", block["measured_evidence"])
    return evidence


def slippage_levels() -> tuple[tuple[str, float], ...]:
    """La rejilla declarada ``(etiqueta, bp)``, leida de #8 y no cableada (A7).

    Primer elemento exactamente ``0`` (la base declarada, sin *slippage*) y despues la mediana,
    el p90 y el maximo de las 59 sesiones que #8 midio: una **cota superior declarada**, jamas
    una medicion del *slippage* de ejecucion.
    """
    evidence = _evidence_block()
    levels: list[tuple[str, float]] = []
    for label, key in GRID_LABELS:
        value = 0.0 if key == "0" else float(cast("float", evidence[key]))
        levels.append((label, value))
    values = [value for _, value in levels]
    if values[0] != 0.0 or any(left >= right for left, right in itertools.pairwise(values)):
        raise InvalidSlippageGridError(
            "la rejilla declarada de *slippage* tiene que arrancar en 0 y ser estrictamente "
            f"ascendente: llego {values} (A7)"
        )
    return tuple(levels)


# ─────────────────────────────────────────────────────────────────────────────
# La semilla declarada de cada metrica (A9): la misma de #28, derivada y no cableada
# ─────────────────────────────────────────────────────────────────────────────
def seed_of(metric: str) -> int:
    """La semilla declarada de esa metrica en #28, derivada de su posicion en #15 (A9)."""
    if metric not in METRIC_NAMES:
        raise Phase2DominanceError(
            f"la metrica {metric!r} no esta en METRIC_NAMES de #28: {list(METRIC_NAMES)} (A9)"
        )
    return (DEFAULT_BOOTSTRAP_SEED + METRIC_NAMES.index(metric) + 1) % (2**32)


def _hit_rate_statistic(sample: Sequence[float]) -> float:
    """La tasa de acierto de una muestra, con la **misma** definicion que #28 (A9)."""
    return sum(1 for value in sample if value > 0.0) / max(len(sample), 1)


def _interval(
    values_pct: Sequence[float],
    *,
    metric: str,
    n_bootstrap: int,
    confidence_level: float,
) -> dict[str, float]:
    """``estimate``/``lower``/``upper`` de una metrica sobre la serie en ``%`` (A9)."""
    decimals = tuple(value / 100.0 for value in values_pct)
    statistic = _hit_rate_statistic if metric == "hit_rate" else sharpe_ratio
    interval = bootstrap_confidence_interval(
        decimals,
        statistic,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed_of(metric),
    )
    return {
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
    }


# ─────────────────────────────────────────────────────────────────────────────
# El calculo puro: rejilla x serie -> celdas -> estado (A10-A14, A17)
# ─────────────────────────────────────────────────────────────────────────────
def _cell_state(*, lower: float, upper: float) -> tuple[str, bool, bool]:
    """``(state, crosses, sharpe_below)`` de un intervalo de Sharpe (A11).

    La mitad del Sharpe decide: un intervalo que excluye el ``0`` **por abajo** es ``fail`` y
    **nunca** un aprobado. El vocabulario de la celda es ``crosses``/``fail``/``not_evaluable``:
    el aprobado de una celda **no** se publica como ``pass`` (A12).
    """
    crosses = lower > 0.0
    below = upper < 0.0
    if crosses:
        return "crosses", True, False
    if below:
        return "fail", False, True
    return "not_evaluable", False, False


def _cell(
    *,
    label: str,
    slippage_bp: float,
    series: DeclaredSeries,
    p_star_fraction: Decimal,
    n_bootstrap: int,
    confidence_level: float,
) -> dict[str, object]:
    """Una celda del barrido: la serie desplazada y sus dos metricas con intervalo (A8-A11)."""
    shifted = series.shifted(slippage_bp=slippage_bp)
    hit = _interval(
        shifted, metric="hit_rate", n_bootstrap=n_bootstrap, confidence_level=confidence_level
    )
    sharpe = _interval(
        shifted, metric="sharpe", n_bootstrap=n_bootstrap, confidence_level=confidence_level
    )
    state, crosses, below = _cell_state(lower=sharpe["lower"], upper=sharpe["upper"])
    is_base = slippage_bp == 0.0
    if crosses:
        code = CODE_CELL_BASE_CROSSES if is_base else CODE_CELL_CROSSES
    elif below:
        code = CODE_CELL_BASE_FAIL if is_base else CODE_CELL_FAIL
    else:
        code = CODE_CELL_BASE_FLAT if is_base else CODE_CELL_FLAT
    hit_excludes = Decimal(repr(hit["lower"])) > p_star_fraction
    return {
        "label": label,
        "slippage_bp": slippage_bp,
        "slippage_pct": slippage_bp * BP_TO_PCT,
        "is_base": is_base,
        "is_measurement": False,
        "n_sessions": series.n_sessions,
        "n_traded_shifted": series.n_traded,
        "state": state,
        "code": code,
        "crosses": crosses,
        "sharpe_excludes_zero_above": crosses,
        "sharpe_below": below,
        "sharpe": dict(sharpe),
        "hit_rate": {
            **dict(hit),
            "p_star_fraction": format(p_star_fraction, "f"),
            "excludes_p_star_above": hit_excludes,
            "role": "publicada, **no** decide: la `hit_rate` de #28 es por sesion (#92)",
        },
        "mean_return_pct": math.fsum(shifted) / len(shifted),
        "seed": {"hit_rate": seed_of("hit_rate"), "sharpe": seed_of("sharpe")},
        "basis": BASIS_DECLARED_COST,
    }


def _dominance_check(cells: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Verifica que la media y la tasa de acierto **no crecen** con el *slippage* (A17)."""
    steps: list[dict[str, object]] = []
    for previous, current in itertools.pairwise(cells):
        delta_mean = float(cast("float", current["mean_return_pct"])) - float(
            cast("float", previous["mean_return_pct"])
        )
        hit_previous = cast("Mapping[str, object]", previous["hit_rate"])
        hit_current = cast("Mapping[str, object]", current["hit_rate"])
        delta_hit = float(cast("float", hit_current["estimate"])) - float(
            cast("float", hit_previous["estimate"])
        )
        step = {
            "from_bp": previous["slippage_bp"],
            "to_bp": current["slippage_bp"],
            "delta_mean_return_pct": delta_mean,
            "delta_hit_rate": delta_hit,
        }
        steps.append(step)
        if delta_mean > 0.0 or delta_hit > 0.0:
            raise DominanceViolationError(
                "la media o la tasa de acierto **crecen** con el *slippage* "
                f"({step['from_bp']} bp -> {step['to_bp']} bp: media {delta_mean:+.6f} pp, tasa de "
                f"acierto {delta_hit:+.6f}): un *slippage* es un coste y no puede mejorar la "
                "serie, asi que el escenario estaria mal declarado (A17)"
            )
    return {
        "mean_non_increasing": True,
        "hit_rate_non_increasing": True,
        "violated": False,
        "n_steps": len(steps),
        "steps": steps,
        "rule": (
            "la media y la tasa de acierto **no** pueden crecer con el *slippage*: es un coste "
            "(se resta una vez por sesion operada) y un barrido que las mejore estaria mal "
            "declarado"
        ),
        "error_type": "DominanceViolationError",
    }


@dataclass(frozen=True, slots=True)
class Dominance:
    """Resultado **puro** del barrido: celdas, veredicto por dominancia y comprobacion (A10-A17)."""

    cells: tuple[dict[str, object], ...]
    crossed_cells: tuple[float, ...]
    no_cell_crosses: bool | None
    state: str
    code: str
    reason: str
    check: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        """El bloque publicado del veredicto por dominancia."""
        return {
            "state": self.state,
            "code": self.code,
            "reason": self.reason,
            "decided_by": "mitad del Sharpe",
            "sharpe_half_rule": SHARPE_HALF_RULE,
            "dominance_rule": DOMINANCE_RULE,
            "verdict_rule": VERDICT_RULE,
            "cell_state_rule": CELL_STATE_RULE,
            "crossed_cells": list(self.crossed_cells),
            "no_cell_crosses": self.no_cell_crosses,
            "n_cells": len(self.cells),
            "note": (
                "el veredicto por dominancia solo puede ser `fail` o `not_evaluable`: `pass` y "
                "`continue` no se publican nunca (A12)"
            ),
        }


def compute_dominance(
    series: DeclaredSeries,
    *,
    levels: Sequence[tuple[str, float]],
    p_star_fraction: Decimal,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> Dominance:
    """Rejilla x serie -> celdas -> estado, como **funcion pura** (A17).

    No toca disco, ni red, ni reloj, y no reejecuta el pipeline: entra la serie declarada y sale
    el veredicto. Una serie degenerada (sin sesiones o todo ceros) sale ``not_evaluable`` con
    ``code: declared_base_degenerate`` y **sin** numeros fabricados (A14).
    """
    if series.is_degenerate:
        return Dominance(
            cells=(),
            crossed_cells=(),
            no_cell_crosses=None,
            state=str(HalfResult.NOT_EVALUABLE),
            code=CODE_DEGENERATE,
            reason=(
                "la serie declarada de la base es degenerada (sin sesiones o con todos los "
                "retornos exactamente 0): no hay celda que calcular y **no** se fabrica ninguna "
                "metrica (A14)"
            ),
            check={
                "mean_non_increasing": None,
                "hit_rate_non_increasing": None,
                "violated": None,
                "n_steps": 0,
                "steps": [],
                "rule": "sin celdas no hay monotonias que comprobar: se declara, no se rellena",
                "error_type": "DominanceViolationError",
            },
        )

    cells = tuple(
        _cell(
            label=label,
            slippage_bp=slippage_bp,
            series=series,
            p_star_fraction=p_star_fraction,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )
        for label, slippage_bp in levels
    )
    check = _dominance_check(cells)
    crossed = tuple(
        float(cast("float", cell["slippage_bp"])) for cell in cells if bool(cell["crosses"])
    )
    base = cells[0]
    if crossed:
        if bool(base["crosses"]):
            code = CODE_BASE_CROSSES
            reason = (
                "la **propia base declarada** cruza la fila principal: eso **no** es aprobar "
                "nada, porque la base declarada no demuestra la base neta, que no esta medida "
                "(A13)"
            )
        else:
            code = CODE_SCENARIO_CROSSES
            reason = (
                "alguna celda del barrido cruza la fila principal: con el *slippage* como coste "
                "seria un resultado a favor, pero sigue siendo un escenario **declarado** y "
                "**no** una medicion de la base neta (A12)"
            )
        state = str(HalfResult.NOT_EVALUABLE)
    elif str(base["state"]) == str(HalfResult.FAIL):
        code = CODE_FAIL
        reason = (
            "la celda de 0 bp **falla** (su IC del Sharpe excluye el 0 por abajo) y ninguna "
            "celda del barrido cruza: el *slippage* solo puede empeorar la serie, asi que "
            "ningun escenario admisible cruza la fila principal (A12)"
        )
        state = str(HalfResult.FAIL)
    else:
        code = CODE_INCONCLUSIVE
        reason = (
            "la celda de 0 bp no falla ni cruza (su IC del Sharpe contiene el 0) y ninguna celda "
            "del barrido cruza: no hay evidencia para fallar por dominancia ni para cruzarla "
            "(A12)"
        )
        state = str(HalfResult.NOT_EVALUABLE)
    return Dominance(
        cells=cells,
        crossed_cells=crossed,
        no_cell_crosses=not crossed,
        state=state,
        code=code,
        reason=reason,
        check=check,
    )


# ─────────────────────────────────────────────────────────────────────────────
# La serie declarada, re-derivada de #28 **en solo lectura** (A3, A6)
# ─────────────────────────────────────────────────────────────────────────────
def declared_series_of(run: BacktestRun) -> DeclaredSeries:
    """La serie declarada de una corrida del motor, en ``%`` y por sesion de *test* (A8).

    ``100 x gross_pct - c_declared_pct``: ``gross_pct`` llega como **fraccion** (la unidad que
    declara #80) y ``c_declared_pct`` en **%**. El modulo **no** lee el P&L declarado del motor:
    lo re-deriva aqui, igual que #28. Las sesiones sin operacion entran como ``0`` exacto y las
    saltadas no entran.
    """
    values: list[float] = []
    traded: list[bool] = []
    for fold in run.folds:
        for outcome in fold.sessions:
            if outcome.status == STATUS_SKIPPED:
                continue
            if outcome.status == STATUS_NO_TRADE:
                values.append(0.0)
                traded.append(False)
                continue
            if outcome.gross_pct is None or outcome.cost is None:
                raise Phase2DominanceError(
                    f"{outcome.session.isoformat()}: una operacion sin `gross_pct` o sin "
                    "`CostBreakdown` no tiene retorno declarado que reconstruir (A8)"
                )
            values.append(100.0 * outcome.gross_pct - float(outcome.cost.c_declared_pct))
            traded.append(outcome.status == STATUS_TRADED)
    return DeclaredSeries(values_pct=tuple(values), traded=tuple(traded))


def derive_declared_series(*, store: Store, reports_dir: Path, as_of: datetime) -> DeclaredSeries:
    """Re-deriva el pipeline de #28 con su API publica y devuelve la serie del brazo base (A3).

    Se reejecuta el pipeline porque el artefacto de #28 **no** publica la serie por sesion
    (→ #94). La corrida es ``write=False``: consume el almacen y los artefactos **en solo
    lectura** y no deja ningun fichero nuevo.
    """
    report = pipeline_report.analyse(store=store, reports_dir=reports_dir, as_of=as_of, write=False)
    arm = report.arm(ARM_COSTE_DECLARADO)
    return declared_series_of(arm.run)


# ─────────────────────────────────────────────────────────────────────────────
# Lectura tipada del artefacto de #28 (el brazo base y sus recuentos) (A6, A9)
# ─────────────────────────────────────────────────────────────────────────────
def _mapping(node: object, *, where: str) -> dict[str, object]:
    """Vista tipada de un nodo que tiene que ser un objeto JSON."""
    if not isinstance(node, dict):
        raise Phase2DominanceError(
            f"{where}: se espera un objeto JSON, no {type(node).__name__} (A6)"
        )
    return cast("dict[str, object]", node)


def _number(node: Mapping[str, object], key: str, *, where: str) -> float:
    """Campo numerico obligatorio del artefacto; su ausencia es un error tipado (A6)."""
    value = node.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase2DominanceError(
            f"`{where}.{key}` deberia ser un numero, no {type(value).__name__}: no se rellena con "
            "un valor por defecto (A6)"
        )
    return float(value)


def _arm_counts(payload: Mapping[str, object], arm: str) -> dict[str, object]:
    """Los recuentos **leidos** del artefacto de #28 para ese brazo (A6)."""
    arms = _mapping(payload.get("arms"), where="arms")
    block = _mapping(arms.get(arm), where=f"arms.{arm}")
    return {
        "traded": int(_number(block, "traded", where=f"arms.{arm}")),
        "no_trade": int(_number(block, "no_trade", where=f"arms.{arm}")),
        "skipped": int(_number(block, "skipped", where=f"arms.{arm}")),
        "n_test": int(_number(block, "n_test", where=f"arms.{arm}")),
        "declared_return_series_all_zero": block.get("declared_return_series_all_zero"),
    }


def _base_arm_block(
    *,
    counts: Mapping[str, object],
    others: Mapping[str, Mapping[str, object]],
    series: DeclaredSeries,
) -> dict[str, object]:
    """El brazo base declarado, con sus recuentos leidos del artefacto (A6)."""
    return {
        "name": BASE_ARM,
        "basis": BASIS_DECLARED_COST,
        "is_measurement": False,
        "is_validation": False,
        "evidence": "artifact",
        "reason": DOMINANCE_RULE,
        "counts": dict(counts),
        "counts_source": (
            f"artefacto de #28, `arms.{BASE_ARM}.traded/no_trade/skipped/n_test`: los recuentos "
            "se **leen**, no se recalculan"
        ),
        "other_arms": {name: dict(block) for name, block in others.items()},
        "other_arms_note": (
            "los otros dos brazos del gate no operan ninguna sesion (0 / 500): sin operaciones no "
            "hay serie sobre la que decidir, y por eso la base declarada es la unica base "
            "admisible hoy"
        ),
        "rederived_counts": {
            "n_sessions": series.n_sessions,
            "n_traded": series.n_traded,
            "matches_artifact": series.n_traded == counts["traded"]
            and series.n_sessions - series.n_traded == counts["no_trade"],
        },
    }


def _reproduction_block(
    *,
    payload: Mapping[str, object],
    artifact: InputArtifact,
    cells: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Comprueba que la celda de ``0`` bp reproduce el artefacto de #28 (A9).

    Si no lo reproduce, se lanza error tipado: publicar una serie reconstruida que no es la del
    artefacto seria publicar otra cosa con el mismo nombre.
    """
    arms = _mapping(payload.get("arms"), where="arms")
    base = _mapping(arms.get(BASE_ARM), where=f"arms.{BASE_ARM}")
    metrics = _mapping(base.get("metrics"), where=f"arms.{BASE_ARM}.metrics")
    if not cells:
        return {
            "checked": False,
            "reason": (
                "la base declarada es degenerada: no hay celda de 0 bp que comparar con el "
                "artefacto (A9, A14)"
            ),
            "artifact_sha256": HASH_PREFIX + artifact.sha256,
        }
    cell = cells[0]
    block: dict[str, object] = {
        "checked": True,
        "source": f"artefacto de #28, `arms.{BASE_ARM}.metrics`",
        "artifact_sha256": HASH_PREFIX + artifact.sha256,
        "rule": (
            "la celda de 0 bp reproduce **exactamente** (igualdad de `float`) la estimacion, el "
            "`lower` y el `upper` de la tasa de acierto y del Sharpe del artefacto, con la "
            "**misma semilla declarada** de #28 (A9)"
        ),
        "metrics": {},
        "seeds": {},
        "reproduces": True,
    }
    compared = cast("dict[str, object]", block["metrics"])
    seeds = cast("dict[str, object]", block["seeds"])
    for metric in ("hit_rate", "sharpe"):
        published = _mapping(metrics.get(metric), where=f"arms.{BASE_ARM}.metrics.{metric}")
        ours = cast("Mapping[str, object]", cell[metric])
        seeds[metric] = {"artifact": published.get("seed"), "here": seed_of(metric)}
        deltas: dict[str, object] = {}
        matches = True
        for key in ("estimate", "lower", "upper"):
            expected = _number(published, key, where=f"arms.{BASE_ARM}.metrics.{metric}")
            found = float(cast("float", ours[key]))
            delta = found - expected
            deltas[key] = {"artifact": expected, "here": found, "delta": delta}
            matches = matches and delta == 0.0
        seed_matches = published.get("seed") == seed_of(metric)
        compared[metric] = {**deltas, "matches": matches and seed_matches}
        block["reproduces"] = bool(block["reproduces"]) and matches and seed_matches
    if not block["reproduces"]:
        raise ReproductionMismatchError(
            "la celda de 0 bp no reproduce el artefacto de #28 (metrica, intervalo o semilla): "
            "la serie reconstruida no es la del artefacto y no se publica como si lo fuera (A9). "
            f"Comparacion: {json.dumps(compared, sort_keys=True, ensure_ascii=False)}"
        )
    return block


# ─────────────────────────────────────────────────────────────────────────────
# El informe: payload canonico, hash y escritura (A4)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class DominanceReport:
    """El informe: payload canonico, hash y los objetos que lo produjeron."""

    as_of: datetime
    report_date: str
    payload: dict[str, object]
    report_sha256: str
    reports_dir: Path
    series: DeclaredSeries
    pipeline: InputArtifact
    model: InputArtifact

    @property
    def report_stem(self) -> str:
        """Nombre base del informe: ``phase2_dominance_<AAAA-MM-DD>``."""
        return f"{REPORT_PREFIX}_{self.report_date}"

    def json_text(self) -> str:
        """JSON determinista: mismo ``as_of`` ⇒ mismo texto byte a byte (A4)."""
        published: dict[str, object] = {**self.payload, "report_sha256": self.report_sha256}
        return json.dumps(published, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    def write(self, directory: Path) -> tuple[Path, Path]:
        """Escribe el informe en JSON y Markdown dentro de ``directory`` (A1)."""
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{self.report_stem}.json"
        markdown_path = directory / f"{self.report_stem}.md"
        json_path.write_text(self.json_text(), encoding="utf-8")
        markdown_path.write_text(render_markdown(self), encoding="utf-8")
        return json_path, markdown_path


def _digest(payload: Mapping[str, object]) -> str:
    """``report_sha256``: el sha256 del texto canonico de #13, con prefijo (A4)."""
    return HASH_PREFIX + hashlib.sha256(canonical_text(payload).encode("utf-8")).hexdigest()


def _as_utc(value: datetime) -> datetime:
    """Normaliza el instante declarado a UTC; sin zona se interpreta como UTC (A2)."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _scenario_block(
    *, levels: Sequence[tuple[str, float]], series: DeclaredSeries, binding: Mapping[str, object]
) -> dict[str, object]:
    """El escenario declarado: rejilla de *slippage*, unidades y bootstrap (A4, A7, A8, A15)."""
    evidence = _evidence_block()
    return {
        "id": "dominancia_base_declarada",
        "title": "Barrido declarado de *slippage* sobre la base declarada",
        "basis": BASIS_DECLARED_COST,
        "is_measurement": False,
        "is_validation": False,
        "slippage_grid_bp": [value for _, value in levels],
        "slippage_grid_labels": [label for label, _ in levels],
        "slippage_grid_ascending": True,
        "grid_provenance": {
            "issue": EVIDENCE_ISSUE,
            "source": EVIDENCE_SOURCE,
            "measured_on": evidence.get("measured_on"),
            "sample": evidence.get("sample"),
            "sessions_measured": evidence.get("sessions_measured"),
            "median_bp": evidence.get("median_bp"),
            "p90_bp": evidence.get("p90_bp"),
            "max_bp": evidence.get("max_bp"),
            "is_measurement": False,
            "why_not_a_measurement": (
                "la rejilla son la mediana, el p90 y el **maximo** de las 59 sesiones que #8 "
                "midio sobre el movimiento de la apertura: es una **cota superior declarada** del "
                "coste, **no** una medicion del *slippage* de ejecucion (#62). El escenario solo "
                "se publica como supuesto (`is_measurement: false`)"
            ),
            "not_part_of_the_assumption": evidence.get("not_part_of_the_assumption"),
        },
        "units": {
            "slippage_unit": "bp del nocional de ida y vuelta",
            "conversion": "1 bp = 0,01 % del nocional",
            "bp_to_pct": BP_TO_PCT,
            "applied": (
                "se resta una vez por **sesion operada** sobre la serie declarada en `%`; las "
                "sesiones sin operacion quedan en `0` exacto y **no** se tocan"
            ),
            "sign": "positivo = coste: cada bp se **resta** de la serie declarada",
            "series_unit": (
                "% del nocional (puntos porcentuales), una entrada por sesion de *test*"
            ),
            "n_sessions": series.n_sessions,
            "n_traded": series.n_traded,
            "n_no_trade": series.n_sessions - series.n_traded,
            "first_cell_is_the_identity": (
                "la celda de 0 bp **no** toca la serie: es la base declarada tal cual, y por eso "
                "puede reproducirse contra el artefacto de #28 (A9)"
            ),
        },
        "bootstrap": {
            "n_bootstrap": DEFAULT_BOOTSTRAP_SAMPLES,
            "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
            "seed": DEFAULT_BOOTSTRAP_SEED,
            "seeds_by_metric": {
                "hit_rate": seed_of("hit_rate"),
                "sharpe": seed_of("sharpe"),
            },
            "seed_source": "cfdtrader.backtest.metrics.DEFAULT_BOOTSTRAP_SEED (#15)",
            "derivation": SEED_DERIVATION,
            "source": "cfdtrader.backtest.metrics.bootstrap_confidence_interval (#15)",
            "note": (
                "`seed` es la declarada de #15 y `seeds_by_metric` las derivadas como las deriva "
                "#28 para cada metrica: son las que reproducen el artefacto (A9)"
            ),
        },
        "p_star_binding": dict(binding),
        "p_star_role": (
            "el `p*` **vinculante** (el escenario de `R` mas exigente) es el que se usa en la "
            "mitad de la tasa de acierto; no decide el veredicto (eso es la mitad del Sharpe)"
        ),
        "decision": "mitad del Sharpe",
        "sharpe_half_rule": SHARPE_HALF_RULE,
    }


def _main_row(
    *, row: Mapping[str, object], dominance: Dominance, payload_p_star: Mapping[str, object]
) -> dict[str, object]:
    """La fila principal de §11.6 **sustituida** por la de dominancia (A16).

    Se conservan los literales del documento (``source_row``, ``threshold``, ``action``) y se
    cambia la evaluacion: el estado sale de la dominancia, nunca del criterio estadistico de #29.
    """
    detail: dict[str, object] = {
        "basis": BASIS_DECLARED_COST,
        "is_measurement": False,
        "decided_by": "mitad del Sharpe",
        "dominance_state": dominance.state,
        "dominance_code": dominance.code,
        "crossed_cells": list(dominance.crossed_cells),
        "no_cell_crosses": dominance.no_cell_crosses,
        "n_cells": len(dominance.cells),
        "p_star_binding": dict(payload_p_star),
        "hit_rate_role": "publicada, no decide",
        "dominance_rule": DOMINANCE_RULE,
    }
    return {
        **row,
        "state": dominance.state,
        "code": dominance.code,
        "basis": BASIS_DECLARED_COST,
        "detail": detail,
        "evaluated_by": "dominancia de la base declarada (#93)",
        "note": dominance.reason,
    }


def _provenance(artifact: InputArtifact) -> dict[str, object]:
    """Procedencia del artefacto consumido, **sin** rutas que dependan del ``--reports-dir``.

    Se publica el **nombre** del fichero en vez de su ruta: la ruta (relativa al almacen o
    absoluta) cambia segun donde viva el directorio de informes de cada corrida, y el informe
    tiene que ser determinista **byte a byte** con el mismo ``--as-of`` (A4). La identidad del
    artefacto la dan su ``sha256``, su ``report_sha256`` y su ``generated_at``.
    """
    block = artifact.provenance()
    block["path"] = artifact.path.name
    block["path_note"] = (
        "se publica el **nombre** del fichero: la ruta de `--reports-dir` cambia entre corridas "
        "y el informe es determinista byte a byte (A4)"
    )
    return block


def _net_metrics_block(payload: Mapping[str, object]) -> dict[str, object]:
    """El estado declarado de las metricas netas: se **copia** de #28 y nunca es ``computed``."""
    source = _mapping(payload.get("net_metrics"), where="net_metrics")
    state = str(source.get("state", "absent"))
    return {
        "state": state,
        "basis": "net",
        "computed_here": False,
        "source": f"artefacto de #28, bloque `net_metrics` (estado `{state}`)",
        "reason": (
            "este informe **no** publica metricas netas: el P&L neto del motor es `null` en todas "
            "las operaciones mientras el *slippage* sea un supuesto (#62) y `R` siga sin decidir "
            "(#60). El estado se copia del artefacto y **nunca** se declara `computed`"
        ),
        "follow_ups": ["#62", "#60"],
        "note": (
            "el escenario de *slippage* de este informe vive en `scenario` y **no** es el P&L "
            "neto del motor: es una cota superior declarada sobre la base declarada"
        ),
    }


def _grid_bp(payload: Mapping[str, object]) -> list[object]:
    """La rejilla de *slippage* en bp publicada en `cells`, en su orden."""
    cells = payload.get("cells")
    if not isinstance(cells, list):
        return []
    out: list[object] = []
    for cell in cast("list[object]", cells):
        if isinstance(cell, dict):
            out.append(cast("dict[str, object]", cell).get("slippage_bp"))
    return out


def _regeneration_block(
    previous: Mapping[str, object], payload: Mapping[str, object], *, name: str
) -> dict[str, object]:
    """La diferencia declarada frente al artefacto previo de este informe (#90).

    Este informe **no** publica P&L declarado: lo unico que cambia al regenerar es el puntero
    al artefacto de #28 consumido. La rejilla y el veredicto se recalculan y tienen que salir
    iguales; el bloque lo declara con banderas **medidas**, no supuestas.
    """
    pointers = [
        regeneration_delta.pointer_delta(
            previous, payload, path=("provenance", "pipeline", "sha256")
        ),
        regeneration_delta.pointer_delta(
            previous, payload, path=("provenance", "pipeline", "report_sha256")
        ),
        regeneration_delta.pointer_delta(
            previous, payload, path=("provenance", "model_comparison", "sha256")
        ),
    ]
    previous_gate = previous.get("gate")
    gate = payload.get("gate")
    gate_aggregate = bool(
        isinstance(previous_gate, dict)
        and isinstance(gate, dict)
        and cast("dict[str, object]", previous_gate).get("aggregate")
        == cast("dict[str, object]", gate).get("aggregate")
    )
    unchanged = {
        "slippage_grid_bp": _grid_bp(previous) == _grid_bp(payload),
        "no_cell_crosses": previous.get("no_cell_crosses") == payload.get("no_cell_crosses"),
        "phase2_ready": previous.get("phase2_ready") == payload.get("phase2_ready"),
        "gate_aggregate": gate_aggregate,
    }
    return regeneration_delta.pointer_deltas_block(
        previous_name=name, pointers=pointers, unchanged=unchanged
    )


def analyse(
    *,
    store: Store,
    reports_dir: Path,
    as_of: datetime,
    write: bool = True,
    previous_artifact: Path | None = None,
) -> DominanceReport:
    """Emite el veredicto por dominancia de la base declarada (A1, A3, A9-A17).

    ``as_of`` es **obligatorio** y es el unico instante de la corrida: ninguna ruta consulta el
    reloj. Los artefactos de #28/#26 se consumen **en solo lectura**; ``write=False`` no escribe
    **nada**. ``previous_artifact`` es la ruta del informe anterior: cuando se declara, el
    payload publica el bloque `regeneration` con el delta de puntero y las banderas de
    invariantes (#90).
    """
    moment = _as_utc(as_of)
    previous = regeneration_delta.load_previous(previous_artifact)
    previous_name = regeneration_delta.artifact_name(previous_artifact)
    table = load_kill_table()
    pipeline = load_input_artifact(reports_dir, PIPELINE_CLASS, store=store)
    model = load_input_artifact(reports_dir, MODEL_CLASS, store=store)
    scenario_payload = _mapping(pipeline.payload.get("scenario"), where="scenario")
    # Fallar rapido: los recuentos que A6 publica se **leen** del artefacto antes de reejecutar
    # el pipeline, que cuesta minutos. Un artefacto mal formado sale como error tipado sin
    # gastar la corrida (A3, A6).
    counts = _arm_counts(pipeline.payload, BASE_ARM)
    others = {name: _arm_counts(pipeline.payload, name) for name in (ARM_OFICIAL, ARM_ESCENARIO)}

    cost_pct = Decimal(str(scenario_payload.get("cost_basis_pct")))
    cost_source = str(scenario_payload.get("cost_provenance"))
    stars = p_star_block(cost_pct=cost_pct, cost_source=cost_source)
    binding = _mapping(stars["binding"], where="p_star.binding")
    p_star_fraction = Decimal(str(binding["p_star_fraction"]))

    levels = slippage_levels()
    series = derive_declared_series(store=store, reports_dir=reports_dir, as_of=moment)
    dominance = compute_dominance(series, levels=levels, p_star_fraction=p_star_fraction)
    reproduction = _reproduction_block(
        payload=pipeline.payload, artifact=pipeline, cells=dominance.cells
    )
    base_arm = _base_arm_block(counts=counts, others=others, series=series)

    criteria = evaluate_criteria(
        pipeline=pipeline.payload, model=model.payload, table=table, stars=stars
    )
    rows = [
        _main_row(row=criteria[0], dominance=dominance, payload_p_star=binding),
        *criteria[1:],
    ]
    gate = gate_block(rows)
    verdict = resolve_verdict(aggregate=str(gate["aggregate"]), criteria=rows)

    payload: dict[str, object] = {
        "analysis": ANALYSIS,
        "task": TASK,
        "title": TITLE,
        "generated_at": moment.isoformat(),
        "report_date": moment.date().isoformat(),
        "hash_format": REPORT_HASH_FORMAT,
        "basis": BASIS_DECLARED_COST,
        "is_measurement": False,
        "is_validation": False,
        "evidence": "artifact",
        "phase2_ready": False,
        "phase2_ready_rule": (
            "`phase2_ready` es `false` **siempre**: el veredicto por dominancia no aprueba la "
            "puerta de Fase 2 y el veredicto sobre la base neta es #88"
        ),
        "net_unmeasured_rule": NET_UNMEASURED_RULE,
        "dominance_rule": DOMINANCE_RULE,
        "sharpe_half_rule": SHARPE_HALF_RULE,
        "verdict_rule": VERDICT_RULE,
        "cell_state_rule": CELL_STATE_RULE,
        "base_arm": base_arm,
        "provenance": {
            "pipeline": _provenance(pipeline),
            "model_comparison": _provenance(model),
        },
        "criteria_source": dict(table.source),
        "kill_criteria": [row.as_dict() for row in table.rows],
        "p_star": stars,
        "scenario": _scenario_block(levels=levels, series=series, binding=binding),
        "series": series.as_dict(),
        "cells": [dict(cell) for cell in dominance.cells],
        "crossed_cells": list(dominance.crossed_cells),
        "no_cell_crosses": dominance.no_cell_crosses,
        "dominance": dominance.as_dict(),
        "dominance_check": dominance.check,
        "reproduction": reproduction,
        "criteria": rows,
        "gate": gate,
        "verdict": verdict,
        "net_metrics": _net_metrics_block(pipeline.payload),
        "limitations": list(REPORT_LIMITATIONS),
        "does_not_do": [dict(entry) for entry in REPORT_DOES_NOT_DO],
        "follow_ups": [dict(entry) for entry in FOLLOW_UPS],
    }
    if previous is not None and previous_name is not None:
        payload["regeneration"] = _regeneration_block(previous, payload, name=previous_name)
    report = DominanceReport(
        as_of=moment,
        report_date=moment.date().isoformat(),
        payload=payload,
        report_sha256=_digest(payload),
        reports_dir=reports_dir,
        series=series,
        pipeline=pipeline,
        model=model,
    )
    if write:
        json_path, markdown_path = report.write(reports_dir)
        logger.info("veredicto por dominancia: {} y {}", json_path, markdown_path)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Informe en prosa (A1)
# ─────────────────────────────────────────────────────────────────────────────
def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Tabla Markdown determinista."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_markdown(report: DominanceReport) -> str:
    """El informe en Markdown: la rejilla, el veredicto y sus limites (A1)."""
    payload = report.payload
    scenario = _mapping(payload["scenario"], where="scenario")
    dominance = _mapping(payload["dominance"], where="dominance")
    gate = _mapping(payload["gate"], where="gate")
    verdict = _mapping(payload["verdict"], where="verdict")
    base_arm = _mapping(payload["base_arm"], where="base_arm")
    counts = _mapping(base_arm["counts"], where="base_arm.counts")
    provenance = _mapping(payload["provenance"], where="provenance")
    binding = _mapping(scenario["p_star_binding"], where="scenario.p_star_binding")
    grid = cast("list[object]", scenario["slippage_grid_bp"])
    grid_provenance = _mapping(scenario["grid_provenance"], where="scenario.grid_provenance")
    units = _mapping(scenario["units"], where="scenario.units")
    bootstrap = _mapping(scenario["bootstrap"], where="scenario.bootstrap")
    check = _mapping(payload["dominance_check"], where="dominance_check")

    def cell_row(item: object) -> list[str]:
        """La fila de una celda del barrido, con su intervalo de Sharpe."""
        cell = _mapping(item, where="cells")
        sharpe = _mapping(cell["sharpe"], where="cells.sharpe")
        lower = float(cast("float", sharpe["lower"]))
        upper = float(cast("float", sharpe["upper"]))
        return [
            str(cell["slippage_bp"]),
            str(cell["label"]),
            f"`{cell['state']}`",
            f"`{cell['code']}`",
            f"{lower:.6f}",
            f"{upper:.6f}",
        ]

    lines: list[str] = [
        "# Veredicto robusto de Fase 2 por dominancia de la base declarada",
        "",
        f"- **Tarea**: {payload['task']}",
        f"- **Instante declarado (`as_of`)**: `{payload['generated_at']}`",
        f"- **`report_sha256`**: `{report.report_sha256}`",
        f"- **Base declarada**: `{base_arm['name']}` "
        f"(`traded` {counts['traded']} / `no_trade` {counts['no_trade']})",
        f"- **`phase2_ready`**: `{str(payload['phase2_ready']).lower()}` · "
        f"**agregado**: `{gate['aggregate']}` (`n_rows` = {gate['n_rows']}) · "
        f"**veredicto**: `{verdict['state']}`",
        "",
        f"> **Supuesto, no medicion.** `is_measurement: false`, `is_validation: false`, "
        f"`basis: {payload['basis']}`. {payload['dominance_rule']}",
        "",
        "## Rejilla declarada de *slippage*",
        "",
    ]
    lines += _table(
        ["bp", "etiqueta", "celda `state`", "`code`", "Sharpe `lower`", "Sharpe `upper`"],
        [cell_row(item) for item in cast("list[object]", payload["cells"])],
    )
    lines += [
        "",
        f"Rejilla: {grid} bp ({grid_provenance['issue']}, {grid_provenance['sample']})",
        "",
        f"> {grid_provenance['why_not_a_measurement']}",
        "",
        f"- Unidades: {units['applied']} (`{units['conversion']}`)",
        f"- Bootstrap: `n_bootstrap` = {bootstrap['n_bootstrap']}, "
        f"`confidence_level` = {bootstrap['confidence_level']}, `seed` = {bootstrap['seed']}",
        f"- `p*` vinculante: `R` = {binding['r_pct']} % ⇒ `p*` = {binding['p_star_pct']} % "
        f"({scenario['p_star_role']})",
        "",
        "## Veredicto por dominancia",
        "",
        f"- **Estado**: `{dominance['state']}` · **`code`**: `{dominance['code']}`",
        f"- Celdas que cruzan: {dominance['crossed_cells']} · "
        f"`no_cell_crosses`: `{dominance['no_cell_crosses']}`",
        f"- Por que: {dominance['reason']}",
        f"- Comprobacion: media no creciente `{check['mean_non_increasing']}`, tasa de acierto "
        f"no creciente `{check['hit_rate_non_increasing']}`",
        "",
        "## Procedencia",
        "",
    ]
    for key in ("pipeline", "model_comparison"):
        block = _mapping(provenance[key], where=key)
        lines.append(
            f"- **{key}**: `{block['path']}` (sha256 `{block['sha256']}`, `report_sha256` "
            f"`{block['report_sha256']}`, `generated_at` `{block['generated_at']}`)"
        )
    lines += [
        f"- **Serie declarada**: `{report.series.digest()}` "
        f"({report.series.n_sessions} sesiones, {report.series.n_traded} operadas) — re-derivada "
        "del pipeline de #28 en solo lectura",
        "",
        "## Limites declarados",
        "",
    ]
    lines += [f"- {item}" for item in cast("list[str]", payload["limitations"])]
    lines += ["", "## Que no hace este informe", ""]
    lines += [
        f"- **{_mapping(entry, where='does_not_do')['id']}** "
        f"({_mapping(entry, where='does_not_do')['issue']}): "
        f"{_mapping(entry, where='does_not_do')['statement']}"
        for entry in cast("list[object]", payload["does_not_do"])
    ]
    lines += ["", "## Seguimientos", ""]
    lines += _table(
        ["issue", "tema", "motivo"],
        [
            [
                f"`{_mapping(entry, where='follow_ups')['issue']}`",
                str(_mapping(entry, where="follow_ups")["topic"]),
                str(_mapping(entry, where="follow_ups")["why"]),
            ]
            for entry in cast("list[object]", payload["follow_ups"])
        ],
    )
    regeneration = payload.get("regeneration")
    if isinstance(regeneration, dict):
        lines += [""]
        lines += regeneration_delta.render_pointer_section(
            cast("Mapping[str, object]", regeneration),
            intro=(
                "Este informe se ha **regenerado**: cambia el puntero al artefacto de #28 "
                "consumido y se declara el delta, junto con las invariantes que **no** cambian."
            ),
            unchanged_labels={
                "slippage_grid_bp": "La rejilla de *slippage* (0.0, 13.2, 26.6, 46.8 bp)",
                "no_cell_crosses": "`no_cell_crosses`",
                "phase2_ready": "`phase2_ready`",
                "gate_aggregate": "El veredicto agregado de la puerta",
            },
        )
    lines += ["", ""]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_as_of(value: str | None) -> datetime:
    """El instante declarado de la CLI: **obligatorio**, y nunca tomado del reloj (A2)."""
    if value is None:
        raise MissingAsOfError(
            "falta --as-of: escribir el informe exige un instante declarado (el modulo no lee el "
            "reloj) y sin el **no se escribe ningun fichero** (A2)"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise InvalidAsOfError(f"--as-of no es un ISO-8601 valido: {error} (A2)") from error
    return _as_utc(parsed)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del veredicto por dominancia.

    Codigos de salida: ``0`` = informe escrito (aunque el veredicto sea ``fail`` o
    ``not_evaluable``, que son resultados legitimos y declarados); ``2`` = falta o no es valido
    ``--as-of``, falta o es ambiguo un artefacto de #28/#26, la tabla de §11.6 no se puede leer,
    el pipeline no se puede reejecutar o el barrido rompe la dominancia ⇒ **no se escribe nada**
    y el motivo sale por ``stderr``.
    """
    parser = argparse.ArgumentParser(
        prog="cfdtrader.analysis.phase2_dominance",
        description="Veredicto robusto de Fase 2 por dominancia de la base declarada",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="raiz del almacen")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="directorio de informes (por defecto <data-root>/derived/reports)",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="instante declarado ISO-8601, obligatorio para escribir (el modulo no lee el reloj)",
    )
    parser.add_argument(
        "--previous-artifact",
        type=Path,
        default=None,
        help=(
            "ruta del informe anterior: si se declara, el payload publica el delta de puntero y "
            "las invariantes de la regeneracion (#90)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        moment = _parse_as_of(cast("str | None", args.as_of))
    except (Phase2DominanceError, Phase2ReportError) as error:
        print(f"no se puede emitir el veredicto por dominancia: {error}", file=sys.stderr)
        return 2

    data_root = Path(args.data_root) if args.data_root is not None else Path("data")
    reports_dir = (
        Path(args.reports_dir)
        if args.reports_dir is not None
        else data_root / "derived" / "reports"
    )
    try:
        report = analyse(
            store=Store(data_root),
            reports_dir=reports_dir,
            as_of=moment,
            write=True,
            previous_artifact=cast("Path | None", args.previous_artifact),
        )
    except (
        Phase2DominanceError,
        Phase2ReportError,
        PipelineReportError,
        BacktestReportError,
        regeneration_delta.RegenerationError,
    ) as error:
        print(f"no se puede emitir el veredicto por dominancia: {error}", file=sys.stderr)
        return 2

    dominance = cast("dict[str, object]", report.payload["dominance"])
    gate = cast("dict[str, object]", report.payload["gate"])
    logger.info(
        "dominancia de la base declarada: {} ({}); agregado {} (n_rows {}); report_sha256 = {}",
        dominance["state"],
        dominance["code"],
        gate["aggregate"],
        gate["n_rows"],
        report.report_sha256,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
