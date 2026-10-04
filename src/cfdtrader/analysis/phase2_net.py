"""Veredicto de Fase 2 sobre la base **neta** bajo el supuesto declarado (#88, tarea T29b).

Este modulo **reemite** la puerta de salida de la Fase 2 —la tabla de criterios de *kill*
pre-registrada de `plan.md` §11.6— evaluando sus nueve filas sobre la base **neta** bajo el
supuesto de *slippage* **declarado** ya cuantificado por el `R` que el propietario decidio en #60
(20 bp, #64). Es el hermano de ``phase2_dominance.py`` (#93, T29c): no toca #29, no toca #28 y **no
mide** nada; sustituye lo que #29 no podia evaluar —porque su artefacto publicaba
``net_metrics: not_computable``— por la evaluacion sobre la base neta que #133 publica.

Que **hace**:

- **lee** la tabla de §11.6 de ``_docs/plan.md`` **en tiempo de ejecucion** (nunca una copia
  cableada), con su ``sha256`` y sus 9 filas literales, y **reutiliza** la maquinaria de #29
  (``load_kill_table``, ``evaluate_criteria``, ``gate_block``, ``resolve_verdict``) y la agregacion
  de #9 (``aggregate_gate``, ``recommendation_is_consistent``): no re-deriva ningun umbral;
- **re-deriva** el pipeline de #28 con su API publica (``analyse(..., write=False)``) para leer la
  **serie neta** que #28 ya publica por brazo (``arms.<brazo>.net_series``, desde #133) y las series
  declaradas de las filas de la tabla: el artefacto **publicado** del 2026-09-23 todavia dice
  ``net_metrics: not_computable`` y refrescarlo es **#108** — este informe no depende de el;
- **reproduce** los intervalos netos (``hit_rate`` por sesion, ``hit_rate_per_trade`` y ``sharpe``)
  desde esa serie con las **semillas declaradas** de #28 y **falla con error tipado** si no igualan
  los que publica #28 (A6): publicar una serie reconstruida que no es la del artefacto seria
  publicar otra cosa con el mismo nombre;
- evalua la **fila principal** (A8) con las **dos** mitades de §11.6: la tasa de acierto **por
  operacion** (``hit_rate_per_trade``, el denominador comparable con el ``p*`` de break-even, #92) y
  el Sharpe; ``pass`` si alguna excluye en el sentido del *edge*, ``fail`` si alguna excluye en el
  contrario, ``not_evaluable`` en otro caso. El codigo de #29 se conserva literal;
- evalua las **tres filas de comparacion** (A9) por **bootstrap pareado de la diferencia** (neto del
  brazo base menos neto de la fila, sesion a sesion, semilla declarada): ``no_trade`` y ``liston_a``
  se deciden; ``liston_b`` se **publica** ``not_evaluable`` porque es una serie de **referencia**
  (cierre a cierre con financiacion, sin ejecucion) y compararla contra una base neta inventaria una
  base — su liston de primera clase es #70;
- **agrega** las nueve filas con #9 y **deriva** ``phase2_ready`` del agregado (no lo cablea): hoy
  es ``false`` porque el agregado no es ``pass``, y las filas de Fase 4 siguen ``not_evaluable``.

Que **no** hace, y por tanto no puede inventar:

- **no** mide el *slippage*: los 20 bp son el **supuesto declarado** de #64 cuantificado por el `R`
  de #60 (``ASSUMED_SLIPPAGE_PCT``, **importado**), etiquetado ``assumed`` e ``is_measurement:
  false``; medirlo son 10-15 ejecuciones reales (#62);
- **no** refresca el artefacto publicado de #28 (#108) ni reescribe su bloque ``net_metrics``;
- **no** convierte un ``not_evaluable`` en un ``pass``, y el aprobado de la fila principal **no** es
  un aprobado de la puerta: ``phase2_ready`` exige el agregado;
- **no** re-decide nada de lo registrado por el propietario: `plan.md` §19.6 (``reframe``) y §19.7
  (carril A / carril B) siguen como estan; este informe solo **reemite** la evaluacion de §11.6 y
  publica el veredicto que sale de ella;
- **no** produce el liston B de primera clase (#70), ni reserva el *holdout* (#68), ni implementa
  CPCV (#67), ni evalua la divergencia paper-vs-backtest ni el cierre de sesion (#45, #84);
- **no** escribe ningun ``None`` como ``0``.

**Reloj prohibido** (A2): ninguna ruta consulta el reloj del sistema; el instante entra por
``--as-of``, obligatorio para escribir, que sale con codigo 2 y sin tocar disco cuando falta o no es
ISO-8601. **Red prohibida** y sin escrituras fuera de ``--reports-dir``. Determinista byte a byte:
``report_sha256`` es el sha256 del texto canonico de #13 sobre el payload sin la clave del hash, con
el prefijo ``sha256:``.
"""

from __future__ import annotations

import argparse
import hashlib
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

from cfdtrader.analysis import phase2_report, pipeline_report, regeneration_delta
from cfdtrader.analysis.backtest_report import BacktestReportError
from cfdtrader.analysis.phase0_report import (
    GateVerdict,
    HalfResult,
)
from cfdtrader.analysis.phase2_report import (
    HASH_PREFIX,
    MODEL_CLASS,
    PIPELINE_CLASS,
    ROW_LISTON_A,
    ROW_LISTON_B,
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
    ASSUMED_SLIPPAGE_PCT,
    BASIS_DECLARED_COST_WITH_ASSUMED_SLIPPAGE,
    METRIC_NAMES,
    NET_METRICS_STATE,
    PipelineReportError,
)
from cfdtrader.backtest.baselines import NO_TRADE
from cfdtrader.backtest.engine import STATUS_SKIPPED, STATUS_TRADED, BacktestRun, canonical_text
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
    "BASIS_NET",
    "BEATS_ROWS",
    "BEATS_RULE",
    "HASH_PREFIX",
    "HIT_HALF_RULE",
    "NET_BASIS_RULE",
    "REPORT_DOES_NOT_DO",
    "REPORT_HASH_FORMAT",
    "REPORT_LIMITATIONS",
    "REPORT_PREFIX",
    "SEED_DERIVATION",
    "TASK",
    "TITLE",
    "VERDICT_RULE",
    "DegenerateNetSeriesError",
    "MissingBeatsSeriesError",
    "MissingNetMetricsError",
    "NetReport",
    "NetSeries",
    "Phase2NetError",
    "ReproductionMismatchError",
    "analyse",
    "difference_seed",
    "main",
    "net_series_of_run",
    "render_markdown",
    "seed_of",
]

#: Identidad del informe: quien lo emite y que tarea lo pide.
ANALYSIS: Final[str] = "cfdtrader.analysis.phase2_net"
TASK: Final[str] = "#88"
TITLE: Final[str] = "T29b — Veredicto de Fase 2 sobre la base neta bajo el supuesto declarado"

#: Prefijo del informe: ``phase2_net_<AAAA-MM-DD>.{json,md}``.
REPORT_PREFIX: Final[str] = "phase2_net"

#: Formato estable del ``report_sha256`` (A4).
REPORT_HASH_FORMAT: Final[str] = (
    "sha256:<64 hex> del texto canonico (``canonical_text`` de #13) del payload **sin** la "
    "clave ``report_sha256``. El prefijo viaja dentro del valor: un digest desnudo lo bloquea "
    "`detect-secrets`"
)

#: La base de este informe: coste declarado **mas** el supuesto de *slippage* ya cuantificado
#: (importado de #28, que a su vez lo deriva del `R` de #60 y del supuesto de #64). Nunca medido.
BASIS_NET: Final[str] = BASIS_DECLARED_COST_WITH_ASSUMED_SLIPPAGE

#: El brazo base: el mismo de #93, el unico que opera (A6).
BASE_ARM: Final[str] = ARM_COSTE_DECLARADO

#: Nombre de la fila de la tabla de #28 que es el baseline de no operar. **No** es el `kind` de la
#: fila de §11.6 (`no_operar`, que vive en la tabla de *criterios*): este es el nombre con el que
#: #28 publica la fila y el que #29 lee en `net_metrics.beats`.
NO_TRADE_ROW: Final[str] = NO_TRADE

#: Nombre de la fila de la tabla de #28 que es el liston A (siempre largo, `open→close`).
LISTON_A_ROW: Final[str] = ROW_LISTON_A

#: Las tres filas de comparacion de §11.6, en el orden en que las evalua #29. ``liston_b`` va
#: la ultima y **no** se decide: su base no es comparable (ver ``BEATS_RULE``).
BEATS_ROWS: Final[tuple[str, ...]] = (NO_TRADE_ROW, LISTON_A_ROW, ROW_LISTON_B)

#: La regla del **supuesto** que sostiene toda la lectura neta (A5).
NET_BASIS_RULE: Final[str] = (
    "La base neta es la declarada **menos** el supuesto de *slippage* que el propietario"
    " cuantifico al decidir `R` en #60 (20 % de `R` = 20 % de 1,00 % = 20 bp = 0,2 % del nocional,"
    " restados una vez por sesion **operada**): es un numero **asumido**, no medido"
    " (`slippage_state: assumed`, `is_measurement: false`), y medirlo es #62"
)

#: Por que la mitad de la tasa de acierto se decide **por operacion** (A8; #92, cerrada).
HIT_HALF_RULE: Final[str] = (
    "el `p*` de §11.6 es el break-even de **una operacion**, asi que la tasa comparable con el es "
    "`hit_rate_per_trade` —la denominacion **por operacion** que #28 publica desde #92—: la "
    "`hit_rate` por **sesion** diluye con los `no_trade` y se **publica** como contexto, no decide"
)

#: La regla declarada de las tres filas de comparacion (A9).
BEATS_RULE: Final[str] = (
    "una fila de comparacion se decide por el **bootstrap pareado de la diferencia** (neto del"
    " brazo base menos neto de la fila, sesion a sesion, con semilla declarada): `pass` si el IC 95"
    " % de la diferencia media excluye el 0 **por arriba**, `fail` si lo excluye **por abajo**, "
    "`not_evaluable` en otro caso. `liston_b` **no** se decide: es la serie de **referencia** de "
    "cierre a cierre con financiacion (sin ejecucion y sin base neta comparable), #29 ya la marca "
    "`insufficient_on_its_own` y su liston de primera clase es #70"
)

#: La regla del veredicto y de la puerta (A12).
VERDICT_RULE: Final[str] = (
    "`phase2_ready` es `true` **solo** si el agregado de las nueve filas es `pass` (#9: un "
    "`not_evaluable` nunca se convierte en `pass`, y hace falta que pasen todas); el aprobado de"
    " la fila principal **no** aprueba la puerta por si sola, y el veredicto sale del vocabulario"
    " de #29 (`continue`/`simplify`/`stop`/`not_evaluable`) con la regla de consistencia de #9"
)

#: De donde salen las semillas de las comparaciones: continuacion declarada de la de #28 (A9).
SEED_DERIVATION: Final[str] = (
    "los intervalos del brazo base usan la semilla que #28 deriva de #15 para cada metrica "
    "(`(DEFAULT_BOOTSTRAP_SEED + posicion de la metrica en METRIC_NAMES + 1) % 2**32`); las "
    "comparaciones continuan esa misma derivacion una y dos posiciones **despues** de la ultima "
    "metrica de #28, asi que no pueden colisionar con ninguna de las once"
)


# ─────────────────────────────────────────────────────────────────────────────
# Textos declarados: limites, lo que no hace y sus seguimientos
# ─────────────────────────────────────────────────────────────────────────────
#: Limites declarados del informe.
REPORT_LIMITATIONS: Final[tuple[str, ...]] = (
    "los 20 bp del supuesto de *slippage* son un **numero asumido** (`assumed`, #64) cuantificado "
    "por el `R` que el propietario decidio en #60, **no** una medicion: la lectura neta de este "
    "informe es la mejor conjetura disponible y `is_measurement` es `false`; medirlo son 10-15 "
    "ejecuciones reales (#62)",
    "la base neta es peor o igual que la declarada **por construccion** (el *slippage* es >= 0): "
    "esto **no** cambia el veredicto por dominancia de #93, que sigue vigente como cota superior,"
    " y lo que aporta este informe es la **cuantificacion** sobre la mejor conjetura",
    "el informe se emite desde una **re-derivacion en vivo** del pipeline de #28 "
    "(`analyse(..., write=False)`) porque el artefacto **publicado** del 2026-09-23 todavia"
    " publica `net_metrics: not_computable` (es anterior a #133): refrescarlo es #108, y hasta"
    " entonces el `sha256` del artefacto publicado no es el de los numeros de este informe",
    "la fila del drawdown se mide sobre `declared_cost` del brazo base, **no** sobre la base neta: "
    "se publica con su `basis` y **no** se presenta como medida neta (la herencia de #29)",
    "el liston B de primera clase (posiciones overnight por el motor) es #70: aqui su fila se "
    "**publica** `not_evaluable` en vez de compararla contra una base que no le corresponde",
    "reemitir la evaluacion de §11.6 **no** re-decide nada de lo ya registrado por el propietario: "
    "`plan.md` §19.6 (`reframe`) y §19.7 (carril A / carril B) siguen como estan y el carril B"
    " sigue bloqueado mientras `phase2_ready` sea `false`",
)

#: Lo que este informe no hace, con la issue que lo cierra.
REPORT_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_mide_el_slippage",
        "issue": "#62",
        "statement": (
            "no mide el *slippage*: publica la lectura neta bajo el supuesto declarado de #64 ya "
            "cuantificado por el `R` de #60 (20 bp) y la etiqueta `is_measurement: false`"
        ),
    },
    {
        "id": "no_refresca_el_artefacto_publicado",
        "issue": "#108",
        "statement": (
            "no refresca `pipeline_backtest_2026-09-23.json`: lo re-deriva en vivo y en solo "
            "lectura. Refrescar los artefactos publicados es #108"
        ),
    },
    {
        "id": "no_decide_el_liston_b_ni_la_fase_4",
        "issue": "#70",
        "statement": (
            "no decide la fila de `liston_b` (serie de referencia sin base neta comparable; su "
            "liston de primera clase es #70) ni las de Fase 4 (paper y cierre), que se publican "
            "`not_evaluable`"
        ),
    },
    {
        "id": "no_reserva_holdout_ni_cpcv",
        "issue": "#68",
        "statement": (
            "no reserva el *holdout* final (#68) ni implementa CPCV (#67): ninguna de las dos "
            "cosas se decide aqui"
        ),
    },
    {
        "id": "no_reabre_lo_registrado",
        "issue": "#29",
        "statement": (
            "no re-decide §19.6 ni §19.7 ni mueve la tabla de §11.6: reemite su evaluacion sobre "
            "la base neta con la misma maquinaria de #29"
        ),
    },
)

#: Seguimientos declarados del informe.
FOLLOW_UPS: Final[tuple[dict[str, str], ...]] = (
    {
        "issue": "#62",
        "topic": "medir el *slippage*",
        "why": "la base neta es un supuesto declarado: la medicion cierra el termino dominante",
    },
    {
        "issue": "#108",
        "topic": "refrescar los artefactos publicados",
        "why": (
            "el informe de #28 publicado el 2026-09-23 todavia dice `net_metrics: not_computable`; "
            "hasta refrescarlo, este informe declara que su base es una re-derivacion en vivo"
        ),
    },
    {
        "issue": "#70",
        "topic": "liston B de primera clase",
        "why": "su fila se publica `not_evaluable` en vez de compararla contra una base inventada",
    },
    {
        "issue": "#45",
        "topic": "divergencia paper vs backtest",
        "why": "es de Fase 4: la fila se publica `not_evaluable`",
    },
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores declarados: un hueco nunca se rellena con un valor por defecto
# ─────────────────────────────────────────────────────────────────────────────
class Phase2NetError(Exception):
    """Error declarado del veredicto sobre la base neta: nunca se rellena el hueco."""


class MissingNetMetricsError(Phase2NetError):
    """El payload de #28 no publica la lectura neta computable bajo el supuesto (A5, A7)."""


class DegenerateNetSeriesError(Phase2NetError):
    """La serie neta del brazo base no tiene sesiones o es todo ceros: no hay nada que medir."""


class MissingBeatsSeriesError(Phase2NetError):
    """No se puede construir la base neta de una fila de comparacion (A9)."""


class ReproductionMismatchError(Phase2NetError):
    """Los intervalos recomputados no igualan los del artefacto de #28 (A6)."""


# ─────────────────────────────────────────────────────────────────────────────
# La serie neta del brazo base (A5, A6, A7)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class NetSeries:
    """La serie **neta** del brazo base: ``%`` por sesion de *test* y mascara de operadas.

    Una sesion sin operacion vale ``0`` **exacto** (no se opero: no hay coste que cobrar) y una
    sesion saltada no entra (la convencion de #15/#28). Los valores **se leen** del artefacto de
    #28 (``arms.<brazo>.net_series``, desde #133); la mascara viene de la corrida en vivo y solo
    sirve para re-comprobar los recuentos que #28 publica.
    """

    values_pct: tuple[float, ...]
    traded: tuple[bool, ...]
    artifact_n: int
    artifact_sha256: str

    def __post_init__(self) -> None:
        if len(self.values_pct) != len(self.traded):
            raise Phase2NetError(
                "la serie neta y su mascara de operadas tienen que tener la misma longitud "
                f"({len(self.values_pct)} != {len(self.traded)})"
            )

    @property
    def n_sessions(self) -> int:
        """Sesiones de *test* que entran en la serie."""
        return len(self.values_pct)

    @property
    def n_traded(self) -> int:
        """Sesiones operadas: las unicas que pagan el supuesto de *slippage*."""
        return sum(1 for flag in self.traded if flag)

    @property
    def is_degenerate(self) -> bool:
        """Serie degenerada: sin sesiones o con todos los retornos exactamente ``0``."""
        return not self.values_pct or all(value == 0.0 for value in self.values_pct)

    def traded_series_pct(self) -> tuple[float, ...]:
        """Las sesiones **operadas**, en orden: la denominacion por operacion (#92)."""
        return tuple(
            value for value, flag in zip(self.values_pct, self.traded, strict=True) if flag
        )


def _traded_mask(run: BacktestRun) -> tuple[bool, ...]:
    """La mascara de sesiones operadas de una corrida, en el orden de #28 (A6).

    Recorre los *folds* y las sesiones como los recorre #28: una sesion saltada **no entra** y una
    sesion sin operacion entra como ``False``.
    """
    mask: list[bool] = []
    for fold in run.folds:
        for outcome in fold.sessions:
            if outcome.status == STATUS_SKIPPED:
                continue
            mask.append(outcome.status == STATUS_TRADED)
    return tuple(mask)


#: Las cuatro claves del **cuerpo** de la serie en #28: su digest se calcula sobre estas solas
#: (las etiquetas del supuesto viajan al lado, pero **no** entran en el digest).
SERIES_BODY_KEYS: Final[tuple[str, ...]] = ("units", "n", "series_pct")


def _published_net_block(payload: Mapping[str, object]) -> dict[str, object]:
    """El bloque ``arms.<brazo>.net_series`` que publica #28, o error tipado si no esta (A5, A7)."""
    arms = _mapping(payload.get("arms"), where="arms")
    block = _mapping(arms.get(BASE_ARM), where=f"arms.{BASE_ARM}")
    series = block.get("net_series")
    if series is None:
        state = _net_state(payload)
        raise MissingNetMetricsError(
            f"`arms.{BASE_ARM}.net_series` no existe y el artefacto de #28 declara "
            f"`net_metrics: {state}`: la lectura neta bajo el supuesto la publica #133 (A5, A7). "
            "Si el artefacto es anterior, hay que re-derivar el pipeline en vivo o refrescarlo "
            "(#108)"
        )
    return _mapping(series, where=f"arms.{BASE_ARM}.net_series")


def _net_state(payload: Mapping[str, object]) -> str:
    """El estado declarado del bloque ``net_metrics`` de #28, sin inventarlo si falta (A5)."""
    node = payload.get("net_metrics")
    if not isinstance(node, dict):
        return "absent"
    return str(cast("dict[str, object]", node).get("state", "absent"))


def _checked_series_digest(block: Mapping[str, object], *, where: str) -> str:
    """Recompone el digest **desde el cuerpo publicado** y lo compara con el que viaja al lado.

    No se fija ningun literal (un artefacto regenerable no se ancla a un dorado): el digest se
    recalcula con ``canonical_text`` de #13 sobre las claves del cuerpo y la comprobacion es de
    **autoconsistencia**.
    """
    body = {key: block[key] for key in SERIES_BODY_KEYS if key in block}
    if len(body) != len(SERIES_BODY_KEYS):
        raise MissingNetMetricsError(
            f"`{where}` no publica el cuerpo completo de la serie: faltan claves de "
            f"{list(SERIES_BODY_KEYS)} (A7)"
        )
    digest = HASH_PREFIX + hashlib.sha256(canonical_text(body).encode("utf-8")).hexdigest()
    published = block.get("series_sha256")
    if published != digest:
        raise ReproductionMismatchError(
            f"`{where}.series_sha256` no es autoconsistente con su propio cuerpo: publicado "
            f"{published!r}, recomputado {digest!r}. No se publica una serie cuyo digest no cuadra "
            "(A6)"
        )
    return digest


def net_series_of_run(*, run: BacktestRun, payload: Mapping[str, object]) -> NetSeries:
    """La serie neta del brazo base: **leida** del artefacto de #28 y con la mascara en vivo (A7).

    La mascara dice en que sesiones hubo operacion y se comprueba contra los recuentos que #28
    publica para el brazo (``traded`` y ``no_trade``): si no cuadran, se lanza error tipado en vez
    de publicar una serie mal alineada.
    """
    block = _published_net_block(payload)
    raw = block.get("series_pct")
    if not isinstance(raw, list):
        raise MissingNetMetricsError(
            f"`arms.{BASE_ARM}.net_series.series_pct` deberia ser una lista, no "
            f"{type(raw).__name__}: el artefacto de #28 no publica la lectura neta (A7)"
        )
    values = tuple(
        _finite(value, where=f"arms.{BASE_ARM}.net_series.series_pct")
        for value in cast("list[object]", raw)
    )
    digest = _checked_series_digest(block, where=f"arms.{BASE_ARM}.net_series")
    traded = _traded_mask(run)
    if len(values) != len(traded):
        raise Phase2NetError(
            f"la serie neta publicada tiene {len(values)} entradas y la corrida en vivo tiene "
            f"{len(traded)} sesiones: no se pueden alinear (A7)"
        )
    series = NetSeries(
        values_pct=values, traded=traded, artifact_n=len(values), artifact_sha256=digest
    )
    counts = _mapping(
        _mapping(payload.get("arms"), where="arms").get(BASE_ARM), where=f"arms.{BASE_ARM}"
    )
    expected_traded = int(_number(counts, "traded", where=f"arms.{BASE_ARM}"))
    expected_no_trade = int(_number(counts, "no_trade", where=f"arms.{BASE_ARM}"))
    no_trade = series.n_sessions - series.n_traded
    if series.n_traded != expected_traded or no_trade != expected_no_trade:
        raise Phase2NetError(
            "los recuentos de la serie neta no cuadran con los de #28 para el brazo base "
            f"({series.n_traded} operadas / {no_trade} sin operar frente a {expected_traded} / "
            f"{expected_no_trade}): no se publica una serie desalineada (A7)"
        )
    return series


# ─────────────────────────────────────────────────────────────────────────────
# Lectura tipada y utilidades de serializacion (A14)
# ─────────────────────────────────────────────────────────────────────────────
def _mapping(node: object, *, where: str) -> dict[str, object]:
    """Vista tipada de un nodo que tiene que ser un objeto JSON."""
    if not isinstance(node, dict):
        raise Phase2NetError(f"{where}: se espera un objeto JSON, no {type(node).__name__} (A7)")
    return cast("dict[str, object]", node)


def _number(node: Mapping[str, object], key: str, *, where: str) -> float:
    """Campo numerico obligatorio del artefacto; su ausencia es un error tipado (A7)."""
    value = node.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase2NetError(
            f"`{where}.{key}` deberia ser un numero, no {type(value).__name__}: no se rellena con "
            "un valor por defecto (A7)"
        )
    return float(value)


def _finite(value: object, *, where: str) -> float:
    """Un valor de serie que tiene que ser finito: el JSON nunca lleva ``nan`` ni ``inf`` (A14)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise Phase2NetError(
            f"`{where}` deberia ser un numero finito, no {value!r}: no se publica `nan` ni `inf` "
            "(A14)"
        )
    return float(value)


# ─────────────────────────────────────────────────────────────────────────────
# Semillas e intervalos: los mismos de #28 para el brazo base (A6, A9)
# ─────────────────────────────────────────────────────────────────────────────
def seed_of(metric: str) -> int:
    """La semilla declarada de esa metrica en #28, derivada de su posicion en #15 (A6)."""
    if metric not in METRIC_NAMES:
        raise Phase2NetError(
            f"la metrica {metric!r} no esta en `METRIC_NAMES` de #28: {list(METRIC_NAMES)} (A6)"
        )
    return (DEFAULT_BOOTSTRAP_SEED + METRIC_NAMES.index(metric) + 1) % (2**32)


def difference_seed(index: int) -> int:
    """La semilla de la comparacion ``index``: la derivacion de #28 continuada (A9).

    Empieza **despues** de la ultima metrica de #28, asi que no puede colisionar con ninguna de las
    once; ningun numero de semilla se teclea.
    """
    if index < 0:
        raise Phase2NetError(
            f"el desplazamiento {index} no vale para derivar una semilla: tiene que ser >= 0 (A9)"
        )
    return (DEFAULT_BOOTSTRAP_SEED + len(METRIC_NAMES) + index + 1) % (2**32)


def _hit_rate_statistic(sample: Sequence[float]) -> float:
    """La tasa de acierto de una muestra, con la **misma** definicion que #28 (A6)."""
    return sum(1 for value in sample if value > 0.0) / max(len(sample), 1)


def _mean_statistic(sample: Sequence[float]) -> float:
    """La media de una muestra (la estadistica de la diferencia de las comparaciones, A9)."""
    return math.fsum(sample) / max(len(sample), 1)


def _interval(
    values_pct: Sequence[float],
    *,
    metric: str,
    seed: int,
    confidence_level: float,
    n_bootstrap: int,
    scale: float = 1.0,
) -> dict[str, float]:
    """``estimate``/``lower``/``upper`` de una metrica sobre una serie en ``%`` (A6, A9)."""
    decimals = tuple(value / 100.0 for value in values_pct)
    statistic = _hit_rate_statistic if metric.startswith("hit_rate") else sharpe_ratio
    interval = bootstrap_confidence_interval(
        decimals,
        statistic,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return {
        "estimate": interval.estimate * scale,
        "lower": interval.lower * scale,
        "upper": interval.upper * scale,
    }


def _difference_interval(
    *,
    left_pct: Sequence[float],
    right_pct: Sequence[float],
    seed: int,
    confidence_level: float,
    n_bootstrap: int,
) -> dict[str, object]:
    """El IC bootstrap **pareado** de la diferencia media ``izquierda - derecha``, en ``%`` (A9).

    Las dos series van alineadas sesion a sesion (mismo orden y misma longitud); el intervalo se
    calcula sobre la diferencia en **unidad decimal** (``0,01`` = 1 %) y se devuelve en ``%``, la
    convencion de #28 para las medias.
    """
    if len(left_pct) != len(right_pct):
        raise MissingBeatsSeriesError(
            f"la comparacion pareada exige la misma longitud, y llegan {len(left_pct)} y "
            f"{len(right_pct)}: no se recorta ninguna serie (A9)"
        )
    difference = tuple(
        (left - right) / 100.0 for left, right in zip(left_pct, right_pct, strict=True)
    )
    interval = bootstrap_confidence_interval(
        difference,
        _mean_statistic,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return {
        "estimate": interval.estimate * 100.0,
        "lower": interval.lower * 100.0,
        "upper": interval.upper * 100.0,
        "n_sessions": len(difference),
    }


# ─────────────────────────────────────────────────────────────────────────────
# El bloque neto: etiquetas, reproduccion y comparaciones (A5, A6, A9)
# ─────────────────────────────────────────────────────────────────────────────
#: El estado que #29 exige en el bloque ``net_metrics`` para evaluar las filas de la base neta.
STATE_COMPUTED: Final[str] = "computed"

#: Nota del estado: el vocabulario de #29 no distingue «asumido» de «medido».
STATE_NOTE: Final[str] = (
    "`computed` es el **vocabulario que #29 exige** para evaluar las filas de la base neta; el "
    "numero que hay debajo es **asumido**, no medido (`basis` con el supuesto, `is_measurement: "
    "false`, #64/#60, medirlo es #62)"
)


def _assumption_labels(payload: Mapping[str, object]) -> dict[str, object]:
    """La **etiqueta** del supuesto, copiada del bloque de #28 y **verificada** (A5).

    Se comprueba que #28 declara la lectura neta como supuesto (``assumed``, ``is_measurement:
    false``, base con el supuesto) y que el termino coincide con el importado
    (``ASSUMED_SLIPPAGE_PCT``, derivado del `R` de #60): si no cuadra, error tipado en vez de
    publicar un numero con la etiqueta de otro.
    """
    net = _mapping(payload.get("net_metrics"), where="net_metrics")
    state = str(net.get("state", "absent"))
    if state != NET_METRICS_STATE:
        raise MissingNetMetricsError(
            f"#28 declara `net_metrics: {state}` y esta tarea espera `{NET_METRICS_STATE}` "
            "(la lectura neta bajo el supuesto declarado, desde #133): si el artefacto es"
            " anterior, hay que re-derivar el pipeline en vivo o refrescarlo (#108) (A5)"
        )
    basis = str(net.get("basis", ""))
    measurement = net.get("is_measurement")
    assumed = net.get("assumed_slippage_pct")
    expected = format(ASSUMED_SLIPPAGE_PCT, "f")
    if basis != BASIS_NET or measurement is not False or assumed != expected:
        raise MissingNetMetricsError(
            f"el bloque neto de #28 no lleva la etiqueta declarada: `basis` {basis!r} (esperada "
            f"{BASIS_NET!r}), `is_measurement` {measurement!r} (esperado `False`) y "
            f"`assumed_slippage_pct` {assumed!r} (esperado {expected!r}) (A5)"
        )
    return {
        "basis": BASIS_NET,
        "slippage_state": str(net.get("slippage_state", "assumed")),
        "is_measurement": False,
        "is_validation": False,
        "assumed_slippage_pct": expected,
        "slippage_pct_of_r": str(net.get("slippage_pct_of_r", "")),
        "r_pct": str(net.get("r_pct", "")),
        "assumption_issue": str(net.get("assumption_issue", "#64")),
        "r_issue": str(net.get("r_issue", "#60")),
        "measuring_issue": str(net.get("measuring_issue", "#62")),
        "rule": NET_BASIS_RULE,
    }


def _net_intervals(
    series: NetSeries, *, n_bootstrap: int, confidence_level: float
) -> dict[str, dict[str, float]]:
    """Los tres intervalos del brazo base sobre la base neta: sesion, operacion y Sharpe (A6).

    ``hit_rate`` y ``sharpe`` se estiman sobre la serie **entera** (una ``no_trade`` entra como 0
    exacto y diluye la tasa, la convencion de #15/#28) y ``hit_rate_per_trade`` sobre la subserie
    **operada** (la denominacion por operacion de #92, la comparable con el `p*` de §11.6).
    """
    return {
        "hit_rate": _interval(
            series.values_pct,
            metric="hit_rate",
            seed=seed_of("hit_rate"),
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
        ),
        "hit_rate_per_trade": _interval(
            series.traded_series_pct(),
            metric="hit_rate_per_trade",
            seed=seed_of("hit_rate_per_trade"),
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
        ),
        "sharpe": _interval(
            series.values_pct,
            metric="sharpe",
            seed=seed_of("sharpe"),
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
        ),
    }


def reproduce_net_metrics(
    *,
    series: NetSeries,
    intervals: Mapping[str, Mapping[str, float]],
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Reproduce los intervalos netos de #28 desde su **propia serie** publicada (A6).

    Exige que los tres intervalos recalculados (``hit_rate`` por sesion, ``hit_rate_per_trade`` por
    operacion y ``sharpe``) igualen (igualdad de ``float``) los que #28 publica en
    ``arms.<brazo>.net_metrics`` con las **mismas semillas declaradas**. Si no cuadran, se lanza
    error tipado: la serie leida no es la que produjo esos numeros.
    """
    arms = _mapping(payload.get("arms"), where="arms")
    block = _mapping(arms.get(BASE_ARM), where=f"arms.{BASE_ARM}")
    published_block = _mapping(block.get("net_metrics"), where=f"arms.{BASE_ARM}.net_metrics")
    compared: dict[str, object] = {}
    reproduces = True
    for metric in ("hit_rate", "hit_rate_per_trade", "sharpe"):
        interval = intervals[metric]
        published = _mapping(
            published_block.get(metric), where=f"arms.{BASE_ARM}.net_metrics.{metric}"
        )
        seeds = {"artifact": published.get("seed"), "here": seed_of(metric)}
        deltas: dict[str, object] = {}
        matches = seeds["artifact"] == seeds["here"]
        for key, found in interval.items():
            expected = _number(published, key, where=f"arms.{BASE_ARM}.net_metrics.{metric}")
            delta = found - expected
            deltas[key] = {"artifact": expected, "here": found, "delta": delta}
            matches = matches and delta == 0.0
        compared[metric] = {
            "interval": dict(interval),
            "seeds": seeds,
            "deltas": deltas,
            "matches": matches,
        }
        reproduces = reproduces and matches
    result: dict[str, object] = {
        "checked": True,
        "source": f"artefacto de #28, `arms.{BASE_ARM}.net_series` y `arms.{BASE_ARM}.net_metrics`",
        "series_sha256": series.artifact_sha256,
        "seeds": {
            metric: seed_of(metric) for metric in ("hit_rate", "hit_rate_per_trade", "sharpe")
        },
        "rule": (
            "los tres intervalos se recomputan desde la serie **publicada** por #28 con las "
            "**mismas semillas declaradas** y tienen que igualarla (igualdad de `float`): si no,"
            " la serie leida no es la que produjo esos numeros y no se publica como si lo fuera"
        ),
        "reproduces": reproduces,
        "metrics": compared,
    }
    if not reproduces:
        raise ReproductionMismatchError(
            "los intervalos netos recomputados desde la serie publicada por #28 no la reproducen "
            "(metrica, intervalo o semilla): no se publica una serie distinta con el mismo nombre "
            f"(A6). Comparacion: {json.dumps(compared, sort_keys=True, ensure_ascii=False)}"
        )
    return result


def _row_net_series_pct(*, row: pipeline_report.TableRow, name: str) -> list[float]:
    """La serie **neta** de una fila de comparacion, o error tipado si su base no es comparable.

    El supuesto se resta una vez por sesion **operada**: una fila que no opera nunca no paga nada
    (su serie declarada **es** su serie neta, y tiene que ser todo ceros) y una fila que opera
    todas las sesiones paga en todas. Una fila con mascara **parcial** exigiria la mascara sesion a
    sesion que #28 no publica para las filas de la tabla: se prefiere el error tipado a inventarla.
    """
    values = [float(value) for value in row.series_pct]
    if row.traded == 0:
        if any(value != 0.0 for value in values):
            raise MissingBeatsSeriesError(
                f"la fila {name!r} no opera ninguna sesion y su serie no es todo ceros: sin "
                "operacion no hay coste que cobrar y no se puede construir su base neta (A9)"
            )
        return values
    if row.traded != row.n_test:
        raise MissingBeatsSeriesError(
            f"la fila {name!r} opera {row.traded} de {row.n_test} sesiones: su mascara es parcial"
            " y #28 no publica la mascara sesion a sesion de las filas de la tabla, asi que su"
            " base neta no se puede construir sin inventarla (A9)"
        )
    return [value - float(ASSUMED_SLIPPAGE_PCT) for value in values]


def _beat_state(interval: Mapping[str, object]) -> tuple[str, str]:
    """``(estado, motivo)`` de una diferencia neta: la regla declarada de ``BEATS_RULE`` (A9)."""
    lower = float(cast("float", interval["lower"]))
    upper = float(cast("float", interval["upper"]))
    if lower > 0.0:
        return str(HalfResult.PASS), "difference_above_zero"
    if upper < 0.0:
        return str(HalfResult.FAIL), "difference_below_zero"
    return str(HalfResult.NOT_EVALUABLE), "difference_interval_contains_zero"


def _beats_block(
    *,
    series: NetSeries,
    report: pipeline_report.PipelineReport,
    n_bootstrap: int,
    confidence_level: float,
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    """Las tres filas de comparacion: el estado que lee #29 y su detalle publicado (A9).

    Devuelve el mapa ``beats`` (nombre de fila -> estado, que es lo que #29 consume) y el bloque de
    detalle con la diferencia pareada, su semilla y su regla.
    """
    states: dict[str, str] = {}
    detail: dict[str, dict[str, object]] = {}
    left = series.values_pct
    for index, name in enumerate(BEATS_ROWS):
        if name == ROW_LISTON_B:
            states[name] = str(HalfResult.NOT_EVALUABLE)
            detail[name] = {
                "state": str(HalfResult.NOT_EVALUABLE),
                "code": "reference_series_not_comparable",
                "decided": False,
                "issue": "#70",
                "reason": (
                    "`liston_b` es la serie de **referencia** de cierre a cierre con financiacion: "
                    "no pasa por el motor, no tiene ejecucion y su base neta no es comparable con"
                    " la del brazo; su liston de primera clase (posiciones overnight por el motor)"
                    " es #70. Se publica `not_evaluable` en vez de inventarle una base"
                ),
                "rule": BEATS_RULE,
            }
            continue
        row = report.row(name)
        right = _row_net_series_pct(row=row, name=name)
        if len(right) != len(left):
            raise MissingBeatsSeriesError(
                f"la fila {name!r} tiene {len(right)} sesiones y el brazo base {len(left)}: la "
                "comparacion pareada exige la misma longitud (A9)"
            )
        seed = difference_seed(index)
        interval = _difference_interval(
            left_pct=left,
            right_pct=right,
            seed=seed,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
        )
        state, code = _beat_state(interval)
        states[name] = state
        detail[name] = {
            "state": state,
            "code": code,
            "decided": True,
            "difference_pct": interval,
            "seed": seed,
            "n_sessions": len(right),
            "basis": BASIS_NET,
            "is_measurement": False,
            "rule": BEATS_RULE,
        }
    return states, detail


def _rate_block(
    *,
    interval: Mapping[str, float],
    denominator: str,
    note: str,
    series: Sequence[float],
    role: str | None = None,
) -> dict[str, object]:
    """Un bloque de tasa de acierto con su denominacion declarada, sus aciertos y su papel (A6)."""
    wins = sum(1 for value in series if value > 0.0)
    block: dict[str, object] = {
        "estimate": interval["estimate"],
        "lower": interval["lower"],
        "upper": interval["upper"],
        "basis": BASIS_NET,
        "n": len(series),
        "n_wins": wins,
        "wins_fraction": f"{wins}/{len(series)}",
        "denominator": denominator,
        "denominator_note": note,
    }
    if role is not None:
        block["role"] = role
    return block


def _net_block(
    *,
    labels: Mapping[str, object],
    series: NetSeries,
    intervals: Mapping[str, Mapping[str, float]],
    beats: Mapping[str, str],
    beats_detail: Mapping[str, Mapping[str, object]],
    reproduction: Mapping[str, object],
    n_bootstrap: int,
    confidence_level: float,
) -> dict[str, object]:
    """El bloque ``net_metrics`` que consume #29: la lectura neta bajo el supuesto, etiquetada (A5).

    Se le entrega a #29 una **vista** del payload con este bloque en lugar del suyo: es la unica
    forma de que sus filas de la base neta (la principal y las tres comparaciones) se evaluen contra
    numeros reales sin reescribir su maquinaria. El ``state`` es ``computed`` porque es el
    vocabulario que #29 exige; la etiqueta del supuesto viaja entera y el detalle lo dice.
    """
    sessions = series.values_pct
    traded = series.traded_series_pct()
    return {
        "state": STATE_COMPUTED,
        "state_note": STATE_NOTE,
        "basis": BASIS_NET,
        "is_measurement": False,
        "is_validation": False,
        "where": f"`arms.{BASE_ARM}.net_series` de #28 (#133), re-derivado en vivo",
        "hit_rate": _rate_block(
            interval=intervals["hit_rate"],
            denominator="session",
            note=(
                "denominacion **por sesion**: entran todas las sesiones de *test* que no estan "
                "`skipped`; una sesion `no_trade` entra como 0 exacto y diluye la tasa. Es la"
                " misma denominacion que publica #28"
            ),
            series=sessions,
            role="publicada como contexto: el `p*` de §11.6 es por operacion y la comparable es "
            "`hit_rate_per_trade` (#92)",
        ),
        "hit_rate_per_trade": _rate_block(
            interval=intervals["hit_rate_per_trade"],
            denominator="trade",
            note=(
                "denominacion **por operacion**: entran solo las sesiones `traded`, una por "
                "operacion. Es la `p_win` de plan.md §11.6 y la tasa con la que decide este"
                " informe (#92)"
            ),
            series=traded,
            role="decide la mitad de la tasa de acierto de la fila principal (A8)",
        ),
        "sharpe": {
            "estimate": intervals["sharpe"]["estimate"],
            "lower": intervals["sharpe"]["lower"],
            "upper": intervals["sharpe"]["upper"],
            "basis": BASIS_NET,
            "n": len(sessions),
            "annualization": "cfdtrader.backtest.metrics.sharpe_ratio (#15)",
            "role": "decide la mitad del Sharpe de la fila principal (A8)",
        },
        "beats": dict(beats),
        "beats_detail": {name: dict(block) for name, block in beats_detail.items()},
        "beats_rule": BEATS_RULE,
        "reproduction": dict(reproduction),
        "bootstrap": {
            "n_bootstrap": n_bootstrap,
            "confidence_level": confidence_level,
            "seed": DEFAULT_BOOTSTRAP_SEED,
            "seeds_by_metric": {
                metric: seed_of(metric) for metric in ("hit_rate", "hit_rate_per_trade", "sharpe")
            },
            "difference_seeds": {
                name: difference_seed(index) for index, name in enumerate(BEATS_ROWS)
            },
            "seed_source": "cfdtrader.backtest.metrics.DEFAULT_BOOTSTRAP_SEED (#15)",
            "derivation": SEED_DERIVATION,
            "source": "cfdtrader.backtest.metrics.bootstrap_confidence_interval (#15)",
        },
        **{
            key: value
            for key, value in labels.items()
            if key not in {"basis", "is_measurement", "is_validation"}
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# La fila principal (A8) y los bloques que la acompanan
# ─────────────────────────────────────────────────────────────────────────────
def _main_row(
    *,
    evaluated: Mapping[str, object],
    net: Mapping[str, object],
    stars: Mapping[str, object],
    decided_r_pct: str,
) -> dict[str, object]:
    """La fila principal de §11.6 evaluada sobre la base **neta**, con las dos mitades (A8).

    Se conservan los literales del documento (``source_row``, ``threshold``, ``action``) y el
    **codigo** y la **precedencia** de #29 (primero el *edge* a favor, despues el contrario, y
    `not_evaluable` si ningun intervalo excluye), pero la mitad de la tasa de acierto se decide con
    la tasa **por operacion** (``hit_rate_per_trade``), que es la comparable con el `p*` de §11.6
    (#92). La tasa por **sesion** se publica en el detalle como contexto.

    El `p*` que **decide** es el **vinculante** (el del escenario declarado de `R` mas exigente), la
    regla de #93/#29; se publica ademas el del `R` que el propietario **decidio** en #60 (S1) para
    que se vea de un vistazo que la decision no depende de cual de los dos se lea.
    """
    binding = _mapping(stars["binding"], where="p_star.binding")
    p_star_fraction = Decimal(str(binding["p_star_fraction"]))
    p_value = float(p_star_fraction)
    scenarios = [
        dict(_mapping(item, where="p_star.scenarios"))
        for item in cast("list[object]", stars["scenarios"])
    ]
    decided = next(
        (item for item in scenarios if str(item.get("r_pct")) == decided_r_pct),
        None,
    )
    per_trade = _mapping(net["hit_rate_per_trade"], where="net_metrics.hit_rate_per_trade")
    session = _mapping(net["hit_rate"], where="net_metrics.hit_rate")
    sharpe = _mapping(net["sharpe"], where="net_metrics.sharpe")
    hit_low = _number(per_trade, "lower", where="net_metrics.hit_rate_per_trade")
    hit_high = _number(per_trade, "upper", where="net_metrics.hit_rate_per_trade")
    sharpe_low = _number(sharpe, "lower", where="net_metrics.sharpe")
    sharpe_high = _number(sharpe, "upper", where="net_metrics.sharpe")
    hit_above = hit_low > p_value
    hit_below = hit_high < p_value
    sharpe_above = sharpe_low > 0.0
    sharpe_below = sharpe_high < 0.0
    if hit_above or sharpe_above:
        state, code = str(HalfResult.PASS), "statistical_edge"
    elif hit_below or sharpe_below:
        state, code = str(HalfResult.FAIL), "edge_against"
    else:
        state, code = str(HalfResult.NOT_EVALUABLE), "no_statistical_edge"
    detail: dict[str, object] = {
        "basis": BASIS_NET,
        "is_measurement": False,
        "decided_by": "las dos mitades de §11.6 sobre la base neta",
        "hit_rate_per_trade": {
            "lower": hit_low,
            "upper": hit_high,
            "n": per_trade.get("n"),
            "denominator": per_trade.get("denominator"),
        },
        "hit_rate_session": {
            "lower": _number(session, "lower", where="net_metrics.hit_rate"),
            "upper": _number(session, "upper", where="net_metrics.hit_rate"),
            "n": session.get("n"),
            "denominator": session.get("denominator"),
            "role": session.get("role"),
        },
        "sharpe": {"lower": sharpe_low, "upper": sharpe_high, "n": sharpe.get("n")},
        "p_star": {
            "p_star_pct": binding.get("p_star_pct"),
            "p_star_fraction": binding.get("p_star_fraction"),
            "r_pct": binding.get("r_pct"),
            "deciding_rule": (
                "el escenario declarado de `R` mas exigente (el `p*` mayor), como #93/#29"
            ),
            "decided_r_pct": decided_r_pct,
            "decided_p_star_pct": None if decided is None else decided.get("p_star_pct"),
            "decided_note": (
                "`R` decidido en #60 (el escenario declarado S1): su `p*` es el central, no el que "
                "decide; se publica para que se vea que el resultado no depende de cual de los "
                "dos se lea"
            ),
        },
        "hit_rate_excludes_p_star_above": hit_above,
        "hit_rate_below_p_star": hit_below,
        "sharpe_excludes_zero_above": sharpe_above,
        "sharpe_below_zero": sharpe_below,
        "precedence": (
            "el codigo y la precedencia son los de #29: primero el *edge* a favor, despues el "
            "contrario, y `not_evaluable` si ninguna mitad excluye"
        ),
        "hit_half_rule": HIT_HALF_RULE,
    }
    return {
        **dict(evaluated),
        "state": state,
        "code": code,
        "basis": BASIS_NET,
        "detail": detail,
        "evaluated_by": "base neta bajo el supuesto declarado (#88)",
        "note": (
            "la fila se evalua sobre la base **neta** bajo el supuesto declarado (#64/#60): "
            "`is_measurement: false`, medirlo es #62"
        ),
    }


def _beat_liston_b(evaluated: list[dict[str, object]]) -> None:
    """Marca la fila de B como *insuficiente por si sola*, **reutilizando** la marca de #29 (A11).

    Se llama a ``phase2_report._mark_liston_b`` en vez de reescribir su texto: es la misma
    declaracion y este informe no debe divergir de ella.
    """
    phase2_report._mark_liston_b(evaluated)  # pyright: ignore[reportPrivateUsage]


def _base_arm_block(
    *, counts: Mapping[str, object], others: Mapping[str, Mapping[str, object]], series: NetSeries
) -> dict[str, object]:
    """El brazo base sobre la base neta: recuentos leidos del artefacto y serie (A6, A7)."""
    return {
        "name": BASE_ARM,
        "basis": BASIS_NET,
        "is_measurement": False,
        "is_validation": False,
        "evidence": "artifact",
        "counts": dict(counts),
        "counts_source": (
            f"artefacto de #28, `arms.{BASE_ARM}.traded/no_trade/skipped/n_test`: los recuentos se "
            "**leen**, no se recalculan"
        ),
        "other_arms": {name: dict(block) for name, block in others.items()},
        "other_arms_note": (
            "los otros dos brazos del gate no operan ninguna sesion: sin operaciones no hay serie "
            "sobre la que decidir, y la base del brazo declarado es la unica admisible"
        ),
        "series": {
            "source": f"`arms.{BASE_ARM}.net_series` de #28 (#133)",
            "series_sha256": series.artifact_sha256,
            "n_sessions": series.n_sessions,
            "n_traded": series.n_traded,
            "n_no_trade": series.n_sessions - series.n_traded,
            "is_degenerate": series.is_degenerate,
            "artifact_n": series.artifact_n,
            "note": (
                "la serie **se lee** del artefacto (no se re-deriva con una formula duplicada); su "
                "digest se recomprueba desde su propio cuerpo y no se publica si no cuadra (A6, A7)"
            ),
        },
        "read_counts_match": (
            series.n_traded == int(cast("float", counts["traded"]))
            and series.n_sessions - series.n_traded == int(cast("float", counts["no_trade"]))
        ),
    }


def _arm_counts(payload: Mapping[str, object], arm: str) -> dict[str, object]:
    """Los recuentos **leidos** del artefacto de #28 para ese brazo (A6)."""
    arms = _mapping(payload.get("arms"), where="arms")
    block = _mapping(arms.get(arm), where=f"arms.{arm}")
    return {
        "traded": int(_number(block, "traded", where=f"arms.{arm}")),
        "no_trade": int(_number(block, "no_trade", where=f"arms.{arm}")),
        "skipped": int(_number(block, "skipped", where=f"arms.{arm}")),
        "n_test": int(_number(block, "n_test", where=f"arms.{arm}")),
    }


def _provenance(artifact: InputArtifact) -> dict[str, object]:
    """Procedencia del artefacto consumido, **sin** rutas que dependan del ``--reports-dir``.

    Se publica el **nombre** del fichero en vez de su ruta: la ruta cambia segun donde viva el
    directorio de informes de cada corrida y el informe tiene que ser determinista byte a byte con
    el mismo ``--as-of`` (A4).
    """
    block = artifact.provenance()
    block["path"] = artifact.path.name
    block["path_note"] = (
        "se publica el **nombre** del fichero: la ruta de `--reports-dir` cambia entre corridas y "
        "el informe es determinista byte a byte (A4)"
    )
    return block


def _published_artifact_block(
    artifact: InputArtifact, *, live_report_sha256: str
) -> dict[str, object]:
    """El artefacto **publicado** de #28: su identidad y el estado de su bloque neto (A5, A7).

    Este informe **no** consume sus numeros (re-deriva el pipeline en vivo, el precedente de #93):
    lo publica para que se vea de que artefacto se parte y, en particular, si su bloque neto es
    anterior a #133 —en cuyo caso refrescarlo es #108—.
    """
    state = _net_state(artifact.payload)
    return {
        "name": artifact.path.name,
        "sha256": artifact.sha256,
        "report_sha256": artifact.report_sha256,
        "generated_at": artifact.generated_at,
        "net_metrics_state": state,
        "is_net_computable": state == NET_METRICS_STATE,
        "rederived_report_sha256": live_report_sha256,
        "rederived_matches_report_sha256": artifact.report_sha256 == live_report_sha256,
        "note": (
            "el artefacto publicado se **consume en solo lectura** para su procedencia; los"
            "numeros de este informe salen de la **re-derivacion en vivo** del mismo pipeline con"
            "el mismo `as_of` (`write=False`), porque el artefacto publicado puede ser anterior a"
            "#133 (refrescarlo es #108)"
        ),
    }


def _regeneration_block(
    previous: Mapping[str, object], payload: Mapping[str, object], *, name: str
) -> dict[str, object]:
    """La diferencia declarada frente al informe previo de este informe (#90).

    Lo unico que cambia al regenerar es el puntero al artefacto de #28 consumido y los numeros de
    la base neta, que se recalculan: el bloque lo declara con banderas **medidas**, no supuestas.
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
        regeneration_delta.pointer_delta(previous, payload, path=("published_artifact", "sha256")),
    ]
    previous_gate = previous.get("gate")
    gate = payload.get("gate")
    previous_rows = (
        cast("dict[str, object]", previous_gate).get("n_rows")
        if isinstance(previous_gate, dict)
        else None
    )
    gate_aggregate = bool(
        isinstance(previous_gate, dict)
        and isinstance(gate, dict)
        and cast("dict[str, object]", previous_gate).get("aggregate")
        == cast("dict[str, object]", gate).get("aggregate")
    )
    previous_net = previous.get("net")
    net = payload.get("net")
    assumed = bool(
        isinstance(previous_net, dict)
        and isinstance(net, dict)
        and cast("dict[str, object]", previous_net).get("assumed_slippage_pct")
        == cast("dict[str, object]", net).get("assumed_slippage_pct")
    )
    unchanged = {
        "phase2_ready": previous.get("phase2_ready") == payload.get("phase2_ready"),
        "gate_aggregate": gate_aggregate,
        "assumed_slippage_pct": assumed,
        "n_rows": isinstance(gate, dict)
        and cast("dict[str, object]", gate).get("n_rows") == previous_rows,
    }
    return regeneration_delta.pointer_deltas_block(
        previous_name=name, pointers=pointers, unchanged=unchanged
    )


# ─────────────────────────────────────────────────────────────────────────────
# El informe: payload canonico, hash y escritura (A4)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class NetReport:
    """El informe: payload canonico, hash y los objetos que lo produjeron."""

    as_of: datetime
    report_date: str
    payload: dict[str, object]
    report_sha256: str
    reports_dir: Path
    series: NetSeries
    pipeline: InputArtifact
    model: InputArtifact

    @property
    def report_stem(self) -> str:
        """Nombre base del informe: ``phase2_net_<AAAA-MM-DD>``."""
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


# ─────────────────────────────────────────────────────────────────────────────
# El veredicto: re-derivar el pipeline, evaluar las nueve filas y agregar (A1, A3, A5-A13)
# ─────────────────────────────────────────────────────────────────────────────
def analyse(
    *,
    store: Store,
    reports_dir: Path,
    as_of: datetime,
    write: bool = True,
    previous_artifact: Path | None = None,
) -> NetReport:
    """Emite el veredicto de Fase 2 sobre la base neta bajo el supuesto declarado (A1, A5-A13).

    ``as_of`` es **obligatorio** y es el unico instante de la corrida: ninguna ruta consulta el
    reloj. Los artefactos de #28/#26 se consumen **en solo lectura** y el pipeline se **re-deriva
    en vivo** con el mismo ``as_of`` (el precedente de #93) para leer la serie neta bajo el supuesto
    que #133 publica; ``write=False`` no escribe **nada**. ``previous_artifact`` es la ruta del
    informe anterior de este mismo informe: cuando se declara, el payload publica el delta de
    puntero y las invariantes de la regeneracion (#90).
    """
    moment = _as_utc(as_of)
    previous = regeneration_delta.load_previous(previous_artifact)
    previous_name = regeneration_delta.artifact_name(previous_artifact)
    table = load_kill_table()
    published = load_input_artifact(reports_dir, PIPELINE_CLASS, store=store)
    model = load_input_artifact(reports_dir, MODEL_CLASS, store=store)
    live = pipeline_report.analyse(store=store, reports_dir=reports_dir, as_of=moment, write=False)
    live_payload = live.payload
    labels = _assumption_labels(live_payload)
    series = net_series_of_run(run=live.arm(BASE_ARM).run, payload=live_payload)
    if series.is_degenerate:
        raise DegenerateNetSeriesError(
            f"la serie neta del brazo {BASE_ARM} no tiene sesiones o es todo ceros "
            f"({series.n_sessions} sesiones, {series.n_traded} operadas): sin dispersion no hay "
            "intervalo que estimar y no se publica un `0` en su lugar (A13, A14)"
        )
    n_bootstrap = DEFAULT_BOOTSTRAP_SAMPLES
    confidence_level = DEFAULT_CONFIDENCE_LEVEL
    intervals = _net_intervals(series, n_bootstrap=n_bootstrap, confidence_level=confidence_level)
    reproduction = reproduce_net_metrics(series=series, intervals=intervals, payload=live_payload)
    beats, beats_detail = _beats_block(
        series=series, report=live, n_bootstrap=n_bootstrap, confidence_level=confidence_level
    )
    net = _net_block(
        labels=labels,
        series=series,
        intervals=intervals,
        beats=beats,
        beats_detail=beats_detail,
        reproduction=reproduction,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )
    scenario = _mapping(live_payload.get("scenario"), where="scenario")
    cost_pct = Decimal(str(scenario.get("cost_basis_pct")))
    cost_source = str(scenario.get("cost_provenance"))
    stars = p_star_block(cost_pct=cost_pct, cost_source=cost_source)
    # A #29 se le pasa una **vista** del payload con el bloque neto sustituido: su maquinaria queda
    # intacta y las filas de la base neta se evaluan contra los numeros de este informe.
    view: dict[str, object] = {**live_payload, "net_metrics": net}
    evaluated = evaluate_criteria(pipeline=view, model=model.payload, table=table, stars=stars)
    _beat_liston_b(evaluated)
    decided_r_pct = str(scenario.get("r_pct", ""))
    rows = [
        _main_row(evaluated=evaluated[0], net=net, stars=stars, decided_r_pct=decided_r_pct),
        *evaluated[1:],
    ]
    gate = gate_block(rows)
    verdict = resolve_verdict(aggregate=str(gate["aggregate"]), criteria=rows)
    ready = str(gate["aggregate"]) == str(GateVerdict.PASS)
    counts = _arm_counts(live_payload, BASE_ARM)
    others = {name: _arm_counts(live_payload, name) for name in (ARM_OFICIAL, ARM_ESCENARIO)}
    payload: dict[str, object] = {
        "analysis": ANALYSIS,
        "task": TASK,
        "title": TITLE,
        "generated_at": moment.isoformat(),
        "report_date": moment.date().isoformat(),
        "hash_format": REPORT_HASH_FORMAT,
        "basis": BASIS_NET,
        "is_measurement": False,
        "is_validation": False,
        "evidence": "artifact",
        "phase2_ready": ready,
        "phase2_ready_rule": VERDICT_RULE,
        "net_basis_rule": NET_BASIS_RULE,
        "hit_half_rule": HIT_HALF_RULE,
        "beats_rule": BEATS_RULE,
        "verdict_rule": VERDICT_RULE,
        "base_arm": _base_arm_block(counts=counts, others=others, series=series),
        "provenance": {
            "pipeline": _provenance(published),
            "model_comparison": _provenance(model),
        },
        "published_artifact": _published_artifact_block(
            published, live_report_sha256=live.report_sha256
        ),
        "criteria_source": dict(table.source),
        "kill_criteria": [row.as_dict() for row in table.rows],
        "p_star": stars,
        "net": net,
        "criteria": rows,
        "gate": gate,
        "verdict": verdict,
        "limitations": list(REPORT_LIMITATIONS),
        "does_not_do": [dict(entry) for entry in REPORT_DOES_NOT_DO],
        "follow_ups": [dict(entry) for entry in FOLLOW_UPS],
    }
    if previous is not None and previous_name is not None:
        payload["regeneration"] = _regeneration_block(previous, payload, name=previous_name)
    report = NetReport(
        as_of=moment,
        report_date=moment.date().isoformat(),
        payload=payload,
        report_sha256=_digest(payload),
        reports_dir=reports_dir,
        series=series,
        pipeline=published,
        model=model,
    )
    if write:
        json_path, markdown_path = report.write(reports_dir)
        logger.info("veredicto sobre la base neta: {} y {}", json_path, markdown_path)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Informe en prosa (A1, A15)
# ─────────────────────────────────────────────────────────────────────────────
def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Tabla Markdown determinista."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_markdown(report: NetReport) -> str:
    """El informe en Markdown: el neto, las nueve filas, el veredicto y sus limites (A1)."""
    payload = report.payload
    net = _mapping(payload["net"], where="net")
    gate = _mapping(payload["gate"], where="gate")
    verdict = _mapping(payload["verdict"], where="verdict")
    base_arm = _mapping(payload["base_arm"], where="base_arm")
    counts = _mapping(base_arm["counts"], where="base_arm.counts")
    series_block = _mapping(base_arm["series"], where="base_arm.series")
    provenance = _mapping(payload["provenance"], where="provenance")
    published = _mapping(payload["published_artifact"], where="published_artifact")
    binding = _mapping(
        _mapping(payload["p_star"], where="p_star")["binding"], where="p_star.binding"
    )
    bootstrap = _mapping(net["bootstrap"], where="net.bootstrap")
    reproduction = _mapping(net["reproduction"], where="net.reproduction")
    sharpe = _mapping(net["sharpe"], where="net.sharpe")

    def rate_row(item: object) -> list[str]:
        """La fila de una tasa de acierto, con su intervalo y su denominacion."""
        block = _mapping(item, where="net.hit_rate")
        return [
            str(block.get("denominator")),
            f"{float(cast('float', block['estimate'])):.6f}",
            f"{float(cast('float', block['lower'])):.6f}",
            f"{float(cast('float', block['upper'])):.6f}",
            str(block.get("wins_fraction")),
        ]

    def beat_row(name: str) -> list[str]:
        """La fila de una comparacion, con su diferencia neta y su estado."""
        detail = _mapping(
            _mapping(net["beats_detail"], where="net.beats_detail")[name],
            where=f"net.beats_detail.{name}",
        )
        states = _mapping(net["beats"], where="net.beats")
        if not detail.get("decided"):
            return [name, f"`{states[name]}`", "no decidida", "-", "-", str(detail.get("issue"))]
        difference = _mapping(detail["difference_pct"], where=f"beats_detail.{name}")
        return [
            name,
            f"`{states[name]}`",
            f"{float(cast('float', difference['estimate'])):.6f}",
            f"{float(cast('float', difference['lower'])):.6f}",
            f"{float(cast('float', difference['upper'])):.6f}",
            str(detail.get("seed")),
        ]

    lines: list[str] = [
        "# Veredicto de Fase 2 sobre la base neta bajo el supuesto declarado",
        "",
        f"- **Tarea**: {payload['task']}",
        f"- **Instante declarado (`as_of`)**: `{payload['generated_at']}`",
        f"- **`report_sha256`**: `{report.report_sha256}`",
        f"- **Brazo base**: `{base_arm['name']}` (`traded` {counts['traded']} / "
        f"`no_trade` {counts['no_trade']})",
        f"- **`phase2_ready`**: `{str(payload['phase2_ready']).lower()}` · "
        f"**agregado**: `{gate['aggregate']}` (`n_rows` = {gate['n_rows']}) · "
        f"**veredicto**: `{verdict['state']}`",
        "",
        f"> **Supuesto, no medicion.** `is_measurement: {str(net['is_measurement']).lower()}`, "
        f"`is_validation: {str(net['is_validation']).lower()}`, `basis: {net['basis']}`. "
        f"{payload['net_basis_rule']}",
        "",
        "## La lectura neta del brazo base",
        "",
    ]
    lines += _table(
        ["denominacion", "`estimate`", "`lower`", "`upper`", "aciertos"],
        [rate_row(net["hit_rate_per_trade"]), rate_row(net["hit_rate"])],
    )
    lines += [
        "",
        f"- Sharpe neto: `estimate` {float(cast('float', sharpe['estimate'])):.6f}, `lower` "
        f"{float(cast('float', sharpe['lower'])):.6f}, `upper` "
        f"{float(cast('float', sharpe['upper'])):.6f} (`n` {sharpe['n']})",
        f"- `p*` vinculante: `R` = {binding['r_pct']} % ⇒ `p*` = {binding['p_star_pct']} %",
        f"- Reproduccion del artefacto de #28: `{reproduction['reproduces']}` "
        f"(serie `{series_block['series_sha256']}`)",
        f"- Bootstrap: `n_bootstrap` = {bootstrap['n_bootstrap']}, `confidence_level` = "
        f"{bootstrap['confidence_level']}, `seed` = {bootstrap['seed']}",
        "",
        f"> {payload['hit_half_rule']}",
        "",
        "## Comparaciones sobre la base neta",
        "",
    ]
    lines += _table(
        ["fila", "estado", "diferencia", "`lower`", "`upper`", "semilla / issue"],
        [beat_row(name) for name in BEATS_ROWS],
    )
    lines += [
        "",
        f"> {payload['beats_rule']}",
        "",
        "## Las nueve filas de §11.6",
        "",
    ]
    lines += _table(
        ["#", "fila", "estado", "`code`", "`basis`"],
        [
            [
                str(_mapping(item, where="criteria")["row_index"]),
                str(_mapping(item, where="criteria")["kind"]),
                f"`{_mapping(item, where='criteria')['state']}`",
                f"`{_mapping(item, where='criteria')['code']}`",
                f"`{_mapping(item, where='criteria')['basis']}`",
            ]
            for item in cast("list[object]", payload["criteria"])
        ],
    )
    lines.append("")
    lines += ["", "## Procedencia", ""]
    for key in ("pipeline", "model_comparison"):
        block = _mapping(provenance[key], where=key)
        lines.append(
            f"- **{key}**: `{block['path']}` (sha256 `{block['sha256']}`, `report_sha256` "
            f"`{block['report_sha256']}`, `generated_at` `{block['generated_at']}`)"
        )
    lines += [
        f"- **Artefacto publicado**: `{published['name']}` (sha256 `{published['sha256']}`, "
        f"bloque neto `{published['net_metrics_state']}`, computable "
        f"`{published['is_net_computable']}`)",
        f"- **Serie neta**: `{series_block['series_sha256']}` "
        f"({series_block['n_sessions']} sesiones, {series_block['n_traded']} operadas)",
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
                str(_mapping(entry, where="follow_ups")["issue"]),
                str(_mapping(entry, where="follow_ups")["topic"]),
                str(_mapping(entry, where="follow_ups")["why"]),
            ]
            for entry in cast("list[object]", payload["follow_ups"])
        ],
    )
    if "regeneration" in payload:
        block = _mapping(payload["regeneration"], where="regeneration")
        lines += [
            "",
            "## Regeneracion",
            "",
            f"- Informe previo: `{block.get('previous_artifact')}`",
            f"- Invariantes: `{block.get('unchanged')}`",
        ]
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
    """Punto de entrada del veredicto sobre la base neta.

    Codigos de salida: ``0`` = informe escrito (aunque el veredicto sea ``fail`` o
    ``not_evaluable``, que son resultados legitimos y declarados); ``2`` = falta o no es valido
    ``--as-of``, falta o es ambiguo un artefacto de #28/#26, la tabla de §11.6 no se puede leer, el
    pipeline no se puede re-derivar, la serie neta no reproduce el artefacto o su base no es
    comparable ⇒ **no se escribe nada** y el motivo sale por ``stderr``.
    """
    parser = argparse.ArgumentParser(
        prog="cfdtrader.analysis.phase2_net",
        description="Veredicto de Fase 2 sobre la base neta bajo el supuesto declarado",
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
            "ruta del informe previo: si se declara, el payload publica el delta de puntero y las "
            "invariantes de la regeneracion (#90)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        moment = _parse_as_of(cast("str | None", args.as_of))
    except (Phase2NetError, Phase2ReportError) as error:
        print(f"no se puede emitir el veredicto sobre la base neta: {error}", file=sys.stderr)
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
        Phase2NetError,
        Phase2ReportError,
        PipelineReportError,
        BacktestReportError,
        regeneration_delta.RegenerationError,
    ) as error:
        print(f"no se puede emitir el veredicto sobre la base neta: {error}", file=sys.stderr)
        return 2

    net = cast("dict[str, object]", report.payload["net"])
    gate = cast("dict[str, object]", report.payload["gate"])
    verdict = cast("dict[str, object]", report.payload["verdict"])
    logger.info(
        "base neta: {} ({}); agregado {} (n_rows {}); phase2_ready {}; veredicto {}; "
        "report_sha256 = {}",
        net["basis"],
        net["assumed_slippage_pct"],
        gate["aggregate"],
        gate["n_rows"],
        report.payload["phase2_ready"],
        verdict["state"],
        report.report_sha256,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
