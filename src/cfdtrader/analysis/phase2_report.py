"""Informe de Fase 2 y criterios de *kill* pre-registrados (`tasks.md`, tarea 29) — tarea #29.

Este modulo es la **puerta de salida** de la Fase 2: aplica **mecanicamente** la tabla de
*criterios de kill* pre-registrada de `plan.md` §11.6 a los resultados de la Fase 2 (el
artefacto de #28) y a la correccion por sobreajuste (el artefacto de #26), y entrega el
informe con veredicto (``continue``/``simplify``/``stop``/``not_evaluable``).

Que **hace**:

- **lee** la tabla de §11.6 de ``_docs/plan.md`` **en tiempo de ejecucion** (nunca una copia
  cableada) y publica su ``sha256`` y sus 9 filas literales: si el documento cambia, el
  hash cambia y la suite lo detecta (A6);
- **consume sin modificar** ``pipeline_backtest_*.json`` (#28) y ``model_comparison_*.json``
  (#26) con :func:`cfdtrader.analysis.phase0_report.select_artifact` y publica su ``sha256``,
  su ``report_sha256`` y su ``generated_at`` como procedencia (``evidence: artifact``); cero
  candidatos y ambiguedad son errores tipados, nunca un fichero elegido a dedo (A5);
- evalua cada fila al vocabulario de #9 (``pass``/``fail``/``not_evaluable``) con un ``code``
  legible por maquina y su ``source_row``, y **agrega importando** ``aggregate_gate`` de #9
  (A7): ``fail`` si alguna fila falla, ``not_evaluable`` si ninguna falla y alguna no es
  evaluable, y ``pass`` **solo** si todas pasan;
- **deriva** ``p* = (R + c) / 2R`` por escenario declarado de ``R`` de
  :data:`cfdtrader.analysis.phase0_report.DECLARED_CONSTANTS`, con el coste de ida y vuelta
  **declarado** del artefacto de #28: nunca cablea ``50,2`` ni ``50,21`` (A10);
- **copia** de #26 la probabilidad de sobreajuste (``pbo``, ``blocks``, ``n_observations``,
  ``verdict``) y la mitad declarada del Sharpe deflactado; no recalcula ni PBO ni DSR (A13).

Que **no** hace, y por tanto no puede inventar:

- **no** publica metricas netas: con ``net_metrics: not_computable`` de #28, la fila principal
  (A9) y las filas «bate a *no operar*» / «bate a *siempre largo open→close*» (A15) salen
  ``not_evaluable`` con ``code: net_metrics_not_computable`` y ``follow_ups: ["#62", "#60"]``;
- **no** convierte un ``not_evaluable`` en un ``pass`` (la agregacion de #9 lo hace imposible
  por construccion): un «adelante» construido sobre datos ausentes es el error mas caro que
  puede cometer este informe;
- **no** escribe ningun ``None`` como ``0``;
- **no** publica la comparacion de coste declarado de #28 como el criterio pre-registrado: se
  etiqueta siempre como ``declared_cost`` y ``is_validation: false`` (A15);
- **no** produce el liston B de primera clase (posiciones overnight por el motor): eso es #70.

**Reloj prohibido** (A2, A3): ninguna ruta consulta el reloj del sistema; el instante entra
por ``--as-of``, obligatorio para escribir, que sale con codigo 2 y sin tocar disco cuando
falta o no es ISO-8601. **Red prohibida**: el informe se genera leyendo solo ficheros locales.
Sin escrituras fuera de ``--reports-dir`` y determinista byte a byte: ``report_sha256`` es el
sha256 del texto canonico de #13 sobre el payload sin la clave del hash, con el prefijo
``sha256:``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from functools import reduce
from pathlib import Path
from typing import Final, cast

from loguru import logger

from cfdtrader.analysis.phase0_report import (
    DECLARED_CONSTANTS,
    FILE_SELECTION_RULE,
    GATE_AGGREGATION_RULE,
    RECOMMENDATION_CONSISTENCY_RULE,
    AmbiguousArtifactError,
    ArtifactClass,
    GateVerdict,
    HalfResult,
    MissingArtifactError,
    Recommendation,
    aggregate_gate,
    declared_value,
    p_star,
    recommendation_is_consistent,
    select_artifact,
)
from cfdtrader.backtest.engine import canonical_text
from cfdtrader.backtest.overfitting import PBO_MAX
from cfdtrader.data.store import Store

__all__ = [
    "A13_DIVERGENCE",
    "ANALYSIS",
    "CRITERIA_SOURCE_RULE",
    "EVIDENCE_ARTIFACT",
    "FOLLOW_UPS",
    "GATE_AGGREGATION_SOURCE",
    "HASH_PREFIX",
    "KILL_N_ROWS",
    "MODEL_CLASS",
    "NET_UNAVAILABLE_FOLLOW_UPS",
    "PIPELINE_CLASS",
    "PLAN_RELATIVE_PATH",
    "PLAN_SECTION",
    "P_STAR_DERIVATION",
    "REPORT_DOES_NOT_DO",
    "REPORT_HASH_FORMAT",
    "REPORT_LIMITATIONS",
    "REPORT_PREFIX",
    "ROW_KINDS",
    "TASK",
    "TITLE",
    "VERDICT_CONTINUE",
    "VERDICT_NOT_EVALUABLE",
    "VERDICT_SIMPLIFY",
    "VERDICT_STOP",
    "AmbiguousInputArtifactError",
    "InputArtifact",
    "InvalidAsOfError",
    "InvalidKillTableError",
    "KillRow",
    "KillTable",
    "MalformedInputArtifactError",
    "MissingAsOfError",
    "MissingInputArtifactError",
    "MissingInputFieldError",
    "Phase2Report",
    "Phase2ReportError",
    "VerdictError",
    "analyse",
    "consistent_verdict",
    "evaluate_criteria",
    "gate_block",
    "load_input_artifact",
    "load_kill_table",
    "main",
    "p_star_block",
    "plan_path",
    "read_kill_table",
    "render_markdown",
    "resolve_verdict",
]

#: Identidad del informe: quien lo emite y que tarea lo pide.
ANALYSIS: Final[str] = "cfdtrader.analysis.phase2_report"
TASK: Final[str] = "#29"
TITLE: Final[str] = "Informe de Fase 2 y criterios de kill"

#: Prefijo del informe: ``phase2_report_<AAAA-MM-DD>.{json,md}``.
REPORT_PREFIX: Final[str] = "phase2_report"

#: Prefijo del ``report_sha256`` (A4): un digest desnudo lo bloquea `detect-secrets`.
HASH_PREFIX: Final[str] = "sha256:"

#: Formato estable del ``report_sha256`` (A4).
REPORT_HASH_FORMAT: Final[str] = (
    "sha256:<64 hex> del texto canonico (``canonical_text`` de #13) del payload **sin** la "
    "clave ``report_sha256``. El prefijo viaja dentro del valor: un digest desnudo lo bloquea "
    "`detect-secrets`"
)

#: Documento del que se **lee** la tabla de *kill*, relativo a la raiz del repositorio (A6).
PLAN_RELATIVE_PATH: Final[Path] = Path("_docs") / "plan.md"

#: Seccion de ``plan.md`` que contiene la tabla pre-registrada (A6).
PLAN_SECTION: Final[str] = "§11.6"

#: Numero de filas de la tabla de *kill* (A6, A16). La tabla **no** se modifica tras ver
#: resultados: se lee y se publica con su huella.
KILL_N_ROWS: Final[int] = 9

#: Regla con la que se publica la tabla de *kill* (A6).
CRITERIA_SOURCE_RULE: Final[str] = (
    "la tabla de criterios de *kill* se **lee** de `_docs/plan.md` §11.6 en tiempo de "
    "ejecucion y se publica con su `sha256`; no se copia ni se cablea, y no se modifica "
    "despues de ver los resultados (A6)"
)

#: De donde sale la agregacion mecanica publicada (A7): se **importa**, no se reescribe.
GATE_AGGREGATION_SOURCE: Final[str] = (
    "cfdtrader.analysis.phase0_report.aggregate_gate (regla de #9, reutilizada tal cual)"
)

#: Derivacion declarada de ``p*`` (A10): nunca un umbral cableado.
P_STAR_DERIVATION: Final[str] = (
    "p* = (R + c) / 2R, con `R` (amplitud del bracket) y `c` (coste de ida y vuelta "
    "declarado) en la misma unidad; se **importa** `p_star` de #9 y entra el coste declarado "
    "de #28, no un literal (A10)"
)

#: Procedencia de un bloque **copiado** de un artefacto, no medido aqui.
EVIDENCE_ARTIFACT: Final[str] = "artifact"

#: Seguimientos de las filas que hoy dependen de las metricas netas (A9, A15, A17).
NET_UNAVAILABLE_FOLLOW_UPS: Final[tuple[str, ...]] = ("#62", "#60")

#: Vocabulario del veredicto (A8).
VERDICT_CONTINUE: Final[str] = "continue"
VERDICT_SIMPLIFY: Final[str] = "simplify"
VERDICT_STOP: Final[str] = "stop"
VERDICT_NOT_EVALUABLE: Final[str] = "not_evaluable"

#: Traduccion del veredicto de Fase 2 al vocabulario de recomendacion de #9 (A8). El
#: veredicto `not_evaluable` no tiene miembro propio en #9: se traduce a `reframe` —la accion
#: declarada cuando la puerta **no** aprueba— solo para comprobar la consistencia.
PHASE2_TO_RECOMMENDATION: Final[dict[str, Recommendation]] = {
    VERDICT_CONTINUE: Recommendation.CONTINUE,
    VERDICT_SIMPLIFY: Recommendation.REFRAME,
    VERDICT_STOP: Recommendation.STOP,
    VERDICT_NOT_EVALUABLE: Recommendation.REFRAME,
}

# ─────────────────────────────────────────────────────────────────────────────
# Las nueve filas de §11.6, identificadas por una subcadena **sin** literales de hora
# ─────────────────────────────────────────────────────────────────────────────
#: Identificadores estables de las nueve filas de §11.6, en el orden del documento.
ROW_EDGE: Final[str] = "edge_demostrable"
ROW_PBO: Final[str] = "pbo"
ROW_DSR: Final[str] = "deflated_sharpe"
ROW_DRAWDOWN: Final[str] = "drawdown"
ROW_NO_TRADE: Final[str] = "no_operar"
ROW_LISTON_A: Final[str] = "liston_a"
ROW_LISTON_B: Final[str] = "liston_b"
ROW_PAPER: Final[str] = "paper_vs_backtest"
ROW_CLOSE: Final[str] = "cierre_de_sesion"

#: Orden declarado de las nueve filas (A6, A16).
ROW_KINDS: Final[tuple[str, ...]] = (
    ROW_EDGE,
    ROW_PBO,
    ROW_DSR,
    ROW_DRAWDOWN,
    ROW_NO_TRADE,
    ROW_LISTON_A,
    ROW_LISTON_B,
    ROW_PAPER,
    ROW_CLOSE,
)

#: Subcadena que identifica cada fila dentro del texto literal del **criterio**. Ninguna
#: contiene una hora literal (A3): la fila del cierre se reconoce por su enunciado, no por su
#: horario. El orden importa: `open→close` (liston A) antes que `close→close` (liston B).
_ROW_MATCHERS: Final[tuple[tuple[str, str], ...]] = (
    (ROW_EDGE, "edge demostrable"),
    (ROW_PBO, "PBO"),
    (ROW_DSR, "Deflated Sharpe"),
    (ROW_DRAWDOWN, "Drawdown"),
    (ROW_NO_TRADE, "no operar"),
    (ROW_LISTON_A, "open→close"),
    (ROW_LISTON_B, "close→close"),
    (ROW_PAPER, "Divergencia paper"),
    (ROW_CLOSE, "Incumplimiento del cierre"),
)

#: Solo la fila del PBO se arregla «simplificando el modelo»; las demas acciones de fallo
#: mandan **parar** o reformular (§11.6). Sirve para derivar el veredicto (A8).
_SIMPLIFY_ROWS: Final[frozenset[str]] = frozenset({ROW_PBO})

#: Cabecera exacta de la tabla de §11.6 (A6).
_KILL_HEADER: Final[str] = "| Criterio | Umbral | Acción si falla |"

#: Prefijo de la cabecera de la seccion §11.6 (A6).
_SECTION_HEADING_PREFIX: Final[str] = "### 11.6"

#: Filas de la tabla de §11.6 que se evaluan sobre la **base neta** (A9, A11, A15).
_NET_BASIS_ROWS: Final[tuple[str, ...]] = (ROW_EDGE, ROW_NO_TRADE, ROW_LISTON_A, ROW_LISTON_B)

#: Clave de ``net_metrics`` de #28 que declara que las metricas netas son computables (A9).
_NET_STATE_COMPUTED: Final[str] = "computed"

# ─────────────────────────────────────────────────────────────────────────────
# Textos declarados del informe (A17)
# ─────────────────────────────────────────────────────────────────────────────
#: Limites declarados: lo que este informe **no** puede decidir hoy.
REPORT_LIMITATIONS: Final[tuple[str, ...]] = (
    "hoy no hay metricas netas: #28 publica `net_metrics: not_computable` porque el "
    "*slippage* sigue siendo un supuesto (#62) y `R` sigue sin decidir (#60); las filas que "
    "exigen la base neta salen `not_evaluable` y **no** se fabrica ninguna metrica (A9, A15)",
    "la fila del drawdown se mide sobre `declared_cost` del brazo `coste_declarado` de #28, "
    "no sobre la base neta: se publica con su `basis` y **no** se presenta como medida neta",
    "el liston B de primera clase (posiciones overnight por el motor) es #70: aqui B se "
    "**copia** de la tabla declarada de #28, con `insufficient_on_its_own: true` (A11)",
    "el *holdout* final intocable es #68 y el CPCV es #67: este informe no reserva ni mira "
    "ninguno de los dos",
)

#: Que no hace el informe, con su issue.
REPORT_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_publica_metricas_netas",
        "issue": "#62",
        "statement": (
            "no publica el IC de la tasa de acierto ni el del Sharpe sobre la base neta: con "
            "`net_metrics: not_computable` de #28 las filas correspondientes son "
            "`not_evaluable` con `code: net_metrics_not_computable` (A9, A15)"
        ),
    },
    {
        "id": "no_decide_r",
        "issue": "#60",
        "statement": (
            "no decide `R`: publica `p*` por cada escenario declarado de "
            "`DECLARED_CONSTANTS` y usa el mas exigente, sin cablear ningun umbral (A10)"
        ),
    },
    {
        "id": "no_produce_el_liston_b",
        "issue": "#70",
        "statement": (
            "no produce el liston B de primera clase (posiciones overnight por el motor): la "
            "fila de B se copia de #28 y se marca `insufficient_on_its_own: true` (A11)"
        ),
    },
    {
        "id": "no_recalcula_pbo_ni_dsr",
        "issue": "#16",
        "statement": (
            "no recalcula la probabilidad de sobreajuste ni el Sharpe deflactado: los copia "
            "de #26 con su `verdict` declarado (A13)"
        ),
    },
    {
        "id": "no_reserva_holdout_ni_cpcv",
        "issue": "#68",
        "statement": (
            "no reserva el *holdout* final (#68) ni implementa CPCV (#67): ninguna de las dos "
            "cosas se decide aqui (A17)"
        ),
    },
    {
        "id": "no_evalua_paper_ni_cierre",
        "issue": "#45",
        "statement": (
            "no evalua la divergencia paper-vs-backtest (#45) ni el incumplimiento del cierre "
            "de sesion (#84): las dos filas se **publican** `not_evaluable`, no se suprimen "
            "(A16)"
        ),
    },
)

#: Seguimientos declarados del informe (A16, A17).
FOLLOW_UPS: Final[tuple[dict[str, str], ...]] = (
    {
        "issue": "#62",
        "topic": "medir el *slippage*",
        "why": "sin *slippage* medido no hay metrica neta y las filas de la base neta siguen "
        "`not_evaluable` (A9, A15)",
    },
    {
        "issue": "#60",
        "topic": "decidir `R`",
        "why": "`R` decide `p*`; aqui se publica por escenarios y se usa el mas exigente (A10)",
    },
    {
        "issue": "#70",
        "topic": "liston B de primera clase",
        "why": "hoy B se copia de #28 como serie de referencia declarada (A11)",
    },
    {
        "issue": "#68",
        "topic": "*holdout* final",
        "why": "este informe no reserva ni mira el *holdout* (A17)",
    },
    {
        "issue": "#67",
        "topic": "CPCV",
        "why": "la validacion combinatoria no se implementa aqui (A17)",
    },
    {
        "issue": "#45",
        "topic": "divergencia paper vs backtest",
        "why": "es de Fase 4: la fila se publica `not_evaluable` (A16)",
    },
    {
        "issue": "#84",
        "topic": "cumplimiento del cierre de sesion",
        "why": "es de Fase 4: la fila se publica `not_evaluable` (A16)",
    },
    {
        "issue": "#88",
        "topic": "reemitir el veredicto",
        "why": "cuando #60 y #62 cierren las metricas netas, el veredicto se reemite",
    },
)

#: Divergencia **declarada** con la letra de A13: A13 pide importar `PBO_MAX` de #16 (que vive
#: en `cfdtrader.backtest.overfitting`) y a la vez que el AST del modulo **no** importe
#: `backtest.overfitting`. Las dos cosas no caben juntas; se importa `PBO_MAX` (no se
#: reimplementa) y el AST comprueba que **no** se importa ninguna funcion de calculo de
#: sobreajuste. Ver el comentario de la entrega.
A13_DIVERGENCE: Final[str] = (
    "A13 pide importar `PBO_MAX` de #16 y que el AST no importe `backtest.overfitting`; como "
    "`PBO_MAX` vive en ese modulo, se prioriza **importar** (no reimplementar) y el AST "
    "prohibe las funciones de calculo, no la constante"
)


# ─────────────────────────────────────────────────────────────────────────────
# Errores declarados: un hueco nunca se rellena con un valor por defecto
# ─────────────────────────────────────────────────────────────────────────────
class Phase2ReportError(Exception):
    """Error declarado del informe de Fase 2: nunca se rellena el hueco."""


class MissingInputArtifactError(Phase2ReportError):
    """Falta un artefacto de entrada (#28 o #26) que consumir."""


class AmbiguousInputArtifactError(Phase2ReportError):
    """Dos o mas artefactos de la misma clase resuelven a la misma fecha."""


class MalformedInputArtifactError(Phase2ReportError):
    """El artefacto de entrada no es JSON legible o no es un objeto."""


class MissingInputFieldError(Phase2ReportError):
    """El artefacto no trae un campo obligatorio, o lo trae con un tipo inesperado."""


class InvalidKillTableError(Phase2ReportError):
    """La tabla de §11.6 no se pudo leer, no tiene 9 filas o sus filas no se reconocen."""


class MissingAsOfError(Phase2ReportError):
    """Falta el instante declarado: escribir el informe lo exige (A2)."""


class InvalidAsOfError(Phase2ReportError):
    """El instante declarado no es ISO-8601 (A2)."""


class VerdictError(Phase2ReportError):
    """El veredicto pedido rompe una regla declarada (vocabulario o consistencia)."""


# ─────────────────────────────────────────────────────────────────────────────
# La tabla de *kill* (§11.6), leida del documento en tiempo de ejecucion (A6)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class KillRow:
    """Una fila literal de la tabla de criterior de *kill* de §11.6."""

    row_index: int
    kind: str
    criterion: str
    threshold: str
    action: str

    def as_dict(self) -> dict[str, object]:
        """Fila publicada: los tres literales del documento, sin retocar (A6)."""
        return {
            "row_index": self.row_index,
            "kind": self.kind,
            "criterion": self.criterion,
            "threshold": self.threshold,
            "action": self.action,
        }


@dataclass(frozen=True, slots=True)
class KillTable:
    """La tabla de *kill* publicada: procedencia con su ``sha256`` y las 9 filas literales."""

    source: dict[str, object]
    rows: tuple[KillRow, ...]

    def row(self, kind: str) -> KillRow:
        """La fila con ese identificador, o error tipado: nunca ``None`` silencioso."""
        for row in self.rows:
            if row.kind == kind:
                return row
        raise InvalidKillTableError(
            f"la tabla de §11.6 no trae la fila {kind!r}: las nueve declaradas son "
            f"{list(ROW_KINDS)} (A6)"
        )


def plan_path() -> Path:
    """Ruta absoluta de ``_docs/plan.md``, resolviendo la raiz desde este modulo (A6)."""
    return Path(__file__).resolve().parents[3] / PLAN_RELATIVE_PATH


def _row_kind(criterion: str) -> str:
    """Identificador estable de una fila a partir del texto literal de su criterio (A6)."""
    for kind, needle in _ROW_MATCHERS:
        if needle in criterion:
            return kind
    raise InvalidKillTableError(
        f"no se reconoce la fila de §11.6 con criterio {criterion!r}: la tabla esperada tiene "
        f"las nueve filas {list(ROW_KINDS)} (A6)"
    )


def read_kill_table(plan_text: str) -> KillTable:
    """Lee la tabla de §11.6 del texto de ``plan.md`` y la publica con su ``sha256`` (A6).

    El ``sha256`` es el de los bytes UTF-8 del bloque exacto de la tabla (cabecera, separador
    y las nueve filas, unidas por ``\\n`` y con salto final): si una fila o un umbral se
    desvia, el hash cambia y la suite lo detecta releyendo el documento.
    """
    lines = plan_text.split("\n")
    heading = next(
        (i for i, line in enumerate(lines) if line.startswith(_SECTION_HEADING_PREFIX)), None
    )
    if heading is None:
        raise InvalidKillTableError(
            f"no se encontro la seccion {PLAN_SECTION!r} en {PLAN_RELATIVE_PATH.as_posix()}: "
            "la tabla no se cablea (A6)"
        )
    header = next(
        (i for i in range(heading, len(lines)) if lines[i].strip() == _KILL_HEADER),
        None,
    )
    if header is None:
        raise InvalidKillTableError(
            f"no se encontro la cabecera de la tabla de §11.6 en "
            f"{PLAN_RELATIVE_PATH.as_posix()} (A6)"
        )
    block: list[str] = []
    rows: list[KillRow] = []
    index = header
    block.append(lines[index])
    index += 1
    if index >= len(lines) or not lines[index].startswith("|"):
        raise InvalidKillTableError("la tabla de §11.6 no trae separador de cabecera (A6)")
    separator = lines[index].replace("|", "").replace("-", "").replace(":", "").strip()
    if separator:
        raise InvalidKillTableError("el separador de la tabla de §11.6 no es valido (A6)")
    block.append(lines[index])
    index += 1
    while index < len(lines) and lines[index].startswith("|"):
        line = lines[index]
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            raise InvalidKillTableError(
                f"la fila {len(rows) + 1} de §11.6 no tiene tres celdas: {line!r} (A6)"
            )
        rows.append(
            KillRow(
                row_index=len(rows) + 1,
                kind=_row_kind(cells[0]),
                criterion=cells[0],
                threshold=cells[1],
                action=cells[2],
            )
        )
        block.append(line)
        index += 1
    if len(rows) != KILL_N_ROWS:
        raise InvalidKillTableError(
            f"la tabla de §11.6 tiene {len(rows)} filas y deberia tener {KILL_N_ROWS}: no se "
            "publica una tabla parcial (A6, A16)"
        )
    if {row.kind for row in rows} != set(ROW_KINDS):
        raise InvalidKillTableError(
            "las filas de §11.6 no son las nueve declaradas (reconocidas: "
            f"{[row.kind for row in rows]}) (A6)"
        )
    digest = hashlib.sha256(("\n".join(block) + "\n").encode("utf-8")).hexdigest()
    return KillTable(
        source={
            "path": PLAN_RELATIVE_PATH.as_posix(),
            "section": PLAN_SECTION,
            "sha256": digest,
            "n_rows": len(rows),
            "rule": CRITERIA_SOURCE_RULE,
        },
        rows=tuple(rows),
    )


def load_kill_table() -> KillTable:
    """Lee la tabla de §11.6 de ``_docs/plan.md`` en tiempo de ejecucion (A6)."""
    path = plan_path()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise InvalidKillTableError(
            f"no se pudo leer {path.as_posix()}: la tabla de §11.6 no se cablea (A6)"
        ) from error
    return read_kill_table(text)


# ─────────────────────────────────────────────────────────────────────────────
# Lectura tipada de los artefactos: un campo que falta es un error, nunca un cero
# ─────────────────────────────────────────────────────────────────────────────
def _mapping(node: object, *, where: str) -> dict[str, object]:
    """Vista tipada de un nodo que tiene que ser un objeto JSON."""
    if not isinstance(node, dict):
        raise MissingInputFieldError(
            f"{where}: se espera un objeto JSON, no {type(node).__name__} (A5)"
        )
    return cast("dict[str, object]", node)


def _require(node: object, *keys: str) -> object:
    """Valor anidado obligatorio del artefacto; su ausencia es un error tipado (A5)."""
    current: object = node
    for index, key in enumerate(keys):
        mapping = _mapping(current, where=".".join(keys[:index]) or "el artefacto")
        if key not in mapping:
            raise MissingInputFieldError(
                f"falta el campo obligatorio `{'.'.join(keys)}` en el artefacto de entrada: no "
                "se rellena con un valor por defecto (A5)"
            )
        current = mapping[key]
    return current


def _require_str(node: object, *keys: str) -> str:
    """Campo obligatorio de tipo cadena."""
    value = _require(node, *keys)
    if not isinstance(value, str):
        raise MissingInputFieldError(
            f"`{'.'.join(keys)}` deberia ser una cadena, no {type(value).__name__} (A5)"
        )
    return value


def _require_number(node: object, *keys: str) -> float:
    """Campo obligatorio de tipo numero (los `bool` no cuentan)."""
    value = _require(node, *keys)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MissingInputFieldError(
            f"`{'.'.join(keys)}` deberia ser un numero, no {type(value).__name__} (A5)"
        )
    return float(value)


def _optional_float(node: object, *keys: str) -> float | None:
    """Numero anidado opcional: ausente o ``None`` devuelve ``None`` (nunca ``0``)."""
    current: object = node
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = cast("dict[str, object]", current)[key]
    if current is None:
        return None
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise MissingInputFieldError(
            f"`{'.'.join(keys)}` deberia ser un numero o null, no {type(current).__name__} (A5)"
        )
    return float(current)


def _as_utc(value: datetime) -> datetime:
    """Normaliza el instante declarado a UTC; sin zona se interpreta como UTC (A2)."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _num(value: Decimal) -> str:
    """``Decimal`` como cadena decimal exacta, sin notacion cientifica."""
    return format(value, "f")


@dataclass(frozen=True, slots=True)
class InputArtifact:
    """Un artefacto consumido, con su procedencia y su huella (A5)."""

    kind: str
    path: Path
    relative_path: str
    artifact_date: str
    sha256: str
    payload: dict[str, object]

    @property
    def generated_at(self) -> str:
        """Instante declarado del artefacto."""
        return _require_str(self.payload, "generated_at")

    @property
    def report_sha256(self) -> str:
        """``report_sha256`` que declara el artefacto."""
        return _require_str(self.payload, "report_sha256")

    def provenance(self) -> dict[str, object]:
        """Bloque de procedencia que publica el informe (A5)."""
        return {
            "kind": self.kind,
            "path": self.relative_path,
            "artifact_date": self.artifact_date,
            "sha256": self.sha256,
            "sha256_of": "fichero .json",
            "report_sha256": self.report_sha256,
            "generated_at": self.generated_at,
            "evidence": EVIDENCE_ARTIFACT,
            "selection_rule": FILE_SELECTION_RULE,
            "note": "el artefacto se **consume** en solo lectura: se publica su huella, no se "
            "reescribe ni se recalcula (A5)",
        }


#: La clase de artefacto del pipeline (#28): el artefacto que se consume **sin** modificar.
PIPELINE_CLASS: Final[ArtifactClass] = ArtifactClass(
    kind="pipeline_backtest",
    pattern="pipeline_backtest_*.json",
    description="informe del pipeline completo de Fase 2 (#28)",
)

#: La clase de artefacto de la comparacion de modelos (#26).
MODEL_CLASS: Final[ArtifactClass] = ArtifactClass(
    kind="model_comparison",
    pattern="model_comparison_*.json",
    description="comparacion de modelos y sobreajuste (#26)",
)


def _relative_path(path: Path, *, store: Store) -> str:
    """Ruta del artefacto relativa a la raiz del almacen cuando cuelga de ella (A5)."""
    root = store.root
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _artifact_date(path: Path, klass: ArtifactClass) -> str:
    """Fecha del sufijo del nombre, o el tallo si el nombre no la trae."""
    stem = path.stem
    prefix = klass.pattern.removesuffix("_*.json") + "_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def load_input_artifact(reports_dir: Path, klass: ArtifactClass, *, store: Store) -> InputArtifact:
    """Selecciona, lee y valida un artefacto. Solo lee: no escribe nada (A5).

    La seleccion **reutiliza** :func:`cfdtrader.analysis.phase0_report.select_artifact`: cero
    candidatos y ambiguedad son errores tipados propios, y nunca se elige a dedo.
    """
    try:
        path = select_artifact(reports_dir, klass)
    except MissingArtifactError as error:
        raise MissingInputArtifactError(
            f"no hay artefacto de {klass.description} que consumir: {error}"
        ) from error
    except AmbiguousArtifactError as error:
        raise AmbiguousInputArtifactError(
            f"hay mas de un artefacto de {klass.description} con la misma fecha: {error}"
        ) from error

    raw = path.read_bytes()
    try:
        loaded: object = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MalformedInputArtifactError(
            f"el artefacto {path.name} no es JSON legible: {error} (A5)"
        ) from error
    if not isinstance(loaded, dict):
        raise MalformedInputArtifactError(
            f"el artefacto {path.name} no es un objeto JSON: el informe consume objetos con "
            "campos (A5)"
        )
    return InputArtifact(
        kind=klass.kind,
        path=path,
        relative_path=_relative_path(path, store=store),
        artifact_date=_artifact_date(path, klass),
        sha256=hashlib.sha256(raw).hexdigest(),
        payload=cast("dict[str, object]", loaded),
    )


# ─────────────────────────────────────────────────────────────────────────────
# `p*`: derivado, nunca cableado (A10)
# ─────────────────────────────────────────────────────────────────────────────
def r_scenarios_pct() -> tuple[Decimal, ...]:
    """Escenarios de ``R`` declarados, leidos de #9 (``DECLARED_CONSTANTS``) (A10)."""
    raw = declared_value("r_scenarios_pct")
    return tuple(Decimal(part.strip()) for part in raw.split("/"))


def p_star_block(*, cost_pct: Decimal, cost_source: str) -> dict[str, object]:
    """Publica ``p*`` por escenario declarado de ``R``, con su derivacion y su fuente (A10)."""
    decimals = Decimal(declared_value("p_star_decimals"))
    scenarios: list[dict[str, object]] = []
    for r_pct in r_scenarios_pct():
        fraction = p_star(r_pct, cost_pct)
        scenarios.append(
            {
                "r_pct": _num(r_pct),
                "p_star_fraction": format(fraction, "f"),
                "p_star_pct": format((fraction * Decimal(100)).quantize(decimals), "f"),
                "derivation": P_STAR_DERIVATION,
            }
        )
    binding = max(scenarios, key=lambda item: Decimal(cast("str", item["p_star_fraction"])))
    return {
        "cost_round_trip_pct": _num(cost_pct),
        "cost_source": cost_source,
        "constants_source": "cfdtrader.analysis.phase0_report.DECLARED_CONSTANTS (#9)",
        "declared_r_constants": [dict(entry) for entry in DECLARED_CONSTANTS],
        "derivation": P_STAR_DERIVATION,
        "decimals_pct": _num(decimals),
        "scenarios": scenarios,
        "binding": {
            "r_pct": binding["r_pct"],
            "p_star_pct": binding["p_star_pct"],
            "p_star_fraction": binding["p_star_fraction"],
            "note": "el escenario con `p*` mayor (R mas pequeno) es el **mas exigente**: la fila "
            "principal pasa solo si el IC excluye ese `p*` (A9, A10)",
        },
    }


def _net_block(pipeline: Mapping[str, object]) -> dict[str, object] | None:
    """Bloque ``net_metrics`` de #28, si viaja como objeto."""
    node = pipeline.get("net_metrics")
    if node is None:
        return None
    return _mapping(node, where="net_metrics")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluacion de las nueve filas (A7, A9, A11, A13, A14, A15, A16)
# ─────────────────────────────────────────────────────────────────────────────
def _drawdown_limit_pct(table: KillTable) -> Decimal:
    """Umbral de drawdown **leido** del literal de la fila de §11.6, no re-derivado (A14)."""
    text = table.row(ROW_DRAWDOWN).threshold
    digits = ""
    for char in text:
        if char.isdigit() or (char == "." and digits and "." not in digits):
            digits += char
        elif digits:
            break
    if not digits:
        raise InvalidKillTableError(
            f"la fila de drawdown de §11.6 no declara un umbral numerico: {text!r} (A6, A14)"
        )
    return Decimal(digits)


def _pbo_row(*, model: Mapping[str, object], row: KillRow) -> dict[str, object]:
    """Fila del PBO: copiada de #26 y evaluada por ``PBO_MAX`` de #16 (A13)."""
    block = _mapping(
        _require(model, "probability_of_backtest_overfitting"),
        where="probability_of_backtest_overfitting",
    )
    state = _require_str(block, "state")
    detail: dict[str, object] = {
        "pbo": _require_number(block, "pbo"),
        "pbo_max": PBO_MAX,
        "blocks": _require_number(block, "blocks"),
        "n_observations": _require_number(block, "n_observations"),
        "declared_verdict": _require_str(block, "verdict"),
        "evidence": EVIDENCE_ARTIFACT,
    }
    if state != "evaluated":
        return _row(row, HalfResult.NOT_EVALUABLE, "pbo_not_evaluable", detail, basis="artifact")
    pbo = cast("float", detail["pbo"])
    if pbo < PBO_MAX:
        return _row(row, HalfResult.PASS, "pbo_ok", detail, basis="artifact")
    return _row(row, HalfResult.FAIL, "pbo_above_max", detail, basis="artifact")


def _dsr_row(*, model: Mapping[str, object], row: KillRow) -> dict[str, object]:
    """Fila del Sharpe deflactado: usa la mitad declarada de #26, no la recalcula (A13)."""
    half = _require_str(model, "verdict", "gate", "halves", "deflated_sharpe_ratio")
    detail: dict[str, object] = {
        "declared_half": half,
        "declared_verdict": _optional_str(model, "deflated_sharpe_ratio", "verdict"),
        "dsr": _optional_float(model, "deflated_sharpe_ratio", "dsr"),
        "evidence": EVIDENCE_ARTIFACT,
    }
    if half == str(HalfResult.PASS):
        return _row(row, HalfResult.PASS, "dsr_significant", detail, basis="artifact")
    if half == str(HalfResult.FAIL):
        return _row(row, HalfResult.FAIL, "dsr_not_significant", detail, basis="artifact")
    return _row(row, HalfResult.NOT_EVALUABLE, "dsr_not_evaluable", detail, basis="artifact")


def _optional_str(node: object, *keys: str) -> str | None:
    """Cadena anidada opcional: ausente o ``None`` devuelve ``None``."""
    current: object = node
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = cast("dict[str, object]", current)[key]
    return current if isinstance(current, str) else None


def _drawdown_row(
    *, pipeline: Mapping[str, object], table: KillTable, row: KillRow
) -> dict[str, object]:
    """Fila del drawdown: lee la metrica de #28 y el umbral del literal de §11.6 (A14)."""
    metrics = _mapping(
        _require(pipeline, "arms", "coste_declarado", "metrics"),
        where="arms.coste_declarado.metrics",
    )
    draw = _mapping(metrics.get("max_drawdown_pct"), where="max_drawdown_pct")
    limit = _drawdown_limit_pct(table)
    estimate = draw.get("estimate")
    detail: dict[str, object] = {
        "arm": "coste_declarado",
        "basis": _optional_str(draw, "basis"),
        "max_drawdown_pct": draw,
        "limit_pct": _num(limit),
        "evidence": EVIDENCE_ARTIFACT,
        "note": "medida sobre `declared_cost` del brazo `coste_declarado` de #28; **no** es una "
        "medida neta (A14, A17)",
    }
    if estimate is None:
        return _row(
            row, HalfResult.NOT_EVALUABLE, "drawdown_not_computable", detail, basis="declared_cost"
        )
    if float(cast("float", estimate)) < float(limit):
        return _row(row, HalfResult.PASS, "drawdown_ok", detail, basis="declared_cost")
    return _row(row, HalfResult.FAIL, "drawdown_above_limit", detail, basis="declared_cost")


def _net_basis_row(
    *, row: KillRow, net: dict[str, object] | None, net_state: str, key: str
) -> dict[str, object]:
    """Fila que exige significacion sobre la base **neta** (A9, A11, A15)."""
    detail: dict[str, object] = {"net_metrics_state": net_state}
    if net is None or net_state != _NET_STATE_COMPUTED:
        return _row(
            row,
            HalfResult.NOT_EVALUABLE,
            "net_metrics_not_computable",
            detail,
            basis="net",
            follow_ups=NET_UNAVAILABLE_FOLLOW_UPS,
        )
    beats = _mapping(_require(net, "beats"), where="net_metrics.beats")
    half = _require_str(beats, key)
    detail["declared_half"] = half
    if half == str(HalfResult.PASS):
        return _row(row, HalfResult.PASS, f"beats_{key}", detail, basis="net")
    if half == str(HalfResult.FAIL):
        return _row(row, HalfResult.FAIL, f"not_significant_vs_{key}", detail, basis="net")
    return _row(row, HalfResult.NOT_EVALUABLE, f"{key}_not_evaluable", detail, basis="net")


def _edge_row(
    *, row: KillRow, net: dict[str, object] | None, net_state: str, stars: Mapping[str, object]
) -> dict[str, object]:
    """Fila principal (A9): el IC de acierto excluye `p*` o el del Sharpe excluye 0."""
    binding = _mapping(stars["binding"], where="p_star.binding")
    detail: dict[str, object] = {
        "net_metrics_state": net_state,
        "p_star_pct": binding["p_star_pct"],
        "p_star_r_pct": binding["r_pct"],
        "basis": "net",
    }
    if net is None or net_state != _NET_STATE_COMPUTED:
        return _row(
            row,
            HalfResult.NOT_EVALUABLE,
            "net_metrics_not_computable",
            detail,
            basis="net",
            follow_ups=NET_UNAVAILABLE_FOLLOW_UPS,
        )
    hit = _mapping(_require(net, "hit_rate"), where="net_metrics.hit_rate")
    hit_low = _require_number(hit, "lower")
    hit_high = _require_number(hit, "upper")
    sharpe = _mapping(_require(net, "sharpe"), where="net_metrics.sharpe")
    sharpe_low = _require_number(sharpe, "lower")
    sharpe_high = _require_number(sharpe, "upper")
    p_value = float(Decimal(cast("str", binding["p_star_fraction"])))
    hit_above = hit_low > p_value
    sharpe_above = sharpe_low > 0.0
    hit_below = hit_high < p_value
    sharpe_below = sharpe_high < 0.0
    detail["hit_rate"] = {"lower": hit_low, "upper": hit_high, "p_star_fraction": p_value}
    detail["sharpe"] = {"lower": sharpe_low, "upper": sharpe_high}
    detail["hit_rate_excludes_p_star_above"] = hit_above
    detail["sharpe_excludes_zero_above"] = sharpe_above
    if hit_above or sharpe_above:
        return _row(row, HalfResult.PASS, "statistical_edge", detail, basis="net")
    if hit_below or sharpe_below:
        return _row(row, HalfResult.FAIL, "edge_against", detail, basis="net")
    return _row(row, HalfResult.NOT_EVALUABLE, "no_statistical_edge", detail, basis="net")


def _row(
    row: KillRow,
    state: HalfResult,
    code: str,
    detail: dict[str, object],
    *,
    basis: str,
    follow_ups: Sequence[str] = (),
    note: str | None = None,
) -> dict[str, object]:
    """Fila evaluada publicada: estado, `code`, `source_row` y su procedencia (A7)."""
    published: dict[str, object] = {
        "row_index": row.row_index,
        "kind": row.kind,
        "source_row": row.criterion,
        "criterion": row.criterion,
        "threshold": row.threshold,
        "action": row.action,
        "state": str(state),
        "code": code,
        "basis": basis,
        "detail": detail,
    }
    if follow_ups:
        published["follow_ups"] = list(follow_ups)
    if note is not None:
        published["note"] = note
    return published


def evaluate_criteria(
    *,
    pipeline: Mapping[str, object],
    model: Mapping[str, object],
    table: KillTable,
    stars: Mapping[str, object],
) -> list[dict[str, object]]:
    """Evalua las nueve filas de §11.6 al vocabulario de #9 (A7, A9-A16).

    Funcion **pura**: entra el payload ya cargado de #28/#26 y la tabla leida del documento,
    y salen las nueve filas publicadas. Que sea pura permite forzar corridas sinteticas de
    resultados opuestos y comprobar que los umbrales no se derivan de los resultados (A6).
    """
    net = _net_block(pipeline)
    net_state = "absent" if net is None else str(net.get("state", "absent"))
    return [
        _edge_row(row=table.row(ROW_EDGE), net=net, net_state=net_state, stars=stars),
        _pbo_row(model=model, row=table.row(ROW_PBO)),
        _dsr_row(model=model, row=table.row(ROW_DSR)),
        _drawdown_row(pipeline=pipeline, table=table, row=table.row(ROW_DRAWDOWN)),
        _net_basis_row(row=table.row(ROW_NO_TRADE), net=net, net_state=net_state, key="no_trade"),
        _net_basis_row(row=table.row(ROW_LISTON_A), net=net, net_state=net_state, key="liston_a"),
        _net_basis_row(
            row=table.row(ROW_LISTON_B),
            net=net,
            net_state=net_state,
            key="liston_b",
        ),
        _row(
            table.row(ROW_PAPER),
            HalfResult.NOT_EVALUABLE,
            "paper_vs_backtest_not_available",
            {"phase": "Fase 4"},
            basis="pre_registered",
            follow_ups=("#45",),
            note="la divergencia paper-vs-backtest es de Fase 4: se **publica**, no se suprime "
            "(A16)",
        ),
        _row(
            table.row(ROW_CLOSE),
            HalfResult.NOT_EVALUABLE,
            "session_close_compliance_not_available",
            {"phase": "Fase 4"},
            basis="pre_registered",
            follow_ups=("#84",),
            note="el cumplimiento del cierre de sesion es de Fase 4: se **publica**, no se "
            "suprime (A16)",
        ),
    ]


def _mark_liston_b(criteria: list[dict[str, object]]) -> None:
    """Marca la fila de B como *insuficiente por si sola* (A11), sin tocar su estado."""
    for entry in criteria:
        if entry.get("kind") == ROW_LISTON_B:
            entry["insufficient_on_its_own"] = True
            entry["note"] = (
                "batir a `siempre largo close→close` **no demuestra nada** sobre el edge "
                "direccional: solo demuestra que no se paga financiacion (§11.6); nunca "
                "convierte la fila principal en `pass` (A11)"
            )


def gate_block(criteria: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Agrega las filas **importando** ``aggregate_gate`` de #9 (A7).

    El plegado es ``reduce(aggregate_gate, estados)``: `fail` si alguna falla, `not_evaluable`
    si ninguna falla y alguna no es evaluable, y `pass` **solo** si todas pasan.
    """
    states = [str(entry["state"]) for entry in criteria]
    if not states:
        raise VerdictError("no hay filas que agregar: la tabla de §11.6 esta vacia (A7)")
    folded: str = reduce(lambda left, right: str(aggregate_gate(left, right)), states)
    counts = {member.value: states.count(member.value) for member in HalfResult}
    return {
        "aggregate": folded,
        "aggregation_source": GATE_AGGREGATION_SOURCE,
        "rule": GATE_AGGREGATION_RULE,
        "n_rows": len(states),
        "counts": counts,
        "folding": "reduce(aggregate_gate, estados)",
        "note": "un `not_evaluable` **nunca** se convierte en `pass`: la agregacion de #9 lo "
        "hace imposible por construccion (A7)",
    }


def resolve_verdict(
    *, aggregate: str, criteria: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Deriva el veredicto de Fase 2 y comprueba la consistencia de #9 (A8).

    Precedencia declarada: `pass` agregado ⇒ `continue`; si la fila principal no es evaluable,
    el veredicto es `not_evaluable` (la evidencia decisiva falta); si hay filas en `fail`,
    `simplify` solo si todas las que fallan se arreglan simplificando (`PBO`) y `stop` en
    cualquier otro caso; si el agregado es `not_evaluable`, `not_evaluable`.
    """
    gate = GateVerdict(str(aggregate))
    if gate is GateVerdict.PASS:
        value = VERDICT_CONTINUE
    else:
        main = str(criteria[0]["state"])
        failing = [entry for entry in criteria if entry["state"] == str(HalfResult.FAIL)]
        if main == str(HalfResult.NOT_EVALUABLE):
            value = VERDICT_NOT_EVALUABLE
        elif failing and all(str(entry["kind"]) in _SIMPLIFY_ROWS for entry in failing):
            value = VERDICT_SIMPLIFY
        elif failing:
            value = VERDICT_STOP
        else:
            value = VERDICT_NOT_EVALUABLE
    return consistent_verdict_block(gate, value)


def consistent_verdict(*, gate: GateVerdict, verdict: str) -> str:
    """Comprueba que el veredicto es coherente con el agregado; si no, error tipado (A8)."""
    if verdict not in PHASE2_TO_RECOMMENDATION:
        raise VerdictError(
            f"veredicto fuera del vocabulario declarado: {verdict!r}; se admite "
            f"{sorted(PHASE2_TO_RECOMMENDATION)} (A8)"
        )
    recommendation = PHASE2_TO_RECOMMENDATION[verdict]
    if not recommendation_is_consistent(gate, recommendation):
        raise VerdictError(
            f"el veredicto {verdict!r} no es coherente con el agregado {gate.value!r}: "
            f"{RECOMMENDATION_CONSISTENCY_RULE} (A8)"
        )
    return verdict


def consistent_verdict_block(gate: GateVerdict, verdict: str) -> dict[str, object]:
    """Bloque publicado del veredicto, ya validado contra la regla de #9 (A8)."""
    checked = consistent_verdict(gate=gate, verdict=verdict)
    return {
        "state": checked,
        "consistent": True,
        "recommendation": PHASE2_TO_RECOMMENDATION[checked].value,
        "recommendation_source": "cfdtrader.analysis.phase0_report.Recommendation (#9)",
        "rule": RECOMMENDATION_CONSISTENCY_RULE,
        "note": "con agregado distinto de `pass` el veredicto **no** puede ser `continue` (A8)",
    }


def _baselines_block(pipeline: Mapping[str, object]) -> dict[str, object]:
    """Copia la tabla de baselines y listones de #28, distinguiendo los tres listones (A12)."""
    table = _mapping(_require(pipeline, "table"), where="table")
    rows = cast("list[object]", _require(table, "rows"))
    copied = [_mapping(row, where=f"table.rows[{index}]") for index, row in enumerate(rows)]
    listones = [row for row in copied if str(row.get("kind", "baseline")) != "baseline"]
    baselines = [row for row in copied if str(row.get("kind", "baseline")) == "baseline"]
    kinds = [str(row.get("kind")) for row in copied]
    liston_c = next((row for row in copied if row.get("kind") == ROW_LISTON_C), None)
    return {
        "evidence": EVIDENCE_ARTIFACT,
        "is_validation": False,
        "rows": copied,
        "baselines": baselines,
        "listones": listones,
        "n_baselines": len(baselines),
        "kinds": kinds,
        "liston_c_is_invertible": None if liston_c is None else bool(liston_c.get("is_invertible")),
        "rule": "la tabla se **copia** de #28 con procedencia `artifact`: no se recalcula; el "
        "liston C es una referencia y **nunca** cuenta como baseline (A12)",
    }


#: Identificador del liston C en la tabla de #28 (referencia `^GSPC`, no invertible).
ROW_LISTON_C: Final[str] = "liston_c"


def _declared_cost_block(pipeline: Mapping[str, object]) -> dict[str, object]:
    """Publica la comparacion de coste declarado de #28 **etiquetada** como tal (A15)."""
    scenario = _mapping(_require(pipeline, "scenario"), where="scenario")
    arms = _mapping(_require(pipeline, "arms"), where="arms")
    coste = _mapping(arms["coste_declarado"], where="arms.coste_declarado")
    return {
        "basis": "declared_cost",
        "is_validation": False,
        "arm": "coste_declarado",
        "cost_basis_pct": _require_str(scenario, "cost_basis_pct"),
        "cost_provenance": _require_str(scenario, "cost_provenance"),
        "traded": _require_number(coste, "traded"),
        "no_trade": _require_number(coste, "no_trade"),
        "evidence": EVIDENCE_ARTIFACT,
        "note": "esta es la comparacion **declarada** de #28: se publica etiquetada como "
        "`declared_cost` y **nunca** como el criterio pre-registrado, que exige la base neta "
        "(A15)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# El informe: payload canonico, hash y escritura (A2, A4)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Phase2Report:
    """El informe: payload canonico, hash y los objetos que lo produjeron."""

    as_of: datetime
    report_date: str
    payload: dict[str, object]
    report_sha256: str
    reports_dir: Path
    table: KillTable
    pipeline: InputArtifact
    model: InputArtifact

    @property
    def report_stem(self) -> str:
        """Nombre base del informe: ``phase2_report_<AAAA-MM-DD>``."""
        return f"{REPORT_PREFIX}_{self.report_date}"

    def json_text(self) -> str:
        """JSON determinista: mismo ``as_of`` ⇒ mismo texto byte a byte (A2, A4)."""
        published: dict[str, object] = {**self.payload, "report_sha256": self.report_sha256}
        return json.dumps(published, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    def write(self, directory: Path) -> tuple[Path, Path]:
        """Escribe el informe en JSON y Markdown dentro de ``directory`` (A2)."""
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{self.report_stem}.json"
        markdown_path = directory / f"{self.report_stem}.md"
        json_path.write_text(self.json_text(), encoding="utf-8")
        markdown_path.write_text(render_markdown(self), encoding="utf-8")
        return json_path, markdown_path


def _digest(payload: Mapping[str, object]) -> str:
    """``report_sha256``: el sha256 del texto canonico de #13, con prefijo (A4)."""
    return HASH_PREFIX + hashlib.sha256(canonical_text(payload).encode("utf-8")).hexdigest()


def _campaign_blocks(
    *, pipeline: InputArtifact, model: InputArtifact, table: KillTable, stars: dict[str, object]
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object], dict[str, object]]:
    """Bloques derivados: criterios, puerta, veredicto y procedencia (A7-A16)."""
    criteria = evaluate_criteria(
        pipeline=pipeline.payload, model=model.payload, table=table, stars=stars
    )
    gate = gate_block(criteria)
    verdict = resolve_verdict(aggregate=cast("str", gate["aggregate"]), criteria=criteria)
    provenance: dict[str, object] = {
        "pipeline": pipeline.provenance(),
        "model_comparison": model.provenance(),
    }
    return criteria, gate, verdict, provenance


def analyse(
    *, store: Store, reports_dir: Path, as_of: datetime, write: bool = True
) -> Phase2Report:
    """Aplica la tabla de §11.6 a #28/#26 y (por defecto) escribe el informe (A1, A2).

    ``as_of`` es **obligatorio** y es el unico instante de la corrida: ninguna ruta consulta el
    reloj. ``write=False`` no escribe **nada**.
    """
    moment = _as_utc(as_of)
    table = load_kill_table()
    pipeline = load_input_artifact(reports_dir, PIPELINE_CLASS, store=store)
    model = load_input_artifact(reports_dir, MODEL_CLASS, store=store)

    cost_pct = Decimal(_require_str(pipeline.payload, "scenario", "cost_basis_pct"))
    cost_source = _require_str(pipeline.payload, "scenario", "cost_provenance")
    stars = p_star_block(cost_pct=cost_pct, cost_source=cost_source)

    criteria, gate, verdict, provenance = _campaign_blocks(
        pipeline=pipeline, model=model, table=table, stars=stars
    )
    _mark_liston_b(criteria)
    listones = _baselines_block(pipeline.payload)
    net_node = _net_block(pipeline.payload)
    net_metrics = dict(net_node) if net_node is not None else {"state": "absent"}

    payload: dict[str, object] = {
        "analysis": ANALYSIS,
        "task": TASK,
        "title": TITLE,
        "generated_at": moment.isoformat(),
        "report_date": moment.date().isoformat(),
        "hash_format": REPORT_HASH_FORMAT,
        "basis": "pre_registered",
        "is_validation": False,
        "criteria_source": dict(table.source),
        "kill_criteria": [row.as_dict() for row in table.rows],
        "provenance": provenance,
        "p_star": stars,
        "criteria": criteria,
        "gate": gate,
        "phase2_ready": str(gate["aggregate"]) == str(GateVerdict.PASS),
        "verdict": verdict,
        "listones": listones,
        "net_metrics": net_metrics,
        "declared_cost": _declared_cost_block(pipeline.payload),
        "limitations": list(REPORT_LIMITATIONS),
        "does_not_do": [dict(entry) for entry in REPORT_DOES_NOT_DO],
        "follow_ups": [dict(entry) for entry in FOLLOW_UPS],
        "a13_divergence": A13_DIVERGENCE,
    }
    report = Phase2Report(
        as_of=moment,
        report_date=moment.date().isoformat(),
        payload=payload,
        report_sha256=_digest(payload),
        reports_dir=reports_dir,
        table=table,
        pipeline=pipeline,
        model=model,
    )
    if write:
        json_path, markdown_path = report.write(reports_dir)
        logger.info("informe de Fase 2: {} y {}", json_path, markdown_path)
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


def render_markdown(report: Phase2Report) -> str:
    """El informe en Markdown: resumen de la puerta, las nueve filas y los limites (A1)."""
    payload = report.payload
    source = _mapping(payload["criteria_source"], where="criteria_source")
    gate = _mapping(payload["gate"], where="gate")
    verdict = _mapping(payload["verdict"], where="verdict")
    provenance = _mapping(payload["provenance"], where="provenance")
    stars = _mapping(payload["p_star"], where="p_star")
    binding = _mapping(stars["binding"], where="p_star.binding")
    criteria = cast("list[object]", payload["criteria"])
    listones = _mapping(payload["listones"], where="listones")

    lines: list[str] = [
        "# Informe de Fase 2 y criterios de *kill*",
        "",
        f"- **Tarea**: {payload['task']}",
        f"- **Instante declarado (`as_of`)**: `{payload['generated_at']}`",
        f"- **Tabla de criterios**: `{source['path']}` {source['section']} "
        f"(sha256 `{source['sha256']}`, `n_rows` = {source['n_rows']})",
        f"- **`report_sha256`**: `{report.report_sha256}`",
        f"- **`phase2_ready`**: `{str(payload['phase2_ready']).lower()}` · "
        f"**agregado**: `{gate['aggregate']}` · **veredicto**: `{verdict['state']}`",
        "",
        "> **La puerta de Fase 2 no se aprueba hoy.** El criterio principal es estadistico "
        "(`plan.md` §4.6) y hoy la base neta no es computable: `not_evaluable` **no** se "
        "convierte en `pass`.",
        "",
        "## Las nueve filas de §11.6",
        "",
    ]
    rows: list[list[str]] = []
    for entry in criteria:
        item = _mapping(entry, where="criterio")
        rows.append(
            [
                str(item["row_index"]),
                f"`{item['state']}`",
                f"`{item['code']}`",
                f"`{item['basis']}`",
                str(item["threshold"]).replace("|", "/"),
            ]
        )
    lines += _table(["fila", "estado", "code", "basis", "umbral"], rows)
    lines += [
        "",
        f"Agregacion: `{gate['aggregation_source']}` — {gate['rule']}",
        "",
        f"Veredicto: `{verdict['state']}` (recomendacion `{verdict['recommendation']}`, "
        f"{verdict['rule']})",
        "",
        "## `p*` derivado (no cableado)",
        "",
        f"- Coste de ida y vuelta declarado: **{stars['cost_round_trip_pct']} %** "
        f"(`{stars['cost_source']}`)",
        f"- Derivacion: {stars['derivation']}",
        f"- Escenario mas exigente: `R` = {binding['r_pct']} % ⇒ `p*` = "
        f"**{binding['p_star_pct']} %**",
        "",
    ]
    star_rows = [
        [
            f"`R` = {_mapping(item, where='escenario')['r_pct']} %",
            str(_mapping(item, where="escenario")["p_star_pct"]),
        ]
        for item in cast("list[object]", stars["scenarios"])
    ]
    lines += _table(["escenario", "p* (%)"], star_rows)
    lines += [
        "",
        "## Baselines y listones (copiado de #28)",
        "",
        f"- Procedencia: `{listones['evidence']}` · `is_validation`: "
        f"`{str(listones['is_validation']).lower()}` · baselines: {listones['n_baselines']}",
        f"- Liston C invertible: `{listones['liston_c_is_invertible']}` (una referencia, nunca "
        "un baseline)",
        "",
        "## Procedencia de los artefactos",
        "",
    ]
    for key in ("pipeline", "model_comparison"):
        block = _mapping(provenance[key], where=key)
        lines.append(
            f"- **{key}**: `{block['path']}` (sha256 `{block['sha256']}`, "
            f"`report_sha256` `{block['report_sha256']}`, `generated_at` `{block['generated_at']}`)"
        )
    lines += ["", "## Limites declarados", ""]
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
    lines += [
        "",
        f"> Nota declarada (A13): {payload['a13_divergence']}",
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_as_of(value: str | None) -> datetime:
    """El instante declarado de la CLI: **obligatorio**, y nunca tomado del reloj (A2)."""
    if value is None:
        raise MissingAsOfError(
            "falta --as-of: escribir el informe exige un instante declarado (el modulo no lee "
            "el reloj) y sin el **no se escribe ningun fichero** (A2)"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise InvalidAsOfError(f"--as-of no es un ISO-8601 valido: {error} (A2)") from error
    return _as_utc(parsed)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del informe de Fase 2.

    Codigos de salida: ``0`` = informe escrito (aunque la puerta **no** se supere, que es un
    resultado legitimo y declarado); ``2`` = falta o no es valido ``--as-of``, falta un
    artefacto de #28/#26, hay ambiguedad o la tabla de §11.6 no se puede leer ⇒ **no se escribe
    nada** y el motivo sale por ``stderr``.
    """
    parser = argparse.ArgumentParser(
        prog="cfdtrader.analysis.phase2_report",
        description="Informe de Fase 2 y criterios de kill pre-registrados",
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
    args = parser.parse_args(argv)

    try:
        moment = _parse_as_of(cast("str | None", args.as_of))
    except Phase2ReportError as error:
        print(f"no se puede emitir el informe de Fase 2: {error}", file=sys.stderr)
        return 2

    data_root = Path(args.data_root) if args.data_root is not None else Path("data")
    reports_dir = (
        Path(args.reports_dir)
        if args.reports_dir is not None
        else data_root / "derived" / "reports"
    )
    try:
        report = analyse(store=Store(data_root), reports_dir=reports_dir, as_of=moment, write=True)
    except Phase2ReportError as error:
        print(f"no se puede emitir el informe de Fase 2: {error}", file=sys.stderr)
        return 2

    gate = cast("dict[str, object]", report.payload["gate"])
    verdict = cast("dict[str, object]", report.payload["verdict"])
    logger.info(
        "Fase 2: agregado {}; phase2_ready {}; veredicto {}; report_sha256 = {}",
        gate["aggregate"],
        report.payload["phase2_ready"],
        verdict["state"],
        report.report_sha256,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
