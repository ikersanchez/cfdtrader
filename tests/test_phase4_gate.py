"""Puerta de la Fase 4: pre-registro, anclaje e implementabilidad (tarea #130).

Lo que se blinda aquí
---------------------
La puerta de salida de la Fase 4 (tarea 45) vivía en `plan.md` §11.6 como **umbral** —«divergencia
paper vs backtest por debajo de 2 desviaciones típicas durante 3 meses»— y no como **regla**: no
decía de qué es esa desviación, sobre qué muestra ni con qué regla se decide. `§16` la
**operacionaliza** en un bloque delimitado por los marcadores
`<!-- puerta-fase-4:begin -->` y `<!-- puerta-fase-4:end -->`.

- **A11** — el bloque existe, declara sus **cinco** piezas y está **anclado**: su `sha256` es un
  literal estable —un bloque del propio documento, con el mismo criterio que el de §11.6—, así que
  mover la puerta falla en vez de pasar desapercibido.
- **A12** — el pre-registro es **implementable**: los campos de `journal.decisions` que declara leer
  existen en el esquema **cerrado** de #39, y el vocabulario que usa (`status`, `direction`) es el
  declarado. Si el esquema cambia, la puerta falla **en voz alta** en lugar de quedarse obsoleta.
- **A13** — operacionalizar esta puerta **no** ha movido la otra: `§11.6` sigue leyéndose con sus
  **nueve** filas.

No hay módulo nuevo de `src/`: esto es una prueba de **contrato de documento**, así que el suelo de
cobertura (90 % sentencias / 85 % ramas) **no aplica** y se declara aquí, como pide la tarea.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from cfdtrader.analysis import phase2_report
from cfdtrader.journal.decision_log import DECISION_STATUSES, DIRECTIONS, TABLE_COLUMNS

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PLAN_PATH: Final[Path] = REPO_ROOT / "_docs" / "plan.md"

#: Los dos marcadores que delimitan el bloque pre-registrado de la Fase 4 (A1).
BEGIN_MARKER: Final[str] = "<!-- puerta-fase-4:begin -->"
END_MARKER: Final[str] = "<!-- puerta-fase-4:end -->"

#: ``sha256`` dorado del bloque de la puerta de la Fase 4 (A11): si la puerta se mueve, esto falla.
#: Es el digest de los bytes UTF-8 de lo que hay **estrictamente entre** los dos marcadores.
GOLDEN_PHASE4_GATE_SHA256: Final[str] = (
    "8bd18442b4866307d199aa2c5b9ddaac365fc48cf7f711d11bb102412c490258"  # pragma: allowlist secret
)

#: Las cinco piezas que el bloque tiene que declarar (A1, A11), con el texto que las identifica.
GATE_PIECES: Final[dict[str, tuple[str, ...]]] = {
    "estadistico": (
        "media del retorno neto por sesión",
        "status = recommendation",
        "direction ∈ {long, short}",
    ),
    "referencia": ("σ que declara el backtest", "NO se recalcula"),  # noqa: RUF001
    "regla": ("2 · σ_backtest / √N",),  # noqa: RUF001
    "muestra_minima": ("N < 30", "not_evaluable"),
    "recomputacion": ("journal.decisions", "trade_date", "stop_pct", "target_pct", "cost_pct"),
}

#: Los campos de ``journal.decisions`` que el pre-registro declara leer (A12). Se comprueban contra
#: el esquema cerrado de #39: si dejan de existir, la puerta **no** es implementable.
DECLARED_DECISION_FIELDS: Final[tuple[str, ...]] = (
    "trade_date",
    "direction",
    "stop_pct",
    "target_pct",
    "cost_pct",
)


def _plan_text() -> str:
    """El texto de ``plan.md`` (vive en git: la prueba corre tambien en el clon limpio del CI)."""
    return PLAN_PATH.read_text(encoding="utf-8")


def gate_block(text: str) -> str:
    """El bloque pre-registrado, **estrictamente entre** sus dos marcadores (A11)."""
    lines = text.split("\n")
    begins = [index for index, line in enumerate(lines) if line.strip() == BEGIN_MARKER]
    ends = [index for index, line in enumerate(lines) if line.strip() == END_MARKER]
    assert len(begins) == 1, f"se espera un unico {BEGIN_MARKER}: hay {len(begins)}"
    assert len(ends) == 1, f"se espera un unico {END_MARKER}: hay {len(ends)}"
    assert begins[0] < ends[0], "los marcadores de la puerta estan invertidos"
    return "\n".join(lines[begins[0] + 1 : ends[0]]) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# A11 · El bloque esta, declara las cinco piezas y esta anclado
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_the_gate_block_declares_its_five_pieces() -> None:
    block = gate_block(_plan_text())
    missing = [
        f"{piece}:{needle}"
        for piece, needles in GATE_PIECES.items()
        for needle in needles
        if needle not in block
    ]
    assert not missing, f"la puerta de la Fase 4 no declara: {missing}"


def test_a11_the_gate_block_is_pinned() -> None:
    digest = hashlib.sha256(gate_block(_plan_text()).encode("utf-8")).hexdigest()
    assert digest == GOLDEN_PHASE4_GATE_SHA256, (
        "la puerta de la Fase 4 se ha movido: §11.6 exige pre-registrarla antes de ver resultados "
        "y no modificarla despues"
    )


def test_a11_moving_the_sample_floor_changes_the_digest() -> None:
    """El anclaje detecta un cambio de regla, no solo un documento distinto."""
    text = _plan_text()
    perturbed = text.replace("N < 30", "N < 20")
    assert perturbed != text, "la sonda no cambio nada: revisa el texto de la regla"
    moved = hashlib.sha256(gate_block(perturbed).encode("utf-8")).hexdigest()
    assert moved != GOLDEN_PHASE4_GATE_SHA256


# ─────────────────────────────────────────────────────────────────────────────
# A12 · El pre-registro es implementable: esquema cerrado y vocabulario declarado
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_the_declared_fields_exist_in_the_closed_schema() -> None:
    columns = TABLE_COLUMNS["decisions"]
    unknown = [field for field in DECLARED_DECISION_FIELDS if field not in columns]
    assert not unknown, f"la puerta declara campos que el esquema de #39 no tiene: {unknown}"


def test_a12_the_fields_it_reads_are_the_ones_it_declares() -> None:
    block = gate_block(_plan_text())
    absent = [field for field in DECLARED_DECISION_FIELDS if field not in block]
    assert not absent, f"la puerta lee campos que el bloque no declara: {absent}"


def test_a12_the_block_speaks_the_declared_vocabulary() -> None:
    block = gate_block(_plan_text())
    missing_statuses = [status for status in DECISION_STATUSES if status not in block]
    missing_directions = [direction for direction in DIRECTIONS if direction not in block]
    assert not missing_statuses, f"la puerta no declara los estados {missing_statuses}"
    assert not missing_directions, f"la puerta no declara las direcciones {missing_directions}"


# ─────────────────────────────────────────────────────────────────────────────
# A13 · Operacionalizar esta puerta no ha movido la de §11.6
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_the_kill_table_still_reads_from_the_edited_document() -> None:
    table = phase2_report.read_kill_table(_plan_text())
    assert table.source["section"] == "§11.6"
    assert table.source["n_rows"] == 9
    assert tuple(row.kind for row in table.rows) == phase2_report.ROW_KINDS
