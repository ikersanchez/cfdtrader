"""Diferencia declarada frente al artefacto previo (#90, tarea T30).

Los informes de `analysis/` que publican P&L declarado se **regeneran** cuando cambia una
decision del motor. Un informe regenerado no dice por si solo que ha cambiado: este modulo
calcula y **declara** la diferencia contra el artefacto anterior, para que el `.md` publicado
lleve la cuenta y nadie tenga que rehacer la regeneracion para auditarla.

El caso que lo motiva es #80: el motor resta `c_declared_pct / 100 = 0.000042` (fraccion del
nocional) y no `c_declared_pct = 0.0042` (porcentaje). Cada sesion operada gana
``0.0042 - 0.000042 = 0.004158`` de suma declarada, asi que la diferencia por operacion es
**constante** y **medible**; aqui se comprueba, fila a fila, contra la constante declarada.

El modulo es **determinista y puro**: no consulta el reloj, no toca la red y no abre el
``Store``. La unica lectura de disco es :func:`load_previous`, que carga el artefacto previo
para que los generadores no vuelvan a calcularlo.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

#: Prefijo obligatorio de un digest (politica `detect-secrets`, #89/#95/#96).
HASH_PREFIX: Final[str] = "sha256:"

#: La constante declarada por operacion: `0.0042 (porcentaje) - 0.000042 (fraccion)`.
DECLARED_COST_PER_OPERATION: Final[float] = 0.004158

#: Tolerancia declarada de las comprobaciones aritmeticas (±1e-9 en la suma agregada).
TOLERANCE: Final[float] = 1e-9

#: Titulo comun de la seccion de regeneracion.
SECTION_TITLE: Final[str] = "## Regeneración"


class RegenerationError(Exception):
    """Raiz de los errores del bloque de regeneracion."""


class PreviousArtifactError(RegenerationError):
    """El artefacto previo declarado no existe o no es un objeto JSON."""


class MissingRowError(RegenerationError):
    """Una fila del artefacto previo no tiene pareja en el informe nuevo."""


# ─────────────────────────────────────────────────────────────────────────────
# El artefacto previo
# ─────────────────────────────────────────────────────────────────────────────
def load_previous(path: Path | None) -> dict[str, object] | None:
    """Carga el artefacto previo declarado, o ``None`` si no se declaro ninguno.

    No consulta el reloj ni la red ni el ``Store``: solo lee el fichero indicado. Si la ruta se
    declara pero no existe, es un error tipado (un informe regenerado sin su "antes" no puede
    declarar la diferencia).
    """
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        raise PreviousArtifactError(
            f"el artefacto previo {candidate} no existe: sin el `antes` no se puede declarar la "
            "diferencia de la regeneracion"
        )
    document: object = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise PreviousArtifactError(
            f"el artefacto previo {candidate} no es un objeto JSON: no se puede leer su payload"
        )
    return cast("dict[str, object]", document)


def artifact_name(path: Path | None) -> str | None:
    """El **nombre** del artefacto previo (nunca la ruta: rompe la determinismo byte a byte)."""
    return None if path is None else Path(path).name


# ─────────────────────────────────────────────────────────────────────────────
# Deltas de la suma declarada, fila a fila
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class DeclaredRowDelta:
    """La suma declarada de una fila antes y despues, con su delta por operacion."""

    label: str
    n_traded: int
    previous_sum: float
    current_sum: float

    @property
    def delta(self) -> float:
        """La diferencia de la suma declarada (`nueva - previa`)."""
        return self.current_sum - self.previous_sum

    @property
    def expected_delta(self) -> float:
        """La diferencia que **deberia** haber: `operadas × 0.004158`."""
        return self.n_traded * DECLARED_COST_PER_OPERATION

    @property
    def per_operation(self) -> float | None:
        """El delta por operacion observado, o ``None`` si la fila no opera."""
        if self.n_traded == 0:
            return None
        return self.delta / self.n_traded

    @property
    def consistent(self) -> bool:
        """``True`` si el delta observado es la constante declarada por las operadas (±tol)."""
        return abs(self.delta - self.expected_delta) <= TOLERANCE

    def to_payload(self) -> dict[str, object]:
        """El bloque JSON de la fila, con el delta **medido** y el esperado."""
        return {
            "label": self.label,
            "n_traded": self.n_traded,
            "previous_pnl_declared_pct_sum": self.previous_sum,
            "current_pnl_declared_pct_sum": self.current_sum,
            "delta_pnl_declared_pct_sum": self.delta,
            "delta_per_operation": self.per_operation,
            "declared_delta_per_operation": DECLARED_COST_PER_OPERATION,
            "consistent_with_declared_term": self.consistent,
        }


def declared_row_deltas(
    previous_rows: Sequence[Mapping[str, object]],
    current_rows: Sequence[Mapping[str, object]],
    *,
    label: Callable[[Mapping[str, object]], str],
    summary: Callable[[Mapping[str, object]], float | None],
    traded: Callable[[Mapping[str, object]], int],
) -> tuple[DeclaredRowDelta, ...]:
    """Empareja las filas por etiqueta y devuelve su delta declarado, en el orden previo.

    Las filas se emparejan por la etiqueta que da ``label`` (que **no** puede cambiar al
    regenerar). Se saltan las que no operan o no publican suma declarada.
    """
    current_by_label = {label(row): row for row in current_rows}
    deltas: list[DeclaredRowDelta] = []
    for previous_row in previous_rows:
        key = label(previous_row)
        current_row = current_by_label.get(key)
        if current_row is None:
            raise MissingRowError(
                f"la fila {key!r} del artefacto previo no tiene pareja en el informe nuevo: las "
                "etiquetas son la identidad de la fila y no pueden cambiar al regenerar"
            )
        operations = traded(current_row)
        if operations <= 0:
            continue
        previous_sum = summary(previous_row)
        current_sum = summary(current_row)
        if previous_sum is None or current_sum is None:
            continue
        deltas.append(
            DeclaredRowDelta(
                label=key,
                n_traded=operations,
                previous_sum=previous_sum,
                current_sum=current_sum,
            )
        )
    return tuple(deltas)


def declared_deltas_block(
    *,
    previous_name: str,
    rows: Sequence[DeclaredRowDelta],
    subject: str,
) -> dict[str, object]:
    """El bloque `regeneration` del payload para un informe de suma declarada."""
    return {
        "previous_artifact": previous_name,
        "subject": subject,
        "declared_delta_per_operation": DECLARED_COST_PER_OPERATION,
        "declared_term_rule": (
            "el motor de #80 resta `c_fraction_of_notional = c_declared_pct / 100 = 0.000042` "
            "(fraccion) y no `0.0042` (porcentaje): cada operada gana `0.0042 - 0.000042 = "
            "0.004158` de suma declarada"
        ),
        "rows": [row.to_payload() for row in rows],
        "all_rows_consistent": all(row.consistent for row in rows),
        "gross_unchanged": (
            "el bruto (`gross_pct`) no depende del coste declarado: solo cambia `pnl_declared_pct`"
        ),
        "note": (
            "el bloque se **mide** contra el artefacto previo declarado; ninguna cifra se copia "
            "y el `report_sha256` del informe se recomputa con el bloque dentro"
        ),
    }


def render_declared_section(
    block: Mapping[str, object],
    *,
    intro: str,
) -> list[str]:
    """La seccion Markdown `## Regeneración` de un informe de suma declarada."""
    rows = [cast_mapping(item) for item in cast_sequence(block.get("rows", ()))]
    lines: list[str] = [
        SECTION_TITLE,
        "",
        intro,
        "",
        f"- Artefacto previo declarado: `{block['previous_artifact']}`.",
        f"- Delta declarado por operacion: `{DECLARED_COST_PER_OPERATION}` "
        "(`0.0042 - 0.000042`, #80).",
        f"- {block['declared_term_rule']}",
        "",
        "| fila | operadas | suma previa | suma nueva | delta | por operación |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        per_operation = row["delta_per_operation"]
        rendered = "`null`" if per_operation is None else f"{float(cast_float(per_operation))!r}"
        lines.append(
            f"| `{row['label']}` | {row['n_traded']} | "
            f"{float(cast_float(row['previous_pnl_declared_pct_sum']))!r} | "
            f"{float(cast_float(row['current_pnl_declared_pct_sum']))!r} | "
            f"{float(cast_float(row['delta_pnl_declared_pct_sum']))!r} | {rendered} |"
        )
    lines.extend(
        [
            "",
            f"- Todas las filas cuadran con la constante declarada: "
            f"`{str(block['all_rows_consistent']).lower()}`.",
            f"- {block['gross_unchanged']}",
            f"- {block['note']}",
        ]
    )
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Deltas de puntero (procedencia de un artefacto consumido)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PointerDelta:
    """Un puntero publicado (p. ej. `provenance.pipeline.sha256`) antes y despues."""

    name: str
    previous: object
    current: object

    @property
    def changed(self) -> bool:
        """``True`` si el puntero cambio entre el artefacto previo y el nuevo."""
        return self.previous != self.current

    def to_payload(self) -> dict[str, object]:
        """El bloque JSON del puntero."""
        return {
            "name": self.name,
            "previous": self.previous,
            "current": self.current,
            "changed": self.changed,
        }


def pointer_delta(
    previous: Mapping[str, object], current: Mapping[str, object], *, path: Sequence[str]
) -> PointerDelta:
    """El delta de un puntero anidado del payload, por su ruta (`("pipeline","sha256")`)."""
    return PointerDelta(
        name=".".join(path),
        previous=_dig(previous, path),
        current=_dig(current, path),
    )


def _dig(document: Mapping[str, object], path: Sequence[str]) -> object:
    """Baja por una ruta anidada del payload; ``None`` si falta algun tramo."""
    node: Mapping[str, object] = document
    for index, key in enumerate(path):
        value = node.get(key)
        if index == len(path) - 1:
            return value
        if not isinstance(value, dict):
            return None
        node = cast("dict[str, object]", value)
    return None


def pointer_deltas_block(
    *,
    previous_name: str,
    pointers: Sequence[PointerDelta],
    unchanged: Mapping[str, bool],
) -> dict[str, object]:
    """El bloque `regeneration` del payload de un informe que solo refresca punteros."""
    return {
        "previous_artifact": previous_name,
        "pointers": [pointer.to_payload() for pointer in pointers],
        "unchanged": {key: bool(value) for key, value in unchanged.items()},
        "all_unchanged": all(bool(value) for value in unchanged.values()),
        "note": (
            "esta regeneracion **solo** refresca el puntero al artefacto consumido: la rejilla, "
            "el veredicto y `phase2_ready` se recomputan y tienen que salir iguales"
        ),
    }


def render_pointer_section(
    block: Mapping[str, object],
    *,
    intro: str,
    unchanged_labels: Mapping[str, str],
) -> list[str]:
    """La seccion Markdown `## Regeneración` de un informe que solo refresca punteros."""
    pointers = [cast_mapping(item) for item in cast_sequence(block.get("pointers", ()))]
    unchanged = cast_mapping(block.get("unchanged", {}))
    lines: list[str] = [
        SECTION_TITLE,
        "",
        intro,
        "",
        f"- Artefacto previo declarado: `{block['previous_artifact']}`.",
        "",
        "| puntero | valor previo | valor nuevo | cambia |",
        "| --- | --- | --- | --- |",
    ]
    for pointer in pointers:
        lines.append(
            f"| `{pointer['name']}` | `{pointer['previous']}` | `{pointer['current']}` | "
            f"`{str(pointer['changed']).lower()}` |"
        )
    lines.extend(["", "- El delta es **de puntero**: cambia el `sha256` del artefacto consumido."])
    for key, text in unchanged_labels.items():
        value = unchanged.get(key)
        lines.append(f"- {text}: **no** cambia (`{str(bool(value)).lower()}`).")
    lines.append(f"- {block['note']}")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades locales (sin dependencias del payload de ningun informe)
# ─────────────────────────────────────────────────────────────────────────────
def cast_mapping(node: object) -> Mapping[str, object]:
    """Un nodo del payload como mapping, con error tipado si no lo es."""
    if not isinstance(node, dict):
        raise RegenerationError(f"se esperaba un objeto JSON y llego {type(node).__name__}")
    return cast("Mapping[str, object]", node)


def cast_sequence(node: object) -> Sequence[object]:
    """Un nodo del payload como secuencia, con error tipado si no lo es."""
    if not isinstance(node, list):
        raise RegenerationError(f"se esperaba una lista JSON y llego {type(node).__name__}")
    return cast("list[object]", node)


def cast_float(node: object) -> float:
    """Un numero del payload como ``float``, con error tipado si no lo es."""
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise RegenerationError(f"se esperaba un numero y llego {type(node).__name__}")
    return float(node)
