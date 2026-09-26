"""Tests del bloque de regeneracion declarada (#90, tarea T30).

El modulo `analysis.regeneration_delta` es **determinista y puro**: calcula la diferencia de la
suma declarada (y los deltas de puntero) contra el artefacto previo, sin reloj, sin red y sin
`Store`. Aqui se comprueba la aritmetica (`0.004158` por operacion), las banderas de
consistencia, los errores tipados, el render Markdown y el determinismo entre procesos.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

import pytest

from cfdtrader.analysis import regeneration_delta
from cfdtrader.backtest.engine import canonical_text

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODULE_PATH: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "analysis" / "regeneration_delta.py"
REAL_REPORTS: Final[Path] = REPO_ROOT / "data" / "derived" / "reports"

needs_store = pytest.mark.skipif(
    not (REPO_ROOT / "data" / "derived" / "labels").exists(),
    reason="el almacen real no esta en el arbol: los artefactos regenerados son los suyos",
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def _row(label: str, *, traded: int, total: float | None) -> dict[str, object]:
    """Una fila sintetica con la forma de `comparison.rows` (baseline o model_comparison)."""
    return {"label": label, "traded": traded, "pnl_declared_pct": {"sum": total}}


def _label_of(row: Mapping[str, object]) -> str:
    """La etiqueta de una fila sintetica."""
    return str(row["label"])


def _traded_of(row: Mapping[str, object]) -> int:
    """Las operaciones de una fila sintetica."""
    return int(cast("int", row["traded"]))


def _sum_of(row: Mapping[str, object]) -> float | None:
    """La suma declarada de una fila sintetica, o ``None`` si no se pudo medir."""
    block = row.get("pnl_declared_pct")
    if not isinstance(block, dict):
        return None
    value = cast("dict[str, object]", block).get("sum")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _deltas(
    previous: Sequence[Mapping[str, object]], current: Sequence[Mapping[str, object]]
) -> tuple[regeneration_delta.DeclaredRowDelta, ...]:
    """Los deltas declarados de dos listas de filas sinteticas."""
    return regeneration_delta.declared_row_deltas(
        previous, current, label=_label_of, summary=_sum_of, traded=_traded_of
    )


def _block(rows: Sequence[regeneration_delta.DeclaredRowDelta]) -> dict[str, object]:
    """El bloque `regeneration` de suma declarada, como lo publican los generadores."""
    return regeneration_delta.declared_deltas_block(
        previous_name="informe_previo.json", rows=rows, subject="prueba"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pureza (sin reloj, sin red, sin Store)
# ─────────────────────────────────────────────────────────────────────────────
def test_t30_a_the_module_is_pure_and_reads_only_the_declared_previous_artifact() -> None:
    """T30-A: sin reloj, sin red y sin `Store`: la unica lectura es el artefacto previo."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert attributes & {"now", "utcnow", "today", "time"} == set()
    roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert roots & {"socket", "urllib", "requests", "http", "httpx"} == set()
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "Store" not in called
    assert "read_text" in attributes  # la unica lectura, del previo
    assert "write_text" not in attributes and "write_bytes" not in attributes


def test_t30_b_the_module_does_not_import_the_store_or_the_report_layers() -> None:
    """T30-B: el modulo nuevo no importa `data`, `backtest` ni los otros informes."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    forbidden = sorted(
        name
        for name in modules
        if name.startswith(("cfdtrader.data", "cfdtrader.backtest", "cfdtrader.analysis"))
    )
    assert not forbidden, f"el modulo puro no importa {forbidden}"


# ─────────────────────────────────────────────────────────────────────────────
# Aritmetica del delta declarado
# ─────────────────────────────────────────────────────────────────────────────
def test_t30_c_the_delta_per_operation_is_the_declared_constant() -> None:
    """T30-C: `delta = operadas x 0.004158`; el delta por operacion es la constante."""
    previous = [_row("fila_a", traded=404, total=-1.663373389040144)]
    current = [_row("fila_a", traded=404, total=-1.663373389040144 + 404 * 0.004158)]
    deltas = _deltas(previous, current)
    assert len(deltas) == 1
    row = deltas[0]
    assert row.label == "fila_a"
    assert row.n_traded == 404
    assert row.delta == pytest.approx(404 * 0.004158)
    assert row.expected_delta == pytest.approx(404 * 0.004158)
    assert row.per_operation == pytest.approx(0.004158)
    assert row.consistent is True
    payload = row.to_payload()
    assert payload["declared_delta_per_operation"] == 0.004158
    assert payload["consistent_with_declared_term"] is True


def test_t30_d_rows_that_do_not_trade_or_have_no_sum_are_skipped() -> None:
    """T30-D: las filas sin operaciones o sin serie declarada no entran en el bloque."""
    previous = [
        _row("opera", traded=3, total=0.0),
        _row("no_opera", traded=0, total=None),
        _row("sin_suma", traded=5, total=None),
    ]
    current = [
        _row("opera", traded=3, total=3 * 0.004158),
        _row("no_opera", traded=0, total=None),
        _row("sin_suma", traded=5, total=0.0),
    ]
    deltas = _deltas(previous, current)
    assert [row.label for row in deltas] == ["opera"]
    assert deltas[0].per_operation == pytest.approx(0.004158)


def test_t30_e_a_row_without_a_pair_in_the_new_report_is_a_typed_error() -> None:
    """T30-E: las etiquetas son la identidad de la fila; perder una es error tipado."""
    with pytest.raises(regeneration_delta.MissingRowError):
        _deltas([_row("fila_a", traded=1, total=0.0)], [_row("fila_b", traded=1, total=0.0)])


def test_t30_f_the_block_flags_rows_that_do_not_match_the_declared_constant() -> None:
    """T30-F: si el delta no cuadra con la constante, la bandera lo declara (no lo esconde)."""
    deltas = _deltas(
        [_row("fila", traded=10, total=0.0)], [_row("fila", traded=10, total=10 * 0.001)]
    )
    assert deltas[0].consistent is False
    assert _block(deltas)["all_rows_consistent"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Render Markdown
# ─────────────────────────────────────────────────────────────────────────────
def test_t30_g_the_declared_section_declares_previous_new_and_per_operation() -> None:
    """T30-G: la seccion `## Regeneración` publica previo, nuevo y `0.004158` por operacion."""
    previous_sum = -1.9823072602926337
    current_sum = previous_sum + 500 * 0.004158
    deltas = _deltas(
        [_row("fila_a", traded=500, total=previous_sum)],
        [_row("fila_a", traded=500, total=current_sum)],
    )
    lines = regeneration_delta.render_declared_section(_block(deltas), intro="intro de prueba")
    text = "\n".join(lines)
    assert regeneration_delta.SECTION_TITLE in lines
    assert "0.004158" in text
    assert "fila_a" in text
    assert repr(previous_sum) in text
    assert repr(current_sum) in text
    assert "informe_previo.json" in text


def test_t30_h_the_declared_section_carries_no_digest_literal() -> None:
    """T30-H: el Markdown no publica ningun literal `sha256:` + 64 hex (regla #89/#95/#96)."""
    deltas = _deltas([_row("fila", traded=1, total=0.0)], [_row("fila", traded=1, total=0.004158)])
    text = "\n".join(regeneration_delta.render_declared_section(_block(deltas), intro="x"))
    assert "sha256:" not in text


# ─────────────────────────────────────────────────────────────────────────────
# Deltas de puntero
# ─────────────────────────────────────────────────────────────────────────────
def test_t30_i_pointer_deltas_declare_previous_and_current() -> None:
    """T30-I: el puntero publica el valor previo, el nuevo y si cambio."""
    previous = {"provenance": {"pipeline": {"sha256": "8fabacbd"}}}
    current = {"provenance": {"pipeline": {"sha256": "87d83421"}}}
    pointer = regeneration_delta.pointer_delta(
        previous, current, path=("provenance", "pipeline", "sha256")
    )
    assert pointer.name == "provenance.pipeline.sha256"
    assert pointer.previous == "8fabacbd"
    assert pointer.current == "87d83421"
    assert pointer.changed is True
    payload = pointer.to_payload()
    assert payload["changed"] is True
    assert payload["name"] == "provenance.pipeline.sha256"


def test_t30_j_a_missing_path_is_none_and_the_block_flags_invariants() -> None:
    """T30-J: una ruta ausente da ``None`` y las banderas de invariantes se publican."""
    previous = {"gate": {"aggregate": "fail"}, "cells": [{"slippage_bp": 0.0}]}
    current = {
        "gate": {"aggregate": "fail"},
        "cells": [{"slippage_bp": 0.0}, {"slippage_bp": 13.2}],
    }
    pointer = regeneration_delta.pointer_delta(previous, current, path=("provenance", "sha256"))
    assert pointer.previous is None
    assert pointer.current is None
    assert pointer.changed is False
    block = regeneration_delta.pointer_deltas_block(
        previous_name="previo.json",
        pointers=[pointer],
        unchanged={"slippage_grid_bp": False, "gate_aggregate": True},
    )
    assert block["all_unchanged"] is False
    lines = regeneration_delta.render_pointer_section(
        block,
        intro="intro",
        unchanged_labels={"slippage_grid_bp": "La rejilla", "gate_aggregate": "El veredicto"},
    )
    text = "\n".join(lines)
    assert regeneration_delta.SECTION_TITLE in lines
    assert "previo.json" in text
    assert "La rejilla: **no** cambia" in text
    assert "`false`" in text


# ─────────────────────────────────────────────────────────────────────────────
# Carga del artefacto previo
# ─────────────────────────────────────────────────────────────────────────────
def test_t30_k_load_previous_reads_the_declared_file_or_returns_none(tmp_path: Path) -> None:
    """T30-K: sin ruta devuelve ``None``; con ruta ausente, error tipado."""
    assert regeneration_delta.load_previous(None) is None
    with pytest.raises(regeneration_delta.PreviousArtifactError):
        regeneration_delta.load_previous(Path("/no/existe/artefacto.json"))


def test_t30_l_load_previous_rejects_a_non_object_and_keeps_the_name_only(tmp_path: Path) -> None:
    """T30-L: un JSON que no es objeto es error; la ruta publicada es solo el **nombre**."""
    document = tmp_path / "previo.json"
    document.write_text(json.dumps({"analysis": "x"}), encoding="utf-8")
    assert regeneration_delta.load_previous(document) == {"analysis": "x"}
    assert regeneration_delta.artifact_name(document) == "previo.json"
    assert regeneration_delta.artifact_name(None) is None
    broken = tmp_path / "lista.json"
    broken.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(regeneration_delta.PreviousArtifactError):
        regeneration_delta.load_previous(broken)


def test_t30_m_the_cast_helpers_are_typed_errors() -> None:
    """T30-M: las vistas tipadas del payload fallan con error tipado, no con `TypeError`."""
    with pytest.raises(regeneration_delta.RegenerationError):
        regeneration_delta.cast_mapping([1])
    with pytest.raises(regeneration_delta.RegenerationError):
        regeneration_delta.cast_sequence({"a": 1})
    with pytest.raises(regeneration_delta.RegenerationError):
        regeneration_delta.cast_float(True)
    assert regeneration_delta.cast_mapping({"a": 1}) == {"a": 1}
    assert regeneration_delta.cast_sequence([1, 2]) == [1, 2]
    assert regeneration_delta.cast_float(2) == 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Determinismo entre procesos frescos
# ─────────────────────────────────────────────────────────────────────────────
CHILD: Final[str] = textwrap.dedent(
    """
    import hashlib, json
    from cfdtrader.analysis import regeneration_delta
    from cfdtrader.backtest.engine import canonical_text

    previous_rows = [
        {"label": "a", "traded": 404, "pnl_declared_pct": {"sum": -1.663373389040144}},
        {"label": "b", "traded": 3, "pnl_declared_pct": {"sum": 0.1}},
    ]
    current_rows = [
        {"label": "a", "traded": 404,
         "pnl_declared_pct": {"sum": -1.663373389040144 + 404 * 0.004158}},
        {"label": "b", "traded": 3, "pnl_declared_pct": {"sum": 0.1 + 3 * 0.004158}},
    ]
    deltas = regeneration_delta.declared_row_deltas(
        previous_rows,
        current_rows,
        label=lambda row: str(row["label"]),
        summary=lambda row: float(row["pnl_declared_pct"]["sum"]),
        traded=lambda row: int(row["traded"]),
    )
    block = regeneration_delta.declared_deltas_block(
        previous_name="previo.json", rows=deltas, subject="prueba"
    )
    markdown = "\\n".join(regeneration_delta.render_declared_section(block, intro="x"))
    print(json.dumps({
        "canonical": canonical_text(block),
        "block_sha256": hashlib.sha256(canonical_text(block).encode("utf-8")).hexdigest(),
        "markdown_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
    }))
    """
)


def _child(hash_seed: str) -> dict[str, str]:
    """Corre el bloque en un proceso fresco con esa semilla de ``hash``."""
    environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
    completed = subprocess.run(  # noqa: S603 - el ejecutable es el interprete de la sesion
        [sys.executable, "-c", CHILD],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return cast("dict[str, str]", json.loads(completed.stdout.strip().splitlines()[-1]))


def test_t30_n_the_block_is_identical_in_two_fresh_processes() -> None:
    """T30-N: `PYTHONHASHSEED` 0 y 1 dan el mismo bloque canonico y el mismo Markdown."""
    zero = _child("0")
    one = _child("1")
    assert zero == one
    assert len(zero["block_sha256"]) == 64


def test_t30_o_the_canonical_text_of_the_block_is_stable() -> None:
    """T30-O: el texto canonico del bloque es una cadena y su digest es reproducible."""
    deltas = _deltas(
        [_row("fila", traded=7, total=0.0)], [_row("fila", traded=7, total=7 * 0.004158)]
    )
    block = _block(deltas)
    text = canonical_text(block)
    assert isinstance(text, str) and text
    first = hashlib.sha256(text.encode("utf-8")).hexdigest()
    second = hashlib.sha256(canonical_text(block).encode("utf-8")).hexdigest()
    assert first == second


# ─────────────────────────────────────────────────────────────────────────────
# Integracion: los artefactos regenerados declaran la diferencia
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
@pytest.mark.parametrize(
    ("name", "prefixed"),
    (
        ("baseline_2026-09-22.json", False),
        ("model_comparison_2026-09-22.json", True),
        ("phase2_dominance_2026-09-24.json", True),
    ),
)
def test_t30_p_the_published_artifacts_carry_a_self_consistent_regeneration_block(
    name: str, prefixed: bool
) -> None:
    """T30-P: los tres `.json` regenerados traen un bloque `regeneration` autoconsistente."""
    path = REAL_REPORTS / name
    if not path.is_file():
        pytest.skip(f"{name} no esta en el arbol")
    document = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    block = cast("Mapping[str, object]", document["regeneration"])
    assert block["previous_artifact"]
    report_sha256 = str(document["report_sha256"])
    assert report_sha256.startswith("sha256:") == prefixed
    body = report_sha256.removeprefix("sha256:")
    without = {key: value for key, value in document.items() if key != "report_sha256"}
    assert hashlib.sha256(canonical_text(without).encode("utf-8")).hexdigest() == body
    markdown = path.with_suffix(".md").read_text(encoding="utf-8")
    assert regeneration_delta.SECTION_TITLE in markdown
