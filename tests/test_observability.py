"""Tests de la observabilidad del pipeline (`#43`): run_log JSONL y manifest.

El artefacto verificable son los **dos ficheros** de una ejecución: ``run_log.jsonl`` (una
línea JSON por etapa, con las columnas de ``ops.run_log``) y ``manifest.json`` (versiones,
hashes y resultado). Todo se prueba con un **reloj de mentira** inyectado, así que las
duraciones son deterministas y no hay dependencia del reloj real.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

import pytest
from pydantic import ValidationError

from cfdtrader.journal.decision_log import TABLE_COLUMNS
from cfdtrader.orchestration.observability import (
    MANIFEST_FILENAME,
    RUN_LOG_COLUMNS,
    RUN_LOG_FILENAME,
    Manifest,
    ObservabilityError,
    RunObserver,
    StageRecord,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODULE: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "orchestration" / "observability.py"

RUN_ID: Final[str] = "run-7f3a"
AS_OF: Final[datetime] = datetime(2026, 10, 3, 12, 45, tzinfo=UTC)
GIT_COMMIT: Final[str] = "b1cfdbf"
FEATURES_VERSION: Final[str] = "sha256:" + "a" * 64
GATE_SHA256: Final[str] = "sha256:" + "b" * 64


class FakeClock:
    """Reloj de mentira: devuelve los valores pautados, en orden (el último, repetido)."""

    def __init__(self, *values: float) -> None:
        self._values = list(values)
        self._index = 0

    def __call__(self) -> float:
        value = self._values[min(self._index, len(self._values) - 1)]
        self._index += 1
        return value


def _observer(root: Path, clock: FakeClock) -> RunObserver:
    return RunObserver(root, run_id=RUN_ID, as_of=AS_OF, git_commit=GIT_COMMIT, timer=clock)


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ─────────────────────────────────────────────────────────────────────────────
# A1 · Contrato: `as_of` explícito, cronómetro inyectable y sin red
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_observer_requires_as_of_and_accepts_an_injected_timer() -> None:
    """`as_of`, `run_id` y `git_commit` son entradas declaradas; el cronómetro se inyecta."""
    parameters = inspect.signature(RunObserver.__init__).parameters

    assert parameters["as_of"].default is inspect.Parameter.empty
    assert parameters["run_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["as_of"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["timer"].default is time.perf_counter


def test_a1_the_module_does_not_talk_to_the_network() -> None:
    """La observabilidad es local: nada de red."""
    source = MODULE.read_text(encoding="utf-8")
    for forbidden in ("import yfinance", "import requests", "import urllib", "import httpx"):
        assert forbidden not in source


# ─────────────────────────────────────────────────────────────────────────────
# A2 · Una ejecución correcta escribe run_log y manifest
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_a_successful_run_writes_the_log_and_the_manifest(tmp_path: Path) -> None:
    """Dos etapas limpias dejan dos líneas `ok` y un manifest con versiones, hashes y estado."""
    clock = FakeClock(0.0, 0.012, 0.012, 0.030)
    with _observer(tmp_path, clock) as observer:
        observer.add_version("features_version", FEATURES_VERSION)
        observer.add_hash("gate_sha256", GATE_SHA256)
        with observer.stage("features"):
            pass
        with observer.stage("gate"):
            pass

    log_path = observer.directory / RUN_LOG_FILENAME
    manifest_path = observer.directory / MANIFEST_FILENAME
    entries = _lines(log_path)
    assert [entry["stage"] for entry in entries] == ["features", "gate"]
    assert [entry["duration_ms"] for entry in entries] == [12, 18]
    assert all(entry["ok"] is True and entry["error"] is None for entry in entries)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == RUN_ID
    assert manifest["git_commit"] == GIT_COMMIT
    assert manifest["versions"] == {"features_version": FEATURES_VERSION}
    assert manifest["hashes"] == {"gate_sha256": GATE_SHA256}
    assert manifest["stages"] == ["features", "gate"]
    assert manifest["ok"] is True


# ─────────────────────────────────────────────────────────────────────────────
# A3 · Un fallo no se pierde: traza completa, re-lanzado y manifest en `ok=False`
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_a_failing_stage_keeps_its_traceback_and_reraises(tmp_path: Path) -> None:
    """Una etapa que lanza deja su traza, marca el manifest y **vuelve a lanzar**."""
    clock = FakeClock(0.0, 0.004)
    observer = _observer(tmp_path, clock)
    with pytest.raises(RuntimeError, match="boom"), observer, observer.stage("gate"):
        raise RuntimeError("boom")

    entry = _lines(observer.directory / RUN_LOG_FILENAME)[0]
    assert entry["stage"] == "gate"
    assert entry["ok"] is False
    assert entry["duration_ms"] == 4
    assert isinstance(entry["error"], str)
    assert "Traceback" in entry["error"] and "RuntimeError: boom" in entry["error"]
    manifest = json.loads((observer.directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["ok"] is False


# ─────────────────────────────────────────────────────────────────────────────
# A4 · JSONL estricto: un objeto por línea, con las columnas de `ops.run_log`
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_the_log_is_strict_jsonl_with_the_ops_columns(tmp_path: Path) -> None:
    """El fichero es JSONL puro: termina en salto y cada línea trae exactamente las columnas."""
    observer = RunObserver(
        tmp_path, run_id=RUN_ID, as_of=AS_OF, git_commit=GIT_COMMIT, timer=FakeClock(0.0)
    )
    observer.record("alpha", duration_ms=1, ok=True)
    observer.record("beta", duration_ms=2, ok=True)
    log_path, _ = observer.finalize()

    text = log_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    lines = text.splitlines()
    assert len(lines) == 2
    for line in lines:
        assert set(json.loads(line)) == set(RUN_LOG_COLUMNS)


# ─────────────────────────────────────────────────────────────────────────────
# A5 · Una sola definición del esquema (el de `ops.run_log`, §12.6)
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_the_columns_are_the_ops_run_log_schema() -> None:
    """Las columnas del `run_log` se importan del diario: no hay una segunda definición."""
    assert tuple(TABLE_COLUMNS["run_log"]) == RUN_LOG_COLUMNS
    assert tuple(StageRecord.model_fields) == RUN_LOG_COLUMNS


# ─────────────────────────────────────────────────────────────────────────────
# A6 · Registro manual y manifest antes de finalizar
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_a_manual_record_and_the_manifest_before_finalize(tmp_path: Path) -> None:
    """`record` añade etapas sin cronómetro y `manifest()` las refleja antes de escribir."""
    observer = RunObserver(
        tmp_path, run_id=RUN_ID, as_of=AS_OF, git_commit=GIT_COMMIT, timer=FakeClock(0.0)
    )
    observer.record("fetched", duration_ms=7, ok=True)
    observer.record("score", duration_ms=3, ok=False, error="boom")

    manifest = observer.manifest()
    assert manifest.stages == ("fetched", "score")
    assert manifest.ok is False
    log_path, manifest_path = observer.finalize()
    assert log_path.is_file() and manifest_path.is_file()
    assert [entry["ok"] for entry in _lines(log_path)] == [True, False]


# ─────────────────────────────────────────────────────────────────────────────
# A7 · Determinismo: mismas entradas y mismo cronómetro ⇒ mismos bytes
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_the_output_is_deterministic(tmp_path: Path) -> None:
    """Dos ejecuciones con las mismas entradas y el mismo reloj escriben bytes idénticos."""

    def produce() -> tuple[bytes, bytes]:
        clock = FakeClock(0.0, 0.005)
        observer = _observer(tmp_path, clock)
        with observer, observer.stage("fetch"):
            pass
        return (
            (observer.directory / RUN_LOG_FILENAME).read_bytes(),
            (observer.directory / MANIFEST_FILENAME).read_bytes(),
        )

    assert produce() == produce()


# ─────────────────────────────────────────────────────────────────────────────
# A8 · Los modelos son inmutables y de esquema cerrado
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_the_models_are_frozen_and_forbid_extra_fields() -> None:
    """Una etapa y un manifest no admiten campos inventados ni se reescriben."""
    assert StageRecord.model_config.get("frozen") is True
    assert Manifest.model_config.get("frozen") is True
    with pytest.raises(ValidationError):
        StageRecord.model_validate(
            {"run_id": "r", "as_of": AS_OF, "stage": "s", "duration_ms": 0, "ok": True, "extra": 1}
        )
    record = StageRecord(run_id="r", as_of=AS_OF, stage="s", duration_ms=0, ok=True)
    with pytest.raises(ValidationError):
        record.stage = "otro"


# ─────────────────────────────────────────────────────────────────────────────
# A9 · Entradas inválidas: errores tipados, nunca un valor silencioso
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_a_blank_run_id_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(ObservabilityError, match="run_id"):
        RunObserver(tmp_path, run_id="  ", as_of=AS_OF, git_commit=GIT_COMMIT)


def test_a9_a_naive_as_of_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(ObservabilityError, match="as_of"):
        RunObserver(
            tmp_path, run_id=RUN_ID, as_of=datetime(2026, 10, 3, 12, 45), git_commit=GIT_COMMIT
        )


def test_a9_a_blank_git_commit_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(ObservabilityError, match="git_commit"):
        RunObserver(tmp_path, run_id=RUN_ID, as_of=AS_OF, git_commit="")


def test_a9_a_non_callable_timer_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(ObservabilityError, match="timer"):
        RunObserver(
            tmp_path,
            run_id=RUN_ID,
            as_of=AS_OF,
            git_commit=GIT_COMMIT,
            timer=cast("Callable[[], float]", object()),
        )


def test_a9_a_negative_or_boolean_duration_is_a_typed_error(tmp_path: Path) -> None:
    observer = RunObserver(
        tmp_path, run_id=RUN_ID, as_of=AS_OF, git_commit=GIT_COMMIT, timer=FakeClock(0.0)
    )
    with pytest.raises(ObservabilityError, match="duration_ms"):
        observer.record("x", duration_ms=-1)
    with pytest.raises(ObservabilityError, match="duration_ms"):
        observer.record("x", duration_ms=True)


# ─────────────────────────────────────────────────────────────────────────────
# A10 · Aislamiento: la observabilidad escribe solo bajo la raíz que se le da
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_the_observer_writes_only_under_the_given_root(tmp_path: Path) -> None:
    """Los dos ficheros van bajo ``<root>/<run_id>`` y nada fuera de la raíz."""
    clock = FakeClock(0.0, 0.001)
    observer = _observer(tmp_path, clock)
    with observer, observer.stage("fetch"):
        pass

    written = sorted(
        str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()
    )
    assert written == [f"{RUN_ID}/{MANIFEST_FILENAME}", f"{RUN_ID}/{RUN_LOG_FILENAME}"]


def test_a9_a_non_datetime_as_of_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(ObservabilityError, match="as_of"):
        RunObserver(
            tmp_path,
            run_id=RUN_ID,
            as_of=cast("datetime", "2026-10-03T12:45:00+00:00"),
            git_commit=GIT_COMMIT,
        )


def test_a9_a_blank_version_or_hash_is_a_typed_error(tmp_path: Path) -> None:
    observer = RunObserver(
        tmp_path, run_id=RUN_ID, as_of=AS_OF, git_commit=GIT_COMMIT, timer=FakeClock(0.0)
    )
    with pytest.raises(ObservabilityError, match="version"):
        observer.add_version("", "x")
    with pytest.raises(ObservabilityError, match="version"):
        observer.add_version("features_version", "")
    with pytest.raises(ObservabilityError, match="hash"):
        observer.add_hash("", "x")
    with pytest.raises(ObservabilityError, match="hash"):
        observer.add_hash("gate_sha256", "")


# ─────────────────────────────────────────────────────────────────────────────
# A11 · `fail` y `records`: marcar un cierre externo sin inventar una etapa
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_fail_marks_the_run_without_recording_a_stage(tmp_path: Path) -> None:
    observer = RunObserver(
        tmp_path, run_id=RUN_ID, as_of=AS_OF, git_commit=GIT_COMMIT, timer=FakeClock(0.0)
    )
    assert observer.run_id == RUN_ID
    assert observer.records == ()
    observer.fail("cierre externo")
    assert observer.manifest().ok is False
    assert observer.records == ()
    with pytest.raises(ObservabilityError, match="error"):
        observer.fail("")


# ─────────────────────────────────────────────────────────────────────────────
# #129 · Los conteos del manifest: cuanto paso, no quien era
# ─────────────────────────────────────────────────────────────────────────────
def test_b5_the_manifest_publishes_the_counters(tmp_path: Path) -> None:
    """Los conteos viajan en el `manifest` y se leen de vuelta desde el fichero."""
    with _observer(tmp_path, FakeClock(0.0)) as observer:
        observer.add_counter("headlines_read", 5)
        observer.add_counter("headlines_sent", 3)

    manifest = json.loads((tmp_path / RUN_ID / "manifest.json").read_text(encoding="utf-8"))
    expected = {"headlines_read": 5, "headlines_sent": 3}
    assert manifest["counters"] == expected
    assert observer.manifest().counters == expected


def test_b5_an_execution_without_counters_declares_an_empty_map(tmp_path: Path) -> None:
    """Sin conteos el mapa va vacio, no ausente: el lector no tiene que defenderse de un `null`."""
    with _observer(tmp_path, FakeClock(0.0)):
        pass
    manifest = json.loads((tmp_path / RUN_ID / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counters"] == {}


@pytest.mark.parametrize("value", [-1, True, "3", 2.0])
def test_b6_add_counter_rejects_what_is_not_a_count(tmp_path: Path, value: object) -> None:
    """Un negativo, un booleano, una cadena o un flotante no son un conteo: se cortan al entrar."""
    observer = RunObserver(tmp_path, run_id=RUN_ID, as_of=AS_OF, git_commit=GIT_COMMIT)
    with pytest.raises(ObservabilityError, match="counter"):
        observer.add_counter("headlines_read", value)  # type: ignore[arg-type]


def test_b6_add_counter_rejects_a_blank_name(tmp_path: Path) -> None:
    """Un conteo sin nombre no se puede leer despues: es un dato perdido con apariencia de dato."""
    observer = RunObserver(tmp_path, run_id=RUN_ID, as_of=AS_OF, git_commit=GIT_COMMIT)
    with pytest.raises(ObservabilityError, match="counter"):
        observer.add_counter("  ", 1)


def test_b8_the_counters_travel_as_integers_with_sorted_keys(tmp_path: Path) -> None:
    """JSON puro: el conteo viaja como entero (no como cadena) y las claves van ordenadas."""
    with _observer(tmp_path, FakeClock(0.0)) as observer:
        observer.add_counter("headlines_sent", 3)
        observer.add_counter("headlines_read", 5)

    text = (tmp_path / RUN_ID / "manifest.json").read_text(encoding="utf-8")
    assert '"headlines_read":5' in text, "un entero, no la cadena '5'"
    assert '"headlines_read":"5"' not in text, "una cadena falsearia cualquier agregado"
    assert text.index('"headlines_read"') < text.index('"headlines_sent"'), "claves ordenadas"
