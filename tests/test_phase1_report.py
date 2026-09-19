"""Pruebas del informe de Fase 1 y de su puerta de salida (`tasks.md`, tarea 18) — #18.

Un test por criterio (``test_a1_...`` … ``test_a34_...``), siempre con ``tmp_path`` y la
fixture de sesion de ``tests/conftest.py`` que **huella ``data/``**. Los bordes incomodos
(artefacto ausente, dos artefactos ambiguos, JSON alterado a mano, ``--as-of`` ausente,
segunda pasada, hash manipulado y ``write=False``) tienen su propio caso: en este proyecto
los defectos aparecen al **reejecutar**, no en la primera pasada.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, cast

import pytest

from cfdtrader.analysis import backtest_report, drift, phase0_report, phase1_report
from cfdtrader.analysis.phase0_report import BlockerCode, GateVerdict, HalfResult
from cfdtrader.analysis.phase1_report import (
    CLOCK_DIVERGENCE,
    EVIDENCE_ARTIFACT,
    EVIDENCE_RERUN,
    FOLLOW_UPS,
    HALF_ALWAYS_LONG,
    HALF_DETERMINISM,
    REPORT_DOES_NOT_DO,
    REPORT_HASH_FORMAT,
    AmbiguousInputArtifactError,
    ConservationError,
    InputArtifact,
    MissingInputArtifactError,
    Phase1Report,
    Phase1ReportError,
    VerdictError,
)
from cfdtrader.backtest.costs import (
    Side,
    SlippageParameter,
    declared_cost_model,
    declared_slippage_assumption,
)
from cfdtrader.backtest.engine import canonical_text
from cfdtrader.backtest.metrics import MetricsInputError, calculate_metrics
from cfdtrader.data.store import Store

MODULE_PATH: Final[Path] = Path(str(phase1_report.__file__))
SOURCE: Final[str] = MODULE_PATH.read_text(encoding="utf-8")
TEST_PATH: Final[Path] = Path(__file__).resolve()
REPO_ROOT: Final[Path] = TEST_PATH.parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"
REAL_REPORTS: Final[Path] = REAL_DATA / "derived" / "reports"
REAL_ARTIFACT: Final[Path] = REAL_REPORTS / f"{backtest_report.REPORT_PREFIX}_2026-09-19.json"

NOW: Final[datetime] = datetime(2026, 9, 19, tzinfo=UTC)

#: Literales de la tabla declarada (#8) que el modulo **no** puede contener (A27).
DECLARED_TABLE_LITERALS: Final[tuple[str, ...]] = (
    "0.42",
    "0.0042",
    "1.82",
    "-0.18",
    "0.24",
    "2.24",
    "0.05",
    "250",
)

#: Claves que el informe **no** puede publicar con valor numerico: son las metricas de #15 (A31).
FORBIDDEN_METRIC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "sharpe",
        "sharpe_ratio",
        "sortino",
        "sortino_ratio",
        "ev",
        "expected_value",
        "expected_value_pct",
        "bootstrap",
        "confidence_interval",
        "equity_curve",
        "max_drawdown",
        "calibration",
        "brier_score",
    }
)

#: Prefijos que la suite tiene que declarar, uno por criterio (A24).
REQUIRED_TEST_PREFIXES: Final[tuple[str, ...]] = tuple(f"test_a{i}_" for i in range(1, 35))


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def _business_days(start: date, count: int) -> list[date]:
    """Dias laborables (lunes a viernes) consecutivos desde ``start``."""
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


SYNTHETIC_SESSIONS: Final[tuple[date, ...]] = tuple(_business_days(date(2024, 1, 2), 600))


def _fingerprint(root: Path) -> dict[str, str]:
    """Ruta relativa -> sha256 de cada fichero (la huella de ``tests/conftest.py``)."""
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _mapping(node: object) -> dict[str, object]:
    """Vista tipada de un nodo del payload (que es JSON puro por contrato)."""
    return cast("dict[str, object]", node)


def _block(report: phase1_report.Phase1Report, *keys: str) -> dict[str, object]:
    """Un bloque anidado del payload, ya tipado."""
    node: object = report.payload
    for key in keys:
        node = _mapping(node)[key]
    return _mapping(node)


def _list_of_mappings(node: object) -> list[dict[str, object]]:
    """Lista de objetos JSON ya tipada."""
    return [cast("dict[str, object]", item) for item in cast("list[object]", node)]


def _rows(report: phase1_report.Phase1Report) -> list[dict[str, object]]:
    """Las filas de la tabla de baselines publicada."""
    return _list_of_mappings(_block(report, "baselines")["rows"])


def _keys(node: object) -> set[str]:
    """Todas las claves de un payload anidado."""
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in cast("dict[object, object]", node).items():
            found.add(str(key))
            found |= _keys(value)
    elif isinstance(node, (list, tuple)):
        for item in cast("list[object] | tuple[object, ...]", node):
            found |= _keys(item)
    return found


def _numeric_items(node: object, path: str = "") -> list[tuple[str, object]]:
    """Pares (ruta, valor) de todo lo numerico del payload."""
    found: list[tuple[str, object]] = []
    if isinstance(node, dict):
        for key, value in cast("dict[object, object]", node).items():
            found += _numeric_items(value, f"{path}.{key}")
    elif isinstance(node, (list, tuple)):
        for index, item in enumerate(cast("list[object] | tuple[object, ...]", node)):
            found += _numeric_items(item, f"{path}[{index}]")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        found.append((path, node))
    return found


def _all_pairs(node: object) -> Iterator[tuple[str, object]]:
    """Todos los pares (clave, valor) del payload, en profundidad."""
    if isinstance(node, dict):
        for key, value in cast("dict[object, object]", node).items():
            yield (str(key), value)
            yield from _all_pairs(value)
    elif isinstance(node, (list, tuple)):
        for item in cast("list[object] | tuple[object, ...]", node):
            yield from _all_pairs(item)


def _artifact_payload(path: Path) -> dict[str, object]:
    """El artefacto de #69 leido de disco, ya tipado."""
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _daily_records(
    sessions: Sequence[date], *, first_close: float = 100.0
) -> list[dict[str, object]]:
    """Filas diarias sinteticas con OHLC sano y **sin** `open` repetido (regla de #52)."""
    records: list[dict[str, object]] = []
    close = first_close
    for session in sessions:
        open_px = close * 1.0009  # nunca igual al cierre previo: no dispara `open_stale`
        close = open_px * 1.0004
        records.append(
            {
                "source": "yfinance",
                "series_id": backtest_report.SERIES_ID,
                "as_of": datetime(session.year, session.month, session.day, 21, tzinfo=UTC),
                "fetched_at": NOW,
                "published_at": None,
                "open": open_px,
                "high": max(open_px, close) * 1.001,
                "low": min(open_px, close) * 0.999,
                "close": close,
                "volume": 1_000.0,
            }
        )
    return records


def _labels_records(sessions: Sequence[date]) -> list[dict[str, object]]:
    """Filas de ``derived.labels``: una por sesion, con su fecha ET."""
    return [
        {
            "source": "cfdtrader.models.labels",
            "series_id": backtest_report.SERIES_ID,
            "as_of": datetime(session.year, session.month, session.day, 21, tzinfo=UTC),
            "fetched_at": NOW,
            "published_at": None,
            "session": session,
        }
        for session in sessions
    ]


def _write_store(
    root: Path,
    *,
    sessions: Sequence[date] = SYNTHETIC_SESSIONS,
    daily: Sequence[Mapping[str, object]] | None = None,
) -> Store:
    """Almacen temporal con las sesiones diarias y sus etiquetas."""
    store = Store(root)
    store.append(
        "raw",
        "market_daily",
        [dict(item) for item in (daily if daily is not None else _daily_records(sessions))],
    )
    store.append("derived", "labels", [dict(item) for item in _labels_records(sessions)])
    return store


def _synthetic_with_artifact(root: Path, sessions: Sequence[date] = SYNTHETIC_SESSIONS) -> Path:
    """Escribe el almacen sintetico y su artefacto de #69; devuelve el directorio de informes."""
    _write_store(root, sessions=sessions)
    reports = root / "derived" / "reports"
    backtest_report.analyse(store=Store(root), reports_dir=reports, as_of=NOW, write=True)
    return reports


def _artifact_pair(reports_dir: Path, source: Path) -> None:
    """Copia el par `.json`/`.md` de un artefacto en el directorio indicado."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".json", ".md"):
        shutil.copy(source.with_suffix(suffix), reports_dir / source.with_suffix(suffix).name)


def _run_module(
    module: str, args: Sequence[str], *, seed: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Ejecuta un modulo del proyecto en un **proceso nuevo**, con su `PYTHONHASHSEED`."""
    environment = dict(os.environ)
    if seed is not None:
        environment["PYTHONHASHSEED"] = seed
    return subprocess.run(  # noqa: S603 - el ejecutable es el interprete de la sesion
        [sys.executable, "-m", module, *args],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Almacenes y corridas de la suite
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Raiz con 600 sesiones sinteticas y el artefacto de #69 ya generado encima."""
    root = tmp_path_factory.mktemp("phase1_report")
    _synthetic_with_artifact(root)
    return root


@pytest.fixture(scope="module")
def synthetic_report(synthetic_root: Path) -> phase1_report.Phase1Report:
    """Corrida del informe sobre el almacen sintetico, **sin escribir nada**."""
    return phase1_report.analyse(
        store=Store(synthetic_root),
        reports_dir=synthetic_root / "derived" / "reports",
        as_of=NOW,
        write=False,
    )


@pytest.fixture(scope="module")
def real_report() -> phase1_report.Phase1Report:
    """Corrida sobre el historico real, en **solo lectura**; se salta si no esta en el arbol."""
    if not REAL_ARTIFACT.is_file():
        pytest.skip("el artefacto real de #69 no esta en el arbol")
    return phase1_report.analyse(
        store=Store(REAL_DATA), reports_dir=REAL_REPORTS, as_of=NOW, write=False
    )


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, phase1_report.Phase1Report]:
    """Copia real del artefacto en un directorio temporal y una corrida con ``write=True``."""
    if not REAL_ARTIFACT.is_file():
        pytest.skip("el artefacto real de #69 no esta en el arbol")
    reports = tmp_path_factory.mktemp("written") / "derived" / "reports"
    _artifact_pair(reports, REAL_ARTIFACT)
    report = phase1_report.analyse(
        store=Store(REAL_DATA), reports_dir=reports, as_of=NOW, write=True
    )
    return reports, report


# ─────────────────────────────────────────────────────────────────────────────
# A1 · Ficheros, API minima y frontera de capas
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_module_api_and_layer_boundary() -> None:
    """A1: el modulo y su test existen, el `__all__` es explicito y la capa no se invierte."""
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    exported = phase1_report.__all__
    assert len(exported) == len(set(exported))
    for name in ("analyse", "main", "Phase1Report", "Phase1ReportError", "REPORT_PREFIX"):
        assert name in exported
    backtest_sources = sorted((REPO_ROOT / "src" / "cfdtrader" / "backtest").glob("*.py"))
    assert backtest_sources
    for path in backtest_sources:
        assert "phase1_report" not in path.read_text(encoding="utf-8")
    # `phase0_report` y `drift` siguen con su contrato publico intacto.
    assert list(inspect.signature(phase0_report.analyse).parameters) == [
        "reports_dir",
        "now",
        "write",
    ]
    assert callable(drift.clean_sample_cutoff)
    assert callable(phase0_report.select_artifact)


# ─────────────────────────────────────────────────────────────────────────────
# A2 · Firmas y reloj prohibido
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_signatures_and_no_clock(synthetic_report: phase1_report.Phase1Report) -> None:
    """A2: las firmas declaradas y **ninguna** lectura del reloj en el fuente."""
    signature = inspect.signature(phase1_report.analyse)
    assert list(signature.parameters) == ["store", "reports_dir", "as_of", "write"]
    for name in ("store", "reports_dir", "as_of", "write"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["write"].default is True
    assert list(inspect.signature(phase1_report.main).parameters) == ["argv"]
    for token in ("datetime.now", "datetime.utcnow", "date.today", "time.time"):
        assert token not in SOURCE
    assert synthetic_report.payload["generated_at"] == NOW.isoformat()
    assert synthetic_report.as_of == NOW


def test_a2_clock_divergence_is_declared() -> None:
    """A2/A3: la divergencia con la CLI de #9 se declara **sin** nombrar el reloj prohibido."""
    assert "phase0_report" in CLOCK_DIVERGENCE
    assert "--now" in CLOCK_DIVERGENCE
    assert "reloj" in CLOCK_DIVERGENCE or "prohibido" in CLOCK_DIVERGENCE


# ─────────────────────────────────────────────────────────────────────────────
# A3 · `--as-of` obligatorio para escribir
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_as_of_is_mandatory_to_write(tmp_path: Path, synthetic_root: Path) -> None:
    """A3: sin `--as-of` (o con uno invalido) sale 2, avisa por `stderr` y no escribe nada."""
    reports = tmp_path / "derived" / "reports"
    _artifact_pair(reports, synthetic_root / "derived" / "reports" / REAL_ARTIFACT.name)
    before = _fingerprint(reports)

    result = phase1_report.main(["--data-root", str(synthetic_root), "--reports-dir", str(reports)])
    assert result == 2
    assert _fingerprint(reports) == before

    invalid = phase1_report.main(
        ["--data-root", str(synthetic_root), "--reports-dir", str(reports), "--as-of", "nope"]
    )
    assert invalid == 2
    assert _fingerprint(reports) == before

    ok = phase1_report.main(
        [
            "--data-root",
            str(synthetic_root),
            "--reports-dir",
            str(reports),
            "--as-of",
            NOW.isoformat(),
        ]
    )
    assert ok == 0
    added = sorted(set(_fingerprint(reports)) - set(before))
    assert added == ["phase1_report_2026-09-19.json", "phase1_report_2026-09-19.md"]


# ─────────────────────────────────────────────────────────────────────────────
# A4 · Nombres de fichero y artefactos ajenos
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_file_names_and_foreign_artifacts(tmp_path: Path, synthetic_root: Path) -> None:
    """A4: exactamente dos ficheros nuevos y ningun artefacto ajeno se toca."""
    reports = tmp_path / "derived" / "reports"
    _artifact_pair(reports, synthetic_root / "derived" / "reports" / REAL_ARTIFACT.name)
    foreign = {
        "phase0_report_2026-09-18.json": "{}\n",
        "drift_decomposition_2026-09-18.json": "{}\n",
        "volatility_forecast_2026-09-18.json": "{}\n",
        "cost_audit_2026-09-18.json": "{}\n",
        "triple_barrier_2026-09-18.json": "{}\n",
    }
    for name, text in foreign.items():
        (reports / name).write_text(text, encoding="utf-8")
    (reports / "phase1_report_2020-01-01.json").write_text("{}\n", encoding="utf-8")
    before = _fingerprint(reports)

    report = phase1_report.analyse(
        store=Store(synthetic_root), reports_dir=reports, as_of=NOW, write=True
    )
    assert report.report_stem == "phase1_report_2026-09-19"
    after = _fingerprint(reports)
    assert sorted(set(after) - set(before)) == [
        "phase1_report_2026-09-19.json",
        "phase1_report_2026-09-19.md",
    ]
    for name in foreign:
        assert after[name] == before[name]
    assert after["phase1_report_2020-01-01.json"] == before["phase1_report_2020-01-01.json"]


# ─────────────────────────────────────────────────────────────────────────────
# A5 · Seleccion del artefacto de entrada reutilizando #9
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_artifact_selection_reuses_phase0(tmp_path: Path) -> None:
    """A5: cero candidatos y ambiguedad son errores tipados propios; la seleccion se importa."""
    assert phase1_report.select_artifact is phase0_report.select_artifact
    assert phase1_report.INPUT_CLASS.pattern == "phase1_backtest_*.json"
    assert "glob(" not in SOURCE

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(MissingInputArtifactError):
        phase1_report.load_input_artifact(empty, store=Store(tmp_path))

    source = _synthetic_with_artifact(tmp_path / "store") / REAL_ARTIFACT.name
    ambiguous = tmp_path / "ambiguous"
    ambiguous.mkdir()
    shutil.copy(source, ambiguous / "phase1_backtest_2026-09-19.json")
    # La misma fecha sin rellenar el mes: `strptime` la lee igual (2026-09-19) ⇒ ambiguedad.
    shutil.copy(source, ambiguous / "phase1_backtest_2026-9-19.json")
    with pytest.raises(AmbiguousInputArtifactError):
        phase1_report.load_input_artifact(ambiguous, store=Store(tmp_path))


def test_a5_missing_markdown_companion_is_an_error(tmp_path: Path) -> None:
    """A5/A23: el entregable de #69 es un par `.json`/`.md`; medio par es un error tipado."""
    source = _synthetic_with_artifact(tmp_path / "store") / REAL_ARTIFACT.name
    reports = tmp_path / "derived" / "reports"
    reports.mkdir(parents=True)
    shutil.copy(source, reports / source.name)
    with pytest.raises(MissingInputArtifactError):
        phase1_report.load_input_artifact(reports, store=Store(tmp_path))


# ─────────────────────────────────────────────────────────────────────────────
# A6 · Consume, no recalcula la tabla
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_consumes_artifact_without_rewriting(
    real_report: phase1_report.Phase1Report, written: tuple[Path, phase1_report.Phase1Report]
) -> None:
    """A6: ruta relativa y huella del artefacto, tabla copiada y par de #69 intacto."""
    reports, _ = written
    raw = REAL_ARTIFACT.read_bytes()
    artifact = _artifact_payload(REAL_ARTIFACT)
    source = _block(real_report, "source_artifact")
    assert source["path"] == f"derived/reports/{REAL_ARTIFACT.name}"
    assert source["sha256"] == hashlib.sha256(raw).hexdigest()
    assert source["sha256_of"] == "fichero .json"
    assert source["report_sha256"] == artifact["report_sha256"]
    assert source["evidence"] == EVIDENCE_ARTIFACT
    assert source["selection_rule"] == phase0_report.FILE_SELECTION_RULE

    published = {cast("str", row["baseline"]): row for row in _rows(real_report)}
    artifact_rows = {
        cast("str", row["baseline"]): row
        for row in _list_of_mappings(_mapping(artifact["baselines"])["rows"])
    }
    assert published.keys() == artifact_rows.keys()
    for name, row in published.items():
        for key in ("n_test", "traded", "no_trade", "skipped", "not_in_any_test", "run_sha256"):
            assert row[key] == artifact_rows[name][key], key
    assert _block(real_report, "baselines")["evidence"] == EVIDENCE_ARTIFACT
    # El par `.json`/`.md` de #69 sigue con la huella de antes de la corrida.
    assert (reports / REAL_ARTIFACT.name).read_bytes() == raw
    markdown = reports / REAL_ARTIFACT.with_suffix(".md").name
    assert markdown.read_bytes() == REAL_ARTIFACT.with_suffix(".md").read_bytes()
    assert REAL_ARTIFACT.read_bytes() == raw


# ─────────────────────────────────────────────────────────────────────────────
# A7 · Auditoria de determinismo con evidencia propia (procesos nuevos)
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_determinism_rerun_in_new_processes(tmp_path: Path) -> None:
    """A7: el test **reproduce** el artefacto de #69 en procesos nuevos, byte a byte."""
    if not REAL_ARTIFACT.is_file():
        pytest.skip("el artefacto real de #69 no esta en el arbol")
    generated_at = cast("str", _artifact_payload(REAL_ARTIFACT)["generated_at"])
    before = REAL_ARTIFACT.read_bytes()
    before_md = REAL_ARTIFACT.with_suffix(".md").read_bytes()
    seen: set[str] = set()
    for index, seed in enumerate(("0", "1", "random")):
        out = tmp_path / f"run{index}"
        out.mkdir()
        result = _run_module(
            "cfdtrader.analysis.backtest_report",
            ["--data-root", str(REAL_DATA), "--reports-dir", str(out), "--as-of", generated_at],
            seed=seed,
        )
        assert result.returncode == 0, result.stderr
        produced = out / REAL_ARTIFACT.name
        assert produced.read_bytes() == before
        assert produced.with_suffix(".md").read_bytes() == before_md
        seen.add(hashlib.sha256(produced.read_bytes()).hexdigest())
    assert len(seen) == 1
    # Segunda pasada **en el mismo directorio**: el defecto tipico aparece al reejecutar.
    second = _run_module(
        "cfdtrader.analysis.backtest_report",
        [
            "--data-root",
            str(REAL_DATA),
            "--reports-dir",
            str(tmp_path / "run0"),
            "--as-of",
            generated_at,
        ],
        seed="7",
    )
    assert second.returncode == 0, second.stderr
    assert (tmp_path / "run0" / REAL_ARTIFACT.name).read_bytes() == before
    assert REAL_ARTIFACT.read_bytes() == before


def test_a7_published_determinism_block_has_observed_hashes(
    real_report: phase1_report.Phase1Report,
) -> None:
    """A7: el bloque publicado trae el estado con las huellas **observadas**."""
    block = _block(real_report, "harness_audit", "determinism")
    assert block["state"] == str(HalfResult.PASS)
    assert block["evidence"] == EVIDENCE_RERUN
    assert block["reproduced_byte_for_byte"] is True
    expected = cast("dict[str, str]", block["expected"])
    observed = cast("dict[str, str]", block["observed"])
    assert set(expected) == {"report_sha256", "json_sha256", "md_sha256"}
    assert observed == expected
    assert observed["report_sha256"] == _block(real_report, "source_artifact")["report_sha256"]
    assert "backtest_report" in cast("str", block["command"])


# ─────────────────────────────────────────────────────────────────────────────
# A8 · Purga y embargo tal cual, con `h = 0` declarado no-op
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_purge_embargo_no_ops(
    real_report: phase1_report.Phase1Report, written: tuple[Path, phase1_report.Phase1Report]
) -> None:
    """A8: los cuatro valores coinciden con el artefacto y se declaran no-ops estructurales."""
    reports, _ = written
    plan = _mapping(_artifact_payload(reports / REAL_ARTIFACT.name)["plan"])
    block = _block(real_report, "harness_audit", "purge_embargo")
    for key in ("purge_total", "embargo_total", "embargo_in_train_total", "exclusions_are_no_op"):
        assert block[key] == plan[key], key
    assert block["label_horizon"] == plan["label_horizon"] == 0
    assert block["structural_no_op"] is True
    assert block["presented_as_active_filter"] is False
    assert "no-op" in cast("str", block["note"])
    assert block["evidence"] == EVIDENCE_ARTIFACT


# ─────────────────────────────────────────────────────────────────────────────
# A9 · Coste declarado reproducido exacto
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_declared_costs_reproduced_exactly(real_report: phase1_report.Phase1Report) -> None:
    """A9: los valores declarados se reproducen con `Decimal` exacto, importando el modelo."""
    block = _block(real_report, "harness_audit", "costs")
    model = declared_cost_model()
    slippage = declared_slippage_assumption()
    assert block["state"] == str(HalfResult.PASS)
    assert block["evidence"] == EVIDENCE_RERUN

    spread = _mapping(block["spread"])
    assert Decimal(cast("str", spread["usd"])) == Decimal("0.42")
    assert Decimal(cast("str", spread["pct"])) == Decimal("0.0042")
    assert Decimal(cast("str", spread["entry_pct"])) == model.spread_entry_pct
    assert Decimal(cast("str", spread["exit_pct"])) == model.spread_exit_pct

    carry = _mapping(block["carry_per_night"])
    short = _mapping(carry[Side.SHORT.value])
    long = _mapping(carry[Side.LONG.value])
    assert Decimal(cast("str", short["usd"])) == Decimal("-0.18")
    assert Decimal(cast("str", short["pct"])) == Decimal("-0.0018")
    assert Decimal(cast("str", long["usd"])) == Decimal("1.82")
    assert Decimal(cast("str", long["pct"])) == Decimal("0.0182")
    assert Decimal(cast("str", short["usd"])) < 0 < Decimal(cast("str", long["usd"]))

    assert Decimal(cast("str", _mapping(block["fx"])["pct"])) == Decimal("0.00")

    round_trip = _mapping(block["round_trip_one_night"])
    assert Decimal(cast("str", _mapping(round_trip[Side.SHORT.value])["usd"])) == Decimal("0.24")
    assert Decimal(cast("str", _mapping(round_trip[Side.LONG.value])["usd"])) == Decimal("2.24")

    published = _mapping(block["slippage"])
    assert published["state"] == slippage.state.value
    assert published["is_measurement"] is False
    assert published["r_pct"] is None
    assert all(entry["ok"] is True for entry in _list_of_mappings(block["checks"]))


# ─────────────────────────────────────────────────────────────────────────────
# A10 · Identidades de conservacion
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_conservation_identities(real_report: phase1_report.Phase1Report) -> None:
    """A10: `traded + no_trade + skipped == n_test` en los seis y la identidad del universo."""
    block = _block(real_report, "harness_audit", "conservation")
    rows = _list_of_mappings(block["rows"])
    assert len(rows) == 6
    n_test = cast("int", block["n_test"])
    not_in_any_test = cast("int", block["not_in_any_test"])
    for row in rows:
        traded = cast("int", row["traded"])
        no_trade = cast("int", row["no_trade"])
        skipped = cast("int", row["skipped"])
        assert traded + no_trade + skipped == n_test
        assert row["total"] == n_test
        assert row["equals_n_test"] is True
        assert row["not_in_any_test"] == not_in_any_test
    assert block["n_sessions"] == n_test + not_in_any_test
    assert block["identity"] == (
        f"{n_test + not_in_any_test} = {n_test} (test) + {not_in_any_test} (fuera de todo test)"
    )
    assert len({row["not_in_any_test"] for row in rows}) == 1
    assert block["evidence"] == EVIDENCE_RERUN


def test_a10_broken_identity_raises(tmp_path: Path) -> None:
    """A10/A23: una sesion que desaparece de un recuento es un error tipado, no un matiz."""
    reports = _synthetic_with_artifact(tmp_path / "store")
    report = phase1_report.analyse(
        store=Store(tmp_path / "store"), reports_dir=reports, as_of=NOW, write=False
    )
    assert _block(report, "harness_audit", "conservation")["state"] == str(HalfResult.PASS)
    rows = _list_of_mappings(_mapping(report.input_artifact.payload["baselines"])["rows"])
    rows[0]["traded"] = cast("int", rows[0]["traded"]) + 1
    with pytest.raises(ConservationError):
        phase1_report.conservation_audit(rerun=report.rerun, artifact=report.input_artifact)


# ─────────────────────────────────────────────────────────────────────────────
# A11 · No-*look-ahead* observable
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_lookahead_two_stores_in_tmp(
    tmp_path: Path, synthetic_report: phase1_report.Phase1Report
) -> None:
    """A11: mutar una sesion posterior no cambia nada anterior (dos almacenes, motor real)."""
    sessions = list(SYNTHETIC_SESSIONS)
    records = _daily_records(sessions)
    position = len(sessions) // 2
    cut = sessions[position]
    hostile = [dict(item) for item in records]
    hostile[position] = {**hostile[position], "close": 999.0, "high": 1_000.0, "low": 998.0}
    intact_root = tmp_path / "intact"
    hostile_root = tmp_path / "hostile"
    _write_store(intact_root, sessions=sessions, daily=records)
    _write_store(hostile_root, sessions=sessions, daily=hostile)

    reference = backtest_report.analyse(
        store=Store(intact_root), reports_dir=intact_root / "reports", as_of=NOW, write=False
    )
    changed = backtest_report.analyse(
        store=Store(hostile_root), reports_dir=hostile_root / "reports", as_of=NOW, write=False
    )
    assert reference.report_sha256 != changed.report_sha256  # la mutacion es observable
    for left, right in zip(reference.outcomes, changed.outcomes, strict=True):
        assert left.baseline == right.baseline
        base_sessions = {item.session: item for fold in left.run.folds for item in fold.sessions}
        hostile_sessions = {
            item.session: item for fold in right.run.folds for item in fold.sessions
        }
        assert base_sessions.keys() == hostile_sessions.keys()
        for session, item in base_sessions.items():
            if session < cut:
                assert item == hostile_sessions[session], (left.baseline, session)
        assert any(
            item != hostile_sessions[session]
            for session, item in base_sessions.items()
            if session >= cut
        )

    block = _block(synthetic_report, "harness_audit", "lookahead")
    assert block["state"] == str(HalfResult.PASS)
    assert block["evidence"] == EVIDENCE_RERUN
    assert block["violations"] == []
    assert cast("int", block["changed_count"]) > 0  # la comprobacion no es vacua
    assert cast("int", block["compared_outcomes"]) > 0
    mutation = _mapping(block["mutation"])
    assert mutation["open_untouched"] is True
    plan = synthetic_report.rerun.split_plan
    positions = sorted({item for fold in plan.folds for item in fold.test})
    middle = positions[len(positions) // 2]
    assert mutation["position"] == middle
    assert mutation["session"] == synthetic_report.rerun.universe.inputs[middle].session.isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# A12 · Mitad (a): determinismo
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_half_a_determinism(real_report: phase1_report.Phase1Report) -> None:
    """A12: con las huellas reproducidas la mitad (a) es `pass`; forzando un hash, `fail`."""
    determinism = _mapping(_block(real_report, "gate_halves")[HALF_DETERMINISM])
    assert determinism["state"] == str(HalfResult.PASS)
    assert determinism["evidence"] == EVIDENCE_RERUN
    assert determinism["block"] == "harness_audit.determinism"

    audit = _block(real_report, "harness_audit", "determinism")
    expected = cast("dict[str, str]", audit["expected"])
    forced = phase1_report.audit_determinism(
        expected=expected,
        observed={**expected, "report_sha256": "0" * 64},
        command="cfdtrader.analysis.backtest_report --as-of X",
        as_of="X",
    )
    assert forced["state"] == str(HalfResult.FAIL)
    assert forced["reproduced_byte_for_byte"] is False
    # El agregado **no** puede quedar mejor que el `fail` de la mitad (a).
    assert (
        phase1_report.mechanical_gate(half_a=str(HalfResult.FAIL), half_b=str(HalfResult.PASS))
        is GateVerdict.FAIL
    )


def test_a12_forced_nondeterminism_makes_the_gate_fail(tmp_path: Path) -> None:
    """A12/A24: con el hash del artefacto manipulado, la mitad (a) cae y el agregado es `fail`."""
    reports = _synthetic_with_artifact(tmp_path / "store")
    artifact_path = reports / REAL_ARTIFACT.name
    payload = _artifact_payload(artifact_path)
    payload["report_sha256"] = "0" * 64
    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report = phase1_report.analyse(
        store=Store(tmp_path / "store"), reports_dir=reports, as_of=NOW, write=False
    )
    halves = _block(report, "gate_halves")
    assert _mapping(halves[HALF_DETERMINISM])["state"] == str(HalfResult.FAIL)
    assert _block(report, "gate")["aggregate"] == str(GateVerdict.FAIL)
    assert report.payload["phase1_ready"] is False
    assert _block(report, "recommendation")["value"] == "stop"


# ─────────────────────────────────────────────────────────────────────────────
# A13 · Mitad (b): `always_long` no evaluable
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_half_b_always_long_not_evaluable(real_report: phase1_report.Phase1Report) -> None:
    """A13: `calculate_metrics` rechaza la corrida real y la mitad (b) es `not_evaluable`."""
    half = _mapping(_block(real_report, "gate_halves")[HALF_ALWAYS_LONG])
    assert half["state"] == str(HalfResult.NOT_EVALUABLE)
    assert half["comparison_performed"] is False
    rejection = _block(real_report, "harness_audit", "net_metrics")
    assert rejection["state"] == "rejected"
    assert rejection["error"] == "MetricsInputError"
    assert rejection["mentions_pnl_net_pct"] is True
    assert rejection["where"] == "cfdtrader.backtest.metrics.calculate_metrics"

    outcome = next(item for item in real_report.rerun.outcomes if item.baseline == HALF_ALWAYS_LONG)
    with pytest.raises(MetricsInputError) as raised:
        calculate_metrics(outcome.run)
    assert "pnl_net_pct" in str(raised.value)
    assert _block(real_report, "net_metrics")["state"] == "not_computable"


# ─────────────────────────────────────────────────────────────────────────────
# A14 · Agregacion mecanica de la puerta
# ─────────────────────────────────────────────────────────────────────────────
def test_a14_gate_aggregation_is_mechanical() -> None:
    """A14: la regla se importa de #9 y `not_evaluable` nunca se convierte en `pass`."""
    assert phase1_report.aggregate_gate is phase0_report.aggregate_gate
    assert "phase0_report.aggregate_gate" in phase1_report.GATE_AGGREGATION_SOURCE
    ok = str(HalfResult.PASS)
    bad = str(HalfResult.FAIL)
    unknown = str(HalfResult.NOT_EVALUABLE)
    undefined = GateVerdict.NOT_EVALUABLE
    assert phase1_report.mechanical_gate(half_a=ok, half_b=ok) is GateVerdict.PASS
    assert phase1_report.mechanical_gate(half_a=ok, half_b=unknown) is undefined
    assert phase1_report.mechanical_gate(half_a=unknown, half_b=unknown) is undefined
    assert phase1_report.mechanical_gate(half_a=unknown, half_b=bad) is GateVerdict.FAIL
    assert phase1_report.mechanical_gate(half_a=bad, half_b=ok) is GateVerdict.FAIL


def test_a14_gate_with_todays_halves_is_not_evaluable(
    real_report: phase1_report.Phase1Report,
) -> None:
    """A14: con (`pass`, `not_evaluable`) el agregado es `not_evaluable`, no `pass`."""
    gate = _block(real_report, "gate")
    assert gate["aggregate"] == str(GateVerdict.NOT_EVALUABLE)
    assert gate["aggregation_rule"] == phase0_report.GATE_AGGREGATION_RULE
    assert gate["halves"] == [HALF_DETERMINISM, HALF_ALWAYS_LONG]


# ─────────────────────────────────────────────────────────────────────────────
# A15 · Veredicto parcial declarado
# ─────────────────────────────────────────────────────────────────────────────
def test_a15_partial_verdict(real_report: phase1_report.Phase1Report) -> None:
    """A15: veredicto parcial, `phase1_ready: false` y prohibido publicar `pass`."""
    gate = _block(real_report, "gate")
    assert real_report.payload["phase1_ready"] is False
    assert gate["phase1_ready"] is False
    assert gate["aggregate"] != str(GateVerdict.PASS)
    verdict = cast("str", gate["partial_verdict"])
    assert "determinismo" in verdict
    assert str(HalfResult.NOT_EVALUABLE) in verdict
    assert "medida hoy" in verdict
    assert "no evaluable" in verdict
    assert gate["is_validation"] is False


# ─────────────────────────────────────────────────────────────────────────────
# A16 · Como se cierra la mitad no evaluable
# ─────────────────────────────────────────────────────────────────────────────
def test_a16_how_to_close(real_report: phase1_report.Phase1Report) -> None:
    """A16: #62, #70 y #60 estan presentes, cada uno con su motivo."""
    entries = _list_of_mappings(real_report.payload["how_to_close"])
    issues = {cast("str", entry["issue"]) for entry in entries}
    assert {"#62", "#70", "#60"} <= issues
    for entry in entries:
        assert cast("str", entry["reason"]).strip()
        assert cast("str", entry["issue"]).startswith("#")


# ─────────────────────────────────────────────────────────────────────────────
# A17 · Limites heredados sin fusionar
# ─────────────────────────────────────────────────────────────────────────────
def test_a17_limits_inherited_without_fusing(
    real_report: phase1_report.Phase1Report, written: tuple[Path, phase1_report.Phase1Report]
) -> None:
    """A17: el bloque de limites se hereda entero y ningun estado se fusiona."""
    reports, _ = written
    inherited = _mapping(_artifact_payload(reports / REAL_ARTIFACT.name)["limits"])
    limits = _block(real_report, "limits")
    for key, value in inherited.items():
        assert limits[key] == value, key
    assert limits["gate"] == "fail"
    assert limits["phase1_ready"] is False
    assert limits["is_validation"] is False
    assert limits["financing_cut"] is None
    assert limits["financing_cut_verified"] is False
    assert limits["financing_cut_issue"] == "#59"
    slippage = _mapping(limits["slippage"])
    assert slippage["state"] == "assumed"
    assert slippage["is_measurement"] is False
    assert slippage["pct_of_r"] == "0.2"
    assert slippage["r_pct"] is None
    assert slippage["issue"] == "#62"
    prices = _mapping(limits["prices"])
    assert prices["series_id"] == "^GSPC"
    assert prices["proxy_of"] == "SPX500:CFD"
    assert prices["is_proxy"] is True
    assert prices["issue"] == "#50"
    assert limits["net_metrics_state"] == "not_computable"
    assert "no son una validacion" in cast("str", limits["statement"])
    assert limits["evidence"] == EVIDENCE_ARTIFACT


# ─────────────────────────────────────────────────────────────────────────────
# A18 · Prohibido maquillar
# ─────────────────────────────────────────────────────────────────────────────
def test_a18_no_cosmetics(real_report: phase1_report.Phase1Report) -> None:
    """A18: sin `measured` en el fuente, sin validaciones y sin `null` convertidos en 0."""
    assert "measured" not in SOURCE
    flagged = [
        key for key, value in _all_pairs(real_report.payload) if key == "is_validation" and value
    ]
    assert flagged == []
    assert _block(real_report, "limits")["is_validation"] is False
    assert _block(real_report, "baselines")["is_validation"] is False
    assert _block(real_report, "gate")["is_validation"] is False
    assert _block(real_report, "net_metrics")["state"] == "not_computable"
    assert _block(real_report, "baselines")["basis"] == "declared_cost"
    # El supuesto sigue siendo supuesto y su `r_pct` sigue siendo `null`, nunca 0.
    assert _mapping(_block(real_report, "harness_audit", "costs")["slippage"])["r_pct"] is None
    assert _mapping(_block(real_report, "limits")["slippage"])["r_pct"] is None


# ─────────────────────────────────────────────────────────────────────────────
# A19 · Determinismo del propio informe entre procesos
# ─────────────────────────────────────────────────────────────────────────────
def test_a19_own_report_determinism_across_processes(tmp_path: Path, synthetic_root: Path) -> None:
    """A19: cuatro procesos (tres `PYTHONHASHSEED` + segunda pasada) dan los mismos bytes."""
    reports = tmp_path / "derived" / "reports"
    _artifact_pair(reports, synthetic_root / "derived" / "reports" / REAL_ARTIFACT.name)
    payloads: list[bytes] = []
    markdowns: list[bytes] = []
    hashes: set[str] = set()
    for index, seed in enumerate(("0", "1", "random", "13")):
        result = _run_module(
            "cfdtrader.analysis.phase1_report",
            [
                "--data-root",
                str(synthetic_root),
                "--reports-dir",
                str(reports),
                "--as-of",
                NOW.isoformat(),
            ],
            seed=seed,
        )
        assert result.returncode == 0, result.stderr
        payloads.append((reports / "phase1_report_2026-09-19.json").read_bytes())
        markdowns.append((reports / "phase1_report_2026-09-19.md").read_bytes())
        hashes.add(cast("str", json.loads(payloads[index].decode("utf-8"))["report_sha256"]))
    assert len(payloads) == 4
    assert len(set(payloads)) == 1
    assert len(set(markdowns)) == 1
    assert len(hashes) == 1


# ─────────────────────────────────────────────────────────────────────────────
# A20 · Hash canonico reutilizando #13
# ─────────────────────────────────────────────────────────────────────────────
def test_a20_canonical_hash_reused(real_report: phase1_report.Phase1Report) -> None:
    """A20: `report_sha256` es el sha256 del canonico de #13, sin la clave del hash."""
    assert phase1_report.canonical_text is canonical_text
    assert "canonical_text" in REPORT_HASH_FORMAT
    assert "report_sha256" in REPORT_HASH_FORMAT
    expected = hashlib.sha256(canonical_text(real_report.payload).encode("utf-8")).hexdigest()
    assert real_report.report_sha256 == expected
    assert "report_sha256" not in real_report.payload
    assert json.loads(real_report.json_text())["report_sha256"] == expected
    assert real_report.payload["hash_format"] == REPORT_HASH_FORMAT
    # Ningun `float` del payload es `nan` ni `inf` (los `Decimal` viajan como cadena).
    floats = [value for _, value in _numeric_items(real_report.payload) if isinstance(value, float)]
    assert floats
    assert all(math.isfinite(value) for value in floats)


# ─────────────────────────────────────────────────────────────────────────────
# A21 · Solo lectura de `raw` y `labels`; `write=False` no escribe
# ─────────────────────────────────────────────────────────────────────────────
def test_a21_read_only_inputs_and_write_false(tmp_path: Path, synthetic_root: Path) -> None:
    """A21: `write=False` no deja nada y `write=True` solo anade el informe."""
    reports = tmp_path / "derived" / "reports"
    _artifact_pair(reports, synthetic_root / "derived" / "reports" / REAL_ARTIFACT.name)
    before = _fingerprint(reports)
    phase1_report.analyse(store=Store(synthetic_root), reports_dir=reports, as_of=NOW, write=False)
    assert _fingerprint(reports) == before

    raw_before = _fingerprint(REAL_DATA / "raw")
    labels_before = _fingerprint(REAL_DATA / "derived" / "labels")
    phase1_report.analyse(store=Store(synthetic_root), reports_dir=reports, as_of=NOW, write=True)
    assert _fingerprint(REAL_DATA / "raw") == raw_before
    assert _fingerprint(REAL_DATA / "derived" / "labels") == labels_before
    assert sorted(set(_fingerprint(reports)) - set(before)) == [
        "phase1_report_2026-09-19.json",
        "phase1_report_2026-09-19.md",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# A22 · Sin red, sin scheduler y sin LLM
# ─────────────────────────────────────────────────────────────────────────────
def test_a22_no_network_no_scheduler_no_llm(real_report: phase1_report.Phase1Report) -> None:
    """A22: el modulo no importa ningun cliente de red, ni agentes, ni planificadores."""
    forbidden = {
        "yfinance",
        "fredapi",
        "requests",
        "httpx",
        "schedule",
        "apscheduler",
        "croniter",
        "openai",
        "langgraph",
        "cfdtrader.agents",
        "cfdtrader.orchestration",
        "cfdtrader.delivery",
    }
    imported: set[str] = set()
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
            imported |= {f"{node.module}.{alias.name}" for alias in node.names}
    assert not (imported & forbidden), sorted(imported & forbidden)
    limits = _block(real_report, "limits")
    assert limits["llm_overlay"] == "disabled"
    assert limits["scheduler"] == "none"


# ─────────────────────────────────────────────────────────────────────────────
# A23 · Errores tipados, nunca un resultado silencioso
# ─────────────────────────────────────────────────────────────────────────────
def test_a23_typed_errors(tmp_path: Path) -> None:
    """A23: errores propios tipados y errores de #12/#13/#14/#15/#69 sin envolver."""
    assert issubclass(MissingInputArtifactError, Phase1ReportError)
    assert issubclass(AmbiguousInputArtifactError, Phase1ReportError)
    assert issubclass(ConservationError, Phase1ReportError)

    reports = _synthetic_with_artifact(tmp_path / "store")

    # Artefacto ausente.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(MissingInputArtifactError):
        phase1_report.load_input_artifact(empty, store=Store(tmp_path))

    # JSON alterado a mano: no es un objeto JSON.
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / REAL_ARTIFACT.name).write_text("[1, 2]", encoding="utf-8")
    (broken / "phase1_backtest_2026-09-19.md").write_text("x", encoding="utf-8")
    with pytest.raises(phase1_report.MalformedInputArtifactError):
        phase1_report.load_input_artifact(broken, store=Store(tmp_path))

    # Falta un campo obligatorio.
    no_hash = tmp_path / "no_hash"
    _artifact_pair(no_hash, reports / REAL_ARTIFACT.name)
    payload = _artifact_payload(no_hash / REAL_ARTIFACT.name)
    del payload["report_sha256"]
    (no_hash / REAL_ARTIFACT.name).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(phase1_report.MissingInputFieldError):
        phase1_report.analyse(
            store=Store(tmp_path / "store"), reports_dir=no_hash, as_of=NOW, write=False
        )

    # Falta `--as-of`: el error tipado se lanza y la CLI lo traduce a codigo 2.
    assert phase1_report.main([]) == 2
    with pytest.raises(phase1_report.MissingAsOfError):
        phase1_report._parse_as_of(None)  # pyright: ignore[reportPrivateUsage]

    # Error de #69 sin envolver: sin `derived.labels` no hay universo que simular.
    bare = tmp_path / "bare"
    Store(bare)
    with pytest.raises(backtest_report.MissingDatasetError):
        phase1_report.analyse(store=Store(bare), reports_dir=reports, as_of=NOW, write=False)


def test_a23_engine_errors_propagate_unwrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A23: el error tipado de #15 llega al llamante con su tipo original."""
    reports = _synthetic_with_artifact(tmp_path / "store")

    def boom(**_kwargs: object) -> object:
        raise MetricsInputError("rechazo declarado de #15")

    monkeypatch.setattr(phase1_report.backtest_report, "analyse", boom)
    with pytest.raises(MetricsInputError):
        phase1_report.analyse(
            store=Store(tmp_path / "store"), reports_dir=reports, as_of=NOW, write=False
        )


# ─────────────────────────────────────────────────────────────────────────────
# A24 · Cobertura de los bordes incomodos
# ─────────────────────────────────────────────────────────────────────────────
def test_a24_edge_cases_are_declared_and_exercised(tmp_path: Path, synthetic_root: Path) -> None:
    """A24: un test por criterio y los bordes incomodos, incluida la segunda pasada."""
    names = {name for name in globals() if name.startswith("test_a")}
    for prefix in REQUIRED_TEST_PREFIXES:
        assert any(name.startswith(prefix) for name in names), prefix

    reports = tmp_path / "derived" / "reports"
    _artifact_pair(reports, synthetic_root / "derived" / "reports" / REAL_ARTIFACT.name)
    first = phase1_report.analyse(
        store=Store(synthetic_root), reports_dir=reports, as_of=NOW, write=True
    )
    before = _fingerprint(reports)
    second = phase1_report.analyse(
        store=Store(synthetic_root), reports_dir=reports, as_of=NOW, write=True
    )
    assert second.report_sha256 == first.report_sha256
    assert second.json_text() == first.json_text()
    assert phase1_report.render_markdown(second) == phase1_report.render_markdown(first)
    assert _fingerprint(reports) == before


# ─────────────────────────────────────────────────────────────────────────────
# A25 · Puertas en verde (lo automatizable desde la suite)
# ─────────────────────────────────────────────────────────────────────────────
def test_a25_only_expected_pragma() -> None:
    """A25: el unico `# pragma: no cover` del modulo es el guard de `__main__`."""
    marked = [
        line.strip()
        for line in SOURCE.splitlines()
        if "# pragma: no cover" in line and not line.strip().startswith("if __name__")
    ]
    assert marked == []


# ─────────────────────────────────────────────────────────────────────────────
# A26 · Fronteras legibles por maquina
# ─────────────────────────────────────────────────────────────────────────────
def test_a26_machine_readable_boundaries() -> None:
    """A26: `REPORT_DOES_NOT_DO` y `FOLLOW_UPS` cubren las fronteras y traen su `issue`."""
    for entry in REPORT_DOES_NOT_DO:
        assert set(entry) == {"id", "issue", "statement"}
        assert entry["issue"].startswith("#")
    for entry in FOLLOW_UPS:
        assert set(entry) == {"issue", "topic", "why"}
        assert entry["issue"].startswith("#")
    covered = {entry["issue"] for entry in (*REPORT_DOES_NOT_DO, *FOLLOW_UPS)}
    for issue in (
        "#17",
        "#16",
        "#62",
        "#60",
        "#70",
        "#50",
        "#51",
        "#28",
        "#29",
        "#65",
        "#67",
        "#68",
        "#69",
    ):
        assert issue in covered, issue


# ─────────────────────────────────────────────────────────────────────────────
# A27 · Reutilizar, no duplicar
# ─────────────────────────────────────────────────────────────────────────────
def test_a27_reuses_rules_without_literals() -> None:
    """A27: el modulo importa las reglas de #9/#8/#11/#13/#15/#52 y no copia sus literales."""
    assert phase1_report.select_artifact is phase0_report.select_artifact
    assert phase1_report.aggregate_gate is phase0_report.aggregate_gate
    assert phase1_report.canonical_text is canonical_text
    assert phase1_report.declared_cost_model is declared_cost_model
    assert phase1_report.declared_slippage_assumption is declared_slippage_assumption
    assert phase1_report.calculate_metrics is calculate_metrics
    assert phase1_report.clean_sample_cutoff is drift.clean_sample_cutoff
    assert phase1_report.backtest_report.build_inputs is backtest_report.build_inputs
    for literal in DECLARED_TABLE_LITERALS:
        assert literal not in SOURCE, literal
    for token in ("STALE_OPEN_TOLERANCE", "MIN_CLEAN_SESSIONS"):
        assert token not in SOURCE


def test_a27_clean_rule_is_cross_checked(synthetic_report: phase1_report.Phase1Report) -> None:
    """A27: el corte publicado es el de la regla compartida de #52, importada."""
    clean = _mapping(_block(synthetic_report, "harness_audit", "lookahead")["clean_sample"])
    assert clean["matches_shared_rule"] is True
    assert "drift" in cast("str", clean["rule_source"])
    assert synthetic_report.rerun.universe.clean_from is not None
    assert clean["clean_from"] == synthetic_report.rerun.universe.clean_from.isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# A28 · Informe en prosa
# ─────────────────────────────────────────────────────────────────────────────
def test_a28_markdown_report(real_report: phase1_report.Phase1Report) -> None:
    """A28: el `.md` resume la puerta, los seis baselines y los limites, y cita el artefacto."""
    markdown = phase1_report.render_markdown(real_report)
    assert markdown.startswith("# Informe de Fase 1 y puerta de salida")
    assert str(GateVerdict.NOT_EVALUABLE) in markdown
    assert HALF_DETERMINISM in markdown and HALF_ALWAYS_LONG in markdown
    for row in _rows(real_report):
        assert cast("str", row["baseline"]) in markdown
        assert cast("str", row["run_sha256"]) in markdown
    assert "purge_total" in markdown
    assert cast("str", _block(real_report, "source_artifact")["path"]) in markdown
    assert "Tabla copiada de" in markdown
    assert "duplica" in markdown
    assert "no es una validacion" in markdown
    assert "Bloqueos" in markdown
    assert "Como se cierra la mitad no evaluable" in markdown
    assert markdown.endswith("\n")


def test_a28_json_and_markdown_come_from_one_payload(
    synthetic_report: phase1_report.Phase1Report,
) -> None:
    """A28/A19: el `.json` y el `.md` salen del **mismo** payload determinista."""
    first = phase1_report.render_markdown(synthetic_report)
    assert first == phase1_report.render_markdown(synthetic_report)
    assert json.loads(synthetic_report.json_text())["report_sha256"] == (
        synthetic_report.report_sha256
    )
    assert synthetic_report.report_sha256 in first


# ─────────────────────────────────────────────────────────────────────────────
# A29 · Recomendacion sujeta a regla
# ─────────────────────────────────────────────────────────────────────────────
def test_a29_recommendation_rule(real_report: phase1_report.Phase1Report) -> None:
    """A29: con `gate != pass` no se recomienda `continue`; forzar el agregado es un error."""
    recommendation = _block(real_report, "recommendation")
    assert recommendation["gate"] != str(GateVerdict.PASS)
    assert recommendation["value"] != "continue"
    assert recommendation["consistent"] is True
    assert recommendation["rule"] == phase0_report.RECOMMENDATION_CONSISTENCY_RULE
    with pytest.raises(VerdictError):
        phase1_report.mechanical_gate(
            half_a=str(HalfResult.PASS),
            half_b=str(HalfResult.NOT_EVALUABLE),
            claimed=str(GateVerdict.PASS),
        )
    with pytest.raises(VerdictError):
        phase1_report.mechanical_gate(
            half_a=str(HalfResult.NOT_EVALUABLE),
            half_b=str(HalfResult.NOT_EVALUABLE),
            claimed=str(GateVerdict.PASS),
        )


# ─────────────────────────────────────────────────────────────────────────────
# A30 · `blockers[]` buscables por maquina
# ─────────────────────────────────────────────────────────────────────────────
def test_a30_blockers(real_report: phase1_report.Phase1Report) -> None:
    """A30: codigos estables que reutilizan `BlockerCode` de #9 y enlazan a su issue."""
    blockers = _list_of_mappings(real_report.payload["blockers"])
    codes = {cast("str", entry["code"]) for entry in blockers}
    assert str(BlockerCode.SLIPPAGE_ASSUMED_NOT_MEASURED) in codes
    assert str(BlockerCode.R_UNDECIDED) in codes
    assert str(BlockerCode.FINANCING_CUT_UNVERIFIED) in codes
    assert str(BlockerCode.BROKER_UNDECIDED) in codes
    assert {"net_metrics_not_computable", "always_long_not_evaluable"} <= codes
    for entry in blockers:
        assert set(entry) == {"code", "issues", "reason"}
        issues = cast("list[str]", entry["issues"])
        assert issues and all(issue.startswith("#") for issue in issues)
        assert cast("str", entry["reason"]).strip()
    assert len(codes) == len(blockers)


# ─────────────────────────────────────────────────────────────────────────────
# A31 · Cero metricas netas
# ─────────────────────────────────────────────────────────────────────────────
def test_a31_zero_net_metrics(real_report: phase1_report.Phase1Report) -> None:
    """A31: ningun Sharpe, Sortino, EV, drawdown ni intervalo; el hueco se declara."""
    block = _block(real_report, "net_metrics")
    assert block["state"] == "not_computable"
    assert block["where"] == "cfdtrader.backtest.metrics.calculate_metrics"
    assert block["follow_up"] == ["#62", "#60"]
    assert block["published_metrics"] == []
    assert cast("str", block["reason"]).strip()
    keys = {key.lower() for key in _keys(real_report.payload)}
    assert not (keys & FORBIDDEN_METRIC_KEYS), sorted(keys & FORBIDDEN_METRIC_KEYS)
    for path, _ in _numeric_items(real_report.payload):
        assert path.rsplit(".", 1)[-1].lower() not in FORBIDDEN_METRIC_KEYS, path


# ─────────────────────────────────────────────────────────────────────────────
# A32 · Procedencia de la evidencia
# ─────────────────────────────────────────────────────────────────────────────
def test_a32_evidence_provenance(real_report: phase1_report.Phase1Report) -> None:
    """A32: cada bloque declara si se midio aqui o se copio del artefacto de #69."""
    assert EVIDENCE_RERUN == "re-run"
    assert EVIDENCE_ARTIFACT == "artifact"
    assert _block(real_report, "harness_audit", "determinism")["evidence"] == EVIDENCE_RERUN
    assert _block(real_report, "harness_audit", "lookahead")["evidence"] == EVIDENCE_RERUN
    assert _block(real_report, "harness_audit", "conservation")["evidence"] == EVIDENCE_RERUN
    assert _block(real_report, "harness_audit", "costs")["evidence"] == EVIDENCE_RERUN
    assert _block(real_report, "baselines")["evidence"] == EVIDENCE_ARTIFACT
    assert _block(real_report, "harness_audit", "purge_embargo")["evidence"] == EVIDENCE_ARTIFACT
    assert _block(real_report, "source_artifact")["evidence"] == EVIDENCE_ARTIFACT


# ─────────────────────────────────────────────────────────────────────────────
# A33 · Sin horas ET literales ni *sizing* derivado
# ─────────────────────────────────────────────────────────────────────────────
def test_a33_no_et_hours_and_no_sizing(real_report: phase1_report.Phase1Report) -> None:
    """A33: ninguna hora ET literal, `financing_cut` a `None` y ningun *sizing* derivado."""
    for token in ("16:00", "09:30", "13:00"):
        assert token not in SOURCE
    assert _block(real_report, "limits")["financing_cut"] is None
    assert Decimal(cast("str", _block(real_report, "baselines")["notional_usd"])) == (
        backtest_report.NOTIONAL_USD
    )
    assert "no_deriva_el_nocional" in {entry["id"] for entry in REPORT_DOES_NOT_DO}
    statements = " ".join(entry["statement"] for entry in REPORT_DOES_NOT_DO)
    assert "sizing" in statements


# ─────────────────────────────────────────────────────────────────────────────
# A34 · Un artefacto, tres verdades separadas
# ─────────────────────────────────────────────────────────────────────────────
def test_a34_three_separate_truths(real_report: phase1_report.Phase1Report) -> None:
    """A34: `source_artifact`, `harness_audit` y `gate` existen separados y sin validar."""
    payload = real_report.payload
    for key in ("source_artifact", "harness_audit", "gate"):
        assert key in payload
        assert isinstance(payload[key], dict)
    assert _block(real_report, "gate")["is_validation"] is False
    assert payload["phase1_ready"] is False
    rendered = json.dumps(payload, ensure_ascii=False).lower()
    assert "no son una validacion" in rendered
    assert '"is_validation": true' not in rendered
    assert "no se fusionan" in cast("str", cast("list[object]", payload["notes"])[0])


# ─────────────────────────────────────────────────────────────────────────────
# A23/A24/A25 · Caminos de error declarados (cobertura del modulo nuevo)
# ─────────────────────────────────────────────────────────────────────────────
def _tampered_artifact(
    tmp_path: Path, name: str, mutate: Callable[[dict[str, object]], None], *, source: Path
) -> tuple[Path, InputArtifact]:
    """Copia el artefacto, lo altera a mano y lo carga: (directorio de informes, artefacto)."""
    reports = tmp_path / name / "derived" / "reports"
    _artifact_pair(reports, source)
    path = reports / source.name
    payload = _artifact_payload(path)
    mutate(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return reports, phase1_report.load_input_artifact(reports, store=Store(tmp_path))


def _synthetic_artifact(synthetic_root: Path) -> Path:
    """El artefacto de #69 del almacen sintetico."""
    return synthetic_root / "derived" / "reports" / REAL_ARTIFACT.name


def _rows_of(payload: dict[str, object]) -> list[object]:
    """Las filas del artefacto, sin tipar (para alterarlas en las pruebas)."""
    return cast("list[object]", _mapping(payload["baselines"])["rows"])


def _plan_of(payload: dict[str, object]) -> dict[str, object]:
    """El bloque `plan` del artefacto, sin tipar."""
    return _mapping(payload["plan"])


def _drop_generated_at(payload: dict[str, object]) -> None:
    """Quita el instante declarado del artefacto (alteracion a mano)."""
    payload.pop("generated_at")


def _drop_last_baseline_row(payload: dict[str, object]) -> None:
    """Quita la ultima fila de la tabla de baselines (alteracion a mano)."""
    _rows_of(payload).pop()


def _shift_the_row_window(payload: dict[str, object]) -> None:
    """Deja la fila autoconsistente pero con un `n_test` que no es el de la corrida."""
    _mapping(_rows_of(payload)[0]).update(n_test=1, traded=1, no_trade=0, skipped=0)


def _declare_the_exclusions_as_active(payload: dict[str, object]) -> None:
    """Marca las exclusiones de #12 como activas (lo que hoy no son)."""
    _plan_of(payload).update(exclusions_are_no_op=False)


def test_a23_malformed_json_is_a_typed_error(tmp_path: Path) -> None:
    """A23: un JSON ilegible o no UTF-8 tambien es un error tipado, no un `None`."""
    reports = tmp_path / "derived" / "reports"
    reports.mkdir(parents=True)
    (reports / REAL_ARTIFACT.name).write_text("{no es json", encoding="utf-8")
    (reports / "phase1_backtest_2026-09-19.md").write_text("x", encoding="utf-8")
    with pytest.raises(phase1_report.MalformedInputArtifactError):
        phase1_report.load_input_artifact(reports, store=Store(tmp_path))

    (reports / REAL_ARTIFACT.name).write_bytes(b"\xff\xfe\x00{}")
    with pytest.raises(phase1_report.MalformedInputArtifactError):
        phase1_report.load_input_artifact(reports, store=Store(tmp_path))


def test_a23_missing_generated_at_is_a_typed_error(tmp_path: Path, synthetic_root: Path) -> None:
    """A23: sin el instante declarado del artefacto no se re-ejecuta: error tipado."""
    reports, _ = _tampered_artifact(
        tmp_path,
        "no_generated",
        _drop_generated_at,
        source=_synthetic_artifact(synthetic_root),
    )
    with pytest.raises(phase1_report.MissingInputFieldError):
        phase1_report.analyse(
            store=Store(synthetic_root), reports_dir=reports, as_of=NOW, write=False
        )


def test_a23_generated_at_must_be_iso(tmp_path: Path, synthetic_root: Path) -> None:
    """A23: un `generated_at` que no es ISO-8601 es un error tipado."""
    reports, _ = _tampered_artifact(
        tmp_path,
        "bad_generated",
        lambda payload: payload.update(generated_at="no-es-iso"),
        source=_synthetic_artifact(synthetic_root),
    )
    with pytest.raises(phase1_report.MalformedInputArtifactError):
        phase1_report.analyse(
            store=Store(synthetic_root), reports_dir=reports, as_of=NOW, write=False
        )


def test_a23_wrong_field_types_are_typed_errors(
    tmp_path: Path, synthetic_root: Path, synthetic_report: Phase1Report
) -> None:
    """A23: un campo obligatorio con el tipo equivocado tambien es un error tipado."""
    _, artifact = _tampered_artifact(
        tmp_path,
        "bad_str",
        lambda payload: payload.update(generated_at=123),
        source=_synthetic_artifact(synthetic_root),
    )
    with pytest.raises(phase1_report.MissingInputFieldError):
        _ = artifact.generated_at

    _, int_artifact = _tampered_artifact(
        tmp_path,
        "bad_int",
        lambda payload: _plan_of(payload).update(purge_total="cero"),
        source=_synthetic_artifact(synthetic_root),
    )
    with pytest.raises(phase1_report.MissingInputFieldError):
        phase1_report.purge_embargo_block(artifact=int_artifact, rerun=synthetic_report.rerun)

    _, bool_artifact = _tampered_artifact(
        tmp_path,
        "bad_bool",
        lambda payload: _plan_of(payload).update(exclusions_are_no_op="si"),
        source=_synthetic_artifact(synthetic_root),
    )
    with pytest.raises(phase1_report.MissingInputFieldError):
        phase1_report.purge_embargo_block(artifact=bool_artifact, rerun=synthetic_report.rerun)

    _, nested_artifact = _tampered_artifact(
        tmp_path,
        "bad_nested",
        lambda payload: payload.update(plan=5),
        source=_synthetic_artifact(synthetic_root),
    )
    with pytest.raises(phase1_report.MissingInputFieldError):
        phase1_report.purge_embargo_block(artifact=nested_artifact, rerun=synthetic_report.rerun)


def test_a23_baselines_rows_must_be_a_list(
    tmp_path: Path, synthetic_root: Path, synthetic_report: Phase1Report
) -> None:
    """A23: `baselines.rows` tiene que ser una lista de objetos."""
    _, artifact = _tampered_artifact(
        tmp_path,
        "bad_rows",
        lambda payload: _mapping(payload["baselines"]).update(rows={}),
        source=_synthetic_artifact(synthetic_root),
    )
    with pytest.raises(phase1_report.MissingInputFieldError):
        phase1_report.conservation_audit(rerun=synthetic_report.rerun, artifact=artifact)


def test_a23_missing_limits_gate_is_a_typed_error(tmp_path: Path, synthetic_root: Path) -> None:
    """A23/A17: un `limits.gate` fuera del vocabulario declarado es un error tipado."""
    reports, _ = _tampered_artifact(
        tmp_path,
        "bad_gate",
        lambda payload: _mapping(payload["limits"]).update(gate="quizas"),
        source=_synthetic_artifact(synthetic_root),
    )
    with pytest.raises(phase1_report.MissingInputFieldError):
        phase1_report.analyse(
            store=Store(synthetic_root), reports_dir=reports, as_of=NOW, write=False
        )


def test_a10_artifact_disagreements_raise(
    tmp_path: Path, synthetic_root: Path, synthetic_report: Phase1Report
) -> None:
    """A10: la tabla de #69 y la corrida no pueden discrepar en un recuento."""
    cases: list[tuple[str, Callable[[dict[str, object]], None]]] = [
        ("bad_n_test", _shift_the_row_window),
        ("bad_row_sum", lambda payload: _mapping(_rows_of(payload)[1]).update(traded=0)),
        ("missing_row", _drop_last_baseline_row),
        ("bad_universe", lambda payload: _mapping(payload["universe"]).update(sessions=1)),
    ]
    for name, mutate in cases:
        _, artifact = _tampered_artifact(
            tmp_path, name, mutate, source=_synthetic_artifact(synthetic_root)
        )
        with pytest.raises(ConservationError):
            phase1_report.conservation_audit(rerun=synthetic_report.rerun, artifact=artifact)


def test_a10_broken_plan_raises(tmp_path: Path, synthetic_root: Path) -> None:
    """A10: un plan que no cuadre con el universo o con los recuentos es un error tipado."""
    reports = _synthetic_with_artifact(tmp_path / "store")
    reference = phase1_report.analyse(
        store=Store(tmp_path / "store"), reports_dir=reports, as_of=NOW, write=False
    )
    artifact = reference.input_artifact
    plan = reference.rerun.split_plan

    with pytest.raises(ConservationError):
        phase1_report.conservation_audit(
            rerun=replace(reference.rerun, split_plan=replace(plan, uncovered=())),
            artifact=artifact,
        )

    no_tests = replace(plan, folds=(), uncovered=tuple(range(plan.n_sessions)))
    with pytest.raises(ConservationError):
        phase1_report.conservation_audit(
            rerun=replace(reference.rerun, split_plan=no_tests), artifact=artifact
        )

    hostile = replace(reference.rerun.outcomes[0].run, not_in_any_test=1)
    first = replace(reference.rerun.outcomes[0], run=hostile)
    with pytest.raises(ConservationError):
        phase1_report.conservation_audit(
            rerun=replace(reference.rerun, outcomes=(first, *reference.rerun.outcomes[1:])),
            artifact=artifact,
        )


def test_a8_purge_embargo_disagreement_raises(
    tmp_path: Path, synthetic_root: Path, synthetic_report: Phase1Report
) -> None:
    """A8/A10: si el artefacto y la re-ejecucion discrepan en purga/embargo, se falla."""
    _, artifact = _tampered_artifact(
        tmp_path,
        "bad_purge",
        lambda payload: _plan_of(payload).update(purge_total=99),
        source=_synthetic_artifact(synthetic_root),
    )
    with pytest.raises(ConservationError):
        phase1_report.purge_embargo_block(artifact=artifact, rerun=synthetic_report.rerun)

    _, no_op_flag = _tampered_artifact(
        tmp_path,
        "noop_flag",
        _declare_the_exclusions_as_active,
        source=_synthetic_artifact(synthetic_root),
    )
    with pytest.raises(ConservationError):
        phase1_report.purge_embargo_block(artifact=no_op_flag, rerun=synthetic_report.rerun)


def test_a8_non_no_op_plan_is_published_as_active(
    tmp_path: Path, synthetic_root: Path, synthetic_report: Phase1Report
) -> None:
    """A8: si las exclusiones **si** quitaran muestra, el informe no las llamaria no-op."""
    _, artifact = _tampered_artifact(
        tmp_path,
        "not_no_op",
        lambda payload: _plan_of(payload).update(exclusions_are_no_op=False),
        source=_synthetic_artifact(synthetic_root),
    )
    plan = replace(synthetic_report.rerun.split_plan, exclusions_are_no_op=False)
    block = phase1_report.purge_embargo_block(
        artifact=artifact, rerun=replace(synthetic_report.rerun, split_plan=plan)
    )
    assert block["structural_no_op"] is False
    assert "no-ops estructurales" not in cast("str", block["note"])


def test_a25_cost_audit_fails_when_the_table_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A25/A9: si la tabla declarada cambiara de signo, la auditoria de coste falla."""
    model = declared_cost_model()
    hostile = model.model_copy(update={"carry_short_pct_per_night": Decimal("0.0018")})
    monkeypatch.setattr(phase1_report, "declared_cost_model", lambda: hostile)
    with pytest.raises(phase1_report.AuditInvariantError):
        phase1_report.costs_audit()


def test_a25_net_metrics_rejection_paths(
    monkeypatch: pytest.MonkeyPatch, synthetic_report: Phase1Report
) -> None:
    """A25/A13: el rechazo de #15 se declara y cualquier otro desenlace es un error."""
    without = tuple(
        item for item in synthetic_report.rerun.outcomes if item.baseline != HALF_ALWAYS_LONG
    )
    with pytest.raises(phase1_report.AuditInvariantError):
        phase1_report.net_metrics_rejection(rerun=replace(synthetic_report.rerun, outcomes=without))

    def unexpected(**_kwargs: object) -> object:
        raise TypeError("otro error, no el declarado")

    monkeypatch.setattr(phase1_report, "calculate_metrics", unexpected)
    with pytest.raises(phase1_report.AuditInvariantError):
        phase1_report.net_metrics_rejection(rerun=synthetic_report.rerun)

    def no_error(*_args: object, **_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(phase1_report, "calculate_metrics", no_error)
    with pytest.raises(phase1_report.AuditInvariantError):
        phase1_report.net_metrics_rejection(rerun=synthetic_report.rerun)


def test_a25_verdict_consistency_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """A25/A29: la recomendacion inconsistente no se publica."""

    def always_inconsistent(*_args: object, **_kwargs: object) -> bool:
        return False

    monkeypatch.setattr(phase1_report, "recommendation_is_consistent", always_inconsistent)
    with pytest.raises(VerdictError):
        phase1_report._verdict(GateVerdict.NOT_EVALUABLE)  # pyright: ignore[reportPrivateUsage]


def test_a11_lookahead_detects_a_violation(
    monkeypatch: pytest.MonkeyPatch, synthetic_root: Path, synthetic_report: Phase1Report
) -> None:
    """A11: si algo anterior cambiase de verdad, el bloque lo declara `fail` y lo lista."""
    original_runs = backtest_report.run_all_baselines

    def hostile(*args: object, **kwargs: object) -> tuple[backtest_report.BaselineOutcome, ...]:
        runs = original_runs(cast("Any", args[0]), **cast("Any", kwargs))
        first = runs[1]  # `always_long`: la fila que la auditoria compara
        fold = first.run.folds[0]
        altered = replace(fold.sessions[0], pnl_declared_pct=123.0)
        trimmed = replace(fold, sessions=(altered, *fold.sessions[1:]))
        run = replace(first.run, folds=(trimmed, *first.run.folds[1:]))
        return (runs[0], replace(first, run=run), *runs[2:])

    monkeypatch.setattr(backtest_report, "run_all_baselines", hostile)
    block = phase1_report.audit_lookahead(store=Store(synthetic_root), base=synthetic_report.rerun)
    assert block["state"] == str(HalfResult.FAIL)
    assert block["violations"]


def test_a25_blockers_without_a_failed_gate() -> None:
    """A25/A30: con la puerta `pass` no se anade el bloqueo del veredicto."""
    codes = {
        entry["code"]
        for entry in phase1_report._blockers(  # pyright: ignore[reportPrivateUsage]
            gate=GateVerdict.PASS
        )
    }
    assert "gate_not_passed" not in codes
    assert "gate_not_passed" in {
        entry["code"]
        for entry in phase1_report._blockers(  # pyright: ignore[reportPrivateUsage]
            gate=GateVerdict.NOT_EVALUABLE
        )
    }


def test_a25_null_slippage_ratios_are_published_as_null() -> None:
    """A25/A18: con un *slippage* sin supuesto, los ratios se publican como `null`."""
    unmeasured = SlippageParameter.unmeasured(reason="sin supuesto declarado")
    ratio = phase1_report._pct_of_r_ratio(unmeasured)  # pyright: ignore[reportPrivateUsage]
    percent = phase1_report._pct_of_r_declared_percent(  # pyright: ignore[reportPrivateUsage]
        unmeasured
    )
    assert ratio is None
    assert percent is None


def test_a25_partial_verdict_covers_both_halves(
    synthetic_report: Phase1Report,
) -> None:
    """A25/A15: el veredicto parcial describe los dos desenlaces posibles de la mitad (a)."""
    undefined = phase1_report._partial_verdict(  # pyright: ignore[reportPrivateUsage]
        half_a=str(HalfResult.NOT_EVALUABLE),
        half_b=str(HalfResult.NOT_EVALUABLE),
        gate=GateVerdict.NOT_EVALUABLE,
    )
    assert "no evaluable hoy" in undefined
    assert "medida hoy" in phase1_report._partial_verdict(  # pyright: ignore[reportPrivateUsage]
        half_a=str(HalfResult.PASS),
        half_b=str(HalfResult.NOT_EVALUABLE),
        gate=GateVerdict.NOT_EVALUABLE,
    )


def test_a25_naive_as_of_is_read_as_utc(tmp_path: Path, synthetic_root: Path) -> None:
    """A25/A2: un `as_of` sin zona se interpreta como UTC (nunca como hora local)."""
    reports = tmp_path / "derived" / "reports"
    _artifact_pair(reports, _synthetic_artifact(synthetic_root))
    report = phase1_report.analyse(
        store=Store(synthetic_root),
        reports_dir=reports,
        as_of=datetime(2026, 9, 19),  # sin zona: se lee como UTC
        write=False,
    )
    assert report.payload["generated_at"] == NOW.isoformat()


def test_a25_cli_reports_missing_datasets(tmp_path: Path, synthetic_root: Path) -> None:
    """A25/A23: la CLI sale con 2 cuando el almacen no tiene los datasets."""
    reports = tmp_path / "derived" / "reports"
    _artifact_pair(reports, _synthetic_artifact(synthetic_root))
    bare = tmp_path / "bare"
    bare.mkdir()
    assert (
        phase1_report.main(
            [
                "--data-root",
                str(bare),
                "--reports-dir",
                str(reports),
                "--as-of",
                NOW.isoformat(),
            ]
        )
        == 2
    )


def test_a25_lookahead_invariants_are_enforced(
    monkeypatch: pytest.MonkeyPatch, synthetic_root: Path, synthetic_report: Phase1Report
) -> None:
    """A25/A11: las invariantes de la mutacion y de la rejilla se comprueban, no se asumen."""
    store = Store(synthetic_root)

    without_folds = replace(synthetic_report.rerun.split_plan, folds=())
    with pytest.raises(phase1_report.AuditInvariantError):
        phase1_report.audit_lookahead(
            store=store, base=replace(synthetic_report.rerun, split_plan=without_folds)
        )

    def ancient_cutoff(_frame: object) -> date:
        return date(1900, 1, 1)

    monkeypatch.setattr(phase1_report, "clean_sample_cutoff", ancient_cutoff)
    with pytest.raises(phase1_report.AuditInvariantError):
        phase1_report.audit_lookahead(store=store, base=synthetic_report.rerun)
    monkeypatch.undo()

    original_inputs = backtest_report.build_inputs

    def shrunk(history: backtest_report.History, *, calendar: object) -> backtest_report.Universe:
        universe = original_inputs(history, calendar=cast("Any", calendar))
        return replace(universe, inputs=universe.inputs[:-1])

    monkeypatch.setattr(backtest_report, "build_inputs", shrunk)
    with pytest.raises(phase1_report.AuditInvariantError):
        phase1_report.audit_lookahead(store=store, base=synthetic_report.rerun)
    monkeypatch.undo()

    original_runs = backtest_report.run_all_baselines

    def fewer_baselines(
        *args: object, **kwargs: object
    ) -> tuple[backtest_report.BaselineOutcome, ...]:
        runs = original_runs(cast("Any", args[0]), **cast("Any", kwargs))
        return runs[:-1]

    monkeypatch.setattr(backtest_report, "run_all_baselines", fewer_baselines)
    with pytest.raises(phase1_report.AuditInvariantError):
        phase1_report.audit_lookahead(store=store, base=synthetic_report.rerun)
    monkeypatch.undo()

    def fewer_folds(*args: object, **kwargs: object) -> tuple[backtest_report.BaselineOutcome, ...]:
        runs = original_runs(cast("Any", args[0]), **cast("Any", kwargs))
        first = replace(runs[0].run, folds=runs[0].run.folds[:-1])
        return (replace(runs[0], run=first), *runs[1:])

    monkeypatch.setattr(backtest_report, "run_all_baselines", fewer_folds)
    with pytest.raises(phase1_report.AuditInvariantError):
        phase1_report.audit_lookahead(store=store, base=synthetic_report.rerun)
    monkeypatch.undo()

    def fewer_sessions(
        *args: object, **kwargs: object
    ) -> tuple[backtest_report.BaselineOutcome, ...]:
        runs = original_runs(cast("Any", args[0]), **cast("Any", kwargs))
        first_fold = runs[0].run.folds[0]
        trimmed = replace(first_fold, sessions=first_fold.sessions[:-1])
        run = replace(runs[0].run, folds=(trimmed, *runs[0].run.folds[1:]))
        return (replace(runs[0], run=run), *runs[1:])

    monkeypatch.setattr(backtest_report, "run_all_baselines", fewer_sessions)
    with pytest.raises(phase1_report.AuditInvariantError):
        phase1_report.audit_lookahead(store=store, base=synthetic_report.rerun)
