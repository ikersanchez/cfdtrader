"""Tests de la fuente de apertura del drift (tarea #52).

Almacenes sintéticos de **varias series**: con estructura conocida se comprueba que
el módulo ancla la ventana con el corte limpio de la serie de referencia, publica
el registro por fuente con la forma del contrato y lee los veredictos **relativos**
(el ETF confirma la referencia; el futuro da otro veredicto). Los tests que usan el
almacén real van ``skipif``, como en el resto del repositorio.

El módulo **no** reimplementa la regla de #52: importa ``load_sessions``,
``decompose``, ``clean_sample`` y ``clean_sample_cutoff`` de
:mod:`cfdtrader.analysis.drift` (comprobado por AST).
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Final, cast

import pytest

from cfdtrader.analysis import drift, drift_open_source, phase0_report
from cfdtrader.backtest.engine import canonical_text
from cfdtrader.data.store import Store

NOW = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)
START = date(2022, 1, 3)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"
REAL_REPORTS: Final[Path] = REAL_DATA / "derived" / "reports"
MODULE_PATH: Final[Path] = Path(str(drift_open_source.__file__))
SOURCE: Final[str] = MODULE_PATH.read_text(encoding="utf-8")

#: Base del guardia de diff (criterio 15): el commit sobre el que se entregó #52.
BASE_COMMIT: Final[str] = "97511c7"

#: Ficheros que la entrega escribe (criterio 15).
WRITTEN: Final[set[str]] = {
    "src/cfdtrader/analysis/drift_open_source.py",
    "tests/test_drift_open_source.py",
    "_docs/data_sources.md",
}

#: Ficheros congelados que la entrega no puede tocar (criterio 15), en las dos
#: formas de ruta (``git diff`` devuelve rutas completas).
FROZEN: Final[set[str]] = {
    # #108 retira `src/cfdtrader/analysis/drift.py`: el barrido de punteros `#50 -> #107` toca
    # sus limitaciones declaradas.
    "backtest/baselines.py",
    "backtest/metrics.py",
    "models/baseline.py",
    "analysis/feature_frame.py",
    "src/cfdtrader/backtest/baselines.py",
    "src/cfdtrader/backtest/metrics.py",
    "src/cfdtrader/models/baseline.py",
    "src/cfdtrader/analysis/feature_frame.py",
}

#: Nombres que el módulo debe **importar** de `drift` y no redefinir (criterio 1).
REUSED: Final[tuple[str, ...]] = (
    "load_sessions",
    "decompose",
    "clean_sample",
    "clean_sample_cutoff",
)

needs_store = pytest.mark.skipif(
    not (REAL_DATA / "derived" / "labels").exists(),
    reason="el almacén real no está en el árbol: los números son los suyos",
)


# ─────────────────────────────────────────────────────────────────────────────
# Datos sintéticos
# ─────────────────────────────────────────────────────────────────────────────
def _weekdays(count: int) -> list[date]:
    """Los primeros ``count`` días laborables desde ``START``."""
    days: list[date] = []
    cursor = START
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _bars(
    count: int,
    *,
    series_id: str,
    overnight_bp: float,
    intraday_cycle: list[float],
    stale_prefix: int = 0,
) -> list[dict[str, object]]:
    """Sesiones sintéticas: sesgo nocturno, ciclo intradía y prefijo con el `open` repetido.

    El ruido nocturno determinista de ±0,4 pb evita la varianza exactamente cero
    (con ella el contraste t degenera). ``stale_prefix`` repite el cierre anterior
    como apertura en las primeras sesiones: es el artefacto de #52.
    """
    previous_close = 5000.0
    records: list[dict[str, object]] = []
    days = _weekdays(count)
    fetched_at = datetime.combine(days[-1], time(23, 0), tzinfo=UTC)
    for index, day in enumerate(days):
        noise_bp = (+0.4, -0.4)[index % 2]
        open_price = previous_close * (1.0 + (overnight_bp + noise_bp) / 10_000.0)
        if index < stale_prefix:
            open_price = previous_close
        intraday_bp = intraday_cycle[index % len(intraday_cycle)]
        close_price = open_price * (1.0 + intraday_bp / 10_000.0)
        records.append(
            {
                "source": "yfinance",
                "series_id": series_id,
                "as_of": datetime(day.year, day.month, day.day, 21, 0, tzinfo=UTC),
                "fetched_at": fetched_at,
                "published_at": None,
                "open": open_price,
                "high": max(open_price, close_price) * 1.0005,
                "low": min(open_price, close_price) * 0.9995,
                "close": close_price,
                "volume": 1_000_000.0,
                "adj_close": close_price,
            }
        )
        previous_close = close_price
    return records


def _write_store(tmp_path: Path, by_series: dict[str, list[dict[str, object]]]) -> Path:
    """Escribe esas series en `raw.market_daily` de un almacén temporal."""
    store = Store(tmp_path)
    records = [record for series_records in by_series.values() for record in series_records]
    store.append("raw", "market_daily", records)
    return tmp_path


def _synthetic_store(tmp_path: Path, *, future_stale_prefix: int = 300) -> Path:
    """`^GSPC`/`SPY` con drift nocturno y un `ES=F` con el `open` repetido y sesión propia."""
    return _write_store(
        tmp_path,
        {
            "^GSPC": _bars(600, series_id="^GSPC", overnight_bp=+6.0, intraday_cycle=[+2.0, -2.0]),
            "SPY": _bars(600, series_id="SPY", overnight_bp=+6.0, intraday_cycle=[+2.0, -2.0]),
            "ES=F": _bars(
                600,
                series_id="ES=F",
                overnight_bp=0.0,
                intraday_cycle=[+5.0, +4.0, +6.0],
                stale_prefix=future_stale_prefix,
            ),
        },
    )


def _report(tmp_path: Path, *, reports_dir: Path | None = None) -> drift_open_source.Report:
    """El informe sintético, sin escribir salvo que se pida."""
    return drift_open_source.analyse(
        data_root=_synthetic_store(tmp_path),
        now=NOW,
        reports_dir=reports_dir,
    )


def _by_series(report: drift_open_source.Report) -> dict[str, dict[str, object]]:
    """Los registros de fuente indexados por `series_id`."""
    sources = cast("list[dict[str, object]]", report.payload["sources"])
    return {str(source["series_id"]): source for source in sources}


# ─────────────────────────────────────────────────────────────────────────────
# La ventana común y la forma del registro
# ─────────────────────────────────────────────────────────────────────────────
def test_the_new_artifact_is_written_and_it_is_not_the_drift_name(tmp_path: Path) -> None:
    """El artefacto nuevo existe con su nombre y no se escribe ningún `drift_decomposition_*`."""
    reports = tmp_path / "derived" / "reports"
    report = _report(tmp_path, reports_dir=reports)

    assert report.report_stem == "drift_open_source_2026-10-01"
    json_path = reports / "drift_open_source_2026-10-01.json"
    markdown_path = reports / "drift_open_source_2026-10-01.md"
    assert json_path.is_file() and markdown_path.is_file()
    assert not list(reports.glob("drift_decomposition_*"))
    assert report.report_sha256.startswith("sha256:") and len(report.report_sha256) == 71


def test_the_window_is_anchored_on_the_reference_series(tmp_path: Path) -> None:
    """`window_from` es el corte limpio de `^GSPC` y es **idéntico** en las tres fuentes."""
    frame = drift.load_sessions(Store(_synthetic_store(tmp_path)), series_id="^GSPC")
    cutoff = drift.clean_sample_cutoff(frame)
    assert cutoff is not None

    report = _report(tmp_path)
    assert report.payload["window_from"] == cutoff.isoformat()
    assert report.payload["reference_series"] == "^GSPC"

    sources = cast("list[dict[str, object]]", report.payload["sources"])
    assert len(sources) == 3
    for source in sources:
        window = cast("dict[str, object]", source["window"])
        assert int(cast("int", window["sessions"])) >= drift.MIN_CLEAN_SESSIONS
        assert int(cast("int", source["full_sessions"])) >= 250
        assert str(source["first_session"]) < str(source["last_session"])
        assert 0.0 <= float(cast("float", source["stale_open_share"])) <= 1.0
        assert str(source["open_quality"]) in {"ok", "degraded", "unusable"}


def test_the_record_shape_matches_the_contract(tmp_path: Path) -> None:
    """Cada fuente publica los 3 tramos con sus 6 campos, la diferencia y los dos veredictos."""
    by_series = _by_series(_report(tmp_path))
    assert set(by_series) == {"^GSPC", "SPY", "ES=F"}
    for source in by_series.values():
        window = cast("dict[str, object]", source["window"])
        segments = cast("list[dict[str, object]]", window["segments"])
        assert [str(segment["name"]) for segment in segments] == ["intraday", "overnight", "total"]
        for segment in segments:
            assert set(segment) == {
                "name",
                "sessions",
                "mean_bp",
                "t_stat",
                "p_value",
                "hit_rate",
            }
            assert int(cast("int", segment["sessions"])) == int(cast("int", window["sessions"]))
        difference = cast("dict[str, object]", window["difference"])
        assert set(difference) == {"mean_difference_bp", "t_stat", "p_value"}
        all_sessions = cast("dict[str, object]", window["all_sessions"])
        assert set(all_sessions) == {"sessions", "verdict", "phase0_gate"}


# ─────────────────────────────────────────────────────────────────────────────
# Lectura relativa de veredictos
# ─────────────────────────────────────────────────────────────────────────────
def test_the_etf_confirms_the_reference_and_the_future_does_not(tmp_path: Path) -> None:
    """`SPY` reproduce `overnight`/`fail`; `ES=F` da otro veredicto y declara su `open` repetido."""
    report = _report(tmp_path)
    by_series = _by_series(report)
    reference = cast("dict[str, object]", by_series["^GSPC"]["window"])
    spy = cast("dict[str, object]", by_series["SPY"]["window"])
    future = cast("dict[str, object]", by_series["ES=F"]["window"])

    assert str(reference["verdict"]) == "overnight"
    assert str(reference["phase0_gate"]) == "fail"
    assert str(spy["verdict"]) == "overnight"
    assert str(spy["phase0_gate"]) == "fail"

    assert str(future["verdict"]) != "overnight"
    assert float(cast("float", future["stale_open_share"])) > 0.0
    declarations = " ".join(
        str(item)
        for item in cast("list[object]", report.payload["limitations"])
        + cast("list[object]", report.payload["notes"])
    )
    assert "futuro" in declarations and "sesión" in declarations


def test_the_decision_keeps_the_reference_and_declares_it_is_not_corrected(
    tmp_path: Path,
) -> None:
    """La decisión mantiene `^GSPC`, no lo corrige, y su razón cita `SPY` y `ES=F`."""
    decision = cast("dict[str, object]", _report(tmp_path).payload["decision"])
    assert decision["source"] == "^GSPC"
    assert decision["corrected"] is False
    rationale = str(decision["rationale"])
    assert rationale
    assert "SPY" in rationale and "ES=F" in rationale


def test_a_custom_series_list_keeps_the_reason_non_empty(tmp_path: Path) -> None:
    """Sin `SPY` ni `ES=F` la decisión sigue siendo válida y razonada."""
    report = drift_open_source.analyse(
        data_root=_synthetic_store(tmp_path),
        now=NOW,
        series_ids=("^GSPC",),
    )
    decision = cast("dict[str, object]", report.payload["decision"])
    assert decision["source"] == "^GSPC"
    assert str(decision["rationale"])


# ─────────────────────────────────────────────────────────────────────────────
# Digest, determinismo y CLI
# ─────────────────────────────────────────────────────────────────────────────
def test_the_digest_is_self_consistent_and_deterministic(tmp_path: Path) -> None:
    """Dos corridas dan el mismo texto y `report_sha256` es el digest canónico sin esa clave."""
    first = _report(tmp_path)
    second = _report(tmp_path)

    assert first.json_text() == second.json_text()
    without = {key: value for key, value in first.payload.items() if key != "report_sha256"}
    expected = "sha256:" + hashlib.sha256(canonical_text(without).encode("utf-8")).hexdigest()
    assert first.report_sha256 == expected
    assert json.loads(first.json_text())["report_sha256"] == expected


def test_the_cli_writes_the_artifact_and_returns_zero(tmp_path: Path) -> None:
    """La CLI con `--now` fijo escribe el artefacto y sale con 0, sin tocar la red."""
    reports = tmp_path / "out"
    code = drift_open_source.main(
        [
            "--data-root",
            str(_synthetic_store(tmp_path)),
            "--reports-dir",
            str(reports),
            "--now",
            "2026-10-01T00:00:00+00:00",
        ]
    )
    assert code == 0
    assert (reports / "drift_open_source_2026-10-01.json").is_file()


def test_the_cli_fails_loudly_with_bad_settings_and_without_data(tmp_path: Path) -> None:
    """Configuración inválida ⇒ 1; almacén sin datos ⇒ 2: nunca un artefacto inventado."""
    assert (
        drift_open_source.main(
            ["--settings", str(tmp_path / "no-existe.yaml"), "--now", "2026-10-01T00:00:00Z"]
        )
        == 1
    )
    assert (
        drift_open_source.main(
            [
                "--data-root",
                str(tmp_path / "vacio"),
                "--reports-dir",
                str(tmp_path / "out"),
                "--now",
                "2026-10-01T00:00:00Z",
            ]
        )
        == 2
    )


# ─────────────────────────────────────────────────────────────────────────────
# Errores declarados
# ─────────────────────────────────────────────────────────────────────────────
def test_a_reference_without_a_clean_era_is_an_error(tmp_path: Path) -> None:
    """Si la referencia no tiene era limpia, no hay ventana común y se falla con el motivo."""
    records = _bars(600, series_id="^GSPC", overnight_bp=+6.0, intraday_cycle=[+2.0, -2.0])
    stale = [
        {**record, "open": previous["close"]}
        for record, previous in zip(records[1:], records, strict=False)
    ]
    _write_store(tmp_path, {"^GSPC": [records[0], *stale]})
    with pytest.raises(drift_open_source.DriftOpenSourceError, match="era limpia"):
        drift_open_source.analyse(data_root=tmp_path, now=NOW, series_ids=("^GSPC",))


def test_a_series_without_enough_clean_sessions_is_an_error(tmp_path: Path) -> None:
    """Una fuente con menos de 250 sesiones limpias en la ventana no se publica como medición."""
    _synthetic_store(tmp_path, future_stale_prefix=520)
    with pytest.raises(drift_open_source.DriftOpenSourceError, match="250"):
        drift_open_source.analyse(data_root=tmp_path, now=NOW)


def test_the_reference_must_be_in_the_series_list(tmp_path: Path) -> None:
    """Sin la serie de referencia no se puede anclar la ventana."""
    with pytest.raises(drift_open_source.DriftOpenSourceError, match="referencia"):
        drift_open_source.analyse(data_root=tmp_path, now=NOW, series_ids=("SPY",))


# ─────────────────────────────────────────────────────────────────────────────
# Reuso por import (criterio 1) y guardia de diff (criterio 15)
# ─────────────────────────────────────────────────────────────────────────────
def test_the_module_reuses_the_drift_rule_by_import() -> None:
    """Los cuatro nombres se importan de `cfdtrader.analysis.drift` y no se redefinen (AST)."""
    tree = ast.parse(SOURCE)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "cfdtrader.analysis.drift":
            imported.update(alias.name for alias in node.names)
    assert set(REUSED) <= imported

    defined: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
    assert defined.isdisjoint(REUSED)


def test_the_delivery_stays_inside_the_allowed_files() -> None:
    """SUBSET + DISJUNTO: lo escrito está en el diff y nada congelado se ha tocado (criterio 15)."""
    changed = set(
        subprocess.run(  # noqa: S603 - el git del sistema, comando fijo
            ["git", "diff", "--name-only", f"{BASE_COMMIT}..HEAD"],  # noqa: S607
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
    )
    assert changed >= WRITTEN, "todo lo escrito debe estar en el diff (WRITTEN <= changed)"
    assert changed.isdisjoint(FROZEN), "ningún fichero congelado puede aparecer en el diff"


# ─────────────────────────────────────────────────────────────────────────────
# Almacén real (skipif)
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_the_real_store_publishes_the_three_sources_with_their_verdicts() -> None:
    """Sobre el almacén real: `2014-01-01`, `SPY` confirma y `ES=F` no sustituye."""
    report = drift_open_source.analyse(
        data_root=REAL_DATA,
        now=datetime(2026, 10, 1, 0, 0, tzinfo=UTC),
    )
    assert report.payload["window_from"] == "2014-01-01"
    by_series = {
        str(source["series_id"]): source
        for source in cast("list[dict[str, object]]", report.payload["sources"])
    }
    assert set(by_series) == {"^GSPC", "SPY", "ES=F"}

    spy = cast("dict[str, object]", by_series["SPY"]["window"])
    assert str(spy["verdict"]) == "overnight"
    assert str(spy["phase0_gate"]) == "fail"
    spy_segments = {
        str(item["name"]): item for item in cast("list[dict[str, object]]", spy["segments"])
    }
    assert float(cast("float", spy_segments["intraday"]["p_value"])) >= drift.ALPHA
    assert float(cast("float", spy_segments["overnight"]["mean_bp"])) > 0.0

    future = cast("dict[str, object]", by_series["ES=F"]["window"])
    assert str(future["verdict"]) != "overnight"
    assert float(cast("float", future["stale_open_share"])) > 0.0
    for source in by_series.values():
        window = cast("dict[str, object]", source["window"])
        assert int(cast("int", window["sessions"])) >= drift.MIN_CLEAN_SESSIONS


@needs_store
def test_the_new_artifact_does_not_shadow_the_fase0_selection(tmp_path: Path) -> None:
    """Tras escribir el artefacto nuevo, la Fase 0 sigue eligiendo el `drift_decomposition`."""
    reports = tmp_path / "reports"
    shutil.copytree(REAL_REPORTS, reports)
    klass = phase0_report.ARTIFACT_CLASSES[0]
    before = phase0_report.select_artifact(reports, klass).name
    assert before == "drift_decomposition_2026-09-18.json"

    drift_open_source.analyse(
        data_root=REAL_DATA,
        now=datetime(2026, 10, 1, 0, 0, tzinfo=UTC),
        reports_dir=reports,
    )
    after = phase0_report.select_artifact(reports, klass).name
    assert after == before


@needs_store
def test_the_frozen_artifacts_are_not_rewritten_by_a_real_run(tmp_path: Path) -> None:
    """Una corrida real escribiendo en un temporal no toca los artefactos congelados."""
    before = {
        name: hashlib.sha256((REAL_REPORTS / name).read_bytes()).hexdigest()
        for name in (
            "drift_decomposition_2026-09-18.json",
            "drift_decomposition_2026-09-18.md",
            "phase0_report_2026-09-18.json",
            "phase0_report_2026-09-18.md",
        )
    }
    drift_open_source.analyse(
        data_root=REAL_DATA,
        now=datetime(2026, 10, 1, 0, 0, tzinfo=UTC),
        reports_dir=tmp_path / "out",
    )
    after = {
        name: hashlib.sha256((REAL_REPORTS / name).read_bytes()).hexdigest() for name in before
    }
    assert after == before
