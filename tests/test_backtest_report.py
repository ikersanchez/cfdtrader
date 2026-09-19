"""Tests del adaptador ``Store`` -> ``SessionInput`` y del informe de Fase 1 (#69).

Un test por criterio (``test_a1_...`` .. ``test_a33_...``), siempre con ``tmp_path`` para los
almacenes sinteticos: la fixture de sesion de ``tests/conftest.py`` huella el ``data/`` del
repositorio antes y despues, asi que **ningun test escribe ahi**. Los que miden los numeros
reales de A6/A7/A14 leen ese ``data/`` en **solo lectura** y se saltan con un motivo declarado
si no esta en el arbol (mismo criterio que ``tests/test_phase0_report.py``).

Los bordes incomodos que exige A31 estan cubiertos: sesion con intradia parcial y ``bars =
None`` (A10), sesion a caballo del cambio de hora (A11), sesion duplicada (A12), mutacion de
una sesion posterior (A13), horizonte distinto de 0 (A15), ``seed`` distinto en
``random_matched`` (A16), ``write=False`` y ``--as-of`` ausente (A3/A27), **segunda pasada** con
los mismos datos (A25, el defecto tipico de este proyecto) y el rechazo de ``calculate_metrics``
(A21).
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Final, cast

import polars as pl
import pytest

from cfdtrader.analysis import backtest_report, drift, phase0_report
from cfdtrader.analysis.backtest_report import (
    ADAPTER_DOES_NOT_DO,
    FOLLOW_UPS,
    INTRADAY_INTERVAL,
    NOTIONAL_USD,
    PHASE1_PLAN,
    PRICE_PROXY_OF,
    RANDOM_MATCHED_FREQUENCY,
    RANDOM_MATCHED_SEED,
    REPORT_HASH_FORMAT,
    REPORT_PREFIX,
    SERIES_ID,
    BacktestReport,
    BacktestReportError,
    BaselineOutcome,
    DuplicateSessionError,
    History,
    InsufficientSampleError,
    InvalidAsOfError,
    LabelHorizonError,
    MissingAsOfError,
    MissingDatasetError,
    MissingPriceError,
    PlanParams,
    Universe,
    analyse,
    build_inputs,
    build_split_plan,
    label_horizon_sequence,
    load_history,
    main,
    run_all_baselines,
)
from cfdtrader.analysis.drift import clean_sample_cutoff, session_stale_open
from cfdtrader.backtest.baselines import (
    BASELINE_IDS,
    BASELINES_DOES_NOT_DO,
    RANDOM_MATCHED,
    InvalidBaselineParameterError,
)
from cfdtrader.backtest.costs import (
    SlippageParameter,
    declared_cost_model,
    declared_slippage_assumption,
)
from cfdtrader.backtest.engine import (
    STATUS_TRADED,
    EngineInputError,
    SessionInput,
    SessionOutcome,
    canonical_text,
)
from cfdtrader.backtest.metrics import MetricsInputError, calculate_metrics
from cfdtrader.backtest.splits import InsufficientSessionsError, walk_forward_splits
from cfdtrader.data.calendar import load_calendar
from cfdtrader.data.store import Store

MODULE_PATH: Final[Path] = Path(str(backtest_report.__file__))
SOURCE: Final[str] = MODULE_PATH.read_text(encoding="utf-8")
TEST_PATH: Final[Path] = Path(__file__).resolve()
TEST_SOURCE: Final[str] = TEST_PATH.read_text(encoding="utf-8")
REPO_ROOT: Final[Path] = TEST_PATH.parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"
REAL_REPORTS: Final[Path] = REAL_DATA / "derived" / "reports"

#: Instante **declarado** de todas las corridas de la suite: el modulo nunca lee el reloj.
NOW: Final[datetime] = datetime(2026, 9, 19, tzinfo=UTC)


def _business_days(start: date, count: int) -> list[date]:
    """``count`` dias laborables consecutivos desde ``start`` (sin fines de semana)."""
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


#: Dias laborables de la muestra sintetica (>= 250 limpias y > 500 para el plan de A14).
SYNTHETIC_SESSIONS: Final[tuple[date, ...]] = tuple(_business_days(date(2024, 1, 2), 600))

#: Literales de la tabla declarada de #8 que el modulo **no** puede contener (A18).
DECLARED_TABLE_LITERALS: Final[tuple[str, ...]] = (
    "0.42",
    "0.0042",
    "1.82",
    "-0.18",
    "0.24",
    "2.24",
)

#: Claves que el informe **no** puede publicar: son las metricas netas de #15 (A20, A21).
FORBIDDEN_METRIC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "sharpe",
        "sharpe_ratio",
        "sortino",
        "sortino_ratio",
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


def _block(report: BacktestReport, *keys: str) -> dict[str, object]:
    """Un bloque anidado del payload, ya tipado."""
    node: object = report.payload
    for key in keys:
        node = _mapping(node)[key]
    return _mapping(node)


def _rows(report: BacktestReport) -> list[dict[str, object]]:
    """Las seis filas de la tabla comparativa, ya tipadas."""
    raw = cast("list[object]", _block(report, "baselines")["rows"])
    return [cast("dict[str, object]", row) for row in raw]


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
                "series_id": SERIES_ID,
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
            "series_id": SERIES_ID,
            "as_of": datetime(session.year, session.month, session.day, 21, tzinfo=UTC),
            "fetched_at": NOW,
            "published_at": None,
            "session": session,
        }
        for session in sessions
    ]


def _intraday_records(session: date, moments: Sequence[tuple[int, int]]) -> list[dict[str, object]]:
    """Barras de 5 minutos de esa sesion, en los instantes UTC indicados."""
    records: list[dict[str, object]] = []
    price = 100.0
    for hour, minute in moments:
        price += 0.05
        moment = datetime(session.year, session.month, session.day, hour, minute, tzinfo=UTC)
        records.append(
            {
                "source": "yfinance",
                "series_id": SERIES_ID,
                "as_of": moment,
                "fetched_at": NOW,
                "published_at": None,
                "open": price,
                "high": price * 1.001,
                "low": price * 0.999,
                "close": price,
                "volume": 10.0,
                "interval": "5m",
                "bid": None,
                "ask": None,
            }
        )
    return records


def _write_store(
    root: Path,
    *,
    sessions: Sequence[date] = SYNTHETIC_SESSIONS,
    daily: Sequence[Mapping[str, object]] | None = None,
    labels: Sequence[Mapping[str, object]] | None = None,
    with_labels: bool = True,
    intraday: Sequence[Mapping[str, object]] | None = None,
) -> Store:
    """Almacen temporal con las sesiones diarias, sus etiquetas y (si se pide) el intradia."""
    store = Store(root)
    store.append(
        "raw",
        "market_daily",
        [dict(item) for item in (daily if daily is not None else _daily_records(sessions))],
    )
    if with_labels:
        store.append(
            "derived",
            "labels",
            [dict(item) for item in (labels if labels is not None else _labels_records(sessions))],
        )
    if intraday:
        store.append("raw", "market_intraday", [dict(item) for item in intraday])
    return store


def _history_from_records(
    daily: Sequence[Mapping[str, object]],
    *,
    label_sessions: Sequence[date],
    intraday: Sequence[Mapping[str, object]] = (),
) -> History:
    """``History`` en memoria: permite fijar el ``as_of`` de cada fila al segundo.

    Las etiquetas se **ordenan** por sesion, igual que el ``ORDER BY session`` de
    ``load_history``: el helper reproduce el ``History`` que produce el adaptador y no uno en
    desorden, que haria saltar la validacion de A12 por culpa de la fixture.
    """
    daily_frame = pl.DataFrame([dict(item) for item in daily]).with_columns(
        pl.col("as_of").cast(pl.Datetime("us", "UTC"))
    )
    labels_frame = (
        pl.DataFrame({"session": list(label_sessions)})
        .with_columns(pl.col("session").cast(pl.Date()))
        .sort("session")
    )
    if intraday:
        intraday_frame = (
            pl.DataFrame([dict(item) for item in intraday])
            .with_columns(pl.col("as_of").cast(pl.Datetime("us", "UTC")))
            .with_columns(
                pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("session")
            )
        )
    else:
        intraday_frame = pl.DataFrame(
            schema={
                "as_of": pl.Datetime("us", "UTC"),
                "high": pl.Float64(),
                "low": pl.Float64(),
                "session": pl.Date(),
            }
        )
    return History(
        series_id=SERIES_ID,
        daily=session_stale_open(daily_frame).sort("session"),
        labels=labels_frame,
        intraday=intraday_frame,
    )


def _traded(report: BacktestReport) -> list[SessionOutcome]:
    """Todas las sesiones ``traded`` de las seis corridas, en orden."""
    return [
        session
        for outcome in report.outcomes
        for fold in outcome.run.folds
        for session in fold.sessions
        if session.status == STATUS_TRADED
    ]


def _keys(node: object) -> set[str]:
    """Todas las claves de un payload anidado (para comprobar que no se publica una metrica)."""
    found: set[str] = set()
    if isinstance(node, dict):
        mapping = cast("dict[object, object]", node)
        for key, value in mapping.items():
            found.add(str(key))
            found |= _keys(value)
    elif isinstance(node, (list, tuple)):
        sequence = cast("list[object] | tuple[object, ...]", node)
        for item in sequence:
            found |= _keys(item)
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Almacenes y corridas de la suite
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Raiz del almacen sintetico (600 sesiones, sin intradia), construida una sola vez."""
    root = tmp_path_factory.mktemp("backtest_report")
    _write_store(root)
    return root


@pytest.fixture(scope="module")
def synthetic_report(synthetic_root: Path) -> BacktestReport:
    """Corrida del arnes sobre el almacen sintetico, **sin escribir nada**."""
    return analyse(
        store=Store(synthetic_root),
        reports_dir=synthetic_root / "reports",
        as_of=NOW,
        write=False,
    )


@pytest.fixture(scope="module")
def real_report() -> BacktestReport:
    """Corrida sobre el historico real, en solo lectura; se salta si no esta en el arbol."""
    if (
        not (REAL_DATA / "raw" / "market_daily").is_dir()
        or not (REAL_DATA / "derived" / "labels").is_dir()
    ):
        pytest.skip("el almacén real no está en el árbol: los números de A6/A7/A14 no se miden")
    return analyse(store=Store(REAL_DATA), reports_dir=REAL_REPORTS, as_of=NOW, write=False)


# ─────────────────────────────────────────────────────────────────────────────
# A1 · Ficheros, API minima y frontera de capas
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_module_api_and_layer_boundary() -> None:
    """A1: existen los dos ficheros, la API minima esta en `__all__` y no hay importe inverso."""
    required = (
        "analyse",
        "main",
        "load_history",
        "build_inputs",
        "run_all_baselines",
        "render_markdown",
        "REPORT_PREFIX",
        "REPORT_HASH_FORMAT",
        "PHASE1_PLAN",
        "ADAPTER_DOES_NOT_DO",
        "FOLLOW_UPS",
        "BacktestReportError",
    )
    assert MODULE_PATH.is_file()
    assert (REPO_ROOT / "tests" / "test_backtest_report.py").is_file()
    assert [name for name in required if name not in backtest_report.__all__] == []
    assert all(hasattr(backtest_report, name) for name in required)

    subclasses = (
        MissingDatasetError,
        InsufficientSampleError,
        DuplicateSessionError,
        LabelHorizonError,
        MissingPriceError,
        MissingAsOfError,
        InvalidAsOfError,
    )
    assert all(issubclass(item, BacktestReportError) for item in subclasses)

    # Frontera de capas: nada de `src/cfdtrader/backtest/**` importa este modulo.
    backtest_dir = REPO_ROOT / "src" / "cfdtrader" / "backtest"
    offenders = [
        path.name
        for path in sorted(backtest_dir.glob("*.py"))
        if "backtest_report" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []

    # `phase0_report` y `drift` siguen con la misma firma y comportamiento.
    assert list(inspect.signature(phase0_report.analyse).parameters) == [
        "reports_dir",
        "now",
        "write",
    ]
    assert list(inspect.signature(drift.main).parameters) == ["argv"]


# ─────────────────────────────────────────────────────────────────────────────
# A2 · Firma declarada, `--as-of` y ninguna ruta lee el reloj
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_signatures_and_no_clock(tmp_path: Path, synthetic_root: Path) -> None:
    """A2: la API declarada, la CLI con `--as-of` y el modulo sin reloj."""
    parameters = inspect.signature(analyse).parameters
    assert list(parameters) == ["store", "reports_dir", "as_of", "write"]
    assert parameters["write"].default is True
    assert all(item.kind is inspect.Parameter.KEYWORD_ONLY for item in parameters.values())
    for option in ("--data-root", "--reports-dir", "--settings", "--as-of"):
        assert option in SOURCE
    for token in ("datetime.now", "datetime.utcnow", "date.today", "time.time", "utcnow()"):
        assert token not in SOURCE

    silent = tmp_path / "silent"
    report = analyse(store=Store(synthetic_root), reports_dir=silent, as_of=NOW, write=False)
    assert not silent.exists()
    assert _block(report)["generated_at"] == NOW.isoformat()
    assert report.as_of == NOW


# ─────────────────────────────────────────────────────────────────────────────
# A3 · `--as-of` obligatorio para escribir
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_as_of_is_mandatory_to_write(
    tmp_path: Path, synthetic_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A3: sin `--as-of` (o con uno no ISO) sale 2, no escribe y el motivo va a stderr."""
    reports = tmp_path / "reports"
    before = _fingerprint(reports)
    for extra in ([], ["--as-of", "no-es-fecha"]):
        code = main(["--data-root", str(synthetic_root), "--reports-dir", str(reports), *extra])
        assert code == 2
        assert _fingerprint(reports) == before
        assert "as-of" in capsys.readouterr().err
    assert not reports.exists()

    stem = f"{REPORT_PREFIX}_{NOW.date().isoformat()}"
    # Sin zona horaria: se interpreta como UTC (A2) y la fecha del nombre no cambia.
    assert (
        main(
            [
                "--data-root",
                str(synthetic_root),
                "--reports-dir",
                str(reports),
                "--as-of",
                "2026-09-19T00:00:00",
            ]
        )
        == 0
    )
    assert sorted(path.name for path in reports.iterdir()) == [f"{stem}.json", f"{stem}.md"]


# ─────────────────────────────────────────────────────────────────────────────
# A4 · Nombres de fichero y artefactos ajenos intactos
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_file_names_and_foreign_artifacts(tmp_path: Path, synthetic_root: Path) -> None:
    """A4: dos ficheros con la fecha del `as_of` en UTC, sin tocar lo que ya estaba."""
    reports = tmp_path / "reports"
    reports.mkdir()
    foreign = reports / "drift_decomposition_2026-09-18.json"
    foreign.write_text('{"ajeno": true}\n', encoding="utf-8")
    before = _fingerprint(reports)

    report = analyse(store=Store(synthetic_root), reports_dir=reports, as_of=NOW)
    stem = f"{REPORT_PREFIX}_{NOW.astimezone(UTC).date().isoformat()}"
    assert report.report_stem == stem
    assert sorted(path.name for path in reports.iterdir()) == [
        foreign.name,
        f"{stem}.json",
        f"{stem}.md",
    ]
    assert _fingerprint(reports)[foreign.name] == before[foreign.name]


# ─────────────────────────────────────────────────────────────────────────────
# A5 · El motor de consulta, nunca la lectura point-in-time
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_history_uses_the_query_engine(
    synthetic_root: Path, synthetic_report: BacktestReport
) -> None:
    """A5: los dos recuentos sobre el **mismo** almacen, no una afirmacion."""
    assert "read_pit" not in SOURCE

    store = Store(synthetic_root)
    historical = datetime(2024, 6, 3, 21, tzinfo=UTC)
    point_in_time_rows = store.read_pit("raw", "market_daily", at=historical).height
    sessions_loaded = load_history(store).daily.height

    assert point_in_time_rows == 0
    assert sessions_loaded == len(SYNTHETIC_SESSIONS)
    reported = _block(synthetic_report, "universe")
    assert reported["sessions"] == len(synthetic_report.universe.inputs)
    assert sessions_loaded > 0 and synthetic_report.universe.inputs


# ─────────────────────────────────────────────────────────────────────────────
# A6 · Universo declarado y contado
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_universe_declared_and_counted(real_report: BacktestReport) -> None:
    """A6: el universo real declarado, con su serie y sus dos extremos."""
    universe = _block(real_report, "universe")
    assert universe["series_id"] == SERIES_ID == "^GSPC"
    assert universe["labelled_sessions"] == 2687
    assert universe["sessions"] == 2687
    assert universe["first_session"] == "2016-01-07"
    assert universe["last_session"] == "2026-09-16"
    assert len(real_report.universe.inputs) == 2687


def test_a6_excluded_sessions_carry_a_reason(tmp_path: Path) -> None:
    """A6: una sesion etiquetada sin OHLC completo se declara, nunca se rellena."""
    sessions = list(SYNTHETIC_SESSIONS)
    records = _daily_records(sessions)
    broken = sessions[10]
    records[10] = {**records[10], "close": None}
    orphan = sessions[-1]
    records = [record for index, record in enumerate(records) if index != len(sessions) - 1]
    labels = _labels_records(sessions)

    store = _write_store(tmp_path / "exclusions", sessions=sessions, daily=records, labels=labels)
    report = analyse(store=store, reports_dir=tmp_path / "r", as_of=NOW, write=False)

    excluded = {
        _mapping(item)["session"]: _mapping(item)["reason"]
        for item in cast("list[object]", _block(report, "universe")["excluded"])
    }
    assert excluded[broken.isoformat()] == "missing_ohlc"
    assert excluded[orphan.isoformat()] == "no_daily_row"
    assert all(item.open_px is not None for item in report.universe.inputs)
    assert broken not in [item.session for item in report.universe.inputs]


def test_a6_clean_rule_reasons_are_declared() -> None:
    """A6/A7: los otros dos motivos de exclusion se declaran con su causa real."""
    sessions = _business_days(date(2023, 1, 2), 600)
    records = _daily_records(sessions)
    stale_2024 = next(index for index, day in enumerate(sessions) if day.year == 2024)
    for index, day in enumerate(sessions):
        if index == 0:
            continue
        # 2023 entero con el `open` repetido de #52 (su cuota anual supera la tolerancia),
        # mas una sola sesion de 2024 (su cuota anual no la supera).
        if day.year == 2023 or index == stale_2024:
            records[index] = {**records[index], "open": records[index - 1]["close"]}

    history = _history_from_records(records, label_sessions=sessions)
    universe = build_inputs(history, calendar=load_calendar(years=(2023, 2024, 2025)))
    assert universe.clean_from == date(2024, 1, 1)
    reasons = {_mapping(item)["session"]: _mapping(item)["reason"] for item in universe.excluded}
    assert reasons[sessions[5].isoformat()] == "before_clean_cutoff"
    assert reasons[sessions[stale_2024].isoformat()] == "stale_open"
    assert reasons[sessions[0].isoformat()] == "no_previous_session"
    assert all(item.open_px is not None for item in universe.inputs)


# ─────────────────────────────────────────────────────────────────────────────
# A7 · Reconciliacion publicada
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_reconciliation_without_filling_gaps(real_report: BacktestReport) -> None:
    """A7: las cifras reales de la reconciliacion, incluida la identidad completa."""
    reconciliation = _block(real_report, "reconciliation")
    assert reconciliation["raw_market_daily_rows"] == 5460
    assert reconciliation["rows_with_previous_session"] == 5459
    assert reconciliation["clean_from"] == "2014-01-01"
    assert reconciliation["clean_sessions"] == 3192
    assert reconciliation["sessions_since_cutoff"] == 3195
    assert reconciliation["excluded_by_clean_rule"] == 3
    assert reconciliation["stale_open_in_window"] == 3
    assert reconciliation["no_forecast"] == 505
    assert reconciliation["labelled"] == 2687
    expected = "3195 = 3 (stale_open) + 505 (no_forecast) + 2687 (etiquetadas)"
    assert reconciliation["identity"] == expected
    assert reconciliation["no_forecast_reason"]


# ─────────────────────────────────────────────────────────────────────────────
# A8 · La regla de muestra limpia se importa, no se copia
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_clean_rule_is_imported_not_copied(real_report: BacktestReport) -> None:
    """A8: el corte sale de `analysis.drift` y el modulo no reimplementa la regla."""
    daily = load_history(Store(REAL_DATA)).daily
    assert real_report.universe.clean_from == clean_sample_cutoff(daily) == date(2014, 1, 1)

    assert "def clean_sample" not in SOURCE
    assert "clean_sample_cutoff" in SOURCE and "session_stale_open" in SOURCE
    assert "STALE_OPEN_TOLERANCE" not in SOURCE and "MIN_CLEAN_SESSIONS" not in SOURCE
    assert "0.05" not in SOURCE
    assert "250" not in SOURCE


# ─────────────────────────────────────────────────────────────────────────────
# A9 · `SessionInput` bien formados
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_session_inputs_match_the_store(real_report: BacktestReport) -> None:
    """A9: los cuatro precios coinciden con la fila del almacen y `context` es `None`."""
    daily = Store(REAL_DATA).sql(
        "SELECT as_of, open, high, low, close FROM raw.market_daily "
        "WHERE series_id = '^GSPC' ORDER BY as_of"
    )
    rows = {
        row["session"]: row
        for row in session_stale_open(daily).sort("session").iter_rows(named=True)
    }
    inputs = real_report.universe.inputs
    assert len(inputs) == 2687
    for item in inputs:
        row = rows[item.session]
        assert (item.open_px, item.high_px, item.low_px, item.close_px) == (
            row["open"],
            row["high"],
            row["low"],
            row["close"],
        )
        assert item.context is None


# ─────────────────────────────────────────────────────────────────────────────
# A10 · Camino intradia
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_intraday_counts_and_window(tmp_path: Path, real_report: BacktestReport) -> None:
    """A10: 59/2.628 reales, y una barra fuera de la ventana no entra en `bars`."""
    universe = _block(real_report, "universe")
    with_intraday = cast("int", universe["sessions_with_intraday_path"])
    with_fallback = cast("int", universe["sessions_with_daily_fallback"])
    assert with_intraday == 59
    assert with_fallback == 2628
    assert with_intraday + with_fallback == universe["sessions"] == 2687

    session = date(2024, 6, 3)  # lunes de sesion completa en EDT
    moments = ((13, 25), (13, 30), (15, 0), (20, 0), (20, 5))
    records = _intraday_records(session, moments)
    assert INTRADAY_INTERVAL == "5m"
    store = _write_store(tmp_path / "window", intraday=records)
    stored = Store(tmp_path / "window").sql("SELECT DISTINCT interval FROM raw.market_intraday")
    assert stored.get_column("interval").to_list() == [INTRADAY_INTERVAL]
    report = analyse(store=store, reports_dir=tmp_path / "r", as_of=NOW, write=False)

    info = load_calendar(years=(2024,)).session(session)
    assert info.open_utc is not None and info.close_utc is not None
    inside = [
        (hour, minute)
        for hour, minute in moments
        if info.open_utc
        <= datetime(session.year, session.month, session.day, hour, minute, tzinfo=UTC)
        <= info.close_utc
    ]
    assert len(inside) == 3
    item = next(entry for entry in report.universe.inputs if entry.session == session)
    assert item.bars is not None
    assert len(item.bars) == 3
    reported = _block(report, "universe")
    assert reported["sessions_with_intraday_path"] == 1
    assert reported["sessions_with_daily_fallback"] == len(report.universe.inputs) - 1
    partial = next(entry for entry in report.universe.inputs if entry.session != session)
    assert partial.bars is None


def test_a10_null_rows_and_non_sessions_are_declared() -> None:
    """A10: una barra con el rango incompleto se descarta y una fecha sin sesion no aporta."""
    sessions = _business_days(date(2024, 1, 2), 260)
    saturday = date(2024, 6, 8)  # no es sesion del mercado
    records = _daily_records([*sessions, saturday])
    intraday = [
        *_intraday_records(sessions[5], ((15, 0),)),
        *_intraday_records(saturday, ((15, 0),)),
    ]
    intraday[0] = {**intraday[0], "high": None, "low": None}

    history = _history_from_records(
        records, label_sessions=[*sessions, saturday], intraday=intraday
    )
    universe = build_inputs(history, calendar=load_calendar(years=(2024,)))
    by_session = {item.session: item for item in universe.inputs}
    assert by_session[sessions[5]].bars is None  # la unica barra venia con el rango incompleto
    assert by_session[saturday].bars is None  # el calendario dice que ese dia no hay sesion
    assert universe.intraday_sessions == 0


# ─────────────────────────────────────────────────────────────────────────────
# A11 · Anclaje a `America/New_York`
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_sessions_are_anchored_to_new_york() -> None:
    """A11: la clave es la fecha ET del `as_of`, y cada fila diaria da una sesion distinta.

    Nota declarada: el cambio de hora de ``America/New_York`` es en **marzo y noviembre**; la
    mencion de octubre del enunciado corresponde al horario de Madrid, que **no** es el ancla
    del instrumento (el calendario y las sesiones van en ET).
    """
    sessions = _business_days(date(2024, 1, 2), 260)
    records = _daily_records(sessions)
    position = 130
    target = sessions[position]
    next_day = target + timedelta(days=1)
    records[position] = {
        **records[position],
        "as_of": datetime(next_day.year, next_day.month, next_day.day, 1, tzinfo=UTC),
    }

    history = _history_from_records(records, label_sessions=sessions)
    frame = history.daily
    assert frame.height == len(sessions) == 260
    session_dates = frame.get_column("session")
    utc_dates = frame.get_column("as_of").dt.date()
    # La fecha ET identifica cada sesion; la fecha UTC en crudo **no**: colisiona.
    assert session_dates.n_unique() == frame.height
    assert utc_dates.n_unique() < frame.height
    moved = frame.filter(pl.col("session") == target)
    assert moved.height == 1
    assert moved.get_column("as_of").dt.date().to_list() == [next_day]
    assert target != next_day

    universe = build_inputs(history, calendar=load_calendar(years=(2024,)))
    assert isinstance(universe, Universe)
    keys = [item.session for item in universe.inputs]
    assert keys.count(target) == 1
    assert len(keys) == len(sessions) - 1  # la primera fila no tiene sesion previa
    for day in (date(2024, 3, 11), date(2024, 11, 4)):
        assert day in sessions
        assert keys.count(day) == 1


# ─────────────────────────────────────────────────────────────────────────────
# A12 · Secuencia creciente y sin duplicados
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_duplicate_sessions_are_rejected(tmp_path: Path) -> None:
    """A12: un duplicado en el origen es error tipado **antes** de llamar a #12."""
    sessions = SYNTHETIC_SESSIONS[:6]
    twin = sessions[3]
    labels = _labels_records(sessions)
    labels.append(
        {
            "source": "cfdtrader.models.labels",
            "series_id": SERIES_ID,
            "as_of": datetime(twin.year, twin.month, twin.day, 22, tzinfo=UTC),
            "fetched_at": NOW,
            "published_at": None,
            "session": twin,
        }
    )
    store = _write_store(tmp_path / "duplicate", sessions=sessions, labels=labels)
    with pytest.raises(DuplicateSessionError):
        build_inputs(load_history(store), calendar=load_calendar(years=(2024,)))

    item = SessionInput(session=twin, open_px=1.0, high_px=1.1, low_px=0.9, close_px=1.0)
    with pytest.raises(DuplicateSessionError):
        build_split_plan((item, item))


# ─────────────────────────────────────────────────────────────────────────────
# A13 · Sin look-ahead
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_mutating_a_later_session_changes_nothing_before(tmp_path: Path) -> None:
    """A13: mutar la sesion `j` (precios o intradia) no toca ningun `SessionInput` de `i < j`."""
    sessions = list(SYNTHETIC_SESSIONS)
    base = _daily_records(sessions)
    position = 5
    later = sessions[10]
    base_intraday = _intraday_records(later, ((15, 0), (16, 0)))

    hostile_price = [dict(item) for item in base]
    hostile_price[position] = {
        **hostile_price[position],
        "open": 999.0,
        "high": 1_000.0,
        "low": 998.0,
        "close": 999.0,
    }
    hostile_intraday = _intraday_records(later, ((15, 0), (16, 0), (17, 0)))

    calendar = load_calendar(years=(2024, 2025, 2026))
    reference = build_inputs(
        load_history(
            _write_store(tmp_path / "base", sessions=sessions, daily=base, intraday=base_intraday)
        ),
        calendar=calendar,
    ).inputs
    mutated = build_inputs(
        load_history(
            _write_store(
                tmp_path / "price", sessions=sessions, daily=hostile_price, intraday=base_intraday
            )
        ),
        calendar=calendar,
    ).inputs
    extended = build_inputs(
        load_history(
            _write_store(
                tmp_path / "intraday",
                sessions=sessions,
                daily=base,
                intraday=hostile_intraday,
            )
        ),
        calendar=calendar,
    ).inputs

    cut = sessions[position]
    assert [item for item in reference if item.session < cut] == [
        item for item in mutated if item.session < cut
    ]
    # La mutacion surte efecto en la sesion mutada: la prueba no es vacua.
    reference_by_session = {item.session: item for item in reference}
    mutated_by_session = {item.session: item for item in mutated}
    extended_by_session = {item.session: item for item in extended}
    assert reference_by_session[cut] != mutated_by_session[cut]
    assert [item for item in reference if item.session < later] == [
        item for item in extended if item.session < later
    ]
    after = extended_by_session[later]
    assert after.bars is not None and len(after.bars) == 3


# ─────────────────────────────────────────────────────────────────────────────
# A14 · El plan declarado
# ─────────────────────────────────────────────────────────────────────────────
def test_a14_plan_is_the_declared_one(real_report: BacktestReport) -> None:
    """A14: los cinco parametros, los recuentos y el `plan_sha256` del plan hecho a mano."""
    plan = _block(real_report, "plan")
    declared = PlanParams(
        n_splits=10, test_size=50, embargo_sessions=5, max_train_size=None, label_horizon=0
    )
    assert declared == PHASE1_PLAN
    assert plan["n_splits"] == 10
    assert plan["test_size"] == 50
    assert plan["embargo_sessions"] == 5
    assert plan["max_train_size"] is None
    assert plan["label_horizon"] == 0
    assert plan["n_test"] == 500
    assert plan["not_in_any_test"] == 2187
    assert plan["purge_total"] == 0
    assert plan["embargo_total"] == 45
    assert plan["embargo_in_train_total"] == 0
    assert plan["exclusions_are_no_op"] is True

    sessions = [item.session for item in real_report.universe.inputs]
    hand = walk_forward_splits(
        sessions,
        label_horizon=[0] * len(sessions),
        n_splits=10,
        test_size=50,
        embargo_sessions=5,
        max_train_size=None,
    )
    assert real_report.split_plan.plan_sha256 == hand.plan_sha256
    assert real_report.split_plan == hand


# ─────────────────────────────────────────────────────────────────────────────
# A15 · Con `h = 0` la purga y el embargo son no-ops, y otro horizonte se rechaza
# ─────────────────────────────────────────────────────────────────────────────
def test_a15_zero_horizon_and_rejection(real_report: BacktestReport) -> None:
    """A15: los no-ops se publican con sus numeros y cualquier otro horizonte es error tipado."""
    plan = _block(real_report, "plan")
    assert plan["exclusions_are_no_op"] is True
    assert plan["purge_total"] == 0
    assert plan["embargo_in_train_total"] == 0
    assert plan["embargo_total"] == 45
    assert "no-ops estructurales" in str(plan["exclusions_note"])

    assert label_horizon_sequence(n_sessions=3, horizon=0) == (0, 0, 0)
    assert label_horizon_sequence(n_sessions=2, horizon=(0, 0)) == (0, 0)
    with pytest.raises(LabelHorizonError):
        label_horizon_sequence(n_sessions=2, horizon=1)
    with pytest.raises(LabelHorizonError):
        label_horizon_sequence(n_sessions=2, horizon=(0, 1))
    with pytest.raises(LabelHorizonError):
        label_horizon_sequence(n_sessions=2, horizon=(0, 0, 0))
    with pytest.raises(LabelHorizonError):
        build_split_plan(real_report.universe.inputs, params=PlanParams(label_horizon=1))
    with pytest.raises(LabelHorizonError):
        build_split_plan(real_report.universe.inputs, params=PlanParams(label_horizon=(0, 0, 0)))


# ─────────────────────────────────────────────────────────────────────────────
# A16 · Los seis baselines
# ─────────────────────────────────────────────────────────────────────────────
def test_a16_six_baselines_in_order(real_report: BacktestReport) -> None:
    """A16: seis filas sin alias, en el orden de `BASELINE_IDS`, y el control aleatorio exacto."""
    rows = _rows(real_report)
    assert (
        [row["baseline"] for row in rows]
        == list(BASELINE_IDS)
        == [
            "no_trade",
            "always_long",
            "always_short",
            "momentum_5d",
            "gap_reversal",
            "random_matched",
        ]
    )
    assert len({row["run_sha256"] for row in rows}) == 6

    random_row = rows[BASELINE_IDS.index(RANDOM_MATCHED)]
    assert Fraction(1, 2) == RANDOM_MATCHED_FREQUENCY
    assert RANDOM_MATCHED_SEED == 42
    assert random_row["frequency"] == "1/2"
    assert random_row["seed"] == RANDOM_MATCHED_SEED
    assert random_row["n_test"] == 500
    assert random_row["traded"] == 250 == math.floor(0.5 * 500)
    assert _block(real_report, "baselines", "random_matched")["traded"] == 250

    other = run_all_baselines(
        real_report.universe.inputs,
        split_plan=real_report.split_plan,
        cost_model=real_report.cost_model,
        slippage=real_report.slippage,
        baselines=(RANDOM_MATCHED,),
        seed=43,
    )
    assert isinstance(other[0], BaselineOutcome)
    assert other[0].frequency == RANDOM_MATCHED_FREQUENCY
    assert other[0].run.run_sha256 != random_row["run_sha256"]


# ─────────────────────────────────────────────────────────────────────────────
# A17 · El nocional es un plano declarado
# ─────────────────────────────────────────────────────────────────────────────
def test_a17_notional_is_flat_and_declared(real_report: BacktestReport) -> None:
    """A17: el nocional declarado, identico en los seis, y ninguna direccion sin el."""
    baselines = _block(real_report, "baselines")
    assert Decimal("10000") == NOTIONAL_USD
    assert baselines["notional_usd"] == "10000"
    assert baselines["notional_provenance"]

    for outcome in real_report.outcomes:
        for session in (item for fold in outcome.run.folds for item in fold.sessions):
            if session.status == STATUS_TRADED:
                assert session.decision is not None
                assert session.decision.notional_usd == Decimal("10000")
                assert session.notional_usd == Decimal("10000")
            else:
                assert session.decision is None or session.decision.notional_usd is None


# ─────────────────────────────────────────────────────────────────────────────
# A18 · El coste se importa
# ─────────────────────────────────────────────────────────────────────────────
def test_a18_cost_model_is_imported(real_report: BacktestReport) -> None:
    """A18: el modelo que viaja a la corrida es **igual** al declarado, y sin literales."""
    assert real_report.cost_model == declared_cost_model()
    assert _block(real_report, "cost_model") == declared_cost_model().model_dump(mode="json")
    assert _block(real_report, "limits", "costs")["source"]
    for token in DECLARED_TABLE_LITERALS:
        assert token not in SOURCE
    assert "CostModel(" not in SOURCE
    assert "declared_cost_model()" in SOURCE


# ─────────────────────────────────────────────────────────────────────────────
# A19 · `pnl_net_pct` es `null` en todas las operaciones
# ─────────────────────────────────────────────────────────────────────────────
def test_a19_pnl_net_is_null_everywhere(real_report: BacktestReport) -> None:
    """A19: el supuesto de #64 no cierra el total, y ningun valor no medido se escribe como 0."""
    traded = _traded(real_report)
    assert traded
    for session in traded:
        assert session.pnl_net_pct is None
        assert session.pnl_net_reason
    for row in _rows(real_report):
        assert row["pnl_net_pct_null_trades"] == row["traded"]
        if row["traded"]:
            assert row["pnl_net_reason"]
        else:
            assert row["pnl_net_reason"] is None

    block = _block(real_report, "slippage")
    assert block["state"] == "assumed"
    assert block["is_measurement"] is False
    assert block["pct_of_r"] == "0.2"
    assert block["r_pct"] is None
    assert block["pct_of_r_declared_percent"] == "20"
    assert block["issue"] == "#62"
    assert real_report.slippage == declared_slippage_assumption()
    assert "SlippageParameter.measured" not in SOURCE
    assert _block(real_report, "limits", "slippage")["r_pct"] is None

    # Un *slippage* sin medicion ni supuesto publica `null`, **nunca** 0 (regla `null != 0`).
    unmeasured = SlippageParameter.unmeasured(reason="prueba: no hay medicion ni supuesto")
    missing = backtest_report._slippage_payload(unmeasured)  # pyright: ignore[reportPrivateUsage]
    assert missing["state"] == "unmeasured"
    assert missing["pct_of_r"] is None
    assert missing["pct_of_r_declared_percent"] is None
    assert missing["r_pct"] is None


# ─────────────────────────────────────────────────────────────────────────────
# A20 · Tabla comparativa declarada
# ─────────────────────────────────────────────────────────────────────────────
def test_a20_comparative_table_declared_basis(real_report: BacktestReport) -> None:
    """A20: una fila por baseline con sus recuentos y su retorno de coste declarado."""
    baselines = _block(real_report, "baselines")
    assert baselines["basis"] == "declared_cost"
    assert baselines["is_validation"] is False
    assert baselines["note"]
    required = (
        "baseline",
        "n_test",
        "traded",
        "no_trade",
        "skipped",
        "trade_rate",
        "exit_reason_counts",
        "pnl_declared_pct",
        "gross_pct",
        "run_sha256",
    )
    rows = _rows(real_report)
    assert len(rows) == 6
    for row in rows:
        assert all(key in row for key in required)
        declared = _mapping(row["pnl_declared_pct"])
        gross = _mapping(row["gross_pct"])
        assert {"mean", "median", "sum"} <= set(declared)
        assert {"mean", "median", "sum"} <= set(gross)
        if row["traded"] == 0:
            assert declared["sum"] is None and declared["mean"] is None
            assert "nunca 0" in str(declared["reason"])
        else:
            assert declared["n"] == row["traded"]

    # Prohibido publicar cualquier cifra que se presente como rendimiento neto medido.
    assert not (FORBIDDEN_METRIC_KEYS & _keys(real_report.payload))


# ─────────────────────────────────────────────────────────────────────────────
# A21 · `net_metrics` declarado como no calculable
# ─────────────────────────────────────────────────────────────────────────────
def test_a21_net_metrics_are_not_computable(real_report: BacktestReport) -> None:
    """A21: el bloque lo declara y `calculate_metrics` **rechaza** la corrida real."""
    net = _block(real_report, "net_metrics")
    assert net["state"] == "not_computable"
    assert "pnl_net_pct" in str(net["reason"]) and "null" in str(net["reason"])
    assert net["where"] == "cfdtrader.backtest.metrics.calculate_metrics"
    assert net["follow_up"] == ["#62", "#60"]

    run = real_report.outcomes[BASELINE_IDS.index("always_long")].run
    assert run.traded == 500
    with pytest.raises(MetricsInputError) as error:
        calculate_metrics(run)
    assert "pnl_net_pct" in str(error.value)

    assert not (FORBIDDEN_METRIC_KEYS & _keys(real_report.payload))


# ─────────────────────────────────────────────────────────────────────────────
# A22 · Identidades de conservacion
# ─────────────────────────────────────────────────────────────────────────────
def test_a22_conservation_identities(real_report: BacktestReport) -> None:
    """A22: ninguna sesion del universo desaparece sin aparecer en un recuento."""
    n_inputs = len(real_report.universe.inputs)
    assert n_inputs == 2687
    not_in_any_test: set[object] = set()
    for row in _rows(real_report):
        traded = cast("int", row["traded"])
        no_trade = cast("int", row["no_trade"])
        skipped = cast("int", row["skipped"])
        assert row["n_sessions"] == traded + no_trade + skipped
        assert row["n_sessions"] == row["n_test"] == 500
        conservation = _mapping(row["conservation"])
        assert conservation["holds"] is True
        assert conservation["n_inputs"] == n_inputs
        assert cast("int", row["n_test"]) + cast("int", row["not_in_any_test"]) == n_inputs
        not_in_any_test.add(row["not_in_any_test"])
    assert not_in_any_test == {2187}


# ─────────────────────────────────────────────────────────────────────────────
# A23 · Toda operacion sale por el cierre de la sesion
# ─────────────────────────────────────────────────────────────────────────────
def test_a23_every_exit_is_the_session_close(real_report: BacktestReport) -> None:
    """A23: el camino intradia es inerte sin barreras declaradas (eso es #27)."""
    for session in _traded(real_report):
        assert session.exit_reason == "session_close"
        assert session.exit_session == session.entry_session

    universe = _block(real_report, "universe")
    assert universe["intraday_path_inert"] is True
    assert "stop_px" in str(universe["intraday_path_note"])
    assert "target_px" in str(universe["intraday_path_note"])

    for row in _rows(real_report):
        counts = _mapping(row["exit_reason_counts"])
        assert counts["session_close"] == row["traded"]
        assert counts["target"] == 0
        assert counts["stop"] == 0
        assert row["exit_session_equals_entry"] is True


# ─────────────────────────────────────────────────────────────────────────────
# A24 · Serializacion canonica y hash
# ─────────────────────────────────────────────────────────────────────────────
def test_a24_canonical_text_and_hash(real_report: BacktestReport) -> None:
    """A24: el hash es el de `canonical_text` y el JSON no lleva `nan` ni `inf`."""
    payload = real_report.payload
    assert "report_sha256" not in payload
    expected = hashlib.sha256(canonical_text(payload).encode("utf-8")).hexdigest()
    assert real_report.report_sha256 == expected
    assert payload["hash_format"] == REPORT_HASH_FORMAT
    assert "canonical_text" in REPORT_HASH_FORMAT
    assert "Decimal" in REPORT_HASH_FORMAT

    text = real_report.json_text()
    assert json.loads(text)["report_sha256"] == real_report.report_sha256
    for token in ("NaN", "Infinity"):
        assert token not in text

    # `Decimal` como cadena decimal exacta y `float` vía `repr`, sin notacion cientifica.
    assert _block(real_report, "baselines")["notional_usd"] == "10000"
    assert _block(real_report, "limits", "slippage")["pct_of_r"] == "0.2"
    model = _block(real_report, "cost_model")
    assert isinstance(model["spread_entry_pct"], str)
    assert "e-" not in json.dumps(model, ensure_ascii=False)

    # El serializador rechaza `nan`/`inf` y cualquier tipo que no sea JSON: la unica via
    # admitida para un valor no medido es `null`, nunca un numero inventado.
    serializer = backtest_report._jsonable  # pyright: ignore[reportPrivateUsage]
    assert serializer(Decimal("0.20"), where="prueba") == "0.20"
    assert serializer(None, where="prueba") is None
    assert serializer([1, (2, "3")], where="prueba") == [1, [2, "3"]]
    for poisoned in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(BacktestReportError):
            serializer(poisoned, where="prueba")
    with pytest.raises(BacktestReportError):
        serializer(object(), where="prueba")


# ─────────────────────────────────────────────────────────────────────────────
# A25 · Determinismo entre procesos y segunda pasada
# ─────────────────────────────────────────────────────────────────────────────
def test_a25_determinism_across_processes_and_second_pass(
    tmp_path: Path, synthetic_root: Path
) -> None:
    """A25: mismo hash y ficheros identicos con `PYTHONHASHSEED` distinto, y al reejecutar."""
    stem = f"{REPORT_PREFIX}_{NOW.date().isoformat()}"

    def _run(directory: Path, hash_seed: str) -> tuple[str, str, str]:
        completed = subprocess.run(  # noqa: S603 - comando fijo (el interprete de la sesion)
            [
                sys.executable,
                "-m",
                "cfdtrader.analysis.backtest_report",
                "--data-root",
                str(synthetic_root),
                "--reports-dir",
                str(directory),
                "--as-of",
                NOW.isoformat(),
            ],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        assert completed.returncode == 0, completed.stderr
        json_path = directory / f"{stem}.json"
        markdown_path = directory / f"{stem}.md"
        digest = json.loads(json_path.read_text(encoding="utf-8"))["report_sha256"]
        return (
            digest,
            hashlib.sha256(json_path.read_bytes()).hexdigest(),
            hashlib.sha256(markdown_path.read_bytes()).hexdigest(),
        )

    first = _run(tmp_path / "r0", "0")
    second = _run(tmp_path / "r1", "1")
    third = _run(tmp_path / "rrandom", "random")
    assert first == second == third
    # Segunda pasada sobre el mismo almacen: el hash no cambia.
    assert _run(tmp_path / "r0", "0") == first


# ─────────────────────────────────────────────────────────────────────────────
# A26 · Sin red, sin scheduler y sin LLM
# ─────────────────────────────────────────────────────────────────────────────
def test_a26_no_network_scheduler_or_llm(real_report: BacktestReport) -> None:
    """A26: el modulo no importa clientes de red ni orquestacion, y lo declara."""
    forbidden = (
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
    )
    for token in forbidden:
        assert f"import {token}" not in SOURCE
        assert f"from {token}" not in SOURCE

    assert real_report.payload["llm_overlay"] == "disabled"
    assert real_report.payload["scheduler"] == "none"
    assert _block(real_report, "limits")["llm_overlay"] == "disabled"
    assert _block(real_report, "limits")["scheduler"] == "none"


# ─────────────────────────────────────────────────────────────────────────────
# A27 · Solo lectura de `raw`
# ─────────────────────────────────────────────────────────────────────────────
def test_a27_raw_is_read_only_and_write_false_is_silent(
    tmp_path: Path, synthetic_root: Path, synthetic_report: BacktestReport
) -> None:
    """A27: `analyse(write=True)` no toca `raw` ni `derived.labels`, y `write=False` no escribe."""
    raw_before = _fingerprint(REAL_DATA / "raw")
    labels_before = _fingerprint(REAL_DATA / "derived" / "labels")
    reports = tmp_path / "reports"
    report = analyse(store=Store(REAL_DATA), reports_dir=reports, as_of=NOW)
    assert _fingerprint(REAL_DATA / "raw") == raw_before
    assert _fingerprint(REAL_DATA / "derived" / "labels") == labels_before
    assert sorted(path.name for path in reports.iterdir()) == [
        f"{report.report_stem}.json",
        f"{report.report_stem}.md",
    ]

    # La corrida del fixture se hizo con `write=False`: su directorio no existe.
    assert synthetic_report.report_sha256
    assert not (synthetic_root / "reports").exists()


# ─────────────────────────────────────────────────────────────────────────────
# A28 · Bloque de limites declarados
# ─────────────────────────────────────────────────────────────────────────────
def test_a28_declared_limits_block(real_report: BacktestReport) -> None:
    """A28: los estados viajan sin fusionar y el veredicto no se edulcora."""
    limits = _block(real_report, "limits")
    assert limits["gate"] == "fail"
    assert limits["phase1_ready"] is False
    assert limits["is_validation"] is False
    assert "no son una validacion de la estrategia" in str(limits["statement"])
    assert _mapping(limits["costs"])["state"] == "declared_not_measured"
    assert limits["slippage"] == {
        "state": "assumed",
        "is_measurement": False,
        "pct_of_r": "0.2",
        "r_pct": None,
        "issue": "#62",
    }
    assert limits["prices"] == {
        "series_id": "^GSPC",
        "proxy_of": PRICE_PROXY_OF,
        "is_proxy": True,
        "issue": "#50",
    }
    assert PRICE_PROXY_OF == "SPX500:CFD"
    assert limits["financing_cut"] is None
    assert limits["financing_cut_verified"] is False
    assert limits["financing_cut_issue"] == "#59"
    assert limits["net_metrics_state"] == "not_computable"
    assert real_report.payload["gate"] == "fail"
    assert real_report.payload["phase1_ready"] is False
    assert real_report.payload["is_validation"] is False


# ─────────────────────────────────────────────────────────────────────────────
# A29 · Ninguna hora de corte asumida ni duracion inventada
# ─────────────────────────────────────────────────────────────────────────────
def test_a29_no_assumed_cut_or_session_length(synthetic_report: BacktestReport) -> None:
    """A29: la ventana sale del `MarketCalendar`; ninguna hora ET literal en el fuente."""
    for token in ("16:00", "09:30", "13:00", "9:30", "16:0"):
        assert token not in SOURCE
    assert "financing_cut=None" in SOURCE
    limits = _block(synthetic_report, "limits")
    assert limits["financing_cut"] is None
    assert limits["financing_cut_verified"] is False
    assert limits["financing_cut_issue"] == "#59"

    # Una media sesion es una sesion mas: no hay trato especial ni duracion inventada.
    half = date(2024, 11, 29)  # viernes despues de Accion de Gracias
    calendar = load_calendar(years=(2024, 2025, 2026))
    assert calendar.is_half_day(half) is True
    assert half in SYNTHETIC_SESSIONS
    keys = [item.session for item in synthetic_report.universe.inputs]
    assert keys.count(half) == 1
    halved = [
        item for item in synthetic_report.universe.inputs if calendar.is_half_day(item.session)
    ]
    assert halved
    assert all(item.open_px is not None and item.close_px is not None for item in halved)


# ─────────────────────────────────────────────────────────────────────────────
# A30 · Errores tipados y contrato de #12/#13/#14 sin envolver
# ─────────────────────────────────────────────────────────────────────────────
def test_a30_typed_errors_and_unwrapped_contract_errors(
    tmp_path: Path, real_report: BacktestReport
) -> None:
    """A30: errores propios donde toca y los de #12/#13/#14 tal cual llegaron."""
    empty = Store(tmp_path / "no_daily")
    empty.append("derived", "labels", _labels_records(SYNTHETIC_SESSIONS[:20]))
    with pytest.raises(MissingDatasetError):
        load_history(empty)

    without_labels = _write_store(
        tmp_path / "no_labels", sessions=SYNTHETIC_SESSIONS[:20], with_labels=False
    )
    with pytest.raises(MissingDatasetError):
        load_history(without_labels)

    small = _write_store(tmp_path / "small", sessions=SYNTHETIC_SESSIONS[:20])
    with pytest.raises(InsufficientSampleError):
        build_inputs(load_history(small), calendar=load_calendar(years=(2024,)))

    bare = SessionInput(
        session=date(2024, 1, 5), open_px=None, high_px=None, low_px=None, close_px=None
    )
    with pytest.raises(MissingPriceError):
        build_split_plan((bare,))

    with pytest.raises(LabelHorizonError):
        build_split_plan(real_report.universe.inputs, params=PlanParams(label_horizon=2))

    # Falta `--as-of` al escribir: codigo 2 y ni un fichero (A3).
    reports = tmp_path / "cli"
    assert main(["--data-root", str(tmp_path / "small"), "--reports-dir", str(reports)]) == 2
    assert not reports.exists()
    # Con `--as-of` valido pero una muestra insuficiente: mismo codigo, sin escribir.
    assert (
        main(
            [
                "--data-root",
                str(tmp_path / "small"),
                "--reports-dir",
                str(reports),
                "--as-of",
                NOW.isoformat(),
            ]
        )
        == 2
    )
    assert not reports.exists()

    # Un dataset que existe pero sin la serie pedida: recuento cero, error tipado.
    other_series = Store(tmp_path / "other_series")
    other_series.append(
        "raw",
        "market_daily",
        [{**record, "series_id": "SPY"} for record in _daily_records(SYNTHETIC_SESSIONS[:20])],
    )
    with pytest.raises(MissingDatasetError):
        load_history(other_series)

    other_labels = Store(tmp_path / "other_labels")
    other_labels.append("raw", "market_daily", _daily_records(SYNTHETIC_SESSIONS[:20]))
    other_labels.append(
        "derived",
        "labels",
        [{**record, "series_id": "SPY"} for record in _labels_records(SYNTHETIC_SESSIONS[:20])],
    )
    with pytest.raises(MissingDatasetError):
        load_history(other_labels)

    # Un plan imposible por tamaño: el `InsufficientSessionsError` de #12 sale tal cual.
    ten = tuple(
        SessionInput(session=day, open_px=1.0, high_px=1.1, low_px=0.9, close_px=1.0)
        for day in SYNTHETIC_SESSIONS[:10]
    )
    with pytest.raises(InsufficientSessionsError):
        build_split_plan(ten)

    # Un baseline desconocido: el `BaselinesError` de #14, sin envolver.
    with pytest.raises(InvalidBaselineParameterError):
        run_all_baselines(
            real_report.universe.inputs,
            split_plan=real_report.split_plan,
            cost_model=real_report.cost_model,
            slippage=real_report.slippage,
            baselines=("desconocido",),
        )

    # Un `SplitPlan` imposible: el `EngineInputError` de #13, sin envolver.
    impossible = dataclasses.replace(real_report.split_plan, n_sessions=1)
    with pytest.raises(EngineInputError):
        run_all_baselines(
            real_report.universe.inputs,
            split_plan=impossible,
            cost_model=real_report.cost_model,
            slippage=real_report.slippage,
            baselines=("no_trade",),
        )


# ─────────────────────────────────────────────────────────────────────────────
# A31 · Un test por criterio y sin escribir en `data/`
# ─────────────────────────────────────────────────────────────────────────────
def test_a31_one_test_per_criterion(tmp_path: Path) -> None:
    """A31: los 33 tests existen y la sesion usa `tmp_path` y la fixture guardiana."""
    missing = [
        index
        for index in range(1, 34)
        if not re.search(rf"^def test_a{index}_", TEST_SOURCE, re.MULTILINE)
    ]
    assert missing == []
    assert "tmp_path" in TEST_SOURCE
    conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "_repository_data_is_untouched" in conftest
    assert "fingerprint" in conftest
    assert _fingerprint(tmp_path) == {}


# ─────────────────────────────────────────────────────────────────────────────
# A32 · Puertas y cobertura
# ─────────────────────────────────────────────────────────────────────────────
def test_a32_public_api_is_exercised_and_no_waived_coverage() -> None:
    """A32: cada nombre publico se ejercita y la unica exencion de cobertura es la entrada."""
    missing = [name for name in backtest_report.__all__ if name not in TEST_SOURCE]
    assert missing == []
    pragmas = [line for line in SOURCE.splitlines() if "pragma: no cover" in line]
    assert len(pragmas) == 1
    assert "__main__" in pragmas[0]


# ─────────────────────────────────────────────────────────────────────────────
# A33 · Fronteras legibles por maquina
# ─────────────────────────────────────────────────────────────────────────────
def test_a33_machine_readable_boundaries() -> None:
    """A33: las dos tuplas de diccionarios, cada una con su issue."""
    for boundary in (ADAPTER_DOES_NOT_DO, FOLLOW_UPS):
        assert isinstance(boundary, tuple)
        assert boundary
        for item in boundary:
            assert isinstance(item, dict)
            assert str(item["issue"]).startswith("#")
    assert ADAPTER_DOES_NOT_DO != BASELINES_DOES_NOT_DO

    covered = {item["issue"] for item in (*ADAPTER_DOES_NOT_DO, *FOLLOW_UPS)}
    assert {
        "#14",
        "#15",
        "#18",
        "#27",
        "#50",
        "#59",
        "#60",
        "#62",
        "#63",
        "#66",
        "#67",
        "#68",
        "#70",
    } <= covered
    assert "#67/#68/#69" in covered
