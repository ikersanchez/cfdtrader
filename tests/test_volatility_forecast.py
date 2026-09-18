"""Tests del estudio de volatilidad (tarea #7).

Datos sintéticos con estructura conocida y un almacén en ``tmp_path`` (A26: nada
del ``data/`` del repositorio). Los tests comprueban lo que de verdad se puede
romper: que el corte de la muestra limpia es **el mismo** que el del estudio del
drift (A2), que un candidato que no se puede estimar no aborta el estudio (A13),
que la regla de selección se aplica tal cual (A16) y que dos ejecuciones dan el
mismo JSON (A22).
"""

from __future__ import annotations

import itertools
import json
import math
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from cfdtrader.analysis.drift import decompose, load_sessions, session_stale_open
from cfdtrader.analysis.volatility_forecast import (
    CANDIDATES,
    MIN_TRAIN,
    REFIT_EVERY,
    CandidateResult,
    Metrics,
    Target,
    Verdict,
    analyse,
    build_sample,
    load_market,
    main,
    report_payload,
    scale_sigma_for_duration,
    select_candidate,
    walk_forward,
)
from cfdtrader.data.calendar import FULL_SESSION_HOURS, HALF_SESSION_HOURS, MarketCalendar
from cfdtrader.data.store import Store

#: Instante de referencia del estudio (fijo ⇒ determinismo comprobable).
NOW = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)

SERIES_ID = "^GSPC"
VIX_SERIES_ID = "^VIX"

#: Primer día laborable de la serie sintética.
START = date(2005, 1, 3)

#: Sesiones sintéticas: las suficientes para que la muestra limpia pase de 500.
SESSIONS = 1200

#: Último año con el `open` repetido del cierre anterior (el artefacto de #52).
STALE_THROUGH = 2006

#: Año desde el que la regla de #52 declara limpia la muestra.
EXPECTED_CUTOFF = "2007-01-01"

EASTERN = ZoneInfo("America/New_York")


# ─────────────────────────────────────────────────────────────────────────────
# Almacén sintético
# ─────────────────────────────────────────────────────────────────────────────
def _weekdays(count: int, start: date = START) -> list[date]:
    """Los primeros ``count`` días laborables desde ``start``."""
    days: list[date] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _session(frame: pl.DataFrame, index: int) -> str:
    """Sesión de esa posición del frame, en ISO."""
    return str(frame.get_column("session").to_list()[index])


def _last_value(frame: pl.DataFrame, column: str) -> float:
    """Último valor de esa columna, como ``float``."""
    value: Any = frame.get_column(column).to_list()[-1]
    assert isinstance(value, float)
    return value


def _dict_of(value: object) -> dict[str, Any]:
    """Diccionario a partir de un valor del payload JSON, para tipar en estricto."""
    assert isinstance(value, dict)
    return cast("dict[str, Any]", value)


def _records(count: int, *, seed: int = 11) -> list[dict[str, object]]:
    """Sesiones con volatilidad variable y el artefacto #52 en los primeros años.

    - En los años ``<= STALE_THROUGH`` el ``open`` es **exactamente** el cierre
      anterior (el artefacto de #52: el tramo nocturno es cero por construcción).
    - Desde el corte, un 2 % de las sesiones siguen siendo *stale* (por debajo de
      la tolerancia del 5 %, así que el corte no cambia) y tres sesiones tienen
      ``high``/``low`` nulos, para poder comprobar las tres exclusiones de A3.
    """
    rng = np.random.default_rng(seed)
    days = _weekdays(count)
    log_vol = np.zeros(count)
    for index in range(1, count):
        log_vol[index] = 0.93 * log_vol[index - 1] + rng.normal(0.0, 0.11)
    vol = 0.009 * np.exp(log_vol)
    step = rng.normal(0.0, 1.0, count) * vol
    close = 3000.0 * np.cumprod(1.0 + step)
    fetched_at = datetime.combine(days[-1], time(23, 0), tzinfo=UTC)

    stale_later = {int(index) for index in rng.choice(count, size=count // 50, replace=False)}
    null_ohlc = {count // 2, count // 2 + 7, count - 30}
    records: list[dict[str, object]] = []
    previous_close: float | None = None
    for index, day in enumerate(days):
        close_value = float(close[index])
        if day.year <= STALE_THROUGH or index in stale_later:
            open_price = close_value if previous_close is None else previous_close
        else:
            open_price = close_value * (1.0 + rng.normal(0.0, 0.004))
        high = max(open_price, close_value) * (1.0 + abs(float(step[index])) * 0.5)
        low = min(open_price, close_value) * (1.0 - abs(float(step[index])) * 0.5)
        records.append(
            {
                "source": "yfinance",
                "series_id": SERIES_ID,
                "as_of": datetime.combine(day, time(21, 0), tzinfo=UTC),
                "fetched_at": fetched_at,
                "published_at": None,
                "open": float(open_price),
                "high": None if index in null_ohlc else float(high),
                "low": None if index in null_ohlc else float(low),
                "close": close_value,
                "volume": 1_000_000.0,
                "adj_close": close_value,
            }
        )
        previous_close = close_value
    return records


def _vix_records(count: int, *, seed: int = 21) -> list[dict[str, object]]:
    """Cierres sintéticos del VIX, correlacionados con la volatilidad del índice."""
    rng = np.random.default_rng(seed)
    days = _weekdays(count)
    level = 17.0 + np.cumsum(rng.normal(0.0, 0.4, count))
    level = np.clip(level, 9.0, 60.0)
    return [
        {
            "source": "yfinance",
            "series_id": VIX_SERIES_ID,
            "as_of": datetime.combine(day, time(21, 0), tzinfo=UTC),
            "fetched_at": datetime.combine(days[-1], time(23, 0), tzinfo=UTC),
            "published_at": None,
            "open": float(level[index]),
            "high": float(level[index] * 1.02),
            "low": float(level[index] * 0.98),
            "close": float(level[index]),
            "volume": 0.0,
            "adj_close": float(level[index]),
        }
        for index, day in enumerate(days)
    ]


def _store(
    root: Path,
    *,
    sessions: int = SESSIONS,
    with_vix: bool = True,
    staleness: bool = True,
) -> Store:
    """Almacén temporal con las sesiones sintéticas ya escritas."""
    records = _records(sessions)
    if not staleness:
        # Sin artefacto: el corte tiene que ser el primer año de la serie.
        records = [_fresh(record) for record in records]
    store = Store(root)
    store.append("raw", "market_daily", records)
    if with_vix:
        store.append("raw", "market_daily", _vix_records(sessions))
    return store


def _fresh(record: dict[str, object]) -> dict[str, object]:
    """Registro con el ``open`` desplazado del cierre anterior (sin artefacto #52)."""
    close = record["close"]
    assert isinstance(close, float)
    record = dict(record)
    record["open"] = close * 0.999
    return record


# ─────────────────────────────────────────────────────────────────────────────
# A2 — Una sola definición de la muestra limpia
# ─────────────────────────────────────────────────────────────────────────────
def test_the_clean_sample_cutoff_is_the_same_as_the_drift_study(tmp_path: Path) -> None:
    """A2: sobre el mismo almacén, drift y volatilidad calculan **el mismo** corte."""
    store = _store(tmp_path)
    drift_study = decompose(
        load_sessions(store, series_id=SERIES_ID),
        series_id=SERIES_ID,
        source="yfinance",
        as_of=NOW,
    )
    study = analyse(data_root=tmp_path, now=NOW, reports_dir=tmp_path / "derived" / "reports")

    assert drift_study.clean_from == EXPECTED_CUTOFF
    assert study.clean_from == drift_study.clean_from
    assert study.clean_sessions == drift_study.clean_sessions


def test_the_report_declares_the_stale_open_share_year_by_year(tmp_path: Path) -> None:
    """A2: por año, la proporción de `open` repetido y qué años son limpios."""
    _store(tmp_path)
    study = analyse(data_root=tmp_path, now=NOW)

    assert study.by_year, "el informe debe declarar la proporción por año"
    first = study.by_year[0]
    assert first["year"] == START.year
    assert float(str(first["stale_open_share"])) > 0.05
    assert first["clean"] is False
    clean_years = [row for row in study.by_year if row["clean"]]
    assert clean_years and clean_years[0]["year"] == int(EXPECTED_CUTOFF[:4])
    assert all(float(str(row["stale_open_share"])) <= 0.05 for row in clean_years)


# ─────────────────────────────────────────────────────────────────────────────
# A3 — Exclusiones declaradas, con motivo
# ─────────────────────────────────────────────────────────────────────────────
def test_every_exclusion_is_declared_with_a_reason(tmp_path: Path) -> None:
    """A3: `open` repetido, nulos y medias sesiones se excluyen y se declaran."""
    _store(tmp_path)
    study = analyse(data_root=tmp_path, now=NOW)

    assert study.exclusions["stale_open"] > 0
    assert study.exclusions["null_ohlc"] == 3
    assert study.exclusions["half_day"] > 0
    assert study.exclusions["total"] == sum(
        study.exclusions[key] for key in ("stale_open", "null_ohlc", "half_day")
    )
    assert set(study.exclusion_reasons) == {"stale_open", "null_ohlc", "half_day"}
    assert all(reason for reason in study.exclusion_reasons.values())


def test_the_excluded_sessions_never_reach_the_evaluated_window(tmp_path: Path) -> None:
    """Las sesiones excluidas no están en la muestra de trabajo: el recuento cuadra."""
    _store(tmp_path)
    frame, summary = build_sample(Store(tmp_path))
    exclusions = cast("dict[str, int]", summary["exclusions"])

    assert int(str(summary["evaluated_sessions"])) == frame.height
    assert frame.height == (
        int(str(summary["clean_sessions"])) - exclusions["null_ohlc"] - exclusions["half_day"]
    )
    calendar = MarketCalendar()
    assert not any(calendar.is_half_day(day) for day in frame.get_column("session").to_list())


# ─────────────────────────────────────────────────────────────────────────────
# A12/A17 — Esquema pre-registrado y bloques legibles por máquina
# ─────────────────────────────────────────────────────────────────────────────
def test_the_walk_forward_scheme_is_the_pre_registered_one(tmp_path: Path) -> None:
    """A12: ventana expansiva, min_train 500, reajuste cada 21 para TODOS."""
    _store(tmp_path)
    frame, _summary = build_sample(Store(tmp_path))
    study = analyse(data_root=tmp_path, now=NOW)

    assert len(study.folds) >= 2
    for index, fold in enumerate(study.folds[:-1]):
        assert fold.index == index
        assert fold.sessions == REFIT_EVERY
        assert fold.start <= fold.end
    assert 1 <= study.folds[-1].sessions <= REFIT_EVERY
    assert study.evaluated_sessions > MIN_TRAIN
    # La primera sesión evaluada es la que ocupa la posición `min_train`:
    # la ventana de entrenamiento es exactamente la declarada, ni una más.
    assert study.first_evaluated == _session(frame, MIN_TRAIN)
    assert study.last_evaluated == _session(frame, -1)


def test_the_json_carries_the_machine_readable_blocks(tmp_path: Path) -> None:
    """A17: `selection`, `candidates` y `folds`, para que #23 no lea prosa."""
    _store(tmp_path)
    study = analyse(data_root=tmp_path, now=NOW, reports_dir=tmp_path / "derived" / "reports")
    payload = report_payload(study)

    selection = _dict_of(payload["selection"])
    assert set(selection) >= {"selected", "verdict", "metric", "rule", "constants"}
    assert selection["metric"] == "qlike"

    candidates = _dict_of(payload["candidates"])
    assert set(candidates) == set(CANDIDATES)
    for block in candidates.values():
        candidate = _dict_of(block)
        assert candidate["status"] in {"ok", "unavailable"}
        for target in (Target.PARKINSON.value, Target.RET_SQ.value):
            metrics = candidate[target]
            if metrics is not None:
                assert set(_dict_of(metrics)) == {"qlike", "mse_log", "sessions"}

    folds = cast("list[Any]", payload["folds"])
    assert folds
    assert set(_dict_of(folds[0])) >= {"index", "start", "end", "sessions", *CANDIDATES}

    path = tmp_path / "derived" / "reports" / f"volatility_forecast_{NOW.date()}.json"
    assert path.exists()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["selection"]["verdict"] == study.selection.verdict


# ─────────────────────────────────────────────────────────────────────────────
# A13 — Un candidato que no se puede estimar no aborta el estudio
# ─────────────────────────────────────────────────────────────────────────────
def test_a_store_without_vix_declares_har_vix_unavailable_and_still_reports(
    tmp_path: Path,
) -> None:
    """A13: sin `^VIX`, `har_vix` es `unavailable` **con el motivo** y el informe sale."""
    _store(tmp_path, with_vix=False)
    reports = tmp_path / "derived" / "reports"
    study = analyse(data_root=tmp_path, now=NOW, reports_dir=reports)

    har_vix = study.candidate("har_vix")
    assert har_vix.status == "unavailable"
    assert har_vix.reason is not None and VIX_SERIES_ID in har_vix.reason
    assert har_vix.metrics[Target.PARKINSON.value] is None
    assert any(result.status == "ok" for result in study.candidates)
    assert (reports / f"volatility_forecast_{NOW.date()}.md").exists()


# ─────────────────────────────────────────────────────────────────────────────
# A14/A15 — Métricas por candidato y por fold, con los dos objetivos
# ─────────────────────────────────────────────────────────────────────────────
def test_both_targets_are_evaluated_and_the_rankings_are_compared(tmp_path: Path) -> None:
    """A15: las dos tablas se calculan y su correlación de rangos se declara."""
    _store(tmp_path)
    study = analyse(data_root=tmp_path, now=NOW)

    for result in study.candidates:
        if result.status == "ok":
            for target in (Target.PARKINSON.value, Target.RET_SQ.value):
                metrics = result.metrics[target]
                assert metrics is not None and math.isfinite(metrics.qlike)
                assert metrics.sessions > 0
        else:
            assert result.reason

    for fold in study.folds:
        for name in CANDIDATES:
            assert set(fold.metrics[name]) == {Target.PARKINSON.value, Target.RET_SQ.value}

    assert isinstance(study.rank_correlation_note, str) and study.rank_correlation_note
    targets = _dict_of(report_payload(study)["targets"])
    assert _dict_of(targets[Target.PARKINSON.value])["immune_to_stale_open_52"] is True
    assert _dict_of(targets[Target.RET_SQ.value])["immune_to_stale_open_52"] is False
    assert "ln(C_t / O_t)" in str(_dict_of(targets[Target.RET_SQ.value])["formula"])


# ─────────────────────────────────────────────────────────────────────────────
# A16 — La regla de selección, aplicada tal cual
# ─────────────────────────────────────────────────────────────────────────────
def _result(name: str, *, primary: float, secondary: float, status: str = "ok") -> CandidateResult:
    """Candidato sintético con las dos métricas puestas a mano."""
    metrics: dict[str, Metrics | None] = {
        Target.PARKINSON.value: Metrics(qlike=primary, mse_log=0.0, sessions=100),
        Target.RET_SQ.value: Metrics(qlike=secondary, mse_log=0.0, sessions=100),
    }
    if status != "ok":
        metrics = {Target.PARKINSON.value: None, Target.RET_SQ.value: None}
    return CandidateResult(
        name=name,
        status=status,
        reason=None if status == "ok" else "no estimable",
        metrics=metrics,
        median_sigma_bp=1.0,
    )


def test_the_selection_rule_picks_a_candidate_that_beats_rw_in_both_targets() -> None:
    """A16: mejor o dentro del 2 % en los dos objetivos **y** >2 % mejor que `rw`."""
    selection = select_candidate(
        [
            _result("rw", primary=1.0, secondary=1.0),
            _result("har", primary=0.80, secondary=0.80),
            _result("garch", primary=0.99, secondary=0.99),
        ]
    )

    assert selection.verdict == Verdict.SELECTED.value
    assert selection.selected == "har"
    arithmetic = {row["candidate"]: row for row in selection.arithmetic}
    assert arithmetic["har"]["eligible"] is True
    assert arithmetic["garch"]["eligible"] is False  # no bate a rw por más del 2 %


def test_the_selection_rule_falls_back_to_rw_when_nobody_beats_it() -> None:
    """A16: si nadie bate a `rw`, el veredicto es `no_better_than_naive`."""
    selection = select_candidate(
        [
            _result("rw", primary=1.0, secondary=1.0),
            _result("har", primary=1.005, secondary=1.005),
        ]
    )

    assert selection.verdict == Verdict.NO_BETTER_THAN_NAIVE.value
    assert selection.selected is None
    assert selection.constants["fallback"] == "rw"


def test_the_selection_rule_is_inconclusive_when_the_targets_contradict() -> None:
    """A16: si cada objetivo señala a un candidato distinto, `inconclusive`."""
    selection = select_candidate(
        [
            _result("rw", primary=1.0, secondary=1.0),
            _result("har", primary=0.80, secondary=1.10),
            _result("garch", primary=1.10, secondary=0.80),
        ]
    )

    assert selection.verdict == Verdict.INCONCLUSIVE.value
    assert selection.selected is None


def test_an_unavailable_candidate_is_never_selected() -> None:
    """Un candidato `unavailable` no entra en la regla, aunque tenga métricas nulas."""
    selection = select_candidate(
        [
            _result("rw", primary=1.0, secondary=1.0),
            _result("har", primary=0.80, secondary=0.80),
            _result("har_vix", primary=0.1, secondary=0.1, status="unavailable"),
        ]
    )

    assert selection.selected == "har"


# ─────────────────────────────────────────────────────────────────────────────
# A19 — DST: la clave de sesión se deriva en America/New_York
# ─────────────────────────────────────────────────────────────────────────────
def test_sessions_across_the_march_dst_change_are_contiguous_and_unique() -> None:
    """A19: con el instante UTC desplazado 1 h, las sesiones siguen contiguas.

    Antes del cambio de hora el cierre de las 16:00 ET es a las 21:00 UTC; después,
    a las 20:00 UTC. La clave de sesión derivada en ``America/New_York`` es la misma
    fecha de sesión en los dos tramos, así que la secuencia es contigua y no tiene
    duplicados: nada de hora fija.
    """
    winter = [date(2023, 3, day) for day in (6, 7, 8, 9, 10)]
    summer = [date(2023, 3, day) for day in (13, 14, 15)]
    instants = [
        *(datetime.combine(day, time(21, 0), tzinfo=UTC) for day in winter),
        *(datetime.combine(day, time(20, 0), tzinfo=UTC) for day in summer),
    ]
    frame = pl.DataFrame(
        {
            "as_of": instants,
            "open": [100.0 + index for index in range(len(instants))],
            "close": [101.0 + index for index in range(len(instants))],
        }
    )
    sessions = session_stale_open(frame).get_column("session").to_list()

    assert sessions == [*winter, *summer]
    assert len(set(sessions)) == len(sessions)
    assert all((later - earlier).days in {1, 3} for earlier, later in itertools.pairwise(sessions))
    # El instante UTC sí se desplaza una hora, y la hora local ET no: eso es el DST.
    assert {instant.hour for instant in instants} == {20, 21}
    assert {instant.astimezone(EASTERN).hour for instant in instants} == {16}


# ─────────────────────────────────────────────────────────────────────────────
# A20 — Medias sesiones: se excluyen al evaluar y se escalan al servir
# ─────────────────────────────────────────────────────────────────────────────
def test_half_days_are_detected_with_the_calendar_and_scaled_at_serving_time(
    tmp_path: Path,
) -> None:
    """A20: medias sesiones fuera del ajuste; el API de servicio escala por √(3,5/6,5)."""
    _store(tmp_path)
    store = Store(tmp_path)
    frame, summary = build_sample(store)
    study = analyse(data_root=tmp_path, now=NOW)

    assert study.exclusions["half_day"] > 0
    calendar = MarketCalendar()
    # Había medias sesiones en el almacén y **ninguna** ha llegado a la muestra de trabajo.
    base = session_stale_open(load_market(store)).filter(pl.col("prev_close").is_not_null())
    assert any(calendar.is_half_day(day) for day in base.get_column("session").to_list())
    assert not any(calendar.is_half_day(day) for day in frame.get_column("session").to_list())
    assert cast("dict[str, int]", summary["exclusions"])["half_day"] == study.exclusions["half_day"]

    full = scale_sigma_for_duration(0.01)
    half = scale_sigma_for_duration(0.01, duration_hours=HALF_SESSION_HOURS)
    assert full == pytest.approx(0.01)
    assert half == pytest.approx(0.01 * math.sqrt(3.5 / 6.5))
    assert scale_sigma_for_duration(0.01, duration_hours=FULL_SESSION_HOURS) == pytest.approx(0.01)
    with pytest.raises(ValueError, match="positiva"):
        scale_sigma_for_duration(0.01, duration_hours=0.0)


def test_half_days_are_post_thanksgiving_or_christmas_eve(tmp_path: Path) -> None:
    """Las medias sesiones salen del calendario de #4, no de una fecha a mano."""
    calendar = MarketCalendar()
    assert calendar.is_half_day(date(2020, 11, 27)) is True  # día después de Acción de Gracias
    assert calendar.is_half_day(date(2020, 12, 24)) is True  # Nochebuena
    assert calendar.is_half_day(date(2021, 3, 15)) is False


# ─────────────────────────────────────────────────────────────────────────────
# A21 — Anclaje para #10
# ─────────────────────────────────────────────────────────────────────────────
def test_the_report_publishes_the_anchors_for_the_barriers(tmp_path: Path) -> None:
    """A21: mediana del forecast (bp/sesión) junto a la mediana de `|open→close|` (bp)."""
    _store(tmp_path)
    study = analyse(data_root=tmp_path, now=NOW)
    anchors = study.anchors

    assert anchors.median_forecast_sigma_bp > 0.0
    assert anchors.median_abs_open_close_bp > 0.0
    assert anchors.absolute_move_sessions > 0
    assert set(anchors.per_candidate_sigma_bp) == set(CANDIDATES)
    assert anchors.used_candidate in CANDIDATES
    payload = report_payload(study)["anchors"]
    assert isinstance(payload, dict)
    assert payload["used_candidate"] == anchors.used_candidate


# ─────────────────────────────────────────────────────────────────────────────
# A18 — Limitaciones declaradas
# ─────────────────────────────────────────────────────────────────────────────
def test_the_five_mandatory_limitations_are_declared(tmp_path: Path) -> None:
    """A18: se declaran al menos las cinco limitaciones exigidas."""
    _store(tmp_path)
    study = analyse(data_root=tmp_path, now=NOW)
    text = " ".join(study.limitations)

    assert len(study.limitations) >= 5
    assert "^GSPC" in text  # (1) se mide sobre el índice, no sobre el CFD
    assert "proxy" in text  # (2) el objetivo es un proxy de la varianza
    assert "#52" in text  # (3) la muestra anterior al corte está contaminada
    assert "pre-registrados" in text or "búsqueda" in text  # (4) 4 candidatos, no una búsqueda
    assert "#9" in text  # (5) esto no dice nada sobre la viabilidad


# ─────────────────────────────────────────────────────────────────────────────
# A10 — No look-ahead también en los candidatos
# ─────────────────────────────────────────────────────────────────────────────
def test_candidate_forecasts_do_not_change_with_a_later_session(tmp_path: Path) -> None:
    """A10: ningún candidato cambia el pronóstico de la sesión `t` al añadir `t+1`."""
    _store(tmp_path)
    frame, _summary = build_sample(Store(tmp_path))
    before = walk_forward(frame).forecasts

    last = frame.tail(1)
    close_last = _last_value(frame, "close")
    open_last = _last_value(frame, "open")
    absurd = last.with_columns(
        pl.lit(close_last * 1.5).alias("open"),
        pl.lit(open_last * 1.5).alias("close"),
        pl.lit(0.5).alias("parkinson_rv"),
        pl.lit(0.25).alias("ret_sq"),
        pl.lit(-10.0).alias("ret_log"),
        pl.lit(0.9).alias("har_lag1"),
        pl.lit(0.9).alias("har_lag4"),
        pl.lit(0.9).alias("har_lag17"),
        pl.lit(0.9).alias("vix_level"),
        pl.lit(3.0).alias("vix_zscore"),
        pl.lit(0.99).alias("vix_percentile"),
    ).with_columns((pl.col("session") + timedelta(days=1)).alias("session"))
    augmented = pl.concat([frame, absurd]).sort("session")
    after = walk_forward(augmented).forecasts

    for name in CANDIDATES:
        np.testing.assert_array_equal(before[name], after[name][: frame.height])


# ─────────────────────────────────────────────────────────────────────────────
# A22 — Determinismo
# ─────────────────────────────────────────────────────────────────────────────
def test_two_runs_with_the_same_now_produce_the_same_json_byte_for_byte(
    tmp_path: Path,
) -> None:
    """A22: mismo `--now` ⇒ mismo `sha256`, GARCH incluido."""
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _store(first_root)
    _store(second_root)
    analyse(data_root=first_root, now=NOW, reports_dir=first_root / "derived" / "reports")
    analyse(data_root=second_root, now=NOW, reports_dir=second_root / "derived" / "reports")

    name = f"volatility_forecast_{NOW.date()}"
    first = (first_root / "derived" / "reports" / f"{name}.json").read_bytes()
    second = (second_root / "derived" / "reports" / f"{name}.json").read_bytes()
    first_md = (first_root / "derived" / "reports" / f"{name}.md").read_bytes()
    second_md = (second_root / "derived" / "reports" / f"{name}.md").read_bytes()

    assert first == second
    assert first_md == second_md


# ─────────────────────────────────────────────────────────────────────────────
# A23 — CLI
# ─────────────────────────────────────────────────────────────────────────────
def test_the_cli_writes_the_report_with_the_expected_name(tmp_path: Path) -> None:
    """A23: `main` lee el almacén y escribe `volatility_forecast_<fecha>.{json,md}`."""
    _store(tmp_path)
    code = main(["--data-root", str(tmp_path), "--now", NOW.isoformat()])
    reports = tmp_path / "derived" / "reports"

    assert code == 0
    assert (reports / f"volatility_forecast_{NOW.date()}.json").exists()
    assert (reports / f"volatility_forecast_{NOW.date()}.md").exists()


def test_the_cli_exits_with_two_and_writes_nothing_when_nothing_can_be_estimated(
    tmp_path: Path,
) -> None:
    """A23: sin muestra para estimar ningún candidato ⇒ código 2 y ningún informe."""
    _store(tmp_path, sessions=300, staleness=False)
    reports = tmp_path / "derived" / "reports"
    code = main(["--data-root", str(tmp_path), "--now", NOW.isoformat()])

    assert code == 2
    assert not reports.exists()


def test_the_cli_fails_with_a_configuration_error_when_the_store_is_empty(
    tmp_path: Path,
) -> None:
    """Sin dataset de mercado, el estudio no se inventa datos: código 2."""
    assert main(["--data-root", str(tmp_path), "--now", NOW.isoformat()]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# A26 — El guardián del `data/` del repositorio
# ─────────────────────────────────────────────────────────────────────────────
def test_the_synthetic_store_lives_under_tmp_path(tmp_path: Path) -> None:
    """A26: el almacén del test vive en `tmp_path` y el estudio escribe solo ahí."""
    _store(tmp_path)
    reports = tmp_path / "derived" / "reports"
    analyse(data_root=tmp_path, now=NOW, reports_dir=reports)

    assert tmp_path in reports.parents
    assert list(reports.iterdir())
