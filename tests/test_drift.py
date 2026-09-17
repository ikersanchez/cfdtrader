"""Tests del estudio del drift (tarea #6).

Datos sintéticos con estructura conocida: si el drift se pone entero en el tramo
nocturno, el veredicto tiene que ser `overnight` y la puerta de la Fase 0 tiene
que decir `fail`. Es la comprobación que de verdad importa: es el estudio que
puede invalidar el planteamiento.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pytest

from cfdtrader.analysis.drift import (
    DriftSegment,
    DriftVerdict,
    analyse,
    decompose,
    load_sessions,
    render_markdown,
)
from cfdtrader.data.settings import ConfigurationError
from cfdtrader.data.store import Store

#: Instante desde el que todo lo anterior ya está "descargado".
NOW = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)

#: Cierres consecutivos (lunes a viernes) desde aquí, para los datos sintéticos.
START = date(2022, 1, 3)

SERIES_ID = "^GSPC"


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
    count: int, *, overnight_bp: float, intraday_cycle: list[float]
) -> list[dict[str, object]]:
    """Sesiones sintéticas: el tramo nocturno con ese sesgo y la sesión con ese ciclo de pb.

    El tramo nocturno lleva un ruido determinista de ±0,4 pb en vez de ser una
    constante: con varianza exactamente cero el contraste t degenera, y el mercado
    real nunca es perfectamente constante.

    El ``fetched_at`` se calcula a partir de la última sesión generada, para que el
    contrato del almacén (``fetched_at >= as_of``) se cumpla con cualquier número
    de sesiones.
    """
    previous_close = 5000.0
    records: list[dict[str, object]] = []
    days = _weekdays(count)
    fetched_at = datetime.combine(days[-1], time(23, 0), tzinfo=UTC)
    for index, day in enumerate(days):
        noise_bp = (+0.4, -0.4)[index % 2]
        open_price = previous_close * (1.0 + (overnight_bp + noise_bp) / 10_000.0)
        intraday_bp = intraday_cycle[index % len(intraday_cycle)]
        close_price = open_price * (1.0 + intraday_bp / 10_000.0)
        records.append(
            {
                "source": "yfinance",
                "series_id": SERIES_ID,
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


def _store_with(tmp_path: Path, records: list[dict[str, object]]) -> Store:
    """Almacén temporal con esas sesiones escritas en `raw.market_daily`."""
    store = Store(tmp_path)
    store.append("raw", "market_daily", records)
    return store


# ─────────────────────────────────────────────────────────────────────────────
# Descomposición aritmética
# ─────────────────────────────────────────────────────────────────────────────
def test_the_three_segments_multiply_to_the_total_return(tmp_path: Path) -> None:
    """`(1+intraday)·(1+overnight) = (1+total)`: los tres tramos son la misma cosa partida."""
    store = _store_with(
        tmp_path,
        _bars(30, overnight_bp=+5.0, intraday_cycle=[+3.0, -2.0, +1.0, -1.5]),
    )
    frame = load_sessions(store, series_id=SERIES_ID)

    intraday = frame.get_column(DriftSegment.INTRADAY.value)
    overnight = frame.get_column(DriftSegment.OVERNIGHT.value)
    total = frame.get_column(DriftSegment.TOTAL.value)
    composed = ((1.0 + intraday) * (1.0 + overnight) - 1.0 - total).abs().max()

    assert frame.height == 29  # la primera sesión no tiene tramo nocturno
    assert isinstance(composed, float) and composed < 1e-9


def test_a_purely_overnight_drift_is_declared_overnight_and_fails_the_gate(
    tmp_path: Path,
) -> None:
    """Si el retorno está todo fuera de la sesión, la estrategia intradía opera la peor parte."""
    store = _store_with(
        tmp_path,
        _bars(320, overnight_bp=+6.0, intraday_cycle=[+2.0, -2.0]),
    )
    study = decompose(
        load_sessions(store, series_id=SERIES_ID),
        series_id=SERIES_ID,
        source="yfinance",
        as_of=NOW,
    )

    assert study.verdict is DriftVerdict.OVERNIGHT
    assert study.phase0_gate == "fail"
    assert study.segment("intraday").mean_bp == pytest.approx(0.0, abs=0.5)
    assert study.segment("overnight").mean_bp == pytest.approx(6.0, abs=0.5)
    assert study.segment("overnight").significant is True


def test_a_purely_intraday_drift_is_declared_intraday_and_passes_the_gate(
    tmp_path: Path,
) -> None:
    """Si el retorno está en la sesión, el planteamiento intradía tiene sentido."""
    store = _store_with(
        tmp_path,
        _bars(320, overnight_bp=0.0, intraday_cycle=[+5.0, +4.0, +6.0]),
    )
    study = decompose(
        load_sessions(store, series_id=SERIES_ID),
        series_id=SERIES_ID,
        source="yfinance",
        as_of=NOW,
    )

    assert study.verdict is DriftVerdict.INTRADAY
    assert study.phase0_gate == "pass"
    assert study.difference.mean_difference_bp > 0
    assert study.difference.significant is True


def test_a_flat_market_is_declared_none(tmp_path: Path) -> None:
    """Sin drift en ningún tramo, el estudio lo dice: `none`, no un falso positivo."""
    store = _store_with(tmp_path, _bars(320, overnight_bp=0.0, intraday_cycle=[+0.5, -0.5]))
    study = decompose(
        load_sessions(store, series_id=SERIES_ID),
        series_id=SERIES_ID,
        source="yfinance",
        as_of=NOW,
    )

    assert study.verdict is DriftVerdict.NONE
    assert study.phase0_gate == "inconclusive"


def test_a_session_without_its_own_drift_loses_the_gate(tmp_path: Path) -> None:
    """Regla pre-registrada: sesión sin drift demostrable + tramo nocturno con drift ⇒ `fail`.

    Es el caso real medido sobre `^GSPC` (sesión `+2,00 pb`, p = 0,19; nocturno
    `+2,91 pb`, p = 0,0008): no hace falta que la sesión tenga media negativa para
    que la puerta se caiga.
    """
    store = _store_with(
        tmp_path,
        _bars(
            400,
            overnight_bp=+6.0,
            intraday_cycle=[+150.0, -148.0, +101.0, -100.0],  # media +0,75 pb, desv. ~126 pb
        ),
    )
    study = decompose(
        load_sessions(store, series_id=SERIES_ID),
        series_id=SERIES_ID,
        source="yfinance",
        as_of=NOW,
    )

    assert study.segment("intraday").mean_bp > 0
    assert study.segment("intraday").significant is False
    assert study.segment("overnight").significant is True
    assert study.verdict is DriftVerdict.OVERNIGHT
    assert study.phase0_gate == "fail"


def test_a_source_that_repeats_the_previous_close_is_detected_and_declared(
    tmp_path: Path,
) -> None:
    """Si la fuente repite el cierre anterior, el tramo nocturno es cero por construcción."""
    records = _bars(60, overnight_bp=+6.0, intraday_cycle=[+3.0, -1.0])
    # La fuente antigua de Yahoo devuelve el cierre anterior como apertura.
    stale = [
        {**record, "open": previous["close"]}
        for record, previous in zip(records[1:], records, strict=False)
    ]
    store = _store_with(tmp_path, [records[0], *stale])
    study = decompose(
        load_sessions(store, series_id=SERIES_ID),
        series_id=SERIES_ID,
        source="yfinance",
        as_of=NOW,
    )

    assert study.stale_open_share == pytest.approx(1.0, abs=0.02)
    assert study.open_quality == "unusable"
    assert study.clean_from is None
    assert study.clean_segments == ()
    # Sin tramo nocturno fiable no se aprueba ni se suspende: `inconclusive`.
    assert study.phase0_gate == "inconclusive"
    assert all(float(str(row["stale_open_share"])) > 0.9 for row in study.by_year)
    assert "open" in " ".join(study.limitations)


def test_the_verdict_uses_only_the_clean_sample(tmp_path: Path) -> None:
    """El veredicto se decide sobre la muestra limpia, no sobre la contaminada.

    Las primeras sesiones llegan con el `open` repetido (su tramo nocturno es cero
    por construcción y la sesión se queda con todo el retorno, que es lo que
    empujaría el veredicto hacia `intraday`); las últimas son limpias y llevan el
    drift puesto **fuera** de la sesión.
    """
    sessions = _bars(1200, overnight_bp=+6.0, intraday_cycle=[+2.0, -2.0])
    records: list[dict[str, object]] = [sessions[0]]
    for index, (record, previous) in enumerate(zip(sessions[1:], sessions, strict=False)):
        records.append({**record, "open": previous["close"]} if index < 600 else record)
    study = decompose(
        load_sessions(_store_with(tmp_path, records), series_id=SERIES_ID),
        series_id=SERIES_ID,
        source="yfinance",
        as_of=NOW,
    )

    assert study.stale_open_share == pytest.approx(0.5, abs=0.02)
    assert study.clean_from is not None
    assert study.clean_sessions >= 250
    # En la muestra completa el tramo nocturno está "arreglado" a cero; en la limpia
    # se ve el drift de verdad, que está fuera de la sesión.
    assert study.verdict is DriftVerdict.OVERNIGHT
    assert study.clean_segment("overnight").mean_bp > study.clean_segment("intraday").mean_bp


# ─────────────────────────────────────────────────────────────────────────────
# Tasa base y segmentados
# ─────────────────────────────────────────────────────────────────────────────
def test_the_base_rate_counts_up_sessions_and_absolute_moves(tmp_path: Path) -> None:
    """Tres de cada cuatro sesiones suben, y la distribución de |open→close| se declara."""
    store = _store_with(
        tmp_path,
        _bars(40, overnight_bp=0.0, intraday_cycle=[+10.0, +10.0, +10.0, -10.0]),
    )
    study = decompose(
        load_sessions(store, series_id=SERIES_ID),
        series_id=SERIES_ID,
        source="yfinance",
        as_of=NOW,
    )

    assert study.base_rate.sessions == 39
    assert study.base_rate.up_share == pytest.approx(0.75, abs=0.02)
    assert study.base_rate.abs_move_median_bp == pytest.approx(10.0, abs=0.1)
    assert study.base_rate.abs_move_p99_bp >= study.base_rate.abs_move_p90_bp


def test_the_study_is_segmented_by_year_weekday_and_volatility(tmp_path: Path) -> None:
    """El informe segmenta por año, día de la semana y régimen de volatilidad."""
    store = _store_with(
        tmp_path,
        _bars(400, overnight_bp=+2.0, intraday_cycle=[+6.0, -4.0, +2.0, +1.0, -5.0]),
    )
    study = decompose(
        load_sessions(store, series_id=SERIES_ID),
        series_id=SERIES_ID,
        source="yfinance",
        as_of=NOW,
    )

    years = [str(row["year"]) for row in study.by_year]
    weekdays = [str(row["weekday"]) for row in study.by_weekday]
    regimes = [str(row["vol_regime"]) for row in study.by_volatility]

    assert len(years) >= 2 and years == sorted(years)
    assert weekdays == ["lunes", "martes", "miércoles", "jueves", "viernes"]
    assert regimes == ["low", "medium", "high"]
    sessions = [row["sessions"] for row in study.by_volatility]
    assert all(isinstance(value, int) and value > 0 for value in sessions)


def test_the_volatility_regime_uses_only_past_sessions(tmp_path: Path) -> None:
    """La volatilidad que clasifica una sesión se mide con las 20 anteriores, nunca con ella."""
    store = _store_with(tmp_path, _bars(40, overnight_bp=0.0, intraday_cycle=[+1.0, -1.0]))
    frame = load_sessions(store, series_id=SERIES_ID)

    volatilities = frame.get_column("vol20").to_list()
    assert len(volatilities) == 39
    assert all(value is None for value in volatilities[:20])  # aún no hay 20 anteriores
    assert all(value is not None for value in volatilities[20:])


# ─────────────────────────────────────────────────────────────────────────────
# Informe
# ─────────────────────────────────────────────────────────────────────────────
def test_the_report_is_written_with_the_verdict_and_the_declared_limits(tmp_path: Path) -> None:
    """El informe queda escrito, empieza por el veredicto y declara la limitación del CFD."""
    _store_with(tmp_path, _bars(320, overnight_bp=+3.0, intraday_cycle=[+1.0, -1.0]))
    study = analyse(
        data_root=tmp_path,
        now=NOW,
        reports_dir=tmp_path / "derived" / "reports",
    )

    json_path = tmp_path / "derived" / "reports" / "drift_decomposition_2024-01-01.json"
    markdown_path = json_path.with_suffix(".md")
    assert json_path.is_file() and markdown_path.is_file()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["verdict"] == "overnight"
    assert payload["phase0_gate"] == "fail"
    assert payload["segments"][0]["name"] == "intraday"
    assert study.verdict is DriftVerdict.OVERNIGHT

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "El drift está en el tramo nocturno" in markdown
    assert "#50" in markdown  # la ausencia de fuente del CFD se declara, no se esconde
    assert "Puerta de la Fase 0" in markdown


def test_the_study_fails_loudly_when_there_is_not_enough_data(tmp_path: Path) -> None:
    """Sin datos no se inventa un informe: se falla con el motivo."""
    with pytest.raises(ConfigurationError, match="tarea #3"):
        load_sessions(Store(tmp_path), series_id=SERIES_ID)

    # Y con un solo día tampoco: no hay tramo nocturno que medir.
    single = _store_with(tmp_path, _bars(1, overnight_bp=0.0, intraday_cycle=[0.0]))
    with pytest.raises(ConfigurationError, match="sesiones suficientes"):
        load_sessions(single, series_id=SERIES_ID)


def test_the_same_reference_instant_gives_the_same_report(tmp_path: Path) -> None:
    """Con el mismo `now`, el estudio es idéntico: es una medición, no una tirada de dados."""
    store = _store_with(tmp_path, _bars(320, overnight_bp=+3.0, intraday_cycle=[+4.0, -2.0]))
    frame = load_sessions(store, series_id=SERIES_ID)

    first = decompose(frame, series_id=SERIES_ID, source="yfinance", as_of=NOW)
    second = decompose(frame, series_id=SERIES_ID, source="yfinance", as_of=NOW)

    assert first.segments == second.segments
    assert first.verdict is second.verdict
    assert first.by_year == second.by_year


def test_the_markdown_starts_with_the_verdict(tmp_path: Path) -> None:
    """Lo primero que se lee es la decisión, no una tabla."""
    store = _store_with(tmp_path, _bars(320, overnight_bp=+3.0, intraday_cycle=[+4.0, -2.0]))
    study = decompose(
        load_sessions(store, series_id=SERIES_ID),
        series_id=SERIES_ID,
        source="yfinance",
        as_of=NOW,
    )

    markdown = render_markdown(study)

    assert "**Veredicto:**" in markdown
    assert "## Los tres tramos" in markdown
    assert "## Limitaciones (declaradas, no escondidas)" in markdown
