"""Tests del etiquetado tri-barrera (tarea #10).

Dos bloques:

- **Casos a mano** sobre la funcion pura (``label_session``, ``barrier_levels``,
  ``order_source_for``): los valores de barrera, salidas y retornos se calculan a
  mano y se escriben en el test.
- **Almacen sintetico** en ``tmp_path`` para el resto: seleccion de #7, muestra
  limpia de #52, medias sesiones, intradia y respaldo, persistencia en
  ``derived.labels``, determinismo, no *look-ahead* y la CLI.

Nada del ``data/`` del repositorio: la fixture guardiana de ``tests/conftest.py``
comprueba que la sesion de tests no lo toca (A34).
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from cfdtrader.analysis.volatility_forecast import (
    MIN_TRAIN,
    Selection,
    build_sample,
    scale_sigma_for_duration,
    select_candidate,
    walk_forward,
)
from cfdtrader.data.calendar import FULL_SESSION_HOURS, HALF_SESSION_HOURS, MarketCalendar
from cfdtrader.data.settings import ConfigurationError
from cfdtrader.data.store import Store
from cfdtrader.models.labels import (
    CARRIER_PREVIOUS_LABELLED,
    CARRIER_WALK_FORWARD,
    COST_SENSITIVITY_SCENARIOS,
    DEFAULT_ENTRY_PRICE_SOURCE,
    ENTRY_PRICE_SOURCES,
    ENTRY_PRICE_T0_SNAPSHOT,
    EV_IDENTITY_TOLERANCE,
    FALLBACK_ORDER_SOURCE,
    INTRADAY_ORDER_SOURCE,
    LABEL_STOP,
    LABEL_TARGET,
    LABEL_TIME,
    LABELS_DATASET,
    LABELS_SOURCE,
    MIN_INTRADAY_COVERAGE,
    PERSISTED_K_SIGMA,
    R_SIGMA_SCENARIOS,
    REASON_CALENDAR_NO_SESSION,
    REASON_NO_FORECAST,
    REASON_NULL_OHLC,
    REASON_SESSION_NOT_CLOSED,
    REASON_STALE_OPEN,
    ROUND_TRIP_SPREAD,
    DailyBar,
    EntryPriceUnavailableError,
    IntradayBar,
    LabelsError,
    LabelsRun,
    SessionLabel,
    barrier_levels,
    label_and_write,
    label_history,
    label_session,
    main,
    order_source_for,
    phase0_context,
    render_markdown,
    report_payload,
    resolve_candidate,
    write_outputs,
)

#: Instante de referencia fijo (determinismo comprobable).
NOW = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)

SERIES_ID = "^GSPC"

#: Primer dia laborable de la serie sintetica.
START = date(2005, 1, 3)

#: Sesiones sinteticas: suficientes para pasar el calentamiento de #7 y evaluar.
SESSIONS = 1600

#: Ultimo ano con el ``open`` repetido del cierre anterior (el artefacto de #52).
STALE_THROUGH = 2006

#: Anio desde el que la regla de #52 declara limpia la muestra.
EXPECTED_CUTOFF = "2007-01-01"

#: Posiciones (dentro de la serie) con el ``open`` repetido despues del corte.
STALE_LATER = (700, 1100)

#: Posiciones con ``high``/``low``/``close`` nulos (excluidas por #7 y por A19).
NULL_OHLC_AT = (1580, 1581)

#: Sesiones con intradia completo (78 barras) al alza, a la baja, y con cobertura corta.
INTRADAY_UP_AT = (-40, -35, -30)
INTRADAY_DOWN_AT = -28
INTRADAY_SHORT_AT = -25

EASTERN = MarketCalendar()


# ─────────────────────────────────────────────────────────────────────────────
# Almacen sintetico
# ─────────────────────────────────────────────────────────────────────────────
def _trading_days(count: int, start: date = START) -> list[date]:
    """Las primeras ``count`` sesiones reales (festivos y fines de semana fuera)."""
    days: list[date] = []
    cursor = start
    while len(days) < count:
        if EASTERN.is_session(cursor):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


DAYS = _trading_days(SESSIONS)


def _records(count: int = SESSIONS, *, seed: int = 11) -> list[dict[str, object]]:
    """Sesiones con volatilidad variable y el artefacto #52 en los primeros anos.

    En los anos ``<= STALE_THROUGH`` el ``open`` es **exactamente** el cierre
    anterior (volatilidad nocturna cero por construccion: el artefacto de #52), y
    ademas hay dos sesiones stale despues del corte y dos con OHLC nulo.
    """
    rng = np.random.default_rng(seed)
    days = _trading_days(count)
    log_vol = np.zeros(count)
    for index in range(1, count):
        log_vol[index] = 0.93 * log_vol[index - 1] + rng.normal(0.0, 0.11)
    vol = 0.009 * np.exp(log_vol)
    step = rng.normal(0.0, 1.0, count) * vol
    close = 3000.0 * np.cumprod(1.0 + step)
    fetched_at = datetime.combine(days[-1], dtime(23, 0), tzinfo=UTC)
    stale_later = set(STALE_LATER)
    null_row: set[int] = set(NULL_OHLC_AT)
    records: list[dict[str, object]] = []
    previous_close: float | None = None
    for index, day in enumerate(days):
        close_value = float(close[index])
        if day.year <= STALE_THROUGH or index in stale_later:
            open_price = close_value if previous_close is None else previous_close
        else:
            open_price = close_value * (1.0 + rng.normal(0.0, 0.004))
        spread = abs(float(step[index])) * 0.5
        high = max(open_price, close_value) * (1.0 + spread)
        low = min(open_price, close_value) * (1.0 - spread)
        records.append(
            {
                "source": "yfinance",
                "series_id": SERIES_ID,
                "as_of": datetime.combine(day, dtime(21, 0), tzinfo=UTC),
                "fetched_at": fetched_at,
                "published_at": None,
                "open": float(open_price),
                # A19/#7 excluyen la sesion por `high`/`low` nulos; `close` sigue
                # siendo valido, si no la sesion ni siquiera entraria en la muestra.
                "high": None if index in null_row else float(high),
                "low": None if index in null_row else float(low),
                "close": close_value,
                "volume": 1_000_000.0,
                "adj_close": close_value,
            }
        )
        previous_close = close_value
    return records


def _intraday_records(day: date, bars: int, *, drift: float) -> list[dict[str, object]]:
    """Barras de 5 minutos desde la apertura real, moviendose a favor o en contra.

    Arrancan en el ``open`` diario de esa sesion (que es el precio de entrada) para
    que el recorrido cruce de verdad las barreras: ``drift > 0`` toca la favorable.
    """
    info = EASTERN.session(day)
    assert info.open_utc is not None
    entry = next(
        cast("float", record["open"]) for record in _records() if record["as_of"] == _at(day)
    )
    records: list[dict[str, object]] = []
    price = entry
    for index in range(bars):
        moment = info.open_utc + timedelta(minutes=5 * index)
        price = price * (1.0 + drift)
        records.append(
            {
                "source": "yfinance",
                "series_id": SERIES_ID,
                "as_of": moment,
                "fetched_at": NOW,
                "published_at": None,
                "open": price,
                "high": price * 1.001 if drift > 0 else price * 1.0005,
                "low": price * 0.999 if drift > 0 else price * 0.9995,
                "close": price,
                "volume": 1_000.0,
                "interval": "5m",
                "bid": None,
                "ask": None,
            }
        )
    return records


def _build_store(root: Path, *, records: list[dict[str, object]] | None = None) -> Store:
    """Almacen temporal con las sesiones sinteticas (y el intradia) ya escritas."""
    store = Store(root)
    store.append("raw", "market_daily", records if records is not None else _records())
    intraday: list[dict[str, object]] = []
    for index in INTRADAY_UP_AT:
        intraday.extend(_intraday_records(DAYS[index], 78, drift=0.01))
    intraday.extend(_intraday_records(DAYS[INTRADAY_DOWN_AT], 78, drift=-0.01))
    intraday.extend(_intraday_records(DAYS[INTRADAY_SHORT_AT], 70, drift=0.01))
    if intraday:
        store.append("raw", "market_intraday", intraday)
    return store


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Raiz del almacen sintetico, construida una sola vez."""
    root = tmp_path_factory.mktemp("labels")
    _build_store(root)
    return root


@pytest.fixture(scope="module")
def run(synthetic_root: Path) -> LabelsRun:
    """Etiquetado del almacen sintetico (no escribe nada)."""
    return label_history(store=Store(synthetic_root), now=NOW)


@pytest.fixture
def writable_root(tmp_path: Path, synthetic_root: Path) -> Path:
    """Copia escribible del almacen sintetico (los tests de escritura no lo comparten)."""
    shutil.copytree(synthetic_root, tmp_path / "data")
    return tmp_path / "data"


def _dict(value: object) -> dict[str, Any]:
    """Diccionario del payload, tipado para el modo estricto."""
    assert isinstance(value, dict), f"se esperaba un diccionario, no {type(value).__name__}"
    return cast("dict[str, Any]", value)


def _number(value: object) -> float:
    """Numero del payload."""
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"no es numero: {value!r}"
    )
    return float(value)


def _at(day: date) -> datetime:
    """Instante del cierre de la sesion (el mismo ``as_of`` que el resto de la serie)."""
    return datetime.combine(day, dtime(21, 0), tzinfo=UTC)


def _row(run_: LabelsRun, day: date) -> Any:
    """Fila etiquetada de esa sesion."""
    found = [item for item in run_.rows if item.session == day]
    assert found, f"la sesion {day} no esta etiquetada"
    return found[0]


# ─────────────────────────────────────────────────────────────────────────────
# Casos a mano sobre la funcion pura (A4, A6-A8, A13-A15, A21, A29)
# ─────────────────────────────────────────────────────────────────────────────
ENTRY = 100.0
SIGMA = 0.01  # k = 1.0 => barreras en 101.0 y 99.0


def _bar(high: float, low: float) -> IntradayBar:
    """Barra con ese ``high`` y ese ``low`` (``open``/``close`` no deciden nada)."""
    return IntradayBar(
        as_of=datetime(2024, 1, 1, 14, 30, tzinfo=UTC),
        open=low,
        high=high,
        low=low,
        close=low,
    )


def _bars(*levels: tuple[float, float], expected: int = 78) -> tuple[IntradayBar, ...]:
    """Barras significativas, rellenadas con barras planas para cubrir la sesion."""
    pairs = [*levels, *([(100.0, 100.0)] * (expected - len(levels)))]
    return tuple(_bar(high, low) for high, low in pairs)


DAILY_NEUTRAL = DailyBar(high=100.2, low=99.8, close=100.05)


def _label(
    *,
    intraday: tuple[IntradayBar, ...] = (),
    daily: DailyBar = DAILY_NEUTRAL,
    sigma: float = SIGMA,
    k_sigma: float = 1.0,
    expected_bars: int = 78,
) -> SessionLabel:
    """``label_session`` con el precio de entrada explicito de los tests a mano."""
    return label_session(
        entry_px=ENTRY,
        sigma=sigma,
        k_sigma=k_sigma,
        daily=daily,
        intraday=intraday,
        expected_bars=expected_bars,
    )


def test_barriers_double_exactly_when_sigma_doubles() -> None:
    """A4: ``target_pct = stop_pct = k * sigma``; duplicar sigma duplica las barreras."""
    simple = barrier_levels(entry_px=ENTRY, sigma=SIGMA, k_sigma=1.0)
    double = barrier_levels(entry_px=ENTRY, sigma=2 * SIGMA, k_sigma=1.0)
    expected = {
        "simple": (simple.target_pct, simple.stop_pct, simple.upper, simple.lower),
        "double": (double.target_pct, double.stop_pct, double.upper, double.lower),
    }
    assert expected["simple"] == (0.01, 0.01, 101.0, 99.0)
    assert expected["double"] == (0.02, 0.02, 102.0, 98.0)


def test_the_scenarios_of_k_are_declared_and_are_not_the_owner_decision(run: LabelsRun) -> None:
    """A4/A5: los escenarios existen, son ilustrativos y solo se persiste ``k = 1.0``."""
    assert R_SIGMA_SCENARIOS == (0.5, 1.0, 1.5)
    assert PERSISTED_K_SIGMA in R_SIGMA_SCENARIOS
    assert all(row.k_sigma == PERSISTED_K_SIGMA for row in run.rows)
    assert set(run.scenarios) == {f"k_{value!r}" for value in R_SIGMA_SCENARIOS}
    assert "ilustrativo" in render_markdown(run).lower()
    doubled = next(row for row in run.scenarios["k_1.0"] if row.session == run.rows[0].session)
    single = run.rows[0]
    assert single.target_pct == pytest.approx(doubled.target_pct)
    half = next(row for row in run.scenarios["k_0.5"] if row.session == single.session)
    assert half.target_pct == pytest.approx(single.target_pct / 2)


def test_target_only_in_both_directions() -> None:
    """A7/A21: sube por encima de la superior => ``target`` en largo y ``stop`` en corto."""
    label = _label(intraday=_bars((101.5, 100.5)))
    fields = {
        "label_long": label.label_long,
        "label_short": label.label_short,
        "exit_long": label.exit_long,
        "exit_short": label.exit_short,
        "ret_long": label.ret_long,
        "ret_short": label.ret_short,
    }
    assert fields["label_long"] == LABEL_TARGET
    assert fields["label_short"] == LABEL_STOP
    # La salida es el precio de la barrera: 101.0 para las dos direcciones.
    assert fields["exit_long"] == 101.0
    assert fields["exit_short"] == 101.0
    assert fields["ret_long"] == pytest.approx(0.01)
    assert fields["ret_short"] == pytest.approx(-0.01)


def test_stop_only_in_both_directions() -> None:
    """A7/A21: baja por debajo de la inferior => ``stop`` en largo y ``target`` en corto."""
    label = _label(intraday=_bars((100.5, 98.5)))
    fields = {
        "label_long": label.label_long,
        "label_short": label.label_short,
        "exit_long": label.exit_long,
        "exit_short": label.exit_short,
        "ret_long": label.ret_long,
        "ret_short": label.ret_short,
    }
    assert fields["label_long"] == LABEL_STOP
    assert fields["label_short"] == LABEL_TARGET
    assert fields["exit_long"] == 99.0
    assert fields["exit_short"] == 99.0
    assert fields["ret_long"] == pytest.approx(-0.01)
    assert fields["ret_short"] == pytest.approx(0.01)


def test_time_label_when_no_barrier_is_touched() -> None:
    """A21: sin tocar ninguna barrera, la posicion sale al cierre de la sesion."""
    label = _label(intraday=_bars((100.5, 99.9)), daily=DailyBar(high=100.6, low=99.5, close=100.4))
    fields = {
        "long": (label.label_long, label.exit_long, label.ret_long),
        "short": (label.label_short, label.exit_short, label.ret_short),
    }
    assert fields["long"] == (LABEL_TIME, 100.4, pytest.approx(0.004))
    assert fields["short"][0] == LABEL_TIME
    assert fields["short"][1] == 100.4
    assert fields["short"][2] == pytest.approx(-0.004)


def test_a_bar_touching_both_barriers_is_the_adverse_label() -> None:
    """A14: empate dentro de la misma barra => ``stop`` en las dos direcciones."""
    label = _label(intraday=_bars((101.5, 98.5)))
    fields = {
        "long": (label.label_long, label.exit_long),
        "short": (label.label_short, label.exit_short),
    }
    assert fields["long"] == (LABEL_STOP, 99.0)
    assert fields["short"] == (LABEL_STOP, 101.0)
    assert label.ties_in_bar == 1
    assert label.fallback_ties == 0
    assert label.decided_bar == 1


def test_a_tie_in_the_opening_bar_is_a_stop() -> None:
    """A14: la barra de apertura que toca las dos => ``stop``."""
    label = _label(intraday=_bars((102.0, 98.0), (101.5, 99.0)))
    assert (label.label_long, label.label_short, label.decided_bar) == (LABEL_STOP, LABEL_STOP, 1)
    assert label.ties_in_bar == 1


def test_favourable_in_the_third_bar_and_adverse_in_the_seventh_is_a_target() -> None:
    """A14: decide la **primera** barra que toca una barrera."""
    label = _label(
        intraday=_bars(
            (100.5, 99.9),
            (100.5, 99.9),
            (101.5, 100.5),
            (100.5, 99.9),
            (100.5, 99.9),
            (100.5, 99.9),
            (100.5, 98.5),
        )
    )
    assert (label.label_long, label.label_short, label.decided_bar) == (LABEL_TARGET, LABEL_STOP, 3)


def test_the_touch_is_inclusive_and_the_barrier_is_not_rounded() -> None:
    """A8: ``high >= barrera`` con el precio de barrera sin redondear."""
    exact = _label(intraday=_bars((101.0, 100.5)))
    below = _label(intraday=_bars((101.0 - 1e-9, 100.5)))
    assert exact.label_long == LABEL_TARGET
    assert below.label_long == LABEL_TIME


def test_without_intraday_the_declared_fallback_is_used() -> None:
    """A13/A15: sin barras intradia, respaldo diario declarado e ``intraday_incomplete``."""
    label = _label(daily=DailyBar(high=101.4, low=98.6, close=100.1))
    assert label.order_source == FALLBACK_ORDER_SOURCE
    assert label.intraday_incomplete is True
    assert label.bars_observed == 0
    assert (label.label_long, label.label_short) == (LABEL_STOP, LABEL_STOP)
    assert label.fallback_ties == 1
    assert label.ties_in_bar == 0


def test_the_fallback_only_picks_the_favourable_when_nothing_else_is_touched() -> None:
    """A15: favorable sola => ``target``; adversa sola => ``stop``; ninguna => ``time``."""
    up = _label(daily=DailyBar(high=101.4, low=99.6, close=101.2))
    down = _label(daily=DailyBar(high=100.4, low=98.6, close=98.8))
    flat = _label(daily=DailyBar(high=100.4, low=99.6, close=100.1))
    assert (up.label_long, up.label_short) == (LABEL_TARGET, LABEL_STOP)
    assert (down.label_long, down.label_short) == (LABEL_STOP, LABEL_TARGET)
    assert (flat.label_long, flat.label_short) == (LABEL_TIME, LABEL_TIME)


def test_the_order_source_needs_the_declared_coverage() -> None:
    """A13: ``MIN_INTRADAY_COVERAGE`` sobre las barras esperadas de la sesion."""
    cases = {
        "completo": order_source_for(observed_bars=78, expected_bars=78),
        "justo": order_source_for(observed_bars=75, expected_bars=78),
        "corto": order_source_for(observed_bars=74, expected_bars=78),
        "vacio": order_source_for(observed_bars=0, expected_bars=78),
        "sin_esperadas": order_source_for(observed_bars=10, expected_bars=0),
    }
    assert MIN_INTRADAY_COVERAGE == 0.95
    assert cases["completo"] == (INTRADAY_ORDER_SOURCE, 1.0, False)
    assert cases["justo"] == (INTRADAY_ORDER_SOURCE, 75 / 78, False)
    assert cases["corto"] == (FALLBACK_ORDER_SOURCE, 74 / 78, True)
    assert cases["vacio"] == (FALLBACK_ORDER_SOURCE, 0.0, True)
    assert cases["sin_esperadas"] == (FALLBACK_ORDER_SOURCE, 0.0, True)


def test_the_expected_bars_of_a_session_depend_on_its_duration() -> None:
    """A11/A13: 78 barras por sesion completa y 42 en media sesion."""
    full = _label(intraday=_bars((100.5, 99.9)), expected_bars=78)
    half = _label(intraday=_bars((100.5, 99.9), expected=42), expected_bars=42)
    assert full.order_source == INTRADAY_ORDER_SOURCE
    assert half.order_source == INTRADAY_ORDER_SOURCE
    assert round(FULL_SESSION_HOURS * 12) == 78
    assert round(HALF_SESSION_HOURS * 12) == 42


def test_the_pure_labelling_runs_in_memory_without_touching_disk() -> None:
    """A30: la etiqueta es una funcion pura de (sigma, entrada, OHLC, barras)."""
    first = _label(intraday=_bars((101.5, 98.5)), daily=DailyBar(101.0, 99.0, 100.5))
    second = _label(intraday=_bars((101.5, 98.5)), daily=DailyBar(101.0, 99.0, 100.5))
    assert first == second


# ─────────────────────────────────────────────────────────────────────────────
# Contexto declarado (A1) y candidato de #7 (A2)
# ─────────────────────────────────────────────────────────────────────────────
def test_the_phase0_decision_is_declared_and_never_sold_as_a_validation(run: LabelsRun) -> None:
    """A1: la Fase 1 se construye con la puerta en ``fail`` y eso se dice con esas palabras."""
    context = phase0_context()
    assert context["gate"] == "fail"
    assert context["phase1_ready"] is False
    assert context["decided_on"] == "2026-09-18"
    assert context["provenance"] == "declaracion del usuario"
    assert context["source_issue"] == 9
    assert "Fase 1" in str(context["owner_decision"])

    payload = report_payload(run)
    assert _dict(payload["phase0_context"])["phase1_ready"] is False
    markdown = render_markdown(run)
    assert "`phase1_ready`: **`false`**" in markdown
    assert "no es una validacion de la estrategia" in markdown
    assert "`gate`: **`fail`**" in markdown


def test_resolve_candidate_covers_the_three_branches_of_task_7() -> None:
    """A2: elegido / ``no_better_than_naive`` => ``rw`` / ``inconclusive`` => error."""

    def selection(selected: str | None, verdict: str) -> Selection:
        return Selection(
            selected=selected,
            verdict=verdict,
            metric="qlike",
            rule=(),
            constants={},
            arithmetic=(),
            leaders={},
        )

    cases = {
        "elegido": resolve_candidate(selection("garch", "selected")),
        "naive": resolve_candidate(selection(None, "no_better_than_naive")),
    }
    assert cases["elegido"] == "garch"
    assert cases["naive"] == "rw"
    with pytest.raises(LabelsError):
        _ = resolve_candidate(selection(None, "inconclusive"))


def test_the_sigma_series_is_the_one_task_7_produces(run: LabelsRun, synthetic_root: Path) -> None:
    """A2/A18: la sigma es ``sqrt`` del *forecast* del candidato de #7, sesion a sesion."""
    store = Store(synthetic_root)
    frame, _ = build_sample(store, series_id=SERIES_ID)
    walk = walk_forward(frame)
    selection = select_candidate(walk.candidates)
    candidate = resolve_candidate(selection)
    sessions = [cast("date", value) for value in frame.get_column("session").to_list()]
    first_index = walk.bounds[0][0]
    expected = {
        sessions[index]: float(np.sqrt(walk.forecasts[candidate][index]))
        for index in range(first_index, frame.height)
    }
    labelled = {
        row.session: row.sigma for row in run.rows if row.sigma_carrier == CARRIER_WALK_FORWARD
    }
    assert run.forecast_candidate == candidate
    assert run.selection_verdict == selection.verdict
    assert labelled
    for day, sigma in labelled.items():
        assert sigma == pytest.approx(expected[day], rel=0, abs=0)


def test_the_units_are_declared(run: LabelsRun) -> None:
    """A3: fraccion en el calculo, bp (x 10^4) al informar."""
    payload = report_payload(run)
    units = _dict(payload["units"])
    assert units["calculation"] == "fraccion"
    assert units["bp_per_unit"] == 10_000.0
    assert "bp" in str(units["report"])
    summary = _dict(_dict(_dict(_dict(payload["summary"])["k_1.0"])["directions"])["long"])["all"]
    block = _dict(summary)
    bp = _dict(block["bp"])
    assert set(bp) >= {"e_gain", "e_loss", "ev_media", "ev_neto", "c"}
    assert _number(bp["ev_media"]) == pytest.approx(_number(block["ev_media"]) * 10_000.0)
    assert _number(bp["c"]) == ROUND_TRIP_SPREAD * 10_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Muestra, sigma y medias sesiones (A16-A20)
# ─────────────────────────────────────────────────────────────────────────────
def test_the_clean_cutoff_is_the_same_as_task_7(run: LabelsRun, synthetic_root: Path) -> None:
    """A19: la definicion de muestra limpia de #52 es una sola en todo el proyecto."""
    _, summary = build_sample(Store(synthetic_root), series_id=SERIES_ID)
    clean = _dict(run.inputs["clean_sample"])
    assert clean["clean_from"] == summary["clean_from"] == EXPECTED_CUTOFF
    assert clean["clean_sessions"] == summary["clean_sessions"]
    assert clean["exclusions"] == summary["exclusions"]


def test_stale_and_null_sessions_are_unlabelled_with_their_reason(run: LabelsRun) -> None:
    """A19: ``open`` repetido y OHLC nulo quedan fuera, con el motivo declarado."""
    reasons = _dict(run.sample["reasons"])
    assert _number(_dict(reasons[REASON_STALE_OPEN])["count"]) == len(STALE_LATER)
    assert _number(_dict(reasons[REASON_NULL_OHLC])["count"]) == len(NULL_OHLC_AT)
    excluded = {DAYS[index] for index in (*STALE_LATER, *NULL_OHLC_AT)}
    labelled = {row.session for row in run.rows}
    assert not (excluded & labelled)
    for reason in (REASON_STALE_OPEN, REASON_NULL_OHLC):
        detail = _dict(reasons[reason])
        assert detail["first"] is not None and detail["last"] is not None


def test_a_row_on_a_holiday_is_unlabelled_with_its_own_reason(tmp_path: Path) -> None:
    """A19: una fila en un festivo no es una sesion: se declara, no se etiqueta."""
    holiday = date(2010, 12, 24)  # viernes: la bolsa americana esta cerrada
    assert EASTERN.is_session(holiday) is False
    records = _records()
    records.insert(
        next(
            index
            for index, record in enumerate(records)
            if cast("datetime", record["as_of"]) > _at(holiday)
        ),
        {
            "source": "yfinance",
            "series_id": SERIES_ID,
            "as_of": _at(holiday),
            "fetched_at": datetime(2024, 1, 1, tzinfo=UTC),
            "published_at": None,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1.0,
            "adj_close": 100.5,
        },
    )
    root = tmp_path / "holiday"
    _build_store(root, records=records)
    result = label_history(store=Store(root), now=NOW)
    reasons = _dict(result.sample["reasons"])
    assert _number(_dict(reasons[REASON_CALENDAR_NO_SESSION])["count"]) == 1
    assert holiday not in {row.session for row in result.rows}


def test_sessions_without_a_forecast_are_unlabelled_and_sigma_is_never_imputed(
    run: LabelsRun,
) -> None:
    """A18: el calentamiento de ``MIN_TRAIN`` queda sin etiqueta y no se rellena con nada."""
    sigma = _dict(run.inputs["sigma"])
    reasons = _dict(run.sample["reasons"])
    assert sigma["min_train"] == MIN_TRAIN == 500
    assert _number(sigma["warmup_sessions"]) == MIN_TRAIN
    assert _number(sigma["sessions_with_forecast"]) == _number(sigma["evaluated_sessions"])
    assert (
        EXPECTED_CUTOFF
        <= str(sigma["warmup_from"])
        < str(sigma["warmup_to"])
        < str(sigma["first_evaluated"])
    )
    assert _number(_dict(reasons[REASON_NO_FORECAST])["count"]) >= MIN_TRAIN
    assert str(_dict(reasons[REASON_NO_FORECAST])["first"]) < str(sigma["first_evaluated"])

    evaluated = {row.session for row in run.rows if row.sigma_carrier == CARRIER_WALK_FORWARD}
    assert evaluated
    assert min(evaluated) >= date.fromisoformat(str(sigma["first_evaluated"]))
    assert len(evaluated) <= _number(sigma["sessions_with_forecast"])
    assert all(row.sigma > 0.0 for row in run.rows)
    assert "prohibido imputar sigma" in str(sigma["no_imputation"])


def test_a_half_session_is_labelled_with_a_carried_and_scaled_sigma(run: LabelsRun) -> None:
    """A20: media sesion con cierre a las 13:00 ET y sigma arrastrada y escalada."""
    halves = [row for row in run.rows if row.is_half_day]
    assert halves, "el almacen sintetico tiene al menos una media sesion etiquetada"
    row = halves[0]
    reference = [
        item
        for item in run.rows
        if item.session < row.session and item.sigma_carrier == CARRIER_WALK_FORWARD
    ][-1]
    expected = scale_sigma_for_duration(reference.sigma, duration_hours=HALF_SESSION_HOURS)
    assert row.sigma_carrier == CARRIER_PREVIOUS_LABELLED
    assert row.sigma == pytest.approx(expected, rel=1e-12)
    assert row.sigma == pytest.approx(reference.sigma * 0.7337993857053428, rel=1e-9)
    info = EASTERN.session(row.session)
    assert info.is_half_day is True
    assert info.close_utc is not None
    assert row.close_utc == info.close_utc
    assert info.close_et is not None and (info.close_et.hour, info.close_et.minute) == (13, 0)
    assert _number(run.sample["labelled_half_sessions"]) == len(halves)


def test_the_time_barrier_comes_from_the_calendar_and_survives_the_dst_week(
    run: LabelsRun,
) -> None:
    """A11: la clave de sesion se deriva en ET y las sesiones del cambio de hora son contiguas."""
    march = [row for row in run.rows if (row.session.month, row.session.day) == (3, 9)]
    sessions = [row.session for row in run.rows]
    assert sessions == sorted(sessions)
    assert len(sessions) == len(set(sessions))
    for session in (date(2007, 3, 9), date(2007, 3, 12), date(2007, 11, 5), date(2007, 11, 6)):
        info = EASTERN.session(session)
        assert info.close_utc is not None and info.close_et is not None
        assert info.close_et.hour == 16 and info.close_et.minute == 0
    assert march, "el almacen sintetico incluye la semana del cambio de hora de marzo"
    full = EASTERN.session(date(2024, 1, 2))
    half = EASTERN.session(date(2024, 11, 29))
    assert full.duration_hours == FULL_SESSION_HOURS and half.duration_hours == HALF_SESSION_HOURS
    assert full.close_utc is not None and full.close_utc.hour == 21  # 16:00 EST
    assert half.close_utc is not None and half.close_utc.hour == 18  # 13:00 EST


def test_the_intraday_ordering_is_used_where_the_coverage_allows_it(run: LabelsRun) -> None:
    """A13/A16/A17: cobertura suficiente => intradia; insuficiente => respaldo declarado."""
    coverage = _dict(run.inputs["intraday_coverage"])
    ordered = _dict(run.sample["by_order_source"])
    with_intraday = len(INTRADAY_UP_AT) + 1
    assert _number(coverage["sessions"]) == with_intraday + 1
    assert _number(coverage["bars"]) == with_intraday * 78 + 70
    assert _number(coverage["ordered_with_intraday"]) == with_intraday
    assert ordered[INTRADAY_ORDER_SOURCE] == with_intraday
    assert ordered[FALLBACK_ORDER_SOURCE] == len(run.rows) - with_intraday
    assert _number(coverage["bars_at_0845_et"]) == 0
    assert _number(coverage["stored_daily_sessions"]) == SESSIONS
    for index in (*INTRADAY_UP_AT, INTRADAY_DOWN_AT):
        assert _row(run, DAYS[index]).order_source == INTRADAY_ORDER_SOURCE
    short = _row(run, DAYS[INTRADAY_SHORT_AT])
    assert short.order_source == FALLBACK_ORDER_SOURCE
    assert short.intraday_incomplete is True
    assert short.bars_observed == 70
    assert short.coverage == pytest.approx(70 / 78)


def test_the_fallback_bias_is_declared_as_a_lower_bound(run: LabelsRun) -> None:
    """A16: el respaldo es conservador y ``p_target`` es una cota inferior."""
    sample = run.sample
    assert sample["bias"] == "conservative_lower_bound_on_p_target"
    assert "cota inferior" in str(sample["bias_sentence"])
    assert 0.0 < _number(sample["fallback_share"]) < 1.0
    markdown = render_markdown(run)
    assert "cota inferior" in markdown
    assert "conservative_lower_bound_on_p_target" in markdown


# ─────────────────────────────────────────────────────────────────────────────
# Resumen: p_target, p_win y EV (A21-A25)
# ─────────────────────────────────────────────────────────────────────────────
def test_the_summary_is_published_per_scenario_direction_and_sample(run: LabelsRun) -> None:
    """A22/A25: ``p_target``/``p_stop``/``p_time`` por direccion y en las dos muestras."""
    scenarios = _dict(run.summary)
    assert set(scenarios) == {f"k_{value!r}" for value in R_SIGMA_SCENARIOS}
    for raw in scenarios.values():
        scenario = _dict(raw)
        assert set(scenario["directions"]) == {"long", "short"}
        long_all = _dict(_dict(scenario["directions"])["long"])
        assert set(long_all) == {"all", "full_only"}
        for sample_name in ("all", "full_only"):
            block = _dict(long_all[sample_name])
            fractions = (
                _number(block["p_target"]) + _number(block["p_stop"]) + _number(block["p_time"])
            )
            assert fractions == pytest.approx(1.0)
            assert _number(block["p_win"]) >= _number(block["p_target"])
            assert "p*" in str(block["p_target_note"])
    assert _number(
        _dict(_dict(_dict(_dict(scenarios["k_1.0"])["directions"])["long"])["all"])["sessions"]
    ) == len(run.rows)
    assert _number(
        _dict(_dict(_dict(_dict(scenarios["k_1.0"])["directions"])["long"])["full_only"])[
            "sessions"
        ]
    ) == _number(run.sample["labelled_full_sessions"])


def test_the_two_ways_of_computing_the_ev_agree(run: LabelsRun) -> None:
    """A23: ``p_win * E[G] - (1 - p_win) * E[P]`` == media de los retornos."""
    directions = _dict(_dict(_dict(run.summary)["k_1.0"])["directions"])
    for direction in ("long", "short"):
        block = _dict(_dict(directions[direction])["all"])
        assert (
            abs(_number(block["ev_identity"]) - _number(block["ev_media"])) <= EV_IDENTITY_TOLERANCE
        )
        assert _number(block["ev_identity_difference"]) <= EV_IDENTITY_TOLERANCE
        assert _number(block["p_win"]) >= _number(block["p_target"])
        # Las salidas `time` cuentan en las dos vias: hay etiquetas `time`.
        assert _number(_dict(block["counts"])[LABEL_TIME]) > 0


def test_the_cost_of_the_net_ev_is_declared_with_its_inconvenients(run: LabelsRun) -> None:
    """A24: ``c`` con procedencia, sensibilidad declarada y nunca fusionada."""
    cost = _dict(run.cost)
    assert cost["round_trip_spread"] == ROUND_TRIP_SPREAD
    assert cost["round_trip_spread"] == 0.000042
    assert "plan.md" in str(cost["provenance"]) and "#8" in str(cost["provenance"])
    assert cost["sensitivity_scenarios"] == COST_SENSITIVITY_SCENARIOS
    assert "nunca" in str(cost["sensitivity_note"])
    assert "#11" in str(cost["engine_cost_model"])
    long_all = _dict(_dict(_dict(_dict(run.summary)["k_1.0"])["directions"])["long"])["all"]
    block = _dict(long_all)
    assert _number(block["ev_neto"]) == pytest.approx(
        _number(block["ev_media"]) - ROUND_TRIP_SPREAD
    )
    sensitivity = _dict(_dict(block["bp"])["sensitivity_one_day"])
    assert set(sensitivity) == set(COST_SENSITIVITY_SCENARIOS)
    assert _number(sensitivity["long_one_day"]) == pytest.approx(
        (_number(block["ev_media"]) - COST_SENSITIVITY_SCENARIOS["long_one_day"]) * 10_000.0
    )


def test_the_returns_of_a_target_and_a_stop_are_the_barrier_distance(run: LabelsRun) -> None:
    """A21: con etiqueta ``target``/``stop`` la salida es el precio de la barrera."""
    targets = [
        row for row in run.rows if row.label_long == LABEL_TARGET and not row.intraday_incomplete
    ]
    assert targets, "el almacen sintetico tiene algun target ordenado con intradia"
    for row in targets:
        assert row.exit_long == pytest.approx(row.entry_px * (1.0 + row.target_pct))
        assert row.ret_long == pytest.approx(row.target_pct)
    stops = [
        row for row in run.rows if row.label_long == LABEL_STOP and not row.intraday_incomplete
    ]
    for row in stops:
        assert row.exit_long == pytest.approx(row.entry_px * (1.0 - row.stop_pct))
        assert row.ret_long == pytest.approx(-row.stop_pct)


# ─────────────────────────────────────────────────────────────────────────────
# Precio de entrada (A9, A10)
# ─────────────────────────────────────────────────────────────────────────────
def test_the_entry_price_sources_are_declared(run: LabelsRun) -> None:
    """A9/A10: registro de fuentes, con estado y procedencia, y la contradiccion declarada."""
    assert set(ENTRY_PRICE_SOURCES) == {DEFAULT_ENTRY_PRICE_SOURCE, ENTRY_PRICE_T0_SNAPSHOT}
    open_source = ENTRY_PRICE_SOURCES[DEFAULT_ENTRY_PRICE_SOURCE]
    t0_source = ENTRY_PRICE_SOURCES[ENTRY_PRICE_T0_SNAPSHOT]
    assert open_source.state == "available" and open_source.is_proxy is True
    assert t0_source.state == "unavailable" and t0_source.tradable is False
    assert t0_source.spot_et == "08:45"
    assert "2026-09-18" in t0_source.provenance

    entry = _dict(report_payload(run)["entry_price"])
    assert entry["owner_decision"] == "t0 a las 08:45 ET (snapshot congelado)"
    assert entry["decided_on"] == "2026-09-18"
    assert entry["provenance"] == "declaracion del usuario"
    assert entry["source_used"] == "session_open"
    assert entry["source_is_proxy"] is True
    assert entry["diverges_from_owner_decision"] is True
    assert entry["contradicts_plan_md_4_1"] is True
    assert entry["not_tradable"] is True
    assert entry["follow_up_issue"] == 61
    evidence = _dict(entry["evidence"])
    assert _number(evidence["bars_at_0845_et"]) == 0
    assert "13:30" in str(evidence["first_bar_utc"]) or evidence["first_bar_utc"] is None
    assert set(_dict(entry["registry"])) == set(ENTRY_PRICE_SOURCES)


def test_the_t0_entry_price_fails_declared_and_never_falls_back_silently(
    writable_root: Path,
) -> None:
    """A9: con ``t0_snapshot_0845_et`` el modulo declara el fallo y **no** produce etiquetas."""
    store = Store(writable_root)
    with pytest.raises(EntryPriceUnavailableError) as error:
        _ = label_history(store=store, now=NOW, entry_price_source=ENTRY_PRICE_T0_SNAPSHOT)
    assert error.value.state == "unavailable"
    assert "08:45" in error.value.reason
    assert store.datasets("derived") == []


def test_an_unknown_entry_price_source_is_rejected(writable_root: Path) -> None:
    """A9: solo las fuentes del registro declarado."""
    with pytest.raises(ConfigurationError):
        _ = label_history(
            store=Store(writable_root), now=NOW, entry_price_source="open_de_la_sesion"
        )


# ─────────────────────────────────────────────────────────────────────────────
# No *look-ahead* y sin overnight (A12, A31)
# ─────────────────────────────────────────────────────────────────────────────
def test_no_overnight_any_label_of_t_survives_a_hostile_open_in_t_plus_one(
    tmp_path: Path,
) -> None:
    """A12: el ``open`` de ``t+1`` no entra en ninguna columna de ``t``."""
    records = _records()
    hostile = [dict(record) for record in records]
    last = cast("float", hostile[-1]["open"])
    hostile[-1]["open"] = last * 1.5
    first = label_history(store=_build_store(tmp_path / "base"), now=NOW)
    second = label_history(
        store=_build_store(tmp_path / "hostile", records=hostile),
        now=NOW,
    )
    assert second.forecast_candidate == first.forecast_candidate
    changed_open = DAYS[-1]
    for row in first.rows:
        if row.session == changed_open:
            continue
        other = _row(second, row.session)
        assert other == row, f"la sesion {row.session} cambio al mover el open de {changed_open}"
    assert _row(second, changed_open).entry_px != _row(first, changed_open).entry_px


def test_appending_a_hostile_session_does_not_change_the_labels_of_the_past(
    tmp_path: Path,
) -> None:
    """A31: una sesion ``t+1`` hostil no cambia ni la sigma ni la etiqueta de ``t``."""
    records = _records()
    extra = dict(records[-1])
    next_day = _trading_days(1, start=DAYS[-1] + timedelta(days=1))[0]
    extra["as_of"] = _at(next_day)
    extra["fetched_at"] = NOW
    extra["open"] = cast("float", records[-1]["close"]) * 1.2
    extra["high"] = cast("float", records[-1]["close"]) * 2.0
    extra["low"] = cast("float", records[-1]["close"]) * 0.5
    extra["close"] = cast("float", records[-1]["close"]) * 1.9
    extra["adj_close"] = extra["close"]
    base = label_history(store=_build_store(tmp_path / "base"), now=NOW)
    extended = label_history(
        store=_build_store(tmp_path / "extended", records=[*records, extra]),
        now=NOW,
    )
    assert extended.forecast_candidate == base.forecast_candidate
    assert len(extended.rows) == len(base.rows) + 1
    for row in base.rows:
        assert _row(extended, row.session) == row
    assert extended.forecast_sha256 != base.forecast_sha256


def test_intraday_bars_of_a_later_session_do_not_change_the_labels_of_t(
    tmp_path: Path,
) -> None:
    """A31: el intradia de ``t+1`` no entra en la etiqueta de ``t`` (la parte intradia)."""
    records = _records()
    base_root = tmp_path / "base"
    hostile_root = tmp_path / "hostile"
    _build_store(base_root, records=records)
    hostile = _build_store(hostile_root, records=records)
    last = DAYS[-1]
    info = EASTERN.session(last)
    assert info.is_half_day is False, "la ultima sesion sintetica es completa"
    # La ultima sesion pasa a tener intradia hostil: antes no tenia ninguna barra.
    hostile.append("raw", "market_intraday", _intraday_records(last, 78, drift=-0.06))

    base = label_history(store=Store(base_root), now=NOW)
    extended = label_history(store=Store(hostile_root), now=NOW)
    assert _row(base, last).order_source == FALLBACK_ORDER_SOURCE
    assert _row(extended, last).order_source == INTRADAY_ORDER_SOURCE
    assert _row(extended, last).bars_observed == 78
    assert _row(extended, last).intraday_incomplete is False
    # El intradia hostil de esa sesion cambia **su** etiqueta (baja => `stop`),
    # pero no la de ninguna sesion anterior.
    assert _row(base, last).label_long == LABEL_TARGET
    assert _row(extended, last).label_long == LABEL_STOP
    # El walk-forward de #7 solo usa el diario: la sigma (y su hash) no cambia.
    assert extended.forecast_sha256 == base.forecast_sha256
    assert extended.forecast_candidate == base.forecast_candidate
    for row in base.rows:
        if row.session == last:
            continue
        assert _row(extended, row.session) == row


# ─────────────────────────────────────────────────────────────────────────────
# Persistencia (A26-A28)
# ─────────────────────────────────────────────────────────────────────────────
def test_the_labels_go_to_derived_labels_with_the_store_conventions(writable_root: Path) -> None:
    """A26: ``derived.labels`` por la API del ``Store``, ``as_of`` = cierre de sesion."""
    store = Store(writable_root)
    result = label_history(store=store, now=NOW)
    outputs = write_outputs(result, store=store, reports_dir=writable_root / "derived" / "reports")
    assert outputs.outcome == "created"
    assert outputs.sessions == len(result.rows)

    frame = store.sql("SELECT * FROM derived.labels ORDER BY session")
    assert frame.height == len(result.rows)
    assert frame.get_column("series_id").unique().to_list() == [SERIES_ID]
    assert frame.get_column("source").unique().to_list() == [LABELS_SOURCE]
    sessions = frame.get_column("session").to_list()
    assert len(sessions) == len(set(sessions)), "una fila por (series_id, session)"
    close_utc = frame.get_column("as_of").to_list()
    published = frame.get_column("published_at").to_list()
    assert close_utc == published, "la etiqueta solo se conoce al cerrar: published_at = as_of"
    for session, moment in zip(sessions, close_utc, strict=True):
        info = EASTERN.session(cast("date", session))
        assert info.close_utc == moment
    assert frame.get_column("version").unique().to_list() == [1]
    assert LABELS_DATASET == "labels"
    assert (writable_root / "derived" / "labels").is_dir()


def test_a_session_that_has_not_closed_is_not_written(writable_root: Path) -> None:
    """A26: ninguna sesion con el cierre posterior a ``now`` se escribe."""
    store = Store(writable_root)
    early = datetime(2009, 6, 1, 12, 0, tzinfo=UTC)
    result = label_history(store=store, now=early)
    assert _number(_dict(_dict(result.sample["reasons"])[REASON_SESSION_NOT_CLOSED])["count"]) > 0
    outputs = write_outputs(result, store=store, reports_dir=writable_root / "derived" / "reports")
    assert outputs.sessions == len(result.rows)
    frame = store.sql("SELECT max(as_of) AS last FROM derived.labels")
    last = frame.get_column("last").to_list()[0]
    assert isinstance(last, datetime) and last <= early
    for row in result.rows:
        assert row.close_utc <= early


def test_an_inconclusive_selection_is_a_declared_error_and_writes_nothing(
    writable_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A2/A32: seleccion `inconclusive` => error declarado y **nada** escrito."""

    def inconclusive(candidates: object) -> Selection:
        return Selection(
            selected=None,
            verdict="inconclusive",
            metric="qlike",
            rule=(),
            constants={},
            arithmetic=(),
            leaders={"parkinson": "har", "ret_sq": "garch"},
        )

    monkeypatch.setattr("cfdtrader.models.labels.select_candidate", inconclusive)
    store = Store(writable_root)
    with pytest.raises(LabelsError):
        _ = label_and_write(
            data_root=writable_root,
            reports_dir=writable_root / "derived" / "reports",
            now=NOW,
        )
    assert store.datasets("derived") == []
    assert not (writable_root / "derived" / "reports").exists()


def test_a_second_identical_run_is_a_no_op_and_another_k_supersedes(writable_root: Path) -> None:
    """A27: idempotencia, *supersede* con ``version`` y ningun fichero borrado."""
    store = Store(writable_root)
    reports = writable_root / "derived" / "reports"
    first = label_and_write(
        data_root=writable_root, reports_dir=reports, now=NOW, k_sigma=PERSISTED_K_SIGMA
    )
    files_before = sorted((writable_root / "derived" / "labels").rglob("*.parquet"))
    assert first[1].outcome == "created"

    def row_count() -> float:
        frame = store.sql("SELECT count(*) AS rows FROM derived.labels")
        return _number(frame.get_column("rows").to_list()[0])

    expected = row_count()

    second = label_and_write(
        data_root=writable_root, reports_dir=reports, now=NOW, k_sigma=PERSISTED_K_SIGMA
    )
    assert second[1].outcome == "unchanged"
    assert sorted((writable_root / "derived" / "labels").rglob("*.parquet")) == files_before
    assert row_count() == expected

    third = label_and_write(data_root=writable_root, reports_dir=reports, now=NOW, k_sigma=1.5)
    assert third[1].outcome == "created"
    assert row_count() == expected, "la vista devuelve una sola fila vigente por sesion"
    versions = (
        store.sql("SELECT DISTINCT version AS version FROM derived.labels")
        .get_column("version")
        .to_list()
    )
    assert versions == [2]
    files_after = sorted((writable_root / "derived" / "labels").rglob("*.parquet"))
    assert set(files_before) < set(files_after), "nada se borra: los ficheros viejos siguen"
    current = (
        store.sql("SELECT DISTINCT k_sigma AS k FROM derived.labels").get_column("k").to_list()
    )
    assert current == [1.5]


def test_every_row_declares_the_forecast_that_produced_it(writable_root: Path) -> None:
    """A28: candidato, veredicto, ``k`` y ``sha256`` de la serie de sigma usada."""
    store = Store(writable_root)
    result = label_history(store=store, now=NOW)
    write_outputs(result, store=store, reports_dir=writable_root / "derived" / "reports")
    frame = store.sql("SELECT * FROM derived.labels")
    columns = set(frame.columns)
    assert {"forecast_candidate", "selection_verdict", "k_sigma", "forecast_sha256"} <= columns
    assert frame.get_column("forecast_sha256").unique().to_list() == [result.forecast_sha256]
    assert frame.get_column("k_sigma").unique().to_list() == [PERSISTED_K_SIGMA]
    assert frame.get_column("forecast_candidate").unique().to_list() == [result.forecast_candidate]

    # Un cambio de sigma cambia el hash y el contenido: la invalidacion es detectable.
    other = _records(seed=77)
    other_root = writable_root.parent / "other"
    _build_store(other_root, records=other)
    other_run = label_history(store=Store(other_root), now=NOW)
    assert other_run.forecast_sha256 != result.forecast_sha256
    different = [
        row for row in other_run.rows if _row(result, row.session).ret_long != row.ret_long
    ]
    assert different, "cambiar la sigma cambia el contenido de las filas"


def test_the_forecast_hash_is_the_hash_of_the_sigma_series_used(run: LabelsRun) -> None:
    """A28: el hash se puede recalcular a mano desde la serie (sesion, sigma)."""
    expected = hashlib.sha256(
        "\n".join(f"{row.session.isoformat()},{row.sigma!r}" for row in run.rows).encode("utf-8")
    ).hexdigest()
    assert run.forecast_sha256 == expected


# ─────────────────────────────────────────────────────────────────────────────
# Determinismo, CLI y limitaciones (A30, A32, A33)
# ─────────────────────────────────────────────────────────────────────────────
def test_two_runs_with_the_same_now_produce_the_same_json_and_the_same_rows(
    tmp_path: Path, synthetic_root: Path
) -> None:
    """A30: mismo ``--now`` => JSON identico byte a byte y mismas filas."""
    digests: list[str] = []
    frames: list[list[dict[str, Any]]] = []
    for name in ("first", "second"):
        root = tmp_path / name
        shutil.copytree(synthetic_root, root)
        store = Store(root)
        reports = root / "derived" / "reports"
        _run, outputs = label_and_write(
            data_root=root, reports_dir=reports, now=NOW, k_sigma=PERSISTED_K_SIGMA
        )
        digests.append(hashlib.sha256(outputs.json_path.read_bytes()).hexdigest())
        frames.append(store.sql("SELECT * FROM derived.labels ORDER BY session").to_dicts())
    assert digests[0] == digests[1]
    assert frames[0] == frames[1]
    assert len(frames[0]) > 0


def test_the_cli_writes_the_report_and_the_dataset(tmp_path: Path, synthetic_root: Path) -> None:
    """A32: codigo 0, informe ``triple_barrier_<fecha>`` y dataset en la raiz pedida."""
    root = tmp_path / "data"
    shutil.copytree(synthetic_root, root)
    assert main(["--data-root", str(root), "--now", NOW.isoformat()]) == 0
    reports = root / "derived" / "reports"
    assert (reports / f"triple_barrier_{NOW.date().isoformat()}.json").is_file()
    assert (reports / f"triple_barrier_{NOW.date().isoformat()}.md").is_file()
    store = Store(root)
    assert LABELS_DATASET in store.datasets("derived")
    assert store.sql("SELECT count(*) AS rows FROM derived.labels").height == 1
    assert store.datasets("raw") == ["market_daily", "market_intraday"]


def test_the_cli_exits_with_2_and_writes_nothing_when_the_entry_price_is_missing(
    tmp_path: Path, synthetic_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A32/A9: falta el precio pedido => salida 2, motivo por ``stderr`` y nada escrito."""
    root = tmp_path / "data"
    shutil.copytree(synthetic_root, root)
    code = main(
        [
            "--data-root",
            str(root),
            "--now",
            NOW.isoformat(),
            "--entry-price-source",
            ENTRY_PRICE_T0_SNAPSHOT,
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "08:45" in captured.err
    assert not (root / "derived").exists()
    assert Store(root).datasets("derived") == []


def test_the_cli_accepts_the_declared_k(tmp_path: Path, synthetic_root: Path) -> None:
    """A32: ``--k`` elige el escenario que se persiste (A5)."""
    root = tmp_path / "data"
    shutil.copytree(synthetic_root, root)
    assert main(["--data-root", str(root), "--now", NOW.isoformat(), "--k", "1.5"]) == 0
    stored = Store(root).sql("SELECT DISTINCT k_sigma AS k FROM derived.labels").get_column("k")
    assert stored.to_list() == [1.5]


def test_the_limitations_are_declared_with_the_measured_numbers(run: LabelsRun) -> None:
    """A33: las limitaciones van en el JSON y en el ``.md``, con sus issues."""
    payload = report_payload(run)
    limitations = cast("list[str]", payload["limitations"])
    assert isinstance(limitations, list) and len(limitations) >= 7
    joined = " ".join(limitations)
    for expected in ("#50", "#57", "#61", "#11", "#52", "#60", "cota inferior", "media"):
        assert expected in joined, f"falta {expected!r} en las limitaciones"
    markdown = render_markdown(run)
    assert "## Limitaciones (declaradas, no escondidas)" in markdown
    for item in limitations:
        assert item in markdown
    sigma = _dict(run.inputs["sigma"])
    assert _number(sigma["warmup_sessions"]) == MIN_TRAIN
    assert MIN_TRAIN == 500


def test_the_report_declares_the_persistence_conventions(run: LabelsRun) -> None:
    """A26/A27: el informe publica grano, convenciones y semantica de recomputacion."""
    persistence = _dict(report_payload(run)["persistence"])
    fields = {
        "layer": persistence["layer"],
        "dataset": persistence["dataset"],
        "source": persistence["source"],
        "sessions": persistence["sessions_persisted"],
        "k": persistence["k_sigma_persisted"],
    }
    assert fields["layer"] == "derived"
    assert fields["dataset"] == LABELS_DATASET
    assert fields["source"] == LABELS_SOURCE
    assert _number(fields["sessions"]) == len(run.rows)
    assert _number(fields["k"]) == PERSISTED_K_SIGMA
    assert "(series_id, session)" in str(persistence["grain"])
    assert "read_pit" in str(persistence["verification"])


def test_the_markdown_states_the_semantics_and_the_divergence(run: LabelsRun) -> None:
    """A1/A6/A10/A16/A33: el ``.md`` dice lo mismo que el JSON, en prosa."""
    markdown = render_markdown(run)
    for text in (
        "favorable",
        "adversa",
        "sin redondear",
        "no del CFD",
        "`not_tradable` = `true`",
        "`diverges_from_owner_decision` = `true`",
        "`contradicts_plan_md_4_1` = `true`",
        "#61",
        "cota inferior",
        "13:00 ET",
        "16:00 ET",
    ):
        assert text in markdown, f"falta {text!r} en el informe legible"


def test_a_custom_k_is_published_as_its_own_scenario(writable_root: Path) -> None:
    """A5/A25: el escenario persistido se publica aunque no este en la lista declarada."""
    result = label_history(store=Store(writable_root), now=NOW, k_sigma=1.25)
    assert set(result.summary) == {*(f"k_{value!r}" for value in R_SIGMA_SCENARIOS), "k_1.25"}
    assert all(row.k_sigma == 1.25 for row in result.rows)
    doubled = label_history(store=Store(writable_root), now=NOW, k_sigma=2.5)
    first = result.rows[0]
    other = _row(doubled, first.session)
    assert other.sigma == first.sigma
    assert other.target_pct == pytest.approx(2 * first.target_pct)


def test_the_run_declares_the_follow_up_issues_of_the_limitations(run: LabelsRun) -> None:
    """A17/A33: ``#50`` (fuente intradia) y ``#57`` (RV intradia) aparecen declaradas."""
    coverage = _dict(run.inputs["intraday_coverage"])
    assert coverage["follow_up_issues"] == [50, 57]
    unused = _dict(_dict(coverage["unused_series"])["ES=F"])
    assert "A13" in str(unused["reason"])
    assert "solo con" in str(unused["reason"])


# ─────────────────────────────────────────────────────────────────────────────
# Almacen real (solo lectura; se salta si no hay datos)
# ─────────────────────────────────────────────────────────────────────────────
REPOSITORY_DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.skipif(
    not (REPOSITORY_DATA / "raw" / "market_daily").is_dir(),
    reason="no hay almacen real en data/ (es gitignored)",
)
def test_the_real_store_reproduces_the_declared_coverage() -> None:
    """A17/A33 sobre el almacen real, **sin escribir nada** (solo lectura)."""
    result = label_history(store=Store(REPOSITORY_DATA), now=datetime(2026, 9, 18, tzinfo=UTC))
    coverage = _dict(result.inputs["intraday_coverage"])
    sigma = _dict(result.inputs["sigma"])
    assert _number(coverage["sessions"]) == 60
    assert "2026-06-24" in str(coverage["first_bar_utc"])
    assert _number(coverage["bars_at_0845_et"]) == 0
    assert _number(coverage["stored_daily_sessions"]) == 5460
    assert _number(_dict(_dict(result.sample["reasons"])[REASON_NO_FORECAST])["count"]) == 505
    assert _number(sigma["warmup_sessions"]) == MIN_TRAIN
    assert result.forecast_candidate == "garch"
    assert 0.97 < _number(result.sample["fallback_share"]) < 0.99
    assert _number(result.sample["labelled"]) > 2600
    assert len(result.limitations) >= 7
    assert _number(_dict(result.inputs["clean_sample"])["clean_sessions"]) == 3192
