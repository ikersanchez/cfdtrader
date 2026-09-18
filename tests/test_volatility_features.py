"""Tests de las features de volatilidad (tarea #7).

Todo se construye **en memoria** (A11): el módulo es puro, entra un ``pl.DataFrame``
y sale otro. Los tests que más importan son el *golden* calculado a mano (A4) y el
de **no look-ahead** (A10): que la sesión ``t`` no cambie al añadir datos de ``t+1``.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from cfdtrader.features import volatility

#: Columnas de feature de las que se exige inmunidad al futuro y al `open`.
FEATURE_COLUMNS = (
    "true_range",
    "atr_norm",
    "parkinson_rv",
    "ret_log",
    "ret_sq",
    "har_lag1",
    "har_lag4",
    "har_lag17",
    "har_forecast",
    "vix_level",
    "vix_zscore",
    "vix_percentile",
)

#: Columnas que NO leen el `open` y por tanto son inmunes al artefacto #52.
OPEN_BLIND_COLUMNS = ("true_range", "atr_norm", "parkinson_rv")


def _sessions(count: int, start: date = date(2015, 1, 1)) -> list[date]:
    """Los primeros ``count`` días laborables desde ``start``."""
    days: list[date] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _frame(
    closes: list[float], *, highs: list[float] | None = None, lows: list[float] | None = None
) -> pl.DataFrame:
    """Frame OHLC con cierres dados; ``high``/``low`` a ``close ± 1`` por defecto."""
    count = len(closes)
    return pl.DataFrame(
        {
            "session": _sessions(count),
            "open": [1000.0] * count,  # absurdo a propósito: no se debe leer
            "high": highs if highs is not None else [close + 1.0 for close in closes],
            "low": lows if lows is not None else [close - 1.0 for close in closes],
            "close": closes,
        }
    )


def _synthetic(count: int, *, seed: int = 3, with_vix: bool = True) -> pl.DataFrame:
    """Frame realista: precios y volatilidad variables, con VIX opcional."""
    import numpy as np

    rng = np.random.default_rng(seed)
    log_vol = np.zeros(count)
    for index in range(1, count):
        log_vol[index] = 0.92 * log_vol[index - 1] + rng.normal(0.0, 0.1)
    step = rng.normal(0.0, 1.0, count) * 0.009 * np.exp(log_vol)
    close = 3000.0 * np.cumprod(1.0 + step)
    open_price = close * (1.0 + rng.normal(0.0, 0.003, count))
    high = np.maximum(close, open_price) * (1.0 + np.abs(step) * 0.5)
    low = np.minimum(close, open_price) * (1.0 - np.abs(step) * 0.5)
    frame = pl.DataFrame(
        {
            "session": _sessions(count),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        }
    )
    if with_vix:
        frame = frame.with_columns(
            pl.Series("vix_close", 18.0 * np.exp(log_vol * 0.8) + rng.normal(0.0, 0.4, count))
        )
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# A4 — ATR normalizado: golden calculado a mano
# ─────────────────────────────────────────────────────────────────────────────
def test_the_declared_default_window_is_fourteen_sessions() -> None:
    """La ventana por defecto está declarada y es de 14 sesiones (A4)."""
    assert volatility.ATR_WINDOW == 14


def test_golden_atr_with_the_default_window(tmp_path: Path) -> None:
    """ATR normalizado con la ventana por defecto, calculado a mano (A4).

    ``close`` alterna 100/110, ``high = close + 1`` y ``low = close - 1``::

        TR_0 = 2 (no hay cierre anterior)
        TR_i = max(2, |111 - 100|, |109 - 100|) = 11  para todo i >= 1

    Con ventana 14, ``atr_norm[14] = media(TR_0 … TR_13) / close_13
    = ((2 + 13·11) / 14) / 110`` y ``atr_norm[15] = 11 / 100``. Las 14 primeras
    sesiones quedan a ``NULL``: no hay ventana parcial.
    """
    closes = [100.0, 110.0] * 10
    frame = volatility.normalised_atr(volatility.true_range(_frame(closes)))
    atr = frame.get_column("atr_norm")

    assert atr.head(14).is_null().all()
    assert atr[14] == pytest.approx(0.09415584415584416, rel=1e-12)
    assert atr[15] == pytest.approx(0.11, rel=1e-12)
    assert frame.get_column("true_range")[0] == pytest.approx(2.0)
    assert frame.get_column("true_range")[1] == pytest.approx(11.0)


def test_golden_atr_with_an_explicit_window_of_three() -> None:
    """Con ventana 3 y un hueco de precio, los dos valores se comprueban a mano (A4)."""
    frame = volatility.normalised_atr(
        volatility.true_range(_frame([100.0, 100.0, 110.0, 100.0, 100.0])), window=3
    )
    atr = frame.get_column("atr_norm")

    assert atr.head(3).is_null().all()
    assert atr[3] == pytest.approx(5.0 / 110.0, rel=1e-12)
    assert atr[4] == pytest.approx(8.0 / 100.0, rel=1e-12)


def test_the_atr_window_must_be_usable() -> None:
    """Una ventana de una sesión no es un ATR: se rechaza en vez de devolver ruido."""
    with pytest.raises(ValueError, match="al menos 2"):
        volatility.normalised_atr(volatility.true_range(_frame([100.0, 101.0])), window=1)


# ─────────────────────────────────────────────────────────────────────────────
# A4/A5 — Parkinson: golden, inmunidad al `open` y caso `high == low`
# ─────────────────────────────────────────────────────────────────────────────
def test_golden_parkinson_variance_matches_the_declared_formula() -> None:
    """``rv = (ln(H/L))² / (4 ln 2)`` con H = 111 y L = 109 (A1, A4)."""
    frame = volatility.parkinson_variance(_frame([100.0], highs=[111.0], lows=[109.0]))
    assert frame.get_column("parkinson_rv")[0] == pytest.approx(0.0001192375647318803, rel=1e-9)


def test_the_atr_and_parkinson_do_not_read_the_open() -> None:
    """A5: sustituir el `open` por el cierre anterior (#52) no cambia ATR ni Parkinson.

    El True Range usa ``H``, ``L`` y ``C``; el Parkinson usa ``H`` y ``L``. Ninguno
    lee ``O``, así que son **inmunes por construcción** al artefacto #52. El test lo
    comprueba bit a bit, no «casi».
    """
    original = _synthetic(120)
    tampered = original.with_columns(pl.col("close").shift(1).alias("open"))
    tampered = tampered.with_columns(pl.col("open").fill_null(pl.col("close")))

    before = volatility.add_features(original)
    after = volatility.add_features(tampered)

    for column in OPEN_BLIND_COLUMNS:
        assert before.get_column(column).to_list() == after.get_column(column).to_list(), column
    # El objetivo secundario sí lee el `open`: tiene que notarse que el artefacto existe.
    assert before.get_column("ret_sq").to_list() != after.get_column("ret_sq").to_list()


def test_parkinson_is_zero_when_high_equals_low() -> None:
    """A5: sin rango no hay varianza, y es 0 exacto, nunca `NaN`."""
    frame = volatility.parkinson_variance(
        volatility.true_range(_frame([100.0, 100.0, 100.0], highs=[100.0] * 3, lows=[100.0] * 3))
    )
    values = frame.get_column("parkinson_rv").to_list()

    assert values == [0.0, 0.0, 0.0]


# ─────────────────────────────────────────────────────────────────────────────
# A6 — HAR: retardos exactos y calentamiento a NULL
# ─────────────────────────────────────────────────────────────────────────────
def test_the_har_lag_constants_are_the_declared_ones() -> None:
    """Los retardos son 1, 4 y 17 sesiones y se exportan como constantes (A6)."""
    assert volatility.HAR_LAG_DAILY == 1
    assert volatility.HAR_LAG_WEEKLY == 4
    assert volatility.HAR_LAG_MONTHLY == 17
    assert volatility.HAR_WARMUP == 22


def test_the_first_twenty_two_sessions_have_null_regressors_and_forecast() -> None:
    """A6: las 22 primeras sesiones quedan a ``NULL``, no a 0 ni interpoladas."""
    frame = volatility.add_features(_synthetic(60))

    for column in ("har_lag1", "har_lag4", "har_lag17", "har_forecast"):
        assert frame.get_column(column).head(volatility.HAR_WARMUP).is_null().all(), column
    assert frame.get_column("har_lag17")[volatility.HAR_WARMUP] is not None


def test_the_har_lags_are_exactly_the_declared_windows() -> None:
    """Los tres retardos son media aritmética de ``rv`` de las ventanas declaradas (A6).

    Se comprueban a mano con una serie de ``rv`` conocida: ``har_lag1 = rv_{t-1}``,
    ``har_lag4 = media(rv_{t-2} … rv_{t-5})`` y ``har_lag17 = media(rv_{t-6} … t-22)``.
    """
    count = 40
    rv = [float(index + 1) for index in range(count)]  # rv_i = i + 1
    frame = pl.DataFrame({"session": _sessions(count)}).with_columns(pl.Series("parkinson_rv", rv))
    frame = volatility.har_regressors(frame)
    row = count - 1  # el último: todas las ventanas están completas

    assert frame.get_column("har_lag1")[row] == pytest.approx(rv[row - 1])
    assert frame.get_column("har_lag4")[row] == pytest.approx(sum(rv[row - 5 : row - 1]) / 4)
    assert frame.get_column("har_lag17")[row] == pytest.approx(sum(rv[row - 22 : row - 5]) / 17)


# ─────────────────────────────────────────────────────────────────────────────
# A7 — HAR en log-varianza: siempre positivo y NULL si no se puede ajustar
# ─────────────────────────────────────────────────────────────────────────────
def test_har_forecasts_are_finite_and_strictly_positive() -> None:
    """El HAR en ``ln(rv)`` da varianza positiva por construcción (A7)."""
    import numpy as np

    frame = volatility.add_features(_synthetic(400))
    values = frame.get_column("har_forecast").drop_nulls().to_numpy()

    assert values.size > 100
    assert bool(np.isfinite(values).all())
    assert bool((values > 0.0).all())


def test_the_har_forecast_is_null_and_not_nan_when_the_fit_is_impossible() -> None:
    """A7: con precio constante el ajuste no es posible y el forecast es ``NULL``.

    Nunca ``0``, ``inf`` ni ``NaN``. Es la prueba de que un ajuste degenerado no
    produce un número inventado.
    """
    frame = volatility.add_features(
        _synthetic(80, with_vix=False).with_columns(
            pl.lit(100.0).alias("open"),
            pl.lit(100.0).alias("high"),
            pl.lit(100.0).alias("low"),
            pl.lit(100.0).alias("close"),
        )
    )
    forecast = frame.get_column("har_forecast")

    assert forecast.is_null().all()
    assert forecast.is_nan().sum() == 0


def test_fit_log_har_rejects_degenerate_designs() -> None:
    """Un rango degenerado no produce coeficientes: ``None`` (A7)."""
    import numpy as np

    target = np.linspace(-10.0, -8.0, 40)
    constant = np.ones((40, 3))
    assert volatility.fit_log_har(constant, target) is None  # columnas redundantes
    assert volatility.fit_log_har(np.ones((3, 3)), target[:3]) is None  # pocas filas
    assert volatility.fit_log_har(np.ones((40, 3)) * np.nan, target) is None


# ─────────────────────────────────────────────────────────────────────────────
# A9 — VIX como feature de régimen
# ─────────────────────────────────────────────────────────────────────────────
def test_the_vix_feature_uses_only_the_previous_session() -> None:
    """A9: la feature de la sesión ``t`` es el cierre del VIX de ``t-1``."""
    frame = volatility.add_features(_synthetic(300))
    vix = frame.get_column("vix_close").to_list()
    level = frame.get_column("vix_level").to_list()

    assert level[0] is None
    for index in range(1, 20):
        assert level[index] == pytest.approx(vix[index - 1])


def test_the_vix_normalisation_needs_two_hundred_and_fifty_sessions() -> None:
    """A9: ventana expandida con mínimo de 250 sesiones, nunca la muestra completa.

    La primera sesión con 250 niveles **anteriores** es la 251 (el nivel de la
    sesión 0 no existe): la sesión 250 sigue a ``NULL``, que es la lectura
    conservadora (el nivel que se normaliza no entra en sus propias estadísticas).
    """
    frame = volatility.add_features(_synthetic(300))
    minimum = volatility.VIX_MIN_SESSIONS

    assert frame.get_column("vix_zscore").head(minimum + 1).is_null().all()
    assert frame.get_column("vix_percentile").head(minimum + 1).is_null().all()
    assert frame.get_column("vix_zscore")[minimum + 1] is not None
    assert frame.get_column("vix_percentile")[minimum + 1] is not None
    percentiles = frame.get_column("vix_percentile").drop_nulls().to_list()
    assert all(0.0 <= value <= 1.0 for value in percentiles)


def test_the_vix_percentile_counts_the_history_before_the_session() -> None:
    """El percentil es la fracción de niveles anteriores que no superan al actual (A9)."""
    frame = volatility.vix_features(
        pl.DataFrame(
            {
                "session": _sessions(6),
                "vix_close": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        ),
        min_sessions=3,
    )
    percentiles = frame.get_column("vix_percentile").to_list()

    assert percentiles[:4] == [None, None, None, None]  # hacen falta 3 de historia
    assert percentiles[4] == pytest.approx(1.0)  # 40 supera a 10, 20 y 30
    assert percentiles[5] == pytest.approx(1.0)  # 50 supera a 10, 20, 30 y 40


# ─────────────────────────────────────────────────────────────────────────────
# A10 — No look-ahead: el test más valioso
# ─────────────────────────────────────────────────────────────────────────────
def test_no_feature_changes_when_a_later_session_is_appended() -> None:
    """A10: añadir una sesión ``t+1`` con precios absurdos (±50 %) no toca ``t``.

    Se compara columna a columna y valor a valor (bit a bit): si alguna feature se
    calculara con datos posteriores, cambiaría.
    """
    count = 300
    frame = _synthetic(count)
    before = volatility.add_features(frame)

    last_close = float(frame.get_column("close")[-1])
    absurd = pl.DataFrame(
        {
            "session": [frame.get_column("session")[-1] + timedelta(days=1)],
            "open": [last_close * 0.5],
            "high": [last_close * 1.5],
            "low": [last_close * 0.5],
            "close": [last_close * 1.5],
            "vix_close": [float(frame.get_column("vix_close")[-1]) * 3.0],
        }
    )
    after = volatility.add_features(pl.concat([frame, absurd]).sort("session"))

    for column in FEATURE_COLUMNS:
        expected = before.get_column(column).to_list()
        actual = after.get_column(column).head(count).to_list()
        assert actual == expected, f"la feature {column} cambió al añadir t+1"
    # La sesión absurda sí aparece: el frame crece, pero no hacia atrás.
    assert after.height == count + 1


# ─────────────────────────────────────────────────────────────────────────────
# A11 — El módulo es puro
# ─────────────────────────────────────────────────────────────────────────────
def test_the_module_computes_from_an_in_memory_frame() -> None:
    """A11: entra un ``pl.DataFrame`` y sale otro, sin almacén ni sistema de ficheros."""
    frame = _synthetic(60)
    result = volatility.add_features(frame)

    assert isinstance(result, pl.DataFrame)
    assert result.height == frame.height
    assert set(FEATURE_COLUMNS).issubset(set(result.columns))


def test_the_module_has_no_io_or_storage_imports() -> None:
    """A11: el módulo no importa el almacén, DuckDB, `httpx` ni toca ficheros."""
    lines = Path(volatility.__file__).read_text(encoding="utf-8").splitlines()
    imports = [line for line in lines if line.startswith(("import ", "from "))]
    forbidden = ("cfdtrader.data", "duckdb", "httpx", "pathlib", "os", "shutil", "socket")

    assert imports, "el módulo debe declarar sus imports de forma explícita"
    for line in imports:
        root = line.split()[1].split(".")[0]
        assert root not in forbidden, f"import prohibido en el módulo puro: {line!r}"
