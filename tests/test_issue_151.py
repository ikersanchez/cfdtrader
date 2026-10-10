"""#151 — el `as_of` diario es el cierre del mercado de la serie, no el de EE. UU.

El bug: `frames.session_close_utc` estampaba **toda** barra diaria con las 16:00 ET
del día, así que a las 08:45 ET del día `t` el cierre asiático de `t` (cerró de
madrugada) quedaba fuera por `as_of > now`. Estas pruebas fijan la disponibilidad
por mercado, que es la base de datos que necesita #147.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from cfdtrader.data.settings import ConfigurationError
from cfdtrader.data.sources import yfinance_adapter
from cfdtrader.data.sources.base import AssetClass, SeriesSpec
from cfdtrader.data.sources.frames import market_close_utc
from cfdtrader.data.sources.registry import load_registry

EASTERN = ZoneInfo("America/New_York")

#: El instante del snapshot de decisión: 08:45 ET del 2026-10-09.
SNAPSHOT = datetime(2026, 10, 9, 8, 45, tzinfo=EASTERN).astimezone(UTC)

#: La sesión que se decide y las barras que su snapshot debe poder leer.
SESSION = date(2026, 10, 9)

#: Cierres declarados por mercado (#151): los cinco índices ajenos a EE. UU.
FOREIGN_CLOSES: dict[str, tuple[time, str]] = {
    "^GDAXI": (time(17, 30), "Europe/Berlin"),
    "^FTSE": (time(16, 30), "Europe/London"),
    "^STOXX50E": (time(17, 30), "Europe/Berlin"),
    "^N225": (time(15, 0), "Asia/Tokyo"),
    "^HSI": (time(16, 0), "Asia/Hong_Kong"),
}

#: El cierre por defecto: EE. UU.
US_CLOSE = (time(16, 0), "America/New_York")


def _specs_by_id() -> dict[str, SeriesSpec]:
    return {spec.series_id: spec for spec in load_registry().series}


def _as_of(spec: SeriesSpec, day: date) -> datetime:
    return market_close_utc(day, at_local=spec.close_local, timezone=spec.close_timezone)


# ─────────────────────────────────────────────────────────────────────────────
# C2 — cierre por mercado declarado
# ─────────────────────────────────────────────────────────────────────────────
def test_c2_the_registry_declares_a_market_close_per_series() -> None:
    """Cada serie declara su cierre; los índices ajenos, el suyo; el resto, EE. UU."""
    by_id = _specs_by_id()
    for series_id, (close_local, zone) in FOREIGN_CLOSES.items():
        assert series_id in by_id, f"falta {series_id} en el registro"
        spec = by_id[series_id]
        assert (spec.close_local, spec.close_timezone) == (close_local, zone)
    # Una serie americana no declara cierre: el por defecto es el de EE. UU.
    assert (by_id["^GSPC"].close_local, by_id["^GSPC"].close_timezone) == US_CLOSE
    assert (by_id["^VIX"].close_local, by_id["^VIX"].close_timezone) == US_CLOSE


def test_c2_a_series_without_the_field_defaults_to_the_us_close() -> None:
    """El valor por defecto del spec es el cierre de EE. UU. (16:00 ET)."""
    spec = SeriesSpec(
        series_id="^TEST",
        dataset="market_daily",
        asset_class=AssetClass.INDEX,
        granularity="daily",
        interval="1d",
        primary="fake",
    )
    assert (spec.close_local, spec.close_timezone) == US_CLOSE


def test_c2_a_non_iana_timezone_is_a_typed_error(tmp_path: Path) -> None:
    """Una zona que no existe es `ConfigurationError`, nunca un cierre silencioso."""
    registry_path = tmp_path / "sources.yaml"
    registry_path.write_text(
        "version: 1\n"
        "series:\n"
        '  - series_id: "^TEST"\n'
        "    dataset: market_daily\n"
        "    asset_class: index\n"
        "    granularity: daily\n"
        '    interval: "1d"\n'
        "    primary: yfinance\n"
        '    close_timezone: "Not/AZone"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="Not/AZone"):
        load_registry(registry_path)


# ─────────────────────────────────────────────────────────────────────────────
# C3 — el `as_of` es el cierre del mercado, y la fecha ET no cambia
# ─────────────────────────────────────────────────────────────────────────────
def test_c3_the_as_of_is_the_market_close_and_the_session_does_not_move() -> None:
    """El cierre de cada mercado es UTC correcto; su fecha ET sigue siendo `t`."""
    by_id = _specs_by_id()
    expected_utc = {
        "^N225": datetime(2026, 10, 9, 6, 0, tzinfo=UTC),
        "^HSI": datetime(2026, 10, 9, 8, 0, tzinfo=UTC),
        "^GDAXI": datetime(2026, 10, 9, 15, 30, tzinfo=UTC),
        "^FTSE": datetime(2026, 10, 9, 15, 30, tzinfo=UTC),
        "^STOXX50E": datetime(2026, 10, 9, 15, 30, tzinfo=UTC),
        "^GSPC": datetime(2026, 10, 9, 20, 0, tzinfo=UTC),
    }
    for series_id, expected in expected_utc.items():
        stamp = _as_of(by_id[series_id], SESSION)
        assert stamp == expected, series_id
        # `session` (la fecha ET del `as_of`) sigue siendo el día de la barra.
        assert stamp.astimezone(EASTERN).date() == SESSION, series_id


def test_c3_the_stamp_is_dst_aware_not_a_fixed_offset() -> None:
    """Tokio cierra a las 06:00 UTC siempre; el `as_of` en ET cambia con el DST de EE. UU."""
    spec = SeriesSpec(
        series_id="^N225",
        dataset="market_daily",
        asset_class=AssetClass.INDEX,
        granularity="daily",
        interval="1d",
        primary="yfinance",
        close_local=time(15, 0),
        close_timezone="Asia/Tokyo",
    )
    summer = _as_of(spec, date(2026, 7, 1))
    winter = _as_of(spec, date(2026, 1, 5))
    assert summer.hour == 6 and winter.hour == 6  # el cierre local no se mueve
    assert summer.astimezone(EASTERN).date() == date(2026, 7, 1)
    assert winter.astimezone(EASTERN).date() == date(2026, 1, 5)


# ─────────────────────────────────────────────────────────────────────────────
# C4/C5 — disponibilidad y sin look-ahead (golden 2026-10-09)
# ─────────────────────────────────────────────────────────────────────────────
def test_c4_at_the_snapshot_asia_is_available_and_the_us_and_europe_are_not() -> None:
    """C4: a las 08:45 ET del 9-oct, Asia del 9-oct ya está; US y Europa del 9-oct no."""
    by_id = _specs_by_id()
    available = {series_id: _as_of(spec, SESSION) <= SNAPSHOT for series_id, spec in by_id.items()}
    assert available["^N225"] is True
    assert available["^HSI"] is True
    for closed_later in ("^GSPC", "^VIX", "^GDAXI", "^FTSE", "^STOXX50E"):
        assert available[closed_later] is False, closed_later


def test_c5_the_filter_never_keeps_a_bar_that_has_not_closed() -> None:
    """C5: `as_of <= now` conserva exactamente las barras ya cerradas de la sesión."""
    by_id = _specs_by_id()
    kept = {sid for sid, spec in by_id.items() if _as_of(spec, SESSION) <= SNAPSHOT}
    assert kept == {"^N225", "^HSI"}


# ─────────────────────────────────────────────────────────────────────────────
# El adaptador usa el cierre del spec (no el de EE. UU.)
# ─────────────────────────────────────────────────────────────────────────────
def test_the_adapter_stamps_a_foreign_daily_bar_with_its_own_close() -> None:
    """`yfinance_adapter._as_of` ancla la barra diaria al cierre del mercado del spec."""
    spec = SeriesSpec(
        series_id="^N225",
        dataset="market_daily",
        asset_class=AssetClass.INDEX,
        granularity="daily",
        interval="1d",
        primary="yfinance",
        close_local=time(15, 0),
        close_timezone="Asia/Tokyo",
    )
    stamp = yfinance_adapter._as_of(pd.Timestamp("2026-10-09"), daily=True, spec=spec)
    assert stamp == datetime(2026, 10, 9, 6, 0, tzinfo=UTC)


def test_the_adapter_leaves_the_us_series_at_the_us_close() -> None:
    """Una serie sin cierre declarado conserva el ancla de EE. UU. (16:00 ET)."""
    spec = SeriesSpec(
        series_id="^GSPC",
        dataset="market_daily",
        asset_class=AssetClass.INDEX,
        granularity="daily",
        interval="1d",
        primary="yfinance",
    )
    stamp = yfinance_adapter._as_of(pd.Timestamp("2026-10-09"), daily=True, spec=spec)
    assert stamp == datetime(2026, 10, 9, 20, 0, tzinfo=UTC)
