"""Tests de los resultados de mega-caps (#126, parte (c) de #114).

Ningún test abre red: la fuente se **simula** y el adaptador de ``yfinance`` se prueba con una
factoría de *tickers* de mentira. El artefacto declarado vive en ``config/mega_caps.yaml``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import ClassVar

import pytest

from cfdtrader.data.earnings import (
    EarningsCertainty,
    EarningsMoment,
    MegaCapIssuer,
    MegaCapsConfig,
    RawEarnings,
    earnings_on,
    ingest,
    load_mega_caps,
)
from cfdtrader.data.settings import ConfigurationError
from cfdtrader.data.sources.yfinance_adapter import YFinanceEarningsAdapter
from cfdtrader.data.store import Store

#: Instante de referencia declarado (nunca del reloj).
NOW = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)


class _FakeSource:
    """Una fuente de resultados simulada: fechas por emisor, sin red."""

    name = "fake"

    def __init__(self, by_symbol: dict[str, tuple[tuple[date, EarningsMoment], ...]]) -> None:
        self._by_symbol = by_symbol

    def earnings_dates(self, symbol: str, *, now: datetime) -> tuple[RawEarnings, ...]:
        return tuple(
            RawEarnings(symbol=symbol, on=on, moment=moment, observed_at=now)
            for on, moment in self._by_symbol.get(symbol, ())
        )


def _config(*, confirmed: bool = False) -> MegaCapsConfig:
    """Un registro de un solo emisor, con o sin una fecha confirmada declarada."""
    confirmed_earnings = ({"date": date(2026, 9, 17), "moment": "amc"},) if confirmed else ()
    issuer = MegaCapIssuer(symbol="NVDA", name="NVIDIA", confirmed_earnings=confirmed_earnings)
    return MegaCapsConfig(version=1, source="fake", declared_on=date(2026, 9, 1), issuers=(issuer,))


def test_126_the_issuer_list_is_a_declared_artifact() -> None:
    """La lista de emisores viene de `config/`, con su fecha de declaración (no del código)."""
    config = load_mega_caps()
    assert config.declared_on == date(2026, 10, 3)
    assert config.source == "yfinance"
    symbols = [issuer.symbol for issuer in config.issuers]
    assert "NVDA" in symbols and "AAPL" in symbols


def test_126_a_date_without_confirmation_is_estimated_never_confirmed(tmp_path: Path) -> None:
    """El valor por defecto de una fecha sin confirmar es `estimated`, nunca `confirmed`."""
    source = _FakeSource({"NVDA": ((date(2026, 9, 17), EarningsMoment.UNKNOWN),)})

    report = ingest(config=_config(), store=Store(tmp_path), source=source, now=NOW)

    (event,) = report.events
    assert event.certainty is EarningsCertainty.ESTIMATED
    assert event.blocking is False
    assert event.moment is EarningsMoment.UNKNOWN


def test_126_a_confirmed_date_blocks_and_the_artifact_overrides_the_source(tmp_path: Path) -> None:
    """Una confirmada bloquea; y si la fuente no la trae, manda el artefacto declarado."""
    # La fuente devuelve OTRA fecha: la confirmada del artefacto se añade igual.
    source = _FakeSource({"NVDA": ((date(2026, 10, 29), EarningsMoment.UNKNOWN),)})

    report = ingest(config=_config(confirmed=True), store=Store(tmp_path), source=source, now=NOW)

    by_date = {event.on: event for event in report.events}
    confirmed = by_date[date(2026, 9, 17)]
    assert confirmed.certainty is EarningsCertainty.CONFIRMED
    assert confirmed.moment is EarningsMoment.AMC
    assert confirmed.blocking is True
    assert by_date[date(2026, 10, 29)].certainty is EarningsCertainty.ESTIMATED


def test_126_the_daily_path_reads_the_store_point_in_time(tmp_path: Path) -> None:
    """`earnings_on` no devuelve una observación que aún no había ocurrido en el `as_of`."""
    source = _FakeSource({"NVDA": ((date(2026, 9, 17), EarningsMoment.UNKNOWN),)})
    store = Store(tmp_path)
    ingest(config=_config(), store=store, source=source, now=NOW)

    day = date(2026, 9, 17)
    assert len(earnings_on(store, session=day, as_of=datetime(2026, 9, 17, 12, 0, tzinfo=UTC))) == 1
    # Antes de la observación (NOW) no hay nada.
    assert earnings_on(store, session=day, as_of=datetime(2026, 9, 1, 12, 0, tzinfo=UTC)) == ()
    # Otro día no trae nada.
    other = date(2026, 9, 18)
    assert earnings_on(store, session=other, as_of=datetime(2026, 9, 18, 12, 0, tzinfo=UTC)) == ()


def test_126_an_empty_store_has_no_earnings(tmp_path: Path) -> None:
    """Un almacén sin `raw.earnings` devuelve `()`: no tener resultados no es un error."""
    assert earnings_on(Store(tmp_path), session=date(2026, 9, 17), as_of=NOW) == ()


def test_126_the_yfinance_adapter_declares_an_unknown_moment() -> None:
    """El adaptador real, con una factoría simulada, no abre red y no asume el momento."""

    class _FakeFrame:
        index: ClassVar[list[datetime]] = [datetime(2026, 10, 22, tzinfo=UTC)]

    class _FakeTicker:
        def get_earnings_dates(self, *, limit: int) -> _FakeFrame:
            return _FakeFrame()

    adapter = YFinanceEarningsAdapter(ticker_factory=lambda _symbol: _FakeTicker())

    dates = adapter.earnings_dates("NVDA", now=NOW)

    assert len(dates) == 1
    assert dates[0].on == date(2026, 10, 22)
    assert dates[0].moment is EarningsMoment.UNKNOWN


def test_126_a_broken_config_fails_at_start(tmp_path: Path) -> None:
    """Una lista de mega-caps mal formada falla al arrancar, no a mitad de la ingesta."""
    broken = tmp_path / "mega.yaml"
    broken.write_text("issuers: [{symbol: X}]\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="mega-caps inválida"):
        load_mega_caps(broken)
