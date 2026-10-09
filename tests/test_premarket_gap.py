"""Tests del gap de pre-mercado del futuro ES como dato declarado de la decision (#141).

Los numeros esperados se calculan **a mano** en el propio test (no se copian de la salida del
codigo): con `ES=F` cerrando a 5050.0 y el `^GSPC` a 5000.0 en la sesion anterior, y una barra de
pre-mercado a 5100.5, el movimiento del futuro es `+1.0 %`, el gap frente al indice `+2.01 %` y el
*basis* `+1.0 %`. La barra **posterior** al snapshot (12:45 UTC) nunca entra: es *look-ahead*.

Todo lo que se escribe va a `tmp_path`: la fixture de sesion de `tests/conftest.py` huella el
`data/` del repositorio y no puede cambiar.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import pytest

from cfdtrader.analysis import premarket_gap as pg
from cfdtrader.analysis.premarket_gap import (
    ES_SERIES,
    INDEX_SERIES,
    PremarketBar,
    PremarketGapError,
    analyse,
    load_bars,
    render_markdown,
    save_bars,
    snapshot_instant,
    write_report,
)
from cfdtrader.data.store import Store

#: La sesion que se decide.
SESSION: Final[date] = date(2026, 10, 9)
#: El `as_of` de la sesion (09:00 ET), posterior al snapshot de las 08:45 ET.
AS_OF: Final[datetime] = datetime(2026, 10, 9, 13, 0, tzinfo=UTC)
#: El snapshot declarado (08:45 ET = 12:45 UTC en EDT).
SNAPSHOT: Final[datetime] = datetime(2026, 10, 9, 12, 45, tzinfo=UTC)
#: El cierre de la sesion anterior (16:00 ET del 2026-10-08).
PREVIOUS_INSTANT: Final[datetime] = datetime(2026, 10, 8, 20, 0, tzinfo=UTC)

#: Cierres de la sesion anterior: `ES=F` a 5050.0 y `^GSPC` a 5000.0 ⇒ *basis* +1.0 %.
ES_PREVIOUS_CLOSE: Final[float] = 5050.0
INDEX_PREVIOUS_CLOSE: Final[float] = 5000.0
#: La barra de pre-mercado que se publica: `5100.5` ⇒ movimiento +1.0 % frente al cierre del ES.
ES_PREMARKET: Final[float] = 5100.5


def _market_records(*, series_id: str, close: float) -> list[dict[str, object]]:
    """Una fila diaria de esa serie, sellada al cierre de la sesion anterior."""
    return [
        {
            "source": "yfinance",
            "series_id": series_id,
            "as_of": PREVIOUS_INSTANT,
            "fetched_at": datetime(2026, 10, 8, 21, tzinfo=UTC),
            "published_at": None,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        }
    ]


def _store(root: Path, *, index: bool = True) -> Store:
    """Almacen sintetico con el cierre de la sesion anterior de `ES=F` (y de `^GSPC`)."""
    store = Store(root)
    records = _market_records(series_id=ES_SERIES, close=ES_PREVIOUS_CLOSE)
    if index:
        records += _market_records(series_id=INDEX_SERIES, close=INDEX_PREVIOUS_CLOSE)
    store.append("raw", "market_daily", records)
    return store


def _bars() -> tuple[PremarketBar, ...]:
    """Barras de pre-mercado: una vieja, la del snapshot y una **posterior** (look-ahead)."""
    return (
        PremarketBar(datetime(2026, 10, 8, 10, 0, tzinfo=UTC), 1.0),  # antes del cierre previo
        PremarketBar(datetime(2026, 10, 9, 11, 0, tzinfo=UTC), 5060.0),  # dentro de la ventana
        PremarketBar(datetime(2026, 10, 9, 12, 40, tzinfo=UTC), ES_PREMARKET),  # la que se publica
        PremarketBar(datetime(2026, 10, 9, 13, 10, tzinfo=UTC), 9999.0),  # DESPUES del snapshot
    )


def _fetcher(bars: tuple[PremarketBar, ...]) -> pg.BarsFetcher:
    """Un *fetcher* inyectable que devuelve esas barras (sin red)."""

    def fetch(_series_id: str, *, as_of: datetime) -> tuple[PremarketBar, ...]:
        del as_of
        return bars

    return fetch


# ─────────────────────────────────────────────────────────────────────────────
# A · El nucleo point-in-time
# ─────────────────────────────────────────────────────────────────────────────
def test_a_snapshot_instant_is_0845_et() -> None:
    """El snapshot declarado son las 08:45 ET de la sesion del `as_of`."""
    session, snapshot = snapshot_instant(AS_OF)
    assert session == SESSION
    assert snapshot == SNAPSHOT


def test_a_uses_only_bars_up_to_the_snapshot(tmp_path: Path) -> None:
    """La barra **posterior** al snapshot (13:10) es *look-ahead* y no se publica."""
    report = analyse(store=_store(tmp_path), as_of=AS_OF, fetcher=_fetcher(_bars()))
    assert report.state == "measured"
    assert report.last_bar is not None
    assert report.last_bar.instant == datetime(2026, 10, 9, 12, 40, tzinfo=UTC)
    assert report.last_bar.price == ES_PREMARKET
    # Entran 11:00 y 12:40; la vieja (antes del cierre previo) y la posterior quedan fuera.
    assert report.bars_considered == 2


def test_a_gap_arithmetic_matches_the_hand_calculation(tmp_path: Path) -> None:
    """Los tres numeros: +1.0 % (futuro), +2.01 % (frente al indice) y +1.0 % (*basis*)."""
    report = analyse(store=_store(tmp_path), as_of=AS_OF, fetcher=_fetcher(_bars()))
    assert report.overnight_move_pct == round((5100.5 / 5050.0 - 1.0) * 100.0, 8)
    assert report.gap_vs_index_pct == round((5100.5 / 5000.0 - 1.0) * 100.0, 8)
    assert report.basis_pct == round((5050.0 / 5000.0 - 1.0) * 100.0, 8)
    assert report.overnight_move_pct == 1.0
    assert report.gap_vs_index_pct == 2.01
    assert report.basis_pct == 1.0


def test_a_reference_is_the_last_close_before_the_snapshot(tmp_path: Path) -> None:
    """El cierre de referencia es la sesion **anterior**, no la del dia que se decide."""
    report = analyse(store=_store(tmp_path), as_of=AS_OF, fetcher=_fetcher(_bars()))
    assert report.es_previous is not None
    assert report.index_previous is not None
    assert report.es_previous.session == date(2026, 10, 8)
    assert report.index_previous.session == date(2026, 10, 8)
    assert report.es_previous.price == ES_PREVIOUS_CLOSE
    assert report.index_previous.price == INDEX_PREVIOUS_CLOSE


# ─────────────────────────────────────────────────────────────────────────────
# B · Ausencias declaradas (nunca un valor de relleno)
# ─────────────────────────────────────────────────────────────────────────────
def test_b_snapshot_ahead_is_declared_and_does_not_fetch(tmp_path: Path) -> None:
    """Un `as_of` anterior al snapshot es `unavailable`: no se adelanta ni se llama a la fuente."""
    calls: list[datetime] = []

    def fetch(_series_id: str, *, as_of: datetime) -> tuple[PremarketBar, ...]:
        calls.append(as_of)
        return _bars()

    early = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)  # 08:00 ET
    report = analyse(store=_store(tmp_path), as_of=early, fetcher=fetch)
    assert report.state == "unavailable"
    assert "snapshot" in report.reason
    assert calls == []


def test_b_missing_reference_is_declared(tmp_path: Path) -> None:
    """Sin el cierre del `^GSPC` no hay gap comparable: `unavailable`, con su motivo."""
    report = analyse(store=_store(tmp_path, index=False), as_of=AS_OF, fetcher=_fetcher(_bars()))
    assert report.state == "unavailable"
    assert report.es_previous is not None
    assert report.index_previous is None
    assert "referencia" in report.reason


def test_b_no_premarket_bars_is_declared(tmp_path: Path) -> None:
    """Sin barras de pre-mercado, `unavailable` (mercado cerrado o ventana agotada)."""
    report = analyse(store=_store(tmp_path), as_of=AS_OF, fetcher=_fetcher(()))
    assert report.state == "unavailable"
    assert "ninguna barra" in report.reason


def test_b_fetch_failure_is_declared_not_raised(tmp_path: Path) -> None:
    """Un fallo de la fuente **no** tumba el camino: se declara con su motivo."""

    def fetch(_series_id: str, *, as_of: datetime) -> tuple[PremarketBar, ...]:
        del as_of
        raise RuntimeError("la red ha caido")

    report = analyse(store=_store(tmp_path), as_of=AS_OF, fetcher=fetch)
    assert report.state == "unavailable"
    assert "la red ha caido" in report.reason


def test_b_payload_does_not_invent_values_when_unavailable(tmp_path: Path) -> None:
    """En `unavailable` los numeros son `null`, jamas `0`."""
    payload = analyse(store=_store(tmp_path, index=False), as_of=AS_OF).payload()
    assert payload["state"] == "unavailable"
    assert payload["es_premarket_price"] is None
    assert payload["overnight_move_pct"] is None
    assert payload["gap_vs_index_pct"] is None
    assert payload["basis_pct"] is None


def test_b_analyse_needs_a_timezone(tmp_path: Path) -> None:
    """Un `as_of` sin zona es un error tipado, no una hora por defecto."""
    with pytest.raises(PremarketGapError):
        analyse(store=_store(tmp_path), as_of=datetime(2026, 10, 9, 13, 0))


# ─────────────────────────────────────────────────────────────────────────────
# C · Captura, publicacion y CLI
# ─────────────────────────────────────────────────────────────────────────────
def test_c_bars_roundtrip_is_exact(tmp_path: Path) -> None:
    """Guardar y releer las barras devuelve exactamente lo mismo (reproducir sin red)."""
    path = save_bars(_bars(), tmp_path / "bars.json")
    assert path.exists()
    assert load_bars(path) == tuple(sorted(_bars(), key=lambda bar: bar.instant))


def test_c_load_bars_assumes_utc_when_the_stamp_has_no_zone(tmp_path: Path) -> None:
    """Un sello sin zona se asume UTC (y se declara asi), no se inventa otra zona."""
    path = tmp_path / "sin_zona.json"
    path.write_text(json.dumps([{"instant": "2026-10-09T12:40:00", "price": 5100.5}]), "utf-8")
    bars = load_bars(path)
    assert bars[0].instant == datetime(2026, 10, 9, 12, 40, tzinfo=UTC)


def test_c_load_bars_rejects_a_broken_file(tmp_path: Path) -> None:
    """Un fichero que no es una lista de barras completas es un error tipado."""
    path = tmp_path / "roto.json"
    path.write_text(json.dumps({"instant": "x"}), "utf-8")
    with pytest.raises(PremarketGapError):
        load_bars(path)
    path.write_text(json.dumps([{"instant": "2026-10-09T12:40:00+00:00"}]), "utf-8")
    with pytest.raises(PremarketGapError):
        load_bars(path)


def test_c_markdown_states_the_three_numbers_and_the_fence(tmp_path: Path) -> None:
    """El informe lleva los tres numeros, el point-in-time y la valla de honestidad."""
    markdown = render_markdown(
        analyse(store=_store(tmp_path), as_of=AS_OF, fetcher=_fetcher(_bars())).payload()
    )
    assert "overnight_move_pct" in markdown
    assert "gap_vs_index_pct" in markdown
    assert "basis_pct" in markdown
    assert "no hay edge demostrado" in markdown
    assert "#107" in markdown
    assert "08:45" in markdown


def test_c_write_report_writes_json_and_markdown(tmp_path: Path) -> None:
    """El artefacto se escribe con su nombre por sesion y un `.json` reparable."""
    payload = analyse(store=_store(tmp_path), as_of=AS_OF, fetcher=_fetcher(_bars())).payload()
    json_path, markdown_path = write_report(payload, tmp_path / "reports")
    assert json_path.name == f"premarket_gap_{SESSION.isoformat()}.json"
    assert markdown_path.name == f"premarket_gap_{SESSION.isoformat()}.md"
    assert json.loads(json_path.read_text(encoding="utf-8"))["state"] == "measured"


def test_c_main_requires_as_of() -> None:
    """Sin `--as-of` (o con uno invalido) el CLI devuelve `2`."""
    assert pg.main([]) == 2
    assert pg.main(["--as-of", "no-es-iso"]) == 2
    assert pg.main(["--as-of", "2026-10-09T13:00:00"]) == 2  # sin zona


def test_c_main_measures_from_captured_bars(tmp_path: Path) -> None:
    """Con `--es-bars` mide sin red y escribe el informe."""
    store_root = tmp_path / "store"
    _store(store_root)
    bars_path = save_bars(_bars(), tmp_path / "bars.json")
    reports = tmp_path / "reports"
    code = pg.main(
        [
            "--as-of",
            AS_OF.isoformat(),
            "--data-root",
            str(store_root),
            "--es-bars",
            str(bars_path),
            "--reports-dir",
            str(reports),
        ]
    )
    assert code == 0
    written = json.loads(
        (reports / f"premarket_gap_{SESSION.isoformat()}.json").read_text(encoding="utf-8")
    )
    assert written["state"] == "measured"
    assert written["overnight_move_pct"] == 1.0
