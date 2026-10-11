"""Tests de la medicion de la serie historica del pre-mercado del ES (#149).

Todo se mide sobre un almacen **sintetico** en ``tmp_path``, con diez sesiones y barras
**inyectadas** (patron de #141): ni red, ni ``data/`` del repositorio, ni reloj. Las barras de cada
candidato se declaran a mano, de modo que la ventana, el recorte point-in-time y la frescura son
calculables a mano y el veredicto se pueda comprobar sesion a sesion.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Final, cast

import polars as pl
import pytest

from cfdtrader.analysis import premarket_gap, premarket_history
from cfdtrader.analysis.premarket_gap import PremarketBar
from cfdtrader.data.calendar import EASTERN
from cfdtrader.data.store import Store

#: Diez sesiones declaradas (dias laborables de septiembre de 2026).
SESSIONS: Final[tuple[date, ...]] = (
    date(2026, 9, 1),
    date(2026, 9, 2),
    date(2026, 9, 3),
    date(2026, 9, 4),
    date(2026, 9, 8),
    date(2026, 9, 9),
    date(2026, 9, 10),
    date(2026, 9, 11),
    date(2026, 9, 14),
    date(2026, 9, 15),
)

#: Cierre declarado del ES (la referencia) y apertura/cierre del indice sinteticos.
ES_CLOSE: Final[float] = 5000.0
INDEX_OPEN: Final[float] = 5000.0

#: El movimiento del pre-mercado de cada sesion: alterna -0,5 / 0 / +0,5 %.
MOVES: Final[dict[date, float]] = {
    day: 0.5 * ((position % 3) - 1) for position, day in enumerate(SESSIONS)
}

#: La sesion sobre la que se mide todo (la ultima con barras completas).
SESSION: Final[date] = SESSIONS[-1]

AS_OF: Final[str] = "2026-09-16T13:00:00+00:00"
FETCHED_AT: Final[datetime] = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


def instant(day: date, hour: int, minute: int) -> datetime:
    """El instante UTC de esa hora **ET** de esa sesion (sin reloj)."""
    return datetime.combine(day, time(hour, minute), tzinfo=EASTERN).astimezone(UTC)


def index_close(day: date) -> float:
    """El cierre del indice de esa sesion: el doble de movimiento que el pre-mercado."""
    return INDEX_OPEN * (1.0 + 2.0 * MOVES[day] / 100.0)


def _market(day: date, close: float, source: str, series_id: str) -> dict[str, object]:
    return {
        "source": source,
        "series_id": series_id,
        "as_of": instant(day, 16, 0),
        "fetched_at": FETCHED_AT,
        "published_at": None,
        "open": INDEX_OPEN,
        "high": max(INDEX_OPEN, close),
        "low": min(INDEX_OPEN, close),
        "close": close,
        "volume": 1000.0,
    }


def build_store(root: Path, *, label_sessions: int = 30) -> Store:
    """El almacen sintetico: cierres diarios del ES y del indice, y la ventana etiquetada."""
    store = Store(root)
    market = [
        *(_market(day, ES_CLOSE, "yfinance", "ES=F") for day in SESSIONS),
        *(_market(day, index_close(day), "yfinance", "^GSPC") for day in SESSIONS),
    ]
    store.append("raw", "market_daily", market)
    store.append(
        "derived",
        "labels",
        [
            {
                "source": "cfdtrader.models.labels",
                "series_id": "^GSPC",
                "as_of": instant(date(2020, 1, 1) + timedelta(days=position), 16, 0),
                "fetched_at": FETCHED_AT,
                "published_at": None,
                "session": date(2020, 1, 1) + timedelta(days=position),
                "ret_long": 0.0,
                "is_half_day": False,
                "k_sigma": 1.0,
            }
            for position in range(label_sessions)
        ],
    )
    return store


def _grid(spec: premarket_history.CandidateSpec) -> tuple[tuple[int, int], ...]:
    """Las barras **antes** del snapshot que declara cada granularidad (a mano)."""
    if spec.interval == "5m":
        return ((7, 0), (8, 0), (8, 40))
    if spec.interval == "30m":
        return ((7, 30), (8, 0), (8, 30))
    if spec.interval == "1h":
        return ((7, 0), (8, 0))
    return ()  # la diaria solo trae el cierre de las 16:00, que es **la referencia**


def provider(spec: premarket_history.CandidateSpec) -> tuple[PremarketBar, ...]:
    """Barras sinteticas del candidato: las de antes del snapshot **y** una posterior."""
    bars: list[PremarketBar] = []
    for day in SESSIONS:
        for hour, minute in _grid(spec):
            bars.append(
                PremarketBar(
                    instant=instant(day, hour, minute),
                    price=ES_CLOSE * (1.0 + MOVES[day] / 100.0),
                )
            )
        # Una barra **despues** del snapshot: existe y no puede usarse (es el *look-ahead*).
        bars.append(PremarketBar(instant=instant(day, 16, 0), price=ES_CLOSE))
    return tuple(bars)


def spy_provider() -> tuple[
    Callable[[premarket_history.CandidateSpec], Sequence[PremarketBar]], list[str]
]:
    """El *provider* de arriba, declarando que candidatos se le pidieron."""
    asked: list[str] = []

    def fetch(spec: premarket_history.CandidateSpec) -> tuple[PremarketBar, ...]:
        asked.append(spec.key)
        return provider(spec)

    return fetch, asked


def report(root: Path, *, label_sessions: int = 30) -> dict[str, object]:
    """El informe completo sobre el almacen sintetico, con el *provider* inyectado."""
    store = build_store(root, label_sessions=label_sessions)
    return premarket_history.analyse(
        store=store, as_of=datetime.fromisoformat(AS_OF), provider=provider
    )


def candidates(payload: Mapping[str, object]) -> list[dict[str, object]]:
    """La tabla de candidatos del informe, ya estrechada."""
    return cast("list[dict[str, object]]", payload["candidates"])


def by_key(payload: Mapping[str, object], key: str) -> dict[str, object]:
    """El candidato del informe por su clave."""
    return next(item for item in candidates(payload) if item["key"] == key)


def rows(payload: Mapping[str, object], field: str) -> list[dict[str, object]]:
    """Una tabla de mapping del informe (`redundancy`, `paid_sources`)."""
    return cast("list[dict[str, object]]", payload[field])


def strings(payload: Mapping[str, object], field: str) -> list[str]:
    """Una lista de prosa del informe (`honesty`, `limitations`)."""
    return cast("list[str]", payload[field])


def verdict(payload: Mapping[str, object]) -> dict[str, object]:
    """El veredicto del informe."""
    return cast("dict[str, object]", payload["verdict"])


# ─────────────────────────────────────────────────────────────────────────────
# C1/C2/C12 — candidatos medidos, recorte documentado y la diaria
# ─────────────────────────────────────────────────────────────────────────────
def test_c1_the_candidates_are_measured_not_declared(tmp_path: Path) -> None:
    """La tabla lleva las **cuatro** granularidades con su ventana medida y su licencia."""
    payload = report(tmp_path / "store")
    assert [item["key"] for item in candidates(payload)] == [
        "es_5m",
        "es_30m",
        "es_1h",
        "es_1d",
    ]
    cinco = by_key(payload, "es_5m")
    assert cinco["state"] == "measured"
    assert cinco["window_start_utc"] == instant(SESSIONS[0], 7, 0).isoformat()
    assert cinco["window_end_utc"] == instant(SESSION, 16, 0).isoformat()
    assert cinco["extended_hours_seen"] is True
    assert "yfinance" in str(cinco["license"])
    assert cinco["bounded"] is True, "diez sesiones son menos de un ano: el veredicto es acotado"


def test_c2_the_snapshot_recorte_is_verified_with_a_real_example(tmp_path: Path) -> None:
    """El ejemplo cita la barra usada, su desfase y la barra posterior que **no** se usa."""
    payload = report(tmp_path / "store")
    ejemplo = cast("dict[str, object]", by_key(payload, "es_5m")["example"])
    assert ejemplo["session"] == SESSION.isoformat()
    assert ejemplo["used_bar_et"] == instant(SESSION, 8, 40).astimezone(EASTERN).isoformat()
    assert ejemplo["staleness_minutes"] == 5.0
    assert datetime.fromisoformat(str(ejemplo["used_bar_utc"])) <= datetime.fromisoformat(
        str(ejemplo["snapshot_utc"])
    )
    assert ejemplo["next_bar_et"] == instant(SESSION, 16, 0).astimezone(EASTERN).isoformat()
    assert "no" in str(ejemplo["next_bar_note"])


def test_c2_a_stale_granularity_declares_its_delay(tmp_path: Path) -> None:
    """La horaria existe pero su ultima barra es de las 08:00: 45 min de desfase, declarados."""
    payload = report(tmp_path / "store")
    horaria = by_key(payload, "es_1h")
    assert horaria["state"] == "measured"
    assert horaria["sessions_fresh"] == len(SESSIONS) - 1, (
        "la primera sesion no tiene referencia previa"
    )
    assert horaria["staleness_minutes_max"] == 45.0
    assert (
        cast("float", horaria["staleness_minutes_max"]) <= premarket_history.MAX_STALENESS_MINUTES
    )


def test_c12_the_daily_series_cannot_give_a_point_in_time_datum(tmp_path: Path) -> None:
    """La diaria viene sellada al inicio de su ventana: su dato a las 08:45 seria *look-ahead*."""
    payload = report(tmp_path / "store")
    diaria = by_key(payload, "es_1d")
    assert diaria["state"] == "unavailable"
    assert diaria["sessions_with_datum"] == 0
    assert diaria["example"] is None, "no puede haber un ejemplo con una barra de la propia sesion"
    assert "look-ahead" in str(diaria["reason"])


# ─────────────────────────────────────────────────────────────────────────────
# C3 — la redundancia, con su n
# ─────────────────────────────────────────────────────────────────────────────
def test_c3_the_redundancy_publishes_its_n(tmp_path: Path) -> None:
    """El gap del pre-mercado frente al movimiento de la sesion del indice, con su `n`."""
    payload = report(tmp_path / "store")
    readings = rows(payload, "redundancy")
    cinco = next(item for item in readings if item["candidate"] == "es_5m")
    assert cinco["sessions"] == len(SESSIONS) - 1, "la primera sesion no tiene cierre previo"
    assert cinco["corr_intraday"] == pytest.approx(1.0), "el indice se movio el doble, en el "
    "sentido del pre-mercado"
    assert isinstance(cinco["corr_close"], float)
    assert cinco["bounded"] is True


# ─────────────────────────────────────────────────────────────────────────────
# C4/C5/C10 — el veredicto, el coste y el cierre del debate
# ─────────────────────────────────────────────────────────────────────────────
def test_c4_without_series_the_verdict_confirms_107(tmp_path: Path) -> None:
    """Con una ventana muy por debajo de la que el conjunto de features usa: **no hay serie**."""
    payload = report(tmp_path / "store")
    requirements = cast("dict[str, object]", payload["requirements"])
    assert requirements["sessions_required"] == 30
    assert cast("str", requirements["sessions_required_source"]).startswith("`derived.labels`")
    decision = verdict(payload)
    assert decision["decision"] == "no_series"
    assert decision["best_candidate"] in ("es_5m", "es_30m")
    assert cast("int", decision["best_sessions_fresh"]) < 30
    assert decision["reopens_107"] is False
    assert decision["feature_follow_up"] is None
    assert "#148" in str(decision["next"])


def test_c4_with_enough_window_the_verdict_opens_the_feature(tmp_path: Path) -> None:
    """Si la ventana alcanza las sesiones que hacen falta, el informe abre el seguimiento."""
    payload = report(tmp_path / "store", label_sessions=5)
    decision = verdict(payload)
    assert decision["decision"] == "series_sufficient"
    assert decision["reopens_107"] is True
    assert "feature" in str(decision["feature_follow_up"])


def test_c5_the_paid_sources_are_declared_and_not_verified(tmp_path: Path) -> None:
    """Las fuentes de pago se enumeran con lo que declaran, marcadas como **no** verificadas."""
    payload = report(tmp_path / "store")
    sources = rows(payload, "paid_sources")
    assert sources
    assert all(item["verified"] is False for item in sources)
    assert all("propietario" in str(item["cost"]) for item in sources)


def test_c10_the_report_says_what_happens_with_107(tmp_path: Path) -> None:
    """El informe dice si #107 se mantiene o se reabre, y enlaza #148 para el mientras tanto."""
    payload = report(tmp_path / "store")
    assert "se mantiene" in str(verdict(payload)["issue_107"])
    text = premarket_history.render_markdown(payload)
    assert "#107" in text
    assert "#148" in text
    assert "no hay edge demostrado" in text


# ─────────────────────────────────────────────────────────────────────────────
# C6/C7/C8/C9 — sin red, reutilizando #141, determinista y con la valla
# ─────────────────────────────────────────────────────────────────────────────
def test_c6_the_report_is_built_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--bars-dir` mide sin red: la descarga real no se llama nunca."""
    store_root = tmp_path / "store"
    build_store(store_root)
    bars_dir = tmp_path / "bars"

    def explode(_spec: premarket_history.CandidateSpec) -> tuple[PremarketBar, ...]:
        raise AssertionError("el informe **no** puede descargar: se reproduce sin red")

    monkeypatch.setattr(premarket_history, "_yfinance_bars", explode)
    premarket_history._capture(bars_dir, provider=provider)  # pyright: ignore[reportPrivateUsage]
    reports = tmp_path / "reports"
    code = premarket_history.main(
        [
            "--as-of",
            AS_OF,
            "--data-root",
            str(store_root),
            "--bars-dir",
            str(bars_dir),
            "--reports-dir",
            str(reports),
        ]
    )
    assert code == 0
    json_path = reports / "premarket_history.json"
    markdown_path = reports / "premarket_history.md"
    assert json_path.is_file() and markdown_path.is_file()
    payload = cast("dict[str, object]", json.loads(json_path.read_text(encoding="utf-8")))
    assert verdict(payload)["decision"] == "no_series"
    assert "Valla de honestidad" in markdown_path.read_text(encoding="utf-8")


def test_the_store_is_read_once_per_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Medir N sesiones **no** escanea el almacen N veces: cada consulta se lee una sola vez."""
    calls: list[str] = []
    original = Store.sql

    def spy(self: Store, query: str) -> pl.DataFrame:
        calls.append(query)
        return original(self, query)

    monkeypatch.setattr(Store, "sql", spy)
    report(tmp_path / "store")
    assert calls == list(dict.fromkeys(calls)), "la misma consulta se leyo mas de una vez"
    assert len(calls) >= 2, "el almacen tiene que leerse: el contador no puede estar vacio"


def test_c7_the_gap_is_measured_by_141(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """La medicion **llama** al `analyse` de #141 (no reimplementa el recorte point-in-time)."""
    calls: list[datetime] = []
    original = premarket_gap.analyse

    def spy(*, store: Store, as_of: datetime, fetcher: object) -> object:
        calls.append(as_of)
        return original(store=store, as_of=as_of, fetcher=fetcher)  # type: ignore[arg-type]

    monkeypatch.setattr(premarket_gap, "analyse", spy)
    report(tmp_path / "store")
    assert calls, "sin llamadas a #141 no hay recorte point-in-time verificado"
    assert all(moment.tzinfo is not None for moment in calls)
    assert all(moment == premarket_history._snapshot(moment.date()) for moment in calls)  # pyright: ignore[reportPrivateUsage]


def test_c8_the_report_is_deterministic(tmp_path: Path) -> None:
    """Mismas entradas ⇒ mismo informe, y el instante entra declarado (sin reloj)."""
    primero = report(tmp_path / "uno")
    segundo = report(tmp_path / "dos")
    assert json.dumps(primero, sort_keys=True) == json.dumps(segundo, sort_keys=True)
    assert primero["as_of_utc"] == datetime.fromisoformat(AS_OF).astimezone(UTC).isoformat()


def test_c9_the_honesty_fence_and_the_absences_are_published(tmp_path: Path) -> None:
    """La valla viaja en el informe y una fuente caida es una **ausencia declarada**."""
    payload = report(tmp_path / "store")
    honesty = strings(payload, "honesty")
    assert any("no" in item and "edge" in item for item in honesty)
    assert any("phase2_ready" in item for item in honesty)
    limits = strings(payload, "limitations")
    assert limits
    assert "08:45" in str(payload["point_in_time"])


def test_a_failure_of_the_source_is_a_declared_absence(tmp_path: Path) -> None:
    """Un fallo de la fuente no rompe la medicion: se publica como ausencia con su motivo."""
    store = build_store(tmp_path / "store")

    def broken(_spec: premarket_history.CandidateSpec) -> tuple[PremarketBar, ...]:
        raise RuntimeError("la fuente no responde")

    payload = premarket_history.analyse(
        store=store, as_of=datetime.fromisoformat(AS_OF), provider=broken
    )
    assert all(item["state"] == "unavailable" for item in candidates(payload))
    assert "la fuente no responde" in str(by_key(payload, "es_5m")["reason"])


# ─────────────────────────────────────────────────────────────────────────────
# C11 — CLI, ficheros y contrato
# ─────────────────────────────────────────────────────────────────────────────
def test_c11_the_cli_rejects_an_invalid_as_of() -> None:
    """Sin `--as-of` (o sin zona) el CLI devuelve `2`."""
    assert premarket_history.main([]) == 2
    assert premarket_history.main(["--as-of", "no-es-iso"]) == 2
    assert premarket_history.main(["--as-of", "2026-09-16T13:00:00"]) == 2


def test_c11_the_cli_reports_a_missing_capture(tmp_path: Path) -> None:
    """Con `--bars-dir` sin ficheros, cada candidato se declara ausente y el informe sale igual."""
    store_root = tmp_path / "store"
    build_store(store_root)
    code = premarket_history.main(
        ["--as-of", AS_OF, "--data-root", str(store_root), "--bars-dir", str(tmp_path / "vacio")]
    )
    assert code == 0


def test_c11_the_module_exports_its_contract() -> None:
    """El contrato declarado existe y el modulo **no** importa la capa de decision."""
    for name in premarket_history.__all__:
        assert hasattr(premarket_history, name), name
    assert {
        "analyse",
        "decide",
        "measure_candidate",
        "render_markdown",
        "sessions_required",
        "write_report",
    } <= set(premarket_history.__all__)
