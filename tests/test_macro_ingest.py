"""Tests de la ingesta macro desde FRED (tarea #5).

El artefacto verificable es el **point-in-time**: tres publicaciones conocidas de
las 08:30 ET no pueden ser visibles antes de su hora, y sí después. Ningún test
abre red: el cliente HTTP se inyecta con ``httpx.MockTransport`` y las respuestas
son *fixtures* de FRED (``tests/fixtures/fred/``).
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from cfdtrader.data.macro import (
    EXIT_MISSING_API_KEY,
    MacroSecrets,
    ingest,
    load_macro_series,
    main,
    render_markdown,
)
from cfdtrader.data.settings import ConfigurationError
from cfdtrader.data.sources.base import SourceStatus
from cfdtrader.data.sources.fred_adapter import FredAdapter, MacroSeriesRegistry, MacroSeriesSpec
from cfdtrader.data.sources.http import CachedHttpClient
from cfdtrader.data.store import Store

FIXTURES = Path(__file__).parent / "fixtures" / "fred"

#: Instante en el que ya se han publicado las tres referencias de enero de 2024.
NOW = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)

#: Las tres publicaciones conocidas de las 08:30 ET que exige la tarea.
#: (serie, fixture, periodo observado, instante del comunicado en UTC)
KNOWN_PUBLICATIONS: tuple[tuple[str, str, date, datetime], ...] = (
    ("CPIAUCSL", "cpi.json", date(2023, 12, 1), datetime(2024, 1, 11, 13, 30, tzinfo=UTC)),
    ("PAYEMS", "payems.json", date(2023, 12, 1), datetime(2024, 1, 5, 13, 30, tzinfo=UTC)),
    ("PCEPI", "pcepi.json", date(2023, 12, 1), datetime(2024, 1, 26, 13, 30, tzinfo=UTC)),
)

Handler = Callable[[httpx.Request], httpx.Response]

#: Clave de mentira para los tests. No es un secreto: la API no se llama nunca.
PROBE_KEY = "no-es-una-clave"  # pragma: allowlist secret


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def _spec(series_id: str, **overrides: object) -> MacroSeriesSpec:
    """Especificación de serie macro, tomada del registro real cuando existe."""
    for spec in load_macro_series().series:
        if spec.series_id == series_id:
            if not overrides:
                return spec
            return spec.model_copy(update=overrides)
    raise AssertionError(f"{series_id!r} no está declarada en config/macro_series.yaml")


def _handler(bodies: Mapping[str, bytes], calls: list[str] | None = None) -> Handler:
    """Transporte simulado que devuelve el cuerpo declarado para cada serie."""

    def handle(request: httpx.Request) -> httpx.Response:
        series_id = request.url.params["series_id"]
        if calls is not None:
            calls.append(series_id)
        body = bodies.get(series_id)
        if body is None:
            return httpx.Response(
                400, json={"error_code": 400, "error_message": f"Bad Request. {series_id}"}
            )
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)

    return handle


def _adapter(bodies: Mapping[str, bytes], *, api_key: str | None = "test-key") -> FredAdapter:
    """Adaptador de FRED con transporte simulado y sin backoff (los tests no duermen)."""
    client = CachedHttpClient(
        source="fred",
        client=httpx.Client(transport=httpx.MockTransport(_handler(bodies))),
        backoff_seconds=0.0,
    )
    return FredAdapter(client, api_key=api_key)


def _registry(*series_ids: str) -> MacroSeriesRegistry:
    """Registro con las series indicadas."""
    registry = load_macro_series()
    return registry.model_copy(
        update={"series": tuple(spec for spec in registry.series if spec.series_id in series_ids)}
    )


def _fixture(name: str) -> bytes:
    """Cuerpo de un fixture de FRED."""
    return (FIXTURES / name).read_bytes()


def _observations(**overrides: object) -> bytes:
    """Respuesta de FRED construida a mano, para los casos que no son un fixture."""
    payload: dict[str, object] = {
        "realtime_start": "2024-06-11",
        "realtime_end": "2024-06-11",
        "observations": [
            {"realtime_start": "2024-06-10", "date": "2024-06-10", "value": "4.28"},
        ],
    }
    payload.update(overrides)
    return json.dumps(payload).encode()


# ─────────────────────────────────────────────────────────────────────────────
# Registro
# ─────────────────────────────────────────────────────────────────────────────
def test_the_registry_declares_rates_inflation_and_employment() -> None:
    """Fed funds, 2 y 10 años, pendiente 2s10s, CPI, PCE y NFP: las siete del enunciado."""
    ids = {spec.series_id for spec in load_macro_series().series}

    assert ids == {"DFF", "DGS2", "DGS10", "T10Y2Y", "CPIAUCSL", "PCEPI", "PAYEMS"}


def test_release_times_are_declared_in_eastern_time() -> None:
    """El CPI/PCE/NFP se publican a las 08:30 ET y toman la fecha de la vintage de ALFRED."""
    for series_id in ("CPIAUCSL", "PCEPI", "PAYEMS"):
        spec = _spec(series_id)
        assert spec.release_time_et.hour == 8
        assert spec.release_time_et.minute == 30
        assert spec.publication_from_realtime_start is True

    # Las series que fija el mercado no tienen rueda de prensa: se declara el desplazamiento.
    assert _spec("DFF").publication_offset_days == 1
    assert _spec("DGS10").release_time_et.hour == 15


def test_the_european_context_is_declared_as_not_implemented() -> None:
    """El contexto europeo es opcional y se declara, no se olvida en silencio."""
    registry = load_macro_series()
    sources = {str(item["source"]) for item in registry.european_context}
    assert sources == {"ecb_sdw", "eurostat"}
    assert all(item["status"] == "not_implemented" for item in registry.european_context)


# ─────────────────────────────────────────────────────────────────────────────
# Point-in-time: las tres publicaciones conocidas de las 08:30 ET
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(("series_id", "fixture", "period", "instant"), KNOWN_PUBLICATIONS)
def test_known_publications_land_at_0830_eastern(
    tmp_path: Path, series_id: str, fixture: str, period: date, instant: datetime
) -> None:
    """El `published_at` guardado es el instante del comunicado, no el periodo observado."""
    ingest(
        registry=_registry(series_id),
        data_root=tmp_path,
        adapter=_adapter({series_id: _fixture(fixture)}),
        now=NOW,
    )

    rows = Store(tmp_path).sql("SELECT as_of, published_at FROM raw.macro").to_dicts()
    observed = [row for row in rows if str(row["as_of"]) == period.isoformat()]

    assert len(observed) == 1
    assert observed[0]["published_at"] == instant
    # El comunicado es posterior al periodo observado: son dos cosas distintas.
    assert instant.date() > period


@pytest.mark.parametrize(("series_id", "fixture", "period", "instant"), KNOWN_PUBLICATIONS)
def test_the_three_publications_are_invisible_before_their_hour(
    tmp_path: Path, series_id: str, fixture: str, period: date, instant: datetime
) -> None:
    """La prueba del algodón: un minuto antes del comunicado, el dato **no existe**."""
    ingest(
        registry=_registry(series_id),
        data_root=tmp_path,
        adapter=_adapter({series_id: _fixture(fixture)}),
        now=NOW,
    )
    store = Store(tmp_path)
    one_minute_before = instant - timedelta(minutes=1)

    before = store.read_pit("raw", "macro", at=one_minute_before, series_id=series_id).to_dicts()
    after = store.read_pit("raw", "macro", at=instant, series_id=series_id).to_dicts()

    assert all(str(row["as_of"]) != period.isoformat() for row in before)
    assert any(str(row["as_of"]) == period.isoformat() for row in after)


def test_a_publication_that_has_not_arrived_is_not_written(tmp_path: Path) -> None:
    """Con `now` anterior al comunicado, la observación se descarta y se declara."""
    # El 10 de enero de 2024 el CPI de diciembre aún no se había publicado.
    before_release = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
    report = ingest(
        registry=_registry("CPIAUCSL"),
        data_root=tmp_path,
        adapter=_adapter({"CPIAUCSL": _fixture("cpi.json")}),
        now=before_release,
    )

    row = report.rows[0]
    assert row.discarded_unpublished == 2  # ni el de diciembre ni el de enero
    assert row.rows_written == 0
    assert row.status is SourceStatus.UNAVAILABLE
    assert "no quedó ninguna observación publicada" in "; ".join(row.notes)


def test_market_determined_series_use_the_declared_publication_rule() -> None:
    """Un rendimiento del Tesoro se conoce el día del que habla; el fed funds, al siguiente."""
    adapter = _adapter({"DGS10": _observations(), "DFF": _observations()})
    now = datetime(2024, 6, 11, 20, 0, tzinfo=UTC)

    treasury = adapter.fetch(_spec("DGS10"), now=now)
    fed_funds = adapter.fetch(_spec("DFF"), now=now)

    assert treasury.frame.get_column("published_at").to_list() == [
        datetime(2024, 6, 10, 19, 30, tzinfo=UTC)  # 15:30 EDT del propio día
    ]
    assert fed_funds.frame.get_column("published_at").to_list() == [
        datetime(2024, 6, 11, 13, 0, tzinfo=UTC)  # 09:00 EDT del día siguiente
    ]
    assert treasury.frame.get_column("as_of").to_list() == [date(2024, 6, 10)]


def test_the_request_asks_for_all_the_vintages_of_a_release_calendar_series() -> None:
    """Sin pedir las *vintages*, FRED devuelve la última y publica **toda** la historia hoy.

    Es el bug que apareció al probar contra la API de verdad (2026-09-17): con la
    consulta por omisión, las 260 observaciones de `CPIAUCSL` desde 2005 traían
    `realtime_start = 2026-09-17`. Lo que se pide ahora es la ventana de vintages
    completa, y solo para las series que publican por calendario.
    """
    seen: list[dict[str, str]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        return httpx.Response(
            200, headers={"content-type": "application/json"}, content=_fixture("cpi.json")
        )

    adapter = FredAdapter(
        CachedHttpClient(
            source="fred",
            client=httpx.Client(transport=httpx.MockTransport(handle)),
            backoff_seconds=0.0,
        ),
        api_key=PROBE_KEY,
    )

    adapter.fetch(_spec("CPIAUCSL"), now=NOW)
    calendar_series = seen[-1]
    assert calendar_series["realtime_start"] == "2005-01-01"
    assert calendar_series["realtime_end"] == "9999-12-31"

    adapter.fetch(_spec("DGS10"), now=NOW)
    market_series = seen[-1]
    assert "realtime_start" not in market_series
    assert "realtime_end" not in market_series


def test_only_the_first_publication_of_each_observation_is_kept() -> None:
    """Con varias *vintages* de la misma observación se guarda el primer comunicado.

    El valor que movió el mercado es el primer publicado; una vintage con `.` no
    es una publicación y no puede «adelantar» la fecha.
    """
    body = json.dumps(
        {
            "realtime_start": "2005-01-01",
            "realtime_end": "9999-12-31",
            "observations": [
                # Primera publicación, luego la revisión, y una vintage sin dato.
                {"realtime_start": "2024-01-11", "date": "2023-12-01", "value": "306.746"},
                {"realtime_start": "2024-02-13", "date": "2023-12-01", "value": "306.900"},
                {"realtime_start": "2024-02-12", "date": "2023-12-01", "value": "."},
                {"realtime_start": "2024-02-13", "date": "2024-01-01", "value": "308.417"},
            ],
        }
    ).encode()

    result = _adapter({"CPIAUCSL": body}).fetch(_spec("CPIAUCSL"), now=NOW)

    assert result.ok is True
    assert result.frame.get_column("as_of").to_list() == [date(2023, 12, 1), date(2024, 1, 1)]
    assert result.frame.get_column("value").to_list() == [306.746, 308.417]
    assert result.frame.get_column("published_at").to_list() == [
        datetime(2024, 1, 11, 13, 30, tzinfo=UTC),
        datetime(2024, 2, 13, 13, 30, tzinfo=UTC),
    ]
    assert any("primera publicación" in note for note in result.notes)


# ─────────────────────────────────────────────────────────────────────────────
# Escritura y revisiones
# ─────────────────────────────────────────────────────────────────────────────
def test_macro_rows_are_dated_series_with_their_publication_instant(tmp_path: Path) -> None:
    """`raw.macro` guarda `as_of` como fecha y `published_at` como instante UTC."""
    ingest(
        registry=_registry("DGS10"),
        data_root=tmp_path,
        adapter=_adapter({"DGS10": _observations()}),
        now=datetime(2024, 6, 12, 12, 0, tzinfo=UTC),
    )

    row = Store(tmp_path).sql("SELECT * FROM raw.macro").to_dicts()[0]

    assert row["series_id"] == "DGS10"
    assert row["source"] == "fred"
    assert str(row["as_of"]) == "2024-06-10"
    assert row["published_at"] == datetime(2024, 6, 10, 19, 30, tzinfo=UTC)
    assert row["version"] == 1


def test_a_second_identical_run_is_a_noop(tmp_path: Path) -> None:
    """Repetir la ingesta no duplica ni reescribe: la segunda vez no entra nada."""
    registry = _registry("CPIAUCSL")
    adapter = _adapter({"CPIAUCSL": _fixture("cpi.json")})

    first = ingest(registry=registry, data_root=tmp_path, adapter=adapter, now=NOW)
    files = sorted((tmp_path / "raw").rglob("*.parquet"))
    second = ingest(registry=registry, data_root=tmp_path, adapter=adapter, now=NOW)

    assert first.rows[0].rows_new == 2
    assert second.rows[0].rows_new == 0
    assert second.rows[0].rows_written == 2
    assert second.rows[0].revisions == 0
    assert any("sin cambios" in note for note in second.rows[0].notes)
    assert sorted((tmp_path / "raw").rglob("*.parquet")) == files


def test_second_run_does_not_send_the_history_to_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La ingesta diaria no reenvía el histórico de FRED: solo lo nuevo y lo revisado.

    DFF tiene 7.928 observaciones. Reenviarlas cada día es correcto (el almacén
    deduplica) pero obliga a comprobar identidad registro a registro para acabar
    sin escribir nada. Aquí se comprueba que la segunda vuelta ni siquiera llama.
    """
    registry = _registry("CPIAUCSL")
    payload = _fixture("cpi.json")
    calls: list[str] = []
    original_append = Store.append
    original_revision = Store.append_revision

    def spy_append(self: Store, layer: str, dataset: str, records: object) -> object:
        calls.append("append")
        return original_append(self, layer, dataset, records)  # type: ignore[arg-type]

    def spy_revision(self: Store, layer: str, dataset: str, records: object) -> object:
        calls.append("append_revision")
        return original_revision(self, layer, dataset, records)  # type: ignore[arg-type]

    monkeypatch.setattr(Store, "append", spy_append)
    monkeypatch.setattr(Store, "append_revision", spy_revision)

    ingest(
        registry=registry,
        data_root=tmp_path,
        adapter=_adapter({"CPIAUCSL": payload}),
        now=datetime(2024, 2, 20, 12, 0, tzinfo=UTC),
    )
    assert calls == ["append"], "la primera vuelta escribe las observaciones nuevas"
    calls.clear()
    ingest(
        registry=registry,
        data_root=tmp_path,
        adapter=_adapter({"CPIAUCSL": payload}),
        now=datetime(2024, 2, 20, 12, 0, tzinfo=UTC),
    )

    assert calls == [], "la segunda vuelta no debe tocar el almacén"


def test_a_revision_from_the_source_becomes_version_two(tmp_path: Path) -> None:
    """FRED revisa CPI/PCE/NFP: la revisión se guarda con `append_revision`."""
    registry = _registry("CPIAUCSL")
    original = json.loads(_fixture("cpi.json").decode("utf-8"))
    revised = {
        **original,
        "observations": [
            {
                # La revisión se publica más tarde: por eso `read_pit` la ve después.
                "realtime_start": "2024-02-29",
                "realtime_end": "9999-12-31",
                "date": "2023-12-01",
                "value": "306.900",  # el valor revisado
            }
        ],
    }
    early = datetime(2024, 2, 20, 12, 0, tzinfo=UTC)
    late = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)

    ingest(
        registry=registry,
        data_root=tmp_path,
        adapter=_adapter({"CPIAUCSL": json.dumps(original).encode()}),
        now=early,
    )
    report = ingest(
        registry=registry,
        data_root=tmp_path,
        adapter=_adapter({"CPIAUCSL": json.dumps(revised).encode()}),
        now=late,
    )
    store = Store(tmp_path)

    assert report.rows[0].revisions == 1
    at_early = store.read_pit("raw", "macro", at=early, series_id="CPIAUCSL").to_dicts()
    at_late = store.read_pit("raw", "macro", at=late, series_id="CPIAUCSL").to_dicts()
    december_early = next(row for row in at_early if str(row["as_of"]) == "2023-12-01")
    december_late = next(row for row in at_late if str(row["as_of"]) == "2023-12-01")

    assert december_early["version"] == 1 and december_early["value"] == 306.746
    assert december_late["version"] == 2 and december_late["value"] == 306.9
    # El estado consultable tiene una sola fila por identidad: la revisión vigente.
    assert store.sql("SELECT * FROM raw.macro WHERE as_of = DATE '2023-12-01'").height == 1


def test_missing_values_are_declared_and_not_written_as_zero() -> None:
    """El marcador de dato ausente de FRED (`.`) no se escribe como 0."""
    body = _observations(
        observations=[
            {"realtime_start": "2024-06-10", "date": "2024-06-10", "value": "."},
            {"realtime_start": "2024-06-11", "date": "2024-06-11", "value": "4.28"},
        ]
    )
    result = _adapter({"DGS10": body}).fetch(
        _spec("DGS10"), now=datetime(2024, 6, 12, 12, 0, tzinfo=UTC)
    )

    assert result.ok is True
    assert result.frame.height == 2
    assert result.frame.get_column("value").to_list() == [None, 4.28]


# ─────────────────────────────────────────────────────────────────────────────
# Estados declarados y CLI
# ─────────────────────────────────────────────────────────────────────────────
def test_without_an_api_key_every_series_is_declared_unavailable() -> None:
    """Sin clave no se inventa nada: estado `unavailable` y el motivo con nombre y apellido."""
    result = _adapter({}, api_key=None).fetch(_spec("CPIAUCSL"), now=NOW)

    assert result.status is SourceStatus.UNAVAILABLE
    assert result.frame.height == 0
    assert result.error is not None and "FRED_API_KEY" in result.error


def test_an_error_from_fred_is_declared_with_its_message() -> None:
    """Un rechazo de FRED (clave inválida, serie no servida) se declara, no se traga."""
    result = _adapter({}).fetch(_spec("DGS10"), now=NOW)

    assert result.status is SourceStatus.ERROR
    assert result.error is not None and "FRED rechazó la petición" in result.error


def test_an_empty_vintage_is_unavailable_not_ok() -> None:
    """Una respuesta sin observaciones no puede figurar como `ok`."""
    body = json.dumps({"realtime_start": "2024-06-11", "observations": []}).encode()
    result = _adapter({"DGS10": body}).fetch(_spec("DGS10"), now=NOW)

    assert result.status is SourceStatus.UNAVAILABLE
    assert result.frame.height == 0
    assert result.error is not None


def test_the_report_is_written_in_json_and_markdown(tmp_path: Path) -> None:
    """El informe macro queda escrito y declara si había clave."""
    report = ingest(
        registry=_registry("CPIAUCSL", "PAYEMS"),
        data_root=tmp_path,
        adapter=_adapter({"CPIAUCSL": _fixture("cpi.json"), "PAYEMS": _fixture("payems.json")}),
        now=NOW,
        reports_dir=tmp_path / "derived" / "reports",
    )

    json_path = tmp_path / "derived" / "reports" / "macro_coverage_2024-03-01.json"
    markdown_path = json_path.with_suffix(".md")
    assert json_path.is_file() and markdown_path.is_file()
    assert json.loads(json_path.read_text(encoding="utf-8"))["api_key_present"] is True
    assert len(report.rows) == 2

    markdown = render_markdown(report)
    assert "CPIAUCSL" in markdown
    assert "no implementado" in markdown


def test_the_cli_exits_with_three_when_the_key_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sin clave, el proceso sale con 3: confundirlo con un éxito sería el peor fallo."""
    monkeypatch.setenv("FRED_API_KEY", "")
    assert MacroSecrets().fred_api_key in (None, "")
    assert EXIT_MISSING_API_KEY == 3

    code = main(
        [
            "--data-root",
            str(tmp_path),
            "--now",
            NOW.isoformat(),
        ]
    )
    assert code == EXIT_MISSING_API_KEY
    assert (tmp_path / "derived" / "reports" / "macro_coverage_2024-03-01.json").is_file()


def test_a_broken_registry_fails_at_start(tmp_path: Path) -> None:
    """Un registro macro mal formado falla al arrancar, no a mitad de la ingesta."""
    broken = tmp_path / "macro.yaml"
    broken.write_text("series: [{series_id: X}]\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="registro macro inválido"):
        load_macro_series(broken)


def test_neither_the_adapter_nor_the_ingest_read_the_clock() -> None:
    """El instante de referencia se pasa desde fuera: ni el adaptador ni la ingesta lo leen."""
    for path in (
        "src/cfdtrader/data/sources/fred_adapter.py",
        "src/cfdtrader/data/macro.py",
    ):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        now_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "now"
        ]
        # `main()` sí lee el reloj una vez, para el `now` por defecto de la ejecución real.
        assert len(now_calls) <= 1, path
