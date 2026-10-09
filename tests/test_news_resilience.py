"""Tests de la tolerancia por fuente de la ingesta de noticias (`#145`).

Antes, un solo `--feed` bloqueado tumbaba el lote entero. Ahora cada fuente se recoge por
separado, su estado se **declara** (`SourceOutcome`) y el resto se escribe igual. El umbral
de fallo está declarado: `1` solo si **ninguna** fuente entregó y alguna falló de verdad.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from cfdtrader.data import news
from cfdtrader.data.news import NewsReport, SourceOutcome, collect
from cfdtrader.data.sources.base import (
    SourceBlockedError,
    SourceRateLimitedError,
    SourceStatus,
    SourceUnavailableError,
)
from cfdtrader.data.sources.news import GdeltAdapter, Headline, RssAdapter
from cfdtrader.data.store import Store

NOW = datetime(2026, 10, 9, 12, 45, tzinfo=UTC)


def _headline(title: str, *, minutes: int = 5) -> Headline:
    """Un titular válido, ya publicado antes de `NOW`."""
    return Headline(
        source="rss",
        feed="cnbc",
        title=title,
        url=f"https://example.invalid/{title}",
        published_at=NOW - timedelta(minutes=minutes),
    )


class _FakeGdelt:
    """Adaptador GDELT de mentira: por consulta, titulares o la excepción a levantar."""

    def __init__(self, behaviour: dict[str, object]) -> None:
        self._behaviour = behaviour

    def fetch(self, *, query: str, now: datetime) -> list[Headline]:
        result = self._behaviour[query]
        if isinstance(result, BaseException):
            raise result
        return list(cast("list[Headline]", result))


class _FakeRss:
    """Adaptador RSS de mentira: por etiqueta, titulares o la excepción a levantar."""

    def __init__(self, behaviour: dict[str, object]) -> None:
        self._behaviour = behaviour

    def fetch(self, *, feed_url: str, label: str, now: datetime) -> list[Headline]:
        result = self._behaviour[label]
        if isinstance(result, BaseException):
            raise result
        return list(cast("list[Headline]", result))


def _collect(
    *, gdelt: dict[str, object] | None = None, rss: dict[str, object] | None = None
) -> tuple[list[Headline], tuple[SourceOutcome, ...]]:
    """`collect` con los dos adaptadores simulados y el orden declarado."""
    gdelt_behaviour = gdelt or {}
    rss_behaviour = rss or {}
    return collect(
        queries=list(gdelt_behaviour),
        feeds=[(label, f"https://example.invalid/{label}") for label in rss_behaviour],
        now=NOW,
        gdelt=cast("GdeltAdapter", _FakeGdelt(gdelt_behaviour)),
        rss=cast("RssAdapter", _FakeRss(rss_behaviour)),
    )


def _adapter(cls: type, behaviour: dict[str, object]):  # type: ignore[no-untyped-def]
    """Fábrica que imita al adaptador real: `main` lo construye con el cliente."""

    def factory(client: object) -> object:
        return cls(behaviour)

    return factory


# ─────────────────────────────────────────────────────────────────────────────
# A1 · El estado declarado por fuente
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_source_outcome_declares_the_state_and_forbids_extra_fields() -> None:
    """A1: `SourceOutcome` lleva ref, kind, status, recuento y error; sin campos de más."""
    outcome = SourceOutcome(ref="cnbc", kind="rss", status=SourceStatus.BLOCKED, error="403")
    assert (outcome.ref, outcome.kind, outcome.headlines, outcome.error) == (
        "cnbc",
        "rss",
        0,
        "403",
    )

    with pytest.raises(ValueError):
        # el modelo prohibe los campos de mas: se comprueba a proposito
        SourceOutcome(
            ref="x",
            kind="rss",
            status=SourceStatus.OK,
            invented=1,  # pyright: ignore[reportCallIssue]
        )

    report = NewsReport(
        as_of=NOW.isoformat(),
        fetched=0,
        discarded_future=0,
        discarded_duplicate=0,
        new=0,
        outcome="unchanged",
    )
    assert report.sources == ()


# ─────────────────────────────────────────────────────────────────────────────
# A2 · Una fuente bloqueada no tumba el lote
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_a_blocked_feed_does_not_take_down_the_batch(tmp_path: Path) -> None:
    """A2: `blocked` se declara y las demás fuentes se escriben igual."""
    headlines, outcomes = _collect(
        rss={
            "blocked": SourceBlockedError("Access Denied", source="news"),
            "good": [_headline("Bueno")],
        }
    )

    assert [outcome.status for outcome in outcomes] == [SourceStatus.BLOCKED, SourceStatus.OK]
    assert [headline.title for headline in headlines] == ["Bueno"]

    store = Store(tmp_path)
    report = news.ingest(store=store, headlines=headlines, now=NOW, sources=outcomes)

    assert report.new == 1
    assert [outcome.status for outcome in report.sources] == [
        SourceStatus.BLOCKED,
        SourceStatus.OK,
    ]
    assert Store(tmp_path).sql("SELECT count(*) AS n FROM raw.news_headlines").item(0, "n") == 1


# ─────────────────────────────────────────────────────────────────────────────
# A3 · El mapeo de excepción a estado
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (SourceRateLimitedError("429", source="news"), SourceStatus.RATE_LIMITED),
        (SourceBlockedError("<html>", source="news"), SourceStatus.BLOCKED),
        (SourceUnavailableError("sin datos", source="news"), SourceStatus.UNAVAILABLE),
        (RuntimeError("boom"), SourceStatus.ERROR),
    ],
)
def test_a3_the_exception_maps_to_its_declared_state(error: Exception, expected: str) -> None:
    """A3: cada error tipado tiene su estado; lo inclasificable es `error`, no un dato."""
    _, outcomes = _collect(rss={"x": error})

    assert outcomes[0].status is expected
    assert outcomes[0].error
    if expected is SourceStatus.ERROR:
        assert "RuntimeError" in str(outcomes[0].error)


# ─────────────────────────────────────────────────────────────────────────────
# A4/A5 · Fuente vacía vs fuente con titulares
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_a_feed_without_headlines_is_unavailable_and_invents_no_rows() -> None:
    """A4: responder sin titulares es `unavailable` con recuento 0."""
    headlines, outcomes = _collect(rss={"quiet": []})

    assert headlines == []
    assert outcomes[0].status is SourceStatus.UNAVAILABLE
    assert outcomes[0].headlines == 0


def test_a5_a_feed_with_headlines_is_ok_with_its_count() -> None:
    """A5: entregar titulares es `ok` con su recuento."""
    headlines, outcomes = _collect(rss={"good": [_headline("A"), _headline("B")]})

    assert len(headlines) == 2
    assert outcomes[0].status is SourceStatus.OK
    assert outcomes[0].headlines == 2


# ─────────────────────────────────────────────────────────────────────────────
# A6/A7 · El CLI: escribe si algo entregó; falla solo si nada entregó y algo falló
# ─────────────────────────────────────────────────────────────────────────────
def _run_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    feeds: dict[str, object],
    queries: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    """Ejecuta el `main` real con los adaptadores simulados y devuelve (código, informe)."""

    class _Context:
        def __enter__(self) -> _Context:
            return self

        def __exit__(self, *arguments: object) -> None:
            return None

    def _client(**_kwargs: object) -> _Context:
        return _Context()

    monkeypatch.setattr(news, "CachedHttpClient", _client)
    monkeypatch.setattr(news, "GdeltAdapter", _adapter(_FakeGdelt, queries or {}))
    monkeypatch.setattr(news, "RssAdapter", _adapter(_FakeRss, feeds))

    argv = ["--as-of", NOW.isoformat(), "--data-root", str(tmp_path)]
    for label in feeds:
        argv += ["--feed", f"{label}=https://example.invalid/{label}"]
    for query in queries or {}:
        argv += ["--query", query]

    code = news.main(argv)
    return code, cast("dict[str, object]", json.loads(capsys.readouterr().out))


def _statuses(report: dict[str, object]) -> list[str]:
    outcomes = cast("list[dict[str, object]]", report["sources"])
    return [str(outcome["status"]) for outcome in outcomes]


def test_a6_main_writes_the_batch_when_one_feed_delivers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A6: un feed bloqueado y otro que entrega ⇒ se escribe y el informe lo declara."""
    code, report = _run_main(
        tmp_path,
        monkeypatch,
        capsys,
        feeds={"blocked": SourceBlockedError("403", source="news"), "good": [_headline("Bueno")]},
    )

    assert code == 0
    assert report["new"] == 1
    assert _statuses(report) == ["blocked", "ok"]
    assert Store(tmp_path).sql("SELECT count(*) AS n FROM raw.news_headlines").item(0, "n") == 1


def test_a7_main_fails_only_when_nothing_delivered_and_something_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A7: ninguna fuente entregó y alguna falló ⇒ `rc=1`, sin escribir filas."""
    code, report = _run_main(
        tmp_path,
        monkeypatch,
        capsys,
        feeds={
            "a": SourceBlockedError("403", source="news"),
            "b": SourceRateLimitedError("429", source="news"),
        },
    )

    assert code == 1
    assert report["new"] == 0
    assert "news_headlines" not in Store(tmp_path).datasets("raw")


def test_a7_a_batch_of_empty_sources_is_a_quiet_day_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A7: todas `unavailable` ⇒ `rc=0`: un día sin noticias no es un fallo."""
    code, report = _run_main(tmp_path, monkeypatch, capsys, feeds={"a": [], "b": []})

    assert code == 0
    assert report["new"] == 0
    assert _statuses(report) == ["unavailable", "unavailable"]


# ─────────────────────────────────────────────────────────────────────────────
# A8/A9/A10 · Orden determinista, escritura intacta y GDELT tolerado
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_the_outcomes_follow_the_declared_order_and_are_json() -> None:
    """A8: primero las consultas GDELT y luego los feeds, en el orden declarado."""
    _, outcomes = _collect(
        gdelt={"q1": [_headline("G")], "q2": []},
        rss={"b": [_headline("R")], "a": []},
    )

    assert [(outcome.kind, outcome.ref) for outcome in outcomes] == [
        ("gdelt", "q1"),
        ("gdelt", "q2"),
        ("rss", "b"),
        ("rss", "a"),
    ]
    payload = json.dumps([outcome.model_dump() for outcome in outcomes])
    assert '"ok"' in payload
    assert '"unavailable"' in payload


def test_a9_the_write_step_and_the_store_schema_are_unchanged(tmp_path: Path) -> None:
    """A9: `ingest` sigue escribiendo lo mismo y **no** añade columnas al almacén."""
    store = Store(tmp_path)
    report = news.ingest(store=store, headlines=[_headline("Uno")], now=NOW)

    assert report.sources == ()
    columns = set(store.sql("SELECT * FROM raw.news_headlines").columns)
    assert columns == {
        "source",
        "series_id",
        "as_of",
        "fetched_at",
        "published_at",
        "version",
        "title",
        "url",
        "headline_hash",
    }


def test_a10_a_gdelt_query_failure_is_tolerated_with_kind_gdelt() -> None:
    """A10: un `--query` caído se declara como `gdelt` y no tumba los feeds."""
    headlines, outcomes = _collect(
        gdelt={"iran": SourceRateLimitedError("429", source="news")},
        rss={"x": [_headline("R")]},
    )

    assert outcomes[0].kind == "gdelt"
    assert outcomes[0].status is SourceStatus.RATE_LIMITED
    assert [headline.title for headline in headlines] == ["R"]
