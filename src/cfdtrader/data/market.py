"""Ingesta de datos de mercado del S&P 500 y su contexto — tarea #3.

Único comando de la tarea (A14)::

    uv run python -m cfdtrader.data.market --data-root data

Qué hace, en orden:

1. Carga y valida la configuración **al arrancar** (``tech_stack.md`` §4.2).
2. Recorre el registro de series y, por cada una, intenta la fuente primaria y
   sus respaldos. **Una fuente que falla no aborta el resto**: cada serie termina
   con su estado, sus intentos y su motivo.
3. Descarta la sesión o la barra en curso (A11) y valida la calidad (A12).
4. Escribe en ``raw`` con ``append``/``append_revision``; ``replace`` no se usa
   (solo vale para ``derived``).
5. Escribe el informe de cobertura en ``<raíz>/derived/reports/``.

La regla de oro: ``SPX500:CFD`` **no se sustituye** por ``^GSPC``, ``ES=F`` ni
``SPY``. La Fase 1 queda bloqueada de forma comprobable (``phase1_ready: false``
+ ``--require-ready`` ⇒ salida 2) y, aun así, el diario y el intradía del índice
y del contexto **se escriben de verdad**.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
import duckdb
import httpx
from loguru import logger

from cfdtrader.data.coverage import CoverageReport, SeriesOutcome, build_report, write_report
from cfdtrader.data.quality import QualityReport, validate
from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.sources.base import (
    FetchRequest,
    FetchResult,
    SeriesSpec,
    SourceAdapter,
    SourceStatus,
)
from cfdtrader.data.sources.http import CachedHttpClient
from cfdtrader.data.sources.registry import SeriesRegistry, load_registry
from cfdtrader.data.sources.stooq_adapter import StooqAdapter
from cfdtrader.data.sources.yfinance_adapter import YFinanceAdapter
from cfdtrader.data.store import ImmutableWriteError, Store, UnknownDatasetError, WriteOutcome

__all__ = ["EXIT_NOT_READY", "EXIT_OK", "build_adapters", "ingest", "main"]

#: Salida normal: la ingesta se hizo, aunque la Fase 1 siga bloqueada (A21).
EXIT_OK: Final[int] = 0

#: Salida con ``--require-ready`` cuando la Fase 1 está bloqueada (A21).
EXIT_NOT_READY: Final[int] = 2

#: Columnas de *payload* por dataset (A8). Las seis obligatorias las añade el almacén.
PAYLOAD_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "market_daily": ("open", "high", "low", "close", "volume", "adj_close"),
    "sectors": ("open", "high", "low", "close", "volume", "adj_close"),
    "market_intraday": ("open", "high", "low", "close", "volume", "interval", "bid", "ask"),
}


def ingest(
    *,
    registry: SeriesRegistry,
    data_root: Path,
    adapters: Mapping[str, SourceAdapter],
    now: datetime,
    reports_dir: Path | None = None,
) -> CoverageReport:
    """Ejecuta la ingesta completa y devuelve el informe (sin escribirlo)."""
    store = Store(data_root)
    outcomes: list[SeriesOutcome] = []

    for spec in registry.series:
        outcomes.append(_ingest_series(spec, store=store, adapters=adapters, now=now))

    report = build_report(
        outcomes=outcomes,
        unavailable=registry.unavailable,
        now=now,
        data_root=data_root,
    )
    if reports_dir is not None:
        json_path, markdown_path = write_report(report, reports_dir)
        logger.info("informe de cobertura: {} y {}", json_path, markdown_path)
    return report


def _ingest_series(
    spec: SeriesSpec,
    *,
    store: Store,
    adapters: Mapping[str, SourceAdapter],
    now: datetime,
) -> SeriesOutcome:
    """Intenta una serie en sus fuentes, en orden, y escribe lo que llegue."""
    notes: list[str] = []
    attempts = 0
    last_error: str | None = None

    for source in spec.sources:
        adapter = adapters.get(source)
        if adapter is None:
            notes.append(f"{source}: no hay adaptador declarado")
            continue
        request = FetchRequest(spec=spec, now=now, start=spec.min_start)
        result = adapter.fetch(request)
        attempts += result.attempts
        notes.extend(f"{source}: {note}" for note in result.notes)

        if not result.ok:
            last_error = result.error or result.status.value
            notes.append(f"{source}: {result.status.value} — {last_error}")
            continue

        outcome = _write_result(
            result, store=store, now=now, attempts=attempts, notes=tuple(notes)
        )
        if outcome is not None:
            return outcome
        notes.append(f"{source}: sin filas utilizables tras el control de calidad")
        last_error = "sin filas utilizables"

    status = SourceStatus.UNAVAILABLE if last_error is not None else SourceStatus.ERROR
    return SeriesOutcome(
        spec=spec,
        source=spec.primary,
        status=status,
        attempts=max(attempts, 1),
        notes=tuple((*notes, last_error or "no se intentó ninguna fuente")),
    )


def _write_result(
    result: FetchResult,
    *,
    store: Store,
    now: datetime,
    attempts: int,
    notes: tuple[str, ...],
) -> SeriesOutcome | None:
    """Descarta lo abierto, valida, escribe y devuelve el resultado de la serie."""
    spec = result.spec
    dataset = spec.dataset

    # (A11) La sesión o la barra en curso no se escribe a medias.
    known = result.frame.filter(pl.col("as_of") <= pl.lit(now))
    discarded = result.frame.height - known.height

    quality: QualityReport = validate(known, spec=spec, source=result.source, now=now)
    frame = quality.accepted
    if frame.height == 0:
        return SeriesOutcome(
            spec=spec,
            source=result.source,
            status=SourceStatus.UNAVAILABLE,
            attempts=attempts,
            discarded_open_session=discarded,
            rejected_rows=quality.rejected_rows,
            issue_codes=quality.issue_codes,
            notes=tuple((*notes, "todas las filas se rechazaron en el control de calidad")),
        )

    records = _records(frame, spec=spec, source=result.source, now=now, dataset=dataset)
    payload = PAYLOAD_COLUMNS[dataset]
    stored = _stored_payload(
        store, dataset, source=result.source, series_id=spec.series_id, columns=payload
    )
    new, revised = _split(records, stored, payload)
    before = _count_rows(store, dataset, source=result.source, series_id=spec.series_id)
    outcome = _write(store, dataset, new=new, revised=revised)
    after = _count_rows(store, dataset, source=result.source, series_id=spec.series_id)

    span, written = _span(store, dataset, source=result.source, series_id=spec.series_id)
    if written == 0:  # pragma: no cover - defensivo: el almacén siempre devuelve algo
        return None

    return SeriesOutcome(
        spec=spec,
        source=result.source,
        status=SourceStatus.OK,
        attempts=attempts,
        rows_written=written,
        rows_new=max(after - before, 0),
        span_start=span[0],
        span_end=span[1],
        stale=quality.stale,
        stale_dates=quality.stale_dates,
        gaps=quality.gaps,
        discarded_open_session=discarded,
        rejected_rows=quality.rejected_rows,
        issue_codes=quality.issue_codes,
        issues=tuple(_issue_payload(quality)),
        notes=tuple((*notes, *quality.notes, f"escritura={outcome}")),
    )


def _records(
    frame: pl.DataFrame, *, spec: SeriesSpec, source: str, now: datetime, dataset: str
) -> list[dict[str, object]]:
    """Convierte el frame validado en registros del almacén para ese dataset."""
    try:
        payload = PAYLOAD_COLUMNS[dataset]
    except KeyError as error:  # pragma: no cover - lo impide la validación del registro
        raise ConfigurationError(
            f"el dataset {dataset!r} no tiene columnas declaradas en PAYLOAD_COLUMNS"
        ) from error
    records: list[dict[str, object]] = []
    for row in frame.iter_rows(named=True):
        record: dict[str, object] = {
            "source": source,
            "series_id": spec.series_id,
            "as_of": row["as_of"],
            "fetched_at": now,
            # La fuente no publica marca por barra: NULL, nunca `fetched_at` (A8).
            "published_at": None,
        }
        for name in payload:
            record[name] = row.get(name)
        records.append(record)
    return records


def _stored_payload(
    store: Store, dataset: str, *, source: str, series_id: str, columns: Sequence[str]
) -> dict[object, tuple[object, ...]]:
    """Estado vigente del almacén para esa serie: ``as_of`` → columnas de *payload*.

    Una sola consulta por serie. Comparar en memoria es lo que evita reenviar el
    histórico completo —5.461 filas por serie— solo para que el almacén descubra
    que ya lo tiene: su comprobación de identidad es registro a registro.
    """
    selected = ", ".join(("as_of", *columns))
    query = (
        f"SELECT {selected} FROM raw.{dataset} "
        f"WHERE source = {_literal(source)} AND series_id = {_literal(series_id)}"
    )
    try:
        frame = store.sql(query)
    except (UnknownDatasetError, duckdb.Error):
        # Dataset todavía inexistente: no hay nada guardado que comparar.
        return {}
    return {
        row["as_of"]: tuple(row[name] for name in columns)
        for row in frame.iter_rows(named=True)
    }


def _split(
    records: Sequence[dict[str, object]],
    stored: dict[object, tuple[object, ...]],
    columns: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Separa lo nuevo de lo **revisado** comparando con el estado vigente.

    Se comparan **todas** las columnas que se envían: una diferencia en cualquiera
    de ellas es contenido distinto para el almacén, y dejarla fuera sería perder
    una revisión silenciosamente.
    """
    new: list[dict[str, object]] = []
    revised: list[dict[str, object]] = []
    for record in records:
        previous = stored.get(record["as_of"])
        if previous is None:
            new.append(record)
        elif tuple(record[name] for name in columns) != previous:
            revised.append(record)
    return new, revised


def _write(
    store: Store,
    dataset: str,
    *,
    new: Sequence[dict[str, object]],
    revised: Sequence[dict[str, object]],
) -> str:
    """Escribe solo lo que cambia: lo nuevo con ``append``, lo revisado con ``append_revision``.

    Enviar el histórico completo en cada ejecución es *correcto* (el almacén
    deduplica) pero carísimo: la ejecución diaria reenviaría ~110.000 filas para
    añadir veinte. Aquí se decide una vez por serie.

    El parche de ``ImmutableWriteError`` no es decorativo: si otro proceso escribió
    la misma identidad entre la comparación y la escritura, ``append`` falla y la
    vía correcta es la revisión (A10), nunca perder el dato.
    """
    outcomes: list[str] = []
    if new:
        try:
            outcomes.append(str(store.append("raw", dataset, list(new))))
        except ImmutableWriteError:
            logger.info("{}: identidad escrita por otro proceso; se escribe como revisión", dataset)
            outcomes.append(str(store.append_revision("raw", dataset, list(new))))
    if revised:
        logger.info("{}: la fuente revisó {} filas", dataset, len(revised))
        outcomes.append(str(store.append_revision("raw", dataset, list(revised))))
    if not outcomes:
        return WriteOutcome.UNCHANGED.value
    return "; ".join(outcomes)


def _count_rows(store: Store, dataset: str, *, source: str, series_id: str) -> int:
    """Filas vigentes en el almacén para ``(source, series_id)``."""
    query = (
        f"SELECT count(*) AS n FROM raw.{dataset} "
        f"WHERE source = {_literal(source)} AND series_id = {_literal(series_id)}"
    )
    try:
        frame = store.sql(query)
    except (UnknownDatasetError, duckdb.Error):
        # Un dataset que todavía no existe cuenta como 0 filas (no se crean vacíos, A7).
        return 0
    if frame.height == 0:
        return 0
    value = frame.item(0, "n")
    return int(value) if isinstance(value, (int, float)) else 0


def _span(
    store: Store, dataset: str, *, source: str, series_id: str
) -> tuple[tuple[str | None, str | None], int]:
    """Primera y última fecha vigentes, y cuántas filas hay, para ``(source, series_id)``."""
    query = (
        f"SELECT min(as_of) AS first, max(as_of) AS last, count(*) AS n FROM raw.{dataset} "
        f"WHERE source = {_literal(source)} AND series_id = {_literal(series_id)}"
    )
    try:
        frame = store.sql(query)
    except (UnknownDatasetError, duckdb.Error):
        return (None, None), 0
    if frame.height == 0:
        return (None, None), 0
    row = frame.to_dicts()[0]
    count = row.get("n")
    return (
        (_as_text(row.get("first")), _as_text(row.get("last"))),
        int(count) if isinstance(count, (int, float)) else 0,
    )


def _as_text(value: object) -> str | None:
    return None if value is None else str(value)


def _literal(value: str) -> str:
    """Literal SQL seguro para los valores que vienen del registro."""
    return "'" + value.replace("'", "''") + "'"


def _issue_payload(quality: QualityReport) -> list[dict[str, object]]:
    """Los problemas de calidad, tal cual, para que el informe no los resuma de más."""
    return [
        {
            "code": issue.code,
            "dates": list(issue.dates),
            "rows": issue.rows,
            "detail": issue.detail,
        }
        for issue in quality.issues
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Cableado y CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_adapters(
    *, cache_root: Path | None, client: httpx.Client | None = None
) -> dict[str, SourceAdapter]:
    """Construye los adaptadores declarados.

    ``client`` permite inyectar el cliente HTTP en los tests (``httpx.MockTransport``).
    """
    stooq_client = CachedHttpClient(source="stooq", client=client, cache_root=cache_root)
    return {
        "yfinance": YFinanceAdapter(),
        "stooq": StooqAdapter(stooq_client),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada. Devuelve el código de salida del proceso."""
    parser = argparse.ArgumentParser(prog="cfdtrader.data.market", description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="raíz del almacén")
    parser.add_argument("--settings", type=Path, default=None, help="ruta de settings.yaml")
    parser.add_argument("--registry", type=Path, default=None, help="ruta de data_sources.yaml")
    parser.add_argument("--now", default=None, help="instante de referencia ISO (tests)")
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="salir con 2 si la Fase 1 sigue bloqueada por falta de fuente del CFD",
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.settings)
        registry = load_registry(args.registry)
    except ConfigurationError as error:
        logger.error("configuración inválida: {}", error)
        return 1

    data_root = args.data_root if args.data_root is not None else settings.data.root
    now = _parse_now(args.now)

    report = ingest(
        registry=registry,
        data_root=data_root,
        adapters=build_adapters(cache_root=data_root / "cache"),
        now=now,
        reports_dir=data_root / "derived" / "reports",
    )

    logger.info("series escritas: {} diario, {} intradía", len(report.daily), len(report.intraday))
    if not report.phase1_ready:
        for blocker in report.blockers:
            logger.warning("Fase 1 bloqueada: {} — {}", blocker.code, blocker.series_id)
    if args.require_ready and not report.phase1_ready:
        return EXIT_NOT_READY
    return EXIT_OK


def _parse_now(value: str | None) -> datetime:
    """Instante de referencia: el de verdad, o el que fije el test."""
    if value is None:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
