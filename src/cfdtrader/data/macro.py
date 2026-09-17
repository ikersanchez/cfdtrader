"""Ingesta de macro americana desde FRED — tarea #5.

Comando único::

    uv run python -m cfdtrader.data.macro --data-root data

Escribe ``raw.macro`` (serie × fecha) con ``as_of`` = periodo observado y
``published_at`` = instante de publicación en UTC, y deja un informe en
``<raíz>/derived/reports/macro_coverage_<AAAA-MM-DD>.json|.md``.

Tres reglas que no se negocian:

- **El dato no existe antes de publicarse.** Una serie de las 08:30 ET no se
  escribe como visible en el snapshot de las 08:45 ET del día anterior, y una
  observación cuyo ``published_at`` aún no ha llegado **no se escribe**.
- **Nunca se rellena ``published_at`` con ``fetched_at``.** Si no se puede
  determinar el instante de publicación, queda ``NULL`` y el almacén cae a
  ``fetched_at`` para decidir visibilidad.
- **Sin clave no se inventa nada.** Si falta ``FRED_API_KEY`` cada serie se
  declara ``unavailable``, el informe se escribe y el proceso sale con **3**
  para que nadie confunda «no hay clave» con «ya está ingestado».
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import duckdb
import httpx
import polars as pl
import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from cfdtrader.data.settings import (
    DEFAULT_MACRO_SERIES_PATH,
    REPO_ROOT,
    ConfigurationError,
    load_settings,
)
from cfdtrader.data.sources.base import SourceStatus
from cfdtrader.data.sources.fred_adapter import (
    FredAdapter,
    MacroFetchResult,
    MacroSeriesRegistry,
    MacroSeriesSpec,
)
from cfdtrader.data.sources.http import CachedHttpClient
from cfdtrader.data.store import ImmutableWriteError, Store, UnknownDatasetError

__all__ = [
    "EXIT_CONFIG_ERROR",
    "EXIT_MISSING_API_KEY",
    "EXIT_OK",
    "MacroRow",
    "MacroReport",
    "ingest",
    "load_macro_series",
    "main",
]

EXIT_OK: Final[int] = 0
EXIT_CONFIG_ERROR: Final[int] = 1
EXIT_MISSING_API_KEY: Final[int] = 3


class MacroSecrets(BaseSettings):
    """Claves del entorno y del ``.env`` (que está en ``.gitignore``)."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    fred_api_key: str | None = None


class MacroRow(BaseModel):
    """Una fila del informe macro: qué pasó con esa serie."""

    model_config = ConfigDict(extra="forbid")

    series_id: str
    name: str
    source: str
    status: SourceStatus
    attempts: int
    rows_written: int
    rows_new: int
    span_start: str | None = None
    span_end: str | None = None
    last_published_at: str | None = None
    first_published_at: str | None = None
    missing_values: int = 0
    discarded_unpublished: int = 0
    revisions: int = 0
    notes: tuple[str, ...] = ()


class MacroReport(BaseModel):
    """Informe de cobertura macro."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    data_root: str
    api_key_present: bool
    rows: tuple[MacroRow, ...] = ()
    unavailable_european_context: tuple[dict[str, object], ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class _Outcome:
    """Acumulador por serie antes de convertirse en fila del informe."""

    spec: MacroSeriesSpec
    status: SourceStatus
    attempts: int = 1
    rows_written: int = 0
    rows_new: int = 0
    span: tuple[str | None, str | None] = (None, None)
    published: tuple[str | None, str | None] = (None, None)
    missing_values: int = 0
    discarded_unpublished: int = 0
    revisions: int = 0
    notes: tuple[str, ...] = field(default=())


def load_macro_series(path: Path | str | None = None) -> MacroSeriesRegistry:
    """Carga y valida ``config/macro_series.yaml``."""
    target = Path(path) if path is not None else DEFAULT_MACRO_SERIES_PATH
    if not target.is_file():
        raise ConfigurationError(f"no existe el registro de series macro: {target}")
    try:
        loaded: object = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"YAML inválido en {target}: {error}") from error
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"{target} debe contener un mapping en la raíz")
    try:
        return MacroSeriesRegistry.model_validate(loaded)
    except ValidationError as error:
        raise ConfigurationError(f"registro macro inválido en {target}: {error}") from error


def ingest(
    *,
    registry: MacroSeriesRegistry,
    data_root: Path,
    adapter: FredAdapter,
    now: datetime,
    reports_dir: Path | None = None,
    api_key_present: bool = True,
) -> MacroReport:
    """Ingesta todas las series declaradas y devuelve el informe."""
    store = Store(data_root)
    outcomes = [
        _ingest_series(spec, store=store, adapter=adapter, now=now) for spec in registry.series
    ]
    report = MacroReport(
        generated_at=now,
        data_root=str(data_root),
        api_key_present=api_key_present,
        rows=tuple(_row(outcome) for outcome in outcomes),
        unavailable_european_context=tuple(
            {str(key): value for key, value in item.items()} for item in registry.european_context
        ),
        notes=(
            "`rows_written` son las filas vigentes en `raw.macro` para esa serie; `rows_new`, las de esta ejecución.",
            "El contexto europeo (ECB SDW, Eurostat) está declarado como no implementado: no es la columna vertebral.",
        ),
    )
    if reports_dir is not None:
        json_path, markdown_path = write_report(report, reports_dir)
        logger.info("informe macro: {} y {}", json_path, markdown_path)
    return report


def _ingest_series(
    spec: MacroSeriesSpec, *, store: Store, adapter: FredAdapter, now: datetime
) -> _Outcome:
    """Descarga una serie, descarta lo no publicado y la escribe en ``raw.macro``."""
    result: MacroFetchResult = adapter.fetch(spec, now=now)
    if not result.ok:
        return _Outcome(
            spec=spec,
            status=result.status,
            attempts=result.attempts,
            notes=tuple((*result.notes, result.error or result.status.value)),
        )

    # Una observación cuyo instante de publicación todavía no ha llegado no se
    # escribe: el almacén exige `published_at <= fetched_at` y, sobre todo, no se
    # puede saber lo que aún no se ha publicado.
    known = result.frame.filter(
        pl.col("published_at").is_null() | (pl.col("published_at") <= pl.lit(now))
    )
    discarded = result.frame.height - known.height
    missing = int(known.filter(pl.col("value").is_null()).height)
    writable = known.filter(pl.col("value").is_not_null())
    if writable.height == 0:
        return _Outcome(
            spec=spec,
            status=SourceStatus.UNAVAILABLE,
            attempts=result.attempts,
            discarded_unpublished=discarded,
            missing_values=missing,
            notes=tuple((*result.notes, "no quedó ninguna observación publicada con valor")),
        )

    records: list[dict[str, object]] = [
        {
            "source": result.source,
            "series_id": spec.series_id,
            "as_of": row["as_of"],
            "fetched_at": now,
            "published_at": row["published_at"],
            "value": row["value"],
            "unit": spec.unit,
            "name": spec.name,
        }
        for row in writable.iter_rows(named=True)
    ]
    before = _count(store, series_id=spec.series_id, source=result.source)
    revisions = _write(store, records)
    written, span, published = _stats(store, series_id=spec.series_id, source=result.source)

    return _Outcome(
        spec=spec,
        status=SourceStatus.OK,
        attempts=result.attempts,
        rows_written=written,
        rows_new=max(written - before, 0),
        span=span,
        published=published,
        missing_values=missing,
        discarded_unpublished=discarded,
        revisions=revisions,
        notes=tuple((*result.notes, f"dataset={spec.dataset}")),
    )


def _write(store: Store, records: list[dict[str, object]]) -> int:
    """Escribe en ``raw.macro``; las revisiones de la fuente van con ``append_revision``.

    FRED revisa CPI, PCE y NFP: es el caso de uso para el que existe
    ``append_revision``. ``replace`` no se usa nunca aquí (solo vale para ``derived``).

    Returns
    -------
    int
        Número de registros escritos como revisión.
    """
    try:
        store.append("raw", "macro", records)
        return 0
    except ImmutableWriteError:
        revisions = _count_revisions(store, records)
        logger.info("macro: {} observaciones revisadas por la fuente", revisions)
        store.append_revision("raw", "macro", records)
        return revisions


def _count_revisions(store: Store, records: list[dict[str, object]]) -> int:
    """Cuántas observaciones **cambian** un valor ya almacenado.

    No basta con mirar si la identidad existe: reenviar el mismo valor es un
    no-op, y contarlo como revisión inflaría el informe.
    """
    try:
        stored = store.sql("SELECT series_id, as_of, value FROM raw.macro").to_dicts()
    except (UnknownDatasetError, duckdb.Error):
        return 0
    current = {(str(row["series_id"]), str(row["as_of"])): row["value"] for row in stored}
    changed = 0
    for record in records:
        key = (str(record["series_id"]), str(record["as_of"]))
        if key in current and current[key] != record["value"]:
            changed += 1
    return changed


def _count(store: Store, *, series_id: str, source: str) -> int:
    """Filas vigentes en ``raw.macro`` para esa serie."""
    return _aggregate(store, series_id=series_id, source=source)[0]


def _stats(
    store: Store, *, series_id: str, source: str
) -> tuple[int, tuple[str | None, str | None], tuple[str | None, str | None]]:
    """Filas vigentes, ventana observada y ventana de publicación."""
    count, first, last, first_pub, last_pub = _aggregate(store, series_id=series_id, source=source)
    return count, (first, last), (first_pub, last_pub)


def _aggregate(
    store: Store, *, series_id: str, source: str
) -> tuple[int, str | None, str | None, str | None, str | None]:
    """Recuento y extremos de la serie en el almacén, o ceros si no existe."""
    query = (
        "SELECT count(*) AS n, min(as_of) AS first, max(as_of) AS last, "
        "min(published_at) AS first_pub, max(published_at) AS last_pub "
        "FROM raw.macro "
        f"WHERE series_id = {_literal(series_id)} AND source = {_literal(source)}"
    )
    try:
        frame = store.sql(query)
    except (UnknownDatasetError, duckdb.Error):
        return 0, None, None, None, None
    if frame.height == 0:
        return 0, None, None, None, None
    row = frame.to_dicts()[0]
    count = row.get("n")
    return (
        int(count) if isinstance(count, (int, float)) else 0,
        _text(row.get("first")),
        _text(row.get("last")),
        _text(row.get("first_pub")),
        _text(row.get("last_pub")),
    )


def _literal(value: str) -> str:
    """Literal SQL seguro: los identificadores vienen del registro validado."""
    return "'" + value.replace("'", "''") + "'"


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _row(outcome: _Outcome) -> MacroRow:
    """Convierte el acumulador en fila del informe."""
    return MacroRow(
        series_id=outcome.spec.series_id,
        name=outcome.spec.name,
        source=FredAdapter.name,
        status=outcome.status,
        attempts=outcome.attempts,
        rows_written=outcome.rows_written,
        rows_new=outcome.rows_new,
        span_start=outcome.span[0],
        span_end=outcome.span[1],
        first_published_at=outcome.published[0],
        last_published_at=outcome.published[1],
        missing_values=outcome.missing_values,
        discarded_unpublished=outcome.discarded_unpublished,
        revisions=outcome.revisions,
        notes=outcome.notes,
    )


def write_report(report: MacroReport, directory: Path) -> tuple[Path, Path]:
    """Escribe el informe macro en JSON y Markdown."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"macro_coverage_{report.generated_at.date().isoformat()}"
    json_path = directory / f"{stem}.json"
    markdown_path = directory / f"{stem}.md"
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(report: MacroReport) -> str:
    """Informe macro legible."""
    lines = [
        "# Informe de cobertura macro (FRED)",
        "",
        f"- **Generado:** {report.generated_at.isoformat()}",
        f"- **Raíz del almacén:** `{report.data_root}`",
        f"- **FRED_API_KEY presente:** {'sí' if report.api_key_present else 'no'}",
        "",
        "| serie | nombre | estado | intentos | filas guardadas | filas nuevas | ventana | "
        "publicado entre | revisadas | valores ausentes | descartadas | notas |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report.rows:
        span = f"{row.span_start or '—'} → {row.span_end or '—'}"
        published = f"{row.first_published_at or '—'} → {row.last_published_at or '—'}"
        notes = "; ".join(row.notes) or "—"
        lines.append(
            f"| `{row.series_id}` | {row.name} | {row.status.value} | {row.attempts} | "
            f"{row.rows_written} | {row.rows_new} | {span} | {published} | {row.revisions} | "
            f"{row.missing_values} | {row.discarded_unpublished} | {notes} |"
        )
    lines.append("")
    if report.unavailable_european_context:
        lines.extend(["## Contexto europeo declarado como no implementado", ""])
        for item in report.unavailable_european_context:
            lines.append(f"- `{item.get('source')}` — {item.get('reason')}")
        lines.append("")
    lines.extend(["## Notas", ""])
    lines.extend(f"- {note}" for note in report.notes)
    lines.append("")
    return "\n".join(lines)


def build_adapter(*, cache_root: Path | None, client: httpx.Client | None = None) -> FredAdapter:
    """Adaptador de FRED con el cliente HTTP inyectable (los tests no abren red)."""
    http_client = CachedHttpClient(source="fred", client=client, cache_root=cache_root)
    return FredAdapter(http_client, api_key=MacroSecrets().fred_api_key)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del comando de ingesta macro."""
    parser = argparse.ArgumentParser(prog="cfdtrader.data.macro", description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="raíz del almacén")
    parser.add_argument("--settings", type=Path, default=None, help="ruta de settings.yaml")
    parser.add_argument("--series", type=Path, default=None, help="ruta de macro_series.yaml")
    parser.add_argument("--now", default=None, help="instante de referencia ISO (tests)")
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.settings)
        registry = load_macro_series(args.series)
    except ConfigurationError as error:
        logger.error("configuración inválida: {}", error)
        return EXIT_CONFIG_ERROR

    data_root = args.data_root if args.data_root is not None else settings.data.root
    now = _parse_now(args.now)
    secrets = MacroSecrets()
    adapter = build_adapter(cache_root=data_root / "cache")

    report = ingest(
        registry=registry,
        data_root=data_root,
        adapter=adapter,
        now=now,
        reports_dir=data_root / "derived" / "reports",
        api_key_present=bool(secrets.fred_api_key),
    )
    if not secrets.fred_api_key:
        logger.error(
            "falta FRED_API_KEY: no se ha ingestado ninguna serie macro. "
            "Ponla en .env (clave gratuita de FRED) y vuelve a ejecutar."
        )
        return EXIT_MISSING_API_KEY
    logger.info("series macro ingestadas: {}", len(report.rows))
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
