"""Estudio de la **fuente de apertura** del drift (`plan.md` §1.1.a) — tarea #52.

El estudio del drift (#6, :mod:`cfdtrader.analysis.drift`) mide la partición
``open→close`` / ``close→open`` sobre ``^GSPC``. Parte del histórico de Yahoo sirve
la **apertura del índice repetida** (el cierre anterior), así que el tramo
nocturno es cero por construcción en esas sesiones y la sesión se queda con todo
el retorno. Este módulo **no rehace #6**: mide la misma descomposición sobre la
**misma era limpia** para `SPY` (el ETF, con apertura de subasta real del mismo
mercado y horario) y para `ES=F` (el futuro, de sesión casi continua), y **decide
y documenta** qué fuente de apertura usa el estudio.

La regla de la muestra limpia vive **solo** en :mod:`cfdtrader.analysis.drift`:
aquí se importan :func:`~cfdtrader.analysis.drift.load_sessions`,
:func:`~cfdtrader.analysis.drift.decompose`,
:func:`~cfdtrader.analysis.drift.clean_sample` y
:func:`~cfdtrader.analysis.drift.clean_sample_cutoff` y **no** se redefinen. La
ventana es común a las tres fuentes: se ancla con el corte limpio de la serie de
referencia (`^GSPC`) y se aplica idéntica a `SPY` y `ES=F`.

El artefacto se llama ``drift_open_source_<fecha>.{json,md}`` **nunca**
``drift_decomposition_*``: la Fase 0 selecciona el ``drift_decomposition_*.json``
más reciente (:func:`cfdtrader.analysis.phase0_report.select_artifact`) y un
nombre nuevo con ese prefijo se convertiría, en silencio, en su entrada.

Ninguna ruta consulta la red: los datos ya están en el almacén
(``raw.market_daily``). El instante de referencia (`now`/``--now``) es el único
reloj de la corrida; el ``report_sha256`` es el sha256 del texto canónico de
:func:`cfdtrader.backtest.engine.canonical_text` sobre el payload sin esa clave.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final, cast

import polars as pl
from loguru import logger

from cfdtrader.analysis.drift import (
    DriftStudy,
    PairedTest,
    Segment,
    clean_sample,
    clean_sample_cutoff,
    decompose,
    load_sessions,
)
from cfdtrader.backtest.engine import canonical_text
from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.store import Store

__all__ = [
    "DEFAULT_SERIES_IDS",
    "REFERENCE_SERIES",
    "REPORT_PREFIX",
    "DriftOpenSourceError",
    "Report",
    "analyse",
    "main",
    "render_markdown",
]

#: Prefijo del nombre del artefacto. **Nunca** ``drift_decomposition``: la Fase 0
#: seleccionaría el artefacto nuevo como su entrada del drift (criterio 11).
REPORT_PREFIX: Final[str] = "drift_open_source"

#: Serie de apertura del estudio: la decisión de #52 es **mantenerla**.
REFERENCE_SERIES: Final[str] = "^GSPC"

#: Las tres fuentes que se comparan sobre la misma ventana.
DEFAULT_SERIES_IDS: Final[tuple[str, ...]] = ("^GSPC", "SPY", "ES=F")

#: Fuente declarada de las series (``config/data_sources.yaml``).
SOURCE: Final[str] = "yfinance"

#: Prefijo del digest (política de ``detect-secrets``): un digest desnudo se bloquea.
HASH_PREFIX: Final[str] = "sha256:"

#: Número de puntos básicos en un retorno fraccional (``0,0001`` fracción = 1 pb).
BP_TO_PCT: Final[float] = 10_000.0


class DriftOpenSourceError(Exception):
    """No se puede medir la fuente de apertura con los datos disponibles."""


@dataclass(frozen=True, slots=True)
class Report:
    """El informe: payload canónico, digest y los objetos que lo produjeron."""

    as_of: datetime
    report_date: str
    payload: dict[str, object]
    report_sha256: str
    reports_dir: Path | None

    @property
    def report_stem(self) -> str:
        """Nombre base del informe: ``drift_open_source_<AAAA-MM-DD>``."""
        return f"{REPORT_PREFIX}_{self.report_date}"

    def json_text(self) -> str:
        """JSON determinista: mismo ``as_of`` ⇒ mismo texto byte a byte (criterio 12)."""
        published: dict[str, object] = {**self.payload, "report_sha256": self.report_sha256}
        return json.dumps(published, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    def write(self, directory: Path) -> tuple[Path, Path]:
        """Escribe el informe en JSON y Markdown dentro de ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{self.report_stem}.json"
        markdown_path = directory / f"{self.report_stem}.md"
        json_path.write_text(self.json_text(), encoding="utf-8")
        markdown_path.write_text(render_markdown(self), encoding="utf-8")
        return json_path, markdown_path


def _digest(payload: Mapping[str, object]) -> str:
    """``report_sha256``: sha256 del texto canónico de #13, con prefijo ``sha256:``."""
    return HASH_PREFIX + hashlib.sha256(canonical_text(payload).encode("utf-8")).hexdigest()


def _as_utc(value: datetime) -> datetime:
    """Normaliza el instante a UTC; sin zona se interpreta como UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_now(value: str | None) -> datetime:
    """Instante de referencia: el de verdad, o el que fije el test."""
    if value is None:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Bloques del payload
# ─────────────────────────────────────────────────────────────────────────────
def _segment_block(segment: Segment) -> dict[str, object]:
    """Un tramo publicado: nombre, sesiones, media, ``t``, ``p`` y tasa de aciertos."""
    return {
        "name": segment.name,
        "sessions": segment.sessions,
        "mean_bp": segment.mean_bp,
        "t_stat": segment.t_stat,
        "p_value": segment.p_value,
        "hit_rate": segment.hit_rate,
    }


def _difference_block(difference: PairedTest) -> dict[str, object]:
    """La diferencia emparejada ``intraday - overnight`` de un estudio."""
    return {
        "mean_difference_bp": difference.mean_difference_bp,
        "t_stat": difference.t_stat,
        "p_value": difference.p_value,
    }


def _window_block(
    *,
    frame: pl.DataFrame,
    cutoff: date,
    window: DriftStudy,
    full: DriftStudy,
) -> dict[str, object]:
    """La ventana común: tramos de la muestra limpia y el contraste con la muestra completa.

    ``stale_open_share`` se mide sobre la **era** común (desde ``cutoff``) *antes* de
    quitar las sesiones con el ``open`` repetido: es lo que deja ver que `ES=F`
    conserva un 4,91 % de contaminación en la ventana. Los tramos, en cambio, se
    calculan sobre la muestra limpia.
    """
    era = frame.filter(pl.col("session") >= cutoff)
    stale_open_share = _float_of(era.get_column("open_stale").mean())
    return {
        "sessions": window.sessions,
        "stale_open_share": stale_open_share,
        "segments": [_segment_block(segment) for segment in window.segments],
        "difference": _difference_block(window.difference),
        "verdict": str(window.verdict),
        "phase0_gate": window.phase0_gate,
        "all_sessions": {
            "sessions": full.sessions,
            "verdict": str(full.verdict),
            "phase0_gate": full.phase0_gate,
        },
    }


def _source_block(
    *,
    frame: pl.DataFrame,
    series_id: str,
    cutoff: date,
    now: datetime,
) -> dict[str, object]:
    """El registro de una fuente: muestra completa + ventana común (muestra limpia)."""
    full = decompose(frame, series_id=series_id, source=SOURCE, as_of=now)
    clean = clean_sample(frame, cutoff=cutoff)
    if clean is None:
        raise DriftOpenSourceError(
            f"la ventana limpia de {series_id!r} desde {cutoff.isoformat()} no alcanza las 250 "
            "sesiones: no se decide la fuente de apertura con menos datos que eso (regla de #52)"
        )
    window = decompose(clean, series_id=series_id, source=SOURCE, as_of=now)
    return {
        "series_id": series_id,
        "source": SOURCE,
        "full_sessions": full.sessions,
        "first_session": full.first_session,
        "last_session": full.last_session,
        "stale_open_share": full.stale_open_share,
        "open_quality": full.open_quality,
        "window": _window_block(frame=frame, cutoff=cutoff, window=window, full=full),
    }


def _float_of(value: object) -> float:
    """Número de un escalar de polars, o 0.0 si no lo es."""
    return float(value) if isinstance(value, (int, float)) else 0.0


def _mapping(value: object) -> dict[str, object]:
    """Un bloque anidado del payload, ya tipado."""
    return cast("dict[str, object]", value)


def _sequence(value: object) -> list[object]:
    """Una lista del payload, ya tipada."""
    return list(cast("list[object]", value))


def _window_of(source: Mapping[str, object]) -> dict[str, object]:
    """El bloque ``window`` de un registro de fuente."""
    return _mapping(source["window"])


def _verdict_of(block: Mapping[str, object]) -> str:
    """El veredicto publicado de un bloque ``window``."""
    return str(block["verdict"])


def _decision_block(reference: str, by_id: Mapping[str, object]) -> dict[str, object]:
    """La decisión del propietario, redactada con los veredictos medidos (criterio 7)."""
    spy = by_id.get("SPY")
    future = by_id.get("ES=F")
    reference_window = _window_of(_mapping(by_id[reference]))
    parts = [
        f"Se mantiene {reference} como fuente de apertura del estudio (#6): el `open` repetido "
        "del índice queda como límite asumido y declarado, no corregido, porque el veredicto ya se "
        "calcula sobre la era limpia.",
    ]
    if isinstance(spy, Mapping):
        spy_window = _window_of(spy)
        parts.append(
            "SPY, la apertura de subasta real del mismo mercado y horario (09:30–16:00 ET), "
            f"reproduce la referencia sobre la ventana común (`{_verdict_of(spy_window)}`/"
            f"`{spy_window['phase0_gate']}`), así que la confirma."
        )
    if isinstance(future, Mapping):
        future_window = _window_of(future)
        stale = _float_of(future_window["stale_open_share"])
        parts.append(
            "ES=F no es sustituto: es el futuro del índice, de sesión casi continua, así que su "
            "`open→close` no es la sesión regular del instrumento y da otro veredicto "
            f"(`{_verdict_of(future_window)}`/`{future_window['phase0_gate']}`); además arrastra "
            f"un {stale:.2%} de `open` repetido en la ventana."
        )
    if not isinstance(spy, Mapping) and not isinstance(future, Mapping):
        parts.append(
            f"La referencia se mantiene con el veredicto medido "
            f"(`{_verdict_of(reference_window)}`/`{reference_window['phase0_gate']}`)."
        )
    return {"source": reference, "corrected": False, "rationale": " ".join(parts)}


#: Limitaciones declaradas del estudio (no se esconden).
LIMITATIONS: Final[tuple[str, ...]] = (
    "Se mantiene ^GSPC como fuente de apertura: su `open` diario es el cierre anterior repetido "
    "en el histórico antiguo de Yahoo, y el `open` repetido queda como límite asumido y declarado, "
    "no corregido. El veredicto de #6 ya se calcula sobre la era limpia.",
    "ES=F no es sustituto: es el futuro del S&P 500, de sesión casi continua, así que su "
    "`open→close` no es la sesión regular del instrumento (el índice cotiza 09:30–16:00 ET) y "
    "da otro veredicto sobre la misma ventana.",
    "`window.stale_open_share` se mide sobre la era común **antes** de quitar las sesiones con el "
    "`open` repetido (por eso ES=F conserva contaminación en la ventana); los tres tramos sí se "
    "calculan sobre la muestra limpia.",
    "Medición offline sobre el almacén local (`raw.market_daily`): ninguna ruta consulta la red.",
)

#: Notas de método declaradas.
NOTES: Final[tuple[str, ...]] = (
    "La ventana es común a las tres fuentes: se ancla con el corte limpio de la serie de "
    "referencia (^GSPC) y se aplica idéntica a SPY y ES=F.",
    "`all_sessions` es el veredicto de `decompose` sobre la serie completa con su propio corte "
    "determinista; `window` es el de la ventana común, que es sobre el que se decide.",
    "`report_sha256` es el sha256 del texto canónico (`cfdtrader.backtest.engine.canonical_text`) "
    "del payload sin esa clave, con prefijo `sha256:`.",
)


# ─────────────────────────────────────────────────────────────────────────────
# Análisis
# ─────────────────────────────────────────────────────────────────────────────
def analyse(
    *,
    data_root: Path,
    now: datetime,
    series_ids: Sequence[str] = DEFAULT_SERIES_IDS,
    reports_dir: Path | None = None,
) -> Report:
    """Mide la fuente de apertura de la ventana común y escribe el artefacto (criterios 1-7).

    ``series_ids`` debe contener la serie de referencia (``^GSPC``); el corte limpio
    de esa serie es el ``window_from`` común. Si ``reports_dir`` es ``None`` no se
    escribe nada (el payload sigue disponible en memoria). ``now`` es el único
    instante de la corrida: ninguna ruta lee el reloj.
    """
    moment = _as_utc(now)
    ids = tuple(series_ids)
    if REFERENCE_SERIES not in ids:
        raise DriftOpenSourceError(
            f"la serie de referencia {REFERENCE_SERIES!r} debe estar en series_ids para fijar la "
            f"ventana común: se pidió {ids!r}"
        )

    store = Store(data_root)
    frames = {series_id: load_sessions(store, series_id=series_id) for series_id in ids}
    cutoff = clean_sample_cutoff(frames[REFERENCE_SERIES])
    if cutoff is None:
        raise DriftOpenSourceError(
            f"la serie de referencia {REFERENCE_SERIES!r} no tiene una era limpia (algún año "
            "posterior conserva el `open` repetido por encima de la tolerancia): no hay ventana "
            "común sobre la que decidir"
        )

    sources = [
        _source_block(frame=frames[series_id], series_id=series_id, cutoff=cutoff, now=moment)
        for series_id in ids
    ]
    by_id: dict[str, object] = dict(zip(ids, sources, strict=True))
    payload: dict[str, object] = {
        "analysis": "drift_open_source",
        "task": "#52",
        "title": "Fuente de apertura del estudio del drift",
        "generated_at": moment.isoformat(),
        "as_of": moment.isoformat(),
        "reference_series": REFERENCE_SERIES,
        "window_from": cutoff.isoformat(),
        "bp_to_pct": BP_TO_PCT,
        "sources": sources,
        "decision": _decision_block(REFERENCE_SERIES, by_id),
        "limitations": list(LIMITATIONS),
        "notes": list(NOTES),
    }
    report = Report(
        as_of=moment,
        report_date=moment.date().isoformat(),
        payload=payload,
        report_sha256=_digest(payload),
        reports_dir=reports_dir,
    )
    if reports_dir is not None:
        json_path, markdown_path = report.write(reports_dir)
        logger.info("fuente de apertura del drift: {} y {}", json_path, markdown_path)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Informe en prosa
# ─────────────────────────────────────────────────────────────────────────────
def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Tabla Markdown determinista."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_markdown(report: Report) -> str:
    """El informe en Markdown: la decisión, las tres fuentes y sus límites (criterios 5-8)."""
    payload = report.payload
    decision = _mapping(payload["decision"])
    sources = _sequence(payload["sources"])
    window_from = str(payload["window_from"])
    reference = str(payload["reference_series"])

    lines = [
        "# Fuente de apertura del estudio del drift (#52)",
        "",
        f"- **Instante de referencia:** `{payload['as_of']}`",
        f"- **Serie de apertura del estudio:** `{reference}`",
        f"- **Ventana común (era limpia):** desde `{window_from}`",
        f"- **`report_sha256`:** `{report.report_sha256}`",
        "",
        "## Decisión",
        "",
        f"- **Fuente:** `{decision['source']}`",
        f"- **¿Corregida?:** `{decision['corrected']}`",
        f"- **Razón:** {decision['rationale']}",
        "",
        "## Las tres fuentes sobre la misma ventana",
        "",
    ]
    summary: list[list[str]] = []
    for source in sources:
        block = _mapping(source)
        window = _window_of(block)
        summary.append(
            [
                f"`{block['series_id']}`",
                str(block["full_sessions"]),
                f"{_float_of(block['stale_open_share']):.2%}",
                str(block["open_quality"]),
                str(window["sessions"]),
                f"{_float_of(window['stale_open_share']):.2%}",
                f"`{window['verdict']}`",
                f"`{window['phase0_gate']}`",
            ]
        )
    lines.extend(
        _table(
            [
                "serie",
                "sesiones (muestra)",
                "`open` repetido (muestra)",
                "calidad",
                "sesiones (ventana)",
                "`open` repetido (ventana)",
                "veredicto (ventana)",
                "puerta Fase 0",
            ],
            summary,
        )
    )
    lines.append("")

    for source in sources:
        block = _mapping(source)
        window = _window_of(block)
        lines.extend([f"## `{block['series_id']}` — ventana limpia", ""])
        rows = [
            [
                f"`{segment['name']}`",
                str(segment["sessions"]),
                f"{_float_of(segment['mean_bp']):+.2f}",
                f"{_float_of(segment['t_stat']):+.2f}",
                f"{_float_of(segment['p_value']):.4f}",
                f"{_float_of(segment['hit_rate']):.1%}",
            ]
            for segment in (_mapping(item) for item in _sequence(window["segments"]))
        ]
        lines.extend(_table(["tramo", "sesiones", "media (pb)", "t", "p", "aciertos"], rows))
        difference = _mapping(window["difference"])
        lines.extend(
            [
                "",
                "**Diferencia emparejada `intraday - overnight`:** "
                f"{_float_of(difference['mean_difference_bp']):+.2f} pb "
                f"(t = {_float_of(difference['t_stat']):+.2f}, "
                f"p = {_float_of(difference['p_value']):.4f}).",
                "",
                f"Veredicto de la ventana: `{window['verdict']}` "
                f"(puerta de Fase 0: `{window['phase0_gate']}`); "
                "muestra completa: "
                f"`{_mapping(window['all_sessions'])['verdict']}` / "
                f"`{_mapping(window['all_sessions'])['phase0_gate']}` "
                f"({_mapping(window['all_sessions'])['sessions']} sesiones).",
                "",
            ]
        )

    lines.extend(["## Limitaciones (declaradas, no escondidas)", ""])
    lines.extend(f"- {item}" for item in _sequence(payload["limitations"]))
    lines.extend(["", "## Notas", ""])
    lines.extend(f"- {item}" for item in _sequence(payload["notes"]))
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del estudio de la fuente de apertura.

    Códigos de salida: ``0`` = artefacto escrito; ``1`` = configuración inválida;
    ``2`` = no hay datos o ventana común suficiente (no se escribe nada).
    """
    parser = argparse.ArgumentParser(
        prog="cfdtrader.analysis.drift_open_source",
        description="Compara la fuente de apertura del drift (^GSPC, SPY, ES=F) sobre una misma "
        "era limpia (#52)",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="raíz del almacén")
    parser.add_argument("--settings", type=Path, default=None, help="ruta de settings.yaml")
    parser.add_argument("--now", default=None, help="instante de referencia ISO-8601 (tests)")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="directorio de informes (por defecto <data-root>/derived/reports)",
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.settings)
    except ConfigurationError as error:
        logger.error("configuración inválida: {}", error)
        return 1

    data_root = args.data_root if args.data_root is not None else settings.data.root
    reports_dir = (
        Path(args.reports_dir)
        if args.reports_dir is not None
        else data_root / "derived" / "reports"
    )
    try:
        report = analyse(
            data_root=data_root,
            now=_parse_now(cast("str | None", args.now)),
            reports_dir=reports_dir,
        )
    except (DriftOpenSourceError, ConfigurationError) as error:
        logger.error("no se puede medir la fuente de apertura: {}", error)
        return 2

    decision = _mapping(report.payload["decision"])
    logger.info(
        "fuente de apertura: {} (corregida: {}); ventana desde {}; report_sha256 = {}",
        decision["source"],
        decision["corrected"],
        report.payload["window_from"],
        report.report_sha256,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
