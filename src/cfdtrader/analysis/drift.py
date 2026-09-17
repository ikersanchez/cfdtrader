"""Estudio de la tasa base y descomposición del drift (`plan.md` §1.1.a y §8.5) — tarea #6.

**Es la medición decisiva del proyecto.** La pregunta no es «¿sube el S&P 500?»,
sino **dónde** sube: si el retorno se concentra en el tramo nocturno
(``close→open``) y no en la sesión (``open→close``), una estrategia intradía pura
está operando sistemáticamente **la peor parte del día**, y hay que rehacerla o
abandonarla (es la primera puerta de salida de la Fase 0, `tasks.md` tarea 9).

Tres tramos, sobre las mismas sesiones:

=============  ==========================  ======================================
Tramo          Retorno                     Qué mide
=============  ==========================  ======================================
``intraday``   ``close_t / open_t - 1``    La sesión. Es lo único que captura el
                                           sistema (entrada en la subasta, cierre
                                           obligatorio a las 16:00 ET).
``overnight``  ``open_t / close_{t-1} - 1``  El hueco nocturno, que el sistema
                                           **no** captura: está plano cada noche.
``total``      ``close_t / close_{t-1} - 1``  ``close→close``: la referencia.
=============  ==========================  ======================================

Se calcula también la **tasa base** (% de sesiones alcistas y distribución de
``|open→close|``) y se segmenta por **día de la semana**, por **año** y por
**régimen de volatilidad** (terciles de volatilidad realizada de 20 sesiones,
medida *antes* de la sesión: nunca con datos del futuro).

Limitación que el informe declara en vez de esconder: esto se mide sobre
**``^GSPC``**, que es el subyacente, no el ``SPX500:CFD``. El CFD lo replica con
diferencial y financiación, así que **el signo de la conclusión se traslada, la
magnitud no**. No hay fuente del CFD (issue #50) y sustituirlo en silencio por
el índice es exactamente lo que la tarea #3 prohíbe.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, cast

import duckdb
import numpy as np
import polars as pl
from loguru import logger
from scipy import stats  # pyright: ignore

from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.store import Store, UnknownDatasetError

__all__ = [
    "DRIFT_SEGMENTS",
    "DriftSegment",
    "DriftStudy",
    "DriftVerdict",
    "Segment",
    "analyse",
    "decompose",
    "load_sessions",
    "main",
    "render_markdown",
]

#: Ventana de volatilidad realizada para clasificar el régimen, en sesiones.
VOL_WINDOW: Final[int] = 20

#: Proporción de sesiones con el `open` repetido que se tolera en un año para
#: considerar su tramo nocturno fiable.
STALE_OPEN_TOLERANCE: Final[float] = 0.05

#: Sesiones mínimas de la muestra limpia para que el veredicto se base en ella.
MIN_CLEAN_SESSIONS: Final[int] = 250

#: Niveles de significación que se declaran en el informe.
ALPHA: Final[float] = 0.05

DRIFT_SEGMENTS: Final[tuple[str, ...]] = ("intraday", "overnight", "total")


class DriftSegment(StrEnum):
    """Los tres tramos que hay que comparar."""

    INTRADAY = "intraday"
    """``open→close``: lo que captura el sistema."""

    OVERNIGHT = "overnight"
    """``close→open``: lo que el sistema se pierde por estar plano cada noche."""

    TOTAL = "total"
    """``close→close``: la referencia."""


class DriftVerdict(StrEnum):
    """Veredicto de la descomposición, con la puerta de salida de la Fase 0."""

    INTRADAY = "intraday"
    """El drift está en la sesión: el planteamiento intradía tiene sentido."""

    OVERNIGHT = "overnight"
    """El drift está fuera de la sesión: la estrategia intradía opera la peor parte."""

    MIXED = "mixed"
    """Los dos tramos son positivos y la diferencia no es concluyente."""

    NONE = "none"
    """Ningún tramo muestra drift positivo relevante."""

    @property
    def phase0_gate(self) -> str:
        """Puerta de salida de la Fase 0 (`tasks.md`, tarea 9)."""
        if self is DriftVerdict.OVERNIGHT:
            return "fail"
        if self is DriftVerdict.INTRADAY:
            return "pass"
        return "inconclusive"


@dataclass(frozen=True, slots=True)
class Segment:
    """Estadísticos de un tramo."""

    name: str
    sessions: int
    mean_bp: float
    median_bp: float
    std_bp: float
    t_stat: float
    p_value: float
    hit_rate: float
    cumulative_bp: float

    @property
    def significant(self) -> bool:
        """``True`` si la media es distinguible de cero al 95 %."""
        return abs(self.t_stat) > 0 and self.p_value < ALPHA


@dataclass(frozen=True, slots=True)
class BaseRate:
    """Tasa base de la sesión y distribución de ``|open→close|``."""

    sessions: int
    up_sessions: int
    up_share: float
    abs_move_median_bp: float
    abs_move_p90_bp: float
    abs_move_p99_bp: float


@dataclass(frozen=True, slots=True)
class PairedTest:
    """Contraste de la diferencia intraday - overnight, emparejada por sesión."""

    mean_difference_bp: float
    t_stat: float
    p_value: float

    @property
    def significant(self) -> bool:
        """``True`` si la diferencia es distinguible de cero al 95 %."""
        return self.p_value < ALPHA


@dataclass(frozen=True, slots=True)
class DriftStudy:
    """Resultado completo del estudio, listo para el informe."""

    series_id: str
    source: str
    first_session: str
    last_session: str
    sessions: int
    as_of: datetime
    segments: tuple[Segment, ...]
    base_rate: BaseRate
    difference: PairedTest
    verdict: DriftVerdict
    stale_open_share: float = 0.0
    open_quality: str = "ok"
    clean_from: str | None = None
    clean_sessions: int = 0
    clean_segments: tuple[Segment, ...] = ()
    clean_difference: PairedTest | None = None
    by_year: tuple[dict[str, object], ...] = ()
    by_weekday: tuple[dict[str, object], ...] = ()
    by_volatility: tuple[dict[str, object], ...] = ()
    limitations: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default=())

    @property
    def phase0_gate(self) -> str:
        """Puerta de salida de la Fase 0 según el veredicto."""
        return self.verdict.phase0_gate

    def segment(self, name: str) -> Segment:
        """Estadísticos de ese tramo en la muestra completa."""
        return _find_segment(self.segments, name)

    def clean_segment(self, name: str) -> Segment:
        """Estadísticos de ese tramo en la muestra limpia (sin el artefacto del `open`)."""
        return _find_segment(self.clean_segments, name)


def _find_segment(segments: tuple[Segment, ...], name: str) -> Segment:
    """El tramo con ese nombre."""
    for segment in segments:
        if segment.name == name:
            return segment
    raise KeyError(name)


# ─────────────────────────────────────────────────────────────────────────────
# Datos
# ─────────────────────────────────────────────────────────────────────────────
def load_sessions(store: Store, *, series_id: str) -> pl.DataFrame:
    """Sesiones de esa serie con los tres tramos calculados, en fracción.

    El retorno nocturno de una sesión se calcula con el cierre de la sesión
    **anterior**: la primera sesión de la muestra no tiene tramo nocturno y se
    descarta del cálculo (no se rellena con nada).
    """
    query = (
        "SELECT as_of, open, close FROM raw.market_daily "  # noqa: S608
        f"WHERE series_id = {_literal(series_id)} AND open IS NOT NULL AND close IS NOT NULL "
        "ORDER BY as_of"
    )
    try:
        frame = store.sql(query).sort("as_of")
    except (UnknownDatasetError, duckdb.Error) as error:
        raise ConfigurationError(
            f"no hay dataset de mercado diario en {store.root}: {error}. "
            "Ejecuta antes la ingesta de mercado (tarea #3)."
        ) from error
    if frame.height < 2:
        raise ConfigurationError(
            f"no hay sesiones suficientes de {series_id!r} en el almacén: "
            f"{frame.height}. Ejecuta antes la ingesta de mercado (tarea #3)."
        )

    frame = frame.with_columns(
        pl.col("close").shift(1).alias("prev_close"),
        pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("session"),
    ).filter(pl.col("prev_close").is_not_null())

    frame = frame.with_columns(
        (pl.col("close") / pl.col("open") - 1.0).alias(DriftSegment.INTRADAY.value),
        (pl.col("open") / pl.col("prev_close") - 1.0).alias(DriftSegment.OVERNIGHT.value),
        (pl.col("close") / pl.col("prev_close") - 1.0).alias(DriftSegment.TOTAL.value),
        # Artefacto de la fuente: en el histórico antiguo de Yahoo el `open` diario
        # del índice es el **cierre anterior repetido**, y eso hace que el tramo
        # nocturno de esa sesión sea 0 por construcción y que la sesión se quede
        # con todo el retorno. Hay que medirlo antes de creerse el veredicto.
        (pl.col("open") == pl.col("prev_close")).alias("open_stale"),
    )
    # Volatilidad realizada de las 20 sesiones ANTERIORES: régimen ex-ante, sin
    # mirar la sesión que se está clasificando.
    return frame.with_columns(
        pl.col(DriftSegment.TOTAL.value)
        .log1p()
        .rolling_std(window_size=VOL_WINDOW, min_samples=VOL_WINDOW)
        .shift(1)
        .alias("vol20")
    )


def _literal(value: str) -> str:
    """Literal SQL seguro: los identificadores vienen del registro validado."""
    return "'" + value.replace("'", "''") + "'"


# ─────────────────────────────────────────────────────────────────────────────
# Estadísticos
# ─────────────────────────────────────────────────────────────────────────────
def decompose(frame: pl.DataFrame, *, series_id: str, source: str, as_of: datetime) -> DriftStudy:
    """Calcula la descomposición completa y emite el veredicto.

    El veredicto se calcula sobre la **muestra limpia** (sin el artefacto del
    `open` repetido) cuando existe: decidir la puerta de la Fase 0 con un tramo
    nocturno que es cero por construcción sería decidir con el dato en contra.
    La muestra completa se reporta igualmente, para que se vea la diferencia.
    """
    segments = tuple(_segment(frame, name) for name in DRIFT_SEGMENTS)
    base_rate = _base_rate(frame)
    difference = _paired_difference(frame)
    by_year = tuple(_by_year(frame))

    stale_share = _float_of(frame.get_column("open_stale").mean())
    clean_from = _clean_from(by_year)
    clean = _clean_frame(frame, clean_from)
    clean_segments = (
        tuple(_segment(clean, name) for name in DRIFT_SEGMENTS) if clean is not None else ()
    )
    clean_difference = _paired_difference(clean) if clean is not None else None

    if clean_segments and clean_difference is not None:
        verdict = _verdict(clean_segments, clean_difference)
    else:
        # Sin tramo nocturno fiable no se puede decidir: `inconclusive`, nunca un
        # aprobado ni un suspenso construidos sobre un artefacto de la fuente.
        verdict = (
            DriftVerdict.NONE
            if segments[0].mean_bp <= 0.0 and segments[1].mean_bp <= 0.0
            else DriftVerdict.MIXED
        )

    return DriftStudy(
        series_id=series_id,
        source=source,
        first_session=str(frame.get_column("session").min()),
        last_session=str(frame.get_column("session").max()),
        sessions=frame.height,
        as_of=as_of,
        segments=segments,
        base_rate=base_rate,
        difference=difference,
        verdict=verdict,
        stale_open_share=stale_share,
        open_quality=_open_quality(stale_share),
        clean_from=None if clean_from is None else clean_from.isoformat(),
        clean_sessions=0 if clean is None else clean.height,
        clean_segments=clean_segments,
        clean_difference=clean_difference,
        by_year=by_year,
        by_weekday=tuple(_by_weekday(frame)),
        by_volatility=tuple(_by_volatility(frame)),
        limitations=(
            "Medido sobre el índice (^GSPC), que es el subyacente del CFD y no el CFD: "
            "el signo de la conclusión se traslada, la magnitud no (diferencial y financiación).",
            "No existe fuente del intradía ni del bid/ask del SPX500:CFD (issue #50); "
            "la tarea #3 prohíbe sustituirlo en silencio por ^GSPC o ES=F.",
            "El `open` diario del índice en Yahoo es el **cierre anterior repetido** en parte "
            "del histórico (96 % de las sesiones de 2005, 0 % desde 2016): en esas sesiones el "
            "tramo nocturno es cero por construcción. Por eso el veredicto se calcula sobre la "
            "muestra limpia y la contaminación se declara año a año. Seguimiento: #52.",
            "Los tramos usan los precios de subasta de apertura y cierre del índice. "
            "El precio de entrada real del CFD (subasta frente a unos minutos después) "
            "lo cierra la tarea #8 con el slippage medido.",
            "El retorno nocturno del sistema es CERO por construcción: opera intradía puro, "
            "sin overnight. Esta descomposición mide dónde está el retorno del índice, "
            "no lo que el sistema capturaría.",
        ),
        notes=(
            "El contraste intraday - overnight está emparejado por sesión: se mide en las "
            "mismas fechas.",
            f"El régimen de volatilidad son terciles de la volatilidad realizada de {VOL_WINDOW} "
            "sesiones anteriores, calculada sin datos del futuro.",
            "El informe separa por año y por régimen para no confundir una década con un régimen.",
            "Un artefacto de fuente detectado (el `open` repetido) se declara y se excluye de la "
            "decisión; no se esconde ni se deja sesgar el veredicto.",
        ),
    )


def _clean_from(by_year: tuple[dict[str, object], ...]) -> date | None:
    """Primer 1 de enero desde el que **ningún** año posterior tiene `open` repetido.

    Se calcula con el dato, no con una fecha a mano: si mañana la fuente arregla el
    histórico antiguo, la muestra limpia se amplía sola.
    """
    years = sorted(int(str(row["year"])) for row in by_year)
    stale = {int(str(row["year"])): _float_of(row["stale_open_share"]) for row in by_year}
    for year in years:
        if all(stale[later] <= STALE_OPEN_TOLERANCE for later in years if later >= year):
            return date(year, 1, 1)
    return None


def _float_of(value: object) -> float:
    """Número de un escalar de polars, o 0.0 si no lo es."""
    return float(value) if isinstance(value, (int, float)) else 0.0


def _clean_frame(frame: pl.DataFrame, clean_from: date | None) -> pl.DataFrame | None:
    """Muestra sin el artefacto del `open` repetido, si alcanza para decidir."""
    if clean_from is None:
        return None
    clean = frame.filter(pl.col("session") >= clean_from).filter(~pl.col("open_stale"))
    return clean if clean.height >= MIN_CLEAN_SESSIONS else None


def _open_quality(stale_share: float) -> str:
    """Calidad del `open` de la muestra: `ok`, `degraded` o `unusable`."""
    if stale_share <= STALE_OPEN_TOLERANCE:
        return "ok"
    if stale_share <= 0.25:
        return "degraded"
    return "unusable"


def _segment(frame: pl.DataFrame, name: str) -> Segment:
    """Estadísticos de un tramo, en puntos básicos."""
    fractions = _fractions(frame, name)
    values = fractions * 10_000.0
    t_stat, p_value = _t_test(values)
    return Segment(
        name=name,
        sessions=int(values.size),
        mean_bp=float(np.mean(values)),
        median_bp=float(np.median(values)),
        std_bp=float(np.std(values, ddof=1)),
        t_stat=t_stat,
        p_value=p_value,
        hit_rate=float(np.mean(values > 0)),
        cumulative_bp=_cumulative_bp(fractions),
    )


def _fractions(frame: pl.DataFrame, name: str) -> np.ndarray:
    """Columna de un tramo como array de ``numpy``, en fracción (0,01 = +1 %)."""
    return np.asarray(frame.get_column(name).to_numpy(), dtype=float)


def _values(frame: pl.DataFrame, name: str) -> np.ndarray:
    """Columna de un tramo en puntos básicos, que es como se reporta."""
    return _fractions(frame, name) * 10_000.0


def _cumulative_bp(fractions: np.ndarray) -> float:
    """Retorno acumulado del tramo, en puntos básicos.

    Se compone en **fracciones**: componer puntos básicos como si fueran
    fracciones da infinito en cuanto hay unas cuantas sesiones.
    """
    cumulative = 1.0
    for value in fractions:
        cumulative *= 1.0 + float(value)
    return (cumulative - 1.0) * 10_000.0


def _t_test(values: np.ndarray, *, other: np.ndarray | None = None) -> tuple[float, float]:
    """Estadístico ``t`` y valor ``p`` de un contraste de medias.

    ``scipy`` no publica anotaciones completas para los objetos que devuelven sus
    contrastes, así que el límite de la librería se aísla aquí: hacia fuera solo
    salen dos ``float``.
    """
    # `cast("Any", …)` es deliberado: los objetos que devuelve scipy no están
    # anotados, y el contrato con el resto del módulo son dos `float`.
    result: Any = cast(
        "Any",
        stats.ttest_1samp(values, 0.0)  # pyright: ignore[reportUnknownMemberType]
        if other is None
        else stats.ttest_rel(values, other),  # pyright: ignore[reportUnknownMemberType]
    )
    return _as_float(result.statistic), _as_float(result.pvalue)


def _as_float(value: object) -> float:
    """Número de un resultado de scipy (``numpy.float64`` incluido)."""
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"se esperaba un número de scipy, no {type(value).__name__}")


def _base_rate(frame: pl.DataFrame) -> BaseRate:
    """% de sesiones alcistas y distribución del movimiento absoluto de la sesión."""
    intraday = _fractions(frame, DriftSegment.INTRADAY.value)
    absolute = np.abs(intraday) * 10_000.0
    return BaseRate(
        sessions=int(intraday.size),
        up_sessions=int(np.sum(intraday > 0)),
        up_share=float(np.mean(intraday > 0)),
        abs_move_median_bp=float(np.median(absolute)),
        abs_move_p90_bp=float(np.percentile(absolute, 90)),
        abs_move_p99_bp=float(np.percentile(absolute, 99)),
    )


def _paired_difference(frame: pl.DataFrame) -> PairedTest:
    """Contraste emparejado ``intraday - overnight``."""
    intraday = _values(frame, DriftSegment.INTRADAY.value)
    overnight = _values(frame, DriftSegment.OVERNIGHT.value)
    t_stat, p_value = _t_test(intraday, other=overnight)
    return PairedTest(
        mean_difference_bp=float(np.mean(intraday - overnight)),
        t_stat=t_stat,
        p_value=p_value,
    )


def _verdict(segments: tuple[Segment, ...], difference: PairedTest) -> DriftVerdict:
    """Regla del veredicto, escrita explícitamente para que sea auditable.

    Es la regla **pre-registrada** de `plan.md` §1.1.a y `tasks.md` tarea 9
    («si el drift se concentra en `close→open` en vez de en `open→close`, no se
    pasa a Fase 1»), traducida a criterios estadísticos:

    1. Si la sesión no aporta nada y el tramo nocturno sí, el veredicto es
       ``overnight``: se opera sistemáticamente la peor parte del día.
    2. Si la sesión **no demuestra** drift propio (su media no es distinguible de
       cero al 95 %) mientras el tramo nocturno sí lo demuestra, el veredicto es
       ``overnight``: «drift nulo en la sesión con drift positivo fuera de ella»
       es exactamente la condición de abandono. No hace falta que la media de la
       sesión sea negativa.
    3. Si la sesión aporta de forma significativa y por encima del tramo
       nocturno, el veredicto es ``intraday``.
    4. Si ninguno aporta, ``none``. En cualquier otro caso, ``mixed``.
    """
    by_name = {segment.name: segment for segment in segments}
    intraday = by_name[DriftSegment.INTRADAY.value]
    overnight = by_name[DriftSegment.OVERNIGHT.value]

    if intraday.mean_bp <= 0.0 and overnight.mean_bp > 0.0:
        return DriftVerdict.OVERNIGHT
    if overnight.mean_bp > 0.0 and overnight.significant and not intraday.significant:
        return DriftVerdict.OVERNIGHT
    if (
        intraday.significant
        and intraday.mean_bp > 0.0
        and difference.significant
        and difference.mean_difference_bp > 0.0
    ):
        return DriftVerdict.INTRADAY
    if intraday.mean_bp <= 0.0 and overnight.mean_bp <= 0.0:
        return DriftVerdict.NONE
    return DriftVerdict.MIXED


def _grouped(
    frame: pl.DataFrame, key: str, *, order: list[str] | None = None
) -> list[dict[str, object]]:
    """Estadísticos de los tres tramos agrupados por esa clave."""
    grouped = frame.group_by(key).agg(
        pl.len().alias("sessions"),
        (pl.col(DriftSegment.INTRADAY.value).mean() * 10_000.0).alias("intraday_mean_bp"),
        (pl.col(DriftSegment.INTRADAY.value).median() * 10_000.0).alias("intraday_median_bp"),
        (pl.col(DriftSegment.OVERNIGHT.value).mean() * 10_000.0).alias("overnight_mean_bp"),
        (pl.col(DriftSegment.OVERNIGHT.value).median() * 10_000.0).alias("overnight_median_bp"),
        (pl.col(DriftSegment.TOTAL.value).mean() * 10_000.0).alias("total_mean_bp"),
        (pl.col(DriftSegment.INTRADAY.value) > 0).mean().alias("intraday_hit_rate"),
        pl.col("open_stale").mean().alias("stale_open_share"),
    )
    rows = [
        {str(name): _scalar(value) for name, value in row.items()}
        for row in grouped.iter_rows(named=True)
    ]
    if order is None:
        return sorted(rows, key=lambda row: str(row[key]))
    positions = {name: index for index, name in enumerate(order)}
    return sorted(rows, key=lambda row: positions.get(str(row[key]), len(positions)))


def _by_year(frame: pl.DataFrame) -> list[dict[str, object]]:
    """Por año: un año suelto no es una conclusión, pero una década sí dice algo."""
    return _grouped(frame.with_columns(pl.col("session").dt.year().alias("year")), "year")


def _by_weekday(frame: pl.DataFrame) -> list[dict[str, object]]:
    """Por día de la semana, con el nombre en claro."""
    names = ["lunes", "martes", "miércoles", "jueves", "viernes"]
    frame = frame.with_columns(
        pl.col("session")
        .dt.weekday()
        .replace_strict(
            {
                1: "lunes",
                2: "martes",
                3: "miércoles",
                4: "jueves",
                5: "viernes",
                6: "sábado",
                7: "domingo",
            }
        )
        .alias("weekday")
    )
    return _grouped(frame, "weekday", order=names)


def _by_volatility(frame: pl.DataFrame) -> list[dict[str, object]]:
    """Por régimen de volatilidad: terciles de la volatilidad realizada de 20 sesiones."""
    with_vol = frame.filter(pl.col("vol20").is_not_null())
    if with_vol.height < 3:
        return []
    low = _quantile(with_vol, 1 / 3)
    high = _quantile(with_vol, 2 / 3)
    labelled = with_vol.with_columns(
        pl.when(pl.col("vol20") <= low)
        .then(pl.lit("low"))
        .when(pl.col("vol20") <= high)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("high"))
        .alias("vol_regime")
    )
    return _grouped(labelled, "vol_regime", order=["low", "medium", "high"])


def _quantile(frame: pl.DataFrame, value: float) -> float:
    """Cuantil de la volatilidad realizada."""
    quantile = frame.get_column("vol20").quantile(value)
    return float(quantile) if quantile is not None else 0.0


def _scalar(value: object) -> object:
    """Escalar serializable en JSON: los tipos de polars se convierten a los de Python."""
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes)):
        return item()
    return value


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución e informe
# ─────────────────────────────────────────────────────────────────────────────
def analyse(
    *,
    data_root: Path,
    now: datetime,
    series_id: str = "^GSPC",
    source: str = "yfinance",
    reports_dir: Path | None = None,
) -> DriftStudy:
    """Lee el diario del almacén, calcula el estudio y escribe el informe."""
    store = Store(data_root)
    frame = load_sessions(store, series_id=series_id)
    study = decompose(frame, series_id=series_id, source=source, as_of=now)
    if reports_dir is not None:
        json_path, markdown_path = write_report(study, reports_dir)
        logger.info("informe del drift: {} y {}", json_path, markdown_path)
    return study


def write_report(study: DriftStudy, directory: Path) -> tuple[Path, Path]:
    """Escribe el informe del estudio en JSON y Markdown."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"drift_decomposition_{study.as_of.date().isoformat()}"
    json_path = directory / f"{stem}.json"
    markdown_path = directory / f"{stem}.md"
    payload = asdict(study)
    payload["verdict"] = str(study.verdict)
    payload["phase0_gate"] = study.phase0_gate
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(study), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(study: DriftStudy) -> str:
    """Informe legible, empezando por el veredicto (que es lo que se decide)."""
    base = study.base_rate
    verdict_line = {
        DriftVerdict.INTRADAY: "**El drift está en la sesión** (`open→close`).",
        DriftVerdict.OVERNIGHT: (
            "**El drift está en el tramo nocturno** (`close→open`): la estrategia intradía "
            "opera sistemáticamente la peor parte del día."
        ),
        DriftVerdict.MIXED: "Los dos tramos aportan; la diferencia no es concluyente.",
        DriftVerdict.NONE: "Ningún tramo muestra un drift positivo relevante.",
    }[study.verdict]

    lines = [
        "# Descomposición del drift — estudio de la tasa base",
        "",
        f"- **Serie:** `{study.series_id}` (fuente `{study.source}`)",
        f"- **Muestras:** {study.sessions} sesiones, de {study.first_session} "
        f"a {study.last_session}",
        f"- **Calculado:** {study.as_of.isoformat()}",
        f"- **Veredicto:** `{study.verdict}` — {verdict_line}",
        f"- **Puerta de la Fase 0 (drift):** `{study.phase0_gate}`",
        f"- **Calidad del `open`:** `{study.open_quality}` "
        f"({study.stale_open_share:.1%} de las sesiones con el `open` repetido del cierre "
        "anterior)",
        "",
        "## Los tres tramos (muestra completa)",
        "",
        "| tramo | sesiones | media (pb) | mediana (pb) | desv. (pb) | "
        "t | p | aciertos | acumulado (pb) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for segment in study.segments:
        lines.append(
            f"| `{segment.name}` | {segment.sessions} | {segment.mean_bp:+.2f} | "
            f"{segment.median_bp:+.2f} | {segment.std_bp:.2f} | {segment.t_stat:+.2f} | "
            f"{segment.p_value:.4f} | {segment.hit_rate:.1%} | {segment.cumulative_bp:+.0f} |"
        )
    lines.extend(
        [
            "",
            "**Diferencia emparejada `intraday - overnight`:** "
            f"{study.difference.mean_difference_bp:+.2f} pb "
            f"(t = {study.difference.t_stat:+.2f}, p = {study.difference.p_value:.4f}).",
            "",
        ]
    )

    lines.extend(_clean_section(study))

    lines.extend(
        [
            "## Tasa base de la sesión",
            "",
            f"- Sesiones alcistas: **{base.up_share:.1%}** "
            f"({base.up_sessions} de {base.sessions}).",
            f"- `|open→close|` mediano: **{base.abs_move_median_bp:.1f} pb**; "
            f"p90: {base.abs_move_p90_bp:.1f} pb; p99: {base.abs_move_p99_bp:.1f} pb.",
            "",
        ]
    )
    lines.extend(_table("Por año", study.by_year))
    lines.extend(_table("Por día de la semana", study.by_weekday))
    lines.extend(_table("Por régimen de volatilidad", study.by_volatility))

    lines.extend(["## Limitaciones (declaradas, no escondidas)", ""])
    lines.extend(f"- {item}" for item in study.limitations)
    lines.extend(["", "## Notas", ""])
    lines.extend(f"- {item}" for item in study.notes)
    lines.append("")
    return "\n".join(lines)


def _clean_section(study: DriftStudy) -> list[str]:
    """Sección de la muestra limpia: es sobre la que se decide."""
    lines = ["## Los tres tramos (muestra limpia: sin el `open` repetido)", ""]
    if not study.clean_segments:
        lines.extend(
            [
                "_No hay tramo nocturno fiable en toda la muestra: el veredicto **no se puede** "
                "decidir con este dato y queda como `inconclusive`._",
                "",
            ]
        )
        return lines

    lines.extend(
        [
            f"Desde **{study.clean_from}**, con {study.clean_sessions} sesiones utilizables "
            "(se excluyen las sesiones en las que el `open` es el cierre anterior).",
            "",
            "| tramo | sesiones | media (pb) | mediana (pb) | desv. (pb) | "
            "t | p | aciertos | acumulado (pb) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for segment in study.clean_segments:
        lines.append(
            f"| `{segment.name}` | {segment.sessions} | {segment.mean_bp:+.2f} | "
            f"{segment.median_bp:+.2f} | {segment.std_bp:.2f} | {segment.t_stat:+.2f} | "
            f"{segment.p_value:.4f} | {segment.hit_rate:.1%} | {segment.cumulative_bp:+.0f} |"
        )
    if study.clean_difference is not None:
        lines.extend(
            [
                "",
                "**Diferencia emparejada `intraday - overnight`:** "
                f"{study.clean_difference.mean_difference_bp:+.2f} pb "
                f"(t = {study.clean_difference.t_stat:+.2f}, "
                f"p = {study.clean_difference.p_value:.4f}).",
            ]
        )
    lines.append("")
    return lines


def _table(title: str, rows: tuple[dict[str, object], ...]) -> list[str]:
    """Tabla de un segmentado."""
    lines = [f"## {title}", ""]
    if not rows:
        lines.extend(["_Sin datos suficientes para segmentar._", ""])
        return lines
    keys = list(rows[0])
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "---|" * len(keys))
    for row in rows:
        cells: list[str] = []
        for key in keys:
            value = row[key]
            if isinstance(value, float) and key.endswith(("_share", "_rate")):
                cells.append(f"{value:.1%}")
            elif isinstance(value, float):
                cells.append(f"{value:+.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del estudio."""
    parser = argparse.ArgumentParser(prog="cfdtrader.analysis.drift", description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="raíz del almacén")
    parser.add_argument("--settings", type=Path, default=None, help="ruta de settings.yaml")
    parser.add_argument("--series", default="^GSPC", help="serie a analizar")
    parser.add_argument("--now", default=None, help="instante de referencia ISO (tests)")
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.settings)
    except ConfigurationError as error:
        logger.error("configuración inválida: {}", error)
        return 1

    data_root = args.data_root if args.data_root is not None else settings.data.root
    now = _parse_now(args.now)
    try:
        study = analyse(
            data_root=data_root,
            now=now,
            series_id=args.series,
            reports_dir=data_root / "derived" / "reports",
        )
    except ConfigurationError as error:
        logger.error("no se puede hacer el estudio: {}", error)
        return 2

    segment = study.segment(DriftSegment.INTRADAY.value)
    logger.info(
        "drift: intraday {:+.2f} pb vs overnight {:+.2f} pb ⇒ {} (puerta Fase 0: {})",
        segment.mean_bp,
        study.segment(DriftSegment.OVERNIGHT.value).mean_bp,
        study.verdict.value,
        study.phase0_gate,
    )
    return 0


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
