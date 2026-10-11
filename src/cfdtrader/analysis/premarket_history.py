"""Serie historica point-in-time del pre-mercado del ES: existe o no existe (tarea #149).

La pregunta que responde este modulo, con **una medicion** en vez de una promesa: **hay alguna
fuente gratuita con la que reconstruir el gap de pre-mercado del ES hacia atras?** De la respuesta
depende (a) si el gap puede **re-medirse** en la Fase 2 y promoverse a **feature**, o (b) si la
decision de #107 (`plan.md` §19.18: el proxy declarado es la ruta definitiva) queda **confirmada**
para este dato.

**Lo que ya se sabe, y por que la pregunta tiene sentido.** #141 mide el gap **del dia** con la
ventana de 5 minutos con horas extendidas de la fuente gratuita, que es **rodante (~60 dias)**; y
la barra **diaria** del ES es una ventana de 24 h que termina a las 16:00 ET, asi que el movimiento
de pre-mercado vive entero en la barra `t`, que a las 08:45 **aun no existe** (*look-ahead*). El
2026-10-09 el modelo decidio `SHORT` con el ES en verde y **no podia verlo**: #148 lo cablea como
**dato declarado** de la decision mientras esto se mide.

**El criterio, declarado antes de medir.** Un candidato sirve si cumple **las tres**:

1. **Point-in-time.** Para la sesion `t` existe una barra con instante ``<=`` 08:45 ET, y
   **ninguna** barra de la sesion que se decide.
2. **Frescura.** El dato no es mas viejo que ``MAX_STALENESS_MINUTES``: una barra horaria que cierra
   a las 09:00 no esta disponible a las 08:45 y no vale como sustituto sin decirlo.
3. **Ventana.** Cubre al menos las sesiones que el **conjunto de features** ya usa
   (``derived.labels``, medido en el almacen): una feature nueva que solo exista en una decima parte
   de la muestra no se puede promover sin truncar el entrenamiento.

El veredicto sale de esas tres cifras, no de una opinion: si **ningun** candidato cumple las tres,
el informe **confirma #107** y lo dice; si alguno las cumple, define la feature y abre seguimiento.

**Honestidad.** Esto **no** mide *edge* ni promueve nada: `gate`, `phase1_ready` y `phase2_ready`
**no** se mueven. Las fuentes de **pago** se enumeran con lo que **declara** el proveedor y se
marcan como **no verificadas** en esta medicion: comprarlas es una **decision del propietario**.
Un «no hay serie» medido vale mas que una feature prometida.

Sin red en el artefacto: las barras se capturan una vez (``--capture-dir``) y el informe se
reproduce **sin red** (``--bars-dir``), reutilizando `analysis/premarket_gap.py` (#141) — incluidos
su ``snapshot_instant`` y el formato de barras.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, Final, cast

import polars as pl

from cfdtrader.analysis import premarket_gap
from cfdtrader.data.calendar import EASTERN
from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.store import Store, UnknownDatasetError

__all__ = [
    "BOUNDED_WINDOW_DAYS",
    "CANDIDATES",
    "MAX_STALENESS_MINUTES",
    "MODULE",
    "PAID_CANDIDATES",
    "REPORT_PREFIX",
    "TASK",
    "CandidateResult",
    "CandidateSpec",
    "PaidCandidate",
    "PremarketHistoryError",
    "Redundancy",
    "analyse",
    "decide",
    "main",
    "measure_candidate",
    "render_markdown",
    "sessions_required",
    "write_report",
]

MODULE: Final[str] = "cfdtrader.analysis.premarket_history"
TASK: Final[str] = "#149"
REPORT_PREFIX: Final[str] = "premarket_history"

#: Instante del snapshot declarado (`plan.md` §13): 08:45 ET. Una barra posterior no existe aun.
#: Un dato mas viejo que esto **no** es el gap del snapshot: hay que declararlo.
MAX_STALENESS_MINUTES: Final[int] = 60

#: Por debajo de esta ventana el veredicto queda **acotado** (no es una conclusion general).
BOUNDED_WINDOW_DAYS: Final[int] = 365

#: La serie del futuro y el proxy declarado del indice (`plan.md` §3.1 y #107).
ES_SERIES: Final[str] = "ES=F"
INDEX_SERIES: Final[str] = "^GSPC"

#: Licencia de la fuente gratuita, copiada de `_docs/data_sources.md` (no reinterpretada).
FREE_LICENSE: Final[str] = (
    "yfinance (Apache-2.0); los datos de Yahoo son de uso **estrictamente personal** y no se "
    "redistribuyen; *scraper* no oficial detras de adaptador"
)


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """Una fuente **gratuita** candidata: se descarga y su ventana se **mide** (no se estima)."""

    key: str
    series: str
    interval: str
    period: str
    extended_hours: bool
    note: str
    stamp_at_window_start: bool = False
    """Si la barra viene sellada al **inicio** de su ventana en vez de a su cierre.

    La barra **diaria** del proveedor llega sellada a las 00:00 ET de **su propia** sesion aunque su
    valor sea el cierre de las 16:00: usar esa barra a las 08:45 seria *look-ahead*, asi que el dato
    **no** se cuenta como point-in-time. El resto de granularidades llevan el sello de su cierre.
    """


#: Los candidatos que se descargan de verdad: la fuente gratuita ya declarada, a cuatro
#: granularidades. El limite de cada una **no** se copia de la documentacion: se mide.
CANDIDATES: Final[tuple[CandidateSpec, ...]] = (
    CandidateSpec(
        key="es_5m",
        series=ES_SERIES,
        interval="5m",
        period="60d",
        extended_hours=True,
        note="la granularidad que #141 mide en vivo: es la unica que da el instante de las 08:45",
    ),
    CandidateSpec(
        key="es_30m",
        series=ES_SERIES,
        interval="30m",
        period="60d",
        extended_hours=True,
        note="misma ventana con barras mas gruesas: comprueba donde corta el proveedor",
    ),
    CandidateSpec(
        key="es_1h",
        series=ES_SERIES,
        interval="1h",
        period="730d",
        extended_hours=True,
        note="la ventana mas larga que el proveedor ofrece **con** horas extendidas",
    ),
    CandidateSpec(
        key="es_1d",
        series=ES_SERIES,
        interval="1d",
        period="max",
        extended_hours=False,
        note=(
            "la barra diaria es una ventana de 24 h sellada al inicio del dia: leerla a las 08:45 "
            "seria *look-ahead*, y su `open(t) ~= close(t-1)` la hace redundante"
        ),
        stamp_at_window_start=True,
    ),
)


@dataclass(frozen=True, slots=True)
class PaidCandidate:
    """Una fuente de **pago**: se enumera lo que **declara** el proveedor, sin verificarlo aqui."""

    vendor: str
    offering: str
    verified: bool
    statement: str


#: Las fuentes de pago **no** se consultan en esta medicion (exigen contratar acceso): se
#: enumeran con lo que declaran y se marcan como **no verificadas**. Comprarlas es del propietario.
PAID_CANDIDATES: Final[tuple[PaidCandidate, ...]] = (
    PaidCandidate(
        vendor="CME DataMine",
        offering="historico del futuro del CME (incluye la sesion nocturna)",
        verified=False,
        statement="ventana, granularidad y licencia **declaradas por el proveedor**: aqui no se "
        "verifican y el coste es una decision del propietario",
    ),
    PaidCandidate(
        vendor="Proveedores de datos de mercado por suscripcion",
        offering="intradia historico con horas extendidas y ajuste de contrato",
        verified=False,
        statement="cada proveedor declara su ventana y su licencia; **ninguna** se ha verificado "
        "aqui porque exige contratar acceso, y el coste es del propietario",
    ),
    PaidCandidate(
        vendor="Barras propias del broker",
        offering="el instrumento que de verdad se opera (el CFD), si el broker da historico",
        verified=False,
        statement="el CFD del proyecto sigue **sin fuente** (#107 §19.18): preguntar al broker es "
        "una accion del propietario, no una tarea de ingenieria",
    ),
)


class PremarketHistoryError(Exception):
    """Argumentos invalidos o almacen ilegible."""


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Lo **medido** de un candidato: su ventana, su dato point-in-time y su frescura."""

    spec: CandidateSpec
    state: str
    reason: str
    bars: int
    window_start: datetime | None = None
    window_end: datetime | None = None
    extended_hours_seen: bool = False
    sessions_with_datum: int = 0
    sessions_fresh: int = 0
    staleness_minutes_max: float | None = None
    example: dict[str, Any] | None = None

    @property
    def window_days(self) -> int:
        """Dias de calendario que cubre la ventana capturada (0 si no hay barras)."""
        if self.window_start is None or self.window_end is None:
            return 0
        return (self.window_end.date() - self.window_start.date()).days

    @property
    def bounded(self) -> bool:
        """Si la ventana es tan corta que el veredicto queda **acotado**."""
        return self.window_days < BOUNDED_WINDOW_DAYS


def _as_utc(stamp: object) -> datetime:
    """Un sello del almacen a ``datetime`` UTC (el almacen guarda en UTC)."""
    if not isinstance(stamp, datetime):
        raise PremarketHistoryError(f"se esperaba un instante del almacen, llego {stamp!r}")
    return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)


def _daily(store: Store) -> pl.DataFrame:
    """``raw.market_daily`` con las columnas que se leen (se filtra en memoria, nunca por SQL)."""
    try:
        return store.sql("SELECT series_id, as_of, open, close FROM raw.market_daily")
    except (UnknownDatasetError, OSError, ValueError) as error:
        raise PremarketHistoryError(f"no se puede leer `raw.market_daily`: {error}") from error


class _CachedStore:
    """El almacen con la consulta que **repite** #141 memorizada: una lectura, no cientos.

    Medir la ventana de un candidato pide el gap **sesion a sesion**, y ``premarket_gap.analyse``
    vuelve a leer ``raw.market_daily`` entera en cada llamada: con el almacen real son cientos de
    lecturas identicas. Este envoltorio **no** cambia el calculo —devuelve la misma tabla—, solo
    deja de repetir la lectura (medido: ~35 min -> ~30 s con el almacen del repositorio).

    ``Store`` esta marcado ``@final``, asi que **no** se hereda: el envoltorio declara el contrato
    que #141 usa de verdad (``sql`` y ``root``) y entra con un ``cast`` explicito en el unico punto
    donde se le pasa a #141. No se le anade ni se le cambia nada al almacen.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._store = Store(self.root)
        self._frames: dict[str, pl.DataFrame] = {}

    def sql(self, query: str) -> pl.DataFrame:
        """La misma consulta, leida una sola vez por ejecucion."""
        if query not in self._frames:
            self._frames[query] = self._store.sql(query)
        return self._frames[query]


def sessions_required(store: Store) -> int:
    """Las sesiones que el **conjunto de features** ya usa: el minimo que la serie debe cubrir.

    Se mide en ``derived.labels`` (la ventana etiquetada): una feature que solo exista en una
    decima parte de la muestra no se puede promover sin truncar el entrenamiento.
    """
    try:
        frame = store.sql("SELECT session FROM derived.labels")
    except (UnknownDatasetError, OSError, ValueError) as error:
        raise PremarketHistoryError(f"no se puede leer `derived.labels`: {error}") from error
    if frame.height == 0:
        return 0
    return int(frame.select(pl.col("session").n_unique()).item())


def _series_closes(frame: pl.DataFrame, series_id: str) -> tuple[list[datetime], list[float]]:
    """Los cierres diarios finitos y positivos de una serie, ordenados por instante."""
    rows = frame.filter(
        (pl.col("series_id") == series_id) & pl.col("close").is_finite() & (pl.col("close") > 0)
    ).sort("as_of")
    instants = [_as_utc(stamp) for stamp in rows.get_column("as_of").to_list()]
    closes = [float(value) for value in rows.get_column("close").to_list()]
    return instants, closes


def _index_ohlc(frame: pl.DataFrame) -> dict[date, tuple[float, float]]:
    """``{sesion ET: (open, close)}`` del **proxy declarado** del indice (#107)."""
    rows = frame.filter(
        (pl.col("series_id") == INDEX_SERIES)
        & pl.col("close").is_finite()
        & pl.col("open").is_finite()
        & (pl.col("close") > 0)
        & (pl.col("open") > 0)
    )
    out: dict[date, tuple[float, float]] = {}
    for row in rows.iter_rows(named=True):
        stamp = _as_utc(row["as_of"])
        out[stamp.astimezone(EASTERN).date()] = (float(row["open"]), float(row["close"]))
    return out


def _previous(
    instants: Sequence[datetime], closes: Sequence[float], limit: datetime
) -> tuple[datetime, float] | None:
    """El ultimo cierre con instante ``<= limit`` (o ``None``): la referencia point-in-time."""
    position = bisect_right(instants, limit)
    if position == 0:
        return None
    return instants[position - 1], closes[position - 1]


def _by_session(
    bars: Sequence[premarket_gap.PremarketBar],
) -> dict[date, list[premarket_gap.PremarketBar]]:
    """Las barras agrupadas por sesion **ET** y ordenadas por instante."""
    buckets: dict[date, list[premarket_gap.PremarketBar]] = {}
    for bar in bars:
        buckets.setdefault(bar.instant.astimezone(EASTERN).date(), []).append(bar)
    for day in buckets.values():
        day.sort(key=lambda bar: bar.instant)
    return buckets


def _snapshot(day: date) -> datetime:
    """El instante del snapshot (08:45 ET) de esa sesion, en UTC. Sin reloj."""
    return datetime.combine(day, premarket_gap.SNAPSHOT_ET, tzinfo=EASTERN).astimezone(UTC)


def _inside_rth(bar: premarket_gap.PremarketBar) -> bool:
    """Si la barra cae en la sesion regular (09:30-16:00 ET). El pre-mercado esta fuera."""
    moment = bar.instant.astimezone(EASTERN).time()
    return time(9, 30) <= moment <= time(16, 0)


def measure_candidate(
    spec: CandidateSpec,
    bars: Sequence[premarket_gap.PremarketBar],
    *,
    store: Store,
    index_instants: Sequence[datetime],
    index_closes: Sequence[float],
    index_ohlc: Mapping[date, tuple[float, float]],
    failure: str = "",
) -> tuple[CandidateResult, Redundancy]:
    """Mide un candidato **reutilizando #141**: una llamada a ``premarket_gap.analyse`` por sesion.

    Para cada sesion ET del candidato se le pide el gap a #141 con el instante del snapshot de esa
    sesion, de modo que el recorte point-in-time (ultima barra ``<=`` 08:45 ET, posterior al cierre
    de referencia) y la aritmetica son **los mismos** que los del camino diario, no una copia.

    Devuelve lo medido del candidato y, con las mismas sesiones, la lectura de redundancia frente al
    indice: el dia del gap **no** se calcula dos veces.
    """
    if not bars:
        vacio = CandidateResult(
            spec=spec,
            state="unavailable",
            reason=(
                "la fuente no devolvio ninguna barra (sin red, ventana agotada o mercado cerrado)"
                if not failure
                else f"no hay barras utilizables: {failure}"
            ),
            bars=0,
        )
        return vacio, _redundancy(
            spec.key, window_days=0, overnight=[], intraday=[], close_close=[]
        )
    ordered = sorted(bars, key=lambda bar: bar.instant)
    buckets = _by_session(ordered)
    fetcher = _fixed_bars(ordered)
    with_datum = 0
    fresh = 0
    worst: float | None = None
    example: dict[str, Any] | None = None
    overnight: list[float] = []
    intraday: list[float] = []
    close_close: list[float] = []
    for day in sorted(buckets):
        snapshot = _snapshot(day)
        bucket = buckets[day]
        if spec.stamp_at_window_start:
            # Su sello es el inicio de la ventana, no su cierre: la barra de la propia sesion no
            # esta disponible a las 08:45 y no se cuenta (seria *look-ahead*).
            bucket = [bar for bar in bucket if bar.instant.astimezone(EASTERN).date() != day]
        if not bucket or bucket[0].instant > snapshot:
            # Ninguna barra de ese dia existe antes del snapshot: el dia no aporta dato, asi que
            # **no** se le pide el gap a #141 (la diaria trae una sola barra, la de las 16:00).
            continue
        gap = premarket_gap.analyse(store=store, as_of=snapshot, fetcher=fetcher)
        datum = gap.last_bar
        if gap.state != "measured" or datum is None:
            continue
        staleness = (snapshot - datum.instant).total_seconds() / 60.0
        with_datum += 1
        fresh += 1 if staleness <= MAX_STALENESS_MINUTES else 0
        worst = staleness if worst is None else max(worst, staleness)
        later = next((bar for bar in bucket if bar.instant > snapshot), None)
        example = {
            "session": day.isoformat(),
            "used_bar_utc": datum.instant.astimezone(UTC).isoformat(),
            "used_bar_et": datum.instant.astimezone(EASTERN).isoformat(),
            "snapshot_utc": snapshot.isoformat(),
            "staleness_minutes": round(staleness, 3),
            "bars_considered": gap.bars_considered,
            "next_bar_et": None if later is None else later.instant.astimezone(EASTERN).isoformat(),
            "next_bar_note": "hay barras posteriores al snapshot y **no** se han usado",
        }
        if staleness > MAX_STALENESS_MINUTES:
            continue
        level = index_ohlc.get(day)
        index_reference = _previous(index_instants, index_closes, snapshot)
        move = gap.overnight_move_pct
        if level is None or index_reference is None or move is None or index_reference[1] <= 0:
            continue
        opened, closed = level
        overnight.append(move)
        intraday.append((closed / opened - 1.0) * 100.0)
        close_close.append((closed / index_reference[1] - 1.0) * 100.0)
    window_days = (ordered[-1].instant.date() - ordered[0].instant.date()).days
    if with_datum == 0:
        reason = (
            "ninguna sesion tiene una barra con instante <= 08:45 ET posterior al cierre de "
            "referencia del ES: la unica barra anterior al snapshot **es** ese cierre y no aporta "
            "dato nuevo"
        )
        if spec.stamp_at_window_start:
            reason = (
                "la barra diaria viene sellada al **inicio** de su ventana (00:00 ET) aunque su "
                "valor sea el cierre de las 16:00: leerla a las 08:45 seria *look-ahead*, asi que "
                "**no** se cuenta como dato point-in-time"
            )
        resultado = CandidateResult(
            spec=spec,
            state="unavailable",
            reason=reason,
            bars=len(ordered),
            window_start=ordered[0].instant,
            window_end=ordered[-1].instant,
            extended_hours_seen=any(not _inside_rth(bar) for bar in ordered),
        )
    else:
        resultado = CandidateResult(
            spec=spec,
            state="measured",
            reason=(
                f"medido con {len(ordered)} barras: {with_datum} sesiones con dato point-in-time y "
                f"{fresh} dentro de la tolerancia de {MAX_STALENESS_MINUTES} min"
            ),
            bars=len(ordered),
            window_start=ordered[0].instant,
            window_end=ordered[-1].instant,
            extended_hours_seen=any(not _inside_rth(bar) for bar in ordered),
            sessions_with_datum=with_datum,
            sessions_fresh=fresh,
            staleness_minutes_max=None if worst is None else round(worst, 3),
            example=example,
        )
    return resultado, _redundancy(
        spec.key,
        window_days=window_days,
        overnight=overnight,
        intraday=intraday,
        close_close=close_close,
    )


@dataclass(frozen=True, slots=True)
class Redundancy:
    """La correlacion entre el gap del pre-mercado y el resultado de la sesion del indice."""

    candidate: str
    sessions: int
    window_days: int
    corr_intraday: float | None
    corr_close: float | None

    @property
    def bounded(self) -> bool:
        """Si la ventana es corta, el numero es una **orientacion acotada**, no una conclusion."""
        return self.window_days < BOUNDED_WINDOW_DAYS


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Pearson con su ``n`` publicado; con menos de dos puntos o varianza cero, ``None``."""
    if len(left) < 2 or len(right) != len(left):
        return None
    frame = pl.DataFrame({"left": list(left), "right": list(right)})
    value = frame.select(pl.corr("left", "right")).item()
    return None if value is None else round(float(value), 6)


def _fixed_bars(bars: Sequence[premarket_gap.PremarketBar]) -> premarket_gap.BarsFetcher:
    """Un *fetcher* de #141 que devuelve las barras ya capturadas (el recorte lo hace `analyse`)."""

    def fetch(_series_id: str, *, as_of: datetime) -> Sequence[premarket_gap.PremarketBar]:
        del as_of
        return bars

    return fetch


def _redundancy(
    key: str,
    *,
    window_days: int,
    overnight: Sequence[float],
    intraday: Sequence[float],
    close_close: Sequence[float],
) -> Redundancy:
    """La lectura de redundancia de un candidato: su ``n``, las dos correlaciones y su ventana."""
    return Redundancy(
        candidate=key,
        sessions=len(overnight),
        window_days=window_days,
        corr_intraday=_pearson(overnight, intraday),
        corr_close=_pearson(overnight, close_close),
    )


def decide(
    results: Sequence[CandidateResult], *, sessions_required: int
) -> tuple[CandidateResult | None, dict[str, Any]]:
    """El **veredicto**: si algun candidato cumple las tres condiciones, o si no hay serie.

    ``best`` es el candidato con mas sesiones frescas (la mejor evidencia disponible), sirva o no:
    el informe publica su cifra para que el «no hay serie» se pueda **comprobar**.
    """
    best = max(results, key=lambda item: item.sessions_fresh, default=None)
    enough = [item for item in results if item.sessions_fresh >= sessions_required > 0]
    if enough:
        winner = max(enough, key=lambda item: item.sessions_fresh)
        return best, {
            "decision": "series_sufficient",
            "statement": (
                f"{winner.spec.key} cubre {winner.sessions_fresh} sesiones point-in-time frescas, "
                f"que alcanzan las {sessions_required} del conjunto de features: la serie existe y "
                "la feature se puede definir y re-medir en la Fase 2"
            ),
            "feature_follow_up": (
                "abrir la tarea de definicion de la feature y su re-medida de Fase 2 (con purga y "
                "embargo) sobre esa ventana"
            ),
            "issue_107": "se reabre para este dato",
        }
    if best is None:  # no puede pasar: CANDIDATES no esta vacio
        raise PremarketHistoryError("no hay ningun candidato declarado que medir")
    return best, {
        "decision": "no_series",
        "statement": (
            f"el mejor candidato ({best.spec.key}) cubre {best.sessions_fresh} sesiones "
            f"point-in-time frescas frente a las {sessions_required} que el conjunto de "
            "features ya usa: la serie **no** existe con la granularidad y la ventana que harian "
            "falta"
        ),
        "feature_follow_up": None,
        "issue_107": "se mantiene para este dato",
    }


# ─────────────────────────────────────────────────────────────────────────────
# El informe: los numeros medidos, el veredicto y la valla
# ─────────────────────────────────────────────────────────────────────────────
def _iso(value: datetime | None) -> str | None:
    """Un instante en UTC ISO-8601, o ``None`` sin inventar un valor."""
    return None if value is None else value.astimezone(UTC).isoformat()


def _candidate_payload(item: CandidateResult) -> dict[str, Any]:
    """Un candidato con su ventana **medida** (nunca copiada de la documentacion)."""
    return {
        "key": item.spec.key,
        "series": item.spec.series,
        "interval": item.spec.interval,
        "period_requested": item.spec.period,
        "extended_hours_requested": item.spec.extended_hours,
        "extended_hours_seen": item.extended_hours_seen,
        "stamp_at_window_start": item.spec.stamp_at_window_start,
        "license": FREE_LICENSE,
        "note": item.spec.note,
        "state": item.state,
        "reason": item.reason,
        "bars": item.bars,
        "window_start_utc": _iso(item.window_start),
        "window_end_utc": _iso(item.window_end),
        "window_days": item.window_days,
        "bounded": item.bounded,
        "sessions_with_datum": item.sessions_with_datum,
        "sessions_fresh": item.sessions_fresh,
        "staleness_minutes_max": item.staleness_minutes_max,
        "example": item.example,
    }


def _redundancy_payload(item: Redundancy) -> dict[str, Any]:
    """La lectura de redundancia de un candidato con su ``n`` y su ventana."""
    return {
        "candidate": item.candidate,
        "sessions": item.sessions,
        "window_days": item.window_days,
        "bounded": item.bounded,
        "corr_intraday": item.corr_intraday,
        "corr_close": item.corr_close,
    }


def analyse(
    *,
    store: Store,
    as_of: datetime,
    provider: Callable[[CandidateSpec], Sequence[premarket_gap.PremarketBar]] | None = None,
) -> dict[str, Any]:
    """Mide los candidatos y publica el informe: **una** decision, con sus numeros.

    ``provider`` es inyectable (patron de #141): sin el se descarga con horas extendidas. Un fallo
    de la fuente es una **ausencia declarada** con su motivo, nunca una excepcion.
    """
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise PremarketHistoryError(f"`as_of` tiene que declarar zona horaria: {as_of!r}")
    session, snapshot = premarket_gap.snapshot_instant(as_of)
    # Una lectura por consulta: medir cientos de sesiones no puede escanear el almacen cientos de
    # veces (el calculo es el de #141, que lee `raw.market_daily` en cada `analyse`). `Store` es
    # `@final`, asi que el envoltorio entra con un `cast` declarado: expone lo unico que #141 usa.
    source = cast("Store", _CachedStore(store.root))
    frame = _daily(source)
    index_instants, index_closes = _series_closes(frame, INDEX_SERIES)
    index_ohlc = _index_ohlc(frame)
    required = sessions_required(source)
    fetch = _yfinance_bars if provider is None else provider
    results: list[CandidateResult] = []
    redundancies: list[Redundancy] = []
    for spec in CANDIDATES:
        try:
            bars: tuple[premarket_gap.PremarketBar, ...] = tuple(fetch(spec))
            failure = ""
        except Exception as error:  # la fuente caida es una ausencia, no una caida
            bars, failure = (), f"{type(error).__name__}: {error}"
        result, redundancy = measure_candidate(
            spec,
            bars,
            store=source,
            index_instants=index_instants,
            index_closes=index_closes,
            index_ohlc=index_ohlc,
            failure=failure,
        )
        results.append(result)
        redundancies.append(redundancy)
    best, verdict = decide(results, sessions_required=required)
    return _payload(
        as_of=as_of,
        session=session,
        snapshot=snapshot,
        results=results,
        redundancies=redundancies,
        required=required,
        best=best,
        verdict=verdict,
    )


def _payload(
    *,
    as_of: datetime,
    session: date,
    snapshot: datetime,
    results: Sequence[CandidateResult],
    redundancies: Sequence[Redundancy],
    required: int,
    best: CandidateResult | None,
    verdict: Mapping[str, Any],
) -> dict[str, Any]:
    """El informe completo: candidatos, redundancia, veredicto, valla de honestidad y limites."""
    return {
        "task": TASK,
        "module": MODULE,
        "as_of_utc": as_of.astimezone(UTC).isoformat(),
        "session": session.isoformat(),
        "snapshot_utc": snapshot.isoformat(),
        "snapshot_et": premarket_gap.SNAPSHOT_ET.isoformat(),
        "requirements": {
            "sessions_required": required,
            "sessions_required_source": (
                "`derived.labels`: la ventana que el conjunto de features ya usa"
            ),
            "max_staleness_minutes": MAX_STALENESS_MINUTES,
            "bounded_window_days": BOUNDED_WINDOW_DAYS,
            "conditions": (
                "point-in-time (una barra <= 08:45 ET y ninguna de la sesion que se decide), "
                f"frescura (<= {MAX_STALENESS_MINUTES} min) y ventana (>= las sesiones del "
                "conjunto de features)"
            ),
        },
        "candidates": [_candidate_payload(item) for item in results],
        "paid_sources": [
            {
                "vendor": item.vendor,
                "offering": item.offering,
                "verified": item.verified,
                "statement": item.statement,
                "cost": "decision del propietario: **no** se contrata nada en esta medicion",
            }
            for item in PAID_CANDIDATES
        ],
        "redundancy": [_redundancy_payload(item) for item in redundancies if item.sessions > 0],
        "verdict": {
            **verdict,
            "candidates_measured": len([item for item in results if item.state == "measured"]),
            "best_candidate": None if best is None else best.spec.key,
            "best_sessions_fresh": None if best is None else best.sessions_fresh,
            "reopens_107": verdict["decision"] == "series_sufficient",
            "next": (
                "**#148** es lo que se hace con el dato mientras tanto: la regla 21 lo consume "
                "como **dato declarado** del dia, sin esperar a ninguna feature"
            ),
        },
        "point_in_time": (
            "solo se usa la ultima barra con instante <= 08:45 ET y posterior al cierre de "
            "referencia; ninguna barra de la sesion que se decide entra en la medicion"
        ),
        "honesty": [
            "no hay edge demostrado: esto mide la **disponibilidad** de una serie, no su poder "
            "predictivo",
            "`gate`, `phase1_ready` y `phase2_ready` **no** se mueven: ninguna feature se promueve "
            "aqui",
            "las fuentes de pago se enumeran con lo que **declara** el proveedor y se marcan como "
            "**no verificadas**: comprarlas es una decision del propietario",
        ],
        "limitations": [
            "la barra **diaria** del ES es una ventana de 24 h que termina a las 16:00 ET: su "
            "unica barra anterior al snapshot **es** el cierre de referencia",
            "la ventana del intradia gratuito es **rodante**: la cifra medida caduca, y el informe "
            "se reproduce desde las barras capturadas (`--bars-dir`)",
            "el `open` diario del indice es un **proxy declarado** (#107), no la subasta de las "
            "09:30",
            "la correlacion de redundancia **no** mide *edge* ni autoriza a operar",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Descarga, captura e informe en prosa
# ─────────────────────────────────────────────────────────────────────────────
def _yfinance_bars(spec: CandidateSpec) -> tuple[premarket_gap.PremarketBar, ...]:
    """Barras del candidato en la fuente gratuita: **la unica pieza que no se reutiliza**.

    #141 descarga el gap del dia con **una** combinacion fija de granularidad y ventana; aqui hay
    que probar cuatro, asi que entra por el candidato. El recorte point-in-time y el formato de
    barras si son los de #141 (`analyse`, `load_bars`, `save_bars`).
    """
    import yfinance as yf

    ticker: Any = yf.Ticker(spec.series)
    frame: Any = ticker.history(
        period=spec.period,
        interval=spec.interval,
        prepost=spec.extended_hours,
        auto_adjust=False,
    )
    if frame is None or len(frame.index) == 0:
        return ()
    index = frame.index
    if getattr(index, "tz", None) is None:
        # La fuente de futuros entrega UTC; un sello sin zona se lee como UTC y se declara asi.
        index = index.tz_localize("UTC")
    stamps = list(index.tz_convert("UTC"))
    closes = cast("list[object]", frame["Close"].tolist())
    bars: list[premarket_gap.PremarketBar] = []
    for stamp, price in zip(stamps, closes, strict=True):
        if price is None:
            continue
        number = float(cast("float", price))
        if not math.isfinite(number):
            continue
        bars.append(premarket_gap.PremarketBar(instant=stamp.to_pydatetime(), price=number))
    return tuple(sorted(bars, key=lambda bar: bar.instant))


def _number(value: object) -> str:
    """Un numero del informe, o ``null`` (nunca un cero inventado)."""
    return "`null`" if value is None else f"`{value}`"


def render_markdown(payload: dict[str, Any]) -> str:
    """El informe en prosa: la tabla de candidatos, la redundancia, el veredicto y la valla."""
    requirements = payload["requirements"]
    lines: list[str] = [
        "# Serie historica point-in-time del pre-mercado del ES (`ES=F`) — `#149`",
        "",
        f"- **Sesion declarada:** {payload['session']} · **as_of:** {payload['as_of_utc']} · "
        f"**snapshot:** {payload['snapshot_utc']} ({payload['snapshot_et']} ET)",
        f"- **Condiciones:** {requirements['conditions']}",
        f"- **Sesiones que el conjunto de features ya usa:** "
        f"**{requirements['sessions_required']}** ({requirements['sessions_required_source']})",
        f"- **Tolerancia de frescura:** {requirements['max_staleness_minutes']} min · "
        f"**ventana acotada** por debajo de {requirements['bounded_window_days']} dias",
        "",
        "## Candidatos de la fuente gratuita (ventana **medida**, no declarada)",
        "",
        "| Candidato | Intervalo | Periodo | Ventana medida | Barras | Horas extendidas | "
        "Sesiones con dato | Sesiones frescas | Peor desfase (min) | Estado |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for candidate in payload["candidates"]:
        window = (
            "sin barras"
            if candidate["window_start_utc"] is None
            else f"{candidate['window_start_utc'][:10]} -> {candidate['window_end_utc'][:10]} "
            f"({candidate['window_days']} d{', acotada' if candidate['bounded'] else ''})"
        )
        lines.append(
            f"| `{candidate['key']}` | {candidate['interval']} | {candidate['period_requested']} | "
            f"{window} | {candidate['bars']} | "
            f"{'si' if candidate['extended_hours_seen'] else 'no'} | "
            f"{candidate['sessions_with_datum']} | {candidate['sessions_fresh']} | "
            f"{_number(candidate['staleness_minutes_max'])} | `{candidate['state']}` |"
        )
    lines.extend(["", "**Motivo de cada candidato.**", ""])
    lines.extend(f"- `{item['key']}`: {item['reason']}" for item in payload["candidates"])
    lines.append("")
    lines.extend(_example_lines(payload))
    lines.extend(_redundancy_lines(payload))
    lines.extend(_paid_lines(payload))
    verdict = payload["verdict"]
    lines.extend(
        [
            "## Veredicto",
            "",
            f"- **Decision:** `{verdict['decision']}` — {verdict['statement']}",
            f"- Mejor candidato medido: `{verdict['best_candidate']}` con "
            f"**{verdict['best_sessions_fresh']}** sesiones frescas frente a las "
            f"{requirements['sessions_required']} que hacen falta.",
            f"- **#107:** {verdict['issue_107']}.",
        ]
    )
    if verdict["feature_follow_up"] is not None:
        lines.append(f"- Seguimiento de la feature: {verdict['feature_follow_up']}")
    lines.append(f"- Mientras tanto: {verdict['next']}")
    lines.extend(["", "## Point-in-time y alcance", "", f"- {payload['point_in_time']}"])
    lines.extend(["", "## Limitaciones (declaradas)", ""])
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.extend(["", "## Valla de honestidad", ""])
    lines.extend(f"- {item}" for item in payload["honesty"])
    lines.append("")
    return "\n".join(lines)


def _example_lines(payload: Mapping[str, Any]) -> list[str]:
    """El ejemplo **real** del recorte point-in-time (C2): la barra usada y las que no."""
    lines = ["## Point-in-time verificado (ejemplo real)", ""]
    shown = 0
    for candidate in cast("list[dict[str, Any]]", payload["candidates"]):
        example = candidate["example"]
        if example is None:
            continue
        shown += 1
        lines.append(
            f"- `{candidate['key']}` · sesion {example['session']}: se usa la barra "
            f"{example['used_bar_et']} ({example['used_bar_utc']}), con "
            f"{example['bars_considered']} barras consideradas y un desfase de "
            f"{example['staleness_minutes']} min frente al "
            f"snapshot {example['snapshot_utc']}. {example['next_bar_note']} (la siguiente es "
            f"{example['next_bar_et']})."
        )
    if shown == 0:
        lines.append("- Ningun candidato tiene ejemplo: ninguno sirve un dato point-in-time.")
    lines.append("")
    return lines


def _redundancy_lines(payload: Mapping[str, Any]) -> list[str]:
    """La redundancia medida, con su ``n`` y su ventana (nunca como promesa de *edge*)."""
    lines = ["## Redundancia medida (con su `n`)", ""]
    if not payload["redundancy"]:
        lines.extend(
            [
                "- No hay sesiones alineadas suficientes: **no** se publica ninguna correlacion.",
                "",
            ]
        )
        return lines
    lines.extend(
        [
            "| Candidato | Sesiones (`n`) | Ventana (dias) | corr(gap, sesion del indice) | "
            "corr(gap, cierre a cierre) |",
            "|---|---|---|---|---|",
        ]
    )
    for item in cast("list[dict[str, Any]]", payload["redundancy"]):
        bounded = " (ventana acotada)" if item["bounded"] else ""
        lines.append(
            f"| `{item['candidate']}` | {item['sessions']}{bounded} | {item['window_days']} | "
            f"{_number(item['corr_intraday'])} | {_number(item['corr_close'])} |"
        )
    lines.extend(
        [
            "",
            "- Ninguna de las dos correlaciones mide *edge*: son una **orientacion acotada** sobre "
            "la ventana que la fuente permite, y el `open` del indice es un proxy declarado "
            "(#107).",
            "",
        ]
    )
    return lines


def _paid_lines(payload: Mapping[str, Any]) -> list[str]:
    """Las fuentes de pago: lo que **declara** el proveedor, sin verificarlo aqui."""
    lines = [
        "## Fuentes de pago (declaradas, **no** verificadas)",
        "",
        "| Proveedor | Oferta | Verificada aqui | Coste |",
        "|---|---|---|---|",
    ]
    for item in cast("list[dict[str, Any]]", payload["paid_sources"]):
        lines.append(
            f"| {item['vendor']} | {item['offering']} | {'si' if item['verified'] else '**no**'} | "
            f"{item['cost']} |"
        )
    lines.append("")
    lines.extend(f"- {item['statement']}" for item in payload["paid_sources"])
    lines.append("")
    return lines


def write_report(payload: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    """Escribe el ``.json`` y el ``.md`` de la medicion y devuelve sus rutas."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"{REPORT_PREFIX}.json"
    markdown_path = reports_dir / f"{REPORT_PREFIX}.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, markdown_path


# ─────────────────────────────────────────────────────────────────────────────
# Ejecucion
# ─────────────────────────────────────────────────────────────────────────────
def _parse_as_of(value: str | None) -> datetime:
    """El instante ISO-8601 **con zona** declarado por el llamante (el modulo no lee el reloj)."""
    if value is None or not value.strip():
        raise PremarketHistoryError("`--as-of` es obligatorio (ISO-8601 con zona horaria)")
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as error:
        raise PremarketHistoryError(f"`--as-of` no es ISO-8601: {value!r}") from error
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise PremarketHistoryError(f"`--as-of` tiene que declarar zona horaria: {value!r}")
    return moment


def _capture(
    directory: Path,
    *,
    provider: Callable[[CandidateSpec], Sequence[premarket_gap.PremarketBar]] | None = None,
) -> dict[str, Path]:
    """Descarga cada candidato y guarda sus barras (una vez) para reproducir **sin red**."""
    fetch = _yfinance_bars if provider is None else provider
    written: dict[str, Path] = {}
    for spec in CANDIDATES:
        try:
            bars = tuple(fetch(spec))
        except Exception as error:
            raise PremarketHistoryError(
                f"no se pueden capturar las barras de `{spec.key}`: {error}"
            ) from error
        written[spec.key] = premarket_gap.save_bars(bars, directory / f"{spec.key}.json")
    return written


def _directory_provider(
    directory: Path,
) -> Callable[[CandidateSpec], Sequence[premarket_gap.PremarketBar]]:
    """Un *provider* que lee las barras capturadas: mide **sin red** (patron de #141)."""

    def fetch(spec: CandidateSpec) -> Sequence[premarket_gap.PremarketBar]:
        path = directory / f"{spec.key}.json"
        if not path.is_file():
            raise PremarketHistoryError(f"no hay barras capturadas de `{spec.key}` en {directory}")
        return premarket_gap.load_bars(path)

    return fetch


def main(argv: Sequence[str] | None = None) -> int:
    """Mide la serie y publica el informe.

    Codigos de salida: ``0`` = informe emitido (con veredicto y ausencias declaradas); ``2`` =
    argumentos invalidos, almacen ilegible o captura fallida, con el motivo por ``stderr``.

    ``--bars-dir`` MIDE SIN RED desde barras capturadas; ``--capture-dir`` las descarga y las
    guarda (una vez) para poder reproducir despues.
    """
    parser = argparse.ArgumentParser(
        prog=MODULE, description="Serie historica point-in-time del pre-mercado del ES (#149)"
    )
    parser.add_argument("--data-root", type=Path, default=None, help="raiz del almacen")
    parser.add_argument("--as-of", default=None, help="instante declarado ISO-8601 con zona")
    parser.add_argument("--reports-dir", type=Path, default=None, help="directorio de informes")
    parser.add_argument(
        "--bars-dir", type=Path, default=None, help="barras capturadas por candidato: mide sin red"
    )
    parser.add_argument(
        "--capture-dir", type=Path, default=None, help="descarga las barras y las guarda aqui"
    )
    parser.add_argument("--settings", type=Path, default=None, help="ruta de settings.yaml")
    args = parser.parse_args(argv)

    try:
        moment = _parse_as_of(cast("str | None", args.as_of))
    except PremarketHistoryError as error:
        print(f"no se puede medir la serie: {error}", file=sys.stderr)
        return 2
    try:
        settings = load_settings(args.settings)
    except ConfigurationError as error:
        print(f"no se puede leer la configuracion: {error}", file=sys.stderr)
        return 2
    data_root = Path(args.data_root) if args.data_root is not None else Path(settings.data.root)

    capture = cast("Path | None", args.capture_dir)
    if capture is not None:
        try:
            written = _capture(capture)
        except PremarketHistoryError as error:
            print(f"no se pueden capturar las barras: {error}", file=sys.stderr)
            return 2
        print(f"barras capturadas: {len(written)} candidatos en {capture}")

    bars_dir = cast("Path | None", args.bars_dir)
    provider = None if bars_dir is None else _directory_provider(bars_dir)
    try:
        payload = analyse(store=Store(data_root), as_of=moment, provider=provider)
    except PremarketHistoryError as error:
        print(f"no se puede medir la serie: {error}", file=sys.stderr)
        return 2

    reports_dir = cast("Path | None", args.reports_dir)
    if reports_dir is not None:
        json_path, markdown_path = write_report(payload, reports_dir)
        print(f"informe: {json_path} y {markdown_path}")
    print(render_markdown(payload))
    verdict = cast("dict[str, Any]", payload["verdict"])
    print(f"veredicto: {verdict['decision']} — #107 {verdict['issue_107']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
