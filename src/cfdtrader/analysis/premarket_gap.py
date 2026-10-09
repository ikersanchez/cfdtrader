"""Gap de pre-mercado del futuro ES como dato **declarado** de la decision (tarea #141).

**Que entrega.** El movimiento del futuro E-mini S&P 500 (``ES=F``) entre el cierre de la sesion
anterior (16:00 ET) y el **instante del snapshot** (08:45 ET) de la sesion que se decide, medido
**point-in-time**: solo se usan barras cuyo instante es ``<=`` el snapshot, nunca el ``open`` ni el
``close`` de la propia sesion.

**Por que existe.** `plan.md` §13 congela el snapshot a las 08:45 ET y entrega a las 09:00 ET, pero
ninguna de las 10 features del modelo (`BASELINE_FEATURES`) usa el precio del futuro: la sesion del
2026-10-09 el camino diario emitio `SHORT` mientras el ES cotizaba en positivo en el pre-mercado y
el modelo **no podia verlo**. Este modulo publica ese dato como **contexto declarado** de la
decision.

**Por que NO es una feature del modelo, y que queda bloqueado.** La feature del modelo (y su
re-medida de la Fase 2) exige una **serie historica** de pre-mercado del ES, y las fuentes gratuitas
no la dan:
la barra **diaria** de ``ES=F`` es una ventana de **24 h** que termina a las **16:00 ET**
(``open(t) ~= close(t-1)``), asi que el movimiento de pre-mercado vive **entero** en la barra ``t``,
que a las 08:45 **aun no existe** (seria *look-ahead*); y el intradia de 5 minutos con horas
extendidas solo cubre una ventana **rodante de ~60 dias** (limite de Yahoo). Es un problema de
**datos**, no de codigo: la parte de modelo queda declarada como bloqueada por **#107**. Medido el
2026-10-09 sobre `data/`, el retorno diario del ES correlaciona **r = 0,972** con el del ``^GSPC``
(2005->2026, n = 5 466): el unico dato diario disponible es redundante.

**Como se mide.** Se descargan las barras de 5 minutos de ``ES=F`` **con horas extendidas**
(``prepost=True``) y se toma la **ultima barra con instante <= 08:45 ET**. El cierre de referencia
es el ultimo ``raw.market_daily`` de cada serie con ``as_of <= 08:45 ET`` (la sesion ``t-1``).
Se publican **tres** numeros, y **nunca se suman**:

- ``overnight_move_pct`` — ``ES(pre-mercado) / ES(cierre t-1) - 1``: el movimiento del **propio**
  futuro, sin el sesgo del *basis*.
- ``gap_vs_index_pct`` — ``ES(pre-mercado) / ^GSPC(cierre t-1) - 1``: el «gap» de la issue, que
  **incluye** el *basis* estructural del futuro.
- ``basis_pct`` — ``ES(cierre t-1) / ^GSPC(cierre t-1) - 1``: cuanto de ese gap es *basis*.

**Estado y honestidad.** ``state`` es ``measured``, ``unavailable`` (sin barra, sin historico o
snapshot en el futuro) o ``not_a_session``. Un ``unavailable`` es una **ausencia declarada** con su
motivo, nunca un ``0``. Esto **no** es una afirmacion de *edge* ni una feature: la Fase 2 sigue
``not_evaluable``/``fail`` con ``phase2_ready = false`` y el carril B sigue bloqueado (`plan.md`
§11.6 / §19.7).

Sin reloj (el instante entra por ``--as-of``); la descarga vive detras de un *fetcher* inyectable
para que el nucleo sea puro y testeable sin red.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, Final, cast

import polars as pl

from cfdtrader.data.calendar import EASTERN
from cfdtrader.data.settings import ConfigurationError, load_settings
from cfdtrader.data.store import Store, UnknownDatasetError

__all__ = [
    "ES_SERIES",
    "INDEX_SERIES",
    "INTERVAL",
    "LOOKBACK_PERIOD",
    "MODULE",
    "REPORT_PREFIX",
    "SNAPSHOT_ET",
    "TASK",
    "BarsFetcher",
    "PremarketBar",
    "PremarketGap",
    "PremarketGapError",
    "analyse",
    "load_bars",
    "main",
    "render_markdown",
    "save_bars",
    "write_report",
]

MODULE: Final[str] = "cfdtrader.analysis.premarket_gap"
TASK: Final[str] = "#141"
REPORT_PREFIX: Final[str] = "premarket_gap"

#: La serie del futuro: un **proxy declarado** del subyacente del CFD (`plan.md` §3.1), no el CFD.
ES_SERIES: Final[str] = "ES=F"

#: La serie del indice, de la que se lee el cierre de la sesion anterior.
INDEX_SERIES: Final[str] = "^GSPC"

#: Instante del snapshot declarado por `plan.md` §13 (08:45 ET), en hora de Nueva York.
SNAPSHOT_ET: Final[time] = time(8, 45)

#: Granularidad de la descarga y su ventana rodante (limite de Yahoo para 5 minutos).
INTERVAL: Final[str] = "5m"
LOOKBACK_PERIOD: Final[str] = "60d"

#: Motivos de una ausencia: se publican en prosa, nunca como un valor de relleno.
_REASON_SNAPSHOT_AHEAD: Final[str] = (
    "el instante declarado `--as-of` es anterior al snapshot de las 08:45 ET de la sesion: el dato "
    "no existe todavia y no se adelanta"
)
_REASON_NO_BARS: Final[str] = (
    "la fuente no devolvio ninguna barra de pre-mercado <= snapshot (sin red, ventana rodante de 5 "
    "minutos agotada o mercado cerrado)"
)
_REASON_NO_REFERENCE: Final[str] = (
    "el almacen no tiene el cierre de la sesion anterior de `ES=F` o de `^GSPC` con `as_of <= "
    "snapshot`: sin referencia no hay gap comparable"
)


class PremarketGapError(Exception):
    """Raiz de los errores del modulo (argumentos invalidos o almacen ilegible)."""


# ─────────────────────────────────────────────────────────────────────────────
# Modelo de datos
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PremarketBar:
    """Una barra de pre-mercado: su instante (UTC) y su precio (el cierre de la barra)."""

    instant: datetime
    price: float


@dataclass(frozen=True, slots=True)
class ReferenceClose:
    """El cierre de una sesion anterior, con su instante y su sesion (ET)."""

    session: date
    instant: datetime
    price: float


#: Firma de un *fetcher*: recibe la serie y el instante tope y devuelve sus barras.
BarsFetcher = Callable[..., Sequence[PremarketBar]]


def _to_utc(stamp: object) -> datetime:
    """Convierte un sello temporal de la fuente a ``datetime`` UTC (o error tipado).

    Acepta lo que devuelve ``pandas``/``yfinance`` (un ``Timestamp`` con ``to_pydatetime``) y un
    ``datetime`` normal. Un sello sin zona se **asume** UTC y se declara asi: la fuente de futuros
    entrega UTC, pero no se inventa otra zona.
    """
    convert = getattr(stamp, "to_pydatetime", None)
    value = convert() if callable(convert) else stamp
    if not isinstance(value, datetime):
        raise PremarketGapError(f"la fuente devolvio un sello temporal no admisible: {stamp!r}")
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Descarga (detras de un *fetcher* inyectable) y lectura del almacen
# ─────────────────────────────────────────────────────────────────────────────
def _yfinance_bars(series_id: str, *, as_of: datetime) -> tuple[PremarketBar, ...]:
    """Barras de 5 minutos **con horas extendidas** de la serie (descarga por defecto).

    ``prepost=True`` es lo que trae el pre-mercado: sin el, Yahoo solo devuelve la sesion regular
    (09:30-16:00 ET), que **no** contiene el movimiento de las 08:45. Se devuelven **todas** las
    barras; el recorte point-in-time lo hace :func:`analyse`.
    """
    import yfinance as yf

    ticker: Any = yf.Ticker(series_id)
    frame: Any = ticker.history(
        period=LOOKBACK_PERIOD,
        interval=INTERVAL,
        prepost=True,
        auto_adjust=False,
    )
    if frame is None or len(frame.index) == 0:
        return ()
    closes = cast("list[object]", frame["Close"].tolist())
    stamps = cast("list[object]", list(frame.index))
    bars: list[PremarketBar] = []
    for stamp, price in zip(stamps, closes, strict=True):
        if price is None:
            continue
        number = float(cast("float", price))
        if not math.isfinite(number):
            continue
        bars.append(PremarketBar(instant=_to_utc(stamp), price=number))
    return tuple(sorted(bars, key=lambda bar: bar.instant))


def _market_frame(store: Store) -> pl.DataFrame:
    """``raw.market_daily`` completo (se filtra en memoria: nunca se compone SQL con literales)."""
    try:
        return store.sql("SELECT series_id, as_of, close FROM raw.market_daily")
    except (UnknownDatasetError, OSError, ValueError) as error:
        raise PremarketGapError(
            f"no se puede leer `raw.market_daily` en {store.root}: {error}"
        ) from error


def _last_close(store: Store, series_id: str, *, instant: datetime) -> ReferenceClose | None:
    """El ultimo ``raw.market_daily`` de la serie con ``as_of <= instant`` (o ``None``).

    Como el sello del diario es el cierre de la sesion (16:00 ET), el ultimo con ``as_of`` anterior
    al snapshot de las 08:45 es la sesion **anterior**: la referencia correcta y point-in-time.
    """
    frame = _market_frame(store)
    rows = frame.filter(
        (pl.col("series_id") == series_id)
        & (pl.col("as_of").cast(pl.Datetime("us", "UTC")) <= instant)
        & pl.col("close").is_finite()
        & (pl.col("close") > 0)
    )
    if rows.height == 0:
        return None
    row = rows.sort("as_of").tail(1).to_dicts()[0]
    stamp = _to_utc(row["as_of"])
    return ReferenceClose(
        session=stamp.astimezone(EASTERN).date(),
        instant=stamp,
        price=float(cast("float", row["close"])),
    )


# ─────────────────────────────────────────────────────────────────────────────
# El resultado
# ─────────────────────────────────────────────────────────────────────────────
def _pct(numerator: float, denominator: float) -> float:
    """Porcentaje ``(numerator / denominator - 1) * 100`` redondeado (determinista)."""
    return round((numerator / denominator - 1.0) * 100.0, 8)


def _reference(value: ReferenceClose | None) -> dict[str, Any] | None:
    """Serializa un cierre de referencia (o ``None``, que es una ausencia declarada)."""
    if value is None:
        return None
    return {
        "session": value.session.isoformat(),
        "instant_utc": value.instant.astimezone(UTC).isoformat(),
        "close": round(value.price, 6),
    }


@dataclass(frozen=True, slots=True)
class PremarketGap:
    """El gap de pre-mercado del ES: sus tres numeros, su estado y su motivo.

    Un ``state`` distinto de ``measured`` deja los numeros a ``None``: una ausencia declarada, jamas
    un ``0``.
    """

    session: date
    as_of: datetime
    snapshot: datetime
    state: str
    reason: str
    bars_considered: int
    last_bar: PremarketBar | None = None
    es_previous: ReferenceClose | None = None
    index_previous: ReferenceClose | None = None

    @property
    def overnight_move_pct(self) -> float | None:
        """``ES(pre-mercado) / ES(cierre t-1) - 1``: el movimiento del futuro, sin *basis*."""
        if self.last_bar is None or self.es_previous is None:
            return None
        return _pct(self.last_bar.price, self.es_previous.price)

    @property
    def gap_vs_index_pct(self) -> float | None:
        """``ES(pre-mercado) / ^GSPC(cierre t-1) - 1``: el gap de la issue, **con** *basis*."""
        if self.last_bar is None or self.index_previous is None:
            return None
        return _pct(self.last_bar.price, self.index_previous.price)

    @property
    def basis_pct(self) -> float | None:
        """``ES(cierre t-1) / ^GSPC(cierre t-1) - 1``: cuanto del gap es *basis* estructural."""
        if self.es_previous is None or self.index_previous is None:
            return None
        return _pct(self.es_previous.price, self.index_previous.price)

    def payload(self) -> dict[str, Any]:
        """El informe como *mapping* JSON-serializable (el artefacto que se escribe)."""
        return {
            "task": TASK,
            "module": MODULE,
            "series": {"es": ES_SERIES, "index": INDEX_SERIES, "interval": INTERVAL},
            "session": self.session.isoformat(),
            "as_of_utc": self.as_of.astimezone(UTC).isoformat(),
            "snapshot_utc": self.snapshot.astimezone(UTC).isoformat(),
            "snapshot_et": SNAPSHOT_ET.isoformat(),
            "state": self.state,
            "reason": self.reason,
            "point_in_time": (
                "solo se usa la ultima barra con instante <= snapshot (08:45 ET): nunca el `open` "
                "ni el `close` de la sesion que se decide"
            ),
            "not_a_model_feature": (
                "es un **dato declarado de la decision**, no una feature: el modelo no puede "
                "consumirlo porque las fuentes gratuitas no dan la **serie historica** de "
                "pre-mercado del ES (la barra diaria es de 24 h y seria *look-ahead*). La parte de "
                "modelo y la re-medida de la Fase 2 quedan bloqueadas por #107"
            ),
            "last_bar_utc": None if self.last_bar is None else self.last_bar.instant.isoformat(),
            "es_premarket_price": None if self.last_bar is None else round(self.last_bar.price, 6),
            "bars_considered": self.bars_considered,
            "es_previous": _reference(self.es_previous),
            "index_previous": _reference(self.index_previous),
            "overnight_move_pct": self.overnight_move_pct,
            "gap_vs_index_pct": self.gap_vs_index_pct,
            "basis_pct": self.basis_pct,
            "limitations": [
                "el ES es un **proxy declarado** del subyacente del CFD (`plan.md` §3.1): "
                "`SPX500:CFD` sigue `unavailable` (#107)",
                "`gap_vs_index_pct` **incluye** el *basis* estructural del futuro; para el "
                "movimiento puro usa `overnight_move_pct` (el *basis* se publica aparte)",
                "la referencia es el cierre de la sesion anterior del indice (16:00 ET), no el "
                "`open` de la subasta: el gap mide el pre-mercado, no la apertura",
                "la ventana de 5 minutos con horas extendidas que da la fuente es **rodante** "
                "(~60 dias): el dato es **de hoy**, no un historico con el que re-medir la Fase 2",
            ],
            "honesty": [
                "no hay edge demostrado: Fase 2 `not_evaluable`/`fail`, `phase2_ready = false`",
                "carril A (apoyo a la decision, ejecucion manual): el carril B sigue bloqueado "
                "(`plan.md` §11.6 / §19.7)",
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Analisis
# ─────────────────────────────────────────────────────────────────────────────
def snapshot_instant(as_of: datetime) -> tuple[date, datetime]:
    """``(sesion ET, instante UTC del snapshot)`` que declara ``as_of`` (sin reloj)."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise PremarketGapError(f"`as_of` tiene que declarar zona horaria: {as_of!r}")
    session = as_of.astimezone(EASTERN).date()
    snapshot = datetime.combine(session, SNAPSHOT_ET, tzinfo=EASTERN).astimezone(UTC)
    return session, snapshot


def _fetch_bars(
    fetcher: BarsFetcher | None, snapshot: datetime
) -> tuple[tuple[PremarketBar, ...], str]:
    """Barras del *fetcher* (por defecto, la descarga real) o ``()`` con el motivo del fallo.

    **Nunca lanza**: un fallo de la fuente es una ausencia declarada, no una caida del diario.
    """
    provider: BarsFetcher = _yfinance_bars if fetcher is None else fetcher
    try:
        return tuple(provider(ES_SERIES, as_of=snapshot)), ""
    except Exception as error:
        return (), f"{type(error).__name__}: {error}"


def analyse(
    *,
    store: Store,
    as_of: datetime,
    fetcher: BarsFetcher | None = None,
) -> PremarketGap:
    """El gap de pre-mercado del ES para la sesion de ``as_of`` (point-in-time, sin *look-ahead*).

    Se toman **solo** las barras con instante ``<=`` el snapshot de las 08:45 ET y **posteriores**
    al cierre de la sesion anterior, y se publica la ultima. ``fetcher`` es inyectable: sin el, se
    descarga de ``yfinance`` con horas extendidas. Un fallo de la fuente devuelve ``state:
    "unavailable"`` con su motivo, no una excepcion.
    """
    session, snapshot = snapshot_instant(as_of)
    moment = as_of.astimezone(UTC)
    if moment < snapshot:
        return PremarketGap(
            session=session,
            as_of=moment,
            snapshot=snapshot,
            state="unavailable",
            reason=_REASON_SNAPSHOT_AHEAD,
            bars_considered=0,
        )
    es_previous = _last_close(store, ES_SERIES, instant=snapshot)
    index_previous = _last_close(store, INDEX_SERIES, instant=snapshot)
    if es_previous is None or index_previous is None:
        return PremarketGap(
            session=session,
            as_of=moment,
            snapshot=snapshot,
            state="unavailable",
            reason=_REASON_NO_REFERENCE,
            bars_considered=0,
            es_previous=es_previous,
            index_previous=index_previous,
        )
    bars, failure = _fetch_bars(fetcher, snapshot)
    window = tuple(
        bar for bar in bars if bar.instant <= snapshot and bar.instant > es_previous.instant
    )
    if not window:
        reason = _REASON_NO_BARS if not failure else f"{_REASON_NO_BARS} — {failure}"
        return PremarketGap(
            session=session,
            as_of=moment,
            snapshot=snapshot,
            state="unavailable",
            reason=reason,
            bars_considered=0,
            es_previous=es_previous,
            index_previous=index_previous,
        )
    return PremarketGap(
        session=session,
        as_of=moment,
        snapshot=snapshot,
        state="measured",
        reason=(
            f"medido con {len(window)} barras de {INTERVAL} con horas extendidas; se publica la "
            "ultima barra <= snapshot"
        ),
        bars_considered=len(window),
        last_bar=window[-1],
        es_previous=es_previous,
        index_previous=index_previous,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Captura y publicacion
# ─────────────────────────────────────────────────────────────────────────────
def load_bars(path: Path | str) -> tuple[PremarketBar, ...]:
    """Lee barras capturadas (JSON: ``[{"instant": ISO, "price": numero}, ...]``).

    Sirve para reproducir un gap **sin red** y para inyectar barras en los tests.
    """
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PremarketGapError(f"no se puede leer {source}: {error}") from error
    if not isinstance(raw, list):
        raise PremarketGapError(f"el fichero de barras {source} no es una lista")
    bars: list[PremarketBar] = []
    for item in cast("list[object]", raw):
        if not isinstance(item, dict):
            raise PremarketGapError(f"una barra de {source} no es un objeto: {item!r}")
        entry = cast("dict[str, object]", item)
        instant = entry.get("instant")
        price = entry.get("price")
        if not isinstance(instant, str) or not isinstance(price, (int, float)):
            raise PremarketGapError(f"barra incompleta en {source}: {item!r}")
        moment = _to_utc(datetime.fromisoformat(instant))
        bars.append(PremarketBar(instant=moment, price=float(price)))
    return tuple(sorted(bars, key=lambda bar: bar.instant))


def save_bars(bars: Sequence[PremarketBar], path: Path | str) -> Path:
    """Guarda las barras capturadas (JSON) para reproducir el gap sin volver a descargar."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"instant": bar.instant.astimezone(UTC).isoformat(), "price": round(bar.price, 6)}
        for bar in bars
    ]
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def render_markdown(payload: dict[str, Any]) -> str:
    """El informe en prosa: el gap, su aritmetica, lo declarado y lo bloqueado."""
    es_previous = payload["es_previous"]
    index_previous = payload["index_previous"]

    def _value(key: str) -> str:
        value = payload[key]
        return "`null` (no medido)" if value is None else f"**{value}** %"

    lines: list[str] = [
        "# Gap de pre-mercado del futuro ES (`ES=F`) — dato declarado de la decision (#141)",
        "",
        f"- **Sesion:** {payload['session']} · **snapshot:** {payload['snapshot_utc']} "
        f"(08:45 ET) · **as_of:** {payload['as_of_utc']}",
        f"- **Estado:** `{payload['state']}` — {payload['reason']}",
        f"- **Barras consideradas:** {payload['bars_considered']} · "
        f"**ultima barra:** {payload['last_bar_utc'] or '`null`'}",
        f"- **ES en el snapshot:** {payload['es_premarket_price'] or '`null`'}",
        "",
        "## Los tres numeros (nunca se suman)",
        "",
        "| Medida | Valor | Que es |",
        "|---|---|---|",
        f"| `overnight_move_pct` | {_value('overnight_move_pct')} | movimiento del **propio** ES "
        "desde el cierre de la sesion anterior (sin *basis*) |",
        f"| `gap_vs_index_pct` | {_value('gap_vs_index_pct')} | ES / cierre previo del `^GSPC` "
        "(**incluye** el *basis*) |",
        f"| `basis_pct` | {_value('basis_pct')} | ES / `^GSPC` en el cierre previo (*basis* "
        "estructural) |",
        "",
        "## Referencias",
        "",
        f"- `ES=F` cierre previo: {es_previous['close'] if es_previous else '`null`'} "
        f"({es_previous['instant_utc'] if es_previous else '`null`'})",
        f"- `^GSPC` cierre previo: {index_previous['close'] if index_previous else '`null`'} "
        f"({index_previous['instant_utc'] if index_previous else '`null`'})",
        "",
        "## Point-in-time y alcance",
        "",
        f"- {payload['point_in_time']}",
        f"- {payload['not_a_model_feature']}",
        "",
        "## Limitaciones (declaradas)",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.extend(["", "## Valla de honestidad", ""])
    lines.extend(f"- {item}" for item in payload["honesty"])
    lines.append("")
    return "\n".join(lines)


def write_report(payload: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    """Escribe el ``.json`` y el ``.md`` del gap y devuelve sus rutas."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{REPORT_PREFIX}_{payload['session']}"
    json_path = reports_dir / f"{stem}.json"
    markdown_path = reports_dir / f"{stem}.md"
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
        raise PremarketGapError("`--as-of` es obligatorio (ISO-8601 con zona horaria)")
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as error:
        raise PremarketGapError(f"`--as-of` no es ISO-8601: {value!r}") from error
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise PremarketGapError(f"`--as-of` tiene que declarar zona horaria: {value!r}")
    return moment


def main(argv: Sequence[str] | None = None) -> int:
    """Descarga el pre-mercado del ES, calcula el gap y publica el informe (#141).

    Codigos de salida: ``0`` = informe emitido (medido o ausencia declarada); ``2`` = argumentos
    invalidos o almacen ilegible, con el motivo por ``stderr``.

    ``--es-bars`` reproduce un gap **sin red** desde barras capturadas; ``--capture-bars`` descarga
    las barras con horas extendidas y las guarda (para reproducir despues).
    """
    parser = argparse.ArgumentParser(
        prog=MODULE, description="Gap de pre-mercado del futuro ES (dato declarado de la decision)"
    )
    parser.add_argument("--data-root", type=Path, default=None, help="raiz del almacen")
    parser.add_argument("--as-of", default=None, help="instante declarado ISO-8601 con zona")
    parser.add_argument("--reports-dir", type=Path, default=None, help="directorio de informes")
    parser.add_argument(
        "--es-bars", type=Path, default=None, help="barras capturadas (JSON): calcula sin red"
    )
    parser.add_argument(
        "--capture-bars",
        type=Path,
        default=None,
        help="descarga las barras con horas extendidas y las guarda aqui",
    )
    parser.add_argument("--settings", type=Path, default=None, help="ruta de settings.yaml")
    args = parser.parse_args(argv)

    try:
        moment = _parse_as_of(cast("str | None", args.as_of))
    except PremarketGapError as error:
        print(f"no se puede medir el gap de pre-mercado: {error}", file=sys.stderr)
        return 2

    try:
        settings = load_settings(args.settings)
    except ConfigurationError as error:
        print(f"no se puede leer la configuracion: {error}", file=sys.stderr)
        return 2
    data_root = Path(args.data_root) if args.data_root is not None else Path(settings.data.root)

    es_bars = cast("Path | None", args.es_bars)
    capture = cast("Path | None", args.capture_bars)
    captured: tuple[PremarketBar, ...] = ()
    if es_bars is not None:
        try:
            captured = load_bars(es_bars)
        except PremarketGapError as error:
            print(f"no se puede leer el fichero de barras: {error}", file=sys.stderr)
            return 2

    def captured_fetcher(_series_id: str, *, as_of: datetime) -> Sequence[PremarketBar]:
        """Devuelve las barras capturadas (el recorte point-in-time lo hace `analyse`)."""
        del as_of
        return captured

    fetcher: BarsFetcher | None = captured_fetcher if es_bars is not None else None
    if capture is not None:
        _, snapshot = snapshot_instant(moment)
        try:
            downloaded = _yfinance_bars(ES_SERIES, as_of=snapshot)
        except Exception as error:
            print(f"no se pueden capturar las barras del ES: {error}", file=sys.stderr)
            return 2
        save_bars(downloaded, capture)
        print(f"barras capturadas: {capture} ({len(downloaded)})")

    try:
        report = analyse(store=Store(data_root), as_of=moment, fetcher=fetcher)
    except PremarketGapError as error:
        print(f"no se puede medir el gap de pre-mercado: {error}", file=sys.stderr)
        return 2

    payload = report.payload()
    reports_dir = cast("Path | None", args.reports_dir)
    if reports_dir is not None:
        json_path, markdown_path = write_report(payload, reports_dir)
        print(f"informe del gap: {json_path} y {markdown_path}")
    print(render_markdown(payload))
    print(f"estado: {payload['state']} — {payload['reason']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de proceso
    sys.exit(main())
