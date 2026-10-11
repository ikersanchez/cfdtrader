"""Ticks gratuitos de Dukascopy (``.bi5``) — la serie de **pre-mercado** del S&P (tarea #153/#150).

**Que entrega.** El precio del S&P 500 en el **instante del snapshot** (08:45 ET, `plan.md` §13)
reconstruido desde los **ticks** que Dukascopy publica gratis de su instrumento ``USA500IDXUSD``. Es
la primera fuente que da `bid`/`ask` de verdad, y su historico arranca a finales de 2013 (medido el
2026-10-11: `_docs/premarket_data_review_2026-10-11.md` §2.1), asi que cubre las ~2 688 sesiones que
el modelo necesita.

**⚠️ Licencia: esto NO es una autorizacion.** Los *Terms of Use* de Dukascopy prohiben usar cualquier
*scraper*, robot o automatismo para acceder a su contenido «*without the prior express written
consent of DUKASCOPY*», y vetar cualquier accion que imponga «*an unreasonable or disproportionately
large load*» (`_docs/data_sources.md`, seccion «Dukascopy»). **El propietario decidio el 2026-10-11
asumir esa decision y seguir por esta via**, con la condicion explicita de **no sobrecargar** el
servicio. Todo lo que sigue existe para eso.

**Contrato de cortesia (la mitigacion de carga, en el codigo y no en un comentario):**

1. **Una peticion por sesion, no veinticuatro.** Solo se pide el fichero de la **hora UTC que
   contiene** las 08:45 ET (`12h` en EDT, `13h` en EST), deducida con ``zoneinfo`` — nunca con un
   offset fijo. Si esa hora no trae tick, se prueba **una** hora antes y se declara.
2. **Intervalo minimo entre peticiones de red** (``min_interval_seconds``, **1 s**; es la **decision
   del propietario** del 2026-10-11). El descargador **duerme** lo que falte antes de cada peticion:
   el backfill completo (~3 000 sesiones) baja de ~25 h a **~50 min**. Es el ritmo mas rapido
   adoptado, y el modulo **no** ofrece concurrencia — subirlo mas es una decision explicita, no un
   accidente de configuracion.
3. **Un solo hilo.** No hay concurrencia: la clase no lanza hilos ni procesos.
4. **Archivo local**: lo descargado se escribe en ``<raw_dir>`` y **nunca se vuelve a pedir**. Una
   re-ejecucion sobre lo ya bajado **no toca la red** (y no duerme: no consume turno).
5. **Tope por ejecucion** (``max_requests``): agotado, el run **para** con error tipado. El backfill
   se retoma en trozos, asi que una ejecucion nunca puede convertirse en un barrido masivo.
6. **Fallo cerrado**: ante un ``429``/``503`` (``SourceRateLimitedError``) el run **aborta**; no
   reintenta en bucle ni sube el ritmo. Es la senal de que hay que ir mas despacio, y se respeta.
7. **`User-Agent` que se identifica**, para que el proveedor sepa quien pide y pueda decir que pare.

**Point-in-time, y lo que NO se hace.** El precio es el **ultimo tick con instante <= 08:45 ET**, y
nunca un tick de la sesion que se decide (*look-ahead*). Sin tick, el dia es una **ausencia
declarada** con su motivo, jamas un ``0``. Esto **no** es una *feature* ni una afirmacion de *edge*:
solo el dato.

Sin red en los tests: el cliente HTTP se **inyecta** (``httpx.MockTransport``) y el sueno tambien
(``sleep``), de modo que el contrato de cortesia se puede comprobar sin esperar ni pedir nada.
"""

from __future__ import annotations

import argparse
import json
import lzma
import struct
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

from cfdtrader.data.calendar import EASTERN
from cfdtrader.data.sources.base import (
    SourceError,
    SourceRateLimitedError,
    SourceUnavailableError,
)
from cfdtrader.data.sources.http import CachedHttpClient

__all__ = [
    "DATA_URL",
    "DEFAULT_MAX_REQUESTS",
    "INSTRUMENT",
    "MIN_INTERVAL_SECONDS",
    "PRICE_SCALE",
    "SNAPSHOT_ET",
    "SOURCE",
    "TICK_BYTES",
    "USER_AGENT",
    "DukascopyBudgetExceededError",
    "DukascopyClient",
    "DukascopyError",
    "Tick",
    "decode_ticks",
    "hour_url",
    "premarket_tick",
    "snapshot_instant",
    "write_bars",
]

#: Nombre de la fuente, tal y como se escribe en la columna ``source``.
SOURCE: Final[str] = "dukascopy"

#: Instrumento: el S&P 500 de Dukascopy, en sesion de 24 h.
INSTRUMENT: Final[str] = "USA500IDXUSD"

#: Base del feed. Se pide **una** URL por sesion (el contrato de cortesia, punto 1).
DATA_URL: Final[str] = "https://datafeed.dukascopy.com/datafeed"

#: Instante del snapshot declarado (`plan.md` §13), en hora de Nueva York.
SNAPSHOT_ET: Final[clock] = clock(8, 45)

#: Cada tick son 20 bytes: ``uint32`` ms, ask, bid, y dos volumenes ``float32``.
TICK_BYTES: Final[int] = 20

#: Los precios del fichero son enteros con 3 decimales: 1919331 -> 1 919,331 (S&P de junio de 2014).
PRICE_SCALE: Final[int] = 1000

#: Intervalo minimo **entre peticiones de red**. **1 s** (decision del propietario, 2026-10-11): el
#: backfill completo (~3 000 sesiones, una peticion cada una) baja de ~25 h a **~50 min**. El limite
#: practico no es el sueno, es el control de tasa del proveedor (medido: corta la conexion tras una
#: rafaga), asi que el modulo **para** en vez de insistir.
MIN_INTERVAL_SECONDS: Final[float] = 1.0

#: Agotado este numero de peticiones, el run para (y se retoma en otra ejecucion). El tope es una
#: guarda contra un bucle desbocado, **no** un troceador: con el turno de 1 s, 4 000 peticiones son
#: ~1 h 07 min, asi que **el backfill entero cabe en una sola pasada**.
DEFAULT_MAX_REQUESTS: Final[int] = 4000

#: El User-Agent se identifica: quien pide, y a que ritmo.
USER_AGENT: Final[str] = (
    f"cfdtrader/0.1 (investigacion personal; una peticion cada {MIN_INTERVAL_SECONDS:.0f}s)"
)

#: Zona horaria de referencia del instrumento (la sesion de EE. UU.).
_EASTERN: Final[ZoneInfo] = EASTERN


class DukascopyError(Exception):
    """Argumentos invalidos o peticion fuera del contrato de cortesia."""


class DukascopyBudgetExceededError(DukascopyError):
    """Se agoto el tope de peticiones de esta ejecucion (punto 5 del contrato)."""


@dataclass(frozen=True, slots=True)
class Tick:
    """Un tick: su instante (UTC) y sus dos precios (ask/bid), ya escalados."""

    instant: datetime
    ask: float
    bid: float


def hour_url(day: date, hour: int, *, instrument: str = INSTRUMENT) -> str:
    """URL del fichero de ticks de **una** hora.

    ⚠️ El **mes va de 0 a 11** en la ruta de Dukascopy: ``/2024/05/03/`` es el **3 de junio** de
    2024, no el 3 de mayo. Se comprueba en los tests porque es la trampa clasica de este feed.
    """
    if not 0 <= hour <= 23:
        raise DukascopyError(f"la hora del fichero tiene que estar entre 0 y 23, llego {hour}")
    return (
        f"{DATA_URL}/{instrument}/{day.year:04d}/{day.month - 1:02d}/{day.day:02d}/"
        f"{hour:02d}h_ticks.bi5"
    )


def decode_ticks(
    payload: bytes,
    *,
    hour_start: datetime,
    price_scale: int = PRICE_SCALE,
) -> tuple[Tick, ...]:
    """Decodifica un ``.bi5`` (LZMA *alone*) en sus ticks, ordenados por instante.

    Un cuerpo **vacio** no es un error: es una hora **sin datos**, y se devuelve la tupla vacia (la
    ausencia se declara aguas arriba, no se rellena con un ``0``). Un cuerpo cuya longitud no es
    multiplo de :data:`TICK_BYTES` si es un error: el formato no es el esperado.
    """
    if not payload:
        return ()
    try:
        raw = lzma.decompress(payload, format=lzma.FORMAT_ALONE)
    except lzma.LZMAError as error:
        raise SourceUnavailableError(
            f"el .bi5 de {hour_start.isoformat()} no se pudo descomprimir: {error}",
            source=SOURCE,
        ) from error
    if len(raw) % TICK_BYTES != 0:
        raise SourceUnavailableError(
            f"el .bi5 de {hour_start.isoformat()} trae {len(raw)} bytes, que no es multiplo de "
            f"{TICK_BYTES}: el formato no es el esperado",
            source=SOURCE,
        )
    ticks: list[Tick] = []
    for offset in range(0, len(raw), TICK_BYTES):
        milliseconds, ask, bid, _ask_volume, _bid_volume = struct.unpack(
            ">IIIff", raw[offset : offset + TICK_BYTES]
        )
        ticks.append(
            Tick(
                instant=hour_start + timedelta(milliseconds=milliseconds),
                ask=ask / price_scale,
                bid=bid / price_scale,
            )
        )
    ticks.sort(key=lambda tick: tick.instant)
    return tuple(ticks)


def snapshot_instant(day: date) -> datetime:
    """El instante del snapshot (08:45 ET) de esa sesion, en UTC (12:45 o 13:45 segun el DST)."""
    return datetime.combine(day, SNAPSHOT_ET, tzinfo=_EASTERN).astimezone(UTC)


def premarket_tick(ticks: Sequence[Tick], snapshot: datetime) -> Tick | None:
    """El **ultimo** tick con instante ``<=`` el snapshot, o ``None`` si no hay ninguno.

    Es el recorte point-in-time: un tick posterior a las 08:45 ET seria *look-ahead* y no se usa.
    """
    found: Tick | None = None
    for tick in ticks:
        if tick.instant <= snapshot and (found is None or tick.instant >= found.instant):
            found = tick
    return found


def _hour_start(day: date, hour: int) -> datetime:
    """El instante UTC en que empieza la hora del fichero (los ticks son milisegundos desde ahi)."""
    return datetime(day.year, day.month, day.day, hour, tzinfo=UTC)


class DukascopyClient:
    """Descargador **de cortesia**: una peticion por sesion, con turno, tope y archivo local.

    El contrato del modulo se implementa aqui, y se puede comprobar sin red ni esperas porque
    ``client`` y ``sleep`` se inyectan.

    Parameters
    ----------
    client:
        Cliente HTTP (``CachedHttpClient``). En los tests, con ``httpx.MockTransport``.
    raw_dir:
        Archivo local de los ``.bi5``. Es lo que hace que una re-ejecucion **no toque la red**.
    min_interval_seconds:
        Turno minimo entre peticiones **de red**. Los aciertos del archivo no consumen turno.
    max_requests:
        Tope de peticiones de red de esta ejecucion (``None`` = sin tope).
    sleep, monotonic:
        Inyectables: los tests comprueban el turno sin esperar.
    """

    def __init__(
        self,
        client: CachedHttpClient,
        *,
        raw_dir: Path | None = None,
        min_interval_seconds: float = MIN_INTERVAL_SECONDS,
        max_requests: int | None = DEFAULT_MAX_REQUESTS,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        user_agent: str = USER_AGENT,
        instrument: str = INSTRUMENT,
    ) -> None:
        if min_interval_seconds < 0:
            raise DukascopyError(
                "el intervalo minimo no puede ser negativo: el ritmo es la mitigacion"
            )
        if max_requests is not None and max_requests < 1:
            raise DukascopyError("el tope de peticiones tiene que ser al menos 1")
        self._client = client
        self._raw_dir = raw_dir
        self._min_interval = float(min_interval_seconds)
        self._max_requests = max_requests
        self._sleep = sleep
        self._monotonic = monotonic
        self._user_agent = user_agent
        self._instrument = instrument
        self._last_request_at: float | None = None
        #: Peticiones **de red** hechas en esta ejecucion (los aciertos del archivo no cuentan).
        self.requests = 0
        #: Horas resueltas desde el archivo local, sin red.
        self.from_archive = 0

    # ── Archivo local (punto 4 del contrato) ─────────────────────────────────
    def _archive_path(self, day: date, hour: int, *, suffix: str = ".bi5") -> Path | None:
        if self._raw_dir is None:
            return None
        return (
            self._raw_dir
            / self._instrument
            / f"{day.year:04d}"
            / f"{day.month:02d}"
            / f"{day.day:02d}"
            / f"{hour:02d}h_ticks{suffix}"
        )

    def _from_archive(self, day: date, hour: int) -> tuple[Tick, ...] | None:
        """Ticks ya archivados (o su marca de ausencia), o ``None`` si hay que pedirlos."""
        stored = self._archive_path(day, hour)
        if stored is not None and stored.is_file():
            self.from_archive += 1
            return decode_ticks(stored.read_bytes(), hour_start=_hour_start(day, hour))
        marker = self._archive_path(day, hour, suffix=".empty")
        if marker is not None and marker.is_file():
            self.from_archive += 1
            return ()
        return None

    def _store(self, day: date, hour: int, payload: bytes) -> None:
        """Archiva el cuerpo (o la **marca de ausencia**, que tambien se cachea)."""
        if self._raw_dir is None:
            return
        target = self._archive_path(day, hour, suffix=".bi5" if payload else ".empty")
        if target is None:  # pragma: no cover - `_raw_dir` ya se comprobo
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    # ── Turno y peticion ─────────────────────────────────────────────────────
    def _wait_turn(self) -> float:
        """Duerme lo que falte para respetar el turno. Devuelve los segundos esperados."""
        elapsed = (
            None if self._last_request_at is None else self._monotonic() - self._last_request_at
        )
        remaining = 0.0 if elapsed is None else self._min_interval - elapsed
        if remaining > 0:
            self._sleep(remaining)
            return remaining
        return 0.0

    def hour_ticks(self, day: date, hour: int, *, now: datetime) -> tuple[Tick, ...]:
        """Los ticks de **una** hora (archivo local, o una peticion de red con su turno).

        Raises
        ------
        DukascopyBudgetExceededError
            Se agoto el tope de peticiones de esta ejecucion (se retoma despues).
        DukascopyError
            La fuente respondio con limite de peticiones: el run **aborta** (fallo cerrado).
        """
        archived = self._from_archive(day, hour)
        if archived is not None:
            return archived

        if self._max_requests is not None and self.requests >= self._max_requests:
            raise DukascopyBudgetExceededError(
                f"tope de {self._max_requests} peticiones de red agotado en esta ejecucion: el run "
                "para aqui y se retoma despues (lo ya bajado esta archivado)"
            )
        url = hour_url(day, hour, instrument=self._instrument)
        self._wait_turn()
        try:
            cached = self._client.get(url, headers={"User-Agent": self._user_agent}, now=now)
        except SourceRateLimitedError as error:
            raise DukascopyError(
                f"{url} respondio con limite de peticiones ({error}): el run se detiene para no "
                "insistir (fallo cerrado)"
            ) from error
        except SourceUnavailableError:
            # Cuerpo vacio: la hora **no tiene datos**. Es una ausencia declarada, no un error.
            self.requests += 1
            self._last_request_at = self._monotonic()
            self._store(day, hour, b"")
            return ()

        payload = cached.response.content
        if cached.from_cache:
            self._last_request_at = None  # no ha habido red: el turno no se ha consumido
        else:
            self.requests += 1
            self._last_request_at = self._monotonic()
            self._store(day, hour, payload)
        return decode_ticks(payload, hour_start=_hour_start(day, hour))

    def session_tick(self, day: date, *, now: datetime) -> tuple[Tick | None, str]:
        """El tick del snapshot (08:45 ET) de esa sesion, y la nota si se fue a la hora anterior.

        Pide **una** URL (la hora UTC que contiene el snapshot); solo si esa hora no trae tick
        prueba **una** hora antes, y lo declara.
        """
        snapshot = snapshot_instant(day)
        hour = snapshot.hour
        tick = premarket_tick(self.hour_ticks(day, hour, now=now), snapshot)
        if tick is not None:
            return tick, ""
        tick = premarket_tick(self.hour_ticks(day, hour - 1, now=now), snapshot)
        if tick is None:
            return None, f"sin ningun tick <= 08:45 ET en las horas {hour - 1}h y {hour}h UTC"
        return tick, f"la hora {hour}h UTC no traia tick: se usa el de la hora anterior"


def write_bars(bars: Iterable[tuple[datetime, float]], path: Path | str) -> Path:
    """Escribe las barras con el **mismo formato** que #141 (`premarket_gap.py::save_bars`).

    Un test de ida y vuelta contra ese modulo mantiene los dos formatos juntos, para que quien
    re-mida el candidato pueda usar el ``--bars-dir`` de #149 **sin conversion**.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"instant": instant.astimezone(UTC).isoformat(), "price": round(price, 6)}
        for instant, price in bars
    ]
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def business_days(start: date, end: date) -> tuple[date, ...]:
    """De lunes a viernes del rango: un festivo saldra como ausencia declarada, no como un ``0``."""
    if end < start:
        raise DukascopyError(f"el rango esta invertido: {start} > {end}")
    days: list[date] = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return tuple(days)


def _parse_date(value: str) -> date:
    """Fecha ISO-8601 del rango (el modulo no lee el reloj)."""
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise DukascopyError(f"fecha no ISO-8601: {value!r}") from error


def _parse_now(value: str) -> datetime:
    """Instante de referencia ISO-8601 **con zona**, obligatorio (A13: nada de ``now()`` dentro)."""
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as error:
        raise DukascopyError(f"`--now` no es ISO-8601: {value!r}") from error
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise DukascopyError(f"`--now` tiene que declarar zona horaria: {value!r}")
    return moment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cfdtrader.data.sources.dukascopy",
        description=(
            "Backfill del tick de pre-mercado del S&P (Dukascopy), **por trozos y con turno**: una "
            "peticion por sesion, un minimo de segundos entre peticiones y un tope por ejecucion."
        ),
    )
    parser.add_argument("--raw-dir", type=Path, required=True, help="archivo local de los .bi5")
    parser.add_argument(
        "--bars-out", type=Path, required=True, help="JSON de barras (formato #141)"
    )
    parser.add_argument("--start", required=True, help="primera sesion (AAAA-MM-DD)")
    parser.add_argument("--end", required=True, help="ultima sesion (AAAA-MM-DD)")
    parser.add_argument("--now", required=True, help="instante de referencia ISO-8601 con zona")
    parser.add_argument(
        "--min-interval",
        type=float,
        default=MIN_INTERVAL_SECONDS,
        help=f"segundos entre peticiones de red (por defecto {MIN_INTERVAL_SECONDS:.0f})",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=DEFAULT_MAX_REQUESTS,
        help=(
            "tope de peticiones de red por ejecucion; 0 = sin tope "
            f"(por defecto {DEFAULT_MAX_REQUESTS})"
        ),
    )
    parser.add_argument("--cache-root", type=Path, default=None, help="raiz de la cache HTTP")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta el backfill de un tramo y deja el resumen en la salida estandar.

    Codigos de salida: ``0`` si completo; ``2`` argumentos invalidos; ``3`` la fuente pidio parar
    (limite de peticiones) o se agoto el tope — en los dos casos **lo bajado queda archivado** y se
    retoma volviendo a ejecutar.
    """
    args = _parser().parse_args(argv)
    try:
        days = business_days(_parse_date(args.start), _parse_date(args.end))
        now = _parse_now(args.now)
        raw_dir = Path(args.raw_dir)
        bars_out = Path(args.bars_out)
    except DukascopyError as error:
        print(f"argumentos invalidos: {error}", file=sys.stderr)
        return 2

    client = DukascopyClient(
        CachedHttpClient(source=SOURCE, cache_root=args.cache_root),
        raw_dir=raw_dir,
        min_interval_seconds=args.min_interval,
        max_requests=None if args.max_requests == 0 else args.max_requests,
    )
    bars: list[tuple[datetime, float]] = []
    absences: list[str] = []
    notes: list[str] = []
    try:
        for day in days:
            tick, note = client.session_tick(day, now=now)
            if tick is None:
                absences.append(f"{day.isoformat()}: {note}")
                continue
            bars.append((tick.instant, tick.bid))
            if note:
                notes.append(f"{day.isoformat()}: {note}")
    except (DukascopyError, SourceError) as error:
        print(f"se detiene el backfill: {error}", file=sys.stderr)
        print(
            f"archivado: {client.requests} peticiones de red, {client.from_archive} horas del "
            "archivo local. Vuelve a ejecutar para retomar donde se quedo.",
            file=sys.stderr,
        )
        return 3

    write_bars(bars, bars_out)
    print(
        json.dumps(
            {
                "sessions_asked": len(days),
                "bars": len(bars),
                "absences": len(absences),
                "notes": len(notes),
                "requests": client.requests,
                "from_archive": client.from_archive,
                "min_interval_seconds": args.min_interval,
                "bars_out": str(bars_out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if absences:
        print("ausencias declaradas: " + "; ".join(absences[:5]), file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada del modulo
    raise SystemExit(main())
