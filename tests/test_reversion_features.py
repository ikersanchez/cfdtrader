"""Features de sobre-extension y reversion (`reversion_v1`, tarea #159): A1-A16.

Blinda lo que el **codigo** declara, no los numeros vivos: la medicion de disponibilidad y el
presupuesto de features viven en ``_docs/reversion_candidates_2026-10-16.md`` y en la issue de
seguimiento. Cubre: el modulo y su nota (A1), el catalogo declarado (A2), la disponibilidad
declarada de las seis (A3), el registro de la familia y el caracter **aditivo** del cambio (A4),
la *term structure* y su redundancia algebraica (A5), el *point-in-time* por los dos lados (A6),
el denominador de `breadth` y su companero `sector_count` (A7), los sectores que aun no existian
(A8), `^VIX9D` ausente (A9), la racha (A10), el percentil expandido contra `vix_percentile` (A11),
el control que no se mueve (A12), el golden (A13), la pureza del modulo (A14), la cobertura
medida contra las series de entrada (A15) y la nota de medicion (A16).
"""

from __future__ import annotations

import ast
import bisect
import json
import math
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import polars as pl
import pytest

from cfdtrader.analysis.feature_frame import COLUMNS_BY_FAMILY, FAMILY_ORDER
from cfdtrader.features import store
from cfdtrader.features.context import context_matrix, context_spec
from cfdtrader.features.reversion import (
    REVERSION_INPUT_COLUMNS,
    reversion_matrix,
    reversion_spec,
)
from cfdtrader.features.technical import technical_matrix, technical_spec
from cfdtrader.features.volatility import VIX_MIN_SESSIONS, vix_features
from cfdtrader.models.baseline import BASELINE_FEATURES

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODULE_PATH: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "features" / "reversion.py"
NOTA_PATH: Final[Path] = REPO_ROOT / "_docs" / "reversion_candidates_2026-10-16.md"
GOLDEN_PATH: Final[Path] = (
    REPO_ROOT / "tests" / "fixtures" / "features" / "matrix_golden_expected.json"
)

#: Las seis columnas de la familia, en el orden del catalogo (contrato de #159).
COLUMNS: Final[tuple[str, ...]] = (
    "vix_ts_slope",
    "vix_contango",
    "vix9d_vix",
    "breadth",
    "streak_up",
    "ret_5_pct",
)

#: El ancla, los tres vertices y los once sectores: la muestra sintetica.
VIX: Final[str] = "^VIX"
VIX9D: Final[str] = "^VIX9D"
VIX3M: Final[str] = "^VIX3M"
ANCHOR: Final[str] = store.REVERSION_ANCHOR_SERIES

#: Sesiones de la muestra corta (ocho sesiones de bolsa).
SESSIONS: Final[tuple[date, ...]] = (
    date(2024, 1, 2),
    date(2024, 1, 3),
    date(2024, 1, 4),
    date(2024, 1, 5),
    date(2024, 1, 8),
    date(2024, 1, 9),
    date(2024, 1, 10),
    date(2024, 1, 11),
)

CLOSE_HOUR_UTC: Final[int] = 21

#: Cierres base de la muestra corta: el ancla sube todas las sesiones (racha creciente), el VIX9D
#: por encima del VIX y el VIX3M por debajo (los dos signos de la *term structure*).
_BASE: Final[dict[str, tuple[float, ...]]] = {
    ANCHOR: (100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0),
    VIX: (20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0),
    VIX9D: (22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0),
    VIX3M: (19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0),
}

#: Tamano del salto de cada sector: el signo del retorno es el del signo declarado.
SIGN_STEP: Final[float] = 0.01


def _input_frame(
    sessions: tuple[date, ...], closes: tuple[float, ...], *, anchor: bool
) -> pl.DataFrame:
    """Frame de entrada: el ancla trae ademas su ``as_of`` (el sello del cierre)."""
    frame = pl.DataFrame(
        {
            "session": pl.Series("session", list(sessions), dtype=pl.Date()),
            "close": pl.Series("close", list(closes), dtype=pl.Float64()),
        }
    )
    if not anchor:
        return frame
    instants = [
        datetime(session.year, session.month, session.day, CLOSE_HOUR_UTC, tzinfo=UTC)
        for session in sessions
    ]
    return frame.with_columns(pl.Series("as_of", instants, dtype=pl.Datetime("us", "UTC")))


def _signs(offset: int) -> tuple[int, ...]:
    """Los siete signos de un sector: sube cuando ``(i + offset) % 3 != 0``."""
    return tuple(1 if (index + offset) % 3 else -1 for index in range(len(SESSIONS) - 1))


def _sector_closes(offset: int, *, base: float) -> tuple[float, ...]:
    """Cierres de un sector a partir de sus signos.

    Cada salto multiplica por ``1 + SIGN_STEP * signo``, de modo que el signo del retorno es
    exactamente el signo declarado (el logaritmo es monotono).
    """
    closes = [base]
    for sign in _signs(offset):
        closes.append(closes[-1] * (1.0 + SIGN_STEP * sign))
    return tuple(closes)


#: `breadth` esperado por fila, calculado **a mano** desde :func:`_signs`: la fila ``i`` lee los
#: retornos de la sesion ``i-1``, de modo que su signo es ``_signs(offset)[i-2]``; con 11 sectores
#: bajan ``#{n : (i-2+n) % 3 == 0}`` (4 o 3). Las dos primeras filas son ``null``: en ``i=1`` la
#: sesion previa es la primera de cada serie y no tiene retorno. Nunca un 0,5 por defecto.
_EXPECTED_BREADTH: Final[tuple[float | None, ...]] = (
    None,
    None,
    7.0 / 11.0,
    8.0 / 11.0,
    7.0 / 11.0,
    7.0 / 11.0,
    8.0 / 11.0,
    7.0 / 11.0,
)

#: Sesiones de la muestra **larga**: hacen falta para que el percentil expandido acumule las 250
#: sesiones de historia que declara el catalogo.
LONG_SESSIONS: Final[int] = 320
LONG_START: Final[date] = date(2024, 1, 2)


def _long_sessions(count: int = LONG_SESSIONS) -> tuple[date, ...]:
    """Sesiones consecutivas de lunes a viernes desde ``LONG_START`` (sin festivos)."""
    sessions: list[date] = []
    current = LONG_START
    while len(sessions) < count:
        if current.weekday() < 5:
            sessions.append(current)
        current = date.fromordinal(current.toordinal() + 1)
    return tuple(sessions)


def _long_closes(count: int, *, phase: float) -> tuple[float, ...]:
    """Cierres deterministas de la muestra larga: una tendencia con una onda declarada."""
    return tuple(
        100.0 + 0.5 * index + 2.0 * math.sin(index / 5.0 + phase) for index in range(count)
    )


def _long_inputs(*, sessions: tuple[date, ...] | None = None) -> dict[str, pl.DataFrame]:
    """El ``Mapping`` de la muestra larga: el ancla, los tres vertices y los once sectores."""
    dates = _long_sessions() if sessions is None else sessions
    count = len(dates)
    frames: dict[str, pl.DataFrame] = {
        ANCHOR: _input_frame(dates, _long_closes(count, phase=0.0), anchor=True),
        VIX: _input_frame(dates, _long_closes(count, phase=0.7), anchor=False),
        VIX9D: _input_frame(dates, _long_closes(count, phase=1.3), anchor=False),
        VIX3M: _input_frame(dates, _long_closes(count, phase=1.9), anchor=False),
    }
    for offset, name in enumerate(store.REVERSION_SECTOR_SERIES):
        frames[name] = _input_frame(dates, _long_closes(count, phase=0.1 * offset), anchor=False)
    return frames


def _sector_series() -> dict[str, tuple[float, ...]]:
    """Los cierres de los once sectores de la muestra corta, con el patron de signos declarado."""
    return {
        name: _sector_closes(offset, base=100.0 + 10.0 * offset)
        for offset, name in enumerate(store.REVERSION_SECTOR_SERIES)
    }


def _sector_frames() -> dict[str, pl.DataFrame]:
    """Los once sectores de la muestra corta, como frames."""
    return {
        name: _input_frame(SESSIONS, closes, anchor=False)
        for name, closes in _sector_series().items()
    }


def _inputs(
    *,
    overrides: dict[str, tuple[float, ...]] | None = None,
    sessions: dict[str, tuple[date, ...]] | None = None,
    drops: tuple[str, ...] = (),
) -> dict[str, pl.DataFrame]:
    """El ``Mapping`` de las quince series de la muestra corta, con lo que se quiera sustituir."""
    closes = {**_BASE, **_sector_series(), **(overrides or {})}
    dates = {ANCHOR: SESSIONS, **(sessions or {})}
    return {
        name: _input_frame(
            dates.get(name, SESSIONS), closes[name], anchor=name == store.REVERSION_ANCHOR_SERIES
        )
        for name in store.REVERSION_SERIES
        if name not in drops
    }


def _matrix(
    *,
    overrides: dict[str, tuple[float, ...]] | None = None,
    sessions: dict[str, tuple[date, ...]] | None = None,
) -> pl.DataFrame:
    """La matriz de la familia con la muestra corta."""
    return reversion_matrix(_inputs(overrides=overrides, sessions=sessions), spec=reversion_spec())


def _long_matrix(frames: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """La matriz de la familia con la muestra larga."""
    return reversion_matrix(_long_inputs() if frames is None else frames, spec=reversion_spec())


def _column(frame: pl.DataFrame, name: str) -> list[float | None]:
    """La columna como lista de ``float``/``None`` (los ``null`` de polars)."""
    values = frame.get_column(name).to_list()
    return [None if value is None else float(value) for value in values]


# ─────────────────────────────────────────────────────────────────────────────
# A1 — el modulo y su nota
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_module_and_its_note_exist() -> None:
    """A1: el modulo vive en el arbol, exporta su API y la nota de medicion existe."""
    assert MODULE_PATH.is_file()
    assert NOTA_PATH.is_file()
    module = __import__("cfdtrader.features.reversion", fromlist=["x"])
    for name in ("REVERSION_INPUT_COLUMNS", "reversion_matrix", "reversion_spec"):
        assert name in module.__all__, name
        assert hasattr(module, name), name


# ─────────────────────────────────────────────────────────────────────────────
# A2 — el catalogo declarado
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_catalog_declares_the_six_candidates() -> None:
    """A2: seis entradas con formula, ventana, fuente, cierre del que dependen y rezago."""
    catalog = store.REVERSION_FEATURE_CATALOG
    assert store.REVERSION_FEATURE_COLUMNS == COLUMNS
    assert tuple(entry.name for entry in catalog) == COLUMNS
    for entry in catalog:
        assert entry.formula, entry.name
        assert entry.source in {"raw.market_daily", "raw.sectors"}, entry.name
        assert entry.required_as_of, entry.name
        assert entry.design_lag == 1, entry.name
    windows = {entry.name: entry.window for entry in catalog}
    assert windows == {
        "vix_ts_slope": 1,
        "vix_contango": 1,
        "vix9d_vix": 1,
        "breadth": 1,
        "streak_up": None,
        "ret_5_pct": store.REVERSION_MIN_SESSIONS,
    }
    assert dict(store.DEFAULT_REVERSION_WINDOWS) == windows
    assert REVERSION_INPUT_COLUMNS == ("session", "close")
    expected_series = (ANCHOR, VIX, VIX9D, VIX3M, *store.REVERSION_SECTOR_SERIES)
    assert expected_series == store.REVERSION_SERIES
    sources = store.DEFAULT_REVERSION_SOURCES
    assert tuple(pair for pair in sources if pair[0] == "raw.sectors") == tuple(
        ("raw.sectors", series_id) for series_id in store.REVERSION_SECTOR_SERIES
    )
    assert ("raw.market_daily", VIX3M) in sources and ("raw.market_daily", VIX9D) in sources


# ─────────────────────────────────────────────────────────────────────────────
# A3 — la disponibilidad declarada
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_every_candidate_declares_lag_one_and_the_close_of_t_minus_one() -> None:
    """A3: las seis declaran `design_lag = 1` y el cierre del que dependen.

    Cinco lo hacen con el cierre de ``t-1``; `ret_5_pct` con el cierre de ``t`` y rezago 1, que es
    **exactamente** lo que declara ``ret_5`` en ``technical_v1``: el valor que entra en la fila de
    diseno de ``t`` se define con el cierre de ``t-1`` en los seis casos.
    """
    catalog = {entry.name: entry for entry in store.REVERSION_FEATURE_CATALOG}
    for name in COLUMNS:
        assert store.DESIGN_LAG_BY_FEATURE[name] == 1, name
        assert catalog[name].design_lag == store.DESIGN_LAG_BY_FEATURE[name], name
    technical = {entry.name: entry for entry in store.TECHNICAL_FEATURE_CATALOG}
    for name in ("vix_ts_slope", "vix_contango", "vix9d_vix", "breadth", "streak_up"):
        assert catalog[name].required_as_of == "cierre de la sesion t-1", name
    assert catalog["ret_5_pct"].required_as_of == technical["ret_5"].required_as_of
    assert store.DESIGN_LAG_BY_FEATURE["ret_5_pct"] == store.DESIGN_LAG_BY_FEATURE["ret_5"]


# ─────────────────────────────────────────────────────────────────────────────
# A4 — el registro de la familia, aditivo
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_the_family_is_registered_appended_and_with_its_own_spec() -> None:
    """A4: la septima familia entra con su `source` y sus columnas van **al final** del catalogo."""
    assert store.REVERSION_FEATURE_SET in store.CATALOG_BY_FEATURE_SET
    assert store.REVERSION_FEATURE_SET in store.SOURCE_BY_FEATURE_SET
    assert (
        store.SOURCE_BY_FEATURE_SET[store.REVERSION_FEATURE_SET] == store.REVERSION_FEATURES_SOURCE
    )
    assert set(store.CATALOG_BY_FEATURE_SET) == set(store.SOURCE_BY_FEATURE_SET)
    assert list(store.CATALOG_BY_FEATURE_SET)[-1] == store.REVERSION_FEATURE_SET
    assert FAMILY_ORDER[-1] == store.REVERSION_FEATURE_SET
    assert COLUMNS_BY_FAMILY[store.REVERSION_FEATURE_SET] == COLUMNS
    # El caracter **aditivo**: las seis nuevas van detras de las que ya estaban, de modo que
    # ninguna columna previa cambia de sitio (y el digest de cualquier prefijo no se mueve).
    assert store.ALL_FEATURE_COLUMNS[-len(COLUMNS) :] == COLUMNS
    assert len(store.ALL_FEATURE_COLUMNS) == len(set(store.ALL_FEATURE_COLUMNS)) == 63
    # Su spec es propia: otro `feature_set`, las ventanas de su catalogo y sus quince fuentes.
    spec = reversion_spec()
    assert spec.feature_set == store.REVERSION_FEATURE_SET
    assert dict(spec.windows) == dict(store.DEFAULT_REVERSION_WINDOWS)
    assert tuple(spec.sources) == tuple(store.DEFAULT_REVERSION_SOURCES)
    with pytest.raises(store.InvalidFeatureMatrixError):
        reversion_matrix(_inputs(), spec=technical_spec())
    # Un `Mapping` que no es el de la familia es un error tipado, con la serie en el mensaje.
    with pytest.raises(store.ReversionInputError):
        reversion_matrix(
            {name: frame for name, frame in _inputs().items() if name != VIX9D},
            spec=reversion_spec(),
        )
    extra = dict(_inputs())
    extra["^SKEW"] = _input_frame(SESSIONS, _BASE[VIX], anchor=False)
    with pytest.raises(store.ReversionInputError):
        reversion_matrix(extra, spec=reversion_spec())


# ─────────────────────────────────────────────────────────────────────────────
# A5 — la term structure y su redundancia algebraica
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_the_term_structure_is_algebraically_dependent() -> None:
    """A5: las tres columnas son las razones de los tres vertices y su producto vale 1."""
    matrix = _matrix()
    slope = _column(matrix, "vix_ts_slope")
    contango = _column(matrix, "vix_contango")
    ratio = _column(matrix, "vix9d_vix")
    # Fila de 2024-01-03: lee el cierre de 2024-01-02 (VIX 20, VIX9D 22, VIX3M 19).
    assert slope[1] == pytest.approx(19.0 / 22.0 - 1.0)
    assert contango[1] == pytest.approx(20.0 / 19.0 - 1.0)
    assert ratio[1] == pytest.approx(22.0 / 20.0 - 1.0)
    # La primera fila tampoco tiene vertices previos: `null` declarado, nunca un cero.
    assert slope[0] is None and contango[0] is None and ratio[0] is None
    for left, middle, right in zip(slope[1:], contango[1:], ratio[1:], strict=True):
        assert left is not None and middle is not None and right is not None
        assert (1.0 + middle) * (1.0 + right) * (1.0 + left) == pytest.approx(1.0, abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# A6 — point-in-time, por los dos lados
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_the_row_of_t_does_not_read_the_close_of_t() -> None:
    """A6: mutar el cierre de los vertices en `t` no mueve la fila de `t`, y si la de `t+1`.

    Los dos lados: la fila de `t` **solo** usa la ultima sesion ``< t`` (el valor de la fila de
    `2024-01-03` es la razon de los cierres de `2024-01-02`, comprobado a mano en A5), y una
    barra posterior al instante no entra en ninguna celda anterior.
    """
    target = 3
    mutated = {name: list(values) for name, values in _BASE.items()}
    # Factores **distintos** por vertice: mutar los tres por el mismo factor dejaria las razones
    # intactas y el test no probaria nada.
    for name, factor in ((VIX, 3.0), (VIX9D, 5.0), (VIX3M, 7.0)):
        mutated[name][target] *= factor
    base = _matrix()
    changed = _matrix(overrides={name: tuple(values) for name, values in mutated.items()})

    for name in ("vix_ts_slope", "vix_contango", "vix9d_vix"):
        assert _column(changed, name)[: target + 1] == _column(base, name)[: target + 1], name
        assert _column(changed, name)[target + 1] != _column(base, name)[target + 1], name
    # La fila de `t` **si** lee las anteriores: el ancla tambien, por la racha.
    anchor = list(_BASE[ANCHOR])
    anchor[target] = 1.0
    streak = _column(_matrix(overrides={ANCHOR: tuple(anchor)}), "streak_up")
    assert streak[: target + 1] == _column(base, "streak_up")[: target + 1]
    assert streak[target + 1] != _column(base, "streak_up")[target + 1]


# ─────────────────────────────────────────────────────────────────────────────
# A7 — el denominador de `breadth` y su companero `sector_count`
# ─────────────────────────────────────────────────────────────────────────────
def _context_frames(sectors: Mapping[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Los 19 frames de ``context_v1``: el ancla y los sectores de la muestra, el resto vacios."""
    frames = {
        name: pl.DataFrame(schema={"session": pl.Date(), "close": pl.Float64()})
        for name in store.CONTEXT_SERIES
    }
    frames[store.CONTEXT_MARKET_SERIES[0]] = _input_frame(SESSIONS, _BASE[ANCHOR], anchor=True)
    frames.update(sectors)
    return frames


def _hand_sector_counts(sectors: Mapping[str, pl.DataFrame]) -> list[int]:
    """Recuento **independiente** de los sectores con retorno computable en ``t-1``, por fila.

    Es la definicion del denominador declarado, escrita otra vez aqui a proposito: el test no
    compara contra un literal del almacen (#135) sino contra las **series de entrada**.
    """
    counts: list[int] = []
    for session in SESSIONS:
        present = 0
        for name in store.REVERSION_SECTOR_SERIES:
            sessions = sectors[name].get_column("session").to_list()
            closes = sectors[name].get_column("close").to_list()
            index = bisect.bisect_left(sessions, session) - 1
            if index >= 1 and closes[index] is not None and closes[index - 1] is not None:
                present += 1
        counts.append(present)
    return counts


def test_a7_breadth_denominator_is_the_companion_sector_count() -> None:
    """A7: el denominador son los sectores **con dato** y coincide con `sector_count`."""
    sectors = _sector_frames()
    matrix = reversion_matrix(_inputs(), spec=reversion_spec())
    context = context_matrix(_context_frames(sectors), spec=context_spec())

    counts = _hand_sector_counts(sectors)
    measured_counts = [int(v) for v in _column(context, "sector_count") if v is not None]
    assert measured_counts == counts
    breadth = _column(matrix, "breadth")
    assert breadth == pytest.approx(_EXPECTED_BREADTH)
    for value, present in zip(breadth, counts, strict=True):
        if present == 0:
            assert value is None, "sin sectores con dato no hay amplitud: `null`, no un 0,5"
            continue
        assert value is not None
        advances = value * present
        assert abs(advances - round(advances)) < 1e-9, "el denominador no es el declarado"


# ─────────────────────────────────────────────────────────────────────────────
# A8 — los sectores que aun no existian
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_a_short_sector_does_not_drag_the_denominator() -> None:
    """A8: `XLRE` nace tarde: antes de su primer cierre no aporta y no se le inventa un cero."""
    late = store.REVERSION_SECTOR_SERIES[-1]
    offset = len(store.REVERSION_SECTOR_SERIES) - 1
    frames = _inputs()
    frames[late] = _input_frame(
        SESSIONS[3:], _sector_closes(offset, base=100.0 + 10.0 * offset)[3:], anchor=False
    )
    matrix = reversion_matrix(frames, spec=reversion_spec())
    sectors = {name: frames[name] for name in store.REVERSION_SECTOR_SERIES}
    counts = _hand_sector_counts(sectors)
    assert counts[:5] == [0, 0, 10, 10, 10] and counts[5:] == [11, 11, 11]
    breadth = _column(matrix, "breadth")
    for value, present in zip(breadth, counts, strict=True):
        assert (value is None) == (present == 0)
        if value is not None:
            assert value * present == pytest.approx(round(value * present), abs=1e-9)
    # Y sin ningun sector con dato la columna es `null`, no `0,5` ni un cero.
    empty = {
        name: pl.DataFrame(schema={"session": pl.Date(), "close": pl.Float64()})
        for name in store.REVERSION_SECTOR_SERIES
    }
    without = reversion_matrix({**_inputs(), **empty}, spec=reversion_spec())
    assert _column(without, "breadth") == [None] * len(SESSIONS)


# ─────────────────────────────────────────────────────────────────────────────
# A9 — un vertice que no existe
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_a_vertex_that_does_not_exist_is_null_and_is_not_filled() -> None:
    """A9: `^VIX9D` antes del 2011-01-01 no existe: sus dos candidatas salen `null`, sin relleno.

    Es la misma regla que el ``min_start`` de #58. La muestra emula el arranque tardio del vertice
    (empieza en la cuarta sesion, como el 2011-01-03 real) y comprueba que las columnas que **no**
    dependen de el siguen publicandose: `vix_contango` solo necesita ``^VIX`` y ``^VIX3M``.
    """
    frames = _inputs()
    frames[VIX9D] = _input_frame(SESSIONS[3:], _BASE[VIX9D][3:], anchor=False)
    matrix = reversion_matrix(frames, spec=reversion_spec())
    for name in ("vix_ts_slope", "vix9d_vix"):
        values = _column(matrix, name)
        assert values[:4] == [None, None, None, None], name
        assert all(value is not None for value in values[4:]), name
    assert all(value is not None for value in _column(matrix, "vix_contango")[1:])
    # El vertice ausente **no** se rellena con otro: sin `^VIX3M` caen las otras dos.
    frames = _inputs()
    frames[VIX3M] = pl.DataFrame(schema={"session": pl.Date(), "close": pl.Float64()})
    without = reversion_matrix(frames, spec=reversion_spec())
    assert _column(without, "vix_ts_slope") == [None] * len(SESSIONS)
    assert _column(without, "vix_contango") == [None] * len(SESSIONS)
    assert all(value is not None for value in _column(without, "vix9d_vix")[1:])


# ─────────────────────────────────────────────────────────────────────────────
# A10 — la racha
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_streak_up_counts_consecutive_up_closes_until_t_minus_one() -> None:
    """A10: la racha son las sesiones seguidas con ``C_s > C_{s-1}`` **hasta ``t-1``**."""
    base = _column(_matrix(), "streak_up")
    # El ancla sube todas las sesiones: la racha de la fila `t` es `t-1`.
    assert base == [None, *[float(index) for index in range(len(SESSIONS) - 1)]]
    # Con el ancla alternando, la racha se corta y vuelve a empezar: `0` donde no sube.
    alternate = (100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0)
    matrix = _matrix(overrides={ANCHOR: alternate})
    assert _column(matrix, "streak_up") == [None, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    # Un hueco en los cierres rompe la racha: no se arrastra el estado.
    holed = pl.when(pl.col("session") == SESSIONS[3]).then(None).otherwise(pl.col("close"))
    frames = _inputs()
    frames[ANCHOR] = _input_frame(SESSIONS, tuple(_BASE[ANCHOR]), anchor=True).with_columns(
        holed.alias("close")
    )
    assert _column(reversion_matrix(frames, spec=reversion_spec()), "streak_up")[4] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# A11 — el percentil del retorno de cinco sesiones
# ─────────────────────────────────────────────────────────────────────────────
def _hand_percentile(values: list[float | None], *, min_sessions: int) -> list[float | None]:
    """Percentil expandido, escrito otra vez aqui: fraccion de los anteriores que no superan."""
    out: list[float | None] = [None] * len(values)
    history: list[float] = []
    for index, value in enumerate(values):
        if value is None:
            continue
        if len(history) >= min_sessions:
            out[index] = sum(1.0 for item in history if item <= value) / len(history)
        history.append(value)
    return out


def test_a11_ret_5_pct_follows_the_vix_percentile_convention() -> None:
    """A11: la candidata es el percentil de un `ret_5` que ya publica `technical_v1`.

    La base se cruza contra la columna real de ``technical_v1`` y el percentil contra la
    convencion de ``vix_percentile`` (#7), las dos veces contra una implementacion escrita en el
    test: no se compara contra un numero del almacen vivo (#135).
    """
    frames = _long_inputs()
    closes = frames[ANCHOR].get_column("close")
    bars = (
        frames[ANCHOR]
        .select("session", "close")
        .with_columns(closes.alias("high"), closes.alias("low"))
    )
    technical = technical_matrix(bars, spec=technical_spec())
    base = _column(technical, "ret_5")
    expected = _hand_percentile(base, min_sessions=store.REVERSION_MIN_SESSIONS)
    measured = _column(_long_matrix(frames), "ret_5_pct")

    warmup = store.REVERSION_RETURN_WINDOW + store.REVERSION_MIN_SESSIONS
    assert measured[:warmup] == [None] * warmup, "sin 250 valores previos no hay percentil"
    assert any(value is not None for value in measured)
    for position in range(warmup, len(measured)):
        assert measured[position] == pytest.approx(expected[position]), position

    # La convencion es la de `vix_percentile`: la fila `t` de #7 publica el percentil del valor de
    # `t-1` entre los anteriores, con el minimo de sesiones que declara esa familia.
    vix_frame = frames[ANCHOR].select("session", pl.col("close").alias("vix_close"))
    percentiles = _column(vix_features(vix_frame), "vix_percentile")
    hand = _hand_percentile(
        [float(value) for value in closes.to_list()], min_sessions=VIX_MIN_SESSIONS
    )
    assert VIX_MIN_SESSIONS == store.REVERSION_MIN_SESSIONS, (
        "las dos familias declaran el mismo minimo de ventana expandida"
    )
    assert any(value is not None for value in percentiles)
    for position in range(1, len(percentiles)):
        assert percentiles[position] == pytest.approx(hand[position - 1]), position


# ─────────────────────────────────────────────────────────────────────────────
# A12 — el control no se mueve
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_the_control_does_not_move() -> None:
    """A12: las diez de produccion y su disponibilidad declarada son las de siempre.

    Los dos literales son **valores estables** (el contrato del modelo y la tabla de
    disponibilidad), no recuentos del almacen: la lista es la declaracion que #24 congelo.
    """
    assert BASELINE_FEATURES == (
        "har_forecast",
        "vix_zscore",
        "dist_sma_20_z",
        "asia_overnight_1",
        "europe_prev_1",
        "dxy_ret_1",
        "sector_dispersion_1",
        "ust_10y_chg_5",
        "garch_forecast_z",
        "is_es_roll_session",
    )
    assert {name: store.DESIGN_LAG_BY_FEATURE[name] for name in BASELINE_FEATURES} == {
        "har_forecast": 0,
        "vix_zscore": 0,
        "dist_sma_20_z": 1,
        "asia_overnight_1": 0,
        "europe_prev_1": 0,
        "dxy_ret_1": 0,
        "sector_dispersion_1": 0,
        "ust_10y_chg_5": 1,
        "garch_forecast_z": 0,
        "is_es_roll_session": 0,
    }
    assert not set(COLUMNS) & set(BASELINE_FEATURES)
    assert FAMILY_ORDER == (
        store.VOLATILITY_FEATURE_SET,
        store.TECHNICAL_FEATURE_SET,
        store.CONTEXT_FEATURE_SET,
        store.MACRO_FEATURE_SET,
        store.REGIME_FEATURE_SET,
        store.COMMODITIES_FEATURE_SET,
        store.REVERSION_FEATURE_SET,
    )


# ─────────────────────────────────────────────────────────────────────────────
# A13 — el golden
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_the_golden_freezes_the_family_and_keeps_the_previous_digests() -> None:
    """A13: el golden de la matriz congela las seis columnas nuevas.

    Sin tocar ningun fixture comprometido: el golden se re-congelo **de forma aditiva** (las seis
    familias previas conservan su digest y sus recuentos de no nulos), que es lo que declara
    ``tests/test_integrity.py``.
    """
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert golden["code_version"] == store.FEATURE_CODE_VERSION
    assert golden["feature_columns"] == ["session", *store.ALL_FEATURE_COLUMNS]
    assert set(golden["feature_spec_sha256"]) == set(FAMILY_ORDER)
    assert golden["feature_spec_sha256"][store.REVERSION_FEATURE_SET] == (
        store.FEATURE_VERSION_PREFIX + store.feature_spec_sha256(reversion_spec())
    )
    for name in COLUMNS:
        assert golden["non_null"][name] > 0, f"'{name}' esta congelada como enteramente nula"


# ─────────────────────────────────────────────────────────────────────────────
# A14 — el modulo es puro
# ─────────────────────────────────────────────────────────────────────────────
def test_a14_the_module_does_not_read_the_clock_the_network_or_write_files() -> None:
    """A14: sin reloj, sin red, sin sistema de ficheros y sin dependencias nuevas."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in ("datetime.now", "utcnow", "time.time", "os.environ", "open("):
        assert forbidden not in source, forbidden
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])
    forbidden_modules = {"requests", "httpx", "yfinance", "urllib", "socket", "os", "pathlib"}
    assert not (forbidden_modules & imported), sorted(imported)
    assert imported <= {
        "__future__",
        "bisect",
        "math",
        "collections",
        "dataclasses",
        "datetime",
        "typing",
        "polars",
        "cfdtrader",
    }, sorted(imported)


# ─────────────────────────────────────────────────────────────────────────────
# A15 — la cobertura, contra las series de entrada
# ─────────────────────────────────────────────────────────────────────────────
def test_a15_coverage_is_relative_to_the_input_series() -> None:
    """A15: la cobertura se afirma contra las series de entrada, nunca contra un literal (#135)."""
    frames = _long_inputs()
    matrix = _long_matrix(frames)
    height = matrix.height
    assert height == len(frames[ANCHOR].get_column("session").to_list())
    non_null = {name: height - matrix.get_column(name).null_count() for name in COLUMNS}

    # Los tres vertices y el ancla empiezan en la primera sesion: la unica fila sin valor es la
    # cero, que no tiene `t-1`. No hay huecos de dato.
    for name in ("vix_ts_slope", "vix_contango", "vix9d_vix", "streak_up"):
        assert non_null[name] == height - 1, name
    # `breadth` necesita ademas un retorno sectorial previo: le falta la segunda fila.
    assert non_null["breadth"] == height - 2
    # `ret_5_pct`: el `ret_5` nace a las 5 sesiones y el percentil necesita 250 valores anteriores.
    assert non_null["ret_5_pct"] == height - (
        store.REVERSION_RETURN_WINDOW + store.REVERSION_MIN_SESSIONS
    )

    # Y el recuento sale de la serie de entrada: si `^VIX9D` empezara en la sesion 40 (el caso real
    # de antes del 2011-01-03) el hueco es exactamente el declarado, sin imputar nada.
    sessions = _long_sessions()
    offset = 40
    late = dict(frames)
    late[VIX9D] = _input_frame(
        sessions[offset:], _long_closes(len(sessions), phase=1.3)[offset:], anchor=False
    )
    late_matrix = _long_matrix(late)
    assert late_matrix.height - late_matrix.get_column("vix_ts_slope").null_count() == (
        height - 1 - offset
    )


# ─────────────────────────────────────────────────────────────────────────────
# A16 — la nota de medicion
# ─────────────────────────────────────────────────────────────────────────────
def test_a16_the_note_declares_the_availability_the_formulas_and_the_fence() -> None:
    """A16: la nota publica la disponibilidad medida, las formulas, la redundancia y la valla."""
    text = NOTA_PATH.read_text(encoding="utf-8")
    for needle in (
        "#159",
        "reversion_v1",
        "vix_ts_slope",
        "vix_contango",
        "vix9d_vix",
        "breadth",
        "streak_up",
        "ret_5_pct",
        "^VIX9D",
        "^VIX3M",
        "sector_count",
        "redundancia",
        "design_lag",
        "percentil",
        "se admite",
        "BASELINE_FEATURES",
        "phase2_ready",
        "carril B",
        "§11.6",
    ):
        assert needle in text, needle


# ─────────────────────────────────────────────────────────────────────────────
# A17 — el contrato de entrada y los cierres que no son cierres
# ─────────────────────────────────────────────────────────────────────────────
def test_a17_the_input_contract_is_typed_and_a_non_positive_close_is_null() -> None:
    """A17: cada campo del contrato tiene su error tipado y un cierre no positivo sale ``null``."""
    base = _inputs()
    cases: tuple[tuple[dict[str, pl.DataFrame], str], ...] = (
        ({**base, VIX: base[VIX].drop("session")}, "session"),
        ({**base, VIX: base[VIX].drop("close")}, "close"),
        (
            {**base, VIX: base[VIX].with_columns(pl.Series("session", [1] * len(SESSIONS)))},
            "no es una fecha",
        ),
        ({**base, VIX: pl.concat([base[VIX], base[VIX]])}, "repite la sesion"),
        (
            {**base, VIX: base[VIX].with_columns(pl.lit("x", dtype=pl.String).alias("close"))},
            "numerica",
        ),
        ({**base, ANCHOR: base[ANCHOR].drop("as_of")}, "as_of"),
        (
            {
                **base,
                ANCHOR: base[ANCHOR].with_columns(
                    pl.Series(
                        "as_of",
                        [datetime(s.year, s.month, s.day, CLOSE_HOUR_UTC) for s in SESSIONS],
                        dtype=pl.Datetime("us"),
                    )
                ),
            },
            "zona horaria",
        ),
        (
            {
                **base,
                ANCHOR: base[ANCHOR].with_columns(
                    pl.Series(
                        "as_of",
                        [
                            datetime(s.year, s.month, s.day, CLOSE_HOUR_UTC, tzinfo=UTC)
                            + timedelta(days=1)
                            for s in SESSIONS
                        ],
                        dtype=pl.Datetime("us", "UTC"),
                    )
                ),
            },
            "no corresponden al mismo dia",
        ),
    )
    for frames, needle in cases:
        with pytest.raises(store.ReversionInputError, match=needle):
            reversion_matrix(frames, spec=reversion_spec())

    # Un cierre no positivo no es un cierre: la razon de esa fila sale `null`, no un cero ni un
    # `inf` por dividir entre cero.
    zeroed = {
        name: (
            base[name].with_columns(
                pl.when(pl.col("session") == SESSIONS[0])
                .then(0.0)
                .otherwise(pl.col("close"))
                .alias("close")
            )
            if name == VIX3M
            else base[name]
        )
        for name in store.REVERSION_SERIES
    }
    matrix = reversion_matrix(zeroed, spec=reversion_spec())
    assert _column(matrix, "vix_ts_slope")[1] is None
    assert _column(matrix, "vix_contango")[1] is None
    assert _column(matrix, "vix_ts_slope")[2] is not None

    # Un hueco en la muestra larga no rompe el percentil: se salta el valor y no lo imputa.
    long_frames = _long_inputs()
    long_sessions = _long_sessions()
    holed = long_frames[ANCHOR].with_columns(
        pl.when(pl.col("session") == long_sessions[100])
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    holed_matrix = _long_matrix({**long_frames, ANCHOR: holed})
    assert holed_matrix.height == long_frames[ANCHOR].height
    assert _column(holed_matrix, "ret_5_pct")[-1] is not None
