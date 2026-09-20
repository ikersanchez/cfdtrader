"""Tests del adaptador de features y del corrimiento de diseno (#24): A2 y A3.

Los dos criterios se miden sobre el **almacen real** (en solo lectura: la fixture de sesion de
``tests/conftest.py`` huella el ``data/`` del repositorio antes y despues). Si el almacen no
esta en el arbol, los tests se **saltan con un motivo declarado**, como en #18/#69.

Todo lo que se escribe va a ``tmp_path``: aqui no se persiste ninguna feature (#73).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final, cast

import polars as pl
import pytest

from cfdtrader.analysis import baseline_report, feature_frame
from cfdtrader.analysis.backtest_report import Universe, build_inputs, load_history
from cfdtrader.analysis.feature_frame import FeatureFrame, FeatureFrameError, build_feature_frame
from cfdtrader.data.calendar import load_calendar
from cfdtrader.data.store import Store
from cfdtrader.features import store as feature_store
from cfdtrader.features.volatility import add_features
from cfdtrader.models.baseline import (
    BASELINE_FEATURES,
    DESIGN_LAG_SESSIONS,
    DESIGN_SESSION_COLUMN,
    SplitAssignment,
    design_frame,
    fit_baseline,
    probabilities,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"

#: El ancla del estudio y el proxy de volatilidad, las dos series que exige ``volatility_v1``.
ANCHOR: Final[str] = "^GSPC"
VIX: Final[str] = "^VIX"

#: Anios que el calendario tiene que materializar para el universo real (A3).
CALENDAR_YEARS: Final[tuple[int, ...]] = tuple(range(2005, 2027))

needs_store = pytest.mark.skipif(
    not (REAL_DATA / "derived" / "labels").exists(),
    reason="el almacen real no esta en el arbol: A2/A3 se miden sobre el (en solo lectura)",
)


@pytest.fixture(scope="session")
def real_frame() -> FeatureFrame:
    """La matriz + el diseno reales, construidos **una vez** para toda la sesion."""
    return build_feature_frame(Store(REAL_DATA))


@pytest.fixture(scope="session")
def real_universe() -> Universe:
    """El universo del arnes (#69) sobre el mismo almacen, para compararlo con el diseno (A3)."""
    history = load_history(Store(REAL_DATA))
    return build_inputs(history, calendar=load_calendar(years=CALENDAR_YEARS))


def _ohlc(store: Store, series_id: str) -> pl.DataFrame:
    """OHLC diario de esa serie con su ``session`` ET derivada de ``as_of``."""
    frame = store.sql(
        "SELECT as_of, open, high, low, close FROM raw.market_daily "  # noqa: S608
        f"WHERE series_id = '{series_id}' ORDER BY as_of"
    )
    return frame.with_columns(
        pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("session")
    )


def _volatility_inputs(store: Store) -> pl.DataFrame:
    """La entrada de ``volatility_v1``: el OHLC del ancla mas ``vix_close``."""
    anchor = _ohlc(store, ANCHOR).select("session", "open", "high", "low", "close")
    vix = _ohlc(store, VIX).select("session", pl.col("close").alias("vix_close"))
    return anchor.join(vix, on="session", how="left")


def _prices_doubled(frame: pl.DataFrame, *, session: date, factor: float) -> pl.DataFrame:
    """Multiplica el OHLC de **una** sesion y deja el resto igual: la mutacion de A2."""
    return frame.with_columns(
        [
            pl.when(pl.col("session") == session)
            .then(pl.col(name) * factor)
            .otherwise(pl.col(name))
            .alias(name)
            for name in ("open", "high", "low", "close")
        ]
    )


def _shock(frame: pl.DataFrame, *, session: date, factor: float) -> pl.DataFrame:
    """Multiplica y desplaza **las 52 columnas** de esa sesion del frame de features."""
    return frame.with_columns(
        [
            pl.when(pl.col("session") == session)
            .then(pl.col(name) * factor + 0.5)
            .otherwise(pl.col(name))
            .alias(name)
            for name in BASELINE_FEATURES
        ]
    )


def _row(frame: pl.DataFrame, session: date) -> dict[str, object]:
    """La fila de esa sesion, como diccionario, para comparar por igualdad."""
    return dict(frame.filter(pl.col("session") == session).row(0, named=True))


def _design_values(frame: pl.DataFrame, session: date) -> dict[str, float]:
    """Las 10 features de la fila de diseno de esa sesion."""
    row = _row(frame, session)
    return {name: float(cast("float", row[name])) for name in BASELINE_FEATURES}


# ─────────────────────────────────────────────────────────────────────────────
# A2 - disponibilidad temporal y no-look-ahead
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a2_the_design_row_of_t_is_the_feature_row_of_t_minus_one(real_frame: FeatureFrame) -> None:
    """A2: una sola regla, sin filtrar columnas, y el corrimiento no pierde sesiones.

    Se comprueba sobre las **2.687** filas, columna a columna: la fila de diseno de cada sesion
    del universo es la fila de features de la sesion anterior del diario, y
    ``n_shifted_rows == 0`` porque la anterior a 2016-01-07 esta en el diario.
    """
    assert DESIGN_LAG_SESSIONS == 1
    assert real_frame.design_lag_sessions == 1
    assert real_frame.design.n_shifted_rows == 0
    assert real_frame.design.n_nulls_in_features == 0

    matrix = real_frame.matrix.frame
    design = real_frame.design.frame
    position = {session: index for index, session in enumerate(matrix["session"].to_list())}
    lookup = matrix.select(pl.col("session").alias("origin"), *BASELINE_FEATURES)
    expected = design.select("session", DESIGN_SESSION_COLUMN).join(
        lookup, left_on=DESIGN_SESSION_COLUMN, right_on="origin", how="left"
    )
    assert expected.height == design.height
    for name in BASELINE_FEATURES:
        assert expected.get_column(name).equals(design.get_column(name)), (
            f"la columna '{name}' del diseno no es la de la sesion de origen: el corrimiento de "
            "A2 esta roto"
        )
    for session, origin in zip(
        cast("list[date]", design["session"].to_list()),
        cast("list[date]", design[DESIGN_SESSION_COLUMN].to_list()),
        strict=True,
    ):
        assert position[session] - position[origin] == 1, (
            f"la sesion {origin} no es la **anterior del diario** a {session}: el corrimiento "
            "usa la lista de etiquetas y no la del diario (A2)"
        )


@needs_store
def test_a2_mutating_the_session_does_not_move_its_design_row_or_its_probability(
    real_frame: FeatureFrame,
) -> None:
    """A2: mutar `close`/`high`/`low`/`ret_long` de `t` no mueve el diseno, ni `p(t)`, ni la
    decision.

    El control positivo es mutar la sesion **anterior**: eso si mueve el diseno y la
    probabilidad. El modelo se ajusta **una vez**, con `t` en el *test* de su fold, de forma que
    la mutacion tampoco pueda propagarse por el ajuste: lo unico que puede mover `p(t)` es la
    fila de diseno de `t`.
    """
    design = real_frame.design.frame
    sessions = real_frame.design.sessions
    model = fit_baseline(
        real_frame.design,
        splits=(
            SplitAssignment(
                index=0,
                train=tuple(range(0, 1000)),
                test=tuple(range(1000, real_frame.design.n_sessions)),
            ),
        ),
    )
    position = 2000
    session = sessions[position]
    previous = sessions[position - 1]
    reference = list(probabilities(model, design))
    assert reference[position] is not None

    labels = real_frame.labels.with_columns((-pl.col("ret_long") - 0.5).alias("ret_long"))
    mutated = _design_with(_shock(real_frame.matrix.frame, session=session, factor=1.9), labels)
    assert mutated.get_column("y").to_list()[position] != design.get_column("y").to_list()[position]
    assert _design_values(mutated, session) == _design_values(design, session)
    assert probabilities(model, mutated)[position] == reference[position]

    moved = _design_with(_shock(real_frame.matrix.frame, session=previous, factor=1.9), labels)
    assert _design_values(moved, session) != _design_values(design, session)
    assert probabilities(model, moved)[position] != reference[position]


def _design_with(matrix: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Reconstruye la matriz de diseno con la regla del modulo (importada, no copiada)."""
    return design_frame(matrix, labels=labels).frame


@needs_store
def test_a2_a_mutated_close_moves_its_own_feature_row_and_not_the_previous_one() -> None:
    """A2 (por la via de los precios): el cierre de `t` mueve la fila de `t`, no la de `t-1`.

    Es el eslabon que cierra el argumento: la mutacion se hace sobre el **precio** y la feature
    se **recalcula** con la funcion de #7, asi que ``design(t) = features(t-1)`` no puede
    depender del cierre de `t` mientras `features(t-1)` sea causal.
    """
    store = Store(REAL_DATA)
    inputs = _volatility_inputs(store)
    sessions = cast("list[date]", inputs.get_column("session").to_list())
    target = sessions[-40]
    previous = sessions[-41]

    before = add_features(inputs)
    after = add_features(_prices_doubled(inputs, session=target, factor=3.0))

    assert _row(before, previous) == _row(after, previous), (
        "mutar el cierre de `t` ha movido la fila de features de `t-1`: hay look-ahead"
    )
    assert _row(before, target) != _row(after, target), (
        "mutar el cierre de `t` no mueve la fila de features de `t`: la mutacion no es efectiva"
    )


# ─────────────────────────────────────────────────────────────────────────────
# A3 - universo y objetivo
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_a3_the_universe_is_the_labelled_sample_and_the_objective_is_ret_long_positive(
    real_frame: FeatureFrame, real_universe: Universe
) -> None:
    """A3: 2.687 sesiones sin nulos, de 2016-01-07 a 2026-09-16, 1.399 positivos.

    Y las sesiones son, **una a una y en orden**, las de ``Universe.inputs`` de #69: el
    adaptador de features y el del arnes tienen que hablar del mismo universo.
    """
    published = baseline_report._universe_block(  # pyright: ignore[reportPrivateUsage]
        real_universe, real_frame
    )
    assert published["n_sessions"] == 2687
    assert published["first_session"] == "2016-01-07"
    assert published["last_session"] == "2026-09-16"
    assert published["n_nulls_in_features"] == 0

    assert real_frame.n_design_rows == len(real_universe.inputs)
    assert real_frame.design.sessions == tuple(item.session for item in real_universe.inputs)
    assert real_frame.n_positives == 1399
    assert real_frame.n_positives / real_frame.n_design_rows == pytest.approx(0.5207, abs=1e-4)
    assert real_frame.n_half_days == 21

    holds = (
        real_frame.design.frame.select(((pl.col("ret_long") > 0) == (pl.col("y") == 1)).all())
        .to_series()
        .item()
    )
    assert holds is True


# ─────────────────────────────────────────────────────────────────────────────
# Bordes del adaptador (apoyo, no un criterio)
# ─────────────────────────────────────────────────────────────────────────────
def test_a_store_without_the_diary_or_the_labels_is_a_typed_error(tmp_path: Path) -> None:
    """Un almacen sin diario ni etiquetas es error tipado, nunca un frame vacio."""
    empty = Store(tmp_path / "vacio")
    with pytest.raises(FeatureFrameError) as error:
        build_feature_frame(empty)
    assert "market_daily" in str(error.value)

    with pytest.raises(FeatureFrameError):
        feature_frame.load_labels(empty)
    with pytest.raises(FeatureFrameError):
        feature_frame.load_labels(empty, series_id="'; DROP TABLE x")


def test_the_two_family_guards_are_reachable_typed_errors() -> None:
    """Las dos guardas de familia (columnas y spec) son alcanzables y tipadas, no codigo muerto.

    Cubrirlas es lo que permite que el unico ``# pragma: no cover`` de los tres modulos de #24
    sea el guard ``__main__`` de la CLI (A15).
    """
    with pytest.raises(FeatureFrameError) as columns:
        feature_frame._family_columns("no_existe")  # pyright: ignore[reportPrivateUsage]
    assert "no_existe" in str(columns.value)

    with pytest.raises(FeatureFrameError) as spec:
        feature_frame._spec_for("no_existe")  # pyright: ignore[reportPrivateUsage]
    assert "no_existe" in str(spec.value)


def _market_records(
    days: Sequence[tuple[int, int, int]], *, series_id: str
) -> list[dict[str, object]]:
    """Filas diarias sinteticas de esa serie, con un ``open`` que no repite el cierre previo."""
    records: list[dict[str, object]] = []
    close = 100.0
    for index, (year, month, day) in enumerate(days):
        open_px = close * 1.0009
        close = open_px * 1.0004
        records.append(
            {
                "source": "yfinance",
                "series_id": series_id,
                "as_of": datetime(year, month, day, 21, tzinfo=UTC),
                "fetched_at": datetime(2026, 9, 20, tzinfo=UTC),
                "published_at": None,
                "open": open_px,
                "high": max(open_px, close) * 1.001,
                "low": min(open_px, close) * 0.999,
                "close": close,
                "volume": 1_000.0 + index,
            }
        )
    return records


def _business_days(count: int) -> list[tuple[int, int, int]]:
    """Dias laborables consecutivos desde 2026-01-05."""
    out: list[tuple[int, int, int]] = []
    current = date(2026, 1, 5)
    while len(out) < count:
        if current.weekday() < 5:
            out.append((current.year, current.month, current.day))
        current += timedelta(days=1)
    return out


def test_a_series_without_history_is_declared_not_invented(tmp_path: Path) -> None:
    """Sin ``raw.macro`` ni ``raw.sectors`` las features salen nulas y se **declaran** (#21)."""
    store = Store(tmp_path / "sin_macro")
    days = _business_days(40)
    store.append(
        "raw",
        "market_daily",
        [
            *_market_records(days, series_id=ANCHOR),
            *_market_records(days, series_id=VIX),
        ],
    )
    store.append(
        "derived",
        "labels",
        [
            {
                "source": "cfdtrader.models.labels",
                "series_id": ANCHOR,
                "as_of": datetime(year, month, day, 21, tzinfo=UTC),
                "fetched_at": datetime(2026, 9, 20, tzinfo=UTC),
                "published_at": None,
                "session": date(year, month, day),
                "ret_long": 0.001,
                "is_half_day": False,
                "k_sigma": 1.0,
            }
            for year, month, day in days
        ],
    )
    matrix = build_feature_frame(store).matrix
    missing = set(matrix.missing_series)
    assert set(feature_store.CONTEXT_SECTOR_SERIES) <= missing
    assert set(feature_store.MACRO_SERIES) <= missing
    assert len(matrix.missing_series) == 23
    assert matrix.n_sessions == 40
    assert matrix.frame.get_column("ust_10y_chg_5").null_count() == 40
    assert matrix.duplicated_columns == ("atr_norm",)
    assert matrix.n_columns == 52
