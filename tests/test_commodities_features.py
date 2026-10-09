"""Familia de features de commodities y FX (`commodities_v1`, tarea #143).

Cubre los quince criterios del *grooming*: el catalogo y el registro (A1, A4, A5), las
redundancias declaradas que sostienen la decision (A2, A3), el *point-in-time* (A6), el
calendario propio de cada serie (A7), el ``null`` en vez de ``0``/``NaN`` (A8), el contrato de
entrada (A9), la pureza del modulo (A10), el cableado al frame (A11), el contrato del almacen
(A12), la produccion intacta (A13) y la identidad de la familia (A14).

La medicion que decide **que** series entran vive en ``_docs/commodity_fx_features_2026-10-09.md``
y no se repite aqui: aqui se blinda lo que el **codigo** declara.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import polars as pl
import pytest

from cfdtrader.analysis import feature_frame
from cfdtrader.data import contracts
from cfdtrader.features import store
from cfdtrader.features.commodities import (
    COMMODITIES_INPUT_COLUMNS,
    COMMODITIES_LAG_SESSIONS,
    commodities_matrix,
    commodities_spec,
)
from cfdtrader.models.baseline import BASELINE_FEATURES

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODULE_PATH: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "features" / "commodities.py"
DECISION_PATH: Final[Path] = REPO_ROOT / "_docs" / "commodity_fx_features_2026-10-09.md"

#: Sesiones de la muestra sintetica (semana y media de dias de bolsa).
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


#: Cierres de la muestra por defecto: crecientes, sin ceros ni nulos.
_BASE: Final[dict[str, tuple[float, ...]]] = {
    "^GSPC": (100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0),
    "CL=F": (70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0),
    "GC=F": (2000.0, 2010.0, 2020.0, 2030.0, 2040.0, 2050.0, 2060.0, 2070.0),
    "EURUSD=X": (1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17),
}

#: Sesiones del ancla, todas. Las series admitidas pueden traer menos (huecos).
_ANCHOR: Final[str] = "^GSPC"


def _inputs(
    *, sessions: dict[str, tuple[date, ...]] | None = None, **overrides: tuple[float, ...]
) -> dict[str, pl.DataFrame]:
    """El ``Mapping`` de las cuatro series, con las que se quieran sustituir."""
    closes = {**_BASE, **overrides}
    dates = {"^GSPC": SESSIONS, **(sessions or {})}
    return {
        name: _input_frame(
            dates.get(name, SESSIONS), closes[name], anchor=name == store.COMMODITIES_ANCHOR_SERIES
        )
        for name in store.COMMODITIES_SERIES
    }


def _matrix(
    *, sessions: dict[str, tuple[date, ...]] | None = None, **overrides: tuple[float, ...]
) -> pl.DataFrame:
    """La matriz de la familia con la muestra sintetica."""
    return commodities_matrix(_inputs(sessions=sessions, **overrides), spec=commodities_spec())


def _column(frame: pl.DataFrame, name: str) -> list[float | None]:
    """La columna como lista de ``float``/``None`` (los ``null`` de polars)."""
    values = frame.get_column(name).to_list()
    return [None if value is None else float(value) for value in values]


# ─────────────────────────────────────────────────────────────────────────────
# A1/A2/A3 — la decision escrita y sus numeros
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_decision_is_written_for_the_four_series() -> None:
    """A1: una linea por serie con ``ENTRA`` o ``SE RECHAZA``, y ninguna sin decidir."""
    text = DECISION_PATH.read_text(encoding="utf-8")
    for series_id in ("CL=F", "BZ=F", "GC=F", "EURUSD=X"):
        assert f"| `{series_id}`" in text, series_id
    decisiones = [
        line for line in text.splitlines() if "| **ENTRA**" in line or "| **SE RECHAZA**" in line
    ]
    assert len(decisiones) == 4, decisiones


def test_a2_the_redundancies_are_published_with_their_number() -> None:
    """A2: los dos pares redundantes traen su ``r`` medido, no una afirmacion."""
    text = DECISION_PATH.read_text(encoding="utf-8")
    for number in ("+0.8596", "-0.3219", "-0.9720", "-0.7955", "-0.9198"):
        assert number in text, number


def test_a3_a_rejected_series_is_really_out() -> None:
    """A3: ``BZ=F`` no esta ni en las series, ni en el catalogo, ni en las fuentes."""
    assert set(store.COMMODITIES_ADMITTED_SERIES) == {"CL=F", "GC=F", "EURUSD=X"}
    assert "BZ=F" not in store.COMMODITIES_SERIES
    assert "BZ=F" not in store.COMMODITIES_FEATURE_COLUMNS
    assert all(series != "BZ=F" for _dataset, series in store.DEFAULT_COMMODITIES_SOURCES)
    assert set(store.COMMODITIES_SERIES) == {
        store.COMMODITIES_ANCHOR_SERIES,
        *store.COMMODITIES_ADMITTED_SERIES,
    }


# ─────────────────────────────────────────────────────────────────────────────
# A4/A5 — catalogo y registro
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_the_family_is_registered_with_its_source() -> None:
    """A4: la familia esta en los dos registros, con su ``source`` propio."""
    assert store.COMMODITIES_FEATURE_SET in store.CATALOG_BY_FEATURE_SET
    assert store.COMMODITIES_FEATURE_SET in store.SOURCE_BY_FEATURE_SET
    assert (
        store.CATALOG_BY_FEATURE_SET[store.COMMODITIES_FEATURE_SET]
        is store.COMMODITIES_FEATURE_CATALOG
    )
    assert (
        store.SOURCE_BY_FEATURE_SET[store.COMMODITIES_FEATURE_SET]
        == store.COMMODITIES_FEATURES_SOURCE
    )
    assert set(store.CATALOG_BY_FEATURE_SET) == set(store.SOURCE_BY_FEATURE_SET)
    assert set(store.COMMODITIES_FEATURE_COLUMNS) <= set(store.ALL_FEATURE_COLUMNS)


def test_a5_the_catalog_declares_every_column() -> None:
    """A5: cinco entradas, con formula, ventana, fuente y cierre del que depende cada una."""
    catalog = store.COMMODITIES_FEATURE_CATALOG
    assert store.COMMODITIES_FEATURE_COLUMNS == (
        "oil_ret_1",
        "oil_ret_5",
        "oil_ret_1_z",
        "gold_ret_1",
        "eurusd_ret_1",
    )
    assert tuple(entry.name for entry in catalog) == store.COMMODITIES_FEATURE_COLUMNS
    for entry in catalog:
        assert entry.formula, entry.name
        assert entry.source == "raw.market_daily", entry.name
        assert entry.required_as_of, entry.name
    windows = {entry.name: entry.window for entry in catalog}
    assert windows["oil_ret_1"] == 1
    assert windows["oil_ret_5"] == store.COMMODITIES_RETURN_WINDOW
    assert windows["oil_ret_1_z"] == store.COMMODITIES_MIN_SESSIONS
    assert dict(store.DEFAULT_COMMODITIES_WINDOWS) == windows
    assert (
        tuple(("raw.market_daily", series_id) for series_id in store.COMMODITIES_SERIES)
        == store.DEFAULT_COMMODITIES_SOURCES
    )
    assert COMMODITIES_INPUT_COLUMNS == ("session", "close")


# ─────────────────────────────────────────────────────────────────────────────
# A6 — point-in-time
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_the_row_of_t_does_not_read_the_close_of_t() -> None:
    """A6: mutar el cierre de ``t`` no mueve la fila de ``t``, y si la de ``t+1``."""
    target = 4
    closes = list(_BASE["CL=F"])
    closes[target] *= 3.0
    base = _matrix()
    mutated = _matrix(**{"CL=F": tuple(closes)})

    for name in store.COMMODITIES_FEATURE_COLUMNS:
        assert _column(mutated, name)[: target + 1] == _column(base, name)[: target + 1], name
    # la sesion siguiente si lo nota: su retorno lee el cierre de `t`
    assert _column(mutated, "oil_ret_1")[target + 1] != _column(base, "oil_ret_1")[target + 1]


def test_a6_mutating_a_later_session_does_not_move_the_previous_rows() -> None:
    """A6: ninguna columna de una fila lee una sesion posterior de su serie."""
    closes = list(_BASE["CL=F"])
    closes[-1] *= 5.0
    closes[-2] *= 2.0
    base = _matrix()
    mutated = _matrix(**{"CL=F": tuple(closes)})

    for name in ("oil_ret_1", "oil_ret_5", "gold_ret_1", "eurusd_ret_1"):
        assert _column(mutated, name)[:-1] == _column(base, name)[:-1], name


def test_a6_the_declared_lag_is_one_session() -> None:
    """A6: la familia declara su rezago, y es el mismo que el de ``dxy_ret_1``."""
    assert COMMODITIES_LAG_SESSIONS == 1
    catalog = {entry.name: entry.required_as_of for entry in store.COMMODITIES_FEATURE_CATALOG}
    context = {entry.name: entry.required_as_of for entry in store.CONTEXT_FEATURE_CATALOG}
    assert set(catalog.values()) == {"cierre de la sesion t-1"}
    assert context["dxy_ret_1"] == "cierre de la sesion t-1"


# ─────────────────────────────────────────────────────────────────────────────
# A7 — calendario propio
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_the_return_is_computed_on_the_calendar_of_its_own_series() -> None:
    """A7: con un hueco, el retorno salta la sesion que falta: nunca un cero."""
    # `CL=F` no cotiza el 2024-01-04; sus cierres son los de las sesiones que si tiene.
    gap = (
        SESSIONS[0],
        SESSIONS[1],
        SESSIONS[3],
        SESSIONS[4],
        SESSIONS[5],
        SESSIONS[6],
        SESSIONS[7],
    )
    frame = _matrix(
        sessions={"CL=F": gap},
        **{"CL=F": (70.0, 71.0, 73.0, 74.0, 75.0, 76.0, 77.0)},
    )

    # la fila del 2024-01-08 (indice 4) lee la ultima sesion del crudo `< t`, que es el 01-05
    row = _column(frame, "oil_ret_1")[4]
    assert row is not None
    assert math.isclose(row, math.log(73.0 / 71.0), rel_tol=1e-12), (
        "el retorno debe saltar el hueco"
    )
    assert 0.0 not in [value for value in _column(frame, "oil_ret_1") if value is not None]


def test_a7_a_series_without_history_is_null_and_never_forward_filled() -> None:
    """A7: una serie que arranca mas tarde no arrastra el cierre anterior hacia adelante."""
    late = SESSIONS[4:]
    frame = _matrix(
        sessions={"GC=F": late},
        **{"GC=F": (2000.0, 2010.0, 2020.0, 2030.0)},
    )
    gold = _column(frame, "gold_ret_1")
    assert gold[:4] == [None, None, None, None]
    assert gold[4] is None, "la primera sesion de la serie no tiene retorno"
    assert gold[5] is None, "la fila siguiente lee esa primera sesion, que tampoco tiene retorno"
    assert gold[6] is not None, "la segunda sesion de la serie ya tiene retorno"


# ─────────────────────────────────────────────────────────────────────────────
# A8 — null, nunca 0 ni NaN
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_what_cannot_be_computed_is_null_and_never_a_number() -> None:
    """A8: sin ventana suficiente la columna es ``null``; nada de ``0``, ``NaN`` ni ``inf``."""
    frame = _matrix()
    infinite = float("inf")
    for name in store.COMMODITIES_FEATURE_COLUMNS:
        column = frame.get_column(name)
        assert column.null_count() > 0, name
        assert not column.is_nan().any(), name
        assert not (column == infinite).any(), name
    # sin ventana de cinco sesiones ni de 250, las columnas que la piden empiezan nulas
    assert _column(frame, "oil_ret_5")[:6] == [None] * 6
    assert _column(frame, "oil_ret_5")[-1] is not None
    assert _column(frame, "oil_ret_1_z") == [None] * len(SESSIONS)


def test_a8_the_store_accepts_the_matrix_of_the_family() -> None:
    """A8: ``daily_records`` acepta la matriz y le pone el ``source`` de la familia."""
    frame = _matrix()
    records = store.daily_records(
        frame,
        spec=commodities_spec(),
        series_id=store.COMMODITIES_ANCHOR_SERIES,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert len(records) == len(SESSIONS)
    assert {record["source"] for record in records} == {store.COMMODITIES_FEATURES_SOURCE}
    assert set(records[0]) - {
        "source",
        "series_id",
        "as_of",
        "fetched_at",
        "published_at",
        "version",
        *store.VERSION_COLUMNS,
    } == set(store.COMMODITIES_FEATURE_COLUMNS)


# ─────────────────────────────────────────────────────────────────────────────
# A9 — el contrato de entrada
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_a_mapping_that_is_not_the_declared_one_names_the_series() -> None:
    """A9: una serie de menos o una de mas es un error tipado que nombra la serie."""
    frames = _inputs()
    with pytest.raises(store.CommoditiesInputError, match="CL=F"):
        commodities_matrix(
            {k: v for k, v in frames.items() if k != "CL=F"}, spec=commodities_spec()
        )
    with pytest.raises(store.CommoditiesInputError, match="BZ=F"):
        commodities_matrix({**frames, "BZ=F": frames["CL=F"]}, spec=commodities_spec())


def test_a9_a_series_without_session_or_close_names_the_series() -> None:
    """A9: falta ``session`` o ``close``: el mensaje dice **de que** serie."""
    frames = _inputs()
    frames["GC=F"] = frames["GC=F"].drop("close")
    with pytest.raises(store.CommoditiesInputError, match="GC=F"):
        commodities_matrix(frames, spec=commodities_spec())

    frames = _inputs()
    frames["EURUSD=X"] = frames["EURUSD=X"].drop("session")
    with pytest.raises(store.CommoditiesInputError, match="EURUSD=X"):
        commodities_matrix(frames, spec=commodities_spec())


def test_a9_a_repeated_session_or_a_foreign_spec_is_rejected() -> None:
    """A9: una sesion repetida y una spec de otra familia son error, no un resultado."""
    frames = _inputs()
    frames["CL=F"] = pl.concat([frames["CL=F"], frames["CL=F"].tail(1)])
    with pytest.raises(store.CommoditiesInputError, match="repite"):
        commodities_matrix(frames, spec=commodities_spec())

    with pytest.raises(store.InvalidFeatureMatrixError, match="commodities_v1"):
        commodities_matrix(_inputs(), spec=store.FeatureSpec())


# ─────────────────────────────────────────────────────────────────────────────
# A10 — el modulo es puro
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_the_module_does_not_read_the_clock_or_the_filesystem() -> None:
    """A10: ni reloj, ni disco, ni entorno: entra un ``Mapping``, sale un ``DataFrame``."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "datetime.now",
        "utcnow",
        "time.time",
        "pathlib",
        "os.environ",
        "read_text",
        "open(",
    ):
        assert forbidden not in source, forbidden
    assert "import os" not in source


# ─────────────────────────────────────────────────────────────────────────────
# A11 — cableada al frame
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_the_family_is_assembled_with_the_others() -> None:
    """A11: la familia entra en el orden de ensamblado, con sus columnas y su spec."""
    assert store.COMMODITIES_FEATURE_SET in feature_frame.FAMILY_ORDER
    assert feature_frame.FAMILY_ORDER[-1] == store.COMMODITIES_FEATURE_SET
    assert (
        feature_frame.COLUMNS_BY_FAMILY[store.COMMODITIES_FEATURE_SET]
        == store.COMMODITIES_FEATURE_COLUMNS
    )
    assert feature_frame.family_spec(store.COMMODITIES_FEATURE_SET) == commodities_spec()


# ─────────────────────────────────────────────────────────────────────────────
# A12 — el contrato del almacen
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_the_layout_of_features_daily_declares_the_new_columns() -> None:
    """A12: las cinco columnas estan en la firma y el ``layout_version`` ha subido."""
    layout = contracts.LAYOUTS[("derived", "features_daily")]
    assert layout.layout_version == 2
    assert set(layout.signature) == {*store.ALL_FEATURE_COLUMNS, *store.VERSION_COLUMNS}
    assert set(store.COMMODITIES_FEATURE_COLUMNS) <= set(layout.signature)


# ─────────────────────────────────────────────────────────────────────────────
# A13 — la produccion no cambia
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_the_model_keeps_its_ten_features() -> None:
    """A13: ``BASELINE_FEATURES`` sigue siendo la misma tupla de diez y sin las nuevas."""
    assert len(BASELINE_FEATURES) == 10
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
    assert not set(store.COMMODITIES_FEATURE_COLUMNS) & set(BASELINE_FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# A14 — la identidad de la familia
# ─────────────────────────────────────────────────────────────────────────────
def test_a14_the_spec_is_deterministic_and_the_code_version_does_not_move() -> None:
    """A14: el digest de la spec es estable y anadir una familia **no** sube la constante."""
    first = store.feature_spec_sha256(commodities_spec())
    assert first == store.feature_spec_sha256(commodities_spec())
    assert len(first) == 64
    assert all(char in "0123456789abcdef" for char in first)
    assert first != store.feature_spec_sha256(store.FeatureSpec())
    assert store.FEATURE_CODE_VERSION == 2
