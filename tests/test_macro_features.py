"""Tests de la familia macro (`macro_v1`, tarea #22).

Un test por criterio, ``test_aN_...``. La raiz del ``Store`` vive siempre bajo
``tmp_path``: la fixture de sesion de ``tests/conftest.py`` huella el ``data/`` y
el ``runs/`` del repositorio y falla si la sesion escribe en ellos.

El *golden dataset* de A13 vive en ``tests/fixtures/features/``: los inputs en
``macro_golden_market.csv`` (las barras de ``^GSPC`` y ``DX-Y.NYB``) y
``macro_golden_series.csv`` (las seis series macro), y el par esperado en
``macro_golden_expected.json``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import subprocess
import sys
from bisect import bisect_right
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

from cfdtrader.data.store import Store, WriteOutcome
from cfdtrader.features import context, macro, store, technical
from cfdtrader.features.macro import macro_matrix, macro_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "features"

#: Primera sesion sintetica. El modulo no conoce el calendario a proposito.
FIRST_SESSION = date(2025, 1, 2)

#: Hora UTC que el test usa como cierre de sesion. Es un valor **del test**.
CLOSE_HOUR_UTC = 20

#: Hora UTC de publicacion del fixture sintetico: despues del cierre de **su** dia.
PUBLISH_HOUR_UTC = 22

#: Instante de captura de los tests: posterior a todas las sesiones usadas (el
#: golden macro termina el 2026-09-16, asi que la captura va despues).
FETCHED_AT = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)

#: ``source`` de la familia macro (el discriminador de familia de #20).
MACRO_SOURCE = store.MACRO_FEATURES_SOURCE

#: Formato de los instantes en los dos CSV del golden.
DT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: Los digests congelados de las tres familias anteriores, tal y como los dejan
#: #19, #20 y #21: unirlas a la cuarta no puede mover ninguno.
FROZEN_19 = (
    "sha256:3c71eb69635589dc7ece74b662170b27104473c3b81f3f21434ac178d7dc539b",
    "sha256:5abdd54362feddfb816ffa2dc566d3039c0a94b597e1cef75e1b2e064d56b947",
)
FROZEN_20 = (
    "sha256:6e4d86aee66be928d860184e65d03f7de03da59fae58a87467b5acdc95b3b78b",
    "sha256:6dbe0a372a64534c07c52e8e44c104163a5f4c27159a40610f5cac9172c9650e",
)
FROZEN_21 = (
    "sha256:513ec8206aea00b5b99341d2c937b38ce41c473a1bfbc864a4fbd63947033614",
    "sha256:09a8b459e444a441bf0aea3a7f5c25e8dd61d00ba58d0c05ae43cb23dc818213",
)

#: La tabla de la issue, literal: ventana y ``required_as_of`` de cada entrada.
CATALOG_TABLE: dict[str, tuple[int | None, str]] = {
    "fed_funds": (None, "ultima publicacion anterior al cierre de la sesion t"),
    "fed_funds_chg_5": (5, "cierre de la sesion t y de la sesion t-5"),
    "ust_10y": (None, "ultima publicacion anterior al cierre de la sesion t"),
    "ust_10y_chg_5": (5, "cierre de la sesion t y de la sesion t-5"),
    "ust_2y": (None, "ultima publicacion anterior al cierre de la sesion t"),
    "ust_2y_chg_5": (5, "cierre de la sesion t y de la sesion t-5"),
    "pendiente_2s10s": (None, "ultima publicacion anterior al cierre de la sesion t"),
    "pendiente_2s10s_chg_5": (5, "cierre de la sesion t y de la sesion t-5"),
    "cpi_yoy": (
        None,
        "ultima publicacion anterior al cierre de la sesion t y su referencia de hace un ano",
    ),
    "pce_yoy": (
        None,
        "ultima publicacion anterior al cierre de la sesion t y su referencia de hace un ano",
    ),
    "dxy": (None, "cierre de la barra del DXY de la sesion t"),
    "ust_10y_z": (250, "cierre de la sesion t y 250 sesiones de historia previa"),
    "dxy_z": (250, "cierre de la sesion t y 250 sesiones de historia previa"),
}

#: Las series macro del enunciado que **no** entran en la familia.
FOREIGN_SERIES = ("PAYEMS",)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def _sessions(count: int, *, start: date = FIRST_SESSION) -> list[date]:
    """``count`` sesiones consecutivas desde ``start``."""
    return [start + timedelta(days=index) for index in range(count)]


def _instant(day: date, hour: int, minute: int = 0) -> datetime:
    """Instante UTC de un dia sintetico."""
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


def _macro_frame(rows: Sequence[tuple[date, datetime | None, float | None]]) -> pl.DataFrame:
    """Frame de una serie macro: ``as_of`` (referencia), ``published_at`` y ``value``."""
    return pl.DataFrame(
        {
            "as_of": pl.Series("as_of", [row[0] for row in rows], dtype=pl.Date()),
            "published_at": pl.Series(
                "published_at", [row[1] for row in rows], dtype=pl.Datetime("us", "UTC")
            ),
            "value": pl.Series("value", [row[2] for row in rows], dtype=pl.Float64()),
        }
    )


def _flat_series(
    position: int,
    sessions: list[date],
    *,
    value: float | None = None,
    with_annual_base: bool = False,
) -> pl.DataFrame:
    """Serie macro plana: cada sesion publica el dia anterior a las 22:00 UTC.

    A las 22:00 UTC del dia anterior ya ha cerrado la sesion siguiente (20:00 UTC),
    asi que el nivel transportado de la sesion ``t`` es el de la referencia
    ``t - 1 dia`` y vale lo mismo en todas las filas.

    ``with_annual_base`` anade, para cada referencia, la de la **misma fecha un ano
    antes** con un valor menor: es lo que necesita ``cpi_yoy``/``pce_yoy`` para no
    salir nulo (la formula exige las dos puntas publicadas).
    """
    level = float(position + 1) if value is None else value
    rows: list[tuple[date, datetime | None, float | None]] = []
    for index, session in enumerate(sessions):
        reference = session - timedelta(days=1)
        # el nivel se mueve despacio: una serie constante dejaria el MAD a cero y
        # las dos `_z` en `null`, que es justo lo que hay que poder ejercitar
        rows.append((reference, _instant(reference, PUBLISH_HOUR_UTC), level + 0.01 * index))
        if not with_annual_base:
            continue
        try:
            base = date(reference.year - 1, reference.month, reference.day)
        except ValueError:  # el 29 de febrero no tiene fecha exacta un ano antes
            continue
        rows.append((base, _instant(base, PUBLISH_HOUR_UTC), level + 0.01 * index - 0.5))
    return _macro_frame(rows)


def _market_frame(
    sessions: list[date],
    *,
    hour: int = CLOSE_HOUR_UTC,
    closes: Sequence[float | None] | None = None,
) -> pl.DataFrame:
    """Barra de mercado: ``session`` + ``as_of`` (y ``close`` si se pide)."""
    data: dict[str, pl.Series] = {
        "session": pl.Series("session", sessions, dtype=pl.Date()),
        "as_of": pl.Series(
            "as_of",
            [_instant(session, hour) for session in sessions],
            dtype=pl.Datetime("us", "UTC"),
        ),
    }
    if closes is not None:
        data["close"] = pl.Series("close", closes, dtype=pl.Float64())
    return pl.DataFrame(data)


def _universe(count: int = 12, *, start: date = FIRST_SESSION) -> dict[str, pl.DataFrame]:
    """Las ocho series del contrato, con datos: niveles planos y DXY con paseo."""
    sessions = _sessions(count, start=start)
    frames = {
        name: _flat_series(position, sessions, with_annual_base=name in ("CPIAUCSL", "PCEPI"))
        for position, name in enumerate(store.MACRO_SERIES)
    }
    frames[macro.ANCHOR_SERIES] = _market_frame(sessions)
    frames[macro.DXY_SERIES] = _market_frame(
        sessions, closes=[100.0 + 0.5 * index for index in range(count)]
    )
    return frames


def _matrix(
    count: int = 12,
    *,
    start: date = FIRST_SESSION,
    overrides: dict[str, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Matriz macro completa con el fixture sintetico."""
    frames = _universe(count, start=start)
    if overrides:
        frames.update(overrides)
    return macro_matrix(frames, spec=macro_spec())


def _column(matrix: pl.DataFrame, name: str) -> list[float | None]:
    """Columna de features como lista de floats con nulos."""
    return [
        None if value is None else float(cast("float", value))
        for value in cast("list[object]", matrix.get_column(name).to_list())
    ]


def _assert_finite(matrix: pl.DataFrame) -> None:
    """Ninguna celda de feature es ``NaN`` ni ``inf``: o numero finito, o ``null``."""
    for name in store.MACRO_FEATURE_COLUMNS:
        for value in cast("list[object]", matrix.get_column(name).to_list()):
            assert value is None or math.isfinite(float(cast("float", value)))


def _source() -> str:
    """Codigo fuente del modulo, del fichero real (no del bytecode)."""
    return Path(macro.__file__).read_text(encoding="utf-8")


def _tree() -> ast.Module:
    """Arbol sintactico del modulo nuevo."""
    return ast.parse(_source())


def _imported_modules() -> set[str]:
    """Modulos importados por el modulo nuevo, tal y como los lee el AST."""
    modules: set[str] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _prefixed(digest: str) -> str:
    """Forma en la que el golden guarda un digest: con el prefijo ``sha256:``.

    El mismo prefijo que publica ``features_version``; ademas, un hex desnudo de 64
    caracteres dispara el hook ``detect-secrets`` (falso positivo de alta entropia).
    """
    return store.FEATURE_VERSION_PREFIX + digest


# ─────────────────────────────────────────────────────────────────────────────
# Golden: lectura de los dos CSV
# ─────────────────────────────────────────────────────────────────────────────
def _golden_market() -> pl.DataFrame:
    """Barras del golden: ``^GSPC`` (ancla) y ``DX-Y.NYB``, una fila por sesion.

    Los CSV se leen con el esquema declarado como texto y se convierten a mano:
    depender de la inferencia de ``pl.read_csv`` con 17 decimales de diferencia es
    exactamente el tipo de fragilidad que un golden no puede permitirse.
    """
    table = pl.read_csv(
        FIXTURES / "macro_golden_market.csv",
        schema_overrides={
            "session": pl.String,
            "gspc_as_of": pl.String,
            "dxy_as_of": pl.String,
            "dxy_close": pl.String,
        },
    )
    return table.with_columns(
        pl.col("session").str.to_date(),
        pl.col("gspc_as_of").str.to_datetime(format=DT_FORMAT, time_zone="UTC"),
        pl.col("dxy_as_of").str.to_datetime(format=DT_FORMAT, time_zone="UTC"),
        pl.col("dxy_close").cast(pl.Float64),
    )


def _golden_series() -> pl.DataFrame:
    """Las seis series macro del golden, en formato largo."""
    table = pl.read_csv(
        FIXTURES / "macro_golden_series.csv",
        schema_overrides={
            "series_id": pl.String,
            "as_of": pl.String,
            "published_at": pl.String,
            "value": pl.String,
        },
    )
    return table.with_columns(
        pl.col("as_of").str.to_date(),
        pl.col("published_at").str.to_datetime(format=DT_FORMAT, time_zone="UTC"),
        pl.col("value").cast(pl.Float64),
    )


def _golden_frames() -> dict[str, pl.DataFrame]:
    """Las ocho series del golden, con la forma que espera ``macro_matrix``."""
    market = _golden_market()
    series = _golden_series()
    frames = {
        name: series.filter(pl.col("series_id") == name).select("as_of", "published_at", "value")
        for name in store.MACRO_SERIES
    }
    frames[macro.ANCHOR_SERIES] = market.select("session", pl.col("gspc_as_of").alias("as_of"))
    frames[macro.DXY_SERIES] = market.drop_nulls("dxy_as_of").select(
        "session", pl.col("dxy_as_of").alias("as_of"), pl.col("dxy_close").alias("close")
    )
    return frames


def _expected() -> dict[str, Any]:
    """Par esperado congelado de A13."""
    raw = (FIXTURES / "macro_golden_expected.json").read_text(encoding="utf-8")
    return dict(json.loads(raw))


def _frozen_spec(expected: dict[str, Any]) -> store.FeatureSpec:
    """Spec exacta que declara un golden: reconstruirla es parte del congelado."""
    return store.FeatureSpec(
        feature_set=str(expected["feature_set"]),
        code_version=int(cast("int", expected["code_version"])),
        parameters=cast("dict[str, object]", expected["parameters"]),
        windows=cast("dict[str, int | None]", expected["windows"]),
        sources=tuple(
            (str(left), str(right)) for left, right in cast("list[list[str]]", expected["sources"])
        ),
    )


def _golden() -> tuple[store.FeatureSpec, pl.DataFrame]:
    """Spec y matriz del golden macro, tal y como los congela A13."""
    spec = _frozen_spec(_expected())
    return spec, macro_matrix(_golden_frames(), spec=spec)


def _golden_matrix() -> pl.DataFrame:
    """La matriz del golden, cuando el test no necesita la spec."""
    _, matrix = _golden()
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# A1 — modulo, firma, pureza y validacion de la entrada
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_module_is_pure() -> None:
    """Sin reloj, sin ficheros y sin almacen: el AST y el texto lo confirman."""
    source = _source()
    for forbidden in ("datetime.now", "utcnow", "date.today", "time.time", "time_ns"):
        assert forbidden not in source
    assert "cfdtrader.data" not in source
    for module in _imported_modules():
        assert not module.startswith("cfdtrader.analysis")
        assert module != "cfdtrader.data.store"

    for node in ast.walk(_tree()):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "open"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"now", "utcnow", "today", "time"}
    assert macro.macro_matrix.__module__ == "cfdtrader.features.macro"
    assert "\t" not in source


def test_a1_the_matrix_is_session_as_of_and_the_catalog_in_order() -> None:
    """``macro_matrix`` publica ``session``, ``as_of`` y las 13 columnas, ordenado."""
    matrix = _matrix(12)
    assert matrix.columns == ["session", "as_of", *store.MACRO_FEATURE_COLUMNS]
    assert matrix.height == 12
    assert matrix.get_column("session").to_list() == _sessions(12)
    assert matrix.get_column("as_of").dtype == pl.Datetime("us", "UTC")
    _assert_finite(matrix)

    shuffled = macro_matrix(
        {name: frame.reverse() for name, frame in _universe(12).items()}, spec=macro_spec()
    )
    assert store.matrix_sha256(shuffled) == store.matrix_sha256(matrix)

    with pytest.raises(store.InvalidFeatureMatrixError, match="es la entrada de"):
        macro_matrix(_universe(12), spec=store.FeatureSpec())
    with pytest.raises(store.InvalidFeatureMatrixError, match="es la entrada de"):
        macro_matrix(_universe(12), spec=technical.technical_spec())


def test_a1_the_input_mapping_is_validated() -> None:
    """Ocho claves exactas, columnas por serie y los dos tipos de ``as_of``."""
    frames = _universe(8)

    missing = dict(frames)
    del missing["PCEPI"]
    with pytest.raises(store.MacroInputError, match="faltan \\['PCEPI'\\]"):
        macro_matrix(missing, spec=macro_spec())

    extra = dict(frames)
    extra["PAYEMS"] = _flat_series(99, _sessions(8))
    with pytest.raises(store.MacroInputError, match="sobran \\['PAYEMS'\\]"):
        macro_matrix(extra, spec=macro_spec())

    without_published = dict(frames)
    without_published["DGS2"] = frames["DGS2"].drop("published_at")
    with pytest.raises(store.MacroInputError, match="'DGS2' no trae las columnas"):
        macro_matrix(without_published, spec=macro_spec())

    without_close = dict(frames)
    without_close[macro.DXY_SERIES] = frames[macro.DXY_SERIES].drop("close")
    with pytest.raises(store.MacroInputError, match="no trae las columnas"):
        macro_matrix(without_close, spec=macro_spec())

    # el `as_of` de una serie macro es una FECHA: un instante es el tipo de una barra
    as_instant = dict(frames)
    as_instant["T10Y2Y"] = frames["T10Y2Y"].with_columns(
        pl.col("as_of").cast(pl.Datetime("us", "UTC"))
    )
    with pytest.raises(
        store.MacroInputError, match=r"tiene que ser pl\.Date \(la fecha de referencia"
    ):
        macro_matrix(as_instant, spec=macro_spec())

    # ... y el de una barra es un INSTANTE con zona
    naive = dict(frames)
    naive[macro.ANCHOR_SERIES] = frames[macro.ANCHOR_SERIES].with_columns(
        pl.Series(
            "as_of",
            [_instant(session, CLOSE_HOUR_UTC).replace(tzinfo=None) for session in _sessions(8)],
        )
    )
    with pytest.raises(store.MacroInputError, match="zona horaria"):
        macro_matrix(naive, spec=macro_spec())

    not_datetime = dict(frames)
    not_datetime[macro.ANCHOR_SERIES] = frames[macro.ANCHOR_SERIES].with_columns(
        pl.col("as_of").cast(pl.Utf8)
    )
    with pytest.raises(store.MacroInputError, match=r"tiene que ser pl\.Datetime"):
        macro_matrix(not_datetime, spec=macro_spec())

    without_session = dict(frames)
    without_session[macro.ANCHOR_SERIES] = frames[macro.ANCHOR_SERIES].drop("session")
    with pytest.raises(store.MacroInputError, match="no trae las columnas"):
        macro_matrix(without_session, spec=macro_spec())

    shifted = dict(frames)
    shifted[macro.ANCHOR_SERIES] = frames[macro.ANCHOR_SERIES].with_columns(
        pl.Series(
            "as_of",
            [_instant(session, CLOSE_HOUR_UTC) + timedelta(hours=6) for session in _sessions(8)],
            dtype=pl.Datetime("us", "UTC"),
        )
    )
    with pytest.raises(store.MacroInputError, match="no corresponden al mismo dia"):
        macro_matrix(shifted, spec=macro_spec())

    repeated = dict(frames)
    anchor = frames[macro.ANCHOR_SERIES]
    repeated[macro.ANCHOR_SERIES] = anchor.vstack(anchor.slice(2, 1))
    with pytest.raises(store.MacroInputError, match="repite estas sesiones"):
        macro_matrix(repeated, spec=macro_spec())

    poisoned = dict(frames)
    poisoned["DFF"] = frames["DFF"].with_columns(
        pl.Series("value", [float("inf"), *[1.0] * (frames["DFF"].height - 1)], dtype=pl.Float64)
    )
    with pytest.raises(store.MacroInputError, match="no finito"):
        macro_matrix(poisoned, spec=macro_spec())


def test_a1_the_per_row_guards_of_the_eight_series() -> None:
    """Los guards por fila: fecha nula, valor de texto, sesion nula y tipo raro."""
    frames = _universe(6)
    sessions = _sessions(6)

    # una fecha de referencia nula no se puede transportar
    null_reference = dict(frames)
    null_reference["DGS10"] = frames["DGS10"].with_columns(
        pl.Series("as_of", [None, *sessions[1:]], dtype=pl.Date())
    )
    with pytest.raises(store.MacroInputError, match="'as_of' nula"):
        macro_matrix(null_reference, spec=macro_spec())

    # una columna de valores de texto
    as_text = dict(frames)
    as_text["DGS10"] = frames["DGS10"].with_columns(pl.col("value").cast(pl.Utf8))
    with pytest.raises(store.MacroInputError, match="no es numerica"):
        macro_matrix(as_text, spec=macro_spec())

    # el ``as_of`` del ancla no puede ser nulo, y su ``session`` sí puede venir
    # como ``Datetime``: se lee el dia
    null_instant = dict(frames)
    null_instant[macro.ANCHOR_SERIES] = frames[macro.ANCHOR_SERIES].with_columns(
        pl.Series("as_of", [None, *_instants_of(frames)[1:]], dtype=pl.Datetime("us", "UTC"))
    )
    with pytest.raises(store.MacroInputError, match="trae un 'as_of' nulo"):
        macro_matrix(null_instant, spec=macro_spec())

    as_datetime = dict(frames)
    as_datetime[macro.ANCHOR_SERIES] = frames[macro.ANCHOR_SERIES].with_columns(
        pl.col("session").cast(pl.Datetime("us", "UTC"))
    )
    assert store.matrix_sha256(macro_matrix(as_datetime, spec=macro_spec())) == store.matrix_sha256(
        _matrix(6)
    )

    # una ``session`` nula o de otro tipo si es un error
    null_session = dict(frames)
    null_session[macro.ANCHOR_SERIES] = frames[macro.ANCHOR_SERIES].with_columns(
        pl.Series("session", [None, *sessions[1:]], dtype=pl.Date())
    )
    with pytest.raises(store.MacroInputError, match="'session' nula"):
        macro_matrix(null_session, spec=macro_spec())

    as_integer = dict(frames)
    as_integer[macro.ANCHOR_SERIES] = frames[macro.ANCHOR_SERIES].with_columns(
        pl.col("session").cast(pl.Int32)
    )
    with pytest.raises(store.MacroInputError, match=r"tiene que ser pl\.Date o pl\.Datetime"):
        macro_matrix(as_integer, spec=macro_spec())

    # y los mismos guards en la barra del DXY
    null_bar = dict(frames)
    null_bar[macro.DXY_SERIES] = frames[macro.DXY_SERIES].with_columns(
        pl.Series(
            "as_of",
            [None, *_instants_of(frames, name=macro.DXY_SERIES)[1:]],
            dtype=pl.Datetime("us", "UTC"),
        )
    )
    with pytest.raises(store.MacroInputError, match="trae un 'as_of' nulo"):
        macro_matrix(null_bar, spec=macro_spec())

    shifted_bar = dict(frames)
    shifted_bar[macro.DXY_SERIES] = frames[macro.DXY_SERIES].with_columns(
        pl.Series(
            "as_of",
            [_instant(session + timedelta(days=1), CLOSE_HOUR_UTC) for session in sessions],
            dtype=pl.Datetime("us", "UTC"),
        )
    )
    with pytest.raises(store.MacroInputError, match="no corresponden al mismo dia"):
        macro_matrix(shifted_bar, spec=macro_spec())


def _instants_of(frames: dict[str, pl.DataFrame], *, name: str = macro.ANCHOR_SERIES) -> list[Any]:
    """Columna ``as_of`` de una barra del fixture, como lista."""
    return frames[name].get_column("as_of").to_list()


def test_a1_an_empty_series_leaves_nulls_and_never_a_zero() -> None:
    """Una serie sin historia deja ``null`` en sus columnas, no un cero."""
    frames = _universe(8)
    empty = _macro_frame([])
    matrix = macro_matrix({**frames, "CPIAUCSL": empty}, spec=macro_spec())
    assert _column(matrix, "cpi_yoy") == [None] * 8
    assert _column(matrix, "fed_funds") != [None] * 8
    _assert_finite(matrix)

    # una barra del DXY sin ninguna fila tampoco inventa un nivel
    without_bars = macro_matrix(
        {**frames, macro.DXY_SERIES: _market_frame([], closes=[])}, spec=macro_spec()
    )
    assert _column(without_bars, "dxy") == [None] * 8
    assert _column(without_bars, "dxy_z") == [None] * 8
    _assert_finite(without_bars)


# ─────────────────────────────────────────────────────────────────────────────
# A2 — registro
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_registry_declares_the_fourth_family() -> None:
    """El registro expone las cuatro familias y la spec macro se construye."""
    assert sorted(store.CATALOG_BY_FEATURE_SET) == [
        "context_v1",
        "macro_v1",
        "technical_v1",
        "volatility_v1",
    ]
    assert store.CATALOG_BY_FEATURE_SET["macro_v1"] is store.MACRO_FEATURE_CATALOG
    assert store.SOURCE_BY_FEATURE_SET["macro_v1"] == MACRO_SOURCE
    assert MACRO_SOURCE not in {
        store.FEATURES_SOURCE,
        store.TECHNICAL_FEATURES_SOURCE,
        store.CONTEXT_FEATURES_SOURCE,
    }
    assert set(store.SOURCE_BY_FEATURE_SET) == set(store.CATALOG_BY_FEATURE_SET)
    assert len(set(store.SOURCE_BY_FEATURE_SET.values())) == 4

    spec = macro_spec()
    assert spec.feature_set == store.MACRO_FEATURE_SET
    assert spec.code_version == store.FEATURE_CODE_VERSION
    assert spec.windows == store.DEFAULT_MACRO_WINDOWS
    assert spec.sources == store.DEFAULT_MACRO_SOURCES
    assert (
        store.FeatureSpec(
            feature_set="macro_v1",
            windows=store.DEFAULT_MACRO_WINDOWS,
            sources=store.DEFAULT_MACRO_SOURCES,
        )
        == spec
    )
    assert store.FEATURE_CODE_VERSION == 1


# ─────────────────────────────────────────────────────────────────────────────
# A3 — catalogo
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_the_catalog_is_exactly_the_thirteen_declared_entries() -> None:
    """Nombre, ventana y ``required_as_of`` de las trece entradas, uno por uno."""
    assert [entry.name for entry in store.MACRO_FEATURE_CATALOG] == list(CATALOG_TABLE)
    assert tuple(CATALOG_TABLE) == store.MACRO_FEATURE_COLUMNS
    assert store.MACRO_CHG_WINDOW == 5
    assert store.MACRO_MIN_SESSIONS == 250
    for entry in store.MACRO_FEATURE_CATALOG:
        window, required = CATALOG_TABLE[entry.name]
        assert entry.window == window
        assert entry.required_as_of == required
        assert entry.formula.strip() == entry.formula
        assert entry.formula and entry.source.startswith("raw.")
    assert {entry.source for entry in store.MACRO_FEATURE_CATALOG} == {
        "raw.macro",
        "raw.market_daily",
    }
    declared = {entry.name: entry.window for entry in store.MACRO_FEATURE_CATALOG}
    assert declared == store.DEFAULT_MACRO_WINDOWS


def test_a3_a_bad_window_or_a_foreign_name_is_a_typed_error() -> None:
    """Ventana contradictoria (``dxy: 5``) y nombre de otra familia."""
    with pytest.raises(store.InvalidFeatureSpecError, match="no coincide"):
        store.FeatureSpec(feature_set="macro_v1", windows={**store.DEFAULT_MACRO_WINDOWS, "dxy": 5})
    with pytest.raises(store.InvalidFeatureSpecError, match="no esta en el catalogo"):
        store.FeatureSpec(
            feature_set="macro_v1", windows={**store.DEFAULT_MACRO_WINDOWS, "ret_1": 1}
        )
    with pytest.raises(store.InvalidFeatureSpecError, match="no esta en el catalogo"):
        store.FeatureSpec(
            feature_set="macro_v1", windows={**store.DEFAULT_MACRO_WINDOWS, "dxy_ret_1": 1}
        )


def test_a3_daily_records_accepts_the_macro_matrix_and_rejects_a_foreign_column() -> None:
    """La matriz macro pasa ``daily_records``; una columna ajena no."""
    matrix = _matrix(12)
    records = store.daily_records(
        matrix, spec=macro_spec(), series_id=macro.ANCHOR_SERIES, fetched_at=FETCHED_AT
    )
    assert set(records[0]) == {
        "source",
        "series_id",
        "as_of",
        "fetched_at",
        "published_at",
        "features_version",
        "feature_spec_sha256",
        *store.MACRO_FEATURE_COLUMNS,
    }
    assert all(record["source"] == MACRO_SOURCE for record in records)

    for foreign in ("ret_1", "corr_dax_60", "vix_level"):
        with pytest.raises(store.InvalidFeatureMatrixError, match="fuera del catalogo"):
            store.daily_records(
                matrix.with_columns(pl.Series(foreign, [0.0] * matrix.height)),
                spec=macro_spec(),
                series_id=macro.ANCHOR_SERIES,
                fetched_at=FETCHED_AT,
            )


def test_a3_a_poisoned_macro_column_is_caught_by_the_digest() -> None:
    """Sin las 13 columnas en ``ALL_FEATURE_COLUMNS`` un ``inf`` no se veria."""
    assert set(store.MACRO_FEATURE_COLUMNS) <= set(store.ALL_FEATURE_COLUMNS)
    matrix = _matrix(12)
    for poisoned in (float("nan"), float("inf")):
        broken = matrix.with_columns(pl.Series("pendiente_2s10s", [poisoned] * matrix.height))
        with pytest.raises(store.InvalidFeatureMatrixError, match="no finito"):
            store.matrix_sha256(broken)


# ─────────────────────────────────────────────────────────────────────────────
# A4 — nombres
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_the_names_do_not_collide_between_the_four_families() -> None:
    """Ninguna de las 13 macro esta en los otros tres catalogos, y no hay `dxy_ret_1`."""
    macros = set(store.MACRO_FEATURE_COLUMNS)
    assert macros & set(store.FEATURE_COLUMNS) == set()
    assert macros & set(store.TECHNICAL_FEATURE_COLUMNS) == set()
    assert macros & set(store.CONTEXT_FEATURE_COLUMNS) == set()
    assert "dxy_ret_1" not in macros
    assert "dxy_ret_1" in store.CONTEXT_FEATURE_COLUMNS
    assert len(macros) == 13

    # el unico repetido entre las cuatro familias sigue siendo `atr_norm` (#72)
    union = (
        set(store.FEATURE_COLUMNS)
        | set(store.TECHNICAL_FEATURE_COLUMNS)
        | set(store.CONTEXT_FEATURE_COLUMNS)
        | macros
    )
    assert len(store.ALL_FEATURE_COLUMNS) == len(union)
    assert len(store.ALL_FEATURE_COLUMNS) == (
        len(store.FEATURE_COLUMNS)
        + len(store.TECHNICAL_FEATURE_COLUMNS)
        + len(store.CONTEXT_FEATURE_COLUMNS)
        + 13
        - 1
    )
    assert set(store.ALL_FEATURE_COLUMNS) == union


# ─────────────────────────────────────────────────────────────────────────────
# A5 — acoplamiento con las aserciones congeladas
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_the_frozen_coupling_is_resolved_with_the_macro_columns() -> None:
    """Las tres aserciones congeladas, re-verificadas desde la familia nueva.

    Registrar ``macro_v1`` rompe **tres** aserciones de las suites anteriores y
    cada una se edita con **una** linea: la lista esperada de
    ``test_technical_features.py::test_a1``, la de
    ``test_context_features.py::test_a1`` y la union de ``ALL_FEATURE_COLUMNS`` de
    ``test_context_features.py::test_a9``. Este test re-verifica los tres hechos
    (registro de cuatro familias, catalogos anteriores intactos y union completa):
    si alguno de esos ficheros se hubiera "arreglado" de otra forma, aqui se ve.
    """
    assert sorted(store.CATALOG_BY_FEATURE_SET) == [
        "context_v1",
        "macro_v1",
        "technical_v1",
        "volatility_v1",
    ]
    assert store.CATALOG_BY_FEATURE_SET["technical_v1"] is store.TECHNICAL_FEATURE_CATALOG
    assert store.CATALOG_BY_FEATURE_SET["context_v1"] is store.CONTEXT_FEATURE_CATALOG
    assert store.CATALOG_BY_FEATURE_SET["volatility_v1"] is store.FEATURE_CATALOG
    assert len(store.FEATURE_COLUMNS) == 12
    assert len(store.TECHNICAL_FEATURE_COLUMNS) == 10
    assert len(store.CONTEXT_FEATURE_COLUMNS) == 11

    union = (
        set(store.FEATURE_COLUMNS)
        | set(store.TECHNICAL_FEATURE_COLUMNS)
        | set(store.CONTEXT_FEATURE_COLUMNS)
        | set(store.MACRO_FEATURE_COLUMNS)
    )
    assert len(store.ALL_FEATURE_COLUMNS) == len(union)


# ─────────────────────────────────────────────────────────────────────────────
# A6 — instantes (R)
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_a_publication_after_the_close_does_not_enter_the_row() -> None:
    """Una publicacion posterior al cierre de ``t`` cambia la fila de la sesion siguiente."""
    count = 6
    sessions = _sessions(count)
    frames = _universe(count)
    # la sesion 2 cierra a las 20:00 UTC; esta publicacion es a las 23:00 UTC
    late = (sessions[2], _instant(sessions[2], 23), 99.0)
    with_late = dict(frames)
    with_late["DFF"] = _macro_frame([*_rows_of(frames["DFF"]), late])

    before = _column(_matrix(count), "fed_funds")
    after = _column(macro_matrix(with_late, spec=macro_spec()), "fed_funds")
    assert after[:3] == before[:3]
    assert after[2] != 99.0
    assert after[3] == 99.0
    # a partir de ahi el dato normal vuelve a ser el mas reciente: la tardia no se queda
    assert after[4:] == before[4:]

    # la unica comparacion temporal es `published_at` contra el `as_of` de la fila:
    # desplazar el cierre de una sesion media hora no cambia nada mas
    other_hour = dict(frames)
    other_hour[macro.ANCHOR_SERIES] = _market_frame(sessions, hour=CLOSE_HOUR_UTC + 1)
    assert macro_matrix(other_hour, spec=macro_spec()).get_column("session").to_list() == sessions


def test_a6_a_publication_exactly_at_the_close_counts() -> None:
    """``published_at == as_of(t)`` entra en la fila: la comparacion es ``<=``."""
    count = 4
    sessions = _sessions(count)
    frames = _universe(count)
    rows = [
        *_rows_of(frames["DFF"]),
        (sessions[1], _instant(sessions[1], CLOSE_HOUR_UTC), 7.0),
    ]
    matrix = macro_matrix({**frames, "DFF": _macro_frame(rows)}, spec=macro_spec())
    assert _column(matrix, "fed_funds")[1] == 7.0
    # una que se publique un microsegundo despues ya no entra
    rows_late = [
        *_rows_of(frames["DFF"]),
        (sessions[1], _instant(sessions[1], CLOSE_HOUR_UTC) + timedelta(microseconds=1), 7.0),
    ]
    late_matrix = macro_matrix({**frames, "DFF": _macro_frame(rows_late)}, spec=macro_spec())
    assert _column(late_matrix, "fed_funds")[1] != 7.0


def _rows_of(frame: pl.DataFrame) -> list[tuple[date, datetime | None, float | None]]:
    """Filas de un frame macro del fixture, como tuplas (para anadirle una)."""
    return [
        (cast("date", row[0]), cast("datetime", row[1]), cast("float", row[2]))
        for row in frame.iter_rows()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# A7 — DST
# ─────────────────────────────────────────────────────────────────────────────
def _transported_reference(series_id: str, instant: datetime) -> date:
    """Referencia transportada de una serie en un instante, tal y como la resuelve R.

    Es una sonda sobre el helper privado del modulo (la misma busqueda binaria que
    la matriz usa) y se cruza con la columna publicada: lo que se mide es la
    alineacion real, no una segunda implementacion.
    """
    frames = _golden_frames()
    series = macro._macro_series(frames[series_id], series_id=series_id)  # pyright: ignore[reportPrivateUsage]
    observation = macro._transported(series, instant)  # pyright: ignore[reportPrivateUsage]
    assert observation is not None
    return observation.reference


def _lag_profile(lo: date, hi: date) -> dict[str, dict[int, list[int]]]:
    """Rezago en **sesiones** del ancla, por dia de la semana, en una ventana.

    ``lag(t) = i(t) - i(ultima sesion del ancla <= referencia)``. Para las series
    macro la referencia sale de R; para el DXY, de la barra elegida (que es la del
    mismo dia).
    """
    matrix = _golden_matrix()
    sessions = [cast("date", value) for value in matrix.get_column("session").to_list()]
    instants = [cast("datetime", value) for value in matrix.get_column("as_of").to_list()]
    dxy = _golden_frames()[macro.DXY_SERIES]
    bar_instants = [cast("datetime", value) for value in dxy.get_column("as_of").to_list()]

    def lag(index: int, reference: date) -> int:
        return index - (bisect_right(sessions, reference) - 1)

    profile: dict[str, dict[int, list[int]]] = {}
    for name, series_id in (("fed_funds", "DFF"), ("ust_10y", "DGS10")):
        per_weekday: dict[int, set[int]] = {}
        for index, session in enumerate(sessions):
            if not (lo <= session <= hi):
                continue
            reference = _transported_reference(series_id, instants[index])
            # la columna publicada es, literalmente, el valor de esa observacion
            assert _column(matrix, name)[index] is not None
            per_weekday.setdefault(session.weekday(), set()).add(lag(index, reference))
        profile[name] = {day: sorted(values) for day, values in sorted(per_weekday.items())}

    dxy_days: dict[int, set[int]] = {}
    for index, session in enumerate(sessions):
        if not (lo <= session <= hi):
            continue
        bar = bar_instants[bisect_right(bar_instants, instants[index]) - 1]
        assert _column(matrix, "dxy")[index] == float(
            cast("float", dxy.get_column("close")[bar_instants.index(bar)])
        )
        dxy_days.setdefault(session.weekday(), set()).add(lag(index, bar.date()))
    profile["dxy"] = {day: sorted(values) for day, values in sorted(dxy_days.items())}
    return profile


def test_a7_the_dst_change_does_not_shift_the_lag() -> None:
    """Marzo (con el cambio del 2026-03-08) y enero dan el mismo rezago, sin corrimiento."""
    march = _lag_profile(date(2026, 3, 1), date(2026, 3, 31))
    january = _lag_profile(date(2026, 1, 1), date(2026, 1, 31))
    assert march == january
    # los tres perfiles son los medidos: fed_funds una sesion atras, los demas el mismo dia
    assert march["fed_funds"] == {0: [1], 1: [1], 2: [1], 3: [1], 4: [1]}
    assert march["ust_10y"] == {0: [0], 1: [0], 2: [0], 3: [0], 4: [0]}
    assert march["dxy"] == {0: [0], 1: [0], 2: [0], 3: [0], 4: [0]}

    # la DFF publica el dia natural **anterior**, en las dos ventanas
    for lo, hi in ((date(2026, 3, 1), date(2026, 3, 31)), (date(2026, 1, 1), date(2026, 1, 31))):
        matrix = _golden_matrix()
        sessions = [cast("date", value) for value in matrix.get_column("session").to_list()]
        instants = [cast("datetime", value) for value in matrix.get_column("as_of").to_list()]
        lags = {
            (session - _transported_reference("DFF", instants[index])).days
            for index, session in enumerate(sessions)
            if lo <= session <= hi
        }
        assert lags == {1}

    # marzo tiene las dos horas (13:00 UTC en EDT y 14:00 en EST); enero solo la de EST
    series = _golden_series().filter(pl.col("series_id") == "DFF")
    march_hours = series.filter(
        pl.col("published_at").dt.date().is_between(date(2026, 3, 1), date(2026, 3, 31))
    )
    january_hours = series.filter(
        pl.col("published_at").dt.date().is_between(date(2026, 1, 1), date(2026, 1, 31))
    )
    assert sorted(march_hours.get_column("published_at").dt.hour().unique().to_list()) == [13, 14]
    assert sorted(january_hours.get_column("published_at").dt.hour().unique().to_list()) == [14]


# ─────────────────────────────────────────────────────────────────────────────
# A8 — no invencion
# ─────────────────────────────────────────────────────────────────────────────
def _streaks(values: list[date]) -> list[int]:
    """Longitudes de las rachas de referencias iguales consecutivas."""
    lengths: list[int] = []
    current = 1
    for left, right in pairwise(values):
        if left == right:
            current += 1
            continue
        lengths.append(current)
        current = 1
    lengths.append(current)
    return lengths


def test_a8_the_gap_is_transported_and_never_interpolated() -> None:
    """En el hueco real (38 sesiones de CPI, 49 de PCE) el valor se transporta."""
    expected = _expected()
    matrix = _golden_matrix()
    instants = [cast("datetime", value) for value in matrix.get_column("as_of").to_list()]
    frames = _golden_frames()

    for name, series_id, streak, column in (
        ("CPI", "CPIAUCSL", 38, "cpi_yoy"),
        ("PCE", "PCEPI", 49, "pce_yoy"),
    ):
        series = macro._macro_series(frames[series_id], series_id=series_id)  # pyright: ignore[reportPrivateUsage]
        references: list[date] = []
        values: list[float] = []
        for instant in instants:
            observation = macro._transported(series, instant)  # pyright: ignore[reportPrivateUsage]
            assert observation is not None
            assert observation.published_at <= instant  # nunca una publicacion futura
            references.append(observation.reference)
            values.append(observation.value)
        assert max(_streaks(references)) == streak, name

        # la unica racha de esa longitud es la del hueco, y dentro de ella la
        # columna publicada es **constante** (transportada, no interpolada)
        start = 0
        best = 0
        for index, length in enumerate(_streaks(references)):
            if length > best:
                best, start = length, sum(_streaks(references)[:index])
        assert best == streak
        window = slice(start, start + streak)
        assert len(set(references[window])) == 1
        assert len(set(values[window])) == 1
        assert len(set(_column(matrix, column)[window])) == 1
        assert _column(matrix, column)[window][0] is not None

    assert expected["non_null"]["cpi_yoy"] == 0
    assert expected["non_null"]["pce_yoy"] == 0
    _assert_finite(matrix)


# ─────────────────────────────────────────────────────────────────────────────
# A9 — doble publicacion
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_a_double_publication_is_broken_by_the_greater_reference() -> None:
    """Dos referencias con el mismo ``published_at``: gana la de mayor referencia."""
    # sintetico: la 2025-03-01 y la 2025-04-01 se publican a la vez; las sesiones
    # van despues de esa publicacion, o ninguna seria candidata
    start = date(2025, 3, 11)
    stamp = _instant(date(2025, 3, 10), 13, 30)
    rows = [
        (date(2025, 1, 1), _instant(date(2025, 2, 10), 13, 30), 100.0),
        (date(2025, 2, 1), _instant(date(2025, 3, 1), 13, 30), 101.0),
        (date(2025, 3, 1), stamp, 102.0),
        (date(2025, 4, 1), stamp, 103.0),
    ]
    frames = _universe(4, start=start)
    matrix = macro_matrix({**frames, "CPIAUCSL": _macro_frame(rows)}, spec=macro_spec())
    # `m` es la mayor referencia publicada (2025-04-01) y `v_{m-12}` es 2024-04-01,
    # que no existe: la columna queda a `null` en vez de elegir la de marzo
    assert _column(matrix, "cpi_yoy") == [None] * 4

    # con la referencia de hace un ano presente, gana la mayor: 100 * (103/95 - 1)
    rows_with_base = [
        (date(2024, 4, 1), _instant(date(2024, 5, 10), 13, 30), 95.0),
        *rows,
    ]
    matrix = macro_matrix({**frames, "CPIAUCSL": _macro_frame(rows_with_base)}, spec=macro_spec())
    assert _column(matrix, "cpi_yoy") == [pytest.approx(100.0 * (103.0 / 95.0 - 1.0))] * 4
    assert _column(matrix, "cpi_yoy")[0] != pytest.approx(100.0 * (102.0 / 95.0 - 1.0))

    # y el dato real del golden: el 2026-01-22 el PCEPI publica dos referencias
    series = _golden_series().filter(pl.col("series_id") == "PCEPI")
    double = series.filter(pl.col("published_at") == _instant(date(2026, 1, 22), 13, 30)).sort(
        "as_of"
    )
    assert double.get_column("as_of").to_list() == [date(2025, 10, 1), date(2025, 11, 1)]
    assert double.height == 2
    base = series.filter(pl.col("as_of") == date(2024, 11, 1)).get_column("value").item()
    golden = _golden_matrix()
    sessions = [cast("date", value) for value in golden.get_column("session").to_list()]
    index = sessions.index(date(2026, 1, 22))
    expected = 100.0 * (128.093 / float(cast("float", base)) - 1.0)
    assert _column(golden, "pce_yoy")[index] == pytest.approx(expected, rel=1e-12)


def test_a9_a_persistent_tie_is_a_typed_error() -> None:
    """Mismo instante **y** misma referencia: el empate no se puede deshacer."""
    stamp = _instant(date(2025, 3, 10), 13, 30)
    rows = [
        (date(2025, 3, 1), stamp, 102.0),
        (date(2025, 3, 1), stamp, 103.0),
    ]
    frames = _universe(4)
    with pytest.raises(store.MacroInputError, match="el empate no se puede deshacer"):
        macro_matrix({**frames, "CPIAUCSL": _macro_frame(rows)}, spec=macro_spec())


# ─────────────────────────────────────────────────────────────────────────────
# A10 — ``_chg_5``
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_the_change_is_positional_and_in_points() -> None:
    """Las cuatro ``_chg_5`` son ``nivel(t) - nivel(t-5)``, en puntos porcentuales."""
    count = 12
    sessions = _sessions(count)
    frames = _universe(count)
    # una serie que sube un punto cada cuatro sesiones: el cambio a 5 sesiones no es 0
    levels = [4.0 + (index // 4) for index in range(count)]
    rows = [
        (
            session - timedelta(days=1),
            _instant(session - timedelta(days=1), PUBLISH_HOUR_UTC),
            4.0 + (index // 4),
        )
        for index, session in enumerate(sessions)
    ]
    matrix = macro_matrix({**frames, "DFF": _macro_frame(rows)}, spec=macro_spec())
    level = _column(matrix, "fed_funds")
    change = _column(matrix, "fed_funds_chg_5")
    assert level == [4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0]
    assert change[:5] == [None] * 5
    for index in range(5, count):
        previous = cast("float", level[index - 5])
        assert change[index] == pytest.approx(cast("float", level[index]) - previous)
    assert change[5:] == [pytest.approx(value) for value in (1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0)]
    # la punta anterior es la sesion ``t-5`` y **no** la ``t-4``: en la sesion 8 las
    # dos definiciones no coinciden (2.00 frente a 1.00)
    assert change[8] == pytest.approx(2.0)
    assert change[8] != pytest.approx(cast("float", level[8]) - cast("float", level[4]))
    assert levels[5] == 5.0

    # puntos porcentuales, **no** % relativo: de 4.00 a 5.00 son 1.00, no 25 %
    assert change[5] == pytest.approx(1.0)
    assert change[5] != pytest.approx(0.25)


def test_a10_the_first_five_sessions_have_no_change_and_the_rest_of_the_catalog() -> None:
    """Con el dato real, las cuatro ``_chg_5`` son la resta posicional y empiezan nulas."""
    expected = _expected()
    matrix = _golden_matrix()
    for level, change in (
        ("fed_funds", "fed_funds_chg_5"),
        ("ust_10y", "ust_10y_chg_5"),
        ("ust_2y", "ust_2y_chg_5"),
        ("pendiente_2s10s", "pendiente_2s10s_chg_5"),
    ):
        levels = _column(matrix, level)
        changes = _column(matrix, change)
        assert changes[:5] == [None] * 5
        assert expected["non_null"][change] == 5
        for index in range(5, matrix.height):
            assert changes[index] == pytest.approx(
                cast("float", levels[index]) - cast("float", levels[index - 5])
            )
        assert any(value is not None and value != 0.0 for value in changes)

    # `pendiente_2s10s` se toma de la serie `T10Y2Y` publicada, **no** de la resta
    # de los dos niveles: se demuestra mutando T10Y2Y y viendo que la columna lo sigue
    # mientras `ust_10y - ust_2y` no se mueve
    frames = _golden_frames()
    mutated = dict(frames)
    mutated["T10Y2Y"] = frames["T10Y2Y"].with_columns(pl.lit(9.99).alias("value"))
    changed = macro_matrix(mutated, spec=macro_spec())
    assert _column(changed, "pendiente_2s10s") == [pytest.approx(9.99)] * changed.height
    assert _column(changed, "ust_10y") == _column(matrix, "ust_10y")
    assert _column(changed, "ust_2y") == _column(matrix, "ust_2y")
    assert _column(changed, "pendiente_2s10s") != _column(matrix, "pendiente_2s10s")


# ─────────────────────────────────────────────────────────────────────────────
# A11 — inflacion
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_the_year_on_year_is_the_hand_computed_ratio() -> None:
    """``100 * (v_m / v_{m-12} - 1)`` con ``m`` la referencia vigente."""
    # las sesiones van despues de la ultima publicacion del fixture: si no, la
    # columna seria `null` por R y el test no probaria la formula
    start = date(2026, 2, 1)
    frames = _universe(6, start=start)
    rows = [
        (date(2024, 6, 1), _instant(date(2024, 7, 10), 12, 30), 95.0),
        (date(2024, 7, 1), _instant(date(2024, 7, 10), 12, 30), 96.0),
        (date(2025, 6, 1), _instant(date(2026, 2, 1), 12, 30), 100.0),
        # la de julio se publica a las 22:00 UTC del 5 de febrero: despues del cierre
        # de la sesion 4 (21:00 UTC) y antes del de la 5
        (date(2025, 7, 1), _instant(date(2026, 2, 5), 22, 0), 105.0),
    ]
    matrix = macro_matrix({**frames, "CPIAUCSL": _macro_frame(rows)}, spec=macro_spec())
    yoy = _column(matrix, "cpi_yoy")
    assert yoy[:5] == [pytest.approx(100.0 * (100.0 / 95.0 - 1.0))] * 5
    assert yoy[5] == pytest.approx(100.0 * (105.0 / 96.0 - 1.0))

    # el valor real del golden, calculado a mano desde el propio fixture
    series = _golden_series().filter(pl.col("series_id") == "CPIAUCSL")
    golden = _golden_matrix()
    instants = [cast("datetime", value) for value in golden.get_column("as_of").to_list()]
    sessions = [cast("date", value) for value in golden.get_column("session").to_list()]
    index = sessions.index(date(2026, 6, 1))
    reference = _transported_reference("CPIAUCSL", instants[index])
    previous = date(reference.year - 1, reference.month, reference.day)
    current = float(cast("float", series.filter(pl.col("as_of") == reference)["value"].item()))
    base = float(cast("float", series.filter(pl.col("as_of") == previous)["value"].item()))
    assert _column(golden, "cpi_yoy")[index] == pytest.approx(100.0 * (current / base - 1.0))
    assert current != base  # el fixture no es degenerado


def test_a11_a_missing_or_unpublished_base_is_a_null() -> None:
    """Sin la referencia de hace 12 meses, o sin publicar todavia, la tasa es ``null``."""
    frames = _universe(6)

    # la referencia de hace un ano no esta en la serie
    without_base = [
        (date(2025, 6, 1), _instant(date(2025, 7, 10), 12, 30), 100.0),
    ]
    matrix = macro_matrix({**frames, "PCEPI": _macro_frame(without_base)}, spec=macro_spec())
    assert _column(matrix, "pce_yoy") == [None] * 6

    # esta, pero se publica despues del cierre: tampoco entra
    unpublished = [
        (date(2024, 6, 1), _instant(date(2026, 1, 1), 12, 30), 95.0),
        (date(2025, 6, 1), _instant(date(2025, 7, 10), 12, 30), 100.0),
    ]
    matrix = macro_matrix({**frames, "PCEPI": _macro_frame(unpublished)}, spec=macro_spec())
    assert _column(matrix, "pce_yoy") == [None] * 6
    assert all(value is None or math.isfinite(value) for value in _column(matrix, "pce_yoy"))

    # base cero y el 29 de febrero (no tiene fecha exacta un ano antes)
    zero = [
        (date(2024, 6, 1), _instant(date(2024, 7, 1), 12, 30), 0.0),
        (date(2025, 6, 1), _instant(date(2025, 7, 1), 12, 30), 100.0),
    ]
    assert (
        _column(macro_matrix({**frames, "PCEPI": _macro_frame(zero)}, spec=macro_spec()), "pce_yoy")
        == [None] * 6
    )

    leap = [
        (date(2024, 2, 29), _instant(date(2024, 3, 10), 12, 30), 110.0),
    ]
    assert (
        _column(macro_matrix({**frames, "PCEPI": _macro_frame(leap)}, spec=macro_spec()), "pce_yoy")
        == [None] * 6
    )

    # un nulo de `value` o de `published_at` no es candidata, y no es un error
    ignored = [
        (date(2024, 6, 1), _instant(date(2025, 7, 1), 12, 30), 95.0),
        (date(2025, 6, 1), None, 100.0),
        (date(2025, 7, 1), _instant(date(2025, 8, 1), 12, 30), None),
    ]
    matrix = macro_matrix({**frames, "PCEPI": _macro_frame(ignored)}, spec=macro_spec())
    assert _column(matrix, "pce_yoy") == [None] * 6


# ─────────────────────────────────────────────────────────────────────────────
# A12 — el DXY
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_the_dxy_is_a_level_and_transports_the_last_close() -> None:
    """Nivel, no variacion; una sesion sin barra transporta el ultimo cierre."""
    count = 6
    sessions = _sessions(count)
    frames = _universe(count)
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]

    # sin la barra de la sesion 3, esa sesion repite el cierre de la 2
    without_third = _market_frame(
        [session for index, session in enumerate(sessions) if index != 3],
        closes=[close for index, close in enumerate(closes) if index != 3],
    )
    matrix = macro_matrix({**frames, macro.DXY_SERIES: without_third}, spec=macro_spec())
    dxy = _column(matrix, "dxy")
    assert dxy[2] == 102.0
    assert dxy[3] == 102.0  # transporta el ultimo cierre, no `null` ni 0.0
    assert dxy[4] == 104.0
    assert dxy == [100.0, 101.0, 102.0, 102.0, 104.0, 105.0]

    # es un nivel: no es el retorno (que seria ~0.0098) ni su variacion
    assert dxy[1] == 101.0
    assert dxy[1] != pytest.approx(math.log(101.0 / 100.0))
    assert dxy[1] != pytest.approx(1.0)
    _assert_finite(matrix)

    # sin ninguna barra candidata si es `null`
    empty = _market_frame([], closes=[])
    assert (
        _column(macro_matrix({**frames, macro.DXY_SERIES: empty}, spec=macro_spec()), "dxy")
        == [None] * count
    )

    # una barra con el cierre nulo no es barra: se transporta la anterior
    null_close = _market_frame(sessions, closes=[100.0, 101.0, None, 103.0, 104.0, 105.0])
    assert (
        _column(macro_matrix({**frames, macro.DXY_SERIES: null_close}, spec=macro_spec()), "dxy")[2]
        == 101.0
    )


def test_a12_the_dxy_is_the_close_of_the_bar_of_the_same_session() -> None:
    """Con el dato real, ``dxy(t)`` es el cierre de la barra de **esa** sesion."""
    market = _golden_market()
    matrix = _golden_matrix()
    dxy = _column(matrix, "dxy")
    assert dxy == [
        pytest.approx(float(cast("float", value)))
        for value in market.get_column("dxy_close").to_list()
    ]
    # y la igualdad de instantes cuenta: la barra del mismo dia se publica
    assert market.get_column("gspc_as_of").to_list() == market.get_column("dxy_as_of").to_list()
    # es nivel: la z de la sesion 250 se calcula sobre 250 niveles, no sobre retornos
    assert _column(matrix, "dxy_z")[MIN_SESSIONS - 1] is not None
    assert _column(matrix, "dxy_z")[MIN_SESSIONS - 2] is None
    _assert_finite(matrix)


MIN_SESSIONS = store.MACRO_MIN_SESSIONS


# ─────────────────────────────────────────────────────────────────────────────
# A13 — golden
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_the_macro_golden_is_frozen() -> None:
    """El par (``code_version``, digests) del golden no puede moverse en silencio."""
    expected = _expected()
    spec, matrix = _golden()

    assert (
        list(matrix.columns)
        == expected["feature_columns"]
        == [
            "session",
            "as_of",
            *store.MACRO_FEATURE_COLUMNS,
        ]
    )
    assert matrix.height == expected["sessions"] == 300
    assert _prefixed(store.feature_spec_sha256(spec)) == expected["feature_spec_sha256"]
    assert _prefixed(store.matrix_sha256(matrix)) == expected["matrix_sha256"]
    assert expected["code_version"] == store.FEATURE_CODE_VERSION
    assert spec == macro_spec()
    assert expected["feature_set"] == store.MACRO_FEATURE_SET
    assert expected["windows"] == store.DEFAULT_MACRO_WINDOWS
    assert expected["sources"] == [list(pair) for pair in store.DEFAULT_MACRO_SOURCES]
    assert len(expected["sources"]) == 8
    assert expected["digest_format"] == "sha256:<hex>"
    assert all(
        digest.startswith(store.FEATURE_VERSION_PREFIX)
        for digest in (expected["feature_spec_sha256"], expected["matrix_sha256"])
    )

    # `non_null` guarda el numero de **nulos** de cada columna (misma clave que #20/#21)
    counts = matrix.null_count().to_dicts()[0]
    assert counts == expected["non_null"]
    assert counts["fed_funds_chg_5"] == counts["ust_10y_chg_5"] == 5
    assert counts["ust_10y_z"] == counts["dxy_z"] == store.MACRO_MIN_SESSIONS - 1

    # el golden falla si el calculo se mueve sin subir la constante declarada
    assert _prefixed(store.matrix_sha256(matrix.drop("dxy"))) != expected["matrix_sha256"]
    assert _prefixed(store.matrix_sha256(matrix.drop("cpi_yoy"))) != expected["matrix_sha256"]


def test_a13_the_golden_inputs_are_the_declared_window() -> None:
    """Las 300 sesiones, las 6 series con lookback y el DXY con su cierre."""
    market = _golden_market()
    series = _golden_series()
    sessions = [cast("date", value) for value in market.get_column("session").to_list()]
    assert market.height == 300
    assert sessions[0] == date(2025, 7, 9)
    assert sessions[-1] == date(2026, 9, 16)
    assert sessions == sorted(sessions)
    assert market.get_column("dxy_close").null_count() == 0
    assert market.get_column("gspc_as_of").null_count() == 0

    assert set(series.get_column("series_id").to_list()) == set(store.MACRO_SERIES)
    assert series.get_column("as_of").min() == date(2024, 1, 1)
    assert "PAYEMS" not in set(series.get_column("series_id").to_list())
    assert series.get_column("published_at").null_count() == 0
    assert series.filter(pl.col("series_id") == "DFF").height == 989
    # marzo de 2026 trae las **dos** horas de publicacion (13:00 y 14:00 UTC)
    march = series.filter(
        (pl.col("series_id") == "DFF")
        & pl.col("published_at").dt.date().is_between(date(2026, 3, 1), date(2026, 3, 31))
    )
    assert sorted(march.get_column("published_at").dt.hour().unique().to_list()) == [13, 14]


def test_a13_the_previous_goldens_are_intact() -> None:
    """Los tres goldens anteriores se recomputan igual: unir una familia no los toca."""
    inputs = pl.read_csv(FIXTURES / "golden_inputs.csv", try_parse_dates=True)
    expected_19 = dict(json.loads((FIXTURES / "golden_expected.json").read_text(encoding="utf-8")))
    expected_20 = dict(
        json.loads((FIXTURES / "technical_golden_expected.json").read_text(encoding="utf-8"))
    )
    expected_21 = dict(
        json.loads((FIXTURES / "context_golden_expected.json").read_text(encoding="utf-8"))
    )

    spec_19 = _frozen_spec(expected_19)
    assert _prefixed(store.feature_spec_sha256(spec_19)) == FROZEN_19[0]
    assert _prefixed(store.matrix_sha256(store.build_matrix(inputs, spec=spec_19))) == FROZEN_19[1]

    spec_20 = _frozen_spec(expected_20)
    assert _prefixed(store.feature_spec_sha256(spec_20)) == FROZEN_20[0]
    assert (
        _prefixed(store.matrix_sha256(technical.technical_matrix(inputs, spec=spec_20)))
        == FROZEN_20[1]
    )

    spec_21 = _frozen_spec(expected_21)
    assert _prefixed(store.feature_spec_sha256(spec_21)) == FROZEN_21[0]
    context_matrix_21 = context.context_matrix(_context_frames(), spec=spec_21)
    assert _prefixed(store.matrix_sha256(context_matrix_21)) == FROZEN_21[1]
    assert store.FEATURE_CODE_VERSION == 1


def _context_frames() -> dict[str, pl.DataFrame]:
    """Las 19 series del golden de contexto, reconstruidas de su CSV."""
    table = pl.read_csv(
        FIXTURES / "context_golden_inputs.csv", try_parse_dates=True, infer_schema_length=None
    )
    frames: dict[str, pl.DataFrame] = {}
    for name in context.CONTEXT_SERIES:
        frame = table.select("session", pl.col(name).alias("close")).drop_nulls("close")
        if name == context.ANCHOR_SERIES:
            frame = frame.with_columns(
                pl.Series(
                    "as_of",
                    [
                        _instant(cast("date", session), CLOSE_HOUR_UTC)
                        for session in frame.get_column("session").to_list()
                    ],
                    dtype=pl.Datetime("us", "UTC"),
                )
            )
        frames[name] = frame
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# A14 — almacen, determinismo y puertas
# ─────────────────────────────────────────────────────────────────────────────
def _parquet_digest(root: Path) -> str:
    """Digest del **contenido** de los Parquet de un almacen, sin mirar los nombres."""
    files = sorted(root.rglob("*.parquet"))
    digests = sorted(hashlib.sha256(path.read_bytes()).hexdigest() for path in files)
    return hashlib.sha256("\n".join(digests).encode("utf-8")).hexdigest()


def _versions(root: Path, *, source: str) -> list[int]:
    """``version`` vigente de las filas de una familia, tal y como la ve ``sql()``."""
    frame = Store(root).sql(
        f"SELECT version FROM {store.FEATURES_LAYER}.{store.FEATURES_DATASET} "  # noqa: S608
        f"WHERE source = '{source}'"
    )
    return sorted(int(value) for value in frame.get_column("version").to_list())


def test_a14_the_family_is_written_and_read_back(tmp_path: Path) -> None:
    """``save_daily``/``load_daily`` con la familia, contra un ``Store`` en ``tmp_path``."""
    root = tmp_path / "store"
    handle = Store(root)
    count = 260
    matrix = _matrix(count)

    assert (
        store.save_daily(
            handle,
            spec=macro_spec(),
            matrix=matrix,
            series_id=macro.ANCHOR_SERIES,
            fetched_at=FETCHED_AT,
        )
        is WriteOutcome.CREATED
    )
    loaded = store.load_daily(handle, series_id=macro.ANCHOR_SERIES, feature_set="macro_v1")
    assert loaded.height == count
    assert set(loaded.get_column("source").to_list()) == {MACRO_SOURCE}
    for name in store.MACRO_FEATURE_COLUMNS:
        assert loaded.get_column(name).is_not_null().any()
    assert _versions(root, source=MACRO_SOURCE) == [1] * count

    # una familia registrada sin filas en un dataset que si existe no devuelve vacio
    with pytest.raises(store.UnknownDatasetError, match="no tiene filas de la familia"):
        store.load_daily(handle, series_id=macro.ANCHOR_SERIES, feature_set="technical_v1")

    records = store.daily_records(
        matrix, spec=macro_spec(), series_id=macro.ANCHOR_SERIES, fetched_at=FETCHED_AT
    )
    for record in records:
        assert "version" not in record
        assert record["fetched_at"] is FETCHED_AT
        assert record["source"] == MACRO_SOURCE
        assert record["published_at"] is None
        assert record["features_version"] == store.features_version(
            macro_spec(), cast("datetime", record["as_of"])
        )


def test_a14_unchanged_is_free_and_a_change_bumps_the_revision(tmp_path: Path) -> None:
    """Contenido identico ⇒ ``UNCHANGED`` sin Parquet nuevo; distinto ⇒ ``version + 1``."""
    root = tmp_path / "store"
    handle = Store(root)
    spec = macro_spec()
    matrix = _matrix(40)

    def save(current: store.FeatureSpec, *, fetched_at: datetime) -> WriteOutcome:
        return store.save_daily(
            handle,
            spec=current,
            matrix=matrix,
            series_id=macro.ANCHOR_SERIES,
            fetched_at=fetched_at,
        )

    assert save(spec, fetched_at=FETCHED_AT) is WriteOutcome.CREATED
    before = _parquet_digest(root)
    files = sorted(root.rglob("*.parquet"))
    assert save(spec, fetched_at=FETCHED_AT + timedelta(days=1)) is WriteOutcome.UNCHANGED
    assert _parquet_digest(root) == before
    assert sorted(root.rglob("*.parquet")) == files
    assert _versions(root, source=MACRO_SOURCE) == [1] * 40

    bumped = store.FeatureSpec(
        feature_set=store.MACRO_FEATURE_SET,
        code_version=spec.code_version + 1,
        windows=store.DEFAULT_MACRO_WINDOWS,
        sources=store.DEFAULT_MACRO_SOURCES,
    )
    assert save(bumped, fetched_at=FETCHED_AT) is WriteOutcome.CREATED
    assert _versions(root, source=MACRO_SOURCE) == [2] * 40
    assert _parquet_digest(root) != before


def test_a14_the_module_does_not_cheat_the_gates() -> None:
    """La cobertura no se maquilla: sin ``pragma``, sin ``type: ignore``."""
    source = _source()
    assert "pragma: no cover" not in source
    assert "type: ignore" not in source
    assert "coverage" not in source.lower()
    assert macro_matrix.__module__ == "cfdtrader.features.macro"
    assert "\t" not in source


_DIGEST_SCRIPT = """
import hashlib, json, pathlib, sys
from datetime import UTC, datetime

import polars as pl

from cfdtrader.features import macro, store

root, market_path, series_path = sys.argv[1], sys.argv[2], sys.argv[3]
dt = "%Y-%m-%dT%H:%M:%SZ"
market = pl.read_csv(
    market_path,
    schema_overrides={"session": pl.String, "gspc_as_of": pl.String,
                      "dxy_as_of": pl.String, "dxy_close": pl.String},
).with_columns(
    pl.col("session").str.to_date(),
    pl.col("gspc_as_of").str.to_datetime(format=dt, time_zone="UTC"),
    pl.col("dxy_as_of").str.to_datetime(format=dt, time_zone="UTC"),
    pl.col("dxy_close").cast(pl.Float64),
)
series = pl.read_csv(
    series_path,
    schema_overrides={"series_id": pl.String, "as_of": pl.String,
                      "published_at": pl.String, "value": pl.String},
).with_columns(
    pl.col("as_of").str.to_date(),
    pl.col("published_at").str.to_datetime(format=dt, time_zone="UTC"),
    pl.col("value").cast(pl.Float64),
)
frames = {
    name: series.filter(pl.col("series_id") == name).select("as_of", "published_at", "value")
    for name in store.MACRO_SERIES
}
frames["^GSPC"] = market.select("session", pl.col("gspc_as_of").alias("as_of"))
frames["DX-Y.NYB"] = market.drop_nulls("dxy_as_of").select(
    "session", pl.col("dxy_as_of").alias("as_of"), pl.col("dxy_close").alias("close")
)

spec = macro.macro_spec()
matrix = macro.macro_matrix(frames, spec=spec)
handle = store.Store(root)
store.save_daily(
    handle,
    spec=spec,
    matrix=matrix,
    series_id="^GSPC",
    fetched_at=datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
)
rows = store.load_daily(handle, series_id="^GSPC", feature_set="macro_v1")
digests = sorted(
    hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(pathlib.Path(root).rglob("*.parquet"))
)
print(json.dumps({
    "feature_spec_sha256": store.feature_spec_sha256(spec),
    "matrix_sha256": store.matrix_sha256(matrix),
    "features_version": rows.get_column("features_version")[0],
    "source": rows.get_column("source")[0],
    "parquet_sha256": hashlib.sha256("\\n".join(digests).encode("utf-8")).hexdigest(),
}))
"""


def _subprocess_digests(seed: str, root: Path) -> dict[str, Any]:
    """Ejecuta el calculo y el guardado completos con ese ``PYTHONHASHSEED``."""
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = seed
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(  # noqa: S603 - el ejecutable es el interprete de la sesion
        [
            sys.executable,
            "-c",
            _DIGEST_SCRIPT,
            str(root),
            str(FIXTURES / "macro_golden_market.csv"),
            str(FIXTURES / "macro_golden_series.csv"),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return dict(json.loads(result.stdout))


def test_a14_determinism_across_processes_and_second_pass(tmp_path: Path) -> None:
    """Mismo input ⇒ mismo digest, mismas ``features_version`` y Parquet byte a byte."""
    digests = [
        _subprocess_digests(seed, tmp_path / f"seed_{seed}") for seed in ("0", "1", "random")
    ]
    assert digests[0] == digests[1] == digests[2]

    spec, matrix = _golden()
    assert digests[0]["feature_spec_sha256"] == store.feature_spec_sha256(spec)
    assert digests[0]["matrix_sha256"] == store.matrix_sha256(matrix)
    assert digests[0]["feature_spec_sha256"] == (
        _expected()["feature_spec_sha256"].removeprefix(store.FEATURE_VERSION_PREFIX)
    )
    assert digests[0]["matrix_sha256"] == (
        _expected()["matrix_sha256"].removeprefix(store.FEATURE_VERSION_PREFIX)
    )
    assert digests[0]["source"] == MACRO_SOURCE

    first, second = tmp_path / "pass_1", tmp_path / "pass_2"
    for root in (first, second):
        store.save_daily(
            Store(root),
            spec=spec,
            matrix=matrix,
            series_id=macro.ANCHOR_SERIES,
            fetched_at=FETCHED_AT,
        )
    assert _parquet_digest(first) == _parquet_digest(second) == digests[0]["parquet_sha256"]
    loaded = store.load_daily(Store(first), series_id=macro.ANCHOR_SERIES, feature_set="macro_v1")
    assert loaded.get_column("features_version")[0] == digests[0]["features_version"]
