"""Tests de la familia de regimen y volatilidad (`regime_v1`, tarea #23).

Un test por criterio, ``test_aN_...``. La raiz del ``Store`` vive siempre bajo
``tmp_path``: la fixture de sesion de ``tests/conftest.py`` huella el ``data/`` y
el ``runs/`` del repositorio y falla si la sesion escribe en ellos.

El *golden dataset* de A12 vive en ``tests/fixtures/features/``:
``regime_golden_inputs.csv`` (800 sesiones reales, 2023-07-11 -> 2026-09-16, con
precios sinteticos deterministas y agrupamiento de volatilidad) y el par esperado
en ``regime_golden_expected.json``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

from cfdtrader.analysis import volatility_forecast
from cfdtrader.data.store import Store, UnknownDatasetError, WriteOutcome
from cfdtrader.features import regime, store
from cfdtrader.features import volatility as volatility_module
from cfdtrader.features.regime import regime_matrix, regime_spec
from cfdtrader.features.volatility import PARKINSON_DENOMINATOR

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "features"

#: Primera sesion sintetica. El modulo no conoce el calendario a proposito.
FIRST_SESSION = date(2025, 1, 2)

#: Hora UTC que el test usa como cierre de sesion. Es un valor **del test**.
CLOSE_HOUR_UTC = 20

#: Instante de captura de los tests: posterior a todas las sesiones usadas.
FETCHED_AT = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

#: Instante de captura del golden: tiene que ser posterior al cierre de 2026-09-16.
GOLDEN_FETCHED_AT = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)

#: ``source`` de la familia de regimen (el discriminador de familia, decision 2 de #20).
REGIME_SOURCE = store.REGIME_FEATURES_SOURCE

#: Sesiones minimas de las ventanas expandidas (percentil y z-score robusto).
MIN_SESSIONS = store.REGIME_MIN_SESSIONS

#: Columnas que **no** son de esta familia: una sola basta para probar el rechazo.
FOREIGN_COLUMNS = ("har_forecast", "vix_percentile", "ret_1")

#: La tabla de la issue, literal: ventana y ``required_as_of`` de cada entrada.
CATALOG_TABLE: dict[str, tuple[int | None, str]] = {
    "rv_percentile": (MIN_SESSIONS, "cierre de la sesion t-1"),
    "garch_forecast": (volatility_module.GARCH_MIN_TRAIN, "cierre de la sesion t-1"),
    "garch_forecast_z": (MIN_SESSIONS, "cierre de la sesion t-1"),
    "efficiency_ratio_20": (store.REGIME_EFFICIENCY_WINDOW, "cierre de la sesion t"),
    "day_of_week": (None, "de antemano"),
    "sessions_to_opex": (None, "de antemano"),
    "is_es_roll_session": (None, "de antemano"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def _sessions(count: int) -> list[date]:
    """``count`` sesiones consecutivas desde la primera del test."""
    return [FIRST_SESSION + timedelta(days=index) for index in range(count)]


def _frame(count: int = 40, *, offset: int = 0) -> pl.DataFrame:
    """Frame sintetico: ``session`` + OHLC coherente (``low < close < high``).

    ``offset`` desplaza la rejilla de precios sin mover las fechas: sirve para
    construir una serie con historia larga y agrupamiento suave de volatilidad.
    """
    closes = [
        100.0 + (index % 17) * 0.7 - (index % 5) * 0.3 + offset * math.sin(index / 3.0)
        for index in range(count)
    ]
    return pl.DataFrame(
        {
            "session": _sessions(count),
            "open": [value * 1.001 for value in closes],
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
        }
    )


def _noise(seed: str, index: int) -> float:
    """Uniforme determinista en ``[0, 1)`` derivado de ``sha256`` (sin RNG global).

    Un generador congruencial lineal con dos semillas distintas produce la misma
    secuencia desplazada una constante (trampa medida en #21): el ruido de estos
    tests es el hash del indice, no un LCG.
    """
    digest = hashlib.sha256(f"{seed}:{index}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _long_frame(count: int = 300) -> pl.DataFrame:
    """Frame largo con retornos **GARCH-like**: el GARCH tiene algo que ajustar.

    Los precios son sinteticos y deterministas, pero con agrupamiento de
    volatilidad real (recursion ``sigma2 = 1e-6 + 0.08 r^2 + 0.88 sigma2``): una
    serie de varianza constante no probaria que el pronostico se mueve.
    """
    closes: list[float] = []
    opens: list[float] = []
    spans: list[float] = []
    close = 100.0
    sigma2 = 1.0e-4
    for index in range(count):
        shock = 2.0 * _noise("regime-long", index) - 1.0
        ret = math.sqrt(sigma2) * shock
        sigma2 = 1.0e-6 + 0.08 * ret * ret + 0.88 * sigma2
        previous = close
        close = previous * math.exp(ret)
        closes.append(close)
        opens.append(previous)
        spans.append(0.5 * math.sqrt(sigma2) * (0.5 + _noise("regime-span", index)))
    return pl.DataFrame(
        {
            "session": _sessions(count),
            "open": opens,
            "high": [
                max(open_px, close_px) * math.exp(span)
                for open_px, close_px, span in zip(opens, closes, spans, strict=True)
            ],
            "low": [
                min(open_px, close_px) / math.exp(span)
                for open_px, close_px, span in zip(opens, closes, spans, strict=True)
            ],
            "close": closes,
        }
    )


def _session_close(session: date) -> datetime:
    """Cierre de sesion en UTC de una sesion sintetica."""
    return datetime(session.year, session.month, session.day, CLOSE_HOUR_UTC, tzinfo=UTC)


def _with_as_of(matrix: pl.DataFrame) -> pl.DataFrame:
    """Anade ``as_of`` (el cierre de sesion en UTC de **cada fila**) a una matriz."""
    sessions = [cast("date", value) for value in matrix.get_column("session").to_list()]
    return matrix.with_columns(
        pl.Series("as_of", [_session_close(session) for session in sessions])
    )


def _matrix(count: int = 40) -> pl.DataFrame:
    """Matriz de regimen persistible: ``session`` + ``as_of`` + las siete features."""
    return _with_as_of(regime_matrix(_frame(count), spec=regime_spec()))


def _volatility_matrix(count: int = 40) -> pl.DataFrame:
    """Matriz de la **otra** familia, para probar la convivencia en el mismo dataset."""
    frame = _frame(count).with_columns(
        pl.Series("vix_close", [20.0 + (index % 11) * 0.5 for index in range(count)])
    )
    return _with_as_of(store.build_matrix(frame, spec=store.FeatureSpec()))


def _prefixed(digest: str) -> str:
    """Forma en la que el golden guarda un digest: con el prefijo ``sha256:``.

    El mismo prefijo que publica ``features_version``; ademas, un hex desnudo de 64
    caracteres dispara el hook ``detect-secrets`` (falso positivo de alta entropia).
    """
    return store.FEATURE_VERSION_PREFIX + digest


def _expected() -> dict[str, Any]:
    """Par esperado congelado de A12."""
    raw = (FIXTURES / "regime_golden_expected.json").read_text(encoding="utf-8")
    return dict(json.loads(raw))


def _frozen_spec(expected: dict[str, Any]) -> store.FeatureSpec:
    """Spec exacta que declara el golden: reconstruirla es parte del congelado."""
    return store.FeatureSpec(
        feature_set=expected["feature_set"],
        code_version=expected["code_version"],
        parameters=expected["parameters"],
        windows=expected["windows"],
        sources=tuple((str(left), str(right)) for left, right in expected["sources"]),
    )


def _golden_frame() -> pl.DataFrame:
    """Inputs del golden: las 800 sesiones reales con precios sinteticos."""
    return pl.read_csv(FIXTURES / "regime_golden_inputs.csv", try_parse_dates=True)


def _golden() -> tuple[store.FeatureSpec, pl.DataFrame]:
    """Spec y matriz del golden de regimen, tal y como los congela A12."""
    spec = _frozen_spec(_expected())
    return spec, regime_matrix(_golden_frame(), spec=spec)


def _source() -> str:
    """Codigo fuente del modulo, del fichero real (no del bytecode)."""
    return Path(regime.__file__).read_text(encoding="utf-8")


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


def _column(matrix: pl.DataFrame, name: str) -> list[float | None]:
    """Columna de features como lista de floats con nulos."""
    return [
        None if value is None else float(cast("float", value))
        for value in cast("list[object]", matrix.get_column(name).to_list())
    ]


def _int_column(matrix: pl.DataFrame, name: str) -> list[int | None]:
    """Columna entera (las tres de calendario) con nulos como ``None``."""
    return [
        None if value is None else int(cast("int", value))
        for value in cast("list[object]", matrix.get_column(name).to_list())
    ]


def _parkinson(frame: pl.DataFrame) -> list[float]:
    """``rv_t = (ln(H_t/L_t))² / (4·ln 2)`` calculado a mano, sin llamar al modulo."""
    highs = cast("list[float]", frame.get_column("high").to_list())
    lows = cast("list[float]", frame.get_column("low").to_list())
    return [
        math.log(h / low) ** 2 / PARKINSON_DENOMINATOR for h, low in zip(highs, lows, strict=True)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# A1 — modulo, firma y pureza
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_matrix_is_session_plus_the_catalog_in_order() -> None:
    """``regime_matrix`` publica ``session`` mas las siete columnas, en orden."""
    matrix = regime_matrix(_frame(40), spec=regime_spec())
    assert matrix.columns == ["session", *store.REGIME_FEATURE_COLUMNS]
    assert matrix.height == 40
    assert matrix.get_column("session").to_list() == _sessions(40)

    shuffled = regime_matrix(_frame(40).reverse(), spec=regime_spec())
    assert store.matrix_sha256(shuffled) == store.matrix_sha256(matrix)

    with pytest.raises(store.InvalidFeatureMatrixError, match="es la entrada de"):
        regime_matrix(_frame(40), spec=store.FeatureSpec())
    # y la matriz de volatilidad no admite la spec de regimen (la otra direccion)
    with pytest.raises(store.InvalidFeatureMatrixError, match="es la entrada de"):
        store.build_matrix(_frame(40), spec=regime_spec())


def test_a1_the_five_input_columns_are_required_and_the_error_names_the_missing_one() -> None:
    """Falta una de las cinco columnas de entrada ⇒ ``RegimeInputError`` que la nombra."""
    for name in regime.REGIME_INPUT_COLUMNS:
        with pytest.raises(store.RegimeInputError, match=name):
            regime_matrix(_frame(40).drop(name), spec=regime_spec())
    # una columna de texto es un error de entrada tipado, no un fallo de polars
    broken = _frame(40).with_columns(pl.Series("low", ["x"] * 40))
    with pytest.raises(store.RegimeInputError, match="no es numerica"):
        regime_matrix(broken, spec=regime_spec())


def test_a1_the_module_is_pure() -> None:
    """Pureza por AST: sin almacen, red, disco ni reloj; y sin reimplementar #7."""
    forbidden = ("cfdtrader.data", "duckdb", "httpx", "pathlib", "os", "shutil", "socket")
    imported = _imported_modules()
    assert imported, "el modulo debe declarar sus imports de forma explicita"
    for module in sorted(imported):
        assert module.split(".")[0] not in forbidden, f"import prohibido: {module!r}"

    source = _source()
    for forbidden_call in ("datetime.now", "utcnow", "date.today", "time.time", "time_ns"):
        assert forbidden_call not in source
    defined = {node.name for node in ast.walk(_tree()) if isinstance(node, ast.FunctionDef)}
    assert not defined & {
        "parkinson_variance",
        "session_returns",
        "normalise_expanding",
        "fit_garch",
        "garch_one_step_forecast",
        "garch_forecasts",
        "garch_fold_bounds",
        "_expanding_percentile",
    }


# ─────────────────────────────────────────────────────────────────────────────
# A2 — registro
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_registry_declares_the_fifth_family() -> None:
    """El registro expone las cinco familias y la spec de regimen se construye."""
    assert sorted(store.CATALOG_BY_FEATURE_SET) == [
        "context_v1",
        "macro_v1",
        "regime_v1",
        "technical_v1",
        "volatility_v1",
    ]
    assert store.CATALOG_BY_FEATURE_SET["regime_v1"] is store.REGIME_FEATURE_CATALOG
    assert store.SOURCE_BY_FEATURE_SET["regime_v1"] == REGIME_SOURCE
    assert REGIME_SOURCE not in {
        store.FEATURES_SOURCE,
        store.TECHNICAL_FEATURES_SOURCE,
        store.CONTEXT_FEATURES_SOURCE,
        store.MACRO_FEATURES_SOURCE,
    }
    assert set(store.SOURCE_BY_FEATURE_SET) == set(store.CATALOG_BY_FEATURE_SET)

    spec = regime_spec()
    assert spec.feature_set == store.REGIME_FEATURE_SET
    assert spec.code_version == store.FEATURE_CODE_VERSION
    assert spec.windows == store.DEFAULT_REGIME_WINDOWS
    assert spec.sources == store.DEFAULT_REGIME_SOURCES
    assert (
        store.FeatureSpec(
            feature_set="regime_v1",
            windows=store.DEFAULT_REGIME_WINDOWS,
            sources=store.DEFAULT_REGIME_SOURCES,
        )
        == spec
    )
    assert store.FEATURE_CODE_VERSION == 1


# ─────────────────────────────────────────────────────────────────────────────
# A3 — catalogo
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_the_catalog_is_exactly_the_seven_declared_entries() -> None:
    """Nombre, ventana y ``required_as_of`` de las siete entradas, una por una."""
    assert [entry.name for entry in store.REGIME_FEATURE_CATALOG] == list(CATALOG_TABLE)
    assert tuple(CATALOG_TABLE) == store.REGIME_FEATURE_COLUMNS
    assert store.REGIME_MIN_SESSIONS == 250
    assert store.REGIME_EFFICIENCY_WINDOW == 20
    for entry in store.REGIME_FEATURE_CATALOG:
        window, required = CATALOG_TABLE[entry.name]
        assert entry.window == window
        assert entry.required_as_of == required
        assert entry.formula.strip() == entry.formula
        assert entry.formula and entry.source == "raw.market_daily"
    assert {entry.name: entry.window for entry in store.REGIME_FEATURE_CATALOG} == (
        store.DEFAULT_REGIME_WINDOWS
    )


def test_a3_a_bad_window_or_a_foreign_name_is_a_typed_error() -> None:
    """Ventana contradictoria (``day_of_week: 5``) y nombre de otra familia."""
    with pytest.raises(store.InvalidFeatureSpecError, match="no coincide"):
        store.FeatureSpec(
            feature_set="regime_v1", windows={**store.DEFAULT_REGIME_WINDOWS, "day_of_week": 5}
        )
    with pytest.raises(store.InvalidFeatureSpecError, match="no esta en el catalogo"):
        store.FeatureSpec(
            feature_set="regime_v1", windows={**store.DEFAULT_REGIME_WINDOWS, "ret_1": 1}
        )
    with pytest.raises(store.InvalidFeatureSpecError, match="no esta en el catalogo"):
        store.FeatureSpec(
            feature_set="regime_v1", windows={**store.DEFAULT_REGIME_WINDOWS, "vix_percentile": 250}
        )


# ─────────────────────────────────────────────────────────────────────────────
# A4 — nombres
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_the_names_do_not_collide_between_the_five_families() -> None:
    """Ninguna de las 7 esta en los otros cuatro catalogos y el unico solape sigue igual."""
    regimes = set(store.REGIME_FEATURE_COLUMNS)
    assert len(regimes) == 7
    assert regimes & set(store.FEATURE_COLUMNS) == set()
    assert regimes & set(store.TECHNICAL_FEATURE_COLUMNS) == set()
    assert regimes & set(store.CONTEXT_FEATURE_COLUMNS) == set()
    assert regimes & set(store.MACRO_FEATURE_COLUMNS) == set()
    assert "rv_percentile" not in store.FEATURE_COLUMNS
    assert "vix_percentile" in store.FEATURE_COLUMNS
    assert "garch_forecast" not in store.FEATURE_COLUMNS
    assert "har_forecast" in store.FEATURE_COLUMNS

    union = (
        set(store.FEATURE_COLUMNS)
        | set(store.TECHNICAL_FEATURE_COLUMNS)
        | set(store.CONTEXT_FEATURE_COLUMNS)
        | set(store.MACRO_FEATURE_COLUMNS)
        | regimes
    )
    assert set(store.ALL_FEATURE_COLUMNS) == union
    assert len(store.ALL_FEATURE_COLUMNS) == len(union)
    # el unico repetido entre las cinco familias sigue siendo `atr_norm` (#72)
    assert len(store.ALL_FEATURE_COLUMNS) == (
        len(store.FEATURE_COLUMNS)
        + len(store.TECHNICAL_FEATURE_COLUMNS)
        + len(store.CONTEXT_FEATURE_COLUMNS)
        + len(store.MACRO_FEATURE_COLUMNS)
        + 7
        - 1
    )


# ─────────────────────────────────────────────────────────────────────────────
# A5 — acoplamiento congelado: los SEIS sitios de una linea
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_the_frozen_coupling_is_resolved_with_the_regime_columns() -> None:
    """Los seis sitios congelados de #20/#21/#22, re-verificados desde la familia nueva.

    Registrar ``regime_v1`` rompe **seis** aserciones de las suites anteriores
    (medido parcheando el registro: ``6 failed, 117 passed``) y cada una se edita
    con **una** linea semantica: la lista esperada de
    ``test_technical_features.py::test_a1`` y de ``test_context_features.py::test_a1``,
    la union de ``ALL_FEATURE_COLUMNS`` de ``test_context_features.py::test_a9``, y
    la lista, el conteo de ``sources``, la union, la aritmetica y los docstrings de
    ``test_macro_features.py::test_a2``, ``::test_a4`` y ``::test_a5`` (``ruff
    format`` refluye el literal de cinco familias a varias lineas fisicas: es
    esperado). Este test re-verifica los hechos: si alguno de esos ficheros se
    hubiera "arreglado" de otra forma, aqui se ve.
    """
    assert sorted(store.CATALOG_BY_FEATURE_SET) == [
        "context_v1",
        "macro_v1",
        "regime_v1",
        "technical_v1",
        "volatility_v1",
    ]
    assert store.CATALOG_BY_FEATURE_SET["technical_v1"] is store.TECHNICAL_FEATURE_CATALOG
    assert store.CATALOG_BY_FEATURE_SET["context_v1"] is store.CONTEXT_FEATURE_CATALOG
    assert store.CATALOG_BY_FEATURE_SET["macro_v1"] is store.MACRO_FEATURE_CATALOG
    assert store.CATALOG_BY_FEATURE_SET["volatility_v1"] is store.FEATURE_CATALOG
    assert len(set(store.SOURCE_BY_FEATURE_SET.values())) == 5
    assert len(store.FEATURE_COLUMNS) == 12
    assert len(store.TECHNICAL_FEATURE_COLUMNS) == 10
    assert len(store.CONTEXT_FEATURE_COLUMNS) == 11
    assert len(store.MACRO_FEATURE_COLUMNS) == 13

    union = (
        set(store.FEATURE_COLUMNS)
        | set(store.TECHNICAL_FEATURE_COLUMNS)
        | set(store.CONTEXT_FEATURE_COLUMNS)
        | set(store.MACRO_FEATURE_COLUMNS)
        | set(store.REGIME_FEATURE_COLUMNS)
    )
    assert len(store.ALL_FEATURE_COLUMNS) == len(union)


# ─────────────────────────────────────────────────────────────────────────────
# A6 — PIT por columna
# ─────────────────────────────────────────────────────────────────────────────
def _mutated(frame: pl.DataFrame, *, column: str, index: int, factor: float) -> pl.DataFrame:
    """Copia del frame con una celda de precio multiplicada."""
    values = [float(value) for value in frame.get_column(column).to_list()]
    values[index] *= factor
    return frame.with_columns(pl.Series(column, values))


def test_a6_the_row_of_t_reads_the_session_t_only_where_the_catalog_says_so() -> None:
    """Mutar el cierre de ``t`` mueve **solo** el *efficiency ratio* de ``t``."""
    frame = _long_frame(1000)
    base = regime_matrix(frame, spec=regime_spec())
    target = 990
    mutated = regime_matrix(
        _mutated(frame, column="close", index=target, factor=1.4), spec=regime_spec()
    )

    # las tres de volatilidad usan sesiones `< t`: la fila mutada no las toca
    for name in ("rv_percentile", "garch_forecast", "garch_forecast_z"):
        assert _column(mutated, name)[: target + 1] == _column(base, name)[: target + 1], name
    assert (
        _column(mutated, "efficiency_ratio_20")[target]
        != _column(base, "efficiency_ratio_20")[target]
    )
    # y la sesion siguiente (`t+1`) si nota la mutacion: `ret_log` lee `close` de `t`
    for name in ("garch_forecast", "garch_forecast_z", "efficiency_ratio_20"):
        assert _column(mutated, name)[target + 1] != _column(base, name)[target + 1], name


def test_a6_mutating_any_price_of_t_plus_one_does_not_move_the_row_of_t() -> None:
    """Ninguna de las 7 columnas de ``t`` depende de una sesion posterior."""
    frame = _long_frame(1000)
    base = regime_matrix(frame, spec=regime_spec())
    for column in ("open", "high", "low", "close"):
        mutated = _mutated(frame, column=column, index=991, factor=1.25)
        shifted = regime_matrix(mutated, spec=regime_spec())
        for name in store.REGIME_FEATURE_COLUMNS:
            assert _column(shifted, name)[:991] == _column(base, name)[:991], (column, name)


def test_a6_mutating_the_session_before_moves_the_volatility_columns() -> None:
    """Mutar la sesion ``t-1`` **si** mueve las columnas que la leen.

    Cada columna lee su parte del OHLC: ``rv_percentile`` sale de ``high``/``low``
    de ``t-1`` (Parkinson no toca el ``close``), y ``garch_forecast`` de
    ``ret_log = ln(C/O)`` hasta ``t-1``. Con la mutacion de las **tres** columnas a
    la vez —que es lo que dice A6— se mueven las tres de volatilidad.
    """
    frame = _long_frame(1000)
    base = regime_matrix(frame, spec=regime_spec())
    target = 990
    for column, factor, moved in (
        ("close", 1.3, ("garch_forecast", "garch_forecast_z")),
        ("high", 1.4, ("rv_percentile",)),
        ("low", 0.7, ("rv_percentile",)),
    ):
        shifted = regime_matrix(
            _mutated(frame, column=column, index=target - 1, factor=factor), spec=regime_spec()
        )
        for name in moved:
            assert _column(shifted, name)[target] != _column(base, name)[target], (column, name)
        # y la calendar es lo unico que no se mueve nunca
        for name in ("day_of_week", "sessions_to_opex", "is_es_roll_session"):
            assert _column(shifted, name) == _column(base, name), name

    # mutando las tres a la vez (lo que pide A6 literalmente) se mueven **las tres**
    mutated = frame
    for column, factor in (("close", 1.3), ("high", 1.4), ("low", 0.7)):
        mutated = _mutated(mutated, column=column, index=target - 1, factor=factor)
    shifted = regime_matrix(mutated, spec=regime_spec())
    for name in ("rv_percentile", "garch_forecast", "garch_forecast_z"):
        assert _column(shifted, name)[target] != _column(base, name)[target], name


# ─────────────────────────────────────────────────────────────────────────────
# A7 — rv_percentile
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_rv_percentile_is_hand_computed() -> None:
    """Fraccion de las RV anteriores a ``t-1`` que no superan la de ``t-1``."""
    count = 300
    frame = _long_frame(count)
    column = _column(regime_matrix(frame, spec=regime_spec()), "rv_percentile")
    realised = _parkinson(frame)

    for index in range(count):
        if index <= MIN_SESSIONS:
            # el percentil de `t-1` necesita 250 RV **estrictamente anteriores** a `t-1`
            assert column[index] is None, index
            continue
        history = realised[: index - 1]
        assert column[index] == pytest.approx(
            sum(1 for value in history if value <= realised[index - 1]) / len(history), rel=1e-12
        )
        assert 0.0 <= cast("float", column[index]) <= 1.0
    assert sum(1 for value in column if value is None) == MIN_SESSIONS + 1


def test_a7_a_bigger_realised_volatility_cannot_lower_the_percentile() -> None:
    """Monotonia: subir la RV de ``t-1`` no puede bajar el percentil de ``t``."""
    frame = _long_frame(300)
    base = _column(regime_matrix(frame, spec=regime_spec()), "rv_percentile")
    for factor in (1.2, 1.5, 3.0):
        risen = regime_matrix(
            _mutated(frame, column="high", index=289, factor=factor), spec=regime_spec()
        )
        assert cast("float", _column(risen, "rv_percentile")[300 - 10]) >= cast(
            "float", base[300 - 10]
        )


def test_a7_no_later_session_changes_a_row() -> None:
    """El percentil de una sesion no mira ninguna RV posterior."""
    frame = _long_frame(300)
    base = _column(regime_matrix(frame, spec=regime_spec()), "rv_percentile")
    shifted = _column(
        regime_matrix(_mutated(frame, column="high", index=270, factor=3.0), spec=regime_spec()),
        "rv_percentile",
    )
    # la fila 271 usa la RV de 270: subirla no puede **bajar** su percentil...
    assert cast("float", shifted[271]) >= cast("float", base[271])
    # ...y ninguna fila anterior se mueve (la RV de 270 no entra en su historia)
    assert shifted[:271] == base[:271]


# ─────────────────────────────────────────────────────────────────────────────
# A8 — garch_forecast
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_the_forecast_is_the_one_step_forecast_with_the_data_of_t_minus_one() -> None:
    """En la fixture: 500 nulos, 300 finitos (500..799), positivos y no constantes."""
    spec, matrix = _golden()
    assert spec == regime_spec()
    column = _column(matrix, "garch_forecast")

    assert column[:500] == [None] * 500
    assert column[500] is not None
    finite = [value for value in column if value is not None]
    assert len(finite) == 300
    assert all(value > 0.0 for value in finite)
    assert len(set(finite)) > 1  # el GARCH ajusta: no devuelve la varianza muestral
    assert [index for index, value in enumerate(column) if value is not None] == list(
        range(500, 800)
    )

    assert volatility_module.GARCH_MIN_TRAIN == 500
    assert volatility_module.GARCH_REFIT_EVERY == 21
    assert len(volatility_module.garch_fold_bounds(800)) == 15
    assert volatility_module.garch_fold_bounds(800)[0] == (500, 521)

    # la fila `t` es exactamente el pronostico a un paso con `ret_log` hasta `t-1`
    frame = _golden_frame().sort("session")
    returns = (frame.get_column("close") / frame.get_column("open")).log().to_numpy()
    assert column[500] == pytest.approx(
        volatility_module.garch_one_step_forecast(cast("Any", returns[:500])), rel=1e-12
    )


def test_a8_a_series_that_cannot_be_estimated_is_null_and_never_raises() -> None:
    """Menos de 500 sesiones, precio plano o no finito ⇒ ``null``, sin excepcion."""
    short = regime_matrix(_frame(120), spec=regime_spec())
    assert _column(short, "garch_forecast") == [None] * 120

    flat = pl.DataFrame(
        {
            "session": _sessions(520),
            "open": [100.0] * 520,
            "high": [100.0] * 520,
            "low": [100.0] * 520,
            "close": [100.0] * 520,
        }
    )
    matrix = regime_matrix(flat, spec=regime_spec())
    forecast = _column(matrix, "garch_forecast")
    assert [value for value in forecast if value is not None] == []

    broken = _long_frame(520).with_columns(pl.Series("close", [None] * 520))
    poisoned = regime_matrix(broken, spec=regime_spec())
    assert [value for value in _column(poisoned, "garch_forecast") if value is not None] == []
    assert all(
        value is None or math.isfinite(value) for value in _column(poisoned, "efficiency_ratio_20")
    )


# ─────────────────────────────────────────────────────────────────────────────
# A9 — garch_forecast_z
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_the_z_score_is_the_imported_expanding_normalisation() -> None:
    """``normalise_expanding`` importada, con ``min_sessions`` explicito, y 749 nulos."""
    imported: set[str] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom) and node.module == "cfdtrader.features.store":
            imported.update(alias.name for alias in node.names)
    assert "normalise_expanding" in imported
    assert "min_sessions=REGIME_MIN_SESSIONS" in _source()

    _, matrix = _golden()
    column = _column(matrix, "garch_forecast_z")
    assert sum(1 for value in column if value is None) == 500 + 249
    assert [index for index, value in enumerate(column) if value is not None] == list(
        range(749, 800)
    )

    forecast = _column(matrix, "garch_forecast")
    history = [value for value in forecast[500:750] if value is not None]
    centre = statistics.median(history)
    mad = statistics.median([abs(value - centre) for value in history])
    assert column[749] == pytest.approx(
        (cast("float", forecast[749]) - centre) / (store.MAD_SCALE * mad), rel=1e-12
    )


# ─────────────────────────────────────────────────────────────────────────────
# A10 — efficiency_ratio_20
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_the_efficiency_ratio_is_hand_computed_and_closes_at_t() -> None:
    """Dos filas a mano, en ``[0, 1]``, 20 nulos y ventana cerrada en ``t``."""
    count = 60
    frame = _frame(count)
    column = _column(regime_matrix(frame, spec=regime_spec()), "efficiency_ratio_20")
    closes = cast("list[float]", frame.get_column("close").to_list())
    window = store.REGIME_EFFICIENCY_WINDOW

    assert column[:window] == [None] * window
    for index in (window, 33, count - 1):
        movement = abs(closes[index] - closes[index - window])
        path = sum(abs(closes[i] - closes[i - 1]) for i in range(index - window + 1, index + 1))
        assert column[index] == pytest.approx(movement / path, rel=1e-12)
        assert 0.0 <= cast("float", column[index]) <= 1.0

    # la ventana **no** es `C_{t-21} ... C_{t-1}`: eso seria no cerrar en `t`. Un
    # salto grande en `t` separa las dos formulas sin ambiguedad.
    jumped = _mutated(frame, column="close", index=count - 1, factor=2.0)
    jumped_column = _column(regime_matrix(jumped, spec=regime_spec()), "efficiency_ratio_20")
    jumped_closes = cast("list[float]", jumped.get_column("close").to_list())
    movement = abs(jumped_closes[-1] - jumped_closes[-1 - window])
    path = sum(abs(jumped_closes[i] - jumped_closes[i - 1]) for i in range(count - window, count))
    assert jumped_column[-1] == pytest.approx(movement / path, rel=1e-12)
    lagged = abs(jumped_closes[-2] - jumped_closes[-2 - window]) / sum(
        abs(jumped_closes[i] - jumped_closes[i - 1]) for i in range(count - 1 - window, count - 1)
    )
    assert abs(cast("float", jumped_column[-1]) - lagged) > 0.01


def test_a10_a_flat_series_has_no_efficiency_ratio() -> None:
    """Denominador cero (serie plana) ⇒ ``null``, nunca ``inf`` ni ``NaN``."""
    flat = pl.DataFrame(
        {
            "session": _sessions(40),
            "open": [100.0] * 40,
            "high": [100.0] * 40,
            "low": [100.0] * 40,
            "close": [100.0] * 40,
        }
    )
    assert _column(regime_matrix(flat, spec=regime_spec()), "efficiency_ratio_20") == [None] * 40


# ─────────────────────────────────────────────────────────────────────────────
# A11 — calendario
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_day_of_week_is_the_isoweekday_of_every_session() -> None:
    """``day_of_week`` coincide con ``session.isoweekday()`` en **800/800**."""
    frame = _golden_frame().sort("session")
    matrix = regime_matrix(frame, spec=regime_spec())
    sessions = [cast("date", value) for value in frame.get_column("session").to_list()]
    assert _int_column(matrix, "day_of_week") == [session.isoweekday() for session in sessions]
    assert set(_int_column(matrix, "day_of_week")) == {1, 2, 3, 4, 5}


def test_a11_the_opex_calendar_is_derived_from_the_frame() -> None:
    """Vencimientos, rodadas y distancia: 38 OPEX, 12 trimestrales, 17 nulos en la cola.

    **Divergencia declarada con A11 del brief**: el criterio pide ``37`` ceros y la
    regla que el propio enunciado fija (tercer viernes de **cada** mes, rodado al
    ultimo dia de sesion ``<=`` el, dentro de la ventana) da **38**: son los 38
    terceros viernes de 2023-07-21 a 2026-08-21, de los que dos (2025-04-18 y
    2026-06-19) no son sesion y ruedan. Para obtener 37 habria que descartar un mes
    entero, lo que cambiaria tambien el maximo (``max = 24`` **si** cuadra: es la
    distancia a la OPEX siguiente desde la sesion anterior a ella). El resto de las
    cifras del criterio coinciden tal cual.
    """
    frame = _golden_frame().sort("session")
    matrix = regime_matrix(frame, spec=regime_spec())
    sessions = [cast("date", value) for value in frame.get_column("session").to_list()]
    to_opex = _int_column(matrix, "sessions_to_opex")
    is_roll = _int_column(matrix, "is_es_roll_session")

    values = [value for value in to_opex if value is not None]
    assert sum(1 for value in values if value == 0) == 38
    assert min(values) == 0
    assert max(values) == 24
    assert sum(1 for value in to_opex if value is None) == 17
    assert _int_column(matrix, "sessions_to_opex")[-17:] == [None] * 17
    assert sum(1 for value in is_roll if value == 1) == 12
    rolls = [index for index, value in enumerate(is_roll) if value == 1]
    assert all(to_opex[index] == 0 for index in rolls)
    assert {(sessions[index].year, sessions[index].month) for index in rolls} == {
        (2023, 9),
        (2023, 12),
        (2024, 3),
        (2024, 6),
        (2024, 9),
        (2024, 12),
        (2025, 3),
        (2025, 6),
        (2025, 9),
        (2025, 12),
        (2026, 3),
        (2026, 6),
    }
    # la sesion anterior a cada vencimiento dista exactamente 1
    assert all(to_opex[index - 1] == 1 for index in rolls if index > 0)
    # las dos rodadas de la ventana, y el dia festivo que **no** es sesion
    index_of = {session: position for position, session in enumerate(sessions)}
    assert date(2026, 6, 19) not in index_of
    assert date(2026, 6, 18) in index_of
    assert is_roll[index_of[date(2026, 6, 18)]] == 1
    assert to_opex[index_of[date(2026, 6, 18)]] == 0
    assert is_roll[index_of[date(2025, 4, 17)]] == 0  # abril no es trimestral, pero si OPEX
    assert to_opex[index_of[date(2025, 4, 17)]] == 0
    assert date(2025, 4, 18) not in index_of
    # la sesion siguiente a la rodada apunta al vencimiento de mayo
    assert to_opex[index_of[date(2025, 4, 17)] + 1] == (
        index_of[date(2025, 5, 16)] - index_of[date(2025, 4, 17)] - 1
    )


# ─────────────────────────────────────────────────────────────────────────────
# A12 — golden
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_the_regime_golden_is_frozen() -> None:
    """El par (``code_version``, digests) del golden no puede moverse en silencio."""
    expected = _expected()
    spec, matrix = _golden()

    assert list(matrix.columns) == expected["feature_columns"]
    assert matrix.columns == ["session", *store.REGIME_FEATURE_COLUMNS]
    assert matrix.height == expected["sessions"] == 800
    assert str(matrix.get_column("session")[0]) == expected["first_session"] == "2023-07-11"
    assert str(matrix.get_column("session")[-1]) == expected["last_session"] == "2026-09-16"
    assert _prefixed(store.feature_spec_sha256(spec)) == expected["feature_spec_sha256"]
    assert _prefixed(store.matrix_sha256(matrix)) == expected["matrix_sha256"]
    assert expected["code_version"] == store.FEATURE_CODE_VERSION
    assert spec == regime_spec()
    assert expected["feature_set"] == store.REGIME_FEATURE_SET
    assert expected["windows"] == store.DEFAULT_REGIME_WINDOWS
    assert expected["sources"] == [list(pair) for pair in store.DEFAULT_REGIME_SOURCES]

    counts = matrix.null_count().to_dicts()[0]
    assert {name: value for name, value in counts.items() if name != "session"} == (
        expected["non_null"]
    )

    # el golden falla si el calculo se mueve sin subir la constante declarada
    assert (
        _prefixed(store.matrix_sha256(matrix.drop("efficiency_ratio_20")))
        != expected["matrix_sha256"]
    )
    assert all(
        digest.startswith(store.FEATURE_VERSION_PREFIX)
        for digest in (expected["feature_spec_sha256"], expected["matrix_sha256"])
    )


def test_a12_the_four_previous_goldens_are_untouched() -> None:
    """#19 a #22 siguen dando su par de digests: la familia nueva no los mueve.

    Los cuatro ficheros conservan su ``feature_set``, su ``code_version`` y su
    ``feature_spec_sha256``. Las matrices de esas familias no dependen de la
    familia nueva, y las suites de #19-#22 las recomputan en cada corrida.
    """
    from cfdtrader.features.context import context_spec
    from cfdtrader.features.macro import macro_spec
    from cfdtrader.features.technical import technical_spec

    specs = {
        "volatility_v1": store.FeatureSpec(),
        "technical_v1": technical_spec(),
        "context_v1": context_spec(),
        "macro_v1": macro_spec(),
    }
    files = {
        "volatility_v1": "golden_expected.json",
        "technical_v1": "technical_golden_expected.json",
        "context_v1": "context_golden_expected.json",
        "macro_v1": "macro_golden_expected.json",
    }
    assert set(specs) | {"regime_v1"} == set(store.CATALOG_BY_FEATURE_SET)
    for feature_set, path in files.items():
        expected = dict(json.loads((FIXTURES / path).read_text(encoding="utf-8")))
        assert expected["feature_set"] == feature_set
        assert expected["code_version"] == store.FEATURE_CODE_VERSION == 1
        assert (
            _prefixed(store.feature_spec_sha256(specs[feature_set]))
            == (expected["feature_spec_sha256"])
        )
        assert expected["matrix_sha256"].startswith(store.FEATURE_VERSION_PREFIX)


# ─────────────────────────────────────────────────────────────────────────────
# A13 — almacen
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_the_matrix_is_persisted_with_the_shared_contract(tmp_path: Path) -> None:
    """``daily_records``/``save_daily``/``load_daily`` con un ``Store`` en ``tmp_path``."""
    root = tmp_path / "store"
    handle = Store(root)
    spec = regime_spec()
    matrix = _matrix(40)

    records = store.daily_records(matrix, spec=spec, series_id="^GSPC", fetched_at=FETCHED_AT)
    assert set(records[0]) == {
        "source",
        "series_id",
        "as_of",
        "fetched_at",
        "published_at",
        "features_version",
        "feature_spec_sha256",
        *store.REGIME_FEATURE_COLUMNS,
    }
    assert all(record["source"] == REGIME_SOURCE for record in records)
    assert all("version" not in record for record in records)

    for foreign in FOREIGN_COLUMNS:
        with pytest.raises(store.InvalidFeatureMatrixError, match="fuera del catalogo"):
            store.daily_records(
                matrix.with_columns(pl.Series(foreign, [0.0] * matrix.height)),
                spec=spec,
                series_id="^GSPC",
                fetched_at=FETCHED_AT,
            )

    assert (
        store.save_daily(handle, spec=spec, matrix=matrix, series_id="^GSPC", fetched_at=FETCHED_AT)
        is WriteOutcome.CREATED
    )
    before = _parquet_digest(root)
    files = sorted(root.rglob("*.parquet"))
    assert (
        store.save_daily(
            handle,
            spec=spec,
            matrix=matrix,
            series_id="^GSPC",
            fetched_at=FETCHED_AT + timedelta(days=1),
        )
        is WriteOutcome.UNCHANGED
    )
    assert _parquet_digest(root) == before
    assert sorted(root.rglob("*.parquet")) == files
    assert _versions(root, source=REGIME_SOURCE) == [1] * 40

    bumped = store.FeatureSpec(
        feature_set="regime_v1",
        code_version=spec.code_version + 1,
        windows=store.DEFAULT_REGIME_WINDOWS,
        sources=store.DEFAULT_REGIME_SOURCES,
    )
    assert (
        store.save_daily(
            handle, spec=bumped, matrix=matrix, series_id="^GSPC", fetched_at=FETCHED_AT
        )
        is WriteOutcome.CREATED
    )
    assert _versions(root, source=REGIME_SOURCE) == [2] * 40

    loaded = store.load_daily(handle, series_id="^GSPC", feature_set="regime_v1")
    assert loaded.height == 40
    assert set(loaded.get_column("source").to_list()) == {REGIME_SOURCE}
    assert set(store.REGIME_FEATURE_COLUMNS) <= set(loaded.columns)

    # la misma sesion puede convivir con la matriz de volatilidad en el mismo dataset
    store.save_daily(
        handle,
        spec=store.FeatureSpec(),
        matrix=_volatility_matrix(40),
        series_id="^GSPC",
        fetched_at=FETCHED_AT,
    )
    rows = Store(root).sql(
        f"SELECT source, count(*) AS n FROM {store.FEATURES_LAYER}.{store.FEATURES_DATASET} "  # noqa: S608
        "GROUP BY source ORDER BY source"
    )
    assert rows.get_column("n").to_list() == [40, 40]

    # una familia sin filas en un dataset que si existe tampoco devuelve vacio
    empty = tmp_path / "empty"
    store.save_daily(
        Store(empty),
        spec=store.FeatureSpec(),
        matrix=_volatility_matrix(40),
        series_id="^GSPC",
        fetched_at=FETCHED_AT,
    )
    with pytest.raises(UnknownDatasetError, match="no tiene filas de la familia"):
        store.load_daily(Store(empty), series_id="^GSPC", feature_set="regime_v1")


# ─────────────────────────────────────────────────────────────────────────────
# A14 — via del GARCH, puertas y cobertura
# ─────────────────────────────────────────────────────────────────────────────
_DIGEST_SCRIPT = """
import hashlib, json, pathlib, sys
from datetime import UTC, datetime

import polars as pl

from cfdtrader.features import store
from cfdtrader.features.regime import regime_matrix, regime_spec

root, csv_path = sys.argv[1], sys.argv[2]
frame = pl.read_csv(csv_path, try_parse_dates=True)
spec = regime_spec()
matrix = regime_matrix(frame, spec=spec)
moments = (frame.get_column("session").cast(pl.Datetime("us"))
           + pl.duration(hours=20)).dt.replace_time_zone("UTC")
persistible = matrix.with_columns(moments.alias("as_of"))
handle = store.Store(root)
store.save_daily(
    handle,
    spec=spec,
    matrix=persistible,
    series_id="^GSPC",
    fetched_at=datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
)
rows = store.load_daily(handle, series_id="^GSPC", feature_set="regime_v1")
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
            str(FIXTURES / "regime_golden_inputs.csv"),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return dict(json.loads(result.stdout))


def test_a14_the_garch_engine_lives_in_the_volatility_module() -> None:
    """Los seis nombres del motor GARCH viven en ``features/volatility.py`` (A14)."""
    for name in (
        "GarchFit",
        "fit_garch",
        "garch_one_step_forecast",
        "garch_forecasts",
        "garch_fold_bounds",
        "GARCH_MIN_TRAIN",
        "GARCH_REFIT_EVERY",
    ):
        assert hasattr(volatility_module, name), name
    assert "import arch" in Path(volatility_module.__file__).read_text(encoding="utf-8")
    assert volatility_module.GARCH_MIN_TRAIN == 500
    assert volatility_module.GARCH_REFIT_EVERY == 21

    # el modulo de estudio **no** define el motor: lo importa y lo re-expone
    study = Path(volatility_forecast.__file__).read_text(encoding="utf-8")
    assert "def fit_garch" not in study
    assert "def _fit_garch" not in study
    assert "def _garch_forecasts" not in study
    assert "def _fold_bounds" not in study
    assert volatility_forecast.MIN_TRAIN == volatility_module.GARCH_MIN_TRAIN
    assert volatility_forecast.REFIT_EVERY == volatility_module.GARCH_REFIT_EVERY
    assert volatility_forecast.garch_one_step_forecast is volatility_module.garch_one_step_forecast
    # y la ruta publica que usan los tests de #7 sigue viva
    assert volatility_module.__all__.count("garch_one_step_forecast") == 1
    assert "garch_one_step_forecast" in volatility_forecast.__all__


def test_a14_determinism_across_processes_and_second_pass(tmp_path: Path) -> None:
    """Mismo input ⇒ mismo digest, mismas ``features_version`` y Parquet byte a byte."""
    digests = [
        _subprocess_digests(seed, tmp_path / f"seed_{seed}") for seed in ("0", "1", "random")
    ]
    assert digests[0] == digests[1] == digests[2]

    spec, matrix = _golden()
    assert digests[0]["feature_spec_sha256"] == store.feature_spec_sha256(spec)
    assert digests[0]["matrix_sha256"] == store.matrix_sha256(matrix)
    assert digests[0]["feature_spec_sha256"] == _expected()["feature_spec_sha256"].removeprefix(
        store.FEATURE_VERSION_PREFIX
    )
    assert digests[0]["source"] == REGIME_SOURCE

    first, second = tmp_path / "pass_1", tmp_path / "pass_2"
    persistible = _with_as_of(matrix)
    for root in (first, second):
        store.save_daily(
            Store(root),
            spec=spec,
            matrix=persistible,
            series_id="^GSPC",
            fetched_at=GOLDEN_FETCHED_AT,
        )
    assert _parquet_digest(first) == _parquet_digest(second) == digests[0]["parquet_sha256"]
    loaded = store.load_daily(Store(first), series_id="^GSPC", feature_set="regime_v1")
    assert loaded.get_column("features_version")[0] == digests[0]["features_version"]


def test_a14_the_module_does_not_cheat_the_gates() -> None:
    """La cobertura no se maquilla: sin ``pragma``, sin ``type: ignore``."""
    source = _source()
    assert "pragma: no cover" not in source
    assert "type: ignore" not in source
    assert "coverage" not in source.lower()
    assert "\t" not in source
    assert "pyright: ignore" in source  # solo la importacion del helper privado de #7
    assert regime_matrix.__module__ == "cfdtrader.features.regime"
