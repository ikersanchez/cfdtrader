"""Tests de la familia tecnica de features (tarea #20).

Un test por criterio, ``test_aN_...``. La raiz del ``Store`` vive siempre bajo
``tmp_path``: la fixture de sesion de ``tests/conftest.py`` huella el ``data/`` y
el ``runs/`` del repositorio y falla si la sesion escribe en ellos.

El *golden dataset* de A13 vive en ``tests/fixtures/features/``: los inputs en
``golden_inputs.csv`` (**reutilizado** de #19, no se duplica) y el par esperado en
``technical_golden_expected.json``.
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

from cfdtrader.data.store import Store, UnknownDatasetError, WriteOutcome
from cfdtrader.features import store, technical
from cfdtrader.features.technical import technical_matrix, technical_spec
from cfdtrader.features.volatility import ATR_WINDOW, normalised_atr, true_range

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "features"

#: Primera sesion sintetica. El modulo no conoce el calendario a proposito.
FIRST_SESSION = date(2025, 1, 2)

#: Hora UTC que el test usa como cierre de sesion. Es un valor **del test**.
CLOSE_HOUR_UTC = 20

#: Instante de captura de los tests: posterior a todas las sesiones usadas.
FETCHED_AT = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

#: ``source`` de la familia tecnica (el discriminador de familia, decision 2 de #20).
TECHNICAL_SOURCE = store.TECHNICAL_FEATURES_SOURCE

#: Sesiones minimas de las dos ``_z`` (ventana expandida de #19).
MIN_SESSIONS = store.TECHNICAL_MIN_SESSIONS

#: La tabla de la issue, literal: ventana y ``required_as_of`` de cada entrada.
CATALOG_TABLE: dict[str, tuple[int | None, str]] = {
    "ret_1": (1, "cierre de la sesion t"),
    "ret_5": (5, "cierre de la sesion t"),
    "ret_21": (21, "cierre de la sesion t"),
    "atr_norm": (ATR_WINDOW, "cierre de la sesion t-1"),
    "dist_sma_20": (20, "cierre de la sesion t"),
    "rsi_14": (store.RSI_WINDOW, "cierre de la sesion t"),
    "range_pos_20": (20, "cierre de la sesion t"),
    "vol_break_20": (20, "cierre de la sesion t"),
    "atr_norm_z": (MIN_SESSIONS, "cierre de la sesion t-1"),
    "dist_sma_20_z": (MIN_SESSIONS, "cierre de la sesion t"),
}

#: Columnas que **no** son de esta familia: una sola basta para probar el rechazo.
FOREIGN_COLUMNS = ("parkinson_rv", "har_lag1", "vix_level")


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def _sessions(count: int) -> list[date]:
    """``count`` sesiones consecutivas desde la primera del test."""
    return [FIRST_SESSION + timedelta(days=index) for index in range(count)]


def _closes(count: int) -> list[float]:
    """Cierres sinteticos y deterministas (sin RNG), siempre positivos."""
    return [100.0 + (index % 17) * 0.7 - (index % 5) * 0.3 for index in range(count)]


def _frame(count: int = 40) -> pl.DataFrame:
    """Frame sintetico de entrada: ``session`` + OHLC coherente (low < close < high)."""
    closes = _closes(count)
    return pl.DataFrame(
        {
            "session": _sessions(count),
            "open": [value * 1.001 for value in closes],
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
        }
    )


def _true_ranges(frame: pl.DataFrame) -> list[float]:
    """``TR`` calculado a mano desde la definicion (no llama al modulo)."""
    highs = cast("list[float]", frame.get_column("high").to_list())
    lows = cast("list[float]", frame.get_column("low").to_list())
    closes = cast("list[float]", frame.get_column("close").to_list())
    ranges: list[float] = []
    for index in range(len(closes)):
        if index == 0:
            ranges.append(highs[0] - lows[0])
            continue
        previous = closes[index - 1]
        ranges.append(
            max(
                highs[index] - lows[index],
                abs(highs[index] - previous),
                abs(lows[index] - previous),
            )
        )
    return ranges


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
    """Matriz tecnica persistible: ``session`` + ``as_of`` + las diez features."""
    return _with_as_of(technical_matrix(_frame(count), spec=technical_spec()))


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
    """Par esperado congelado de A13."""
    raw = (FIXTURES / "technical_golden_expected.json").read_text(encoding="utf-8")
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
    """Inputs del golden: el **mismo** CSV de #19."""
    return pl.read_csv(FIXTURES / "golden_inputs.csv", try_parse_dates=True)


def _golden() -> tuple[store.FeatureSpec, pl.DataFrame]:
    """Spec y matriz del golden tecnico, tal y como los congela A13."""
    spec = _frozen_spec(_expected())
    return spec, technical_matrix(_golden_frame(), spec=spec)


def _source() -> str:
    """Codigo fuente del modulo, del fichero real (no del bytecode)."""
    return Path(technical.__file__).read_text(encoding="utf-8")


def _tree() -> ast.Module:
    """Arbol sintactico del modulo nuevo."""
    return ast.parse(_source())


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


# ─────────────────────────────────────────────────────────────────────────────
# A1 — registro por familia
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_registry_declares_the_two_families() -> None:
    """El registro expone las dos familias y la spec tecnica se construye."""
    # #21 anade `context_v1`: el registro deja de ser solo el de esta familia
    assert sorted(store.CATALOG_BY_FEATURE_SET) == [
        "context_v1",
        "macro_v1",
        "technical_v1",
        "volatility_v1",
    ]
    assert store.CATALOG_BY_FEATURE_SET["volatility_v1"] is store.FEATURE_CATALOG
    assert store.CATALOG_BY_FEATURE_SET["technical_v1"] is store.TECHNICAL_FEATURE_CATALOG
    assert store.SOURCE_BY_FEATURE_SET["technical_v1"] == TECHNICAL_SOURCE
    assert TECHNICAL_SOURCE != store.FEATURES_SOURCE

    spec = technical_spec()
    assert spec.feature_set == store.TECHNICAL_FEATURE_SET
    assert spec.windows == store.DEFAULT_TECHNICAL_WINDOWS
    assert spec.sources == store.DEFAULT_TECHNICAL_SOURCES
    assert (
        store.FeatureSpec(
            feature_set="technical_v1",
            windows=store.DEFAULT_TECHNICAL_WINDOWS,
            sources=store.DEFAULT_TECHNICAL_SOURCES,
        )
        == spec
    )


def test_a1_a_bad_family_or_a_bad_window_is_a_typed_error(tmp_path: Path) -> None:
    """Familia sin registrar, clave fuera del catalogo y ventana discrepante."""
    # una familia sin registrar no se puede ni declarar con las ventanas de otra
    with pytest.raises(store.InvalidFeatureSpecError, match="no esta en el catalogo"):
        store.FeatureSpec(feature_set="technical_v2", windows=store.DEFAULT_TECHNICAL_WINDOWS)
    # una clave que no es de **su** catalogo
    with pytest.raises(store.InvalidFeatureSpecError, match="no esta en el catalogo"):
        store.FeatureSpec(
            feature_set="technical_v1",
            windows={**store.DEFAULT_TECHNICAL_WINDOWS, "parkinson_rv": 1},
        )
    # una ventana que contradice al catalogo
    with pytest.raises(store.InvalidFeatureSpecError, match="no coincide"):
        store.FeatureSpec(
            feature_set="technical_v1",
            windows={**store.DEFAULT_TECHNICAL_WINDOWS, "ret_5": 3},
        )

    # una familia con forma valida pero sin registrar cae al **resolverla** (#19 exige
    # que su spec siga siendo hasheable, asi que el error no puede estar en el hash)
    unregistered = store.FeatureSpec(feature_set="volatility_v2")
    assert store.feature_spec_sha256(unregistered)
    with pytest.raises(store.InvalidFeatureSpecError, match="no esta registrada"):
        store.daily_records(
            _matrix(25), spec=unregistered, series_id="^GSPC", fetched_at=FETCHED_AT
        )
    with pytest.raises(store.InvalidFeatureSpecError, match="no esta registrada"):
        store.load_daily(Store(tmp_path / "store"), series_id="^GSPC", feature_set="volatility_v2")


# ─────────────────────────────────────────────────────────────────────────────
# A2 — #19 intacto
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_frozen_19_contract_is_untouched() -> None:
    """Los dos digests de #19 y su ``code_version`` siguen clavados."""
    assert store.FEATURE_CODE_VERSION == 1
    expected = dict(json.loads((FIXTURES / "golden_expected.json").read_text(encoding="utf-8")))
    assert (
        _prefixed(store.feature_spec_sha256(store.FeatureSpec())) == expected["feature_spec_sha256"]
    )

    spec = store.FeatureSpec(
        feature_set=expected["feature_set"],
        code_version=expected["code_version"],
        parameters=expected["parameters"],
        windows=expected["windows"],
        sources=tuple((str(left), str(right)) for left, right in expected["sources"]),
    )
    matrix = store.build_matrix(_golden_frame(), spec=spec)
    assert _prefixed(store.feature_spec_sha256(spec)) == expected["feature_spec_sha256"]
    assert _prefixed(store.matrix_sha256(matrix)) == expected["matrix_sha256"]
    counts = matrix.null_count().to_dicts()[0]
    assert {name: value for name, value in counts.items() if name != "session"} == (
        expected["non_null"]
    )

    # el registro por familia no toca el catalogo de volatilidad ni sus defaults
    assert tuple(entry.name for entry in store.FEATURE_CATALOG) == store.FEATURE_COLUMNS
    assert {entry.name: entry.window for entry in store.FEATURE_CATALOG} == store.DEFAULT_WINDOWS
    assert store.FeatureSpec() == spec


# ─────────────────────────────────────────────────────────────────────────────
# A3 — catalogo y matriz
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_the_catalog_is_exactly_the_ten_declared_entries() -> None:
    """Nombre, ventana y ``required_as_of`` de las diez entradas, uno por uno."""
    assert [entry.name for entry in store.TECHNICAL_FEATURE_CATALOG] == list(CATALOG_TABLE)
    assert tuple(CATALOG_TABLE) == store.TECHNICAL_FEATURE_COLUMNS
    for entry in store.TECHNICAL_FEATURE_CATALOG:
        window, required = CATALOG_TABLE[entry.name]
        assert entry.window == window
        assert entry.required_as_of == required
        assert entry.formula.strip() == entry.formula
        assert entry.formula and entry.source.startswith("raw.")


def test_a3_the_matrix_is_session_plus_the_catalog_in_order() -> None:
    """``technical_matrix`` publica ``session`` mas las diez columnas, ordenado."""
    matrix = technical_matrix(_frame(40), spec=technical_spec())
    assert matrix.columns == ["session", *store.TECHNICAL_FEATURE_COLUMNS]
    assert matrix.height == 40
    assert matrix.get_column("session").to_list() == _sessions(40)

    shuffled = technical_matrix(_frame(40).reverse(), spec=technical_spec())
    assert store.matrix_sha256(shuffled) == store.matrix_sha256(matrix)

    with pytest.raises(store.InvalidFeatureMatrixError, match="faltan columnas de entrada"):
        technical_matrix(_frame(40).drop("high"), spec=technical_spec())
    with pytest.raises(store.InvalidFeatureMatrixError, match="es la entrada de"):
        technical_matrix(_frame(40), spec=store.FeatureSpec())
    # y la matriz de volatilidad no admite la spec tecnica (la otra direccion)
    with pytest.raises(store.InvalidFeatureMatrixError, match="es la entrada de"):
        store.build_matrix(_frame(40), spec=technical_spec())


def test_a3_daily_records_rejects_a_column_outside_the_technical_catalog() -> None:
    """Una feature de la otra familia en la matriz tecnica no se persiste."""
    matrix = _matrix(40)
    for foreign in FOREIGN_COLUMNS:
        with pytest.raises(store.InvalidFeatureMatrixError, match="fuera del catalogo"):
            store.daily_records(
                matrix.with_columns(pl.Series(foreign, [0.0] * matrix.height)),
                spec=technical_spec(),
                series_id="^GSPC",
                fetched_at=FETCHED_AT,
            )
    records = store.daily_records(
        matrix, spec=technical_spec(), series_id="^GSPC", fetched_at=FETCHED_AT
    )
    assert set(records[0]) == {
        "source",
        "series_id",
        "as_of",
        "fetched_at",
        "published_at",
        "features_version",
        "feature_spec_sha256",
        *store.TECHNICAL_FEATURE_COLUMNS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# A4 — retornos multi-ventana
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_multi_window_returns_are_hand_computed() -> None:
    """``ln(C_t / C_{t-k})`` sobre el fixture, con las ``k`` primeras a ``null``."""
    count = 40
    closes = _closes(count)
    matrix = technical_matrix(_frame(count), spec=technical_spec())

    for name, lag in store.RETURN_LAGS:
        column = _column(matrix, name)
        for index in range(count):
            if index < lag:
                assert column[index] is None
            else:
                assert column[index] == pytest.approx(math.log(closes[index] / closes[index - lag]))
        assert matrix.get_column(name).null_count() == lag


def test_a4_the_returns_read_only_the_close() -> None:
    """Mutar ``open`` no cambia **nada**, y el frame sin ``open`` tambien vale."""
    frame = _frame(40)
    base = technical_matrix(frame, spec=technical_spec())
    mutated = technical_matrix(
        frame.with_columns((pl.col("open") * 3.0).alias("open")), spec=technical_spec()
    )
    assert store.matrix_sha256(mutated) == store.matrix_sha256(base)

    without_open = technical_matrix(frame.drop("open"), spec=technical_spec())
    assert store.matrix_sha256(without_open) == store.matrix_sha256(base)


# ─────────────────────────────────────────────────────────────────────────────
# A5 — ATR importado, no reimplementado
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_the_atr_is_imported_from_the_volatility_module() -> None:
    """El AST importa ``true_range`` y ``normalised_atr`` y no define una copia."""
    imported: set[str] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom) and node.module == "cfdtrader.features.volatility":
            imported.update(alias.name for alias in node.names)
    assert {"true_range", "normalised_atr"} <= imported

    defined = {node.name for node in ast.walk(_tree()) if isinstance(node, ast.FunctionDef)}
    assert not defined & {
        "true_range",
        "normalised_atr",
        "parkinson_variance",
        "har_regressors",
        "har_forecast",
        "vix_features",
    }


def test_a5_the_atr_column_matches_the_volatility_function_value_by_value() -> None:
    """Columna ``atr_norm`` == ``normalised_atr`` sobre el mismo frame, valor a valor."""
    frame = _frame(40)
    expected = normalised_atr(true_range(frame), window=ATR_WINDOW)
    matrix = technical_matrix(frame, spec=technical_spec())
    assert _column(matrix, "atr_norm") == _column(expected, "atr_norm")


def test_a5_the_atr_of_a_session_ignores_that_session() -> None:
    """La fila ``t`` de ``atr_norm`` no cambia al mutar ``open/high/low/close`` de ``t``."""
    frame = _frame(40)
    base = _column(technical_matrix(frame, spec=technical_spec()), "atr_norm")
    target = 30
    mutated = frame.clone()
    for name, factor in (("open", 1.5), ("high", 1.2), ("low", 0.8), ("close", 1.3)):
        values = [float(value) for value in frame.get_column(name).to_list()]
        values[target] *= factor
        mutated = mutated.with_columns(pl.Series(name, values))

    shifted = _column(technical_matrix(mutated, spec=technical_spec()), "atr_norm")
    assert shifted[target] == base[target]
    assert shifted[target + 1] != base[target + 1]  # la sesion mutada si mueve el ATR siguiente


# ─────────────────────────────────────────────────────────────────────────────
# A6 — distancia a medias
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_distance_to_sma_is_hand_computed() -> None:
    """``C_t / media(C_{t-19} ... C_t) - 1``, con las 19 primeras a ``null``."""
    count = 40
    closes = _closes(count)
    column = _column(technical_matrix(_frame(count), spec=technical_spec()), "dist_sma_20")

    for index in range(count):
        if index < 19:
            assert column[index] is None
        else:
            average = statistics.fmean(closes[index - 19 : index + 1])
            assert column[index] == pytest.approx(closes[index] / average - 1.0)
    assert sum(value is None for value in column) == 19


# ─────────────────────────────────────────────────────────────────────────────
# A7 — RSI de Wilder
# ─────────────────────────────────────────────────────────────────────────────
def _alternating_frame() -> pl.DataFrame:
    """17 sesiones cuyo ``d = C_t - C_{t-1}`` alterna ``+1`` y ``-1`` desde ``t-1``.

    Con esa serie el RSI se calcula a mano: las 14 primeras diferencias dan 7
    ganancias y 7 perdidas de 1, asi que la semilla es ``(1, 1)``.
    """
    closes = [10.0]
    step = 1.0
    for _ in range(16):
        closes.append(closes[-1] + step)
        step = -step
    return pl.DataFrame(
        {
            "session": _sessions(17),
            "open": closes,
            "high": [value + 0.5 for value in closes],
            "low": [value - 0.5 for value in closes],
            "close": closes,
        }
    )


def test_a7_wilder_rsi_is_hand_computed() -> None:
    """Semilla por media simple de los 14 primeros ``d`` y suavizado ``alpha = 1/14``.

    La semilla es ``suma de ganancias / 14`` (no la media de las ganancias
    distintas): con 7 ganancias de 1 y 7 perdidas de 1, las dos medias valen
    ``0.5`` y el RSI arranca en 50. A partir de ahi, cada valor se calcula con el
    suavizado de Wilder sobre su delta.
    """
    column = _column(technical_matrix(_alternating_frame(), spec=technical_spec()), "rsi_14")
    assert column[:14] == [None] * 14
    assert column[14] == pytest.approx(50.0)  # semilla (0.5, 0.5)
    assert column[15] == pytest.approx(100.0 * 15.0 / 28.0)  # gain=1: (15/28, 13/28)
    assert column[16] == pytest.approx(100.0 * 195.0 / 392.0)  # loss=1: (195/392, 197/392)
    assert sum(value is None for value in column) == 14


def test_a7_the_three_edges_and_the_range() -> None:
    """Sin perdidas ``100.0``, sin ganancias ``0.0``, serie plana ``50.0``, rango [0, 100]."""
    count = 30

    def rsi(closes: list[float]) -> list[float | None]:
        frame = pl.DataFrame(
            {
                "session": _sessions(len(closes)),
                "open": closes,
                "high": [value + 0.5 for value in closes],
                "low": [value - 0.5 for value in closes],
                "close": closes,
            }
        )
        return _column(technical_matrix(frame, spec=technical_spec()), "rsi_14")

    rising = rsi([100.0 + index for index in range(count)])
    falling = rsi([100.0 - index for index in range(count)])
    flat = rsi([100.0] * count)
    assert set(rising[14:]) == {100.0}
    assert set(falling[14:]) == {0.0}
    assert set(flat[14:]) == {50.0}

    scaled = rsi([100.0 + math.sin(index / 3.0) * 5.0 for index in range(count)])
    assert all(value is None or 0.0 <= value <= 100.0 for value in scaled)


# ─────────────────────────────────────────────────────────────────────────────
# A8 — rango y ruptura
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_range_position_and_volatility_break_are_hand_computed() -> None:
    """Las dos formulas, a mano, mas los dos bordes que dan ``null``."""
    count = 40
    frame = _frame(count)
    matrix = technical_matrix(frame, spec=technical_spec())
    highs = cast("list[float]", frame.get_column("high").to_list())
    lows = cast("list[float]", frame.get_column("low").to_list())
    closes = cast("list[float]", frame.get_column("close").to_list())
    ranges = _true_ranges(frame)

    position = _column(matrix, "range_pos_20")
    brk = _column(matrix, "vol_break_20")
    for index in range(count):
        if index < 19:
            assert position[index] is None
        else:
            lowest = min(lows[index - 19 : index + 1])
            highest = max(highs[index - 19 : index + 1])
            assert position[index] == pytest.approx((closes[index] - lowest) / (highest - lowest))
            assert 0.0 <= cast("float", position[index]) <= 1.0
        if index < 20:
            assert brk[index] is None
        else:
            average = statistics.fmean(ranges[index - 20 : index])
            assert brk[index] == pytest.approx(ranges[index] / average)

    # `max(high) == min(low)` y un denominador cero se publican `null`, no `NaN` ni `inf`
    flat = pl.DataFrame(
        {
            "session": _sessions(count),
            "open": [100.0] * count,
            "high": [100.0] * count,
            "low": [100.0] * count,
            "close": [100.0] * count,
        }
    )
    flat_matrix = technical_matrix(flat, spec=technical_spec())
    assert flat_matrix.get_column("range_pos_20").to_list() == [None] * count
    assert flat_matrix.get_column("vol_break_20").to_list() == [None] * count


def test_a8_no_column_holds_nan_or_inf_after_persistence() -> None:
    """Tras ``daily_records`` ninguna columna del catalogo lleva ``NaN`` ni ``inf``."""
    records = store.daily_records(
        _matrix(40), spec=technical_spec(), series_id="^GSPC", fetched_at=FETCHED_AT
    )
    for record in records:
        for name in store.TECHNICAL_FEATURE_COLUMNS:
            value = record[name]
            assert value is None or math.isfinite(cast("float", value))

    # un frame con un precio no positivo tampoco produce `inf`: produce `null`
    zeroed = _frame(40).with_columns(
        pl.when(pl.int_range(pl.len()) == 25).then(0.0).otherwise(pl.col("close")).alias("close")
    )
    broken = technical_matrix(zeroed, spec=technical_spec())
    for name in store.TECHNICAL_FEATURE_COLUMNS:
        for value in _column(broken, name):
            assert value is None or math.isfinite(value)


def test_a8_a_broken_input_column_is_a_typed_error_or_a_null() -> None:
    """El texto es un error tipado; un ``NaN``/``inf`` de entrada es un hueco."""
    frame = _frame(40)
    for name in ("close", "high", "low"):
        texts: list[object] = ["hola"] * frame.height
        with pytest.raises(store.InvalidFeatureMatrixError, match="no es numerica"):
            technical_matrix(frame.with_columns(pl.Series(name, texts)), spec=technical_spec())

    for broken in (float("nan"), float("inf"), float("-inf")):
        values = [float(value) for value in frame.get_column("close").to_list()]
        values[25] = broken
        matrix = technical_matrix(
            frame.with_columns(pl.Series("close", values)), spec=technical_spec()
        )
        assert matrix.get_column("rsi_14").to_list()[25] is None
        for name in store.TECHNICAL_FEATURE_COLUMNS:
            for value in _column(matrix, name):
                assert value is None or math.isfinite(value)

    # un hueco (`null`) es un dato: no rompe la matriz y el RSI vuelve a sembrarse
    holes: list[float | None] = [float(value) for value in frame.get_column("close").to_list()]
    holes[25] = None
    holed = technical_matrix(
        frame.with_columns(pl.Series("close", holes, dtype=pl.Float64)), spec=technical_spec()
    )
    rsi = holed.get_column("rsi_14").to_list()
    base = technical_matrix(frame, spec=technical_spec()).get_column("rsi_14").to_list()
    assert rsi[:25] == base[:25]
    assert all(value is None for value in rsi[25:])
    assert holed.get_column("ret_1").to_list()[25] is None
    for name in store.TECHNICAL_FEATURE_COLUMNS:
        for value in _column(holed, name):
            assert value is None or math.isfinite(value)


# ─────────────────────────────────────────────────────────────────────────────
# A9 — las `_z` expanden
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_the_z_columns_come_from_the_imported_normalisation() -> None:
    """La normalizacion es la de #19, con ``min_sessions = 250`` y ventana expandida."""
    imported: set[str] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom) and node.module == "cfdtrader.features.store":
            imported.update(alias.name for alias in node.names)
    assert "normalise_expanding" in imported

    frame = _golden_frame()
    matrix = technical_matrix(frame, spec=technical_spec())
    # las dos `_z` son exactamente `normalise_expanding` sobre las columnas base de la
    # propia matriz: no hay una segunda normalizacion en el modulo
    base = pl.DataFrame(
        {
            "session": matrix.get_column("session"),
            "atr_norm": matrix.get_column("atr_norm"),
            "dist_sma_20": matrix.get_column("dist_sma_20"),
        }
    )
    for source_column, produced in (("atr_norm", "atr_norm_z"), ("dist_sma_20", "dist_sma_20_z")):
        expected = store.normalise_expanding(
            base.select("session", source_column), source_column, min_sessions=MIN_SESSIONS
        )
        assert _column(matrix, produced) == _column(expected, produced)

    # las 249 primeras sesiones son `null`, y el minimo declarado son 250 sesiones:
    # los nulos de la columna base no cuentan como historia (`atr_norm` empieza en 14)
    assert matrix.get_column("atr_norm_z").null_count() == 14 + MIN_SESSIONS - 1
    assert matrix.get_column("dist_sma_20_z").null_count() == 19 + MIN_SESSIONS - 1
    for produced in ("atr_norm_z", "dist_sma_20_z"):
        assert matrix.get_column(produced).to_list()[: MIN_SESSIONS - 1] == [None] * (
            MIN_SESSIONS - 1
        )


def test_a9_a_prefix_is_identical_to_the_full_matrix() -> None:
    """El valor de la fila ``t`` no cambia al anadir sesiones posteriores."""
    frame = _golden_frame()
    full = technical_matrix(frame, spec=technical_spec())
    assert full.get_column("atr_norm_z").null_count() < full.height  # hay valores, no todo nulo
    for prefix in (1, 14, 20, 21, MIN_SESSIONS, MIN_SESSIONS + 14, frame.height):
        partial = technical_matrix(frame.head(prefix), spec=technical_spec())
        for name in ("atr_norm_z", "dist_sma_20_z"):
            assert _column(partial, name) == _column(full, name)[:prefix]


# ─────────────────────────────────────────────────────────────────────────────
# A10 — sin look-ahead
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_every_prefix_reproduces_the_full_matrix() -> None:
    """Todos los prefijos ``1…N`` dan, fila a fila, las ``N`` primeras filas."""
    count = 60
    frame = _frame(count)
    full = technical_matrix(frame, spec=technical_spec())
    for prefix in range(1, count + 1):
        partial = technical_matrix(frame.head(prefix), spec=technical_spec())
        assert partial.height == prefix
        for name in full.columns:
            assert partial.get_column(name).to_list() == full.get_column(name).to_list()[:prefix]


def test_a10_no_shift_with_a_negative_argument() -> None:
    """El AST no puede contener un ``shift`` hacia adelante."""
    shifts = [
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "shift"
    ]
    assert shifts  # hay retardos, y todos miran al pasado
    for node in shifts:
        first = node.args[0]
        assert not (isinstance(first, ast.UnaryOp) and isinstance(first.op, ast.USub))
        assert not (
            isinstance(first, ast.Constant) and isinstance(first.value, int) and first.value < 0
        )


# ─────────────────────────────────────────────────────────────────────────────
# A11 — `as_of` y `session`
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_the_row_anchor_is_the_session_close_in_utc() -> None:
    """``as_of`` es el cierre de sesion y ``as_of.date() == session``."""
    records = store.daily_records(
        _matrix(25), spec=technical_spec(), series_id="^GSPC", fetched_at=FETCHED_AT
    )
    for record, session in zip(records, _sessions(25), strict=True):
        as_of = cast("datetime", record["as_of"])
        assert as_of == _session_close(session)
        assert as_of.date() == session
        assert as_of.tzinfo is UTC

    # una discrepancia es un error tipado, no una fila mal anclada
    sessions = _sessions(25)
    shifted = _matrix(25).with_columns(
        pl.Series("as_of", [_session_close(session + timedelta(days=1)) for session in sessions])
    )
    with pytest.raises(store.InvalidFeatureMatrixError, match="no corresponden al mismo dia"):
        store.daily_records(
            shifted, spec=technical_spec(), series_id="^GSPC", fetched_at=FETCHED_AT
        )


def test_a11_the_required_as_of_is_the_declared_one() -> None:
    """Las dos features que solo necesitan el cierre de ``t-1`` son ``atr_norm`` y su ``_z``."""
    needs_previous_close = {
        entry.name
        for entry in store.TECHNICAL_FEATURE_CATALOG
        if entry.required_as_of.endswith("t-1")
    }
    assert needs_previous_close == {"atr_norm", "atr_norm_z"}
    for entry in store.TECHNICAL_FEATURE_CATALOG:
        assert entry.required_as_of == CATALOG_TABLE[entry.name][1]


# ─────────────────────────────────────────────────────────────────────────────
# A12 — persistencia y convivencia
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_the_two_families_coexist_in_the_same_dataset(tmp_path: Path) -> None:
    """La familia tecnica no sustituye a la de volatilidad: las separa el ``source``."""
    root = tmp_path / "store"
    handle = Store(root)
    count = 40

    assert (
        store.save_daily(
            handle,
            spec=store.FeatureSpec(),
            matrix=_volatility_matrix(count),
            series_id="^GSPC",
            fetched_at=FETCHED_AT,
        )
        is WriteOutcome.CREATED
    )
    assert (
        store.save_daily(
            handle,
            spec=technical_spec(),
            matrix=_matrix(count),
            series_id="^GSPC",
            fetched_at=FETCHED_AT,
        )
        is WriteOutcome.CREATED
    )

    rows = Store(root).sql(
        f"SELECT source, count(*) AS n FROM {store.FEATURES_LAYER}.{store.FEATURES_DATASET} "  # noqa: S608
        "GROUP BY source ORDER BY source"
    )
    assert rows.get_column("source").to_list() == sorted([store.FEATURES_SOURCE, TECHNICAL_SOURCE])
    assert rows.get_column("n").to_list() == [count, count]

    volatility = store.load_daily(handle, series_id="^GSPC")
    technical_rows = store.load_daily(handle, series_id="^GSPC", feature_set="technical_v1")
    # las filas son de una sola familia: eso lo decide el ``source`` (decision 2)
    assert volatility.height == technical_rows.height == count
    assert set(volatility.get_column("source").to_list()) == {store.FEATURES_SOURCE}
    assert set(technical_rows.get_column("source").to_list()) == {TECHNICAL_SOURCE}
    # el dataset es **compartido**, asi que su esquema de lectura es la union de las
    # dos familias (`SELECT *`; el ``feature_set`` no forma parte de la identidad del
    # almacen, #49). Las columnas de la otra familia estan, pero **nulas** en estas filas
    for foreign in FOREIGN_COLUMNS:
        assert technical_rows.get_column(foreign).is_null().all()
        assert volatility.get_column(foreign).is_not_null().any()
    for name in ("ret_1", "atr_norm", "dist_sma_20", "rsi_14", "range_pos_20", "vol_break_20"):
        assert technical_rows.get_column(name).is_not_null().any()
    assert set(store.FEATURE_COLUMNS) <= set(volatility.columns)
    assert set(store.TECHNICAL_FEATURE_COLUMNS) <= set(technical_rows.columns)
    assert not set(store.TECHNICAL_FEATURE_COLUMNS) & set(FOREIGN_COLUMNS)
    assert sorted(technical_rows.get_column("version").to_list()) == [1] * count
    assert _versions(root, source=store.FEATURES_SOURCE) == [1] * count

    # una familia sin filas en un dataset que si existe tampoco devuelve vacio
    volatility_only = tmp_path / "volatility_only"
    store.save_daily(
        Store(volatility_only),
        spec=store.FeatureSpec(),
        matrix=_volatility_matrix(count),
        series_id="^GSPC",
        fetched_at=FETCHED_AT,
    )
    with pytest.raises(UnknownDatasetError, match="no tiene filas de la familia"):
        store.load_daily(Store(volatility_only), series_id="^GSPC", feature_set="technical_v1")


def test_a12_unchanged_is_free_and_a_change_bumps_the_revision(tmp_path: Path) -> None:
    """Contenido identico ⇒ ``UNCHANGED`` sin Parquet nuevo; distinto ⇒ ``version + 1``."""
    root = tmp_path / "store"
    handle = Store(root)
    spec = technical_spec()
    matrix = _matrix(40)

    def save(current: store.FeatureSpec, *, fetched_at: datetime) -> WriteOutcome:
        return store.save_daily(
            handle, spec=current, matrix=matrix, series_id="^GSPC", fetched_at=fetched_at
        )

    assert save(spec, fetched_at=FETCHED_AT) is WriteOutcome.CREATED
    before = _parquet_digest(root)
    files = sorted(root.rglob("*.parquet"))
    assert save(spec, fetched_at=FETCHED_AT + timedelta(days=1)) is WriteOutcome.UNCHANGED
    assert _parquet_digest(root) == before
    assert sorted(root.rglob("*.parquet")) == files
    assert _versions(root, source=TECHNICAL_SOURCE) == [1] * 40

    bumped = store.FeatureSpec(
        feature_set="technical_v1",
        code_version=spec.code_version + 1,
        windows=store.DEFAULT_TECHNICAL_WINDOWS,
        sources=store.DEFAULT_TECHNICAL_SOURCES,
    )
    assert save(bumped, fetched_at=FETCHED_AT) is WriteOutcome.CREATED
    assert _versions(root, source=TECHNICAL_SOURCE) == [2] * 40

    records = store.daily_records(matrix, spec=spec, series_id="^GSPC", fetched_at=FETCHED_AT)
    for record in records:
        assert "version" not in record
        assert record["fetched_at"] is FETCHED_AT
        assert record["source"] == TECHNICAL_SOURCE
        assert record["features_version"] == store.features_version(
            spec, cast("datetime", record["as_of"])
        )

    source = _source()
    for forbidden in ("datetime.now", "utcnow", "date.today", "time.time", "time_ns"):
        assert forbidden not in source


# ─────────────────────────────────────────────────────────────────────────────
# A13 — golden tecnico
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_the_technical_golden_is_frozen() -> None:
    """El par (``code_version``, digests) del golden no puede moverse en silencio."""
    expected = _expected()
    spec, matrix = _golden()

    assert list(matrix.columns) == expected["feature_columns"]
    assert matrix.height == expected["sessions"] == 300
    assert _prefixed(store.feature_spec_sha256(spec)) == expected["feature_spec_sha256"]
    assert _prefixed(store.matrix_sha256(matrix)) == expected["matrix_sha256"]
    assert expected["code_version"] == store.FEATURE_CODE_VERSION
    assert spec == technical_spec()
    assert expected["feature_set"] == store.TECHNICAL_FEATURE_SET
    assert expected["windows"] == store.DEFAULT_TECHNICAL_WINDOWS
    assert expected["sources"] == [list(pair) for pair in store.DEFAULT_TECHNICAL_SOURCES]

    counts = matrix.null_count().to_dicts()[0]
    assert {name: value for name, value in counts.items() if name != "session"} == (
        expected["non_null"]
    )

    # el golden falla si el calculo se mueve sin subir la constante declarada
    assert _prefixed(store.matrix_sha256(matrix.drop("rsi_14"))) != expected["matrix_sha256"]
    assert all(
        digest.startswith(store.FEATURE_VERSION_PREFIX)
        for digest in (
            expected["feature_spec_sha256"],
            expected["matrix_sha256"],
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# A14 — determinismo, estilo y cobertura
# ─────────────────────────────────────────────────────────────────────────────
_DIGEST_SCRIPT = """
import hashlib, json, pathlib, sys
from datetime import UTC, datetime

import polars as pl

from cfdtrader.features import store
from cfdtrader.features.technical import technical_matrix, technical_spec

root, csv_path = sys.argv[1], sys.argv[2]
frame = pl.read_csv(csv_path, try_parse_dates=True)
spec = technical_spec()
matrix = technical_matrix(frame, spec=spec)
moments = (
    frame.get_column("session").cast(pl.Datetime("us")) + pl.duration(hours=20)
).dt.replace_time_zone("UTC")
persistible = matrix.with_columns(moments.alias("as_of"))
handle = store.Store(root)
store.save_daily(
    handle,
    spec=spec,
    matrix=persistible,
    series_id="^GSPC",
    fetched_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
)
rows = store.load_daily(handle, series_id="^GSPC", feature_set="technical_v1")
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
        [sys.executable, "-c", _DIGEST_SCRIPT, str(root), str(FIXTURES / "golden_inputs.csv")],
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
    assert digests[0]["feature_spec_sha256"] == _expected()["feature_spec_sha256"].removeprefix(
        store.FEATURE_VERSION_PREFIX
    )
    assert digests[0]["source"] == TECHNICAL_SOURCE

    # segunda pasada en el mismo proceso, a una raiz distinta
    first, second = tmp_path / "pass_1", tmp_path / "pass_2"
    persistible = _with_as_of(matrix)
    for root in (first, second):
        store.save_daily(
            Store(root),
            spec=spec,
            matrix=persistible,
            series_id="^GSPC",
            fetched_at=FETCHED_AT,
        )
    assert _parquet_digest(first) == _parquet_digest(second) == digests[0]["parquet_sha256"]
    loaded = store.load_daily(Store(first), series_id="^GSPC", feature_set="technical_v1")
    assert loaded.get_column("features_version")[0] == digests[0]["features_version"]


def test_a14_the_module_does_not_cheat_the_gates() -> None:
    """La cobertura no se maquilla: sin ``pragma``, sin ``type: ignore``."""
    source = _source()
    assert "pragma: no cover" not in source
    assert "type: ignore" not in source
    assert "coverage" not in source.lower()
    assert technical_matrix.__module__ == "cfdtrader.features.technical"
    assert "\t" not in source
