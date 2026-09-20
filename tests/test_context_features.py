"""Tests de la familia de contexto de mercado (tarea #21).

Un test por criterio, ``test_aN_...``. La raiz del ``Store`` vive siempre bajo
``tmp_path``: la fixture de sesion de ``tests/conftest.py`` huella el ``data/`` y
el ``runs/`` del repositorio y falla si la sesion escribe en ellos.

El *golden dataset* de A13 vive en ``tests/fixtures/features/``: los inputs en
``context_golden_inputs.csv`` (una fila por sesion del calendario **union** y
celda vacia cuando ese mercado no cotizo) y el par esperado en
``context_golden_expected.json``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

from cfdtrader.data.store import Store, UnknownDatasetError, WriteOutcome
from cfdtrader.features import context, store, technical
from cfdtrader.features.context import context_matrix, context_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "features"

#: Primera sesion sintetica. El modulo no conoce el calendario a proposito.
FIRST_SESSION = date(2025, 1, 2)

#: Hora UTC que el test usa como cierre de sesion. Es un valor **del test**.
CLOSE_HOUR_UTC = 20

#: Instante de captura de los tests: posterior a todas las sesiones usadas.
FETCHED_AT = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

#: ``source`` de la familia de contexto (el discriminador de familia de #20).
CONTEXT_SOURCE = store.CONTEXT_FEATURES_SOURCE

#: Ventana de las correlaciones y de la beta, y minimo de la ventana expandida.
WINDOW = store.CONTEXT_CORRELATION_WINDOW
MIN_SESSIONS = store.CONTEXT_MIN_SESSIONS

#: El digest congelado de #19: la constante declarada **no** sube por unir familias.
FROZEN_SPEC_19 = "sha256:3c71eb69635589dc7ece74b662170b27104473c3b81f3f21434ac178d7dc539b"

#: La tabla de la issue, literal: ventana y ``required_as_of`` de cada entrada.
CATALOG_TABLE: dict[str, tuple[int | None, str]] = {
    "corr_dax_60": (WINDOW, "cierre de la sesion t-1"),
    "corr_ftse_60": (WINDOW, "cierre de la sesion t-1"),
    "corr_stoxx_60": (WINDOW, "cierre de la sesion t-1"),
    "corr_nikkei_60": (WINDOW, "cierre de la sesion t-1"),
    "asia_overnight_1": (1, "cierre asiatico de la sesion t"),
    "europe_prev_1": (1, "cierre europeo de la sesion t-1"),
    "beta_vix_60": (WINDOW, "cierre de la sesion t-1"),
    "dxy_ret_1": (1, "cierre de la sesion t-1"),
    "sector_dispersion_1": (1, "cierre de la sesion t-1"),
    "sector_count": (1, "cierre de la sesion t-1"),
    "sector_dispersion_1_z": (MIN_SESSIONS, "cierre de la sesion t-1"),
}

#: El rezago declarado, literal: dos mercados asiaticos con 0 y cinco series con 1.
MARKET_LAG_TABLE: dict[str, int] = {
    "^N225": 0,
    "^HSI": 0,
    "^GDAXI": 1,
    "^FTSE": 1,
    "^STOXX50E": 1,
    "DX-Y.NYB": 1,
    "^VIX": 1,
}

#: Columnas que **no** son de esta familia: una sola basta para probar el rechazo.
FOREIGN_COLUMNS = ("ret_1", "parkinson_rv", "har_lag1", "vix_level")


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def _sessions(count: int, *, start: date = FIRST_SESSION) -> list[date]:
    """``count`` sesiones consecutivas desde ``start``."""
    return [start + timedelta(days=index) for index in range(count)]


def _noise(*, seed: int, index: int) -> float:
    """Ruido determinista en ``[-1, 1]`` **sin RNG**: un sha256 del par (semilla, indice).

    Un generador congruencial lineal no vale aqui: dos semillas distintas darian la
    misma secuencia desplazada una constante, y las series sinteticas saldrian
    correlacionadas entre si sin querer.
    """
    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 63) - 1.0


def _walk(count: int, *, seed: int, base: float = 100.0, scale: float = 0.01) -> list[float]:
    """Cierres de un paseo aleatorio determinista, siempre positivos."""
    closes = [base]
    for index in range(1, count):
        closes.append(closes[-1] * math.exp(scale * _noise(seed=seed, index=index)))
    return closes


def _series(
    count: int,
    *,
    seed: int,
    base: float = 100.0,
    skip: tuple[int, ...] = (),
    late: int = 0,
) -> tuple[list[date], list[float]]:
    """Sesiones y cierres: ``skip`` son vacaciones **suyas**, ``late`` su nacimiento."""
    sessions = _sessions(count)
    closes = _walk(count, seed=seed, base=base)
    pairs = [
        (session, close)
        for index, (session, close) in enumerate(zip(sessions, closes, strict=True))
        if index not in skip and index >= late
    ]
    return [item[0] for item in pairs], [item[1] for item in pairs]


def _log_returns(closes: list[float]) -> list[float | None]:
    """``ln(C_i / C_i-1)`` calculado en el test, sin llamar al modulo."""
    returns: list[float | None] = [None]
    for index in range(1, len(closes)):
        returns.append(math.log(closes[index] / closes[index - 1]))
    return returns


def _session_close(session: date) -> datetime:
    """Cierre de sesion en UTC de una sesion sintetica."""
    return datetime(session.year, session.month, session.day, CLOSE_HOUR_UTC, tzinfo=UTC)


def _frame(sessions: list[date], closes: list[float], *, with_as_of: bool = False) -> pl.DataFrame:
    """Frame de una serie: ``session`` + ``close`` (y ``as_of`` si es el ancla)."""
    data: dict[str, pl.Series] = {
        "session": pl.Series("session", sessions, dtype=pl.Date()),
        "close": pl.Series("close", closes, dtype=pl.Float64()),
    }
    if with_as_of:
        data["as_of"] = pl.Series(
            "as_of",
            [_session_close(session) for session in sessions],
            dtype=pl.Datetime("us", "UTC"),
        )
    return pl.DataFrame(data)


def _empty_frame() -> pl.DataFrame:
    """Frame **sin filas**: una serie sin historia (el ETF que aun no existia)."""
    return pl.DataFrame(
        {
            "session": pl.Series("session", [], dtype=pl.Date()),
            "close": pl.Series("close", [], dtype=pl.Float64()),
        }
    )


def _frames(overrides: dict[str, tuple[list[date], list[float]]]) -> dict[str, pl.DataFrame]:
    """Mapping de las **19** series; las que no se nombran van sin historia.

    El ``as_of`` solo lo lleva el ancla: es lo unico que el modulo lee de mas.
    """
    frames: dict[str, pl.DataFrame] = {}
    for name in context.CONTEXT_SERIES:
        if name in overrides:
            sessions, closes = overrides[name]
            frames[name] = _frame(sessions, closes, with_as_of=name == context.ANCHOR_SERIES)
        else:
            frames[name] = _empty_frame()
    return frames


def _universe(count: int = 80, *, offset: int = 0) -> dict[str, tuple[list[date], list[float]]]:
    """Las 19 series con las mismas sesiones y una semilla distinta por serie."""
    return {
        name: _series(count, seed=index + 1 + offset, base=100.0 + 10.0 * index)
        for index, name in enumerate(context.CONTEXT_SERIES)
    }


def _matrix(
    count: int = 80, *, overrides: dict[str, tuple[list[date], list[float]]] | None = None
) -> pl.DataFrame:
    """Matriz de contexto completa: ``session`` + ``as_of`` + las once features."""
    data = _universe(count)
    if overrides:
        data.update(overrides)
    return context_matrix(_frames(data), spec=context_spec())


def _column(matrix: pl.DataFrame, name: str) -> list[float | None]:
    """Columna de features como lista de floats con nulos."""
    return [
        None if value is None else float(cast("float", value))
        for value in cast("list[object]", matrix.get_column(name).to_list())
    ]


def _assert_finite(matrix: pl.DataFrame) -> None:
    """Ninguna celda de feature es ``NaN`` ni ``inf``: o numero finito, o ``null``."""
    for name in store.CONTEXT_FEATURE_COLUMNS:
        for value in cast("list[object]", matrix.get_column(name).to_list()):
            assert value is None or math.isfinite(float(cast("float", value)))


def _source() -> str:
    """Codigo fuente del modulo, del fichero real (no del bytecode)."""
    return Path(context.__file__).read_text(encoding="utf-8")


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


def _prefixed(digest: str) -> str:
    """Forma en la que el golden guarda un digest: con el prefijo ``sha256:``.

    El mismo prefijo que publica ``features_version``; ademas, un hex desnudo de 64
    caracteres dispara el hook ``detect-secrets`` (falso positivo de alta entropia).
    """
    return store.FEATURE_VERSION_PREFIX + digest


def _expected() -> dict[str, Any]:
    """Par esperado congelado de A13."""
    raw = (FIXTURES / "context_golden_expected.json").read_text(encoding="utf-8")
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


def _golden_table() -> pl.DataFrame:
    """Inputs del golden: una fila por sesion del calendario union.

    ``infer_schema_length=None`` es obligatorio: ``XLC`` nace en la sesion 180, asi
    que con la ventana de inferencia por defecto se leería como texto.
    """
    return pl.read_csv(
        FIXTURES / "context_golden_inputs.csv", try_parse_dates=True, infer_schema_length=None
    )


def _golden_frames() -> dict[str, pl.DataFrame]:
    """Las 19 series del golden, cada una con **su** calendario (celdas vacias fuera)."""
    table = _golden_table()
    frames: dict[str, pl.DataFrame] = {}
    for name in context.CONTEXT_SERIES:
        frame = table.select("session", pl.col(name).alias("close")).drop_nulls("close")
        if name == context.ANCHOR_SERIES:
            frame = frame.with_columns(
                pl.Series(
                    "as_of",
                    [
                        _session_close(cast("date", session))
                        for session in frame.get_column("session").to_list()
                    ],
                    dtype=pl.Datetime("us", "UTC"),
                )
            )
        frames[name] = frame
    return frames


def _golden() -> tuple[store.FeatureSpec, pl.DataFrame]:
    """Spec y matriz del golden de contexto, tal y como los congela A13."""
    spec = _frozen_spec(_expected())
    return spec, context_matrix(_golden_frames(), spec=spec)


def _golden_matrix() -> pl.DataFrame:
    """La matriz del golden, cuando el test no necesita la spec."""
    _, matrix = _golden()
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# A1 — registro por familia
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_registry_declares_the_third_family() -> None:
    """El registro expone las tres familias y la spec de contexto se construye."""
    assert sorted(store.CATALOG_BY_FEATURE_SET) == ["context_v1", "technical_v1", "volatility_v1"]
    assert store.CATALOG_BY_FEATURE_SET["context_v1"] is store.CONTEXT_FEATURE_CATALOG
    assert store.SOURCE_BY_FEATURE_SET["context_v1"] == CONTEXT_SOURCE
    assert CONTEXT_SOURCE not in {store.FEATURES_SOURCE, store.TECHNICAL_FEATURES_SOURCE}
    assert set(store.SOURCE_BY_FEATURE_SET) == set(store.CATALOG_BY_FEATURE_SET)

    spec = context_spec()
    assert spec.feature_set == store.CONTEXT_FEATURE_SET
    assert spec.windows == store.DEFAULT_CONTEXT_WINDOWS
    assert spec.sources == store.DEFAULT_CONTEXT_SOURCES
    assert (
        store.FeatureSpec(
            feature_set="context_v1",
            windows=store.DEFAULT_CONTEXT_WINDOWS,
            sources=store.DEFAULT_CONTEXT_SOURCES,
        )
        == spec
    )


def test_a1_all_feature_columns_covers_the_eleven_new_ones() -> None:
    """Sin las 11 columnas en ``ALL_FEATURE_COLUMNS`` el digest no veria un ``NaN``."""
    assert set(store.CONTEXT_FEATURE_COLUMNS) <= set(store.ALL_FEATURE_COLUMNS)
    assert len(set(store.ALL_FEATURE_COLUMNS)) == len(store.ALL_FEATURE_COLUMNS)
    assert set(store.CONTEXT_FEATURE_COLUMNS) & {"ret_1", "vix_level"} == set()

    matrix = _matrix(25)
    for poisoned in (float("nan"), float("inf")):
        broken = matrix.with_columns(pl.Series("dxy_ret_1", [poisoned] * matrix.height))
        with pytest.raises(store.InvalidFeatureMatrixError, match="no finito"):
            store.matrix_sha256(broken)


def test_a1_a_bad_family_or_a_bad_window_is_a_typed_error() -> None:
    """Clave fuera del catalogo de contexto y ventana discrepante."""
    with pytest.raises(store.InvalidFeatureSpecError, match="no esta en el catalogo"):
        store.FeatureSpec(
            feature_set="context_v1", windows={**store.DEFAULT_CONTEXT_WINDOWS, "ret_1": 1}
        )
    with pytest.raises(store.InvalidFeatureSpecError, match="no coincide"):
        store.FeatureSpec(
            feature_set="context_v1", windows={**store.DEFAULT_CONTEXT_WINDOWS, "corr_dax_60": 30}
        )


# ─────────────────────────────────────────────────────────────────────────────
# A2 — #19 y #20 intactos
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_frozen_19_contract_is_untouched(tmp_path: Path) -> None:
    """El digest congelado de #19, su ``code_version`` y una familia sin registrar."""
    assert store.FEATURE_CODE_VERSION == 1
    assert _prefixed(store.feature_spec_sha256(store.FeatureSpec())) == FROZEN_SPEC_19

    golden = dict(json.loads((FIXTURES / "golden_expected.json").read_text(encoding="utf-8")))
    assert golden["feature_spec_sha256"] == FROZEN_SPEC_19
    assert golden["code_version"] == store.FEATURE_CODE_VERSION

    # una spec de familia **no** registrada se sigue construyendo y hasheando: el
    # error de "familia sin registrar" vive en la resolucion, no en el hash (#20)
    unregistered = store.FeatureSpec(feature_set="context_v2")
    assert store.feature_spec_sha256(unregistered)
    with pytest.raises(store.InvalidFeatureSpecError, match="no esta registrada"):
        store.load_daily(Store(tmp_path / "store"), series_id="^GSPC", feature_set="context_v2")


def test_a2_the_frozen_20_golden_does_not_move() -> None:
    """El golden tecnico se recomputa igual: unir una familia no lo toca."""
    expected = dict(
        json.loads((FIXTURES / "technical_golden_expected.json").read_text(encoding="utf-8"))
    )
    inputs = pl.read_csv(FIXTURES / "golden_inputs.csv", try_parse_dates=True)
    spec = store.FeatureSpec(
        feature_set=expected["feature_set"],
        code_version=expected["code_version"],
        parameters=expected["parameters"],
        windows=expected["windows"],
        sources=tuple((str(left), str(right)) for left, right in expected["sources"]),
    )
    matrix = technical.technical_matrix(inputs, spec=spec)
    assert _prefixed(store.feature_spec_sha256(spec)) == expected["feature_spec_sha256"]
    assert _prefixed(store.matrix_sha256(matrix)) == expected["matrix_sha256"]
    assert expected["windows"] == store.DEFAULT_TECHNICAL_WINDOWS
    assert expected["sources"] == [list(pair) for pair in store.DEFAULT_TECHNICAL_SOURCES]


# ─────────────────────────────────────────────────────────────────────────────
# A3 — catalogo y matriz
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_the_catalog_is_exactly_the_eleven_declared_entries() -> None:
    """Nombre, ventana y ``required_as_of`` de las once entradas, uno por uno."""
    assert [entry.name for entry in store.CONTEXT_FEATURE_CATALOG] == list(CATALOG_TABLE)
    assert tuple(CATALOG_TABLE) == store.CONTEXT_FEATURE_COLUMNS
    for entry in store.CONTEXT_FEATURE_CATALOG:
        window, required = CATALOG_TABLE[entry.name]
        assert entry.window == window
        assert entry.required_as_of == required
        assert entry.formula.strip() == entry.formula
        assert entry.formula and entry.source.startswith("raw.")
    assert {entry.source for entry in store.CONTEXT_FEATURE_CATALOG} == {
        "raw.market_daily",
        "raw.sectors",
    }


def test_a3_the_matrix_is_session_as_of_and_the_catalog_in_order() -> None:
    """``context_matrix`` publica ``session``, ``as_of`` y las once columnas, ordenado."""
    matrix = _matrix(80)
    assert matrix.columns == ["session", "as_of", *store.CONTEXT_FEATURE_COLUMNS]
    assert matrix.height == 80
    assert matrix.get_column("session").to_list() == _sessions(80)

    shuffled = context_matrix(
        {name: frame.reverse() for name, frame in _frames(_universe(80)).items()},
        spec=context_spec(),
    )
    assert store.matrix_sha256(shuffled) == store.matrix_sha256(matrix)

    with pytest.raises(store.InvalidFeatureMatrixError, match="es la entrada de"):
        context_matrix(_frames(_universe(25)), spec=store.FeatureSpec())
    with pytest.raises(store.InvalidFeatureMatrixError, match="es la entrada de"):
        store.build_matrix(
            pl.DataFrame(
                {
                    "session": pl.Series("session", _sessions(25), dtype=pl.Date()),
                    "open": pl.Series("open", [1.0] * 25, dtype=pl.Float64()),
                    "high": pl.Series("high", [1.0] * 25, dtype=pl.Float64()),
                    "low": pl.Series("low", [1.0] * 25, dtype=pl.Float64()),
                    "close": pl.Series("close", [1.0] * 25, dtype=pl.Float64()),
                    "vix_close": pl.Series("vix_close", [20.0] * 25, dtype=pl.Float64()),
                }
            ),
            spec=context_spec(),
        )


def test_a3_daily_records_accepts_the_context_matrix_and_rejects_a_foreign_column() -> None:
    """La matriz de contexto pasa ``daily_records``; una columna ajena no."""
    matrix = _matrix(25)
    records = store.daily_records(
        matrix, spec=context_spec(), series_id="^GSPC", fetched_at=FETCHED_AT
    )
    assert set(records[0]) == {
        "source",
        "series_id",
        "as_of",
        "fetched_at",
        "published_at",
        "features_version",
        "feature_spec_sha256",
        *store.CONTEXT_FEATURE_COLUMNS,
    }
    assert all(record["source"] == CONTEXT_SOURCE for record in records)

    for foreign in FOREIGN_COLUMNS:
        with pytest.raises(store.InvalidFeatureMatrixError, match="fuera del catalogo"):
            store.daily_records(
                matrix.with_columns(pl.Series(foreign, [0.0] * matrix.height)),
                spec=context_spec(),
                series_id="^GSPC",
                fetched_at=FETCHED_AT,
            )


# ─────────────────────────────────────────────────────────────────────────────
# A4 — interfaz pura
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_the_module_is_pure() -> None:
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
    assert context.context_matrix.__module__ == "cfdtrader.features.context"
    assert "\t" not in source


def test_a4_the_input_mapping_is_validated() -> None:
    """Falta una serie, sobra una clave, falta ``session``/``close``, sesion repetida."""
    frames = _frames(_universe(25))

    missing = dict(frames)
    del missing["XLC"]
    with pytest.raises(store.ContextInputError, match="faltan \\['XLC'\\]"):
        context_matrix(missing, spec=context_spec())

    extra = dict(frames)
    extra["^IXIC"] = _frame(_sessions(25), [1.0] * 25)
    with pytest.raises(store.ContextInputError, match="sobran \\['\\^IXIC'\\]"):
        context_matrix(extra, spec=context_spec())

    without_close = dict(frames)
    without_close["^GDAXI"] = frames["^GDAXI"].drop("close")
    with pytest.raises(
        store.ContextInputError, match=re.escape("'^GDAXI' no trae la columna 'close'")
    ):
        context_matrix(without_close, spec=context_spec())

    without_session = dict(frames)
    without_session["^N225"] = frames["^N225"].drop("session")
    with pytest.raises(
        store.ContextInputError, match=re.escape("'^N225' no trae la columna 'session'")
    ):
        context_matrix(without_session, spec=context_spec())

    without_as_of = dict(frames)
    without_as_of["^GSPC"] = frames["^GSPC"].drop("as_of")
    with pytest.raises(store.ContextInputError, match="'as_of'"):
        context_matrix(without_as_of, spec=context_spec())

    # una sesion repetida dentro de **una** serie: el mensaje dice cual
    duplicated = frames["^FTSE"].vstack(frames["^FTSE"].slice(2, 1))
    with pytest.raises(store.ContextInputError, match="'\\^FTSE' repite estas sesiones"):
        context_matrix({**frames, "^FTSE": duplicated}, spec=context_spec())


def test_a4_a_bad_column_type_or_a_broken_close_is_a_typed_error() -> None:
    """Una columna de texto, un ``NaN``, un ``inf`` y una ``session`` sin zona."""
    frames = _frames(_universe(25))

    as_text = dict(frames)
    as_text["DX-Y.NYB"] = as_text["DX-Y.NYB"].with_columns(pl.col("close").cast(pl.Utf8))
    with pytest.raises(store.ContextInputError, match="no es numerica"):
        context_matrix(as_text, spec=context_spec())

    for poisoned in (float("nan"), float("inf")):
        broken = dict(frames)
        broken["^VIX"] = broken["^VIX"].with_columns(
            pl.Series("close", [poisoned] * 25, dtype=pl.Float64())
        )
        with pytest.raises(store.ContextInputError, match="no finito"):
            context_matrix(broken, spec=context_spec())

    naive = dict(frames)
    anchor = naive["^GSPC"]
    naive["^GSPC"] = anchor.with_columns(
        pl.Series(
            "as_of",
            [datetime(session.year, session.month, session.day) for session in _sessions(25)],
        )
    )
    with pytest.raises(store.ContextInputError, match="zona horaria"):
        context_matrix(naive, spec=context_spec())

    not_datetime = dict(frames)
    not_datetime["^GSPC"] = anchor.with_columns(pl.col("as_of").cast(pl.Utf8))
    with pytest.raises(store.ContextInputError, match=re.escape("tiene que ser pl.Datetime")):
        context_matrix(not_datetime, spec=context_spec())

    null_session = dict(frames)
    null_session["^GDAXI"] = frames["^GDAXI"].with_columns(
        pl.Series("session", [None, *_sessions(25)[1:]], dtype=pl.Date())
    )
    with pytest.raises(store.ContextInputError, match="'session' nula"):
        context_matrix(null_session, spec=context_spec())

    bad_dtype = dict(frames)
    bad_dtype["^N225"] = frames["^N225"].with_columns(pl.col("session").cast(pl.Int32))
    with pytest.raises(
        store.ContextInputError, match=re.escape("tiene que ser pl.Date o pl.Datetime")
    ):
        context_matrix(bad_dtype, spec=context_spec())


def test_a4_only_session_and_close_are_read() -> None:
    """Mutar cualquier otra columna de una serie no cambia una sola celda."""
    frames = _frames(_universe(80))
    base = context_matrix(frames, spec=context_spec())

    noisy = dict(frames)
    for name in context.CONTEXT_SERIES:
        if name == context.ANCHOR_SERIES:
            continue
        noisy[name] = frames[name].with_columns(
            pl.Series("close_bis", [13.0] * frames[name].height, dtype=pl.Float64()),
            pl.Series("open", [1.0] * frames[name].height, dtype=pl.Float64()),
        )
    assert store.matrix_sha256(context_matrix(noisy, spec=context_spec())) == store.matrix_sha256(
        base
    )

    # una `session` en `Datetime` (con o sin zona) se lee igual: solo cuenta el dia
    as_datetime = dict(frames)
    as_datetime["^GDAXI"] = frames["^GDAXI"].with_columns(
        pl.col("session").cast(pl.Datetime("us", "UTC"))
    )
    assert store.matrix_sha256(
        context_matrix(as_datetime, spec=context_spec())
    ) == store.matrix_sha256(base)

    null_as_of = dict(frames)
    anchor = null_as_of["^GSPC"]
    null_as_of["^GSPC"] = anchor.with_columns(
        pl.Series(
            "as_of",
            [None, *anchor.get_column("as_of").to_list()[1:]],
            dtype=pl.Datetime("us", "UTC"),
        )
    )
    with pytest.raises(store.ContextInputError, match="'as_of' nulo"):
        context_matrix(null_as_of, spec=context_spec())


# ─────────────────────────────────────────────────────────────────────────────
# A5 — alineamiento y huecos
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_the_lag_table_is_the_declared_one() -> None:
    """El rezago de cada mercado ajeno: 0 para Asia, 1 para Europa, el DXY y el VIX."""
    assert context.CONTEXT_MARKET_LAG == MARKET_LAG_TABLE
    assert set(context.CONTEXT_MARKET_LAG) == set(MARKET_LAG_TABLE)
    # el ancla no aparece: su rezago es 0 por definicion, no se elige
    assert context.ANCHOR_SERIES not in context.CONTEXT_MARKET_LAG
    assert set(store.CONTEXT_SECTOR_SERIES) & set(context.CONTEXT_MARKET_LAG) == set()
    assert set(context.CONTEXT_MARKET_LAG) | {context.ANCHOR_SERIES} | set(
        store.CONTEXT_SECTOR_SERIES
    ) == set(context.CONTEXT_SERIES)


def test_a5_a_foreign_holiday_takes_the_last_published_session() -> None:
    """Un festivo ajeno no da un cero: se toma la ultima sesion publicada."""
    count = 8
    sessions = _sessions(count)
    # El DAX cierra por vacaciones en la sesion 6 (indice 5) del calendario del S&P.
    dax = _series(count, seed=11, skip=(5,))
    assert dax[0] == [*sessions[:5], sessions[6], sessions[7]]

    data = _universe(count)
    data["^GDAXI"] = dax
    data["^FTSE"] = _series(count, seed=12)
    data["^STOXX50E"] = _series(count, seed=13)
    matrix = context_matrix(_frames(data), spec=context_spec())

    closes = dax[1]
    gdax_returns = _log_returns(closes)
    ftse_returns = _log_returns(data["^FTSE"][1])
    stoxx_returns = _log_returns(data["^STOXX50E"][1])
    # en la sesion 7 (indice 7) el ultimo cierre del DAX es el de la sesion 6 (indice 5)
    expected = statistics.fmean(
        [
            cast("float", gdax_returns[5]),
            cast("float", ftse_returns[6]),
            cast("float", stoxx_returns[6]),
        ]
    )
    europe = _column(matrix, "europe_prev_1")
    assert europe[7] == pytest.approx(expected)
    assert europe[7] != 0.0
    assert matrix.height == count
    assert all(name in matrix.columns for name in store.CONTEXT_FEATURE_COLUMNS)

    # una serie sin historia deja la feature a `null`, no a cero
    without_history = context_matrix(_frames({"^GSPC": _series(8, seed=1)}), spec=context_spec())
    assert _column(without_history, "europe_prev_1") == [None] * 8
    assert _column(without_history, "asia_overnight_1") == [None] * 8
    assert _column(without_history, "corr_dax_60") == [None] * 8


# ─────────────────────────────────────────────────────────────────────────────
# A6 — correlaciones moviles
# ─────────────────────────────────────────────────────────────────────────────
def _correlation_fixture(count: int = 80) -> dict[str, pl.DataFrame]:
    """Fixture con historia de sobra: las 19 series presentes desde la sesion 0."""
    return _frames(_universe(count))


def test_a6_the_moving_correlation_is_a_prefix_invariant_window() -> None:
    """(a) la fila ``t`` no cambia con el prefijo; (b) el pasado hondo tampoco la mueve."""
    count = 80
    frames = _correlation_fixture(count)
    full = context_matrix(frames, spec=context_spec())

    for prefix in (62, 70, count):
        cut = _sessions(count)[:prefix]
        truncated = {
            name: frame.filter(pl.col("session").is_in(cut)) for name, frame in frames.items()
        }
        partial = context_matrix(truncated, spec=context_spec())
        assert store.matrix_sha256(partial) == store.matrix_sha256(full.head(prefix))

    # (b) cambiar los pares **anteriores** a la ventana no altera la fila `t`
    target = count - 1
    window_start = target - WINDOW
    deep_past = dict(_universe(count))
    for name in ("^GSPC", "^GDAXI"):
        sessions, closes = deep_past[name]
        # el factor toca los cierres hasta la sesion 17: el primer retorno que entra
        # en la ventana es el de la sesion 19, asi que el retorno 18 sigue intacto
        deep_past[name] = (
            sessions,
            [2.0 * close for close in closes[: window_start - 1]] + closes[window_start - 1 :],
        )
    shifted = context_matrix(_frames(deep_past), spec=context_spec())
    for name, partner in (
        ("corr_dax_60", "^GDAXI"),
        ("corr_ftse_60", "^FTSE"),
        ("beta_vix_60", "^VIX"),
    ):
        assert _column(shifted, name)[target] == _column(full, name)[target], name
        assert _column(full, name)[target] is not None, partner
    assert full.height == count


def test_a6_the_correlation_window_and_its_edge_cases() -> None:
    """(c) proporcionales dan 1, varianza 0 da ``null``; (d) sin ventana completa, ``null``."""
    count = 80
    anchor_sessions, anchor_closes = _series(count, seed=1, base=100.0)
    data = _universe(count)
    data["^GSPC"] = (anchor_sessions, anchor_closes)
    # el DAX vale exactamente el doble: mismos retornos, correlacion 1
    data["^GDAXI"] = (anchor_sessions, [2.0 * close for close in anchor_closes])
    data["^FTSE"] = (anchor_sessions, anchor_closes)  # retornos identicos: 1 tambien
    matrix = context_matrix(_frames(data), spec=context_spec())

    assert _column(matrix, "corr_dax_60")[-1] == pytest.approx(1.0, rel=1e-12)
    assert _column(matrix, "corr_ftse_60")[-1] == pytest.approx(1.0, rel=1e-12)
    assert cast("float", _column(matrix, "corr_nikkei_60")[-1]) < 1.0

    # varianza cero en la otra serie (cierre constante) ⇒ null, nunca NaN ni inf
    flat = dict(data)
    flat["^GDAXI"] = (anchor_sessions, [50.0] * count)
    flat_matrix = context_matrix(_frames(flat), spec=context_spec())
    assert _column(flat_matrix, "corr_dax_60")[-1] is None
    _assert_finite(flat_matrix)

    corr = _column(matrix, "corr_dax_60")
    assert corr[: WINDOW + 1] == [None] * (WINDOW + 1)
    assert corr[WINDOW + 1] is not None
    assert statistics.fmean(cast("list[float]", corr[WINDOW + 1 :])) > 0.5

    # una correlacion que en doble precision no existe (escala subnormal) ⇒ null
    tiny = context._correlation([1e-200, 2e-200], [1e-200, 3e-200])  # pyright: ignore[reportPrivateUsage]
    assert tiny is None


# ─────────────────────────────────────────────────────────────────────────────
# A7 — Asia y Europa
# ─────────────────────────────────────────────────────────────────────────────
def _two_continents() -> dict[str, tuple[list[date], list[float]]]:
    """Tres sesiones: los **dos** mercados asiaticos y los **tres** europeos.

    Asia necesita dos mercados y Europa tres (``^GDAXI``, ``^FTSE`` y
    ``^STOXX50E``: su media es la de los tres). El DAX no cotiza la ultima sesion,
    para que se vea que el cierre europeo es el de la ultima sesion **anterior**.
    """
    sessions = _sessions(3)
    return {
        "^GSPC": (sessions, [100.0, 101.0, 103.0]),
        "^N225": (sessions, [39000.0, 39200.0, 39450.0]),
        "^HSI": (sessions, [20000.0, 19900.0, 20100.0]),
        "^GDAXI": (sessions[:2], [20000.0, 20200.0]),
        "^FTSE": (sessions, [8200.0, 8230.0, 8250.0]),
        "^STOXX50E": (sessions, [4900.0, 4915.0, 4930.0]),
    }


def test_a7_asia_and_europe_match_the_hand_computed_mean() -> None:
    """La media de los indices de cada continente, calculada a mano."""
    data = _two_continents()
    matrix = context_matrix(_frames(data), spec=context_spec())

    nikkei = cast("list[float]", _log_returns(data["^N225"][1]))
    hang_seng = cast("list[float]", _log_returns(data["^HSI"][1]))
    dax = cast("list[float]", _log_returns(data["^GDAXI"][1]))
    ftse = cast("list[float]", _log_returns(data["^FTSE"][1]))
    stoxx = cast("list[float]", _log_returns(data["^STOXX50E"][1]))

    asia = _column(matrix, "asia_overnight_1")
    europe = _column(matrix, "europe_prev_1")
    assert asia[0] is None and europe[0] is None
    # el cierre europeo de la sesion 1 seria el de la sesion 0, que no tiene retorno
    assert europe[1] is None
    assert asia[1] == pytest.approx(statistics.fmean([nikkei[1], hang_seng[1]]))
    assert asia[2] == pytest.approx(statistics.fmean([nikkei[2], hang_seng[2]]))
    # en la sesion 2 el DAX ya no cierra: entra su ultima sesion publicada, la 1
    assert dax[0] is None
    assert europe[2] == pytest.approx(statistics.fmean([dax[1], ftse[1], stoxx[1]]))
    assert ftse[1] != ftse[2]  # el fixture no es degenerado


def test_a7_only_asia_overnight_needs_the_cierre_of_t() -> None:
    """``asia_overnight_1`` es la unica columna cuyo ``required_as_of`` es la sesion ``t``."""
    same_session = {
        entry.name
        for entry in store.CONTEXT_FEATURE_CATALOG
        if not entry.required_as_of.endswith("t-1")
    }
    assert same_session == {"asia_overnight_1"}
    for entry in store.CONTEXT_FEATURE_CATALOG:
        assert entry.required_as_of == CATALOG_TABLE[entry.name][1]


# ─────────────────────────────────────────────────────────────────────────────
# A8 — beta del VIX
# ─────────────────────────────────────────────────────────────────────────────
def _vix_closes(count: int) -> list[float]:
    """Cierres del VIX del fixture de A8: se mueven **contra** el S&P."""
    spx_returns = _log_returns(_walk(count, seed=1, base=5000.0))
    closes = [18.0]
    for index in range(1, count):
        move = cast("float", spx_returns[index])
        closes.append(closes[-1] * math.exp(-2.0 * move + 0.0005 * _noise(seed=5, index=index)))
    return closes


def _beta_fixture(count: int = 70) -> dict[str, pl.DataFrame]:
    """El VIX se mueve contra el S&P: la pendiente tiene que salir negativa."""
    spx_sessions, spx_closes = _series(count, seed=1, base=5000.0)
    data = _universe(count)
    data["^GSPC"] = (spx_sessions, spx_closes)
    data["^VIX"] = (spx_sessions, _vix_closes(count))
    return _frames(data)


def test_a8_the_vix_beta_is_the_hand_computed_slope() -> None:
    """Pendiente OLS sobre los 60 pares ``s <= t-1``, calculada a mano (signo incluido)."""
    count = 70
    matrix = context_matrix(_beta_fixture(count), spec=context_spec())

    spx_returns = _log_returns(_walk(count, seed=1, base=5000.0))
    vix_returns = _log_returns(_vix_closes(count))
    assert vix_returns[0] is None and spx_returns[0] is None

    beta = _column(matrix, "beta_vix_60")
    assert beta[: WINDOW + 1] == [None] * (WINDOW + 1)
    for target in (WINDOW + 1, WINDOW + 5, count - 1):
        xs = [cast("float", spx_returns[index]) for index in range(target - WINDOW, target)]
        ys = [cast("float", vix_returns[index]) for index in range(target - WINDOW, target)]
        mean_x = statistics.fmean(xs)
        mean_y = statistics.fmean(ys)
        numerator = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
        denominator = math.fsum((x - mean_x) ** 2 for x in xs)
        assert beta[target] == pytest.approx(numerator / denominator)
    assert cast("float", beta[-1]) < 0.0


def test_a8_a_degenerate_scale_is_a_null() -> None:
    """Sin varianza no hay pendiente: ni con varianza exacta cero ni por desbordamiento."""
    # el mismo caso que la correlacion: en doble precision la varianza se va a cero
    assert (
        context._slope(  # pyright: ignore[reportPrivateUsage]
            [1e-200, 2e-200], [1e-200, 3e-200]
        )
        is None
    )
    assert (
        context._slope(  # pyright: ignore[reportPrivateUsage]
            [0.5, 0.5], [1.0, 2.0]
        )
        is None
    )


def test_a8_a_zero_variance_in_either_leg_is_a_null() -> None:
    """VIX constante o retornos del S&P constantes ⇒ ``null``, nunca ``NaN`` ni ``inf``."""
    count = 70
    sessions = _sessions(count)
    spx_sessions, spx_closes = _series(count, seed=1, base=5000.0)

    flat_vix = dict(_universe(count))
    flat_vix["^GSPC"] = (spx_sessions, spx_closes)
    flat_vix["^VIX"] = (sessions, [18.0] * count)
    flat_matrix = context_matrix(_frames(flat_vix), spec=context_spec())
    assert _column(flat_matrix, "beta_vix_60")[-1] is None
    _assert_finite(flat_matrix)

    flat_spx = dict(_universe(count))
    flat_spx["^GSPC"] = (sessions, [5000.0] * count)
    flat_spx["^VIX"] = (sessions, _walk(count, seed=3, base=18.0))
    spx_matrix = context_matrix(_frames(flat_spx), spec=context_spec())
    assert _column(spx_matrix, "beta_vix_60")[-1] is None
    _assert_finite(spx_matrix)


# ─────────────────────────────────────────────────────────────────────────────
# A9 — el DXY, sin duplicar retornos del S&P
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_the_dxy_return_is_the_last_session_before_t() -> None:
    """``ln(DXY/DXY_prev)`` de la ultima sesion del DXY ``< t``, contra el fixture."""
    count = 5
    # El DXY no cotiza la sesion 2 (indice 2) del calendario del S&P.
    dollar = _series(count, seed=4, base=104.0, skip=(2,))
    data = _universe(count)
    data["DX-Y.NYB"] = dollar
    matrix = context_matrix(_frames(data), spec=context_spec())

    sessions = _sessions(count)
    assert dollar[0] == [sessions[0], sessions[1], sessions[3], sessions[4]]
    returns = _log_returns(dollar[1])
    column = _column(matrix, "dxy_ret_1")
    assert column[0] is None and column[1] is None
    assert column[2] == pytest.approx(cast("float", returns[1]))
    # la sesion 3 no tiene cierre del DXY: repite su **ultimo retorno publicado**
    assert column[3] == pytest.approx(cast("float", returns[1]))
    assert column[4] == pytest.approx(cast("float", returns[2]))
    assert column[4] != column[3]
    assert column[3] != 0.0


def test_a9_the_sp500_return_is_not_duplicated_between_families() -> None:
    """``ret_1``/``ret_5``/``ret_21`` siguen solo en ``technical_v1`` y el solape no crece."""
    spx_returns = {"ret_1", "ret_5", "ret_21"}
    assert spx_returns <= set(store.TECHNICAL_FEATURE_COLUMNS)
    assert spx_returns & set(store.CONTEXT_FEATURE_COLUMNS) == set()
    assert "dxy_ret_1" in store.CONTEXT_FEATURE_COLUMNS

    overlap = set(store.TECHNICAL_FEATURE_COLUMNS) & set(store.CONTEXT_FEATURE_COLUMNS)
    assert overlap == set()
    # el unico solape entre los tres catalogos sigue siendo `atr_norm` (#72)
    assert set(store.FEATURE_COLUMNS) & set(store.TECHNICAL_FEATURE_COLUMNS) == {"atr_norm"}
    assert set(store.FEATURE_COLUMNS) & set(store.CONTEXT_FEATURE_COLUMNS) == set()
    assert len(store.ALL_FEATURE_COLUMNS) == len(
        set(store.FEATURE_COLUMNS)
        | set(store.TECHNICAL_FEATURE_COLUMNS)
        | set(store.CONTEXT_FEATURE_COLUMNS)
    )


# ─────────────────────────────────────────────────────────────────────────────
# A10 — dispersion sectorial
# ─────────────────────────────────────────────────────────────────────────────
def _sector_fixture(*, with_xlre: bool = True, with_xlc: bool = True) -> dict[str, pl.DataFrame]:
    """Las 19 series, con ``XLRE`` y ``XLC`` presentes o sin nacer todavia."""
    data = _universe(6)
    for index, name in enumerate(store.CONTEXT_SECTOR_SERIES):
        data[name] = _series(6, seed=30 + index, base=90.0 + index)
    data[context.CONTEXT_SERIES[0]] = _series(6, seed=1)
    if not with_xlre:
        data["XLRE"] = ([], [])
    if not with_xlc:
        data["XLC"] = ([], [])
    return _frames(data)


def test_a10_the_dispersion_uses_the_sectors_with_data() -> None:
    """9, 10 y 11 sectores: la dispersion es un numero en los tres casos (A10)."""
    for with_xlre, with_xlc, expected in (
        (False, False, 9),
        (True, False, 10),
        (True, True, 11),
    ):
        matrix = context_matrix(
            _sector_fixture(with_xlre=with_xlre, with_xlc=with_xlc), spec=context_spec()
        )
        counts = matrix.get_column("sector_count").to_list()
        assert counts[-1] == expected, (with_xlre, with_xlc)
        dispersion = _column(matrix, "sector_dispersion_1")[-1]
        assert dispersion is not None
        assert dispersion > 0.0

        # la dispersion es la desviacion estandar muestral (ddof=1) de los que tienen dato
        target = 5
        returns = [
            cast(
                "float",
                _log_returns(_series(6, seed=30 + index, base=90.0 + index)[1])[target - 1],
            )
            for index, name in enumerate(store.CONTEXT_SECTOR_SERIES)
            if not (name == "XLRE" and not with_xlre) and not (name == "XLC" and not with_xlc)
        ]
        assert dispersion == pytest.approx(statistics.stdev(returns))
        assert len(returns) == expected


def test_a10_less_than_two_sectors_is_a_null_with_a_real_count() -> None:
    """Con un solo sector la dispersion no existe, pero el recuento sigue siendo real."""
    data = _universe(6)
    for index, name in enumerate(store.CONTEXT_SECTOR_SERIES):
        data[name] = _series(6, seed=30 + index, base=90.0 + index)
    for name in store.CONTEXT_SECTOR_SERIES[1:]:
        data[name] = ([], [])
    matrix = context_matrix(_frames(data), spec=context_spec())
    assert matrix.get_column("sector_count").to_list()[-1] == 1
    assert _column(matrix, "sector_dispersion_1")[-1] is None
    assert _column(matrix, "sector_dispersion_1_z")[-1] is None
    _assert_finite(matrix)


def test_a10_the_z_column_comes_from_the_imported_normalisation() -> None:
    """El AST importa ``normalise_expanding`` y **no** define una normalizacion propia."""
    imported: set[str] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom) and node.module == "cfdtrader.features.store":
            imported.update(alias.name for alias in node.names)
    assert "normalise_expanding" in imported
    assert "MAD_SCALE" not in imported  # la escala vive en #19, no se copia

    defined = {node.name for node in ast.walk(_tree()) if isinstance(node, ast.FunctionDef)}
    assert not defined & {"normalise_expanding", "median", "mad"}

    matrix = _matrix(80)
    dispersion = _column(matrix, "sector_dispersion_1")
    expected = store.normalise_expanding(
        pl.DataFrame(
            {
                "disp": pl.Series("disp", dispersion, dtype=pl.Float64()),
            }
        ),
        "disp",
        min_sessions=MIN_SESSIONS,
    )
    assert _column(expected, "disp_z") == _column(matrix, "sector_dispersion_1_z")
    assert expected.height == 80


# ─────────────────────────────────────────────────────────────────────────────
# A11 — universo, ``as_of`` y nulos
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_the_universe_is_exactly_the_anchor_sessions() -> None:
    """Una fila por sesion de ``^GSPC``, en orden, con el ``as_of`` copiado en UTC."""
    expected = _expected()
    matrix = _golden_matrix()
    table = _golden_table()
    sessions = [cast("date", value) for value in table.get_column("session").to_list()]

    assert matrix.height == expected["sessions"] == len(sessions) == 300
    assert matrix.get_column("session").to_list() == sessions
    assert matrix.get_column("as_of").dtype == pl.Datetime("us", "UTC")

    for session, instant in zip(
        matrix.get_column("session").to_list(),
        matrix.get_column("as_of").to_list(),
        strict=True,
    ):
        moment = cast("datetime", instant)
        assert moment.tzinfo is not None and moment.utcoffset() == timedelta(0)
        assert moment.date() == session


def test_a11_every_cell_is_finite_or_null_and_nothing_is_filled_with_zero() -> None:
    """Ninguna columna con ``NaN`` ni ``inf``, tampoco tras ``daily_records``."""
    _, matrix = _golden()
    _assert_finite(matrix)
    records = store.daily_records(
        matrix, spec=context_spec(), series_id="^GSPC", fetched_at=FETCHED_AT
    )
    for record in records:
        for name in store.CONTEXT_FEATURE_COLUMNS:
            value = cast("float | None", record[name])
            assert value is None or isinstance(value, float)
            if value is not None:
                assert math.isfinite(value)

    expected = _expected()
    counts = {name: matrix.get_column(name).null_count() for name in store.CONTEXT_FEATURE_COLUMNS}
    assert counts["corr_stoxx_60"] == expected["non_null"]["corr_stoxx_60"] == 131
    assert counts["dxy_ret_1"] == 2
    assert counts["sector_count"] == 0

    # los huecos son `null`, nunca un cero de relleno
    assert _column(matrix, "dxy_ret_1")[0] is None
    assert _column(matrix, "asia_overnight_1")[0] is None
    assert _column(matrix, "corr_stoxx_60")[0] is None
    assert _column(matrix, "sector_dispersion_1")[0] is None
    assert matrix.get_column("sector_count").to_list()[0] == 0


def test_a11_a_late_series_leaves_null_and_the_clean_sample_cutoff_is_not_applied() -> None:
    """``^STOXX50E`` entra tarde ⇒ ``null``; y 2005 no se recorta (no es este modulo)."""
    _, matrix = _golden()
    stoxx = _column(matrix, "corr_stoxx_60")
    assert stoxx[:131] == [None] * 131
    assert stoxx[131] is not None

    # una sesion de 2005 (muy anterior al corte de muestra limpia de 2014-01-01) se
    # publica igual: el corte es una restriccion de estudio, no del almacen
    old_sessions = _sessions(3, start=date(2005, 1, 3))
    data = _universe(3)
    data["^GSPC"] = (old_sessions, _walk(3, seed=1))
    old = context_matrix(_frames(data), spec=context_spec())
    assert old.height == 3
    assert old.get_column("session").to_list() == old_sessions
    assert not any(module.startswith("cfdtrader.analysis") for module in _imported_modules())


def test_a11_an_anchor_as_of_that_does_not_match_its_session_is_an_error() -> None:
    """El ``as_of`` de la fila tiene que ser el de su sesion: desplazarlo es un error."""
    frames = _frames(_universe(25))
    anchor = frames["^GSPC"]
    instants = [cast("datetime", value) for value in anchor.get_column("as_of").to_list()]
    shifted = dict(frames)
    shifted["^GSPC"] = anchor.with_columns(
        pl.Series(
            "as_of",
            [instant + timedelta(hours=6) for instant in instants],
            dtype=pl.Datetime("us", "UTC"),
        )
    )
    with pytest.raises(store.ContextInputError, match="no corresponden al mismo dia"):
        context_matrix(shifted, spec=context_spec())


def test_a11_a_null_or_non_positive_close_leaves_the_return_as_null() -> None:
    """Un cierre nulo, cero o negativo no se convierte en un retorno: la feature es ``null``."""
    count = 7
    sessions = _sessions(count)
    frames = _frames(_universe(count))
    frames["^GDAXI"] = pl.DataFrame(
        {
            "session": pl.Series("session", sessions, dtype=pl.Date()),
            "close": pl.Series(
                "close",
                [100.0, 101.0, None, 103.0, 0.0, 105.0, -1.0],
                dtype=pl.Float64(),
            ),
        }
    )
    matrix = context_matrix(frames, spec=context_spec())
    europe = _column(matrix, "europe_prev_1")
    assert europe[2] is not None  # el unico retorno que llega a existir (sesion 1)
    assert europe[3] is None  # la sesion 2 no tiene cierre
    assert europe[4] is None  # 0.0 no es un precio
    assert europe[5] is None  # el cierre anterior era 0.0
    assert europe[6] is None  # -1.0 tampoco
    assert _column(matrix, "corr_dax_60") == [None] * count
    _assert_finite(matrix)


# ─────────────────────────────────────────────────────────────────────────────
# A12 — persistencia
# ─────────────────────────────────────────────────────────────────────────────
def _volatility_matrix(count: int) -> pl.DataFrame:
    """Matriz de la familia de volatilidad, para probar la convivencia."""
    sessions = _sessions(count)
    closes = _walk(count, seed=9)
    frame = pl.DataFrame(
        {
            "session": pl.Series("session", sessions, dtype=pl.Date()),
            "open": pl.Series("open", closes, dtype=pl.Float64()),
            "high": pl.Series("high", [value * 1.01 for value in closes], dtype=pl.Float64()),
            "low": pl.Series("low", [value * 0.99 for value in closes], dtype=pl.Float64()),
            "close": pl.Series("close", closes, dtype=pl.Float64()),
            "vix_close": pl.Series("vix_close", [18.0 + (index % 7) for index in range(count)]),
        }
    )
    matrix = store.build_matrix(frame, spec=store.FeatureSpec())
    return matrix.with_columns(
        pl.Series("as_of", [_session_close(s) for s in sessions], dtype=pl.Datetime("us", "UTC"))
    )


def test_a12_the_three_families_coexist_in_the_same_dataset(tmp_path: Path) -> None:
    """La familia de contexto no sustituye a las otras dos: las separa el ``source``."""
    root = tmp_path / "store"
    handle = Store(root)
    count = 260
    context_rows = _matrix(count)

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
    technical_frame = pl.DataFrame(
        {
            "session": pl.Series("session", _sessions(count), dtype=pl.Date()),
            "high": pl.Series("high", _walk(count, seed=3), dtype=pl.Float64()),
            "low": pl.Series("low", _walk(count, seed=4), dtype=pl.Float64()),
            "close": pl.Series("close", _walk(count, seed=5), dtype=pl.Float64()),
        }
    )
    technical_matrix_frame = technical.technical_matrix(
        technical_frame, spec=technical.technical_spec()
    )
    store.save_daily(
        handle,
        spec=technical.technical_spec(),
        matrix=technical_matrix_frame.with_columns(
            pl.Series(
                "as_of",
                [_session_close(s) for s in _sessions(count)],
                dtype=pl.Datetime("us", "UTC"),
            )
        ),
        series_id="^GSPC",
        fetched_at=FETCHED_AT,
    )
    store.save_daily(
        handle,
        spec=context_spec(),
        matrix=context_rows,
        series_id="^GSPC",
        fetched_at=FETCHED_AT,
    )

    rows = Store(root).sql(
        f"SELECT source, count(*) AS n FROM {store.FEATURES_LAYER}.{store.FEATURES_DATASET} "  # noqa: S608
        "GROUP BY source ORDER BY source"
    )
    assert rows.get_column("source").to_list() == sorted(
        [store.FEATURES_SOURCE, store.TECHNICAL_FEATURES_SOURCE, CONTEXT_SOURCE]
    )
    assert rows.get_column("n").to_list() == [count, count, count]

    context_rows_loaded = store.load_daily(handle, series_id="^GSPC", feature_set="context_v1")
    assert context_rows_loaded.height == count
    assert set(context_rows_loaded.get_column("source").to_list()) == {CONTEXT_SOURCE}
    for name in store.CONTEXT_FEATURE_COLUMNS:
        assert context_rows_loaded.get_column(name).is_not_null().any()
    for foreign in FOREIGN_COLUMNS:
        assert context_rows_loaded.get_column(foreign).is_null().all()
    assert _versions(root, source=store.FEATURES_SOURCE) == [1] * count
    assert _versions(root, source=CONTEXT_SOURCE) == [1] * count

    with pytest.raises(store.InvalidFeatureSpecError, match="no esta registrada"):
        store.load_daily(handle, series_id="^GSPC", feature_set="volatility_v2")

    # una familia sin filas en un dataset que si existe tampoco devuelve vacio
    context_only = tmp_path / "context_only"
    store.save_daily(
        Store(context_only),
        spec=context_spec(),
        matrix=context_rows,
        series_id="^GSPC",
        fetched_at=FETCHED_AT,
    )
    with pytest.raises(UnknownDatasetError, match="no tiene filas de la familia"):
        store.load_daily(
            Store(context_only), series_id="^GSPC", feature_set=store.TECHNICAL_FEATURE_SET
        )


def test_a12_unchanged_is_free_and_a_change_bumps_the_revision(tmp_path: Path) -> None:
    """Contenido identico ⇒ ``UNCHANGED`` sin Parquet nuevo; distinto ⇒ ``version + 1``."""
    root = tmp_path / "store"
    handle = Store(root)
    spec = context_spec()
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
    assert _versions(root, source=CONTEXT_SOURCE) == [1] * 40

    bumped = store.FeatureSpec(
        feature_set=store.CONTEXT_FEATURE_SET,
        code_version=spec.code_version + 1,
        windows=store.DEFAULT_CONTEXT_WINDOWS,
        sources=store.DEFAULT_CONTEXT_SOURCES,
    )
    assert save(bumped, fetched_at=FETCHED_AT) is WriteOutcome.CREATED
    assert _versions(root, source=CONTEXT_SOURCE) == [2] * 40

    records = store.daily_records(matrix, spec=spec, series_id="^GSPC", fetched_at=FETCHED_AT)
    for record in records:
        assert "version" not in record
        assert record["fetched_at"] is FETCHED_AT
        assert record["source"] == CONTEXT_SOURCE
        assert record["features_version"] == store.features_version(
            spec, cast("datetime", record["as_of"])
        )


# ─────────────────────────────────────────────────────────────────────────────
# A13 — golden de contexto
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_the_context_golden_is_frozen() -> None:
    """El par (``code_version``, digests) del golden no puede moverse en silencio."""
    expected = _expected()
    spec, matrix = _golden()

    assert (
        list(matrix.columns)
        == expected["feature_columns"]
        == [
            "session",
            "as_of",
            *store.CONTEXT_FEATURE_COLUMNS,
        ]
    )
    assert matrix.height == expected["sessions"] == 300
    assert _prefixed(store.feature_spec_sha256(spec)) == expected["feature_spec_sha256"]
    assert _prefixed(store.matrix_sha256(matrix)) == expected["matrix_sha256"]
    assert expected["code_version"] == store.FEATURE_CODE_VERSION
    assert spec == context_spec()
    assert expected["feature_set"] == store.CONTEXT_FEATURE_SET
    assert expected["windows"] == store.DEFAULT_CONTEXT_WINDOWS
    assert expected["sources"] == [list(pair) for pair in store.DEFAULT_CONTEXT_SOURCES]
    assert expected["digest_format"] == "sha256:<hex>"

    counts = matrix.null_count().to_dicts()[0]
    assert {name: value for name, value in counts.items() if name != "session"} == (
        expected["non_null"]
    )
    assert [int(value) for value in matrix.get_column("sector_count").to_list()] == (
        expected["sector_count"]
    )
    assert sorted(set(expected["sector_count"])) == [0, 9, 10, 11]

    # el golden falla si el calculo se mueve sin subir la constante declarada
    assert _prefixed(store.matrix_sha256(matrix.drop("dxy_ret_1"))) != expected["matrix_sha256"]
    assert _prefixed(store.matrix_sha256(matrix.drop("sector_count"))) != expected["matrix_sha256"]
    assert all(
        digest.startswith(store.FEATURE_VERSION_PREFIX)
        for digest in (expected["feature_spec_sha256"], expected["matrix_sha256"])
    )
    assert _golden_table().height == 300


def test_a13_the_golden_fixture_declares_the_nineteen_series() -> None:
    """El fixture de entrada es **propio** (19 series) y trae los huecos de verdad."""
    table = _golden_table()
    assert table.columns == ["session", *context.CONTEXT_SERIES]
    assert len(table.columns) - 1 == 19
    assert table.get_column("^GSPC").null_count() == 0
    assert table.get_column("^GDAXI").null_count() == 3
    assert table.get_column("^STOXX50E").null_count() == 72
    assert table.get_column("XLRE").null_count() == 120
    assert table.get_column("XLC").null_count() == 180


# ─────────────────────────────────────────────────────────────────────────────
# A14 — determinismo y puertas
# ─────────────────────────────────────────────────────────────────────────────
_DIGEST_SCRIPT = """
import hashlib, json, pathlib, sys
from datetime import UTC, datetime

import polars as pl

from cfdtrader.features import context, store

root, csv_path = sys.argv[1], sys.argv[2]
table = pl.read_csv(csv_path, try_parse_dates=True, infer_schema_length=None)
frames = {}
for name in context.CONTEXT_SERIES:
    frame = table.select("session", pl.col(name).alias("close")).drop_nulls("close")
    if name == context.ANCHOR_SERIES:
        frame = frame.with_columns(
            pl.Series(
                "as_of",
                [
                    datetime(value.year, value.month, value.day, 20, tzinfo=UTC)
                    for value in frame.get_column("session").to_list()
                ],
                dtype=pl.Datetime("us", "UTC"),
            )
        )
    frames[name] = frame

spec = context.context_spec()
matrix = context.context_matrix(frames, spec=spec)
handle = store.Store(root)
store.save_daily(
    handle,
    spec=spec,
    matrix=matrix,
    series_id="^GSPC",
    fetched_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
)
rows = store.load_daily(handle, series_id="^GSPC", feature_set="context_v1")
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
            str(FIXTURES / "context_golden_inputs.csv"),
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
    assert digests[0]["source"] == CONTEXT_SOURCE

    first, second = tmp_path / "pass_1", tmp_path / "pass_2"
    for root in (first, second):
        store.save_daily(
            Store(root),
            spec=spec,
            matrix=matrix,
            series_id="^GSPC",
            fetched_at=FETCHED_AT,
        )
    assert _parquet_digest(first) == _parquet_digest(second) == digests[0]["parquet_sha256"]
    loaded = store.load_daily(Store(first), series_id="^GSPC", feature_set="context_v1")
    assert loaded.get_column("features_version")[0] == digests[0]["features_version"]


def test_a14_the_module_does_not_cheat_the_gates() -> None:
    """La cobertura no se maquilla: sin ``pragma``, sin ``type: ignore``."""
    source = _source()
    assert "pragma: no cover" not in source
    assert "type: ignore" not in source
    assert "coverage" not in source.lower()
    assert context_matrix.__module__ == "cfdtrader.features.context"
    assert "\t" not in source
