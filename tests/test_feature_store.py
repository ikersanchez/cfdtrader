"""Tests del *feature store* con versionado y hash (tarea #19).

Un test por criterio, ``test_aN_...``. La raiz del ``Store`` vive siempre bajo
``tmp_path``: la fixture de ``tests/conftest.py`` huella el ``data/`` **y** el
``runs/`` del repositorio y falla si la sesion escribe en ellos.

El *golden dataset* de A4 vive en ``tests/fixtures/features/``: los inputs en
``golden_inputs.csv`` y el par esperado (``code_version`` + digests) en
``golden_expected.json``.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import polars as pl
import pytest

from cfdtrader.data.store import ImmutableWriteError, Store, UnknownDatasetError, WriteOutcome
from cfdtrader.features import store
from cfdtrader.features.volatility import add_features

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "features"

#: Primera sesion sintetica. El modulo no conoce el calendario a proposito.
FIRST_SESSION = date(2025, 1, 2)


def _sessions(count: int) -> list[date]:
    """``count`` sesiones consecutivas desde la primera del test."""
    return [FIRST_SESSION + timedelta(days=index) for index in range(count)]


#: 40 sesiones consecutivas.
SESSIONS: tuple[date, ...] = tuple(_sessions(40))

#: 300 sesiones: suficientes para que la ventana expandida del VIX (250) deje valores.
LONG_SESSIONS: tuple[date, ...] = tuple(_sessions(300))

#: Hora UTC que el test usa como cierre de sesion. Es un valor **del test**: el
#: modulo no conoce el calendario ni lee el reloj.
CLOSE_HOUR_UTC = 20

#: Instante de captura de los tests: posterior a todas las sesiones usadas.
FETCHED_AT = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def _session_close(session: date) -> datetime:
    """Cierre de sesion en UTC de una sesion sintetica."""
    return datetime(session.year, session.month, session.day, CLOSE_HOUR_UTC, tzinfo=UTC)


def _inputs(sessions: Sequence[date]) -> pl.DataFrame:
    """Frame de entrada sintetico y determinista (sin RNG) de la familia de volatilidad."""
    closes = [100.0 + index * 0.5 + (index % 5) * 0.25 for index in range(len(sessions))]
    return pl.DataFrame(
        {
            "session": list(sessions),
            "open": [value * 1.001 for value in closes],
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "vix_close": [20.0 + (index % 11) * 0.5 for index in range(len(sessions))],
        }
    )


def _with_as_of(matrix: pl.DataFrame, sessions: Sequence[date]) -> pl.DataFrame:
    """Anade la columna ``as_of`` (cierre de sesion en UTC) a una matriz."""
    instants = pl.Series("as_of", [_session_close(session) for session in sessions])
    return matrix.with_columns(instants)


def _matrix(
    sessions: Sequence[date] = SESSIONS, *, spec: store.FeatureSpec | None = None
) -> pl.DataFrame:
    """Matriz persistible (session + as_of + features del catalogo)."""
    frame = _inputs(sessions)
    matrix = store.build_matrix(frame, spec=spec or store.FeatureSpec())
    return _with_as_of(matrix, sessions)


def _prefixed(digest: str) -> str:
    """Forma en la que el golden guarda un digest: con el prefijo ``sha256:``.

    Es el mismo prefijo que publica ``features_version`` y, ademas, un hex desnudo
    de 64 caracteres dispara el hook ``detect-secrets`` (falso positivo de alta
    entropia). El test lo quita al comparar.
    """
    return store.FEATURE_VERSION_PREFIX + digest


def _expected() -> dict[str, Any]:
    """Par esperado congelado de A4."""
    raw = (FIXTURES / "golden_expected.json").read_text(encoding="utf-8")
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


def _golden() -> tuple[store.FeatureSpec, pl.DataFrame]:
    """Matriz del golden dataset sin ``as_of``, tal y como la congela A4."""
    spec = _frozen_spec(_expected())
    frame = pl.read_csv(FIXTURES / "golden_inputs.csv", try_parse_dates=True)
    return spec, store.build_matrix(frame, spec=spec)


def _golden_sessions() -> list[date]:
    """Sesiones del golden, para poder situar su ``as_of``."""
    frame = pl.read_csv(FIXTURES / "golden_inputs.csv", try_parse_dates=True)
    return list(frame.get_column("session").to_list())


def _parquet_digest(root: Path) -> str:
    """Digest del **contenido** de los Parquet de un almacen, sin mirar los nombres."""
    files = sorted(root.rglob("*.parquet"))
    digests = sorted(hashlib.sha256(path.read_bytes()).hexdigest() for path in files)
    return hashlib.sha256("\n".join(digests).encode("utf-8")).hexdigest()


def _stored_rows(root: Path) -> pl.DataFrame:
    """Todo lo que hay en disco, sin pasar por las vistas (para ver las revisiones superadas)."""
    files = sorted(root.rglob("*.parquet"))
    return pl.concat([pl.read_parquet(path) for path in files])


def _versions(root: Path) -> list[int]:
    """``version`` de la revision vigente de cada sesion, tal y como la ve ``sql()``."""
    frame = Store(root).sql(
        f"SELECT version FROM {store.FEATURES_LAYER}.{store.FEATURES_DATASET}"  # noqa: S608
    )
    return sorted(int(value) for value in frame.get_column("version").to_list())


def _module_copy(tmp_path: Path, name: str, source: str) -> Any:
    """Importa una **copia** del modulo desde ``tmp_path``, para tocar su texto.

    Devuelve ``Any`` a proposito: el objeto es un modulo dinamico y sus miembros
    no se pueden tipar estaticamente.
    """
    path = tmp_path / f"feature_store_{name}.py"
    path.write_text(source, encoding="utf-8")
    module_name = f"_feature_store_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    # `dataclass` resuelve las anotaciones diferidas (`from __future__ import
    # annotations`) buscando el modulo en `sys.modules`: sin registrarlo, la copia
    # no se puede ni importar.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _source() -> str:
    """Codigo fuente del modulo, del fichero real (no del bytecode)."""
    return Path(store.__file__).read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# A1 — que entra en el hash
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_features_version_covers_the_six_fields() -> None:
    """Mutar por separado cada uno de los seis campos da seis valores distintos."""
    base = store.FeatureSpec()
    instant = _session_close(SESSIONS[0])
    without_ret_sq = {name: value for name, value in base.windows.items() if name != "ret_sq"}
    variants = {
        "feature_set": (replace(base, feature_set="volatility_v2"), instant),
        "code_version": (replace(base, code_version=base.code_version + 1), instant),
        "parameters": (replace(base, parameters={"atr_window": 21}), instant),
        "windows": (replace(base, windows=without_ret_sq), instant),
        "sources": (replace(base, sources=(*base.sources, ("raw.market_daily", "SPY"))), instant),
        "as_of": (base, _session_close(SESSIONS[1])),
    }
    versions = {
        name: store.features_version(spec, moment) for name, (spec, moment) in variants.items()
    }

    assert sorted(versions) == [
        "as_of",
        "code_version",
        "feature_set",
        "parameters",
        "sources",
        "windows",
    ]
    assert len(set(versions.values())) == 6
    assert store.features_version(base, instant) not in versions.values()
    assert all(value.startswith(store.FEATURE_VERSION_PREFIX) for value in versions.values())


def test_a1_reordering_an_input_dict_changes_nothing() -> None:
    """El hash es de un JSON canonico: reordenar una entrada no puede cambiar el digest."""
    base = store.FeatureSpec()
    parameters = {"alpha": 1, "beta": 2.5, "gamma": [1, 2, 3]}
    reordered_parameters = {"gamma": [1, 2, 3], "beta": 2.5, "alpha": 1}
    reversed_windows = dict(reversed(list(base.windows.items())))
    reversed_sources = tuple(reversed(base.sources))

    assert store.feature_spec_sha256(store.FeatureSpec(parameters=reordered_parameters)) == (
        store.feature_spec_sha256(store.FeatureSpec(parameters=parameters))
    )
    assert store.feature_spec_sha256(store.FeatureSpec(windows=reversed_windows)) == (
        store.feature_spec_sha256(base)
    )
    assert store.feature_spec_sha256(store.FeatureSpec(sources=reversed_sources)) == (
        store.feature_spec_sha256(base)
    )


# ─────────────────────────────────────────────────────────────────────────────
# A2 — "codigo" es una constante declarada, no los bytes del fichero
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_code_version_is_a_declared_constant(tmp_path: Path) -> None:
    """Reescribir el modulo no cambia el hash; subir la constante, si."""
    assert store.FEATURE_CODE_VERSION == 1
    assert store.FeatureSpec().code_version == store.FEATURE_CODE_VERSION

    source = _source()
    instant = _session_close(SESSIONS[0])
    baseline = store.features_version(store.FeatureSpec(), instant)

    rewritten_source = source.replace(
        '"""Persistencia versionada', '"""  Persistencia versionada'
    ).replace("    #: Capa del almacen", "    #:  Capa del almacen")
    rewritten_source += "\n# reescritura deliberada: comentario y formato no entran en el hash\n"
    assert rewritten_source != source

    rewritten = _module_copy(tmp_path, "rewritten", rewritten_source)
    assert rewritten.features_version(rewritten.FeatureSpec(), instant) == baseline

    bumped_source = source.replace(
        "FEATURE_CODE_VERSION: Final[int] = 1", "FEATURE_CODE_VERSION: Final[int] = 2"
    )
    bumped = _module_copy(tmp_path, "bumped", bumped_source)
    assert bumped.FEATURE_CODE_VERSION == store.FEATURE_CODE_VERSION + 1
    assert bumped.FeatureSpec().code_version == 2
    assert bumped.features_version(bumped.FeatureSpec(), instant) != baseline


# ─────────────────────────────────────────────────────────────────────────────
# A3 — dos valores, no uno
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_two_values_not_one() -> None:
    """``feature_spec_sha256`` no mira el dia; ``features_version`` si."""
    spec = store.FeatureSpec()
    first, second = (
        store.features_version(spec, _session_close(SESSIONS[0])),
        store.features_version(spec, _session_close(SESSIONS[1])),
    )
    digest = store.feature_spec_sha256(spec)

    assert first != second
    assert digest == store.feature_spec_sha256(store.FeatureSpec())
    assert store.FEATURE_VERSION_PREFIX not in digest
    assert len(digest) == 64
    assert int(digest, 16) >= 0
    assert first != digest
    assert first.removeprefix(store.FEATURE_VERSION_PREFIX) != digest


# ─────────────────────────────────────────────────────────────────────────────
# A4 — golden dataset congelado
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_golden_dataset_matches_the_frozen_pair() -> None:
    """El par (``code_version``, digest) del golden no puede moverse en silencio."""
    expected = _expected()
    spec, matrix = _golden()

    assert list(matrix.columns) == expected["feature_columns"]
    assert matrix.height == expected["sessions"]
    assert _prefixed(store.feature_spec_sha256(spec)) == expected["feature_spec_sha256"]
    assert _prefixed(store.matrix_sha256(matrix)) == expected["matrix_sha256"]

    counts = matrix.null_count().to_dicts()[0]
    non_null = {name: value for name, value in counts.items() if name != "session"}
    assert non_null == expected["non_null"]


def test_a4_a_silent_change_fails_and_a_bump_requires_updating_the_pair() -> None:
    """Las dos direcciones del contrato: sin subir la constante no se puede cambiar el calculo."""
    expected = _expected()
    _, matrix = _golden()

    # (a) la constante declarada tiene que ser la del golden: subirla obliga a actualizar el par
    assert expected["code_version"] == store.FEATURE_CODE_VERSION
    # (b) la spec por defecto es la del golden: cambiar defaults obliga a lo mismo
    assert store.FeatureSpec() == _frozen_spec(expected)
    # (c) si el digest cambia sin tocar la constante, el test falla
    assert _prefixed(store.matrix_sha256(matrix)) == expected["matrix_sha256"]
    assert _prefixed(store.matrix_sha256(matrix.drop("ret_sq"))) != expected["matrix_sha256"]


# ─────────────────────────────────────────────────────────────────────────────
# A5 — esquema persistido
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_persisted_schema_is_wide_and_one_row_per_session(tmp_path: Path) -> None:
    """Las seis columnas del ``Store`` + los dos hashes + una columna por feature."""
    root = tmp_path / "store"
    matrix = _matrix()
    outcome = store.save_daily(
        Store(root),
        spec=store.FeatureSpec(),
        matrix=matrix,
        series_id="^GSPC",
        fetched_at=FETCHED_AT,
    )
    assert outcome is WriteOutcome.CREATED

    frame = Store(root).sql(
        f"SELECT * FROM {store.FEATURES_LAYER}.{store.FEATURES_DATASET}"  # noqa: S608
    )
    expected_columns = {
        "source",
        "series_id",
        "as_of",
        "fetched_at",
        "published_at",
        "version",
        *store.VERSION_COLUMNS,
        *store.FEATURE_COLUMNS,
    }
    assert set(frame.columns) == expected_columns
    assert frame.height == len(SESSIONS)
    assert "session" not in frame.columns  # el ancla es `as_of`, no una columna de payload
    assert set(frame.get_column("source").to_list()) == {store.FEATURES_SOURCE}
    assert set(frame.get_column("series_id").to_list()) == {"^GSPC"}
    assert sorted(frame.get_column("version").to_list()) == [1] * len(SESSIONS)
    assert frame.get_column("published_at").to_list() == [None] * len(SESSIONS)
    assert frame.get_column("as_of").n_unique() == len(SESSIONS)

    persisted = frame.sort("as_of").get_column("parkinson_rv").to_list()
    assert persisted == matrix.sort("session").get_column("parkinson_rv").to_list()
    assert frame.get_column("features_version").n_unique() == len(SESSIONS)
    assert frame.get_column("feature_spec_sha256").n_unique() == 1


# ─────────────────────────────────────────────────────────────────────────────
# A6 — inmutabilidad del Store respetada
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_immutability_is_respected_and_revisions_supersede(tmp_path: Path) -> None:
    """Escribir dos veces lo mismo no toca el disco; un cambio sube la revision."""
    root = tmp_path / "store"
    handle = Store(root)
    spec = store.FeatureSpec()
    matrix = _matrix()

    def save(current: store.FeatureSpec, *, fetched_at: datetime) -> WriteOutcome:
        return store.save_daily(
            handle, spec=current, matrix=matrix, series_id="^GSPC", fetched_at=fetched_at
        )

    assert save(spec, fetched_at=FETCHED_AT) is WriteOutcome.CREATED
    first_digest = _parquet_digest(root)
    first_files = sorted(root.rglob("*.parquet"))
    assert len(first_files) == 1

    # (1) mismo contenido (aunque cambie `fetched_at`) ⇒ UNCHANGED, mismo version, sin Parquet nuevo
    later = FETCHED_AT + timedelta(days=1)
    assert save(spec, fetched_at=later) is WriteOutcome.UNCHANGED
    assert _parquet_digest(root) == first_digest
    assert sorted(root.rglob("*.parquet")) == first_files
    assert _versions(root) == [1] * len(SESSIONS)

    # (2) el modulo escribe con `replace`, nunca con `append`: si lo llamara, esto lanzaria
    def forbidden(*_: object, **__: object) -> WriteOutcome:
        raise AssertionError("save_daily no puede escribir features con Store.append")

    original = Store.append
    cast("Any", Store).append = forbidden
    try:
        bumped = replace(spec, code_version=spec.code_version + 1)
        assert save(bumped, fetched_at=later) is WriteOutcome.CREATED
    finally:
        cast("Any", Store).append = original

    # (3) una sola revision vigente por sesion, y la anterior sigue en disco
    assert _versions(root) == [2] * len(SESSIONS)
    assert len(sorted(root.rglob("*.parquet"))) > len(first_files)
    assert _stored_rows(root).height == 2 * len(SESSIONS)
    assert set(_stored_rows(root).get_column("version").to_list()) == {1, 2}

    # (4) load_daily devuelve la vigente
    loaded = store.load_daily(handle, series_id="^GSPC")
    assert sorted(loaded.get_column("version").to_list()) == [2] * len(SESSIONS)
    assert loaded.get_column("feature_spec_sha256").to_list() == [
        store.feature_spec_sha256(bumped)
    ] * len(SESSIONS)


def test_a6_a_new_identity_is_written_even_with_append_forbidden(tmp_path: Path) -> None:
    """El camino de identidad nueva tampoco puede pasar por ``append``."""
    root = tmp_path / "store"

    def forbidden(*_: object, **__: object) -> WriteOutcome:
        raise AssertionError("identidad nueva: tampoco aqui se usa append")

    original = Store.append
    cast("Any", Store).append = forbidden
    try:
        outcome = store.save_daily(
            Store(root),
            spec=store.FeatureSpec(),
            matrix=_matrix(),
            series_id="^GSPC",
            fetched_at=FETCHED_AT,
        )
    finally:
        cast("Any", Store).append = original
    assert outcome is WriteOutcome.CREATED
    assert _versions(root) == [1] * len(SESSIONS)


# ─────────────────────────────────────────────────────────────────────────────
# A7 — el almacen posee `version` y `fetched_at`
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_the_module_neither_owns_version_nor_invents_fetched_at(tmp_path: Path) -> None:
    """Los registros no llevan ``version`` y ``fetched_at`` es el que pasa el llamante."""
    spec = store.FeatureSpec()
    matrix = _matrix()
    records = store.daily_records(matrix, spec=spec, series_id="^GSPC", fetched_at=FETCHED_AT)

    assert len(records) == len(SESSIONS)
    for record, session in zip(records, SESSIONS, strict=True):
        assert "version" not in record
        assert record["fetched_at"] is FETCHED_AT
        assert record["as_of"] == _session_close(session)
        assert record["source"] == store.FEATURES_SOURCE
        # el `as_of` de la fila es **exactamente** el que entra en el hash
        assert record["features_version"] == store.features_version(
            spec, cast("datetime", record["as_of"])
        )
    assert {record["feature_spec_sha256"] for record in records} == {
        store.feature_spec_sha256(spec)
    }

    # la capa es siempre `derived`: ninguna ruta toca `raw.*`
    root = tmp_path / "store"
    store.save_daily(
        Store(root), spec=spec, matrix=matrix, series_id="^GSPC", fetched_at=FETCHED_AT
    )
    assert store.FEATURES_LAYER == "derived"
    assert not (root / "raw").exists()
    stored = Store(root).sql(
        f"SELECT * FROM {store.FEATURES_LAYER}.{store.FEATURES_DATASET}"  # noqa: S608
    )
    assert all(
        fetched >= as_of
        for fetched, as_of in zip(
            stored.get_column("fetched_at").to_list(),
            stored.get_column("as_of").to_list(),
            strict=True,
        )
    )

    # y el modulo no lee el reloj
    source = _source()
    for forbidden in ("datetime.now", "utcnow", "date.today", "time.time", "time_ns"):
        assert forbidden not in source


def test_a7_an_as_of_discrepancy_is_a_typed_error() -> None:
    """Un `as_of` que no cuadra con la fila (o con el hash ya calculado) no se escribe."""
    spec = store.FeatureSpec()
    matrix = _matrix()
    shifted = matrix.with_columns(
        pl.Series("as_of", [_session_close(session + timedelta(days=1)) for session in SESSIONS])
    )

    mismatched_version = matrix.with_columns(
        pl.Series("features_version", [store.FEATURE_VERSION_PREFIX + "0" * 64] * len(SESSIONS))
    )
    mismatched_spec = matrix.with_columns(
        pl.Series("feature_spec_sha256", ["0" * 64] * len(SESSIONS))
    )
    naive = matrix.with_columns(pl.Series("as_of", [datetime(2025, 1, 2, 20, 0)] * len(SESSIONS)))

    cases: dict[str, pl.DataFrame] = {
        "sesion y as_of de dias distintos": shifted,
        "features_version que no cuadra": mismatched_version,
        "feature_spec_sha256 que no cuadra": mismatched_spec,
        "as_of sin zona horaria": naive,
    }
    for frame in cases.values():
        with pytest.raises(store.InvalidFeatureMatrixError):
            store.daily_records(frame, spec=spec, series_id="^GSPC", fetched_at=FETCHED_AT)

    with pytest.raises(store.InvalidFeatureMatrixError):
        store.daily_records(
            matrix, spec=spec, series_id="^GSPC", fetched_at=_session_close(SESSIONS[0])
        )


# ─────────────────────────────────────────────────────────────────────────────
# A8 — normalizacion robusta de ventana expandida
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_expanding_robust_normalisation() -> None:
    """``z_t = (x_t - mediana_{<=t}) / (1,4826 * MAD_{<=t})``, con ``min_sessions - 1`` nulos."""
    assert store.MAD_SCALE == 1.4826
    values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0]
    frame = pl.DataFrame({"session": list(SESSIONS)[: len(values)], "x": values})
    result = store.normalise_expanding(frame, "x", min_sessions=5)
    column = f"x{store.NORMALISED_SUFFIX}"

    assert set(result.columns) == {"session", "x", column}
    assert result.get_column("session").to_list() == list(SESSIONS)[: len(values)]
    assert result.get_column(column).to_list()[:4] == [None] * 4

    window = values[:5]
    centre = statistics.median(window)
    mad = statistics.median([abs(value - centre) for value in window])
    assert result.get_column(column).to_list()[4] == pytest.approx(
        (values[4] - centre) / (store.MAD_SCALE * mad)
    )

    # la ventana crece: el valor de la sesion 6 usa las seis primeras
    window = values[:6]
    centre = statistics.median(window)
    mad = statistics.median([abs(value - centre) for value in window])
    assert result.get_column(column).to_list()[5] == pytest.approx(
        (values[5] - centre) / (store.MAD_SCALE * mad)
    )

    # `min_sessions` es obligatorio y se valida
    with pytest.raises(TypeError):
        store.normalise_expanding(frame, "x")  # pyright: ignore[reportCallIssue]
    for bad in (0, -1, True, 1.5, "5"):
        with pytest.raises(store.InvalidFeatureSpecError):
            store.normalise_expanding(frame, "x", min_sessions=bad)  # pyright: ignore[reportArgumentType]
    with pytest.raises(store.InvalidFeatureMatrixError):
        store.normalise_expanding(frame, "no_existe", min_sessions=2)


def test_a8_a_null_does_not_count_as_history_and_stays_null() -> None:
    """Un hueco no inventa historia: se publica ``null`` y no avanza el contador."""
    values = [1.0, None, 2.0, 3.0, 4.0, 5.0]
    frame = pl.DataFrame({"session": _sessions(len(values)), "x": values})
    column = f"x{store.NORMALISED_SUFFIX}"
    result = store.normalise_expanding(frame, "x", min_sessions=3).get_column(column).to_list()

    # con el hueco, la tercera sesion **con dato** es la cuarta fila (indice 3)
    assert result[:3] == [None, None, None]
    assert result[3] is not None
    window = [1.0, 2.0, 3.0]
    centre = statistics.median(window)
    mad = statistics.median([abs(value - centre) for value in window])
    assert result[3] == pytest.approx((3.0 - centre) / (store.MAD_SCALE * mad))
    assert result[4] is not None


# ─────────────────────────────────────────────────────────────────────────────
# A9 — nunca sobre la muestra completa
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_prefix_values_do_not_change_when_sessions_are_appended() -> None:
    """Prefijo contra matriz completa: la parte comun sale byte a byte identica."""
    values = [10.0 + math.sin(index / 3.0) * 4.0 + (index % 7) * 0.3 for index in range(60)]
    column = f"x{store.NORMALISED_SUFFIX}"

    def zs(count: int) -> list[float | None]:
        frame = pl.DataFrame({"session": _sessions(count), "x": values[:count]})
        return store.normalise_expanding(frame, "x", min_sessions=7).get_column(column).to_list()

    full = zs(len(values))
    assert full[:10] == [None] * 6 + full[6:10]
    for count in (7, 13, 30, 60):
        assert zs(count) == full[:count]


def test_a9_the_implementation_does_not_use_the_whole_sample() -> None:
    """Ni ``mean`` ni ``std`` de la columna entera: la ventana es ``<= t``."""
    tree = ast.parse(inspect.getsource(store.normalise_expanding))
    used = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not used & {"mean", "std", "var", "quantile", "median_low"}
    assert "median" in used


# ─────────────────────────────────────────────────────────────────────────────
# A10 — borde y basura
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_mad_zero_is_null_never_inf_or_nan() -> None:
    """Sin dispersion no hay z: ``null``, nunca ``inf`` ni ``NaN``."""
    column = f"x{store.NORMALISED_SUFFIX}"
    constant = pl.DataFrame({"session": list(SESSIONS)[:10], "x": [5.0] * 10})
    result = store.normalise_expanding(constant, "x", min_sessions=3)

    assert result.get_column(column).to_list() == [None] * 10
    mixed = pl.DataFrame(
        {"session": list(SESSIONS)[:8], "x": [4.0, 4.0, 4.0, 4.0, 9.0, 9.0, 9.0, 9.0]}
    )
    for value in store.normalise_expanding(mixed, "x", min_sessions=4).get_column(column).to_list():
        assert value is None or math.isfinite(value)


def test_a10_broken_values_are_typed_errors() -> None:
    """``NaN``/``inf``, una columna fuera del catalogo y una sesion repetida."""
    spec = store.FeatureSpec()
    matrix = _matrix()

    broken_values = (float("nan"), float("inf"), float("-inf"))
    for broken in broken_values:
        for column in ("parkinson_rv", "atr_norm"):
            corrupt = matrix.with_columns(
                pl.Series(column, [broken] * matrix.height, dtype=pl.Float64)
            )
            with pytest.raises(store.InvalidFeatureMatrixError):
                store.daily_records(corrupt, spec=spec, series_id="^GSPC", fetched_at=FETCHED_AT)

    outside = matrix.with_columns(pl.Series("no_catalogada", [0.0] * matrix.height))
    with pytest.raises(store.InvalidFeatureMatrixError, match="fuera del catalogo"):
        store.daily_records(outside, spec=spec, series_id="^GSPC", fetched_at=FETCHED_AT)

    missing = matrix.drop("ret_log")
    with pytest.raises(store.InvalidFeatureMatrixError, match="faltan columnas"):
        store.daily_records(missing, spec=spec, series_id="^GSPC", fetched_at=FETCHED_AT)

    repeated = pl.concat([matrix, matrix.head(1)])
    with pytest.raises(store.InvalidFeatureMatrixError, match="repite"):
        store.daily_records(repeated, spec=spec, series_id="^GSPC", fetched_at=FETCHED_AT)

    text = matrix.with_columns(pl.Series("true_range", ["hola"] * matrix.height))
    with pytest.raises(store.InvalidFeatureMatrixError, match="no es numerica"):
        store.daily_records(text, spec=spec, series_id="^GSPC", fetched_at=FETCHED_AT)


def test_a10_not_computable_is_published_as_null_and_survives_parquet(tmp_path: Path) -> None:
    """Un ``null`` es un dato: no se convierte en 0 al escribir y leer."""
    root = tmp_path / "store"
    matrix = _matrix()
    nulls_before = {name: matrix.get_column(name).null_count() for name in store.FEATURE_COLUMNS}
    assert nulls_before["vix_zscore"] > 0  # 40 sesiones < 250: no computable, y se dice

    store.save_daily(
        Store(root),
        spec=store.FeatureSpec(),
        matrix=matrix,
        series_id="^GSPC",
        fetched_at=FETCHED_AT,
    )
    loaded = store.load_daily(Store(root), series_id="^GSPC")
    assert loaded.height == len(SESSIONS)
    assert {name: loaded.get_column(name).null_count() for name in store.FEATURE_COLUMNS} == (
        nulls_before
    )
    expected = store.matrix_sha256(matrix.drop("session").sort("as_of"))
    round_tripped = loaded.select("as_of", *store.FEATURE_COLUMNS)
    assert store.matrix_sha256(round_tripped) == expected


# ─────────────────────────────────────────────────────────────────────────────
# A11 — catalogo y politica documentada
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_catalog_is_complete_and_documented() -> None:
    """Cada feature publica nombre, formula, ventana, fuente y ``as_of`` requerido."""
    names = [entry.name for entry in store.FEATURE_CATALOG]
    assert len(names) == len(set(names))
    for entry in store.FEATURE_CATALOG:
        assert entry.name and entry.formula and entry.source and entry.required_as_of
        assert entry.window is None or entry.window >= 1

    computed = add_features(_inputs(SESSIONS))
    assert set(store.FEATURE_COLUMNS) <= set(computed.columns)
    assert set(store.FEATURE_COLUMNS) == set(names)
    for entry in store.FEATURE_CATALOG:
        assert entry.name in computed.columns


def test_a11_the_catalog_window_of_the_vix_is_the_one_that_is_used() -> None:
    """Con 249 sesiones el z del VIX no es computable; con 300, si."""
    short = store.build_matrix(_inputs(_sessions(249)), spec=store.FeatureSpec())
    long = store.build_matrix(_inputs(LONG_SESSIONS), spec=store.FeatureSpec())

    assert short.get_column("vix_zscore").null_count() == 249
    assert long.get_column("vix_zscore").null_count() < 300
    assert long.get_column("vix_level").null_count() == 1


def test_a11_a_window_that_contradicts_the_catalog_is_an_error() -> None:
    """El catalogo y la spec no pueden decir cosas distintas."""
    base = store.FeatureSpec()
    with pytest.raises(store.InvalidFeatureSpecError, match="no coincide"):
        store.FeatureSpec(windows={**base.windows, "atr_norm": 21})
    with pytest.raises(store.InvalidFeatureSpecError, match="no esta en el catalogo"):
        store.FeatureSpec(windows={**base.windows, "no_existe": 3})
    with pytest.raises(store.InvalidFeatureSpecError, match="no puede estar vacio"):
        store.FeatureSpec(windows={})
    with pytest.raises(store.InvalidFeatureSpecError, match="code_version"):
        store.FeatureSpec(code_version=0)
    with pytest.raises(store.InvalidFeatureSpecError, match="feature_set"):
        store.FeatureSpec(feature_set="   ")
    with pytest.raises(store.InvalidFeatureSpecError, match="JSON canonico"):
        store.feature_spec_sha256(store.FeatureSpec(parameters={"peso": object()}))
    with pytest.raises(store.InvalidFeatureSpecError, match="datetime UTC"):
        store.features_version(base, date(2025, 1, 2))  # pyright: ignore[reportArgumentType]


def test_a11_the_module_docstring_carries_the_versioning_policy() -> None:
    """La politica tiene que estar donde se lee: en el docstring del modulo."""
    docstring = store.__doc__ or ""
    for phrase in (
        "Politica de versionado",
        "FEATURE_CODE_VERSION",
        "invalida los backtests",
        "golden",
        "JSON",
    ):
        assert phrase in docstring


# ─────────────────────────────────────────────────────────────────────────────
# A12 — determinismo
# ─────────────────────────────────────────────────────────────────────────────
_DIGEST_SCRIPT = """
import hashlib, json, pathlib, sys
from datetime import UTC, datetime

import polars as pl

from cfdtrader.features import store

root, csv_path = sys.argv[1], sys.argv[2]
spec = store.FeatureSpec()
frame = pl.read_csv(csv_path, try_parse_dates=True)
matrix = store.build_matrix(frame, spec=spec)
moments = (
    frame.get_column("session").cast(pl.Datetime("us")) + pl.duration(hours=20)
).dt.replace_time_zone("UTC")
matrix = matrix.with_columns(moments.alias("as_of"))
handle = store.Store(root)
store.save_daily(
    handle,
    spec=spec,
    matrix=matrix,
    series_id="^GSPC",
    fetched_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
)
rows = store.load_daily(handle, series_id="^GSPC")
digests = sorted(
    hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(pathlib.Path(root).rglob("*.parquet"))
)
print(json.dumps({
    "feature_spec_sha256": store.feature_spec_sha256(spec),
    "features_version": rows.get_column("features_version")[0],
    "matrix_sha256": store.matrix_sha256(matrix.drop("as_of")),
    "parquet_sha256": hashlib.sha256("\\n".join(digests).encode("utf-8")).hexdigest(),
}))
"""


def _subprocess_digests(seed: str, root: Path) -> dict[str, Any]:
    """Ejecuta el guardado completo en un proceso nuevo con ese ``PYTHONHASHSEED``."""
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = seed
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(  # noqa: S603 - el ejecutable es el interprete de la sesion
        [
            sys.executable,
            "-c",
            _DIGEST_SCRIPT,
            str(root),
            str(FIXTURES / "golden_inputs.csv"),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return dict(json.loads(result.stdout))


def test_a12_determinism_across_processes_and_second_pass(tmp_path: Path) -> None:
    """Mismo input y mismo ``as_of`` ⇒ mismos hashes y Parquet byte a byte."""
    digests = [
        _subprocess_digests(seed, tmp_path / f"seed_{seed}") for seed in ("0", "1", "random")
    ]
    assert digests[0] == digests[1] == digests[2]

    spec, base = _golden()
    sessions = _golden_sessions()
    matrix = _with_as_of(base, sessions)
    assert digests[0]["feature_spec_sha256"] == store.feature_spec_sha256(spec)
    assert _prefixed(digests[0]["feature_spec_sha256"]) == _expected()["feature_spec_sha256"]
    assert digests[0]["matrix_sha256"] == store.matrix_sha256(base)
    assert _prefixed(digests[0]["matrix_sha256"]) == _expected()["matrix_sha256"]

    # segunda pasada en el mismo proceso, y a una raiz distinta
    first, second = tmp_path / "pass_1", tmp_path / "pass_2"
    for root in (first, second):
        store.save_daily(
            Store(root), spec=spec, matrix=matrix, series_id="^GSPC", fetched_at=FETCHED_AT
        )
    assert _parquet_digest(first) == _parquet_digest(second) == digests[0]["parquet_sha256"]

    again = store.save_daily(
        Store(first), spec=spec, matrix=matrix, series_id="^GSPC", fetched_at=FETCHED_AT
    )
    assert again is WriteOutcome.UNCHANGED
    assert _parquet_digest(first) == digests[0]["parquet_sha256"]

    loaded = store.load_daily(Store(first), series_id="^GSPC")
    assert loaded.get_column("features_version")[0] == digests[0]["features_version"]


# ─────────────────────────────────────────────────────────────────────────────
# A13 — puertas
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_the_module_does_not_cheat_the_gates() -> None:
    """A13 se mide con el comando documentado; lo que si se comprueba es como se mide.

    La cobertura (>= 90 % sentencias / 85 % ramas del modulo nuevo) y la suite
    completa no las puede afirmar un test: las afirma la puerta. Lo que **si** se
    puede comprobar aqui es que la cobertura no se ha maquillado.
    """
    source = _source()
    assert "pragma: no cover" not in source
    assert "type: ignore" not in source
    assert "coverage" not in source.lower()


def test_a13_load_daily_without_data_is_a_typed_error(tmp_path: Path) -> None:
    """Leer una matriz que no existe no puede devolver un frame vacio en silencio."""
    with pytest.raises(UnknownDatasetError):
        store.load_daily(Store(tmp_path / "store"), series_id="^GSPC")
    for bad in ("", "  ", "serie'; DROP TABLE x", "a/b"):
        with pytest.raises(store.InvalidSeriesIdError):
            store.load_daily(Store(tmp_path / "store"), series_id=bad)
        with pytest.raises(store.InvalidSeriesIdError):
            store.daily_records(
                _matrix(), spec=store.FeatureSpec(), series_id=bad, fetched_at=FETCHED_AT
            )


def test_a13_replace_never_touches_raw(tmp_path: Path) -> None:
    """La capa es una decision del modulo, no del llamante: `raw` es inmutable."""
    with pytest.raises(ImmutableWriteError):
        Store(tmp_path / "store").replace("raw", store.FEATURES_DATASET, [])


# ─────────────────────────────────────────────────────────────────────────────
# Apoyo: las formas degeneradas de la matriz y de la spec
# ─────────────────────────────────────────────────────────────────────────────
def test_support_degenerate_inputs_are_typed_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cada rechazo posible tiene su error tipado, no un `TypeError` de polars."""
    spec = store.FeatureSpec()
    matrix = _matrix()

    with pytest.raises(store.InvalidFeatureSpecError, match="code_version"):
        store.FeatureSpec(code_version="1")  # pyright: ignore[reportArgumentType]
    with pytest.raises(store.InvalidFeatureSpecError, match="zona horaria"):
        store.features_version(spec, datetime(2025, 1, 2, 20, 0))

    missing_inputs = _inputs(SESSIONS).drop("high", "vix_close")
    with pytest.raises(store.InvalidFeatureMatrixError, match="faltan columnas de entrada"):
        store.build_matrix(missing_inputs, spec=spec)

    monkeypatch.setattr(store, "add_features", lambda frame: frame)
    with pytest.raises(store.InvalidFeatureMatrixError, match="no produjo estas features"):
        store.build_matrix(_inputs(SESSIONS), spec=spec)
    monkeypatch.undo()

    as_datetime = matrix.with_columns(pl.col("session").cast(pl.Datetime("us")))
    assert store.daily_records(as_datetime, spec=spec, series_id="^GSPC", fetched_at=FETCHED_AT)
    as_text = matrix.with_columns(pl.col("session").cast(pl.String()))
    with pytest.raises(store.InvalidFeatureMatrixError, match="'session' debe ser"):
        store.daily_records(as_text, spec=spec, series_id="^GSPC", fetched_at=FETCHED_AT)
    as_date = matrix.with_columns(pl.Series("as_of", list(SESSIONS)))
    with pytest.raises(store.InvalidFeatureMatrixError, match="'as_of' debe ser"):
        store.daily_records(as_date, spec=spec, series_id="^GSPC", fetched_at=FETCHED_AT)

    # el digest canonico admite matrices parciales y frames sin columnas de identidad
    partial = matrix.drop("ret_sq", "session")
    assert store.matrix_sha256(partial) == store.matrix_sha256(partial)
    assert store.matrix_sha256(pl.DataFrame({"suelta": [1.0, 2.0]})) != store.matrix_sha256(
        pl.DataFrame({"suelta": [2.0, 1.0]})
    )
