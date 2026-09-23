"""Suite de integridad de la matriz de las cinco familias (#17): A1-A16.

Blinda los cuatro errores que invalidarian el proyecto **en silencio**:

1. *no-look-ahead* **cross-familia**: un feature en `t` no puede cambiar al anadir datos de
   `t+1`, y la ventana de la normalizacion robusta es **expansiva** (A2-A5).
2. *golden dataset* de la matriz de las 52 columnas, congelado con hash, y una **puerta por
   `FEATURE_CODE_VERSION`** (A6-A8).
3. determinismo **byte a byte** del gate y de la matriz, tambien entre procesos con
   `PYTHONHASHSEED` distinto (A9-A10).
4. los tres escenarios de coste calculados **a mano** y los tres estados del *slippage*, que no
   se fusionan (A11-A13).

Todo se mide sobre un almacen **sintetico** en `tmp_path`, ensamblado con `Store.append` a
partir de los cinco fixtures comprometidos; el modulo no toca el almacen real del repositorio,
ni el registro de experimentos, ni el reloj, ni la red (A14). La puerta del golden es **local**
(no hay CI todavia, #85): un hook `pre-push` de pre-commit (A15).

La identidad de la matriz **no se reimplementa**: se recomputa con
``analysis.feature_frame.build_feature_matrix`` y se lee con ``store.sql()``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Final, cast

import polars as pl
import pytest
import yaml

from cfdtrader.analysis.feature_frame import FAMILY_ORDER, FeatureMatrix, build_feature_matrix
from cfdtrader.backtest.costs import (
    RATIO_QUANTUM,
    CostBreakdown,
    Side,
    SlippageParameter,
    cost_breakdown,
    declared_cost_model,
    declared_slippage_assumption,
)
from cfdtrader.backtest.engine import canonical_text
from cfdtrader.data.calendar import MarketCalendar
from cfdtrader.data.store import Store
from cfdtrader.decision import gate
from cfdtrader.decision.gate import GateOutput, GateParameters, evaluate_gate
from cfdtrader.features import store as feature_store
from cfdtrader.features.context import context_spec
from cfdtrader.features.macro import macro_spec
from cfdtrader.features.regime import regime_spec
from cfdtrader.features.technical import technical_spec

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

#: Los cinco fixtures comprometidos; el almacen sintetico se ensambla **solo** con ellos.
FIXTURES: Final[Path] = REPO_ROOT / "tests" / "fixtures" / "features"

#: El golden de la matriz completa (A6-A8).
GOLDEN: Final[Path] = FIXTURES / "matrix_golden_expected.json"

#: La puerta local de A15.
PRE_COMMIT: Final[Path] = REPO_ROOT / ".pre-commit-config.yaml"

#: Identificador del hook local de la etapa `pre-push` (A15).
PRE_PUSH_HOOK: Final[str] = "integrity-golden"

#: El ancla del estudio, su proxy de volatilidad y el indice dolar.
ANCHOR: Final[str] = "^GSPC"
VIX: Final[str] = "^VIX"
DXY: Final[str] = "DX-Y.NYB"

#: Series de mercado de la familia de contexto que **si** se toman del fixture de contexto: las
#: otras tres columnas de ese CSV (`^GSPC`, `^VIX` y `DX-Y.NYB`) chocan con series que ya tienen
#: otro origen declarado, y la identidad del almacen es `(source, series_id, as_of)`.
CONTEXT_MARKET_FIXTURE: Final[tuple[str, ...]] = ("^GDAXI", "^FTSE", "^STOXX50E", "^N225", "^HSI")

#: Instantes declarados: **nunca** del reloj. El cierre de sesion va a las 21:00 UTC, que en ET
#: es la misma fecha todo el anio (16:00 EST en invierno, 17:00 EDT en verano).
FETCHED_AT: Final[datetime] = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
CLOSE_HOUR_UTC: Final[int] = 21
DT_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%SZ"

#: Procedencias declaradas de cada dataset del almacen sintetico.
SOURCE_MARKET: Final[str] = "yfinance"
SOURCE_SECTORS: Final[str] = "stooq"
SOURCE_MACRO: Final[str] = "fred"
SOURCE_LABELS: Final[str] = "cfdtrader.models.labels"

#: El catalogo tiene 52 columnas y la matriz publica `session` + esas 52.
N_COLUMNS: Final[int] = 52

#: Suelo de A3: por debajo de esto el fixture seria degenerado.
MIN_SESSIONS: Final[int] = 250

#: Sesiones que tienen que quedar **despues** de la frontera para que `sessions_to_opex` del
#: primer dia posterior tenga un vencimiento por delante (el control de A3).
MIN_SESSIONS_AFTER_FRONTIER: Final[int] = 25

#: Modulos de red, nombres de reloj y metodos de escritura que el modulo **no** puede usar (A14).
NETWORK_MODULES: Final[tuple[str, ...]] = ("http", "httpx", "requests", "socket", "urllib")
CLOCK_NAMES: Final[tuple[str, ...]] = ("now", "utcnow", "today")
WRITE_METHODS: Final[tuple[str, ...]] = ("write_text", "write_bytes", "unlink", "mkdir", "touch")

#: Cadenas prohibidas en el codigo del modulo (fuera de los docstrings): el almacen real del
#: repositorio y la lectura *point-in-time*, que no se usan aqui.
FORBIDDEN_TEXT: Final[tuple[str, ...]] = ("data/", "read_pit")

#: El caso canonico del gate (A9-A10): sesion, instante y calendario declarados a mano.
GATE_SESSION: Final[date] = date(2026, 9, 23)
GATE_AS_OF: Final[datetime] = datetime(2026, 9, 23, 12, 45, tzinfo=UTC)
GATE_CALENDAR: Final[MarketCalendar] = MarketCalendar(years=tuple(range(2020, 2030)))

#: El *slippage* medido de las sondas: `Decimal("0.2")` en la escala de #11 (no `0.02`).
GATE_SLIPPAGE_PCT: Final[Decimal] = Decimal("0.2")

#: Claves del golden (A6), en el orden en que se publican. `n_columns` es la unica anadida a la
#: convencion de #19-#23: deja el `n_columns == 52` de A7 escrito en el propio fixture.
GOLDEN_KEYS: Final[tuple[str, ...]] = (
    "_comment",
    "generator",
    "code_version",
    "feature_spec_sha256",
    "matrix_sha256",
    "feature_columns",
    "sessions",
    "n_columns",
    "first_session",
    "last_session",
    "non_null",
    "sources",
    "duplicated_columns",
)

#: Criterio → trozo de nombre que tiene que seleccionarlo con `-k` (A16).
SELECTORS: Final[dict[str, tuple[str, ...]]] = {
    "A2": ("no_look_ahead",),
    "A3": ("no_look_ahead",),
    "A4": ("negative_control",),
    "A5": ("expanding_window",),
    "A6": ("golden_fixture",),
    "A7": ("golden_matrix",),
    "A8": ("version_gate",),
    "A9": ("gate_determinism",),
    "A10": ("across_processes",),
    "A11": ("costs",),
    "A12": ("costs",),
    "A13": ("costs",),
    "A14": ("no_clock_no_io",),
    "A15": ("pre_push",),
}


# ─────────────────────────────────────────────────────────────────────────────
# El almacen sintetico (A1)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Substrate:
    """El almacen sintetico y su matriz; la raiz vive siempre bajo `tmp_path`."""

    root: Path
    matrix: FeatureMatrix


def _instant(session: date) -> datetime:
    """Cierre de sesion declarado de esa fecha, en UTC (no se consulta el reloj)."""
    return datetime(session.year, session.month, session.day, CLOSE_HOUR_UTC, tzinfo=UTC)


def _bar(
    *,
    series_id: str,
    session: date,
    source: str,
    prices: tuple[float, float, float, float],
) -> dict[str, object]:
    """Una barra diaria sintetica, con las columnas que lee `raw.market_daily`."""
    opened, high, low, close = prices
    return {
        "source": source,
        "series_id": series_id,
        "as_of": _instant(session),
        "fetched_at": FETCHED_AT,
        "published_at": None,
        "open": opened,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1_000.0,
    }


def _flat(close: float) -> tuple[float, float, float, float]:
    """OHLC de una serie de la que el fixture solo trae el cierre: se declara, no se inventa.

    `context_v1` y `macro_v1` consumen el **cierre** de esas series; las otras tres columnas se
    rellenan con el mismo valor para que el almacen tenga el esquema que lee el adaptador.
    """
    return (close, close, close, close)


def _anchor() -> pl.DataFrame:
    """OHLC del ancla: el fixture de volatilidad manda y el de regimen rellena las 500 previas.

    `golden_inputs.csv` (300 sesiones desde 2025-01-02) es el ancla de #19; el GARCH de
    `regime_v1` necesita **500** sesiones de entrenamiento, asi que las anteriores las aporta
    `regime_golden_inputs.csv` (800 sesiones reales desde 2023-07-11). Las dos series son el mismo
    `^GSPC` y no se pueden escribir por separado (la identidad del almacen es
    `(source, series_id, as_of)`): la ventana del ancla es la union y, donde las dos traen precio,
    manda `golden_inputs.csv`.
    """
    columns = ("session", "open", "high", "low", "close")
    primary = pl.read_csv(FIXTURES / "golden_inputs.csv", try_parse_dates=True).select(*columns)
    filler = (
        pl.read_csv(FIXTURES / "regime_golden_inputs.csv", try_parse_dates=True)
        .select(*columns)
        .join(primary.select("session"), on="session", how="anti")
    )
    return pl.concat([filler, primary]).sort("session")


def _volatility_inputs() -> pl.DataFrame:
    """`golden_inputs.csv`: el ancla de #19 y su columna `vix_close`."""
    return pl.read_csv(FIXTURES / "golden_inputs.csv", try_parse_dates=True)


def _regime_inputs() -> pl.DataFrame:
    """`regime_golden_inputs.csv`: las 800 sesiones reales con precios sinteticos."""
    return pl.read_csv(FIXTURES / "regime_golden_inputs.csv", try_parse_dates=True)


def _context_inputs() -> pl.DataFrame:
    """`context_golden_inputs.csv`: una fila por sesion del calendario union de contexto.

    ``infer_schema_length=None`` es obligatorio: ``XLC`` nace en la sesion 180 y con la ventana de
    inferencia por defecto se leeria como texto.
    """
    return pl.read_csv(
        FIXTURES / "context_golden_inputs.csv", try_parse_dates=True, infer_schema_length=None
    )


def _macro_market() -> pl.DataFrame:
    """`macro_golden_market.csv`: las barras del indice dolar, con su `as_of` declarado."""
    table = pl.read_csv(
        FIXTURES / "macro_golden_market.csv",
        schema_overrides={"session": pl.String, "dxy_as_of": pl.String, "dxy_close": pl.String},
    )
    return table.with_columns(
        pl.col("session").str.to_date(),
        pl.col("dxy_as_of").str.to_datetime(format=DT_FORMAT, time_zone="UTC"),
        pl.col("dxy_close").cast(pl.Float64),
    )


def _macro_series() -> pl.DataFrame:
    """`macro_golden_series.csv`: las seis series de `raw.macro`, en formato largo."""
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


def _visible_at(record: Mapping[str, object]) -> datetime:
    """Instante en el que el almacen conoce un registro: `published_at`, o `as_of` si es barra."""
    published = record["published_at"]
    return cast("datetime", record["as_of"]) if published is None else cast("datetime", published)


def _records() -> dict[str, list[dict[str, object]]]:
    """Todos los registros del almacen sintetico, por dataset y **sin** recortar."""
    anchor = _anchor()
    market: list[dict[str, object]] = [
        _bar(
            series_id=ANCHOR,
            session=cast("date", row["session"]),
            source=SOURCE_MARKET,
            prices=(
                float(cast("float", row["open"])),
                float(cast("float", row["high"])),
                float(cast("float", row["low"])),
                float(cast("float", row["close"])),
            ),
        )
        for row in anchor.iter_rows(named=True)
    ]
    for row in _volatility_inputs().iter_rows(named=True):
        market.append(
            _bar(
                series_id=VIX,
                session=cast("date", row["session"]),
                source=SOURCE_MARKET,
                prices=_flat(float(cast("float", row["vix_close"]))),
            )
        )
    context = _context_inputs()
    for name in CONTEXT_MARKET_FIXTURE:
        for row in context.select("session", name).drop_nulls(name).iter_rows(named=True):
            market.append(
                _bar(
                    series_id=name,
                    session=cast("date", row["session"]),
                    source=SOURCE_MARKET,
                    prices=_flat(float(cast("float", row[name]))),
                )
            )
    for row in _macro_market().drop_nulls("dxy_as_of").iter_rows(named=True):
        close = float(cast("float", row["dxy_close"]))
        market.append(
            {
                "source": SOURCE_MARKET,
                "series_id": DXY,
                "as_of": row["dxy_as_of"],
                "fetched_at": FETCHED_AT,
                "published_at": None,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1_000.0,
            }
        )

    sectors: list[dict[str, object]] = [
        _bar(
            series_id=name,
            session=cast("date", row["session"]),
            source=SOURCE_SECTORS,
            prices=_flat(float(cast("float", row[name]))),
        )
        for name in feature_store.CONTEXT_SECTOR_SERIES
        for row in context.select("session", name).drop_nulls(name).iter_rows(named=True)
    ]

    macro: list[dict[str, object]] = [
        {
            "source": SOURCE_MACRO,
            "series_id": str(row["series_id"]),
            "as_of": row["as_of"],
            "fetched_at": FETCHED_AT,
            "published_at": row["published_at"],
            "value": float(cast("float", row["value"])),
        }
        for row in _macro_series().iter_rows(named=True)
    ]

    # `derived.labels` sintetico **declarado**: no entra en la matriz, pero el adaptador exige
    # que el dataset exista, y una fila por sesion del ancla deja el almacen coherente.
    labels: list[dict[str, object]] = [
        {
            "source": SOURCE_LABELS,
            "series_id": ANCHOR,
            "as_of": _instant(cast("date", row["session"])),
            "fetched_at": FETCHED_AT,
            "published_at": None,
            "session": row["session"],
            "ret_long": 0.001 * float(position % 7) - 0.003,
            "is_half_day": False,
            "k_sigma": 1.0,
        }
        for position, row in enumerate(anchor.iter_rows(named=True))
    ]
    return {"market_daily": market, "sectors": sectors, "macro": macro, "labels": labels}


def build_store(root: Path, *, visible_at: datetime | None = None) -> Store:
    """Materializa el almacen sintetico en ``root`` y lo devuelve.

    Con ``visible_at`` se escribe solo lo que el almacen **ya conocia** en ese instante: eso es lo
    que convierte el almacen en un prefijo temporal del completo y lo que usa A2 para anadir
    despues las sesiones posteriores. La capa es un argumento aparte, como en el contrato.
    """
    handle = Store(root)
    for dataset, dataset_records in _records().items():
        selected = (
            dataset_records
            if visible_at is None
            else [record for record in dataset_records if _visible_at(record) <= visible_at]
        )
        handle.append("derived" if dataset == "labels" else "raw", dataset, selected)
    return handle


def substrate(root: Path, *, visible_at: datetime | None = None) -> Substrate:
    """El almacen y su matriz de las cinco familias, sin escribir nada fuera de ``root``."""
    return Substrate(
        root=root, matrix=build_feature_matrix(build_store(root, visible_at=visible_at))
    )


def _specs() -> dict[str, feature_store.FeatureSpec]:
    """Las cinco specs por defecto de las familias, en el orden de ensamblado."""
    return {
        feature_store.VOLATILITY_FEATURE_SET: feature_store.FeatureSpec(),
        feature_store.TECHNICAL_FEATURE_SET: technical_spec(),
        feature_store.CONTEXT_FEATURE_SET: context_spec(),
        feature_store.MACRO_FEATURE_SET: macro_spec(),
        feature_store.REGIME_FEATURE_SET: regime_spec(),
    }


def _catalog_sources() -> list[list[str]]:
    """Las fuentes de entrada de las cinco familias, sin repetir y en orden de ensamblado."""
    seen: dict[tuple[str, str], None] = {}
    for family in FAMILY_ORDER:
        for pair in _specs()[family].sources:
            seen.setdefault(pair, None)
    return [[dataset, series] for dataset, series in seen]


# ─────────────────────────────────────────────────────────────────────────────
# El golden completo (A6-A8)
# ─────────────────────────────────────────────────────────────────────────────
GOLDEN_COMMENT: Final[tuple[str, ...]] = (
    "Golden dataset congelado de la matriz de las cinco familias (#17). Los INPUTS son los cinco",
    "fixtures comprometidos de tests/fixtures/features: el ancla `^GSPC` es la union de",
    "`golden_inputs.csv` (300 sesiones; manda donde las dos traen precio) y",
    "`regime_golden_inputs.csv` (las 500+ anteriores que necesita el GARCH de `regime_v1`);",
    "`vix_close` sale de `golden_inputs.csv`, las series de contexto y los 11 ETF sectoriales de",
    "`context_golden_inputs.csv`, el indice dolar de `macro_golden_market.csv` y las seis series",
    "macro de `macro_golden_series.csv`. No hay ninguna semilla y ningun reloj: el almacen es una",
    "funcion determinista de esos ficheros.",
    "El par (code_version, digests) esta congelado entero, asi que el test falla si el calculo se",
    "mueve sin subir FEATURE_CODE_VERSION (y tambien si sube sin actualizar este fichero). Nunca",
    "se regenera un golden 'para que pase'.",
)

GOLDEN_GENERATOR: Final[str] = (
    "tests/fixtures/features/*.csv ensamblados con Store.append y "
    "cfdtrader.analysis.feature_frame.build_feature_matrix; para reproducirlo: "
    "json.dumps(tests.test_integrity.golden_payload(root), indent=2, ensure_ascii=False) sobre un"
    " root nuevo dentro del workspace"
)


def golden_payload(root: Path) -> dict[str, object]:
    """El contenido **completo** del golden, recomputado desde el almacen sintetico.

    Es la unica definicion del fichero: el test de A7 y el comando de reproduccion trabajan sobre
    esto, no sobre una copia escrita a mano.
    """
    matrix = substrate(root).matrix
    frame = matrix.frame
    return {
        "_comment": list(GOLDEN_COMMENT),
        "generator": GOLDEN_GENERATOR,
        "code_version": matrix.feature_code_version,
        "feature_spec_sha256": {
            family: feature_store.FEATURE_VERSION_PREFIX + matrix.feature_spec_sha256[family]
            for family in FAMILY_ORDER
        },
        "matrix_sha256": feature_store.FEATURE_VERSION_PREFIX + matrix.matrix_sha256,
        "feature_columns": ["session", *feature_store.ALL_FEATURE_COLUMNS],
        "sessions": matrix.n_sessions,
        "n_columns": matrix.n_columns,
        "first_session": matrix.first_session.isoformat(),
        "last_session": matrix.last_session.isoformat(),
        "non_null": {
            name: frame.height - frame.get_column(name).null_count()
            for name in feature_store.ALL_FEATURE_COLUMNS
        },
        "sources": _catalog_sources(),
        "duplicated_columns": list(matrix.duplicated_columns),
    }


def _read_golden() -> dict[str, Any]:
    """El golden tal y como esta en disco, sin normalizar nada."""
    return cast("dict[str, Any]", json.loads(GOLDEN.read_text(encoding="utf-8")))


def _version_gate(*, digest: str, golden: Mapping[str, Any], code_version: int) -> None:
    """A8: la puerta del par (``FEATURE_CODE_VERSION``, ``matrix_sha256``).

    Tres desenlaces y ninguno se calla:

    * el digest recomputado **no** coincide y la constante no subio ⇒ el calculo se movio sin
      declararlo: hay que subir ``FEATURE_CODE_VERSION`` y reevaluar;
    * la constante **subio** (el digest haya cambiado o no) ⇒ el fixture esta obsoleto: hay que
      re-congelarlo con el digest nuevo;
    * el digest y la constante coinciden con el fixture ⇒ la puerta pasa.
    """
    frozen = str(golden["matrix_sha256"]).removeprefix(feature_store.FEATURE_VERSION_PREFIX)
    declared = int(cast("int", golden["code_version"]))
    if digest != frozen:
        if code_version == declared:
            raise AssertionError(
                f"la matriz cambio (matrix_sha256 {digest} != {frozen}) y FEATURE_CODE_VERSION "
                f"sigue en {code_version}: es un cambio de calculo **sin declarar**. Sube la "
                "constante y reevalua; no regeneres el golden para que pase"
            )
        raise AssertionError(
            f"la matriz cambio (matrix_sha256 {digest} != {frozen}) y FEATURE_CODE_VERSION subio "
            f"({declared} -> {code_version}): re-congela el fixture con el digest nuevo"
        )
    if code_version != declared:
        raise AssertionError(
            f"el fixture declara code_version {declared} y la constante vale {code_version}: "
            "re-congela el fixture con el par (code_version, digests) nuevo"
        )


# ─────────────────────────────────────────────────────────────────────────────
# A2-A5: el harness de *no-look-ahead*
# ─────────────────────────────────────────────────────────────────────────────
def _cell_key(value: object) -> object:
    """Clave de comparacion **byte a byte**: el valor, y los nulos con un centinela propio."""
    if value is None:
        return ("null",)
    if isinstance(value, date):
        return ("date", value.isoformat())
    return ("number", float(cast("float", value)).hex())


def _assert_rows_unchanged(before: pl.DataFrame, after: pl.DataFrame, *, upto: date) -> None:
    """A2/A4: el **unico** punto de comparacion de las dos matrices.

    Comprueba el conjunto **exacto** de columnas (``session`` + las 52 del catalogo) y publica el
    **primer** desajuste como ``(session, columna, antes, despues)``. Nada de comparar solo las
    columnas presentes: un frame al que le falta una columna del catalogo tiene que fallar.
    """
    expected = ("session", *feature_store.ALL_FEATURE_COLUMNS)
    for label, frame in (("antes", before), ("despues", after)):
        missing = sorted(set(expected) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(expected))
        assert not (missing or extra), (
            f"el frame de '{label}' no publica el catalogo exacto: faltan {missing}, sobran {extra}"
        )
        assert tuple(frame.columns) == expected, (
            f"el frame de '{label}' publica el catalogo en otro orden"
        )

    left = before.filter(pl.col("session") <= upto).sort("session")
    right = after.filter(pl.col("session") <= upto).sort("session")
    sessions = cast("list[date]", left.get_column("session").to_list())
    assert sessions == right.get_column("session").to_list(), (
        "las dos matrices no tienen las mismas sesiones hasta la frontera: la comparacion seria "
        "entre rejillas distintas"
    )
    for column in left.columns:
        for session, one, other in zip(
            sessions,
            left.get_column(column).to_list(),
            right.get_column(column).to_list(),
            strict=True,
        ):
            if _cell_key(one) != _cell_key(other):
                detail = (session.isoformat(), column, one, other)
                raise AssertionError(
                    "look-ahead: una sesion <= la frontera cambio al anadir sesiones posteriores. "
                    f"(session, columna, antes, despues) = {detail!r}"
                )


def _frontier(matrix: FeatureMatrix) -> date:
    """La frontera `t0` de A2/A3, con la regla declarada: la **ultima** sesion OPEX con margen.

    `sessions_to_opex` es la unica columna que mira el final del frame (declara `null` el
    vencimiento que cae fuera), asi que `t0` tiene que ser un vencimiento: asi toda sesion `<= t0`
    tiene su proximo OPEX dentro del frame y el prefijo es comparable.
    """
    frame = matrix.frame.with_row_index("position")
    margin = frame.height - 1 - MIN_SESSIONS_AFTER_FRONTIER
    candidates = frame.filter((pl.col("sessions_to_opex") == 0.0) & (pl.col("position") <= margin))
    assert candidates.height > 0, (
        "el fixture no tiene ninguna sesion OPEX con margen suficiente para el control de A3"
    )
    return cast("date", candidates.get_column("session")[-1])


@dataclass(frozen=True, slots=True)
class Frontier:
    """El par congelado de A2/A3: el prefijo temporal, el completo y la frontera."""

    before: pl.DataFrame
    after: pl.DataFrame
    t0: date

    @property
    def next_session(self) -> date:
        """La sesion inmediatamente posterior a la frontera, presente en los dos frames."""
        sessions = cast("list[date]", self.after.get_column("session").to_list())
        return sessions[sessions.index(self.t0) + 1]


@pytest.fixture(scope="module")
def frontier(tmp_path_factory: pytest.TempPathFactory) -> Frontier:
    """El par de A2/A3, construido **una vez** por sesion: dos almacenes y dos matrices."""
    root = tmp_path_factory.mktemp("integridad")
    complete = substrate(root / "completo").matrix
    t0 = _frontier(complete)
    sessions = cast("list[date]", complete.frame.get_column("session").to_list())
    prefix = substrate(root / "prefijo", visible_at=_instant(sessions[sessions.index(t0) + 1]))
    assert prefix.matrix.n_sessions < complete.n_sessions, (
        "el prefijo temporal no es mas corto que el almacen completo: A2 no probaria nada"
    )
    return Frontier(before=prefix.matrix.frame, after=complete.frame, t0=t0)


# ─────────────────────────────────────────────────────────────────────────────
# A9-A10: el determinismo del gate y de la matriz
# ─────────────────────────────────────────────────────────────────────────────
def _gate_parameters() -> GateParameters:
    """Parametros **declarados por el llamante** (las decisiones de #60 no las toma este test)."""
    return GateParameters(
        broker="cfd-broker-declarado",
        risk_per_trade_pct=Decimal("1"),
        ev_threshold_pct=Decimal("0.0084"),
        max_daily_loss_pct=Decimal("2"),
        max_weekly_loss_pct=Decimal("5"),
        max_monthly_loss_pct=Decimal("10"),
        r_pct=Decimal("1"),
        tier_a_cost_multiple=Decimal("3"),
        tier_b_cost_multiple=Decimal("2"),
        tier_a_min_probability=Decimal("0.58"),
        authorized_tiers=("A",),
    )


def _gate_cost() -> CostBreakdown:
    """El coste del caso canonico, calculado por el motor de costes de #11."""
    return cost_breakdown(
        model=declared_cost_model(),
        slippage=SlippageParameter.measured(
            pct_of_notional=GATE_SLIPPAGE_PCT,
            source="tests/test_integrity.py (#17)",
            reason="slippage medido declarado a mano para el caso canonico de determinismo",
        ),
        notional_usd=Decimal("10000"),
        side=Side.LONG,
        nights=0,
    )


def gate_case() -> GateOutput:
    """Los **19** argumentos keyword-only de `evaluate_gate`, todos declarados.

    Ni reloj (la sesion, su instante y su fecha de hoy llegan escritos), ni azar, ni red: la misma
    entrada tiene que dar la misma salida byte a byte.
    """
    return evaluate_gate(
        session=GATE_SESSION,
        as_of=GATE_AS_OF,
        today=GATE_SESSION,
        calendar=GATE_CALENDAR,
        prob_up_calibrated=0.6,
        expected_move_pct=Decimal("1.0"),
        expected_move_basis="sigma_k: k declarado por el llamante (#60)",
        cost=_gate_cost(),
        capital_usd=Decimal("10000"),
        snapshot_ok=True,
        stop_pct=Decimal("0.5"),
        target_pct=Decimal("1.0"),
        fomc_dates=(),
        params=_gate_parameters(),
    )


def process_digests(root: Path) -> dict[str, str]:
    """Los **dos** digests que A10 exige de un proceso: el del gate y el de la matriz.

    Es una funcion publica del modulo a proposito: la sonda hija la importa y la llama, de modo
    que el hijo mide con el **mismo** codigo que el padre y no con una copia.
    """
    return {
        "gate_sha256": gate_case().gate_sha256,
        "matrix_sha256": substrate(root).matrix.matrix_sha256,
    }


_PROBE: Final[str] = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    import test_integrity

    digests = test_integrity.process_digests(Path(sys.argv[1]))
    sys.stdout.write(digests["gate_sha256"] + "\\n")
    sys.stdout.write(digests["matrix_sha256"] + "\\n")
    """
)


def _probe(seed: str, root: Path) -> list[str]:
    """Ejecuta la sonda con ese `PYTHONHASHSEED` y devuelve sus lineas de `stdout`."""
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = seed
    environment["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT / "tests")])
    completed = subprocess.run(  # noqa: S603 - el ejecutable es el interprete de la sesion
        [sys.executable, "-c", _PROBE, str(root)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.splitlines()


# ─────────────────────────────────────────────────────────────────────────────
# A11-A13: los costes calculados a mano
# ─────────────────────────────────────────────────────────────────────────────
#: Los tres escenarios declarados, con los totales **escritos como literales** (A11).
DECLARED_SCENARIOS: Final[dict[str, tuple[Side, int, Decimal, Decimal]]] = {
    "intradia": (Side.LONG, 0, Decimal("0.42"), Decimal("0.0042")),
    "corto": (Side.SHORT, 1, Decimal("0.24"), Decimal("0.0024")),
    "largo": (Side.LONG, 1, Decimal("2.24"), Decimal("0.0224")),
}

#: Nocional declarado de `plan.md` §3.3: los importes de la tabla estan a 10.000 $.
DECLARED_NOTIONAL: Final[Decimal] = Decimal("10000")


def _declared_breakdown(
    *, scenario: str, slippage: SlippageParameter, notional_usd: Decimal = DECLARED_NOTIONAL
) -> CostBreakdown:
    """Coste declarado de un escenario, con el modelo importado de #11 (no reimplementado)."""
    side, nights, _usd, _pct = DECLARED_SCENARIOS[scenario]
    return cost_breakdown(
        model=declared_cost_model(),
        slippage=slippage,
        notional_usd=notional_usd,
        side=side,
        nights=nights,
        overnight_reason=None if nights == 0 else "escenario declarado a mano en el test (#17)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# A1 — sustrato hermetico
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_synthetic_store_assembles_the_five_families(tmp_path: Path) -> None:
    """A1: cinco familias en un almacen de `tmp_path`, desde los fixtures, sin tocar el mundo."""
    root = tmp_path / "almacen"
    built = substrate(root)
    matrix = built.matrix

    assert root.is_relative_to(tmp_path)
    assert built.root == root
    assert Store(root).datasets("raw") == ["macro", "market_daily", "sectors"]
    assert Store(root).datasets("derived") == ["labels"]
    assert matrix.n_sessions >= MIN_SESSIONS
    assert matrix.n_columns == N_COLUMNS
    assert matrix.duplicated_columns == ("atr_norm",)
    assert matrix.missing_series == ()
    assert matrix.feature_code_version == feature_store.FEATURE_CODE_VERSION
    assert set(matrix.feature_spec_sha256) == set(FAMILY_ORDER)

    # La matriz sale del almacen escrito (`store.sql()`), no de una lectura point-in-time: el
    # ancla tiene exactamente las sesiones de los dos fixtures de los que se compone.
    sessions = cast("list[date]", matrix.frame.get_column("session").to_list())
    expected = set(cast("list[date]", _volatility_inputs()["session"].to_list())) | set(
        cast("list[date]", _regime_inputs()["session"].to_list())
    )
    assert set(sessions) == expected
    assert matrix.first_session == min(expected)
    assert matrix.last_session == max(expected)
    stored = Store(root).sql("SELECT count(*) AS n FROM raw.market_daily WHERE series_id = '^GSPC'")
    assert int(stored.get_column("n")[0]) == matrix.n_sessions

    # Y los valores son **los del fixture**: el `^VIX` del almacen es la columna `vix_close` de
    # `golden_inputs.csv`, fila a fila.
    vix = Store(root).sql(
        "SELECT close FROM raw.market_daily WHERE series_id = '^VIX' ORDER BY as_of"
    )
    assert [float(value) for value in vix.get_column("close").to_list()] == [
        float(value) for value in _volatility_inputs().get_column("vix_close").to_list()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# A2 — el test central: *no-look-ahead* cross-familia
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_no_look_ahead_cross_family(frontier: Frontier) -> None:
    """A2: anadidas las sesiones posteriores, las filas `session <= t0` no se mueven.

    Se comparan **las 53 columnas** (las 52 features del catalogo y `session`), incluidas las
    `*_z`: son las que romperian el test si la normalizacion robusta usara la muestra completa
    (`_docs/plan.md` §9) en vez de la ventana expansiva.
    """
    assert frontier.before.height < frontier.after.height
    assert frontier.next_session > frontier.t0
    _assert_rows_unchanged(frontier.before, frontier.after, upto=frontier.t0)


# ─────────────────────────────────────────────────────────────────────────────
# A3 — el harness no es tautologico
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_no_look_ahead_controls_are_not_tautological(frontier: Frontier) -> None:
    """A3: los datos posteriores se consumen de verdad y el fixture no es degenerado.

    El control es la sesion inmediatamente posterior a la frontera: **tiene** que cambiar al
    anadir las sesiones siguientes (si no, la comparacion de A2 seria `None == None` sobre un
    fixture muerto). La unica columna que puede moverse es `sessions_to_opex`, que declara `null`
    el vencimiento que cae fuera del frame: cualquier otra que se moviera seria *look-ahead*.
    """
    after = frontier.after
    target = frontier.next_session

    def row(frame: pl.DataFrame, session: date) -> dict[str, object]:
        return dict(frame.filter(pl.col("session") == session).row(0, named=True))

    one = row(frontier.before, target)
    other = row(after, target)
    moved = sorted(
        column for column in after.columns if _cell_key(one[column]) != _cell_key(other[column])
    )
    assert moved == ["sessions_to_opex"], (
        "la sesion posterior a la frontera tiene que cambiar en el vencimiento y solo ahi: "
        f"cambio en {moved}"
    )
    assert one["sessions_to_opex"] is None and other["sessions_to_opex"] is not None

    assert after.height >= MIN_SESSIONS
    dead = [
        name
        for name in feature_store.ALL_FEATURE_COLUMNS
        if after.get_column(name).null_count() == after.height
    ]
    assert dead == [], f"el fixture congela columnas enteramente nulas: {dead}"


# ─────────────────────────────────────────────────────────────────────────────
# A4 — control negativo del comparador
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_negative_control_of_the_comparator(frontier: Frontier) -> None:
    """A4: el comparador **falla** con una celda corrompida y con una columna que falta."""
    # El control positivo: con el par real, el prefijo pasa.
    _assert_rows_unchanged(frontier.before, frontier.after, upto=frontier.t0)

    corrupted = frontier.before.with_columns(
        pl.when(pl.col("session") == frontier.t0)
        .then(pl.col("ret_1") * 1.5)
        .otherwise(pl.col("ret_1"))
        .alias("ret_1")
    )
    with pytest.raises(AssertionError) as cell:
        _assert_rows_unchanged(corrupted, frontier.after, upto=frontier.t0)
    assert "ret_1" in str(cell.value)
    assert frontier.t0.isoformat() in str(cell.value)
    assert "look-ahead" in str(cell.value)

    with pytest.raises(AssertionError) as dropped:
        _assert_rows_unchanged(frontier.before.drop("ret_1"), frontier.after, upto=frontier.t0)
    assert "ret_1" in str(dropped.value)


# ─────────────────────────────────────────────────────────────────────────────
# A5 — la frontera de la ventana expansiva
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_expanding_window_boundary(frontier: Frontier) -> None:
    """A5: en el minimo de sesiones, anadir la siguiente **no** mueve la `*_z` de `t`.

    El valor comparado es no nulo en los dos lados (`None == None` no demuestra nada) y con una
    sesion menos el borde no existe: el minimo se comprueba por los dos extremos, sobre una serie
    declarada y sobre la columna real del almacen.
    """
    windows = {
        entry.name: entry.window
        for family in FAMILY_ORDER
        for entry in feature_store.CATALOG_BY_FEATURE_SET[family]
        if entry.name.endswith(feature_store.NORMALISED_SUFFIX)
    }
    assert windows, "el catalogo no declara ninguna columna normalizada"
    assert set(windows.values()) == {feature_store.CONTEXT_MIN_SESSIONS}
    minimum = feature_store.CONTEXT_MIN_SESSIONS

    values = [1.0 + 0.25 * position + 0.01 * (position % 5) for position in range(minimum + 1)]
    sessions = [date(2026, 1, 1) + timedelta(days=position) for position in range(minimum + 1)]

    def normalised(count: int) -> pl.DataFrame:
        frame = pl.DataFrame(
            {
                "session": pl.Series("session", sessions[:count], dtype=pl.Date()),
                "x": pl.Series("x", values[:count], dtype=pl.Float64()),
            }
        )
        return feature_store.normalise_expanding(frame, "x", min_sessions=minimum)

    boundary = normalised(minimum)
    assert boundary.get_column("x_z")[minimum - 1] is not None
    assert normalised(minimum - 1).get_column("x_z")[minimum - 2] is None

    grown = normalised(minimum + 1)
    assert _cell_key(boundary.get_column("x_z")[minimum - 1]) == _cell_key(
        grown.get_column("x_z")[minimum - 1]
    )
    assert grown.get_column("x_z")[minimum - 1] is not None
    assert grown.get_column("x_z")[minimum] is not None

    # El mismo borde sobre la columna real del almacen sintetico: `atr_norm` del ancla.
    real = frontier.after.select("session", "atr_norm")
    with_z = feature_store.normalise_expanding(real, "atr_norm", min_sessions=minimum)
    position = next(
        index
        for index in range(with_z.height)
        if with_z.get_column("atr_norm_z")[index] is not None
    )
    assert position >= minimum - 1, "la ventana expansiva no puede cruzarse antes del minimo"
    trimmed = feature_store.normalise_expanding(
        real.head(position + 1), "atr_norm", min_sessions=minimum
    )
    extended = feature_store.normalise_expanding(
        real.head(position + 2), "atr_norm", min_sessions=minimum
    )
    assert trimmed.get_column("atr_norm_z")[position] is not None
    assert _cell_key(trimmed.get_column("atr_norm_z")[position]) == _cell_key(
        extended.get_column("atr_norm_z")[position]
    )


# ─────────────────────────────────────────────────────────────────────────────
# A6 — el fixture del golden
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_the_golden_fixture_follows_the_convention() -> None:
    """A6: claves declaradas, digests con prefijo y ninguna columna fuera del catalogo."""
    raw = GOLDEN.read_text(encoding="utf-8")
    golden = _read_golden()

    assert set(golden) == set(GOLDEN_KEYS), sorted(set(golden) ^ set(GOLDEN_KEYS))
    assert golden["code_version"] == feature_store.FEATURE_CODE_VERSION
    assert golden["feature_columns"] == ["session", *feature_store.ALL_FEATURE_COLUMNS]
    assert golden["n_columns"] == N_COLUMNS
    assert golden["duplicated_columns"] == ["atr_norm"]
    assert golden["first_session"] == "2023-07-11"
    assert golden["last_session"] == "2026-09-16"
    assert int(cast("int", golden["sessions"])) >= MIN_SESSIONS
    assert set(cast("dict[str, object]", golden["feature_spec_sha256"])) == set(FAMILY_ORDER)
    assert set(cast("dict[str, object]", golden["non_null"])) == set(
        feature_store.ALL_FEATURE_COLUMNS
    )
    assert golden["sources"] == _catalog_sources()
    assert golden["_comment"] == list(GOLDEN_COMMENT)
    assert golden["generator"] == GOLDEN_GENERATOR

    digests = [
        cast("str", golden["matrix_sha256"]),
        *cast("dict[str, str]", golden["feature_spec_sha256"]).values(),
    ]
    assert len(digests) == len(FAMILY_ORDER) + 1
    for digest in digests:
        assert digest.startswith(feature_store.FEATURE_VERSION_PREFIX)
        assert len(digest) == len(feature_store.FEATURE_VERSION_PREFIX) + 64
    # Ningun hex desnudo de 64 caracteres: es lo que bloquea `detect-secrets`, y un JSON no
    # admite `pragma: allowlist secret`, por eso todos los digests van con prefijo.
    assert not re.search(rf"(?<!{feature_store.FEATURE_VERSION_PREFIX})[0-9a-f]{{64}}", raw)


# ─────────────────────────────────────────────────────────────────────────────
# A7 — el golden recomputa
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_the_golden_matrix_recomputes(tmp_path: Path) -> None:
    """A7: la matriz recomputada desde el almacen coincide con el fixture, campo a campo."""
    golden = _read_golden()
    payload = golden_payload(tmp_path / "golden")

    assert payload["matrix_sha256"] == golden["matrix_sha256"]
    assert payload["feature_spec_sha256"] == golden["feature_spec_sha256"]
    assert payload["feature_columns"] == golden["feature_columns"]
    assert payload["sessions"] == golden["sessions"]
    assert payload["n_columns"] == golden["n_columns"]
    assert payload["n_columns"] == N_COLUMNS == 52
    assert len(cast("list[str]", payload["feature_columns"])) == N_COLUMNS + 1
    assert payload["first_session"] == golden["first_session"]
    assert payload["last_session"] == golden["last_session"]
    assert payload["non_null"] == golden["non_null"]
    assert payload["sources"] == golden["sources"]
    assert payload["duplicated_columns"] == golden["duplicated_columns"]
    assert payload == golden, "el golden completo (comentario y generador incluidos) cambio"


# ─────────────────────────────────────────────────────────────────────────────
# A8 — la puerta hash/version
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_version_gate_both_branches(tmp_path: Path) -> None:
    """A8: el digest cambia sin subir la constante ⇒ error; la constante sube ⇒ re-congelar.

    Las dos ramas se prueban con una **copia perturbada** del fixture en `tmp_path`: la puerta lee
    el par (``code_version``, ``matrix_sha256``) del fichero, no de una constante del test.
    """
    golden = _read_golden()
    digest = substrate(tmp_path / "almacen").matrix.matrix_sha256

    # La rama que pasa: el digest recomputado y la constante son los del fixture.
    _version_gate(digest=digest, golden=golden, code_version=feature_store.FEATURE_CODE_VERSION)

    silent = dict(golden)
    silent["matrix_sha256"] = feature_store.FEATURE_VERSION_PREFIX + "0" * 64
    (tmp_path / "golden_sin_declarar.json").write_text(
        json.dumps(silent, indent=2), encoding="utf-8"
    )
    perturbed = cast(
        "dict[str, Any]",
        json.loads((tmp_path / "golden_sin_declarar.json").read_text(encoding="utf-8")),
    )
    with pytest.raises(AssertionError) as undeclared:
        _version_gate(
            digest=digest, golden=perturbed, code_version=feature_store.FEATURE_CODE_VERSION
        )
    assert "sin declarar" in str(undeclared.value)
    assert "FEATURE_CODE_VERSION" in str(undeclared.value)

    # La segunda rama: la constante **subio** y el fixture sigue con el par viejo, asi que hay
    # que re-congelarlo (el digest que trae ya no es el de la constante nueva).
    with pytest.raises(AssertionError) as refreezed:
        _version_gate(
            digest=digest, golden=perturbed, code_version=feature_store.FEATURE_CODE_VERSION + 1
        )
    assert "re-congela" in str(refreezed.value)
    assert "FEATURE_CODE_VERSION subio" in str(refreezed.value)

    # Y un fixture que declara una version que ya no es la constante tambien pide re-congelarlo,
    # aunque el digest coincida.
    outdated = dict(golden)
    outdated["code_version"] = feature_store.FEATURE_CODE_VERSION + 1
    with pytest.raises(AssertionError) as stale_pair:
        _version_gate(
            digest=digest, golden=outdated, code_version=feature_store.FEATURE_CODE_VERSION
        )
    assert "re-congela" in str(stale_pair.value)


# ─────────────────────────────────────────────────────────────────────────────
# A9 — determinismo del gate, en proceso
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_gate_determinism_in_process() -> None:
    """A9: el mismo input declarado, cinco veces, da el mismo hash y el mismo texto canonico."""
    outputs = [gate_case() for _ in range(5)]
    digests = {output.gate_sha256 for output in outputs}
    assert len(digests) == 1
    assert outputs[0].gate_sha256.startswith(gate.GATE_HASH_PREFIX)
    assert outputs[0].gate_sha256 == gate.gate_sha256(outputs[0])
    assert outputs[0].status.value == "recommendation"

    texts = {
        canonical_text(gate._json_payload(output))  # pyright: ignore[reportPrivateUsage]
        for output in outputs
    }
    assert len(texts) == 1
    text = texts.pop()
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == outputs[0].gate_sha256.removeprefix(
        gate.GATE_HASH_PREFIX
    )
    for output in outputs:
        assert output == outputs[0]


# ─────────────────────────────────────────────────────────────────────────────
# A10 — determinismo entre procesos
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_across_processes_determinism(tmp_path: Path) -> None:
    """A10: con `PYTHONHASHSEED` 0, 1 y `random`, los dos digests son los mismos.

    El hijo imprime **solo** los dos digests por `stdout`: nada de logs que podrian tapar una
    diferencia entre procesos.
    """
    root = tmp_path / "padre"
    assert root.is_relative_to(tmp_path)
    local = process_digests(root)
    outputs = [_probe(seed, tmp_path / f"semilla_{seed}") for seed in ("0", "1", "random")]

    assert [len(lines) for lines in outputs] == [2, 2, 2]
    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[0][0] == local["gate_sha256"] == gate_case().gate_sha256
    assert outputs[0][1] == local["matrix_sha256"]
    assert outputs[0][1] == _read_golden()["matrix_sha256"].removeprefix(
        feature_store.FEATURE_VERSION_PREFIX
    )


# ─────────────────────────────────────────────────────────────────────────────
# A11-A13 — los costes a mano
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_costs_three_declared_scenarios() -> None:
    """A11: los totales declarados, exactos, con el modelo y el supuesto importados de #11."""
    assumption = declared_slippage_assumption()
    for scenario, (_side, _nights, usd, pct) in DECLARED_SCENARIOS.items():
        breakdown = _declared_breakdown(scenario=scenario, slippage=assumption)
        assert breakdown.c_declared_usd == usd
        assert breakdown.c_declared_pct == pct
        assert breakdown.notional_usd == DECLARED_NOTIONAL
    assert assumption.state.value == "assumed"


def test_a12_costs_dominance_ratio() -> None:
    """A12: el ratio de dominancia es `Decimal("0.2") / Decimal("0.0042")`, tambien literal."""
    measured = SlippageParameter.measured(
        pct_of_notional=GATE_SLIPPAGE_PCT,
        source="tests/test_integrity.py (#17)",
        reason="slippage medido declarado a mano para el ratio de dominancia",
    )
    breakdown = _declared_breakdown(scenario="intradia", slippage=measured)
    ratio = breakdown.slippage_over_spread_ratio
    assert ratio is not None
    assert ratio == (GATE_SLIPPAGE_PCT / Decimal("0.0042")).quantize(
        RATIO_QUANTUM, rounding=ROUND_HALF_UP
    )
    assert ratio == Decimal("47.619")
    assert breakdown.slippage_pct == GATE_SLIPPAGE_PCT
    assert breakdown.slippage_dominates is True


def test_a13_costs_slippage_states_are_never_merged() -> None:
    """A13: `assumed` deja el total nulo con motivo, `measured` lo cierra y `unmeasured` nulo."""
    model = declared_cost_model()
    assumed = declared_slippage_assumption()
    measured = SlippageParameter.measured(
        pct_of_notional=Decimal("0.01"),
        source="tests/test_integrity.py (#17)",
        reason="medicion declarada a mano para el estado medido",
    )
    unmeasured = SlippageParameter.unmeasured(reason="no hay ninguna ejecucion real que medir")

    assumed_breakdown = cost_breakdown(
        model=model, slippage=assumed, notional_usd=DECLARED_NOTIONAL, side=Side.LONG, nights=0
    )
    assert assumed_breakdown.c_total_pct is None
    assert assumed_breakdown.c_total_usd is None
    assert assumed_breakdown.c_declared_usd == Decimal("0.42")
    assert assumed_breakdown.nulls
    assert "supuesto" in assumed_breakdown.nulls[0]["reason"]

    measured_breakdown = cost_breakdown(
        model=model, slippage=measured, notional_usd=DECLARED_NOTIONAL, side=Side.LONG, nights=0
    )
    assert measured_breakdown.c_total_pct == measured_breakdown.c_declared_pct + Decimal("0.01")
    assert measured_breakdown.c_total_usd == Decimal("1.42")
    assert measured_breakdown.nulls == ()

    unmeasured_breakdown = cost_breakdown(
        model=model, slippage=unmeasured, notional_usd=DECLARED_NOTIONAL, side=Side.LONG, nights=0
    )
    assert unmeasured_breakdown.c_total_pct is None
    assert unmeasured_breakdown.c_total_usd is None
    assert unmeasured_breakdown.nulls
    assert "#62" in unmeasured_breakdown.nulls[0]["reason"]

    states = {
        assumed_breakdown.slippage.state,
        measured_breakdown.slippage.state,
        unmeasured_breakdown.slippage.state,
    }
    assert len(states) == 3
    assert measured_breakdown.c_total_pct is not None
    assert measured_breakdown.c_total_pct > Decimal(0)


# ─────────────────────────────────────────────────────────────────────────────
# A14 — el modulo no toca el mundo
# ─────────────────────────────────────────────────────────────────────────────
def _literal_strings(tree: ast.Module) -> list[str]:
    """Constantes de texto del modulo, **sin** sus docstrings (donde la prosa puede hablar)."""
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_a14_no_clock_no_io(tmp_path: Path) -> None:
    """A14: por AST, el modulo no lee el almacen real, ni el reloj, ni la red, ni escribe fuera."""
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    for text in _literal_strings(tree):
        if text in FORBIDDEN_TEXT:
            continue  # la constante que **declara** el patron es el unico sitio donde cabe
        for forbidden in FORBIDDEN_TEXT:
            assert forbidden not in text, f"el modulo menciona {forbidden!r}: {text!r}"

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported & set(NETWORK_MODULES) == set(), sorted(imported & set(NETWORK_MODULES))

    called = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name | ast.Attribute)
    }
    assert called & set(CLOCK_NAMES) == set(), sorted(called & set(CLOCK_NAMES))
    assert {"open", "urlopen"} & called == set()

    writes = [
        ast.unparse(node.func.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in WRITE_METHODS
    ]
    assert writes, "el modulo no declara ninguna escritura: la comprobacion no probaria nada"
    for target in writes:
        assert "tmp_path" in target, f"el modulo escribe en {target!r}, que no cuelga de tmp_path"

    # Y la prueba de comportamiento: la raiz del almacen que construye este test cuelga de
    # `tmp_path`, que es lo unico que el modulo escribe. El almacen real del repositorio y el
    # registro de experimentos los huella `tests/conftest.py` antes y despues de la sesion.
    root = tmp_path / "mundo"
    substrate(root)
    assert root.is_relative_to(tmp_path)
    assert root.exists()


# ─────────────────────────────────────────────────────────────────────────────
# A15 — la puerta local de `pre-push`
# ─────────────────────────────────────────────────────────────────────────────
def test_a15_the_pre_push_hook_is_declared() -> None:
    """A15: el hook local de `pre-push` existe, corre siempre y **no** recibe ficheros.

    Las tres cosas importan: `stages: [pre-push]` lo saca del commit, y `always_run: true` con
    `pass_filenames: false` lo convierten en una puerta de verdad (sin ellas seria un no-op
    silencioso que no comprueba nada).
    """
    config = cast("dict[str, Any]", yaml.safe_load(PRE_COMMIT.read_text(encoding="utf-8")))
    hooks = [
        hook
        for repo in cast("list[dict[str, Any]]", config["repos"])
        if repo.get("repo") == "local"
        for hook in cast("list[dict[str, Any]]", repo["hooks"])
    ]
    declared = [hook for hook in hooks if hook["id"] == PRE_PUSH_HOOK]
    assert len(declared) == 1, f"el hook local {PRE_PUSH_HOOK!r} no esta declarado una sola vez"
    hook = declared[0]
    assert hook["stages"] == ["pre-push"]
    assert hook["always_run"] is True
    assert hook["pass_filenames"] is False
    assert hook["language"] == "system"
    entry = str(hook["entry"])
    assert "tests/test_integrity.py" in entry
    assert "version_gate" in entry or "golden_matrix" in entry
    assert "uv run pytest" in entry


# ─────────────────────────────────────────────────────────────────────────────
# A16 — cierre: cada criterio tiene su test y su selector
# ─────────────────────────────────────────────────────────────────────────────
def test_a16_every_criterion_has_its_test() -> None:
    """A16: `test_aN_...` para cada criterio y los `-k` del enunciado seleccionan algo.

    La mitad que se puede comprobar desde dentro de la suite es el mapeo criterio → test y que
    cada selector de `-k` del enunciado case con **al menos** un test. La otra mitad (la suite
    entera en verde sobre un arbol commiteado y limpio) se mide corriendo la suite.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    assert len(names) == len(set(names)), "hay dos tests con el mismo nombre"

    grouped: dict[str, list[str]] = {}
    for name in names:
        match = re.match(r"test_(a\d+)_", name)
        if match:
            grouped.setdefault(match.group(1), []).append(name)
    assert set(grouped) == {f"a{number}" for number in range(1, 17)}, sorted(grouped)

    for criterion, selectors in SELECTORS.items():
        for selector in selectors:
            matched = [name for name in names if selector in name]
            assert matched, f"el selector -k {selector!r} de {criterion} no selecciona ningun test"
