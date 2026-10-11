"""Tests de #153: las familias ancladas al S&P sirven su fila `t` desde la sesion pendiente.

Cubre los criterios de la issue:

- **C1** — el catalogo declara `design_lag = 0` para `har_forecast`, `vix_zscore`,
  `garch_forecast_z` e `is_es_roll_session`; los dos **forzados** siguen en `1`.
- **C2** — el valor pendiente es **la fila `t` de la matriz** (una regla, un valor); la matriz
  **cerrada** no gana ninguna fila ni se mueve.
- **C3** — sin *look-ahead*: una columna con `design_lag = 1` **no** se sirve en el instantaneo y
  una barra posterior a `t` no cambia el valor leido en `t`.
- **C5** — lo no computable sale ``null``, nunca un cero.
- **C6** — determinismo (dos procesos, `PYTHONHASHSEED` distinto).

Los casos de almacen se saltan si el `data/` real no esta en el arbol (en solo lectura, como el
resto de la suite: `tests/conftest.py` huella `data/` y `runs/`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Final, cast

import polars as pl
import pytest

from cfdtrader.analysis import feature_frame as ff
from cfdtrader.data.store import Store
from cfdtrader.delivery import run_daily
from cfdtrader.features import context as context_features
from cfdtrader.features import store as feature_store
from cfdtrader.models.baseline import BASELINE_FEATURES

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"

#: Las cuatro columnas que #153 mueve: su fila `t` se calcula solo con sesiones `<= t-1`.
MOVED_COLUMNS: Final[tuple[str, ...]] = (
    "har_forecast",
    "vix_zscore",
    "garch_forecast_z",
    "is_es_roll_session",
)

#: Las dos cuyo desfase es **forzado**: la barra del ancla de `t`, o el `as_of` de la familia macro.
FORCED_COLUMNS: Final[tuple[str, ...]] = ("dist_sma_20_z", "ust_10y_chg_5")

#: Las columnas que la **auditoria de disponibilidad** (punto 1) anadio despues de #153: su fila
#: `t` solo usa sesiones `<= t-1` —o el calendario—, asi que se sirven desde la fila pendiente con
#: la misma regla que las cuatro de #153. No estan en `BASELINE_FEATURES`: el catalogo las declara
#: para que, si algun dia se promueven al modelo, no haya *train/serve skew*.
AUDITED_COLUMNS: Final[tuple[str, ...]] = (
    "har_lag1",
    "har_lag4",
    "har_lag17",
    "vix_level",
    "vix_percentile",
    "atr_norm",
    "atr_norm_z",
    "garch_forecast",
    "rv_percentile",
    "day_of_week",
)

#: El **motivo** de cada columna movida: su fila `t` se calcula con sesiones `<= t-1`, o es una
#: marca del calendario. Es la auditoria leida de la **formula** (`volatility.py`, `technical.py`,
#: `regime.py`), no de la prosa del `req_as_of` (que en `atr_norm` decia `t` y es `t-1`).
AUDIT_REASONS: Final[dict[str, str]] = {
    "har_lag1": "parkinson_rv de t-1: la ventana ya esta desplazada una sesion",
    "har_lag4": "media de parkinson_rv de t-2 ... t-5: toda la ventana es < t",
    "har_lag17": "media de parkinson_rv de t-6 ... t-22: toda la ventana es < t",
    "vix_level": "el cierre del VIX de t-1 (shift(1)): publicado antes del snapshot",
    "vix_percentile": "percentil del nivel de VIX de t-1 sobre su historia <= t-1",
    "atr_norm": "true_range de t-20 ... t-1 normalizado por C_{t-1}: la sesion t no entra",
    "atr_norm_z": "normalizacion expandida de atr_norm, que ya es <= t-1",
    "garch_forecast": "omega + alpha*eps^2_T + beta*sigma^2_T con T = t-1",
    "rv_percentile": "percentil de la volatilidad realizada de t-1 sobre la historia < t-1",
    "day_of_week": "isoweekday() de la sesion: calendario, no depende de ninguna barra",
}

#: Las columnas que **no** pueden servirse en el instantaneo porque su fila `t` necesita la barra
#: del ancla de `t` (cierre a las 16:00 ET): leerlas antes seria *look-ahead*. Su desfase es
#: **forzado** por el reloj, no evitable.
FORCED_BY_ANCHOR_CLOCK: Final[tuple[str, ...]] = (
    "true_range",
    "parkinson_rv",
    "ret_log",
    "ret_sq",
    "ret_1",
    "ret_5",
    "ret_21",
    "dist_sma_20",
    "dist_sma_20_z",
    "rsi_14",
    "range_pos_20",
    "vol_break_20",
    "efficiency_ratio_20",
)

#: Las de la familia **macro**: su corte *point-in-time* es el `as_of` de la fila (16:00 ET): una
#: publicacion posterior al instantaneo sigue entrando en la sesion; adelantarlas seria
#: *look-ahead* (la auditoria razonada de `ust_10y_chg_5` esta en #150).
FORCED_BY_MACRO_CUT: Final[tuple[str, ...]] = (
    "fed_funds",
    "fed_funds_chg_5",
    "ust_10y",
    "ust_10y_chg_5",
    "ust_2y",
    "ust_2y_chg_5",
    "pendiente_2s10s",
    "pendiente_2s10s_chg_5",
    "cpi_yoy",
    "pce_yoy",
    "dxy",
    "ust_10y_z",
    "dxy_z",
)

#: Las que **podrian** declararse `0` (su input cierra antes del snapshot) pero **aun no tienen
#: lector pendiente**: las commodities/FX y las de contexto que no son de las cuatro ya publicadas.
#: Conservan el `1` conservador y su auditoria es una tarea aparte. `sessions_to_opex` es de
#: calendario pero arrastra nulos de cola (#79), asi que tambien espera.
DEFERRED_READER: Final[tuple[str, ...]] = (
    "corr_dax_60",
    "corr_ftse_60",
    "corr_stoxx_60",
    "corr_nikkei_60",
    "beta_vix_60",
    "sector_count",
    "sector_dispersion_1_z",
    "sessions_to_opex",
    "oil_ret_1",
    "oil_ret_5",
    "oil_ret_1_z",
    "gold_ret_1",
    "eurusd_ret_1",
)

needs_store = pytest.mark.skipif(
    not (REAL_DATA / "derived" / "labels").exists(),
    reason="el almacen real no esta en el arbol: C2/C3/C6 se miden sobre el (en solo lectura)",
)


# ─────────────────────────────────────────────────────────────────────────────
# C1 - la declaracion del catalogo
# ─────────────────────────────────────────────────────────────────────────────
def test_c1_the_moved_columns_are_declared_readable_at_the_snapshot() -> None:
    """Las cuatro columnas de #153 declaran `design_lag = 0` y tienen lector pendiente (C1)."""
    declared = feature_store.DESIGN_LAG_BY_FEATURE
    assert all(declared[name] == 0 for name in MOVED_COLUMNS)
    assert set(MOVED_COLUMNS) <= set(ff.PENDING_FAMILY_COLUMNS)
    # #153 + la auditoria de disponibilidad del punto 1: el lector pendiente es **exactamente**
    # las cuatro de #153 mas las auditadas (la union completa la cierra
    # `test_issue_150::test_c2_...`).
    assert set(ff.PENDING_FAMILY_COLUMNS) == set(MOVED_COLUMNS) | set(AUDITED_COLUMNS)


def test_c1_the_forced_lag_columns_are_untouched() -> None:
    """Los dos forzados siguen en `1` y el rezago de referencia del diseno no se mueve (C1)."""
    declared = feature_store.DESIGN_LAG_BY_FEATURE
    assert all(declared[name] == 1 for name in FORCED_COLUMNS)
    assert max(declared[name] for name in BASELINE_FEATURES) == 1


def test_c1_the_pending_families_are_the_snap_anchored_ones() -> None:
    """Cada columna pendiente declara de que familia sale, y son las ancladas al S&P (C1)."""
    assert set(ff.PENDING_FAMILY_COLUMNS.values()) == {
        feature_store.VOLATILITY_FEATURE_SET,
        feature_store.TECHNICAL_FEATURE_SET,
        feature_store.REGIME_FEATURE_SET,
    }
    # Las cuatro de #153 **si** son del contrato de #24; las auditadas despues, no.
    assert set(MOVED_COLUMNS) <= set(BASELINE_FEATURES)
    assert set(AUDITED_COLUMNS).isdisjoint(BASELINE_FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# La auditoria de disponibilidad: las 57 columnas, en tres grupos declarados
# ─────────────────────────────────────────────────────────────────────────────
def test_the_audit_partitions_every_catalog_column() -> None:
    """Cada una de las 57 columnas cae en **exactamente** un grupo declarado.

    Los tres grupos son: las legibles en el instantaneo (`design_lag = 0`), las forzadas por el
    reloj del ancla o por el corte macro, y las conservadoras a la espera de su lector. Si una
    columna nueva del catalogo no cae en ninguno, esta prueba falla: eso es la auditoria.
    """
    declared = feature_store.DESIGN_LAG_BY_FEATURE
    readable = {name for name, lag in declared.items() if lag == 0}
    forced = set(FORCED_BY_ANCHOR_CLOCK) | set(FORCED_BY_MACRO_CUT)
    deferred = set(DEFERRED_READER)
    assert readable == set(context_features.PENDING_COLUMNS) | set(ff.PENDING_FAMILY_COLUMNS)
    assert not (forced & deferred)
    assert not (readable & forced)
    assert not (readable & deferred)
    assert readable | forced | deferred == set(declared)
    assert all(declared[name] == 1 for name in forced | deferred)


def test_the_audited_columns_have_a_declared_reason() -> None:
    """Cada columna movida a `0` por la auditoria declara su motivo, y su lectura coincide."""
    assert set(AUDIT_REASONS) == set(AUDITED_COLUMNS)
    for name, reason in AUDIT_REASONS.items():
        assert len(reason) > 30, name
        assert feature_store.DESIGN_LAG_BY_FEATURE[name] == 0, name


# ─────────────────────────────────────────────────────────────────────────────
# C3 - sin look-ahead: lo que no se declara no se sirve
# ─────────────────────────────────────────────────────────────────────────────
def test_c3_a_lag_one_column_is_not_readable_at_the_snapshot(tmp_path: Path) -> None:
    """Una columna con `design_lag = 1` **no** se sirve en el instantaneo: error tipado (C3).

    El rechazo ocurre **antes** de tocar el almacen, asi que la prueba es hermética.
    """
    with pytest.raises(feature_store.ContextInputError):
        ff.pending_session_values(
            Store(tmp_path), session=date(2026, 10, 9), columns=[FORCED_COLUMNS[0]]
        )
    with pytest.raises(feature_store.ContextInputError):
        ff.pending_session_values(Store(tmp_path), session=date(2026, 10, 9), columns=[])
    with pytest.raises(feature_store.ContextInputError):
        ff.pending_session_values(
            Store(tmp_path), session=date(2026, 10, 9), columns=["una_columna_que_no_existe"]
        )


def test_c3_a_missing_anchor_is_a_typed_error(tmp_path: Path) -> None:
    """El camino pendiente no inventa una fila: sin ancla en el almacen, error tipado (C3)."""
    with pytest.raises((ff.MissingFeatureDatasetError, feature_store.FeatureStoreError)):
        ff.pending_session_values(
            Store(tmp_path), session=date(2026, 10, 9), columns=[MOVED_COLUMNS[0]]
        )


# ─────────────────────────────────────────────────────────────────────────────
# C5 - lo no computable sale `null`, nunca un cero
# ─────────────────────────────────────────────────────────────────────────────
def _synthetic_inputs(
    rows: int,
) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
    """Un ancla sintetica con precios positivos y su VIX: lo minimo para las dos familias."""
    days = [date(2020, 1, 1) + timedelta(days=index) for index in range(rows)]
    closes = [100.0 + index for index in range(rows)]
    anchor = pl.DataFrame(
        {
            "session": days,
            "open": closes,
            "high": [value + 1.0 for value in closes],
            "low": [value - 1.0 for value in closes],
            "close": closes,
        },
        schema_overrides={"session": pl.Date()},
    )
    vix = pl.DataFrame(
        {"session": days, "close": [20.0 + (index % 5) for index in range(rows)]},
        schema_overrides={"session": pl.Date()},
    )
    return {"^GSPC": anchor, "^VIX": vix}, {}, {}


def _patch_load_raw(monkeypatch: pytest.MonkeyPatch, rows: int) -> date:
    """Sustituye la lectura del almacen por el frame sintetico y devuelve la sesion pendiente."""
    market, sectors, macro = _synthetic_inputs(rows)
    pending = cast("date", market["^GSPC"].get_column("session").to_list()[-1]) + timedelta(days=1)

    def fake(
        store: Store,
    ) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
        del store
        return market, sectors, macro

    monkeypatch.setattr(ff, "_load_raw", fake)  # pyright: ignore[reportPrivateUsage]
    return pending


def test_c5_without_history_the_family_columns_are_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con cinco sesiones no hay pronostico: ``null``, nunca un cero (C5)."""
    session = _patch_load_raw(monkeypatch, 5)
    values = ff.pending_session_values(
        Store(tmp_path), session=session, columns=list(MOVED_COLUMNS)
    )
    assert values["har_forecast"] is None
    assert values["vix_zscore"] is None
    assert values["garch_forecast_z"] is None
    assert values["is_es_roll_session"] in {0.0, 1.0}


def test_c5_with_history_the_family_columns_are_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con historia suficiente, las tres numericas se publican y son finitas (C5)."""
    session = _patch_load_raw(monkeypatch, 520)
    values = ff.pending_session_values(
        Store(tmp_path), session=session, columns=list(MOVED_COLUMNS)
    )
    for name in ("har_forecast", "vix_zscore"):
        assert values[name] is not None, name
    assert values["is_es_roll_session"] in {0.0, 1.0}
    assert all(value is None or float(value) == float(value) for value in values.values())


def test_c3_a_bar_after_the_pending_session_does_not_change_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Una barra **posterior** a `t` no entra: el valor de `t` solo usa sesiones `<= t-1` (C3)."""
    plain = _patch_load_raw(monkeypatch, 60)
    at_t = ff.pending_session_values(Store(tmp_path), session=plain, columns=list(MOVED_COLUMNS))
    market, sectors, macro = _synthetic_inputs(60)
    anchor = market["^GSPC"]
    after = pl.DataFrame(
        {
            "session": [plain + timedelta(days=1)],
            "open": [9_999.0],
            "high": [10_000.0],
            "low": [9_998.0],
            "close": [9_997.0],
        },
        schema_overrides={"session": pl.Date()},
    )
    shocked_market = {**market, "^GSPC": pl.concat([anchor, after], how="vertical")}

    def shocked(
        store: Store,
    ) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
        del store
        return shocked_market, sectors, macro

    monkeypatch.setattr(ff, "_load_raw", shocked)  # pyright: ignore[reportPrivateUsage]
    with_after = ff.pending_session_values(
        Store(tmp_path), session=plain, columns=list(MOVED_COLUMNS)
    )
    assert at_t == with_after


# ─────────────────────────────────────────────────────────────────────────────
# C2 - sobre el almacen real: el valor pendiente es la fila `t`, y la matriz no se mueve
# ─────────────────────────────────────────────────────────────────────────────
def _hide_anchor_bar(monkeypatch: pytest.MonkeyPatch, session: date) -> None:
    """Deja el almacen como en el instantaneo: el ancla y el VIX no tienen la barra de `session`."""
    real = ff._load_raw  # pyright: ignore[reportPrivateUsage]
    hidden = {ff.ANCHOR_SERIES, ff.VIX_SERIES}

    def truncated(
        store: Store,
    ) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
        market, sectors, macro = real(store)
        for name in hidden:
            frame = market.get(name)
            if frame is not None:
                market[name] = frame.filter(pl.col("session") < session)
        return market, sectors, macro

    monkeypatch.setattr(ff, "_load_raw", truncated)  # pyright: ignore[reportPrivateUsage]


@needs_store
def test_c2_the_pending_value_is_the_matrix_row_of_the_same_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El valor servido en `t` es **el mismo** que el de la fila `t` de la matriz (C2).

    Es el invariante de una sola regla: la funcion de familia es la misma y la fila pendiente solo
    anade la sesion que el ancla aun no ha cerrado.
    """
    store = Store(REAL_DATA)
    matrix = ff.build_feature_matrix(store)
    session = matrix.last_session
    _hide_anchor_bar(monkeypatch, session)
    columns = (*MOVED_COLUMNS, *AUDITED_COLUMNS)
    served = ff.pending_session_values(store, session=session, columns=list(columns))
    row = matrix.frame.filter(pl.col("session") == session).row(0, named=True)
    for name in columns:
        assert float(cast("float", served[name])) == float(cast("float", row[name])), name


@needs_store
def test_c2_the_closed_matrix_does_not_gain_a_row() -> None:
    """La fila pendiente es *opt-in*: la matriz cerrada tiene exactamente las sesiones del ancla."""
    store = Store(REAL_DATA)
    matrix = ff.build_feature_matrix(store)
    market, _sectors, _macro = ff._load_raw(store)  # pyright: ignore[reportPrivateUsage]
    anchor = market[ff.ANCHOR_SERIES]
    assert matrix.frame.height == anchor.height
    assert matrix.last_session == cast("date", anchor.get_column("session").to_list()[-1])


@needs_store
def test_c2_the_served_vector_carries_the_fresh_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`run_daily._served_vector` sirve las cuatro columnas de `t`, no las de `t-1` (C2)."""
    store = Store(REAL_DATA)
    matrix = ff.build_feature_matrix(store)
    session = matrix.last_session
    sessions = [cast("date", value) for value in matrix.frame.get_column("session").to_list()]
    previous = session - timedelta(days=1)
    while previous not in set(sessions):
        previous -= timedelta(days=1)
    row = matrix.frame.filter(pl.col("session") == previous).row(0, named=True)
    reference = matrix.frame.filter(pl.col("session") == session).row(0, named=True)
    _hide_anchor_bar(monkeypatch, session)
    served = run_daily._served_vector(  # pyright: ignore[reportPrivateUsage]
        store, session=session, row=row
    )
    assert set(served) == set(BASELINE_FEATURES)
    for name in MOVED_COLUMNS:
        assert float(cast("float", served[name])) == float(cast("float", reference[name])), name
    # Las tres numericas **no** pueden venir de la fila anterior: si vinieran, el desfase seguiria
    # ahi. `is_es_roll_session` es una marca 0/1 del calendario y puede coincidir con la de `t-1`
    # (la sesion medida no es OPEX), asi que su prueba es la igualdad con la fila `t`, no el cambio.
    for name in MOVED_COLUMNS[:3]:
        assert served[name] != row[name], f"{name} sigue llegando de la fila anterior"


# ─────────────────────────────────────────────────────────────────────────────
# C6 - determinismo
# ─────────────────────────────────────────────────────────────────────────────
CHILD: Final[str] = (
    "import json, polars as pl\n"
    "from cfdtrader.analysis import feature_frame as ff\n"
    "from cfdtrader.data.store import Store\n"
    "real = ff._load_raw\n"
    "store = Store('data')\n"
    "matrix = ff.build_feature_matrix(store)\n"
    "session = matrix.last_session\n"
    "def truncated(s):\n"
    "    market, sectors, macro = real(s)\n"
    "    for name in (ff.ANCHOR_SERIES, ff.VIX_SERIES):\n"
    "        market[name] = market[name].filter(pl.col('session') < session)\n"
    "    return market, sectors, macro\n"
    "ff._load_raw = truncated\n"
    "columns = sorted(ff.PENDING_FAMILY_COLUMNS)\n"
    "values = ff.pending_session_values(store, session=session, columns=columns)\n"
    "print(json.dumps({k: repr(v) for k, v in values.items()}, sort_keys=True))\n"
)


def _child_pending(hashseed: str) -> dict[str, object]:
    """Las columnas de la fila pendiente en un proceso fresco con esa semilla de `hash`."""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", CHILD],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": hashseed},
        cwd=str(REPO_ROOT),
    )
    return cast("dict[str, object]", json.loads(completed.stdout))


@needs_store
def test_c6_two_processes_agree_on_the_pending_family_read() -> None:
    """La lectura pendiente es pura: dos procesos con `PYTHONHASHSEED` distinto coinciden (C6)."""
    first = _child_pending("0")
    second = _child_pending("1")
    assert first == second
    assert set(first) == set(MOVED_COLUMNS) | set(AUDITED_COLUMNS)
    assert all(value != "None" for value in first.values())
