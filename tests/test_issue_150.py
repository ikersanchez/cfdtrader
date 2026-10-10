"""Tests de #150: los cierres de mercado ajeno que ya estan publicados entran en la decision.

- **C2** la regla es la de #147: una sola declaracion (`design_lag`) y un solo lector.
- **C3** el diseno de `t` lee el cierre **europeo** de `t-1` (y el del dolar y la dispersion
  sectorial), no el de `t-2`; golden medido del caso 2026-10-09.
- **C4** el cierre de la sesion que se decide —11:30 ET, **despues** del snapshot— no entra.
- **C5** la auditoria de las 10 features queda declarada, columna a columna y con su motivo.
- **C6** las columnas sin desfase evitable no cambian de valor.
- **C7** una sola regla, dos sitios: el vector servido y la fila de diseno de la misma sesion.
- **C8** la lectura pendiente es determinista (dos procesos, `PYTHONHASHSEED` distinto).

Los casos de almacen se saltan si el `data/` real no esta en el arbol (en solo lectura, como el
resto de la suite: `tests/conftest.py` huella `data/` y `runs/`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Final, cast

import polars as pl
import pytest

from cfdtrader.analysis import feature_frame as ff
from cfdtrader.analysis.feature_frame import build_feature_frame, pending_session_values
from cfdtrader.data.store import Store
from cfdtrader.delivery import run_daily
from cfdtrader.features import context as context_features
from cfdtrader.features import store as feature_store
from cfdtrader.models.baseline import (
    BASELINE_FEATURES,
    DESIGN_SESSION_COLUMN,
    design_frame,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"

#: El snapshot que abrio la issue: a las 08:45 ET del 2026-10-09 el ultimo cierre europeo publicado
#: era el del **8** (11:30 ET del 8), no el del 7.
GOLDEN_SESSION: Final[date] = date(2026, 10, 9)
GOLDEN_PREVIOUS: Final[date] = date(2026, 10, 8)

#: Las tres columnas que #150 mueve: su input cierra **antes** del snapshot, asi que la fila `t`
#: ya lo tiene publicado.
MOVED_COLUMNS: Final[tuple[str, ...]] = (
    "europe_prev_1",
    "dxy_ret_1",
    "sector_dispersion_1",
)

#: El caso 2026-10-09 medido, columna a columna: ``(lo que usaba el modelo, lo ya publicado)``. El
#: primero es la fila de `t-1` de la matriz (lo que leia el desplazamiento uniforme de #24 —Europa
#: del 7) y el segundo la fila de `t`, que es la que la lectura pendiente sirve. Son **valores del
#: caso**, no digests de un artefacto regenerable (`_docs/team/pm.md`).
GOLDEN: Final[dict[str, tuple[float, float]]] = {
    "europe_prev_1": (-0.012110974841203624, -0.00741270524678685),
    "dxy_ret_1": (0.004018195569891882, -0.0009785544910906408),
    "sector_dispersion_1": (0.008562132362673374, 0.012348854177046863),
}

needs_store = pytest.mark.skipif(
    not (REAL_DATA / "derived" / "labels").exists(),
    reason="el almacen real no esta en el arbol: C3/C7 se miden sobre el (en solo lectura)",
)


@pytest.fixture(scope="session")
def real_frame() -> ff.FeatureFrame:
    """La matriz + el diseno reales, construidos **una vez** para toda la sesion."""
    return build_feature_frame(Store(REAL_DATA))


def _row(frame: pl.DataFrame, session: date) -> Mapping[str, object]:
    """La fila de esa sesion, como `Mapping` (o un fallo claro si no esta)."""
    selected = frame.filter(pl.col("session") == session)
    assert selected.height == 1, f"la sesion {session} no esta exactamente una vez"
    return selected.row(0, named=True)


def _synthetic(rows: int = 12) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Features con valores que dicen **de que sesion** viene cada celda (como en #147)."""
    days = [date(2024, 1, 2) + timedelta(days=index) for index in range(rows)]
    data: dict[str, list[object]] = {"session": list(days)}
    for position, name in enumerate(BASELINE_FEATURES):
        data[name] = [float(index * 10 + position) for index in range(rows)]
    labels = pl.DataFrame(
        {"session": days[1:], "ret_long": [0.01 * (index % 3 - 1) for index in range(1, rows)]}
    )
    return pl.DataFrame(data), labels


#: La auditoria de #150 (C5): para **cada** feature de #24, el rezago declarado y su motivo. Los
#: cuatro `0` son las que ya estan publicadas en el snapshot; los seis `1` se justifican uno a uno
#: —dos **forzados** por el reloj del ancla y cuatro **evitables** pero bloqueados por la
#: arquitectura de la matriz (seguimiento en #153)—. La tabla es el contrato declarado, no una
#: medicion de un artefacto regenerable.
AUDIT: Final[dict[str, tuple[int, str]]] = {
    "asia_overnight_1": (
        0,
        "Tokio cierra a las 06:00 UTC y Hong Kong a las 08:00 UTC: ya publicados (#147)",
    ),
    "europe_prev_1": (
        0,
        "Europa cierra a las 11:30 ET del dia anterior, antes del snapshot (#150)",
    ),
    "dxy_ret_1": (0, "el DXY cierra a las 17:00 ET del dia anterior, antes del snapshot (#150)"),
    "sector_dispersion_1": (
        0,
        "los ETF sectoriales cierran a las 16:00 ET del dia anterior, antes del snapshot (#150)",
    ),
    "vix_zscore": (
        1,
        "evitable: la ventana ya esta desplazada (`vix_close` de `t-1`), pero es la familia de "
        "volatilidad, anclada al calendario del S&P, y su fila `t` no es publicable sin una fila "
        "pendiente en la matriz (#153)",
    ),
    "har_forecast": (
        1,
        "evitable: se ajusta con las sesiones `< t`, pero vive en la familia de volatilidad "
        "(misma razon que `vix_zscore`, #153)",
    ),
    "garch_forecast_z": (
        1,
        "evitable: usa `ret_log` hasta `t-1`, pero vive en la familia de regimen (misma razon, "
        "#153)",
    ),
    "is_es_roll_session": (
        1,
        "evitable: es el calendario (`de antemano`), pero vive en la familia de regimen "
        "(misma razon, #153)",
    ),
    "dist_sma_20_z": (
        1,
        "forzado: su fila `t` necesita el cierre del ancla de `t`, que es a las 16:00 ET",
    ),
    "ust_10y_chg_5": (
        1,
        "forzado: usa el `ust_10y` de `t`, y el corte *point-in-time* de la familia macro es el "
        "`as_of` de la fila (16:00 ET): usarla antes seria look-ahead",
    ),
}

#: La declaracion de #147, antes de #150: todo `1` salvo el overnight asiatico.
BEFORE: Final[dict[str, int]] = {
    **dict.fromkeys(BASELINE_FEATURES, 1),
    "asia_overnight_1": 0,
}


# ─────────────────────────────────────────────────────────────────────────────
# C5 - la auditoria, declarada columna a columna
# ─────────────────────────────────────────────────────────────────────────────
def test_c5_the_audit_is_declared_column_by_column() -> None:
    """Cada feature de #24 declara su rezago y su motivo; el catalogo dice lo mismo que la tabla."""
    declared = feature_store.DESIGN_LAG_BY_FEATURE
    assert set(AUDIT) == set(BASELINE_FEATURES)
    for name, (lag, reason) in AUDIT.items():
        assert declared[name] == lag, name
        assert len(reason) > 40, name
    assert {name for name, (lag, _) in AUDIT.items() if lag == 0} == set(MOVED_COLUMNS) | {
        "asia_overnight_1"
    }


def test_c5_the_avoidable_gaps_are_tracked_and_the_forced_ones_are_not() -> None:
    """La auditoria separa lo **evitable** de lo **forzado**, y el seguimiento existe."""
    evitable = (AUDIT["vix_zscore"][1], AUDIT["har_forecast"][1], AUDIT["garch_forecast_z"][1])
    assert all("#153" in reason for reason in evitable)
    assert all(AUDIT[name][0] == 1 for name in ("vix_zscore", "har_forecast", "garch_forecast_z"))
    assert all(AUDIT[name][0] == 1 for name in ("dist_sma_20_z", "ust_10y_chg_5"))
    assert "forzado" in AUDIT["dist_sma_20_z"][1]
    assert "forzado" in AUDIT["ust_10y_chg_5"][1]


# ─────────────────────────────────────────────────────────────────────────────
# C3 / C6 - cada columna sale de la sesion que declara, y las demas no se mueven
# ─────────────────────────────────────────────────────────────────────────────
def test_c3_the_design_of_t_reads_where_each_availability_says() -> None:
    """La fila de `t` lleva el valor de `t` en las cuatro publicadas y el de `t-1` en el resto."""
    features, labels = _synthetic()
    design = design_frame(features, labels=labels, availability=feature_store.DESIGN_LAG_BY_FEATURE)
    days = [cast("date", value) for value in features["session"].to_list()]
    declared = feature_store.DESIGN_LAG_BY_FEATURE
    for index in range(1, len(days)):
        row = _row(design.frame, days[index])
        for position, name in enumerate(BASELINE_FEATURES):
            lag = declared[name]
            expected = float((index - lag) * 10 + position)
            assert row[name] == expected, f"{name} (design_lag={lag}) no viene de `t-{lag}`"
        assert row[DESIGN_SESSION_COLUMN] == days[index - 1]


def test_c6_the_columns_without_avoidable_lag_keep_their_value() -> None:
    """Solo se mueven las tres columnas cuyo input ya estaba publicado; el resto, byte a byte."""
    features, labels = _synthetic()
    before = design_frame(features, labels=labels, availability=BEFORE)
    after = design_frame(features, labels=labels, availability=feature_store.DESIGN_LAG_BY_FEATURE)
    untouched = [name for name in BASELINE_FEATURES if name not in MOVED_COLUMNS]
    assert after.frame.select("session", "y", *untouched).equals(
        before.frame.select("session", "y", *untouched)
    )
    for name in MOVED_COLUMNS:
        assert after.frame[name].to_list() != before.frame[name].to_list(), name
    assert after.design_lag_by_feature != before.design_lag_by_feature


# ─────────────────────────────────────────────────────────────────────────────
# C4 - sin look-ahead: el cierre de la sesion que se decide no entra
# ─────────────────────────────────────────────────────────────────────────────
def _europe(extra: Mapping[date, float] | None = None) -> dict[str, pl.DataFrame]:
    """Los tres indices europeos, con las sesiones que se anadan (para el caso de look-ahead)."""
    days = [date(2026, 10, 6), date(2026, 10, 7), date(2026, 10, 8)]
    frames: dict[str, pl.DataFrame] = {}
    for position, name in enumerate(("^GDAXI", "^FTSE", "^STOXX50E")):
        sessions = list(days)
        closes = [100.0 + position, 101.0 + position, 102.0 + position]
        for session, close in (extra or {}).items():
            sessions.append(session)
            closes.append(close)
        frames[name] = pl.DataFrame({"session": sessions, "close": closes})
    return frames


def test_c4_the_close_of_the_session_being_decided_does_not_enter() -> None:
    """El cierre europeo de `t` es a las 11:30 ET, **despues** del snapshot: no puede entrar.

    Se anade la barra que seria el look-ahead (la del 9) y otra posterior (la del 12): el valor
    leido en `t` no cambia, porque la formula es «su ultima sesion **< t**».
    """
    at_snapshot = context_features.pending_context_values(
        _europe(), session=GOLDEN_SESSION, columns=("europe_prev_1",)
    )
    with_look_ahead = context_features.pending_context_values(
        _europe({GOLDEN_SESSION: 10.0, date(2026, 10, 12): 5.0}),
        session=GOLDEN_SESSION,
        columns=("europe_prev_1",),
    )
    assert at_snapshot == with_look_ahead
    assert at_snapshot["europe_prev_1"] is not None


def test_c4_the_moved_columns_are_never_read_from_a_later_session() -> None:
    """Ninguna de las tres lee una sesion posterior al instantaneo (el techo es `< t`)."""
    days = [date(2026, 10, 6), date(2026, 10, 7), date(2026, 10, 8)]
    plain = {
        name: pl.DataFrame({"session": days, "close": [100.0, 101.0, 102.0]})
        for name in ("^GDAXI", "^FTSE", "^STOXX50E", "DX-Y.NYB")
    }
    shocked = {
        **plain,
        **{
            name: pl.DataFrame(
                {"session": [*days, date(2026, 10, 9)], "close": [100.0, 101.0, 102.0, 0.01]}
            )
            for name in ("^GDAXI", "^FTSE", "^STOXX50E", "DX-Y.NYB")
        },
    }
    columns = ("europe_prev_1", "dxy_ret_1")
    assert context_features.pending_context_values(
        plain, session=GOLDEN_SESSION, columns=columns
    ) == context_features.pending_context_values(shocked, session=GOLDEN_SESSION, columns=columns)


# ─────────────────────────────────────────────────────────────────────────────
# C2 - la regla es la de #147, sin un segundo mecanismo
# ─────────────────────────────────────────────────────────────────────────────
def test_c2_every_declared_zero_has_a_pending_reader_and_only_context_columns_do() -> None:
    """Una columna con `design_lag = 0` **tiene** lector pendiente, y el lector solo lee esas.

    Es el cruce que impide que el catalogo diga «disponible en `t`» y el camino diario no sepa
    calcularla (o al reves). Solo puede declararse `0` una columna de **contexto**: las demas
    familias se anclan al calendario del S&P y su fila `t` no es publicable (#153).
    """
    declared = feature_store.DESIGN_LAG_BY_FEATURE
    zero = {name for name, lag in declared.items() if lag == 0}
    assert zero == set(context_features.PENDING_COLUMNS)
    assert zero <= set(feature_store.CONTEXT_FEATURE_COLUMNS)
    formulas = context_features._PENDING_FORMULAS  # pyright: ignore[reportPrivateUsage]
    assert set(formulas) == zero


# ─────────────────────────────────────────────────────────────────────────────
# C3 / C7 - el caso 2026-10-09 y la ausencia de *train/serve skew*, sobre el almacen real
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_c3_the_golden_case_of_2026_10_09_reads_the_close_of_the_8th(
    real_frame: ff.FeatureFrame,
) -> None:
    """El caso de la issue, medido: la decision del 9-oct lee el cierre **del 8**, no el del 7.

    Los dos literales son valores del caso: el que usaba el modelo (la fila del 8 de la matriz, que
    es lo que leia el desplazamiento uniforme de #24 —Europa del 7) y el que ya estaba publicado a
    las 08:45 ET del 9 (la fila del 9, que es la que sirve la lectura pendiente). El `-0.01211` de
    `europe_prev_1` es el que la issue cita, con su SHAP de `-0.1252`.
    """
    matrix = real_frame.matrix.frame
    for name, (stale, fresh) in GOLDEN.items():
        assert float(cast("float", _row(matrix, GOLDEN_PREVIOUS)[name])) == pytest.approx(
            stale, abs=1e-15
        ), name
        assert float(cast("float", _row(matrix, GOLDEN_SESSION)[name])) == pytest.approx(
            fresh, abs=1e-15
        ), name
        served = pending_session_values(Store(REAL_DATA), session=GOLDEN_SESSION, columns=(name,))
        assert served[name] == pytest.approx(fresh, abs=1e-15), name
        assert served[name] != pytest.approx(stale, abs=1e-15), name


@needs_store
def test_c7_the_served_vector_matches_the_design_for_the_moved_columns(
    real_frame: ff.FeatureFrame,
) -> None:
    """El vector servido y la fila de diseno de `t` coinciden en las tres columnas (C7).

    Es el cierre del argumento de #147 aplicado a #150: la regla es **una**, y la lee tanto el
    diseno como el camino diario. Ademas, el valor servido **no** es el de la fila anterior: si
    fuera el mismo, el desfase de dos sesiones seguiria ahi.
    """
    store = Store(REAL_DATA)
    design = real_frame.design
    session = design.sessions[-1]
    reference = _row(design.frame, session)
    previous = cast("date", reference[DESIGN_SESSION_COLUMN])
    row = _row(real_frame.matrix.frame, previous)
    served = run_daily._served_vector(  # pyright: ignore[reportPrivateUsage]
        store, session=session, row=row
    )
    for name in MOVED_COLUMNS:
        assert float(cast("float", served[name])) == float(cast("float", reference[name])), name
        assert served[name] != row[name], f"{name} sigue llegando de la fila anterior"


# ─────────────────────────────────────────────────────────────────────────────
# C8 - determinismo de la lectura pendiente
# ─────────────────────────────────────────────────────────────────────────────
CHILD: Final[str] = textwrap.dedent(
    """
    import hashlib, json
    from datetime import date
    import polars as pl
    from cfdtrader.features import context as context_features

    SECTORS = ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")

    def noise(seed, index):
        digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
        return 100.0 + int.from_bytes(digest[:4], "big") / (2 ** 24)

    days = [date(2026, 10, day) for day in range(1, 10)]
    frames = {
        name: pl.DataFrame({"session": days, "close": [noise(name, i) for i in range(len(days))]})
        for name in ("^N225", "^HSI", "^GDAXI", "^FTSE", "^STOXX50E", "DX-Y.NYB", *SECTORS)
    }
    values = context_features.pending_context_values(
        frames, session=date(2026, 10, 9), columns=sorted(context_features.PENDING_COLUMNS)
    )
    print(json.dumps({name: repr(value) for name, value in values.items()}, sort_keys=True))
    """
)


def _child_pending(hashseed: str) -> dict[str, object]:
    """Las cuatro columnas pendientes calculadas en un proceso fresco con esa semilla de `hash`."""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", CHILD],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": hashseed},
        cwd=str(REPO_ROOT),
    )
    return cast("dict[str, object]", json.loads(completed.stdout))


def test_c8_two_processes_agree_on_the_pending_read() -> None:
    """La lectura pendiente es pura: dos procesos con `PYTHONHASHSEED` distinto dan lo mismo."""
    first = _child_pending("0")
    second = _child_pending("1")
    assert first == second
    assert set(first) == set(MOVED_COLUMNS) | {"asia_overnight_1"}
    assert all(value != "None" for value in first.values())
