"""Tests de #147: `asia_overnight_1` usa el Asia de `t`, con disponibilidad declarada por feature.

Cubre los criterios de la issue:

- **C2** — la disponibilidad (`design_lag`) esta declarada en el catalogo y `design_frame` la
  consume; una columna sin declaracion es error tipado, nunca un rezago por defecto.
- **C3** — el diseno de `t` lee el overnight asiatico de `t` (su ultima sesion `<= t`), no el de
  `t-1`, con el caso 2026-10-09 medido sobre el almacen real.
- **C4** — sin *look-ahead*: una barra posterior a `t` no entra, y una columna no declarada como
  pendiente no se sirve.
- **C5** — las otras 9 features de `BASELINE_FEATURES` no cambian de valor.
- **C6** — una sola regla, dos sitios: el vector **servido** por `run_daily` y la fila de **diseno**
  de la misma sesion coinciden (sin *train/serve skew*).
- **C7** — determinismo: el orden del mapa de disponibilidad no cambia el diseno y dos procesos con
  distinto `PYTHONHASHSEED` dan el mismo resultado.
- **C11** — el informe del pipeline publica el `design_lag` efectivo por columna.

Los casos de almacen se saltan con un `skipif` si el `data/` real no esta en el arbol (en solo
lectura, como el resto de la suite: `tests/conftest.py` huella `data/` y `runs/`).
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
from cfdtrader.analysis import pipeline_report
from cfdtrader.analysis.feature_frame import build_feature_frame
from cfdtrader.data.store import Store
from cfdtrader.delivery import run_daily
from cfdtrader.features import context as context_features
from cfdtrader.features import store as feature_store
from cfdtrader.features.store import ContextInputError
from cfdtrader.models.baseline import (
    BASELINE_FEATURES,
    DESIGN_LAG_SESSIONS,
    DESIGN_SESSION_COLUMN,
    InvalidDesignFrameError,
    SplitAssignment,
    UndeclaredAvailabilityError,
    design_frame,
    fit_baseline,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"

#: La sesion del caso que la issue mide a mano (el snapshot del 2026-10-09 leyo el Asia del 8).
GOLDEN_SESSION: Final[date] = date(2026, 10, 9)
GOLDEN_PREVIOUS: Final[date] = date(2026, 10, 8)

#: El catalogo declara la disponibilidad de **todas** sus columnas, y hoy solo el overnight
#: asiatico es legible en el instantaneo de decision (`design_lag = 0`). El literal es estable: es
#: el contrato de #147, no un digest de un artefacto regenerable.
ZERO_LAG_COLUMNS: Final[tuple[str, ...]] = ("asia_overnight_1",)

needs_store = pytest.mark.skipif(
    not (REAL_DATA / "derived" / "labels").exists(),
    reason="el almacen real no esta en el arbol: C3/C6 se miden sobre el (en solo lectura)",
)


@pytest.fixture(scope="session")
def real_frame() -> ff.FeatureFrame:
    """La matriz + el diseno reales, construidos **una vez** para toda la sesion."""
    return build_feature_frame(Store(REAL_DATA))


def _synthetic(rows: int = 6) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Features (las 10) y etiquetas con valores que dicen **de que sesion** viene cada celda.

    Cada columna vale ``posicion_en_la_sesion * 10 + posicion_de_la_columna``: asi la fila de
    diseno de `t` se puede leer a mano sin ambiguedad (¿viene de `t` o de `t-1`?).
    """
    days = [date(2024, 1, 2) + timedelta(days=index) for index in range(rows)]
    data: dict[str, list[object]] = {"session": list(days)}
    for position, name in enumerate(BASELINE_FEATURES):
        data[name] = [float(index * 10 + position) for index in range(rows)]
    labels = pl.DataFrame(
        {"session": days[1:], "ret_long": [0.01 * (index % 3 - 1) for index in range(1, rows)]}
    )
    return pl.DataFrame(data), labels


def _row(frame: pl.DataFrame, session: date) -> Mapping[str, object]:
    """La fila de esa sesion, como `Mapping` (o un fallo claro si no esta)."""
    selected = frame.filter(pl.col("session") == session)
    assert selected.height == 1, f"la sesion {session} no esta exactamente una vez"
    return selected.row(0, named=True)


# ─────────────────────────────────────────────────────────────────────────────
# C2 - la disponibilidad declarada en el catalogo
# ─────────────────────────────────────────────────────────────────────────────
def test_c2_every_column_declares_its_availability() -> None:
    """Las 57 columnas del catalogo declaran `design_lag`, y solo el Asia es de la sesion `t`."""
    declared = feature_store.DESIGN_LAG_BY_FEATURE
    assert set(declared) == set(feature_store.ALL_FEATURE_COLUMNS)
    assert sorted({name for name, lag in declared.items() if lag == 0}) == list(ZERO_LAG_COLUMNS)
    assert set(declared.values()) == {0, 1}


def test_c2_two_catalogs_declaring_the_same_name_cannot_disagree() -> None:
    """``atr_norm`` vive en dos familias (#72): el mapa exige el **mismo** rezago en las dos."""
    for catalog in feature_store.CATALOG_BY_FEATURE_SET.values():
        for entry in catalog:
            assert entry.design_lag == feature_store.DESIGN_LAG_BY_FEATURE[entry.name]


def test_c2_the_pending_columns_match_the_declared_zero_lag() -> None:
    """Las columnas legibles en el instantaneo y las declaradas con rezago 0 son las mismas.

    Es el cruce que impide que el catalogo diga «disponible en `t`» y el codigo no sepa calcularla
    (o al reves): `PENDING_COLUMNS` es la formula y `design_lag`, la declaracion.
    """
    assert tuple(sorted(context_features.PENDING_COLUMNS)) == ZERO_LAG_COLUMNS
    for column, series in context_features.PENDING_COLUMNS.items():
        assert column in feature_store.CONTEXT_FEATURE_COLUMNS
        assert series
        assert all(name in feature_store.CONTEXT_MARKET_SERIES for name in series)


def test_c2_a_column_without_a_declaration_is_a_typed_error() -> None:
    """Sin declaracion no hay rezago por defecto: `UndeclaredAvailabilityError`."""
    features, labels = _synthetic()
    availability = dict(feature_store.DESIGN_LAG_BY_FEATURE)
    availability.pop("asia_overnight_1")
    with pytest.raises(UndeclaredAvailabilityError):
        design_frame(features, labels=labels, availability=availability)


def test_c2_an_impossible_declaration_is_a_typed_error() -> None:
    """Un rezago negativo (leeria una sesion futura) o no entero no describe ninguna sesion."""
    features, labels = _synthetic()
    availability: dict[str, object] = dict(feature_store.DESIGN_LAG_BY_FEATURE)
    availability["asia_overnight_1"] = -1
    with pytest.raises(InvalidDesignFrameError):
        design_frame(features, labels=labels, availability=availability)  # type: ignore[arg-type]
    availability["asia_overnight_1"] = 1.0
    with pytest.raises(InvalidDesignFrameError):
        design_frame(features, labels=labels, availability=availability)  # type: ignore[arg-type]


def test_c2_a_non_mapping_availability_is_a_typed_error() -> None:
    """El mapa es un `Mapping`: cualquier otra cosa es error tipado, no `AttributeError`."""
    features, labels = _synthetic()
    with pytest.raises(InvalidDesignFrameError):
        design_frame(features, labels=labels, availability=[1, 1])  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# C3 / C5 - el Asia llega de `t` y nada mas se mueve
# ─────────────────────────────────────────────────────────────────────────────
def test_c3_the_design_of_t_reads_the_asia_of_t_and_the_rest_of_t_minus_one() -> None:
    """La fila de `t` lleva el overnight de `t` y las otras 9 columnas de `t-1` (C3 + C5)."""
    features, labels = _synthetic()
    design = design_frame(features, labels=labels, availability=feature_store.DESIGN_LAG_BY_FEATURE)
    days = [cast("date", value) for value in features["session"].to_list()]
    asia = BASELINE_FEATURES.index("asia_overnight_1")
    for index in range(1, len(days)):
        row = _row(design.frame, days[index])
        assert row["asia_overnight_1"] == float(index * 10 + asia), "el Asia no es el de `t`"
        for position, name in enumerate(BASELINE_FEATURES):
            if name == "asia_overnight_1":
                continue
            assert row[name] == float((index - 1) * 10 + position), f"{name} no es el de `t-1`"
        assert row[DESIGN_SESSION_COLUMN] == days[index - 1]

    assert design.design_lag_by_feature == tuple(
        (name, 0 if name == "asia_overnight_1" else 1) for name in BASELINE_FEATURES
    )
    assert design.design_lag_sessions == DESIGN_LAG_SESSIONS == 1


def test_c5_the_other_nine_columns_are_byte_identical_to_the_previous_rule() -> None:
    """Solo el Asia se mueve: las otras 9 columnas son **las mismas** que con el rezago uniforme."""
    features, labels = _synthetic(rows=12)
    uniform = dict.fromkeys(BASELINE_FEATURES, DESIGN_LAG_SESSIONS)
    declared = dict(feature_store.DESIGN_LAG_BY_FEATURE)
    before = design_frame(features, labels=labels, availability=uniform)
    after = design_frame(features, labels=labels, availability=declared)

    others = [name for name in BASELINE_FEATURES if name != "asia_overnight_1"]
    assert after.frame.select("session", "y", *others).equals(
        before.frame.select("session", "y", *others)
    )
    assert after.frame["asia_overnight_1"].to_list() != before.frame["asia_overnight_1"].to_list()
    assert after.design_lag_by_feature != before.design_lag_by_feature


# ─────────────────────────────────────────────────────────────────────────────
# C4 - sin look-ahead
# ─────────────────────────────────────────────────────────────────────────────
def test_c4_the_pending_read_ignores_a_later_bar() -> None:
    """Una barra **posterior** a `t` no cambia el valor leido en `t` (el techo es `<= t`)."""
    days = [date(2026, 10, 8), date(2026, 10, 9)]
    nikkei = pl.DataFrame({"session": days, "close": [40_000.0, 41_000.0]})
    hang_seng = pl.DataFrame({"session": days, "close": [25_000.0, 25_500.0]})
    at_t = context_features.pending_context_values(
        {"^N225": nikkei, "^HSI": hang_seng},
        session=GOLDEN_SESSION,
        columns=["asia_overnight_1"],
    )
    later = pl.DataFrame(
        {
            "session": [*days, date(2026, 10, 12)],
            "close": [40_000.0, 41_000.0, 10.0],
        }
    )
    with_later = context_features.pending_context_values(
        {"^N225": later, "^HSI": hang_seng},
        session=GOLDEN_SESSION,
        columns=["asia_overnight_1"],
    )
    assert at_t == with_later
    assert at_t["asia_overnight_1"] is not None


def test_c4_a_column_that_is_not_readable_at_the_snapshot_is_rejected() -> None:
    """Solo se sirve lo declarado pendiente: `europe_prev_1` no lo es (#150 lo declara)."""
    days = [date(2026, 10, 8), date(2026, 10, 9)]
    frames = {
        "^N225": pl.DataFrame({"session": days, "close": [40_000.0, 41_000.0]}),
        "^HSI": pl.DataFrame({"session": days, "close": [25_000.0, 25_500.0]}),
    }
    with pytest.raises(ContextInputError):
        context_features.pending_context_values(
            frames, session=GOLDEN_SESSION, columns=["europe_prev_1"]
        )
    with pytest.raises(ContextInputError):
        context_features.pending_context_values(frames, session=GOLDEN_SESSION, columns=[])
    with pytest.raises(ContextInputError):
        context_features.pending_context_values(
            {"^N225": frames["^N225"]}, session=GOLDEN_SESSION, columns=["asia_overnight_1"]
        )


def test_c4_a_session_without_history_publishes_null_and_never_a_zero() -> None:
    """Sin historia no hay valor: ``None``, nunca un cero «porque toca»."""
    days = [date(2026, 10, 8)]
    frames = {
        "^N225": pl.DataFrame({"session": days, "close": [40_000.0]}),
        "^HSI": pl.DataFrame({"session": days, "close": [25_000.0]}),
    }
    assert context_features.pending_context_values(
        frames, session=GOLDEN_PREVIOUS, columns=["asia_overnight_1"]
    ) == {"asia_overnight_1": None}


def test_c4_the_design_never_reads_a_session_after_the_label() -> None:
    """Ninguna celda de la fila de `t` viene de `t+1` ni de mas alla (el rezago no es negativo)."""
    features, labels = _synthetic(rows=8)
    design = design_frame(features, labels=labels, availability=feature_store.DESIGN_LAG_BY_FEATURE)
    days = [cast("date", value) for value in features["session"].to_list()]
    for index, session in enumerate(days[1:], start=1):
        row = _row(design.frame, session)
        for name in BASELINE_FEATURES:
            assert float(cast("float", row[name])) <= float(index * 10 + 9)


# ─────────────────────────────────────────────────────────────────────────────
# C7 - determinismo
# ─────────────────────────────────────────────────────────────────────────────
def test_c7_the_order_of_the_availability_map_does_not_change_the_design() -> None:
    """El mapa declara, no ordena: el mismo diseno con el mapa en orden inverso (C7)."""
    features, labels = _synthetic(rows=10)
    declared = dict(feature_store.DESIGN_LAG_BY_FEATURE)
    reversed_map = dict(reversed(list(declared.items())))
    first = design_frame(features, labels=labels, availability=declared)
    second = design_frame(features, labels=labels, availability=reversed_map)
    assert first.frame.equals(second.frame)
    assert first.design_lag_by_feature == second.design_lag_by_feature


CHILD: Final[str] = textwrap.dedent(
    """
    import hashlib, json
    from datetime import date, timedelta
    import polars as pl
    from cfdtrader.features.store import DESIGN_LAG_BY_FEATURE
    from cfdtrader.models.baseline import BASELINE_FEATURES, design_frame

    def noise(seed, index):
        digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / (2**63) - 1.0

    rows = 40
    days = []
    current = date(2024, 1, 2)
    while len(days) < rows:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    data = {"session": days}
    for name in BASELINE_FEATURES:
        data[name] = [noise(f"a147:{name}", i) for i in range(rows)]
    labels = pl.DataFrame(
        {"session": days[1:], "ret_long": [0.01 * (i % 4 - 1) for i in range(1, rows)]}
    )
    design = design_frame(pl.DataFrame(data), labels=labels, availability=DESIGN_LAG_BY_FEATURE)
    print(json.dumps({
        "lag": [list(pair) for pair in design.design_lag_by_feature],
        "values": [
            [repr(value) for value in row]
            for row in design.frame.select(*BASELINE_FEATURES).iter_rows()
        ],
    }, sort_keys=True))
    """
)


def _child_design(hashseed: str) -> dict[str, object]:
    """El diseno calculado en un proceso fresco con ese ``PYTHONHASHSEED``."""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", CHILD],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": hashseed},
        cwd=str(REPO_ROOT),
    )
    return cast("dict[str, object]", json.loads(completed.stdout))


def test_c7_two_processes_with_different_hash_seeds_agree() -> None:
    """Dos procesos con `PYTHONHASHSEED` distinto dan el **mismo** diseno y la misma auditoria.

    No hay reloj ni estado global: el diseno es una funcion de los datos y del mapa declarado.
    """
    first = _child_design("0")
    second = _child_design("1")
    assert first == second
    assert first["lag"] == [
        [name, 0 if name == "asia_overnight_1" else 1] for name in BASELINE_FEATURES
    ]
    assert len(cast("list[object]", first["values"])) == 39


def _asia_frames(store: Store) -> dict[str, pl.DataFrame]:
    """Los frames de las dos series asiaticas, como los lee el camino diario."""
    frames = ff.load_context_inputs(store)
    return {name: frames[name] for name in context_features.PENDING_COLUMNS["asia_overnight_1"]}


# ─────────────────────────────────────────────────────────────────────────────
# C3 / C6 - el caso 2026-10-09 y la ausencia de *train/serve skew*, sobre el almacen real
# ─────────────────────────────────────────────────────────────────────────────
@needs_store
def test_c3_the_golden_case_of_2026_10_09_reads_the_asia_of_the_9th(
    real_frame: ff.FeatureFrame,
) -> None:
    """El caso de la issue, medido: el snapshot del 9-oct lee el Asia del **9**, no el del **8**.

    Los dos literales son los del enunciado, en retorno logaritmico: el Asia del 8-oct
    (``-0.01434``, el que empujo la decision a `SHORT`) y el del 9-oct (``+0.00879``, el que ya
    estaba publicado y **no** entraba). El primero es la fila del 8 en la matriz; el segundo, la
    fila del 9, que es la que la lectura pendiente sirve en `t = 2026-10-09`.
    """
    matrix = real_frame.matrix.frame
    asia_of_the_8th = float(cast("float", _row(matrix, GOLDEN_PREVIOUS)["asia_overnight_1"]))
    asia_of_the_9th = float(cast("float", _row(matrix, GOLDEN_SESSION)["asia_overnight_1"]))
    assert asia_of_the_8th == pytest.approx(-0.014338482676228757, abs=1e-15)
    assert asia_of_the_9th == pytest.approx(0.008785579646215156, abs=1e-15)

    served = context_features.pending_context_values(
        _asia_frames(Store(REAL_DATA)), session=GOLDEN_SESSION, columns=["asia_overnight_1"]
    )
    assert served["asia_overnight_1"] == pytest.approx(asia_of_the_9th, abs=1e-15)
    assert served["asia_overnight_1"] != pytest.approx(asia_of_the_8th, abs=1e-15)


@needs_store
def test_c6_the_served_vector_is_the_design_row_of_the_same_session(
    real_frame: ff.FeatureFrame,
) -> None:
    """El vector que sirve el camino diario y la fila de diseno de `t` son **el mismo** (C6).

    Es el cierre del argumento: no hay *train/serve skew*. La sesion se elige con etiqueta (la
    ultima del diseno) y el vector se construye como en el camino diario —la fila de la sesion
    anterior para las columnas con `design_lag = 1` y la lectura de la sesion `t` para las de
    `design_lag = 0`— con la **misma** funcion que usa `run_daily`.
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
    assert set(served) == set(BASELINE_FEATURES)
    for name in BASELINE_FEATURES:
        assert float(cast("float", served[name])) == float(cast("float", reference[name])), name
    assert served["asia_overnight_1"] != row["asia_overnight_1"]


# ─────────────────────────────────────────────────────────────────────────────
# C11 - el informe publica el rezago efectivo por columna
# ─────────────────────────────────────────────────────────────────────────────
def test_c11_the_temporal_mapping_declares_the_per_column_rule() -> None:
    """La prosa que publica el informe declara la regla por columna (C11)."""
    prose = pipeline_report.TEMPORAL_MAPPING["features"]
    assert "design_lag" in prose
    assert "asia_overnight_1" in prose
    assert "t-1" in prose


@needs_store
def test_c11_the_model_block_publishes_the_effective_lag_per_column(
    real_frame: ff.FeatureFrame,
) -> None:
    """El bloque `model` del informe publica el `design_lag` **efectivo** de cada columna (C11)."""
    design = real_frame.design
    rows = design.n_sessions
    splits = (
        SplitAssignment(
            index=0, train=tuple(range(0, rows - 20)), test=tuple(range(rows - 20, rows))
        ),
    )
    model = fit_baseline(design, splits=splits)
    payload = pipeline_report._model_payload(  # pyright: ignore[reportPrivateUsage]
        model,
        features=real_frame,
        probabilities={},
        moves={},
        test_sessions=(),
    )
    published = cast("Mapping[str, int]", payload["design_lag_by_feature"])
    assert published == dict(design.design_lag_by_feature)
    assert published["asia_overnight_1"] == 0
    assert payload["design_lag_sessions"] == DESIGN_LAG_SESSIONS
