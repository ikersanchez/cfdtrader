"""Tests del camino diario ``delivery/run_daily.py`` (#110): A1-A14.

Todo se mide sobre un almacen **sintetico** en ``tmp_path`` (los cinco fixtures de features
comprometidos) y un registro de modelos tambien en ``tmp_path``: la sesion de tests **no** toca
el ``data/`` ni el ``runs/`` del repositorio (lo blinda ``tests/conftest.py``), no usa red y no
consulta el reloj.

La prueba de render del estado ``recommendation`` construye el ``GateOutput`` llamando al
**gate real** con un ``SlippageParameter.measured(...)``, sin almacen. La determinismo se
verifica con ``subprocess`` y ``PYTHONHASHSEED`` distinto.

Desde #111 este fichero cubre ademas la eleccion del modelo por ``--variant-id`` sobre un
registro sintetico y la familia LightGBM, puntuada con un **booster real entrenado en el test**
(semilla fija, un hilo): nada de ``runs/`` del repositorio y sin red.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final, cast

import lightgbm
import numpy as np
import polars as pl
import pytest

from cfdtrader.analysis import experiment_log, model_comparison
from cfdtrader.analysis.backtest_report import NOTIONAL_USD
from cfdtrader.analysis.model_comparison import BASELINE_VARIANT_ID, VARIANT_ID
from cfdtrader.analysis.pipeline_report import (
    EXPECTED_MOVE_BASIS,
    GARCH_COLUMN,
    SCENARIO_STOP_SIGMA_MULTIPLE,
    SCENARIO_TARGET_STOP_MULTIPLE,
    scenario_parameters,
)
from cfdtrader.backtest.costs import (
    Side,
    SlippageParameter,
    cost_breakdown,
    declared_cost_model,
    declared_slippage_assumption,
)
from cfdtrader.data.calendar import EASTERN, load_calendar
from cfdtrader.data.store import Store
from cfdtrader.decision.gate import GateOutput, GateStatus, evaluate_gate, gate_sha256
from cfdtrader.delivery import run_daily
from cfdtrader.features import store as feature_store
from cfdtrader.features.store import FEATURE_VERSION_PREFIX
from cfdtrader.journal.decision_log import Journal, build_decision, read_decision
from cfdtrader.models.baseline import BASELINE_FEATURES

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
FIXTURES: Final[Path] = REPO_ROOT / "tests" / "fixtures" / "features"

#: El ancla del estudio, su proxy de volatilidad y el indice dolar.
ANCHOR: Final[str] = "^GSPC"
VIX: Final[str] = "^VIX"
DXY: Final[str] = "DX-Y.NYB"

#: Series de contexto de mercado que **solo** aporta el fixture de contexto.
CONTEXT_MARKET: Final[tuple[str, ...]] = ("^GDAXI", "^FTSE", "^STOXX50E", "^N225", "^HSI")

#: Instantes y raices declarados: nunca del reloj.
FETCHED_AT: Final[datetime] = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
CLOSE_HOUR_UTC: Final[int] = 21
DT_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%SZ"
SOURCE_MARKET: Final[str] = "yfinance"
SOURCE_SECTORS: Final[str] = "stooq"
SOURCE_MACRO: Final[str] = "fred"
SOURCE_LABELS: Final[str] = "cfdtrader.models.labels"

#: La ultima sesion del almacen sintetico y la siguiente (la que se evalua a las 08:45 ET).
SNAPSHOT_SESSION: Final[date] = date(2026, 9, 16)
NEXT_SESSION: Final[date] = date(2026, 9, 17)
STALE_SESSION: Final[date] = date(2026, 9, 18)
AS_OF_NEXT: Final[str] = "2026-09-17T12:00:00+00:00"
AS_OF_STALE: Final[str] = "2026-09-18T12:00:00+00:00"

#: #40: una sesion tres dias despues del almacen (las sesiones 17, 18 y 21) y el inicio de la
#: ausencia larga con la que se siembra la ventana de observacion de la regla 15.
MONDAY_SESSION: Final[date] = date(2026, 9, 21)
AS_OF_MONDAY: Final[str] = "2026-09-21T12:00:00+00:00"
ABSENCE_START: Final[date] = date(2026, 9, 1)

#: Identidad declarada de la corrida sintetica del modelo.
RUN_ID: Final[str] = "1" * 64

#: Commit inyectado por el llamante (el modulo no lee git, #112).
GIT_COMMIT: Final[str] = "5" * 40

#: Guarda de diff (criterio 12): forma sancionada SUBSET + DISJOINT.
BASE_COMMIT: Final[str] = "388ac87"
WRITTEN: Final[frozenset[str]] = frozenset(
    {
        "src/cfdtrader/delivery/run_daily.py",
        "src/cfdtrader/delivery/staleness.py",
        "tests/test_run_daily.py",
        "tests/test_staleness.py",
    }
)
FROZEN: Final[frozenset[str]] = frozenset(
    {
        "src/cfdtrader/delivery/__init__.py",
        "src/cfdtrader/decision/gate.py",
        "src/cfdtrader/analysis/feature_frame.py",
        "src/cfdtrader/analysis/pipeline_report.py",
        "src/cfdtrader/analysis/model_comparison.py",
        "src/cfdtrader/analysis/backtest_report.py",
        "src/cfdtrader/models/baseline.py",
        "src/cfdtrader/models/calibration.py",
        "src/cfdtrader/models/labels.py",
        "src/cfdtrader/models/lightgbm_model.py",
        "src/cfdtrader/backtest/costs.py",
        "src/cfdtrader/backtest/engine.py",
        "src/cfdtrader/backtest/metrics.py",
        "src/cfdtrader/backtest/baselines.py",
        "src/cfdtrader/data/store.py",
        "src/cfdtrader/data/calendar.py",
        "src/cfdtrader/data/settings.py",
    }
)

#: La guardia de calidad de fila del modulo, por su nombre, para la prueba directa.
_ROW_PROBLEM = run_daily._row_problem  # pyright: ignore[reportPrivateUsage]
_PREDICTIONS = run_daily._predictions  # pyright: ignore[reportPrivateUsage]


# ─────────────────────────────────────────────────────────────────────────────
# El almacen sintetico (fixtures de features, `tmp_path`)
# ─────────────────────────────────────────────────────────────────────────────
def _instant(session: date) -> datetime:
    """Cierre de sesion declarado de esa fecha, en UTC (no se consulta el reloj)."""
    return datetime(session.year, session.month, session.day, CLOSE_HOUR_UTC, tzinfo=UTC)


def _bar(
    series_id: str, session: date, source: str, prices: tuple[float, float, float, float]
) -> dict[str, object]:
    """Una barra diaria sintetica con las columnas que lee ``raw.market_daily``."""
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
        "volume": 1000.0,
    }


def _flat(close: float) -> tuple[float, float, float, float]:
    """OHLC de una serie de la que el fixture solo trae el cierre."""
    return (close, close, close, close)


def _anchor() -> pl.DataFrame:
    """OHLC del ancla: el fixture de volatilidad manda y el de regimen rellena lo anterior."""
    columns = ("session", "open", "high", "low", "close")
    primary = pl.read_csv(FIXTURES / "golden_inputs.csv", try_parse_dates=True).select(*columns)
    filler = (
        pl.read_csv(FIXTURES / "regime_golden_inputs.csv", try_parse_dates=True)
        .select(*columns)
        .join(primary.select("session"), on="session", how="anti")
    )
    return pl.concat([filler, primary]).sort("session")


def _records(*, vix_mode: str) -> dict[str, list[dict[str, object]]]:
    """Todos los registros del almacen sintetico. ``vix_mode`` decide el VIX del ancla.

    ``constant`` deja ``vix_zscore`` a ``null`` (varianza 0 ⇒ sin ``std``): es la fila
    incompleta que prueba el estado de calidad de datos.
    """
    anchor = _anchor()
    market: list[dict[str, object]] = [
        _bar(
            ANCHOR,
            cast("date", row["session"]),
            SOURCE_MARKET,
            (
                float(cast("float", row["open"])),
                float(cast("float", row["high"])),
                float(cast("float", row["low"])),
                float(cast("float", row["close"])),
            ),
        )
        for row in anchor.iter_rows(named=True)
    ]
    sessions = cast("list[date]", anchor.get_column("session").to_list())
    for position, session in enumerate(sessions):
        if vix_mode == "constant":
            level = 20.0
        else:
            level = 15.0 + 4.0 * math.sin(position / 11.0) + math.cos(position / 5.0)
        market.append(_bar(VIX, session, SOURCE_MARKET, _flat(level)))

    context = pl.read_csv(
        FIXTURES / "context_golden_inputs.csv", try_parse_dates=True, infer_schema_length=None
    )
    for name in CONTEXT_MARKET:
        for row in context.select("session", name).drop_nulls(name).iter_rows(named=True):
            market.append(
                _bar(name, cast("date", row["session"]), SOURCE_MARKET, _flat(float(row[name])))
            )
    macro_market = pl.read_csv(
        FIXTURES / "macro_golden_market.csv",
        schema_overrides={"session": pl.String, "dxy_as_of": pl.String, "dxy_close": pl.String},
    ).with_columns(
        pl.col("dxy_as_of").str.to_datetime(format=DT_FORMAT, time_zone="UTC"),
        pl.col("dxy_close").cast(pl.Float64),
    )
    for row in macro_market.drop_nulls("dxy_as_of").iter_rows(named=True):
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
                "volume": 1000.0,
            }
        )

    sectors: list[dict[str, object]] = [
        _bar(name, cast("date", row["session"]), SOURCE_SECTORS, _flat(float(row[name])))
        for name in feature_store.CONTEXT_SECTOR_SERIES
        for row in context.select("session", name).drop_nulls(name).iter_rows(named=True)
    ]

    macro_series = pl.read_csv(
        FIXTURES / "macro_golden_series.csv",
        schema_overrides={
            "series_id": pl.String,
            "as_of": pl.String,
            "published_at": pl.String,
            "value": pl.String,
        },
    ).with_columns(
        pl.col("as_of").str.to_date(),
        pl.col("published_at").str.to_datetime(format=DT_FORMAT, time_zone="UTC"),
        pl.col("value").cast(pl.Float64),
    )
    macro: list[dict[str, object]] = [
        {
            "source": SOURCE_MACRO,
            "series_id": str(row["series_id"]),
            "as_of": row["as_of"],
            "fetched_at": FETCHED_AT,
            "published_at": row["published_at"],
            "value": float(cast("float", row["value"])),
        }
        for row in macro_series.iter_rows(named=True)
    ]

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


def _build_store(root: Path, *, vix_mode: str) -> Store:
    """Materializa el almacen sintetico en ``root`` y lo devuelve."""
    handle = Store(root)
    for dataset, records in _records(vix_mode=vix_mode).items():
        handle.append("derived" if dataset == "labels" else "raw", dataset, records)
    return handle


@pytest.fixture(scope="module")
def store_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """El almacen sintetico **completo** (la ultima fila trae las diez features y el GARCH)."""
    root = tmp_path_factory.mktemp("run_daily_store")
    _build_store(root, vix_mode="varying")
    return root


@pytest.fixture(scope="module")
def constant_vix_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Un almacen con ``vix_zscore`` a ``null`` en la ultima fila (calidad de datos)."""
    root = tmp_path_factory.mktemp("run_daily_constant")
    _build_store(root, vix_mode="constant")
    return root


# ─────────────────────────────────────────────────────────────────────────────
# El modelo sintetico
# ─────────────────────────────────────────────────────────────────────────────
def _model_document(*, calibration: dict[str, object] | None) -> dict[str, object]:
    """Un documento de la familia lineal con un ultimo fold y, si se pide, su calibrador."""
    fold: dict[str, object] = {
        "index": 0,
        "coefficients": [0.5] * 10,
        "intercept": 0.25,
        "mean": [1.0] * 10,
        "scale": [2.0] * 10,
        "test_positions": [0],
    }
    if calibration is not None:
        fold["calibration"] = calibration
    return {"model": {"features": list(BASELINE_FEATURES), "folds": [fold]}, "note": ""}


_PLATT: Final[dict[str, object]] = {
    "method": "platt",
    "reason": None,
    "n_fit": 10,
    "n_calibration": 5,
    "n_positives": 3,
    "calibration_positions": [0, 1, 2, 3, 4],
    "purge_sessions": 0,
    "exclusions_are_no_op": True,
    "parameters": {"coef": 2.0, "intercept": -1.0, "mean": 0.0, "scale": 1.0},
}

_NONE: Final[dict[str, object]] = {
    "method": "none",
    "reason": "single_class_calibration",
    "n_fit": 10,
    "n_calibration": 5,
    "n_positives": 0,
    "calibration_positions": [0, 1, 2, 3, 4],
    "purge_sessions": 0,
    "exclusions_are_no_op": True,
    "parameters": None,
}

_FEATURES: Final[dict[str, float]] = dict.fromkeys(BASELINE_FEATURES, 3.0)


def _write_model(root: Path, document: Mapping[str, object], *, run_id: str = RUN_ID) -> Path:
    """Escribe ``root/<run_id>/model.json`` y devuelve la ruta del documento."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "model.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def runs_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Un registro sintetico con una corrida lineal valida (con calibrador de Platt)."""
    root = tmp_path_factory.mktemp("run_daily_runs")
    _write_model(root, _model_document(calibration=_PLATT))
    return root


# ─────────────────────────────────────────────────────────────────────────────
# A1: CLI, `--as-of` obligatorio y sin reloj interno
# ─────────────────────────────────────────────────────────────────────────────
def test_the_cli_requires_as_of_and_has_no_internal_clock(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A1: `--as-of` es obligatorio y con zona; el modulo no consulta el reloj."""
    assert run_daily.main([]) == 2
    assert "as-of" in capsys.readouterr().err

    assert run_daily.main(["--as-of", "2026-09-17T12:00:00", "--model-run", RUN_ID]) == 2
    assert "zona horaria" in capsys.readouterr().err

    assert run_daily.main(["--as-of", "no-es-fecha", "--model-run", RUN_ID]) == 2
    assert "ISO-8601" in capsys.readouterr().err

    assert run_daily.main(["--as-of", AS_OF_NEXT]) == 2
    assert "--model-run" in capsys.readouterr().err

    source = Path(run_daily.__file__).read_text(encoding="utf-8")
    assert re.search(r"datetime\.now|date\.today|time\.time", source) is None
    assert re.search(r"\bimport\s+(yfinance|requests|urllib)\b", source) is None


# ─────────────────────────────────────────────────────────────────────────────
# A2: modelo obligatorio y errores tipados
# ─────────────────────────────────────────────────────────────────────────────
def test_a_missing_model_is_a_typed_error(
    store_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A2: un `model.json` ausente es `MissingModelError`; el CLI publica `estado: error`."""
    code = run_daily.main(
        [
            "--as-of",
            AS_OF_NEXT,
            "--model-run",
            "deadbeef",
            "--journal-root",
            str(tmp_path / "journal"),
            "--git-commit",
            GIT_COMMIT,
            "--data-root",
            str(store_root),
            "--runs-root",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "estado: error" in captured.err
    assert "no hay edge demostrado" in captured.err
    assert captured.out == ""

    with pytest.raises(run_daily.MissingModelError):
        run_daily.predict(tmp_path / "nope.json", _FEATURES)

    lightgbm = _write_model(
        tmp_path,
        cast(
            "Mapping[str, object]",
            {"model": {"library": "lightgbm", "features": list(BASELINE_FEATURES), "folds": []}},
        ),
        run_id="lightgbm",
    )
    with pytest.raises(run_daily.UnsupportedModelError):
        run_daily.predict(lightgbm, _FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# A3: `predict` aplica el ultimo fold y su calibrador
# ─────────────────────────────────────────────────────────────────────────────
def test_predict_applies_the_last_fold_and_its_calibrator(tmp_path: Path) -> None:
    """A3: score del ultimo fold y su calibrador, sin reajustar y determinista."""
    # score = sum(((3 - 1) / 2) * 0.5) + 0.25 = 5.25
    raw = run_daily.sigmoid(5.25)
    expected = 1.0 / (1.0 + math.exp(-(2.0 * 5.25 - 1.0)))

    model = _write_model(tmp_path, _model_document(calibration=_PLATT))
    assert run_daily.predict(model, _FEATURES) == pytest.approx(expected)
    assert run_daily.predict(model, _FEATURES) != pytest.approx(raw)

    uncalibrated = _write_model(tmp_path, _model_document(calibration=_NONE), run_id="uncalibrated")
    assert run_daily.predict(uncalibrated, _FEATURES) == pytest.approx(raw)

    no_block = _write_model(tmp_path, _model_document(calibration=None), run_id="noblock")
    assert run_daily.predict(no_block, _FEATURES) == pytest.approx(raw)

    wrong_features = _write_model(
        tmp_path,
        cast("Mapping[str, object]", {"model": {"features": ["solo_una"], "folds": [{}]}}),
        run_id="wrong",
    )
    with pytest.raises(run_daily.UnsupportedModelError):
        run_daily.predict(wrong_features, _FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# A7: guardia de obsolescencia (§8.4)
# ─────────────────────────────────────────────────────────────────────────────
def test_a_stale_store_has_no_hint(
    store_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A4/A7: si la ultima sesion no es la anterior, `no_recommendation_stale_data` y sin pista."""
    code = run_daily.main(
        [
            "--as-of",
            AS_OF_STALE,
            "--model-run",
            "deadbeef",
            "--journal-root",
            str(tmp_path / "journal"),
            "--git-commit",
            GIT_COMMIT,
            "--data-root",
            str(store_root),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "estado: no_recommendation_stale_data" in captured.out
    assert f"sesion: {STALE_SESSION.isoformat()}" in captured.out
    assert f"snapshot_sesion: {SNAPSHOT_SESSION.isoformat()}" in captured.out
    assert "direccion:" not in captured.out
    assert "no hay edge demostrado" in captured.out
    assert captured.err == ""


def test_a_null_feature_row_is_a_data_quality_state(
    constant_vix_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A7: una fila con un `null` en las diez features es `no_recommendation_data_quality`."""
    code = run_daily.main(
        [
            "--as-of",
            AS_OF_NEXT,
            "--model-run",
            "deadbeef",
            "--journal-root",
            str(tmp_path / "journal"),
            "--git-commit",
            GIT_COMMIT,
            "--data-root",
            str(constant_vix_root),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "estado: no_recommendation_data_quality" in captured.out
    assert "vix_zscore" in captured.out
    assert "direccion:" not in captured.out
    assert "no hay edge demostrado" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# A6/A8: informe de la pista y de los cuatro estados
# ─────────────────────────────────────────────────────────────────────────────
def _gate_output(*, measured: bool, probability: float) -> GateOutput:
    """Una salida del **gate real**: con *slippage* medido o con el supuesto declarado (S1)."""
    slippage = (
        SlippageParameter.measured(
            pct_of_notional=Decimal("0.0001"),
            source="prueba unitaria",
            reason="medicion declarada en la prueba",
        )
        if measured
        else declared_slippage_assumption()
    )
    cost = cost_breakdown(
        model=declared_cost_model(),
        slippage=slippage,
        notional_usd=NOTIONAL_USD,
        side=Side.LONG,
        nights=0,
    )
    params = scenario_parameters(cost_pct=cost.c_declared_pct)
    stop_pct = Decimal("1.5") * SCENARIO_STOP_SIGMA_MULTIPLE
    return evaluate_gate(
        session=NEXT_SESSION,
        as_of=datetime(2026, 9, 17, 12, 0, tzinfo=UTC).astimezone(EASTERN),
        today=NEXT_SESSION,
        calendar=load_calendar(),
        prob_up_calibrated=probability,
        expected_move_pct=Decimal("1.5"),
        expected_move_basis=EXPECTED_MOVE_BASIS,
        cost=cost,
        capital_usd=NOTIONAL_USD,
        snapshot_ok=True,
        stop_pct=stop_pct,
        target_pct=SCENARIO_TARGET_STOP_MULTIPLE * stop_pct,
        fomc_dates=(),
        params=params,
        trades_today=0,
    )


def test_the_report_renders_a_recommendation_and_its_fields() -> None:
    """A6/A8: el informe de una recomendacion trae todas las claves y el `gate_sha256` real."""
    output = _gate_output(measured=True, probability=0.62)
    assert output.status is GateStatus.RECOMMENDATION
    assert output.direction is not None
    assert output.direction.name == "LONG"

    text = run_daily.render(
        status=output.status,
        session=NEXT_SESSION,
        as_of=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
        snapshot_session=SNAPSHOT_SESSION,
        model_source="runs/model.json",
        message="pista evaluada",
        output=output,
    )
    for key in (
        "estado:",
        "sesion:",
        "as_of:",
        "snapshot_sesion:",
        "direccion:",
        "prob_calibrada:",
        "ev_declarado_pct:",
        "ev_neto_pct:",
        "stop_pct:",
        "objetivo_pct:",
        "tier:",
        "gate_sha256:",
        "modelo:",
    ):
        assert key in text
    assert "direccion: LONG" in text
    assert f"gate_sha256: {gate_sha256(output)}" in text
    assert "ev_neto_pct: " in text
    assert "no hay edge demostrado" in text

    declared = _gate_output(measured=False, probability=0.62)
    declared_text = run_daily.render(
        status=declared.status,
        session=NEXT_SESSION,
        as_of=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
        snapshot_session=SNAPSHOT_SESSION,
        model_source="runs/model.json",
        message="supuesto declarado",
        output=declared,
    )
    assert "direccion: NOTHING" in declared_text
    assert "ev_neto_pct: null" in declared_text


def test_the_report_declares_the_honesty_fence_in_every_state() -> None:
    """A9 (#109): los cuatro estados imprimen la valla de honestidad."""
    moment = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
    for status in (
        GateStatus.RECOMMENDATION,
        GateStatus.NO_RECOMMENDATION_STALE_DATA,
        GateStatus.NO_RECOMMENDATION_DATA_QUALITY,
        GateStatus.ERROR,
    ):
        text = run_daily.render(
            status=status,
            session=NEXT_SESSION,
            as_of=moment,
            snapshot_session=SNAPSHOT_SESSION,
            model_source="runs/model.json",
            message="motivo de la prueba",
        )
        assert f"estado: {status.value}" in text
        assert "no hay edge demostrado" in text
        assert "manual" in text
        assert "apoyo a la decision" in text
        assert "S1" in text
        assert "#59" in text
        assert "#60" in text
        assert "OPEN" in text


# ─────────────────────────────────────────────────────────────────────────────
# A10: determinismo entre procesos
# ─────────────────────────────────────────────────────────────────────────────
def test_the_report_is_deterministic_across_hash_seeds(
    store_root: Path, runs_root: Path, tmp_path: Path
) -> None:
    """A10: dos CLI con `PYTHONHASHSEED` distinto imprimen exactamente el mismo texto."""
    outputs: list[str] = []
    digests: list[bytes] = []
    journal_file = tmp_path / "journal" / "decisions" / f"{NEXT_SESSION.isoformat()}.json"
    for seed in ("0", "1"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        completed = subprocess.run(  # noqa: S603 - el ejecutable es el interprete de la sesion
            [
                sys.executable,
                "-m",
                "cfdtrader.delivery.run_daily",
                "--as-of",
                AS_OF_NEXT,
                "--model-run",
                RUN_ID,
                "--journal-root",
                str(tmp_path / "journal"),
                "--git-commit",
                GIT_COMMIT,
                "--data-root",
                str(store_root),
                "--runs-root",
                str(runs_root),
            ],
            capture_output=True,
            text=True,
            env=environment,
            cwd=REPO_ROOT,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout)
        digests.append(journal_file.read_bytes())

    assert outputs[0] == outputs[1]
    assert digests[0] == digests[1]  # A14: el diario es determinista byte a byte
    assert "estado: recommendation" in outputs[0]
    assert "gate_sha256: sha256:" in outputs[0]


# ─────────────────────────────────────────────────────────────────────────────
# A12: guarda de diff (SUBSET + DISJOINT)
# ─────────────────────────────────────────────────────────────────────────────
def test_the_delivery_modules_are_the_only_ones_written() -> None:
    """A12/A13: los cuatro ficheros son los escritos y ninguno del conjunto congelado se toca."""

    def _git(*arguments: str) -> str:
        completed = subprocess.run(  # noqa: S603 - el git del sistema, uso fijo
            ["git", *arguments],  # noqa: S607 - el git del sistema, uso fijo
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        )
        return completed.stdout

    assert _git("status", "--porcelain").strip() == ""
    changed = {
        line.strip() for line in _git("diff", "--name-only", f"{BASE_COMMIT}..HEAD").splitlines()
    }
    assert set(WRITTEN) <= changed
    assert changed.isdisjoint(FROZEN)


# ─────────────────────────────────────────────────────────────────────────────
# Extra: la familia lineal y la guardia de obsolescencia
# ─────────────────────────────────────────────────────────────────────────────
def test_main_emits_the_recommendation_state(
    store_root: Path, runs_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A6: el CLI completa el flujo y publica el estado `recommendation`."""
    code = run_daily.main(
        [
            "--as-of",
            AS_OF_NEXT,
            "--model-run",
            RUN_ID,
            "--journal-root",
            str(tmp_path / "journal"),
            "--git-commit",
            GIT_COMMIT,
            "--data-root",
            str(store_root),
            "--runs-root",
            str(runs_root),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "estado: recommendation" in captured.out
    assert "direccion: NOTHING" in captured.out
    assert "ev_neto_pct: null" in captured.out
    assert "gate_sha256: sha256:" in captured.out
    assert "no hay edge demostrado" in captured.out
    assert captured.err == ""


def test_main_reports_a_configuration_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A1: `--settings` inexistente es un error de configuracion (salida 2)."""
    code = run_daily.main(
        [
            "--as-of",
            AS_OF_NEXT,
            "--model-run",
            RUN_ID,
            "--journal-root",
            str(tmp_path / "journal"),
            "--git-commit",
            GIT_COMMIT,
            "--settings",
            str(tmp_path / "no-existe.yaml"),
        ]
    )
    assert code == 2
    assert "no se puede emitir la pista diaria" in capsys.readouterr().err


def test_main_reports_a_missing_dataset_as_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A1/A7: un almacen sin datasets es un `error` del pipeline (salida 2, sin traceback)."""
    code = run_daily.main(
        [
            "--as-of",
            AS_OF_NEXT,
            "--model-run",
            RUN_ID,
            "--journal-root",
            str(tmp_path / "journal"),
            "--git-commit",
            GIT_COMMIT,
            "--data-root",
            str(tmp_path / "almacen-vacio"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "estado: error" in captured.err
    assert "no hay edge demostrado" in captured.err


def test_row_problem_flags_a_missing_or_non_positive_garch() -> None:
    """A7: sin un `garch_forecast` positivo no hay movimiento, stop ni objetivo."""
    row: dict[str, object] = dict.fromkeys(BASELINE_FEATURES, 1.0)
    row[GARCH_COLUMN] = None
    assert "es null" in cast("str", _ROW_PROBLEM(row))
    row[GARCH_COLUMN] = 0.0
    assert "no es positivo" in cast("str", _ROW_PROBLEM(row))
    row[GARCH_COLUMN] = 1e-6
    assert _ROW_PROBLEM(row) is None


def test_predict_rejects_malformed_documents(tmp_path: Path) -> None:
    """A2/A3: una familia que no es la lineal o un fold incompleto son errores tipados."""
    with pytest.raises(run_daily.UnsupportedModelError):
        run_daily.predict(
            _write_model(
                tmp_path,
                cast("Mapping[str, object]", {"model": {"features": list(BASELINE_FEATURES)}}),
                run_id="sin-folds",
            ),
            _FEATURES,
        )
    with pytest.raises(run_daily.UnsupportedModelError):
        run_daily.predict(
            _write_model(
                tmp_path,
                cast(
                    "Mapping[str, object]",
                    {"model": {"features": list(BASELINE_FEATURES), "folds": []}},
                ),
                run_id="folds-vacios",
            ),
            _FEATURES,
        )
    with pytest.raises(run_daily.UnsupportedModelError):
        run_daily.predict(
            _write_model(
                tmp_path,
                cast(
                    "Mapping[str, object]",
                    {
                        "model": {
                            "features": list(BASELINE_FEATURES),
                            "folds": [
                                {"mean": [0.0] * 10, "coefficients": [0.0] * 10, "intercept": 0.0}
                            ],
                        }
                    },
                ),
                run_id="sin-scale",
            ),
            _FEATURES,
        )
    with pytest.raises(run_daily.UnsupportedModelError):
        run_daily.predict(
            _write_model(
                tmp_path,
                cast(
                    "Mapping[str, object]",
                    {
                        "model": {
                            "features": list(BASELINE_FEATURES),
                            "folds": [
                                {
                                    "mean": [0.0] * 10,
                                    "scale": [1.0] * 10,
                                    "coefficients": [0.0] * 9,
                                    "intercept": 0.0,
                                }
                            ],
                        }
                    },
                ),
                run_id="longitudes",
            ),
            _FEATURES,
        )
    with pytest.raises(run_daily.DeliveryError):
        run_daily.predict(
            _write_model(tmp_path, _model_document(calibration=None), run_id="sin-fila"), {}
        )

    bad_json = tmp_path / "bad-json" / "model.json"
    bad_json.parent.mkdir(parents=True, exist_ok=True)
    bad_json.write_text("{ no es json", encoding="utf-8")
    with pytest.raises(run_daily.UnsupportedModelError):
        run_daily.predict(bad_json, _FEATURES)

    not_a_document = tmp_path / "lista" / "model.json"
    not_a_document.parent.mkdir(parents=True, exist_ok=True)
    not_a_document.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(run_daily.UnsupportedModelError):
        run_daily.predict(not_a_document, _FEATURES)

    without_model = tmp_path / "sin-modelo" / "model.json"
    without_model.parent.mkdir(parents=True, exist_ok=True)
    without_model.write_text("{}", encoding="utf-8")
    with pytest.raises(run_daily.UnsupportedModelError):
        run_daily.predict(without_model, _FEATURES)


def test_the_linear_family_rejects_lightgbm_documents(tmp_path: Path) -> None:
    """A2: un documento con `booster_model` (LightGBM) es `UnsupportedModelError`."""
    booster = _write_model(
        tmp_path,
        cast(
            "Mapping[str, object]",
            {
                "model": {
                    "library": "lightgbm",
                    "features": list(BASELINE_FEATURES),
                    "folds": [{"booster_model": "..."}],
                }
            },
        ),
        run_id="booster",
    )
    with pytest.raises(run_daily.UnsupportedModelError):
        run_daily.predict(booster, _FEATURES)


def test_the_module_exports_are_the_declared_contract() -> None:
    """A1: el modulo publica el contrato pedido."""
    module = importlib.import_module("cfdtrader.delivery.run_daily")
    for name in (
        "DeliveryError",
        "MissingModelError",
        "UnsupportedModelError",
        "predict",
        "render",
        "main",
    ):
        assert hasattr(module, name)
    assert issubclass(run_daily.MissingModelError, run_daily.DeliveryError)
    assert issubclass(run_daily.UnsupportedModelError, run_daily.DeliveryError)


# ─────────────────────────────────────────────────────────────────────────────
# #112: el diario de decisiones en el camino diario (A1-A11)
# ─────────────────────────────────────────────────────────────────────────────
def test_the_cli_requires_a_journal_root_and_a_git_commit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A1: sin `--journal-root` y sin `--git-commit` el CLI sale con 2 y lo dice por stderr."""
    assert run_daily.main(["--as-of", AS_OF_NEXT, "--model-run", RUN_ID]) == 2
    assert "--journal-root" in capsys.readouterr().err

    assert (
        run_daily.main(
            [
                "--as-of",
                AS_OF_NEXT,
                "--model-run",
                RUN_ID,
                "--journal-root",
                str(tmp_path / "journal"),
            ]
        )
        == 2
    )
    assert "--git-commit" in capsys.readouterr().err


def test_the_module_does_not_read_git_or_the_network() -> None:
    """A2: la raiz del diario es el unico destino; el modulo no lanza procesos ni usa red."""
    source = Path(run_daily.__file__).read_text(encoding="utf-8")
    assert re.search(r"subprocess|os\.system|os\.popen", source) is None
    assert re.search(r"import\s+(yfinance|requests|urllib)", source) is None


def test_the_raw_and_calibrated_probabilities_come_from_the_last_fold(tmp_path: Path) -> None:
    """A4: `_predictions` devuelve el `sigmoid(score)` crudo y su calibrador, sin reajustar."""
    model = _write_model(tmp_path, _model_document(calibration=_PLATT))
    raw, calibrated = _PREDICTIONS(model, _FEATURES)
    assert raw == pytest.approx(run_daily.sigmoid(5.25))
    assert calibrated == pytest.approx(1.0 / (1.0 + math.exp(-(2.0 * 5.25 - 1.0))))
    assert raw != pytest.approx(calibrated)


def test_a_recommendation_is_journaled(
    store_root: Path, runs_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A3/A4/A8/A9/A10/A12: la fila lleva estado, versiones, las dos probabilidades y el informe."""
    journal = tmp_path / "journal"
    code = run_daily.main(
        [
            "--as-of",
            AS_OF_NEXT,
            "--model-run",
            RUN_ID,
            "--journal-root",
            str(journal),
            "--git-commit",
            GIT_COMMIT,
            "--data-root",
            str(store_root),
            "--runs-root",
            str(runs_root),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0

    record = read_decision(journal, NEXT_SESSION)
    assert record["trade_date"] == NEXT_SESSION.isoformat()
    assert record["status"] == "recommendation"
    assert record["direction"] is not None
    assert record["model_version"] == RUN_ID
    assert record["git_commit"] == GIT_COMMIT
    assert cast("str", record["features_version"]).startswith(FEATURE_VERSION_PREFIX)
    assert f"prob_calibrada: {record['prob_up_calibrated']!r}" in captured.out
    raw = cast("float", record["prob_up_raw"])
    calibrated = cast("float", record["prob_up_calibrated"])
    assert raw != pytest.approx(calibrated)
    # A4: la calibrada es el transform de Platt sobre la cruda: sigmoid(2 * logit(raw) - 1).
    logit = math.log(raw / (1.0 - raw))
    assert calibrated == pytest.approx(1.0 / (1.0 + math.exp(-(2.0 * logit - 1.0))))
    # A10: el informe se persiste verbatim (el `print` anade un salto de linea al de `render`).
    assert captured.out == cast("str", record["report_text"]) + "\n"
    assert "journal" not in captured.out.lower()  # A12: la salida no gana lineas


def test_the_no_recommendation_states_are_journaled_without_direction(
    store_root: Path, constant_vix_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A5/A6: los estados "no se" se registran con su `status` y `direction = null`."""
    stale = tmp_path / "stale"
    assert (
        run_daily.main(
            [
                "--as-of",
                AS_OF_STALE,
                "--model-run",
                "deadbeef",
                "--journal-root",
                str(stale),
                "--git-commit",
                GIT_COMMIT,
                "--data-root",
                str(store_root),
            ]
        )
        == 0
    )
    capsys.readouterr()
    stale_record = read_decision(stale, STALE_SESSION)
    assert stale_record["status"] == "no_recommendation_stale_data"
    assert stale_record["direction"] is None
    assert stale_record["prob_up_calibrated"] is None

    quality = tmp_path / "quality"
    assert (
        run_daily.main(
            [
                "--as-of",
                AS_OF_NEXT,
                "--model-run",
                "deadbeef",
                "--journal-root",
                str(quality),
                "--git-commit",
                GIT_COMMIT,
                "--data-root",
                str(constant_vix_root),
            ]
        )
        == 0
    )
    capsys.readouterr()
    quality_record = read_decision(quality, NEXT_SESSION)
    assert quality_record["status"] == "no_recommendation_data_quality"
    assert quality_record["direction"] is None


def test_the_error_state_is_journaled_with_a_declared_features_version(
    store_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A7: el estado `error` tambien registra; sin matriz, `features_version` es el centinela."""
    empty = tmp_path / "empty"
    code = run_daily.main(
        [
            "--as-of",
            AS_OF_NEXT,
            "--model-run",
            RUN_ID,
            "--journal-root",
            str(empty),
            "--git-commit",
            GIT_COMMIT,
            "--data-root",
            str(tmp_path / "almacen-vacio"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "estado: error" in captured.err
    record = read_decision(empty, NEXT_SESSION)
    assert record["status"] == "error"
    assert record["features_version"] == run_daily.FEATURES_VERSION_UNAVAILABLE
    assert captured.err == cast("str", record["report_text"]) + "\n"

    missing_model = tmp_path / "missing-model"
    assert (
        run_daily.main(
            [
                "--as-of",
                AS_OF_NEXT,
                "--model-run",
                "deadbeef",
                "--journal-root",
                str(missing_model),
                "--git-commit",
                GIT_COMMIT,
                "--data-root",
                str(store_root),
                "--runs-root",
                str(tmp_path / "runs"),
            ]
        )
        == 2
    )
    capsys.readouterr()
    with_matrix = read_decision(missing_model, NEXT_SESSION)
    assert with_matrix["status"] == "error"
    assert cast("str", with_matrix["features_version"]).startswith(FEATURE_VERSION_PREFIX)


def test_the_journal_is_immutable_across_reruns(
    store_root: Path, runs_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A11: misma entrada -> UNCHANGED; otro contenido -> error tipado, bytes intactos."""
    journal = tmp_path / "journal"
    args = [
        "--as-of",
        AS_OF_NEXT,
        "--model-run",
        RUN_ID,
        "--journal-root",
        str(journal),
        "--git-commit",
        GIT_COMMIT,
        "--data-root",
        str(store_root),
        "--runs-root",
        str(runs_root),
    ]
    assert run_daily.main(args) == 0
    capsys.readouterr()
    path = journal / "decisions" / f"{NEXT_SESSION.isoformat()}.json"
    first = path.read_bytes()

    assert run_daily.main(args) == 0  # la misma identidad con el mismo contenido: UNCHANGED
    capsys.readouterr()
    assert path.read_bytes() == first

    changed = list(args)
    changed[changed.index("--as-of") + 1] = "2026-09-17T13:00:00+00:00"
    assert run_daily.main(changed) == 2
    assert "diario" in capsys.readouterr().err
    assert path.read_bytes() == first


# ─────────────────────────────────────────────────────────────────────────────
# #111: elegir el modelo del registro (`--variant-id`) y puntuar LightGBM (A1-A13)
# ─────────────────────────────────────────────────────────────────────────────
def _fingerprint(root: Path) -> dict[str, str]:
    """Ruta relativa -> sha256 de cada fichero: demuestra que un arbol no cambio (A4, A9, A12)."""
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def booster_text() -> str:
    """Un booster LightGBM **real**, entrenado en el test con semilla fija (A12)."""
    rng = np.random.default_rng(11)
    design = rng.normal(size=(48, len(BASELINE_FEATURES)))
    outcome = (design[:, 0] > 0.0).astype(int)
    booster = lightgbm.train(  # pyright: ignore[reportUnknownMemberType]
        {
            "objective": "binary",
            "num_leaves": 3,
            "min_data_in_leaf": 4,
            "learning_rate": 0.3,
            "num_threads": 1,
            "seed": 7,
            "deterministic": True,
            "verbose": -1,
        },
        lightgbm.Dataset(design, label=outcome),
        num_boost_round=4,
    )
    return str(booster.model_to_string())


def _lightgbm_document(
    *, booster_model: str, calibration: dict[str, object] | None
) -> dict[str, object]:
    """El documento de la familia LightGBM: `library` con nombre/version y un fold con el texto."""
    fold: dict[str, object] = {
        "index": 0,
        "n_train": 24,
        "n_test": 2,
        "train_first_session": "2026-08-03",
        "train_last_session": "2026-08-28",
        "train_positives": 12,
        "train_base_rate": 0.5,
        "n_trees": 4,
        "booster_model": booster_model,
        "test_positions": [0, 1],
        "test_probabilities": [0.4, 0.6],
        "test_scores": [-0.4, 0.4],
    }
    if calibration is not None:
        fold["calibration"] = calibration
    return {
        "model": {
            "library": {"name": "lightgbm", "version": lightgbm.__version__},
            "features": list(BASELINE_FEATURES),
            "hyperparameters": {"objective": "binary", "num_leaves": 3, "seed": 7},
            "seed": 7,
            "design_lag_sessions": 1,
            "decision_threshold": 0.55,
            "folds": [fold],
        },
        "note": "",
    }


def _register(
    root: Path,
    *,
    variant_id: str,
    sharpe: float,
    index: int,
    document: Mapping[str, object],
    n_observations: int = 40,
) -> experiment_log.ExperimentRecord:
    """Registra una variante con `record_experiment` y le escribe su `model.json` (A12)."""
    record = experiment_log.record_experiment(
        runs_root=root,
        config=experiment_log.ExperimentConfig(
            variant_id=variant_id,
            features=BASELINE_FEATURES,
            hyperparameters={"variant": variant_id, "index": index},
            seed=index,
            series_id=ANCHOR,
            window={"kind": "index", "start": 0, "stop": n_observations},
        ),
        result=experiment_log.ExperimentResult(
            sharpe_per_session=sharpe, n_observations=n_observations
        ),
        as_of=FETCHED_AT,
    )
    _write_model(root, document, run_id=record.run_sha256)
    return record


@pytest.fixture(scope="module")
def registry_root(tmp_path_factory: pytest.TempPathFactory, booster_text: str) -> Path:
    """Un registro sintetico: dos lineales del mismo `variant_id` y una de LightGBM (A3, A12)."""
    root = tmp_path_factory.mktemp("run_daily_registry")
    _register(
        root,
        variant_id=BASELINE_VARIANT_ID,
        sharpe=0.05,
        index=0,
        document=_model_document(calibration=_PLATT),
    )
    _register(
        root,
        variant_id=BASELINE_VARIANT_ID,
        sharpe=0.20,
        index=1,
        document=_model_document(calibration=_PLATT),
    )
    _register(
        root,
        variant_id=VARIANT_ID,
        sharpe=0.10,
        index=2,
        document=_lightgbm_document(booster_model=booster_text, calibration=_PLATT),
    )
    return root


def test_the_model_selector_is_exactly_one_of_the_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A1: sin selector -> 2 nombrando los dos; con los dos -> 2 con el motivo; sin traceback."""
    assert run_daily.main(["--as-of", AS_OF_NEXT]) == 2
    missing = capsys.readouterr().err
    assert "Traceback" not in missing
    assert "--model-run" in missing
    assert "--variant-id" in missing

    assert (
        run_daily.main(
            [
                "--as-of",
                AS_OF_NEXT,
                "--model-run",
                RUN_ID,
                "--variant-id",
                BASELINE_VARIANT_ID,
                "--journal-root",
                str(tmp_path / "journal"),
                "--git-commit",
                GIT_COMMIT,
            ]
        )
        == 2
    )
    both = capsys.readouterr().err
    assert "Traceback" not in both
    assert "no los dos" in both


def test_the_registry_is_read_through_the_imported_loader() -> None:
    """A2/D6: el registro lo lee `load_registry` importada y las familias, del comparador."""
    assert run_daily.load_registry is experiment_log.load_registry
    assert run_daily.SUPPORTED_VARIANTS == (BASELINE_VARIANT_ID, VARIANT_ID)
    assert run_daily.SUPPORTED_VARIANTS == model_comparison.FAMILY_ORDER
    assert run_daily.VARIANT_SELECTION_RULE


def test_the_selection_criterion_is_declared_and_deterministic(
    registry_root: Path, tmp_path: Path
) -> None:
    """A3: gana el mayor `sharpe_per_session`; empate -> `run_sha256` menor; determinista."""
    chosen = run_daily.resolve_run(runs_root=registry_root, variant_id=BASELINE_VARIANT_ID)
    assert chosen.variant_id == BASELINE_VARIANT_ID
    assert chosen.sharpe_per_session == pytest.approx(0.20)
    assert chosen == run_daily.resolve_run(runs_root=registry_root, variant_id=BASELINE_VARIANT_ID)

    tie = tmp_path / "tie"
    first = _register(
        tie,
        variant_id=BASELINE_VARIANT_ID,
        sharpe=0.10,
        index=0,
        document=_model_document(calibration=None),
    )
    second = _register(
        tie,
        variant_id=BASELINE_VARIANT_ID,
        sharpe=0.10,
        index=1,
        document=_model_document(calibration=None),
    )
    assert first.run_sha256 != second.run_sha256
    expected = min(first.run_sha256, second.run_sha256)
    tied = run_daily.resolve_run(runs_root=tie, variant_id=BASELINE_VARIANT_ID)
    assert tied.run_sha256 == expected


def test_an_unresolvable_variant_is_a_typed_error_and_touches_nothing(
    store_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A4: registro vacio o sin esa familia -> error tipado, salida 2 y `runs/` intacto."""
    empty = tmp_path / "empty-runs"
    empty.mkdir()
    assert (
        run_daily.main(
            [
                "--as-of",
                AS_OF_NEXT,
                "--variant-id",
                BASELINE_VARIANT_ID,
                "--journal-root",
                str(tmp_path / "journal-empty"),
                "--git-commit",
                GIT_COMMIT,
                "--data-root",
                str(store_root),
                "--runs-root",
                str(empty),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "estado: error" in captured.err
    assert "no se pudo leer el registro" in captured.err
    assert "Traceback" not in captured.err
    assert _fingerprint(empty) == {}
    empty_row = read_decision(tmp_path / "journal-empty", NEXT_SESSION)
    assert empty_row["status"] == "error"
    assert empty_row["model_version"] == BASELINE_VARIANT_ID  # D5: el selector declarado

    only_baseline = tmp_path / "only-baseline"
    _register(
        only_baseline,
        variant_id=BASELINE_VARIANT_ID,
        sharpe=0.10,
        index=0,
        document=_model_document(calibration=None),
    )
    before = _fingerprint(only_baseline)
    assert (
        run_daily.main(
            [
                "--as-of",
                AS_OF_NEXT,
                "--variant-id",
                VARIANT_ID,
                "--journal-root",
                str(tmp_path / "journal-absent"),
                "--git-commit",
                GIT_COMMIT,
                "--data-root",
                str(store_root),
                "--runs-root",
                str(only_baseline),
            ]
        )
        == 2
    )
    absent = capsys.readouterr().err
    assert "no trae ninguna variante" in absent
    assert "Traceback" not in absent
    assert _fingerprint(only_baseline) == before


def test_a_variant_outside_the_two_families_is_a_typed_error(
    store_root: Path, registry_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A5: un `variant_id` que no publican las dos familias -> error tipado, salida 2."""
    with pytest.raises(run_daily.RegistrySelectionError):
        run_daily.resolve_run(runs_root=registry_root, variant_id="random_forest_v9")

    assert (
        run_daily.main(
            [
                "--as-of",
                AS_OF_NEXT,
                "--variant-id",
                "random_forest_v9",
                "--journal-root",
                str(tmp_path / "journal"),
                "--git-commit",
                GIT_COMMIT,
                "--data-root",
                str(store_root),
                "--runs-root",
                str(registry_root),
            ]
        )
        == 2
    )
    err = capsys.readouterr().err
    assert "no es un `variant_id` soportado" in err
    assert "Traceback" not in err


def test_the_family_is_decided_by_the_payload(tmp_path: Path) -> None:
    """A6: sin `library` es lineal; `library.name == lightgbm` es LightGBM; otro, error."""
    linear = _write_model(tmp_path, _model_document(calibration=_PLATT), run_id="linear")
    expected = 1.0 / (1.0 + math.exp(-(2.0 * 5.25 - 1.0)))
    assert _PREDICTIONS(linear, _FEATURES) == (
        pytest.approx(run_daily.sigmoid(5.25)),
        pytest.approx(expected),
    )

    unknown = _write_model(
        tmp_path,
        cast(
            "Mapping[str, object]",
            {
                "model": {
                    "library": {"name": "xgboost"},
                    "features": list(BASELINE_FEATURES),
                    "folds": [{"index": 0}],
                }
            },
        ),
        run_id="unknown-family",
    )
    with pytest.raises(run_daily.UnsupportedModelError):
        _PREDICTIONS(unknown, _FEATURES)

    text_library = _write_model(
        tmp_path,
        cast(
            "Mapping[str, object]",
            {"model": {"library": "lightgbm", "features": list(BASELINE_FEATURES), "folds": []}},
        ),
        run_id="text-library",
    )
    with pytest.raises(run_daily.UnsupportedModelError):
        _PREDICTIONS(text_library, _FEATURES)


def test_the_lightgbm_family_reloads_the_published_booster(
    booster_text: str, tmp_path: Path
) -> None:
    """A7/A9: la cruda se recomputa aparte con `raw_score=True`; el artefacto no cambia."""
    document = _lightgbm_document(booster_model=booster_text, calibration=_PLATT)
    path = _write_model(tmp_path, document, run_id="lightgbm")
    before = path.read_bytes()
    raw, calibrated = _PREDICTIONS(path, _FEATURES)
    assert path.read_bytes() == before  # A9: solo se lee

    model_block = cast("Mapping[str, object]", document["model"])
    fold = cast("Mapping[str, object]", cast("list[object]", model_block["folds"])[0])
    published = cast("str", fold["booster_model"])
    row = np.asarray([[3.0] * len(BASELINE_FEATURES)], dtype=np.float64)
    margin = float(
        np.asarray(lightgbm.Booster(model_str=published).predict(row, raw_score=True)).ravel()[0]
    )
    assert raw == pytest.approx(math.exp(margin) / (1.0 + math.exp(margin)))
    assert calibrated == pytest.approx(1.0 / (1.0 + math.exp(-(2.0 * margin - 1.0))))


def test_the_lightgbm_calibration_is_the_published_one(booster_text: str, tmp_path: Path) -> None:
    """A8: `none` y la ausencia de bloque pasan la cruda; un texto corrupto es error tipado."""
    none_path = _write_model(
        tmp_path,
        _lightgbm_document(booster_model=booster_text, calibration=_NONE),
        run_id="gbm-none",
    )
    raw, calibrated = _PREDICTIONS(none_path, _FEATURES)
    assert calibrated == pytest.approx(raw)
    assert run_daily.predict(none_path, _FEATURES) == pytest.approx(raw)

    bare_path = _write_model(
        tmp_path,
        _lightgbm_document(booster_model=booster_text, calibration=None),
        run_id="gbm-bare",
    )
    bare_raw, bare_calibrated = _PREDICTIONS(bare_path, _FEATURES)
    assert bare_calibrated == pytest.approx(bare_raw)

    broken_path = _write_model(
        tmp_path,
        _lightgbm_document(booster_model="no-es-un-booster", calibration=None),
        run_id="gbm-broken",
    )
    with pytest.raises(run_daily.UnsupportedModelError):
        _PREDICTIONS(broken_path, _FEATURES)

    empty_path = _write_model(
        tmp_path,
        _lightgbm_document(booster_model="   ", calibration=None),
        run_id="gbm-empty",
    )
    with pytest.raises(run_daily.UnsupportedModelError):
        _PREDICTIONS(empty_path, _FEATURES)


def test_the_cli_scores_a_lightgbm_variant_from_the_registry(
    store_root: Path, registry_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A10/A11: `--variant-id` resuelve la corrida, el informe se anota y el diario la registra."""
    journal = tmp_path / "journal"
    before = _fingerprint(registry_root)
    code = run_daily.main(
        [
            "--as-of",
            AS_OF_NEXT,
            "--variant-id",
            VARIANT_ID,
            "--journal-root",
            str(journal),
            "--git-commit",
            GIT_COMMIT,
            "--data-root",
            str(store_root),
            "--runs-root",
            str(registry_root),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert f"(variant_id: {VARIANT_ID})" in captured.out

    entry = run_daily.resolve_run(runs_root=registry_root, variant_id=VARIANT_ID)
    record = read_decision(journal, NEXT_SESSION)
    assert f"estado: {record['status']}" in captured.out
    assert record["model_version"] == entry.run_sha256
    assert record["status"] == "recommendation"
    assert captured.out == cast("str", record["report_text"]) + "\n"
    assert _fingerprint(registry_root) == before

    raw = cast("float", record["prob_up_raw"])
    calibrated = cast("float", record["prob_up_calibrated"])
    logit = math.log(raw / (1.0 - raw))
    assert calibrated == pytest.approx(1.0 / (1.0 + math.exp(-(2.0 * logit - 1.0))))

    # D5/A13: la misma corrida por identidad da las mismas probabilidades (cambia `modelo:`).
    other = tmp_path / "by-identity"
    assert (
        run_daily.main(
            [
                "--as-of",
                AS_OF_NEXT,
                "--model-run",
                entry.run_sha256,
                "--journal-root",
                str(other),
                "--git-commit",
                GIT_COMMIT,
                "--data-root",
                str(store_root),
                "--runs-root",
                str(registry_root),
            ]
        )
        == 0
    )
    capsys.readouterr()
    same = read_decision(other, NEXT_SESSION)
    assert same["status"] == record["status"]
    assert same["prob_up_raw"] == record["prob_up_raw"]
    assert same["prob_up_calibrated"] == record["prob_up_calibrated"]


def test_the_cli_picks_the_highest_sharpe_entry_from_the_registry(
    store_root: Path, registry_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A3/A10: con `--variant-id` el CLI usa la entrada de mayor Sharpe de esa familia."""
    assert (
        run_daily.main(
            [
                "--as-of",
                AS_OF_NEXT,
                "--variant-id",
                BASELINE_VARIANT_ID,
                "--journal-root",
                str(tmp_path / "journal"),
                "--git-commit",
                GIT_COMMIT,
                "--data-root",
                str(store_root),
                "--runs-root",
                str(registry_root),
            ]
        )
        == 0
    )
    capsys.readouterr()
    chosen = run_daily.resolve_run(runs_root=registry_root, variant_id=BASELINE_VARIANT_ID)
    assert chosen.sharpe_per_session == pytest.approx(0.20)
    record = read_decision(tmp_path / "journal", NEXT_SESSION)
    assert record["model_version"] == chosen.run_sha256
    assert record["status"] == "recommendation"


def test_the_lightgbm_fixture_is_a_real_booster_in_a_tmp_registry(
    booster_text: str, registry_root: Path
) -> None:
    """A12: el booster se entrena en el test y el registro sintetico no es el `runs/` del repo."""
    assert lightgbm.Booster(model_str=booster_text)
    assert "tree_sizes" in booster_text
    assert registry_root.name.startswith("run_daily_registry")
    assert not registry_root.is_relative_to(REPO_ROOT / "runs")
    assert not registry_root.is_relative_to(REPO_ROOT / "data")
    names = {path.name for path in registry_root.rglob("*") if path.is_file()}
    assert names <= {"config.json", "result.json", "summary.md", "model.json"}


def test_the_variant_path_is_deterministic_across_hash_seeds(
    store_root: Path, registry_root: Path, tmp_path: Path
) -> None:
    """A13: dos procesos con `PYTHONHASHSEED` distinto dan el mismo informe y el mismo diario."""
    outputs: list[str] = []
    digests: list[bytes] = []
    for seed in ("0", "1"):
        journal = tmp_path / f"journal-{seed}"
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        completed = subprocess.run(  # noqa: S603 - el ejecutable es el interprete de la sesion
            [
                sys.executable,
                "-m",
                "cfdtrader.delivery.run_daily",
                "--as-of",
                AS_OF_NEXT,
                "--variant-id",
                VARIANT_ID,
                "--journal-root",
                str(journal),
                "--git-commit",
                GIT_COMMIT,
                "--data-root",
                str(store_root),
                "--runs-root",
                str(registry_root),
            ],
            capture_output=True,
            text=True,
            env=environment,
            cwd=REPO_ROOT,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout)
        digests.append((journal / "decisions" / f"{NEXT_SESSION.isoformat()}.json").read_bytes())

    assert outputs[0] == outputs[1]
    assert digests[0] == digests[1]


# ─────────────────────────────────────────────────────────────────────────────
# #40: la guardia de obsolescencia (festivo, media sesion, ausencia y observacion)
# ─────────────────────────────────────────────────────────────────────────────
def _write_calendar(
    tmp_path: Path, *, holidays: Sequence[date] = (), half_days: Sequence[date] = ()
) -> Path:
    """Un `calendar.yaml` con las excepciones **declaradas** del propio `CalendarConfig` (#40).

    Es el mecanismo del proyecto para lo que ninguna regla deduce ("cierres por luto nacional o
    medias sesiones anunciadas a la ultima hora"): asi la prueba declara un festivo o una media
    sesion sin tocar `data/calendar.py`, que esta congelado para esta entrega.
    """
    lines: list[str] = []
    for key, days in (("extra_holidays", holidays), ("extra_half_days", half_days)):
        if days:
            lines.append(f"{key}:")
            lines.extend(f"  - {day.isoformat()}" for day in days)
    path = tmp_path / "calendar.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _seed_journal(root: Path, days: Sequence[date]) -> None:
    """Siembra el diario con la capa de #39: una fila por sesion, sin `GateOutput`."""
    for day in days:
        payload = build_decision(
            trade_date=day,
            as_of=datetime(day.year, day.month, day.day, 12, 0, tzinfo=UTC),
            features_version="features_de_prueba",
            model_version="modelo_de_prueba",
            git_commit=GIT_COMMIT,
            report_text="informe sembrado en la prueba (#40)",
            status=GateStatus.NO_RECOMMENDATION_STALE_DATA,
        )
        Journal(root).write("decisions", payload)


def test_a3_a_market_holiday_does_not_run_the_daily_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A3: con el mercado cerrado no se lee el almacen, no hay informe y no hay fila."""
    calendar_path = _write_calendar(tmp_path, holidays=(NEXT_SESSION,))
    journal_root = tmp_path / "journal"
    code = run_daily.main(
        [
            "--as-of",
            AS_OF_NEXT,
            "--model-run",
            RUN_ID,
            "--journal-root",
            str(journal_root),
            "--git-commit",
            GIT_COMMIT,
            "--data-root",
            str(tmp_path / "almacen-que-no-existe"),
            "--calendar",
            str(calendar_path),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert f"sin sesion: {NEXT_SESSION.isoformat()}" in captured.out
    assert "festivo" in captured.out
    for token in ("estado:", "direccion:", "gate_sha256:", "modelo:", "no hay edge demostrado"):
        assert token not in captured.out, token
    assert captured.err == ""
    assert not journal_root.exists()


def test_a5_three_sessions_off_are_stale_and_journaled(
    store_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A5: tres sesiones sin ejecutar son `no_recommendation_stale_data`, con su fila y motivo."""
    journal_root = tmp_path / "journal"
    code = run_daily.main(
        [
            "--as-of",
            AS_OF_MONDAY,
            "--model-run",
            RUN_ID,
            "--journal-root",
            str(journal_root),
            "--git-commit",
            GIT_COMMIT,
            "--data-root",
            str(store_root),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "estado: no_recommendation_stale_data" in captured.out
    assert f"sesion: {MONDAY_SESSION.isoformat()}" in captured.out
    assert "direccion:" not in captured.out
    assert "no hay edge demostrado" in captured.out
    assert captured.err == ""

    record = read_decision(journal_root, MONDAY_SESSION)
    assert record["status"] == "no_recommendation_stale_data"
    assert record["direction"] is None
    # A5: el informe se persiste verbatim (el `print` anade un salto al de `render`).
    assert captured.out == cast("str", record["report_text"]) + "\n"
    assert str(record["features_version"]).startswith(FEATURE_VERSION_PREFIX)


def test_a5_a_long_absence_carries_the_reincorporation_notice(
    store_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A5/A8 (§8.4 fila 3): el "no se" de una ausencia larga lleva su aviso de reincorporacion."""
    journal_root = tmp_path / "journal"
    _seed_journal(journal_root, (ABSENCE_START, date(2026, 9, 2)))
    code = run_daily.main(
        [
            "--as-of",
            AS_OF_MONDAY,
            "--model-run",
            RUN_ID,
            "--journal-root",
            str(journal_root),
            "--git-commit",
            GIT_COMMIT,
            "--data-root",
            str(store_root),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "estado: no_recommendation_stale_data" in captured.out
    assert "direccion:" not in captured.out
    assert "reincorporacion" in captured.out
    assert "5 de 5" in captured.out

    record = read_decision(journal_root, MONDAY_SESSION)
    assert record["status"] == "no_recommendation_stale_data"
    assert "reincorporacion" in cast("str", record["report_text"])


def test_a6_a_half_session_is_a_justified_nothing(
    store_root: Path, runs_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A6: media sesion ⇒ `NOTHING` justificado (regla 18), no un "no se"; y se registra."""
    # El festivo del 17 y la media sesion del 18: la sesion anterior a la evaluada sigue siendo
    # el cierre del 16 que trae el almacen sintetico.
    calendar_path = _write_calendar(tmp_path, holidays=(NEXT_SESSION,), half_days=(STALE_SESSION,))
    journal_root = tmp_path / "journal"
    code = run_daily.main(
        [
            "--as-of",
            AS_OF_STALE,
            "--model-run",
            RUN_ID,
            "--journal-root",
            str(journal_root),
            "--git-commit",
            GIT_COMMIT,
            "--data-root",
            str(store_root),
            "--runs-root",
            str(runs_root),
            "--calendar",
            str(calendar_path),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "estado: recommendation" in captured.out
    assert "direccion: NOTHING" in captured.out
    assert "bloqueo: 18:media_sesion" in captured.out
    assert "no_recommendation" not in captured.out
    assert captured.err == ""

    record = read_decision(journal_root, STALE_SESSION)
    assert record["status"] == "recommendation"
    assert record["direction"] == "nothing"
    assert record["blocking_events"] == ["media_sesion"]
    assert captured.out == cast("str", record["report_text"]) + "\n"


def test_a8_observation_mode_is_not_an_actionable_hint(
    store_root: Path, runs_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A7/A8: la vuelta de una ausencia de mas de una semana deja la pista en observacion."""
    journal_root = tmp_path / "journal"
    _seed_journal(journal_root, (ABSENCE_START, date(2026, 9, 2)))
    code = run_daily.main(
        [
            "--as-of",
            AS_OF_NEXT,
            "--model-run",
            RUN_ID,
            "--journal-root",
            str(journal_root),
            "--git-commit",
            GIT_COMMIT,
            "--data-root",
            str(store_root),
            "--runs-root",
            str(runs_root),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "estado: recommendation" in captured.out
    assert "direccion: NOTHING" in captured.out
    assert "bloqueo: 15:modo_observacion" in captured.out
    assert "reincorporacion" in captured.out
    assert "5 de 5" in captured.out
    assert "no_recommendation" not in captured.out
    assert captured.err == ""

    record = read_decision(journal_root, NEXT_SESSION)
    assert record["status"] == "recommendation"
    assert record["direction"] == "nothing"
    assert record["blocking_events"] == ["modo_observacion"]
    assert captured.out == cast("str", record["report_text"]) + "\n"


def test_a9_the_report_shows_the_blockers_only_when_there_are_any() -> None:
    """A9: una linea `bloqueo:` por bloqueo del gate, y ninguna cuando el gate no bloquea."""
    blocked = _gate_output(measured=False, probability=0.62)
    assert blocked.blockers
    moment = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
    text = run_daily.render(
        status=blocked.status,
        session=NEXT_SESSION,
        as_of=moment,
        snapshot_session=SNAPSHOT_SESSION,
        model_source="runs/model.json",
        message="motivo de la prueba",
        output=blocked,
    )
    expected = [f"bloqueo: {entry['rule']}:{entry['code']}" for entry in blocked.blockers]
    assert [line for line in text.splitlines() if line.startswith("bloqueo:")] == expected

    clean = _gate_output(measured=True, probability=0.62)
    assert clean.blockers == ()
    clean_text = run_daily.render(
        status=clean.status,
        session=NEXT_SESSION,
        as_of=moment,
        snapshot_session=SNAPSHOT_SESSION,
        model_source="runs/model.json",
        message="motivo de la prueba",
        output=clean,
    )
    assert "bloqueo:" not in clean_text
    assert "direccion: LONG" in clean_text


def test_the_guard_paths_are_deterministic_across_hash_seeds(
    store_root: Path, runs_root: Path, tmp_path: Path
) -> None:
    """A11 (#40): el festivo (no escribe) y la media sesion (escribe) son deterministas."""
    holiday_calendar = _write_calendar(tmp_path, holidays=(NEXT_SESSION,))
    half_calendar = _write_calendar(tmp_path, holidays=(NEXT_SESSION,), half_days=(STALE_SESSION,))
    scenarios = (
        (AS_OF_NEXT, holiday_calendar, None),
        (AS_OF_STALE, half_calendar, STALE_SESSION),
    )
    for as_of, calendar_path, written in scenarios:
        outputs: list[str] = []
        digests: list[bytes] = []
        for seed in ("0", "1"):
            journal = tmp_path / f"journal-{seed}-{as_of}"
            environment = {**os.environ, "PYTHONHASHSEED": seed}
            completed = subprocess.run(  # noqa: S603 - el ejecutable es el interprete de la sesion
                [
                    sys.executable,
                    "-m",
                    "cfdtrader.delivery.run_daily",
                    "--as-of",
                    as_of,
                    "--model-run",
                    RUN_ID,
                    "--journal-root",
                    str(journal),
                    "--git-commit",
                    GIT_COMMIT,
                    "--data-root",
                    str(store_root),
                    "--runs-root",
                    str(runs_root),
                    "--calendar",
                    str(calendar_path),
                ],
                capture_output=True,
                text=True,
                env=environment,
                cwd=REPO_ROOT,
                check=False,
            )
            assert completed.returncode == 0, completed.stderr
            outputs.append(completed.stdout)
            if written is None:
                assert not journal.exists(), "el festivo no deja fila en el diario"
            else:
                digests.append((journal / "decisions" / f"{written.isoformat()}.json").read_bytes())

        assert outputs[0] == outputs[1], as_of
        if written is not None:
            assert digests[0] == digests[1], as_of
