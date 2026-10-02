"""Tests del camino diario ``delivery/run_daily.py`` (#110): A1-A14.

Todo se mide sobre un almacen **sintetico** en ``tmp_path`` (los cinco fixtures de features
comprometidos) y un registro de modelos tambien en ``tmp_path``: la sesion de tests **no** toca
el ``data/`` ni el ``runs/`` del repositorio (lo blinda ``tests/conftest.py``), no usa red y no
consulta el reloj.

La prueba de render del estado ``recommendation`` construye el ``GateOutput`` llamando al
**gate real** con un ``SlippageParameter.measured(...)``, sin almacen. La determinismo se
verifica con ``subprocess`` y ``PYTHONHASHSEED`` distinto.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final, cast

import polars as pl
import pytest

from cfdtrader.analysis.backtest_report import NOTIONAL_USD
from cfdtrader.analysis.pipeline_report import (
    EXPECTED_MOVE_BASIS,
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

#: Identidad declarada de la corrida sintetica del modelo.
RUN_ID: Final[str] = "1" * 64

#: Guarda de diff (criterio 12): forma sancionada SUBSET + DISJOINT.
BASE_COMMIT: Final[str] = "388ac87"
WRITTEN: Final[frozenset[str]] = frozenset(
    {"src/cfdtrader/delivery/run_daily.py", "tests/test_run_daily.py"}
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
def test_a_stale_store_has_no_hint(store_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A4/A7: si la ultima sesion no es la anterior, `no_recommendation_stale_data` y sin pista."""
    code = run_daily.main(
        [
            "--as-of",
            AS_OF_STALE,
            "--model-run",
            "deadbeef",
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
    constant_vix_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A7: una fila con un `null` en las diez features es `no_recommendation_data_quality`."""
    code = run_daily.main(
        [
            "--as-of",
            AS_OF_NEXT,
            "--model-run",
            "deadbeef",
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
def test_the_report_is_deterministic_across_hash_seeds(store_root: Path, runs_root: Path) -> None:
    """A10: dos CLI con `PYTHONHASHSEED` distinto imprimen exactamente el mismo texto."""
    outputs: list[str] = []
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

    assert outputs[0] == outputs[1]
    assert "estado: recommendation" in outputs[0]
    assert "gate_sha256: sha256:" in outputs[0]


# ─────────────────────────────────────────────────────────────────────────────
# A12: guarda de diff (SUBSET + DISJOINT)
# ─────────────────────────────────────────────────────────────────────────────
def test_the_delivery_modules_are_the_only_ones_written() -> None:
    """A12: los dos ficheros son los escritos y ninguno del conjunto congelado se toca."""

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
