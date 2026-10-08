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

import ast
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Final, cast

import lightgbm
import numpy as np
import polars as pl
import pytest

from cfdtrader.agents import report as report_agent
from cfdtrader.agents.event_calendar import (
    EventCalendarInputError,
    EventKind,
    calendar_signal,
)
from cfdtrader.agents.news import PROMPT_TEMPLATE_NAME
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
from cfdtrader.data.calendar import EASTERN, MADRID, load_calendar
from cfdtrader.data.earnings import EarningsCertainty, EarningsEvent, EarningsMoment
from cfdtrader.data.macro import MacroPublication, publications_on
from cfdtrader.data.news import ingest as ingest_news
from cfdtrader.data.sources.fred_adapter import MacroSeriesRegistry, MacroSeriesSpec
from cfdtrader.data.sources.news import Headline
from cfdtrader.data.store import Store
from cfdtrader.decision.gate import GateOutput, GateStatus, evaluate_gate, gate_sha256
from cfdtrader.decision.overlay import OverlayDecision, OverlayState, disabled_overlay
from cfdtrader.delivery import run_daily
from cfdtrader.features import store as feature_store
from cfdtrader.features.store import FEATURE_VERSION_PREFIX
from cfdtrader.journal.decision_log import Journal, build_decision, read_decision
from cfdtrader.llm.base import LLMRequest, LLMResponse
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

#: #113: un sabado real del calendario (fin de semana sin excepciones declaradas).
SATURDAY_SESSION: Final[date] = date(2026, 9, 19)
AS_OF_SATURDAY: Final[str] = "2026-09-19T12:00:00+00:00"

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
        # #113 retira `decision/gate.py` y `analysis/pipeline_report.py` de este conjunto: la
        # entrega del dia de mercado cerrado toca el gate (regla 19) y el informe (SESSION_RULES),
        # el mismo criterio que #80 aplico en `tests/test_pipeline_report.py`.
        "src/cfdtrader/analysis/feature_frame.py",
        # #136 retira `analysis/model_comparison.py` de este conjunto: el guardian de reconstruccion
        # (A7) se relativiza —contra el resultado que publica la **propia** entrada, no contra un
        # literal `FROZEN_BASELINE` que caduca con cada ingesta—, el mismo criterio que #113, #124
        # y #131 aplicaron al gate, al calendario y al motor de costes.
        # #108 retira `analysis/backtest_report.py` y `models/labels.py`: el barrido de punteros
        # `#50 -> #107` toca su prosa y sus punteros, el mismo criterio que #113/#124/#131.
        "src/cfdtrader/models/baseline.py",
        "src/cfdtrader/models/calibration.py",
        "src/cfdtrader/models/lightgbm_model.py",
        # #131 retira `backtest/costs.py` de este conjunto: cablear el `R` decidido en #60 en el
        # motor de costes —para que el supuesto de *slippage* tenga numero y el gate pueda
        # verificar la regla 9 sobre el coste declarado (`plan.md` §19.12)— es exactamente ese
        # fichero, el mismo criterio que #113 aplico con el gate y #124 con el calendario.
        # #108 retira `backtest/engine.py`: el barrido de punteros `#50 -> #107` toca su
        # `does_not_do` y su prosa de follow-ups.
        "src/cfdtrader/backtest/metrics.py",
        "src/cfdtrader/backtest/baselines.py",
        "src/cfdtrader/data/store.py",
        # #124 retira `data/calendar.py` de este conjunto: el esquema y el cargador del calendario
        # de FOMC declarado viven en ese modulo, el mismo criterio que #113 aplico con el gate
        # (`decision/gate.py`) y con el informe (`analysis/pipeline_report.py`).
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
    # Este camino sale **antes** de la etapa del calendario, asi que el aviso de FOMC de #124 no
    # llega a imprimirse: aqui el `stderr` sigue vacio.
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
        assert "#62" in text
        assert "#107" in text
        # #137: la valla deja de declarar abierta una decision ya cerrada. Se comprueba sobre
        # la valla, no sobre el informe entero (donde «OPEN» podria aparecer por otro motivo).
        assert "OPEN" not in "\n".join(run_daily.HONESTY_FENCE)


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
    # #131: con el `R` que decidio #60 cableado, la sesion de los fixtures **autoriza** sobre el
    # coste declarado (`plan.md` §19.12), asi que la direccion deja de ser NOTHING y el EV bajo el
    # supuesto se publica como **sensibilidad** en vez de `null`.
    assert "direccion: SHORT" in captured.out
    assert "base_del_ev: declared" in captured.out
    assert "ev_neto_es_sensibilidad: true" in captured.out
    assert "ev_neto_pct: null" not in captured.out
    assert "gate_sha256: sha256:" in captured.out
    assert "no hay edge demostrado" in captured.out
    # 2026 **ya esta declarado** (tarea #132): el aviso **desaparece**. Era
    # `assert "..." in captured.err` mientras el año estaba pendiente, y `err == ""` antes de #124.
    assert "no esta declarado en el calendario de FOMC" not in captured.err


# ─────────────────────────────────────────────────────────────────────────────
# El overlay en el camino diario (#35): A7, A8, A9, A11 y A12
# ─────────────────────────────────────────────────────────────────────────────
JOURNAL_DATE: Final[date] = date(2026, 9, 17)


def _daily_run(store_root: Path, runs_root: Path, journal_root: Path) -> int:
    """Una ejecucion completa del camino diario sobre los fixtures, con su diario propio."""
    return run_daily.main(
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


def _journal_row(journal_root: Path) -> dict[str, object]:
    return Journal(journal_root).read_decision(JOURNAL_DATE)


def test_132_a_declared_year_does_not_warn_and_a_pending_one_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#132/#114: declarado → sin aviso; pendiente → aviso que **nombra** el hueco y su fecha."""
    from cfdtrader.data.calendar import FomcCalendarConfig, PendingYear

    dates, note, declared = run_daily._declared_fomc_dates(2026)  # pyright: ignore[reportPrivateUsage]
    assert len(dates) == 16, "2026 esta declarado con sus ocho reuniones"
    assert declared is True
    assert "pendiente" not in note

    pending = FomcCalendarConfig(
        version=1,
        source="https://example.invalid/fomc",
        verified_on=date(2026, 10, 4),
        tentative_note="tentative",
        pending={
            2030: PendingYear(
                reason="la fuente se corta antes de 2030", attempted_on=date(2026, 10, 4)
            )
        },
    )

    def _pending_config(path: Path | str | None = None) -> FomcCalendarConfig:
        """El calendario pautado: se prueba el **aviso**, no la lectura del fichero."""
        return pending

    monkeypatch.setattr(run_daily, "load_fomc_calendar", _pending_config)

    dates, named, declared = run_daily._declared_fomc_dates(2030)  # pyright: ignore[reportPrivateUsage]
    assert dates == () and declared is False
    assert "no esta declarado en el calendario de FOMC" in named, "la frase de #114 se conserva"
    assert "pendiente" in named and "2026-10-04" in named, "con la fecha del intento"
    assert "se corta antes de 2030" in named, "el motivo declarado viaja al aviso"

    _, generic, _ = run_daily._declared_fomc_dates(2040)  # pyright: ignore[reportPrivateUsage]
    assert "pendiente" not in generic, "un año sin intento declarado no finge que lo hubo"


def test_b7_the_manifest_declares_the_headline_counts(
    store_root: Path,
    runs_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#129: el `manifest` de la sesion declara cuantos titulares vio el lote y cuantos envio.

    Se pauta el lote y se **agota el presupuesto** a proposito: el overlay no llega a llamar, y aun
    asi el conteo tiene que salir entero con `headlines_sent = 0`. Que la capa LLM este caida no es
    motivo para no saber cuantos titulares habia.
    """
    moment = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
    batch = (
        Headline(
            source="rss",
            feed="qa",
            title="La Fed sube los tipos",
            url="https://example.invalid/a",
            published_at=moment,
        ),
        Headline(
            source="rss",
            feed="qa",
            title="La Fed sube los tipos",
            url="https://example.invalid/b",
            published_at=moment,
        ),
        Headline(
            source="rss",
            feed="qa",
            title="Nvidia presenta resultados",
            url="https://example.invalid/c",
            published_at=moment,
        ),
        Headline(
            source="rss",
            feed="qa",
            title="Titular del mes pasado",
            url="https://example.invalid/d",
            published_at=moment - timedelta(days=30),
        ),
    )

    def _pinned_batch(*arguments: object, **keywords: object) -> tuple[Headline, ...]:
        """El lote pautado, sin tocar el almacen: se prueba el **conteo**, no el lector."""
        return batch

    monkeypatch.setattr(run_daily, "load_headlines", _pinned_batch)
    monkeypatch.setenv("LLM_MAX_CALLS_PER_RUN", "0")
    journal_root = tmp_path / "journal"

    assert _daily_run(store_root, runs_root, journal_root) == 0
    capsys.readouterr()

    manifest = json.loads(
        (journal_root / "ops" / NEXT_SESSION.isoformat() / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    counters = manifest["counters"]
    assert counters["headlines_read"] == 4
    assert counters["headlines_duplicates"] == 1, "los dos titulares identicos colapsan"
    assert counters["headlines_out_of_window"] == 1, "el del mes pasado queda fuera"
    assert counters["headlines_prepared"] == 2
    assert counters["headlines_sent"] == 0, "sin llamada no se envio nada, y se declara"
    assert (
        counters["headlines_read"]
        == counters["headlines_duplicates"]
        + counters["headlines_out_of_window"]
        + counters["headlines_prepared"]
    ), "el conteo del lote cierra en el propio manifest"


def _stub_overlay(result: OverlayDecision, hashes: Mapping[str, object]) -> object:
    """Un `_compute_overlay` de mentira: devuelve la decision pautada sin tocar el proveedor.

    Devuelve tambien los conteos del lote (#129) —aqui vacios, porque el doble no lee titulares—,
    que es el tercer elemento que el camino diario espera.
    """

    def _fake(
        *arguments: object, **keywords: object
    ) -> tuple[OverlayDecision, Mapping[str, object], dict[str, int]]:
        return result, hashes, {}

    return _fake


def _no_report(*args: object, **kwargs: object) -> tuple[None, dict[str, str]]:
    """Doble sin red del informe redactado (#37): no-op que nunca redacta."""
    return (None, {})


@pytest.fixture(autouse=True)
def _no_llm_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """La redaccion del informe no abre red en las pruebas: se sustituye por un no-op (#37).

    El camino diario SI consulta el estado antes de llamar, pero la clave del `.env` existe: sin
    esta sustitucion, cada prueba del pipeline pagaria una llamada real. Las pruebas de #37
    sobreescriben este doble con una redaccion pautada.
    """
    monkeypatch.setattr(run_daily, "_compose_report", _no_report)


def test_a7_the_journal_records_what_the_overlay_did(
    store_root: Path,
    runs_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """El diario deja de mentir por omision: `llm_overlay` y `prompt_hashes` poblados (#35 A7)."""
    prompt_hash = "sha256:" + "c" * 64
    monkeypatch.setattr(
        run_daily,
        "_compute_overlay",
        _stub_overlay(
            OverlayDecision(
                state=OverlayState.VETO,
                reasons=("monetary_policy/high: una frase",),
                prompt_hash=prompt_hash,
            ),
            {PROMPT_TEMPLATE_NAME: prompt_hash},
        ),
    )
    journal_root = tmp_path / "journal"

    assert _daily_run(store_root, runs_root, journal_root) == 0
    capsys.readouterr()

    row = _journal_row(journal_root)
    assert row["llm_overlay"] == "veto"
    assert row["prompt_hashes"] == {PROMPT_TEMPLATE_NAME: prompt_hash}
    assert row["direction"] == "nothing", "un veto convierte el dia en NOTHING"
    report = cast("str", row["report_text"])
    assert "bloqueo: 20:overlay_veto" in report, "el veto es trazable como regla 20"


def test_a8_without_news_the_recommendation_still_comes_out(
    store_root: Path,
    runs_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A8: el overlay es opcional por diseno. Sin titulares, la recomendacion sigue saliendo.

    Se declara una config de LLM valida por entorno para que la prueba **no** dependa del `.env`
    del desarrollador; no se llama al proveedor porque no hay titulares que enviar.
    """
    monkeypatch.setenv("LLM_API_KEY", "clave-solo-para-esta-prueba")
    monkeypatch.setenv("LLM_MODEL_EXTRACT", "modelo-falso-de-prueba")
    journal_root = tmp_path / "journal"

    assert _daily_run(store_root, runs_root, journal_root) == 0
    captured = capsys.readouterr()

    assert "estado: recommendation" in captured.out
    row = _journal_row(journal_root)
    assert row["llm_overlay"] == "applied", "sin noticias no hay nada que vetar"
    assert row["prompt_hashes"] == {}, "sin llamada no hay hash de prompt que registrar"


def test_a9_a_spent_budget_never_blocks_the_pipeline(
    store_root: Path,
    runs_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """⭐ El criterio B9 de #33, que #33 no podia cerrar: superar un tope **no** bloquea.

    La recomendacion del dia tiene que ser la **misma** con el overlay aplicado y con el overlay
    desactivado por presupuesto: la decision no depende de que quede dinero.
    """
    monkeypatch.setattr(
        run_daily,
        "_compute_overlay",
        _stub_overlay(OverlayDecision(state=OverlayState.APPLIED), {}),
    )
    baseline = tmp_path / "baseline"
    assert _daily_run(store_root, runs_root, baseline) == 0
    capsys.readouterr()

    monkeypatch.setattr(
        run_daily,
        "_compute_overlay",
        _stub_overlay(
            disabled_overlay(OverlayState.DISABLED_BUDGET, reasons=("llamadas: 0 >= 0",)), {}
        ),
    )
    stopped = tmp_path / "stopped"
    assert _daily_run(store_root, runs_root, stopped) == 0, "el tope no puede abortar la ejecucion"
    capsys.readouterr()

    applied_row = _journal_row(baseline)
    stopped_row = _journal_row(stopped)
    for field in ("status", "direction", "tier", "size_notional_eur", "prob_up_calibrated"):
        assert applied_row[field] == stopped_row[field], (
            f"{field} no puede depender del presupuesto"
        )
    assert applied_row["llm_overlay"] == "applied"
    assert stopped_row["llm_overlay"] == "disabled_budget"


def test_a11_reads_from_store_and_not_from_the_network(
    store_root: Path,
    runs_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A11: el camino diario lee `raw.news_headlines`; no ingesta en linea."""
    seen: list[datetime] = []
    real = run_daily.load_headlines

    def _spy(store: Store, *, as_of: datetime, window_hours: int = 24) -> tuple[object, ...]:
        seen.append(as_of)
        return real(store, as_of=as_of, window_hours=window_hours)

    monkeypatch.setattr(run_daily, "load_headlines", _spy)
    monkeypatch.setenv("LLM_API_KEY", "clave-solo-para-esta-prueba")
    monkeypatch.setenv("LLM_MODEL_EXTRACT", "modelo-falso-de-prueba")

    assert _daily_run(store_root, runs_root, tmp_path / "journal") == 0
    capsys.readouterr()

    assert len(seen) == 1, "los titulares se leen del almacen, una vez, con el instante declarado"
    assert seen[0] == datetime(2026, 9, 17, 12, 0, tzinfo=UTC)


def test_a12_the_daily_path_uses_the_public_bridge() -> None:
    """A12: el camino diario usa el puente declarado y **no** define el suyo."""
    source = (REPO_ROOT / "src" / "cfdtrader" / "delivery" / "run_daily.py").read_text(
        encoding="utf-8"
    )
    classes = {node.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ClassDef)}
    assert not [name for name in classes if "Bridge" in name or "Puente" in name]
    assert "OverlayClient(" in source, "el puente publico de llm.budget es el que se usa"


def test_a9_without_a_provider_the_pipeline_still_recommends(
    store_root: Path,
    runs_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A8/A9: sin clave de LLM el overlay queda desactivado y **el dia se recomienda igual**.

    Es la red de seguridad de `_compute_overlay`: cualquier fallo de configuracion, de proveedor o
    de plantilla tiene que degradar a `disabled_error` y no romper el pipeline (§4.9).
    """
    monkeypatch.setenv("LLM_API_KEY", "")
    journal_root = tmp_path / "journal"

    assert _daily_run(store_root, runs_root, journal_root) == 0
    captured = capsys.readouterr()

    assert "estado: recommendation" in captured.out
    row = _journal_row(journal_root)
    assert row["llm_overlay"] == "disabled_error"
    assert row["prompt_hashes"] == {}


def test_the_whole_chain_runs_when_there_are_headlines(
    store_root: Path,
    runs_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """La cadena completa con proveedor simulado: titulares -> agente -> overlay -> gate -> diario.

    Se copia el almacen de fixtures para **no contaminar** el compartido: si el dataset de
    noticias quedase en `store_root`, las demas pruebas del fichero intentarian ejecutar el overlay
    de verdad.
    """
    data_root = tmp_path / "store"
    shutil.copytree(store_root, data_root)

    moment = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
    ingest_news(
        store=Store(data_root),
        headlines=(
            Headline(
                source="rss",
                feed="qa",
                title="La Fed mantiene los tipos sin cambios",
                url="https://example.invalid/fed",
                published_at=moment,
            ),
        ),
        now=moment,
    )

    class _FakeProvider:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, request: LLMRequest) -> LLMResponse:
            self.calls += 1
            return LLMResponse(content='{"events": []}', model="falso", system_fingerprint="fp-qa")

    provider = _FakeProvider()
    monkeypatch.setenv("LLM_API_KEY", "clave-solo-para-esta-prueba")
    monkeypatch.setenv("LLM_MODEL_EXTRACT", "modelo-falso-de-prueba")

    def _fake_build(settings: object) -> _FakeProvider:
        return provider

    monkeypatch.setattr(run_daily, "build_client", _fake_build)

    journal_root = tmp_path / "journal"
    assert _daily_run(data_root, runs_root, journal_root) == 0
    capsys.readouterr()

    assert provider.calls == 1, "un lote de un titular es una llamada"
    row = _journal_row(journal_root)
    assert row["llm_overlay"] == "applied"
    hashes = cast("Mapping[str, object]", row["prompt_hashes"])
    assert list(hashes) == [PROMPT_TEMPLATE_NAME], "el hash del prompt queda registrado"


def test_a12_the_cost_layer_still_does_not_touch_the_decision_layer() -> None:
    """El reparto sigue en pie: #33 prohibe `llm -> decision` y #35 prohibe `decision -> llm`."""
    budget = (REPO_ROOT / "src" / "cfdtrader" / "llm" / "budget.py").read_text(encoding="utf-8")
    overlay = (REPO_ROOT / "src" / "cfdtrader" / "decision" / "overlay.py").read_text(
        encoding="utf-8"
    )
    budget_modules = {
        node.module
        for node in ast.walk(ast.parse(budget))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    overlay_modules = {
        node.module
        for node in ast.walk(ast.parse(overlay))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not [name for name in budget_modules if name.startswith("cfdtrader.decision")]
    assert not [name for name in overlay_modules if name.startswith("cfdtrader.llm")]


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
    sesion sin tocar el calendario de mercado (`MarketCalendar`).
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


def test_a3_a_market_closed_day_is_a_justified_nothing_in_the_journal(
    store_root: Path, runs_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A3 (#113): un dia de mercado cerrado se ejecuta y deja un `NOTHING` justificado (regla 19).

    Vale para un festivo de EE. UU. (declarado via ``--calendar`` como ``extra_holidays``) y para
    un fin de semana real: en los dos, la sesion **no** es sesion, el gate lo justifica y la fila
    del diario queda con `status: recommendation`, `direction: nothing` y el bloqueo
    `mercado_cerrado`.
    """
    holiday_calendar = _write_calendar(tmp_path, holidays=(NEXT_SESSION,))
    cases = (
        (NEXT_SESSION, AS_OF_NEXT, holiday_calendar),
        (SATURDAY_SESSION, AS_OF_SATURDAY, None),
    )
    for session, as_of, calendar_path in cases:
        journal_root = tmp_path / f"journal-{session.isoformat()}"
        arguments = [
            "--as-of",
            as_of,
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
        if calendar_path is not None:
            arguments += ["--calendar", str(calendar_path)]
        code = run_daily.main(arguments)
        captured = capsys.readouterr()
        assert code == 0, captured.err
        assert "estado: recommendation" in captured.out, session
        assert "direccion: NOTHING" in captured.out, session
        assert "bloqueo: 19:mercado_cerrado" in captured.out, session
        assert "no_recommendation" not in captured.out, session
        assert "direccion: LONG" not in captured.out, session
        assert "direccion: SHORT" not in captured.out, session
        # 2026 **ya esta declarado** (#132): el aviso de #124 ya no sale para ese año.
        assert "no esta declarado en el calendario de FOMC" not in captured.err, session

        record = read_decision(journal_root, session)
        assert record["status"] == "recommendation"
        assert record["direction"] == "nothing"
        assert record["blocking_events"] == ["mercado_cerrado"]
        assert captured.out == cast("str", record["report_text"]) + "\n"


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
    # 2026 **ya esta declarado** (#132): el aviso de #124 ya no sale para ese año.
    assert "no esta declarado en el calendario de FOMC" not in captured.err

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
    # 2026 **ya esta declarado** (#132): el aviso de #124 ya no sale para ese año.
    assert "no esta declarado en el calendario de FOMC" not in captured.err

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
    """A11 (#40/#113): el festivo y la media sesion escriben y son deterministas byte a byte."""
    holiday_calendar = _write_calendar(tmp_path, holidays=(NEXT_SESSION,))
    half_calendar = _write_calendar(tmp_path, holidays=(NEXT_SESSION,), half_days=(STALE_SESSION,))
    scenarios = (
        (AS_OF_NEXT, holiday_calendar, NEXT_SESSION),
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
            digests.append((journal / "decisions" / f"{written.isoformat()}.json").read_bytes())

        assert outputs[0] == outputs[1], as_of
        assert digests[0] == digests[1], as_of


# ─────────────────────────────────────────────────────────────────────────────
# #43: la traza estructurada del camino diario (run_log + manifest)
# ─────────────────────────────────────────────────────────────────────────────
def test_the_run_writes_the_observability_trace_under_the_journal_root(
    store_root: Path, runs_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A15 (#43): la ejecución deja su traza (`run_log.jsonl` + `manifest.json`)."""
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
            str(store_root),
            "--runs-root",
            str(runs_root),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err

    run_dir = journal_root / "ops" / NEXT_SESSION.isoformat()
    log_path = run_dir / "run_log.jsonl"
    manifest_path = run_dir / "manifest.json"
    assert log_path.is_file() and manifest_path.is_file()

    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    stages = [entry["stage"] for entry in entries]
    assert {"features", "predict", "gate", "journal"} <= set(stages)
    assert all(entry["ok"] is True for entry in entries)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == NEXT_SESSION.isoformat()
    assert manifest["git_commit"] == GIT_COMMIT
    assert manifest["ok"] is True
    assert cast("str", manifest["hashes"]["gate_sha256"]).startswith("sha256:")
    assert cast("str", manifest["versions"]["features_version"]).startswith(FEATURE_VERSION_PREFIX)
    assert manifest["versions"]["model_version"] == RUN_ID


def test_observability_root_overrides_the_default_location(
    store_root: Path, runs_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A16 (#43): `--observability-root` desvía la traza fuera del diario."""
    journal_root = tmp_path / "journal"
    observability_root = tmp_path / "trace"
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
            "--observability-root",
            str(observability_root),
        ]
    )
    assert code == 0, capsys.readouterr().err

    assert (observability_root / NEXT_SESSION.isoformat() / "run_log.jsonl").is_file()
    assert not (journal_root / "ops").exists()


def test_a_pipeline_failure_is_recorded_in_the_trace(
    store_root: Path, runs_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A17 (#43): un modelo ausente deja la etapa fallida con su traza y `manifest.ok = False`."""
    journal_root = tmp_path / "journal"
    code = run_daily.main(
        [
            "--as-of",
            AS_OF_NEXT,
            "--model-run",
            "0" * 64,  # no existe en `runs_root`: el modelo falta
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
    assert code == 2, captured.out

    run_dir = journal_root / "ops" / NEXT_SESSION.isoformat()
    log_text = (run_dir / "run_log.jsonl").read_text(encoding="utf-8")
    entries = [json.loads(line) for line in log_text.splitlines()]
    failed = [entry for entry in entries if entry["ok"] is False]
    assert failed, entries
    assert any("Traceback" in cast("str", entry["error"]) for entry in failed)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ok"] is False


# ─────────────────────────────────────────────────────────────────────────────
# #121 · Los eventos del dia en el informe, sin contaminar `blocking_events`
# ─────────────────────────────────────────────────────────────────────────────
_DAY_EVENTS = run_daily._day_events  # pyright: ignore[reportPrivateUsage]


def _third_friday(year: int, month: int) -> date:
    """El tercer viernes del mes: OPEX nominal y triple *witching* (`agents/event_calendar.py`)."""
    first = date(year, month, 1)
    return date(year, month, 1 + ((4 - first.weekday()) % 7) + 14)


def test_121_the_report_publishes_the_day_events_and_opex_is_not_a_block() -> None:
    """Un dia de OPEX **se lee** en el informe, y se distingue de un bloqueo.

    Es la decision declarada en `plan.md` §19.8: OPEX, triple *witching* y roll del ES son
    **informativos**. Si esto fallara con un `evento_bloqueante`, la decision se habria cambiado
    sin registrarla.
    """
    calendar = load_calendar()
    opex_day = _third_friday(2026, 9)
    signal = calendar_signal(calendar, opex_day, as_of=_instant(opex_day))
    kinds = {event.kind for event in signal.events}
    assert EventKind.OPEX in kinds, f"premisa del test: {opex_day} deberia ser OPEX"

    text = run_daily.render(
        status=GateStatus.RECOMMENDATION,
        session=opex_day,
        as_of=_instant(opex_day),
        snapshot_session=opex_day,
        model_source="modelo",
        message="motivo",
        calendar_events=signal,
    )

    assert "evento: opex |" in text
    assert "evento_bloqueante:" not in text, "OPEX NO bloquea: es informativo (§19.8)"
    assert "bloqueo:" not in text, "la seccion de eventos no inventa bloqueos"


def test_121_a_half_session_is_published_as_blocking() -> None:
    """La media sesion **si** bloquea (regla 18), y el informe lo publica con su prefijo."""
    calendar = load_calendar()
    half = date(2026, 11, 27)  # el dia despues de Accion de Gracias
    signal = calendar_signal(calendar, half, as_of=_instant(half))
    kinds = {event.kind for event in signal.events}
    assert EventKind.HALF_SESSION in kinds, f"premisa del test: {half} deberia ser media sesion"

    text = run_daily.render(
        status=GateStatus.RECOMMENDATION,
        session=half,
        as_of=_instant(half),
        snapshot_session=half,
        model_source="modelo",
        message="motivo",
        calendar_events=signal,
    )

    assert "evento_bloqueante: half_session |" in text


def test_121_without_a_calendar_signal_the_report_has_no_events_section() -> None:
    """Sin senal el informe sale igual: la seccion es opcional, no una obligacion del formato."""
    text = run_daily.render(
        status=GateStatus.NO_RECOMMENDATION_STALE_DATA,
        session=date(2026, 9, 17),
        as_of=_instant(date(2026, 9, 17)),
        snapshot_session=date(2026, 9, 17),
        model_source="modelo",
        message="motivo",
    )

    assert "evento:" not in text
    assert "evento_bloqueante:" not in text
    assert "motivo: motivo" in text


def test_121_a_broken_calendar_degrades_and_never_blocks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Si la senal falla, el dia se emite **sin** su seccion: §7.4, degradacion gracil."""

    def _boom(*args: object, **kwargs: object) -> object:
        raise EventCalendarInputError("calendario roto")

    monkeypatch.setattr(run_daily, "calendar_signal", _boom)
    moment = _instant(date(2026, 9, 17))

    assert _DAY_EVENTS(load_calendar(), date(2026, 9, 17), moment) is None
    assert "no se puede calcular la senal del calendario" in capsys.readouterr().err


def test_121_the_journal_row_keeps_only_what_blocks(
    store_root: Path, runs_root: Path, tmp_path: Path
) -> None:
    """El cableado **no** contamina `blocking_events`: lo informativo no entra ahi nunca."""
    journal_root = tmp_path / "journal"
    assert _daily_run(store_root, runs_root, journal_root) == 0

    row = _journal_row(journal_root)
    blockers = cast("list[str]", row["blocking_events"])
    informative = {EventKind.OPEX.value, EventKind.TRIPLE_WITCHING.value, EventKind.ES_ROLL.value}

    assert set(blockers).isdisjoint(informative), f"un informativo se colo: {blockers}"
    # #131: con el `R` de #60 cableado la sesion de los fixtures ya **autoriza** sobre el coste
    # declarado (§19.12), asi que `blocking_events` puede venir **vacio** (venia con las reglas 9 y
    # 10 mientras el supuesto no estaba cuantificado). Lo que este test fija —y sigue fijando— es
    # que lo informativo (OPEX, triple witching, roll de ES) **nunca** entra ahi. Que un bloqueo
    # si aterrice en `blocking_events` lo cubren los casos con bloqueo de este mismo fichero
    # (reglas 13, 1, 15 y las dos de calendario/beneficios).
    assert all(isinstance(entry, str) and entry for entry in blockers)


# ─────────────────────────────────────────────────────────────────────────────
# #124 · El conjunto de FOMC declarado llega al gate (parte (a) de #114)
# ─────────────────────────────────────────────────────────────────────────────
_DECLARED_FOMC = run_daily._declared_fomc_dates  # pyright: ignore[reportPrivateUsage]


def test_124_a_declared_year_reaches_the_gate_and_an_undeclared_one_is_announced(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """El literal `fomc_dates=()` no distinguía «hoy no hay FOMC» de «no hay calendario»."""
    dates, note, declared = _DECLARED_FOMC(2027)
    assert declared is True
    assert len(dates) == 16
    assert "federalreserve.gov" in note and "verificado" in note
    assert capsys.readouterr().err == "", "un año declarado no avisa de nada"

    events, absence, is_declared = _DECLARED_FOMC(2030)
    assert events == () and is_declared is False
    assert "no esta declarado" in absence
    assert "regla 17" in absence, "el aviso dice que la regla 17 no puede dispararse"


def test_124_a_declared_fomc_day_blocks_with_rule_17() -> None:
    """La regla 17 **se dispara** con el calendario declarado, no con un conjunto vacío."""
    dates, _, declared = _DECLARED_FOMC(2027)
    assert declared is True
    fomc_day = dates[0]
    cost = cost_breakdown(
        model=declared_cost_model(),
        slippage=declared_slippage_assumption(),
        notional_usd=NOTIONAL_USD,
        side=Side.LONG,
        nights=0,
    )
    output = evaluate_gate(
        session=fomc_day,
        as_of=datetime.combine(fomc_day, datetime.min.time(), tzinfo=EASTERN),
        today=fomc_day,
        calendar=load_calendar(),
        prob_up_calibrated=0.6,
        expected_move_pct=Decimal("1.0"),
        expected_move_basis="sigma_k",
        cost=cost,
        capital_usd=NOTIONAL_USD,
        snapshot_ok=True,
        stop_pct=Decimal("0.5"),
        target_pct=Decimal("1.0"),
        fomc_dates=dates,
        params=scenario_parameters(cost_pct=cost.c_declared_pct),
    )
    assert ("17", "dia_de_fomc") in [(entry["rule"], entry["code"]) for entry in output.blockers]


# ─────────────────────────────────────────────────────────────────────────────
# #125 · Las publicaciones macro del dia (parte (b) de #114)
# ─────────────────────────────────────────────────────────────────────────────
#: Un registro con una unica serie que **no declara** hora de publicacion.
_MISSING_PUBLICATION_HOUR: Final[MacroSeriesRegistry] = MacroSeriesRegistry(
    version=1,
    series=(MacroSeriesSpec(series_id="SINHORA", name="serie sin hora declarada"),),
)


def _macro_store(root: Path, *, series_id: str, session: date) -> Store:
    """Un almacen minimo con una publicacion macro cuyo dia **en ET** es `session`."""
    published_at = datetime.combine(session, time(8, 30), tzinfo=EASTERN).astimezone(UTC)
    store = Store(root)
    store.append(
        "raw",
        "macro",
        [
            {
                "source": SOURCE_MACRO,
                "series_id": series_id,
                "as_of": date(session.year, session.month, 1),
                # `fetched_at` posterior al comunicado: el almacen exige published_at <= fetched_at.
                "fetched_at": datetime.combine(session, time(12, 0), tzinfo=EASTERN).astimezone(
                    UTC
                ),
                "published_at": published_at,
                "value": 1.0,
            }
        ],
    )
    return store


def _as_of_et(session: date, at: time) -> datetime:
    """El instante de decision de ese dia en ET (08:45 ET por defecto)."""
    return datetime.combine(session, at, tzinfo=EASTERN)


@pytest.mark.parametrize(
    ("session", "madrid_hour"),
    [
        (date(2026, 3, 10), 13),  # ventana DST de marzo: ET ya cambio, Madrid todavia no
        (date(2026, 6, 10), 14),  # alineados: Madrid va dos horas por delante de UTC
        (date(2026, 10, 27), 13),  # ventana DST de finales de octubre
    ],
)
def test_125_the_et_hour_is_fixed_and_madrid_moves(
    tmp_path: Path, session: date, madrid_hour: int
) -> None:
    """La hora ET del comunicado no se mueve; la de Madrid si (con `zoneinfo`, no un offset)."""
    store = _macro_store(tmp_path, series_id="CPIAUCSL", session=session)
    published = publications_on(store=store, session=session, as_of=_as_of_et(session, time(8, 45)))

    assert len(published) == 1
    (entry,) = published
    assert entry.series_id == "CPIAUCSL"
    assert entry.release_at_et is not None and entry.release_at_et.hour == 8
    assert entry.release_at_utc is not None and entry.release_at_utc.hour == 12
    assert entry.release_at_madrid is not None and entry.release_at_madrid.hour == madrid_hour
    assert entry.available_at_as_of is True


def test_125_a_publication_after_the_as_of_is_not_available(tmp_path: Path) -> None:
    """A las 08:00 ET el dato de las 08:30 ET todavia no esta: la disponibilidad lo dice."""
    session = date(2026, 6, 10)
    store = _macro_store(tmp_path, series_id="CPIAUCSL", session=session)

    published = publications_on(store=store, session=session, as_of=_as_of_et(session, time(8, 0)))

    assert published[0].available_at_as_of is False


def test_125_a_series_without_a_declared_hour_is_published_without_a_hour(tmp_path: Path) -> None:
    """Una serie que se publico y no declara hora **no la inventa**: declara su ausencia."""
    session = date(2026, 6, 10)
    store = _macro_store(tmp_path, series_id="SINHORA", session=session)

    published = publications_on(
        store=store,
        session=session,
        as_of=_as_of_et(session, time(8, 45)),
        registry=_MISSING_PUBLICATION_HOUR,
    )

    assert len(published) == 1
    entry = published[0]
    assert entry.release_time_et is None
    assert entry.release_at_utc is None
    assert entry.available_at_as_of is None


def test_125_a_day_without_publications_adds_no_line(tmp_path: Path) -> None:
    """Un dia sin publicaciones no anade ninguna linea (ni en el lector ni en el informe)."""
    store = _macro_store(tmp_path, series_id="CPIAUCSL", session=date(2026, 6, 10))

    other = date(2026, 6, 11)
    assert publications_on(store=store, session=other, as_of=_as_of_et(other, time(8, 45))) == ()

    text = run_daily.render(
        status=GateStatus.RECOMMENDATION,
        session=other,
        as_of=_instant(other),
        snapshot_session=other,
        model_source="modelo",
        message="motivo",
    )
    assert "publicacion_macro:" not in text


def test_125_the_report_publishes_the_publications_without_blocking() -> None:
    """La publicacion se lee en el informe y **no** anade ningun bloqueo."""
    session = date(2026, 6, 10)
    at_et = datetime.combine(session, time(8, 30), tzinfo=EASTERN)
    publication = MacroPublication(
        series_id="CPIAUCSL",
        name="CPI",
        unit="index",
        release_time_et=time(8, 30),
        release_at_et=at_et,
        release_at_utc=at_et.astimezone(UTC),
        release_at_madrid=at_et.astimezone(MADRID),
        available_at_as_of=True,
    )
    without_hour = MacroPublication(series_id="SINHORA", name="serie sin hora declarada")

    without = run_daily.render(
        status=GateStatus.RECOMMENDATION,
        session=session,
        as_of=_instant(session),
        snapshot_session=session,
        model_source="modelo",
        message="motivo",
        publications=(),
    )
    text = run_daily.render(
        status=GateStatus.RECOMMENDATION,
        session=session,
        as_of=_instant(session),
        snapshot_session=session,
        model_source="modelo",
        message="motivo",
        publications=(publication, without_hour),
    )

    assert (
        "publicacion_macro: CPIAUCSL | CPI | 08:30 ET (12:30 UTC / 14:30 Madrid) | "
        "disponible en el as_of: si" in text
    )
    assert (
        "publicacion_macro: SINHORA | serie sin hora declarada | hora no declarada | "
        "disponible en el as_of: no evaluable" in text
    )
    # Un dia sin publicaciones no anade lineas; con publicaciones, **ninguna** es un bloqueo.
    assert "publicacion_macro:" not in without
    assert "bloqueo:" not in text
    assert "evento_bloqueante:" not in text


# ─────────────────────────────────────────────────────────────────────────────
# #126 · Resultados de mega-caps: estimado no bloquea, confirmado si (parte (c) de #114)
# ─────────────────────────────────────────────────────────────────────────────
def _store_with_earnings(root: Path, *, certainty: str, session: date) -> None:
    """El almacen sintetico **mas** una observacion de resultado de NVDA para `session`."""
    _build_store(root, vix_mode="varying")
    observed = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
    Store(root).append(
        "raw",
        "earnings",
        [
            {
                "source": "yfinance",
                "series_id": f"NVDA@{session.isoformat()}",
                "as_of": observed,
                "fetched_at": FETCHED_AT,
                "published_at": observed,
                "name": "NVIDIA",
                "event_date": session,
                "moment": "amc",
                "certainty": certainty,
                "observed_at": observed,
            }
        ],
    )


def test_126_the_report_publishes_earnings_with_moment_and_certainty() -> None:
    """El informe publica el momento (BMO/AMC/unknown) y la certeza, sin asumir el momento."""
    session = date(2026, 9, 17)
    observed = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
    confirmed = EarningsEvent(
        symbol="NVDA",
        name="NVIDIA",
        on=session,
        moment=EarningsMoment.AMC,
        certainty=EarningsCertainty.CONFIRMED,
        observed_at=observed,
    )
    estimated = EarningsEvent(
        symbol="AAPL",
        name="Apple",
        on=session,
        moment=EarningsMoment.UNKNOWN,
        certainty=EarningsCertainty.ESTIMATED,
        observed_at=observed,
    )

    text = run_daily.render(
        status=GateStatus.RECOMMENDATION,
        session=session,
        as_of=_instant(session),
        snapshot_session=session,
        model_source="modelo",
        message="motivo",
        earnings=(confirmed, estimated),
    )

    assert (
        "resultado_mega_cap: NVDA | NVIDIA | 2026-09-17 | momento: amc | certeza: confirmed | "
        "bloquea: si" in text
    )
    assert (
        "resultado_mega_cap: AAPL | Apple | 2026-09-17 | momento: unknown | certeza: estimated | "
        "bloquea: no" in text
    )
    assert "bloqueo:" not in text, "la seccion de earnings no inventa bloqueos del gate"


def test_126_only_a_confirmed_earnings_enters_blocking_events(
    runs_root: Path, tmp_path: Path
) -> None:
    """Una fecha confirmada deja su codigo en `blocking_events`; una estimada, no."""
    _store_with_earnings(tmp_path / "estimated_store", certainty="estimated", session=NEXT_SESSION)
    estimated_journal = tmp_path / "estimated_journal"
    assert _daily_run(tmp_path / "estimated_store", runs_root, estimated_journal) == 0
    estimated = cast("list[str]", _journal_row(estimated_journal)["blocking_events"])
    assert not any(code.startswith("earnings_confirmado") for code in estimated)

    _store_with_earnings(tmp_path / "confirmed_store", certainty="confirmed", session=NEXT_SESSION)
    confirmed_journal = tmp_path / "confirmed_journal"
    assert _daily_run(tmp_path / "confirmed_store", runs_root, confirmed_journal) == 0
    confirmed = cast("list[str]", _journal_row(confirmed_journal)["blocking_events"])
    assert "earnings_confirmado:NVDA" in confirmed


def test_126_a_day_without_earnings_adds_no_line() -> None:
    """Un dia sin resultados no anade ninguna linea."""
    text = run_daily.render(
        status=GateStatus.RECOMMENDATION,
        session=NEXT_SESSION,
        as_of=_instant(NEXT_SESSION),
        snapshot_session=NEXT_SESSION,
        model_source="modelo",
        message="motivo",
    )
    assert "resultado_mega_cap:" not in text


# ─────────────────────────────────────────────────────────────────────────────
# #37 · El informe redactado por el LLM entra en el informe y en el diario
# ─────────────────────────────────────────────────────────────────────────────
def test_37_the_composed_report_reaches_the_report_and_the_journal(
    store_root: Path,
    runs_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Con una redaccion disponible, se publica y el diario guarda el texto redactado verbatim."""
    draft = report_agent.ReportDraft(
        narrative="sin edge demostrado: la pista es apoyo a la decision",
        bull_case=("el soporte aguanta",),
        bear_case=("la subasta falla",),
        prompt_hash="sha256:" + "a" * 64,
        model="report-model-v1",
    )

    def _compose(
        *args: object, **kwargs: object
    ) -> tuple[report_agent.ReportDraft, dict[str, str]]:
        return (draft, {report_agent.PROMPT_TEMPLATE_NAME: draft.prompt_hash})

    monkeypatch.setattr(run_daily, "_compose_report", _compose)
    journal_root = tmp_path / "journal"

    assert _daily_run(store_root, runs_root, journal_root) == 0

    captured = capsys.readouterr()
    assert "redaccion: sin edge demostrado" in captured.out
    assert "contra_argumento: la subasta falla" in captured.out
    row = _journal_row(journal_root)
    assert "redaccion: sin edge demostrado" in cast("str", row["report_text"])
    hashes = cast("dict[str, str]", row["prompt_hashes"])
    assert hashes[report_agent.PROMPT_TEMPLATE_NAME] == draft.prompt_hash


def test_37_without_a_redaction_the_report_is_still_emitted(
    store_root: Path, runs_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sin redaccion (el doble autouse), el informe sale igual: la capa es opcional."""
    journal_root = tmp_path / "journal"

    assert _daily_run(store_root, runs_root, journal_root) == 0

    captured = capsys.readouterr()
    assert "redaccion:" not in captured.out
    assert "estado: recommendation" in captured.out
    assert "no hay edge demostrado" in captured.out
