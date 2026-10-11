"""Tests de la busqueda pre-registrada de especificaciones de volatilidad (tarea #58).

Todo se escribe en ``tmp_path`` (informes y registro de ensayos) y los almacenes son
**sinteticos**: el test no abre red y no necesita el ``data/`` del repositorio. Las dos
features que podrian depender de el (la ingesta de la *term structure* y el fin a fin real) se
comprueban leyendo el registro declarado y, cuando el almacen esta, con la fixture ``needs_store``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import polars as pl
import pytest
import yaml  # type: ignore[import-untyped]
from arch import arch_model

from cfdtrader.analysis import volatility_forecast
from cfdtrader.analysis import volatility_spec_search as search
from cfdtrader.analysis.experiment_log import (
    ExperimentConfig,
    Registry,
    TrialsMismatchError,
    load_registry,
    require_trials_match_registry,
    run_sha256,
)
from cfdtrader.analysis.volatility_forecast import (
    CandidateResult,
    Metrics,
    Status,
    Target,
    walk_forward,
)
from cfdtrader.backtest.metrics import sharpe_ratio
from cfdtrader.backtest.overfitting import (
    VERDICT_DETECTED,
    VERDICT_NOT_DETECTED,
    VERDICT_NOT_SIGNIFICANT,
    VERDICT_SIGNIFICANT,
)
from cfdtrader.data.settings import ConfigurationError
from cfdtrader.data.store import Store

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"
NOW: Final[datetime] = datetime(2026, 10, 11, 21, tzinfo=UTC)

#: Sesiones sinteticas: mas de ``MIN_TRAIN`` + tres tramos de reajuste, y por encima de
#: ``MIN_CLEAN_SESSIONS``.
SESSIONS: Final[int] = 560
START: Final[date] = date(2016, 1, 4)

needs_store = pytest.mark.skipif(
    not (REAL_DATA / "derived" / "reports" / "volatility_forecast_2026-09-18.json").exists(),
    reason="el almacen real no esta en el arbol: el fin a fin sobre el se mide solo cuando esta",
)


def _sessions(count: int = SESSIONS) -> list[date]:
    """Dias laborables consecutivos desde ``START`` (sin festivos: la muestra limpia los tolera)."""
    days: list[date] = []
    current = START
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _returns(count: int, *, seed: int = 58) -> np.ndarray:
    """Serie de retornos con agrupamiento de volatilidad, determinista."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, 0.01, count)
    return np.asarray(
        np.clip(shocks * (1.0 + 0.6 * np.abs(np.sin(np.arange(count) / 17.0))), -0.05, 0.05),
        dtype=float,
    )


def _market_records(
    series_id: str, sessions: list[date], closes: list[float]
) -> list[dict[str, object]]:
    """Filas de ``raw.market_daily`` para una serie, con ``as_of`` a cierre de sesion ET.

    El ``open`` es el cierre anterior **con un hueco minimo pero no nulo**: si fuera exactamente
    el cierre anterior, toda la serie seria el artefacto de #52 (`open` repetido) y la muestra
    limpia quedaria vacia — que es justo lo que el arnes de #7 rechaza.
    """
    records: list[dict[str, object]] = []
    previous = closes[0]
    for index, (session, close) in enumerate(zip(sessions, closes, strict=True)):
        open_px = previous * (1.0 + 0.001 * math.sin(index))
        records.append(
            {
                "source": "yfinance",
                "series_id": series_id,
                "as_of": datetime(session.year, session.month, session.day, 21, tzinfo=UTC),
                "fetched_at": NOW,
                "published_at": None,
                "open": open_px,
                "high": max(open_px, close) * 1.002,
                "low": min(open_px, close) * 0.998,
                "close": close,
                "volume": 1_000.0,
            }
        )
        previous = close
    return records


def _prices(count: int, *, start: float = 100.0, seed: int = 7) -> list[float]:
    """Camino de precios determinista (los ``open`` no se repiten: la muestra sale limpia)."""
    rng = np.random.default_rng(seed)
    steps = np.exp(rng.normal(0.0, 0.008, count).cumsum())
    return [float(start * step) for step in steps]


def _store(root: Path, *, count: int = SESSIONS, with_term_structure: bool = True) -> Store:
    """Almacen sintetico con ``^GSPC``, ``^VIX`` y (si se pide) la *term structure* del VIX.

    El VIX y sus dos vertices **oscilan** (no son rampas): con una rampa monótona los niveles
    serian colineales y el ajuste del HAR-VIX no seria estimable, que es una propiedad del dato
    sintetico, no de la especificacion.
    """
    store = Store(root)
    sessions = _sessions(count)
    store.append("raw", "market_daily", _market_records("^GSPC", sessions, _prices(count)))
    store.append(
        "raw",
        "market_daily",
        _market_records(
            "^VIX", sessions, [20.0 + 3.0 * math.sin(index / 7.0) for index in range(count)]
        ),
    )
    if with_term_structure:
        store.append(
            "raw",
            "market_daily",
            _market_records(
                "^VIX9D",
                sessions,
                [15.0 + 2.0 * math.sin(index / 5.0 + 1.0) for index in range(count)],
            ),
        )
        store.append(
            "raw",
            "market_daily",
            _market_records(
                "^VIX3M",
                sessions,
                [22.0 + 4.0 * math.sin(index / 9.0 + 0.5) for index in range(count)],
            ),
        )
    return store


def _frame(count: int = 200, *, seed: int = 58) -> pl.DataFrame:
    """Marco con las columnas que consume el arnes de #7 y ninguna mas (los tests puros).

    El proxy de Parkinson se sortea **aparte** de los retornos: en el dato real son estimadores
    distintos, y si se atara a ``ret^2`` el regresor del salto saldria constante o colineal con
    ``har_lag1`` (una propiedad del dato sintetico, no de la especificacion).
    """
    returns = _returns(count, seed=seed)
    rng = np.random.default_rng(seed + 1)
    variance = np.abs(rng.normal(0.0, 0.006, count)) + 1e-6
    frame = pl.DataFrame(
        {
            "session": _sessions(count),
            "ret_log": returns,
            "ret_sq": returns**2,
            "parkinson_rv": variance,
        }
    )
    base = pl.col("parkinson_rv")
    return frame.with_columns(
        base.shift(1).alias("har_lag1"),
        base.shift(2).rolling_mean(window_size=4, min_samples=4).alias("har_lag4"),
        base.shift(6).rolling_mean(window_size=17, min_samples=17).alias("har_lag17"),
        pl.lit(20.0).alias("vix_level"),
        pl.lit(0.0).alias("vix_zscore"),
        pl.lit(0.5).alias("vix_percentile"),
    )


def _constant_frame(count: int = 60) -> pl.DataFrame:
    """Marco degenerado: varianza cero, sin ajuste posible en ninguna familia."""
    frame = pl.DataFrame(
        {
            "session": _sessions(count),
            "ret_log": [0.0] * count,
            "ret_sq": [0.0] * count,
            "parkinson_rv": [0.0] * count,
        }
    )
    base = pl.col("parkinson_rv")
    return frame.with_columns(
        base.shift(1).alias("har_lag1"),
        base.shift(2).rolling_mean(window_size=4, min_samples=4).alias("har_lag4"),
        base.shift(6).rolling_mean(window_size=17, min_samples=17).alias("har_lag17"),
    )


def _bounds(frame: pl.DataFrame, *, min_train: int = 80, size: int = 20) -> list[tuple[int, int]]:
    """Tramos de reajuste para un marco sintetico (mismo esquema, ventana mas corta)."""
    return [
        (start, min(start + size, frame.height)) for start in range(min_train, frame.height, size)
    ]


def _targets(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Los dos objetivos de #7 como arrays de numpy."""
    return {
        target.value: np.asarray(
            frame.get_column(volatility_forecast.TARGET_COLUMNS[target])
            .cast(pl.Float64)
            .fill_null(float("nan"))
            .to_numpy(),
            dtype=float,
        )
        for target in volatility_forecast.TARGETS
    }


def _spec(name: str) -> search.Spec:
    """La especificacion del catalogo con ese nombre."""
    return next(spec for spec in search.SPECS if spec.name == name)


# ─────────────────────────────────────────────────────────────────────────────
# A1 - catalogo pre-registrado, sin busqueda de hiperparametros
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_catalogue_is_pre_registered_and_free_of_hyperparameter_search() -> None:
    assert [spec.name for spec in search.SPECS] == [
        "harq",
        "har_jump",
        "har_vix_ts",
        "ridge_har",
        "lasso_har",
        "quantile_har",
        "egarch",
        "aparch",
        "har_intraday_rv",
    ]
    assert len(search.SPECS) == 9
    assert len(search.EVALUABLE_SPECS) == 8
    assert [spec.name for spec in search.TRIAL_SPECS] == [
        "har",
        "har_vix",
        "garch",
        *(spec.name for spec in search.EVALUABLE_SPECS),
    ]
    intraday = search.SPECS[-1]
    assert intraday.evaluable is False
    assert intraday.follow_up == "#57"
    assert intraday.reason is not None and intraday.reason
    for spec in search.EVALUABLE_SPECS:
        assert spec.formula, spec.name
        assert spec.hyperparameters, spec.name
        # Ninguna rejilla: cada valor declarado es un escalar o una cadena, nunca una lista.
        assert all(
            isinstance(value, (int, float, str)) for value in spec.hyperparameters.values()
        ), spec.name
    payload = intraday.to_payload()
    assert payload["status"] == Status.UNAVAILABLE.value
    assert payload["source"] == "#58"
    assert payload["follow_up"] == "#57"
    # La no evaluable no produce pronostico: sale a NaN con su motivo, nunca un numero.
    frame = _frame()
    forecast, reason = search.spec_forecasts(frame, spec=intraday, bounds=_bounds(frame))
    assert not np.isfinite(forecast).any()
    assert reason == intraday.reason
    # El CLI no acepta un numero de ensayos propio: `n_trials` se deriva del registro.
    with pytest.raises(SystemExit):
        search.main(["--n-trials", "5"])


# ─────────────────────────────────────────────────────────────────────────────
# A2 - la term structure del VIX: declarada, ingerida y sin look-ahead
# ─────────────────────────────────────────────────────────────────────────────
def _declared_series() -> dict[str, dict[str, object]]:
    """Las series declaradas en `config/data_sources.yaml`."""
    document = yaml.safe_load((REPO_ROOT / "config" / "data_sources.yaml").read_text("utf-8"))
    return {item["series_id"]: item for item in document["series"]}


def test_a2_term_structure_is_declared_and_reads_only_yesterday(tmp_path: Path) -> None:
    declared = _declared_series()
    for series_id, min_start in (("^VIX9D", "2011-01-01"), ("^VIX3M", "2006-07-01")):
        spec = declared[series_id]
        # El `min_start` declara lo que la fuente tiene de verdad: `^VIX9D` empieza en 2011.
        assert str(spec["min_start"]) == min_start, series_id
        assert spec["dataset"] == "market_daily"
        assert spec["primary"] == "yfinance"
    closes = pl.DataFrame(
        {
            "session": _sessions(20),
            "vix9d_close": [15.0 + index for index in range(20)],
            "vix3m_close": [22.0 + index for index in range(20)],
        }
    )
    before = search.term_structure_regressors(closes)
    mutated = closes.with_columns(
        pl.when(pl.col("session") == _sessions(20)[10])
        .then(pl.col("vix9d_close") * 3.0)
        .otherwise(pl.col("vix9d_close"))
        .alias("vix9d_close")
    )
    after = search.term_structure_regressors(mutated)
    # El regresor de `t` es el cierre de `t-1`: mutar la sesion `t` no lo cambia...
    assert before["vix9d_level"][10] == after["vix9d_level"][10]
    assert before["vix9d_level"][10] == pytest.approx(closes["vix9d_close"][9])
    # ...y el de `t+1` si cambia, porque ese si usa la sesion mutada.
    assert before["vix9d_level"][11] != after["vix9d_level"][11]
    # La pendiente declarada es `vix3m / vix9d - 1`, tambien de `t-1`.
    assert before["vix_ts_slope"][10] == pytest.approx(
        closes["vix3m_close"][9] / closes["vix9d_close"][9] - 1.0
    )
    # Sin las dos series, la especificacion es no evaluable con motivo: nunca un numero.
    frame = _frame()
    forecast, reason = search.spec_forecasts(frame, spec=_spec("har_vix_ts"), bounds=_bounds(frame))
    assert not np.isfinite(forecast).any()
    assert reason is not None and "vix9d_level" in reason
    # El almacen sintetico si las tiene: el fin a fin las usa (A9).
    assert search.load_term_structure(_store(tmp_path / "store")) is not None
    # Y sin almacen de por medio, el marco nuevo trae los dos regresores derivados (A4).
    spec_frame = search.build_spec_frame(frame, term_structure=None)
    assert "har_quarticity" in spec_frame.columns and "har_jump" in spec_frame.columns


# ─────────────────────────────────────────────────────────────────────────────
# A3 - la metrica y los folds son los de #7, importados
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_metric_folds_and_the_four_rows_are_the_ones_of_issue_7(tmp_path: Path) -> None:
    forecast = np.array([1e-4, 2e-4, 3e-4, -1.0])
    target = np.array([1e-4, 1e-4, 4e-4, 1e-5])
    # `evaluate_forecast` es la **misma** funcion que la metrica privada de #7, no una copia.
    assert volatility_forecast.evaluate_forecast(forecast, target) == volatility_forecast._metrics(  # pyright: ignore[reportPrivateUsage]
        forecast, target
    )
    assert search.evaluate_forecast is volatility_forecast.evaluate_forecast
    # Los folds y las constantes son las importadas: el mismo objeto, no un valor copiado.
    assert search.garch_fold_bounds is volatility_forecast.garch_fold_bounds
    for name in ("MIN_TRAIN", "REFIT_EVERY", "RELATIVE_TOLERANCE", "REFERENCE"):
        if name == "REFERENCE":
            continue
        assert getattr(search, name) == getattr(volatility_forecast, name), name
    assert volatility_forecast.Candidate.RW.value == search.REFERENCE
    assert search.DECLARED_CANDIDATE is volatility_forecast.DECLARED_CANDIDATE
    # Las cuatro filas de #7 se recalculan llamando a su `walk_forward` sobre su muestra:
    # si la metrica o los folds se hubieran movido, dejarian de coincidir.
    store = _store(tmp_path / "store")
    frame, _ = volatility_forecast.build_sample(store)
    report = search.analyse(
        store=store,
        reports_dir=tmp_path / "reports",
        trials_root=tmp_path / "trials",
        as_of=NOW,
        write=False,
    )
    expected = walk_forward(frame).candidates
    assert [row.name for row in expected] == ["rw", "har", "har_vix", "garch"]
    assert list(report.rows[:4]) == list(expected)
    assert [row.name for row in report.rows[4:]] == [spec.name for spec in search.SPECS]
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "trials").exists()


# ─────────────────────────────────────────────────────────────────────────────
# A4 - cada familia mide o declara por que no
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_every_family_produces_a_forecast_or_declares_a_reason() -> None:
    frame = search.build_spec_frame(_frame(220), term_structure=None)
    bounds = _bounds(frame)
    open_at = bounds[0][0]
    for spec in search.EVALUABLE_SPECS:
        if spec.name == "har_vix_ts":
            continue  # sin las series del VIX: se comprueba en A2
        forecast, reason = search.spec_forecasts(frame, spec=spec, bounds=bounds)
        evaluated = forecast[open_at:]
        assert np.isfinite(evaluated).any(), (spec.name, reason)
        finite = evaluated[np.isfinite(evaluated)]
        # Un pronostico o es una varianza positiva y finita, o no esta: nunca un 0 ni un inf.
        assert (finite > 0.0).all(), spec.name
        assert np.isfinite(finite).all(), spec.name


def test_a4b_the_regressors_are_the_declared_formulas() -> None:
    base = _frame(120)
    with_regressors = search.build_spec_frame(base, term_structure=None)
    quarticity = base["parkinson_rv"].to_numpy()
    jumps = base["ret_sq"].to_numpy() - base["parkinson_rv"].to_numpy()
    for index in range(23, 120):
        expected_quarticity = np.log((4.0 / 3.0) * quarticity[index - 1] ** 2)
        assert float(with_regressors["har_quarticity"][index]) == pytest.approx(expected_quarticity)
        assert float(with_regressors["har_jump"][index]) == pytest.approx(
            max(0.0, jumps[index - 1])
        )


def test_a4c_a_degenerate_series_is_declared_not_invented() -> None:
    frame = _constant_frame()
    bounds = [(30, 60)]
    evaluated = np.zeros(frame.height, dtype=bool)
    evaluated[30:] = True
    targets = _targets(frame)
    for name in ("harq", "har_jump", "ridge_har", "lasso_har", "quantile_har", "egarch"):
        spec = _spec(name)
        forecast, reason = search.spec_forecasts(frame, spec=spec, bounds=bounds)
        assert not np.isfinite(forecast).any(), name
        row = search.spec_row(
            spec=spec,
            forecast=forecast,
            reason=reason,
            targets=targets,
            evaluated=evaluated,
        )
        assert row.status == Status.UNAVAILABLE.value, name
        assert row.reason, name
        assert row.metrics[Target.PARKINSON.value] is None
        assert row.median_sigma_bp is None


def test_a4d_the_arch_recursions_are_the_ones_of_arch() -> None:
    """La recursion manual se contrasta contra el `forecast` de `arch`: si divergiera, el
    numero publicado seria plausible y falso."""
    returns = _returns(620, seed=11)
    for name in ("EGARCH", "APARCH"):
        fit = search.fit_arch(returns[:500], vol=name)
        model = arch_model(
            returns[:500],
            mean="Zero",
            vol=cast("Any", name),
            p=search.ARCH_P,
            o=search.ARCH_O,
            q=search.ARCH_Q,
            dist="normal",
            rescale=False,
        )
        result = cast("Any", model.fit(disp="off", show_warning=False))
        expected = float(
            np.asarray(result.forecast(horizon=1, reindex=False).variance, dtype=float).reshape(-1)[
                0
            ]
        )
        assert search.arch_step(fit, vol=name) == pytest.approx(expected, rel=1e-9)
        assert fit.params["omega"] == pytest.approx(float(result.params["omega"]))


def test_a4e_no_family_reads_the_future() -> None:
    frame = search.build_spec_frame(_frame(200), term_structure=None)
    bounds = [(80, 200)]
    cut = 140
    after = np.arange(frame.height) >= cut
    mutated = frame.with_columns(
        pl.Series(
            name,
            np.where(after, frame[name].to_numpy() * 3.0, frame[name].to_numpy()),
        )
        for name in ("parkinson_rv", "ret_sq", "ret_log")
    )
    for name in ("harq", "har_jump", "ridge_har", "lasso_har", "quantile_har", "egarch", "aparch"):
        spec = _spec(name)
        base, _ = search.spec_forecasts(frame, spec=spec, bounds=bounds)
        changed, _ = search.spec_forecasts(mutated, spec=spec, bounds=bounds)
        for index in range(bounds[0][0], cut + 1):
            left, right = base[index], changed[index]
            if np.isnan(left) and np.isnan(right):
                continue
            assert left == pytest.approx(right, rel=1e-12), (name, index)


# ─────────────────────────────────────────────────────────────────────────────
# A5 - registro de ensayos: su propia raiz y un `n_trials` derivado
# ─────────────────────────────────────────────────────────────────────────────
def _registry_with(root: Path, count: int) -> Path:
    """Registro sintetico de `count` variantes, cada una con su Sharpe de diferencial."""
    for index in range(count):
        search.register_spec(
            trials_root=root,
            config=ExperimentConfig(
                variant_id=f"synthetic_v1#{index}",
                features=("x",),
                hyperparameters={"index": index},
                seed=index,
                series_id="^GSPC",
                window={"index": index, "units": search.BASIS_LOSS_DIFFERENTIAL},
            ),
            sharpe_per_session=0.02 + 0.01 * index,
            n_observations=60,
            as_of=NOW,
            write=True,
        )
    return root


def test_a5_trials_go_to_their_own_root_and_n_trials_is_derived(tmp_path: Path) -> None:
    config = search.spec_config(
        spec=_spec("harq"),
        summary={"clean_from": "2016-01-01", "clean_sessions": 560},
        sessions=560,
        evaluated_sessions=60,
        folds=3,
        reference=search.REFERENCE,
    )
    assert config.variant_id == "volatility_spec_v1#harq"
    assert config.features == _spec("harq").regressors
    assert config.hyperparameters["estimator"] == "log_lstsq"
    assert config.seed == search.SPEC_SEED > 0
    assert config.window["units"] == search.BASIS_LOSS_DIFFERENTIAL
    assert config.window["reference"] == "rw"
    assert config.window["min_train"] == volatility_forecast.MIN_TRAIN
    record = search.register_spec(
        trials_root=tmp_path / "trials",
        config=config,
        sharpe_per_session=0.01,
        n_observations=60,
        as_of=NOW,
        write=True,
    )
    # La identidad es el contenido y los tres ficheros son los que lee `load_registry`.
    assert record.run_sha256 == run_sha256(config)
    run_dir = tmp_path / "trials" / record.run_sha256
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "result.json").is_file()
    assert (run_dir / "summary.md").is_file()
    document = json.loads((run_dir / "result.json").read_text("utf-8"))
    assert document["result"]["basis"] == search.BASIS_LOSS_DIFFERENTIAL
    registry = load_registry(tmp_path / "trials")
    assert registry.n_trials == 1
    # `n_trials` no se inventa: un valor propio es error tipado (con dos variantes ya hay V[SR]).
    with_two = load_registry(_registry_with(tmp_path / "reg_two", 2))
    with pytest.raises(TrialsMismatchError):
        require_trials_match_registry(
            n_trials=99, sr_variance=with_two.sr_variance, registry=with_two
        )
    # El registro por defecto **no** es `runs/` (la cadena de modelos lo enumera entero).
    assert search.DEFAULT_TRIALS_ROOT == "runs_volatility"
    assert search.DEFAULT_TRIALS_ROOT != "runs"
    # Mas variantes probadas => mas deflacion sobre la **misma** serie (con una ventaja modesta,
    # que es donde el DSR no se satura).
    rng = np.random.default_rng(58)
    noise = rng.normal(0.0, 1.0, 240)
    series = (noise - noise.mean()) + 0.03
    common = np.ones(series.size, dtype=bool)
    dsr: dict[int, float] = {}
    for count in (3, 10):
        block, reason = search.deflated_block(
            selected="harq",
            differentials={"harq": series},
            common=common,
            registry=load_registry(_registry_with(tmp_path / f"reg{count}", count)),
        )
        assert reason is None and block is not None
        assert block.n_trials == count
        dsr[count] = block.dsr
    assert dsr[10] < dsr[3]


# ─────────────────────────────────────────────────────────────────────────────
# A6 - DSR: unidades declaradas y aritmetica
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_differentials_are_the_qlike_summands_and_the_dsr_declares_units(
    tmp_path: Path,
) -> None:
    forecast = np.array([1e-4, 2e-4, 3e-4, 1e-3])
    target = np.array([1e-4, 1e-4, 4e-4, 2e-4])
    losses = search.qlike_losses(forecast, target)
    metric = cast("Metrics", volatility_forecast.evaluate_forecast(forecast, target))
    # La serie por sesion es exactamente el sumando del QLIKE de #7.
    assert float(np.nanmean(losses)) == pytest.approx(metric.qlike)
    assert int(np.isfinite(losses).sum()) == metric.sessions
    # Una perdida no finita queda fuera de la serie, nunca vale 0.
    with_nan = search.qlike_losses(np.array([1e-4, np.nan]), np.array([1e-4, 1e-4]))
    assert np.isnan(with_nan[1])
    reference = np.full(4, 50.0)
    # Positivo = la especificacion pierde menos que la referencia.
    assert (search.loss_differentials(reference, losses) > 0.0).all()
    assert (search.loss_differentials(losses, reference) < 0.0).all()
    series = 0.001 * np.sin(np.arange(240.0) / 5.0) + 0.0004
    common = np.ones(series.size, dtype=bool)
    registry = load_registry(_registry_with(tmp_path / "trials", 4))
    block, reason = search.deflated_block(
        selected="harq", differentials={"harq": series}, common=common, registry=registry
    )
    assert reason is None and block is not None
    assert block.units == "per_session"
    assert block.n_observations == series.size
    assert block.n_trials == 4
    assert block.sr_variance == pytest.approx(registry.sr_variance)
    assert block.sr_observed == pytest.approx(
        sharpe_ratio([float(value) for value in series], annualization=1)
    )
    # Sin registro no hay DSR: se declara el motivo, no se publica un numero.
    empty, empty_reason = search.deflated_block(
        selected="harq",
        differentials={"harq": series},
        common=common,
        registry=Registry(entries=(), registry_sha256="sha256:" + "0" * 64),
    )
    assert empty is None and empty_reason is not None


# ─────────────────────────────────────────────────────────────────────────────
# A7 - PBO: los bloques se calculan de `T`
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_pbo_blocks_are_computed_from_t_and_never_faked() -> None:
    # El mayor par admisible: bloques, no un divisor de `T` (con T impar no habria ninguno).
    assert search.blocks_for(2700) == 16
    assert search.blocks_for(60) == 16
    assert search.blocks_for(16) == 16
    assert search.blocks_for(10) == 10
    assert search.blocks_for(9) == 8
    assert search.blocks_for(2435) == 16
    assert search.blocks_for(1) is None
    series = {
        "harq": np.linspace(-1.0, 1.0, 60),
        "ridge_har": np.linspace(1.0, -1.0, 60),
    }
    common = np.ones(60, dtype=bool)
    block, reason = search.pbo_block(names=list(series), differentials=series, common=common)
    assert reason is None and block is not None
    assert block.n_variants == 2
    assert block.blocks == 16
    # Las 12 sesiones que sobran del principio se recortan para que los bloques cuadren.
    assert block.n_observations == 48
    assert 0.0 <= block.pbo <= 1.0
    single, single_reason = search.pbo_block(names=["harq"], differentials=series, common=common)
    assert single is None
    assert single_reason is not None and "pbo: 0" in single_reason
    thin = {"harq": np.arange(1.0), "ridge_har": np.arange(1.0)}
    none, thin_reason = search.pbo_block(
        names=list(thin), differentials=thin, common=np.ones(1, dtype=bool)
    )
    assert none is None
    assert thin_reason is not None and "bloques del CSCV" in thin_reason


def test_a7b_loss_units_never_rescales_a_degenerate_series() -> None:
    """Un diferencial sin magnitud se deja tal cual: de ahi sale un Sharpe 0, no un cociente."""
    zeros = np.zeros(5)
    assert np.allclose(search.loss_units(zeros), zeros)
    assert search.loss_units(np.array([])).size == 0
    raw = np.array([-3.0, 1.0, 2.0])
    scaled = search.loss_units(raw)
    # El escalado es positivo y comun: el signo y el orden no cambian...
    assert (np.sign(scaled) == np.sign(raw)).all()
    assert np.all(np.abs(scaled) <= 1.0)
    # ...y el Sharpe es **el mismo** (es invariante a un escalado positivo), que es lo unico que
    # justifica la transformacion.
    assert sharpe_ratio(list(scaled), annualization=1) == pytest.approx(
        float(np.mean(raw) / np.std(raw, ddof=1))
    )


# ─────────────────────────────────────────────────────────────────────────────
# A8 - el veredicto exige las tres condiciones
# ─────────────────────────────────────────────────────────────────────────────
def _row(
    name: str, qlike: float, *, status: str = Status.OK.value, mse: float = 0.5
) -> CandidateResult:
    """Fila sintetica de la tabla."""
    metrics: dict[str, Metrics | None] = {
        target.value: Metrics(qlike=qlike, mse_log=mse, sessions=10)
        for target in volatility_forecast.TARGETS
    }
    return CandidateResult(
        name=name,
        status=status,
        reason=None if status == Status.OK.value else "no medible",
        metrics=metrics,
        median_sigma_bp=100.0,
    )


def test_a8_the_verdict_needs_the_three_conditions(tmp_path: Path) -> None:
    rows = [
        _row("harq", -10.0),
        _row("har_jump", -10.05),
        _row("ridge_har", -9.0),
        _row("har_intraday_rv", -12.0, status=Status.UNAVAILABLE.value),
    ]
    new_names = ["harq", "har_jump", "ridge_har"]
    # La mejor es el menor QLIKE primario; el empate lo rompe el orden del catalogo.
    assert search.best_new_spec(rows, names=new_names) == "har_jump"
    assert (
        search.best_new_spec(
            [_row("harq", -10.0), _row("har_jump", -10.0)], names=["harq", "har_jump"]
        )
        == "harq"
    )
    assert (
        search.best_new_spec(
            [_row("harq", -10.0, mse=0.2), _row("har_jump", -10.0, mse=0.1)],
            names=["harq", "har_jump"],
        )
        == "har_jump"
    )
    assert (
        search.best_new_spec([_row("harq", -10.0, status=Status.UNAVAILABLE.value)], names=["harq"])
        is None
    )
    # Bloques reales de #16, mutados para recorrer las cuatro ramas del veredicto.
    series = 0.001 * np.sin(np.arange(240.0) / 5.0) + 0.0004
    common = np.ones(series.size, dtype=bool)
    registry = load_registry(_registry_with(tmp_path / "trials", 4))
    deflated, deflated_reason = search.deflated_block(
        selected="harq", differentials={"harq": series}, common=common, registry=registry
    )
    pbo, pbo_reason = search.pbo_block(
        names=["harq", "ridge_har", "lasso_har"],
        differentials={
            "harq": series,
            "ridge_har": series[::-1].copy(),
            "lasso_har": series * 0.5,
        },
        common=common,
    )
    assert deflated_reason is None and pbo_reason is None
    assert deflated is not None and pbo is not None
    significant = dataclasses.replace(deflated, verdict=VERDICT_SIGNIFICANT)
    not_significant = dataclasses.replace(deflated, verdict=VERDICT_NOT_SIGNIFICANT)
    not_detected = dataclasses.replace(pbo, verdict=VERDICT_NOT_DETECTED)
    detected = dataclasses.replace(pbo, verdict=VERDICT_DETECTED)
    beating = {Target.PARKINSON.value: -0.5, Target.RET_SQ.value: -0.5}
    losing = {Target.PARKINSON.value: 0.01, Target.RET_SQ.value: -0.5}
    improves = search.improvement_verdict(
        reference="garch", best_new="harq", gaps=beating, deflated=significant, pbo=not_detected
    )
    assert improves["verdict"] == search.VERDICT_IMPROVES
    assert improves["improves"] is True
    assert (
        cast("Mapping[str, object]", improves["conditions"])["beats_reference_beyond_tolerance"]
        is True
    )
    keeps = search.improvement_verdict(
        reference="garch", best_new="harq", gaps=losing, deflated=significant, pbo=not_detected
    )
    assert keeps["verdict"] == search.VERDICT_KEEPS_7 and keeps["improves"] is False
    assert "no bate a `garch`" in cast(str, keeps["reason"])
    deflated_off = search.improvement_verdict(
        reference="garch", best_new="harq", gaps=beating, deflated=not_significant, pbo=not_detected
    )
    assert deflated_off["verdict"] == search.VERDICT_KEEPS_7
    assert "DSR" in cast(str, deflated_off["reason"])
    pbo_off = search.improvement_verdict(
        reference="garch", best_new="harq", gaps=beating, deflated=significant, pbo=detected
    )
    assert pbo_off["verdict"] == search.VERDICT_KEEPS_7
    assert "PBO" in cast(str, pbo_off["reason"])
    inconclusive = search.improvement_verdict(
        reference="garch", best_new="harq", gaps=beating, deflated=None, pbo=not_detected
    )
    assert inconclusive["verdict"] == search.VERDICT_INCONCLUSIVE
    assert inconclusive["improves"] is False
    # Un `not_evaluable` nunca se escribe como aprobado, ni con la brecha a favor.
    conditions = cast("Mapping[str, object]", inconclusive["conditions"])
    assert conditions["deflated_sharpe_significant"] is False
    assert conditions["beats_reference_beyond_tolerance"] is True
    # Pero con la brecha ya medida y fallando el veredicto **no** es una inferencia: aunque no
    # haya ningun bloque, lo medido dice que se queda el de #7.
    measured = search.improvement_verdict(
        reference="garch", best_new="harq", gaps=losing, deflated=None, pbo=None
    )
    assert measured["verdict"] == search.VERDICT_KEEPS_7
    assert measured["improves"] is False
    assert "no bate a `garch`" in cast(str, measured["reason"])


# ─────────────────────────────────────────────────────────────────────────────
# A9 - el informe: catalogo, tabla, correccion y veredicto
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_the_report_carries_the_catalogue_the_table_and_the_verdict(tmp_path: Path) -> None:
    _store(tmp_path / "store")
    code = search.main(
        [
            "--data-root",
            str(tmp_path / "store"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--trials-root",
            str(tmp_path / "trials"),
            "--as-of",
            NOW.isoformat(),
        ]
    )
    assert code == 0
    reports = sorted((tmp_path / "reports").glob("volatility_spec_search_*.json"))
    assert len(reports) == 1
    assert reports[0].name == "volatility_spec_search_2026-10-11.json"
    document = json.loads(reports[0].read_text("utf-8"))
    published = dict(document)
    digest = published.pop("report_sha256")
    assert (
        isinstance(digest, str)
        and digest.startswith("sha256:")
        and len(digest) == len("sha256:") + 64
    )
    # Autoconsistencia: el digest se recalcula sobre el propio documento (sin su propia clave).
    assert digest == search._report_digest(published)  # pyright: ignore[reportPrivateUsage]
    # Catalogo pre-registrado, con formulas y sin rejillas.
    catalogue = cast("list[dict[str, object]]", document["catalogue"])
    assert len(catalogue) == 9
    for entry in catalogue:
        assert entry["formula"] and entry["source"] == "#58"
    assert document["not_evaluable_specs"] == ["har_intraday_rv"]
    # Tabla: 4 filas de #7 + 8 nuevas + 1 declarada no evaluable.
    rows = cast("Mapping[str, Mapping[str, object]]", document["rows"])
    assert len(rows) == 13
    assert rows["har_intraday_rv"]["status"] == Status.UNAVAILABLE.value
    assert rows["har_intraday_rv"]["metrics"] == {"parkinson": None, "ret_sq": None}
    assert rows["rw"]["is_differential_reference"] is True
    # Correccion: 11 variantes, con el `n_trials` del registro y las unidades declaradas.
    correction = cast("Mapping[str, object]", document["correction"])
    assert correction["n_variants"] == 11
    assert correction["units_note"] == search.CORRECTION_UNITS_NOTE
    assert correction["basis"] == search.BASIS_LOSS_DIFFERENTIAL
    registry = cast("Mapping[str, object]", correction["registry"])
    assert registry["n_trials"] == 11
    assert len(list((tmp_path / "trials").glob("*/config.json"))) == 11
    assert correction["deflated_sharpe"] is not None
    assert correction["pbo"] is not None
    assert cast("int", correction["common_sessions"]) > 0
    # Veredicto con las tres condiciones y con su referencia declarada.
    verdict = cast("Mapping[str, object]", document["verdict"])
    assert verdict["verdict"] in {
        search.VERDICT_IMPROVES,
        search.VERDICT_KEEPS_7,
        search.VERDICT_INCONCLUSIVE,
    }
    assert isinstance(verdict["reason"], str) and verdict["reason"]
    assert verdict["improves"] is False or verdict["verdict"] == search.VERDICT_IMPROVES
    references = cast("Mapping[str, object]", document["references"])
    assert references["differential"] == "rw"
    assert references["declared_candidate"] == "garch"
    markdown = reports[0].with_suffix(".md").read_text("utf-8")
    for section in (
        "## Catalogo pre-registrado",
        "## Tabla comparativa",
        "## Correccion por comparaciones multiples (#16)",
        "## Veredicto",
        "## Limitaciones",
    ):
        assert section in markdown
    assert verdict["reason"] in markdown
    assert document["not_a_pnl_note"] in markdown
    assert search.CORRECTION_UNITS_NOTE in markdown


# ─────────────────────────────────────────────────────────────────────────────
# A10 - determinismo entre procesos y `--dry-run`
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT: Final[str] = """
import sys
from datetime import UTC, datetime
from pathlib import Path

from cfdtrader.analysis import volatility_spec_search as search
from cfdtrader.data.store import Store

report = search.analyse(
    store=Store(Path(sys.argv[1])),
    reports_dir=Path(sys.argv[2]),
    trials_root=Path(sys.argv[3]),
    as_of=datetime(2026, 10, 11, 21, tzinfo=UTC),
    write=False,
)
print(report.report_sha256)
"""


def test_a10_determinism_across_processes_and_dry_run(tmp_path: Path) -> None:
    _store(tmp_path / "store")
    digests: list[str] = []
    for seed in ("0", "1"):
        result = subprocess.run(  # noqa: S603 - comando propio, sin entrada externa
            [
                sys.executable,
                "-c",
                _SCRIPT,
                str(tmp_path / "store"),
                str(tmp_path / f"reports{seed}"),
                str(tmp_path / f"trials{seed}"),
            ],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        )
        digests.append(result.stdout.strip().splitlines()[-1])
    assert digests[0] == digests[1]
    assert digests[0].startswith("sha256:")
    # Y no depende de la ruta de los informes: `write=False` no escribe nada.
    assert not (tmp_path / "reports0").exists()
    assert not (tmp_path / "reports1").exists()
    code = search.main(
        [
            "--data-root",
            str(tmp_path / "store"),
            "--reports-dir",
            str(tmp_path / "dry_reports"),
            "--trials-root",
            str(tmp_path / "dry_trials"),
            "--as-of",
            NOW.isoformat(),
            "--dry-run",
        ]
    )
    assert code == 0
    assert not (tmp_path / "dry_reports").exists()
    assert not (tmp_path / "dry_trials").exists()


# ─────────────────────────────────────────────────────────────────────────────
# A11 - produccion intacta y escrituras acotadas a las raices que se le dan
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_a_run_writes_only_under_the_roots_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[Path] = []
    original = Path.write_text

    def spy(path: Path, data: str, *args: object, **kwargs: object) -> int:
        written.append(path)
        return original(path, data, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", spy)
    report = search.analyse(
        store=_store(tmp_path / "store"),
        reports_dir=tmp_path / "reports",
        trials_root=tmp_path / "trials",
        as_of=NOW,
        write=True,
    )
    # Nada se escribe fuera de las dos raices que se le dan: ni `runs/` ni el almacen real.
    assert written
    assert all(path.is_relative_to(tmp_path) for path in written)
    assert report.registry.n_trials == 11
    # La produccion no se toca: el candidato declarado y la declaracion siguen siendo los de #7.
    assert search.DECLARED_CANDIDATE is volatility_forecast.DECLARED_CANDIDATE
    assert search.DECLARED_CANDIDATE.candidate == "garch"
    source = Path(str(search.__file__)).read_text("utf-8")
    assert "DEFAULT_RUNS_ROOT" not in source
    assert 'DEFAULT_TRIALS_ROOT: Final[str] = "runs_volatility"' in source
    assert "runs_volatility/" in (REPO_ROOT / ".gitignore").read_text("utf-8")


@needs_store
def test_a11b_the_real_store_carries_the_term_structure_on_the_same_window() -> None:
    """La ingesta de A2 dejo las dos series en el almacen real, con la ventana de `^GSPC`."""
    store = Store(REAL_DATA)
    spans = store.sql(
        "SELECT series_id, min(as_of) AS first, max(as_of) AS last FROM raw.market_daily "
        "WHERE series_id IN ('^GSPC', '^VIX9D', '^VIX3M') GROUP BY 1"
    )
    by_series = {str(row["series_id"]): row for row in spans.iter_rows(named=True)}
    assert set(by_series) == {"^GSPC", "^VIX9D", "^VIX3M"}
    for series_id, earliest in (("^VIX9D", date(2011, 1, 1)), ("^VIX3M", date(2006, 7, 1))):
        assert by_series[series_id]["first"].date() >= earliest
        assert by_series[series_id]["last"].date() == by_series["^GSPC"]["last"].date()
    assert search.load_term_structure(store) is not None


# ─────────────────────────────────────────────────────────────────────────────
# A12 - puertas del CLI, caminos sin bloque y cobertura del modulo nuevo
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_cli_guards_declare_and_write_nothing(tmp_path: Path) -> None:
    """Sin `--as-of`, con un `--as-of` invalido o sin almacen: salida 2 y **nada** escrito."""
    store = _store(tmp_path / "store")
    assert search.main(["--data-root", str(tmp_path / "store")]) == 2
    assert search.main(["--data-root", str(tmp_path / "store"), "--as-of", "ayer"]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert search.main(["--data-root", str(empty), "--as-of", NOW.isoformat()]) == 2
    assert not (empty / "derived").exists()
    # Un almacen sin mercado diario no es un informe vacio: es error tipado, y no se escribe nada.
    with pytest.raises(ConfigurationError):
        search.analyse(
            store=Store(empty),
            reports_dir=tmp_path / "reports",
            trials_root=tmp_path / "trials",
            as_of=NOW,
            write=True,
        )
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "trials").exists()
    # Una muestra que no llega al minimo de entrenamiento tampoco: error tipado, nunca un cero.
    with pytest.raises(search.SpecSearchError):
        search.analyse(
            store=_store(tmp_path / "short", count=300),
            reports_dir=tmp_path / "reports_short",
            trials_root=tmp_path / "trials_short",
            as_of=NOW,
            write=True,
        )
    assert not (tmp_path / "reports_short").exists()
    # Sin la *term structure*, `har_vix_ts` se queda sin regresores y sin almacen de la misma no
    # hay serie que unir.
    assert search.load_term_structure(_store(tmp_path / "plain", with_term_structure=False)) is None
    assert search.load_term_structure(store) is not None
    # Una variante que no esta en el catalogo ni en #7 no se puede registrar.
    with pytest.raises(search.SpecSearchError):
        search.trial_spec("no_existe")
    assert search.best_new_spec([_row("harq", -10.0)], names=["otra"]) is None


def test_a12b_the_report_declares_a_missing_block_instead_of_inventing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con DSR y PBO `not_evaluable` el informe lo dice, y el veredicto no aprueba nada."""

    def no_deflated(*_args: object, **_kwargs: object) -> tuple[None, str]:
        return None, "DSR no disponible en esta prueba"

    def no_pbo(*_args: object, **_kwargs: object) -> tuple[None, str]:
        return None, "PBO no disponible en esta prueba"

    monkeypatch.setattr(search, "deflated_block", no_deflated)
    monkeypatch.setattr(search, "pbo_block", no_pbo)
    report = search.analyse(
        store=_store(tmp_path / "store"),
        reports_dir=tmp_path / "reports",
        trials_root=tmp_path / "trials",
        as_of=NOW,
        write=True,
    )
    correction = cast("Mapping[str, object]", report.payload["correction"])
    assert correction["deflated_sharpe_state"] == "not_evaluable"
    assert correction["deflated_sharpe"] is None
    assert correction["pbo_state"] == "not_evaluable"
    assert correction["pbo"] is None
    assert correction["trimmed_sessions"] is None
    assert correction["sessions_per_block"] is None
    verdict = cast("Mapping[str, object]", report.payload["verdict"])
    assert verdict["improves"] is False
    assert verdict["verdict"] in {search.VERDICT_KEEPS_7, search.VERDICT_INCONCLUSIVE}
    markdown = (tmp_path / "reports" / "volatility_spec_search_2026-10-11.md").read_text("utf-8")
    assert "**DSR no evaluable**" in markdown
    assert "**PBO no evaluable**" in markdown
    assert f"`{verdict['verdict']}`" in markdown


def test_a12c_extreme_values_are_formatted_compactly() -> None:
    """Un QLIKE desbordado no puede llenar la tabla del informe de trescientos digitos."""
    fmt = search._fmt  # pyright: ignore[reportPrivateUsage]
    assert fmt(5.9e281) == "5.900000e+281"
    assert fmt(-8.9805, digits=4) == "-8.9805"
    assert fmt(0.0) == "0.000000"
    assert fmt(12) == "12"
    assert fmt(True) == "si"
    assert fmt(None) == "n/d"
    assert fmt(float("nan")) == "n/d"
    assert fmt("texto") == "texto"
