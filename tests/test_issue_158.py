"""Tests del informe y del registro del espejo corto (#158, seguimiento de #78).

Todo lo que se escribe va a ``tmp_path``: el ``data/`` del repositorio se lee en **solo lectura**
(la fixture de sesion lo huella antes y despues) y el ``runs/`` del repositorio no se toca.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final, cast

import polars as pl
import pytest

from cfdtrader.analysis import mirror_report
from cfdtrader.analysis.experiment_log import (
    ExperimentConfig,
    ExperimentRecord,
    ExperimentResult,
    Registry,
    run_sha256,
)
from cfdtrader.backtest.engine import BacktestRun, DecisionError, Direction, SessionView
from cfdtrader.backtest.splits import SplitPlan
from cfdtrader.data.store import Store
from cfdtrader.models.baseline import BaselineModel, DesignFrame
from cfdtrader.models.mirror import (
    MIRROR_FEATURES,
    MIRROR_TARGET_COLUMN,
    MIRROR_TARGET_RULE,
    MIRROR_VARIANT_ID,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"

needs_store = pytest.mark.skipif(
    not (REAL_DATA / "derived" / "labels").exists(),
    reason="el almacen real no esta en el arbol: el fin a fin se mide sobre el (en solo lectura)",
)


def _business_days(count: int) -> list[date]:
    days: list[date] = []
    current = date(2026, 1, 5)
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _design(*, rows: int = 4, positives: int = 2) -> DesignFrame:
    """Un frame de diseno minimo del objetivo corto (para el render, sin ajustar nada)."""
    days = _business_days(rows)
    frame = pl.DataFrame(
        {
            "session": days,
            "y": [1] * positives + [0] * (rows - positives),
            **{name: [0.0] * rows for name in MIRROR_FEATURES},
        }
    )
    return DesignFrame(
        frame=frame,
        n_sessions=rows,
        n_labels=rows,
        n_shifted_rows=0,
        n_nulls_in_features=0,
        design_lag_sessions=1,
        features=MIRROR_FEATURES,
        design_lag_by_feature=tuple((name, 1) for name in MIRROR_FEATURES),
        target_column=MIRROR_TARGET_COLUMN,
    )


def _view(probability: object) -> SessionView:
    """La vista que ve el decider: la probabilidad viaja en la carga opaca."""
    return SessionView(session=date(2026, 1, 5), open_px=100.0, gap_px=None, context=probability)


def _report() -> mirror_report.MirrorReport:
    """Un informe construido a mano (objetos minimos) para probar el payload y el Markdown."""
    config = ExperimentConfig(
        variant_id=MIRROR_VARIANT_ID,
        features=MIRROR_FEATURES,
        hyperparameters={"solver": "saga"},
        seed=1,
        series_id="^GSPC",
        window={"direction": "short", "target_rule": MIRROR_TARGET_RULE},
    )
    result = ExperimentResult(sharpe_per_session=0.0105, n_observations=3)
    record = ExperimentRecord(
        as_of=datetime(2026, 10, 11, 22, tzinfo=UTC),
        run_sha256="a" * 64,
        directory=Path("/no/existe") / ("a" * 64),
        config=config,
        result=result,
        outcome=None,
        written=False,
    )
    payload: dict[str, object] = {"variant_id": MIRROR_VARIANT_ID}
    return mirror_report.MirrorReport(
        as_of=datetime(2026, 10, 11, 22, tzinfo=UTC),
        report_date=date(2026, 10, 11),
        payload=payload,
        report_sha256=mirror_report._report_digest(payload),  # pyright: ignore[reportPrivateUsage]
        design=_design(),
        plan=SplitPlan(
            n_sessions=4,
            inputs={},
            folds=(),
            uncovered=(),
            purge_total=0,
            embargo_total=0,
            embargo_in_train_total=0,
            exclusions_are_no_op=True,
            plan_sha256="plan-de-prueba",
        ),
        model=BaselineModel(features=MIRROR_FEATURES, hyperparameters={}, seed=1, folds=()),
        run=BacktestRun(
            folds=(),
            plan_sha256="plan-de-prueba",
            purge_total=0,
            embargo_total=0,
            embargo_in_train_total=0,
            exclusions_are_no_op=True,
            uncovered=(),
            not_in_any_test=0,
            n_sessions=4,
            traded=3,
            no_trade=1,
            skipped=0,
            run_sha256="a" * 64,
            report={},
        ),
        record=record,
        registry=Registry(entries=(), registry_sha256="b" * 64),
        config=config,
        result=result,
    )


def test_c1_the_decider_goes_short_at_the_declared_threshold() -> None:
    """A1: el decider del espejo declara `SHORT` con `p_cal >= 0,5` y `NOTHING` por debajo."""
    decide = mirror_report.mirror_decider(0)
    short = decide(_view(0.6))
    assert short.direction is Direction.SHORT
    assert short.notional_usd is not None
    assert short.probability == pytest.approx(0.6)
    assert decide(_view(0.5)).direction is Direction.SHORT
    assert decide(_view(0.499)).direction is Direction.NOTHING


def test_c2_the_decider_declares_what_it_cannot_read() -> None:
    """A2: sin probabilidad no se decide (``NOTHING`` con su motivo), y fuera de rango es error."""
    decide = mirror_report.mirror_decider(0)
    missing = decide(_view(None))
    assert missing.direction is Direction.NOTHING
    assert missing.probability is None
    # Un booleano **no** es una probabilidad: la vista no trae un numero.
    assert decide(_view(True)).direction is Direction.NOTHING
    with pytest.raises(DecisionError):
        decide(_view(1.5))


def test_c3_the_report_publishes_the_objective_and_the_cost_asymmetry() -> None:
    """A3: el informe declara el objetivo corto, la asimetria de coste y que no es un signo."""
    markdown = mirror_report.render_markdown(_report())
    assert MIRROR_VARIANT_ID in markdown
    assert MIRROR_TARGET_RULE in markdown
    assert mirror_report.NOT_A_SIGN_FLIP_NOTE in markdown
    assert mirror_report.COST_ASYMMETRY_NOTE in markdown
    assert mirror_report.REGISTRY_COUPLING_NOTE in markdown
    assert "no hay edge demostrado" in markdown.lower()


def test_c4_the_report_hash_is_self_consistent() -> None:
    """A4: el `report_sha256` es el del texto canonico sin esa clave, y el texto es estable."""
    report = _report()
    publish = json.loads(report.json_text())
    assert publish["report_sha256"] == report.report_sha256
    assert report.report_sha256.startswith("sha256:")
    assert report.json_text() == report.json_text()
    assert report.report_stem == "mirror_short_2026-10-11"


def test_c5_the_cli_requires_as_of_and_reports_an_unreadable_store(tmp_path: Path) -> None:
    """A5: sin `--as-of` (o con uno invalido) no se escribe nada; un almacen vacio es rc=2."""
    assert mirror_report.main([]) == 2
    assert mirror_report.main(["--as-of", "no-es-una-fecha"]) == 2
    code = mirror_report.main(
        [
            "--data-root",
            str(tmp_path / "vacio"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--as-of",
            "2026-10-11T22:00:00+00:00",
        ]
    )
    assert code == 2
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "runs").exists()


@needs_store
def test_c6_end_to_end_registers_the_mirror_with_its_own_identity(tmp_path: Path) -> None:
    """A6: el fin a fin registra el espejo con su `variant_id` propio y su propio `run_sha256`."""
    report = mirror_report.analyse(
        store=Store(REAL_DATA),
        reports_dir=tmp_path / "reports",
        runs_root=tmp_path / "runs",
        as_of=datetime(2026, 10, 11, 22, tzinfo=UTC),
        write=False,
    )
    assert report.record.config.variant_id == MIRROR_VARIANT_ID
    assert report.record.run_sha256 == run_sha256(report.config)
    assert report.config.window["target_rule"] == MIRROR_TARGET_RULE
    assert report.config.window["target_column"] == MIRROR_TARGET_COLUMN
    # La identidad es **propia**: con el `variant_id` de la larga el sha cambia.
    swapped = dataclasses.replace(report.config, variant_id="baseline_logit_elasticnet_v1")
    assert run_sha256(swapped) != report.record.run_sha256
    assert report.run.traded > 0
    assert report.result.n_observations == report.run.traded
    run_block = cast("dict[str, object]", report.payload["run"])
    objective_block = cast("dict[str, object]", report.payload["objective"])
    assert run_block["direction"] == "short"
    assert objective_block["rule"] == MIRROR_TARGET_RULE
    # `write=False` no escribe **nada** (ni el informe ni la carpeta del registro).
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "runs").exists()


@needs_store
def test_c7_end_to_end_write_publishes_the_report_and_the_run(tmp_path: Path) -> None:
    """A7: el CLI real (write) deja el informe con prefijo propio y la fila del registro."""
    code = mirror_report.main(
        [
            "--data-root",
            str(REAL_DATA),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--as-of",
            "2026-10-11T22:00:00+00:00",
        ]
    )
    assert code == 0
    reports = sorted((tmp_path / "reports").glob("mirror_short_*.json"))
    assert len(reports) == 1
    assert reports[0].name == "mirror_short_2026-10-11.json"
    document = json.loads(reports[0].read_text(encoding="utf-8"))
    registry_block = cast("dict[str, object]", document["registry"])
    run_sha = registry_block["run_sha256"]
    assert isinstance(run_sha, str)
    run_dir = tmp_path / "runs" / run_sha
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "result.json").is_file()
    assert (run_dir / "summary.md").is_file()
    markdown = reports[0].with_suffix(".md").read_text(encoding="utf-8")
    assert MIRROR_TARGET_RULE in markdown
    assert mirror_report.REGISTRY_COUPLING_NOTE in markdown
