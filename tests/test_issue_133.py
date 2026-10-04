"""Tests del contrato de #133: la lectura **neta** bajo el supuesto declarado (20 bp).

El informe del pipeline publicaba ``net_metrics: not_computable`` culpando a #60 (cerrada). Desde
#133 publica el neto **bajo el supuesto declarado** ya cuantificado por el `R` de #60, etiquetado
con su estado (``assumed``, ``is_measurement = false``) y su valor (20 bp), y **sin** tocar las
decisiones de ningún brazo ni ningún artefacto ya publicado.

Los tests caros viven en ``tests/test_pipeline_report.py`` (sobre su fixture del almacén real); aquí
se fija la **derivación** del supuesto, la aritmética de la serie neta y la etiqueta del bloque.
"""

from __future__ import annotations

import subprocess
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Final

from cfdtrader.analysis import pipeline_report
from cfdtrader.analysis.pipeline_report import (
    ASSUMED_SLIPPAGE_PCT,
    BASIS_DECLARED_COST_WITH_ASSUMED_SLIPPAGE,
    NET_METRICS_STATE,
    SCENARIO_R_PCT,
)
from cfdtrader.backtest.costs import SlippageParameter, declared_slippage_assumption

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SOURCE: Final[Path] = REPO_ROOT / "src" / "cfdtrader" / "analysis" / "pipeline_report.py"
PHASE1_JSON: Final[Path] = (
    REPO_ROOT / "data" / "derived" / "reports" / "phase1_backtest_2026-09-19.json"
)

#: Los tres helpers privados que fijan la aritmetica y la etiqueta (por su nombre).
_NET_OF_RUN = pipeline_report._net_series_of_run  # pyright: ignore[reportPrivateUsage]
_NET_RETURN = pipeline_report._net_return_pct  # pyright: ignore[reportPrivateUsage]
_NET_PAYLOAD = pipeline_report._net_series_payload  # pyright: ignore[reportPrivateUsage]
_DECLARED_SERIES = pipeline_report._series_of_run  # pyright: ignore[reportPrivateUsage]
_DECLARED_PAYLOAD = pipeline_report._declared_series_payload  # pyright: ignore[reportPrivateUsage]

_STATUS_NO_TRADE: Final[str] = "no_trade"
_STATUS_SKIPPED: Final[str] = "skipped"
_STATUS_TRADED: Final[str] = "traded"


def _outcome(*, gross: float, declared_pct: str, status: str = _STATUS_TRADED) -> SimpleNamespace:
    """Una sesion minima con lo que la derivacion necesita (no hace falta el motor entero)."""
    return SimpleNamespace(
        session=date(2026, 9, 17),
        status=status,
        gross=gross,
        cost=SimpleNamespace(c_declared_pct=Decimal(declared_pct)),
    )


def _run(outcomes: list[SimpleNamespace]) -> SimpleNamespace:
    """Un ``BacktestRun`` minimo: ``_sessions_of_run`` solo recorre ``folds[].sessions``."""
    return SimpleNamespace(folds=[SimpleNamespace(sessions=tuple(outcomes))])


# ─────────────────────────────────────────────────────────────────────────────
# A1 · El supuesto se **deriva**, no se teclea
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_assumed_slippage_is_derived_not_typed() -> None:
    assumption = declared_slippage_assumption()
    expected = Decimal(str(assumption.pct_of_r)) * SCENARIO_R_PCT / Decimal(100)
    assert expected == ASSUMED_SLIPPAGE_PCT
    assert Decimal("0.2") == ASSUMED_SLIPPAGE_PCT
    source = SOURCE.read_text(encoding="utf-8")
    assert 'Decimal("0.2")' not in source, "una cifra de coste tecleada en el modulo"
    assert "ASSUMED_SLIPPAGE_PCT" in source


# ─────────────────────────────────────────────────────────────────────────────
# A2 · La aritmetica de la serie neta
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_net_return_is_the_declared_one_minus_the_assumption() -> None:
    traded = _outcome(gross=0.01, declared_pct="0.0042")
    assert _NET_RETURN(traded) == 100.0 * 0.01 - 0.0042 - 0.2


def test_a2_a_no_trade_session_is_zero_in_both_series_and_skipped_is_out() -> None:
    run = _run(
        [
            _outcome(gross=0.01, declared_pct="0.0042"),
            _outcome(gross=0.0, declared_pct="0.0042", status=_STATUS_NO_TRADE),
            _outcome(gross=0.02, declared_pct="0.0042", status=_STATUS_SKIPPED),
        ]
    )
    net = _NET_OF_RUN(run)
    declared = _DECLARED_SERIES(run)
    assert len(net) == len(declared) == 2, "las `skipped` no entran en ninguna de las dos series"
    assert net[1] == 0.0 == declared[1], "sin operacion no hay coste que cobrar"
    assert net[0] == declared[0] - float(ASSUMED_SLIPPAGE_PCT)
    assert net[0] < declared[0], "el supuesto **encarece**: la lectura neta es menor"


# ─────────────────────────────────────────────────────────────────────────────
# A3 · El bloque neto lleva la etiqueta del supuesto (nunca una medicion)
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_the_net_block_is_labelled_as_an_assumption() -> None:
    block = _NET_PAYLOAD((0.5, -0.2))
    assert block["basis"] == BASIS_DECLARED_COST_WITH_ASSUMED_SLIPPAGE
    assert block["slippage_state"] == "assumed"
    assert block["is_measurement"] is False
    assert block["assumed_slippage_pct"] == "0.2"
    assert block["slippage_pct_of_r"] == "20"
    assert block["r_pct"] == "1"
    assert block["assumption_issue"] == "#64"
    assert block["r_issue"] == "#60"
    assert block["measuring_issue"] == "#62"
    # El cuerpo y su digest son los mismos que los de la serie declarada: misma convencion.
    declared = _DECLARED_PAYLOAD((0.5, -0.2))
    assert {key: block[key] for key in declared} == declared
    assert set(declared) < set(block)


def test_a3_the_state_is_no_longer_not_computable_and_stops_blaming_60() -> None:
    assert NET_METRICS_STATE == "computed_under_declared_assumption"
    reason = pipeline_report.NET_METRICS_REASON
    assert "#62" in reason and "#88" in reason
    assert "no se puede cobrar sin" not in reason, "la razon obsoleta que culpaba a #60"
    follow_ups = [entry["issue"] for entry in pipeline_report.FOLLOW_UPS]
    assert "#88" in follow_ups
    assert "#60" not in follow_ups, "#60 esta cerrada: no es un seguimiento vivo"
    # El bloque publicado declara los dos: lo comprueba, sobre el almacen real, el A9 de
    # `tests/test_pipeline_report.py` (`net["follow_ups"] == ["#62", "#88"]`).


def test_a3_the_informe_still_does_not_quantify_the_model() -> None:
    assumption = declared_slippage_assumption()
    assert isinstance(assumption, SlippageParameter)
    assert assumption.is_measurement is False
    assert assumption.pct_of_notional is None, "el informe **deriva**; no cuantifica el supuesto"
    assert assumption.r_pct is None


# ─────────────────────────────────────────────────────────────────────────────
# A9 · Ningun artefacto publicado queda invalidado por este cambio
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_no_published_artifact_is_invalidated(tmp_path: Path) -> None:
    """El `phase1_backtest` publicado se regenera **byte a byte** (arnes de #29)."""
    if not PHASE1_JSON.is_file():
        return
    result = subprocess.run(  # noqa: S603 - el `uv` del entorno, uso fijo
        [  # noqa: S607 - el `uv` del entorno, uso fijo
            "uv",
            "run",
            "python",
            "-m",
            "cfdtrader.analysis.backtest_report",
            "--data-root",
            str(REPO_ROOT / "data"),
            "--reports-dir",
            str(tmp_path),
            "--as-of",
            "2026-09-19T00:00:00+00:00",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    produced = tmp_path / PHASE1_JSON.name
    assert produced.read_bytes() == PHASE1_JSON.read_bytes()
    assert produced.with_suffix(".md").read_bytes() == PHASE1_JSON.with_suffix(".md").read_bytes()
