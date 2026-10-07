"""Tests de la serie de referencia del listón **B** (#70).

Un test por criterio (A1-A14). Lo importante que se comprueba aquí:

- el identificador ``close_to_close_long`` (listón B) **no** entra en los seis baselines de
  #14 y ``baselines`` no lo expone (A2: B «por el motor» sigue declarado como frontera de #70);
- el retorno ``gross_pct`` es ``100 · (close / close_previo - 1)``, calculado **a mano** (A3);
- la financiación por sesión es la **tenencia declarada por noche** del ``CostModel`` (una
  noche por sesión) y ``net_pct = gross_pct - financing_pct`` (A4, A5);
- ``declared_hold_cost`` es el ``CostBreakdown`` de #11 con ``nights = nights_total`` (el
  diferencial **una vez**), y bajo el supuesto de #64 ``c_total`` queda ``null`` (A6);
- la primera sesión sin ``starting_close_px`` **se declara** (``None``), no se inventa (A7);
- el módulo es **puro** (biblioteca estándar y los contratos de #11/#13) y **determinista**
  byte a byte, también entre procesos con ``PYTHONHASHSEED`` distinto (A9, A11);
- la equivalencia de financiación (§11.2 6,66 %/año vs #8 0,0182 %/noche) se **declara**, y el
  módulo usa el valor por noche (A12).

Los importes declarados se comprueban **contra la tabla de #11**, no contra el resultado del
módulo.
"""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
import textwrap
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from cfdtrader.backtest import baselines, close_to_close
from cfdtrader.backtest.close_to_close import (
    CLOSE_TO_CLOSE,
    CLOSE_TO_CLOSE_DOES_NOT_DO,
    FINANCING_EQUIVALENCE,
    FOLLOW_UPS,
    LIMITATIONS,
    NIGHTS_PER_SESSION,
    OVERNIGHT_REASON,
    CloseToCloseError,
    InvalidCloseToCloseParameterError,
    ReferenceSeries,
    close_to_close_series,
    declared_financing_cut,
)
from cfdtrader.backtest.costs import (
    CostModel,
    FinancingCut,
    Side,
    SlippageParameter,
    cost_breakdown,
    declared_cost_model,
    declared_slippage_assumption,
)
from cfdtrader.backtest.engine import SessionInput

MODULE_PATH = Path(str(close_to_close.__file__))
SOURCE = MODULE_PATH.read_text(encoding="utf-8")

REQUIRED_PUBLIC = (
    "CLOSE_TO_CLOSE",
    "CLOSE_TO_CLOSE_DOES_NOT_DO",
    "FINANCING_EQUIVALENCE",
    "FOLLOW_UPS",
    "LIMITATIONS",
    "NIGHTS_PER_SESSION",
    "OVERNIGHT_REASON",
    "CloseToCloseError",
    "InvalidCloseToCloseParameterError",
    "ReferenceRow",
    "ReferenceSeries",
    "close_to_close_series",
    "declared_financing_cut",
)

#: El modelo declarado de #11 (la tabla de §3.3) y el supuesto de *slippage* de #64.
DECLARED = declared_cost_model()
ASSUMED = declared_slippage_assumption()
NOTIONAL = Decimal("10000")

_SECOND_PROCESS_SCRIPT = textwrap.dedent(
    """
    from datetime import date, timedelta
    from decimal import Decimal

    from cfdtrader.backtest.close_to_close import close_to_close_series, declared_financing_cut
    from cfdtrader.backtest.costs import declared_cost_model, declared_slippage_assumption
    from cfdtrader.backtest.engine import SessionInput

    days = []
    day = date(2026, 1, 5)
    while len(days) < 5:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    inputs = [
        SessionInput(session=item, open_px=195.0 + index, high_px=196.0 + index,
                     low_px=194.0 + index, close_px=195.5 + index)
        for index, item in enumerate(days)
    ]
    series = close_to_close_series(
        inputs,
        notional_usd=Decimal("10000"),
        cost_model=declared_cost_model(),
        slippage=declared_slippage_assumption(),
        financing_cut=declared_financing_cut(),
        starting_close_px=195.0,
    )
    print(series.series_sha256)
    """
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades del test
# ─────────────────────────────────────────────────────────────────────────────
def business_sessions(count: int) -> list[date]:
    """``count`` sesiones consecutivas de lunes a viernes, desde el 2026-01-05."""
    days: list[date] = []
    day = date(2026, 1, 5)
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def inputs_for(closes: list[float | None]) -> list[SessionInput]:
    """Un ``SessionInput`` por cierre declarado; ``open``/``high``/``low`` son de relleno."""
    days = business_sessions(len(closes))
    return [
        SessionInput(
            session=days[index],
            open_px=None if close is None else close,
            high_px=None if close is None else close,
            low_px=None if close is None else close,
            close_px=close,
        )
        for index, close in enumerate(closes)
    ]


def series_for(
    closes: list[float | None],
    *,
    starting: float | None = None,
    notional: Decimal = NOTIONAL,
    cost_model: CostModel = DECLARED,
    slippage: SlippageParameter = ASSUMED,
    financing_cut: FinancingCut | None = None,
) -> ReferenceSeries:
    """Construye la serie con el corte **ya verificado** de #87 salvo que se declare otro."""
    return close_to_close_series(
        inputs_for(closes),
        notional_usd=notional,
        cost_model=cost_model,
        slippage=slippage,
        financing_cut=declared_financing_cut() if financing_cut is None else financing_cut,
        starting_close_px=starting,
    )


def _imported_modules() -> set[str]:
    """Módulos importados por el módulo (para demostrar que A11 se cumple)."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def _code_only() -> str:
    """El código del módulo sin **ningún** literal de cadena (ni docstrings ni declaraciones)."""
    masked: set[int] = set()
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            masked.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return "\n".join(
        "" if number in masked else row for number, row in enumerate(SOURCE.splitlines(), start=1)
    )


# ─────────────────────────────────────────────────────────────────────────────
# A1 — fichero y API pública
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_module_file_and_public_api() -> None:
    assert MODULE_PATH.is_file()
    assert MODULE_PATH.name == "close_to_close.py"
    assert MODULE_PATH.parent.name == "backtest"
    assert (Path(__file__).resolve().parents[1] / "tests" / "test_close_to_close.py").is_file()
    for name in REQUIRED_PUBLIC:
        assert name in close_to_close.__all__, f"{name} falta en __all__"
    assert issubclass(CloseToCloseError, Exception)
    assert issubclass(InvalidCloseToCloseParameterError, CloseToCloseError)
    assert InvalidCloseToCloseParameterError is not CloseToCloseError
    assert callable(close_to_close_series)
    assert callable(declared_financing_cut)
    assert NIGHTS_PER_SESSION == 1
    assert CLOSE_TO_CLOSE == "close_to_close_long"


# ─────────────────────────────────────────────────────────────────────────────
# A2 — B no es un baseline de #14: no entra en BASELINE_IDS ni lo expone baselines
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_b_is_not_one_of_the_six_baselines() -> None:
    assert CLOSE_TO_CLOSE not in baselines.BASELINE_IDS
    assert len(baselines.BASELINE_IDS) == 6
    for token in ("close", "overnight", "index", "hold"):
        for name in baselines.__all__:
            assert token not in name.lower(), name
    for candidate in (CLOSE_TO_CLOSE, "close_to_close", "buy_and_hold"):
        with pytest.raises(baselines.InvalidBaselineParameterError):
            baselines.run_baseline(
                inputs_for([100.0, 101.0, 102.0]),
                split_plan=cast("Any", None),
                cost_model=DECLARED,
                slippage=ASSUMED,
                notional_usd=NOTIONAL,
                baseline=candidate,
            )


# ─────────────────────────────────────────────────────────────────────────────
# A3 — gross_pct es cierre a cierre, calculado a mano
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_gross_close_to_close_is_hand_computed() -> None:
    series = series_for([200.0, 202.0, 201.0], starting=200.0)
    assert series.gross_pct == (
        0.0,
        100.0 * (202.0 / 200.0 - 1.0),
        100.0 * (201.0 / 202.0 - 1.0),
    )
    assert series.rows[1].previous_close_px == 200.0
    assert series.rows[1].close_px == 202.0
    assert series.nights_total == 3
    assert all(row.nights == NIGHTS_PER_SESSION for row in series.rows)


# ─────────────────────────────────────────────────────────────────────────────
# A4 — la financiación es la tenencia declarada por noche; net = gross - financing
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_financing_is_the_declared_carry_and_net_subtracts_it() -> None:
    carry = Decimal(cast("Any", DECLARED.carry_long_pct_per_night))
    assert carry == Decimal("0.0182")
    series = series_for([200.0, 202.0], starting=200.0)
    assert series.financing_pct_per_night == "0.0182"
    for row in series.rows:
        assert row.financing_pct == float(carry)
        assert row.gross_pct is not None and row.net_pct is not None
        assert row.net_pct == row.gross_pct - float(carry)
    assert series.net_pct == tuple(
        cast("float", row.gross_pct) - float(carry) for row in series.rows
    )


# ─────────────────────────────────────────────────────────────────────────────
# A5 — una noche por sesión; el diferencial se cobra UNA vez en declared_hold_cost
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_one_night_per_session_and_spread_charged_once() -> None:
    series = series_for([100.0, 101.0, 102.0, 103.0], starting=100.0)
    cost = series.declared_hold_cost
    assert cost.nights == series.nights_total == 4
    assert cost.side is Side.LONG
    assert cost.overnight_reason is not None and "listón B" in cost.overnight_reason
    # Diferencial **una** vez (dos mitades), no una por sesión.
    assert cost.spread_pct == Decimal("0.0042")
    # Tenencia = por-noche × noches (exacta).
    assert cost.carry_pct_per_night == Decimal("0.0182")
    assert cost.carry_pct == Decimal("0.0182") * 4 == Decimal("0.0728")


# ─────────────────────────────────────────────────────────────────────────────
# A6 — declared_hold_cost es el CostBreakdown de #11, y c_total sigue null (supuesto #64)
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_hold_cost_is_the_number_11_breakdown_and_total_stays_null() -> None:
    series = series_for([100.0, 110.0, 121.0], starting=100.0)
    expected = cost_breakdown(
        model=DECLARED,
        slippage=ASSUMED,
        notional_usd=NOTIONAL,
        side=Side.LONG,
        nights=3,
        overnight_reason=OVERNIGHT_REASON,
        financing_cut=declared_financing_cut(),
    )
    assert series.declared_hold_cost == expected
    assert series.declared_hold_cost.c_total_pct is None
    assert series.declared_hold_cost.c_declared_pct is not None
    # No se rellena el hueco con 0: el supuesto no es una medición.
    assert series.declared_hold_cost.slippage.state == "assumed"
    assert series.declared_hold_cost.slippage.is_measurement is False


# ─────────────────────────────────────────────────────────────────────────────
# A7 — la primera sesión sin cierre previo se declara, no se inventa
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_first_session_without_starting_close_is_declared() -> None:
    series = series_for([200.0, 202.0])
    first = series.rows[0]
    assert first.previous_close_px is None
    assert first.gross_pct is None and first.net_pct is None and first.nights == 0
    assert first.note is not None and "cierre previo" in first.note
    assert series.gross_pct[0] is None
    assert series.nights_total == 1
    # El coste declarado solo paga la noche que existe.
    assert series.declared_hold_cost.nights == 1
    # Con el cierre previo declarado, la fila sí se calcula.
    with_start = series_for([200.0, 202.0], starting=200.0)
    assert with_start.rows[0].gross_pct == 0.0
    assert with_start.nights_total == 2


# ─────────────────────────────────────────────────────────────────────────────
# A8 — una sesión sin cierre se declara (None), nunca se rellena con 0
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_a_session_without_close_is_declared() -> None:
    series = series_for([200.0, None, 204.0], starting=200.0)
    middle = series.rows[1]
    assert middle.close_px is None
    assert middle.gross_pct is None and middle.net_pct is None and middle.nights == 0
    assert middle.note is not None and "sin cierre" in middle.note
    # La siguiente sesión pierde su cierre previo: también se declara.
    after = series.rows[2]
    assert after.previous_close_px is None
    assert after.gross_pct is None and "cierre previo" in cast("str", after.note)
    assert series.nights_total == 1
    assert series.declared_hold_cost.nights == 1


# ─────────────────────────────────────────────────────────────────────────────
# A9 — determinismo byte a byte, también entre procesos con PYTHONHASHSEED distinto
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_the_series_is_deterministic_across_processes() -> None:
    first = series_for([195.5, 196.5, 197.5, 198.5, 199.5], starting=195.0)
    again = series_for([195.5, 196.5, 197.5, 198.5, 199.5], starting=195.0)
    assert first.series_sha256 == again.series_sha256
    assert first.series_sha256.startswith("sha256:")
    assert len(first.series_sha256) == len("sha256:") + 64
    env = dict(os.environ, PYTHONHASHSEED="12345")
    completed = subprocess.run(  # noqa: S603 - reproduccion controlada del escenario
        [sys.executable, "-c", _SECOND_PROCESS_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert completed.stdout.strip() == first.series_sha256


# ─────────────────────────────────────────────────────────────────────────────
# A10 — entradas no declarables: error tipado, nunca un resultado silencioso
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_invalid_parameters_raise_a_typed_error() -> None:
    good = inputs_for([200.0, 202.0])
    cut = declared_financing_cut()
    with pytest.raises(InvalidCloseToCloseParameterError):
        close_to_close_series(
            cast("Any", "nfo"),
            notional_usd=NOTIONAL,
            cost_model=DECLARED,
            slippage=ASSUMED,
            financing_cut=cut,
        )
    with pytest.raises(InvalidCloseToCloseParameterError):
        close_to_close_series(
            [],
            notional_usd=NOTIONAL,
            cost_model=DECLARED,
            slippage=ASSUMED,
            financing_cut=cut,
        )
    with pytest.raises(InvalidCloseToCloseParameterError):
        close_to_close_series(
            cast("Any", [object()]),
            notional_usd=NOTIONAL,
            cost_model=DECLARED,
            slippage=ASSUMED,
            financing_cut=cut,
        )
    duplicated = [good[0], good[0]]
    with pytest.raises(InvalidCloseToCloseParameterError):
        close_to_close_series(
            duplicated,
            notional_usd=NOTIONAL,
            cost_model=DECLARED,
            slippage=ASSUMED,
            financing_cut=cut,
        )
    for bad_notional in (cast("Any", 10000.0), cast("Any", 0), Decimal("-1")):
        with pytest.raises(InvalidCloseToCloseParameterError):
            close_to_close_series(
                good,
                notional_usd=bad_notional,
                cost_model=DECLARED,
                slippage=ASSUMED,
                financing_cut=cut,
            )
    base: dict[str, Any] = {
        "notional_usd": NOTIONAL,
        "cost_model": DECLARED,
        "slippage": ASSUMED,
        "financing_cut": cut,
    }
    for override in ("cost_model", "slippage", "financing_cut"):
        with pytest.raises(InvalidCloseToCloseParameterError):
            close_to_close_series(good, **cast("Any", {**base, override: "x"}))
    for bad_start in (0.0, -3.0, cast("Any", "x")):
        with pytest.raises(InvalidCloseToCloseParameterError):
            close_to_close_series(
                good,
                notional_usd=NOTIONAL,
                cost_model=DECLARED,
                slippage=ASSUMED,
                financing_cut=cut,
                starting_close_px=bad_start,
            )


# ─────────────────────────────────────────────────────────────────────────────
# A11 — el módulo es puro: biblioteca estándar y los contratos de #11/#13
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_the_module_is_pure_stdlib_and_touches_no_disk() -> None:
    allowed = {
        "__future__",
        "hashlib",
        "json",
        "collections.abc",
        "dataclasses",
        "datetime",
        "decimal",
        "typing",
        "cfdtrader.backtest.costs",
        "cfdtrader.backtest.engine",
    }
    assert _imported_modules() <= allowed, _imported_modules() - allowed
    for forbidden in (
        "polars",
        "duckdb",
        "numpy",
        "pandas",
        "sklearn",
        "cfdtrader.analysis",
        "cfdtrader.data",
        "cfdtrader.agents",
    ):
        assert forbidden not in _imported_modules()
        assert f"import {forbidden}" not in SOURCE
    code = _code_only()
    for token in (
        "datetime.now",
        "utcnow",
        "date.today",
        "time.time",
        "uuid",
        "Store(",
        "Store.",
        "read_pit",
        "open(",
        "Path(",
        "config/",
        "run_walk_forward",
    ):
        assert token not in code, token


# ─────────────────────────────────────────────────────────────────────────────
# A12 — la equivalencia de financiación se DECLARA (no se deriva)
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_the_financing_equivalence_is_declared() -> None:
    assert FINANCING_EQUIVALENCE["used"] == "per_night"
    assert FINANCING_EQUIVALENCE["derived_from_daily"] == "false"
    assert FINANCING_EQUIVALENCE["per_night_pct"] == "0.0182"
    assert FINANCING_EQUIVALENCE["per_year_pct"] == "6.6647"
    assert FINANCING_EQUIVALENCE["plan_112_pct"] == "6.66"
    # No se deriva: 365 noches del valor por-noche no dan la anualización declarada.
    assert Decimal("0.0182") * 365 != Decimal("6.6647")
    series = series_for([200.0, 202.0], starting=200.0)
    assert series.financing_equivalence["per_night_pct"] == series.financing_pct_per_night
    assert series.financing_cut.state == "measured"
    assert series.financing_cut.cut_et is not None
    assert series.financing_cut.cut_et.hour == 17


# ─────────────────────────────────────────────────────────────────────────────
# A13 — las fronteras se declaran con su issue, legibles por máquina
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_the_frontiers_are_declared_with_their_issue() -> None:
    for table in (CLOSE_TO_CLOSE_DOES_NOT_DO, FOLLOW_UPS):
        assert isinstance(table, tuple) and table
        for entry in table:
            assert str(entry["issue"]).startswith("#")
            assert len(entry) >= 3
    frontiers = {entry["issue"] for entry in CLOSE_TO_CLOSE_DOES_NOT_DO}
    for issue in ("#13", "#15", "#18", "#28", "#69", "#70"):
        assert issue in frontiers, issue
    follow_ups = {entry["issue"] for entry in FOLLOW_UPS}
    for issue in ("#15", "#18", "#69"):
        assert issue in follow_ups, issue
    assert isinstance(LIMITATIONS, tuple) and LIMITATIONS
    doc = close_to_close.__doc__ or ""
    for token in ("**B**", "§11.2", "#70", "regla 6", "#87", "#13", "#11"):
        assert token in doc, f"el docstring no traza {token}"


# ─────────────────────────────────────────────────────────────────────────────
# A14 — la frontera de #70 («B por el motor») queda declarada, no fingida
# ─────────────────────────────────────────────────────────────────────────────
def test_a14_the_engine_overnight_frontier_is_declared() -> None:
    statement = " ".join(entry["statement"] for entry in CLOSE_TO_CLOSE_DOES_NOT_DO).lower()
    assert "por el motor" in statement
    assert "regla 6" in statement
    assert any("overnight" in item.lower() or "listón b" in item.lower() for item in LIMITATIONS)
    # El módulo usa la tenencia por noche (una por sesión), sin reescribir la tabla de #11.
    assert NIGHTS_PER_SESSION == 1
    assert "listón B" in OVERNIGHT_REASON
    assert "17:00" in OVERNIGHT_REASON
    # No es un motor: no expone ni usa la corrida de #13.
    assert "run_walk_forward" not in close_to_close.__all__
    assert not hasattr(close_to_close, "run_walk_forward")
    assert isinstance(inspect.getsource(close_to_close), str)
