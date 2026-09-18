"""Pruebas del informe de Fase 0 (`tasks.md`, tarea 9) — tarea #9.

Se cubren los 33 criterios de aceptación de la tarea, incluidos los **negativos**: un
artefacto ausente, ambiguo o contradictorio, y —sobre todo— el caso en que un
``not_evaluable`` intenta colarse como ``pass``.

Todos los artefactos sintéticos se escriben en ``tmp_path``: ningún test necesita los
informes reales de las tareas #6, #7 y #8 para pasar. El único test que los usa está
*skip*-gated, para poder comprobar la conclusión real sin que la suite dependa del árbol.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from cfdtrader.analysis import phase0_report
from cfdtrader.analysis.phase0_report import (
    FILE_SELECTION_RULE,
    GATE_AGGREGATION_RULE,
    HALF_A_MAPPING,
    MEASURE_KEYS,
    OPEN_DECISIONS,
    RECOMMENDATION_CONSISTENCY_RULE,
    AmbiguousArtifactError,
    GateVerdict,
    InputConflictError,
    MissingArtifactError,
    Phase0Inputs,
    Phase0ReportError,
    Recommendation,
    aggregate_gate,
    confidence_interval,
    consolidate,
    half_a_state,
    load_inputs,
    main,
    p_star,
    recommendation_is_consistent,
    render_markdown,
    required_operations,
    select_artifact,
    threshold_breached,
)

NOW = datetime(2026, 9, 18, tzinfo=UTC)
REPORT_STEM = "phase0_report_2026-09-18"
REPO_REPORTS = Path(__file__).resolve().parents[1] / "data" / "derived" / "reports"


# ─────────────────────────────────────────────────────────────────────────────
# Artefactos sintéticos
# ─────────────────────────────────────────────────────────────────────────────
def _patch(payload: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Aplica cambios por ruta con puntos; `lista[3]` para un índice de lista."""
    for path, value in updates.items():
        parts = path.split(".")
        node: Any = payload
        for part in parts[:-1]:
            key, _, index = part.partition("[")
            node = node[key]
            if index:
                node = node[int(index.rstrip("]"))]
        node[parts[-1]] = value
    return payload


def _segment(
    name: str,
    *,
    sessions: int,
    mean_bp: float,
    median_bp: float,
    std_bp: float,
    t_stat: float,
    p_value: float,
    hit_rate: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "sessions": sessions,
        "mean_bp": mean_bp,
        "median_bp": median_bp,
        "std_bp": std_bp,
        "t_stat": t_stat,
        "p_value": p_value,
        "hit_rate": hit_rate,
        "cumulative_bp": mean_bp * sessions,
    }


def drift_payload() -> dict[str, Any]:
    """Artefacto del drift con los valores de hoy (2026-09-18)."""
    return {
        "series_id": "^GSPC",
        "source": "yfinance",
        "first_session": "2005-01-04",
        "last_session": "2026-09-16",
        "sessions": 5459,
        "as_of": "2026-09-18 00:00:00+00:00",
        "segments": [
            _segment(
                "intraday",
                sessions=5459,
                mean_bp=2.47,
                median_bp=5.76,
                std_bp=105.3,
                t_stat=1.73,
                p_value=0.083,
                hit_rate=0.542,
            )
        ],
        "base_rate": {
            "sessions": 5459,
            "up_sessions": 2960,
            "up_share": 0.5422238505220737,
            "abs_move_median_bp": 44.77098345899044,
            "abs_move_p90_bp": 153.69341277472515,
            "abs_move_p99_bp": 389.21369021265957,
        },
        "difference": {"mean_difference_bp": 0.94, "t_stat": 0.67, "p_value": 0.5036},
        "verdict": "overnight",
        "stale_open_share": 0.10111742077303536,
        "open_quality": "degraded",
        "clean_from": "2014-01-01",
        "clean_sessions": 3192,
        "clean_segments": [
            _segment(
                "intraday",
                sessions=3192,
                mean_bp=1.9961071699776327,
                median_bp=4.485921889414435,
                std_bp=86.58007701357035,
                t_stat=1.3025587726526136,
                p_value=0.19281943803201562,
                hit_rate=0.5366541353383458,
            ),
            _segment(
                "overnight",
                sessions=3192,
                mean_bp=2.9094959219551573,
                median_bp=3.5951194855943225,
                std_bp=49.073873204994726,
                t_stat=3.34964558105888,
                p_value=0.0008185504141527522,
                hit_rate=0.5679824561403509,
            ),
            _segment(
                "total",
                sessions=3192,
                mean_bp=5.002826283840864,
                median_bp=8.34874609681637,
                std_bp=104.4162234482033,
                t_stat=2.6058466846821966,
                p_value=0.009346384598214695,
                hit_rate=0.5664160401002506,
            ),
        ],
        "clean_difference": {
            "mean_difference_bp": -0.9133887519775243,
            "t_stat": -0.5780097332780469,
            "p_value": 0.56329833331013,
        },
        "by_year": [],
        "by_weekday": [],
        "by_volatility": [],
        "limitations": ["medido sobre el índice, no sobre el CFD"],
        "notes": ["nota del artefacto"],
        "phase0_gate": "fail",
    }


def volatility_payload() -> dict[str, Any]:
    """Artefacto de volatilidad con los valores de hoy."""
    return {
        "as_of": "2026-09-18 00:00:00+00:00",
        "series_id": "^GSPC",
        "vix_series_id": "^VIX",
        "source": "yfinance",
        "targets": {"parkinson": {"immune_to_stale_open_52": True}},
        "sample": {
            "first_session": "2014-01-02",
            "last_session": "2026-09-16",
            "sessions": 5459,
            "stale_open_share": 0.10111742077303536,
            "clean_from": "2014-01-01",
            "clean_sessions": 3192,
            "by_year": [],
            "exclusions": {"stale_open": 3, "null_ohlc": 0, "half_day": 26, "total": 29},
            "exclusion_reasons": {},
        },
        "features": {
            "evaluated_sessions": 3166,
            "first_evaluated": "2016-01-07",
            "last_evaluated": "2026-09-16",
        },
        "walk_forward": {"window": "expanding", "min_train": 500, "folds": 127},
        "selection": {"selected": "garch", "verdict": "selected", "metric": "qlike"},
        "candidates": {},
        "folds": [],
        "rank_correlation": {"spearman": 1.0},
        "anchors": {
            "used_candidate": "garch",
            "median_forecast_sigma_bp": 65.99165226749633,
            "median_abs_open_close_bp": 40.86664837130438,
            "absolute_move_sessions": 3166,
            "per_candidate_sigma_bp": {"garch": 65.99165226749633},
        },
        "limitations": ["medido sobre el índice"],
        "notes": ["nota del artefacto"],
    }


def costs_payload() -> dict[str, Any]:
    """Artefacto de costes con los valores de hoy (plantilla vacía)."""
    return {
        "task": "#8",
        "artifact": "cost_audit",
        "as_of_utc": "2026-09-18T00:00:00+00:00",
        "template_path": "config/cost_observations.yaml",
        "session": {"day": "2026-09-18"},
        "confirmed_inputs": {},
        "fx_cost": {
            "state": "measured",
            "value_pct": "0",
            "reason": "nocional liquidado en USD: no hay conversión de divisa que costear",
        },
        "financing_cut": {
            "state": "unverified",
            "value_utc": None,
            "value_et": None,
            "reason": "no verificado a 2026-09-18: el documento no fija el instante de corte",
        },
        "declared_table": {
            "source": "plan.md §3.3 (documento del bróker recogido en el plan)",
            "reference_notional_usd": "10000",
            "holding_nights": 1,
            "rows": [
                {
                    "concept": "diferencial (spread) declarado",
                    "direction": None,
                    "per": "ida y vuelta (media al entrar + media al salir)",
                    "amount": {
                        "usd": "0.42",
                        "pct": "0.0042",
                        "notional_usd": "10000",
                    },
                },
                {
                    "concept": "tenencia declarada en CORTO",
                    "direction": "short",
                    "per": "por noche",
                    "amount": {"usd": "-0.18", "pct": "-0.0018", "notional_usd": "10000"},
                },
                {
                    "concept": "tenencia declarada en LARGO",
                    "direction": "long",
                    "per": "por noche",
                    "amount": {"usd": "1.82", "pct": "0.0182", "notional_usd": "10000"},
                },
            ],
            "round_trip": {
                "holding_nights": 1,
                "short": {
                    "direction": "short",
                    "formula": "0.42 $ + (-0.18 $/noche × 1 noche)",
                    "amount": {"usd": "0.24", "pct": "0.0024", "notional_usd": "10000"},
                },
                "long": {
                    "direction": "long",
                    "formula": "0.42 $ + (1.82 $/noche × 1 noche)",
                    "amount": {"usd": "2.24", "pct": "0.0224", "notional_usd": "10000"},
                },
            },
            "annualised": {
                "short": {"pct": "-0.6528", "ratio_annualised_over_per_night": "362.6667"},
                "long": {"pct": "6.6647", "ratio_annualised_over_per_night": "366.1923"},
            },
            "annualisation": {
                "pct": {"short": "-0.6528", "long": "6.6647"},
                "annualisation_consistent": True,
                "relative_difference": "0.009628",
                "tolerance": "0.01",
                "note": "las cifras anualizadas son declaradas: no se derivan de las diarias",
            },
            "note": "la tabla es el documento del bróker, no una medición",
        },
        "spread_cotizado": {
            "state": "unmeasured",
            "value_pct": None,
            "observations": 0,
            "reason": "no hay observaciones de bid/ask en la plantilla",
            "how_to_fill": "anotar `ask - bid` en los cinco instantes de `plan.md` §8.5",
        },
        "tracking_difference": {
            "state": "unmeasured",
            "value_pct": None,
            "observations": 0,
            "reason": "no hay pares CFD/índice en la plantilla",
            "how_to_fill": "anotar la cotización del CFD y la del índice",
        },
        "slippage_ejecucion": {
            "state": "unmeasured",
            "value_pct": None,
            "observations": 0,
            "reason": (
                "no existe ninguna ejecución real: a 2026-09-18 no se ha operado, así que el "
                "slippage no se ha medido y no se emite ningún valor de relleno"
            ),
            "how_to_fill": "anotar el precio obtenido frente al de referencia, 10-15 veces",
            "forbidden": "prohibido cualquier valor de relleno",
        },
        "by_size": {"percentage_is_constant": True},
        "phase0_gate_b": {
            "criterion": "slippage sistemático > ~20 % de R (tasks.md, tarea 9, puerta (b))",
            "threshold_pct_of_r": "20",
            "evaluable": False,
            "reason": (
                "el slippage de ejecución está sin medir (no existe ninguna ejecución real), "
                "así que la condición (b) no es evaluable con este artefacto"
            ),
            "verdict_owner": "#9",
            "note": "aquí solo se deja dicho si la condición es evaluable",
        },
        "broker_questions": [],
        "limitations": ["no hay ejecución real ⇒ el slippage no está medido"],
        "notes": ["los importes se publican como cadenas decimales exactas"],
    }


def write_artifacts(
    directory: Path,
    *,
    drift: dict[str, Any] | None = None,
    volatility: dict[str, Any] | None = None,
    costs: dict[str, Any] | None = None,
    date: str = "2026-09-18",
    drift_name: str | None = None,
    volatility_name: str | None = None,
    costs_name: str | None = None,
) -> Path:
    """Escribe los tres artefactos (los que se le pasen) en el directorio."""
    directory.mkdir(parents=True, exist_ok=True)
    items = (
        (drift_name or f"drift_decomposition_{date}.json", drift),
        (volatility_name or f"volatility_forecast_{date}.json", volatility),
        (costs_name or f"cost_audit_{date}.json", costs),
    )
    for name, payload in items:
        if payload is not None:
            (directory / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    return directory


def ready_inputs(directory: Path, **overrides: dict[str, Any]) -> Phase0Inputs:
    """Entradas cargadas de disco; `overrides` permite parchear cada artefacto."""
    drift = _patch(drift_payload(), overrides.get("drift", {}))
    volatility = _patch(volatility_payload(), overrides.get("volatility", {}))
    costs = _patch(costs_payload(), overrides.get("costs", {}))
    write_artifacts(directory, drift=drift, volatility=volatility, costs=costs)
    return load_inputs(directory)


def in_memory_inputs(**overrides: dict[str, Any]) -> Phase0Inputs:
    """Entradas construidas **en memoria**, sin tocar disco."""
    return Phase0Inputs.from_payloads(
        drift=_patch(drift_payload(), overrides.get("drift", {})),
        volatility=_patch(volatility_payload(), overrides.get("volatility", {})),
        costs=_patch(costs_payload(), overrides.get("costs", {})),
    )


def report_for(inputs: Phase0Inputs) -> dict[str, Any]:
    """Consolida y devuelve el *payload* del informe."""
    return consolidate(inputs, now=NOW).payload


def _repository_reports_fingerprint() -> dict[str, str]:
    """Huella del directorio de informes del repositorio, para detectar escrituras propias."""
    if not REPO_REPORTS.is_dir():
        return {}
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(REPO_REPORTS.iterdir())
        if path.is_file()
    }


def _cli_args(directory: Path, *extra: str) -> list[str]:
    return ["--data-root", str(directory), "--now", "2026-09-18T00:00:00+00:00", *extra]


def _cli_reports(directory: Path) -> Path:
    """Directorio donde el CLI busca (y escribe) los informes con `--data-root`."""
    return directory / "derived" / "reports"


# ─────────────────────────────────────────────────────────────────────────────
# A1 · consumo de los tres artefactos, regla de selección declarada
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_selection_rule_is_a_named_constant() -> None:
    assert FILE_SELECTION_RULE.startswith("de los ficheros de esa clase")
    assert "más reciente" in FILE_SELECTION_RULE
    assert phase0_report.ARTIFACT_DATE_FORMAT == "%Y-%m-%d"
    assert len(phase0_report.ARTIFACT_CLASSES) == 3


def test_a1_the_most_recent_artifact_of_each_class_is_used(tmp_path: Path) -> None:
    old = tmp_path / "old"
    write_artifacts(
        old,
        drift=_patch(drift_payload(), {"clean_sessions": 111}),
        volatility=volatility_payload(),
        costs=costs_payload(),
        date="2026-09-10",
    )
    new = tmp_path / "new"
    write_artifacts(
        new,
        drift=_patch(drift_payload(), {"clean_sessions": 222}),
        volatility=volatility_payload(),
        costs=costs_payload(),
        date="2026-09-18",
    )
    # El artefacto nuevo y el viejo conviven en el mismo directorio: gana el de fecha mayor.
    for name in ("drift_decomposition", "volatility_forecast", "cost_audit"):
        for source in (new / f"{name}_2026-09-18.json", old / f"{name}_2026-09-10.json"):
            (tmp_path / source.name).write_bytes(source.read_bytes())

    inputs = load_inputs(tmp_path)
    assert inputs.drift.date == "2026-09-18"
    assert report_for(inputs)["gate_a"]["clean_sample"]["clean_sessions"] == 222


def test_a1_a_missing_artifact_aborts_with_code_2_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reports = write_artifacts(_cli_reports(tmp_path), drift=drift_payload(), costs=costs_payload())

    assert main(_cli_args(tmp_path)) == 2
    error = capsys.readouterr().err
    assert "volatilidad y forecast (#7)" in error
    assert "No se rellena con un valor por defecto" in error
    assert not list(reports.glob(f"{REPORT_STEM}.*"))


def test_a1_an_ambiguous_artifact_aborts_with_code_2_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reports = write_artifacts(
        _cli_reports(tmp_path),
        drift=drift_payload(),
        volatility=volatility_payload(),
        costs=costs_payload(),
        drift_name="drift_decomposition_2026-09-18.json",
    )
    # Segunda copia de la misma clase que resuelve a la misma fecha.
    (reports / "drift_decomposition_2026-9-18.json").write_text(
        json.dumps(drift_payload()), encoding="utf-8"
    )

    assert main(_cli_args(tmp_path)) == 2
    error = capsys.readouterr().err
    assert "misma fecha" in error
    assert "Ambigüedad declarada" in error
    assert not list(reports.glob(f"{REPORT_STEM}.*"))


def test_a1_the_selection_helpers_raise_typed_errors(tmp_path: Path) -> None:
    klass = phase0_report.ARTIFACT_CLASSES[0]
    with pytest.raises(MissingArtifactError):
        select_artifact(tmp_path / "no-existe", klass)
    with pytest.raises(MissingArtifactError):
        select_artifact(tmp_path, klass)

    write_artifacts(tmp_path, drift=drift_payload())
    (tmp_path / "drift_decomposition_2026-9-18.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AmbiguousArtifactError):
        select_artifact(tmp_path, klass)


def test_a1_a_malformed_artifact_aborts_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reports = write_artifacts(
        _cli_reports(tmp_path), volatility=volatility_payload(), costs=costs_payload()
    )
    (reports / "drift_decomposition_2026-09-18.json").write_text("[1, 2]", encoding="utf-8")

    assert main(_cli_args(tmp_path)) == 2
    assert "no es un objeto JSON" in capsys.readouterr().err
    assert not list(reports.glob(f"{REPORT_STEM}.*"))


def test_a1_a_missing_required_field_aborts_instead_of_defaulting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    drift = drift_payload()
    del drift["verdict"]
    reports = write_artifacts(
        _cli_reports(tmp_path),
        drift=drift,
        volatility=volatility_payload(),
        costs=costs_payload(),
    )

    assert main(_cli_args(tmp_path)) == 2
    error = capsys.readouterr().err
    assert "`verdict`" in error
    assert "no lo rellena con un valor por defecto" in error
    assert not list(reports.glob(f"{REPORT_STEM}.*"))


# ─────────────────────────────────────────────────────────────────────────────
# A2 · ninguna cifra derivada es un literal del módulo
# ─────────────────────────────────────────────────────────────────────────────
FORBIDDEN_LITERALS = (
    # derivadas de los artefactos
    "44.77",
    "40.86",
    "65.99",
    "3192",
    "5459",
    "3166",
    "2014-01-01",
    "0.9133",
    "0.5633",
    "0.1928",
    "0.0008",
    "0.42",
    "0.24",
    "2.24",
    "1.82",
    "0.18",
    "0.0042",
    "0.0024",
    "0.0224",
    "0.6528",
    "6.6647",
    "362.6667",
    "366.1923",
    # calculadas a partir de ellas
    "50.42",
    "50.21",
    "50.14",
    "51.12",
    "4892",
    "19.6",
    "0.463",
    "0.576",
    "33.1",
)


def test_a2_no_derived_number_is_written_as_a_literal_in_the_module() -> None:
    source = Path(phase0_report.__file__).read_text(encoding="utf-8")
    present = [literal for literal in FORBIDDEN_LITERALS if literal in source]
    assert present == [], f"estos números deben calcularse, no escribirse: {present}"


def test_a2_changing_the_artifacts_changes_the_report(tmp_path: Path) -> None:
    baseline = report_for(ready_inputs(tmp_path / "base"))
    changed = report_for(
        ready_inputs(
            tmp_path / "otro",
            drift={"clean_sessions": 4321, "clean_segments[0].mean_bp": -12.5},
            volatility={"anchors.median_forecast_sigma_bp": 51.25},
            costs={"declared_table.rows[0].amount.pct": "0.0100"},
        )
    )

    assert baseline["gate_a"]["clean_sample"]["clean_sessions"] == 3192
    assert changed["gate_a"]["clean_sample"]["clean_sessions"] == 4321
    baseline_mean = baseline["gate_a"]["clean_sample"]["segments"][0]["mean_bp"]
    assert baseline_mean == pytest.approx(1.9961, abs=1e-3)
    assert changed["gate_a"]["clean_sample"]["segments"][0]["mean_bp"] == -12.5
    assert (
        baseline["volatility_anchor"]["anchors"]["median_forecast_sigma_bp"]
        != changed["volatility_anchor"]["anchors"]["median_forecast_sigma_bp"]
    )
    assert baseline["p_star"]["cost_used_pct"] == "0.0042"
    assert changed["p_star"]["cost_used_pct"] == "0.0100"
    assert baseline["p_star"]["rows"][0]["p_star_pct"] == "50.42"
    assert changed["p_star"]["rows"][0]["p_star_pct"] == "51.00"


def test_a2_declared_constants_are_published_with_name_value_and_provenance(
    tmp_path: Path,
) -> None:
    payload = report_for(ready_inputs(tmp_path))
    constants = payload["declared_constants"]
    for entry in constants:
        assert entry["name"] and entry["value"] and entry["unit"] and entry["provenance"]
        assert entry["note"]

    by_name = {entry["name"]: entry["value"] for entry in constants}
    assert by_name["r_scenarios_pct"] == "0.5 / 1.0 / 1.5"
    assert by_name["z_alpha_two_sided"] == "1.96"
    assert by_name["z_beta"] == "0.84"
    assert by_name["phase0_gate_b_threshold_pct_of_r"] == "20"
    assert by_name["declared_spread_round_trip_pct"] == "0.0042"
    assert by_name["declared_spread_round_trip_usd"] == "0.42"


# ─────────────────────────────────────────────────────────────────────────────
# A3 · procedencia de cada entrada
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_each_input_publishes_path_as_of_source_series_and_sha256(tmp_path: Path) -> None:
    write_artifacts(
        tmp_path, drift=drift_payload(), volatility=volatility_payload(), costs=costs_payload()
    )
    payload = report_for(load_inputs(tmp_path))
    entries = payload["inputs"]

    for kind, name in (
        ("drift", "drift_decomposition_2026-09-18.json"),
        ("volatility", "volatility_forecast_2026-09-18.json"),
        ("costs", "cost_audit_2026-09-18.json"),
    ):
        entry = entries[kind]
        expected = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert entry["path"] == str(tmp_path / name)
        assert entry["sha256"] == expected
        assert entry["sha256_of"] == "fichero"
        assert entry["artifact_date"] == "2026-09-18"

    assert entries["drift"]["source"] == "yfinance"
    assert entries["drift"]["series_id"] == "^GSPC"
    assert entries["drift"]["as_of"] == "2026-09-18 00:00:00+00:00"
    assert entries["volatility"]["series_id"] == "^GSPC"
    # El artefacto de costes no declara `source` ni `series_id`: no se inventan.
    assert entries["costs"]["source"] is None
    assert entries["costs"]["series_id"] is None
    assert entries["costs"]["as_of"] == "2026-09-18T00:00:00+00:00"


# ─────────────────────────────────────────────────────────────────────────────
# A4 · decisiones abiertas 4, 5 y 6
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_open_decisions_4_5_and_6_are_declared_unresolved(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))
    decisions = {entry["id"]: entry for entry in payload["open_decisions"]}

    assert set(decisions) == {"4", "5", "6"}
    assert decisions["4"]["issue"] == "#59"
    assert decisions["5"]["issue"] == "#60"
    assert decisions["6"]["issue"] == "#61"
    for entry in decisions.values():
        assert entry["state"] == "unresolved"
        assert entry["missing_information"]
        assert entry["depends"]
    assert "R" in decisions["5"]["name"]
    assert "#60" in decisions["5"]["depends"]
    assert payload["gate_b"]["threshold_in_comparable_units"]["applied"] is False


def test_a4_the_module_declares_the_same_decisions() -> None:
    assert {entry["id"] for entry in OPEN_DECISIONS} == {"4", "5", "6"}


# ─────────────────────────────────────────────────────────────────────────────
# A5 · prohibido el valor por defecto silencioso
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_measured_slippage_without_declared_r_is_not_evaluable_not_pass(
    tmp_path: Path,
) -> None:
    payload = report_for(
        ready_inputs(
            tmp_path,
            costs={
                "slippage_ejecucion.state": "measured",
                "slippage_ejecucion.value_pct": "0.05",
                "phase0_gate_b.evaluable": True,
            },
        )
    )
    assert payload["gate_b"]["state"] == "not_evaluable"
    assert payload["gate_a"]["state"] == "fail"
    reason = payload["gate_b"]["reason"]
    assert "decisión abierta 5" in reason
    assert "#60" in reason
    assert payload["verdict"]["half_b"] != "pass"
    assert payload["recommendation"]["value"] != "continue"


def test_a5_no_silent_defaults_are_baked_into_the_module() -> None:
    source = Path(phase0_report.__file__).read_text(encoding="utf-8")
    assert "DECLARED_R" not in source
    assert payload_free_default(source)
    decisions = {entry["id"]: entry for entry in OPEN_DECISIONS}
    assert decisions["5"]["state"] == "unresolved"
    assert decisions["6"]["missing_information"]


def payload_free_default(source: str) -> bool:
    """El módulo no decide el bróker, ni `R`, ni el precio de entrada, ni el slippage."""
    return all(
        forbidden not in source
        for forbidden in ("subasta de apertura por defecto", "broker =", "R = 1 %", "r_pct = 1.0")
    )


def test_a5_a_broken_verdict_vocabulary_is_refused() -> None:
    with pytest.raises(phase0_report.VerdictError):
        aggregate_gate("adelante", "not_evaluable")


# ─────────────────────────────────────────────────────────────────────────────
# A6 · una sola muestra limpia
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_divergent_clean_cutoffs_are_a_declared_conflict_and_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = ready_inputs(tmp_path, volatility={"sample.clean_from": "2015-01-01"})
    with pytest.raises(InputConflictError) as error:
        consolidate(inputs, now=NOW)
    assert "no coinciden" in str(error.value)
    assert "2014-01-01" in str(error.value)
    assert "2015-01-01" in str(error.value)
    assert error.value.conflict["kind"] == "input_conflict"
    assert error.value.conflict["field"] == "clean_from"
    assert error.value.conflict["drift"] == "2014-01-01"
    assert error.value.conflict["volatility"] == "2015-01-01"

    write_artifacts(
        _cli_reports(tmp_path / "cli"),
        drift=drift_payload(),
        volatility=_patch(volatility_payload(), {"sample.clean_from": "2015-01-01"}),
        costs=costs_payload(),
    )
    assert main(_cli_args(tmp_path / "cli")) == 2
    emitted = capsys.readouterr().err
    assert "input_conflict" in emitted
    assert "2014-01-01" in emitted
    assert "2015-01-01" in emitted
    assert not list(_cli_reports(tmp_path / "cli").glob(f"{REPORT_STEM}.*"))


def test_a6_matching_cutoffs_publish_the_shared_definition(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))
    sample = payload["sample_consistency"]
    assert sample["clean_from"] == "2014-01-01"
    assert sample["drift_clean_from"] == sample["volatility_clean_from"]
    assert sample["consistent"] is True
    assert sample["clean_sessions_match"] is True
    assert sample["stale_open_share_match"] is True
    assert "clean_sample_cutoff" in sample["definition"]
    assert "importa" in sample["definition"]


# ─────────────────────────────────────────────────────────────────────────────
# A7 · las dos anclas de |open→close| no se confunden
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_both_anchors_are_published_with_their_own_window(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))
    drift_anchor = payload["gate_a"]["base_rate"]
    volatility_anchor = payload["volatility_anchor"]["anchors"]

    assert drift_anchor["abs_move_median_bp_full_sample"] == pytest.approx(44.77, abs=1e-2)
    assert drift_anchor["sessions"] == 5459
    assert volatility_anchor["median_abs_open_close_bp"] == pytest.approx(40.87, abs=1e-2)
    assert volatility_anchor["absolute_move_sessions"] == 3166
    assert "abs_move_median_bp" not in payload["gate_a"]
    assert payload["limitations"]


def test_a7_there_is_no_windowless_open_close_median(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))
    for windowless in ("abs_move_median_bp", "median_abs_open_close_bp"):
        assert windowless not in payload
    text = render_markdown(consolidate(ready_inputs(tmp_path / "md"), now=NOW))
    assert "muestra **completa**" in text
    assert "ventana **evaluada limpia**" in text


# ─────────────────────────────────────────────────────────────────────────────
# A8 · solo se consumen los `.json`
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_the_report_is_identical_without_the_other_markdown_files(tmp_path: Path) -> None:
    inputs = ready_inputs(tmp_path)
    with_md = consolidate(inputs, now=NOW).json_text()

    for name in (
        "drift_decomposition_2026-09-18.md",
        "volatility_forecast_2026-09-18.md",
        "cost_audit_2026-09-18.md",
    ):
        (tmp_path / name).unlink(missing_ok=True)

    assert consolidate(load_inputs(tmp_path), now=NOW).json_text() == with_md


def test_a8_the_markdown_prose_of_another_report_is_never_read(tmp_path: Path) -> None:
    inputs = ready_inputs(tmp_path)
    poisoned = consolidate(inputs, now=NOW).json_text()

    for name in (
        "drift_decomposition_2026-09-18.md",
        "volatility_forecast_2026-09-18.md",
        "cost_audit_2026-09-18.md",
    ):
        (tmp_path / name).write_text("prosa con cifras falsas: 9999 pb", encoding="utf-8")

    assert consolidate(load_inputs(tmp_path), now=NOW).json_text() == poisoned


# ─────────────────────────────────────────────────────────────────────────────
# A9 · correspondencia declarada de la mitad (a)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("verdict", "gate", "expected"),
    [
        ("overnight", "fail", "fail"),
        ("intraday", "pass", "pass"),
        ("mixed", "inconclusive", "not_evaluable"),
        ("none", "inconclusive", "not_evaluable"),
        ("overnight", "pass", "not_evaluable"),
        ("intraday", "fail", "not_evaluable"),
        ("cualquier-cosa", "fail", "not_evaluable"),
    ],
)
def test_a9_the_declared_mapping_reads_the_artifact(verdict: str, gate: str, expected: str) -> None:
    assert half_a_state(verdict, gate) == expected


def test_a9_the_mapping_is_published_in_the_report() -> None:
    assert HALF_A_MAPPING == (
        ("overnight", "fail", "fail"),
        ("intraday", "pass", "pass"),
    )


def test_a9_a_mixed_verdict_leaves_half_a_not_evaluable(tmp_path: Path) -> None:
    payload = report_for(
        ready_inputs(tmp_path, drift={"verdict": "mixed", "phase0_gate": "inconclusive"})
    )
    assert payload["gate_a"]["state"] == "not_evaluable"
    assert payload["gate_a"]["mapping"]["fallback"] in str(payload["gate_a"])


# ─────────────────────────────────────────────────────────────────────────────
# A10 · detalle que sostiene la mitad (a)
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_the_clean_sample_detail_comes_from_the_artifact(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))
    clean = payload["gate_a"]["clean_sample"]
    assert clean["clean_from"] == "2014-01-01"
    assert clean["clean_sessions"] == 3192
    assert [row["name"] for row in clean["segments"]] == ["intraday", "overnight", "total"]
    intraday = clean["segments"][0]
    assert intraday["mean_bp"] == pytest.approx(1.9961, abs=1e-4)
    assert intraday["p_value"] == pytest.approx(0.19282, abs=1e-5)
    assert intraday["significant"] is False
    assert clean["segments"][1]["significant"] is True


def test_a10_the_paired_difference_is_declared_not_significant(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))
    difference = payload["gate_a"]["paired_difference"]
    assert difference["mean_difference_bp"] == pytest.approx(-0.91339, abs=1e-5)
    assert difference["p_value"] == pytest.approx(0.56330, abs=1e-5)
    assert difference["significant"] is False
    assert "no es significativa" in difference["statement"]
    assert "no se apoya en una diferencia demostrada" in difference["statement"]


# ─────────────────────────────────────────────────────────────────────────────
# A11 · discrepancia de redacción de la puerta (a)
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_both_wordings_are_declared_with_the_one_that_applies(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))
    discrepancy = payload["gate_a"]["wording_discrepancy"]
    sources = " ".join(entry["source"] for entry in discrepancy["declared"])
    assert "tasks.md" in sources
    assert "plan.md" in sources
    assert "concentra" in discrepancy["declared"][0]["wording"]
    assert "negativo o nulo" in discrepancy["declared"][1]["wording"]
    assert discrepancy["applies"] == "`plan.md` §1.1.a"
    assert "**no** está demostrada" in discrepancy["why"]
    assert "No se elige en silencio" in discrepancy["consequence"]


def test_a11_the_why_is_built_from_artifact_numbers(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))
    why = payload["gate_a"]["wording_discrepancy"]["why"]
    assert "+2.00 pb" in why
    assert "p = 0.1928" in why
    assert "-0.91 pb" in why
    assert "p = 0.5633" in why


# ─────────────────────────────────────────────────────────────────────────────
# A12 · hoy la mitad (a) es `fail` y no se puede suavizar
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_half_a_is_fail_and_not_softened(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))
    assert payload["gate_a"]["state"] == "fail"
    assert payload["verdict"]["half_a"] == "fail"
    assert payload["verdict"]["half_a"] not in {"pass", "not_evaluable"}
    markdown = render_markdown(consolidate(ready_inputs(tmp_path / "md"), now=NOW))
    assert "mitad (a) = `fail`" in markdown


# ─────────────────────────────────────────────────────────────────────────────
# A13 · limitaciones de (a), con su enlace
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_gate_a_limitations_are_published_with_their_links(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))
    issues = {entry["issue"] for entry in payload["gate_a"]["limitations"]}
    assert {"#50", "#52", "#61"} <= issues
    text = " ".join(entry["limitation"] for entry in payload["gate_a"]["limitations"])
    assert "^GSPC" in text
    assert "nocturno del **sistema** es cero" in text
    assert payload["gate_a"]["artifact_limitations"] == ["medido sobre el índice, no sobre el CFD"]


# ─────────────────────────────────────────────────────────────────────────────
# A14 · la mitad (b) nace del artefacto
# ─────────────────────────────────────────────────────────────────────────────
def test_a14_half_b_uses_the_artifact_reason_when_not_evaluable(tmp_path: Path) -> None:
    inputs = ready_inputs(tmp_path)
    payload = report_for(inputs)
    artifact_reason = inputs.costs.payload["phase0_gate_b"]["reason"]
    assert payload["gate_b"]["state"] == "not_evaluable"
    assert payload["gate_b"]["artifact"]["evaluable"] is False
    assert artifact_reason in payload["gate_b"]["reason"]
    assert "no es un aprobado condicional" in payload["gate_b"]["reason"]
    assert "no evaluable no es un aprobado" in payload["gate_b"]["not_presented_as"]
    markdown = render_markdown(consolidate(inputs, now=NOW))
    assert "La mitad (b) es `not_evaluable`" in markdown


# ─────────────────────────────────────────────────────────────────────────────
# A15 · prohibido el falso pase
# ─────────────────────────────────────────────────────────────────────────────
def test_a15_unmeasured_slippage_never_produces_pass_or_continue(tmp_path: Path) -> None:
    payload = report_for(
        ready_inputs(tmp_path, drift={"verdict": "intraday", "phase0_gate": "pass"})
    )
    assert payload["gate_a"]["state"] == "pass"
    assert payload["verdict"]["half_b"] == "not_evaluable"
    assert payload["verdict"]["half_b"] != "pass"
    assert payload["verdict"]["gate"] != "pass"
    assert payload["verdict"]["gate"] == "not_evaluable"
    assert payload["recommendation"]["value"] != "continue"
    assert payload["verdict"]["phase1_ready"] is False
    markdown = render_markdown(
        consolidate(
            ready_inputs(tmp_path / "md", drift={"verdict": "intraday", "phase0_gate": "pass"}),
            now=NOW,
        )
    )
    assert "**`reframe`**" in markdown


# ─────────────────────────────────────────────────────────────────────────────
# A16 · el umbral, en unidades comparables
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("slippage", "r_value", "expected"),
    [
        ("0.05", "1.0", False),
        ("0.20", "1.0", False),
        ("0.21", "1.0", True),
        ("0.10", "0.5", False),
        ("0.11", "0.5", True),
        ("0.30", "1.5", False),
        ("0.31", "1.5", True),
    ],
)
def test_a16_the_threshold_compares_comparable_units(
    slippage: str, r_value: str, expected: bool
) -> None:
    assert (
        threshold_breached(
            slippage_pct=Decimal(slippage),
            r_pct=Decimal(r_value),
            threshold_pct_of_r=Decimal("20"),
        )
        is expected
    )


def test_a16_missing_r_keeps_half_b_not_evaluable_even_with_measured_slippage(
    tmp_path: Path,
) -> None:
    payload = report_for(
        ready_inputs(
            tmp_path,
            costs={
                "slippage_ejecucion.state": "measured",
                "slippage_ejecucion.value_pct": "0.50",
                "slippage_ejecucion.observations": 12,
                "phase0_gate_b.evaluable": True,
            },
        )
    )
    threshold = payload["gate_b"]["threshold_in_comparable_units"]
    assert payload["gate_b"]["state"] == "not_evaluable"
    assert threshold["applied"] is False
    assert "decisión abierta 5" in payload["gate_b"]["reason"]
    assert "#60" in payload["gate_b"]["reason"]
    assert payload["gate_b"]["slippage"]["state"] == "measured"
    assert payload["gate_b"]["slippage"]["value_pct"] == "0.50"
    scenarios = {row["r_pct"]: row for row in threshold["absolute_threshold_pct_per_scenario"]}
    assert scenarios["1.0"]["threshold_pct"] == "0.20"
    assert "no decisión del propietario" in scenarios["1.0"]["r_label"]


# ─────────────────────────────────────────────────────────────────────────────
# A17 · tres medidas separadas y el corte de financiación
# ─────────────────────────────────────────────────────────────────────────────
def test_a17_the_three_measures_are_separate_and_never_summed(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))
    measures = payload["declared_costs"]["measures"]
    assert set(measures) == set(MEASURE_KEYS)
    for key in MEASURE_KEYS:
        assert measures[key]["state"] == "unmeasured"
        assert measures[key]["value_pct"] is None
    assert "nunca sumadas" in payload["declared_costs"]["measures_rule"]
    assert "total" not in measures
    assert set(payload["declared_costs"]) == {
        "source",
        "declared_table",
        "fx_cost",
        "round_trip",
        "measures",
        "measures_rule",
        "annualisation",
    }
    assert "slippage" in payload["gate_b"]["dominance"]["statement"]
    assert "~50×" in payload["gate_b"]["dominance"]["statement"]


def test_a17_the_financing_cut_is_unverified_and_its_consequence_is_published(
    tmp_path: Path,
) -> None:
    payload = report_for(ready_inputs(tmp_path))
    cut = payload["gate_b"]["financing_cut"]
    assert cut["state"] == "unverified"
    assert cut["value_et"] is None
    assert "**no** se puede afirmar" in cut["consequence"]
    assert cut["issue"] == "#59"
    codes = {blocker["code"] for blocker in payload["verdict"]["blockers"]}
    assert "financing_cut_unverified" in codes


# ─────────────────────────────────────────────────────────────────────────────
# A18 · la tabla declarada, tal cual
# ─────────────────────────────────────────────────────────────────────────────
def test_a18_the_declared_table_is_reproduced_exactly(tmp_path: Path) -> None:
    inputs = ready_inputs(tmp_path)
    payload = report_for(inputs)
    origin = inputs.costs.payload
    assert payload["declared_costs"]["declared_table"] == origin["declared_table"]
    assert payload["declared_costs"]["fx_cost"] == origin["fx_cost"]
    round_trip = payload["declared_costs"]["round_trip"]
    assert round_trip["spread_usd"] == "0.42"
    assert round_trip["spread_pct"] == "0.0042"
    assert round_trip["short_usd"] == "0.24"
    assert round_trip["short_pct"] == "0.0024"
    assert round_trip["long_usd"] == "2.24"
    assert round_trip["long_pct"] == "0.0224"
    assert round_trip["holding_nights"] == 1
    annualisation = payload["declared_costs"]["annualisation"]
    assert annualisation["consistent"] is True
    assert annualisation["relative_difference"] == "0.009628"
    assert payload["declared_costs"]["fx_cost"]["value_pct"] == "0"


def test_a18_the_mutation_of_the_artifact_does_not_leak_into_the_report(tmp_path: Path) -> None:
    inputs = ready_inputs(tmp_path)
    payload = report_for(inputs)
    payload["declared_costs"]["declared_table"]["rows"][0]["amount"]["pct"] = "9.9"
    assert inputs.costs.payload["declared_table"]["rows"][0]["amount"]["pct"] == "0.0042"


# ─────────────────────────────────────────────────────────────────────────────
# A19 · p* con aritmética exacta
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("r_value", "c_value", "expected"),
    [
        ("0.5", "0.0042", "0.5042"),
        ("1.0", "0.0042", "0.5021"),
        ("1.5", "0.0042", "0.5014"),
        ("1.0", "0.0224", "0.5112"),
        ("1.0", "0.0024", "0.5012"),
    ],
)
def test_a19_p_star_matches_the_hand_computed_values(
    r_value: str, c_value: str, expected: str
) -> None:
    assert p_star(Decimal(r_value), Decimal(c_value)) == Decimal(expected)


def test_a19_the_report_publishes_the_declared_scenarios(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))["p_star"]
    assert payload["formula"] == "p* = (R + c) / 2R"
    assert "Decimal" in payload["arithmetic"]
    intraday = {
        row["r_pct"]: row["p_star_pct"]
        for row in payload["rows"]
        if row["c_label"].startswith("intradía")
    }
    assert intraday == {"0.5": "50.42", "1.0": "50.21", "1.5": "50.14"}
    assert payload["bad_night_scenario"]["p_star_pct"] == "51.12"
    assert payload["bad_night_scenario"]["c_pct"] == "0.0224"
    assert payload["bad_night_scenario"]["r_pct"] == "1.0"
    assert payload["cost_used_pct"] == "0.0042"


def test_a19_every_scenario_is_labeled_as_a_scenario_not_a_decision(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))["p_star"]
    assert "no decisión del propietario" in payload["r_scenarios_source"]
    for row in payload["rows"]:
        assert row["r_label"] == "escenario declarado, no decisión del propietario (-> #60)"
        assert row["p_star_fraction"]


# ─────────────────────────────────────────────────────────────────────────────
# A20 · un p* bajo no es criterio de viabilidad
# ─────────────────────────────────────────────────────────────────────────────
def test_a20_low_p_star_is_not_a_viability_criterion(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))["p_star"]
    viability = payload["viability"]
    assert "ya no es criterio de viabilidad" in viability["statement"]
    assert "plan.md" in viability["statement"]
    assert viability["forbidden_use"] == "usar un `p*` bajo como señal de viabilidad"
    assert "0.0042" in viability["reason"]


def test_a20_the_old_p_star_gate_is_marked_derogated(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))["p_star"]["viability"]["derogated_gate"]
    assert payload["criterion"] == "p* > 60 % ⇒ parar"
    assert payload["status"] == "derogada"
    assert payload["provenance"]


# ─────────────────────────────────────────────────────────────────────────────
# A21 · consecuencia estadística
# ─────────────────────────────────────────────────────────────────────────────
def test_a21_the_statistical_arithmetic_is_exact() -> None:
    operations = required_operations(
        edge=Decimal("0.52"),
        null_probability=Decimal("0.50"),
        z_alpha=Decimal("1.96"),
        z_beta=Decimal("0.84"),
    )
    assert operations == Decimal("4892.16")
    assert int(operations) == 4892

    low, high = confidence_interval(
        observed=Decimal("0.52"),
        operations=300,
        z=Decimal("1.96"),
        decimals=Decimal("0.001"),
    )
    assert (low, high) == (Decimal("0.463"), Decimal("0.576"))
    assert low < Decimal("0.50") < high


def test_a21_the_report_publishes_the_consequence(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))["statistics"]
    assert payload["operations"] == 4892
    assert payload["operations_exact"] == "4892.16"
    assert payload["years"] == "19.6"
    assert payload["years_approx"] == 20
    assert payload["interval_95"] == {"low": "0.463", "high": "0.576"}
    assert payload["interval_contains_null"] is True
    assert "4892 operaciones" in payload["statement"]
    assert "indistinguible del azar" in payload["statement"]
    assert "no será validable por resultado" in payload["conclusion"]


# ─────────────────────────────────────────────────────────────────────────────
# A22 · anclaje de volatilidad (#7)
# ─────────────────────────────────────────────────────────────────────────────
def test_a22_the_volatility_anchor_is_published_and_not_sold_as_viability(
    tmp_path: Path,
) -> None:
    payload = report_for(ready_inputs(tmp_path))["volatility_anchor"]
    assert payload["selection"]["selected"] == "garch"
    assert payload["anchors"]["median_forecast_sigma_bp"] == pytest.approx(65.9917, abs=1e-3)
    assert payload["anchors"]["median_abs_open_close_bp"] == pytest.approx(40.8666, abs=1e-3)
    assert payload["session_drift_mean_bp"] == pytest.approx(1.9961, abs=1e-4)
    assert payload["signal_to_noise_ratio"] == pytest.approx(33.1, abs=0.05)
    assert "mucho menor" in payload["statement"]
    assert "viabilidad direccional" in payload["forbidden_use"]


# ─────────────────────────────────────────────────────────────────────────────
# A23 · regla de agregación declarada
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("half_a", "half_b", "expected"),
    [
        ("pass", "pass", "pass"),
        ("pass", "fail", "fail"),
        ("fail", "pass", "fail"),
        ("fail", "fail", "fail"),
        ("pass", "not_evaluable", "not_evaluable"),
        ("not_evaluable", "pass", "not_evaluable"),
        ("not_evaluable", "not_evaluable", "not_evaluable"),
        ("fail", "not_evaluable", "fail"),
        ("not_evaluable", "fail", "fail"),
    ],
)
def test_a23_the_aggregation_rule_covers_every_combination(
    half_a: str, half_b: str, expected: str
) -> None:
    assert aggregate_gate(half_a, half_b) == expected


def test_a23_the_aggregation_rule_is_published(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))["verdict"]
    assert payload["aggregation_rule"] == GATE_AGGREGATION_RULE
    assert payload["half_vocabulary"] == ["pass", "fail", "not_evaluable"]
    assert payload["half_a"] in payload["half_vocabulary"]
    assert payload["half_b"] in payload["half_vocabulary"]


def test_a23_pass_with_not_evaluable_is_never_pass(tmp_path: Path) -> None:
    payload = report_for(
        ready_inputs(tmp_path, drift={"verdict": "intraday", "phase0_gate": "pass"})
    )
    assert aggregate_gate("pass", "not_evaluable") is GateVerdict.NOT_EVALUABLE
    assert payload["verdict"]["gate"] == "not_evaluable"


# ─────────────────────────────────────────────────────────────────────────────
# A24 · veredicto y `blockers` de hoy
# ─────────────────────────────────────────────────────────────────────────────
def test_a24_with_today_like_artifacts_the_verdict_block_is_the_expected_one(
    tmp_path: Path,
) -> None:
    payload = report_for(ready_inputs(tmp_path))
    verdict = payload["verdict"]
    assert verdict["half_a"] == "fail"
    assert verdict["half_b"] == "not_evaluable"
    assert verdict["gate"] == "fail"
    assert verdict["phase1_ready"] is False
    assert payload["phases"]["phase1_ready"] is False

    codes = {blocker["code"] for blocker in verdict["blockers"]}
    assert {"drift_overnight", "slippage_unmeasured", "r_undecided"} <= codes
    for blocker in verdict["blockers"]:
        assert blocker["half"] in {"a", "b"}
        assert blocker["reason"]
        assert blocker["issues"]


# ─────────────────────────────────────────────────────────────────────────────
# A25 · «no se pasa a Fase 1» es un resultado, no un error
# ─────────────────────────────────────────────────────────────────────────────
def test_a25_no_phase1_is_expected_and_the_markdown_says_it(tmp_path: Path) -> None:
    report = consolidate(ready_inputs(tmp_path), now=NOW)
    assert report.payload["verdict"]["statement"] == "no se pasa a Fase 1"
    markdown = render_markdown(report)
    assert "no se pasa a Fase 1" in markdown


def test_a25_against_the_repository_artifacts_the_states_hold(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Contra los artefactos del almacén, si están en el árbol (no se depende de ellos)."""
    needed = [
        REPO_REPORTS.glob("drift_decomposition_*.json"),
        REPO_REPORTS.glob("volatility_forecast_*.json"),
        REPO_REPORTS.glob("cost_audit_*.json"),
    ]
    if not all(any(pattern) for pattern in needed):
        pytest.skip("los informes de #6/#7/#8 no están en el árbol; no se depende de ellos")

    payload = report_for(load_inputs(REPO_REPORTS))
    verdict = payload["verdict"]
    assert verdict["half_a"] == "fail"
    assert verdict["half_b"] == "not_evaluable"
    assert verdict["gate"] == "fail"
    assert verdict["phase1_ready"] is False
    assert "no se pasa a Fase 1" in render_markdown(consolidate(load_inputs(REPO_REPORTS), now=NOW))
    capsys.readouterr()


# ─────────────────────────────────────────────────────────────────────────────
# A26 · recomendación y regla de consistencia
# ─────────────────────────────────────────────────────────────────────────────
def test_a26_the_consistency_rule_rejects_continue_when_the_gate_is_not_pass() -> None:
    assert "no puede" in RECOMMENDATION_CONSISTENCY_RULE
    assert recommendation_is_consistent(GateVerdict.FAIL, Recommendation.REFRAME) is True
    assert recommendation_is_consistent(GateVerdict.FAIL, Recommendation.STOP) is True
    assert recommendation_is_consistent(GateVerdict.FAIL, Recommendation.CONTINUE) is False
    assert recommendation_is_consistent(GateVerdict.NOT_EVALUABLE, Recommendation.CONTINUE) is False
    assert recommendation_is_consistent(GateVerdict.PASS, Recommendation.CONTINUE) is True


def test_a26_the_recommendation_is_reasoned_with_the_arithmetic(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))["recommendation"]
    assert payload["value"] == "reframe"
    assert "**prohibido**" in payload["reason"]
    assert "4892" in payload["reason"]
    assert "no significativa" in payload["reason"]
    options = {option["option"]: option for option in payload["options"]}
    assert set(options) == {"continue", "reframe", "stop"}
    assert options["continue"]["allowed"] is False
    assert options["reframe"]["allowed"] is True
    assert options["stop"]["requires"]


def test_a26_both_runs_never_recommend_continue_off_a_passing_gate(tmp_path: Path) -> None:
    for half_a, half_b in (("fail", "not_evaluable"), ("pass", "not_evaluable")):
        gate = aggregate_gate(half_a, half_b)
        payload = report_for(
            ready_inputs(
                tmp_path / f"{half_a}-{half_b}",
                drift=({"verdict": "intraday", "phase0_gate": "pass"} if half_a == "pass" else {}),
                costs={"phase0_gate_b.evaluable": True, "slippage_ejecucion.state": "measured"},
            )
        )
        assert payload["verdict"]["gate"] == str(gate)
        assert payload["recommendation"]["value"] != "continue"
        assert payload["recommendation"]["rule"] == RECOMMENDATION_CONSISTENCY_RULE


# ─────────────────────────────────────────────────────────────────────────────
# A27 · caso «sin evidencia»
# ─────────────────────────────────────────────────────────────────────────────
def test_a27_no_evidence_case_never_recommends_continue(tmp_path: Path) -> None:
    payload = report_for(
        ready_inputs(
            tmp_path,
            drift={"verdict": "mixed", "phase0_gate": "inconclusive"},
            costs={"phase0_gate_b.evaluable": False},
        )
    )
    assert payload["verdict"]["half_a"] == "not_evaluable"
    assert payload["verdict"]["half_b"] == "not_evaluable"
    assert payload["verdict"]["gate"] == "not_evaluable"
    assert payload["recommendation"]["value"] != "continue"

    basis = payload["no_basis_for_continuation"]
    assert "No hay base para recomendar «continuar»" in basis["statement"]
    issues = " ".join(item["issue"] for item in basis["missing"])
    assert "#50" in issues
    assert "#60" in issues
    assert all(item["what"] for item in basis["missing"])


def test_a27_a_failing_gate_also_publishes_the_missing_data(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))
    assert "no_basis_for_continuation" in payload


# ─────────────────────────────────────────────────────────────────────────────
# A28 · qué cambiaría el veredicto
# ─────────────────────────────────────────────────────────────────────────────
def test_a28_what_would_change_the_verdict_is_concrete_and_owned(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))["what_would_change_the_verdict"]
    assert len(payload) == 3
    halves = {entry["half"] for entry in payload}
    assert halves == {"a", "b"}
    links = " ".join(entry["issues"] for entry in payload)
    for issue in ("#50", "#52", "#62", "#60"):
        assert issue in links
    for entry in payload:
        assert entry["condition"]
        assert entry["owner"]


# ─────────────────────────────────────────────────────────────────────────────
# A29 · CLI, códigos de salida y `--data-root` aislado
# ─────────────────────────────────────────────────────────────────────────────
def test_a29_the_cli_writes_the_report_and_returns_0(tmp_path: Path) -> None:
    reports = write_artifacts(
        _cli_reports(tmp_path),
        drift=drift_payload(),
        volatility=volatility_payload(),
        costs=costs_payload(),
    )
    assert main(_cli_args(tmp_path)) == 0

    json_path = reports / f"{REPORT_STEM}.json"
    markdown_path = reports / f"{REPORT_STEM}.md"
    assert json_path.is_file()
    assert markdown_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["artifact"] == "phase0_report"
    assert payload["verdict"]["gate"] == "fail"


def test_a29_the_cli_honours_an_explicit_reports_dir(tmp_path: Path) -> None:
    artifacts = write_artifacts(
        tmp_path / "artefactos",
        drift=drift_payload(),
        volatility=volatility_payload(),
        costs=costs_payload(),
    )
    output = tmp_path / "salida"
    assert (
        main(
            [
                "--data-root",
                str(tmp_path / "almacen"),
                "--reports-dir",
                str(artifacts),
                "--now",
                "2026-09-18T00:00:00+00:00",
            ]
        )
        == 0
    )
    assert (artifacts / f"{REPORT_STEM}.json").is_file()
    assert not output.exists()


def test_a29_a_temp_data_root_does_not_touch_the_repository_data(tmp_path: Path) -> None:
    before = _repository_reports_fingerprint()
    write_artifacts(
        tmp_path / "derived" / "reports",
        drift=drift_payload(),
        volatility=volatility_payload(),
        costs=costs_payload(),
    )
    assert main(_cli_args(tmp_path)) == 0
    assert (tmp_path / "derived" / "reports" / f"{REPORT_STEM}.json").is_file()
    assert _repository_reports_fingerprint() == before


@pytest.mark.parametrize("missing", ["drift", "volatility", "costs"])
def test_a29_a_missing_artifact_never_writes_a_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    payloads: dict[str, dict[str, Any] | None] = {
        "drift": drift_payload(),
        "volatility": volatility_payload(),
        "costs": costs_payload(),
    }
    payloads[missing] = None
    write_artifacts(tmp_path, **payloads)  # type: ignore[arg-type]

    assert main(_cli_args(tmp_path)) == 2
    assert capsys.readouterr().err.strip()
    assert not list(tmp_path.glob(f"{REPORT_STEM}.*"))


# ─────────────────────────────────────────────────────────────────────────────
# A30 · determinismo
# ─────────────────────────────────────────────────────────────────────────────
def test_a30_two_runs_produce_byte_identical_json_and_markdown(tmp_path: Path) -> None:
    inputs = ready_inputs(tmp_path)
    first = consolidate(inputs, now=NOW)
    second = consolidate(load_inputs(tmp_path), now=NOW)

    assert first.json_text() == second.json_text()
    assert render_markdown(first) == render_markdown(second)
    assert (
        hashlib.sha256(first.json_text().encode()).hexdigest()
        == hashlib.sha256(second.json_text().encode()).hexdigest()
    )
    assert (
        hashlib.sha256(render_markdown(first).encode()).hexdigest()
        == hashlib.sha256(render_markdown(second).encode()).hexdigest()
    )


def test_a30_the_report_depends_only_on_now_and_the_artifacts(tmp_path: Path) -> None:
    inputs = ready_inputs(tmp_path)
    other_day = consolidate(inputs, now=datetime(2026, 10, 1, tzinfo=UTC))
    assert other_day.report_stem == "phase0_report_2026-10-01"
    assert other_day.payload["as_of_utc"] == "2026-10-01T00:00:00+00:00"


# ─────────────────────────────────────────────────────────────────────────────
# A31 · consolidación pura
# ─────────────────────────────────────────────────────────────────────────────
def test_a31_the_consolidation_is_pure_and_touches_no_disk(tmp_path: Path) -> None:
    before = _repository_reports_fingerprint()
    inputs = in_memory_inputs()
    report = consolidate(inputs, now=NOW)

    assert list(tmp_path.iterdir()) == [], "consolidar no puede escribir nada"
    assert _repository_reports_fingerprint() == before
    assert report.report_stem == REPORT_STEM
    provenance = report.payload["inputs"]
    for kind in ("drift", "volatility", "costs"):
        assert provenance[kind]["path"].startswith("<memoria>")
        assert len(provenance[kind]["sha256"]) == 64
        assert provenance[kind]["sha256_of"] == "payload"


def test_a31_the_inputs_can_be_reloaded_from_the_declared_artifacts(tmp_path: Path) -> None:
    inputs = ready_inputs(tmp_path)
    payload = report_for(inputs)
    declared = {kind: entry["path"] for kind, entry in payload["inputs"].items()}
    for path in declared.values():
        assert Path(path).is_file()
    assert (
        consolidate(load_inputs(tmp_path), now=NOW).json_text()
        == consolidate(inputs, now=NOW).json_text()
    )


# ─────────────────────────────────────────────────────────────────────────────
# A32 · el `.md` y el `.json` dicen lo mismo
# ─────────────────────────────────────────────────────────────────────────────
def test_a32_the_markdown_and_the_json_agree_on_verdict_and_key_figures(
    tmp_path: Path,
) -> None:
    report = consolidate(ready_inputs(tmp_path), now=NOW)
    payload = report.payload
    markdown = render_markdown(report)

    verdict = payload["verdict"]
    for label, value in (
        ("mitad (a)", verdict["half_a"]),
        ("mitad (b)", verdict["half_b"]),
        ("agregado", verdict["gate"]),
        ("phase1_ready", str(verdict["phase1_ready"]).lower()),
        ("recomendación", payload["recommendation"]["value"]),
    ):
        assert f"`{value}`" in markdown, label
    for blocker in verdict["blockers"]:
        assert f"`{blocker['code']}`" in markdown
    assert f"**{payload['p_star']['bad_night_scenario']['p_star_pct']} %**" in markdown
    for row in payload["p_star"]["rows"]:
        assert row["p_star_pct"] in markdown
    assert payload["statistics"]["statement"].split(" ⇒ ")[0] in markdown


def test_a32_the_markdown_includes_every_required_section(tmp_path: Path) -> None:
    markdown = render_markdown(consolidate(ready_inputs(tmp_path), now=NOW))
    for section in (
        "## Estado de la doble puerta",
        "## Puerta (a) — el drift",
        "## Puerta (b) — el *slippage*",
        "## Coste declarado y `p*`",
        "## Consecuencia estadística",
        "## Recomendación",
        "## Qué cambiaría el veredicto",
        "## Decisiones abiertas",
        "## Limitaciones",
        "## Entradas (ruta y `sha256`)",
    ):
        assert section in markdown, section


def test_a32_the_module_docstring_declares_what_it_does_not_decide() -> None:
    docstring = phase0_report.__doc__
    assert docstring is not None
    assert "#11" in docstring
    assert "#59" in docstring
    assert "#60" in docstring
    assert "#61" in docstring
    assert "**no** decide" in docstring


# ─────────────────────────────────────────────────────────────────────────────
# A33 · guardián de `data/` y artefactos sintéticos
# ─────────────────────────────────────────────────────────────────────────────
def test_a33_the_new_tests_only_write_into_tmp_path(tmp_path: Path) -> None:
    before = _repository_reports_fingerprint()
    reports = write_artifacts(
        _cli_reports(tmp_path),
        drift=drift_payload(),
        volatility=volatility_payload(),
        costs=costs_payload(),
    )
    assert main(_cli_args(tmp_path)) == 0
    assert _repository_reports_fingerprint() == before
    assert list(reports.glob("phase0_report_*"))


def test_a33_the_module_is_covered_by_ruff_with_an_explicit_path() -> None:
    """La trampa de #55: el fichero se revisa pasando su ruta explícita."""
    import shutil
    import subprocess

    uv = shutil.which("uv")
    if uv is None:  # pragma: no cover - el entorno del proyecto siempre lo tiene
        pytest.skip("`uv` no está en el PATH; la comprobación se hace a mano")

    result = subprocess.run(  # noqa: S603 - comando fijo, sin entrada ajena
        [uv, "run", "ruff", "check", str(Path(phase0_report.__file__))],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_sanity_check_the_expected_verdict_is_not_a_pass(tmp_path: Path) -> None:
    payload = report_for(ready_inputs(tmp_path))
    assert payload["verdict"]["gate"] != "pass"
    assert "not_evaluable" in payload["verdict"]["half_b"]


def test_the_report_error_is_exported_for_the_cli_contract() -> None:
    assert issubclass(phase0_report.MissingArtifactError, Phase0ReportError)
    assert issubclass(phase0_report.AmbiguousArtifactError, Phase0ReportError)
    assert issubclass(phase0_report.InputConflictError, Phase0ReportError)
    assert issubclass(phase0_report.VerdictError, Phase0ReportError)
