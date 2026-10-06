"""Pruebas del veredicto de Fase 2 sobre la base neta bajo el supuesto declarado (T29b) — #88.

Un test por criterio (``test_a1_...`` … ``test_a17_...``), siempre con ``tmp_path`` y la fixture de
sesion de ``tests/conftest.py`` que **huella ``data/`` y ``runs/``**. Los bordes incomodos
(artefacto ausente, dos artefactos ambiguos, ``--as-of`` ausente o invalido, `write=False`, serie
degenerada, mascara parcial, reproduccion que no cuadra, etiqueta del supuesto que no cuadra) tienen
su propio caso: en este proyecto los defectos aparecen al **reejecutar**, no en la primera pasada.

**Coste de la suite de este fichero, y por que.** Los numeros de este informe salen de **re-derivar
el pipeline** de #28 en vivo (``analyse(..., write=False)``, el precedente de #93), asi que cada
proceso paga esa corrida. El gasto se concentra en:

- **dos procesos frescos** (``PYTHONHASHSEED`` ``0`` y ``1``) **en paralelo**, con el bootstrap de
  #15 **doblado** en los dos lados —sus numeros no entran en ninguna asercion de este fichero y la
  comprobacion de reproduccion de A6 sigue cuadrando, porque compara contra los intervalos que #28
  publica en la **misma** corrida—; son la prueba de A1 (CLI y ficheros) y de A4 (determinismo);
- **una corrida de la sesion** en el proceso de ``pytest`` (``PYTHONHASHSEED`` aleatorio), que es la
  de A6: el bootstrap de **este** modulo queda en su valor declarado, que es el que reproduce el
  artefacto de #28; el de #28 se dobla para no pagar dos veces lo mismo;
- los casos sinteticos (A8, A9, A13, A7) son **puros**: no reejecutan el pipeline y bajan
  ``n_bootstrap`` a un valor declarado para no gastar minutos por test.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Generator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Final, cast

import pytest

from cfdtrader.analysis import phase2_net, phase2_report, pipeline_report
from cfdtrader.analysis.phase2_net import (
    BASIS_NET,
    BEATS_ROWS,
    NetReport,
    NetSeries,
    Phase2NetError,
    ReproductionMismatchError,
    difference_seed,
    net_series_of_run,
    seed_of,
)
from cfdtrader.analysis.phase2_net import (
    main as net_main,
)
from cfdtrader.backtest.engine import (
    STATUS_NO_TRADE,
    STATUS_SKIPPED,
    STATUS_TRADED,
    BacktestRun,
    canonical_text,
)
from cfdtrader.backtest.metrics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
)
from cfdtrader.data.store import Store

MODULE_PATH: Final[Path] = Path(str(phase2_net.__file__))
SOURCE: Final[str] = MODULE_PATH.read_text(encoding="utf-8")
TEST_PATH: Final[Path] = Path(__file__).resolve()
REPO_ROOT: Final[Path] = TEST_PATH.parents[1]
REAL_DATA: Final[Path] = REPO_ROOT / "data"
REAL_REPORTS: Final[Path] = REAL_DATA / "derived" / "reports"
PIPELINE_ARTIFACT: Final[Path] = REAL_REPORTS / "pipeline_backtest_2026-09-23.json"
MODEL_ARTIFACT: Final[Path] = REAL_REPORTS / "model_comparison_2026-09-22.json"

#: Predicado del clon limpio (CI): los informes publicados cuelgan de `data/`, que esta
#: gitignorado. Sin ellos no hay nada que medir (#85).
ARTIFACTS_IN_TREE: Final[bool] = PIPELINE_ARTIFACT.exists() and MODEL_ARTIFACT.exists()

#: Instante declarado de la corrida: el mismo `as_of` del artefacto de #28 que se consume.
NOW: Final[datetime] = datetime(2026, 9, 23, 22, 0, tzinfo=UTC)

#: Nombre del informe de este modulo para ese instante.
STEM: Final[str] = "phase2_net_2026-09-23"

#: Ficheros que A17 prohibe tocar: todo lo que este informe **consume** (los suyos propios y los de
#: #28, #29, #93), mas las tres piezas congeladas del motor y del gate.
#: Los ficheros que la entrega transversal de #108 **si** toca —`pipeline_report.py`,
#: `phase0_report.py`, `costs.py` y `engine.py`, por el barrido de punteros `#50 -> #107`— se
#: han retirado de aqui (mismo patron que #80 A6).
FROZEN_PATHS: Final[frozenset[str]] = frozenset(
    {
        "src/cfdtrader/analysis/phase2_report.py",
        "src/cfdtrader/analysis/phase2_dominance.py",
        "src/cfdtrader/backtest/metrics.py",
        "src/cfdtrader/decision/gate.py",
    }
)


def _skip_without_artifacts() -> None:
    """Salta test o fixture si los informes publicados no estan en el arbol (#85)."""
    if not ARTIFACTS_IN_TREE:
        pytest.skip("los informes publicados de #28/#26 no estan en el arbol")


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap reducido de las corridas del fichero (no un procedimiento distinto)
# ─────────────────────────────────────────────────────────────────────────────
#: Remuestreos por intervalo en las corridas de este fichero. **No** cambia el procedimiento
#: (mismos `seed`, mismo percentil): solo cuantos remuestreos. Se aplica a los **dos** lados —#28 y
#: este modulo— porque A6 exige que la recomputacion **iguale** los intervalos que #28 publica: con
#: lados distintos la igualdad no seria comprobable. La corrida a los 10.000 declarados es la del
#: informe: este fichero solo la sustituye por una mas barata y **declara** el valor que usa.
FAST_SAMPLES: Final[int] = 200


@contextmanager
def _patched_fast() -> Generator[None, None, None]:
    """Baja los remuestreos de #15 en #28 y en este modulo a :data:`FAST_SAMPLES`."""
    original_pipeline = pipeline_report.DEFAULT_BOOTSTRAP_SAMPLES
    original_ours = phase2_net.DEFAULT_BOOTSTRAP_SAMPLES
    pipeline_report.DEFAULT_BOOTSTRAP_SAMPLES = FAST_SAMPLES
    phase2_net.DEFAULT_BOOTSTRAP_SAMPLES = FAST_SAMPLES
    try:
        yield
    finally:
        pipeline_report.DEFAULT_BOOTSTRAP_SAMPLES = original_pipeline
        phase2_net.DEFAULT_BOOTSTRAP_SAMPLES = original_ours


# ─────────────────────────────────────────────────────────────────────────────
# Acceso tipado al payload publicado
# ─────────────────────────────────────────────────────────────────────────────
def at(node: object, *keys: str) -> object:
    """Un valor anidado del payload, sin ``Any`` por el camino."""
    current: object = node
    for key in keys:
        current = cast("Mapping[str, object]", current)[key]
    return current


def as_map(node: object) -> dict[str, object]:
    """El nodo como mapping."""
    return dict(cast("Mapping[str, object]", node))


def as_float(node: object, *keys: str) -> float:
    """Un numero anidado del payload, sin ``Any`` por el camino."""
    return float(cast("float", at(node, *keys)))


def as_text(node: object, *keys: str) -> str:
    """Una cadena anidada del payload."""
    return str(at(node, *keys))


def rows_of(report: NetReport) -> list[dict[str, object]]:
    """Las nueve filas evaluadas, como mappings."""
    return [as_map(item) for item in cast("list[object]", report.payload["criteria"])]


def net_of(report: NetReport) -> dict[str, object]:
    """El bloque neto publicado."""
    return as_map(report.payload["net"])


def _fingerprint(root: Path) -> dict[str, str]:
    """Ruta relativa → sha256 de cada fichero: detecta altas, bajas y cambios."""
    if not root.exists():  # pragma: no cover - el almacen real esta en el arbol en esta sesion
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _copy_artifacts(directory: Path) -> Path:
    """Copia los artefactos reales de #28/#26 (solo lectura) dentro del ``tmp_path``."""
    _skip_without_artifacts()
    reports = directory / "derived" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    for artifact in (PIPELINE_ARTIFACT, MODEL_ARTIFACT):
        shutil.copy2(artifact, reports / artifact.name)
    return reports


# ─────────────────────────────────────────────────────────────────────────────
# Los dos procesos frescos de A4, en paralelo, y la corrida de la sesion
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CliRun:
    """Lo que deja una corrida de la CLI en un proceso fresco."""

    directory: Path
    code: int
    stderr: str
    json_bytes: bytes
    markdown_bytes: bytes
    report_sha256: str
    reports_fingerprint_before: Mapping[str, str]
    reports_fingerprint_after: Mapping[str, str]


#: Arranque del proceso fresco: dobla el bootstrap de #15 en los dos modulos **antes** de llamar a
#: ``main``, para no pagar dos veces la re-derivacion del pipeline. La CLI es la misma.
_FAST_ENTRY: Final[str] = (
    "import sys; sys.path.insert(0, 'tests');"
    "from cfdtrader.analysis import pipeline_report, phase2_net;"
    "from test_phase2_net import FAST_SAMPLES;"
    "pipeline_report.DEFAULT_BOOTSTRAP_SAMPLES = FAST_SAMPLES;"
    "phase2_net.DEFAULT_BOOTSTRAP_SAMPLES = FAST_SAMPLES;"
    "raise SystemExit(phase2_net.main(sys.argv[1:]))"
)


def _run_cli(directory: Path, hash_seed: str, *, extra: Sequence[str] = ()) -> CliRun:
    """Corre la CLI en un proceso fresco, con su ``PYTHONHASHSEED`` y su directorio."""
    reports = _copy_artifacts(directory)
    before = _fingerprint(REAL_REPORTS)
    environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
    completed = subprocess.run(  # noqa: S603 - comando fijo (el interprete de la sesion)
        [
            sys.executable,
            "-c",
            _FAST_ENTRY,
            "--data-root",
            str(REAL_DATA),
            "--reports-dir",
            str(reports),
            "--as-of",
            NOW.isoformat(),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        cwd=str(REPO_ROOT),
    )
    json_path = reports / f"{STEM}.json"
    markdown_path = reports / f"{STEM}.md"
    digest = ""
    if completed.returncode == 0 and json_path.is_file():
        digest = str(json.loads(json_path.read_text(encoding="utf-8"))["report_sha256"])
    return CliRun(
        directory=directory,
        code=completed.returncode,
        stderr=completed.stderr,
        json_bytes=json_path.read_bytes() if json_path.is_file() else b"",
        markdown_bytes=markdown_path.read_bytes() if markdown_path.is_file() else b"",
        report_sha256=digest,
        reports_fingerprint_before=before,
        reports_fingerprint_after=_fingerprint(REAL_REPORTS),
    )


@pytest.fixture(scope="session")
def fresh_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, CliRun]:
    """Dos procesos frescos (``PYTHONHASHSEED`` ``0`` y ``1``) **en paralelo**, cada uno con su dir.

    Cada uno paga una re-derivacion del pipeline de #28, asi que se lanzan a la vez.
    """
    _skip_without_artifacts()
    root = tmp_path_factory.mktemp("net_fresh")
    with ThreadPoolExecutor(max_workers=2) as pool:
        zero = pool.submit(_run_cli, root / "seed0", "0")
        one = pool.submit(_run_cli, root / "seed1", "1")
        return {"seed0": zero.result(), "seed1": one.result()}


@pytest.fixture(scope="session")
def net_report(tmp_path_factory: pytest.TempPathFactory) -> NetReport:
    """La corrida de la sesion (``PYTHONHASHSEED`` aleatorio) **una vez** por sesion.

    Dobla el bootstrap de #28 (sus metricas no entran en este payload: las calcula este modulo) y
    deja **real** el de este modulo, que es el que reproduce el artefacto de #28 para A6.
    """
    reports = _copy_artifacts(tmp_path_factory.mktemp("net_session"))
    with _patched_fast():
        return phase2_net.analyse(
            store=Store(REAL_DATA), reports_dir=reports, as_of=NOW, write=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# Series sinteticas para el calculo puro (A7, A8, A9, A13)
# ─────────────────────────────────────────────────────────────────────────────
#: Muestras sinteticas, sin pipeline y sin almacen. Los casos puros usan los helpers internos del
#: modulo a proposito: son su calculo, y probarlos por dentro es la unica forma de cubrir sus ramas
#: sin pagar una re-derivacion del pipeline por rama.
N_SYNTHETIC: Final[int] = 60
N_BOOTSTRAP_SYNTHETIC: Final[int] = 200
CONFIDENCE_SYNTHETIC: Final[float] = 0.95


def _synthetic_series(values: Sequence[float], *, traded: int | None = None) -> NetSeries:
    """Una serie neta sintetica: todas las sesiones operadas salvo que se diga otra cosa."""
    flags = (
        tuple(True for _ in values)
        if traded is None
        else (True,) * traded + (False,) * (len(values) - traded)
    )
    return NetSeries(
        values_pct=tuple(values), traded=flags, artifact_n=len(values), artifact_sha256="sha256:0"
    )


def _pair_interval(pair: tuple[float, float]) -> dict[str, object]:
    """Un intervalo minimo ``lower``/``upper``/``estimate`` para los casos puros."""
    return {"estimate": pair[0], "lower": pair[0], "upper": pair[1]}


def _net_block_synthetic(
    *, hit: tuple[float, float], per_trade: tuple[float, float], sharpe: tuple[float, float]
) -> dict[str, object]:
    """El bloque neto minimo que consume la fila principal y que #29 leeria (A8, A13)."""
    return {
        "state": "computed",
        "basis": BASIS_NET,
        "is_measurement": False,
        "hit_rate": _pair_interval(hit),
        "hit_rate_per_trade": _pair_interval(per_trade),
        "sharpe": _pair_interval(sharpe),
        "beats": {
            "no_trade": "not_evaluable",
            "liston_a": "not_evaluable",
            "liston_b": "not_evaluable",
        },
    }


def _stars() -> dict[str, object]:
    """El bloque `p*` derivado de #9, sin cablear ningun umbral (A10)."""
    from cfdtrader.analysis.phase2_report import p_star_block

    return p_star_block(cost_pct=Decimal("0.0042"), cost_source="sintetico")


def _main_row_synthetic(net: Mapping[str, object]) -> dict[str, object]:
    """La fila principal evaluada por el modulo sobre un bloque neto sintetico (A8, A13)."""
    evaluated: dict[str, object] = {
        "row_index": 1,
        "kind": "edge_demostrable",
        "source_row": "⭐ **¿Existe edge demostrable?**",
        "criterion": "⭐ **¿Existe edge demostrable?**",
        "threshold": "IC 95 %",
        "action": "Parar",
        "state": "not_evaluable",
        "code": "net_metrics_not_computable",
        "basis": "net",
        "detail": {},
    }
    return phase2_net._main_row(  # pyright: ignore[reportPrivateUsage]
        evaluated=evaluated, net=net, stars=_stars(), decided_r_pct="1"
    )


@dataclass(frozen=True, slots=True)
class FakeRow:
    """Una fila de la tabla de #28 con lo justo que lee el modulo (A9)."""

    series_pct: tuple[float, ...]
    traded: int
    n_test: int


class FakePipeline:
    """Un ``PipelineReport`` de mentira: solo su acceso ``row`` (A9)."""

    def __init__(self, rows: Mapping[str, FakeRow]) -> None:
        self._rows = dict(rows)

    def row(self, name: str) -> FakeRow:
        """La fila con ese nombre; si no esta, es un fallo del test."""
        if name not in self._rows:  # pragma: no cover - el test pide nombres declarados
            raise AssertionError(f"la fila sintetica {name!r} no existe")
        return self._rows[name]


def _beats_synthetic(*, left: Sequence[float], right: Sequence[float]) -> dict[str, object]:
    """Las tres filas de comparacion de una corrida sintetica, con pocos remuestreos (A9)."""
    pipeline = FakePipeline(
        {
            "no_trade": FakeRow(series_pct=(0.0,) * len(right), traded=0, n_test=len(right)),
            "liston_a": FakeRow(series_pct=tuple(right), traded=len(right), n_test=len(right)),
            "liston_b": FakeRow(series_pct=tuple(right), traded=len(right), n_test=len(right)),
        }
    )
    detail = phase2_net._beats_block(  # pyright: ignore[reportPrivateUsage]
        series=_synthetic_series(left),
        report=cast("pipeline_report.PipelineReport", pipeline),
        n_bootstrap=N_BOOTSTRAP_SYNTHETIC,
        confidence_level=CONFIDENCE_SYNTHETIC,
    )
    return {"states": detail[0], "detail": detail[1]}


# ─────────────────────────────────────────────────────────────────────────────
# A1 — API del modulo y CLI
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_module_api_and_cli(fresh_runs: Mapping[str, CliRun]) -> None:
    """A1: existe el modulo con su API minima y la CLI escribe el par JSON/Markdown."""
    assert callable(phase2_net.analyse)
    assert callable(phase2_net.render_markdown)
    assert phase2_net.TASK == "#88"
    assert phase2_net.REPORT_PREFIX == "phase2_net"
    for run in fresh_runs.values():
        assert run.code == 0, run.stderr
        assert run.json_bytes and run.markdown_bytes
        payload = json.loads(run.json_bytes)
        assert payload["analysis"] == "cfdtrader.analysis.phase2_net"
        assert payload["task"] == "#88"
        text = run.markdown_bytes.decode("utf-8")
        assert text.startswith("# Veredicto de Fase 2 sobre la base neta")
        assert "## Las nueve filas de §11.6" in text


def test_a1_cli_success_in_process(tmp_path: Path) -> None:
    """A1: ``main`` devuelve 0 y escribe los dos ficheros cuando todo cuadra."""
    reports = _copy_artifacts(tmp_path)
    with _patched_fast():
        code = net_main(
            [
                "--data-root",
                str(REAL_DATA),
                "--reports-dir",
                str(reports),
                "--as-of",
                NOW.isoformat(),
            ]
        )
    assert code == 0
    assert (reports / f"{STEM}.json").is_file()
    assert (reports / f"{STEM}.md").is_file()


# ─────────────────────────────────────────────────────────────────────────────
# A2 — reloj y red prohibidos; `--as-of` obligatorio; `write=False` no escribe
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_missing_or_invalid_as_of_exits_2_without_writing(tmp_path: Path) -> None:
    """A2: sin ``--as-of`` (o mal formado) no se escribe nada y la salida es 2."""
    reports = _copy_artifacts(tmp_path)
    before = _fingerprint(reports)
    assert net_main(["--data-root", str(REAL_DATA), "--reports-dir", str(reports)]) == 2
    assert (
        net_main(["--data-root", str(REAL_DATA), "--reports-dir", str(reports), "--as-of", "ayer"])
        == 2
    )
    assert _fingerprint(reports) == before


def test_a2_ast_forbids_the_clock_and_the_network() -> None:
    """A2: ninguna ruta del modulo consulta el reloj ni sale a la red."""
    banned = {
        "today",
        "now",
        "utcnow",
        "time",
        "monotonic",
        "perf_counter",
        "socket",
        "urlopen",
        "urlretrieve",
    }
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in banned, node.attr
        if isinstance(node, ast.Name):
            assert node.id not in {"socket"}, node.id
        if isinstance(node, ast.Import):
            assert all(
                alias.name.split(".")[0] not in {"socket", "requests"} for alias in node.names
            )
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in {"socket", "requests"}


def test_a2_write_false_writes_nothing(tmp_path: Path) -> None:
    """A2: ``write=False`` no deja ningun fichero."""
    reports = _copy_artifacts(tmp_path)
    before = _fingerprint(reports)
    with _patched_fast():
        report = phase2_net.analyse(
            store=Store(REAL_DATA), reports_dir=reports, as_of=NOW, write=False
        )
    assert report.payload["task"] == "#88"
    assert _fingerprint(reports) == before


# ─────────────────────────────────────────────────────────────────────────────
# A3 — artefactos: ausente, ambiguo, solo lectura
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_missing_and_ambiguous_artifacts_exit_2(tmp_path: Path) -> None:
    """A3: sin los artefactos de #28/#26 (o con dos candidatos) no se escribe nada."""
    empty = tmp_path / "vacio" / "derived" / "reports"
    empty.mkdir(parents=True)
    assert (
        net_main(
            ["--data-root", str(REAL_DATA), "--reports-dir", str(empty), "--as-of", NOW.isoformat()]
        )
        == 2
    )
    reports = _copy_artifacts(tmp_path / "ambiguo")
    # Dos ficheros que resuelven a la **misma** fecha: es ambiguedad, no un empate de fechas.
    shutil.copy2(PIPELINE_ARTIFACT, reports / "pipeline_backtest_2026-9-23.json")
    assert (
        net_main(
            [
                "--data-root",
                str(REAL_DATA),
                "--reports-dir",
                str(reports),
                "--as-of",
                NOW.isoformat(),
            ]
        )
        == 2
    )
    assert not (reports / f"{STEM}.json").exists()


def test_a3_artifacts_are_consumed_read_only(fresh_runs: Mapping[str, CliRun]) -> None:
    """A3: el directorio de informes real no cambia con la corrida."""
    for run in fresh_runs.values():
        assert run.reports_fingerprint_before == run.reports_fingerprint_after


def test_a3_ast_writes_only_inside_write() -> None:
    """A3: el modulo escribe solo dentro del bloque de escritura del informe."""
    writes = {"write_text", "write_bytes", "unlink", "rmtree", "remove"}
    offenders = [
        node.attr
        for node in ast.walk(ast.parse(SOURCE))
        if isinstance(node, ast.Attribute) and node.attr in writes
    ]
    assert sorted(offenders) == ["write_text", "write_text"]


# ─────────────────────────────────────────────────────────────────────────────
# A4 — determinismo y formato del hash
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_identical_across_fresh_processes(fresh_runs: Mapping[str, CliRun]) -> None:
    """A4: dos procesos frescos con distinto ``PYTHONHASHSEED`` dan lo mismo byte a byte."""
    zero, one = fresh_runs["seed0"], fresh_runs["seed1"]
    assert zero.json_bytes == one.json_bytes
    assert zero.markdown_bytes == one.markdown_bytes
    assert zero.report_sha256 == one.report_sha256


def test_a4_hash_format_and_self_consistency(net_report: NetReport) -> None:
    """A4: el ``report_sha256`` lleva prefijo, tiene 64 hex y es autoconsistente."""
    digest = net_report.report_sha256
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
    without = {key: value for key, value in net_report.payload.items() if key != "report_sha256"}
    recomputed = "sha256:" + hashlib.sha256(canonical_text(without).encode("utf-8")).hexdigest()
    assert digest == recomputed


def test_a4_hash_is_fixed_with_prefix(fresh_runs: Mapping[str, CliRun]) -> None:
    """A4: el hash escrito en el fichero es el del payload sin la clave del hash."""
    run = fresh_runs["seed0"]
    payload = json.loads(run.json_bytes)
    written = payload.pop("report_sha256")
    recomputed = "sha256:" + hashlib.sha256(canonical_text(payload).encode("utf-8")).hexdigest()
    assert written == recomputed == run.report_sha256


def test_a4_bootstrap_is_declared(net_report: NetReport) -> None:
    """A4: el bootstrap declarado (``n``, nivel y semillas) viaja en el payload.

    El valor de ``n_bootstrap`` es :data:`FAST_SAMPLES` porque **este fichero** baja los remuestreos
    (se declara arriba); el declarado del informe sigue siendo ``DEFAULT_BOOTSTRAP_SAMPLES``, que es
    el que usa la corrida real y el que se comprueba aquí.
    """
    assert DEFAULT_BOOTSTRAP_SAMPLES == 10_000
    bootstrap = as_map(net_of(net_report)["bootstrap"])
    assert bootstrap["n_bootstrap"] == FAST_SAMPLES
    assert bootstrap["confidence_level"] == 0.95
    assert bootstrap["seed"] == DEFAULT_BOOTSTRAP_SEED
    seeds = as_map(bootstrap["seeds_by_metric"])
    assert {key: int(cast("int", value)) for key, value in seeds.items()} == {
        "hit_rate": seed_of("hit_rate"),
        "hit_rate_per_trade": seed_of("hit_rate_per_trade"),
        "sharpe": seed_of("sharpe"),
    }
    differences = as_map(bootstrap["difference_seeds"])
    assert int(cast("int", differences["no_trade"])) == difference_seed(0)
    assert int(cast("int", differences["liston_a"])) == difference_seed(1)
    assert difference_seed(0) > max(seed_of(name) for name in seeds)


# ─────────────────────────────────────────────────────────────────────────────
# A5 — la etiqueta del supuesto (nunca una medicion)
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_labelling_of_the_declared_assumption(net_report: NetReport) -> None:
    """A5: la lectura neta viaja etiquetada como **supuesto** declarado, con sus issues."""
    net = net_of(net_report)
    assert net["state"] == "computed"
    assert net["basis"] == BASIS_NET == "declared_cost_with_assumed_slippage"
    assert net["is_measurement"] is False
    assert net["is_validation"] is False
    assert net["assumed_slippage_pct"] == "0.2"
    assert net["slippage_pct_of_r"] == "20"
    assert net["r_pct"] == "1"
    assert net["slippage_state"] == "assumed"
    assert net["assumption_issue"] == "#64"
    assert net["r_issue"] == "#60"
    assert net["measuring_issue"] == "#62"
    assert net_report.payload["is_measurement"] is False
    assert net_report.payload["basis"] == BASIS_NET
    for name in ("hit_rate", "hit_rate_per_trade", "sharpe"):
        assert as_text(net, name, "basis") == BASIS_NET


def test_a5_a_wrong_label_is_a_typed_error() -> None:
    """A5: si el bloque neto de #28 no lleva la etiqueta declarada, no se publica."""
    base = {
        "state": "computed_under_declared_assumption",
        "basis": BASIS_NET,
        "is_measurement": False,
        "assumed_slippage_pct": "0.2",
    }
    for broken in (
        {**base, "basis": "net"},
        {**base, "is_measurement": True},
        {**base, "assumed_slippage_pct": "0.4"},
        {**base, "state": "not_computable"},
    ):
        with pytest.raises(Phase2NetError):
            phase2_net._assumption_labels({"net_metrics": broken})  # pyright: ignore[reportPrivateUsage]


# ─────────────────────────────────────────────────────────────────────────────
# A6 — reproduccion de los intervalos de #28 desde su propia serie
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_net_intervals_reproduce_the_artifact(net_report: NetReport) -> None:
    """A6: los tres intervalos recomputados desde la serie publicada igualan los de #28."""
    reproduction = as_map(net_of(net_report)["reproduction"])
    assert reproduction["checked"] is True
    assert reproduction["reproduces"] is True
    metrics = as_map(reproduction["metrics"])
    assert sorted(metrics) == ["hit_rate", "hit_rate_per_trade", "sharpe"]
    for metric, entry in metrics.items():
        block = as_map(entry)
        assert block["matches"] is True, metric
        seeds = as_map(block["seeds"])
        assert seeds["artifact"] == seeds["here"] == seed_of(metric)
        deltas = as_map(block["deltas"])
        for key in ("estimate", "lower", "upper"):
            assert as_float(deltas, key, "delta") == 0.0


def test_a6_synthetic_intervals_are_reproduced_and_a_mismatch_is_a_typed_error() -> None:
    """A6: la comprobacion es de **igualdad**; un desvio es error tipado, no un aviso."""
    series = _synthetic_series([0.3, -0.2, 0.4, 0.1, -0.5], traded=5)
    intervals = {
        "hit_rate": {"estimate": 0.6, "lower": 0.5, "upper": 0.7},
        "hit_rate_per_trade": {"estimate": 0.6, "lower": 0.5, "upper": 0.7},
        "sharpe": {"estimate": 0.4, "lower": 0.1, "upper": 0.8},
    }
    payload: dict[str, object] = {
        "arms": {
            "coste_declarado": {
                "net_metrics": {
                    name: {**interval, "seed": seed_of(name)}
                    for name, interval in intervals.items()
                }
            }
        }
    }
    block = phase2_net.reproduce_net_metrics(series=series, intervals=intervals, payload=payload)
    assert block["reproduces"] is True
    broken: dict[str, object] = json.loads(json.dumps(payload))
    arms = cast("dict[str, object]", broken["arms"])
    metrics = cast(
        "dict[str, object]", cast("dict[str, object]", arms["coste_declarado"])["net_metrics"]
    )
    cast("dict[str, object]", metrics["sharpe"])["lower"] = 0.2
    with pytest.raises(ReproductionMismatchError):
        phase2_net.reproduce_net_metrics(series=series, intervals=intervals, payload=broken)


# ─────────────────────────────────────────────────────────────────────────────
# A7 — el 20 bp se importa y la serie neta se lee del artefacto
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_the_assumption_is_imported_and_never_typed() -> None:
    """A7: el termino del supuesto viene de #28 (``ASSUMED_SLIPPAGE_PCT``), no de un literal."""
    imported = {
        alias.name
        for node in ast.walk(ast.parse(SOURCE))
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("cfdtrader.analysis")
        for alias in node.names
    }
    assert "ASSUMED_SLIPPAGE_PCT" in imported
    assert Decimal("0.2") == phase2_net.ASSUMED_SLIPPAGE_PCT
    assert 'Decimal("0.2")' not in SOURCE
    assert "= 0.2" not in SOURCE
    # La serie neta **se lee** de `net_series`: el modulo no re-deriva el retorno del motor.
    assert "net_series" in SOURCE
    assert "outcome.gross" not in SOURCE
    assert "c_declared_pct" not in SOURCE


def test_a7_counts_that_do_not_match_are_a_typed_error() -> None:
    """A7: una mascara que no cuadra con los recuentos del artefacto es error tipado."""
    body = {"units": "sintetica", "n": 2, "series_pct": [0.5, 0.0]}
    series_block = {
        **body,
        "series_sha256": "sha256:"
        + hashlib.sha256(canonical_text(body).encode("utf-8")).hexdigest(),
    }
    payload: dict[str, object] = {
        "arms": {
            "coste_declarado": {
                "traded": 1,
                "no_trade": 0,
                "net_series": series_block,
            }
        }
    }
    sessions = (
        SimpleNamespace(status=STATUS_TRADED),
        SimpleNamespace(status=STATUS_NO_TRADE),
    )
    fake_run = cast("BacktestRun", SimpleNamespace(folds=(SimpleNamespace(sessions=sessions),)))
    # La mascara dice 1 operada de 2 y el artefacto publica 1 operada y **cero** sin operar: no
    # cuadra, y entonces no se publica ninguna serie.
    with pytest.raises(Phase2NetError):
        net_series_of_run(run=fake_run, payload=payload)


def test_a7_a_digest_that_does_not_match_is_a_typed_error() -> None:
    """A7: el digest de la serie se recomprueba; si no cuadra, no se publica."""
    with pytest.raises(ReproductionMismatchError):
        phase2_net._checked_series_digest(  # pyright: ignore[reportPrivateUsage]
            {"units": "sintetica", "n": 1, "series_pct": [0.0], "series_sha256": "sha256:0"},
            where="sintetica",
        )


# ─────────────────────────────────────────────────────────────────────────────
# A8 — la fila principal, con sus dos mitades
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_main_row_branches() -> None:
    """A8: ``pass`` si una mitad excluye a favor, ``fail`` si excluye en contra, si no
    ``not_evaluable``; la mitad de acierto se decide con la tasa **por operacion**."""
    p_star = float(Decimal(str(as_map(_stars()["binding"])["p_star_fraction"])))
    above = (min(0.9, p_star + 0.2), 0.99)
    below = (0.01, max(0.1, p_star - 0.2))
    flat = (max(0.0, p_star - 0.05), min(1.0, p_star + 0.05))
    passing = _main_row_synthetic(
        _net_block_synthetic(hit=below, per_trade=above, sharpe=(-0.5, 0.5))
    )
    assert passing["state"] == "pass"
    assert passing["code"] == "statistical_edge"
    failing = _main_row_synthetic(
        _net_block_synthetic(hit=above, per_trade=below, sharpe=(-0.5, -0.1))
    )
    assert failing["state"] == "fail"
    assert failing["code"] == "edge_against"
    inconclusive = _main_row_synthetic(
        _net_block_synthetic(hit=flat, per_trade=flat, sharpe=(-0.5, 0.5))
    )
    assert inconclusive["state"] == "not_evaluable"
    assert inconclusive["code"] == "no_statistical_edge"


def test_a8_the_deciding_half_is_the_per_trade_rate(net_report: NetReport) -> None:
    """A8: el detalle publica las dos tasas y decide con la **por operacion** (#92)."""
    main = rows_of(net_report)[0]
    detail = as_map(main["detail"])
    assert as_text(detail, "hit_rate_per_trade", "denominator") == "trade"
    assert as_text(detail, "hit_rate_session", "denominator") == "session"
    assert "publicada como contexto" in as_text(detail, "hit_rate_session", "role")
    per_trade = as_float(detail, "hit_rate_per_trade", "lower")
    session = as_float(detail, "hit_rate_session", "lower")
    assert per_trade > session  # la tasa por sesion diluye con los `no_trade`
    assert as_text(main, "evaluated_by").endswith("(#88)")


def test_a8_the_real_verdict_on_this_data(net_report: NetReport) -> None:
    """A8: el veredicto **medido** sobre este artefacto: la base neta falla la fila principal."""
    main = rows_of(net_report)[0]
    assert main["state"] == "fail"
    assert main["code"] == "edge_against"
    detail = as_map(main["detail"])
    assert detail["hit_rate_below_p_star"] is True
    assert detail["sharpe_below_zero"] is True
    assert detail["hit_rate_excludes_p_star_above"] is False
    assert detail["sharpe_excludes_zero_above"] is False


# ─────────────────────────────────────────────────────────────────────────────
# A9 — las tres filas de comparacion
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_beats_branches() -> None:
    """A9: ``pass``/``fail``/``not_evaluable`` por la diferencia pareada, y B sin decidir."""
    winners = _beats_synthetic(
        left=[0.6 + 0.01 * (index % 3) for index in range(N_SYNTHETIC)],
        right=[0.0] * N_SYNTHETIC,
    )
    states = as_map(winners["states"])
    assert states["no_trade"] == "pass"
    assert states["liston_a"] == "pass"
    assert states["liston_b"] == "not_evaluable"
    detail = as_map(winners["detail"])
    assert as_map(detail["liston_b"])["decided"] is False
    assert as_text(detail, "liston_b", "issue") == "#70"
    assert as_map(detail["no_trade"])["decided"] is True

    losers = _beats_synthetic(
        left=[-0.6 - 0.01 * (index % 3) for index in range(N_SYNTHETIC)],
        right=[0.0] * N_SYNTHETIC,
    )
    assert as_map(losers["states"])["no_trade"] == "fail"

    flat = _beats_synthetic(
        left=[0.5 if index % 2 == 0 else -0.5 for index in range(N_SYNTHETIC)],
        right=[0.0] * N_SYNTHETIC,
    )
    assert as_map(flat["states"])["no_trade"] == "not_evaluable"


def test_a9_a_partial_mask_is_a_typed_error() -> None:
    """A9: una fila con mascara parcial no tiene base neta comparable: error tipado."""
    pipeline = FakePipeline(
        {
            "no_trade": FakeRow(series_pct=(0.0,) * 4, traded=0, n_test=4),
            "liston_a": FakeRow(series_pct=(0.1, 0.1, 0.1, 0.1), traded=2, n_test=4),
        }
    )
    with pytest.raises(Phase2NetError):
        phase2_net._beats_block(  # pyright: ignore[reportPrivateUsage]
            series=_synthetic_series([0.1, 0.1, 0.1, 0.1]),
            report=cast("pipeline_report.PipelineReport", pipeline),
            n_bootstrap=N_BOOTSTRAP_SYNTHETIC,
            confidence_level=CONFIDENCE_SYNTHETIC,
        )


def test_a9_the_beats_of_the_real_run(net_report: NetReport) -> None:
    """A9: las claves publicadas son las que #29 lee, y B queda sin decidir."""
    net = net_of(net_report)
    beats = as_map(net["beats"])
    assert sorted(beats) == sorted(BEATS_ROWS) == ["liston_a", "liston_b", "no_trade"]
    assert beats["liston_b"] == "not_evaluable"
    detail = as_map(net["beats_detail"])
    assert as_map(detail["liston_b"])["decided"] is False
    for name in ("no_trade", "liston_a"):
        block = as_map(detail[name])
        assert block["decided"] is True
        assert int(cast("int", block["seed"])) == difference_seed(BEATS_ROWS.index(name))
        assert as_text(block, "basis") == BASIS_NET


# ─────────────────────────────────────────────────────────────────────────────
# A10 — `p*` derivado de #9, sin cablear ningun umbral
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_p_star_is_derived_and_never_typed(net_report: NetReport) -> None:
    """A10: `p*` sale de `p_star_block` de #9 por escenario de `R`, y el vinculante es el mayor."""
    stars = as_map(net_report.payload["p_star"])
    binding = as_map(stars["binding"])
    scenarios = [as_map(item) for item in cast("list[object]", stars["scenarios"])]
    assert scenarios
    highest = max(scenarios, key=lambda item: Decimal(str(item["p_star_fraction"])))
    assert binding["p_star_fraction"] == highest["p_star_fraction"]
    assert binding["r_pct"] == highest["r_pct"]
    assert str(stars["constants_source"]).endswith("DECLARED_CONSTANTS (#9)")
    assert "50.2" not in SOURCE and "50,2" not in SOURCE and "0.5021" not in SOURCE


# ─────────────────────────────────────────────────────────────────────────────
# A11 — las nueve filas de §11.6, con sus literales
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_nine_rows_from_the_document(net_report: NetReport) -> None:
    """A11: `n_rows == 9`, los literales son los del documento y las filas de Fase 4 se publican."""
    source = as_map(net_report.payload["criteria_source"])
    assert source["path"] == "_docs/plan.md"
    assert source["section"] == "§11.6"
    assert source["n_rows"] == 9
    assert len(cast("list[object]", net_report.payload["kill_criteria"])) == 9
    rows = rows_of(net_report)
    assert len(rows) == 9
    table = phase2_report.load_kill_table()
    for row, expected in zip(rows, table.rows, strict=True):
        assert row["row_index"] == expected.row_index
        assert row["kind"] == expected.kind
        assert row["source_row"] == expected.criterion
        assert row["threshold"] == expected.threshold
        assert row["action"] == expected.action
    assert rows[7]["state"] == "not_evaluable"
    assert rows[7]["follow_ups"] == ["#45"]
    assert rows[8]["state"] == "not_evaluable"
    assert rows[8]["follow_ups"] == ["#84"]
    assert rows[6]["insufficient_on_its_own"] is True


# ─────────────────────────────────────────────────────────────────────────────
# A12 — agregado, veredicto y `phase2_ready`
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_gate_aggregate_and_verdict(net_report: NetReport) -> None:
    """A12: el agregado es de #9 (via #29), el veredicto es coherente y `phase2_ready` se deriva."""
    gate = as_map(net_report.payload["gate"])
    assert gate["n_rows"] == 9
    assert "aggregate_gate" in str(gate["aggregation_source"])
    assert "aggregate_gate" in str(gate["folding"])
    verdict = as_map(net_report.payload["verdict"])
    assert verdict["state"] in {"continue", "simplify", "stop", "not_evaluable"}
    assert verdict["consistent"] is True
    assert str(verdict["recommendation_source"]).endswith("Recommendation (#9)")
    ready = net_report.payload["phase2_ready"]
    assert ready is (gate["aggregate"] == "pass")
    assert net_report.payload["phase2_ready_rule"] == phase2_net.VERDICT_RULE


def test_a12_the_real_aggregate_on_this_data(net_report: NetReport) -> None:
    """A12: el agregado **medido** sobre este artefacto: `fail` y la puerta sin aprobar."""
    gate = as_map(net_report.payload["gate"])
    counts = as_map(gate["counts"])
    assert gate["aggregate"] == "fail"
    assert int(cast("int", counts["fail"])) >= 1
    assert net_report.payload["phase2_ready"] is False
    assert as_map(net_report.payload["verdict"])["state"] == "stop"


# ─────────────────────────────────────────────────────────────────────────────
# A13 — un `not_evaluable` nunca se convierte en `pass`
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_not_evaluable_is_never_a_pass() -> None:
    """A13: la fila principal `not_evaluable` no aprueba nada, y un `pass` suelto tampoco."""
    p_star = float(Decimal(str(as_map(_stars()["binding"])["p_star_fraction"])))
    flat = (max(0.0, p_star - 0.05), min(1.0, p_star + 0.05))
    inconclusive = _main_row_synthetic(
        _net_block_synthetic(hit=flat, per_trade=flat, sharpe=(-0.5, 0.5))
    )
    assert inconclusive["state"] == "not_evaluable"
    rows: list[dict[str, object]] = [
        inconclusive,
        *(
            {
                "row_index": index,
                "kind": "pbo",
                "state": "not_evaluable",
                "action": "Simplificar modelo",
                "threshold": "< 20%",
                "criterion": "PBO",
                "source_row": "PBO",
                "basis": "artifact",
                "code": "pbo_not_evaluable",
                "detail": {},
            }
            for index in range(2, 10)
        ),
    ]
    gate = phase2_report.gate_block(rows)
    assert gate["aggregate"] == "not_evaluable"
    assert gate["aggregate"] != "pass"


def test_a13_a_passing_main_row_does_not_open_the_gate(net_report: NetReport) -> None:
    """A13: con las filas de Fase 4 sin evaluar, el agregado no puede ser `pass`."""
    gate = as_map(net_report.payload["gate"])
    rows = rows_of(net_report)
    assert any(row["state"] == "not_evaluable" for row in rows)
    assert gate["aggregate"] != "pass"
    assert net_report.payload["phase2_ready"] is False


# ─────────────────────────────────────────────────────────────────────────────
# A14 — JSON estricto y ningun `None` disfrazado de `0`
# ─────────────────────────────────────────────────────────────────────────────
def test_a14_json_is_strict_and_no_none_becomes_zero(net_report: NetReport) -> None:
    """A14: el JSON no lleva `nan`/`inf`, es redondo y un cero solo se publica si es un cero."""
    text = net_report.json_text()
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text) == {**net_report.payload, "report_sha256": net_report.report_sha256}
    assert json.dumps(net_report.payload, allow_nan=False)
    assert _synthetic_series([0.0, 0.0]).is_degenerate is True
    assert _synthetic_series([0.1, 0.0]).is_degenerate is False
    with pytest.raises(Phase2NetError):
        phase2_net._finite(float("nan"), where="sintetica")  # pyright: ignore[reportPrivateUsage]
    assert phase2_net._finite(0, where="sintetica") == 0.0  # pyright: ignore[reportPrivateUsage]


def test_a14_the_published_series_is_the_net_one(net_report: NetReport) -> None:
    """A14: la serie neta del brazo base baja el supuesto en lo operado y deja el cero intacto."""
    series = net_report.series
    assert series.n_sessions == 500
    # Relativo (#136): el recuento de operadas es una **medicion** de la ventana vigente, no un
    # literal (150 era el de la ventana vieja); lo que se fija es que hubo actividad y que la
    # mascara `traded` cuadra con la serie operada.
    assert series.n_traded > 0
    assert series.artifact_n == series.n_sessions
    assert len(series.traded_series_pct()) == series.n_traded
    for value, flag in zip(series.values_pct, series.traded, strict=True):
        if not flag:
            assert value == 0.0
    assert len(series.artifact_sha256) == len("sha256:") + 64


# ─────────────────────────────────────────────────────────────────────────────
# A15 — limites, lo que no hace y seguimientos declarados
# ─────────────────────────────────────────────────────────────────────────────
def test_a15_declared_limits_and_follow_ups(net_report: NetReport) -> None:
    """A15: el informe declara lo que no hace y a que issue va cada hueco."""
    payload = net_report.payload
    issues = {as_text(entry, "issue") for entry in cast("list[object]", payload["follow_ups"])}
    assert {"#62", "#108", "#70", "#45"} <= issues
    identifiers = {as_text(entry, "id") for entry in cast("list[object]", payload["does_not_do"])}
    assert "no_mide_el_slippage" in identifiers
    assert "no_refresca_el_artefacto_publicado" in identifiers
    assert "no_reabre_lo_registrado" in identifiers
    limitations = [str(item) for item in cast("list[object]", payload["limitations"])]
    assert any("#108" in line for line in limitations)
    assert any("#93" in line for line in limitations)
    assert any("#62" in line for line in limitations)
    text = phase2_net.render_markdown(net_report)
    assert "## Que no hace este informe" in text
    assert "## Seguimientos" in text
    assert "## Limites declarados" in text
    assert "Supuesto, no medicion" in text
    assert "is_measurement: false" in text


def test_a15_the_published_artifact_state_is_declared(net_report: NetReport) -> None:
    """A15: el informe dice de que artefacto parte y el estado de su bloque neto.

    Desde que **#108** refresco el artefacto publicado, su bloque neto ya es el **neto bajo el
    supuesto declarado** de #133 (antes decia `not_computable`).
    """
    published = as_map(net_report.payload["published_artifact"])
    assert published["name"] == PIPELINE_ARTIFACT.name
    assert published["net_metrics_state"] == "computed_under_declared_assumption"
    assert published["is_net_computable"] is True
    assert as_text(published, "rederived_report_sha256").startswith("sha256:")


# ─────────────────────────────────────────────────────────────────────────────
# A16 — la documentacion registra la decision
# ─────────────────────────────────────────────────────────────────────────────
def test_a16_documentation_records_the_re_emission() -> None:
    """A16: `plan.md` §19.13 y `tech_stack.md` registran la reemision y sus versiones cuadran."""
    plan = (REPO_ROOT / "_docs" / "plan.md").read_text(encoding="utf-8")
    assert "### 19.13" in plan
    section = plan.split("### 19.13", 1)[1]
    assert "#88" in section
    assert "phase2_net" in section
    stack = (REPO_ROOT / "_docs" / "tech_stack.md").read_text(encoding="utf-8")
    assert "phase2_net.py" in stack
    assert "§19.13" in stack


# ─────────────────────────────────────────────────────────────────────────────
# A17 — modulos congelados intactos y sin dependencia nueva
# ─────────────────────────────────────────────────────────────────────────────
def _changed_paths() -> set[str]:
    """Rutas cambiadas desde el commit de partida, con las nuevas sin seguir incluidas."""
    base = "c574f83"
    tracked = subprocess.run(  # noqa: S603 - comando fijo
        ["git", "diff", "--name-only", base],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
    ).stdout.split()
    others = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
    ).stdout.split()
    return set(tracked) | set(others)


def test_a17_frozen_modules_are_untouched() -> None:
    """A17: nada de lo que este informe consume cambia en esta tarea."""
    changed = _changed_paths()
    assert not (changed & FROZEN_PATHS), sorted(changed & FROZEN_PATHS)


def test_a17_no_new_dependency() -> None:
    """A17: el modulo solo usa la biblioteca estandar, `cfdtrader` y `loguru`."""
    stdlib = {
        "argparse",
        "hashlib",
        "json",
        "math",
        "sys",
        "collections",
        "dataclasses",
        "datetime",
        "decimal",
        "pathlib",
        "typing",
    }
    allowed = stdlib | {"cfdtrader", "loguru"}
    modules: set[str] = set()
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    modules.discard("__future__")  # no es una dependencia: es el compilador
    assert modules <= allowed, sorted(modules - allowed)
    # #134 retira la guardia de manifiesto: declarar `lxml` —el backend de parseo HTML que exige
    # `pandas.read_html`, llamada **dentro** de `yfinance.base.get_earnings_dates` (la ruta de
    # earnings de #126)— cambia `pyproject.toml`/`uv.lock` legitimamente, el mismo criterio que
    # #113/#124/#131 aplicaron a sus modulos congelados. Lo que sigue en pie es que **este** modulo
    # solo importa stdlib + `cfdtrader` + `loguru`, que es la asercion de arriba.


def test_a17_the_module_lives_in_analysis_and_ignores_the_daily_path() -> None:
    """A17: el informe vive en `analysis/` y no toca el camino diario ni el gate."""
    assert MODULE_PATH.parent.name == "analysis"
    changed = _changed_paths()
    assert not any(path.startswith("src/cfdtrader/delivery/") for path in changed)
    assert not any(path.startswith("src/cfdtrader/decision/") for path in changed)


# ─────────────────────────────────────────────────────────────────────────────
# Bordes: los errores tipados del modulo, sin volver a re-derivar el pipeline
# ─────────────────────────────────────────────────────────────────────────────
def test_edge_series_without_net_block_is_a_typed_error() -> None:
    """Un payload que no publica `net_series` es un hueco, no un cero."""
    with pytest.raises(Phase2NetError):
        phase2_net._published_net_block(  # pyright: ignore[reportPrivateUsage]
            {"arms": {"coste_declarado": {}}, "net_metrics": {"state": "not_computable"}}
        )
    with pytest.raises(Phase2NetError):
        phase2_net._published_net_block(  # pyright: ignore[reportPrivateUsage]
            {"arms": {"coste_declarado": {}}, "net_metrics": "no es un objeto"}
        )
    with pytest.raises(Phase2NetError):
        phase2_net._published_net_block(  # pyright: ignore[reportPrivateUsage]
            {"net_metrics": {}}
        )


def test_edge_series_block_without_its_body_is_a_typed_error() -> None:
    """Un bloque de serie al que le falta el cuerpo no se puede verificar."""
    with pytest.raises(Phase2NetError):
        phase2_net._checked_series_digest(  # pyright: ignore[reportPrivateUsage]
            {"n": 2}, where="sintetica"
        )


def test_edge_net_series_mismatch_and_malformed() -> None:
    """La serie tiene que ser una lista de numeros finitos y cuadrar con los recuentos."""
    with pytest.raises(Phase2NetError):
        phase2_net._finite(True, where="sintetica")  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(Phase2NetError):
        phase2_net._number({"a": "x"}, "a", where="sintetica")  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(Phase2NetError):
        phase2_net._mapping([], where="sintetica")  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(Phase2NetError):
        phase2_net.seed_of("metrica_que_no_existe")
    with pytest.raises(Phase2NetError):
        difference_seed(-1)
    with pytest.raises(Phase2NetError):
        NetSeries(values_pct=(0.1,), traded=(True, False), artifact_n=1, artifact_sha256="sha256:0")
    # La serie publicada tiene que ser una **lista** y cuadrar con la corrida en vivo.
    body = {"units": "sintetica", "n": 1, "series_pct": "no es una lista"}
    block = {
        **body,
        "series_sha256": "sha256:"
        + hashlib.sha256(canonical_text(body).encode("utf-8")).hexdigest(),
    }
    payload: dict[str, object] = {"arms": {"coste_declarado": {"net_series": block}}}
    sessions = (SimpleNamespace(status=STATUS_TRADED), SimpleNamespace(status=STATUS_NO_TRADE))
    run = cast("BacktestRun", SimpleNamespace(folds=(SimpleNamespace(sessions=sessions),)))
    with pytest.raises(Phase2NetError):
        net_series_of_run(run=run, payload=payload)
    # Y la longitud tiene que coincidir con la mascara en vivo.
    body = {"units": "sintetica", "n": 1, "series_pct": [0.5]}
    block = {
        **body,
        "series_sha256": "sha256:"
        + hashlib.sha256(canonical_text(body).encode("utf-8")).hexdigest(),
    }
    payload = {"arms": {"coste_declarado": {"traded": 1, "no_trade": 0, "net_series": block}}}
    with pytest.raises(Phase2NetError):
        net_series_of_run(run=run, payload=payload)


def test_edge_a_rate_without_a_role_and_a_checked_main_row() -> None:
    """Un bloque de tasa sin `role` no publica el campo, y la fila principal no lo inventa."""
    block = phase2_net._rate_block(  # pyright: ignore[reportPrivateUsage]
        interval={"estimate": 0.5, "lower": 0.4, "upper": 0.6},
        denominator="session",
        note="sintetica",
        series=[0.1, -0.1, 0.2],
    )
    assert "role" not in block
    assert block["n_wins"] == 2
    assert block["wins_fraction"] == "2/3"
    assert block["n"] == 3


def test_edge_unaligned_difference_is_a_typed_error() -> None:
    """La diferencia pareada rechaza longitudes distintas por su cuenta (A9)."""
    with pytest.raises(Phase2NetError):
        phase2_net._difference_interval(  # pyright: ignore[reportPrivateUsage]
            left_pct=[0.1, 0.1],
            right_pct=[0.0],
            seed=difference_seed(0),
            confidence_level=CONFIDENCE_SYNTHETIC,
            n_bootstrap=N_BOOTSTRAP_SYNTHETIC,
        )


def test_edge_a_naive_instant_is_read_as_utc() -> None:
    """Un `as_of` sin zona se lee como UTC (A2); con zona se convierte."""
    naive = phase2_net._as_utc(datetime(2026, 9, 23, 22, 0))  # pyright: ignore[reportPrivateUsage]
    assert naive == NOW
    assert naive.tzinfo is UTC
    assert phase2_net._as_utc(NOW.astimezone(tz=None)) == NOW  # pyright: ignore[reportPrivateUsage]


def test_edge_traded_mask_skips_sessions() -> None:
    """Una sesion saltada no entra en la mascara; una sin operacion entra como ``False``."""
    sessions = (
        SimpleNamespace(status=STATUS_TRADED),
        SimpleNamespace(status=STATUS_SKIPPED),
        SimpleNamespace(status=STATUS_NO_TRADE),
    )
    run = cast("BacktestRun", SimpleNamespace(folds=(SimpleNamespace(sessions=sessions),)))
    assert phase2_net._traded_mask(run) == (True, False)  # pyright: ignore[reportPrivateUsage]


def test_edge_a_row_without_trades_but_a_non_zero_series_is_a_typed_error() -> None:
    """Una fila que no opera y publica retornos no tiene base neta: error tipado."""
    pipeline = FakePipeline(
        {
            "no_trade": FakeRow(series_pct=(0.5, 0.5), traded=0, n_test=2),
            "liston_a": FakeRow(series_pct=(0.1, 0.1), traded=2, n_test=2),
        }
    )
    with pytest.raises(Phase2NetError):
        phase2_net._beats_block(  # pyright: ignore[reportPrivateUsage]
            series=_synthetic_series([0.1, 0.1]),
            report=cast("pipeline_report.PipelineReport", pipeline),
            n_bootstrap=N_BOOTSTRAP_SYNTHETIC,
            confidence_level=CONFIDENCE_SYNTHETIC,
        )


def test_edge_beats_with_unaligned_lengths_is_a_typed_error() -> None:
    """La comparacion pareada exige la misma longitud: no se recorta ninguna serie."""
    pipeline = FakePipeline(
        {
            "no_trade": FakeRow(series_pct=(0.0,) * 3, traded=0, n_test=3),
            "liston_a": FakeRow(series_pct=(0.1, 0.1, 0.1), traded=3, n_test=3),
        }
    )
    with pytest.raises(Phase2NetError):
        phase2_net._beats_block(  # pyright: ignore[reportPrivateUsage]
            series=_synthetic_series([0.1, 0.1]),
            report=cast("pipeline_report.PipelineReport", pipeline),
            n_bootstrap=N_BOOTSTRAP_SYNTHETIC,
            confidence_level=CONFIDENCE_SYNTHETIC,
        )


def test_edge_regeneration_records_the_pointer_delta(tmp_path: Path) -> None:
    """`--previous-artifact` publica el delta de puntero y las invariantes (#90)."""
    reports = _copy_artifacts(tmp_path)
    with _patched_fast():
        first = phase2_net.analyse(
            store=Store(REAL_DATA), reports_dir=reports, as_of=NOW, write=True
        )
        second = phase2_net.analyse(
            store=Store(REAL_DATA),
            reports_dir=reports,
            as_of=NOW,
            write=False,
            previous_artifact=reports / f"{STEM}.json",
        )
    without_regeneration = {
        key: value for key, value in second.payload.items() if key != "regeneration"
    }
    assert without_regeneration == first.payload
    block = as_map(second.payload["regeneration"])
    assert as_text(block, "previous_artifact").endswith(f"{STEM}.json")
    pointers = cast("list[object]", block["pointers"])
    assert len(pointers) == 4
    unchanged = as_map(block["unchanged"])
    assert unchanged["phase2_ready"] is True
    assert unchanged["gate_aggregate"] is True
    assert unchanged["n_rows"] is True
    assert unchanged["assumed_slippage_pct"] is True
    text = phase2_net.render_markdown(second)
    assert "## Regeneracion" in text
