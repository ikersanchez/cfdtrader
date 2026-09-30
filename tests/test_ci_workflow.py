"""Guarda estructural del CI de GitHub Actions (#85, seguimiento de #17).

El CI es un fichero de configuración, no código: la forma de que no se degrade en
silencio (un job que se cae, un selector `-k` que deja de casar con ningún test,
un `continue-on-error` que convierte una puerta en un adorno) es auditar el YAML.

Lo que **no** se puede comprobar desde aquí es que el runner ejecute los comandos
de verdad: eso lo mide la propia ejecución del workflow. Esta suite fija la
estructura; el run fija el resultado.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, cast

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PRE_COMMIT = REPO_ROOT / ".pre-commit-config.yaml"
INTEGRITY = REPO_ROOT / "tests" / "test_integrity.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: Los cuatro jobs declarados por #85.
JOB_LINT = "lint"
JOB_TYPECHECK = "typecheck"
JOB_GOLDEN = "feature-golden"
JOB_TESTS = "tests"
JOBS = (JOB_LINT, JOB_TYPECHECK, JOB_GOLDEN, JOB_TESTS)

#: Versión de Python que fija el proyecto (`pyproject.toml` requires-python y `.python-version`).
PYTHON_VERSION = "3.12"


def _workflow() -> dict[str, Any]:
    return cast("dict[str, Any]", yaml.safe_load(WORKFLOW.read_text(encoding="utf-8")))


def _job(name: str) -> dict[str, Any]:
    jobs = cast("dict[str, Any]", _workflow()["jobs"])
    assert name in jobs, f"el workflow no declara el job {name!r}"
    return cast("dict[str, Any]", jobs[name])


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return cast("list[dict[str, Any]]", job["steps"])


def _run_text(job: dict[str, Any]) -> str:
    """Todo el texto de los `run` de un job, unido (para greps de comandos)."""
    return "\n".join(str(step["run"]) for step in _steps(job) if "run" in step)


def _on_block() -> dict[str, Any]:
    """El bloque `on:` del workflow.

    PyYAML aplica la resolución YAML 1.1 y convierte la clave `on` en el booleano
    `True`: hay que aceptar las dos formas o el test miente sobre lo que ve.
    """
    raw = cast("dict[Any, Any]", _workflow())
    block = raw.get("on")
    if block is None:
        block = raw.get(True)
    assert isinstance(block, dict), "el workflow no declara disparadores (`on`)"
    return cast("dict[str, Any]", block)


def _integrity_test_names() -> list[str]:
    source = INTEGRITY.read_text(encoding="utf-8")
    return [str(name) for name in re.findall(r"^def (test_\w+)", source, re.MULTILINE)]


def _pytest_selector(job: dict[str, Any]) -> list[str]:
    """Los términos del selector `-k "..."` declarado en un job."""
    match = re.search(r'-k\s+"([^"]+)"', _run_text(job))
    assert match is not None, "el job no declara un selector -k entre comillas"
    return [term.strip() for term in match.group(1).split(" or ")]


# ─────────────────────────────────────────────────────────────────────────────
# A1 — el fichero existe donde GitHub lo busca
# ─────────────────────────────────────────────────────────────────────────────
def test_a1_the_workflow_file_is_where_github_looks_for_it() -> None:
    assert WORKFLOW == REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert WORKFLOW.is_file(), "no existe .github/workflows/ci.yml"
    assert not list((REPO_ROOT / ".github" / "workflows").glob("*.yaml")), (
        "GitHub solo reconoce .yml/.yaml; se declara un unico fichero .yml"
    )


# ─────────────────────────────────────────────────────────────────────────────
# A2 — es YAML valido y declara los cuatro jobs
# ─────────────────────────────────────────────────────────────────────────────
def test_a2_the_workflow_parses_and_declares_the_four_jobs() -> None:
    workflow = _workflow()
    assert isinstance(workflow.get("name"), str) and workflow["name"]
    assert isinstance(workflow.get("jobs"), dict)
    assert set(cast("dict[str, Any]", workflow["jobs"])) == set(JOBS), sorted(
        cast("dict[str, Any]", workflow["jobs"])
    )
    for name in JOBS:
        assert _steps(_job(name)), f"el job {name!r} no tiene pasos"


# ─────────────────────────────────────────────────────────────────────────────
# A3 — dispara en push a main y en pull_request
# ─────────────────────────────────────────────────────────────────────────────
def test_a3_it_triggers_on_push_to_main_and_on_pull_request() -> None:
    on = _on_block()
    assert "pull_request" in on, "sin `pull_request` el CI no protege un PR"
    assert "push" in on, "sin `push` la rama principal no se comprueba sola"
    push = on["push"]
    assert isinstance(push, dict)
    assert cast("dict[str, Any]", push).get("branches") == ["main"]


# ─────────────────────────────────────────────────────────────────────────────
# A4 — permisos minimos y cancelacion de ejecuciones obsoletas
# ─────────────────────────────────────────────────────────────────────────────
def test_a4_least_privilege_and_concurrency() -> None:
    workflow = _workflow()
    permissions = workflow["permissions"]
    assert permissions == {"contents": "read"}, permissions
    concurrency = workflow["concurrency"]
    assert isinstance(concurrency, dict)
    assert "group" in concurrency
    assert cast("dict[str, Any]", concurrency)["cancel-in-progress"] is True


# ─────────────────────────────────────────────────────────────────────────────
# A5 — todo corre en el runner gratuito y en el mismo SO
# ─────────────────────────────────────────────────────────────────────────────
def test_a5_every_job_runs_on_ubuntu_latest() -> None:
    for name in JOBS:
        assert _job(name)["runs-on"] == "ubuntu-latest", name


# ─────────────────────────────────────────────────────────────────────────────
# A6 — todo job parte de un clon limpio
# ─────────────────────────────────────────────────────────────────────────────
def test_a6_every_job_checks_out_the_full_history() -> None:
    """Las guardias historicas de la suite hacen `git diff <commit>..HEAD`.

    `actions/checkout` clona con `fetch-depth: 1` por defecto: en un clon superficial
    commits como `6aa582d`, `9cb5068` o `35d2592` no existen y las guardias fallan con
    `fatal: ambiguous argument` (visto en la primera ejecucion real de #85).
    """
    for name in JOBS:
        checkouts = [
            cast("dict[str, Any]", step.get("with", {}))
            for step in _steps(_job(name))
            if str(step.get("uses", "")).startswith("actions/checkout@")
        ]
        assert len(checkouts) == 1, f"el job {name!r} no hace checkout exactamente una vez"
        assert checkouts[0].get("fetch-depth") == 0, (
            f"el job {name!r} no trae el historial completo"
        )


# ─────────────────────────────────────────────────────────────────────────────
# A7 — todo job instala uv con la cache activada y Python 3.12
# ─────────────────────────────────────────────────────────────────────────────
def test_a7_every_job_sets_up_uv_with_cache_and_python_312() -> None:
    for name in JOBS:
        setups = [
            cast("dict[str, Any]", step["with"])
            for step in _steps(_job(name))
            if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
        ]
        assert len(setups) == 1, f"el job {name!r} no instala uv exactamente una vez"
        with_block = setups[0]
        assert with_block.get("enable-cache") is True, f"el job {name!r} no cachea uv"
        assert with_block.get("python-version") == PYTHON_VERSION, name


# ─────────────────────────────────────────────────────────────────────────────
# A8 — ninguna etapa usa otra version de Python
# ─────────────────────────────────────────────────────────────────────────────
def test_a8_no_job_pins_another_python_version() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "setup-python" not in text, "el Python lo fija `setup-uv`, no `actions/setup-python`"
    for match in re.findall(r"python-version:\s*\"?'?([0-9.]+)", text):
        assert match == PYTHON_VERSION, f"version de Python inesperada: {match}"


# ─────────────────────────────────────────────────────────────────────────────
# A9 — las dependencias vienen del lockfile, nunca de una resolucion nueva
# ─────────────────────────────────────────────────────────────────────────────
def test_a9_every_job_installs_from_the_lockfile() -> None:
    env = cast("dict[str, Any]", _workflow().get("env") or {})
    assert env.get("UV_FROZEN") == "1"
    for name in JOBS:
        assert "uv sync --locked" in _run_text(_job(name)), name


# ─────────────────────────────────────────────────────────────────────────────
# A10 — el job de lint corre las dos mitades de ruff
# ─────────────────────────────────────────────────────────────────────────────
def test_a10_lint_runs_both_ruff_halves() -> None:
    text = _run_text(_job(JOB_LINT))
    assert "uv run ruff check ." in text
    assert "uv run ruff format --check ." in text


# ─────────────────────────────────────────────────────────────────────────────
# A11 — el job de tipos corre pyright en modo estricto
# ─────────────────────────────────────────────────────────────────────────────
def test_a11_typecheck_runs_pyright() -> None:
    assert "uv run pyright" in _run_text(_job(JOB_TYPECHECK))


# ─────────────────────────────────────────────────────────────────────────────
# A12 — la puerta del golden no puede quedar en un no-op silencioso
# ─────────────────────────────────────────────────────────────────────────────
def test_a12_the_golden_gate_selector_matches_real_tests() -> None:
    text = _run_text(_job(JOB_GOLDEN))
    assert "tests/test_integrity.py" in text
    assert "uv run pytest" in text
    terms = _pytest_selector(_job(JOB_GOLDEN))
    names = _integrity_test_names()
    assert names, "no se han encontrado tests en test_integrity.py"
    for term in terms:
        assert any(term in name for name in names), (
            f"el selector {term!r} de la puerta del golden no casa con ningun test"
        )
    assert terms, "el selector de la puerta del golden esta vacio"


# ─────────────────────────────────────────────────────────────────────────────
# A13 — la suite completa corre sin data/ y no puede pasar en vacio
# ─────────────────────────────────────────────────────────────────────────────
def test_a13_the_full_suite_runs_and_cannot_pass_vacuously() -> None:
    job = _job(JOB_TESTS)
    text = _run_text(job)
    assert "uv run pytest -q" in text
    assert "pipefail" in text, "sin pipefail, `| tee` enmascara el fallo de pytest"
    assert 'tee "$RUNNER_TEMP/pytest.log"' in text, "el log tiene que ir fuera del arbol"
    assert "tee pytest.log" not in text, (
        "el log no puede escribirse dentro del arbol: lo ensucia y rompe las guardias "
        "de arbol limpio (test_metrics.py::test_a35, test_gate_sweep.py::test_a15, ...)"
    )
    floor = re.search(r"-lt\s+(\d+)", text)
    assert floor is not None, "el job de tests no declara el umbral de vacio"
    assert int(floor.group(1)) > 0, "un umbral de 0 no protege de nada"
    names = [str(step.get("name", "")) for step in _steps(job)]
    assert any("vacio" in name.lower() or "vacío" in name.lower() for name in names), names


# ─────────────────────────────────────────────────────────────────────────────
# A14 — el workflow no depende de data/ (que no existe en el runner)
# ─────────────────────────────────────────────────────────────────────────────
def test_a14_the_workflow_never_requires_the_gitignored_data() -> None:
    for name in JOBS:
        assert "data/" not in _run_text(_job(name)), (
            f"el job {name!r} menciona data/ en un `run`: en el runner no existe"
        )


# ─────────────────────────────────────────────────────────────────────────────
# A15 — ninguna puerta se degrada a `continue-on-error`
# ─────────────────────────────────────────────────────────────────────────────
def test_a15_no_step_is_allowed_to_fail_silently() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "continue-on-error" not in text, (
        "`continue-on-error` convierte una comprobacion en un adorno"
    )
    assert "|| true" not in text


# ─────────────────────────────────────────────────────────────────────────────
# A16 — la barrera local no se mueve: `pre-push` sigue siendo solo el golden
# ─────────────────────────────────────────────────────────────────────────────
def test_a16_the_pre_push_stage_stays_fast() -> None:
    config = cast("dict[str, Any]", yaml.safe_load(PRE_COMMIT.read_text(encoding="utf-8")))
    hooks = [
        hook
        for repo in cast("list[dict[str, Any]]", config["repos"])
        if repo.get("repo") == "local"
        for hook in cast("list[dict[str, Any]]", repo["hooks"])
        if hook["id"] == "integrity-golden"
    ]
    assert len(hooks) == 1
    hook = hooks[0]
    assert hook["stages"] == ["pre-push"]
    entry = str(hook["entry"])
    assert "tests/test_integrity.py" in entry
    # La suite completa se queda en el CI: `pre-push` no la ejecuta (decision de #85).
    assert "-k" in entry and ("golden_matrix" in entry or "version_gate" in entry)
    assert "testpaths" not in entry
    assert "uv run pytest tests" not in entry.replace("tests/test_integrity.py", "")


# ─────────────────────────────────────────────────────────────────────────────
# A17 — el `tmp_path` de la suite sobrevive a un clon limpio
# ─────────────────────────────────────────────────────────────────────────────
def test_a17_the_basetemp_parent_exists_in_a_clean_clone() -> None:
    """`--basetemp=.scratch/pytest` exige que `.scratch/` exista.

    `.scratch/` está gitignorado, así que no viaja al clon: pytest hace
    `mkdir(mode=0o700)` sin `parents=True` y toda la suite que usa `tmp_path` caía con
    `FileNotFoundError` (385 errores en el clon limpio de #85). Lo crea
    `tests/conftest.py` al importarse; esta guarda lo declara.
    """
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    options = cast("dict[str, Any]", config["tool"]["pytest"]["ini_options"])
    addopts = cast("list[str]", options["addopts"])
    flags = [opt for opt in addopts if opt.startswith("--basetemp=")]
    assert len(flags) == 1, f"se esperaba un unico --basetemp, hay: {addopts}"
    basetemp = REPO_ROOT / flags[0].split("=", 1)[1]
    assert basetemp.is_relative_to(REPO_ROOT), (
        f"el --basetemp {basetemp} debe quedar dentro del workspace"
    )
    assert basetemp.parent.is_dir(), (
        f"el padre de --basetemp ({basetemp.parent}) no existe: en un clon limpio "
        "pytest falla con FileNotFoundError"
    )
