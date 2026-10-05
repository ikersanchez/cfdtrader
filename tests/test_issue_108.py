"""Guardian del barrido de punteros de #108.

Comprueba sobre los artefactos **publicados** (``data/derived/reports/``) que ningun
informe presenta la ausencia de la fuente del ``SPX500:CFD`` como una **frontera
pendiente** de ``#50`` (esa frontera es ``#107``), ni cita ``#60`` (cerrada) como causa
de un ``null``. Sin ``data/derived/reports/`` (un clon limpio, CI) las pruebas se saltan.

Los literales que se anclan son **numeros de issue** (valores estables), nunca el digest
de un artefacto regenerable (``_docs/team/pm.md``, #89/#95/#96).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final, cast

import pytest

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPORTS_DIR: Final[Path] = REPO_ROOT / "data" / "derived" / "reports"

#: Claves estructuradas que declaran un seguimiento/frontera en un payload.
ISSUE_KEYS: Final[frozenset[str]] = frozenset(
    {"issue", "issues", "follow_up_issue", "follow_up_issues"}
)

#: La issue cerrada cuyo puntero no puede sobrevivir como **frontera pendiente**.
CLOSED_FRONTIER: Final[str] = "#50"

#: La adquisicion real del dato del CFD, que es el seguimiento vigente (#107).
CFD_FOLLOW_UP: Final[str] = "#107"


def _artifacts() -> list[Path]:
    """Los artefactos JSON publicados, o *skip* si no hay almacen en el arbol."""
    if not REPORTS_DIR.is_dir():
        pytest.skip("no hay artefactos publicados (clon sin data/)")
    files = sorted(REPORTS_DIR.glob("*.json"))
    if not files:
        pytest.skip("no hay artefactos JSON publicados")
    return files


def _issue_values(node: object, key: str | None = None) -> Iterator[str]:
    """Recorre el payload y devuelve los valores de las claves de seguimiento."""
    if isinstance(node, Mapping):
        for child_key, child in cast("Mapping[object, object]", node).items():
            yield from _issue_values(child, str(child_key))
    elif isinstance(node, list):
        for item in cast("list[object]", node):
            yield from _issue_values(item, key)
    elif isinstance(node, str) and key in ISSUE_KEYS:
        yield node


def test_108_no_published_artifact_presents_50_as_a_closed_frontier() -> None:
    """Ningun artefacto publicado declara `#50` (cerrada) como frontera pendiente."""
    offenders: dict[str, list[str]] = {}
    for path in _artifacts():
        payload = cast("Mapping[str, object]", json.loads(path.read_text(encoding="utf-8")))
        hits = [value for value in _issue_values(payload) if CLOSED_FRONTIER in value]
        if hits:
            offenders[path.name] = hits
    assert not offenders, f"artefactos con `{CLOSED_FRONTIER}` como frontera pendiente: {offenders}"


def test_108_the_cfd_follow_up_is_107_somewhere() -> None:
    """Algun artefacto publicado declara `#107` como seguimiento del CFD."""
    seen = False
    for path in _artifacts():
        payload = cast("Mapping[str, object]", json.loads(path.read_text(encoding="utf-8")))
        if any(CFD_FOLLOW_UP in value for value in _issue_values(payload)):
            seen = True
    assert seen, f"ningun artefacto declara `{CFD_FOLLOW_UP}` como seguimiento del CFD"


def test_108_the_pipeline_does_not_present_60_as_an_open_blocker() -> None:
    """El `net_metrics` del pipeline no presenta `#60` (cerrada) como bloqueante **abierto**.

    `#60` puede seguir citandose como **procedencia** del `R` (``r_issue`` y el motivo), pero
    no como el bloqueante pendiente: eso es lo que dejaba el artefacto viejo (`not_computable`
    culpando a `#60`).
    """
    pipelines = sorted(REPORTS_DIR.glob("pipeline_backtest_*.json"))
    if not pipelines:
        pytest.skip("no hay artefacto del pipeline publicado")
    for path in pipelines:
        payload = cast("Mapping[str, object]", json.loads(path.read_text(encoding="utf-8")))
        net = payload.get("net_metrics")
        if not isinstance(net, Mapping):
            continue
        net_map = cast("Mapping[str, object]", net)
        state = net_map.get("state")
        assert state != "not_computable", f"{path.name}: `net_metrics` sigue no computable"
        follow_ups = net_map.get("follow_ups")
        if isinstance(follow_ups, list):
            assert "#60" not in cast("list[object]", follow_ups), (
                f"{path.name} presenta #60 (cerrada) como seguimiento"
            )
