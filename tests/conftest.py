"""Configuración común a toda la sesión de tests.

**A22: ningún test escribe en el `data/` del repositorio.** Comprobar que el
directorio no existe o no tiene Parquet no demuestra eso: la ingesta real (A14)
escribe ahí a propósito, así que un `data/` poblado hacía fallar la puerta sin
que ningún test hubiera hecho nada (#54). Lo que hay que demostrar es que la
sesión **no cambia** nada, y eso se mide con una huella antes y después.

**A27 de #16: la misma guardia para el `runs/` del repositorio**, que es el registro de
experimentos. La huella de `data/` no lo cubría y `runs/` está gitignorado
(`.gitignore` línea 55): sin esta segunda fixture, un test podría escribir en el registro
del repositorio sin que ninguna puerta se enterara.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directorio de datos del repositorio (el real, no el de los tests).
REPOSITORY_DATA = REPO_ROOT / "data"

#: Registro de experimentos del repositorio (gitignorado, `.gitignore` línea 55).
REPOSITORY_RUNS = REPO_ROOT / "runs"


def fingerprint(root: Path) -> dict[str, str]:
    """Ruta relativa → sha256 de cada fichero. Detecta altas, bajas y cambios."""
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="session", autouse=True)
def _repository_data_is_untouched() -> Iterator[None]:
    """La sesión de tests no puede crear, borrar ni reescribir nada de `data/`."""
    before = fingerprint(REPOSITORY_DATA)
    yield
    after = fingerprint(REPOSITORY_DATA)

    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(path for path in set(before) & set(after) if before[path] != after[path])
    assert not (added or removed or changed), (
        "la sesión de tests ha modificado el data/ del repositorio: "
        f"creados={added} borrados={removed} modificados={changed}"
    )


@pytest.fixture(scope="session", autouse=True)
def _repository_runs_is_untouched() -> Iterator[None]:
    """La sesión de tests tampoco puede tocar la huella el `runs/` del repositorio (A27).

    `runs/` es el registro de experimentos de #16 y está gitignorado: la fixture de `data/`
    no lo cubría, así que el registro se huella aquí con la **misma** función.
    """
    before = fingerprint(REPOSITORY_RUNS)
    yield
    after = fingerprint(REPOSITORY_RUNS)

    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(path for path in set(before) & set(after) if before[path] != after[path])
    assert not (added or removed or changed), (
        "la sesión de tests ha modificado el runs/ del repositorio: "
        f"creados={added} borrados={removed} modificados={changed}"
    )
