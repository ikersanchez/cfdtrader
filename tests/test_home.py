"""Test trivial de bootstrap (tarea #1).

No vale por lo que comprueba, sino porque demuestra que el repositorio es
instalable y que ``uv run pytest`` recorre el ciclo completo sin fallar.
"""

from __future__ import annotations

import cfdtrader


def test_package_is_importable() -> None:
    assert cfdtrader.__name__ == "cfdtrader"


def test_version_is_defined() -> None:
    assert isinstance(cfdtrader.__version__, str)
    assert cfdtrader.__version__.count(".") == 2
