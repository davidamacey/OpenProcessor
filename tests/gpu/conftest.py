"""The GPU tier runs only when asked for with ``-m gpu``.

Every offline gate selects its tests with its own ``-m 'not live and not families ...'`` expression
(Makefile, CI, ``scripts/dev/gate.sh``), which replaces the ``addopts`` one; deselecting here keeps
all of them from collecting tests that need a running Triton.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if 'gpu' in (config.getoption('-m') or '').replace('not gpu', ''):
        return
    gpu = [i for i in items if i.get_closest_marker('gpu')]
    if gpu:
        config.hook.pytest_deselected(items=gpu)
        items[:] = [i for i in items if i not in gpu]
