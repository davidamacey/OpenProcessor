"""The health module probes one dependency after another: no task fan-out (owner rule D-84).
asyncio docs: https://docs.python.org/3/library/asyncio-task.html"""

from __future__ import annotations

import ast
from pathlib import Path


HEALTH = Path(__file__).resolve().parents[1] / 'src' / 'routers' / 'health.py'
FORBIDDEN = {'create_task', 'gather', 'TaskGroup'}


def test_health_router_has_no_fanout() -> None:
    tree = ast.parse(HEALTH.read_text())
    used = {
        node.attr if isinstance(node, ast.Attribute) else node.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Attribute, ast.Name))
    }
    assert not used & FORBIDDEN, sorted(used & FORBIDDEN)
