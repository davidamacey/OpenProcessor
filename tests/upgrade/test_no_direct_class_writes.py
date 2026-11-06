"""2F-5 (moved into 2E, D-84): the upgrade writes ``project_classes`` only through lane 2B's repository."""

from __future__ import annotations

import ast
from pathlib import Path


UPGRADE = Path(__file__).resolve().parents[2] / 'src' / 'services' / 'upgrade'
WRITERS = {'insert', 'update', 'delete'}
MODELS = {'ProjectClass'}


def _writes_classes(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
        )
        first = node.args[0] if node.args else None
        if name in MODELS:
            lines.append(node.lineno)  # constructing a row
        elif name in WRITERS and isinstance(first, ast.Name) and first.id in MODELS:
            lines.append(node.lineno)
    return lines


def test_no_module_of_the_upgrade_writes_class_rows_itself() -> None:
    offenders = {
        str(path.relative_to(UPGRADE)): lines
        for path in sorted(UPGRADE.rglob('*.py'))
        if (lines := _writes_classes(ast.parse(path.read_text(encoding='utf-8'))))
    }
    assert offenders == {}


def test_the_check_would_see_a_planted_write() -> None:
    planted = ast.parse(
        'session.add(ProjectClass(project_id=1))\nsession.execute(insert(ProjectClass))'
    )
    assert _writes_classes(planted) == [1, 2]
