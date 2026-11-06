"""Every writer that puts a class on an item takes the class from the registry (D-47).

An item stores the registry's id and spelling. The class fields are built by ``ItemLabel`` /
``human_move_class_update`` / ``candidate_move_update``. A call to one of them outside
``src/services/classes/`` that passes a literal class name or id (a class that was made up, not
read from or ensured in the registry) is refused. The id and the name must arrive as values, so
they can only come from ``src.services.classes`` (``get_class_registry``, ``resolve``) or from a
plan that was resolved against it.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ITEM_LABEL_BUILDERS = frozenset({'human', 'imported', 'propagated'})
FUNCTION_BUILDERS = frozenset({'human_move_class_update', 'candidate_move_update'})
CLASS_ARGS = frozenset({'class_id', 'class_name'})


def _is_builder(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in ITEM_LABEL_BUILDERS:
        return isinstance(func.value, ast.Name) and func.value.id == 'ItemLabel'
    return isinstance(func, ast.Name) and func.id in FUNCTION_BUILDERS


def _made_up(value: ast.expr) -> bool:
    return isinstance(value, ast.Constant | ast.JoinedStr) and not (
        isinstance(value, ast.Constant) and value.value is None
    )


def offenders(paths: list[Path]) -> list[str]:
    bad: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        bad.extend(
            f'{path.name}:{node.lineno}'
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _is_builder(node)
            and any(kw.arg in CLASS_ARGS and _made_up(kw.value) for kw in node.keywords)
        )
    return bad


def _tree_paths() -> list[Path]:
    skip = (
        REPO_ROOT / 'src' / 'services' / 'classes',
        REPO_ROOT / 'src' / 'services' / 'curation' / 'class_label.py',
    )
    return [
        p
        for root in ('src', 'scripts')
        for p in (REPO_ROOT / root).rglob('*.py')
        if not any(p.is_relative_to(s) for s in skip)
    ]


def test_a_planted_writer_with_made_up_names_fails(tmp_path: Path) -> None:
    planted = tmp_path / 'writer.py'
    planted.write_text(
        'def go():\n'
        "    a = ItemLabel.human(class_id=3, class_name='made up')\n"
        "    b = human_move_class_update({}, class_id=1, class_name=f'x{1}', now='t')\n"
        '    return a, b\n',
        encoding='utf-8',
    )
    assert offenders([planted]) == ['writer.py:2', 'writer.py:3']


def test_a_writer_that_passes_resolved_values_is_not_flagged(tmp_path: Path) -> None:
    ok = tmp_path / 'writer.py'
    ok.write_text(
        'async def go(reg):\n'
        '    entry = (await reg.snapshot()).get(1)\n'
        '    return ItemLabel.human(class_id=entry.class_id, class_name=entry.class_name)\n',
        encoding='utf-8',
    )
    assert offenders([ok]) == []


def test_the_real_tree_has_no_writer_with_made_up_names() -> None:
    assert offenders(_tree_paths()) == []
