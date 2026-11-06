"""The platform never calls Celery's ``revoke``.

Cancel and pause are database writes only: the claim refuses a cancelled, paused or terminal row.
A revoke is remembered by workers, so a later ``send_task`` with the same task id (a resume, a
re-dispatch) would be skipped as well (spike S1 item 5, review RR-1). Any call named ``revoke``
in ``src/`` or ``scripts/`` fails, ``terminate=`` or not (owner decision D-71).
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_ROOTS = ('src', 'scripts')


def revoke_calls(source: str) -> list[int]:
    """Line numbers of every call to something named ``revoke``."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', '')
            if name == 'revoke':
                lines.append(node.lineno)
    return lines


def test_a_planted_revoke_is_found() -> None:
    planted = (
        'def cancel(app, job_id):\n'
        '    app.control.revoke(job_id)\n'
        '    revoke(job_id, terminate=True)\n'
    )
    assert revoke_calls(planted) == [2, 3]


def test_a_task_level_revoke_is_found() -> None:
    assert revoke_calls('result.revoke()\n') == [1]


def test_other_calls_are_not_flagged() -> None:
    assert revoke_calls('app.control.inspect().active()\nrevoked = 1\n') == []


def test_the_real_tree_never_calls_revoke() -> None:
    offenders = [
        f'{path.relative_to(REPO_ROOT)}:{line}'
        for root in SCANNED_ROOTS
        for path in (REPO_ROOT / root).rglob('*.py')
        for line in revoke_calls(path.read_text(encoding='utf-8'))
    ]
    assert offenders == []
