"""A ``jobs`` row is born in one place and a task message is sent from one place (02b section 2.4).

``src/db/repos/jobs.py`` (and ``jobs_load.py``, the 0.6.0 upgrade's load) are the only modules that insert into ``jobs`` (every start goes through
``start_job`` and ``insert_queued``), and ``src/clients/task_queue.py`` is the only one that calls
``send_task``. An AST scan fails on any other insert, on a ``Job(...)`` model construction, on
``insert(Job)``, and on any ``send_task`` call, so a second start path cannot grow back.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_ROOTS = ('src', 'scripts', 'docker', 'export')
# The repository: the start path, and the upgrade's load next to it (split out for size).
JOBS_WRITERS = ('src/db/repos/jobs.py', 'src/db/repos/jobs_load.py')
SENDER = 'src/clients/task_queue.py'
_INSERT_SQL = re.compile(r'\binsert\s+into\s+"?jobs"?\b', re.IGNORECASE)


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return getattr(func, 'id', '')


def jobs_inserts(source: str) -> list[int]:
    """Lines that insert into ``jobs``: SQL text, ``insert(Job)`` or a ``Job(...)`` construction."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _INSERT_SQL.search(node.value):
                lines.append(node.lineno)
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            first = node.args[0] if node.args else None
            if name == 'Job' or (
                name == 'insert' and isinstance(first, ast.Name) and first.id == 'Job'
            ):
                lines.append(node.lineno)
    return sorted(lines)


def send_task_calls(source: str) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and _call_name(node) == 'send_task'
    ]


def _offenders(scan, allowed: tuple[str, ...]) -> list[str]:
    found = []
    for root in SCANNED_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob('*.py')):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in allowed:
                continue
            found += [f'{rel}:{line}' for line in scan(path.read_text(encoding='utf-8'))]
    return found


def test_a_planted_insert_is_found() -> None:
    planted = (
        "session.execute(text('INSERT INTO jobs (kind) VALUES (1)'))\n"
        'session.execute(text("insert into  jobs(kind) values (\'x\')"))\n'
        'session.execute(insert(Job).values(kind=1))\n'
        'session.add(Job(kind=1))\n'
        'models.Job(kind=1)\n'
    )
    assert jobs_inserts(planted) == [1, 2, 3, 4, 5]


def test_other_tables_and_reads_are_not_flagged() -> None:
    clean = (
        "session.execute(text('INSERT INTO job_attempts (job_id) VALUES (1)'))\n"
        "session.execute(text('UPDATE jobs SET status = 1'))\n"
        'session.execute(select(Job))\n'
        'JobRow(id=1)\n'
    )
    assert jobs_inserts(clean) == []


def test_a_planted_send_task_is_found() -> None:
    planted = "app.send_task('op.x', args=['1'])\nsend_task('op.y')\n"
    assert send_task_calls(planted) == [1, 2]


def test_the_real_tree_inserts_into_jobs_only_in_the_repository() -> None:
    assert _offenders(jobs_inserts, JOBS_WRITERS) == []


def test_the_real_tree_sends_tasks_only_from_the_queue_client() -> None:
    assert _offenders(send_task_calls, (SENDER,)) == []


def test_the_allowed_modules_do_use_their_calls() -> None:
    """The exemption is not vacuous: each allowed module really holds the call."""
    for writer in JOBS_WRITERS:
        assert jobs_inserts((REPO_ROOT / writer).read_text(encoding='utf-8'))
    assert send_task_calls((REPO_ROOT / SENDER).read_text(encoding='utf-8'))
