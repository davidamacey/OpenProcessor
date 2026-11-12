#!/usr/bin/env python3
"""Pre-commit guard: the API keeps no per-process state that another process would not share.

The API runs many uvicorn workers and replicas, and any request may land on any of them. Two
patterns break "the same answers with 1 and N workers":

* ``asyncio.create_task`` / ``loop.create_task`` in ``src/routers``, ``src/services`` and
  ``src/app``: a task spawned for a request dies with its process and is lost to every other
  one. Concurrency that must finish before the answer uses ``asyncio.TaskGroup``
  (https://docs.python.org/3/library/asyncio-task.html#task-groups); work that outlives the
  request goes to a job (Celery).
* A module-level ``asyncio`` / ``threading`` lock, semaphore, event or condition, or a
  module-level mutable ``dict`` / ``set`` / ``list`` (a lower-case name, the project's mark of
  a variable rather than a constant): each process holds its own copy, so it is correct only as
  a per-process cache with a revision check, a staleness window or a local throttle.

Read with ``ast``, so a docstring that names one does not count. Today's uses live in
``baselines/process_state.txt`` (``path (kind)<TAB>count<TAB>reason``; the reason says why that
use is correct per process) and can only shrink (see ``_baseline.py``). A lock never guards
shared state across processes: use a database claim, OCC or a lease (``src/db/repos/jobs.py``).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / 'scripts' / 'codegen' / 'baselines' / 'process_state.txt'

STATE_SCOPE = ('src/',)
TASK_SCOPE = ('src/routers/', 'src/services/', 'src/app/')

SYNC_PRIMITIVES = frozenset(
    {'Lock', 'RLock', 'Semaphore', 'BoundedSemaphore', 'Event', 'Condition', 'Barrier'}
)
PRIMITIVE_MODULES = frozenset({'asyncio', 'threading'})
CONTAINER_CALLS = frozenset(
    {
        'dict',
        'set',
        'list',
        'defaultdict',
        'OrderedDict',
        'deque',
        'Counter',
        'WeakValueDictionary',
        'WeakKeyDictionary',
        'WeakSet',
    }
)

KIND_TASK = 'create_task'
KIND_PRIMITIVE = 'module lock'
KIND_CONTAINER = 'module container'


def tracked_files(repo_root: Path) -> list[str]:
    out = subprocess.run(  # nosec B603 B607  fixed argv, git from PATH
        ['git', 'ls-files', '-z'], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.split('\0') if p]


def _call_name(node: ast.Call) -> tuple[str, str]:
    """``(owner, name)`` of a call: ``asyncio.Lock()`` is ``('asyncio', 'Lock')``."""
    func = node.func
    if isinstance(func, ast.Attribute):
        owner = func.value.id if isinstance(func.value, ast.Name) else ''
        return owner, func.attr
    return '', getattr(func, 'id', '')


def _is_primitive(value: ast.expr, imported: dict[str, str]) -> bool:
    if not isinstance(value, ast.Call):
        return False
    owner, name = _call_name(value)
    if name not in SYNC_PRIMITIVES:
        return False
    return owner in PRIMITIVE_MODULES or imported.get(name, '') in PRIMITIVE_MODULES


def _is_container(value: ast.expr) -> bool:
    if isinstance(value, ast.Dict | ast.Set | ast.List | ast.DictComp | ast.SetComp | ast.ListComp):
        return True
    return isinstance(value, ast.Call) and _call_name(value)[1] in CONTAINER_CALLS


def _targets(node: ast.stmt) -> list[ast.expr]:
    if isinstance(node, ast.Assign):
        return list(node.targets)
    if isinstance(node, ast.AnnAssign):
        return [node.target]
    return []


def _value(node: ast.stmt) -> ast.expr | None:
    return node.value if isinstance(node, ast.Assign | ast.AnnAssign) else None


def _is_empty(value: ast.expr) -> bool:
    """An empty container exists to be filled, whatever its name."""
    if isinstance(value, ast.Dict):
        return not value.keys
    if isinstance(value, ast.Set | ast.List):
        return not value.elts
    return isinstance(value, ast.Call) and not value.args and not value.keywords


def _is_variable_name(name: str) -> bool:
    return name.lstrip('_') != name.lstrip('_').upper()


def _is_loop_task(node: ast.Call) -> bool:
    """``asyncio.create_task(..)`` or ``<loop>.create_task(..)``; ``TaskGroup.create_task`` is the pattern."""
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == 'create_task'):
        return False
    owner = func.value
    if isinstance(owner, ast.Call):
        return _call_name(owner)[1] in {'get_running_loop', 'get_event_loop'}
    return isinstance(owner, ast.Name) and (owner.id == 'asyncio' or 'loop' in owner.id)


def _create_tasks(tree: ast.AST) -> int:
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call) and _is_loop_task(node))


def _primitive_count(tree: ast.Module) -> tuple[int, int]:
    """``(locks, containers)`` at module level."""
    imported = {
        a.asname or a.name: node.module or ''
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for a in node.names
    }
    locks = containers = 0
    for node in tree.body:
        value = _value(node)
        names = [t.id for t in _targets(node) if isinstance(t, ast.Name)]
        if value is None or not names or names == ['__all__']:
            continue
        if _is_primitive(value, imported):
            locks += 1
        elif _is_container(value) and (
            _is_empty(value) or any(_is_variable_name(n) for n in names)
        ):
            containers += 1
    return locks, containers


def scan(repo_root: Path, files: list[str] | None = None) -> tuple[dict[str, int], list[str]]:
    """Return ``({"path (kind)": uses}, unreadable files)``."""
    found: dict[str, int] = {}
    broken: list[str] = []
    for rel in files if files is not None else tracked_files(repo_root):
        if not (rel.endswith('.py') and rel.startswith(STATE_SCOPE)):
            continue
        try:
            tree = ast.parse((repo_root / rel).read_text(encoding='utf-8'), filename=rel)
        except (SyntaxError, UnicodeDecodeError, OSError):
            broken.append(rel)
            continue
        locks, containers = _primitive_count(tree)
        tasks = _create_tasks(tree) if rel.startswith(TASK_SCOPE) else 0
        for kind, n in ((KIND_PRIMITIVE, locks), (KIND_CONTAINER, containers), (KIND_TASK, tasks)):
            if n:
                found[f'{rel} ({kind})'] = n
    return found, broken


def main(argv: list[str] | None = None) -> int:
    return int(
        _baseline.run_scan_ratchet(
            argv,
            doc=__doc__,
            baseline_path=BASELINE_PATH,
            repo_root=REPO_ROOT,
            scan=scan,
            unit='use(s)',
            unreadable='cannot be parsed, so its process state is unchecked',
            failure=(
                'new per-process state in the API. A task spawned for a request, a lock or a '
                'module-level mutable container exists once per worker. Use asyncio.TaskGroup, a '
                'job, or a database claim; if the use is a per-process cache with a revision check '
                'or a local throttle, add it to baselines/process_state.txt with that reason.'
            ),
        )
    )


if __name__ == '__main__':
    sys.exit(main())
