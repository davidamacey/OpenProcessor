"""check_process_state.py: the API keeps no per-process state outside the reasoned baseline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import check_process_state as guard
import pytest


if TYPE_CHECKING:
    from collections.abc import Callable

    from conftest import ScratchRepo

REPO_ROOT = Path(__file__).resolve().parents[2]


def run(repo: ScratchRepo, baseline: Path, *extra: str) -> int:
    return guard.main(['--root', str(repo.root), '--baseline', str(baseline), *extra])


@pytest.mark.parametrize(
    ('path', 'source'),
    [
        ('src/services/a.py', '_cache: dict[str, int] = {}\n'),
        ('src/services/a.py', 'cache = {"a": 1}\n'),
        ('src/utils/a.py', 'import threading\n\n_registry = set()\n'),
        ('src/services/a.py', 'from collections import OrderedDict\n\n_LRU = OrderedDict()\n'),
        ('src/services/a.py', 'import threading\n\n_lock = threading.Lock()\n'),
        ('src/services/a.py', 'import asyncio\n\nGATE = asyncio.Semaphore(4)\n'),
        ('src/services/a.py', 'from asyncio import Event\n\nstop = Event()\n'),
        ('src/routers/a.py', 'import asyncio\n\nasync def f(c):\n    asyncio.create_task(c)\n'),
        (
            'src/app/a.py',
            'import asyncio\n\nasync def f(c):\n    asyncio.get_running_loop().create_task(c)\n',
        ),
        ('src/services/a.py', 'async def f(loop, c):\n    loop.create_task(c)\n'),
    ],
)
def test_new_per_process_state_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path], path: str, source: str
) -> None:
    scratch_repo.write(path, source)
    assert run(scratch_repo, baseline_file('')) == 1


@pytest.mark.parametrize(
    ('path', 'source'),
    [
        ('src/services/a.py', 'LIMITS = {"a": 1}\nNAMES = ["x"]\n__all__ = ["LIMITS"]\n'),
        (
            'src/services/a.py',
            'import asyncio\n\nasync def f(c):\n    async with asyncio.TaskGroup() as g:\n        g.create_task(c)\n',
        ),
        (
            'src/services/a.py',
            'import asyncio\n\nasync def f():\n    lock = asyncio.Lock()\n    cache = {}\n    return lock, cache\n',
        ),
        ('src/services/a.py', 'class C:\n    seen = {}\n'),
        ('src/clients/a.py', 'import asyncio\n\nasync def f(c):\n    asyncio.create_task(c)\n'),
        ('src/services/a.py', '"""asyncio.create_task and threading.Lock() in prose."""\n'),
        ('tests/a.py', '_cache = {}\n'),
    ],
)
def test_constants_taskgroups_locals_and_out_of_scope_pass(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path], path: str, source: str
) -> None:
    scratch_repo.write(path, source)
    assert run(scratch_repo, baseline_file('')) == 0


def test_an_allowlisted_entry_with_its_reason_passes(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/services/a.py', '_cache: dict[str, int] = {}\n')
    baseline = baseline_file(
        'src/services/a.py (module container)\t1\tper-process cache with a revision check\n'
    )
    assert run(scratch_repo, baseline) == 0


def test_an_allowlisted_file_that_gains_a_second_use_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/services/a.py', '_a = {}\n_b = []\n')
    baseline = baseline_file('src/services/a.py (module container)\t1\tper-process cache\n')
    assert run(scratch_repo, baseline) == 1


def test_a_stale_entry_fails_until_tightened(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/services/a.py', 'X = 1\n')
    baseline = baseline_file('src/services/a.py (module container)\t1\tper-process cache\n')
    assert run(scratch_repo, baseline) == 1
    assert run(scratch_repo, baseline, '--tighten') == 0
    assert baseline.read_text() == ''


def test_a_baseline_line_without_a_reason_is_rejected(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/services/a.py', '_cache = {}\n')
    with pytest.raises(ValueError, match='reason'):
        run(scratch_repo, baseline_file('src/services/a.py (module container)\t1\t\n'))


def test_the_real_tree_passes_with_the_committed_baseline() -> None:
    assert guard.main([]) == 0
