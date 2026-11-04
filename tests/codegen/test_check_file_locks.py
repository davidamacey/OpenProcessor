"""check_file_locks.py: no file lock guards cross-process state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import check_file_locks as guard
import pytest


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from conftest import ScratchRepo


def run(repo: ScratchRepo, baseline: Path, *extra: str) -> int:
    return guard.main(['--root', str(repo.root), '--baseline', str(baseline), *extra])


@pytest.mark.parametrize(
    ('path', 'source'),
    [
        ('src/a.py', 'import fcntl\n\ndef f(fd):\n    fcntl.flock(fd, fcntl.LOCK_EX)\n'),
        ('src/a.py', 'import fcntl\n\ndef f(fd):\n    fcntl.lockf(fd, fcntl.LOCK_EX)\n'),
        (
            'scripts/a.py',
            'from x import exclusive_start_lock\n\nwith exclusive_start_lock(p):\n    pass\n',
        ),
        ('export/a.py', 'import job_lock\n\nwith job_lock.exclusive_file_lock(p):\n    pass\n'),
        ('docker/a.py', 'import filelock\n'),
        ('src/a.py', 'from portalocker import Lock\n'),
    ],
)
def test_a_new_file_lock_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path], path: str, source: str
) -> None:
    scratch_repo.write(path, source)
    assert run(scratch_repo, baseline_file('')) == 1


@pytest.mark.parametrize(
    ('path', 'source'),
    [
        ('src/a.py', '"""Not a flock: fcntl.flock(fd) in a docstring."""\n'),
        ('src/a.py', '# exclusive_start_lock(x) was removed\nx = 1\n'),
        ('src/a.py', 'import fcntl\n\nfcntl.fcntl(1, fcntl.F_GETFD)\n'),
        ('tests/a.py', 'import fcntl\nfcntl.flock(1, 2)\n'),
        ('docs/a.py', 'import filelock\n'),
    ],
)
def test_a_mention_or_an_out_of_scope_file_passes(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path], path: str, source: str
) -> None:
    scratch_repo.write(path, source)
    assert run(scratch_repo, baseline_file('')) == 0


def test_a_baselined_file_may_not_gain_a_use(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/a.py', 'import fcntl\nfcntl.flock(1, 2)\nfcntl.flock(1, 8)\n')
    assert run(scratch_repo, baseline_file('src/a.py\t1\tB4: moves to a stream\n')) == 1


def test_a_baselined_use_passes_and_a_removed_one_must_leave_the_baseline(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/a.py', 'import fcntl\nfcntl.flock(1, 2)\n')
    baseline = baseline_file('src/a.py\t1\tB4: moves to a stream\n')
    assert run(scratch_repo, baseline) == 0
    scratch_repo.write('src/a.py', 'x = 1\n')
    assert run(scratch_repo, baseline) == 1  # stale entry


def test_the_real_tree_passes() -> None:
    assert guard.main([]) == 0
