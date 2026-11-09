"""check_file_locks.py: no file lock guards cross-process state, and there is no baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import check_file_locks as guard
import pytest


if TYPE_CHECKING:
    from conftest import ScratchRepo


def run(repo: ScratchRepo) -> int:
    return guard.main(['--root', str(repo.root)])


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
def test_a_planted_file_lock_fails(scratch_repo: ScratchRepo, path: str, source: str) -> None:
    scratch_repo.write(path, source)
    assert run(scratch_repo) == 1


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
    scratch_repo: ScratchRepo, path: str, source: str
) -> None:
    scratch_repo.write(path, source)
    assert run(scratch_repo) == 0


def test_an_unparsable_file_is_not_waved_through(scratch_repo: ScratchRepo) -> None:
    scratch_repo.write('src/a.py', 'def broken(:\n')
    assert run(scratch_repo) == 1


def test_the_baseline_file_is_gone() -> None:
    assert not (guard.REPO_ROOT / 'scripts' / 'codegen' / 'baselines' / 'file_locks.txt').exists()


def test_the_real_tree_passes() -> None:
    assert guard.main([]) == 0
