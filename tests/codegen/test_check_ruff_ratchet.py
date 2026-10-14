"""check_ruff_ratchet.py: per-function complexity and per-file broad-except counts may not grow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import check_ruff_ratchet as guard
import pytest


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from conftest import ScratchRepo


def branches(name: str, n: int) -> str:
    body = ''.join(f'    if x == {i}:\n        return {i}\n' for i in range(n))
    return f'def {name}(x):\n{body}    return -1\n'


def broad(n: int) -> str:
    handler = '    try:\n        pass\n    except Exception:\n        return None\n'
    return 'def f():\n' + handler * n


def run(repo: ScratchRepo, complexity: Path, broad_except: Path, *extra: str) -> int:
    return guard.main(
        [
            '--root',
            str(repo.root),
            '--complexity-baseline',
            str(complexity),
            '--broad-except-baseline',
            str(broad_except),
            *extra,
        ]
    )


@pytest.fixture
def empty(baseline_file: Callable[[str], Path], tmp_path: Path) -> tuple[Path, Path]:
    other = tmp_path / 'other.txt'
    other.write_text('', encoding='utf-8')
    return baseline_file(''), other


def test_a_16_branch_function_fails(scratch_repo: ScratchRepo, empty: tuple[Path, Path]) -> None:
    scratch_repo.write('src/_probe.py', branches('f', 16))
    assert run(scratch_repo, *empty) == 1


def test_a_14_branch_function_passes(scratch_repo: ScratchRepo, empty: tuple[Path, Path]) -> None:
    scratch_repo.write('src/_probe.py', branches('f', 14))
    assert run(scratch_repo, *empty) == 0


def test_a_baselined_function_may_stay_but_not_grow(
    scratch_repo: ScratchRepo, empty: tuple[Path, Path]
) -> None:
    scratch_repo.write('src/_probe.py', branches('f', 20))
    complexity, other = empty
    complexity.write_text('src/_probe.py::f\t21\ttop function, split in PR 6\n', encoding='utf-8')
    assert run(scratch_repo, complexity, other) == 0
    scratch_repo.write('src/_probe.py', branches('f', 25))
    assert run(scratch_repo, complexity, other) == 1


def test_a_stale_complexity_entry_fails_and_tighten_drops_it(
    scratch_repo: ScratchRepo, empty: tuple[Path, Path]
) -> None:
    scratch_repo.write('src/_probe.py', branches('f', 3))
    complexity, other = empty
    complexity.write_text('src/_probe.py::f\t21\tsplit in PR 6\n', encoding='utf-8')
    assert run(scratch_repo, complexity, other) == 1
    assert run(scratch_repo, complexity, other, '--tighten') == 0
    assert complexity.read_text() == ''


def test_a_new_broad_except_in_a_baselined_file_fails(
    scratch_repo: ScratchRepo, empty: tuple[Path, Path]
) -> None:
    scratch_repo.write('src/_probe.py', broad(2))
    complexity, other = empty
    other.write_text('src/_probe.py\t2\ttriaged in PR 3\n', encoding='utf-8')
    assert run(scratch_repo, complexity, other) == 0
    scratch_repo.write('src/_probe.py', broad(3))
    assert run(scratch_repo, complexity, other) == 1


def test_a_broad_except_in_a_new_file_fails(
    scratch_repo: ScratchRepo, empty: tuple[Path, Path]
) -> None:
    scratch_repo.write('scripts/_probe.py', broad(1))
    assert run(scratch_repo, *empty) == 1


def test_fewer_broad_excepts_than_the_baseline_fails_until_tightened(
    scratch_repo: ScratchRepo, empty: tuple[Path, Path]
) -> None:
    scratch_repo.write('src/_probe.py', broad(1))
    complexity, other = empty
    other.write_text('src/_probe.py\t4\ttriaged in PR 3\n', encoding='utf-8')
    assert run(scratch_repo, complexity, other) == 1
    assert run(scratch_repo, complexity, other, '--tighten') == 0
    assert other.read_text() == 'src/_probe.py\t1\ttriaged in PR 3\n'


def test_same_named_methods_get_separate_keys(
    scratch_repo: ScratchRepo, empty: tuple[Path, Path]
) -> None:
    src = branches('f', 16) + branches('f', 17)
    scratch_repo.write('src/_probe.py', src)
    found = guard.complexity_violations(scratch_repo.root)
    assert sorted(found) == ['src/_probe.py::f', 'src/_probe.py::f#2']


def test_third_party_patches_are_excluded(
    scratch_repo: ScratchRepo, empty: tuple[Path, Path]
) -> None:
    scratch_repo.write('src/ultralytics_patches/p.py', branches('f', 30) + broad(5))
    assert run(scratch_repo, *empty) == 0


def test_a_ruff_failure_fails_closed(scratch_repo: ScratchRepo, empty: tuple[Path, Path]) -> None:
    scratch_repo.write('src/_probe.py', 'def broken(:\n')
    assert run(scratch_repo, *empty) == 1


def test_the_real_repo_passes() -> None:
    assert guard.main([]) == 0
