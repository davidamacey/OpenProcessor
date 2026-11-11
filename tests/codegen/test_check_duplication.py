"""check_duplication.py: new copy/paste clones fail, baselined ones may not grow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import check_duplication as guard
import pytest


if TYPE_CHECKING:
    from pathlib import Path

    from conftest import ScratchRepo


pytestmark = pytest.mark.skipif(
    not (guard.REPO_ROOT / guard.JSCPD).exists(), reason='frontend npm ci not run'
)


def block(prefix: str, n: int = 30) -> str:
    return ''.join(
        f'    total_{i} = compute_{i}(value) + other_{i}(value) * {i}\n' for i in range(n)
    )


def module(extra: str = '') -> str:
    return f'def work(value):\n{block("a")}    return value\n{extra}'


def link_tooling(repo: ScratchRepo) -> None:
    (repo.root / 'frontend').mkdir(exist_ok=True)
    link = repo.root / 'frontend' / 'node_modules'
    if not link.exists():
        link.symlink_to(guard.REPO_ROOT / 'frontend' / 'node_modules')


def run(repo: ScratchRepo, baseline: Path, *extra: str) -> int:
    link_tooling(repo)
    return guard.main(['--root', str(repo.root), '--baseline', str(baseline), *extra])


def test_a_copied_block_fails(scratch_repo: ScratchRepo, baseline_file) -> None:
    scratch_repo.write('src/one.py', module())
    scratch_repo.write('src/two.py', module())
    assert run(scratch_repo, baseline_file('')) == 1


def test_distinct_code_passes(scratch_repo: ScratchRepo, baseline_file) -> None:
    scratch_repo.write('src/one.py', module())
    scratch_repo.write('src/two.py', 'def other(x):\n    return x + 1\n')
    assert run(scratch_repo, baseline_file('')) == 0


def test_a_baselined_pair_may_stay_but_not_grow(scratch_repo: ScratchRepo, baseline_file) -> None:
    scratch_repo.write('src/one.py', module())
    scratch_repo.write('src/two.py', module())
    base = baseline_file('')
    link_tooling(scratch_repo)
    found = guard.clone_pairs(scratch_repo.root)
    assert len(found) == 1
    ((key, lines),) = found.items()
    base.write_text(f'{key}\t{lines}\tlegacy\n', encoding='utf-8')
    assert run(scratch_repo, base) == 0
    bigger = module().replace('return value', block('b', 20) + '    return value')
    scratch_repo.write('src/one.py', bigger)
    scratch_repo.write('src/two.py', bigger)
    assert run(scratch_repo, base) == 1


def test_a_stale_entry_fails_and_tighten_drops_it(scratch_repo: ScratchRepo, baseline_file) -> None:
    scratch_repo.write('src/one.py', 'x = 1\n')
    base = baseline_file('src/one.py <-> src/two.py\t30\tlegacy\n')
    assert run(scratch_repo, base) == 1
    assert run(scratch_repo, base, '--tighten') == 0
    assert base.read_text() == ''
