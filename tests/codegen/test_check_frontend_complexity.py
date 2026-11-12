"""check_frontend_complexity.py: frontend functions over 15 may not be new or grow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import check_frontend_complexity as guard
import pytest


if TYPE_CHECKING:
    from pathlib import Path

    from conftest import ScratchRepo


pytestmark = pytest.mark.skipif(
    not (guard.REPO_ROOT / 'frontend' / 'node_modules' / '.bin' / 'eslint').exists(),
    reason='frontend npm ci not run',
)


def branches(name: str, n: int) -> str:
    body = ''.join(f'  if (x === {i}) return {i};\n' for i in range(n))
    return f'export function {name}(x: number): number {{\n{body}  return -1;\n}}\n'


def run(repo: ScratchRepo, baseline: Path, *extra: str) -> int:
    frontend = repo.root / 'frontend'
    for name in (
        'node_modules',
        'eslint.config.js',
        'eslint.complexity.config.js',
        'sveltekit.options.js',
        'svelte.config.js',
        'tsconfig.json',
        'package.json',
    ):
        link = frontend / name
        if not link.exists() and not link.is_symlink():
            frontend.mkdir(parents=True, exist_ok=True)
            link.symlink_to(guard.REPO_ROOT / 'frontend' / name)
    return guard.main(['--root', str(repo.root), '--baseline', str(baseline), *extra])


def test_a_16_branch_function_fails(scratch_repo: ScratchRepo, baseline_file) -> None:
    scratch_repo.write('frontend/src/probe.ts', branches('f', 16))
    assert run(scratch_repo, baseline_file('')) == 1


def test_a_10_branch_function_passes(scratch_repo: ScratchRepo, baseline_file) -> None:
    scratch_repo.write('frontend/src/probe.ts', branches('f', 10))
    assert run(scratch_repo, baseline_file('')) == 0


def test_a_baselined_function_may_stay_but_not_grow(
    scratch_repo: ScratchRepo, baseline_file
) -> None:
    scratch_repo.write('frontend/src/probe.ts', branches('f', 20))
    base = baseline_file('frontend/src/probe.ts::f\t21\tlegacy\n')
    assert run(scratch_repo, base) == 0
    scratch_repo.write('frontend/src/probe.ts', branches('f', 25))
    assert run(scratch_repo, base) == 1


def test_a_stale_entry_fails_and_tighten_drops_it(scratch_repo: ScratchRepo, baseline_file) -> None:
    scratch_repo.write('frontend/src/probe.ts', branches('f', 3))
    base = baseline_file('frontend/src/probe.ts::f\t21\tlegacy\n')
    assert run(scratch_repo, base) == 1
    assert run(scratch_repo, base, '--tighten') == 0
    assert base.read_text() == ''


def test_the_committed_baseline_matches_the_tree() -> None:
    """The hook passes on the real tree in both modes, so ``--all-files`` and a staged run agree."""
    assert guard.main([]) == 0
    files = [
        f'frontend/{p.relative_to(guard.REPO_ROOT / "frontend").as_posix()}'
        for p in sorted((guard.REPO_ROOT / 'frontend' / 'src').rglob('*'))
        if p.suffix in {'.js', '.ts', '.svelte'}
    ]
    assert guard.main(files) == 0
