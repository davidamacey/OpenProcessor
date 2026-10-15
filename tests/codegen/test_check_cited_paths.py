"""check_cited_paths.py: a comment or docstring that cites a repo file must cite one that exists."""

from __future__ import annotations

from typing import TYPE_CHECKING

import check_cited_paths as guard
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
        ('src/a.py', '# see docs/design/gone.md for why\nx = 1\n'),
        ('src/a.py', '"""Mirrors scripts/lib/missing.sh."""\n'),
        ('scripts/a.sh', '# see docker/nowhere/x.py\necho hi\n'),
        ('frontend/src/lib/a.ts', '// see docs/design/gone-plan.md\nexport const a = 1;\n'),
    ],
)
def test_a_citation_of_a_missing_file_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path], path: str, source: str
) -> None:
    scratch_repo.write(path, source)
    assert run(scratch_repo, baseline_file('')) == 1


@pytest.mark.parametrize(
    ('path', 'source'),
    [
        ('src/a.py', '# see docs/real.md\nx = 1\n'),
        ('src/a.py', "x = 'docs/design/gone.md is data, not a comment'\n"),
        ('src/a.py', '# a glob like docs/design/*.md or docs/<name>.md is not a path\n'),
        ('src/a.py', '# see https://example.test/docs/design/gone.md\n'),
    ],
)
def test_real_paths_data_and_globs_pass(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path], path: str, source: str
) -> None:
    scratch_repo.write('docs/real.md', '# real\n')
    scratch_repo.write(path, source)
    assert run(scratch_repo, baseline_file('')) == 0


def test_a_baselined_citation_passes_and_a_fixed_one_must_be_removed(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/a.py', '# see docs/design/gone.md\n')
    base = baseline_file('src/a.py\t1\tcites a plan that left the tree\n')
    assert run(scratch_repo, base) == 0
    scratch_repo.write('src/a.py', '# no citation now\n')
    assert run(scratch_repo, base) == 1
