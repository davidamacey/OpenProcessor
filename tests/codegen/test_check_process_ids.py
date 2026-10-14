"""check_process_ids.py: no work-package ids or history names in comments, docstrings or file names."""

from __future__ import annotations

from typing import TYPE_CHECKING

import check_process_ids as guard
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
        ('src/a.py', 'x = 1  # W8.2: keep the queue ordered\n'),
        ('src/a.py', '# round-3 review: keep\nx = 1\n'),
        ('src/a.py', 'def f():\n    """Parses the file (R7-2 fix)."""\n'),
        ('src/a.py', '"""Module docstring, see WP-A12 for the plan."""\n'),
        ('src/a.py', '# PC-14 says so\n'),
        ('src/a.py', '# Chunk 4 and Wave 3 landed this\n'),
        ('src/a.py', '# F-12 fix\n'),
        ('src/a.py', '# dqM3 regression\n'),
        ('scripts/a.sh', '# W3.1: copy first\necho hi\n'),
        ('Dockerfile', '# R5-1: pin the base\nFROM x\n'),
        ('frontend/src/lib/a.ts', '// W9.4 keeps this ordering\nexport const a = 1;\n'),
        ('frontend/src/lib/a.ts', '/**\n * Round-2 behavior.\n * round-2 fix\n */\n'),
        ('frontend/src/lib/A.svelte', '<!-- W2.1: header -->\n<p>hi</p>\n'),
        ('frontend/src/lib/A.svelte', '<script>\n  // R4-9 guard\n</script>\n'),
    ],
)
def test_a_process_id_in_a_comment_or_docstring_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path], path: str, source: str
) -> None:
    scratch_repo.write(path, source)
    assert run(scratch_repo, baseline_file('')) == 1


@pytest.mark.parametrize(
    ('path', 'source'),
    [
        ('src/a.py', "x = 'W8.2 is a string, not a comment'\n"),
        ('src/a.py', "NAME = 'F-12'\n"),
        ('src/a.py', '# the W3C spec and RGB values\n'),
        ('src/a.py', '# rounds 3 times, wavelet\n'),
        ('frontend/src/lib/a.ts', "const url = 'https://x.test/W8.2'; // fine\n"),
        ('docs/note.md', '# W8.2 is a doc, out of scope\n'),
        ('tests/a.py', '"""Plain docstring."""\n'),
    ],
)
def test_ids_outside_comments_and_docstrings_pass(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path], path: str, source: str
) -> None:
    scratch_repo.write(path, source)
    assert run(scratch_repo, baseline_file('')) == 0


@pytest.mark.parametrize(
    'path',
    [
        'tests/test_r9_x.py',
        'tests/projects/test_r7_clone.py',
        'tests/test_w3_thing.py',
        'tests/test_wave2_thing.py',
        'tests/test_round3.py',
        'scripts/bench/wave0_probes.py',
        'frontend/src/lib/logicMoves.ts',
        'frontend/e2e/test_interactivePass.py',
        'frontend/src/lib/a1b2c3d.test.ts',
        'tests/dqM3_check.py',
    ],
)
def test_a_history_named_file_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path], path: str
) -> None:
    scratch_repo.write(path, 'x = 1\n')
    assert run(scratch_repo, baseline_file('')) == 1


@pytest.mark.parametrize(
    'path',
    [
        'tests/test_router_wiring.py',
        'tests/test_release_notes.py',
        'scripts/bench/probes.py',
        'frontend/src/lib/api.ts',
        'docs/design/r7_plan.md',
        'src/services/rewrite.py',
    ],
)
def test_ordinary_file_names_pass(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path], path: str
) -> None:
    scratch_repo.write(path, 'x = 1\n')
    assert run(scratch_repo, baseline_file('')) == 0


def test_a_baselined_file_may_not_gain_a_hit(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/a.py', '# W8.2: a\n')
    base = baseline_file('src/a.py\t1\tstripped in PR 2\n')
    assert run(scratch_repo, base) == 0
    scratch_repo.write('src/a.py', '# W8.2: a\n# W8.3: b\n')
    assert run(scratch_repo, base) == 1


def test_a_stale_entry_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/a.py', '# clean\n')
    assert run(scratch_repo, baseline_file('src/a.py\t1\tstripped in PR 2\n')) == 1


def test_a_syntax_error_fails_closed(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/a.py', 'def broken(:\n')
    assert run(scratch_repo, baseline_file('')) == 1


def test_the_real_repo_passes() -> None:
    assert guard.main([]) == 0
