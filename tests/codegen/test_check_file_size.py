"""check_file_size.py: the 700-line cap over all first-party code, with a shrink-only baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import check_file_size as guard
import pytest


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from conftest import ScratchRepo


def run(repo: ScratchRepo, baseline: Path, *extra: str) -> int:
    return guard.main(['--root', str(repo.root), '--baseline', str(baseline), *extra])


@pytest.mark.parametrize(
    'planted',
    [
        'export/_probe.py',
        'frontend/src/lib/_probe.ts',
        'frontend/e2e/_probe.py',
        'frontend/scripts/_probe.mjs',
        'tests/_probe.py',
        'scripts/lib/_probe.sh',
        'models/m/1/_probe.py',
        'docker/x/_probe.py',
        'src/_probe.py',
        'src/lib/_probe.svelte',
    ],
)
def test_701_lines_fail_in_every_first_party_tree(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path], planted: str
) -> None:
    scratch_repo.write_lines(planted, 701)
    assert run(scratch_repo, baseline_file('')) == 1


def test_700_lines_pass(scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]) -> None:
    scratch_repo.write_lines('export/ok.py', 700)
    assert run(scratch_repo, baseline_file('')) == 0


def test_root_shell_programs_are_in_scope(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write_lines('openprocessor', 701)
    assert run(scratch_repo, baseline_file('')) == 1


def test_vendored_and_unrelated_files_are_out_of_scope(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write_lines('src/static/docs/bundle.js', 5000)
    scratch_repo.write_lines('docs/gen.py', 5000)
    scratch_repo.write_lines('tests/data.json', 5000)
    assert run(scratch_repo, baseline_file('')) == 0


def test_the_warn_band_never_fails(
    scratch_repo: ScratchRepo,
    baseline_file: Callable[[str], Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    scratch_repo.write_lines('export/near.py', 650)
    assert run(scratch_repo, baseline_file('')) == 0
    assert 'export/near.py: 650' in capsys.readouterr().err


def test_a_claude_md_over_400_lines_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write_lines('frontend/CLAUDE.md', 401)
    assert run(scratch_repo, baseline_file('')) == 1


def test_a_baselined_file_may_stay_but_not_grow(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write_lines('export/big.py', 800)
    base = baseline_file('export/big.py\t800\tsplit in PR 7\n')
    assert run(scratch_repo, base) == 0
    scratch_repo.write_lines('export/big.py', 801)
    assert run(scratch_repo, base) == 1


def test_a_stale_entry_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write_lines('export/was_big.py', 600)
    assert run(scratch_repo, baseline_file('export/was_big.py\t800\tsplit in PR 7\n')) == 1
    assert run(scratch_repo, baseline_file('export/gone.py\t800\tsplit in PR 7\n')) == 1


def test_a_loosened_entry_fails_and_tighten_repairs_it(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write_lines('export/big.py', 750)
    scratch_repo.write_lines('export/done.py', 100)
    base = baseline_file('export/big.py\t800\tsplit in PR 7\nexport/done.py\t900\tsplit in PR 7\n')
    assert run(scratch_repo, base) == 1
    assert run(scratch_repo, base, '--tighten') == 0
    assert base.read_text() == 'export/big.py\t750\tsplit in PR 7\n'
    assert run(scratch_repo, base) == 0


def test_a_baseline_line_without_a_reason_is_rejected(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write_lines('export/big.py', 800)
    with pytest.raises(ValueError, match='reason'):
        run(scratch_repo, baseline_file('export/big.py\t800\t\n'))


def test_the_real_repo_passes() -> None:
    assert guard.main([]) == 0
