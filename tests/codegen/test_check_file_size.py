"""check_file_size.py: warn at 700, fail above 1000 unless a cohesive baseline entry, over all first-party code."""

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


APP_FILES = [
    'src/_probe.py',
    'export/_probe.py',
    'models/m/1/_probe.py',
    'docker/x/_probe.py',
    'frontend/src/lib/_probe.ts',
    'frontend/src/lib/_probe.svelte',
]
TEST_FILES = ['tests/_probe.py', 'frontend/e2e/_probe.py', 'frontend/src/lib/_probe.test.ts']
OUT_OF_SCOPE = [
    'scripts/codegen/_probe.py',
    'scripts/lib/_probe.sh',
    'src/_probe.sh',
    'frontend/scripts/_probe.mjs',
    'openprocessor',
    'setup-openprocessor.sh',
]


@pytest.mark.parametrize('planted', APP_FILES)
def test_app_code_over_1000_lines_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path], planted: str
) -> None:
    scratch_repo.write_lines(planted, 1001)
    assert run(scratch_repo, baseline_file('')) == 1


@pytest.mark.parametrize('planted', APP_FILES)
def test_app_code_at_1000_lines_passes_with_a_warning(
    scratch_repo: ScratchRepo,
    baseline_file: Callable[[str], Path],
    capsys: pytest.CaptureFixture[str],
    planted: str,
) -> None:
    scratch_repo.write_lines(planted, 1000)
    assert run(scratch_repo, baseline_file('')) == 0
    assert f'{planted}: 1000' in capsys.readouterr().err


@pytest.mark.parametrize('planted', TEST_FILES)
def test_test_code_warns_from_700_and_fails_over_1500(
    scratch_repo: ScratchRepo,
    baseline_file: Callable[[str], Path],
    capsys: pytest.CaptureFixture[str],
    planted: str,
) -> None:
    scratch_repo.write_lines(planted, 1500)
    assert run(scratch_repo, baseline_file('')) == 0
    assert f'{planted}: 1500' in capsys.readouterr().err
    scratch_repo.write_lines(planted, 1501)
    assert run(scratch_repo, baseline_file('')) == 1


@pytest.mark.parametrize('planted', OUT_OF_SCOPE)
def test_scripts_shell_and_the_root_cli_are_out_of_scope(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path], planted: str
) -> None:
    scratch_repo.write_lines(planted, 5000)
    assert run(scratch_repo, baseline_file('')) == 0


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
    scratch_repo.write_lines('export/near.py', 750)
    assert run(scratch_repo, baseline_file('')) == 0
    assert 'export/near.py: 750' in capsys.readouterr().err


def test_a_claude_md_over_400_lines_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write_lines('frontend/CLAUDE.md', 401)
    assert run(scratch_repo, baseline_file('')) == 1


def test_699_lines_do_not_warn(
    scratch_repo: ScratchRepo,
    baseline_file: Callable[[str], Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    scratch_repo.write_lines('export/fine.py', 699)
    assert run(scratch_repo, baseline_file('')) == 0
    assert 'fine.py' not in capsys.readouterr().err


def test_a_baseline_reason_must_say_cohesive(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write_lines('export/big.py', 1200)
    assert run(scratch_repo, baseline_file('export/big.py\t1200\tsplit later\n')) == 1


def test_a_baselined_file_may_stay_but_not_grow(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write_lines('export/big.py', 1200)
    base = baseline_file('export/big.py\t1200\tcohesive: one parser state machine\n')
    assert run(scratch_repo, base) == 0
    scratch_repo.write_lines('export/big.py', 1201)
    assert run(scratch_repo, base) == 1


def test_a_stale_entry_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write_lines('export/was_big.py', 600)
    assert (
        run(
            scratch_repo,
            baseline_file('export/was_big.py\t1200\tcohesive: one parser state machine\n'),
        )
        == 1
    )
    assert (
        run(
            scratch_repo,
            baseline_file('export/gone.py\t1200\tcohesive: one parser state machine\n'),
        )
        == 1
    )


def test_a_loosened_entry_fails_and_tighten_repairs_it(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write_lines('export/big.py', 1100)
    scratch_repo.write_lines('export/done.py', 100)
    base = baseline_file(
        'export/big.py\t1200\tcohesive: one parser state machine\nexport/done.py\t900\tcohesive: one parser state machine\n'
    )
    assert run(scratch_repo, base) == 1
    assert run(scratch_repo, base, '--tighten') == 0
    assert base.read_text() == 'export/big.py\t1100\tcohesive: one parser state machine\n'
    assert run(scratch_repo, base) == 0


def test_a_baseline_line_without_a_reason_is_rejected(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write_lines('export/big.py', 1200)
    with pytest.raises(ValueError, match='reason'):
        run(scratch_repo, baseline_file('export/big.py\t1200\t\n'))


def test_the_real_repo_passes() -> None:
    assert guard.main([]) == 0
