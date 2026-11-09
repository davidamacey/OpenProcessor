"""The mypy-skip list for tests only shrinks: the pyproject block and the baseline must agree."""

from __future__ import annotations

from typing import TYPE_CHECKING

import check_mypy_tests


if TYPE_CHECKING:
    from pathlib import Path


BLOCK = '[tool.mypy]\n\n[[tool.mypy.overrides]]\nmodule = [\n{mods}]\nignore_errors = true\n'


def _repo(tmp_path: Path, listed: list[str], baseline: list[str]) -> tuple[Path, Path]:
    (tmp_path / 'tests').mkdir()
    for m in {*listed, *baseline}:
        (tmp_path / (m.replace('.', '/') + '.py')).write_text('', encoding='utf-8')
    pyproject = tmp_path / 'pyproject.toml'
    pyproject.write_text(
        BLOCK.replace('{mods}', ''.join(f'    "{m}",\n' for m in listed)), encoding='utf-8'
    )
    base = tmp_path / 'base.txt'
    base.write_text(''.join(f'{m}\t1\treason\n' for m in baseline), encoding='utf-8')
    return pyproject, base


def test_matching_lists_pass(tmp_path: Path) -> None:
    pyproject, base = _repo(tmp_path, ['tests.test_a'], ['tests.test_a'])
    assert check_mypy_tests.check(tmp_path, pyproject, base) == []


def test_new_ignored_module_is_blocked(tmp_path: Path) -> None:
    pyproject, base = _repo(tmp_path, ['tests.test_a', 'tests.test_b'], ['tests.test_a'])
    errors = check_mypy_tests.check(tmp_path, pyproject, base)
    assert len(errors) == 1
    assert 'tests.test_b' in errors[0]
    assert 'new debt' in errors[0]


def test_baseline_line_left_after_a_fix_is_flagged(tmp_path: Path) -> None:
    pyproject, base = _repo(tmp_path, [], ['tests.test_a'])
    assert 'delete the line' in check_mypy_tests.check(tmp_path, pyproject, base)[0]


def test_deleted_module_is_flagged_and_tightened(tmp_path: Path) -> None:
    pyproject, base = _repo(tmp_path, ['tests.test_a'], ['tests.test_a'])
    (tmp_path / 'tests' / 'test_a.py').unlink()
    assert any('no longer exists' in e for e in check_mypy_tests.check(tmp_path, pyproject, base))
    assert check_mypy_tests.tighten(tmp_path, pyproject, base) == 1
    assert check_mypy_tests.check(tmp_path, pyproject, base) == []


def test_non_test_module_cannot_be_ignored(tmp_path: Path) -> None:
    (tmp_path / 'src').mkdir()
    pyproject, base = _repo(tmp_path, ['src.x'], ['src.x'])
    assert any('only tests.*' in e for e in check_mypy_tests.check(tmp_path, pyproject, base))


def _fake_runner(output: str) -> check_mypy_tests.MypyRunner:
    def run(repo_root: Path, config: Path, files: list[Path]) -> str:
        text = config.read_text(encoding='utf-8')
        assert 'ignore_errors = false' in text
        assert 'ignore_errors = true' not in text
        return output

    return run


def test_clean_module_still_listed_is_flagged(tmp_path: Path) -> None:
    pyproject, _ = _repo(
        tmp_path, ['tests.test_a', 'tests.test_b'], ['tests.test_a', 'tests.test_b']
    )
    output = f'{tmp_path}/tests/test_a.py:3: error: boom  [misc]\nFound 1 error in 1 file\n'
    errors = check_mypy_tests.clean_but_listed(tmp_path, pyproject, _fake_runner(output))
    assert len(errors) == 1
    assert 'tests.test_b' in errors[0]
    assert 'no mypy error' in errors[0]


def test_listed_modules_that_still_fail_pass(tmp_path: Path) -> None:
    pyproject, _ = _repo(tmp_path, ['tests.test_a'], ['tests.test_a'])
    output = f'{tmp_path}/tests/test_a.py:3:5: error: boom  [misc]\n'
    assert check_mypy_tests.clean_but_listed(tmp_path, pyproject, _fake_runner(output)) == []


def test_empty_list_does_not_run_mypy(tmp_path: Path) -> None:
    pyproject, _ = _repo(tmp_path, [], [])

    def boom(repo_root: Path, config: Path, files: list[Path]) -> str:
        raise AssertionError('mypy must not run for an empty list')

    assert check_mypy_tests.clean_but_listed(tmp_path, pyproject, boom) == []
