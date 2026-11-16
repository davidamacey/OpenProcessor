"""The pending list (``tests/pending_rust.txt``): a listed test is an expected failure, strictly.

pytest's ``xfail(strict=True)`` turns a listed test that passes into a failure
(https://docs.pytest.org/en/stable/how-to/skipping.html#strict-parameter), so an entry cannot outlive
the slice that fixes it. The hook runs here in a child pytest over a scratch project, because the
behaviour under test is the outcome of a whole run.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pending_rust
import pytest


TESTS_DIR = Path(__file__).resolve().parent

CONFTEST = """
import sys
sys.path.insert(0, {tests_dir!r})
import pending_rust


def pytest_collection_modifyitems(items):
    pending_rust.mark_pending(items, pending_rust.read_pending(__import__('pathlib').Path({pending!r})),
                              __import__('pathlib').Path({root!r}))
"""

SUITE = """
import pytest


def test_red():
    assert False


def test_green():
    assert True


@pytest.mark.parametrize('n', [1, 2])
def test_param_red(n):
    assert False


def test_unlisted_red():
    assert False
"""


def run_child(tmp_path: Path, pending_text: str, *, rootdir: Path | None = None) -> tuple[int, str]:
    project = tmp_path / 'project'
    (project / 'sub').mkdir(parents=True)
    pending = tmp_path / 'pending.txt'
    pending.write_text(pending_text, encoding='utf-8')
    suite_dir = project / 'sub'
    (suite_dir / 'conftest.py').write_text(
        CONFTEST.format(
            tests_dir=str(TESTS_DIR), pending=str(pending), root=str(rootdir or project)
        ),
        encoding='utf-8',
    )
    (suite_dir / 'test_suite.py').write_text(textwrap.dedent(SUITE), encoding='utf-8')
    (suite_dir / 'pytest.ini').write_text(
        '[pytest]\naddopts = -p no:cacheprovider\n', encoding='utf-8'
    )
    done = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            'test_suite.py',
            '-rxX',
            '-q',
            '--no-header',
            '-p',
            'no:randomly',
        ],
        cwd=suite_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    return done.returncode, done.stdout + done.stderr


def test_a_listed_failing_test_is_an_expected_failure(tmp_path: Path) -> None:
    code, out = run_child(
        tmp_path,
        'sub/test_suite.py::test_red R1c\n'
        'sub/test_suite.py::test_param_red R5\n'
        'sub/test_suite.py::test_green R1c\n',
    )
    # test_green passes although listed: strict, so the run fails on it alone.
    assert code == 1, out
    assert 'XPASS(strict)' in out, out
    assert 'pending R1c' in out, out
    assert '2 failed, 3 xfailed' in out, out
    # the unlisted failing test still fails the run on its own terms
    assert 'test_unlisted_red' in out, out


def test_when_only_failing_tests_are_listed_the_run_passes_the_listed_and_fails_the_rest(
    tmp_path: Path,
) -> None:
    code, out = run_child(
        tmp_path,
        'sub/test_suite.py::test_red R1c\nsub/test_suite.py::test_param_red R5\n',
    )
    assert code == 1, out
    assert 'XPASS(strict)' not in out, out
    assert '3 xfailed' in out, out
    assert '1 failed' in out, out


def test_a_listed_test_that_passes_fails_the_run(tmp_path: Path) -> None:
    code, out = run_child(
        tmp_path,
        'sub/test_suite.py::test_red R1c\n'
        'sub/test_suite.py::test_param_red R1c\n'
        'sub/test_suite.py::test_unlisted_red R1c\n'
        'sub/test_suite.py::test_green R2b\n',
    )
    assert code == 1, out
    assert 'XPASS(strict)' in out, out
    assert '1 failed' in out, out
    assert '4 xfailed' in out, out


def test_an_empty_list_marks_nothing(tmp_path: Path) -> None:
    code, out = run_child(tmp_path, '# nothing is pending\n')
    assert code == 1, out
    assert 'xfailed' not in out, out
    assert '4 failed, 1 passed' in out, out


def test_ids_are_relative_to_the_repo_root_not_the_pytest_rootdir(tmp_path: Path) -> None:
    # frontend/e2e runs with its own rootdir; the list names every test from the repo root.
    code, out = run_child(
        tmp_path,
        'project/sub/test_suite.py::test_red R1c\n',
        rootdir=tmp_path,
    )
    assert code == 1, out
    assert '1 xfailed' in out, out


@pytest.mark.parametrize(
    'text',
    [
        'tests/x.py::test_a\n',
        'tests/x.py::test_a R1c extra words\n',
        'tests/x.py::test_a R1c\ntests/x.py::test_a R5\n',
    ],
)
def test_a_malformed_or_duplicate_line_is_refused(tmp_path: Path, text: str) -> None:
    path = tmp_path / 'p.txt'
    path.write_text(text, encoding='utf-8')
    with pytest.raises(ValueError, match='pending'):
        pending_rust.read_pending(path)


def test_comments_and_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / 'p.txt'
    path.write_text('# header\n\ntests/x.py::test_a[1 2] R1c\n', encoding='utf-8')
    assert pending_rust.read_pending(path) == {'tests/x.py::test_a[1 2]': 'R1c'}


def test_every_committed_entry_names_a_test_that_still_exists() -> None:
    repo = TESTS_DIR.parent
    for node_id in pending_rust.read_pending():
        file_part, _, name = node_id.partition('::')
        path = repo / file_part
        assert path.is_file(), f'{node_id}: {file_part} is not a file (stale entry)'
        function = name.split('::')[-1].split('[')[0]
        assert f'def {function}(' in path.read_text(encoding='utf-8'), f'{node_id}: no such test'


class _Item:
    """Just enough of a collected test for the hook: an id, a path and ``add_marker``."""

    def __init__(self, nodeid: str, path: Path) -> None:
        self.nodeid = nodeid
        self.path = path
        self.markers: list[pytest.Mark] = []

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        self.markers.append(marker.mark)


def test_the_tests_conftest_applies_the_committed_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import conftest

    listed = tmp_path / 'pending.txt'
    listed.write_text('tests/test_probe.py::test_red R1c\n', encoding='utf-8')
    monkeypatch.setattr(pending_rust, 'PENDING_FILE', listed)
    red = _Item('test_probe.py::test_red', TESTS_DIR / 'test_probe.py')
    other = _Item('test_probe.py::test_other', TESTS_DIR / 'test_probe.py')

    conftest.pytest_collection_modifyitems([red, other])

    assert [m.name for m in red.markers] == ['xfail']
    assert red.markers[0].kwargs == {'strict': True, 'reason': 'pending R1c'}
    assert other.markers == []


def test_the_browser_suite_conftest_applies_the_same_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    repo = TESTS_DIR.parent
    spec = importlib.util.spec_from_file_location('e2e_conftest', repo / 'frontend/e2e/conftest.py')
    assert spec is not None
    assert spec.loader is not None
    e2e_conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2e_conftest)

    listed = tmp_path / 'pending.txt'
    listed.write_text('frontend/e2e/live/test_probe.py::test_red R4\n', encoding='utf-8')
    monkeypatch.setattr(pending_rust, 'PENDING_FILE', listed)
    # the browser suite runs with rootdir frontend/e2e, so its nodeids start below it
    red = _Item('live/test_probe.py::test_red', repo / 'frontend/e2e/live/test_probe.py')

    e2e_conftest.pytest_collection_modifyitems([red])

    assert red.markers[0].kwargs == {'strict': True, 'reason': 'pending R4'}
