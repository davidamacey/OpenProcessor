"""The ``db`` marker policy: an unmarked database test fails collection, the default selection
holds no ``db`` test. Runs pytest in a subprocess on a planted tree."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests' / 'fixtures'


def _pytest(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # fixed argv, our own interpreter
        [sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', '--rootdir', str(cwd), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={'PYTHONPATH': str(FIXTURES), 'PATH': '/usr/bin'},
    )


def _plant(root: Path, body: str, *, where: str) -> Path:
    tests = root / 'tests' / where if where else root / 'tests'
    tests.mkdir(parents=True)
    (root / 'tests' / 'conftest.py').write_text(
        "import pytest\npytest_plugins = ['db_marker_policy']\n\n"
        '@pytest.fixture\ndef pg():\n    return None\n'
    )
    (root / 'pytest.ini').write_text('[pytest]\nmarkers = db: needs a database\n')
    (tests / 'test_planted.py').write_text(body)
    return root


def test_unmarked_test_under_tests_db_fails_collection(tmp_path: Path) -> None:
    root = _plant(tmp_path, 'def test_x():\n    pass\n', where='db')
    result = _pytest(root, '--collect-only', '-q')
    assert result.returncode != 0
    assert 'lack @pytest.mark.db' in result.stdout + result.stderr
    assert 'test_planted.py::test_x' in result.stdout + result.stderr


def test_unmarked_test_using_pg_fails_collection(tmp_path: Path) -> None:
    root = _plant(tmp_path, 'def test_x(pg):\n    pass\n', where='')
    result = _pytest(root, '--collect-only', '-q')
    assert result.returncode != 0
    assert 'lack @pytest.mark.db' in result.stdout + result.stderr


def test_marked_tests_collect(tmp_path: Path) -> None:
    root = _plant(
        tmp_path,
        'import pytest\npytestmark = pytest.mark.db\n\ndef test_x(pg):\n    pass\n',
        where='db',
    )
    result = _pytest(root, '--collect-only', '-q')
    assert result.returncode == 0, result.stdout + result.stderr


def test_default_selection_contains_no_db_test() -> None:
    """The real tree: the ``addopts`` expression deselects every test under tests/db."""
    result = subprocess.run(  # fixed argv, our own interpreter
        [sys.executable, '-m', 'pytest', 'tests/db', '--collect-only', '-q', '--no-cov'],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    selected = [ln for ln in result.stdout.splitlines() if '::' in ln]
    assert selected == []
    assert 'deselected' in result.stdout
