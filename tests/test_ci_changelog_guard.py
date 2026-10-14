"""The CI changelog step, run for real: the ``run:`` block of ci.yml against throwaway git repos."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _step_script() -> str:
    workflow = yaml.safe_load((REPO_ROOT / '.github/workflows/ci.yml').read_text(encoding='utf-8'))
    steps = workflow['jobs']['changelog']['steps']
    return next(s['run'] for s in steps if 'run' in s)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-c', 'user.name=t', '-c', 'user.email=t@example.com', *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_step(repo: Path, message: str, *files: str) -> subprocess.CompletedProcess[str]:
    """Commit ``files`` on top of a base commit and run the step as a pull_request event."""
    (repo / 'README.md').write_text('base\n', encoding='utf-8')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-q', '-m', 'chore: base')
    base = _git(repo, 'rev-parse', 'HEAD')
    for name in files:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('changed\n', encoding='utf-8')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-q', '-m', message)
    head = _git(repo, 'rev-parse', 'HEAD')
    env = {
        'EVENT_NAME': 'pull_request',
        'BASE_SHA': base,
        'HEAD_SHA': head,
        'BEFORE_SHA': '',
        'AFTER_SHA': head,
        'PATH': '/usr/bin:/bin',
    }
    return subprocess.run(
        ['bash', '-e', '-c', _step_script()],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, 'init', '-q')
    return tmp_path


@pytest.mark.parametrize(
    'changed',
    ['src/a.py', 'scripts/b.py', 'docker/c/Dockerfile', 'export/d.py', 'models/e/config.pbtxt'],
)
def test_a_backend_change_without_a_changelog_entry_fails(repo: Path, changed: str) -> None:
    result = _run_step(repo, 'fix(x): change', changed)
    assert result.returncode == 1
    assert 'does not update CHANGELOG.md' in result.stdout


def test_a_backend_change_with_a_changelog_entry_passes(repo: Path) -> None:
    assert _run_step(repo, 'fix(x): change', 'src/a.py', 'CHANGELOG.md').returncode == 0


def test_skip_changelog_in_the_last_commit_message_passes(repo: Path) -> None:
    assert _run_step(repo, 'chore: rename [skip-changelog]', 'src/a.py').returncode == 0


@pytest.mark.parametrize('changed', ['docs/a.md', 'tests/a.py', 'frontend/src/a.ts', 'README.md'])
def test_changes_outside_the_backend_trees_pass(repo: Path, changed: str) -> None:
    assert _run_step(repo, 'docs: change', changed).returncode == 0
