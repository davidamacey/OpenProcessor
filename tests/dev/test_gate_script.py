"""The local gate script against a temp tree with fake step commands.

The fakes append "<start> <end> <cwd-basename> <command line>" to one log, so the
tests assert ordering and overlap without running any real gate step.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'dev' / 'gate.sh'

FAKE = """#!/usr/bin/env bash
name=$(basename "$0")
start=$(date +%s.%N)
sleep "${FAKE_SLEEP:-0.1}"
rc=0
case "$name $*" in *"${FAKE_FAIL:-@@never@@}"*) rc=7 ;; esac
echo "$start $(date +%s.%N) $(basename "$PWD") $name $*" >> "$FAKE_LOG"
exit $rc
"""


def _exe(path: Path, body: str = FAKE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def tree(tmp_path: Path):
    root = tmp_path / 'repo'
    for d in ('frontend', 'docs-site', 'scripts/dev'):
        (root / d).mkdir(parents=True)
    (root / 'requirements-test.txt').write_text('')
    (root / 'pyproject.toml').write_text(
        '[tool.pytest.ini_options]\naddopts = [\n    "-v",\n    "-m",\n'
        '    "not live and not db",\n    "--tb=short",\n]\n'
    )
    for n in ('pip', 'pre-commit', 'python'):
        _exe(root / '.venv' / 'bin' / n)
    bindir = tmp_path / 'bin'
    for n in ('npm', 'npx', 'make'):
        _exe(bindir / n)
    load = tmp_path / 'loadavg'
    load.write_text('0.10 0.10 0.10 1/1 1\n')
    log = tmp_path / 'fake.log'
    env = {
        **os.environ,
        'PATH': f'{bindir}:{os.environ["PATH"]}',
        'GATE_ROOT': str(root),
        'GATE_LOADAVG': str(load),
        'GATE_HOLD_SECS': '0.1',
        'GATE_NICE': '0',
        'FAKE_LOG': str(log),
    }
    for k in ('GATE_OUT', 'GATE_MAX_LOAD', 'GATE_XDIST'):
        env.pop(k, None)
    return {'root': root, 'env': env, 'log': log, 'out': tmp_path / 'out', 'load': load}


def _run(tree, *args, extra=None, timeout=60):
    env = {**tree['env'], **(extra or {})}
    return subprocess.run(
        ['bash', str(SCRIPT), *args, str(tree['out'])],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _calls(tree):
    rows = []
    for line in tree['log'].read_text().splitlines():
        start, end, cwd, cmd = line.split(' ', 3)
        rows.append((float(start), float(end), cwd, cmd))
    return sorted(rows)


def test_script_exists():
    assert SCRIPT.is_file()


def test_installs_run_before_every_other_step(tree):
    res = _run(tree, 'all')
    assert res.returncode == 0, res.stdout + res.stderr
    calls = _calls(tree)
    installs = [c for c in calls if c[3].startswith(('pip ', 'npm ci'))]
    others = [c for c in calls if c not in installs]
    assert {c[3].split()[0] for c in installs} == {'pip', 'npm'}
    assert len(installs) == 4  # pip install, pip check, npm ci x2
    assert max(c[1] for c in installs) <= min(c[0] for c in others)


def test_tracks_run_in_parallel(tree):
    res = _run(tree, 'all')
    assert res.returncode == 0, res.stdout + res.stderr
    calls = _calls(tree)
    py = [c for c in calls if c[3].startswith(('pre-commit', 'python', 'make'))]
    fe = [c for c in calls if c[3].startswith(('npm run', 'npx', 'npm test'))]
    assert py
    assert fe
    assert min(c[0] for c in fe) < max(c[1] for c in py)
    assert min(c[0] for c in py) < max(c[1] for c in fe)


def test_failed_step_recorded_and_track_continues(tree):
    res = _run(tree, 'all', extra={'FAKE_FAIL': 'run lint'})
    assert res.returncode != 0
    summary = (tree['out'] / 'summary_fe.txt').read_text()
    assert '== fe_lint rc=7 ' in summary
    assert '== fe_knip rc=0 ' in summary
    assert '== fe_lint rc=7' in res.stdout


def test_summary_lists_every_step_with_logs(tree):
    res = _run(tree, 'all')
    assert res.returncode == 0, res.stdout + res.stderr
    py = (tree['out'] / 'summary_py.txt').read_text()
    fe = (tree['out'] / 'summary_fe.txt').read_text()
    for step in ('precommit', 'pytest', 'contracts', 'docs_check'):
        assert f'== {step} rc=0 ' in py
        assert (tree['out'] / f'{step}.log').is_file()
    for step in ('fe_check', 'fe_lint', 'fe_knip', 'fe_test', 'fe_build', 'fe_e2e', 'site_build'):
        assert f'== {step} rc=0 ' in fe
    assert 'load=' in py
    assert '== pytest' in res.stdout


def test_pytest_marker_expression_read_from_pyproject(tree):
    _run(tree, 'py', extra={'GATE_XDIST': '3'})
    pytest_call = next(c[3] for c in _calls(tree) if c[3].startswith('python -m pytest'))
    assert '-m not live and not db' in pytest_call
    assert '-n 3' in pytest_call


def test_single_track_argument(tree):
    _run(tree, 'py')
    assert not any(c[3].startswith(('npx', 'npm')) for c in _calls(tree))


def test_load_hold_is_recorded(tree):
    tree['load'].write_text('99.0 99.0 99.0 1/1 1\n')
    env = {**tree['env'], 'GATE_NICE': '1', 'GATE_MAX_LOAD': '5'}
    if shutil.which('ionice') is None or shutil.which('nice') is None:
        pytest.skip('nice/ionice missing')

    def relieve():
        time.sleep(1.0)
        tree['load'].write_text('0.1 0.1 0.1 1/1 1\n')

    t = threading.Thread(target=relieve)
    t.start()
    res = subprocess.run(
        ['bash', str(SCRIPT), 'py', str(tree['out'])],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    t.join()
    assert res.returncode == 0, res.stdout + res.stderr
    summary = ''.join(p.read_text() for p in tree['out'].glob('summary_*.txt'))
    assert 'hold' in summary
    assert '99.0' in summary
