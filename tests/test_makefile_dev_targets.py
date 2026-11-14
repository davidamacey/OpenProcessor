"""The dev-* targets run a real compose command: 3f36a3b2a renamed the variable they use to
DEV_DEV_COMPOSE and left every `$(DEV_COMPOSE)` empty, so `make dev-up` ran `up -d --build ...` as a
shell command."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = (REPO_ROOT / 'Makefile').read_text()
# Used in recipes but given on the command line or by make itself.
FROM_OUTSIDE = {
    'ACC_ARGS', 'BATCH', 'COMPOSE_PROJECT_NAME', 'CONFIG', 'CURDIR', 'ID', 'MAKE', 'MAKEFILE_LIST',
    'MODEL', 'NAME', 'OP_CROPWRIGHT_URL', 'OP_DATA_DIR', 'OP_FACES_ENABLED', 'REPO', 'SERVICES',
    'SOURCE',
}  # fmt: skip


def _dry_run(*args: str, env: dict[str, str] | None = None) -> str:
    done = subprocess.run(
        ['make', '-n', '-s', *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_every_makefile_variable_is_defined() -> None:
    defined = set(re.findall(r'^([A-Z_][A-Z0-9_]*)\s*[:?+]?=', MAKEFILE, re.M))
    used = set(re.findall(r'\$\(([A-Z_][A-Z0-9_]*)\)', MAKEFILE))
    assert used - defined - FROM_OUTSIDE == set()


@pytest.mark.parametrize(
    ('target', 'tail'),
    [
        ('dev-up', 'up -d --build --remove-orphans'),
        ('dev-monitoring-up', '--profile monitoring up -d --build --remove-orphans'),
        ('dev-down', 'down'),
        ('dev-ps', 'ps'),
    ],
)
def test_dev_targets_run_docker_compose_with_the_dev_overlay(target: str, tail: str) -> None:
    lines = [ln for ln in _dry_run(target).splitlines() if 'compose' in ln and ln.endswith(tail)]
    assert lines, target
    assert lines[-1].startswith('docker compose -f docker-compose.yml -f docker-compose.dev.yml')


def test_dev_targets_load_the_data_dir_overlay_when_op_data_dir_is_set() -> None:
    import os

    out = _dry_run('dev-down', env={**os.environ, 'OP_DATA_DIR': '/srv/x'})
    assert '-f docker-compose.data-dir.yml' in out
