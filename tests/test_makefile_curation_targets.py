"""``make curation-up`` takes the same optional service names as ``./openprocessor curation up`` (#285),
and every compose call from the Makefile loads the user's docker-compose.local.yml (#283)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _dry_run(*args: str, cwd: Path = REPO_ROOT) -> str:
    done = subprocess.run(
        ['make', '-n', '-s', *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_curation_up_without_services_starts_the_whole_profile() -> None:
    out = _dry_run('curation-up')
    assert '--profile curation up -d' in [
        ln.strip()[-len('--profile curation up -d') :] for ln in out.splitlines()
    ]


def test_curation_up_with_services_starts_only_those() -> None:
    out = _dry_run('curation-up', 'SERVICES=curation-detection-worker curation-vlm-worker')
    assert '--profile curation up -d curation-detection-worker curation-vlm-worker\n' in out


def test_curation_up_refuses_a_service_that_is_not_a_curation_one() -> None:
    done = subprocess.run(
        ['make', '-n', '-s', 'curation-up', 'SERVICES=api'],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode != 0


@pytest.mark.parametrize('present', [True, False])
def test_makefile_loads_a_local_override_only_when_it_exists(tmp_path: Path, present: bool) -> None:
    for name in ('Makefile', 'docker-compose.yml'):
        (tmp_path / name).write_text((REPO_ROOT / name).read_text())
    if present:
        (tmp_path / 'docker-compose.local.yml').write_text('services: {}\n')
    out = _dry_run('curation-up', cwd=tmp_path)
    assert ('-f docker-compose.local.yml' in out) is present
