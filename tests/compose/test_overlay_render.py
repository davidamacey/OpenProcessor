"""Every compose file combination the Makefile and the CLI use must render.

Render only (`docker compose config`): no container is created. Skipped where
Docker or the Compose plugin is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from compose._helpers import REPO_ROOT


BASE = ['docker-compose.yml']
DEV = [*BASE, 'docker-compose.dev.yml']
ALL_PROFILES = ['curation', 'segmenter', 'vlm', 'training', 'monitoring']

# (files, profiles): the Makefile's COMPOSE, cluster-gpu, cluster-cpu and
# DEV_COMPOSE variables and the CLI's DC_FILES, each with the profiles it enables.
COMBINATIONS = {
    'base': (BASE, ALL_PROFILES),
    'dev': (DEV, ALL_PROFILES),
    'cluster-gpu': ([*DEV, 'docker-compose.gpu-clustering.yml'], ['curation']),
    'cluster-gpu-all-profiles': ([*DEV, 'docker-compose.gpu-clustering.yml'], ALL_PROFILES),
    'dev-compose-arbiter': (
        [*DEV, 'docker-compose.gpu-arbiter.yml'],
        ['curation', 'segmenter', 'vlm', 'training'],
    ),
}


@pytest.mark.parametrize('name', sorted(COMBINATIONS))
def test_the_overlay_combination_renders(name: str) -> None:
    docker = shutil.which('docker')
    if not docker:
        pytest.skip('docker CLI not installed')
    files, profiles = COMBINATIONS[name]
    cmd = [docker, 'compose', '-p', 'overlay-render-test']
    for f in files:
        cmd += ['-f', str(REPO_ROOT / f)]
    for p in profiles:
        cmd += ['--profile', p]
    result = subprocess.run(
        [*cmd, '--project-directory', str(REPO_ROOT), 'config', '--quiet'],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if 'docker: unknown command' in result.stderr or 'is not a docker command' in result.stderr:
        pytest.skip('docker compose plugin unavailable')
    assert result.returncode == 0, result.stderr
