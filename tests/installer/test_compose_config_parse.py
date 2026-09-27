"""The installer's `compose config --format json` parsers, checked against
the real Compose output for this repo's docker-compose.yml. Only `config` is
run (read-only, project opinst-test); skipped where Docker is unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest
from installer_harness import REPO_ROOT, SCRIPT


if TYPE_CHECKING:
    from pathlib import Path


def real_config(tmp_path: Path) -> str:
    docker = shutil.which('docker') or ''
    if not docker:
        pytest.skip('docker CLI not installed')
    shutil.copy(REPO_ROOT / 'docker-compose.yml', tmp_path / 'docker-compose.yml')
    (tmp_path / '.env').write_text(
        'COMPOSE_PROJECT_NAME=opinst-test\nCOMPOSE_PROFILES=curation,segmenter,vlm,training\n'
    )
    result = subprocess.run(
        [
            docker,
            'compose',
            '-p',
            'opinst-test',
            '--env-file',
            str(tmp_path / '.env'),
            '--project-directory',
            str(tmp_path),
            '-f',
            str(tmp_path / 'docker-compose.yml'),
            'config',
            '--format',
            'json',
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.skip(f'docker compose config unavailable: {result.stderr[:200]}')
    return result.stdout


def parse(fn: str, text: str) -> list[str]:
    result = subprocess.run(
        ['bash', '-c', f'OP_SOURCE_ONLY=1 source "{SCRIPT}"; {fn} "$1"', '_', text],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.split()


def test_container_names_match_the_real_config(tmp_path: Path) -> None:
    text = real_config(tmp_path)
    cfg = json.loads(text)
    expected = sorted(s['container_name'] for s in cfg['services'].values())
    assert sorted(parse('compose_config_container_names', text)) == expected
    assert all(n.startswith('opinst-test-') for n in expected)


def test_network_names_match_the_real_config(tmp_path: Path) -> None:
    text = real_config(tmp_path)
    cfg = json.loads(text)
    assert parse('compose_config_network_names', text) == [
        n['name'] for n in cfg['networks'].values()
    ]
    assert parse('compose_config_network_names', text) == ['opinst-test_triton_net']
