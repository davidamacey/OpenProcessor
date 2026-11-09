"""Shell helpers find a container through its Compose labels, never a container name (D-54).

Docs: https://docs.docker.com/reference/compose-file/services/#labels
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from installer_harness import REPO_ROOT


if TYPE_CHECKING:
    from pathlib import Path


CLI_GPU = REPO_ROOT / 'scripts' / 'lib' / 'cli_gpu.sh'

# A fake docker that knows one container, labelled for project `acme`, service `triton-server`,
# named like Compose names it (no container_name): `acme-triton-server-1`.
FAKE_DOCKER = """#!/bin/bash
args="$*"
[[ "$1" == ps ]] || exit 1
[[ "$args" == *"label=com.docker.compose.project=acme"* ]] || exit 0
[[ "$args" == *"label=com.docker.compose.service=triton-server"* ]] || exit 0
echo abc123
"""


def _running(tmp_path: Path, project: str, service: str) -> bool:
    docker = tmp_path / 'docker'
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    done = subprocess.run(
        [
            'bash',
            '-c',
            f'source "{CLI_GPU}"; _compose_service_running "$1" "$2"',
            '_',
            project,
            service,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={'PATH': f'{tmp_path}:/usr/bin:/bin'},
    )
    return done.returncode == 0


def test_a_running_service_is_found_by_its_labels(tmp_path: Path) -> None:
    assert _running(tmp_path, 'acme', 'triton-server')


def test_another_project_or_service_is_not_found(tmp_path: Path) -> None:
    assert not _running(tmp_path, 'other', 'triton-server')
    assert not _running(tmp_path, 'acme', 'api')


def test_no_shell_helper_builds_a_container_name() -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in [*(REPO_ROOT / 'scripts' / 'lib').glob('*.sh'), REPO_ROOT / 'openprocessor']
        if '-triton"' in path.read_text() or '}-api"' in path.read_text()
    ]
    assert not offenders, offenders
