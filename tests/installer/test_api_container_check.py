"""A running api container is found through compose, not by a bare name match.

The container is named ``${COMPOSE_PROJECT_NAME}-api``, so ``docker ps | grep
'^api$'`` never matched and download.sh / export_paddleocr.sh always took the
temporary-container branch (#187). Everything runs against the docker shim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from installer_harness import run_bash


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed

PROJECT = 'opcheck'
DOWNLOAD_SH = 'scripts/lib/download.sh'


def _running_api(shimmed: Shimmed) -> None:
    shimmed.containers([(PROJECT, '/srv/op', f'{PROJECT}-api', '')])
    shimmed.flag('api_container')
    shimmed.flag('allow_mutations')


def _compose_calls(shimmed: Shimmed) -> list[str]:
    return shimmed.log_lines('docker compose')


def test_download_paddleocr_execs_in_the_running_project_prefixed_api(
    shimmed: Shimmed, repo_root: Path, tmp_path: Path
) -> None:
    _running_api(shimmed)
    script = f"""
set -u
PROJECT_DIR={tmp_path}
source {repo_root}/{DOWNLOAD_SH}
dc() {{ docker compose -p {PROJECT} "$@"; }}
download_paddleocr_models
"""
    result = run_bash(script, env=shimmed.env())
    assert result.returncode == 0, result.stdout + result.stderr
    calls = _compose_calls(shimmed)
    assert any(f'-p {PROJECT} exec -T api' in c for c in calls), calls
    assert not any(' run ' in c for c in calls), calls


def test_download_paddleocr_uses_a_temporary_container_when_api_is_down(
    shimmed: Shimmed, repo_root: Path, tmp_path: Path
) -> None:
    shimmed.flag('allow_mutations')
    script = f"""
set -u
PROJECT_DIR={tmp_path}
source {repo_root}/{DOWNLOAD_SH}
dc() {{ docker compose -p {PROJECT} "$@"; }}
download_paddleocr_models
"""
    result = run_bash(script, env=shimmed.env())
    assert result.returncode == 0, result.stdout + result.stderr
    calls = _compose_calls(shimmed)
    assert any(f'-p {PROJECT} run --rm --no-deps -T api' in c for c in calls), calls
    assert not any(' exec ' in c for c in calls), calls


@pytest.mark.parametrize('service', ['api', 'triton-server'])
def test_export_paddleocr_check_container_resolves_through_the_project(
    shimmed: Shimmed, repo_root: Path, tmp_path: Path, service: str
) -> None:
    shimmed.containers([(PROJECT, '/srv/op', f'{PROJECT}-{service}', '')])
    shimmed.flag(f'{service}_container')
    script = f"""
set -u
PROJECT_DIR={tmp_path}
source {repo_root}/scripts/lib/ports.sh
source <(sed -n '/^dc() {{/,/^}}/p;/^check_container() {{/,/^}}/p' {repo_root}/scripts/export_paddleocr.sh)
COMPOSE_PROJECT_NAME={PROJECT}
check_container {service} && echo FOUND || echo MISSING
check_container nothing-here && echo FOUND || echo MISSING
"""
    result = run_bash(script, env=shimmed.env())
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.split() == ['FOUND', 'MISSING']
    assert any(f'-p {PROJECT} ' in c and ' ps -q ' in c for c in _compose_calls(shimmed))
