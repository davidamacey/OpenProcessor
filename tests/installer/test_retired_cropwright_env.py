"""The Cropwright variables moved under OP_CROPWRIGHT_*; a .env that still sets an old
name is refused by the installer and the deploy CLI, naming the new variable."""

from __future__ import annotations

import re
import subprocess
from typing import TYPE_CHECKING

import pytest
from _setup_dry_run_helpers import configure, dry
from installer_harness import REPO_ROOT


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed

RETIRED = {
    'CROPWRIGHT_IMAGE': 'OP_CROPWRIGHT_IMAGE',
    'CROPWRIGHT_BIND_ADDRESS': 'OP_CROPWRIGHT_BIND_ADDRESS',
    'CROPWRIGHT_PORT': 'OP_CROPWRIGHT_PORT',
    'CROPWRIGHT_INGEST_MAX_REQUEST_MB': 'OP_CROPWRIGHT_INGEST_MAX_REQUEST_MB',
    'CROPWRIGHT_DATASET_UPLOAD_MAX_MB': 'OP_CROPWRIGHT_DATASET_UPLOAD_MAX_MB',
}


def _install(shimmed: Shimmed) -> Path:
    assert configure(shimmed).returncode == 0
    return (shimmed.root / 'inst').resolve()


def _cli(shimmed: Shimmed, inst: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(inst / 'openprocessor'), *args],
        check=False,
        cwd=str(shimmed.root),
        env=shimmed.env(),
        capture_output=True,
        text=True,
        timeout=60,
        start_new_session=True,
    )


@pytest.mark.parametrize(('old', 'new'), sorted(RETIRED.items()))
def test_installer_refuses_a_retired_name_in_env(shimmed: Shimmed, old: str, new: str) -> None:
    inst = _install(shimmed)
    with (inst / '.env').open('a') as fh:
        fh.write(f'{old}=1\n')
    result = dry(shimmed)
    assert result.returncode == 2, result.stderr
    assert f'{old} was renamed to {new}' in result.stderr


@pytest.mark.parametrize(('old', 'new'), sorted(RETIRED.items()))
def test_cli_refuses_a_retired_name_in_env(shimmed: Shimmed, old: str, new: str) -> None:
    inst = _install(shimmed)
    with (inst / '.env').open('a') as fh:
        fh.write(f'{old}=1\n')
    result = _cli(shimmed, inst, 'status')
    assert result.returncode == 2, result.stderr
    assert f'{old} was renamed to {new}' in result.stdout + result.stderr


def test_cli_accepts_a_current_env(shimmed: Shimmed) -> None:
    inst = _install(shimmed)
    with (inst / '.env').open('a') as fh:
        fh.write('OP_CROPWRIGHT_PORT=5190\n')
    result = _cli(shimmed, inst, 'status')
    assert 'renamed' not in result.stdout + result.stderr


def _names(text: str) -> set[str]:
    return set(re.findall(r'\bCROPWRIGHT_[A-Z_]+', text))


def test_retired_lists_agree_everywhere() -> None:
    lib = (REPO_ROOT / 'scripts' / 'lib' / 'retired_env.sh').read_text()
    compose = (REPO_ROOT / 'docker-compose.yml').read_text()
    entrypoint = (REPO_ROOT / 'frontend' / 'docker-entrypoint.sh').read_text()
    block = re.search(r'RETIRED_ENV_NAMES=\((.*?)\)', lib, re.S)
    assert block, 'scripts/lib/retired_env.sh has no RETIRED_ENV_NAMES'
    assert _names(block.group(1)) == set(RETIRED)
    passthrough = set(re.findall(r'^\s+- (CROPWRIGHT_[A-Z_]+)=\$\{\1:-\}$', compose, re.M))
    assert passthrough == set(RETIRED)
    loop = re.search(r'for retired in ([A-Z_ ]+); do', entrypoint)
    assert loop, 'the cropwright entrypoint has no retired-name loop'
    assert {f'CROPWRIGHT_{n}' for n in loop.group(1).split()} == set(RETIRED)


@pytest.mark.parametrize('old', sorted(RETIRED))
def test_cropwright_entrypoint_refuses_a_retired_name(old: str) -> None:
    result = subprocess.run(
        ['sh', str(REPO_ROOT / 'frontend' / 'docker-entrypoint.sh')],
        check=False,
        env={'PATH': '/usr/bin:/bin', old: '1'},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert f'{old} was renamed to OP_{old}' in result.stderr
