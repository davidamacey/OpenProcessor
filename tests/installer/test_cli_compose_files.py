"""The deploy CLI's compose file list and fresh-checkout project name, and the subset form of
``curation up`` (issues #283, #285, #291)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import test_cli_project
from test_cli_project import cli, set_env_project


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed

inst = test_cli_project.inst


def _make_checkout(inst: Path) -> None:
    (inst / 'src').mkdir()
    (inst / 'src' / 'main.py').write_text('')
    (inst / 'docker-compose.dev.yml').write_text('services: {}\n')


def test_a_local_override_file_is_loaded_after_the_stock_files(
    shimmed: Shimmed, inst: Path
) -> None:
    shimmed.flag('allow_mutations')
    (inst / 'docker-compose.local.yml').write_text('services: {}\n')
    done = cli(shimmed, inst, 'stop')
    assert done.returncode == 0, done.stderr
    down = [ln for ln in shimmed.mutating_docker_calls() if ln.endswith(' down')]
    assert down, shimmed.mutating_docker_calls()
    line = down[0]
    assert f'-f {inst}/docker-compose.local.yml' in line
    assert line.index('docker-compose.yml') < line.index('docker-compose.local.yml')


def test_without_a_local_file_the_file_list_is_unchanged(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    assert cli(shimmed, inst, 'stop').returncode == 0
    assert all('docker-compose.local.yml' not in ln for ln in shimmed.mutating_docker_calls())


def _stop_line(shimmed: Shimmed, inst: Path) -> str:
    done = cli(shimmed, inst, 'stop')
    assert done.returncode == 0, done.stderr
    down = [ln for ln in shimmed.mutating_docker_calls() if ln.endswith(' down')]
    assert down, shimmed.mutating_docker_calls()
    return down[0]


def test_the_ingest_load_profile_adds_its_overlay_before_the_local_file(
    shimmed: Shimmed, inst: Path
) -> None:
    shimmed.flag('allow_mutations')
    (inst / 'docker-compose.ingest-focus.yml').write_text('services: {}\n')
    (inst / 'docker-compose.local.yml').write_text('services: {}\n')
    with (inst / '.env').open('a') as fh:
        fh.write('\nOP_LOAD_PROFILE=ingest\n')
    line = _stop_line(shimmed, inst)
    assert f'-f {inst}/docker-compose.ingest-focus.yml' in line
    assert line.index('docker-compose.ingest-focus.yml') < line.index('docker-compose.local.yml')


@pytest.mark.parametrize('env_line', ['', 'OP_LOAD_PROFILE=full', '# OP_LOAD_PROFILE=ingest'])
def test_without_the_ingest_profile_the_overlay_is_not_loaded(
    shimmed: Shimmed, inst: Path, env_line: str
) -> None:
    shimmed.flag('allow_mutations')
    (inst / 'docker-compose.ingest-focus.yml').write_text('services: {}\n')
    with (inst / '.env').open('a') as fh:
        fh.write(f'\n{env_line}\n')
    assert 'ingest-focus' not in _stop_line(shimmed, inst)


def test_a_fresh_checkout_without_a_project_name_uses_the_documented_default(
    shimmed: Shimmed, inst: Path
) -> None:
    shimmed.flag('allow_mutations')
    _make_checkout(inst)
    set_env_project(inst, None)
    done = cli(shimmed, inst, 'stop')
    assert done.returncode == 0, done.stderr
    assert 'COMPOSE_PROJECT_NAME' in done.stdout + done.stderr
    assert any(' -p openprocessor ' in ln for ln in shimmed.mutating_docker_calls())


def test_an_install_dir_without_a_project_name_still_refuses(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    set_env_project(inst, None)
    done = cli(shimmed, inst, 'stop')
    assert done.returncode == 3
    assert shimmed.mutating_docker_calls() == []


def test_curation_up_without_names_starts_the_curation_profile(
    shimmed: Shimmed, inst: Path
) -> None:
    shimmed.flag('allow_mutations')
    done = cli(shimmed, inst, 'curation', 'up')
    assert done.returncode == 0, done.stderr
    ups = [ln for ln in shimmed.mutating_docker_calls() if ' up -d' in ln]
    assert ups, shimmed.mutating_docker_calls()
    assert ups[0].endswith(' --profile curation up -d')


def test_curation_up_accepts_service_names(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    done = cli(shimmed, inst, 'curation', 'up', 'curation-detection-worker', 'curation-evaluator')
    assert done.returncode == 0, done.stderr
    ups = [ln for ln in shimmed.mutating_docker_calls() if ' up -d' in ln]
    assert ups[0].endswith(' up -d curation-detection-worker curation-evaluator')


@pytest.mark.parametrize('bad', ['api', 'triton-server', 'nonsense'])
def test_curation_up_refuses_a_name_that_is_not_a_curation_service(
    shimmed: Shimmed, inst: Path, bad: str
) -> None:
    shimmed.flag('allow_mutations')
    done = cli(shimmed, inst, 'curation', 'up', bad)
    assert done.returncode != 0
    assert bad in done.stderr + done.stdout
    assert shimmed.mutating_docker_calls() == []
