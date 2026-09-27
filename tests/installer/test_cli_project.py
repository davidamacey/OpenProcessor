"""B3: the deploy CLI (`openprocessor`) must only ever act on the compose
project its own install created. It resolves the name from the install's
.env (cross-checked with .install/state.json), never from a default, and
refuses every state-changing command when anything disagrees.
"""

from __future__ import annotations

import stat
import subprocess
from typing import TYPE_CHECKING

import pytest
from installer_harness import PROJECT, RELEASE


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed


@pytest.fixture
def inst(shimmed: Shimmed) -> Path:
    """A real install dir produced by a dry run of the installer."""
    result = shimmed.run(
        [
            '--dry-run',
            '--unattended',
            '--dir',
            'inst',
            '--project',
            PROJECT,
            '--version',
            RELEASE,
            '--tiers',
            'core',
        ]
    )
    assert result.returncode == 0, result.stderr
    (shimmed.state / 'mutations.log').unlink(missing_ok=True)
    shimmed.log.write_text('')
    return (shimmed.root / 'inst').resolve()


def cli(shimmed: Shimmed, inst: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(inst / 'openprocessor'), *args],
        check=False,
        cwd=str(shimmed.root),
        env=shimmed.env(**env),
        capture_output=True,
        text=True,
        timeout=60,
        start_new_session=True,
    )


def set_env_project(inst: Path, value: str | None) -> None:
    env = inst / '.env'
    lines = [
        ln for ln in env.read_text().splitlines() if not ln.startswith('COMPOSE_PROJECT_NAME=')
    ]
    if value is not None:
        lines.append(f'COMPOSE_PROJECT_NAME={value}')
    env.write_text('\n'.join(lines) + '\n')


def set_state_project(inst: Path, value: str) -> None:
    state = inst / '.install' / 'state.json'
    state.write_text(state.read_text().replace(f'"project": "{PROJECT}"', f'"project": "{value}"'))


def test_stop_uses_the_project_from_the_installs_env(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    result = cli(shimmed, inst, 'stop')
    assert result.returncode == 0, result.stderr
    downs = [ln for ln in shimmed.mutating_docker_calls() if ln.endswith(' down')]
    assert downs, shimmed.mutating_docker_calls()
    assert all(
        f'docker compose -p {PROJECT} --env-file {inst}/.env --project-directory {inst} ' in ln
        for ln in downs
    )


def test_shell_compose_project_name_is_ignored(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    result = cli(shimmed, inst, 'stop', COMPOSE_PROJECT_NAME='openprocessor')
    assert result.returncode == 0, result.stderr
    assert all(f' -p {PROJECT} ' in ln for ln in shimmed.mutating_docker_calls())


HOSTILE = [
    'no_env',
    'env_without_project',
    'op_project_env',
    'state_mismatch',
    'live_owned',
    'docker_down',
    'ps_fail',
]


def make_hostile(shimmed: Shimmed, inst: Path, case: str) -> dict[str, str]:
    env: dict[str, str] = {}
    if case == 'no_env':
        (inst / '.env').unlink()
    elif case == 'env_without_project':
        set_env_project(inst, None)
    elif case == 'op_project_env':
        env['OP_PROJECT'] = 'openprocessor'
    elif case == 'state_mismatch':
        set_env_project(inst, 'openprocessor')
    elif case == 'live_owned':
        # Even an install that really is named "openprocessor" may not touch
        # containers of that name created from another directory.
        set_env_project(inst, 'openprocessor')
        set_state_project(inst, 'openprocessor')
        shimmed.containers(
            [('openprocessor', '/srv/live-stack', 'openprocessor-api', '127.0.0.1:4603->8000/tcp')]
        )
    elif case == 'docker_down':
        shimmed.flag('docker_down')
    elif case == 'ps_fail':
        shimmed.flag('ps_fail')
    return env


@pytest.mark.parametrize('case', HOSTILE)
@pytest.mark.parametrize(
    'command', [['stop'], ['start'], ['restart'], ['curation', 'down'], ['train-mode', 'on']]
)
def test_cli_can_never_touch_a_project_it_did_not_create(
    shimmed: Shimmed, inst: Path, case: str, command: list[str]
) -> None:
    shimmed.flag('allow_mutations')
    env = make_hostile(shimmed, inst, case)
    result = cli(shimmed, inst, *command, **env)
    assert result.returncode != 0, f'{command} succeeded in hostile case {case}'
    assert shimmed.mutating_docker_calls() == [], f'{case}: {shimmed.mutating_docker_calls()}'
    assert not [ln for ln in shimmed.log_lines('docker compose') if ' -p openprocessor ' in ln]


def test_a_real_install_named_openprocessor_still_works_when_it_owns_the_project(
    shimmed: Shimmed, inst: Path
) -> None:
    shimmed.flag('allow_mutations')
    set_env_project(inst, 'openprocessor')
    set_state_project(inst, 'openprocessor')
    shimmed.containers([('openprocessor', str(inst), 'openprocessor-api', '')])
    result = cli(shimmed, inst, 'stop')
    assert result.returncode == 0, result.stderr
    assert [
        ln
        for ln in shimmed.mutating_docker_calls()
        if ' -p openprocessor ' in ln and ln.endswith(' down')
    ]


def test_public_bind_from_the_shell_needs_the_installs_consent(
    shimmed: Shimmed, inst: Path
) -> None:
    shimmed.flag('allow_mutations')
    result = cli(shimmed, inst, 'start', OP_BIND_ADDRESS='0.0.0.0')
    assert result.returncode == 9
    assert shimmed.mutating_docker_calls() == []


def test_train_mode_reports_failure_instead_of_success(shimmed: Shimmed, inst: Path) -> None:
    result = cli(shimmed, inst, 'train-mode', 'on')
    assert result.returncode != 0
    assert 'train-mode on (stopped' not in result.stdout


def test_train_mode_stops_a_colocated_segmenter_and_restores_it(
    shimmed: Shimmed, inst: Path
) -> None:
    shimmed.flag('allow_mutations')
    result = cli(shimmed, inst, 'train-mode', 'on')
    assert result.returncode == 0, result.stderr
    assert any(ln.endswith(' stop vlm segmenter') for ln in shimmed.mutating_docker_calls())
    assert (inst / '.install' / 'train_mode').read_text().startswith('on vlm segmenter')
    result = cli(shimmed, inst, 'train-mode', 'off')
    assert result.returncode == 0, result.stderr
    assert any(ln.endswith(' up -d vlm segmenter') for ln in shimmed.mutating_docker_calls())


def test_models_install_runs_the_shared_library_groups(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    result = cli(shimmed, inst, 'models', 'install', '--only', 'yolo')
    assert result.returncode == 0, result.stderr
    m = shimmed.mutating_docker_calls()
    assert [ln for ln in m if 'export/export_models.py' in ln and f' -p {PROJECT} ' in ln]
    assert not [ln for ln in m if 'export_mobileclip' in ln]


def test_uninstall_runs_this_installs_setup_script(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    result = cli(shimmed, inst, 'uninstall', '--unattended')
    assert result.returncode == 0, result.stderr
    assert [
        ln
        for ln in shimmed.mutating_docker_calls()
        if ln.endswith('down --remove-orphans') and f' -p {PROJECT} ' in ln
    ]


def test_config_show_redacts_every_secret(shimmed: Shimmed, inst: Path) -> None:
    env = inst / '.env'
    env.write_text(
        env.read_text() + 'HF_TOKEN=hf_abcdefghijkl\nexport OPENAI_API_KEY=sk-1\nDB_PASSWORD=pw\n'
        'MY_SECRET=s\nOP_VLM_API_KEY=k\nAPI_PORT=4603\n'
    )
    result = cli(shimmed, inst, 'config', 'show')
    assert result.returncode == 0, result.stderr
    for leaked in ('hf_abcdefghijkl', 'sk-1', '=pw', '=s\n', '=k\n'):
        assert leaked not in result.stdout + '\n', leaked
    assert 'API_PORT=4603' in result.stdout


def test_vlm_key_slug_cannot_escape_secrets(shimmed: Shimmed, inst: Path) -> None:
    result = cli(shimmed, inst, 'vlm', 'key', 'set', '../../.bashrc')
    assert result.returncode == 2
    assert not (inst.parent / '.bashrc').exists()


def test_cli_files_keep_their_modes(inst: Path) -> None:
    assert stat.S_IMODE((inst / 'openprocessor').stat().st_mode) == 0o700
    assert stat.S_IMODE((inst / 'setup-openprocessor.sh').stat().st_mode) == 0o700
