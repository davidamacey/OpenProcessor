"""The dry run plans every step and changes nothing; project, container and port collisions fail closed."""

from __future__ import annotations

import stat
from typing import TYPE_CHECKING

from _setup_dry_run_helpers import compose_lines, configure, dry, env_file
from installer_harness import PROJECT, RELEASE


if TYPE_CHECKING:
    from installer_harness import Shimmed


# --- the dry run plans everything and changes nothing --------------------------


def test_dry_run_makes_no_mutating_docker_call_and_plans_every_step(shimmed: Shimmed) -> None:
    result = dry(shimmed)
    assert result.returncode == 0, result.stderr
    assert shimmed.mutating_docker_calls() == []
    dry_lines = [ln for ln in result.stdout.splitlines() if ln.startswith('DRY: ')]
    joined = '\n'.join(dry_lines)
    for expected in (
        'docker pull davidamacey/openprocessor@sha256:',
        'docker pull davidamacey/openprocessor-triton@sha256:',
        'docker run --rm --gpus device=0',
        'up -d triton-server',
        'api python -m export.preflight',
        '--entrypoint python api /app/export/sync_model_seed.py /opt/openprocessor/model_repo_seed /app/models',
        'export/export_models.py',
        'export_paddleocr_rec.py',
        'up -d --remove-orphans',
    ):
        assert expected in joined, f'missing planned step: {expected}'
    assert len(dry_lines) >= 20
    assert 'installed and healthy' not in result.stdout
    assert 'nothing was written, pulled, started or removed' in result.stdout
    assert not (shimmed.root / 'inst').exists()


def test_every_compose_call_uses_exactly_the_install_project(shimmed: Shimmed) -> None:
    result = dry(shimmed)
    assert result.returncode == 0, result.stderr
    lines = compose_lines(shimmed, result)
    assert lines
    for line in lines:
        assert f' -p {PROJECT} ' in f'{line} ', line
        assert ' -p openprocessor ' not in f'{line} ', line
        assert '--env-file' in line
        assert '--project-directory' in line, line
    assert not (shimmed.root / 'inst').exists()
    assert configure(shimmed).returncode == 0
    assert env_file(shimmed)['COMPOSE_PROJECT_NAME'] == PROJECT


def test_env_state_and_install_meta_permissions(shimmed: Shimmed) -> None:
    result = configure(shimmed)
    assert result.returncode == 0, result.stderr
    inst = shimmed.root / 'inst'
    assert stat.S_IMODE((inst / '.env').stat().st_mode) == 0o600
    assert stat.S_IMODE((inst / '.install').stat().st_mode) == 0o700
    assert stat.S_IMODE((inst / '.install' / 'state.json').stat().st_mode) == 0o600
    assert stat.S_IMODE((inst / '.install' / 'install.log').stat().st_mode) == 0o600
    # R5: secrets are private; release files are readable, because some are
    # bind-mounted into containers running as other users.
    assert stat.S_IMODE((inst / '.install' / 'managed.env').stat().st_mode) == 0o600
    assert stat.S_IMODE((inst / 'openprocessor').stat().st_mode) == 0o755
    assert stat.S_IMODE((inst / 'setup-openprocessor.sh').stat().st_mode) == 0o755
    assert stat.S_IMODE((inst / 'docker-compose.yml').stat().st_mode) == 0o644
    assert stat.S_IMODE((inst / 'monitoring' / 'prometheus.yml').stat().st_mode) == 0o644
    assert stat.S_IMODE((inst / 'monitoring' / 'loki-config.yml').stat().st_mode) == 0o644
    assert stat.S_IMODE((inst / 'monitoring' / 'dashboards').stat().st_mode) == 0o755
    assert stat.S_IMODE((inst / 'models').stat().st_mode) == 0o755
    assert env_file(shimmed)['OP_BIND_ADDRESS'] == '127.0.0.1'


def test_shell_compose_project_name_cannot_rename_the_containers(shimmed: Shimmed) -> None:
    # compose interpolates container_name from COMPOSE_PROJECT_NAME, and a
    # shell value beats .env: dc() must pin it to the install's project.
    result = dry(shimmed, COMPOSE_PROJECT_NAME='openprocessor')
    assert result.returncode == 0, result.stderr + result.stdout
    assert "all 'opinst-test-*'" in result.stdout


# --- collisions fail closed ------------------------------------------------------


def test_default_project_owned_by_another_dir_exits_3(shimmed: Shimmed) -> None:
    shimmed.containers(
        [('openprocessor', '/srv/live-stack', 'openprocessor-api', '127.0.0.1:4603->8000/tcp')]
    )
    result = shimmed.run(
        ['--dry-run', '--unattended', '--dir', 'inst', '--version', RELEASE, '--tiers', 'core']
    )
    assert result.returncode == 3, result.stderr
    assert '--project' in result.stderr
    assert shimmed.mutating_docker_calls() == []


def test_docker_unreachable_fails_closed(shimmed: Shimmed) -> None:
    shimmed.flag('docker_down')
    result = dry(shimmed)
    assert result.returncode == 5
    assert 'Docker' in result.stderr


def test_docker_ps_failure_fails_closed(shimmed: Shimmed) -> None:
    shimmed.flag('ps_fail')
    result = dry(shimmed)
    assert result.returncode == 5, result.stderr
    assert not (shimmed.root / 'inst' / 'docker-compose.yml').exists()


def test_container_name_owned_by_another_project_exits_3(shimmed: Shimmed) -> None:
    shimmed.containers([('someone-else', '/x', f'{PROJECT}-api', '')])
    result = dry(shimmed)
    assert result.returncode == 3
    assert f"'{PROJECT}-api'" in result.stderr


def test_unprefixed_container_names_exit_3(shimmed: Shimmed) -> None:
    shimmed.flag(
        'config.json',
        '{\n  "networks": {\n    "triton_net": {\n      "name": "opinst-test_triton_net"\n    }\n  },\n'
        '  "services": {\n    "api": {\n      "container_name": "openprocessor-api"\n    }\n  }\n}\n',
    )
    result = dry(shimmed)
    assert result.returncode == 3
    assert "does not start with 'opinst-test-'" in result.stderr


def test_wrong_network_name_exits_3(shimmed: Shimmed) -> None:
    shimmed.flag(
        'config.json',
        '{\n  "networks": {\n    "triton_net": {\n      "name": "openprocessor_triton_net"\n    }\n  },\n'
        '  "services": {\n    "api": {\n      "container_name": "opinst-test-api"\n    }\n  }\n}\n',
    )
    assert dry(shimmed).returncode == 3


# --- ports -----------------------------------------------------------------------


def test_port_conflict_shifts_and_prints_the_mapping(shimmed: Shimmed) -> None:
    shimmed.flag('ports_in_use', '4603\n')
    result = configure(shimmed)
    assert result.returncode == 0, result.stderr
    assert 'port 4603 (API_PORT) is in use: using 4604' in result.stderr
    assert env_file(shimmed)['API_PORT'] == '4604'


def test_port_held_by_another_container_counts_as_used(shimmed: Shimmed) -> None:
    shimmed.containers([('other', '/o', 'other-os', '127.0.0.1:4607->9200/tcp')])
    result = configure(shimmed)
    assert result.returncode == 0, result.stderr
    assert env_file(shimmed)['OPENSEARCH_PORT'] != '4607'


def test_this_projects_own_ports_count_as_free(shimmed: Shimmed) -> None:
    inst = (shimmed.root / 'inst').resolve()
    inst.mkdir()
    shimmed.containers([(PROJECT, str(inst), f'{PROJECT}-api', '127.0.0.1:4603->8000/tcp')])
    shimmed.flag('ports_in_use', '4603\n')
    result = configure(shimmed)
    assert result.returncode == 0, result.stderr
    assert env_file(shimmed)['API_PORT'] == '4603'


def test_port_base_moves_the_whole_block(shimmed: Shimmed) -> None:
    result = configure(shimmed, '--port-base', '4970')
    assert result.returncode == 0, result.stderr
    env = env_file(shimmed)
    assert (env['TRITON_HTTP_PORT'], env['API_PORT'], env['OPENSEARCH_PORT']) == (
        '4970',
        '4973',
        '4977',
    )
