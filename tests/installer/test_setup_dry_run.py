"""Dry-run and invariant tests for setup-openprocessor.sh (installer plan 9.2).

Every run goes through PATH shims (tests/installer/shims): the docker shim
records any state-changing call in mutations.log and refuses it, so a dry
run that tried to change anything would both fail and be visible.
"""

from __future__ import annotations

import shutil
import stat
from pathlib import Path
from typing import TYPE_CHECKING

from installer_harness import GPU_HOST, PROJECT, RELEASE, build_fake_release, fake_digest


if TYPE_CHECKING:
    from installer_harness import Shimmed


def dry(shimmed: Shimmed, *extra: str, tiers: str = 'core', **env: str):
    return shimmed.run(
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
            tiers,
            *extra,
        ],
        **env,
    )


def configure(shimmed: Shimmed, *extra: str, tiers: str = 'core', **env: str):
    """A real (non-dry) run that writes the install but starts nothing; the
    docker shim accepts the pulls/probe and records them."""
    shimmed.flag('allow_mutations')
    return shimmed.run(
        [
            '--unattended',
            '--no-start',
            '--dir',
            'inst',
            '--project',
            PROJECT,
            '--version',
            RELEASE,
            '--tiers',
            tiers,
            *extra,
        ],
        **env,
    )


def env_file(shimmed: Shimmed) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (shimmed.root / 'inst' / '.env').read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            out[k] = v
    return out


def compose_lines(shimmed: Shimmed, result) -> list[str]:
    lines = [
        ln[len('DRY: ') :]
        for ln in result.stdout.splitlines()
        if ln.startswith('DRY: docker compose')
    ]
    lines += list(shimmed.log_lines('docker compose'))
    return lines


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
        'yolo-api python -m export.preflight',
        '--entrypoint cp yolo-api -rn /opt/openprocessor/model_repo_seed/. /app/models/',
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
        assert f' -p {PROJECT} ' in f'{line} ' or f' -p {PROJECT}-cw ' in f'{line} ', line
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


def test_cropwright_project_owned_elsewhere_exits_3(shimmed: Shimmed) -> None:
    shimmed.containers([(f'{PROJECT}-cw', '/elsewhere/cropwright', 'x', '')])
    assert dry(shimmed).returncode == 3


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
        '  "services": {\n    "yolo-api": {\n      "container_name": "openprocessor-api"\n    }\n  }\n}\n',
    )
    result = dry(shimmed)
    assert result.returncode == 3
    assert "does not start with 'opinst-test-'" in result.stderr


def test_wrong_network_name_exits_3(shimmed: Shimmed) -> None:
    shimmed.flag(
        'config.json',
        '{\n  "networks": {\n    "triton_net": {\n      "name": "openprocessor_triton_net"\n    }\n  },\n'
        '  "services": {\n    "yolo-api": {\n      "container_name": "opinst-test-api"\n    }\n  }\n}\n',
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


# --- .env content ------------------------------------------------------------------


def test_opensearch_heap_is_ram_over_8_clamped(shimmed: Shimmed) -> None:
    result = configure(shimmed)
    assert result.returncode == 0, result.stderr
    meminfo = Path('/proc/meminfo').read_text().splitlines()
    mem_kib = next(int(ln.split()[1]) for ln in meminfo if ln.startswith('MemTotal:'))
    expected = min(8, max(1, mem_kib // 1024 // 1024 // 8))
    assert env_file(shimmed)['OPENSEARCH_HEAP'] == f'{expected}g'


def test_rerun_is_idempotent_and_keeps_user_values(shimmed: Shimmed) -> None:
    assert configure(shimmed).returncode == 0
    env_path = shimmed.root / 'inst' / '.env'
    first = env_path.read_text()
    assert configure(shimmed).returncode == 0
    assert env_path.read_text() == first, 'a re-run changed .env'
    assert list((shimmed.root / 'inst' / 'backups').iterdir()), 'a re-run must back up first'

    edited = first.replace('OPENSEARCH_HEAP=', 'OPENSEARCH_HEAP=3g\n#was ', 1)
    env_path.write_text(edited)
    result = configure(shimmed)
    assert result.returncode == 0, result.stderr
    assert env_file(shimmed)['OPENSEARCH_HEAP'] == '3g'
    assert 'keeping your OPENSEARCH_HEAP=3g' in result.stdout


def test_vlm_tier_writes_the_catalog_keys(shimmed: Shimmed) -> None:
    result = configure(shimmed, tiers='vlm')
    assert result.returncode == 0, result.stderr
    env = env_file(shimmed)
    lock = dict(
        ln.split('=', 1)
        for ln in (shimmed.root / 'inst' / 'images.lock').read_text().splitlines()
        if ln and not ln.startswith('#')
    )
    assert env['VLM_SERVED_MODEL_NAME'] == 'local-vlm'
    assert env['OP_VLM_MODEL'] == 'local-vlm'
    assert env['OP_VLM_URL'] == 'http://vlm:8000/v1'
    assert env['VLM_IMAGE'] == lock['vlm_gemma4']
    assert env['VLM_GPU_MEMORY_UTILIZATION'] == '0.42'
    assert env['VLM_LIMIT_MM_IMAGES'] == env['OP_VLM_MAX_IMAGES_PER_CALL'] == '8'
    assert 'vlm' in env['COMPOSE_PROFILES'].split(',')


def test_vlm_tier_refused_where_no_tested_entry_fits(shimmed: Shimmed) -> None:
    shimmed.gpus('0, NVIDIA GeForce RTX 4090, 24564, 0, 8.9\n')
    result = dry(shimmed, tiers='vlm')
    assert result.returncode == 4
    assert 'no tested catalog entry fits' in result.stderr


def test_unverified_vlm_only_by_explicit_id_and_force(shimmed: Shimmed) -> None:
    shimmed.gpus('0, NVIDIA GeForce RTX 4090, 24564, 0, 8.9\n')
    assert dry(shimmed, '--vlm-model-id', 'qwen2.5-vl-7b-awq', tiers='vlm').returncode == 4
    result = configure(shimmed, '--vlm-model-id', 'qwen2.5-vl-7b-awq', '--force', tiers='vlm')
    assert result.returncode == 0, result.stderr
    assert 'not yet verified' in result.stderr
    assert env_file(shimmed)['VLM_CATALOG_ID'] == 'qwen2.5-vl-7b-awq'


def test_host_shaped_three_gpus_write_the_placement(shimmed: Shimmed, tmp_path: Path) -> None:
    shimmed.gpus(GPU_HOST)
    tok = tmp_path / 'tok'
    tok.write_text('hf_TESTSECRET1234567890\n')
    tok.chmod(0o600)
    result = configure(shimmed, tiers='segmenter,vlm,trainer', HF_TOKEN_FILE=str(tok))
    assert result.returncode == 0, result.stderr
    env = env_file(shimmed)
    assert env['TRITON_GPU_ID'] == env['API_GPU_ID'] == '1'
    assert env['VLM_GPU_ID'] == '0'
    assert env['SEGMENTER_GPU_ID'] == env['OP_TRAIN_GPU_ORDER'] == env['EVALUATOR_GPU_ID'] == '2'
    assert env['OP_GPU_ALLOWED_IDS'] == '0,1,2'
    assert env['OP_GPU_LABELS'].startswith('0=NVIDIA RTX A6000,1=NVIDIA GeForce RTX 3080 Ti')


# --- HF token ----------------------------------------------------------------------


def test_gated_tier_without_token_exits_6(shimmed: Shimmed) -> None:
    result = dry(shimmed, tiers='segmenter')
    assert result.returncode == 6
    assert 'HF_TOKEN_FILE' in result.stderr


def test_rejected_token_exits_6_with_licence_hint(shimmed: Shimmed, tmp_path: Path) -> None:
    tok = tmp_path / 'tok'
    tok.write_text('hf_TESTSECRET1234567890\n')
    tok.chmod(0o600)
    shimmed.flag('hf_code', '403')
    result = dry(shimmed, tiers='segmenter', HF_TOKEN_FILE=str(tok))
    assert result.returncode == 6
    assert 'accept the licence at https://huggingface.co/facebook/sam3' in result.stderr


def test_core_never_asks_for_a_token(shimmed: Shimmed) -> None:
    result = dry(shimmed)
    assert result.returncode == 0
    assert 'no HuggingFace token needed' in result.stdout
    assert not [ln for ln in shimmed.log_lines('curl') if 'huggingface.co' in ln]


# --- flags -------------------------------------------------------------------------


def test_unknown_flag_exits_2(shimmed: Shimmed) -> None:
    assert shimmed.run(['--not-a-real-flag']).returncode == 2


def test_missing_flag_value_exits_2_with_a_message(shimmed: Shimmed) -> None:
    result = shimmed.run(['--dir'])
    assert result.returncode == 2
    assert '--dir needs a value' in result.stderr
    assert 'unbound variable' not in result.stderr


def test_invalid_values_exit_2(shimmed: Shimmed) -> None:
    for args in (
        ['--project', 'Bad Name'],
        ['--port-base', 'abc'],
        ['--profile', 'huge'],
        ['--bind', 'example.com'],
        ['--bind', '::1'],
        ['--image-tag', 'latest'],
        ['--purge-data'],
    ):
        assert shimmed.run(['--dry-run', *args]).returncode == 2, args


def test_unattended_truthy_values(shimmed: Shimmed) -> None:
    for value in ('true', 'yes', '1'):
        result = shimmed.run(
            [
                '--dry-run',
                '--dir',
                'inst',
                '--project',
                PROJECT,
                '--version',
                RELEASE,
                '--tiers',
                'core',
                '--bind',
                '0.0.0.0',
            ],
            OP_UNATTENDED=value,
        )
        assert 'OP_ALLOW_PUBLIC_BIND=1' in result.stderr, value


def test_cpu_exits_4_with_the_honest_message(shimmed: Shimmed) -> None:
    result = dry(shimmed, '--cpu')
    assert result.returncode == 4
    assert 'no CPU inference path' in result.stderr


def test_no_gpu_exits_4(shimmed: Shimmed) -> None:
    shimmed.gpus(None)
    assert dry(shimmed).returncode == 4


def test_control_plane_only_starts_api_and_opensearch_only(shimmed: Shimmed) -> None:
    shimmed.gpus(None)
    result = dry(shimmed, '--cpu', '--control-plane-only')
    assert result.returncode == 0, result.stderr
    dry_compose = [ln for ln in result.stdout.splitlines() if ln.startswith('DRY: docker compose')]
    assert any(ln.endswith(' up -d opensearch') for ln in dry_compose)
    assert any(ln.endswith(' up -d --no-deps yolo-api') for ln in dry_compose)
    assert not any('triton-server' in ln for ln in dry_compose)
    assert all('docker-compose.cpu.yml' in ln for ln in dry_compose)
    assert configure(shimmed, '--cpu', '--control-plane-only').returncode == 0
    assert '!reset' in (shimmed.root / 'inst' / 'docker-compose.cpu.yml').read_text()
    assert 'NOT a functional inference install' in result.stdout + result.stderr


# --- bind / external consent ---------------------------------------------------------


def test_public_bind_unattended_needs_explicit_consent(shimmed: Shimmed) -> None:
    result = dry(shimmed, '--bind', '0.0.0.0')
    assert result.returncode == 9
    assert 'OP_ALLOW_PUBLIC_BIND=1' in result.stderr
    ok = configure(shimmed, '--bind', '0.0.0.0', OP_ALLOW_PUBLIC_BIND='1')
    assert ok.returncode == 0, ok.stderr
    assert env_file(shimmed)['OP_BIND_ADDRESS'] == '0.0.0.0'


def test_localhost_bind_is_normalised_to_an_ip(shimmed: Shimmed) -> None:
    assert configure(shimmed, '--bind', 'localhost').returncode == 0
    assert env_file(shimmed)['OP_BIND_ADDRESS'] == '127.0.0.1'


def test_external_vlm_needs_consent(shimmed: Shimmed) -> None:
    for url in (
        'https://api.example.invalid/v1',
        'http://127.0.0.1@evil.example/v1',
        'http://10.evil.example/v1',
    ):
        result = dry(shimmed, '--vlm-remote', url, '--vlm-model', 'some-model')
        assert result.returncode == 9, url
        assert 'OP_ALLOW_EXTERNAL_VLM=1' in result.stderr, url
    ok = dry(shimmed, '--vlm-remote', 'http://192.168.1.20:8000/v1', '--vlm-model', 'm')
    assert ok.returncode == 0, ok.stderr


def test_remote_vlm_key_lands_only_in_secrets(shimmed: Shimmed, tmp_path: Path) -> None:
    key = 'sk-TESTVLMKEY0123456789'  # gitleaks:allow
    key_file = tmp_path / 'vlm.key'
    key_file.write_text(key)
    key_file.chmod(0o600)
    result = configure(
        shimmed,
        '--vlm-remote',
        'https://api.example.invalid/v1',
        '--vlm-model',
        'm',
        '--vlm-key-file',
        str(key_file),
        OP_ALLOW_EXTERNAL_VLM='1',
    )
    assert result.returncode == 0, result.stderr
    inst = shimmed.root / 'inst'
    secret = inst / 'secrets' / 'vlm' / 'env'
    assert secret.read_text() == key
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600
    for text in (
        (inst / '.env').read_text(),
        result.stdout,
        result.stderr,
        (inst / '.install' / 'install.log').read_text(),
        shimmed.log.read_text(),
    ):
        assert key not in text


def test_remote_and_local_vlm_are_exclusive(shimmed: Shimmed) -> None:
    result = dry(shimmed, '--vlm-remote', 'http://127.0.0.1:9/v1', '--vlm-model', 'm', tiers='vlm')
    assert result.returncode == 2


# --- Cropwright ----------------------------------------------------------------------


def _cw_env(shimmed: Shimmed) -> str:
    return (shimmed.root / 'inst' / 'cropwright' / '.env').read_text()


def test_cropwright_is_reachable_from_the_lan_by_default_with_a_warning(shimmed: Shimmed) -> None:
    # Owner decision (plan 7 addendum): the UI serves this computer AND the LAN.
    result = configure(shimmed, tiers='cropwright')
    assert result.returncode == 0, result.stderr
    cw_env = _cw_env(shimmed)
    assert 'CROPWRIGHT_BIND_ADDRESS=0.0.0.0' in cw_env
    assert f'CROPWRIGHT_IMAGE=davidamacey/cropwright@{fake_digest("cropwright")}' in cw_env
    assert f'CROPWRIGHT_CONTAINER_NAME={PROJECT}-cropwright' in cw_env
    assert f'OP_DOCKER_NETWORK={PROJECT}_triton_net' in cw_env
    assert stat.S_IMODE((shimmed.root / 'inst' / 'cropwright' / '.env').stat().st_mode) == 0o600
    assert not (shimmed.root / 'inst' / 'cropwright' / 'docker-compose.bind.yml').exists()
    # The backend stays on loopback; only Cropwright is on the LAN.
    assert env_file(shimmed)['OP_BIND_ADDRESS'] == '127.0.0.1'
    assert 'reachable from your LAN and has NO login' in result.stderr
    assert 'never port-forward' in result.stderr
    dry_run = dry(shimmed, tiers='cropwright')
    assert dry_run.returncode == 0, dry_run.stderr
    cw = [
        ln
        for ln in dry_run.stdout.splitlines()
        if ln.startswith(f'DRY: docker compose -p {PROJECT}-cw')
    ]
    assert cw
    assert not [ln for ln in cw if 'bind.yml' in ln]


def test_local_only_keeps_cropwright_on_this_computer(shimmed: Shimmed) -> None:
    result = configure(shimmed, '--local-only', tiers='cropwright')
    assert result.returncode == 0, result.stderr
    assert 'CROPWRIGHT_BIND_ADDRESS=127.0.0.1' in _cw_env(shimmed)
    assert 'reachable from your LAN' not in result.stderr


def test_a_shell_bind_cannot_override_the_chosen_cropwright_bind(shimmed: Shimmed) -> None:
    result = configure(
        shimmed, '--local-only', tiers='cropwright', CROPWRIGHT_BIND_ADDRESS='0.0.0.0'
    )
    assert result.returncode == 0, result.stderr
    assert 'CROPWRIGHT_BIND_ADDRESS=127.0.0.1' in _cw_env(shimmed)


def test_cropwright_compose_that_ignores_the_bind_is_refused(shimmed: Shimmed) -> None:
    shimmed.flag('cw_ignores_bind')
    result = configure(shimmed, '--local-only', tiers='cropwright')
    assert result.returncode == 7
    assert 'must honour CROPWRIGHT_BIND_ADDRESS' in result.stderr


def test_cropwright_file_with_wrong_checksum_is_refused(shimmed: Shimmed, tmp_path: Path) -> None:
    release = tmp_path / 'rel'
    shutil.copytree(shimmed.release, release)
    compose = next(release.glob('cw/*/docker-compose.yml'))
    compose.write_text(compose.read_text() + '# tampered\n')
    shimmed.release = release
    result = dry(shimmed, tiers='cropwright')
    assert result.returncode == 7
    assert 'docker-compose.yml failed checksum verification' in result.stderr


def test_cropwright_sha256sums_must_match_the_lock(shimmed: Shimmed, tmp_path: Path) -> None:
    release = tmp_path / 'rel'
    shutil.copytree(shimmed.release, release)
    sums = next(release.glob('cw/*/SHA256SUMS'))
    sums.write_text(sums.read_text() + '# tampered\n')
    shimmed.release = release
    result = dry(shimmed, tiers='cropwright')
    assert result.returncode == 7
    assert 'SHA256SUMS does not match cropwright.lock' in result.stderr


def _release_copy(shimmed: Shimmed, tmp_path: Path) -> Path:
    release = tmp_path / 'rel'
    shutil.copytree(shimmed.release, release)
    shimmed.release = release
    return release / 'assets' / RELEASE


def test_truncated_tarball_fails_before_anything_is_installed(
    shimmed: Shimmed, tmp_path: Path
) -> None:
    assets = _release_copy(shimmed, tmp_path)
    tarball = assets / f'openprocessor-deploy-{RELEASE}.tar.gz'
    data = tarball.read_bytes()
    tarball.write_bytes(data[: len(data) // 2])
    result = dry(shimmed)
    assert result.returncode == 7
    inst = shimmed.root / 'inst'
    assert not (inst / '.env').exists()
    assert not (inst / 'docker-compose.yml').exists()
    assert shimmed.mutating_docker_calls() == []


def test_raw_fallback_verifies_every_file(shimmed: Shimmed, tmp_path: Path) -> None:
    assets = _release_copy(shimmed, tmp_path)
    (assets / f'openprocessor-deploy-{RELEASE}.tar.gz').unlink()
    raw_env = shimmed.release / 'raw' / RELEASE / 'env.template'
    ok = dry(shimmed)
    assert ok.returncode == 0, ok.stderr
    assert 'fetching verified raw files' in ok.stderr
    raw_env.write_text(raw_env.read_text() + 'OP_BIND_ADDRESS=0.0.0.0\n')
    assert not (shimmed.root / 'inst').exists()
    bad = dry(shimmed)
    assert bad.returncode == 7
    assert 'env.template failed verification' in bad.stderr


def test_images_lock_with_latest_is_rejected(shimmed: Shimmed, tmp_path: Path) -> None:
    shimmed.release = build_fake_release(
        tmp_path / 'r',
        lock_override='api=davidamacey/openprocessor:latest@sha256:' + 'a' * 64 + '\n',
    )
    result = dry(shimmed)
    assert result.returncode == 7
    assert 'images.lock' in result.stderr


def test_committed_placeholder_lock_is_refused_not_installed(
    shimmed: Shimmed, tmp_path: Path, repo_root: Path
) -> None:
    shimmed.release = build_fake_release(
        tmp_path / 'r', lock_override=(repo_root / 'images.lock').read_text()
    )
    result = dry(shimmed)
    assert result.returncode == 7
    assert '--image-tag' in result.stderr


def _local_images(shimmed: Shimmed, repo: str, tag: str) -> None:
    images = shimmed.state / 'images'
    images.mkdir(exist_ok=True)
    for name in ('openprocessor', 'openprocessor-triton'):
        (images / f'{repo}_{name}_{tag}').write_text(f'{repo}/{name}@sha256:{"0" * 64}\n')


def test_image_tag_mode_uses_local_tags_without_lock_pins(shimmed: Shimmed) -> None:
    _local_images(shimmed, 'opinst', 'localtest')
    result = configure(shimmed, '--image-tag', 'localtest', OP_IMAGE_REPO='opinst')
    assert result.returncode == 0, result.stderr
    env = env_file(shimmed)
    assert env['OP_IMAGE_REPO'] == 'opinst'
    assert env['OP_IMAGE_TAG'] == 'localtest'
    assert env['OP_API_IMAGE'] == ''
    assert not [ln for ln in shimmed.mutating_docker_calls() if 'pull opinst/' in ln]
    assert 'UNPINNED install' in result.stderr


def test_image_tag_mode_never_pulls_an_unpinned_image_of_ours(shimmed: Shimmed) -> None:
    # r11: a missing local build must not be fetched by tag from a registry.
    result = configure(shimmed, '--image-tag', 'localtest', OP_IMAGE_REPO='opinst')
    assert result.returncode == 7
    assert 'local builds only' in result.stderr
    assert not [
        ln for ln in shimmed.mutating_docker_calls() if ln.startswith('docker pull opinst/')
    ]


def test_non_https_artifact_base_is_refused(shimmed: Shimmed) -> None:
    result = dry(shimmed, OP_ARTIFACT_BASE_URL='http://release.test/assets')
    assert result.returncode == 2
    assert shimmed.log_lines('curl') == []


def test_version_must_be_a_release_tag(shimmed: Shimmed) -> None:
    result = shimmed.run(
        [
            '--dry-run',
            '--unattended',
            '--dir',
            'inst',
            '--project',
            PROJECT,
            '--version',
            '../../other/repo/main',
            '--tiers',
            'core',
        ]
    )
    assert result.returncode == 2
    assert shimmed.log_lines('curl') == []


def test_branch_mode_uses_the_full_sha_and_a_sha_image_tag(
    shimmed: Shimmed, tmp_path: Path
) -> None:
    sha = 'ab' * 20
    _release_copy(shimmed, tmp_path)
    shutil.copytree(shimmed.release / 'raw' / RELEASE, shimmed.release / 'raw' / sha)
    shimmed.flag('allow_mutations')
    result = shimmed.run(
        [
            '--no-start',
            '--unattended',
            '--dir',
            'inst',
            '--project',
            PROJECT,
            '--branch',
            'main',
            '--tiers',
            'core',
        ],
        SHIM_BRANCH_SHA=sha,
    )
    assert result.returncode == 0, result.stderr
    assert any(
        f'release.test/raw/{sha}/release-manifest.txt' in ln for ln in shimmed.log_lines('curl')
    )
    assert env_file(shimmed)['OP_IMAGE_TAG'] == f'sha-{sha[:12]}'
    assert 'TESTING install' in result.stderr


def test_branch_mode_refuses_an_unparseable_api_answer(shimmed: Shimmed) -> None:
    result = shimmed.run(
        [
            '--dry-run',
            '--unattended',
            '--dir',
            'inst',
            '--project',
            PROJECT,
            '--branch',
            'main',
            '--tiers',
            'core',
        ],
        SHIM_BRANCH_SHA='',
    )
    assert result.returncode != 0
    assert 'head commit' in result.stderr
