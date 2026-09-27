"""Non-dry-run lifecycle through accepting shims: install, re-run, repair,
rollback, uninstall and purge. The docker shim accepts state-changing calls
here (allow_mutations) and records every one of them, so these tests check
what the installer actually tells Docker to do, in order.
"""

from __future__ import annotations

import shutil
import stat
from typing import TYPE_CHECKING

import pytest
from installer_harness import PROJECT, RELEASE


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed


PLAN_FILES = [
    'yolov11_small_trt_end2end/1/model.plan',
    'mobileclip2_s2_image_encoder/1/model.plan',
    'mobileclip2_s2_text_encoder/1/model.plan',
    'scrfd_10g_bnkps/1/model.plan',
    'arcface_w600k_r50/1/model.plan',
    'paddleocr_det_trt/1/model.plan',
    'paddleocr_rec_trt/1/model.plan',
]


def install(shimmed: Shimmed, *extra: str, tiers: str = 'core', **env: str):
    shimmed.flag('allow_mutations')
    env.setdefault('OP_HEALTH_TIMEOUT', '1')
    return shimmed.run(
        [
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


def reset_mutations(shimmed: Shimmed) -> None:
    (shimmed.state / 'mutations.log').unlink(missing_ok=True)


def index_of(lines: list[str], needle: str) -> int:
    for i, ln in enumerate(lines):
        if needle in ln:
            return i
    raise AssertionError(f'{needle!r} never happened:\n' + '\n'.join(lines))


def fake_engines(inst: Path) -> None:
    for rel in PLAN_FILES:
        p = inst / 'models' / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('engine')


def test_install_performs_every_step_in_order(shimmed: Shimmed) -> None:
    result = install(shimmed)
    assert result.returncode == 0, result.stderr[-3000:]
    assert f'OpenProcessor {RELEASE} is installed and healthy' in result.stdout
    m = shimmed.mutating_docker_calls()
    pull = index_of(m, 'docker pull davidamacey/openprocessor-triton@sha256:')
    probe = index_of(m, 'docker run --rm --gpus device=0 --entrypoint nvidia-smi')
    triton = index_of(m, ' up -d triton-server')
    preflight = index_of(m, 'python -m export.preflight')
    yolo = index_of(m, 'export/export_models.py')
    up = index_of(m, ' up -d --remove-orphans')
    probes = index_of(m, ' exec -T yolo-api python -c')
    assert pull < probe < triton < preflight < yolo < up < probes
    for line in m:
        if 'docker compose' in line:
            assert f' -p {PROJECT} ' in line, line
    state = (shimmed.root / 'inst' / '.install' / 'state.json').read_text()
    assert '"health": "ok"' in state
    assert '"yolo": {"status": "ok"' in state
    log = shimmed.root / 'inst' / '.install' / 'logs' / 'yolo_export.log'
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    assert stat.S_IMODE(log.parent.stat().st_mode) == 0o700


def test_digest_mismatch_after_pull_stops_before_anything_starts(shimmed: Shimmed) -> None:
    shimmed.flag('pull_bad_digest')
    result = install(shimmed)
    assert result.returncode == 7
    assert 'digest mismatch' in result.stderr
    assert not [ln for ln in shimmed.mutating_docker_calls() if ' up ' in f'{ln} ']


def test_unhealthy_stack_exits_8_and_never_claims_success(shimmed: Shimmed) -> None:
    shimmed.flag('health_down')
    result = install(shimmed, '--skip-models')
    assert result.returncode == 8
    assert 'installed and healthy' not in result.stdout
    assert 'health: ' in result.stderr


def test_wrong_vlm_model_served_fails_health(shimmed: Shimmed) -> None:
    tok = shimmed.root / 'tok'
    tok.write_text('hf_TESTSECRET1234567890\n')
    tok.chmod(0o600)
    result = install(shimmed, '--skip-models', tiers='vlm', SHIM_VLM_ROOT='someone/else')
    assert result.returncode == 8
    assert 'vlm not serving local-vlm' in result.stderr


def test_no_start_starts_nothing(shimmed: Shimmed) -> None:
    result = install(shimmed, '--no-start')
    assert result.returncode == 0, result.stderr
    assert not [
        ln
        for ln in shimmed.mutating_docker_calls()
        if ' up ' in f'{ln} ' or ' run --rm --no-deps' in ln
    ]
    assert 'Nothing was started' in result.stdout


def test_image_tag_mode_runs_local_images_without_pulling(shimmed: Shimmed) -> None:
    images = shimmed.state / 'images'
    images.mkdir()
    for name in ('openprocessor', 'openprocessor-triton'):
        (images / f'opinst_{name}_localtest').write_text(f'opinst/{name}@sha256:{"0" * 64}\n')
    result = install(shimmed, '--image-tag', 'localtest', OP_IMAGE_REPO='opinst')
    assert result.returncode == 0, result.stderr[-2000:]
    pulls = [ln for ln in shimmed.mutating_docker_calls() if ln.startswith('docker pull')]
    assert pulls == ['docker pull opensearchproject/opensearch:3.6.0']
    assert 'UNPINNED' in result.stderr


def test_rerun_skips_up_to_date_groups_and_repair_rebuilds_only_the_missing_one(
    shimmed: Shimmed,
) -> None:
    assert install(shimmed).returncode == 0
    inst = shimmed.root / 'inst'
    fake_engines(inst)
    reset_mutations(shimmed)
    again = install(shimmed)
    assert again.returncode == 0, again.stderr[-2000:]
    assert 'group yolo: up to date, skipped' in again.stdout
    assert not [ln for ln in shimmed.mutating_docker_calls() if 'export_models.py' in ln]

    (inst / 'models' / PLAN_FILES[0]).unlink()
    reset_mutations(shimmed)
    repair = shimmed.run(['--repair', '--unattended', '--dir', 'inst'], OP_HEALTH_TIMEOUT='1')
    assert repair.returncode == 0, repair.stderr[-2000:]
    m = shimmed.mutating_docker_calls()
    assert [ln for ln in m if 'export_models.py' in ln]
    assert not [ln for ln in m if 'export_mobileclip' in ln or 'export_scrfd' in ln]
    assert all(f' -p {PROJECT} ' in ln for ln in m if 'docker compose' in ln)


def test_repair_needs_an_existing_install(shimmed: Shimmed) -> None:
    result = shimmed.run(['--repair', '--unattended', '--dir', 'inst'])
    assert result.returncode != 0
    assert 'needs an existing install' in result.stderr


def test_rollback_restores_the_newest_backup(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    inst = shimmed.root / 'inst'
    env = inst / '.env'
    original = env.read_text()
    assert install(shimmed, '--port-base', '4970').returncode == 0
    assert 'API_PORT=4973' in env.read_text()
    reset_mutations(shimmed)
    result = shimmed.run(['--rollback', '--unattended', '--dir', 'inst'])
    assert result.returncode == 0, result.stderr[-2000:]
    assert env.read_text() == original
    assert [
        ln
        for ln in shimmed.mutating_docker_calls()
        if ln.endswith(' up -d --remove-orphans') and f'-p {PROJECT} ' in ln
    ]
    assert list((inst / '.install' / 'rollback-undo').iterdir())


def test_rollback_without_a_backup_fails(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    result = shimmed.run(['--rollback', '--unattended', '--dir', 'inst'])
    assert result.returncode != 0
    assert 'no backups/ entry' in result.stderr


# --- uninstall ----------------------------------------------------------------------


def test_uninstall_targets_the_installs_own_project(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    reset_mutations(shimmed)
    result = shimmed.run(['--uninstall', '--unattended', '--dir', 'inst'])
    assert result.returncode == 0, result.stderr
    m = shimmed.mutating_docker_calls()
    assert [ln for ln in m if ln.endswith('down --remove-orphans') and f' -p {PROJECT} ' in ln]
    assert not [ln for ln in m if ' -p openprocessor ' in ln]
    assert f'uninstalled project {PROJECT}' in result.stdout


def test_uninstall_project_flag_must_match(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    reset_mutations(shimmed)
    result = shimmed.run(
        ['--uninstall', '--unattended', '--dir', 'inst', '--project', 'openprocessor']
    )
    assert result.returncode == 2
    assert shimmed.mutating_docker_calls() == []


def test_uninstall_refuses_a_dir_that_is_not_an_install(shimmed: Shimmed) -> None:
    other = shimmed.root / 'other'
    other.mkdir()
    (other / '.env').write_text('API_PORT=4603\n')
    shimmed.flag('allow_mutations')
    result = shimmed.run(['--uninstall', '--unattended', '--dir', 'other'])
    assert result.returncode != 0
    assert shimmed.mutating_docker_calls() == []


def test_uninstall_refuses_a_foreign_owned_project(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    reset_mutations(shimmed)
    shimmed.containers([(PROJECT, '/srv/someone-else', f'{PROJECT}-api', '')])
    result = shimmed.run(['--uninstall', '--unattended', '--dir', 'inst'])
    assert result.returncode == 3
    assert shimmed.mutating_docker_calls() == []


def test_purge_needs_confirmation_when_unattended(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    reset_mutations(shimmed)
    result = shimmed.run(['--uninstall', '--purge-data', '--unattended', '--dir', 'inst'])
    assert result.returncode == 9
    assert f'OP_CONFIRM_PURGE={PROJECT}' in result.stderr
    assert shimmed.mutating_docker_calls() == []
    assert (shimmed.root / 'inst' / 'models').exists()


def test_purge_data_deletes_only_inside_the_install(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    inst = shimmed.root / 'inst'
    outside = shimmed.root / 'photos'
    outside.mkdir()
    (outside / 'keep.jpg').write_text('x')
    (inst / 'data' / 'd.txt').write_text('x')
    (inst / 'models' / 'm.txt').write_text('x')
    env = inst / '.env'
    env.write_text(
        env.read_text().replace('OP_SOURCE_ROOT_HOST=./data', 'OP_SOURCE_ROOT_HOST=../photos')
    )
    vols = f'{PROJECT}_opensearch_data\n'
    shimmed.flag('volumes.txt', vols)
    result = shimmed.run(
        ['--uninstall', '--purge-data', '--purge-volumes', '--unattended', '--dir', 'inst'],
        OP_CONFIRM_PURGE=PROJECT,
    )
    assert result.returncode == 0, result.stderr
    assert not (inst / 'data').exists()
    assert not (inst / 'models').exists()
    assert (outside / 'keep.jpg').exists()
    assert f'volumes to delete: {PROJECT}_opensearch_data' in result.stdout
    assert f'keeping OP_SOURCE_ROOT_HOST ({outside.resolve()})' in result.stdout + result.stderr
    assert [ln for ln in shimmed.mutating_docker_calls() if ln.endswith('down --remove-orphans -v')]
    assert (inst / '.env').exists()


def test_purge_never_follows_a_symlinked_data_dir(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    inst = shimmed.root / 'inst'
    library = shimmed.root / 'library'
    library.mkdir()
    (library / 'precious.jpg').write_text('x')
    (inst / 'data').rename(inst / 'data.orig')
    (inst / 'data').symlink_to(library)
    result = shimmed.run(
        ['--uninstall', '--purge-data', '--unattended', '--dir', 'inst'], OP_CONFIRM_PURGE=PROJECT
    )
    assert result.returncode == 0, result.stderr
    assert (library / 'precious.jpg').exists()
    assert 'it is a symlink' in result.stderr


def test_purge_refuses_without_state_json(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    inst = shimmed.root / 'inst'
    (inst / '.install' / 'state.json').unlink()
    result = shimmed.run(
        ['--uninstall', '--purge-data', '--unattended', '--dir', 'inst'], OP_CONFIRM_PURGE=PROJECT
    )
    assert result.returncode != 0
    assert 'not an installer-managed directory' in result.stderr
    assert (inst / 'models').exists()


@pytest.mark.parametrize('target', ['/', '~'])
def test_purge_refuses_root_and_home(shimmed: Shimmed, target: str) -> None:
    shimmed.flag('allow_mutations')
    d = '/' if target == '/' else str(shimmed.home)
    (shimmed.home / '.env').write_text(f'COMPOSE_PROJECT_NAME={PROJECT}\n')
    (shimmed.home / '.install').mkdir()
    (shimmed.home / '.install' / 'state.json').write_text('{}')
    result = shimmed.run(
        ['--uninstall', '--purge-data', '--unattended', '--dir', d], OP_CONFIRM_PURGE=PROJECT
    )
    assert result.returncode != 0
    assert shimmed.mutating_docker_calls() == []


def test_secrets_need_their_own_confirmation(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    inst = shimmed.root / 'inst'
    (inst / 'secrets' / 'vlm').mkdir(parents=True)
    (inst / 'secrets' / 'vlm' / 'k').write_text('x')
    result = shimmed.run(
        ['--uninstall', '--purge-data', '--unattended', '--dir', 'inst'], OP_CONFIRM_PURGE=PROJECT
    )
    assert result.returncode == 0
    assert (inst / 'secrets' / 'vlm' / 'k').exists()
    result = shimmed.run(
        ['--uninstall', '--purge-data', '--unattended', '--dir', 'inst'],
        OP_CONFIRM_PURGE=PROJECT,
        OP_CONFIRM_PURGE_SECRETS=PROJECT,
    )
    assert result.returncode == 0
    assert not (inst / 'secrets').exists()


def test_remove_images_only_removes_this_installs_lock_images(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    reset_mutations(shimmed)
    result = shimmed.run(['--uninstall', '--remove-images', '--unattended', '--dir', 'inst'])
    assert result.returncode == 0, result.stderr
    rmis = [ln for ln in shimmed.mutating_docker_calls() if ln.startswith('docker rmi')]
    assert rmis
    assert all('@sha256:' in ln and ('davidamacey/' in ln or 'vllm/' in ln) for ln in rmis)
    assert not [ln for ln in rmis if 'opensearch' in ln]


def test_dry_run_uninstall_removes_nothing(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    reset_mutations(shimmed)
    result = shimmed.run(
        ['--uninstall', '--purge-data', '--dry-run', '--unattended', '--dir', 'inst'],
        OP_CONFIRM_PURGE=PROJECT,
    )
    assert result.returncode == 0, result.stderr
    assert shimmed.mutating_docker_calls() == []
    assert (shimmed.root / 'inst' / 'models').exists()
    assert 'nothing was stopped or deleted' in result.stdout


def test_local_release_dir_and_local_image_tags_install_end_to_end(shimmed: Shimmed) -> None:
    # The pre-publication path: release assets built locally with
    # scripts/release/build_deploy_bundle.sh and images built under
    # local-only tags. Nothing is fetched from GitHub or Docker Hub.
    images = shimmed.state / 'images'
    images.mkdir()
    for name in ('openprocessor', 'openprocessor-triton'):
        (images / f'opinst_{name}_localtest').write_text(f'opinst/{name}@sha256:{"0" * 64}\n')
    (images / 'opensearchproject_opensearch_3.6.0').write_text(
        'opensearchproject/opensearch@sha256:' + '1' * 64 + '\n'
    )
    assets = shimmed.release / 'assets' / RELEASE
    result = install(
        shimmed,
        '--release-dir',
        str(assets),
        '--image-tag',
        'localtest',
        OP_IMAGE_REPO='opinst',
        OP_ARTIFACT_BASE_URL='https://unreachable.invalid/x',
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert [
        ln for ln in shimmed.log_lines('curl') if 'release.test' in ln or 'unreachable' in ln
    ] == []
    assert not [ln for ln in shimmed.mutating_docker_calls() if ln.startswith('docker pull')]
    assert 'installed and healthy' in result.stdout


def test_release_dir_is_still_checksum_verified(shimmed: Shimmed, tmp_path: Path) -> None:
    assets = tmp_path / 'assets'
    shutil.copytree(shimmed.release / 'assets' / RELEASE, assets)
    tb = assets / f'openprocessor-deploy-{RELEASE}.tar.gz'
    tb.write_bytes(tb.read_bytes()[:-100])
    result = install(shimmed, '--release-dir', str(assets))
    assert result.returncode == 7
    assert not (shimmed.root / 'inst' / '.env').exists()


def test_release_dir_needs_a_version(shimmed: Shimmed) -> None:
    result = shimmed.run(
        ['--dry-run', '--unattended', '--dir', 'inst', '--release-dir', str(shimmed.release)]
    )
    assert result.returncode == 2


def test_install_refuses_a_project_flag_that_contradicts_the_install(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    reset_mutations(shimmed)
    result = shimmed.run(
        [
            '--unattended',
            '--dir',
            'inst',
            '--project',
            'openprocessor',
            '--version',
            RELEASE,
            '--tiers',
            'core',
        ]
    )
    assert result.returncode == 2
    assert shimmed.mutating_docker_calls() == []
