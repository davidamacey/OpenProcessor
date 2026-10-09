"""Re-review items R1-R3 and the minor items r6-r15: the installer acts only on
directories and compose projects it created, a dry run writes nothing, a
re-run keeps what is installed, and several edge cases fail closed.
"""

from __future__ import annotations

import hashlib
import subprocess
import tarfile
from typing import TYPE_CHECKING

import pytest
from installer_harness import PROJECT, RELEASE, SCRIPT


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed


def install(
    shimmed: Shimmed, *extra: str, tiers: str | None = 'core', dirname: str = 'inst', **env: str
):
    shimmed.flag('allow_mutations')
    env.setdefault('OP_HEALTH_TIMEOUT', '1')
    args = ['--unattended', '--dir', dirname, '--project', PROJECT, '--version', RELEASE]
    if tiers is not None:
        args += ['--tiers', tiers]
    return shimmed.run([*args, *extra], **env)


def tree_digest(root: Path) -> str:
    """sha256 over every path, mode and file content under root."""
    h = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        st = p.lstat()
        h.update(f'{p.relative_to(root)}|{oct(st.st_mode)}'.encode())
        if p.is_file() and not p.is_symlink():
            h.update(p.read_bytes())
    return h.hexdigest()


def make_dev_checkout(path: Path, project: str) -> None:
    (path / '.git').mkdir(parents=True)
    (path / 'src').mkdir()
    (path / 'src' / 'main.py').write_text('app = None\n')
    (path / 'docker-compose.yml').write_text("services: {}  # the developer's compose\n")
    (path / '.env').write_text(f'COMPOSE_PROJECT_NAME={project}\nAPI_PORT=4603\n')


def env_value(path: Path, key: str) -> str:
    for line in path.read_text().splitlines():
        if line.startswith(f'{key}='):
            return line.split('=', 1)[1]
    return ''


def reset(shimmed: Shimmed) -> None:
    (shimmed.state / 'mutations.log').unlink(missing_ok=True)


# --- R1: never act on a project or dir the installer did not create ------------


def test_uninstall_refuses_a_dev_checkout(shimmed: Shimmed) -> None:
    checkout = shimmed.root / 'devstack'
    make_dev_checkout(checkout, 'devstack')
    shimmed.containers(
        [('devstack', str(checkout.resolve()), 'devstack-api', '127.0.0.1:4603->8000/tcp')]
    )
    shimmed.flag('allow_mutations')
    before = tree_digest(checkout)
    for extra in ([], ['--unattended'], ['--yes', '--purge-data']):
        result = shimmed.run(
            ['--uninstall', '--dir', 'devstack', *extra], OP_CONFIRM_PURGE='devstack'
        )
        assert result.returncode == 3, (extra, result.stderr)
        assert 'not an install made by this installer' in result.stderr
    assert shimmed.mutating_docker_calls() == []
    assert tree_digest(checkout) == before


def test_install_refuses_a_dev_checkout_at_the_default_dir(shimmed: Shimmed) -> None:
    checkout = shimmed.root / 'openprocessor'
    make_dev_checkout(checkout, 'openprocessor')
    shimmed.containers([('openprocessor', str(checkout.resolve()), 'openprocessor-api', '')])
    shimmed.flag('allow_mutations')
    before = tree_digest(checkout)
    for args in (['--unattended'], ['--unattended', '--dry-run'], ['--repair', '--unattended']):
        result = shimmed.run([*args, '--version', RELEASE])
        assert result.returncode == 3, (args, result.stderr)
        assert 'git checkout' in result.stderr or 'not an install made' in result.stderr
    assert shimmed.mutating_docker_calls() == []
    assert tree_digest(checkout) == before


def test_force_existing_dir_backs_up_before_adopting(shimmed: Shimmed) -> None:
    checkout = shimmed.root / 'devstack'
    make_dev_checkout(checkout, 'devstack')
    result = install(shimmed, '--no-start', '--force-existing-dir', dirname='devstack')
    assert result.returncode == 0, result.stderr[-2000:]
    backups = list((checkout / 'backups').glob('*-pre-install.tar.gz'))
    assert len(backups) == 1
    with tarfile.open(backups[0]) as tar:
        saved = tar.extractfile('./docker-compose.yml').read().decode()  # type: ignore[union-attr]
    assert "the developer's compose" in saved
    assert (checkout / 'src' / 'main.py').exists()


def test_a_missing_terminal_is_not_consent(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    reset(shimmed)
    for args in (['--uninstall'], ['--rollback']):
        result = shimmed.run([*args, '--dir', 'inst'])
        assert result.returncode in (9, 1), result.stderr
        assert shimmed.mutating_docker_calls() == []
    result = shimmed.run(['--uninstall', '--dir', 'inst'])
    assert result.returncode == 9
    assert '--yes' in result.stderr
    ok = shimmed.run(['--uninstall', '--yes', '--dir', 'inst'])
    assert ok.returncode == 0, ok.stderr
    assert [ln for ln in shimmed.mutating_docker_calls() if ln.endswith('down --remove-orphans')]


def test_state_recorded_for_another_dir_is_refused(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    subprocess.run(['cp', '-a', str(shimmed.root / 'inst'), str(shimmed.root / 'copy')], check=True)
    reset(shimmed)
    for args in (
        ['--uninstall', '--yes'],
        ['--repair', '--unattended'],
        ['--unattended', '--tiers', 'core'],
    ):
        result = shimmed.run(
            [*args, '--dir', 'copy', '--version', RELEASE]
            if '--uninstall' not in args
            else [*args, '--dir', 'copy']
        )
        assert result.returncode == 3, (args, result.stderr)
    assert shimmed.mutating_docker_calls() == []


def test_interrupted_install_can_be_resumed_by_the_installer(shimmed: Shimmed) -> None:
    shimmed.flag('pull_bad_digest')
    assert install(shimmed).returncode == 7
    (shimmed.state / 'pull_bad_digest').unlink()
    (shimmed.state / 'images').exists() and subprocess.run(
        ['rm', '-rf', str(shimmed.state / 'images')], check=True
    )
    assert install(shimmed).returncode == 0


# --- R2: a dry run writes nothing --------------------------------------------------


def test_dry_run_on_an_installed_dir_changes_no_byte(shimmed: Shimmed) -> None:
    assert install(shimmed, tiers='trainer').returncode == 0
    inst = shimmed.root / 'inst'
    (inst / 'docker-compose.yml').write_text(
        (inst / 'docker-compose.yml').read_text() + '# user edit\n'
    )
    before = tree_digest(shimmed.root / 'inst')
    tmp = shimmed.root / 'tmp'
    tmp.mkdir()
    reset(shimmed)
    result = shimmed.run(
        ['--dry-run', '--unattended', '--dir', 'inst', '--tiers', 'core'], TMPDIR=str(tmp)
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert tree_digest(shimmed.root / 'inst') == before
    assert list(tmp.iterdir()) == []
    assert shimmed.mutating_docker_calls() == []
    assert 'would change docker-compose.yml' in result.stdout
    assert 'would set    .env COMPOSE_PROFILES' in result.stdout


def test_dry_run_on_a_fresh_dir_creates_nothing(shimmed: Shimmed) -> None:
    tmp = shimmed.root / 'tmp'
    tmp.mkdir()
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
        ],
        TMPDIR=str(tmp),
    )
    assert result.returncode == 0, result.stderr
    assert not (shimmed.root / 'inst').exists()
    assert list(tmp.iterdir()) == []


# --- R3: a re-run keeps the installed tiers -----------------------------------------


def test_rerun_without_tiers_keeps_the_installed_tiers(shimmed: Shimmed) -> None:
    assert (
        install(shimmed, '--no-start', '--with-monitoring', tiers='trainer,cropwright').returncode
        == 0
    )
    inst = shimmed.root / 'inst'
    profiles = env_value(inst / '.env', 'COMPOSE_PROFILES')
    assert set(profiles.split(',')) == {'curation', 'training', 'cropwright', 'monitoring'}
    again = install(shimmed, '--no-start', tiers=None)
    assert again.returncode == 0, again.stderr[-2000:]
    assert 'keeping the installed tiers' in again.stdout
    assert env_value(inst / '.env', 'COMPOSE_PROFILES') == profiles
    state = (inst / '.install' / 'state.json').read_text()
    assert '"tiers": "core curation trainer cropwright"' in state
    assert '"with_monitoring": "1"' in state


# --- minor items -------------------------------------------------------------------


def test_container_without_a_working_dir_label_fails_closed(shimmed: Shimmed) -> None:
    shimmed.containers([(PROJECT, '', f'{PROJECT}-api', '')])
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
    assert result.returncode == 3
    assert 'no working_dir label' in result.stderr


def test_non_compose_container_with_our_name_is_refused(shimmed: Shimmed) -> None:
    shimmed.containers([('', '', f'{PROJECT}-api', '')])
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
    assert result.returncode == 3
    assert 'not created by Compose' in result.stderr


def test_source_only_env_cannot_silence_the_one_liner(shimmed: Shimmed) -> None:
    result = shimmed.run(['--help'], piped=True, OP_SOURCE_ONLY='1')
    assert result.returncode == 0
    assert 'Usage: setup-openprocessor.sh' in result.stdout


def test_bootstrap_env_is_ignored_outside_the_bootstrapped_child(shimmed: Shimmed) -> None:
    victim = shimmed.root / 'victim'
    victim.mkdir()
    (victim / 'keep').write_text('x')
    result = shimmed.run(
        ['--dry-run', '--unattended', '--dir', 'inst', '--project', PROJECT, '--tiers', 'core'],
        OP_BOOTSTRAP_TMP=str(victim),
        OP_BOOTSTRAP_REF='v0.0.1',
        OP_BOOTSTRAP_MODE='release',
    )
    assert result.returncode == 0, result.stderr
    assert (victim / 'keep').exists()
    assert f'release     : {RELEASE}' in result.stdout


def test_bootstrapped_env_does_not_skip_the_checksum_bootstrap(shimmed: Shimmed) -> None:
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
        ],
        piped=True,
        OP_BOOTSTRAPPED='1',
    )
    assert result.returncode == 0, result.stderr
    assert f'running the verified {RELEASE} installer' in result.stdout


def test_unattended_recommendation_drops_the_gated_segmenter_without_a_token(
    shimmed: Shimmed,
) -> None:
    result = install(shimmed, '--no-start', tiers=None)
    assert result.returncode == 0, result.stderr[-2000:]
    assert 'segmenter tier needs a HuggingFace token' in result.stderr
    assert 'segmenter' not in env_value(shimmed.root / 'inst' / '.env', 'COMPOSE_PROFILES')


def test_purge_of_container_owned_files_runs_in_a_container(
    shimmed: Shimmed, tmp_path: Path
) -> None:
    target = tmp_path / 'inst' / 'models'
    target.mkdir(parents=True)
    env_path = tmp_path / 'inst' / '.env'
    env_path.write_text('OP_API_IMAGE=davidamacey/openprocessor@sha256:' + 'a' * 64 + '\n')
    shimmed.flag('allow_mutations')
    result = subprocess.run(
        [
            'bash',
            '-c',
            f'OP_SOURCE_ONLY=1 source "{SCRIPT}"; OP_DRY_RUN=0; ENV_FILE="{env_path}"; '
            '_tree_has_foreign_files() { return 0; }; safe_rm_tree "$1"',
            '_',
            str(target),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=shimmed.env(),
    )
    assert result.returncode == 0, result.stderr
    runs = [ln for ln in shimmed.mutating_docker_calls() if ln.startswith('docker run')]
    assert runs == [
        f'docker run --rm --user 0 --entrypoint rm -v {target.parent}:/purge '
        f'davidamacey/openprocessor@sha256:{"a" * 64} -rf --one-file-system -- /purge/models'
    ]


@pytest.mark.parametrize('bind', ['010.0.0.1', '127.000.000.001', '1.2.3.04'])
def test_leading_zero_octets_are_rejected(shimmed: Shimmed, bind: str) -> None:
    assert shimmed.run(['--dry-run', '--bind', bind]).returncode == 2


def test_monitoring_is_refused_with_control_plane_only(shimmed: Shimmed) -> None:
    result = shimmed.run(['--dry-run', '--cpu', '--control-plane-only', '--with-monitoring'])
    assert result.returncode == 2
    assert 'not available with --control-plane-only' in result.stderr


def test_monitoring_health_verifies_the_loaded_config(shimmed: Shimmed) -> None:
    ok = install(shimmed, '--with-monitoring', '--skip-models')
    assert ok.returncode == 0, ok.stderr[-2000:]
    assert any('/api/v1/status/config' in ln for ln in shimmed.log_lines('curl'))
    shimmed.flag('monitoring_bad')
    bad = install(shimmed, '--with-monitoring', '--skip-models')
    assert bad.returncode == 8
    assert 'prometheus did not load monitoring/prometheus.yml' in bad.stderr


def test_nothing_env_driven_remains_in_the_bootstrap_guard() -> None:
    text = SCRIPT.read_text()
    assert 'OP_BOOTSTRAPPED' not in text
    assert 'trap "rm -rf \'' not in text
    assert SCRIPT.name == 'setup-openprocessor.sh'


# --- #108: a re-run never silently swaps the installed VLM / counts itself as "other" ---


def test_rerun_never_silently_switches_the_installed_vlm(shimmed: Shimmed) -> None:
    first = install(shimmed, '--no-start', tiers='vlm')
    assert first.returncode == 0, first.stderr[-2000:]
    inst = shimmed.root / 'inst'
    installed = env_value(inst / '.env', 'VLM_CATALOG_ID')
    assert installed
    # 8 GB used: planning from scratch would fall back to the smaller qwen entry.
    shimmed.gpus('0, NVIDIA RTX A6000, 49140, 8000, 8.6\n')
    again = install(shimmed, '--no-start', tiers=None)
    assert 'keeping the installed VLM' in again.stdout
    assert env_value(inst / '.env', 'VLM_CATALOG_ID') == installed, again.stdout[-1500:]
    # the kept model is refused (not swapped) when it no longer fits
    assert again.returncode != 0
    assert f'{installed} needs' in again.stderr


def test_rerun_subtracts_its_own_vram_so_the_installed_vlm_still_fits(shimmed: Shimmed) -> None:
    assert install(shimmed, '--no-start', tiers='vlm').returncode == 0
    inst = shimmed.root / 'inst'
    installed = env_value(inst / '.env', 'VLM_CATALOG_ID')
    shimmed.containers([(PROJECT, str(inst), f'{PROJECT}-vlm', '')])
    shimmed.flag('own_pids', '4242\n')
    shimmed.flag('gpu_uuids.csv', '0, GPU-aaaa\n')
    # 31 GB used, 26 GB of it by this project's own VLM process; pid 999 is someone else's.
    shimmed.flag('compute_apps.csv', '4242, GPU-aaaa, 26000\n999, GPU-aaaa, 5000\n')
    shimmed.gpus('0, NVIDIA RTX A6000, 49140, 31000, 8.6\n')
    again = install(shimmed, '--no-start', tiers=None)
    assert again.returncode == 0, again.stderr[-1500:]
    assert env_value(inst / '.env', 'VLM_CATALOG_ID') == installed


def test_rerun_without_gpu_plan_keeps_the_installed_placement(shimmed: Shimmed) -> None:
    two = '0, NVIDIA RTX A6000, 49140, 0, 8.6\n1, NVIDIA RTX A6000, 49140, 0, 8.6\n'
    shimmed.gpus(two)
    first = install(shimmed, '--no-start', '--gpu-plan', 'triton=1', tiers='core')
    assert first.returncode == 0, first.stderr[-1500:]
    inst = shimmed.root / 'inst'
    assert env_value(inst / '.env', 'TRITON_GPU_ID') == '1'
    again = install(shimmed, '--no-start', tiers=None)
    assert again.returncode == 0, again.stderr[-1500:]
    assert 'keeping the installed GPU placement' in again.stdout
    assert env_value(inst / '.env', 'TRITON_GPU_ID') == '1'


def test_rerun_without_the_segmenter_tier_keeps_its_gpu_assignment(shimmed: Shimmed) -> None:
    assert install(shimmed, '--no-start').returncode == 0
    inst = shimmed.root / 'inst'
    env = inst / '.env'
    kept = [ln for ln in env.read_text().splitlines() if not ln.startswith('SEGMENTER_GPU_ID=')]
    env.write_text('\n'.join([*kept, 'SEGMENTER_GPU_ID=7']) + '\n')
    shimmed.gpus('0, NVIDIA RTX A6000, 49140, 0, 8.6\n1, NVIDIA RTX A6000, 49140, 0, 8.6\n')
    again = install(shimmed, '--no-start', '--gpu-plan', 'triton=0', tiers=None)
    assert again.returncode == 0, again.stderr[-1500:]
    assert env_value(inst / '.env', 'SEGMENTER_GPU_ID') == '7'


def test_gpu_subtract_own_removes_the_installs_own_vram() -> None:
    out = subprocess.run(
        [
            'bash',
            '-c',
            f'OP_SOURCE_ONLY=1 source "{SCRIPT}"; '
            'gpu_subtract_own "$(printf "0 49140 1000\\n2 49140 22528\\n")" "$(printf "2 22000\\n")"',
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.stdout.split('\n')[:2] == ['0 49140 1000', '2 49140 528']


def test_fresh_install_sets_a_working_primary_ingest_detector(shimmed: Shimmed) -> None:
    assert install(shimmed, '--no-start').returncode == 0
    inst = shimmed.root / 'inst'
    assert (
        env_value(inst / '.env', 'OP_INGEST_PRIMARY_DETECTOR_MODEL') == 'yolov11_small_trt_end2end'
    )


def test_rerun_from_inside_the_install_dir_needs_no_dir_flag(shimmed: Shimmed) -> None:
    assert install(shimmed, '--no-start').returncode == 0
    inst = shimmed.root / 'inst'
    shimmed.flag('allow_mutations')
    result = shimmed.run(
        ['--unattended', '--project', PROJECT, '--version', RELEASE, '--no-start'],
        cwd=inst,
        OP_HEALTH_TIMEOUT='1',
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert 'keeping the installed tiers' in result.stdout
