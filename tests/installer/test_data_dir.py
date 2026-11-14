"""OP_DATA_DIR: the database, search and queue data on a chosen directory (docker-compose.data-dir.yml),
the refusal to start empty over a volume that holds data, and the rotating-disk warning."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest
import test_cli_project
from installer_harness import PROJECT, REPO_ROOT, run_bash
from test_cli_project import cli


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed

inst = test_cli_project.inst
OVERLAY = 'docker-compose.data-dir.yml'
LIB = REPO_ROOT / 'scripts' / 'lib' / 'data_dir.sh'
VOLUMES = ('openprocessor-pg', 'opensearch_data', 'openprocessor-queue')


def _set_data_dir(inst: Path, value: str) -> None:
    with (inst / '.env').open('a') as fh:
        fh.write(f'OP_DATA_DIR={value}\n')


def _up_lines(shimmed: Shimmed) -> list[str]:
    return [ln for ln in shimmed.mutating_docker_calls() if ln.endswith(' up -d --remove-orphans')]


def _volume_with_data(shimmed: Shimmed) -> None:
    (shimmed.state / 'volumes.txt').write_text(''.join(f'{PROJECT}_{v}\n' for v in VOLUMES))
    (shimmed.state / 'volume_data').mkdir()
    for v in VOLUMES:
        (shimmed.state / 'volume_data' / f'{PROJECT}_{v}').write_text('base\n')


def test_the_overlay_is_added_exactly_when_op_data_dir_is_set(
    shimmed: Shimmed, inst: Path, tmp_path: Path
) -> None:
    shimmed.flag('allow_mutations')
    assert cli(shimmed, inst, 'stop').returncode == 0
    assert all(OVERLAY not in ln for ln in shimmed.mutating_docker_calls())
    _set_data_dir(inst, str(tmp_path / 'fast'))
    assert cli(shimmed, inst, 'stop').returncode == 0
    downs = [ln for ln in shimmed.mutating_docker_calls() if ln.endswith(' down')]
    assert f'-f {inst}/{OVERLAY}' in downs[-1]
    assert downs[-1].index('docker-compose.yml') < downs[-1].index(OVERLAY)


def test_start_creates_the_three_directories_and_uses_the_overlay(
    shimmed: Shimmed, inst: Path, tmp_path: Path
) -> None:
    shimmed.flag('allow_mutations')
    fast = tmp_path / 'fast'
    _set_data_dir(inst, str(fast))
    done = cli(shimmed, inst, 'start')
    assert done.returncode == 0, done.stderr
    for sub in ('postgres', 'opensearch', 'valkey'):
        assert (fast / sub).is_dir()
    assert OVERLAY in _up_lines(shimmed)[0]


def test_start_refuses_an_empty_directory_over_volumes_that_hold_data(
    shimmed: Shimmed, inst: Path, tmp_path: Path
) -> None:
    shimmed.flag('allow_mutations')
    _volume_with_data(shimmed)
    _set_data_dir(inst, str(tmp_path / 'fast'))
    done = cli(shimmed, inst, 'start')
    assert done.returncode != 0
    assert 'would start with empty databases' in done.stderr + done.stdout
    assert f'{PROJECT}_openprocessor-pg' in done.stderr + done.stdout
    assert 'operations/storage' in done.stderr + done.stdout
    assert _up_lines(shimmed) == []


def test_start_proceeds_once_the_directory_holds_the_copied_data(
    shimmed: Shimmed, inst: Path, tmp_path: Path
) -> None:
    shimmed.flag('allow_mutations')
    _volume_with_data(shimmed)
    fast = tmp_path / 'fast'
    for sub in ('postgres', 'opensearch', 'valkey'):
        (fast / sub).mkdir(parents=True)
        (fast / sub / 'PG_VERSION').write_text('x')
    _set_data_dir(inst, str(fast))
    done = cli(shimmed, inst, 'start')
    assert done.returncode == 0, done.stderr
    assert len(_up_lines(shimmed)) == 1


def test_a_relative_data_dir_is_refused(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    _set_data_dir(inst, 'fast')
    done = cli(shimmed, inst, 'start')
    assert done.returncode != 0
    assert 'not an absolute path' in done.stderr + done.stdout
    assert _up_lines(shimmed) == []


# --- rotational detection against a fake /sys ---------------------------------------------


def _blk(sys: Path, name: str, rota: str | None, slaves: tuple[str, ...] = ()) -> None:
    base = sys / 'class' / 'block' / name
    (base / 'queue').mkdir(parents=True)
    if rota is not None:
        (base / 'queue' / 'rotational').write_text(rota + '\n')
    (base / 'slaves').mkdir()
    for s in slaves:
        (base / 'slaves' / s).mkdir()


def _rotational(shimmed: Shimmed, sys: Path, source: str | None, path: Path) -> str:
    f = shimmed.state / 'findmnt_source'
    f.unlink(missing_ok=True)
    if source is not None:
        f.write_text(source + '\n')
    done = run_bash(
        f'source {LIB}; storage_rotational {path}',
        env=shimmed.env(SYS_ROOT=str(sys)),
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@pytest.mark.parametrize(
    ('source', 'devices', 'want'),
    [
        ('/dev/sda1', {'sda': '1'}, 'yes'),
        ('/dev/nvme0n1p2', {'nvme0n1': '0'}, 'no'),
        ('/dev/nvme0n1', {'nvme0n1': '0'}, 'no'),
        ('/dev/md0', {'md0': '0', 'sda': '0', 'sdb': '1'}, 'yes'),
        ('/dev/md0', {'md0': '0', 'sda': '0', 'sdb': '0'}, 'no'),
        ('/dev/sdz', {}, 'unknown'),
        ('/dev/sdz', {'sdz': None}, 'unknown'),
        ('tmpfs', {}, 'unknown'),
        (None, {}, 'unknown'),
    ],
    ids=[
        'disk',
        'nvme-partition',
        'nvme',
        'md-rotational-member',
        'md-ssd',
        'no-sys-entry',
        'no-flag',
        'not-a-device',
        'no-mount',
    ],
)
def test_rotational_detection(
    shimmed: Shimmed, tmp_path: Path, source: str | None, devices: dict, want: str
) -> None:
    sys = tmp_path / 'sys'
    members = {'md0': ('sda', 'sdb')}
    for name, rota in devices.items():
        _blk(sys, name, rota, members.get(name, ()) if len(devices) > 1 else ())
    if source == '/dev/sda1':
        part = sys / 'devices' / 'sda' / 'sda1'
        part.mkdir(parents=True)
        (part / 'partition').write_text('1\n')
        (sys / 'devices' / 'sda' / 'queue').mkdir()
        (sys / 'class' / 'block' / 'sda1').symlink_to(part)
    if source == '/dev/nvme0n1p2':
        part = sys / 'devices' / 'nvme0n1' / 'nvme0n1p2'
        part.mkdir(parents=True)
        (part / 'partition').write_text('2\n')
        (sys / 'class' / 'block' / 'nvme0n1p2').symlink_to(part)
    assert _rotational(shimmed, sys, source, tmp_path) == want


def test_a_device_mapper_stack_walks_to_its_rotational_member(
    shimmed: Shimmed, tmp_path: Path
) -> None:
    sys = tmp_path / 'sys'
    _blk(sys, 'dm-0', '0', ('md0',))
    _blk(sys, 'md0', '0', ('sda',))
    _blk(sys, 'sda', '1')
    assert _rotational(shimmed, sys, '/dev/dm-0', tmp_path) == 'yes'


def test_the_warning_is_two_plain_lines_and_never_a_failure(
    shimmed: Shimmed, tmp_path: Path
) -> None:
    sys = tmp_path / 'sys'
    _blk(sys, 'sda', '1')
    (shimmed.state / 'findmnt_source').write_text('/dev/sda\n')
    done = run_bash(
        f'log_warn() {{ echo "WARN $*"; }}; source {LIB}; storage_warn_if_rotational {tmp_path} "Docker data root"',
        env=shimmed.env(SYS_ROOT=str(sys)),
    )
    assert done.returncode == 0
    lines = done.stdout.strip().splitlines()
    assert len(lines) == 2
    assert 'rotating disk' in lines[0]
    assert 'several times slower' in lines[0]
    assert 'OP_DATA_DIR' in lines[1]


def test_an_unknown_device_says_it_could_not_determine(shimmed: Shimmed, tmp_path: Path) -> None:
    (shimmed.state / 'findmnt_source').write_text('tmpfs\n')
    done = run_bash(
        f'log_warn() {{ echo "WARN $*"; }}; source {LIB}; storage_warn_if_rotational {tmp_path}',
        env=shimmed.env(SYS_ROOT=str(tmp_path)),
    )
    assert 'could not determine' in done.stdout


def test_status_warns_when_the_data_directory_is_on_a_rotating_disk(
    shimmed: Shimmed, inst: Path, tmp_path: Path
) -> None:
    sys = tmp_path / 'sys'
    _blk(sys, 'sda', '1')
    (shimmed.state / 'findmnt_source').write_text('/dev/sda\n')
    _set_data_dir(inst, str(tmp_path / 'fast'))
    done = cli(shimmed, inst, 'status', SYS_ROOT=str(sys))
    assert 'rotating disk' in done.stdout + done.stderr


def test_overlay_file_resolves_the_three_mounts_and_leaves_the_base_alone(tmp_path: Path) -> None:
    import json
    import shutil

    docker = shutil.which('docker')
    if not docker:
        pytest.skip('docker CLI not installed')

    def render(*files: str) -> dict:
        cmd = [docker, 'compose', '-p', 'data-dir-test']
        for f in files:
            cmd += ['-f', str(REPO_ROOT / f)]
        done = subprocess.run(
            [*cmd, '--project-directory', str(REPO_ROOT), 'config', '--format', 'json'],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
            env={
                'PATH': '/usr/bin:/bin:/usr/local/bin',
                'HOME': str(tmp_path),
                'OP_DATA_DIR': '/srv/fast',
            },
        )
        if 'unknown command' in done.stderr or 'not a docker command' in done.stderr:
            pytest.skip('docker compose plugin unavailable')
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout)['services']

    base = render('docker-compose.yml')
    over = render('docker-compose.yml', OVERLAY)

    def mounts(svc: dict) -> dict[str, tuple[str, str]]:
        return {v['target']: (v['type'], v['source']) for v in svc['volumes']}

    for service, target, sub in (
        ('postgres', '/var/lib/postgresql', 'postgres'),
        ('valkey', '/data', 'valkey'),
        ('opensearch', '/usr/share/opensearch/data', 'opensearch'),
    ):
        assert mounts(over[service])[target] == ('bind', f'/srv/fast/{sub}')
        assert mounts(base[service])[target][0] == 'volume'
