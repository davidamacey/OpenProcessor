"""`openprocessor db backup|restore` against the docker shim (pg_dump -Fc, pg_restore, keep-N)."""

from __future__ import annotations

import stat
from typing import TYPE_CHECKING

import test_cli_project
from installer_harness import PROJECT, RELEASE, REPO_ROOT
from test_cli_project import cli


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed

inst = test_cli_project.inst

DUMP = 'compose -p opinst-test'


def _docker_calls(shimmed: Shimmed) -> list[str]:
    return [ln for ln in shimmed.log.read_text().splitlines() if ln.startswith('docker compose')]


def _dumps(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.glob('openprocessor-*.dump'))


def test_backup_runs_pg_dump_custom_format_into_the_default_directory(
    shimmed: Shimmed, inst: Path
) -> None:
    shimmed.flag('postgres_container')
    done = cli(shimmed, inst, 'db', 'backup')
    assert done.returncode == 0, done.stderr + done.stdout
    calls = ' '.join(_docker_calls(shimmed))
    assert 'exec -T postgres pg_dump -U op_owner -d openprocessor -Fc' in calls
    files = _dumps(inst / 'backups' / 'postgres')
    assert len(files) == 1
    path = inst / 'backups' / 'postgres' / files[0]
    assert path.read_text().startswith('PGDMP-shim')
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(path.parent.glob('*.part'))


def test_backup_keeps_only_the_newest_n(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('postgres_container')
    out = inst / 'backups' / 'postgres'
    out.mkdir(parents=True)
    for day in range(1, 10):
        (out / f'openprocessor-2026010{day}T000000Z.dump').write_text('old')
    (out / 'notes.txt').write_text('not a dump')
    with (inst / '.env').open('a') as env:
        env.write('OP_DB_BACKUP_KEEP=3\n')
    done = cli(shimmed, inst, 'db', 'backup')
    assert done.returncode == 0, done.stderr + done.stdout
    kept = _dumps(out)
    assert len(kept) == 3
    assert 'openprocessor-20260109T000000Z.dump' in kept
    assert 'openprocessor-20260101T000000Z.dump' not in kept
    assert (out / 'notes.txt').exists()


def test_backup_directory_comes_from_out_then_env(
    shimmed: Shimmed, inst: Path, tmp_path: Path
) -> None:
    shimmed.flag('postgres_container')
    other = tmp_path / 'other-disk'
    with (inst / '.env').open('a') as env:
        env.write(f'OP_DB_BACKUP_DIR={other}\n')
    assert cli(shimmed, inst, 'db', 'backup').returncode == 0
    assert len(_dumps(other)) == 1
    explicit = tmp_path / 'explicit'
    assert cli(shimmed, inst, 'db', 'backup', '--out', str(explicit)).returncode == 0
    assert len(_dumps(explicit)) == 1
    assert len(_dumps(other)) == 1


def test_backup_without_a_running_postgres_writes_nothing(shimmed: Shimmed, inst: Path) -> None:
    done = cli(shimmed, inst, 'db', 'backup')
    assert done.returncode != 0
    assert 'postgres is not running' in done.stderr + done.stdout
    assert not (inst / 'backups' / 'postgres').exists()
    assert not any('pg_dump' in c for c in _docker_calls(shimmed))


def test_a_bad_keep_value_is_refused(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('postgres_container')
    with (inst / '.env').open('a') as env:
        env.write('OP_DB_BACKUP_KEEP=0\n')
    done = cli(shimmed, inst, 'db', 'backup')
    assert done.returncode != 0
    assert 'OP_DB_BACKUP_KEEP' in done.stderr + done.stdout


def test_restore_runs_the_documented_sequence(shimmed: Shimmed, inst: Path, tmp_path: Path) -> None:
    shimmed.flag('allow_mutations')
    shimmed.flag('services', 'opensearch\ntriton-server\napi\njobs-worker\npostgres\n')
    dump = tmp_path / 'x.dump'
    dump.write_text('PGDMP-test')
    done = cli(shimmed, inst, 'db', 'restore', str(dump), '--yes')
    assert done.returncode == 0, done.stderr + done.stdout
    calls = _docker_calls(shimmed)
    marks = [
        next(i for i, c in enumerate(calls) if needle in c)
        for needle in (
            ' stop api jobs-worker',
            ' up -d --wait postgres',
            ' run --rm --no-deps db-migrate',
            'pg_restore -U op_super --clean --if-exists --no-owner --no-privileges --role=op_owner -d openprocessor',
            'psql -U op_super -d openprocessor -v ON_ERROR_STOP=1 -c TRUNCATE events',
        )
    ]
    assert marks == sorted(marks)
    assert sum('db-migrate' in c for c in calls) == 2
    assert calls[-1].endswith(' up -d')


def test_restore_without_yes_and_without_a_terminal_changes_nothing(
    shimmed: Shimmed, inst: Path, tmp_path: Path
) -> None:
    dump = tmp_path / 'x.dump'
    dump.write_text('PGDMP-test')
    done = cli(shimmed, inst, 'db', 'restore', str(dump))
    assert done.returncode != 0
    assert not (shimmed.state / 'mutations.log').exists()


def test_restore_refuses_a_missing_file(shimmed: Shimmed, inst: Path) -> None:
    done = cli(shimmed, inst, 'db', 'restore', '/nonexistent.dump', '--yes')
    assert done.returncode != 0
    assert not (shimmed.state / 'mutations.log').exists()


def test_the_backup_settings_are_read_by_the_db_command() -> None:
    text = (REPO_ROOT / 'scripts' / 'lib' / 'cli_db.sh').read_text()
    assert 'OP_DB_BACKUP_DIR' in text
    assert 'OP_DB_BACKUP_KEEP' in text


def test_the_db_lib_ships_in_the_release() -> None:
    assert 'scripts/lib/cli_db.sh' in (REPO_ROOT / 'release-manifest.txt').read_text().split()


def test_a_rerun_of_the_installer_dumps_a_running_postgres_first(
    shimmed: Shimmed, inst: Path
) -> None:
    shimmed.flag('allow_mutations')
    shimmed.flag('postgres_container')
    done = shimmed.run(
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
            'core',
        ]
    )
    assert done.returncode == 0, done.stderr
    assert len(_dumps(inst / 'backups' / 'postgres')) == 1
    calls = ' '.join(_docker_calls(shimmed))
    assert 'pg_dump -U op_owner -d openprocessor -Fc' in calls


def test_an_install_without_postgres_takes_no_dump(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    done = shimmed.run(
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
            'core',
        ]
    )
    assert done.returncode == 0, done.stderr
    assert not (inst / 'backups' / 'postgres').exists()
