"""The installer moves the 0.6.0 control data for the user: an upgrade of a recorded 0.6.0 install
runs db-migrate, the upgrade step, then db-migrate again before it starts anything; a fresh install
and a same-version re-run never run it; a step that stopped is owed again on the next run; and a
rollback restores the control data (and only then abandons the PostgreSQL data) before the older
files come back."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import test_cli_project
from installer_harness import PROJECT
from test_cli_project import cli
from test_ownership_and_rerun import install


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed

inst = test_cli_project.inst

UPGRADE = 'scripts.db.upgrade_from_060'
ROLLBACK = 'scripts.db.rollback_to_060'


def _calls(shimmed: Shimmed) -> list[str]:
    return shimmed.mutating_docker_calls()


def _reset(shimmed: Shimmed) -> None:
    (shimmed.state / 'mutations.log').unlink(missing_ok=True)


def _index(calls: list[str], *needles: str) -> int:
    for i, ln in enumerate(calls):
        if all(n in ln for n in needles):
            return i
    raise AssertionError(f'no call with {needles}: {calls}')


def _indexes(calls: list[str], *needles: str) -> list[int]:
    return [i for i, ln in enumerate(calls) if all(n in ln for n in needles)]


def _pretend_060(shimmed: Shimmed) -> None:
    state = shimmed.root / 'inst' / '.install' / 'state.json'
    state.write_text(re.sub(r'"version": "[^"]*"', '"version": "v0.6.0"', state.read_text()))


def _final_up(calls: list[str]) -> int:
    return [
        i
        for i, ln in enumerate(calls)
        if ln.endswith(' up -d --remove-orphans') and f'-p {PROJECT} ' in ln
    ][-1]


def test_a_fresh_install_never_runs_the_step(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    assert not [ln for ln in _calls(shimmed) if UPGRADE in ln]


def test_a_same_version_rerun_runs_no_upgrade(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    _reset(shimmed)
    assert install(shimmed).returncode == 0
    assert not [ln for ln in _calls(shimmed) if UPGRADE in ln]


def test_an_upgrade_runs_migrate_then_the_step_then_migrate_again_before_the_stack_starts(
    shimmed: Shimmed,
) -> None:
    assert install(shimmed).returncode == 0
    _pretend_060(shimmed)
    _reset(shimmed)

    result = install(shimmed)

    assert result.returncode == 0, result.stderr[-2000:]
    calls = _calls(shimmed)
    migrates = _indexes(calls, ' run ', 'db-migrate')
    step = _index(calls, UPGRADE)
    assert len(migrates) == 2
    assert migrates[0] < step < migrates[1] < _final_up(calls)
    assert _index(calls, ' stop') < migrates[0]  # the 0.6.0 services are stopped first
    assert _index(calls, ' up -d --wait postgres valkey opensearch') < migrates[0]
    assert '-v ' in calls[step]
    assert '/install' in calls[step]  # the bundle lands in the install directory


def test_a_failed_step_stops_before_the_stack_starts_names_the_report_and_is_owed_again(
    shimmed: Shimmed,
) -> None:
    assert install(shimmed).returncode == 0
    _pretend_060(shimmed)
    shimmed.flag('upgrade_fail')
    _reset(shimmed)

    failed = install(shimmed)

    assert failed.returncode == 8, failed.stderr[-2000:]
    assert 'upgrade_060' in failed.stderr
    assert 'nothing of the 0.6.0 control data was deleted' in failed.stderr
    assert 'is installed and healthy' not in failed.stdout
    calls = _calls(shimmed)
    assert len(_indexes(calls, ' run ', 'db-migrate')) == 1  # no second migrate after a failure
    assert not [ln for ln in calls if ln.endswith(' up -d --remove-orphans')]

    # the recorded version is already the new one, but the marker keeps the step owed
    (shimmed.state / 'upgrade_fail').unlink()
    _reset(shimmed)
    again = install(shimmed)
    assert again.returncode == 0, again.stderr[-2000:]
    assert [ln for ln in _calls(shimmed) if UPGRADE in ln]
    assert not (shimmed.root / 'inst' / '.install' / 'upgrade_060.pending').exists()


def test_an_upgrade_with_no_start_is_refused_because_the_step_needs_the_stack(
    shimmed: Shimmed,
) -> None:
    assert install(shimmed).returncode == 0
    _pretend_060(shimmed)
    result = install(shimmed, '--no-start')
    assert result.returncode != 0
    assert 'without --no-start' in result.stderr


def _two_installs_with_a_bundle(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    assert install(shimmed, '--port-base', '4970').returncode == 0
    bundle = shimmed.root / 'inst' / '.install' / 'upgrade-0.6.0-20260201T000000Z'
    bundle.mkdir(parents=True)
    (bundle / 'MANIFEST.json').write_text('{}')
    _reset(shimmed)


def test_rollback_restores_the_control_data_then_abandons_postgres_then_the_old_files(
    shimmed: Shimmed,
) -> None:
    _two_installs_with_a_bundle(shimmed)

    result = shimmed.run(['--rollback', '--unattended', '--dir', 'inst'])

    assert result.returncode == 0, result.stderr[-2000:]
    calls = _calls(shimmed)
    restore = _index(calls, ROLLBACK, '--bundle /install/upgrade-0.6.0-20260201T000000Z')
    abandon = _index(calls, 'DROP DATABASE')
    assert _index(calls, ' up -d --wait opensearch') < restore < abandon < _final_up(calls)
    assert 'CREATE DATABASE openprocessor' in calls[abandon]


def test_a_restore_that_does_not_match_the_bundle_restores_nothing(shimmed: Shimmed) -> None:
    _two_installs_with_a_bundle(shimmed)
    shimmed.flag('rollback_fail')

    result = shimmed.run(['--rollback', '--unattended', '--dir', 'inst'])

    assert result.returncode != 0
    calls = _calls(shimmed)
    assert not [ln for ln in calls if 'DROP DATABASE' in ln]
    assert not [ln for ln in calls if ln.endswith(' up -d --remove-orphans')]


def test_a_rollback_with_no_bundle_has_no_control_data_step(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    assert install(shimmed, '--port-base', '4970').returncode == 0
    _reset(shimmed)
    result = shimmed.run(['--rollback', '--unattended', '--dir', 'inst'])
    assert result.returncode == 0, result.stderr[-2000:]
    assert not [ln for ln in _calls(shimmed) if ROLLBACK in ln]


def test_db_verify_runs_the_report_in_the_api_image(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    done = cli(shimmed, inst, 'db', 'verify')
    assert done.returncode == 0, done.stderr + done.stdout
    assert [ln for ln in _calls(shimmed) if ' run ' in ln and 'scripts.db.verify' in ln]
