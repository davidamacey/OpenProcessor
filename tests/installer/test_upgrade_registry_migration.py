"""The installer moves the class registry for the user: an upgrade or repair of an existing install
runs the migration after the stack is healthy (dry-run report first), a fresh install runs none,
a failed migration stops with the report path and exit 8, and a rollback writes the registry back
to class_registry.json before it restores any file."""

from __future__ import annotations

from typing import TYPE_CHECKING

from installer_harness import PROJECT, RELEASE
from test_ownership_and_rerun import install


if TYPE_CHECKING:
    from installer_harness import Shimmed

MODULE = 'scripts.curation.migrate_class_registry'


def _migrations(shimmed: Shimmed) -> list[str]:
    return [ln for ln in shimmed.mutating_docker_calls() if MODULE in ln]


def _reset(shimmed: Shimmed) -> None:
    (shimmed.state / 'mutations.log').unlink(missing_ok=True)


def test_a_fresh_install_runs_no_migration(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    assert _migrations(shimmed) == []


def test_a_rerun_migrates_after_health_dry_run_first_and_keeps_the_reports(
    shimmed: Shimmed,
) -> None:
    assert install(shimmed).returncode == 0
    _reset(shimmed)

    again = install(shimmed)

    assert again.returncode == 0, again.stderr[-2000:]
    runs = _migrations(shimmed)
    assert len(runs) == 2, runs
    assert runs[0].rstrip().endswith('--all --dry-run')
    assert runs[1].rstrip().endswith('--all')
    reports = sorted((shimmed.root / 'inst' / '.install' / 'registry_migration').iterdir())
    assert [p.name.split('-', 1)[1] for p in reports] == ['dry-run.json', 'run.json']


def test_repair_migrates_too(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    _reset(shimmed)
    repair = shimmed.run(['--repair', '--unattended', '--dir', 'inst'], OP_HEALTH_TIMEOUT='1')
    assert repair.returncode == 0, repair.stderr[-2000:]
    assert len(_migrations(shimmed)) == 2


def test_a_failed_migration_exits_8_names_the_report_and_never_claims_success(
    shimmed: Shimmed,
) -> None:
    assert install(shimmed).returncode == 0
    shimmed.flag('migrate_fail')

    again = install(shimmed)

    assert again.returncode == 8, again.stderr[-2000:]
    assert 'registry_migration' in again.stderr
    assert 'dry-run' in again.stderr
    assert 'is installed and healthy' not in again.stdout
    assert len(_migrations(shimmed)) >= 1


def test_rollback_writes_the_registry_back_before_it_restores_anything(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    assert install(shimmed, '--port-base', '4970').returncode == 0
    shimmed.flag('opensearch_container')
    _reset(shimmed)

    result = shimmed.run(['--rollback', '--unattended', '--dir', 'inst'])

    assert result.returncode == 0, result.stderr[-2000:]
    calls = shimmed.mutating_docker_calls()
    reverse = [
        i for i, ln in enumerate(calls) if MODULE in ln and ln.rstrip().endswith('--reverse')
    ]
    restart = [
        i
        for i, ln in enumerate(calls)
        if ln.endswith(' up -d --remove-orphans') and f'-p {PROJECT} ' in ln
    ]
    assert reverse
    assert restart
    assert reverse[0] < restart[0]
    assert '--all --reverse' in calls[reverse[0]]


def test_rollback_without_opensearch_says_how_to_finish_by_hand(shimmed: Shimmed) -> None:
    assert install(shimmed).returncode == 0
    assert install(shimmed, '--port-base', '4970').returncode == 0
    _reset(shimmed)

    result = shimmed.run(['--rollback', '--unattended', '--dir', 'inst'])

    assert result.returncode == 0, result.stderr[-2000:]
    assert _migrations(shimmed) == []
    assert 'migrate-registry --all --reverse' in result.stderr + result.stdout
    assert RELEASE
