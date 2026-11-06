"""`openprocessor status` lists Postgres, Valkey and the Celery services of the control plane."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import test_cli_project
from test_cli_project import cli


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed

inst = test_cli_project.inst

ALL = 'opensearch\ntriton-server\napi\npostgres\nvalkey\ndb-migrate\njobs-worker\nperiodic-worker\nopen-vocab-worker\njobs-beat\n'


def _plain(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def test_status_reports_each_new_service(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('services', ALL)
    shimmed.flag(
        'compose_ps',
        'postgres|running|healthy|0\n'
        'valkey|running|healthy|0\n'
        'db-migrate|exited||0\n'
        'jobs-worker|running|unhealthy|0\n'
        'periodic-worker|running|starting|0\n'
        'jobs-beat|running||0\n',
    )
    done = cli(shimmed, inst, 'status')
    assert done.returncode == 0, done.stderr
    out = _plain(done.stdout)
    assert re.search(r'postgres:\s+healthy', out)
    assert re.search(r'valkey:\s+healthy', out)
    assert re.search(r'db-migrate:\s+completed', out)
    assert re.search(r'jobs-worker:\s+unhealthy', out)
    assert re.search(r'periodic-worker:\s+starting', out)
    assert re.search(r'jobs-beat:\s+running', out)
    assert re.search(r'open-vocab-worker:\s+not created', out)
    assert re.search(r'flower:\s+not part of this install', out)


def test_a_failed_migration_is_not_reported_as_completed(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('services', ALL)
    shimmed.flag('compose_ps', 'db-migrate|exited||1\n')
    out = _plain(cli(shimmed, inst, 'status').stdout)
    assert re.search(r'db-migrate:\s+exited', out)
    assert not re.search(r'db-migrate:\s+completed', out)
