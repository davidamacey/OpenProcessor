"""``op.periodic.job_stats``: the queue gauges come from the database and are never missing.

Two queued jobs, one 20 minutes old, give ``op_jobs{status="queued"} 2`` and an oldest age of at
least 1,200 s; an empty table gives zeros for every kind, not missing series.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

import pytest
from fixtures.job_rows import insert_row, make_project
from prometheus_client import REGISTRY

from src.config.job_kinds import KINDS
from src.services.periodic import job_stats
from src.services.periodic.runner import run_periodic


if TYPE_CHECKING:
    import uuid


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_env')]

KIND = 'open_vocab_drain'


def sample(name: str, **labels: str) -> float | None:
    return REGISTRY.get_sample_value(name, labels)


def insert(pg, project: uuid.UUID, status: str, *, queued_ago: dt.timedelta = dt.timedelta(0)):
    columns: dict[str, Any] = {'queued_at': dt.datetime.now(dt.UTC) - queued_ago}
    if status == 'failed':
        columns |= {
            'error_code': 'x',
            'error_message': 'x',
            'finished_at': dt.datetime.now(dt.UTC) - dt.timedelta(hours=1),
        }
    insert_row(pg, project, status, kind=KIND, **columns)


def test_an_empty_table_sets_zero_for_every_registered_kind(pg) -> None:
    run_periodic('job_stats', job_stats.tick())
    for kind in KINDS:
        assert sample('op_jobs', kind=kind, status='queued') == 0
        assert sample('op_jobs_oldest_queued_seconds', kind=kind) == 0
        assert sample('op_jobs_failed_24h', kind=kind) == 0


def test_queued_jobs_and_their_oldest_age_are_read_from_the_table(pg) -> None:
    project = make_project(pg)
    insert(pg, project, 'queued', queued_ago=dt.timedelta(minutes=20))
    insert(pg, project, 'queued')
    insert(pg, project, 'failed')

    run_periodic('job_stats', job_stats.tick())

    assert sample('op_jobs', kind=KIND, status='queued') == 2
    assert sample('op_jobs_oldest_queued_seconds', kind=KIND) >= 1200
    assert sample('op_jobs_failed_24h', kind=KIND) == 1
