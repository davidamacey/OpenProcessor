"""Startup reconcile is a lease takeover: a sibling's live job is left alone, a dead holder's
(or a pre-lease job's) stale record is repaired.

The jobs run against the in-memory lease store of ``tests/conftest.py``.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import pytest

from src.clients import job_lease
from src.domain.job_lease import Acquired, LeaseKind
from src.services.curation import startup_reconcile
from src.services.curation.file_job import HEARTBEAT_STALE_S, FileJob
from src.services.curation.item_scores import job as scores


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FileJob:
    job = FileJob(tmp_path / 'scores')
    monkeypatch.setattr(scores, '_job', lambda: job)
    monkeypatch.setattr(
        startup_reconcile,
        'JOB_MODULES',
        (('src.services.curation.item_scores.job', LeaseKind.ITEM_SCORES),),
    )
    return job


def _age_heartbeat(job: FileJob, seconds: float) -> None:
    job.touch_heartbeat()
    old = time.time() - seconds
    os.utime(job.heartbeat_file, (old, old))


@pytest.mark.asyncio
async def test_a_running_record_with_no_lease_and_a_stale_heartbeat_is_repaired(
    job: FileJob,
) -> None:
    job.write({'job_id': 'j1', 'status': 'running'})
    _age_heartbeat(job, HEARTBEAT_STALE_S + 60)

    await startup_reconcile.reconcile_all_jobs(object())

    assert job.read()['status'] == 'interrupted'


@pytest.mark.asyncio
async def test_a_running_record_with_no_lease_and_a_fresh_heartbeat_is_left_busy(
    job: FileJob,
) -> None:
    job.write({'job_id': 'j1', 'status': 'running'})
    _age_heartbeat(job, 5)

    await startup_reconcile.reconcile_all_jobs(object())

    assert job.read()['status'] == 'running'


@pytest.mark.asyncio
async def test_a_siblings_fresh_job_with_a_live_lease_and_no_heartbeat_yet_is_left_alone(
    job: FileJob,
) -> None:
    held = await job_lease.acquire(object(), LeaseKind.ITEM_SCORES, None, 'j1')
    assert isinstance(held, Acquired)
    job.write({'job_id': 'j1', 'status': 'running'})  # no heartbeat file yet

    await startup_reconcile.reconcile_all_jobs(object())

    assert job.read()['status'] == 'running'
    assert (await job_lease.renew(object(), held.lease)).__class__.__name__ == 'Renewed'


@pytest.mark.asyncio
async def test_an_expired_lease_is_taken_over_and_its_record_repaired(
    job: FileJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = time.time
    monkeypatch.setattr(job_lease.time, 'time', lambda: real() - 1000)
    held = await job_lease.acquire(object(), LeaseKind.ITEM_SCORES, None, 'j1')
    monkeypatch.setattr(job_lease.time, 'time', real)
    assert isinstance(held, Acquired)
    job.write({'job_id': 'j1', 'status': 'running'})

    await startup_reconcile.reconcile_all_jobs(object())

    assert job.read()['status'] == 'interrupted'
    again = await job_lease.acquire(object(), LeaseKind.ITEM_SCORES, None, 'j2')
    assert isinstance(again, Acquired)  # the reconcile released what it took
    assert again.lease.generation == 3
