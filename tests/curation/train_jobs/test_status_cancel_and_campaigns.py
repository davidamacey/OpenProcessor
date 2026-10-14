"""Reading status (stale heartbeat), cancel sentinels, run listing, campaigns and the run log tail."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from curation.train_jobs._helpers import _write_status
from src.services.training import jobs as train_jobs
from src.services.training.job_models import CampaignRunSpec, TrainCampaignSpec, TrainJobSpec


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_read_status_returns_none_for_unknown_job(jobs_dir: Path) -> None:
    result = await train_jobs.read_status('does_not_exist')
    assert result is None


@pytest.mark.asyncio
async def test_read_status_returns_queued_when_only_spec_exists(jobs_dir: Path) -> None:
    spec = TrainJobSpec(
        job_id='spec_only',
        dataset_export_dir='/data/exports/x',
    )
    await train_jobs.write_job(spec)
    result = await train_jobs.read_status('spec_only')
    assert result is not None
    assert result.state == 'queued'
    assert result.job_id == 'spec_only'


@pytest.mark.asyncio
async def test_read_status_marks_stale_heartbeat_as_lost(jobs_dir: Path) -> None:
    """heartbeat_at >60s old + state=running -> API view flips to 'lost'."""
    stale_ts = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    _write_status(jobs_dir, 'stale', state='running', heartbeat_at=stale_ts)
    result = await train_jobs.read_status('stale')
    assert result is not None
    assert result.state == 'lost', 'stale heartbeat should flip state to lost'


@pytest.mark.asyncio
async def test_read_status_keeps_fresh_heartbeat(jobs_dir: Path) -> None:
    fresh_ts = datetime.now(UTC).isoformat()
    _write_status(jobs_dir, 'fresh', state='running', heartbeat_at=fresh_ts)
    result = await train_jobs.read_status('fresh')
    assert result is not None
    assert result.state == 'running'


@pytest.mark.asyncio
async def test_read_status_rejects_invalid_id(jobs_dir: Path) -> None:
    with pytest.raises(ValueError, match='invalid job_id'):
        await train_jobs.read_status('../escaping')


@pytest.mark.asyncio
async def test_write_cancel_creates_sentinel(jobs_dir: Path) -> None:
    await train_jobs.write_cancel('some_job')
    sentinel = jobs_dir / 'some_job.cancel'
    assert sentinel.exists(), 'cancel sentinel was not written'


@pytest.mark.asyncio
async def test_write_cancel_idempotent(jobs_dir: Path) -> None:
    await train_jobs.write_cancel('twice')
    await train_jobs.write_cancel('twice')
    assert (jobs_dir / 'twice.cancel').exists()


@pytest.mark.asyncio
async def test_list_runs_returns_newest_first(jobs_dir: Path) -> None:
    # Write three specs out-of-order; mtime ordering should win.
    for jid, state in (('a', 'finished'), ('b', 'running'), ('c', 'queued')):
        spec = TrainJobSpec(job_id=jid, dataset_export_dir='/x')
        await train_jobs.write_job(spec)
        _write_status(jobs_dir, jid, state=state, heartbeat_at=datetime.now(UTC).isoformat())

    runs = await train_jobs.list_runs(limit=10)
    assert len(runs) == 3
    # Newest first means c, then b, then a (mtime order).
    assert {r.job_id for r in runs} == {'a', 'b', 'c'}


@pytest.mark.asyncio
async def test_get_active_job_returns_running_run(jobs_dir: Path) -> None:
    for jid, state in (('done1', 'finished'), ('live', 'running')):
        spec = TrainJobSpec(job_id=jid, dataset_export_dir='/x')
        await train_jobs.write_job(spec)
        _write_status(jobs_dir, jid, state=state, heartbeat_at=datetime.now(UTC).isoformat())

    active = await train_jobs.get_active_job()
    assert active is not None
    assert active.job_id == 'live'
    assert active.state == 'running'


@pytest.mark.asyncio
async def test_get_active_job_falls_back_to_most_recent(jobs_dir: Path) -> None:
    spec = TrainJobSpec(job_id='done', dataset_export_dir='/x')
    await train_jobs.write_job(spec)
    _write_status(jobs_dir, 'done', state='finished')
    result = await train_jobs.get_active_job()
    assert result is not None
    assert result.job_id == 'done'
    assert result.state == 'finished'


@pytest.mark.asyncio
async def test_get_active_job_raises_when_two_active(jobs_dir: Path) -> None:
    for jid in ('a', 'b'):
        spec = TrainJobSpec(job_id=jid, dataset_export_dir='/x')
        await train_jobs.write_job(spec)
        _write_status(jobs_dir, jid, state='running', heartbeat_at=datetime.now(UTC).isoformat())
    with pytest.raises(RuntimeError, match='multiple active runs'):
        await train_jobs.get_active_job()


@pytest.mark.asyncio
async def test_write_campaign_creates_n_files_with_shared_id(jobs_dir: Path) -> None:
    campaign = TrainCampaignSpec(
        campaign_id='2026-05-09_sweep',
        dataset_export_dir='/data/exports/x',
        runs=[
            CampaignRunSpec(profile='nano', model_size='n'),
            CampaignRunSpec(profile='small', model_size='s'),
            CampaignRunSpec(profile='medium', model_size='m'),
        ],
    )
    campaign_id, job_ids = await train_jobs.write_campaign(campaign)
    assert campaign_id == '2026-05-09_sweep'
    assert len(job_ids) == 3
    for jid in job_ids:
        path = jobs_dir / f'{jid}.job.json'
        assert path.exists()
        raw = json.loads(path.read_text())
        assert raw['campaign_id'] == '2026-05-09_sweep'


@pytest.mark.asyncio
async def test_cancel_campaign_writes_sentinels(jobs_dir: Path) -> None:
    campaign = TrainCampaignSpec(
        campaign_id='2026-05-09_cancel_me',
        dataset_export_dir='/data/exports/x',
        runs=[
            CampaignRunSpec(profile='nano', model_size='n'),
            CampaignRunSpec(profile='small', model_size='s'),
        ],
    )
    campaign_id, job_ids = await train_jobs.write_campaign(campaign)
    # Mark first as finished, second still queued.
    _write_status(jobs_dir, job_ids[0], state='finished')
    _write_status(jobs_dir, job_ids[1], state='running', heartbeat_at=datetime.now(UTC).isoformat())

    n = await train_jobs.cancel_campaign(campaign_id)
    assert n == 1, 'should only cancel the non-terminal run'
    assert (jobs_dir / f'{job_ids[1]}.cancel').exists()
    assert not (jobs_dir / f'{job_ids[0]}.cancel').exists()


@pytest.mark.asyncio
async def test_tail_run_log_returns_last_n(jobs_dir: Path) -> None:
    log = jobs_dir / 'logged.run.log'
    log.write_text('\n'.join(f'line {i}' for i in range(50)))
    out = await train_jobs.tail_run_log('logged', lines=5)
    assert out == ['line 45', 'line 46', 'line 47', 'line 48', 'line 49']


@pytest.mark.asyncio
async def test_tail_run_log_missing_file_returns_empty(jobs_dir: Path) -> None:
    out = await train_jobs.tail_run_log('absent', lines=5)
    assert out == []
