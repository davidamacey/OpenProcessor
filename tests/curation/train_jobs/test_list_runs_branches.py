"""``list_runs`` branch by branch: which files are listed, skipped or rehydrated, and in what order."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from curation.train_jobs._helpers import _write_status
from src.services.training import jobs as train_jobs


if TYPE_CHECKING:
    from pathlib import Path


def _touch(path: Path, mtime: float) -> None:
    os.utime(path, (mtime, mtime))


def _write_job_json(jobs_dir: Path, job_id: str, payload: Any) -> Path:
    path = jobs_dir / f'{job_id}.job.json'
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.asyncio
async def test_non_positive_limit_returns_nothing(jobs_dir: Path) -> None:
    _write_status(jobs_dir, 'a')
    assert await train_jobs.list_runs(limit=0) == []
    assert await train_jobs.list_runs(limit=-3) == []


@pytest.mark.asyncio
async def test_missing_jobs_dir_returns_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path / 'never_created'))
    assert await train_jobs.list_runs() == []


@pytest.mark.asyncio
async def test_job_without_status_is_listed_queued_with_its_campaign(jobs_dir: Path) -> None:
    _write_job_json(jobs_dir, 'queued_one', {'job_id': 'queued_one', 'campaign_id': 'camp'})
    runs = await train_jobs.list_runs()
    assert [(r.job_id, r.state, r.campaign_id) for r in runs] == [('queued_one', 'queued', 'camp')]


@pytest.mark.asyncio
async def test_job_without_job_id_falls_back_to_the_file_stem(jobs_dir: Path) -> None:
    _write_job_json(jobs_dir, 'stem_name', {'dataset_export_dir': '/x'})
    runs = await train_jobs.list_runs()
    # Path.stem of "stem_name.job.json" is "stem_name.job"
    assert [(r.job_id, r.state, r.campaign_id) for r in runs] == [('stem_name.job', 'queued', None)]


@pytest.mark.asyncio
async def test_status_wins_over_the_job_file_of_the_same_run(jobs_dir: Path) -> None:
    _write_job_json(jobs_dir, 'both', {'job_id': 'both'})
    _write_status(jobs_dir, 'both', state='finished')
    runs = await train_jobs.list_runs()
    assert [(r.job_id, r.state) for r in runs] == [('both', 'finished')]


@pytest.mark.asyncio
async def test_unreadable_status_is_skipped_and_its_job_file_is_listed(jobs_dir: Path) -> None:
    (jobs_dir / 'broken.status.json').write_text('{not json')
    _write_job_json(jobs_dir, 'broken', {'job_id': 'broken'})
    runs = await train_jobs.list_runs()
    assert [(r.job_id, r.state) for r in runs] == [('broken', 'queued')]


@pytest.mark.asyncio
async def test_non_object_status_and_job_files_are_skipped(jobs_dir: Path) -> None:
    (jobs_dir / 'listy.status.json').write_text('[1, 2]')
    (jobs_dir / 'scalar.job.json').write_text('"text"')
    (jobs_dir / 'broken.job.json').write_text('{oops')
    assert await train_jobs.list_runs() == []


@pytest.mark.asyncio
async def test_status_that_fails_validation_is_skipped_with_a_warning(
    jobs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        train_jobs.logger, 'warning', lambda event, **kw: warnings.append((event, kw))
    )
    (jobs_dir / 'bad.status.json').write_text(json.dumps({'job_id': 'bad', 'state': 'nonsense'}))
    _write_status(jobs_dir, 'good', state='finished')
    runs = await train_jobs.list_runs()
    assert [r.job_id for r in runs] == ['good']
    assert [event for event, _ in warnings] == ['curation_train_status_invalid']


@pytest.mark.asyncio
async def test_campaign_id_is_rehydrated_from_the_job_file(jobs_dir: Path) -> None:
    _write_status(jobs_dir, 'run1', state='finished')
    _write_job_json(jobs_dir, 'run1', {'job_id': 'run1', 'campaign_id': 'camp_a'})
    (run,) = await train_jobs.list_runs()
    assert run.campaign_id == 'camp_a'


@pytest.mark.asyncio
async def test_campaign_id_in_the_status_is_not_overwritten(jobs_dir: Path) -> None:
    (jobs_dir / 'run2.status.json').write_text(
        json.dumps({'job_id': 'run2', 'state': 'finished', 'campaign_id': 'from_status'})
    )
    _write_job_json(jobs_dir, 'run2', {'job_id': 'run2', 'campaign_id': 'from_job'})
    (run,) = await train_jobs.list_runs()
    assert run.campaign_id == 'from_status'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'job_text',
    [None, '{broken', '[1]', json.dumps({'job_id': 'run3'}), json.dumps({'campaign_id': ''})],
    ids=['no_job_file', 'invalid_json', 'not_an_object', 'no_campaign_key', 'empty_campaign'],
)
async def test_campaign_id_stays_unset_when_the_job_file_cannot_supply_one(
    jobs_dir: Path, job_text: str | None
) -> None:
    _write_status(jobs_dir, 'run3', state='finished')
    if job_text is not None:
        (jobs_dir / 'run3.job.json').write_text(job_text)
    (run,) = await train_jobs.list_runs()
    assert run.campaign_id is None


@pytest.mark.asyncio
async def test_stale_heartbeat_is_listed_as_lost(jobs_dir: Path) -> None:
    stale = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    _write_status(jobs_dir, 'stale', state='running', heartbeat_at=stale)
    (run,) = await train_jobs.list_runs()
    assert run.state == 'lost'


@pytest.mark.asyncio
async def test_runs_are_newest_first_and_sliced_by_offset_and_limit(jobs_dir: Path) -> None:
    for index, job_id in enumerate(['oldest', 'older', 'newer', 'newest']):
        _write_status(jobs_dir, job_id, state='finished')
        _touch(jobs_dir / f'{job_id}.status.json', 1_700_000_000 + index * 10)
    ordered = [r.job_id for r in await train_jobs.list_runs(limit=10)]
    assert ordered == ['newest', 'newer', 'older', 'oldest']
    assert [r.job_id for r in await train_jobs.list_runs(limit=2)] == ['newest', 'newer']
    assert [r.job_id for r in await train_jobs.list_runs(limit=2, offset=1)] == ['newer', 'older']
    assert await train_jobs.list_runs(limit=5, offset=9) == []


@pytest.mark.asyncio
async def test_status_and_queued_job_files_sort_together_by_mtime(jobs_dir: Path) -> None:
    _write_status(jobs_dir, 'with_status', state='finished')
    _touch(jobs_dir / 'with_status.status.json', 1_700_000_000)
    path = _write_job_json(jobs_dir, 'only_job', {'job_id': 'only_job'})
    _touch(path, 1_700_000_100)
    assert [r.job_id for r in await train_jobs.list_runs()] == ['only_job', 'with_status']
