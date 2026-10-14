"""Run-artifact resolution (``GET {api_prefix}/train/artifacts/{job_id}/{name}``)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.services.training import job_wire, jobs as train_jobs


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_resolve_run_dir_from_checkpoint_path(jobs_dir: Path) -> None:
    run_dir = jobs_dir / 'a_run'
    (run_dir / 'weights').mkdir(parents=True)
    (jobs_dir / 'ck.status.json').write_text(
        json.dumps(
            {
                'job_id': 'ck',
                'state': 'finished',
                'checkpoint_path': str(run_dir / 'weights' / 'best.pt'),
            }
        )
    )
    resolved = await train_jobs.resolve_run_dir('ck')
    assert resolved == run_dir.resolve()


@pytest.mark.asyncio
async def test_resolve_run_dir_falls_back_to_confusion_matrix_path(jobs_dir: Path) -> None:
    run_dir = jobs_dir / 'b_run'
    run_dir.mkdir()
    (jobs_dir / 'cm.status.json').write_text(
        json.dumps(
            {
                'job_id': 'cm',
                'state': 'failed',
                'eval': {'confusion_matrix_path': str(run_dir / 'confusion_matrix.png')},
            }
        )
    )
    resolved = await train_jobs.resolve_run_dir('cm')
    assert resolved == run_dir.resolve()


@pytest.mark.asyncio
async def test_resolve_run_dir_none_when_no_hints(jobs_dir: Path) -> None:
    (jobs_dir / 'bare.status.json').write_text(json.dumps({'job_id': 'bare', 'state': 'running'}))
    assert await train_jobs.resolve_run_dir('bare') is None


@pytest.mark.asyncio
async def test_resolve_run_dir_none_when_no_status(jobs_dir: Path) -> None:
    assert await train_jobs.resolve_run_dir('never_ran') is None


@pytest.mark.asyncio
async def test_read_artifact_serves_a_whitelisted_file(jobs_dir: Path) -> None:
    run_dir = jobs_dir / 'good_run'
    run_dir.mkdir()
    (run_dir / 'confusion_matrix.png').write_bytes(b'fake-png')
    (jobs_dir / 'good.status.json').write_text(
        json.dumps(
            {
                'job_id': 'good',
                'state': 'finished',
                'eval': {'confusion_matrix_path': str(run_dir / 'confusion_matrix.png')},
            }
        )
    )
    path = await train_jobs.read_artifact('good', 'confusion_matrix.png')
    assert path == (run_dir / 'confusion_matrix.png').resolve()


@pytest.mark.asyncio
async def test_read_artifact_rejects_a_non_whitelisted_name(jobs_dir: Path) -> None:
    run_dir = jobs_dir / 'good_run2'
    run_dir.mkdir()
    (run_dir / 'best.pt').write_bytes(b'weights')
    (jobs_dir / 'good2.status.json').write_text(
        json.dumps(
            {'job_id': 'good2', 'state': 'finished', 'checkpoint_path': str(run_dir / 'best.pt')}
        )
    )
    assert await train_jobs.read_artifact('good2', 'best.pt') is None


@pytest.mark.asyncio
async def test_read_artifact_none_for_unknown_job(jobs_dir: Path) -> None:
    assert await train_jobs.read_artifact('does_not_exist', 'confusion_matrix.png') is None


@pytest.mark.asyncio
async def test_read_artifact_none_when_file_never_written(jobs_dir: Path) -> None:
    run_dir = jobs_dir / 'no_file_run'
    run_dir.mkdir()
    (jobs_dir / 'nofile.status.json').write_text(
        json.dumps(
            {
                'job_id': 'nofile',
                'state': 'running',
                'eval': {'confusion_matrix_path': str(run_dir / 'confusion_matrix.png')},
            }
        )
    )
    assert await train_jobs.read_artifact('nofile', 'confusion_matrix.png') is None


@pytest.mark.asyncio
async def test_read_artifact_rejects_traversal_attempt(jobs_dir: Path) -> None:
    run_dir = jobs_dir / 'traversal_run'
    run_dir.mkdir()
    secret = jobs_dir / 'confusion_matrix.png'  # sibling of run_dir, outside it
    secret.write_bytes(b'not-this-one')
    (jobs_dir / 'trav.status.json').write_text(
        json.dumps(
            {
                'job_id': 'trav',
                'state': 'finished',
                'eval': {'confusion_matrix_path': str(run_dir / 'confusion_matrix.png')},
            }
        )
    )
    # Not in the whitelist (a slash isn't a valid character in any
    # whitelisted basename), so this must resolve to None rather than
    # escaping run_dir to serve the sibling file.
    assert await train_jobs.read_artifact('trav', '../confusion_matrix.png') is None


@pytest.mark.asyncio
async def test_read_artifact_rejects_invalid_job_id() -> None:
    with pytest.raises(ValueError, match='invalid job_id'):
        await train_jobs.read_artifact('../escaping', 'confusion_matrix.png')


def test_artifact_media_type_png_and_csv() -> None:
    assert job_wire.artifact_media_type('confusion_matrix.png') == 'image/png'
    assert job_wire.artifact_media_type('results.csv') == 'text/csv'
    assert job_wire.artifact_media_type('unknown.bin') == 'application/octet-stream'
