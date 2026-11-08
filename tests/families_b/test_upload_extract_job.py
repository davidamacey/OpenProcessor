"""``POST /datasets/uploads`` streams to storage and queues ``dataset_upload_extract`` (4B-2).

The request never extracts: ``safe_extract`` is called by the task, once.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import uuid
import zipfile
from pathlib import Path

import pytest
from families_b.support import project
from fixtures.job_rows import sql

from src.services.curation.dataset_import import upload


pytestmark = pytest.mark.db


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _roots(tmp_path, monkeypatch):
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))


def _project(app, pg, tmp_path):
    """A project whose extracted datasets land under ``tmp_path`` (not the system default)."""
    slug, pid = project(app, pg)
    record = app.state.records[slug]
    resources = dataclasses.replace(record.resources, upload_root=tmp_path / 'uploads')
    app.state.records[slug] = dataclasses.replace(record, resources=resources)
    return slug, pid


async def _upload(client, slug, data: bytes, name: str = 'ds.zip'):
    return await client.post(
        f'/curation/projects/{slug}/datasets/uploads', files={'file': (name, data)}
    )


@pytest.mark.asyncio
async def test_the_route_streams_to_storage_queues_one_job_and_does_not_extract(
    app, client, pg, producer, monkeypatch, tmp_path
) -> None:
    slug, pid = _project(app, pg, tmp_path)
    called: list[str] = []
    real = upload.safe_extract
    monkeypatch.setattr(upload, 'safe_extract', lambda *a, **k: called.append('x') or real(*a, **k))

    reply = await _upload(client, slug, _zip({'ds/data.yaml': b'names: [a]'}))

    assert reply.status_code == 202, reply.text
    body = reply.json()
    assert producer.sent() == [body['job_id']]
    assert called == []  # no safe_extract in the request
    [(status, size, sha, job, key)] = sql(
        pg,
        'SELECT status, size_bytes, sha256, extract_job_id, storage_key FROM dataset_uploads '
        'WHERE id = %s',
        uuid.UUID(body['upload_id']),
    )
    assert status == 'received'
    assert size > 0
    assert sha
    assert str(job) == body['job_id']
    assert sql(pg, 'SELECT kind, status FROM jobs WHERE project_id = %s', pid) == [
        ('dataset_upload_extract', 'queued')
    ]
    assert key.startswith(f'projects/{slug}/uploads/')


@pytest.mark.asyncio
async def test_the_task_extracts_once_records_the_result_and_removes_the_archive(
    app, client, pg, producer, worker_app, monkeypatch, tmp_path
) -> None:
    from families_b.support import bind_worker_to

    slug, _pid = _project(app, pg, tmp_path)
    bind_worker_to(monkeypatch, app, None)
    calls: list[Path] = []
    real = upload.safe_extract
    monkeypatch.setattr(upload, 'safe_extract', lambda a, d: calls.append(a) or real(a, d))
    reply = (await _upload(client, slug, _zip({'ds/data.yaml': b'names: [a]'}))).json()

    outcome = await asyncio.to_thread(
        worker_app.tasks['op.families_b.dataset_upload_extract'].apply,
        args=[reply['job_id']],
        task_id=reply['job_id'],
    )

    assert outcome.successful(), outcome.traceback
    assert len(calls) == 1
    [(status, result)] = sql(
        pg, 'SELECT status, result FROM jobs WHERE id = %s', uuid.UUID(reply['job_id'])
    )
    assert status == 'completed'
    assert result['files'] == 1
    assert (Path(result['dataset_path']) / 'data.yaml').read_bytes() == b'names: [a]'
    assert sql(
        pg, 'SELECT status FROM dataset_uploads WHERE id = %s', uuid.UUID(reply['upload_id'])
    ) == [('extracted',)]
    assert not calls[0].exists()  # the received archive is gone


@pytest.mark.asyncio
async def test_an_unsafe_archive_fails_the_job_and_the_row(
    app, client, pg, producer, worker_app, monkeypatch, tmp_path
) -> None:
    from families_b.support import bind_worker_to

    slug, _pid = _project(app, pg, tmp_path)
    bind_worker_to(monkeypatch, app, None)
    reply = (await _upload(client, slug, _zip({'../escape.txt': b'x'}))).json()

    await asyncio.to_thread(
        worker_app.tasks['op.families_b.dataset_upload_extract'].apply,
        args=[reply['job_id']],
        task_id=reply['job_id'],
    )

    [(status, message)] = sql(
        pg, 'SELECT status, error_message FROM jobs WHERE id = %s', uuid.UUID(reply['job_id'])
    )
    assert status == 'failed'
    assert 'unsafe member name' in message
    assert sql(
        pg, 'SELECT status FROM dataset_uploads WHERE id = %s', uuid.UUID(reply['upload_id'])
    ) == [('failed',)]


@pytest.mark.asyncio
async def test_an_upload_over_the_cap_is_413_and_leaves_a_failed_row_and_no_blob(
    app, client, pg, producer, monkeypatch, tmp_path
) -> None:
    slug, pid = project(app, pg)
    monkeypatch.setenv('OP_DATASET_UPLOAD_MAX_BYTES', '100')

    reply = await _upload(client, slug, b'x' * 5000)

    assert reply.status_code == 413
    assert reply.json()['detail']['error'] == 'upload_too_large'
    assert sql(pg, 'SELECT status FROM dataset_uploads WHERE project_id = %s', pid) == [('failed',)]
    assert producer.sent() == []
    assert [p for p in (tmp_path / 'projects_data').rglob('*') if p.is_file()] == []


@pytest.mark.asyncio
async def test_broker_down_fails_the_row_and_answers_503(app, client, pg, producer) -> None:
    slug, pid = project(app, pg)
    producer.fail_send = True

    reply = await _upload(client, slug, _zip({'ds/data.yaml': b'names: [a]'}))

    assert reply.status_code == 503
    assert reply.json()['detail']['error'] == 'queue_unavailable'
    assert sql(pg, 'SELECT status FROM dataset_uploads WHERE project_id = %s', pid) == [('failed',)]
