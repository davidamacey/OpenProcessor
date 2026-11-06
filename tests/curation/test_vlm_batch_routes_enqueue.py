# ruff: noqa: F401, F811 - pytest fixtures are imported by name and used as test arguments

from __future__ import annotations

import base64
import os
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.job_fixtures import job_env, kinds, producer, wakes
from fixtures.job_rows import make_project, sql

from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import new_project_record
from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.routers.curation import vlm as vlm_mod
from src.routers.curation._vlm_route_models import (
    VlmLabelBatchRequest,
    VlmVerifyRegionBatchItem,
    VlmVerifyRegionBatchRequest,
)
from src.services.curation import vlm_batch
from src.services.jobs import vlm_batch_sweep


pytestmark = [pytest.mark.db, pytest.mark.asyncio]

PINS = {'pack_name': None, 'vlm_endpoint': 'ep', 'vlm_endpoint_revision': 1}


@pytest.fixture
def bound(pg, job_env, tmp_path, monkeypatch):
    slug = f'vb-{uuid.uuid4().hex[:6]}'
    project = make_project(pg, slug)
    record = replace(new_project_record(slug, base_curation_config()), id=project)
    cfg = replace(base_curation_config(), crop_cache_dir=tmp_path / 'cache')
    monkeypatch.setattr(vlm_batch, 'base_curation_config', lambda: cfg)

    async def pins(*_a, **_k):
        return dict(PINS)

    monkeypatch.setattr(vlm_mod, 'request_pins', pins)
    with bind_project(record):
        yield project


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


async def _label(crop_ids):
    async with async_session_factory()() as session:
        return await vlm_mod.vlm_label_batch(VlmLabelBatchRequest(crop_ids=crop_ids), session, None)


async def _verify(*images: bytes):
    items = [
        VlmVerifyRegionBatchItem(crop_id=f'c{i}', region_image_b64=_b64(b), candidate_text='t')
        for i, b in enumerate(images)
    ]
    async with async_session_factory()() as session:
        return await vlm_mod.vlm_verify_region_batch(
            VlmVerifyRegionBatchRequest(items=items), session, None, None
        )


async def test_label_batch_answers_with_a_job_and_queues_one_row(pg, bound, producer, wakes):
    out = await _label(['a', 'b'])
    assert out['status'] == 'queued'
    rows = sql(pg, 'SELECT kind, status, params FROM jobs WHERE project_id = %s', bound)
    assert [(r[0], r[1]) for r in rows] == [('vlm_label_batch', 'queued')]
    assert rows[0][2]['crop_ids'] == ['a', 'b']
    assert rows[0][2]['vlm_endpoint'] == 'ep'
    assert producer.sent() == [out['job_id']]


async def test_verify_region_batch_stages_the_images_and_keeps_them_out_of_params(
    pg, bound, producer, wakes
):
    out = await _verify(b'img-one', b'img-two')
    (params,) = [r[0] for r in sql(pg, 'SELECT params FROM jobs WHERE project_id = %s', bound)]
    assert 'img-one' not in str(params)
    assert _b64(b'img-one') not in str(params)
    assert [vlm_batch.read_staged(i['key']) for i in params['items']] == [b'img-one', b'img-two']
    assert [i['candidate_text'] for i in params['items']] == ['t', 't']
    assert out['status'] == 'queued'


async def test_a_second_label_batch_while_one_is_active_is_not_blocked_but_a_broker_outage_is_503(
    pg, bound, producer, wakes
):
    await _label(['a'])
    await _label(['b'])  # no exclusivity: both queued
    assert len(producer.sent()) == 2
    producer.fail_send = True
    with pytest.raises(RequestRejectedError) as down:
        await _label(['c'])
    assert (down.value.status_code, down.value.detail['error']) == (503, 'queue_unavailable')
    states = sql(pg, 'SELECT status FROM jobs WHERE project_id = %s ORDER BY created_at', bound)
    assert [s[0] for s in states] == ['queued', 'queued', 'failed']


async def test_a_failed_enqueue_removes_the_staged_images(pg, bound, producer, wakes):
    producer.fail_send = True
    with pytest.raises(RequestRejectedError):
        await _verify(b'x')
    assert vlm_batch.staging_store().list_prefix('cache/vlm_batch') == []


async def test_the_sweep_removes_a_cancelled_queued_jobs_images_and_keeps_a_running_ones(
    pg, bound, producer, wakes
):
    cancelled = await _verify(b'gone')
    running = await _verify(b'kept')
    ids = {
        r[0]: r[1]
        for r in sql(pg, "SELECT id, params ->> 'batch_id' FROM jobs WHERE project_id = %s", bound)
    }
    by_job = {str(k): v for k, v in ids.items()}
    sql(pg, "UPDATE jobs SET status = 'cancelled', finished_at = now() WHERE id = %s",
        uuid.UUID(cancelled['job_id']))  # fmt: skip
    sql(pg, "UPDATE jobs SET status = 'running', fencing_token = 1, attempt = 1, owner = 'w', "
        "started_at = now(), heartbeat_at = now(), lease_expires_at = now() + interval '1 hour' "
        'WHERE id = %s', uuid.UUID(running['job_id']))  # fmt: skip
    removed = vlm_batch_sweep.run_sweep()
    assert removed == [by_job[cancelled['job_id']]]
    left = vlm_batch.staging_store().list_prefix('cache/vlm_batch')
    assert [k.split('/')[2] for k in left] == [by_job[running['job_id']]]


async def test_an_orphan_prefix_is_removed_only_after_the_grace(pg, bound):
    keys = vlm_batch.stage_images('orphan', [b'z'])
    assert vlm_batch_sweep.run_sweep() == []
    old = time.time() - vlm_batch_sweep.MISSING_JOB_GRACE_S - 5
    os.utime(vlm_batch.staging_store().local_path(keys[0]), (old, old))
    assert vlm_batch_sweep.run_sweep() == ['orphan']
