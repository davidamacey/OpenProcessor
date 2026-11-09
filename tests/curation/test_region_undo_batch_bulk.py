"""``POST /crops/region/undo_batch`` writes through the shared paged bulk loop.

A batch up to ``OP_BATCH_SYNC_MAX`` is one ``_mget`` plus one conditional ``_bulk``; a larger one
queues the ``edit_undo_batch`` job (202) whose handler undoes the same pages. Each crop keeps its
own outcome: ``undone``, ``nothing_to_undo``, ``conflicts`` or ``not_found``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from curation.label_ops_env import OpsEnv, build_env
from curation.queued_label import run_job
from src.clients.occ import OCC_BULK_PAGE_SIZE
from src.config import get_region_fields
from src.domain.region_boxes import RegionBox, boxes_write_fields


if TYPE_CHECKING:
    from pathlib import Path


pytest_plugins = ['_records_world', 'curation.jobs_a.autoqueue']
pytestmark = [
    pytest.mark.db,
    pytest.mark.usefixtures('world', 'reference_region_profile'),
]

F = get_region_fields()
UNDO = 'crops/region/undo_batch'


def _detected(crop_id: str) -> dict[str, Any]:
    box = RegionBox(
        box_id='b1',
        bbox_norm=(0.2, 0.2, 0.4, 0.4),
        state='accepted',
        score=0.9,
        detector='det_model',
        detector_version='3',
        source='detector',
        detected_at='2026-09-24T03:08:19+00:00',
    )
    return {
        'crop_id': crop_id,
        F.status: 'detected',
        F.verified: True,
        F.validated: False,
        **boxes_write_fields([box], current_src={}),
    }


def _ids(count: int) -> list[str]:
    return [f'c{i:04d}' for i in range(count)]


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, count: int) -> OpsEnv:
    """``count`` detected crops, each given one human region write (the thing to undo)."""
    docs = {i: _detected(i) for i in _ids(count)}
    env = build_env(monkeypatch, tmp_path, docs)
    env.docs()['fresh'] = _detected('fresh')
    written = env.client.post(
        f'{env.prefix}/regions/batch_status',
        json={'crop_ids': _ids(count), 'region_status': 'false_positive'},
    )
    assert written.status_code == 200, written.text
    return env


def _undo(env: OpsEnv, crop_ids: list[str]) -> Any:
    return env.client.post(f'{env.prefix}/{UNDO}', json={'crop_ids': crop_ids})


def test_100_crops_are_one_mget_and_one_bulk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path, 100)
    env.fake.bulk_sizes.clear()
    env.fake.max_in_flight = 0
    singles, mgets = env.fake.update_calls, env.fake.mget_calls

    resp = _undo(env, [*_ids(100), 'fresh', 'nope'])

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['undone'] == 100
    assert body['nothing_to_undo'] == ['fresh']
    assert body['not_found'] == ['nope']
    assert body['conflicts'] == []
    assert env.fake.update_calls == singles, 'no per-crop conditional updates'
    # one read for the page, one for the response items
    assert env.fake.mget_calls - mgets == 2
    assert env.fake.bulk_sizes == [100]
    assert env.fake.max_in_flight == 1
    assert all(env.docs()[i][F.status] == 'detected' for i in _ids(100))


def test_a_write_that_keeps_conflicting_is_a_conflicts_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path, 4)
    env.fake.conflict_ids = {'c0002'}
    body = _undo(env, _ids(4)).json()

    assert (body['undone'], body['conflicts']) == (3, ['c0002'])
    assert env.docs()['c0002'][F.status] == 'false_positive'


def test_nothing_to_undo_anywhere_is_409(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = _env(monkeypatch, tmp_path, 1)
    assert _undo(env, ['fresh']).status_code == 409


def test_the_id_cap_of_the_request_is_5000(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = _env(monkeypatch, tmp_path, 1)
    assert _undo(env, _ids(5001)).status_code == 422


def test_a_single_crop_undo_is_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = _env(monkeypatch, tmp_path, 2)
    resp = env.client.post(f'{env.prefix}/crops/c0001/region/undo')

    assert resp.status_code == 200, resp.text
    assert env.docs()['c0001'][F.status] == 'detected'
    assert env.docs()['c0000'][F.status] == 'false_positive'
    assert env.client.post(f'{env.prefix}/crops/c0001/region/undo').status_code == 409


def test_501_crops_queue_a_job_and_the_handler_undoes_the_same_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from src.services.curation.edit_undo import run_edit_undo_job

    env = _env(monkeypatch, tmp_path, 501)
    env.fake.bulk_sizes.clear()
    resp = _undo(env, [*_ids(501), 'fresh', 'nope'])

    assert resp.status_code == 202, resp.text
    accepted = resp.json()
    assert accepted['status'] == 'queued'
    assert env.fake.bulk_sizes == [], 'the request wrote nothing'
    busy = _undo(env, _ids(501))
    assert (busy.status_code, busy.json()['detail']['error']) == (409, 'job_busy')

    result = asyncio.run(run_job(accepted['job_id'], lambda ctx: run_edit_undo_job(ctx, env.fake)))
    assert result['undone'] == 501
    assert (result['nothing_to_undo'], result['not_found'], result['conflicts']) == (
        ['fresh'],
        ['nope'],
        [],
    )
    assert env.fake.bulk_sizes == [OCC_BULK_PAGE_SIZE, 1]
    assert env.docs()['c0500'][F.status] == 'detected'
