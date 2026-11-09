"""``POST /review/new_class_proposals/resolve`` writes through the shared paged bulk loop.

A batch up to ``OP_BATCH_SYNC_MAX`` is one ``_mget`` plus one conditional ``_bulk``; a larger one
queues the ``review_resolve_batch`` job (202) whose handler writes the same pages. A write that
keeps conflicting is a ``conflicts`` entry; an item no longer pending the proposal is ``skipped``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from curation.label_ops_env import OpsEnv, build_env
from curation.queued_label import run_job
from src.clients.occ import OCC_BULK_PAGE_SIZE


if TYPE_CHECKING:
    from pathlib import Path


pytest_plugins = ['_records_world', 'curation.jobs_a.autoqueue']
pytestmark = [pytest.mark.db, pytest.mark.usefixtures('world')]

ROUTE = 'review/new_class_proposals/resolve'


def _pending(crop_id: str, label: str = 'sidecar', **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'class_source': 'vlm_new_class_pending',
        'vlm_proposed_class': label,
        'needs_new_class': True,
        **extra,
    }


def _ids(count: int) -> list[str]:
    return [f'c{i:04d}' for i in range(count)]


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, count: int) -> OpsEnv:
    env = build_env(monkeypatch, tmp_path, {i: _pending(i) for i in _ids(count)})
    monkeypatch.setattr(
        'src.routers.curation.review_resolve.get_class_registry', lambda: env.registry
    )
    return env


def _resolve(env: OpsEnv, *, dry_run: bool = False) -> Any:
    body = {'label': 'sidecar', 'class_id': env.class_ids['widget']}
    return env.client.post(f'{env.prefix}/{ROUTE}', json=body, params={'dry_run': dry_run})


def test_100_crops_are_one_mget_and_one_bulk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path, 100)
    resp = _resolve(env)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body['matched'], body['updated'], body['conflicts'], body['skipped']) == (
        100,
        100,
        [],
        [],
    )
    assert env.fake.update_calls == 0, 'no per-crop conditional updates'
    assert env.fake.mget_calls == 1
    assert env.fake.bulk_sizes == [100]
    assert env.fake.max_in_flight == 1
    assert all(env.docs()[i]['class_source'] == 'human' for i in _ids(100))
    assert all(env.docs()[i]['needs_new_class'] is False for i in _ids(100))


def test_a_write_that_keeps_conflicting_is_a_conflicts_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path, 10)
    env.fake.conflict_ids = {'c0003'}
    body = _resolve(env).json()

    assert body['updated'] == 9
    assert body['conflicts'] == [{'crop_id': 'c0003', 'current_source': 'vlm_new_class_pending'}]
    assert env.docs()['c0003']['class_source'] == 'vlm_new_class_pending'


def test_an_item_no_longer_pending_is_skipped_and_never_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from src.services.curation.new_class_resolve import resolve_pending_proposals

    env = _env(monkeypatch, tmp_path, 3)
    env.docs()['c0001'].update({'class_source': 'human', 'class_validated': True, 'class_id': 99})
    outcome = asyncio.run(
        resolve_pending_proposals(
            env.fake,
            [*_ids(3), 'gone'],
            label='sidecar',
            class_id=env.class_ids['widget'],
            class_name='widget',
            label_source='human',
        )
    )

    assert outcome.updated_ids == ['c0000', 'c0002']
    assert sorted(outcome.skipped) == ['c0001', 'gone']
    assert env.docs()['c0001']['class_id'] == 99


def test_a_single_crop_resolve_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path, 1)
    body = _resolve(env).json()

    assert body['updated_ids'] == ['c0000']
    assert body['class_name'] == 'widget'
    assert env.docs()['c0000']['class_id'] == env.class_ids['widget']
    assert env.docs()['c0000']['class_validated'] is True


def test_501_crops_queue_a_job_and_the_handler_writes_the_same_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from src.services.curation.new_class_resolve import run_resolve_job

    env = _env(monkeypatch, tmp_path, 501)
    resp = _resolve(env)

    assert resp.status_code == 202, resp.text
    accepted = resp.json()
    assert accepted['status'] == 'queued'
    assert env.fake.bulk_sizes == [], 'the request wrote nothing'

    busy = _resolve(env)
    assert (busy.status_code, busy.json()['detail']['error']) == (409, 'job_busy')

    result = asyncio.run(run_job(accepted['job_id'], lambda ctx: run_resolve_job(ctx, env.fake)))
    assert (result['matched'], result['updated'], result['conflicts'], result['skipped']) == (
        501,
        501,
        [],
        [],
    )
    assert env.fake.bulk_sizes == [OCC_BULK_PAGE_SIZE, 1]
    assert env.docs()['c0500']['class_id'] == env.class_ids['widget']


def test_the_sync_bound_is_the_environment_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('OP_BATCH_SYNC_MAX', '5')
    env = _env(monkeypatch, tmp_path, 6)
    assert _resolve(env, dry_run=True).json()['matched'] == 6  # a dry run never queues
    assert _resolve(env).status_code == 202
