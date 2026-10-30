"""The batch undo and batch discard routes write through the shared paged bulk loop,
not one conditional update per id, and answer a typed body."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from curation.label_ops_env import build_env, proposal
from src.clients.occ import OCC_BULK_PAGE_SIZE
from src.schemas.curation.label_ops import CropUndoBatchResponse


if TYPE_CHECKING:
    from pathlib import Path

    import pytest

N = 1_300


def _ids(count: int) -> list[str]:
    return [f'c{i:05d}' for i in range(count)]


def test_undo_batch_is_typed_bulk_and_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = build_env(monkeypatch, tmp_path, {i: proposal(i) for i in _ids(N)})
    env.docs()['unlabelled'] = proposal('unlabelled')
    env.label('widget', crop_ids=_ids(N))
    env.fake.bulk_sizes.clear()
    env.fake.max_in_flight = 0
    singles = env.fake.update_calls

    resp = env.client.post(
        f'{env.prefix}/crops/label/undo_batch',
        json={'crop_ids': [*_ids(N), 'unlabelled', 'missing']},
    )

    assert resp.status_code == 200, resp.text
    body = CropUndoBatchResponse.model_validate(resp.json())
    assert body.undone == N
    assert len(body.items) == N
    assert body.nothing_to_undo == ['unlabelled']
    assert body.not_found == ['missing']
    assert body.conflicts == []
    assert env.fake.update_calls == singles, 'no per-id conditional updates'
    assert max(env.fake.bulk_sizes) <= OCC_BULK_PAGE_SIZE
    assert len(env.fake.bulk_sizes) == math.ceil(N / OCC_BULK_PAGE_SIZE)
    assert env.fake.max_in_flight == 1
    assert all(not env.docs()[i].get('class_validated') for i in _ids(N))

    too_many = env.client.post(
        f'{env.prefix}/crops/label/undo_batch', json={'crop_ids': _ids(5_001)}
    )
    assert too_many.status_code == 422, 'the 5,000-id cap of this route stays'


def test_undo_batch_reports_conflicts_and_409_when_nothing_to_undo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = build_env(monkeypatch, tmp_path, {i: proposal(i) for i in _ids(4)})
    nothing = env.client.post(f'{env.prefix}/crops/label/undo_batch', json={'crop_ids': _ids(4)})
    assert nothing.status_code == 409
    env.label('widget', crop_ids=_ids(4))
    env.fake.conflict_ids = {'c00002'}
    resp = env.client.post(f'{env.prefix}/crops/label/undo_batch', json={'crop_ids': _ids(4)})
    body = resp.json()
    assert (body['undone'], body['conflicts']) == (3, ['c00002'])
    assert env.docs()['c00002']['class_validated'] is True


def test_discard_batch_is_bulk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = build_env(monkeypatch, tmp_path, {i: proposal(i) for i in _ids(N)})
    env.label('widget', crop_ids=_ids(N))
    env.fake.bulk_sizes.clear()
    env.fake.max_in_flight = 0
    singles = env.fake.update_calls

    resp = env.client.post(
        f'{env.prefix}/crops/discard_batch', json={'crop_ids': [*_ids(N), 'missing']}
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['discarded'] == N
    assert body['not_found'] == ['missing']
    assert body['conflicts'] == []
    assert env.fake.update_calls == singles
    assert max(env.fake.bulk_sizes) <= OCC_BULK_PAGE_SIZE
    assert env.fake.max_in_flight == 1
    assert env.docs()['c00000']['class_id'] is None
    # The discard is still one human write that the batch undo reverses.
    undo = env.client.post(f'{env.prefix}/crops/label/undo_batch', json={'crop_ids': _ids(5)})
    assert undo.json()['undone'] == 5
    assert env.docs()['c00000']['class_name'] == 'widget'
