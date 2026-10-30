"""Bulk label and move are recorded as operations: tagged items, a stored record,
the list route and retention."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from curation.label_ops_env import build_env, proposal
from src.domain.history import find_undo_snapshot
from src.domain.label_ops import LABEL_OP_ID_RE
from src.services.curation.label_ops import store


if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from curation.label_ops_env import OpsEnv


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, n: int = 6) -> OpsEnv:
    return build_env(monkeypatch, tmp_path, {f'c{i}': proposal(f'c{i}') for i in range(n)})


def test_batch_label_returns_an_op_id_and_tags_every_written_item(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    body = env.label('widget', crop_ids=['c0', 'c1', 'c2'])
    op_id = body['op_id']
    assert LABEL_OP_ID_RE.fullmatch(op_id)
    assert sorted(body['updated_ids']) == ['c0', 'c1', 'c2']
    for crop_id in ('c0', 'c1', 'c2'):
        doc = env.docs()[crop_id]
        assert doc['label_op_id'] == op_id
        newest = doc['class_id_history'][-1]
        assert newest['op_id'] == op_id
        assert find_undo_snapshot(doc['class_id_history']) is newest
    assert 'label_op_id' not in env.docs()['c3']

    record = json.loads((env.state() / op_id / 'state.json').read_text())
    assert record['status'] == 'written'
    assert (record['kind'], record['count'], record['conflicts'], record['selected']) == (
        'label',
        3,
        0,
        3,
    )
    members = json.loads((env.state() / op_id / 'members.json').read_text())['ids']
    assert sorted(members) == ['c0', 'c1', 'c2']


def test_dry_run_and_empty_selection_record_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    dry = env.client.put(
        f'{env.prefix}/crops/batch_label',
        json={'class_id': env.class_ids['widget'], 'crop_ids': ['c0'], 'dry_run': True},
    )
    assert dry.json() == {'dry_run': True, 'selected': 1}
    empty = env.label('widget', selection={'filter': {'class_names': ['no_such_class']}})
    assert empty['op_id'] is None
    assert empty['updated'] == 0
    assert not env.state().exists() or not list(env.state().iterdir())
    assert env.list_ops().json()['items'] == []
    assert env.fake.write_calls == 0


def test_a_write_without_a_record_never_happens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)

    def refuse(_new: Any) -> Any:
        raise OSError('disk full')

    monkeypatch.setattr(store, 'save_op', refuse)
    resp = env.client.put(
        f'{env.prefix}/crops/batch_label',
        json={'class_id': env.class_ids['widget'], 'crop_ids': ['c0']},
    )
    assert resp.status_code == 503, resp.text
    assert env.fake.write_calls == 0
    assert 'label_op_id' not in env.docs()['c0']


def test_label_ops_list_newest_first_counts_and_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    first = env.label('widget', crop_ids=['c0', 'c1', 'c2'])['op_id']
    second = env.move(env.class_ids['gadget'], crop_ids=['c2', 'c3'])['op_id']
    # A person relabels c1 afterwards: the first operation no longer owns it.
    assert (
        env.client.put(
            f'{env.prefix}/crops/c1/label', json={'class_id': env.class_ids['gizmo']}
        ).status_code
        == 200
    )

    body = env.list_ops().json()
    assert body['limit'] == 20
    assert body['retention'] == store.LABEL_OPS_KEEP == 200
    assert [o['op_id'] for o in body['items']] == [second, first]
    newer, older = body['items']
    assert (newer['kind'], newer['class_name'], newer['target_kind'], newer['current']) == (
        'move',
        'gadget',
        'class',
        2,
    )
    # c2 moved on to the second operation, c1 to a person: only c0 still carries the first.
    assert (older['kind'], older['class_name'], older['count'], older['current']) == (
        'label',
        'widget',
        3,
        1,
    )
    assert older['undoable'] is True
    assert older['undone'] is False
    assert older['last_undo'] is None
    assert older['status'] == 'written'
    assert 'class_id' not in older

    assert [o['op_id'] for o in env.list_ops(limit=1).json()['items']] == [second]
    assert env.list_ops(limit=51).status_code == 422
    assert env.list_ops(limit=0).status_code == 422

    # Nothing is left of the first operation once c0 is taken over as well.
    assert (
        env.client.put(
            f'{env.prefix}/crops/c0/label', json={'class_id': env.class_ids['gizmo']}
        ).status_code
        == 200
    )
    only = env.list_ops(undoable='true', limit=5).json()['items']
    assert [o['op_id'] for o in only] == [second]
    assert env.list_ops(undoable='true', limit=1).json()['items'][0]['op_id'] == second


def test_interrupted_and_writing_operations_are_served_and_not_undoable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    op_id = env.label('widget', crop_ids=['c0'])['op_id']
    path = env.state() / op_id / 'state.json'
    record = json.loads(path.read_text())

    path.write_text(json.dumps({**record, 'status': 'writing'}))
    item = env.list_ops().json()['items'][0]
    assert item['status'] == 'interrupted'  # nothing holds write.lock
    assert item['undoable'] is True

    with store.write_lock(op_id) as got:
        assert got
        held = env.list_ops().json()['items'][0]
    assert held['status'] == 'writing'
    assert held['undoable'] is False


def test_retention_keeps_the_newest_200_and_never_prunes_a_live_op(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path, n=3)
    monkeypatch.setattr(store, 'LABEL_OPS_KEEP', 5)
    first = env.label('widget', crop_ids=['c0'])['op_id']
    first_dir = env.state() / first
    record = json.loads((first_dir / 'state.json').read_text())
    (first_dir / 'state.json').write_text(json.dumps({**record, 'status': 'undoing'}))
    with store.write_lock(first):  # any lock alone does not make it live: the undo lock does
        pass
    with store.undo_lock() as got:
        assert got
        for _ in range(7):
            env.label('gadget', crop_ids=['c1'])
        kept = sorted(p.name for p in env.state().iterdir() if p.is_dir())
        assert first in kept, 'an operation that is undoing is never pruned'
    assert len(kept) == 6  # the live one plus the newest five

    for _ in range(2):
        env.label('gadget', crop_ids=['c1'])
    kept = sorted(p.name for p in env.state().iterdir() if p.is_dir())
    assert first not in kept, 'a dead undoing record is prunable once the undo lock is free'
    assert len(kept) == 5
    ids = [o['op_id'] for o in env.list_ops(limit=50).json()['items']]
    assert ids == sorted(ids, reverse=True)
