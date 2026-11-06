"""Bulk label and move are recorded as operations: tagged items, a stored record,
the list route and retention."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest
from _lease_hold import held

from curation.label_ops_env import build_env, proposal
from src.domain.history import find_undo_snapshot
from src.domain.label_ops import LABEL_OP_ID_RE
from src.services.curation.label_ops import store


if TYPE_CHECKING:
    from pathlib import Path

    from curation.label_ops_env import OpsEnv


pytest_plugins = ['_records_world']
pytestmark = [pytest.mark.db, pytest.mark.usefixtures('world')]


def _members(op_id: str) -> dict[str, Any]:
    """The recorded members of an operation with the state each one had before it."""
    import asyncio

    from src.services.curation.label_ops import store

    async def read() -> dict[str, Any]:
        out: dict[str, Any] = {}
        async for page in store.member_pages(await store.read_op(op_id)):
            out.update({m.item_id: m.prior for m in page})
        return out

    return asyncio.run(read())


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

    record = env.record(op_id)
    assert record['status'] == 'written'
    assert (record['kind'], record['count'], record['conflicts'], record['selected']) == (
        'label',
        3,
        0,
        3,
    )
    members = _members(op_id)
    assert sorted(members) == ['c0', 'c1', 'c2']
    assert all(p['op_id'] == op_id and p['restorable'] is True for p in members.values())
    assert members['c0']['class_validated'] is False, 'the state the label replaced'


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
    assert env.row_count() == 0
    assert env.list_ops().json()['items'] == []
    assert env.fake.write_calls == 0


def test_a_write_without_a_record_never_happens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)

    async def refuse(_new: Any) -> Any:
        raise RuntimeError('database down')

    monkeypatch.setattr('src.services.curation.label_ops.record.create_op', refuse)
    resp = env.client.put(
        f'{env.prefix}/crops/batch_label',
        json={'class_id': env.class_ids['widget'], 'crop_ids': ['c0']},
    )
    assert resp.status_code == 500, resp.text
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
    assert body['next_after'] is None


def test_interrupted_and_writing_operations_are_served_and_not_undoable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    op_id = env.label('widget', crop_ids=['c0'])['op_id']
    env.set_status(op_id, 'writing')
    item = env.list_ops().json()['items'][0]
    assert item['status'] == 'interrupted'  # no live lease guards the write
    assert item['undoable'] is True

    with held('label_op', op_id):
        live = env.list_ops().json()['items'][0]
    assert live['status'] == 'writing'
    assert live['undoable'] is False


def test_retention_keeps_the_newest_200_and_never_prunes_a_live_op(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path, n=3)
    monkeypatch.setattr(store, 'LABEL_OPS_KEEP', 5)
    first = env.label('widget', crop_ids=['c0'])['op_id']
    env.set_status(first, 'undoing')
    for _ in range(7):
        env.label('gadget', crop_ids=['c1'])
    kept = [o['op_id'] for o in env.list_ops(limit=50).json()['items']]
    assert first in kept, 'an operation that is undoing is never pruned'
    assert len(kept) == 6  # the undoing one plus the newest five

    for _ in range(2):
        env.label('gadget', crop_ids=['c1'])
    kept = [o['op_id'] for o in env.list_ops(limit=50).json()['items']]
    assert first in kept, (
        'an undoing record is not pruned, even a dead one: its next undo cleans it'
    )
    ids = [o['op_id'] for o in env.list_ops(limit=50).json()['items']]
    assert ids == sorted(ids, reverse=True)


def _ops(env: OpsEnv, n: int) -> list[str]:
    """``n`` operations on distinct crops, oldest first."""
    return [env.label('widget', crop_ids=[f'c{i}'])['op_id'] for i in range(n)]


def _page(env: OpsEnv, **params: Any) -> dict[str, Any]:
    resp = env.list_ops(**params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_label_ops_list_pages_with_after_to_the_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    _ops(env, 5)
    full = [o['op_id'] for o in _page(env, limit=50)['items']]
    one = _page(env, limit=2)
    assert [o['op_id'] for o in one['items']] == full[:2]
    assert one['next_after'] == full[1]
    two = _page(env, limit=2, after=one['next_after'])
    assert [o['op_id'] for o in two['items']] == full[2:4]
    assert two['next_after'] == full[3]
    three = _page(env, limit=2, after=two['next_after'])
    assert [o['op_id'] for o in three['items']] == full[4:]
    assert three['next_after'] is None
    paged = [o['op_id'] for pg in (one, two, three) for o in pg['items']]
    assert paged == full
    assert len(set(paged)) == 5


def test_label_ops_last_full_page_has_no_next_after(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    ids = _ops(env, 4)
    first = _page(env, limit=2)
    assert first['next_after'] is not None
    last = _page(env, limit=2, after=first['next_after'])
    assert len(last['items']) == 2
    assert last['next_after'] is None
    past = _page(env, after=ids[0])
    assert past['items'] == []
    assert past['next_after'] is None


def test_label_ops_same_timestamp_is_ordered_by_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    monkeypatch.setattr(store, 'now_iso', lambda: '2026-10-31T00:00:00+00:00')
    a, b = _ops(env, 2)
    assert [o['op_id'] for o in _page(env)['items']] == [b, a]
    assert {o['at'] for o in _page(env)['items']} == {'2026-10-31T00:00:00+00:00'}
    first = _page(env, limit=1)
    assert ([o['op_id'] for o in first['items']], first['next_after']) == ([b], b)
    rest = _page(env, limit=1, after=b)
    assert ([o['op_id'] for o in rest['items']], rest['next_after']) == ([a], None)


def test_label_ops_malformed_after_is_422(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = _env(monkeypatch, tmp_path)
    _ops(env, 1)
    for bad in ('x', '../decoy', 'lop_20261031T000000000000_00000a', str(uuid.uuid4()).upper()):
        resp = env.list_ops(after=bad)
        assert (resp.status_code, resp.json()['detail']['error']) == (422, 'invalid_request'), bad


def test_label_ops_pruned_cursor_still_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    oldest, middle, _newest = _ops(env, 3)
    _delete(middle)
    body = _page(env, after=middle)
    assert [o['op_id'] for o in body['items']] == [oldest]
    assert body['next_after'] is None


def _delete(op_id: str) -> None:
    import asyncio

    from sqlalchemy import text

    from src.config.project_context import current_project_id
    from src.db.session import async_session_factory, project_tx

    async def drop() -> None:
        async with async_session_factory()() as session, project_tx(session, current_project_id()):
            await session.execute(
                text('DELETE FROM label_operations WHERE id = :id'), {'id': uuid.UUID(op_id)}
            )

    asyncio.run(drop())


def test_label_ops_undoable_paging_skips_non_undoable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    ids = _ops(env, 4)  # oldest first; take over c0 and c2 so those two ops have nothing left
    for crop in ('c0', 'c2'):
        resp = env.client.put(
            f'{env.prefix}/crops/{crop}/label', json={'class_id': env.class_ids['gizmo']}
        )
        assert resp.status_code == 200, resp.text
    served: list[str] = []
    after: str | None = None
    for _ in range(5):
        params: dict[str, Any] = {'undoable': 'true', 'limit': 1}
        if after:
            params['after'] = after
        body = _page(env, **params)
        served += [o['op_id'] for o in body['items']]
        after = body['next_after']
        if after is None:
            break
    assert served == [ids[3], ids[1]]
    assert after is None
