"""The freeze pulls the near-duplicates of a held-out item into the holdout with it."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest

from src.core.exceptions import RequestRejectedError
from src.services.near_duplicates import holdout
from src.services.near_duplicates.build import full_build

from .engine import IMAGES, basis, make_env, near
from .fake_opensearch import FakeNearDupOpenSearch


pytestmark = [pytest.mark.db, pytest.mark.asyncio]

ITEMS = 'op_prj_nd__items'


def _item(image: str, vector: list[float]) -> dict[str, Any]:
    return {'pe_embedding': vector, 'image_id': image, 'detector_class_name': 'car'}


def _docs() -> dict[str, dict[str, Any]]:
    d = basis(3)
    return {
        # i1 (validated, selected) and i2 are crops of one vehicle seen in two frames
        'i1': _item('a', near(d[0], 1)),
        'i2': _item('b', near(d[0], 2)),
        'i3': _item('c', near(d[0], 3)),
        # i4 and i5 are near-duplicates of each other, neither selected
        'i4': _item('d', near(d[1], 4)),
        'i5': _item('e', near(d[1], 5)),
        'i6': _item('f', d[2].tolist()),
    }


class _SeqNoOpenSearch(FakeNearDupOpenSearch):
    """The stand-in with the ``_seq_no`` an OCC write reads, and a conflict switch."""

    conflict: set[str] = set()

    async def mget(
        self, *, body: dict[str, Any], index: str | None = None, **kw: Any
    ) -> dict[str, Any]:
        if 'ids' in body:
            return await super().mget(index=index or '', body=body, **kw)
        out = []
        for ref in body['docs']:
            source = self.docs[ref['_index']].get(ref['_id'])
            out.append(
                {'_id': ref['_id'], 'found': source is not None, '_source': source,
                 '_seq_no': 1, '_primary_term': 1}
            )  # fmt: skip
        return {'docs': out}

    async def bulk(self, *, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        keep, items = [], []
        for action, payload in zip(body[::2], body[1::2], strict=True):
            if action['update']['_id'] in self.conflict:
                items.append({'update': {'status': 409}})
            else:
                keep += [action, payload]
        done = await super().bulk(body=keep, **kw)
        return {'items': items + done['items']}


async def _built(pg, project):
    # the item level groups inside the image level's leak groups: build the images first
    images = {
        i: {'pe_embedding': d['pe_embedding']}
        for i, d in zip('abcdef', _docs().values(), strict=True)
    }
    image_env, _ = await make_env(pg, project, {IMAGES: images})
    await full_build(image_env)
    env, _ = await make_env(pg, project, {ITEMS: _docs()}, level='item', index=ITEMS)
    await full_build(env)
    return env


async def test_a_near_duplicate_of_a_selected_item_is_held_out_and_counted(pg, project) -> None:
    env = await _built(pg, project)
    async with env.scope() as session:
        closure = await holdout.leak_closure(session, project, {'i1'})

    assert closure.moved_ids == ['i2', 'i3']
    assert {m.anchor_item_id for m in closure.moves} == {'i1'}
    # the other pair has no selected member: unchanged
    assert not {'i4', 'i5', 'i6'} & set(closure.moved_ids)


async def test_the_moves_and_the_freeze_record_are_written(pg, project) -> None:
    env = await _built(pg, project)
    async with env.scope() as session:
        closure = await holdout.leak_closure(session, project, {'i1'})
        freeze_id = _make_freeze(pg, project)
        await holdout.record_closure(session, project, freeze_id, closure)

    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        rows = conn.execute(
            'SELECT item_id, anchor_item_id, freeze_id, run_id FROM holdout_near_dup_moves '
            'WHERE project_id = %s ORDER BY item_id',
            (project,),
        ).fetchall()
        freeze = conn.execute(
            'SELECT moved_by_near_dup, near_dup_item_run_id, near_dup_image_run_id '
            'FROM holdout_freezes WHERE id = %s',
            (freeze_id,),
        ).fetchone()
    assert [(r[0], r[1]) for r in rows] == [('i2', 'i1'), ('i3', 'i1')]
    assert {r[2] for r in rows} == {freeze_id}
    assert freeze[0] == 2
    assert freeze[1] == closure.item_run_id


async def test_an_item_without_a_near_duplicate_moves_nothing(pg, project) -> None:
    env = await _built(pg, project)
    async with env.scope() as session:
        closure = await holdout.leak_closure(session, project, {'i6'})
    assert closure.moves == ()


async def test_a_project_without_an_item_run_moves_nothing(project, pg_registry) -> None:
    from .conftest import scope_for

    async with scope_for(project)() as session:
        closure = await holdout.leak_closure(session, project, {'i1'})
    assert closure.moves == ()
    assert closure.item_run_id is None


async def test_the_flag_goes_through_the_occ_writer(pg, project) -> None:
    client = _SeqNoOpenSearch({ITEMS: _docs()})
    await holdout.flag_held_out(client, ITEMS, ['i1', 'i2'])
    assert client.docs[ITEMS]['i1']['test_holdout'] is True
    assert client.docs[ITEMS]['i2']['test_holdout'] is True
    assert 'test_holdout' not in client.docs[ITEMS]['i3']


async def test_a_conflicting_write_refuses_the_freeze() -> None:
    client = _SeqNoOpenSearch({ITEMS: _docs()})
    client.conflict = {'i2'}
    with pytest.raises(RequestRejectedError) as raised:
        await holdout.flag_held_out(client, ITEMS, ['i1', 'i2'])
    assert raised.value.status_code == 502


def _make_freeze(pg, project):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        return conn.execute(
            'INSERT INTO holdout_freezes (project_id, is_current, source, item_count, frozen_by) '
            "VALUES (%s, true, 'manual', 3, 'test') RETURNING id",
            (project,),
        ).fetchone()[0]
