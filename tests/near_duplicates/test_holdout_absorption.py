"""After a freeze, an item-level run moves the rest of a mixed leak group into the holdout."""

from __future__ import annotations

import dataclasses

import psycopg
import pytest

from src.db.repos import near_dup as repo
from src.services.near_duplicates.build import full_build

from .test_holdout_near_duplicates import ITEMS, _built, _make_freeze, _SeqNoOpenSearch


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


async def _rebuild_after_freeze(pg, project, held: str):
    env = await _built(pg, project)
    _make_freeze(pg, project)
    docs = env.client.docs  # type: ignore[attr-defined]
    docs[ITEMS][held]['test_holdout'] = True
    client = _SeqNoOpenSearch(docs)
    summary = await full_build(dataclasses.replace(env, client=client))  # type: ignore[arg-type]
    return env, client, summary


def _rows(pg, project):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        moves = conn.execute(
            'SELECT item_id, anchor_item_id, run_id FROM holdout_near_dup_moves '
            'WHERE project_id = %s ORDER BY item_id',
            (project,),
        ).fetchall()
        events = conn.execute(
            "SELECT count(*) FROM events WHERE type = 'holdout.near_dup_absorbed' "
            'AND project_id = %s',
            (project,),
        ).fetchone()[0]
    return moves, events


async def test_a_mixed_leak_group_is_absorbed_into_the_holdout(pg, project) -> None:
    env, client, summary = await _rebuild_after_freeze(pg, project, held='i1')

    assert summary.holdout_absorbed == 2
    assert client.docs[ITEMS]['i2']['test_holdout'] is True
    assert client.docs[ITEMS]['i3']['test_holdout'] is True
    assert 'test_holdout' not in client.docs[ITEMS]['i4']
    moves, events = _rows(pg, project)
    assert [(m[0], m[1]) for m in moves] == [('i2', 'i1'), ('i3', 'i1')]
    assert {m[2] for m in moves} == {summary.run_id}
    assert events == 1
    async with env.scope() as session:
        run = await repo.get_run(session, summary.run_id)
    assert run is not None
    assert run.holdout_absorbed == 2


async def test_a_run_with_no_mixed_group_moves_nothing(pg, project) -> None:
    _, client, summary = await _rebuild_after_freeze(pg, project, held='i6')

    assert summary.holdout_absorbed == 0
    assert not [i for i, d in client.docs[ITEMS].items() if d.get('test_holdout') and i != 'i6']
    assert _rows(pg, project) == ([], 0)


async def test_a_project_without_a_freeze_absorbs_nothing(pg, project) -> None:
    env = await _built(pg, project)
    env.client.docs[ITEMS]['i1']['test_holdout'] = True  # type: ignore[attr-defined]
    summary = await full_build(dataclasses.replace(env, client=_SeqNoOpenSearch(env.client.docs)))  # type: ignore[arg-type, attr-defined]
    assert summary.holdout_absorbed == 0
