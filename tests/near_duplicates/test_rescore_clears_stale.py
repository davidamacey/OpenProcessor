"""After a rescore that ungroups an item, no near_dup field remains on it (#473 part a)."""

from __future__ import annotations

import numpy as np
import pytest

from src.services.near_duplicates.build import full_build
from src.services.near_duplicates.pending import enqueue_pending
from src.services.near_duplicates.projection import FIELDS, apply_run
from src.services.near_duplicates.rescore import rescore

from .engine import IMAGES, image_docs, make_env, snapshot


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


def _fields(doc: dict) -> set[str]:
    return {f for f in FIELDS if f in doc}


async def test_an_ungrouped_member_loses_every_field_and_a_new_group_gets_a_new_id(
    pg, project
) -> None:
    docs = {IMAGES: image_docs({'g': ['a1', 'a2', 'a3'], 'h': ['b1', 'b2']}, ['s1', 's2'])}
    env, _ = await make_env(pg, project, docs)
    first = await full_build(env)
    await apply_run(env, first.run_id, None)
    assert _fields(docs[IMAGES]['a3']) == set(FIELDS)
    old_ids = {d.get('near_dup_group_id') for d in docs[IMAGES].values()} - {None}
    assert len(old_ids) == 2

    docs[IMAGES]['a3']['pe_embedding'] = docs[IMAGES]['s1']['pe_embedding']  # now a lone copy of s1
    docs[IMAGES]['s2']['pe_embedding'] = docs[IMAGES]['s1']['pe_embedding']  # and s2 joins it
    async with env.scope() as session:
        await enqueue_pending(session, project, 'image', ['a3', 's2'], 'vector_changed')
    outcome = await rescore(env)
    assert outcome.summary is not None
    result = await apply_run(env, outcome.summary.run_id, None)

    assert not result.skipped
    assert _fields(docs[IMAGES]['a3']) == _fields(docs[IMAGES]['s2'])  # a3 now sits with s1 and s2
    new_group = {frozenset(m) for m in snapshot(pg, outcome.summary.run_id, 'dedup')}
    assert frozenset({'a3', 's1', 's2'}) in new_group
    assert docs[IMAGES]['a3']['near_dup_group_id'] not in old_ids
    assert docs[IMAGES]['a3']['near_dup_run_id'] == str(outcome.summary.run_id)
    # b1/b2 were not touched: the incremental projection left them alone
    assert docs[IMAGES]['b1']['near_dup_run_id'] == str(first.run_id)


async def test_a_member_that_leaves_every_group_is_cleared(pg, project) -> None:
    docs = {IMAGES: image_docs({'g': ['a1', 'a2']}, ['s1'])}
    env, _ = await make_env(pg, project, docs)
    first = await full_build(env)
    await apply_run(env, first.run_id, None)
    docs[IMAGES]['a2']['pe_embedding'] = np.eye(32, dtype=np.float32)[0].tolist()
    async with env.scope() as session:
        await enqueue_pending(session, project, 'image', ['a2'], 'vector_changed')
    outcome = await rescore(env)
    await apply_run(env, outcome.summary.run_id, None)
    assert _fields(docs[IMAGES]['a1']) == set()
    assert _fields(docs[IMAGES]['a2']) == set()


async def test_an_older_run_cannot_overwrite_a_newer_projection(pg, project) -> None:
    docs = {IMAGES: image_docs({'g': ['a1', 'a2']}, ['s1'])}
    env, _ = await make_env(pg, project, docs)
    first = await full_build(env)
    second = await full_build(env)
    skipped = await apply_run(env, first.run_id, None)
    assert skipped.skipped
    assert not (await apply_run(env, second.run_id, None)).skipped
