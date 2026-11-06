"""An incremental run equals a full build on the same state (risk R4), and keeps untouched groups."""

from __future__ import annotations

import pytest

from src.services.near_duplicates.build import full_build
from src.services.near_duplicates.pending import enqueue_pending
from src.services.near_duplicates.rescore import rescore

from .engine import IMAGES, image_docs, make_env, near, snapshot


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


def _members(snap):
    return set(snap)


async def test_incremental_equals_full(pg, project) -> None:
    groups = {
        'g1': ['i01', 'i02', 'i03', 'i04', 'i05'],
        'g2': ['i06', 'i07', 'i08'],
        'g3': ['i09', 'i10'],
        'g4': ['i11', 'i12'],  # untouched below
    }
    singles = [f's{n:02d}' for n in range(15)]
    docs = {IMAGES: image_docs(groups, singles)}
    env, _ = await make_env(pg, project, docs)
    first = await full_build(env)
    before = snapshot(pg, first.run_id, 'dedup')
    assert _members(before) == {frozenset(m) for m in groups.values()}

    # changes: two newcomers join g1, g2 loses a member, g3 splits apart, a new pair appears
    store = docs[IMAGES]
    direction_g1 = store['i01']['pe_embedding']
    store['n1'] = {'pe_embedding': near(__import__('numpy').asarray(direction_g1), 101)}
    store['n2'] = {'pe_embedding': near(__import__('numpy').asarray(direction_g1), 102)}
    del store['i08']
    store['i10']['pe_embedding'] = store['s00']['pe_embedding']  # now identical to a single
    store['p1'] = {
        'pe_embedding': near(__import__('numpy').asarray(store['s05']['pe_embedding']), 103)
    }
    changed = ['n1', 'n2', 'i08', 'i10', 'p1']
    async with env.scope() as session:
        await enqueue_pending(session, project, 'image', changed[:3], 'added')
        await enqueue_pending(session, project, 'image', ['i08'], 'deleted')
        await enqueue_pending(session, project, 'image', ['i10', 'p1'], 'vector_changed')

    outcome = await rescore(env)
    assert outcome.summary is not None
    assert outcome.claimed == 5
    incremental = {p: snapshot(pg, outcome.summary.run_id, p) for p in ('dedup', 'leak')}

    full = await full_build(env)
    rebuilt = {p: snapshot(pg, full.run_id, p) for p in ('dedup', 'leak')}
    for purpose in ('dedup', 'leak'):
        assert {m: rep for m, (rep, _) in incremental[purpose].items()} == {
            m: rep for m, (rep, _) in rebuilt[purpose].items()
        }
    # the pending rows were drained, and the untouched group g4 kept its lineage id
    assert incremental['dedup'][frozenset(groups['g4'])][1] == before[frozenset(groups['g4'])][1]
    assert rebuilt['dedup'][frozenset(groups['g4'])][1] == before[frozenset(groups['g4'])][1]


async def test_a_group_that_gains_members_keeps_its_lineage(pg, project) -> None:
    docs = {IMAGES: image_docs({'g': ['a1', 'a2', 'a3']}, ['s1', 's2'])}
    env, _ = await make_env(pg, project, docs)
    first = await full_build(env)
    lineage = next(iter(snapshot(pg, first.run_id, 'dedup').values()))[1]
    import numpy as np

    docs[IMAGES]['a4'] = {'pe_embedding': near(np.asarray(docs[IMAGES]['a1']['pe_embedding']), 9)}
    async with env.scope() as session:
        await enqueue_pending(session, project, 'image', ['a4'], 'added')
    outcome = await rescore(env)
    ((members, (_, carried)),) = snapshot(pg, outcome.summary.run_id, 'dedup').items()
    assert members == {'a1', 'a2', 'a3', 'a4'}
    assert carried == lineage
