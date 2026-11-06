"""A human decision binds every later run: 'different' splits, 'same' joins."""

from __future__ import annotations

import pytest

from src.services.near_duplicates.build import full_build, group_stored

from .engine import IMAGES, image_docs, make_env, snapshot


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


async def _decide(env, a: str, b: str, decision: str) -> None:
    from src.db.models.near_dup import NearDupDecision

    lo, hi = sorted((a, b))
    async with env.scope() as session:
        session.add(
            NearDupDecision(
                project_id=env.project_id,
                level='image',
                a_id=lo,
                b_id=hi,
                decision=decision,
                decided_by='t',
            )
        )


async def test_different_splits_and_same_joins_in_a_full_build_and_a_regroup(pg, project) -> None:
    docs = {IMAGES: image_docs({'g': ['a1', 'a2'], 'h': ['b1', 'b2']}, ['s1', 's2'])}
    env, _ = await make_env(pg, project, docs)
    await _decide(env, 'a1', 'a2', 'different')
    await _decide(env, 's1', 's2', 'same')

    built = await full_build(env)
    assert {frozenset(m) for m in snapshot(pg, built.run_id, 'dedup')} == {
        frozenset({'b1', 'b2'}),
        frozenset({'s1', 's2'}),
    }
    regrouped = await group_stored(env, scored=0)
    assert {frozenset(m) for m in snapshot(pg, regrouped.run_id, 'dedup')} == {
        frozenset({'b1', 'b2'}),
        frozenset({'s1', 's2'}),
    }
