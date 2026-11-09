"""A freeze or a training export refuses (409 ``near_dup_stale``) while the groups are stale."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.services.near_duplicates import holdout, status
from src.services.near_duplicates.build import full_build
from src.services.near_duplicates.pending import enqueue_pending

from .engine import IMAGES, image_docs, make_env, plan


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def _space(monkeypatch: pytest.MonkeyPatch) -> None:
    async def resolved(_target, _key=None):
        return SimpleNamespace(space_key=plan().space_key)

    monkeypatch.setattr(status, 'embedding_space', resolved)


async def _refusal(project) -> dict:
    async with async_session_factory()() as session:
        with pytest.raises(RequestRejectedError) as raised:
            await holdout.require_fresh(session, project)
    assert raised.value.status_code == 409
    return raised.value.detail


async def _build_both_levels(pg, project) -> None:
    env, _ = await make_env(pg, project, {IMAGES: image_docs({'g': ['a1', 'a2']}, ['s1'])})
    await full_build(env)
    # an empty item level that is current, so only the case under test is stale
    from .test_item_level_chain import _build_item_run

    await _build_item_run(env)


async def test_a_project_with_no_run_is_refused_and_names_the_levels(project) -> None:
    detail = await _refusal(project)
    assert detail['code'] == 'near_dup_stale'
    assert {(lv['level'], tuple(lv['stale_reasons'])) for lv in detail['levels']} == {
        ('image', ('no_run',)),
        ('item', ('no_run',)),
    }


async def test_pending_rows_refuse(pg, project) -> None:
    await _build_both_levels(pg, project)
    env, _ = await make_env(pg, project, {IMAGES: {}})
    async with env.scope() as session:
        await enqueue_pending(session, project, 'image', ['a1'], 'vector_changed')
    detail = await _refusal(project)
    assert detail['levels'] == [
        {'level': 'image', 'stale_reasons': ['pending'], 'pending_count': 1}
    ]


async def test_a_fresh_project_passes(pg, project) -> None:
    await _build_both_levels(pg, project)
    async with async_session_factory()() as session:
        await holdout.require_fresh(session, project)
