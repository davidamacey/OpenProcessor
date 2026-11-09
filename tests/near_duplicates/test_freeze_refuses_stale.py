"""The ``near_dup_stale`` gate (owner decision 2026-11-08): it applies only once a holdout is frozen.

Before any freeze an export is never refused for near-duplicate state; the first freeze needs a
current item-level run to pull in (and is refused without one); a re-freeze is refused while stale.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.services.near_duplicates import holdout, status
from src.services.near_duplicates.build import full_build
from src.services.near_duplicates.pending import enqueue_pending

from .engine import IMAGES, image_docs, make_env, plan
from .test_holdout_near_duplicates import _make_freeze


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def _space(monkeypatch: pytest.MonkeyPatch) -> None:
    async def resolved(_target, _key=None):
        return SimpleNamespace(space_key=plan().space_key)

    monkeypatch.setattr(status, 'embedding_space', resolved)


async def _refusal(project, check=None) -> dict:
    check = check or holdout.require_fresh_when_frozen
    async with async_session_factory()() as session:
        with pytest.raises(RequestRejectedError) as raised:
            await check(session, project)
    assert raised.value.status_code == 409
    return raised.value.detail


async def _build_both_levels(pg, project) -> None:
    env, _ = await make_env(pg, project, {IMAGES: image_docs({'g': ['a1', 'a2']}, ['s1'])})
    await full_build(env)
    # an empty item level that is current, so only the case under test is stale
    from .test_item_level_chain import _build_item_run

    await _build_item_run(env)


async def test_a_project_with_no_run_is_refused_and_names_the_levels(pg, project) -> None:
    _make_freeze(pg, project)
    detail = await _refusal(project)
    assert detail['code'] == 'near_dup_stale'
    assert {(lv['level'], tuple(lv['stale_reasons'])) for lv in detail['levels']} == {
        ('image', ('no_run',)),
        ('item', ('no_run',)),
    }


async def _enqueue_pending(pg, project) -> None:
    env, _ = await make_env(pg, project, {IMAGES: {}})
    async with env.scope() as session:
        await enqueue_pending(session, project, 'image', ['a1'], 'vector_changed')


async def test_pending_rows_refuse(pg, project) -> None:
    await _build_both_levels(pg, project)
    _make_freeze(pg, project)
    await _enqueue_pending(pg, project)
    detail = await _refusal(project)
    assert detail['levels'] == [
        {'level': 'image', 'stale_reasons': ['pending'], 'pending_count': 1}
    ]


async def test_a_fresh_project_passes(pg, project) -> None:
    await _build_both_levels(pg, project)
    _make_freeze(pg, project)
    async with async_session_factory()() as session:
        await holdout.require_fresh_when_frozen(session, project)


async def test_before_any_freeze_an_export_is_never_refused(project) -> None:
    async with async_session_factory()() as session:
        await holdout.require_fresh_when_frozen(session, project)


async def test_the_first_freeze_needs_a_current_item_run(project) -> None:
    detail = await _refusal(project, holdout.require_freezable)
    assert detail['code'] == 'near_dup_stale'
    assert [lv['level'] for lv in detail['levels']] == ['item']
    assert detail['levels'][0]['stale_reasons'] == ['no_run']


async def test_the_first_freeze_runs_on_stale_but_present_groups(pg, project) -> None:
    await _build_both_levels(pg, project)
    await _enqueue_pending(pg, project)
    async with async_session_factory()() as session:
        await holdout.require_freezable(session, project)


async def test_a_refreeze_is_refused_while_stale(pg, project) -> None:
    await _build_both_levels(pg, project)
    _make_freeze(pg, project)
    await _enqueue_pending(pg, project)
    detail = await _refusal(project, holdout.require_freezable)
    assert detail['levels'][0]['stale_reasons'] == ['pending']
