"""An image-level run starts the item level that follows it (the item level groups inside the
image level's leak groups)."""

from __future__ import annotations

import uuid

import psycopg
import pytest

from src.config.curation_accessors import get_curation_config
from src.db.repos import near_dup as repo
from src.db.session import async_session_factory
from src.schemas.curation import near_duplicates as wire
from src.services.near_duplicates import chain, status
from src.services.near_duplicates.build import full_build

from .engine import IMAGES, image_docs, make_env


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


def _jobs(pg, project):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        return c.execute(
            'SELECT kind, params, parent_job_id FROM jobs WHERE project_id = %s '
            "AND kind <> 'noop' ORDER BY created_at",
            (project,),
        ).fetchall()


def _pending(pg, project):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        return c.execute(
            'SELECT member_id, reason FROM near_dup_pending WHERE project_id = %s '
            "AND level = 'item' ORDER BY member_id",
            (project,),
        ).fetchall()


async def _build_item_run(env) -> None:
    """A current item-level run, so the item level counts as built."""
    async with env.scope() as session:
        run_id = await repo.start_run(
            session,
            project_id=env.project_id,
            level='item',
            mode='full',
            job_id=None,
            parent_run_id=None,
            profile=(env.profile.config_id, env.profile.revision),
            params={},
            space=env.plan.space_key,
            algorithm_version='nd-1',
        )
        run = await repo.get_run(session, run_id)
        await repo.flip_current(session, run, {})


async def test_a_full_image_build_starts_a_full_item_build(
    pg, project, job_env, producer, wakes
) -> None:
    docs = {IMAGES: image_docs({'g': ['a1', 'a2']}, ['s1'])}
    env, _ = await make_env(pg, project, docs)
    run = await full_build(env)

    started = await chain.after_image_run(env, run.run_id, touched=None)

    assert started is not None
    assert producer.sent() == [str(started)]
    ((kind, params, parent),) = _jobs(pg, project)
    assert (kind, params, parent) == ('near_dup_score', {'level': 'item'}, env.job_id)
    assert _pending(pg, project) == []


async def test_an_incremental_run_marks_the_items_of_the_regrouped_images(
    pg, project, job_env, producer, wakes
) -> None:
    items = get_curation_config().items_index
    docs = {
        IMAGES: image_docs({'g': ['a1', 'a2']}, ['s1']),
        items: {
            'i1': {'image_id': 'a1'},
            'i2': {'image_id': 'a1'},
            'i3': {'image_id': 'a2'},
            'i4': {'image_id': 's1'},
        },
    }
    env, _ = await make_env(pg, project, docs)
    run = await full_build(env)
    await _build_item_run(env)

    started = await chain.after_image_run(env, run.run_id, touched={'a1', 'a2'})

    assert started is not None
    assert _pending(pg, project) == [
        ('i1', 'group_changed'),
        ('i2', 'group_changed'),
        ('i3', 'group_changed'),
    ]
    ((kind, _, _),) = _jobs(pg, project)
    assert kind == 'near_dup_rescore'


async def test_an_item_level_never_built_gets_a_full_build_not_a_rescore(
    pg, project, job_env, producer, wakes
) -> None:
    docs = {IMAGES: image_docs({'g': ['a1', 'a2']}, [])}
    env, _ = await make_env(pg, project, docs)
    run = await full_build(env)

    await chain.after_image_run(env, run.run_id, touched={'a1'})

    ((kind, _, _),) = _jobs(pg, project)
    assert kind == 'near_dup_score'


async def test_a_busy_item_level_is_not_an_error_and_the_start_is_idempotent(
    pg, project, job_env, producer, wakes
) -> None:
    docs = {IMAGES: image_docs({'g': ['a1', 'a2']}, [])}
    env, _ = await make_env(pg, project, docs)
    run = await full_build(env)
    async with async_session_factory()() as session:
        await status.start_run(
            session, project, wire.NearDupRunRequest(level='item', mode='full'), 'tester'
        )

    assert await chain.after_image_run(env, run.run_id, touched=None) is None
    assert len(_jobs(pg, project)) == 1


async def test_the_item_level_does_not_chain(pg, project, job_env, producer, wakes) -> None:
    items = get_curation_config().items_index
    env, _ = await make_env(pg, project, {items: {}}, level='item', index=items)
    assert await chain.after_image_run(env, uuid.uuid4(), touched=None) is None
    assert _jobs(pg, project) == []
