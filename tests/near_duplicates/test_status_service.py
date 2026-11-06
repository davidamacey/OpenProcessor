"""The near-duplicate reads and writes behind the routes: status, runs, groups, decisions."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import psycopg
import pytest

from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.schemas.curation import near_duplicates as wire
from src.services.near_duplicates import status as service
from src.services.near_duplicates.build import full_build
from src.services.near_duplicates.pending import enqueue_pending

from .conftest import make_project
from .engine import IMAGES, image_docs, make_env, plan


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def _space(monkeypatch: pytest.MonkeyPatch) -> None:
    async def resolved(_target, _key=None):
        return SimpleNamespace(space_key=plan().space_key)

    monkeypatch.setattr(service, 'embedding_space', resolved)


async def _session():
    return async_session_factory()()


async def _reasons(project, level='image'):
    async with async_session_factory()() as session:
        status = await service.status(session, project)
    return next(lv for lv in status.levels if lv.level == level)


async def test_a_new_project_asks_for_a_first_build(project) -> None:
    async with async_session_factory()() as session:
        status = await service.status(session, project)
    assert status.profile is not None
    assert status.profile.revision == 1
    assert [(lv.level, lv.stale_reasons, lv.pending_count) for lv in status.levels] == [
        ('image', ['no_run'], 0),
        ('item', ['no_run'], 0),
    ]


async def test_status_follows_pending_rows_and_a_changed_space(pg, project, monkeypatch) -> None:
    docs = {IMAGES: image_docs({'g': ['a1', 'a2']}, ['s1'])}
    env, _ = await make_env(pg, project, docs)
    await full_build(env)
    assert (await _reasons(project)).stale_reasons == []
    async with env.scope() as session:
        await enqueue_pending(session, project, 'image', ['a1'], 'vector_changed')
    level = await _reasons(project)
    assert (level.stale_reasons, level.pending_count) == (['pending'], 1)

    async def other(_target, _key=None):
        return SimpleNamespace(space_key='some_other_space')

    monkeypatch.setattr(service, 'embedding_space', other)
    assert (await _reasons(project)).stale_reasons == ['rebuild_required', 'pending']


async def test_a_run_is_a_job_and_a_second_start_is_busy(
    pg, project, job_env, producer, wakes
) -> None:
    body = wire.NearDupRunRequest(level='image', mode='full')
    async with async_session_factory()() as session:
        started = await service.start_run(session, project, body, 'tester')
    assert started.kind == 'near_dup_score'
    assert producer.sent() == [started.job_id]
    with pytest.raises(RequestRejectedError) as busy:
        async with async_session_factory()() as session:
            await service.start_run(
                session,
                project,
                wire.NearDupRunRequest(level='image', mode='incremental'),
                'tester',
            )
    assert (busy.value.status_code, busy.value.detail['error']) == (409, 'job_busy')


async def test_groups_page_by_size_and_open_by_lineage(pg, project) -> None:
    docs = {IMAGES: image_docs({'big': ['a1', 'a2', 'a3'], 'small': ['b1', 'b2']}, ['s1'])}
    env, _ = await make_env(pg, project, docs)
    run = await full_build(env)
    async with async_session_factory()() as session:
        first = await service.groups(
            session, project, level='image', purpose='dedup', min_size=2, has_holdout=None,
            cursor=None, limit=1,
        )  # fmt: skip
    assert [g.size for g in first.items] == [3]
    assert first.next_cursor is not None
    async with async_session_factory()() as session:
        second = await service.groups(
            session, project, level='image', purpose='dedup', min_size=2, has_holdout=None,
            cursor=first.next_cursor, limit=1,
        )  # fmt: skip
    assert [g.size for g in second.items] == [2]
    assert second.next_cursor is None
    async with async_session_factory()() as session:
        detail = await service.group(session, project, 'image', uuid.UUID(first.items[0].group_id))
    assert {m.member_id for m in detail.members} == {'a1', 'a2', 'a3'}
    assert sum(m.is_representative for m in detail.members) == 1
    async with async_session_factory()() as session:
        listed = await service.runs(session, project, 'image', 10)
        one = await service.run(session, project, run.run_id)
    assert [r.id for r in listed.items] == [str(run.run_id)]
    assert one.dedup_groups == 2


async def test_another_projects_runs_and_groups_are_not_found(pg, project) -> None:
    docs = {IMAGES: image_docs({'g': ['a1', 'a2']}, ['s1'])}
    env, _ = await make_env(pg, project, docs)
    run = await full_build(env)
    other = await make_project()
    async with async_session_factory()() as session:
        with pytest.raises(RequestRejectedError) as missing:
            await service.run(session, other, run.run_id)
    assert missing.value.status_code == 404
    async with async_session_factory()() as session:
        page = await service.groups(
            session, other, level='image', purpose='dedup', min_size=2, has_holdout=None,
            cursor=None, limit=10,
        )  # fmt: skip
    assert page.items == []


async def test_a_bad_cursor_is_422(project) -> None:
    async with async_session_factory()() as session:
        with pytest.raises(RequestRejectedError) as bad:
            await service.groups(
                session, project, level='image', purpose='dedup', min_size=2, has_holdout=None,
                cursor='not-a-cursor', limit=10,
            )  # fmt: skip
    assert bad.value.status_code == 422


async def test_a_decision_is_stored_once_and_marks_both_members(pg, project) -> None:
    body = wire.NearDupDecisionIn(level='image', a_id='z', b_id='a', decision='different')
    async with async_session_factory()() as session:
        first = await service.decide(session, project, body, 'tester')
    async with async_session_factory()() as session:
        again = await service.decide(
            session, project, body.model_copy(update={'decision': 'same'}), 'tester'
        )
    assert (first.a_id, first.b_id) == ('a', 'z')
    assert again.id == first.id
    assert again.decision == 'same'
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        pend = c.execute(
            'SELECT member_id FROM near_dup_pending WHERE project_id = %s ORDER BY member_id',
            (project,),
        ).fetchall()
    assert pend == [('a',), ('z',)]
    async with async_session_factory()() as session:
        await service.undecide(session, project, uuid.UUID(first.id))
    async with async_session_factory()() as session:
        with pytest.raises(RequestRejectedError) as gone:
            await service.undecide(session, project, uuid.UUID(first.id))
    assert gone.value.status_code == 404


def test_a_decision_needs_two_different_members() -> None:
    with pytest.raises(ValueError, match='two different members'):
        wire.NearDupDecisionIn(level='item', a_id='x', b_id='x', decision='same')
