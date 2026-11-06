"""Stream-worker replicas share the projects: fair shares, one holder each, takeover (3A-10)."""

from __future__ import annotations

import uuid

import pytest
from fixtures.job_rows import make_project, sql

from src.services.jobs.stream_claims import Candidate, ProjectClaims


pytestmark = pytest.mark.db

KIND = 'detection'


@pytest.fixture
def env(pg, pg_env):
    sql(pg, 'DELETE FROM stream_worker_claims')
    sql(pg, "DELETE FROM worker_capabilities WHERE queue IN ('detection', 'vlm')")
    yield pg
    import asyncio

    from src.db.engine import dispose_engines

    asyncio.run(dispose_engines())


def projects(env, count: int) -> list[Candidate]:
    return [
        Candidate(make_project(env, f'sc{uuid.uuid4().hex[:6]}'), f'p{i}') for i in range(count)
    ]


def replica(host: str, **kwargs) -> ProjectClaims:
    return ProjectClaims(KIND, host=host, **kwargs)


def holders(env) -> dict[uuid.UUID, str]:
    rows = sql(
        env, 'SELECT project_id, worker_id FROM stream_worker_claims WHERE worker_kind = %s', KIND
    )
    return dict(rows)


def tokens(env) -> dict[uuid.UUID, int]:
    return dict(sql(env, 'SELECT project_id, fencing_token FROM stream_worker_claims'))


def test_one_worker_with_three_projects_holds_all_of_them(env) -> None:
    offered = projects(env, 3)
    only = replica('w1')
    held = only.run_pass(offered, idle_slugs=[c.slug for c in offered])
    assert set(held) == {c.project_id for c in offered}
    assert set(holders(env).values()) == {'detection@w1'}  # as the round-robin did, over everything


def test_a_second_worker_gets_projects_within_two_passes(env) -> None:
    offered = projects(env, 3)
    idle = [c.slug for c in offered]
    first, second = replica('w1'), replica('w2')
    first.run_pass(offered, idle)
    assert len(first.held) == 3

    second.run_pass(offered, idle)  # registers; nothing is free yet
    assert len(second.held) == 0
    first.run_pass(offered, idle)  # sees two live replicas: fair = 2, releases one
    assert len(first.held) == 2
    second.run_pass(offered, idle)  # takes the released project
    assert len(second.held) == 1

    owners = holders(env)
    assert len(owners) == 3  # each project held by exactly one worker
    assert sorted(owners.values()) == ['detection@w1', 'detection@w1', 'detection@w2']
    assert set(first.held) | set(second.held) == {c.project_id for c in offered}
    assert not set(first.held) & set(second.held)


def test_a_project_with_work_in_flight_is_not_released(env) -> None:
    offered = projects(env, 2)
    first, second = replica('w1'), replica('w2')
    first.run_pass(offered, [])
    second.run_pass(offered, [])
    first.run_pass(offered, idle_slugs=[])  # nothing idle: nothing is given up
    assert len(first.held) == 2
    first.run_pass(offered, idle_slugs=[offered[0].slug])
    assert len(first.held) == 1  # only the idle project, at its boundary


def test_the_oldest_claim_is_released_first(env) -> None:
    offered = projects(env, 3)
    idle = [c.slug for c in offered]
    clock = {'now': 0.0}
    first = replica('w1', clock=lambda: clock['now'])
    for index, _candidate in enumerate(offered):  # claimed one after the other
        clock['now'] = float(index)
        first.run_pass(offered[: index + 1], idle)
    clock['now'] = 10.0
    second = replica('w2')
    second.run_pass(offered, idle)
    first.run_pass(offered, idle)  # fair = ceil(3 / 2) = 2: one goes, the oldest
    assert set(first.held) == {offered[1].project_id, offered[2].project_id}


def test_a_killed_workers_projects_move_after_the_lease_expires(env) -> None:
    offered = projects(env, 2)
    idle = [c.slug for c in offered]
    dead, alive = replica('w1'), replica('w2')
    dead.run_pass(offered, idle)
    before = tokens(env)
    alive.run_pass(offered, idle)
    assert len(alive.held) == 0  # the dead worker's leases are still live
    # time passes: the dead worker's leases and liveness row age past the TTL
    sql(env, "UPDATE stream_worker_claims SET lease_expires_at = now() - interval '1 second'")
    sql(
        env,
        "UPDATE worker_capabilities SET seen_at = now() - interval '5 minutes' WHERE worker_id = 'detection@w1'",
    )
    alive.run_pass(offered, idle)
    assert set(alive.held) == {c.project_id for c in offered}
    after = tokens(env)
    assert all(after[p] == before[p] + 1 for p in before)  # a takeover bumps the token


def test_a_renewal_keeps_the_token_and_a_takeover_bumps_it(env) -> None:
    offered = projects(env, 1)
    idle = [offered[0].slug]
    holder = replica('w1')
    holder.run_pass(offered, idle)
    first = tokens(env)
    holder.run_pass(offered, idle)
    holder.run_pass(offered, idle)
    assert tokens(env) == first  # renewals leave it alone

    other = replica('w2')
    sql(env, "UPDATE stream_worker_claims SET lease_expires_at = now() - interval '1 second'")
    sql(env, "UPDATE worker_capabilities SET seen_at = now() - interval '5 minutes'")
    other.run_pass(offered, idle)
    assert tokens(env) == {offered[0].project_id: first[offered[0].project_id] + 1}
    # the old holder renews nothing it no longer owns, and drops the project
    holder.run_pass(offered, idle)
    assert holder.held == {}


def test_a_paused_project_is_skipped_and_a_project_that_goes_away_is_released(env) -> None:
    live, paused = projects(env, 2)
    paused = Candidate(paused.project_id, paused.slug, paused=True)
    worker = replica('w1')
    held = worker.run_pass([live, paused], [live.slug, paused.slug])
    assert set(held) == {live.project_id}
    worker.run_pass([paused], [live.slug])  # the live one was archived or deleted
    assert holders(env) == {}


def test_shutdown_releases_every_claim_and_the_liveness_row(env) -> None:
    offered = projects(env, 2)
    worker = replica('w1')
    worker.run_pass(offered, [c.slug for c in offered])
    assert worker.release_all() == 2
    assert holders(env) == {}
    assert sql(
        env, "SELECT count(*) FROM worker_capabilities WHERE worker_id = 'detection@w1'"
    ) == [(0,)]
    other = replica('w2')
    other.run_pass(offered, [c.slug for c in offered])
    assert len(other.held) == 2  # the departed replica no longer halves its share


def test_the_two_kinds_do_not_count_each_other(env) -> None:
    offered = projects(env, 2)
    idle = [c.slug for c in offered]
    detection, vlm = replica('w1'), ProjectClaims('vlm', host='w1')
    detection.run_pass(offered, idle)
    vlm.run_pass(offered, idle)
    assert len(detection.held) == 2
    assert len(vlm.held) == 2


@pytest.mark.asyncio
async def test_without_the_database_the_held_claims_last_until_the_fence_then_none(env) -> None:
    from sqlalchemy.exc import OperationalError

    offered = projects(env, 2)
    clock = {'now': 0.0}
    worker = replica('w1', ttl_s=120, margin_s=30, pass_every_s=15, clock=lambda: clock['now'])
    assert len(await worker.refresh(offered)) == 2

    def down() -> None:
        raise OperationalError('select 1', {}, OSError('database is down'))

    worker.session_factory = down  # type: ignore[assignment]
    clock['now'] = 20.0
    assert len(await worker.refresh(offered)) == 2  # young enough: keep serving what is held
    clock['now'] = 95.0
    assert await worker.refresh(offered) == set()  # past TTL minus the margin: stop


@pytest.mark.asyncio
async def test_select_returns_the_records_held_and_never_one_without_a_database_id(env) -> None:
    from types import SimpleNamespace

    offered = projects(env, 2)
    records = [SimpleNamespace(id=c.project_id, slug=c.slug, paused_at=None) for c in offered] + [
        SimpleNamespace(id=None, slug='no-id', paused_at=None)
    ]
    worker = replica('w1')
    served = await worker.select(records)
    assert [r.slug for r in served] == [c.slug for c in offered]
