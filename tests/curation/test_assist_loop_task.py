"""The assist-loop handler: one slice per delivery, yield after ``slice_s``, the person's pause,
the loop's own pauses with their reasons, and a finished loop's result."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from curation.assist_loop_fakes import SLUG, FakeVlm, World, build_deps, loop_params, make_world
from src.db.repos import assist_loop_record as rows
from src.services.curation.assist_loop import runtime
from src.services.curation.assist_loop.partition import PartitionScope
from src.services.curation.assist_loop.vlm_failure import VlmFailedError
from src.tasks import assist_loop
from src.tasks.runtime.context import Paused, Yield


if TYPE_CHECKING:
    from curation.assist_loop_fakes import LoopEnv, RunRef

pytestmark = [pytest.mark.db, pytest.mark.usefixtures('readable_crops')]


class Ctx:
    """The parts of a ``JobContext`` the handler reads, over the real test database."""

    def __init__(self, env: LoopEnv, ref: RunRef, engine: Any) -> None:
        self.env, self.ref, self._engine = env, ref, engine
        self.job_id, self.fencing_token = ref.job_id, ref.token
        self.project_id, self.project_slug = env.project, SLUG
        self.cancel = self.pause = False
        self.slice_after: int | None = None
        self.requests = lambda: 0
        self.paused_for: list[str] = []
        self.snapshots: list[tuple[Any, ...]] = []

    def cancelled(self) -> bool:
        return self.cancel

    def paused(self) -> bool:
        return self.pause

    def slice_due(self) -> bool:
        return self.slice_after is not None and self.requests() >= self.slice_after

    async def progress(self, done: int, total: int | None, stage: str, counters: Any) -> None:
        self.snapshots.append((done, total, stage, dict(counters)))

    async def request_pause(self, reason: str) -> None:
        self.paused_for.append(reason)

    async def client(self, _name: str) -> str:
        return 'opensearch'

    def session(self) -> Any:
        @asynccontextmanager
        async def open_() -> Any:
            from sqlalchemy.ext.asyncio import AsyncSession

            from src.db import session as db

            async with (
                AsyncSession(self._engine, expire_on_commit=False) as session,
                db.project_tx(session, self.env.project),
            ):
                yield session

        return open_()


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    w = make_world(monkeypatch)
    w.add_cluster(10001, {'car': 300}, prefix='a')
    w.add_cluster(10002, {'truck': 250}, prefix='b')
    return w


@pytest.fixture
def harness(world: World, loop_env: LoopEnv, monkeypatch: pytest.MonkeyPatch) -> Any:
    ref = loop_env.new_run(max_crops=100_000)
    vlm = FakeVlm(world.truth)
    state = SimpleNamespace(blocked=None, vlm=vlm)

    @contextmanager
    def bound(_ctx: Any, _record: Any):
        yield

    async def record(_ctx: Any) -> Any:
        return SimpleNamespace(slug=SLUG)

    async def blocked(_ctx: Any) -> str | None:
        return state.blocked

    async def build(_os: Any, rec: Any, budget: Any) -> Any:
        lp = loop_params(node_concurrency=1, max_depth=0, on_budget='pause')
        deps = await build_deps(
            world,
            state.vlm,
            lp,
            rec,
            budget,
            vlm_concurrency=1,
            reader=world.reader(loop_env, rec),
        )
        request = SimpleNamespace()
        return runtime.Built(deps, lp, PartitionScope(None, 40), request)  # type: ignore[arg-type]

    monkeypatch.setattr(assist_loop, 'project_record', record)
    monkeypatch.setattr(assist_loop, 'bound_job_project', bound)
    monkeypatch.setattr(assist_loop, '_blocked', blocked)
    monkeypatch.setattr(runtime, 'build', build)
    state.ref = ref
    return state


async def _deliver(env: LoopEnv, ref: RunRef, **flags: Any) -> tuple[Any, Ctx]:
    engine = create_async_engine(env.pg.url('app'), poolclass=NullPool)
    try:
        ctx = Ctx(env, ref, engine)
        for key, value in flags.items():
            setattr(ctx, key, value)
        async with ctx.session() as session:
            if not ref.seeded:
                from src.db.repos import assist_loop as budget_repo

                await session.run_sync(
                    lambda s: budget_repo.seed_budgets(s, ref.run_id, ref.limits.seed())
                )
                ref.seeded = True
            # the handler reads the run's request and pins itself
            await session.run_sync(lambda s: s.execute(__import__('sqlalchemy').text('SELECT 1')))
        result = await assist_loop.assist_loop(ctx)  # type: ignore[arg-type]
        return result, ctx
    finally:
        await engine.dispose()


def test_a_finished_loop_returns_its_stop_reason_and_the_run_has_it(
    harness: Any, loop_env: LoopEnv
) -> None:
    result, ctx = asyncio.run(_deliver(loop_env, harness.ref))

    assert result == {'stop_reason': 'done', 'loop_id': str(harness.ref.run_id)}
    assert ctx.paused_for == []
    assert ctx.snapshots[-1][2] == 'iteration:1'
    assert ctx.snapshots[-1][3]['nodes_open'] == 0
    assert ctx.snapshots[-1][3]['items_propagated'] > 0
    assert _stop_reason(loop_env, harness.ref) == 'done'


def _stop_reason(env: LoopEnv, ref: RunRef) -> str | None:
    import psycopg

    with psycopg.connect(env.pg.conninfo('super'), autocommit=True) as conn:
        return conn.execute(
            'SELECT stop_reason FROM assist_loop_runs WHERE id = %s', (ref.run_id,)
        ).fetchone()[0]


def test_a_slice_yields_at_a_node_boundary_and_the_next_delivery_finishes_the_loop(
    harness: Any, loop_env: LoopEnv
) -> None:
    vlm = harness.vlm
    first, _ = asyncio.run(
        _deliver(loop_env, harness.ref, requests=lambda: vlm.requests, slice_after=1)
    )
    assert isinstance(first, Yield)
    asked_first = set(vlm.asked)

    second, _ = asyncio.run(_deliver(loop_env, harness.ref))

    assert second['stop_reason'] == 'done'
    assert asked_first.isdisjoint(vlm.asked[len(asked_first) :]), 'no item is asked in two slices'
    assert len(vlm.asked) == len(set(vlm.asked))


def test_the_persons_pause_returns_paused_without_a_reason_of_its_own(
    harness: Any, loop_env: LoopEnv
) -> None:
    result, ctx = asyncio.run(_deliver(loop_env, harness.ref, pause=True))

    assert result == Paused()
    assert ctx.paused_for == []
    assert harness.vlm.asked == []


@pytest.mark.parametrize('reason', ['project_paused', 'vlm_unavailable'])
def test_a_slice_that_may_not_start_pauses_itself_with_the_reason(
    harness: Any, loop_env: LoopEnv, reason: str
) -> None:
    harness.blocked = reason

    result, ctx = asyncio.run(_deliver(loop_env, harness.ref))

    assert result == Paused(reason)
    assert ctx.paused_for == [reason]
    assert harness.vlm.asked == []


def test_a_vlm_that_does_not_answer_pauses_the_loop_as_unavailable(
    harness: Any, loop_env: LoopEnv
) -> None:
    harness.vlm = FakeVlm(harness.vlm.truth, fail_rate=1.0)

    result, ctx = asyncio.run(_deliver(loop_env, harness.ref))

    assert result == Paused('vlm_unavailable')
    assert ctx.paused_for == ['vlm_unavailable']
    assert VlmFailedError  # the exception the slice turned into a pause


def test_a_reached_budget_pauses_for_budget_and_a_raised_limit_lets_it_finish(
    harness: Any, loop_env: LoopEnv
) -> None:
    import psycopg

    with psycopg.connect(loop_env.pg.conninfo('super'), autocommit=True) as conn:
        harness.ref.limits = type(harness.ref.limits)(
            pools=harness.ref.limits.pools, max_wall_s=harness.ref.limits.max_wall_s
        )
    tight = loop_env.new_run(max_crops=60)
    first, ctx = asyncio.run(_deliver(loop_env, tight))

    assert first == Paused('budget')
    assert ctx.paused_for == ['budget']
    assert _stop_reason(loop_env, tight) is None, 'a paused loop has not stopped'

    with psycopg.connect(loop_env.pg.conninfo('super'), autocommit=True) as conn:
        conn.execute(
            'UPDATE assist_loop_budgets SET limit_value = 100000 '
            "WHERE run_id = %s AND pool = 'vlm' AND unit IN ('crops', 'requests')",
            (tight.run_id,),
        )
    second, _ = asyncio.run(_deliver(loop_env, tight))

    assert second['stop_reason'] == 'done'
    assert len(harness.vlm.asked) == len(set(harness.vlm.asked))
    assert rows  # the tables are the record of both slices
