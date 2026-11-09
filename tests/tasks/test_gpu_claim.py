"""A GPU job reserves its cards and waits for the arbiter tick; the wait is bounded and fails closed.

The arbiter tick records which claims it confirmed in
``deployment_settings('vlm_paused_for_training')``; the handler polls that record with a bounded
exponential backoff and ends ``gpu_busy`` when the wait is spent.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

import psycopg
import pytest
from fixtures.job_rows import insert_row, make_project, sql

from src.config import GpuArbiterConfig, gpu_arbiter as gpu_arbiter_config_module
from src.db.repos import gpu_devices
from src.db.session import async_session_factory, project_tx
from src.domain import gpu_arbiter_state as state
from src.tasks.runtime import gpu_claim


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


class Ctx:
    """The slice of ``JobContext`` the claim helpers use, over the real database."""

    def __init__(self, project: uuid.UUID, job_id: uuid.UUID) -> None:
        self.project_id = project
        self.job_id = job_id
        self.cancel = False

    def cancelled(self) -> bool:
        return self.cancel

    @asynccontextmanager
    async def session(self):
        async with async_session_factory()() as session, project_tx(session, self.project_id):
            yield session


class Clock:
    """A clock the fake sleep advances, so the backoff runs in no time."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def world(pg, monkeypatch):
    for table in ('gpu_reservations', 'gpu_devices', 'jobs'):
        sql(pg, f'DELETE FROM {table}')
    sql(pg, 'DELETE FROM deployment_settings WHERE key = %s', state.KEY)
    monkeypatch.setattr(
        gpu_arbiter_config_module,
        '_default_gpu_arbiter_config',
        GpuArbiterConfig(containers=('vlm',), container_gpus=(('vlm', None),)),
    )
    monkeypatch.setenv(gpu_claim.WAIT_ENV, '60')
    project = make_project(pg)
    job = insert_row(pg, project, 'running', 'bakeoff')
    return pg, project, job, Ctx(project, job), Clock()


def acknowledge(pg, *jobs: uuid.UUID) -> None:
    body = state.ArbiterRecord(
        claims=tuple(str(j) for j in jobs), acknowledged=tuple(str(j) for j in jobs)
    ).to_json()
    sql(
        pg,
        'INSERT INTO deployment_settings (key, value, updated_by) VALUES (%s, %s, %s) '
        'ON CONFLICT (key) DO UPDATE SET value = excluded.value',
        state.KEY,
        psycopg.types.json.Jsonb({state.ARBITER: body}),
        'test',
    )


def card(pg, n: int) -> str:
    gpu_uuid = f'GPU-{uuid.uuid4().hex[:8]}'
    sql(
        pg,
        'INSERT INTO gpu_devices (gpu_uuid, host, nvml_index, name, compute_capability, '
        'memory_total_mib, hw_jpeg_decoder, driver_version, cuda_version) '
        "VALUES (%s, 'h', %s, 'RTX', '8.6', 49140, false, '580', '13.0')",
        gpu_uuid,
        n,
    )
    return gpu_uuid


def holders(pg) -> list[tuple[Any, ...]]:
    return sql(pg, 'SELECT gpu_uuid, holder_job_id FROM gpu_reservations ORDER BY gpu_uuid')


@pytest.mark.asyncio
async def test_a_confirmed_claim_returns_without_sleeping(world) -> None:
    pg, _project, job, ctx, clock = world
    acknowledge(pg, job)

    await gpu_claim.wait_for_arbiter(ctx, sleep=clock.sleep, clock=clock)

    assert clock.slept == []


@pytest.mark.asyncio
async def test_an_unconfirmed_claim_backs_off_then_fails_closed_with_gpu_busy(world) -> None:
    pg, _project, _job, ctx, clock = world
    sql(
        pg,
        'INSERT INTO deployment_settings (key, value, updated_by) VALUES (%s, %s, %s)',
        state.KEY,
        psycopg.types.json.Jsonb(
            {state.ARBITER: state.ArbiterRecord(error='stop vlm: daemon refused').to_json()}
        ),
        'test',
    )

    with pytest.raises(gpu_claim.GpuBusyError, match=r'gpu_busy.*daemon refused'):
        await gpu_claim.wait_for_arbiter(ctx, sleep=clock.sleep, clock=clock)

    assert clock.slept == [2.0, 4.0, 8.0, 15.0, 15.0, 15.0, 1.0]  # bounded, spends the 60 s
    assert clock.now - 1000.0 == 60.0


@pytest.mark.asyncio
async def test_the_wait_ends_as_soon_as_the_arbiter_confirms(world) -> None:
    pg, _project, job, ctx, clock = world
    plain_sleep = clock.sleep

    async def sleep(seconds: float) -> None:
        await plain_sleep(seconds)
        if len(clock.slept) == 2:
            acknowledge(pg, job)

    await gpu_claim.wait_for_arbiter(ctx, sleep=sleep, clock=clock)

    assert clock.slept == [2.0, 4.0]


@pytest.mark.asyncio
async def test_a_cancel_ends_the_wait_without_an_error(world) -> None:
    _pg, _project, _job, ctx, clock = world
    ctx.cancel = True

    await gpu_claim.wait_for_arbiter(ctx, sleep=clock.sleep, clock=clock)

    assert clock.slept == []


@pytest.mark.asyncio
async def test_without_an_arbiter_nothing_is_reserved_or_waited_for(world, monkeypatch) -> None:
    pg, _project, _job, ctx, clock = world
    monkeypatch.setattr(
        gpu_arbiter_config_module, '_default_gpu_arbiter_config', GpuArbiterConfig()
    )
    card(pg, 0)

    found = await gpu_claim.claim_cards(
        ctx, lambda s: gpu_devices.on_host(s, 'h'), sleep=clock.sleep, clock=clock
    )
    await gpu_claim.release_cards(ctx)

    assert (found, holders(pg), clock.slept) == ([], [], [])


@pytest.mark.asyncio
async def test_claim_cards_reserves_every_card_then_waits_for_the_confirmation(world) -> None:
    pg, _project, job, ctx, clock = world
    a, b = card(pg, 0), card(pg, 1)
    plain_sleep = clock.sleep

    async def sleep(seconds: float) -> None:
        await plain_sleep(seconds)
        acknowledge(pg, job)

    found = await gpu_claim.claim_cards(
        ctx, lambda s: gpu_devices.on_host(s, 'h'), sleep=sleep, clock=clock
    )

    assert sorted(found) == sorted([a, b])
    assert holders(pg) == sorted([(a, job), (b, job)])
    await gpu_claim.release_cards(ctx)
    assert holders(pg) == []


@pytest.mark.asyncio
async def test_a_card_another_job_holds_is_waited_for_then_gpu_busy_and_nothing_is_kept(
    world,
) -> None:
    pg, project, _job, ctx, clock = world
    free, taken = card(pg, 0), card(pg, 1)
    other = insert_row(pg, project, 'running', 'train')
    sql(pg, 'INSERT INTO gpu_reservations (gpu_uuid, holder_job_id) VALUES (%s, %s)', taken, other)

    with pytest.raises(gpu_claim.GpuBusyError, match=f'gpu_busy: GPU {taken} is reserved'):
        await gpu_claim.claim_cards(
            ctx, lambda s: gpu_devices.on_host(s, 'h'), sleep=clock.sleep, clock=clock
        )

    assert holders(pg) == [(taken, other)]  # all or none: the free card was not kept
    assert free not in [h[0] for h in holders(pg)]
    assert clock.slept[:3] == [2.0, 4.0, 8.0]


@pytest.mark.asyncio
async def test_cards_the_inventory_does_not_know_are_a_lookup_error(world) -> None:
    _pg, _project, _job, ctx, clock = world

    with pytest.raises(LookupError, match='no GPU card is recorded'):
        await gpu_claim.claim_cards(
            ctx, lambda s: gpu_devices.on_host(s, 'nobody'), sleep=clock.sleep, clock=clock
        )
