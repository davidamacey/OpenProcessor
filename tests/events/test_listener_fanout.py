"""One Valkey subscription per process fans committed rows out to local subscribers."""

from __future__ import annotations

import asyncio

import pytest
from _events_support import seed_project
from _listener_support import assert_quiet, running_listener, take

from src.services.events.listener import Overflow
from src.services.events.publish import GLOBAL, apublish


pytestmark = pytest.mark.db


@pytest.mark.asyncio
async def test_subscribers_get_only_their_projects_events(pg_env, broker) -> None:
    seed_project(pg_env, 'fan-a')
    seed_project(pg_env, 'fan-b')
    async with running_listener(broker) as listener:
        a1 = await listener.subscribe('fan-a')
        a2 = await listener.subscribe('fan-a')
        b = await listener.subscribe('fan-b')
        await apublish('crop.created', {'crop_id': 'a-1'}, project='fan-a')
        await apublish('crop.created', {'crop_id': 'b-1'}, project='fan-b')
        await apublish('crop.created', {'crop_id': 'a-2'}, project='fan-a')
        for sub in (a1, a2):
            assert [r.payload['crop_id'] for r in await take(sub, 2)] == ['a-1', 'a-2']
        assert [r.payload['crop_id'] for r in await take(b, 1)] == ['b-1']
        await assert_quiet(a1)
        await assert_quiet(b)


@pytest.mark.asyncio
async def test_the_global_stream_gets_only_global_events(pg_env, broker) -> None:
    seed_project(pg_env, 'fan-g')
    async with running_listener(broker) as listener:
        scoped = await listener.subscribe('fan-g')
        glob = await listener.subscribe(None)
        await apublish('crop.created', {}, project='fan-g')
        await apublish('project.created', {'target': 'x'}, project=GLOBAL)
        assert [r.type for r in await take(glob, 1)] == ['project.created']
        assert [r.type for r in await take(scoped, 1)] == ['crop.created']
        await assert_quiet(glob)
        await assert_quiet(scoped)


@pytest.mark.asyncio
async def test_with_every_wakeup_dropped_the_two_second_tick_delivers_each_event_once(
    pg_env, broker
) -> None:
    seed_project(pg_env, 'tick')
    broker.drop_wakeups = True
    async with running_listener(broker) as listener:  # the default 2 s safety tick
        sub = await listener.subscribe('tick')
        await apublish('crop.created', {'crop_id': '1'}, project='tick')
        await apublish('crop.created', {'crop_id': '2'}, project='tick')
        assert [r.payload['crop_id'] for r in await take(sub, 2, timeout=5)] == ['1', '2']
        await assert_quiet(sub, wait=2.5)


@pytest.mark.asyncio
async def test_a_dropped_subscription_recovers_and_the_catch_up_delivers_the_gap(
    pg_env, broker
) -> None:
    seed_project(pg_env, 'recover')
    async with running_listener(broker, tick_s=3600) as listener:  # only a wake-up can deliver
        sub = await listener.subscribe('recover')
        await asyncio.sleep(1.2)  # let the first subscription settle
        assert listener.pubsub_connected
        broker.disconnect_subscribers()
        await apublish('crop.created', {'crop_id': 'during-outage'}, project='recover')
        got = await take(sub, 1, timeout=8)
        assert [r.payload['crop_id'] for r in got] == ['during-outage']
        await assert_quiet(sub)


@pytest.mark.asyncio
async def test_with_valkey_down_the_tick_still_delivers_and_start_does_not_fail(
    pg_env, broker
) -> None:
    seed_project(pg_env, 'down')
    broker.down = True
    async with running_listener(broker, tick_s=0.3) as listener:
        sub = await listener.subscribe('down')
        await apublish('crop.created', {'crop_id': 'x'}, project='down')
        assert [r.payload['crop_id'] for r in await take(sub, 1)] == ['x']
        assert not listener.pubsub_connected


@pytest.mark.asyncio
async def test_a_slow_client_is_dropped_with_a_final_marker_and_never_blocks_the_others(
    pg_env, broker
) -> None:
    seed_project(pg_env, 'slow')
    async with running_listener(broker, tick_s=0.3, queue_max=3) as listener:
        slow = await listener.subscribe('slow')
        fast = await listener.subscribe('slow')
        drained = []

        async def drain() -> None:
            while len(drained) < 6:
                drained.append(await fast.queue.get())

        task = asyncio.create_task(drain())
        for i in range(6):
            await apublish('crop.created', {'crop_id': str(i)}, project='slow')
            await asyncio.sleep(0.4)  # one catch-up per event keeps `fast` below its cap
        await asyncio.wait_for(task, 5)
        assert [r.payload['crop_id'] for r in drained] == [str(i) for i in range(6)]
        assert isinstance(slow.queue.get_nowait(), Overflow)
        assert slow.queue.empty(), 'nothing is queued behind the final marker'
        assert listener.dropped_clients == 1


@pytest.mark.asyncio
async def test_subscribing_while_stopped_is_refused(pg_env, broker) -> None:
    from src.services.events.listener import EventListener, ListenerNotRunningError

    with pytest.raises(ListenerNotRunningError):
        await EventListener().subscribe(None)
