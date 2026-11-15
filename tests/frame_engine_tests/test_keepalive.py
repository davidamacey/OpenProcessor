"""The API's channels to the engine and the engine's server agree on keepalive pings.

gRPC keepalive guide (https://grpc.io/docs/guides/keepalive/) and the C-core keepalive document
(https://github.com/grpc/grpc/blob/master/doc/keepalive.md): a server answers a ping it does not
allow with GOAWAY ``ENHANCE_YOUR_CALM`` / ``too_many_pings`` once the strikes pass
``http2.max_ping_strikes``. A ping with no call in flight is not allowed unless the server sets
``keepalive_permit_without_calls``, and no ping is allowed sooner than the server's
``http2.min_ping_interval_without_data_ms`` after the last. The engine's server ran with the
defaults (no pings without calls, 5 minutes apart) while the API's channels pinged every 30 s even
when idle: the 354 GOAWAYs and the two ``503`` of the 2026-11-13 speed chain.

The wire test runs the production option sets with every duration divided by :data:`SCALE`
(gRPC does not send keepalive pings more often than once a second, so 30 is the largest divisor
that keeps the client's ping period at its floor). The ratio between the two sides, the thing
the rule is about, is unchanged.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import grpc
import pytest
from tritonclient.grpc import service_pb2, service_pb2_grpc

from frame_engine.front import SERVER_OPTIONS, FrontService, make_server
from frame_engine.stage_metrics import EngineMetrics
from src.clients.frame_engine import ENGINE_CHANNEL_OPTIONS
from src.clients.triton_pool_core import GRPC_CHANNEL_OPTIONS


SCALE = 30
_TIMES = {
    'grpc.keepalive_time_ms',
    'grpc.keepalive_timeout_ms',
    'grpc.http2.min_ping_interval_without_data_ms',
}
#: How long the idle channel is watched: a server that refuses the pings strikes the third one
#: about 3 s in (probed: GOAWAY at 3.0 s with the engine's server on gRPC defaults).
IDLE_WATCH_S = 5.0


def _scaled(options: Any) -> list[tuple[str, Any]]:
    return [(k, max(v // SCALE, 100) if k in _TIMES else v) for k, v in options]


class _NoEngine:
    def submit(self, request: Any) -> Any:  # a ping test makes no photo call
        raise AssertionError('no photo is sent')


async def _states_of_an_idle_channel(server_options: Any) -> list[grpc.ChannelConnectivity]:
    """The distinct connectivity states of a channel with one finished call, then idle."""
    service = FrontService(
        _NoEngine(), lambda: True, EngineMetrics(), max_in_flight=1, request_deadline=5.0
    )
    server, port = make_server(
        service, address='127.0.0.1:0', max_message_bytes=2**24, options=tuple(server_options)
    )
    await server.start()
    channel = grpc.aio.insecure_channel(
        f'127.0.0.1:{port}', options=_scaled(ENGINE_CHANNEL_OPTIONS)
    )
    try:
        await service_pb2_grpc.GRPCInferenceServiceStub(channel).ServerLive(
            service_pb2.ServerLiveRequest()
        )
        seen: list[grpc.ChannelConnectivity] = []
        deadline = time.monotonic() + IDLE_WATCH_S
        while time.monotonic() < deadline:
            state = channel.get_state(try_to_connect=False)
            if not seen or seen[-1] != state:
                seen.append(state)
                if len(seen) > 1:  # the connection was closed: nothing more to learn
                    break
            await asyncio.sleep(0.02)
        return seen
    finally:
        await channel.close()
        await server.stop(0)


@pytest.mark.asyncio
async def test_an_idle_engine_channel_is_not_sent_away_for_its_keepalive_pings() -> None:
    seen = await _states_of_an_idle_channel(_scaled(SERVER_OPTIONS))
    assert seen == [grpc.ChannelConnectivity.READY], 'the engine closed a channel for its pings'


@pytest.mark.asyncio
async def test_the_wire_test_would_see_a_server_on_grpc_defaults_send_the_channel_away() -> None:
    """The control: the same channel against a server without the engine's ping policy loses its
    connection (GOAWAY too_many_pings), so the test above proves the options, not the timing."""
    seen = await _states_of_an_idle_channel(())
    assert grpc.ChannelConnectivity.IDLE in seen


def test_the_two_sides_agree_by_the_documented_rules() -> None:
    client, server = dict(ENGINE_CHANNEL_OPTIONS), dict(SERVER_OPTIONS)
    # a client that pings with no call in flight needs a server that permits it
    assert client['grpc.keepalive_permit_without_calls'] == 1
    assert server['grpc.keepalive_permit_without_calls'] == 1
    # no ping sooner than the server's minimum interval, with a margin of a few
    assert (
        client['grpc.keepalive_time_ms']
        >= 2 * server['grpc.http2.min_ping_interval_without_data_ms']
    )
    # the client may send every ping its keepalive time calls for
    assert client['grpc.http2.max_pings_without_data'] == 0


def test_the_engine_options_are_the_pools_but_for_the_keepalive_policy() -> None:
    """The engine's set is its own (the model server's keepalive is not copied), and every other
    value is the pool's: message sizes, reconnect bound, flow-control windows."""
    pool = dict(GRPC_CHANNEL_OPTIONS)
    engine = dict(ENGINE_CHANNEL_OPTIONS)
    own = {
        k
        for k in engine
        if k.startswith('grpc.keepalive') or k == 'grpc.http2.max_pings_without_data'
    }
    assert {k: engine[k] for k in engine if k not in own} == {
        k: pool[k] for k in engine if k not in own
    }
