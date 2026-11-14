"""A restarted model server is reached again within seconds, not after gRPC's default backoff.

gRPC's reconnect delay grows towards 120 s after repeated failed connects
(https://grpc.github.io/grpc/core/md_doc_connection-backoff.html). The pool's channel
options cap it (``grpc.max_reconnect_backoff_ms``); this runs a real ``grpc.aio`` dummy
server on a loopback port, stops it long enough for the backoff to grow, restarts it and
times the first successful call.
"""

from __future__ import annotations

import asyncio
import time

import grpc
import pytest
from tritonclient.grpc import service_pb2, service_pb2_grpc

from src.clients.triton_pool_core import (
    GRPC_CHANNEL_OPTIONS,
    RECONNECT_MAX_BACKOFF_MS,
    AsyncTritonPool,
)


class _Live(service_pb2_grpc.GRPCInferenceServiceServicer):
    async def _live(self, request, context):
        return service_pb2.ServerLiveResponse(live=True)

    ServerLive = _live  # the generated RPC name


async def _serve(port: int) -> tuple[grpc.aio.Server, int]:
    server = grpc.aio.server()
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(_Live(), server)
    bound = server.add_insecure_port(f'127.0.0.1:{port}')
    await server.start()
    return server, bound


def test_reconnect_backoff_options_are_bounded() -> None:
    options = dict(GRPC_CHANNEL_OPTIONS)
    assert options['grpc.max_reconnect_backoff_ms'] == RECONNECT_MAX_BACKOFF_MS <= 5000
    assert options['grpc.initial_reconnect_backoff_ms'] <= options['grpc.max_reconnect_backoff_ms']
    assert options['grpc.min_reconnect_backoff_ms'] <= options['grpc.max_reconnect_backoff_ms']
    # channel_args replace tritonclient's defaults, so the ones it sets must stay.
    for key in (
        'grpc.max_send_message_length',
        'grpc.max_receive_message_length',
        'grpc.keepalive_time_ms',
        'grpc.keepalive_timeout_ms',
    ):
        assert key in options


@pytest.mark.asyncio
async def test_pool_reaches_restarted_server_within_bound() -> None:
    async def scenario() -> float:
        server, port = await _serve(0)
        pool = AsyncTritonPool(f'127.0.0.1:{port}', pool_size=1)
        await pool.initialize()
        try:
            assert await pool.health_check()
            await server.stop(None)
            # Long enough for gRPC's default delay to grow past the bound asserted below.
            down_until = time.monotonic() + 28
            while time.monotonic() < down_until:
                assert not await pool.health_check()
                await asyncio.sleep(0.2)
            server, _ = await _serve(port)
            listening = time.monotonic()
            while not await pool.health_check():
                await asyncio.sleep(0.1)
            return time.monotonic() - listening
        finally:
            await pool.close()
            await server.stop(None)

    recovery = await asyncio.wait_for(scenario(), timeout=90)
    # one capped backoff plus scheduling slack on a loaded host; the uncapped default gave 16 s
    assert recovery < RECONNECT_MAX_BACKOFF_MS / 1000 + 5
