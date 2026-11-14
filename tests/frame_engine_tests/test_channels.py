"""The engine's channels to the model server reach a restarted server within seconds.

gRPC's reconnect delay grows towards 120 s after repeated failed connects
(https://grpc.github.io/grpc/core/md_doc_connection-backoff.html); the engine passed only the
message sizes as ``channel_args``, which tritonclient uses INSTEAD of its own defaults, so its
channels had the default backoff and no keepalive. This runs a real gRPC server on a loopback
port, stops it long enough for the default backoff to grow, starts it again and times the first
successful call over a channel made by :func:`frame_engine.transport.channels`.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.utils import InferenceServerException

from frame_engine.transport import CHANNEL_OPTIONS, RECONNECT_MAX_BACKOFF_MS, channels
from src.clients.triton_pool_core import GRPC_CHANNEL_OPTIONS


class _Live(service_pb2_grpc.GRPCInferenceServiceServicer):
    def _live(self, request: Any, context: Any) -> Any:
        return service_pb2.ServerLiveResponse(live=True)

    ServerLive = _live  # the generated RPC name


def _serve(port: int) -> tuple[grpc.Server, int]:
    server = grpc.server(ThreadPoolExecutor(2))
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(_Live(), server)
    bound = server.add_insecure_port(f'127.0.0.1:{port}')
    server.start()
    return server, bound


def _live(client: Any) -> bool:
    try:
        return bool(client.is_server_live(client_timeout=0.5))
    except InferenceServerException:
        return False


def test_keepalive_and_reconnect_values_are_the_api_pools() -> None:
    """One set of values on both sides of the model server (the engine may not import
    ``src.clients``): keepalive above Triton's 20 s ping floor, bounded reconnect."""
    engine, api = dict(CHANNEL_OPTIONS), dict(GRPC_CHANNEL_OPTIONS)
    shared = (
        'grpc.keepalive_time_ms',
        'grpc.keepalive_timeout_ms',
        'grpc.keepalive_permit_without_calls',
        'grpc.http2.max_pings_without_data',
        'grpc.initial_reconnect_backoff_ms',
        'grpc.min_reconnect_backoff_ms',
        'grpc.max_reconnect_backoff_ms',
        'grpc.max_send_message_length',
        'grpc.max_receive_message_length',
    )
    assert {k: engine[k] for k in shared} == {k: api[k] for k in shared}
    assert set(engine) == set(shared)  # the whole set the engine passes
    # docker-compose.yml: --grpc-http2-min-recv-ping-interval-without-data=20000
    assert engine['grpc.keepalive_time_ms'] >= 20000
    assert engine['grpc.max_reconnect_backoff_ms'] == RECONNECT_MAX_BACKOFF_MS <= 5000


def test_a_restarted_model_server_is_reached_within_the_backoff_bound() -> None:
    server, port = _serve(0)
    [client] = channels(f'127.0.0.1:{port}', 1)
    try:
        assert _live(client)
        server.stop(None).wait(5)
        down_until = time.monotonic() + 28  # long enough for the default delay to pass the bound
        while time.monotonic() < down_until:
            assert not _live(client)
            time.sleep(0.2)
        server, _ = _serve(port)
        listening = time.monotonic()
        give_up = listening + 60
        while not _live(client) and time.monotonic() < give_up:
            time.sleep(0.1)
        recovery = time.monotonic() - listening
    finally:
        client.close()
        server.stop(None).wait(5)
    # one capped backoff plus scheduling slack on a loaded host
    assert recovery < RECONNECT_MAX_BACKOFF_MS / 1000 + 5
