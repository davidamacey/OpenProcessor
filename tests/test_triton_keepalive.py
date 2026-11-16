"""The API's channels to the model server and the model server agree on keepalive pings.

gRPC keepalive guide (https://grpc.io/docs/guides/keepalive/) and the C-core keepalive document
(https://github.com/grpc/grpc/blob/master/doc/keepalive.md): a server answers a ping it does not
allow with GOAWAY ``ENHANCE_YOUR_CALM`` / ``too_many_pings``. A ping with no call in flight is not
allowed unless the server sets ``keepalive_permit_without_calls`` (default off), and no ping is
allowed sooner than the server's ``http2.min_recv_ping_interval_without_data_ms`` (default 5
minutes) after the last. Triton exposes both as ``--grpc-keepalive-permit-without-calls`` and
``--grpc-http2-min-recv-ping-interval-without-data`` (Triton 26.10 inference-protocols page, which
refers to the gRPC document for the defaults). The harness's fake model server answered every idle
API channel with that GOAWAY (live sweep 2026-11-15: ``503 semantic search unavailable``).

An offline wire test was tried and dropped: on a loopback server on gRPC defaults the pool's own
options (``max_pings_without_data`` 2: two idle pings, then none) did not provoke the GOAWAY in
8 s, with or without calls in between, and a control with ``max_pings_without_data`` 0 would prove a
different option set. The 2026-11-15 GOAWAYs are in the sweep's API log; the live tier reruns them.

The client options in ``src/clients/triton_pool_core.py`` are the reference; every place that starts
a model server is pinned to them here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.clients.triton_pool_core import GRPC_CHANNEL_OPTIONS


ROOT = Path(__file__).resolve().parents[1]
TRITON_COMPOSE_FILES = (
    'docker-compose.yml',
    'docker-compose.faces-off.yml',
    'docker-compose.ingest-focus.yml',
)
PERMIT_FLAG = '--grpc-keepalive-permit-without-calls='
INTERVAL_FLAG = '--grpc-http2-min-recv-ping-interval-without-data='


def _fake_triton() -> Any:
    spec = importlib.util.spec_from_file_location(
        'fake_triton_keepalive', ROOT / 'docker' / 'test' / 'fake_triton.py'
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compose_flag(path: str, prefix: str) -> str:
    command = yaml.safe_load((ROOT / path).read_text())['services']['triton-server']['command']
    found = [arg.split('=', 1)[1] for arg in command if arg.startswith(prefix)]
    assert len(found) == 1, f'{path}: {prefix} must be set exactly once, got {found}'
    return found[0]


@pytest.mark.parametrize('path', TRITON_COMPOSE_FILES)
def test_every_triton_the_stack_starts_accepts_the_clients_idle_pings(path: str) -> None:
    client = dict(GRPC_CHANNEL_OPTIONS)
    assert client['grpc.keepalive_permit_without_calls'] == 1
    assert _compose_flag(path, PERMIT_FLAG) == 'true'
    assert int(_compose_flag(path, INTERVAL_FLAG)) <= client['grpc.keepalive_time_ms']


def test_the_harness_fake_model_server_accepts_the_clients_idle_pings() -> None:
    client = dict(GRPC_CHANNEL_OPTIONS)
    server = dict(_fake_triton().GRPC_OPTIONS)
    assert server['grpc.keepalive_permit_without_calls'] == 1
    assert (
        server['grpc.http2.min_ping_interval_without_data_ms'] <= client['grpc.keepalive_time_ms']
    )
