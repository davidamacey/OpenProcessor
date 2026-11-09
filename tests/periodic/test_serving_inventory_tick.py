"""``op.periodic.serving_inventory``: ``serving_reports`` follows what Triton says, and an
unreachable server is marked ``unavailable`` rather than left looking ready."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fixtures.job_rows import sql

from src.services.periodic import serving_inventory
from src.services.periodic.runner import run_periodic


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_env')]

BODY = {
    'servers': [
        {'name': 'gpu0', 'grpc_url': 'triton-a:8001', 'metrics_url': 'triton-a:8002'},
        {'name': 'gpu1', 'grpc_url': 'triton-b:8001', 'metrics_url': 'triton-b:8002'},
    ],
    'instances': {'yolo': 2},
}
METRICS = (
    '# TYPE nv_gpu_utilization gauge\n'
    'nv_gpu_utilization{gpu_uuid="GPU-aaaa"} 0.5\n'
    'nv_gpu_utilization{gpu_uuid="GPU-bbbb"} 0.1\n'
)


@pytest.fixture
def profile(monkeypatch: pytest.MonkeyPatch) -> None:
    async def active() -> Any:
        return SimpleNamespace(body=BODY), 1

    monkeypatch.setattr(serving_inventory, 'active_serving_profile', active)


def client(index: list[dict[str, str]], *, down: tuple[str, ...] = ()) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host in down:
            raise httpx.ConnectError('refused', request=request)
        if request.url.path == '/v2/repository/index':
            return httpx.Response(200, json=index)
        return httpx.Response(200, text=METRICS)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def reports(pg) -> list[tuple[Any, ...]]:
    return sql(
        pg,
        'SELECT server, model, version, state, instances FROM serving_reports '
        'ORDER BY server, model, version',
    )


def add_card(pg, uuid_: str) -> None:
    sql(
        pg,
        'INSERT INTO gpu_devices (gpu_uuid, host, nvml_index, name, compute_capability, '
        'memory_total_mib, hw_jpeg_decoder, driver_version, cuda_version, seen_at) '
        "VALUES (%s, 'h', 0, 'card', '8.6', 1, false, '1', '1', now() - interval '2 hours') "
        'ON CONFLICT DO NOTHING',
        uuid_,
    )


def tick(http: httpx.AsyncClient) -> Any:
    return run_periodic('serving_inventory', serving_inventory.tick(client=http))


def test_states_instances_and_a_model_that_left_the_index(pg, profile) -> None:
    index = [
        {'name': 'yolo', 'version': '1', 'state': 'READY'},
        {'name': 'scrfd', 'version': '1', 'state': 'LOADING'},
        {'name': 'ocr', 'version': '1', 'state': 'UNAVAILABLE'},
    ]
    assert tick(client(index)) == {'servers': 2, 'unreachable': 0}
    rows = reports(pg)
    assert ('gpu0', 'yolo', 1, 'ready', 2) in rows
    assert ('gpu0', 'scrfd', 1, 'loading', None) in rows
    assert ('gpu1', 'ocr', 1, 'unavailable', None) in rows

    tick(client([{'name': 'yolo', 'version': '1', 'state': 'READY'}]))
    assert [r[1] for r in reports(pg) if r[0] == 'gpu0'] == ['yolo']


def test_an_unreachable_server_is_marked_unavailable_and_the_others_still_refresh(
    pg, profile
) -> None:
    ready = [{'name': 'yolo', 'version': '1', 'state': 'READY'}]
    tick(client(ready))
    assert tick(client(ready, down=('triton-b',))) == {'servers': 2, 'unreachable': 1}
    states = {r[0]: r[3] for r in reports(pg)}
    assert states == {'gpu0': 'ready', 'gpu1': 'unavailable'}


def test_the_cards_the_metrics_name_get_a_fresh_seen_at(pg, profile) -> None:
    add_card(pg, 'GPU-aaaa')
    add_card(pg, 'GPU-zzzz')
    tick(client([{'name': 'yolo', 'version': '1', 'state': 'READY'}]))
    fresh = {
        row[0]: row[1]
        for row in sql(pg, "SELECT gpu_uuid, seen_at > now() - interval '1 hour' FROM gpu_devices")
    }
    assert fresh['GPU-aaaa'] is True
    assert fresh['GPU-zzzz'] is False


def test_no_active_profile_does_nothing(pg, monkeypatch: pytest.MonkeyPatch) -> None:
    async def none() -> None:
        return None

    monkeypatch.setattr(serving_inventory, 'active_serving_profile', none)
    assert tick(client([])) == {'servers': 0, 'unreachable': 0}
