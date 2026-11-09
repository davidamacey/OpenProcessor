"""Trainer reachability: a live ``gpu_train`` consumer plus the capabilities its worker recorded.

The broker is replaced by a fake ``inspect().active_queues()`` answer; the capabilities and the
card inventory are real rows.
"""

from __future__ import annotations

import json

import pytest
from fixtures.job_rows import sql

from src.clients import task_queue
from src.services.training import trainer_reachability as reach


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]

HOST = 'trainer-host-1'
CARD_A, CARD_B = 'GPU-aaaa', 'GPU-bbbb'


def _consumers(monkeypatch, *nodes: str, queue: str = 'gpu_train') -> None:
    monkeypatch.setattr(
        reach,
        '_inspect_active_queues',
        lambda: {node: [{'name': queue}] for node in nodes} or None,
    )


@pytest.fixture
def capabilities(pg_env):
    for statement in (
        'DELETE FROM worker_capabilities',
        'DELETE FROM gpu_reservations',
        'DELETE FROM gpu_devices',
    ):
        sql(pg_env, statement)
    for uuid_, nvml in ((CARD_A, 0), (CARD_B, 2)):
        sql(
            pg_env,
            'INSERT INTO gpu_devices (gpu_uuid, host, nvml_index, name, compute_capability, '
            'memory_total_mib, hw_jpeg_decoder, driver_version, cuda_version) '
            "VALUES (%s, %s, %s, 'RTX A6000', '8.6', 49140, false, '580', '13.0')",
            uuid_,
            HOST,
            nvml,
        )
    sql(
        pg_env,
        'INSERT INTO worker_capabilities (worker_id, queue, host, capabilities) '
        "VALUES (%s, 'gpu_train', %s, %s::jsonb)",
        f'gpu_train@{HOST}',
        HOST,
        json.dumps({'gpu_uuids': [CARD_A, CARD_B], 'device_uuid': None}),
    )
    return pg_env


@pytest.mark.asyncio
async def test_no_consumer_means_the_trainer_is_not_running(capabilities, monkeypatch) -> None:
    _consumers(monkeypatch)  # the broker answers, nobody consumes
    severity, detail = await reach.probe_trainer_reachable()
    assert severity == 'block'
    assert 'not running' in detail


@pytest.mark.asyncio
async def test_a_consumer_of_another_queue_does_not_count(capabilities, monkeypatch) -> None:
    _consumers(monkeypatch, f'celery@{HOST}', queue='jobs')
    assert (await reach.probe_trainer_reachable())[0] == 'block'


@pytest.mark.asyncio
async def test_a_consumer_with_its_capabilities_row_is_available(capabilities, monkeypatch) -> None:
    _consumers(monkeypatch, f'celery@{HOST}')
    severity, _detail = await reach.probe_trainer_reachable()
    assert severity == 'ok'
    assert await reach.read_trainer_gpu_order() == [0, 2]


@pytest.mark.asyncio
async def test_a_consumer_on_a_host_without_a_row_is_only_a_warning(
    capabilities, monkeypatch
) -> None:
    _consumers(monkeypatch, 'celery@some-other-host')
    assert (await reach.probe_trainer_reachable())[0] == 'warn'
    assert await reach.read_trainer_gpu_order() is None


@pytest.mark.asyncio
async def test_an_unreachable_broker_is_a_warning_not_a_block(capabilities, monkeypatch) -> None:
    def down():
        raise task_queue.QueueUnavailable('broker is down')

    monkeypatch.setattr(reach, '_inspect_active_queues', down)
    assert (await reach.probe_trainer_reachable())[0] == 'warn'
    assert await reach.read_trainer_gpu_order() is None
