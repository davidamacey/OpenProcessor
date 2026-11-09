"""``op.periodic.gpu_arbiter``: intent and outcome around the Docker calls, in two transactions.

A claim is a running job that holds cards. The tick stops the services the claim needs (found by
Compose labels), records the claim as confirmed in
``deployment_settings('vlm_paused_for_training')``, and starts, once nothing claims them, only
the services it stopped itself. A Docker failure confirms nothing it did not stop.
"""

from __future__ import annotations

import uuid
from typing import Any

import docker.errors
import psycopg
import pytest
from fixtures.job_rows import insert_row, make_project, sql

from src.config import GpuArbiterConfig, gpu_arbiter as gpu_arbiter_config_module
from src.config.advisory_locks import GPU_ARBITER
from src.domain import gpu_arbiter_state as state
from src.services.periodic import gpu_arbiter, gpu_arbiter_docker as docker_control
from src.services.periodic.runner import run_periodic


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_env')]

PROJECT_LABEL = docker_control.PROJECT_LABEL
SERVICE_LABEL = docker_control.SERVICE_LABEL


class FakeContainer:
    def __init__(self, service: str, status: str, *, fail_stop: bool = False) -> None:
        self.labels = {PROJECT_LABEL: 'op', SERVICE_LABEL: service}
        self.status = status
        self.fail_stop = fail_stop
        self.calls: list[str] = []

    def reload(self) -> None:
        return None

    def stop(self, timeout: int) -> None:
        self.calls.append(f'stop-timeout-{timeout}')
        if self.fail_stop:
            raise docker.errors.APIError('daemon refused')
        self.status = 'exited'

    def start(self) -> None:
        self.calls.append('start')
        self.status = 'running'


class FakeClient:
    def __init__(self, *containers: FakeContainer) -> None:
        self.all = [*containers]
        self.containers = self

    def list(self, **kwargs: Any) -> list[FakeContainer]:
        wanted = dict(item.split('=', 1) for item in kwargs['filters']['label'])
        return [c for c in self.all if kwargs['all'] and c.labels == {**c.labels, **wanted}]

    def by(self, service: str) -> FakeContainer:
        return next(c for c in self.all if c.labels[SERVICE_LABEL] == service)


@pytest.fixture
def world(pg, monkeypatch):
    for table in ('gpu_reservations', 'gpu_devices', 'training_runs', 'jobs'):
        sql(pg, f'DELETE FROM {table}')
    sql(pg, 'DELETE FROM deployment_settings WHERE key = %s', state.KEY)
    config = GpuArbiterConfig(
        containers=('vlm', 'triton-server'),
        container_gpus=(('vlm', frozenset({2})), ('triton-server', None)),
        bakeoff_host_gpus=None,
    )
    monkeypatch.setattr(gpu_arbiter_config_module, '_default_gpu_arbiter_config', config)
    vlm = FakeContainer('vlm', 'running')
    triton = FakeContainer('triton-server', 'running')
    client = FakeClient(vlm, triton)
    monkeypatch.setattr(docker_control, 'docker_client', lambda: client)
    monkeypatch.setattr(docker_control, 'compose_project', lambda _client: 'op')
    return pg, make_project(pg), client


def _card(pg, n: int) -> str:
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


def hold(
    pg, project, kind: str = 'probe', *, gpu: str | None = '2', status: str = 'running'
) -> uuid.UUID:
    params = psycopg.types.json.Jsonb({'gpu': gpu})
    job = insert_row(pg, project, status, kind, params=params)
    sql(
        pg,
        'INSERT INTO gpu_reservations (gpu_uuid, holder_job_id) VALUES (%s, %s)',
        _card(pg, 2),
        job,
    )
    return job


def section(pg) -> dict[str, Any] | None:
    rows = sql(pg, 'SELECT value FROM deployment_settings WHERE key = %s', state.KEY)
    return rows[0][0].get(state.ARBITER) if rows else None


def tick() -> dict[str, Any]:
    return run_periodic('gpu_arbiter', gpu_arbiter.tick())


def test_a_scoped_claim_stops_its_service_and_is_confirmed(world) -> None:
    pg, project, client = world
    job = hold(pg, project, 'probe', gpu='2')

    result = tick()

    assert client.by('vlm').status == 'exited'
    assert client.by('triton-server').status == 'running'  # unscoped: only a multi-GPU claim
    record = section(pg)
    assert record['claims'] == [str(job)]
    assert record['acknowledged'] == [str(job)]
    assert record['stopped'] == ['vlm']
    assert record['error'] is None
    assert result['acknowledged'] == 1


def test_a_multi_gpu_train_claim_stops_every_service(world) -> None:
    pg, project, client = world
    job = insert_row(pg, project, 'running', 'train')
    sql(
        pg,
        'INSERT INTO training_runs (project_id, job_id, spec, class_names, family, base_model, '
        "artifact_prefix) VALUES (%s, %s, %s, ARRAY['a'], 'yolo26', 'm', 'p')",
        project,
        job,
        psycopg.types.json.Jsonb({'cuda_visible_devices': '0,2'}),
    )
    sql(
        pg,
        'INSERT INTO gpu_reservations (gpu_uuid, holder_job_id) VALUES (%s, %s)',
        _card(pg, 0),
        job,
    )

    tick()

    assert {c.labels[SERVICE_LABEL]: c.status for c in client.all} == {
        'vlm': 'exited',
        'triton-server': 'exited',
    }
    assert section(pg)['acknowledged'] == [str(job)]


def test_a_bakeoff_claims_every_configured_service_without_a_host_scope(world) -> None:
    pg, project, client = world
    job = insert_row(pg, project, 'running', 'bakeoff')
    sql(
        pg,
        'INSERT INTO gpu_reservations (gpu_uuid, holder_job_id) VALUES (%s, %s)',
        _card(pg, 0),
        job,
    )

    tick()

    assert {c.status for c in client.all} == {'exited'}
    assert section(pg)['acknowledged'] == [str(job)]


def test_when_the_claim_ends_only_what_the_arbiter_stopped_starts_again(world) -> None:
    pg, project, client = world
    client.by('triton-server').status = 'exited'  # somebody else stopped it
    job = hold(pg, project, 'bakeoff')
    tick()
    assert section(pg)['stopped'] == ['vlm']

    sql(pg, 'DELETE FROM gpu_reservations WHERE holder_job_id = %s', job)
    tick()

    assert client.by('vlm').status == 'running'
    assert client.by('triton-server').calls == []
    assert client.by('triton-server').status == 'exited'  # never ours to start
    assert section(pg) is None


def test_a_docker_failure_confirms_nothing_it_did_not_stop(world) -> None:
    pg, project, client = world
    client.by('vlm').fail_stop = True
    job = hold(pg, project, 'bakeoff')

    tick()

    record = section(pg)
    assert record['claims'] == [str(job)]
    assert record['acknowledged'] == []
    assert 'stop vlm: daemon refused' in record['error']


def test_an_unreachable_daemon_confirms_nothing(world, monkeypatch) -> None:
    pg, project, _client = world

    def boom() -> None:
        raise docker_control.ArbiterDockerError('the Docker daemon is not reachable: no socket')

    monkeypatch.setattr(docker_control, 'docker_client', boom)
    hold(pg, project, 'probe')

    tick()

    record = section(pg)
    assert record['acknowledged'] == []
    assert 'not reachable' in record['error']


def test_a_reservation_of_a_job_that_is_no_longer_running_is_released(world) -> None:
    pg, project, client = world
    hold(pg, project, 'probe', status='failed')

    result = tick()

    assert sql(pg, 'SELECT count(*) FROM gpu_reservations') == [(0,)]
    assert result['claims'] == 0
    assert client.by('vlm').status == 'running'
    assert section(pg) is None


def test_with_no_claim_and_nothing_stopped_the_tick_writes_nothing(world) -> None:
    pg, _project, client = world

    tick()

    assert sql(pg, 'SELECT count(*) FROM deployment_settings WHERE key = %s', state.KEY) == [(0,)]
    assert [c.calls for c in client.all] == [[], []]


def test_a_tick_that_cannot_take_the_lock_does_nothing(world) -> None:
    pg, project, client = world
    hold(pg, project, 'probe')
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as holder:
        holder.execute('SELECT pg_advisory_lock(%s, 0)', (GPU_ARBITER,))
        assert tick() == {'skipped': 'busy'}
        holder.execute('SELECT pg_advisory_unlock(%s, 0)', (GPU_ARBITER,))
    assert client.by('vlm').status == 'running'


def test_an_unconfigured_deployment_is_a_noop(world, monkeypatch) -> None:
    pg, project, client = world
    monkeypatch.setattr(
        gpu_arbiter_config_module, '_default_gpu_arbiter_config', GpuArbiterConfig()
    )
    hold(pg, project, 'probe')

    assert tick() == {'skipped': 'no GPU-resident services configured'}
    assert client.by('vlm').status == 'running'


def test_the_vlm_switch_section_survives_an_arbiter_pass(world) -> None:
    pg, project, _client = world
    sql(
        pg,
        'INSERT INTO deployment_settings (key, value, updated_by) VALUES (%s, %s, %s)',
        state.KEY,
        psycopg.types.json.Jsonb({'vlm_switch': {'at': 1.0e12}}),
        'vlm-switch',
    )
    hold(pg, project, 'probe')

    tick()

    [(value,)] = sql(pg, 'SELECT value FROM deployment_settings WHERE key = %s', state.KEY)
    assert value['vlm_switch'] == {'at': 1.0e12}
    assert value[state.ARBITER]['stopped'] == ['vlm']
