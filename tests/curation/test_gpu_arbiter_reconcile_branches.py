"""``reconcile_on_startup`` branch by branch: what counts as an active run, which containers
stop or start, in what order, and which project is reported as the holder."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import pytest

from src.config import GpuArbiterConfig, gpu_arbiter as gpu_arbiter_config_module
from src.services.training import gpu_arbiter as ga, gpu_arbiter_docker, gpu_arbiter_policy


if TYPE_CHECKING:
    from pathlib import Path


class _Container:
    def __init__(self, name: str, status: str, calls: list[tuple[str, str]]) -> None:
        self.name = name
        self.status = status
        self._calls = calls

    def stop(self, timeout: int = 30) -> None:  # noqa: ARG002 - matches the docker SDK
        self._calls.append(('stop', self.name))
        self.status = 'exited'

    def start(self) -> None:
        self._calls.append(('start', self.name))
        self.status = 'running'

    def reload(self) -> None:
        return None


class _Containers:
    def __init__(self, registry: dict[str, _Container]) -> None:
        self._registry = registry

    def get(self, name: str) -> _Container:
        import docker.errors

        try:
            return self._registry[name]
        except KeyError as exc:
            raise docker.errors.NotFound(name) from exc


class _Client:
    def __init__(self, registry: dict[str, _Container]) -> None:
        self.containers = _Containers(registry)


class _Env:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.calls: list[tuple[str, str]] = []
        self.registry: dict[str, _Container] = {}
        self.sentinel = tmp_path / 'pause.sentinel'
        monkeypatch.setattr(gpu_arbiter_policy, '_state_dir', lambda: tmp_path)
        monkeypatch.setattr(gpu_arbiter_docker, '_docker_client', lambda: _Client(self.registry))

    def configure(
        self,
        scoped: dict[str, frozenset[int] | None],
        *,
        status: str,
        bakeoff_host_gpus: str | None = None,
    ) -> None:
        self.registry.update({n: _Container(n, status, self.calls) for n in scoped})
        cfg = GpuArbiterConfig(
            containers=tuple(scoped),
            container_gpus=tuple(scoped.items()),
            bakeoff_host_gpus=bakeoff_host_gpus,
        )
        self.monkeypatch.setattr(gpu_arbiter_config_module, '_default_gpu_arbiter_config', cfg)

    def jobs_dir(self, name: str = 'jobs') -> Path:
        path = self.tmp_path / name
        path.mkdir(exist_ok=True)
        return path

    async def reconcile(self, jobs_dir: Path | None = None) -> Any:
        return await ga.reconcile_on_startup(
            train_jobs_dir=jobs_dir if jobs_dir is not None else self.jobs_dir(),
            sentinel=self.sentinel,
        )

    def statuses(self) -> dict[str, str]:
        return {name: c.status for name, c in self.registry.items()}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Env:
    return _Env(tmp_path, monkeypatch)


def _job(jobs_dir: Path, job_id: str, cvd: str | None = None, *, raw: str | None = None) -> None:
    text = raw if raw is not None else json.dumps({'job_id': job_id, 'cuda_visible_devices': cvd})
    (jobs_dir / f'{job_id}.job.json').write_text(text)


def _status(jobs_dir: Path, job_id: str, state: str | None) -> None:
    (jobs_dir / f'{job_id}.status.json').write_text(json.dumps({'job_id': job_id, 'state': state}))


@pytest.mark.asyncio
async def test_run_without_a_device_list_stops_every_configured_container(env: _Env) -> None:
    env.configure({'svc-a': None, 'svc-b': frozenset({2})}, status='running')
    jobs_dir = env.jobs_dir()
    _job(jobs_dir, 'run', None)
    _status(jobs_dir, 'run', 'running')
    result = await env.reconcile(jobs_dir)
    assert result.action == 'gpu_services_stopped'
    assert result.detail == 'svc-a,svc-b'
    assert env.statuses() == {'svc-a': 'exited', 'svc-b': 'exited'}
    assert env.sentinel.exists()


@pytest.mark.asyncio
async def test_unreadable_job_file_counts_as_an_unscoped_claim(env: _Env) -> None:
    env.configure({'svc-a': None, 'svc-b': frozenset({2})}, status='running')
    jobs_dir = env.jobs_dir()
    _job(jobs_dir, 'run', raw='{not json')
    result = await env.reconcile(jobs_dir)
    assert result.action == 'gpu_services_stopped'
    assert env.statuses() == {'svc-a': 'exited', 'svc-b': 'exited'}


@pytest.mark.asyncio
async def test_start_runs_before_stop_and_stop_follows_configured_order(env: _Env) -> None:
    env.configure({'first': frozenset({2}), 'idle': None, 'last': frozenset({2})}, status='running')
    env.registry['idle'].status = 'exited'
    jobs_dir = env.jobs_dir()
    _job(jobs_dir, 'run', '2')
    result = await env.reconcile(jobs_dir)
    assert env.calls == [('start', 'idle'), ('stop', 'first'), ('stop', 'last')]
    # the reported action is the start result when anything was started
    assert result.action == 'gpu_services_started'
    assert result.detail == 'idle'
    assert env.sentinel.exists()


@pytest.mark.asyncio
async def test_only_stops_reports_the_stop_result(env: _Env) -> None:
    env.configure({'svc': frozenset({2})}, status='running')
    jobs_dir = env.jobs_dir()
    _job(jobs_dir, 'run', '2')
    result = await env.reconcile(jobs_dir)
    assert result.action == 'gpu_services_stopped'
    assert env.statuses() == {'svc': 'exited'}


@pytest.mark.asyncio
async def test_active_run_with_nothing_configured_only_pauses_the_worker(env: _Env) -> None:
    env.configure({}, status='running')
    jobs_dir = env.jobs_dir()
    _job(jobs_dir, 'run', '0')
    result = await env.reconcile(jobs_dir)
    assert result.action == 'sentinel_set'
    assert env.sentinel.exists()
    again = await env.reconcile(jobs_dir)
    assert again.action == 'noop'


@pytest.mark.asyncio
@pytest.mark.parametrize('state', ['finished', 'failed', 'cancelled', 'skipped'])
async def test_terminal_status_means_the_run_is_over(env: _Env, state: str) -> None:
    env.configure({'svc': frozenset({2})}, status='exited')
    env.sentinel.touch()
    jobs_dir = env.jobs_dir()
    _job(jobs_dir, 'run', '2')
    _status(jobs_dir, 'run', state)
    result = await env.reconcile(jobs_dir)
    assert result.action == 'gpu_services_started'
    assert env.statuses() == {'svc': 'running'}
    assert not env.sentinel.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('state', ['queued', 'running', 'exporting', 'brand_new_state', None])
async def test_any_non_terminal_status_keeps_the_run_live(env: _Env, state: str | None) -> None:
    env.configure({'svc': frozenset({2})}, status='running')
    jobs_dir = env.jobs_dir()
    _job(jobs_dir, 'run', '2')
    _status(jobs_dir, 'run', state)
    result = await env.reconcile(jobs_dir)
    assert result.action == 'gpu_services_stopped'
    assert env.statuses() == {'svc': 'exited'}


@pytest.mark.asyncio
async def test_status_file_without_a_job_file_is_not_a_run(env: _Env) -> None:
    env.configure({'svc': frozenset({2})}, status='exited')
    jobs_dir = env.jobs_dir()
    _status(jobs_dir, 'orphan', 'running')
    (jobs_dir / 'junk.status.json').write_text('{broken')
    result = await env.reconcile(jobs_dir)
    assert result.action == 'gpu_services_started'
    assert env.statuses() == {'svc': 'running'}


@pytest.mark.asyncio
async def test_missing_jobs_dir_is_skipped(env: _Env) -> None:
    env.configure({'svc': frozenset({2})}, status='exited')
    result = await ga.reconcile_on_startup(
        train_jobs_dir=env.tmp_path / 'absent', sentinel=env.sentinel
    )
    assert result.action == 'gpu_services_started'
    assert result.holder_project is None


@pytest.mark.asyncio
async def test_fresh_training_lock_keeps_its_devices_claimed(env: _Env) -> None:
    env.configure({'svc': frozenset({2}), 'other': frozenset({0})}, status='running')
    gpu_arbiter_policy.set_training_lock('2')
    result = await env.reconcile()
    assert env.statuses() == {'svc': 'exited', 'other': 'running'}
    # 'other' is outside the claim, so it is (re)started first: that is the reported result
    assert result.action == 'gpu_services_started'
    assert result.detail == 'all already running'
    assert gpu_arbiter_policy.read_training_lock() is not None


@pytest.mark.asyncio
async def test_fresh_lock_without_devices_claims_everything(env: _Env) -> None:
    env.configure({'svc': frozenset({2}), 'other': frozenset({0})}, status='running')
    target = gpu_arbiter_policy.lock_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'claimed_at': time.time(), 'cuda_visible_devices': None}))
    await env.reconcile()
    assert env.statuses() == {'svc': 'exited', 'other': 'exited'}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'payload',
    [
        {'claimed_at': 1.0, 'cuda_visible_devices': '2'},
        {'cuda_visible_devices': '2'},
        {'claimed_at': None, 'cuda_visible_devices': '2'},
    ],
    ids=['expired', 'no_timestamp', 'null_timestamp'],
)
async def test_stale_training_lock_is_ignored_and_cleared(env: _Env, payload: dict) -> None:
    env.configure({'svc': frozenset({2})}, status='exited')
    target = gpu_arbiter_policy.lock_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload))
    result = await env.reconcile()
    assert result.action == 'gpu_services_started'
    assert env.statuses() == {'svc': 'running'}
    assert not target.exists()


@pytest.mark.asyncio
async def test_queued_bakeoff_claims_every_container_without_a_host_scope(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.configure({'svc': frozenset({2}), 'other': frozenset({0})}, status='running')
    monkeypatch.setattr(ga, 'bakeoff_active', lambda: True)
    await env.reconcile()
    assert env.statuses() == {'svc': 'exited', 'other': 'exited'}


@pytest.mark.asyncio
async def test_queued_bakeoff_with_a_host_scope_claims_only_intersecting_containers(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.configure(
        {'svc': frozenset({2}), 'other': frozenset({0})}, status='running', bakeoff_host_gpus='2'
    )
    monkeypatch.setattr(ga, 'bakeoff_active', lambda: True)
    await env.reconcile()
    assert env.statuses() == {'svc': 'exited', 'other': 'running'}


@pytest.mark.asyncio
async def test_claims_from_several_runs_are_unioned(env: _Env) -> None:
    env.configure({'a': frozenset({0}), 'b': frozenset({1}), 'c': frozenset({2})}, status='running')
    jobs_dir = env.jobs_dir()
    _job(jobs_dir, 'run1', '0')
    _job(jobs_dir, 'run2', '2')
    await env.reconcile(jobs_dir)
    assert env.statuses() == {'a': 'exited', 'b': 'running', 'c': 'exited'}


@pytest.mark.asyncio
async def test_idle_reconcile_clears_lock_and_sentinel_and_starts_everything(env: _Env) -> None:
    env.configure({'svc': frozenset({2})}, status='exited')
    env.sentinel.touch()
    default_sentinel = gpu_arbiter_policy.sentinel_path()
    default_sentinel.parent.mkdir(parents=True, exist_ok=True)
    default_sentinel.touch()
    lock = gpu_arbiter_policy.lock_path()
    lock.write_text(json.dumps({'claimed_at': 1.0, 'cuda_visible_devices': '2'}))
    result = await env.reconcile()
    assert result.action == 'gpu_services_started'
    assert not lock.exists()
    assert not env.sentinel.exists()
    # only the sentinel passed in is resumed; the arbiter's start step then resumes the default one
    assert not default_sentinel.exists()


@pytest.mark.asyncio
async def test_holder_project_is_the_first_project_with_an_active_run(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.configure({'svc': frozenset({2})}, status='running')
    idle, first, second = env.jobs_dir('p_idle'), env.jobs_dir('p_first'), env.jobs_dir('p_second')
    _job(idle, 'done', '2')
    _status(idle, 'done', 'finished')
    _job(first, 'live1', '2')
    _job(second, 'live2', '2')
    monkeypatch.setattr(
        ga, 'all_train_jobs_dirs', lambda: {'idle': idle, 'first': first, 'second': second}
    )
    result = await ga.reconcile_on_startup(sentinel=env.sentinel)
    assert result.holder_project == 'first'
    assert result.action == 'gpu_services_stopped'


@pytest.mark.asyncio
async def test_an_explicit_jobs_dir_is_attributed_to_no_project(env: _Env) -> None:
    env.configure({'svc': frozenset({2})}, status='running')
    jobs_dir = env.jobs_dir()
    _job(jobs_dir, 'live', '2')
    result = await env.reconcile(jobs_dir)
    assert result.holder_project is None


@pytest.mark.asyncio
async def test_lock_alone_reports_no_holder_project(env: _Env) -> None:
    env.configure({'svc': frozenset({2})}, status='running')
    gpu_arbiter_policy.set_training_lock('2')
    result = await env.reconcile()
    assert result.holder_project is None


@pytest.mark.asyncio
async def test_status_states_are_shared_across_project_dirs_by_stem(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Current behavior: a terminal status in one project dir also ends a same-named job
    listed later in another project dir."""
    env.configure({'svc': frozenset({2})}, status='exited')
    first, second = env.jobs_dir('p_first'), env.jobs_dir('p_second')
    _status(first, 'same', 'finished')
    _job(second, 'same', '2')
    monkeypatch.setattr(ga, 'all_train_jobs_dirs', lambda: {'first': first, 'second': second})
    result = await ga.reconcile_on_startup(sentinel=env.sentinel)
    assert result.action == 'gpu_services_started'
    assert env.statuses() == {'svc': 'running'}
