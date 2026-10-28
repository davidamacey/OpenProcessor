"""The application lifespan: the order of its startup and shutdown steps and how each
step's failure is handled.

Every collaborator is replaced where the lifespan looks it up (the lazily imported
service functions at their own modules, the pool and client factories on the lifespan
module), so the test records the steps without touching Triton, OpenSearch or Docker.
"""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI

from src.app import lifespan as lifespan_module
from src.app.resources import AppResources


class World:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.fail: set[str] = set()
        self.reload_log_events: list[str] = []
        self.arbiter_calls = 0

    def step(self, name: str) -> None:
        self.events.append(name)
        if name in self.fail:
            raise RuntimeError(f'{name} failed')


def _async_step(world: World, name: str, result: Any = None):
    async def _run(*_a: Any, **_k: Any) -> Any:
        world.step(name)
        return result

    return _run


def _sync_step(world: World, name: str, result: Any = None):
    def _run(*_a: Any, **_k: Any) -> Any:
        world.step(name)
        return result

    return _run


async def _forever() -> None:
    await asyncio.sleep(3600)


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    w = World()
    saved = {
        k: v for k, v in vars(AppResources).items() if not k.startswith('__') and not callable(v)
    }

    class FakePool:
        def __init__(self, **kwargs: Any) -> None:
            w.events.append(f'pool_created:{kwargs["pool_size"]}x{kwargs["max_concurrent"]}')

        async def initialize(self) -> None:
            w.step('pool_initialize')

        async def close(self) -> None:
            w.step('pool_close')

    class FakeTritonFactory:
        @staticmethod
        async def close_all() -> None:
            w.step('triton_clients_close')

    class FakeOpenSearchFactory:
        @staticmethod
        async def close() -> None:
            w.step('opensearch_close')

    class FakeHub:
        async def start_tail(self) -> None:
            w.step('event_hub_start')

        async def stop_tail(self) -> None:
            w.step('event_hub_stop')

    class FakeEncoder:
        text_backend = 'fake'

        def __init__(self, triton_pool: Any) -> None:
            w.events.append('pe_encoder_created')
            self.triton_pool = triton_pool

        def warm_text_encoder(self) -> None:
            w.step('pe_encoder_warm')

    async def fake_reconcile_on_startup() -> Any:
        w.arbiter_calls += 1
        w.step('arbiter_reconcile')
        return SimpleNamespace(action='noop', detail='stub')

    async def fake_reload(*, log_event: str) -> None:
        w.reload_log_events.append(log_event)
        w.step(f'reload:{log_event}')

    async def fake_stop_jobs(jobs: list[Any]) -> None:
        w.events.append(f'stop_background_jobs:{jobs}')

    async def fake_ops_refresh_loop() -> None:
        w.step('ops_metrics_loop')
        await _forever()

    arbiter_cfg = SimpleNamespace(
        allowed_gpu_ids=frozenset({2, 0}), containers=('c1',), trainer_container='trainer'
    )

    def fake_get_arbiter_config() -> Any:
        w.step('arbiter_config')
        return arbiter_cfg

    patches: list[tuple[str, str, Any]] = [
        ('src.config.retired_env', 'reject_retired_env', _sync_step(w, 'retired_env')),
        (
            'src.services.config_store.global_store',
            'global_configs_index',
            _sync_step(w, 'global_configs_index'),
        ),
        (
            'src.services.detection.profile_registry',
            'ensure_env_region_profile',
            _sync_step(w, 'region_profile'),
        ),
        (
            'src.services.projects.bootstrap',
            'startup_bootstrap_project_registry_safe',
            _async_step(w, 'project_registry_bootstrap'),
        ),
        ('src.services.curation.ops_metrics_refresh', 'refresh_loop', fake_ops_refresh_loop),
        (
            'src.core.dependencies',
            'bootstrap_opensearch_indexes',
            _async_step(w, 'opensearch_bootstrap'),
        ),
        (
            'src.services.config_store',
            'startup_bootstrap_config_store_safe',
            _async_step(w, 'config_store_bootstrap'),
        ),
        ('src.services.curation.event_hub', 'get_event_hub', FakeHub),
        (
            'src.services.curation.startup_reconcile',
            'reconcile_all_jobs',
            _sync_step(w, 'reconcile_jobs'),
        ),
        (
            'src.services.model_export',
            'reconcile_orphaned_export_tasks',
            _sync_step(w, 'reconcile_exports', 0),
        ),
        ('src.config', 'get_gpu_arbiter_config', fake_get_arbiter_config),
        ('src.services.training.gpu_arbiter', 'reconcile_on_startup', fake_reconcile_on_startup),
        ('src.services.training.triton_reload', 'reload_promoted_models_best_effort', fake_reload),
        ('src.clients.pe_encoder', 'PEEncoder', FakeEncoder),
        (
            'src.services.curation.background_jobs',
            'start_background_jobs',
            _sync_step(w, 'start_background_jobs', ['job-a']),
        ),
        ('src.services.curation.background_jobs', 'stop_background_jobs', fake_stop_jobs),
        (
            'src.services.projects.bootstrap',
            'shutdown_project_registry',
            _async_step(w, 'project_registry_shutdown'),
        ),
        (
            'src.services.config_store',
            'shutdown_config_store_poll',
            _async_step(w, 'config_store_shutdown'),
        ),
        ('src.app.lifespan', 'AsyncTritonPool', FakePool),
        ('src.app.lifespan', 'TritonClientFactory', FakeTritonFactory),
        ('src.app.lifespan', 'OpenSearchClientFactory', FakeOpenSearchFactory),
        ('src.app.lifespan', 'shutdown_cpu_executor', _sync_step(w, 'cpu_executor_shutdown')),
    ]
    for module, name, value in patches:
        monkeypatch.setattr(importlib.import_module(module), name, value)
    # importing the patched modules can already call a patched step
    w.events.clear()
    yield w
    for key, value in saved.items():
        setattr(AppResources, key, value)


def _run_lifespan(body=None) -> FastAPI:
    app = FastAPI()

    async def _drive() -> None:
        async with lifespan_module.lifespan(app):
            if body is not None:
                await body()

    asyncio.run(_drive())
    return app


EXPECTED_STARTUP = [
    'retired_env',
    'global_configs_index',
    'region_profile',
    'pool_created:4x64',
    'pool_initialize',
    'project_registry_bootstrap',
    'opensearch_bootstrap',
    'config_store_bootstrap',
    'event_hub_start',
    'reconcile_jobs',
    'reconcile_exports',
    'arbiter_config',
    'arbiter_reconcile',
    'reload:promoted_models_reload_on_startup',
    'pe_encoder_created',
    'pe_encoder_warm',
    'start_background_jobs',
]
EXPECTED_SHUTDOWN = [
    "stop_background_jobs:['job-a']",
    'project_registry_shutdown',
    'config_store_shutdown',
    'event_hub_stop',
    'pool_close',
    'cpu_executor_shutdown',
    'triton_clients_close',
    'opensearch_close',
]


def _normalized(events: list[str]) -> list[str]:
    # the ops-metrics task starts as soon as the loop first yields, between two steps
    return [e for e in events if e != 'ops_metrics_loop']


def test_startup_then_shutdown_run_every_step_in_order(world: World) -> None:
    seen: dict[str, Any] = {}

    async def body() -> None:
        seen['startup'] = list(world.events)
        seen['executor'] = AppResources.shared_executor
        seen['pool'] = AppResources.async_triton_pool
        seen['arbiter'] = AppResources.arbiter_task
        seen['ops'] = AppResources.ops_metrics_task
        seen['jobs'] = AppResources.curation_background_jobs
        seen['bus'] = AppResources.event_bus_started
        await asyncio.sleep(0)

    app = _run_lifespan(body)

    assert _normalized(seen['startup']) == EXPECTED_STARTUP
    assert 'ops_metrics_loop' in world.events
    assert seen['executor'] is not None
    assert seen['pool'] is not None
    assert seen['arbiter'] is not None
    assert seen['ops'] is not None
    assert seen['jobs'] == ['job-a']
    assert seen['bus'] is True
    assert app.state.pe_encoder.triton_pool is seen['pool']
    assert _normalized(world.events)[len(EXPECTED_STARTUP) :] == EXPECTED_SHUTDOWN
    # shutdown clears the tasks and flags but keeps the pool and executor references
    assert AppResources.ops_metrics_task is None
    assert AppResources.arbiter_task is None
    assert AppResources.curation_background_jobs == []
    assert AppResources.project_registry_poll_task is None
    assert AppResources.config_store_poll_task is None
    assert AppResources.event_bus_started is False
    assert AppResources.async_triton_pool is seen['pool']
    assert AppResources.shared_executor is seen['executor']
    assert seen['arbiter'].cancelled() or seen['arbiter'].done()


def test_a_failing_event_hub_start_is_skipped_and_its_stop_is_not_attempted(
    world: World,
) -> None:
    world.fail = {'event_hub_start'}
    _run_lifespan()
    assert 'event_hub_start' in world.events
    assert 'event_hub_stop' not in world.events
    assert 'reconcile_jobs' in world.events
    assert 'start_background_jobs' in world.events


def test_a_failing_event_hub_stop_does_not_stop_the_rest_of_shutdown(world: World) -> None:
    world.fail = {'event_hub_stop'}
    _run_lifespan()
    tail = _normalized(world.events)[len(EXPECTED_STARTUP) :]
    assert tail == EXPECTED_SHUTDOWN


def test_a_failing_export_reconcile_is_skipped(world: World) -> None:
    world.fail = {'reconcile_exports'}
    _run_lifespan()
    assert 'arbiter_config' in world.events
    assert 'start_background_jobs' in world.events


def test_a_failing_arbiter_setup_skips_the_loop_but_not_the_reload(world: World) -> None:
    world.fail = {'arbiter_reconcile'}
    seen: dict[str, Any] = {}

    async def body() -> None:
        seen['arbiter'] = AppResources.arbiter_task

    _run_lifespan(body)
    assert seen['arbiter'] is None
    assert world.reload_log_events == ['promoted_models_reload_on_startup']
    assert 'pe_encoder_warm' in world.events


def test_a_failing_pe_encoder_warmup_does_not_block_startup(world: World) -> None:
    world.fail = {'pe_encoder_warm'}
    app = _run_lifespan()
    assert app.state.pe_encoder is not None
    assert 'start_background_jobs' in world.events


@pytest.mark.parametrize('failing', ['pool_close', 'triton_clients_close', 'opensearch_close'])
def test_a_failing_shutdown_step_does_not_skip_the_ones_after_it(
    world: World, failing: str
) -> None:
    world.fail = {failing}
    _run_lifespan()
    assert _normalized(world.events)[len(EXPECTED_STARTUP) :] == EXPECTED_SHUTDOWN


def test_a_failing_cpu_pool_shutdown_is_logged_by_step_and_skips_nothing(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    logged: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        lifespan_module.logger, 'warning', lambda event, **kw: logged.append((event, kw))
    )
    world.fail = {'cpu_executor_shutdown'}
    _run_lifespan()
    assert _normalized(world.events)[len(EXPECTED_STARTUP) :] == EXPECTED_SHUTDOWN
    assert (
        'shutdown_step_failed',
        {'step': 'cpu_executor', 'error': 'cpu_executor_shutdown failed'},
    ) in logged


@pytest.mark.parametrize('failing', ['project_registry_shutdown', 'config_store_shutdown'])
def test_a_failing_background_stop_does_not_skip_the_rest(world: World, failing: str) -> None:
    world.fail = {failing}
    _run_lifespan()
    assert _normalized(world.events)[len(EXPECTED_STARTUP) :] == EXPECTED_SHUTDOWN


def test_a_failing_background_job_stop_does_not_skip_the_rest(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(_jobs: list[Any]) -> None:
        world.events.append('stop_background_jobs:boom')
        raise RuntimeError('jobs stuck')

    monkeypatch.setattr('src.services.curation.background_jobs.stop_background_jobs', boom)
    _run_lifespan()
    tail = _normalized(world.events)[len(EXPECTED_STARTUP) :]
    assert tail == ['stop_background_jobs:boom', *EXPECTED_SHUTDOWN[1:]]


def test_startup_failures_before_serving_propagate(world: World) -> None:
    world.fail = {'region_profile'}
    with pytest.raises(RuntimeError, match='region_profile failed'):
        _run_lifespan()
    assert 'pool_created:4x64' not in world.events


def test_the_arbiter_loop_reconciles_on_a_timer_and_survives_a_failing_tick(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('src.app.lifespan.ARBITER_RECONCILE_INTERVAL_SECONDS', 0.01)
    calls = {'n': 0}

    async def flaky_reconcile() -> Any:
        calls['n'] += 1
        if calls['n'] == 2:
            raise RuntimeError('tick failed')
        return SimpleNamespace(action='noop', detail='stub')

    monkeypatch.setattr('src.services.training.gpu_arbiter.reconcile_on_startup', flaky_reconcile)

    async def body() -> None:
        for _ in range(100):
            if calls['n'] >= 4:
                return
            await asyncio.sleep(0.01)

    _run_lifespan(body)
    assert calls['n'] >= 4
    # one reload at startup, one per tick (the failing tick still reloads)
    assert world.reload_log_events[0] == 'promoted_models_reload_on_startup'
    assert world.reload_log_events.count('promoted_models_reload_tick') >= 3


def _app_with_nested_routers() -> FastAPI:
    from fastapi import APIRouter

    app = FastAPI()
    outer = APIRouter(prefix='/outer')
    inner = APIRouter(prefix='/inner')

    @inner.get('/ping')
    async def ping() -> dict[str, bool]:
        return {'ok': True}

    outer.include_router(inner)
    app.include_router(outer)
    app.include_router(APIRouter(prefix='/other'))
    return app


def _unbuilt_included_routers(routes: list[Any]) -> list[str]:
    """Included routers whose lazily built route state does not exist yet (FastAPI >= 0.143)."""
    from fastapi.routing import _IncludedRouter

    missing: list[str] = []
    for route in routes:
        if not isinstance(route, _IncludedRouter):
            continue
        if route._effective_candidates_version is None:
            missing.append(route.include_context.prefix)
            continue
        missing.extend(_unbuilt_included_routers(route._effective_candidates))
    return missing


def test_startup_builds_the_lazy_route_state_of_every_router(world: World) -> None:
    app = _app_with_nested_routers()
    assert _unbuilt_included_routers(app.router.routes)  # lazily built: nothing exists yet
    seen: dict[str, Any] = {}

    async def _drive() -> None:
        async with lifespan_module.lifespan(app):
            seen['unbuilt'] = _unbuilt_included_routers(app.router.routes)

    asyncio.run(_drive())
    assert seen['unbuilt'] == []


def test_a_failing_route_warmup_never_blocks_startup(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError('asgi warm-up failed')

    app = _app_with_nested_routers()
    monkeypatch.setattr(type(app), '__call__', _boom)
    seen: dict[str, Any] = {}

    async def _drive() -> None:
        async with lifespan_module.lifespan(app):
            seen['up'] = 'start_background_jobs' in world.events

    asyncio.run(_drive())
    assert seen['up'] is True
