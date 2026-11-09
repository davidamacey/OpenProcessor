"""The application lifespan: the order of its startup and shutdown steps and how each
step's failure is handled.

Every collaborator is replaced where the lifespan looks it up (the lazily imported
service functions at their own modules, the pool and client factories on the lifespan
module), so the test records the steps without touching Triton, OpenSearch or Docker.
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest
from fastapi import FastAPI

from src.app import lifespan as lifespan_module
from src.app.resources import AppResources


class World:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.fail: set[str] = set()

    def step(self, name: str) -> None:
        self.events.append(name)
        if name in self.fail:
            raise RuntimeError(f'{name} failed')


def _async_step(world: World, name: str, result: Any = None):
    async def _run(*_a: Any, **_k: Any) -> Any:
        world.step(name)
        return result

    return _run


async def _fake_lease_client() -> object:
    return object()


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

    class FakeEncoder:
        text_backend = 'fake'

        def __init__(self, triton_pool: Any) -> None:
            w.events.append('pe_encoder_created')
            self.triton_pool = triton_pool

        def warm_text_encoder(self) -> None:
            w.step('pe_encoder_warm')

    async def fake_ops_refresh_loop() -> None:
        w.step('ops_metrics_loop')
        await _forever()

    patches: list[tuple[str, str, Any]] = [
        ('src.config.retired_env', 'reject_retired_env', _sync_step(w, 'retired_env')),
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
        ('src.services.events.listener', 'start_listener', _async_step(w, 'event_listener_start')),
        ('src.services.events.listener', 'stop_listener', _async_step(w, 'event_listener_stop')),
        (
            'src.services.projects.guard',
            'make_curation_opensearch',
            _fake_lease_client,
        ),
        ('src.clients.pe_encoder', 'PEEncoder', FakeEncoder),
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
    'region_profile',
    'pool_created:4x64',
    'pool_initialize',
    'project_registry_bootstrap',
    'opensearch_bootstrap',
    'config_store_bootstrap',
    'event_listener_start',
    'pe_encoder_created',
    'pe_encoder_warm',
]
EXPECTED_SHUTDOWN = [
    'project_registry_shutdown',
    'config_store_shutdown',
    'event_listener_stop',
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
        seen['ops'] = AppResources.ops_metrics_task
        seen['bus'] = AppResources.event_bus_started
        await asyncio.sleep(0)

    app = _run_lifespan(body)

    assert _normalized(seen['startup']) == EXPECTED_STARTUP
    assert 'ops_metrics_loop' in world.events
    assert seen['executor'] is not None
    assert seen['pool'] is not None
    assert seen['ops'] is not None
    assert seen['bus'] is True
    assert app.state.pe_encoder.triton_pool is seen['pool']
    assert _normalized(world.events)[len(EXPECTED_STARTUP) :] == EXPECTED_SHUTDOWN
    # shutdown clears the tasks and flags but keeps the pool and executor references
    assert AppResources.ops_metrics_task is None
    assert AppResources.project_registry_poll_task is None
    assert AppResources.config_store_poll_task is None
    assert AppResources.event_bus_started is False
    assert AppResources.async_triton_pool is seen['pool']
    assert AppResources.shared_executor is seen['executor']


def test_a_failing_event_listener_start_is_skipped_and_its_stop_is_not_attempted(
    world: World,
) -> None:
    world.fail = {'event_listener_start'}
    _run_lifespan()
    assert 'event_listener_start' in world.events
    assert 'event_listener_stop' not in world.events
    assert 'pe_encoder_warm' in world.events


def test_a_failing_event_listener_stop_does_not_stop_the_rest_of_shutdown(world: World) -> None:
    world.fail = {'event_listener_stop'}
    _run_lifespan()
    tail = _normalized(world.events)[len(EXPECTED_STARTUP) :]
    assert tail == EXPECTED_SHUTDOWN


def test_a_failing_pe_encoder_warmup_does_not_block_startup(world: World) -> None:
    world.fail = {'pe_encoder_warm'}
    app = _run_lifespan()
    assert app.state.pe_encoder is not None
    assert 'pe_encoder_warm' in world.events


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


def test_startup_failures_before_serving_propagate(world: World) -> None:
    world.fail = {'region_profile'}
    with pytest.raises(RuntimeError, match='region_profile failed'):
        _run_lifespan()
    assert 'pool_created:4x64' not in world.events


def test_the_api_runs_no_arbiter_loop() -> None:
    """The GPU arbiter is the ``op.periodic.gpu_arbiter`` beat tick in gpu-arbiter-worker; the API
    lifespan starts none and the API holds no Docker socket."""
    assert not hasattr(lifespan_module, '_start_gpu_arbiter')
    assert not hasattr(AppResources, 'arbiter_task')


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
            seen['up'] = 'pe_encoder_warm' in world.events

    asyncio.run(_drive())
    assert seen['up'] is True
