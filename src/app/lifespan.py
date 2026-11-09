"""Application lifecycle: startup steps and shutdown."""

import asyncio
import contextlib
import inspect
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.app import startup_gate
from src.app.resources import AppResources
from src.clients.triton_pool_core import AsyncTritonPool
from src.config import Settings, get_settings
from src.core.dependencies import OpenSearchClientFactory, TritonClientFactory
from src.core.logging import get_logger
from src.db.engine import dispose_engines
from src.utils.cpu_pool import shutdown_cpu_executor
from src.utils.file_read import shutdown_read_executor
from src.utils.metrics_processes import forget_process


logger = get_logger(__name__)


def _preflight() -> None:
    """Refuse to start on a misconfiguration that would otherwise serve a broken store."""
    # Fail loudly on any retired env-var name before anything else initializes.
    from src.config.retired_env import reject_retired_env

    reject_retired_env()

    # A malformed OP_FACES_ENABLED refuses startup instead of guessing a privacy switch.
    from src.config.faces import faces_enabled

    faces_enabled()

    # The lifespan runs unbound; steps that touch project data bind each
    # project in turn (src.services.projects.bootstrap). Resolve the region
    # profile before serving: a bad profile file fails startup up front.
    from src.services.detection.profile_registry import ensure_env_region_profile

    ensure_env_region_profile()


async def _start_pools(settings: Settings) -> None:
    """The shared CPU executor and the high-throughput async Triton pool."""
    # Shared ThreadPoolExecutor for CPU-bound tasks (JPEG decode, resize).
    AppResources.shared_executor = ThreadPoolExecutor(
        max_workers=64,
        thread_name_prefix='ingest-worker-',
    )
    logger.info('executor_initialized', workers=64, type='ThreadPoolExecutor')

    # High-throughput async Triton pool: 4 gRPC channels = 4 TCP connections.
    AppResources.async_triton_pool = AsyncTritonPool(
        url=settings.triton_url,
        pool_size=4,
        max_concurrent=64,
        verbose=False,
    )
    await AppResources.async_triton_pool.initialize()
    logger.info('triton_pool_initialized', channels=4, max_concurrent=64)


async def _report_stored_face_data() -> None:
    """Faces off: say at startup whether face data is still stored (counts only)."""
    from src.config.faces import faces_enabled

    if faces_enabled():
        return
    from opensearchpy import OpenSearchException

    from src.services.face_privacy import stored_face_counts, stored_summary

    try:
        counts = await stored_face_counts()
    except (TimeoutError, OSError, RuntimeError, OpenSearchException) as exc:
        logger.warning('faces_disabled_state_unknown', error_type=type(exc).__name__)
        return
    if counts.is_empty():
        logger.info('faces_disabled')
    else:
        logger.warning('faces_disabled_with_stored_data', detail=stored_summary(counts))


async def _start_background_tasks() -> None:
    """Project registry, ops gauges, curation indexes and the config store poll."""
    # Projects foundation (P1): default record + poll loop, best-effort.
    from src.services.projects.bootstrap import startup_bootstrap_project_registry_safe

    AppResources.project_registry_poll_task = await startup_bootstrap_project_registry_safe()

    # Snapshot gauges (queue depth, embedding state, shards) for the dashboards.
    from src.services.curation.ops_metrics_refresh import refresh_loop as ops_metrics_refresh_loop

    AppResources.ops_metrics_task = asyncio.create_task(ops_metrics_refresh_loop())

    # Refuse to start when an active project's index mapping stamp is not current.
    from src.core.dependencies import bootstrap_opensearch_indexes

    await bootstrap_opensearch_indexes()
    await _report_stored_face_data()

    # Config store poll task: fans out per active project, not `default`.
    from src.services.config_store import startup_bootstrap_config_store_safe

    AppResources.config_store_poll_task = await startup_bootstrap_config_store_safe()


async def _start_event_listener() -> None:
    # One Valkey subscription per process; a Valkey outage leaves its 2 s database tick running.
    try:
        from src.services.events.listener import start_listener

        await start_listener()
        AppResources.event_bus_started = True
        logger.info('event_listener_started')
    except Exception as exc:
        logger.warning('event_listener_start_skipped', error=str(exc))


def _warm_pe_encoder(app: FastAPI) -> None:
    # Best-effort: warm the PE-Core text encoder for GET /curation/search/text
    # (ONNX Runtime when OP_PE_TEXT_ONNX_PATH exists, else Triton's
    # pe_text_encoder if ready, else PyTorch — see src/clients/pe_encoder.py).
    # Non-fatal if nothing can load — the search endpoint surfaces a 503 in
    # that case rather than the whole service failing to start.
    from src.clients.pe_encoder import PEEncoder

    app.state.pe_encoder = PEEncoder(triton_pool=AppResources.async_triton_pool)
    try:
        app.state.pe_encoder.warm_text_encoder()
        app.state.pe_encoder.prime_text_backend()
        logger.info('pe_text_encoder_warmed', backend=app.state.pe_encoder.text_backend)
    except Exception as exc:
        logger.warning('pe_text_encoder_warm_skipped', error=str(exc))


async def _warm_routes(app: FastAPI) -> None:
    """Build FastAPI's lazily constructed route state before the first real request.

    FastAPI 0.143 builds the effective route tables of every included router on the first
    request that is matched against them (about 0.9 s for 321 routes, once per uvicorn
    worker). One request that matches no route walks every router, so the cost is paid here
    instead of by the first search a worker answers (#356). A no-match request runs no
    handler and no dependency, so there is no side effect and no auth or project guard to
    bypass. Best-effort: a failure is logged and startup continues.
    """
    status: list[int] = []

    async def receive() -> dict[str, object]:
        return {'type': 'http.request', 'body': b'', 'more_body': False}

    async def send(message: dict[str, object]) -> None:
        if message['type'] == 'http.response.start':
            status.append(int(str(message['status'])))

    scope = {
        'type': 'http',
        'asgi': {'version': '3.0'},
        'http_version': '1.1',
        'method': 'GET',
        'scheme': 'http',
        'path': '/_route_warmup_no_match',
        'raw_path': b'/_route_warmup_no_match',
        'query_string': b'',
        'root_path': '',
        'headers': [(b'host', b'localhost')],
        'client': ('127.0.0.1', 0),
        'server': ('localhost', 80),
        'app': app,
    }
    try:
        await app(scope, receive, send)  # type: ignore[arg-type]
        logger.info('route_state_warmed', status=status[0] if status else None)
    except (RuntimeError, OSError, ValueError) as exc:
        logger.warning('route_state_warm_skipped', error=str(exc))


async def _shutdown_step(step: str, run: Callable[[], Awaitable[None] | None]) -> None:
    """Run one shutdown step on its own: a failure is logged with the step name and
    never stops the steps after it. Startup is not routed through here, so a startup
    failure still propagates."""
    try:
        result = run()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        logger.warning('shutdown_step_failed', step=step, error=str(exc))


async def _stop_ops_metrics() -> None:
    if AppResources.ops_metrics_task is not None:
        AppResources.ops_metrics_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await AppResources.ops_metrics_task
        AppResources.ops_metrics_task = None


async def _stop_project_registry() -> None:
    from src.services.projects.bootstrap import shutdown_project_registry

    task = AppResources.project_registry_poll_task
    AppResources.project_registry_poll_task = None
    await shutdown_project_registry(task)


async def _stop_config_store_poll() -> None:
    from src.services.config_store import shutdown_config_store_poll

    task = AppResources.config_store_poll_task
    AppResources.config_store_poll_task = None
    await shutdown_config_store_poll(task)


async def _stop_event_bus() -> None:
    if AppResources.event_bus_started:
        AppResources.event_bus_started = False
        from src.services.events.listener import stop_listener

        await stop_listener()
        logger.info('event_listener_stopped')


async def _close_triton_pool() -> None:
    if AppResources.async_triton_pool is not None:
        await AppResources.async_triton_pool.close()
        logger.info('triton_pool_closed')


def _shutdown_shared_executor() -> None:
    if AppResources.shared_executor is not None:
        AppResources.shared_executor.shutdown(wait=True)
        logger.info('executor_shutdown')


async def _run_shutdown() -> None:
    steps: list[tuple[str, Callable[[], Awaitable[None] | None]]] = [
        ('ops_metrics', _stop_ops_metrics),
        ('project_registry', _stop_project_registry),
        ('config_store_poll', _stop_config_store_poll),
        ('event_bus', _stop_event_bus),
        ('triton_pool', _close_triton_pool),
        # the ingest CPU pool: let running pixel work finish
        ('cpu_executor', shutdown_cpu_executor),
        ('read_executor', shutdown_read_executor),
        ('shared_executor', _shutdown_shared_executor),
        ('triton_clients', TritonClientFactory.close_all),
        ('opensearch_clients', OpenSearchClientFactory.close),
        ('database', dispose_engines),
        ('metrics_files', lambda: forget_process(os.getpid())),
    ]
    for step, run in steps:
        await _shutdown_step(step, run)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifecycle manager.

    Args:
        app: FastAPI application instance (required by API contract, not used in implementation)

    Startup:
    - Refuse to start unless the database, queue, OpenSearch and Triton answer and the
      database is at this image's schema head
    - Create shared ThreadPoolExecutor for CPU-bound tasks
    - Create AsyncTritonPool for high-throughput inference
    - Shared Triton gRPC client auto-created on first use
    - Reconcile orphaned job state left behind by a killed process

    Shutdown:
    - Close AsyncTritonPool
    - Shutdown shared ThreadPoolExecutor
    - Close all Triton gRPC connections
    - Close OpenSearch connections
    - Dispose the database pools
    """
    settings = get_settings()

    # =========================================================================
    # STARTUP
    # =========================================================================
    logger.info('startup_begin', phase='initialization')

    _preflight()
    await startup_gate.require_control_plane(settings)
    await _start_pools(settings)
    await _start_background_tasks()
    await _start_event_listener()
    _warm_pe_encoder(app)
    await _warm_routes(app)

    logger.info(
        'service_ready',
        triton_url=settings.triton_url,
        opensearch_url=settings.opensearch_url,
    )

    yield

    # =========================================================================
    # SHUTDOWN
    # =========================================================================
    logger.info('shutdown_begin', phase='cleanup')

    await _run_shutdown()

    logger.info('shutdown_complete')
