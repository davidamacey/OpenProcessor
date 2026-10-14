"""Application lifecycle: startup steps, the GPU-arbiter reconcile loop and shutdown."""

import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.app.resources import AppResources
from src.clients.triton_pool_core import AsyncTritonPool
from src.config import get_settings
from src.core.dependencies import OpenSearchClientFactory, TritonClientFactory
from src.core.logging import get_logger
from src.utils.cpu_pool import shutdown_cpu_executor


logger = get_logger(__name__)

# How often the GPU-arbiter reconcile loop re-asserts the desired state of
# the deployment's GPU-resident containers. Short enough that a crashed
# trainer's claim is released within seconds, long enough to be free in
# steady state (each tick is a directory glob plus, at most, no-op docker
# stop/start calls on containers already in the target state).
ARBITER_RECONCILE_INTERVAL_SECONDS = 15.0


def _preflight() -> None:
    """Refuse to start on a misconfiguration that would otherwise serve a broken store."""
    # Fail loudly on any retired env-var name before anything else initializes.
    from src.config.retired_env import reject_retired_env

    reject_retired_env()

    # A global config index name inside the project namespace is a
    # misconfiguration: refuse to start rather than serve a broken store.
    from src.services.config_store.global_store import global_configs_index

    global_configs_index()

    # The lifespan runs unbound; steps that touch project data bind each
    # project in turn (src.services.projects.bootstrap). Resolve the region
    # profile before serving: a bad profile file fails startup up front.
    from src.services.detection.profile_registry import ensure_env_region_profile

    ensure_env_region_profile()


async def _start_pools(settings) -> None:
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


async def _start_background_tasks() -> None:
    """Project registry, ops gauges, curation indexes and the config store poll."""
    # Projects foundation (P1): default record + poll loop, best-effort.
    from src.services.projects.bootstrap import startup_bootstrap_project_registry_safe

    AppResources.project_registry_poll_task = await startup_bootstrap_project_registry_safe()

    # Snapshot gauges (queue depth, embedding state, shards) for the dashboards.
    from src.services.curation.ops_metrics_refresh import refresh_loop as ops_metrics_refresh_loop

    AppResources.ops_metrics_task = asyncio.create_task(ops_metrics_refresh_loop())

    # Best-effort: create each project's curation indexes + kNN-warmup.
    from src.core.dependencies import bootstrap_opensearch_indexes

    AppResources.curation_knn_warmup_task = await bootstrap_opensearch_indexes()

    # Config store poll task: fans out per active project, not `default`.
    from src.services.config_store import startup_bootstrap_config_store_safe

    AppResources.config_store_poll_task = await startup_bootstrap_config_store_safe()


async def _start_event_bus_tail() -> None:
    # S-3: tail the shared cross-process event log so this uvicorn
    # worker's SSE clients see events from any other worker/job. Best-effort.
    try:
        from src.services.curation.event_hub import get_event_hub

        await get_event_hub().start_tail()
        AppResources.event_bus_started = True
        logger.info('event_bus_tail_started')
    except Exception as exc:
        logger.warning('event_bus_tail_start_skipped', error=str(exc))


def _reconcile_orphaned_state() -> None:
    """Fix up job and export state left behind by a killed process."""
    # Reconcile job state.json files left at status='running' by a killed
    # process; best-effort and isolated per module (see each module's
    # reconcile_orphaned_jobs() docstring).
    from src.services.curation.startup_reconcile import reconcile_all_jobs

    reconcile_all_jobs()

    # Gap 2 (model export): same idea, different shape — see
    # src.services.model_export's module docstring.
    try:
        from src.services.model_export import reconcile_orphaned_export_tasks

        n_reconciled = reconcile_orphaned_export_tasks()
        if n_reconciled:
            logger.warning('orphaned_export_tasks_reconciled', count=n_reconciled)
    except Exception as exc:
        logger.warning('orphaned_export_tasks_reconcile_skipped', error=str(exc))


async def _arbiter_reconcile_loop(reconcile_on_startup, reload_promoted_models_best_effort) -> None:
    """Re-assert the desired GPU-service state on a fixed interval.

    Also re-loads any promoted Triton model that's dropped out of
    READY (see ``reload_promoted_models_best_effort``) -- this is
    the only periodic tick in the process, so a Triton restart
    between startups still gets picked up without an API bounce.
    Idempotent by construction (stop/start only act on containers
    not already in the target state; reloading an already-READY
    model is a no-op), so running this in every uvicorn worker is
    safe re-enforcement rather than a race. One failing tick must
    never kill the loop — the next one retries.
    """
    while True:
        await asyncio.sleep(ARBITER_RECONCILE_INTERVAL_SECONDS)
        try:
            await reconcile_on_startup()
        except Exception as exc:
            logger.warning('gpu_arbiter_reconcile_loop_error', error=str(exc))
        await reload_promoted_models_best_effort(log_event='promoted_models_reload_tick')


async def _start_gpu_arbiter() -> None:
    # GPU arbiter. A training run claims the deployment's GPU-resident
    # containers by writing a lock + pause sentinel and stopping them
    # (src.services.training.gpu_arbiter). Nothing in the trainer releases
    # that claim if it dies mid-run, so without this the containers stay
    # down and the sentinel stays set *forever*. Reconcile once here (crash
    # recovery), then keep enforcing on a periodic loop: while a run is live
    # it re-stops any claimed container that comes back up, and once no run
    # is active it clears the lock + sentinel and restarts the containers.
    #
    # Deliberately parameter-free: every deployment-specific fact (which
    # containers, which GPU ids, the trainer jobs dir) is resolved by the
    # arbiter itself from GpuArbiterConfig / OP_TRAIN_JOBS_DIR, so no
    # container name, NAS path or GPU id is hardcoded here. An install that
    # configures none of it degrades to a no-op. Imported inside the
    # lifespan (not at module scope) so the whole block stays optional and
    # testable — same shape as the reconcile blocks above.
    try:
        from src.config import get_gpu_arbiter_config
        from src.services.training.gpu_arbiter import reconcile_on_startup

        # Resolved via GpuArbiterConfig.from_env() (OP_GPU_ALLOWED_IDS,
        # OP_GPU_ARBITER_CONTAINERS, OP_GPU_ARBITER_TRAINER_CONTAINER).
        # Logged so an operator can confirm the GPU
        # fence actually took effect.
        arbiter_cfg = get_gpu_arbiter_config()
        logger.info(
            'gpu_arbiter_config',
            allowed_gpu_ids=sorted(arbiter_cfg.allowed_gpu_ids) or 'unrestricted',
            containers=list(arbiter_cfg.containers),
            trainer_container=arbiter_cfg.trainer_container,
        )

        action = await reconcile_on_startup()
        logger.info('gpu_arbiter_reconciled', action=action.action, detail=action.detail)

        from src.services.training.triton_reload import reload_promoted_models_best_effort

        AppResources.arbiter_task = asyncio.create_task(
            _arbiter_reconcile_loop(reconcile_on_startup, reload_promoted_models_best_effort)
        )
        logger.info('gpu_arbiter_loop_started', interval_s=ARBITER_RECONCILE_INTERVAL_SECONDS)
    except Exception as exc:
        logger.warning('gpu_arbiter_reconcile_skipped', error=str(exc))


async def _reload_promoted_models() -> None:
    # Triton in explicit-control mode only loads its --load-model
    # list at startup. A model promoted through POST
    # {api_prefix}/train/promote/{job_id} stays on disk (with a
    # promote.json marker) but drops to UNAVAILABLE after any Triton
    # restart until something POSTs /load again -- without this, a
    # Triton bounce silently strands every previously-promoted model.
    # One-shot, best-effort: never blocks API startup. Also re-run on
    # every arbiter reconcile tick above, and available on demand via
    # POST {api_prefix}/train/reload_promoted (make reload-promoted).
    from src.services.training.triton_reload import reload_promoted_models_best_effort

    await reload_promoted_models_best_effort(log_event='promoted_models_reload_on_startup')


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
        logger.info('pe_text_encoder_warmed', backend=app.state.pe_encoder.text_backend)
    except Exception as exc:
        logger.warning('pe_text_encoder_warm_skipped', error=str(exc))


async def _stop_ops_metrics_and_arbiter() -> None:
    if AppResources.ops_metrics_task is not None:
        AppResources.ops_metrics_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await AppResources.ops_metrics_task
        AppResources.ops_metrics_task = None

    # Cancel the GPU-arbiter reconcile loop. Cleared afterwards so a second
    # app lifecycle in the same process (tests, embedded runs) never sees a
    # stale finished task.
    if AppResources.arbiter_task is not None:
        AppResources.arbiter_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await AppResources.arbiter_task
        AppResources.arbiter_task = None
        logger.info('gpu_arbiter_loop_stopped')


async def _stop_background_tasks() -> None:
    from src.services.curation.background_jobs import stop_background_jobs

    await stop_background_jobs(AppResources.curation_background_jobs)
    AppResources.curation_background_jobs = []

    from src.services.projects.bootstrap import shutdown_project_registry

    await shutdown_project_registry(AppResources.project_registry_poll_task)
    AppResources.project_registry_poll_task = None

    from src.services.config_store import shutdown_config_store_poll

    await shutdown_config_store_poll(AppResources.config_store_poll_task)
    AppResources.config_store_poll_task = None

    # Stop the event-bus tail task.
    if AppResources.event_bus_started:
        try:
            from src.services.curation.event_hub import get_event_hub

            await get_event_hub().stop_tail()
            logger.info('event_bus_tail_stopped')
        except Exception as exc:
            logger.warning('event_bus_tail_stop_error', error=str(exc))
        AppResources.event_bus_started = False


async def _close_pools() -> None:
    # Close AsyncTritonPool
    if AppResources.async_triton_pool is not None:
        try:
            await AppResources.async_triton_pool.close()
            logger.info('triton_pool_closed')
        except Exception as e:
            logger.warning('triton_pool_close_error', error=str(e))

    # The ingest CPU pool: let running pixel work finish
    shutdown_cpu_executor()

    # Shutdown shared executor
    if AppResources.shared_executor is not None:
        try:
            AppResources.shared_executor.shutdown(wait=True)
            logger.info('executor_shutdown')
        except Exception as e:
            logger.warning('executor_shutdown_error', error=str(e))


async def _close_clients() -> None:
    # Close Triton connections
    try:
        await TritonClientFactory.close_all()
    except Exception as e:
        logger.warning('triton_client_close_error', error=str(e))

    # Close OpenSearch connections
    try:
        await OpenSearchClientFactory.close()
    except Exception as e:
        logger.warning('opensearch_close_error', error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.

    Args:
        app: FastAPI application instance (required by API contract, not used in implementation)

    Startup:
    - Create shared ThreadPoolExecutor for CPU-bound tasks
    - Create AsyncTritonPool for high-throughput inference
    - Shared Triton gRPC client auto-created on first use
    - Reconcile orphaned job state left behind by a killed process
    - Reconcile the GPU arbiter and start its periodic reconcile loop

    Shutdown:
    - Cancel the GPU-arbiter reconcile loop
    - Close AsyncTritonPool
    - Shutdown shared ThreadPoolExecutor
    - Close all Triton gRPC connections
    - Close OpenSearch connections
    """
    settings = get_settings()

    # =========================================================================
    # STARTUP
    # =========================================================================
    logger.info('startup_begin', phase='initialization')

    _preflight()
    await _start_pools(settings)
    await _start_background_tasks()
    await _start_event_bus_tail()
    _reconcile_orphaned_state()
    await _start_gpu_arbiter()
    await _reload_promoted_models()
    _warm_pe_encoder(app)

    from src.services.curation.background_jobs import start_background_jobs

    AppResources.curation_background_jobs = start_background_jobs()

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

    await _stop_ops_metrics_and_arbiter()
    await _stop_background_tasks()
    await _close_pools()
    await _close_clients()

    logger.info('shutdown_complete')
