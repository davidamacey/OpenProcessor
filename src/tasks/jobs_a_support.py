"""What the lane-4A handlers need around a claimed job: the bound project and the clients.

The wrapper opens a database session and hands the handler a ``JobContext``; it binds no project
and builds no OpenSearch or Triton client (the runtime never imports them, review R-2). The
handlers do it here, inside the task's own event loop, and the context closes what they built.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from src.config.project_context import bind_project


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from src.tasks.runtime.context import JobContext


class ProjectGoneError(RuntimeError):
    """The job's project is not in the registry (deleted or never existed): the job fails."""


@asynccontextmanager
async def bound_project(ctx: JobContext) -> AsyncIterator[None]:
    """Bind the job's project for the handler's code (index names, class registry, config)."""
    from src.services.projects.registry import get_project_registry

    slug = ctx.project_slug
    record = None if slug is None else await get_project_registry().lookup(slug)
    if record is None:
        raise ProjectGoneError(f'the project {slug!r} of job {ctx.job_id} is not in the registry')
    with bind_project(record):
        yield


async def opensearch_client() -> Any:
    """A project-guarded ``AsyncOpenSearch`` for this task (the context closes it)."""
    from src.config.settings import get_settings
    from src.services.projects.guard import make_script_opensearch

    settings = get_settings()
    return make_script_opensearch([settings.opensearch_url], timeout=settings.opensearch_timeout)


class WorkerIngest:
    """Builds the ingest service ``detect`` and ``embed`` need, for a worker with no FastAPI app.

    The Triton pool belongs to the task's event loop, so it is created here and closed with the
    context; ``AppResources`` points at it for the shared code that reads it from there.
    """

    def __init__(self) -> None:
        self._pool: Any = None
        self._executor: ThreadPoolExecutor | None = None

    async def service(self, opensearch: Any) -> Any:
        from src.app.resources import AppResources
        from src.clients.pe_encoder import PEEncoder
        from src.clients.triton_pool_core import AsyncTritonPool
        from src.config.settings import get_settings
        from src.services.classes.registry import get_class_registry
        from src.services.curation.ingest import CurationIngestService
        from src.services.curation.ingest_detector import refreshed_ingest_profiles
        from src.services.curation.ingest_policy_store import get_ingest_policy

        if self._pool is None:
            self._pool = AsyncTritonPool(
                url=get_settings().triton_url, pool_size=4, max_concurrent=64, verbose=False
            )
            await self._pool.initialize()
            self._executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix='job-ingest-')
            AppResources.async_triton_pool = self._pool
            AppResources.shared_executor = self._executor
        profiles = await refreshed_ingest_profiles()
        if not profiles.primary.detector_model:
            raise RuntimeError(
                'no detector is configured for ingest; activate an ingest profile for this project'
            )
        return CurationIngestService(
            opensearch=opensearch,
            triton_pool=self._pool,
            registry=await get_class_registry().snapshot(),
            profile=profiles.primary,
            secondary_profile=profiles.secondary,
            pe_encoder=PEEncoder(triton_pool=self._pool),
            policy=await get_ingest_policy(),
        )

    async def aclose(self) -> None:
        from src.app.resources import AppResources

        if self._pool is not None:
            await self._pool.close()
            AppResources.async_triton_pool = None
            self._pool = None
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            AppResources.shared_executor = None
            self._executor = None


async def worker_ingest() -> WorkerIngest:
    return WorkerIngest()
