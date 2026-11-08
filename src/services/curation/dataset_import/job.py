"""The ``dataset_import`` and ``dataset_import_undo`` jobs as a worker runs them.

The handler is already in the project's scope. Everything a run needs comes from the rows the
start wrote (:mod:`.rows`): the request, the resolved mapping with the pinned region profile, the
scan facts. The dataset is scanned again at the start of every attempt and must still have the
``source_sha`` it had when the import was accepted (a changed dataset fails the job).
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config.curation_accessors import get_curation_config
from src.services.classes.registry import get_class_registry
from src.services.curation.dataset_import import rows
from src.services.curation.dataset_import.context import ImportContext
from src.services.curation.dataset_import.holdout_record import (
    mark_freeze_undone,
    persist_import_freeze,
)
from src.services.curation.dataset_import.options import DatasetImportRequest
from src.services.curation.dataset_import.paths import dataset_path_guard
from src.services.curation.dataset_import.prepare import (
    mapping_to_dict,
    materialize_created_classes,
)
from src.services.curation.dataset_import.runner import (
    ImportOutcome,
    load_pinned,
    rescan,
    run_import_job,
)
from src.services.curation.dataset_import.undo import UndoContext, undo_import
from src.services.curation.next_steps import cluster_regions
from src.tasks.runtime.context import JobResult


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.clients.pe_encoder import PEEncoder
    from src.clients.triton_pool_core import AsyncTritonPool
    from src.tasks.runtime.context import JobContext


class ImportRunError(RuntimeError):
    """The import ended without importing every image (the job fails; a resume continues)."""


class ImportRowMissingError(RuntimeError):
    """The job names an import the project does not have (fail closed, nothing is written)."""


@dataclass
class IngestRuntime:
    """What the ingest service needs that a worker builds for itself: Triton and the encoder."""

    pool: AsyncTritonPool
    pe_encoder: PEEncoder

    async def close(self) -> None:
        await self.pool.close()


async def open_ingest_runtime() -> IngestRuntime:
    from src.clients.pe_encoder import PEEncoder
    from src.clients.triton_pool_core import AsyncTritonPool
    from src.config.settings import get_settings

    pool = AsyncTritonPool(url=get_settings().triton_url, pool_size=2, max_concurrent=16)
    await pool.initialize()
    return IngestRuntime(pool, PEEncoder(triton_pool=pool))


async def _import_service(opensearch: Any, runtime: IngestRuntime, *, need_detector: bool) -> Any:
    from src.services.curation.ingest import CurationIngestService
    from src.services.curation.ingest_detector import refreshed_ingest_profiles
    from src.services.curation.ingest_policy_store import get_ingest_policy

    profile = (await refreshed_ingest_profiles()).primary
    if need_detector and not profile.detector_model:
        raise ImportRunError('this import runs the detector; none is configured')
    return CurationIngestService(
        opensearch=opensearch,
        triton_pool=runtime.pool if need_detector else None,
        registry=await get_class_registry().snapshot(),
        profile=profile,
        pe_encoder=runtime.pe_encoder,
        policy=await get_ingest_policy(),
    )


def _enqueue_for(ctx: JobContext) -> Callable[[list[str]], Awaitable[None]]:
    async def enqueue(image_ids: list[str]) -> None:
        from src.services.near_duplicates.pending import enqueue_pending

        assert ctx.project_id is not None
        async with ctx.session() as session:
            await enqueue_pending(session, ctx.project_id, 'image', image_ids, 'added')

    return enqueue


async def _load(ctx: JobContext) -> rows.ImportRecord:
    import_id = uuid.UUID(str(ctx.params['import_id']))
    async with ctx.session() as session:
        record = await rows.get(session, import_id)
    if record is None:
        raise ImportRowMissingError(f'job {ctx.job_id} has no import {import_id}')
    return record


async def run_import(ctx: JobContext, opensearch: Any, runtime: IngestRuntime) -> Any:
    record = await _load(ctx)
    cfg = get_curation_config()
    request = DatasetImportRequest(**record.request)
    entries = await asyncio.to_thread(
        rescan,
        request,
        record.scan.get('source_sha'),
        path_guard=dataset_path_guard(cfg.export_root),
    )
    resolved, profile, parents = load_pinned(record.mapping)
    # a start that died after creating some classes but before recording them
    await materialize_created_classes(resolved, get_class_registry())
    async with ctx.session() as session:
        await rows.set_mapping(session, record.id, {**record.mapping, **mapping_to_dict(resolved)})
    has_region = any(t.kind == 'region' for t in resolved.targets.values())
    service = await _import_service(
        opensearch,
        runtime,
        need_detector=request.options.processing == 'propose'
        or (has_region and parents == 'detect'),
    )
    import_ctx = ImportContext(
        import_id=str(record.id),
        options=request.options,
        resolved=resolved,
        profile=profile,
        parents=parents,
        source_sha=str(record.scan['source_sha']),
        source_format=str(record.scan.get('source_format', '')),
        source_root=Path(str(record.scan.get('source_root', ''))),
        opensearch=service.opensearch,
        service=service,
        images_index=cfg.images_index,
        items_index=cfg.items_index,
        crop_cache_dir=cfg.crop_cache_dir,
        upload_root=Path(cfg.upload_root),
        export_root=Path(cfg.export_root) if cfg.export_root else None,
        freeze_test=bool(record.scan.get('freeze_test')),
        ledger=rows.Ledger(ctx.session, record.id),
        enqueue_pending=_enqueue_for(ctx),
    )
    outcome = await run_import_job(ctx, import_ctx, entries)
    return await finish(ctx.session, import_ctx, record, outcome, total=len(entries))


async def finish(
    sessions: rows.Sessions,
    import_ctx: ImportContext,
    record: rows.ImportRecord,
    outcome: ImportOutcome,
    *,
    total: int,
) -> Any:
    """Turn how the loop ended into the job's end: cancel, failure, or the result."""
    if outcome.cancelled:
        return {'import_id': str(record.id), 'cancelled': True}
    if outcome.error or outcome.failed_chunks:
        raise ImportRunError(outcome.error or f'chunks failed: {outcome.failed_chunks}')
    if import_ctx.freeze_test and not record.scan.get('freeze_record'):
        freeze = await persist_import_freeze(import_ctx, record.scan.get('test_split') or {})
        if freeze is not None:
            async with sessions() as session:
                await rows.merge_scan(session, record.id, {'freeze_record': str(freeze)})
    report = outcome.report
    next_steps = []
    if report.boxes_written or report.standalone_regions:
        next_steps.append(cluster_regions())
    return JobResult(
        {'import_id': str(record.id), 'images_total': total, 'next_steps': next_steps},
        with_errors=bool(report.images_failed or report.images_skipped),
    )


async def run_undo(ctx: JobContext, opensearch: Any) -> Any:
    record = await _load(ctx)
    cfg = get_curation_config()
    from src.config.region_fields import get_region_fields

    undo_ctx = UndoContext(
        import_id=str(record.id),
        opensearch=getattr(opensearch, 'client', opensearch),
        images_index=cfg.images_index,
        items_index=cfg.items_index,
        crop_cache_dir=cfg.crop_cache_dir,
        region_fields=get_region_fields(),
        registry=get_class_registry(),
    )
    ledger = rows.Ledger(ctx.session, record.id)
    report = await undo_import(
        undo_ctx,
        ledger,
        dry_run=False,
        remove_images=bool(ctx.params.get('remove_images', True)),
        deprecate_created_classes=bool(ctx.params.get('deprecate_created_classes', True)),
        created_classes=record.mapping.get('created_classes') or {},
    )
    await mark_freeze_undone(record.scan.get('freeze_record'))
    await ledger.mark_undone()
    return {'import_id': str(record.id), 'undo': report.to_dict()}


__all__ = [
    'ImportRowMissingError',
    'ImportRunError',
    'IngestRuntime',
    'finish',
    'open_ingest_runtime',
    'run_import',
    'run_undo',
]
