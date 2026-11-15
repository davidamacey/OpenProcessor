"""The finish phase of an ingest: what happens to a batch of photos after the inference results exist.

:func:`finish_photos` runs the steps for a batch in a fixed order, the per-photo work
:data:`~src.services.curation.ingest_helpers.MAX_INGEST_CONCURRENCY` wide and the database steps
once for the whole batch:

1. per photo: identity and capture (:meth:`CurationIngestService.ingest_image`), then the detector's
   items and the project's detect filter (:meth:`CurationIngestService.detect_for_ingest`);
2. once: the registry classes the detector labels need, created in one transaction
   (:meth:`CurationIngestService.resolve_classes`);
3. per photo: crops, embeddings, blur, ids (:func:`~src.services.curation.ingest_index.prepare_image`);
4. once each, for the batch as one commit group (:func:`~src.services.curation.ingest_index.commit_batch`):
   placement, the content claims with the pending rows (one transaction), the writes through the
   coalescing OpenSearch writer, the release of failed claims, the ``crop.created`` events.

A photo of a batch used to run all of this alone, with four or more PostgreSQL transactions of its
own, each ending in a COMMIT that waits for the WAL. ``ingest_one`` and a dataset import's
``index_items`` are a batch of one through the same functions.

What a photo ends with when a step fails, in the per-photo path this replaced (OLD) and now
(NEW). "request fails" is an exception out of the call: the route answers 500, or 503
``database_busy`` for a pool timeout, and what was stored stays stored::

    failure point            OLD                                NEW
    -----------------------  ---------------------------------  --------------------------------
    class create fails       that photo ``failed``              the photos that carry that class
                             (``detector_infer``); others go    name ``failed`` (``detector_infer``);
                             on                                 others go on. The batch create is
                                                                tried again name by name first
    detector fails           that photo ``failed``              same
    placement fails          request fails (photos ahead of     request fails before any photo of
                             the failure may be stored)         the batch is stored
    claim: content held      ``duplicate`` of the holder; no    same
    by another image         pending row, no write
    claim: key changed or    ``failed`` (``bulk_index``);       same
    refused                  nothing stored
    claim or pending rows    ``failed`` (``bulk_index``);       a batch transaction that fails as
    fail (database)          a claim already taken is           a whole is rolled back and the
                             released                           batch runs photo by photo with
                                                                the OLD behaviour (counted in
                                                                ``op_ingest_batch_fallback_total``)
    pool timeout in the      ``failed`` (``bulk_index``)        request fails (503): a retry per
    claim or pending rows    for each photo that waited         photo would wait the pool timeout
                             out the pool wait                  again for every photo
    pool timeout elsewhere   as the step above (class create    same
                             and detector: ``failed``;
                             placement: request fails, 503)
    OpenSearch write fails   ``failed`` (``bulk_index``);       same; the claims of the batch's
                             claim released; pending rows       failed photos are released in one
                             kept (the next rescore is          transaction
                             idempotent)
    crop cache fails         request fails; documents and       same
                             claim stay
    ``crop.created``         request fails; documents and       same; the events of the photos
    publish fails            claim stay                         stored before a failure are still
                                                                announced

A photo of one content as another (byte-identical) answers ``duplicate`` from the first one's
result; two photos with one ``imohash`` and other bytes finish in separate rounds.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.logging import get_logger
from src.services.curation.ingest_helpers import MAX_INGEST_CONCURRENCY
from src.services.curation.ingest_index import (
    ImageContext,
    IndexOutcome,
    PreparedImage,
    commit_batch,
    prepare_image,
)
from src.services.curation.ingest_models import ERROR_KIND_DETECTOR_INFER, IngestResult
from src.services.curation.ops_metrics import record_ingest_result
from src.utils.task_group import run_all


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.services.curation.ingest import CurationIngestService
    from src.services.curation.ingest_dedup import Resolved
    from src.services.curation.ingest_detect import SecondaryOutput
    from src.services.curation.ingest_gpu_decode import GpuDecodeFrame
    from src.services.curation.ingest_items import DetectResult
    from src.services.curation.item_doc import DetectedItem
    from src.utils.image_decode import DecodedImage


logger = get_logger(__name__)


@dataclass
class PhotoJob:
    """One photo of a batch and what the batch already knows about it."""

    image_bytes: bytes
    image_path: str
    source: str = 'unknown'
    prefilled_image: DecodedImage | None = None
    prefilled_items: list[DetectedItem] | None = None
    prefilled_secondary: SecondaryOutput | None = None
    prefilled_frame: GpuDecodeFrame | None = None
    decode_route: str | None = None
    source_identifier: str | None = None
    ingest_run_id: str | None = None
    resolved: Resolved | None = None
    """The photo's content looked up against the stored contents (the batch's one lookup); a
    photo without it is looked up on its own."""


@dataclass
class _Detected:
    ctx: ImageContext
    detected: DetectResult


def _failed_detector(job: PhotoJob, ctx: ImageContext, exc: Exception) -> IngestResult:
    logger.error('ingest_primary_detector_failed', path=job.image_path, error=str(exc))
    return IngestResult(
        status='failed',
        image_path=job.image_path,
        source_identifier=job.source_identifier,
        error=str(exc),
        error_kind=ERROR_KIND_DETECTOR_INFER,
        decode_route=ctx.decode_route,
    )


async def _detect(
    service: CurationIngestService, job: PhotoJob, sem: asyncio.Semaphore
) -> _Detected | IngestResult:
    async with sem:
        ctx = await service.ingest_image(
            job.image_bytes,
            job.image_path,
            job.source,
            prefilled_image=job.prefilled_image,
            prefilled_frame=job.prefilled_frame,
            decode_route=job.decode_route,
            source_identifier=job.source_identifier,
            ingest_run_id=job.ingest_run_id,
            resolved=job.resolved,
        )
        if isinstance(ctx, IngestResult):
            return ctx
        detected = await service.detect_for_ingest(
            ctx, job.image_path, job.prefilled_items, job.prefilled_secondary
        )
        if isinstance(detected, Exception):
            return _failed_detector(job, ctx, detected)
        return _Detected(ctx, detected)


async def _prepare(
    service: CurationIngestService, found: _Detected, sem: asyncio.Semaphore
) -> PreparedImage | IndexOutcome:
    async with sem:
        return await prepare_image(
            service,
            found.ctx,
            found.detected.items,
            secondary_detector_error=found.detected.secondary_detector_error,
        )


async def finish_photos(
    service: CurationIngestService, jobs: Sequence[PhotoJob]
) -> list[IngestResult]:
    """Ingest ``jobs`` and answer one result per job, in order (see the module docstring)."""
    sem = asyncio.Semaphore(MAX_INGEST_CONCURRENCY)
    staged = await run_all([_detect(service, job, sem) for job in jobs])
    results: list[IngestResult | None] = [
        s if isinstance(s, IngestResult) else None for s in staged
    ]
    found = [(i, s) for i, s in enumerate(staged) if isinstance(s, _Detected)]

    class_errors = await service.resolve_classes([s.detected.items for _, s in found])
    live: list[tuple[int, _Detected]] = []
    for (i, s), error in zip(found, class_errors, strict=True):
        if error is None:
            live.append((i, s))
        else:
            results[i] = _failed_detector(jobs[i], s.ctx, error)

    prepared = await run_all([_prepare(service, s, sem) for _, s in live])
    finished: dict[int, IndexOutcome] = {}
    ready: list[tuple[int, PreparedImage]] = []
    for (i, _), p in zip(live, prepared, strict=True):
        if isinstance(p, IndexOutcome):
            finished[i] = p
        else:
            ready.append((i, p))
    for (i, _), outcome in zip(
        ready, await commit_batch(service, [p for _, p in ready]), strict=True
    ):
        finished[i] = outcome
    for i, s in live:
        result = finished[i].result
        result.n_filtered = s.detected.n_filtered
        if result.status != 'duplicate':
            result.decode_route = s.ctx.decode_route
        results[i] = result
    return [record_ingest_result(r) for r in results if r is not None]


__all__ = ['PhotoJob', 'finish_photos']
