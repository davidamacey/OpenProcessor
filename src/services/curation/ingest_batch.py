"""Batch orchestration for the curation ingest pipeline.

Split out of :mod:`src.services.curation.ingest` so that module keeps one
concern (the per-image pipeline) and this one keeps another: how a batch
of images is fanned out efficiently.

``CurationIngestService.ingest_batch`` is a thin delegate to
:func:`run_ingest_batch` here. The batch path is *not* merely
"``ingest_one`` N times concurrently" — that would issue N single-image
Triton round-trips and is measurably slower for identical output. It is:

1. **Dedup once, for everyone** — a single query resolves every
   image's ``imohash`` against ``image_contents`` (only files on a match are compared
   by SHA-256). This ONLY resolves against
   the index, i.e. images already committed before this batch started —
   see :func:`run_ingest_batch` for the separate in-batch dedup pass (D3)
   that also collapses byte-identical repeats *within* one request onto a
   single representative. Copies in *concurrent* requests are kept apart
   at write time: the images doc is created under a content-derived id, and
   the copy that loses the create is a ``duplicate``.
2. **Decode once** — non-duplicates are decoded up front, and the PIL
   image is handed to ``ingest_one`` so it does not decode again. With the
   GPU decode on, photos at or above its size gate are decoded on the GPU
   instead (:mod:`~src.services.curation.ingest_gpu_decode`), which also
   returns their detections, vectors, blur sums and crop JPEGs; an image it
   declines falls back to the CPU decode here.
3. **Batched whole-image inference** — the decoded images go through
   :class:`~src.services.curation.ingest_detect.WholeImageDetector`'s
   batched methods, one Triton call per ``DetectionProfile.batch_limit``
   chunk per detector, with both detectors launched concurrently.
4. **Finish, once per batch where it touches PostgreSQL** — per-image work (capture, crops,
   embeddings, quality metrics) runs under a semaphore, consuming the prefilled results; the
   registry classes, the placement, the content claims with the pending rows and the
   ``crop.created`` events are one transaction each for the whole batch
   (:mod:`~src.services.curation.ingest_finish`). Photos of one content that cannot be decided in
   one transaction (two files with one ``imohash`` and other bytes) finish in later waves.

Ingesting an already-labeled dataset is ``POST /datasets/imports`` — this module carries no
label-import machinery.

If the batched inference raises, the prefilled detections are dropped
entirely and every image falls back to its own single-image call — a
half-batched hybrid would be harder to reason about than either path.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Literal

from src.core.logging import get_logger
from src.domain.decode_route import CPU_GPU_OFF, CPU_NOT_IMAGE, CPU_SMALL, GPU, prep_status_route
from src.services.curation.ingest_dedup import held_back_result
from src.services.curation.ingest_finish import PhotoJob
from src.services.curation.ingest_models import BatchIngestResult, IngestResult, IngestSummary
from src.services.curation.ops_metrics import record_ingest_result
from src.utils.cpu_pool import encoded_nbytes, ingest_gate, mib, run_cpu
from src.utils.stage_timing import observe_stage, stage_timer
from src.utils.task_group import run_all, run_settled


if TYPE_CHECKING:
    from PIL import Image

    from src.services.curation.ingest import CurationIngestService
    from src.services.curation.ingest_dedup import Resolved
    from src.services.curation.ingest_detect import SecondaryOutput
    from src.services.curation.ingest_gpu_decode import GpuDecodeFrame, GpuRoute
    from src.services.curation.item_doc import DetectedItem
    from src.utils.image_decode import DecodedImage


logger = get_logger(__name__)


async def _prefill_gpu_decode(
    service: CurationIngestService, images: list[bytes], indices: list[int], route: GpuRoute
) -> tuple[dict[int, GpuDecodeFrame], dict[int, list[DetectedItem]], dict[int, str]]:
    """Run the GPU decode on the images ``route`` chose; the third map is the decode route of
    every image of ``indices`` (counted once each), whatever path it takes."""
    from src.services.curation.ingest_gpu_decode import count_route, run_gpu_decode

    frames: dict[int, GpuDecodeFrame] = {}
    items: dict[int, list[DetectedItem]] = {}
    routes: dict[int, str] = {}

    def decided(idx: int, name: str) -> None:
        routes[idx] = name
        count_route(name)

    if not route.ready:
        for idx in indices:
            decided(idx, CPU_GPU_OFF)
        return frames, items, routes
    chosen = set(route.chosen)
    unreadable = set(route.unreadable)
    for idx in indices:
        if idx not in chosen:
            decided(idx, CPU_NOT_IMAGE if idx in unreadable else CPU_SMALL)
    taken = [idx for idx in indices if idx in chosen]
    embed_crops = service.policy.embedding.mode not in ('lazy', 'none')

    async def timed(idx: int) -> tuple[GpuDecodeFrame, list[DetectedItem]] | str:
        # Wall time of the call, which includes its wait in the Triton queue.
        with stage_timer('gpu_decode'):
            return await run_gpu_decode(service, images[idx], embed_crops=embed_crops)

    outcomes = await run_all([timed(idx) for idx in taken])
    for idx, outcome in zip(taken, outcomes, strict=True):
        if isinstance(outcome, str):
            decided(idx, prep_status_route(outcome))
            continue
        decided(idx, GPU)
        frames[idx], items[idx] = outcome
    return frames, items, routes


async def _prefill_detections(
    service: CurationIngestService,
    images: list[bytes],
    image_paths: list[str],
    indices: list[int],
    sizes: dict[int, int | None],
    route: GpuRoute,
) -> tuple[
    dict[int, DecodedImage],
    dict[int, list[DetectedItem]],
    dict[int, SecondaryOutput],
    dict[int, GpuDecodeFrame],
    dict[int, str],
]:
    """Batch-decode ``indices`` and run the whole-image detectors batched.

    Returns the index-keyed maps (and the decode route of each image) consumed by
    :meth:`~src.services.curation.ingest.CurationIngestService.ingest_one`'s
    ``prefilled_*`` arguments. On a batched-inference failure both
    detection maps come back empty so every image falls back to the
    per-image path rather than running a half-batched hybrid.
    """
    from src.services.curation.ingest_helpers import _decode_image

    prefilled_imgs: dict[int, DecodedImage] = {}
    prefilled_secondary: dict[int, SecondaryOutput] = {}
    frames, prefilled_items, routes = await _prefill_gpu_decode(service, images, indices, route)
    indices = [idx for idx in indices if idx not in frames]
    if not indices:
        return prefilled_imgs, prefilled_items, prefilled_secondary, frames, routes

    valid_imgs: list[Image.Image] = []
    valid_indices: list[int] = []
    decoded = await run_settled(
        [run_cpu(_decode_image, images[idx], nbytes=sizes[idx]) for idx in indices]
    )
    for idx, outcome in zip(indices, decoded, strict=True):
        if isinstance(outcome, Exception):
            # ingest_one re-decodes this byte slice and produces a structured IngestResult
            # with the right error_kind, so just skip prefilling it here.
            logger.warning('ingest_batch_decode_failed', path=image_paths[idx], error=str(outcome))
            continue
        prefilled_imgs[idx] = outcome
        valid_imgs.append(outcome.pil)
        valid_indices.append(idx)

    if not valid_imgs:
        return prefilled_imgs, prefilled_items, prefilled_secondary, frames, routes

    detector = service.detector
    secondary_per: list[SecondaryOutput | None]
    try:
        if detector.secondary_profile is not None:
            async with asyncio.TaskGroup() as group:
                primary_task = group.create_task(detector.run_primary_batch(valid_imgs))
                secondary_task = group.create_task(detector.run_secondary_raw_batch(valid_imgs))
            items_per = primary_task.result()
            secondary_per = [*secondary_task.result()]
        else:
            items_per = await detector.run_primary_batch(valid_imgs)
            secondary_per = [None] * len(valid_imgs)
    except Exception as exc:
        logger.warning(
            'ingest_batch_detector_failed_falling_back',
            error=str(exc),
            n_images=len(valid_imgs),
        )
        gpu_decode_items = {idx: prefilled_items[idx] for idx in frames}
        return prefilled_imgs, gpu_decode_items, {}, frames, routes

    for idx, items, secondary in zip(valid_indices, items_per, secondary_per, strict=False):
        prefilled_items[idx] = items
        if secondary is not None:
            prefilled_secondary[idx] = secondary
    return prefilled_imgs, prefilled_items, prefilled_secondary, frames, routes


def _group_batch(
    hashes: list[str], images: list[bytes], existing_ids: list[str | None]
) -> tuple[dict[int, int], list[int]]:
    """``(batch_dup_of, non_dup)`` for one batch.

    D3: ``existing_ids`` only resolves content already committed *before* this batch started --
    two byte-identical files uploaded together share a content neither has stored yet, and
    both would otherwise land as separate 'success' items with 0 duplicates reported.
    Same-batch repeats collapse onto one representative index per content (the first
    occurrence) before deciding what needs detection/inference at all (``batch_dup_of``). The
    imohash is only the candidate key: a repeat is a duplicate when its bytes are equal, and a
    same-imohash file with other bytes (an imohash collision) is its own image, which
    :func:`_rounds` finishes after the earlier one is stored, so the content key sees the
    collision instead of two writers racing it.
    """
    by_hash: dict[str, list[int]] = {}
    batch_dup_of: dict[int, int] = {}
    non_dup: list[int] = []
    for i, h in enumerate(hashes):
        if existing_ids[i] is not None:
            continue
        peers = by_hash.setdefault(h, [])
        same = next((p for p in peers if images[p] == images[i]), None)
        if same is not None:
            batch_dup_of[i] = same
            continue
        peers.append(i)
        non_dup.append(i)
    return batch_dup_of, non_dup


def _rounds(indices: list[int], hashes: list[str]) -> list[list[int]]:
    """``indices`` split so that no round holds two photos of one ``imohash``: a batch decides one
    content key per ``imohash`` in one transaction, and the later photo of a pair is decided once
    the earlier one is stored."""
    seen: dict[str, int] = {}
    rounds: list[list[int]] = []
    for i in indices:
        n = seen.get(hashes[i], 0)
        seen[hashes[i]] = n + 1
        while len(rounds) <= n:
            rounds.append([])
        rounds[n].append(i)
    return rounds


def _result_of(
    i: int,
    image_paths: list[str],
    hashes: list[str],
    source_identifiers: list[str | None] | None,
    existing_ids: list[str | None],
    batch_dup_of: dict[int, int],
    stored: dict[int, IngestResult],
) -> IngestResult:
    """Photo ``i``'s result: its own when it was ingested; else the duplicate it is of."""
    if i in stored:
        return stored[i]
    holder = existing_ids[i] if existing_ids[i] else stored[batch_dup_of[i]].image_id
    return record_ingest_result(
        IngestResult(
            status='duplicate',
            image_id=holder,
            image_path=image_paths[i],
            imohash=hashes[i],
            source_identifier=source_identifiers[i] if source_identifiers else None,
        )
    )


async def run_ingest_batch(
    service: CurationIngestService,
    images: list[bytes],
    image_paths: list[str],
    source: str = 'batch',
    source_identifiers: list[str | None] | None = None,
    ingest_run_id: str | None = None,
) -> BatchIngestResult:
    """Implementation behind :meth:`CurationIngestService.ingest_batch`.

    See that method's docstring for the argument contract.
    """
    from src.services.curation.ingest_gpu_decode import gpu_route
    from src.services.near_duplicates.content_key import content_imohash

    if len(images) != len(image_paths):
        raise ValueError('images and image_paths must be same length')
    if source_identifiers is not None and len(source_identifiers) != len(images):
        raise ValueError('source_identifiers must match images length')

    summary = IngestSummary()
    hashes = [content_imohash(b) for b in images]
    resolved = await service.resolve_contents(list(zip(hashes, images, strict=True)))
    # A stored image that misses items is no duplicate: one nobody is writing is ingested again
    # (it completes the image), one a writer may still be writing is answered at once.
    existing_ids = [
        r.existing['image_id'] if r.existing is not None and r.completeness != 'repair' else None
        for r in resolved
    ]
    stored: dict[int, IngestResult] = {
        i: record_ingest_result(
            held_back_result(
                r, image_paths[i], hashes[i], source_identifiers[i] if source_identifiers else None
            )
        )
        for i, r in enumerate(resolved)
        if r.completeness == 'in_flight'
    }

    batch_dup_of, non_dup = _group_batch(hashes, images, existing_ids)

    # Hold the decoded megabytes this request will bring against the process's in-flight cap,
    # so more clients queue here instead of all decoding at once (#241).
    # A photo the GPU decode takes is never decoded here, so it holds only its encoded bytes.
    sizes = {i: encoded_nbytes(images[i]) for i in non_dup}
    route = await gpu_route(service, images, non_dup)
    held = sum(mib(len(images[i]) if i in route.chosen else sizes[i]) for i in non_dup)
    wait_start = time.perf_counter()
    async with ingest_gate().hold(held):
        observe_stage('admission_wait', time.perf_counter() - wait_start)
        # Only pay the (expensive) batched decode + inference for images that
        # dedup did not already resolve to an existing document or an
        # earlier-in-batch representative.
        (
            prefilled_imgs,
            prefilled_items,
            prefilled_secondary,
            frames,
            routes,
        ) = await _prefill_detections(service, images, image_paths, non_dup, sizes, route)

        def job(i: int, *, prefilled: bool = True, looked_up: Resolved | None = None) -> PhotoJob:
            # Clients match results to request entries by source_identifier, so
            # every duplicate carries its own, even though image_id is shared.
            return PhotoJob(
                images[i],
                image_paths[i],
                source,
                prefilled_image=prefilled_imgs.get(i) if prefilled else None,
                prefilled_items=prefilled_items.get(i) if prefilled else None,
                prefilled_secondary=prefilled_secondary.get(i) if prefilled else None,
                prefilled_frame=frames.get(i) if prefilled else None,
                decode_route=routes.get(i) if prefilled else None,
                source_identifier=source_identifiers[i] if source_identifiers else None,
                ingest_run_id=ingest_run_id,
                resolved=looked_up,
            )

        for n, wave in enumerate(_rounds(non_dup, hashes)):
            # The first wave's files were resolved above, before anything of this batch was
            # stored; a later wave resolves again, now that the earlier file it waited for is.
            looked_up = (
                [resolved[i] for i in wave]
                if n == 0
                else await service.resolve_contents([(hashes[i], images[i]) for i in wave])
            )
            jobs = [job(i, looked_up=r) for i, r in zip(wave, looked_up, strict=True)]
            stored.update(zip(wave, await service.ingest_photos(jobs), strict=True))
        # A copy of a representative that failed (e.g. undecodable bytes) gets its own real
        # attempt instead of a guessed duplicate-of relationship; it was not prefilled.
        retry = [i for i, rep in batch_dup_of.items() if stored[rep].status == 'failed']
        for again in _rounds(retry, hashes):
            jobs = [job(i, prefilled=False) for i in again]
            stored.update(zip(again, await service.ingest_photos(jobs), strict=True))

    results = [
        _result_of(i, image_paths, hashes, source_identifiers, existing_ids, batch_dup_of, stored)
        for i in range(len(images))
    ]

    await service.refresh_indexes()

    for res in results:
        if res.status == 'duplicate':
            summary.duplicates += 1
        elif res.status == 'success':
            summary.successful += 1
            summary.crops_indexed += res.n_crops
            summary.n_embedded += res.n_embedded
            summary.n_not_embedded += res.n_not_embedded
            summary.n_embed_failed += res.n_embed_failed
            summary.n_filtered += res.n_filtered
        else:
            summary.failed += 1
        # an otherwise-'successful' ingest that silently skipped a
        # configured secondary detector call must still be visible in the
        # batch summary, not just a per-item field a caller has to notice.
        if res.secondary_detector_error:
            summary.secondary_detector_failures += 1
        if res.decode_route is not None:
            summary.decode_routes[res.decode_route] = (
                summary.decode_routes.get(res.decode_route, 0) + 1
            )

    if summary.failed == 0:
        status: Literal['success', 'partial', 'error'] = 'success'
    elif summary.successful == 0:
        status = 'error'
    else:
        status = 'partial'

    return BatchIngestResult(status=status, summary=summary, results=results)


__all__ = ['run_ingest_batch']
