"""Batch orchestration for the curation ingest pipeline.

Split out of :mod:`src.services.curation.ingest` so that module keeps one
concern (the per-image pipeline) and this one keeps another: how a batch
of images is fanned out efficiently.

``CurationIngestService.ingest_batch`` is a thin delegate to
:func:`run_ingest_batch` here. The batch path is *not* merely
"``ingest_one`` N times concurrently" — that would issue N single-image
Triton round-trips and is measurably slower for identical output. It is:

1. **Dedup once, for everyone** — a single ``msearch`` resolves every
   image's imohash instead of N term queries.
2. **Decode once** — non-duplicates are decoded up front, and the PIL
   image is handed to ``ingest_one`` so it does not decode again.
3. **Batched whole-image inference** — the decoded images go through
   :class:`~src.services.curation.ingest_detect.WholeImageDetector`'s
   batched methods, one Triton call per ``DetectionProfile.batch_limit``
   chunk per detector, with both detectors launched concurrently.
4. **Bounded-concurrency finish** — per-image work (crops, embeddings,
   quality metrics, bulk index) runs under a semaphore, consuming the
   prefilled results via ``ingest_one``'s ``prefilled_*`` arguments.
5. **Optional ground-truth import** — companion YOLO ``.txt`` label
   paths are imported through
   :mod:`src.services.curation.label_import` in the same call, so
   ingesting an already-labeled dataset (and reporting where the
   detector disagreed with it) is one request, not two. The images and
   items indexes are refreshed first so the importer's searches can see
   the documents this batch just wrote.

If the batched inference raises, the prefilled detections are dropped
entirely and every image falls back to its own single-image call — a
half-batched hybrid would be harder to reason about than either path.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.core.logging import get_logger
from src.services.curation.ingest_models import BatchIngestResult, IngestResult, IngestSummary


if TYPE_CHECKING:
    from PIL import Image

    from src.services.curation.ingest import CurationIngestService
    from src.services.curation.ingest_detect import SecondaryOutput
    from src.services.curation.item_doc import DetectedItem


logger = get_logger(__name__)


async def _prefill_detections(
    service: CurationIngestService,
    images: list[bytes],
    image_paths: list[str],
    indices: list[int],
) -> tuple[
    dict[int, Image.Image],
    dict[int, list[DetectedItem]],
    dict[int, SecondaryOutput],
]:
    """Batch-decode ``indices`` and run the whole-image detectors batched.

    Returns three index-keyed maps consumed by
    :meth:`~src.services.curation.ingest.CurationIngestService.ingest_one`'s
    ``prefilled_*`` arguments. On a batched-inference failure both
    detection maps come back empty so every image falls back to the
    per-image path rather than running a half-batched hybrid.
    """
    from src.services.curation.ingest import _decode_image

    prefilled_imgs: dict[int, Image.Image] = {}
    prefilled_items: dict[int, list[DetectedItem]] = {}
    prefilled_secondary: dict[int, SecondaryOutput] = {}
    if not indices:
        return prefilled_imgs, prefilled_items, prefilled_secondary

    valid_imgs: list[Image.Image] = []
    valid_indices: list[int] = []
    for idx in indices:
        try:
            img, _w, _h = _decode_image(images[idx])
        except Exception as exc:
            # Broad catch: ingest_one re-decodes this byte slice and
            # produces a structured IngestResult with the right
            # error_kind, so just skip prefilling it here.
            logger.warning('ingest_batch_decode_failed', path=image_paths[idx], error=str(exc))
            continue
        prefilled_imgs[idx] = img
        valid_imgs.append(img)
        valid_indices.append(idx)

    if not valid_imgs:
        return prefilled_imgs, prefilled_items, prefilled_secondary

    detector = service.detector
    secondary_per: list[SecondaryOutput | None]
    try:
        if detector.secondary_profile is not None:
            items_per, secondary_batch = await asyncio.gather(
                detector.run_primary_batch(valid_imgs),
                detector.run_secondary_raw_batch(valid_imgs),
            )
            secondary_per = [*secondary_batch]
        else:
            items_per = await detector.run_primary_batch(valid_imgs)
            secondary_per = [None] * len(valid_imgs)
    except Exception as exc:
        logger.warning(
            'ingest_batch_detector_failed_falling_back',
            error=str(exc),
            n_images=len(valid_imgs),
        )
        return prefilled_imgs, {}, {}

    for idx, items, secondary in zip(valid_indices, items_per, secondary_per, strict=False):
        prefilled_items[idx] = items
        if secondary is not None:
            prefilled_secondary[idx] = secondary
    return prefilled_imgs, prefilled_items, prefilled_secondary


async def _refresh_items_for_label_import(service: CurationIngestService) -> None:
    """Make this batch's just-written item docs searchable.

    F-26: the images-index refresh is no longer needed — ``_import_batch_labels``
    now passes each result's ``image_id`` straight through to
    ``import_labels_batch`` (this batch already knows it; no need to
    search the images index to rediscover it). The detector items still
    need one: the label importer's IoU match does a *search* on
    ``image_id`` against the items index, which — bulk-written with
    ``refresh=False`` — is invisible until refreshed. Without this, every
    label would silently skip its IoU match ("no detector item found")
    and no disagreement would ever be detected.
    """
    cfg = service.config
    try:
        await service.opensearch.indices.refresh(index=cfg.items_index)
    except Exception as exc:
        # Broad on purpose: the import below still runs; if the docs are
        # not yet visible it reports labels_imported=0, which the caller
        # sees — this does not hide the failure, it just does not abort
        # the already-completed ingest.
        logger.warning('ingest_batch_label_refresh_failed', error=str(exc))


async def _import_batch_labels(
    service: CurationIngestService,
    image_paths: list[str],
    label_paths: list[str | None],
    results: list[IngestResult],
    *,
    label_source: str,
    detect_mismatches: bool,
    disagreement_sink: list[dict[str, Any]],
) -> dict[str, int]:
    """Import companion YOLO ``.txt`` labels for the images that ingested OK."""
    from src.services.curation.label_import import DEFAULT_LABEL_SOURCE, import_labels_batch

    pairs: list[tuple[Path, Path]] = []
    image_docs: dict[str, dict[str, Any]] = {}
    for image_path, label_path, res in zip(image_paths, label_paths, results, strict=False):
        if not (label_path and res.status == 'success'):
            continue
        pairs.append((Path(image_path), Path(label_path)))
        # F-26: this ingest batch already knows the image_id it just wrote
        # — hand it straight to the importer instead of making it search
        # the images index to rediscover what this call already knows.
        if res.image_id:
            image_docs[image_path] = {'image_id': res.image_id, '_id': res.image_id}
    if not pairs:
        return {}
    await _refresh_items_for_label_import(service)
    try:
        return await import_labels_batch(
            pairs,
            service.registry,
            service.opensearch,
            label_source=label_source or DEFAULT_LABEL_SOURCE,
            detect_mismatches=detect_mismatches,
            disagreement_sink=disagreement_sink,
            image_docs=image_docs,
        )
    except Exception as exc:
        logger.warning('ingest_batch_label_import_failed', error=str(exc), n_pairs=len(pairs))
        return {}


async def run_ingest_batch(
    service: CurationIngestService,
    images: list[bytes],
    image_paths: list[str],
    label_paths: list[str | None] | None = None,
    source: str = 'batch',
    label_source: str = '',
    detect_mismatches: bool = False,
    whole_frame_from_bytes: bool = False,
    source_identifiers: list[str | None] | None = None,
    ingest_run_id: str | None = None,
) -> BatchIngestResult:
    """Implementation behind :meth:`CurationIngestService.ingest_batch`.

    See that method's docstring for the argument contract.
    """
    from src.services.curation.ingest import MAX_INGEST_CONCURRENCY, _imohash_bytes

    if len(images) != len(image_paths):
        raise ValueError('images and image_paths must be same length')
    if label_paths is not None and len(label_paths) != len(images):
        raise ValueError('label_paths must match images length')
    if source_identifiers is not None and len(source_identifiers) != len(images):
        raise ValueError('source_identifiers must match images length')

    summary = IngestSummary()
    hashes = [_imohash_bytes(b) for b in images]
    hash_to_existing = await service._check_duplicates_msearch(hashes)

    # Only pay the (expensive) batched decode + inference for images that
    # dedup did not already resolve to an existing document.
    non_dup = [i for i, h in enumerate(hashes) if hash_to_existing.get(h) is None]
    prefilled_imgs, prefilled_items, prefilled_secondary = await _prefill_detections(
        service, images, image_paths, non_dup
    )

    sem = asyncio.Semaphore(MAX_INGEST_CONCURRENCY)

    async def _one(i: int, image_bytes: bytes, image_path: str, image_hash: str) -> IngestResult:
        existing_id = hash_to_existing.get(image_hash)
        if existing_id:
            return IngestResult(
                status='duplicate',
                image_id=existing_id,
                image_path=image_path,
                imohash=image_hash,
            )
        async with sem:
            return await service.ingest_one(
                image_bytes,
                image_path,
                source=source,
                prefilled_image=prefilled_imgs.get(i),
                prefilled_items=prefilled_items.get(i),
                prefilled_secondary=prefilled_secondary.get(i),
                whole_frame_from_bytes=whole_frame_from_bytes,
                source_identifier=source_identifiers[i] if source_identifiers else None,
                ingest_run_id=ingest_run_id,
            )

    results = list(
        await asyncio.gather(
            *[
                _one(i, b, p, h)
                for i, (b, p, h) in enumerate(zip(images, image_paths, hashes, strict=False))
            ]
        )
    )

    for res in results:
        if res.status == 'duplicate':
            summary.duplicates += 1
        elif res.status == 'success':
            summary.successful += 1
            summary.crops_indexed += res.n_crops
        else:
            summary.failed += 1

    disagreements: list[dict[str, Any]] = []
    if label_paths is not None:
        label_summary: dict[str, Any] = await _import_batch_labels(
            service,
            image_paths,
            label_paths,
            results,
            label_source=label_source,
            detect_mismatches=detect_mismatches,
            disagreement_sink=disagreements,
        )
        summary.labels_imported += label_summary.get('labels_imported', 0)
        summary.mismatches += label_summary.get('mismatches', 0)
        summary.missed_labels += label_summary.get('missed_labels', 0)
        summary.unmatched_detections += label_summary.get('unmatched_detections', 0)

    if summary.failed == 0:
        status: Literal['success', 'partial', 'error'] = 'success'
    elif summary.successful == 0:
        status = 'error'
    else:
        status = 'partial'

    return BatchIngestResult(
        status=status, summary=summary, results=results, disagreements=disagreements
    )


__all__ = ['run_ingest_batch']
