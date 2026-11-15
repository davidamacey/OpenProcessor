"""Batch orchestration for the curation ingest pipeline.

Split out of :mod:`src.services.curation.ingest` so that module keeps one
concern (the per-image pipeline) and this one keeps another: how a request's
photos are brought to the flow.

``CurationIngestService.ingest_batch`` is a thin delegate to
:func:`run_ingest_batch` here:

1. **Dedup once, for everyone**: a single query resolves every image's ``imohash`` against
   ``image_contents`` (only files on a match are compared by SHA-256). This resolves against
   what was stored before the request started; byte-identical repeats *within* one request
   collapse onto a single representative (``batch_dup_of``). Copies in *concurrent* requests are
   kept apart at write time: the images doc is created under a content-derived id, and the copy
   that loses the create is a ``duplicate``.
2. **Admission**: the encoded megabytes the request will hold are charged against the process's
   in-flight cap, so more clients queue here. When the project uses the frame engine and the
   request has a new photo but the engine is not ready, the request answers 503 before any work.
3. **One flow per wave**: every new photo moves on its own from the engine's answer to its write
   (:mod:`~src.services.curation.ingest_finish`): the API never decodes it (the engine does), and
   the database work is at most three commit groups. Photos of one ``imohash`` that cannot be
   decided in one transaction (two files with one ``imohash`` and other bytes) finish in later
   waves.
4. **No refresh**: the request does not refresh the indexes (a bulk run reads nothing of its
   writes by search); a caller that promises read-your-writes refreshes itself.

Ingesting an already-labeled dataset is ``POST /datasets/imports``: this module carries no
label-import machinery.

When every new photo of the request ended ``engine_unavailable`` nothing was stored, so the
request answers 503 with ``Retry-After`` (:class:`~src.core.exceptions.EngineUnavailableError`)
and the caller resends.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

from src.core.exceptions import EngineUnavailableError
from src.core.logging import get_logger
from src.services.curation.ingest_dedup import held_back_result
from src.services.curation.ingest_finish import PhotoJob
from src.services.curation.ingest_gpu_decode import engine_ready, engine_stage_applies
from src.services.curation.ingest_models import (
    ERROR_KIND_ENGINE_UNAVAILABLE,
    TRANSIENT_ERROR_KINDS,
    BatchIngestResult,
    IngestResult,
    IngestSummary,
)
from src.services.curation.ops_metrics import record_ingest_result
from src.utils.cpu_pool import encoded_nbytes, ingest_gate, mib
from src.utils.stage_timing import observe_stage


if TYPE_CHECKING:
    from src.services.curation.ingest import CurationIngestService
    from src.services.curation.ingest_dedup import Resolved


logger = get_logger(__name__)


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


def _copies_to_retry(
    batch_dup_of: dict[int, int],
    stored: dict[int, IngestResult],
    image_paths: list[str],
    source_identifiers: list[str | None] | None,
) -> list[int]:
    """The copies that get their own real attempt: a copy of a representative that failed (e.g.
    undecodable bytes) must not be answered by a guessed duplicate-of relationship. A representative
    that failed for a reason that says nothing about the file (the engine or the database was busy)
    is not tried again here: its copies are answered with the same failure (set into ``stored``)
    and the client resends."""
    retry: list[int] = []
    for i, rep in batch_dup_of.items():
        if stored[rep].status != 'failed':
            continue
        if stored[rep].error_kind in TRANSIENT_ERROR_KINDS:
            stored[i] = record_ingest_result(
                stored[rep].model_copy(
                    update={
                        'image_path': image_paths[i],
                        'source_identifier': source_identifiers[i] if source_identifiers else None,
                    }
                )
            )
        else:
            retry.append(i)
    return retry


def _summarize(results: list[IngestResult]) -> IngestSummary:
    summary = IngestSummary()
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

    return summary


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
    from src.services.near_duplicates.content_key import content_imohash

    if len(images) != len(image_paths):
        raise ValueError('images and image_paths must be same length')
    if source_identifiers is not None and len(source_identifiers) != len(images):
        raise ValueError('source_identifiers must match images length')

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

    # Hold the megabytes this request will bring against the process's in-flight cap, so more
    # clients queue here instead of all working at once (#241). A photo of the engine stage is
    # never decoded in the API, so it holds only its encoded bytes.
    use_engine = engine_stage_applies(service)
    if use_engine and non_dup and not await engine_ready(service):
        msg = 'the image engine is not ready'
        raise EngineUnavailableError(msg)
    held = sum(mib(len(images[i]) if use_engine else encoded_nbytes(images[i])) for i in non_dup)
    wait_start = time.perf_counter()
    async with ingest_gate().hold(held):
        observe_stage('admission_wait', time.perf_counter() - wait_start)

        def job(i: int, looked_up: Resolved | None = None) -> PhotoJob:
            # Clients match results to request entries by source_identifier, so
            # every duplicate carries its own, even though image_id is shared.
            return PhotoJob(
                images[i],
                image_paths[i],
                source,
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
            jobs = [job(i, r) for i, r in zip(wave, looked_up, strict=True)]
            stored.update(zip(wave, await service.ingest_photos(jobs), strict=True))
        if non_dup and all(stored[i].error_kind == ERROR_KIND_ENGINE_UNAVAILABLE for i in non_dup):
            msg = 'the image engine took none of the new photos of this request'
            raise EngineUnavailableError(msg)
        retry = _copies_to_retry(batch_dup_of, stored, image_paths, source_identifiers)
        for again in _rounds(retry, hashes):
            jobs = [job(i) for i in again]
            stored.update(zip(again, await service.ingest_photos(jobs), strict=True))

    results = [
        _result_of(i, image_paths, hashes, source_identifiers, existing_ids, batch_dup_of, stored)
        for i in range(len(images))
    ]

    summary = _summarize(results)

    if summary.failed == 0:
        status: Literal['success', 'partial', 'error'] = 'success'
    elif summary.successful == 0:
        status = 'error'
    else:
        status = 'partial'

    return BatchIngestResult(status=status, summary=summary, results=results)


__all__ = ['run_ingest_batch']
