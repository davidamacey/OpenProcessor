"""One photo's stage of an ingest: from its bytes to a prepared image, alone.

A stage worker (:func:`~src.services.curation.ingest_finish.finish_photos`) takes one photo at a
time and runs :func:`stage_photo` on it, so the photos of a request overlap: while one waits for
the frame engine another is being written. There are two implementations, picked per project and
deployment by :func:`~src.services.curation.ingest_gpu_decode.engine_stage_applies`:

* **engine stage**, the rule: the frame engine decodes the photo (nvJPEG for a baseline JPEG, a
  CPU decode inside the engine for every other image), runs the detector and the PE encoder on
  it, and returns boxes, vectors, blur sums and crop JPEGs. The API never touches the pixels. The
  engine call and the capture read (ExifTool, new photos only) run together.
* **host stage**, only for a project with a secondary detector or a letterbox fill that is not
  one grey, and for ``OP_INGEST_GPU_DECODE=0``: the API decodes, detects (one Triton call per
  photo; the model server's dynamic batcher groups the concurrent ones) and embeds.

Both end in the same place: the detect filter applied, the embedding states stamped, blur sums and
ids computed (:func:`~src.services.curation.ingest_index.prepare_image`). Classes are not resolved
here: the embedding mask reads the label (``class_name or proposal_name``, slug-normalized), which
resolution leaves equal (a detector that assigns its class sets the same name itself), and the
documents are built after the committer resolved the classes of each commit group.

A failure of one photo is a value (an :class:`~src.services.curation.ingest_models.IngestResult`),
never an exception: it must not cancel the photos around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.logging import get_logger
from src.services.curation.ingest_gpu_decode import (
    EngineFailure,
    EngineStaged,
    count_route,
    engine_failure_kind,
    run_engine_stage,
)
from src.services.curation.ingest_index import IndexOutcome, PreparedImage, prepare_image
from src.services.curation.ingest_models import (
    ERROR_KIND_DETECTOR_INFER,
    ERROR_KIND_ENGINE_UNAVAILABLE,
    IngestResult,
)
from src.utils.stage_timing import stage_timer
from src.utils.task_group import run_all


if TYPE_CHECKING:
    from src.domain.capture_metadata import CaptureMetadata
    from src.services.curation.ingest import CurationIngestService
    from src.services.curation.ingest_finish import PhotoJob
    from src.services.curation.ingest_index import ImageContext


logger = get_logger(__name__)


@dataclass
class StagedPhoto:
    """A photo ready for its commit group: embedded, scored, named; classes still to resolve."""

    prepared: PreparedImage
    n_filtered: int


def failed_result(
    job: PhotoJob, kind: str, message: str, decode_route: str | None = None
) -> IngestResult:
    return IngestResult(
        status='failed',
        image_path=job.image_path,
        source_identifier=job.source_identifier,
        error=message,
        error_kind=kind,
        decode_route=decode_route,
    )


def _prepared_or_result(
    prepared: PreparedImage | IndexOutcome, ctx: ImageContext, n_filtered: int
) -> StagedPhoto | IngestResult:
    if isinstance(prepared, IndexOutcome):  # the blur sums could not be computed
        prepared.result.n_filtered = n_filtered
        prepared.result.decode_route = ctx.decode_route
        return prepared.result
    return StagedPhoto(prepared, n_filtered)


async def _capture(
    service: CurationIngestService, data: bytes, *, new_photo: bool
) -> CaptureMetadata | None:
    if not new_photo:
        return None
    with stage_timer('capture'):
        return await service.read_capture(data)


async def stage_engine(
    service: CurationIngestService, job: PhotoJob, *, engine_ready: bool
) -> StagedPhoto | IngestResult:
    """Identity, then the engine call together with the capture read, then the detect filter and
    the preparation. ``engine_ready`` False answers ``engine_unavailable`` for a photo that needs
    the engine (a duplicate is still answered as one)."""
    identity = await service.identify(
        job.image_bytes, job.image_path, job.source_identifier, resolved=job.resolved
    )
    if isinstance(identity, IngestResult):
        return identity
    if not engine_ready:
        return failed_result(
            job, ERROR_KIND_ENGINE_UNAVAILABLE, 'the image engine is not ready; retry later'
        )
    produced, capture = await run_all(
        [
            run_engine_stage(service, job.image_bytes),
            _capture(service, job.image_bytes, new_photo=identity.resolved.existing is None),
        ]
    )
    assert isinstance(produced, EngineStaged | EngineFailure)
    if isinstance(produced, EngineFailure):
        logger.warning('ingest_engine_stage_failed', path=job.image_path, reason=produced.reason)
        return failed_result(job, engine_failure_kind(produced), produced.detail)
    count_route(produced.decode_route)
    ctx = service.context_for(
        identity,
        job.image_bytes,
        job.image_path,
        job.source,
        width=produced.frame.width,
        height=produced.frame.height,
        gpu_frame=produced.frame,
        decode_route=produced.decode_route,
        capture=capture,  # type: ignore[arg-type]
        source_identifier=job.source_identifier,
        ingest_run_id=job.ingest_run_id,
    )
    detected = await service.filter_detections(produced.items, ctx.width, ctx.height, resolve=False)
    prepared = await prepare_image(service, ctx, detected.items)
    return _prepared_or_result(prepared, ctx, detected.n_filtered)


async def stage_host(service: CurationIngestService, job: PhotoJob) -> StagedPhoto | IngestResult:
    """The host stage: decode, identity and capture, the detectors, then the preparation."""
    ctx = await service.ingest_image(
        job.image_bytes,
        job.image_path,
        job.source,
        source_identifier=job.source_identifier,
        ingest_run_id=job.ingest_run_id,
        resolved=job.resolved,
    )
    if isinstance(ctx, IngestResult):
        return ctx
    detected = await service.detect_for_ingest(ctx, job.image_path)
    if isinstance(detected, Exception):
        logger.error('ingest_primary_detector_failed', path=job.image_path, error=str(detected))
        return failed_result(job, ERROR_KIND_DETECTOR_INFER, str(detected), ctx.decode_route)
    count_route(ctx.decode_route or '')
    prepared = await prepare_image(
        service,
        ctx,
        detected.items,
        secondary_detector_error=detected.secondary_detector_error,
    )
    return _prepared_or_result(prepared, ctx, detected.n_filtered)


__all__ = ['StagedPhoto', 'failed_result', 'stage_engine', 'stage_host']
