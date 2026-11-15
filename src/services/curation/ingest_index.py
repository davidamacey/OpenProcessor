"""The index half of ingest: everything after a detector (or a dataset
label) has produced items for one decoded image.

:class:`~src.services.curation.ingest.CurationIngestService` splits one
image's ingest into ``ingest_image`` (decode, dedup, image identity) and
:func:`index_items` (this module): per-crop and whole-frame PE embeddings,
rank/blur quality, the serving-partition placement, the crop cache, item docs and the
bulk index. ``ingest_one`` runs ``ingest_image`` -> detector ->
``index_items``; a dataset import runs ``ingest_image`` -> its own
label boxes -> the same ``index_items``, so an imported item is built by
exactly the code a detected one is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from opensearchpy.exceptions import OpenSearchException

from src.clients.curation_opensearch.capture_fields import capture_image_block, capture_item_fields
from src.core.logging import get_logger, get_request_id
from src.domain.embedding_state import EMBEDDED, FAILED
from src.domain.image_geometry import crop_rect
from src.domain.location_policy import apply_location_policy
from src.services.curation.image_completeness import holders_completeness
from src.services.curation.ingest_gpu_decode import GpuDecodeFrame
from src.services.curation.ingest_models import (
    ERROR_KIND_BLUR,
    ERROR_KIND_BULK_INDEX,
    ERROR_KIND_INCOMPLETE_WRITE,
    IngestResult,
)
from src.services.curation.ingest_policy import select_for_embedding
from src.services.curation.ingest_writer import REPAIR_AFTER_S, IngestWriteError, RepairTarget
from src.services.curation.item_doc import (
    DetectedItem,
    build_image_doc,
    build_item_doc,
    region_seed_status,
)
from src.services.curation.ops_metrics import count_whole_event
from src.services.curation.source_image_cache import project_crop_cache_dir, write_crop_cache
from src.services.detection.crop_quality import blur_ratio, crop_lap_var, image_lap_var
from src.services.detection.geometry import bbox_norm as _bbox_norm_fn, crop_id as _crop_id
from src.services.embedding_spaces.readers import WriteFields, frame_vector_field, item_write_fields
from src.services.near_duplicates import content_key, ingest_claim
from src.services.near_duplicates.content_key import ClaimOutcome, ContentClaim
from src.utils.cpu_pool import run_cpu
from src.utils.retry import TRITON_CALL_ERRORS
from src.utils.stage_timing import stage_timer
from src.utils.task_group import run_all


# What the Laplacian-variance maths raises: OpenCV, a missing host frame, a degenerate frame.
BLUR_ERRORS = (cv2.error, ValueError, ArithmeticError)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import Image

    from src.domain.capture_metadata import CaptureMetadata
    from src.services.clustering.assign import Placement
    from src.services.curation.ingest import CurationIngestService


logger = get_logger(__name__)


@dataclass
class ImageContext:
    """One decoded image, ready for :func:`index_items`.

    ``created`` is False when ``ingest_image`` resolved the bytes to an
    image doc that already exists (exact-content dedup): its ``image_id`` and
    ``image_path`` are the existing doc's, and ``index_items`` writes no
    images doc for it.
    """

    image_id: str
    image_path: str
    created: bool
    pil: Image.Image | None
    """The decoded frame; ``None`` when the GPU decode decoded the image (``gpu_frame``)."""
    width: int
    height: int
    imohash: str
    content: bytes
    """The file's bytes: the claim compares them with a stored file on an ``imohash`` match."""
    source: str
    source_identifier: str | None = None
    ingest_run_id: str | None = None
    rgb: np.ndarray | None = None
    """``pil``'s pixels as an HWC uint8 array when the decoder produced one (no copy for blur)."""
    gpu_frame: GpuDecodeFrame | None = None
    dataset_split: str | None = None
    """The split the images doc is filed under (a dataset import's); items
    indexed on this image later inherit it."""
    image_extra: dict[str, Any] = field(default_factory=dict)
    """Extra fields merged into the images doc this call creates."""
    decode_route: str | None = None
    """Which decode route the pixels took (:mod:`src.domain.decode_route`); stored on a created images doc."""
    capture: CaptureMetadata | None = None
    """The file's capture metadata as read, location included (``repr`` hides it); ``None`` for
    an image that already has a doc, whose capture is already stored."""
    repair: RepairTarget | None = None
    """The stored image this ingest completes (``created`` is False: its doc exists, written
    incomplete by an ingest that failed or was killed). Its items are written and its list set;
    no other field of the doc is written."""

    def require_pil(self) -> Image.Image:
        """The host-side frame; raises when the GPU decode kept the pixels on the device."""
        if self.pil is None:
            msg = f'image {self.image_id} was decoded on the GPU and has no host frame'
            raise ValueError(msg)
        return self.pil


@dataclass
class IndexOutcome:
    result: IngestResult
    crop_ids: list[str] = field(default_factory=list)
    created_ids: list[str] = field(default_factory=list)


def crop_pil(img: Image.Image, bbox_pixel: tuple[float, float, float, float]) -> Image.Image:
    """The pixels of the box that is stored: ``bbox_pixel`` cut to the frame, so the crop (and its
    PE vector) never carries the zero padding Pillow adds past the edge (#259)."""
    rect = crop_rect(bbox_pixel, *img.size)
    with stage_timer('crop'):
        return img.crop(rect)


def _box_bytes(items: list[DetectedItem]) -> int:
    """RGB bytes of the item boxes: what cutting, blur-scoring and encoding the crops touch."""
    return int(
        sum(
            max(0.0, x2 - x1) * max(0.0, y2 - y1) * 3
            for x1, y1, x2, y2 in (i.bbox_pixel for i in items)
        )
    )


def _crop_all(img: Image.Image, items: list[DetectedItem]) -> list[Image.Image]:
    return [crop_pil(img, item.bbox_pixel) for item in items]


def _full_frame_blur(img: Image.Image, rgb: np.ndarray | None) -> tuple[np.ndarray, float]:
    """The RGB frame and its Laplacian variance (the denominator of every box's ratio)."""
    with stage_timer('blur'):
        if rgb is None:
            rgb = np.asarray(img if img.mode == 'RGB' else img.convert('RGB'))
        return rgb, image_lap_var(rgb, 'rgb')


def _box_blurs(
    rgb: np.ndarray, boxes: list[tuple[float, float, float, float]]
) -> list[float | None]:
    with stage_timer('blur'):
        return [crop_lap_var(rgb, box, 'rgb') for box in boxes]


async def place_vectors(item_ids: Sequence[str], vectors: np.ndarray) -> list[Placement]:
    """Place new items by the serving partition's prototypes (4F ``assign.place_new``); empty when
    the project has no serving partition for the item target. A failure of the read refuses the
    call: a placement that silently vanished would leave an item looking unclustered."""
    from src.db.session import async_session_factory, project_tx
    from src.services.clustering.assign import place_new
    from src.services.config_store.bound_project import bound_project_id

    project = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project):
        return await place_new(
            session, project=project, target='item', item_ids=item_ids, vectors=vectors
        )


async def place_prepared(service: CurationIngestService, prepared: Sequence[PreparedImage]) -> None:
    """Stamp every embedded item of the batch with its place in the serving partition, all in one
    read (:func:`place_vectors`): the prototypes are read once, not once per photo.

    Nothing is written for an item without a vector, and for all of them when the project has no
    serving partition for the item target (an unclustered project behaves as before).
    """
    if not service.policy.clustering.automatic:
        return
    owners = [
        (p.cids[i], item)
        for p in prepared
        for i, item in enumerate(p.items)
        if item.pe_embedding is not None
    ]
    if not owners:
        return
    with stage_timer('placement'):
        placements = await place_vectors(
            [item_id for item_id, _ in owners],
            np.asarray([item.pe_embedding for _, item in owners], dtype=np.float32),
        )
    if not placements:
        return
    for (_, item), placement in zip(owners, placements, strict=True):
        item.placement = placement


async def embed_items(
    service: CurationIngestService,
    items: list[DetectedItem],
    crops: list[Image.Image] | GpuDecodeFrame,
    width: int,
    height: int,
    image_path: str,
) -> None:
    """Give the items the project's embedding policy selects a vector and
    stamp every item's ``embedding_state``: ``embedded``, ``failed`` (the
    encoder raised) or the policy's skip state. Only selected crops reach the
    encoder; a skipped item keeps no vector of any kind."""
    states = select_for_embedding(items, service.policy.embedding, width, height)
    wanted: list[int] = []
    for i, state in enumerate(states):
        if state == EMBEDDED:
            wanted.append(i)
        else:
            items[i].embedding_state = state
            items[i].pe_embedding = None
    try:
        if isinstance(crops, GpuDecodeFrame):
            for i in wanted:
                items[i].pe_embedding = crops.crop_embedding(items[i])
        elif wanted:
            embeddings = await service.pe_encoder.embed_crops(
                [np.asarray(crops[i]) for i in wanted], max_batch=service.profile.batch_limit
            )
            for i, emb in zip(wanted, embeddings, strict=False):
                items[i].pe_embedding = emb
    except TRITON_CALL_ERRORS as exc:
        logger.warning('ingest_embed_crops_failed', path=image_path, error=str(exc))
    for i in wanted:
        items[i].embedding_state = EMBEDDED if items[i].pe_embedding is not None else FAILED


async def _cache_crops(
    service: CurationIngestService, cids: list[str], crops: Sequence[Image.Image | bytes]
) -> None:
    """Fill the crop cache for the items just written. After the write, so a copy that
    lost the images-doc ``create`` leaves no crops behind; a reader that comes first
    re-crops from the source, as on any cache miss."""
    project_dir = project_crop_cache_dir(service.config.crop_cache_dir)
    await run_all(
        [
            run_cpu(
                write_crop_cache,
                cid,
                crop,
                project_dir,
                max_side=service.config.crop_max_side,
                nbytes=len(crop) if isinstance(crop, bytes) else crop.width * crop.height * 3,
            )
            for cid, crop in zip(cids, crops, strict=True)
        ]
    )


async def _whole_frame_vector(service: CurationIngestService, ctx: ImageContext) -> Any:
    """The image's whole-frame vector, or ``None`` for detection only (mode ``none``
    stores no vector of any kind; the GPU decode computes this one in its single pass
    and it is dropped here) or when the encoder fails."""
    if service.policy.embedding.mode == 'none':
        return None
    try:
        if ctx.gpu_frame is not None:
            return ctx.gpu_frame.whole_embedding
        assert ctx.pil is not None
        return await service.pe_encoder.embed_whole_frame_image(ctx.pil)
    except TRITON_CALL_ERRORS as exc:
        logger.warning('ingest_embed_whole_frame_failed', path=ctx.image_path, error=str(exc))
        return None


async def _held_elsewhere(
    service: CurationIngestService, ctx: ImageContext, holder: str
) -> IndexOutcome:
    """Another copy claimed this content first: nothing of this one is written. The answer is
    ``duplicate`` only when that image is whole (the rule of the content lookup, the same
    function); a holder that has claimed and not written yet, or whose items are not all there,
    is ``incomplete_write``: send the file again later."""
    try:
        state = (
            await holders_completeness(
                service.opensearch,
                service.config.images_index,
                service.config.items_index,
                [holder],
                repair_after_s=REPAIR_AFTER_S,
            )
        )[holder]
    except OpenSearchException as exc:
        logger.error('ingest_holder_check_failed', path=ctx.image_path, error=str(exc))
        return _failed_write(ctx, exc, ERROR_KIND_BULK_INDEX)
    if state != 'complete':
        count_whole_event('held_back')
        msg = f'image {holder} holds these bytes and is not stored whole yet; retry later'
        return _failed_write(ctx, ValueError(msg), ERROR_KIND_INCOMPLETE_WRITE)
    return IndexOutcome(
        result=IngestResult(
            status='duplicate',
            image_id=holder,
            image_path=ctx.image_path,
            source_identifier=ctx.source_identifier,
            imohash=ctx.imohash,
        )
    )


@dataclass
class PreparedImage:
    """One image with everything computed before its documents are built: embeddings, blur, ids."""

    ctx: ImageContext
    items: list[DetectedItem]
    crops: Sequence[Image.Image | bytes]
    cids: list[str]
    bnorms: list[list[float]]
    areas: list[float]
    rank_by_idx: dict[int, int]
    full_var: float
    box_vars: list[float | None]
    whole_frame_embedding: Any
    seed_region: bool
    secondary_detector_error: str | None


async def prepare_image(
    service: CurationIngestService,
    ctx: ImageContext,
    items: list[DetectedItem],
    *,
    seed_region: bool = True,
    secondary_detector_error: str | None = None,
) -> PreparedImage | IndexOutcome:
    """Embed, score and name the items of one decoded image; a failed result when the blur
    cannot be computed."""
    img, frame = ctx.pil, ctx.gpu_frame
    full_w, full_h = ctx.width, ctx.height
    image_path = ctx.image_path
    crops: list[Image.Image] | list[bytes]
    if frame is not None:
        crops = [frame.crop_jpeg(it) for it in items]
        await embed_items(service, items, frame, full_w, full_h, image_path)
    else:
        assert img is not None
        crops = await run_cpu(_crop_all, img, items, nbytes=_box_bytes(items))
        await embed_items(service, items, crops, full_w, full_h, image_path)

    whole_frame_embedding = await _whole_frame_vector(service, ctx) if ctx.created else None
    rgb_frame = None
    full_var = 0.0
    try:
        if frame is not None:
            full_var = frame.full_var()
        else:
            assert img is not None
            rgb_frame, full_var = await run_cpu(
                _full_frame_blur, img, ctx.rgb, nbytes=full_w * full_h * 3
            )
    except BLUR_ERRORS as exc:
        logger.error('ingest_blur_failed', path=image_path, error=str(exc))
        return IndexOutcome(
            result=IngestResult(
                status='failed',
                image_path=image_path,
                source_identifier=ctx.source_identifier,
                imohash=ctx.imohash,
                error=str(exc),
                error_kind=ERROR_KIND_BLUR,
            )
        )

    bnorms = [_bbox_norm_fn(it.bbox_pixel, full_w, full_h) for it in items]
    areas = [max(0.0, bn[2] - bn[0]) * max(0.0, bn[3] - bn[1]) for bn in bnorms]
    cids = [_crop_id(ctx.image_id, bn) for bn in bnorms]
    rank_by_idx = {
        idx: rank
        for rank, idx in enumerate(
            sorted(range(len(items)), key=lambda i: (-areas[i], cids[i])), start=1
        )
    }
    box_vars: list[float | None]
    if frame is not None:
        box_vars = [frame.box_var(item) for item in items]
    elif rgb_frame is not None:
        box_vars = await run_cpu(
            _box_blurs, rgb_frame, [item.bbox_pixel for item in items], nbytes=_box_bytes(items)
        )
    else:
        box_vars = [None] * len(items)
    return PreparedImage(
        ctx, items, crops, cids, bnorms, areas, rank_by_idx, full_var, box_vars,
        whole_frame_embedding, seed_region, secondary_detector_error,
    )  # fmt: skip


@dataclass(frozen=True)
class DocContext:
    """What a batch's documents share, read once per batch: the vector fields and the location
    policy of the bound project."""

    frame_vector_field: str | None
    write_fields: WriteFields
    location_policy: str | None


async def doc_context(
    service: CurationIngestService, prepared: Sequence[PreparedImage]
) -> DocContext:
    created = any(p.ctx.created for p in prepared)
    located = any(p.ctx.created and p.ctx.capture is not None for p in prepared)
    return DocContext(
        frame_vector_field=await frame_vector_field() if created else None,
        write_fields=await item_write_fields(),
        location_policy=await service.location_policy() if located else None,
    )


@dataclass
class BuiltDocuments:
    image_doc: dict[str, Any] | None
    crop_docs: list[dict[str, Any]]


def build_documents(
    service: CurationIngestService, p: PreparedImage, shared: DocContext
) -> BuiltDocuments:
    """The images document (a new image only) and the item documents of one prepared image."""
    ctx, items = p.ctx, p.items
    now = datetime.now(UTC).isoformat()
    image_doc: dict[str, Any] | None = None
    if ctx.created:
        assert shared.frame_vector_field is not None
        with stage_timer('doc_build'):
            image_doc = build_image_doc(
                vector_field=shared.frame_vector_field,
                image_id=ctx.image_id,
                image_path=ctx.image_path,
                source=ctx.source,
                width=ctx.width,
                height=ctx.height,
                imohash=ctx.imohash,
                now=now,
                whole_frame_embedding=p.whole_frame_embedding,
                source_identifier=ctx.source_identifier,
                ingest_run_id=ctx.ingest_run_id,
                decode_route=ctx.decode_route,
            )
            image_doc.update(ctx.image_extra)
            if ctx.capture is not None:
                assert shared.location_policy is not None
                stored = apply_location_policy(ctx.capture, shared.location_policy)
                image_doc['capture'] = capture_image_block(stored)
    crop_docs: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        box_var = p.box_vars[idx]
        with stage_timer('doc_build'):
            doc = build_item_doc(
                vector_field=shared.write_fields.vector_field,
                state_field=shared.write_fields.state_field,
                crop_id=p.cids[idx],
                image_id=ctx.image_id,
                image_path=ctx.image_path,
                source=ctx.source,
                request_id=get_request_id(),
                bbox_norm=p.bnorms[idx],
                item=item,
                now=now,
                crop_area_norm=p.areas[idx],
                crop_rank_in_image=p.rank_by_idx[idx],
                blur_full_var=p.full_var,
                blur_lap_var=box_var,
                blur_lap_ratio=blur_ratio(box_var, p.full_var),
                region_status=(
                    region_seed_status(item)
                    if p.seed_region and service.region_seed_status is not None
                    else None
                ),
            )
            if ctx.dataset_split and 'dataset_split' not in doc:
                doc['dataset_split'] = ctx.dataset_split
            if ctx.capture is not None:
                doc.update(capture_item_fields(ctx.capture))
        crop_docs.append(doc)
    return BuiltDocuments(image_doc, crop_docs)


def _failed_write(ctx: ImageContext, exc: Exception, kind: str) -> IndexOutcome:
    return IndexOutcome(
        result=IngestResult(
            status='failed',
            image_path=ctx.image_path,
            source_identifier=ctx.source_identifier,
            imohash=ctx.imohash,
            error=str(exc),
            error_kind=kind,
        )
    )


async def write_prepared(
    service: CurationIngestService,
    p: PreparedImage,
    docs: BuiltDocuments,
    claim: ClaimOutcome,
    unclaimed: list[str],
) -> IndexOutcome:
    """Write one photo's documents after its claim, the crop cache after them.

    The terminal result when another copy holds the content, the claim was refused or the write
    fails: a photo is ``success`` only when all of its documents were stored. A claim this photo
    took is appended to ``unclaimed`` only when the store refused its images doc (it is known
    absent), so the batch releases it and a later attempt can store the content. Every other
    failure keeps the claim: the images doc may exist, and the claim is what sends the next
    ingest of the same bytes to the repair of it (a claim without a doc is released by that
    lookup)."""
    ctx = p.ctx
    if claim.holder is not None:
        return await _held_elsewhere(service, ctx, claim.holder)
    created_ids: list[str] = []
    try:
        if claim.error is not None:
            raise claim.error
        bulk_result = await service._bulk_index(
            docs.image_doc,
            docs.crop_docs,
            created_ids,
            seed_region=p.seed_region,
            repair=ctx.repair,
        )
    except IngestWriteError as exc:
        logger.error('ingest_bulk_index_failed', path=ctx.image_path, error=str(exc))
        if exc.release_claim and docs.image_doc is not None:
            unclaimed.append(ctx.image_id)
        return _failed_write(ctx, exc, ERROR_KIND_BULK_INDEX)
    except Exception as exc:
        logger.error('ingest_bulk_index_failed', path=ctx.image_path, error=str(exc))
        return _failed_write(ctx, exc, ERROR_KIND_BULK_INDEX)
    if bulk_result.get('incomplete'):
        # Another copy's doc misses items and its writer may still be writing: nothing was
        # written, the photo is not stored yet, and a later attempt decides.
        count_whole_event('held_back')
        msg = f'image {ctx.image_id} is stored incomplete and may still be written; retry later'
        return _failed_write(ctx, ValueError(msg), ERROR_KIND_INCOMPLETE_WRITE)
    if bulk_result.get('duplicate'):
        # Another copy of this content created the images doc first (#358): this copy
        # wrote nothing, so it answers as the dedup lookup would have.
        return IndexOutcome(
            result=IngestResult(
                status='duplicate',
                image_id=ctx.image_id,
                image_path=ctx.image_path,
                source_identifier=ctx.source_identifier,
                imohash=ctx.imohash,
            )
        )
    if bulk_result.get('repaired'):
        count_whole_event('repaired')
    await _cache_crops(service, p.cids, p.crops)
    items = p.items
    return IndexOutcome(
        result=IngestResult(
            status='success',
            image_id=ctx.image_id,
            image_path=ctx.image_path,
            source_identifier=ctx.source_identifier,
            imohash=ctx.imohash,
            n_crops=len(docs.crop_docs),
            n_embedded=sum(it.embedding_state == EMBEDDED for it in items),
            n_not_embedded=sum(it.embedding_state != EMBEDDED for it in items),
            n_embed_failed=sum(it.embedding_state == FAILED for it in items),
            crops_created=bulk_result.get('crops_created', 0),
            crops_updated=bulk_result.get('crops_updated', 0),
            crops_preserved_human=bulk_result.get('crops_preserved_human', 0),
            crops_final_conflicts=bulk_result.get('crops_final_conflicts', 0),
            n_region_queued=bulk_result.get('region_queued', 0),
            secondary_detector_error=p.secondary_detector_error,
        ),
        crop_ids=p.cids,
        created_ids=created_ids,
    )


async def commit_prepared(
    service: CurationIngestService, prepared: Sequence[PreparedImage]
) -> list[IndexOutcome]:
    """Place, claim, write and announce a batch of prepared images: the per-batch database steps
    run once for the batch (placement, the claim with the pending rows, the release of failed
    claims, the ``crop.created`` events), and the documents go through the coalescing writer
    photo by photo.

    What a photo ends with at each failure point is the table in
    :mod:`~src.services.curation.ingest_finish`. The batch's claim transaction, if it fails as a
    whole, runs photo by photo (:mod:`~src.services.near_duplicates.ingest_claim`).
    """
    if not prepared:
        return []
    await place_prepared(service, prepared)
    shared = await doc_context(service, prepared)
    built = [build_documents(service, p, shared) for p in prepared]
    claims = await ingest_claim.claim_and_mark(
        [
            ingest_claim.PhotoWrite(
                item_ids=p.cids,
                claim=ContentClaim(p.ctx.imohash, p.ctx.image_id, p.ctx.image_path, p.ctx.content)
                if docs.image_doc is not None
                else None,
            )
            for p, docs in zip(prepared, built, strict=True)
        ]
    )
    unclaimed: list[str] = []
    written: dict[int, IndexOutcome] = {}

    async def write(i: int) -> IndexOutcome:
        written[i] = await write_prepared(service, prepared[i], built[i], claims[i], unclaimed)
        return written[i]

    try:
        outcomes = await run_all([write(i) for i in range(len(prepared))])
    finally:
        if unclaimed:
            with stage_timer('release'):
                await content_key.release_bound(unclaimed)
        # Photos stored before a failure are announced too, as when each ran alone.
        announced = [
            (crop_id, prepared[i].ctx.image_path)
            for i, outcome in sorted(written.items())
            for crop_id in outcome.created_ids
        ]
        if announced:
            with stage_timer('publish'):
                await service._publish_created(announced)
    return outcomes


async def index_items(
    service: CurationIngestService,
    ctx: ImageContext,
    items: list[DetectedItem],
    *,
    seed_region: bool = True,
    secondary_detector_error: str | None = None,
) -> IndexOutcome:
    """Embed, score, place and bulk-index ``items`` (plus the images doc
    when ``ctx.created``) for one decoded image.

    ``seed_region`` False writes no ``pending_detection`` seed even under an
    active region profile (a ``processing: none`` import: the items are
    validated labels the region worker must never see). A batch of one through the same steps
    :func:`commit_prepared` runs for a whole batch.
    """
    prepared = await prepare_image(
        service,
        ctx,
        items,
        seed_region=seed_region,
        secondary_detector_error=secondary_detector_error,
    )
    if isinstance(prepared, IndexOutcome):
        return prepared
    [outcome] = await commit_prepared(service, [prepared])
    return outcome


__all__ = [
    'ImageContext',
    'IndexOutcome',
    'PreparedImage',
    'commit_prepared',
    'crop_pil',
    'index_items',
    'place_vectors',
    'prepare_image',
]
