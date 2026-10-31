"""The index half of ingest: everything after a detector (or a dataset
label) has produced items for one decoded image.

:class:`~src.services.curation.ingest.CurationIngestService` splits one
image's ingest into ``ingest_image`` (decode, dedup, image identity) and
:func:`index_items` (this module): per-crop and whole-frame PE embeddings,
rank/blur quality, cluster placement, the crop cache, item docs and the
bulk index. ``ingest_one`` runs ``ingest_image`` -> detector ->
``index_items``; a dataset import runs ``ingest_image`` -> its own
label boxes -> the same ``index_items``, so an imported item is built by
exactly the code a detected one is.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.logging import get_logger, get_request_id
from src.domain.embedding_state import EMBEDDED, FAILED
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.clustering.ivf_ingest import get_ivf_ingest_store, ingest_passes_gate
from src.services.curation.ingest_gpu_decode import GpuDecodeFrame
from src.services.curation.ingest_models import ERROR_KIND_BULK_INDEX, IngestResult
from src.services.curation.ingest_policy import select_for_embedding
from src.services.curation.item_doc import (
    DetectedItem,
    build_image_doc,
    build_item_doc,
    region_seed_status,
)
from src.services.curation.source_image_cache import (
    maybe_prune_crop_cache,
    project_crop_cache_dir,
    write_crop_cache,
)
from src.services.detection.crop_quality import blur_ratio, crop_lap_var, image_lap_var
from src.services.detection.geometry import (
    bbox_norm as _bbox_norm_fn,
    crop_id as _crop_id,
    crop_rect,
)
from src.utils.cpu_pool import run_cpu
from src.utils.stage_timing import stage_timer


if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import Image

    from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore
    from src.services.curation.ingest import CurationIngestService


logger = get_logger(__name__)

PARKED_CLUSTER_ID = -3


@dataclass
class ImageContext:
    """One decoded image, ready for :func:`index_items`.

    ``created`` is False when ``ingest_image`` resolved the bytes to an
    image doc that already exists (imohash dedup): its ``image_id`` and
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


def residual_placement(
    store: IVFCentroidStore, embedding: Any, rank: int | None, blur_ratio: float | None
) -> tuple[int, float | None] | None:
    """Where an embedded item with no class goes: ``(cluster_id, distance)``,
    the parked cluster when it fails the ingest clustering gate, or ``None``
    when the store cannot place it. The one placement rule, shared by ingest
    and the embed step so a later-embedded item lands where an ingest-embedded
    one would have."""
    if not ingest_passes_gate(rank, blur_ratio):
        return PARKED_CLUSTER_ID, None
    try:
        centroid, distance = store.assign_one_with_distance(embedding)
    except Exception as exc:
        logger.debug('ingest_ivf_assign_failed', error=str(exc))
        return None
    return int(centroid) + RESIDUAL_CLUSTER_ID_OFFSET, distance


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
    except Exception as exc:
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
    await asyncio.gather(
        *(
            run_cpu(
                write_crop_cache,
                cid,
                crop,
                project_dir,
                nbytes=len(crop) if isinstance(crop, bytes) else crop.width * crop.height * 3,
            )
            for cid, crop in zip(cids, crops, strict=True)
        )
    )
    await asyncio.to_thread(
        maybe_prune_crop_cache, service.config.crop_cache_dir, service.config.crop_cache_max_bytes
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
    except Exception as exc:
        logger.warning('ingest_embed_whole_frame_failed', path=ctx.image_path, error=str(exc))
        return None


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
    validated labels the region worker must never see).
    """
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

    crop_items: Sequence[Image.Image | bytes] = crops
    now = datetime.now(UTC).isoformat()
    image_doc: dict[str, Any] | None = None
    if ctx.created:
        whole_frame_embedding = await _whole_frame_vector(service, ctx)
        with stage_timer('doc_build'):
            image_doc = build_image_doc(
                image_id=ctx.image_id,
                image_path=image_path,
                source=ctx.source,
                width=full_w,
                height=full_h,
                imohash=ctx.imohash,
                now=now,
                whole_frame_embedding=whole_frame_embedding,
                source_identifier=ctx.source_identifier,
                ingest_run_id=ctx.ingest_run_id,
            )
            image_doc.update(ctx.image_extra)

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
    except Exception as exc:
        logger.warning('ingest_blur_full_var_failed', path=image_path, error=str(exc))

    bnorms = [_bbox_norm_fn(it.bbox_pixel, full_w, full_h) for it in items]
    areas = [max(0.0, bn[2] - bn[0]) * max(0.0, bn[3] - bn[1]) for bn in bnorms]
    cids = [_crop_id(ctx.image_id, bn) for bn in bnorms]
    rank_by_idx = {
        idx: rank
        for rank, idx in enumerate(
            sorted(range(len(items)), key=lambda i: (-areas[i], cids[i])), start=1
        )
    }

    store: IVFCentroidStore | None = None
    if service.policy.clustering.automatic and any(
        it.class_id is None and it.pe_embedding is not None for it in items
    ):
        store = get_ivf_ingest_store()

    box_vars: list[float | None]
    if frame is not None:
        box_vars = [frame.box_var(item) for item in items]
    elif rgb_frame is not None:
        box_vars = await run_cpu(
            _box_blurs, rgb_frame, [item.bbox_pixel for item in items], nbytes=_box_bytes(items)
        )
    else:
        box_vars = [None] * len(items)
    crop_docs: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        bn = bnorms[idx]
        cid = cids[idx]
        box_var = box_vars[idx]
        ratio = blur_ratio(box_var, full_var)

        if item.class_id is not None:
            item.cluster_id = int(item.class_id)
        elif item.pe_embedding is not None and store is not None:
            placed = residual_placement(store, item.pe_embedding, rank_by_idx[idx], ratio)
            if placed is not None:
                item.cluster_id, item.cluster_distance = placed

        with stage_timer('doc_build'):
            doc = build_item_doc(
                crop_id=cid,
                image_id=ctx.image_id,
                image_path=image_path,
                source=ctx.source,
                request_id=get_request_id(),
                bbox_norm=bn,
                item=item,
                now=now,
                crop_area_norm=areas[idx],
                crop_rank_in_image=rank_by_idx[idx],
                blur_full_var=full_var,
                blur_lap_var=box_var,
                blur_lap_ratio=ratio,
                region_status=(
                    region_seed_status(item)
                    if seed_region and service.region_seed_status is not None
                    else None
                ),
            )
            if ctx.dataset_split and 'dataset_split' not in doc:
                doc['dataset_split'] = ctx.dataset_split
        crop_docs.append(doc)

    created_ids: list[str] = []
    try:
        bulk_result = await service._bulk_index(
            image_doc, crop_docs, created_ids, seed_region=seed_region
        )
    except Exception as exc:
        logger.error('ingest_bulk_index_failed', path=image_path, error=str(exc))
        return IndexOutcome(
            result=IngestResult(
                status='failed',
                image_path=image_path,
                source_identifier=ctx.source_identifier,
                imohash=ctx.imohash,
                error=str(exc),
                error_kind=ERROR_KIND_BULK_INDEX,
            )
        )
    if bulk_result.get('duplicate'):
        # Another copy of this content created the images doc first (#358): this copy
        # wrote nothing, so it answers as the dedup lookup would have.
        return IndexOutcome(
            result=IngestResult(
                status='duplicate',
                image_id=ctx.image_id,
                image_path=image_path,
                source_identifier=ctx.source_identifier,
                imohash=ctx.imohash,
            )
        )
    await _cache_crops(service, cids, crop_items)
    service._publish_created(created_ids, image_path)

    return IndexOutcome(
        result=IngestResult(
            status='success',
            image_id=ctx.image_id,
            image_path=image_path,
            source_identifier=ctx.source_identifier,
            imohash=ctx.imohash,
            n_crops=len(crop_docs),
            n_embedded=sum(it.embedding_state == EMBEDDED for it in items),
            n_not_embedded=sum(it.embedding_state != EMBEDDED for it in items),
            n_embed_failed=sum(it.embedding_state == FAILED for it in items),
            crops_created=bulk_result.get('crops_created', 0),
            crops_updated=bulk_result.get('crops_updated', 0),
            crops_preserved_human=bulk_result.get('crops_preserved_human', 0),
            crops_final_conflicts=bulk_result.get('crops_final_conflicts', 0),
            n_region_queued=bulk_result.get('region_queued', 0),
            secondary_detector_error=secondary_detector_error,
        ),
        crop_ids=cids,
        created_ids=created_ids,
    )


def content_image_id(imohash: str) -> str:
    """``sha256(imohash)[:32]``: the images-doc id every ingest path (detector
    ingest and dataset import) derives. It names the content, not the path, so
    the ``create`` of the images doc is what keeps one image per content when
    copies race past the imohash lookup (#358)."""
    return hashlib.sha256(imohash.encode()).hexdigest()[:32]


__all__ = [
    'PARKED_CLUSTER_ID',
    'ImageContext',
    'IndexOutcome',
    'content_image_id',
    'crop_pil',
    'index_items',
    'residual_placement',
]
