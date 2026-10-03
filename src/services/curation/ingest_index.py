"""The index half of ingest: everything after a detector (or a dataset
label) has produced items for one decoded image.

:class:`~src.services.curation.ingest.CurationIngestService` splits one
image's ingest into ``ingest_image`` (decode, dedup, image identity) and
:func:`index_items` (this module): per-crop and whole-frame PE embeddings,
rank/blur quality, cluster placement, the crop cache, item docs and the
bulk index. ``ingest_one`` runs ``ingest_image`` -> detector ->
``index_items``; a dataset import (W10) runs ``ingest_image`` -> its own
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
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.clustering.ivf_ingest import get_ivf_ingest_store, ingest_passes_gate
from src.services.curation.embedding_state import EMBEDDED, FAILED
from src.services.curation.ingest_models import ERROR_KIND_BULK_INDEX, IngestResult
from src.services.curation.ingest_policy import select_for_embedding
from src.services.curation.item_doc import (
    DetectedItem,
    build_image_doc,
    build_item_doc,
    region_seed_status,
)
from src.services.curation.source_image_cache import maybe_prune_crop_cache, write_crop_cache
from src.services.detection.crop_quality import blur_ratio, crop_lap_var, image_lap_var
from src.services.detection.geometry import bbox_norm as _bbox_norm_fn, crop_id as _crop_id


if TYPE_CHECKING:
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
    pil: Image.Image
    width: int
    height: int
    imohash: str
    image_bytes: bytes
    source: str
    source_identifier: str | None = None
    ingest_run_id: str | None = None
    whole_frame_from_bytes: bool = False
    dataset_split: str | None = None
    """The split the images doc is filed under (a dataset import's); items
    indexed on this image later inherit it."""
    image_extra: dict[str, Any] = field(default_factory=dict)
    """Extra fields merged into the images doc this call creates."""


@dataclass
class IndexOutcome:
    result: IngestResult
    crop_ids: list[str] = field(default_factory=list)
    created_ids: list[str] = field(default_factory=list)


def crop_pil(img: Image.Image, bbox_pixel: tuple[float, float, float, float]) -> Image.Image:
    x1, y1, x2, y2 = bbox_pixel
    x1i = max(0, round(x1))
    y1i = max(0, round(y1))
    x2i = max(x1i + 1, round(x2))
    y2i = max(y1i + 1, round(y2))
    return img.crop((x1i, y1i, x2i, y2i))


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
    crops_pil: list[Image.Image],
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
            items[i].backbone_embedding = None
    try:
        if wanted:
            embeddings = await service.pe_encoder.embed_crops(
                [np.asarray(crops_pil[i]) for i in wanted], max_batch=service.profile.batch_limit
            )
            for i, emb in zip(wanted, embeddings, strict=False):
                items[i].pe_embedding = emb
    except Exception as exc:
        logger.warning('ingest_embed_crops_failed', path=image_path, error=str(exc))
    for i in wanted:
        items[i].embedding_state = EMBEDDED if items[i].pe_embedding is not None else FAILED


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
    img = ctx.pil
    full_w, full_h = ctx.width, ctx.height
    image_path = ctx.image_path
    crops_pil = [crop_pil(img, item.bbox_pixel) for item in items]
    await embed_items(service, items, crops_pil, full_w, full_h, image_path)

    now = datetime.now(UTC).isoformat()
    image_doc: dict[str, Any] | None = None
    if ctx.created:
        whole_frame_embedding = None
        try:
            if ctx.whole_frame_from_bytes:
                whole_frame_embedding = await service.pe_encoder.embed_whole_frame_bytes(
                    ctx.image_bytes
                )
            else:
                whole_frame_embedding = await service.pe_encoder.embed_whole_frame(image_path)
        except Exception as exc:
            logger.warning('ingest_embed_whole_frame_failed', path=image_path, error=str(exc))
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

    try:
        img_bgr = np.ascontiguousarray(np.asarray(img.convert('RGB'))[:, :, ::-1])
        full_var = image_lap_var(img_bgr)
    except Exception as exc:
        logger.warning('ingest_blur_full_var_failed', path=image_path, error=str(exc))
        img_bgr = None
        full_var = 0.0

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
    if any(it.class_id is None and it.pe_embedding is not None for it in items):
        store = get_ivf_ingest_store()

    crop_docs: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        bn = bnorms[idx]
        cid = cids[idx]
        box_var = crop_lap_var(img_bgr, item.bbox_pixel) if img_bgr is not None else None
        ratio = blur_ratio(box_var, full_var)

        write_crop_cache(cid, crops_pil[idx], service.config.crop_cache_dir)

        if item.class_id is not None:
            item.cluster_id = int(item.class_id)
        elif item.pe_embedding is not None and store is not None:
            placed = residual_placement(store, item.pe_embedding, rank_by_idx[idx], ratio)
            if placed is not None:
                item.cluster_id, item.cluster_distance = placed

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

    await asyncio.to_thread(
        maybe_prune_crop_cache, service.config.crop_cache_dir, service.config.crop_cache_max_bytes
    )

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


def image_id_for(image_path: str, imohash: str) -> str:
    """``sha256(path|imohash)[:32]``: the images-doc id every ingest path
    (detector ingest and dataset import) derives."""
    return hashlib.sha256(f'{image_path}|{imohash}'.encode()).hexdigest()[:32]


__all__ = [
    'PARKED_CLUSTER_ID',
    'ImageContext',
    'IndexOutcome',
    'crop_pil',
    'image_id_for',
    'index_items',
    'residual_placement',
]
