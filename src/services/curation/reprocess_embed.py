"""The ``embed`` scope: recompute derived vectors.

Embeddings are derived data, not labels, so locked items are included. Per
item: the PE crop embedding (``pe_embedding``) and one region embedding per
embeddable box (accepted and false-positive; ``region_box_embeddings``,
written through ``write_box_embeddings`` so the box list is never touched);
per image: the whole-frame PE embedding on the images doc. Written: the
vectors, ``embedding_state`` and, for an item that had neither a class nor a
cluster (stored without a vector), the cluster placement ingest would have
given it. Class fields are never written.

``reembed_items`` is the one function behind the ``embed`` reprocess scope
and ``scripts/curation/backfill_region_embeddings.py``. Given a GPU route
(:mod:`~src.services.curation.reprocess_gpu_embed`), a photo at or above the
ingest size gate is decoded and its stored boxes embedded on the GPU; every
other photo, and any the GPU decode declines, is decoded here.
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageOps

from src.config import get_curation_config
from src.core.logging import get_logger
from src.domain.embedding_state import EMBEDDED
from src.domain.region_boxes import read_boxes
from src.services.curation.clustering.ivf_ingest import get_ivf_ingest_store
from src.services.curation.image_serving import is_servable_image_path
from src.services.curation.ingest_index import crop_pil, residual_placement
from src.services.curation.region_box_embeddings import (
    embeddable,
    entry_for,
    missing_boxes,
    write_box_embeddings,
)
from src.services.curation.reprocess_gpu_embed import embed_on_gpu
from src.services.detection.region_embed import embed_region_crops
from src.utils.cpu_pool import offload_always, run_cpu
from src.utils.vectors import float_list


if TYPE_CHECKING:
    from collections.abc import Sequence

    from opensearchpy import AsyncOpenSearch

    from src.clients.pe_encoder import PEEncoder
    from src.domain.region_boxes import RegionBox
    from src.services.curation.reprocess_gpu_embed import GpuEmbedRoute

logger = get_logger(__name__)

ALL_PARTS = frozenset({'crop', 'frame', 'region'})
_BATCH = 32
_IMAGES_IN_FLIGHT = 16
"""Images embedded at once: enough crops in flight to fill an engine batch, few enough that
their decoded pixels stay small."""


@dataclass
class EmbedTarget:
    """One image and the items on it to re-embed."""

    image_id: str
    image_path: str
    items: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    """``(crop_id, _source)``; the source needs ``bbox_norm`` for the crop
    part, the region boxes (and, for ``only_missing``, the stored box
    embeddings' ids and geometry) for the region part, and ``class_id`` /
    ``cluster_id`` / ``crop_rank_in_image`` / ``blur_lap_ratio`` for the
    cluster placement of an item that had no vector."""
    skip_crop: frozenset[str] = frozenset()
    """Crop ids that already have a vector and keep it (``only_missing``)."""


def target_has_work(target: EmbedTarget, parts: frozenset[str], *, only_missing: bool) -> bool:
    """Whether embedding ``target`` writes anything: the one test behind the
    applied run's ``images`` count and the dry run's ``images_to_embed``."""
    if 'frame' in parts:
        return True
    if 'crop' in parts and any(
        s.get('bbox_norm') and cid not in target.skip_crop for cid, s in target.items
    ):
        return True
    return 'region' in parts and any(
        (missing_boxes(s) if only_missing else embeddable(read_boxes(s))) for _, s in target.items
    )


def _read_image(path: str) -> bytes | None:
    """The encoded file at ``path`` if it is a servable path, else ``None``
    (fail closed: an unservable stored path is never read)."""
    if not is_servable_image_path(path):
        return None
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        logger.warning('reembed_image_unreadable', path=path, error=str(exc))
        return None


def _decode_image(data: bytes, path: str) -> Image.Image | None:
    """The EXIF-transposed RGB image encoded in ``data``, ``None`` when undecodable."""
    try:
        with Image.open(io.BytesIO(data)) as raw:
            raw.load()
            img = ImageOps.exif_transpose(raw)
            return img if img.mode == 'RGB' else img.convert('RGB')
    except Exception as exc:
        logger.warning('reembed_image_unreadable', path=path, error=str(exc))
        return None


def _pixel_box(
    bbox_norm: Sequence[float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in bbox_norm)
    return (x1 * width, y1 * height, x2 * width, y2 * height)


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=92)
    return buf.getvalue()


def _crop_arrays(img: Image.Image, boxes: Sequence[Sequence[float]]) -> list[np.ndarray]:
    return [np.asarray(crop_pil(img, _pixel_box(box, *img.size))) for box in boxes]


def _crop_jpegs(img: Image.Image, boxes: Sequence[Sequence[float]]) -> list[bytes]:
    return [_jpeg(crop_pil(img, _pixel_box(box, *img.size))) for box in boxes]


@dataclass
class _Vectors:
    """One photo's vectors: the frame (``None`` when not asked), one per crop box and one per
    region box in the order asked (``None`` for a region crop the encoder could not decode)."""

    frame: np.ndarray | None
    crops: Sequence[np.ndarray] | np.ndarray
    regions: Sequence[Sequence[float] | np.ndarray | None]


async def _cpu_vectors(
    pe: PEEncoder | Any,
    img: Image.Image,
    crop_boxes: list[Sequence[float]],
    region_boxes: list[Sequence[float]],
    *,
    frame: bool,
) -> _Vectors:
    whole = await pe.embed_whole_frame_image(img) if frame else None
    crops = await run_cpu(_crop_arrays, img, crop_boxes) if crop_boxes else []
    jpegs = await run_cpu(_crop_jpegs, img, region_boxes) if region_boxes else []
    return _Vectors(
        frame=whole,
        crops=await pe.embed_crops(crops, max_batch=_BATCH) if crops else [],
        regions=await embed_region_crops(pe, jpegs) if jpegs else [],
    )


def _key(box: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(v) for v in box)


async def _gpu_vectors(
    gpu: GpuEmbedRoute,
    data: bytes,
    crop_boxes: list[Sequence[float]],
    region_boxes: list[Sequence[float]],
    *,
    frame: bool,
) -> _Vectors | None:
    """The GPU decode's vectors (a box asked twice, crop and region, is embedded once), or
    ``None`` when this photo takes the CPU path."""
    keys = list(dict.fromkeys(_key(b) for b in [*crop_boxes, *region_boxes]))
    found = await embed_on_gpu(gpu, data, keys)
    if found is None:
        return None
    row = {k: found.boxes[i] for i, k in enumerate(keys)}
    return _Vectors(
        frame=found.frame if frame else None,
        crops=[row[_key(b)] for b in crop_boxes],
        regions=[row[_key(b)] for b in region_boxes],
    )


async def _target_vectors(
    pe: PEEncoder | Any,
    gpu: GpuEmbedRoute | None,
    path: str,
    crop_boxes: list[Sequence[float]],
    region_boxes: list[Sequence[float]],
    *,
    frame: bool,
) -> _Vectors | None:
    """The vectors of the photo at ``path``, or ``None`` when it cannot be read or decoded."""
    data = await asyncio.to_thread(_read_image, path)
    if data is None:
        return None
    if gpu is not None:
        found = await _gpu_vectors(gpu, data, crop_boxes, region_boxes, frame=frame)
        if found is not None:
            return found
    img = await asyncio.to_thread(_decode_image, data, path)
    if img is None:
        return None
    return await _cpu_vectors(pe, img, crop_boxes, region_boxes, frame=frame)


async def _bulk_update(
    opensearch: AsyncOpenSearch, index: str, docs: list[tuple[str, dict[str, Any]]]
) -> None:
    for start in range(0, len(docs), 500):
        body: list[dict[str, Any]] = []
        for doc_id, fields in docs[start : start + 500]:
            body.append({'update': {'_index': index, '_id': doc_id}})
            body.append({'doc': fields})
        resp = await opensearch.bulk(body=body, refresh=False)
        if isinstance(resp, dict) and resp.get('errors'):
            raise RuntimeError(f'embedding bulk update failed: {str(resp.get("items"))[:300]}')


def _placement(store: Any, src: dict[str, Any], vector: Any) -> dict[str, Any]:
    """Cluster fields for an item that has neither a class nor a cluster yet (it
    was stored without a vector), placed by the rule ingest uses. Anything
    already clustered or classed is left where it is."""
    if store is None or src.get('class_id') is not None or src.get('cluster_id') is not None:
        return {}
    placed = residual_placement(
        store, vector, src.get('crop_rank_in_image'), src.get('blur_lap_ratio')
    )
    if placed is None:
        return {}
    cluster_id, distance = placed
    out: dict[str, Any] = {'cluster_id': cluster_id, 'cluster_distance': distance}
    if distance is not None:
        out['cluster_distance_cluster_id'] = cluster_id
    return out


@dataclass
class _Writes:
    """What a pass writes once every image is embedded, and its counters."""

    ivf_store: Any
    counts: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(
            (
                'images',
                'items',
                'crop_written',
                'frame_written',
                'region_written',
                'missing_image',
                'decode_failed',
            ),
            0,
        )
    )
    items: dict[str, dict[str, Any]] = field(default_factory=dict)
    images: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    boxes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def add(
        self,
        target: EmbedTarget,
        boxed: list[tuple[str, dict[str, Any]]],
        work: list[tuple[str, RegionBox]],
        vectors: _Vectors,
    ) -> None:
        self.counts['images'] += 1
        self.counts['items'] += len(target.items)
        if vectors.frame is not None:
            self.images.append((target.image_id, {'pe_embedding': float_list(vectors.frame)}))
            self.counts['frame_written'] += 1
        for (cid, src), vec in zip(boxed, vectors.crops, strict=True):
            update = self.items.setdefault(cid, {})
            update.update(pe_embedding=float_list(vec), embedding_state=EMBEDDED)
            update.update(_placement(self.ivf_store, src, vec))
            self.counts['crop_written'] += 1
        for (cid, box), region_vec in zip(work, vectors.regions, strict=True):
            if region_vec is None:
                self.counts['decode_failed'] += 1
                continue
            self.boxes.setdefault(cid, []).append(entry_for(box, region_vec))
            self.counts['region_written'] += 1


async def reembed_items(
    opensearch: AsyncOpenSearch,
    pe: PEEncoder | Any,
    targets: list[EmbedTarget],
    *,
    parts: frozenset[str] = ALL_PARTS,
    only_missing: bool = False,
    place_in_clusters: bool = True,
    gpu: GpuEmbedRoute | None = None,
) -> dict[str, int]:
    """Recompute the vectors named by ``parts`` (``crop``, ``frame``,
    ``region``) for every target; ``only_missing`` limits the region part to
    boxes with no valid stored vector; ``place_in_clusters`` false leaves the
    newly embedded crops out of the clusters; ``gpu`` sends photos at or above
    the ingest size gate to the GPU decode. Returns counters:
    ``images``, ``items``, ``crop_written``, ``frame_written``,
    ``region_written``, ``missing_image`` (path unservable or unreadable)
    and ``decode_failed`` (region crop the encoder could not decode).
    """
    cfg = get_curation_config()
    writes = _Writes(get_ivf_ingest_store() if 'crop' in parts and place_in_clusters else None)
    gate = asyncio.Semaphore(_IMAGES_IN_FLIGHT)

    async def embed_target(target: EmbedTarget) -> None:
        if not target_has_work(target, parts, only_missing=only_missing):
            return
        boxed = [
            (cid, s)
            for cid, s in target.items
            if 'crop' in parts and s.get('bbox_norm') and cid not in target.skip_crop
        ]
        work = [
            (cid, box)
            for cid, s in target.items
            if 'region' in parts
            for box in (missing_boxes(s) if only_missing else embeddable(read_boxes(s)))
        ]
        async with gate:
            vectors = await _target_vectors(
                pe,
                gpu,
                target.image_path,
                [s['bbox_norm'] for _, s in boxed],
                [box.bbox_norm for _, box in work],
                frame='frame' in parts,
            )
        if vectors is None:
            writes.counts['missing_image'] += 1
            return
        writes.add(target, boxed, work, vectors)

    # Images run side by side so the encoder client can pack their crops into shared engine
    # batches; a failure still ends the pass before anything is written, as it did serially.
    # The loop thread was the limit (#259): crop, resize and pack on the pool, whatever the size.
    with offload_always():
        outcomes = await asyncio.gather(*(embed_target(t) for t in targets), return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome

    await _bulk_update(opensearch, cfg.items_index, list(writes.items.items()))
    await _bulk_update(opensearch, cfg.images_index, writes.images)
    if writes.boxes:
        await write_box_embeddings(opensearch, index=cfg.items_index, by_crop=writes.boxes)
    return writes.counts


__all__ = ['ALL_PARTS', 'EmbedTarget', 'reembed_items', 'target_has_work']
