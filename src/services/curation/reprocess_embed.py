"""The ``embed`` scope (W10.13): recompute derived vectors.

Embeddings are derived data, not labels, so locked items are included. Per
item: the PE crop embedding (``pe_embedding``) and the item-level region
embedding (``RegionFields.embedding``, the best accepted -- else
false-positive -- box, the same representative the backfill script picks);
per image: the whole-frame PE embedding on the images doc. Only those
vector fields are written, with plain partial updates.

``reembed_items`` is the one function behind the ``embed`` reprocess scope
and ``scripts/curation/backfill_region_embeddings.py``.
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageOps

from src.config import get_curation_config
from src.config.region_fields import RegionFields, get_region_fields
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.curation.image_serving import is_servable_image_path
from src.services.curation.ingest_index import crop_pil
from src.services.curation.region_boxes import read_boxes
from src.services.detection.region_embed import embed_region_crops


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.clients.pe_encoder import PEEncoder

logger = get_logger(__name__)

ALL_PARTS = frozenset({'crop', 'frame', 'region'})
_BATCH = 32


@dataclass
class EmbedTarget:
    """One image and the items on it to re-embed."""

    image_id: str
    image_path: str
    items: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    """``(crop_id, _source)``; the source needs ``bbox_norm`` and the
    region boxes (``region_boxes``) for the region part."""


def best_region_bbox(source: dict[str, Any], F: RegionFields | None = None) -> list[float] | None:
    """The representative region box's ``bbox_norm``: the highest-``score``
    accepted box, else the highest-``score`` false-positive box, else
    ``None``. One item-level embedding needs exactly one crop."""
    F = F or get_region_fields()
    boxes = read_boxes(source, F)
    candidates = [b for b in boxes if b.state == 'accepted']
    if not candidates:
        candidates = [b for b in boxes if b.state == RegionStatus.FALSE_POSITIVE.value]
    if not candidates:
        return None
    best = max(candidates, key=lambda b: b.score if b.score is not None else -1.0)
    return list(best.bbox_norm)


def _load_image(path: str) -> Image.Image | None:
    """The EXIF-transposed RGB image at ``path`` if it is a servable path,
    else ``None`` (fail closed: an unservable stored path is never read)."""
    if not is_servable_image_path(path):
        return None
    try:
        with Image.open(path) as raw:
            raw.load()
            img = ImageOps.exif_transpose(raw)
            return img if img.mode == 'RGB' else img.convert('RGB')
    except Exception as exc:
        logger.warning('reembed_image_unreadable', path=path, error=str(exc))
        return None


def _pixel_box(
    bbox_norm: list[float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in bbox_norm)
    return (x1 * width, y1 * height, x2 * width, y2 * height)


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=92)
    return buf.getvalue()


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


async def reembed_items(
    opensearch: AsyncOpenSearch,
    pe: PEEncoder | Any,
    targets: list[EmbedTarget],
    *,
    parts: frozenset[str] = ALL_PARTS,
) -> dict[str, int]:
    """Recompute the vectors named by ``parts`` (``crop``, ``frame``,
    ``region``) for every target. Returns counters:
    ``images``, ``items``, ``crop_written``, ``frame_written``,
    ``region_written``, ``missing_image`` (path unservable or unreadable)
    and ``decode_failed`` (region crop the encoder could not decode).
    """
    cfg = get_curation_config()
    F = get_region_fields()
    counts = {
        'images': 0,
        'items': 0,
        'crop_written': 0,
        'frame_written': 0,
        'region_written': 0,
        'missing_image': 0,
        'decode_failed': 0,
    }
    item_updates: dict[str, dict[str, Any]] = {}
    image_updates: list[tuple[str, dict[str, Any]]] = []

    for target in targets:
        img = await asyncio.to_thread(_load_image, target.image_path)
        if img is None:
            counts['missing_image'] += 1
            continue
        counts['images'] += 1
        counts['items'] += len(target.items)
        if 'frame' in parts:
            frame = await pe.embed_whole_frame(target.image_path)
            if frame is not None:
                image_updates.append((target.image_id, {'pe_embedding': [float(v) for v in frame]}))
                counts['frame_written'] += 1
        if 'crop' in parts:
            boxed = [(cid, s) for cid, s in target.items if s.get('bbox_norm')]
            crops = [
                np.asarray(crop_pil(img, _pixel_box(s['bbox_norm'], *img.size))) for _, s in boxed
            ]
            if crops:
                vectors = await pe.embed_crops(crops, max_batch=_BATCH)
                for (cid, _), vec in zip(boxed, vectors, strict=True):
                    item_updates.setdefault(cid, {})['pe_embedding'] = [float(v) for v in vec]
                    counts['crop_written'] += 1
        if 'region' in parts:
            ids: list[str] = []
            jpegs: list[bytes] = []
            for cid, s in target.items:
                bbox = best_region_bbox(s, F)
                if bbox is None or len(bbox) != 4:
                    continue
                ids.append(cid)
                jpegs.append(_jpeg(crop_pil(img, _pixel_box(bbox, *img.size))))
            if jpegs:
                for cid, vec in zip(ids, await embed_region_crops(pe, jpegs), strict=True):
                    if vec is None:
                        counts['decode_failed'] += 1
                        continue
                    item_updates.setdefault(cid, {})[F.embedding] = vec
                    counts['region_written'] += 1

    await _bulk_update(opensearch, cfg.items_index, list(item_updates.items()))
    await _bulk_update(opensearch, cfg.images_index, image_updates)
    return counts


__all__ = ['ALL_PARTS', 'EmbedTarget', 'best_region_bbox', 'reembed_items']
