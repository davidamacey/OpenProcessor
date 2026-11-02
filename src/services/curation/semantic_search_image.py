"""Query-by-picture semantic search: ``POST /curation/projects/{project}/search/image``.

The picture is decoded once (EXIF-upright RGB), optionally cut to a normalized ``box``, and
encoded through the PE-Core **image** tower with the *crop* preprocessing -- the one the indexed
item and box vectors were made with -- then ranked by the same filtered kNN page the text and
similar searches use. Nothing is written: not to disk, not to the index.

The picture is either the uploaded bytes or an image already in the bound project
(``image_id``, read from the stored path under the configured source roots).
"""

from __future__ import annotations

import io
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from opensearchpy.exceptions import NotFoundError
from PIL import Image

from src.config import CurationConfig, get_curation_config
from src.config.region_fields import RegionFields, get_region_fields
from src.core.logging import get_logger
from src.schemas.curation.search_target import SearchTarget
from src.services.curation.image_decode import ImageDecodeError, decode_image
from src.services.curation.semantic_search import _filter_for, _knn_page
from src.services.detection.geometry import crop_rect
from src.services.rejections import rejection
from src.utils.cpu_pool import run_cpu


if TYPE_CHECKING:
    from src.services.curation.item_filter import ItemFilter

logger = get_logger(__name__)

Box = tuple[float, float, float, float]


def parse_box(text: str | None) -> Box | None:
    """``"x1,y1,x2,y2"`` (normalized) as a box; 400 ``bad_box`` when it is not four finite
    numbers with ``0 <= x1 < x2 <= 1`` and ``0 <= y1 < y2 <= 1``."""
    if text is None or not text.strip():
        return None
    try:
        values = [float(p) for p in text.split(',')]
    except ValueError:
        values = []
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise rejection(400, 'bad_box', 'box must be four numbers: x1,y1,x2,y2 (normalized)')
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise rejection(400, 'bad_box', 'box must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1')
    return (x1, y1, x2, y2)


def _too_large(what: str, limit: str) -> Exception:
    return rejection(413, 'upload_too_large', f'{what} exceeds {limit}')


def _decode_checked(data: bytes, max_pixels: int) -> np.ndarray:
    """The upright RGB array of ``data``, refusing a picture over ``max_pixels`` from its header
    before the full decode."""
    try:
        with Image.open(io.BytesIO(data)) as header:
            width, height = header.size
    except Exception as exc:
        raise rejection(400, 'image_unreadable', 'not a decodable image') from exc
    if width * height > max_pixels:
        raise _too_large(f'{width}x{height} picture', f'{max_pixels} pixels')
    try:
        decoded = decode_image(data)
    except ImageDecodeError as exc:
        raise rejection(400, 'image_unreadable', str(exc)) from exc
    rgb = decoded.rgb if decoded.rgb is not None else np.asarray(decoded.pil)
    return np.ascontiguousarray(rgb)


def _crop(rgb: np.ndarray, box: Box | None) -> np.ndarray:
    if box is None:
        return rgb
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = box
    if (x2 - x1) * width < 1.0 or (y2 - y1) * height < 1.0:
        raise rejection(400, 'bad_box', 'box is smaller than one pixel of this picture')
    left, top, right, bottom = crop_rect(
        (x1 * width, y1 * height, x2 * width, y2 * height), width, height
    )
    return rgb[top:bottom, left:right]


def _prepare(data: bytes, box: Box | None, max_pixels: int) -> np.ndarray:
    return _crop(_decode_checked(data, max_pixels), box)


async def _stored_image_bytes(opensearch: Any, image_id: str, cfg: CurationConfig) -> bytes:
    """The bytes of project image ``image_id``: looked up in the bound project's images index,
    its stored path resolved under a configured source root (never a client-supplied path)."""
    from src.services.curation.crop_bytes import resolved_image_path

    try:
        resp = await opensearch.get(index=cfg.images_index, id=image_id)
    except Exception as exc:
        if not isinstance(exc, NotFoundError):
            raise
        resp = {'found': False}
    source = resp.get('_source') if resp.get('found', True) else None
    image_path = (source or {}).get('image_path')
    if not image_path:
        raise rejection(404, 'image_not_found', f"no image '{image_id}' in this project")
    path = resolved_image_path(image_path)
    try:
        size = path.stat().st_size
        if size > cfg.search_image_max_bytes:
            raise _too_large(f'image {image_id}', f'{cfg.search_image_max_bytes} bytes')
        data: bytes = await run_cpu(path.read_bytes, nbytes=size)
    except FileNotFoundError as exc:
        raise rejection(404, 'image_not_found', f"image '{image_id}' has no readable file") from exc
    return data


async def semantic_image_search(
    *,
    opensearch: Any,
    pe_encoder: Any,
    data: bytes | None,
    image_id: str | None,
    box: Box | None,
    page: int,
    page_size: int,
    target: SearchTarget = SearchTarget.ITEM,
    cluster_id: int | None = None,
    tab: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    max_rank: int | None = None,
    min_blur_ratio: float | None = None,
    hide_near_duplicates: bool = False,
    min_score: float | None = None,
    include_test: bool = False,
    item_filter: ItemFilter | None = None,
    config: CurationConfig | None = None,
    fields: RegionFields | None = None,
) -> dict[str, Any]:
    """Rank items (or, with ``target=box``, their boxes) by similarity to a picture.

    Exactly one of ``data`` (already size-checked bytes) and ``image_id`` names the picture. An
    encoder that cannot answer is 503 ``image_encoder_unavailable``, never an empty page.
    """
    if (data is None) == (image_id is None):
        raise rejection(400, 'invalid_request', 'send exactly one of image and image_id')
    cfg = config or get_curation_config()
    region_fields = fields or get_region_fields()

    if image_id is not None:
        data = await _stored_image_bytes(opensearch, image_id, cfg)
    assert data is not None
    crop = await run_cpu(_prepare, data, box, cfg.search_image_max_pixels, nbytes=len(data))

    if pe_encoder is None:
        raise rejection(503, 'image_encoder_unavailable', 'the PE image encoder is not loaded')
    try:
        vectors = await pe_encoder.embed_crops([crop])
    except Exception as exc:
        logger.warning('search_image_encode_failed', error=str(exc))
        raise rejection(503, 'image_encoder_unavailable', 'the PE image encoder failed') from exc

    filter_clause = _filter_for(
        item_filter=item_filter,
        tab=tab,
        cluster_id=cluster_id,
        date_from=date_from,
        date_to=date_to,
        max_rank=max_rank,
        min_blur_ratio=min_blur_ratio,
        hide_near_duplicates=hide_near_duplicates,
        include_test=include_test,
    )
    return await _knn_page(
        opensearch=opensearch,
        vector=vectors[0].tolist(),
        filter_clause=filter_clause,
        page=page,
        page_size=page_size,
        min_score=min_score,
        cfg=cfg,
        fields=region_fields,
        target=target,
    )
