"""Curation image-serving router.

Exposes:

- ``GET {prefix}/images/{image_id}/file``       — clean source pixels of an image (no overlay)
- ``GET {prefix}/images/{image_id}/thumbnail``  — whole-frame thumbnail of an image (LRU cached)
- ``GET {prefix}/images/cache/stats``           — thumbnail cache hit/miss stats
- ``GET {prefix}/crops/{id}/thumbnail``         — 128px item-crop thumbnail (LRU cached)
- ``GET {prefix}/crops/{id}/region_thumbnail``  — region-of-interest sub-bbox thumbnail

Exports **two** routers — ``router`` (images) and ``crops_router``
(crops), registered separately in ``src/app/factory.py`` rather than nesting one
under the other.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response

from src.config import get_curation_config, get_region_fields
from src.core.dependencies import get_curation_opensearch as _raw_opensearch_dep
from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.domain.region_boxes import read_boxes
from src.routers._errors import server_error
from src.services.curation.image_detail import image_source
from src.services.curation.image_serving import (
    THUMBNAIL_CACHE,
    _fetch_crop,
    render_source_image,
    resolve_crop_root,
    resolve_safe_path,
    source_missing_detail,
)


if TYPE_CHECKING:
    from pathlib import Path


OpenSearchDep = Annotated[Any, Depends(_raw_opensearch_dep)]


logger = get_logger(__name__)

config = get_curation_config()


router = APIRouter(
    prefix='/images',
    tags=[f'{config.api_tag} - Images'],
)


# Shared headers for image responses — 1h cache, public.
_IMAGE_CACHE_HEADERS = {'Cache-Control': 'public, max-age=3600'}
_WHOLE_FRAME = (0.0, 0.0, 1.0, 1.0)


def _region_thumbnail_headers(image_path: Any, bbox: Any, size: int) -> dict[str, str]:
    """Revalidate-every-time headers for a box close-up. The URL is only
    ``crop_id`` + ``box_id`` and a human can move the box, so a fixed
    max-age would show the old crop; the ETag names the image, geometry and
    size, so an unmoved box is a cheap 304 and a moved one is a new body."""
    tag = hashlib.sha1(
        f'{image_path}|{list(bbox)}|{size}'.encode(), usedforsecurity=False
    ).hexdigest()[:16]
    return {'Cache-Control': 'no-cache', 'ETag': f'"{tag}"'}


def resolve_or_source_missing(path: str, root: Path) -> Path:
    """``resolve_safe_path`` with a not-found answer turned into the typed
    ``source_missing`` body so the frontend can name the unreachable root."""
    try:
        return resolve_safe_path(path, root)
    except RequestRejectedError as exc:
        if exc.status_code == 404:
            raise HTTPException(status_code=404, detail=source_missing_detail(root, path)) from exc
        raise


async def _image_path(opensearch: Any, image_id: str) -> str:
    """The stored path of the bound project's image ``image_id`` (404 for any other id, so an
    image another project ingested is neither readable nor distinguishable from a missing one)."""
    src = await image_source(opensearch, image_id, ['image_path'])
    if src is None or not src.get('image_path'):
        raise HTTPException(status_code=404, detail=f'image not found: {image_id}')
    return str(src['image_path'])


def _source_missing(image_path: str) -> HTTPException:
    return HTTPException(
        status_code=404, detail=source_missing_detail(resolve_crop_root(image_path), image_path)
    )


@router.get('/{image_id}/file', response_class=Response)
async def image_file(
    image_id: str,
    opensearch: OpenSearchDep,
    max_dim: Annotated[
        int | None,
        Query(
            ge=128,
            le=8192,
            description=(
                'Cap the longest edge at this many pixels. Omit for full resolution; '
                'about 1280 suits a review view (a 2 MB photo becomes a few hundred KB).'
            ),
        ),
    ] = None,
) -> Response:
    """The clean source pixels of an image: EXIF orientation applied and an optional resize,
    nothing drawn on them. A client draws the boxes `GET .../images/{image_id}` serves in image
    coordinates. A file that is not reachable answers the typed `source_missing` 404."""
    image_path = await _image_path(opensearch, image_id)
    path = resolve_or_source_missing(image_path, resolve_crop_root(image_path))
    try:
        jpeg = await render_source_image(path, max_dim=max_dim)
    except FileNotFoundError as exc:
        raise _source_missing(image_path) from exc
    except Exception as exc:
        logger.error('image_file_failed', image_id=image_id, error=str(exc))
        raise server_error(500, 'image render failed', exc) from exc
    return Response(content=jpeg, media_type='image/jpeg', headers=_IMAGE_CACHE_HEADERS)


@router.get('/{image_id}/thumbnail', response_class=Response)
async def image_thumbnail(
    image_id: str,
    opensearch: OpenSearchDep,
    size: Annotated[
        int, Query(ge=32, le=512, description='Longest edge of the thumbnail, in pixels.')
    ] = 256,
) -> Response:
    """The whole frame, shrunk to fit `size` with its aspect kept (so a box in image coordinates
    lands on the same place), LRU cached."""
    image_path = await _image_path(opensearch, image_id)
    path = resolve_or_source_missing(image_path, resolve_crop_root(image_path))
    try:
        jpeg = await THUMBNAIL_CACHE.get_or_compute_offloaded(path, _WHOLE_FRAME, size=size)
    except FileNotFoundError as exc:
        raise _source_missing(image_path) from exc
    except Exception as exc:
        logger.error('image_thumbnail_failed', image_id=image_id, error=str(exc))
        raise server_error(500, 'thumbnail render failed', exc) from exc
    return Response(content=jpeg, media_type='image/jpeg', headers=_IMAGE_CACHE_HEADERS)


# =============================================================================
# /images/cache/stats
# =============================================================================


@router.get('/cache/stats')
async def thumbnail_cache_stats() -> dict[str, int]:
    """Return current thumbnail cache hit/miss counts and size."""
    return THUMBNAIL_CACHE.cache_info()


# =============================================================================
# /crops/* — separate router so it can mount under its own prefix
# =============================================================================


crops_router = APIRouter(
    prefix='/crops',
    tags=[f'{config.api_tag} - Crops'],
)


def _resolve_image_for_crop(crop_doc: dict[str, Any]) -> Path:
    """Resolve a crop's ``image_path`` to an absolute, validated Path."""
    image_path = crop_doc.get('image_path')
    if not image_path:
        raise HTTPException(status_code=500, detail='crop is missing image_path')
    root = resolve_crop_root(image_path)
    return resolve_or_source_missing(image_path, root)


def _source_missing_error(crop_doc: dict[str, Any]) -> HTTPException:
    """The typed 404 for a file that vanished between resolve and render."""
    image_path = str(crop_doc.get('image_path', ''))
    return HTTPException(
        status_code=404,
        detail=source_missing_detail(resolve_crop_root(image_path), image_path),
    )


@crops_router.get('/{crop_id}/thumbnail')
async def crop_thumbnail(
    crop_id: str,
    opensearch: OpenSearchDep,
    size: Annotated[int, Query(ge=32, le=512, description='Square thumbnail size')] = 128,
    v: Annotated[
        int | None,
        Query(description='The item `geometry_revision`: a cache key only, the server ignores it.'),
    ] = None,
) -> Response:
    """128px JPEG thumbnail of an item crop, LRU-cached."""
    del v  # a cache key for the browser; the server's cache is keyed by the box itself
    crop = await _fetch_crop(crop_id, opensearch)
    bbox_norm = crop.get('bbox_norm')
    if not bbox_norm or len(bbox_norm) != 4:
        raise HTTPException(status_code=500, detail='crop has invalid bbox_norm')

    image_path = _resolve_image_for_crop(crop)

    try:
        jpeg = await THUMBNAIL_CACHE.get_or_compute_offloaded(
            image_path, tuple(bbox_norm), size=size
        )
    except ValueError as exc:
        # Bad bbox / corrupt geometry → 400 (client visible).
        logger.warning('crop_thumbnail_invalid_bbox', crop_id=crop_id, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise _source_missing_error(crop) from exc
    except Exception as exc:
        logger.error('crop_thumbnail_failed', crop_id=crop_id, error=str(exc))
        raise server_error(500, 'thumbnail render failed', exc) from exc

    return Response(content=jpeg, media_type='image/jpeg', headers=_IMAGE_CACHE_HEADERS)


@crops_router.get('/{crop_id}/region_thumbnail')
async def crop_region_thumbnail(
    crop_id: str,
    request: Request,
    opensearch: OpenSearchDep,
    box_id: Annotated[str | None, Query(description='The region box to render (required).')] = None,
    size: Annotated[int, Query(ge=32, le=512, description='Square thumbnail size')] = 128,
) -> Response:
    """128px JPEG thumbnail of one region box of the crop.

    Used by the region-verification UI in the labeler. ``box_id`` names
    the box (422 ``box_id_required`` without it, 404 ``unknown_box_id``
    for a box the crop does not hold); any state renders, a rejected box
    included, since a rejected box is still reviewable. The cache key
    and the ``ETag`` include the box's own coordinates
    (``ThumbnailCache.get_or_compute``), so a later edit or re-detection
    that changes the box never serves a stale image; the response is
    ``no-cache`` and a matching ``If-None-Match`` gets a 304.
    """
    if not box_id:
        raise HTTPException(
            status_code=422, detail={'error': 'box_id_required', 'message': 'box_id is required'}
        )
    crop = await _fetch_crop(crop_id, opensearch)
    region_bbox = next(
        (b.bbox_norm for b in read_boxes(crop, get_region_fields()) if b.box_id == box_id), None
    )
    if region_bbox is None:
        raise HTTPException(
            status_code=404,
            detail={'error': 'unknown_box_id', 'message': f'the item has no box {box_id!r}'},
        )

    image_path = _resolve_image_for_crop(crop)
    headers = _region_thumbnail_headers(image_path, region_bbox, size)
    if request.headers.get('if-none-match') == headers['ETag']:
        return Response(status_code=304, headers=headers)

    try:
        jpeg = await THUMBNAIL_CACHE.get_or_compute_offloaded(
            image_path, tuple(region_bbox), size=size
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise _source_missing_error(crop) from exc
    except Exception as exc:
        logger.error('region_thumbnail_failed', crop_id=crop_id, error=str(exc))
        raise server_error(500, 'region thumbnail render failed', exc) from exc

    return Response(content=jpeg, media_type='image/jpeg', headers=headers)


__all__ = ['crops_router', 'router']
