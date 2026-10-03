"""Curation image-serving router.

Exposes:

- ``GET {prefix}/images/serve``                — stream a source JPEG (auto root detection)
- ``GET {prefix}/images/root/{alias}``          — stream an image from a named source-path alias
- ``GET {prefix}/images/cache/stats``           — thumbnail cache hit/miss stats
- ``GET {prefix}/crops/{id}/thumbnail``         — 128px item-crop thumbnail (LRU cached)
- ``GET {prefix}/crops/{id}/image``             — clean full source image (no overlay)
- ``GET {prefix}/crops/{id}/region_thumbnail``  — region-of-interest sub-bbox thumbnail

Exports **two** routers — ``router`` (images) and ``crops_router``
(crops), registered separately in ``src/main.py`` rather than nesting one
under the other.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, ORJSONResponse, Response

from src.config import get_curation_config, get_region_fields
from src.core.dependencies import get_curation_opensearch as _raw_opensearch_dep
from src.core.logging import get_logger
from src.services.curation.image_serving import (
    THUMBNAIL_CACHE,
    _fetch_crop,
    render_source_image,
    resolve_crop_root,
    resolve_safe_path,
    serve_source_image,
)
from src.services.curation.region_boxes import read_boxes


if TYPE_CHECKING:
    from pathlib import Path


OpenSearchDep = Annotated[Any, Depends(_raw_opensearch_dep)]


logger = get_logger(__name__)

config = get_curation_config()


router = APIRouter(
    prefix='/images',
    tags=[f'{config.api_tag} - Images'],
    default_response_class=ORJSONResponse,
)


# Shared headers for image responses — 1h cache, public.
_IMAGE_CACHE_HEADERS = {'Cache-Control': 'public, max-age=3600'}


def _region_thumbnail_headers(image_path: Any, bbox: Any, size: int) -> dict[str, str]:
    """Revalidate-every-time headers for a box close-up. The URL is only
    ``crop_id`` + ``box_id`` and a human can move the box, so a fixed
    max-age would show the old crop; the ETag names the image, geometry and
    size, so an unmoved box is a cheap 304 and a moved one is a new body."""
    tag = hashlib.sha1(
        f'{image_path}|{list(bbox)}|{size}'.encode(), usedforsecurity=False
    ).hexdigest()[:16]
    return {'Cache-Control': 'no-cache', 'ETag': f'"{tag}"'}


# =============================================================================
# /images/serve — auto-detect root
# =============================================================================


@router.get('/serve', response_class=FileResponse)
async def serve_image(
    path: Annotated[str, Query(description='Path relative to a configured source root')],
) -> FileResponse:
    """Stream a source image, auto-detecting its root.

    Root selection: the longest matching configured root for absolute
    paths, or a ``source_path_aliases`` prefix match on the leading path
    segment for relative paths — falls back to ``source_root``. See
    ``src.services.curation.image_serving.resolve_crop_root``.
    """
    root = resolve_crop_root(path)
    return await serve_source_image(path, root)


@router.get('/root/{alias}', response_class=FileResponse)
async def serve_aliased_image(
    alias: str,
    path: Annotated[str, Query(description='Path relative to the named alias root')],
) -> FileResponse:
    """Stream an image from an explicitly-named ``source_path_aliases`` root."""
    alias_root = config.source_path_aliases.get(alias)
    if alias_root is None:
        raise HTTPException(status_code=404, detail=f'unknown source root alias: {alias}')
    return await serve_source_image(path, alias_root)


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
    default_response_class=ORJSONResponse,
)


def _resolve_image_for_crop(crop_doc: dict[str, Any]) -> Path:
    """Resolve a crop's ``image_path`` to an absolute, validated Path."""
    image_path = crop_doc.get('image_path')
    if not image_path:
        raise HTTPException(status_code=500, detail='crop is missing image_path')
    root = resolve_crop_root(image_path)
    return resolve_safe_path(image_path, root)


@crops_router.get('/{crop_id}/thumbnail')
async def crop_thumbnail(
    crop_id: str,
    opensearch: OpenSearchDep,
    size: Annotated[int, Query(ge=32, le=512, description='Square thumbnail size')] = 128,
) -> Response:
    """128px JPEG thumbnail of an item crop, LRU-cached."""
    crop = await _fetch_crop(crop_id, opensearch)
    bbox_norm = crop.get('bbox_norm')
    if not bbox_norm or len(bbox_norm) != 4:
        raise HTTPException(status_code=500, detail='crop has invalid bbox_norm')

    image_path = _resolve_image_for_crop(crop)

    try:
        jpeg = THUMBNAIL_CACHE.get_or_compute(image_path, tuple(bbox_norm), size=size)
    except ValueError as exc:
        # Bad bbox / corrupt geometry → 400 (client visible).
        logger.warning('crop_thumbnail_invalid_bbox', crop_id=crop_id, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f'image missing: {exc}') from exc
    except Exception as exc:
        logger.error('crop_thumbnail_failed', crop_id=crop_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f'thumbnail render failed: {exc}') from exc

    return Response(content=jpeg, media_type='image/jpeg', headers=_IMAGE_CACHE_HEADERS)


@crops_router.get('/{crop_id}/image')
async def crop_full_image(
    crop_id: str,
    opensearch: OpenSearchDep,
    max_dim: Annotated[
        int | None,
        Query(
            ge=128,
            le=8192,
            description=(
                'Cap longest edge at this many pixels. Omit for full '
                'resolution; pass ~1024 for review-friendly previews '
                '(drops ~2.2 MB sources to ~200 KB).'
            ),
        ),
    ] = None,
) -> Response:
    """The clean source image for this crop's parent frame — no box, label
    or other overlay drawn. Only resize, EXIF-transpose and RGB
    conversion apply.

    Defaults to full resolution to preserve compatibility with callers
    that need pixel-accurate frames. Pass ``max_dim`` for the labeler
    review queue, where the user is glancing at one item at a time and a
    sub-MB preview is plenty. The frontend draws the item box, the
    region box and any rejected-candidate box itself, from the geometry
    ``GET {prefix}/crops/{id}/context`` serves in source-image-normalized
    coordinates.
    """
    crop = await _fetch_crop(crop_id, opensearch)
    image_path = _resolve_image_for_crop(crop)

    try:
        jpeg = await render_source_image(image_path, max_dim=max_dim)
    except Exception as exc:
        logger.error('crop_full_image_failed', crop_id=crop_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f'image render failed: {exc}') from exc

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
        jpeg = THUMBNAIL_CACHE.get_or_compute(image_path, tuple(region_bbox), size=size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f'image missing: {exc}') from exc
    except Exception as exc:
        logger.error('region_thumbnail_failed', crop_id=crop_id, error=str(exc))
        raise HTTPException(
            status_code=500, detail=f'region thumbnail render failed: {exc}'
        ) from exc

    return Response(content=jpeg, media_type='image/jpeg', headers=headers)


__all__ = ['crops_router', 'router']
