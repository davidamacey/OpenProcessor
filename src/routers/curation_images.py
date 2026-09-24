"""Curation image-serving router.

Exposes:

- ``GET {prefix}/images/serve``                — stream a source JPEG (auto root detection)
- ``GET {prefix}/images/root/{alias}``          — stream an image from a named source-path alias
- ``GET {prefix}/images/cache/stats``           — thumbnail cache hit/miss stats
- ``GET {prefix}/crops/{id}/thumbnail``         — 128px item-crop thumbnail (LRU cached)
- ``GET {prefix}/crops/{id}/image``             — full source image with bbox overlay(s)
- ``GET {prefix}/crops/{id}/region_thumbnail``  — region-of-interest sub-bbox thumbnail

Exports **two** routers — ``router`` (images) and ``crops_router``
(crops) — mirroring the reference implementation, which registers them
separately in ``src/main.py`` rather than nesting one under the other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, ORJSONResponse, Response

from src.config import get_curation_config, get_region_fields
from src.core.dependencies import get_opensearch
from src.core.logging import get_logger
from src.services.curation.image_serving import (
    THUMBNAIL_CACHE,
    _fetch_crop,
    render_image_with_bbox,
    render_image_with_multiple_bboxes,
    resolve_crop_root,
    resolve_safe_path,
    serve_source_image,
)


if TYPE_CHECKING:
    from pathlib import Path


async def _raw_opensearch_dep() -> Any:
    """Unwrap the OpenSearchClient wrapper to the underlying AsyncOpenSearch.

    ``image_serving._fetch_crop`` calls ``client.get(...)`` directly, which
    only exists on the raw async client. The default ``OpenSearchDep``
    yields the wrapper, so we unwrap it here.
    """
    wrapper = await get_opensearch()
    return getattr(wrapper, 'client', wrapper)


OpenSearchDep = Annotated[Any, Depends(_raw_opensearch_dep)]


logger = get_logger(__name__)

config = get_curation_config()


router = APIRouter(
    prefix=f'{config.api_prefix}/images',
    tags=[f'{config.api_tag} - Images'],
    default_response_class=ORJSONResponse,
)


# Shared headers for image responses — 1h cache, public.
_IMAGE_CACHE_HEADERS = {'Cache-Control': 'public, max-age=3600'}


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
    prefix=f'{config.api_prefix}/crops',
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
    """Source image with the item bbox drawn.

    Defaults to full resolution to preserve compatibility with callers
    that need pixel-accurate frames. Pass ``max_dim`` for the labeler
    review queue, where the user is glancing at one bbox at a time and
    a sub-MB preview is plenty. If the item also carries a
    region-of-interest sub-bbox (``RegionFields.bbox_norm``), it is
    drawn as a second overlay.
    """
    crop = await _fetch_crop(crop_id, opensearch)
    bbox_norm = crop.get('bbox_norm')
    if not bbox_norm or len(bbox_norm) != 4:
        raise HTTPException(status_code=500, detail='crop has invalid bbox_norm')

    image_path = _resolve_image_for_crop(crop)
    region_bbox = crop.get(get_region_fields().bbox_norm)

    try:
        # If a region-of-interest bbox is also present, draw both for
        # richer context.
        if region_bbox and len(region_bbox) == 4:
            jpeg = await render_image_with_multiple_bboxes(
                image_path,
                [
                    {
                        'bbox_norm': bbox_norm,
                        'class_name': crop.get('class_name', 'item'),
                        'color': (255, 80, 80),
                    },
                    {
                        'bbox_norm': region_bbox,
                        'class_name': 'region',
                        'color': (80, 200, 255),
                    },
                ],
                max_dim=max_dim,
            )
        else:
            jpeg = await render_image_with_bbox(image_path, list(bbox_norm), max_dim=max_dim)
    except Exception as exc:
        logger.error('crop_full_image_failed', crop_id=crop_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f'overlay render failed: {exc}') from exc

    return Response(content=jpeg, media_type='image/jpeg', headers=_IMAGE_CACHE_HEADERS)


@crops_router.get('/{crop_id}/region_thumbnail')
async def crop_region_thumbnail(
    crop_id: str,
    opensearch: OpenSearchDep,
    size: Annotated[int, Query(ge=32, le=512, description='Square thumbnail size')] = 128,
) -> Response:
    """128px JPEG thumbnail of the crop's region-of-interest sub-bbox.

    Used by the region-verification UI in the labeler. Falls back to the
    verifier-rejected candidate box (``RegionFields.candidate_bbox_norm``)
    when there is no accepted region box (``RegionFields.bbox_norm``) --
    a ``verify_rejected`` item never has the latter, so this route used to
    404 for every one of them even though the item is still reviewable
    (DQ-B2 follow-up). 404 only when the crop has neither box. The cache
    key includes the box's own coordinates (``ThumbnailCache.get_or_compute``),
    so a later promotion or re-detection that changes the box never
    serves a stale image -- it's a different cache key.
    """
    crop = await _fetch_crop(crop_id, opensearch)
    fields = get_region_fields()
    region_bbox = crop.get(fields.bbox_norm)
    if not region_bbox or len(region_bbox) != 4:
        region_bbox = crop.get(fields.candidate_bbox_norm)
    if not region_bbox or len(region_bbox) != 4:
        raise HTTPException(status_code=404, detail='crop has no region bbox')

    image_path = _resolve_image_for_crop(crop)

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

    return Response(content=jpeg, media_type='image/jpeg', headers=_IMAGE_CACHE_HEADERS)


__all__ = ['crops_router', 'router']
