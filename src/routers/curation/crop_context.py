"""Curation router sub-module — an item's source-image context."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from src.routers.curation._common import (
    CURATION_IMAGES_INDEX,
    CURATION_ITEMS_INDEX,
    OpenSearchDep,
    is_not_found,
    router,
)
from src.services.curation.wire import (
    item_list_source_excludes,
    item_source_excludes,
    serialize_item,
)


_MAX_SIBLINGS = 500


async def _get_source(opensearch: Any, index: str, doc_id: str, **kw: Any) -> dict[str, Any] | None:
    try:
        resp = await opensearch.get(index=index, id=doc_id, **kw)
    except Exception as exc:
        if is_not_found(exc):
            return None
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
    if not resp.get('found', True):
        return None
    return resp.get('_source') or {}


@router.get('/crops/{crop_id}/image')
async def crop_image_context(crop_id: str, opensearch: OpenSearchDep) -> dict[str, Any]:
    """The item's source image and every item detected in it.

    ``{image: {image_id, image_path, width, height, source, indexed_at} |
    null, items: [wire item, ...]}`` — items ordered by
    ``crop_rank_in_image`` (largest first), at most 500. ``image`` is null
    when the images index has no record of the frame.
    """
    # F-25 (adjacent fix): this endpoint's bare .get() had zero _source
    # excludes at all -- not even the embedding vectors every other item
    # endpoint drops. Bring it in line: exclude vectors like a
    # single-item fetch (item_source_excludes), and the sibling list
    # below additionally drops class_id_history (item_list_source_excludes
    # -- undo-only, no list renderer reads it).
    item = await _get_source(
        opensearch, CURATION_ITEMS_INDEX, crop_id, _source_excludes=item_source_excludes()
    )
    if item is None:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}')
    image_id = item.get('image_id')
    image: dict[str, Any] | None = None
    items: list[dict[str, Any]] = [serialize_item(item, crop_id)]
    if image_id:
        img = await _get_source(opensearch, CURATION_IMAGES_INDEX, image_id)
        if img is not None:
            image = {
                'image_id': image_id,
                'image_path': img.get('image_path'),
                'width': img.get('width'),
                'height': img.get('height'),
                'source': img.get('hdd_source') or img.get('source') or '',
                'indexed_at': img.get('indexed_at'),
            }
        body = {
            'size': _MAX_SIBLINGS,
            'query': {'term': {'image_id': image_id}},
            'sort': [{'crop_rank_in_image': {'order': 'asc', 'missing': '_last'}}],
            '_source': {'excludes': item_list_source_excludes()},
        }
        try:
            resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
        hits = (resp.get('hits') or {}).get('hits') or []
        items = [serialize_item(h.get('_source') or {}, h.get('_id', '')) for h in hits]
    return {'image': image, 'items': items}
