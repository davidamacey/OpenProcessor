"""The stored per-image item counts the images index carries (R-13).

``counts`` holds the five numbers an image card shows; ``counts_at`` says when they were last
recomputed from the items, and an image without it has never been counted. Only the drain
(``services/curation/image_counts_refresh``) writes them, always by recomputing from the items,
never by incrementing. They serve the gallery's sort and ``label_state`` filter, so those need no
join on the items index at 100,000+ images. Numeric fields are doc-values-backed sort and range
targets: https://docs.opensearch.org/latest/field-types/supported-field-types/numeric/
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)

COUNT_KEYS = ('items', 'labeled', 'validated', 'unlabeled', 'in_holdout')

IMAGE_COUNTS_MAPPING: dict[str, Any] = {
    'counts': {'properties': {key: {'type': 'integer'} for key in COUNT_KEYS}},
    'counts_at': {'type': 'date'},
}


async def ensure_images_count_fields(client: AsyncOpenSearch) -> dict[str, Any]:
    """PUT the count fields onto the bound project's existing images mapping (additive)."""
    from src.config.curation_accessors import images_index

    index = images_index()
    resp = await client.indices.put_mapping(index=index, body={'properties': IMAGE_COUNTS_MAPPING})
    ack = bool(resp.get('acknowledged', False))
    logger.info(
        'curation_mapping_migration',
        index=index,
        fields=list(IMAGE_COUNTS_MAPPING),
        acknowledged=ack,
    )
    return {'acknowledged': ack, 'index': index, 'fields_added': list(IMAGE_COUNTS_MAPPING)}


__all__ = ['COUNT_KEYS', 'IMAGE_COUNTS_MAPPING', 'ensure_images_count_fields']
