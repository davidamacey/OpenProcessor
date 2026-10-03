"""Index fields the open-vocabulary pass writes: the items' provenance and
outline, and the images' pass status.

Mapped explicitly (not left to dynamic mapping, which would index a name as
``text``) because reprocess and the review filters ``term``-query them.
Additive and idempotent; :func:`ensure_open_vocab_fields` migrates indexes
created before the pass existed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)

OPEN_VOCAB_ITEM_MAPPING: dict[str, Any] = {
    # The target prompt that found the item, and the set it belongs to.
    'source_prompt': {'type': 'keyword'},
    'open_vocab_set': {'type': 'keyword'},
    'open_vocab_revision': {'type': 'integer'},
    # Normalized outline in the source frame: stored, never indexed.
    'mask_polygon': {'type': 'object', 'enabled': False},
}

#: ``pending`` (queued by ingest, or the segmenter was down) | ``done`` | ``failed``.
OPEN_VOCAB_IMAGE_MAPPING: dict[str, Any] = {
    'open_vocab_status': {'type': 'keyword'},
}


async def _put_each(client: AsyncOpenSearch, index: str, specs: dict[str, Any]) -> None:
    from src.clients.curation_opensearch import _is_recoverable_mapping_conflict

    for field, spec in specs.items():
        try:
            await client.indices.put_mapping(index=index, body={'properties': {field: spec}})
        except Exception as exc:
            if not _is_recoverable_mapping_conflict(str(exc)):
                raise


async def ensure_open_vocab_fields(client: AsyncOpenSearch) -> None:
    """PUT the item and image fields onto the bound project's indexes. One
    ``PUT _mapping`` per field so a dynamic mapping an older index already
    picked up for one of them cannot block the others."""
    from src.config import get_curation_config

    cfg = get_curation_config()
    await _put_each(client, cfg.items_index, OPEN_VOCAB_ITEM_MAPPING)
    await _put_each(client, cfg.images_index, OPEN_VOCAB_IMAGE_MAPPING)
    logger.info('curation_mapping_migration', fields=[*OPEN_VOCAB_ITEM_MAPPING])


__all__ = ['OPEN_VOCAB_IMAGE_MAPPING', 'OPEN_VOCAB_ITEM_MAPPING', 'ensure_open_vocab_fields']
