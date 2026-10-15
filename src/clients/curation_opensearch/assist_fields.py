"""The item field assist runs write: ``assist``, the VLM's suggestions for an
item (its class answer and the pack-declared attributes), stored apart from
the item's class and boxes so no export, training or holdout path reads it.

Mapped explicitly: ``attributes`` are ``nested`` so a ``key`` and its ``value``
stay paired. Additive and idempotent; :func:`ensure_assist_fields` migrates
indexes created before assist runs existed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)

_ATTRIBUTE_MAPPING: dict[str, Any] = {
    'type': 'nested',
    'properties': {
        'key': {'type': 'keyword'},
        'value': {'type': 'keyword'},
        'confidence': {'type': 'keyword'},
    },
}

ASSIST_ITEM_MAPPING: dict[str, Any] = {
    'assist': {
        'properties': {
            'run_id': {'type': 'keyword'},
            'run_created_at': {'type': 'date'},
            'vlm_endpoint': {'type': 'keyword'},
            'vlm_model': {'type': 'keyword'},
            'prompt_pack': {'type': 'keyword'},
            'prompt_pack_revision': {'type': 'integer'},
            'class_name': {'type': 'keyword'},
            'class_confidence': {'type': 'keyword'},
            'attributes': _ATTRIBUTE_MAPPING,
            'boxes': {
                'type': 'nested',
                'properties': {'box_id': {'type': 'keyword'}, 'attributes': _ATTRIBUTE_MAPPING},
            },
        }
    }
}


async def ensure_assist_fields(client: AsyncOpenSearch) -> None:
    """PUT the ``assist`` mapping onto the bound project's items index."""
    from src.clients.curation_opensearch.base import _is_recoverable_mapping_conflict
    from src.config import get_curation_config

    try:
        await client.indices.put_mapping(
            index=get_curation_config().items_index, body={'properties': ASSIST_ITEM_MAPPING}
        )
    except Exception as exc:
        if not _is_recoverable_mapping_conflict(str(exc)):
            raise
    logger.info('curation_mapping_migration', fields=[*ASSIST_ITEM_MAPPING])


__all__ = ['ASSIST_ITEM_MAPPING', 'ensure_assist_fields']
