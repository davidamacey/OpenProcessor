"""Additive ``PUT _mapping`` migration for the SETTINGS-role index: the ``lease`` and
``class_registry`` object keys."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException

from src.clients.curation_opensearch.base import config, logger
from src.clients.curation_opensearch.bodies_other import SETTINGS_OBJECT_KEYS_MAPPING


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


async def ensure_settings_object_keys(client: AsyncOpenSearch) -> dict[str, Any]:
    """PUT the ``lease`` and ``class_registry`` object keys onto the project's configs
    index and, for a project whose stored resources predate the fold, onto its separate
    settings index (once: a folded project resolves SETTINGS to the configs index).

    Unmapped, a dynamic legacy settings index would map the fields of a lease document
    on its own. Additive ``PUT <index>/_mapping``, idempotent.
    """
    indexes = [config.configs_index]
    if config.settings_index not in indexes:
        indexes.append(config.settings_index)
    fields = list(SETTINGS_OBJECT_KEYS_MAPPING)
    for index in indexes:
        try:
            await client.indices.put_mapping(
                index=index, body={'properties': SETTINGS_OBJECT_KEYS_MAPPING}
            )
        except OpenSearchException as exc:
            msg = str(exc)
            logger.error('curation_mapping_migration_failed', index=index, error=msg)
            return {'acknowledged': False, 'index': index, 'fields_added': fields, 'error': msg}
    logger.info('curation_mapping_migration', indexes=indexes, fields=fields)
    return {'acknowledged': True, 'indexes': indexes, 'fields_added': fields}
