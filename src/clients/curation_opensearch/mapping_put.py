"""The additive ``PUT <index>/_mapping`` calls the ``ensure_*`` migrations share: one
acknowledged result dict, with a recoverable field conflict logged at info and never raised."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException

from src.clients.curation_opensearch.base import _is_recoverable_mapping_conflict, logger


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


async def put_properties(
    client: AsyncOpenSearch, index: str, properties: dict[str, Any]
) -> dict[str, Any]:
    """One ``PUT _mapping`` of ``properties``. Returns ``acknowledged`` / ``index`` /
    ``fields_added`` (and ``error`` on an OpenSearch failure, which never raises)."""
    fields = list(properties)
    try:
        resp = await client.indices.put_mapping(index=index, body={'properties': properties})
        ack = bool(resp.get('acknowledged', False))
        logger.info('curation_mapping_migration', index=index, fields=fields, acknowledged=ack)
        return {'acknowledged': ack, 'index': index, 'fields_added': fields}
    except OpenSearchException as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {'acknowledged': False, 'index': index, 'fields_added': fields, 'error': msg}


async def put_properties_one_by_one(
    client: AsyncOpenSearch, index: str, properties: dict[str, Any]
) -> dict[str, Any]:
    """One ``PUT _mapping`` per field of ``properties``: indexes created before these
    were mapped may already carry a dynamic mapping for a field, and a conflict on one
    must not block the others."""
    added: list[str] = []
    conflicts: list[str] = []
    for field, spec in properties.items():
        try:
            await client.indices.put_mapping(index=index, body={'properties': {field: spec}})
            added.append(field)
        except OpenSearchException as exc:
            msg = str(exc)
            if not _is_recoverable_mapping_conflict(msg):
                logger.error('curation_mapping_migration_failed', index=index, error=msg)
                return {'acknowledged': False, 'index': index, 'fields_added': added, 'error': msg}
            conflicts.append(field)
    logger.info(
        'curation_mapping_migration', index=index, fields=added, existing_conflicts=conflicts
    )
    return {'acknowledged': True, 'index': index, 'fields_added': added, 'conflicts': conflicts}
