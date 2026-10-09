"""Items-index mapping fields added for label confirmation (#119), plus the
``ensure_*`` helpers that add the same fields to indexes created before they
existed (one ``PUT _mapping`` per field, additive and idempotent).

The constants are spread into the index bodies in ``bodies_core.py`` and
``bodies_other.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch.base import _is_recoverable_mapping_conflict, config, logger
from src.services.curation.audit_math import AUDIT_FIELDS


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


# The detector's own answer, written once at ingest
# (src/services/curation/item_doc.py DETECTOR_FIELDS).
DETECTOR_MAPPING: dict[str, Any] = {
    'detector_class_name': {'type': 'keyword'},
    'detector_class_id': {'type': 'integer'},
    'detector_confidence': {'type': 'float'},
}

# The accuracy audit's sample marker and verdict
# (src/services/curation/audit_math.py AUDIT_FIELDS).
AUDIT_MAPPING: dict[str, Any] = dict(AUDIT_FIELDS)

ITEMS_EXTRA_MAPPING: dict[str, Any] = {**DETECTOR_MAPPING, **AUDIT_MAPPING}

# Per-project policy documents inside the settings document: stored, never indexed
# (services/curation/ingest_policy_store.py, vlm_policy_store.py).
POLICY_DOC_MAPPING: dict[str, Any] = {
    'ingest_policy': {'type': 'object', 'enabled': False},
    'vlm_policy': {'type': 'object', 'enabled': False},
}


async def _put_fields(client: AsyncOpenSearch, mapping: dict[str, Any]) -> dict[str, Any]:
    index = config.items_index
    added: list[str] = []
    for field, spec in mapping.items():
        try:
            await client.indices.put_mapping(index=index, body={'properties': {field: spec}})
            added.append(field)
        except Exception as exc:
            msg = str(exc)
            if not _is_recoverable_mapping_conflict(msg):
                logger.error('curation_mapping_migration_failed', index=index, error=msg)
                return {'acknowledged': False, 'index': index, 'fields_added': added, 'error': msg}
    logger.info('curation_mapping_migration', index=index, fields=added)
    return {'acknowledged': True, 'index': index, 'fields_added': added}


async def ensure_items_detector_fields(client: AsyncOpenSearch) -> dict[str, Any]:
    """Map :data:`DETECTOR_MAPPING` onto the existing items index."""
    return await _put_fields(client, DETECTOR_MAPPING)


async def ensure_items_audit_fields(client: AsyncOpenSearch) -> dict[str, Any]:
    """Map :data:`AUDIT_MAPPING` onto the existing items index."""
    return await _put_fields(client, AUDIT_MAPPING)
