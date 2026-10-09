"""Additive ``PUT _mapping`` migrations for the items and labels_confirmed
indexes: label, provenance, region, text and embedding fields."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch.base import (
    F,
    _is_recoverable_mapping_conflict,
    _knn_field,
    config,
    logger,
)
from src.clients.curation_opensearch.bodies_core import (
    _CLASS_HISTORY_MAPPING,
    _EXCLUSION_MAPPING,
    CLUSTER_GEOMETRY_MAPPING,
    _region_auto_confirm_mapping,
    _region_boxes_mapping,
)
from src.clients.curation_opensearch.bodies_other import LABELS_CONFIRMED_EXTRA_MAPPING
from src.config import BACKBONE_EMBEDDING_FIELD
from src.services.curation.item_text import ITEM_TEXT_MAPPING
from src.services.curation.vlm_class_attempt import VLM_CLASS_ATTEMPT_MAPPING


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


async def ensure_items_vlm_raw_label_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the ``vlm_raw_label`` + ``vlm_raw_label_conf`` fields onto the
    existing items mapping.

    OpenSearch ``PUT <index>/_mapping`` is idempotent for additive field
    changes — running it repeatedly is a no-op once the fields exist. This
    helper exists so deployments don't have to drop / recreate the index just
    to start capturing the VLM's raw output.

    Returns:
        Dict with ``acknowledged`` (bool from OpenSearch) plus ``index`` and
        ``fields_added`` (the field names this call attempted to add).
    """
    index = config.items_index
    body = {
        'properties': {
            'vlm_raw_label': {'type': 'keyword'},
            'vlm_raw_label_conf': {'type': 'float'},
            'vlm_verify_completed_at': {'type': 'date'},
            **VLM_CLASS_ATTEMPT_MAPPING,
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info(
            'curation_mapping_migration',
            index=index,
            fields=list(body['properties'].keys()),
            acknowledged=ack,
        )
        return {
            'acknowledged': ack,
            'index': index,
            'fields_added': list(body['properties'].keys()),
        }
    except Exception as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {
            'acknowledged': False,
            'index': index,
            'fields_added': list(body['properties'].keys()),
            'error': msg,
        }


async def ensure_items_label_cluster_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the ``vlm_label_cluster_*`` fields onto the existing items mapping.

    Why a dedicated helper: ``PUT <index>/_mapping`` is idempotent for
    additive field changes, so this can run on every cold start without
    triggering an index drop. A background clustering job calls this before
    writing back cluster ids to guarantee the fields exist on long-lived
    deployments created before this migration shipped.

    Returns:
        Dict with ``acknowledged`` / ``index`` / ``fields_added`` (and
        ``error`` on failure).
    """
    index = config.items_index
    body = {
        'properties': {
            'vlm_label_cluster_id': {'type': 'integer'},
            'vlm_label_cluster_name': {'type': 'keyword'},
            'vlm_label_cluster_distance': {'type': 'float'},
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info(
            'curation_mapping_migration',
            index=index,
            fields=list(body['properties'].keys()),
            acknowledged=ack,
        )
        return {
            'acknowledged': ack,
            'index': index,
            'fields_added': list(body['properties'].keys()),
        }
    except Exception as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {
            'acknowledged': False,
            'index': index,
            'fields_added': list(body['properties'].keys()),
            'error': msg,
        }


async def ensure_items_provenance_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the region + class provenance fields onto the existing items
    mapping.

    Additive ``PUT <index>/_mapping`` — idempotent: re-running on a mapping
    that already has the fields is a silent no-op. The list of fields here
    mirrors the entries added to :func:`_items_body`, so a freshly created
    index already has them and this call simply confirms. Adds for
    long-lived deployments that were created before this migration shipped.

    ``mapper_parsing_exception`` (raised on the rare case where a field
    name collides with an incompatible existing type — should never happen
    in production but is the documented failure mode) is swallowed with a
    log line; this helper never crashes the cold-start bootstrap.
    """
    index = config.items_index
    body = {
        'properties': {
            F.detector_chain: {'type': 'keyword'},
            F.detected_at: {'type': 'date'},
            F.verifier: {'type': 'keyword'},
            F.verifier_version: {'type': 'keyword'},
            F.verified_at: {'type': 'date'},
            F.rejection_reason: {'type': 'keyword'},
            F.status_legacy: {'type': 'keyword'},
            'class_detector': {'type': 'keyword'},
            'class_detector_version': {'type': 'keyword'},
            'class_labeler': {'type': 'keyword'},
            'class_labeled_at': {'type': 'date'},
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info(
            'curation_mapping_migration',
            index=index,
            fields=list(body['properties'].keys()),
            acknowledged=ack,
        )
        return {
            'acknowledged': ack,
            'index': index,
            'fields_added': list(body['properties'].keys()),
        }
    except Exception as exc:
        # OpenSearch's ``mapper_parsing_exception`` arrives wrapped in an
        # ``opensearchpy`` transport error; we don't import the client class
        # here so we string-match the error type. Either way, the failure
        # is logged and the bootstrap continues — additive mapping
        # collisions are non-fatal for the rest of the API surface.
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {
            'acknowledged': False,
            'index': index,
            'fields_added': list(body['properties'].keys()),
            'error': msg,
        }


async def ensure_items_validation_split_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the split ``class_validated`` + region ``validated`` boolean
    fields onto the existing items mapping.

    The conflated ``label_validated`` field is being split because a
    worker's region writes were silently un-validating human class labels
    by sharing the flag. The new fields are independent: a region-edit
    write touches only the region ``validated`` field, a class-edit write
    touches only ``class_validated``.

    Additive PUT — idempotent. The legacy ``label_validated`` field
    is preserved during the one-release deprecation window; new
    writers SHOULD NOT set it (the pre-commit guard blocks new writes).
    """
    index = config.items_index
    fields = ['class_validated', F.validated]
    body = {
        'properties': {
            'class_validated': {'type': 'boolean'},
            F.validated: {'type': 'boolean'},
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info('curation_mapping_migration', index=index, fields=fields, acknowledged=ack)
        return {
            'acknowledged': ack,
            'index': index,
            'fields_added': fields,
        }
    except Exception as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {
            'acknowledged': False,
            'index': index,
            'fields_added': fields,
            'error': msg,
        }


async def ensure_items_history_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the ``class_id_history`` object field onto the existing items
    mapping.

    Additive ``PUT <index>/_mapping`` — idempotent. Writers land in
    ``src/services/curation/history.py``; this helper exists so a
    re-ingested or migrated index has the field ready when the writers go
    live.

    A field's type can't change in place — an index built before this
    field went from ``nested`` to ``object enabled:false`` still has it
    mapped ``nested``, and OpenSearch would 400 on a conflicting
    ``put_mapping`` every cold start. No-op whenever the field is already
    present, regardless of its type; the type change itself only takes
    effect on a reindex.
    """
    index = config.items_index
    try:
        existing = await client.indices.get_mapping(index=index)
    except Exception as exc:
        logger.info('curation_mapping_precheck_failed', index=index, error=str(exc))
        existing = {}
    for mapping in (existing or {}).values():
        if 'class_id_history' in (mapping.get('mappings', {}).get('properties') or {}):
            return {
                'acknowledged': True,
                'index': index,
                'fields_added': [],
                'skipped': 'field_already_present',
            }
    body = {
        'properties': {
            'class_id_history': _CLASS_HISTORY_MAPPING,
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info(
            'curation_mapping_migration',
            index=index,
            fields=list(body['properties']),
            acknowledged=ack,
        )
        return {
            'acknowledged': ack,
            'index': index,
            'fields_added': list(body['properties']),
        }
    except Exception as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {
            'acknowledged': False,
            'index': index,
            'fields_added': list(body['properties']),
            'error': msg,
        }


async def ensure_items_exclusion_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the exclusion fields (:data:`_EXCLUSION_MAPPING`) onto the items mapping.

    One ``PUT _mapping`` per field: indexes created before these were
    mapped may already carry a dynamic ``text`` mapping for the string
    fields, and a conflict on one must not block the others.
    """
    index = config.items_index
    added: list[str] = []
    conflicts: list[str] = []
    for field, spec in _EXCLUSION_MAPPING.items():
        try:
            await client.indices.put_mapping(index=index, body={'properties': {field: spec}})
            added.append(field)
        except Exception as exc:
            msg = str(exc)
            if not _is_recoverable_mapping_conflict(msg):
                logger.error('curation_mapping_migration_failed', index=index, error=msg)
                return {'acknowledged': False, 'index': index, 'fields_added': added, 'error': msg}
            conflicts.append(field)
    logger.info(
        'curation_mapping_migration', index=index, fields=added, existing_conflicts=conflicts
    )
    return {'acknowledged': True, 'index': index, 'fields_added': added, 'conflicts': conflicts}


async def ensure_items_cluster_geometry_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT :data:`CLUSTER_GEOMETRY_MAPPING` onto the items mapping — one
    ``PUT _mapping`` per field, additive and idempotent."""
    index = config.items_index
    added: list[str] = []
    for field, spec in CLUSTER_GEOMETRY_MAPPING.items():
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


async def ensure_labels_confirmed_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT :data:`LABELS_CONFIRMED_EXTRA_MAPPING` onto the labels_confirmed
    mapping — one ``PUT _mapping`` per field, additive and idempotent."""
    index = config.labels_confirmed_index
    added: list[str] = []
    for field, spec in LABELS_CONFIRMED_EXTRA_MAPPING.items():
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


async def ensure_items_text_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the region auto-confirm flag and the item-text fields onto the
    items mapping.

    One ``PUT _mapping`` per field (as :func:`ensure_items_exclusion_fields`)
    so a dynamic mapping one of them already picked up on an older index
    cannot block the others. Additive and idempotent.
    """
    index = config.items_index
    added: list[str] = []
    conflicts: list[str] = []
    specs = {**_region_auto_confirm_mapping(), **ITEM_TEXT_MAPPING}
    for field, spec in specs.items():
        try:
            await client.indices.put_mapping(index=index, body={'properties': {field: spec}})
            added.append(field)
        except Exception as exc:
            msg = str(exc)
            if not _is_recoverable_mapping_conflict(msg):
                logger.error('curation_mapping_migration_failed', index=index, error=msg)
                return {'acknowledged': False, 'index': index, 'fields_added': added, 'error': msg}
            conflicts.append(field)
    logger.info(
        'curation_mapping_migration', index=index, fields=added, existing_conflicts=conflicts
    )
    return {'acknowledged': True, 'index': index, 'fields_added': added, 'conflicts': conflicts}


async def ensure_items_embedding_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the secondary + backbone embedding fields onto an existing items
    mapping.

    Additive ``PUT <index>/_mapping`` — idempotent.
    """
    index = config.items_index
    body = {
        'properties': {
            'pe_embedding': _knn_field(dim=config.encoder_embedding_dim),
            BACKBONE_EMBEDDING_FIELD: _knn_field(dim=config.backbone_embedding_dim),
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info(
            'curation_mapping_migration',
            index=index,
            fields=['pe_embedding', BACKBONE_EMBEDDING_FIELD],
            acknowledged=ack,
        )
        return {
            'acknowledged': ack,
            'index': index,
            'fields_added': ['pe_embedding', BACKBONE_EMBEDDING_FIELD],
        }
    except Exception as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {
            'acknowledged': False,
            'index': index,
            'fields_added': ['pe_embedding', BACKBONE_EMBEDDING_FIELD],
            'error': msg,
        }


async def ensure_items_region_boxes_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the multi-box region fields (``region_boxes``,
    ``region_box_embeddings`` and the item-level summary) onto an existing
    items mapping.

    One ``PUT _mapping`` per field (as :func:`ensure_items_exclusion_fields`)
    so a field the index already carries cannot block the others. Additive
    and idempotent; a type conflict on an existing field is recoverable
    (logged at info, see ``_is_recoverable_mapping_conflict``) because a
    field's type cannot change in place.
    """
    index = config.items_index
    added: list[str] = []
    conflicts: list[str] = []
    for field, spec in _region_boxes_mapping().items():
        try:
            await client.indices.put_mapping(index=index, body={'properties': {field: spec}})
            added.append(field)
        except Exception as exc:
            msg = str(exc)
            if not _is_recoverable_mapping_conflict(msg):
                logger.error('curation_mapping_migration_failed', index=index, error=msg)
                return {'acknowledged': False, 'index': index, 'fields_added': added, 'error': msg}
            conflicts.append(field)
    logger.info(
        'curation_mapping_migration', index=index, fields=added, existing_conflicts=conflicts
    )
    return {'acknowledged': True, 'index': index, 'fields_added': added, 'conflicts': conflicts}
