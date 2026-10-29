"""Additive ``PUT _mapping`` migrations for the items and images overlay
fields (quality, scores, probes, viz, upload ids) and the inner-result window."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch.base import _is_recoverable_mapping_conflict, config, logger
from src.clients.curation_opensearch.bodies_core import IMAGES_WRITE_ID_MAPPING, ITEMS_VIZ_MAPPING


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


_INNER_RESULT_WINDOWS: dict[str, int] = {}


def inner_result_window(index: str) -> int | None:
    """The ``index.max_inner_result_window`` :func:`ensure_items_inner_result_window`
    last read back for ``index`` (``None`` before it ran in this process)."""
    return _INNER_RESULT_WINDOWS.get(index)


async def ensure_items_inner_result_window(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """Raise ``index.max_inner_result_window`` on the items index to at least
    ``region_max_boxes_per_write``.

    The region list routes ask a nested query for ``inner_hits`` (which
    boxes of an item matched) sized to the most boxes one item may hold;
    OpenSearch rejects an ``inner_hits.size`` above this dynamic setting
    (default 100). Never lowers it, and issues no write when the current
    value already suffices. The effective value is remembered per index
    (:func:`inner_result_window`) so a reader can clamp to it.
    """
    index = config.items_index
    wanted = int(config.region_max_boxes_per_write)
    try:
        # No `name=`: the project guard allowlists `GET /{index}/_settings` only.
        resp = await client.indices.get_settings(index=index)
        per_index: dict[str, Any] = next(iter(resp.values()), {})
        if not adopt_inner_result_window(index, per_index.get('settings', {})):
            await client.indices.put_settings(
                index=index, body={'index': {'max_inner_result_window': wanted}}
            )
            _INNER_RESULT_WINDOWS[index] = wanted
    except Exception as exc:
        logger.warning('curation_inner_result_window_failed', index=index, error=str(exc))
        return {'acknowledged': False, 'index': index, 'error': str(exc)}
    return {
        'acknowledged': True,
        'index': index,
        'max_inner_result_window': _INNER_RESULT_WINDOWS[index],
    }


def adopt_inner_result_window(index: str, settings: dict[str, Any]) -> bool:
    """Record the ``index.max_inner_result_window`` a settings read of ``index`` shows
    (``settings`` is that index's ``settings`` object); True when it already reaches
    ``region_max_boxes_per_write``, False (nothing recorded) when it must be raised."""
    current = int(settings.get('index', {}).get('max_inner_result_window', 100))
    if current < int(config.region_max_boxes_per_write):
        return False
    _INNER_RESULT_WINDOWS[index] = current
    return True


async def ensure_images_upload_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the BA-1/BA-4 ``source_identifier`` / ``ingest_run_id`` keyword
    fields and the ingest write id onto the existing images mapping.

    Additive ``PUT <index>/_mapping`` — idempotent.
    """
    index = config.images_index
    body = {
        'properties': {
            'source_identifier': {'type': 'keyword'},
            'ingest_run_id': {'type': 'keyword'},
            **IMAGES_WRITE_ID_MAPPING,
        }
    }
    fields = list(body['properties'])
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info(
            'curation_mapping_migration',
            index=index,
            fields=fields,
            acknowledged=ack,
        )
        return {'acknowledged': ack, 'index': index, 'fields_added': fields}
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


async def ensure_items_request_id_field(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the ``request_id`` keyword field onto the existing items mapping.

    Additive ``PUT <index>/_mapping`` — idempotent.
    """
    index = config.items_index
    body = {'properties': {'request_id': {'type': 'keyword'}}}
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info(
            'curation_mapping_migration',
            index=index,
            fields=['request_id'],
            acknowledged=ack,
        )
        return {'acknowledged': ack, 'index': index, 'fields_added': ['request_id']}
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
            'fields_added': ['request_id'],
            'error': msg,
        }


async def ensure_items_quality_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the primary-subject rank + blur quality fields onto the existing
    items mapping.

    Adds ``crop_area_norm``, ``crop_rank_in_image``, ``blur_lap_var``,
    ``blur_lap_ratio`` and ``blur_full_var``. Additive
    ``PUT <index>/_mapping`` — idempotent. Mirrors the entries in
    :func:`_items_body`; legacy items are populated by backfill scripts.
    """
    index = config.items_index
    body = {
        'properties': {
            'crop_area_norm': {'type': 'float'},
            'crop_rank_in_image': {'type': 'byte'},
            'blur_lap_var': {'type': 'float'},
            'blur_lap_ratio': {'type': 'float'},
            'blur_full_var': {'type': 'float'},
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


async def ensure_items_score_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the curation-scoring overlay fields onto the existing items
    mapping.

    Adds the ``uniqueness_*`` (k-NN density/typicality), ``mistakenness_*``
    (confident-learning margin), and ``dup_*`` (item-level near-duplicate
    grouping) field families. Every field carries a ``{value, method,
    version, scored_at}``-shaped provenance quad so partial re-scores and
    mixed-version pools are visible via ``GET /curation/scores/coverage``.

    Purely additive — these fields are written **only** by the dedicated
    scoring job (``src/services/curation/item_scores/``) and never by the
    production clustering pipeline. No overlay/scorer writes ``cluster_id``
    / ``cluster_subid`` / ``cluster_distance``.

    Additive ``PUT <index>/_mapping`` — idempotent.
    """
    index = config.items_index
    fields = [
        'uniqueness_score',
        'uniqueness_method',
        'uniqueness_version',
        'uniqueness_scored_at',
        'mistakenness_score',
        'mistakenness_method',
        'mistakenness_version',
        'mistakenness_scored_at',
        'dup_group_id',
        'dup_group_size',
        'dup_is_representative',
        'dup_threshold',
        'dup_method',
        'dup_scored_at',
    ]
    body = {
        'properties': {
            'uniqueness_score': {'type': 'float'},
            'uniqueness_method': {'type': 'keyword'},
            'uniqueness_version': {'type': 'keyword'},
            'uniqueness_scored_at': {'type': 'date'},
            'mistakenness_score': {'type': 'float'},
            'mistakenness_method': {'type': 'keyword'},
            'mistakenness_version': {'type': 'keyword'},
            'mistakenness_scored_at': {'type': 'date'},
            'dup_group_id': {'type': 'keyword'},
            'dup_group_size': {'type': 'integer'},
            'dup_is_representative': {'type': 'boolean'},
            'dup_threshold': {'type': 'float'},
            'dup_method': {'type': 'keyword'},
            'dup_scored_at': {'type': 'date'},
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info('curation_mapping_migration', index=index, fields=fields, acknowledged=ack)
        return {'acknowledged': ack, 'index': index, 'fields_added': fields}
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
        return {'acknowledged': False, 'index': index, 'fields_added': fields, 'error': msg}


async def ensure_items_probe_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the probe-model provenance fields onto the existing items
    mapping.

    ``probe_pred_class`` and ``probe_pred_entropy`` are already declared in
    :func:`_items_body`. ``probe_pred_confidence`` and
    ``probe_disagreement`` are written by
    :func:`src.services.curation.probe_predictions.run_probe_inference` —
    this migration declares them explicitly alongside
    ``probe_pred_margin`` (``p(top1) - p(top2)`` from the real per-class
    posterior) and provenance (``probe_model_version`` /
    ``probe_scored_at``).

    Additive ``PUT <index>/_mapping`` — idempotent.
    """
    index = config.items_index
    fields = [
        'probe_pred_class_id',
        'probe_pred_confidence',
        'probe_disagreement',
        'probe_pred_margin',
        'probe_model_version',
        'probe_scored_at',
    ]
    body = {
        'properties': {
            'probe_pred_class_id': {'type': 'integer'},
            'probe_pred_confidence': {'type': 'float'},
            'probe_disagreement': {'type': 'boolean'},
            'probe_pred_margin': {'type': 'float'},
            'probe_model_version': {'type': 'keyword'},
            'probe_scored_at': {'type': 'date'},
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info('curation_mapping_migration', index=index, fields=fields, acknowledged=ack)
        return {'acknowledged': ack, 'index': index, 'fields_added': fields}
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
        return {'acknowledged': False, 'index': index, 'fields_added': fields, 'error': msg}


async def ensure_items_viz_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the UMAP-visualization overlay fields onto the existing items
    mapping.

    ``viz_x`` / ``viz_y`` are the cached 2-d coordinates from the
    visualization-only UMAP projection
    (:mod:`src.services.curation.embedding_viz`); ``viz_projection_version``
    stamps which fitted projection produced them so a partial re-fit (e.g.
    a cluster-scoped rebuild that only touches some items) is visible the
    same way the score-fields ``*_version`` fields make partial re-scores
    visible.

    **Purely additive and purely cosmetic** — these three fields are
    written **only** by :mod:`embedding_viz`'s background fit job and are
    never read by any clustering code path.

    Additive ``PUT <index>/_mapping`` — idempotent.
    """
    index = config.items_index
    fields = list(ITEMS_VIZ_MAPPING)
    body = {'properties': ITEMS_VIZ_MAPPING}
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info('curation_mapping_migration', index=index, fields=fields, acknowledged=ack)
        return {'acknowledged': ack, 'index': index, 'fields_added': fields}
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
        return {'acknowledged': False, 'index': index, 'fields_added': fields, 'error': msg}
