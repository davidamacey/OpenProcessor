"""The curation API's item wire format — one serializer for every endpoint.

Every endpoint that returns an item (``GET /crops``, ``GET /crops/{id}``,
``GET /review/{tab}``, ``GET /regions``, ``GET /regions/training_candidates``,
``GET /search/text``) and the ``crop.region_verified`` SSE event emit the
same keys for the same data, built here.

Wire names are FIXED. Region attributes always go out as the stock
``RegionFields()`` default names (``region_status``, ``region_bbox_norm``,
…) no matter what ``OP_REGION_FIELD_*`` storage override a deployment
sets: the storage instance only picks which OpenSearch key a value is
read from. With stock defaults the translation is the identity.

See ``docs/design/curation_api_contract.md``.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any

from src.config.region_fields import RegionFields, get_region_fields
from src.services.curation.class_sources import vlm_suggestion


# Stock defaults double as the wire vocabulary. Never build this from env.
WIRE_REGION_FIELDS = RegionFields()

# `prefix` is not a document key; the embedding is a 1024-d vector no client
# needs; `*_legacy` columns are rollback-only storage.
_NON_WIRE_REGION_ATTRS = frozenset(
    {'prefix', 'embedding', 'bbox_norm_legacy', 'score_legacy', 'status_legacy'}
)

REGION_WIRE_ATTRS: tuple[str, ...] = tuple(
    f.name for f in dataclass_fields(RegionFields) if f.name not in _NON_WIRE_REGION_ATTRS
)


def region_wire_key(attr: str) -> str:
    """Wire JSON key for a ``RegionFields`` attribute (e.g. ``'status'`` ->
    ``'region_status'``), independent of any storage override."""
    return str(getattr(WIRE_REGION_FIELDS, attr))


REGION_WIRE_KEYS: tuple[str, ...] = tuple(region_wire_key(a) for a in REGION_WIRE_ATTRS)

# Heavy vectors never shipped to a client. Callers pass the storage
# instance so an overridden embedding key is excluded too.
_EMBEDDING_SOURCE_EXCLUDES = ('pe_embedding', 'v6_embedding')


def item_source_excludes(storage: RegionFields | None = None) -> list[str]:
    """``_source.excludes`` list for any item search/get that feeds
    :func:`serialize_item`."""
    f = storage or get_region_fields()
    return [*_EMBEDDING_SOURCE_EXCLUDES, f.embedding]


def region_to_wire(src: dict[str, Any], storage: RegionFields | None = None) -> dict[str, Any]:
    """Read every wire region attribute from a stored doc, keyed by its
    fixed wire name."""
    f = storage or get_region_fields()
    return {region_wire_key(a): src.get(getattr(f, a)) for a in REGION_WIRE_ATTRS}


def _api_prefix() -> str:
    from src.config import get_curation_config

    return get_curation_config().api_prefix


def serialize_item(
    src: dict[str, Any],
    fallback_id: str = '',
    *,
    storage: RegionFields | None = None,
    api_prefix: str | None = None,
) -> dict[str, Any]:
    """Project one stored items-index document onto the wire item.

    ``storage`` is the ``RegionFields`` instance the document was written
    with (default: the process-wide one); ``api_prefix`` defaults to the
    configured ``OP_API_PREFIX`` so server-built URLs match the mount.
    """
    f = storage or get_region_fields()
    prefix = _api_prefix() if api_prefix is None else api_prefix
    crop_id = src.get('crop_id') or fallback_id
    image_path = src.get('image_path', '')
    vlm_class_id, vlm_class_name = vlm_suggestion(src)
    item: dict[str, Any] = {
        'id': crop_id,
        'crop_id': crop_id,
        'image_id': src.get('image_id', ''),
        'image_path': image_path,
        'source_image_path': image_path,
        'bbox_norm': src.get('bbox_norm') or [],
        'class_id': src.get('class_id'),
        'class_name': src.get('class_name', ''),
        'class_source': src.get('class_source', ''),
        'confidence': float(src.get('confidence') or 0.0),
        'classifier_raw_confidence': src.get('classifier_raw_confidence'),
        'label_source': src.get('label_source', ''),
        # Derived: either the class or the region was confirmed.
        'label_validated': bool(
            src.get('class_validated') or src.get(f.validated) or src.get('label_validated', False)
        ),
        'class_validated': bool(src.get('class_validated', False)),
        'class_detector': src.get('class_detector'),
        'class_detector_version': src.get('class_detector_version'),
        'class_labeled_at': src.get('class_labeled_at'),
        'class_labeler': src.get('class_labeler'),
        # Categorical VLM confidence (high/medium/low).
        'vlm_confidence': src.get('vlm_confidence'),
        # The VLM's unvalidated class choice (registry id + name), or a
        # proposed new class (name only). Null otherwise.
        'vlm_proposed_class_id': vlm_class_id,
        'vlm_proposed_class_name': vlm_class_name,
        'cluster_id': src.get('cluster_id'),
        'cluster_distance': src.get('cluster_distance'),
        'cluster_subid': src.get('cluster_subid'),
        'test_holdout': bool(src.get('test_holdout', False)),
        'crop_rank_in_image': src.get('crop_rank_in_image'),
        'crop_area_norm': src.get('crop_area_norm'),
        'blur_lap_ratio': src.get('blur_lap_ratio'),
        'proposal_name': src.get('proposal_name'),
        'probe_pred_class': src.get('probe_pred_class'),
        'probe_pred_entropy': src.get('probe_pred_entropy'),
        'mistakenness_score': src.get('mistakenness_score'),
        'mistakenness_method': src.get('mistakenness_method'),
        'mistakenness_version': src.get('mistakenness_version'),
        'mistakenness_scored_at': src.get('mistakenness_scored_at'),
        'uniqueness_score': src.get('uniqueness_score'),
        'dup_group_id': src.get('dup_group_id'),
        'dup_group_size': src.get('dup_group_size'),
        'dup_is_representative': src.get('dup_is_representative'),
        'updated_at': src.get('updated_at') or '',
        'thumbnail_url': f'{prefix}/crops/{crop_id}/thumbnail',
        'region_thumbnail_url': f'{prefix}/crops/{crop_id}/region_thumbnail',
    }
    item.update(region_to_wire(src, f))
    return item


ITEM_WIRE_KEYS: frozenset[str] = frozenset(serialize_item({}, 'x', api_prefix=''))

# Endpoint-specific keys layered on top of the shared item. Everything
# else is identical across endpoints.
REVIEW_EXTRA_KEYS = frozenset({'reason', 'proposed_class_id', 'proposed_class_name'})
TRAINING_CANDIDATE_EXTRA_KEYS = frozenset({'selection_reason'})
SEARCH_EXTRA_KEYS = frozenset({'semantic_score'})


def region_event_payload(
    crop_id: str, *, region_status: str | None, region_text: str | None = None
) -> dict[str, Any]:
    """``crop.region_verified`` SSE payload. Keys are wire names, same as
    the item's."""
    return {
        'type': 'crop.region_verified',
        'topic': region_wire_key('status'),
        'crop_id': crop_id,
        region_wire_key('status'): region_status,
        region_wire_key('text'): region_text,
    }


__all__ = [
    'ITEM_WIRE_KEYS',
    'REGION_WIRE_ATTRS',
    'REGION_WIRE_KEYS',
    'REVIEW_EXTRA_KEYS',
    'SEARCH_EXTRA_KEYS',
    'TRAINING_CANDIDATE_EXTRA_KEYS',
    'WIRE_REGION_FIELDS',
    'item_source_excludes',
    'region_event_payload',
    'region_to_wire',
    'region_wire_key',
    'serialize_item',
]
