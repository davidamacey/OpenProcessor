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

from src.clients.occ_locks import is_locked_box, is_locked_item
from src.config.curation import BACKBONE_EMBEDDING_FIELD, ITEM_EMBEDDING_FIELD
from src.config.region_fields import RegionFields, get_region_fields
from src.services.curation.class_sources import (
    class_confidence,
    vlm_suggestion,
    vlm_suggestion_dismissed,
)
from src.services.curation.cluster_ids import CORE_SIMILARITY_MIN, cluster_kind, cluster_similarity
from src.services.curation.item_text import ITEM_TEXT_LINES_FIELD, item_text_lines_to_wire


# Stock defaults double as the wire vocabulary. Never build this from env.
WIRE_REGION_FIELDS = RegionFields()

# `prefix` is not a document key; the embedding is a 1024-d vector no client
# needs; `*_legacy` columns are rollback-only storage.
#
# The per-item box-list summary fields and the list itself are excluded
# from the GENERIC per-attribute loop below (:func:`region_to_wire`)
# because they're served by :func:`region_boxes_to_wire` instead, with
# their own fixed wire keys (``region_boxes``, ``region_count``, …) --
# not the ``region_wire_key(attr)`` derivation this module uses for the
# legacy per-box scalars. ``boxes_state`` is never a top-level document
# key at all (it names the FIXED element key `state` inside one
# ``region_boxes`` list entry, used only to build nested queries via
# ``region_boxes.box_query``).
_NON_WIRE_REGION_ATTRS = frozenset(
    {
        'prefix',
        'status_legacy',
        'boxes',
        'boxes_state',
        'box_embeddings',
        'count',
        'rejected_count',
        'max_score',
        'set_complete',
        'revision',
        'box_seq',
    }
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
_EMBEDDING_SOURCE_EXCLUDES = (ITEM_EMBEDDING_FIELD, BACKBONE_EMBEDDING_FIELD)


def item_source_excludes(storage: RegionFields | None = None) -> list[str]:
    """``_source.excludes`` list for any item search/get that feeds
    :func:`serialize_item`."""
    f = storage or get_region_fields()
    return [*_EMBEDDING_SOURCE_EXCLUDES, f.box_embeddings]


def item_list_source_excludes(storage: RegionFields | None = None) -> list[str]:
    """``_source.excludes`` for a *list* endpoint (card grid / review queue
    / regions browse / semantic search) — :func:`item_source_excludes`
    plus ``class_id_history``.

    ``class_id_history`` (up to 32 entries) is only ever read by the
    undo path (``label_undo.py``, which stays on
    :func:`item_source_excludes` — it needs the history) — no list
    renderer reads it. Shipping it in every row of a paginated list
    response decompresses + serializes a field nobody displays.
    """
    return [*item_source_excludes(storage), 'class_id_history']


def region_to_wire(src: dict[str, Any], storage: RegionFields | None = None) -> dict[str, Any]:
    """Read every wire region attribute from a stored doc, keyed by its
    fixed wire name."""
    f = storage or get_region_fields()
    return {region_wire_key(a): src.get(getattr(f, a)) for a in REGION_WIRE_ATTRS}


def _xyxy(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _source_to_parent(src: dict[str, Any], region: tuple[float, float, float, float]) -> Any:
    parent = _xyxy(src.get('bbox_norm'))
    if parent is None:
        return None
    px1, py1, px2, py2 = parent
    pw, ph = px2 - px1, py2 - py1
    rx1, ry1, rx2, ry2 = region
    return [
        min(max((rx1 - px1) / pw, 0.0), 1.0),
        min(max((ry1 - py1) / ph, 0.0), 1.0),
        min(max((rx2 - px1) / pw, 0.0), 1.0),
        min(max((ry2 - py1) / ph, 0.0), 1.0),
    ]


def _proposed_class(
    src: dict[str, Any], vlm_class_id: int | None, vlm_class_name: str | None
) -> dict[str, Any]:
    if vlm_class_name is not None:
        return {'proposed_class_id': vlm_class_id, 'proposed_class_name': vlm_class_name}
    if vlm_suggestion_dismissed(src):
        # The current class *is* the rejected VLM pick; confirming must not apply it.
        return {'proposed_class_id': None, 'proposed_class_name': ''}
    return {
        'proposed_class_id': src.get('class_id'),
        'proposed_class_name': src.get('vlm_raw_class') or src.get('class_name') or '',
    }


def current_cluster_distance(src: dict[str, Any]) -> Any:
    """The stored ``cluster_distance`` if it was measured against the item's
    current cluster (``cluster_distance_cluster_id`` absent — written
    before that field existed — or equal to ``cluster_id``), else ``None``."""
    ref = src.get('cluster_distance_cluster_id')
    if ref is not None and ref != src.get('cluster_id'):
        return None
    return src.get('cluster_distance')


def _api_prefix() -> str:
    """The bound project's base (``{api_prefix}/projects/{slug}``): every
    served URL is scoped to the project that served it."""
    from src.config.project_context import project_api_base

    return project_api_base()


def _probe_actionable_min_confidence() -> float:
    from src.config import get_curation_config

    return get_curation_config().probe_actionable_min_confidence


def probe_actionable(
    *,
    probe_pred_class: Any,
    probe_in_scope: bool | None,
    probe_disagreement: bool | None,
    probe_pred_confidence: float | None,
    min_confidence: float | None = None,
) -> bool | None:
    """Whether a client should offer "accept model's class" for this item.

    ``None`` when the probe hasn't scored the item at all (mirrors
    ``probe_in_scope``/``probe_disagreement``'s own null). Once scored,
    ``True`` only when all three hold: the item's class is one the probe
    was trained on (``probe_in_scope``), the probe's top-1 differs from it
    (``probe_disagreement``), and the probe is confident enough in its
    own top-1 (``probe_pred_confidence >= min_confidence``) -- a
    disagreeing-but-unsure probe (e.g. near-uniform posterior) must not
    be offered as a one-click accept. ``False`` for every other case:
    in-scope + agreeing, out-of-scope, or disagreeing-but-unsure.

    Confidence, not entropy, gates this -- see
    ``CurationConfig.probe_actionable_min_confidence``'s docstring for why
    ``probe_pred_entropy`` isn't a reliable normalized signal here.
    """
    if probe_pred_class is None:
        return None
    threshold = _probe_actionable_min_confidence() if min_confidence is None else min_confidence
    return bool(
        probe_in_scope
        and probe_disagreement is True
        and probe_pred_confidence is not None
        and probe_pred_confidence >= threshold
    )


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
    distance = current_cluster_distance(src)
    similarity = cluster_similarity(distance)
    label_conf, label_conf_source = class_confidence(src)
    probe_pred_class = src.get('probe_pred_class')
    probe_disagreement = src.get('probe_disagreement')
    probe_in_scope = None if probe_pred_class is None else probe_disagreement is not None
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
        # The detector/classifier score, whatever wrote the label.
        'confidence': float(src.get('confidence') or 0.0),
        # The confidence of the writer that set the label (VLM category
        # mapped to a number, or the classifier score); null for human /
        # merge / import labels.
        'class_confidence': label_conf,
        'class_confidence_source': label_conf_source,
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
        # Provenance of the VLM's most recent write: prompt pack, and which
        # endpoint / model answered.
        'vlm_prompt_pack': src.get('vlm_prompt_pack'),
        'vlm_endpoint': src.get('vlm_endpoint'),
        'vlm_model': src.get('vlm_model'),
        # Last VLM class attempt, and why it gave no class (null = it did).
        'vlm_class_attempted_at': src.get('vlm_class_attempted_at'),
        'vlm_class_empty_reason': src.get('vlm_class_empty_reason'),
        # The VLM's class answer verbatim — for vlm_unmatched, the label it
        # named outside the registry (class_name is whatever the item
        # already carried).
        'vlm_raw_class': src.get('vlm_raw_class'),
        # The VLM's unvalidated class choice (registry id + name), or a
        # proposed new class (name only). Null otherwise.
        'vlm_proposed_class_id': vlm_class_id,
        'vlm_proposed_class_name': vlm_class_name,
        # The class a one-key confirm would apply: the VLM suggestion when
        # there is one, else the current class (or the raw unmatched VLM
        # answer as the name).
        **_proposed_class(src, vlm_class_id, vlm_class_name),
        # Human flagged "needs a class the registry doesn't have yet".
        'needs_new_class': bool(src.get('needs_new_class', False)),
        'needs_new_class_note': src.get('needs_new_class_note'),
        'cluster_id': src.get('cluster_id'),
        'cluster_kind': cluster_kind(src.get('cluster_id')),
        # Null when measured against a cluster the item has since left.
        'cluster_distance': distance,
        'cluster_similarity': similarity,
        'cluster_is_core': None if similarity is None else similarity >= CORE_SIMILARITY_MIN,
        # Cluster whose centroid is nearest this item (== cluster_id when
        # it sits best where it is); null unless measured for its cluster.
        'cluster_nearest_id': (
            src.get('cluster_nearest_id')
            if src.get('cluster_distance_cluster_id') == src.get('cluster_id')
            else None
        ),
        'cluster_subid': src.get('cluster_subid'),
        'class_excluded': bool(src.get('class_excluded', False)),
        'excluded_reason': src.get('excluded_reason'),
        'excluded_at': src.get('excluded_at'),
        # Set = hidden from every /review tab (discard / review_dismiss).
        'review_dismissed_at': src.get('review_dismissed_at'),
        'source': src.get('source') or '',
        'test_holdout': bool(src.get('test_holdout', False)),
        # Server-computed lock rule (src.clients.occ_locks.is_locked_item): a
        # human- or import-owned label, box or verdict that no automated
        # writer will touch. The client never re-derives it.
        'label_locked': is_locked_item(src, f),
        # Dataset-import provenance (W10.10).
        'import_ids': list(src.get('import_ids') or []),
        'dataset_split': src.get('dataset_split'),
        'imported_at': src.get('imported_at'),
        'proposed_by_import': src.get('proposed_by_import'),
        'on_negative_frame': bool(src.get('on_negative_frame', False)),
        'import_standalone_region': bool(src.get('import_standalone_region', False)),
        'proposal_chain': list(src.get('proposal_chain') or []),
        'crop_rank_in_image': src.get('crop_rank_in_image'),
        'crop_area_norm': src.get('crop_area_norm'),
        'blur_lap_ratio': src.get('blur_lap_ratio'),
        'proposal_name': src.get('proposal_name'),
        'probe_pred_class': probe_pred_class,
        'probe_pred_class_id': src.get('probe_pred_class_id'),
        'probe_pred_entropy': src.get('probe_pred_entropy'),
        # D1: null means the probe has no opinion (either it hasn't scored
        # this item, or — when this item's current class isn't one the probe
        # was trained on — it structurally can't disagree with a class it
        # was never shown). The UI must treat null as "no opinion" and never
        # offer an "accept model's class" action for it; only a real
        # True/False value is an actual probe opinion. See
        # src.services.curation.probe_predictions for how this is computed.
        'probe_disagreement': probe_disagreement,
        # Explicit, so a client never has to infer scope from
        # probe_disagreement being null: None when the probe hasn't scored
        # this item at all; once scored, True iff the item's class was one
        # the probe was trained on (probe_disagreement is then a real
        # True/False), False when the probe scored the item but the item's
        # class is outside the probe's class set (probe_disagreement is
        # then null — structurally no opinion, not an agreement).
        'probe_in_scope': probe_in_scope,
        # The probe checkpoint's version tag (see probe_predictions.py's
        # probe_model_version) -- the closest thing this system has to a
        # "probe run id" today; null before the probe has ever scored this
        # item.
        'probe_model_version': src.get('probe_model_version'),
        # D1 follow-up: the backend's own accept/no-accept decision, so the
        # frontend never re-derives a threshold. Null mirrors
        # probe_in_scope/probe_disagreement (probe hasn't scored this item);
        # true only when in-scope + disagreeing + confident enough (see
        # CurationConfig.probe_actionable_min_confidence); false for every
        # other case, including disagreeing-but-unsure.
        'probe_actionable': probe_actionable(
            probe_pred_class=probe_pred_class,
            probe_in_scope=probe_in_scope,
            probe_disagreement=probe_disagreement,
            probe_pred_confidence=src.get('probe_pred_confidence'),
        ),
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
        # Every OCR line on the item crop; [] when none / not yet read.
        'item_text_lines': item_text_lines_to_wire(src.get(ITEM_TEXT_LINES_FIELD)),
    }
    item.update(region_to_wire(src, f))
    item.update(region_boxes_to_wire(src, f, crop_id=crop_id, prefix=prefix))
    return item


def box_thumbnail_url(prefix: str, crop_id: str, box_id: str) -> str:
    """The region close-up of one box (``GET /crops/{id}/region_thumbnail``
    needs its ``box_id``): the one place that URL is built."""
    return f'{prefix}/crops/{crop_id}/region_thumbnail?box_id={box_id}'


def region_boxes_to_wire(
    src: dict[str, Any],
    storage: RegionFields | None = None,
    *,
    crop_id: str = '',
    prefix: str = '',
) -> dict[str, Any]:
    """The W8 per-item box list plus its item-level summary fields
    (any_domain_plan.md W8.9). Additive alongside the legacy per-box
    scalar wire keys above -- see the W8 handback report for why those
    are not removed yet. Each ``RegionBoxWire`` element also carries
    ``bbox_in_parent`` (the item-crop frame, derived; ``None`` with no
    usable item box) and ``thumbnail_url`` (``crop_id`` unset -> both
    keys omitted, e.g. a bare doc with no crop identity)."""
    from src.services.curation.region_boxes import read_boxes

    f = storage or get_region_fields()
    boxes = []
    for box in read_boxes(src, f):
        doc = box.to_doc()
        doc['locked'] = is_locked_box(box)
        doc['bbox_in_parent'] = _source_to_parent(src, box.bbox_norm)
        doc['thumbnail_url'] = box_thumbnail_url(prefix, crop_id, box.box_id) if crop_id else None
        boxes.append(doc)
    return {
        'region_boxes': boxes,
        'region_count': int(src.get(f.count) or 0),
        'region_rejected_count': int(src.get(f.rejected_count) or 0),
        'region_max_score': src.get(f.max_score),
        'region_set_complete': src.get(f.set_complete),
        'region_revision': int(src.get(f.revision) or 0),
    }


ITEM_WIRE_KEYS: frozenset[str] = frozenset(serialize_item({}, 'x', api_prefix=''))

# Endpoint-specific keys layered on top of the shared item. Everything
# else is identical across endpoints.
REVIEW_EXTRA_KEYS = frozenset({'reason'})
# A region row is the item plus the box it is about (``None`` for an item row).
REGION_ROW_EXTRA_KEYS = frozenset({'region_box_id'})
TRAINING_CANDIDATE_EXTRA_KEYS = REGION_ROW_EXTRA_KEYS | {'selection_reason'}
SEARCH_EXTRA_KEYS = frozenset({'semantic_score'})


def region_event_payload(
    crop_id: str, *, region_status: str | None, region_count: int | None = None
) -> dict[str, Any]:
    """``crop.region_verified`` SSE payload. Keys are wire names, same as
    the item's: the item status and how many accepted boxes it now holds."""
    return {
        'type': 'crop.region_verified',
        'topic': 'region_status',
        'crop_id': crop_id,
        'region_status': region_status,
        'region_count': region_count,
    }


__all__ = [
    'ITEM_WIRE_KEYS',
    'REGION_ROW_EXTRA_KEYS',
    'REGION_WIRE_ATTRS',
    'REGION_WIRE_KEYS',
    'REVIEW_EXTRA_KEYS',
    'SEARCH_EXTRA_KEYS',
    'TRAINING_CANDIDATE_EXTRA_KEYS',
    'WIRE_REGION_FIELDS',
    'box_thumbnail_url',
    'current_cluster_distance',
    'item_list_source_excludes',
    'item_source_excludes',
    'region_boxes_to_wire',
    'region_event_payload',
    'region_to_wire',
    'region_wire_key',
    'serialize_item',
]
