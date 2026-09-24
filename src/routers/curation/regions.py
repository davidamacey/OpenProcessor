"""Curation router sub-module — region-of-interest browse endpoints.

Ported from the reference implementation's region router. Covers browsing
items that carry a region-of-interest sub-bbox (filtered by
provenance/score/text/status), the curated training-cohort picker, and the
served region lifecycle / vocabulary catalogs. The human-edit endpoints
that set, clear and patch the sub-bbox live in
:mod:`src.routers.curation.regions_edit`.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query

from src.config import DetectionProfile, get_region_fields
from src.config.region_state import RegionStatus, region_status_catalog
from src.routers.curation._common import (
    CURATION_ITEMS_INDEX,
    OpenSearchDep,
    _ensure_indexes,
    guard_page_depth,
    router,
)
from src.routers.curation._region_vocabulary_models import RegionVocabularyResponse
from src.services.curation.region_vocabulary import region_vocabulary_catalog
from src.services.curation.review_queries import region_text_clause
from src.services.curation.training_cohorts import REGION_LOW_SCORE_MAX, TRAINING_CANDIDATE_MODES
from src.services.curation.wire import item_list_source_excludes, serialize_item
from src.services.detection.profile_registry import region_profile_or_neutral


_TRAINING_CANDIDATE_MODES = tuple(TRAINING_CANDIDATE_MODES)
_STATUS_VALUES = frozenset(s.value for s in RegionStatus)


def _reason(mode: str) -> str:
    return TRAINING_CANDIDATE_MODES[mode].description


def _region_item(src: dict[str, Any], crop_id: str) -> dict[str, Any]:
    """Wire item for /regions + /regions/training_candidates — the shared
    item serializer, identical to /crops and /review."""
    return serialize_item(src, crop_id)


# Large embedding fields (1024 floats) + class_id_history (F-25, a
# list-only field no browse renderer reads) — excluded from browse _source.
_REGION_SOURCE_EXCLUDES = item_list_source_excludes()


@router.get('/regions')
async def list_regions(
    opensearch: OpenSearchDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    class_id: int | None = Query(None),
    cluster_id: int | None = Query(None),
    region_cluster_id: int | None = Query(None, description='Filter by region_cluster_id bucket.'),
    region_cluster_subid: str | None = Query(None, description='Filter by AHC region sub-cluster.'),
    sort_by_subid: bool = Query(
        False,
        description=(
            'Within a bucket, order by region_cluster_subid so AHC sub-clusters '
            'come back contiguous across pages (the UI groups them with '
            'separators). Overrides the default outliers-first ordering.'
        ),
    ),
    max_rank: int | None = Query(
        None, ge=1, description='Only regions on top-N largest crops (crop_rank_in_image<=N).'
    ),
    min_score: float | None = Query(None, ge=0.0, le=1.0),
    max_score: float | None = Query(None, ge=0.0, le=1.0),
    verified: bool | None = Query(None),
    detector: str | None = Query(None, description='Filter by region_detector keyword.'),
    text: str | None = Query(None, description='Substring search on region_text.'),
    status: str | None = Query(
        None,
        description=(
            'Only items with this region_status (see GET /regions/statuses). Without it '
            'only items carrying a region box are listed; with it every item in that '
            'status is, e.g. status=verify_rejected lists the verifier-rejected items '
            '(their candidate box is region_candidate_bbox_norm).'
        ),
    ),
    include_test: bool = False,
) -> dict[str, Any]:
    """Browse crops that have a region bbox, filtered by provenance/score/text.

    Used by the labeler's /clusters page when the region-of-interest
    class is the selected class filter, and by any region-centric
    review tooling. ``status`` selects one lifecycle status instead of
    "has a box" (a rejected or absent region has none).
    """
    F = get_region_fields()
    if status is not None and status not in _STATUS_VALUES:
        raise HTTPException(
            status_code=400,
            detail=f'status must be one of {sorted(_STATUS_VALUES)}; got {status!r}',
        )
    await _ensure_indexes(opensearch)
    # F-19: every clause here is a pure predicate (exists/term/range/
    # wildcard-as-boolean-match) -- filter context, not must.
    filt: list[dict[str, Any]] = (
        [{'exists': {'field': F.bbox_norm}}] if status is None else [{'term': {F.status: status}}]
    )
    must_not: list[dict[str, Any]] = []
    if not include_test:
        must_not.append({'term': {'test_holdout': True}})
    if class_id is not None:
        filt.append({'term': {'class_id': class_id}})
    if cluster_id is not None:
        filt.append({'term': {'cluster_id': cluster_id}})
    if region_cluster_id is not None:
        filt.append({'term': {F.cluster_id: region_cluster_id}})
    if region_cluster_subid is not None:
        filt.append({'term': {F.cluster_subid: region_cluster_subid}})
    if max_rank is not None:
        filt.append({'range': {'crop_rank_in_image': {'lte': max_rank}}})
    if min_score is not None or max_score is not None:
        rng: dict[str, float] = {}
        if min_score is not None:
            rng['gte'] = min_score
        if max_score is not None:
            rng['lte'] = max_score
        filt.append({'range': {F.score: rng}})
    if verified is not None:
        filt.append({'term': {F.verified: verified}})
    if detector is not None:
        filt.append({'term': {F.detector: detector}})
    if text:
        filt.append(region_text_clause(F.text, text))

    # In a bucket, either group by sub-cluster (subid asc, then outliers within
    # each subid) so refine results render as contiguous, paginated groups — or
    # float regions farthest from the centroid (outliers) first.
    _distance_sort = {
        F.cluster_distance: {'order': 'desc', 'missing': '_last', 'unmapped_type': 'float'}
    }
    if region_cluster_id is not None and sort_by_subid:
        sort = [
            {
                F.cluster_subid: {
                    'order': 'asc',
                    'missing': '_last',
                    'unmapped_type': 'keyword',
                }
            },
            _distance_sort,
        ]
    elif region_cluster_id is not None:
        sort = [_distance_sort]
    else:
        sort = [{F.detected_at: {'order': 'desc', 'missing': '_last'}}]
    sort.append({'crop_id': {'order': 'asc'}})
    guard_page_depth(page, page_size)
    body: dict[str, Any] = {
        'from': (page - 1) * page_size,
        'size': page_size,
        '_source': {'excludes': _REGION_SOURCE_EXCLUDES},
        'query': {'bool': {'filter': filt, 'must_not': must_not}},
        'sort': sort,
        'track_total_hits': True,
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc

    hits = (resp.get('hits') or {}).get('hits') or []
    total = int((resp.get('hits') or {}).get('total', {}).get('value', 0))
    items = [_region_item(h.get('_source') or {}, h.get('_id') or '') for h in hits]
    return {'total': total, 'page': page, 'page_size': page_size, 'items': items}


def _training_candidate_query(
    mode: str, profile: DetectionProfile | None = None
) -> tuple[dict[str, Any], str]:
    """Return the OpenSearch query body + selection_reason for a mode.

    ``profile`` defaults to the deployment's active region profile; with
    none configured the detector-keyed modes simply match nothing.
    """
    F = get_region_fields()
    if profile is None:
        profile = region_profile_or_neutral()
    if mode == 'detector_blind_spots':
        # Primary detector missed but the secondary segmenter found a
        # region, the VLM verified. These are the high-signal training
        # examples — the next primary-detector training cycle needs
        # exactly these crops to expand its recall.
        return (
            {
                'bool': {
                    'filter': [
                        {'exists': {'field': F.bbox_norm}},
                        {'term': {F.detector: profile.segmenter_name}},
                        {'term': {F.verified: True}},
                        {
                            'term': {
                                F.detector_chain: f'{profile.detector_model}:miss',
                            }
                        },
                    ],
                    'must_not': [{'term': {'test_holdout': True}}],
                }
            },
            _reason('detector_blind_spots'),
        )
    if mode == 'low_conf_correct':
        return (
            {
                'bool': {
                    'filter': [
                        {'term': {F.detector: profile.detector_model}},
                        {'term': {F.verified: True}},
                        {'range': {F.score: {'lt': REGION_LOW_SCORE_MAX}}},
                    ],
                    'must_not': [{'term': {'test_holdout': True}}],
                }
            },
            _reason('low_conf_correct'),
        )
    if mode == 'disagreement':
        # Both the primary detector and secondary segmenter fired. The
        # actual IoU disagreement filter is applied client-side (or in a
        # follow-up endpoint with a script_score) — here we narrow the
        # pool to crops that ran through both. Tag both presences via
        # chain entries.
        return (
            {
                'bool': {
                    'filter': [
                        {'exists': {'field': F.bbox_norm}},
                        {'term': {F.detector_chain: f'{profile.detector_model}:hit'}},
                        {'term': {F.detector_chain: f'{profile.segmenter_name}:hit'}},
                    ],
                    'must_not': [{'term': {'test_holdout': True}}],
                }
            },
            _reason('disagreement'),
        )
    if mode == 'human_corrected':
        # Rows where a human PUT a region AND there was a prior detector
        # chain. These are the gold-standard training rows — a human
        # reviewed a model's output and corrected it.
        return (
            {
                'bool': {
                    'filter': [
                        {'exists': {'field': F.label_source}},
                        {'exists': {'field': F.detector_chain}},
                    ],
                    'must_not': [{'term': {'test_holdout': True}}],
                }
            },
            _reason('human_corrected'),
        )
    if mode == 'false_positives':
        # Human marked a detected box as "not a region" but the box +
        # provenance were KEPT (region status='false_positive'). These
        # feed the dedicated detector training run as hard negatives —
        # the detector fired here and should learn not to.
        return (
            {
                'bool': {
                    'filter': [
                        {'term': {F.status: RegionStatus.FALSE_POSITIVE}},
                        {'exists': {'field': F.bbox_norm}},
                    ],
                    'must_not': [{'term': {'test_holdout': True}}],
                }
            },
            _reason('false_positives'),
        )
    raise HTTPException(
        status_code=400,
        detail=f'unknown training-cohort mode: {mode}. Choose from {_TRAINING_CANDIDATE_MODES}.',
    )


@router.get('/regions/training_candidates')
async def training_candidates(
    opensearch: OpenSearchDep,
    mode: str = Query(
        ...,
        description=(
            'detector_blind_spots | low_conf_correct | disagreement | '
            'human_corrected | false_positives'
        ),
    ),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    class_id: int | None = Query(None),
) -> dict[str, Any]:
    """Return regions filtered for the next training cycle.

    Powers the /train cockpit's Training-cohort picker. ``mode`` selects
    a curated slice of the items index (see :func:`_training_candidate_query`).
    """
    F = get_region_fields()
    await _ensure_indexes(opensearch)
    query, reason = _training_candidate_query(mode)
    if class_id is not None:
        # Tack the class filter onto the bool.filter of the mode query.
        query['bool']['filter'] = [*query['bool']['filter'], {'term': {'class_id': class_id}}]

    guard_page_depth(page, page_size)
    body: dict[str, Any] = {
        'from': (page - 1) * page_size,
        'size': page_size,
        '_source': {'excludes': _REGION_SOURCE_EXCLUDES},
        'query': query,
        'sort': [
            {F.detected_at: {'order': 'desc', 'missing': '_last'}},
            {'crop_id': {'order': 'asc'}},
        ],
        'track_total_hits': True,
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc

    hits = (resp.get('hits') or {}).get('hits') or []
    total = int((resp.get('hits') or {}).get('total', {}).get('value', 0))
    items: list[dict[str, Any]] = []
    for h in hits:
        item = _region_item(h.get('_source') or {}, h.get('_id') or '')
        item['selection_reason'] = reason
        items.append(item)
    return {
        'total': total,
        'page': page,
        'page_size': page_size,
        'mode': mode,
        'selection_reason': reason,
        'items': items,
    }


@router.get('/regions/statuses')
async def region_statuses() -> dict[str, Any]:
    """The region lifecycle vocabulary: every status with its ``label``,
    ``role``, ``terminal``, ``human_writable``, ``clears_box`` and
    ``wants_reason``, plus the ``confirm_status`` / ``reject_status`` /
    ``false_positive_status`` values. Generated from
    ``src/config/region_state.py``; the human writers enforce it."""
    return region_status_catalog()


@router.get('/regions/vocabulary', response_model=RegionVocabularyResponse)
async def regions_vocabulary() -> dict[str, Any]:
    """The deployment-configured detector/segmenter/verifier vocabulary
    (W0: naming sweep finding m9).

    ``{detectors: [{id, label, role, filterable}], region_sources:
    [{id, label, role}], chain_actors: [{id, label, role}], text_rules,
    text_choices, rejection_reasons: [{id, label, kind, match,
    label_template}]}``. ``kind`` is ``model_verdict`` / ``automatic`` /
    ``needs_human``; ``match`` is ``exact`` or ``prefix``. Built from
    the active region profile / ingest profiles / ``OP_VLM_MODEL`` --
    never a hardcoded model id. ``filterable`` marks the values that can
    appear in stored ``region_detector`` (the detector filter's exact
    option list). The frontend renders this instead of hardcoding a
    label/palette map keyed on private model ids."""
    return region_vocabulary_catalog()
