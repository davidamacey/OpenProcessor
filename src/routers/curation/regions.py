"""Curation router sub-module — region-of-interest browse endpoints.

Covers browsing items that carry a region-of-interest sub-bbox
(filtered by provenance/score/text/status), the curated training-cohort
picker, and the served region lifecycle / vocabulary catalogs. The
human-edit endpoints that set, clear and patch the sub-bbox live in
:mod:`src.routers.curation.regions_edit`.
"""

from __future__ import annotations

import contextlib
from typing import Any

from fastapi import HTTPException, Query

from src.config import DetectionProfile, get_region_fields
from src.config.region_state import RegionStatus, region_status_catalog
from src.routers.curation._common import (
    OpenSearchDep,
    RegionProfileDep,
    _ensure_indexes,
    guard_page_depth,
    items_index,
    router,
)
from src.routers.curation._error_models import REGION_PROFILE_RESPONSES
from src.routers.curation._item_filter_params import ItemFilterQuery  # noqa: TC001 - FastAPI
from src.routers.curation._region_row_models import RegionRowPage
from src.routers.curation._region_vocabulary_models import RegionVocabularyResponse
from src.services.curation.item_filter import item_filter_clauses
from src.services.curation.region_boxes import BOX_STATES, box_query
from src.services.curation.region_rows import search_region_rows
from src.services.curation.region_vocabulary import region_vocabulary_catalog
from src.services.curation.review_queries import region_text_clause
from src.services.curation.training_cohorts import REGION_LOW_SCORE_MAX, TRAINING_CANDIDATE_MODES
from src.services.curation.wire import item_list_source_excludes
from src.services.detection.profile_registry import region_profile_or_neutral
from src.services.labeling.vlm_endpoints import refresh_vlm_state


_TRAINING_CANDIDATE_MODES = tuple(TRAINING_CANDIDATE_MODES)
_STATUS_VALUES = frozenset(s.value for s in RegionStatus)


def _reason(mode: str) -> str:
    return TRAINING_CANDIDATE_MODES[mode].description


# Large embedding fields (1024 floats) + class_id_history (a
# list-only field no browse renderer reads) — excluded from browse _source.
_REGION_SOURCE_EXCLUDES = item_list_source_excludes()


def _box_clause(parts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The one per-box predicate: every part must hold for the SAME box."""
    return {'bool': {'filter': parts}} if parts else None


@router.get(
    '/regions',
    response_model=None,
    responses={**REGION_PROFILE_RESPONSES, 200: {'model': RegionRowPage}},
)
async def list_regions(
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
    item_filter: ItemFilterQuery,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    cluster_id: int | None = Query(None, description='Filter by the ITEM cluster.'),
    region_cluster_id: int | None = Query(None, description='Box filter: region cluster bucket.'),
    region_cluster_subid: str | None = Query(None, description='Box filter: AHC sub-cluster.'),
    sort_by_subid: bool = Query(
        False,
        description=(
            "Within a bucket, order by the boxes' sub-cluster so AHC sub-clusters "
            'come back contiguous across pages (the UI groups them with '
            'separators). Overrides the default outliers-first ordering.'
        ),
    ),
    min_score: float | None = Query(None, ge=0.0, le=1.0, description='Box filter.'),
    max_score: float | None = Query(None, ge=0.0, le=1.0, description='Box filter.'),
    verified: bool | None = Query(None),
    detector: str | None = Query(None, description='Box filter: the detector that found it.'),
    text: str | None = Query(None, description='Box filter: substring search on the box text.'),
    box_state: str | None = Query(
        None, description='Box filter: a box state (GET /regions/statuses box_states).'
    ),
    status: str | None = Query(
        None,
        description=(
            'Item filter: only items with this region_status (see GET /regions/statuses). '
            'Without it, and without a box filter, the rows are the accepted and '
            'false_positive boxes; with it every item in that status is listed (e.g. '
            'status=verify_rejected lists the verifier-rejected items) -- an item row has '
            'region_box_id null unless a box filter selects boxes too.'
        ),
    ),
    include_test: bool = False,
) -> dict[str, Any]:
    """Browse region rows filtered by provenance/score/text.

    Rows (:class:`RegionRowPage`): the box filters (``detector``,
    ``min_score``, ``max_score``, ``text``, ``region_cluster_id``,
    ``region_cluster_subid``, ``box_state``) all select the SAME box, and
    each matching box is its own row (``region_box_id``). The item filters
    (``status``, ``cluster_id``, ``verified`` and the shared item
    filter: ``class_name``, ``conf_min``/``conf_max``, ``min_area``/``max_area``,
    ``max_rank``, ``origin``, ``embedding_state``, ``review_status``)
    select items. ``page`` / ``page_size`` page items; ``total`` counts
    items, ``total_rows`` rows.
    """
    F = get_region_fields()
    if status is not None and status not in _STATUS_VALUES:
        raise HTTPException(
            status_code=400,
            detail=f'status must be one of {sorted(_STATUS_VALUES)}; got {status!r}',
        )
    if box_state is not None and box_state not in BOX_STATES:
        raise HTTPException(
            status_code=400,
            detail=f'box_state must be one of {sorted(BOX_STATES)}; got {box_state!r}',
        )
    await _ensure_indexes(opensearch)

    parts: list[dict[str, Any]] = []
    if box_state is not None:
        parts.append({'term': {f'{F.boxes}.{F.boxes_state}': box_state}})
    elif status is None:
        # No status and no explicit box state: the boxes a region browse is
        # about -- accepted, or false_positive (kept for FP analysis).
        parts.append(
            {
                'terms': {
                    f'{F.boxes}.{F.boxes_state}': ['accepted', RegionStatus.FALSE_POSITIVE.value]
                }
            }
        )
    if region_cluster_id is not None:
        parts.append({'term': {f'{F.boxes}.cluster_id': region_cluster_id}})
    if region_cluster_subid is not None:
        parts.append({'term': {f'{F.boxes}.cluster_subid': region_cluster_subid}})
    if min_score is not None or max_score is not None:
        rng: dict[str, float] = {}
        if min_score is not None:
            rng['gte'] = min_score
        if max_score is not None:
            rng['lte'] = max_score
        parts.append({'range': {f'{F.boxes}.score': rng}})
    if detector is not None:
        parts.append({'term': {f'{F.boxes}.detector': detector}})
    if text:
        parts.append(region_text_clause(f'{F.boxes}.text', text))

    filt: list[dict[str, Any]] = []
    must_not: list[dict[str, Any]] = []
    if status is not None:
        filt.append({'term': {F.status: status}})
    if not include_test:
        must_not.append({'term': {'test_holdout': True}})
    if cluster_id is not None:
        filt.append({'term': {'cluster_id': cluster_id}})
    try:
        filt.extend(item_filter_clauses(item_filter))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if verified is not None:
        filt.append({'term': {F.verified: verified}})

    # In a bucket, either group by sub-cluster (subid asc, then outliers within
    # each subid) so refine results render as contiguous, paginated groups -- or
    # float regions farthest from the centroid (outliers) first. The sort reads
    # the boxes of the bucket only (the nested filter), so an item with a box
    # in another bucket doesn't sort by that box.
    def _bucket_sort(field: str, order: str, mode: str, unmapped: str) -> dict[str, Any]:
        return {
            f'{F.boxes}.{field}': {
                'order': order,
                'mode': mode,
                'missing': '_last',
                'unmapped_type': unmapped,
                'nested': {
                    'path': F.boxes,
                    'filter': {'term': {f'{F.boxes}.cluster_id': region_cluster_id}},
                },
            }
        }

    sort: list[dict[str, Any]]
    if region_cluster_id is not None:
        distance_sort = _bucket_sort('cluster_distance', 'desc', 'max', 'float')
        sort = (
            [_bucket_sort('cluster_subid', 'asc', 'min', 'keyword'), distance_sort]
            if sort_by_subid
            else [distance_sort]
        )
    else:
        # Items order by their most recently detected box: a nested sort
        # with mode='max' (region_boxes carries detected_at per box).
        sort = [
            {
                f'{F.boxes}.detected_at': {
                    'order': 'desc',
                    'missing': '_last',
                    'unmapped_type': 'date',
                    'nested': {'path': F.boxes},
                    'mode': 'max',
                }
            }
        ]
    sort.append({'crop_id': {'order': 'asc'}})
    guard_page_depth(page, page_size)
    try:
        return await search_region_rows(
            opensearch,
            index=items_index(),
            filters=filt,
            must_not=must_not,
            sort=sort,
            page=page,
            page_size=page_size,
            box_clause=_box_clause(parts),
            source_excludes=_REGION_SOURCE_EXCLUDES,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc


def _training_candidate_query(
    mode: str, profile: DetectionProfile | None = None
) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    """``(item query, box clause, selection_reason)`` for a cohort mode.

    The item query selects items (item-level predicates only); the box
    clause, when the mode's predicate is about ONE box, selects the boxes
    -- every part of it must hold for the same box -- and each matching box
    becomes its own row. A mode whose predicate is item-level
    (``disagreement``, ``human_corrected``) has no box clause: one item
    row, ``region_box_id: null``.

    ``profile`` defaults to the deployment's active region profile; with
    none configured the detector-keyed modes simply match nothing.
    """
    F = get_region_fields()
    if profile is None:
        profile = region_profile_or_neutral()
    state = f'{F.boxes}.{F.boxes_state}'
    holdout = [{'term': {'test_holdout': True}}]

    def item_query(*filters: dict[str, Any]) -> dict[str, Any]:
        return {'bool': {'filter': list(filters), 'must_not': holdout}}

    if mode == 'detector_blind_spots':
        # The primary detector missed but the secondary segmenter found a
        # region and the VLM verified: the high-signal examples the next
        # primary-detector training cycle needs to expand its recall. The
        # segmenter's box is the row.
        return (
            item_query(
                {'term': {F.verified: True}},
                {'term': {F.detector_chain: f'{profile.detector_model}:miss'}},
            ),
            {
                'bool': {
                    'filter': [
                        {'term': {f'{F.boxes}.detector': profile.segmenter_name}},
                        {'term': {state: 'accepted'}},
                    ]
                }
            },
            _reason('detector_blind_spots'),
        )
    if mode == 'low_conf_correct':
        # The primary detector's accepted, low-confidence box on a verified
        # item. The box must itself be accepted: a rejected low-score box on
        # an item that has another accepted box would otherwise contaminate
        # this "primary detector correct but low-confidence" cohort.
        return (
            item_query({'term': {F.verified: True}}),
            {
                'bool': {
                    'filter': [
                        {'term': {f'{F.boxes}.detector': profile.detector_model}},
                        {'range': {f'{F.boxes}.score': {'lt': REGION_LOW_SCORE_MAX}}},
                        {'term': {state: 'accepted'}},
                    ]
                }
            },
            _reason('low_conf_correct'),
        )
    if mode == 'disagreement':
        # Both the primary detector and secondary segmenter fired. The
        # actual IoU disagreement filter is applied client-side (or in a
        # follow-up endpoint with a script_score) -- here we narrow the
        # pool to items that ran through both (chain entries). Item-level.
        return (
            item_query(
                box_query({'term': {state: 'accepted'}}, F),
                {'term': {F.detector_chain: f'{profile.detector_model}:hit'}},
                {'term': {F.detector_chain: f'{profile.segmenter_name}:hit'}},
            ),
            None,
            _reason('disagreement'),
        )
    if mode == 'human_corrected':
        # Items where a human PUT a region AND there was a prior detector
        # chain: a human reviewed a model's output and corrected it.
        # Item-level.
        return (
            item_query(
                {'exists': {'field': F.label_source}},
                {'exists': {'field': F.detector_chain}},
            ),
            None,
            _reason('human_corrected'),
        )
    if mode == 'false_positives':
        # Boxes a human marked "not a region" whose geometry + provenance
        # were KEPT (box state false_positive). They feed the dedicated
        # detector training run as hard negatives -- the detector fired
        # here and should learn not to. Whatever the item status: an item
        # with an accepted sibling box is `detected`, not false_positive.
        return (
            item_query(),
            {'term': {state: RegionStatus.FALSE_POSITIVE.value}},
            _reason('false_positives'),
        )
    raise HTTPException(
        status_code=400,
        detail=f'unknown training-cohort mode: {mode}. Choose from {_TRAINING_CANDIDATE_MODES}.',
    )


@router.get(
    '/regions/training_candidates',
    response_model=None,
    responses={**REGION_PROFILE_RESPONSES, 200: {'model': RegionRowPage}},
)
async def training_candidates(
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
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
    a curated slice of the items index (see :func:`_training_candidate_query`):
    the box-predicate modes return one row per matching box, the
    item-predicate modes one row per item (``region_box_id: null``). Every
    row carries ``selection_reason``.
    """
    F = get_region_fields()
    await _ensure_indexes(opensearch)
    query, box_clause, reason = _training_candidate_query(mode)
    filters = list(query['bool']['filter'])
    if class_id is not None:
        filters.append({'term': {'class_id': class_id}})

    guard_page_depth(page, page_size)
    sort: list[dict[str, Any]] = [
        # Items order by their most recently detected box (nested, mode max).
        {
            f'{F.boxes}.detected_at': {
                'order': 'desc',
                'missing': '_last',
                'unmapped_type': 'date',
                'nested': {'path': F.boxes},
                'mode': 'max',
            }
        },
        {'crop_id': {'order': 'asc'}},
    ]
    try:
        page_body = await search_region_rows(
            opensearch,
            index=items_index(),
            filters=filters,
            must_not=query['bool']['must_not'],
            sort=sort,
            page=page,
            page_size=page_size,
            box_clause=box_clause,
            source_excludes=_REGION_SOURCE_EXCLUDES,
            row_keys={'selection_reason': reason},
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
    return {**page_body, 'mode': mode, 'selection_reason': reason}


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
    """The deployment-configured detector/segmenter/verifier vocabulary.

    ``{detectors: [{id, label, role, filterable}], region_sources:
    [{id, label, role}], chain_actors: [{id, label, role}], text_rules,
    text_choices, rejection_reasons: [{id, label, kind, match,
    label_template}]}``. ``kind`` is ``model_verdict`` / ``automatic`` /
    ``needs_human``; ``match`` is ``exact`` or ``prefix``. Built from
    the active region profile / ingest profiles / the VLM endpoint registry --
    never a hardcoded model id. ``filterable`` marks the values that can
    appear in stored ``region_detector`` (the detector filter's exact
    option list). The frontend renders this instead of hardcoding a
    label/palette map keyed on private model ids."""
    with contextlib.suppress(Exception):
        # Best-effort: the verifier entries come from the endpoint registry;
        # an unreachable store just leaves them out.
        from src.services.projects.guard import make_curation_opensearch

        await refresh_vlm_state(await make_curation_opensearch())
    return region_vocabulary_catalog()
