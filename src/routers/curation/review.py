"""Curation review-queue router — ``GET/POST /curation/review/*``.

``POST /review/new_class_proposals/resolve`` lives in the sibling
``review_resolve.py`` module and ``GET /review/new_class_proposals/summary``
in ``review_proposals.py`` (LOC-ceiling splits), both registered on this
same shared ``router``.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import HTTPException, Path as PathParam, Query

from src.config import get_region_fields
from src.routers.curation._common import (
    CURATION_ITEMS_INDEX,
    OpenSearchDep,
    TestHoldoutFreezeRequest,
    TestHoldoutFreezeResponse,
    _ensure_indexes,
    _now_iso,
    get_class_registry,
    guard_page_depth,
    is_not_found,
    logger,
    router,
)
from src.routers.curation._review_tab_models import ReviewTabsResponse
from src.services.curation import review_empty_reason, review_queries
from src.services.curation.dataset_thresholds import MIN_TEST_CROPS_PER_CLASS
from src.services.curation.holdout import (
    MIN_TEST_PER_CLASS,
    build_cohort_query,
    compute_holdout_sha,
    fetch_cohort_strata,
    persist_freeze_record,
    select_test_holdout,
)
from src.services.curation.raw_label_clusters import (
    CLUSTER_ID_FIELD,
    CLUSTER_NAME_FIELD,
    RAW_LABEL_FIELD,
    UNMATCHED_CLASS_SOURCE,
)
from src.services.curation.review_request import (
    ReviewFilters,
    UnlocatableSortError,
    before_query,
    build_review_request,
)
from src.services.curation.wire import item_list_source_excludes, serialize_item


@router.get('/review/unmatched_terms')
async def review_unmatched_terms(
    opensearch: OpenSearchDep,
    size: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    """Aggregate the VLM's raw labels across every ``vlm_unmatched`` crop.

    Returns the top-N most common raw labels the VLM produced for crops the
    registry could not resolve. This is the main input to growing the
    registry: high-count labels are obvious candidates for new class
    entries (or new ``SYNONYMS`` mappings if the raw label is just a
    phrasing of an existing class).

    Once the registry (or the prompt pack's synonyms) has grown, run
    ``scripts/curation/reclassify_after_registry_growth.py`` with
    ``--label-prefix`` set to the prefix of the ``class_source`` /
    raw-label pair aggregated here; it promotes every ``<prefix>_unmatched``
    item whose raw label now resolves to ``<prefix>_reclassified`` (never
    setting ``class_validated``).

    Args:
        size: Maximum number of distinct raw labels to return. Capped at
            1000; default 100 (the long tail past 100 is usually noise).

    Returns:
        ``{"total_unmatched": <int>, "top_terms": [{"label": str, "count": int}, ...]}``
    """
    await _ensure_indexes(opensearch)
    body = {
        'size': 0,
        'query': {'term': {'class_source': 'vlm_unmatched'}},
        'aggs': {
            'top_raw': {
                'terms': {
                    'field': 'vlm_raw_label',
                    'size': size,
                    # Push rare/unknowns to the bottom and avoid empty buckets.
                    'min_doc_count': 1,
                }
            },
        },
        'track_total_hits': True,
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc

    total = (resp.get('hits') or {}).get('total', {}).get('value', 0)
    buckets = (((resp.get('aggregations') or {}).get('top_raw') or {}).get('buckets')) or []
    top_terms = [
        {'label': str(b.get('key', '')), 'count': int(b.get('doc_count', 0))} for b in buckets
    ]
    return {'total_unmatched': int(total), 'top_terms': top_terms}


@router.get('/review/raw_label_clusters')
async def review_raw_label_clusters(
    opensearch: OpenSearchDep,
    size: int = Query(50, ge=1, le=500),
    samples_per_cluster: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    """Top-N hierarchical clusters of ``vlm_raw_label`` for the labeler UI.

    Surfaces the output of ``scripts/curation/cluster_raw_labels.py``
    (field contract: :mod:`src.services.curation.raw_label_clusters`) so
    the labeler can:

    1. Show fine-grained sub-classes the registry doesn't have yet
       (e.g. "ford f150" + "ford ranger" rolled up under
       ``ford_pickup`` → suggested registry parent ``pickup``).
    2. Surface candidate registry promotions ranked by crop volume.
    3. Let curators bulk-relabel a whole cluster at once instead of
       clicking through individual ``vlm_unmatched`` crops.

    The endpoint runs purely against the configured items index — no clustering
    happens here, only aggregation. Cluster ids / names are written by
    the offline script; if no crops have ``vlm_label_cluster_id`` yet
    the endpoint returns an empty list with a populated ``hint``.

    NOTE on route ordering: this endpoint MUST be declared before the
    parameterised ``/review/{tab}`` below — FastAPI dispatches the first
    matching route, and ``raw_label_clusters`` would otherwise be eaten
    by the ``{tab}`` matcher.

    Args:
        size: Number of clusters to return (default 50).
        samples_per_cluster: How many ``vlm_raw_label`` samples to
            include per cluster (default 5 — enough to read at a glance).

    Returns:
        ``{"status": "ok|empty", "clusters": [{cluster_id, cluster_name,
        n_crops, n_unmatched, sample_terms, parent_class_suggestion}, ...]}``
    """
    await _ensure_indexes(opensearch)
    body = {
        'size': 0,
        'query': {'exists': {'field': CLUSTER_ID_FIELD}},
        'aggs': {
            'by_cluster': {
                'terms': {
                    'field': CLUSTER_ID_FIELD,
                    'size': size,
                    'order': {'_count': 'desc'},
                },
                'aggs': {
                    'name': {'terms': {'field': CLUSTER_NAME_FIELD, 'size': 1}},
                    'samples': {
                        'terms': {
                            'field': RAW_LABEL_FIELD,
                            'size': samples_per_cluster,
                        }
                    },
                    'unmatched': {'filter': {'term': {'class_source': UNMATCHED_CLASS_SOURCE}}},
                    # Most common already-resolved class within the cluster — used
                    # as the ``parent_class_suggestion`` hint. If the cluster is
                    # 100% vlm_unmatched the bucket is empty and we return None.
                    'parent': {
                        # ``class_name`` is mapped ``keyword`` directly on the
                        # live index — no ``.keyword`` subfield exists (the
                        # migration that assumed a ``text`` mapping has been
                        # failing silently on every startup; see
                        # the reference clusters router's top_class agg for the full story).
                        'terms': {'field': 'class_name', 'size': 1},
                    },
                },
            },
        },
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc

    buckets = (((resp.get('aggregations') or {}).get('by_cluster') or {}).get('buckets')) or []
    clusters: list[dict[str, Any]] = []
    for b in buckets:
        cid = int(b.get('key', 0))
        n_crops = int(b.get('doc_count', 0))
        name_buckets = (b.get('name') or {}).get('buckets') or []
        cluster_name = str(name_buckets[0]['key']) if name_buckets else f'cluster_{cid}'
        sample_buckets = (b.get('samples') or {}).get('buckets') or []
        samples = [
            {'label': str(sb.get('key', '')), 'count': int(sb.get('doc_count', 0))}
            for sb in sample_buckets
        ]
        unmatched = int((b.get('unmatched') or {}).get('doc_count', 0))
        parent_buckets = (b.get('parent') or {}).get('buckets') or []
        parent = str(parent_buckets[0]['key']) if parent_buckets else None
        clusters.append(
            {
                'cluster_id': cid,
                'cluster_name': cluster_name,
                'n_crops': n_crops,
                'n_unmatched': unmatched,
                'sample_terms': samples,
                'parent_class_suggestion': parent,
            }
        )
    status_str = 'ok' if clusters else 'empty'
    return {
        'status': status_str,
        'clusters': clusters,
        'hint': (
            'Run scripts/curation/cluster_raw_labels.py to populate '
            f'{CLUSTER_ID_FIELD} on items if status=empty.'
        )
        if not clusters
        else None,
    }


_TAB_DESCRIPTION = 'One of: ' + ' | '.join(review_queries.KNOWN_TABS)


def _filters(
    include_test: bool,
    text: str | None,
    max_rank: int | None,
    min_blur_ratio: float | None,
    min_mistakenness: float | None,
    hide_near_duplicates: bool,
    class_id: int | None,
    source: str | None,
    conf_min: float | None,
    conf_max: float | None,
    region_status: str | None,
) -> ReviewFilters:
    return ReviewFilters(
        include_test=include_test,
        text=text,
        max_rank=max_rank,
        min_blur_ratio=min_blur_ratio,
        min_mistakenness=min_mistakenness,
        hide_near_duplicates=hide_near_duplicates,
        class_id=class_id,
        source=source,
        conf_min=conf_min,
        conf_max=conf_max,
        region_status=region_status,
    )


# Filter params shared by the queue and locate routes (Annotated defaults,
# so direct Python callers get plain values).
IncludeTest = Annotated[bool, Query()]
TextQ = Annotated[
    str | None,
    Query(description='Regions tab only: case-insensitive substring search on region_text.'),
]
MaxRankQ = Annotated[
    int | None,
    Query(
        ge=1,
        description=(
            'Keep crop_rank_in_image <= this (every tab). Omitted: no limit, '
            "except a tab's served filter_defaults (GET /review/tabs)."
        ),
    ),
]
BlurQ = Annotated[float | None, Query(ge=0.0, description='Clarity floor (null-safe).')]
MistakeQ = Annotated[float | None, Query(ge=0.0, description='Mistakenness floor (null-safe).')]
NearDupQ = Annotated[bool, Query(description='Hide non-representative near-duplicates.')]
ClassIdQ = Annotated[int | None, Query(description='Only items of this class.')]
SourceQ = Annotated[str | None, Query(description='Only items with this ingest source tag.')]
ConfQ = Annotated[float | None, Query(ge=0.0, le=1.0, description='Inclusive confidence band.')]
RegionStatusQ = Annotated[
    str | None,
    Query(
        description=(
            "Regions tab only (ignored elsewhere). One of 'all' (default: "
            'accepted-but-unvalidated boxes plus a verifier-rejected '
            "candidate that still has a box), 'detected', 'verify_rejected'. "
            'See GET /review/tabs filter_specs (param region_status).'
        )
    ),
]
SortQ = Annotated[
    str | None,
    Query(
        description=(
            'Review-sort id from GET /curation/methods (axis=sort). Omitted or '
            "'default': the tab's own default. Unknown / shadow / disabled -> 400."
        )
    ),
]


async def _request(tab: str, filters: ReviewFilters, sort: str | None, opensearch: Any) -> Any:
    try:
        return await build_review_request(tab, filters, sort, opensearch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get('/review/tabs', response_model=ReviewTabsResponse)
async def review_tabs(opensearch: OpenSearchDep) -> dict[str, Any]:
    """Every review tab's ``id``/``label``/``description`` (W0: naming
    sweep finding m9) plus ``filters`` (the query parameters it honours)
    ``filter_defaults`` (values it applies when one is omitted) and
    ``filter_specs`` (self-describing enum filters) — the frontend renders this instead of hardcoding tab labels or assuming a
    filter works everywhere. Must be registered before ``GET /review/{tab}``
    so it isn't shadowed as ``tab='tabs'``.

    C3: also carries ``empty_state`` -- ``{has_probe_predictions,
    has_item_scores}`` -- computed once here so a client can annotate any
    tab's zero-result state (the same reasons ``GET /review/{tab}``'s own
    ``empty_reason`` uses) without a per-tab round trip.
    """
    empty_state = await review_empty_reason.review_tabs_empty_state(opensearch)
    return {'tabs': review_queries.review_tab_catalog(), 'empty_state': empty_state}


@router.get('/review/{tab}')
async def review_queue(
    tab: Annotated[str, PathParam(description=_TAB_DESCRIPTION)],
    opensearch: OpenSearchDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 30,
    include_test: IncludeTest = False,
    text: TextQ = None,
    max_rank: MaxRankQ = None,
    min_blur_ratio: BlurQ = None,
    min_mistakenness: MistakeQ = None,
    hide_near_duplicates: NearDupQ = False,
    class_id: ClassIdQ = None,
    source: SourceQ = None,
    conf_min: ConfQ = None,
    conf_max: ConfQ = None,
    sort: SortQ = None,
    region_status: RegionStatusQ = None,
) -> dict[str, Any]:
    """Human review queue for the labeler ``/review`` page.

    Each tab maps to a deterministic OpenSearch query against the
    configured items index plus a ``reason`` string the labeler renders to
    explain why the crop landed in this queue. Items already validated by
    a human are excluded from every tab — except ``model_disagreements``,
    where validated crops are exactly the input set, and ``regions``,
    which reviews the region annotation independently of the item's
    class validation. Order: the applied sort (``sort_applied``), then
    ``crop_id``.
    """
    await _ensure_indexes(opensearch)
    filters = _filters(
        include_test,
        text,
        max_rank,
        min_blur_ratio,
        min_mistakenness,
        hide_near_duplicates,
        class_id,
        source,
        conf_min,
        conf_max,
        region_status,
    )
    guard_page_depth(page, page_size)
    req = await _request(tab, filters, sort, opensearch)
    body = {
        'from': (page - 1) * page_size,
        'size': page_size,
        'query': req.query,
        'sort': req.sort,
        # Exact totals: the default 10k cap makes large queues look smaller.
        'track_total_hits': True,
        # Never ship the 1024-d embedding vectors or class_id_history to
        # the review grid (F-25 — history is undo-only).
        '_source': {'excludes': item_list_source_excludes()},
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc

    total_obj = (resp.get('hits') or {}).get('total') or {}
    total = int(total_obj.get('value', 0))
    hits = (resp.get('hits') or {}).get('hits') or []
    registry_names: frozenset[str] = frozenset()
    if tab == 'mismatches':
        from src.services.curation.new_class_terms import normalize_term

        registry_names = frozenset(
            normalize_term(c.class_name)
            for c in get_class_registry().load().classes
            if not c.deprecated
        )
    items: list[dict[str, Any]] = []
    region_fields = get_region_fields() if tab == 'regions' else None
    for h in hits:
        src = h.get('_source') or {}
        item = serialize_item(src, h.get('_id', ''))
        # Review-only extra on top of the shared wire item.
        if tab == 'mismatches':
            item['reason'] = review_queries.mismatch_reason(src, registry_names, req.reason)
        elif tab == 'regions':
            item['reason'] = review_queries.region_reason(src, region_fields, req.reason)
        else:
            item['reason'] = req.reason
        items.append(item)
    # C3: a real-state reason a bare "Queue empty." can't tell the
    # operator -- only computed on the empty path (a few extra `count`
    # calls that only fire when there's nothing else to show anyway).
    empty_reason = (
        await review_empty_reason.compute_empty_reason(tab, filters, opensearch)
        if total == 0
        else None
    )
    return {
        'total': int(total),
        'page': page,
        'page_size': page_size,
        'items': items,
        'sort_applied': req.sort_applied,
        # Set when the default sort's field has 0% coverage and
        # sort_applied is its fallback.
        'sort_fallback_reason': req.sort_fallback_reason,
        'empty_reason': empty_reason,
    }


@router.get('/review/{tab}/locate')
async def review_locate(
    tab: Annotated[str, PathParam(description=_TAB_DESCRIPTION)],
    crop_id: Annotated[str, Query(description='The item to find.')],
    opensearch: OpenSearchDep,
    page_size: Annotated[int, Query(ge=1, le=200)] = 30,
    include_test: IncludeTest = False,
    text: TextQ = None,
    max_rank: MaxRankQ = None,
    min_blur_ratio: BlurQ = None,
    min_mistakenness: MistakeQ = None,
    hide_near_duplicates: NearDupQ = False,
    class_id: ClassIdQ = None,
    source: SourceQ = None,
    conf_min: ConfQ = None,
    conf_max: ConfQ = None,
    sort: SortQ = None,
    region_status: RegionStatusQ = None,
) -> dict[str, Any]:
    """Where ``crop_id`` sits in the queue ``GET /review/{tab}`` would serve
    for the same filters and sort.

    ``{crop_id, in_queue, rank, page, page_size, total, reason,
    sort_applied, sort_fallback_reason}``: ``rank`` is 0-based, ``page`` the 1-based page of
    ``page_size`` holding it. Out of the queue: ``rank``/``page`` null and
    ``reason`` ``not_found`` (no such item) or ``filtered_out``. Counts the
    items sorting before it, so any depth costs the same.
    """
    await _ensure_indexes(opensearch)
    filters = _filters(
        include_test,
        text,
        max_rank,
        min_blur_ratio,
        min_mistakenness,
        hide_near_duplicates,
        class_id,
        source,
        conf_min,
        conf_max,
        region_status,
    )
    req = await _request(tab, filters, sort, opensearch)
    out: dict[str, Any] = {
        'crop_id': crop_id,
        'in_queue': False,
        'rank': None,
        'page': None,
        'page_size': page_size,
        'total': None,
        'reason': None,
        'sort_applied': req.sort_applied,
        'sort_fallback_reason': req.sort_fallback_reason,
    }

    async def _count(query: dict[str, Any]) -> int:
        try:
            resp = await opensearch.count(index=CURATION_ITEMS_INDEX, body={'query': query})
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
        return int(resp.get('count', 0))

    try:
        doc = await opensearch.get(index=CURATION_ITEMS_INDEX, id=crop_id)
    except Exception as exc:
        if is_not_found(exc):
            return {**out, 'reason': 'not_found'}
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
    if not doc.get('found', True):
        return {**out, 'reason': 'not_found'}
    source_doc = {**(doc.get('_source') or {}), 'crop_id': crop_id}
    out['total'] = await _count(req.query)
    member = {'bool': {'filter': [req.query, {'term': {'crop_id': crop_id}}]}}
    if not await _count(member):
        return {**out, 'reason': 'filtered_out'}
    try:
        before = before_query(req.sort, source_doc)
    except UnlocatableSortError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    rank = await _count({'bool': {'filter': [req.query, before]}})
    return {**out, 'in_queue': True, 'rank': rank, 'page': rank // page_size + 1}


@router.post('/test_holdout/freeze', response_model=TestHoldoutFreezeResponse)
async def freeze_test_holdout(
    payload: TestHoldoutFreezeRequest,
    opensearch: OpenSearchDep,
    force: bool = Query(False, description='Allow re-running freeze (overwrites existing).'),
) -> TestHoldoutFreezeResponse:
    """Deterministic per-class sample of human-validated crops to ``test_holdout=true``.

    Cohort: ``class_validated=true AND class_source='human'`` (Appendix C
    Decision 2). Selection: SHA1-of-``crop_id`` deterministic per-class
    sampling with a 5-crop floor via :func:`select_test_holdout` — the
    single canonical holdout algorithm shared with
    an offline promotion script (Appendix C
    Decision 1). No seed (a ``seed`` field is a ``422``): the same cohort
    always freezes the same set. The response names the method in
    ``selection`` (``sha1_per_class``).

    Refuses to run if a holdout already exists unless ``force=true``.
    Refuses (422) to freeze zero rows — a freeze that freezes nothing is
    never a success, and silently returning 200 with ``sha256('')`` hid
    that this endpoint has never actually frozen anything. Persists a
    durable freeze record under ``OP_STATE_DIR/test_holdout/`` so a bad
    freeze can be diagnosed and reverted from the recorded crop-id list.
    """
    await _ensure_indexes(opensearch)
    # Refuse re-run if an existing holdout exists.
    if not force:
        try:
            existing = await opensearch.count(
                index=CURATION_ITEMS_INDEX,
                body={'query': {'term': {'test_holdout': True}}},
            )
            if (existing or {}).get('count', 0) > 0:
                raise HTTPException(
                    status_code=409,
                    detail='test holdout already exists; pass ?force=true to re-run',
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning('curation_test_holdout_count_failed', error=str(exc))

    cohort_query = build_cohort_query()
    try:
        strata = await fetch_cohort_strata(opensearch, CURATION_ITEMS_INDEX, cohort_query)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch error: {exc}') from exc

    crop_ids_by_class: dict[int, list[str]] = {}
    for bucket in strata:
        ids = bucket.get('crop_ids') or []
        if not ids:
            continue
        crop_ids_by_class.setdefault(bucket['class_id'], []).extend(ids)

    fraction = payload.percent / 100.0
    chosen_ids, per_class = select_test_holdout(crop_ids_by_class, fraction=fraction)

    if not chosen_ids:
        cohort_size = sum(len(v) for v in crop_ids_by_class.values())
        raise HTTPException(
            status_code=422,
            detail=(
                f'zero crops selected from a cohort of {cohort_size} across '
                f'{len(crop_ids_by_class)} classes; refusing to freeze nothing'
            ),
        )

    # Bulk-flag.
    bulk: list[dict[str, Any]] = []
    now = _now_iso()
    for crop_id in chosen_ids:
        bulk.append({'update': {'_index': CURATION_ITEMS_INDEX, '_id': crop_id}})
        bulk.append({'doc': {'test_holdout': True, 'updated_at': now}})
    try:
        await opensearch.bulk(body=bulk, refresh=True)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'bulk failed: {exc}') from exc

    digest = compute_holdout_sha(chosen_ids)

    persist_freeze_record(
        crop_ids=chosen_ids,
        sha=digest,
        cohort_spec={
            'query': cohort_query,
            'selection': 'sha1_per_class',
            'percent': payload.percent,
            'fraction': fraction,
            'force': force,
        },
        per_class_counts=per_class,
    )

    return TestHoldoutFreezeResponse(
        n_frozen=len(chosen_ids),
        n_classes_covered=len(per_class),
        test_holdout_sha=digest,
        per_class_counts=per_class,
        percent=payload.percent,
        min_per_class=MIN_TEST_PER_CLASS,
    )


@router.get('/test_holdout/stats')
async def test_holdout_stats(opensearch: OpenSearchDep) -> dict[str, Any]:
    """Per-class test-set counts."""
    body = {
        'size': 0,
        'query': {'term': {'test_holdout': True}},
        'aggs': {'by_class': {'terms': {'field': 'class_id', 'size': 1000}}},
        'track_total_hits': True,
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch error: {exc}') from exc
    total = (resp.get('hits') or {}).get('total', {}).get('value', 0)
    buckets = (resp.get('aggregations') or {}).get('by_class', {}).get('buckets', [])
    return {
        'total': int(total),
        # deficient: below the per-class test minimum preflight warns on.
        'by_class': [
            {**b, 'deficient': int(b.get('doc_count', 0)) < MIN_TEST_CROPS_PER_CLASS}
            for b in buckets
        ],
        'min_test_per_class': MIN_TEST_CROPS_PER_CLASS,
    }


# =============================================================================
# Export (stub — Wave 5/6)
# =============================================================================
