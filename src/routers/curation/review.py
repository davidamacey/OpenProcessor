"""Curation review-queue router — ``GET/POST /curation/review/*``."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import HTTPException, Path as PathParam, Query

from src.config.region_fields import get_region_fields
from src.routers.curation._common import (
    CURATION_ITEMS_INDEX,
    OpenSearchDep,
    TestHoldoutFreezeRequest,
    TestHoldoutFreezeResponse,
    _ensure_indexes,
    _now_iso,
    logger,
    router,
)
from src.services.curation import review_queries, review_sorts
from src.services.curation.holdout import (
    build_cohort_query,
    compute_holdout_sha,
    fetch_cohort_strata,
    persist_freeze_record,
    select_test_holdout,
)


@router.get('/review/unmatched_terms')
async def review_unmatched_terms(
    opensearch: OpenSearchDep,
    size: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    """Aggregate Gemma's raw labels across every ``gemma_unmatched`` crop.

    Returns the top-N most common raw labels Gemma produced for crops the
    registry could not resolve. This is the main input to growing the
    registry: high-count labels are obvious candidates for new
    :py:class:`LegacyClassEntry` entries (or new ``SYNONYMS`` mappings if the
    raw label is just a phrasing of an existing class).

    Pair with an offline registry-reclassification script (see ``scripts/``)
    once the registry has been updated to convert matched crops from
    ``class_source='gemma_unmatched'`` to ``class_source='gemma_reclassified'``.

    Args:
        size: Maximum number of distinct raw labels to return. Capped at
            1000; default 100 (the long tail past 100 is usually noise).

    Returns:
        ``{"total_unmatched": <int>, "top_terms": [{"label": str, "count": int}, ...]}``
    """
    await _ensure_indexes(opensearch)
    body = {
        'size': 0,
        'query': {'term': {'class_source': 'gemma_unmatched'}},
        'aggs': {
            'top_raw': {
                'terms': {
                    'field': 'gemma_raw_label',
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
    """Top-N hierarchical clusters of ``gemma_raw_label`` for the labeler UI.

    Task #91 — surfaces the output of
    an offline raw-label clustering script so the labeler can:

    1. Show fine-grained sub-classes the registry doesn't have yet
       (e.g. "ford f150" + "ford ranger" rolled up under
       ``ford_pickup`` → suggested registry parent ``pickup``).
    2. Surface candidate registry promotions ranked by crop volume.
    3. Let curators bulk-relabel a whole cluster at once instead of
       clicking through individual ``gemma_unmatched`` crops.

    The endpoint runs purely against the configured items index — no clustering
    happens here, only aggregation. Cluster ids / names are written by
    the offline script; if no crops have ``gemma_label_cluster_id`` yet
    the endpoint returns an empty list with a populated ``hint``.

    NOTE on route ordering: this endpoint MUST be declared before the
    parameterised ``/review/{tab}`` below — FastAPI dispatches the first
    matching route, and ``raw_label_clusters`` would otherwise be eaten
    by the ``{tab}`` matcher.

    Args:
        size: Number of clusters to return (default 50).
        samples_per_cluster: How many ``gemma_raw_label`` samples to
            include per cluster (default 5 — enough to read at a glance).

    Returns:
        ``{"status": "ok|empty", "clusters": [{cluster_id, cluster_name,
        n_crops, n_unmatched, sample_terms, parent_class_suggestion}, ...]}``
    """
    await _ensure_indexes(opensearch)
    body = {
        'size': 0,
        'query': {'exists': {'field': 'gemma_label_cluster_id'}},
        'aggs': {
            'by_cluster': {
                'terms': {
                    'field': 'gemma_label_cluster_id',
                    'size': size,
                    'order': {'_count': 'desc'},
                },
                'aggs': {
                    'name': {'terms': {'field': 'gemma_label_cluster_name', 'size': 1}},
                    'samples': {
                        'terms': {
                            'field': 'gemma_raw_label',
                            'size': samples_per_cluster,
                        }
                    },
                    'unmatched': {'filter': {'term': {'class_source': 'gemma_unmatched'}}},
                    # Most common already-resolved class within the cluster — used
                    # as the ``parent_class_suggestion`` hint. If the cluster is
                    # 100% gemma_unmatched the bucket is empty and we return None.
                    'parent': {
                        # ``class_name`` is mapped ``keyword`` directly on the
                        # live index — no ``.keyword`` subfield exists (the
                        # migration that assumed a ``text`` mapping has been
                        # failing silently on every startup; see
                        # legacy_clusters.py's top_class agg for the full story).
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
            'Run the offline raw-label clustering script to populate '
            'gemma_label_cluster_id on crops if status=empty.'
        )
        if not clusters
        else None,
    }


@router.get('/review/{tab}')
async def review_queue(
    tab: Annotated[
        str,
        PathParam(
            description=(
                'One of: all | mismatches | gemma_low_conf | outliers | '
                'uncertainty | model_disagreements | plates | '
                'primary_low_conf | coco_blind_spots'
            )
        ),
    ],
    opensearch: OpenSearchDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=200),
    include_test: bool = False,
    text: str | None = Query(
        None,
        description=(
            'Plate-tab only: case-insensitive substring search on '
            'plate_text. Ignored on other tabs.'
        ),
    ),
    # max_rank: keep crop_rank_in_image <= this (primary tabs default 2).
    # min_blur_ratio: clarity slider (null-safe). Both apply across tabs.
    max_rank: int | None = Query(None, ge=1),
    min_blur_ratio: float | None = Query(None, ge=0.0),
    # min_mistakenness: null-safe floor on mistakenness_score, same
    # missing-field-stays-visible pattern as min_blur_ratio above.
    min_mistakenness: float | None = Query(None, ge=0.0),
    # hide_near_duplicates: drops crops a near-dup scoring pass explicitly
    # marked as a non-representative duplicate (dup_is_representative ==
    # False). No-op wherever the field hasn't been backfilled yet — never
    # hides a crop just because near-dup scoring hasn't run on it.
    hide_near_duplicates: bool = False,
    sort: str | None = Query(
        None,
        description=(
            'Review-sort strategy id from GET /curation/methods (axis=sort). Omitted '
            "or 'default' uses this tab's legacy default sort. An unknown, "
            'shadow, or disabled id returns 400.'
        ),
    ),
) -> dict[str, Any]:
    """Human review queue for the labeler ``/review`` page.

    Each tab maps to a deterministic OpenSearch query against the
    configured items index plus a ``reason`` string the labeler renders to
    explain why the crop landed in this queue. Items already validated by
    a human are excluded from every tab — except ``model_disagreements``,
    where validated crops are exactly the input set (we want to know
    where the new model thinks the human was wrong).
    """
    await _ensure_indexes(opensearch)
    fields = get_region_fields()

    # Per-tab must/must_not/reason construction lives in review_queries.py
    # (split out so this file stays under the 700-LOC pre-commit ceiling —
    # see that module's docstring). Raises HTTPException(400) for an
    # unrecognized tab, same as before the split.
    must, must_not, reason = review_queries.build_tab_query(
        tab, include_test=include_test, text=text, max_rank=max_rank
    )

    # Clarity slider — applies to every tab when set. Null-safe: crops with
    # no blur score are never hidden by the slider.
    if min_blur_ratio is not None:
        must.append(
            {
                'bool': {
                    'should': [
                        {'range': {'blur_lap_ratio': {'gte': min_blur_ratio}}},
                        {'bool': {'must_not': {'exists': {'field': 'blur_lap_ratio'}}}},
                    ],
                    'minimum_should_match': 1,
                }
            }
        )

    # Mistakenness floor — same null-safe pattern as min_blur_ratio: a crop
    # with no mistakenness_score yet (not backfilled) is never hidden by
    # the filter, only crops scored below the floor are.
    if min_mistakenness is not None:
        must.append(
            {
                'bool': {
                    'should': [
                        {'range': {'mistakenness_score': {'gte': min_mistakenness}}},
                        {'bool': {'must_not': {'exists': {'field': 'mistakenness_score'}}}},
                    ],
                    'minimum_should_match': 1,
                }
            }
        )

    # Near-dup filter — hides only crops a scoring pass explicitly marked
    # as a non-representative duplicate (dup_is_representative == False).
    # No-op wherever the field hasn't been backfilled: a crop with no
    # dup_is_representative at all is never hidden.
    if hide_near_duplicates:
        must.append(
            {
                'bool': {
                    'should': [
                        {'term': {'dup_is_representative': True}},
                        {'bool': {'must_not': {'exists': {'field': 'dup_is_representative'}}}},
                    ],
                    'minimum_should_match': 1,
                }
            }
        )

    try:
        sort_clause, sort_applied, sort_fallback_reason = review_sorts.build_sort(sort, tab=tab)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    body = {
        'from': (page - 1) * page_size,
        'size': page_size,
        'query': {'bool': {'must': must, 'must_not': must_not}},
        'sort': sort_clause,
        # OpenSearch caps hits.total.value at 10000 by default. The
        # labeler displays the queue total in the page header; capping
        # at 10k makes large queues look smaller than they are. Cost is
        # one extra count pass per search — fine at typical deployment QPS.
        'track_total_hits': True,
        # Never ship the 1024-d embedding vectors to the review grid.
        '_source': {'excludes': ['pe_embedding', 'v6_embedding']},
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc

    total_obj = (resp.get('hits') or {}).get('total') or {}
    total = int(total_obj.get('value', 0))
    hits = (resp.get('hits') or {}).get('hits') or []
    items: list[dict[str, Any]] = []
    for h in hits:
        src = h.get('_source') or {}
        crop_id = src.get('crop_id') or h.get('_id', '')
        proposed_id = src.get('gemma_proposed_class_id') or src.get('class_id')
        proposed_name = (
            src.get('gemma_proposed_class')
            or src.get('gemma_raw_class')
            or src.get('class_name')
            or ''
        )
        items.append(
            {
                # LegacyCrop fields the labeler ReviewItem extends.
                'id': crop_id,
                'crop_id': crop_id,
                'source_image_path': src.get('image_path', ''),
                'image_path': src.get('image_path', ''),
                'bbox_norm': src.get('bbox_norm') or [],
                'class_id': src.get('class_id'),
                'class_name': src.get('class_name', ''),
                'class_source': src.get('class_source', ''),
                'confidence': float(src.get('confidence') or 0.0),
                # Categorical Gemma confidence (high/medium/low) shown alongside
                # the numeric v6 confidence — labeler renders "v6: 95.9 %,
                # gemma: medium" so the rows aren't ambiguous.
                'gemma_confidence': src.get('gemma_confidence'),
                'label_source': src.get('label_source', ''),
                # Plan §1.3, A-PR2: legacy label_validated derived; expose
                # the split fields directly so Slice C can migrate.
                'label_validated': bool(
                    src.get('class_validated')
                    or src.get(fields.validated)
                    or src.get('label_validated', False)
                ),
                'class_validated': bool(src.get('class_validated', False)),
                fields.validated: bool(src.get(fields.validated, False)),
                'cluster_id': src.get('cluster_id'),
                'cluster_distance': src.get('cluster_distance'),
                fields.bbox_norm: src.get(fields.bbox_norm),
                fields.score: src.get(fields.score),
                'test_holdout': bool(src.get('test_holdout', False)),
                # Primary-subject rank + blur + COCO hint for the new tabs.
                'crop_rank_in_image': src.get('crop_rank_in_image'),
                'crop_area_norm': src.get('crop_area_norm'),
                'blur_lap_ratio': src.get('blur_lap_ratio'),
                'v6_raw_confidence': src.get('v6_raw_confidence'),
                'coco_proposal_name': src.get('coco_proposal_name'),
                'updated_at': src.get('updated_at', ''),
                'thumbnail_url': f'/curation/crops/{crop_id}/thumbnail',
                # ReviewItem extras.
                'reason': reason,
                'proposed_class_id': proposed_id,
                'proposed_class_name': proposed_name,
                # Phase 5 active-learning loop: surfaces in the
                # model_disagreements tab so the user sees what the new
                # model thought (and how confident it was).
                'probe_pred_class': src.get('probe_pred_class'),
                'probe_pred_entropy': src.get('probe_pred_entropy'),
                # Region-detection outputs — needed by the `plates` review tab
                # so the labeler can render the bbox on the source image
                # for human confirmation.
                fields.status: src.get(fields.status),
                fields.verified: src.get(fields.verified),
                # Region provenance (Wave 1) — labeler chips render which
                # detector + verifier produced the stored bbox.
                fields.detector: src.get(fields.detector),
                fields.detector_version: src.get(fields.detector_version),
                fields.detector_chain: src.get(fields.detector_chain),
                fields.bbox_frame: src.get(fields.bbox_frame),
                fields.detected_at: src.get(fields.detected_at),
                fields.verifier: src.get(fields.verifier),
                fields.verifier_version: src.get(fields.verifier_version),
                fields.verified_at: src.get(fields.verified_at),
                fields.rejection_reason: src.get(fields.rejection_reason),
                fields.visible: src.get(fields.visible),
                # Region OCR (Wave 2b — fields may be absent until that
                # phase ships; pass through unconditionally).
                fields.text: src.get(fields.text),
                fields.text_raw: src.get(fields.text_raw),
                fields.text_source: src.get(fields.text_source),
                fields.text_confidence: src.get(fields.text_confidence),
                fields.text_engine_version: src.get(fields.text_engine_version),
                # Class provenance (Wave 1).
                'class_detector': src.get('class_detector'),
                'class_detector_version': src.get('class_detector_version'),
                'class_labeled_at': src.get('class_labeled_at'),
                'class_labeler': src.get('class_labeler'),
                # Curation-score overlays (Phase 3, review_sorts.py) — pass
                # through unconditionally; absent on any crop no scoring
                # job has touched yet.
                'mistakenness_score': src.get('mistakenness_score'),
                'uniqueness_score': src.get('uniqueness_score'),
                'dup_group_id': src.get('dup_group_id'),
                'dup_group_size': src.get('dup_group_size'),
                'dup_is_representative': src.get('dup_is_representative'),
            }
        )
    return {
        'total': int(total),
        'page': page,
        'page_size': page_size,
        'items': items,
        # Phase 3 (review_sorts.py) — additive envelope fields, never
        # removes/retypes an existing key. sort_fallback_reason is always
        # None today; reserved for a future graceful-degradation case.
        'sort_applied': sort_applied,
        'sort_fallback_reason': sort_fallback_reason,
    }


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
    Decision 1). No seed: the same cohort always freezes the same set.

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
            logger.warning('legacy_test_holdout_count_failed', error=str(exc))

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
    )


@router.get('/test_holdout/stats')
async def test_holdout_stats(opensearch: OpenSearchDep) -> dict[str, Any]:
    """Per-class test-set counts."""
    body = {
        'size': 0,
        'query': {'term': {'test_holdout': True}},
        'aggs': {'by_class': {'terms': {'field': 'class_id', 'size': 1000}}},
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch error: {exc}') from exc
    total = (resp.get('hits') or {}).get('total', {}).get('value', 0)
    return {
        'total': int(total),
        'by_class': (resp.get('aggregations') or {}).get('by_class', {}).get('buckets', []),
    }


# =============================================================================
# Export (stub — Wave 5/6)
# =============================================================================
