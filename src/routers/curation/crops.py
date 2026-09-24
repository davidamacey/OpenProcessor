"""Curation router sub-module — item (crop) browse, label, move, exclude."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import HTTPException, Query

from src.clients.occ import OCCFinalConflictError, occ_update_one
from src.clients.occ_bulk import occ_update_bulk
from src.routers.curation._common import (
    CURATION_ITEMS_INDEX,
    CropBatchLabelRequest,
    CropExcludeRequest,
    CropFlagNewClassRequest,
    CropLabelRequest,
    CropMoveRequest,
    CropsPageResponse,
    CropUnexcludeRequest,
    ItemDoc,
    OpenSearchDep,
    RegistryDep,
    _ensure_indexes,
    _now_iso,
    get_class_registry,
    guard_page_depth,
    logger,
    router,
)
from src.services.curation.cluster_ids import cluster_kind
from src.services.curation.crop_browse import (
    classifier_low_confidence_clause,
    confidence_band,
    crops_page,
    embedding_pool_query_and_count,
    parse_crop_sort,
)
from src.services.curation.human_label import (
    candidate_move_update,
    human_class_provenance,
    human_label_update,
)
from src.services.curation.item_text import item_text_query
from src.services.curation.wire import (
    item_list_source_excludes,
    item_source_excludes,
    serialize_item,
)


_MAX_IDS = 500


async def _occ_bulk_human_relabel(
    opensearch: Any,
    crop_ids: list[str],
    merger: Any,
    *,
    writer_id: str,
    max_retries: int = 5,
) -> dict[str, Any]:
    """Batch OCC relabel (F-17): one mget page + one bulk call instead of
    one ``occ_update_one`` round-trip per crop. ``refresh='wait_for'`` is
    attached only to the final bulk call of each retry round (see
    :func:`src.clients.occ_bulk.occ_update_bulk`), not per crop.

    ``current_source`` on each conflict entry is looked up with one
    follow-up ``mget`` restricted to the (normally empty) conflict set,
    to preserve the pre-existing response shape without re-adding a
    per-crop read.
    """
    status_map = await occ_update_bulk(
        opensearch,
        index=CURATION_ITEMS_INDEX,
        ids=list(crop_ids),
        merge_fn=lambda _crop_id, source: merger(source),
        max_retries=max_retries,
        refresh='wait_for',
    )
    updated_ids = [cid for cid in crop_ids if status_map.get(cid) == 'updated']
    conflict_ids = [cid for cid in crop_ids if status_map.get(cid) != 'updated']

    conflicts: list[dict[str, Any]] = []
    if conflict_ids:
        from src.clients.curation_opensearch import mget_crops

        docs = await mget_crops(
            opensearch,
            conflict_ids,
            index=CURATION_ITEMS_INDEX,
            source_includes=['class_source'],
        )
        for cid in conflict_ids:
            source = (docs.get(cid) or {}).get('_source') or {}
            conflicts.append({'crop_id': cid, 'current_source': source.get('class_source')})
            logger.warning(f'{writer_id}_update_failed', crop_id=cid)

    return {'updated': len(updated_ids), 'updated_ids': updated_ids, 'conflicts': conflicts}


@router.get('/crops', response_model=None, responses={200: {'model': CropsPageResponse}})
async def list_crops(
    opensearch: OpenSearchDep,
    # Annotated defaults (not `= Query(...)`) so direct Python callers such
    # as GET /classes/{id}/crops get real values, not FieldInfo objects.
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=500)] = 50,
    limit: Annotated[
        int | None, Query(ge=1, le=500, description='Alias for page_size; wins when both set.')
    ] = None,
    sort: Annotated[
        str | None,
        Query(
            description=(
                "'<field>[:asc|desc]', default 'updated_at:desc'. Fields: "
                'updated_at, created_at, confidence, '
                'crop_rank_in_image, crop_area_norm, blur_lap_ratio, cluster_distance, '
                'mistakenness_score, uniqueness_score. Ignored by order=outliers|diverse.'
            )
        ),
    ] = None,
    class_id: int | None = None,
    cluster_id: int | None = None,
    label_source: str | None = None,
    class_source: str | None = None,
    label_validated: bool | None = None,
    source: Annotated[str | None, Query(description='Ingest source tag (wire `source`).')] = None,
    needs_new_class: bool | None = None,
    review_dismissed: Annotated[
        bool | None, Query(description='true = only items hidden from review.')
    ] = None,
    ids: Annotated[
        str | None,
        Query(
            description=(
                'Comma-separated crop ids (max 500): return exactly these items in this '
                'order, missing ids dropped. Every other filter is ignored.'
            )
        ),
    ] = None,
    include_test: bool = False,
    include_excluded: bool = False,
    max_rank: Annotated[int | None, Query(ge=1)] = None,
    min_blur_ratio: Annotated[float | None, Query(ge=0.0)] = None,
    classifier_conf_lt: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    conf_min: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    conf_max: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    item_text: Annotated[
        str | None,
        Query(
            max_length=200,
            description=(
                'Text read on the item crop: every word must be a case-insensitive '
                'prefix of a stored item text token.'
            ),
        ),
    ] = None,
    order: Annotated[
        str,
        Query(
            description=(
                "'outliers' ranks a cluster's members farthest-from-centroid first. "
                "'diverse' ranks the matched pool by k-center-greedy coverage "
                '(gated on OP_SELECT_DIVERSE_ENABLED; behaves like an unrecognized '
                'order value when the flag is off).'
            )
        ),
    ] = 'default',
    k: Annotated[
        int | None,
        Query(ge=1, le=10_000, description='order=diverse only: rank just the first k picks.'),
    ] = None,
) -> dict[str, Any]:
    """Paginated crop browse with the standard filter set.

    ``test_holdout=true`` rows are filtered out unless ``include_test``.
    ``class_excluded=true`` (human-ignored) rows are filtered out unless
    ``include_excluded`` — set it to review the ignore bucket.

    Primary-subject filters (all default off → no behavior change):
    ``max_rank`` keeps only crops whose ``crop_rank_in_image <= max_rank``
    (e.g. 1 = largest only, 2 = largest + 2nd). ``min_blur_ratio`` keeps crops
    at or above a clarity threshold (the labeler slider); crops with no blur
    score are NOT dropped. ``classifier_conf_lt`` mines the "model wasn't sure" pool —
    crops a classifier scored below the value (``confidence``) OR that have no classifier
    prediction at all (blind spots).
    """
    await _ensure_indexes(opensearch)
    if ids is not None:
        wanted = list(dict.fromkeys(i for i in (x.strip() for x in ids.split(',')) if i))
        if len(wanted) > _MAX_IDS:
            raise HTTPException(status_code=400, detail=f'at most {_MAX_IDS} ids per request')
        found = await _crops_by_ids(opensearch, wanted)
        return crops_page(total=len(found), page=1, page_size=len(wanted), crops=found)
    if limit is not None:
        page_size = limit
    guard_page_depth(page, page_size)
    try:
        sort_clause = parse_crop_sort(sort)
        conf_clause = confidence_band(conf_min, conf_max)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # F-19: every clause below is a pure predicate (term/exists/range/
    # must_not-wrapped-term) — none score — so all of it lives in filter
    # context, not must.
    filt: list[dict[str, Any]] = []
    if conf_clause is not None:
        filt.append(conf_clause)
    if item_text is not None:
        text_clause = item_text_query(item_text)
        if text_clause is None:
            raise HTTPException(status_code=400, detail='item_text must contain a letter or digit')
        filt.append(text_clause)
    if class_id is not None:
        filt.append({'term': {'class_id': class_id}})
    if cluster_id is not None:
        filt.append({'term': {'cluster_id': cluster_id}})
    if label_source:
        filt.append({'term': {'label_source': label_source}})
    if class_source:
        # class_source is mapped keyword directly on the live index — no
        # .keyword subfield exists.
        filt.append({'term': {'class_source': class_source}})
    if label_validated is not None:
        # Legacy query param maps to class_validated (the class-side flag —
        # the common case for the labeler /clusters filter).
        filt.append({'term': {'class_validated': label_validated}})
    if source:
        filt.append({'term': {'source': source}})
    if review_dismissed is not None:
        dismissed: dict[str, Any] = {'exists': {'field': 'review_dismissed_at'}}
        filt.append(dismissed if review_dismissed else {'bool': {'must_not': dismissed}})
    if needs_new_class is not None:
        clause: dict[str, Any] = {'term': {'needs_new_class': True}}
        filt.append(clause if needs_new_class else {'bool': {'must_not': clause}})
    if not include_test:
        filt.append({'bool': {'must_not': {'term': {'test_holdout': True}}}})
    if not include_excluded:
        filt.append({'bool': {'must_not': {'term': {'class_excluded': True}}}})
    if max_rank is not None:
        filt.append({'range': {'crop_rank_in_image': {'lte': max_rank}}})
    if min_blur_ratio is not None:
        # Crops with no blur score must NOT be hidden by the slider — only
        # exclude crops that have a score and fall below it.
        filt.append(
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
    if classifier_conf_lt is not None:
        filt.append(classifier_low_confidence_clause(classifier_conf_lt))

    bool_q: dict[str, Any] = {}
    if filt:
        bool_q['filter'] = filt
    query_clause: dict[str, Any] = {'bool': bool_q} if bool_q else {'match_all': {}}
    body = {
        'from': (page - 1) * page_size,
        'size': page_size,
        'query': query_clause,
        'sort': sort_clause,
        # Exact total (not the default 10k cap) so the labeler shows real
        # queue sizes for filtered views — one count pass per query, fine at
        # this scale and matches the /curation/review endpoint.
        'track_total_hits': True,
        # Never ship the 1024-d embedding vectors or class_id_history
        # to the card grid (F-25 -- history is undo-only).
        '_source': {'excludes': item_list_source_excludes()},
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
    total = (resp.get('hits') or {}).get('total', {}).get('value', 0)
    # Outlier ordering: rank this cluster's members by distance from their
    # centroid (most atypical first). Computed on-the-fly + cached; falls
    # through to the default newest-first sort if too large / no embeddings.
    if order == 'outliers' and cluster_id is not None:
        from src.services.curation.clustering.outliers import (
            OUTLIER_EMBEDDING_FIELD,
            compute_outlier_order,
        )

        # F-16: pool query + exact count scoped to the embedding-bearing subset.
        pool_query, pool_count = await embedding_pool_query_and_count(
            opensearch, CURATION_ITEMS_INDEX, query_clause, OUTLIER_EMBEDDING_FIELD
        )
        ordered_ids = await compute_outlier_order(
            opensearch, CURATION_ITEMS_INDEX, pool_query, current_count=pool_count
        )
        if ordered_ids is not None:
            page_ids = ordered_ids[(page - 1) * page_size : (page - 1) * page_size + page_size]
            crops = await _crops_by_ids(opensearch, page_ids)
            return crops_page(
                total=len(ordered_ids),
                page=page,
                page_size=page_size,
                crops=crops,
                method='outliers',
                n_pool=int(total),
            )

    # Diversity ordering: k-center-greedy coverage over the matched pool,
    # same fallback contract as 'outliers' above.
    if order == 'diverse':
        from src.routers.curation.select import EMBEDDING_FIELD, compute_diverse_order

        pool_query, pool_count = await embedding_pool_query_and_count(
            opensearch, CURATION_ITEMS_INDEX, query_clause, EMBEDDING_FIELD
        )
        diverse_ids = await compute_diverse_order(
            opensearch, CURATION_ITEMS_INDEX, pool_query, current_count=pool_count, k=k
        )
        if diverse_ids is not None:
            page_ids = diverse_ids[(page - 1) * page_size : (page - 1) * page_size + page_size]
            crops = await _crops_by_ids(opensearch, page_ids)
            return crops_page(
                total=len(diverse_ids),
                page=page,
                page_size=page_size,
                crops=crops,
                method='diverse',
                n_pool=int(total),
            )

    hits = (resp.get('hits') or {}).get('hits') or []
    crops = [serialize_item(h.get('_source') or {}, h.get('_id', '')) for h in hits]
    return crops_page(total=int(total), page=page, page_size=page_size, crops=crops)


async def _crops_by_ids(opensearch: Any, ids: list[str]) -> list[dict[str, Any]]:
    """mget item docs preserving the supplied id order (drops missing)."""
    if not ids:
        return []
    resp = await opensearch.mget(
        index=CURATION_ITEMS_INDEX,
        body={'ids': ids},
        _source_excludes=item_list_source_excludes(),
    )
    return [
        serialize_item(d.get('_source') or {}, d.get('_id', ''))
        for d in (resp.get('docs') or [])
        if d.get('found')
    ]


@router.get('/crops/{crop_id}', response_model=None, responses={200: {'model': ItemDoc}})
async def get_crop(
    crop_id: str,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Return the authoritative item document by id.

    Used by the labeler's review-queue "Back" path so the operator sees
    what was actually persisted (not a stale local snapshot). Same wire
    item as ``GET /crops`` and ``GET /review/{tab}`` — never the raw
    OpenSearch ``_source``, whose region keys follow ``RegionFields``
    storage names.
    """
    try:
        # F-25: still excludes the 1024-d embedding vectors (matching
        # every other item endpoint's behavior) -- but not
        # class_id_history, since a single-item view may legitimately
        # want it, unlike a paginated list.
        resp = await opensearch.get(
            index=CURATION_ITEMS_INDEX, id=crop_id, _source_excludes=item_source_excludes()
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc
    src = (resp.get('_source') or {}) if isinstance(resp, dict) else {}
    return serialize_item(src, crop_id)


@router.put('/crops/{crop_id}/label')
async def label_crop(
    crop_id: str,
    payload: CropLabelRequest,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
) -> dict[str, Any]:
    """Set the validated class on a single item."""
    reg = registry
    if not reg.validate_id(payload.class_id):
        raise HTTPException(status_code=400, detail=f'unknown class_id {payload.class_id}')
    entry = reg.get(payload.class_id)
    class_name = entry.class_name if entry is not None else ''

    def _merge_label(current: dict[str, Any]) -> dict[str, Any]:
        return human_label_update(
            current,
            class_id=payload.class_id,
            class_name=class_name,
            label_source=payload.label_source,
            writer='human:label_crop',
        )

    try:
        await occ_update_one(
            opensearch,
            doc_id=crop_id,
            merger=_merge_label,
            refresh='wait_for',
            writer_id='human:label_crop',
        )
    except OCCFinalConflictError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc
    return {'crop_id': crop_id, 'class_id': payload.class_id, 'class_name': class_name}


@router.put('/crops/batch_label')
async def batch_label_crops(
    payload: CropBatchLabelRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Bulk label many crops with a single class_id.

    Each crop is updated via ``occ_update_one`` so a concurrent worker
    write can't silently clobber the human edit and the prior class
    assignment is snapshotted into ``class_id_history``. Returns
    ``{updated, conflicts: [{crop_id, current_source}]}`` — conflicts
    list crops whose OCC retries were exhausted.
    """
    reg = get_class_registry()
    if not reg.validate_id(payload.class_id):
        raise HTTPException(status_code=400, detail=f'unknown class_id {payload.class_id}')
    entry = reg.get(payload.class_id)
    class_name = entry.class_name if entry is not None else ''

    if not payload.crop_ids:
        return {'updated': 0, 'updated_ids': [], 'conflicts': []}

    def _merge(current: dict[str, Any]) -> dict[str, Any]:
        return human_label_update(
            current,
            class_id=payload.class_id,
            class_name=class_name,
            label_source=payload.label_source,
            writer='human:batch_label_crops',
        )

    # F-17: one mget page + one bulk call (regardless of batch size) via
    # occ_update_bulk, instead of one occ_update_one round-trip per crop.
    return await _occ_bulk_human_relabel(
        opensearch,
        payload.crop_ids,
        _merge,
        writer_id='human:batch_label_crops',
    )


@router.post('/crops/move')
async def move_crops(
    payload: CropMoveRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Move crops to a different cluster.

    * Class cluster (``cluster_id == class_id``): also a relabel — the
      crops get that registry class (``class_source='human_move'``,
      ``class_validated=True``): "move this to the X cluster" means "this
      is an X".
    * Candidate cluster: placement only. The crops join the group; a
      human-owned class is cleared (the human just said it isn't that
      class), a machine suggestion is kept, and nothing is validated —
      the group has no class until one is assigned.
    * ``400`` for an unassigned (negative) target (use exclude/discard)
      a class-range id that isn't in the registry, or a candidate id with
      no current members (it doesn't exist). Nothing is written.
    """
    target_id = int(payload.cluster_id)
    kind = cluster_kind(target_id)
    target = None
    if kind == 'unassigned':
        raise HTTPException(
            status_code=400,
            detail=f'cannot move into unassigned cluster {target_id}; use exclude or discard',
        )
    if kind == 'class':
        target = get_class_registry().get(target_id)
        if target is None:
            raise HTTPException(status_code=400, detail=f'unknown class_id {target_id}')
    if kind == 'candidate':
        from src.services.curation.exclusion import cluster_member_count

        if await cluster_member_count(opensearch, CURATION_ITEMS_INDEX, target_id) == 0:
            raise HTTPException(
                status_code=400, detail=f'candidate cluster {target_id} has no members'
            )
    if not payload.crop_ids:
        return {'updated': 0, 'updated_ids': [], 'conflicts': []}

    from src.services.curation.history import record_class_snapshot

    target_name = target.class_name if target is not None else ''

    def _merge(current: dict[str, Any]) -> dict[str, Any]:
        if kind == 'candidate':
            return candidate_move_update(current, cluster_id=target_id, now=_now_iso())
        history = record_class_snapshot(current, writer='human:move_crops', restorable=True)
        return {
            'cluster_id': target_id,
            'class_id': target_id,
            'class_name': target_name,
            'class_source': 'human_move',
            # Move-from-cluster is a class gesture.
            'class_validated': True,
            'label_source': 'human',
            'class_id_history': history,
            # See label_crop above — subid only applies inside the crop's
            # original cluster; clear on move.
            'cluster_subid': None,
            **human_class_provenance(),
            'updated_at': _now_iso(),
        }

    # F-17: batched via occ_update_bulk — see batch_label_crops above.
    return await _occ_bulk_human_relabel(
        opensearch,
        payload.crop_ids,
        _merge,
        writer_id='human:move_crops',
    )


@router.post('/crops/flag_new_class')
async def flag_new_class(
    payload: CropFlagNewClassRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Flag crops as needing a new class label (curator queue).

    Source imagery can contain object types beyond the current class
    registry. Labelers flag a crop; a curator later batch-reviews the
    queue, decides if a new class warrants creation via
    ``POST /curation/classes``, and labels the flagged crops.
    """
    if not payload.crop_ids:
        return {'updated': 0}
    now = _now_iso()
    bulk: list[dict[str, Any]] = []
    for crop_id in payload.crop_ids:
        bulk.append({'update': {'_index': CURATION_ITEMS_INDEX, '_id': crop_id}})
        bulk.append(
            {
                'doc': {
                    'needs_new_class': True,
                    'needs_new_class_note': payload.note,
                    'needs_new_class_at': now,
                    'updated_at': now,
                }
            }
        )
    try:
        resp = await opensearch.bulk(body=bulk, refresh=False)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch error: {exc}') from exc
    n_errors = sum(1 for it in resp.get('items', []) if any('error' in v for v in it.values()))
    return {'flagged': len(payload.crop_ids) - n_errors, 'errors': n_errors}


@router.post('/crops/batch_exclude')
async def batch_exclude_crops(
    payload: CropExcludeRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Exclude crops from training + clustering (reversible).

    Sets ``class_excluded=true`` plus provenance (``excluded_at``,
    ``excluded_by``, ``excluded_reason``) and moves the crop out of its
    candidate bucket (``cluster_id=-2``, the excluded sentinel). The
    residual-pool fetch, the trainer export, the cluster listing, and
    the default crop browse all filter ``class_excluded=true`` out, so
    an excluded crop never re-appears in a cluster or training set until
    it's un-excluded.

    Non-destructive: the crop document stays in OpenSearch for audit and
    provenance, and the pre-exclusion validation + cluster placement are
    recorded (``excluded_prior_*``) so un-exclude can restore them.
    ``reason`` defaults to ``'ignore'``; pass a tag like
    ``'blurry'`` to record why (e.g. a whole cluster of blurry items).
    """
    if not payload.crop_ids:
        return {'excluded': 0, 'errors': 0}
    from src.services.curation.exclusion import exclusion_update

    now = _now_iso()
    reason = payload.reason or 'ignore'
    n_errors = await _occ_bulk_human_write(
        opensearch,
        payload.crop_ids,
        lambda _id, cur: exclusion_update(cur, reason=reason, now=now),
        writer_id='human:batch_exclude_crops',
    )
    return {'excluded': len(payload.crop_ids) - n_errors, 'errors': n_errors}


async def _occ_bulk_human_write(
    opensearch: Any,
    crop_ids: list[str],
    merger: Any,
    *,
    writer_id: str,
) -> int:
    """Read-modify-write ``crop_ids`` via OCC bulk; return the failure count.

    A version conflict (a concurrent write landed between read and write)
    counts as a failure the caller reports, never a silent success.
    """
    from src.clients.occ import occ_skip_on_conflict_bulk

    try:
        resp = await occ_skip_on_conflict_bulk(
            opensearch,
            doc_ids=list(crop_ids),
            merger=merger,
            index=CURATION_ITEMS_INDEX,
            refresh='wait_for',
            writer_id=writer_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch error: {exc}') from exc
    return len(resp.get('errors') or []) + int(resp.get('skipped_due_to_conflict') or 0)


@router.post('/crops/batch_unexclude')
async def batch_unexclude_crops(
    payload: CropUnexcludeRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Reverse an exclusion (Undo path for Ignore).

    Clears ``class_excluded`` + provenance and restores the validation
    recorded at exclude time. A validated crop goes straight back to its
    class cluster (``cluster_id == class_id``); an unvalidated one returns
    to the candidate cluster it was excluded from while that cluster still
    has members, else drops to the residual pool (``cluster_id=null``) for
    a fresh assignment on the next recluster. Crops that aren't excluded
    are left untouched.
    """
    if not payload.crop_ids:
        return {'unexcluded': 0, 'errors': 0}
    from src.services.curation.exclusion import live_candidate_ids, unexclusion_update

    try:
        live = await live_candidate_ids(opensearch, CURATION_ITEMS_INDEX, payload.crop_ids)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch error: {exc}') from exc
    now = _now_iso()
    n_errors = await _occ_bulk_human_write(
        opensearch,
        payload.crop_ids,
        lambda _id, cur: unexclusion_update(cur, now=now, live_candidate_ids=live),
        writer_id='human:batch_unexclude_crops',
    )
    return {'unexcluded': len(payload.crop_ids) - n_errors, 'errors': n_errors}


@router.post('/crops/{crop_id}/review_dismiss')
async def review_dismiss_crop(crop_id: str, opensearch: OpenSearchDep) -> dict[str, Any]:
    """Permanently dismiss a crop from every /review queue.

    Stamps ``review_dismissed_at`` + ``review_dismissed_by``. The
    review_queue handler excludes any crop where ``review_dismissed_at``
    exists, so this is one-way: once an operator says 'I never want to
    see this again in review', it's gone from every tab. The crop's
    underlying class / region state is left intact — only the review
    visibility changes.
    """
    body = {
        'doc': {
            'review_dismissed_at': _now_iso(),
            'review_dismissed_by': 'human',
            'updated_at': _now_iso(),
        }
    }
    try:
        await opensearch.update(
            index=CURATION_ITEMS_INDEX,
            id=crop_id,
            body=body,
            refresh='wait_for',
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc
    return {'crop_id': crop_id, 'dismissed': True}
