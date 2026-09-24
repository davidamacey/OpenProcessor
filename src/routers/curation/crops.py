"""Curation router sub-module — item (crop) browse, label, move, exclude."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import HTTPException, Query

from src.clients.occ import OCCFinalConflictError, occ_update_one
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
    logger,
    router,
)
from src.services.curation.crop_browse import confidence_band, crops_page, parse_crop_sort
from src.services.curation.ingest_class_sources import HUMAN_CLASS_SOURCE
from src.services.curation.wire import item_source_excludes, serialize_item
from src.services.detection.cascade_detect import class_provenance


def _human_class_provenance() -> dict[str, Any]:
    """Class provenance for every human class write in this module."""
    from src.services.detection.profile_registry import region_profile_or_neutral

    human = region_profile_or_neutral()
    return class_provenance(
        detector=human.human_detector_name,
        detector_version=human.human_detector_version,
        labeler='human',
    )


_MAX_IDS = 500


def _batch_outcome(
    crop_ids: list[str], results: list[tuple[bool, dict[str, Any] | None]]
) -> dict[str, Any]:
    """``updated_ids`` lists exactly the crops written (the ones an undo of
    this batch should pass); ``conflicts`` the ones that were not."""
    updated_ids = [cid for cid, (ok, _c) in zip(crop_ids, results, strict=True) if ok]
    conflicts = [c for ok, c in results if not ok and c is not None]
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
                'updated_at, created_at, confidence, classifier_raw_confidence, '
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
    hdd_source: str | None = None,
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
    crops whose ``classifier_raw_confidence`` is below the value OR that have no classifier
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
    try:
        sort_clause = parse_crop_sort(sort)
        conf_clause = confidence_band(conf_min, conf_max)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    must: list[dict[str, Any]] = []
    # Filter-context clauses (cached bitsets, no scoring) for the new
    # primary-subject filters.
    filt: list[dict[str, Any]] = []
    if conf_clause is not None:
        filt.append(conf_clause)
    if class_id is not None:
        must.append({'term': {'class_id': class_id}})
    if cluster_id is not None:
        must.append({'term': {'cluster_id': cluster_id}})
    if label_source:
        must.append({'term': {'label_source': label_source}})
    if class_source:
        # class_source is mapped keyword directly on the live index — no
        # .keyword subfield exists.
        must.append({'term': {'class_source': class_source}})
    if label_validated is not None:
        # Legacy query param maps to class_validated (the class-side flag —
        # the common case for the labeler /clusters filter).
        must.append({'term': {'class_validated': label_validated}})
    if source or hdd_source:
        must.append({'term': {'hdd_source': source or hdd_source}})
    if review_dismissed is not None:
        dismissed: dict[str, Any] = {'exists': {'field': 'review_dismissed_at'}}
        must.append(dismissed if review_dismissed else {'bool': {'must_not': dismissed}})
    if needs_new_class is not None:
        clause: dict[str, Any] = {'term': {'needs_new_class': True}}
        must.append(clause if needs_new_class else {'bool': {'must_not': clause}})
    if not include_test:
        must.append({'bool': {'must_not': {'term': {'test_holdout': True}}}})
    if not include_excluded:
        must.append({'bool': {'must_not': {'term': {'class_excluded': True}}}})
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
        # Low-confidence band OR no v6 prediction at all (COCO/VLM-only
        # blind spots) — not silently dropped by a plain range clause.
        filt.append(
            {
                'bool': {
                    'should': [
                        {'range': {'classifier_raw_confidence': {'lt': classifier_conf_lt}}},
                        {'bool': {'must_not': {'exists': {'field': 'classifier_raw_confidence'}}}},
                    ],
                    'minimum_should_match': 1,
                }
            }
        )

    bool_q: dict[str, Any] = {}
    if must:
        bool_q['must'] = must
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
        # Never ship the 1024-d embedding vectors to the card grid.
        '_source': {'excludes': item_source_excludes()},
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
    total = (resp.get('hits') or {}).get('total', {}).get('value', 0)
    # Outlier ordering: rank this cluster's members by distance from their
    # centroid (most atypical first) so operators can cherry-pick the worst
    # offenders. Computed on-the-fly + cached; falls through to the default
    # newest-first sort if the cluster is too large or has no embeddings.
    if order == 'outliers' and cluster_id is not None:
        from src.services.curation.clustering.outliers import compute_outlier_order

        ordered_ids = await compute_outlier_order(
            opensearch, CURATION_ITEMS_INDEX, query_clause, current_count=int(total)
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
    # same fallback contract as 'outliers' above — None means "disabled, or
    # pool too large for an inline full-pool ranking", and the caller falls
    # back to the default newest-first sort computed below.
    if order == 'diverse':
        from src.routers.curation.select import compute_diverse_order

        diverse_ids = await compute_diverse_order(
            opensearch, CURATION_ITEMS_INDEX, query_clause, current_count=int(total), k=k
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
        _source_excludes=item_source_excludes(),
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
        resp = await opensearch.get(index=CURATION_ITEMS_INDEX, id=crop_id)
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
    from src.services.curation.history import record_class_snapshot

    def _merge_label(current: dict[str, Any]) -> dict[str, Any]:
        history = record_class_snapshot(current, writer='human:label_crop', restorable=True)
        return {
            'class_id': payload.class_id,
            'class_name': class_name,
            'class_source': HUMAN_CLASS_SOURCE,
            # Human class label. Sets class_validated; the region-side
            # validated flag is independent and unaffected.
            'class_validated': True,
            'label_source': payload.label_source,
            'class_id_history': history,
            # cluster_id mirrors class_id in the default ensemble — without
            # this the relabeled crop stays visually in its old cluster
            # bucket on the next page reload, even though its class is now
            # different.
            'cluster_id': payload.class_id,
            # cluster_subid is only meaningful within its origin cluster.
            # A class change moves the crop to a new bucket; the prior
            # AHC sub-cluster grouping no longer applies.
            'cluster_subid': None,
            **_human_class_provenance(),
            'updated_at': _now_iso(),
        }

    try:
        await occ_update_one(
            opensearch,
            doc_id=crop_id,
            merger=_merge_label,
            refresh=True,
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

    from src.services.curation.history import record_class_snapshot

    def _merge(current: dict[str, Any]) -> dict[str, Any]:
        history = record_class_snapshot(current, writer='human:batch_label_crops', restorable=True)
        return {
            'class_id': payload.class_id,
            'class_name': class_name,
            'class_source': HUMAN_CLASS_SOURCE,
            'class_validated': True,
            'label_source': payload.label_source,
            'class_id_history': history,
            'cluster_id': payload.class_id,
            # See label_crop above — subid is cluster-local.
            'cluster_subid': None,
            **_human_class_provenance(),
            'updated_at': _now_iso(),
        }

    async def _label_one(crop_id: str) -> tuple[bool, dict[str, Any] | None]:
        try:
            await occ_update_one(
                opensearch,
                doc_id=crop_id,
                merger=_merge,
                # Cluster drag-drop can race a background worker often
                # enough that a short backoff isn't enough to outlast a
                # single worker-batch write. Five retries covers ~2 s of
                # contention.
                max_retries=5,
                refresh=True,
                writer_id='human:batch_label_crops',
            )
            return True, None
        except OCCFinalConflictError:
            try:
                doc = await opensearch.get(index=CURATION_ITEMS_INDEX, id=crop_id)
                current_source = (doc.get('_source') or {}).get('class_source')
            except Exception:
                current_source = None
            return False, {'crop_id': crop_id, 'current_source': current_source}
        except Exception as exc:
            logger.warning('batch_label_update_failed', crop_id=crop_id, error=str(exc))
            return False, {'crop_id': crop_id, 'current_source': None}

    # Parallelize the per-crop OCC writes. A serialized 10-crop drag-drop
    # would otherwise take one round-trip per crop; asyncio.gather makes it
    # ~one round-trip total. OpenSearch handles dozens of concurrent updates
    # fine on a single index; this only ever runs on human-bounded
    # drag-drop sizes.
    results = await asyncio.gather(*(_label_one(cid) for cid in payload.crop_ids))
    return _batch_outcome(payload.crop_ids, results)


@router.post('/crops/move')
async def move_crops(
    payload: CropMoveRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Move crops to a different cluster.

    In the default ensemble cluster_id == class_id, so this is also a
    relabel: the moved crops get the destination cluster's class assigned
    (with class_source='human_move' and class_validated=True). This
    matches the user's mental model — "move this crop to the X cluster"
    should also mean "this is an X, validated by me".
    """
    if not payload.crop_ids:
        return {'updated': 0, 'updated_ids': [], 'conflicts': []}

    # Resolve the destination class so the crops also get relabeled.
    reg = get_class_registry()
    target = reg.get(int(payload.cluster_id))
    target_name = target.class_name if target is not None else ''

    from src.services.curation.history import record_class_snapshot

    def _merge(current: dict[str, Any]) -> dict[str, Any]:
        history = record_class_snapshot(current, writer='human:move_crops', restorable=True)
        return {
            'cluster_id': int(payload.cluster_id),
            'class_id': int(payload.cluster_id),
            'class_name': target_name,
            'class_source': 'human_move',
            # Move-from-cluster is a class gesture.
            'class_validated': True,
            'label_source': 'human',
            'class_id_history': history,
            # See label_crop above — subid only applies inside the crop's
            # original cluster; clear on move.
            'cluster_subid': None,
            **_human_class_provenance(),
            'updated_at': _now_iso(),
        }

    async def _move_one(crop_id: str) -> tuple[bool, dict[str, Any] | None]:
        try:
            await occ_update_one(
                opensearch,
                doc_id=crop_id,
                merger=_merge,
                # Match batch_label_crops — see note there.
                max_retries=5,
                refresh=True,
                writer_id='human:move_crops',
            )
            return True, None
        except OCCFinalConflictError:
            try:
                doc = await opensearch.get(index=CURATION_ITEMS_INDEX, id=crop_id)
                current_source = (doc.get('_source') or {}).get('class_source')
            except Exception:
                current_source = None
            return False, {'crop_id': crop_id, 'current_source': current_source}
        except Exception as exc:
            logger.warning('move_crops_update_failed', crop_id=crop_id, error=str(exc))
            return False, {'crop_id': crop_id, 'current_source': None}

    results = await asyncio.gather(*(_move_one(cid) for cid in payload.crop_ids))
    return _batch_outcome(payload.crop_ids, results)


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
            refresh=True,
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
    class cluster (``cluster_id == class_id``); an unvalidated one drops
    to the residual pool (``cluster_id=null``) and gets a fresh candidate
    assignment on the next recluster. Crops that aren't excluded are left
    untouched.
    """
    if not payload.crop_ids:
        return {'unexcluded': 0, 'errors': 0}
    from src.services.curation.exclusion import unexclusion_update

    now = _now_iso()
    n_errors = await _occ_bulk_human_write(
        opensearch,
        payload.crop_ids,
        lambda _id, cur: unexclusion_update(cur, now=now),
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
            refresh=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc
    return {'crop_id': crop_id, 'dismissed': True}
