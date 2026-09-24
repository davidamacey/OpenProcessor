"""Curation router sub-module — region-of-interest browse + human edit endpoints.

Ported from the reference implementation's region router. Covers browsing
items that carry a region-of-interest sub-bbox (filtered by
provenance/score/text), the curated training-cohort picker, and the
human-edit endpoints for setting/clearing/patching that sub-bbox.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query

from src.clients.occ import OCCFinalConflictError, occ_update_one
from src.config import DetectionProfile, get_region_fields
from src.config.region_state import RegionStatus, region_status_catalog
from src.routers.curation._common import (
    CURATION_ITEMS_INDEX,
    HUMAN_REGION_STATUS_VALUES,
    CropBatchStatusRequest,
    ItemBatchRegionRequest,
    ItemRegionMetaRequest,
    ItemRegionRequest,
    OpenSearchDep,
    _ensure_indexes,
    _now_iso,
    logger,
    router,
)
from src.services.curation.region_writes import (
    RegionWriteError,
    human_status_fields,
    parent_to_source_bbox,
    post_write_item,
    region_box_doc,
    validate_bbox_norm,
)
from src.services.curation.wire import item_source_excludes, region_wire_key, serialize_item
from src.services.detection.profile_registry import region_profile_or_neutral


_TRAINING_CANDIDATE_MODES = (
    'detector_blind_spots',
    'low_conf_correct',
    'disagreement',
    'human_corrected',
    'false_positives',
)


def _region_item(src: dict[str, Any], crop_id: str) -> dict[str, Any]:
    """Wire item for /regions + /regions/training_candidates — the shared
    item serializer, identical to /crops and /review."""
    return serialize_item(src, crop_id)


# Large embedding fields (1024 floats) — excluded from browse _source.
_REGION_SOURCE_EXCLUDES = item_source_excludes()


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
    include_test: bool = False,
) -> dict[str, Any]:
    """Browse crops that have a region bbox, filtered by provenance/score/text.

    Used by the labeler's /clusters page when the region-of-interest
    class is the selected class filter, and by any region-centric
    review tooling.
    """
    F = get_region_fields()
    await _ensure_indexes(opensearch)
    must: list[dict[str, Any]] = [{'exists': {'field': F.bbox_norm}}]
    must_not: list[dict[str, Any]] = []
    if not include_test:
        must_not.append({'term': {'test_holdout': True}})
    if class_id is not None:
        must.append({'term': {'class_id': class_id}})
    if cluster_id is not None:
        must.append({'term': {'cluster_id': cluster_id}})
    if region_cluster_id is not None:
        must.append({'term': {F.cluster_id: region_cluster_id}})
    if region_cluster_subid is not None:
        must.append({'term': {F.cluster_subid: region_cluster_subid}})
    if max_rank is not None:
        must.append({'range': {'crop_rank_in_image': {'lte': max_rank}}})
    if min_score is not None or max_score is not None:
        rng: dict[str, float] = {}
        if min_score is not None:
            rng['gte'] = min_score
        if max_score is not None:
            rng['lte'] = max_score
        must.append({'range': {F.score: rng}})
    if verified is not None:
        must.append({'term': {F.verified: verified}})
    if detector is not None:
        must.append({'term': {F.detector: detector}})
    if text:
        must.append({'wildcard': {F.text: f'*{text.upper()}*'}})

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
    body: dict[str, Any] = {
        'from': (page - 1) * page_size,
        'size': page_size,
        '_source': {'excludes': _REGION_SOURCE_EXCLUDES},
        'query': {'bool': {'must': must, 'must_not': must_not}},
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
                    'must': [
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
            'Primary detector missed; secondary segmenter found the region, VLM confirmed',
        )
    if mode == 'low_conf_correct':
        return (
            {
                'bool': {
                    'must': [
                        {'term': {F.detector: profile.detector_model}},
                        {'term': {F.verified: True}},
                        {'range': {F.score: {'lt': 0.6}}},
                    ],
                    'must_not': [{'term': {'test_holdout': True}}],
                }
            },
            'Primary detector hit but with low confidence; useful as high-loss training rows',
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
                    'must': [
                        {'exists': {'field': F.bbox_norm}},
                        {'term': {F.detector_chain: f'{profile.detector_model}:hit'}},
                        {'term': {F.detector_chain: f'{profile.segmenter_name}:hit'}},
                    ],
                    'must_not': [{'term': {'test_holdout': True}}],
                }
            },
            'Primary detector + secondary segmenter both fired; bboxes may disagree',
        )
    if mode == 'human_corrected':
        # Rows where a human PUT a region AND there was a prior detector
        # chain. These are the gold-standard training rows — a human
        # reviewed a model's output and corrected it.
        return (
            {
                'bool': {
                    'must': [
                        {'exists': {'field': F.label_source}},
                        {'exists': {'field': F.detector_chain}},
                    ],
                    'must_not': [{'term': {'test_holdout': True}}],
                }
            },
            'Human corrected a prior detector output',
        )
    if mode == 'false_positives':
        # Human marked a detected box as "not a region" but the box +
        # provenance were KEPT (region status='false_positive'). These
        # feed the dedicated detector training run as hard negatives —
        # the detector fired here and should learn not to.
        return (
            {
                'bool': {
                    'must': [
                        {'term': {F.status: RegionStatus.FALSE_POSITIVE}},
                        {'exists': {'field': F.bbox_norm}},
                    ],
                    'must_not': [{'term': {'test_holdout': True}}],
                }
            },
            'Human marked a detector box as a false positive (box retained)',
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
        # Tack the class filter onto the bool.must of the mode query.
        query['bool']['must'] = [*query['bool']['must'], {'term': {'class_id': class_id}}]

    body: dict[str, Any] = {
        'from': (page - 1) * page_size,
        'size': page_size,
        '_source': {'excludes': _REGION_SOURCE_EXCLUDES},
        'query': query,
        'sort': [{F.detected_at: {'order': 'desc', 'missing': '_last'}}],
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


def _write_error(exc: RegionWriteError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


def _validate_status(region_status: str | None) -> None:
    if region_status not in HUMAN_REGION_STATUS_VALUES:
        raise HTTPException(
            status_code=400,
            detail=f'region_status must be one of {sorted(HUMAN_REGION_STATUS_VALUES)}; '
            f'got {region_status!r}',
        )


class _Recorder:
    """OCC merger wrapper that remembers the doc it last merged onto, so the
    handler can return the post-write item without a second read."""

    def __init__(self, build: Any) -> None:
        self._build = build
        self.current: dict[str, Any] = {}
        self.update: dict[str, Any] = {}

    def __call__(self, current: dict[str, Any]) -> dict[str, Any]:
        self.current = current
        self.update = self._build(current)
        return self.update

    def item(self, crop_id: str) -> dict[str, Any]:
        return post_write_item(self.current, self.update, crop_id)


async def _write_one(
    opensearch: Any, crop_id: str, rec: _Recorder, writer_id: str, **kw: Any
) -> None:
    """One OCC write; RegionWriteError -> 422, a missing doc -> 404."""
    try:
        await occ_update_one(
            opensearch, doc_id=crop_id, merger=rec, refresh=True, writer_id=writer_id, **kw
        )
    except OCCFinalConflictError:
        raise
    except RegionWriteError as exc:
        raise _write_error(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc


def _box_builder(payload: ItemRegionRequest | ItemBatchRegionRequest) -> Any:
    """Merger body for a box write. Range errors are a 400 up front; a
    parent-frame box is projected per item inside the merger."""
    box = None if payload.region_bbox_norm is None else list(payload.region_bbox_norm)
    if box is not None:
        try:
            validate_bbox_norm(box)
        except RegionWriteError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    now = _now_iso()
    source = payload.region_label_source
    if box is None or payload.frame == 'source':
        doc = region_box_doc(box, label_source=source, now=now)
        return lambda _current: doc

    def _build(current: dict[str, Any]) -> dict[str, Any]:
        projected = parent_to_source_bbox(box, current.get('bbox_norm'))
        return region_box_doc(projected, label_source=source, now=now)

    return _build


@router.get('/regions/statuses')
async def region_statuses() -> dict[str, Any]:
    """The region lifecycle vocabulary: every status with its ``label``,
    ``role``, ``terminal``, ``human_writable``, ``clears_box`` and
    ``wants_reason``, plus the ``confirm_status`` / ``reject_status`` /
    ``false_positive_status`` values. Generated from
    ``src/config/region_state.py``; the human writers enforce it."""
    return region_status_catalog()


@router.put('/crops/{crop_id}/region')
async def set_crop_region(
    crop_id: str,
    payload: ItemRegionRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Set or clear the region sub-bbox on a single crop.

    ``region_bbox_norm`` is in ``frame`` (``source`` default, or
    ``parent`` = the item crop, projected server-side). ``None`` clears the box and marks the crop
    ``region_status='no_region_visible'``. Returns ``item``, the post-write
    wire item.
    """
    F = get_region_fields()
    rec = _Recorder(_box_builder(payload))
    await _write_one(opensearch, crop_id, rec, 'human:set_crop_region')
    return {
        'crop_id': crop_id,
        region_wire_key('bbox_norm'): rec.update[F.bbox_norm],
        region_wire_key('status'): rec.update[F.status],
        'item': rec.item(crop_id),
    }


@router.patch('/crops/{crop_id}/region_meta')
async def patch_crop_region_meta(
    crop_id: str,
    payload: ItemRegionMetaRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Patch region metadata (text / status / rejection reason).

    Bbox edits go through ``PUT /crops/{crop_id}/region``. Only the fields
    present in the payload are written; a status write applies the
    lifecycle invariants (see :mod:`src.services.curation.region_writes`).
    Returns ``updated_fields`` (wire names) and ``item`` (post-write).
    """
    F = get_region_fields()
    fields_set = payload.model_fields_set
    if not (fields_set - {'region_label_source'}):
        raise HTTPException(
            status_code=400,
            detail='at least one of region_text, region_status, '
            'region_rejection_reason must be provided',
        )
    if 'region_status' in fields_set:
        _validate_status(payload.region_status)

    base: dict[str, Any] = {'updated_at': _now_iso()}
    # Wire names of the fields this request changed — never `doc.keys()`,
    # which are RegionFields storage keys.
    wire_fields: list[str] = []
    if 'region_text' in fields_set:
        # Human-typed text is the ground truth; mark the source so the
        # OCR pipeline knows not to overwrite it. Human OCR is 1.0
        # confidence — null would read as "unknown".
        base[F.text] = payload.region_text
        base[F.text_source] = 'human'
        base[F.text_confidence] = 1.0 if payload.region_text else None
        wire_fields.append('region_text')
    if 'region_status' in fields_set:
        base[F.label_source] = payload.region_label_source
        wire_fields.append('region_status')
    if 'region_rejection_reason' in fields_set:
        base[F.rejection_reason] = payload.region_rejection_reason
        wire_fields.append('region_rejection_reason')
    # Operator-initiated edits are terminal — keep the row out of the
    # /review/regions queue. AI-source patches (auto-relabel jobs) skip
    # this so they remain reviewable. Region signal only.
    if (payload.region_label_source or '').lower().startswith('human'):
        base[F.validated] = True

    def _build(current: dict[str, Any]) -> dict[str, Any]:
        doc = dict(base)
        if 'region_status' in fields_set:
            doc.update(human_status_fields(str(payload.region_status), current))
        return doc

    rec = _Recorder(_build)
    await _write_one(opensearch, crop_id, rec, 'human:patch_region_meta')
    return {'crop_id': crop_id, 'updated_fields': sorted(wire_fields), 'item': rec.item(crop_id)}


async def _batch_write(
    opensearch: Any,
    crop_ids: list[str],
    build: Any,
    writer_id: str,
) -> dict[str, Any]:
    """Apply ``build`` to every crop; per-crop outcomes, post-write items.

    Only refreshes once at the end (the per-doc writes use refresh=False).
    """
    F = get_region_fields()
    updated = 0
    conflicts: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for crop_id in crop_ids:
        rec = _Recorder(build)
        try:
            await occ_update_one(
                opensearch,
                doc_id=crop_id,
                merger=rec,
                max_retries=2,
                refresh=False,
                writer_id=writer_id,
            )
        except RegionWriteError as exc:
            invalid.append({'crop_id': crop_id, 'detail': str(exc)})
            continue
        except OCCFinalConflictError:
            try:
                doc = await opensearch.get(index=CURATION_ITEMS_INDEX, id=crop_id)
                current_source = (doc.get('_source') or {}).get(F.label_source)
            except Exception:
                current_source = None
            conflicts.append({'crop_id': crop_id, 'current_source': current_source})
            continue
        except Exception as exc:
            logger.warning('batch_region_write_failed', crop_id=crop_id, error=str(exc))
            conflicts.append({'crop_id': crop_id, 'current_source': None})
            continue
        updated += 1
        items.append(rec.item(crop_id))
    if updated:
        try:
            await opensearch.indices.refresh(index=CURATION_ITEMS_INDEX)
        except Exception as exc:
            logger.debug('batch_region_refresh_failed', error=str(exc))
    return {'updated': updated, 'conflicts': conflicts, 'invalid': invalid, 'items': items}


@router.put('/crops/batch_region')
async def batch_set_crop_region(
    payload: ItemBatchRegionRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Bulk variant of ``PUT /crops/{crop_id}/region`` (typically
    ``region_bbox_norm=null``: "no region visible on these N crops").

    Returns ``updated``, ``conflicts``, ``invalid`` and ``items`` (the
    post-write wire items of the updated crops).
    """
    if not payload.crop_ids:
        return {'updated': 0, 'conflicts': [], 'invalid': [], 'items': []}
    return await _batch_write(
        opensearch, payload.crop_ids, _box_builder(payload), 'human:batch_set_crop_region'
    )


@router.post('/regions/batch_status')
async def batch_set_region_status(
    payload: CropBatchStatusRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Bulk-set region status over many crops — the cluster-view triage op.

    Applies the same lifecycle invariants as ``PATCH region_meta``:
    ``no_region_visible`` clears the box, ``region_verified`` follows the
    status (a request's ``region_verified`` is ignored), ``detected``
    without a box lands in ``invalid``. Human edits are terminal
    (``region_validated=True``). Returns ``updated``, ``conflicts``,
    ``invalid`` and ``items`` (post-write wire items).
    """
    F = get_region_fields()
    if not payload.crop_ids:
        return {'updated': 0, 'conflicts': [], 'invalid': [], 'items': []}
    _validate_status(payload.region_status)
    base: dict[str, Any] = {
        F.label_source: payload.region_label_source,
        'updated_at': _now_iso(),
    }
    if (payload.region_label_source or '').lower().startswith('human'):
        base[F.validated] = True

    def _build(current: dict[str, Any]) -> dict[str, Any]:
        return {**base, **human_status_fields(payload.region_status, current)}

    return await _batch_write(opensearch, payload.crop_ids, _build, 'human:batch_set_region_status')
