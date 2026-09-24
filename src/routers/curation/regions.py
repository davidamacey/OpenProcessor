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
from src.config.region_state import RegionStatus
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
from src.services.curation.wire import item_source_excludes, region_wire_key, serialize_item
from src.services.detection.cascade_detect import region_provenance
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


def _fp_cluster_fields(region_status: str | None) -> dict[str, Any]:
    """Cluster-field side effects of a human region-status write.

    Marking ``false_positive`` parks the item in the permanent FP bucket so
    it leaves the normal region clusters and survives re-clustering. Moving
    an item *off* ``false_positive`` nulls the cluster id so the next
    re-cluster re-absorbs it — otherwise an un-marked item would be stuck in
    the FP bucket.
    """
    from src.services.curation.clustering.orchestrator import FALSE_POSITIVE_REGION_CLUSTER_ID

    F = get_region_fields()
    if region_status == RegionStatus.FALSE_POSITIVE:
        return {
            F.cluster_id: FALSE_POSITIVE_REGION_CLUSTER_ID,
            F.cluster_subid: None,
            F.cluster_distance: 0.0,
        }
    return {F.cluster_id: None, F.cluster_subid: None}


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
                        {'term': {f'{F.status}.keyword': RegionStatus.FALSE_POSITIVE}},
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


def _validate_bbox_norm(bbox: tuple[float, float, float, float]) -> None:
    """Reject malformed region bboxes before they reach OpenSearch."""
    x1, y1, x2, y2 = bbox
    for name, v in (('x1', x1), ('y1', y1), ('x2', x2), ('y2', y2)):
        if not 0.0 <= float(v) <= 1.0:
            raise HTTPException(
                status_code=400,
                detail=f'region bbox {name}={v} out of [0, 1] range',
            )
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(
            status_code=400,
            detail=f'region bbox is degenerate: ({x1}, {y1}, {x2}, {y2})',
        )


def _region_doc(payload: ItemRegionRequest | ItemBatchRegionRequest) -> dict[str, Any]:
    """Build the OpenSearch ``doc`` body for a region set/clear update."""
    F = get_region_fields()
    human = region_profile_or_neutral()
    now = _now_iso()
    if payload.region_bbox_norm is None:
        # Human says "no region visible" — preserve that against re-runs of
        # detection. region score=null distinguishes it from a
        # yet-to-detect item (where score and bbox are simply missing).
        return {
            'doc': {
                F.bbox_norm: None,
                F.score: None,
                F.status: RegionStatus.NO_REGION_VISIBLE,
                F.label_source: payload.region_label_source,
                F.detector: human.human_detector_name,
                F.detector_version: human.human_detector_version,
                F.verifier: human.human_detector_name,
                F.verifier_version: human.human_detector_version,
                F.verified_at: now,
                F.detected_at: now,
                F.bbox_frame: 'source',
                # Human reviewed and said "no region visible" — terminal
                # decision. Region-only signal so we touch ONLY
                # RegionFields.validated; class_validated is unaffected.
                F.validated: True,
                'updated_at': now,
            }
        }
    # Human writes are exempt from the sanity gate (operators can
    # intentionally set unusual boxes).
    _validate_bbox_norm(payload.region_bbox_norm)
    return {
        'doc': {
            F.bbox_norm: list(payload.region_bbox_norm),
            F.score: 1.0,  # human-set boxes are ground truth
            F.status: RegionStatus.DETECTED,
            F.label_source: payload.region_label_source,
            F.verified: True,
            # Human confirmation is terminal — region signal only.
            # RegionFields.validated isolates region edits from class
            # edits (audit: SAM-worker class-clobber bug).
            F.validated: True,
            **region_provenance(
                detector=human.human_detector_name,
                detector_version=human.human_detector_version,
                bbox_frame='source',
                verifier=human.human_detector_name,
                verifier_version=human.human_detector_version,
                detected_at=now,
                verified_at=now,
            ),
            'updated_at': now,
        }
    }


@router.put('/crops/{crop_id}/region')
async def set_crop_region(
    crop_id: str,
    payload: ItemRegionRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Set or clear the region sub-bbox on a single crop.

    ``region_bbox_norm`` is in the **source-image** coordinate frame. The
    client converts crop-frame → source-frame before sending; the API
    never sees crop-frame coords. ``None`` clears the box and marks the
    crop ``region_status='no_region_visible'``.
    """
    F = get_region_fields()
    body = _region_doc(payload)
    region_doc = body['doc']
    try:
        await occ_update_one(
            opensearch,
            doc_id=crop_id,
            merger=lambda _current: region_doc,
            refresh=True,
            writer_id='human:set_crop_region',
        )
    except OCCFinalConflictError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc
    return {
        'crop_id': crop_id,
        region_wire_key('bbox_norm'): payload.region_bbox_norm,
        region_wire_key('status'): region_doc[F.status],
    }


@router.patch('/crops/{crop_id}/region_meta')
async def patch_crop_region_meta(
    crop_id: str,
    payload: ItemRegionMetaRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Patch region metadata (text / status / rejection reason).

    Bbox edits go through ``PUT /crops/{crop_id}/region``; this endpoint
    is for operator corrections of the surrounding fields. Only the
    fields explicitly present in the payload are written.
    """
    F = get_region_fields()
    fields_set = payload.model_fields_set
    if not (fields_set - {'region_label_source'}):
        raise HTTPException(
            status_code=400,
            detail='at least one of region_text, region_status, '
            'region_rejection_reason must be provided',
        )

    doc: dict[str, Any] = {'updated_at': _now_iso()}
    # Wire names of the fields this request changed — never `doc.keys()`,
    # which are RegionFields storage keys.
    wire_fields: list[str] = []
    if 'region_text' in fields_set:
        # Human-typed text is the ground truth; mark the source so the
        # region thumbnail / OCR pipeline knows not to overwrite it.
        doc[F.text] = payload.region_text
        doc[F.text_source] = 'human'
        # Human OCR is by definition 1.0 confidence — null would imply
        # "unknown" which is misleading when a human typed it.
        doc[F.text_confidence] = 1.0 if payload.region_text else None
        wire_fields.append('region_text')
    if 'region_status' in fields_set:
        if (
            payload.region_status is not None
            and payload.region_status not in HUMAN_REGION_STATUS_VALUES
        ):
            raise HTTPException(
                status_code=400,
                detail=f'region_status must be one of {sorted(HUMAN_REGION_STATUS_VALUES)}; '
                f'got {payload.region_status!r}',
            )
        doc[F.status] = payload.region_status
        doc[F.label_source] = payload.region_label_source
        # Route FP marks into the permanent FP bucket (and release on un-mark).
        # Only when region_status is in the payload — never clobber the
        # cluster id on a text-only edit.
        doc.update(_fp_cluster_fields(payload.region_status))
        wire_fields.append('region_status')
    if 'region_rejection_reason' in fields_set:
        doc[F.rejection_reason] = payload.region_rejection_reason
        wire_fields.append('region_rejection_reason')

    # Operator-initiated edits are terminal — keep the row out of the
    # /review/regions queue. AI-source patches (auto-relabel jobs) skip
    # this so they remain reviewable. Region signal only.
    if (payload.region_label_source or '').lower().startswith('human'):
        doc[F.validated] = True

    try:
        await occ_update_one(
            opensearch,
            doc_id=crop_id,
            merger=lambda _current: doc,
            refresh=True,
            writer_id='human:patch_region_meta',
        )
    except OCCFinalConflictError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc
    return {'crop_id': crop_id, 'updated_fields': sorted(wire_fields)}


@router.put('/crops/batch_region')
async def batch_set_crop_region(
    payload: ItemBatchRegionRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Bulk variant of ``PUT /crops/{crop_id}/region``.

    Most useful for the curator's "mark these N crops as no region
    visible" hotkey on the cluster page (via ``bbox_norm=null``);
    setting the same source-frame bbox on many crops doesn't make
    sense in the typical flow but the endpoint accepts it.
    """
    F = get_region_fields()
    if not payload.crop_ids:
        return {'updated': 0, 'conflicts': []}
    if payload.region_bbox_norm is not None:
        _validate_bbox_norm(payload.region_bbox_norm)

    doc_body = _region_doc(payload)['doc']

    updated = 0
    conflicts: list[dict[str, Any]] = []
    for crop_id in payload.crop_ids:
        try:
            await occ_update_one(
                opensearch,
                doc_id=crop_id,
                merger=lambda _current: doc_body,
                max_retries=2,
                refresh=True,
                writer_id='human:batch_set_crop_region',
            )
            updated += 1
        except OCCFinalConflictError:
            try:
                doc = await opensearch.get(index=CURATION_ITEMS_INDEX, id=crop_id)
                current_source = (doc.get('_source') or {}).get(F.label_source)
            except Exception:
                current_source = None
            conflicts.append({'crop_id': crop_id, 'current_source': current_source})
        except Exception as exc:
            logger.warning('batch_region_update_failed', crop_id=crop_id, error=str(exc))
            conflicts.append({'crop_id': crop_id, 'current_source': None})
    return {'updated': updated, 'conflicts': conflicts}


@router.post('/regions/batch_status')
async def batch_set_region_status(
    payload: CropBatchStatusRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Bulk-set region status over many crops — the cluster-view triage op.

    Mark regions ``false_positive`` / ``no_region_visible`` in one call, or
    bulk-confirm with ``region_status='detected'`` + ``region_verified=True``.
    Mirrors ``patch_crop_region_meta`` (human edits are terminal →
    ``region_validated=True`` so they leave the /review queue).
    """
    F = get_region_fields()
    if not payload.crop_ids:
        return {'updated': 0, 'conflicts': []}
    if payload.region_status not in HUMAN_REGION_STATUS_VALUES:
        raise HTTPException(
            status_code=400,
            detail=f'region_status must be one of {sorted(HUMAN_REGION_STATUS_VALUES)}; '
            f'got {payload.region_status!r}',
        )

    doc: dict[str, Any] = {
        F.status: payload.region_status,
        F.label_source: payload.region_label_source,
        'updated_at': _now_iso(),
    }
    if payload.region_verified is not None:
        doc[F.verified] = payload.region_verified
    # Operator-initiated status changes are terminal (keep out of /review).
    if (payload.region_label_source or '').lower().startswith('human'):
        doc[F.validated] = True
    # Route FP marks into the permanent FP bucket (and release on un-mark).
    doc.update(_fp_cluster_fields(payload.region_status))

    updated = 0
    conflicts: list[dict[str, Any]] = []
    for crop_id in payload.crop_ids:
        try:
            await occ_update_one(
                opensearch,
                doc_id=crop_id,
                merger=lambda _current: doc,
                max_retries=2,
                refresh=False,
                writer_id='human:batch_set_region_status',
            )
            updated += 1
        except OCCFinalConflictError:
            conflicts.append({'crop_id': crop_id, 'current_source': None})
        except Exception as exc:
            logger.warning('batch_region_status_failed', crop_id=crop_id, error=str(exc))
            conflicts.append({'crop_id': crop_id, 'current_source': None})
    if updated:
        try:
            await opensearch.indices.refresh(index=CURATION_ITEMS_INDEX)
        except Exception as exc:
            logger.debug('batch_region_status_refresh_failed', error=str(exc))
    return {'updated': updated, 'conflicts': conflicts}
