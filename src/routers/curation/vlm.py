"""VLM-based labeling endpoints.

``POST /curation/vlm/label_batch`` classifies item crops via the shared
:class:`~src.services.labeling.vlm_labeler.VlmLabeler` singleton;
``/verify_region_batch`` and ``/region_visible_batch`` judge a
sub-region-of-interest (e.g. a printed label or sticker) from caller-supplied crops;
per-box verification of stored boxes is the ``verify`` action of an assist run.
"""

from __future__ import annotations

import base64
import binascii
from typing import Annotated, Any

from fastapi import HTTPException, Query

from src.clients.occ import occ_skip_on_conflict_bulk
from src.config import get_curation_config, get_region_fields
from src.routers.curation._api_error import api_error
from src.routers.curation._common import (
    OpenSearchDep,
    RegionProfileDep,
    get_class_registry,
    items_index,
    logger,
    router,
)
from src.routers.curation._error_models import REGION_PROFILE_RESPONSES
from src.routers.curation._vlm_route_models import (
    VlmLabelBatchRequest,
    VlmRegionVisibleBatchRequest,
    VlmRegionVisibleBatchResponse,
    VlmVerifyRegionBatchRequest,
    VlmVerifyRegionBatchResponse,
    VlmVerifyRegionBatchResult,
)
from src.routers.curation.pipeline_params import ACKNOWLEDGE_EXTERNAL_DESC, VLM_DESC
from src.services.curation.class_write_guard import (
    CLASS_GUARD_SOURCE_FIELDS,
    ClassWriteGuard,
    class_write_locked,
)
from src.services.curation.crop_bytes import load_vlm_item_jpeg
from src.services.curation.label_batch_write import label_batch_merge, label_batch_update
from src.services.curation.source_image_cache import crop_cache_file
from src.services.labeling.detector_hint import DETECTOR_HINT_FIELDS, hinted_crop
from src.services.labeling.run_labeler import get_vlm_labeler, resolve_pack, vlm_stamp
from src.utils.timeutil import now_iso


_F = get_region_fields()


async def request_labeler(opensearch: Any, vlm: Any, acknowledge_external: Any) -> Any:
    """The labeler for one VLM route call: the settings-default pack, and the
    endpoint the request selected (``?vlm=``) or the project default, both
    through the shared gate. 409 ``vlm_not_configured`` when the project's
    VLM is off."""
    from src.services.curation.autolabel.run_vlm import labeler_unavailable, resolve_run_vlm
    from src.services.labeling.vlm_endpoints import VlmEndpointUnavailableError
    from src.services.labeling.vlm_factory import assert_may_connect

    pack_name = await _default_pack_name(opensearch)
    run = await resolve_run_vlm(
        opensearch,
        vlm,
        pack=resolve_pack(pack_name),
        acknowledge_external=acknowledge_external,
    )
    try:
        if run.endpoint is not None:
            await assert_may_connect(run.endpoint)
        return get_vlm_labeler(pack_name, endpoint=run.endpoint)
    except VlmEndpointUnavailableError as exc:
        raise labeler_unavailable(exc) from exc


def _class_locked(source: dict[str, Any]) -> bool:
    """True when a VLM class write must not touch this item.

    Besides human-owned classes, any validated class (cluster auto-promote,
    label import, ...) is locked: a VLM write only sets some class fields
    (``vlm_unmatched`` sets class_source but not class_id), so letting it
    through left items with ``class_source='vlm_unmatched'`` yet a
    validated class_id set by a different writer.
    """
    return class_write_locked(source)


async def _default_pack_name(opensearch: Any) -> str | None:
    """The ``prompt_pack`` axis's effective default: the config store's
    active pack, else the process default pack."""
    from src.services.curation.strategy_defaults import resolve_effective_default

    return await resolve_effective_default('prompt_pack', opensearch)


def _is_frozen_test_holdout(current_source: dict[str, Any]) -> bool:
    """Guard predicate: True when a doc's current ``_source`` is a
    frozen test_holdout crop.

    ``vlm_label_batch`` fetches crops by explicit crop_id (no OpenSearch
    query to attach a ``must_not`` clause to), so this writer's
    test_holdout guard lives here instead, consulted per-doc inside the
    OCC merger. This writer only ever touches class fields — never
    region fields — so an unconditional per-doc skip is the correct
    scope.
    """
    return bool(current_source.get('test_holdout'))


def _refuse_when_unanswered(predictions: list[Any], labeler: Any) -> None:
    """503 when no crop got an answer: the endpoint is down. Nothing was written, so
    every crop stays pending; the caller backs off instead of retrying hot."""
    if predictions and all(p.failure == 'request_failed' for p in predictions):
        raise api_error(
            503,
            'vlm_unreachable',
            f'the VLM endpoint {labeler.identity.endpoint_ref} did not answer; no crop was labelled',
        )


@router.post('/vlm/label_batch')
async def vlm_label_batch(
    payload: VlmLabelBatchRequest,
    opensearch: OpenSearchDep,
    vlm: Annotated[str | None, Query(description=VLM_DESC)] = None,
    acknowledge_external: Annotated[bool, Query(description=ACKNOWLEDGE_EXTERNAL_DESC)] = False,
) -> dict[str, Any]:
    """Send up to 64 crops to the VLM — chunked at ``max_images_per_call``."""
    if not payload.crop_ids:
        return {'predicted': 0, 'updated': 0}
    if len(payload.crop_ids) > 64:
        raise HTTPException(status_code=400, detail='maximum 64 crop_ids per call')
    # Resolved (and gated) before any work: an unknown or unacknowledged
    # endpoint is a 422 up front, never after the crops were read.
    labeler = await request_labeler(opensearch, vlm, acknowledge_external)

    reg = get_class_registry().load()
    from src.services.curation.region_class import item_classes

    labelable = item_classes(reg.classes)
    class_names = [c.class_name for c in labelable]
    if not class_names:
        # A fresh project: the VLM worker already has unlabelled items to
        # send, but there is nothing to label them as yet.
        raise HTTPException(
            status_code=409,
            detail={
                'error': 'no_classes',
                'message': 'this project has no classes yet; add classes before VLM labeling',
            },
        )

    # Pull image_path + bbox_norm for each crop, then build ItemCrop list
    # with the LRU-thumbnail JPEG bytes (128px is enough for the VLM).
    # Same OP_CROP_CACHE_DIR / CurationConfig.crop_cache_dir the worker
    # (scripts/curation/worker/state.py) writes into -- this used to read a
    # different env var with a different default, which meant a 100% cache
    # miss out of the box.
    crop_cache_dir = str(get_curation_config().crop_cache_dir)

    crops = []
    cache_hits = 0
    cache_misses = 0
    # One mget_crops() call instead of N separate opensearch.get()
    # round trips. The guard's fields are fetched too: its read token must be
    # built from the same class state the write-time re-check compares.
    from src.clients.curation_opensearch.crops import mget_crops

    docs_by_id = await mget_crops(
        opensearch,
        list(payload.crop_ids),
        index=items_index(),
        source_includes=sorted(
            {'class_source', 'class_validated', 'image_path', 'bbox_norm', *DETECTOR_HINT_FIELDS}
            | set(CLASS_GUARD_SOURCE_FIELDS)
        ),
    )
    guard = ClassWriteGuard('vlm_label_batch')
    for crop_id in payload.crop_ids:
        doc = docs_by_id.get(crop_id)
        if doc is None:
            logger.warning('curation_vlm_crop_missing', crop_id=crop_id)
            continue
        src = doc.get('_source') or {}
        # Never send a human-owned or class-validated crop to the VLM for
        # reclassification — there is no upstream query filter on this
        # caller-supplied-id endpoint, so this check runs per-crop here.
        if _class_locked(src):
            continue
        guard.remember(crop_id, src)
        image_path = src.get('image_path', '')
        bbox = src.get('bbox_norm')
        if not image_path or not bbox or len(bbox) != 4:
            continue
        cache_path = crop_cache_file(crop_cache_dir, crop_id) if crop_cache_dir else None
        if cache_path and cache_path.exists():
            cache_hits += 1
        else:
            cache_misses += 1
        jpeg = load_vlm_item_jpeg(crop_id, image_path, tuple(bbox), cache_dir=crop_cache_dir)
        if jpeg is None:
            continue
        crops.append(hinted_crop(crop_id, jpeg, src, labeler._pack))

    if cache_hits + cache_misses > 0:
        logger.info(
            'curation_vlm_label_batch_cache',
            hits=cache_hits,
            misses=cache_misses,
            hit_rate=round(cache_hits / (cache_hits + cache_misses), 3),
        )

    if not crops:
        return {'predicted': 0, 'updated': 0}

    from src.services.labeling.vlm_class_names import resolve_class_name as _resolve_class_name_fn
    from src.services.labeling.vlm_prompts import prompt_pack_stamp

    _pack_stamp = prompt_pack_stamp(labeler._pack)
    # Use the open-vocabulary path so the VLM can flag genuinely-unknown
    # items instead of silently snapping them to the wrong class.
    predictions = await labeler.label_or_propose_batch(crops, class_names)
    _refuse_when_unanswered(predictions, labeler)
    name_to_id = {c.class_name: c.class_id for c in labelable}
    # Collect per-doc updates, then dispatch via occ_skip_on_conflict_bulk
    # so a concurrent human edit always wins.
    updates_by_id: dict[str, dict[str, Any]] = {}
    proposals: list[dict[str, Any]] = []
    now = now_iso()

    # Track force-fit bypasses (low-confidence VLM replies where we skip
    # the synonym/fuzzy match and route to the unmatched cohort).
    _force_fit_bypass = {'low_conf_skipped': 0, 'attempted': 0}

    def _resolve(raw: str, *, confidence: str | None = None) -> str | None:
        if confidence == 'low':
            _force_fit_bypass['low_conf_skipped'] += 1
        else:
            _force_fit_bypass['attempted'] += 1
        return _resolve_class_name_fn(raw, name_to_id, confidence=confidence)  # type: ignore[arg-type]

    empty_answers = 0
    stamp = vlm_stamp(labeler)
    for p in predictions:
        update, proposal = label_batch_update(
            p,
            name_to_id=name_to_id,
            resolve=_resolve,
            now=now,
            pack_stamp=_pack_stamp,
            vlm_endpoint=stamp['vlm_endpoint'],
            vlm_model=stamp['vlm_model'],
        )
        if update is None:
            continue
        if 'class_source' not in update:
            empty_answers += 1
        if proposal is not None:
            proposals.append(proposal)
        updates_by_id[p.img_id] = update
    if updates_by_id:

        def _merge_label_batch(doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
            # Never relabel a frozen test_holdout crop. Returning {} is
            # a documented noop in occ_skip_on_conflict_bulk.
            if _is_frozen_test_holdout(current):
                return {}
            # The VLM call takes seconds: write only onto the exact class
            # state this batch read (a human undo/relabel in between wins),
            # never onto a human-owned or validated class.
            if not guard.allows(doc_id, current):
                return {}
            return label_batch_merge(updates_by_id[doc_id], current)

        try:
            await occ_skip_on_conflict_bulk(
                opensearch,
                doc_ids=list(updates_by_id.keys()),
                merger=_merge_label_batch,
                index=items_index(),
                refresh=False,
                writer_id='vlm_label_batch',
            )
        except Exception as exc:
            logger.warning('curation_vlm_bulk_failed', error=str(exc))
    logger.info(
        'curation_vlm_force_fit_bypass',
        attempted=_force_fit_bypass['attempted'],
        low_conf_skipped=_force_fit_bypass['low_conf_skipped'],
    )
    return {
        'predicted': len(predictions),
        'updated': len(updates_by_id),
        # Replies with no class: class fields left as they were, attempt recorded.
        'empty_answers': empty_answers,
        'new_class_proposals': proposals,
        'force_fit_bypass': dict(_force_fit_bypass),
    }


@router.post(
    '/vlm/verify_region_batch',
    response_model=VlmVerifyRegionBatchResponse,
    responses=REGION_PROFILE_RESPONSES,
)
async def vlm_verify_region_batch(
    payload: VlmVerifyRegionBatchRequest,
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
    vlm: Annotated[str | None, Query(description=VLM_DESC)] = None,
    acknowledge_external: Annotated[bool, Query(description=ACKNOWLEDGE_EXTERNAL_DESC)] = False,
) -> VlmVerifyRegionBatchResponse:
    """Verify region crops in batches of ``max_images_per_call`` per upstream VLM call.

    Why this endpoint exists
    ------------------------
    A one-crop-at-a-time verify bottlenecks a
    high-throughput worker because every verify pins an upstream slot
    for the whole prompt+decode round-trip. This endpoint mirrors the
    chunked-fan-out pattern from ``/curation/vlm/label_batch``: pack
    several region JPEGs per upstream call (the deployment's
    images-per-prompt cap), fire chunks in parallel via
    :py:meth:`VlmLabeler.verify_region_batch`, and return verdicts
    ordered by input ``crop_id``.

    Unlike the assist run's verify action (which reads the region JPEGs of
    stored boxes), this endpoint expects the caller to supply
    the region JPEG directly — cheap for a worker that already holds
    the cropped JPEG in memory.

    A ``crop_id`` the VLM gave no verdict for is omitted from
    ``results`` — never emitted as a synthesized ``is_region=False``
    reject. The caller must diff the response against its request
    ``crop_id``s and retry whatever is missing.
    """

    items = payload.items
    if not items:
        return VlmVerifyRegionBatchResponse(results=[])
    if len(items) > 64:
        raise HTTPException(status_code=400, detail='maximum 64 items per call')

    from src.services.labeling.vlm_models import RegionCrop

    crops: list[RegionCrop] = []
    candidate_text_by_id: dict[str, str | None] = {}
    seen: set[str] = set()
    for item in items:
        if item.crop_id in seen:
            raise HTTPException(status_code=400, detail=f'duplicate crop_id: {item.crop_id}')
        seen.add(item.crop_id)
        try:
            jpeg_bytes = base64.b64decode(item.region_image_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f'invalid base64 for crop_id={item.crop_id}: {exc}',
            ) from exc
        if not jpeg_bytes:
            raise HTTPException(
                status_code=400,
                detail=f'empty region_image_b64 for crop_id={item.crop_id}',
            )
        crops.append(RegionCrop(crop_id=item.crop_id, jpeg_bytes=jpeg_bytes))
        candidate_text_by_id[item.crop_id] = item.candidate_text

    labeler = await request_labeler(opensearch, vlm, acknowledge_external)
    verdicts = await labeler.verify_region_batch(crops)

    # Re-order to input order (verify_region_batch already preserves it,
    # but the explicit reorder defends against future implementation
    # changes and gives a deterministic contract).
    by_id = {v.crop_id: v for v in verdicts}
    results: list[VlmVerifyRegionBatchResult] = []
    for item in items:
        v = by_id.get(item.crop_id)
        if v is None:
            # No verdict for this crop_id at all -- omit it so the
            # caller retries, rather than recording a rejection the VLM
            # never gave.
            continue
        results.append(
            VlmVerifyRegionBatchResult(
                crop_id=v.crop_id,
                is_region=v.is_region,
                confidence=v.confidence,
                reason=v.reason,
                candidate_text=candidate_text_by_id.get(v.crop_id),
            )
        )
    return VlmVerifyRegionBatchResponse(results=results)


@router.post(
    '/vlm/region_visible_batch',
    response_model=VlmRegionVisibleBatchResponse,
    responses=REGION_PROFILE_RESPONSES,
)
async def vlm_region_visible_batch(
    payload: VlmRegionVisibleBatchRequest,
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
    vlm: Annotated[str | None, Query(description=VLM_DESC)] = None,
    acknowledge_external: Annotated[bool, Query(description=ACKNOWLEDGE_EXTERNAL_DESC)] = False,
) -> VlmRegionVisibleBatchResponse:
    """Pre-filter item crops by asking the VLM whether a sub-region is visible.

    Why this endpoint exists
    ------------------------
    A full region-of-interest detector is often the throughput
    bottleneck. A non-trivial fraction of item crops contain no visible
    sub-region at all. Routing those crops through the detector +
    verify pipeline is pure waste. A yes/no VLM call packed several
    crops at a time costs roughly an order of magnitude less per crop
    than a verify call.

    The endpoint returns ``{crop_id: bool}``: ``True`` means the crop
    should continue to the detector; ``False`` means the caller should
    write a terminal "not visible" status directly and skip the
    detector entirely. On an RPC failure or a garbled entry the verdict
    defaults to ``True`` (fail-open) so we never silently drop a crop that
    might have a real sub-region. A crop missing from the map got no
    verdict (the VLM answered its chunk with nothing): retry it.
    """

    items = payload.items
    if not items:
        return VlmRegionVisibleBatchResponse(visible={})
    if len(items) > 64:
        raise HTTPException(status_code=400, detail='maximum 64 items per call')

    from src.services.labeling.vlm_models import RegionCrop

    crops: list[RegionCrop] = []
    seen: set[str] = set()
    for item in items:
        if item.crop_id in seen:
            raise HTTPException(status_code=400, detail=f'duplicate crop_id: {item.crop_id}')
        seen.add(item.crop_id)
        try:
            jpeg_bytes = base64.b64decode(item.image_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f'invalid base64 for crop_id={item.crop_id}: {exc}',
            ) from exc
        if not jpeg_bytes:
            raise HTTPException(
                status_code=400,
                detail=f'empty image_b64 for crop_id={item.crop_id}',
            )
        crops.append(RegionCrop(crop_id=item.crop_id, jpeg_bytes=jpeg_bytes))

    labeler = await request_labeler(opensearch, vlm, acknowledge_external)
    visible = await labeler.region_visible_batch(crops)
    return VlmRegionVisibleBatchResponse(visible=visible)


__all__ = ['get_vlm_labeler', 'request_labeler']
