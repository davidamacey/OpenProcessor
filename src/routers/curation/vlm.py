"""VLM-based labeling endpoints — ported from ``legacy_gemma.py`` (§5 Chunk 7).

``POST /curation/vlm/label_batch`` classifies item crops via the shared
:class:`~src.services.labeling.vlm_labeler.VlmLabeler` singleton;
``/verify_regions``, ``/verify_region_batch`` and ``/region_visible_batch``
verify (and, for the first two, read) the crop's sub-region-of-interest
(e.g. a printed label, a license plate).

Deviation from the plan's file-for-file mapping: the reference
``legacy_gemma.py`` derives its class-label provenance dict from
``plate_detect.class_provenance``, which lives in
``src.services.detection.cascade_detect`` — not ported until Chunk 8.
Importing it here would either forward-reference a module that doesn't
exist yet (breaking ``import src.main`` for every wave between Chunk 7
and Chunk 8) or force Chunk 8 to land early. The helper is four lines
(build a provenance dict), so it is inlined as ``_class_provenance``
below rather than deferred-imported; Chunk 8 is unaffected since
``cascade_detect.py``'s own copy is unrelated to this router.
"""

from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from src.clients.occ import is_human_owned_class, occ_skip_on_conflict_bulk
from src.config import get_curation_config, get_region_fields
from src.routers.curation._common import (
    CURATION_ITEMS_INDEX,
    OpenSearchDep,
    _now_iso,
    get_class_registry,
    logger,
    router,
)
from src.services.curation.history import record_class_history
from src.services.curation.image_serving import (
    THUMBNAIL_CACHE,
    resolve_crop_root,
    resolve_safe_path,
)


ITEMS_INDEX = CURATION_ITEMS_INDEX
_F = get_region_fields()


def _get_vlm_labeler(pack_name: str | None = None) -> Any:
    """Lazy per-pack ``VlmLabeler`` cache — imported so VLM routes don't
    pull httpx for the whole router on cold start.

    ``pack_name=None`` uses the process default pack
    (:func:`~src.services.labeling.vlm_prompts.resolve_prompt_pack` — the
    ``OP_PROMPT_PACK_PATH`` pack, or the built-in generic pack). A name
    selects any pack :func:`~src.services.labeling.vlm_prompts.
    available_prompt_packs` advertises; an unknown name raises
    ``ValueError``. One labeler instance is cached per pack name.
    """
    from src.services.labeling.vlm_labeler import VlmLabeler
    from src.services.labeling.vlm_prompts import get_prompt_pack, resolve_prompt_pack

    pack = resolve_prompt_pack() if pack_name is None else get_prompt_pack(pack_name)
    if pack is None:
        msg = f'unknown prompt pack {pack_name!r}'
        raise ValueError(msg)
    cache: dict[str, Any] = _get_vlm_labeler.__dict__.setdefault('_insts', {})
    inst = cache.get(pack.name)
    if inst is None or inst._pack != pack:
        inst = cache[pack.name] = VlmLabeler(pack=pack)
    return inst


async def _default_pack_name(opensearch: Any) -> str | None:
    """The ``prompt_pack`` axis's effective default (settings-doc override
    when set and advertised, else the process default pack)."""
    from src.services.curation.strategy_defaults import resolve_effective_default

    return await resolve_effective_default('prompt_pack', opensearch)


def _class_provenance(
    detector: str,
    detector_version: str,
    *,
    labeler: str,
    labeled_at: str | None = None,
) -> dict[str, Any]:
    """Build the class-provenance dict for crop class label writers.

    Inlined here rather than imported from ``cascade_detect.py`` — see
    module docstring. Mirrors the reference ``class_provenance`` helper
    exactly (field names are class-label provenance, not
    ``RegionFields``-governed).
    """
    return {
        'class_detector': detector,
        'class_detector_version': detector_version,
        'class_labeler': labeler,
        'class_labeled_at': labeled_at or _now_iso(),
    }


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


class VlmLabelBatchRequest(BaseModel):
    crop_ids: list[str]


class VlmVerifyRegionsRequest(BaseModel):
    crop_ids: list[str]


class VlmVerifyRegionBatchItem(BaseModel):
    """One item in a batched region-verify request.

    Mirrors the single-crop ``/curation/vlm/verify_region`` shape. The
    caller is responsible for cropping the sub-region out of its source
    crop and base64-encoding the JPEG bytes — the API does not re-derive
    the region JPEG from OpenSearch on this path so the batch endpoint
    can serve callers (e.g. a detection worker, training scripts) that
    already hold the JPEG in memory.
    """

    crop_id: str
    region_image_b64: str = Field(
        ...,
        description='Base64-encoded JPEG of the sub-region crop (no data: prefix).',
    )
    candidate_text: str | None = Field(
        default=None,
        description='Optional caller-supplied candidate text (e.g. from a text-detection '
        'pre-pass) echoed back on the result.',
    )


class VlmVerifyRegionBatchRequest(BaseModel):
    """Request body for ``POST /curation/vlm/verify_region_batch``."""

    items: list[VlmVerifyRegionBatchItem]


class VlmVerifyRegionBatchResult(BaseModel):
    """One ordered result in the verify_region_batch response."""

    crop_id: str
    is_region: bool
    confidence: str
    reason: str = ''
    candidate_text: str | None = None


class VlmVerifyRegionBatchResponse(BaseModel):
    results: list[VlmVerifyRegionBatchResult]


class VlmRegionVisibleBatchItem(BaseModel):
    crop_id: str
    image_b64: str = Field(
        ...,
        description='Base64-encoded JPEG of the item crop (no data: prefix).',
    )


class VlmRegionVisibleBatchRequest(BaseModel):
    """Request body for ``POST /curation/vlm/region_visible_batch``."""

    items: list[VlmRegionVisibleBatchItem]


class VlmRegionVisibleBatchResponse(BaseModel):
    """``{crop_id: bool}`` mapping — True means a sub-region is visible."""

    visible: dict[str, bool]


@router.post('/vlm/label_batch')
async def vlm_label_batch(
    payload: VlmLabelBatchRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Send up to 64 crops to the VLM — chunked at ``max_images_per_call``."""
    if not payload.crop_ids:
        return {'predicted': 0, 'updated': 0}
    if len(payload.crop_ids) > 64:
        raise HTTPException(status_code=400, detail='maximum 64 crop_ids per call')

    # Pull image_path + bbox_norm for each crop, then build ItemCrop list
    # with the LRU-thumbnail JPEG bytes (128px is enough for the VLM).
    from src.services.labeling.vlm_labeler import ItemCrop

    # Same OP_CROP_CACHE_DIR / CurationConfig.crop_cache_dir the worker
    # (scripts/curation/worker/state.py) writes into -- this used to read a
    # different env var with a different default (GEMMA_CROP_CACHE_DIR),
    # which meant a 100% cache miss out of the box (CFG-2).
    crop_cache_dir = str(get_curation_config().crop_cache_dir)

    def _vlm_jpeg_for(crop_id: str, image_path: str, bbox: tuple) -> bytes | None:
        # Phase A: prefer the RAM crop cache populated by ingest, if any.
        if crop_cache_dir:
            cache_path = Path(crop_cache_dir) / f'{crop_id}.jpg'
            try:
                cached_bytes = cache_path.read_bytes()
                img = Image.open(io.BytesIO(cached_bytes))
                img.thumbnail((224, 224))
                buf = io.BytesIO()
                img.convert('RGB').save(buf, format='JPEG', quality=90)
                return buf.getvalue()
            except FileNotFoundError:
                pass  # fall through to slow path
            except Exception as exc:
                logger.warning('curation_vlm_cache_read_failed', crop_id=crop_id, error=str(exc))
        # Slow path: open source from disk, EXIF-transpose, crop, resize.
        try:
            root = resolve_crop_root(image_path)
            safe = resolve_safe_path(image_path, root)
            return THUMBNAIL_CACHE.get_or_compute(safe, tuple(bbox), size=224)
        except Exception as exc:
            logger.warning('curation_vlm_thumb_failed', crop_id=crop_id, error=str(exc))
            return None

    reg = get_class_registry().load()
    class_names = [c.class_name for c in reg.classes if not c.deprecated]
    crops: list[ItemCrop] = []
    cache_hits = 0
    cache_misses = 0
    for crop_id in payload.crop_ids:
        try:
            doc = await opensearch.get(index=ITEMS_INDEX, id=crop_id)
        except Exception as exc:
            logger.warning('curation_vlm_crop_missing', crop_id=crop_id, error=str(exc))
            continue
        src = doc.get('_source') or {}
        # Never send a human-owned crop to the VLM for reclassification —
        # there is no upstream query filter on this caller-supplied-id
        # endpoint, so this check runs per-crop here.
        if is_human_owned_class(src):
            continue
        image_path = src.get('image_path', '')
        bbox = src.get('bbox_norm')
        if not image_path or not bbox or len(bbox) != 4:
            continue
        cache_path = Path(crop_cache_dir) / f'{crop_id}.jpg' if crop_cache_dir else None
        if cache_path and cache_path.exists():
            cache_hits += 1
        else:
            cache_misses += 1
        jpeg = _vlm_jpeg_for(crop_id, image_path, tuple(bbox))
        if jpeg is None:
            continue
        crops.append(ItemCrop(img_id=crop_id, jpeg_bytes=jpeg))

    if cache_hits + cache_misses > 0:
        logger.info(
            'curation_vlm_label_batch_cache',
            hits=cache_hits,
            misses=cache_misses,
            hit_rate=round(cache_hits / (cache_hits + cache_misses), 3),
        )

    if not crops:
        return {'predicted': 0, 'updated': 0}

    from src.services.labeling.vlm_labeler import resolve_class_name as _resolve_class_name_fn

    labeler = _get_vlm_labeler(await _default_pack_name(opensearch))
    # Use the open-vocabulary path so the VLM can flag genuinely-unknown
    # items instead of silently snapping them to the wrong class.
    predictions = await labeler.label_or_propose_batch(crops, class_names)
    name_to_id = {c.class_name: c.class_id for c in reg.classes if not c.deprecated}
    # Collect per-doc updates, then dispatch via occ_skip_on_conflict_bulk
    # so a concurrent human edit always wins.
    updates_by_id: dict[str, dict[str, Any]] = {}
    proposals: list[dict[str, Any]] = []
    now = _now_iso()
    _vlm_class_prov = _class_provenance(
        detector='vlm',
        detector_version='1',
        labeler='vlm',
        labeled_at=now,
    )

    # Track force-fit bypasses (low-confidence VLM replies where we skip
    # the synonym/fuzzy match and route to the unmatched cohort).
    _force_fit_bypass = {'low_conf_skipped': 0, 'attempted': 0}

    def _resolve(raw: str, *, confidence: str | None = None) -> str | None:
        if confidence == 'low':
            _force_fit_bypass['low_conf_skipped'] += 1
        else:
            _force_fit_bypass['attempted'] += 1
        return _resolve_class_name_fn(raw, name_to_id, confidence=confidence)  # type: ignore[arg-type]

    for p in predictions:
        # Capture the VLM's raw answer on EVERY branch (not just the
        # unmatched exit) so a terms agg can quantify the long tail.
        raw_label = p.proposed_class if p.class_name == '__new__' else p.class_name
        if p.class_name == '__new__' and p.proposed_class:
            proposed_resolved = _resolve(p.proposed_class, confidence=p.confidence)
            if proposed_resolved is not None:
                cid = name_to_id[proposed_resolved]
                updates_by_id[p.img_id] = {
                    'class_id': cid,
                    'class_name': proposed_resolved,
                    'class_source': 'vlm',
                    'label_source': 'vlm',
                    'vlm_confidence': p.confidence,
                    'vlm_raw_class': p.proposed_class,
                    'vlm_raw_label': raw_label,
                    **_vlm_class_prov,
                    'updated_at': now,
                }
                continue
            proposals.append({'crop_id': p.img_id, 'proposed_class': p.proposed_class})
            updates_by_id[p.img_id] = {
                'class_source': 'vlm_new_class_pending',
                'label_source': 'vlm',
                'vlm_proposed_class': p.proposed_class,
                'vlm_raw_label': raw_label,
                'vlm_confidence': p.confidence,
                'needs_new_class': True,
                'updated_at': now,
            }
            continue
        resolved = _resolve(p.class_name, confidence=p.confidence)
        if resolved is None:
            updates_by_id[p.img_id] = {
                'class_source': 'vlm_unmatched',
                'label_source': 'vlm',
                'vlm_raw_class': p.class_name,
                'vlm_raw_label': raw_label,
                'vlm_confidence': p.confidence,
                'updated_at': now,
            }
            continue
        cid = name_to_id[resolved]
        updates_by_id[p.img_id] = {
            'class_id': cid,
            'class_name': resolved,
            'class_source': 'vlm',
            'label_source': 'vlm',
            'vlm_confidence': p.confidence,
            'vlm_raw_label': raw_label,
            **_vlm_class_prov,
            'updated_at': now,
        }
    if updates_by_id:

        def _merge_label_batch(doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
            # Never relabel a frozen test_holdout crop. Returning {} is
            # a documented noop in occ_skip_on_conflict_bulk.
            if _is_frozen_test_holdout(current):
                return {}
            # Defense-in-depth: the fetch loop above already skips
            # human-owned crops before they ever reach `updates_by_id`,
            # but re-check here against the freshest `current` (OCC
            # re-fetches with seq_no) in case a human write landed
            # between the fetch loop and this merge.
            if is_human_owned_class(current):
                return {}
            update = dict(updates_by_id[doc_id])
            if 'class_id' in update:
                update['class_id_history'] = record_class_history(current, writer='vlm_label_batch')
            return update

        try:
            await occ_skip_on_conflict_bulk(
                opensearch,
                doc_ids=list(updates_by_id.keys()),
                merger=_merge_label_batch,
                index=ITEMS_INDEX,
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
        'new_class_proposals': proposals,
        'force_fit_bypass': dict(_force_fit_bypass),
    }


@router.post('/vlm/verify_regions')
async def vlm_verify_regions(
    payload: VlmVerifyRegionsRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Verify whether each crop's region-of-interest contains a real region."""
    if not payload.crop_ids:
        return {'verified': 0}
    if len(payload.crop_ids) > 64:
        raise HTTPException(status_code=400, detail='maximum 64 crop_ids per call')

    from src.services.labeling.vlm_labeler import RegionCrop

    labeler = _get_vlm_labeler(await _default_pack_name(opensearch))
    n_verified = 0
    bulk: list[dict[str, Any]] = []
    now = _now_iso()
    for crop_id in payload.crop_ids:
        try:
            doc = await opensearch.get(index=ITEMS_INDEX, id=crop_id)
        except Exception:
            logger.debug('curation_vlm_region_verify_skip_missing', crop_id=crop_id)
            continue
        src = doc.get('_source') or {}
        region_box = src.get(_F.bbox_norm)
        image_path = src.get('image_path', '')
        if not region_box or len(region_box) != 4 or not image_path:
            continue
        root = resolve_crop_root(image_path)
        try:
            safe = resolve_safe_path(image_path, root)
            jpeg = THUMBNAIL_CACHE.get_or_compute(safe, tuple(region_box), size=224)
        except Exception as exc:
            logger.warning('curation_vlm_region_thumb_failed', crop_id=crop_id, error=str(exc))
            continue
        verdict = await labeler.verify_plate(RegionCrop(crop_id=crop_id, jpeg_bytes=jpeg))
        n_verified += 1
        bulk.append({'update': {'_index': ITEMS_INDEX, '_id': crop_id}})
        bulk.append(
            {
                'doc': {
                    _F.verified: verdict.is_region,
                    _F.reason: verdict.reason,
                    'updated_at': now,
                }
            }
        )
    if bulk:
        try:
            await opensearch.bulk(body=bulk, refresh=False)
        except Exception as exc:
            logger.warning('curation_vlm_region_bulk_failed', error=str(exc))
    return {'verified': n_verified}


@router.post('/vlm/verify_region_batch', response_model=VlmVerifyRegionBatchResponse)
async def vlm_verify_region_batch(
    payload: VlmVerifyRegionBatchRequest,
    opensearch: OpenSearchDep,
) -> VlmVerifyRegionBatchResponse:
    """Verify region crops in batches of ``max_images_per_call`` per upstream VLM call.

    Why this endpoint exists
    ------------------------
    The single-crop ``/curation/vlm/verify_regions`` path bottlenecks a
    high-throughput worker because every verify pins an upstream slot
    for the whole prompt+decode round-trip. This endpoint mirrors the
    chunked-fan-out pattern from ``/curation/vlm/label_batch``: pack
    several region JPEGs per upstream call (the deployment's
    images-per-prompt cap), fire chunks in parallel via
    :py:meth:`VlmLabeler.verify_plate_batch`, and return verdicts
    ordered by input ``crop_id``.

    Unlike ``/curation/vlm/verify_regions`` (which re-derives the region
    JPEG from OpenSearch), this endpoint expects the caller to supply
    the region JPEG directly — cheap for a worker that already holds
    the cropped JPEG in memory.
    """

    items = payload.items
    if not items:
        return VlmVerifyRegionBatchResponse(results=[])
    if len(items) > 64:
        raise HTTPException(status_code=400, detail='maximum 64 items per call')

    from src.services.labeling.vlm_labeler import RegionCrop

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

    labeler = _get_vlm_labeler(await _default_pack_name(opensearch))
    verdicts = await labeler.verify_plate_batch(crops)

    # Re-order to input order (verify_plate_batch already preserves it,
    # but the explicit reorder defends against future implementation
    # changes and gives a deterministic contract).
    by_id = {v.crop_id: v for v in verdicts}
    results: list[VlmVerifyRegionBatchResult] = []
    for item in items:
        v = by_id.get(item.crop_id)
        if v is None:
            results.append(
                VlmVerifyRegionBatchResult(
                    crop_id=item.crop_id,
                    is_region=False,
                    confidence='low',
                    reason='no_response',
                    candidate_text=candidate_text_by_id.get(item.crop_id),
                )
            )
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


@router.post('/vlm/region_visible_batch', response_model=VlmRegionVisibleBatchResponse)
async def vlm_region_visible_batch(
    payload: VlmRegionVisibleBatchRequest,
    opensearch: OpenSearchDep,
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
    detector entirely. On any RPC / parse failure the verdict defaults
    to ``True`` (fail-open) so we never silently drop a crop that might
    have a real sub-region.
    """

    items = payload.items
    if not items:
        return VlmRegionVisibleBatchResponse(visible={})
    if len(items) > 64:
        raise HTTPException(status_code=400, detail='maximum 64 items per call')

    from src.services.labeling.vlm_labeler import RegionCrop

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

    labeler = _get_vlm_labeler(await _default_pack_name(opensearch))
    visible = await labeler.plate_visible_batch(crops)
    return VlmRegionVisibleBatchResponse(visible=visible)


__all__ = ['_get_vlm_labeler']
