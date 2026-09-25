"""VLM-based labeling endpoints.

``POST /curation/vlm/label_batch`` classifies item crops via the shared
:class:`~src.services.labeling.vlm_labeler.VlmLabeler` singleton;
``/verify_regions``, ``/verify_region_batch`` and ``/region_visible_batch``
verify (and, for the first two, read) the crop's sub-region-of-interest
(e.g. a printed label or sticker).

``_class_provenance`` (below) builds the class-label provenance dict
inline rather than importing it from
``src.services.detection.cascade_detect``, which carries its own,
unrelated copy of the same four-line helper — importing across modules
for four lines isn't worth the coupling.
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

from src.clients.occ import occ_skip_on_conflict_bulk
from src.config import get_curation_config, get_region_fields
from src.routers.curation._common import (
    CURATION_ITEMS_INDEX,
    OpenSearchDep,
    RegionProfileDep,
    _now_iso,
    get_class_registry,
    logger,
    router,
)
from src.services.curation.class_write_guard import (
    CLASS_GUARD_SOURCE_FIELDS,
    ClassWriteGuard,
    class_write_locked,
)
from src.services.curation.clustering.id_normalize import class_cluster_placement
from src.services.curation.image_serving import (
    THUMBNAIL_CACHE,
    resolve_crop_root,
    resolve_safe_path,
)
from src.services.curation.vlm_class_attempt import prediction_class_update, with_class_snapshot


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
    module docstring. Field names are class-label provenance, not
    ``RegionFields``-governed.
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
    crop_ids: list[str] = Field(..., max_length=5000)


class VlmVerifyRegionsRequest(BaseModel):
    crop_ids: list[str] = Field(..., max_length=5000)


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
    """One result in the verify_region_batch response.

    A ``crop_id`` from the request that got no usable VLM answer at all
    (whole-chunk upstream failure, empty/unparseable/misaligned reply,
    or an individual crop missing from an otherwise-aligned reply) is
    absent from ``results`` entirely -- never emitted with a
    synthesized ``is_region=False``. Callers must treat a missing
    crop_id as "retry later", the same contract
    ``/vlm/region_visible_batch`` uses for its map.
    """

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
    # different env var with a different default, which meant a 100% cache
    # miss out of the box.
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
    # One mget_crops() call instead of N separate opensearch.get()
    # round trips. The guard's fields are fetched too: its read token must be
    # built from the same class state the write-time re-check compares.
    from src.clients.curation_opensearch import mget_crops

    docs_by_id = await mget_crops(
        opensearch,
        list(payload.crop_ids),
        index=ITEMS_INDEX,
        source_includes=sorted(
            {'class_source', 'class_validated', 'image_path', 'bbox_norm'}
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

    empty_answers = 0
    for p in predictions:
        update, proposal = prediction_class_update(
            p, name_to_id=name_to_id, resolve=_resolve, now=now, provenance=_vlm_class_prov
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
            update = dict(updates_by_id[doc_id])
            # A registry class moves the item into its class cluster now,
            # as the worker's combined call does (DQ-m3).
            update.update(class_cluster_placement(update, current))
            return with_class_snapshot(update, current, writer='vlm_label_batch')

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
        # Replies with no class: class fields left as they were, attempt recorded.
        'empty_answers': empty_answers,
        'new_class_proposals': proposals,
        'force_fit_bypass': dict(_force_fit_bypass),
    }


@router.post('/vlm/verify_regions')
async def vlm_verify_regions(
    payload: VlmVerifyRegionsRequest,
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
) -> dict[str, Any]:
    """Verify whether each crop's region-of-interest contains a real region.

    A crop the VLM gave no usable answer for (upstream failure, empty or
    unparseable reply) is skipped entirely -- its verify state is left
    untouched for a later retry rather than written as ``verified=False``,
    a verdict the VLM never actually gave.
    """
    if not payload.crop_ids:
        return {'verified': 0}
    if len(payload.crop_ids) > 64:
        raise HTTPException(status_code=400, detail='maximum 64 crop_ids per call')

    from src.services.labeling.vlm_labeler import RegionCrop

    labeler = _get_vlm_labeler(await _default_pack_name(opensearch))
    n_verified = 0
    # Keyed by crop_id rather than written straight to a plain bulk
    # body -- the actual write goes through occ_skip_on_conflict_bulk
    # below so a human verify/label landing on the same crop while this
    # loop's VLM round-trips are in flight wins outright (conflict -> skip,
    # never retried against).
    updates_by_id: dict[str, dict[str, Any]] = {}
    now = _now_iso()
    # One mget_crops() call instead of N separate opensearch.get()
    # round trips.
    from src.clients.curation_opensearch import mget_crops

    docs_by_id = await mget_crops(
        opensearch,
        list(payload.crop_ids),
        index=ITEMS_INDEX,
        source_includes=[_F.bbox_norm, 'image_path'],
    )
    for crop_id in payload.crop_ids:
        doc = docs_by_id.get(crop_id)
        if doc is None:
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
        verdict = await labeler.verify_region(RegionCrop(crop_id=crop_id, jpeg_bytes=jpeg))
        if verdict is None:
            # No usable answer at all -- leave this crop's verify state
            # untouched for a retry rather than writing a verified=False
            # the VLM never actually said.
            continue
        n_verified += 1
        updates_by_id[crop_id] = {
            _F.verified: verdict.is_region,
            _F.reason: verdict.reason,
            'updated_at': now,
        }
    if updates_by_id:

        def _merge_verify_regions(doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
            # A human verify/label landing on this crop while the VLM
            # round-trip was in flight must win outright, never be
            # overwritten by this write -- re-check against the freshest
            # `current` (occ_skip_on_conflict_bulk re-fetches with
            # seq_no) rather than the stale per-crop doc read at the top
            # of the loop above. Mirrors the region-write human guard the
            # clustering orchestrator's bulk writers use.
            if current.get(_F.verifier) == 'human' or current.get(_F.label_source) == 'human':
                return {}
            return updates_by_id[doc_id]

        try:
            await occ_skip_on_conflict_bulk(
                opensearch,
                doc_ids=list(updates_by_id.keys()),
                merger=_merge_verify_regions,
                index=ITEMS_INDEX,
                refresh=False,
                writer_id='vlm_verify_regions',
            )
        except Exception as exc:
            logger.warning('curation_vlm_region_bulk_failed', error=str(exc))
    return {'verified': n_verified}


@router.post('/vlm/verify_region_batch', response_model=VlmVerifyRegionBatchResponse)
async def vlm_verify_region_batch(
    payload: VlmVerifyRegionBatchRequest,
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
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
    :py:meth:`VlmLabeler.verify_region_batch`, and return verdicts
    ordered by input ``crop_id``.

    Unlike ``/curation/vlm/verify_regions`` (which re-derives the region
    JPEG from OpenSearch), this endpoint expects the caller to supply
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


@router.post('/vlm/region_visible_batch', response_model=VlmRegionVisibleBatchResponse)
async def vlm_region_visible_batch(
    payload: VlmRegionVisibleBatchRequest,
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
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
    visible = await labeler.region_visible_batch(crops)
    return VlmRegionVisibleBatchResponse(visible=visible)


__all__ = ['_get_vlm_labeler']
