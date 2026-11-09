"""The VLM batch routes' work, run by the job handlers: label a list of crops, verify region crops.

Moved out of the routes so a worker runs it (``vlm_label_batch`` and ``vlm_verify_region_batch``
jobs). A project without classes or an endpoint that answered nothing is a *result*
(``no_classes``, ``vlm_unreachable``), not a failure, so the caller backs off as it always did.
Region images reach the job through the storage seam, never through ``jobs.params``.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException

from src.clients.occ import occ_skip_on_conflict_bulk
from src.clients.storage.roots import RootedStore
from src.config import get_curation_config
from src.config.curation_accessors import base_curation_config, items_index
from src.core.logging import get_logger
from src.domain.storage_keys import make_key
from src.services.classes.registry import get_class_registry
from src.services.curation.class_write_guard import (
    CLASS_GUARD_SOURCE_FIELDS,
    ClassWriteGuard,
    class_write_locked,
)
from src.services.curation.crop_bytes import load_vlm_item_jpeg
from src.services.curation.label_batch_write import label_batch_merge, label_batch_update
from src.services.curation.source_image_cache import crop_cache_file
from src.services.embedding_spaces.readers import vector_source_excludes
from src.services.labeling.detector_hint import DETECTOR_HINT_FIELDS, hinted_crop
from src.services.labeling.run_labeler import vlm_stamp
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

STAGING_ROOT = 'cache'
STAGING_DIR = 'vlm_batch'


def is_frozen_test_holdout(current_source: dict[str, Any]) -> bool:
    """A frozen test-holdout crop is never relabelled: this writer touches class fields only,
    so the per-document skip is unconditional."""
    return bool(current_source.get('test_holdout'))


def unanswered(predictions: Sequence[Any]) -> bool:
    """No crop got an answer: the endpoint is down. Nothing is written, so every crop stays
    pending and the caller backs off instead of retrying hot."""
    return bool(predictions) and all(p.failure == 'request_failed' for p in predictions)


def staging_store() -> RootedStore:
    """The deployment-wide cache root (not a project's), so the route, the handler and the
    unbound sweep all see one place."""
    return RootedStore({STAGING_ROOT: base_curation_config().crop_cache_dir})


def staging_prefix(batch_id: str) -> str:
    return make_key(STAGING_ROOT, STAGING_DIR, batch_id)


def stage_images(batch_id: str, images: Sequence[bytes]) -> list[str]:
    """Write ``images`` under the batch's prefix; the key of each, in order."""
    store = staging_store()
    keys = []
    for index, data in enumerate(images):
        key = make_key(STAGING_ROOT, STAGING_DIR, batch_id, f'{index:04d}.jpg')
        store.write_atomic(key, data)
        keys.append(key)
    return keys


def read_staged(key: str) -> bytes:
    with staging_store().open_read(key) as handle:
        return handle.read()


def drop_staged(batch_id: str) -> int:
    return staging_store().delete_prefix(staging_prefix(batch_id))


def new_batch_id() -> str:
    return uuid.uuid4().hex


async def _load_crops(
    opensearch: Any, labeler: Any, crop_ids: list[str], guard: ClassWriteGuard
) -> list[Any]:
    """The sendable crops of ``crop_ids``: not locked, with an image and a box, JPEG read.

    One ``mget_crops`` call instead of N gets. The guard's fields are fetched too: its read
    token must be built from the same class state the write-time re-check compares."""
    from src.clients.curation_opensearch.crops import mget_crops

    crop_cache_dir = str(get_curation_config().crop_cache_dir)
    docs_by_id = await mget_crops(
        opensearch,
        crop_ids,
        index=items_index(),
        source_includes=sorted(
            {'class_source', 'class_validated', 'image_path', 'bbox_norm', *DETECTOR_HINT_FIELDS}
            | set(CLASS_GUARD_SOURCE_FIELDS)
        ),
    )
    crops = []
    hits = misses = 0
    for crop_id in crop_ids:
        doc = docs_by_id.get(crop_id)
        if doc is None:
            logger.warning('curation_vlm_crop_missing', crop_id=crop_id)
            continue
        src = doc.get('_source') or {}
        # Never send a human-owned or class-validated crop to the VLM for reclassification:
        # the caller supplies the ids, so there is no upstream query filter.
        if class_write_locked(src):
            continue
        guard.remember(crop_id, src)
        image_path = src.get('image_path', '')
        bbox = src.get('bbox_norm')
        if not image_path or not bbox or len(bbox) != 4:
            continue
        cache_path = crop_cache_file(crop_cache_dir, crop_id) if crop_cache_dir else None
        if cache_path and cache_path.exists():
            hits += 1
        else:
            misses += 1
        jpeg = load_vlm_item_jpeg(crop_id, image_path, tuple(bbox), cache_dir=crop_cache_dir)
        if jpeg is not None:
            crops.append(hinted_crop(crop_id, jpeg, src, labeler._pack))
    if hits + misses > 0:
        logger.info(
            'curation_vlm_label_batch_cache',
            hits=hits,
            misses=misses,
            hit_rate=round(hits / (hits + misses), 3),
        )
    return crops


def _build_updates(
    predictions: Sequence[Any], labelable: Sequence[Any], labeler: Any
) -> dict[str, Any]:
    """The per-document updates for ``predictions`` and the counters the result reports."""
    from src.services.labeling.vlm_class_names import resolve_class_name
    from src.services.labeling.vlm_prompts import prompt_pack_stamp

    pack_stamp = prompt_pack_stamp(labeler._pack)
    name_to_id = {c.class_name: c.class_id for c in labelable}
    # Low-confidence replies skip the synonym/fuzzy match and route to the unmatched cohort.
    bypass = {'low_conf_skipped': 0, 'attempted': 0}

    def resolve(raw: str, *, confidence: str | None = None) -> str | None:
        bypass['low_conf_skipped' if confidence == 'low' else 'attempted'] += 1
        return resolve_class_name(raw, name_to_id, confidence=confidence)  # type: ignore[arg-type]

    updates: dict[str, dict[str, Any]] = {}
    proposals: list[dict[str, Any]] = []
    empty_answers = 0
    now = now_iso()
    stamp = vlm_stamp(labeler)
    for p in predictions:
        update, proposal = label_batch_update(
            p,
            name_to_id=name_to_id,
            resolve=resolve,
            now=now,
            pack_stamp=pack_stamp,
            vlm_endpoint=stamp['vlm_endpoint'],
            vlm_model=stamp['vlm_model'],
        )
        if update is None:
            continue
        if 'class_source' not in update:
            empty_answers += 1
        if proposal is not None:
            proposals.append(proposal)
        updates[p.img_id] = update
    return {
        'updates': updates,
        'proposals': proposals,
        'empty_answers': empty_answers,
        'bypass': bypass,
    }


async def _write_updates(
    opensearch: Any, updates: dict[str, dict[str, Any]], guard: ClassWriteGuard
) -> None:
    """Write through the OCC skip-on-conflict bulk so a concurrent human edit always wins."""

    def merge(doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
        # Returning {} is a documented no-op. A frozen test-holdout crop is never relabelled,
        # and the write lands only on the exact class state this batch read (the VLM call
        # takes seconds): a human relabel in between wins, and so does a human-owned class.
        if is_frozen_test_holdout(current) or not guard.allows(doc_id, current):
            return {}
        return label_batch_merge(updates[doc_id], current)

    try:
        await occ_skip_on_conflict_bulk(
            opensearch,
            vector_excludes=await vector_source_excludes(),
            doc_ids=list(updates),
            merger=merge,
            index=items_index(),
            refresh=False,
            writer_id='vlm_label_batch',
        )
    except OpenSearchException as exc:
        logger.warning('curation_vlm_bulk_failed', error=str(exc))


async def run_label_batch(opensearch: Any, labeler: Any, crop_ids: Sequence[str]) -> dict[str, Any]:
    """Label ``crop_ids`` (at most 64) and write the answers; the route's former body."""
    from src.services.curation.region_class import item_classes

    reg = await get_class_registry().snapshot()
    labelable = item_classes(reg.classes)
    class_names = [c.class_name for c in labelable]
    if not class_names:
        # A fresh project: the VLM worker already has unlabelled items to send, but there is
        # nothing to label them as yet.
        return {'no_classes': True}
    guard = ClassWriteGuard('vlm_label_batch')
    crops = await _load_crops(opensearch, labeler, list(crop_ids), guard)
    if not crops:
        return {'predicted': 0, 'updated': 0}
    # The open-vocabulary path lets the VLM flag genuinely-unknown items instead of
    # silently snapping them to the wrong class.
    predictions = await labeler.label_or_propose_batch(crops, class_names)
    if unanswered(predictions):
        return {
            'vlm_unreachable': True,
            'message': f'the VLM endpoint {labeler.identity.endpoint_ref} did not answer',
        }
    built = _build_updates(predictions, labelable, labeler)
    if built['updates']:
        await _write_updates(opensearch, built['updates'], guard)
    logger.info(
        'curation_vlm_force_fit_bypass',
        attempted=built['bypass']['attempted'],
        low_conf_skipped=built['bypass']['low_conf_skipped'],
    )
    return {
        'predicted': len(predictions),
        'updated': len(built['updates']),
        # Replies with no class: class fields left as they were, attempt recorded.
        'empty_answers': built['empty_answers'],
        'new_class_proposals': built['proposals'],
        'force_fit_bypass': dict(built['bypass']),
    }


async def run_verify_region_batch(labeler: Any, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Verify the staged region crops of ``items`` (``crop_id``, ``key``, ``candidate_text``).

    A ``crop_id`` the VLM gave no verdict for is left out of ``results``, never turned into a
    reject; the caller diffs the result against what it sent and retries the rest."""
    from src.services.labeling.vlm_models import RegionCrop

    crops = [RegionCrop(crop_id=i['crop_id'], jpeg_bytes=read_staged(i['key'])) for i in items]
    by_id = {v.crop_id: v for v in await labeler.verify_region_batch(crops)}
    results = []
    for item in items:
        verdict = by_id.get(item['crop_id'])
        if verdict is None:
            continue
        results.append(
            {
                'crop_id': verdict.crop_id,
                'is_region': verdict.is_region,
                'confidence': verdict.confidence,
                'reason': verdict.reason,
                'candidate_text': item.get('candidate_text'),
            }
        )
    return {'results': results}
