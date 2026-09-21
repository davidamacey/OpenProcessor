"""Generic curation ingest pipeline.

Ported (generic half only) from the private reference ingest service —
see ``docs/design/curation_design_rationale.md`` §2.1 for the citation
convention. The reference file mixed a generic per-image pipeline
(decode, dedup, detect, embed, quality-score, bulk-index) with ~700 LOC
of domain-specific logic (a hardcoded vehicle-class allowlist, a
dual-head vehicle-detector runner, a region-status assignment policy,
and a mismatch-report sink). None of that domain-specific logic is
ported here — a deployment that needs it builds its own overlay on top
of :class:`CurationIngestService` instead.

Pipeline, per image:

1. Decode (EXIF-transposed) + imohash dedup against the images index.
2. Run the primary detector (:class:`~src.config.DetectionProfile`,
   an end2end model whose Triton response is already NMS'd) over the
   full image.
3. If a secondary ``DetectionProfile`` is configured, run it too (a
   raw-output ensemble/backbone detector requiring client-side NMS via
   :func:`~src.services.detection.ensemble_nms.apply_ensemble_nms`) and
   resolve overlap against the primary boxes by IoU — a secondary hit
   above its own confidence floor overrides the primary's class
   assignment; below the floor, the box stays an unlabeled proposal
   with a raw score for lineage.
4. Per-crop PE embedding (:meth:`~src.clients.pe_encoder.PEEncoder.embed_crops`)
   plus a whole-frame PE embedding
   (:meth:`~src.clients.pe_encoder.PEEncoder.embed_whole_frame`).
5. Primary-subject rank + blur quality
   (:mod:`~src.services.detection.crop_quality`), and residual-pool
   cluster assignment via the ingest-time IVF cache
   (:mod:`~src.services.curation.clustering.ivf_ingest`).
6. Crop-cache write (:func:`~src.services.curation.source_image_cache.write_crop_cache`).
7. Bulk index: the images doc is a blind upsert (nothing else writes
   that index); items docs go through
   :func:`~src.clients.occ.occ_upsert_bulk` with human-field guards so a
   re-ingest never clobbers a human-applied label.

``ingest_batch`` parallelizes dedup (msearch) and the whole-image
detector calls (one batched Triton call per detector), then runs the
per-image finishing work under a bounded semaphore.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from src.config import get_curation_config
from src.core.logging import get_logger, get_request_id
from src.services.curation.clustering.ivf_ingest import get_ivf_ingest_store, ingest_passes_gate
from src.services.curation.item_doc import DetectedItem, build_image_doc, build_item_doc
from src.services.curation.source_image_cache import write_crop_cache
from src.services.detection.crop_quality import blur_ratio, crop_lap_var, image_lap_var
from src.services.detection.geometry import (
    bbox_norm as _bbox_norm_fn,
    crop_id as _crop_id_fn,
    iou as _iou_fn,
    letterbox_to_square,
    undo_letterbox,
)


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.clients.curation_opensearch import ClassRegistry
    from src.clients.pe_encoder import PEEncoder
    from src.clients.triton_pool import AsyncTritonPool
    from src.config import CurationConfig, DetectionProfile
    from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore


logger = get_logger(__name__)


# Per-batch in-flight `ingest_one` calls. Triton's own dynamic batching
# coalesces the actual GPU work; this cap just keeps the request
# pipeline from swamping Triton's queue. Override with
# OP_MAX_INGEST_CONCURRENCY (no rebuild required).
MAX_INGEST_CONCURRENCY = int(os.getenv('OP_MAX_INGEST_CONCURRENCY', '16'))

# Secondary-detector confidence floor for an ensemble box to override the
# primary detector's proposal. Kept independent of the primary profile's
# own confidence_floor since the two detectors are calibrated differently.
SECONDARY_IOU_MATCH = 0.3

RESIDUAL_CLUSTER_ID_OFFSET = 10000
PARKED_CLUSTER_ID = -3


# =============================================================================
# Pydantic I/O models
# =============================================================================


class IngestSummary(BaseModel):
    successful: int = 0
    duplicates: int = 0
    failed: int = 0
    labels_imported: int = 0
    crops_indexed: int = 0


class IngestResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: Literal['success', 'duplicate', 'failed'] = 'success'
    image_id: str = ''
    image_path: str = ''
    imohash: str = ''
    n_crops: int = 0
    crops_created: int = 0
    crops_updated: int = 0
    crops_preserved_human: int = 0
    crops_final_conflicts: int = 0
    error: str | None = None
    error_kind: str | None = None


class BatchIngestResult(BaseModel):
    status: Literal['success', 'partial', 'error'] = 'success'
    summary: IngestSummary = Field(default_factory=IngestSummary)
    results: list[IngestResult] = Field(default_factory=list)


# =============================================================================
# Helpers
# =============================================================================


def _imohash_bytes(data: bytes) -> str:
    """Compute a fingerprint for raw image bytes, used for dedup.

    Uses ``imohash`` if available; otherwise falls back to a sha256
    prefix (slim test environments only — production installs imohash).
    """
    try:
        import imohash  # type: ignore[import-untyped]

        return imohash.hashfileobject(io.BytesIO(data)).hex()
    except Exception:
        return hashlib.sha256(data).hexdigest()[:32]


def _decode_image(image_bytes: bytes) -> tuple[Image.Image, int, int]:
    """Decode bytes -> EXIF-transposed RGB PIL image + (width, height)."""
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    w, h = img.size
    return img, w, h


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _crop_id(image_id: str, bbox_norm: list[float]) -> str:
    """Stable item id = sha256(image_id + bbox)[:32].

    Delegates to :func:`src.services.detection.geometry.crop_id` — the
    same function :mod:`src.services.curation.label_import` uses, so
    detector-created and label-created items for the same (image, bbox)
    always collide onto the same document.
    """
    return _crop_id_fn(image_id, bbox_norm)


# =============================================================================
# Service
# =============================================================================


class CurationIngestService:
    """Generic per-image ingest orchestrator.

    All collaborators are dependency-injected; tests pass mocks/fakes.
    The service does not own any collaborator's lifecycle.
    """

    # Fields whose presence on an existing items doc signals a human
    # write that ingest must never clobber on re-ingest.
    _CROP_HUMAN_FIELD_GUARDS = ('label_source', 'class_source')

    def __init__(
        self,
        *,
        opensearch: AsyncOpenSearch | Any,
        triton_pool: AsyncTritonPool | Any,
        registry: ClassRegistry | Any,
        profile: DetectionProfile,
        pe_encoder: PEEncoder | Any,
        secondary_profile: DetectionProfile | None = None,
        config: CurationConfig | None = None,
    ) -> None:
        # Accept either a wrapper exposing `.client` or a raw AsyncOpenSearch.
        self.opensearch = getattr(opensearch, 'client', opensearch)
        self.triton_pool = triton_pool
        self.registry = registry
        self.profile = profile
        self.secondary_profile = secondary_profile
        self.pe_encoder = pe_encoder
        self.config = config or get_curation_config()

    # ------------------------------------------------------------------
    # Dedup
    # ------------------------------------------------------------------

    async def _check_duplicate(self, image_hash: str) -> str | None:
        """Term-query the images index on imohash. Returns existing image_id or None."""
        body = {
            'size': 1,
            'query': {'term': {'imohash': image_hash}},
            '_source': ['image_id'],
        }
        try:
            resp = await self.opensearch.search(index=self.config.images_index, body=body)
        except Exception as exc:
            logger.warning('ingest_dedup_query_failed', error=str(exc))
            return None
        hits = (resp.get('hits') or {}).get('hits') or []
        if not hits:
            return None
        return (hits[0].get('_source') or {}).get('image_id')

    async def _check_duplicates_msearch(
        self,
        image_hashes: list[str],
    ) -> dict[str, str | None]:
        """Batch dedup via msearch. Maps imohash -> existing image_id (or None).

        Falls back to per-image lookup on an msearch failure so dedup is
        never silently skipped.
        """
        if not image_hashes:
            return {}
        body_lines: list[dict[str, Any]] = []
        for image_hash in image_hashes:
            body_lines.append({'index': self.config.images_index})
            body_lines.append(
                {'size': 1, 'query': {'term': {'imohash': image_hash}}, '_source': ['image_id']}
            )
        try:
            resp = await self.opensearch.msearch(body=body_lines)
        except Exception as exc:
            logger.warning('ingest_msearch_failed', error=str(exc), n=len(image_hashes))
            return {h: await self._check_duplicate(h) for h in image_hashes}
        out: dict[str, str | None] = {}
        responses = resp.get('responses', [])
        for image_hash, sub in zip(image_hashes, responses, strict=False):
            hits = ((sub or {}).get('hits') or {}).get('hits') or []
            out[image_hash] = (hits[0].get('_source') or {}).get('image_id') if hits else None
        return out

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    async def _run_primary_detector(self, img: Image.Image) -> list[DetectedItem]:
        """Run the primary end2end detector over the full image.

        Contract: the model returns already-NMS'd detections as
        ``num_dets`` / ``det_boxes`` (normalized ``[0, 1]`` of the network
        input) / ``det_scores`` / ``det_classes`` — the same Ultralytics
        TensorRT end2end export shape this repo's own ``/detect`` endpoint
        serves from.
        """
        from tritonclient.grpc import InferInput, InferRequestedOutput

        chw, scale, pad = letterbox_to_square(img, target=self.profile.input_size)
        inp = InferInput('images', list(chw.shape), 'FP32')
        inp.set_data_from_numpy(chw)
        outs = [
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
        ]
        result = await self.triton_pool.infer(self.profile.detector_model, [inp], outputs=outs)
        return self._decode_primary_result(result, scale, pad, self.profile.input_size)

    def _decode_primary_result(
        self,
        result: Any,
        scale: float,
        pad: tuple[float, float],
        net_size: int,
    ) -> list[DetectedItem]:
        num_dets = int(result.as_numpy('num_dets')[0][0])
        boxes = result.as_numpy('det_boxes')[0][:num_dets]
        scores = result.as_numpy('det_scores')[0][:num_dets]
        classes = result.as_numpy('det_classes')[0][:num_dets]

        out: list[DetectedItem] = []
        for box, score, cls in zip(boxes, scores, classes, strict=False):
            full = undo_letterbox(
                (
                    float(box[0]) * net_size,
                    float(box[1]) * net_size,
                    float(box[2]) * net_size,
                    float(box[3]) * net_size,
                ),
                scale,
                pad,
            )
            cls_id = int(cls)
            conf = float(score)
            entry = self.registry.get(cls_id)
            class_name = entry.class_name if entry is not None else None
            if conf >= self.profile.confidence_floor:
                class_source = f'{self.profile.name}_model'
            else:
                class_source = f'{self.profile.name}_low_conf'
            out.append(
                DetectedItem(
                    bbox_pixel=full,
                    score=conf,
                    class_id=cls_id if conf >= self.profile.confidence_floor else None,
                    class_name=class_name if conf >= self.profile.confidence_floor else None,
                    class_source=class_source,
                    proposal_name=class_name,
                )
            )
        return out

    async def _run_secondary_detector_raw(self, img: Image.Image) -> np.ndarray | None:
        """Run the secondary (raw-output) ensemble detector; return its raw tensor."""
        from tritonclient.grpc import InferInput, InferRequestedOutput

        assert self.secondary_profile is not None
        chw, _scale, _pad = letterbox_to_square(img, target=self.secondary_profile.input_size)
        inp = InferInput('images', list(chw.shape), 'FP32')
        inp.set_data_from_numpy(chw)
        outs = [InferRequestedOutput('output0')]
        result = await self.triton_pool.infer(
            self.secondary_profile.detector_model, [inp], outputs=outs
        )
        return result.as_numpy('output0')

    def _resolve_with_secondary(
        self,
        items: list[DetectedItem],
        raw_output: np.ndarray,
        scale: float,
        pad: tuple[float, float],
    ) -> None:
        """Enrich ``items`` in place using the secondary ensemble detector's NMS output.

        A secondary detection above its own confidence floor overrides
        the matched primary box's class assignment; this is how a
        two-detector ``DetectionProfile`` pair takes
        :func:`apply_ensemble_nms` off zero production callers.
        """
        from src.services.detection.ensemble_nms import apply_ensemble_nms

        profile = self.secondary_profile
        assert profile is not None
        per_image = apply_ensemble_nms(
            raw_output[None, ...] if raw_output.ndim == 2 else raw_output,
            conf_thres=profile.confidence_floor,
        )
        detections = per_image[0] if per_image else []
        secondary_boxes: list[tuple[tuple[float, float, float, float], float, int]] = []
        for det in detections:
            full = undo_letterbox(tuple(det['box']), scale, pad)
            secondary_boxes.append((full, float(det['score']), int(det['class_id'])))

        for item in items:
            best_iou = 0.0
            best: tuple[tuple[float, float, float, float], float, int] | None = None
            for sec_box, sec_score, sec_cls in secondary_boxes:
                score = _iou_fn(item.bbox_pixel, sec_box)
                if score > best_iou:
                    best_iou = score
                    best = (sec_box, sec_score, sec_cls)
            if best is None or best_iou < SECONDARY_IOU_MATCH:
                continue
            _, sec_score, sec_cls = best
            entry = self.registry.get(sec_cls)
            class_name = entry.class_name if entry is not None else None
            item.class_id = sec_cls
            item.class_name = class_name
            item.class_source = f'{profile.name}_model'
            item.score = sec_score

    # ------------------------------------------------------------------
    # Bulk index
    # ------------------------------------------------------------------

    async def _bulk_index(
        self,
        image_doc: dict[str, Any] | None,
        crop_docs: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Index 1 images doc (blind) + N items docs (OCC upsert)."""
        from src.clients.occ import occ_upsert_bulk

        result = {
            'images_indexed': 0,
            'crops_created': 0,
            'crops_updated': 0,
            'crops_preserved_human': 0,
            'crops_final_conflicts': 0,
        }

        if image_doc:
            image_body = [
                {'index': {'_index': self.config.images_index, '_id': image_doc['image_id']}},
                image_doc,
            ]
            resp = await self.opensearch.bulk(body=image_body, refresh=False)
            if isinstance(resp, dict) and resp.get('errors'):
                logger.warning('ingest_bulk_partial_errors', items=resp.get('items', [])[:3])
            else:
                result['images_indexed'] = 1

        if crop_docs:
            upsert = await occ_upsert_bulk(
                self.opensearch,
                crop_docs,
                index=self.config.items_index,
                human_field_guards=list(self._CROP_HUMAN_FIELD_GUARDS),
                writer_id='ingest',
            )
            result['crops_created'] = upsert['created']
            result['crops_updated'] = upsert['updated']
            result['crops_preserved_human'] = upsert['preserved_human']
            result['crops_final_conflicts'] = upsert['final_conflicts']

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_one(
        self,
        image_bytes: bytes,
        image_path: str,
        source: str = 'unknown',
        *,
        prefilled_image: Image.Image | None = None,
    ) -> IngestResult:
        """Run the full pipeline on a single image."""
        if not image_bytes:
            return IngestResult(
                status='failed',
                image_path=image_path,
                error='empty image bytes',
                error_kind='empty',
            )

        if prefilled_image is not None:
            img = prefilled_image
            full_w, full_h = img.size
        else:
            try:
                img, full_w, full_h = _decode_image(image_bytes)
            except UnidentifiedImageError as exc:
                return IngestResult(
                    status='failed',
                    image_path=image_path,
                    error=str(exc),
                    error_kind='unidentified_image',
                )
            except Exception as exc:
                return IngestResult(
                    status='failed',
                    image_path=image_path,
                    error=str(exc),
                    error_kind='decode_error',
                )

        image_hash = _imohash_bytes(image_bytes)
        existing_id = await self._check_duplicate(image_hash)
        if existing_id:
            return IngestResult(
                status='duplicate', image_id=existing_id, image_path=image_path, imohash=image_hash
            )

        try:
            items = await self._run_primary_detector(img)
        except Exception as exc:
            logger.error('ingest_primary_detector_failed', path=image_path, error=str(exc))
            return IngestResult(
                status='failed', image_path=image_path, error=str(exc), error_kind='detector_infer'
            )

        if self.secondary_profile is not None and items:
            try:
                raw = await self._run_secondary_detector_raw(img)
                if raw is not None:
                    _, sec_scale, sec_pad = letterbox_to_square(
                        img, target=self.secondary_profile.input_size
                    )
                    self._resolve_with_secondary(items, raw, sec_scale, sec_pad)
            except Exception as exc:
                logger.warning('ingest_secondary_detector_failed', path=image_path, error=str(exc))

        crops_pil = [self._crop_pil(img, item.bbox_pixel) for item in items]
        try:
            if crops_pil:
                crop_arrays = [np.asarray(c) for c in crops_pil]
                embeddings = await self.pe_encoder.embed_crops(
                    crop_arrays, max_batch=self.profile.batch_limit
                )
                for item, emb in zip(items, embeddings, strict=False):
                    item.pe_embedding = emb
        except Exception as exc:
            logger.warning('ingest_embed_crops_failed', path=image_path, error=str(exc))

        whole_frame_embedding = None
        try:
            whole_frame_embedding = await self.pe_encoder.embed_whole_frame(image_path)
        except Exception as exc:
            logger.warning('ingest_embed_whole_frame_failed', path=image_path, error=str(exc))

        image_id = hashlib.sha256(f'{image_path}|{image_hash}'.encode()).hexdigest()[:32]
        now = _now_iso()
        image_doc = build_image_doc(
            image_id=image_id,
            image_path=image_path,
            source=source,
            width=full_w,
            height=full_h,
            imohash=image_hash,
            now=now,
            whole_frame_embedding=whole_frame_embedding,
        )

        try:
            img_bgr = np.ascontiguousarray(np.asarray(img.convert('RGB'))[:, :, ::-1])
            full_var = image_lap_var(img_bgr)
        except Exception as exc:
            logger.warning('ingest_blur_full_var_failed', path=image_path, error=str(exc))
            img_bgr = None
            full_var = 0.0

        bnorms = [_bbox_norm_fn(it.bbox_pixel, full_w, full_h) for it in items]
        areas = [max(0.0, bn[2] - bn[0]) * max(0.0, bn[3] - bn[1]) for bn in bnorms]
        cids = [_crop_id(image_id, bn) for bn in bnorms]
        rank_by_idx = {
            idx: rank
            for rank, idx in enumerate(
                sorted(range(len(items)), key=lambda i: (-areas[i], cids[i])), start=1
            )
        }

        store: IVFCentroidStore | None = None
        if any(it.class_id is None and it.pe_embedding is not None for it in items):
            store = get_ivf_ingest_store()

        crop_docs: list[dict[str, Any]] = []
        for idx, item in enumerate(items):
            bn = bnorms[idx]
            cid = cids[idx]
            box_var = crop_lap_var(img_bgr, item.bbox_pixel) if img_bgr is not None else None
            ratio = blur_ratio(box_var, full_var)

            write_crop_cache(cid, crops_pil[idx], self.config.crop_cache_dir)

            if item.class_id is not None:
                item.cluster_id = int(item.class_id)
            elif item.pe_embedding is not None and store is not None:
                if not ingest_passes_gate(rank_by_idx[idx], ratio):
                    item.cluster_id = PARKED_CLUSTER_ID
                else:
                    try:
                        centroid, distance = store.assign_one_with_distance(item.pe_embedding)
                        item.cluster_id = int(centroid) + RESIDUAL_CLUSTER_ID_OFFSET
                        item.cluster_distance = distance
                    except Exception as exc:
                        logger.debug('ingest_ivf_assign_failed', error=str(exc))

            crop_docs.append(
                build_item_doc(
                    crop_id=cid,
                    image_id=image_id,
                    image_path=image_path,
                    source=source,
                    request_id=get_request_id(),
                    bbox_norm=bn,
                    item=item,
                    now=now,
                    crop_area_norm=areas[idx],
                    crop_rank_in_image=rank_by_idx[idx],
                    blur_full_var=full_var,
                    blur_lap_var=box_var,
                    blur_lap_ratio=ratio,
                )
            )

        try:
            bulk_result = await self._bulk_index(image_doc, crop_docs)
        except Exception as exc:
            logger.error('ingest_bulk_index_failed', path=image_path, error=str(exc))
            return IngestResult(
                status='failed',
                image_path=image_path,
                imohash=image_hash,
                error=str(exc),
                error_kind='bulk_index',
            )

        return IngestResult(
            status='success',
            image_id=image_id,
            image_path=image_path,
            imohash=image_hash,
            n_crops=len(crop_docs),
            crops_created=bulk_result.get('crops_created', 0),
            crops_updated=bulk_result.get('crops_updated', 0),
            crops_preserved_human=bulk_result.get('crops_preserved_human', 0),
            crops_final_conflicts=bulk_result.get('crops_final_conflicts', 0),
        )

    @staticmethod
    def _crop_pil(img: Image.Image, bbox_pixel: tuple[float, float, float, float]) -> Image.Image:
        x1, y1, x2, y2 = bbox_pixel
        x1i = max(0, round(x1))
        y1i = max(0, round(y1))
        x2i = max(x1i + 1, round(x2))
        y2i = max(y1i + 1, round(y2))
        return img.crop((x1i, y1i, x2i, y2i))

    async def ingest_batch(
        self,
        images: list[bytes],
        image_paths: list[str],
        source: str = 'batch',
    ) -> BatchIngestResult:
        """Batch ingest with parallel msearch dedup + bounded-concurrency finishing work."""
        if len(images) != len(image_paths):
            raise ValueError('images and image_paths must be same length')

        summary = IngestSummary()
        hashes = [_imohash_bytes(b) for b in images]
        hash_to_existing = await self._check_duplicates_msearch(hashes)

        sem = asyncio.Semaphore(MAX_INGEST_CONCURRENCY)

        async def _one(image_bytes: bytes, image_path: str, image_hash: str) -> IngestResult:
            existing_id = hash_to_existing.get(image_hash)
            if existing_id:
                return IngestResult(
                    status='duplicate',
                    image_id=existing_id,
                    image_path=image_path,
                    imohash=image_hash,
                )
            async with sem:
                return await self.ingest_one(image_bytes, image_path, source=source)

        results = list(
            await asyncio.gather(
                *[_one(b, p, h) for b, p, h in zip(images, image_paths, hashes, strict=False)]
            )
        )

        for res in results:
            if res.status == 'duplicate':
                summary.duplicates += 1
            elif res.status == 'success':
                summary.successful += 1
                summary.crops_indexed += res.n_crops
            else:
                summary.failed += 1

        if summary.failed == 0:
            status: Literal['success', 'partial', 'error'] = 'success'
        elif summary.successful == 0:
            status = 'error'
        else:
            status = 'partial'

        return BatchIngestResult(status=status, summary=summary, results=results)


__all__ = [
    'MAX_INGEST_CONCURRENCY',
    'PARKED_CLUSTER_ID',
    'RESIDUAL_CLUSTER_ID_OFFSET',
    'BatchIngestResult',
    'CurationIngestService',
    'IngestResult',
    'IngestSummary',
]
