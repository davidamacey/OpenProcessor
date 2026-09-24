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
   re-ingest never clobbers a human-applied label. With a region profile
   active, new items are seeded ``pending_detection`` (never overwriting
   an existing region status) so the region-detection worker picks them
   up — see :func:`~src.services.curation.item_doc.region_seed_status`.
8. After a successful write, one advisory ``crop.created`` event per
   newly created item on the in-process event hub
   (:func:`~src.services.curation.event_hub.publish_crop_created`).

Sibling modules, split out of this one to keep each to one concern:

* :mod:`src.services.curation.ingest_models` — the wire models.
* :mod:`src.services.curation.ingest_detect` — steps 2-3, the
  ``DetectionProfile``-driven Triton detector runners (single-image and
  **batched**) and their tensor decoding.
* :mod:`src.services.curation.ingest_batch` — ``ingest_batch``'s
  implementation: one msearch dedup, one batched decode, **one batched
  Triton call per detector per ``batch_limit`` chunk** (not one per
  image), the results fed back into ``ingest_one`` through its
  ``prefilled_image`` / ``prefilled_items`` / ``prefilled_secondary``
  arguments, plus optional companion-YOLO-label import.
"""

from __future__ import annotations

import hashlib
import io
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import get_curation_config, get_region_fields
from src.core.logging import get_logger, get_request_id
from src.services.curation.clustering.ivf_ingest import get_ivf_ingest_store, ingest_passes_gate
from src.services.curation.event_hub import publish_crop_created
from src.services.curation.ingest_detect import (
    SECONDARY_IOU_MATCH,
    SecondaryOutput,
    WholeImageDetector,
)
from src.services.curation.ingest_models import BatchIngestResult, IngestResult, IngestSummary
from src.services.curation.item_doc import (
    DetectedItem,
    build_image_doc,
    build_item_doc,
    region_seed_status,
)
from src.services.curation.source_image_cache import write_crop_cache
from src.services.detection.crop_quality import blur_ratio, crop_lap_var, image_lap_var
from src.services.detection.geometry import (
    bbox_norm as _bbox_norm_fn,
    crop_id as _crop_id_fn,
    letterbox_params,
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

RESIDUAL_CLUSTER_ID_OFFSET = 10000
PARKED_CLUSTER_ID = -3


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
        self.detector = WholeImageDetector(
            triton_pool=triton_pool,
            registry=registry,
            profile=profile,
            secondary_profile=secondary_profile,
            backbone_embedding_dim=self.config.backbone_embedding_dim,
        )
        self.region_seed_status = region_seed_status()

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
    # Bulk index
    # ------------------------------------------------------------------

    async def _bulk_index(
        self,
        image_doc: dict[str, Any] | None,
        crop_docs: list[dict[str, Any]],
        created_ids: list[str] | None = None,
    ) -> dict[str, int]:
        """Index 1 images doc (blind) + N items docs (OCC upsert).

        ``created_ids`` (optional out-list) receives the crop_id of every
        items doc this call newly created.
        """
        from src.clients.occ import occ_upsert_bulk

        result = {
            'images_indexed': 0,
            'crops_created': 0,
            'crops_updated': 0,
            'crops_preserved_human': 0,
            'crops_final_conflicts': 0,
            'region_queued': 0,
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
                created_ids=created_ids,
                fill_if_absent=(
                    (get_region_fields().status,) if self.region_seed_status is not None else ()
                ),
            )
            result['crops_created'] = upsert['created']
            result['crops_updated'] = upsert['updated']
            result['crops_preserved_human'] = upsert['preserved_human']
            result['crops_final_conflicts'] = upsert['final_conflicts']
            if self.region_seed_status is not None:
                result['region_queued'] = upsert['created'] + upsert['filled_absent']

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
        prefilled_items: list[DetectedItem] | None = None,
        prefilled_secondary: SecondaryOutput | None = None,
        whole_frame_from_bytes: bool = False,
    ) -> IngestResult:
        """Run the full pipeline on a single image.

        Args:
            image_bytes: Raw JPEG/PNG bytes. Required even when
                ``prefilled_image`` is supplied — it is what the imohash
                dedup fingerprint is computed from.
            image_path: Source path, persisted verbatim on the images doc.
            source: Free-form provenance tag.
            prefilled_image: Already-decoded PIL image; skips the decode
                + EXIF transpose.
            prefilled_items: Primary-detector output from
                :meth:`_run_primary_detector_batch`; skips this image's
                own single-image Triton round-trip. An empty list is
                meaningful (the detector found nothing) and is *not*
                treated as "not prefilled".
            prefilled_secondary: Secondary-detector output (raw tensor
                plus optional backbone feature map) from
                :meth:`WholeImageDetector.run_secondary_raw_batch`; same deal.
            whole_frame_from_bytes: Compute the whole-frame embedding from
                ``image_bytes`` instead of re-reading ``image_path``. Set by
                byte-upload ingest, where ``image_path`` is a client-side
                identifier the server cannot open.

        Every ``prefilled_*`` argument defaults to ``None``, in which
        case this method does the work itself — so direct callers
        (``POST /curation/ingest/image``, scripts) behave exactly as they
        did before the batch path existed.
        """
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

        if prefilled_items is not None:
            items = prefilled_items
        else:
            try:
                items = await self.detector.run_primary(img)
            except Exception as exc:
                logger.error('ingest_primary_detector_failed', path=image_path, error=str(exc))
                return IngestResult(
                    status='failed',
                    image_path=image_path,
                    error=str(exc),
                    error_kind='detector_infer',
                )

        if self.secondary_profile is not None and items:
            secondary = prefilled_secondary
            if secondary is None:
                try:
                    secondary = await self.detector.run_secondary_raw(img)
                except Exception as exc:
                    logger.warning(
                        'ingest_secondary_detector_failed', path=image_path, error=str(exc)
                    )
            if secondary is not None:
                sec_scale, sec_pad = letterbox_params(img, target=self.secondary_profile.input_size)
                # Class override and embedding pooling fail independently —
                # an NMS failure must not also drop the embeddings.
                try:
                    self.detector.resolve_with_secondary(items, secondary.raw, sec_scale, sec_pad)
                except Exception as exc:
                    logger.warning(
                        'ingest_secondary_resolve_failed', path=image_path, error=str(exc)
                    )
                if secondary.feature_map is not None:
                    try:
                        self.detector.attach_backbone_embeddings(
                            items, secondary.feature_map, sec_scale, sec_pad
                        )
                    except Exception as exc:
                        logger.warning(
                            'ingest_backbone_embedding_failed', path=image_path, error=str(exc)
                        )

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
            if whole_frame_from_bytes:
                whole_frame_embedding = await self.pe_encoder.embed_whole_frame_bytes(image_bytes)
            else:
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
                    region_status=self.region_seed_status,
                )
            )

        created_ids: list[str] = []
        try:
            bulk_result = await self._bulk_index(image_doc, crop_docs, created_ids)
        except Exception as exc:
            logger.error('ingest_bulk_index_failed', path=image_path, error=str(exc))
            return IngestResult(
                status='failed',
                image_path=image_path,
                imohash=image_hash,
                error=str(exc),
                error_kind='bulk_index',
            )
        self._publish_created(created_ids, image_path)

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
            n_region_queued=bulk_result.get('region_queued', 0),
        )

    @staticmethod
    def _publish_created(crop_ids: list[str], image_path: str) -> None:
        """Announce newly written items on the live-update event hub.

        Only ids ``occ_upsert_bulk`` confirmed as created are published —
        a re-ingest update or a rejected create is not a new crop. Events
        are advisory (the hub never blocks: bounded per-subscriber queues,
        drop-oldest), so a publish error is logged and swallowed rather
        than failing an ingest whose writes already succeeded.
        """
        for crop_id in crop_ids:
            try:
                publish_crop_created(crop_id, image_path)
            except Exception as exc:
                logger.warning('ingest_event_publish_failed', crop_id=crop_id, error=str(exc))

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
        label_paths: list[str | None] | None = None,
        source: str = 'batch',
        label_source: str = '',
        detect_mismatches: bool = False,
        whole_frame_from_bytes: bool = False,
    ) -> BatchIngestResult:
        """Batch ingest: msearch dedup, batched detector inference, per-image finish.

        Delegates to :func:`src.services.curation.ingest_batch.run_ingest_batch`
        — see that module's docstring for why the batch path is more than
        ``ingest_one`` run N times concurrently.

        Args:
            images: Raw image bytes, one per entry.
            image_paths: Source paths, index-aligned with ``images``.
            label_paths: Optional companion YOLO ``.txt`` paths,
                index-aligned with ``images`` (``None`` per entry to skip
                that image). Supplying them ingests images *and* their
                ground-truth labels in one call, which is what a
                re-ingest-and-verify pass over an already-labeled dataset
                needs.
            source: Provenance tag stamped on every document.
            label_source: ``label_source`` recorded on the imported
                labels; defaults to the label importer's own default.
            detect_mismatches: Record (and count) labels whose IoU-matched
                item carried a different detector class — the
                model-vs-ground-truth disagreement report.
            whole_frame_from_bytes: See :meth:`ingest_one`.

        Raises:
            ValueError: If ``image_paths`` or ``label_paths`` is not the
                same length as ``images``.
        """
        from src.services.curation.ingest_batch import run_ingest_batch

        return await run_ingest_batch(
            self,
            images,
            image_paths,
            label_paths=label_paths,
            source=source,
            label_source=label_source,
            detect_mismatches=detect_mismatches,
            whole_frame_from_bytes=whole_frame_from_bytes,
        )


__all__ = [
    'MAX_INGEST_CONCURRENCY',
    'PARKED_CLUSTER_ID',
    'RESIDUAL_CLUSTER_ID_OFFSET',
    'SECONDARY_IOU_MATCH',
    'BatchIngestResult',
    'CurationIngestService',
    'IngestResult',
    'IngestSummary',
]
