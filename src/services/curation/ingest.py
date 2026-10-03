"""Generic curation ingest pipeline.

Keeps a generic per-image pipeline (decode, dedup, detect, embed,
quality-score, bulk-index) fully separate from any domain-specific logic
(a hardcoded class allowlist, a dual-head detector runner, a
region-status assignment policy, a mismatch-report sink). See
``docs/design/curation_design_rationale.md`` §2.1 for the design
approach — none of that domain-specific logic lives here; a deployment
that needs it builds its own overlay on top of
:class:`CurationIngestService` instead.

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
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageOps

from src.config import get_curation_config, get_region_fields
from src.core.logging import get_logger
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.event_hub import publish_crop_created
from src.services.curation.ingest_detect import (
    SECONDARY_IOU_MATCH,
    SecondaryOutput,
    WholeImageDetector,
)
from src.services.curation.ingest_index import (
    PARKED_CLUSTER_ID,
    ImageContext,
    IndexOutcome,
    image_id_for,
    index_items,
)
from src.services.curation.ingest_models import (
    ERROR_KIND_DECODE_FAILED,
    ERROR_KIND_DETECTOR_INFER,
    ERROR_KIND_EMPTY,
    BatchIngestResult,
    IngestResult,
    IngestSummary,
)
from src.services.curation.ingest_policy import (
    IngestPolicy,
    apply_detect_filter,
    assign_classes_by_name,
    registry_name_index,
)
from src.services.curation.item_doc import DetectedItem, region_seed_status
from src.services.detection.geometry import crop_id as _crop_id_fn, letterbox_params


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.clients.curation_opensearch import ClassRegistry
    from src.clients.pe_encoder import PEEncoder
    from src.clients.triton_pool import AsyncTritonPool
    from src.config import CurationConfig, DetectionProfile


logger = get_logger(__name__)


# Per-batch in-flight `ingest_one` calls. Triton's own dynamic batching
# coalesces the actual GPU work; this cap just keeps the request
# pipeline from swamping Triton's queue. Override with
# OP_MAX_INGEST_CONCURRENCY (no rebuild required).
MAX_INGEST_CONCURRENCY = int(os.getenv('OP_MAX_INGEST_CONCURRENCY', '16'))


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


@dataclass
class DetectResult:
    """``detect_items``' output: the items to store, a secondary failure (not
    raised; the primary's items stand) and how many detections the project's
    detect filter dropped."""

    items: list[DetectedItem]
    secondary_detector_error: str | None = None
    n_filtered: int = 0


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
        policy: IngestPolicy | None = None,
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
        # The project's detect filter and embedding policy; defaults = today's behaviour.
        self.policy = policy or IngestPolicy()

    # ------------------------------------------------------------------
    # Dedup
    # ------------------------------------------------------------------

    async def _lookup_image(self, image_hash: str) -> dict[str, Any] | None:
        """Term-query the images index on imohash. Returns the existing
        doc's ``image_id`` / ``image_path`` or None."""
        body = {
            'size': 1,
            'query': {'term': {'imohash': image_hash}},
            '_source': ['image_id', 'image_path', 'dataset_split'],
        }
        try:
            resp = await self.opensearch.search(index=self.config.images_index, body=body)
        except Exception as exc:
            logger.warning('ingest_dedup_query_failed', error=str(exc))
            return None
        hits = (resp.get('hits') or {}).get('hits') or []
        if not hits:
            return None
        source = hits[0].get('_source') or {}
        return source if source.get('image_id') else None

    async def _check_duplicate(self, image_hash: str) -> str | None:
        """Existing image_id for ``image_hash``, or None."""
        existing = await self._lookup_image(image_hash)
        return existing['image_id'] if existing is not None else None

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
        *,
        seed_region: bool = True,
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
                    (get_region_fields().status,)
                    if seed_region and self.region_seed_status is not None
                    else ()
                ),
            )
            result['crops_created'] = upsert['created']
            result['crops_updated'] = upsert['updated']
            result['crops_preserved_human'] = upsert['preserved_human']
            result['crops_final_conflicts'] = upsert['final_conflicts']
            if seed_region and self.region_seed_status is not None:
                status_field = get_region_fields().status
                seeded = {d['crop_id'] for d in crop_docs if d.get(status_field) is not None}
                created_seeded = len([cid for cid in created_ids or () if cid in seeded])
                result['region_queued'] = created_seeded + upsert['filled_absent']

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_image(
        self,
        image_bytes: bytes,
        image_path: str,
        source: str = 'unknown',
        *,
        prefilled_image: Image.Image | None = None,
        whole_frame_from_bytes: bool = False,
        source_identifier: str | None = None,
        ingest_run_id: str | None = None,
        adopt_existing: bool = False,
        known_existing: dict[str, Any] | None = None,
    ) -> ImageContext | IngestResult:
        """Decode, fingerprint and dedup one image.

        Returns the :class:`ImageContext` :func:`index_items` consumes, or
        a terminal :class:`IngestResult`: ``failed`` for empty or
        undecodable bytes, ``duplicate`` for an image the index already
        holds. ``known_existing`` is a lookup result the caller already has
        (an image it indexed moments ago that search cannot see yet).
        ``adopt_existing=True`` (a dataset import, which must attach
        labels to an already-indexed image) turns a duplicate into a
        context with ``created=False`` carrying the existing doc's
        ``image_id`` and ``image_path``.
        """
        if not image_bytes:
            return IngestResult(
                status='failed',
                image_path=image_path,
                source_identifier=source_identifier,
                error='empty image bytes',
                error_kind=ERROR_KIND_EMPTY,
            )

        if prefilled_image is not None:
            img = prefilled_image
            full_w, full_h = img.size
        else:
            try:
                img, full_w, full_h = _decode_image(image_bytes)
            except Exception as exc:
                return IngestResult(
                    status='failed',
                    image_path=image_path,
                    source_identifier=source_identifier,
                    error=str(exc),
                    error_kind=ERROR_KIND_DECODE_FAILED,
                )

        image_hash = _imohash_bytes(image_bytes)
        existing = known_existing or await self._lookup_image(image_hash)
        if existing is not None and not adopt_existing:
            return IngestResult(
                status='duplicate',
                image_id=existing['image_id'],
                image_path=image_path,
                source_identifier=source_identifier,
                imohash=image_hash,
            )
        if existing is None:
            image_id = image_id_for(image_path, image_hash)
            resolved_path = image_path
        else:
            image_id = existing['image_id']
            resolved_path = existing.get('image_path') or image_path
        return ImageContext(
            image_id=image_id,
            image_path=resolved_path,
            created=existing is None,
            dataset_split=(existing or {}).get('dataset_split'),
            pil=img,
            width=full_w,
            height=full_h,
            imohash=image_hash,
            image_bytes=image_bytes,
            source=source,
            source_identifier=source_identifier,
            ingest_run_id=ingest_run_id,
            whole_frame_from_bytes=whole_frame_from_bytes,
        )

    async def detect_items(
        self,
        img: Image.Image,
        *,
        image_path: str = '',
        prefilled_items: list[DetectedItem] | None = None,
        prefilled_secondary: SecondaryOutput | None = None,
    ) -> DetectResult:
        """The ingest detectors on one decoded image: the primary, then the
        optional secondary's class override and backbone embeddings, then the
        project's detect filter.

        Returns a :class:`DetectResult`. A primary failure
        raises (the caller decides whether that fails the image); a
        secondary failure is returned, not raised, and the primary's items
        stand. ``ingest_one``, a dataset import's ``propose`` mode and the
        ``detect`` reprocess scope all run this one function.
        """
        if prefilled_items is not None:
            items = prefilled_items
        else:
            items = await self.detector.run_primary(img)

        # F-43: surfaced on the response (IngestResult.secondary_detector_error)
        # and rolled into IngestSummary.secondary_detector_failures instead of
        # only ever reaching a 'warning' log line.
        secondary_detector_error: str | None = None
        if self.secondary_profile is not None and items:
            secondary = prefilled_secondary
            if secondary is None:
                try:
                    secondary = await self.detector.run_secondary_raw(img)
                except Exception as exc:
                    secondary_detector_error = str(exc)
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
        items, n_filtered = apply_detect_filter(items, self.policy.detect, img.width, img.height)
        if self.policy.detect.class_resolution == 'by_name':
            assign_classes_by_name(items, registry_name_index(self.registry), self.profile.name)
        return DetectResult(items, secondary_detector_error, n_filtered)

    async def index_items(
        self,
        ctx: ImageContext,
        items: list[DetectedItem],
        *,
        seed_region: bool = True,
    ) -> IndexOutcome:
        """Embed, score, place and bulk-index ``items`` for ``ctx`` (see
        :func:`~src.services.curation.ingest_index.index_items`)."""
        return await index_items(self, ctx, items, seed_region=seed_region)

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
        source_identifier: str | None = None,
        ingest_run_id: str | None = None,
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
        ctx = await self.ingest_image(
            image_bytes,
            image_path,
            source,
            prefilled_image=prefilled_image,
            whole_frame_from_bytes=whole_frame_from_bytes,
            source_identifier=source_identifier,
            ingest_run_id=ingest_run_id,
        )
        if isinstance(ctx, IngestResult):
            return ctx
        img = ctx.pil

        try:
            detected = await self.detect_items(
                img,
                image_path=image_path,
                prefilled_items=prefilled_items,
                prefilled_secondary=prefilled_secondary,
            )
        except Exception as exc:
            logger.error('ingest_primary_detector_failed', path=image_path, error=str(exc))
            return IngestResult(
                status='failed',
                image_path=image_path,
                source_identifier=source_identifier,
                error=str(exc),
                error_kind=ERROR_KIND_DETECTOR_INFER,
            )

        outcome = await index_items(
            self,
            ctx,
            detected.items,
            secondary_detector_error=detected.secondary_detector_error,
        )
        outcome.result.n_filtered = detected.n_filtered
        return outcome.result

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

    async def ingest_batch(
        self,
        images: list[bytes],
        image_paths: list[str],
        source: str = 'batch',
        whole_frame_from_bytes: bool = False,
        source_identifiers: list[str | None] | None = None,
        ingest_run_id: str | None = None,
    ) -> BatchIngestResult:
        """Batch ingest: msearch dedup, batched detector inference, per-image finish.

        Delegates to :func:`src.services.curation.ingest_batch.run_ingest_batch`
        — see that module's docstring for why the batch path is more than
        ``ingest_one`` run N times concurrently. To ingest an
        already-labeled dataset, use ``POST /datasets/imports``
        (planned, W10 route not yet built).

        Args:
            images: Raw image bytes, one per entry.
            image_paths: Source paths, index-aligned with ``images``.
            source: Provenance tag stamped on every document.
            whole_frame_from_bytes: See :meth:`ingest_one`.

        Raises:
            ValueError: If ``image_paths`` is not the same length as
                ``images``.
        """
        from src.services.curation.ingest_batch import run_ingest_batch

        return await run_ingest_batch(
            self,
            images,
            image_paths,
            source=source,
            whole_frame_from_bytes=whole_frame_from_bytes,
            source_identifiers=source_identifiers,
            ingest_run_id=ingest_run_id,
        )


__all__ = [
    'MAX_INGEST_CONCURRENCY',
    'PARKED_CLUSTER_ID',
    'RESIDUAL_CLUSTER_ID_OFFSET',
    'SECONDARY_IOU_MATCH',
    'BatchIngestResult',
    'CurationIngestService',
    'DetectResult',
    'IngestResult',
    'IngestSummary',
]
