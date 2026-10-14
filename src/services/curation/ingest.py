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
   (:meth:`~src.clients.pe_encoder.PEEncoder.embed_whole_frame_image`).
5. Primary-subject rank + blur quality
   (:mod:`~src.services.detection.crop_quality`), and residual-pool
   cluster assignment via the ingest-time IVF cache
   (:mod:`~src.services.curation.clustering.ivf_ingest`).
6. Crop-cache write (:func:`~src.services.curation.source_image_cache.write_crop_cache`).
7. Bulk index: the images doc is a blind upsert (nothing else writes
   that index); items docs go through
   :func:`~src.clients.occ_upsert.occ_upsert_bulk` with human-field guards so a
   re-ingest never clobbers a human-applied label. With a region profile
   active, new items are seeded ``pending_detection`` (never overwriting
   an existing region status) so the region-detection worker picks them
   up — see :func:`~src.services.curation.item_doc.region_seed_status`.
8. After a successful write, one advisory ``crop.created`` event per
   newly created item on the in-process event hub
   (:func:`~src.services.curation.event_hub.publish_crop_created`).

Sibling modules, split out of this one to keep each to one concern:

* :mod:`src.services.curation.ingest_helpers` — the imohash fingerprint, the
  upright decode and ``MAX_INGEST_CONCURRENCY``.
* :mod:`src.services.curation.ingest_dedup` — the dedup lookups
  (:class:`IngestDedupMixin`).
* :mod:`src.services.curation.ingest_items` — detecting one image's items and
  bulk-writing its documents (:class:`IngestItemsMixin`, ``DetectResult``).
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

from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.event_hub import publish_crop_created
from src.services.curation.ingest_dedup import IngestDedupMixin
from src.services.curation.ingest_detect import (
    SECONDARY_IOU_MATCH,
    SecondaryOutput,
    WholeImageDetector,
)
from src.services.curation.ingest_helpers import _decode_image, _imohash_bytes
from src.services.curation.ingest_index import (
    PARKED_CLUSTER_ID,
    ImageContext,
    IndexOutcome,
    image_id_for,
    index_items,
)
from src.services.curation.ingest_items import IngestItemsMixin
from src.services.curation.ingest_models import (
    ERROR_KIND_DECODE_FAILED,
    ERROR_KIND_DETECTOR_INFER,
    ERROR_KIND_EMPTY,
    BatchIngestResult,
    IngestResult,
    IngestSummary,
)
from src.services.curation.ingest_policy import IngestPolicy
from src.services.curation.item_doc import DetectedItem, region_seed_status
from src.services.curation.ops_metrics import record_ingest_result
from src.utils.cpu_pool import encoded_nbytes, run_cpu


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.clients.curation_opensearch.registry import ClassRegistry
    from src.clients.pe_encoder import PEEncoder
    from src.clients.triton_pool import AsyncTritonPool
    from src.config import CurationConfig, DetectionProfile
    from src.services.curation.image_decode import DecodedImage
    from src.services.curation.ingest_gpu_decode import GpuDecodeFrame


logger = get_logger(__name__)


class CurationIngestService(IngestDedupMixin, IngestItemsMixin):
    """Generic per-image ingest orchestrator.

    All collaborators are dependency-injected; tests pass mocks/fakes.
    The service does not own any collaborator's lifecycle.
    """

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
    # Public API
    # ------------------------------------------------------------------

    async def ingest_image(
        self,
        image_bytes: bytes,
        image_path: str,
        source: str = 'unknown',
        *,
        prefilled_image: DecodedImage | None = None,
        prefilled_frame: GpuDecodeFrame | None = None,
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

        decoded: DecodedImage | None = None
        if prefilled_frame is not None:
            pass
        elif prefilled_image is not None:
            decoded = prefilled_image
        else:
            try:
                decoded = await run_cpu(
                    _decode_image, image_bytes, nbytes=encoded_nbytes(image_bytes)
                )
            except Exception as exc:
                return IngestResult(
                    status='failed',
                    image_path=image_path,
                    source_identifier=source_identifier,
                    error=str(exc),
                    error_kind=ERROR_KIND_DECODE_FAILED,
                )

        if decoded is not None:
            width, height = decoded.pil.size
        else:
            assert prefilled_frame is not None
            width, height = prefilled_frame.width, prefilled_frame.height

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
            pil=decoded.pil if decoded is not None else None,
            rgb=decoded.rgb if decoded is not None else None,
            gpu_frame=prefilled_frame,
            width=width,
            height=height,
            imohash=image_hash,
            source=source,
            source_identifier=source_identifier,
            ingest_run_id=ingest_run_id,
        )

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
        prefilled_image: DecodedImage | None = None,
        prefilled_items: list[DetectedItem] | None = None,
        prefilled_secondary: SecondaryOutput | None = None,
        prefilled_frame: GpuDecodeFrame | None = None,
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
            prefilled_image: The already-decoded frame; skips the decode.
            prefilled_items: Primary-detector output from
                :meth:`_run_primary_detector_batch`; skips this image's
                own single-image Triton round-trip. An empty list is
                meaningful (the detector found nothing) and is *not*
                treated as "not prefilled".
            prefilled_secondary: Secondary-detector output (raw tensor
                plus optional backbone feature map) from
                :meth:`WholeImageDetector.run_secondary_raw_batch`; same deal.
            prefilled_frame: The GPU decode's products for this image
                (:mod:`~src.services.curation.ingest_gpu_decode`), with its
                items in ``prefilled_items``; the API never decodes it.

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
            prefilled_frame=prefilled_frame,
            source_identifier=source_identifier,
            ingest_run_id=ingest_run_id,
        )
        if isinstance(ctx, IngestResult):
            return record_ingest_result(ctx)

        try:
            if ctx.pil is None:
                detected = self.filter_detections(prefilled_items or [], ctx.width, ctx.height)
            else:
                detected = await self.detect_items(
                    ctx.pil,
                    image_path=image_path,
                    prefilled_items=prefilled_items,
                    prefilled_secondary=prefilled_secondary,
                )
        except Exception as exc:
            logger.error('ingest_primary_detector_failed', path=image_path, error=str(exc))
            return record_ingest_result(
                IngestResult(
                    status='failed',
                    image_path=image_path,
                    source_identifier=source_identifier,
                    error=str(exc),
                    error_kind=ERROR_KIND_DETECTOR_INFER,
                )
            )

        outcome = await index_items(
            self,
            ctx,
            detected.items,
            secondary_detector_error=detected.secondary_detector_error,
        )
        outcome.result.n_filtered = detected.n_filtered
        return record_ingest_result(outcome.result)

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

    async def refresh_indexes(self) -> None:
        """Make this call's writes visible to search (stats, previews) once the
        ingest call ends: one refresh per call, never per item. A failure is
        logged, not raised: the writes already succeeded."""
        try:
            await self.opensearch.indices.refresh(
                index=f'{self.config.images_index},{self.config.items_index}'
            )
        except Exception as exc:
            logger.warning('ingest_refresh_failed', error=str(exc))

    async def ingest_batch(
        self,
        images: list[bytes],
        image_paths: list[str],
        source: str = 'batch',
        source_identifiers: list[str | None] | None = None,
        ingest_run_id: str | None = None,
    ) -> BatchIngestResult:
        """Batch ingest: msearch dedup, batched detector inference, per-image finish.

        Delegates to :func:`src.services.curation.ingest_batch.run_ingest_batch`
        — see that module's docstring for why the batch path is more than
        ``ingest_one`` run N times concurrently. To ingest an
        already-labeled dataset, use ``POST /datasets/imports``
        (planned, route not yet built).

        Args:
            images: Raw image bytes, one per entry.
            image_paths: Source paths, index-aligned with ``images``.
            source: Provenance tag stamped on every document.

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
            source_identifiers=source_identifiers,
            ingest_run_id=ingest_run_id,
        )


__all__ = [
    'PARKED_CLUSTER_ID',
    'RESIDUAL_CLUSTER_ID_OFFSET',
    'SECONDARY_IOU_MATCH',
    'BatchIngestResult',
    'CurationIngestService',
    'IngestResult',
    'IngestSummary',
]
