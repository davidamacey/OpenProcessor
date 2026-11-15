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

1. Decode (EXIF-transposed) + exact-content dedup (``imohash`` against ``image_contents``, confirmed by SHA-256).
2. Run the primary detector (:class:`~src.config.DetectionProfile`,
   a fused model whose ``[300, 6]`` Triton output is already NMS'd) over the
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
   (:mod:`~src.services.detection.crop_quality`), and the item's place in the serving
   partition (:func:`~src.services.clustering.assign.place_new`).
6. Bulk index: the images doc is a ``create`` under an id derived from the
   content (:func:`~src.services.near_duplicates.content_key.content_image_id`), so
   a copy that raced past step 1's lookup (a search does not see an
   unrefreshed write) is refused 409 and reported ``duplicate`` with none of
   its items or crops stored (#358). Items docs go through
   :func:`~src.clients.occ_upsert.occ_upsert_bulk` with human-field guards so a
   re-ingest never clobbers a human-applied label. With a region profile
   active, new items are seeded ``pending_detection`` (never overwriting
   an existing region status) so the region-detection worker picks them
   up — see :func:`~src.services.curation.item_doc.region_seed_status`.
7. Crop-cache write (:func:`~src.services.curation.source_image_cache.write_crop_cache`).
8. After a successful write, one advisory ``crop.created`` event per
   newly created item on the in-process event hub
   (:func:`~src.services.events.publish.apublish_crops_created`).

Sibling modules, split out of this one to keep each to one concern:

* :mod:`src.services.curation.ingest_helpers` — the
  upright decode and ``MAX_INGEST_CONCURRENCY``.
* :mod:`src.services.curation.ingest_dedup` — the dedup lookups
  (:class:`IngestDedupMixin`).
* :mod:`src.services.curation.ingest_items` — detecting one image's items and
  bulk-writing its documents (:class:`IngestItemsMixin`, ``DetectResult``).
* :mod:`src.services.curation.ingest_models` — the wire models.
* :mod:`src.services.curation.ingest_detect` — steps 2-3 of the host stage, the
  ``DetectionProfile``-driven Triton detector runner (one image per call) and its tensor decoding.
* :mod:`src.services.curation.ingest_gpu_decode` — the frame engine as the decoder, detector and
  embedder of the engine stage (every photo of a project without a secondary detector).
* :mod:`src.services.curation.ingest_batch` — ``ingest_batch``'s implementation: one lookup of the
  stored contents, admission, then the flow per wave.
* :mod:`src.services.curation.ingest_stage` and :mod:`src.services.curation.ingest_finish` — one
  photo's stage (engine or host) and the flow of a request (``ingest_one`` is a request of one
  photo): each photo moves on its own to its write, the PostgreSQL steps (classes, placement,
  content claims with the pending rows) run in at most three commit groups, then the release and
  the events once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException

from src.config import get_curation_config
from src.core.exceptions import EngineUnavailableError
from src.core.logging import get_logger
from src.domain.capture_metadata import unknown_capture
from src.domain.decode_route import CPU_HOST
from src.services.curation.ingest_dedup import IngestDedupMixin, Resolved, held_back_result
from src.services.curation.ingest_detect import SECONDARY_IOU_MATCH, WholeImageDetector
from src.services.curation.ingest_finish import PhotoJob, finish_photos
from src.services.curation.ingest_helpers import _decode_image
from src.services.curation.ingest_index import ImageContext, IndexOutcome, index_items
from src.services.curation.ingest_items import DetectResult, IngestItemsMixin
from src.services.curation.ingest_models import (
    ERROR_KIND_DECODE_FAILED,
    ERROR_KIND_EMPTY,
    ERROR_KIND_ENGINE_UNAVAILABLE,
    BatchIngestResult,
    IngestResult,
    IngestSummary,
)
from src.services.curation.ingest_policy import IngestPolicy
from src.services.curation.ingest_writer import RepairTarget
from src.services.curation.item_doc import DetectedItem, region_seed_status
from src.services.events.publish import apublish_crops_created
from src.services.near_duplicates.content_key import content_image_id, content_imohash
from src.utils.cpu_pool import encoded_nbytes, run_cpu
from src.utils.image_decode import IMAGE_DECODE_ERRORS
from src.utils.stage_timing import stage_timer


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from opensearchpy import AsyncOpenSearch

    from src.clients.curation_opensearch import ClassRegistryFile
    from src.clients.pe_encoder import PEEncoder
    from src.clients.triton_pool_core import AsyncTritonPool
    from src.config import CurationConfig, DetectionProfile
    from src.domain.capture_metadata import CaptureMetadata
    from src.services.curation.ingest_gpu_decode import GpuDecodeFrame
    from src.services.curation.ingest_items import ClassEnsurer
    from src.utils.image_decode import DecodedImage


logger = get_logger(__name__)


@dataclass(frozen=True)
class ImageIdentity:
    """Who a file is: its ``imohash`` and its content's lookup (``resolved``)."""

    image_hash: str
    resolved: Resolved
    adopt_existing: bool = False


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
        registry: ClassRegistryFile,
        profile: DetectionProfile,
        pe_encoder: PEEncoder | Any,
        secondary_profile: DetectionProfile | None = None,
        config: CurationConfig | None = None,
        policy: IngestPolicy | None = None,
        capture_reader: Callable[[bytes], Awaitable[CaptureMetadata]] | None = None,
        location_policy: Callable[[], Awaitable[str]] | None = None,
        class_ensurer: ClassEnsurer | None = None,
        frame_engine: AsyncTritonPool | Any | None = None,
    ) -> None:
        # Accept either a wrapper exposing `.client` or a raw AsyncOpenSearch.
        self.opensearch = getattr(opensearch, 'client', opensearch)
        self.triton_pool = triton_pool
        # A pool on the frame engine's address (src/clients/frame_engine.py); None: no GPU decode.
        self.frame_engine = frame_engine
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
        )
        self.region_seed_status = region_seed_status()
        # The project's detect filter and embedding policy; defaults = today's behaviour.
        self.policy = policy or IngestPolicy()
        self.class_ensurer = class_ensurer
        self.ensured_classes = {}
        # How a new image's capture metadata is read and which location policy it is stored under.
        # Without a reader the capture is unknown (the capture backfill fills it); without a
        # policy source no coordinate is stored (fail closed).
        self._capture_reader = capture_reader
        self._location_policy = location_policy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def identify(
        self,
        image_bytes: bytes,
        image_path: str,
        source_identifier: str | None,
        *,
        adopt_existing: bool = False,
        known_existing: dict[str, Any] | None = None,
        resolved: Resolved | None = None,
    ) -> ImageIdentity | IngestResult:
        """Fingerprint and dedup one file, without touching its pixels.

        Returns the :class:`ImageIdentity` the context is built from, or a terminal
        :class:`IngestResult`: ``failed`` for empty bytes, ``duplicate`` for an image the index
        holds whole, ``incomplete_write`` for one a writer may still be writing. ``known_existing``
        is a lookup result the caller already has (an image it indexed moments ago that search
        cannot see yet); ``resolved`` is the whole lookup of this file the caller already made (a
        batch resolves all its files at once). ``adopt_existing=True`` (a dataset import, which
        must attach labels to an already-indexed image) never answers a duplicate and never
        repairs.
        """
        if not image_bytes:
            return IngestResult(
                status='failed',
                image_path=image_path,
                source_identifier=source_identifier,
                error='empty image bytes',
                error_kind=ERROR_KIND_EMPTY,
            )
        image_hash = content_imohash(image_bytes)
        if resolved is None:
            resolved = (
                Resolved(known_existing, image_hash)
                if known_existing is not None
                else await self._lookup_image(image_hash, image_bytes)
            )
        existing = resolved.existing
        if existing is not None and not adopt_existing:
            if resolved.completeness == 'in_flight':
                return held_back_result(resolved, image_path, image_hash, source_identifier)
            if resolved.completeness == 'complete':
                return IngestResult(
                    status='duplicate',
                    image_id=existing['image_id'],
                    image_path=image_path,
                    source_identifier=source_identifier,
                    imohash=image_hash,
                )
        return ImageIdentity(image_hash, resolved, adopt_existing)

    def context_for(
        self,
        identity: ImageIdentity,
        image_bytes: bytes,
        image_path: str,
        source: str,
        *,
        width: int,
        height: int,
        decode_route: str,
        gpu_frame: GpuDecodeFrame | None = None,
        decoded: DecodedImage | None = None,
        capture: CaptureMetadata | None = None,
        source_identifier: str | None = None,
        ingest_run_id: str | None = None,
    ) -> ImageContext:
        """The :class:`ImageContext` of an identified image whose pixels were decoded (by the
        engine: ``gpu_frame``; or in the API: ``decoded``). ``capture`` is the file's capture
        metadata for a new image."""
        resolved = identity.resolved
        existing = resolved.existing
        repair: RepairTarget | None = None
        if existing is None:
            image_id = content_image_id(resolved.key)
            resolved_path = image_path
        else:
            image_id = existing['image_id']
            resolved_path = existing.get('image_path') or image_path
            if not identity.adopt_existing:  # an incomplete image nobody is writing: complete it
                repair = RepairTarget(image_id, resolved.listed)
        return ImageContext(
            image_id=image_id,
            image_path=resolved_path,
            created=existing is None,
            dataset_split=(existing or {}).get('dataset_split'),
            pil=decoded.pil if decoded is not None else None,
            rgb=decoded.rgb if decoded is not None else None,
            gpu_frame=gpu_frame,
            decode_route=decode_route,
            width=width,
            height=height,
            imohash=identity.image_hash,
            content=image_bytes,
            source=source,
            source_identifier=source_identifier,
            ingest_run_id=ingest_run_id,
            capture=capture,
            repair=repair,
        )

    async def ingest_image(
        self,
        image_bytes: bytes,
        image_path: str,
        source: str = 'unknown',
        *,
        source_identifier: str | None = None,
        ingest_run_id: str | None = None,
        adopt_existing: bool = False,
        known_existing: dict[str, Any] | None = None,
        resolved: Resolved | None = None,
    ) -> ImageContext | IngestResult:
        """Decode (in the API), fingerprint and dedup one image: the host stage's first step, and
        the first step of a dataset import and the reprocess scopes.

        Returns the :class:`ImageContext` :func:`index_items` consumes, or a terminal
        :class:`IngestResult` (see :meth:`identify`; ``decode_failed`` for undecodable bytes).
        ``adopt_existing=True`` turns a duplicate into a context with ``created=False`` carrying
        the existing doc's ``image_id`` and ``image_path``; it never repairs an incomplete image.
        Without it, a stored image that misses items comes back as a context with ``created``
        False and ``repair`` set: the ingest completes the image.
        """
        decoded: DecodedImage | None = None
        if image_bytes:
            try:
                decoded = await run_cpu(
                    _decode_image, image_bytes, nbytes=encoded_nbytes(image_bytes)
                )
            except IMAGE_DECODE_ERRORS as exc:
                return IngestResult(
                    status='failed',
                    image_path=image_path,
                    source_identifier=source_identifier,
                    error=str(exc),
                    error_kind=ERROR_KIND_DECODE_FAILED,
                )
        identity = await self.identify(
            image_bytes,
            image_path,
            source_identifier,
            adopt_existing=adopt_existing,
            known_existing=known_existing,
            resolved=resolved,
        )
        if isinstance(identity, IngestResult):
            return identity
        assert decoded is not None
        capture: CaptureMetadata | None = None
        if identity.resolved.existing is None:
            with stage_timer('capture'):
                capture = await self.read_capture(image_bytes)
        width, height = decoded.pil.size
        return self.context_for(
            identity,
            image_bytes,
            image_path,
            source,
            width=width,
            height=height,
            decode_route=CPU_HOST,
            decoded=decoded,
            capture=capture,
            source_identifier=source_identifier,
            ingest_run_id=ingest_run_id,
        )

    async def read_capture(self, image_bytes: bytes) -> CaptureMetadata:
        """The capture metadata of a new image's file (unknown when no reader is wired)."""
        if self._capture_reader is None:
            return unknown_capture()
        return await self._capture_reader(image_bytes)

    async def location_policy(self) -> str:
        """The location policy a new image is stored under (``drop`` when none is wired)."""
        return 'drop' if self._location_policy is None else await self._location_policy()

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
        source_identifier: str | None = None,
        ingest_run_id: str | None = None,
    ) -> IngestResult:
        """Run the full pipeline on a single image: a request of one photo through the same flow
        as a batch (:func:`~src.services.curation.ingest_finish.finish_photos`).

        Args:
            image_bytes: Raw image bytes.
            image_path: Source path, persisted verbatim on the images doc.
            source: Free-form provenance tag.

        Raises:
            EngineUnavailableError: the frame engine could not take the photo and nothing was
                stored (a person is waiting: the route answers 503 with ``Retry-After``).
        """
        [result] = await self.ingest_photos(
            [
                PhotoJob(
                    image_bytes,
                    image_path,
                    source,
                    source_identifier=source_identifier,
                    ingest_run_id=ingest_run_id,
                )
            ],
        )
        if result.error_kind == ERROR_KIND_ENGINE_UNAVAILABLE:
            msg = result.error or 'the image engine is unavailable'
            raise EngineUnavailableError(msg)
        return result

    async def ingest_photos(self, jobs: Sequence[PhotoJob]) -> list[IngestResult]:
        """Ingest a request's photos (:func:`~src.services.curation.ingest_finish.finish_photos`):
        one result per job, in order; each photo moves on its own, the database steps run in at
        most three commit groups."""
        return await finish_photos(self, jobs)

    async def detect_for_ingest(
        self, ctx: ImageContext, image_path: str
    ) -> DetectResult | Exception:
        """The host stage's detector items for a decoded image, the project's detect filter
        applied; the class names are left to :meth:`resolve_classes`, which the committer runs
        per commit group. A failure of the detector is returned (the photo fails, the others go
        on)."""
        try:
            return await self.detect_items(ctx.require_pil(), image_path=image_path, resolve=False)
        except Exception as exc:
            return exc

    @staticmethod
    async def _publish_created(crops: Sequence[tuple[str, str]]) -> None:
        """Announce newly written items, ``(crop_id, image_path)`` each, on the live-update event
        stream.

        Only ids ``occ_upsert_bulk`` confirmed as created are published —
        a re-ingest update or a rejected create is not a new crop. One
        transaction per call (a batch announces all its crops in one); a failed write is
        raised, not swallowed.
        """
        await apublish_crops_created(crops)

    async def refresh_indexes(self) -> None:
        """Make this call's writes visible to search at once, for a caller that promises it
        (``/ingest/image``, ``/ingest/upload``, ``/ingest/objects``): one refresh per call,
        never per item, and none in a batch ingest. A failure is logged, not raised: the writes
        already succeeded."""
        try:
            await self.opensearch.indices.refresh(
                index=f'{self.config.images_index},{self.config.items_index}'
            )
        except OpenSearchException as exc:
            logger.warning('ingest_refresh_failed', error=str(exc))

    async def ingest_batch(
        self,
        images: list[bytes],
        image_paths: list[str],
        source: str = 'batch',
        source_identifiers: list[str | None] | None = None,
        ingest_run_id: str | None = None,
    ) -> BatchIngestResult:
        """Batch ingest: one dedup lookup, then every new photo through the per-photo flow.

        Delegates to :func:`src.services.curation.ingest_batch.run_ingest_batch`
        — see that module's docstring for why the batch path is more than
        ``ingest_one`` run N times concurrently. To ingest an
        already-labeled dataset, use ``POST /datasets/imports``.

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
    'SECONDARY_IOU_MATCH',
    'BatchIngestResult',
    'CurationIngestService',
    'IngestResult',
    'IngestSummary',
]
