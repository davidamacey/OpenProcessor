"""Ingest items: detection of one image's items and the bulk write of its documents.

:class:`IngestItemsMixin` is mixed into
:class:`~src.services.curation.ingest.CurationIngestService`, which provides the
collaborators declared below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config import get_region_fields
from src.core.logging import get_logger
from src.services.curation.ingest_policy import (
    IngestPolicy,
    apply_detect_filter,
    assign_classes_by_name,
    registry_name_index,
)
from src.services.curation.ingest_writer import get_ingest_writer
from src.services.detection.geometry import letterbox_params
from src.utils.stage_timing import stage_timer


if TYPE_CHECKING:
    from PIL import Image

    from src.clients.curation_opensearch.registry import ClassRegistry
    from src.config import CurationConfig, DetectionProfile
    from src.services.curation.ingest_detect import SecondaryOutput, WholeImageDetector
    from src.services.curation.item_doc import DetectedItem


logger = get_logger(__name__)


@dataclass
class DetectResult:
    """``detect_items``' output: the items to store, a secondary failure (not
    raised; the primary's items stand) and how many detections the project's
    detect filter dropped."""

    items: list[DetectedItem]
    secondary_detector_error: str | None = None
    n_filtered: int = 0


class IngestItemsMixin:
    """Detection and bulk-write steps of :class:`~src.services.curation.ingest.CurationIngestService`."""

    opensearch: Any
    registry: ClassRegistry | Any
    profile: DetectionProfile
    secondary_profile: DetectionProfile | None
    config: CurationConfig
    detector: WholeImageDetector
    region_seed_status: Any
    policy: IngestPolicy

    # Fields whose presence on an existing items doc signals a human
    # write that ingest must never clobber on re-ingest.
    _CROP_HUMAN_FIELD_GUARDS = ('label_source', 'class_source')

    async def _bulk_index(
        self,
        image_doc: dict[str, Any] | None,
        crop_docs: list[dict[str, Any]],
        created_ids: list[str] | None = None,
        *,
        seed_region: bool = True,
    ) -> dict[str, int]:
        """Write 1 images doc (create) + N items docs (OCC upsert) through the process's
        coalescing writer (:mod:`~src.services.curation.ingest_writer`), which batches them
        with other images' documents and answers for this image alone. ``duplicate`` is 1
        when the images doc already existed: then nothing was written.

        ``created_ids`` (optional out-list) receives the crop_id of every
        items doc this call newly created. A flush that fails as a whole
        raises, as the per-image path did.
        """
        seeding = seed_region and self.region_seed_status is not None
        with stage_timer('opensearch_write'):
            unit = await get_ingest_writer(self.opensearch).write(
                image_doc,
                crop_docs,
                images_index=self.config.images_index,
                items_index=self.config.items_index,
                guards=self._CROP_HUMAN_FIELD_GUARDS,
                fill_if_absent=(get_region_fields().status,) if seeding else (),
            )
        if unit.refused:
            logger.warning('ingest_bulk_partial_errors', refused=unit.refused[:3])
        if created_ids is not None:
            created_ids.extend(unit.created_ids)
        queued = 0
        if seeding:
            status_field = get_region_fields().status
            seeded = {d['crop_id'] for d in crop_docs if d.get(status_field) is not None}
            queued = len(seeded.intersection(unit.created_ids)) + unit.filled_absent
        return {
            'duplicate': int(unit.duplicate),
            'images_indexed': unit.images_indexed,
            'crops_created': unit.created,
            'crops_updated': unit.updated,
            'crops_preserved_human': unit.preserved_human,
            'crops_final_conflicts': unit.final_conflicts,
            'region_queued': queued,
        }

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

        # surfaced on the response (IngestResult.secondary_detector_error)
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
        return self.filter_detections(items, img.width, img.height, secondary_detector_error)

    def filter_detections(
        self,
        items: list[DetectedItem],
        width: int,
        height: int,
        secondary_detector_error: str | None = None,
    ) -> DetectResult:
        """The project's detect filter and class resolution over detected items."""
        items, n_filtered = apply_detect_filter(items, self.policy.detect, width, height)
        if self.policy.detect.class_resolution == 'by_name':
            assign_classes_by_name(items, registry_name_index(self.registry), self.profile.name)
        return DetectResult(items, secondary_detector_error, n_filtered)
