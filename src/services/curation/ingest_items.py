"""Ingest items: detection of one image's items and the bulk write of its documents.

:class:`IngestItemsMixin` is mixed into
:class:`~src.services.curation.ingest.CurationIngestService`, which provides the
collaborators declared below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy.exc import SQLAlchemyError, TimeoutError as PoolTimeoutError

from src.config import get_region_fields
from src.core.logging import get_logger
from src.db.repos.classes import ClassRegistryError
from src.services.curation.ingest_policy import (
    IngestPolicy,
    apply_detect_filter,
    assign_classes_by_name,
    awaits_class_by_name,
    registry_name_index,
)
from src.services.curation.ingest_writer import IngestWriteError, RepairTarget, get_ingest_writer
from src.services.curation.ops_metrics import count_batch_fallback
from src.services.detection.geometry import letterbox_params
from src.utils.class_names import class_name_deprecation_index, normalize_class_name
from src.utils.retry import TRITON_CALL_ERRORS
from src.utils.stage_timing import stage_timer


if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import Image

    from src.clients.curation_opensearch import ClassRegistryFile
    from src.config import CurationConfig, DetectionProfile
    from src.services.curation.ingest_detect import WholeImageDetector
    from src.services.curation.item_doc import DetectedItem


logger = get_logger(__name__)

#: What a failed class create is: the database, or the registry refusing a name.
CLASS_CREATE_ERRORS = (SQLAlchemyError, OSError, ClassRegistryError, ValueError)


class ClassEnsurer(Protocol):
    """Adds the registry classes for detector labels, all in one transaction: ``wanted`` holds
    ``(name, label)`` pairs (``name`` in registry-name form, ``label`` as the detector spells it);
    answers ``{name: (class_id, class_name)}``. A name the registry has is not added again."""

    async def __call__(self, wanted: Sequence[tuple[str, str]]) -> dict[str, tuple[int, str]]: ...


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
    registry: ClassRegistryFile
    profile: DetectionProfile
    secondary_profile: DetectionProfile | None
    config: CurationConfig
    detector: WholeImageDetector
    region_seed_status: Any
    policy: IngestPolicy
    # Creates a detector label's registry class; ``None`` resolves only against the registry as
    # it is (a caller with no registry to write to).
    class_ensurer: ClassEnsurer | None
    # Classes this service already created, by registry name: one registry write per name.
    ensured_classes: dict[str, tuple[int, str]]

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
        repair: RepairTarget | None = None,
    ) -> dict[str, int]:
        """Write 1 images doc (create) + N items docs (OCC upsert) through the process's
        coalescing writer (:mod:`~src.services.curation.ingest_writer`), which batches them
        with other images' documents and answers for this image alone. ``duplicate`` is 1
        when the images doc already existed and is whole: then nothing was written;
        ``incomplete`` is 1 when it exists and misses items a live writer may still be
        writing: nothing was written either.

        ``repair`` completes an image already stored incomplete (``image_doc`` is then None): the
        items no existing item stands for are created, none is rewritten; ``repaired`` is 1 when
        this call completed one, also when a 409 on the create found an incomplete doc nobody is
        writing.

        ``created_ids`` (optional out-list) receives the crop_id of every items doc this call
        newly created. A unit with any document refused, or whose flush failed, raises
        :class:`~src.services.curation.ingest_writer.IngestWriteError`: a photo is reported
        stored only when all of it is.
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
                repair=repair,
            )
        if unit.refused or unit.failure:
            logger.warning('ingest_bulk_failed', refused=unit.refused[:3], failure=unit.failure)
            reason = '; '.join(unit.refused[:3]) or str(unit.failure)
            raise IngestWriteError(reason, release_claim=unit.image_absent)
        if created_ids is not None:
            created_ids.extend(unit.created_ids)
        queued = 0
        if seeding:
            status_field = get_region_fields().status
            seeded = {d['crop_id'] for d in crop_docs if d.get(status_field) is not None}
            queued = len(seeded.intersection(unit.created_ids)) + unit.filled_absent
        return {
            'duplicate': int(unit.duplicate),
            'incomplete': int(unit.incomplete),
            'repaired': int(unit.repaired),
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
        resolve: bool = True,
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
        items = await self.detector.run_primary(img)

        # surfaced on the response (IngestResult.secondary_detector_error)
        # and rolled into IngestSummary.secondary_detector_failures instead of
        # only ever reaching a 'warning' log line.
        secondary_detector_error: str | None = None
        if self.secondary_profile is not None and items:
            secondary = None
            try:
                secondary = await self.detector.run_secondary_raw(img)
            except TRITON_CALL_ERRORS as exc:
                secondary_detector_error = str(exc)
                logger.warning('ingest_secondary_detector_failed', path=image_path, error=str(exc))
            if secondary is not None:
                sec_scale, sec_pad = letterbox_params(img, target=self.secondary_profile.input_size)
                try:
                    self.detector.resolve_with_secondary(items, secondary.raw, sec_scale, sec_pad)
                except (ValueError, IndexError) as exc:
                    secondary_detector_error = str(exc)
                    logger.warning(
                        'ingest_secondary_resolve_failed', path=image_path, error=str(exc)
                    )
        return await self.filter_detections(
            items, img.width, img.height, secondary_detector_error, resolve=resolve
        )

    async def filter_detections(
        self,
        items: list[DetectedItem],
        width: int,
        height: int,
        secondary_detector_error: str | None = None,
        *,
        resolve: bool = True,
    ) -> DetectResult:
        """The project's detect filter and class resolution over detected items.

        ``resolve=False`` leaves the class resolution to the caller, which resolves the items of
        a whole batch at once (:meth:`resolve_classes`)."""
        items, n_filtered = apply_detect_filter(items, self.policy.detect, width, height)
        if resolve:
            [error] = await self.resolve_classes([items])
            if error is not None:
                raise error
        return DetectResult(items, secondary_detector_error, n_filtered)

    async def resolve_classes(
        self, item_lists: Sequence[list[DetectedItem]]
    ) -> list[Exception | None]:
        """Give every item its registry class by name (``detect.class_resolution = by_name``),
        creating the classes the registry lacks first, all of them in one transaction for all
        the lists. Answers, per list, the error that kept a label of it from getting its class
        (the list's items stay without that class), else ``None``.

        A deprecated class of a label's name, or the region class, counts as known: neither is
        brought back or duplicated (the rule ``seed_from_detector`` follows). A name this service
        created is not created again; a name another request created is in the registry snapshot
        the service was built with, or is found by name by the idempotent create. When the one
        transaction fails, each name is created alone, so a failure stays with the lists that
        carry that name.
        """
        if self.policy.detect.class_resolution != 'by_name':
            return [None] * len(item_lists)
        low_conf = f'{self.profile.name}_low_conf'
        failed: dict[str, Exception] = {}
        missing = self._missing_classes(item_lists, low_conf)
        if missing and self.class_ensurer is not None:
            with stage_timer('class_create'):
                try:
                    self.ensured_classes.update(await self.class_ensurer(list(missing.items())))
                except PoolTimeoutError as exc:  # no connection to retry with: fail them all now
                    failed = dict.fromkeys(missing, exc)
                except CLASS_CREATE_ERRORS as exc:
                    count_batch_fallback('class_create', exc)
                    logger.warning('ingest_batch_class_create_fell_back', n_names=len(missing))
                    for name, label in missing.items():
                        try:
                            self.ensured_classes.update(await self.class_ensurer([(name, label)]))
                        except CLASS_CREATE_ERRORS as one:
                            failed[name] = one
        index = {**registry_name_index(self.registry), **self.ensured_classes}
        for items in item_lists:
            assign_classes_by_name(items, index, low_conf)
        return [self._first_failure(items, failed, low_conf) for items in item_lists]

    def _missing_classes(
        self, item_lists: Sequence[list[DetectedItem]], low_conf: str
    ) -> dict[str, str]:
        """``{name: label}`` of the detector labels of these items whose class nobody has."""
        known = class_name_deprecation_index(self.registry.classes)
        missing: dict[str, str] = {}
        for items in item_lists:
            for item in items:
                if not awaits_class_by_name(item, low_conf):
                    continue
                label = item.proposal_name or ''
                slug = normalize_class_name(label)
                if slug and slug not in known and slug not in self.ensured_classes:
                    missing.setdefault(slug, label)
        return missing

    @staticmethod
    def _first_failure(
        items: list[DetectedItem], failed: dict[str, Exception], low_conf: str
    ) -> Exception | None:
        for item in items:
            if awaits_class_by_name(item, low_conf):
                error = failed.get(normalize_class_name(item.proposal_name or ''))
                if error is not None:
                    return error
        return None
