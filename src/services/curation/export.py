"""Generic YOLO-format dataset export service.

Splits into two halves (see ``docs/design/curation_design_rationale.md``
for the design approach): artifact filenames, class lists and split
ratios are deployment data (an ``ExportProfile``, extracted here), while
the YOLO-format writer, split logic and manifest/checksum mechanism are
generic algorithm code that stays code. Domain-specific
whole-frame-vs-crop / near-dup-collapsing export features are
intentionally NOT reproduced here; a deployment-specific overlay can
extend :class:`GenericYoloExportService` directly if it needs them — the
generic curation stack ships with a thinner export path by design,
tracked as the most likely first follow-up after any such feature is
needed.

**Layout: one image, one label file, per source image.** Validated items
(one object each) are grouped by ``image_id``
(:mod:`src.services.curation.export_images`); each exported image gets one
``images/<split>/<image_id>.<ext>`` and one ``labels/<split>/<image_id>.txt``
carrying a ``cls cx cy w h`` line for every validated, non-excluded,
non-dismissed object on it. Writing a frame once per object, each copy
labeled with only that object, taught the detector that the frame's other
objects were background.

**Partial frames.** A frame can also hold objects the export does not
label (unreviewed, or on a class with no dense id). By default the frame
is exported with its validated objects labeled, and the manifest counts
the rest (``unlabeled_items_on_exported_images``,
``images_with_unlabeled_items``) so training preflight can warn that they
will be learned as background. ``require_fully_labeled_images=True``
drops such frames instead and records how many
(``images_dropped_not_fully_labeled``).

Split assignment is a deterministic, **stratified** hash-order bucket
rather than a stored crop->split mapping, so re-running an export with the
same recorded seed reproduces the same split without needing to persist
per-item split assignments anywhere. The split groups on ``image_id``, so
every object of an image lands in that image's split. An image carrying a
frozen ``test_holdout`` item always lands in the ``test`` split, honoring
whatever holdout freeze a deployment has already committed to (see
``src.services.curation.holdout``); a class with a frozen holdout splits
its other images between train and val only. The exact per-class rules
are on :func:`~src.services.curation.export_split.stratified_split`.

Dense export ids (``class_registry.json:export_id_map``) are resolved from
the live :class:`~src.clients.curation_opensearch.registry.ClassRegistry` at export
time and frozen into the export directory, so a subset-training request's
``include_classes`` (always expressed in REGISTRY ids) can be translated to
the dense ids that actually appear in the written label files — see
``src/services/training/preflight_scan.py`` and
``src/services/training/preflight_checks.py``, both of which read this file back.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.clients.curation_opensearch.registry import ClassRegistry, get_class_registry
from src.config import CurationConfig, get_curation_config
from src.core.logging import get_logger
from src.services.curation.export_images import copy_export_images, select_export_images
from src.services.curation.export_models import ARTIFACT_FILENAMES, ExportProfile, ExportResult
from src.services.curation.export_negatives import (
    plan_multi_class_negatives,
    scroll_negative_frames,
    write_negative_frames,
)
from src.services.curation.export_readiness import (
    MANIFEST_GENERATION_KEY,
    NO_VALIDATED_ITEMS_REASON,
    NothingToExportError,
    items_index_generation,
    validated_export_query,
)
from src.services.curation.export_retention import prune_exports_after_write
from src.services.curation.export_rows import dense_class_names, hits_to_rows, to_dense_ids
from src.services.curation.export_split import (
    DEFAULT_SPLIT_GROUP_KEY,
    SplitMode,
    pins_from_rows,
    stratified_split_with_stats,
)
from src.services.curation.export_support import (
    _code_sha,
    _ExportRow,
    allocate_export_dir,
    atomic_symlink_flip,
    atomic_write_text,
    frozen_test_sha_of,
    label_content_sha,
    scroll_hits,
)
from src.services.curation.export_writers import (
    WrittenLabels,
    class_split_rows,
    source_frozen_manifest,
    write_data_yaml,
    write_image_labels,
    write_label_stats,
    write_registry_snapshot,
)
from src.services.curation.holdout import compute_holdout_sha


if TYPE_CHECKING:
    from src.services.curation.export_images import _ExportImage
    from src.services.curation.item_filter import ItemFilter


logger = get_logger(__name__)


def resolve_current_export_dir(config: CurationConfig | None = None) -> Path:
    """Resolve the current export directory via the ``current`` symlink.

    Raises:
        FileNotFoundError: no ``current`` symlink/dir exists yet.
    """
    cfg = config or get_curation_config()
    current = cfg.export_root / 'current'
    if not (current.is_symlink() or current.exists()):
        msg = 'no current export symlink found'
        raise FileNotFoundError(msg)
    return current.resolve()


class GenericYoloExportService:
    """Builds a YOLO-format detection dataset from the curation items index.

    Writes one label ``.txt`` per exported source image (one line per
    object: dense export class id + normalized cx/cy/w/h), copies
    (optionally resizing) that image's pixels once into
    ``images/<split>/``, a ``data.yaml`` class map, a
    ``class_registry.json`` snapshot + dense ``export_id_map``, a
    ``label_stats.json`` per-class object count, and a ``manifest.json``
    reproducibility envelope (dataset checksum, image and object counts
    overall, per split and per class, unlabeled-object counts, split
    group key, seed, code sha, frozen-holdout sha, timestamps). Deliberately narrower than the
    reference exporter — see module docstring.
    """

    def __init__(
        self,
        opensearch: Any,
        *,
        config: CurationConfig | None = None,
        profile: ExportProfile | None = None,
        registry: ClassRegistry | None = None,
    ) -> None:
        self.opensearch = opensearch
        self.config = config or get_curation_config()
        self.profile = profile or ExportProfile()
        self.registry = registry or get_class_registry()

    async def _scroll_items(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        return await scroll_hits(
            self.opensearch,
            index=self.config.items_index,
            query=query,
            source=[
                'crop_id',
                'image_id',
                'image_path',
                'bbox_norm',
                'class_id',
                'class_name',
                'test_holdout',
                'dataset_split',
                'import_source_stem',
            ],
        )

    async def export_dataset(
        self,
        *,
        export_dir: Path | None = None,
        version_tag: str = '',
        seed: int = 42,
        max_images: int | None = None,
        dedup_threshold: float | None = None,
        require_fully_labeled_images: bool = False,
        resize_mode: Literal['aspect'] | None = None,
        image_size: int = 640,
        copy_images: bool = True,
        max_image_workers: int = 4,
        split_mode: SplitMode = 'keep_imported',
        include_negative_frames: bool = True,
        item_filter: ItemFilter | None = None,
    ) -> ExportResult:
        """Export every validated, non-excluded, non-dismissed item as a
        multi-class YOLO detection dataset, one image + one label file per
        source image.

        Raises :class:`NothingToExportError` (before writing anything) when
        no image survives the selection: no validated item, none with an
        image, a box and a class, every one on an unregistered / deprecated
        class, or (with ``require_fully_labeled_images``) no fully labeled
        image.

        Splits are assigned per image (:func:`stratified_split` grouped on
        ``image_id``); an image carrying a frozen ``test_holdout`` item goes
        to ``test`` with all its objects. ``split_mode='keep_imported'`` (the
        default) keeps the split a dataset import filed each frame under
        (``dataset_split``); ``'recompute'`` ignores it. A frame an import
        marked a reviewed negative is written as an empty label file when
        ``include_negative_frames`` and its ``negative_for`` covers every
        class this export has objects for (see
        :mod:`~src.services.curation.export_negatives`). The manifest records
        ``split_counts`` (images per split), ``split_object_counts``
        (objects per split) and ``class_split_counts`` (objects per class
        per split).

        ``require_fully_labeled_images`` drops every image that also
        carries an unlabeled object (see
        :mod:`~src.services.curation.export_images`). Off by default: such
        an image is exported with its validated objects labeled and counted
        in ``unlabeled_items_on_exported_images`` /
        ``images_with_unlabeled_items``.

        ``item_filter`` (the shared item filter) narrows the validated items that
        are exported; the manifest records it.

        ``dedup_threshold`` (if given) collapses whole-frame near-duplicate
        images (cosine >= threshold on the images index's secondary
        embedding) to one representative image, which keeps all its
        objects, via
        :func:`~src.services.detection.frame_dedup.dedup_rows_by_embedding`.

        ``max_images`` caps the number of images via
        :func:`~src.services.curation.export_support.even_stratified_sample`
        — an even round-robin over each image's rarest class, applied after
        the partial-frame policy and dedup, so a rare class can't be
        squeezed out and the final count is exactly
        ``min(max_images, pool)``. The manifest records ``sampling_mode``
        (``'all'`` / ``'stratified_even'``).

        ``resize_mode='aspect'`` (default ``None`` = copy as-is) shrinks the
        longest side to ``image_size`` without padding, so the normalized
        labels stay valid. A letterbox pad would shift every box relative
        to the written labels, so it is refused.
        """
        if resize_mode not in (None, 'aspect'):
            msg = (
                f"resize_mode must be None or 'aspect' (labels are frame-relative): {resize_mode!r}"
            )
            raise ValueError(msg)
        started_at = datetime.now(UTC).isoformat()
        generation = await items_index_generation(self.opensearch, self.config.items_index)
        rows, skipped_items = await self._load_rows(validated_export_query(item_filter))
        registry_file = self.registry.load()
        rows, id_map, dropped_unregistered = to_dense_ids(rows, registry_file)

        images, unlabeled, selection = await select_export_images(
            self.opensearch,
            self.config,
            rows,
            require_fully_labeled_images=require_fully_labeled_images,
            dedup_threshold=dedup_threshold,
            max_images=max_images,
            seed=seed,
        )
        names = dense_class_names(registry_file, id_map)

        resolved_export_dir = (
            Path(export_dir)
            if export_dir
            else allocate_export_dir(self.config.export_root, datetime.now(UTC))
        )
        images_root = resolved_export_dir / 'images'
        labels_root = resolved_export_dir / 'labels'
        for split in ('train', 'val', 'test'):
            (images_root / split).mkdir(parents=True, exist_ok=True)
            (labels_root / split).mkdir(parents=True, exist_ok=True)

        # Grouped on image_id, so all of an image's objects share one split.
        split_rows = [obj for image in images for obj in image.objects]
        item_split, split_stats = stratified_split_with_stats(
            split_rows,
            seed=seed,
            train_ratio=self.profile.train_ratio,
            val_ratio=self.profile.val_ratio,
            group_key=DEFAULT_SPLIT_GROUP_KEY,
            pinned=pins_from_rows(split_rows, split_mode),
        )

        written = write_image_labels(
            images,
            item_split,
            names=names,
            labels_root=labels_root,
            images_root=images_root,
            config=self.config,
            copy_images=copy_images,
        )
        negatives_written = 0
        negatives_skipped_partial = 0
        if include_negative_frames:
            negatives_written, negatives_skipped_partial = await self._write_negatives(
                images,
                names,
                written,
                labels_root=labels_root,
                images_root=images_root,
                split_mode=split_mode,
                seed=seed,
                copy_images=copy_images,
            )

        image_copy_stats = await copy_export_images(
            written.image_jobs,
            resize_mode=resize_mode,
            image_size=image_size,
            max_workers=max_image_workers,
        )

        # Computed AFTER every label file is on disk — hashes what was
        # actually written (frames, splits, boxes), plus the ordered class
        # names, not just which item ids were selected. See
        # export_support.label_content_sha.
        checksum = label_content_sha(resolved_export_dir, names, truncate=None)
        frozen_test_sha = await asyncio.to_thread(frozen_test_sha_of, resolved_export_dir)
        test_label_sha = await asyncio.to_thread(
            label_content_sha, resolved_export_dir, None, truncate=16, split='test'
        )
        finished_at = datetime.now(UTC).isoformat()

        data_yaml_path = write_data_yaml(resolved_export_dir, names)
        classes_with_objects = write_label_stats(resolved_export_dir, names, written.per_class)
        write_registry_snapshot(resolved_export_dir, registry_file, id_map, exported_at=finished_at)

        exported_unlabeled = [unlabeled.get(im.image_id, 0) for im in images]
        partial = {
            'require_fully_labeled_images': require_fully_labeled_images,
            'unlabeled_items_on_exported_images': sum(exported_unlabeled),
            'images_with_unlabeled_items': sum(1 for n in exported_unlabeled if n),
            'images_dropped_not_fully_labeled': selection['images_dropped_not_fully_labeled'],
        }
        manifest = {
            'version_tag': version_tag,
            'seed': seed,
            'group_key': DEFAULT_SPLIT_GROUP_KEY,
            'dataset_sha': checksum,
            'frozen_test_sha': frozen_test_sha,
            'test_label_sha': test_label_sha,
            'image_count': len(images) + negatives_written,
            'negative_images': negatives_written,
            'negative_frames_skipped_partial': negatives_skipped_partial,
            **split_stats.to_manifest(split_mode),
            **source_frozen_manifest(written.test_stems),
            'object_count': len(written.item_ids),
            'split_counts': written.counts.to_dict(),
            'split_object_counts': written.object_counts.to_dict(),
            'class_split_counts': class_split_rows(written.per_class, id_map, names),
            'class_count': len(names),
            'classes_with_objects': classes_with_objects,
            **partial,
            'started_at': started_at,
            'finished_at': finished_at,
            'exported_at': finished_at,
            'code_sha': _code_sha(),
            'frozen_holdout_sha': compute_holdout_sha(written.holdout_item_ids)
            if written.holdout_item_ids
            else None,
            'dedup': selection['dedup'],
            'max_images': max_images,
            'item_filter': item_filter.model_dump(exclude_defaults=True) if item_filter else None,
            'sampling_mode': selection['sampling_mode'],
            'image_copy': image_copy_stats,
            'resize_mode': resize_mode,
            'dropped_unregistered_class_ids': dropped_unregistered,
            'skipped_items': skipped_items,
            # The items index this dataset was read from (staleness check
            # at training preflight — see export_readiness).
            MANIFEST_GENERATION_KEY: generation,
        }
        manifest_path = resolved_export_dir / ARTIFACT_FILENAMES['manifest']
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2))

        current_symlink = self.config.export_root / 'current'
        atomic_symlink_flip(current_symlink, resolved_export_dir)
        await self._prune_old_exports()

        return ExportResult(
            export_dir=str(resolved_export_dir),
            version_tag=version_tag,
            manifest_path=str(manifest_path),
            data_yaml_path=str(data_yaml_path),
            dataset_sha=checksum,
            split_counts=written.counts,
            image_count=len(images) + negatives_written,
            class_count=len(names),
            classes_with_objects=classes_with_objects,
            started_at=started_at,
            finished_at=finished_at,
            current_symlink=str(current_symlink),
            object_count=len(written.item_ids),
            split_object_counts=written.object_counts,
            unlabeled_items_on_exported_images=partial['unlabeled_items_on_exported_images'],
            images_with_unlabeled_items=partial['images_with_unlabeled_items'],
            images_dropped_not_fully_labeled=partial['images_dropped_not_fully_labeled'],
            skipped_items=skipped_items,
        )

    async def _load_rows(self, query: dict[str, Any]) -> tuple[list[_ExportRow], dict[str, int]]:
        """Scroll the cohort and turn it into object rows; refuses an empty export."""
        # Scroll the FULL cohort — never cap here. Truncating the raw hit
        # list would hand the budget to whatever the scroll returned first
        # and let a rare class vanish; the cap is a class-balanced sample
        # of images applied once the pool has settled.
        hits = await self._scroll_items(query)
        if not hits:
            raise NothingToExportError(NO_VALIDATED_ITEMS_REASON)

        rows, skipped_items = hits_to_rows(hits)
        if not rows:
            raise NothingToExportError(
                f'{len(hits)} validated items, but none has an image id, a box and a class id'
            )
        return rows, skipped_items

    async def _write_negatives(
        self,
        images: list[_ExportImage],
        names: list[str],
        written: WrittenLabels,
        *,
        labels_root: Path,
        images_root: Path,
        split_mode: SplitMode,
        seed: int,
        copy_images: bool,
    ) -> tuple[int, int]:
        """Reviewed-negative frames as empty labels; returns ``(written, skipped_partial)``."""
        planned, skipped_partial = plan_multi_class_negatives(
            await scroll_negative_frames(self.opensearch, images_index=self.config.images_index),
            exported_class_names={names[o.export_class_id] for im in images for o in im.objects},
            already_exported={im.image_id for im in images},
            split_mode=split_mode,
            seed=seed,
            train_ratio=self.profile.train_ratio,
            val_ratio=self.profile.val_ratio,
        )
        negatives_written = write_negative_frames(
            planned,
            labels_root=labels_root,
            images_root=images_root,
            config=self.config,
            split_counts=written.counts,
            test_stems=written.test_stems,
            image_jobs=written.image_jobs,
            copy_images=copy_images,
        )
        return negatives_written, skipped_partial

    async def _prune_old_exports(self) -> None:
        # ST-4: best-effort keep-last prune of auto-named export dirs.
        # Never raises -- a prune failure must never fail an otherwise
        # successful export.
        try:
            await asyncio.to_thread(prune_exports_after_write, self.config)
        except Exception as exc:
            logger.warning('export_retention_failed', error=str(exc))
