"""Generic YOLO-format dataset export service.

The reference implementation this was ported from
(see ``docs/design/curation_design_rationale.md`` for the genericization
approach) splits into two halves: artifact filenames, class
lists and split ratios are deployment data (an ``ExportProfile``, extracted
here), while the YOLO-format writer, split logic and manifest/checksum
mechanism are generic algorithm code that stays code. The bespoke
whole-frame-vs-crop / near-dup-collapsing features of the reference
exporter (a 1143-LOC domain-specific service, never ported) are
intentionally NOT reproduced here; a deployment-specific overlay can
extend :class:`GenericYoloExportService` directly if it needs them (plan
§7 R5 — the generic curation stack ships with a thinner export path than
the reference by design, tracked as the most likely first follow-up after
merge).

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
are on :func:`~src.services.curation.export_support.stratified_split`.

Dense export ids (``class_registry.json:export_id_map``) are resolved from
the live :class:`~src.clients.curation_opensearch.ClassRegistry` at export
time and frozen into the export directory, so a subset-training request's
``include_classes`` (always expressed in REGISTRY ids) can be translated to
the dense ids that actually appear in the written label files — see
``src/services/training/preflight_scan.py`` and
``src/routers/curation_train.py``, both of which read this file back.
"""

from __future__ import annotations

import asyncio
import json
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from src.clients.curation_opensearch import ClassRegistry, get_class_registry
from src.config import CurationConfig, get_curation_config
from src.core.logging import get_logger
from src.services.curation.export_images import (
    _ExportImage,
    group_rows_by_image,
    image_file_stem,
    normalized_box,
    rarest_class_key,
    unlabeled_items_by_image,
    yolo_line,
)
from src.services.curation.export_readiness import (
    MANIFEST_GENERATION_KEY,
    NothingToExportError,
    items_index_generation,
)
from src.services.curation.export_support import (
    DEFAULT_SPLIT_GROUP_KEY,
    _build_export_id_map,
    _code_sha,
    _copy_or_resize_one,
    _ExportRow,
    _remap_rows_to_export_ids,
    _resolve_source_path,
    atomic_symlink_flip,
    atomic_write_text,
    even_stratified_sample,
    frozen_test_sha_of,
    hash_split,
    label_content_sha,
    scroll_hits,
    stratified_split,
)
from src.services.curation.holdout import compute_holdout_sha
from src.services.detection.frame_dedup import dedup_rows_by_embedding


logger = get_logger(__name__)

# Artifact filenames served read-only by `GET /curation/export/registry/{artifact}`.
# Fixed, whitelisted set — never derived from user input.
ARTIFACT_FILENAMES: dict[str, str] = {
    'manifest': 'manifest.json',
    'data_yaml': 'data.yaml',
    'class_registry': 'class_registry.json',
    'label_stats': 'label_stats.json',
}

REGISTRY_ARTIFACT_CONTENT_TYPES: dict[str, str] = {
    'class_registry.json': 'application/json',
    'data.yaml': 'application/x-yaml',
    'manifest.json': 'application/json',
    'label_stats.json': 'application/json',
}


@dataclass(frozen=True)
class ExportProfile:
    """Deployment-tunable export knobs (plan §3.5 extraction)."""

    name: str = 'default'
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    # test_ratio is whatever remains: 1 - train_ratio - val_ratio.


@dataclass
class SplitCounts:
    train: int = 0
    val: int = 0
    test: int = 0

    def to_dict(self) -> dict[str, int]:
        return {'train': self.train, 'val': self.val, 'test': self.test}


@dataclass
class ExportResult:
    """What one export wrote. ``split_counts`` / ``image_count`` count
    images; ``split_object_counts`` / ``object_count`` count label lines."""

    export_dir: str
    version_tag: str
    manifest_path: str
    data_yaml_path: str
    dataset_sha: str
    split_counts: SplitCounts
    image_count: int
    class_count: int
    classes_with_objects: int
    started_at: str
    finished_at: str
    current_symlink: str
    object_count: int = 0
    split_object_counts: SplitCounts = field(default_factory=SplitCounts)
    unlabeled_items_on_exported_images: int = 0
    images_with_unlabeled_items: int = 0
    images_dropped_not_fully_labeled: int = 0
    skipped_items: dict[str, int] = field(default_factory=dict)


def _class_split_rows(
    per_class: dict[int, SplitCounts], id_map: dict[int, int], names: list[str]
) -> list[dict[str, Any]]:
    """Per-class object counts per split, one row per dense export id.

    Every class in the export's vocabulary gets a row, including one with
    no instances at all — training preflight reads these rows to block a
    class the run would train on but that has nothing in train or val.
    """
    registry_id_of = {dense_id: registry_id for registry_id, dense_id in id_map.items()}
    return [
        {
            'class_id': registry_id_of[dense_id],
            'export_id': dense_id,
            'class_name': names[dense_id],
            **per_class[dense_id].to_dict(),
        }
        for dense_id in range(len(names))
    ]


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
            ],
        )

    def _hits_to_rows(self, hits: list[dict[str, Any]]) -> tuple[list[_ExportRow], dict[str, int]]:
        """Object rows for every hit with an image, a usable box and a class.

        Returns ``(rows, skipped)``; ``skipped`` counts the hits left out
        by reason. A skipped hit that sits on an exported image is still
        counted there as an unlabeled object.
        """
        rows: list[_ExportRow] = []
        skipped = {'no_image_id': 0, 'no_usable_box_or_class': 0}
        for hit in hits:
            src = hit.get('_source') or {}
            item_id = src.get('crop_id') or hit.get('_id')
            box = normalized_box(src.get('bbox_norm'))
            class_id = src.get('class_id')
            if not item_id or box is None or class_id is None:
                skipped['no_usable_box_or_class'] += 1
                continue
            if not src.get('image_id'):
                # No image key to group on or to name files by.
                skipped['no_image_id'] += 1
                continue
            rows.append(
                _ExportRow(
                    item_id=str(item_id),
                    image_id=str(src['image_id']),
                    image_path=str(src.get('image_path') or ''),
                    bbox_norm=list(box),
                    class_id=int(class_id),
                    class_name=str(src.get('class_name') or class_id),
                    has_test_crop=bool(src.get('test_holdout')),
                )
            )
        return rows, skipped

    async def _apply_dedup(
        self, images: list[_ExportImage], dedup_threshold: float | None
    ) -> tuple[list[_ExportImage], dict[str, Any]]:
        """Collapse near-duplicate images; the stats' ``*_rows`` count images."""
        if dedup_threshold is None:
            return images, {'enabled': False}
        kept, stats = await dedup_rows_by_embedding(
            self.opensearch, images, threshold=dedup_threshold, config=self.config
        )
        return sorted(kept, key=lambda im: im.image_id), stats

    async def _copy_images(
        self,
        jobs: list[tuple[str, str]],
        *,
        resize_mode: Literal['aspect'] | None,
        image_size: int,
        max_workers: int,
    ) -> dict[str, Any]:
        if not jobs:
            return {'attempted': 0, 'copied': 0, 'failed': 0, 'errors': []}
        loop = asyncio.get_running_loop()
        copied = 0
        failed = 0
        errors: list[str] = []
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                loop.run_in_executor(pool, _copy_or_resize_one, src, dest, resize_mode, image_size)
                for src, dest in jobs
            ]
            for fut in asyncio.as_completed(futures):
                _dest, ok, err = await fut
                if ok:
                    copied += 1
                else:
                    failed += 1
                    if err:
                        errors.append(err)
        return {'attempted': len(jobs), 'copied': copied, 'failed': failed, 'errors': errors[:20]}

    async def _select_images(
        self,
        rows: list[_ExportRow],
        *,
        require_fully_labeled_images: bool,
        dedup_threshold: float | None,
        max_images: int | None,
        seed: int,
    ) -> tuple[list[_ExportImage], dict[str, int], dict[str, Any]]:
        """Group dense-id rows into images, then apply the partial-frame
        policy, the near-dup collapse and the ``max_images`` cap, in that
        order — so a dropped partial frame can't knock out a fully labeled
        near-duplicate, and the cap lands on exactly
        ``min(max_images, pool)`` images.

        Returns ``(images, unlabeled_by_image, info)``.
        """
        images = group_rows_by_image(rows)
        unlabeled = await unlabeled_items_by_image(
            self.opensearch,
            index=self.config.items_index,
            image_ids=[im.image_id for im in images],
            labeled_item_ids={r.item_id for r in rows},
        )
        dropped_partial = 0
        if require_fully_labeled_images:
            full = [im for im in images if not unlabeled.get(im.image_id)]
            dropped_partial = len(images) - len(full)
            if not full:
                raise NothingToExportError(
                    f'require_fully_labeled_images: all {len(images)} image(s) with a validated '
                    'object also carry an unreviewed or unexported object; none is fully labeled'
                )
            images = full
        images, dedup_stats = await self._apply_dedup(images, dedup_threshold)

        sampling_mode = 'all'
        if max_images is not None and len(images) > max_images:
            frequency: dict[int, int] = {}
            for image in images:
                for obj in image.objects:
                    frequency[obj.export_class_id] = frequency.get(obj.export_class_id, 0) + 1
            images = even_stratified_sample(
                images, max_images, lambda im: rarest_class_key(im, frequency), random.Random(seed)
            )
            images.sort(key=lambda im: im.image_id)
            sampling_mode = 'stratified_even'
            logger.info('export_sampled', n_images=len(images), max_images=max_images)
        info = {
            'dedup': dedup_stats,
            'sampling_mode': sampling_mode,
            'images_dropped_not_fully_labeled': dropped_partial,
        }
        return images, unlabeled, info

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
        to ``test`` with all its objects. The manifest records
        ``split_counts`` (images per split), ``split_object_counts``
        (objects per split) and ``class_split_counts`` (objects per class
        per split).

        ``require_fully_labeled_images`` drops every image that also
        carries an unlabeled object (see
        :mod:`~src.services.curation.export_images`). Off by default: such
        an image is exported with its validated objects labeled and counted
        in ``unlabeled_items_on_exported_images`` /
        ``images_with_unlabeled_items``.

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
        query = {
            'bool': {
                'filter': [{'term': {'class_validated': True}}],
                'must_not': [
                    {'exists': {'field': 'review_dismissed_at'}},
                    {'term': {'class_excluded': True}},
                ],
            }
        }
        # Scroll the FULL cohort — never cap here. Truncating the raw hit
        # list would hand the budget to whatever the scroll returned first
        # and let a rare class vanish; the cap is a class-balanced sample
        # of images applied once the pool has settled.
        hits = await self._scroll_items(query)
        if not hits:
            raise NothingToExportError(
                '0 items are class_validated (and not excluded or review-dismissed); '
                'validate labels before exporting'
            )

        rows, skipped_items = self._hits_to_rows(hits)
        if not rows:
            raise NothingToExportError(
                f'{len(hits)} validated items, but none has an image id, a box and a class id'
            )

        registry_file = self.registry.load()
        id_map = _build_export_id_map(registry_file.classes)
        dropped_unregistered: dict[str, int] = {}
        for row in rows:
            if row.class_id not in id_map:
                key = str(row.class_id)
                dropped_unregistered[key] = dropped_unregistered.get(key, 0) + 1
        rows = _remap_rows_to_export_ids(rows, id_map)
        if not rows:
            raise NothingToExportError(
                'every validated item has a class id that is not in the class registry '
                f'(or is deprecated): {dict(sorted(dropped_unregistered.items()))}'
            )

        images, unlabeled, selection = await self._select_images(
            rows,
            require_fully_labeled_images=require_fully_labeled_images,
            dedup_threshold=dedup_threshold,
            max_images=max_images,
            seed=seed,
        )

        name_by_registry_id = {c.class_id: c.class_name for c in registry_file.classes}
        names: list[str] = [''] * len(id_map)
        for registry_id, dense_id in id_map.items():
            names[dense_id] = name_by_registry_id.get(registry_id, str(registry_id))

        resolved_export_dir = (
            Path(export_dir)
            if export_dir
            else (self.config.export_root / datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ'))
        )
        images_root = resolved_export_dir / 'images'
        labels_root = resolved_export_dir / 'labels'
        for split in ('train', 'val', 'test'):
            (images_root / split).mkdir(parents=True, exist_ok=True)
            (labels_root / split).mkdir(parents=True, exist_ok=True)

        # Grouped on image_id, so all of an image's objects share one split.
        item_split = stratified_split(
            [obj for image in images for obj in image.objects],
            seed=seed,
            train_ratio=self.profile.train_ratio,
            val_ratio=self.profile.val_ratio,
            group_key=DEFAULT_SPLIT_GROUP_KEY,
        )

        counts = SplitCounts()
        object_counts = SplitCounts()
        per_class = {dense_id: SplitCounts() for dense_id in range(len(names))}
        item_ids: list[str] = []
        holdout_item_ids: list[str] = []
        image_jobs: list[tuple[str, str]] = []

        for image in images:
            split = item_split[image.objects[0].item_id]
            stem = image_file_stem(image.image_id)
            lines = [yolo_line(o.export_class_id, o.bbox_norm) for o in image.objects]
            (labels_root / split / f'{stem}.txt').write_text('\n'.join(lines) + '\n')
            setattr(counts, split, getattr(counts, split) + 1)
            setattr(object_counts, split, getattr(object_counts, split) + len(image.objects))
            for obj in image.objects:
                class_counts = per_class[obj.export_class_id]
                setattr(class_counts, split, getattr(class_counts, split) + 1)
                item_ids.append(obj.item_id)
                if obj.has_test_crop:
                    holdout_item_ids.append(obj.item_id)

            if copy_images:
                src_path = _resolve_source_path(image.image_path, self.config)
                if src_path is not None:
                    ext = src_path.suffix or '.jpg'
                    dest = images_root / split / f'{stem}{ext}'
                    image_jobs.append((str(src_path), str(dest)))
                else:
                    logger.warning(
                        'export_source_image_unresolved',
                        image_id=image.image_id,
                        image_path=image.image_path,
                    )

        image_copy_stats = await self._copy_images(
            image_jobs,
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

        data_yaml_path = resolved_export_dir / ARTIFACT_FILENAMES['data_yaml']
        data_yaml_path.write_text(
            f'path: {resolved_export_dir}\n'
            'train: images/train\n'
            'val: images/val\n'
            'test: images/test\n'
            f'nc: {len(names)}\n'
            f'names: {json.dumps(names)}\n'
        )

        # Objects per class, i.e. label lines across every split.
        label_stats: dict[str, int] = dict.fromkeys(names, 0)
        for dense_id, class_counts in per_class.items():
            label_stats[names[dense_id]] += sum(class_counts.to_dict().values())
        (resolved_export_dir / ARTIFACT_FILENAMES['label_stats']).write_text(
            json.dumps(label_stats, indent=2)
        )
        # E2: class_count is the registry size written into data.yaml
        # (nc/names) -- it stays that way for back-compat, but a class
        # with zero labeled objects in this export inflates that number
        # into looking like "84 classes of real data" when only a
        # handful have any objects at all. classes_with_objects is the
        # honest denominator for a per-class table.
        classes_with_objects = sum(1 for v in label_stats.values() if v > 0)

        class_registry_payload = {
            'version': 1,
            'exported_at': finished_at,
            'classes': [
                {
                    'class_id': c.class_id,
                    'class_name': c.class_name,
                    'deprecated': c.deprecated,
                }
                for c in registry_file.classes
            ],
            'export_id_map': {str(k): v for k, v in id_map.items()},
        }
        class_registry_path = resolved_export_dir / ARTIFACT_FILENAMES['class_registry']
        class_registry_path.write_text(json.dumps(class_registry_payload, indent=2))

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
            'image_count': len(images),
            'object_count': len(item_ids),
            'split_counts': counts.to_dict(),
            'split_object_counts': object_counts.to_dict(),
            'class_split_counts': _class_split_rows(per_class, id_map, names),
            'class_count': len(names),
            'classes_with_objects': classes_with_objects,
            **partial,
            'started_at': started_at,
            'finished_at': finished_at,
            'exported_at': finished_at,
            'code_sha': _code_sha(),
            'frozen_holdout_sha': compute_holdout_sha(holdout_item_ids)
            if holdout_item_ids
            else None,
            'dedup': selection['dedup'],
            'max_images': max_images,
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

        return ExportResult(
            export_dir=str(resolved_export_dir),
            version_tag=version_tag,
            manifest_path=str(manifest_path),
            data_yaml_path=str(data_yaml_path),
            dataset_sha=checksum,
            split_counts=counts,
            image_count=len(images),
            class_count=len(names),
            classes_with_objects=classes_with_objects,
            started_at=started_at,
            finished_at=finished_at,
            current_symlink=str(current_symlink),
            object_count=len(item_ids),
            split_object_counts=object_counts,
            unlabeled_items_on_exported_images=partial['unlabeled_items_on_exported_images'],
            images_with_unlabeled_items=partial['images_with_unlabeled_items'],
            images_dropped_not_fully_labeled=partial['images_dropped_not_fully_labeled'],
            skipped_items=skipped_items,
        )


__all__ = [
    'ARTIFACT_FILENAMES',
    'REGISTRY_ARTIFACT_CONTENT_TYPES',
    'ExportProfile',
    'ExportResult',
    'GenericYoloExportService',
    'SplitCounts',
    'atomic_symlink_flip',
    'atomic_write_text',
    'even_stratified_sample',
    'frozen_test_sha_of',
    'hash_split',
    'label_content_sha',
    'resolve_current_export_dir',
    'stratified_split',
]
