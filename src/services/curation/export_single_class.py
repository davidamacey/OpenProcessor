"""Single-class / class-subset YOLO dataset export with an integrity envelope.

The counterpart to :mod:`src.services.curation.export`'s
:class:`~src.services.curation.export.GenericYoloExportService`, which
exports the *whole* registry as a multi-class detection dataset. This
module answers the narrower question a focused detector-training cycle
asks: **"give me a dataset for these specific class(es), and prove to me
afterwards exactly what was in it."**

Two things make it a distinct capability rather than a flag on the
multi-class exporter:

1. **A narrowed vocabulary needs real backgrounds.** A detector trained
   on one class out of eighty must learn to *not* fire on the other
   seventy-nine. So alongside positives this exporter deliberately emits
   label-free background frames — and prioritises the ones where a
   detector already fired and a human said "no" (``false_positive``
   regions), which are the hard negatives worth the most per image.
   Generic empty frames are added only as a small
   ``empty_bg_ratio``-scaled sample so they can't swamp the positives.
2. **A stronger integrity envelope.** ``dataset_sha`` hashes the actual
   written label *content* (not just the set of item ids), and
   ``frozen_test_sha`` hashes the test split's on-disk identity, so
   "the test set never changed between these two runs" is a checkable
   claim rather than a convention. The current-export symlink is flipped
   atomically, so a trainer resolving it mid-export never sees a
   half-written dataset.

Everything else is shared with the multi-class exporter rather than
reimplemented: the class-balanced cap
(:func:`~src.services.curation.export_support.even_stratified_sample`),
the group-aware deterministic splitter
(:func:`~src.services.curation.export_split.stratified_split`), the
near-duplicate frame collapse
(:func:`~src.services.detection.frame_dedup.dedup_rows_by_embedding`),
the resize worker and the atomic-write primitives.

**Nothing here is domain-specific.** Which classes to export, what the
region vocabulary is called, where the output lands — all of it comes
from :class:`SingleClassExportProfile` and the call's arguments. Region
document fields are read through :class:`~src.config.region_fields.RegionFields`
and region states through :class:`~src.config.region_state.RegionStatus`,
so a deployment whose index uses different field names configures them
rather than forking this file.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.clients.curation_opensearch.registry import ClassRegistry, get_class_registry
from src.config import CurationConfig, get_curation_config
from src.config.region_fields import RegionFields, get_region_fields
from src.core.logging import get_logger
from src.services.curation.export_models import (
    DATA_YAML_FILENAME,
    MANIFEST_FILENAME,
    STRATUM_MAP_FILENAME,
    SingleClassExportProfile,
    SingleClassExportResult,
    SplitCounts,
)
from src.services.curation.export_readiness import (
    MANIFEST_GENERATION_KEY,
    NothingToExportError,
    items_index_generation,
)
from src.services.curation.export_single_class_rows import ImageMode, RowCollector, _FrameRow
from src.services.curation.export_split import (
    SplitStats,
    pins_from_rows,
    stratified_split_with_stats,
)
from src.services.curation.export_support import (
    _code_sha,
    _resolve_source_path,
    allocate_export_dir,
    atomic_symlink_flip,
    atomic_write_text,
    even_stratified_sample,
    frozen_test_sha_of,
    label_content_sha,
    source_frozen_test_sha,
)
from src.services.curation.export_writers import (
    copy_single_class_images,
    render_data_yaml,
    render_label,
    write_class_registry,
    write_label_stats_from_disk,
)
from src.services.detection.frame_dedup import dedup_rows_by_embedding


logger = get_logger(__name__)


class SingleClassExportService:
    """Builds a narrowed-vocabulary YOLO dataset with an integrity envelope."""

    def __init__(
        self,
        opensearch: Any,
        *,
        profile: SingleClassExportProfile,
        config: CurationConfig | None = None,
        registry: ClassRegistry | None = None,
        region_fields: RegionFields | None = None,
        max_image_workers: int = 4,
    ) -> None:
        self.opensearch = opensearch
        self.profile = profile
        self.config = config or get_curation_config()
        self.registry = registry or get_class_registry()
        self.max_image_workers = max_image_workers
        self.collector = RowCollector(
            opensearch,
            profile=profile,
            config=self.config,
            region_fields=region_fields or get_region_fields(),
            class_names=self._resolve_class_names(),
        )

    # ----------------------------------------------------------------- public

    async def export(
        self,
        *,
        export_dir: Path | None = None,
        version_tag: str = '',
        seed: int = 42,
        skip_test_split: bool = False,
        empty_bg_ratio: float = 0.1,
        max_positive_images: int | None = None,
        dedup_threshold: float | None = None,
        image_mode: ImageMode = 'whole_frame',
        img_max_side: int = 1280,
        copy_images: bool = True,
    ) -> SingleClassExportResult:
        """Materialize the dataset and return its integrity envelope.

        Args:
            export_dir: Staging-dir override. ``None`` = a timestamped dir
                under the profile's output root.
            version_tag: Free-form tag recorded in the manifest.
            seed: RNG seed for both the cap sample and the split, so
                recording it makes the export re-derivable.
            skip_test_split: Emit only train/val, for quick iteration.
                Waives the frozen test-holdout pin, so ``frozen_test_sha``
                comes back empty — an export made this way is explicitly
                not comparable against one that has a test split.
            empty_bg_ratio: Label-free frames to add as a fraction of
                positives (``0.1`` = one per ten). Hard negatives are kept
                in full regardless and are not counted against this.
            max_positive_images: Cap on positive frames, applied as an
                even round-robin over strata so a rare stratum survives
                the cap intact. ``None`` = every positive.
            dedup_threshold: Cosine cut for collapsing near-duplicate
                source frames before the split, so a burst of near-identical
                frames neither inflates the dataset nor leaks across splits.
                ``None`` disables.
            image_mode: ``'whole_frame'`` writes the full source frame —
                the distribution a deployed detector actually sees.
                ``'item_crop'`` writes the parent item's crop with the
                region re-projected into crop coordinates, for the second
                stage of a detect-then-crop pipeline. Only meaningful when
                the profile's ``box_source`` is ``'region'``.
            img_max_side: Longest output side in px, aspect preserved and
                never padded — the normalized labels stay valid because a
                letterbox offset would silently shift every box.
            copy_images: Write image pixels. ``False`` writes labels and
                artifacts only (fast dry run).
        """
        self._validate(image_mode)
        started = datetime.now(UTC)

        generation = await items_index_generation(self.opensearch, self.config.items_index)
        names = self._resolve_class_names()
        rows = await self.collector.collect(image_mode=image_mode, empty_bg_ratio=empty_bg_ratio)
        if not rows:
            raise NothingToExportError(
                f'no item matches class ids {list(self.profile.class_ids)} '
                f'with box_source={self.profile.box_source!r}; nothing to export'
            )
        rng = random.Random(seed)  # nosec B311 - reproducible sampling, not cryptography
        rows = self._select(
            rows,
            rng=rng,
            empty_bg_ratio=empty_bg_ratio,
            max_positive_images=max_positive_images,
        )
        sampling_mode = 'stratified_even' if max_positive_images is not None else 'all'

        dedup_stats: dict[str, Any] = {'enabled': False}
        if dedup_threshold is not None:
            rows, dedup_stats = await dedup_rows_by_embedding(
                self.opensearch, rows, threshold=dedup_threshold, config=self.config
            )

        n_pos = sum(1 for r in rows if r.is_positive)
        n_bg = len(rows) - n_pos
        n_hard_neg = sum(1 for r in rows if not r.is_positive and r.is_hard_negative)
        if n_pos == 0:
            # A background-only dataset cannot train a detector. Surface it
            # loudly here AND in the manifest, so an operator notices before
            # committing GPU hours to a doomed run.
            logger.warning(
                'single_class_export_zero_positives',
                background_images=n_bg,
                hard_negative_images=n_hard_neg,
            )

        resolved_dir = (
            Path(export_dir) if export_dir else allocate_export_dir(self._output_root(), started)
        )
        for split in ('train', 'val', 'test'):
            (resolved_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (resolved_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

        splits, split_stats = self._assign_splits(rows, seed=seed, skip_test_split=skip_test_split)
        counts, stratum_distribution, image_jobs = self._write_labels(
            rows, splits, resolved_dir, copy_images=copy_images
        )
        image_copy_stats = await copy_single_class_images(
            image_jobs,
            max_workers=self.max_image_workers,
            img_max_side=img_max_side,
            copy_images=copy_images,
        )

        atomic_write_text(
            resolved_dir / STRATUM_MAP_FILENAME,
            json.dumps({r.item_id: r.stratum for r in rows}, indent=2, sort_keys=True),
        )
        data_yaml_path = resolved_dir / DATA_YAML_FILENAME
        data_yaml_path.write_text(render_data_yaml(resolved_dir, names))
        write_class_registry(resolved_dir, names, profile=self.profile, registry=self.registry)
        write_label_stats_from_disk(resolved_dir, names)

        dataset_sha = await asyncio.to_thread(label_content_sha, resolved_dir, names)
        frozen_test_sha = await asyncio.to_thread(frozen_test_sha_of, resolved_dir)
        test_label_sha = await asyncio.to_thread(
            label_content_sha, resolved_dir, None, truncate=16, split='test'
        )
        finished = datetime.now(UTC)

        manifest = {
            'dataset_kind': self.profile.dataset_kind,
            'version_tag': version_tag,
            'started_at': started.isoformat(),
            'finished_at': finished.isoformat(),
            'exported_at': finished.isoformat(),
            'class_count': len(names),
            'class_name': names[0] if len(names) == 1 else '',
            'class_names': names,
            'class_ids': list(self.profile.class_ids),
            'box_source': self.profile.box_source,
            'image_mode': image_mode,
            'img_max_side': img_max_side,
            'empty_bg_ratio': empty_bg_ratio,
            'max_positive_images': max_positive_images,
            'sampling_mode': sampling_mode,
            'seed': seed,
            'group_key': 'image_id',
            'dedup': dedup_stats,
            'positive_images': n_pos,
            'background_images': n_bg,
            'false_positive_background_images': n_hard_neg,
            'positives_zero_warning': n_pos == 0,
            'dataset_sha': dataset_sha,
            'frozen_test_sha': frozen_test_sha,
            'test_label_sha': test_label_sha,
            'code_sha': _code_sha(),
            'split_counts': counts.to_dict(),
            'stratum_distribution': dict(sorted(stratum_distribution.items())),
            'stratum_count': len(stratum_distribution),
            'image_count': counts.train + counts.val + counts.test,
            'image_copy': image_copy_stats,
            **split_stats.to_manifest(self.profile.split_mode),
            **self._source_frozen_manifest(rows, splits),
            MANIFEST_GENERATION_KEY: generation,
        }
        manifest_path = resolved_dir / MANIFEST_FILENAME
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))

        current_symlink = self._output_root() / self.profile.current_link_name
        atomic_symlink_flip(current_symlink, resolved_dir)

        logger.info(
            'single_class_export_done',
            export_dir=str(resolved_dir),
            positive_images=n_pos,
            background_images=n_bg,
            **counts.to_dict(),
        )
        return SingleClassExportResult(
            export_dir=str(resolved_dir),
            version_tag=version_tag,
            manifest_path=str(manifest_path),
            data_yaml_path=str(data_yaml_path),
            dataset_sha=dataset_sha,
            frozen_test_sha=frozen_test_sha,
            split_counts=counts,
            image_count=counts.train + counts.val + counts.test,
            class_count=len(names),
            positive_images=n_pos,
            background_images=n_bg,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            current_symlink=str(current_symlink),
        )

    # ------------------------------------------------------------- validation

    def _validate(self, image_mode: str) -> None:
        if image_mode not in ('whole_frame', 'item_crop'):
            msg = f'image_mode must be whole_frame|item_crop, got {image_mode!r}'
            raise ValueError(msg)
        if self.profile.box_source not in ('item', 'region'):
            msg = f'box_source must be item|region, got {self.profile.box_source!r}'
            raise ValueError(msg)
        if self.profile.box_source == 'item':
            if not self.profile.class_ids:
                msg = "box_source='item' requires a non-empty profile.class_ids"
                raise ValueError(msg)
            if image_mode == 'item_crop':
                # The item's own box IS the label, so cropping to it would
                # make every label the full image -- a degenerate dataset.
                msg = "image_mode='item_crop' is only valid with box_source='region'"
                raise ValueError(msg)

    def _output_root(self) -> Path:
        return self.profile.output_root or (self.config.export_root / self.profile.name)

    def _resolve_class_names(self) -> list[str]:
        """Dataset class names, ordered to match the dense label ids."""
        if self.profile.box_source == 'region':
            return [self.profile.region_class_name]
        by_id = {c.class_id: c.class_name for c in self.registry.load().classes}
        return [by_id.get(cid, str(cid)) for cid in self.profile.class_ids]

    # ---------------------------------------------------------------- select

    def _select(
        self,
        rows: list[_FrameRow],
        *,
        rng: random.Random,
        empty_bg_ratio: float,
        max_positive_images: int | None,
    ) -> list[_FrameRow]:
        """Cap positives evenly; keep every hard negative; sample the rest.

        The cap runs as an even round-robin over strata rather than a flat
        random cut, so a smaller dataset version still represents every
        labeled stratum instead of letting the biggest ones eat the budget.
        """
        positives = [r for r in rows if r.is_positive]
        backgrounds = [r for r in rows if not r.is_positive]
        if max_positive_images is not None and len(positives) > max_positive_images:
            positives = even_stratified_sample(
                positives, max_positive_images, lambda r: r.stratum, rng
            )
        hard_negatives = [r for r in backgrounds if r.is_hard_negative]
        empties = [r for r in backgrounds if not r.is_hard_negative]
        rng.shuffle(empties)
        n_empty = round(len(positives) * max(0.0, empty_bg_ratio))
        return positives + hard_negatives + empties[:n_empty]

    # ----------------------------------------------------------------- split

    def _source_frozen_manifest(
        self, rows: list[_FrameRow], splits: dict[str, str]
    ) -> dict[str, str]:
        """``source_frozen_test_sha`` when any test row came from an import
        (see :func:`~src.services.curation.export_support.source_frozen_test_sha`)."""
        test_rows = [r for r in rows if splits[r.item_id] == 'test']
        sha = source_frozen_test_sha(
            (r.import_source_stem for r in test_rows), (r.item_id for r in test_rows)
        )
        return {'source_frozen_test_sha': sha} if sha else {}

    def _assign_splits(
        self, rows: list[_FrameRow], *, seed: int, skip_test_split: bool
    ) -> tuple[dict[str, str], SplitStats]:
        """Group-aware stratified split over the shared splitter.

        Strata are this exporter's human-readable stratum strings, mapped
        to stable ordinals (sorted, so the mapping depends only on which
        strata exist and never on scroll order) because the shared splitter
        stratifies on an integer ``class_id``. Grouping on ``image_id``
        keeps every row of one source frame in the same split.
        """
        ordinals = {key: i for i, key in enumerate(sorted({r.stratum for r in rows}))}
        for row in rows:
            row.class_id = ordinals[row.stratum]
        if skip_test_split:
            # Absorb the test share into train and drop the holdout pin, so
            # the caller gets train/val only. frozen_test_sha comes back
            # empty for such a run -- see export()'s docstring.
            for row in rows:
                row.has_test_crop = False
            train_ratio, val_ratio = 0.89, 0.11
        else:
            train_ratio, val_ratio = self.profile.train_ratio, self.profile.val_ratio
        splits, stats = stratified_split_with_stats(
            rows,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            group_key='image_id',
            pinned=pins_from_rows(rows, self.profile.split_mode),
        )
        if skip_test_split:
            splits = {k: ('train' if v == 'test' else v) for k, v in splits.items()}
        return splits, stats

    # ----------------------------------------------------------------- write

    def _write_labels(
        self,
        rows: list[_FrameRow],
        splits: dict[str, str],
        export_dir: Path,
        *,
        copy_images: bool,
    ) -> tuple[SplitCounts, dict[str, dict[str, int]], list[tuple[str, str, Any]]]:
        counts = SplitCounts()
        distribution: dict[str, dict[str, int]] = defaultdict(
            lambda: {'train': 0, 'val': 0, 'test': 0}
        )
        image_jobs: list[tuple[str, str, Any]] = []
        for row in rows:
            split = splits[row.item_id]
            label_path = export_dir / 'labels' / split / f'{row.item_id}.txt'
            label_path.write_text(render_label(row.boxes), encoding='utf-8')
            setattr(counts, split, getattr(counts, split) + 1)
            distribution[row.stratum][split] += 1
            if not copy_images:
                continue
            src_path = _resolve_source_path(row.image_path, self.config)
            if src_path is None:
                logger.warning(
                    'single_class_export_source_unresolved',
                    item_id=row.item_id,
                    image_path=row.image_path,
                )
                continue
            dest = export_dir / 'images' / split / f'{row.item_id}.jpg'
            image_jobs.append((str(src_path), str(dest), row.crop_norm))
        return counts, dict(distribution), image_jobs


def resolve_current_single_class_dir(
    profile: SingleClassExportProfile,
    config: CurationConfig | None = None,
) -> Path:
    """Resolve a profile's current export dir via its own ``current`` symlink.

    Scoped to the profile's own output root — never the multi-class
    export root, so the two exports can never be confused for each other.

    Raises:
        FileNotFoundError: this profile has never produced an export.
    """
    cfg = config or get_curation_config()
    root = profile.output_root or (cfg.export_root / profile.name)
    current = root / profile.current_link_name
    if not (current.is_symlink() or current.exists()):
        msg = f'no current export symlink for profile {profile.name!r}'
        raise FileNotFoundError(msg)
    return current.resolve()
