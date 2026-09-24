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
(:func:`~src.services.curation.export_support.stratified_split`), the
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
import hashlib
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.clients.curation_opensearch import ClassRegistry, get_class_registry
from src.config import CurationConfig, get_curation_config
from src.config.region_fields import RegionFields, get_region_fields
from src.core.logging import get_logger
from src.services.curation.export_readiness import (
    MANIFEST_GENERATION_KEY,
    NothingToExportError,
    items_index_generation,
)
from src.services.curation.export_single_class_rows import ImageMode, RowCollector, _FrameRow
from src.services.curation.export_support import (
    _code_sha,
    _copy_or_resize_one,
    _resolve_source_path,
    atomic_symlink_flip,
    atomic_write_text,
    even_stratified_sample,
    label_content_sha,
    stratified_split,
)
from src.services.detection.frame_dedup import dedup_rows_by_embedding


if TYPE_CHECKING:
    from collections.abc import Iterable


logger = get_logger(__name__)

BoxSource = Literal['item', 'region']

# Artifacts written next to the dataset. Same filenames as the
# multi-class exporter, so `GET /curation/export/registry/{artifact}`'s
# whitelist and any consumer reading an export directory work unchanged.
MANIFEST_FILENAME = 'manifest.json'
DATA_YAML_FILENAME = 'data.yaml'
CLASS_REGISTRY_FILENAME = 'class_registry.json'
LABEL_STATS_FILENAME = 'label_stats.json'
STRATUM_MAP_FILENAME = 'stratum_map.json'


@dataclass(frozen=True)
class SingleClassExportProfile:
    """Deployment-supplied configuration for one narrowed export.

    This is the whole reason the exporter is generic: a deployment names
    its own target classes, its own region vocabulary and its own output
    location here, instead of the exporter hardcoding any of them.
    """

    # Directory name under the export root, and the default dataset
    # identity recorded in the manifest. Keeps a narrowed export's
    # artifacts (and its own `current` symlink) out of the multi-class
    # export root, so the two can never clobber each other.
    name: str = 'single_class'

    # Ordered registry class ids forming this export's vocabulary. One id
    # is the single-class case; several is a class-subset export. The
    # dense label id written into the .txt files is the INDEX INTO THIS
    # TUPLE, so the ordering here is the dataset's class order and is
    # frozen into class_registry.json's export_id_map.
    #
    # Required for box_source='item'. For box_source='region' it is
    # instead an optional filter on which parent items' regions are
    # collected (empty = every item's region).
    class_ids: tuple[int, ...] = ()

    # Where each label box comes from: the item's own `bbox_norm`, or the
    # item's region-of-interest sub-annotation (`RegionFields.bbox_norm`).
    box_source: BoxSource = 'item'

    # Class name for the region vocabulary in region mode, where the
    # single exported class is the region itself rather than a registry
    # entry. Purely a label in data.yaml.
    region_class_name: str = 'region'

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    # test_ratio is whatever remains: 1 - train_ratio - val_ratio.

    # Absolute output root override. None = `CurationConfig.export_root / name`.
    output_root: Path | None = None
    current_link_name: str = 'current'

    # Recorded in the manifest so a consumer can tell a narrowed export
    # from a full multi-class one without inspecting data.yaml. Read by
    # `src/routers/curation_train.py`'s single-class preflight branch.
    dataset_kind: str = 'single_class'


@dataclass
class SingleClassSplitCounts:
    train: int = 0
    val: int = 0
    test: int = 0

    def to_dict(self) -> dict[str, int]:
        return {'train': self.train, 'val': self.val, 'test': self.test}


@dataclass
class SingleClassExportResult:
    export_dir: str
    version_tag: str
    manifest_path: str
    data_yaml_path: str
    dataset_sha: str
    frozen_test_sha: str
    split_counts: SingleClassSplitCounts
    image_count: int
    class_count: int
    positive_images: int
    background_images: int
    started_at: str
    finished_at: str
    current_symlink: str


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

        resolved_dir = Path(export_dir) if export_dir else self._default_export_dir(started)
        for split in ('train', 'val', 'test'):
            (resolved_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (resolved_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

        splits = self._assign_splits(rows, seed=seed, skip_test_split=skip_test_split)
        counts, stratum_distribution, image_jobs = self._write_labels(
            rows, splits, resolved_dir, copy_images=copy_images
        )
        image_copy_stats = await self._copy_images(
            image_jobs, img_max_side=img_max_side, copy_images=copy_images
        )

        atomic_write_text(
            resolved_dir / STRATUM_MAP_FILENAME,
            json.dumps({r.item_id: r.stratum for r in rows}, indent=2, sort_keys=True),
        )
        data_yaml_path = resolved_dir / DATA_YAML_FILENAME
        data_yaml_path.write_text(_render_data_yaml(resolved_dir, names))
        self._write_class_registry(resolved_dir, names)
        self._write_label_stats(resolved_dir, names)

        dataset_sha = await asyncio.to_thread(label_content_sha, resolved_dir, names)
        frozen_test_sha = await asyncio.to_thread(frozen_test_sha_of, resolved_dir)
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
            'code_sha': _code_sha(),
            'split_counts': counts.to_dict(),
            'stratum_distribution': dict(sorted(stratum_distribution.items())),
            'stratum_count': len(stratum_distribution),
            'image_count': counts.train + counts.val + counts.test,
            'image_copy': image_copy_stats,
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

    def _default_export_dir(self, started: datetime) -> Path:
        return self._output_root() / started.strftime('%Y%m%dT%H%M%SZ')

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

    def _assign_splits(
        self, rows: list[_FrameRow], *, seed: int, skip_test_split: bool
    ) -> dict[str, str]:
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
        splits = stratified_split(
            rows,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            group_key='image_id',
        )
        if skip_test_split:
            splits = {k: ('train' if v == 'test' else v) for k, v in splits.items()}
        return splits

    # ----------------------------------------------------------------- write

    def _write_labels(
        self,
        rows: list[_FrameRow],
        splits: dict[str, str],
        export_dir: Path,
        *,
        copy_images: bool,
    ) -> tuple[SingleClassSplitCounts, dict[str, dict[str, int]], list[tuple[str, str, Any]]]:
        counts = SingleClassSplitCounts()
        distribution: dict[str, dict[str, int]] = defaultdict(
            lambda: {'train': 0, 'val': 0, 'test': 0}
        )
        image_jobs: list[tuple[str, str, Any]] = []
        for row in rows:
            split = splits[row.item_id]
            label_path = export_dir / 'labels' / split / f'{row.item_id}.txt'
            label_path.write_text(_render_label(row.boxes), encoding='utf-8')
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

    async def _copy_images(
        self,
        jobs: list[tuple[str, str, Any]],
        *,
        img_max_side: int,
        copy_images: bool,
    ) -> dict[str, Any]:
        if not copy_images or not jobs:
            return {'attempted': 0, 'copied': 0, 'failed': 0, 'errors': []}
        loop = asyncio.get_running_loop()
        copied = failed = 0
        errors: list[str] = []
        with ProcessPoolExecutor(max_workers=self.max_image_workers) as pool:
            futures = [
                # 'aspect' resizes the longest side down without padding, so
                # the normalized labels stay valid -- a letterbox offset here
                # would silently shift every box in the dataset.
                loop.run_in_executor(
                    pool, _copy_or_resize_one, src, dest, 'aspect', img_max_side, crop
                )
                for src, dest, crop in jobs
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

    def _write_class_registry(self, export_dir: Path, names: list[str]) -> None:
        """Freeze the vocabulary + dense-id map next to the dataset.

        A promote step needs to know which REGISTRY class each dense label
        id means; reading the live registry instead would silently describe
        a different dataset once the registry moves on.
        """
        export_id_map = {str(cid): i for i, cid in enumerate(self.profile.class_ids)}
        payload = {
            'version': 1,
            'exported_at': datetime.now(UTC).isoformat(),
            'dataset_kind': self.profile.dataset_kind,
            'box_source': self.profile.box_source,
            'classes': [
                {'class_id': cid, 'class_name': name}
                for cid, name in zip(self.profile.class_ids, names, strict=False)
            ],
            'export_id_map': export_id_map,
            'names': names,
        }
        atomic_write_text(
            export_dir / CLASS_REGISTRY_FILENAME, json.dumps(payload, indent=2, sort_keys=True)
        )

    @staticmethod
    def _write_label_stats(export_dir: Path, names: list[str]) -> None:
        stats: dict[str, int] = dict.fromkeys(names, 0)
        for split in ('train', 'val', 'test'):
            for label_file in sorted((export_dir / 'labels' / split).glob('*.txt')):
                for line in label_file.read_text(encoding='utf-8').splitlines():
                    if not line.strip():
                        continue
                    dense = int(line.split()[0])
                    if 0 <= dense < len(names):
                        stats[names[dense]] += 1
        atomic_write_text(
            export_dir / LABEL_STATS_FILENAME, json.dumps(stats, indent=2, sort_keys=True)
        )


# =============================================================================
# Integrity + rendering helpers
# =============================================================================


def frozen_test_sha_of(export_dir: Path) -> str:
    """Checksum over the test split's *identity* — which frames are in it.

    Deliberately filenames only, not content: the guarantee being made is
    "the held-out evaluation set is the same set of frames as last time",
    which must keep holding after a label correction inside the test set.
    Content changes there are caught by ``dataset_sha`` instead. Returns
    ``''`` when there is no test split.
    """
    test_labels = export_dir / 'labels' / 'test'
    if not test_labels.is_dir():
        return ''
    names = sorted(p.name for p in test_labels.glob('*.txt'))
    if not names:
        return ''
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()[:16]


def _render_label(boxes: Iterable[tuple[int, float, float, float, float]]) -> str:
    """YOLO label text. An empty string (empty file) means a background frame."""
    lines = [f'{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}' for cid, cx, cy, w, h in boxes]
    return '\n'.join(lines) + '\n' if lines else ''


def _render_data_yaml(export_dir: Path, names: list[str]) -> str:
    """``data.yaml`` for the YOLO trainer, with an absolute ``path``.

    Absolute so ``yolo train data=<dir>/data.yaml`` resolves the splits
    regardless of the working directory the trainer happens to start in.
    """
    name_lines = ''.join(f'  {i}: {name}\n' for i, name in enumerate(names))
    return (
        f'path: {export_dir}\n'
        'train: images/train\n'
        'val: images/val\n'
        'test: images/test\n'
        f'nc: {len(names)}\n'
        'names:\n' + name_lines
    )


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


__all__ = [
    'CLASS_REGISTRY_FILENAME',
    'DATA_YAML_FILENAME',
    'LABEL_STATS_FILENAME',
    'MANIFEST_FILENAME',
    'STRATUM_MAP_FILENAME',
    'SingleClassExportProfile',
    'SingleClassExportResult',
    'SingleClassExportService',
    'SingleClassSplitCounts',
    'frozen_test_sha_of',
    'label_content_sha',
    'resolve_current_single_class_dir',
]
