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

Split assignment is a deterministic, **stratified** hash-order bucket
rather than a stored crop->split mapping, so re-running an export with the
same recorded seed reproduces the same split without needing to persist
per-item split assignments anywhere. Items already carrying a frozen
``test_holdout`` flag always land in the ``test`` split regardless of the
hash, honoring whatever holdout freeze a deployment has already committed
to (see ``src.services.curation.holdout``).

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
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from src.clients.curation_opensearch import ClassRegistry, get_class_registry
from src.config import CurationConfig, get_curation_config
from src.core.logging import get_logger
from src.services.curation.export_support import (
    _build_export_id_map,
    _code_sha,
    _copy_or_resize_one,
    _ExportRow,
    _remap_rows_to_export_ids,
    _resolve_source_path,
    dataset_checksum,
    even_stratified_sample,
    hash_split,
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
    export_dir: str
    version_tag: str
    manifest_path: str
    data_yaml_path: str
    dataset_sha: str
    split_counts: SplitCounts
    image_count: int
    class_count: int
    started_at: str
    finished_at: str
    current_symlink: str


def _write_yolo_label(path: Path, class_id: int, bbox_norm: list[float]) -> None:
    x1, y1, x2, y2 = bbox_norm
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    path.write_text(f'{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')


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

    Writes one label ``.txt`` per exported item (dense export class id +
    normalized cx/cy/w/h), copies (optionally resizing) the source pixels
    into ``images/<split>/``, a ``data.yaml`` class map, a
    ``class_registry.json`` snapshot + dense ``export_id_map``, a
    ``label_stats.json`` per-class count, and a ``manifest.json``
    reproducibility envelope (dataset checksum, split counts, seed, code
    sha, frozen-holdout sha, timestamps). Deliberately narrower than the
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
        body: dict[str, Any] = {
            'size': 500,
            'query': query,
            '_source': [
                'crop_id',
                'image_id',
                'image_path',
                'bbox_norm',
                'class_id',
                'class_name',
                'test_holdout',
                'cluster_id',
            ],
        }
        resp = await self.opensearch.search(index=self.config.items_index, body=body, scroll='5m')
        scroll_id = resp.get('_scroll_id')
        hits = list((resp.get('hits') or {}).get('hits') or [])
        out = list(hits)
        try:
            while hits:
                resp = await self.opensearch.scroll(scroll_id=scroll_id, scroll='5m')
                scroll_id = resp.get('_scroll_id')
                hits = list((resp.get('hits') or {}).get('hits') or [])
                out.extend(hits)
        finally:
            if scroll_id:
                try:
                    await self.opensearch.clear_scroll(scroll_id=scroll_id)
                except Exception as exc:
                    logger.warning('export_clear_scroll_failed', err=str(exc))
        return out

    def _hits_to_rows(self, hits: list[dict[str, Any]]) -> list[_ExportRow]:
        rows: list[_ExportRow] = []
        for hit in hits:
            src = hit.get('_source') or {}
            item_id = src.get('crop_id') or hit.get('_id')
            bbox = src.get('bbox_norm')
            class_id = src.get('class_id')
            if not item_id or bbox is None or len(bbox) != 4 or class_id is None:
                continue
            rows.append(
                _ExportRow(
                    item_id=str(item_id),
                    image_id=str(src.get('image_id') or ''),
                    image_path=str(src.get('image_path') or ''),
                    bbox_norm=list(bbox),
                    class_id=int(class_id),
                    class_name=str(src.get('class_name') or class_id),
                    has_test_crop=bool(src.get('test_holdout')),
                    cluster_id=src.get('cluster_id'),
                )
            )
        return rows

    async def _apply_dedup(
        self, rows: list[_ExportRow], dedup_threshold: float | None
    ) -> tuple[list[_ExportRow], dict[str, Any]]:
        if dedup_threshold is None:
            return rows, {'enabled': False}
        kept, stats = await dedup_rows_by_embedding(
            self.opensearch, rows, threshold=dedup_threshold, config=self.config
        )
        return kept, stats

    async def _copy_images(
        self,
        jobs: list[tuple[str, str]],
        *,
        resize_mode: Literal['letterbox', 'aspect'] | None,
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

    async def export_dataset(
        self,
        *,
        export_dir: Path | None = None,
        version_tag: str = '',
        seed: int = 42,
        max_images: int | None = None,
        dedup_threshold: float | None = None,
        group_key: str | None = 'cluster_id',
        resize_mode: Literal['letterbox', 'aspect'] | None = None,
        image_size: int = 640,
        copy_images: bool = True,
        max_image_workers: int = 4,
    ) -> ExportResult:
        """Export every validated, non-dismissed item as a multi-class YOLO
        detection dataset.

        Honors a frozen ``test_holdout`` flag for the test split; every
        other item's split comes from :func:`stratified_split` — a
        deterministic per-class, per-``group_key`` bucket assignment
        (:func:`hash_split` still exists as the underlying single-key
        primitive it's built from).

        ``dedup_threshold`` (if given) collapses whole-frame near-duplicate
        bursts (cosine >= threshold on the images index's secondary
        embedding) to one representative frame before the split is
        computed, via
        :func:`~src.services.detection.frame_dedup.dedup_rows_by_embedding`.

        ``max_images`` caps the export via
        :func:`~src.services.curation.export_support.even_stratified_sample`
        — an even round-robin over ``class_id`` applied *after* dedup and
        the dense-id remap, so a rare class can't be squeezed out and the
        final count is exactly ``min(max_images, pool)``. The manifest
        records which of the two modes ran as ``sampling_mode``
        (``'all'`` / ``'stratified_even'``).

        ``resize_mode`` (``'letterbox'`` or ``'aspect'``, default ``None``
        = copy as-is) controls how source pixels are resized into
        ``images/<split>/`` when ``copy_images`` is true.
        """
        started_at = datetime.now(UTC).isoformat()
        query = {
            'bool': {
                'must': [{'term': {'class_validated': True}}],
                'must_not': [{'exists': {'field': 'review_dismissed_at'}}],
            }
        }
        # Scroll the FULL cohort — never cap here. Truncating the raw hit
        # list would hand the budget to whatever the scroll returned first
        # and let a rare class vanish; the cap is a class-balanced sample
        # applied below, once dedup and the id remap have settled the pool.
        hits = await self._scroll_items(query)

        rows = self._hits_to_rows(hits)
        rows, dedup_stats = await self._apply_dedup(rows, dedup_threshold)

        registry_file = self.registry.load()
        id_map = _build_export_id_map(registry_file.classes)
        rows = _remap_rows_to_export_ids(rows, id_map)

        # Cap AFTER dedup + remap, so the final count lands at exactly
        # min(max_images, pool) instead of drifting below it.
        sampling_mode = 'all'
        if max_images is not None and len(rows) > max_images:
            rows = even_stratified_sample(
                rows, max_images, lambda r: str(r.class_id), random.Random(seed)
            )
            sampling_mode = 'stratified_even'
            logger.info('export_sampled', n_images=len(rows), max_images=max_images)

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

        item_split = stratified_split(
            rows,
            seed=seed,
            train_ratio=self.profile.train_ratio,
            val_ratio=self.profile.val_ratio,
            group_key=group_key,
        )

        counts = SplitCounts()
        item_ids: list[str] = []
        holdout_item_ids: list[str] = []
        image_jobs: list[tuple[str, str]] = []

        for row in rows:
            split = item_split[row.item_id]
            label_path = labels_root / split / f'{row.item_id}.txt'
            _write_yolo_label(label_path, row.export_class_id, row.bbox_norm)
            setattr(counts, split, getattr(counts, split) + 1)
            item_ids.append(row.item_id)
            if row.has_test_crop:
                holdout_item_ids.append(row.item_id)

            if copy_images:
                src_path = _resolve_source_path(row.image_path, self.config)
                if src_path is not None:
                    ext = src_path.suffix or '.jpg'
                    dest = images_root / split / f'{row.item_id}{ext}'
                    image_jobs.append((str(src_path), str(dest)))
                else:
                    logger.warning(
                        'export_source_image_unresolved',
                        item_id=row.item_id,
                        image_path=row.image_path,
                    )

        image_copy_stats = await self._copy_images(
            image_jobs,
            resize_mode=resize_mode,
            image_size=image_size,
            max_workers=max_image_workers,
        )

        checksum = dataset_checksum(item_ids)
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

        label_stats: dict[str, int] = dict.fromkeys(names, 0)
        for split_name in ('train', 'val', 'test'):
            for label_file in (labels_root / split_name).glob('*.txt'):
                cid = int(label_file.read_text().split()[0])
                label_stats[names[cid]] += 1
        (resolved_export_dir / ARTIFACT_FILENAMES['label_stats']).write_text(
            json.dumps(label_stats, indent=2)
        )

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

        manifest = {
            'version_tag': version_tag,
            'seed': seed,
            'group_key': group_key,
            'dataset_sha': checksum,
            'image_count': len(item_ids),
            'split_counts': counts.to_dict(),
            'class_count': len(names),
            'started_at': started_at,
            'finished_at': finished_at,
            'exported_at': finished_at,
            'code_sha': _code_sha(),
            'frozen_holdout_sha': compute_holdout_sha(holdout_item_ids)
            if holdout_item_ids
            else None,
            'dedup': dedup_stats,
            'max_images': max_images,
            'sampling_mode': sampling_mode,
            'image_copy': image_copy_stats,
            'resize_mode': resize_mode,
        }
        manifest_path = resolved_export_dir / ARTIFACT_FILENAMES['manifest']
        self._atomic_write_text(manifest_path, json.dumps(manifest, indent=2))

        current_symlink = self.config.export_root / 'current'
        self._atomic_symlink_flip(current_symlink, resolved_export_dir)

        return ExportResult(
            export_dir=str(resolved_export_dir),
            version_tag=version_tag,
            manifest_path=str(manifest_path),
            data_yaml_path=str(data_yaml_path),
            dataset_sha=checksum,
            split_counts=counts,
            image_count=len(item_ids),
            class_count=len(names),
            started_at=started_at,
            finished_at=finished_at,
            current_symlink=str(current_symlink),
        )

    @staticmethod
    def _atomic_write_text(path: Path, payload: str) -> None:
        """tmp-write + fsync + rename — never leaves a partially-written
        manifest visible to a concurrent reader (e.g. preflight scan)."""
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        with tmp_path.open('w', encoding='utf-8') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)

    @staticmethod
    def _atomic_symlink_flip(symlink_path: Path, target: Path) -> None:
        """Point ``symlink_path`` at ``target`` via a write-then-rename.

        A reader that resolves ``current`` mid-flip always sees either the
        old or the new export directory, never a missing/half-written
        symlink.
        """
        symlink_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_symlink = symlink_path.with_name(f'.{symlink_path.name}.tmp.{os.getpid()}')
        if tmp_symlink.exists() or tmp_symlink.is_symlink():
            tmp_symlink.unlink()
        tmp_symlink.symlink_to(target, target_is_directory=True)
        tmp_symlink.replace(symlink_path)


__all__ = [
    'ARTIFACT_FILENAMES',
    'REGISTRY_ARTIFACT_CONTENT_TYPES',
    'ExportProfile',
    'ExportResult',
    'GenericYoloExportService',
    'SplitCounts',
    'dataset_checksum',
    'even_stratified_sample',
    'hash_split',
    'resolve_current_export_dir',
    'stratified_split',
]
