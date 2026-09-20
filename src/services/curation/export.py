"""Generic YOLO-format dataset export service.

The reference implementation this was ported from
(see ``docs/design/curation_design_rationale.md`` for the genericization
approach) splits into two halves: artifact filenames, class
lists and split ratios are deployment data (an ``ExportProfile``, extracted
here), while the YOLO-format writer, split logic and manifest/checksum
mechanism are generic algorithm code that stays code. The bespoke
letterbox-resize / whole-frame-vs-crop / near-dup-collapsing features of
the reference exporter (a 1143-LOC domain-specific service, never
ported) are intentionally NOT reproduced here; a
deployment-specific overlay can extend :class:`GenericYoloExportService`
directly if it needs them (plan §7 R5 — the generic curation stack ships
with a thinner export path than the reference by design, tracked as the
most likely first follow-up after merge).

Split assignment is a deterministic ``sha256(seed:item_id)`` hash bucket
rather than a stored crop->split mapping, so re-running an export with the
same recorded seed reproduces the same split without needing to persist
per-item split assignments anywhere. Items already carrying a frozen
``test_holdout`` flag always land in the ``test`` split regardless of the
hash, honoring whatever holdout freeze a deployment has already committed
to (see ``src.services.curation.holdout``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import CurationConfig, get_curation_config
from src.core.logging import get_logger


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


def hash_split(key: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    """Deterministic ``(seed, key)`` -> ``{'train','val','test'}`` bucket.

    Stable across export re-runs with the same seed — the split is
    re-derivable from the manifest's recorded seed rather than requiring a
    persisted item->split mapping.
    """
    digest = hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    if frac < train_ratio:
        return 'train'
    if frac < train_ratio + val_ratio:
        return 'val'
    return 'test'


def dataset_checksum(item_ids: list[str]) -> str:
    """Checksum over the sorted set of item ids that went into an export."""
    return hashlib.sha256('\n'.join(sorted(item_ids)).encode()).hexdigest()


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

    Writes one label ``.txt`` per exported item (class id + normalized
    cx/cy/w/h), a ``data.yaml`` class map, a ``label_stats.json`` per-class
    count, and a ``manifest.json`` reproducibility envelope (dataset
    checksum, split counts, seed, timestamps). Deliberately narrower than
    the reference exporter — see module docstring.
    """

    def __init__(
        self,
        opensearch: Any,
        *,
        config: CurationConfig | None = None,
        profile: ExportProfile | None = None,
    ) -> None:
        self.opensearch = opensearch
        self.config = config or get_curation_config()
        self.profile = profile or ExportProfile()

    async def _scroll_items(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            'size': 500,
            'query': query,
            '_source': [
                'crop_id',
                'image_path',
                'bbox_norm',
                'class_id',
                'class_name',
                'test_holdout',
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

    async def export_dataset(
        self,
        *,
        export_dir: Path | None = None,
        version_tag: str = '',
        seed: int = 42,
        max_images: int | None = None,
        dedup_threshold: float | None = None,  # noqa: ARG002 - call-site compat; near-dup collapsing is an overlay hook, not implemented generically here
        class_names: list[str] | None = None,
    ) -> ExportResult:
        """Export every validated, non-dismissed item as a multi-class YOLO
        detection dataset.

        Honors a frozen ``test_holdout`` flag for the test split; every
        other item's split is a deterministic ``(seed, item_id)`` hash
        bucket (:func:`hash_split`).
        """
        started_at = datetime.now(UTC).isoformat()
        query = {
            'bool': {
                'must': [{'term': {'class_validated': True}}],
                'must_not': [{'exists': {'field': 'review_dismissed_at'}}],
            }
        }
        hits = await self._scroll_items(query)
        if max_images is not None:
            hits = hits[:max_images]

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

        names: list[str] = list(class_names) if class_names is not None else []
        counts = SplitCounts()
        item_ids: list[str] = []

        for hit in hits:
            src = hit.get('_source') or {}
            item_id = src.get('crop_id') or hit.get('_id')
            bbox = src.get('bbox_norm')
            class_id = src.get('class_id')
            if not item_id or bbox is None or len(bbox) != 4 or class_id is None:
                continue
            class_name = src.get('class_name') or str(class_id)
            if class_name not in names:
                names.append(class_name)
            resolved_class_id = names.index(class_name)

            split = (
                'test'
                if src.get('test_holdout')
                else hash_split(
                    str(item_id), seed, self.profile.train_ratio, self.profile.val_ratio
                )
            )
            label_path = labels_root / split / f'{item_id}.txt'
            _write_yolo_label(label_path, resolved_class_id, bbox)
            setattr(counts, split, getattr(counts, split) + 1)
            item_ids.append(str(item_id))

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

        manifest = {
            'version_tag': version_tag,
            'seed': seed,
            'dataset_sha': checksum,
            'image_count': len(item_ids),
            'split_counts': counts.to_dict(),
            'class_count': len(names),
            'started_at': started_at,
            'finished_at': finished_at,
            'exported_at': finished_at,
        }
        manifest_path = resolved_export_dir / ARTIFACT_FILENAMES['manifest']
        manifest_path.write_text(json.dumps(manifest, indent=2))

        current_symlink = self.config.export_root / 'current'
        current_symlink.parent.mkdir(parents=True, exist_ok=True)
        if current_symlink.is_symlink() or current_symlink.exists():
            current_symlink.unlink()
        current_symlink.symlink_to(resolved_export_dir, target_is_directory=True)

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


__all__ = [
    'ARTIFACT_FILENAMES',
    'REGISTRY_ARTIFACT_CONTENT_TYPES',
    'ExportProfile',
    'ExportResult',
    'GenericYoloExportService',
    'SplitCounts',
    'dataset_checksum',
    'hash_split',
    'resolve_current_export_dir',
]
