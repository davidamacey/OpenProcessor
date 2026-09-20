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

Dense export ids (``class_registry.json:export_id_map``) are resolved from
the live :class:`~src.clients.curation_opensearch.ClassRegistry` at export
time and frozen into the export directory, so a subset-training request's
``include_classes`` (always expressed in REGISTRY ids) can be translated to
the dense ids that actually appear in the written label files — see
``src/services/training/preflight_scan.py`` and
``src/routers/curation_train.py``, both of which read this file back. This
closes a real bug: ``export_dataset`` previously never wrote
``class_registry.json`` at all, so both readers' ``include_classes``
filtering was silently inert on every export this codebase produced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch import ClassRegistry, get_class_registry
from src.config import CurationConfig, get_curation_config
from src.core.logging import get_logger


if TYPE_CHECKING:
    from src.clients.curation_opensearch import RegistryClassEntry


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


@dataclass
class _ExportRow:
    """One label row scrolled off the items index, en route to a label file."""

    item_id: str
    image_path: str
    bbox_norm: list[float]
    class_id: int
    class_name: str
    has_test_crop: bool = False
    export_class_id: int = -1


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


def _build_export_id_map(classes: list[RegistryClassEntry]) -> dict[int, int]:
    """Registry ``class_id`` -> contiguous dense export id (0-indexed).

    Deprecated classes are skipped entirely — they never occupy a dense
    slot. Order is ascending registry ``class_id``, so the mapping only
    changes when the live registry's set of non-deprecated classes
    changes, not on export-to-export scroll-order noise (the previous
    behavior assigned dense ids in first-seen scroll order, which was
    silently non-deterministic).
    """
    live_ids = sorted(c.class_id for c in classes if not c.deprecated)
    return {registry_id: dense_id for dense_id, registry_id in enumerate(live_ids)}


def _remap_rows_to_export_ids(rows: list[_ExportRow], id_map: dict[int, int]) -> list[_ExportRow]:
    """Apply ``id_map`` to every row's ``class_id``, in place.

    Rows whose ``class_id`` has no entry in ``id_map`` (deprecated or
    otherwise not in the live registry) are dropped — there is no valid
    dense id to write into a label file for them.
    """
    kept: list[_ExportRow] = []
    for row in rows:
        dense = id_map.get(row.class_id)
        if dense is None:
            logger.warning(
                'export_row_dropped_unmapped_class',
                item_id=row.item_id,
                class_id=row.class_id,
            )
            continue
        row.export_class_id = dense
        kept.append(row)
    return kept


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
    normalized cx/cy/w/h), a ``data.yaml`` class map, a
    ``class_registry.json`` snapshot + dense ``export_id_map``, a
    ``label_stats.json`` per-class count, and a ``manifest.json``
    reproducibility envelope (dataset checksum, split counts, seed,
    timestamps). Deliberately narrower than the reference exporter — see
    module docstring.
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
                    image_path=str(src.get('image_path') or ''),
                    bbox_norm=list(bbox),
                    class_id=int(class_id),
                    class_name=str(src.get('class_name') or class_id),
                    has_test_crop=bool(src.get('test_holdout')),
                )
            )
        return rows

    async def export_dataset(
        self,
        *,
        export_dir: Path | None = None,
        version_tag: str = '',
        seed: int = 42,
        max_images: int | None = None,
        dedup_threshold: float | None = None,  # noqa: ARG002 - call-site compat; near-dup collapsing is an overlay hook, not implemented generically here
    ) -> ExportResult:
        """Export every validated, non-dismissed item as a multi-class YOLO
        detection dataset.

        Honors a frozen ``test_holdout`` flag for the test split; every
        other item's split is a deterministic ``(seed, item_id)`` hash
        bucket (:func:`hash_split`). Class ids written to the label files
        are DENSE export ids resolved from the live
        :class:`~src.clients.curation_opensearch.ClassRegistry` at export
        time (see :func:`_build_export_id_map`) — the ``class_registry.json``
        artifact this writes is what lets a subset-training request's
        ``include_classes`` (expressed in registry ids) translate to those
        dense ids.
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

        rows = self._hits_to_rows(hits)

        registry_file = self.registry.load()
        id_map = _build_export_id_map(registry_file.classes)
        rows = _remap_rows_to_export_ids(rows, id_map)

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

        counts = SplitCounts()
        item_ids: list[str] = []

        for row in rows:
            split = (
                'test'
                if row.has_test_crop
                else hash_split(row.item_id, seed, self.profile.train_ratio, self.profile.val_ratio)
            )
            label_path = labels_root / split / f'{row.item_id}.txt'
            _write_yolo_label(label_path, row.export_class_id, row.bbox_norm)
            setattr(counts, split, getattr(counts, split) + 1)
            item_ids.append(row.item_id)

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
