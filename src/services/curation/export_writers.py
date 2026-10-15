"""Writers for the artifacts each exporter puts next to a dataset.

Label text, ``data.yaml``, the frozen class registry, per-class stats and
the process-pool image copy. The two exporters build different artifacts
(a whole-registry dataset versus a narrowed vocabulary with backgrounds),
so each keeps its own writers here rather than sharing one.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.config.project_context import run_in_executor_bound
from src.core.logging import get_logger
from src.services.curation.export_images import image_file_stem, yolo_line
from src.services.curation.export_models import (
    ARTIFACT_FILENAMES,
    CLASS_REGISTRY_FILENAME,
    LABEL_STATS_FILENAME,
    SingleClassExportProfile,
    SplitCounts,
)
from src.services.curation.export_support import (
    _copy_or_resize_one,
    _resolve_source_path,
    atomic_write_text,
    source_frozen_test_sha,
)


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from src.clients.curation_opensearch import ClassRegistryFile
    from src.clients.curation_opensearch.registry import ClassRegistry
    from src.config import CurationConfig
    from src.services.curation.export_images import _ExportImage


logger = get_logger(__name__)


def class_split_rows(
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


def render_label(boxes: Iterable[tuple[int, float, float, float, float]]) -> str:
    """YOLO label text. An empty string (empty file) means a background frame."""
    lines = [f'{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}' for cid, cx, cy, w, h in boxes]
    return '\n'.join(lines) + '\n' if lines else ''


def render_data_yaml(export_dir: Path, names: list[str]) -> str:
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


def write_label_stats_from_disk(export_dir: Path, names: list[str]) -> None:
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


def write_class_registry(
    export_dir: Path,
    names: list[str],
    *,
    profile: SingleClassExportProfile,
    registry: ClassRegistry,
) -> None:
    """Freeze the vocabulary + dense-id map next to the dataset.

    A promote step needs to know which REGISTRY class each dense label
    id means; reading the live registry instead would silently describe
    a different dataset once the registry moves on.
    """
    export_id_map = {str(cid): i for i, cid in enumerate(profile.class_ids)}
    payload = {
        'version': 1,
        'exported_at': datetime.now(UTC).isoformat(),
        'dataset_kind': profile.dataset_kind,
        'box_source': profile.box_source,
        'classes': class_entries(profile, registry, names),
        'export_id_map': export_id_map,
        'names': names,
    }
    atomic_write_text(
        export_dir / CLASS_REGISTRY_FILENAME, json.dumps(payload, indent=2, sort_keys=True)
    )


def class_entries(
    profile: SingleClassExportProfile, registry: ClassRegistry, names: list[str]
) -> list[dict[str, Any]]:
    """``class_registry.json``'s ``classes``: the dataset's classes, or --
    in region mode, where the one dataset class is the region itself -- the
    parent-class filter, named by the live registry (never by the region
    class name, which a positional zip with ``names`` would give them)."""
    if profile.box_source != 'region':
        return [
            {'class_id': cid, 'class_name': name}
            for cid, name in zip(profile.class_ids, names, strict=False)
        ]
    by_id = {c.class_id: c.class_name for c in registry.load().classes}
    return [{'class_id': cid, 'class_name': by_id.get(cid, str(cid))} for cid in profile.class_ids]


async def copy_single_class_images(
    jobs: list[tuple[str, str, Any]],
    *,
    max_workers: int,
    img_max_side: int,
    copy_images: bool,
) -> dict[str, Any]:
    if not copy_images or not jobs:
        return {'attempted': 0, 'copied': 0, 'failed': 0, 'errors': []}
    loop = asyncio.get_running_loop()
    copied = failed = 0
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            # 'aspect' resizes the longest side down without padding, so
            # the normalized labels stay valid -- a letterbox offset here
            # would silently shift every box in the dataset.
            run_in_executor_bound(
                loop, pool, _copy_or_resize_one, src, dest, 'aspect', img_max_side, crop
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


@dataclass
class WrittenLabels:
    """What :func:`write_image_labels` counted and which image copies it planned."""

    counts: SplitCounts = field(default_factory=SplitCounts)
    object_counts: SplitCounts = field(default_factory=SplitCounts)
    per_class: dict[int, SplitCounts] = field(default_factory=dict)
    item_ids: list[str] = field(default_factory=list)
    holdout_item_ids: list[str] = field(default_factory=list)
    image_jobs: list[tuple[str, str]] = field(default_factory=list)
    # (import_source_stem, own label stem) of every test image, for the
    # manifest's source_frozen_test_sha.
    test_stems: list[tuple[str | None, str]] = field(default_factory=list)


def write_image_labels(
    images: list[_ExportImage],
    item_split: dict[str, str],
    *,
    names: list[str],
    labels_root: Path,
    images_root: Path,
    config: CurationConfig,
    copy_images: bool,
) -> WrittenLabels:
    """One label file per image under ``labels_root/<split>``; plans the pixel copies."""
    written = WrittenLabels(per_class={dense_id: SplitCounts() for dense_id in range(len(names))})
    for image in images:
        split = item_split[image.objects[0].item_id]
        stem = image_file_stem(image.image_id)
        if split == 'test':
            written.test_stems.append((image.objects[0].import_source_stem, stem))
        lines = [yolo_line(o.export_class_id, o.bbox_norm) for o in image.objects]
        (labels_root / split / f'{stem}.txt').write_text('\n'.join(lines) + '\n')
        setattr(written.counts, split, getattr(written.counts, split) + 1)
        setattr(
            written.object_counts, split, getattr(written.object_counts, split) + len(image.objects)
        )
        for obj in image.objects:
            class_counts = written.per_class[obj.export_class_id]
            setattr(class_counts, split, getattr(class_counts, split) + 1)
            written.item_ids.append(obj.item_id)
            if obj.has_test_crop:
                written.holdout_item_ids.append(obj.item_id)

        if copy_images:
            src_path = _resolve_source_path(image.image_path, config)
            if src_path is not None:
                ext = src_path.suffix or '.jpg'
                dest = images_root / split / f'{stem}{ext}'
                written.image_jobs.append((str(src_path), str(dest)))
            else:
                logger.warning(
                    'export_source_image_unresolved',
                    image_id=image.image_id,
                    image_path=image.image_path,
                )
    return written


def write_data_yaml(export_dir: Path, names: list[str]) -> Path:
    """The multi-class ``data.yaml`` (flow-style ``names``); returns its path."""
    data_yaml_path = export_dir / ARTIFACT_FILENAMES['data_yaml']
    data_yaml_path.write_text(
        f'path: {export_dir}\n'
        'train: images/train\n'
        'val: images/val\n'
        'test: images/test\n'
        f'nc: {len(names)}\n'
        f'names: {json.dumps(names)}\n'
    )
    return data_yaml_path


def write_label_stats(export_dir: Path, names: list[str], per_class: dict[int, SplitCounts]) -> int:
    """``label_stats.json`` (objects per class across every split); returns
    the number of classes that have any object."""
    label_stats: dict[str, int] = dict.fromkeys(names, 0)
    for dense_id, class_counts in per_class.items():
        label_stats[names[dense_id]] += sum(class_counts.to_dict().values())
    (export_dir / ARTIFACT_FILENAMES['label_stats']).write_text(json.dumps(label_stats, indent=2))
    # class_count is the registry size written into data.yaml
    # (nc/names) -- it stays that way for back-compat, but a class
    # with zero labeled objects in this export inflates that number
    # into looking like "84 classes of real data" when only a
    # handful have any objects at all. classes_with_objects is the
    # honest denominator for a per-class table.
    return sum(1 for v in label_stats.values() if v > 0)


def write_registry_snapshot(
    export_dir: Path,
    registry_file: ClassRegistryFile,
    id_map: dict[int, int],
    *,
    exported_at: str,
) -> None:
    """``class_registry.json``: the registry as exported plus the dense id map."""
    payload = {
        'version': 1,
        'exported_at': exported_at,
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
    (export_dir / ARTIFACT_FILENAMES['class_registry']).write_text(json.dumps(payload, indent=2))


def source_frozen_manifest(test_stems: list[tuple[str | None, str]]) -> dict[str, str]:
    """``source_frozen_test_sha`` when any test image came from an import."""
    sha = source_frozen_test_sha((s for s, _ in test_stems), (o for _, o in test_stems))
    return {'source_frozen_test_sha': sha} if sha else {}
