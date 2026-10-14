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
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.config.project_context import run_in_executor_bound
from src.services.curation.export_models import (
    CLASS_REGISTRY_FILENAME,
    LABEL_STATS_FILENAME,
    SingleClassExportProfile,
    SplitCounts,
)
from src.services.curation.export_support import _copy_or_resize_one, atomic_write_text


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from src.clients.curation_opensearch.registry import ClassRegistry


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
