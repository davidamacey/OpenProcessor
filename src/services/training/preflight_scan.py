"""Real preflight label-file scan checks.

``/curation/train/preflight``'s ``empty_labels`` and ``region_pairing``
checks used to be hardcoded to always report ``'ok'`` -- stubbed, never
implemented. This module does the real work: scan an export's label
``.txt`` files once (cached per export dir + manifest hash, since
exports are immutable once written) and compute:

* how many images have zero label rows once a subset ``include_classes``
  filter is applied (catches "the requested subset excludes every class in
  this image -> 0 training rows for it", a real prior failure mode) -- the
  scan runs against the export's DENSE ids, so a subset filter is
  translated through ``export_id_map`` before comparing, matching the
  trainer's own subsetting logic.
* how many region-of-interest boxes (the active region profile's
  ``region_class_name``, e.g. a wheel on a car) have no matching parent
  item box in the same image (a pairing/parity signal). With no active
  region profile, or one whose ``region_class_name`` is empty, this
  check is not applicable: ``region_boxes`` / ``unpaired_region_boxes``
  stay ``0`` rather than guessing a class to pair against.

Item-class-only. Single-class exports are handled entirely by the
router's own additive ``dataset_kind == 'single_class'`` branch -- this
module never reads a single-class export.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


SPLITS = ('train', 'val', 'test')

# Past this many label files, a full per-line scan risks hanging preflight
# for a very large export. Report 'unknown' instead of blocking the request
# indefinitely. Override via env for hosts with a larger/faster disk.
DEFAULT_SCAN_CAP = int(os.environ.get('OP_PREFLIGHT_SCAN_CAP', '200000'))

# Small in-process cache: exports are immutable once written, so a repeat
# preflight call for the same export dir (same manifest content) never
# needs to re-scan the label files. Keyed on (export_dir, manifest sha256).
_scan_cache: dict[tuple[str, str, tuple[int, ...] | None], ScanResult] = {}


@dataclass(frozen=True)
class ScanResult:
    """Result of one label-file scan pass."""

    status: str  # 'ok' | 'unknown'
    total_images: int = 0
    empty_label_images: int = 0
    region_boxes: int = 0
    unpaired_region_boxes: int = 0
    reason: str | None = None  # populated only when status == 'unknown'


def _manifest_fingerprint(export_dir: Path) -> str:
    """A cheap, stable fingerprint for cache invalidation.

    Exports are frozen once written (design invariant), so the manifest's
    own bytes are a fine proxy for "has this export changed" without
    hashing every label file on every preflight call.
    """
    manifest_path = export_dir / 'manifest.json'
    try:
        data = manifest_path.read_bytes()
    except OSError:
        return 'no-manifest'
    return hashlib.sha256(data).hexdigest()


def _region_export_id(class_registry_payload: dict, class_name: str) -> int | None:
    """Resolve ``class_name``'s DENSE export id from a loaded
    ``class_registry.json`` payload. ``None`` if ``class_name`` is empty,
    this export has no such class, or the payload predates the
    ``export_id_map`` field."""
    if not class_name:
        return None
    export_id_map = class_registry_payload.get('export_id_map')
    if not isinstance(export_id_map, dict):
        return None
    registry_id: int | None = None
    for entry in class_registry_payload.get('classes') or []:
        if isinstance(entry, dict) and entry.get('class_name') == class_name:
            try:
                registry_id = int(entry['class_id'])
            except (KeyError, TypeError, ValueError):
                return None
            break
    if registry_id is None:
        return None
    val = export_id_map.get(str(registry_id))
    return int(val) if val is not None else None


def _box_center_inside(
    parent_box: tuple[float, float, float, float], region_cx: float, region_cy: float
) -> bool:
    """True if the region box's center falls inside the parent item box.

    Cheap containment heuristic (normalized YOLO cx/cy/w/h) rather than a
    full IoU -- a region is "paired" with a parent item when it visually
    sits on that item's box, which containment approximates well enough
    for a preflight parity signal (not a training-time correctness gate).
    """
    pcx, pcy, pw, ph = parent_box
    return (pcx - pw / 2) <= region_cx <= (pcx + pw / 2) and (pcy - ph / 2) <= region_cy <= (
        pcy + ph / 2
    )


def scan_export_labels(
    export_dir: Path,
    *,
    include_classes: list[int] | None = None,
    scan_cap: int = DEFAULT_SCAN_CAP,
) -> ScanResult:
    """Scan an export's label files once, cached per export.

    ``include_classes`` (if given) is a REGISTRY id list, translated
    through this export's own ``class_registry.json:export_id_map`` before
    filtering -- never compared directly against the dense ids the label
    files actually carry (the same rule ``subset_dataset.py`` follows).
    """
    if not export_dir.is_dir():
        return ScanResult('unknown', reason=f'{export_dir} not found or unreadable')

    # The class filter changes the result, so it is part of the key.
    class_filter = tuple(sorted(set(include_classes))) if include_classes else None
    cache_key = (str(export_dir), _manifest_fingerprint(export_dir), class_filter)
    cached = _scan_cache.get(cache_key)
    if cached is not None:
        return cached

    label_files: list[Path] = []
    for split in SPLITS:
        split_dir = export_dir / 'labels' / split
        if split_dir.is_dir():
            label_files.extend(split_dir.glob('*.txt'))

    if len(label_files) > scan_cap:
        result = ScanResult(
            'unknown',
            total_images=len(label_files),
            reason=f'{len(label_files)} label files exceeds scan cap {scan_cap}',
        )
        _scan_cache[cache_key] = result
        return result

    registry_path = export_dir / 'class_registry.json'
    try:
        registry_payload = json.loads(registry_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        registry_payload = {}
    export_id_map_raw = registry_payload.get('export_id_map')
    export_id_map: dict[int, int] = (
        {int(k): int(v) for k, v in export_id_map_raw.items()}
        if isinstance(export_id_map_raw, dict)
        else {}
    )
    from src.services.detection.profile_registry import get_active_region_profile

    active_profile = get_active_region_profile()
    region_class_name = active_profile.region_class_name if active_profile else ''
    region_export_id = _region_export_id(registry_payload, region_class_name)

    keep_dense_ids: set[int] | None = None
    if include_classes:
        keep_dense_ids = {export_id_map[c] for c in include_classes if c in export_id_map}

    total = 0
    empty = 0
    region_boxes = 0
    unpaired = 0
    for txt in label_files:
        total += 1
        try:
            lines = txt.read_text(encoding='utf-8').splitlines()
        except OSError:
            continue
        rows: list[tuple[int, tuple[float, float, float, float]]] = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
                box = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            except (ValueError, IndexError):
                continue
            if keep_dense_ids is not None and cid not in keep_dense_ids:
                continue
            rows.append((cid, box))

        if not rows:
            empty += 1
            continue

        if region_export_id is not None:
            regions = [box for cid, box in rows if cid == region_export_id]
            parents = [box for cid, box in rows if cid != region_export_id]
            region_boxes += len(regions)
            for rcx, rcy, _rw, _rh in regions:
                if not any(_box_center_inside(pb, rcx, rcy) for pb in parents):
                    unpaired += 1

    result = ScanResult(
        'ok',
        total_images=total,
        empty_label_images=empty,
        region_boxes=region_boxes,
        unpaired_region_boxes=unpaired,
    )
    _scan_cache[cache_key] = result
    return result


__all__ = ['DEFAULT_SCAN_CAP', 'ScanResult', 'scan_export_labels']
