"""Pure COCO subset selection: license resolution, primary class per image,
seeded per-class, side-set and negative sampling, and the pinned-manifest check.
No network; unit-testable on a small synthetic COCO-shaped fixture."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from scripts.datasets._common import FetchError, load_pinned_manifest, seeded_sample, write_json


if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CLASSES = (
    'bicycle',
    'car',
    'motorcycle',
    'bus',
    'truck',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
)

# CLI --licenses token -> COCO licenses[].name (resolved by name, not id --
# see module docstring point 4).
LICENSE_ALIASES = {
    'by': 'Attribution License',
    'by-sa': 'Attribution-ShareAlike License',
    'no-known': 'No known copyright restrictions',
    'usgov': 'United States Government Work',
}
DEFAULT_LICENSES = ('by', 'by-sa', 'no-known', 'usgov')

MIN_BOX_AREA_FRACTION = 0.01


def resolve_license_names(tokens: list[str]) -> set[str]:
    bad = [t for t in tokens if t not in LICENSE_ALIASES]
    if bad:
        raise FetchError(f'unknown --licenses token(s) {bad}; valid: {sorted(LICENSE_ALIASES)}')
    return {LICENSE_ALIASES[t] for t in tokens}


def primary_class_per_image(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    category_id_to_name: dict[int, str],
    target_class_ids: set[int],
) -> dict[int, str]:
    """``image_id -> primary class name`` for images with a qualifying box.

    The primary class is whichever target-class box has the largest area,
    restricted to boxes covering >= ``MIN_BOX_AREA_FRACTION`` of the frame.
    Images with no qualifying box are absent from the result.
    """
    dims = {im['id']: (im['width'], im['height']) for im in images}
    best_area: dict[int, float] = {}
    best_cat: dict[int, int] = {}
    for ann in annotations:
        if ann.get('iscrowd', 0):
            continue
        cat_id = ann['category_id']
        if cat_id not in target_class_ids:
            continue
        image_id = ann['image_id']
        w, h = dims.get(image_id, (0, 0))
        frame_area = w * h
        if frame_area <= 0:
            continue
        _, _, bw, bh = ann['bbox']
        area = bw * bh
        if area / frame_area < MIN_BOX_AREA_FRACTION:
            continue
        if area > best_area.get(image_id, -1.0):
            best_area[image_id] = area
            best_cat[image_id] = cat_id
    return {img_id: category_id_to_name[cat_id] for img_id, cat_id in best_cat.items()}


def eligible_images(
    images: list[dict[str, Any]],
    primary_class: dict[int, str],
    license_id_to_name: dict[int, str],
    allowed_license_names: set[str],
    split: str,
) -> list[dict[str, Any]]:
    """Images with a primary class and an allowed license, sorted by id
    for deterministic downstream sampling."""
    out = []
    for im in images:
        cls = primary_class.get(im['id'])
        if cls is None:
            continue
        lic_name = license_id_to_name.get(im['license'])
        if lic_name not in allowed_license_names:
            continue
        out.append(
            {
                'image_id': im['id'],
                'file_name': im['file_name'],
                'split': split,
                'primary_class': cls,
                'license_name': lic_name,
                'flickr_url': im.get('flickr_url', ''),
                'coco_url': im.get('coco_url', ''),
                'width': im['width'],
                'height': im['height'],
            }
        )
    out.sort(key=lambda r: r['image_id'])
    return out


def select_per_class(
    val_pool: list[dict[str, Any]],
    train_pool: list[dict[str, Any]],
    classes: list[str],
    per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Val2017 first, topped up from train2017, seeded per class."""
    selected: list[dict[str, Any]] = []
    for cls in classes:
        val_cls = [r for r in val_pool if r['primary_class'] == cls]
        train_cls = [r for r in train_pool if r['primary_class'] == cls]
        chosen = seeded_sample(val_cls, per_class, seed)
        remaining = per_class - len(chosen)
        if remaining > 0:
            chosen += seeded_sample(train_cls, remaining, seed + 1)
        if len(chosen) < per_class:
            logger.warning(
                'class %r: only %d/%d eligible images available', cls, len(chosen), per_class
            )
        selected.extend(chosen)
    selected.sort(key=lambda r: r['image_id'])
    return selected


def select_side_sets(
    val_pool: list[dict[str, Any]],
    train_pool: list[dict[str, Any]],
    main_ids: set[int],
    side_sets: dict[str, int],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Disjoint side-set samples drawn from the leftover eligible pool
    (any class), seeded, in image-id order for determinism."""
    leftover = [r for r in (*val_pool, *train_pool) if r['image_id'] not in main_ids]
    leftover.sort(key=lambda r: r['image_id'])
    out: dict[str, list[dict[str, Any]]] = {}
    used: set[int] = set()
    for offset, (name, n) in enumerate(side_sets.items()):
        pool = [r for r in leftover if r['image_id'] not in used]
        chosen = seeded_sample(pool, n, seed + 100 + offset)
        used.update(r['image_id'] for r in chosen)
        out[name] = sorted(chosen, key=lambda r: r['image_id'])
    return out


def select_negatives(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    target_class_ids: set[int],
    license_id_to_name: dict[int, str],
    allowed_license_names: set[str],
    split: str,
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Seeded sample of allowed-license images with no box (crowd included)
    of any target class. Their other-category annotations are kept in
    ``coco_gt.json``, so they are negatives only for the chosen classes."""
    annotated = {a['image_id'] for a in annotations if a['category_id'] in target_class_ids}
    pool = []
    for im in images:
        if im['id'] in annotated:
            continue
        lic_name = license_id_to_name.get(im['license'])
        if lic_name not in allowed_license_names:
            continue
        pool.append(
            {
                'image_id': im['id'],
                'file_name': im['file_name'],
                'split': split,
                'primary_class': '',
                'license_name': lic_name,
                'flickr_url': im.get('flickr_url', ''),
                'coco_url': im.get('coco_url', ''),
                'width': im['width'],
                'height': im['height'],
            }
        )
    pool.sort(key=lambda r: r['image_id'])
    return sorted(seeded_sample(pool, n, seed + 200), key=lambda r: r['image_id'])


def verify_or_write_manifest(manifest_path: Path, selected: list[dict[str, Any]]) -> None:
    """Compare ``selected`` against the committed pin, or write it if this
    is the first run (bootstrap). Raises on drift."""
    pinned = load_pinned_manifest(manifest_path)
    pin_rows = [
        {
            'image_id': r['image_id'],
            'file_name': r['file_name'],
            'split': r['split'],
            'primary_class': r['primary_class'],
            'license_name': r['license_name'],
            'flickr_url': r['flickr_url'],
        }
        for r in selected
    ]
    if pinned is None:
        write_json(manifest_path, pin_rows)
        logger.info('wrote new pinned manifest %s (%d images)', manifest_path, len(pin_rows))
        return
    pinned_ids = [r['image_id'] for r in pinned]
    selected_ids = [r['image_id'] for r in pin_rows]
    if pinned_ids != selected_ids:
        raise FetchError(
            f'{manifest_path}: selection drift -- pinned {len(pinned_ids)} image ids, '
            f'recomputed {len(selected_ids)} differ. The annotation source or selection '
            'code changed since this manifest was pinned; investigate before re-pinning.'
        )


def parse_side_sets(spec: str) -> dict[str, int]:
    if not spec:
        return {}
    out: dict[str, int] = {}
    for part in spec.split(','):
        name, _, n = part.partition('=')
        if not name or not n.isdigit():
            raise FetchError(f'invalid --side-sets entry {part!r}; expected name=count')
        out[name] = int(n)
    return out
