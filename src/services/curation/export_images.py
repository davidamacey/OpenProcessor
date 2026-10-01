"""Per-image grouping for the multi-class YOLO export.

The standard YOLO layout is one image file and one label file per source
image, with one line per object on it. The exporter scrolls validated
*items* (one object each) off the items index; this module turns those
into *images*: it groups objects by ``image_id``, renders their label
lines, counts the objects on each exported image that are not labeled,
and picks the stratum an image counts toward when a ``max_images`` cap
samples images.

Why grouping matters: writing one full-frame copy per object, each
labeled with only that object, teaches a detector that every other object
in the frame is background.

**Unlabeled objects.** An object on an exported image is *unlabeled* when
it is an item on that image that the export does not write a line for —
not validated yet, validated on a class the export has no dense id for
(unregistered or deprecated), or validated without a usable box. It is
still in the pixels, so training learns it as background. Items that are
``class_excluded`` or review-dismissed are not objects to label and are
never counted.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from src.config.project_context import run_in_executor_bound
from src.core.logging import get_logger
from src.services.curation.export_readiness import NothingToExportError
from src.services.curation.export_support import (
    _copy_or_resize_one,
    even_stratified_sample,
    scroll_hits,
)
from src.services.detection.frame_dedup import dedup_rows_by_embedding


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from src.config import CurationConfig
    from src.services.curation.export_support import _ExportRow


logger = get_logger(__name__)

# Keeps an unlabeled-items query's ``terms`` clause well under OpenSearch's
# ``index.max_terms_count`` default (65,536).
UNLABELED_QUERY_CHUNK = 1000

_SAFE_STEM = re.compile(r'[^A-Za-z0-9._-]')


@dataclass
class _ExportImage:
    """One source image and every validated object the export writes for it.

    ``image_id`` + ``has_test_crop`` make it a ``frame_dedup._DedupRow``,
    so near-duplicate collapse runs on images directly. ``has_test_crop``
    is true when any object on the image is a frozen ``test_holdout``
    item; the whole image then goes to ``test``.
    """

    image_id: str
    image_path: str
    objects: list[_ExportRow] = field(default_factory=list)
    has_test_crop: bool = False


def group_rows_by_image(rows: Iterable[_ExportRow]) -> list[_ExportImage]:
    """Group object rows by ``image_id``, sorted by ``image_id``.

    Objects inside an image are sorted by ``item_id`` so the written label
    file is byte-identical across re-runs regardless of scroll order.
    """
    images: dict[str, _ExportImage] = {}
    for row in rows:
        image = images.get(row.image_id)
        if image is None:
            image = images[row.image_id] = _ExportImage(row.image_id, row.image_path)
        image.objects.append(row)
        image.has_test_crop = image.has_test_crop or row.has_test_crop
    for image in images.values():
        image.objects.sort(key=lambda r: r.item_id)
    return [images[k] for k in sorted(images)]


def normalized_box(bbox: Any) -> tuple[float, float, float, float] | None:
    """A stored ``bbox_norm`` as a clamped ``(x1, y1, x2, y2)``, or ``None``.

    ``bbox_norm`` is normalized to the full source frame (``[0, 1]`` on
    both axes). Corners are clamped to the frame before the box is
    judged, so a slightly out-of-frame box is kept at its visible extent
    and a box with no area inside the frame is rejected.
    """
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (min(max(float(v), 0.0), 1.0) for v in bbox)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def yolo_line(export_class_id: int, bbox: Sequence[float]) -> str:
    """One YOLO label line, ``cls cx cy w h``, relative to the full source image."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return f'{export_class_id} {cx:.6f} {cy:.6f} {x2 - x1:.6f} {y2 - y1:.6f}'


def image_file_stem(image_id: str) -> str:
    """Filesystem-safe stem for an image's label and image files.

    Ingest ids are hex digests, which pass through unchanged. Anything
    else has unsafe characters replaced, plus a short digest of the raw id
    so two ids that differ only in replaced characters can't collide.
    """
    safe = _SAFE_STEM.sub('_', image_id)
    if safe == image_id and image_id not in ('', '.', '..'):
        return image_id
    return f'{safe}-{hashlib.sha256(image_id.encode()).hexdigest()[:12]}'


def rarest_class_key(image: _ExportImage, class_frequency: dict[int, int]) -> str:
    """The stratum an image counts toward when a ``max_images`` cap samples.

    The image's rarest class across the pool (ties to the lowest dense id),
    so an image carrying a rare class is sampled as that class and a
    common class sharing the frame can't crowd it out.
    """
    rarest = min(
        {o.export_class_id for o in image.objects},
        key=lambda cid: (class_frequency.get(cid, 0), cid),
    )
    return str(rarest)


async def unlabeled_items_by_image(
    opensearch: Any,
    *,
    index: str,
    image_ids: list[str],
    labeled_item_ids: set[str],
) -> dict[str, int]:
    """``{image_id: n}`` for the images with at least one unlabeled object.

    Reads every non-excluded, non-dismissed item on ``image_ids`` and
    counts those the export does not write (not in ``labeled_item_ids``).
    The excluded / dismissed filter is re-applied to each hit so a
    partial result can never count one.
    """
    counts: dict[str, int] = {}
    wanted = set(image_ids)
    for start in range(0, len(image_ids), UNLABELED_QUERY_CHUNK):
        chunk = image_ids[start : start + UNLABELED_QUERY_CHUNK]
        hits = await scroll_hits(
            opensearch,
            index=index,
            query={
                'bool': {
                    'filter': [{'terms': {'image_id': chunk}}],
                    'must_not': [
                        {'exists': {'field': 'review_dismissed_at'}},
                        {'term': {'class_excluded': True}},
                    ],
                }
            },
            source=['crop_id', 'image_id', 'class_excluded', 'review_dismissed_at'],
        )
        for hit in hits:
            src = hit.get('_source') or {}
            item_id = str(src.get('crop_id') or hit.get('_id') or '')
            image_id = str(src.get('image_id') or '')
            if (
                image_id not in wanted
                or item_id in labeled_item_ids
                or src.get('class_excluded')
                or src.get('review_dismissed_at') is not None
            ):
                continue
            counts[image_id] = counts.get(image_id, 0) + 1
    return counts


async def copy_export_images(
    jobs: list[tuple[str, str]],
    *,
    resize_mode: Literal['aspect'] | None,
    image_size: int,
    max_workers: int,
) -> dict[str, Any]:
    """Copy (optionally aspect-resize) every ``(src, dest)`` pair in a
    process pool; returns the ``image_copy`` manifest block."""
    if not jobs:
        return {'attempted': 0, 'copied': 0, 'failed': 0, 'errors': []}
    loop = asyncio.get_running_loop()
    copied = 0
    failed = 0
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            run_in_executor_bound(
                loop, pool, _copy_or_resize_one, src, dest, resize_mode, image_size
            )
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


async def select_export_images(
    opensearch: Any,
    config: CurationConfig,
    rows: list[_ExportRow],
    *,
    require_fully_labeled_images: bool,
    dedup_threshold: float | None,
    max_images: int | None,
    seed: int,
) -> tuple[list[_ExportImage], dict[str, int], dict[str, Any]]:
    """Group dense-id rows into images, then apply the partial-frame
    policy, the near-dup collapse and the ``max_images`` cap, in that
    order -- so a dropped partial frame can't knock out a fully labeled
    near-duplicate, and the cap lands on exactly ``min(max_images, pool)``
    images.

    Returns ``(images, unlabeled_by_image, info)``.
    """
    images = group_rows_by_image(rows)
    unlabeled = await unlabeled_items_by_image(
        opensearch,
        index=config.items_index,
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
    dedup_stats: dict[str, Any] = {'enabled': False}
    if dedup_threshold is not None:
        kept, dedup_stats = await dedup_rows_by_embedding(
            opensearch, images, threshold=dedup_threshold, config=config
        )
        images = sorted(kept, key=lambda im: im.image_id)

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


__all__ = [
    'UNLABELED_QUERY_CHUNK',
    'copy_export_images',
    'group_rows_by_image',
    'image_file_stem',
    'normalized_box',
    'rarest_class_key',
    'select_export_images',
    'unlabeled_items_by_image',
    'yolo_line',
]
