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

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.services.curation.export_support import scroll_hits


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from src.services.curation.export_support import _ExportRow


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


__all__ = [
    'UNLABELED_QUERY_CHUNK',
    'group_rows_by_image',
    'image_file_stem',
    'normalized_box',
    'rarest_class_key',
    'unlabeled_items_by_image',
    'yolo_line',
]
