"""The boxes an image view draws: one per item and one per region box, with the label and the
tone (how the box looks) decided here so the browser only paints what it is given."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from src.clients.occ_locks import is_locked_box
from src.config import get_region_fields
from src.domain.region_boxes import read_boxes
from src.schemas.curation.images import BoxTone, BoxView
from src.services.curation.wire import parse_xyxy


if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.config import RegionFields

UNLABELED_LABEL: Final = 'unlabeled'


def is_labeled(src: dict[str, Any]) -> bool:
    """An item with a class (the registry id is the mark; an unlabeled proposal has none)."""
    return src.get('class_id') is not None


def item_label(src: dict[str, Any]) -> str:
    return str(src.get('class_name') or UNLABELED_LABEL) if is_labeled(src) else UNLABELED_LABEL


def item_tone(src: dict[str, Any]) -> BoxTone:
    if src.get('class_excluded'):
        return 'rejected'
    if src.get('test_holdout'):
        return 'holdout'
    if src.get('class_validated'):
        return 'validated'
    return 'machine' if is_labeled(src) else 'unlabeled'


def item_box_views(
    src: dict[str, Any], crop_id: str, storage: RegionFields | None = None
) -> list[BoxView]:
    """The item's own box, then each region box it holds."""
    f = storage or get_region_fields()
    label = item_label(src)
    class_name = str(src['class_name']) if is_labeled(src) and src.get('class_name') else None
    tone = item_tone(src)
    views: list[BoxView] = []
    if (item_box := parse_xyxy(src.get('bbox_norm'))) is not None:
        views.append(
            BoxView(
                crop_id=crop_id,
                kind='item',
                bbox_norm=item_box,
                label=label,
                class_name=class_name,
                tone=tone,
            )
        )
    for box in read_boxes(src, f):
        if (region_box := parse_xyxy(list(box.bbox_norm))) is None:
            continue
        box_tone: BoxTone = tone
        if box.state == 'rejected':
            box_tone = 'rejected'
        elif tone in ('machine', 'unlabeled') and is_locked_box(box):
            box_tone = 'validated'
        views.append(
            BoxView(
                crop_id=crop_id,
                box_id=box.box_id,
                kind='region',
                bbox_norm=region_box,
                label=label,
                class_name=class_name,
                tone=box_tone,
                state=box.state,
            )
        )
    return views


def boxes_of(docs: Iterable[tuple[str, dict[str, Any]]]) -> list[BoxView]:
    """Every box of ``(crop_id, stored item)`` pairs, in the order given."""
    return [view for crop_id, src in docs for view in item_box_views(src, crop_id)]
