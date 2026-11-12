"""The boxes an image view draws: one per item and one per region box, with the label and the
tone (how the box looks) decided here so the browser only paints what it is given."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, cast

from src.clients.occ_locks import is_locked_box, is_locked_geometry
from src.config import get_region_fields
from src.domain.region_boxes import read_boxes
from src.schemas.curation.images import BoxLabelKind, BoxTone, BoxView
from src.services.curation.geometry_wire import geometry_lock_label, region_box_lock_label
from src.services.curation.wire import item_display_name, parse_xyxy, region_box_name


if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.config import RegionFields

UNLABELED_LABEL: Final = 'unlabeled'


def is_labeled(src: dict[str, Any]) -> bool:
    """An item with a class (the registry id is the mark; an unlabeled proposal has none)."""
    return src.get('class_id') is not None


def item_label(src: dict[str, Any]) -> tuple[str, BoxLabelKind]:
    """The item's served display name and its kind; ``unlabeled`` when nothing names it."""
    named = item_display_name(src)
    kind = named['display_name_kind']
    return named['display_name'] or UNLABELED_LABEL, cast('BoxLabelKind', kind)


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
    label, label_kind = item_label(src)
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
                label_kind=label_kind,
                class_name=class_name,
                tone=tone,
                locked=is_locked_geometry(src),
                lock_label=geometry_lock_label(src),
                geometry_revision=int(src.get('geometry_revision') or 0),
            )
        )
    for ordinal, box in enumerate(read_boxes(src, f), start=1):
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
                label=region_box_name(box, ordinal),
                label_kind='region',
                class_name=class_name,
                tone=box_tone,
                state=box.state,
                locked=is_locked_box(box),
                lock_label=region_box_lock_label(box, locked=is_locked_box(box)),
            )
        )
    return views


def boxes_of(docs: Iterable[tuple[str, dict[str, Any]]]) -> list[BoxView]:
    """Every box of ``(crop_id, stored item)`` pairs, in the order given."""
    return [view for crop_id, src in docs for view in item_box_views(src, crop_id)]
