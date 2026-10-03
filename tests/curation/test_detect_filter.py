"""The detect filter: what is dropped before anything is stored."""

from __future__ import annotations

from src.services.curation.ingest_policy import DetectFilter, apply_detect_filter
from src.services.curation.item_doc import DetectedItem


def _item(proposal: str, score: float = 0.9, size: float = 100.0) -> DetectedItem:
    return DetectedItem(bbox_pixel=(0, 0, size, size), score=score, proposal_name=proposal)


ITEMS = [
    _item('person', 0.9, 200),
    _item('car', 0.8, 100),
    _item('dog', 0.3, 20),
    _item('hot dog', 0.7, 50),
]


def _names(items: list[DetectedItem]) -> list[str | None]:
    return [i.proposal_name for i in items]


def test_default_filter_keeps_every_detection_unchanged() -> None:
    kept, dropped = apply_detect_filter(list(ITEMS), DetectFilter(), 1000, 1000)
    assert kept == ITEMS
    assert dropped == 0


def test_allow_list_by_name() -> None:
    kept, dropped = apply_detect_filter(
        list(ITEMS), DetectFilter(classes=['Hot_Dog', 'car']), 1000, 1000
    )
    assert _names(kept) == ['car', 'hot dog']
    assert dropped == 2


def test_deny_list_wins_over_allow_list() -> None:
    flt = DetectFilter(classes=['person', 'car'], exclude_classes=['person'])
    kept, _ = apply_detect_filter(list(ITEMS), flt, 1000, 1000)
    assert _names(kept) == ['car']


def test_confidence_area_and_cap() -> None:
    flt = DetectFilter(min_confidence=0.5, min_box_area_frac=0.003, max_per_image=2)
    kept, dropped = apply_detect_filter(list(ITEMS), flt, 1000, 1000)
    assert _names(kept) == ['person', 'car']  # dog: low conf; hot dog: 0.0025 area; cap is moot
    assert dropped == 2
    kept, _ = apply_detect_filter(list(ITEMS), DetectFilter(max_per_image=1), 1000, 1000)
    assert _names(kept) == ['person']


def test_kept_items_keep_their_original_order() -> None:
    kept, _ = apply_detect_filter(list(ITEMS), DetectFilter(max_per_image=3), 1000, 1000)
    assert _names(kept) == ['person', 'car', 'hot dog']
