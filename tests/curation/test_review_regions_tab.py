"""``GET /review/regions`` match logic.

Region review is independent of the item's class validation (VLM and
cluster agreement validate most classes automatically), and the region
text search must not depend on the stored text's case.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from curation.query_fakes import matches
from src.config.region_fields import get_region_fields
from src.config.region_state import RegionStatus
from src.services.curation.review_queries import build_tab_query


F = get_region_fields()


def _region_item(*, text: str | None = None, **extra: Any) -> dict[str, Any]:
    box: dict[str, Any] = {
        'box_id': 'b1',
        'bbox_norm': [0.2, 0.2, 0.4, 0.3],
        'state': 'accepted',
    }
    if text is not None:
        box['text'] = text
    return {
        F.boxes: [box],
        F.status: RegionStatus.DETECTED.value,
        F.validated: False,
        'class_validated': False,
        'test_holdout': False,
        **extra,
    }


def _rejected_candidate_item(**extra: Any) -> dict[str, Any]:
    """A realistic ``verify_rejected`` item (DQ-B2): no accepted box, only
    the candidate box the verifier rejected (kept, ``state='rejected'``)."""
    from src.services.curation.region_boxes import RegionBox, boxes_write_fields

    doc = _region_item(**extra)
    doc[F.status] = RegionStatus.VERIFY_REJECTED.value
    rejected = RegionBox(box_id='b1', bbox_norm=(0.3, 0.6, 0.4, 0.65), state='rejected')
    doc.update(boxes_write_fields([rejected], current_src={}))
    return doc


def _in_queue(
    doc: dict[str, Any], *, text: str | None = None, region_status: str | None = None
) -> bool:
    must, must_not, _reason = build_tab_query(
        'regions', include_test=False, text=text, max_rank=None, region_status=region_status
    )
    return matches(doc, {'bool': {'must': must, 'must_not': must_not}})


def test_class_validated_item_with_unreviewed_region_is_queued() -> None:
    assert _in_queue(_region_item(class_validated=True))


def test_region_validated_item_is_not_queued() -> None:
    validated = _region_item()
    validated[F.validated] = True
    assert not _in_queue(validated)


def test_false_positive_regions_are_never_queued() -> None:
    assert not _in_queue(_region_item(**{F.status: RegionStatus.FALSE_POSITIVE.value}))
    for mode in ('all', 'detected', 'verify_rejected'):
        assert not _in_queue(
            _region_item(**{F.status: RegionStatus.FALSE_POSITIVE.value}), region_status=mode
        )


def test_rejected_candidate_is_reachable_from_the_default_queue() -> None:
    """DQ-B2 follow-up: a verify_rejected item with a kept candidate box
    used to be unreachable from every review tab. It must now surface in
    the default ('all') queue, in the verify_rejected-only queue, but NOT
    in the detected-only queue."""
    doc = _rejected_candidate_item()
    assert _in_queue(doc)
    assert _in_queue(doc, region_status='all')
    assert _in_queue(doc, region_status='verify_rejected')
    assert not _in_queue(doc, region_status='detected')


def test_rejected_status_without_a_candidate_box_is_not_queued() -> None:
    """A legacy verify_rejected row with no candidate (rejected before the
    candidate was kept) has nothing to show -- correctly excluded."""
    doc = _region_item(**{F.status: RegionStatus.VERIFY_REJECTED.value})
    doc[F.boxes] = []
    assert not _in_queue(doc)
    assert not _in_queue(doc, region_status='verify_rejected')


def test_detected_item_is_reachable_in_every_mode_except_verify_rejected_only() -> None:
    doc = _region_item()
    assert _in_queue(doc)
    assert _in_queue(doc, region_status='all')
    assert _in_queue(doc, region_status='detected')
    assert not _in_queue(doc, region_status='verify_rejected')


def test_unknown_region_status_filter_raises_400() -> None:
    with pytest.raises(HTTPException) as exc:
        build_tab_query(
            'regions', include_test=False, text=None, max_rank=None, region_status='bogus'
        )
    assert exc.value.status_code == 400


def test_text_search_is_case_insensitive() -> None:
    doc = _region_item(text='Ab12Cd')
    assert _in_queue(doc, text='b12c')
    assert _in_queue(doc, text='AB12')
    assert not _in_queue(doc, text='zz')


def test_text_search_treats_wildcard_characters_literally() -> None:
    assert not _in_queue(_region_item(text='AB12'), text='A*2')
    assert _in_queue(_region_item(text='xA*2y'), text='a*2')


def _boxed_item(*states: str, status: RegionStatus) -> dict[str, Any]:
    from src.services.curation.region_boxes import RegionBox, boxes_write_fields

    boxes = [
        RegionBox(box_id=f'b{i + 1}', bbox_norm=(0.1, 0.1, 0.2, 0.2), state=state)
        for i, state in enumerate(states)
    ]
    return {
        **boxes_write_fields(boxes, current_src={}),
        F.status: status.value,
        F.validated: False,
        'class_validated': False,
        'test_holdout': False,
    }


def test_has_rejected_box_reaches_a_rejected_box_on_a_detected_item() -> None:
    mixed = _boxed_item('accepted', 'rejected', status=RegionStatus.DETECTED)
    assert _in_queue(mixed, region_status='has_rejected_box')
    # ...but the item is not "only rejected boxes".
    assert not _in_queue(mixed, region_status=RegionStatus.VERIFY_REJECTED.value)


def test_an_all_rejected_item_is_in_both_rejected_options() -> None:
    rejected = _boxed_item('rejected', 'rejected', status=RegionStatus.VERIFY_REJECTED)
    assert _in_queue(rejected, region_status='has_rejected_box')
    assert _in_queue(rejected, region_status=RegionStatus.VERIFY_REJECTED.value)


def test_has_rejected_box_ignores_items_without_a_rejected_box() -> None:
    clean = _boxed_item('accepted', status=RegionStatus.DETECTED)
    assert not _in_queue(clean, region_status='has_rejected_box')
