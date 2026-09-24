"""``GET /review/regions`` match logic.

Region review is independent of the item's class validation (VLM and
cluster agreement validate most classes automatically), and the region
text search must not depend on the stored text's case.
"""

from __future__ import annotations

from typing import Any

from curation.query_fakes import matches
from src.config.region_fields import get_region_fields
from src.config.region_state import RegionStatus
from src.services.curation.review_queries import build_tab_query


F = get_region_fields()


def _region_item(**extra: Any) -> dict[str, Any]:
    return {
        F.bbox_norm: [0.2, 0.2, 0.4, 0.3],
        F.status: RegionStatus.DETECTED.value,
        F.validated: False,
        'class_validated': False,
        'test_holdout': False,
        **extra,
    }


def _in_queue(doc: dict[str, Any], *, text: str | None = None) -> bool:
    must, must_not, _reason = build_tab_query(
        'regions', include_test=False, text=text, max_rank=None
    )
    return matches(doc, {'bool': {'must': must, 'must_not': must_not}})


def test_class_validated_item_with_unreviewed_region_is_queued() -> None:
    assert _in_queue(_region_item(class_validated=True))


def test_region_validated_item_is_not_queued() -> None:
    assert not _in_queue(_region_item(**{F.validated: True}))


def test_rejected_and_false_positive_regions_are_not_queued() -> None:
    assert not _in_queue(_region_item(**{F.status: RegionStatus.VERIFY_REJECTED.value}))
    assert not _in_queue(_region_item(**{F.status: RegionStatus.FALSE_POSITIVE.value}))


def test_text_search_is_case_insensitive() -> None:
    doc = _region_item(**{F.text: 'Ab12Cd'})
    assert _in_queue(doc, text='b12c')
    assert _in_queue(doc, text='AB12')
    assert not _in_queue(doc, text='zz')


def test_text_search_treats_wildcard_characters_literally() -> None:
    assert not _in_queue(_region_item(**{F.text: 'AB12'}), text='A*2')
    assert _in_queue(_region_item(**{F.text: 'xA*2y'}), text='a*2')
