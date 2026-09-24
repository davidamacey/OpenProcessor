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


def _region_item(**extra: Any) -> dict[str, Any]:
    return {
        F.bbox_norm: [0.2, 0.2, 0.4, 0.3],
        F.status: RegionStatus.DETECTED.value,
        F.validated: False,
        'class_validated': False,
        'test_holdout': False,
        **extra,
    }


def _rejected_candidate_item(**extra: Any) -> dict[str, Any]:
    """A realistic ``verify_rejected`` item (DQ-B2): no ``bbox_norm``, only
    the candidate box the verifier rejected."""
    doc = _region_item(**extra)
    doc.pop(F.bbox_norm, None)
    doc[F.status] = RegionStatus.VERIFY_REJECTED.value
    doc[F.candidate_bbox_norm] = [0.3, 0.6, 0.4, 0.65]
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
    assert not _in_queue(_region_item(**{F.validated: True}))


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
    doc.pop(F.bbox_norm, None)
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
    doc = _region_item(**{F.text: 'Ab12Cd'})
    assert _in_queue(doc, text='b12c')
    assert _in_queue(doc, text='AB12')
    assert not _in_queue(doc, text='zz')


def test_text_search_treats_wildcard_characters_literally() -> None:
    assert not _in_queue(_region_item(**{F.text: 'AB12'}), text='A*2')
    assert _in_queue(_region_item(**{F.text: 'xA*2y'}), text='a*2')
