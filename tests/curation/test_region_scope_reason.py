"""The out-of-scope sentence names the item name the worker evaluated."""

from __future__ import annotations

from src.services.curation.region_scope import in_parent_classes, out_of_scope_reason


def test_reason_names_the_proposal_when_the_item_has_no_class() -> None:
    reason = out_of_scope_reason(['car', 'Bus'], class_name=None, proposal_name='person')
    assert reason is not None
    assert "'person'" in reason
    assert '(none)' not in reason


def test_reason_prefers_the_class_name_and_falls_back_to_none_label() -> None:
    both = out_of_scope_reason(['car'], class_name='dog', proposal_name='cat')
    assert both is not None
    assert "'dog'" in both
    neither = out_of_scope_reason(['car'], class_name='', proposal_name=None)
    assert neither is not None
    assert "'(none)'" in neither


def test_proposal_name_matches_case_insensitively() -> None:
    assert in_parent_classes(['Car'], class_name=None, proposal_name='CAR')
    assert out_of_scope_reason(['Car'], class_name='', proposal_name=' cAr ') is None
