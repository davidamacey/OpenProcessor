"""W3/W4/W9 shared: every ``ValidationIssue.bypassable`` equals membership
in that validator's ``BYPASSABLE_CODES``, and ``force_allowed`` is true
only when every error present is bypassable. Each wave that adds a
validator extends this file with its own rows -- W3 adds
``pack_validation``; W4 adds ``profile_validation``.
"""

from __future__ import annotations

from src.config import DetectionProfile
from src.services.config_store.pack_validation import (
    BYPASSABLE_CODES as PACK_BYPASSABLE,
    check_multi_region_keys,
    validate_pack,
)


def test_pack_bypassable_codes_are_empty() -> None:
    """The pack side's only activation-only error
    (``pack_multi_region_keys_missing``) is never bypassable -- there is
    no pack error a ``force: true`` can wave through."""
    assert frozenset() == PACK_BYPASSABLE


def test_pack_multi_region_keys_missing_is_never_bypassable() -> None:
    issue = check_multi_region_keys({}, max_regions_per_item=4, for_activation=True)
    assert issue is not None
    assert issue.code == 'pack_multi_region_keys_missing'
    assert issue.bypassable is False


def test_pack_every_issue_bypassable_flag_matches_the_constant() -> None:
    body = {'class_system': ''}  # trips several error codes at once
    profile = DetectionProfile(name='p', max_regions_per_item=4)
    report = validate_pack(None, body, profile=profile, for_activation=True)
    for issue in [*report.errors, *report.warnings]:
        assert issue.bypassable == (issue.code in PACK_BYPASSABLE)


def test_pack_force_allowed_true_only_when_every_error_bypassable() -> None:
    body = {'class_system': ''}
    report = validate_pack(None, body)
    assert report.errors  # sanity: this draft really has errors
    # None of the pack codes are bypassable, so force_allowed must be False
    # whenever there is at least one error.
    assert report.force_allowed is False
