"""W3/W4/W9 shared: every ``ValidationIssue.bypassable`` equals membership
in that validator's ``BYPASSABLE_CODES``, and ``force_allowed`` is true
only when every error present is bypassable. Each wave that adds a
validator extends this file with its own rows -- W3 adds
``pack_validation``; W4 adds ``profile_validation``.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from src.config import DetectionProfile
from src.services.config_store.pack_validation import (
    BYPASSABLE_CODES as PACK_BYPASSABLE,
    check_multi_region_keys,
    validate_pack,
)
from src.services.config_store.profile_validation import (
    BYPASSABLE_CODES as PROFILE_BYPASSABLE,
    validate_profile,
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


def _profile_body(**overrides: Any) -> dict[str, Any]:
    raw = asdict(DetectionProfile(name='p'))
    raw.pop('name')
    for key, value in raw.items():
        if isinstance(value, frozenset):
            raw[key] = sorted(value)
        elif isinstance(value, tuple):
            raw[key] = list(value)
    raw.update(overrides)
    return raw


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _unreachable() -> list[dict[str, Any]]:
    raise ConnectionError('down')


def test_profile_every_issue_bypassable_flag_matches_the_constant() -> None:
    body = _profile_body(detector_model='m', text_reader='none')
    report = _run(
        validate_profile(None, body, get_repository_index=_unreachable, for_activation=True)
    )
    assert report.errors  # sanity: triton_unreachable fired
    for issue in [*report.errors, *report.warnings]:
        assert issue.bypassable == (issue.code in PROFILE_BYPASSABLE)


def test_profile_force_allowed_true_only_when_every_error_bypassable() -> None:
    body = _profile_body(detector_model='m', text_reader='none')
    report = _run(
        validate_profile(None, body, get_repository_index=_unreachable, for_activation=True)
    )
    assert report.errors
    assert report.force_allowed is True
    # A non-bypassable error alongside a bypassable one flips force_allowed False.
    body2 = _profile_body(detector_model='', segmenter_text_prompt='', text_reader='none')
    report2 = _run(
        validate_profile(None, body2, get_repository_index=_unreachable, for_activation=True)
    )
    codes2 = {e.code for e in report2.errors}
    assert 'no_candidate_source' in codes2  # never bypassable
    assert report2.force_allowed is False


def test_vlm_max_images_exceeds_server_is_reserved_never_bypassable() -> None:
    """W9 owns raising this code; pinned here so a future BYPASSABLE_CODES
    addition can never accidentally include it (any_domain_plan.md §3.3)."""
    assert 'vlm_max_images_exceeds_server' not in PACK_BYPASSABLE
    assert 'vlm_max_images_exceeds_server' not in PROFILE_BYPASSABLE
