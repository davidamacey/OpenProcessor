"""M-4 fix (W3/W4 review 2026-09-28): pin the actual computed
``ActivationImpact`` numbers against a known-buckets fake search response.

Before this file, no test asserted any number `compute_activation_impact`
returns -- the router test only checked that the response keys exist
(`test_region_profiles_router.py::test_active_impact_route`), so the
reviewer's mutation M5 (corrupting `validated_items` to read the pending
agg and dropping unseeded counts) left every test green. This file is the
regression: it stubs the OpenSearch client directly with a seeded
aggregation response and pins every field, including `items_total` (also
verifying `track_total_hits: True` is actually sent, per M-4's other
finding that the default 10k cap silently truncated it).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from src.config import DetectionProfile
from src.config.curation import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.curation.region_impact import compute_activation_impact


def _record(slug: str) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base_curation_config()),
    )


class _StubOpenSearch:
    """Returns fixed, hand-authored aggregation buckets -- no fake index,
    no query-matching logic, just "here is the response OpenSearch would
    give for these exact numbers," so the test pins the *arithmetic* in
    `compute_activation_impact`, not a fake search engine's correctness."""

    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []

    async def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        self.search_calls.append(body)
        return {
            'hits': {'total': {'value': 12345, 'relation': 'eq'}},
            'aggregations': {
                'by_profile': {
                    'buckets': [
                        {
                            'key': 'wheel',
                            'doc_count': 100,
                            'by_revision': {
                                'buckets': [
                                    {'key': 3, 'doc_count': 70},
                                    {'key': 2, 'doc_count': 30},
                                ]
                            },
                        },
                        {
                            'key': '__unseeded__',
                            'doc_count': 25,
                            'by_revision': {'buckets': []},
                        },
                    ]
                },
                'validated': {'doc_count': 40},
                'pending': {'doc_count': 55},
            },
        }

    async def count(self, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        self.count_calls.append(body)
        return {'count': 7}


@pytest.mark.asyncio
async def test_activation_impact_arithmetic_is_pinned() -> None:
    with bind_project(_record('impact-test')):
        client = _StubOpenSearch()
        profile = DetectionProfile(name='wheel', parent_classes=frozenset({'car'}))
        impact = await compute_activation_impact(client, profile=profile)

    assert impact.items_total == 12345
    assert impact.unseeded_items == 25
    assert impact.validated_items == 40
    assert impact.pending_items == 55
    assert impact.pending_not_matching == 7
    assert {(b.name, b.revision, b.count) for b in impact.by_profile} == {
        ('wheel', 3, 70),
        ('wheel', 2, 30),
    }

    # M-4: the 10k-cap trap this fix closes -- the search body must ask
    # OpenSearch to actually count past 10,000, not silently truncate.
    assert client.search_calls[0]['track_total_hits'] is True


@pytest.mark.asyncio
async def test_activation_impact_skips_pending_not_matching_without_profile() -> None:
    with bind_project(_record('impact-test-2')):
        client = _StubOpenSearch()
        impact = await compute_activation_impact(client, profile=None)

    assert impact.pending_not_matching == 0
    assert client.count_calls == []


# ---------------------------------------------------------- suggested_reprocess


def _stale_corpus() -> list[dict[str, Any]]:
    from curation.reprocess_fixtures import F, RegionStatus, box, item

    detected = RegionStatus.DETECTED.value
    accepted = (box('b1', state='accepted'),)
    return [
        item('old', detected, boxes=accepted, **{F.profile: 'wheel', F.profile_revision: 1}),
        item('cur', detected, boxes=accepted, **{F.profile: 'wheel', F.profile_revision: 2}),
        item('other', detected, boxes=accepted, **{F.profile: 'plate', F.profile_revision: 3}),
        item('failed', RegionStatus.DETECTION_FAILED.value, **{F.profile: 'plate'}),
        item('pending', RegionStatus.PENDING_DETECTION.value),
        item('human', detected, validated=True, verifier='human', **{F.profile: 'plate'}),
        item('unseeded'),
    ]


@pytest.mark.asyncio
async def test_activation_impact_suggests_the_reprocess_that_reruns_exactly_the_stale_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The impact carries a ``ReprocessRequest`` a client POSTs to
    ``/reprocess`` verbatim: unlocked, machine-written items the active
    ``wheel@2`` did not produce (not pending ones: the new profile processes
    those anyway; not validated ones; not unseeded ones)."""
    from curation.reprocess_fixtures import make_fake
    from src.services.curation import region_impact
    from src.services.curation.reprocess import plan_reprocess

    monkeypatch.setattr(region_impact, '_active_revision', lambda _profile: 2)
    fake = make_fake(_stale_corpus())

    impact = await compute_activation_impact(fake, profile=DetectionProfile(name='wheel'))

    assert impact.stale_items == 3  # old (rev 1), other (plate), failed (plate)
    suggestion = impact.suggested_reprocess
    assert suggestion is not None
    assert suggestion.scopes == ['region']
    assert suggestion.dry_run is True
    assert suggestion.targets.filter is not None
    assert suggestion.targets.filter.profile_not == 'wheel'
    assert suggestion.targets.filter.profile_revision_below == 2
    # what the GUI POSTs: the dumped request, verbatim
    posted = type(suggestion).model_validate(impact.model_dump()['suggested_reprocess'])
    region = (await plan_reprocess(fake, posted)).results[0]
    # the human-validated plate item matches the selector but is locked: the
    # re-run reports it as skipped, and `stale_items` counts only what it moves
    assert (region.selected, region.locked_skipped) == (4, 1)
    assert region.selected - region.locked_skipped == impact.stale_items


@pytest.mark.asyncio
async def test_activation_impact_suggests_nothing_without_a_profile_or_stale_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from curation.reprocess_fixtures import make_fake
    from src.services.curation import region_impact

    monkeypatch.setattr(region_impact, '_active_revision', lambda _profile: 2)
    fake = make_fake(_stale_corpus())
    assert (await compute_activation_impact(fake, profile=None)).suggested_reprocess is None
    only_current = make_fake([d for d in _stale_corpus() if d['crop_id'] == 'cur'])
    impact = await compute_activation_impact(only_current, profile=DetectionProfile(name='wheel'))
    assert (impact.stale_items, impact.suggested_reprocess) == (0, None)
