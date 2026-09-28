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
