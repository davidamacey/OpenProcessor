"""A project count that could not be read is ``None`` (unknown), never ``0``: an
OpenSearch outage must not look like an empty project."""

from __future__ import annotations

from typing import Any

import pytest

from src.schemas.curation.projects import ProjectCounts, ProjectStatsCounts
from src.services.projects.stats import _term_count, index_count
from tests.projects.test_project_counts import _record


class _Down:
    async def count(self, **_: Any) -> dict[str, int]:
        raise RuntimeError('opensearch is down')


@pytest.mark.asyncio
async def test_index_count_is_none_when_the_count_fails() -> None:
    assert await index_count(_Down(), 'items') is None


@pytest.mark.asyncio
async def test_term_count_is_none_when_the_count_fails() -> None:
    assert await _term_count(_Down(), 'items', 'region_status', 'pending_detection') is None


@pytest.mark.asyncio
async def test_a_zero_count_stays_zero() -> None:
    class _Empty:
        async def count(self, **_: Any) -> dict[str, int]:
            return {'count': 0}

    assert await index_count(_Empty(), 'items') == 0
    assert await _term_count(_Empty(), 'items', 'f', True) == 0


@pytest.mark.asyncio
async def test_project_list_counts_are_unknown_during_an_outage() -> None:
    from src.routers.curation.projects import _fetch_counts

    counts = await _fetch_counts(_Down(), {'alpha': _record('alpha')})

    assert counts['alpha'].images is None
    assert counts['alpha'].items is None
    assert counts['alpha'].validated is None
    assert counts['alpha'].items_embedded is None


def test_the_wire_models_carry_null() -> None:
    ProjectCounts(images=None, items=None)
    ProjectStatsCounts(
        images=None,
        items=None,
        validated=None,
        items_embedded=None,
        pending_detection=None,
        holdout_items=None,
        classes=0,
        promoted_models=0,
    )
