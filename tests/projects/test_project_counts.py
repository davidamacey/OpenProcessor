"""Found live: ``GET /projects`` reported 491 items for a project holding 234
(``_cat/indices`` ``docs.count`` also counts the nested region boxes and box
embeddings)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from src.config.curation import IndexRole, base_curation_config
from src.config.projects import ProjectRecord, resources_for_new
from src.routers.curation.projects import _fetch_counts


class _Client:
    """``_count`` answers with top-level documents; ``_cat`` (not offered
    here) would have counted nested ones too."""

    def __init__(self, counts: dict[str, int], validated: int) -> None:
        self._counts = counts
        self._validated = validated

    async def count(self, *, index: str, body: Any = None) -> dict[str, int]:
        return {'count': self._validated if body else self._counts[index]}


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


@pytest.mark.asyncio
async def test_counts_are_top_level_documents_per_project() -> None:
    alpha, beta = _record('alpha'), _record('beta')
    idx = {r.slug: r.resources.indexes for r in (alpha, beta)}
    client = _Client(
        {
            idx['alpha'][IndexRole.IMAGES]: 60,
            idx['alpha'][IndexRole.ITEMS]: 234,
            idx['beta'][IndexRole.IMAGES]: 1,
            idx['beta'][IndexRole.ITEMS]: 2,
        },
        validated=5,
    )

    counts = await _fetch_counts(client, {'alpha': alpha, 'beta': beta})

    assert (counts['alpha'].images, counts['alpha'].items, counts['alpha'].validated) == (
        60,
        234,
        5,
    )
    assert (counts['beta'].images, counts['beta'].items) == (1, 2)


@pytest.mark.asyncio
async def test_embedded_count_is_none_when_uncountable() -> None:
    from src.services.projects.stats import embedded_count

    class _Down:
        async def count(self, **_: Any) -> dict[str, int]:
            raise RuntimeError('down')

    assert await embedded_count(_Down(), 'items') is None
    assert await embedded_count(_Client({'items': 9}, validated=4), 'items') == 4
